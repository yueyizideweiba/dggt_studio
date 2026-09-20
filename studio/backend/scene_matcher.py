"""把 output/waymo_eval 里的重建场景匹配回 Waymo 原始 segment（以及起始帧）。

原理：重建场景的世界系是"首帧自车系"（推理时归一化过），所以**自车系下的逐帧位移向量**
（dx 横向 / dz 纵向）与原始 segment 的自车系位移是同一个量（只差一个全局朝向 + 可能的尺度）。
于是可以用 seq = 逐帧局部位移 的序列做滑动窗口匹配。

用途：拿到场景对应的 segment 后，就能取到同一时间戳的**其它相机真实图像**
（`data/waymo/processed/validation/<seg>/images/NNN_C.jpg`），从而做真正的 novel-view 评测。
"""
from __future__ import annotations

import json
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _load_matrix(path_or_json):
    if isinstance(path_or_json, str) and path_or_json.endswith(".json"):
        d = json.load(open(path_or_json))
        m = np.asarray(d["camera_extrinsics_world"], dtype=np.float64)
    elif isinstance(path_or_json, str):
        m = np.loadtxt(path_or_json)
    else:
        m = np.asarray(path_or_json, dtype=np.float64)
    if m.shape == (3, 4):
        m = np.vstack([m, np.array([[0.0, 0.0, 0.0, 1.0]])])
    return m


def local_displacements(poses: List[np.ndarray], convention: str = "y_up") -> np.ndarray:
    """自车系下的逐帧位移 [n-1, 2] = (纵向 forward, 横向 right)。

    convention:
      - "y_up"：DGGT 重建场景的相机/自车系（x=右, y=下, z=前）→ forward=z, right=+x
      - "z_up"：Waymo 预处理 ego pose（x=前, y=左, z=上）→ forward=x, right=-y
    """
    out = []
    for a, b in zip(poses[:-1], poses[1:]):
        R = a[:3, :3]
        d = b[:3, 3] - a[:3, 3]
        d_local = R.T @ d
        if convention == "z_up":
            out.append([float(d_local[0]), float(-d_local[1])])
        else:
            out.append([float(d_local[2]), float(d_local[0])])
    return np.asarray(out, dtype=np.float64)


def scene_displacements(scene_dir: str, flat_step: int = 1, max_frames: Optional[int] = None) -> np.ndarray:
    """重建场景（view0）在自车系下的逐帧位移；多视角数据按 flat_step 跨视角取 view0。"""
    ego_dir = os.path.join(scene_dir, "ego_pose")
    # 多视角场景的 ego_pose 是逐视角 flat 排列的：必须显式取 view0，
    # 不能直接把所有文件按顺序当成连续帧（否则每帧在 3 台相机之间跳）。
    try:
        import scene_frames as _sf
        n_real = _sf.num_real_frames(scene_dir)
        idxs = list(range(n_real))[::max(1, int(flat_step))]
        if max_frames:
            idxs = idxs[:max_frames]
        poses = []
        for i in idxs:
            mc = _sf.load_ego_pose(scene_dir, i, 0)
            if mc is not None:
                poses.append(_load_matrix(_sf.ego_pose_path(scene_dir, i, 0)))
    except Exception:  # noqa: BLE001
        files = sorted(f for f in os.listdir(ego_dir) if f.endswith("_ego.json"))
        if flat_step > 1:
            files = files[::flat_step]
        if max_frames:
            files = files[:max_frames]
        poses = [_load_matrix(os.path.join(ego_dir, f)) for f in files]
    return local_displacements(poses, convention="y_up")


def segment_local_cache(data_root: str, cache_path: str = "/tmp/waymo_seg_local.npz") -> Dict[str, np.ndarray]:
    """把所有 segment 的逐帧局部位移缓存起来（一次 ~40k 小文件读取）。"""
    if os.path.exists(cache_path):
        z = np.load(cache_path, allow_pickle=True)
        good = {}
        for k in z.files:
            v = z[k]
            if len(v) >= 2 and float(np.median(np.linalg.norm(v, axis=1))) >= 0.01:
                good[k] = v
        if good:
            return good
    out = {}
    for seg in sorted(os.listdir(data_root)):
        ego = os.path.join(data_root, seg, "ego_pose")
        if not os.path.isdir(ego):
            continue
        fs = sorted(f for f in os.listdir(ego) if f.endswith(".txt"))
        if len(fs) < 5:
            continue
        poses = [_load_matrix(os.path.join(ego, f)) for f in fs]
        disp = local_displacements(poses, convention="z_up")
        # 跳过退化的段（有些预处理目录里的 ego_pose 是同一个位姿复制出来的）
        if len(disp) < 2 or float(np.median(np.linalg.norm(disp, axis=1))) < 0.01:
            continue
        out[seg] = disp
    np.savez_compressed(cache_path, **out)
    return out


def _norm(v: np.ndarray) -> Tuple[np.ndarray, float]:
    s = float(np.median(np.linalg.norm(v, axis=1))) or 1.0
    return v / s, s


def match_segment(scene_disp: np.ndarray, seg_cache: Dict[str, np.ndarray],
                  top_k: int = 5, stride: int = 3) -> List[Dict]:
    """滑动窗口匹配：返回 [{segment, offset, score, scale}]（score 越小越像）。"""
    q, q_scale = _norm(scene_disp)
    q_flip = q * np.array([1.0, -1.0])       # 横向符号不确定（左/右约定）
    n = len(q)
    results = []
    for seg, disp in seg_cache.items():
        m = len(disp)
        if m < n + 1:
            continue
        cand, c_scale = _norm(disp)
        for k in range(0, m - n, stride):
            w = cand[k:k + n]
            s1 = float(np.mean(np.abs(w - q)))
            s2 = float(np.mean(np.abs(w - q_flip)))
            score = min(s1, s2)
            results.append((score, seg, k, c_scale / q_scale, "same" if s1 <= s2 else "flip"))
    results.sort(key=lambda x: x[0])
    out, used = [], []
    for score, seg, k, sc, lat in results:
        if any(u[1] == seg and abs(k - u[2]) < n for u in used):
            continue
        used.append((score, seg, k))
        out.append({"segment": seg, "offset": int(k), "score": round(score, 4),
                    "lateral_convention": lat,
                    "scale_seg_over_scene": round(float(sc), 4)})
        if len(out) >= top_k:
            break
    return out


def gt_image_paths(data_root: str, segment: str, frame_offset: int, camera: int,
                   count: int) -> List[str]:
    """取某 segment 的某相机在 [frame_offset, frame_offset+count) 的 GT 图像路径。"""
    d = os.path.join(data_root, segment, "images")
    out = []
    for i in range(frame_offset, frame_offset + count):
        p = os.path.join(d, f"{i:03d}_{camera}.jpg")
        out.append(p if os.path.exists(p) else "")
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser(description="把重建场景匹配回 Waymo segment")
    ap.add_argument("scene_dir", help="如 output/waymo_eval/000/test1")
    ap.add_argument("--data_root", default=os.path.join(ROOT, "data/waymo/processed/validation"))
    ap.add_argument("--flat_step", type=int, default=1, help="多视角场景：跨视角步长（num_views）")
    ap.add_argument("--max_frames", type=int, default=0)
    args = ap.parse_args()
    disp = scene_displacements(args.scene_dir, args.flat_step, args.max_frames or None)
    cache = segment_local_cache(args.data_root)
    hits = match_segment(disp, cache)
    print(json.dumps({"scene": args.scene_dir, "num_steps": len(disp), "candidates": hits},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
