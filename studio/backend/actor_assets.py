"""actor 资产库：把"逐帧、视角相关"的动态高斯规范化成**对象中心的 3D 资产**。

为什么需要（P1 核心）：
- 重建输出的动态物体是"每帧一份、只在源视角附近有形"的高斯（实测：从上往下看会退化成条纹），
  而且 `object_id` 只是**每帧重排的簇号**（同一辆车相邻帧会换号）→ 既不能跨帧拼，也不能换视角看。
- 本模块做三件事：
  1. `associate_tracks`：用"匀速预测 + 最近邻 + 尺寸相容"把每帧簇号关联成持续轨迹；
  2. 多视角感知：`flat = real * num_views + view`，时间维只走 view 0，其它视角按世界位置并进来
     （同一物体被多个视角看到 → 角度覆盖更全）；
  3. `solidify_surfels`：把单目 surfel 体素化成**体积代理**，使资产在任意视角都是实体；
  4. 写标准 3DGS ply（**原始值**：opacity=logit / scale=log / f_dc=pre-sigmoid），
     直接走渲染器的 `_load_sam3d_ply` 通道，与 SAM3D 资产（ego/person）同一套 schema，
     写 `bank.json` 兼容 `dggt/scene_edit/asset_bank.SceneObjectAssetBank`。

渲染任意摆位的资产：
`extra_objects=[{"ply_path": asset, "transform": pose_world, "synth_track_id": id}]`
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


# ==================== ply IO ====================

def _read_dynamic_ply(path: str):
    """读一帧的 dynamic ply，返回 dict(object_id -> arrays)。"""
    from plyfile import PlyData
    v = PlyData.read(path)["vertex"]
    names = v.data.dtype.names
    keys = ("x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "opacity",
            "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3")
    if not all(k in names for k in keys) or "object_id" not in names:
        return {}
    arrs = {k: np.asarray(v[k], dtype=np.float32) for k in keys}
    ids = np.asarray(v["object_id"])
    out = {}
    for uid in np.unique(ids):
        uid_i = int(uid)
        if uid_i < 0:
            continue                      # -1 = 全局/背景，不做资产
        m = ids == uid
        out[uid_i] = {k: a[m] for k, a in arrs.items()}
    return out


def _logit(x: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    x = np.clip(x, eps, 1.0 - eps)
    return np.log(x / (1.0 - x)).astype(np.float32)


def _write_gs_ply(path: str, means, scales, quats, opacity, colors):
    """标准 3DGS ply（原始值）：scale=log、opacity=logit、f_dc=logit(color)。"""
    from plyfile import PlyElement, PlyData
    n = len(means)
    fields = [("x", "f4"), ("y", "f4"), ("z", "f4"),
              ("nx", "f4"), ("ny", "f4"), ("nz", "f4"),
              ("f_dc_0", "f4"), ("f_dc_1", "f4"), ("f_dc_2", "f4"),
              ("opacity", "f4"),
              ("scale_0", "f4"), ("scale_1", "f4"), ("scale_2", "f4"),
              ("rot_0", "f4"), ("rot_1", "f4"), ("rot_2", "f4"), ("rot_3", "f4")]
    elem = np.empty(n, dtype=fields)
    elem["x"], elem["y"], elem["z"] = means[:, 0], means[:, 1], means[:, 2]
    elem["nx"] = elem["ny"] = elem["nz"] = 0.0
    col = _logit(np.clip(colors, 0, 1))
    elem["f_dc_0"], elem["f_dc_1"], elem["f_dc_2"] = col[:, 0], col[:, 1], col[:, 2]
    elem["opacity"] = _logit(np.clip(opacity, 0, 1))
    sc = np.log(np.clip(scales, 1e-8, None))
    elem["scale_0"], elem["scale_1"], elem["scale_2"] = sc[:, 0], sc[:, 1], sc[:, 2]
    elem["rot_0"], elem["rot_1"] = quats[:, 0], quats[:, 1]
    elem["rot_2"], elem["rot_3"] = quats[:, 2], quats[:, 3]
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    PlyData([PlyElement.describe(elem, "vertex")], text=False).write(path)


# ==================== 几何/统计小工具 ====================

def _voxel_dedup(pts: np.ndarray, keep_score: np.ndarray, voxel: float) -> np.ndarray:
    """体素去重：每格只留 keep_score 最大的一个，返回保留的索引。"""
    if voxel <= 0 or len(pts) == 0:
        return np.arange(len(pts))
    key = np.floor(pts / float(voxel)).astype(np.int64)
    order = np.argsort(-keep_score)
    seen = {}
    keep = []
    for i in order:
        k = (int(key[i, 0]), int(key[i, 1]), int(key[i, 2]))
        if k in seen:
            continue
        seen[k] = int(i)
        keep.append(int(i))
    return np.asarray(sorted(keep), dtype=np.int64)


def solidify_surfels(means, scales, quats, opacity, colors, dims, voxel=0.0, op_floor=0.25):
    """把"贴片式"（视角相关）的高斯体素化成**体积代理**。

    实测动机：从重建里拼出来的 actor 资产仍是单目 surfel —— 从正上方看几乎没有投影面积、
    换视角就看不见。体素化给每个被占体素放一个各向同性高斯，填满物体体积，
    于是任意角度都能渲染出实体（位置/尺寸/颜色来自原数据，代价是外观块状化）。
    """
    dims = np.asarray(dims, dtype=np.float64)
    if voxel <= 0:
        m = float(min(dims)) if np.all(np.isfinite(dims)) else 0.4
        voxel = float(np.clip(m / 10.0, 0.08, 0.35))
    lo, hi = means.min(0), means.max(0)
    n = np.maximum(np.ceil((hi - lo) / voxel).astype(int), 1)
    if int(np.prod(n)) > 400000:
        voxel *= 1.6
        n = np.maximum(np.ceil((hi - lo) / voxel).astype(int), 1)
    key = np.floor((means - lo) / voxel).astype(np.int64)
    seen = {}
    for i in np.argsort(-opacity):
        k = (int(key[i, 0]), int(key[i, 1]), int(key[i, 2]))
        if k not in seen:
            seen[k] = int(i)
    if not seen:
        return means, scales, quats, opacity, colors
    idx = np.asarray(list(seen.values()), dtype=np.int64)
    keys = np.asarray(list(seen.keys()), dtype=np.float64)
    new_means = (lo + (keys + 0.5) * voxel).astype(np.float32)
    N = len(idx)
    return (new_means,
            np.full((N, 3), float(voxel) * 0.62, dtype=np.float32),
            np.tile(np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32), (N, 1)),
            np.clip(opacity[idx], op_floor, 1.0).astype(np.float32),
            colors[idx].astype(np.float32))


# ==================== 跨帧关联 ====================

def associate_tracks(frames_objs: Dict[int, Dict[int, dict]], fps: float = 10.0,
                     max_speed: float = 40.0, dim_tol: float = 0.6, min_frames: int = 3,
                     verbose: bool = False):
    """把"每帧重排的簇号"关联成持续 track（SORT 式：匀速预测 + 最近邻 + 尺寸相容）。"""
    tracks = []
    for t in sorted(frames_objs.keys()):
        preds = []
        for tr in tracks:
            if not tr["frames"]:
                continue
            last = tr["frames"][-1]
            dt = max(1e-3, (t - last["frame"]) / float(fps))
            p = np.asarray(last["pose"], dtype=np.float64)
            c_last = p[:3, 3] if p.ndim == 2 else p
            v = np.asarray(tr.get("vel", np.zeros(3)), dtype=np.float64)
            preds.append((tr, c_last + v * dt, float(np.linalg.norm(v))))
        used = set()
        for oid, meta in frames_objs[t].items():
            if meta.get("pose") is None:
                continue
            p = np.asarray(meta["pose"], dtype=np.float64)
            c = p[:3, 3] if p.ndim == 2 else p
            dims = np.asarray(meta.get("dims") or [4.5, 2.0, 1.6], dtype=np.float64)
            best, best_d = None, None
            for tr, c_pred, spd in preds:
                if id(tr) in used:
                    continue
                gate = max(2.5, 1.5 * min(spd, max_speed) / float(fps) + 2.0) \
                    + 0.5 * float(np.linalg.norm(dims[:2]))
                d = float(np.linalg.norm(c - c_pred))
                if d > gate:
                    continue
                dl = float(np.linalg.norm(np.asarray(tr["frames"][-1]["dims"], dtype=np.float64) - dims))
                if dl > dim_tol * float(np.linalg.norm(dims) + 1e-6) * 3.0:
                    continue
                if best_d is None or d < best_d:
                    best, best_d = tr, d
            if best is None:
                tracks.append({"track_id": len(tracks), "frames": [], "vel": np.zeros(3)})
                best = tracks[-1]
            used.add(id(best))
            prev = best["frames"][-1] if best["frames"] else None
            if prev is not None:
                dt = max(1e-3, (t - prev["frame"]) / float(fps))
                pp = np.asarray(prev["pose"], dtype=np.float64)
                cp = pp[:3, 3] if pp.ndim == 2 else pp
                best["vel"] = (c - cp) / dt
            best["frames"].append({"frame": int(t), "object_id": int(oid),
                                   "pose": meta.get("pose"), "dims": list(dims)})
    out = [tr for tr in tracks if len(tr["frames"]) >= int(min_frames)]
    for i, tr in enumerate(out):
        tr["track_id"] = i
    if verbose:
        print(f"[track] 关联出 {len(out)} 条持续 track"
              f"（原始逐帧簇 {sum(len(v) for v in frames_objs.values())} 个）")
    return out


# ==================== 主流程 ====================

def build_scene_asset_bank(scene_dir: str, out_dir: str, voxel: float = 0.04,
                           min_frames: int = 3, max_gaussians: int = 80000,
                           min_opacity: float = 0.05, verbose: bool = True,
                           clip_margin: float = 0.35, min_asset_gaussians: int = 200,
                           fps: float = 10.0, track: bool = True,
                           solidify: bool = True, solid_voxel: float = 0.0,
                           num_views: int = 0) -> Dict:
    """把一个重建场景的动态物体规范化为对象中心资产，写入 out_dir。"""
    scene_dir, out_dir = str(scene_dir), str(out_dir)
    gs_dir = os.path.join(scene_dir, "gaussians")
    obj_dir = os.path.join(scene_dir, "dynamic_objects")
    frames = sorted(
        int(f.split("_")[1]) for f in os.listdir(gs_dir)
        if f.startswith("frame_") and f.endswith("_dynamic.ply")
    ) if os.path.isdir(gs_dir) else []
    if not frames:
        raise FileNotFoundError(f"{gs_dir} 下没有 frame_XXXX_dynamic.ply")

    per_ply: Dict[int, Dict[int, dict]] = {}
    per_meta: Dict[int, Dict[int, dict]] = {}
    for t in frames:
        per_ply[t] = _read_dynamic_ply(os.path.join(gs_dir, f"frame_{t:04d}_dynamic.ply"))
        metas = {}
        mp = os.path.join(obj_dir, f"frame_{t:04d}_objects.json")
        if os.path.exists(mp):
            for o in json.load(open(mp)):
                try:
                    metas[int(o.get("object_id", -1))] = o
                except Exception:  # noqa: BLE001
                    pass
        per_meta[t] = metas

    # --- 多视角：flat = real * V + view ---
    if num_views <= 0:
        try:
            npy = len([f for f in os.listdir(scene_dir) if f.startswith("view_") and f.endswith(".npy")])
            png = len([f for f in os.listdir(scene_dir) if f.startswith("view_") and f.endswith(".png")])
            num_views = int(npy // png) if png else 1
        except Exception:  # noqa: BLE001
            num_views = 1
    num_views = max(1, int(num_views))
    owner = {t: {"real": t // num_views, "view": t % num_views} for t in frames}
    primary = [t for t in frames if owner[t]["view"] == 0]
    if verbose and num_views > 1:
        print(f"[asset] 多视角场景：num_views={num_views}, real frames={len(set(v['real'] for v in owner.values()))}")

    def _world_center(t, oid):
        pw = per_meta[t].get(oid, {}).get("pose_world")
        if pw is None:
            return None
        P = np.asarray(pw, dtype=np.float64)
        return P[:3, 3] if P.ndim == 2 else P

    frames_objs = {t: {oid: {"pose": per_meta[t].get(oid, {}).get("pose_world"),
                             "dims": per_meta[t].get(oid, {}).get("dimensions")}
                       for oid in per_ply[t].keys()}
                   for t in primary}
    if track:
        tracks = associate_tracks(frames_objs, fps=fps, min_frames=min_frames, verbose=verbose)
    else:
        # 不做跨帧关联时：每个 (帧, 物体) 各自成一条 track（调试用）。
        # 注意这里原来是 `m.get(...)`——`m` 在这个作用域根本不存在，会直接 NameError
        # （`python actor_assets.py --no_track` 必崩）。按原意取 frames_objs 里的那份。
        tracks = [{"track_id": i,
                   "frames": [{"frame": t, "object_id": oid,
                               "pose": frames_objs[t][oid].get("pose"),
                               "dims": frames_objs[t][oid].get("dims")}],
                   "vel": np.zeros(3)}
                  for i, (t, oid) in enumerate(sorted((t, oid) for t in primary for oid in per_ply[t]))]

    # 其它视角看到的同一物体并进来（角度覆盖更全）
    if num_views > 1:
        merged = 0
        for tr in tracks:
            for fr in tr["frames"]:
                t0 = int(fr["frame"])
                real = owner.get(t0, {}).get("real", t0 // num_views)
                c0 = _world_center(t0, fr["object_id"])
                if c0 is None:
                    continue
                for t2 in frames:
                    if owner[t2]["view"] == 0 or owner[t2]["real"] != real:
                        continue
                    for oid2 in per_ply.get(t2, {}).keys():
                        c2 = _world_center(t2, oid2)
                        if c2 is None or float(np.linalg.norm(c2 - c0)) > 1.5:
                            continue
                        fr.setdefault("extra", []).append({"frame": t2, "object_id": oid2})
                        merged += 1
        if verbose and merged:
            print(f"[asset] 跨视角并入 {merged} 个局部观测")

    os.makedirs(out_dir, exist_ok=True)
    assets = []
    for tr in tracks:
        entries = []
        for fr in tr["frames"]:
            arrs = per_ply.get(fr["frame"], {}).get(fr["object_id"])
            if arrs is not None:
                entries.append(arrs)
            for extra in fr.get("extra", []):
                a2 = per_ply.get(extra["frame"], {}).get(extra["object_id"])
                if a2 is not None:
                    entries.append(a2)
        if len(entries) < int(min_frames):
            continue

        means = np.concatenate([np.stack([e["x"], e["y"], e["z"]], 1) for e in entries], 0)
        sc = np.concatenate([np.stack([e["scale_0"], e["scale_1"], e["scale_2"]], 1) for e in entries], 0)
        quat = np.concatenate([np.stack([e["rot_0"], e["rot_1"], e["rot_2"], e["rot_3"]], 1) for e in entries], 0)
        op = np.concatenate([e["opacity"] for e in entries], 0)
        col = np.concatenate([np.stack([e["f_dc_0"], e["f_dc_1"], e["f_dc_2"]], 1) for e in entries], 0)

        # 尺寸取该 track 各帧 dims 的中位数；bbox 裁掉跑飞的孤立高斯
        dims_hist = np.asarray([f["dims"] for f in tr["frames"] if f.get("dims")], dtype=np.float64)
        dims_ref = np.median(dims_hist, axis=0) if len(dims_hist) else None
        if dims_ref is None or not np.all(np.isfinite(dims_ref)):
            q = np.percentile(means, [1, 99], axis=0)
            dims_ref = (q[1] - q[0]).astype(np.float64)
        half = np.asarray(dims_ref, dtype=np.float64) / 2.0 * (1.0 + float(clip_margin))
        inside = np.all(np.abs(means) <= half[None, :], axis=1)
        keep_ratio = float(inside.mean())
        means, sc, quat, op, col = means[inside], sc[inside], quat[inside], op[inside], col[inside]
        keep = op >= float(min_opacity)
        means, sc, quat, op, col = means[keep], sc[keep], quat[keep], op[keep], col[keep]
        if len(means) == 0:
            continue
        idx = _voxel_dedup(means, op, voxel)
        means, sc, quat, op, col = means[idx], sc[idx], quat[idx], op[idx], col[idx]

        # 一致性诊断：局部形状波动 + 世界轨迹跳变
        ext_hist = np.asarray([np.percentile(np.stack([e["x"], e["y"], e["z"]], 1), 99, axis=0)
                               - np.percentile(np.stack([e["x"], e["y"], e["z"]], 1), 1, axis=0)
                               for e in entries], dtype=np.float64)
        ext_cv = float(np.mean(np.std(ext_hist, axis=0) / (np.median(ext_hist, axis=0) + 1e-6))) \
            if len(ext_hist) > 1 else 0.0
        centres, fs = [], []
        for f in sorted(tr["frames"], key=lambda d: d["frame"]):
            if f.get("pose") is None:
                continue
            P = np.asarray(f["pose"], dtype=np.float64)
            centres.append(P[:3, 3] if P.ndim == 2 else P)
            fs.append(f["frame"])
        jumps = [float(np.linalg.norm(centres[i] - centres[i - 1])) for i in range(1, len(centres))]
        speeds = [jumps[i] / max(1e-3, (fs[i + 1] - fs[i]) / float(fps)) for i in range(len(jumps))]
        max_jump = max(jumps) if jumps else 0.0
        mean_speed = float(np.mean(speeds)) if speeds else 0.0
        span_m = float(np.linalg.norm(centres[-1] - centres[0])) if len(centres) > 1 else 0.0
        consistent = bool(ext_cv <= 0.75 and max_jump <= 3.0)

        n_surfels = int(len(means))
        if solidify:
            means, sc, quat, op, col = solidify_surfels(means, sc, quat, op, col, dims_ref,
                                                        voxel=float(solid_voxel))
        if len(means) > int(max_gaussians):
            sel = np.argsort(-op)[: int(max_gaussians)]
            means, sc, quat, op, col = means[sel], sc[sel], quat[sel], op[sel], col[sel]
        usable = bool(consistent and len(means) >= int(min_asset_gaussians))

        tid = int(tr["track_id"])
        name = f"{os.path.basename(scene_dir.rstrip('/'))}_track{tid:03d}"
        ply_path = os.path.join(out_dir, f"{name}.ply")
        _write_gs_ply(ply_path, means, sc, quat, op, col)
        assets.append({
            "asset_id": name,
            "source_type": "canonical_actor",
            # 绝对路径：渲染器只认存在的文件，相对路径在别的 cwd 下会被静默跳过
            "source_path": os.path.abspath(ply_path),
            "source_path_rel": ply_path,
            "default_pose": tr["frames"][0].get("pose") or np.eye(4).tolist(),
            "metadata": {
                "track_id": tid,
                "scene_dir": scene_dir,
                "num_views": num_views,
                "frames": [int(f["frame"]) for f in tr["frames"]],
                # (flat_frame, 该帧的簇号) → 便于把资产和 TrackManager/渲染器里的物体对起来
                "object_ids_per_frame": [[int(f["frame"]), int(f["object_id"])] for f in tr["frames"]],
                "real_frames": [int(owner.get(int(f["frame"]), {}).get("real", f["frame"])) for f in tr["frames"]],
                "poses_per_frame": [f.get("pose") for f in tr["frames"]],
                "num_frames": len(tr["frames"]),
                "num_gaussians": int(len(means)),
                "num_surfels_before_solidify": n_surfels,
                "solidified": bool(solidify),
                "dimensions": [round(float(v), 3) for v in dims_ref],
                "extent_local_xyz": [round(float(v), 3) for v in (means.max(0) - means.min(0)).tolist()],
                "centroid_local": [round(float(v), 4) for v in means.mean(0).tolist()],
                "mean_opacity": round(float(op.mean()), 4),
                "bbox_keep_ratio": round(float(keep_ratio), 4),
                "extent_cv": round(float(ext_cv), 4),
                "max_jump_m": round(float(max_jump), 3),
                "mean_speed_mps": round(float(mean_speed), 3),
                "span_m": round(float(span_m), 3),
                "consistent": consistent,
                "usable": usable,
            },
        })
        if verbose:
            print(f"[asset] track{tid:>3d}: frames={len(tr['frames']):3d} surfels={n_surfels:5d} "
                  f"gs={len(means):6d} keep={keep_ratio:4.2f} ext_cv={ext_cv:4.2f} jump={max_jump:4.2f}m "
                  f"v~{mean_speed:4.1f}m/s usable={usable} solid={solidify}")

    bank = {
        "schema": "SceneObjectAssetBank/1",
        "generator": "studio/backend/actor_assets.py",
        "scene_dir": scene_dir,
        "voxel": float(voxel),
        "fps": float(fps),
        "solidified": bool(solidify),
        "num_tracks": len(assets),
        "num_usable": sum(1 for a in assets if a["metadata"]["usable"]),
        "assets": assets,
    }
    bank_path = os.path.join(out_dir, "bank.json")
    with open(bank_path, "w", encoding="utf-8") as f:
        json.dump(bank, f, ensure_ascii=False, indent=2)
    if verbose:
        print(f"[asset] 写出 {len(assets)} 个资产（可用 {bank['num_usable']}）-> {bank_path}")
    return bank


# ==================== SAM3D 资产登记（统一资产库） ====================

SAM3D_SAMPLES = [
    {"asset_id": "sam3d_ego", "kind": "ego",
     "source_path": "/root/autodl-fs/dggt-main/sam-3d-objects/ego.ply",
     "note": "主车（Chrysler Pacifica 风格），按车长 4.9m 缩放"},
    {"asset_id": "sam3d_person_0", "kind": "pedestrian",
     "source_path": "/root/autodl-fs/dggt-main/sam-3d-objects/person_0.ply",
     "note": "行人模型 0，按身高 1.75m 缩放"},
    {"asset_id": "sam3d_person_1", "kind": "pedestrian",
     "source_path": "/root/autodl-fs/dggt-main/sam-3d-objects/person_1.ply",
     "note": "行人模型 1，按身高 1.75m 缩放"},
]


def register_sam3d_assets(out_path: str, extra=None, verbose: bool = True) -> Dict:
    """把已有的 SAM3D 真 3D 资产（主车 / 行人）登记进**同一份**资产库 schema。

    说明：这类资产是真正的三维模型（不是单目 surfel），任意视角都成立；
    与 `canonical_actor` 资产放一起，下游可以用同一套读取/摆放逻辑。
    """
    items = list(SAM3D_SAMPLES) + list(extra or [])
    assets = []
    for it in items:
        p = str(it["source_path"])
        if not os.path.exists(p):
            if verbose:
                print(f"[sam3d] 跳过（文件不存在）: {p}")
            continue
        ext = None
        try:
            from plyfile import PlyData
            v = PlyData.read(p)["vertex"]
            x = np.asarray(v["x"], dtype=np.float64)
            y = np.asarray(v["y"], dtype=np.float64)
            z = np.asarray(v["z"], dtype=np.float64)
            ext = [float(x.max() - x.min()), float(y.max() - y.min()), float(z.max() - z.min())]
        except Exception:  # noqa: BLE001
            pass
        assets.append({
            "asset_id": it["asset_id"],
            "source_type": "sam3d",
            "source_path": os.path.abspath(p),
            "default_pose": np.eye(4).tolist(),
            "metadata": {
                "kind": it.get("kind", "unknown"),
                "note": it.get("note", ""),
                "extent_units_xyz": [round(v, 4) for v in ext] if ext else None,
                # SAM3D canonical(up=-Z, front=-Y, right=+X) → DGGT 物体局部系
                "model_corr": [[-1.0, 0, 0], [0, 0, -1.0], [0, -1.0, 0]],
                "usable": True,
                "solidified": False,
            },
        })
    bank = {"schema": "SceneObjectAssetBank/1",
            "generator": "studio/backend/actor_assets.py::register_sam3d_assets",
            "kind": "sam3d_assets", "num_tracks": len(assets), "num_usable": len(assets),
            "assets": assets}
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(bank, f, ensure_ascii=False, indent=2)
    if verbose:
        print(f"[sam3d] 登记 {len(assets)} 个真 3D 资产 -> {out_path}")
    return bank


def main():
    ap = argparse.ArgumentParser(description="把重建场景的动态物体规范化为对象中心 3D 资产（资产库）")
    ap.add_argument("scene_dir", help="如 output/trust/scene001_cam0/001")
    ap.add_argument("--out", default="", help="输出目录（默认 <repo>/output/actor_assets/<scene 名>）")
    ap.add_argument("--voxel", type=float, default=0.04, help="去重体素边长")
    ap.add_argument("--min_frames", type=int, default=3)
    ap.add_argument("--max_gaussians", type=int, default=80000)
    ap.add_argument("--no_solidify", action="store_true",
                    help="不做体素化（保留原始单目 surfel；只适合就在原视角附近看）")
    ap.add_argument("--solid_voxel", type=float, default=0.0, help="体素化边长（米），0=自动")
    ap.add_argument("--num_views", type=int, default=0,
                    help="多视角场景的视角数（0=自动由 view_*.npy/png 推断）")
    ap.add_argument("--no_track", action="store_true", help="不做跨帧关联（调试用）")
    ap.add_argument("--register_sam3d", action="store_true",
                    help="额外把 SAM3D 真 3D 资产（主车/行人）登记进同一份资产库")
    args = ap.parse_args()
    out = args.out or os.path.join(str(ROOT), "output", "actor_assets",
                                   os.path.basename(args.scene_dir.rstrip("/")))
    bank = build_scene_asset_bank(args.scene_dir, out, voxel=args.voxel,
                                  min_frames=args.min_frames, max_gaussians=args.max_gaussians,
                                  solidify=not args.no_solidify, solid_voxel=args.solid_voxel,
                                  num_views=args.num_views, track=not args.no_track)
    if args.register_sam3d:
        register_sam3d_assets(os.path.join(out, "bank_sam3d.json"))
    print(json.dumps({"num_assets": len(bank["assets"]), "num_usable": bank["num_usable"],
                      "assets": [{"id": a["asset_id"], "gaussians": a["metadata"]["num_gaussians"],
                                  "dims": a["metadata"]["dimensions"],
                                  "usable": a["metadata"]["usable"]} for a in bank["assets"]]},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
