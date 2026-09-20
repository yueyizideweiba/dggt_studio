"""novel-view 可信域评测（P1 核心交付物）

目的：回答"自车/相机偏离训练轨迹后，重渲染的画面还可信吗"，并给闭环仿真提供
**每帧可信度 + 越界截断**依据。

两条互补的度量：

1. **有 GT 的 novel-view 误差**（可信度的"真值"）：
   用标定文件里的相机 rig（`extrinsics/C.txt`）把"留出相机"的位姿推算出来，
   在**只用部分视角重建**的场景上渲染该相机，与预处理后的真实图像比 PSNR/SSIM/LPIPS。
   —— 留出相机在重建时没被喂进去，所以这是货真价实的 novel view。

2. **无 GT 的覆盖度 / 稳定性**（可在线用、用于截断）：
   - `coverage`：把该帧所有高斯中心投影到画面上，看有多少网格被"命中"（重建支撑不足=空洞）；
   - `sensitivity`：把相机位姿扰动 δ（横移/偏航）后再渲染一次，比两幅图的差异
     （对稀疏区域，小小扰动就会导致画面大幅变化）。

输出 `trust_report.json`：per-camera / per-frame 的误差与覆盖度，以及"误差 ~ 位姿偏离"的
拟合曲线 → 可信域阈值（供 `ScenarioSpec.trust_envelope` 与 rollout 截断使用）。
"""
from __future__ import annotations

import argparse
import scene_frames as _sf
import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# ---------------- 相机标定 / rig ----------------


def load_cam_extrinsics(data_root: str, segment: str, camera: int) -> np.ndarray:
    """`extrinsics/C.txt`：相机在**自车系**下的位姿（4x4）。"""
    p = os.path.join(data_root, segment, "extrinsics", f"{camera}.txt")
    m = np.loadtxt(p)
    if m.shape == (3, 4):
        m = np.vstack([m, np.array([[0.0, 0.0, 0.0, 1.0]])])
    return m.astype(np.float64)


def load_cam_intrinsics(data_root: str, segment: str, camera: int, scale: float = 1.0) -> np.ndarray:
    """`intrinsics/C.txt`：fx, fy, cx, cy, ... → 3x3（乘 scale 可缩放到预处理后的分辨率）。"""
    p = os.path.join(data_root, segment, "intrinsics", f"{camera}.txt")
    v = np.atleast_1d(np.loadtxt(p)).ravel()
    fx, fy, cx, cy = [float(x) for x in v[:4]]
    K = np.array([[fx * scale, 0.0, cx * scale],
                  [0.0, fy * scale, cy * scale],
                  [0.0, 0.0, 1.0]], dtype=np.float64)
    return K


# Waymo/DGGT 预处理里的 `extrinsics/C.txt` 用的是 **x=前, y=左, z=上** 的相机系
# （实测：cam1/2 与 cam0 视线夹角 44.9°/45.0°，cam3/4 为 90.0°/90.7°，与 Waymo rig 完全一致），
# 而 DGGT 重建/渲染用的是 OpenCV 相机系（x=右, y=下, z=前）。
# 因此在把 rig 用到渲染位姿之前，必须做一次基变换共轭。
_CAL_TO_OPENCV = np.array([[0.0, -1.0, 0.0],
                           [0.0, 0.0, -1.0],
                           [1.0, 0.0, 0.0]], dtype=np.float64)


def _conj(T: np.ndarray) -> np.ndarray:
    """把标定系(x前,y左,z上)下的刚体变换转成 OpenCV 相机系(x右,y下,z前)。"""
    B = np.eye(4)
    B[:3, :3] = _CAL_TO_OPENCV
    return B @ np.asarray(T, dtype=np.float64) @ np.linalg.inv(B)


def rig_transform(data_root: str, segment: str, ref_camera: int, camera: int,
                  to_opencv: bool = True) -> np.ndarray:
    """ref 相机 → 目标相机 的相对位姿。

    `to_opencv=True`（默认）返回可直接左乘到 DGGT 相机位姿上的形式（OpenCV 系）。
    """
    a = load_cam_extrinsics(data_root, segment, ref_camera)
    b = load_cam_extrinsics(data_root, segment, camera)
    T = np.linalg.inv(a) @ b
    return _conj(T) if to_opencv else T


# ---------------- GT 图像预处理（与 DGGT 数据集一致） ----------------


def preprocess_gt_image(path: str, target_width: int = 518) -> np.ndarray:
    """复刻 datasets.dataset.load_and_preprocess_images：宽度缩到 518、保持比例、过高则中心裁剪。"""
    from PIL import Image
    img = Image.open(path).convert("RGB")
    w, h = img.size
    new_h = int(round(h * (target_width / w) / 14) * 14)
    img = img.resize((target_width, new_h), Image.Resampling.BICUBIC)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    if new_h > target_width:
        s = (new_h - target_width) // 2
        arr = arr[s:s + target_width]
    return arr  # (H, W, 3) in [0,1]


# ---------------- 指标 ----------------

_LPIPS = None


def _lpips_fn(device):
    global _LPIPS
    if _LPIPS is None:
        import lpips
        _LPIPS = lpips.LPIPS(net="alex").to(device).eval()
    return _LPIPS


def compute_metrics(img_a: np.ndarray, img_b: np.ndarray, device="cuda") -> Dict[str, float]:
    """PSNR / SSIM / LPIPS（输入 (H,W,3) float[0,1]）。"""
    import torch
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity
    a = np.clip(img_a, 0, 1).astype(np.float32)
    b = np.clip(img_b, 0, 1).astype(np.float32)
    h = min(a.shape[0], b.shape[0])
    w = min(a.shape[1], b.shape[1])
    a, b = a[:h, :w], b[:h, :w]
    psnr = float(peak_signal_noise_ratio(b, a, data_range=1.0))
    ssim = float(structural_similarity(b, a, channel_axis=2, data_range=1.0))
    ta = torch.from_numpy(b).permute(2, 0, 1).unsqueeze(0).to(device) * 2 - 1
    tb = torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0).to(device) * 2 - 1
    with torch.no_grad():
        lp = float(_lpips_fn(device)(ta, tb).mean().item())
    return {"psnr": round(psnr, 3), "ssim": round(ssim, 4), "lpips": round(lp, 4)}


# ---------------- 渲染 / 覆盖度 ----------------


def render_view(renderer, frame_idx: int, c2w: np.ndarray, K: np.ndarray,
                width: int, height: int, object_overrides=None, extra_objects=None) -> np.ndarray:
    """用引擎在给定相机位姿/内参下渲染一帧（返回 float[0,1] (H,W,3)）。"""
    import torch
    img = renderer._render_frame_with_object_overrides(
        int(frame_idx), object_overrides or {},
        c2w_override=torch.tensor(np.asarray(c2w, dtype=np.float32), device=renderer.device),
        K_override=torch.tensor(np.asarray(K, dtype=np.float32), device=renderer.device),
        width_override=int(width), height_override=int(height),
        extra_objects=extra_objects or [])
    return np.asarray(img, dtype=np.float32) / 255.0


def _ply_means(path: str) -> Optional[np.ndarray]:
    """只读位置列，返回 (N,3) float32。"""
    try:
        from plyfile import PlyData
        v = PlyData.read(path)["vertex"]
        return np.stack([np.asarray(v["x"]), np.asarray(v["y"]), np.asarray(v["z"])], 1).astype(np.float32)
    except Exception:  # noqa: BLE001
        return None


def coverage_map(scene_dir: str, frame_idx: int, c2w: np.ndarray, K: np.ndarray,
                 width: int, height: int, cell: int = 8) -> Dict[str, float]:
    """无 GT 覆盖度：把静态 + 动态高斯中心投到画面，统计被命中的网格比例。"""
    pts = []
    static = os.path.join(scene_dir, "gaussians", "static_scene.ply")
    if os.path.exists(static):
        m = _ply_means(static)
        if m is not None:
            pts.append(m)
    # 多视角场景：gaussians/dynamic_objects 是 flat 排列的（flat = real * V + view0）
    _nv = _sf.num_views(scene_dir)
    _flat = int(frame_idx) * max(1, _nv)
    dyn = os.path.join(scene_dir, "gaussians", f"frame_{_flat:04d}_dynamic.ply")
    meta = os.path.join(scene_dir, "dynamic_objects", f"frame_{_flat:04d}_objects.json")
    if not os.path.exists(dyn):
        dyn = os.path.join(scene_dir, "gaussians", f"frame_{int(frame_idx):04d}_dynamic.ply")
        meta = os.path.join(scene_dir, "dynamic_objects", f"frame_{int(frame_idx):04d}_objects.json")
    if os.path.exists(dyn) and os.path.exists(meta):
        m = _ply_means(dyn)
        if m is not None:
            objs = json.load(open(meta))
            ids = None
            try:
                from plyfile import PlyData
                v = PlyData.read(dyn)["vertex"]
                if "object_id" in v.data.dtype.names:
                    ids = np.asarray(v["object_id"])
            except Exception:  # noqa: BLE001
                ids = None
            if ids is not None:
                for o in objs:
                    oid = int(o.get("object_id", -1))
                    if oid < 0:
                        continue
                    mask = ids == oid
                    if not mask.any():
                        continue
                    pose = np.asarray(o["pose_world"], dtype=np.float64)
                    if pose.shape == (3, 4):
                        pose = np.vstack([pose, [0, 0, 0, 1]])
                    pts.append((m[mask] @ pose[:3, :3].T.astype(np.float32)) + pose[:3, 3].astype(np.float32))
    if not pts:
        return {"coverage": 0.0, "num_gaussians": 0}
    P = np.concatenate(pts, 0)
    R = np.asarray(c2w, dtype=np.float64)[:3, :3]
    t = np.asarray(c2w, dtype=np.float64)[:3, 3]
    cam = (P - t) @ R
    z = cam[:, 2]
    ok = z > 0.1
    if ok.sum() == 0:
        return {"coverage": 0.0, "num_gaussians": int(len(P))}
    K = np.asarray(K, dtype=np.float64)
    u = K[0, 0] * cam[ok, 0] / z[ok] + K[0, 2]
    v = K[1, 1] * cam[ok, 1] / z[ok] + K[1, 2]
    gi = np.floor(u / cell).astype(np.int64)
    gj = np.floor(v / cell).astype(np.int64)
    gw, gh = int(math.ceil(width / cell)), int(math.ceil(height / cell))
    inside = (gi >= 0) & (gi < gw) & (gj >= 0) & (gj < gh)
    hit = set(zip(gi[inside].tolist(), gj[inside].tolist()))
    return {"coverage": round(len(hit) / float(max(1, gw * gh)), 4),
            "num_gaussians": int(len(P))}


def pose_deviation(c2w_a: np.ndarray, c2w_b: np.ndarray) -> Dict[str, float]:
    """两个相机位姿的偏差，全部在 **A 相机的光学系**里表达。

    注意：不要假设"哪个轴是前/上"——不同标定文件的轴向约定不同（Waymo 的相机外参
    是 color 系，绕光轴的朝向可能落在 x/y 上）。这里给出与约定无关的量：
    位移三分量（A 相机系）、位移模长、**相对旋转角**（度）、以及"视线夹角"。
    """
    A, B = np.asarray(c2w_a, float), np.asarray(c2w_b, float)
    d = B[:3, 3] - A[:3, 3]
    R = np.linalg.inv(A[:3, :3]) @ B[:3, :3]
    angle = math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(R) - 1.0) / 2.0))))
    # 视线夹角：两个相机光轴（OpenCV 约定下是局部 +z）的夹角
    z_a = A[:3, 2] / (np.linalg.norm(A[:3, 2]) + 1e-9)
    z_b = B[:3, 2] / (np.linalg.norm(B[:3, 2]) + 1e-9)
    view_angle = math.degrees(math.acos(max(-1.0, min(1.0, float(np.dot(z_a, z_b))))))
    return {"t_in_ref_cam": [round(float(x), 3) for x in d],
            "d_trans_m": round(float(np.linalg.norm(d)), 3),
            "d_rot_deg": round(float(angle), 2),
            "d_view_angle_deg": round(float(view_angle), 2)}


# ---------------- 主评测 ----------------

def _calib_check(data_root: str, segment: str, scene_dir: str, start_idx: int,
                 ref_camera: int, target_width: int) -> Dict:
    """对比"重建预测的内参"与"标定文件的真实内参"（度量重投影/尺度误差的来源）。"""
    out: Dict = {}
    try:
        from PIL import Image
        p0 = os.path.join(data_root, segment, "images", f"{start_idx:03d}_{ref_camera}.jpg")
        w0, h0 = Image.open(p0).size
        K_gt = load_cam_intrinsics(data_root, segment, ref_camera, scale=target_width / float(w0))
        ego_p = _sf.ego_pose_path(scene_dir, 0, 0)
        d = json.load(open(ego_p))
        K_pred = np.asarray(d["camera_intrinsics"], dtype=np.float64)
        out = {
            "fx_gt": round(float(K_gt[0, 0]), 2), "fx_pred": round(float(K_pred[0, 0]), 2),
            "fx_rel_err": round(float(K_pred[0, 0] / K_gt[0, 0] - 1.0), 4),
            "cx_gt": round(float(K_gt[0, 2]), 2), "cx_pred": round(float(K_pred[0, 2]), 2),
            "cy_gt": round(float(K_gt[1, 2]), 2), "cy_pred": round(float(K_pred[1, 2]), 2),
            "scale_factor": round(float(d.get("scale_factor", 1.0)), 4),
            "pred_size": [int(d["camera"]["width"]), int(d["camera"]["height"])],
        }
    except Exception as e:  # noqa: BLE001
        out = {"error": str(e)}
    return out


def _make_refiner(kind: str):
    """取一个"渲染精修"函数：difix（真的扩散精修）/ stub（轻量去噪，仅验证 A/B 管道）。"""
    if kind in (None, "", "none"):
        return None
    if kind == "difix":
        import diffusion_refine as D
        st = D.status()
        if not st.get("ready"):
            raise RuntimeError(f"Difix 未就绪：{json.dumps(st.get('sd_turbo'), ensure_ascii=False)[:200]}")
        return lambda img: D.refine_image_rgb(img)
    if kind == "stub":
        def _stub(img):
            import cv2
            x = (np.clip(img, 0, 1) * 255).astype(np.uint8)
            y = cv2.bilateralFilter(x, 5, 40, 40)
            y = cv2.addWeighted(y, 1.15, cv2.GaussianBlur(y, (0, 0), 1.0), -0.15, 0)
            return np.clip(y.astype(np.float32) / 255.0, 0, 1)
        return _stub
    raise ValueError(f"未知精修类型: {kind}")


def evaluate(data_root: str, segment: str, scene_dir: str, start_idx: int, num_frames: int,
             ref_camera: int, heldout_cameras: List[int], out_json: str,
             target_width: int = 518, sensitivity_step_m: float = 1.0,
             sensitivity_yaw_deg: float = 5.0, include_ref: bool = True,
             refine: str = "none") -> Dict:
    from dggt_engine import DGGTRenderer

    renderer = DGGTRenderer(scene_path=scene_dir, device="cuda", load_sky=True, verbose=False)
    refiner = _make_refiner(refine)
    report: Dict = {"segment": segment, "refine": refine, "scene_dir": scene_dir, "start_idx": start_idx,
                    "num_frames": num_frames, "ref_camera": ref_camera, "per_view": [],
                    "calibration": _calib_check(data_root, segment, scene_dir, start_idx, ref_camera,
                                                target_width)}

    # ---- 参考视角（=重建时喂进去的那台相机，位姿就是训练位姿）：这是可信域的"上界锚点" ----
    if include_ref:
        rows = []
        for i in range(num_frames):
            ego_p = _sf.ego_pose_path(scene_dir, i, 0)
            gtp = os.path.join(data_root, segment, "images", f"{start_idx + i:03d}_{ref_camera}.jpg")
            if not (os.path.exists(ego_p) and os.path.exists(gtp)):
                continue
            d = json.load(open(ego_p))
            c2w = np.asarray(d["camera_extrinsics_world"], dtype=np.float64)
            if c2w.shape == (3, 4):
                c2w = np.vstack([c2w, np.array([[0.0, 0.0, 0.0, 1.0]])])
            K = np.asarray(d["camera_intrinsics"], dtype=np.float64)
            W, H = int(d["camera"]["width"]), int(d["camera"]["height"])
            img = render_view(renderer, i, c2w, K, W, H)
            gt = preprocess_gt_image(gtp, target_width)
            met = compute_metrics(gt, img)
            met_ref = None
            if refiner is not None:
                try:
                    met_ref = compute_metrics(gt, refiner(img))
                except Exception as e:  # noqa: BLE001
                    print(f"[trust] 精修失败（参考视角）frame{start_idx + i}: {e}")
            cov = coverage_map(scene_dir, i, c2w, K, W, H)
            rows.append({"frame": start_idx + i, "metrics": met, "metrics_refined": met_ref,
                         "coverage": cov["coverage"],
                         "deviation": pose_deviation(c2w, c2w), "sensitivity": 0.0})
        if rows:
            _ref_entry = {
                "camera": f"{ref_camera}(ref)", "is_reference": True,
                "mean_psnr": round(float(np.mean([r["metrics"]["psnr"] for r in rows])), 3),
                "mean_ssim": round(float(np.mean([r["metrics"]["ssim"] for r in rows])), 4),
                "mean_lpips": round(float(np.mean([r["metrics"]["lpips"] for r in rows])), 4),
                "mean_coverage": round(float(np.mean([r["coverage"] for r in rows])), 4),
                "frames": rows,
            }
            _rr = [r["metrics_refined"] for r in rows if r.get("metrics_refined")]
            if _rr:
                _ref_entry["refined"] = {
                    "num": len(_rr),
                    "mean_psnr": round(float(np.mean([m["psnr"] for m in _rr])), 3),
                    "mean_ssim": round(float(np.mean([m["ssim"] for m in _rr])), 4),
                    "mean_lpips": round(float(np.mean([m["lpips"] for m in _rr])), 4),
                }
                _ref_entry["refined"]["delta_psnr"] = round(
                    _ref_entry["refined"]["mean_psnr"] - _ref_entry["mean_psnr"], 3)
                _ref_entry["refined"]["delta_ssim"] = round(
                    _ref_entry["refined"]["mean_ssim"] - _ref_entry["mean_ssim"], 4)
                _ref_entry["refined"]["delta_lpips"] = round(
                    _ref_entry["refined"]["mean_lpips"] - _ref_entry["mean_lpips"], 4)
            report["per_view"].append(_ref_entry)

    for cam in heldout_cameras:
        T_rel = rig_transform(data_root, segment, ref_camera, cam)
        # 该相机预处理后的分辨率
        gt0 = os.path.join(data_root, segment, "images", f"{start_idx:03d}_{cam}.jpg")
        if not os.path.exists(gt0):
            continue
        from PIL import Image
        w0, h0 = Image.open(gt0).size
        new_h = int(round(h0 * (target_width / w0) / 14) * 14)
        W_img = target_width
        H_img = min(new_h, target_width)
        K_cam = load_cam_intrinsics(data_root, segment, cam, scale=target_width / float(w0))
        rows = []
        for i in range(num_frames):
            f = start_idx + i
            ego_p = _sf.ego_pose_path(scene_dir, i, 0)
            if not os.path.exists(ego_p):
                continue
            d = json.load(open(ego_p))
            c2w_ref = np.asarray(d["camera_extrinsics_world"], dtype=np.float64)
            if c2w_ref.shape == (3, 4):
                c2w_ref = np.vstack([c2w_ref, [0, 0, 0, 1]])
            c2w_cam = c2w_ref @ T_rel
            img = render_view(renderer, i, c2w_cam, K_cam, W_img, H_img)
            gtp = os.path.join(data_root, segment, "images", f"{f:03d}_{cam}.jpg")
            if not os.path.exists(gtp):
                continue
            gt = preprocess_gt_image(gtp, target_width)
            met = compute_metrics(gt, img)
            cov = coverage_map(scene_dir, i, c2w_cam, K_cam, W_img, H_img)
            dev = pose_deviation(c2w_ref, c2w_cam)
            # 稳定性：位姿扰动后重渲染的差异
            c2w_p = c2w_cam.copy()
            c2w_p[0, 3] += sensitivity_step_m
            img_p = render_view(renderer, i, c2w_p, K_cam, W_img, H_img)
            sens = float(np.mean(np.abs(img_p - img)))
            met_ref = None
            if refiner is not None:
                try:
                    met_ref = compute_metrics(gt, refiner(img))
                except Exception as e:  # noqa: BLE001
                    print(f"[trust] 精修失败 cam{cam} frame{f}: {e}")
            rows.append({"frame": f, "metrics": met, "metrics_refined": met_ref,
                         "coverage": cov["coverage"],
                         "deviation": dev, "sensitivity": round(sens, 4)})
        if rows:
            entry = {
                "camera": cam,
                "T_ref_to_cam": np.asarray(T_rel).round(4).tolist(),
                "mean_psnr": round(float(np.mean([r["metrics"]["psnr"] for r in rows])), 3),
                "mean_ssim": round(float(np.mean([r["metrics"]["ssim"] for r in rows])), 4),
                "mean_lpips": round(float(np.mean([r["metrics"]["lpips"] for r in rows])), 4),
                "mean_coverage": round(float(np.mean([r["coverage"] for r in rows])), 4),
                "frames": rows,
            }
            ref_rows = [r["metrics_refined"] for r in rows if r.get("metrics_refined")]
            if ref_rows:
                entry["refined"] = {
                    "num": len(ref_rows),
                    "mean_psnr": round(float(np.mean([m["psnr"] for m in ref_rows])), 3),
                    "mean_ssim": round(float(np.mean([m["ssim"] for m in ref_rows])), 4),
                    "mean_lpips": round(float(np.mean([m["lpips"] for m in ref_rows])), 4),
                }
                entry["refined"]["delta_psnr"] = round(entry["refined"]["mean_psnr"] - entry["mean_psnr"], 3)
                entry["refined"]["delta_ssim"] = round(entry["refined"]["mean_ssim"] - entry["mean_ssim"], 4)
                entry["refined"]["delta_lpips"] = round(entry["refined"]["mean_lpips"] - entry["mean_lpips"], 4)
            report["per_view"].append(entry)
    if out_json:
        os.makedirs(os.path.dirname(os.path.abspath(out_json)), exist_ok=True)
        with open(out_json, "w", encoding="utf-8") as fp:
            json.dump(report, fp, ensure_ascii=False, indent=2)
    return report


def main():
    ap = argparse.ArgumentParser(description="novel-view 可信域评测（留出相机 vs 真实图像）")
    ap.add_argument("--scene_dir", required=True, help="重建输出目录（含 gaussians/ego_pose）")
    ap.add_argument("--data_root", default=str(ROOT / "data/waymo/processed/validation"))
    ap.add_argument("--segment", required=True, help="该重建对应的预处理场景名（如 001）")
    ap.add_argument("--start_idx", type=int, default=0, help="重建用到的第一个原始帧")
    ap.add_argument("--num_frames", type=int, default=8)
    ap.add_argument("--ref_camera", type=int, default=0, help="重建时喂进去的相机（作为参考系）")
    ap.add_argument("--heldout", default="1,2,3,4", help="留出相机（逗号分隔）")
    ap.add_argument("--out", default="", help="trust_report.json 路径")
    ap.add_argument("--no_ref", action="store_true", help="不评测参考视角（重建上界锚点）")
    ap.add_argument("--refine", default="none", choices=["none", "difix", "stub"],
                    help="渲染精修（difix=扩散精修；stub=轻量去噪，仅验证 A/B 管道）")
    args = ap.parse_args()
    held = [int(x) for x in args.heldout.split(",") if x.strip() != ""]
    rep = evaluate(args.data_root, args.segment, args.scene_dir, args.start_idx, args.num_frames,
                   args.ref_camera, held, args.out, include_ref=not args.no_ref,
                   refine=args.refine)
    brief = {}
    for v in rep["per_view"]:
        d = {"psnr": v["mean_psnr"], "ssim": v["mean_ssim"],
             "lpips": v["mean_lpips"], "coverage": v["mean_coverage"]}
        if v.get("refined"):
            d["refined"] = v["refined"]
        brief[str(v["camera"])] = d
    print(json.dumps({"segment": rep["segment"], "num_frames": rep["num_frames"],
                      "heldout": brief}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
