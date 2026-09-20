"""可信域运行时（P1 的"运行时"部分）：给任意自车/相机轨迹算 **per-frame trust + 覆盖率**，
并在可信度跌破阈值时给出 **越界截断点** 与覆盖率报告。

三块能力：

1. **无 GT 在线指标**（任意位姿都能算，不需要真实图像）：
   - `coverage_map`（复用 `novelview_trust`）：把该帧所有高斯中心投到画面，统计被命中的网格比例；
   - `pose_novelty`：当前相机相对"重建/训练视角"的偏离（横向位移 + 视线夹角），
     这是"离训练分布多远"的直接度量。
2. **有 GT 标定的 trust 模型**：用留出相机的真实误差（PSNR/SSIM/LPIPS）与 coverage 的
   成对样本拟合 `SSIM ≈ f(coverage)`（单调 saturating 曲线），把 coverage 换算成"预期画质"。
   报告里会给出样本数、拟合参数与 R² —— 预测是**数据标定**出来的，不是拍脑袋。
3. **轨迹级判定**：给定一条相机轨迹 → 逐帧 coverage/novelty/预期 SSIM →
   连续 `patience` 帧低于阈值就截断，输出 `coverage_report`
   （信任帧比例、截断帧、越界时的位姿偏移、以及可选的"最大安全横移"envelope）。

用法（CLI）：
    python trust_runtime.py --report output/trust/trust_scene001_cam0.json \
        --scene output/trust/scene001_cam0/001 --sweep 0,0.5,1,2,4 --out json
"""
from __future__ import annotations

import argparse
import scene_frames as _sf
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
import sys  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))

from novelview_trust import coverage_map  # noqa: E402


# ==================== 1) 拟合 trust 模型 ====================


def load_trust_rows(report_paths: Sequence[str]) -> List[Dict[str, Any]]:
    """把若干 `trust_report.json` 摊平成 (coverage, ssim, lpips, psnr, 位姿偏离) 样本。"""
    rows: List[Dict[str, Any]] = []
    for p in report_paths:
        if not os.path.exists(p):
            continue
        rep = json.load(open(p, encoding="utf-8"))
        for view in rep.get("per_view", []):
            for fr in view.get("frames", []):
                m = fr.get("metrics") or {}
                dv = fr.get("deviation") or {}
                rows.append({
                    "report": os.path.basename(p),
                    "camera": view.get("camera"),
                    "frame": fr.get("frame"),
                    "coverage": float(fr.get("coverage", 0.0)),
                    "psnr": float(m.get("psnr", float("nan"))),
                    "ssim": float(m.get("ssim", float("nan"))),
                    "lpips": float(m.get("lpips", float("nan"))),
                    "d_view_angle_deg": float(dv.get("d_view_angle_deg", 0.0)),
                    "d_trans_m": float(dv.get("d_trans_m", 0.0)),
                })
    return rows


def _fit_monotone(x: np.ndarray, y: np.ndarray) -> Tuple[Dict[str, float], float]:
    """拟合 y = y0 + (y1-y0) * x^p（x∈[0,1]，p>0）→ 单调递增的 saturating 曲线。

    用网格搜 p + 线性最小二乘解 y0/y1，避免引入额外依赖；返回 (params, r2)。
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if len(x) < 4:
        return {"y0": 0.0, "y1": 1.0, "p": 1.0, "n": int(len(x))}, float("nan")
    best = None
    for p in np.linspace(0.2, 3.0, 57):
        z = np.power(np.clip(x, 0, 1), p)
        A = np.stack([np.ones_like(z), z], axis=1)
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        pred = A @ coef
        ss_res = float(np.sum((y - pred) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2)) + 1e-12
        r2 = 1.0 - ss_res / ss_tot
        if best is None or r2 > best[0]:
            best = (r2, float(coef[0]), float(coef[1]), float(p))
    r2, y0, y1, p = best
    return {"y0": round(y0, 4), "y1": round(y1, 4), "p": round(p, 3), "n": int(len(x))}, round(r2, 4)


def _fit_two_feature(cov: np.ndarray, ang: np.ndarray, y: np.ndarray) -> Tuple[Dict[str, float], float]:
    """拟合 y = y0 + (y1-y0) * cov^p * exp(-ang/tau)（对 cov 单调增、对视角偏离单调减）。

    网格搜 (p, tau) + 线性最小二乘解 y0/y1。比"只有 coverage"的模型更贴近物理：
    同样的覆盖率，离训练视角越远画质越差。
    """
    if len(cov) < 6:
        return {"y0": 0.0, "y1": 1.0, "p": 1.0, "tau_deg": 45.0, "n": int(len(cov))}, float("nan")
    best = None
    for p in (0.6, 0.8, 1.0, 1.25, 1.5):
        for tau in (15.0, 25.0, 40.0, 60.0, 90.0, 180.0, 1e6):
            f = np.power(np.clip(cov, 0, 1), p) * np.exp(-np.clip(ang, 0, None) / tau)
            A = np.stack([np.ones_like(f), f], axis=1)
            coef, *_ = np.linalg.lstsq(A, y, rcond=None)
            pred = A @ coef
            ss_res = float(np.sum((y - pred) ** 2))
            ss_tot = float(np.sum((y - y.mean()) ** 2)) + 1e-12
            r2 = 1.0 - ss_res / ss_tot
            if best is None or r2 > best[0]:
                best = (r2, float(coef[0]), float(coef[1]), float(p), float(tau))
    r2, y0, y1, p, tau = best
    return {"y0": round(y0, 4), "y1": round(y1, 4), "p": round(p, 3),
            "tau_deg": (None if tau > 1e5 else round(tau, 1)), "n": int(len(cov))}, round(r2, 4)


def fit_trust_model(rows: Sequence[Dict[str, Any]], metric: str = "ssim") -> Dict[str, Any]:
    """用 (coverage, 视角偏离, metric) 成对样本拟合 trust 模型；同时给出单特征模型的 R² 做对比。"""
    xs, angs, ys = [], [], []
    for r in rows:
        c, v = r.get("coverage"), r.get(metric)
        if c is None or v is None or not np.isfinite(v):
            continue
        xs.append(float(c))
        angs.append(float(r.get("d_view_angle_deg", 0.0)))
        ys.append(float(v))
    cov, ang, y = np.asarray(xs), np.asarray(angs), np.asarray(ys)
    params, r2 = _fit_two_feature(cov, ang, y)
    single, r2_single = _fit_monotone(cov, y)
    return {"metric": metric, **params, "r2": r2,
            "r2_coverage_only": r2_single,
            "single_feature_params": single,
            "coverage_range": [round(float(cov.min()), 4), round(float(cov.max()), 4)] if len(cov) else [0.0, 0.0]}


def predict_trust(model: Dict[str, Any], coverage: float, view_angle_deg: float = 0.0) -> float:
    """由 (coverage, 视角偏离) 预测画质指标（默认 SSIM）。"""
    y0, y1, p = float(model["y0"]), float(model["y1"]), float(model["p"])
    tau = model.get("tau_deg")
    f = max(0.0, min(1.0, coverage)) ** p
    if tau:
        f *= math.exp(-max(0.0, float(view_angle_deg)) / float(tau))
    return float(y0 + (y1 - y0) * f)


# ==================== 2) 位姿"离训练分布多远" ====================


def training_poses(scene_dir: str, num_views: int = 1) -> List[np.ndarray]:
    """重建时用过的相机位姿（view 0 的所有真实帧）。"""
    ego_dir = os.path.join(scene_dir, "ego_pose")
    files = sorted(f for f in os.listdir(ego_dir) if f.endswith("_ego.json"))
    if num_views > 1:
        files = files[::num_views]
    out = []
    for f in files:
        d = json.load(open(os.path.join(ego_dir, f)))
        m = np.asarray(d["camera_extrinsics_world"], dtype=np.float64)
        if m.shape == (3, 4):
            m = np.vstack([m, np.array([[0.0, 0.0, 0.0, 1.0]])])
        out.append(m)
    return out


def pose_novelty(pose: np.ndarray, train: Sequence[np.ndarray]) -> Dict[str, float]:
    """当前相机相对最近训练视角的偏离：横向位移、纵向位移、高度差、视线夹角（度）。"""
    P = np.asarray(pose, dtype=np.float64)
    c = P[:3, 3]
    z = P[:3, 2] / (np.linalg.norm(P[:3, 2]) + 1e-9)
    best = None
    for T in train:
        ct = T[:3, 3]
        d = c - ct
        # 在"最近训练相机"的坐标系里表达位移
        R = T[:3, :3]
        d_local = R.T @ d
        zt = T[:3, 2] / (np.linalg.norm(T[:3, 2]) + 1e-9)
        ang = math.degrees(math.acos(max(-1.0, min(1.0, float(np.dot(z, zt))))))
        score = float(np.linalg.norm(d)) + 0.05 * ang
        if best is None or score < best[0]:
            best = (score, d_local, ang)
    if best is None:
        return {"d_lateral_m": 0.0, "d_up_m": 0.0, "d_forward_m": 0.0,
                "d_view_angle_deg": 0.0, "d_trans_m": 0.0}
    _, d_local, ang = best
    return {"d_lateral_m": round(float(d_local[0]), 3),
            "d_up_m": round(float(d_local[1]), 3),
            "d_forward_m": round(float(d_local[2]), 3),
            "d_view_angle_deg": round(float(ang), 2),
            "d_trans_m": round(float(np.linalg.norm(d_local)), 3)}


# ==================== 3) 轨迹级判定 / 截断 ====================


def trust_of_trajectory(scene_dir: str, poses: Sequence[Tuple[int, np.ndarray]],
                        K: np.ndarray, W: int, H: int, model: Dict[str, Any],
                        min_trust: float = 0.55, patience: int = 2,
                        cell: int = 8, num_views: int = 1) -> Dict[str, Any]:
    """逐帧算 coverage → 预测 trust → 给出截断点与覆盖率报告。

    poses: [(frame_idx, c2w_4x4), ...]（可以是"仿真里自车实际走出来的轨迹"）
    """
    train = training_poses(scene_dir, num_views=num_views)
    frames: List[Dict[str, Any]] = []
    truncate_at = None
    bad_streak = 0
    for f, pose in poses:
        cov = coverage_map(scene_dir, int(f), pose, K, W, H, cell=cell)
        nov = pose_novelty(pose, train)
        trust = predict_trust(model, cov["coverage"], nov["d_view_angle_deg"])
        ok = bool(trust >= float(min_trust))
        if ok:
            bad_streak = 0
        else:
            bad_streak += 1
            if truncate_at is None and bad_streak >= int(patience):
                truncate_at = int(f)
        frames.append({"frame": int(f), "coverage": cov["coverage"],
                       "num_gaussians": cov.get("num_gaussians", 0),
                       "pred_ssim": round(float(trust), 4), "trusted": ok, **nov})
    n = max(1, len(frames))
    trusted = sum(1 for x in frames if x["trusted"])
    rep = {
        "scene_dir": scene_dir,
        "num_frames": len(frames),
        "trusted_frames": trusted,
        "trusted_ratio": round(trusted / n, 4),
        "truncate_frame": truncate_at,
        "min_trust": float(min_trust),
        "model": model,
        "frames": frames,
    }
    if truncate_at is not None:
        bad = next(x for x in frames if x["frame"] == truncate_at)
        rep["truncate_reason"] = (f"coverage 过低：pred_ssim={bad['pred_ssim']} < {min_trust}"
                                 f"（连续 {patience} 帧）；此时离最近训练视角 "
                                 f"{bad['d_trans_m']}m / {bad['d_view_angle_deg']}°")
    return rep


def annotate_camera_track(scene_dir: str, poses: Sequence[Tuple[int, np.ndarray]],
                          K: np.ndarray, W: int, H: int, model: Dict[str, Any],
                          min_trust: float = 0.55, num_views: int = 1) -> Dict[str, Any]:
    """给"某条相机轨迹"打一个紧凑的可信度标注（供批量数据落盘用）。

    Returns: coverage 均值/最小、预测 SSIM 均值/最小、可信帧比例、首个不可信帧、平均位姿偏离。
    """
    rep = trust_of_trajectory(scene_dir, poses, K, W, H, model, min_trust=min_trust,
                              patience=2, num_views=num_views)
    frames = rep.get("frames", [])
    if not frames:
        return {"available": False}
    cov = [f["coverage"] for f in frames]
    ssim = [f["pred_ssim"] for f in frames]
    return {
        "available": True,
        "num_frames": len(frames),
        "coverage_mean": round(float(np.mean(cov)), 4),
        "coverage_min": round(float(np.min(cov)), 4),
        "pred_ssim_mean": round(float(np.mean(ssim)), 4),
        "pred_ssim_min": round(float(np.min(ssim)), 4),
        "trusted_ratio": rep.get("trusted_ratio"),
        "first_untrusted_frame": next((f["frame"] for f in frames if not f["trusted"]), None),
        "mean_novelty_trans_m": round(float(np.mean([f["d_trans_m"] for f in frames])), 3),
        "mean_view_angle_deg": round(float(np.mean([f["d_view_angle_deg"] for f in frames])), 2),
        "min_trust": float(min_trust),
    }


def lateral_sweep(scene_dir: str, offsets: Sequence[float], K: np.ndarray, W: int, H: int,
                  model: Dict[str, Any], min_trust: float = 0.55, frame: int = 0,
                  num_views: int = 1, fps: float = 10.0) -> Dict[str, Any]:
    """横向平移自车轨迹（模拟"自车为了避让/变道偏离了重建轨迹"）→ 可信域 envelope。"""
    _ego_d = _sf.load_ego_pose(scene_dir, int(frame), 0)
    if _ego_d is None:
        raise FileNotFoundError(f"找不到真实帧 {frame} 的自车位姿（多视角场景需按 flat_index 取）")
    ego = _ego_d
    c2w = np.asarray(ego["camera_extrinsics_world"], dtype=np.float64)
    if c2w.shape == (3, 4):
        c2w = np.vstack([c2w, np.array([[0.0, 0.0, 0.0, 1.0]])])
    train = training_poses(scene_dir, num_views=num_views)
    out = []
    max_safe = None
    for off in offsets:
        p = c2w.copy()
        p[:3, 3] = p[:3, 3] + c2w[:3, 0] * float(off)
        cov = coverage_map(scene_dir, int(frame), p, K, W, H)
        nov = pose_novelty(p, train)
        tr = predict_trust(model, cov["coverage"])
        ok = bool(tr >= float(min_trust))
        if ok:
            max_safe = float(off)
        out.append({"lateral_m": float(off), "coverage": cov["coverage"],
                    "pred_ssim": round(float(tr), 4), "trusted": ok, **nov})
    return {"scene_dir": scene_dir, "frame": int(frame), "min_trust": float(min_trust),
            "max_safe_lateral_m": max_safe, "rows": out}


def main():
    ap = argparse.ArgumentParser(description="可信域运行时：per-frame trust / 越界截断 / 覆盖率报告")
    ap.add_argument("--report", nargs="+", default=[], help="用于标定 trust 模型的 trust_report.json")
    ap.add_argument("--scene", default="", help="重建场景目录（算 coverage 用）")
    ap.add_argument("--frame", type=int, default=0)
    ap.add_argument("--num_views", type=int, default=1)
    ap.add_argument("--sweep", default="0,0.5,1,2,4", help="横向平移档位（米）")
    ap.add_argument("--min_trust", type=float, default=0.55)
    ap.add_argument("--out", default="", help="输出 json")
    args = ap.parse_args()

    rows = load_trust_rows(args.report)
    model = fit_trust_model(rows)
    result: Dict[str, Any] = {"num_rows": len(rows), "model": model}
    if args.scene:
        ego = _sf.load_ego_pose(args.scene, int(args.frame), 0)
        K = np.asarray(ego["camera_intrinsics"], dtype=np.float64)
        W, H = int(ego["camera"]["width"]), int(ego["camera"]["height"])
        offs = [float(x) for x in args.sweep.split(",") if x.strip() != ""]
        result["envelope"] = lateral_sweep(args.scene, offs, K, W, H, model,
                                           min_trust=args.min_trust, frame=args.frame,
                                           num_views=args.num_views)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        json.dump(result, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
