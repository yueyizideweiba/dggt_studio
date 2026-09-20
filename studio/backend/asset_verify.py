"""验证 actor 资产库：用对象中心资产替换逐帧动态高斯，检查
① 还原度（同视角下与原渲染的差异）② 新视角下是否仍有实体（对比"逐帧高斯退化成条纹"）
③ 可摆位（把某个 actor 平移到别处、渲染新视角）

用法：python asset_verify.py --scene <重建场景目录> --bank <bank.json> --out <图片输出目录>
"""
from __future__ import annotations

import argparse
import scene_frames as _sf
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from novelview_trust import compute_metrics  # noqa: E402


def load_bank(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def pose_at(asset: dict, frame: int):
    """资产在该帧的位姿（来自跟踪结果）；没有就退回 default_pose。"""
    md = asset.get("metadata", {})
    frames = md.get("frames") or []
    poses = md.get("poses_per_frame") or []
    for f, p in zip(frames, poses):
        if int(f) == int(frame) and p is not None:
            return np.asarray(p, dtype=np.float32)
    dp = asset.get("default_pose")
    return np.asarray(dp, dtype=np.float32) if dp is not None else np.eye(4, dtype=np.float32)


def resolve_asset_path(p: str) -> str:
    """资产 ply 路径解析：绝对路径优先，其次相对 ROOT。"""
    if os.path.isabs(p) and os.path.exists(p):
        return p
    cand = os.path.join(str(ROOT), p)
    return cand if os.path.exists(cand) else p


def build_extras(bank: dict, frame: int, shift: dict | None = None):
    """把资产变成引擎的 extra_objects（可对指定 track 施加平移/旋转）。"""
    extras = []
    shift = shift or {}
    for a in bank.get("assets", []):
        tid = a.get("metadata", {}).get("track_id")
        if tid is None or not a.get("metadata", {}).get("usable", False):
            continue
        pose = pose_at(a, frame)
        if tid in shift:
            d = np.asarray(shift[tid], dtype=np.float32)
            pose = pose.copy()
            pose[:3, 3] = pose[:3, 3] + d
        path = resolve_asset_path(a["source_path"])
        if not os.path.exists(path):
            print(f"[asset_verify] 警告：资产文件不存在，会被渲染器静默跳过 -> {a['source_path']}")
            continue
        extras.append({"ply_path": path, "transform": pose, "synth_track_id": int(tid)})
    return extras


def render(renderer, frame: int, extras, include_dynamic: bool, c2w=None, K=None, W=None, H=None):
    import torch
    kw = {}
    if c2w is not None:
        kw["c2w_override"] = torch.tensor(np.asarray(c2w, dtype=np.float32), device=renderer.device)
    if K is not None:
        kw["K_override"] = torch.tensor(np.asarray(K, dtype=np.float32), device=renderer.device)
    if W:
        kw["width_override"] = int(W)
    if H:
        kw["height_override"] = int(H)
    exs = []
    for e in (extras or []):
        e = dict(e)
        if e.get("transform") is not None:
            e["transform"] = torch.tensor(np.asarray(e["transform"], dtype=np.float32),
                                          device=renderer.device, dtype=torch.float32)
        exs.append(e)
    return renderer._render_frame_with_object_overrides(
        int(frame), {}, extra_objects=exs, include_dynamic=bool(include_dynamic), **kw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True, help="重建场景目录（含 gaussians/ego_pose）")
    ap.add_argument("--bank", required=True, help="bank.json")
    ap.add_argument("--out", default="", help="输出图片目录")
    ap.add_argument("--frame", type=int, default=0)
    ap.add_argument("--shift", default="", help="挪动某个 track，如 1:3,0,0（track:dx,dy,dz，单位米）")
    args = ap.parse_args()
    out_dir = args.out or os.path.join(str(ROOT), "output", "asset_verify")
    os.makedirs(out_dir, exist_ok=True)

    import cv2
    from dggt_engine import DGGTRenderer

    renderer = DGGTRenderer(scene_path=args.scene, device="cuda", load_sky=True, verbose=False)
    bank = load_bank(args.bank)
    frame = int(args.frame)
    ego = _sf.load_ego_pose(args.scene, int(frame), 0)
    c2w = np.asarray(ego["camera_extrinsics_world"], dtype=np.float32)
    if c2w.shape == (3, 4):
        c2w = np.vstack([c2w, np.array([[0, 0, 0, 1]], dtype=np.float32)])
    K = np.asarray(ego["camera_intrinsics"], dtype=np.float32)
    W, H = int(ego["camera"]["width"]), int(ego["camera"]["height"])

    shift = {}
    if args.shift:
        tid, vec = args.shift.split(":")
        shift[int(tid)] = [float(x) for x in vec.split(",")]

    # 静态背景基线（不含任何动态物体）→ 用"与基线的差异"衡量动态物体究竟画出来了多少
    def as_img(x):
        return np.asarray(x, dtype=np.float32) / 255.0

    img_static = as_img(render(renderer, frame, [], False, c2w, K, W, H))
    img_orig = as_img(render(renderer, frame, [], True, c2w, K, W, H))
    img_asset = as_img(render(renderer, frame, build_extras(bank, frame), False, c2w, K, W, H))
    # 新视角：从自车相机沿右侧平移 3m、抬高 1.5m
    dev = c2w.copy()
    dev[:3, 3] = dev[:3, 3] + c2w[:3, 0] * 3.0 + np.array([0.0, 1.5, 0.0], dtype=np.float32)
    img_orig_novel = as_img(render(renderer, frame, [], True, dev, K, W, H))
    img_asset_novel = as_img(render(renderer, frame, build_extras(bank, frame), False, dev, K, W, H))
    img_asset_moved = as_img(render(renderer, frame, build_extras(bank, frame, shift), False, dev, K, W, H))

    # 俯视相机（正上方朝下，90° FOV，覆盖所有动态物体的范围）：
    # 这是"逐帧动态高斯会退化成条纹"的典型视角
    pts = []
    for a in bank.get("assets", []):
        md = a.get("metadata", {})
        for f, p in zip(md.get("frames") or [], md.get("poses_per_frame") or []):
            if p is not None and int(f) == int(frame):
                P = np.asarray(p, dtype=np.float64)
                pts.append(P[:3, 3] if P.ndim == 2 else P)
    if pts:
        cen = np.mean(np.asarray(pts), axis=0)
        span_z = float(np.max(np.asarray(pts)[:, 2]) - np.min(np.asarray(pts)[:, 2])) + 20.0
    else:
        cen = c2w[:3, 3] + c2w[:3, 2] * 20.0
        span_z = 60.0
    height = max(12.0, 0.5 * span_z / math.tan(math.radians(45.0)) + 4.0)
    top = np.eye(4, dtype=np.float32)
    top[:3, 0] = (-1.0, 0.0, 0.0)
    top[:3, 1] = (0.0, 0.0, -1.0)
    top[:3, 2] = (0.0, -1.0, 0.0)
    top[:3, 3] = (float(cen[0]), float(cen[1]) + height, float(cen[2]))
    fov = math.radians(90.0)
    Kt = np.array([[0.5 * W / math.tan(fov / 2), 0.0, W / 2.0],
                   [0.0, 0.5 * H / math.tan(fov / 2), H / 2.0],
                   [0.0, 0.0, 1.0]], dtype=np.float32)
    img_static_top = as_img(render(renderer, frame, [], False, top, Kt, W, H))
    img_orig_top = as_img(render(renderer, frame, [], True, top, Kt, W, H))
    img_asset_top = as_img(render(renderer, frame, build_extras(bank, frame), False, top, Kt, W, H))

    def mass(a, b):
        return round(float(np.mean(np.abs(a - b))), 5)

    def cover(a, b, thr=0.02):
        return round(float(np.mean(np.abs(a - b).max(axis=2) > thr)), 4)

    rep = {
        "scene": args.scene, "bank": args.bank, "frame": frame,
        "num_usable_assets": sum(1 for a in bank.get("assets", []) if a.get("metadata", {}).get("usable")),
        "same_view_orig_vs_asset": compute_metrics(img_orig, img_asset),
        # 动态物体"画出来了多少"：与静态基线的差异强度 / 覆盖像素比例
        "dynamic_energy": {
            "ego_view_perframe": mass(img_orig, img_static),
            "ego_view_assets": mass(img_asset, img_static),
            "topdown_perframe": mass(img_orig_top, img_static_top),
            "topdown_assets": mass(img_asset_top, img_static_top),
        },
        "dynamic_coverage_px": {
            "ego_view_perframe": cover(img_orig, img_static),
            "ego_view_assets": cover(img_asset, img_static),
            "topdown_perframe": cover(img_orig_top, img_static_top),
            "topdown_assets": cover(img_asset_top, img_static_top),
        },
    }
    for name, im in (("orig_same", img_orig), ("asset_same", img_asset),
                     ("orig_novel", img_orig_novel), ("asset_novel", img_asset_novel),
                     ("asset_novel_moved", img_asset_moved),
                     ("orig_topdown", img_orig_top), ("asset_topdown", img_asset_top),
                     ("static_topdown", img_static_top)):
        cv2.imwrite(os.path.join(out_dir, f"f{frame:04d}_{name}.png"),
                    cv2.cvtColor((np.clip(im, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
    # 上下拼图便于人工检查
    stack = np.vstack([img_orig, img_asset, img_orig_top, img_asset_top, img_orig_novel, img_asset_moved])
    cv2.imwrite(os.path.join(out_dir, f"f{frame:04d}_montage.png"),
                cv2.cvtColor((np.clip(stack, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
    print(json.dumps(rep, ensure_ascii=False, indent=2))
    print("图片 ->", out_dir)


if __name__ == "__main__":
    main()
