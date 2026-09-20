"""可摆位 actor 演示（闭环仿真最关键的一步）：把资产挪到重建覆盖良好的地方（自车车道前方），
从"自车视角"和"俯视"两个视角渲染，验证"资产可以按剧本重新摆位并可信渲染"。

用法：python asset_repose.py --scene <场景> --bank <bank.json> --track 4 --ahead 12 --out <目录>
"""
import argparse
import scene_frames as _sf
import json
import math
import os
import sys

import numpy as np

ROOT = "/root/autodl-fs/dggt-main"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "studio/backend"))

import cv2  # noqa: E402
import torch  # noqa: E402
from dggt_engine import DGGTRenderer  # noqa: E402
from asset_verify import resolve_asset_path  # noqa: E402


def render(r, frame, extras, include_dynamic, c2w, K, W, H):
    exs = []
    for e in extras:
        e = dict(e)
        e["transform"] = torch.tensor(np.asarray(e["transform"], np.float32), device=r.device)
        exs.append(e)
    return np.asarray(r._render_frame_with_object_overrides(
        frame, {}, extra_objects=exs, include_dynamic=include_dynamic,
        c2w_override=torch.tensor(np.asarray(c2w, np.float32), device=r.device),
        K_override=torch.tensor(np.asarray(K, np.float32), device=r.device),
        width_override=W, height_override=H), np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--bank", required=True)
    ap.add_argument("--track", type=int, default=-1)
    ap.add_argument("--frame", type=int, default=0)
    ap.add_argument("--ahead", type=float, default=12.0)
    ap.add_argument("--lateral", type=float, default=0.0)
    ap.add_argument("--out", default=os.path.join(ROOT, "output", "asset_verify", "repose"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    bank = json.load(open(args.bank))
    usable = [a for a in bank["assets"] if a["metadata"].get("usable")]
    asset = next((a for a in usable if a["metadata"]["track_id"] == args.track), None) if args.track >= 0 else None
    if asset is None:
        asset = max(usable, key=lambda a: a["metadata"]["num_gaussians"])
    tid = asset["metadata"]["track_id"]

    r = DGGTRenderer(scene_path=args.scene, device="cuda", load_sky=True, verbose=False)
    ego = _sf.load_ego_pose(args.scene, int(args.frame), 0)
    c2w = np.asarray(ego["camera_extrinsics_world"], np.float32)
    if c2w.shape == (3, 4):
        c2w = np.vstack([c2w, np.array([[0, 0, 0, 1]], np.float32)])
    K = np.asarray(ego["camera_intrinsics"], np.float32)
    W, H = int(ego["camera"]["width"]), int(ego["camera"]["height"])

    # 把资产摆到自车前方 args.ahead 米（自车相机系：z 前, x 右, y 下）
    target = c2w[:3, 3] + c2w[:3, 2] * args.ahead + c2w[:3, 0] * args.lateral
    pose = np.eye(4, dtype=np.float32)
    yaw = math.atan2(float(c2w[0, 2]), float(c2w[2, 2]))          # 与自车同向
    cy, sy = math.cos(yaw), math.sin(yaw)
    pose[:3, :3] = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], np.float32)
    # 高度用**资产原本的路面高度**（重建里物体真实落在路面上），不要瞎猜偏移，
    # 否则会出现"车浮在半空"。朝向与自车一致。
    y_orig = None
    for f, p in zip(asset["metadata"].get("frames") or [], asset["metadata"].get("poses_per_frame") or []):
        if p is not None and int(f) == int(args.frame):
            Pp = np.asarray(p, float)
            y_orig = float((Pp[:3, 3] if Pp.ndim == 2 else Pp)[1])
            break
    if y_orig is None:
        y_orig = float(np.asarray(asset["default_pose"], float)[1, 3])
    pose[:3, 3] = [float(target[0]), float(y_orig), float(target[2])]
    print(f"[repose] 使用资产原始高度 y={y_orig:.2f}（相机高度 y={float(c2w[1,3]):.2f}）")
    path = resolve_asset_path(asset["source_path"])
    extras = [{"ply_path": path, "transform": pose, "synth_track_id": int(tid)}]

    img_base = render(r, args.frame, [], False, c2w, K, W, H)          # 静态背景
    img_reposed = render(r, args.frame, extras, False, c2w, K, W, H)   # 摆位后的资产
    # 俯视：对着摆放点
    top = np.eye(4, dtype=np.float32)
    top[:3, 0] = (-1.0, 0.0, 0.0); top[:3, 1] = (0.0, 0.0, -1.0); top[:3, 2] = (0.0, -1.0, 0.0)
    top[:3, 3] = (float(target[0]), float(y_orig) + 16.0, float(target[2]))
    fov = math.radians(70.0)
    Kt = np.array([[0.5 * W / math.tan(fov / 2), 0, W / 2], [0, 0.5 * H / math.tan(fov / 2), H / 2], [0, 0, 1]], np.float32)
    img_top_base = render(r, args.frame, [], False, top, Kt, W, H)
    img_top_reposed = render(r, args.frame, extras, False, top, Kt, W, H)

    m = np.vstack([np.hstack([img_base, img_reposed]), np.hstack([img_top_base, img_top_reposed])])
    out = os.path.join(args.out, f"track{tid:03d}_repose_ahead{args.ahead:g}.png")
    cv2.imwrite(out, cv2.cvtColor(m, cv2.COLOR_RGB2BGR))
    d_ego = float(np.mean(np.abs(img_reposed.astype(np.float32) - img_base.astype(np.float32))))
    d_top = float(np.mean(np.abs(img_top_reposed.astype(np.float32) - img_top_base.astype(np.float32))))
    print(json.dumps({"track": tid, "dims": asset["metadata"]["dimensions"],
                      "ahead_m": args.ahead, "ego_view_delta": round(d_ego, 4),
                      "top_view_delta": round(d_top, 4), "out": out}, ensure_ascii=False))


if __name__ == "__main__":
    main()
