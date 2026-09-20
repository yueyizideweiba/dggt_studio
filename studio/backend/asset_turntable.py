"""资产"转台"验证：围着某个 actor 转几个角度渲染，对比
  (a) 逐帧动态高斯（重建原始形态，视角相关）
  (b) 对象中心资产（跨帧规范化 + 可摆位）
用法：python asset_turntable.py --scene <场景> --bank <bank.json> --track 4 --frame 0 --out <目录>
"""
import argparse
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
from asset_verify import build_extras, resolve_asset_path  # noqa: E402


def look_at(eye, target, up=(0.0, 1.0, 0.0)):
    f = np.asarray(target, float) - np.asarray(eye, float)
    f = f / (np.linalg.norm(f) + 1e-9)
    r = np.cross(np.asarray(up, float), f)
    r = r / (np.linalg.norm(r) + 1e-9)
    d = np.cross(f, r)
    m = np.eye(4)
    m[:3, 0], m[:3, 1], m[:3, 2], m[:3, 3] = r, d, f, np.asarray(eye, float)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--bank", required=True)
    ap.add_argument("--track", type=int, default=-1, help="-1 = 用 gaussians 最多的可用资产")
    ap.add_argument("--frame", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(ROOT, "output", "asset_verify", "turntable"))
    ap.add_argument("--dist", type=float, default=9.0)
    ap.add_argument("--elev", type=float, default=18.0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    bank = json.load(open(args.bank))
    usable = [a for a in bank["assets"] if a["metadata"].get("usable")]
    if not usable:
        raise SystemExit("没有可用资产")
    asset = next((a for a in usable if a["metadata"]["track_id"] == args.track), None) if args.track >= 0 else None
    if asset is None:
        asset = max(usable, key=lambda a: a["metadata"]["num_gaussians"])
    tid = asset["metadata"]["track_id"]
    md = asset["metadata"]
    pose = None
    for f, p in zip(md.get("frames") or [], md.get("poses_per_frame") or []):
        if p is not None and int(f) == int(args.frame):
            pose = np.asarray(p, float)
            break
    if pose is None:
        pose = np.asarray(asset["default_pose"], float)
    cen = pose[:3, 3]
    print(f"[turntable] track{tid} dims={md['dimensions']} gs={md['num_gaussians']} center={np.round(cen,2)}")

    r = DGGTRenderer(scene_path=args.scene, device="cuda", load_sky=True, verbose=False)
    W, H = 518, 350
    fov = math.radians(45.0)
    K = np.array([[0.5 * W / math.tan(fov / 2), 0, W / 2], [0, 0.5 * H / math.tan(fov / 2), H / 2], [0, 0, 1]], np.float32)
    extras = build_extras(bank, args.frame)
    rows = []
    for az in (0, 60, 120, 180, 240, 300):
        a = math.radians(az)
        eye = cen + np.array([math.sin(a) * args.dist, args.dist * math.tan(math.radians(args.elev)),
                              math.cos(a) * args.dist])
        c2w = look_at(eye, cen)
        img_pf = r._render_frame_with_object_overrides(
            args.frame, {}, include_dynamic=True,
            c2w_override=torch.tensor(c2w, dtype=torch.float32, device=r.device),
            K_override=torch.tensor(K, dtype=torch.float32, device=r.device),
            width_override=W, height_override=H)
        exs = []
        for e in extras:
            e = dict(e)
            e["transform"] = torch.tensor(np.asarray(e["transform"], np.float32), device=r.device)
            exs.append(e)
        img_as = r._render_frame_with_object_overrides(
            args.frame, {}, extra_objects=exs, include_dynamic=False,
            c2w_override=torch.tensor(c2w, dtype=torch.float32, device=r.device),
            K_override=torch.tensor(K, dtype=torch.float32, device=r.device),
            width_override=W, height_override=H)
        rows.append((np.asarray(img_pf, np.float32), np.asarray(img_as, np.float32), az))
    top = np.hstack([x[0] for x in rows])
    bot = np.hstack([x[1] for x in rows])
    m = np.vstack([top, bot]).astype(np.uint8)
    out = os.path.join(args.out, f"track{tid:03d}_frame{args.frame:04d}_turntable.png")
    cv2.imwrite(out, cv2.cvtColor(m, cv2.COLOR_RGB2BGR))
    print("上排=逐帧动态高斯（重建原形态），下排=对象中心资产；方位角:",
          [x[2] for x in rows])
    print("->", out)


if __name__ == "__main__":
    main()
