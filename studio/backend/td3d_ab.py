"""俯视 3D「真模型 vs 方框替身」的**决定性对比**：把车摆到自车前方近处、俯视视野调小，
这样车在画面里足够大，肉眼能直接判断"是渲染出来的模型"还是"一个矩形"。

用法：python td3d_ab.py --scene <重建场景> --bank <bank.json> --out <目录> [--ahead 10] [--span 16]
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
import api_server as S  # noqa: E402
from dggt_engine import DGGTRenderer  # noqa: E402
from track_manager import TrackManager  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--bank", default="", help="资产库路径（默认自动查找）")
    ap.add_argument("--out", default=os.path.join(ROOT, "output", "asset_verify", "td3d_ab"))
    ap.add_argument("--ahead", type=float, default=10.0, help="把资产摆到自车前方多少米")
    ap.add_argument("--span", type=float, default=16.0, help="俯视纵向视野（米），越小车越大")
    ap.add_argument("--frame", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    r = DGGTRenderer(scene_path=args.scene, device="cuda", load_sky=True, verbose=False)
    tm = TrackManager(r)
    bank = None
    if args.bank:
        bank = json.load(open(args.bank, encoding="utf-8"))
    else:
        bank = S.load_actor_bank(args.scene)
    if not bank:
        print("没有找到 actor 资产库（先跑 actor_assets.py）")
        return 1
    usable = [a for a in bank["assets"] if (a.get("metadata") or {}).get("usable")]
    # 选一个"车"（最长边 3~6m 的资产）
    cars = [a for a in usable if 2.5 <= float((a["metadata"]["dimensions"] or [0, 0, 0])[2]) <= 6.5] or usable
    car = max(cars, key=lambda a: a["metadata"]["num_gaussians"])
    print(f"[td3d] 用资产 {car['asset_id']}（dims={car['metadata']['dimensions']}, "
          f"gs={car['metadata']['num_gaussians']}）")

    # 自车相机位姿 → 把车摆到前方 ahead 米、原地朝向与车道一致
    d = json.load(open(os.path.join(args.scene, "ego_pose", f"frame_{args.frame:04d}_ego.json")))
    c2w = np.asarray(d["camera_extrinsics_world"], dtype=np.float64)
    if c2w.shape == (3, 4):
        c2w = np.vstack([c2w, np.array([0.0, 0.0, 0.0, 1.0])])
    yaw = math.atan2(float(c2w[0, 2]), float(c2w[2, 2]))
    cy, sy = math.cos(yaw), math.sin(yaw)
    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], np.float32)
    target = c2w[:3, 3] + c2w[:3, 2] * args.ahead
    pose[:3, 3] = [float(target[0]), float(target[1]) - 0.6, float(target[2])]

    # 俯视相机：正上方朝下，框住自车与摆放点
    cen = 0.5 * (c2w[:3, 3] + target)
    height = max(10.0, 0.5 * args.span / math.tan(math.radians(35.0)) + 2.0)
    top = np.eye(4, dtype=np.float32)
    top[:3, 0] = (-1.0, 0.0, 0.0)
    top[:3, 1] = (0.0, 0.0, -1.0)
    top[:3, 2] = (0.0, -1.0, 0.0)
    top[:3, 3] = (float(cen[0]), float(cen[1]) + height, float(cen[2]))
    W, H = 480, 720
    K = np.array([[0.5 * W / math.tan(math.radians(35.0)), 0, W / 2],
                  [0, 0.5 * H / math.tan(math.radians(35.0)), H / 2], [0, 0, 1]], np.float32)

    extras = [{"ply_path": car["source_path"], "transform": pose, "synth_track_id": 1}]

    def render(with_asset: bool, draw_boxes: bool):
        exs = extras if with_asset else []
        # 引擎渲染（真模型）
        img = r._render_frame_with_object_overrides(
            args.frame, {}, extra_objects=[dict(e, transform=torch.tensor(np.asarray(e["transform"], np.float32),
                                                                          device=r.device)) for e in exs],
            include_dynamic=False,
            c2w_override=torch.tensor(top, device=r.device),
            K_override=torch.tensor(K, device=r.device),
            width_override=W, height_override=H)
        img = np.asarray(img, np.float32) / 255.0
        if draw_boxes:   # 叠加"方框替身"作为对照
            corners = S._object_top_face_corners(np.asarray(pose, np.float64),
                                                 car["metadata"]["dimensions"])
            pts, ok = S._project_points(corners, top, K, W, H)
            if ok and pts is not None:
                cv2.fillPoly(img, [np.round(pts).astype(np.int32)], (0.35, 0.38, 0.42))
                cv2.polylines(img, [np.round(pts).astype(np.int32)], True, (0.9, 0.92, 0.95), 2)
        return img

    img_model = render(True, False)
    img_box = render(False, True)
    def mass(a, b):
        return round(float(np.mean(np.abs(a - b))), 4)
    print(f"[td3d] 真模型 vs 方框替身 画面差异 = {mass(img_model, img_box)}")
    m = np.hstack([img_box, img_model])
    out = os.path.join(args.out, "model_vs_box.png")
    cv2.imwrite(out, cv2.cvtColor((np.clip(m, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
    print("->", out, "（左=方框替身，右=真模型；俯视视野", args.span, "米，车在自车前方", args.ahead, "米）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
