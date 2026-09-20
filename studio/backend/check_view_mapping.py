"""验证多视角场景的"真实帧↔flat 帧"映射：用**真实渲染 vs GT 图**的 SSIM 判定对错。

做法：对同一个真实帧 t，分别用
  (A) 直接用 frame_{t:04d}_ego.json（错误：多视角下这是别的相机）
  (B) flat_index(t, 0)（正确：view0 = 重建时的参考相机）
渲染自车视角，和 `data/waymo/processed/validation/<segment>/images/{t}_0.jpg` 比 SSIM/PSNR。
正确的那一路应该明显更高（本机 3 视角场景实测 ~0.85 vs ~0.1）。
"""
import argparse
import json
import os
import sys

import numpy as np

ROOT = "/root/autodl-fs/dggt-main"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "studio/backend"))

import scene_frames as SF  # noqa


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--segment", default="000")
    ap.add_argument("--frames", type=int, default=4)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    import torch
    import cv2
    from dggt_engine import DGGTRenderer

    r = DGGTRenderer(scene_path=args.scene, device="cuda", load_sky=True, verbose=False)
    nv = SF.num_views(args.scene)
    print(f"场景 {args.scene}：views={nv}（renderer.num_views={r.num_views}）")
    data_root = os.path.join(ROOT, "data/waymo/processed/validation", args.segment)
    rows = []
    for t in range(args.frames):
        gtp = os.path.join(data_root, "images", f"{t:03d}_0.jpg")
        if not os.path.exists(gtp):
            continue
        gt = cv2.cvtColor(cv2.imread(gtp), cv2.COLOR_BGR2RGB)
        out = {}
        for tag, path in (("raw", os.path.join(args.scene, "ego_pose", f"frame_{t:04d}_ego.json")),
                          ("flat", SF.ego_pose_path(args.scene, t, 0))):
            if not os.path.exists(path):
                out[tag] = None
                continue
            j = json.load(open(path))
            c2w = np.asarray(j["camera_extrinsics_world"], np.float64)
            if c2w.shape == (3, 4):
                c2w = np.vstack([c2w, np.array([[0.0, 0.0, 0.0, 1.0]])])
            K = np.asarray(j["camera_intrinsics"], np.float64)
            W, H = int(j["camera"]["width"]), int(j["camera"]["height"])
            img = r._render_frame_with_object_overrides(
                t, {}, c2w_override=torch.tensor(c2w, dtype=torch.float32, device=r.device),
                K_override=torch.tensor(K, dtype=torch.float32, device=r.device),
                width_override=W, height_override=H)
            img = np.asarray(img, np.uint8)
            if img.shape[:2] != gt.shape[:2]:
                img = cv2.resize(img, (gt.shape[1], gt.shape[0]))
            g = cv2.cvtColor(gt, cv2.COLOR_RGB2GRAY).astype(np.float64) / 255.0
            a = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float64) / 255.0
            mu1, mu2 = g.mean(), a.mean()
            v1, v2 = g.var(), a.var()
            cov = ((g - mu1) * (a - mu2)).mean()
            c1, c2 = 0.01 ** 2, 0.03 ** 2
            ssim = ((2 * mu1 * mu2 + c1) * (2 * cov + c2)) / ((mu1 ** 2 + mu2 ** 2 + c1) * (v1 + v2 + c2))
            mse = float(((g - a) ** 2).mean())
            psnr = 10 * np.log10(1.0 / max(mse, 1e-12))
            out[tag] = (round(float(ssim), 4), round(float(psnr), 2))
        rows.append((t, out.get("raw"), out.get("flat")))
        print(f"  帧 {t}: 直接用真实帧号 SSIM/PSNR = {out.get('raw')}   |  flat_index(,0) = {out.get('flat')}")
    if args.out:
        json.dump({"scene": args.scene, "num_views": nv,
                   "rows": [{"frame": a, "raw": b, "flat": c} for a, b, c in rows]},
                  open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    good = [r for r in rows if r[1] and r[2]]
    if good:
        mr = np.mean([r[1][0] for r in good]); mf = np.mean([r[2][0] for r in good])
        print(f"平均 SSIM：直接用真实帧号 {mr:.4f} | flat_index(,0) {mf:.4f} -> "
              + ("✓ 映射正确" if mf > mr else "✗ 反了"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
