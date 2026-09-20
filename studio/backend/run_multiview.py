"""可复现的多视角重建 + 评测一键脚本（P1 的 ③：把 5 相机多视角数据接入管线）。

做四件事，并把"用了什么命令、产出在哪、指标多少"写成一份 `run_manifest.json`：
1. **显存预算检查**：实测 24 张图（如 3 视角×8 帧）可行、40 张（5 视角×8 帧）会 OOM（24GB 卡）。
   本脚本按 `frames × views ≤ MAX_IMAGES(默认 24)` 把关，超了就报错而不是崩；
2. 调 `inference.py`（自动带上 PATH 上的 ninja 与 `PYTORCH_CUDA_ALLOC_CONF`）；
3. 可选 `--holdout C`：把相机 C 留出来不喂进去（真正的 novel-view 评测）；
4. 可选 `--eval`（跑 novelview_trust 算留出相机真实误差）与 `--bank`（跑 actor_assets 建资产库）。

用法：
    python run_multiview.py --scene 001 --cameras 0,1,2 --frames 8 --eval --holdout 3,4
    python run_multiview.py --scene 001 --cameras 0,1,2,3 --frames 4 --holdout 4 --eval --bank
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parents[2]
PY = sys.executable
MAX_IMAGES_DEFAULT = 24          # 24GB 卡上的安全预算（frames × views）
CKPT = "pretrained/model_latest_waymo.pt"


def _run(cmd: List[str], cwd: str, log) -> int:
    log.write(f"$ {' '.join(cmd)}\n")
    log.flush()
    env = dict(os.environ)
    # gsplat 需要 ninja 在 PATH 上；expandable_segments 减少碎片导致的 OOM
    env["PATH"] = str(Path(PY).parent) + os.pathsep + env.get("PATH", "")
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    t0 = time.time()
    p = subprocess.run(cmd, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT)
    log.write(f"[exit={p.returncode} 用时 {time.time() - t0:.1f}s]\n")
    log.flush()
    return p.returncode


def main():
    ap = argparse.ArgumentParser(description="多视角重建 + 评测一键脚本（可复现）")
    ap.add_argument("--scene", required=True, help="预处理场景名（如 001）")
    ap.add_argument("--cameras", default="0,1,2", help="喂进去的相机（逗号分隔）")
    ap.add_argument("--holdout", default="", help="留出不喂的相机（逗号分隔，用于 novel-view 评测）")
    ap.add_argument("--frames", type=int, default=8, help="每个样本的帧数（sequence_length）")
    ap.add_argument("--start_idx", type=int, default=0)
    ap.add_argument("--data_root", default=str(ROOT / "data/waymo/processed/validation"))
    ap.add_argument("--out_root", default=str(ROOT / "output/trust"))
    ap.add_argument("--run_name", default="")
    ap.add_argument("--max_images", type=int, default=MAX_IMAGES_DEFAULT)
    ap.add_argument("--eval", action="store_true", help="跑 novel-view 可信域评测")
    ap.add_argument("--bank", action="store_true", help="建 actor 资产库")
    ap.add_argument("--diffusion", action="store_true", help="启用 diffusion 渲染精修（需权重就位）")
    args = ap.parse_args()

    cams_in = [int(x) for x in args.cameras.split(",") if x.strip() != ""]
    holdouts = [int(x) for x in args.holdout.split(",") if x.strip() != ""]
    n_images = len(cams_in) * int(args.frames)
    if n_images > args.max_images:
        print(f"[budget] {len(cams_in)} 视角 × {args.frames} 帧 = {n_images} 张图 > 预算 "
              f"{args.max_images} → 会 OOM。请减帧数/视角（实测 24 可行、40 OOM）。", file=sys.stderr)
        return 2

    run_name = args.run_name or f"scene{args.scene}_cam{''.join(str(c) for c in cams_in)}_f{args.frames}"
    out_dir = os.path.join(args.out_root, run_name)
    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, "run.log")
    scene_dir = os.path.join(out_dir, args.scene)

    manifest = {
        "scene": args.scene, "cameras_in": cams_in, "holdout": holdouts,
        "frames": args.frames, "start_idx": args.start_idx,
        "num_images": n_images, "max_images": args.max_images,
        "diffusion": bool(args.diffusion), "out_dir": out_dir, "scene_dir": scene_dir,
        "steps": [],
    }
    with open(log_path, "w", encoding="utf-8") as log:
        cmd = [PY, "-u", "inference.py", "--image_dir", args.data_root,
               "--scene_names", str(args.scene), "--mode", "2", "--ckpt_path", CKPT,
               "--output_path", out_dir, "--sequence_length", str(args.frames),
               "--start_idx", str(args.start_idx), "--camera_ids", ",".join(str(c) for c in cams_in),
               "-images", "-depth"]
        if args.diffusion:
            cmd.append("-diffusion")
        manifest["inference_cmd"] = " ".join(cmd)
        rc = _run(cmd, str(ROOT), log)
        manifest["steps"].append({"name": "inference", "exit": rc})
        have_scene = os.path.isdir(os.path.join(scene_dir, "ego_pose")) and \
            os.path.isdir(os.path.join(scene_dir, "gaussians"))
        if rc != 0 and not have_scene:
            json.dump(manifest, open(os.path.join(out_dir, "run_manifest.json"), "w"), ensure_ascii=False, indent=2)
            print(f"推理失败，见 {log_path}", file=sys.stderr)
            return rc
        if rc != 0:
            log.write("[warn] 推理进程退出码非 0，但主产物已生成 → 继续评测/建库\n")
            print(f"[warn] 推理退出码 {rc}（主产物已生成，继续）", file=sys.stderr)

        if args.eval and holdouts:
            rep_path = os.path.join(args.out_root, f"trust_{run_name}.json")
            cmd = [PY, "-u", os.path.join(ROOT, "studio/backend/novelview_trust.py"),
                   "--scene_dir", scene_dir, "--segment", str(args.scene),
                   "--start_idx", str(args.start_idx), "--num_frames", str(args.frames),
                   "--ref_camera", str(cams_in[0]), "--heldout", ",".join(str(c) for c in holdouts),
                   "--out", rep_path]
            rc = _run(cmd, str(ROOT), log)
            manifest["steps"].append({"name": "trust_eval", "exit": rc, "report": rep_path})
            if rc == 0 and os.path.exists(rep_path):
                rep = json.load(open(rep_path, encoding="utf-8"))
                manifest["trust"] = {str(v["camera"]): {"ssim": v["mean_ssim"], "psnr": v["mean_psnr"],
                                                       "lpips": v["mean_lpips"],
                                                       "coverage": v["mean_coverage"]}
                                     for v in rep.get("per_view", [])}

        if args.bank:
            bank_dir = os.path.join(ROOT, "output", "actor_assets", run_name)
            cmd = [PY, os.path.join(ROOT, "studio/backend/actor_assets.py"), scene_dir, "--out", str(bank_dir)]
            rc = _run(cmd, str(ROOT), log)
            manifest["steps"].append({"name": "actor_assets", "exit": rc, "bank_dir": str(bank_dir)})
            bank_json = os.path.join(bank_dir, "bank.json")
            if rc == 0 and os.path.exists(bank_json):
                b = json.load(open(bank_json, encoding="utf-8"))
                manifest["actor_assets"] = {"num_tracks": b.get("num_tracks"),
                                            "num_usable": b.get("num_usable")}

    json.dump(manifest, open(os.path.join(out_dir, "run_manifest.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
