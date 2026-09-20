"""闭环仿真能力**一键自检脚本**：把这一阶段做的所有能力串起来跑一遍，输出一份总报告。

依次执行（全部走库函数，不依赖后端服务）：
  1. 可信域模型标定（用 `output/trust/trust_*.json` 里留出视角的真实误差）
  2. 可信域 envelope（横向平移 sweep → `max_safe_lateral_m`）
  3. actor 资产库构建（对象中心 / 体素化资产，供俯视 3D 与可摆位渲染）
  4. 相机 rig 传感器渲染（Waymo 多相机标定 → 多视角观测）
  5. 闭环 rollout（直行 / 转向两组动作 → 逐帧 trust + 越界截断 + 覆盖率报告）
  6. 标准格式导出（OpenSCENARIO + CommonRoad + 世界坐标 JSON，含 round-trip 校验）
  7. 汇总成 `closedloop_demo_report.json`

用法：
    python run_closedloop_demo.py --scene <重建场景目录> --segment 001 \
        [--scenario rear-end] [--cameras 0,1,2,3,4] [--steps 6] [--out_dir output/demo]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import actor_assets            # noqa: E402
import diffusion_refine        # noqa: E402
import trust_runtime           # noqa: E402


def _t0() -> float:
    return time.time()


def main():
    ap = argparse.ArgumentParser(description="闭环仿真能力一键自检")
    ap.add_argument("--scene", required=True, help="重建场景目录（含 gaussians/ego_pose）")
    ap.add_argument("--segment", default="001", help="标定来源的预处理场景名")
    ap.add_argument("--data_root", default=str(ROOT / "data/waymo/processed/validation"))
    ap.add_argument("--scenario", default="rear-end", help="导出用的事故类型（空字符串=不生成）")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--frames", type=int, default=8)
    ap.add_argument("--cameras", default="0,1,2,3,4")
    ap.add_argument("--rig_cameras", default="0,1,2", help="rig 渲染的相机（少渲几台更快）")
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--min_trust_frac", type=float, default=0.8)
    ap.add_argument("--out_dir", default=str(ROOT / "output" / "demo"))
    ap.add_argument("--skip_export", action="store_true")
    ap.add_argument("--skip_rig", action="store_true")
    # ↓ 可选：导出后把场景直接送进 CARLA 渲染事故回放视频（需先起 CARLA server）
    ap.add_argument("--carla", action="store_true",
                    help="导出后再把场景送进 CARLA 渲染 MP4（carla_bridge/run_bridge.sh）")
    ap.add_argument("--carla_out", default=None, help="CARLA 产物目录（默认 output/carla/<tag>）")
    ap.add_argument("--carla_map", default="Town10HD_Opt", help="CARLA 内置地图")
    ap.add_argument("--carla_focus", default="crash", choices=["crash", "ego", "all"],
                    help="CARLA 相机跟随目标")
    args = ap.parse_args()

    scene = args.scene if os.path.isabs(args.scene) else str((ROOT / args.scene).resolve())
    parent = os.path.basename(os.path.dirname(os.path.normpath(scene)))
    name = os.path.basename(os.path.normpath(scene))
    tag = f"{parent}_{name}" if parent else name      # 避免不同重建（cam0/cam012）共用同名目录
    out_dir = Path(args.out_dir) / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    report: Dict[str, Any] = {"scene": scene, "segment": args.segment, "steps": {}}

    # ---- 1. trust 模型 ----
    t = _t0()
    reports = sorted(str(p) for p in (ROOT / "output" / "trust").glob("trust_*.json"))
    rows = trust_runtime.load_trust_rows(reports) if reports else []
    model = trust_runtime.fit_trust_model(rows)
    report["steps"]["trust_model"] = {"seconds": round(_t0() - t, 1), "reports": [os.path.basename(r) for r in reports],
                                      "num_rows": len(rows), "model": model}
    print(f"[1/7] trust 模型：样本 {len(rows)}，R²={model.get('r2')}（仅 coverage {model.get('r2_coverage_only')}）")

    # ---- 2. 可信域 envelope ----
    t = _t0()
    from dggt_engine import DGGTRenderer
    from track_manager import TrackManager
    renderer = DGGTRenderer(scene_path=scene, device="cuda", load_sky=True, verbose=False)
    tm = TrackManager(renderer)
    ego = json.load(open(os.path.join(scene, "ego_pose", "frame_0000_ego.json")))
    K = np.asarray(ego["camera_intrinsics"], dtype=np.float64)
    W, H = int(ego["camera"]["width"]), int(ego["camera"]["height"])
    env = trust_runtime.lateral_sweep(scene, [0, 0.5, 1, 1.5, 2, 3, 4, 6], K, W, H, model,
                                      min_trust=0.55, frame=0)
    report["steps"]["trust_envelope"] = {"seconds": round(_t0() - t, 1),
                                        "max_safe_lateral_m": env["max_safe_lateral_m"],
                                        "rows": [{k: r[k] for k in ("lateral_m", "coverage", "pred_ssim", "trusted")}
                                                 for r in env["rows"]]}
    print(f"[2/7] 可信域：max_safe_lateral={env['max_safe_lateral_m']}m")

    # ---- 3. actor 资产库 ----
    t = _t0()
    bank_dir = ROOT / "output" / "actor_assets" / tag
    bank = actor_assets.build_scene_asset_bank(scene, str(bank_dir), verbose=False)
    report["steps"]["actor_bank"] = {"seconds": round(_t0() - t, 1), "out_dir": str(bank_dir),
                                     "num_tracks": bank["num_tracks"], "num_usable": bank["num_usable"],
                                     "assets": [{"asset_id": a["asset_id"],
                                                 "track_id": a["metadata"]["track_id"],
                                                 "gaussians": a["metadata"]["num_gaussians"],
                                                 "dimensions": a["metadata"]["dimensions"],
                                                 "usable": a["metadata"]["usable"]}
                                                for a in bank["assets"]]}
    print(f"[3/7] actor 资产库：{bank['num_tracks']} 条 track，可用 {bank['num_usable']}（{bank_dir}）")

    # ---- 4. 相机 rig ----
    if not args.skip_rig:
        t = _t0()
        import sensor_rig
        cams = [int(x) for x in args.rig_cameras.split(",") if x.strip() != ""]
        rig = sensor_rig.load_rig(args.data_root, args.segment, cams)
        imgs = sensor_rig.render_rig_frame(renderer, 0, rig, cams[0], cams, size=(960, 640))
        for c, img in imgs.items():
            import cv2
            cv2.imwrite(str(out_dir / f"rig_cam{c}.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        import math
        report["steps"]["sensor_rig"] = {
            "seconds": round(_t0() - t, 1), "cameras": cams,
            "view_angle_vs_ref_deg": {int(c): round(math.degrees(math.acos(max(-1.0, min(1.0,
                float(sensor_rig.camera_to_ref(rig, cams[0], int(c))[2, 2]))))), 2) for c in cams},
            "out_dir": str(out_dir)}
        print(f"[4/7] 相机 rig：{len(imgs)} 台相机已渲染（{out_dir}）")

    # ---- 5. 闭环 rollout ----
    t = _t0()
    import sim_runtime
    rollouts = {}
    for label, act in (("straight", {"speed": 8.0, "steer": 0.0}),
                       ("aggressive", {"speed": 12.0, "steer": 0.3})):
        sim = sim_runtime.ScenarioSim(scene, min_trust_frac=args.min_trust_frac,
                                      max_steps=args.steps, width=640, height=440)
        obs = sim.reset()
        log = [{"step": 0, "frame": obs["frame_idx"], "coverage": obs["trust"]["coverage"],
                "pred_ssim": obs["trust"]["pred_ssim"], "trusted": obs["trust"]["trusted"],
                "threshold": obs["trust"]["threshold"]}]
        for _ in range(args.steps):
            obs, r, done, info = sim.step(dict(act))
            log.append({"step": info["step"], "frame": info["frame"], "coverage": info["coverage"],
                        "pred_ssim": info["pred_ssim"], "trusted": info["trusted"],
                        "near_m": info["min_actor_distance_m"], "collision": info["collision_tracks"],
                        "truncated": info["truncated"], "reason": info["truncate_reason"]})
            if done:
                break
        rep = sim.coverage_report()
        rollouts[label] = {"action": act, "report": {k: v for k, v in rep.items() if k != "history"},
                           "log": log}
    report["steps"]["rollout"] = {"seconds": round(_t0() - t, 1), "runs": rollouts}
    for label, r in rollouts.items():
        print(f"[5/7] rollout({label}): {r['report']['steps']} 步，可信比例 "
              f"{r['report']['trusted_ratio']}，截断={r['log'][-1].get('reason') or '无'}")

    # ---- 6. 导出 ----
    wj = None
    if not args.skip_export and args.scenario:
        t = _t0()
        import scene_graph
        import corner_case
        import scenario_export
        g = scene_graph.SceneGraph(tm, fps=10.0)
        g.build(frame_idx=0, window=10)
        cands = g.propose_participants(args.scenario, max_candidates=2)
        if cands:
            tm.push_history()
            res = corner_case.generate(tm, args.scenario, cands[0], 0, args.frames,
                                      enable_physics=True, fps=10.0, sampling_seed=args.seed)
            frames = list(range(0, args.frames + 1))
            keep = ([tm.ego_track_id] if tm.ego_track_id else []) + \
                   [int(x) for x in (res.get("affected_tracks") or [])] + \
                   [int(x) for x in (res.get("synthesized_tracks") or [])]
            tracks = scenario_export.collect_tracks(tm, frames, ego_track=tm.ego_track_id,
                                                    only_tracks=keep)
            nm = f"{args.scenario}_s{args.seed}"
            osc = scenario_export.export_openscenario(tracks, str(out_dir / f"{nm}.xosc"),
                                                      dt=0.1, name=nm)
            cr = scenario_export.export_commonroad(tracks, str(out_dir / f"{nm}.cr.xml"),
                                                   dt=0.1, name=nm)
            wj = scenario_export.export_world_json(tracks, str(out_dir / f"{nm}.world.json"), dt=0.1)
            val = scenario_export.validate_roundtrip(osc, cr, tracks, dt=0.1)
            tm.undo()
            report["steps"]["export"] = {"seconds": round(_t0() - t, 1), "scenario": args.scenario,
                                         "num_tracks": len(tracks), "files": {"openscenario": osc,
                                         "commonroad": cr, "world_json": wj}, "validation": val}
            print(f"[6/7] 导出：{len(tracks)} 条轨迹，round-trip "
                  f"{'通过 ✓' if val.get('ok') else '失败 ✗'}（{out_dir}）")
        else:
            report["steps"]["export"] = {"error": f"{args.scenario} 没有候选参与者"}
            print(f"[6/7] 导出：{args.scenario} 没有候选参与者，跳过")
    else:
        print("[6/7] 导出：已跳过")

    # ---- 6.5 （可选）把导出的事故场景送进 CARLA，渲染回放视频 ----
    if args.carla:
        t = _t0()
        bridge = ROOT / "carla_bridge" / "run_bridge.sh"
        if not wj:
            report["steps"]["carla"] = {"skipped": "没有可用的 world.json（先导出场景）"}
            print("[carla] 跳过：没有 world.json")
        elif not bridge.exists():
            report["steps"]["carla"] = {"skipped": f"找不到 {bridge}"}
            print(f"[carla] 跳过：找不到 {bridge}")
        else:
            carla_out = args.carla_out or str(ROOT / "output" / "carla" / tag)
            cmd = ["bash", str(bridge), "--scenario", wj, "--out-dir", carla_out,
                   "--map", args.carla_map, "--focus", args.carla_focus,
                   "--reload-world", "auto"]
            print(f"[carla] 送进 CARLA 渲染：{carla_out}")
            try:
                r = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
                tail = (r.stdout or "").strip().splitlines()[-10:]
                report["steps"]["carla"] = {
                    "seconds": round(_t0() - t, 1), "ok": r.returncode == 0,
                    "out_dir": carla_out, "log_tail": tail,
                    "returncode": r.returncode}
                print(f"[carla] {'完成 ✓' if r.returncode == 0 else '失败 ✗'}（{carla_out}）")
                for line in tail:
                    print("        " + line)
            except Exception as e:                       # noqa: BLE001
                report["steps"]["carla"] = {"failed": f"{type(e).__name__}: {e}"}
                print(f"[carla] 失败：{e}")

    # ---- 7. 扩散精修状态 ----
    st = diffusion_refine.status()
    report["steps"]["diffusion"] = {"ready": st["ready"], "difix_ckpt": st["difix_ckpt"],
                                    "sd_turbo_missing": (st.get("sd_turbo") or {}).get("missing"),
                                    "fix": (st.get("sd_turbo") or {}).get("fix")}
    print(f"[7/7] 扩散精修：{'就绪 ✓' if st['ready'] else '未就绪（缺 ' + str((st.get('sd_turbo') or {}).get('missing')) + '）'}")

    report["ok"] = all("error" not in v for v in report["steps"].values() if isinstance(v, dict))
    rp = out_dir / "closedloop_demo_report.json"
    json.dump(report, open(rp, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\n总报告 -> {rp}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
