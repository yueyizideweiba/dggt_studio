"""确定性检查：同一个 seed / 同一条场景，重跑是否得到**完全一致**的结果。

为什么重要：闭环仿真/大规模训练的数据必须可复现。踩过的坑：
- 行人模型以前是 `random.choice` → 同 seed 重跑外观不同（已改成位姿稳定哈希）；
- 批量里的 `seed_i` 由请求 seed 派生（`_random.Random(seed)`）→ 可复现；
- 采样参数全部由 `sampling_seed` 派生（`random.Random(seed)`）→ 可复现。

用法：
    python determinism_check.py --scene <重建场景目录> --scenario rear-end --seed 7 --runs 2
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dggt_engine import DGGTRenderer  # noqa: E402
from track_manager import TrackManager  # noqa: E402
import corner_case  # noqa: E402
import scene_graph  # noqa: E402


def generate_once(renderer, tm, scenario: str, seed: int, start: int, frames: int,
                  fps: float) -> Dict[str, Any]:
    """跑一次生成，返回可比较的"结果指纹"（参与者、采样参数、逐帧位姿、合成物体外观）。"""
    graph = scene_graph.SceneGraph(tm, fps=fps)
    graph.build(frame_idx=start, window=10)
    cands = graph.propose_participants(scenario, max_candidates=3)
    if not cands:
        raise RuntimeError(f"{scenario} 没有候选参与者")
    tm.push_history()
    res = corner_case.generate(tm, scenario, cands[0], start, frames, intensity=1.0,
                              enable_physics=True, fps=fps, sampling_seed=seed)
    out = {"roles": {k: int(v) for k, v in (res.get("roles") or cands[0]).items()},
           "sampling": {k: (round(float(v), 6) if isinstance(v, (int, float)) else v)
                        for k, v in (res.get("sampling_params") or {}).items()},
           "collision_frame": res.get("collision_frame"),
           "affected": [int(t) for t in res.get("affected_tracks", [])],
           "synthesized": [int(t) for t in res.get("synthesized_tracks", [])],
           "trajectories": {}, "synthetic_assets": []}
    for tid in out["affected"]:
        traj = {}
        for f in range(start, start + frames + 1):
            p = tm.get_track_pose(tid, f)
            if p is not None:
                traj[str(f)] = [round(float(v), 6) for v in np.asarray(p).reshape(-1)]
        out["trajectories"][str(tid)] = traj
    for sid in out["synthesized"]:
        s = tm.synthetic_tracks.get(int(sid)) or {}
        out["synthetic_assets"].append({"track_id": int(sid),
                                        "ply": os.path.basename(str(s.get("ply_path") or "")),
                                        "dims": [round(float(v), 4) for v in (s.get("dimensions") or [])]})
    tm.undo()
    return out


def diff_fingerprints(a: Dict[str, Any], b: Dict[str, Any]) -> List[str]:
    """返回不一致之处（空列表 = 完全一致）。"""
    bad = []
    for k in ("roles", "sampling", "collision_frame", "affected", "synthesized", "synthetic_assets"):
        if json.dumps(a.get(k), sort_keys=True, ensure_ascii=False) != \
           json.dumps(b.get(k), sort_keys=True, ensure_ascii=False):
            bad.append(f"{k} 不一致: {a.get(k)} vs {b.get(k)}")
    ta, tb = a.get("trajectories", {}), b.get("trajectories", {})
    if set(ta) != set(tb):
        bad.append(f"受影响 track 集合不一致: {sorted(ta)} vs {sorted(tb)}")
    for tid in set(ta) & set(tb):
        fa, fb = ta[tid], tb[tid]
        if set(fa) != set(fb):
            bad.append(f"track {tid} 帧集合不一致")
            continue
        m = 0.0
        for f in fa:
            m = max(m, float(np.max(np.abs(np.asarray(fa[f]) - np.asarray(fb[f])))))
        if m > 1e-5:
            bad.append(f"track {tid} 逐帧位姿最大差异 {m:.3e}（应≈0）")
    return bad


def main():
    ap = argparse.ArgumentParser(description="确定性检查：同 seed 重跑是否一致")
    ap.add_argument("--scene", required=True)
    ap.add_argument("--scenario", default="rear-end")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--frames", type=int, default=12)
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--runs", type=int, default=2)
    args = ap.parse_args()

    renderer = DGGTRenderer(scene_path=args.scene, device="cuda", load_sky=True, verbose=False)
    tm = TrackManager(renderer)
    tm.fps = float(args.fps)
    prints = [generate_once(renderer, tm, args.scenario, args.seed, args.start, args.frames, args.fps)
              for _ in range(max(2, args.runs))]
    bad = diff_fingerprints(prints[0], prints[1])
    rep = {"scene": args.scene, "scenario": args.scenario, "seed": args.seed,
           "runs": len(prints), "deterministic": not bad, "diffs": bad,
           "roles": prints[0]["roles"], "sampling": prints[0]["sampling"],
           "synthesized_assets": prints[0]["synthetic_assets"]}
    print(json.dumps(rep, ensure_ascii=False, indent=2))
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
