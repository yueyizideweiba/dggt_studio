"""闭环仿真运行时（P3）：gym 风格 `reset/step`，带 trust/coverage 越界截断。

为什么需要：到这一步，场景已经是"可渲染 + 可摆位 + 有可信度度量"的了，但还没有**闭环**——
规划器每走一步要能拿到观测、得到反馈、并且知道"这一步还可不可信"。

设计（刻意保持最小可用、无额外依赖）：
- **观测 obs**：`{frame_idx, images: {cam: uint8}, ego_pose, actors: [...], trust: {...}}`
  - `images` 来自 `sensor_rig.render_rig_frame`（单相机或多相机 rig，可加畸变）；
  - `actors` 从 TrackManager 的轨迹读（超出记录帧窗口时按最后速度外推）；
  - `trust` = coverage + 相对最近"训练视角"的偏离（复用 `trust_runtime`）。
- **动作 action**：`{"speed": m/s, "steer": rad, "dt": s(可选)}` → 简易运动学自行车模型积分，
  作用在**自车相机位姿**上（重建场景是静态的，所以"自车移动"= 新视角渲染；
  一旦离开重建覆盖，trust 就会掉下来并被截断）。
  也支持 `{"pose": 4x4}` 直接给定自车相机位姿（接规划器输出更方便）。
- **奖励 reward**：占位实现（碰撞/near-miss 罚分 + 前进里程），文档里写明这是给 RL/评测的接口位。
- **终止 done / truncated**：
  - `truncated=True`：trust 连续 `patience` 步低于阈值（越界），或超出记录帧窗口；
  - 碰撞：与任一 actor 的 OBB 相交（`dggt.scene_edit.collision_physics.check_collision`）。

用法：
    python sim_runtime.py --scene output/trust/scene001_cam0/001 --steps 20 --speed 8 --steer 0
"""
from __future__ import annotations

import argparse
import scene_frames as _sf
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from novelview_trust import coverage_map, pose_deviation          # noqa: E402
from trust_runtime import fit_trust_model, load_trust_rows, predict_trust, pose_novelty  # noqa: E402


def _auto_trust_model() -> Dict[str, Any]:
    try:
        auto = sorted(str(p) for p in (ROOT / "output" / "trust").glob("trust_*.json"))
        if auto:
            return fit_trust_model(load_trust_rows(auto))
    except Exception:  # noqa: BLE001
        pass
    return {"metric": "ssim", "y0": 0.0, "y1": 1.0, "p": 1.0, "tau_deg": 90.0, "n": 0, "r2": None}


class ScenarioSim:
    """一个场景 + 一条（可选）事故剧本的闭环仿真环境。"""

    def __init__(self, scene_dir: str, cameras: Sequence[int] = (0,), width: int = 960,
                 height: int = 640, fps: float = 10.0, min_trust: float = 0.55,
                 patience: int = 2, max_steps: int = 60, num_views: int = 1,
                 trust_model: Optional[Dict[str, Any]] = None, wobble: float = 0.0,
                 min_trust_frac: Optional[float] = None):
        """min_trust_frac 非空时用**相对阈值**：trusted ⟺ pred_ssim ≥ frac × 起始帧的 pred_ssim。

        为什么：不同重建（1 视角 / 3 视角 / 4 视角）的绝对画质差别很大，绝对阈值会把
        质量本来就不高的场景"一开局就判越界"。相对阈值衡量的是"相对起点退化了多少"。
        """
        from dggt_engine import DGGTRenderer
        from track_manager import TrackManager
        self.scene_dir = str(scene_dir)
        self.renderer = DGGTRenderer(scene_path=self.scene_dir, device="cuda", load_sky=True, verbose=False)
        self.tm = TrackManager(self.renderer)
        self.cameras = [int(c) for c in cameras]
        self.width, self.height = int(width), int(height)
        self.fps = float(fps)
        self.dt = 1.0 / self.fps
        self.min_trust = float(min_trust)
        self.patience = int(patience)
        self.max_steps = int(max_steps)
        self.num_views = int(max(1, num_views))
        self.trust_model = trust_model or _auto_trust_model()
        self.n_frames = int(getattr(self.renderer, "num_real_frames", 0) or 0) or self.renderer.frame_count()
        self._train_poses_cache = None
        self._ego_rel = None
        self._state: Dict[str, Any] = {}
        self.wobble = float(wobble)
        self.min_trust_frac = None if min_trust_frac is None else float(min_trust_frac)
        self.baseline_ssim: Optional[float] = None

    # ---------------- 内部工具 ----------------

    def _ego_c2w(self, frame: int) -> Optional[np.ndarray]:
        d = _sf.load_ego_pose(self.scene_dir, int(frame), 0)
        if d is None:
            return None
        m = np.asarray(d["camera_extrinsics_world"], dtype=np.float64)
        if m.shape == (3, 4):
            m = np.vstack([m, np.array([0.0, 0.0, 0.0, 1.0])])
        return m

    def _train_poses(self) -> List[np.ndarray]:
        if self._train_poses_cache is None:
            out = []
            for f in range(self.n_frames):
                p = self._ego_c2w(f)
                if p is not None:
                    out.append(p)
            self._train_poses_cache = out
        return self._train_poses_cache

    def _actors(self, frame: int) -> List[Dict[str, Any]]:
        """该帧的交通参与者（含 OBB 尺寸）。超出记录窗口时按最后速度外推。"""
        out = []
        try:
            objs = self.tm.get_frame_objects(int(frame))
        except Exception:  # noqa: BLE001
            return out
        for o in objs:
            if o.get("ego"):
                continue
            tid = int(o["track_id"])
            pose = np.asarray(o["pose_world"], dtype=np.float64)
            # 速度：最近两帧的世界位移 / dt
            v = 0.0
            p_prev = None
            for back in range(1, 4):
                p_prev = self.tm.get_track_pose(tid, int(frame) - back)
                if p_prev is not None:
                    v = float(np.linalg.norm(np.asarray(p_prev)[:3, 3] - pose[:3, 3]) /
                              (back * self.dt))
                    break
            out.append({"track_id": tid, "pose": pose.tolist(),
                        "dimensions": list(o.get("dimensions") or [4.5, 2.0, 1.6]),
                        "speed_mps": round(v, 2), "synthetic": bool(o.get("synthetic"))})
        return out

    def _collisions(self, ego_pose: np.ndarray, actors: List[Dict[str, Any]]) -> List[int]:
        from dggt.scene_edit.collision_physics import check_collision
        ego_dims = [4.9, 2.0, 1.6]      # 自车（Waymo 车辆尺度近似）
        hit = []
        for a in actors:
            h, _ = check_collision(ego_pose, ego_dims, np.asarray(a["pose"], dtype=np.float64),
                                   a["dimensions"])
            if h:
                hit.append(int(a["track_id"]))
        return hit

    def _observe(self, ego_pose: np.ndarray, frame: int, step: int) -> Dict[str, Any]:
        from sensor_rig import render_rig_frame, load_rig, camera_to_ref, scaled_intrinsics
        # 单相机（或没有标定）时直接用引擎渲染；多相机时用 rig（需要标定文件）
        images: Dict[str, np.ndarray] = {}
        try:
            images = {"cam%d" % c: img for c, img in render_rig_frame(
                self.renderer, int(frame), {"0": {"M": np.eye(4), "K": np.eye(3), "dist": []}},
                0, [0], ego_c2w=ego_pose, size=(self.width, self.height))}
        except Exception:  # noqa: BLE001
            import torch
            d = self._ego_intrinsics(frame)
            img = self.renderer._render_frame_with_object_overrides(
                int(frame), {},
                c2w_override=torch.tensor(ego_pose, dtype=torch.float32, device=self.renderer.device),
                K_override=torch.tensor(d[0], dtype=torch.float32, device=self.renderer.device),
                width_override=self.width, height_override=self.height)
            images = {"cam0": np.asarray(img, dtype=np.uint8)}
        return {"frame_idx": int(frame), "step": int(step), "images": images,
                "ego_pose": np.asarray(ego_pose).tolist(),
                "actors": self._actors(frame), "camera_ids": self.cameras}

    def _ego_intrinsics(self, frame: int):
        d = _sf.load_ego_pose(self.scene_dir, int(frame), 0)
        if d is None:
            raise FileNotFoundError(f"找不到真实帧 {frame} 的自车相机参数")
        K = np.asarray(d["camera_intrinsics"], dtype=np.float64).copy()
        w0, h0 = float(d["camera"]["width"]), float(d["camera"]["height"])
        K[0, 0] *= self.width / w0
        K[0, 2] *= self.width / w0
        K[1, 1] *= self.height / h0
        K[1, 2] *= self.height / h0
        return K, self.width, self.height

    def _trust(self, frame: int, ego_pose: np.ndarray) -> Dict[str, Any]:
        K, W, H = self._ego_intrinsics(frame)
        cov = coverage_map(self.scene_dir, int(frame), ego_pose, K, W, H)
        nov = pose_novelty(ego_pose, self._train_poses())
        ssim = predict_trust(self.trust_model, cov["coverage"], nov["d_view_angle_deg"])
        thr = self.min_trust
        if self.min_trust_frac is not None and self.baseline_ssim:
            thr = float(self.min_trust_frac) * float(self.baseline_ssim)
        return {"coverage": cov["coverage"], "pred_ssim": round(float(ssim), 4),
                "threshold": round(float(thr), 4), **nov, "trusted": bool(ssim >= thr)}

    # ---------------- gym 接口 ----------------

    def reset(self) -> Dict[str, Any]:
        self._state = {"step": 0, "frame": 0.0, "bad_streak": 0,
                       "ego_pose": self._ego_c2w(0).copy(), "collided": False,
                       "trusted_frames": 0, "history": []}
        ego = self._state["ego_pose"]
        try:
            obs = self._observe(ego, 0, 0)
        except Exception:  # noqa: BLE001
            obs = {"images": {}}
        tr = self._trust(0, ego)
        self.baseline_ssim = float(tr["pred_ssim"])
        tr = self._trust(0, ego)          # 用基线重算一次 threshold
        obs["trust"] = tr
        self._state["history"].append({"step": 0, "frame": 0, "trust": tr})
        return obs

    def step(self, action: Dict[str, Any]) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        if not self._state:
            self.reset()
        st = self._state
        st["step"] += 1
        dt = float(action.get("dt") or self.dt)
        ego = np.asarray(st["ego_pose"], dtype=np.float64).copy()

        if action.get("pose") is not None:                      # 直接给位姿（接规划器）
            ego = np.asarray(action["pose"], dtype=np.float64)
        else:                                                   # 简易运动学自行车模型
            speed = float(action.get("speed", 0.0))
            steer = float(action.get("steer", 0.0))
            wheelbase = 2.8
            yaw = math.atan2(float(ego[0, 2]), float(ego[2, 2]))
            yaw += speed / wheelbase * math.tan(steer) * dt
            fwd = np.array([math.sin(yaw), 0.0, math.cos(yaw)])
            ego[:3, 3] = ego[:3, 3] + fwd * speed * dt
            cy, sy = math.cos(yaw), math.sin(yaw)
            ego[:3, 0] = np.array([cy, 0.0, -sy])
            ego[:3, 2] = np.array([sy, 0.0, cy])
            if self.wobble:
                ego[:3, 3] += ego[:3, 0] * self.wobble * math.sin(0.7 * st["step"])

        frame_f = st["frame"] + 1.0                 # 帧随步进推进（1 步 = 1 帧）
        over_window = frame_f > (self.n_frames - 1)
        frame = int(min(self.n_frames - 1, max(0, round(frame_f))))
        st["frame"] = frame_f
        st["ego_pose"] = ego

        obs = self._observe(ego, frame, st["step"])
        tr = self._trust(frame, ego)
        obs["trust"] = tr

        actors = obs.get("actors") or []
        hits = self._collisions(ego, actors)
        if hits:
            st["collided"] = True
        truncated = False
        reason = None
        if over_window:
            truncated, reason = True, f"超出记录帧窗口（{self.n_frames} 帧，不做时间外推）"
        if not tr["trusted"] and not truncated:
            st["bad_streak"] += 1
            if st["bad_streak"] >= self.patience:
                truncated, reason = True, (f"预测画质 {tr['pred_ssim']} < 阈值 {tr['threshold']}"
                                          f" 连续 {self.patience} 步（越界）")
        else:
            st["bad_streak"] = 0

        near = 0.0
        for a in actors:
            d = float(np.linalg.norm(np.asarray(a["pose"], dtype=np.float64)[:3, 3] - ego[:3, 3]))
            near = d if near == 0 else min(near, d)
        done = bool(st["collided"] or truncated or st["step"] >= self.max_steps)
        # 奖励占位：前进里程 - 碰撞罚 - 贴太近罚（给 RL/评测留接口）
        reward = float(action.get("speed", 0.0) * dt)
        if st["collided"]:
            reward -= 10.0
        if near and near < 3.0:
            reward -= (3.0 - near)
        info = {"coverage": tr["coverage"], "pred_ssim": tr["pred_ssim"], "trusted": tr["trusted"],
                "truncated": truncated, "truncate_reason": reason, "collision_tracks": hits,
                "min_actor_distance_m": round(near, 3), "frame": frame, "step": st["step"]}
        st["history"].append({"step": st["step"], "frame": frame, "trust": tr,
                              "collision": hits, "near_m": round(near, 3)})
        return obs, reward, done, info

    def coverage_report(self) -> Dict[str, Any]:
        """episode 级别的可信度/覆盖率报告（越界截断的依据）。"""
        st = self._state or {}
        hist = st.get("history", [])
        trusted = [h for h in hist if h.get("trust", {}).get("trusted")]
        return {"steps": len(hist), "trusted_steps": len(trusted),
                "trusted_ratio": round(len(trusted) / max(1, len(hist)), 4),
                "min_coverage": round(min([h["trust"]["coverage"] for h in hist] or [0.0]), 4),
                "mean_coverage": round(float(np.mean([h["trust"]["coverage"] for h in hist] or [0.0])), 4),
                "min_pred_ssim": round(min([h["trust"]["pred_ssim"] for h in hist] or [0.0]), 4),
                "collided": bool(st.get("collided")), "history": hist}


def speed_or_zero(action: Dict[str, Any]) -> float:
    return float(action.get("speed", 0.0))


def main():
    ap = argparse.ArgumentParser(description="闭环仿真运行时（reset/step + trust 截断）")
    ap.add_argument("--scene", required=True)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--speed", type=float, default=8.0)
    ap.add_argument("--steer", type=float, default=0.0)
    ap.add_argument("--min_trust", type=float, default=0.55)
    ap.add_argument("--min_trust_frac", type=float, default=None,
                    help="相对阈值（× 起始帧 pred_ssim），比绝对阈值更适合跨重建比较")
    ap.add_argument("--out_dir", default="", help="报告输出目录（默认仓库 output/sim）")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    sim = ScenarioSim(args.scene, min_trust=args.min_trust, max_steps=args.steps,
                      min_trust_frac=args.min_trust_frac)
    obs = sim.reset()
    print(f"[sim] reset: frame 0, trust={obs['trust']}")
    for i in range(args.steps):
        obs, r, done, info = sim.step({"speed": args.speed, "steer": args.steer})
        print(f"[sim] step {info['step']:>3d} frame {info['frame']:>3d} "
              f"cov={info['coverage']:.3f} pred_ssim={info['pred_ssim']:.3f} "
              f"trusted={info['trusted']} near={info['min_actor_distance_m']:.1f}m "
              f"collision={info['collision_tracks']} r={r:.2f}"
              + (f"  TRUNCATED: {info['truncate_reason']}" if info["truncated"] else ""))
        if done:
            break
    rep = sim.coverage_report()
    print(json.dumps({k: v for k, v in rep.items() if k != "history"}, ensure_ascii=False, indent=2))
    out = args.out
    if not out and args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)
        out = os.path.join(args.out_dir, f"sim_s{args.speed:g}_st{args.steer:g}.json")
    if out:
        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
        json.dump(rep, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        print("报告 ->", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
