"""自然冲突生成：**保留原轨迹语义，只调时序/速度**制造或避免冲突。

与"套模板"（`scenario_engine.prim_regularize` 把轨迹擦成匀速直线再重摆）的本质区别：

* 这里**不改路径**：a 在路口左转就还是左转、对向 b 直行就还是直行；
* 只对车辆做**纵向重定时**（沿它自己的轨迹加速/减速），让两车在它们**本来就交汇**的
  冲突点同时到达（碰撞）、差一点到达（险情/近失）、或一车刹停（避免）。

这正是"研究如何避免事故"关心的东西：关键帧/最晚反应点由 `compute_critical_frame`
给出（碰撞帧 + critical_frame + TTC + 严重度），而不是关注撞完之后怎么飞。

用法（由 scenario_engine.plan_natural_conflict 调用）：
    generate_natural_conflict(tm, ego, other, frames, fps,
                              outcome="collide|near_miss|avoid",
                              time_gap_s=0.5, adjust="auto", ego_brake_decel=6.0)
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from dggt.scene_edit.collision_physics import check_collision, compute_critical_frame

from corner_case import (
    _get_dimensions,
    _pose_with_center,
    _write_pose,
    _heading,
    _speed_mps,
    _velocity_vec,
)

# 两条路径"最近距离"超过这个值就认为它们本不相交，无法用纯时序制造冲突（米）
CONFLICT_MAX_DIST = 4.5


def _dedup_polyline(pts: np.ndarray, eps: float = 0.05) -> np.ndarray:
    if len(pts) < 2:
        return pts
    keep = [0]
    for i in range(1, len(pts)):
        if float(np.linalg.norm(pts[i] - pts[keep[-1]])) > eps:
            keep.append(i)
    if keep[-1] != len(pts) - 1:
        keep.append(len(pts) - 1)
    return np.asarray(pts, dtype=np.float64)[keep]


def _cum_arc(pts: np.ndarray) -> np.ndarray:
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(seg)])


def _sample(pts: np.ndarray, arc: np.ndarray, s: float) -> Tuple[np.ndarray, np.ndarray]:
    """按弧长 s 在折线上取 (点 xyz, 单位切向 xyz)。超界时线性外推。"""
    if len(pts) == 1:
        return pts[0].copy(), np.array([0.0, 0.0, 1.0])
    total = float(arc[-1])
    if s <= 0.0:
        seg = float(arc[1] - arc[0]) if len(pts) > 2 else 1e-6
        p = pts[0] + (pts[1] - pts[0]) * (s / max(1e-9, seg))
        v = pts[1] - pts[0]
    elif s >= total:
        seg = float(arc[-1] - arc[-2]) if len(pts) > 2 else 1e-6
        p = pts[-1] + (pts[-1] - pts[-2]) * ((s - total) / max(1e-9, seg))
        v = pts[-1] - pts[-2]
    else:
        i = int(np.searchsorted(arc, s) - 1)
        i = max(0, min(i, len(pts) - 2))
        u = (s - arc[i]) / max(1e-9, arc[i + 1] - arc[i])
        p = pts[i] * (1.0 - u) + pts[i + 1] * u
        v = pts[i + 1] - pts[i]
    v = np.array([v[0], 0.0, v[2]], dtype=np.float64)
    n = float(np.linalg.norm(v))
    if n < 1e-9:
        v = np.array([0.0, 0.0, 1.0])
    else:
        v = v / n
    return p, v


def _smooth_pts(pts: np.ndarray, passes: int = 2) -> np.ndarray:
    """对逐帧中心点做 [1,2,1]/4 的轻度平滑（端点固定），去掉检测噪声。

    不做平滑的话，`retime_track` 沿带噪折线重采样会在每个顶点处方向突变——
    实测 accel 22 m/s²、jerk 62 m/s³（质量门直接判废）。
    """
    p = np.asarray(pts, dtype=np.float64).copy()
    n = len(p)
    if n < 3:
        return p
    for _ in range(max(0, int(passes))):
        q = p.copy()
        q[1:-1] = (p[:-2] + 2.0 * p[1:-1] + p[2:]) / 4.0
        p = q
    return p


def extract_path(tm, track_id, frames, fps: float) -> Dict[str, Any]:
    """把某 track 在窗口内的位姿抽成一条"路径"（折线 + 弧长参数化）。

    返回 {'pts'(N,3), 'arc'(N), 'frame_arc'(len(frames)), 'f0', 'fps',
          'speed_mps'(自然平均速度), 'n_frames'}
    """
    f0 = frames[0]
    centers = []
    for f in frames:
        p = tm.get_track_pose(int(track_id), int(f))
        if p is None:
            centers.append(None)
        else:
            centers.append(np.asarray(p, dtype=np.float32)[:3, 3])
    valid = [(i, c) for i, c in enumerate(centers) if c is not None]
    if len(valid) < 2:
        return {"pts": np.zeros((0, 3)), "arc": np.zeros(0), "frame_arc": np.zeros(0),
                "f0": int(f0), "fps": float(fps), "speed_mps": 0.0, "n_frames": 0, "valid": False}
    # 有效帧可能不连续（检测漏帧）：对缺失帧线性插值，得到逐帧连续路径
    frame_pts = []
    vi = 0
    for i in range(len(frames)):
        if centers[i] is not None:
            frame_pts.append(np.asarray(centers[i], dtype=np.float64))
        else:
            # 在前后最近的有效帧之间插值
            lo = None
            for j in range(i, -1, -1):
                if centers[j] is not None:
                    lo = (j, np.asarray(centers[j], dtype=np.float64)); break
            hi = None
            for j in range(i, len(frames)):
                if centers[j] is not None:
                    hi = (j, np.asarray(centers[j], dtype=np.float64)); break
            if lo is not None and hi is not None:
                a = (i - lo[0]) / float(max(1, hi[0] - lo[0]))
                frame_pts.append(lo[1] * (1 - a) + hi[1] * a)
            elif lo is not None:
                frame_pts.append(lo[1].copy())
            else:
                frame_pts.append(hi[1].copy())
    pts = _dedup_polyline(np.asarray(frame_pts, dtype=np.float64))
    # 平滑后再参数化：直接把"逐帧累积弧长"作为 frame_arc，而不是"最近顶点弧长"。
    # 后者是量化台阶（相邻帧可能落在同一顶点、再整段跳一次），重定时后就是一顿一顿的。
    sm = _smooth_pts(np.asarray(frame_pts, dtype=np.float64))
    fa = _cum_arc(sm)
    arc = _cum_arc(pts)
    if len(fa) >= 2 and float(fa[-1]) > 1e-6 and float(arc[-1]) > 1e-6:
        fa = fa * (float(arc[-1]) / float(fa[-1]))     # 与折线弧长对齐
    speed = 0.0
    if len(fa) >= 2:
        speed = float((fa[-1] - fa[0]) / max(1e-6, (len(fa) - 1) / float(fps)))
    return {"pts": pts, "arc": arc, "frame_arc": fa, "f0": int(f0), "fps": float(fps),
            "speed_mps": speed, "n_frames": len(frames), "valid": True}


def _closest_approach(path_a: Dict[str, Any], path_b: Dict[str, Any]):
    """两折线的最近点对（XZ 平面），返回 (s_a, s_b, dist, head_a, head_b)。"""
    pa, pb = path_a["pts"], path_b["pts"]
    if len(pa) < 2 or len(pb) < 2:
        return None
    best = None
    for i in range(len(pa)):
        d = np.linalg.norm(pb[:, [0, 2]] - pa[i][[0, 2]], axis=1)
        j = int(np.argmin(d))
        if best is None or d[j] < best[2]:
            best = (float(path_a["arc"][i]), float(path_b["arc"][j]), float(d[j]))
    if best is None:
        return None
    s_a, s_b, dist = best
    _, ha = _sample(pa, path_a["arc"], s_a)
    _, hb = _sample(pb, path_b["arc"], s_b)
    return {"s_a": s_a, "s_b": s_b, "dist": dist, "head_a": ha, "head_b": hb}


def _conflict_type(head_a: np.ndarray, head_b: np.ndarray) -> Tuple[str, float]:
    """由两车在冲突点处的航向判场景类型。"""
    ha = np.array([head_a[0], head_a[2]])
    hb = np.array([head_b[0], head_b[2]])
    na, nb = float(np.linalg.norm(ha)), float(np.linalg.norm(hb))
    if na < 1e-6 or nb < 1e-6:
        return "unknown", 0.0
    c = float(np.clip(float(np.dot(ha / na, hb / nb)), -1.0, 1.0))
    ang = math.degrees(math.acos(c))     # 0=同向 180=对向
    if ang >= 135.0:
        return "head-on", ang
    if 45.0 <= ang <= 135.0:
        return "crossing", ang
    return "same-dir", ang


def _arrival_time(path: Dict[str, Any], s_target: float) -> float:
    """自然时序下，从窗口起点到达弧长 s_target 的秒数。"""
    fa = path["frame_arc"]
    fps = float(path["fps"])
    if len(fa) == 0:
        return 0.0
    if s_target <= fa[0]:
        return 0.0
    if s_target >= fa[-1]:
        return (len(fa) - 1) / fps
    for i in range(len(fa) - 1):
        if fa[i] <= s_target <= fa[i + 1]:
            u = (s_target - fa[i]) / max(1e-6, fa[i + 1] - fa[i])
            return (i + u) / fps
    return (len(fa) - 1) / fps


def _pose_at(xyz: np.ndarray, yaw: float) -> np.ndarray:
    cy, sy = math.cos(yaw), math.sin(yaw)
    P = np.eye(4, dtype=np.float32)
    P[0, 0], P[0, 2] = cy, sy
    P[1, 1] = 1.0
    P[2, 0], P[2, 2] = -sy, cy
    P[:3, 3] = np.asarray(xyz, dtype=np.float32)
    return P


def retime_track(tm, track_id, frames, path: Dict[str, Any], speed_factor: float,
                 height_mode: str = "path"):
    """沿自己的轨迹按 speed_factor 重定时（>1 加速、<1 减速），路径形状不变。"""
    if not path.get("valid"):
        return False
    pts, arc, fa = path["pts"], path["arc"], path["frame_arc"]
    s0 = float(fa[0])
    for i, f in enumerate(frames):
        s = s0 + (float(fa[i]) - s0) * float(speed_factor)
        xyz, tang = _sample(pts, arc, s)
        yaw = math.atan2(float(tang[0]), float(tang[2]))
        _write_pose(tm, int(track_id), int(f), _pose_at(xyz, yaw))
    return True


def _brake_along_path(tm, track_id, frames, path: Dict[str, Any], stop_arc: float,
                      decel_mps2: float, fps: float) -> bool:
    """沿自身路径按减速度刹停（停在弧长 stop_arc 之前），**保持路径形状**。

    比 `corner_case._apply_brake_decel`（沿起始航向拉直线）更贴合真实轨迹。
    """
    if not path.get("valid"):
        return False
    pts, arc, fa = path["pts"], path["arc"], path["frame_arc"]
    dt = 1.0 / float(fps)
    s = float(fa[0])
    v = float(path.get("speed_mps") or 0.0)
    writes = []
    for i, f in enumerate(frames):
        if i > 0:
            v = max(0.0, v - float(decel_mps2) * dt)
            s = min(s + v * dt, float(stop_arc))
        xyz, tang = _sample(pts, arc, s)
        yaw = math.atan2(float(tang[0]), float(tang[2]))
        writes.append((int(f), _pose_at(xyz, yaw)))
    for f, wp in writes:
        _write_pose(tm, int(track_id), f, wp)
    return True


def _collision_analysis(tm, a, b, frames, fps):
    """基于当前已写入的位姿做碰撞 + 关键帧/最晚反应点分析。"""
    dims_a = _get_dimensions(tm, a)
    dims_b = _get_dimensions(tm, b)
    pa, pb = [], []
    for f in frames:
        xa, xb = tm.get_track_pose(a, f), tm.get_track_pose(b, f)
        if xa is None or xb is None:
            continue
        pa.append(np.asarray(xa, dtype=np.float32))
        pb.append(np.asarray(xb, dtype=np.float32))
    if len(pa) < 2:
        return None
    # 逐帧查实际碰撞 + 最近接近
    collided_frame = None
    min_dist, closest_frame = float('inf'), None
    for f, xa, xb in zip(frames, pa, pb):
        ca, cb = xa[:3, 3], xb[:3, 3]
        d = float(np.linalg.norm(ca - cb))
        if d < min_dist:
            min_dist, closest_frame = d, int(f)
        hit, _ = check_collision(xa, dims_a, xb, dims_b)
        if hit and collided_frame is None:
            collided_frame = int(f)
    info = compute_critical_frame(pa, dims_a, pb, dims_b, list(frames), fps=fps)
    if info is None:
        # 没撞、也没达到 compute_critical_frame 的"near"阈值（gap 较大）：
        # 自算"最近接近帧 + TTC"，保证 near_miss / avoid 也有可读的回避指标。
        closing = 0.0
        if closest_frame is not None:
            i = frames.index(closest_frame)
            va = _velocity_vec(tm, a, closest_frame, fps)
            vb = _velocity_vec(tm, b, closest_frame, fps)
            closing = float(np.linalg.norm(va - vb))
        ttc = (min_dist / closing) if closing > 1e-3 else None
        sev = "near_miss" if min_dist < 3.0 else ("close" if min_dist < 6.0 else "safe")
        info = {"critical_frame": closest_frame, "collision_frame": None,
                "time_to_collision": ttc, "distance_at_critical": min_dist,
                "collision_severity": sev, "reaction_frames": None}
    info["attacker"] = int(a)
    info["victim"] = int(b)
    info["collided"] = bool(collided_frame is not None)
    if collided_frame is not None and info.get("collision_frame") is None:
        info["collision_frame"] = int(collided_frame)
    if info.get("critical_frame") is None:
        info["critical_frame"] = closest_frame
    return info


def generate_natural_conflict(tm, ego, other, frames, fps, outcome="collide",
                              time_gap_s=0.4, adjust="auto", ego_brake_decel=6.0):
    """核心：保留两车原轨迹，用纵向重定时制造/避免冲突。"""
    from corner_case import _apply_brake_decel
    frames = list(frames)
    path_a = extract_path(tm, ego, frames, fps)
    path_b = extract_path(tm, other, frames, fps)
    if not path_a.get("valid") or not path_b.get("valid"):
        return {"ok": False, "reason": "参与者轨迹过短/缺失"}

    approach = _closest_approach(path_a, path_b)
    if approach is None or approach["dist"] > CONFLICT_MAX_DIST:
        return {"ok": False, "reason": "两车轨迹不相交（最近 %.1fm），无法用纯时序制造冲突；"
                                       "请换一对会交汇的车，或用模板化场景" % (approach or {"dist": float('inf')})["dist"]}

    ctype, ang = _conflict_type(approach["head_a"], approach["head_b"])
    t_a = _arrival_time(path_a, approach["s_a"])
    t_b = _arrival_time(path_b, approach["s_b"])
    delta = t_a - t_b           # >0: ego 后到；<0: ego 先到

    # 决定谁动、动多少
    if outcome == "avoid":
        # 不重定时，让 ego 制动到冲突点之前停下（"可避免"场景，critical frame 有意义）
        stop_before = max(0.5, approach["s_a"] - 2.0)
        # 先把两车都写回各自**去噪后的原路径**（形状不变），再沿路径减速
        retime_track(tm, ego, frames, path_a, 1.0)
        retime_track(tm, other, frames, path_b, 1.0)
        _brake_along_path(tm, ego, frames, path_a, stop_before, float(ego_brake_decel), fps)
        result = {
            "ok": True, "outcome": "avoid", "conflict_type": ctype,
            "approach_dist_m": round(approach["dist"], 2), "conflict_angle_deg": round(ang, 1),
            "natural_delta_s": round(delta, 2), "braked": int(ego),
        }
    else:
        # 目标时间差：collide=0，near_miss=time_gap_s（一车先过）
        target_gap = 0.0 if outcome == "collide" else float(time_gap_s)
        # 让 ego 以 target_gap 领先/落后于 other：t_a' = t_b + target_gap（ego 晚 target_gap 到）
        # 或让 other 动。选改动幅度小的一方。
        k_ego = t_a / max(0.15, t_b + target_gap)          # ego 到达时刻调成 t_b+gap
        k_other = t_b / max(0.15, t_a - target_gap)        # other 到达时刻调成 t_a-gap
        dev_ego = abs(k_ego - 1.0)
        dev_other = abs(k_other - 1.0)
        if adjust == "ego":
            do_ego = True
        elif adjust == "other":
            do_ego = False
        else:
            do_ego = dev_ego <= dev_other
        # 限幅：别把速度改得太离谱（0.2~3.0 倍）。夹得太紧会导致"到不了冲突点"，
        # 明明该撞却没撞（实测 0.35 下限时最近距离 3.1m、TTC 0.27s 的假近失）。
        if do_ego:
            k = float(np.clip(k_ego, 0.2, 3.0))
            retime_track(tm, ego, frames, path_a, k)
            retime_track(tm, other, frames, path_b, 1.0)   # 去噪但不改时序
        else:
            k = float(np.clip(k_other, 0.2, 3.0))
            retime_track(tm, other, frames, path_b, k)
            retime_track(tm, ego, frames, path_a, 1.0)
        result = {
            "ok": True, "outcome": outcome, "conflict_type": ctype,
            "approach_dist_m": round(approach["dist"], 2), "conflict_angle_deg": round(ang, 1),
            "natural_delta_s": round(delta, 2), "target_gap_s": round(target_gap, 2),
            "adjusted_track": int(ego if do_ego else other),
            "speed_factor": round(k, 3),
            "natural_speed_a": round(path_a["speed_mps"], 2),
            "natural_speed_b": round(path_b["speed_mps"], 2),
        }

    info = _collision_analysis(tm, ego, other, frames, fps)
    if info is not None:
        result["collision_analysis"] = info
        result["critical_frame"] = info.get("critical_frame")
        result["collision_frame"] = info.get("collision_frame")
    return result
