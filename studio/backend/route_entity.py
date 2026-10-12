"""在高精地图道路上放置"严格沿路线行驶"的车辆实体。

设计：
- **路线** = 车道中心线折线（可选沿后继车道延伸一段，直行优先）；
- **行驶方向** = 沿折线正向 / 反向；
- **位姿** = 按弧长匀速采样折线：位置 = 折线采样点、朝向 = 折线切线、y = 路面高度（贴地），
  所以车辆**严格贴合道路**，不会飘、不会斜切；
- **外观模型** = 默认（主车 ego.ply，就是主车那个备用模型）或自定义 PLY（如自然语言/SAM3D 生成）。

与"文本添加实体"的区别：那条链路是"相对自车/参照车"摆放；这里是"**给定车道 + 方向 → 沿路线走**"。
"""
from __future__ import annotations

import math
import os
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from road_rules import sample_polyline, polyline_total, _dedup_polyline, _dir_at_start, _dir_at_end


def _yaw_pose(x: float, y: float, z: float, yaw: float) -> np.ndarray:
    """世界系位姿：列 2 = (sin,0,cos) 为车头前向（与全局约定一致）。"""
    c, s = math.cos(yaw), math.sin(yaw)
    P = np.eye(4, dtype=np.float32)
    P[:3, 0] = (c, 0.0, -s)
    P[:3, 1] = (0.0, 1.0, 0.0)
    P[:3, 2] = (s, 0.0, c)
    P[:3, 3] = (float(x), float(y), float(z))
    return P


def list_lanes(road_model, min_len: float = 6.0, limit: int = 600,
               include_pts: bool = True) -> List[Dict[str, Any]]:
    """可选车道列表（id / 长度 / 限速 / 折线），供前端选路线。"""
    if road_model is None or not getattr(road_model, "available", False):
        return []
    out: List[Dict[str, Any]] = []
    for lid in road_model.lane_ids():
        try:
            L = float(road_model.lane_length(lid))
        except Exception:  # noqa: BLE001
            continue
        if L < float(min_len):
            continue
        P = road_model.lane_pts(lid)
        if P is None or len(P) < 2:
            continue
        item: Dict[str, Any] = {
            "lane_id": str(lid), "length_m": round(L, 1),
            "speed_limit_mps": round(float(road_model.speed_limit_mps(lid)), 1),
            "start": [round(float(x), 2) for x in np.asarray(P)[0][:3]],
            "end": [round(float(x), 2) for x in np.asarray(P)[-1][:3]],
        }
        if include_pts:
            item["pts"] = [[round(float(v), 2) for v in p[:3]] for p in np.asarray(P)]
        out.append(item)
        if len(out) >= int(limit):
            break
    return out


def build_route(road_model, lane_id, extend_m: float = 0.0,
                max_lanes: int = 8) -> Optional[np.ndarray]:
    """取一条车道的折线；extend_m>0 时沿后继车道继续延伸（多后继取"最直"的）。"""
    if road_model is None:
        return None
    P = road_model.lane_pts(lane_id)
    if P is None or len(P) < 2:
        return None
    P = np.asarray(P, dtype=np.float64)
    if extend_m and float(extend_m) > 0:
        cur = str(lane_id)
        seen = {cur}
        while polyline_total(P) < float(extend_m) and len(seen) < int(max_lanes):
            try:
                succ = road_model.pure_successors(cur)
            except Exception:  # noqa: BLE001
                succ = []
            if not succ:
                break
            if len(succ) == 1:
                nxt = succ[0]
            else:
                end_dir = _dir_at_end(P)

                def _ang(s):
                    Q = road_model.lane_pts(s)
                    if Q is None or len(Q) < 2:
                        return 1e9
                    d = _dir_at_start(np.asarray(Q, dtype=np.float64))
                    return float(np.arccos(np.clip(float(np.dot(d, end_dir)), -1.0, 1.0)))

                nxt = min(succ, key=_ang)
            if nxt in seen:
                break
            Q = road_model.lane_pts(nxt)
            if Q is None or len(Q) < 2:
                break
            seen.add(nxt)
            P = np.vstack([P, np.asarray(Q, dtype=np.float64)[1:]])
            cur = nxt
    return _dedup_polyline(P)


def route_poses(tm, route_pts, frames: List[int], fps: float, speed_mps: float,
                backward: bool = False, start_arc: float = 0.0,
                ground: Optional[Callable[[float, float], float]] = None) -> Dict[int, np.ndarray]:
    """沿路线逐帧生成"严格按路线"的位姿（位置=折线采样、朝向=切线、y=路面高）。"""
    P = np.asarray(route_pts, dtype=np.float64)
    if P is None or len(P) < 2:
        return {}
    if backward:
        P = P[::-1].copy()
    total = polyline_total(P)
    dt = 1.0 / max(1e-6, float(fps))
    poses: Dict[int, np.ndarray] = {}
    for i, f in enumerate(frames):
        s = float(start_arc) + float(speed_mps) * dt * i
        if s < 0.0:
            continue
        if s > total:
            break
        c, tang = sample_polyline(P, s)
        if c is None:
            continue
        yaw = math.atan2(float(tang[0]), float(tang[2]))
        x, z = float(c[0]), float(c[2])
        y = float(c[1])
        if ground is not None:
            try:
                y = float(ground(x, z))
            except Exception:  # noqa: BLE001
                pass
        poses[int(f)] = _yaw_pose(x, y, z, yaw)
    return poses


def create_route_vehicle(tm, road_model, lane_id, direction: str = "forward",
                         speed_mps: float = 8.0, start_frame: int = 0, num_frames: int = 30,
                         fps: float = 10.0, extend_m: float = 0.0, start_arc: float = 0.0,
                         model: str = "default", ply_path: Optional[str] = None,
                         dims: Optional[List[float]] = None, scale: Optional[float] = None,
                         model_corr=None, type_name: Optional[str] = None,
                         source: Optional[str] = None) -> Dict[str, Any]:
    """创建一辆沿指定路线行驶的车辆实体，返回 {ok, track_id, ...}。

    model='default' → 用主车 ego.ply（含主车的 scale/dims，看上去就是"那辆备用车"）；
    model='custom'  → 用传入的 ply_path/dims/scale。
    """
    from track_manager import EGO_PLY_DEFAULT, SAM3D_MODEL_CORR

    route = build_route(road_model, lane_id, extend_m=extend_m)
    if route is None or len(route) < 2:
        return {"ok": False, "reason": "该车道没有可用的中心线折线（或地图不可用）"}

    frames = list(range(int(start_frame), int(start_frame) + int(num_frames) + 1))
    backward = str(direction).strip().lower() in ("backward", "back", "reverse", "neg", "-1")
    ground = getattr(tm, "ground_y_at", None)
    poses = route_poses(tm, route, frames, fps, float(speed_mps), backward=backward,
                        start_arc=float(start_arc), ground=ground if callable(ground) else None)
    if len(poses) < 2:
        return {"ok": False, "reason": "路线太短或速度过慢，生成的帧数不足 2 帧"}

    if str(model).lower() == "custom" and ply_path:
        ply = str(ply_path)
        sc = float(scale if scale is not None else 1.0)
        dm = list(dims or [1.8, 1.5, 4.6])
        corr = model_corr if model_corr is not None else SAM3D_MODEL_CORR
        tn = type_name or "道路车辆(自定义模型)"
    else:
        ego = getattr(tm, "ego", None) or {}
        ply = ego.get("ply_path") or os.environ.get("DGGT_EGO_PLY", EGO_PLY_DEFAULT)
        sc = float(ego.get("scale") or (scale if scale is not None else 1.0))
        dm = list(ego.get("dimensions") or dims or [1.7, 1.26, 4.01])
        corr = SAM3D_MODEL_CORR
        tn = type_name or "道路车辆(默认模型)"
    if not ply or not os.path.exists(str(ply)):
        return {"ok": False, "reason": "模型 PLY 不存在: %s" % ply}

    tm.push_history()
    obj_id = tm.add_sam3d_track(ply, poses, dm, scale=sc, model_corr=corr, type_name=tn)
    try:
        spec = tm.synthetic_tracks[int(obj_id)]
        spec["route_lane_id"] = str(lane_id)
        spec["route_direction"] = "backward" if backward else "forward"
        spec["route_speed_mps"] = float(speed_mps)
        spec["route_extend_m"] = float(extend_m)
        spec["route_len_m"] = round(polyline_total(route), 2)
        if source:
            spec["source"] = str(source)
    except Exception:  # noqa: BLE001
        pass
    return {
        "ok": True, "track_id": int(obj_id), "lane_id": str(lane_id),
        "direction": "backward" if backward else "forward",
        "speed_mps": float(speed_mps), "num_frames": len(poses),
        "route_len_m": round(polyline_total(route), 2),
        "model": "custom" if (str(model).lower() == "custom" and ply_path) else "default",
        "first_frame": int(min(poses.keys())), "last_frame": int(max(poses.keys())),
    }
