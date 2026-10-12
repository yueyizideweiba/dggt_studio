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

from road_rules import sample_polyline, polyline_total, project_arc, _dedup_polyline, _dir_at_start, _dir_at_end


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
                ground: Optional[Callable[[float, float], float]] = None,
                dims: Optional[List[float]] = None, up_sign: float = 1.0) -> Dict[int, np.ndarray]:
    """沿路线逐帧生成"严格按路线"的位姿（位置=折线采样、朝向=切线、y=物体**中心**高）。

    pose 的 y 是物体包围盒**中心**（不是路面接触点）：路面高 + up_sign * 半高。
    否则车会"陷进地里"半截。
    """
    P = np.asarray(route_pts, dtype=np.float64)
    if P is None or len(P) < 2:
        return {}
    if backward:
        P = P[::-1].copy()
    dh = (0.5 * float(dims[1])) if (dims and len(dims) >= 2) else 0.0
    # 地面参考：路线折线自带的路面高（平滑、无点云噪声），再用 4DGS 点云在若干
    # 参考点处校准一个**全局垂直偏移**（高精地图与 4DGS 重建可能差一个常数 datum）。
    # 这样既不会像"逐点用 ground_y_at"那样一颠一颠，也不会像"纯用折线 y"那样整体悬浮/下沉。
    offset = 0.0
    if ground is not None:
        refs = []
        n = len(P)
        for i in np.linspace(0, n - 1, min(6, n)).astype(int):
            try:
                gy = float(ground(float(P[i][0]), float(P[i][2])))
                if gy is not None:
                    refs.append(gy - float(P[i][1]))
            except Exception:  # noqa: BLE001
                pass
        if refs:
            offset = float(np.median(refs))
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
        y = float(c[1]) + float(offset) + float(up_sign) * dh
        poses[int(f)] = _yaw_pose(x, y, z, yaw)
    return poses


def _resolve_model(tm, model: str, ply_path: Optional[str],
                   dims: Optional[List[float]], scale: Optional[float],
                   model_corr, type_name: Optional[str]):
    """解析车辆外观 → (ply, scale, dims, model_corr, type_name)。"""
    from track_manager import EGO_PLY_DEFAULT, SAM3D_MODEL_CORR

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
    return ply, sc, dm, corr, tn


def _up_sign(tm) -> float:
    fn = getattr(tm, "world_up_sign", None)
    return float(fn()) if callable(fn) else 1.0


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
    route = build_route(road_model, lane_id, extend_m=extend_m)
    if route is None or len(route) < 2:
        return {"ok": False, "reason": "该车道没有可用的中心线折线（或地图不可用）"}

    ply, sc, dm, corr, tn = _resolve_model(tm, model, ply_path, dims, scale, model_corr, type_name)
    if not ply or not os.path.exists(str(ply)):
        return {"ok": False, "reason": "模型 PLY 不存在: %s" % ply}

    frames = list(range(int(start_frame), int(start_frame) + int(num_frames) + 1))
    backward = str(direction).strip().lower() in ("backward", "back", "reverse", "neg", "-1")
    ground = getattr(tm, "ground_y_at", None)
    poses = route_poses(tm, route, frames, fps, float(speed_mps), backward=backward,
                        start_arc=float(start_arc), ground=ground if callable(ground) else None,
                        dims=dm, up_sign=_up_sign(tm))
    if len(poses) < 2:
        return {"ok": False, "reason": "路线太短或速度过慢，生成的帧数不足 2 帧"}

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


# --------------------------------------------------------------------------- 起终点自动算路
def _trim_lane(P: np.ndarray, a0: float, a1: float) -> Optional[np.ndarray]:
    """取车道折线上 [a0, a1] 弧长区间（按 ~1m 采样）。"""
    P = np.asarray(P, dtype=np.float64)
    if len(P) < 2:
        return None
    total = polyline_total(P)
    a0 = max(0.0, float(a0))
    a1 = min(total, float(a1))
    if a1 - a0 < 0.5:
        return None
    n = max(2, int((a1 - a0) / 1.0) + 1)
    out = []
    for s in np.linspace(a0, a1, n):
        c, _ = sample_polyline(P, float(s))
        if c is not None:
            out.append(c)
    return np.asarray(out, dtype=np.float64) if len(out) >= 2 else None


def _resample(P: np.ndarray, step: float = 1.0) -> np.ndarray:
    """按固定弧长步长重采样折线。"""
    P = np.asarray(P, dtype=np.float64)
    if len(P) < 2:
        return P
    total = polyline_total(P)
    if total < 1e-6:
        return P
    n = max(2, int(total / max(0.2, float(step))) + 1)
    out = []
    for s in np.linspace(0.0, total, n):
        c, _ = sample_polyline(P, float(s))
        if c is not None:
            out.append(c)
    return np.asarray(out, dtype=np.float64)


def _smooth_polyline(P: np.ndarray, passes: int = 3, window: int = 12,
                     max_window: int = 28) -> np.ndarray:
    """窗口移动平均（端点固定），**窗宽随局部转角自适应**：

    - 直行段（转角≈0°）用 `window` 米的小窗 → 保持贴路、不削弯；
    - 路口/变道的折角（转角大）每 30° 加 6m 窗，上限 `max_window` → 把锐角磨成圆角，
      符合真实车辆"转弯切圆"的样子（路口盒子内本来就没有车道，切角不会掉 off-road）。

    窗必须对称且随位置收缩，否则端点附近会把 p[1] 推到前方 ~w/2 米处产生"瞬移"。
    """
    p = np.asarray(P, dtype=np.float64).copy()
    n = len(p)
    if n < 3:
        return p
    w = max(1, int(window))
    for _ in range(max(1, int(passes))):
        q = p.copy()
        for i in range(1, n - 1):
            v1 = p[i][[0, 2]] - p[i - 1][[0, 2]]
            v2 = p[i + 1][[0, 2]] - p[i][[0, 2]]
            n1 = float(np.linalg.norm(v1))
            n2 = float(np.linalg.norm(v2))
            ang = 0.0
            if n1 > 1e-6 and n2 > 1e-6:
                c = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
                ang = math.degrees(math.acos(c))
            extra = int(ang / 30.0) * 6           # 每 30° 转角多加 6m 平滑窗
            wi = min(w + extra, int(max_window), i, n - 1 - i)
            q[i] = p[i - wi:i + wi + 1].mean(axis=0)
        p = q
    return p


def _lane_mean_dir(road_model, lid):
    """车道的整体行驶方向（单位向量，XZ 平面）。曲线车道取各段方向的平均，比端点方向稳。"""
    try:
        P = road_model.lane_pts(lid)
        if P is None or len(P) < 2:
            return None
        P = np.asarray(P, dtype=np.float64)
        v = np.diff(P[:, [0, 2]], axis=0)
        n = np.linalg.norm(v, axis=1, keepdims=True)
        v = np.divide(v, np.maximum(n, 1e-9))
        s = v.sum(axis=0)
        sl = float(np.linalg.norm(s))
        return (s / sl) if sl > 1e-6 else None
    except Exception:  # noqa: BLE001
        return None


def _neighbor_ids(road_model, lid) -> List[str]:
    """车道 `lid` 的**全部**相邻车道（left_neighbors ∪ right_neighbors）。

    不要用 `neighbor_lane()`：它每侧只返回"离中点最近的**一条**"邻居，路口车道的中点
    往往离**转弯进口道**最近（dot≈0.3），于是返回了转弯道、把真正平行的邻道漏掉——
    再叠加方向门槛，就会让路口车道"完全无法变道"。这里把地图里声明的邻居全列出来，
    由调用方逐条做"平行同向"过滤。
    """
    try:
        lanes = getattr(road_model, "map", {}).get("lanes", {})
    except Exception:  # noqa: BLE001
        lanes = {}
    L = lanes.get(str(lid))
    if not L:
        return []
    out: List[str] = []
    for key in ("left_neighbors", "right_neighbors"):
        for x in (L.get(key) or []):
            s = str(x)
            if s in lanes and s not in out:
                out.append(s)
    return out


def _lane_conn(road_model, a, b):
    """后继边 a→b 的几何连续性：(gap, dot)；(gap, dot) 判定失败返回 (None, None)。

    gap = a 末点到 b 首点的距离；dot = 两段方向余弦（1=直行，-1=掉头/反向车道）。

    地图的 `exit_lanes` 有时会把**平行的相邻车道**也列成后继（例如 81→85 其实是并排的
    两条车道），硬接会得到"开到末端再倒回 50m"的 177° 折角，这里用 dot 否决掉。
    """
    try:
        PA = road_model.lane_pts(a)
        PB = road_model.lane_pts(b)
        if PA is None or PB is None or len(PA) < 2 or len(PB) < 2:
            return None, None
        PA = np.asarray(PA, dtype=np.float64)
        PB = np.asarray(PB, dtype=np.float64)
        gap = float(np.linalg.norm(PA[-1][[0, 2]] - PB[0][[0, 2]]))
        dot = float(np.dot(_dir_at_end(PA), _dir_at_start(PB)))
        return gap, dot
    except Exception:  # noqa: BLE001
        return None, None


def _astar_lanes(road_model, start_lane, end_lane, allow_lane_change: bool = True,
                 max_lanes: int = 25, lane_change_cost: float = 30.0,
                 turn_penalty: float = 8.0, lane_change_len: float = 18.0,
                 gap_penalty: float = 0.5):
    """车道图最短路：节点=车道；边 = 后继（含转弯代价）+ 邻道变道（变道代价）。

    **代价必须"按米记账"**（这是本函数的关键约束）：
      后继边 a→b 代价 ≈ 走完整条车道 a 的真实行驶距离 `lane_length(a)`；
      变道边 a→b 代价 ≈ 变道横向过渡实际前进的 `lane_change_len` 米 + 变道机动惩罚。
    早期版本把变道代价写成常数 14、而后继代价是整条车道长度（几十米），于是 A* 会
    **为了省掉车道长度而疯狂变道**（"从起点一路变道变到终点"），完全脱离道路走向。
    现在变道每米的代价恒高于直行 → 只有"必要变道"（目标车道不连通）才会被选中。

    返回 ([(lane_id, edge_kind), ...], cost)；不可达返回 (None, None)。
    """
    import heapq
    a, b = str(start_lane), str(end_lane)
    if a == b:
        return [(a, "start")], 0.0

    def lane_pts2(lid):
        P = road_model.lane_pts(lid)
        if P is None or len(P) == 0:
            return None
        return np.asarray(P, dtype=np.float64)

    Pb = lane_pts2(b)
    tgt = Pb[len(Pb) // 2][[0, 2]] if Pb is not None else None

    _hcache: Dict[str, float] = {}

    def h(lid):
        """可采纳启发：本车道任意点到目标车道中点的最小欧氏距离 × 0.9。

        任何从本车道到目标的路线至少要走过这段直线距离，所以不会高估。
        """
        if tgt is None:
            return 0.0
        if lid in _hcache:
            return _hcache[lid]
        P = lane_pts2(lid)
        val = 0.0
        if P is not None:
            val = float(np.min(np.linalg.norm(P[:, [0, 2]] - tgt, axis=1))) * 0.9
        _hcache[lid] = val
        return val

    def edges(lid):
        out = []
        # 后继边只取 `exit_lanes`（地图的"真实下一车道"），**不要**用 `successors()`——
        # 后者会把 left/right_neighbors（并排邻道/变道候选）也混进后继里，而这些本该由
        # 下面的"变道边"处理。混进来之后再靠 gap/dot 猜"是不是真后继"，会：
        #   (a) 把 90° 的左右转 exit 误判成"平行邻道"丢掉 → 无法左转/右转；
        #   (b) 把 50m 外的平行邻道当后继 → 走出"开到尽头再倒回去"的诡异绕圈。
        try:
            for nxt in road_model.pure_successors(lid):
                if str(nxt) == lid:
                    continue
                gap, dot = _lane_conn(road_model, lid, str(nxt))
                if dot is not None and dot < -0.35:
                    continue               # 反向（本图不存在，仅兜底）
                d = float(dot) if dot is not None else 1.0
                turn = float(turn_penalty) * float(1.0 - max(-1.0, min(1.0, d)))
                cost = float(road_model.lane_length(lid)) + turn + float(gap_penalty) * (gap or 0.0)
                out.append((str(nxt), cost, "successor"))
        except Exception:  # noqa: BLE001
            pass
        if allow_lane_change:
            da = _lane_mean_dir(road_model, lid)
            for nb_id in _neighbor_ids(road_model, lid):
                if str(nb_id) == str(lid):
                    continue
                # 变道只能并入**平行同向**车道：
                #  - 双向道路的"左邻道"往往是对向车道（dot≈-1）→ 逆行，必须拒绝；
                #  - 路口的 left/right_neighbors 还会把**转弯进口道**（与当前车道成 60~130°，
                #    dot≈0.3~0.5）也列成邻居；把这种当"变道"就会斜穿路口、走出诡异绕圈。
                #    真正的变道发生在两段**基本平行**的车道之间（dot≈1），故要求 dot>0.6。
                db = _lane_mean_dir(road_model, nb_id)
                if da is None or db is None or float(np.dot(da, db)) <= 0.6:
                    continue
                # 变道只前进 lane_change_len 米 → 代价按"米 + 机动惩罚"记账
                out.append((nb_id, float(lane_change_len) + float(lane_change_cost),
                            "lane_change"))
        return out

    pq = [(h(a), 0.0, a, [])]
    best = {a: 0.0}
    while pq:
        _f, cost, lid, path = heapq.heappop(pq)
        if lid == b:
            return [(a, "start")] + path, cost
        if cost > best.get(lid, 1e18) + 1e-9:
            continue
        if len(path) >= int(max_lanes):
            continue
        for nxt, w, kind in edges(lid):
            nc = cost + float(w)
            if nc < best.get(nxt, 1e18) - 1e-9:
                best[nxt] = nc
                heapq.heappush(pq, (nc + h(nxt), nc, nxt, path + [(nxt, kind)]))
    return None, None


def _remove_backtracks(P: np.ndarray, max_turn_deg: float = 150.0, passes: int = 4) -> np.ndarray:
    """删除折线里的"折返尖点"（发卡弯）。

    路口附近车道中心线可能首尾相接得很别扭（A 的末点在 B 上更靠后、或两条邻道几乎反向），
    拼接后会出现"往前十几厘米又倒回来"的尖点：既不合法（等于掉头），也会让轨迹曲率爆表。
    这里把转角 > `max_turn_deg` 的顶点删掉——真实路口转弯不会超过这个角度，
    而且折返必然使路径变长，删掉一定更短。
    """
    p = [np.asarray(v, dtype=np.float64) for v in np.asarray(P, dtype=np.float64)]
    if len(p) < 4:
        return np.asarray(p, dtype=np.float64)
    thr = math.cos(math.radians(float(max_turn_deg)))
    for _ in range(max(1, int(passes))):
        if len(p) < 4:
            break
        out = [p[0]]
        removed = 0
        i = 1
        while i < len(p) - 1:
            a, b, c = out[-1], p[i], p[i + 1]
            v1 = b[[0, 2]] - a[[0, 2]]
            v2 = c[[0, 2]] - b[[0, 2]]
            n1, n2 = float(np.linalg.norm(v1)), float(np.linalg.norm(v2))
            if n1 > 1e-6 and n2 > 1e-6 and float(np.dot(v1, v2)) / (n1 * n2) < thr:
                removed += 1          # 折返尖点 → 跳过 b
                i += 1
                continue
            out.append(b)
            i += 1
        out.append(p[-1])
        p = out
        if not removed:
            break
    return np.asarray(p, dtype=np.float64)


def _smoothstep(u: float) -> float:
    u = max(0.0, min(1.0, float(u)))
    return u * u * (3.0 - 2.0 * u)


def _append_pts(pts: List[np.ndarray], seg: np.ndarray, sample_step: float):
    """把一段折线接到 pts 末尾（先补上两段之间的小间隙）。"""
    if seg is None or len(seg) < 1:
        return
    if pts:
        last = pts[-1]
        gap = float(np.linalg.norm(seg[0][[0, 2]] - last[[0, 2]]))
        if gap > 0.05:
            n = max(1, int(gap / max(0.2, sample_step)))
            for k in range(1, n + 1):
                pts.append(last + (seg[0] - last) * (k / n))
    pts.extend(list(seg))


def plan_route(road_model, start_lane, start_arc: float, end_lane, end_arc: float,
               allow_lane_change: bool = True, max_lanes: int = 25,
               lane_change_cost: float = 30.0, turn_penalty: float = 8.0,
               lane_change_len: float = 18.0,
               sample_step: float = 1.0, smooth_passes: int = 3,
               smooth_window: int = 12) -> Optional[Dict[str, Any]]:
    """起终点自动算路：给定起/终点所在车道与弧长，规划一条**严格沿车道中心线**的路线。

    车道图 = 后继边（代价=真实行驶距离，含转弯/连接缺口惩罚）+ 邻道变道边（代价=变道前进
    距离 + 机动惩罚）。**变道每米代价高于直行**，因此只有目标车道不连通时才会变道。
    变道在"当前位置"做横向平滑过渡（不是走到车道末端再硬接），所以变道是 S 形；
    路口转弯由后继车道自然衔接，最后统一重采样 + 平滑，折角被磨圆。
    """
    if road_model is None or not getattr(road_model, "available", False):
        return None
    path, cost = _astar_lanes(road_model, start_lane, end_lane,
                              allow_lane_change=allow_lane_change, max_lanes=max_lanes,
                              lane_change_cost=lane_change_cost, turn_penalty=turn_penalty,
                              lane_change_len=lane_change_len)
    if path is None:
        return None
    pts: List[np.ndarray] = []
    lane_ids: List[str] = []
    cur_lane: Optional[str] = None
    cur_arc = 0.0
    for i, (lid, kind) in enumerate(path):
        P = road_model.lane_pts(lid)
        if P is None or len(P) < 2:
            continue
        P = np.asarray(P, dtype=np.float64)
        total = polyline_total(P)
        is_last = (i == len(path) - 1)
        if cur_lane is None:
            cur_lane, cur_arc = lid, float(start_arc)
        elif kind.startswith("lane_change"):
            # 在 (cur_lane, cur_arc) 处平滑变道到 lid：两条中心线按平滑权重横向混合
            PA = np.asarray(road_model.lane_pts(cur_lane), dtype=np.float64) \
                if road_model.lane_pts(cur_lane) is not None else None
            if PA is None or len(PA) < 2:
                cur_lane, cur_arc = lid, 0.0
            else:
                a_total = polyline_total(PA)
                L = min(float(lane_change_len), max(8.0, a_total - cur_arc), max(8.0, total))
                segA = _trim_lane(PA, cur_arc, min(a_total, cur_arc + L))
                s0B = project_arc(P, segA[0][[0, 2]]) if segA is not None else 0.0
                # 最后一条边就是变道时：变道不要"冲过头"跑到用户点的终点之后
                # （否则车会停在终点前方几米到十几米处）
                if is_last:
                    L = min(L, max(3.0, float(end_arc) - float(s0B)))
                segB = _trim_lane(P, s0B, min(total, s0B + L))
                if segA is not None and segB is not None and len(segA) >= 2 and len(segB) >= 2:
                    n = min(len(segA), len(segB))
                    blend = []
                    for k in range(n):
                        w = _smoothstep(k / max(1, n - 1))
                        blend.append(segA[k] * (1.0 - w) + segB[k] * w)
                    _append_pts(pts, np.asarray(blend, dtype=np.float64), sample_step)
                    # 变道段的几何起点在"源车道"上，所以 lane_ids 也要记上源车道
                    if cur_lane is not None and (not lane_ids or str(lane_ids[-1]) != str(cur_lane)):
                        lane_ids.append(cur_lane)
                    cur_lane, cur_arc = lid, s0B + L
                else:
                    cur_lane, cur_arc = lid, 0.0
        elif str(cur_lane) != str(lid):
            cur_lane, cur_arc = lid, 0.0
        # 走当前车道 [cur_arc, a1]
        # 关键：若**下一条边是变道**，本段不要走到车道末端，否则变道会从车道起点开始混合
        # （等于"先开完整条车道、再倒回去变道"→ 180° 折角）。
        next_kind = path[i + 1][1] if (i + 1 < len(path)) else None
        if is_last:
            a1 = float(end_arc)
        elif next_kind is not None and str(next_kind).startswith("lane_change"):
            a1 = float(cur_arc)
        else:
            a1 = total
        if a1 > cur_arc + 0.3:
            seg = _trim_lane(P, cur_arc, a1)
            if seg is not None:
                _append_pts(pts, seg, sample_step)
                lane_ids.append(lid)
        cur_lane, cur_arc = lid, (0.0 if a1 >= total - 0.3 else a1)
    if len(pts) < 2:
        return None
    # 目标车道可能因为"最后一段太短/经变道衔接"而没被记入 lane_ids，这里补上末尾车道，
    # 保证 lane_ids 的首尾就是实际起点/终点所在车道。
    last_lane = str(path[-1][0])
    if not lane_ids or str(lane_ids[-1]) != last_lane:
        lane_ids.append(last_lane)
    # 先去折返尖点（只删真正的反向折返），再重采样 + 平滑。
    # 注意：**不要在平滑之后再删顶点**——删点会让相邻转角变得更尖锐，反而制造新的尖点。
    P = _remove_backtracks(np.asarray(pts, dtype=np.float64))
    P = _dedup_polyline(P)
    P = _resample(np.asarray(P, dtype=np.float64), sample_step)
    P = _smooth_polyline(P, smooth_passes, smooth_window)
    return {"pts": P, "lane_ids": lane_ids, "num_lanes": len(lane_ids),
            "cost": round(float(cost), 1), "length_m": round(polyline_total(P), 1)}


def create_vehicle_on_path(tm, path_pts, direction: str = "forward",
                           speed_mps: float = 8.0, start_frame: int = 0, num_frames: int = 30,
                           fps: float = 10.0, start_arc: float = 0.0,
                           model: str = "default", ply_path: Optional[str] = None,
                           dims: Optional[List[float]] = None, scale: Optional[float] = None,
                           model_corr=None, type_name: Optional[str] = None,
                           source: Optional[str] = None) -> Dict[str, Any]:
    """在**给定的路线折线**上创建一辆严格沿路线行驶的车辆（用于"起终点自动算路"的结果）。"""
    P = np.asarray(path_pts, dtype=np.float64)
    if P is None or len(P) < 2:
        return {"ok": False, "reason": "路线折线为空"}

    ply, sc, dm, corr, tn = _resolve_model(tm, model, ply_path, dims, scale, model_corr, type_name)
    if not ply or not os.path.exists(str(ply)):
        return {"ok": False, "reason": "模型 PLY 不存在: %s" % ply}

    frames = list(range(int(start_frame), int(start_frame) + int(num_frames) + 1))
    backward = str(direction).strip().lower() in ("backward", "back", "reverse", "-1")
    ground = getattr(tm, "ground_y_at", None)
    poses = route_poses(tm, P, frames, fps, float(speed_mps), backward=backward,
                        start_arc=float(start_arc), ground=ground if callable(ground) else None,
                        dims=dm, up_sign=_up_sign(tm))
    if len(poses) < 2:
        return {"ok": False, "reason": "路线太短或速度过慢，生成的帧数不足 2 帧"}

    tm.push_history()
    obj_id = tm.add_sam3d_track(ply, poses, dm, scale=sc, model_corr=corr, type_name=tn)
    try:
        spec = tm.synthetic_tracks[int(obj_id)]
        spec["route_direction"] = "backward" if backward else "forward"
        spec["route_speed_mps"] = float(speed_mps)
        spec["route_len_m"] = round(polyline_total(P), 2)
        spec["route_pts"] = [[round(float(v), 2) for v in p[:3]] for p in P]
        if source:
            spec["source"] = str(source)
    except Exception:  # noqa: BLE001
        pass
    return {"ok": True, "track_id": int(obj_id),
            "direction": "backward" if backward else "forward",
            "speed_mps": float(speed_mps), "num_frames": len(poses),
            "route_len_m": round(polyline_total(P), 2),
            "model": "custom" if (str(model).lower() == "custom" and ply_path) else "default",
            "first_frame": int(min(poses.keys())), "last_frame": int(max(poses.keys()))}
