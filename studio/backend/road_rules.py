"""把高精地图当作一套**道路约束规则**，服务给路径编辑与事故生成。

设计原则（对既有流程是"加法"而不是"改写"）：
  * `RoadModel` 只做**查询 + 约束检查 + 合法路径规划**，不碰 TrackManager 的状态；
  * 轨迹类接口统一收/发 `{frame: 4x4 pose}` 或 `Nx3 中心点`，调用方拿到结果直接写回；
  * 没有地图（老场景）时 `RoadModel.available == False`，所有接口都安全退化成 no-op，
    调用方不需要写 if。

规则清单（都对应 Waymo 地图里真实存在的几何）：
  R1 车道走廊：轨迹每个点应落在某条 LaneCenter 走廊内（默认 ±3.5m）；超出就横向吸附回去。
  R2 航向一致：同一车道上，车辆前进方向应与该车道中心线方向一致；反向 = 逆行（wrong_way）。
  R3 车道连通：跨车道只能走 LaneCenter 的 entry/exit/左右邻居边 —— 不许"凭空横跳"。
  R4 路口转向：左/右转只允许发生在"出口>=2 的路口进口"处，且要沿出口支路的几何转弯。
  R5 限速：车速上限取所在车道的 `speed_limit_mph`（没写就用默认值）。
  R6 路口锚点：停车标志/斑马线/路口进口可以作为"在某处发生什么"的语义锚点。

对外主要接口：
    rm = RoadModel(scene_dir)                    # 或 get_road_model(scene_dir) 带缓存
    rm.available
    rm.lane_at(xz)                               # (lane_id, dist)
    rm.lane_dir(lane_id, xz)                     # 单位方向 (dx, dz)
    rm.snap_centers(centers)                     # 轨迹横向吸附 + 报告
    rm.classify(centers)                         # 在不在路上/逆行/超速
    rm.on_road_report(centers)                   # 给前端 warning 用的紧凑报告
    rm.junction_ahead(xz, heading)               # 前方路口锚点
    rm.plan_along_lane(xz, lane_id, back_m, fwd_m)
    rm.plan_turn(xz, heading, direction)         # 沿出口支路真实几何转弯
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

import waymo_map as wm

# 道路约束默认参数
LANE_TOL_M = 3.5          # 距离车道中心线多远算"在路上"
SNAP_MAX_M = 2.6          # 单帧横向吸附的最大修正量（超过就不动，只报警）
DEFAULT_SPEED_LIMIT_MPS = 16.7   # 60 km/h
_CACHE: Dict[str, Optional["RoadModel"]] = {}


def get_road_model(scene_dir: str, use_cache: bool = True) -> Optional["RoadModel"]:
    """按场景取道路模型（带缓存）。场景没有高精地图时返回 None。"""
    if not scene_dir:
        return None
    if use_cache and scene_dir in _CACHE:
        return _CACHE[scene_dir]
    rm: Optional[RoadModel] = None
    try:
        rm = RoadModel(scene_dir)
    except Exception as e:  # noqa: BLE001
        print('[road_rules] 场景 %s 没有可用道路数据: %s' % (scene_dir, e))
        rm = None
    if use_cache:
        _CACHE[scene_dir] = rm
    return rm


def clear_cache():
    _CACHE.clear()


def _wrap_pi(a: float) -> float:
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a


def turn_side(h0: np.ndarray, h1: np.ndarray, up_sign: float) -> float:
    """h1 相对 h0 的水平转向角（弧度，带符号）。

    **>0 = 左转，<0 = 右转**（与地图数据实测一致：scene 016 路口处，左转支路测得 +85°~+111°，
    右转支路 -85°~-94°）。
    """
    c = float(np.cross(np.asarray(h0, float), np.asarray(h1, float))[1])
    ang = math.atan2(c, float(np.dot(h0, h1)))
    return ang * float(up_sign)


class RoadModel:
    """场景世界系下的道路约束模型。"""

    def __init__(self, scene_dir: str):
        self.scene_dir = scene_dir
        self.map = wm.load_scene_map(scene_dir)
        self.up_sign = float(self.map['align'].get('up_sign') or -1.0)
        self._lane_len_cache: Dict[str, float] = {}

    # ------------------------------------------------------------ 基础查询

    @property
    def available(self) -> bool:
        return True

    @property
    def n_lanes(self) -> int:
        return len(self.map['lanes'])

    def lane_ids(self) -> List[str]:
        return list(self.map['lanes'].keys())

    def lane_at(self, xz, max_d: float = 12.0, heading=None,
                max_ang: float = 75.0) -> Tuple[Optional[str], float]:
        """取 xz 附近的车道；给了 `heading` 就只在**大致同向**的车道里挑。

        为什么必须要方向过滤：路口附近不同方向的车道中心线会挨得极近。实测 scene 005 的
        自车在路口处到"最近车道中心线"只有 0.2m，但那条是**横穿**的车道（方向差 88°）——
        纯按距离选会把正常行驶的车判成逆行，横向吸附时还可能把车吸到对面的车道上。
        """
        q = np.asarray(xz, dtype=np.float64).reshape(-1)[:2]
        h2 = None
        if heading is not None:
            hh = np.asarray(heading, dtype=np.float64).reshape(-1)
            hh = np.array([hh[0], hh[2]]) if hh.size >= 3 else hh[:2]
            n = float(np.linalg.norm(hh))
            if n > 1e-9:
                h2 = hh / n
        best, bd, any_best, any_d = None, float('inf'), None, float('inf')
        for lid in self.map['lanes']:
            P = self._lane_xz(lid)
            if P is None or len(P) == 0:
                continue
            d = float(np.min(np.linalg.norm(P - q, axis=1)))
            if d < any_d:
                any_best, any_d = lid, d
            if d > max_d or d >= bd:
                continue
            if h2 is not None:
                ld = self.lane_dir(lid, q)
                if ld is not None:
                    cc = float(np.clip(float(np.dot(h2, ld)), -1.0, 1.0))
                    if math.degrees(math.acos(cc)) > max_ang:
                        continue
            best, bd = lid, d
        if best is not None:
            return best, bd
        # 方向过滤后没有候选：返回最近的（调用方从 lane_dir_ok 之类的体检里能看出不对）
        return (any_best, any_d) if any_d <= max_d else (None, any_d)

    def _lane_xz(self, lane_id) -> Optional[np.ndarray]:
        P = self.lane_pts(lane_id)
        return None if P is None else P[:, [0, 2]]

    def lane_dir(self, lane_id, xz) -> Optional[np.ndarray]:
        return wm.lane_heading(self.map, lane_id, xz)

    def lane_length(self, lane_id) -> float:
        k = str(lane_id)
        if k not in self._lane_len_cache:
            self._lane_len_cache[k] = wm.lane_length(self.map, lane_id)
        return self._lane_len_cache[k]

    def lane_pts(self, lane_id) -> Optional[np.ndarray]:
        return wm.lane_polyline(self.map, lane_id)

    def successors(self, lane_id) -> List[str]:
        return wm.successors(self.map, lane_id)

    def pure_successors(self, lane_id) -> List[str]:
        l = self.map['lanes'].get(str(lane_id))
        if not l:
            return []
        return [str(x) for x in l.get('exit_lanes') or [] if str(x) in self.map['lanes']]

    def speed_limit_mps(self, lane_id) -> float:
        l = self.map['lanes'].get(str(lane_id)) or {}
        mph = l.get('speed_limit_mph')
        try:
            v = float(mph)
            if v > 0:
                return v * 0.44704
        except (TypeError, ValueError):
            pass
        return DEFAULT_SPEED_LIMIT_MPS

    # ------------------------------------------------------------ R1/R2/R5：约束检查

    def _right_vec(self, heading) -> np.ndarray:
        """场景系里"车的右方"= forward × up。"""
        h = np.asarray(heading, dtype=np.float64).reshape(-1)
        f = np.array([h[0], 0.0, h[2]])
        n = float(np.linalg.norm(f))
        f = f / n if n > 1e-9 else np.array([0.0, 0.0, 1.0])
        up = np.array([0.0, -1.0, 0.0]) if self.up_sign < 0 else np.array([0.0, 1.0, 0.0])
        return np.cross(f, up)

    def neighbor_lane(self, lane_id, side: str, xz=None) -> Optional[Dict[str, Any]]:
        """取车道的左/右相邻车道（规则 R3：跨车道只能走地图里真的连着的邻居边）。

        map 里的 `left_neighbors` / `right_neighbors` 是按**行驶方向**定义的，这里再用
        几何复核一遍（邻居中心线相对本车道的横向偏移方向要和 `side` 一致），
        避免左右命名与场景系朝向不一致时选错。
        """
        L = self.map['lanes'].get(str(lane_id))
        if not L:
            return None
        cands = []
        for key in ('left_neighbors', 'right_neighbors'):
            cands += [str(x) for x in (L.get(key) or []) if str(x) in self.map['lanes']]
        P = np.asarray(L['pts'], dtype=np.float64)
        if xz is None:
            i = len(P) // 2
            xz = P[i][[0, 2]]
        else:
            xz = np.asarray(xz, dtype=np.float64)[:2]
        h = self.lane_dir(lane_id, xz)
        if h is None:
            h = _dir_at_start(P)[[0, 2]]
        right = self._right_vec(np.array([h[0], 0.0, h[1]]))[[0, 2]]
        want = 1.0 if str(side).lower().startswith('r') else -1.0

        # 地图没给邻居边时的几何兜底：找一条**方向平行**、且横向偏在指定一侧的车道。
        if not cands:
            for c, L2 in self.map['lanes'].items():
                if str(c) == str(lane_id):
                    continue
                Q = self.lane_pts(c)
                if Q is None or len(Q) < 2:
                    continue
                j = int(np.argmin(np.linalg.norm(Q[:, [0, 2]] - xz, axis=1)))
                h2 = self.lane_dir(c, Q[j][[0, 2]])
                if h2 is None:
                    continue
                par = abs(float(np.dot(h2, h)))
                if par < 0.75:                     # 只接受大致同向的车道（不是对向/横穿）
                    continue
                off = float(np.dot(Q[j][[0, 2]] - xz, right))
                if off * want <= 0.5 or abs(off) > 8.0:
                    continue
                cands.append(str(c))
        best, bs = None, None
        for c in cands:
            Q = self.lane_pts(c)
            if Q is None or len(Q) == 0:
                continue
            j = int(np.argmin(np.linalg.norm(Q[:, [0, 2]] - xz, axis=1)))
            off = float(np.dot(Q[j][[0, 2]] - xz, right))
            if off * want <= 0.2:          # 方向不对（或几乎重合）就不要
                continue
            if abs(off) > 6.0:             # 一次变道最多跨 1~2 条车道，十几米外的不算
                continue
            score = abs(off)
            if bs is None or score < bs:
                best, bs = {'lane': c, 'offset_m': off, 'score': score}, score
        return best

    def classify_point(self, xz, heading: Optional[np.ndarray] = None,
                       tol: float = LANE_TOL_M) -> Dict[str, Any]:
        lid, d = self.lane_at(xz, max_d=1e9, heading=heading)
        out = {'lane': lid, 'dist_m': d, 'on_road': bool(d <= tol)}
        if lid is not None:
            out['speed_limit_mps'] = self.speed_limit_mps(lid)
            ld = self.lane_dir(lid, xz)
            if ld is not None and heading is not None:
                h = np.asarray(heading, dtype=np.float64).reshape(-1)
                h = np.array([h[0], h[2]]) if h.size >= 3 else h[:2]
                n = float(np.linalg.norm(h))
                if n > 1e-9:
                    h = h / n
                    c = float(np.clip(float(np.dot(h, ld)), -1.0, 1.0))
                    out['heading_err_deg'] = math.degrees(math.acos(c))
                    out['wrong_way'] = bool(c < -0.3)
        return out

    def classify(self, centers: Dict[int, np.ndarray], tol: float = LANE_TOL_M) -> Dict[str, Any]:
        """对一条轨迹（frame → 中心点）做道路合规性体检。"""
        if not centers:
            return {'n': 0, 'on_road_ratio': 0.0, 'lanes': [], 'wrong_way': False}
        frames = sorted(centers)
        ds, offs, lanes, errs = [], [], [], []
        wrong = 0
        for i, f in enumerate(frames):
            c = np.asarray(centers[f], dtype=np.float64).reshape(3)
            h = None
            if i > 0:
                p = np.asarray(centers[frames[i - 1]], dtype=np.float64).reshape(3)
                h = np.array([c[0] - p[0], 0.0, c[2] - p[2]])
            if i == 0 and len(frames) > 1:
                p = np.asarray(centers[frames[1]], dtype=np.float64).reshape(3)
                h = np.array([p[0] - c[0], 0.0, p[2] - c[2]])
            r = self.classify_point(c[[0, 2]], h, tol=tol)
            ds.append(r['dist_m'])
            lanes.append(r.get('lane'))
            if r.get('wrong_way'):
                wrong += 1
            if 'heading_err_deg' in r:
                errs.append(r['heading_err_deg'])
                ld = self.lane_dir(r['lane'], c[[0, 2]])
                if ld is not None:
                    # 带符号的横向偏移（左正右负），用于判断压线/占道
                    nrm = np.array([-ld[1], ld[0]])   # 车道方向的左法向（场景 xz）
                    offs.append(float(np.dot(c[[0, 2]] - _closest_on_lane(self.lane_pts(r['lane']), c[[0, 2]]), nrm)))
        ds = np.asarray(ds, dtype=np.float64) if ds else np.zeros(1)
        return {
            'n': len(frames),
            'on_road_ratio': float(np.mean(ds <= tol)),
            'dist_med_m': float(np.median(ds)),
            'dist_max_m': float(np.max(ds)),
            'lanes': sorted({l for l in lanes if l is not None}),
            'heading_err_med_deg': float(np.median(errs)) if errs else None,
            'offset_med_m': float(np.median(offs)) if offs else None,
            'wrong_way_frames': int(wrong),
            'wrong_way': bool(wrong > max(1, 0.25 * len(frames))),
        }

    def on_road_report(self, centers: Dict[int, np.ndarray],
                       tol: float = LANE_TOL_M) -> Dict[str, Any]:
        """紧凑报告（给前端 warning 用）：只保留需要提醒用户的点。"""
        c = self.classify(centers, tol=tol)
        issues = []
        if c['n'] and c['on_road_ratio'] < 0.8:
            issues.append('有 %d%% 的帧偏离车道中心线超过 %.1fm（最大 %.1fm）' % (
                round(100 * (1 - c['on_road_ratio'])), tol, c['dist_max_m']))
        if c['wrong_way']:
            issues.append('检测到逆行（%d 帧航向与车道方向相反）' % c['wrong_way_frames'])
        if c.get('heading_err_med_deg') is not None and c['heading_err_med_deg'] > 35:
            issues.append('航向与车道方向平均偏差 %.0f°' % c['heading_err_med_deg'])
        return {'available': True, 'issues': issues, **c}

    # ------------------------------------------------------------ 横向吸附

    def _traj_headings(self, centers: Dict[int, np.ndarray]):
        """由中心点序列算每帧的前进方向（场景 xz）。"""
        frames = sorted(centers)
        out: Dict[int, Optional[np.ndarray]] = {}
        for i, f in enumerate(frames):
            a = np.asarray(centers[f], dtype=np.float64).reshape(3)
            if i + 1 < len(frames):
                b = np.asarray(centers[frames[i + 1]], dtype=np.float64).reshape(3)
                v = np.array([b[0] - a[0], 0.0, b[2] - a[2]])
            elif i > 0:
                p = np.asarray(centers[frames[i - 1]], dtype=np.float64).reshape(3)
                v = np.array([a[0] - p[0], 0.0, a[2] - p[2]])
            else:
                v = None
            n = float(np.linalg.norm(v)) if v is not None else 0.0
            out[f] = (v / n) if n > 1e-9 else None
        return out

    def snap_point(self, xz, lane: Optional[str] = None,
                   max_shift: float = SNAP_MAX_M, heading=None) -> Dict[str, Any]:
        """把一个点横向吸附到最近（或指定）车道的中心线上。

        只做**横向**修正（保留沿车道方向的进度），且修正量超过 `max_shift` 就不动 ——
        宁可保留原样并报警，也不能把车"瞬移"到几十米外的另一条路上。
        `heading` 用于挑车道（路口附近横向车道挨得极近，不按方向过滤会吸错）。
        """
        xz = np.asarray(xz, dtype=np.float64)[:2]
        if lane is None:
            lid, _d = self.lane_at(xz, heading=heading)
            lane = lid
        if lane is None:
            return {'xz': xz, 'lane': None, 'shift': 0.0, 'applied': False, 'dist': float('inf')}
        P = self.lane_pts(lane)
        if P is None or len(P) == 0:
            return {'xz': xz, 'lane': None, 'shift': 0.0, 'applied': False, 'dist': float('inf')}
        Q = P[:, [0, 2]]
        dd = np.linalg.norm(Q - xz, axis=1)
        j = int(np.argmin(dd))
        shift = float(dd[j])
        if shift > max_shift:
            return {'xz': xz, 'lane': lane, 'shift': shift, 'applied': False, 'dist': shift,
                    'reason': 'shift_too_large'}
        return {'xz': Q[j], 'lane': lane, 'shift': shift, 'applied': True, 'dist': shift}

    def snap_centers(self, centers: Dict[int, np.ndarray], lane: Optional[str] = None,
                     max_shift: float = SNAP_MAX_M, per_frame: bool = False) -> Dict[str, Any]:
        """对整条轨迹做横向吸附（车道走廊约束 R1）。

        `per_frame=True` 时每帧吸附到**各自最近**的车道（用于变道轨迹，保留变道结构）；
        默认吸附到"整条轨迹的主导车道"（更稳，避免逐帧跳到隔壁车道）。

        返回 {'centers': 修正后的中心点, 'applied', 'snapped_frames', 'skipped_frames',
              'lane', 'report', 'report_after'}
        """
        if not centers:
            return {'centers': {}, 'applied': False, 'snapped_frames': 0, 'skipped_frames': 0,
                    'lane': None, 'report': {'n': 0}}
        before = self.classify(centers)
        frames = sorted(centers)
        heads = self._traj_headings(centers)
        # 主导车道：整条轨迹中出现最多的车道（更稳，避免逐帧跳到隔壁车道）。
        # 选车道时按每帧的前进方向过滤，否则路口处会统计到横穿的车道。
        dom = lane
        if dom is None and not per_frame:
            votes: Dict[str, int] = {}
            for f in frames:
                lid, _d = self.lane_at(np.asarray(centers[f]).reshape(3)[[0, 2]],
                                       heading=heads.get(f))
                if lid is not None:
                    votes[lid] = votes.get(lid, 0) + 1
            if votes:
                dom = max(votes.items(), key=lambda kv: kv[1])[0]
        if dom is None and not per_frame:
            return {'centers': dict(centers), 'applied': False, 'snapped_frames': 0,
                    'skipped_frames': len(frames), 'lane': None, 'report': before}
        out: Dict[int, np.ndarray] = {}
        snapped = skipped = 0
        used_lanes: Dict[str, int] = {}
        for f in frames:
            c = np.asarray(centers[f], dtype=np.float64).reshape(3).copy()
            r = self.snap_point(c[[0, 2]], lane=dom if not per_frame else None,
                                max_shift=max_shift, heading=heads.get(f))
            if r['applied']:
                c[0], c[2] = float(r['xz'][0]), float(r['xz'][1])
                snapped += 1
                if r.get('lane'):
                    used_lanes[r['lane']] = used_lanes.get(r['lane'], 0) + 1
            else:
                skipped += 1
            out[f] = c
        after = self.classify(out)
        improved = (after['dist_med_m'] < before['dist_med_m'] - 0.02
                    or after['dist_max_m'] < before['dist_max_m'] - 0.02)
        applied = bool(snapped and improved and skipped <= 0.2 * len(frames))
        return {'centers': out if applied else dict(centers), 'applied': applied,
                'snapped_frames': snapped, 'skipped_frames': skipped, 'lane': dom,
                'lanes_used': sorted(used_lanes, key=lambda k: -used_lanes[k])[:6],
                'report': before, 'report_after': after}

    # ------------------------------------------------------------ R4：路口

    def junction_ahead(self, xz, heading, max_dist: float = 60.0) -> Optional[Dict[str, Any]]:
        """沿"当前车道方向"往前找最近的**路口进口**（出口 >= 2 的车道末端）。

        比"离某个锚点最近"更符合语义：车在路口之前，左边那个路口不算"前方路口"。
        """
        lane, d = self.lane_at(xz, max_d=8.0, heading=heading)
        if lane is None:
            return None
        xz = np.asarray(xz, dtype=np.float64)[:2]
        h = np.asarray(heading, dtype=np.float64)
        h = np.array([h[0], 0.0, h[2]]) if h.size >= 3 else np.array([0.0, 0.0, 1.0])
        n = float(np.linalg.norm(h[[0, 2]]))
        h = h / n if n > 1e-9 else np.array([0.0, 0.0, 1.0])
        acc = 0.0
        cur = lane
        for _hop in range(24):
            L = self.map['lanes'].get(str(cur))
            if not L:
                return None
            P = np.asarray(L['pts'], dtype=np.float64)
            if len(P) < 2:
                return None
            exits = [str(x) for x in (L.get('exit_lanes') or []) if str(x) in self.map['lanes']]
            if len(exits) >= 2:
                # 该进口的末端就是路口位置；确认它在车辆"前方"
                end = P[-1][[0, 2]]
                v = end - xz
                fwd = float(v[0] * h[0] + v[1] * h[2])
                dist = float(np.linalg.norm(v))
                if fwd > -2.0 and dist <= max_dist:
                    return {'lane': str(cur), 'at': [float(P[-1][0]), float(P[-1][1]), float(P[-1][2])],
                            'exits': exits, 'dist_m': dist, 'lane_ids': [str(cur)]}
                return None
            nxt = [str(x) for x in (L.get('exit_lanes') or []) if str(x) in self.map['lanes']]
            if len(nxt) != 1:
                return None
            acc += self.lane_length(cur)
            if acc > max_dist:
                return None
            cur = nxt[0]
        return None

    def plan_turn(self, xz, heading, direction: str, *, forward_m: float = 55.0,
                  back_m: float = 8.0, max_hops: int = 6, max_paths: int = 96,
                  ) -> Optional[Dict[str, Any]]:
        """沿地图真实几何规划一次转弯（"在路口左转/右转/直行"）。

        为什么要在**车道图**上搜、而不是直接在路口进口挑出口：Waymo 的路口结构是
        "进口车道先扇出成若干条平行车道，再各自接到不同去向"（实测 scene 016：
        进口 49 → {40,52,53,54} → 分别接 39 / 58(左, +111°) / 50 / …）。只差一跳
        看不出转向，必须往前搜几跳，用**路径走 forward_m 之后的净航向变化**判定左右。

        返回一条路线折线（场景系）+ 用到的车道序列；调用方按弧长铺逐帧位姿，
        位置和航向都来自地图 —— 所以是真的"在路口拐弯"，而不是原地转方向。
        """
        lane0, _d = self.lane_at(xz, max_d=8.0, heading=heading)
        if lane0 is None:
            return None
        xz = np.asarray(xz, dtype=np.float64)[:2]
        P0 = self.lane_pts(lane0)
        if P0 is None or len(P0) < 2:
            return None
        s0 = project_arc(P0, xz)
        h_start = _dir_at_start(P0)

        # ---- 在车道图上做有界 BFS，收集候选路径 ----
        frontier = [{'lanes': [str(lane0)], 'segs': [P0]}]
        results: List[Dict[str, Any]] = []
        for _hop in range(int(max_hops)):
            nxt_list = []
            for p in frontier:
                ex = self.pure_successors(p['lanes'][-1])
                if not ex:
                    results.append(p)
                    continue
                for e in ex:
                    Q = self.lane_pts(e)
                    if Q is None or len(Q) < 2:
                        continue
                    nxt_list.append({'lanes': p['lanes'] + [e], 'segs': p['segs'] + [Q]})
            if not nxt_list:
                break
            results.extend(nxt_list)
            frontier = nxt_list
            if len(results) > int(max_paths):
                break
        if not results:
            return None

        target = {'left': math.radians(88.0), 'right': math.radians(-88.0),
                  'straight': 0.0}.get(direction, 0.0)
        best, best_score, best_ang = None, None, 0.0
        for p in results:
            route = _dedup_polyline(np.concatenate(p['segs'], axis=0))
            total = polyline_total(route)
            if total - s0 < 8.0:
                continue
            probe = min(total, s0 + float(forward_m))
            if probe - s0 < 8.0:
                continue
            _c, h_end = sample_polyline(route, probe)
            ang = turn_side(h_start, h_end, self.up_sign)     # >0 左转
            score = -abs(ang - target)
            if direction == 'straight' and len(p['lanes']) == 2:
                score += 0.25      # 直行时优先用真正连着的下一跳，别绕环岛一整圈
            if best_score is None or score > best_score:
                best, best_score, best_ang = p, score, ang
        if best is None:
            return None
        route = _dedup_polyline(np.concatenate(best['segs'], axis=0))
        want_deg = math.degrees(target)
        got_deg = math.degrees(best_ang)
        # 诚实回报：真正匹配的支路可能根本不存在（这一侧没有左转口），
        # 那就只是"沿本车道往前开"，必须显式告诉调用方，不能假装转过去了。
        short = abs(got_deg - want_deg) > 40.0 and direction != 'straight'
        return {'pts': route, 'lane_ids': best['lanes'], 'direction': direction,
                'turn_deg': float(got_deg), 'target_turn_deg': float(want_deg),
                'short_turn': bool(short),
                'entry_lane': str(lane0), 'start_xz': xz.tolist(),
                'start_arc_m': float(s0), 'back_m': float(back_m),
                'forward_m': float(forward_m), 'n_lanes': len(best['lanes']),
                'up_sign': self.up_sign, 'source': 'lane_graph_bfs'}

    def turn_for_track(self, tm, track_id: int, direction: str, frame_idx: int, *,
                       forward_m: float = 55.0, back_m: float = 8.0):
        """给"某辆车在路口怎么走"直接算路线：取该 track 当前帧的位置/朝向去搜。"""
        try:
            P = tm.get_track_pose(int(track_id), int(frame_idx))
        except Exception:  # noqa: BLE001
            P = None
        if P is None:
            return None
        P = np.asarray(P, dtype=np.float64)
        c = P[:3, 3]
        f = P[:3, 2]
        return self.plan_turn(c[[0, 2]], f, direction, forward_m=forward_m, back_m=back_m)

    def plan_along_lane(self, xz, lane: Optional[str] = None, *, back_m: float = 10.0,
                        fwd_m: float = 60.0) -> Optional[Dict[str, Any]]:
        """沿"当前车道"取一段中心线（前后各若干米），用作铺轨迹的干净骨架。"""
        lane, _ = self.lane_at(xz) if lane is None else (lane, 0.0)
        if lane is None:
            return None
        segs = [self.lane_pts(lane)]
        ids = [str(lane)]
        cur = str(lane)
        acc = 0.0
        for _hop in range(24):
            if acc >= fwd_m:
                break
            nxt = self.pure_successors(cur)
            if len(nxt) != 1:
                break
            Q = self.lane_pts(nxt[0])
            if Q is None:
                break
            segs.append(Q)
            ids.append(nxt[0])
            acc += self.lane_length(nxt[0])
            cur = nxt[0]
        route = _dedup_polyline(np.concatenate([s for s in segs if s is not None], axis=0))
        return {'pts': route, 'lane_ids': ids}


# ---------------------------------------------------------------- 小工具

def _closest_on_lane(P: Optional[np.ndarray], xz) -> np.ndarray:
    if P is None or len(P) == 0:
        return np.asarray(xz, dtype=np.float64)
    Q = P[:, [0, 2]]
    j = int(np.argmin(np.linalg.norm(Q - np.asarray(xz, dtype=np.float64), axis=1)))
    return Q[j]


def _dir_at_start(P: Optional[np.ndarray]) -> np.ndarray:
    if P is None or len(P) < 2:
        return np.array([0.0, 0.0, 1.0])
    for i in range(1, len(P)):
        v = P[i] - P[0]
        v = np.array([v[0], 0.0, v[2]])
        n = float(np.linalg.norm(v))
        if n > 0.5:
            return v / n
    v = P[-1] - P[0]
    v = np.array([v[0], 0.0, v[2]])
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-6 else np.array([0.0, 0.0, 1.0])


def _dir_at_end(P: Optional[np.ndarray]) -> np.ndarray:
    if P is None or len(P) < 2:
        return np.array([0.0, 0.0, 1.0])
    for i in range(len(P) - 2, -1, -1):
        v = P[-1] - P[i]
        v = np.array([v[0], 0.0, v[2]])
        n = float(np.linalg.norm(v))
        if n > 0.5:
            return v / n
    return np.array([0.0, 0.0, 1.0])


def _dedup_polyline(P: np.ndarray, eps: float = 0.05) -> np.ndarray:
    if P is None or len(P) == 0:
        return np.zeros((0, 3), dtype=np.float64)
    keep = [0]
    for i in range(1, len(P)):
        if float(np.linalg.norm(P[i] - P[keep[-1]])) > eps:
            keep.append(i)
    if keep[-1] != len(P) - 1:
        keep.append(len(P) - 1)
    return np.asarray(P, dtype=np.float64)[keep]


# ---------------------------------------------------------------- 采样工具

def sample_polyline(P: np.ndarray, s: float):
    """按弧长 s 在折线上取 (点, 单位方向)。"""
    P = np.asarray(P, dtype=np.float64)
    if len(P) == 0:
        return None, None
    if len(P) == 1:
        return P[0], np.array([0.0, 0.0, 1.0])
    seg = np.linalg.norm(np.diff(P[:, [0, 2]], axis=0), axis=1)
    arc = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(arc[-1])
    if s <= 0:
        c = P[0] + (P[1] - P[0]) * (s / max(1e-9, seg[0])) if seg[0] > 1e-9 else P[0]
        return c, _dir_at_start(P)
    if s >= total:
        last = seg[-1]
        c = P[-1] + (P[-1] - P[-2]) * ((s - total) / last) if last > 1e-9 else P[-1]
        return c, _dir_at_end(P)
    i = int(np.searchsorted(arc, s) - 1)
    i = max(0, min(i, len(seg) - 1))
    u = (s - arc[i]) / max(1e-9, seg[i])
    c = P[i] * (1 - u) + P[i + 1] * u
    v = P[i + 1] - P[i]
    v = np.array([v[0], 0.0, v[2]])
    n = float(np.linalg.norm(v))
    return c, (v / n if n > 1e-9 else _dir_at_start(P))


def project_arc(P: np.ndarray, xz) -> float:
    """点 xz 在折线 P 上的弧长坐标（最近的投影点）。"""
    P = np.asarray(P, dtype=np.float64)
    if len(P) < 2:
        return 0.0
    Q = P[:, [0, 2]]
    q = np.asarray(xz, dtype=np.float64)[:2]
    d = np.linalg.norm(Q - q, axis=1)
    j = int(np.argmin(d))
    seg = np.linalg.norm(np.diff(Q, axis=0), axis=1)
    arc = np.concatenate([[0.0], np.cumsum(seg)])
    return float(arc[j])


def polyline_total(P: np.ndarray) -> float:
    P = np.asarray(P, dtype=np.float64)
    if len(P) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(P[:, [0, 2]], axis=0), axis=1).sum())
