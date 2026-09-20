"""动态物体关系图（scene graph）与 Corner Case 参与者提案。

目标：从"硬编码的逐类型生成"升级为"先理解场景里动态物体之间的关系，再据此
挑选参与者、采样参数、批量生成多样事故"。

- 节点 = 动态物体（真实 track + 合成 track，跳过已删除/已被替换的）。
- 边 = 任意两物体在**当前帧附近窗口**内的关系特征（相对位置/速度、距离、
  接近速率、航向差、交角、TTC、最近距离、关系类型、冲突关键度）。
- `propose_participants()` 用关系类型 + 冲突关键度，为每种事故类型挑选候选的
  参与者元组，从而让"同类事故换位置/换速度"成为可能（多样性来源之一）。

图结构/特征向量是**图神经网络可消费的表示**：后续训练 GNN 时可直接用本模块产出的
节点特征、边特征与关系标签作为输入。
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# 关系类型（边的语义标签）
REL_FOLLOWING = "following"        # 同向，前后跟车
REL_ADJACENT = "adjacent_parallel"  # 同向但横向并列（相邻车道）
REL_ONCOMING = "oncoming"           # 对向
REL_CROSSING = "crossing"           # 交叉（有夹角，30°~150°）
REL_STATIONARY = "stationary"       # 一方静止
REL_RECEDING = "receding"           # 相互远离
REL_UNRELATED = "unrelated"         # 无显著关系


class SceneGraph:
    """基于 TrackManager 构建的动态物体关系图。"""

    def __init__(self, tm, fps: float = 10.0):
        self.tm = tm
        self.fps = float(fps)
        self.nodes: List[Dict[str, Any]] = []
        self.edges: List[Dict[str, Any]] = []
        self._node_by_tid: Dict[int, int] = {}
        self.frame_idx = 0
        self.window = 10

    # ------------------------------------------------------------------ 构建
    def all_track_ids(self) -> List[int]:
        tm = self.tm
        ids = set(int(t) for t in getattr(tm, "track_meta", {}).keys())
        ids |= set(int(t) for t in getattr(tm, "synthetic_tracks", {}).keys())
        ids -= set(int(t) for t in getattr(tm, "track_deleted", set()))
        ids -= set(int(t) for t in getattr(tm, "track_replaced", set()))
        # 主车（EGO 实体）不是"交通参与者"，不能作为事故的候选参与者
        ego_id = getattr(tm, "ego_track_id", None)
        if ego_id is not None:
            ids.discard(int(ego_id))
        return sorted(ids)

    def build(self, frame_idx: int = 0, window: int = 10) -> "SceneGraph":
        self.frame_idx = int(frame_idx)
        self.window = int(window)
        tm = self.tm
        for tid in self.all_track_ids():
            frames = tm.get_track_frames(tid)
            if not frames:
                continue
            near = [f for f in frames if abs(f - frame_idx) <= window]
            if not near:
                continue          # 该帧附近没有此物体，不计入图
            center, vel, speed = self._motion_state(tid, frame_idx, near, window)
            if center is None:
                continue
            dims = list(tm.get_track_dimensions(tid))
            # 可用性：起始帧有位姿 + 尺寸合理 + 速度合理。
            # 生成器（_apply_initial_pair_layout/_ensure_participant）需要"起始帧位姿"作锚点，
            # 否则会静默不生效（参与者沿用原始稀疏噪声轨迹），必然过不了质量门。
            has_pose_at_start = tm.get_track_pose(tid, frame_idx) is not None
            try:
                dims_ok = all(abs(float(d)) > 0.05 for d in dims[:3])
            except (TypeError, ValueError, IndexError):
                dims_ok = False
            usable = bool(has_pose_at_start and dims_ok and speed <= 45.0)
            self._node_by_tid[tid] = len(self.nodes)
            self.nodes.append({
                "track_id": int(tid),
                "type": self._node_type(tid),
                "class": self._semantic_class(dims),
                "dimensions": dims,
                "center": [float(center[0]), float(center[1]), float(center[2])],
                "velocity": [float(vel[0]), float(vel[1]), float(vel[2])],
                "speed": float(speed),
                "yaw": float(math.atan2(vel[0], vel[2])) if speed > 1e-3 else self._pose_yaw(tid, frame_idx),
                "num_frames": len(frames),
                "synthetic": bool(tm.is_synthetic(tid)),
                "has_pose_at_start": bool(has_pose_at_start),
                "usable": usable,
            })

        n = len(self.nodes)
        for i in range(n):
            for j in range(i + 1, n):
                edge = self._relational_edge(i, j, frame_idx, window)
                if edge is not None:
                    self.edges.append(edge)

        # 按冲突关键度降序排列，方便直接取"最危险"的组合
        self.edges.sort(key=lambda e: -e["criticality"])
        return self

    # ------------------------------------------------------------------ 节点/运动
    def _node_type(self, tid: int) -> str:
        tm = self.tm
        if tm.is_synthetic(tid):
            s = tm.synthetic_tracks.get(int(tid)) or {}
            return str(s.get("type") or "合成物体")
        meta = tm.track_meta.get(int(tid)) or {}
        return str(meta.get("type") or "未知")

    @staticmethod
    def _semantic_class(dims: List[float]) -> str:
        try:
            vol = float(abs(dims[0]) * abs(dims[1]) * abs(dims[2]))
        except (IndexError, TypeError, ValueError):
            return "other"
        if vol < 2.0:
            return "pedestrian"      # 行人/锥桶/小目标
        if vol < 30.0:
            return "vehicle"
        return "large_vehicle"

    def _pose_yaw(self, tid: int, frame_idx: int) -> float:
        p = self.tm.get_track_pose(tid, frame_idx)
        if p is None:
            return 0.0
        return float(math.atan2(float(p[0, 2]), float(p[2, 2])))

    def _motion_state(self, tid: int, frame_idx: int, frames: List[int], window: int):
        """返回 (center, velocity, speed)：用 frame_idx 附近窗口内的轨迹中心做一阶拟合估计速度。"""
        tm = self.tm
        idxs = sorted(f for f in frames if abs(f - frame_idx) <= window)
        if not idxs:
            return None, np.zeros(3, dtype=np.float32), 0.0
        pts = []
        for f in idxs:
            p = tm.get_track_pose(tid, f)
            if p is not None:
                pts.append((f, np.asarray(p, dtype=np.float32)[:3, 3]))
        if not pts:
            return None, np.zeros(3, dtype=np.float32), 0.0
        if len(pts) == 1:
            return pts[0][1].astype(float), np.zeros(3, dtype=np.float32), 0.0
        t = np.array([(f - frame_idx) / self.fps for f, _ in pts], dtype=np.float64)
        xyz = np.array([c for _, c in pts], dtype=np.float64)
        # 一阶最小二乘：位置 = center + v*t
        vel = np.zeros(3, dtype=np.float64)
        center = xyz.mean(axis=0)
        if np.var(t) > 1e-12:
            for k in range(3):
                slope = float(np.polyfit(t, xyz[:, k], 1)[0])
                vel[k] = slope
        # 用 frame_idx 处（或最近帧）的中心作为节点位置
        near = min(pts, key=lambda x: abs(x[0] - frame_idx))[1]
        return near.astype(float), vel.astype(np.float32), float(np.linalg.norm(vel))

    # ------------------------------------------------------------------ 关系
    def _relational_edge(self, i: int, j: int, frame_idx: int, window: int) -> Optional[Dict[str, Any]]:
        a = self.nodes[i]
        b = self.nodes[j]
        ca = np.array(a["center"], dtype=np.float64)
        cb = np.array(b["center"], dtype=np.float64)
        va = np.array(a["velocity"], dtype=np.float64)
        vb = np.array(b["velocity"], dtype=np.float64)

        r = cb - ca
        r[1] = 0.0
        dist = float(np.linalg.norm(r))
        va2 = va.copy(); va2[1] = 0.0
        vb2 = vb.copy(); vb2[1] = 0.0
        rel_v = vb2 - va2
        rel_speed = float(np.linalg.norm(rel_v))
        sa, sb = float(np.linalg.norm(va2)), float(np.linalg.norm(vb2))

        # 接近速率 = 距离随时间的导数（负值 = 正在接近）
        approach_rate = 0.0
        if dist > 1e-6:
            approach_rate = float(-np.dot(r, rel_v) / dist)

        # 航向差 / 交角
        heading_a = math.atan2(va2[0], va2[2]) if sa > 1e-3 else math.atan2(r[0], r[2])
        heading_b = math.atan2(vb2[0], vb2[2]) if sb > 1e-3 else math.atan2(-r[0], -r[2])
        heading_diff = abs((heading_b - heading_a + math.pi) % (2 * math.pi) - math.pi)
        crossing_angle = math.degrees(heading_diff)

        # 关系类型
        rel = self._classify_relation(sa, sb, heading_diff, approach_rate)

        # TTC（匀速、把物体近似成地面圆盘的闭式解）
        r_contact = self._footprint_radius(a["dimensions"]) + self._footprint_radius(b["dimensions"])
        ttc, min_dist, is_collision_course = self._ttc_closed_form(r, rel_v, rel_speed, dist, r_contact)

        # 冲突关键度 [0,1]
        criticality = self._criticality(ttc, min_dist, dist, approach_rate, rel_speed, rel)

        return {
            "track_pair": [int(a["track_id"]), int(b["track_id"])],
            "relation": rel,
            "distance": float(dist),
            "relative_speed": float(rel_speed),
            "approach_rate": float(approach_rate),
            "heading_diff_deg": float(crossing_angle),
            "ttc_s": None if ttc is None else float(ttc),
            "min_distance_m": float(min_dist),
            "on_collision_course": bool(is_collision_course),
            "criticality": float(criticality),
            # GNN 可用的定长边特征向量（相对位置2D + 相对速度2D + 标量组）
            "features": [
                float(r[0]), float(r[2]),
                float(rel_v[0]), float(rel_v[2]),
                float(dist), float(rel_speed), float(approach_rate),
                float(crossing_angle) / 180.0,
                float(ttc) if ttc is not None else 10.0,
                float(min_dist),
            ],
        }

    @staticmethod
    def _classify_relation(sa: float, sb: float, heading_diff: float, approach_rate: float) -> str:
        if sa < 0.5 or sb < 0.5:
            return REL_STATIONARY
        if approach_rate < -0.05:
            pass  # 接近中
        elif approach_rate > 0.05:
            return REL_RECEDING
        deg = math.degrees(heading_diff)
        if deg < 30.0:
            return REL_FOLLOWING
        if deg < 45.0:
            return REL_ADJACENT
        if deg > 150.0:
            return REL_ONCOMING
        return REL_CROSSING

    @staticmethod
    def _footprint_radius(dims: List[float]) -> float:
        try:
            w, _, l = abs(float(dims[0])), abs(float(dims[1])), abs(float(dims[2]))
        except (IndexError, ValueError, TypeError):
            return 1.0
        return 0.5 * math.sqrt(w * w + l * l)

    @staticmethod
    def _ttc_closed_form(r, rel_v, rel_speed, dist, contact):
        if rel_speed < 1e-3:
            return (None, dist, False)
        # 最近点时间 t* = -r·v / |v|²
        t_star = -float(np.dot(r, rel_v)) / (rel_speed ** 2)
        closest = float(np.linalg.norm(r + rel_v * t_star))
        # 到达接触半径的时间（解 |r + v t| = contact）
        b = float(np.dot(r, rel_v))
        a = rel_speed ** 2
        disc = b * b - a * (float(np.dot(r, r)) - contact * contact)
        ttc = None
        if disc >= 0:
            sq = math.sqrt(disc)
            t1 = (-b - sq) / a
            t2 = (-b + sq) / a
            candidates = [t for t in (t1, t2) if t >= 0]
            if candidates:
                ttc = min(candidates)
        on_course = ttc is not None and 0.0 <= ttc <= 5.0 and b < 0.0
        return (ttc, closest, on_course)

    @staticmethod
    def _criticality(ttc, min_dist, dist, approach_rate, rel_speed, rel):
        score = 0.0
        # TTC 越短越危险
        if ttc is not None and ttc < 3.0:
            score += (1.0 - ttc / 3.0) * 0.6
        # 最近距离越近越危险
        score += max(0.0, 1.0 - min_dist / 8.0) * 0.25
        # 正在接近且相对速度快 → 更危险
        if approach_rate < -0.1:
            score += min(1.0, -approach_rate / 10.0) * 0.1
        score += min(1.0, rel_speed / 20.0) * 0.05
        if rel == REL_STATIONARY:
            score *= 0.5  # 静止目标危险度打折
        return float(min(1.0, score))

    # ------------------------------------------------------------------ 提案
    # 每种事故类型 → 角色键顺序（对应 _pairs_for 返回的 0/1 语义）
    ROLE_KEYS = {
        "rear-end": ("attacker", "victim"),
        "head-on": ("attacker", "victim"),
        "intersection-tbone": ("attacker", "victim"),
        "lane-change-cutin": ("cutter", "target"),
        "pedestrian-crossing": ("pedestrian", "vehicle"),
        "hard-brake": ("braker",),
    }

    # 该类型"必须指定"的主角色（其余角色可自动合成）
    PRIMARY_ROLE = {
        "rear-end": "attacker",
        "head-on": "attacker",
        "intersection-tbone": "attacker",
        "lane-change-cutin": "cutter",
        "pedestrian-crossing": "vehicle",
        "hard-brake": "braker",
        "cut-out-reveal": "blocker",
        "chain-reaction-rear-end": "rear",
        "cutin-brake-pileup": "cutter",
        "occluded-pedestrian-pileup": "vehicle",
    }

    # 这些角色必须是"车辆"类物体（不能拿行人/锥桶/杆当车）
    VEHICLE_ROLES = {"attacker", "victim", "cutter", "target", "blocker", "obstacle",
                     "lead", "middle", "rear", "follower", "braker", "vehicle"}

    def _roles_class_ok(self, roles: Dict[str, int]) -> bool:
        for rk, tid in roles.items():
            idx = self._node_by_tid.get(int(tid))
            if idx is None:
                return False
            nd = self.nodes[idx]
            if rk in self.VEHICLE_ROLES and nd.get("class") == "pedestrian":
                return False
        return True

    def usable_node_ids(self):
        return {int(n["track_id"]) for n in self.nodes if n.get("usable")}

    def propose_anchors(self, scenario_type: str, max_candidates: int = 8):
        """退回方案：只给"主角色"一个可用物体，其余参与者由生成器自动合成。

        当关系图里没有"两个都在起始帧有位姿"的可用组合时，用这种方式仍能生成
        （自动合成出来的参与者轨迹干净，质量门更容易通过）。
        """
        role = self.PRIMARY_ROLE.get(scenario_type)
        if role is None:
            return []
        out = []
        for nd in sorted(self.nodes, key=lambda x: -float(x.get("speed") or 0.0)):
            if not nd.get("usable"):
                continue
            cand = {role: int(nd["track_id"])}
            if not self._roles_class_ok(cand):
                continue
            out.append(cand)
            if len(out) >= max_candidates:
                break
        return out

    def propose_participants(self, scenario_type: str, max_candidates: int = 8) -> List[Dict[str, int]]:
        """为某事故类型挑选候选参与者，返回 `{role_key: track_id}` 列表（按关键度降序）。

        只使用 `usable`（起始帧有位姿 + 尺寸/速度合理）的物体；若没有可用的**成对**组合，
        自动退回 `propose_anchors`（1 个可用锚点 + 其余自动合成），保证批量仍有产出。
        """
        if scenario_type == "rear-end":
            raw = self._pairs_for(REL_FOLLOWING, order="front_back")   # 0=front, 1=back
            raw = [{t: 1 if v == 0 else 0 for t, v in d.items()} for d in raw]  # 0=attacker(后车), 1=victim(前车)
        elif scenario_type == "head-on":
            raw = self._pairs_for(REL_ONCOMING, order="speed_desc")
        elif scenario_type == "intersection-tbone":
            raw = self._pairs_for(REL_CROSSING, order="crosser_crossed")
        elif scenario_type == "lane-change-cutin":
            # 0=cutter(后车), 1=target(前车)
            raw = self._pairs_for(REL_FOLLOWING, order="front_back")
            raw = [{t: 1 if v == 0 else 0 for t, v in d.items()} for d in raw]
            raw += [{t: 1 if v == 0 else 0 for t, v in d.items()}
                    for d in self._pairs_for(REL_ADJACENT, order="front_back")]
        elif scenario_type == "pedestrian-crossing":
            raw = self._pairs_for(REL_CROSSING, order="ped_veh")       # 0=ped, 1=veh
        elif scenario_type == "hard-brake":
            raw = []
            for nd in sorted(self.nodes, key=lambda x: -x["speed"]):
                if nd.get("usable") and nd["speed"] > 2.0:
                    raw.append({0: int(nd["track_id"])})
        else:
            raw = []

        keys = self.ROLE_KEYS.get(scenario_type, ())
        out: List[Dict[str, int]] = []
        for d in raw:
            roles = {}
            for tid, idx in d.items():
                if idx < len(keys):
                    roles[keys[idx]] = int(tid)
            if roles:
                out.append(roles)
        # 去重
        seen, dedup = set(), []
        for roles in out:
            key = tuple(sorted(roles.items()))
            if key not in seen:
                seen.add(key)
                dedup.append(roles)
        # 车辆类角色不能是行人/锥桶
        dedup = [r for r in dedup if self._roles_class_ok(r)]
        if max_candidates and len(dedup) > max_candidates:
            dedup = dedup[:max_candidates]
        # 没有任何可用成对组合 → 退回"单锚点 + 自动合成"
        if not dedup:
            dedup = self.propose_anchors(scenario_type, max_candidates=max_candidates)
        return dedup

    def _pairs_for(self, relation: str, order: str) -> List[Dict[int, int]]:
        usable = self.usable_node_ids()
        pairs = [e for e in self.edges if e["relation"] == relation]
        result = []
        for e in pairs:
            ta, tb = [int(x) for x in e["track_pair"]]
            if ta not in usable or tb not in usable:
                continue          # 只在"起始帧有位姿且数据合理"的物体之间配对
            na = self.nodes[self._node_by_tid[ta]]
            nb = self.nodes[self._node_by_tid[tb]]
            ca = np.array(na["center"], dtype=np.float64)
            cb = np.array(nb["center"], dtype=np.float64)
            head = np.array(na["velocity"], dtype=np.float64); head[1] = 0.0
            hn = float(np.linalg.norm(head))
            if hn < 1e-3:
                continue
            proj = float(np.dot(cb - ca, head / hn))
            if order == "front_back":
                front, back = (ta, tb) if proj > 0 else (tb, ta)
                result.append({front: 0, back: 1})
            elif order == "speed_desc":
                first, second = (ta, tb) if na["speed"] >= nb["speed"] else (tb, ta)
                result.append({first: 0, second: 1})
            elif order == "crosser_crossed":
                # 谁更"横穿"（速度方向与连线更垂直）谁是 crosser
                cross_a = abs(math.sin(math.atan2(head[0], head[2]) - math.atan2(cb[0]-ca[0], cb[2]-ca[2])))
                crosser, crossed = (ta, tb) if cross_a >= 0.5 else (tb, ta)
                result.append({crosser: 0, crossed: 1})
            elif order == "ped_veh":
                # 0=pedestrian, 1=vehicle
                a_ped = na["class"] == "pedestrian"
                b_ped = nb["class"] == "pedestrian"
                if a_ped and not b_ped:
                    result.append({ta: 0, tb: 1})
                elif b_ped and not a_ped:
                    result.append({tb: 0, ta: 1})
                else:
                    result.append({ta: 0, tb: 1})
        seen, dedup = set(), []
        for d in result:
            key = tuple(sorted(d.items()))
            if key not in seen:
                seen.add(key)
                dedup.append(d)
        return dedup

    # ------------------------------------------------------------------ 序列化
    def to_dict(self) -> Dict[str, Any]:
        return {
            "num_nodes": len(self.nodes),
            "num_edges": len(self.edges),
            "nodes": self.nodes,
            "edges": self.edges,
        }
