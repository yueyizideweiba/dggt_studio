"""多阶段交通事故交互结构：预警区 / 危险区 / 发生区（+ 遮挡 / 撞击 / 安全）。

论文核心："corner case 的生成由参数控制"。本模块把**一次事故的演化过程**进一步参数化为
几段可判定的区域，给自动驾驶一个可执行的、分级的响应参考：

  安全   (safe)       —— 当前预测轨迹本就无冲突，正常行驶
  预警区 (warning)    —— 舒适制动(≤a_comf)即可避免，系统应"监测 + 准备"
  危险区 (danger)     —— 必须紧急制动(a_comf < a ≤ a_emerg)，系统应"立即干预"
  发生区 (occurrence) —— 即使最大制动也来不及（不可避免碰撞态），只能"最小化损失"
  撞击   (impact)     —— 已经发生接触，进入"碰撞后稳定 / 最小风险停车"处置
  遮挡   (occluded)   —— 参与者尚未可见（鬼探头），无有效感知

核心判据是**所需最小减速度 a\***：在反应主体当前的相对运动（速度、方向、物体尺寸）下，
要使它在反应时间 t_r 之后开始制动、且在预测时域内不与对方包围盒(OBB)接触，所需的
最小恒定减速度。a\* 把速度/方向/尺寸/反应时间**一次折算成一个可比较的标量**，再和两级
能力阈值（舒适 a_comf、紧急 a_emerg）比较，就自然切出三个区域：

    a* = 0                   → 安全（预测轨迹本就无接触）
    0      < a* ≤ a_comf     → 预警区
    a_comf < a* ≤ a_emerg    → 危险区
    a_emerg < a*             → 发生区（制动不可解 → 不可避免碰撞态 ICS）

这与 ISO 15623（前向碰撞预警/AEB）、RSS（责任敏感安全）以及不可避免碰撞状态 ICS
（Fraichard & Asama 2004）一致，并且把原先 `compute_critical_frame` 的"单一最晚反应帧"
推广成了**连续的三段式时程**。

两点关键设计：
  1. "撞击"帧**直接采用生成器已给出的 `collision_frame`**（仿真在分离前判定的），而不是从
     最终姿态重推——因为防穿模 `separate_pair` 会把碰撞后的两车推到"零间隙"，此时 SAT 判为
     "分离"，重推会漏掉撞击帧。
  2. 除"接触"外，还考虑"危险接近"（near-miss）：若预测时域内无接触、但两心最近距离
     小于 `d_near_miss_m`，也记为"预警"（有意义的近距离擦过），否则这类"回避/险情"场景
     会全部被判成 safe。

"corner case 什么时候发生、发生或不发生都由参数控制"在这里的体现：
  - 反应时间 t_r 越大（分心/新手）→ 危险区与发生区越早到来；
  - 舒适阈值 a_comf / 紧急阈值 a_emerg（路面附着）→ 决定预警/危险/发生的分界；
  - 相对速度（对向=速度之和、同向=速度之差、横穿=矢量）→ 决定三区宽度；
  - 物体尺寸（OBB）→ 决定接触判定与 a* 的边界；
  - 遮挡/鬼探头（occlusion）→ 参与者不可见期间根本没有预警区，一出现就可能直接落进
    危险区/发生区（"预警区被压缩或跳过"的典型 corner case）。
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from dggt.scene_edit.collision_physics import check_collision

# --------------------------------------------------------------------------- 区域枚举
ZONE_SAFE = "safe"
ZONE_WARNING = "warning"
ZONE_DANGER = "danger"
ZONE_OCCURRENCE = "occurrence"
ZONE_IMPACT = "impact"
ZONE_OCCLUDED = "occluded"
ZONE_NA = "n/a"

# 严重度排序（越大越紧急），用于多参与者聚合时取"最危险"的那个
_ZONE_ORDER = {
    ZONE_NA: -2,
    ZONE_OCCLUDED: -1,
    ZONE_SAFE: 0,
    ZONE_WARNING: 1,
    ZONE_DANGER: 2,
    ZONE_OCCURRENCE: 3,
    ZONE_IMPACT: 4,
}

# 各区域对应的自动驾驶响应建议（论文里可逐条引用）
_ZONE_ADVICE = {
    ZONE_SAFE: {
        "action": "normal", "level": 0,
        "text": "正常行驶，保持观察（预测时域内无冲突）",
    },
    ZONE_WARNING: {
        "action": "monitor_brake", "level": 1,
        "text": "预警区：保持监测，准备以舒适减速度(≤a_comf)减速，并评估变道空间",
    },
    ZONE_DANGER: {
        "action": "emergency_brake", "level": 2,
        "text": "危险区：立即紧急制动(AEB，≤a_emerg)或按规则变道避让",
    },
    ZONE_OCCURRENCE: {
        "action": "minimize_damage", "level": 3,
        "text": "发生区：碰撞不可避免，全力制动以减小相对速度，并调整接触角度以最小化损失",
    },
    ZONE_IMPACT: {
        "action": "post_impact", "level": 4,
        "text": "已发生接触：稳住方向、持续制动、开启双闪、执行最小风险停车",
    },
    ZONE_OCCLUDED: {
        "action": "occluded", "level": 0,
        "text": "遮挡（鬼探头）：参与者不可见，无有效感知，应降低车速/依赖遮挡推理",
    },
    ZONE_NA: {
        "action": "unknown", "level": -1,
        "text": "位姿缺失，无法判定",
    },
}

# 默认参数（均可在 sampling 里覆盖）
PHASE_PARAMS_DEFAULT = {
    "t_reaction_s": 0.7,          # 反应时间（感知+决策+执行延迟）
    "a_comfort_mps2": 3.0,        # 舒适减速度上限（预警/危险分界）
    "a_emergency_mps2": 8.0,      # 紧急减速度上限（危险/发生分界）
    "horizon_s": 4.0,             # 预测时域
    "dt_pred_s": 0.1,             # 前向仿真步长
    "a_scan_step_mps2": 0.25,     # a* 搜索粒度
    "d_near_miss_m": 3.0,         # 无接触但"近距离擦过"的最近距离阈值
    "post_impact_frames": 2,      # 撞击帧之后的"碰撞后稳定"帧数
}

# 场景级的相态画像：不同事故类型的反应特性不同（行人横穿反应更慢、对向需要更狠的制动等）
SCENARIO_PHASE_PROFILES: Dict[str, Dict[str, float]] = {
    "pedestrian-crossing": {"t_reaction_s": 0.9, "a_emergency_mps2": 9.0, "d_near_miss_m": 3.5},
    "occluded-pedestrian-pileup": {"t_reaction_s": 1.0, "d_near_miss_m": 3.5},
    "head-on": {"a_emergency_mps2": 9.0, "a_comfort_mps2": 3.5},
    "hard-brake": {"t_reaction_s": 1.2},
    "chain-reaction-rear-end": {"t_reaction_s": 0.9},
    "cutin-brake-pileup": {"t_reaction_s": 0.9},
}


# --------------------------------------------------------------------------- 工具
def _dims(tm, tid: int) -> List[float]:
    """取物体尺寸 [宽, 高, 长]；取不到给一个合理的默认轿车尺寸。"""
    try:
        d = tm.get_track_dimensions(int(tid))
        if d and all(abs(float(x)) > 1e-3 for x in d[:3]):
            return [float(d[0]), float(d[1]), float(d[2])]
    except Exception:  # noqa: BLE001
        pass
    return [2.0, 1.7, 4.6]


def _radius(dims: List[float]) -> float:
    """OBB 在 XZ 平面的"半对角"（用于中心距粗筛的保守外接圆半径）。"""
    return 0.5 * math.hypot(float(dims[0]), float(dims[2]))


def _contact(pa: np.ndarray, dims_a: List[float], pb: np.ndarray, dims_b: List[float]) -> bool:
    """两物体当前是否接触（先中心距粗筛，再 SAT 精判）。"""
    ca, cb = pa[:3, 3], pb[:3, 3]
    if float(np.linalg.norm(ca - cb)) > _radius(dims_a) + _radius(dims_b) + 1e-3:
        return False
    hit, _ = check_collision(np.asarray(pa, dtype=np.float32), dims_a,
                             np.asarray(pb, dtype=np.float32), dims_b)
    return bool(hit)


def _motion_series(tm, tid: int, frames: List[int], fps: float):
    """某 track 在窗口内的逐帧 (中心, 速度, 加速度)，用中心差分估计。"""
    dt = 1.0 / float(fps)
    centers: Dict[int, np.ndarray] = {}
    for f in frames:
        p = tm.get_track_pose(int(tid), int(f))
        if p is not None:
            centers[int(f)] = np.asarray(p, dtype=np.float32)[:3, 3].copy()
    fs = sorted(centers)
    vel: Dict[int, np.ndarray] = {}
    acc: Dict[int, np.ndarray] = {}
    for i, f in enumerate(fs):
        if i > 0 and i < len(fs) - 1:
            v = (centers[fs[i + 1]] - centers[fs[i - 1]]) / (2.0 * dt)
        elif i < len(fs) - 1:
            v = (centers[fs[i + 1]] - centers[f]) / dt
        elif i > 0:
            v = (centers[f] - centers[fs[i - 1]]) / dt
        else:
            v = np.zeros(3, dtype=np.float32)
        vel[f] = v.astype(np.float64)
    for i, f in enumerate(fs):
        acc[f] = ((vel[f] - vel[fs[i - 1]]) / dt).astype(np.float64) if i > 0 else np.zeros(3, dtype=np.float64)
    return centers, vel, acc


def _predict_contact(pa: np.ndarray, va: np.ndarray, dims_a: List[float],
                     pb: np.ndarray, vb: np.ndarray, dims_b: List[float],
                     horizon: float, dt_pred: float) -> Tuple[Optional[float], float]:
    """匀速外推下，两物体的最早接触时间与最近中心距。

    Returns:
        (ttc, min_center_dist)：ttc 为预测时域内最早接触时间（无接触则 None），
        min_center_dist 为该时域内两心的最近距离。
    """
    pa = np.array(pa, dtype=np.float64).copy()
    pb = np.array(pb, dtype=np.float64).copy()
    va = np.asarray(va, dtype=np.float64)
    vb = np.asarray(vb, dtype=np.float64)
    t = 0.0
    min_d = float(np.linalg.norm(pa[:3, 3] - pb[:3, 3]))
    while t <= horizon:
        d = float(np.linalg.norm(pa[:3, 3] - pb[:3, 3]))
        min_d = min(min_d, d)
        if _contact(pa, dims_a, pb, dims_b):
            return t, min_d
        pa[:3, 3] += va * dt_pred
        pb[:3, 3] += vb * dt_pred
        t += dt_pred
    return None, min_d


def _contact_with_braking(pa: np.ndarray, va: np.ndarray, dims_a: List[float],
                          pb: np.ndarray, vb: np.ndarray, dims_b: List[float],
                          a: float, t_reaction: float, horizon: float, dt_pred: float) -> bool:
    """主车反应时间 t_reaction 后以恒定减速度 a 制动（沿当前航向），对方匀速，
    预测时域内是否仍会接触。制动只沿当前航向（AEB 原语），转向避让作为扩展说明。
    """
    pa = np.array(pa, dtype=np.float64).copy()
    pb = np.array(pb, dtype=np.float64).copy()
    v = np.asarray(va, dtype=np.float64).copy()
    vb = np.asarray(vb, dtype=np.float64)
    t = 0.0
    while t <= horizon:
        if _contact(pa, dims_a, pb, dims_b):
            return True
        speed = float(np.linalg.norm(v))
        if t >= t_reaction and speed > 1e-6:
            heading = v / speed
            speed = max(0.0, speed - a * dt_pred)
            v = heading * speed
        pa[:3, 3] += v * dt_pred
        pb[:3, 3] += vb * dt_pred
        t += dt_pred
    return False


def min_required_deceleration(pa: np.ndarray, va: np.ndarray, dims_a: List[float],
                              pb: np.ndarray, vb: np.ndarray, dims_b: List[float],
                              params: Dict[str, Any]) -> float:
    """所需最小减速度 a*：使主车（反应时间后）制动仍能在预测时域内避免接触的最小 a。

    返回 0.0 表示当前预测轨迹本就无接触（无需制动）；返回 inf 表示直到 a_emerg 仍无法
    避免（不可避免碰撞态）。
    """
    a_emerg = float(params["a_emergency_mps2"])
    step = float(params["a_scan_step_mps2"])
    t_r = float(params["t_reaction_s"])
    horizon = float(params["horizon_s"])
    dt_pred = float(params["dt_pred_s"])
    if not _contact_with_braking(pa, va, dims_a, pb, vb, dims_b, 0.0, t_r, horizon, dt_pred):
        return 0.0
    a = step
    while a <= a_emerg + 1e-6:
        if not _contact_with_braking(pa, va, dims_a, pb, vb, dims_b, a, t_r, horizon, dt_pred):
            return a
        a += step
    return float("inf")


def classify_zone(a_star: float, params: Dict[str, Any]) -> str:
    if a_star == 0.0:
        return ZONE_SAFE
    if a_star <= float(params["a_comfort_mps2"]):
        return ZONE_WARNING
    if a_star <= float(params["a_emergency_mps2"]):
        return ZONE_DANGER
    return ZONE_OCCURRENCE


# --------------------------------------------------------------------------- 主入口
def compute_phase_timeline(tm, actor: int, others: List[int], frames: List[int], fps: float,
                           params: Optional[Dict[str, Any]] = None,
                           occlusion: Optional[Dict[int, int]] = None,
                           collision_frame: Optional[int] = None) -> Dict[str, Any]:
    """计算反应主体(actor)相对其它参与者(others)的逐帧相态时程。

    Args:
        tm: TrackManager
        actor: 反应主体（决策主体，通常是肇事车/主车）track_id
        others: 其它参与者 track_id 列表
        frames: 帧区间
        fps: 帧率
        params: 覆盖 PHASE_PARAMS_DEFAULT 的参数
        occlusion: {other_id: first_visible_frame}，该帧之前该参与者对 actor 不可见（鬼探头）
        collision_frame: 生成器已知的撞击帧（用于把"撞击"标出来，见模块 docstring）

    Returns:
        {actor, others, fps, params, timeline[], boundaries, summary}
    """
    params = dict(PHASE_PARAMS_DEFAULT)
    if params is not None:
        for k, v in params.items():
            if v is not None:
                params[k] = v
    others = [int(o) for o in others if int(o) != int(actor)]
    occlusion = {int(k): int(v) for k, v in (occlusion or {}).items()}
    cfr = None if collision_frame is None else int(collision_frame)
    post = int(params["post_impact_frames"])
    d_near = float(params["d_near_miss_m"])

    dims_actor = _dims(tm, int(actor))
    _, vel_actor, acc_actor = _motion_series(tm, int(actor), frames, fps)
    mot = {int(o): _motion_series(tm, int(o), frames, fps) for o in others}
    dims = {int(o): _dims(tm, int(o)) for o in others}

    timeline: List[Dict[str, Any]] = []
    for f in frames:
        pa = tm.get_track_pose(int(actor), int(f))
        if pa is None:
            timeline.append({"frame": int(f), "zone": ZONE_NA, "a_required": None,
                             "advice": _ZONE_ADVICE[ZONE_NA], "advice_level": -1})
            continue
        pa = np.asarray(pa, dtype=np.float64)
        va = vel_actor.get(int(f), np.zeros(3, dtype=np.float64))
        aa = acc_actor.get(int(f), np.zeros(3, dtype=np.float64))

        # 撞击帧及随后短暂"碰撞后稳定"窗口 → 直接标 impact（不再跑 a* 预测）
        if cfr is not None and cfr <= int(f) <= cfr + post:
            timeline.append({
                "frame": int(f), "zone": ZONE_IMPACT, "a_required": None, "ttc_s": 0.0,
                "relative_speed_mps": None, "distance_m": None, "other": others[0] if others else None,
                "actor_speed_mps": round(float(np.linalg.norm(va)), 3),
                "actor_accel_mps2": round(float(np.linalg.norm(aa)), 3),
                "advice": _ZONE_ADVICE[ZONE_IMPACT], "advice_level": 4,
            })
            continue
        # 撞击事件结束之后：相态故事已讲完，不再用匀速预测（撞击后的速度噪声会产生
        # "再次接近"的假 occurrence），直接记为安全/事件已结束。
        if cfr is not None and int(f) > cfr + post:
            timeline.append({
                "frame": int(f), "zone": ZONE_SAFE, "a_required": 0.0, "ttc_s": None,
                "relative_speed_mps": None, "distance_m": None, "other": None,
                "actor_speed_mps": round(float(np.linalg.norm(va)), 3),
                "actor_accel_mps2": round(float(np.linalg.norm(aa)), 3),
                "advice": _ZONE_ADVICE[ZONE_SAFE], "advice_level": 0, "event_concluded": True,
            })
            continue

        best = {"zone": ZONE_SAFE, "a_required": 0.0, "ttc": None, "other": None,
                "relative_speed": 0.0, "distance": None, "near_miss": False}
        best_order = _ZONE_ORDER[ZONE_SAFE]
        any_occluded = False
        for other in others:
            pb = tm.get_track_pose(other, int(f))
            if pb is None:
                continue
            if int(f) < occlusion.get(other, -1):
                any_occluded = True
                if _ZONE_ORDER[ZONE_OCCLUDED] > best_order:
                    best = {"zone": ZONE_OCCLUDED, "a_required": None, "ttc": None,
                            "other": int(other), "relative_speed": None, "distance": None,
                            "near_miss": False}
                    best_order = _ZONE_ORDER[ZONE_OCCLUDED]
                continue
            pb = np.asarray(pb, dtype=np.float64)
            vb = mot[other][1].get(int(f), np.zeros(3, dtype=np.float64))
            dims_b = dims[other]
            dist = float(np.linalg.norm(pa[:3, 3] - pb[:3, 3]))
            rel_spd = float(np.linalg.norm(va - vb))
            ttc, min_d = _predict_contact(pa, va, dims_actor, pb, vb, dims_b,
                                          float(params["horizon_s"]), float(params["dt_pred_s"]))
            if ttc is not None:
                a_star = min_required_deceleration(pa, va, dims_actor, pb, vb, dims_b, params)
                zone = classify_zone(a_star, params)
                cand = {"zone": zone,
                        "a_required": (None if a_star == float("inf") else round(float(a_star), 3)),
                        "ttc": round(float(ttc), 3), "other": int(other),
                        "relative_speed": round(rel_spd, 3), "distance": round(dist, 3),
                        "near_miss": False}
            elif min_d < d_near:
                # 无接触但预测最近距离很近 → 近距离擦过（near-miss），记为预警
                cand = {"zone": ZONE_WARNING, "a_required": 0.0, "ttc": None, "other": int(other),
                        "relative_speed": round(rel_spd, 3), "distance": round(min_d, 3),
                        "near_miss": True}
            else:
                cand = {"zone": ZONE_SAFE, "a_required": 0.0, "ttc": None, "other": int(other),
                        "relative_speed": round(rel_spd, 3), "distance": round(min_d, 3),
                        "near_miss": False}
            if _ZONE_ORDER[cand["zone"]] > best_order:
                best = cand
                best_order = _ZONE_ORDER[cand["zone"]]
        advice = _ZONE_ADVICE.get(best["zone"], _ZONE_ADVICE[ZONE_NA])
        timeline.append({
            "frame": int(f), "zone": best["zone"],
            "a_required": best["a_required"], "ttc_s": best["ttc"],
            "relative_speed_mps": best["relative_speed"], "distance_m": best["distance"],
            "other": best["other"], "near_miss": bool(best.get("near_miss")),
            "actor_speed_mps": round(float(np.linalg.norm(va)), 3),
            "actor_accel_mps2": round(float(np.linalg.norm(aa)), 3),
            "advice": advice, "advice_level": int(advice.get("level", -1)),
        })

    return _finish_timeline(tm, actor, others, frames, fps, params, timeline)


def _finish_timeline(tm, actor, others, frames, fps, params, timeline) -> Dict[str, Any]:
    """汇总边界帧、各区持续时间，并生成一句"论文可用"的相态故事。"""
    def first_frame(zone):
        for e in timeline:
            if e["zone"] == zone:
                return e["frame"]
        return None

    def last_frame(zone):
        last = None
        for e in timeline:
            if e["zone"] == zone:
                last = e["frame"]
        return last

    def duration(zone):
        lo, hi = first_frame(zone), last_frame(zone)
        if lo is None or hi is None:
            return 0.0
        return max(0.0, (hi - lo) / float(fps))

    w_f = first_frame(ZONE_WARNING)
    d_f = first_frame(ZONE_DANGER)
    o_f = first_frame(ZONE_OCCURRENCE)
    i_f = first_frame(ZONE_IMPACT)
    oc_f = first_frame(ZONE_OCCLUDED)

    event = "collision" if i_f is not None else (
        "near_miss" if o_f is not None else (
            "avoid" if (d_f is not None or w_f is not None) else "none"))

    dw, dd, do = duration(ZONE_WARNING), duration(ZONE_DANGER), duration(ZONE_OCCURRENCE)
    compressed_warning = (dw <= 0.3) and (dd > 0.0 or do > 0.0 or i_f is not None)
    skipped_warning = (dw == 0.0) and (dd > 0.0 or do > 0.0 or i_f is not None)

    story = _make_story(w_f, d_f, o_f, i_f, oc_f, dw, dd, do, compressed_warning, skipped_warning, event)

    summary = {
        "first_warning_frame": w_f,
        "first_danger_frame": d_f,
        "first_occurrence_frame": o_f,
        "collision_frame": i_f,
        "first_occluded_frame": oc_f,
        "zone_durations_s": {"warning": round(dw, 2), "danger": round(dd, 2),
                             "occurrence": round(do, 2)},
        "compressed_warning": bool(compressed_warning),
        "skipped_warning": bool(skipped_warning),
        "event": event,
        "story": story,
    }
    return {
        "actor": int(actor),
        "others": [int(o) for o in others],
        "fps": float(fps),
        "params": params,
        "timeline": timeline,
        "boundaries": {
            "warning_frame": w_f, "danger_frame": d_f,
            "occurrence_frame": o_f, "collision_frame": i_f,
        },
        "summary": summary,
    }


def _make_story(w_f, d_f, o_f, i_f, oc_f, dw, dd, do, compressed, skipped, event) -> str:
    if event == "none":
        return "全程安全，预测时域内未出现冲突或危险接近，无需干预。"
    # 按时间顺序叙述（有的场景会"先危险后缓解"，如闪开前车：发生区 → 预警）
    seg = []
    if oc_f is not None:
        seg.append((oc_f, "参与者直到第 %d 帧才可见（遮挡/鬼探头）" % oc_f))
    if w_f is not None:
        seg.append((w_f, "预警区 第 %d 帧起，持续 %.1fs" % (w_f, dw)))
    if d_f is not None:
        seg.append((d_f, "危险区 第 %d 帧起，持续 %.1fs" % (d_f, dd)))
    if o_f is not None:
        seg.append((o_f, "发生区 第 %d 帧起，持续 %.1fs" % (o_f, do)))
    if i_f is not None:
        seg.append((i_f, "第 %d 帧发生接触" % i_f))
    seg.sort(key=lambda x: x[0])
    if not seg:
        return "无可读相态（参与者轨迹缺失）。"
    s = "，".join(txt for _, txt in seg) + "。"
    if skipped:
        s += " 预警区被完全跳过，直接进入危险/发生区——典型 corner case。"
    elif compressed:
        s += " 预警区被压缩到 %.1fs 以内——典型 corner case。" % dw
    return s
