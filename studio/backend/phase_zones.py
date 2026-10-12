"""多阶段交通事故交互结构：预警区 / 危险区 / 发生区（多因素判据模型）。

论文核心："corner case 的生成由参数控制"，事故的演化过程也应由**多个可解释的因素**共同判定，
而不是单一标量。本模块把每帧的交互状态分解为若干**物理因素**，再由"能力包络 + 能量严重度"
综合成一个**二维风险**（紧急度 × 严重度），映射到可执行的区域与反应建议：

  安全 safe / 预警区 warning / 危险区 danger / 发生区 occurrence / 撞击 impact / 遮挡 occluded

--------------------------------------------------------------
一、判据因素（每个因素都逐帧计算、逐帧可解释、逐帧可追溯）
--------------------------------------------------------------
1. 纵向制动所需减速度 a_brake  —— 反应时间 t_r 后以恒定减速度制动，能在预测时域内不与对方
   OBB 接触所需的最小 a。这是"刹不刹得住"的纵向能力判据。数值用**二分法精确求根**；
   一维闭合（对向/追尾）另有解析上界 a_stop = v_rel²/(2·(gap - v_rel·t_r))（brake-to-stop）。
2. 横向转向所需加速度 a_lat     —— 要在 (TTC - t_r) 秒内横向避让 OBB 净空 W_clr（两车半宽之和
   + 余量，而非固定一个车道宽），所需的最小横向加速度 a_lat = 2·W_clr / (TTC - t_r)²。
   这是"转不转得开"的横向能力判据。
3. 撞击严重度（能量）Δv + 接触几何 —— 预测接触时的相对速度 |v_a - v_b|，与接触几何
   （对向 head-on / 侧面 side / 追尾 rear / 静止 stationary）一起决定后果严重度。
4. 物体类型（VRU 弱势道路使用者）—— 行人类体积的参与者给更大的安全余量与更保守的严重度。
5. 路面附着 μ —— 决定纵向制动上限 cap_brake=μ·g 与横向转向上限 cap_lat≈0.9·μ·g（湿滑/冰雪
   直接压缩能力包络 → 危险区与发生区变宽）。
6. TTC 与反应裕度 —— TTC 是时间紧迫度；反应裕度 = TTC - t_r 是"反应之后还剩多少时间"。
7. 机动余量 μ —— μ_brake = 1 - a_brake/cap_brake、μ_lat = 1 - a_lat/cap_lat（1=零负担，0=用满能力，
   <0=能力不足）。最优机动余量 μ* = max(μ_brake, μ_lat) 是"还剩多少能力"的连续、可比较指标。

--------------------------------------------------------------
二、能力包络 → 区域（紧急度维度）
--------------------------------------------------------------
对每个选项（制动 / 转向）评一个"能力档位"：
    制动：a_brake=0→无冲突；≤a_comfort→舒适；≤cap_brake→紧急；>cap_brake→不足
    转向：a_lat ≤ a_lat_comfort→舒适；≤cap_lat→紧急；>cap_lat→不足
主车取**最好的那个选项**（能力包络的边界）：
    最优=舒适 → 预警区；最优=紧急 → 危险区；两都不足 → 发生区（不可避免碰撞态 ICS）
（若预测本就无接触 → safe；若已接触 → impact。）

这与 ISO 15623、RSS（"合理响应"含制动与转向）、不可避免碰撞态 ICS（Fraichard & Asama 2004）
一致，且比"单一最晚反应帧"更完整：它同时回答"该不该动、能动哪种、还来不来得及"。

--------------------------------------------------------------
三、严重度维度 → 风险等级（二维风险）
--------------------------------------------------------------
risk_level = 紧急度(区域) × 严重度(Δv·几何·类型)：
    occurrence + 高严重度 → critical；occurrence → severe；danger → high；
    warning → elevated；safe → low。
区域回答"多紧急"，严重度回答"多严重"，两者共同决定响应优先级与"最小化损失"的具体动作。

--------------------------------------------------------------
四、参数（全部可调，全部决定"corner case 是否/何时发生"）
--------------------------------------------------------------
t_reaction_s / a_comfort_mps2 / a_emergency_mps2 / mu / lane_width_m / a_lat_comfort_mps2 /
horizon_s / dt_pred_s / a_scan_step_mps2 / d_near_miss_m / vru_margin_m / post_impact_frames /
severity_delta_v_high / severity_delta_v_med
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

_ZONE_ORDER = {
    ZONE_NA: -2, ZONE_OCCLUDED: -1, ZONE_SAFE: 0, ZONE_WARNING: 1,
    ZONE_DANGER: 2, ZONE_OCCURRENCE: 3, ZONE_IMPACT: 4,
}

_ZONE_ADVICE = {
    ZONE_SAFE: {"action": "normal", "level": 0,
                "text": "正常行驶，保持观察（预测时域内无冲突）"},
    ZONE_WARNING: {"action": "monitor_brake", "level": 1,
                   "text": "预警区：保持监测，准备以舒适减速度减速，并评估变道空间"},
    ZONE_DANGER: {"action": "emergency_maneuver", "level": 2,
                  "text": "危险区：立即紧急制动(AEB)或按规则变道避让"},
    ZONE_OCCURRENCE: {"action": "minimize_damage", "level": 3,
                      "text": "发生区：碰撞不可避免，全力制动减小相对速度，并调整接触角度以最小化损失"},
    ZONE_IMPACT: {"action": "post_impact", "level": 4,
                  "text": "已发生接触：稳住方向、持续制动、开启双闪、执行最小风险停车"},
    ZONE_OCCLUDED: {"action": "occluded", "level": 0,
                    "text": "遮挡（鬼探头）：参与者不可见，无有效感知，应降低车速/依赖遮挡推理"},
    ZONE_NA: {"action": "unknown", "level": -1, "text": "位姿缺失，无法判定"},
}

PHASE_PARAMS_DEFAULT = {
    "t_reaction_s": 0.7,
    "a_comfort_mps2": 3.0,        # 舒适纵向减速度（预警/危险分界）
    "a_emergency_mps2": 8.0,      # 纵向紧急制动上限（再受 μ·g 约束）
    "mu": 0.8,                    # 路面附着系数
    "g_mps2": 9.8,
    "lane_width_m": 3.5,          # 转向避让的横向位移（一个车道）
    "a_lat_comfort_mps2": 2.0,    # 舒适横向加速度
    "horizon_s": 4.0,
    "dt_pred_s": 0.1,
    "a_scan_step_mps2": 0.25,
    "d_near_miss_m": 3.0,         # 无接触但"近距离擦过"的最近距离阈值
    "vru_margin_m": 1.0,          # 行人类物体的额外安全余量
    "max_region_radius_m": 40.0,  # 区域环半径封顶（避免高速对向时环铺满整个场景）
    "max_pair_distance_m": 50.0,  # 全局两两：超过该间距的对接"接近过程"无意义（判 safe，去噪）
    "thw_warning_s": 1.0,         # 车头时距阈值：低于它且无预测接触时判"预警"（跟车/贴车过近）
    "thw_danger_s": 0.5,          # 车头时距阈值：低于它判"危险"
    "post_impact_frames": 2,
    "severity_delta_v_high": 14.0,
    "severity_delta_v_med": 7.0,
    "accel_prediction": True,     # 二阶预测：用实测加速度（饱和）外推
    "accel_sat_mps2": 3.0,        # 预测加速度饱和值
}

SCENARIO_PHASE_PROFILES: Dict[str, Dict[str, float]] = {
    "pedestrian-crossing": {"t_reaction_s": 0.9, "a_emergency_mps2": 9.0, "d_near_miss_m": 3.5},
    "occluded-pedestrian-pileup": {"t_reaction_s": 1.0, "d_near_miss_m": 3.5},
    "head-on": {"a_emergency_mps2": 9.0, "a_comfort_mps2": 3.5},
    "hard-brake": {"t_reaction_s": 1.2},
    "chain-reaction-rear-end": {"t_reaction_s": 0.9},
    "cutin-brake-pileup": {"t_reaction_s": 0.9},
    "intersection-tbone": {"mu": 0.7},
}


# --------------------------------------------------------------------------- 工具
def _dims(tm, tid: int) -> List[float]:
    try:
        d = tm.get_track_dimensions(int(tid))
        if d and all(abs(float(x)) > 1e-3 for x in d[:3]):
            return [float(d[0]), float(d[1]), float(d[2])]
    except Exception:  # noqa: BLE001
        pass
    return [2.0, 1.7, 4.6]


def _class_of(dims: List[float]) -> str:
    try:
        vol = abs(float(dims[0]) * float(dims[1]) * float(dims[2]))
    except Exception:  # noqa: BLE001
        return "vehicle"
    if vol < 2.0:
        return "pedestrian"
    if vol < 30.0:
        return "vehicle"
    return "large_vehicle"


def _radius(dims: List[float]) -> float:
    return 0.5 * math.hypot(float(dims[0]), float(dims[2]))


def _contact(pa, dims_a, pb, dims_b) -> bool:
    ca, cb = pa[:3, 3], pb[:3, 3]
    if float(np.linalg.norm(ca - cb)) > _radius(dims_a) + _radius(dims_b) + 1e-3:
        return False
    hit, _ = check_collision(np.asarray(pa, dtype=np.float32), dims_a,
                             np.asarray(pb, dtype=np.float32), dims_b)
    return bool(hit)


def _motion_series(tm, tid, frames, fps):
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


def _advance(pose: np.ndarray, vel: np.ndarray, acc: np.ndarray, dt: float,
             accel_prediction: bool, accel_sat: float):
    """二阶外推一步：速度沿航向以饱和后的纵向加速度更新（速度不回退到负）。"""
    p = np.array(pose, dtype=np.float64).copy()
    v = np.asarray(vel, dtype=np.float64).copy()
    if accel_prediction:
        spd = float(np.linalg.norm(v))
        if spd > 1e-6:
            h = v / spd
            a_long = float(np.dot(np.asarray(acc, dtype=np.float64), h))
            a_long = max(-accel_sat, min(accel_sat, a_long))
            spd = max(0.0, spd + a_long * dt)
            v = h * spd
    p[:3, 3] += v * dt
    return p, v


def _predict_contact(pa, va, aa, dims_a, pb, vb, ab, dims_b, params) -> Tuple[Optional[float], float]:
    """二阶外推下两物体最早接触时间与最近中心距。"""
    pa = np.array(pa, dtype=np.float64).copy()
    pb = np.array(pb, dtype=np.float64).copy()
    va = np.asarray(va, dtype=np.float64).copy()
    vb = np.asarray(vb, dtype=np.float64).copy()
    aa = np.asarray(aa, dtype=np.float64)
    ab = np.asarray(ab, dtype=np.float64)
    t = 0.0
    min_d = float(np.linalg.norm(pa[:3, 3] - pb[:3, 3]))
    dt = float(params["dt_pred_s"])
    horizon = float(params["horizon_s"])
    ap = bool(params.get("accel_prediction", True))
    sat = float(params.get("accel_sat_mps2", 3.0))
    while t <= horizon:
        d = float(np.linalg.norm(pa[:3, 3] - pb[:3, 3]))
        min_d = min(min_d, d)
        if _contact(pa, dims_a, pb, dims_b):
            return t, min_d
        pa, va = _advance(pa, va, aa, dt, ap, sat)
        pb, vb = _advance(pb, vb, ab, dt, ap, sat)
        t += dt
    return None, min_d


def _contact_with_braking(pa, va, aa, dims_a, pb, vb, ab, dims_b, a, params) -> bool:
    """主车反应时间后以恒定减速度 a 制动（沿当前航向），对方二阶外推，是否仍会接触。

    a=0 表示"不干预"，此时主车也按二阶自然运动推进（与 `_predict_contact` 完全一致），
    保证 `min_required_deceleration` 的"无需制动"判据与"预测会撞"判据严格自洽。
    """
    pa = np.array(pa, dtype=np.float64).copy()
    pb = np.array(pb, dtype=np.float64).copy()
    v = np.asarray(va, dtype=np.float64).copy()
    vb = np.asarray(vb, dtype=np.float64).copy()
    aa = np.asarray(aa, dtype=np.float64)
    ab = np.asarray(ab, dtype=np.float64)
    t = 0.0
    dt = float(params["dt_pred_s"])
    horizon = float(params["horizon_s"])
    t_r = float(params["t_reaction_s"])
    ap = bool(params.get("accel_prediction", True))
    sat = float(params.get("accel_sat_mps2", 3.0))
    while t <= horizon:
        if _contact(pa, dims_a, pb, dims_b):
            return True
        if a > 0.0 and t >= t_r:
            speed = float(np.linalg.norm(v))
            if speed > 1e-6:
                h = v / speed
                speed = max(0.0, speed - a * dt)
                v = h * speed
            pa[:3, 3] += v * dt
        else:
            pa, v = _advance(pa, v, aa, dt, ap, sat)
        pb, vb = _advance(pb, vb, ab, dt, ap, sat)
        t += dt
    return False


def _deceleration_closed_form_1d(gap: float, v_rel: float, t_r: float) -> Optional[float]:
    """一维闭合场景（对向/追尾，B 匀速或静止，转化为 B 的参考系）的**解析上界**。

    设有效接触间隙 `gap`（已扣掉两车半长：OBB 相切时的中心距），相对速度 v_rel。
    反应时间 t_r 内相对速度不变，剩余间隙 d = gap - v_rel·t_r；之后要以恒定减速度 a 在
    相撞前把相对速度降到 0，最小 a 满足：

        v_rel² = 2·a·d   ⟹   a* = v_rel² / (2d)  =  v_rel² / (2·(gap - v_rel·t_r))

    这是"恒定减速度刹停"的标准运动学闭式（RSS / ISO 15623 制动判据的连续版本）。
    它是"brake-to-stop（把相对速度降到 0）"的**解析上界**：数值解 `min_required_deceleration`
    只需在预测时域 H 内避免接触（不必完全刹停），因此当 H 不足以完全刹停时，闭式 ≥ 数值解。
    d ≤ 0 表示反应期内就已接触（a* = ∞）；一般二阶加速度 / 横穿情形无闭式解。
    """
    d = float(gap) - float(v_rel) * float(t_r)
    if d <= 1e-9:
        return None if v_rel <= 1e-9 else float("inf")
    if float(v_rel) <= 1e-9:
        return 0.0
    return float(v_rel) * float(v_rel) / (2.0 * d)


def min_required_deceleration(pa, va, aa, dims_a, pb, vb, ab, dims_b, params) -> float:
    """最小避免接触的纵向减速度 a*（对 a 的**二分精确解**）。

    判据 `P(a)` = "主车反应 t_r 后以恒定减速度 a 制动，预测时域内不再接触"。
    对一维闭合（对向/追尾）P 关于 a 单调（减速度越大越早停），故二分收敛到阈值 a*；
    a* = 0 表示无需干预，a* = ∞ 表示直到能力上限仍不可避免（ICS）。
    解析参考值见 `_deceleration_closed_form_1d`；一般情形（二阶加速度/横穿）用二分。
    """
    cap_brake = float(params["cap_brake_mps2"])
    if not _contact_with_braking(pa, va, aa, dims_a, pb, vb, ab, dims_b, 0.0, params):
        return 0.0
    if _contact_with_braking(pa, va, aa, dims_a, pb, vb, ab, dims_b, cap_brake, params):
        return float("inf")
    lo, hi = 0.0, cap_brake    # 不变式：P(lo)=仍接触，P(hi)=已避免
    for _ in range(36):
        mid = 0.5 * (lo + hi)
        if _contact_with_braking(pa, va, aa, dims_a, pb, vb, ab, dims_b, mid, params):
            lo = mid
        else:
            hi = mid
    return hi


def _lateral_clearance(dims_a, dims_b, params) -> float:
    """转向避让所需的最小横向位移（OBB 感知，而非固定一个车道宽）。

    要避免接触，主车需横向移动"两车半宽之和 + 安全余量"（从完全重叠到刚好错开）。
    相比固定 lane_width（≈3.5 m），OBB 感知的横移更贴近真实：两辆 2 m 宽的车只需
    ~2 m 横移即可错开，而非整整一个车道。
    """
    try:
        wa = float(dims_a[0]) if dims_a and len(dims_a) > 0 else float(params["lane_width_m"])
        wb = float(dims_b[0]) if dims_b and len(dims_b) > 0 else float(params["lane_width_m"])
        margin = float(params["d_near_miss_m"]) * 0.5   # 半侧安全余量
        return 0.5 * (wa + wb) + margin
    except Exception:  # noqa: BLE001
        return float(params["lane_width_m"])


def _a_lat_required(ttc: float, t_reaction: float, clearance: float) -> float:
    t_avail = float(ttc) - float(t_reaction)
    if t_avail <= 0.05:
        return float("inf")
    return 2.0 * float(clearance) / (t_avail * t_avail)


def _geometry(va, vb) -> str:
    na, nb = float(np.linalg.norm(va)), float(np.linalg.norm(vb))
    if na < 0.3 or nb < 0.3:
        return "stationary"
    cos = float(np.clip(np.dot(va / na, vb / nb), -1.0, 1.0))
    ang = math.degrees(math.acos(cos))
    if ang >= 135.0:
        return "head-on"
    if ang >= 45.0:
        return "side"
    return "rear"


def _severity(delta_v: float, geometry: str, vru: bool, params) -> str:
    dv = float(delta_v)
    if vru:
        dv *= 1.6
    if geometry == "head-on":
        dv *= 1.2
    if dv >= float(params["severity_delta_v_high"]):
        return "high"
    if dv >= float(params["severity_delta_v_med"]):
        return "medium"
    return "low"


def _risk_level(zone: str, severity: str) -> str:
    if zone == ZONE_OCCURRENCE and severity == "high":
        return "critical"
    if zone == ZONE_OCCURRENCE:
        return "severe"
    if zone == ZONE_DANGER:
        return "high"
    if zone == ZONE_WARNING:
        return "elevated"
    if zone == ZONE_IMPACT:
        return "critical"
    return "low"


# --------------------------------------------------------------------------- 主入口
def _prepare_params(params: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """合并默认参数并算能力包络上限（受 μ 约束的制动/转向）。"""
    p = dict(PHASE_PARAMS_DEFAULT)
    if params is not None:
        for k, v in params.items():
            if v is not None:
                p[k] = v
    p["cap_brake_mps2"] = min(float(p["a_emergency_mps2"]),
                              float(p["mu"]) * float(p["g_mps2"]))
    p["cap_lat_mps2"] = 0.9 * float(p["mu"]) * float(p["g_mps2"])
    return p


def compute_phase_timeline(tm, actor: int, others: List[int], frames: List[int], fps: float,
                           params: Optional[Dict[str, Any]] = None,
                           occlusion: Optional[Dict[int, int]] = None,
                           collision_frames: Optional[List[int]] = None) -> Dict[str, Any]:
    """计算反应主体(actor)相对其它参与者(others)的逐帧多因素相态时程。

    Args:
        collision_frames: 撞击帧列表（生成器已知的接触时刻，可多个，用于连环事故）。
    """
    params = _prepare_params(params)

    others = [int(o) for o in others if int(o) != int(actor)]
    occlusion = {int(k): int(v) for k, v in (occlusion or {}).items()}
    cfrs = sorted({int(c) for c in (collision_frames or []) if c is not None})
    post = int(params["post_impact_frames"])

    dims_actor = _dims(tm, int(actor))
    _, vel_actor, acc_actor = _motion_series(tm, int(actor), frames, fps)
    mot = {int(o): _motion_series(tm, int(o), frames, fps) for o in others}
    dims = {int(o): _dims(tm, int(o)) for o in others}
    cls = {int(o): _class_of(dims[o]) for o in others}

    timeline: List[Dict[str, Any]] = []
    for f in frames:
        pa = tm.get_track_pose(int(actor), int(f))
        if pa is None:
            timeline.append({"frame": int(f), "zone": ZONE_NA, "factors": None,
                             "advice": _ZONE_ADVICE[ZONE_NA], "advice_level": -1})
            continue
        pa = np.asarray(pa, dtype=np.float64)
        va = vel_actor.get(int(f), np.zeros(3, dtype=np.float64))
        aa = acc_actor.get(int(f), np.zeros(3, dtype=np.float64))
        v_abs = float(np.linalg.norm(va))
        rw0, rd0 = _envelope_radii(v_abs, params)   # 无配对/撞击帧兜底：用主车绝对车速

        if cfrs and any(c <= int(f) <= c + post for c in cfrs):
            timeline.append(_impact_entry(f, va, aa, rw0, rd0))
            continue
        if cfrs and int(f) > max(cfrs) + post:
            timeline.append(_concluded_entry(f, va, aa, rw0, rd0))
            continue

        # 无其它参与者 → 单参与者急刹（hard-brake 类）：按主车自身制动强度给相态
        if not others:
            timeline.append(_single_vehicle_entry(f, va, aa, params, rw0, rd0))
            continue

        best = _best_of_others(tm, pa, va, aa, dims_actor, others, mot, dims, cls, params,
                               occlusion, int(f))
        # 区域半径用"闭合速度"（实际接近速度），而不是主车绝对车速——否则会大得离谱
        _fac = best.get("factors") or {}
        _vc = float(_fac.get("closing_speed_mps") if _fac.get("closing_speed_mps") is not None
                    else (_fac.get("relative_speed_mps") or v_abs))
        rw, rd = _envelope_radii(_vc, params)
        advice = _ZONE_ADVICE.get(best["zone"], _ZONE_ADVICE[ZONE_NA])
        timeline.append({
            "frame": int(f), "zone": best["zone"],
            "factors": best.get("factors"),
            "other": best.get("other"),
            "binding": best.get("binding"),
            "risk_level": best.get("risk_level"),
            "risk_score": best.get("risk_score"),
            "decision": best.get("decision"),
            "r_warning_m": rw, "r_danger_m": rd,
            "advice": advice, "advice_level": int(advice.get("level", -1)),
        })

    return _finish_timeline(actor, others, frames, fps, params, timeline)


def _envelope_radii(v, params):
    """主车"制动能力包络"的两个半径（用于空间可视化）。

    用**闭合速度**（两车实际接近速度）而非主车绝对车速：追尾时闭合速度可能只有 5m/s，
    而主车绝对车速 20m/s 会把环画成 80m、铺满整个场景。

    r_danger = v·t_r + v²/(2·cap_brake)：紧急制动刚好能停住的最远距离（危险区外边界）；
    r_warning = v·t_r + v²/(2·a_comfort)：舒适制动刚好能停住（预警区外边界）。
    半径按 max_region_radius_m 封顶，避免高速对向/远距离时环覆盖整个画面。
    """
    v = max(0.0, float(v))
    t_r = float(params["t_reaction_s"])
    a_comf = float(params["a_comfort_mps2"])
    cap_brake = float(params["cap_brake_mps2"])
    r_warn = v * t_r + v * v / (2.0 * a_comf)
    r_dang = v * t_r + v * v / (2.0 * cap_brake)
    rmax = float(params.get("max_region_radius_m", 40.0))
    return round(min(r_warn, rmax), 2), round(min(r_dang, rmax), 2)


def _impact_entry(f, va, aa, rw, rd):
    return {"frame": int(f), "zone": ZONE_IMPACT, "factors": None, "other": None,
            "binding": "impact", "risk_level": "critical",
            "r_warning_m": rw, "r_danger_m": rd,
            "advice": _ZONE_ADVICE[ZONE_IMPACT], "advice_level": 4}


def _concluded_entry(f, va, aa, rw, rd):
    return {"frame": int(f), "zone": ZONE_SAFE, "factors": None, "other": None,
            "binding": None, "risk_level": "low",
            "r_warning_m": rw, "r_danger_m": rd,
            "advice": _ZONE_ADVICE[ZONE_SAFE], "advice_level": 0, "event_concluded": True}


def _single_vehicle_entry(f, va, aa, params, rw, rd):
    """单参与者（急刹）：把主车自身的制动强度映射成相态（对后车构成追尾风险）。"""
    spd = float(np.linalg.norm(va))
    decel = 0.0
    if spd > 1e-3:
        h = np.asarray(va) / spd
        decel = max(0.0, -float(np.dot(np.asarray(aa), h)))
    cap_brake = float(params["cap_brake_mps2"])
    a_comf = float(params["a_comfort_mps2"])
    if decel > cap_brake:
        zone = ZONE_DANGER
    elif decel > a_comf:
        zone = ZONE_WARNING
    else:
        zone = ZONE_SAFE
    return {"frame": int(f), "zone": zone,
            "factors": {"self_decel_mps2": round(decel, 3), "cap_brake_mps2": round(cap_brake, 3)},
            "other": None, "binding": "self_brake", "risk_level": _risk_level(zone, "low"),
            "r_warning_m": rw, "r_danger_m": rd,
            "advice": _ZONE_ADVICE[zone], "advice_level": _ZONE_ADVICE[zone]["level"]}


# 风险标量：紧急度(区域) × 严重度(Δv·几何·VRU)，0~1 单调递增（可解释、可比较）
_ZONE_URGENCY = {ZONE_SAFE: 0.0, ZONE_OCCLUDED: 0.0, ZONE_NA: 0.0,
                 ZONE_WARNING: 0.33, ZONE_DANGER: 0.67, ZONE_OCCURRENCE: 0.9, ZONE_IMPACT: 1.0}
_SEV_WEIGHT = {"low": 0.6, "medium": 0.8, "high": 1.0}


def _risk_score(zone, severity):
    """标量风险指数 = 紧急度(区域) × 严重度权重，0~1。"""
    urg = _ZONE_URGENCY.get(zone, 0.0)
    sev = _SEV_WEIGHT.get(severity, 0.8)
    return round(urg * sev, 3)


def _mk(zone, binding, factors, decision, sev):
    return {"zone": zone, "binding": binding, "factors": factors,
            "risk_level": _risk_level(zone, sev),
            "risk_score": _risk_score(zone, sev),
            "decision": decision}


def analyze_pair(pa, va, aa, dims_a, pb, vb, ab, dims_b, class_b, params):
    """单对参与者的多因素相态判定（最小可解释单元）。

    返回 {zone, binding, factors, risk_level, risk_score, decision}。
    `decision` 是**决策链路**：记录每个因素的取值、能力档位、以及得出该区域的规则，
    让每一条区域判定都可被追溯、可被评审（论文里"为什么是这个区域"的直接证据）。
    """
    t_r = float(params["t_reaction_s"])
    a_comf = float(params["a_comfort_mps2"])
    a_lat_comf = float(params["a_lat_comfort_mps2"])
    cap_brake = float(params["cap_brake_mps2"])
    cap_lat = float(params["cap_lat_mps2"])
    vru = (class_b == "pedestrian")
    margin = float(params["d_near_miss_m"]) + (float(params["vru_margin_m"]) if vru else 0.0)

    rel_spd = float(np.linalg.norm(va - vb))
    closing_spd = _closing_speed(pa, pb, va, vb)
    geo = _geometry(va, vb)
    sev = _severity(rel_spd, geo, vru, params)

    def fac(kin, thw=None):
        return _factors(kin, rel_spd, closing_spd, va, vb, vru, params, thw=thw)

    if _contact(pa, dims_a, pb, dims_b):
        return _mk(ZONE_IMPACT, "impact", fac(None),
                   {"rule": "两物体 OBB 已接触（撞击）"}, sev)

    ttc, min_d = _predict_contact(pa, va, aa, dims_a, pb, vb, ab, dims_b, params)
    if ttc is None:
        ang = _heading_angle(va, vb)
        if ang < 45.0:
            thw = _time_headway(pa, va, pb)
            if thw is None:
                thw = _time_headway(pb, vb, pa)
            if thw is not None:
                if thw < float(params["thw_danger_s"]):
                    return _mk(ZONE_DANGER, "headway", fac(None, thw),
                               {"rule": "同向贴车：THW=%.2fs < thw_danger=%.1f → 危险" % (thw, params["thw_danger_s"]),
                                "thw_s": round(thw, 3), "heading_angle_deg": round(ang, 1)}, sev)
                if thw < float(params["thw_warning_s"]):
                    return _mk(ZONE_WARNING, "headway", fac(None, thw),
                               {"rule": "同向贴车：THW=%.2fs < thw_warning=%.1f → 预警" % (thw, params["thw_warning_s"]),
                                "thw_s": round(thw, 3), "heading_angle_deg": round(ang, 1)}, sev)
        if min_d < margin and closing_spd > 0.5:
            return _mk(ZONE_WARNING, "near_miss", fac(None),
                       {"rule": "无接触但近距擦过：min_d=%.2fm < margin=%.2fm 且闭合=%.2fm/s → 预警"
                                % (min_d, margin, closing_spd),
                        "min_d_m": round(min_d, 3), "closing_speed_mps": round(closing_spd, 3)}, sev)
        return _mk(ZONE_SAFE, None, fac(None),
                   {"rule": "无预测接触、时距足够、非近距 → 安全"}, sev)

    a_brake = min_required_deceleration(pa, va, aa, dims_a, pb, vb, ab, dims_b, params)
    clearance = _lateral_clearance(dims_a, dims_b, params)
    a_lat = _a_lat_required(ttc, t_r, clearance)
    bt = 0 if a_brake == 0.0 else (1 if a_brake <= a_comf else (2 if a_brake <= cap_brake else 3))
    st = 1 if a_lat <= a_lat_comf else (2 if a_lat <= cap_lat else 3)
    tier = max(1, min(bt, st))
    zone = {1: ZONE_WARNING, 2: ZONE_DANGER, 3: ZONE_OCCURRENCE}[tier]
    binding = "brake" if bt <= st else "steer"
    if tier == 3:
        binding = "none"
    mb = (1.0 - float(a_brake) / cap_brake) if a_brake != float("inf") else None
    ml = (1.0 - float(a_lat) / cap_lat) if a_lat != float("inf") else None
    a_brake_s = "∞" if a_brake == float("inf") else "%.2f" % a_brake
    a_lat_s = "∞" if a_lat == float("inf") else "%.2f" % a_lat
    # 一维闭合（对向/追尾）的解析参考值（brake-to-stop 上界，已扣 OBB 半长）
    a_closed = None
    if geo in ("head-on", "rear"):
        gap = float(np.linalg.norm(np.asarray(pb[:3, 3]) - np.asarray(pa[:3, 3])))
        eff_gap = gap - 0.5 * (float(dims_a[2]) + float(dims_b[2]))
        a_closed = _deceleration_closed_form_1d(eff_gap, rel_spd, t_r)
    decision = {
        "a_brake_mps2": None if a_brake == float("inf") else round(float(a_brake), 3),
        "a_lat_mps2": None if a_lat == float("inf") else round(float(a_lat), 3),
        "lateral_clearance_m": round(float(clearance), 3),
        "a_closed_form_mps2": None if (a_closed is None or a_closed == float("inf")) else round(float(a_closed), 3),
        "tier_brake": int(bt), "tier_steer": int(st), "tier": int(tier),
        "margin_brake": None if mb is None else round(float(mb), 3),
        "margin_lat": None if ml is None else round(float(ml), 3),
        "ttc_s": round(float(ttc), 3),
        "rule": "能力包络：制动档=%d(a*=%s, 舒适%.1f/紧急%.1f/余量%s)，转向档=%d(a_lat=%s, 横移%.2fm, 舒适%.1f/紧急%.1f/余量%s) → tier=min=%d → %s"
                % (bt, a_brake_s, a_comf, cap_brake, ("%.2f" % mb if mb is not None else "∞"),
                   st, a_lat_s, clearance, a_lat_comf, cap_lat, ("%.2f" % ml if ml is not None else "∞"),
                   tier, zone),
    }
    return _mk(zone, binding, fac((a_brake, a_lat, ttc, clearance)), decision, sev)


def _best_of_others(tm, pa, va, aa, dims_actor, others, mot, dims, cls, params, occlusion, f):
    best = {"zone": ZONE_SAFE, "factors": None, "other": None, "binding": None,
            "risk_level": "low"}
    best_order = _ZONE_ORDER[ZONE_SAFE]
    for other in others:
        if f < occlusion.get(other, -1):
            if _ZONE_ORDER[ZONE_OCCLUDED] > best_order:
                best = {"zone": ZONE_OCCLUDED, "factors": None, "other": int(other),
                        "binding": "occluded", "risk_level": "low"}
                best_order = _ZONE_ORDER[ZONE_OCCLUDED]
            continue
        pb_raw = tm.get_track_pose(int(other), int(f))
        if pb_raw is None:
            continue
        pb = np.asarray(pb_raw, dtype=np.float64)
        vb = mot[other][1].get(int(f), np.zeros(3, dtype=np.float64))
        ab = mot[other][2].get(int(f), np.zeros(3, dtype=np.float64))
        cand = analyze_pair(pa, va, aa, dims_actor, pb, vb, ab, dims[other], cls[other], params)
        cand["other"] = int(other)
        if _ZONE_ORDER[cand["zone"]] > best_order:
            best, best_order = cand, _ZONE_ORDER[cand["zone"]]
    return best


def _closing_speed(pa, pb, va, vb) -> float:
    """两物体沿连线的闭合速度（gap 缩小的速率，≥0）。

    追尾时是"速度差"、对向时是"速度和"、横穿时随夹角变化——用这个算区域半径
    才能反映**实际接近情况**，而不是主车绝对车速（那会把追尾 20m/s 的车画成 80m 的环）。
    """
    d = np.asarray(pb[:3, 3], dtype=np.float64) - np.asarray(pa[:3, 3], dtype=np.float64)
    dd = float(np.linalg.norm(d))
    if dd < 1e-3:
        return 0.0
    u = d / dd
    return max(0.0, float(np.dot(np.asarray(va, dtype=np.float64) - np.asarray(vb, dtype=np.float64), u)))


def _time_headway(pa, va, pb) -> Optional[float]:
    """车头时距 THW = 沿主车航向的纵向间距 / 主车速度。

    只对"对方在主车前方至少 2m"有效（long_gap≤2m 排除并排/后方），不再按横向车道限制：
    同向贴行的车不一定同车道，纵向接近就应有预警。
    """
    head = _heading_unit(va)
    rel = np.asarray(pb[:3, 3], dtype=np.float64) - np.asarray(pa[:3, 3], dtype=np.float64)
    long_gap = float(np.dot(rel, head))
    if long_gap <= 2.0:
        return None
    spd = float(np.linalg.norm(va))
    if spd < 0.5:
        return None
    return long_gap / spd


def _heading_angle(va, vb) -> float:
    """两车航向夹角（度，0=同向，180=对向）。"""
    na, nb = float(np.linalg.norm(va)), float(np.linalg.norm(vb))
    if na < 0.3 or nb < 0.3:
        return 0.0
    ha = np.array([va[0], va[2]], dtype=np.float64)
    hb = np.array([vb[0], vb[2]], dtype=np.float64)
    ha = ha / np.linalg.norm(ha)
    hb = hb / np.linalg.norm(hb)
    cos = float(np.clip(np.dot(ha, hb), -1.0, 1.0))
    return math.degrees(math.acos(cos))


def _factors(kin, rel_spd, closing_spd, va, vb, vru, params, thw=None):
    """拼一个可追溯的因素字典。kin = (a_brake, a_lat, ttc, clearance) 或 None。"""
    f = {
        "relative_speed_mps": round(rel_spd, 3),
        "closing_speed_mps": round(closing_spd, 3),
        "delta_v_mps": round(rel_spd, 3),
        "geometry": _geometry(va, vb),
        "class_other": "pedestrian" if vru else "vehicle",
        "mu": float(params["mu"]),
        "cap_brake_mps2": round(float(params["cap_brake_mps2"]), 2),
        "cap_lat_mps2": round(float(params["cap_lat_mps2"]), 2),
    }
    if thw is not None:
        f["thw_s"] = round(float(thw), 2)
    if kin is not None:
        a_brake, a_lat, ttc, clearance = kin
        f["ttc_s"] = round(float(ttc), 3)
        f["reaction_margin_s"] = round(max(0.0, float(ttc) - float(params["t_reaction_s"])), 3)
        f["a_brake_mps2"] = None if a_brake == float("inf") else round(float(a_brake), 3)
        f["a_lat_mps2"] = None if a_lat == float("inf") else round(float(a_lat), 3)
        f["lateral_clearance_m"] = round(float(clearance), 3)
        cb = float(params["cap_brake_mps2"])
        cl = float(params["cap_lat_mps2"])
        # 机动余量：1=零负担，0=用满能力，<0=能力不足（可解释的"还剩多少能力"）
        f["margin_brake"] = None if a_brake == float("inf") else round(1.0 - float(a_brake) / cb, 3)
        f["margin_lat"] = None if a_lat == float("inf") else round(1.0 - float(a_lat) / cl, 3)
        _m = [x for x in (f["margin_brake"], f["margin_lat"]) if x is not None]
        f["maneuver_margin"] = round(float(max(_m)), 3) if _m else None
    return f


def _finish_timeline(actor, others, frames, fps, params, timeline) -> Dict[str, Any]:
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
        return 0.0 if lo is None or hi is None else max(0.0, (hi - lo) / float(fps))

    w_f, d_f = first_frame(ZONE_WARNING), first_frame(ZONE_DANGER)
    o_f, i_f = first_frame(ZONE_OCCURRENCE), first_frame(ZONE_IMPACT)
    oc_f = first_frame(ZONE_OCCLUDED)
    # 最晚反应点 (Latest Reaction Point, LRP)：最后一个"危险区"帧 —— 进入发生/撞击前，
    # 仍可用紧急动作规避的**最后时刻**；若无危险区，退而取最后一个预警区。
    # 附上该点的 TTC 与机动余量，把 LRP 从"一个帧号"升级为"帧号 + 还剩多少时间 + 还剩多少能力"。
    latest_reaction = None
    latest_reaction_ctx: Dict[str, Any] = {}
    for e in timeline:
        if e["zone"] == ZONE_DANGER:
            latest_reaction = e["frame"]
            latest_reaction_ctx = e
    if latest_reaction is None:
        for e in timeline:
            if e["zone"] == ZONE_WARNING:
                latest_reaction = e["frame"]
                latest_reaction_ctx = e
    _lf = latest_reaction_ctx.get("factors") or {}
    lrp_ttc = _lf.get("ttc_s")
    lrp_margin = _lf.get("maneuver_margin")
    # 峰值风险（0~1），用于量化整条交互的危险程度
    max_risk = max((e.get("risk_score") or 0.0 for e in timeline), default=0.0)
    event = ("collision" if i_f is not None else
             "near_miss" if o_f is not None else
             "avoid" if (d_f is not None or w_f is not None) else "none")
    dw, dd, do = duration(ZONE_WARNING), duration(ZONE_DANGER), duration(ZONE_OCCURRENCE)
    compressed = (dw <= 0.3) and (dd > 0.0 or do > 0.0 or i_f is not None)
    skipped = (dw == 0.0) and (dd > 0.0 or do > 0.0 or i_f is not None)
    story = _make_story(w_f, d_f, o_f, i_f, oc_f, dw, dd, do, compressed, skipped, event)
    return {
        "actor": int(actor), "others": [int(o) for o in others], "fps": float(fps),
        "params": params, "timeline": timeline,
        "boundaries": {"warning_frame": w_f, "danger_frame": d_f,
                       "occurrence_frame": o_f, "collision_frame": i_f,
                       "latest_reaction_frame": latest_reaction},
        "summary": {
            "first_warning_frame": w_f, "first_danger_frame": d_f,
            "first_occurrence_frame": o_f, "collision_frame": i_f,
            "first_occluded_frame": oc_f,
            "latest_reaction_frame": latest_reaction,
            "latest_reaction": {"frame": latest_reaction, "ttc_s": lrp_ttc,
                                "maneuver_margin": lrp_margin},
            "zone_durations_s": {"warning": round(dw, 2), "danger": round(dd, 2),
                                 "occurrence": round(do, 2)},
            "max_risk_score": round(max_risk, 3),
            "compressed_warning": bool(compressed), "skipped_warning": bool(skipped),
            "event": event, "story": story,
        },
    }


def _make_story(w_f, d_f, o_f, i_f, oc_f, dw, dd, do, compressed, skipped, event) -> str:
    if event == "none":
        return "全程安全，预测时域内未出现冲突或危险接近，无需干预。"
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


# --------------------------------------------------------------------------- 全局两两相态
def _tracks_at_frame(tm, f: int) -> List[int]:
    """该帧存在的所有动态物体（真实 + 合成，跳过 ego 与已删除/替换）。"""
    tids = set()
    deleted = set(getattr(tm, "track_deleted", set()) or set())
    replaced = set(getattr(tm, "track_replaced", set()) or set())
    for tid, fm in (getattr(tm, "track_to_frames", {}) or {}).items():
        if int(f) in (fm or {}) and int(tid) not in deleted and int(tid) not in replaced:
            tids.add(int(tid))
    for sid, s in (getattr(tm, "synthetic_tracks", {}) or {}).items():
        if s.get("ego"):
            continue
        if int(f) in (s.get("poses") or {}):
            tids.add(int(sid))
    return sorted(tids)


def compute_pairwise_phases(tm, frames: List[int], fps: float,
                            params: Optional[Dict[str, Any]] = None,
                            max_pairs_per_frame: int = 40,
                            track_ids: Optional[List[int]] = None) -> Dict[int, List[Dict[str, Any]]]:
    """全局两两相态：在**指定的参与者集合**内做两两多阶段区域判定。

    Args:
        track_ids: 显式指定"哪些物体是车辆/参与者"；只在这些物体之间判定。
                   为 None（默认）时对**场景里所有物体**全局两两判定（不按类型过滤）。

    返回 {frame: [ {a, b, zone, binding, factors, risk_level, r_warning_m, r_danger_m,
                   pos_a, pos_b} ... ]}，只保留非 safe 的冲突对（按严重度降序，封顶）。
    """
    params = _prepare_params(params)
    frames = list(frames)
    tids = set()
    for f in frames:
        tids |= set(_tracks_at_frame(tm, f))
    mot, dims, cls = {}, {}, {}
    for tid in tids:
        mot[tid] = _motion_series(tm, tid, frames, fps)
        dims[tid] = _dims(tm, tid)
        cls[tid] = _class_of(dims[tid])

    # 参与者集合：显式指定用 track_ids；否则默认全局所有物体
    if track_ids:
        allowed = {int(t) for t in track_ids}
    else:
        allowed = set(tids)

    order = {ZONE_IMPACT: 5, ZONE_OCCURRENCE: 4, ZONE_DANGER: 3,
             ZONE_WARNING: 2, ZONE_OCCLUDED: 1}
    out: Dict[int, List[Dict[str, Any]]] = {}
    for f in frames:
        present = [t for t in _tracks_at_frame(tm, f) if t in allowed]
        rows = []
        for i in range(len(present)):
            for j in range(i + 1, len(present)):
                ta, tb = present[i], present[j]
                pa_raw = tm.get_track_pose(ta, f)
                pb_raw = tm.get_track_pose(tb, f)
                if pa_raw is None or pb_raw is None:
                    continue
                pa = np.asarray(pa_raw, dtype=np.float64)
                pb = np.asarray(pb_raw, dtype=np.float64)
                # 空间距离过滤：多阶段区域是"接近过程"，超过 max_pair_distance_m 的对接 safe，
                # 也避免远处高速噪声轨迹被误判成冲突。
                if float(np.linalg.norm(pa[:3, 3] - pb[:3, 3])) > float(params["max_pair_distance_m"]):
                    continue
                va = mot[ta][1].get(int(f), np.zeros(3, dtype=np.float64))
                vb = mot[tb][1].get(int(f), np.zeros(3, dtype=np.float64))
                aa = mot[ta][2].get(int(f), np.zeros(3, dtype=np.float64))
                ab = mot[tb][2].get(int(f), np.zeros(3, dtype=np.float64))
                # 反应主体取速度更快的一方
                if float(np.linalg.norm(va)) >= float(np.linalg.norm(vb)):
                    a, b = ta, tb
                    r = analyze_pair(pa, va, aa, dims[ta], pb, vb, ab, dims[tb], cls[tb], params)
                else:
                    a, b = tb, ta
                    r = analyze_pair(pb, vb, ab, dims[tb], pa, va, aa, dims[ta], cls[ta], params)
                if r["zone"] in (ZONE_SAFE, ZONE_NA, ZONE_IMPACT):
                    # 全局模型只报"接近过程"的三区 + 险情；"已接触"（impact）由碰撞关键帧单独标，
                    # 否则 raw 场景里行人贴着车、重复跟踪等静态重叠会被当成"撞击"（噪声极大）。
                    continue
                if r.get("binding") == "near_miss":
                    # 全局概览只保留"真的会撞 / 需要制动"的冲突对；"擦过"(<3m)在密集场景里
                    # 太多（相邻车道、行人贴着车），单条事故时程里已保留，这里去掉去噪。
                    continue
                fac = r.get("factors") or {}
                vc = float(fac.get("closing_speed_mps") if fac.get("closing_speed_mps") is not None
                           else (fac.get("relative_speed_mps") or 0.0))
                rw, rd = _envelope_radii(vc, params)
                ca = np.asarray(tm.get_track_pose(a, f))[:3, 3]
                cb = np.asarray(tm.get_track_pose(b, f))[:3, 3]
                rows.append({
                    "a": int(a), "b": int(b), "zone": r["zone"], "binding": r["binding"],
                    "factors": fac, "risk_level": r["risk_level"],
                    "risk_score": r.get("risk_score"), "decision": r.get("decision"),
                    "r_warning_m": rw, "r_danger_m": rd,
                    "pos_a": [float(x) for x in ca],
                    "pos_b": [float(x) for x in cb],
                })
        rows.sort(key=lambda x: -order.get(x["zone"], 0))
        out[int(f)] = rows[:max_pairs_per_frame]
    return out


# --------------------------------------------------------------------------- 规避控制（为 AD 训练提供"每阶段该做什么反应"）
_ZONE_SEV = {ZONE_SAFE: 0, ZONE_WARNING: 1, ZONE_DANGER: 2,
             ZONE_OCCURRENCE: 3, ZONE_IMPACT: 4, ZONE_OCCLUDED: -1, ZONE_NA: -2}


def _local_write_pose(tm, tid, f, pose):
    pose = np.asarray(pose, dtype=np.float32)
    try:
        if tm.is_synthetic(int(tid)):
            tm.set_synthetic_pose(int(tid), int(f), pose)
        else:
            tm.set_track_pose(int(tid), int(f), pose)
    except Exception:  # noqa: BLE001
        pass


def _pose_with_center(base, center):
    p = np.array(base, dtype=np.float32).copy()
    p[0, 3], p[1, 3], p[2, 3] = float(center[0]), float(center[1]), float(center[2])
    return p


def _ground_right(head):
    """航向的右向量（世界 XZ 平面）。"""
    h = np.asarray(head, dtype=np.float64)
    return np.array([h[2], 0.0, -h[0]], dtype=np.float64)


def _heading_unit(va):
    v = np.asarray(va, dtype=np.float64)
    h = np.array([v[0], 0.0, v[2]])
    n = float(np.linalg.norm(h))
    return h / n if n > 1e-6 else np.array([0.0, 0.0, 1.0])


def _brake_evasion(tm, actor, frames, fps, begin_frame, decel, va, t_reaction=0.0):
    """沿当前航向以恒定减速度制动（标准 AEB 原语），改写 actor 从 begin_frame 起的轨迹。

    **反应延迟 t_reaction**：反应期内保持原速，之后才开始制动——这与 `min_required_deceleration`
    里"反应 t_r 后再以 a 制动"的假设严格一致，保证"模型说不可规避(occurrence)"与
    "规避后仍相撞"的口径一致（不会出现"发生区却规避成功"的自相矛盾）。
    """
    spd = float(np.linalg.norm(va))
    if spd < 1e-3:
        return
    head = _heading_unit(va)
    pose0 = tm.get_track_pose(int(actor), int(begin_frame))
    if pose0 is None:
        return
    pose0 = np.asarray(pose0, dtype=np.float64)
    c0 = pose0[:3, 3].copy()
    t_r = max(0.0, float(t_reaction))
    t_brake = spd / max(1e-6, float(decel))
    d_react = spd * t_r
    d_brake = 0.5 * spd * t_brake   # v²/(2a) = 0.5*v*(v/a)
    for f in frames:
        if int(f) < int(begin_frame):
            continue
        t = (int(f) - int(begin_frame)) / float(fps)
        if t <= t_r:
            d = spd * t
        else:
            tb = t - t_r
            d = d_react + (d_brake if tb >= t_brake else (spd * tb - 0.5 * float(decel) * tb * tb))
        c = c0 + head * d
        c[1] = c0[1]
        _local_write_pose(tm, actor, int(f), _pose_with_center(pose0, c))


def _steer_evasion(tm, actor, frames, fps, begin_frame, va, lateral, direction=None,
                   dur=0.8, t_reaction=0.0):
    """横向 smoothstep 变道（转向避让）：纵向保持原速，横向在 `dur` 秒内平移 `lateral`。

    平滑曲线用 smoothstep（C¹ 连续，位移/速度/加速度都连续），这是转向避让的标准
    横移剖面；`dur` 由"在 TTC−t_r 内完成横移"反推，保证在接触前完成避让。
    `t_reaction` 是反应延迟：反应期内保持原速、不横移，之后才开始变道。
    """
    head = _heading_unit(va)
    if direction is None:
        direction = _ground_right(head)
    else:
        direction = np.asarray(direction, dtype=np.float64)
        direction[1] = 0.0
    spd = float(np.linalg.norm(va))
    pose0 = tm.get_track_pose(int(actor), int(begin_frame))
    if pose0 is None:
        return
    pose0 = np.asarray(pose0, dtype=np.float64)
    c0 = pose0[:3, 3].copy()
    dur = max(0.2, float(dur))
    t_r = max(0.0, float(t_reaction))
    for f in frames:
        if int(f) < int(begin_frame):
            continue
        t = (int(f) - int(begin_frame)) / float(fps)
        if t <= t_r:
            u = 0.0
        else:
            u = min(1.0, (t - t_r) / dur)
        sm = u * u * (3.0 - 2.0 * u)      # smoothstep
        c = c0 + head * (spd * t) + direction * (float(lateral) * sm)
        c[1] = c0[1]
        _local_write_pose(tm, actor, int(f), _pose_with_center(pose0, c))


def _recompute_zone(tm, actor, others, frames, fps, params):
    f0 = int(frames[0])
    pa = tm.get_track_pose(int(actor), f0)
    if pa is None:
        return ZONE_NA, None
    pa = np.asarray(pa, dtype=np.float64)
    dims_actor = _dims(tm, int(actor))
    _, vel, acc = _motion_series(tm, int(actor), frames, fps)
    mot = {int(o): _motion_series(tm, int(o), frames, fps) for o in others}
    dims = {int(o): _dims(tm, int(o)) for o in others}
    cls = {int(o): _class_of(dims[o]) for o in others}
    va = vel.get(f0, np.zeros(3, dtype=np.float64))
    aa = acc.get(f0, np.zeros(3, dtype=np.float64))
    best = _best_of_others(tm, pa, va, aa, dims_actor, others, mot, dims, cls, params, {}, f0)
    return best["zone"], best.get("binding")


def _pair_collides_window(tm, a, b, frames) -> bool:
    """窗口内两物体是否有任何一帧发生接触（用于判断"规避后是否还相撞"）。

    碰撞后 `separate_pair` 会把两车推到"零间隙 + 容差"，此时 SAT 判为"分离"；所以这里
    把包围盒在宽/长方向各外扩 tol，用"接触 + 容差"来识别"确实相撞过"。
    """
    if a is None or b is None:
        return False
    tol = 0.2
    da = _dims(tm, int(a))
    db = _dims(tm, int(b))
    da_inf = [da[0] + tol, da[1], da[2] + tol]
    db_inf = [db[0] + tol, db[1], db[2] + tol]
    for f in frames:
        pa = tm.get_track_pose(int(a), int(f))
        pb = tm.get_track_pose(int(b), int(f))
        if pa is None or pb is None:
            continue
        if _contact(np.asarray(pa, np.float64), da_inf, np.asarray(pb, np.float64), db_inf):
            return True
    return False


def _maneuver_decision(before_zone, a_brake_req, a_lat_req, cap_brake, cap_lat,
                       margin_brake, margin_lat, action, steer_penalty=0.15):
    """能力包络上的**风险加权最小成本**机动决策（可解释、可写论文）。

    对每个机动定义"努力比" e = 1 - μ = 所需 / 能力，再加**机动风险惩罚**：

        cost_brake = e_brake
        cost_steer = e_lat + c_steer          (c_steer ≥ 0)

    其中 c_steer 是**换道风险惩罚**：转向要横向进入相邻车道交通，风险和不确定性高于"在本车道
    内减速"，所以只有在"转向明显更省力（cost 更低）"或"制动不可行"时才选转向。这与真实
    AEB"制动优先、转向兜底"的分级响应一致，也是把"可解释的工程偏好"显式写进目标函数的做法。

    取 cost 更小者；两者都不行（e>1 或需求=∞）即不可避免碰撞态 ICS → 全力制动最小化 Δv。
    返回 (action, 原则说明)。
    """
    if action != "auto":
        return action, "外部显式指定机动：%s" % action
    if before_zone in (ZONE_SAFE, ZONE_NA, ZONE_IMPACT, ZONE_OCCLUDED):
        return "none", "处于【%s】阶段，无需规避（无预测冲突/已接触/不可见）" % before_zone
    brake_ok = a_brake_req is not None and 0.0 < float(a_brake_req) <= float(cap_brake)
    steer_ok = a_lat_req is not None and float(a_lat_req) <= float(cap_lat)
    if before_zone == ZONE_OCCURRENCE or (not brake_ok and not steer_ok):
        return "brake", "发生区（ICS）：制动与转向都不可解 → 全力制动最小化 Δv"

    def _cost(ok, margin, penalty):
        return (1.0 - float(margin) + float(penalty)) if (ok and margin is not None) else float("inf")
    cost_brake = _cost(brake_ok, margin_brake, 0.0)
    cost_steer = _cost(steer_ok, margin_lat, float(steer_penalty))
    pick = "brake" if cost_brake <= cost_steer else "steer"
    return pick, ("风险加权最小成本：cost_brake=%s vs cost_steer=%s（含换道风险惩罚 c_steer=%.2f）→ %s"
                  % ("∞" if cost_brake == float("inf") else round(cost_brake, 3),
                     "∞" if cost_steer == float("inf") else round(cost_steer, 3),
                     float(steer_penalty), pick))


def apply_evasion(tm, actor, others, frames, fps, params=None, action="auto",
                  steer_lateral_m=None, safety_factor=1.15, steer_penalty=0.15):
    """对某辆车执行**规避控制**：由多阶段模型直接给出"每阶段该做什么、做多少"的最优动作，
    改写其轨迹并验证是否规避成功——这正是自动驾驶训练时要学的响应策略。

    与旧版的区别（学术化/效果优化）：
      - **制动量 = 精确所需 a\* × 安全系数**（二分精确解），而不是一刀切用 a_comfort / a_emerg；
        预警区就用"刚好够"的舒适减速，危险区用"刚好够"的紧急减速，既不刹过头、也不刹不够。
      - **转向量 = OBB 感知的横向净空**（而非固定一个车道宽），转向时长由 `TTC - t_r` 反推，
        保证在接触前完成横移。
      - **主机动选择 = 风险加权最小成本**（努力比 + 换道风险惩罚，制动优先），与区域判定同源，
        决策链路可追溯。

    阶段 → 反应：
      warning    → 舒适制动（精确 a*，≤ a_comfort）或转向
      danger     → 紧急制动（精确 a*，≤ cap_brake）或转向
      occurrence → 全力制动（不可避免碰撞态，最小化 Δv）
      impact     → 无动作

    返回 {ok, action, maneuver, plan, before_zone, after_zone, before_collides,
          after_collides, avoided, story}。
    """
    params = _prepare_params(params)
    frames = [int(f) for f in frames]
    f0 = int(frames[0])
    actor = int(actor)
    others = [int(o) for o in others if int(o) != actor]

    pa = tm.get_track_pose(actor, f0)
    if pa is None:
        return {"ok": False, "reason": "actor 在起始帧无位姿"}
    pa = np.asarray(pa, dtype=np.float64)
    dims_actor = _dims(tm, actor)
    _, vel_actor, acc_actor = _motion_series(tm, actor, frames, fps)
    mot = {o: _motion_series(tm, o, frames, fps) for o in others}
    dims = {o: _dims(tm, o) for o in others}
    cls = {o: _class_of(dims[o]) for o in others}
    va = vel_actor.get(f0, np.zeros(3, dtype=np.float64))
    aa = acc_actor.get(f0, np.zeros(3, dtype=np.float64))
    best = _best_of_others(tm, pa, va, aa, dims_actor, others, mot, dims, cls, params, {}, f0)
    before_zone = best["zone"]
    binding = best.get("binding") or "brake"
    worst_other = best.get("other")
    fac = best.get("factors") or {}
    before_collides = _pair_collides_window(tm, actor, worst_other, frames)

    cap_brake = float(params["cap_brake_mps2"])
    cap_lat = float(params["cap_lat_mps2"])
    a_brake_req = fac.get("a_brake_mps2")
    a_lat_req = fac.get("a_lat_mps2")
    clearance = fac.get("lateral_clearance_m") or float(params["lane_width_m"])
    ttc = fac.get("ttc_s")
    margin_brake = fac.get("margin_brake")
    margin_lat = fac.get("margin_lat")
    v_rel_before = float(fac.get("delta_v_mps") or 0.0)

    action, principle = _maneuver_decision(
        before_zone, a_brake_req, a_lat_req, cap_brake, cap_lat,
        margin_brake, margin_lat, action, steer_penalty=float(steer_penalty))

    maneuver = {"type": "none"}
    a_applied = None
    lateral_applied = None
    dur_steer = None
    if action == "brake":
        if a_brake_req is not None and float(a_brake_req) > 0:
            a_applied = round(min(float(a_brake_req) * float(safety_factor), cap_brake), 3)
        else:
            a_applied = round(cap_brake, 3)   # occurrence / 无解：全力制动
        _brake_evasion(tm, actor, frames, fps, f0, a_applied, va,
                       t_reaction=float(params["t_reaction_s"]))
        maneuver = {"type": "brake", "decel_mps2": a_applied}
    elif action == "steer":
        lat = float(steer_lateral_m or clearance)
        t_avail = (float(ttc) - float(params["t_reaction_s"])) if ttc is not None else 1.0
        dur_steer = float(min(1.2, max(0.3, 0.8 * t_avail))) if t_avail > 0 else 0.8
        _steer_evasion(tm, actor, frames, fps, f0, va, lat, dur=dur_steer,
                       t_reaction=float(params["t_reaction_s"]))
        lateral_applied = round(lat, 3)
        maneuver = {"type": "steer", "lateral_m": lateral_applied, "dur_s": round(dur_steer, 3)}

    after_zone, after_binding = _recompute_zone(tm, actor, others, frames, fps, params)
    after_collides = _pair_collides_window(tm, actor, worst_other, frames)
    avoided = bool(before_collides and not after_collides)

    plan = {
        "zone": before_zone, "binding": binding, "worst_other": worst_other,
        "a_brake_req_mps2": None if a_brake_req is None else round(float(a_brake_req), 3),
        "a_lat_req_mps2": None if a_lat_req is None else round(float(a_lat_req), 3),
        "lateral_clearance_m": round(float(clearance), 3),
        "ttc_s": None if ttc is None else round(float(ttc), 3),
        "margin_brake": margin_brake, "margin_lat": margin_lat,
        "delta_v_before_mps": round(v_rel_before, 3),
        "chosen": action,
        "a_applied_mps2": a_applied,
        "lateral_applied_m": lateral_applied,
        "steer_dur_s": dur_steer,
        "principle": principle,
    }

    zl = {ZONE_SAFE: "安全", ZONE_WARNING: "预警", ZONE_DANGER: "危险",
          ZONE_OCCURRENCE: "发生", ZONE_IMPACT: "撞击"}
    mdesc = maneuver["type"]
    if maneuver["type"] == "brake":
        mdesc = "制动 %.2f m/s²" % maneuver["decel_mps2"]
    elif maneuver["type"] == "steer":
        mdesc = "转向避让 %.2f m（%.1fs 内完成）" % (maneuver["lateral_m"], maneuver["dur_s"])
    if action == "none":
        outcome = "无需规避（%s 阶段）" % zl.get(before_zone, before_zone)
    elif avoided:
        outcome = "规避成功（窗口内不再相撞）"
    elif before_zone == ZONE_OCCURRENCE and after_collides:
        outcome = "已最小化损失（发生区不可避免，只能降低 Δv）"
    elif before_collides and after_collides:
        outcome = "未规避（仍相撞）"
    elif not before_collides:
        outcome = "本无接触冲突"
    else:
        outcome = "未改善"
    story = "在【%s】阶段对 T%d 施加【%s】：%s" % (
        zl.get(before_zone, before_zone), actor, mdesc, outcome)
    return {"ok": True, "actor": actor, "action": action, "maneuver": maneuver,
            "plan": plan,
            "before_zone": before_zone, "after_zone": after_zone,
            "before_binding": binding, "after_binding": after_binding,
            "before_collides": bool(before_collides), "after_collides": bool(after_collides),
            "avoided": avoided, "story": story}
