"""统一参数化生成引擎（motion primitives + 声明式场景计划）。

把事故生成拆成两层，替换 `corner_case` 里逐类型硬编码的 `_gen_*`：

1. **运动原语（primitives）**：可复用的轨迹写入逻辑（布局、追击碰撞、横移、刹停、静止）。
2. **场景计划（plans）**：声明式地"组合原语 + 采样参数"来描述一种事故类型。

新增/调事故类型时，只需写一个 `plan_xxx(ctx)`，复用原语，不再新写一套硬编码循环。

- 已迁移：rear-end / head-on / intersection-tbone / lane-change-cutin /
  hard-brake / cut-out-reveal（覆盖主要原语）。
- 尚未迁移（仍走 corner_case._gen_* 的复合场景）：pedestrian-crossing、
  chain-reaction-rear-end / cutin-brake-pileup / occluded-pedestrian-pileup。
"""
from __future__ import annotations

import numpy as np

from corner_case import (
    _apply_initial_pair_layout,
    _apply_brake_decel,
    _brake_track_after,
    _simulate_pursuit_collision,
    _make_intercepting_pedestrian,
    _retarget_pedestrian,
    _simulate_pedestrian,
    _ensure_participant,
    _ensure_chain_roles,
    _ensure_cutin_roles,
    _ensure_occluded_roles,
    _obb_radius_along,
    _sample_float,
    _severity_scale,
    _ground_right,
    _heading,
    _speed_mps,
    _get_dimensions,
    _pose_with_center,
    _write_pose,
    _smoothstep,
)


class _Ctx:
    """执行上下文：解析角色、采样参数，收集 affected/synth/sim/collision_pair。"""

    def __init__(self, tm, scenario, roles, frames, fps, intensity, enable_physics, sampling):
        self.tm = tm
        self.scenario = scenario
        self.frames = frames
        self.fps = fps
        self.intensity = intensity
        self.enable_physics = enable_physics
        self.sampling = sampling
        self.roles = {k: (int(v) if v is not None else None) for k, v in roles.items()}
        self.synth: list = []
        self.affected: list = []
        self.collision_pair = None
        self.sim = None
        self.natural = None   # 自然冲突生成的附加结果（冲突类型/速度因子等）

    def ensure(self, key, anchor_key=None):
        """若角色未指定则自动合成；返回 track_id。"""
        tid = self.roles.get(key)
        if tid is not None:
            return int(tid)
        anchor = self.roles.get(anchor_key)
        if anchor is None:
            raise ValueError(f"{self.scenario} 缺少角色 '{key}'（且无锚点 '{anchor_key}'）")
        tid, made = _ensure_participant(self.tm, None, self.scenario, key, anchor,
                                        self.frames, self.fps, self.intensity)
        if made:
            self.synth.append(int(tid))
        self.roles[key] = int(tid)
        return int(tid)

    def finish(self):
        return (self.affected, self.synth, self.collision_pair, self.sim, self.natural)


# ==================== 运动原语 ====================

def prim_regularize(ctx, track, speed=None, heading=None):
    """把 track 在整段生成窗口内的轨迹重写为"从起始帧出发的匀速直线"。

    这是关键一步：corner case 生成必须从**干净的基轨迹**出发，不能沿用原始数据里的
    噪声/稀疏轨迹——否则 (a) 动力学门槛必然超限（原始噪声加速度可达数百 m/s²），
    (b) 帧覆盖率不足导致标注一致性不过。朝向保留起始帧位姿的旋转部分。
    """
    f0 = ctx.frames[0]
    pose0 = ctx.tm.get_track_pose(track, f0)
    if pose0 is None:
        return False
    pose0 = np.asarray(pose0, dtype=np.float32).copy()
    c0 = pose0[:3, 3].copy()
    if heading is None:
        heading = _heading(ctx.tm, track, f0)
    heading = np.asarray(heading, dtype=np.float32).copy()
    heading[1] = 0.0
    hn = float(np.linalg.norm(heading))
    heading = np.array([1.0, 0.0, 0.0], dtype=np.float32) if hn < 1e-6 else heading / hn
    if speed is None:
        speed = max(_speed_mps(ctx.tm, track, f0, ctx.fps), 0.0)
    dt = 1.0 / ctx.fps
    for i, f in enumerate(ctx.frames):
        c = c0 + heading * (float(speed) * dt * i)
        c[1] = c0[1]
        _write_pose(ctx.tm, track, f, _pose_with_center(pose0, c))
    return True


def prim_layout_pair(ctx, rear, front, gap, lateral, rear_speed, front_speed, front_heading_sign=1.0):
    """把两车沿 rear 航向摆成 gap/lateral/速度（同时写满整段窗口，天然覆盖 100%）。"""
    _apply_initial_pair_layout(ctx.tm, rear, front, ctx.frames, ctx.fps, gap,
                               lateral_offset_m=lateral,
                               rear_speed_mps=rear_speed,
                               front_speed_mps=front_speed,
                               front_heading_sign=front_heading_sign)


def prim_pursue(ctx, attacker, victim, severity_scale=1.0, victim_brake_decel=0.0,
                victim_brake_frame=None, attacker_lateral=0.0, victim_lateral=0.0,
                lateral_ramp=(0.0, 1.0)):
    """限加速度追击碰撞（含可选前车制动 + 可选横向偏移）。返回 sim dict。"""
    return _simulate_pursuit_collision(
        ctx.tm, attacker, victim, ctx.frames, ctx.fps, ctx.intensity * severity_scale,
        enable_physics=ctx.enable_physics,
        victim_brake_decel=victim_brake_decel,
        victim_brake_frame=victim_brake_frame,
        attacker_lateral=attacker_lateral,
        victim_lateral=victim_lateral,
        lateral_ramp=lateral_ramp,
    )


def prim_lateral_sweep(ctx, track, lat_from, lat_to, delay_s=0.0, duration=1.0,
                       direction=None):
    """沿 direction（默认物体右向量）把横向量从 lat_from 平滑渐变到 lat_to。"""
    n = len(ctx.frames)
    dt_total = max(0.1, n / ctx.fps)
    if direction is None:
        direction = _ground_right(_heading(ctx.tm, track, ctx.frames[0]))
    direction = np.asarray(direction, dtype=np.float32)
    direction[1] = 0.0
    for i, f in enumerate(ctx.frames):
        base_pose = ctx.tm.get_track_pose(track, f)
        if base_pose is None:
            continue
        base_c = np.asarray(base_pose, dtype=np.float32)[:3, 3].copy()
        t = i / max(1, n - 1)
        u = (t - delay_s / dt_total) / max(1e-6, duration)
        sm = _smoothstep(max(0.0, min(1.0, u)))
        off = float(lat_from) + (float(lat_to) - float(lat_from)) * sm
        new_c = base_c + direction * off
        new_c[1] = base_c[1]
        _write_pose(ctx.tm, track, f, _pose_with_center(base_pose, new_c))


def prim_lateral_shift(ctx, track, lateral, delay_s=0.0, duration=1.0):
    """把 track 沿其右侧做 smoothstep 横移（变道/闪开）。

    横向"右"向量**在起始帧固定一次**（不用逐帧航向），否则噪声轨迹会让横移方向乱跳、
    产生巨额加速度。要求调用方先把轨迹 `prim_regularize` 成干净基轨迹。
    """
    prim_lateral_sweep(ctx, track, 0.0, lateral, delay_s=delay_s, duration=duration)


def prim_brake_to_stop(ctx, track, decel, begin_frame):
    """从 begin_frame 起以 decel 减速。"""
    _apply_brake_decel(ctx.tm, track, ctx.frames, ctx.fps, begin_frame, decel)


def prim_hold_still(ctx, track):
    """把 track 冻结在首帧位置（静止障碍）。"""
    f0 = ctx.frames[0]
    pose0 = ctx.tm.get_track_pose(track, f0)
    if pose0 is None:
        return
    c0 = np.array(pose0, dtype=np.float32)[:3, 3].copy()
    for f in ctx.frames:
        p = ctx.tm.get_track_pose(track, f)
        if p is not None:
            _write_pose(ctx.tm, track, f, _pose_with_center(p, c0))


# ==================== 场景计划 ====================

def plan_rear_end(ctx: _Ctx):
    s = ctx.sampling
    severity = _severity_scale(s, ctx.intensity)
    if ctx.roles.get("attacker") is None and ctx.roles.get("victim") is None:
        raise ValueError("追尾事故至少需要指定肇事车或受害车")
    if ctx.roles.get("attacker") is None:
        ctx.roles["attacker"] = ctx.ensure("attacker", "victim")
    if ctx.roles.get("victim") is None:
        ctx.roles["victim"] = ctx.ensure("victim", "attacker")
    attacker, victim = int(ctx.roles["attacker"]), int(ctx.roles["victim"])

    gap = _sample_float(s, "initial_gap_m", 12.0)
    rel_speed = _sample_float(s, "relative_speed_mps", 6.0) * severity
    lateral = _sample_float(s, "lateral_offset_m", 0.0)
    decel = _sample_float(s, "brake_decel_mps2", 6.0)
    reaction_delay = _sample_float(s, "reaction_delay_s", 0.4)
    victim_speed = max(_speed_mps(ctx.tm, victim, ctx.frames[0], ctx.fps), 2.0)
    attacker_speed = max(victim_speed + rel_speed, 3.0)
    if s.get("collision_severity") == "near_miss":
        lateral = lateral if abs(lateral) >= 0.8 else (0.9 if lateral >= 0 else -0.9)
    gap = max(gap, rel_speed * 0.6)

    prim_layout_pair(ctx, attacker, victim, gap, lateral, attacker_speed, victim_speed)
    brake_frame = ctx.frames[min(len(ctx.frames) - 1, max(0, int(round(reaction_delay * ctx.fps))))]
    ctx.sim = prim_pursue(ctx, attacker, victim, severity, decel, brake_frame)
    ctx.collision_pair = (attacker, victim)
    ctx.affected = [attacker, victim]


def plan_head_on(ctx: _Ctx):
    s = ctx.sampling
    severity = _severity_scale(s, ctx.intensity)
    if ctx.roles.get("attacker") is None:
        raise ValueError("对向碰撞需要至少指定肇事车")
    ctx.roles["attacker"] = int(ctx.roles["attacker"])
    ctx.roles["victim"] = ctx.ensure("victim", "attacker")
    attacker, victim = int(ctx.roles["attacker"]), int(ctx.roles["victim"])

    gap = _sample_float(s, "initial_gap_m", 30.0)
    rel_speed = _sample_float(s, "relative_speed_mps", 12.0) * severity
    lateral = _sample_float(s, "lateral_offset_m", 0.0)
    if s.get("collision_severity") == "near_miss":
        lateral = lateral if abs(lateral) >= 0.8 else (0.9 if lateral >= 0 else -0.9)
    speed_each = max(rel_speed / 2.0, 3.0)

    prim_layout_pair(ctx, attacker, victim, gap, lateral, speed_each, speed_each, front_heading_sign=-1.0)
    ctx.sim = prim_pursue(ctx, attacker, victim, severity)
    ctx.collision_pair = (attacker, victim)
    ctx.affected = [attacker, victim]


def plan_tbone(ctx: _Ctx):
    s = ctx.sampling
    severity = _severity_scale(s, ctx.intensity)
    if ctx.roles.get("attacker") is None:
        raise ValueError("路口侧碰需要至少指定冲撞车")
    ctx.roles["attacker"] = int(ctx.roles["attacker"])
    ctx.roles["victim"] = ctx.ensure("victim", "attacker")
    attacker, victim = int(ctx.roles["attacker"]), int(ctx.roles["victim"])

    lateral = _sample_float(s, "lateral_offset_m", 0.0)
    rel_speed = _sample_float(s, "relative_speed_mps", 8.0) * severity
    gap = max(10.0, rel_speed * max(1.0, len(ctx.frames) / ctx.fps) * 0.45)
    base_speed = max(_speed_mps(ctx.tm, attacker, ctx.frames[0], ctx.fps), 3.0)

    prim_layout_pair(ctx, attacker, victim, gap, lateral, base_speed + rel_speed,
                     max(base_speed * 0.6, 2.0))
    ctx.sim = prim_pursue(ctx, attacker, victim, severity)
    ctx.collision_pair = (attacker, victim)
    ctx.affected = [attacker, victim]


def plan_lane_change(ctx: _Ctx):
    """变道加塞：cutter 从相邻车道强行并入 target 所在车道（前车加塞）→ 追尾/刮擦。

    做法（与 plan_rear_end 同源，但多一个"从邻道并进来"的横向过程）：
      1. 布局：cutter 位于 target **前方 gap** 处、**相邻车道**（横向 side*lane_off），
         两车各自匀速（cutter 略慢，让 target 逐渐逼近）——干净基轨迹，覆盖率 100%；
      2. 追击：让 target 追 cutter（`prim_pursue`），同时在仿真内部给 cutter 叠加一个
         **横向并入**偏移（`victim_lateral`，0 → -side*lane_off），即"加塞车挤进 target 车道"；
      3. 碰撞/险情由追击动力学自然产生（不再依赖原始噪声轨迹的偶然接近）。
    """
    s = ctx.sampling
    if ctx.roles.get("cutter") is None:
        raise ValueError("变道加塞需要指定加塞车")
    ctx.roles["cutter"] = int(ctx.roles["cutter"])
    ctx.roles["target"] = ctx.ensure("target", "cutter")
    cutter = int(ctx.roles["cutter"])
    target = int(ctx.roles["target"]) if ctx.roles.get("target") is not None else None

    prim_regularize(ctx, cutter)
    if target is not None:
        prim_regularize(ctx, target)
    if target is None or not ctx.enable_physics:
        ctx.affected = [cutter] if target is None else [cutter, target]
        ctx.collision_pair = (cutter, target)
        return

    f0 = ctx.frames[0]
    severity = _severity_scale(s, ctx.intensity)
    # 一条车道 ≈3.5m（按 intensity 缩放），限制在 2.4~5.0m
    lane_off = abs(_sample_float(s, "lateral_offset_m", 3.5 * ctx.intensity)) * severity
    lane_off = max(2.4, min(5.0, lane_off))
    gap = max(4.0, _sample_float(s, "initial_gap_m", 6.0))
    rel = max(1.2, _sample_float(s, "relative_speed_mps", 4.0)) * severity
    delay = max(0.0, _sample_float(s, "reaction_delay_s", 0.2))

    tgt_speed = max(_speed_mps(ctx.tm, target, f0, ctx.fps), 6.0)
    cut_speed = max(tgt_speed - rel, 2.5)

    # 切入方向：默认从 cutter 当前所在的一侧切入（几何决定），几何退化时取右侧
    right = _ground_right(_heading(ctx.tm, target, f0))
    pc = ctx.tm.get_track_pose(cutter, f0)
    pt = ctx.tm.get_track_pose(target, f0)
    side = 1.0
    if pc is not None and pt is not None:
        d = np.asarray(pc, dtype=np.float32)[:3, 3] - np.asarray(pt, dtype=np.float32)[:3, 3]
        d[1] = 0.0
        u = float(np.dot(d, right))
        if abs(u) >= 0.5:
            side = 1.0 if u > 0 else -1.0

    # 1) 布局：cutter 在 target 前方 gap、相邻车道
    prim_layout_pair(ctx, target, cutter, gap, side * lane_off, tgt_speed, cut_speed)
    # 2) 追击 + 横向并入：target 追 cutter，cutter 同时挤进 target 车道
    ramp = (min(0.45, delay / max(0.1, len(ctx.frames) / ctx.fps)),
            min(0.95, delay / max(0.1, len(ctx.frames) / ctx.fps) + 0.5))
    ctx.sim = prim_pursue(ctx, target, cutter, severity,
                          victim_lateral=-side * lane_off, lateral_ramp=ramp)
    ctx.affected = [cutter, target]
    ctx.collision_pair = (target, cutter)


def plan_hard_brake(ctx: _Ctx):
    s = ctx.sampling
    if ctx.roles.get("braker") is None:
        raise ValueError("紧急刹车需要指定急刹车辆")
    braker = int(ctx.roles["braker"])
    if ctx.tm.get_track_pose(braker, ctx.frames[0]) is None:
        raise ValueError("该帧无此车辆")
    begin_idx = min(len(ctx.frames) - 1, max(0, int(round(_sample_float(s, "reaction_delay_s", 0.0) * ctx.fps))))
    begin_frame = ctx.frames[begin_idx]
    decel = _sample_float(s, "brake_decel_mps2", 6.0) * _severity_scale(s, ctx.intensity)
    # 先用干净基轨迹铺满整段窗口（否则制动帧之前的原始噪声轨迹会拉爆动力学/覆盖率）
    prim_regularize(ctx, braker)
    prim_brake_to_stop(ctx, braker, decel, begin_frame)
    ctx.affected = [braker]


def plan_cut_out(ctx: _Ctx):
    s = ctx.sampling
    if ctx.roles.get("blocker") is None:
        raise ValueError("该场景需要指定前方遮挡车")
    ctx.roles["blocker"] = int(ctx.roles["blocker"])
    ctx.roles["obstacle"] = ctx.ensure("obstacle", "blocker")
    blocker, obstacle = int(ctx.roles["blocker"]), int(ctx.roles["obstacle"])

    lateral = _sample_float(s, "lateral_offset_m", 3.5 * ctx.intensity)
    delay = _sample_float(s, "reaction_delay_s", 0.4)
    prim_regularize(ctx, blocker)
    prim_regularize(ctx, obstacle)
    prim_lateral_shift(ctx, blocker, lateral, delay_s=delay, duration=1.2)
    prim_hold_still(ctx, obstacle)
    ctx.affected = [blocker, obstacle]
    ctx.collision_pair = (blocker, obstacle)


def plan_pedestrian(ctx: _Ctx):
    s = ctx.sampling
    if ctx.roles.get("vehicle") is None:
        raise ValueError("行人横穿需要指定受影响车辆")
    vehicle = int(ctx.roles["vehicle"])
    prim_regularize(ctx, vehicle)          # 车辆先铺成干净基轨迹
    ped_speed = _sample_float(s, "pedestrian_speed_mps", 1.6 * max(0.6, ctx.intensity))
    sampled_intensity = max(0.3, ped_speed / 1.6) * _severity_scale(s, 1.0)
    pedestrian = ctx.roles.get("pedestrian")
    if pedestrian is None:
        pedestrian = _make_intercepting_pedestrian(ctx.tm, vehicle, ctx.frames, ctx.fps, sampled_intensity)
        ctx.synth.append(int(pedestrian))
    else:
        _retarget_pedestrian(ctx.tm, pedestrian, vehicle, ctx.frames, ctx.fps, sampled_intensity)
    collided = _simulate_pedestrian(ctx.tm, vehicle, pedestrian, ctx.frames, ctx.fps, sampled_intensity)
    ctx.roles["pedestrian"] = int(pedestrian)
    ctx.sim = collided if isinstance(collided, dict) else None
    ctx.collision_pair = (vehicle, int(pedestrian))
    ctx.affected = [vehicle, int(pedestrian)]


def plan_chain(ctx: _Ctx):
    """三车连环追尾：rear -> middle -> lead。"""
    lead, middle, rear, synth = _ensure_chain_roles(ctx.tm, ctx.roles, ctx.frames, ctx.fps, ctx.intensity)
    ctx.synth.extend(int(t) for t in synth)
    for _t in (lead, middle, rear):
        prim_regularize(ctx, _t)
    stop_frame = ctx.frames[len(ctx.frames) // 4]
    _brake_track_after(ctx.tm, lead, ctx.frames, ctx.fps, stop_frame, severity=1.1)
    sim1 = prim_pursue(ctx, rear, middle, 1.0)
    sim2 = prim_pursue(ctx, middle, lead, 0.9)
    ctx.affected = [int(lead), int(middle), int(rear)]
    ctx.collision_pair = (int(middle), int(lead))
    ctx.sim = sim2 if (sim2 or {}).get("collided") else sim1


def plan_cutin_brake(ctx: _Ctx):
    """加塞急刹追尾：cutter 从相邻车道并入目标车前方 → target 急刹 → follower 追尾。

    关键：**目标车只被一次追击仿真改写**；cutter 只做"布局 + 横向并入"，不参与追击。
    """
    s = ctx.sampling
    cutter, target, follower, synth = _ensure_cutin_roles(ctx.tm, ctx.roles, ctx.frames, ctx.fps, ctx.intensity)
    ctx.synth.extend(int(t) for t in synth)
    for _t in (cutter, target, follower):
        prim_regularize(ctx, _t)

    f0 = ctx.frames[0]
    dims_t = _get_dimensions(ctx.tm, target)
    dims_c = _get_dimensions(ctx.tm, cutter)
    gap = dims_t[0] / 2.0 + dims_c[0] / 2.0 + 5.0      # 插到目标车前方（留出安全间距）
    tgt_speed = max(_speed_mps(ctx.tm, target, f0, ctx.fps), 2.0)
    cut_speed = tgt_speed * 0.95
    # 1) 布局：cutter 在 target 前方 gap 处（同车道、干净基轨迹）
    prim_layout_pair(ctx, target, cutter, gap, 0.0, tgt_speed, cut_speed)
    # 2) 并入目标车道：横向量从"相邻车道"渐变到 0
    right = _ground_right(_heading(ctx.tm, target, f0))
    lane_off = 3.5 * ctx.intensity
    prim_lateral_sweep(ctx, cutter, -lane_off, 0.0, delay_s=0.15, duration=0.7, direction=right)
    # 3) 目标车急刹 + 后车追尾（制动在追击仿真内部）
    brake_frame = ctx.frames[int(len(ctx.frames) * 0.45)]
    ctx.sim = prim_pursue(ctx, follower, target, 1.0,
                          victim_brake_decel=6.0, victim_brake_frame=brake_frame)
    ctx.affected = [int(cutter), int(target), int(follower)]
    ctx.collision_pair = (int(follower), int(target))


def plan_natural_conflict(ctx: _Ctx):
    """自然冲突：保留两车原轨迹（a 左转还是左转、b 直行还是直行），只调时序/速度制造
    或避免冲突 —— 不再套模板把轨迹拉直。

    roles: ego（保留轨迹的主车）/ other（冲突车，可自动找）
    参数：outcome=collide|near_miss|avoid, time_gap_s, adjust=auto|ego|other, ego_brake_decel_mps2
    """
    import natural_conflict as nc
    s = ctx.sampling
    if ctx.roles.get("ego") is None:
        raise ValueError("自然冲突需要指定 ego（保留轨迹的那辆）")
    ego = int(ctx.roles["ego"])
    other = ctx.roles.get("other")
    if other is None:
        other = _pick_conflicting_track(ctx.tm, ego, ctx.frames, ctx.fps)
        if other is None:
            raise ValueError("找不到与 ego 轨迹交汇的其它车辆（可显式指定 other，或用模板化场景）")
        ctx.roles["other"] = int(other)
    other = int(other)

    outcome = s.get("outcome", "collide")
    time_gap = _sample_float(s, "time_gap_s", 0.4)
    adjust = s.get("adjust", "auto")
    brake = _sample_float(s, "ego_brake_decel_mps2", 6.0)

    res = nc.generate_natural_conflict(
        ctx.tm, ego, other, ctx.frames, ctx.fps,
        outcome=outcome, time_gap_s=time_gap, adjust=adjust, ego_brake_decel=brake)

    ctx.affected = [ego, other]
    ctx.collision_pair = (ego, other)
    ca = res.get("collision_analysis") or {}
    if ca:
        ctx.sim = {"collided": bool(ca.get("collided")),
                   "collision_frame": ca.get("collision_frame"),
                   "frames": list(ctx.frames), "fps": float(ctx.fps),
                   "attacker": int(ego), "victim": int(other),
                   "dims_a": _get_dimensions(ctx.tm, ego),
                   "dims_v": _get_dimensions(ctx.tm, other)}
    ctx.natural = res


def _pick_conflicting_track(tm, ego, frames, fps):
    """找一条与 ego 轨迹**交汇**的其它真实车辆（用于自然冲突的 other 角色）。"""
    import natural_conflict as nc
    path_ego = nc.extract_path(tm, ego, frames, fps)
    if not path_ego.get("valid"):
        return None
    best, best_dist = None, None
    skip = {int(ego)}
    skip |= set(getattr(tm, "track_deleted", set()) or set())
    skip |= set(getattr(tm, "track_replaced", set()) or set())
    skip |= set(getattr(tm, "synthetic_tracks", {}) or {})
    for tid in list(getattr(tm, "track_to_frames", {}) or {}):
        if int(tid) in skip:
            continue
        p = nc.extract_path(tm, tid, frames, fps)
        if not p.get("valid"):
            continue
        ap = nc._closest_approach(path_ego, p)
        if ap is None:
            continue
        if best_dist is None or ap["dist"] < best_dist:
            best, best_dist = int(tid), ap["dist"]
    return best if (best is not None and best_dist is not None
                    and best_dist <= nc.CONFLICT_MAX_DIST) else None


def plan_occluded_ped(ctx: _Ctx):
    """遮挡行人横穿追尾：vehicle / pedestrian / follower。"""
    vehicle, pedestrian, follower, synth = _ensure_occluded_roles(ctx.tm, ctx.roles, ctx.frames, ctx.fps, ctx.intensity)
    ctx.synth.extend(int(t) for t in synth)
    for _t in (vehicle, follower):
        prim_regularize(ctx, _t)
    ped_sim = _simulate_pedestrian(ctx.tm, vehicle, pedestrian, ctx.frames, ctx.fps, ctx.intensity)
    collision_frame = ped_sim.get("collision_frame") if isinstance(ped_sim, dict) else None
    if collision_frame is None:
        collision_frame = ctx.frames[int(len(ctx.frames) * 0.55)]
    _brake_track_after(ctx.tm, vehicle, ctx.frames, ctx.fps, collision_frame, severity=1.35)
    rear_sim = prim_pursue(ctx, follower, vehicle, 1.0)
    ctx.affected = [int(vehicle), int(pedestrian), int(follower)]
    ctx.collision_pair = (int(follower), int(vehicle))
    ctx.sim = rear_sim if (rear_sim or {}).get("collided") else ped_sim


# ==================== 执行入口 ====================

SCENARIO_PLANS = {
    "rear-end": plan_rear_end,
    "head-on": plan_head_on,
    "intersection-tbone": plan_tbone,
    "lane-change-cutin": plan_lane_change,
    "hard-brake": plan_hard_brake,
    "cut-out-reveal": plan_cut_out,
    "pedestrian-crossing": plan_pedestrian,
    "chain-reaction-rear-end": plan_chain,
    "cutin-brake-pileup": plan_cutin_brake,
    "occluded-pedestrian-pileup": plan_occluded_ped,
    "natural-conflict": plan_natural_conflict,
}


def execute(tm, scenario_type, roles, frames, fps, intensity, enable_physics, sampling):
    """执行某个场景计划，返回 (affected, synthesized, collision_pair, sim_result)。"""
    plan = SCENARIO_PLANS.get(scenario_type)
    if plan is None:
        return None  # 未迁移到引擎的场景，交由调用方走旧 _gen_*
    ctx = _Ctx(tm, scenario_type, roles, frames, fps, intensity, enable_physics, sampling)
    plan(ctx)
    return ctx.finish()
