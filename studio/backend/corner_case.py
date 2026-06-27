"""
Corner Case（交通事故场景）生成模块

底层逻辑：通过 **修改动态物体的轨迹**（写入 TrackManager 的 track 编辑）来制造
事故效果，与轨迹编辑/渲染管线完全统一。

核心特性：
- **两体前向物理仿真**：肇事车追击/接近受害车，在包围盒接触时按动量守恒发生
  非弹性碰撞，受害车被撞后获得速度并沿合理方向运动，之后双方因摩擦逐渐减速
  直至停止。整个过程逐帧积分，不会出现"瞬移"。
- **参与者自动合成**：若事故所需的某个参与者未指定，可基于已有物体克隆出一个
  合成参与者（synthetic track），自动放置初始轨迹并参与仿真。
- **碰撞关键帧识别**：识别最晚反应帧/碰撞帧，供自动驾驶评估。

坐标约定（与数据集一致）：
- 世界 Y 轴为竖直方向，车辆在 XZ 地平面运动；位姿 pose_world 为 4x4。
"""

import random
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from dggt.scene_edit.collision_physics import (
    check_collision,
    compute_critical_frame,
)


SCENARIOS = {
    "rear-end": {
        "name": "追尾事故",
        "desc": "后车加速撞向前车，前车被撞后向前位移",
        "roles": [
            {"key": "attacker", "label": "肇事车（后车）"},
            {"key": "victim", "label": "受害车（前车）", "optional": True, "auto": True},
        ],
    },
    "hard-brake": {
        "name": "紧急刹车",
        "desc": "目标车辆急减速直至停止",
        "roles": [
            {"key": "braker", "label": "急刹车辆"},
        ],
    },
    "lane-change-cutin": {
        "name": "变道加塞",
        "desc": "车辆向相邻车道横向切入",
        "roles": [
            {"key": "cutter", "label": "加塞车"},
            {"key": "target", "label": "被加塞车", "optional": True, "auto": True},
        ],
    },
    "intersection-tbone": {
        "name": "路口侧碰 (T-Bone)",
        "desc": "冲撞车横向撞向被撞车侧面，被撞车被侧推",
        "roles": [
            {"key": "attacker", "label": "冲撞车"},
            {"key": "victim", "label": "被撞车", "optional": True, "auto": True},
        ],
    },
    "head-on": {
        "name": "对向碰撞",
        "desc": "两车相向行驶直至碰撞，双方反弹减速",
        "roles": [
            {"key": "attacker", "label": "肇事车"},
            {"key": "victim", "label": "对向车", "optional": True, "auto": True},
        ],
    },
    "pedestrian-crossing": {
        "name": "行人横穿",
        "desc": "行人/物体突然横穿至车辆前方路径",
        "roles": [
            {"key": "pedestrian", "label": "横穿者", "optional": True, "auto": True},
            {"key": "vehicle", "label": "受影响车辆"},
        ],
    },
    "cut-out-reveal": {
        "name": "前车闪开露出障碍",
        "desc": "前车突然变道闪开，露出前方静止障碍车",
        "roles": [
            {"key": "blocker", "label": "前方遮挡车（闪开）"},
            {"key": "obstacle", "label": "被遮挡的静止障碍车", "optional": True, "auto": True},
        ],
    },
    "chain-reaction-rear-end": {
        "name": "连环追尾（三车）",
        "desc": "前车急停，中车被后车撞击后继续撞向前车，形成三车连环追尾",
        "roles": [
            {"key": "lead", "label": "前车（急停）", "optional": True, "auto": True},
            {"key": "middle", "label": "中车（被夹击）", "optional": True, "auto": True},
            {"key": "rear", "label": "后车（肇事车）", "optional": True, "auto": True},
        ],
    },
    "cutin-brake-pileup": {
        "name": "加塞急刹追尾（三车）",
        "desc": "加塞车突然切入并急刹，目标车急减速，后方跟车追尾目标车",
        "roles": [
            {"key": "cutter", "label": "加塞车", "optional": True, "auto": True},
            {"key": "target", "label": "被加塞目标车", "optional": True, "auto": True},
            {"key": "follower", "label": "后方跟车", "optional": True, "auto": True},
        ],
    },
    "occluded-pedestrian-pileup": {
        "name": "遮挡行人横穿追尾（车-人-车）",
        "desc": "前方车辆因被遮挡行人横穿急刹，后方跟车反应不及追尾，涉及行人和两车",
        "roles": [
            {"key": "vehicle", "label": "前车/受影响车辆", "optional": True, "auto": True},
            {"key": "pedestrian", "label": "横穿行人", "optional": True, "auto": True},
            {"key": "follower", "label": "后方跟车", "optional": True, "auto": True},
        ],
    },
}


def list_scenarios():
    out = []
    for key, s in SCENARIOS.items():
        out.append({
            "type": key,
            "name": s["name"],
            "desc": s["desc"],
            "roles": s["roles"],
            "sampling_schema": s.get("sampling_schema", {}),
        })
    return out


# ==================== 几何 / 运动学辅助 ====================

def _heading(tm, track_id, frame_idx):
    """估计某 track 在该帧的行进方向（地面 XZ 平面，单位向量）。"""
    frames = tm.get_track_frames(track_id)
    if not frames:
        return np.array([1.0, 0.0, 0.0])
    f1 = frame_idx + 1 if (frame_idx + 1) in frames else frame_idx
    fm = frame_idx - 1 if (frame_idx - 1) in frames else frame_idx
    p1 = tm.get_track_pose(track_id, f1)
    p0 = tm.get_track_pose(track_id, fm)
    if p1 is None or p0 is None:
        return np.array([1.0, 0.0, 0.0])
    d = np.array(p1)[:3, 3] - np.array(p0)[:3, 3]
    d[1] = 0.0
    n = np.linalg.norm(d)
    if n < 1e-4:
        pose = np.array(tm.get_track_pose(track_id, frame_idx))
        d = pose[:3, 2].copy()
        d[1] = 0.0
        n = np.linalg.norm(d)
        if n < 1e-4:
            return np.array([1.0, 0.0, 0.0])
    return d / n


def _speed_mps(tm, track_id, frame_idx, fps):
    """估计某 track 在该帧的速率（米/秒）。"""
    frames = tm.get_track_frames(track_id)
    if len(frames) < 2:
        return 0.0
    f1 = frame_idx + 1 if (frame_idx + 1) in frames else frame_idx
    fm = frame_idx - 1 if (frame_idx - 1) in frames else frame_idx
    p1 = tm.get_track_pose(track_id, f1)
    p0 = tm.get_track_pose(track_id, fm)
    if p1 is None or p0 is None:
        return 0.0
    d = np.array(p1)[:3, 3] - np.array(p0)[:3, 3]
    d[1] = 0.0
    span = max(1, (f1 - fm))
    return float(np.linalg.norm(d) / (span / fps))


def _ground_right(heading):
    """地面上垂直于 heading 的右向量（绕竖直 Y 轴）。"""
    up = np.array([0.0, -1.0, 0.0])
    r = np.cross(up, heading)
    n = np.linalg.norm(r)
    if n < 1e-4:
        return np.array([0.0, 0.0, 1.0])
    return r / n


def _pose_with_center(base_pose, center):
    p = np.array(base_pose, dtype=np.float32).copy()
    p[0, 3] = center[0]
    p[1, 3] = center[1]
    p[2, 3] = center[2]
    return p


def _get_dimensions(tm, track_id):
    """获取某 track 的包围盒尺寸 [length, width, height]，缺省给小汽车尺寸。"""
    dims = tm.get_track_dimensions(track_id)
    if dims and len(dims) >= 3:
        return [float(dims[0]), float(dims[1]), float(dims[2])]
    return [4.5, 2.0, 1.6]


def _estimate_mass(dims):
    """由包围盒体积估计质量（kg）。行人/小目标轻，车辆重。"""
    l, w, h = (abs(dims[0]), abs(dims[1]), abs(dims[2])) if len(dims) >= 3 else (4.5, 2.0, 1.6)
    vol = max(0.1, l * w * h)
    if vol < 2.0:       # 行人 / 自行车
        return 80.0
    return float(np.clip(120.0 * vol, 800.0, 12000.0))


def _vehicle_dimensions_from_reference(tm, ref_track=None, fallback=None):
    """返回合成车辆包围盒尺寸，让旧 bbox 画法下的框贴近克隆车体。"""
    candidates = []
    if ref_track is not None:
        candidates.append(_get_dimensions(tm, ref_track))
    if fallback is not None:
        candidates.append(fallback)
    for t in tm.list_tracks():
        candidates.append(t.get("dimensions") or [])
    for dims in candidates:
        if dims and len(dims) >= 3:
            l, w, h = [float(abs(v)) for v in dims[:3]]
            if l >= 2.5 and w >= 1.2 and h >= 1.0:
                # _get_bbox_corners_2d 的旧约定直接把 dimensions 映射到本地 XYZ。
                # 为了只修合成车、不影响原车框，把车辆 [length,width,height]
                # 转成旧画法下更贴近车体的 [width,height,length]。
                return [w, h, l]
    return [2.0, 1.7, 4.6]


def _pedestrian_dimensions_from_reference(tm, donor_track=None):
    """行人/小目标尺寸，若 donor 是小目标则沿用，否则使用默认站立行人盒。"""
    if donor_track is not None:
        dims = _get_dimensions(tm, donor_track)
        if dims and len(dims) >= 3:
            vol = abs(dims[0] * dims[1] * dims[2])
            if vol < 2.0:
                return [float(abs(dims[0])), float(abs(dims[1])), float(abs(dims[2]))]
    return [0.6, 0.6, 1.7]


def _smoothstep(t):
    return t * t * (3 - 2 * t)


def _write_pose(tm, track_id, frame_idx, pose):
    """把位姿写回：真实 track 用 set_track_pose，合成 track 用 set_synthetic_pose。"""
    if tm.is_synthetic(track_id):
        tm.set_synthetic_pose(track_id, frame_idx, pose)
    else:
        tm.set_track_pose(track_id, frame_idx, pose)


# ==================== 参与者自动合成 ====================

def _pick_donor(tm, prefer_track=None, prefer_pedestrian=False):
    """挑选一个用于克隆外观的 donor track。

    优先使用 prefer_track；行人场景优先挑选小尺寸物体；否则任选一个真实 track。
    """
    if prefer_track is not None and not tm.is_synthetic(prefer_track):
        return int(prefer_track)
    tracks = tm.list_tracks()
    if not tracks:
        raise ValueError("场景中没有可克隆的物体，无法自动合成参与者")
    if prefer_pedestrian:
        # 找体积最小的物体当行人
        best, best_vol = None, float("inf")
        for t in tracks:
            d = t.get("dimensions") or [4.5, 2.0, 1.6]
            if len(d) < 3:
                d = [4.5, 2.0, 1.6]
            vol = abs(d[0]) * abs(d[1]) * abs(d[2])
            if vol < best_vol:
                best_vol = vol
                best = t["track_id"]
        if best is not None:
            return int(best)
    return int(tracks[0]["track_id"])


def _default_anchor_track(tm):
    """当用户未指定任何角色时，选择场景中的第一个真实 track 作为布局锚点。"""
    tracks = tm.list_tracks()
    if not tracks:
        raise ValueError("场景中没有可用物体，无法自动生成 corner case 参与者")
    return int(tracks[0]["track_id"])


def _make_synthetic_relative(tm, anchor_track, frames, fps, offset_world, heading=None,
                             speed_scale=1.0, speed_mps=None, type_name="合成车辆",
                             dims=None, prefer_pedestrian=False):
    """相对锚点快速合成一个参与者。"""
    f0 = frames[0]
    anchor_pose = tm.get_track_pose(anchor_track, f0)
    if anchor_pose is None:
        raise ValueError("锚点物体在起始帧不存在，无法自动合成参与者")
    if heading is None:
        heading = _heading(tm, anchor_track, f0)
    if speed_mps is None:
        speed_mps = max(_speed_mps(tm, anchor_track, f0, fps), 3.0) * speed_scale
    donor = _pick_donor(
        tm,
        prefer_track=None if prefer_pedestrian else anchor_track,
        prefer_pedestrian=prefer_pedestrian,
    )
    if dims is None:
        dims = (_pedestrian_dimensions_from_reference(tm, donor)
                if prefer_pedestrian else
                _vehicle_dimensions_from_reference(tm, anchor_track, _get_dimensions(tm, donor)))
    poses = _synth_initial_poses(tm, anchor_pose, offset_world, heading, speed_mps, frames, fps)
    return tm.create_synthetic_track(donor, poses, dimensions=dims, type_name=type_name)


def _synth_initial_poses(tm, ref_pose, offset_world, heading, speed_mps, frames, fps):
    """生成合成参与者的初始（未碰撞）逐帧位姿：从 ref_pose 平移 offset，沿 heading 匀速。"""
    base = np.asarray(ref_pose, dtype=np.float32).copy()
    start_c = base[:3, 3] + np.asarray(offset_world, dtype=np.float32)
    dt = 1.0 / fps
    poses = {}
    for i, f in enumerate(frames):
        c = start_c + heading * (speed_mps * dt * i)
        poses[f] = _pose_with_center(base, c)
    return poses


def _ensure_participant(tm, role_track, scenario_type, role_key, anchor_track, frames, fps,
                        intensity):
    """确保某个参与者存在；若未指定（None），则自动合成并返回新建的 synth track_id。

    anchor_track: 已存在的参照物（用于决定合成者的相对位置/朝向/donor）。
    """
    if role_track is not None:
        return role_track, False  # 已指定，无需合成

    f0 = frames[0]
    anchor_pose = tm.get_track_pose(anchor_track, f0)
    if anchor_pose is None:
        raise ValueError("参照物在起始帧不存在，无法自动合成参与者")
    anchor_head = _heading(tm, anchor_track, f0)
    anchor_right = _ground_right(anchor_head)
    anchor_speed = max(_speed_mps(tm, anchor_track, f0, fps), 3.0)
    dims_anchor = _get_dimensions(tm, anchor_track)

    pedestrian = (role_key == "pedestrian")
    donor = _pick_donor(
        tm,
        prefer_track=None if pedestrian else anchor_track,
        prefer_pedestrian=pedestrian,
    )
    donor_dims = _get_dimensions(tm, donor)
    vehicle_dims = _vehicle_dimensions_from_reference(tm, anchor_track, donor_dims)
    ped_dims = _pedestrian_dimensions_from_reference(tm, donor)

    # 根据场景与角色决定合成者初始位置/朝向/速度
    if scenario_type == "rear-end" and role_key == "victim":
        # 前车：在肇事车前方一段距离，速度略低
        offset = anchor_head * (dims_anchor[0] + 12.0)
        poses = _synth_initial_poses(tm, anchor_pose, offset, anchor_head,
                                     anchor_speed * 0.5, frames, fps)
        dims = vehicle_dims
    elif scenario_type == "rear-end" and role_key == "attacker":
        offset = -anchor_head * (dims_anchor[0] + 12.0)
        poses = _synth_initial_poses(tm, anchor_pose, offset, anchor_head,
                                     anchor_speed * 1.3, frames, fps)
        dims = vehicle_dims
    elif scenario_type == "head-on" and role_key == "victim":
        # 对向车：前方较远，朝向相反，相向驶来
        offset = anchor_head * 30.0
        poses = _synth_initial_poses(tm, anchor_pose, offset, -anchor_head,
                                     anchor_speed, frames, fps)
        dims = vehicle_dims
    elif scenario_type == "intersection-tbone" and role_key == "victim":
        # 被撞车：从侧向驶来，穿过路口
        offset = anchor_right * 16.0 + anchor_head * (dims_anchor[0] * 1.5)
        poses = _synth_initial_poses(tm, anchor_pose, offset, -anchor_right,
                                     anchor_speed, frames, fps)
        dims = vehicle_dims
    elif scenario_type == "pedestrian-crossing" and role_key == "pedestrian":
        # 行人：在车辆前方路径侧边，横向穿出
        ahead = dims_anchor[0] * 2.0
        offset = anchor_head * ahead + anchor_right * 6.0
        ped_speed = 1.5 * max(0.6, intensity)  # 人速 ~1.5 m/s
        poses = _synth_initial_poses(tm, anchor_pose, offset, -anchor_right,
                                     ped_speed, frames, fps)
        dims = ped_dims
    elif scenario_type == "cut-out-reveal" and role_key == "obstacle":
        # 静止障碍车：在遮挡车前方，速度 0
        offset = anchor_head * (dims_anchor[0] + 14.0)
        poses = _synth_initial_poses(tm, anchor_pose, offset, anchor_head, 0.0, frames, fps)
        dims = vehicle_dims
    elif scenario_type == "lane-change-cutin" and role_key == "target":
        # 被加塞车：相邻车道、与加塞车同向
        offset = anchor_right * 3.5 + anchor_head * 4.0
        poses = _synth_initial_poses(tm, anchor_pose, offset, anchor_head,
                                     anchor_speed * 0.9, frames, fps)
        dims = vehicle_dims
    else:
        # 兜底：在参照物前方合成一个同向物体
        offset = anchor_head * 15.0
        poses = _synth_initial_poses(tm, anchor_pose, offset, anchor_head,
                                     anchor_speed, frames, fps)
        dims = vehicle_dims

    type_name = "合成行人" if pedestrian else "合成车辆"
    synth_id = tm.create_synthetic_track(donor, poses, dimensions=dims, type_name=type_name)
    return synth_id, True


# ==================== 两体物理仿真 ====================

def _simulate_pursuit_collision(tm, attacker, victim, frames, fps, intensity,
                                enable_physics=True,
                                victim_evasive=False):
    """通用两体追击-碰撞仿真（适用于追尾/侧碰/对向）。

    enable_physics=True：碰撞按动量守恒分配速度，受害车被撞后真实位移，
        双方因摩擦逐渐减速。
    enable_physics=False：碰撞后简单"粘连"——受害车被推到接触面外但不获得
        明显动量（不会被撞飞），用于对比物理效果。

    Returns: sim dict
    """
    dt = 1.0 / fps
    f0 = frames[0]

    dims_a = _get_dimensions(tm, attacker)
    dims_v = _get_dimensions(tm, victim)
    mass_a = _estimate_mass(dims_a)
    mass_v = _estimate_mass(dims_v)

    pose_a0 = tm.get_track_pose(attacker, f0)
    pose_v0 = tm.get_track_pose(victim, f0)
    if pose_a0 is None or pose_v0 is None:
        return {"collided": False, "collision_frame": None, "frames": list(frames),
                "fps": fps, "attacker": int(attacker), "victim": int(victim),
                "dims_a": dims_a, "dims_v": dims_v}

    pos_a = np.array(pose_a0, dtype=np.float32)[:3, 3].copy()
    pos_v = np.array(pose_v0, dtype=np.float32)[:3, 3].copy()
    y_a, y_v = pos_a[1], pos_v[1]  # 锁定高度

    head_a = _heading(tm, attacker, f0)
    speed_a0 = max(_speed_mps(tm, attacker, f0, fps), 2.0)
    speed_v0 = _speed_mps(tm, victim, f0, fps)
    head_v = _heading(tm, victim, f0)

    # 追击速度：保证能在窗口内追上（强度提升）
    n = len(frames)
    dist0 = np.linalg.norm((pos_v - pos_a) * np.array([1, 0, 1]))
    need_speed = dist0 / max(1e-3, (n * 0.55) * dt) + speed_v0
    vel_a = head_a * max(speed_a0, need_speed) * (0.9 + 0.3 * intensity)
    vel_v = head_v * speed_v0

    friction_decel = 7.0  # m/s^2 摩擦减速度
    collided = False
    collision_frame = None

    pose_a_cur = np.array(pose_a0, dtype=np.float32).copy()
    pose_v_cur = np.array(pose_v0, dtype=np.float32).copy()

    for i, f in enumerate(frames):
        if i > 0:
            if not collided:
                # 追击：肇事车持续指向受害车当前位置
                to_v = pos_v - pos_a
                to_v[1] = 0.0
                d = np.linalg.norm(to_v)
                if d > 1e-4:
                    aim = to_v / d
                    spd = np.linalg.norm(vel_a)
                    vel_a = aim * spd
                pos_a = pos_a + vel_a * dt
                pos_v = pos_v + vel_v * dt
            else:
                # 碰后：双方摩擦减速
                sa = np.linalg.norm(vel_a)
                sv = np.linalg.norm(vel_v)
                if sa > 1e-3:
                    vel_a = vel_a / sa * max(0.0, sa - friction_decel * dt)
                if sv > 1e-3:
                    vel_v = vel_v / sv * max(0.0, sv - friction_decel * dt)
                pos_a = pos_a + vel_a * dt
                pos_v = pos_v + vel_v * dt

            pos_a[1] = y_a
            pos_v[1] = y_v

        # 写入位姿（保持各自朝向，仅更新中心）
        pa = _pose_with_center(pose_a_cur, pos_a)
        pv = _pose_with_center(pose_v_cur, pos_v)

        # 碰撞检测（用当前帧位姿）
        if not collided and i > 0:
            hit, _ = check_collision(pa, dims_a, pv, dims_v)
            if hit:
                collided = True
                collision_frame = f
                normal = pos_v - pos_a
                normal[1] = 0.0
                nn = np.linalg.norm(normal)
                normal = normal / nn if nn > 1e-4 else head_a
                if enable_physics:
                    # 非弹性碰撞：动量守恒分配速度，受害车被撞飞
                    rel = np.dot(vel_a - vel_v, normal)
                    restitution = 0.15  # 接近完全非弹性
                    if rel > 0:  # 正在接近才有冲量
                        j = -(1 + restitution) * rel / (1 / mass_a + 1 / mass_v)
                        impulse = j * normal
                        vel_a = vel_a + impulse / mass_a
                        vel_v = vel_v - impulse / mass_v
                else:
                    # 无物理：双方碰撞后都停下（受害车不被撞飞）
                    vel_a = vel_a * 0.0
                    vel_v = vel_v * 0.0
                # 防穿模：把受害车沿法线推到接触面外
                contact = dims_a[0] / 2.0 + min(dims_v[0], dims_v[1]) / 2.0
                pos_v = pos_a + normal * contact
                pos_v[1] = y_v
                pv = _pose_with_center(pose_v_cur, pos_v)

        _write_pose(tm, attacker, f, pa)
        _write_pose(tm, victim, f, pv)

    return {"collided": collided, "collision_frame": collision_frame,
            "frames": list(frames), "fps": fps,
            "attacker": int(attacker), "victim": int(victim),
            "dims_a": dims_a, "dims_v": dims_v}


# ==================== 入口 ====================

def _analysis_from_sim(tm, sim, safety_margin=1.5):
    """根据仿真结果构建碰撞关键帧分析（碰撞帧由仿真直接给出，最晚反应帧反推）。"""
    if not sim or not sim.get("collided") or sim.get("collision_frame") is None:
        # 仿真未碰撞，退回基于距离的分析
        return None
    attacker = sim["attacker"]
    victim = sim["victim"]
    frames = sim["frames"]
    fps = sim["fps"]
    collision_frame = sim["collision_frame"]
    dims_a = sim["dims_a"]
    dims_v = sim["dims_v"]
    dt = 1.0 / fps

    collision_idx = frames.index(collision_frame) if collision_frame in frames else len(frames) - 1

    # 从碰撞帧向前找"最晚反应帧"：相对速度下刹车距离+安全裕度仍可避免
    critical_idx = collision_idx
    for i in range(collision_idx - 1, -1, -1):
        f = frames[i]
        pa = tm.get_track_pose(attacker, f)
        pv = tm.get_track_pose(victim, f)
        if pa is None or pv is None:
            continue
        ca = np.asarray(pa, dtype=np.float32)[:3, 3]
        cv = np.asarray(pv, dtype=np.float32)[:3, 3]
        distance = float(np.linalg.norm(ca - cv))
        va = _speed_mps(tm, attacker, f, fps)
        vv = _speed_mps(tm, victim, f, fps)
        rel_vel = abs(va - vv)
        braking = (rel_vel ** 2) / (2 * 6.0) if rel_vel > 0 else 0.0
        if distance > braking + safety_margin:
            critical_idx = i + 1
            break
        critical_idx = i

    critical_frame = frames[critical_idx]
    reaction_frames = collision_idx - critical_idx
    time_to_collision = reaction_frames * dt
    pa_c = tm.get_track_pose(attacker, critical_frame)
    pv_c = tm.get_track_pose(victim, critical_frame)
    dist_at_critical = float(np.linalg.norm(
        np.asarray(pa_c, dtype=np.float32)[:3, 3] - np.asarray(pv_c, dtype=np.float32)[:3, 3]
    )) if pa_c is not None and pv_c is not None else 0.0

    if reaction_frames <= 3:
        severity = "high"
    elif reaction_frames <= 6:
        severity = "medium"
    else:
        severity = "low"

    return {
        "critical_frame": int(critical_frame),
        "collision_frame": int(collision_frame),
        "time_to_collision": float(time_to_collision),
        "distance_at_critical": dist_at_critical,
        "collision_severity": severity,
        "reaction_frames": int(reaction_frames),
        "attacker": int(attacker),
        "victim": int(victim),
    }


def generate(tm, scenario_type, roles, start_frame, num_frames, intensity=1.0,
             enable_physics=True, fps=10.0, sampling_params=None, sampling_seed=None):
    """生成 corner case，写入 TrackManager 的 track 编辑/合成参与者。

    Returns: dict 结果摘要（含碰撞分析、关键帧、合成参与者列表）
    """
    if scenario_type not in SCENARIOS:
        raise ValueError(f"未知场景类型: {scenario_type}")

    end_frame = start_frame + num_frames
    frames = list(range(start_frame, end_frame + 1))
    sampling = _normalize_sampling_params(scenario_type, sampling_params, sampling_seed)

    affected = []
    synthesized = []           # 自动合成的 track_id 列表
    collision_pair = None
    sim_result = None          # 物理仿真结果（若有）

    if scenario_type == "rear-end":
        affected, synthesized, collision_pair, sim_result = _gen_rear_end(tm, roles, frames, fps, intensity, enable_physics, sampling)
    elif scenario_type == "hard-brake":
        affected = _gen_hard_brake(tm, roles, frames, intensity, fps=fps, sampling=sampling)
    elif scenario_type == "lane-change-cutin":
        affected, synthesized, collision_pair = _gen_lane_change(tm, roles, frames, fps, intensity, enable_physics, sampling)
    elif scenario_type == "intersection-tbone":
        affected, synthesized, collision_pair, sim_result = _gen_tbone(tm, roles, frames, fps, intensity, enable_physics, sampling)
    elif scenario_type == "head-on":
        affected, synthesized, collision_pair, sim_result = _gen_head_on(tm, roles, frames, fps, intensity, enable_physics, sampling)
    elif scenario_type == "pedestrian-crossing":
        affected, synthesized, collision_pair, sim_result = _gen_pedestrian_crossing(tm, roles, frames, fps, intensity, enable_physics, sampling)
    elif scenario_type == "cut-out-reveal":
        affected, synthesized, collision_pair = _gen_cut_out_reveal(tm, roles, frames, fps, intensity, enable_physics, sampling)
    elif scenario_type == "chain-reaction-rear-end":
        affected, synthesized, collision_pair, sim_result = _gen_chain_reaction_rear_end(tm, roles, frames, fps, intensity, enable_physics)
    elif scenario_type == "cutin-brake-pileup":
        affected, synthesized, collision_pair, sim_result = _gen_cutin_brake_pileup(tm, roles, frames, fps, intensity, enable_physics)
    elif scenario_type == "occluded-pedestrian-pileup":
        affected, synthesized, collision_pair, sim_result = _gen_occluded_pedestrian_pileup(tm, roles, frames, fps, intensity, enable_physics)

    result = {
        "scenario_type": scenario_type,
        "affected_tracks": affected,
        "synthesized_tracks": synthesized,
        "start_frame": start_frame,
        "num_frames": num_frames,
        "physics_enabled": enable_physics,
        "sampling_params": sampling,
        "sampling_seed": sampling_seed,
    }

    # 碰撞关键帧分析：优先用物理仿真结果，否则退回基于距离的检测
    collision_info = None
    if sim_result is not None:
        collision_info = _analysis_from_sim(tm, sim_result)
    if collision_info is None and collision_pair and collision_pair[0] is not None and collision_pair[1] is not None:
        collision_info = analyze_collision(tm, collision_pair[0], collision_pair[1], frames, fps=fps)
    if collision_info is not None:
        result["collision_analysis"] = collision_info
        result["critical_frame"] = collision_info.get("critical_frame")
        result["collision_frame"] = collision_info.get("collision_frame")
        if scenario_type in ("chain-reaction-rear-end", "cutin-brake-pileup", "occluded-pedestrian-pileup"):
            result["collision_tracks"] = [int(t) for t in affected]
        elif collision_pair:
            result["collision_tracks"] = [int(collision_pair[0]), int(collision_pair[1])]

    return result


def analyze_collision(tm, attacker, victim, frames, fps=10.0, safety_margin=1.5):
    """分析两参与者在帧区间内的碰撞，返回关键帧信息（含涉及的 track）。"""
    if attacker is None or victim is None:
        return None
    common = sorted(set(tm.get_track_frames(attacker)) & set(tm.get_track_frames(victim)) & set(frames))
    if not common:
        return None
    pa = [tm.get_track_pose(attacker, f) for f in common]
    pv = [tm.get_track_pose(victim, f) for f in common]
    if any(p is None for p in pa) or any(p is None for p in pv):
        pairs = [(f, a, v) for f, a, v in zip(common, pa, pv) if a is not None and v is not None]
        if not pairs:
            return None
        common = [p[0] for p in pairs]
        pa = [p[1] for p in pairs]
        pv = [p[2] for p in pairs]
    dims_a = _get_dimensions(tm, attacker)
    dims_v = _get_dimensions(tm, victim)
    info = compute_critical_frame(pa, dims_a, pv, dims_v, common,
                                  safety_margin=safety_margin, fps=fps)
    if info is not None:
        info["attacker"] = int(attacker)
        info["victim"] = int(victim)
    return info


# ==================== 各场景生成 ====================

def _gen_rear_end(tm, roles, frames, fps, intensity, enable_physics, sampling=None):
    attacker = roles.get("attacker")
    victim = roles.get("victim")
    if attacker is None and victim is None:
        raise ValueError("追尾事故至少需要指定肇事车或受害车")
    synth = []
    # 自动合成缺失参与者
    if attacker is None:
        attacker, made = _ensure_participant(tm, None, "rear-end", "attacker", victim, frames, fps, intensity)
        if made:
            synth.append(int(attacker))
    if victim is None:
        victim, made = _ensure_participant(tm, None, "rear-end", "victim", attacker, frames, fps, intensity)
        if made:
            synth.append(int(victim))

    sampling = sampling or {}
    severity = _severity_scale(sampling, intensity)
    gap = _sample_float(sampling, "initial_gap_m", 12.0)
    rel_speed = _sample_float(sampling, "relative_speed_mps", 6.0) * severity
    lateral = _sample_float(sampling, "lateral_offset_m", 0.0)
    decel = _sample_float(sampling, "brake_decel_mps2", 6.0)
    reaction_delay = _sample_float(sampling, "reaction_delay_s", 0.4)
    victim_speed = max(_speed_mps(tm, victim, frames[0], fps), 2.0)
    attacker_speed = max(victim_speed + rel_speed, 3.0)
    if sampling.get("collision_severity") == "near_miss":
        lateral = lateral if abs(lateral) >= 0.8 else (0.9 if lateral >= 0 else -0.9)
    _apply_initial_pair_layout(
        tm, attacker, victim, frames, fps, gap,
        lateral_offset_m=lateral,
        rear_speed_mps=attacker_speed,
        front_speed_mps=victim_speed,
    )
    brake_frame = frames[min(len(frames) - 1, max(0, int(round(reaction_delay * fps))))]
    _apply_brake_decel(tm, victim, frames, fps, brake_frame, decel)

    collided = _simulate_pursuit_collision(
        tm, attacker, victim, frames, fps, intensity * severity,
        enable_physics=enable_physics,
    )
    return [int(attacker), int(victim)], synth, (attacker, victim), collided


def _gen_tbone(tm, roles, frames, fps, intensity, enable_physics, sampling=None):
    attacker = roles.get("attacker")
    victim = roles.get("victim")
    if attacker is None:
        raise ValueError("路口侧碰需要至少指定冲撞车")
    synth = []
    if victim is None:
        victim, made = _ensure_participant(tm, None, "intersection-tbone", "victim", attacker, frames, fps, intensity)
        if made:
            synth.append(int(victim))
    sampling = sampling or {}
    severity = _severity_scale(sampling, intensity)
    lateral = _sample_float(sampling, "lateral_offset_m", 0.0)
    rel_speed = _sample_float(sampling, "relative_speed_mps", 8.0) * severity
    gap = max(10.0, rel_speed * max(1.0, len(frames) / fps) * 0.45)
    base_speed = max(_speed_mps(tm, attacker, frames[0], fps), 3.0)
    _apply_initial_pair_layout(
        tm, attacker, victim, frames, fps, gap,
        lateral_offset_m=lateral,
        rear_speed_mps=base_speed + rel_speed,
        front_speed_mps=max(base_speed * 0.6, 2.0),
    )
    collided = _simulate_pursuit_collision(tm, attacker, victim, frames, fps, intensity * severity, enable_physics=enable_physics)
    return [int(attacker), int(victim)], synth, (attacker, victim), collided


def _gen_head_on(tm, roles, frames, fps, intensity, enable_physics, sampling=None):
    attacker = roles.get("attacker")
    victim = roles.get("victim")
    if attacker is None:
        raise ValueError("对向碰撞需要至少指定肇事车")
    synth = []
    if victim is None:
        victim, made = _ensure_participant(tm, None, "head-on", "victim", attacker, frames, fps, intensity)
        if made:
            synth.append(int(victim))
    sampling = sampling or {}
    severity = _severity_scale(sampling, intensity)
    gap = _sample_float(sampling, "initial_gap_m", 30.0)
    rel_speed = _sample_float(sampling, "relative_speed_mps", 12.0) * severity
    lateral = _sample_float(sampling, "lateral_offset_m", 0.0)
    if sampling.get("collision_severity") == "near_miss":
        lateral = lateral if abs(lateral) >= 0.8 else (0.9 if lateral >= 0 else -0.9)
    speed_each = max(rel_speed / 2.0, 3.0)
    _apply_initial_pair_layout(
        tm, attacker, victim, frames, fps, gap,
        lateral_offset_m=lateral,
        rear_speed_mps=speed_each,
        front_speed_mps=speed_each,
        front_heading_sign=-1.0,
    )
    collided = _simulate_pursuit_collision(tm, attacker, victim, frames, fps, intensity * severity, enable_physics=enable_physics)
    return [int(attacker), int(victim)], synth, (attacker, victim), collided


def _gen_pedestrian_crossing(tm, roles, frames, fps, intensity, enable_physics, sampling=None):
    pedestrian = roles.get("pedestrian")
    vehicle = roles.get("vehicle")
    if vehicle is None:
        raise ValueError("行人横穿需要指定受影响车辆")
    synth = []
    sampling = sampling or {}
    ped_speed = _sample_float(sampling, "pedestrian_speed_mps", 1.6 * max(0.6, intensity))
    sampled_intensity = max(0.3, ped_speed / 1.6) * _severity_scale(sampling, 1.0)
    if pedestrian is None:
        pedestrian = _make_intercepting_pedestrian(tm, vehicle, frames, fps, sampled_intensity)
        synth.append(int(pedestrian))
    else:
        # 已指定行人：重排其轨迹去拦截车辆路径
        _retarget_pedestrian(tm, pedestrian, vehicle, frames, fps, sampled_intensity)

    collided = _simulate_pedestrian(tm, vehicle, pedestrian, frames, fps, sampled_intensity)
    sim = None
    if isinstance(collided, dict):
        sim = collided
    return [int(vehicle), int(pedestrian)], synth, (vehicle, pedestrian), sim


def _interception_setup(tm, vehicle, frames, fps, ped_speed):
    """计算行人拦截车辆路径的几何：返回 (起点, 横穿方向, 拦截帧)。"""
    n = len(frames)
    # 拦截发生在窗口约 60% 处
    fc_idx = int(n * 0.6)
    fc = frames[fc_idx]
    veh_pose_fc = tm.get_track_pose(vehicle, fc)
    f0 = frames[0]
    if veh_pose_fc is None:
        veh_pose_fc = tm.get_track_pose(vehicle, f0)
    veh_c_fc = np.array(veh_pose_fc, dtype=np.float32)[:3, 3].copy()
    veh_head = _heading(tm, vehicle, fc)
    cross_dir = _ground_right(veh_head)  # 垂直于车辆前向横穿
    dt = 1.0 / fps
    # 行人从一侧出发，在 fc 抵达车辆路径点
    travel = ped_speed * dt * fc_idx
    start = veh_c_fc + cross_dir * travel        # 起点在右侧 travel 处
    move_dir = -cross_dir                         # 朝车辆路径横穿
    return start, move_dir, fc


def _make_intercepting_pedestrian(tm, vehicle, frames, fps, intensity):
    """合成一个会拦截车辆路径的行人。"""
    ped_speed = 1.6 * max(0.6, intensity)
    start, move_dir, fc = _interception_setup(tm, vehicle, frames, fps, ped_speed)
    f0 = frames[0]
    ref_pose = tm.get_track_pose(vehicle, f0)
    y_ground = np.array(ref_pose, dtype=np.float32)[1, 3]
    dt = 1.0 / fps
    poses = {}
    base = np.array(ref_pose, dtype=np.float32).copy()
    for i, f in enumerate(frames):
        c = start + move_dir * (ped_speed * dt * i)
        c[1] = y_ground
        poses[f] = _pose_with_center(base, c)
    donor = _pick_donor(tm, prefer_pedestrian=True)
    donor_dims = _get_dimensions(tm, donor)
    dims = donor_dims if (donor_dims[0] * donor_dims[1] * donor_dims[2] < 2.0) else [0.6, 0.6, 1.7]
    return tm.create_synthetic_track(donor, poses, dimensions=dims, type_name="合成行人")


def _retarget_pedestrian(tm, pedestrian, vehicle, frames, fps, intensity):
    """把已指定的行人轨迹重排为拦截车辆路径。"""
    ped_speed = 1.6 * max(0.6, intensity)
    start, move_dir, fc = _interception_setup(tm, vehicle, frames, fps, ped_speed)
    f0 = frames[0]
    ped_pose0 = tm.get_track_pose(pedestrian, f0)
    if ped_pose0 is None:
        return
    y_ground = np.array(ped_pose0, dtype=np.float32)[1, 3]
    start[1] = y_ground
    dt = 1.0 / fps
    base = np.array(ped_pose0, dtype=np.float32).copy()
    for i, f in enumerate(frames):
        c = start + move_dir * (ped_speed * dt * i)
        c[1] = y_ground
        _write_pose(tm, pedestrian, f, _pose_with_center(base, c))


def _simulate_pedestrian(tm, vehicle, pedestrian, frames, fps, intensity):
    """行人横穿仿真：行人沿已设定的拦截轨迹前进；碰撞后行人被车辆带飞并减速。

    前置条件：pedestrian 的逐帧位姿已被设置为拦截车辆路径的轨迹
    （由 _make_intercepting_pedestrian / _retarget_pedestrian 写入）。
    本函数只负责检测碰撞并改写碰撞之后的帧（撞飞效果），不重算碰撞前轨迹。
    """
    dt = 1.0 / fps
    dims_v = _get_dimensions(tm, vehicle)
    dims_p = _get_dimensions(tm, pedestrian)

    # 预读行人既定轨迹
    preset = {f: np.array(tm.get_track_pose(pedestrian, f), dtype=np.float32)
              for f in frames if tm.get_track_pose(pedestrian, f) is not None}
    if not preset:
        return {"collided": False, "collision_frame": None, "frames": list(frames),
                "fps": fps, "attacker": int(vehicle), "victim": int(pedestrian),
                "dims_a": dims_v, "dims_v": dims_p}

    collided = False
    collision_frame = None
    vel_p = None
    pos_p = None
    pose_p_cur = None

    for i, f in enumerate(frames):
        veh_pose = tm.get_track_pose(vehicle, f)

        if not collided:
            # 碰撞前：沿既定拦截轨迹
            if f not in preset:
                continue
            pp = preset[f]
            if veh_pose is not None and i > 0:
                hit, _ = check_collision(np.array(veh_pose, dtype=np.float32), dims_v, pp, dims_p)
                if hit:
                    collided = True
                    collision_frame = f
                    pose_p_cur = pp.copy()
                    pos_p = pp[:3, 3].copy()
                    veh_head = _heading(tm, vehicle, f)
                    veh_speed = max(_speed_mps(tm, vehicle, f, fps), 5.0)
                    # 行人被撞获得车辆方向速度（被带飞）
                    vel_p = veh_head * veh_speed * 1.1
            # 写回（保持既定）
            _write_pose(tm, pedestrian, f, pp)
        else:
            # 碰撞后：沿被撞方向运动并减速
            sp = np.linalg.norm(vel_p)
            if sp > 1e-3:
                vel_p = vel_p / sp * max(0.0, sp - 5.0 * dt)
            pos_p = pos_p + vel_p * dt
            _write_pose(tm, pedestrian, f, _pose_with_center(pose_p_cur, pos_p))

    return {"collided": collided, "collision_frame": collision_frame,
            "frames": list(frames), "fps": fps,
            "attacker": int(vehicle), "victim": int(pedestrian),
            "dims_a": dims_v, "dims_v": dims_p}


def _gen_lane_change(tm, roles, frames, fps, intensity, enable_physics, sampling=None):
    cutter = roles.get("cutter")
    if cutter is None:
        raise ValueError("变道加塞需要指定加塞车")
    target = roles.get("target")
    synth = []
    if target is None:
        # 变道加塞默认不强制合成；仅当用户要求(role auto)时合成被加塞车
        target, made = _ensure_participant(tm, None, "lane-change-cutin", "target", cutter, frames, fps, intensity)
        if made:
            synth.append(int(target))

    sampling = sampling or {}
    f0 = frames[0]
    head = _heading(tm, cutter, f0)
    right = _ground_right(head)
    lateral = _sample_float(sampling, "lateral_offset_m", 3.5 * intensity)
    lateral *= _severity_scale(sampling, 1.0)
    delay = _sample_float(sampling, "reaction_delay_s", 0.0)
    n = len(frames)
    for i, f in enumerate(frames):
        t = i / max(1, n - 1)
        t = max(0.0, min(1.0, t - delay / max(0.1, len(frames) / fps)))
        s = _smoothstep(t)
        base_pose = tm.get_track_pose(cutter, f)
        if base_pose is None:
            continue
        base_c = np.array(base_pose)[:3, 3].copy()
        new_c = base_c + right * (lateral * s)
        if enable_physics and target is not None:
            tgt_pose = tm.get_track_pose(target, f)
            if tgt_pose is not None and s > 0.5:
                tgt_head = _heading(tm, target, f)
                contact = _get_dimensions(tm, cutter)[0] / 2.0 + _get_dimensions(tm, target)[0] / 2.0
                tgt_c = np.array(tgt_pose)[:3, 3]
                front_pos = tgt_c + tgt_head * contact
                blend = _smoothstep(max(0.0, (s - 0.5) / 0.5))
                new_c = new_c * (1 - blend) + front_pos * blend
        new_c[1] = base_c[1]
        tm.set_track_pose(cutter, f, _pose_with_center(base_pose, new_c))

    affected = [int(cutter)]
    if target is not None:
        affected.append(int(target))
    return affected, synth, (cutter, target)


def _gen_hard_brake(tm, roles, frames, intensity, fps=10.0, sampling=None):
    braker = roles.get("braker")
    if braker is None:
        raise ValueError("紧急刹车需要指定急刹车辆")
    f0 = frames[0]
    base_pose = tm.get_track_pose(braker, f0)
    if base_pose is None:
        raise ValueError("该帧无此车辆")
    sampling = sampling or {}
    begin_idx = min(len(frames) - 1, max(0, int(round(_sample_float(sampling, "reaction_delay_s", 0.0) * fps))))
    begin_frame = frames[begin_idx]
    decel = _sample_float(sampling, "brake_decel_mps2", 6.0) * _severity_scale(sampling, intensity)
    _apply_brake_decel(tm, braker, frames, fps, begin_frame, decel)
    return [int(braker)]


def _gen_cut_out_reveal(tm, roles, frames, fps, intensity, enable_physics, sampling=None):
    blocker = roles.get("blocker")
    obstacle = roles.get("obstacle")
    if blocker is None:
        raise ValueError("该场景需要指定前方遮挡车")
    synth = []
    if obstacle is None:
        obstacle, made = _ensure_participant(tm, None, "cut-out-reveal", "obstacle", blocker, frames, fps, intensity)
        if made:
            synth.append(int(obstacle))

    sampling = sampling or {}
    f0 = frames[0]
    head = _heading(tm, blocker, f0)
    right = _ground_right(head)
    lateral = _sample_float(sampling, "lateral_offset_m", 3.5 * intensity)
    delay = _sample_float(sampling, "reaction_delay_s", 0.4)
    n = len(frames)
    for i, f in enumerate(frames):
        t = i / max(1, n - 1)
        delay_t = delay / max(0.1, len(frames) / fps)
        s = _smoothstep(max(0.0, min(1.0, (t - delay_t) / 0.4)))
        base_pose = tm.get_track_pose(blocker, f)
        if base_pose is None:
            continue
        base_c = np.array(base_pose)[:3, 3].copy()
        new_c = base_c + right * (lateral * s)
        new_c[1] = base_c[1]
        tm.set_track_pose(blocker, f, _pose_with_center(base_pose, new_c))

    affected = [int(blocker)]
    if obstacle is not None:
        # 障碍车保持静止
        if not tm.is_synthetic(obstacle):
            obs_pose0 = tm.get_track_pose(obstacle, f0)
            if obs_pose0 is not None:
                obs_c0 = np.array(obs_pose0)[:3, 3].copy()
                for f in frames:
                    op = tm.get_track_pose(obstacle, f)
                    if op is None:
                        continue
                    tm.set_track_pose(obstacle, f, _pose_with_center(op, obs_c0))
        affected.append(int(obstacle))

    # cut-out 不一定发生碰撞，但分析遮挡车闪开后障碍与自车关系交给前端按需
    return affected, synth, (blocker, obstacle)


# ==================== 多交通要素 Corner Cases（三个及以上参与者） ====================

def _ensure_chain_roles(tm, roles, frames, fps, intensity):
    """三车链式场景角色补全：lead/middle/rear，允许用户全部不指定。"""
    lead, middle, rear = roles.get("lead"), roles.get("middle"), roles.get("rear")
    synth = []
    anchor = rear or middle or lead or _default_anchor_track(tm)
    f0 = frames[0]
    head = _heading(tm, anchor, f0)
    dims = _get_dimensions(tm, anchor)
    if rear is None:
        rear = anchor if anchor not in (lead, middle) else _make_synthetic_relative(
            tm, anchor, frames, fps, -head * (dims[0] + 10.0), head, speed_scale=1.15)
        if rear != anchor:
            synth.append(int(rear))
    if middle is None:
        middle = _make_synthetic_relative(tm, rear, frames, fps, head * (dims[0] + 9.0), head, speed_scale=0.85)
        synth.append(int(middle))
    if lead is None:
        lead = _make_synthetic_relative(tm, middle, frames, fps, head * (dims[0] + 9.0), head, speed_scale=0.45)
        synth.append(int(lead))
    return int(lead), int(middle), int(rear), synth


def _ensure_cutin_roles(tm, roles, frames, fps, intensity):
    """加塞急刹追尾：cutter/target/follower，允许全部自动生成。"""
    cutter, target, follower = roles.get("cutter"), roles.get("target"), roles.get("follower")
    synth = []
    anchor = target or follower or cutter or _default_anchor_track(tm)
    f0 = frames[0]
    head = _heading(tm, anchor, f0)
    right = _ground_right(head)
    dims = _get_dimensions(tm, anchor)
    if target is None:
        target = anchor if anchor not in (cutter, follower) else _make_synthetic_relative(
            tm, anchor, frames, fps, np.zeros(3), head, speed_scale=0.9)
        if target != anchor:
            synth.append(int(target))
    if follower is None:
        follower = _make_synthetic_relative(tm, target, frames, fps, -head * (dims[0] + 12.0), head, speed_scale=1.15)
        synth.append(int(follower))
    if cutter is None:
        cutter = _make_synthetic_relative(tm, target, frames, fps, right * 3.5 + head * 8.0, head, speed_scale=1.0)
        synth.append(int(cutter))
    return int(cutter), int(target), int(follower), synth


def _ensure_occluded_roles(tm, roles, frames, fps, intensity):
    """遮挡行人横穿追尾：vehicle/pedestrian/follower，允许全部自动生成。"""
    vehicle, pedestrian, follower = roles.get("vehicle"), roles.get("pedestrian"), roles.get("follower")
    synth = []
    anchor = vehicle or follower or pedestrian or _default_anchor_track(tm)
    f0 = frames[0]
    head = _heading(tm, anchor, f0)
    dims = _get_dimensions(tm, anchor)
    if vehicle is None:
        vehicle = anchor if anchor not in (pedestrian, follower) else _make_synthetic_relative(
            tm, anchor, frames, fps, np.zeros(3), head, speed_scale=0.9)
        if vehicle != anchor:
            synth.append(int(vehicle))
    if pedestrian is None:
        pedestrian = _make_intercepting_pedestrian(tm, vehicle, frames, fps, intensity)
        synth.append(int(pedestrian))
    else:
        _retarget_pedestrian(tm, pedestrian, vehicle, frames, fps, intensity)
    if follower is None:
        follower = _make_synthetic_relative(tm, vehicle, frames, fps, -head * (dims[0] + 12.0), head, speed_scale=1.15)
        synth.append(int(follower))
    return int(vehicle), int(pedestrian), int(follower), synth


def _brake_track_after(tm, track_id, frames, fps, begin_frame, severity=1.0):
    """在 begin_frame 后让车辆急减速并最终保持，供多车事故中的二次反应用。"""
    base_pose = tm.get_track_pose(track_id, begin_frame)
    if base_pose is None:
        return
    base_c = np.asarray(base_pose, dtype=np.float32)[:3, 3].copy()
    head = _heading(tm, track_id, begin_frame)
    v0 = max(_speed_mps(tm, track_id, begin_frame, fps), 2.0) * severity
    post_frames = [f for f in frames if f >= begin_frame]
    dist = 0.0
    n = max(1, len(post_frames) - 1)
    for i, f in enumerate(post_frames):
        t = i / n
        v = v0 * max(0.0, 1.0 - 2.2 * t)
        dist += v / fps
        c = base_c + head * dist
        c[1] = base_c[1]
        _write_pose(tm, track_id, f, _pose_with_center(base_pose, c))


def _gen_chain_reaction_rear_end(tm, roles, frames, fps, intensity, enable_physics):
    """三车连环追尾：rear -> middle -> lead。"""
    lead, middle, rear, synth = _ensure_chain_roles(tm, roles, frames, fps, intensity)
    # 前车急停，制造中车二次碰撞条件
    stop_frame = frames[len(frames) // 4]
    _brake_track_after(tm, lead, frames, fps, stop_frame, severity=1.1)
    sim1 = _simulate_pursuit_collision(tm, rear, middle, frames, fps, intensity, enable_physics=enable_physics)
    # 后车撞中车后，中车继续追上前车；二次碰撞分析使用 middle->lead
    sim2 = _simulate_pursuit_collision(tm, middle, lead, frames, fps, intensity * 0.9, enable_physics=enable_physics)
    return [lead, middle, rear], synth, (middle, lead), sim2 if sim2.get("collided") else sim1


def _gen_cutin_brake_pileup(tm, roles, frames, fps, intensity, enable_physics):
    """加塞车切入急刹，目标车急减速，后车追尾目标车。"""
    cutter, target, follower, synth = _ensure_cutin_roles(tm, roles, frames, fps, intensity)
    f0 = frames[0]
    head = _heading(tm, target, f0)
    right = _ground_right(head)
    n = len(frames)
    # 1) 加塞车从相邻车道切入目标车前方
    for i, f in enumerate(frames):
        t = i / max(1, n - 1)
        s = _smoothstep(max(0.0, min(1.0, (t - 0.10) / 0.45)))
        pose = tm.get_track_pose(cutter, f)
        tgt = tm.get_track_pose(target, f)
        if pose is None or tgt is None:
            continue
        c0 = np.asarray(pose, dtype=np.float32)[:3, 3].copy()
        tgt_c = np.asarray(tgt, dtype=np.float32)[:3, 3]
        desired = tgt_c + head * (_get_dimensions(tm, target)[0] + 3.0)
        new_c = c0 * (1.0 - s) + desired * s
        new_c[1] = c0[1]
        _write_pose(tm, cutter, f, _pose_with_center(pose, new_c))
    # 2) 加塞完成后目标车急刹，后方跟车追尾
    brake_frame = frames[int(len(frames) * 0.45)]
    _brake_track_after(tm, target, frames, fps, brake_frame, severity=1.2)
    sim = _simulate_pursuit_collision(tm, follower, target, frames, fps, intensity, enable_physics=enable_physics)
    return [cutter, target, follower], synth, (follower, target), sim


def _gen_occluded_pedestrian_pileup(tm, roles, frames, fps, intensity, enable_physics):
    """行人横穿导致前车急刹，后方车辆追尾，至少三要素：车-人-车。"""
    vehicle, pedestrian, follower, synth = _ensure_occluded_roles(tm, roles, frames, fps, intensity)
    ped_sim = _simulate_pedestrian(tm, vehicle, pedestrian, frames, fps, intensity)
    collision_frame = ped_sim.get("collision_frame") if isinstance(ped_sim, dict) else None
    if collision_frame is None:
        collision_frame = frames[int(len(frames) * 0.55)]
    _brake_track_after(tm, vehicle, frames, fps, collision_frame, severity=1.35)
    rear_sim = _simulate_pursuit_collision(tm, follower, vehicle, frames, fps, intensity, enable_physics=enable_physics)
    return [vehicle, pedestrian, follower], synth, (follower, vehicle), rear_sim if rear_sim.get("collided") else ped_sim

# NOTE: sampling schemas are attached after SCENARIOS is defined so the base
# scenario metadata stays compact near the top of the file.
SCENARIO_SAMPLING_SCHEMAS = {
    "rear-end": {
        "initial_gap_m": {"type": "range", "label": "初始车距", "min": 6.0, "max": 30.0, "step": 0.5, "default": [10.0, 18.0], "unit": "m"},
        "relative_speed_mps": {"type": "range", "label": "相对速度", "min": 3.0, "max": 15.0, "step": 0.5, "default": [5.0, 10.0], "unit": "m/s"},
        "brake_decel_mps2": {"type": "range", "label": "前车减速度", "min": 3.0, "max": 9.0, "step": 0.5, "default": [5.0, 8.0], "unit": "m/s²"},
        "reaction_delay_s": {"type": "range", "label": "后车反应延迟", "min": 0.0, "max": 1.5, "step": 0.1, "default": [0.3, 0.9], "unit": "s"},
        "lateral_offset_m": {"type": "range", "label": "横向偏置", "min": -1.2, "max": 1.2, "step": 0.1, "default": [-0.3, 0.3], "unit": "m"},
        "collision_severity": {"type": "enum", "label": "碰撞强度", "options": ["near_miss", "minor", "severe"], "default": "minor"},
        "weather_context": {"type": "enum", "label": "环境上下文", "options": ["clear", "rain", "fog", "night"], "default": "clear"},
        "road_context": {"type": "enum", "label": "道路上下文", "options": ["straight", "curve", "intersection", "ramp"], "default": "straight"},
    },
    "hard-brake": {
        "brake_decel_mps2": {"type": "range", "label": "制动减速度", "min": 3.0, "max": 10.0, "step": 0.5, "default": [5.0, 8.0], "unit": "m/s²"},
        "reaction_delay_s": {"type": "range", "label": "触发延迟", "min": 0.0, "max": 1.5, "step": 0.1, "default": [0.0, 0.5], "unit": "s"},
        "collision_severity": {"type": "enum", "label": "风险等级", "options": ["near_miss", "minor", "severe"], "default": "near_miss"},
    },
    "lane-change-cutin": {
        "lateral_offset_m": {"type": "range", "label": "切入横向距离", "min": 2.0, "max": 5.0, "step": 0.1, "default": [3.0, 4.2], "unit": "m"},
        "relative_speed_mps": {"type": "range", "label": "切入相对速度", "min": -3.0, "max": 8.0, "step": 0.5, "default": [0.0, 4.0], "unit": "m/s"},
        "reaction_delay_s": {"type": "range", "label": "切入开始延迟", "min": 0.0, "max": 1.2, "step": 0.1, "default": [0.1, 0.5], "unit": "s"},
        "collision_severity": {"type": "enum", "label": "切入强度", "options": ["near_miss", "minor", "severe"], "default": "minor"},
    },
    "intersection-tbone": {
        "relative_speed_mps": {"type": "range", "label": "冲撞相对速度", "min": 3.0, "max": 16.0, "step": 0.5, "default": [6.0, 12.0], "unit": "m/s"},
        "lateral_offset_m": {"type": "range", "label": "侧向偏置", "min": -1.0, "max": 1.0, "step": 0.1, "default": [-0.3, 0.3], "unit": "m"},
        "collision_severity": {"type": "enum", "label": "碰撞强度", "options": ["near_miss", "minor", "severe"], "default": "severe"},
        "road_context": {"type": "enum", "label": "道路上下文", "options": ["intersection", "straight"], "default": "intersection"},
    },
    "head-on": {
        "initial_gap_m": {"type": "range", "label": "初始距离", "min": 15.0, "max": 60.0, "step": 1.0, "default": [25.0, 45.0], "unit": "m"},
        "relative_speed_mps": {"type": "range", "label": "相向相对速度", "min": 6.0, "max": 25.0, "step": 0.5, "default": [10.0, 18.0], "unit": "m/s"},
        "lateral_offset_m": {"type": "range", "label": "越线偏置", "min": -1.0, "max": 1.0, "step": 0.1, "default": [-0.4, 0.4], "unit": "m"},
        "collision_severity": {"type": "enum", "label": "碰撞强度", "options": ["near_miss", "minor", "severe"], "default": "severe"},
    },
    "pedestrian-crossing": {
        "pedestrian_speed_mps": {"type": "range", "label": "横穿速度", "min": 0.8, "max": 3.5, "step": 0.1, "default": [1.2, 2.4], "unit": "m/s"},
        "reaction_delay_s": {"type": "range", "label": "车辆反应延迟", "min": 0.0, "max": 1.2, "step": 0.1, "default": [0.2, 0.8], "unit": "s"},
        "occlusion": {"type": "enum", "label": "遮挡条件", "options": ["none", "vehicle", "roadside"], "default": "vehicle"},
        "collision_severity": {"type": "enum", "label": "风险等级", "options": ["near_miss", "minor", "severe"], "default": "minor"},
    },
    "cut-out-reveal": {
        "initial_gap_m": {"type": "range", "label": "障碍距离", "min": 8.0, "max": 35.0, "step": 0.5, "default": [12.0, 24.0], "unit": "m"},
        "lateral_offset_m": {"type": "range", "label": "闪开横向距离", "min": 2.5, "max": 5.0, "step": 0.1, "default": [3.2, 4.5], "unit": "m"},
        "reaction_delay_s": {"type": "range", "label": "闪开开始延迟", "min": 0.0, "max": 1.2, "step": 0.1, "default": [0.2, 0.6], "unit": "s"},
        "collision_severity": {"type": "enum", "label": "风险等级", "options": ["near_miss", "minor", "severe"], "default": "near_miss"},
    },
}

for _scenario_key, _schema in SCENARIO_SAMPLING_SCHEMAS.items():
    if _scenario_key in SCENARIOS:
        SCENARIOS[_scenario_key]["sampling_schema"] = _schema


def _normalize_sampling_params(scenario_type, sampling_params=None, sampling_seed=None):
    schema = SCENARIOS.get(scenario_type, {}).get("sampling_schema", {})
    rng = random.Random(sampling_seed) if sampling_seed is not None else random.Random()
    provided = sampling_params or {}
    resolved = {}
    for key, spec in schema.items():
        if key in provided and provided[key] is not None:
            value = provided[key]
            if spec.get("type") == "range" and isinstance(value, (list, tuple)) and len(value) == 2:
                lo, hi = float(value[0]), float(value[1])
                if lo > hi:
                    lo, hi = hi, lo
                resolved[key] = rng.uniform(lo, hi)
            else:
                resolved[key] = value
            continue
        default = spec.get("default")
        if spec.get("type") == "range":
            if isinstance(default, (list, tuple)) and len(default) == 2:
                lo, hi = float(default[0]), float(default[1])
            else:
                lo, hi = float(spec.get("min", 0.0)), float(spec.get("max", 1.0))
            resolved[key] = rng.uniform(lo, hi)
        elif spec.get("type") == "enum":
            options = spec.get("options") or []
            resolved[key] = default if default is not None else (rng.choice(options) if options else None)
        else:
            resolved[key] = default
    for key, value in provided.items():
        if key not in resolved:
            resolved[key] = value
    return resolved


def _sample_float(params, key, default):
    try:
        return float(params.get(key, default))
    except (TypeError, ValueError):
        return float(default)


def _severity_scale(params, default=1.0):
    sev = params.get("collision_severity")
    return {"near_miss": 0.72, "minor": 1.0, "severe": 1.3}.get(sev, default)


def _apply_initial_pair_layout(tm, rear_track, front_track, frames, fps, gap_m,
                               lateral_offset_m=0.0, rear_speed_mps=None,
                               front_speed_mps=None, front_heading_sign=1.0):
    """Place two tracks as a controllable initial pair while preserving poses."""
    f0 = frames[0]
    rear_pose0 = tm.get_track_pose(rear_track, f0)
    front_pose0 = tm.get_track_pose(front_track, f0)
    if rear_pose0 is None or front_pose0 is None:
        return
    rear_pose0 = np.asarray(rear_pose0, dtype=np.float32)
    front_pose0 = np.asarray(front_pose0, dtype=np.float32)
    head = _heading(tm, rear_track, f0)
    right = _ground_right(head)
    rear_start = rear_pose0[:3, 3].copy()
    front_start = rear_start + head * float(gap_m) + right * float(lateral_offset_m)
    front_start[1] = front_pose0[1, 3]
    rear_y = rear_pose0[1, 3]
    front_y = front_pose0[1, 3]
    rear_speed = rear_speed_mps if rear_speed_mps is not None else _speed_mps(tm, rear_track, f0, fps)
    front_speed = front_speed_mps if front_speed_mps is not None else _speed_mps(tm, front_track, f0, fps)
    dt = 1.0 / fps
    for i, f in enumerate(frames):
        rp = tm.get_track_pose(rear_track, f)
        fp = tm.get_track_pose(front_track, f)
        if rp is not None:
            rc = rear_start + head * (float(rear_speed) * dt * i)
            rc[1] = rear_y
            _write_pose(tm, rear_track, f, _pose_with_center(rp, rc))
        if fp is not None:
            fc = front_start + head * float(front_heading_sign) * (float(front_speed) * dt * i)
            fc[1] = front_y
            _write_pose(tm, front_track, f, _pose_with_center(fp, fc))


def _apply_brake_decel(tm, track_id, frames, fps, begin_frame, decel_mps2):
    pose0 = tm.get_track_pose(track_id, begin_frame)
    if pose0 is None:
        return
    pose0 = np.asarray(pose0, dtype=np.float32)
    start_c = pose0[:3, 3].copy()
    head = _heading(tm, track_id, begin_frame)
    speed = max(_speed_mps(tm, track_id, begin_frame, fps), 0.0)
    dist = 0.0
    prev_f = begin_frame
    for f in frames:
        if f < begin_frame:
            continue
        dt = max(0, f - prev_f) / float(fps)
        if f == begin_frame:
            dt = 0.0
        speed = max(0.0, speed - float(decel_mps2) * dt)
        dist += speed * dt
        pose = tm.get_track_pose(track_id, f)
        if pose is None:
            prev_f = f
            continue
        c = start_c + head * dist
        c[1] = start_c[1]
        _write_pose(tm, track_id, f, _pose_with_center(pose, c))
        prev_f = f
