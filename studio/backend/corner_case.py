"""
Corner Case（交通事故场景）生成模块

底层逻辑：通过 **修改动态物体的轨迹**（写入 TrackManager 的 track 编辑）来制造
事故效果，与轨迹编辑/渲染管线完全统一。

坐标约定（与数据集一致）：
- 世界 Y 轴为竖直方向（向下为正），车辆在 XZ 地平面运动。
- 物体 pose_world 为 4x4，前向由其旋转列决定；我们用相邻帧位移估计行进方向。

每种场景定义若干"角色"（role），由前端分别指定 track_id：
- rear-end（追尾）：attacker（肇事/后车）、victim（受害/前车）
- hard-brake（紧急刹车）：braker（急刹车辆）
- lane-change-cutin（变道加塞）：cutter（加塞车）、target（被加塞车，可选）
- intersection-tbone（路口侧碰）：attacker（直行冲撞车）、victim（被撞车）
- head-on（对向碰撞）：attacker、victim
"""

import numpy as np


SCENARIOS = {
    "rear-end": {
        "name": "追尾事故",
        "desc": "后车加速撞向前车",
        "roles": [
            {"key": "attacker", "label": "肇事车（后车）"},
            {"key": "victim", "label": "受害车（前车）"},
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
            {"key": "target", "label": "被加塞车（可选）"},
        ],
    },
    "intersection-tbone": {
        "name": "路口侧碰 (T-Bone)",
        "desc": "冲撞车横向撞向被撞车侧面",
        "roles": [
            {"key": "attacker", "label": "冲撞车"},
            {"key": "victim", "label": "被撞车"},
        ],
    },
    "head-on": {
        "name": "对向碰撞",
        "desc": "两车相向行驶直至碰撞",
        "roles": [
            {"key": "attacker", "label": "肇事车"},
            {"key": "victim", "label": "对向车"},
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
        })
    return out


def _heading(tm, track_id, frame_idx):
    """估计某 track 在该帧的行进方向（地面 XZ 平面，单位向量）。"""
    frames = tm.get_track_frames(track_id)
    if not frames:
        return np.array([1.0, 0.0, 0.0])
    # 用前后帧位移估计
    f0 = frame_idx
    f1 = frame_idx + 1 if (frame_idx + 1) in frames else frame_idx
    fm = frame_idx - 1 if (frame_idx - 1) in frames else frame_idx
    p1 = tm.get_track_pose(track_id, f1)
    p0 = tm.get_track_pose(track_id, fm)
    if p1 is None or p0 is None:
        return np.array([1.0, 0.0, 0.0])
    d = np.array(p1)[:3, 3] - np.array(p0)[:3, 3]
    d[1] = 0.0  # 投影到地面
    n = np.linalg.norm(d)
    if n < 1e-4:
        # 退化：用 pose 的某一列作为前向
        pose = np.array(tm.get_track_pose(track_id, frame_idx))
        d = pose[:3, 2].copy()
        d[1] = 0.0
        n = np.linalg.norm(d)
        if n < 1e-4:
            return np.array([1.0, 0.0, 0.0])
    return d / n


def _ground_right(heading):
    """地面上垂直于 heading 的右向量（绕竖直 Y 轴）。"""
    # world up（竖直）取 -Y（数据集 +Y 向下）。right = up × heading
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


def _smoothstep(t):
    return t * t * (3 - 2 * t)


def generate(tm, scenario_type, roles, start_frame, num_frames, intensity=1.0):
    """生成 corner case，写入 TrackManager 的 track 编辑。

    Args:
        tm: TrackManager
        scenario_type: 场景类型 key
        roles: {role_key: track_id}
        start_frame, num_frames: 事故发生的帧区间
        intensity: 强度系数（影响位移/速度幅度）

    Returns:
        dict 结果摘要
    """
    if scenario_type not in SCENARIOS:
        raise ValueError(f"未知场景类型: {scenario_type}")

    end_frame = start_frame + num_frames
    frames = list(range(start_frame, end_frame + 1))

    affected = []

    if scenario_type == "rear-end":
        affected = _gen_rear_end(tm, roles, frames, intensity)
    elif scenario_type == "hard-brake":
        affected = _gen_hard_brake(tm, roles, frames, intensity)
    elif scenario_type == "lane-change-cutin":
        affected = _gen_lane_change(tm, roles, frames, intensity)
    elif scenario_type == "intersection-tbone":
        affected = _gen_tbone(tm, roles, frames, intensity)
    elif scenario_type == "head-on":
        affected = _gen_head_on(tm, roles, frames, intensity)

    return {
        "scenario_type": scenario_type,
        "affected_tracks": affected,
        "start_frame": start_frame,
        "num_frames": num_frames,
    }


def _gen_rear_end(tm, roles, frames, intensity):
    """追尾：肇事车（后车）逐渐加速贴近受害车（前车），直至重合。"""
    attacker = roles.get("attacker")
    victim = roles.get("victim")
    if attacker is None or victim is None:
        raise ValueError("追尾事故需要指定肇事车和受害车")

    f0 = frames[0]
    affected = [attacker]

    # 受害车保持原轨迹（不强行修改），用其每帧位置作为追尾目标
    for i, f in enumerate(frames):
        t = i / max(1, len(frames) - 1)
        s = _smoothstep(t)

        atk_pose = tm.get_track_pose(attacker, f)
        vic_pose = tm.get_track_pose(victim, f)
        if atk_pose is None or vic_pose is None:
            continue
        atk_c = np.array(atk_pose)[:3, 3]
        vic_c = np.array(vic_pose)[:3, 3]

        # 受害车尾部稍前一点（沿受害车前向回退一个车身）
        head = _heading(tm, victim, f)
        target = vic_c - head * (2.0)  # 贴在前车尾后约 2m
        # 从原始位置插值到目标位置（越接近末尾越贴近）
        new_c = atk_c * (1 - s) + target * s
        new_c[1] = atk_c[1]  # 锁定高度
        tm.set_track_pose(attacker, f, _pose_with_center(atk_pose, new_c))

    return affected


def _gen_hard_brake(tm, roles, frames, intensity):
    """紧急刹车：车辆沿自身前向逐渐减速，位移按减速曲线压缩直至停止。"""
    braker = roles.get("braker")
    if braker is None:
        raise ValueError("紧急刹车需要指定急刹车辆")

    f0 = frames[0]
    base_pose = tm.get_track_pose(braker, f0)
    if base_pose is None:
        raise ValueError("该帧无此车辆")
    base_c = np.array(base_pose)[:3, 3].copy()
    head = _heading(tm, braker, f0)

    # 估计初速度（每帧位移）
    p1 = tm.get_track_pose(braker, min(f0 + 1, frames[-1]))
    v0 = np.linalg.norm((np.array(p1)[:3, 3] - base_c)) if p1 is not None else 1.0
    v0 = max(v0, 0.5) * intensity

    n = len(frames)
    # 减速：速度线性降到 0，位移为累积
    dist = 0.0
    for i, f in enumerate(frames):
        t = i / max(1, n - 1)
        v = v0 * (1 - t)  # 线性减速
        dist += max(0.0, v)
        new_c = base_c + head * dist
        new_c[1] = base_c[1]
        tm.set_track_pose(braker, f, _pose_with_center(base_pose, new_c))

    return [braker]


def _gen_lane_change(tm, roles, frames, intensity):
    """变道加塞：加塞车沿当前前向继续行驶，同时横向切入相邻车道。"""
    cutter = roles.get("cutter")
    if cutter is None:
        raise ValueError("变道加塞需要指定加塞车")

    f0 = frames[0]
    head = _heading(tm, cutter, f0)
    right = _ground_right(head)
    lateral = 3.5 * intensity  # 一个车道宽约 3.5m

    n = len(frames)
    for i, f in enumerate(frames):
        t = i / max(1, n - 1)
        s = _smoothstep(t)
        base_pose = tm.get_track_pose(cutter, f)
        if base_pose is None:
            continue
        base_c = np.array(base_pose)[:3, 3].copy()
        new_c = base_c + right * (lateral * s)
        new_c[1] = base_c[1]
        tm.set_track_pose(cutter, f, _pose_with_center(base_pose, new_c))

    return [cutter]


def _gen_tbone(tm, roles, frames, intensity):
    """路口侧碰：冲撞车横向（垂直于受害车前向）移动撞向受害车侧面。"""
    attacker = roles.get("attacker")
    victim = roles.get("victim")
    if attacker is None or victim is None:
        raise ValueError("路口侧碰需要指定冲撞车和被撞车")

    f0 = frames[0]
    n = len(frames)
    for i, f in enumerate(frames):
        t = i / max(1, n - 1)
        s = _smoothstep(t)
        atk_pose = tm.get_track_pose(attacker, f)
        vic_pose = tm.get_track_pose(victim, f)
        if atk_pose is None or vic_pose is None:
            continue
        atk_c = np.array(atk_pose)[:3, 3]
        vic_c = np.array(vic_pose)[:3, 3]
        # 冲撞车直接冲向受害车当前位置
        new_c = atk_c * (1 - s) + vic_c * s
        new_c[1] = atk_c[1]
        tm.set_track_pose(attacker, f, _pose_with_center(atk_pose, new_c))

    return [attacker]


def _gen_head_on(tm, roles, frames, intensity):
    """对向碰撞：肇事车跨越到对向，与对向车相向接近直至重合。"""
    attacker = roles.get("attacker")
    victim = roles.get("victim")
    if attacker is None or victim is None:
        raise ValueError("对向碰撞需要指定肇事车和对向车")

    n = len(frames)
    for i, f in enumerate(frames):
        t = i / max(1, n - 1)
        s = _smoothstep(t)
        atk_pose = tm.get_track_pose(attacker, f)
        vic_pose = tm.get_track_pose(victim, f)
        if atk_pose is None or vic_pose is None:
            continue
        atk_c = np.array(atk_pose)[:3, 3]
        vic_c = np.array(vic_pose)[:3, 3]
        # 肇事车逐渐移动到对向车前方一点（相向接近）
        head_v = _heading(tm, victim, f)
        target = vic_c + head_v * 2.0
        new_c = atk_c * (1 - s) + target * s
        new_c[1] = atk_c[1]
        tm.set_track_pose(attacker, f, _pose_with_center(atk_pose, new_c))

    return [attacker]
