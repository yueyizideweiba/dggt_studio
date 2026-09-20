"""
物理碰撞检测和模拟模块

基于包围盒的碰撞检测和物理模拟，让corner case生成更真实。
支持：
1. 3D OBB（定向包围盒）碰撞检测
2. 碰撞响应（动量守恒、碰撞反弹）
3. 碰撞关键帧识别（最晚反应时刻）
4. 轨迹预测和碰撞时间估算
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass


@dataclass
class BoundingBox:
    """定向包围盒 (OBB)"""
    center: np.ndarray  # 3D中心位置 [x, y, z]
    dimensions: np.ndarray  # 尺寸 [length, width, height]
    rotation: np.ndarray  # 3x3旋转矩阵
    
    @classmethod
    def from_pose(cls, pose_world: np.ndarray, dimensions: List[float]):
        """从4x4位姿矩阵和尺寸创建包围盒"""
        pose = np.asarray(pose_world, dtype=np.float32)
        return cls(
            center=pose[:3, 3].copy(),
            dimensions=np.asarray(dimensions, dtype=np.float32),
            rotation=pose[:3, :3].copy()
        )
    
    def get_corners(self) -> np.ndarray:
        """获取8个角点的世界坐标 (8x3)"""
        # 局部坐标系下的8个角点
        half_dims = self.dimensions / 2.0
        corners_local = np.array([
            [-half_dims[0], -half_dims[1], -half_dims[2]],
            [+half_dims[0], -half_dims[1], -half_dims[2]],
            [+half_dims[0], +half_dims[1], -half_dims[2]],
            [-half_dims[0], +half_dims[1], -half_dims[2]],
            [-half_dims[0], -half_dims[1], +half_dims[2]],
            [+half_dims[0], -half_dims[1], +half_dims[2]],
            [+half_dims[0], +half_dims[1], +half_dims[2]],
            [-half_dims[0], +half_dims[1], +half_dims[2]],
        ])
        # 转换到世界坐标系
        corners_world = (self.rotation @ corners_local.T).T + self.center
        return corners_world
    
    def get_axes(self) -> np.ndarray:
        """获取OBB的三个主轴，**按列返回**（每一列是一个面法线）。

        注意：角点是按 `rotation @ corners_local.T` 生成的，所以盒子的三个主轴是
        rotation 的**列**。早期这里直接返回 rotation 并被调用方 `extend()` 展开成了
        **行**，等于在错误的三根轴上做 SAT——实测会造成约 0.5% 的**误判碰撞**
        （真实已分离却判为碰撞），是"包围盒明明没碰到却算撞了"的一个来源。
        """
        return self.rotation.T.copy()


def separating_axis_test(box1: BoundingBox, box2: BoundingBox) -> Tuple[bool, float]:
    """分离轴定理 (SAT) 检测两个OBB是否碰撞
    
    Returns:
        (is_colliding, penetration_depth): 碰撞状态和穿透深度（米）
    """
    # 15个潜在的分离轴：6个面法线 + 9个边叉积
    axes = []
    
    # Box1和Box2的三个主轴
    axes.extend(box1.get_axes())
    axes.extend(box2.get_axes())
    
    # 边的叉积（9个）
    for i in range(3):
        for j in range(3):
            axis = np.cross(box1.rotation[:, i], box2.rotation[:, j])
            norm = np.linalg.norm(axis)
            if norm > 1e-6:  # 避免平行边
                axes.append(axis / norm)
    
    min_penetration = float('inf')
    is_colliding = True

    corners1 = box1.get_corners()
    corners2 = box2.get_corners()

    for axis in axes:
        proj1 = corners1 @ axis
        min1, max1 = proj1.min(), proj1.max()
        proj2 = corners2 @ axis
        min2, max2 = proj2.min(), proj2.max()
        if max1 < min2 or max2 < min1:
            return False, 0.0  # 找到分离轴，不碰撞
        penetration = min(max1 - min2, max2 - min1)
        min_penetration = min(min_penetration, penetration)

    return is_colliding, min_penetration


def separating_axis(pose1: np.ndarray, dims1: List[float],
                    pose2: np.ndarray, dims2: List[float]):
    """SAT 且**返回分离轴**：(是否碰撞, 穿透深度, 单位轴向量)。

    轴向量取 15 条候选轴里穿透最浅的那条（最小平移向量方向），并统一定向为
    "把 box2 推离 box1"（与两心连线同向）。用于碰撞后的防穿模分离：
    沿这条轴把两车推开 penetration 米，它们就刚好接触而不重叠。
    """
    box1 = BoundingBox.from_pose(pose1, dims1)
    box2 = BoundingBox.from_pose(pose2, dims2)
    axes = []
    axes.extend(box1.get_axes())
    axes.extend(box2.get_axes())
    for i in range(3):
        for j in range(3):
            axis = np.cross(box1.rotation[:, i], box2.rotation[:, j])
            norm = np.linalg.norm(axis)
            if norm > 1e-6:
                axes.append(axis / norm)

    corners1 = box1.get_corners()
    corners2 = box2.get_corners()
    best_axis = None
    best_depth = float('inf')
    for axis in axes:
        a = np.asarray(axis, dtype=np.float64)
        n = float(np.linalg.norm(a))
        if n < 1e-9:
            continue
        a = a / n
        proj1 = corners1 @ a
        proj2 = corners2 @ a
        if proj1.max() < proj2.min() or proj2.max() < proj1.min():
            return False, 0.0, None
        depth = float(min(proj1.max() - proj2.min(), proj2.max() - proj1.min()))
        if depth < best_depth:
            best_depth = depth
            best_axis = a
    if best_axis is None:
        return False, 0.0, None
    d = np.asarray(pose2, dtype=np.float64)[:3, 3] - np.asarray(pose1, dtype=np.float64)[:3, 3]
    if float(np.dot(d, best_axis)) < 0.0:
        best_axis = -best_axis
    return True, best_depth, best_axis


def check_collision(pose1: np.ndarray, dims1: List[float],
                    pose2: np.ndarray, dims2: List[float]) -> Tuple[bool, float]:
    """检测两个物体是否碰撞
    
    Args:
        pose1, pose2: 4x4位姿矩阵
        dims1, dims2: [length, width, height]尺寸
    
    Returns:
        (is_colliding, penetration_depth)
    """
    box1 = BoundingBox.from_pose(pose1, dims1)
    box2 = BoundingBox.from_pose(pose2, dims2)
    return separating_axis_test(box1, box2)


def estimate_velocity(poses: List[np.ndarray], dt: float = 0.1) -> np.ndarray:
    """从位姿序列估计速度（米/秒）
    
    Args:
        poses: 连续帧的位姿列表
        dt: 帧间时间间隔（秒）
    
    Returns:
        3D速度向量 [vx, vy, vz]
    """
    if len(poses) < 2:
        return np.zeros(3, dtype=np.float32)
    
    # 用最近两帧估计速度
    p1 = np.asarray(poses[-2], dtype=np.float32)[:3, 3]
    p2 = np.asarray(poses[-1], dtype=np.float32)[:3, 3]
    velocity = (p2 - p1) / dt
    return velocity


def predict_collision_time(pose1: np.ndarray, vel1: np.ndarray, dims1: List[float],
                           pose2: np.ndarray, vel2: np.ndarray, dims2: List[float],
                           max_time: float = 5.0, dt: float = 0.1) -> Optional[float]:
    """预测两个运动物体的碰撞时间
    
    Args:
        pose1, pose2: 当前位姿
        vel1, vel2: 速度向量 (m/s)
        dims1, dims2: 包围盒尺寸
        max_time: 最大预测时间（秒）
        dt: 时间步长（秒）
    
    Returns:
        碰撞时间（秒），若不碰撞则返回None
    """
    p1 = np.asarray(pose1, dtype=np.float32).copy()
    p2 = np.asarray(pose2, dtype=np.float32).copy()
    v1 = np.asarray(vel1, dtype=np.float32)
    v2 = np.asarray(vel2, dtype=np.float32)
    
    num_steps = int(max_time / dt)
    
    for step in range(num_steps):
        # 更新位置
        p1[:3, 3] += v1 * dt
        p2[:3, 3] += v2 * dt
        
        # 检测碰撞
        colliding, _ = check_collision(p1, dims1, p2, dims2)
        if colliding:
            return step * dt
    
    return None


def compute_critical_frame(poses1: List[np.ndarray], dims1: List[float],
                          poses2: List[np.ndarray], dims2: List[float],
                          frame_indices: List[int],
                          safety_margin: float = 1.5,
                          fps: float = 10.0) -> Optional[Dict]:
    """计算碰撞的关键帧（最晚反应时刻）
    
    Args:
        poses1, poses2: 两个物体的位姿序列
        dims1, dims2: 包围盒尺寸
        frame_indices: 帧索引列表
        safety_margin: 安全距离（米），考虑刹车距离
        fps: 帧率
    
    Returns:
        {
            'critical_frame': int,  # 最晚反应帧
            'collision_frame': int,  # 实际碰撞帧
            'time_to_collision': float,  # 到碰撞的时间（秒）
            'distance_at_critical': float,  # 关键帧时的距离
            'collision_severity': str  # 'high'/'medium'/'low'
        }
    """
    if len(poses1) != len(poses2) or len(poses1) == 0:
        return None
    
    dt = 1.0 / fps
    collision_frame = None
    min_distance = float('inf')
    
    # 找到实际碰撞帧
    for i, (p1, p2, fi) in enumerate(zip(poses1, poses2, frame_indices)):
        colliding, penetration = check_collision(p1, dims1, p2, dims2)
        
        # 计算中心距离
        c1 = np.asarray(p1, dtype=np.float32)[:3, 3]
        c2 = np.asarray(p2, dtype=np.float32)[:3, 3]
        distance = np.linalg.norm(c1 - c2)
        min_distance = min(min_distance, distance)
        
        if colliding and collision_frame is None:
            collision_frame = fi
    
    if collision_frame is None:
        # 没有实际碰撞，返回最接近的帧
        if min_distance < safety_margin * 2:
            closest_idx = int(np.argmin([np.linalg.norm(
                np.asarray(p1, dtype=np.float32)[:3, 3] - 
                np.asarray(p2, dtype=np.float32)[:3, 3]
            ) for p1, p2 in zip(poses1, poses2)]))
            return {
                'critical_frame': int(frame_indices[closest_idx]),
                'collision_frame': None,
                'time_to_collision': None,
                'distance_at_critical': float(min_distance),
                'collision_severity': 'near_miss',
                'warning': '未发生实际碰撞，但距离很近'
            }
        return None
    
    # 从碰撞帧向前搜索关键帧
    collision_idx = frame_indices.index(collision_frame)
    critical_idx = collision_idx
    
    for i in range(collision_idx - 1, -1, -1):
        p1, p2 = poses1[i], poses2[i]
        c1 = np.asarray(p1, dtype=np.float32)[:3, 3]
        c2 = np.asarray(p2, dtype=np.float32)[:3, 3]
        distance = np.linalg.norm(c1 - c2)
        
        # 估计相对速度
        if i + 1 < len(poses1):
            v1 = estimate_velocity([poses1[i], poses1[i+1]], dt)
            v2 = estimate_velocity([poses2[i], poses2[i+1]], dt)
            rel_vel = np.linalg.norm(v1 - v2)
            
            # 刹车距离估算：v^2 / (2 * a)，假设减速度 a = 6 m/s^2
            braking_distance = (rel_vel ** 2) / (2 * 6.0) if rel_vel > 0 else 0
            required_distance = braking_distance + safety_margin
            
            if distance > required_distance:
                critical_idx = i + 1
                break
        else:
            if distance > safety_margin * 2:
                critical_idx = i + 1
                break
    
    critical_frame = frame_indices[critical_idx]
    time_to_collision = (collision_idx - critical_idx) * dt
    distance_at_critical = np.linalg.norm(
        np.asarray(poses1[critical_idx], dtype=np.float32)[:3, 3] -
        np.asarray(poses2[critical_idx], dtype=np.float32)[:3, 3]
    )
    
    # 评估碰撞严重程度
    if critical_idx + 3 >= collision_idx:
        severity = 'high'  # 反应时间<0.3秒
    elif critical_idx + 6 >= collision_idx:
        severity = 'medium'  # 0.3-0.6秒
    else:
        severity = 'low'
    
    return {
        'critical_frame': int(critical_frame),
        'collision_frame': int(collision_frame),
        'time_to_collision': float(time_to_collision),
        'distance_at_critical': float(distance_at_critical),
        'collision_severity': severity,
        'reaction_frames': int(collision_idx - critical_idx)
    }


def apply_collision_response(pose1: np.ndarray, vel1: np.ndarray, mass1: float,
                             pose2: np.ndarray, vel2: np.ndarray, mass2: float,
                             restitution: float = 0.3) -> Tuple[np.ndarray, np.ndarray]:
    """应用碰撞响应（动量守恒）
    
    Args:
        pose1, pose2: 碰撞时的位姿
        vel1, vel2: 碰撞前的速度
        mass1, mass2: 物体质量（kg）
        restitution: 恢复系数（0=完全非弹性，1=完全弹性）
    
    Returns:
        (new_vel1, new_vel2): 碰撞后的速度
    """
    # 碰撞法线：从物体1指向物体2
    c1 = np.asarray(pose1, dtype=np.float32)[:3, 3]
    c2 = np.asarray(pose2, dtype=np.float32)[:3, 3]
    normal = c2 - c1
    normal_norm = np.linalg.norm(normal)
    if normal_norm < 1e-6:
        return vel1.copy(), vel2.copy()
    normal = normal / normal_norm
    
    # 相对速度在法线方向的分量
    v1 = np.asarray(vel1, dtype=np.float32)
    v2 = np.asarray(vel2, dtype=np.float32)
    rel_vel = v1 - v2
    vel_along_normal = np.dot(rel_vel, normal)
    
    # 如果物体正在分离，不处理
    if vel_along_normal > 0:
        return v1.copy(), v2.copy()
    
    # 碰撞冲量（一维弹性碰撞公式）
    j = -(1 + restitution) * vel_along_normal / (1/mass1 + 1/mass2)
    
    # 应用冲量
    impulse = j * normal
    new_vel1 = v1 + impulse / mass1
    new_vel2 = v2 - impulse / mass2
    
    return new_vel1, new_vel2


def smooth_trajectory_adaptive(centers: List[np.ndarray], 
                               fixed_indices: List[int],
                               smoothness: float = 0.5) -> List[np.ndarray]:
    """智能轨迹平滑：拖动一个节点时，相邻节点自适应调整
    
    Args:
        centers: 轨迹点列表 (Nx3)
        fixed_indices: 固定不动的节点索引（用户拖动的点和端点）
        smoothness: 平滑系数 (0=不平滑, 1=最大平滑)
    
    Returns:
        平滑后的轨迹点列表
    """
    if len(centers) < 3:
        return centers
    
    centers = [np.asarray(c, dtype=np.float32).copy() for c in centers]
    fixed_set = set(fixed_indices)
    n = len(centers)
    
    # 多次迭代平滑
    iterations = max(3, int(10 * smoothness))
    for _ in range(iterations):
        new_centers = [c.copy() for c in centers]
        
        for i in range(1, n - 1):
            if i in fixed_set:
                continue
            
            # 相邻点的平均位置
            prev_c = centers[i - 1]
            next_c = centers[i + 1]
            avg = (prev_c + next_c) / 2.0
            
            # 混合原位置和平均位置
            alpha = smoothness * 0.5
            new_centers[i] = (1 - alpha) * centers[i] + alpha * avg
        
        centers = new_centers
    
    return centers


def catmull_rom_spline(points: List[np.ndarray], num_samples: int = 50) -> np.ndarray:
    """Catmull-Rom样条插值，生成平滑轨迹
    
    Args:
        points: 控制点 (Nx3)
        num_samples: 每段的采样数
    
    Returns:
        插值后的轨迹 (Mx3)
    """
    if len(points) < 2:
        return np.array(points)
    
    points = [np.asarray(p, dtype=np.float32) for p in points]
    
    # 扩展端点以保持切线
    extended = [points[0] - (points[1] - points[0])] + points + \
               [points[-1] + (points[-1] - points[-2])]
    
    result = []
    for i in range(1, len(extended) - 2):
        p0, p1, p2, p3 = extended[i-1:i+3]
        
        for t in np.linspace(0, 1, num_samples, endpoint=(i == len(extended) - 3)):
            t2, t3 = t * t, t * t * t
            # Catmull-Rom公式
            point = 0.5 * (
                (2 * p1) +
                (-p0 + p2) * t +
                (2*p0 - 5*p1 + 4*p2 - p3) * t2 +
                (-p0 + 3*p1 - 3*p2 + p3) * t3
            )
            result.append(point)
    
    return np.array(result)
