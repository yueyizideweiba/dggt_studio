import os
import json
import math
import os
import struct
import random
import numpy as np

# ==================== 主车（EGO）实体化 ====================
# 把"主车"也当成一个可渲染/可编辑的物体：3D 视图与 BEV 里能看到这辆车本身，
# 编辑它的轨迹后可以沿新轨迹录制"主车视角"视频。
EGO_TRACK_ID = 900000
EGO_PLY_DEFAULT = "/root/autodl-fs/dggt-main/sam-3d-objects/ego.ply"
# SAM3D canonical(up=-Z, front=-Y, right=+X) → DGGT 物体局部系(X=宽, Y=上, Z=前)
SAM3D_MODEL_CORR = np.array([[-1.0, 0.0, 0.0],
                             [0.0, 0.0, -1.0],
                             [0.0, -1.0, 0.0]], dtype=np.float32)
# 相机 → 车体中心（相机坐标系：x右, y下, z前）：约在相机下方 1.15m、后方 0.55m
# 多视角数据把 object_id 编码成 view * VIEW_ID_STRIDE + raw_object_id（与 dggt_engine 一致）
VIEW_ID_STRIDE = 1000000
EGO_CAM_OFFSET = np.array([0.0, 1.15, -0.55], dtype=np.float32)
EGO_LENGTH_M = 4.9
# 行人 3D 模型（SAM3D 导出）：需要行人时随机取其中一个
PED_PLY_DEFAULT = ("/root/autodl-fs/dggt-main/sam-3d-objects/person_0.ply:"
                   "/root/autodl-fs/dggt-main/sam-3d-objects/person_1.ply")
PED_TARGET_HEIGHT_M = 1.75


def _ply_axis_extents(ply_path):
    """只读 PLY 头 + 顶点位置，返回 (x, y, z) 三轴范围（用于把模型缩放到真实尺寸）。"""
    try:
        with open(ply_path, "rb") as f:
            head = b""
            while b"end_header" not in head:
                chunk = f.read(4096)
                if not chunk:
                    break
                head += chunk
            end = head.index(b"end_header") + len(b"end_header") + 1
            text = head[:end].decode("ascii", "replace")
            types = {"float": "f4", "double": "f8", "uchar": "u1", "int": "i4",
                     "uint": "u4", "short": "i2", "ushort": "u2"}
            props = []
            n = 0
            for line in text.splitlines():
                parts = line.split()
                if len(parts) >= 3 and parts[0] == "element" and parts[1] == "vertex":
                    n = int(parts[2])
                elif len(parts) >= 3 and parts[0] == "property":
                    props.append((parts[2], types.get(parts[1], "f4")))
            dt = np.dtype(props)
            f.seek(end)
            need = ["x", "y", "z"]
            if not all(k in dt.names for k in need):
                return None
            arr = np.frombuffer(f.read(dt.itemsize * n), dtype=dt)
            cs = [arr[k].astype(np.float64) for k in need]
            return (float(cs[0].max() - cs[0].min()),
                    float(cs[1].max() - cs[1].min()),
                    float(cs[2].max() - cs[2].min()))
    except Exception:  # noqa: BLE001
        return None


def _pose_from_camera(c2w, offset=None):
    """由自车相机位姿推导"车体中心位姿"：朝向取相机前向在地面的投影。"""
    c2w = np.asarray(c2w, dtype=np.float32)
    if c2w.shape == (3, 4):
        c2w = np.vstack([c2w, np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32)])
    off = EGO_CAM_OFFSET if offset is None else np.asarray(offset, dtype=np.float32)
    Rc, tc = c2w[:3, :3], c2w[:3, 3]
    fwd = Rc[:, 2].copy()
    fwd[1] = 0.0
    n = float(np.linalg.norm(fwd))
    fwd = np.array([0.0, 0.0, 1.0], dtype=np.float32) if n < 1e-6 else (fwd / n).astype(np.float32)
    up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    x = np.cross(up, fwd).astype(np.float32)
    pose = np.eye(4, dtype=np.float32)
    pose[:3, 0], pose[:3, 1], pose[:3, 2] = x, up, fwd
    pose[:3, 3] = tc + Rc @ off
    return pose


# ==================== 主车自动贴地（逐场景自适应） ====================
# 不同 4DGS 场景里自车相机的安装高度差别很大（实测 1.1m ~ 1.9m），而车体相对相机的
# 安装关系 EGO_CAM_OFFSET 是固定值，于是同一套偏移会让主车在部分场景里"沉进路面"。
# 这里直接从静态场景高斯估计自车路径下方的路面高度，把车体放到地面上。
EGO_AUTO_GROUND = os.environ.get("DGGT_EGO_AUTO_GROUND", "1").strip().lower() not in (
    "0", "false", "no", "off", "")
EGO_GROUND_CLEARANCE = float(os.environ.get("DGGT_EGO_GROUND_CLEARANCE", "0.0"))
EGO_GROUND_RADIUS = float(os.environ.get("DGGT_EGO_GROUND_RADIUS", "6.0"))
EGO_GROUND_MAX_POINTS = int(os.environ.get("DGGT_EGO_GROUND_MAX_POINTS", "600000"))


def _scene_vehicle_length(track_meta):
    """场景里已有"车辆类"物体的中位车长（米）；不足 3 辆返回 None。

    自车模型默认按固定 4.9m 缩放，比场景里的车明显大；这里改成参照场景车辆。
    """
    lens = []
    for _tid, meta in (track_meta or {}).items():
        d = (meta or {}).get("dimensions") or []
        if len(d) != 3:
            continue
        try:
            w, h, l = float(d[0]), float(d[1]), float(d[2])
        except (TypeError, ValueError):
            continue
        if 2.5 <= l <= 6.5 and 1.0 <= w <= 2.8 and 0.7 <= h <= 2.5 and l > w:
            lens.append(l)
    if len(lens) < 3:
        return None
    return float(np.median(np.array(lens)))


def _scene_vehicle_dims(track_meta):
    """场景里已有"车辆类"物体的中位尺寸 [宽,高,长]（米）；不足 3 辆返回 None。"""
    rows = []
    for _tid, meta in (track_meta or {}).items():
        d = (meta or {}).get("dimensions") or []
        if len(d) != 3:
            continue
        try:
            w, h, l = float(d[0]), float(d[1]), float(d[2])
        except (TypeError, ValueError):
            continue
        if 2.5 <= l <= 6.5 and 1.0 <= w <= 2.8 and 0.7 <= h <= 2.5 and l > w:
            rows.append((w, h, l))
    if len(rows) < 3:
        return None
    return [float(x) for x in np.median(np.array(rows), axis=0)]


def _scene_points_cpu(renderer, max_points=EGO_GROUND_MAX_POINTS):
    """把静态场景高斯的中心点下采样到 CPU，供估计路面高度（避免每帧拷显存）。"""
    gs = getattr(renderer, "static_gs", None)
    if not gs or "means" not in gs:
        return None
    means = gs["means"]
    try:
        n = int(means.shape[0])
    except Exception:  # noqa: BLE001
        return None
    if n <= 0:
        return None
    step = max(1, n // max(1, int(max_points)))
    try:
        arr = means[::step].detach().float().cpu().numpy()
    except Exception:  # noqa: BLE001
        return None
    return arr if arr.size else None


def _ground_height_near(points, eye, radius, up_sign, min_pts=30, max_tries=5):
    """在相机水平邻域内估计路面高度（世界 y）。

    `up_sign`：世界 +y 是"上"(+1) 还是"下"(-1)。地面在相机的"下"方向，
    所以 +y 向下时取偏大的分位、+y 向上时取偏小的分位（路面高斯最密）。

    **邻域逐级放大**：4DGS 静态点只覆盖"相机走过/看过"的区域，物体轨迹常常会走到
    覆盖边缘（例如自车起点后方、对向车道远端），此时固定 2 倍半径会拿到 <10 个点而
    返回 None。旧实现一旦返回 None，`plan_trajectory` 会退化成"用自车**车体中心**高度
    当路面"（自车的 y 是车身中心，比路面高半个车高），于是生成的车肉眼可见地"浮在
    空中"半个车高。这里改成 6→12→24→48→96m 逐级放大：只要视野里还有路面点就给出
    一个估计；只有连最远一级都几乎没有点时才返回 None（真·场景外）。
    """
    q = 55.0 if up_sign < 0 else 45.0
    dx = points[:, 0] - float(eye[0])
    dz = points[:, 2] - float(eye[2])
    d2 = dx * dx + dz * dz
    r = max(1.0, float(radius))
    last = None
    for _ in range(max(1, int(max_tries))):
        m = d2 <= r * r
        n = int(m.sum())
        if n >= int(min_pts):
            return float(np.percentile(points[m, 1], q))
        if n >= 8:
            last = float(np.percentile(points[m, 1], q))
        r *= 2.0
    return last


def _smooth_series(values, window=5):
    """滑动均值（端点收窄），抑制逐帧地面估计的抖动。"""
    n = len(values)
    if n <= 2 or window <= 1:
        return list(values)
    half = int(window) // 2
    out = []
    for i in range(n):
        lo, hi = max(0, i - half), min(n, i + half + 1)
        out.append(float(np.mean(values[lo:hi])))
    return out


import torch

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class TrackManager:
    def __init__(self, renderer):
        self.renderer = renderer
        self.num_frames = self._count_frames()

        # track_id -> {frame_idx: raw_object_id}
        self.track_to_frames = {}
        # (frame_idx, raw_object_id) -> track_id
        self.frame_obj_to_track = {}
        # track_id -> 元信息（type, dimensions, 首次出现帧）
        self.track_meta = {}
        # track_id -> {frame_idx: 原始 pose_world(4x4 np)}
        self.track_raw_poses = {}

        # 编辑存储
        # track_id -> {frame_idx: pose 4x4 np}  位姿关键帧
        self.track_edits = {}
        # 被删除的 track_id 集合（整段删除）
        self.track_deleted = set()
        # track_id -> 3x3 np 局部旋转偏移（全局作用于该物体所有帧，用于 360° 旋转）
        self.track_rotations = {}

        # 合成参与者（自动生成的事故参与者，原始数据中不存在）
        # synth_track_id -> {
        #   "donor_object_id": int(首帧 fallback raw object_id),
        #   "donor_track_id": int(克隆来源 track),
        #   "donor_pose0": 4x4 np(首帧 fallback pose),
        #   "dimensions": [l,w,h], "type": str,
        #   "poses": {frame_idx: 4x4 np}  # 逐帧世界位姿
        # }
        self.synthetic_tracks = {}
        self._next_synth_id = 100000  # 合成 track_id 从 10 万起，避免与真实 track 冲突

        # 撤销/重做快照栈（保存 (track_edits, track_deleted) 的深拷贝）
        self._undo_stack = []
        self._redo_stack = []
        self._max_history = 50

        # 车头自动朝向运动方向（平滑）。auto_heading_tracks=None 表示对所有动态物体生效。
        # 这是**默认开启**的功能（用户无需手动应用）。
        self.auto_heading = True
        self.auto_heading_smoothing = 0.65   # 0~1，越大越平滑
        self.auto_heading_tracks = None
        self.auto_heading_window = 6         # 方向估计的 ±帧窗口（稳健 PCA 拟合）
        self.auto_heading_sustain = 4        # 目标方向反转需连续确认的帧数（抗噪声来回掉头）
        # 朝向限速（rad/s）：质量门要求 max_yaw_rate ≤ 1.2 rad/s，这里留一点余量。
        # 逐帧最大转角 = max_yaw_rate / fps（帧率由 fps 决定，corner case 生成时会同步）。
        self.auto_heading_max_yaw_rate = 1.0
        self.fps = 10.0
        self._heading_cache = {}             # track_id -> {frame_idx: yaw(rad)}
        self._forward_sign_cache = {}        # track_id -> ±1（车头是 +Z / -Z）
        # 已被 SAM3D 替换掉的原始 track：即使被编辑也不允许复活（避免"替换前后两个模型"）
        self.track_replaced = set()
        # corner case 生成前的持久合成物体轨迹备份（清除时用它还原，保留手动编辑）
        self._corner_poses_backup = {}

        # ---- 可扩展时间轴 ----
        # track_id -> {frame_idx: pose}：**超出磁盘数据帧**的外推位姿。与 track_edits 分开存，
        # 这样"延长出来的轨迹"不会被当成用户手动编辑（但仍可被编辑覆盖）。
        self.track_extended = {}
        self._road_model_cache = None    # 场景高精地图模型（懒加载、缓存；没有地图时为 None）
        # track_id -> 该 track 用于渲染的稳定 raw object_id（超范围帧没有逐帧数据，
        # 外观沿用最后一次出现的那一份）
        self._track_render_raw = {}
        # (view, raw_object_id) -> track_id（供超范围帧把复合 id 反查回 track）
        self._view_raw_to_track = {}
        self.extended_from_frame = None      # 从第几帧开始是"外推出来的"

        # 主车（EGO）实体化的物体 track（900000）：ply + 逐帧车体位姿 + 相机安装关系
        self.ego_track_id = None
        self.ego = {}
        # 以谁的视角渲染（None = 真实主车/自车；也可以指定任意动态物体 track）
        self.ego_source_track = None
        # 自动贴地用：逐帧自车相机外参 + 自动贴地前的车体位姿（供 refit 重算）
        self._ego_cams = {}
        self._ego_pose_no_ground = {}

        self._build()
        self._init_ego_track()
        self._register_render_providers()

    # ==================== 可扩展时间轴（超出原场景帧数） ====================

    def _register_render_providers(self):
        """把"超出数据帧时怎么渲染"的两个回调挂到渲染器上。

        - `camera_provider(f)`：按自车轨迹给出相机位姿（超范围帧本来是读不到 ego json 的）
        - `object_pose_provider(composite_id, f)`：按轨迹给出物体位姿
        这样"延长帧"不需要任何磁盘数据也能出图。
        """
        try:
            self.renderer.camera_provider = self.get_ego_camera
            self.renderer.object_pose_provider = self._pose_for_composite
        except Exception:  # noqa: BLE001
            pass

    def data_frame_count(self):
        """磁盘上真实存在的帧数。"""
        try:
            return int(self.renderer.data_frame_count())
        except Exception:  # noqa: BLE001
            return int(self.num_frames)

    def _pose_for_composite(self, composite_id, frame_idx):
        """超范围帧：把渲染器的复合 object id 反查回 track，再取该帧位姿。"""
        try:
            cid = int(composite_id)
        except (TypeError, ValueError):
            return None
        nv = int(getattr(self.renderer, "num_views", 1) or 1)
        view, raw = (cid // VIEW_ID_STRIDE, cid % VIEW_ID_STRIDE) if nv > 1 else (0, cid)
        tid = self._view_raw_to_track.get((view, raw))
        if tid is None:
            return None
        if tid in self.track_deleted or tid in self.track_replaced:
            return None
        pose = self.get_track_pose(tid, int(frame_idx))
        if pose is None:
            return None
        return np.asarray(pose, dtype=np.float32)

    @staticmethod
    def _yaw_of(pose):
        f = np.asarray(pose, dtype=np.float32)[:3, 2]
        return float(np.arctan2(float(f[0]), float(f[2])))

    @staticmethod
    def _wrap_pi(a):
        while a > np.pi:
            a -= 2.0 * np.pi
        while a < -np.pi:
            a += 2.0 * np.pi
        return float(a)

    def _extrapolate_poses(self, poses, targets, mode="extrapolate", window=8,
                           max_speed=35.0, ground_snap=True):
        """由已有逐帧位姿外推 `targets`（都大于已有最后一帧）的位姿。

        默认 `extrapolate`：用末尾 `window` 帧估计"常速度 + 常航向角速度"，沿地面继续前进
        （高度保持不变，可选按静态地面重新贴合）。速度/角速度都取中位数，抗噪声。
        `hold` 则冻结在最后一帧不动。
        """
        keys = sorted(poses.keys())
        tgt = [int(f) for f in targets if int(f) > keys[-1]]
        if not tgt:
            return {}
        last_f = keys[-1]
        last = np.asarray(poses[last_f], dtype=np.float32)
        if mode == "hold":
            return {f: last.copy() for f in tgt}

        win = keys[-max(2, min(int(window), len(keys))):]
        vs, dys = [], []
        for f0, f1 in zip(win[:-1], win[1:]):
            p0 = np.asarray(poses[f0], dtype=np.float32)[:3, 3]
            p1 = np.asarray(poses[f1], dtype=np.float32)[:3, 3]
            d = p1 - p0
            d[1] = 0.0                       # 只在地面平面内延续
            vs.append(d / float(max(1, f1 - f0)))
            dys.append(self._wrap_pi(self._yaw_of(poses[f1]) - self._yaw_of(poses[f0]))
                       / float(max(1, f1 - f0)))
        v = np.median(np.asarray(vs), axis=0) if vs else np.zeros(3, dtype=np.float32)
        spd = float(np.linalg.norm(v))
        if spd > max_speed:                  # 估计出来的速度离谱就限幅（数据噪声）
            v = v / spd * max_speed
        dyaw = float(np.median(np.asarray(dys))) if dys else 0.0

        out = {}
        last_f = int(last_f)
        base_pos = last[:3, 3].astype(np.float64).copy()
        base_yaw = self._yaw_of(last)
        for f in tgt:
            # **按帧号步进**（而不是"每个 target 走一步"）：`extend_timeline` 会被调用多次
            # （例如先按 num_frames=30 延到 29 帧，再按生成物体的 40 帧延到 40），第二次的
            # targets 从 29 开始、锚点却是最后一个**数据**帧 24 —— 一次一步会少走 5 帧，
            # 于是外推出来的轨迹在拼接处**重复前 5 帧**（用户看到"轨迹莫名来回跳"）。
            n = int(f) - last_f
            pos = base_pos + v.astype(np.float64) * n
            yaw = base_yaw + dyaw * n
            c, sn = math.cos(yaw), math.sin(yaw)
            P = last.copy()
            P[:3, 0] = (c, 0.0, -sn)
            P[:3, 1] = (0.0, 1.0, 0.0)
            P[:3, 2] = (sn, 0.0, c)
            P[:3, 3] = pos.astype(np.float32)
            if ground_snap:
                gy = self.ground_y_at(float(pos[0]), float(pos[2]))
                if gy is not None:
                    dims = self.get_track_dimensions_local(pose=P)
                    P[1, 3] = float(gy) + self.world_up_sign() * (0.5 * float(dims[1]))
            out[int(f)] = P.astype(np.float32)
        return out

    def _road_model(self):
        """场景的高精地图道路模型（`road_rules.RoadModel`）；没有地图时返回 None（带缓存）。"""
        rm = getattr(self, "_road_model_cache", None)
        if rm is None:
            try:
                import road_rules
                sd = getattr(self.renderer, "scene_path", None)
                self._road_model_cache = road_rules.get_road_model(str(sd)) if sd else None
            except Exception:  # noqa: BLE001
                self._road_model_cache = None
            rm = self._road_model_cache
        return rm

    def _extrapolate_along_lane(self, poses, targets, max_speed=35.0, dims=None):
        """沿**车道中心线**外推位姿（替代常速度直线外推）。

        解决"延长一辆正在转弯的车，结果它直着开出去"的问题：找该车最后所在的车道，
        沿车道几何继续往前走，位置取车道中心线、航向取车道方向 —— 转弯的车就延成转弯。

        返回 `None` 表示"这辆车不在车道上/没有地图"，调用方应退回直线外推。
        """
        road = self._road_model()
        if road is None:
            return None
        import road_rules as rr
        keys = sorted(poses)
        tgt = [int(f) for f in targets if int(f) > keys[-1]]
        if not tgt:
            return {}
        last_f = int(keys[-1])
        last = np.asarray(poses[last_f], dtype=np.float32)
        c0 = last[:3, 3]
        fwd = last[:3, 2]
        lane, d = road.lane_at(c0[[0, 2]], heading=fwd, max_d=6.0)
        if lane is None:
            return None
        route = road.route_forward(c0[[0, 2]], lane=lane, heading=fwd, distance=400.0)
        if not route or len(route['pts']) < 2:
            return None
        P = np.asarray(route['pts'], dtype=np.float64)
        s0 = rr.project_arc(P, c0[[0, 2]])
        # 速度：末尾若干帧的中位步长；夹在 [0.5, min(max_speed, 车道限速)]
        dt = 1.0 / max(1e-6, float(getattr(self, "fps", 10.0)))
        win = keys[-max(2, min(8, len(keys))):]
        steps = []
        for fa, fb in zip(win[:-1], win[1:]):
            a = np.asarray(poses[fa], np.float32)[:3, 3]
            b = np.asarray(poses[fb], np.float32)[:3, 3]
            steps.append(float(np.linalg.norm((b - a)[[0, 2]]) / max(1, fb - fa)))
        spd = float(np.median(steps)) / dt if steps else 0.0
        lim = float(road.speed_limit_mps(lane) or max_speed)
        spd = float(np.clip(spd, 0.5, min(max_speed, max(lim, 3.0))))
        dd = list(dims) if dims and len(dims) >= 3 else [1.86, 1.45, 4.9]
        out = {}
        for f in tgt:
            s = s0 + spd * dt * (int(f) - last_f)
            c, h = rr.sample_polyline(P, s)
            if c is None:
                continue
            yaw = math.atan2(float(h[0]), float(h[2]))
            cy, sy = math.cos(yaw), math.sin(yaw)
            Pm = last.copy()
            Pm[:3, 0] = (cy, 0.0, -sy)
            Pm[:3, 1] = (0.0, 1.0, 0.0)
            Pm[:3, 2] = (sy, 0.0, cy)
            gy = self.ground_y_at(float(c[0]), float(c[2]))
            Pm[1, 3] = float(gy) + self.world_up_sign() * (0.5 * float(dd[1])) if gy is not None \
                else float(last[1, 3])
            Pm[:3, 3] = np.array([float(c[0]), float(Pm[1, 3]), float(c[2])], dtype=np.float32)
            out[int(f)] = Pm.astype(np.float32)
        return out

    def _extrapolate_poses_smart(self, poses, targets, mode="extrapolate", window=8,
                                 max_speed=35.0, dims=None):
        """先沿车道外推，不行再退回直线外推。`hold` 模式不沿车道。"""
        if mode == "hold":
            return self._extrapolate_poses(poses, targets, mode="hold",
                                           window=window, max_speed=max_speed)
        lane = self._extrapolate_along_lane(poses, targets, max_speed=max_speed, dims=dims)
        if lane:
            return lane
        return self._extrapolate_poses(poses, targets, mode=mode, window=window, max_speed=max_speed)

    def get_track_dimensions_local(self, track_id=None, pose=None):
        """外推时用：拿到尺寸（拿不到就给一个轿车默认值）。"""
        if track_id is not None:
            try:
                d = self.get_track_dimensions(int(track_id))
                if d and len(d) >= 3:
                    return d
            except Exception:  # noqa: BLE001
                pass
        return [1.86, 1.45, 4.9]

    def extend_timeline(self, extra_frames=None, total_frames=None, mode="extrapolate",
                        ego_mode=None, extend_tracks=True):
        """把时间轴延长到目标帧数：相机按自车轨迹外推，物体按各自轨迹外推。

        这是"原场景多少帧不再重要"的核心：静态场景是单份高斯（与帧无关），
        动态物体复用最后出现的外观，位姿由轨迹给出，相机由自车轨迹给出。

        Returns: {"data_frames", "total_frames", "old_total_frames", "extended_tracks", "mode"}
        """
        data_frames = self.data_frame_count()
        old_total = int(self.num_frames)
        target = None
        if total_frames is not None:
            target = int(total_frames)
        elif extra_frames is not None:
            target = old_total + int(extra_frames)
        if target is None or target <= old_total:
            return {"data_frames": data_frames, "total_frames": old_total,
                    "old_total_frames": old_total, "extended_tracks": 0, "mode": mode}
        cap = int(os.environ.get("DGGT_MAX_FRAMES", "4000"))
        target = max(data_frames, min(cap, target))
        targets = list(range(old_total, target))
        if not targets:
            return {"data_frames": data_frames, "total_frames": old_total,
                    "old_total_frames": old_total, "extended_tracks": 0, "mode": mode}

        # 1) 自车（相机 + 车体）：优先保证"车后面的行动能延续"
        ego_poses = None
        if self.ego_track_id is not None:
            s = self.synthetic_tracks.get(self.ego_track_id)
            if s and s.get("poses"):
                ego_poses = {int(f): np.asarray(v, dtype=np.float32) for f, v in s["poses"].items()}
                new = self._extrapolate_poses_smart(ego_poses, targets,
                                                    mode=(ego_mode or mode),
                                                    dims=(s.get("dimensions") or None))
                s["poses"].update(new)
                s["base_poses"].update({f: v.copy() for f, v in new.items()})
                self._invalidate_heading_cache(self.ego_track_id)
                self._ego_cams = dict(getattr(self, "_ego_cams", {}))

        # 2) 真实 track：外推位姿，并记下稳定渲染 raw id
        n_ext = 0
        if extend_tracks:
            for tid in list(self.track_raw_poses.keys()):
                if tid in self.track_deleted or tid in self.track_replaced:
                    continue
                poses = {int(f): np.asarray(v, dtype=np.float32)
                         for f, v in (self.track_raw_poses.get(tid) or {}).items()}
                # 之前外推出来的帧也要算"已有"：否则多次延长时锚点永远停在最后一个**数据**帧，
                # 每次都要重算一遍，稍有不一致就会在拼接处出现重复/错位。
                for f, v in (self.track_extended.get(int(tid)) or {}).items():
                    poses[int(f)] = np.asarray(v, dtype=np.float32)
                edits = self.track_edits.get(tid) or {}
                for f, v in edits.items():                     # 编辑过的帧也算"已有"
                    poses[int(f)] = np.asarray(v, dtype=np.float32)
                if len(poses) < 2:
                    continue
                try:
                    tdims = list(self.get_track_dimensions(int(tid)))
                except Exception:  # noqa: BLE001
                    tdims = None
                new = self._extrapolate_poses_smart(poses, targets, mode=mode, dims=tdims)
                if not new:
                    continue
                self.track_extended.setdefault(int(tid), {}).update(new)
                self._invalidate_heading_cache(int(tid))
                n_ext += 1

            # 3) 合成物体（含 SAM3D 替换物体、事故参与者）
            for sid, spec in list(self.synthetic_tracks.items()):
                if spec.get("ego"):
                    continue
                poses = spec.get("poses") or {}
                if len(poses) < 2:
                    continue
                new = self._extrapolate_poses_smart({int(f): v for f, v in poses.items()}, targets,
                                                    mode=mode,
                                                    dims=(spec.get("dimensions") or None))
                if new:
                    spec["poses"].update(new)
                    self._invalidate_heading_cache(int(sid))
                    n_ext += 1

        self.num_frames = int(target)
        self.extended_from_frame = old_total
        try:
            self.renderer.extend_frames(total_frames=target)
        except Exception:  # noqa: BLE001
            pass
        self._rebuild_render_id_map()
        return {"data_frames": data_frames, "total_frames": int(self.num_frames),
                "old_total_frames": old_total, "extended_tracks": int(n_ext), "mode": mode}

    def _rebuild_render_id_map(self):
        """建立 (view, raw_id) -> track_id 与 track -> 稳定 raw_id 的映射。

        超范围帧没有逐帧元数据，渲染器只能拿"该物体最后一次出现"的外观；
        这里把复合 id 反查回 track，位姿就能继续由轨迹给。
        """
        self._view_raw_to_track = {}
        self._track_render_raw = {}
        nv = int(getattr(self.renderer, "num_views", 1) or 1)
        for tid, frames in self.track_to_frames.items():
            if not frames:
                continue
            f_last = max(frames.keys())
            raw = int(frames[f_last])
            view = raw // VIEW_ID_STRIDE if nv > 1 else 0
            raw_id = raw % VIEW_ID_STRIDE if nv > 1 else raw
            self._track_render_raw[int(tid)] = raw_id
            self._view_raw_to_track[(view, raw_id)] = int(tid)

    def ensure_frames(self, frame_idx=None, num_frames=None):
        """按需把时间轴延长到能覆盖 frame_idx / num_frames（超出就自动外推）。"""
        need = 0
        if frame_idx is not None:
            need = max(need, int(frame_idx) + 1)
        if num_frames is not None:
            need = max(need, int(num_frames))
        if need > int(self.num_frames):
            return self.extend_timeline(total_frames=need)
        return None

    # ==================== 主车（EGO）实体化 ====================

    def _init_ego_track(self):
        """把主车实体化：从逐帧自车相机外参推导车体位姿，作为 900000 号合成物体。

        之后它会像其它物体一样：出现在 3D 视图 / 2D 叠加 / BEV 里，可以被拖动编辑轨迹，
        也可以沿编辑后的轨迹重新渲染"主车视角"视频（相机与车体保持固定安装关系）。
        """
        try:
            ply = os.environ.get("DGGT_EGO_PLY", EGO_PLY_DEFAULT)
            renderer = self.renderer
            ego_dir = getattr(renderer, "ego_dir", None)
            if not ply or not os.path.exists(ply) or not ego_dir:
                return
            cams, poses = {}, {}
            for f in range(int(getattr(self, "num_frames", 0))):
                try:
                    flat = renderer.flat_index(f, 0)
                except Exception:  # noqa: BLE001
                    continue
                fp = os.path.join(ego_dir, f"frame_{flat:04d}_ego.json")
                if not os.path.exists(fp):
                    continue
                with open(fp, "r") as fh:
                    data = json.load(fh)
                ext = data.get("camera_extrinsics_world")
                if ext is None:
                    continue
                c2w = np.asarray(ext, dtype=np.float32)
                if c2w.shape == (3, 4):
                    c2w = np.vstack([c2w, np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32)])
                cams[int(f)] = c2w
                poses[int(f)] = _pose_from_camera(c2w)
            if len(poses) < 2:
                return
            # 按真实车长缩放模型；车长优先参照"场景里已有的车"，避免自车比场景车明显大
            ext = _ply_axis_extents(ply)
            ref_dims = _scene_vehicle_dims(getattr(self, "track_meta", {}))
            ego_fit = float(os.environ.get("DGGT_EGO_FIT", "1.0"))
            # fit_inside（默认）：等比缩到"塞进场景中位车的包围盒"（取三轴最小比例）；
            # length：只按车长对齐（旧行为，自车会比场景车高/宽一圈）。
            ego_fit_mode = str(os.environ.get("DGGT_EGO_FIT_MODE", "fit_inside")).strip().lower()
            ego_len = (ref_dims[2] if ref_dims else None) or _scene_vehicle_length(
                getattr(self, "track_meta", {})) or EGO_LENGTH_M
            ego_len *= ego_fit
            ego_len = float(np.clip(ego_len, 1.5, 5.2))
            scale, dims = 1.0, [1.86, 1.45, ego_len]
            if ext and ext[0] > 1e-3 and ext[1] > 1e-3 and ext[2] > 1e-3:
                # ext: SAM3D canonical → x=车宽, y=车长, z=车高
                # **等比**缩放到"能套进场景车辆的中位包围盒"（取三轴比例的最小值），
                # 只按车长缩会让自车比场景车高出一截/宽出一圈（车模的三维比例和
                # 4DGS 拟合出来的包围盒比例不一样），用户看到的就是"主车过大"。
                # 取 min 之后：长宽高都 ≤ 场景车，比例不变（不变形）。
                if ref_dims and ego_fit_mode != "length":
                    s_fit = min(float(ref_dims[0]) / float(ext[0]),
                                float(ref_dims[1]) / float(ext[2]),
                                float(ref_dims[2]) / float(ext[1]))
                    s_fit *= ego_fit
                else:
                    s_fit = ego_len / float(ext[1])
                scale = float(max(1e-6, s_fit))
                dims = [round(float(ext[0]) * scale, 3), round(float(ext[2]) * scale, 3),
                        round(float(ext[1]) * scale, 3)]
            # 自动贴地：不同场景自车相机安装高度不同，用 4DGS 静态地面把主车放到路面上，
            # 避免"车体沉进地面/穿模"。贴地前的位姿留作 refit/回退用。
            pose_no_ground = {f: p.copy() for f, p in poses.items()}
            self._ego_cams = {int(f): np.asarray(c, dtype=np.float32).copy() for f, c in cams.items()}
            self._ego_pose_no_ground = pose_no_ground
            ground_info = {"applied": False, "reason": "auto_ground_disabled"}
            if EGO_AUTO_GROUND:
                poses, ground_info = self._auto_ground_ego(poses, cams, dims)
            f_first = sorted(poses.keys())[0]
            cam_rel = np.linalg.inv(poses[f_first]) @ cams[f_first]
            self.synthetic_tracks[EGO_TRACK_ID] = {
                "ply_path": str(ply),
                "ego": True,
                "type": "主车(EGO)",
                "dimensions": list(dims),
                "scale": float(scale),
                "base_scale": float(scale),
                "base_dimensions": list(dims),
                "base_scale_vec": None,
                "center": None,
                "model_corr": SAM3D_MODEL_CORR.copy(),
                "poses": {f: p.copy() for f, p in poses.items()},
                "base_poses": {f: p.copy() for f, p in poses.items()},
                "cam_rel": cam_rel.astype(np.float32),
                "auto_ground": bool(ground_info.get("applied")),
                "ground": ground_info,
                "visible": True,
            }
            self.ego_track_id = EGO_TRACK_ID
            self.ego = {"track_id": EGO_TRACK_ID, "ply_path": str(ply), "scale": float(scale),
                        "dimensions": list(dims), "num_frames": len(poses),
                        "auto_ground": bool(ground_info.get("applied")),
                        "ground": ground_info}
            self._invalidate_heading_cache(EGO_TRACK_ID)
        except Exception as e:  # noqa: BLE001
            print(f"[ego] 初始化主车实体失败: {e}")

    def _auto_ground_ego(self, poses, cams, dims, clearance=None, radius=None, points=None):
        """把主车逐帧放到"自车路径下方的 4DGS 地面"上，返回 (新位姿, 说明)。

        做法：在每帧相机 (x,z) 邻域内取静态高斯的 y 分位数作为路面高度，
        令车体中心 = 地面 + 半个车高（沿世界"上"方向），再换算成相机坐标系下
        的下向距离（EGO_CAM_OFFSET 的 y 分量）。这样相机安装高度不同的场景都能自适应。
        """
        if not poses:
            return poses, {"applied": False, "reason": "no_poses"}
        clearance = EGO_GROUND_CLEARANCE if clearance is None else float(clearance)
        radius = EGO_GROUND_RADIUS if radius is None else float(radius)
        if points is None:
            points = _scene_points_cpu(self.renderer, EGO_GROUND_MAX_POINTS)
        if points is None or len(points) == 0:
            return poses, {"applied": False, "reason": "no_static_points"}
        car_h = float(dims[1]) if len(dims) > 1 and dims[1] else 1.55

        frames = sorted(poses.keys())
        down_ys = []
        for f in frames:
            c = cams.get(f)
            if c is not None:
                down_ys.append(float(np.asarray(c, dtype=np.float32)[1, 1]))
        if not down_ys:
            return poses, {"applied": False, "reason": "no_camera"}
        # 相机 y 轴（图像"下"）在世界里的 y 分量；>0 表示世界 +y 向下。
        down_y = float(np.median(down_ys))
        up_sign = -1.0 if down_y > 0 else 1.0
        self._world_up_sign = float(up_sign)

        # 逐帧估计地面 → 车体中心的**世界 y**，对世界 y 平滑（而不是对"相机到下向距离"
        # 平滑），这样相机外参偶发的跳变不会把车带偏。
        raw_center, raw_ground, raw_cam_h = {}, {}, {}
        for f in frames:
            c = cams.get(f)
            if c is None:
                continue
            c = np.asarray(c, dtype=np.float32)
            eye = c[:3, 3]
            gy = _ground_height_near(points, eye, radius, up_sign)
            if gy is None:
                continue
            # 相机到地面沿"世界下"方向的距离（地面在下为正）
            cam_to_ground = (gy - float(eye[1])) * (-up_sign)
            if not np.isfinite(cam_to_ground):
                continue
            cam_to_ground = float(np.clip(cam_to_ground, 0.3, 4.0))
            raw_ground[f] = float(gy)
            raw_cam_h[f] = cam_to_ground
            # 目标车体中心：地面 + 半个车高 + 离地余量（沿世界"上"）
            raw_center[f] = float(gy) + up_sign * (0.5 * car_h + clearance)
        if not raw_center:
            return poses, {"applied": False, "reason": "no_ground_estimate"}

        ordered = sorted(raw_center.keys())
        smoothed_center = _smooth_series([raw_center[f] for f in ordered], window=5)
        new_poses = dict(poses)
        down_dists = []
        for f, cy in zip(ordered, smoothed_center):
            c = cams.get(f)
            if c is None:
                continue
            c = np.asarray(c, dtype=np.float32)
            # 由"目标中心的世界 y"反算相机坐标系下的下向偏移（相机装得低时可为负）
            down = (float(cy) - float(c[1, 3])) * (-up_sign)
            down = float(np.clip(down, -1.5, 4.0))
            down_dists.append(down)
            off = EGO_CAM_OFFSET.copy()
            off[1] = down                            # 相机 y（下）方向的距离
            new_poses[f] = _pose_from_camera(c, off)
        gvals = np.asarray([raw_ground[f] for f in ordered], dtype=np.float64)
        info = {
            "applied": True,
            "method": "static_point_percentile",
            "radius": float(radius),
            "clearance": clearance,
            "car_height": car_h,
            "up_sign": int(up_sign),
            "ground_y_median": float(np.median(gvals)),
            "cam_height_median": float(np.median([raw_cam_h[f] for f in ordered])),
            "cam_offset_down_median": float(np.median(down_dists)) if down_dists else 0.0,
            "cam_offset_down_default": float(EGO_CAM_OFFSET[1]),
            "num_frames": len(ordered),
        }
        return new_poses, info

    def refit_ego_ground(self, clearance=None, radius=None, points=None):
        """重新按当前场景地面调整主车高度（保留相机轨迹，忽略用户对主车的轨迹编辑）。

        返回 ground 说明 dict；主车不存在时返回 None。
        """
        if self.ego_track_id is None:
            return None
        s = self.synthetic_tracks.get(self.ego_track_id)
        if not s:
            return None
        base = self._ego_pose_no_ground or {f: p for f, p in (s.get("poses") or {}).items()}
        cams = self._ego_cams or {}
        dims = s.get("dimensions") or [1.95, 1.55, EGO_LENGTH_M]
        new_poses, info = self._auto_ground_ego(base, cams, dims,
                                                clearance=clearance, radius=radius, points=points)
        if not info.get("applied"):
            return info
        self.push_history()
        s["poses"] = {f: p.copy() for f, p in new_poses.items()}
        s["base_poses"] = {f: p.copy() for f, p in new_poses.items()}
        s["auto_ground"] = True
        s["ground"] = info
        self.ego["auto_ground"] = True
        self.ego["ground"] = info
        self._invalidate_heading_cache(self.ego_track_id)
        return info

    def reset_ego_ground(self):
        """还原到"自动贴地之前"的相机轨迹（关闭自动贴地时用）。"""
        if self.ego_track_id is None:
            return {"applied": False, "reason": "no_ego"}
        s = self.synthetic_tracks.get(self.ego_track_id)
        if not s or not self._ego_pose_no_ground:
            return {"applied": False, "reason": "no_base_poses"}
        self.push_history()
        s["poses"] = {f: p.copy() for f, p in self._ego_pose_no_ground.items()}
        s["base_poses"] = {f: p.copy() for f, p in self._ego_pose_no_ground.items()}
        s["auto_ground"] = False
        s["ground"] = {"applied": False, "reason": "revoked"}
        self.ego["auto_ground"] = False
        self.ego["ground"] = s["ground"]
        self._invalidate_heading_cache(self.ego_track_id)
        return s["ground"]


    def is_ego_track(self, track_id):
        try:
            return self.ego_track_id is not None and int(track_id) == int(self.ego_track_id)
        except (TypeError, ValueError):
            return False

    def world_up_sign(self):
        """世界 +y 是"上"(+1) 还是"下"(-1)；阴影/贴地等都要用。"""
        up = getattr(self, "_world_up_sign", None)
        if up is not None:
            return float(up)
        cams = getattr(self, "_ego_cams", {}) or {}
        if cams:
            down_ys = [float(np.asarray(c, dtype=np.float32)[1, 1]) for c in cams.values()]
            self._world_up_sign = -1.0 if float(np.median(down_ys)) > 0 else 1.0
        else:
            self._world_up_sign = -1.0
        return float(self._world_up_sign)

    def ground_y_at(self, x, z, radius=None):
        """估计 (x, z) 处 4DGS 路面的世界 y（没有静态点/邻域太稀时返回 None）。

        接触阴影必须贴在**真实路面**上，而不是物体包围盒底面：4DGS 重建出的包围盒
        在高度方向普遍不准（偏扁、还可能整体悬浮），照包围盒底面画阴影会飘在空中。
        """
        try:
            pts = getattr(self, "_ground_points", None)
            if pts is None:
                pts = _scene_points_cpu(self.renderer, EGO_GROUND_MAX_POINTS)
                self._ground_points = pts
            if pts is None or len(pts) == 0:
                return None
            r = EGO_GROUND_RADIUS if radius is None else float(radius)
            eye = (float(x), 0.0, float(z))
            return _ground_height_near(pts, eye, r, self.world_up_sign())
        except Exception:  # noqa: BLE001
            return None

    def get_ego_camera(self, frame_idx):
        """"主车视角"在 frame_idx 的相机位姿（c2w）= 车体位姿 × 固定安装关系。

        默认车体 = 真实主车实体（900000，沿原始/编辑后的相机轨迹）；
        `set_ego_source(track_id)` 之后，车体 = 该动态物体的位姿 —— 即"以这个物体的视角"。
        """
        if self.ego_track_id is None:
            return None
        s = self.synthetic_tracks.get(self.ego_track_id)
        if not s:
            return None
        src = self.ego_source_track
        track = self.ego_track_id if src is None else int(src)
        pose = self.get_track_pose(track, frame_idx)
        if pose is None:
            return None
        rel = s.get("cam_rel")
        if rel is None:
            return None
        return (np.asarray(pose, dtype=np.float32) @ np.asarray(rel, dtype=np.float32))

    def set_ego_source(self, track_id=None):
        """切换"用谁的眼睛看"：None/900000 = 真实主车；其它 = 指定动态物体。"""
        if track_id in (None, 0, "0", "", self.ego_track_id):
            self.ego_source_track = None
            return self.get_ego_config()
        tid = int(track_id)
        if tid not in self.track_meta and tid not in self.synthetic_tracks:
            raise ValueError(f"track {tid} 不存在")
        if tid in self.track_deleted or tid in self.track_replaced:
            raise ValueError(f"track {tid} 已被删除/替换")
        if not self.get_track_frames(tid):
            raise ValueError(f"track {tid} 没有轨迹")
        self.ego_source_track = tid
        return self.get_ego_config()

    def _track_display_type(self, tid):
        meta = self.track_meta.get(int(tid), {}) or {}
        tname = meta.get("type") or "动态物体"
        return "动态物体" if str(tname) in ("未知", "unknown", "None") else tname

    def get_ego_config(self):
        """主车视角配置 + 可选的"视角来源"列表（用于前端下拉框）。"""
        opts = []
        if self.ego_track_id is not None:
            opts.append({"track_id": int(self.ego_track_id), "type": "真实主车(EGO)",
                         "num_frames": len(self.synthetic_tracks[self.ego_track_id].get("poses") or {})})
        for tid, meta in sorted(self.track_meta.items()):
            if tid in self.track_deleted or tid in self.track_replaced:
                continue
            n = len(self.get_track_frames(tid))
            if n < 2:
                continue
            tname = meta.get("type") or "动态物体"
            if str(tname) in ("未知", "unknown", "None"):
                tname = "动态物体"
            opts.append({"track_id": int(tid), "type": tname,
                         "num_frames": n,
                         "first_frame": meta.get("first_frame")})
        src = self.ego_source_track
        return {
            "default_track_id": None if self.ego_track_id is None else int(self.ego_track_id),
            "source_track_id": None if src is None else int(src),
            "source_type": None if src is None else (
                "真实主车(EGO)" if src == self.ego_track_id
                else self._track_display_type(src)),
            "options": opts,
        }

    def ego_is_edited(self):
        """主车轨迹是否被编辑过（相对初始化时的相机轨迹）。"""
        if self.ego_track_id is None:
            return False
        s = self.synthetic_tracks.get(self.ego_track_id) or {}
        base, cur = s.get("base_poses") or {}, s.get("poses") or {}
        if set(base.keys()) != set(cur.keys()):
            return True
        for f, p in cur.items():
            b = base.get(f)
            if b is None or not np.allclose(np.asarray(p), np.asarray(b), atol=1e-3):
                return True
        return False

    def set_ego_visible(self, visible):
        if self.ego_track_id is None:
            return False
        self.synthetic_tracks[self.ego_track_id]["visible"] = bool(visible)
        return True

    def reset_ego_track(self):
        """把主车轨迹还原为原始相机轨迹（只重置轨迹，保留物体）。"""
        if self.ego_track_id is None:
            return False
        self._invalidate_heading_cache(self.ego_track_id)
        return self.reset_synthetic_to_base(self.ego_track_id)

    # ==================== 撤销/重做 ====================

    def _copy_track(self, s):
        sc = dict(s)
        sc["dimensions"] = list(s.get("dimensions", []))
        sc["poses"] = {f: p.copy() for f, p in s["poses"].items()}
        if s.get("base_poses"):
            sc["base_poses"] = {f: np.asarray(p, dtype=np.float32).copy() for f, p in s["base_poses"].items()}
        if sc.get("center") is not None:
            sc["center"] = np.asarray(sc["center"], dtype=np.float32).copy()
        if sc.get("model_corr") is not None:
            sc["model_corr"] = np.asarray(sc["model_corr"], dtype=np.float32).copy()
        if sc.get("scale_vec") is not None:
            sc["scale_vec"] = np.asarray(sc["scale_vec"], dtype=np.float32).copy()
        if sc.get("base_scale_vec") is not None:
            sc["base_scale_vec"] = np.asarray(sc["base_scale_vec"], dtype=np.float32).copy()
        if sc.get("donor_pose0") is not None:
            sc["donor_pose0"] = np.asarray(sc["donor_pose0"], dtype=np.float32).copy()
        return sc

    def _snapshot(self):
        edits = {tid: {f: p.copy() for f, p in fp.items()} for tid, fp in self.track_edits.items()}
        synth = {sid: self._copy_track(s) for sid, s in self.synthetic_tracks.items()}
        rots = {tid: np.asarray(R, dtype=np.float32).copy() for tid, R in self.track_rotations.items()}
        return (edits, set(self.track_deleted), synth, self._next_synth_id, rots,
                set(getattr(self, "track_replaced", set())))

    def _restore(self, snap):
        edits, deleted, synth, next_id, rots = snap[:5]
        self.track_edits = {tid: {f: p.copy() for f, p in fp.items()} for tid, fp in edits.items()}
        self.track_deleted = set(deleted)
        self.synthetic_tracks = {sid: self._copy_track(s) for sid, s in synth.items()}
        self._next_synth_id = next_id
        self.track_rotations = {tid: np.asarray(R, dtype=np.float32).copy() for tid, R in rots.items()}
        self.track_replaced = set(snap[5]) if len(snap) > 5 else set()
        self._invalidate_heading_cache()

    def push_history(self):
        """在一次编辑操作 **之前** 调用，保存当前状态以便撤销。"""
        self._undo_stack.append(self._snapshot())
        if len(self._undo_stack) > self._max_history:
            self._undo_stack.pop(0)
        self._redo_stack.clear()

    def undo(self):
        if not self._undo_stack:
            return False
        self._redo_stack.append(self._snapshot())
        self._restore(self._undo_stack.pop())
        return True

    def redo(self):
        if not self._redo_stack:
            return False
        self._undo_stack.append(self._snapshot())
        self._restore(self._redo_stack.pop())
        return True

    def can_undo(self):
        return len(self._undo_stack) > 0

    def can_redo(self):
        return len(self._redo_stack) > 0


    def _count_frames(self):
        # 多视角数据：使用合并后的“真实帧数”
        return self.renderer.frame_count()

    def _build(self):
        """构建 track 映射（一次性）。"""
        track_frames = self.renderer._build_track_id_map(self.num_frames, start_idx=0)
        for frame_idx, matched in track_frames:
            for obj in matched:
                tid = int(obj["track_id"])
                raw_id = int(obj["object_id"])
                self.track_to_frames.setdefault(tid, {})[frame_idx] = raw_id
                self.frame_obj_to_track[(frame_idx, raw_id)] = tid

                pose = np.asarray(obj["pose_world"], dtype=np.float32)
                self.track_raw_poses.setdefault(tid, {})[frame_idx] = pose

                if tid not in self.track_meta:
                    self.track_meta[tid] = {
                        "track_id": tid,
                        "type": obj.get("type", "未知"),
                        "dimensions": obj.get("dimensions", []),
                        "first_frame": frame_idx,
                    }

        self._rebuild_render_id_map()

    # ==================== 查询 ====================

    def get_track_id(self, frame_idx, raw_object_id):
        return self.frame_obj_to_track.get((int(frame_idx), int(raw_object_id)))

    def get_raw_object_id(self, track_id, frame_idx):
        """返回某 track 在指定帧的 raw object_id。合成 track 无 raw id（返回 None）。

        超出数据帧时该帧没有逐帧元数据，返回"该物体最后一次出现"的 raw id —— 渲染器
        用同一 id 复用它的外观，位姿则由轨迹（track_extended / object_pose_provider）给。
        """
        tid = int(track_id)
        rid = self.track_to_frames.get(tid, {}).get(int(frame_idx))
        if rid is not None:
            return rid
        if (self.track_extended.get(tid) or {}).get(int(frame_idx)) is not None:
            return self._track_render_raw.get(tid)
        return None

    def is_synthetic(self, track_id):
        return int(track_id) in self.synthetic_tracks

    def get_track_frames(self, track_id):
        """该 track 有轨迹的帧（含**超出原数据范围的外推帧**）。

        延长过的 track 在数据帧之外也有位姿，所以这里要并上 `track_extended`，
        否则下游（轨迹显示、碰撞分析、语言编辑…）仍会以为它"到原场景末尾就没了"。
        """
        track_id = int(track_id)
        if track_id in self.synthetic_tracks:
            return sorted(self.synthetic_tracks[track_id]["poses"].keys())
        frames = set(self.track_to_frames.get(track_id, {}).keys())
        frames |= set((self.track_extended.get(track_id) or {}).keys())
        return sorted(frames)

    def get_track_dimensions(self, track_id):
        track_id = int(track_id)
        if track_id in self.synthetic_tracks:
            return self.synthetic_tracks[track_id].get("dimensions", [4.5, 2.0, 1.6])
        meta = self.track_meta.get(track_id, {})
        dims = meta.get("dimensions")
        return dims if dims else [4.5, 2.0, 1.6]

    # ==================== 合成参与者 ====================

    def create_synthetic_track(self, donor_track_id, poses, dimensions=None, type_name="合成参与者"):
        """创建一个合成参与者：克隆 donor_track 的高斯外观，按给定逐帧位姿放置。

        Args:
            donor_track_id: 用于克隆外观的真实 track（提供 raw object 高斯）
            poses: {frame_idx: 4x4 np} 逐帧世界位姿
            dimensions: 包围盒尺寸，默认沿用 donor 的
            type_name: 物体类型名

        Returns:
            新建的合成 track_id
        """
        donor_track_id = int(donor_track_id)
        # 找一个 donor 在某帧的 raw object_id 作为高斯克隆来源
        donor_frames = self.track_to_frames.get(donor_track_id, {})
        if not donor_frames:
            raise ValueError(f"donor track {donor_track_id} 无可克隆的高斯")
        donor_raw_id = donor_frames[sorted(donor_frames.keys())[0]]
        donor_pose0 = self.track_raw_poses[donor_track_id][sorted(donor_frames.keys())[0]]

        if dimensions is None:
            dimensions = self.get_track_dimensions(donor_track_id)

        sid = self._next_synth_id
        self._next_synth_id += 1
        self.synthetic_tracks[sid] = {
            "donor_object_id": int(donor_raw_id),
            "donor_track_id": donor_track_id,
            "donor_pose0": np.asarray(donor_pose0, dtype=np.float32).copy(),
            "dimensions": list(dimensions),
            "type": type_name,
            "poses": {int(f): np.asarray(p, dtype=np.float32) for f, p in poses.items()},
        }
        self._invalidate_heading_cache(sid)
        return sid

    def add_sam3d_track(self, ply_path, poses, dimensions, scale=1.0, center=None,
                        model_corr=None, type_name="SAM3D物体", scale_vec=None):
        """创建一个 SAM3D 重建物体 track：自带 .ply 高斯（非克隆），按逐帧位姿摆放。

        Args:
            ply_path: 重建出的 3D Gaussian Splat (.ply) 路径
            poses: {frame_idx: 4x4 np} 逐帧世界位姿（与目标物体轨迹一致）
            dimensions: 包围盒尺寸 [宽, 高, 长]（沿用目标物体）
            scale: 均匀缩放系数（用 scale_vec 时仅作兼容展示）
            center: 重建物体质心 [x,y,z]（渲染时先归零）
            model_corr: 3x3 模型朝向修正（只作用于 gaussians，不影响包围盒）
            scale_vec: [sx,sy,sz] **逐轴**缩放（在 model_corr 之后的物体局部系里作用）。
                       用于把重建模型精确套进目标物体的包围盒（宽/高/长分别对齐），
                       避免 SAM3D 模型比原模型"大一圈"。
            type_name: 类型名

        Returns:
            新建的合成 track_id（>= 100000）
        """
        sid = self._next_synth_id
        self._next_synth_id += 1
        poses_np = {int(f): np.asarray(p, dtype=np.float32) for f, p in poses.items()}
        sv = None if scale_vec is None else np.asarray(scale_vec, dtype=np.float32).reshape(3)
        if sv is not None:
            scale = float(np.cbrt(max(1e-9, float(sv[0]) * float(sv[1]) * float(sv[2]))))
        self.synthetic_tracks[sid] = {
            "ply_path": str(ply_path),
            "dimensions": list(dimensions),
            "type": type_name,
            "scale": float(scale),
            # 替换/创建时自动算出的基准尺寸：前端滑块按倍数调整时用它做基准
            "base_scale": float(scale),
            "base_dimensions": list(dimensions),
            "scale_vec": sv,
            "base_scale_vec": None if sv is None else sv.copy(),
            "center": None if center is None else np.asarray(center, dtype=np.float32).reshape(3),
            "model_corr": None if model_corr is None else np.asarray(model_corr, dtype=np.float32),
            "poses": poses_np,
            # 替换时刻的原始轨迹备份：corner case 等编辑后可用它把物体还原，
            # 而不会把物体本身删掉。
            "base_poses": {f: p.copy() for f, p in poses_np.items()},
        }
        self._invalidate_heading_cache(sid)
        return sid

    def add_pedestrian_track(self, poses, dimensions=None, type_name="行人", ply_path=None,
                             choice_key=None):
        """用行人 3D 模型（person_*.ply，随机一个）创建一个合成的行人物体。

        没有可用模型时抛异常，调用方可以退回"克隆 donor 外观"的老做法。
        """
        spec = ply_path or os.environ.get("DGGT_PED_PLYS", PED_PLY_DEFAULT)
        plies = [p for p in str(spec).split(":") if p and os.path.exists(p)]
        if not plies:
            raise FileNotFoundError("没有可用的行人模型(.ply)，可用 DGGT_PED_PLYS 指定")
        # 外观多样 + **可复现**：有 choice_key 时用稳定哈希选（同一条轨迹 → 同一个模型），
        # 没有才退回真随机。避免"同 seed 重跑结果不同"。
        if choice_key is not None:
            import zlib
            ply = plies[zlib.crc32(str(choice_key).encode("utf-8")) % len(plies)]
        else:
            ply = random.choice(plies)
        ext = _ply_axis_extents(ply)
        if not ext or ext[2] < 1e-3:
            dims = list(dimensions or [0.6, PED_TARGET_HEIGHT_M, 0.5])
            scale = 1.0
        else:
            # SAM3D canonical: z = 身高方向
            scale = float(PED_TARGET_HEIGHT_M) / float(ext[2])
            dims = [round(ext[0] * scale, 3), round(ext[2] * scale, 3), round(ext[1] * scale, 3)]
        sid = self.add_sam3d_track(ply, poses, dims, scale=float(scale), center=None,
                                   model_corr=SAM3D_MODEL_CORR, type_name=type_name)
        # 事故临时合成的物体：清除生成轨迹时应整段移除，而不是"保留模型"
        self.synthetic_tracks[int(sid)]["ephemeral"] = True
        return sid

    def reset_synthetic_to_base(self, track_id):
        """把合成物体（如 SAM3D 替换物体）的轨迹还原到创建时的原始轨迹。

        与 remove_synthetic_track 不同：**保留物体本身**，只撤销后续对轨迹的修改。
        返回是否成功还原。
        """
        tid = int(track_id)
        s = self.synthetic_tracks.get(tid)
        if not s:
            return False
        base = s.get("base_poses")
        if not base:
            return False
        s["poses"] = {f: np.asarray(p, dtype=np.float32).copy() for f, p in base.items()}
        self._invalidate_heading_cache(tid)
        return True

    def snapshot_synthetic_poses(self):
        """在生成 corner case **之前**调用：备份持久合成物体（SAM3D 替换物体）的当前轨迹。

        这样"清除生成的事故轨迹"只撤销事故造成的轨迹改动，而不会把用户之前
        手动调整过的轨迹一起清掉。
        """
        self._corner_poses_backup = {
            int(sid): {int(f): np.asarray(p, dtype=np.float32).copy() for f, p in s.get("poses", {}).items()}
            for sid, s in self.synthetic_tracks.items()
            if s.get("ply_path") and not s.get("ephemeral") and not s.get("ego")
        }
        return sorted(self._corner_poses_backup.keys())

    def restore_synthetic_from_corner_backup(self, track_id):
        """优先用 corner case 之前的备份还原合成物体轨迹；没有备份则退回 base_poses。

        `ephemeral=True` 的物体（事故临时合成的行人等）不参与还原 → 清除时会被删除。
        """
        tid = int(track_id)
        s0 = self.synthetic_tracks.get(tid)
        if s0 is not None and s0.get("ephemeral"):
            return False
        backup = getattr(self, "_corner_poses_backup", {}).get(tid)
        s = self.synthetic_tracks.get(tid)
        if s is not None and backup:
            s["poses"] = {f: np.asarray(p, dtype=np.float32).copy() for f, p in backup.items()}
            self._invalidate_heading_cache(tid)
            return True
        return self.reset_synthetic_to_base(tid)

    def is_sam3d_track(self, track_id):
        s = self.synthetic_tracks.get(int(track_id))
        return bool(s and s.get("ply_path"))

    def list_sam3d_tracks(self):
        """列出所有 SAM3D 重建物体 track 的概要。"""
        out = []
        for sid, s in self.synthetic_tracks.items():
            if not s.get("ply_path") or s.get("ephemeral") or s.get("ego"):
                continue
            frames = sorted(s["poses"].keys())
            pose = s["poses"][frames[0]] if frames else np.eye(4, dtype=np.float32)
            out.append({
                "object_id": int(sid),
                "track_id": int(sid),
                "ply_path": s["ply_path"],
                "scale": s.get("scale", 1.0),
                "center": s.get("center").tolist() if s.get("center") is not None else None,
                "dimensions": s.get("dimensions", []),
                "pose_world": pose.tolist(),
                "num_frames": len(frames),
                "meta": {"replaced": True},
            })
        return out

    def get_sam3d_track_ply(self, track_id):
        s = self.synthetic_tracks.get(int(track_id))
        return s.get("ply_path") if s else None

    def set_synthetic_pose(self, synth_id, frame_idx, pose):
        synth_id = int(synth_id)
        if synth_id not in self.synthetic_tracks:
            return
        self.synthetic_tracks[synth_id]["poses"][int(frame_idx)] = np.asarray(pose, dtype=np.float32)
        self._invalidate_heading_cache(synth_id)

    def remove_synthetic_track(self, synth_id):
        self.synthetic_tracks.pop(int(synth_id), None)
        self._invalidate_heading_cache(synth_id)

    def is_deleted(self, track_id):
        return int(track_id) in self.track_deleted

    # ==================== 位姿求值（含编辑） ====================

    # ==================== 车头自动朝向运动方向 ====================

    def _invalidate_heading_cache(self, track_id=None):
        if track_id is None:
            self._heading_cache.clear()
        else:
            self._heading_cache.pop(int(track_id), None)

    def set_auto_heading(self, enabled=None, track_ids=None, smoothing=None,
                         window=None, all_tracks=False, max_yaw_rate=None, fps=None):
        """配置“车头自动朝向运动方向”。

        enabled: 总开关；track_ids: 只对这些 track 生效（all_tracks=True 或 None 表示全部）；
        smoothing: 0~1 越大越平滑；window: 方向估计的 ±帧窗口。
        """
        if enabled is not None:
            self.auto_heading = bool(enabled)
        if smoothing is not None:
            self.auto_heading_smoothing = float(min(1.0, max(0.0, smoothing)))
        if window is not None:
            self.auto_heading_window = int(max(1, min(15, window)))
        if max_yaw_rate is not None:
            self.auto_heading_max_yaw_rate = float(max(0.05, min(6.0, max_yaw_rate)))
            self._invalidate_heading_cache()
        if fps is not None:
            self.fps = float(max(1e-3, fps))
            self._invalidate_heading_cache()
        if all_tracks or track_ids is None:
            self.auto_heading_tracks = None
        else:
            self.auto_heading_tracks = {int(t) for t in track_ids}
        self._invalidate_heading_cache()
        return self.get_auto_heading_config()

    def get_auto_heading_config(self):
        return {
            "enabled": bool(self.auto_heading),
            "smoothing": float(self.auto_heading_smoothing),
            "window": int(self.auto_heading_window),
            "max_yaw_rate_radps": float(self.auto_heading_max_yaw_rate),
            "fps": float(self.fps),
            "track_ids": None if self.auto_heading_tracks is None else sorted(self.auto_heading_tracks),
            "all_tracks": self.auto_heading_tracks is None,
        }

    def _is_auto_heading_track(self, track_id):
        if not self.auto_heading:
            return False
        if self.auto_heading_tracks is not None and int(track_id) not in self.auto_heading_tracks:
            return False
        return bool(self.get_track_frames(track_id))

    def _forward_sign(self, track_id):
        """物体局部系的“车头轴”：+1 表示车头是 +Z，-1 表示车头是 -Z。

        本数据集的物体位姿旋转全是单位阵（4D 重建把每个物体 canonical 化到世界轴），
        所以“车头朝 +Z 还是 -Z”**逐物体不同**，不能用一个全局约定。这里逐个推断：

        - SAM3D 重建物体：`model_corr` 已把车头对齐到 +Z → +1；
        - 克隆合成参与者：沿用 donor 的推断结果；
        - 真实物体：用**不可变的原始位姿**判断——原始数据里物体是真实朝向（车朝行驶方向），
          因此比较“原始局部 +Z”与“原始运动方向”的平均夹角符号即可确定车头轴。
        """
        tid = int(track_id)
        if self.is_synthetic(tid):
            s = self.synthetic_tracks.get(tid) or {}
            if s.get("ply_path"):
                return 1.0
            donor = s.get("donor_track_id")
            if donor is not None and not self.is_synthetic(donor):
                return self._forward_sign_from_raw(int(donor))
            return 1.0
        return self._forward_sign_from_raw(tid)

    def _forward_sign_from_raw(self, track_id):
        tid = int(track_id)
        cached = self._forward_sign_cache.get(tid)
        if cached is not None:
            return cached
        raw = self.track_raw_poses.get(tid, {})
        frames = sorted(raw.keys())
        # 用“净位移在原始 +Z 轴上的投影”判断车头轴：等价于整段轨迹的总体前进方向，
        # 比逐帧夹角平均稳健（逐帧平均在来回抖动时会互相抵消成噪声）。
        acc = 0.0
        prev_c = prev_z = None
        for f in frames:
            p = np.asarray(raw[f], dtype=np.float32)
            c = p[:3, 3]
            if prev_c is not None:
                d = c - prev_c
                d[1] = 0.0
                n = float(np.linalg.norm(d))
                z = prev_z.copy()
                z[1] = 0.0
                zn = float(np.linalg.norm(z))
                if n > 0.05 and zn > 1e-6:
                    acc += float(np.dot(d, z / zn))
            prev_c = c
            prev_z = p[:3, 2].copy()
        sign = 1.0 if acc > 0 else -1.0
        self._forward_sign_cache[tid] = sign
        return sign

    def _motion_yaw_map(self, track_id):
        """该 track 逐帧的**平滑**运动朝向 yaw（弧度，绕世界 Y），带缓存。

        两步：
        1. 方向估计：以每个轨迹点为中，取前后各 `window` 个**有效轨迹点**之间的位移
           得到瞬时运动方向（位移 < 5cm 视为抖动，不参与）。
        2. 平滑：对朝向角做**限速转向**（slew-rate limit）——每帧最多转动
           `max_step` 度，`smoothing` 越大转得越慢；没有方向的帧保持上一朝向。
           这样即使物体掉头，朝向也是逐帧小步转过，不会突然翻转。
        """
        tid = int(track_id)
        cached = self._heading_cache.get(tid)
        if cached is not None:
            return cached

        frames = self.get_track_frames(tid)
        centers = []
        for f in frames:
            p = self._get_track_pose_base(tid, f)
            centers.append(None if p is None
                           else np.asarray(p, dtype=np.float32)[:3, 3].copy())
        n = len(frames)
        # 近乎静止的物体（杆/静止车/锥桶）不做自动朝向，避免噪声导致乱转
        valid = [c for c in centers if c is not None]
        path_len = float(sum(np.linalg.norm(valid[i + 1] - valid[i])
                             for i in range(len(valid) - 1))) if len(valid) > 1 else 0.0
        if path_len < 0.5:
            self._heading_cache[tid] = {}
            return {}

        wf = max(1, int(self.auto_heading_window))   # 方向估计的 ±帧窗口（10fps 时 6 ≈ 0.6s）
        vs = [k for k in range(n) if centers[k] is not None]
        dirs = [None] * n
        for k in vs:
            fi = frames[k]
            sel = [q for q in vs if abs(frames[q] - fi) <= wf]
            if len(sel) < 3:
                continue
            pts = np.array([centers[q][[0, 2]] for q in sel], dtype=np.float64)
            mean = pts.mean(axis=0)
            c0 = pts - mean
            cov = (c0.T @ c0) / len(pts)
            try:
                evals, evecs = np.linalg.eigh(cov)
            except np.linalg.LinAlgError:
                continue
            # 主轴方向 = 物体在该窗口内的整体运动方向；主轴散布太小说明几乎没动/噪声
            if math.sqrt(max(0.0, float(evals[-1]))) < 0.4:
                continue
            d = evecs[:, -1].astype(np.float64)
            net = pts[-1] - pts[0]
            if float(np.dot(d, net)) < 0.0:
                d = -d
            dirs[k] = d

        sm = float(min(1.0, max(0.0, self.auto_heading_smoothing)))
        # 第一步：方向**向量平均**（±sf 帧）——去掉逐帧抖动，且天然处理角度环绕
        sf = int(round(sm * 6))
        tgt_yaw = [None] * n
        for i in range(n):
            fi = frames[i]
            acc = np.zeros(2, dtype=np.float64)
            for k in vs:
                if dirs[k] is not None and abs(frames[k] - fi) <= sf:
                    acc += dirs[k]
            if float(np.linalg.norm(acc)) > 1e-6:
                tgt_yaw[i] = math.atan2(float(acc[0]), float(acc[1]))

        # 第二步：**限速转向**（slew-rate limit）——即使目标方向突然反向（碰撞反弹/掉头），
        # 每帧也最多转 max_step 度，保证"平滑、不突然"；没有方向的帧保持上一朝向。
        # 另外：目标方向与当前朝向相差 >100° 时，必须连续 `sustain` 帧都指向新方向才接受，
        # 避免轨迹噪声导致物体来回掉头。
        max_step = math.pi if sm <= 0.01 else math.radians(4.0 + 46.0 * (1.0 - sm))
        # 物理上限：车不可能瞬间转过很大角度（质量门要求 ≤1.2 rad/s，
        # 否则即使"平滑"也会因为一帧转 20° 而被判轨迹动力学不合理）。
        try:
            rate_cap = float(self.auto_heading_max_yaw_rate) / max(1e-3, float(self.fps))
            max_step = min(max_step, max(rate_cap, 1e-3))
        except Exception:  # noqa: BLE001
            pass
        sustain = max(1, int(self.auto_heading_sustain))
        out = {}
        prev = None
        pend_yaw = None
        pend_cnt = 0
        for i in range(n):
            t = tgt_yaw[i]
            if t is None:
                yaw = prev          # 没有方向（静止/抖动/掉头瞬间）：保持上一朝向
            elif prev is None:
                yaw = t
            else:
                raw_d = (t - prev + math.pi) % (2.0 * math.pi) - math.pi
                if abs(raw_d) > math.radians(100.0) and sustain > 1:
                    if pend_yaw is not None and abs((t - pend_yaw + math.pi) % (2.0 * math.pi) - math.pi) < math.radians(60.0):
                        pend_cnt += 1
                    else:
                        pend_yaw = t
                        pend_cnt = 1
                    if pend_cnt < sustain:
                        t = None        # 还没确认，先保持原朝向
                else:
                    pend_yaw = None
                    pend_cnt = 0
            if t is None:
                yaw = prev
            elif prev is None:
                yaw = t
            else:
                d = (t - prev + math.pi) % (2.0 * math.pi) - math.pi
                step = max(-max_step, min(max_step, d))
                yaw = (prev + step + math.pi) % (2.0 * math.pi) - math.pi
            if yaw is not None:
                prev = yaw
                out[int(frames[i])] = float(yaw)

        self._heading_cache[tid] = out
        return out

    def get_motion_yaw(self, track_id, frame_idx):
        """取某帧的运动朝向 yaw；该帧不在轨迹上时取最近帧。"""
        m = self._motion_yaw_map(track_id)
        if not m:
            return None
        f = int(frame_idx)
        if f in m:
            return m[f]
        return m[min(m.keys(), key=lambda k: abs(k - f))]

    def _apply_auto_heading(self, track_id, frame_idx, pose):
        """把位姿的偏航角改成“车头朝向运动方向”，保留俯仰/滚转。"""
        if pose is None or not self._is_auto_heading_track(track_id):
            return pose
        yaw = self.get_motion_yaw(track_id, frame_idx)
        if yaw is None:
            return pose
        if self._forward_sign(track_id) < 0:
            yaw = yaw + math.pi          # 车头是 -Z：把 +Z 指向运动反方向
        out = np.asarray(pose, dtype=np.float32).copy()
        R = out[:3, :3]
        cur = math.atan2(float(R[0, 2]), float(R[2, 2]))
        d = yaw - cur
        c, s = math.cos(d), math.sin(d)
        Ry = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float32)
        out[:3, :3] = Ry @ R
        return out

    def list_motion_headings(self, frame_idx):
        """列出所有动态物体在该帧的当前/目标朝向（用于前端展示）。"""
        out = []
        tids = set(self.track_meta.keys()) | set(self.synthetic_tracks.keys())
        for tid in sorted(tids):
            frames = self.get_track_frames(tid)
            if not frames:
                continue
            f = int(frame_idx)
            if f not in frames:
                f = min(frames, key=lambda k: abs(k - f))
            yaw = self.get_motion_yaw(tid, f)
            out.append({
                "track_id": int(tid),
                "frame_idx": int(f),
                "motion_yaw_deg": None if yaw is None else float(math.degrees(yaw)),
                "forward_sign": self._forward_sign(tid),
                "auto": self._is_auto_heading_track(tid),
            })
        return out

    # ==================== 全局旋转偏移（持久 360° 旋转）====================

    def get_track_rotation(self, track_id):
        """返回某 track 的 3x3 局部旋转偏移（无则单位阵）。"""
        R = self.track_rotations.get(int(track_id))
        return R if R is not None else np.eye(3, dtype=np.float32)

    def set_track_rotation(self, track_id, R):
        """设置某 track 的全局局部旋转偏移（作用于该物体所有帧）。"""
        track_id = int(track_id)
        R = np.asarray(R, dtype=np.float32)
        if np.allclose(R, np.eye(3), atol=1e-6):
            self.track_rotations.pop(track_id, None)
        else:
            self.track_rotations[track_id] = R
        self.track_deleted.discard(track_id)

    def accumulate_track_rotation(self, track_id, delta_R):
        """在当前旋转偏移上叠加一个旋转（delta_R @ R）。"""
        track_id = int(track_id)
        R = self.get_track_rotation(track_id)
        self.set_track_rotation(track_id, np.asarray(delta_R, dtype=np.float32) @ R)

    def _apply_track_rotation(self, track_id, pose):
        R = self.track_rotations.get(int(track_id))
        if R is None or pose is None:
            return pose
        out = np.asarray(pose, dtype=np.float32).copy()
        out[:3, :3] = out[:3, :3] @ R
        return out

    def get_track_pose(self, track_id, frame_idx):
        """获取某 track 在指定帧的最终位姿（基础轨迹 → 车头朝向运动方向 → 全局旋转偏移）。"""
        pose = self._get_track_pose_base(track_id, frame_idx)
        pose = self._apply_auto_heading(track_id, frame_idx, pose)
        return self._apply_track_rotation(track_id, pose)

    def _get_track_pose_base(self, track_id, frame_idx):
        """获取某 track 在指定帧的基础位姿（未叠加全局旋转偏移）。

        语义（逐顶点编辑）：
        - 该帧有编辑关键帧 → 直接用编辑值；
        - 否则 → 用该帧原始位姿（未被编辑的帧保持不动）。
        这样拖动某一帧的轨迹顶点只改变该帧，物体沿"被编辑后的逐帧轨迹"运动。
        """
        track_id = int(track_id)
        frame_idx = int(frame_idx)

        # 合成参与者：直接返回其逐帧位姿（缺帧则在已有帧间插值/夹取）
        if track_id in self.synthetic_tracks:
            poses = self.synthetic_tracks[track_id]["poses"]
            if frame_idx in poses:
                return np.asarray(poses[frame_idx], dtype=np.float32)
            keys = sorted(poses.keys())
            if not keys:
                return None
            if frame_idx <= keys[0]:
                return np.asarray(poses[keys[0]], dtype=np.float32)
            if frame_idx >= keys[-1]:
                return np.asarray(poses[keys[-1]], dtype=np.float32)
            for f0, f1 in zip(keys[:-1], keys[1:]):
                if f0 <= frame_idx <= f1:
                    alpha = (frame_idx - f0) / float(max(1, f1 - f0))
                    return np.asarray(self.renderer._interp_pose(poses[f0], poses[f1], alpha), dtype=np.float32)
            return None

        edits = self.track_edits.get(track_id)
        if edits and frame_idx in edits:
            return np.asarray(edits[frame_idx], dtype=np.float32)

        # 超出数据范围的帧：用延长时外推出来的位姿
        ext = self.track_extended.get(track_id)
        if ext and frame_idx in ext:
            return np.asarray(ext[frame_idx], dtype=np.float32)

        # 原始位姿（未编辑帧保持不变）
        raw = self.track_raw_poses.get(track_id, {}).get(frame_idx)
        if raw is not None:
            return np.asarray(raw, dtype=np.float32)

        # 该帧没有原始位姿 → **资产化**：在相邻"有检测"的帧之间插值，让物体在漏检的帧
        # 里也持续存在（不因某一帧没识别到就整帧消失）。只覆盖"首次检测 ~ 末次检测"
        # 之间的内部空隙；末次检测之后不夹取（物体可能真的离开场景了）。
        raw_frames = sorted(self.track_raw_poses.get(track_id, {}).keys())
        if raw_frames:
            if frame_idx < raw_frames[0] or frame_idx > raw_frames[-1]:
                return None
            if frame_idx in self.track_raw_poses[track_id]:
                return np.asarray(self.track_raw_poses[track_id][frame_idx], dtype=np.float32)
            for f0, f1 in zip(raw_frames[:-1], raw_frames[1:]):
                if f0 < frame_idx < f1:
                    alpha = (frame_idx - f0) / float(max(1, f1 - f0))
                    return np.asarray(self.renderer._interp_pose(
                        self.track_raw_poses[track_id][f0],
                        self.track_raw_poses[track_id][f1], alpha), dtype=np.float32)
            return None

        # 该帧无原始位姿（稀疏关键帧轨迹）：在编辑关键帧之间插值
        if edits:
            frames = sorted(edits.keys())
            if frame_idx <= frames[0]:
                return np.asarray(edits[frames[0]], dtype=np.float32)
            if frame_idx >= frames[-1]:
                return np.asarray(edits[frames[-1]], dtype=np.float32)
            for f0, f1 in zip(frames[:-1], frames[1:]):
                if f0 <= frame_idx <= f1:
                    alpha = (frame_idx - f0) / float(max(1, f1 - f0))
                    return np.asarray(self.renderer._interp_pose(edits[f0], edits[f1], alpha), dtype=np.float32)
        return None

    def get_track_trajectory(self, track_id, use_edits=True):
        """返回 track 的逐帧中心点列表 [{frame_idx, center[x,y,z], edited}]。"""
        track_id = int(track_id)
        result = []
        if track_id in self.synthetic_tracks:
            for frame_idx in self.get_track_frames(track_id):
                pose = self.get_track_pose(track_id, frame_idx)
                if pose is None:
                    continue
                result.append({
                    "frame_idx": int(frame_idx),
                    "center": [float(pose[0, 3]), float(pose[1, 3]), float(pose[2, 3])],
                    "edited": True,
                })
            return result
        for frame_idx in self.get_track_frames(track_id):
            if use_edits:
                pose = self.get_track_pose(track_id, frame_idx)
            else:
                pose = self.track_raw_poses.get(track_id, {}).get(frame_idx)
            if pose is None:
                continue
            edited = bool(self.track_edits.get(track_id, {}).get(frame_idx) is not None)
            result.append({
                "frame_idx": int(frame_idx),
                "center": [float(pose[0, 3]), float(pose[1, 3]), float(pose[2, 3])],
                "edited": edited,
            })
        return result

    # ==================== 编辑 ====================

    def _strip_user_rotation(self, track_id, pose):
        """写入前去掉“用户全局旋转偏移”，读时会重新叠加，避免反复编辑时旋转被累加。"""
        R_user = self.track_rotations.get(int(track_id))
        if R_user is None or pose is None:
            return pose
        try:
            Rinv = np.linalg.inv(np.asarray(R_user, dtype=np.float32))
        except np.linalg.LinAlgError:
            return pose
        out = np.asarray(pose, dtype=np.float32).copy()
        out[:3, :3] = out[:3, :3] @ Rinv
        return out

    def is_track_replaced(self, track_id):
        """该 track 是否是被 SAM3D 替换掉的原始物体（不允许再被编辑/复活）。"""
        return int(track_id) in self.track_replaced

    def set_track_pose(self, track_id, frame_idx, pose_matrix):
        """为某 track 在指定帧设置位姿。

        - **合成物体（SAM3D 替换物体、事故合成参与者）**：直接写入它自己的逐帧位姿
          （它们不使用 `track_edits`，否则编辑会被忽略、看起来"拖拽后自动复原"）。
        - 真实 track：写位姿关键帧。
        两种都会先去掉用户旋转偏移，保持读时语义一致。
        """
        track_id = int(track_id)
        pose = np.asarray(pose_matrix, dtype=np.float32)
        if track_id in self.synthetic_tracks:
            self.synthetic_tracks[track_id]["poses"][int(frame_idx)] = \
                np.asarray(self._strip_user_rotation(track_id, pose), dtype=np.float32)
            self._invalidate_heading_cache(track_id)
            return
        self.track_edits.setdefault(track_id, {})[int(frame_idx)] = \
            np.asarray(self._strip_user_rotation(track_id, pose), dtype=np.float32)
        # 被替换掉的原始物体不允许通过编辑"复活"
        if track_id not in self.track_replaced:
            self.track_deleted.discard(track_id)
        self._invalidate_heading_cache(track_id)

    def set_track_trajectory(self, track_id, keyframes):
        """用关键帧列表 [(frame_idx, pose4x4)] 设置整条轨迹。"""
        track_id = int(track_id)
        kfs = {int(f): np.asarray(self._strip_user_rotation(track_id, np.asarray(p, dtype=np.float32)),
                                  dtype=np.float32) for f, p in keyframes}
        if track_id in self.synthetic_tracks:
            self.synthetic_tracks[track_id]["poses"] = kfs
            self._invalidate_heading_cache(track_id)
            return
        self.track_edits[track_id] = kfs
        if track_id not in self.track_replaced:
            self.track_deleted.discard(track_id)
        self._invalidate_heading_cache(track_id)

    def delete_track(self, track_id):
        track_id = int(track_id)
        self.track_deleted.add(track_id)
        self.track_edits.pop(track_id, None)
        self._invalidate_heading_cache(track_id)

    def clear_track_edits(self, track_id):
        """撤销某 track 的轨迹编辑。

        - 合成物体：把轨迹还原到创建时的 `base_poses`（SAM3D 替换物体用）；
        - 真实 track：删除位姿编辑，回到原始轨迹。
        """
        track_id = int(track_id)
        if track_id in self.synthetic_tracks:
            self.reset_synthetic_to_base(track_id)
            return
        self.track_edits.pop(track_id, None)
        if track_id not in self.track_replaced:
            self.track_deleted.discard(track_id)
        self._invalidate_heading_cache(track_id)

    def drag_point_adaptive(self, track_id, frame_idx, new_center,
                            influence=6, falloff="smooth"):
        """智能轨迹编辑：拖动某一帧的点，相邻帧按影响范围自适应平滑跟随。

        Args:
            track_id: 目标 track
            frame_idx: 被拖动的帧
            new_center: 新的世界中心 [x,y,z]
            influence: 影响半径（前后多少帧跟随）
            falloff: 衰减方式 'smooth'(平滑) | 'linear'(线性) | 'gaussian'(高斯)

        语义：被拖动帧位移最大，越远的帧位移越小，整条路径平滑过渡，
        避免用户逐个调整每个节点。
        """
        track_id = int(track_id)
        frame_idx = int(frame_idx)
        frames = self.get_track_frames(track_id)
        cur_pose = self.get_track_pose(track_id, frame_idx)
        if cur_pose is None:
            return []
        if frame_idx not in frames:
            # 允许在"插值帧"上新增轨迹点：合成物体（SAM3D 替换物体）只在目标轨迹的
            # 稀疏帧上有位姿，但渲染时会插值出现在所有帧，所以这里也要能编辑。
            self.set_track_pose(track_id, frame_idx, cur_pose)
            frames = sorted(set(frames) | {frame_idx})
        cur_center = np.asarray(cur_pose, dtype=np.float32)[:3, 3]
        delta = np.asarray(new_center, dtype=np.float32) - cur_center

        center_pos = frames.index(frame_idx)
        edited_frames = []

        for offset in range(-influence, influence + 1):
            idx = center_pos + offset
            if idx < 0 or idx >= len(frames):
                continue
            f = frames[idx]
            # 影响权重
            dist = abs(offset) / float(max(1, influence))
            if dist > 1.0:
                w = 0.0
            elif falloff == "linear":
                w = 1.0 - dist
            elif falloff == "gaussian":
                w = float(np.exp(-(dist * 2.0) ** 2))
            else:  # smooth (smoothstep 反向)
                w = 1.0 - (dist * dist * (3 - 2 * dist))
            if w <= 1e-4:
                continue
            base_pose = self.get_track_pose(track_id, f)
            if base_pose is None:
                continue
            base_pose = np.asarray(base_pose, dtype=np.float32).copy()
            base_pose[0, 3] += delta[0] * w
            base_pose[1, 3] += delta[1] * w
            base_pose[2, 3] += delta[2] * w
            self.set_track_pose(track_id, f, base_pose)
            edited_frames.append(int(f))

        return edited_frames

    def smooth_track(self, track_id, smoothness=0.5, keep_endpoints=True):
        """对整条 track 轨迹做平滑（消除手动编辑产生的抖动）。"""
        from dggt.scene_edit.collision_physics import smooth_trajectory_adaptive
        track_id = int(track_id)
        frames = self.get_track_frames(track_id)
        if len(frames) < 3:
            return []
        poses = [self.get_track_pose(track_id, f) for f in frames]
        centers = [np.asarray(p, dtype=np.float32)[:3, 3].copy() for p in poses if p is not None]
        if len(centers) < 3:
            return []
        fixed = [0, len(centers) - 1] if keep_endpoints else []
        smoothed = smooth_trajectory_adaptive(centers, fixed, smoothness)
        edited = []
        for f, p, c in zip(frames, poses, smoothed):
            if p is None:
                continue
            new_pose = np.asarray(p, dtype=np.float32).copy()
            new_pose[:3, 3] = c
            self.set_track_pose(track_id, f, new_pose)
            edited.append(int(f))
        return edited

    def is_track_edited(self, track_id):
        track_id = int(track_id)
        return track_id in self.track_edits or track_id in self.track_deleted

    # ==================== 渲染 override ====================

    def build_object_overrides(self, frame_idx):
        """构建该帧的 raw object_id -> 变换（或 None 表示删除）的 override 字典。

        返回的位姿为 torch.Tensor（在 renderer.device 上），以匹配引擎的
        _transform_gaussians（其内部做 R @ means.T 等张量运算）。
        """
        frame_idx = int(frame_idx)
        device = self.renderer.device
        overrides = {}

        # 删除：把该帧属于已删除 track 的 raw object 标记为 None
        for tid in set(self.track_deleted) | set(self.track_replaced):
            raw_id = self.get_raw_object_id(tid, frame_idx)
            if raw_id is not None:
                overrides[raw_id] = None

        # 延长出来的帧：位姿直接交给渲染器（数据帧之外没有逐帧外观，靠外观缓存复用）
        for tid, ext in self.track_extended.items():
            if tid in self.track_deleted or tid in self.track_replaced or tid in self.synthetic_tracks:
                continue
            if frame_idx not in ext:
                continue
            raw_id = self.get_raw_object_id(tid, frame_idx)
            if raw_id is None:
                continue
            overrides[int(raw_id)] = torch.tensor(
                np.asarray(ext[frame_idx], dtype=np.float32), device=device, dtype=torch.float32
            )

        # 位姿编辑 / 全局旋转偏移
        for tid in set(self.track_edits) | set(self.track_rotations):
            if tid in self.track_deleted or tid in self.track_replaced or tid in self.synthetic_tracks:
                continue
            raw_id = self.get_raw_object_id(tid, frame_idx)
            if raw_id is None:
                continue
            pose = self.get_track_pose(tid, frame_idx)
            if pose is not None:
                overrides[raw_id] = torch.tensor(
                    np.asarray(pose, dtype=np.float32), device=device, dtype=torch.float32
                )

        return overrides

    def prewarm_render_appearance(self, max_frames: int = 40):
        """渲染前预热 donor 外观缓存，避免合成参与者"整段消失"。

        渲染器按帧顺序加载 PLY 并顺手缓存每个物体的外观（`_obj_appearance`）。合成参与者
        是"克隆 donor 高斯"渲染的，若它的 donor 在该帧还没出现过（或该帧的逐帧 PLY 里没有
        donor），就会去查外观缓存；**缓存还没有 → 直接把该物体整帧跳过**（画面里就是
        "参与者消失"）。事故生成经常让参与者出现在窗口最开头，而 donor 可能到窗口中部才被
        检测到，所以这里先按"每个 donor 首次出现的帧"把缓存预热一遍。
        """
        needed = set()
        for s in self.synthetic_tracks.values():
            if s.get("ply_path"):
                continue                       # SAM3D 物体自带高斯，不需要 donor 缓存
            donor = s.get("donor_track_id")
            if donor is None:
                continue
            fm = self.track_to_frames.get(int(donor)) or {}
            if fm:
                needed.add(int(min(fm.keys())))
        # 真实 track 被编辑（事故生成）后也可能需要"补帧"渲染，用它们的首个检测帧预热
        for tid in set(self.track_edits) | set(self.track_rotations):
            if int(tid) in self.synthetic_tracks:
                continue
            fm = self.track_to_frames.get(int(tid)) or {}
            if fm:
                needed.add(int(min(fm.keys())))
        ok = 0
        for f in sorted(needed)[:max(1, int(max_frames))]:
            try:
                self.renderer._appearance_objects(int(f), 0)
                ok += 1
            except Exception:  # noqa: BLE001
                continue
        return ok

    def build_extra_objects(self, frame_idx, include_ego=True):
        """构建该帧需要额外渲染的合成参与者列表（克隆 donor 高斯）。

        动态物体高斯在 PLY 中以物体局部坐标存储，正常渲染时直接乘该帧
        pose_world。因此合成参与者也直接使用目标 pose_world，并按 donor track
        在当前帧的 raw object_id 克隆外观，避免 raw object_id 跨帧不连续导致闪烁。
        """
        frame_idx = int(frame_idx)
        device = self.renderer.device
        extras = []
        for sid, s in self.synthetic_tracks.items():
            if s.get("ego") and not include_ego:
                continue
            if s.get("ego") and s.get("visible") is False:
                continue
            target = self.get_track_pose(sid, frame_idx)
            if target is None:
                continue
            transform = np.asarray(target, dtype=np.float32)
            # SAM3D 重建物体：自带 .ply 高斯，直接渲染
            if s.get("ply_path"):
                extras.append({
                    "ply_path": s["ply_path"],
                    "transform": torch.tensor(transform, device=device, dtype=torch.float32),
                    "scale": float(s.get("scale", 1.0)),
                    "scale_vec": s.get("scale_vec"),
                    "center": s.get("center"),
                    "model_corr": s.get("model_corr"),
                    "dims": s.get("dimensions"),
                    "up_sign": self.world_up_sign(),
                    # 接触阴影要贴在真实路面上（包围盒底面在高度方向不可靠）
                    "ground_y": self.ground_y_at(transform[0, 3], transform[2, 3]),
                    # 接触阴影可逐物体开关（SAM3D 模型本身没有烘焙阴影，默认补一圈）
                    "shadow": bool(s.get("shadow", True)),
                    "synth_track_id": int(sid),
                })
                continue
            # 普通合成参与者：克隆 donor 高斯
            donor_track_id = s.get("donor_track_id")
            donor_raw_id = None
            if donor_track_id is not None:
                donor_raw_id = self.get_raw_object_id(donor_track_id, frame_idx)
            if donor_raw_id is None:
                donor_raw_id = s.get("donor_object_id")
            extras.append({
                "donor_object_id": int(donor_raw_id),
                "transform": torch.tensor(transform, device=device, dtype=torch.float32),
                "synth_track_id": int(sid),
            })

        # ---- 资产化：补回"这一帧漏检、但 track 仍应存在"的真实物体 ----
        # 单目检测偶发漏掉某一帧，导致那帧的逐帧元数据里没有该物体。这里用
        # "该物体最后一次出现的外观 + 相邻检测帧插值出的位姿"把它补回来，避免整帧消失。
        try:
            present = {int(o["object_id"]) for o in self.renderer.load_real_frame_objects(frame_idx)}
        except Exception:  # noqa: BLE001
            present = set()
        for tid, frames_map in self.track_to_frames.items():
            if tid in self.track_deleted or tid in self.track_replaced or tid in self.synthetic_tracks:
                continue
            if not frames_map:
                continue
            fmin, fmax = min(frames_map.keys()), max(frames_map.keys())
            # 编辑帧即使落在该 track 的检测跨度之外也要补：事故生成会给参与者写满整个窗口，
            # 只按 fmin..fmax 判断的话，跨度不足的参与者会在窗口两端整段消失。
            edited_at = int(frame_idx) in (self.track_edits.get(int(tid)) or {})
            if not (fmin <= frame_idx <= fmax or edited_at):
                continue
            rid_this = frames_map.get(frame_idx)
            if rid_this is not None and int(rid_this) in present:
                continue                     # 这一帧本来就有，不用补
            pose = self.get_track_pose(int(tid), frame_idx)
            if pose is None:
                continue
            # donor 用"这一帧之前最近一次检测"的 raw id：渲染是顺序进行的，它的外观一定
            # 已经在 `_obj_appearance` 缓存里。别用 `_track_render_raw`（那是**末次**检测的
            # id，可能还没渲染到，缓存里没有 → 回退失败）。
            donor_raw = None
            for f2 in sorted(frames_map.keys(), reverse=True):
                if f2 <= frame_idx:
                    donor_raw = frames_map[f2]
                    break
            if donor_raw is None:
                donor_raw = frames_map[min(frames_map.keys())]
            if donor_raw is None:
                continue
            extras.append({
                "donor_object_id": int(donor_raw),
                "transform": torch.tensor(np.asarray(pose, dtype=np.float32),
                                          device=device, dtype=torch.float32),
                "synth_track_id": None,      # 不是合成物体，别被 ego/synth 过滤误伤
                "persistent": True,
            })
        return extras

    # ==================== 帧级物体列表（带 track_id） ====================

    def get_frame_objects(self, frame_idx):
        """返回该帧动态物体列表，附带 track_id、当前位姿（含编辑）、是否编辑/删除。

        包含原始 track 与合成参与者（synthetic）。
        """
        frame_idx = int(frame_idx)
        result = []
        seen_tracks = set()
        for obj in self.renderer.load_real_frame_objects(frame_idx):
            raw_id = int(obj["object_id"])
            tid = self.get_track_id(frame_idx, raw_id)
            if tid is None:
                continue
            if tid in self.track_deleted or tid in self.track_replaced:
                continue  # 已删除/已被替换的不返回
            pose = self.get_track_pose(tid, frame_idx)
            if pose is None:
                pose = np.asarray(obj["pose_world"], dtype=np.float32)
            pose_list = pose.astype(float).tolist()
            seen_tracks.add(int(tid))
            result.append({
                "track_id": tid,
                "raw_object_id": raw_id,
                "type": obj.get("type", "未知"),
                "dimensions": obj.get("dimensions", []),
                "pose_world": pose_list,
                "center": [pose_list[0][3], pose_list[1][3], pose_list[2][3]],
                "edited": self.is_track_edited(tid),
                "synthetic": False,
            })

        # 延长出来的帧：磁盘上没有逐帧元数据，但轨迹被外推过，这里照样把它们列出来
        # （否则超出原场景帧数后前端选不中物体、也画不出包围盒）
        for tid, ext in self.track_extended.items():
            if int(tid) in seen_tracks:
                continue
            if frame_idx not in ext:
                continue
            if tid in self.track_deleted or tid in self.track_replaced:
                continue
            pose = self.get_track_pose(int(tid), frame_idx)
            if pose is None:
                continue
            pose_list = np.asarray(pose, dtype=np.float32).astype(float).tolist()
            meta = self.track_meta.get(int(tid), {}) or {}
            result.append({
                "track_id": int(tid),
                "raw_object_id": self.get_raw_object_id(int(tid), frame_idx),
                "type": meta.get("type", "动态物体"),
                "dimensions": meta.get("dimensions", []) or self.get_track_dimensions(int(tid)),
                "pose_world": pose_list,
                "center": [pose_list[0][3], pose_list[1][3], pose_list[2][3]],
                "edited": self.is_track_edited(int(tid)),
                "synthetic": False,
                "extended": True,      # 该帧是"延长出来"的，位姿由轨迹外推
            })

        # 资产化：**内部漏检帧** —— track 的检测区间覆盖这一帧、但这一帧的逐帧元数据里
        # 没有它（单目检测偶发漏掉），用相邻检测帧插值位姿把它补回来，别让物体"整帧消失"。
        for tid, frames_map in self.track_to_frames.items():
            if int(tid) in seen_tracks:
                continue
            if tid in self.track_deleted or tid in self.track_replaced:
                continue
            if not frames_map:
                continue
            fmin, fmax = min(frames_map.keys()), max(frames_map.keys())
            if not (fmin <= frame_idx <= fmax):
                continue
            pose = self.get_track_pose(int(tid), frame_idx)
            if pose is None:
                continue
            pose_list = np.asarray(pose, dtype=np.float32).astype(float).tolist()
            meta = self.track_meta.get(int(tid), {}) or {}
            result.append({
                "track_id": int(tid),
                "raw_object_id": self.get_raw_object_id(int(tid), frame_idx),
                "type": meta.get("type", "动态物体"),
                "dimensions": meta.get("dimensions", []) or self.get_track_dimensions(int(tid)),
                "pose_world": pose_list,
                "center": [pose_list[0][3], pose_list[1][3], pose_list[2][3]],
                "edited": self.is_track_edited(int(tid)),
                "synthetic": False,
                "persisted": True,      # 该帧是"补出来的"（检测漏帧，位姿由相邻帧插值）
            })

        # 合成参与者 + 主车（EGO 实体）
        for sid, s in self.synthetic_tracks.items():
            if s.get("ego") and s.get("visible") is False:
                continue
            pose = self.get_track_pose(sid, frame_idx)
            if pose is None:
                continue
            pose_list = np.asarray(pose, dtype=np.float32).astype(float).tolist()
            result.append({
                "track_id": int(sid),
                "raw_object_id": None,
                "type": s.get("type", "合成参与者"),
                "dimensions": s.get("dimensions", []),
                "pose_world": pose_list,
                "center": [pose_list[0][3], pose_list[1][3], pose_list[2][3]],
                "edited": True,
                "synthetic": True,
                "ego": bool(s.get("ego")),
                "viewer": bool(s.get("ego") and self.ego_source_track is None),
            })
        # 真实 track 中"当前被当作主车视角"的那一个
        if self.ego_source_track is not None:
            for o in result:
                if int(o["track_id"]) == int(self.ego_source_track):
                    o["viewer"] = True
        return result

    def list_tracks(self):
        """列出所有 track 的概要。"""
        out = []
        for tid, meta in sorted(self.track_meta.items()):
            out.append({
                "track_id": tid,
                "type": meta.get("type", "未知"),
                "dimensions": meta.get("dimensions", []),
                "first_frame": meta.get("first_frame"),
                "num_frames": len(self.track_to_frames.get(tid, {})),
                "edited": self.is_track_edited(tid),
            })
        return out
