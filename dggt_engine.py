import os
import json
import argparse
import math

import torch
import numpy as np
import cv2
import imageio.v2 as imageio
from plyfile import PlyData
from gsplat.rendering import rasterization
from scipy.spatial.transform import Rotation as R, Slerp

from dggt.scene_edit import SceneEditExecutor, load_scene_edit_spec, load_scene_edit_specs


class TrajectoryController:
    def __init__(self, device="cuda"):
        self.device = device
        self.plans = {}

    def add_keyframe(self, object_id, frame_idx, pose_world):
        self.plans.setdefault(object_id, {})[int(frame_idx)] = np.asarray(pose_world, dtype=np.float32)

    def set_trajectory(self, object_id, keyframes, **kwargs):
        self.plans[object_id] = {int(f): np.asarray(p, dtype=np.float32) for f, p in keyframes}

    def get_object_ids(self):
        return list(self.plans.keys())

    def _interp_pose(self, pose_a, pose_b, alpha):
        pose_a = np.asarray(pose_a, dtype=np.float32)
        pose_b = np.asarray(pose_b, dtype=np.float32)
        out = np.eye(4, dtype=np.float32)
        out[:3, 3] = (1 - alpha) * pose_a[:3, 3] + alpha * pose_b[:3, 3]
        rot = Slerp([0.0, 1.0], R.from_matrix(np.stack([pose_a[:3, :3], pose_b[:3, :3]], axis=0)))([alpha])[0]
        out[:3, :3] = rot.as_matrix()
        return out

    def evaluate_pose(self, object_id, frame_idx):
        plan = self.plans.get(object_id)
        if not plan:
            return None
        frames = sorted(plan.keys())
        if frame_idx <= frames[0]:
            return plan[frames[0]]
        if frame_idx >= frames[-1]:
            return plan[frames[-1]]
        for f0, f1 in zip(frames[:-1], frames[1:]):
            if f0 <= frame_idx <= f1:
                alpha = (frame_idx - f0) / float(max(1, f1 - f0))
                return self._interp_pose(plan[f0], plan[f1], alpha)
        return None

    def to_spec(self):
        return {
            "device": self.device,
            "trajectories": [
                {
                    "object_id": oid,
                    "keyframes": [{"frame_idx": int(f), "pose_world": pose.tolist()} for f, pose in sorted(plan.items())],
                }
                for oid, plan in self.plans.items()
            ],
        }

    @classmethod
    def from_spec(cls, spec, device="cuda"):
        controller = cls(device=device)
        for item in spec.get("trajectories", []):
            controller.set_trajectory(
                item["object_id"],
                [(kf["frame_idx"], kf["pose_world"]) for kf in item.get("keyframes", [])],
            )
        return controller

    @classmethod
    def load_spec(cls, input_path, device="cuda"):
        with open(input_path, "r") as f:
            return cls.from_spec(json.load(f), device=device)

    def save_spec(self, output_path, metadata=None):
        spec = self.to_spec()
        spec["metadata"] = dict(metadata or {})
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(spec, f, indent=2)
        return spec


class CornerCaseGenerator:
    def __init__(self, controller):
        self.controller = controller

    def generate(self, scenario_type, selected_objects, base_poses, start_idx, num_frames, metadata=None):
        metadata = dict(metadata or {})
        if not selected_objects:
            return
        if scenario_type == "hard-brake":
            oid = selected_objects[0]
            p = np.asarray(base_poses[oid], dtype=np.float32)
            self.controller.set_trajectory(oid, [(start_idx, p), (start_idx + num_frames // 2, p.copy()), (start_idx + num_frames - 1, p.copy())])
        elif scenario_type == "rear-end-precrash" and len(selected_objects) >= 2:
            lead, follow = selected_objects[:2]
            p_lead = np.asarray(base_poses[lead], dtype=np.float32)
            p_follow = np.asarray(base_poses[follow], dtype=np.float32)
            p_follow2 = p_follow.copy(); p_follow2[0, 3] += 8.0
            self.controller.set_trajectory(lead, [(start_idx, p_lead), (start_idx + num_frames - 1, p_lead)])
            self.controller.set_trajectory(follow, [(start_idx, p_follow2), (start_idx + num_frames - 1, p_follow)])
        else:
            oid = selected_objects[0]
            p = np.asarray(base_poses[oid], dtype=np.float32)
            p2 = p.copy(); p2[0, 3] += 5.0
            self.controller.set_trajectory(oid, [(start_idx, p), (start_idx + num_frames - 1, p2)])


# 多视角合成时，用于给不同 view 的原始 object_id 加偏移，避免跨 view 冲突：
#   composite_object_id = view_idx * VIEW_ID_STRIDE + raw_object_id
VIEW_ID_STRIDE = 1000000


def _look_at_world2cam(eye, center, up):
    """世界->相机 4x4 视图矩阵（相机位于 eye、看向 center）。

    gsplat 相机朝 +z（正深度 = 前方），故 forward 映射到 +z。
    """
    f = torch.nn.functional.normalize(center - eye, dim=0)
    s = torch.nn.functional.normalize(torch.cross(f, up), dim=0)
    u = torch.cross(s, f)
    R = torch.stack([s, u, f], dim=0)  # 3x3, world -> cam（f 映射到 +z）
    t = torch.stack([-torch.dot(s, eye), -torch.dot(u, eye), -torch.dot(f, eye)])
    viewmat = torch.eye(4, device=eye.device, dtype=eye.dtype)
    viewmat[:3, :3] = R
    viewmat[:3, 3] = t
    return viewmat


class DGGTRenderer:
    def __init__(self, scene_path, device="cuda", load_sky=True, verbose=True, static_only=False):
        self.scene_path = scene_path
        self.device = device
        self.verbose = verbose
        self.static_only = static_only
        self.static_ply = os.path.join(scene_path, "gaussians", "static_scene.ply")
        self.sky_ply = os.path.join(scene_path, "gaussians", "sky_scene.ply")
        self.dynamic_dir = os.path.join(scene_path, "gaussians")
        self.meta_dir = os.path.join(scene_path, "dynamic_objects")
        self.ego_dir = os.path.join(scene_path, "ego_pose")

        # ---- 多视角（multi-view）检测 ----
        # 某些 4DGS 数据由多个相机视角合成：磁盘上的 frame_XXXX_* 文件是
        # “逐视角展开”的（flat），排列方式为 flat = real_frame * num_views + view。
        # 所有视角共享同一世界坐标系（单个 static_scene.ply），因此可以把同一真实帧
        # 下所有视角的动态物体合并到一帧中渲染/播放。
        self.num_flat_frames = self._count_flat_frames()
        self.num_views = self._detect_num_views()
        if self.num_views < 1:
            self.num_views = 1
        self.num_real_frames = max(1, self.num_flat_frames // self.num_views) if self.num_flat_frames else 0
        if self.verbose and self.num_views > 1:
            print(f"[DGGT] 检测到多视角数据：num_views={self.num_views}, "
                  f"real_frames={self.num_real_frames}（flat_frames={self.num_flat_frames}）")

        self.static_gs = self._load_ply(self.static_ply)
        self.sky_gs = self._load_ply(self.sky_ply) if load_sky and os.path.exists(self.sky_ply) else None
        self.cam_overrides = {}
        self.obj_overrides = {}
        self.trajectory_controller = TrajectoryController(device=self.device)
        self.corner_case_generator = CornerCaseGenerator(self.trajectory_controller)

        # ---- 可扩展时间轴 ----
        # 磁盘上真实存在的帧数（相机/动态物体 PLY/物体元数据）是"数据帧"；而**可渲染帧**
        # 可以按需延长：静态场景是单份高斯（与帧无关），动态物体可以复用"最后出现的外观"
        # 并把位姿交给轨迹（object_pose_provider），相机则按自车轨迹外推。
        # 于是"原场景多少帧"不再限制后续编辑/生成。
        self.num_data_frames = self.num_real_frames if self.num_real_frames else self.num_flat_frames
        self.total_frames = int(self.num_data_frames)
        # 超出数据帧时，用来合成相机位姿的回调：(frame) -> 4x4 c2w（numpy 或 tensor）
        self.camera_provider = None
        # 超出数据帧时，用来取物体位姿的回调：(composite_id, frame) -> 4x4 或 None
        self.object_pose_provider = None
        # (view, raw_object_id) -> 该物体"最后出现"的高斯切片与位姿（跨帧复用外观）
        self._obj_appearance = {}
        self._ego_template = None

        # ---- SAM 3D 导入物体（自带 .ply 高斯，独立于原始/合成 track） ----
        self.sam3d_objects = {}       # object_id -> {"ply_path", "pose"(4x4 np), "scale", "meta"}
        self._sam3d_ply_cache = {}    # ply_path -> gs dict（缓存，避免重复读盘）
        self._next_sam3d_id = 200000  # SAM3D 物体 id 从 20 万起，避开真实/合成 track id

    def _load_ply(self, path):
        plydata = PlyData.read(path)
        v = plydata["vertex"]
        data = {
            "means": torch.stack([torch.tensor(v["x"]), torch.tensor(v["y"]), torch.tensor(v["z"])], dim=-1).to(self.device),
            "scales": torch.stack([torch.tensor(v["scale_0"]), torch.tensor(v["scale_1"]), torch.tensor(v["scale_2"])], dim=-1).to(self.device),
            "quats": torch.stack([torch.tensor(v["rot_0"]), torch.tensor(v["rot_1"]), torch.tensor(v["rot_2"]), torch.tensor(v["rot_3"])], dim=-1).to(self.device),
            "opacities": torch.tensor(v["opacity"]).to(self.device),
            "colors": torch.stack([torch.tensor(v["f_dc_0"]), torch.tensor(v["f_dc_1"]), torch.tensor(v["f_dc_2"])], dim=-1).to(self.device),
        }
        if "object_id" in v:
            data["object_ids"] = torch.tensor(v["object_id"]).to(self.device)
        return data

    # ==================== SAM 3D 导入物体 ====================

    def _load_sam3d_ply(self, ply_path):
        """加载 SAM 3D 重建的 .ply（带缓存），并把 SAM 3D 的"原始值"转成 DGGT 的"激活值"。

        SAM 3D 的 Gaussian.save_ply 存的是未激活值：opacity=logit、scale=log、f_dc=未激活
        SH；而 DGGT _load_ply + gsplat(sh_degree=None) 期望的是激活值（opacity∈[0,1]、
        scale=正值、color∈[0,1]）。故这里做 sigmoid / exp 转换，否则会渲染成黑块。
        """
        ply_path = str(ply_path)
        if ply_path in self._sam3d_ply_cache:
            return self._sam3d_ply_cache[ply_path]
        if not os.path.exists(ply_path):
            return None
        gs = self._load_ply(ply_path)
        # 一些 SAM3D 导出的 ply 里存在 inf/nan（例如 opacity）或非单位四元数，
        # 单个坏高斯会让整幅渲染变成 NaN/纯黑，这里统一清洗一遍。
        for key in ("means", "scales", "quats", "opacities", "colors"):
            if key in gs:
                gs[key] = torch.nan_to_num(gs[key], nan=0.0, posinf=0.0, neginf=0.0)
        gs["scales"] = torch.exp(torch.clamp(gs["scales"], min=-30.0, max=2.0))
        gs["opacities"] = torch.sigmoid(torch.clamp(gs["opacities"], min=-30.0, max=30.0))
        gs["colors"] = torch.sigmoid(torch.clamp(gs["colors"], min=-30.0, max=30.0))
        qn = torch.linalg.norm(gs["quats"], dim=-1, keepdim=True).clamp(min=1e-6)
        gs["quats"] = gs["quats"] / qn
        self._sam3d_ply_cache[ply_path] = gs
        return gs

    def add_sam3d_object(self, ply_path, pose_world, scale=1.0, object_id=None, meta=None, center=None):
        """导入一个 SAM 3D 重建的 .ply 作为场景内可渲染物体。

        Args:
            ply_path:   重建出的 3D Gaussian Splat (.ply) 绝对路径
            pose_world: 4x4 世界变换矩阵（旋转 + 平移，不含缩放）
            scale:      均匀缩放系数（SAM 3D 重建尺度与场景世界尺度可能不同）
            object_id:  可选，自定义物体 id；默认自动分配（200000 起）
            meta:       附加元信息（名称、来源等）
            center:     可选 [x,y,z]，把重建物体质心平移归零后再缩放/摆位，
                        用于"替换已有物体"时让两物体质心对齐。

        Returns:
            object_id: int
        """
        pose_world = np.asarray(pose_world, dtype=np.float32)
        if pose_world.shape != (4, 4):
            raise ValueError("pose_world 必须是 4x4 矩阵")
        if object_id is None:
            object_id = self._next_sam3d_id
            self._next_sam3d_id += 1
        object_id = int(object_id)
        if center is not None:
            center = np.asarray(center, dtype=np.float32).reshape(3)
        self.sam3d_objects[object_id] = {
            "ply_path": str(ply_path),
            "pose": pose_world,
            "scale": float(scale),
            "center": center,
            "meta": meta or {},
        }
        return object_id

    def remove_sam3d_object(self, object_id):
        """移除已导入的 SAM 3D 物体。"""
        return self.sam3d_objects.pop(int(object_id), None) is not None

    def list_sam3d_objects(self):
        """列出当前场景内所有 SAM 3D 导入物体。"""
        out = []
        for obj_id, spec in self.sam3d_objects.items():
            pose = spec["pose"]
            out.append({
                "object_id": int(obj_id),
                "ply_path": spec["ply_path"],
                "scale": spec.get("scale", 1.0),
                "pose_world": pose.tolist(),
                "center": [float(pose[0, 3]), float(pose[1, 3]), float(pose[2, 3])],
                "meta": spec.get("meta", {}),
            })
        return out

    def render_ply_preview(self, ply_path, num_frames=36, size=512):
        """渲染某个 .ply 的旋转预览（turntable），返回 list[uint8 HxWx3 RGB]。

        单独渲染该物体的高斯（不带场景），相机绕物体中心水平旋转 num_frames 帧。
        """
        gs = self._load_sam3d_ply(ply_path)
        if gs is None:
            raise ValueError("ply 加载失败")

        means = gs["means"]
        quats = gs["quats"]
        scales = gs["scales"]
        opac = gs["opacities"]
        colors = gs["colors"]

        center = means.mean(dim=0)
        radius = torch.norm(means - center, dim=1).max().clamp(min=0.5).item()
        dist = float(radius) * 3.0
        fov = math.radians(40.0)
        focal = (size / 2.0) / math.tan(fov / 2.0)
        K = torch.tensor(
            [[focal, 0.0, size / 2.0], [0.0, focal, size / 2.0], [0.0, 0.0, 1.0]],
            device=self.device, dtype=torch.float32,
        )
        up = torch.tensor([0.0, 1.0, 0.0], device=self.device, dtype=torch.float32)
        bg = torch.tensor([0.0, 0.0, 0.0], device=self.device, dtype=torch.float32)

        frames = []
        for i in range(int(num_frames)):
            yaw = 2.0 * math.pi * i / float(num_frames)
            eye = center + torch.tensor(
                [dist * math.sin(yaw), dist * 0.12, dist * math.cos(yaw)],
                device=self.device, dtype=torch.float32,
            )
            viewmat = _look_at_world2cam(eye, center, up)
            renders, _, _ = rasterization(
                means=means, quats=quats, scales=scales, opacities=opac, colors=colors,
                viewmats=viewmat.unsqueeze(0), Ks=K.unsqueeze(0),
                width=int(size), height=int(size), render_mode="RGB", backgrounds=bg,
            )
            frames.append((renders[0].detach().cpu().clamp(0, 1).numpy() * 255).astype(np.uint8))
        return frames

    # ==================== 多视角（multi-view）帧映射 ====================

    def _count_flat_frames(self):
        """统计磁盘上展开后的帧文件数量（逐视角）。"""
        if not os.path.exists(self.ego_dir):
            return 0
        files = [f for f in os.listdir(self.ego_dir)
                 if f.startswith("frame_") and f.endswith("_ego.json")]
        return len(files)

    def _detect_num_views(self):
        """检测该场景由几个视角合成。

        依据：数据导出时每个真实帧会导出一张聚合预览图 view_XXXX.png，同时
        每个视角导出一个 view_XXXX_<view>.npy。因此：
            num_views = (#view_*.npy) / (#view_*.png)
        若无这些文件则回退为单视角（num_views=1）。
        """
        try:
            entries = os.listdir(self.scene_path)
        except OSError:
            return 1

        png_frames = set()
        npy_views = {}  # frame_str -> set(view_idx)
        for name in entries:
            if not name.startswith("view_"):
                continue
            stem, ext = os.path.splitext(name)
            ext = ext.lower()
            parts = stem.split("_")  # ["view", "0000"] 或 ["view", "0000", "1"]
            if ext == ".png" and len(parts) == 2:
                png_frames.add(parts[1])
            elif ext == ".npy" and len(parts) == 3:
                npy_views.setdefault(parts[1], set()).add(parts[2])

        if npy_views:
            counts = {len(v) for v in npy_views.values()}
            # 取众数（最常见的视角数），避免个别帧缺文件干扰
            best = max(counts, key=lambda c: sum(1 for v in npy_views.values() if len(v) == c))
            if best >= 1:
                return best
        if png_frames and self.num_flat_frames:
            # 回退：flat 帧数 / 真实帧数（png 张数）
            ratio = self.num_flat_frames / float(len(png_frames))
            if ratio >= 1 and abs(ratio - round(ratio)) < 1e-6:
                return int(round(ratio))
        return 1

    def flat_index(self, real_frame, view=0):
        """真实帧 + 视角 -> 磁盘上的展开帧索引。"""
        return int(real_frame) * self.num_views + int(view)

    def real_and_view(self, flat_frame):
        """磁盘展开帧索引 -> (真实帧, 视角)。"""
        return int(flat_frame) // self.num_views, int(flat_frame) % self.num_views

    def frame_count(self):
        """对外统一的"可渲染帧数"（多视角已合并，且可被 extend_frames 延长）。"""
        return int(self.total_frames)

    def data_frame_count(self):
        """磁盘上真实存在的帧数（相机/动态 PLY 的上限）。"""
        return int(self.num_data_frames)

    def extend_frames(self, extra_frames=None, total_frames=None, cap=None):
        """把"可渲染帧数"延长到指定值，返回 (旧值, 新值)。

        - `total_frames` 直接给目标总数；`extra_frames` 给增量。
        - `cap` 是硬上限（默认 DGGT_MAX_FRAMES，4000），防止误传超大值。
        - 只改计数；超出数据帧的相机位姿与物体外观分别由 camera_provider /
          object_pose_provider + 外观缓存兜底（见 `_ego_data`、`_appearance_objects`）。
        """
        cap = int(os.environ.get("DGGT_MAX_FRAMES", "4000") if cap is None else cap)
        old = int(self.total_frames)
        if total_frames is not None:
            target = int(total_frames)
        elif extra_frames is not None:
            target = old + int(extra_frames)
        else:
            return old, old
        target = max(int(self.num_data_frames), min(cap, target))
        self.total_frames = target
        return old, target

    def _ego_template_data(self):
        """取最后一份存在的自车 json 作为"相机参数模板"（内参/分辨率/兜底位姿）。"""
        if self._ego_template is not None:
            return self._ego_template
        last = None
        for flat in range(self.num_flat_frames - 1, -1, -1):
            fp = os.path.join(self.ego_dir, f"frame_{flat:04d}_ego.json")
            if os.path.exists(fp):
                last = fp
                break
        if last is None:
            return None
        with open(last, "r") as f:
            data = json.load(f)
        self._ego_template = {
            "camera_intrinsics": data.get("camera_intrinsics"),
            "camera": data.get("camera", {}),
            "camera_extrinsics_world": data.get("camera_extrinsics_world"),
        }
        return self._ego_template

    def _ego_data(self, real_frame):
        """返回某真实帧的自车相机数据（含内参/分辨率/外参）。

        数据帧：直接读 `frame_XXXX_ego.json`。
        **超出数据帧**：用模板里的内参与分辨率，位姿优先向 `camera_provider`（自车轨迹外推）
        要，拿不到就冻结在最后一帧。这样"延长帧"也能出图，而不是直接抛 FileNotFoundError。
        """
        cam_flat = self.flat_index(int(real_frame), 0)
        path = os.path.join(self.ego_dir, f"frame_{cam_flat:04d}_ego.json")
        if os.path.exists(path):
            with open(path, "r") as f:
                data = json.load(f)
            if self._ego_template is None:
                self._ego_template = {
                    "camera_intrinsics": data.get("camera_intrinsics"),
                    "camera": data.get("camera", {}),
                    "camera_extrinsics_world": data.get("camera_extrinsics_world"),
                }
            return data
        tmpl = self._ego_template_data()
        if tmpl is None:
            raise FileNotFoundError(
                f"帧 {real_frame} 超出数据范围({self.num_data_frames})，且找不到可用的自车相机模板")
        c2w = None
        if self.camera_provider is not None:
            try:
                c2w = self.camera_provider(int(real_frame))
            except Exception as e:  # noqa: BLE001
                if self.verbose:
                    print(f"[DGGT] 帧 {real_frame} 相机外推失败，冻结最后一帧: {e}")
        if c2w is None:
            c2w = tmpl.get("camera_extrinsics_world")
        if hasattr(c2w, "detach"):          # torch.Tensor -> list
            c2w = c2w.detach().cpu().numpy()
        return {
            "camera_extrinsics_world": np.asarray(c2w, dtype=np.float32).tolist()
            if c2w is not None else None,
            "camera_intrinsics": tmpl.get("camera_intrinsics"),
            "camera": tmpl.get("camera", {}),
            "synthesized": True,
        }

    def _appearance_objects(self, real_frame, view):
        """某帧某视角该渲染的高斯：当帧 PLY 存在就用它（并刷新外观缓存），
        否则回退到缓存里"该物体最后一次出现"的外观。

        返回 ([(raw_id, gs_slice_dict)], 是否来自缓存, 该视角"整帧"gs)；
        第三项供"克隆 donor 外观"的合成参与者复用（缓存回退时用缓存拼一个等价结构）。
        """
        flat = self.flat_index(int(real_frame), int(view))
        dyn_ply_path = os.path.join(self.dynamic_dir, f"frame_{flat:04d}_dynamic.ply")
        if os.path.exists(dyn_ply_path):
            dyn_gs = self._load_ply(dyn_ply_path)
            if "object_ids" not in dyn_gs:
                return [], False, None
            out = []
            for uid in torch.unique(dyn_gs["object_ids"]):
                uid_int = int(uid.item())
                if uid_int == -1:
                    continue
                mask = dyn_gs["object_ids"] == uid
                part = {
                    "means": dyn_gs["means"][mask],
                    "quats": dyn_gs["quats"][mask],
                    "scales": dyn_gs["scales"][mask],
                    "opacities": dyn_gs["opacities"][mask],
                    "colors": dyn_gs["colors"][mask],
                }
                self._obj_appearance[(int(view), uid_int)] = part
                out.append((uid_int, part))
            return out, False, dyn_gs

        cached = [(uid, part) for (v, uid), part in self._obj_appearance.items() if v == int(view)]
        cached.sort(key=lambda t: t[0])
        if not cached:
            return [], True, None
        # 用缓存拼一个等价的"整帧 gs"，这样 donor 克隆路径在超范围帧也能用
        synth = {k: torch.cat([part[k] for _, part in cached], dim=0)
                 for k in ("means", "quats", "scales", "opacities", "colors")}
        synth["object_ids"] = torch.cat(
            [torch.full((int(part["means"].shape[0]),), int(uid), device=self.device,
                        dtype=torch.int64) for uid, part in cached], dim=0)
        return cached, True, synth

    def _load_frame_objects(self, frame_idx):
        obj_meta_path = os.path.join(self.meta_dir, f"frame_{frame_idx:04d}_objects.json")
        if not os.path.exists(obj_meta_path):
            return []
        with open(obj_meta_path, "r") as f:
            return json.load(f)

    def load_real_frame_objects(self, real_frame):
        """合并同一真实帧下所有视角的动态物体。

        为避免不同视角的 object_id 冲突，合并后的 object_id 采用
        view * VIEW_ID_STRIDE + raw_object_id 编码（单视角时保持原始 id 不变）。
        返回列表中每个元素额外带 "view" 与 "raw_object_id" 字段。
        """
        merged = []
        for view in range(self.num_views):
            flat = self.flat_index(real_frame, view)
            for obj in self._load_frame_objects(flat):
                raw_id = int(obj["object_id"])
                composite = raw_id if self.num_views <= 1 else view * VIEW_ID_STRIDE + raw_id
                item = dict(obj)
                item["object_id"] = composite
                item["raw_object_id"] = raw_id
                item["view"] = view
                merged.append(item)
        return merged

    def _default_pose_for(self, real_frame, view, raw_object_id):
        """返回某真实帧下指定视角、指定 raw object 的默认世界位姿（4x4 list）。

        超出数据帧时该帧没有元数据：退回"该物体最后一次出现"的位姿（冻结），
        而不是单位阵——否则物体会被瞬移到世界原点。
        """
        flat = self.flat_index(real_frame, view)
        for obj in self._load_frame_objects(flat):
            if int(obj["object_id"]) == int(raw_object_id):
                pose = obj["pose_world"]
                self._obj_last_pose = getattr(self, "_obj_last_pose", {})
                self._obj_last_pose[(int(view), int(raw_object_id))] = pose
                return pose
        last = getattr(self, "_obj_last_pose", {}).get((int(view), int(raw_object_id)))
        if last is not None:
            return last
        return np.eye(4, dtype=np.float32).tolist()

    def _get_frame_object_pose(self, frame_idx, object_id):
        """**内部**接口：frame_idx 是 flat 帧号（多视角场景由调用方换算）。"""
        pose = self.trajectory_controller.evaluate_pose(object_id, frame_idx)
        if pose is not None:
            return torch.tensor(pose, device=self.device).float() if not isinstance(pose, torch.Tensor) else pose
        for obj in self._load_frame_objects(frame_idx):
            if obj["object_id"] == object_id:
                return torch.tensor(obj["pose_world"], device=self.device).float()
        return None

    def get_real_frame_object_pose(self, real_frame, object_id):
        """**对外**接口：真实帧号。

        多视角场景下 `ego_pose/gaussians/dynamic_objects` 都是 flat 排列
        （flat = real * num_views + view），而"编辑过的轨迹"是按真实帧号存的，
        所以这里分两步：先按真实帧查编辑轨迹，再按 view0 的 flat 帧查原始位姿。
        """
        rf = int(real_frame)
        pose = self.trajectory_controller.evaluate_pose(object_id, rf)
        if pose is not None:
            return torch.tensor(pose, device=self.device).float() if not isinstance(pose, torch.Tensor) else pose
        try:
            flat = self.flat_index(rf, 0) if self.num_views > 1 else rf
        except Exception:  # noqa: BLE001
            flat = rf
        for obj in self._load_frame_objects(flat):
            if obj["object_id"] == object_id:
                return torch.tensor(obj["pose_world"], device=self.device).float()
        return None

    def _interp_pose(self, pose_a, pose_b, alpha):
        pose_a = np.asarray(pose_a, dtype=np.float32)
        pose_b = np.asarray(pose_b, dtype=np.float32)
        out = np.eye(4, dtype=np.float32)
        out[:3, 3] = (1 - alpha) * pose_a[:3, 3] + alpha * pose_b[:3, 3]
        rot = Slerp([0.0, 1.0], R.from_matrix(np.stack([pose_a[:3, :3], pose_b[:3, :3]], axis=0)))([alpha])[0]
        out[:3, :3] = rot.as_matrix()
        return out

    def _build_track_id_map(self, num_frames, start_idx=0):
        """构建 track_id 映射（增强版：多帧历史 + 运动预测 + 全局优化）。
        
        改进点：
        1. 维护多帧历史缓存，支持物体短暂消失后重关联
        2. 基于历史轨迹预测下一帧位置
        3. 自适应距离阈值（根据运动速度调整）
        4. 全局后处理：合并断裂的track
        """
        # 第一遍：使用增强的逐帧匹配
        all_frames = []
        track_history = {}  # track_id -> {centers: [], dimensions: [], last_frame: int, velocity: np.array}
        next_track_id = 0
        
        for i in range(num_frames):
            frame_idx = start_idx + i
            objs = self.load_real_frame_objects(frame_idx)
            matched, track_history, next_track_id = self._match_objects_enhanced(
                objs, track_history, next_track_id, frame_idx
            )
            all_frames.append((frame_idx, matched))
        
        # 第二遍：全局优化 - 合并断裂的track
        all_frames = self._merge_broken_tracks(all_frames, start_idx)
        
        return all_frames

    def _match_objects_enhanced(self, frame_objects, track_history, next_track_id, frame_idx,
                                 max_history=15, base_dist_thresh=8.0, size_thresh=2.0,
                                 velocity_weight=0.6, lost_frames_tolerance=8):
        """增强版跨帧匹配算法。
        
        Args:
            frame_objects: 当前帧物体列表
            track_history: 轨迹历史 {track_id: {centers, dimensions, last_frame, velocity}}
            next_track_id: 下一个可用的track_id
            frame_idx: 当前帧索引
            max_history: 保留的最大历史帧数（增加到15以获得更稳定的速度估计）
            base_dist_thresh: 基础距离阈值（增加到8.0以适应更快运动的物体）
            size_thresh: 尺寸变化阈值（增加到2.0以容忍更大的测量误差）
            velocity_weight: 速度预测权重（降低到0.6以平衡预测和实际位置）
            lost_frames_tolerance: 允许物体消失的最大帧数（增加到8以处理短暂遮挡）
        
        Returns:
            matched: 匹配后的物体列表（带track_id）
            track_history: 更新后的轨迹历史
            next_track_id: 更新后的下一个track_id
        """
        matched = []
        used_tracks = set()
        
        # 清理过期的track（消失超过lost_frames_tolerance帧）
        expired = [tid for tid, info in track_history.items() 
                   if frame_idx - info["last_frame"] > lost_frames_tolerance]
        for tid in expired:
            del track_history[tid]
        
        # 为当前帧的每个物体寻找最佳匹配
        for obj in frame_objects:
            center = np.asarray(obj["pose_world"], dtype=np.float32)[:3, 3]
            dims = np.asarray(obj["dimensions"], dtype=np.float32)
            
            best_track, best_score = None, float("inf")
            
            for tid, info in track_history.items():
                if tid in used_tracks:
                    continue
                
                # 尺寸匹配
                avg_dims = np.mean(info["dimensions"], axis=0)
                size_dist = np.linalg.norm(dims - avg_dims)
                if size_dist > size_thresh:
                    continue
                
                # 计算预测位置
                predicted_center = info["centers"][-1].copy()
                if info.get("velocity") is not None and len(info["centers"]) >= 2:
                    # 使用速度预测
                    gap = frame_idx - info["last_frame"]
                    predicted_center = predicted_center + info["velocity"] * gap * velocity_weight
                
                # 计算距离（考虑预测位置）
                actual_dist = np.linalg.norm(center - info["centers"][-1])
                predicted_dist = np.linalg.norm(center - predicted_center)
                
                # 使用预测距离和实际距离的加权（增加对预测的信任）
                dist = min(actual_dist, predicted_dist * 0.8 + actual_dist * 0.2)
                
                # 自适应阈值：根据历史速度调整
                if info.get("velocity") is not None:
                    speed = np.linalg.norm(info["velocity"])
                    # 速度越快，阈值越大，增加系数从1.5到2.5
                    adaptive_thresh = base_dist_thresh + speed * 2.5
                else:
                    adaptive_thresh = base_dist_thresh
                
                if dist > adaptive_thresh:
                    continue
                
                # 综合评分：距离 + 尺寸 + 时间间隔惩罚（降低时间惩罚）
                time_gap = frame_idx - info["last_frame"]
                time_penalty = time_gap * 0.15  # 降低时间惩罚，允许更长的间隙
                # 降低尺寸权重，增加距离权重
                score = predicted_dist * 1.2 + 0.2 * size_dist + time_penalty
                
                if score < best_score:
                    best_score = score
                    best_track = tid
            
            # 分配track_id
            if best_track is None:
                best_track = next_track_id
                next_track_id += 1
            else:
                used_tracks.add(best_track)
            
            matched.append({**obj, "track_id": best_track})
            
            # 更新轨迹历史
            if best_track not in track_history:
                track_history[best_track] = {
                    "centers": [],
                    "dimensions": [],
                    "last_frame": frame_idx,
                    "velocity": None
                }
            
            history = track_history[best_track]
            history["centers"].append(center)
            history["dimensions"].append(dims)
            history["last_frame"] = frame_idx
            
            # 限制历史长度
            if len(history["centers"]) > max_history:
                history["centers"] = history["centers"][-max_history:]
                history["dimensions"] = history["dimensions"][-max_history:]
            
            # 计算速度（使用最近几帧的平均速度）
            if len(history["centers"]) >= 2:
                recent_centers = history["centers"][-3:]  # 最近3帧
                if len(recent_centers) >= 2:
                    velocities = []
                    for j in range(1, len(recent_centers)):
                        velocities.append(recent_centers[j] - recent_centers[j-1])
                    history["velocity"] = np.mean(velocities, axis=0)
        
        return matched, track_history, next_track_id

    def _merge_broken_tracks(self, all_frames, start_idx, merge_dist_thresh=5.0, merge_time_thresh=12):
        """全局后处理：合并断裂的track。
        
        检测同一物理物体因跟踪失败而分裂成多个track的情况，进行合并。
        
        Args:
            all_frames: [(frame_idx, matched_objects)]
            start_idx: 起始帧索引
            merge_dist_thresh: 合并的距离阈值（增加到5.0）
            merge_time_thresh: 合并的时间阈值（增加到12帧）
        """
        # 构建track的首次/末次出现信息
        track_info = {}
        for frame_idx, objs in all_frames:
            for obj in objs:
                tid = obj["track_id"]
                center = np.asarray(obj["pose_world"], dtype=np.float32)[:3, 3]
                dims = np.asarray(obj["dimensions"], dtype=np.float32)
                
                if tid not in track_info:
                    track_info[tid] = {
                        "first_frame": frame_idx,
                        "last_frame": frame_idx,
                        "first_center": center.copy(),
                        "last_center": center.copy(),
                        "avg_dims": dims.copy(),
                        "dim_sum": dims.copy(),  # 累加和
                        "count": 1,
                        "centers": [center.copy()],  # 保存所有中心点用于计算速度
                        "frames": [frame_idx]
                    }
                else:
                    track_info[tid]["last_frame"] = frame_idx
                    track_info[tid]["last_center"] = center.copy()
                    track_info[tid]["dim_sum"] = track_info[tid]["dim_sum"] + dims
                    track_info[tid]["count"] += 1
                    track_info[tid]["avg_dims"] = track_info[tid]["dim_sum"] / track_info[tid]["count"]
                    track_info[tid]["centers"].append(center.copy())
                    track_info[tid]["frames"].append(frame_idx)
        
        # 检测可合并的track对
        merge_map = {}  # old_tid -> new_tid
        track_ids = sorted(track_info.keys())
        
        for i, tid1 in enumerate(track_ids):
            if tid1 in merge_map:
                continue
            info1 = track_info[tid1]
            
            for j in range(i + 1, len(track_ids)):
                tid2 = track_ids[j]
                if tid2 in merge_map:
                    continue
                info2 = track_info[tid2]
                
                # 检查时间连续性：track1结束后track2开始
                time_gap = info2["first_frame"] - info1["last_frame"]
                if time_gap <= 0 or time_gap > merge_time_thresh:
                    continue
                
                # 检查尺寸相似性（放宽到1.5）
                size_dist = np.linalg.norm(info1["avg_dims"] - info2["avg_dims"])
                if size_dist > 1.5:
                    continue
                
                # 检查位置连续性：track1的末尾位置与track2的起始位置
                # 使用运动预测
                if len(info1["centers"]) >= 2:
                    # 计算track1的平均速度
                    recent_centers = info1["centers"][-3:]
                    if len(recent_centers) >= 2:
                        velocity = np.mean([recent_centers[k] - recent_centers[k-1] 
                                           for k in range(1, len(recent_centers))], axis=0)
                        # 预测track1在track2起始帧的位置
                        predicted_center = info1["last_center"] + velocity * time_gap
                        dist = np.linalg.norm(predicted_center - info2["first_center"])
                    else:
                        dist = np.linalg.norm(info1["last_center"] - info2["first_center"])
                else:
                    dist = np.linalg.norm(info1["last_center"] - info2["first_center"])
                
                # 使用自适应距离阈值
                adaptive_merge_thresh = merge_dist_thresh + time_gap * 0.3
                if dist < adaptive_merge_thresh:
                    # 合并：将tid2合并到tid1
                    merge_map[tid2] = tid1
        
        # 应用合并
        if merge_map:
            # 递归查找最终track_id
            def find_final_tid(tid):
                if tid not in merge_map:
                    return tid
                return find_final_tid(merge_map[tid])
            
            for frame_idx, objs in all_frames:
                for obj in objs:
                    old_tid = obj["track_id"]
                    if old_tid in merge_map:
                        obj["track_id"] = find_final_tid(old_tid)
        
        return all_frames

    def export_track_id_legend(self, track_frames, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        lines = []
        seen = {}
        for frame_idx, objs in track_frames:
            for obj in objs:
                tid = obj["track_id"]
                if tid not in seen:
                    seen[tid] = obj
                    c = np.asarray(obj["pose_world"], dtype=np.float32)[:3, 3]
                    lines.append(f"track_id={tid}, first_seen_frame={frame_idx}, raw_object_id={obj['object_id']}, center=({c[0]:.2f}, {c[1]:.2f}, {c[2]:.2f}), dims=({obj['dimensions'][0]:.2f}, {obj['dimensions'][1]:.2f}, {obj['dimensions'][2]:.2f})")
        with open(os.path.join(output_dir, "track_id_legend.txt"), "w") as f:
            f.write("\n".join(lines))
        return seen

    def set_camera_pose(self, frame_idx, pose_matrix):
        if pose_matrix is None:
            self.cam_overrides.pop(frame_idx, None)
        else:
            self.cam_overrides[frame_idx] = torch.as_tensor(pose_matrix, device=self.device, dtype=torch.float32)

    def set_object_pose(self, frame_idx, object_id, pose_matrix):
        self.obj_overrides.setdefault(frame_idx, {})
        if pose_matrix is None:
            self.obj_overrides[frame_idx].pop(object_id, None)
        else:
            self.obj_overrides[frame_idx][object_id] = torch.as_tensor(pose_matrix, device=self.device, dtype=torch.float32)

    def export_scene_edit_spec(self, output_path, name, scenario_type, actors, start_frame, duration, metadata=None):
        spec = {
            "name": name,
            "scene_path": self.scene_path,
            "scenario_type": scenario_type,
            "actors": list(map(int, actors)),
            "start_frame": int(start_frame),
            "duration": int(duration),
            "metadata": dict(metadata or {}),
            "trajectories": self.trajectory_controller.to_spec()["trajectories"],
        }
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(spec, f, indent=2)
        return spec

    def export_trajectory_controller_spec(self, output_path, metadata=None):
        return self.trajectory_controller.save_spec(output_path, metadata=metadata)

    def _render_frame_with_object_overrides(self, t, object_overrides, c2w_override=None, K_override=None, width_override=None, height_override=None, extra_objects=None, include_dynamic=True):
        """渲染真实帧 t。

        多视角数据下，t 为“真实帧”索引；相机默认取该真实帧 view0 的自车位姿，
        动态物体则合并该真实帧下所有视角的高斯（共享世界坐标系）。
        object_overrides / extra_objects 的物体 id 采用合并后的 composite id
        （view * VIEW_ID_STRIDE + raw_object_id，单视角时即原始 id）。
        """
        cam_flat = self.flat_index(t, 0)
        # 数据帧直接读盘；超出数据帧时按自车轨迹合成为相机（见 _ego_data）
        ego_data = self._ego_data(t)
        c2w = c2w_override if c2w_override is not None else self.cam_overrides.get(t, torch.tensor(ego_data["camera_extrinsics_world"], device=self.device).float())
        if c2w.shape == (3, 4):
            c2w = torch.cat([c2w, torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=self.device, dtype=c2w.dtype)], dim=0)
        K = K_override if K_override is not None else torch.tensor(ego_data["camera_intrinsics"], device=self.device).float()
        if width_override is not None and height_override is not None:
            width, height = int(width_override), int(height_override)
        else:
            width, height = ego_data["camera"]["width"], ego_data["camera"]["height"]
        viewmat = torch.inverse(c2w)

        dyn_means, dyn_quats, dyn_scales, dyn_opac, dyn_cols = [], [], [], [], []
        # 缓存各视角已加载的动态高斯，供合成参与者克隆复用
        view_dyn_gs = {}

        if include_dynamic and not self.static_only:
            for view in range(self.num_views):
                objs, from_cache, dyn_gs = self._appearance_objects(t, view)
                # 克隆 donor 外观（合成参与者）时要用"该视角的整帧 gs"；
                # 缓存回退时上面已经把缓存拼成了等价结构，所以这条路也能用
                if dyn_gs is not None:
                    view_dyn_gs[view] = dyn_gs
                id_offset = 0 if self.num_views <= 1 else view * VIEW_ID_STRIDE
                for uid_int, part in objs:
                    composite = uid_int + id_offset
                    # 显式传入 None 表示删除该物体（不渲染）
                    if composite in object_overrides and object_overrides[composite] is None:
                        continue
                    transform = object_overrides.get(composite)
                    if transform is None and self.object_pose_provider is not None:
                        # 超出数据帧：位姿交给轨迹（TrackManager），物体就能"按路线"继续动
                        try:
                            transform = self.object_pose_provider(composite, int(t))
                        except Exception:  # noqa: BLE001
                            transform = None
                    if transform is not None and not isinstance(transform, torch.Tensor):
                        transform = torch.as_tensor(np.asarray(transform, dtype=np.float32),
                                                    device=self.device, dtype=torch.float32)
                    if transform is None:
                        # 再兜底：该物体最后一次出现的位姿（冻结），最后才用当帧默认
                        transform = torch.tensor(
                            self._default_pose_for(t, view, uid_int),
                            device=self.device,
                        ).float()
                    p_world, q_world = self._transform_gaussians(part["means"], part["quats"], transform)
                    dyn_means.append(p_world)
                    dyn_quats.append(q_world)
                    dyn_scales.append(part["scales"])
                    dyn_opac.append(part["opacities"])
                    dyn_cols.append(part["colors"])

        # 渲染合成参与者（克隆 donor 高斯，或 SAM3D 自带 .ply 高斯）
        if extra_objects and not self.static_only:
            for ex in extra_objects:
                transform = ex.get("transform")
                if transform is None:
                    continue
                # 允许传入 numpy：统一转到当前 device 的 float32 tensor
                if not isinstance(transform, torch.Tensor):
                    transform = torch.as_tensor(np.asarray(transform, dtype=np.float32),
                                                device=self.device, dtype=torch.float32)
                # SAM3D 重建物体：自带 .ply 高斯
                if ex.get("ply_path"):
                    gs = self._load_sam3d_ply(ex["ply_path"])
                    if gs is None:
                        continue
                    means = gs["means"]
                    quats = gs["quats"]
                    scales = gs["scales"]
                    center = ex.get("center")
                    scale = float(ex.get("scale", 1.0))
                    scale_vec = ex.get("scale_vec")
                    model_corr = ex.get("model_corr")
                    if center is not None:
                        means = means - torch.as_tensor(center, device=self.device, dtype=means.dtype)
                    # 模型朝向修正：只作用于 gaussians（先质心归零，再旋转），不影响包围盒
                    if model_corr is not None:
                        T4 = torch.eye(4, device=self.device, dtype=torch.float32)
                        T4[:3, :3] = torch.as_tensor(model_corr, device=self.device, dtype=torch.float32)
                        means, quats = self._transform_gaussians(means, quats, T4)
                    if scale_vec is not None:
                        # 逐轴缩放：把重建模型精确套进目标包围盒（宽/高/长分别对齐）。
                        # 高斯半径按逐轴缩放的几何平均近似（半径远小于物体尺寸，视觉无差）。
                        sv = torch.as_tensor(np.asarray(scale_vec, dtype=np.float32),
                                             device=self.device, dtype=means.dtype).reshape(3).clamp(min=1e-6)
                        means = means * sv
                        scales = scales * float(torch.prod(sv).item() ** (1.0 / 3.0))
                    elif scale != 1.0:
                        means = means * scale
                        scales = scales * scale
                    p_world, q_world = self._transform_gaussians(means, quats, transform)
                    dyn_means.append(p_world)
                    dyn_quats.append(q_world)
                    dyn_scales.append(scales)
                    dyn_opac.append(gs["opacities"])
                    dyn_cols.append(gs["colors"])
                    # SAM3D 模型没有烘焙阴影，补一圈接触阴影，避免看起来"飘"
                    if ex.get("shadow", True):
                        try:
                            sdims = ex.get("dims") or [2.0, 1.5, 4.5]
                            sm, sq, ss, so, sc = self._contact_shadow(
                                transform, sdims, float(ex.get("up_sign", -1.0)),
                                ground_y=ex.get("ground_y"))
                            dyn_means.append(sm)
                            dyn_quats.append(sq)
                            dyn_scales.append(ss)
                            dyn_opac.append(so)
                            dyn_cols.append(sc)
                        except Exception as _e:  # noqa: BLE001
                            if not getattr(self, "_shadow_warned", False):
                                self._shadow_warned = True
                                print("[shadow] contact shadow failed, skipping:", repr(_e), flush=True)
                    continue
                donor_id = ex.get("donor_object_id")
                if donor_id is None:
                    continue
                donor_view = int(donor_id) // VIEW_ID_STRIDE if self.num_views > 1 else 0
                donor_raw = int(donor_id) % VIEW_ID_STRIDE if self.num_views > 1 else int(donor_id)
                dyn_gs = view_dyn_gs.get(donor_view)
                if dyn_gs is None or "object_ids" not in dyn_gs:
                    continue
                mask = dyn_gs["object_ids"] == donor_raw
                if mask.any():
                    p_world, q_world = self._transform_gaussians(dyn_gs["means"][mask], dyn_gs["quats"][mask], transform)
                    dyn_means.append(p_world)
                    dyn_quats.append(q_world)
                    dyn_scales.append(dyn_gs["scales"][mask])
                    dyn_opac.append(dyn_gs["opacities"][mask])
                    dyn_cols.append(dyn_gs["colors"][mask])

        # 渲染 SAM 3D 导入物体（自带 .ply 高斯，无需 donor 克隆）
        if self.sam3d_objects and not self.static_only:
            for _obj_id, spec in self.sam3d_objects.items():
                gs = self._load_sam3d_ply(spec["ply_path"])
                if gs is None:
                    continue
                pose = torch.as_tensor(spec["pose"], device=self.device, dtype=torch.float32)
                scale = float(spec.get("scale", 1.0))
                means = gs["means"]
                quats = gs["quats"]
                scales = gs["scales"]
                center = spec.get("center")
                if center is not None:
                    means = means - torch.as_tensor(center, device=self.device, dtype=means.dtype)
                if scale != 1.0:
                    means = means * scale
                    # scales 已是线性（实际）尺度（_load_sam3d_ply 里做了 exp），直接相乘
                    scales = scales * scale
                p_world, q_world = self._transform_gaussians(means, quats, pose)
                dyn_means.append(p_world)
                dyn_quats.append(q_world)
                dyn_scales.append(scales)
                dyn_opac.append(gs["opacities"])
                dyn_cols.append(gs["colors"])
        all_means, all_quats, all_scales, all_opac, all_cols = [], [], [], [], []
        if self.sky_gs is not None:
            all_means.append(self.sky_gs["means"])
            all_quats.append(self.sky_gs["quats"])
            all_scales.append(self.sky_gs["scales"])
            all_opac.append(self.sky_gs["opacities"])
            all_cols.append(self.sky_gs["colors"])
        all_means.append(self.static_gs["means"])
        all_quats.append(self.static_gs["quats"])
        all_scales.append(self.static_gs["scales"])
        all_opac.append(self.static_gs["opacities"])
        all_cols.append(self.static_gs["colors"])
        if dyn_means:
            all_means.extend(dyn_means)
            all_quats.extend(dyn_quats)
            all_scales.extend(dyn_scales)
            all_opac.extend(dyn_opac)
            all_cols.extend(dyn_cols)
        renders, _, _ = rasterization(means=torch.cat(all_means, dim=0), quats=torch.cat(all_quats, dim=0), scales=torch.cat(all_scales, dim=0), opacities=torch.cat(all_opac, dim=0), colors=torch.cat(all_cols, dim=0), viewmats=viewmat.unsqueeze(0), Ks=K.unsqueeze(0), width=width, height=height, render_mode="RGB")
        return (renders[0].detach().cpu().clamp(0, 1).numpy() * 255).astype(np.uint8)

    def _transform_gaussians(self, means, quats, transform_mat):
        R_mat = transform_mat[:3, :3]
        T_vec = transform_mat[:3, 3]
        new_means = (R_mat @ means.T).T + T_vec
        q_trans_np = R.from_matrix(R_mat.cpu().numpy()).as_quat()
        q_trans_wxyz = np.r_[q_trans_np[3], q_trans_np[:3]]
        q_trans = torch.tensor(q_trans_wxyz, device=self.device, dtype=torch.float32)
        def quat_mult(q1, q2):
            w1, x1, y1, z1 = q1.unbind(-1)
            w2, x2, y2, z2 = q2.unbind(-1)
            return torch.stack((w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2, w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2), dim=-1)
        return new_means, quat_mult(q_trans.unsqueeze(0).expand_as(quats), quats)

    def _contact_shadow(self, pose, dims, up_sign=-1.0, strength=0.42, n=48, seed=1234,
                        ground_y=None):
        """在物体底部生成一圈"接触阴影"高斯（SAM3D 模型本身没有烘焙阴影，所以看起来发飘）。

        用确定性采样（固定随机种子）避免逐帧闪烁；扁平高斯拼成一个椭圆暗斑。
        `ground_y` 给定时直接贴到该世界 y 上（由 TrackManager 从静态高斯估计出的**真实路面**），
        否则退回"包围盒底面"——4DGS 包围盒在高度方向常不准，会让阴影飘在空中。
        """
        dev = self.device
        # 注意：extra_objects 里的 transform 是 CUDA tensor，不能走 np.asarray（会 TypeError
        # 并被外层的 except 静默吞掉，结果就是"阴影完全不出现"）。
        if isinstance(pose, torch.Tensor):
            pose_t = pose.detach().to(device=dev, dtype=torch.float32)
        else:
            pose_t = torch.as_tensor(np.asarray(pose, dtype=np.float32), device=dev,
                                     dtype=torch.float32)
        c, R = pose_t[:3, 3], pose_t[:3, :3]
        w = float(dims[0]) if len(dims) > 0 else 1.8
        h = float(dims[1]) if len(dims) > 1 else 1.5
        l = float(dims[2]) if len(dims) > 2 else 4.5
        if ground_y is not None and np.isfinite(float(ground_y)):
            # 贴到真实路面：再沿"上"抬 2cm 避免与地面 z-fighting
            bottom = torch.tensor(
                [float(c[0]), float(ground_y) + float(up_sign) * 0.02, float(c[2])],
                device=dev, dtype=torch.float32)
        else:
            # 底部位置：中心沿"下"方向 h/2，再朝"上"抬 2cm 避免与地面 z-fighting
            bottom = c + torch.tensor(
                [0.0, (-float(up_sign)) * (h / 2.0) + float(up_sign) * 0.02, 0.0],
                device=dev, dtype=torch.float32)
        g = torch.Generator(device="cpu")
        g.manual_seed(int(seed))
        r = torch.sqrt(torch.rand(n, generator=g))
        th = torch.rand(n, generator=g) * (2.0 * math.pi)
        lx = torch.as_tensor(r * torch.cos(th), device=dev, dtype=torch.float32) * (w * 0.60)
        lz = torch.as_tensor(r * torch.sin(th), device=dev, dtype=torch.float32) * (l * 0.54)
        ex, ez = R[:, 0], R[:, 2]
        means = bottom.unsqueeze(0) + lx.unsqueeze(1) * ex.unsqueeze(0) + lz.unsqueeze(1) * ez.unsqueeze(0)
        fall = (1.0 - r.to(dev)).clamp(0.15, 1.0)
        base = 0.05 * max(w, l)
        sx = base * fall + 0.05
        scales = torch.stack([sx, torch.full_like(sx, 0.012), sx * (l / max(w, 1e-3)) * 0.9 + 0.05], dim=-1)
        quats = torch.tensor([1.0, 0.0, 0.0, 0.0], device=dev).expand(n, 4).clone()
        opac = (strength * fall).clamp(0.05, 0.65).to(dev)
        cols = torch.full((n, 3), 0.03, device=dev, dtype=torch.float32)
        return means, quats, scales, opac, cols

    def _get_bbox_corners_2d(self, pose, dimensions, viewmat, K, W, H):
        dx, dy, dz = dimensions[0] / 2.0, dimensions[1] / 2.0, dimensions[2] / 2.0
        corners_local = torch.tensor([[-dx, -dy, -dz], [dx, -dy, -dz], [dx, dy, -dz], [-dx, dy, -dz], [-dx, -dy, dz], [dx, -dy, dz], [dx, dy, dz], [-dx, dy, dz]], device=self.device, dtype=torch.float32)
        corners_world = torch.cat([corners_local, torch.ones((8, 1), device=self.device)], dim=1) @ pose.T
        corners_cam = corners_world @ viewmat.T
        xyz = corners_cam[:, :3]
        if (xyz[:, 2] < 0.1).any():
            return None
        uv_w = (K @ xyz.T).T
        return (uv_w[:, :2] / (uv_w[:, 2:3] + 1e-6)).detach().cpu().numpy()

    def _draw_bbox(self, image, corners_2d, color=(0, 255, 0), thickness=2, label=None):
        if corners_2d is None:
            return image
        pts = corners_2d.astype(np.int32)
        edges = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
        h, w = image.shape[:2]
        for start, end in edges:
            pt1 = tuple(pts[start])
            pt2 = tuple(pts[end])
            if -w < pt1[0] < 2 * w and -h < pt1[1] < 2 * h:
                cv2.line(image, pt1, pt2, color, thickness)
        if label is not None:
            anchor = tuple(pts[0])
            cv2.putText(image, str(label), (int(anchor[0]), int(anchor[1]) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        return image

    def render_sequence(self, num_frames, output_dir, draw_bboxes=False, draw_ids=False, start_idx=0, save_video=False, video_name="rendered_video.mp4", video_fps=10, export_id_map=False, temporal_upsample_factor=1):
        os.makedirs(output_dir, exist_ok=True)
        track_frames = self._build_track_id_map(num_frames, start_idx=start_idx) if export_id_map else None
        if export_id_map:
            self.export_track_id_legend(track_frames, output_dir)
        frame_records = []
        for i in range(num_frames):
            frame_idx = start_idx + i
            frame_records.append((frame_idx, 0.0, None))
            if temporal_upsample_factor > 1 and i < num_frames - 1:
                for k in range(1, temporal_upsample_factor):
                    frame_records.append((frame_idx, k / float(temporal_upsample_factor), frame_idx + 1))
        rendered_frames = []
        for base_frame_idx, alpha, next_frame_idx in frame_records:
            object_overrides = {}
            c2w_override = None
            if next_frame_idx is None:
                for oid in self.trajectory_controller.get_object_ids():
                    pose = self._get_frame_object_pose(base_frame_idx, oid)
                    if pose is not None:
                        object_overrides[oid] = pose
                render_frame_idx = base_frame_idx
            else:
                for oid in self.trajectory_controller.get_object_ids():
                    pose_a = self._get_frame_object_pose(base_frame_idx, oid)
                    pose_b = self._get_frame_object_pose(next_frame_idx, oid)
                    if pose_a is None or pose_b is None:
                        continue
                    object_overrides[oid] = torch.tensor(self._interp_pose(pose_a.cpu().numpy(), pose_b.cpu().numpy(), alpha), device=self.device).float()
                render_frame_idx = base_frame_idx
            image_out = self._render_frame_with_object_overrides(render_frame_idx, object_overrides, c2w_override=c2w_override)
            rendered_frames.append(image_out)
            import imageio
            imageio.imwrite(os.path.join(output_dir, f"render_frame_{base_frame_idx:04d}_{int(alpha * 1000):03d}.png"), image_out)
        if save_video and rendered_frames:
            video_path = os.path.join(output_dir, video_name)
            try:
                with imageio.get_writer(video_path, fps=video_fps, codec="libx264", format="FFMPEG") as writer:
                    for frame in rendered_frames:
                        writer.append_data(frame)
            except Exception:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                first = rendered_frames[0]
                video_writer = cv2.VideoWriter(video_path, fourcc, video_fps, (first.shape[1], first.shape[0]))
                for frame in rendered_frames:
                    video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                video_writer.release()


def parse_args():
    parser = argparse.ArgumentParser(description="DGGT scene renderer")
    parser.add_argument("--scene_path", type=str, help="场景根目录；当未使用 spec 时必填")
    parser.add_argument("--output_dir", type=str, help="输出目录；当未使用 spec 时必填")
    parser.add_argument("--num_frames", type=int, default=20)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--draw_bboxes", action="store_true")
    parser.add_argument("--draw_ids", action="store_true")
    parser.add_argument("--no_sky", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--save_video", action="store_true")
    parser.add_argument("--export_id_map", action="store_true")
    parser.add_argument("--static_only", action="store_true")
    parser.add_argument("--video_name", type=str, default="rendered_video.mp4")
    parser.add_argument("--video_fps", type=int, default=10)
    parser.add_argument("--temporal_upsample_factor", type=int, default=1)
    parser.add_argument("--edit_spec", type=str, default=None, help="scene edit spec JSON 路径")
    parser.add_argument("--edit_spec_dir", type=str, default=None, help="scene edit spec 目录，目录内每个 json 会按顺序批量执行")
    parser.add_argument("--trajectory_spec", type=str, default=None, help="trajectory spec JSON 路径")
    parser.add_argument("--corner_case", type=str, default=None, help="自动生成的 corner case 类型")
    parser.add_argument("--corner_case_output_spec", type=str, default=None, help="corner case spec 导出路径")
    parser.add_argument("--corner_case_metadata", type=str, default=None, help="corner case 元数据 JSON")
    parser.add_argument("--corner_batch_root", type=str, default=None, help="批量 corner case 数据集输出根目录")
    parser.add_argument("--corner_batch_scenarios", type=str, default=None, help="批量生成的场景列表，逗号分隔")
    parser.add_argument("--corner_batch_severities", type=str, default=None, help="批量生成的强度列表，逗号分隔")
    return parser.parse_args()


def run_legacy_mode(args):
    engine = DGGTRenderer(args.scene_path, device=args.device, load_sky=not args.no_sky, verbose=not args.quiet, static_only=args.static_only)
    engine.render_sequence(args.num_frames, args.output_dir, draw_bboxes=args.draw_bboxes, draw_ids=args.draw_ids, start_idx=args.start_idx, save_video=args.save_video, video_name=args.video_name, video_fps=args.video_fps, export_id_map=args.export_id_map, temporal_upsample_factor=args.temporal_upsample_factor)


def main():
    args = parse_args()
    if args.scene_path is None or args.output_dir is None:
        raise ValueError("请提供 --scene_path 和 --output_dir")
    run_legacy_mode(args)


if __name__ == "__main__":
    main()
