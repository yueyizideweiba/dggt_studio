import os
import json
import argparse

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

        self.static_gs = self._load_ply(self.static_ply)
        self.sky_gs = self._load_ply(self.sky_ply) if load_sky and os.path.exists(self.sky_ply) else None
        self.cam_overrides = {}
        self.obj_overrides = {}
        self.trajectory_controller = TrajectoryController(device=self.device)
        self.corner_case_generator = CornerCaseGenerator(self.trajectory_controller)

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

    def _load_frame_objects(self, frame_idx):
        obj_meta_path = os.path.join(self.meta_dir, f"frame_{frame_idx:04d}_objects.json")
        if not os.path.exists(obj_meta_path):
            return []
        with open(obj_meta_path, "r") as f:
            return json.load(f)

    def _get_frame_object_pose(self, frame_idx, object_id):
        pose = self.trajectory_controller.evaluate_pose(object_id, frame_idx)
        if pose is not None:
            return torch.tensor(pose, device=self.device).float() if not isinstance(pose, torch.Tensor) else pose
        for obj in self._load_frame_objects(frame_idx):
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
        all_frames = []
        prev_tracks = {}
        for i in range(num_frames):
            frame_idx = start_idx + i
            objs = self._load_frame_objects(frame_idx)
            matched, prev_tracks = self._match_objects_across_frames(objs, prev_tracks)
            all_frames.append((frame_idx, matched))
        return all_frames

    def _match_objects_across_frames(self, frame_objects, prev_tracks, dist_thresh=3.5, size_thresh=1.2):
        matched = []
        used_prev = set()
        next_tracks = {}
        next_track_id = 0 if not prev_tracks else (max(prev_tracks.keys()) + 1)
        for obj in frame_objects:
            center = np.asarray(obj["pose_world"], dtype=np.float32)[:3, 3]
            dims = np.asarray(obj["dimensions"], dtype=np.float32)
            best_track, best_score = None, float("inf")
            for tid, track in prev_tracks.items():
                if tid in used_prev:
                    continue
                center_dist = np.linalg.norm(center - track["center"])
                size_dist = np.linalg.norm(dims - track["dimensions"])
                if center_dist > dist_thresh or size_dist > size_thresh:
                    continue
                score = center_dist + 0.5 * size_dist
                if score < best_score:
                    best_score = score
                    best_track = tid
            if best_track is None:
                best_track = next_track_id
                next_track_id += 1
            else:
                used_prev.add(best_track)
            matched.append({**obj, "track_id": best_track})
            next_tracks[best_track] = {"center": center, "dimensions": dims}
        return matched, next_tracks

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

    def _render_frame_with_object_overrides(self, t, object_overrides, c2w_override=None, K_override=None, width_override=None, height_override=None):
        ego_path = os.path.join(self.ego_dir, f"frame_{t:04d}_ego.json")
        with open(ego_path, "r") as f:
            ego_data = json.load(f)
        c2w = c2w_override if c2w_override is not None else self.cam_overrides.get(t, torch.tensor(ego_data["camera_extrinsics_world"], device=self.device).float())
        if c2w.shape == (3, 4):
            c2w = torch.cat([c2w, torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=self.device, dtype=c2w.dtype)], dim=0)
        K = K_override if K_override is not None else torch.tensor(ego_data["camera_intrinsics"], device=self.device).float()
        if width_override is not None and height_override is not None:
            width, height = int(width_override), int(height_override)
        else:
            width, height = ego_data["camera"]["width"], ego_data["camera"]["height"]
        viewmat = torch.inverse(c2w)
        dyn_ply_path = os.path.join(self.dynamic_dir, f"frame_{t:04d}_dynamic.ply")
        obj_meta_path = os.path.join(self.meta_dir, f"frame_{t:04d}_objects.json")
        dyn_means, dyn_quats, dyn_scales, dyn_opac, dyn_cols = [], [], [], [], []
        if not self.static_only and os.path.exists(dyn_ply_path):
            dyn_gs = self._load_ply(dyn_ply_path)
            with open(obj_meta_path, "r") as f:
                obj_meta = json.load(f)
            default_poses = {o["object_id"]: torch.tensor(o["pose_world"], device=self.device).float() for o in obj_meta}
            for uid in torch.unique(dyn_gs["object_ids"]):
                uid_int = int(uid.item())
                if uid_int == -1:
                    continue
                # 显式传入 None 表示删除该物体（不渲染）
                if uid_int in object_overrides and object_overrides[uid_int] is None:
                    continue
                mask = dyn_gs["object_ids"] == uid
                transform = object_overrides.get(uid_int, default_poses.get(uid_int, torch.eye(4, device=self.device)))
                p_world, q_world = self._transform_gaussians(dyn_gs["means"][mask], dyn_gs["quats"][mask], transform)
                dyn_means.append(p_world)
                dyn_quats.append(q_world)
                dyn_scales.append(dyn_gs["scales"][mask])
                dyn_opac.append(dyn_gs["opacities"][mask])
                dyn_cols.append(dyn_gs["colors"][mask])
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
