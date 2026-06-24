from __future__ import annotations

from typing import Optional

import torch

from .asset_bank import SceneObjectAssetBank
from .geometry import make_translation, make_yaw_rotation, to_pose_matrix
from .specs import EditAction, SceneEditSpec


class SceneEditExecutor:
    def __init__(self, engine, asset_bank: SceneObjectAssetBank | None = None):
        self.engine = engine
        self.asset_bank = asset_bank

    def apply_spec(self, spec: SceneEditSpec) -> None:
        for action in spec.actions:
            self.apply_action(action, start_idx=spec.start_idx, num_frames=spec.num_frames)

    def apply_action(self, action: EditAction, start_idx: int = 0, num_frames: Optional[int] = None) -> None:
        action_type = action.type
        if action_type == "translate_track":
            self._translate_track(action, start_idx, num_frames)
        elif action_type == "insert_object":
            self._insert_object(action)
        elif action_type == "delete_object":
            self._delete_object(action, start_idx, num_frames)
        elif action_type == "replace_object":
            self._replace_object(action, start_idx, num_frames)
        elif action_type == "swap_tracks":
            self._swap_tracks(action, start_idx, num_frames)
        elif action_type == "interpolate_track":
            self._interpolate_track(action, start_idx, num_frames)
        elif action_type == "clip_track":
            self._clip_track(action, start_idx, num_frames)
        elif action_type == "velocity_perturb":
            self._velocity_perturb(action, start_idx, num_frames)
        elif action_type == "heading_perturb":
            self._heading_perturb(action, start_idx, num_frames)
        elif action_type == "time_shift":
            self._time_shift(action, start_idx, num_frames)
        elif action_type == "stop_track":
            self._stop_track(action, start_idx, num_frames)
        elif action_type == "collision_course":
            self._collision_course(action, start_idx, num_frames)
        else:
            raise ValueError(f"Unsupported action type: {action_type}")

    def _track_frames(self, start_idx: int, num_frames: Optional[int]):
        if num_frames is None:
            raise ValueError("num_frames is required for track-level actions")
        return self.engine._build_track_id_map(num_frames, start_idx=start_idx)

    def _translate_track(self, action: EditAction, start_idx: int, num_frames: Optional[int]):
        tid = action.target.track_id
        if tid is None:
            raise ValueError("translate_track requires target.track_id")
        dx = float(action.params.get("dx", 0.0))
        dy = float(action.params.get("dy", 0.0))
        dz = float(action.params.get("dz", 0.0))
        pose = make_translation(dx, dy, dz, device=self.engine.device)
        for frame_idx, objs in self._track_frames(start_idx, num_frames):
            for obj in objs:
                if obj.get("track_id") == tid:
                    self.engine.set_object_pose(frame_idx, obj["object_id"], pose)

    def _insert_object(self, action: EditAction):
        frame_idx = action.target.frame_idx
        object_id = action.target.object_id
        if frame_idx is None or object_id is None:
            raise ValueError("insert_object requires target.frame_idx and target.object_id")
        if self.asset_bank is not None and "asset_id" in action.params:
            _ = self.asset_bank.resolve_pose(action.params["asset_id"], action.params.get("pose_world"))
        pose = to_pose_matrix(action.params.get("pose_world"), device=self.engine.device)
        self.engine.set_object_pose(frame_idx, object_id, pose)

    def _delete_object(self, action: EditAction, start_idx: int, num_frames: Optional[int]):
        tid = action.target.track_id
        object_id = action.target.object_id
        if tid is None and object_id is None:
            raise ValueError("delete_object requires target.track_id or target.object_id")
        for frame_idx, objs in self._track_frames(start_idx, num_frames):
            for obj in objs:
                if (tid is not None and obj.get("track_id") == tid) or (object_id is not None and obj.get("object_id") == object_id):
                    self.engine.set_object_pose(frame_idx, obj["object_id"], None)

    def _replace_object(self, action: EditAction, start_idx: int, num_frames: Optional[int]):
        self._delete_object(action, start_idx, num_frames)
        replacement = action.params.get("replacement", [])
        for item in replacement:
            pose = to_pose_matrix(item["pose_world"], device=self.engine.device)
            self.engine.set_object_pose(int(item["frame_idx"]), int(item["object_id"]), pose)

    def _swap_tracks(self, action: EditAction, start_idx: int, num_frames: Optional[int]):
        track_a = action.params.get("track_a")
        track_b = action.params.get("track_b")
        if track_a is None or track_b is None:
            raise ValueError("swap_tracks requires params.track_a and params.track_b")
        for frame_idx, objs in self._track_frames(start_idx, num_frames):
            pose_a = None
            pose_b = None
            raw_a = None
            raw_b = None
            for obj in objs:
                if obj.get("track_id") == track_a:
                    raw_a = obj["object_id"]
                    pose_a = to_pose_matrix(obj["pose_world"], device=self.engine.device)
                elif obj.get("track_id") == track_b:
                    raw_b = obj["object_id"]
                    pose_b = to_pose_matrix(obj["pose_world"], device=self.engine.device)
            if raw_a is not None and raw_b is not None:
                self.engine.set_object_pose(frame_idx, raw_a, pose_b)
                self.engine.set_object_pose(frame_idx, raw_b, pose_a)

    def _interpolate_track(self, action: EditAction, start_idx: int, num_frames: Optional[int]):
        tid = action.target.track_id
        if tid is None:
            raise ValueError("interpolate_track requires target.track_id")
        begin_frame = int(action.params.get("begin_frame", start_idx))
        end_frame = int(action.params.get("end_frame", start_idx + num_frames - 1 if num_frames is not None else begin_frame))
        start_shift = action.params.get("start_shift", [0.0, 0.0, 0.0])
        end_shift = action.params.get("end_shift", start_shift)
        for frame_idx, objs in self._track_frames(start_idx, num_frames):
            if frame_idx < begin_frame or frame_idx > end_frame:
                continue
            alpha = 0.0 if end_frame == begin_frame else (frame_idx - begin_frame) / float(end_frame - begin_frame)
            shift = [float(s0) * (1.0 - alpha) + float(s1) * alpha for s0, s1 in zip(start_shift, end_shift)]
            delta_pose = make_translation(*shift, device=self.engine.device)
            for obj in objs:
                if obj.get("track_id") == tid:
                    base_pose = to_pose_matrix(obj["pose_world"], device=self.engine.device)
                    self.engine.set_object_pose(frame_idx, obj["object_id"], delta_pose @ base_pose)

    def _clip_track(self, action: EditAction, start_idx: int, num_frames: Optional[int]):
        tid = action.target.track_id
        if tid is None:
            raise ValueError("clip_track requires target.track_id")
        begin_frame = int(action.params.get("begin_frame", start_idx))
        end_frame = int(action.params.get("end_frame", start_idx + num_frames - 1 if num_frames is not None else begin_frame))
        for frame_idx, objs in self._track_frames(start_idx, num_frames):
            if begin_frame <= frame_idx <= end_frame:
                continue
            for obj in objs:
                if obj.get("track_id") == tid:
                    self.engine.set_object_pose(frame_idx, obj["object_id"], None)

    def _velocity_perturb(self, action: EditAction, start_idx: int, num_frames: Optional[int]):
        tid = action.target.track_id
        if tid is None:
            raise ValueError("velocity_perturb requires target.track_id")
        scale = float(action.params.get("scale", 1.0))
        base_shift = action.params.get("shift", [0.0, 0.0, 0.0])
        begin_frame = int(action.params.get("begin_frame", start_idx))
        for frame_idx, objs in self._track_frames(start_idx, num_frames):
            if frame_idx < begin_frame:
                continue
            step = frame_idx - begin_frame
            shift = [float(v) * scale * step for v in base_shift]
            delta_pose = make_translation(*shift, device=self.engine.device)
            for obj in objs:
                if obj.get("track_id") == tid:
                    base_pose = to_pose_matrix(obj["pose_world"], device=self.engine.device)
                    self.engine.set_object_pose(frame_idx, obj["object_id"], delta_pose @ base_pose)

    def _heading_perturb(self, action: EditAction, start_idx: int, num_frames: Optional[int]):
        tid = action.target.track_id
        if tid is None:
            raise ValueError("heading_perturb requires target.track_id")
        yaw_deg = float(action.params.get("yaw_deg", 0.0))
        begin_frame = int(action.params.get("begin_frame", start_idx))
        if yaw_deg == 0.0:
            return
        rot = make_yaw_rotation(yaw_deg, device=self.engine.device)
        for frame_idx, objs in self._track_frames(start_idx, num_frames):
            if frame_idx < begin_frame:
                continue
            for obj in objs:
                if obj.get("track_id") == tid:
                    base_pose = to_pose_matrix(obj["pose_world"], device=self.engine.device)
                    self.engine.set_object_pose(frame_idx, obj["object_id"], rot @ base_pose)

    def _time_shift(self, action: EditAction, start_idx: int, num_frames: Optional[int]):
        tid = action.target.track_id
        if tid is None:
            raise ValueError("time_shift requires target.track_id")
        shift_frames = int(action.params.get("shift_frames", 0))
        for frame_idx, objs in self._track_frames(start_idx, num_frames):
            src_frame = frame_idx - shift_frames
            if src_frame < start_idx:
                continue
            src_obj = self._find_track_object_at_frame(tid, src_frame)
            if src_obj is None:
                continue
            src_pose = to_pose_matrix(src_obj["pose_world"], device=self.engine.device)
            for obj in objs:
                if obj.get("track_id") == tid:
                    self.engine.set_object_pose(frame_idx, obj["object_id"], src_pose)

    def _find_track_object_at_frame(self, track_id: int, frame_idx: int):
        for obj in self.engine._load_frame_objects(frame_idx):
            if obj.get("track_id") == track_id:
                return obj
        return None

    def _extract_positions(self, track_id: int, start_idx: int, num_frames: int):
        frames = []
        for frame_idx, objs in self._track_frames(start_idx, num_frames):
            for obj in objs:
                if obj.get("track_id") == track_id:
                    pose = to_pose_matrix(obj["pose_world"], device=self.engine.device)
                    frames.append((frame_idx, obj, pose))
                    break
        return frames

    def _stop_track(self, action: EditAction, start_idx: int, num_frames: Optional[int]):
        tid = action.target.track_id
        if tid is None:
            raise ValueError("stop_track requires target.track_id")
        stop_frame = int(action.params.get("stop_frame", start_idx))
        hold_until = int(action.params.get("hold_until", start_idx + num_frames - 1 if num_frames is not None else stop_frame))
        base = self._extract_positions(tid, start_idx, num_frames)
        if not base:
            return
        hold_pose = None
        for frame_idx, obj, pose in base:
            if frame_idx <= stop_frame:
                hold_pose = pose
                continue
            if frame_idx > hold_until:
                break
            if hold_pose is None:
                hold_pose = pose
            self.engine.set_object_pose(frame_idx, obj["object_id"], hold_pose)

    def _collision_course(self, action: EditAction, start_idx: int, num_frames: Optional[int]):
        lead_tid = action.params.get("lead_track_id")
        follow_tid = action.params.get("follow_track_id")
        if lead_tid is None or follow_tid is None:
            raise ValueError("collision_course requires params.lead_track_id and params.follow_track_id")

        mode = action.params.get("mode", "rear_end")
        collision_frame = int(action.params.get("collision_frame", start_idx + (num_frames or 1) // 2))
        hold_after = int(action.params.get("hold_after", collision_frame))
        decel_start = int(action.params.get("decel_start", max(start_idx, collision_frame - 6)))
        lateral_offset = float(action.params.get("lateral_offset", 0.0))
        slow_down_scale = float(action.params.get("slow_down_scale", 0.35))
        follow_gain = float(action.params.get("follow_gain", 1.0))

        lead_tracks = self._extract_positions(lead_tid, start_idx, num_frames)
        follow_tracks = self._extract_positions(follow_tid, start_idx, num_frames)
        if not lead_tracks or not follow_tracks:
            return

        lead_start_pose = lead_tracks[0][2]
        follow_start_pose = follow_tracks[0][2]
        lead_xy = lead_start_pose[:3, 3].clone()
        follow_xy = follow_start_pose[:3, 3].clone()
        delta = lead_xy - follow_xy
        if mode == "lane_change":
            lateral_axis = action.params.get("lateral_axis", "x")
            offset_value = lateral_offset if lateral_offset != 0.0 else 1.8
            target_follow = follow_xy.clone()
            if lateral_axis == "x":
                target_follow[0] = lead_xy[0] - offset_value
                target_follow[1] = follow_xy[1]
            else:
                target_follow[1] = lead_xy[1] - offset_value
                target_follow[0] = follow_xy[0]
            target_follow[2] = follow_xy[2]
        else:
            target_follow = lead_xy - delta * 0.35

        for idx, (frame_idx, obj, pose) in enumerate(follow_tracks):
            if frame_idx < decel_start:
                continue
            if frame_idx > hold_after:
                break
            alpha = 0.0 if hold_after == decel_start else (frame_idx - decel_start) / float(hold_after - decel_start)
            alpha = max(0.0, min(1.0, alpha))
            if mode == "lane_change":
                interp_target = follow_xy + (target_follow - follow_xy) * alpha
                interp_target[0] = follow_xy[0] + (lead_xy[0] - follow_xy[0]) * alpha * follow_gain
            else:
                interp_target = follow_xy + (target_follow - follow_xy) * (alpha ** slow_down_scale)
            new_pose = pose.clone()
            new_pose[:3, 3] = interp_target
            self.engine.set_object_pose(frame_idx, obj["object_id"], new_pose)

        for frame_idx, obj, pose in lead_tracks:
            if frame_idx < decel_start:
                continue
            if frame_idx > hold_after:
                break
            if mode == "rear_end":
                hold_pose = pose.clone()
                hold_pose[:3, 3] = lead_xy
                self.engine.set_object_pose(frame_idx, obj["object_id"], hold_pose)
            elif mode == "lane_change":
                lead_pose = pose.clone()
                lead_pose[:3, 3] = lead_xy
                self.engine.set_object_pose(frame_idx, obj["object_id"], lead_pose)
