"""
轨迹/Track 管理模块

DGGT 每帧的动态物体 object_id 在不同帧之间 **不连续**（同一物理物体在不同帧
可能有不同的 raw object_id）。为了让"编辑某个物体""显示运动轨迹"在整个序列中
指向同一物理物体，需要用 **track_id** 跨帧追踪。

本模块：
- 基于 DGGTRenderer 的跨帧匹配（_match_objects_across_frames）构建稳定的 track_id。
- 维护 (frame_idx, raw_object_id) <-> track_id 的双向映射。
- 保存基于 track_id 的编辑（位姿关键帧 / 删除）。
- 在渲染某一帧时，把 track 级编辑翻译成该帧 raw object_id 的 override。
"""

import os
import json
import numpy as np
import torch


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

        # 撤销/重做快照栈（保存 (track_edits, track_deleted) 的深拷贝）
        self._undo_stack = []
        self._redo_stack = []
        self._max_history = 50

        self._build()

    # ==================== 撤销/重做 ====================

    def _snapshot(self):
        edits = {tid: {f: p.copy() for f, p in fp.items()} for tid, fp in self.track_edits.items()}
        return (edits, set(self.track_deleted))

    def _restore(self, snap):
        edits, deleted = snap
        self.track_edits = {tid: {f: p.copy() for f, p in fp.items()} for tid, fp in edits.items()}
        self.track_deleted = set(deleted)

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
        ego_dir = self.renderer.ego_dir
        if not os.path.exists(ego_dir):
            return 0
        files = [f for f in os.listdir(ego_dir)
                 if f.startswith("frame_") and f.endswith("_ego.json")]
        return len(files)

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

    # ==================== 查询 ====================

    def get_track_id(self, frame_idx, raw_object_id):
        return self.frame_obj_to_track.get((int(frame_idx), int(raw_object_id)))

    def get_raw_object_id(self, track_id, frame_idx):
        """返回某 track 在指定帧的 raw object_id（不存在则 None）。"""
        return self.track_to_frames.get(int(track_id), {}).get(int(frame_idx))

    def get_track_frames(self, track_id):
        return sorted(self.track_to_frames.get(int(track_id), {}).keys())

    def is_deleted(self, track_id):
        return int(track_id) in self.track_deleted

    # ==================== 位姿求值（含编辑） ====================

    def get_track_pose(self, track_id, frame_idx):
        """获取某 track 在指定帧的位姿。

        语义（逐顶点编辑）：
        - 该帧有编辑关键帧 → 直接用编辑值；
        - 否则 → 用该帧原始位姿（未被编辑的帧保持不动）。
        这样拖动某一帧的轨迹顶点只改变该帧，物体沿"被编辑后的逐帧轨迹"运动。
        """
        track_id = int(track_id)
        frame_idx = int(frame_idx)

        edits = self.track_edits.get(track_id)
        if edits and frame_idx in edits:
            return np.asarray(edits[frame_idx], dtype=np.float32)

        # 原始位姿（未编辑帧保持不变）
        raw = self.track_raw_poses.get(track_id, {}).get(frame_idx)
        if raw is not None:
            return np.asarray(raw, dtype=np.float32)

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

    def set_track_pose(self, track_id, frame_idx, pose_matrix):
        """为某 track 在指定帧设置位姿关键帧。"""
        track_id = int(track_id)
        pose = np.asarray(pose_matrix, dtype=np.float32)
        self.track_edits.setdefault(track_id, {})[int(frame_idx)] = pose
        self.track_deleted.discard(track_id)

    def set_track_trajectory(self, track_id, keyframes):
        """用关键帧列表 [(frame_idx, pose4x4)] 设置整条编辑轨迹。"""
        track_id = int(track_id)
        self.track_edits[track_id] = {
            int(f): np.asarray(p, dtype=np.float32) for f, p in keyframes
        }
        self.track_deleted.discard(track_id)

    def delete_track(self, track_id):
        track_id = int(track_id)
        self.track_deleted.add(track_id)
        self.track_edits.pop(track_id, None)

    def clear_track_edits(self, track_id):
        track_id = int(track_id)
        self.track_edits.pop(track_id, None)
        self.track_deleted.discard(track_id)

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
        for tid in self.track_deleted:
            raw_id = self.get_raw_object_id(tid, frame_idx)
            if raw_id is not None:
                overrides[raw_id] = None

        # 位姿编辑
        for tid in self.track_edits:
            if tid in self.track_deleted:
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

    # ==================== 帧级物体列表（带 track_id） ====================

    def get_frame_objects(self, frame_idx):
        """返回该帧动态物体列表，附带 track_id、当前位姿（含编辑）、是否编辑/删除。"""
        frame_idx = int(frame_idx)
        result = []
        for obj in self.renderer._load_frame_objects(frame_idx):
            raw_id = int(obj["object_id"])
            tid = self.get_track_id(frame_idx, raw_id)
            if tid is None:
                continue
            if tid in self.track_deleted:
                continue  # 已删除的不返回
            pose = self.get_track_pose(tid, frame_idx)
            if pose is None:
                pose = np.asarray(obj["pose_world"], dtype=np.float32)
            pose_list = pose.astype(float).tolist()
            result.append({
                "track_id": tid,
                "raw_object_id": raw_id,
                "type": obj.get("type", "未知"),
                "dimensions": obj.get("dimensions", []),
                "pose_world": pose_list,
                "center": [pose_list[0][3], pose_list[1][3], pose_list[2][3]],
                "edited": self.is_track_edited(tid),
            })
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
