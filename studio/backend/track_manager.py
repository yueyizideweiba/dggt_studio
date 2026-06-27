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

        self._build()

    # ==================== 撤销/重做 ====================

    def _snapshot(self):
        edits = {tid: {f: p.copy() for f, p in fp.items()} for tid, fp in self.track_edits.items()}
        synth = {}
        for sid, s in self.synthetic_tracks.items():
            synth[sid] = {
                "donor_object_id": s["donor_object_id"],
                "donor_track_id": s.get("donor_track_id"),
                "donor_pose0": np.asarray(s.get("donor_pose0", np.eye(4)), dtype=np.float32).copy(),
                "dimensions": list(s.get("dimensions", [])),
                "type": s.get("type", "合成"),
                "poses": {f: p.copy() for f, p in s["poses"].items()},
            }
        return (edits, set(self.track_deleted), synth, self._next_synth_id)

    def _restore(self, snap):
        edits, deleted, synth, next_id = snap
        self.track_edits = {tid: {f: p.copy() for f, p in fp.items()} for tid, fp in edits.items()}
        self.track_deleted = set(deleted)
        self.synthetic_tracks = {}
        for sid, s in synth.items():
            self.synthetic_tracks[sid] = {
                "donor_object_id": s["donor_object_id"],
                "donor_track_id": s.get("donor_track_id"),
                "donor_pose0": np.asarray(s.get("donor_pose0", np.eye(4)), dtype=np.float32).copy(),
                "dimensions": list(s.get("dimensions", [])),
                "type": s.get("type", "合成"),
                "poses": {f: p.copy() for f, p in s["poses"].items()},
            }
        self._next_synth_id = next_id

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
        """返回某 track 在指定帧的 raw object_id（不存在则 None）。合成 track 无 raw id。"""
        return self.track_to_frames.get(int(track_id), {}).get(int(frame_idx))

    def is_synthetic(self, track_id):
        return int(track_id) in self.synthetic_tracks

    def get_track_frames(self, track_id):
        track_id = int(track_id)
        if track_id in self.synthetic_tracks:
            return sorted(self.synthetic_tracks[track_id]["poses"].keys())
        return sorted(self.track_to_frames.get(track_id, {}).keys())

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
        return sid

    def set_synthetic_pose(self, synth_id, frame_idx, pose):
        synth_id = int(synth_id)
        if synth_id not in self.synthetic_tracks:
            return
        self.synthetic_tracks[synth_id]["poses"][int(frame_idx)] = np.asarray(pose, dtype=np.float32)

    def remove_synthetic_track(self, synth_id):
        self.synthetic_tracks.pop(int(synth_id), None)

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
        if frame_idx not in frames:
            return []

        cur_pose = self.get_track_pose(track_id, frame_idx)
        if cur_pose is None:
            return []
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

    def build_extra_objects(self, frame_idx):
        """构建该帧需要额外渲染的合成参与者列表（克隆 donor 高斯）。

        动态物体高斯在 PLY 中以物体局部坐标存储，正常渲染时直接乘该帧
        pose_world。因此合成参与者也直接使用目标 pose_world，并按 donor track
        在当前帧的 raw object_id 克隆外观，避免 raw object_id 跨帧不连续导致闪烁。
        """
        frame_idx = int(frame_idx)
        device = self.renderer.device
        extras = []
        for sid, s in self.synthetic_tracks.items():
            target = self.get_track_pose(sid, frame_idx)
            if target is None:
                continue
            donor_track_id = s.get("donor_track_id")
            donor_raw_id = None
            if donor_track_id is not None:
                donor_raw_id = self.get_raw_object_id(donor_track_id, frame_idx)
            if donor_raw_id is None:
                donor_raw_id = s.get("donor_object_id")
            transform = np.asarray(target, dtype=np.float32)
            extras.append({
                "donor_object_id": int(donor_raw_id),
                "transform": torch.tensor(transform, device=device, dtype=torch.float32),
                "synth_track_id": int(sid),
            })
        return extras

    # ==================== 帧级物体列表（带 track_id） ====================

    def get_frame_objects(self, frame_idx):
        """返回该帧动态物体列表，附带 track_id、当前位姿（含编辑）、是否编辑/删除。

        包含原始 track 与合成参与者（synthetic）。
        """
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
                "synthetic": False,
            })

        # 合成参与者
        for sid, s in self.synthetic_tracks.items():
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
