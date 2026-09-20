"""多视角重建场景的"真实帧 ↔ 文件帧"映射（统一入口，别再各写一份）。

背景（踩过的坑）：多视角重建时，`ego_pose/`、`gaussians/`、`dynamic_objects/` 里的文件是
**逐视角展开（flat）** 的，排列为：

    flat = real_frame * num_views + view

而 `view_XXXX.png`（预览图）是**每个真实帧一张**，`view_XXXX_<view>.npy` 是每个视角一份。
所以 `num_views = #view_*.npy / #view_*.png`。

推论：读 `ego_pose` 时**不能**直接用真实帧号，必须 `flat_index(real, view)`；
view 0 是重建时喂进去的第一台相机（studio 里的"自车视角"就是它）。
直接在 3 视角场景里用真实帧号读，会拿到"别的相机的位姿"——位置几乎一样、**朝向差 45~90°**，
表现出来就是"多视角数据加载不对/轨迹乱跳"。
"""
from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np


def num_views(scene_dir: str) -> int:
    """从文件名推断视角数（没有多视角文件时返回 1）。"""
    try:
        names = os.listdir(scene_dir)
    except OSError:
        return 1
    npng = sum(1 for n in names if n.startswith("view_") and n.endswith(".png"))
    nnpy = sum(1 for n in names if n.startswith("view_") and n.endswith(".npy"))
    if npng > 0 and nnpy > 0:
        return max(1, nnpy // npng)
    if nnpy > 0 and npng == 0:
        # 只有 npy：按"每个真实帧的视角数"猜不了，退回 1（由调用方另行指定）
        return 1
    return 1


def num_real_frames(scene_dir: str, view: int = 0) -> int:
    """真实帧数：优先用 view_*.png 的个数，否则用 ego_pose 文件数 / 视角数。"""
    nv = num_views(scene_dir)
    try:
        npng = sum(1 for n in os.listdir(scene_dir)
                   if n.startswith("view_") and n.endswith(".png"))
    except OSError:
        npng = 0
    if npng:
        return npng
    ego = os.path.join(scene_dir, "ego_pose")
    try:
        n = sum(1 for f in os.listdir(ego) if f.endswith("_ego.json"))
    except OSError:
        return 0
    return max(1, n // max(1, nv))


def flat_index(real_frame: int, view: int, n_views: int) -> int:
    return int(real_frame) * int(max(1, n_views)) + int(view)


def unflat(flat_frame: int, n_views: int) -> Tuple[int, int]:
    nv = int(max(1, n_views))
    return int(flat_frame) // nv, int(flat_frame) % nv


def ego_pose_path(scene_dir: str, real_frame: int, view: int = 0,
                  n_views: Optional[int] = None) -> str:
    nv = int(n_views) if n_views else num_views(scene_dir)
    return os.path.join(scene_dir, "ego_pose",
                        f"frame_{flat_index(real_frame, view, nv):04d}_ego.json")


def load_ego_pose(scene_dir: str, real_frame: int, view: int = 0,
                  n_views: Optional[int] = None) -> Optional[Dict]:
    p = ego_pose_path(scene_dir, real_frame, view, n_views)
    if not os.path.exists(p):
        return None
    try:
        return json.load(open(p, "r"))
    except Exception:  # noqa: BLE001
        return None


def load_ego_cam(scene_dir: str, real_frame: int, view: int = 0,
                 n_views: Optional[int] = None) -> Optional[Tuple[np.ndarray, np.ndarray, int, int]]:
    """→ (c2w 4x4, K 3x3, W, H)，与 studio 的 `_frame_camera` 语义一致（4x4）。"""
    ego = load_ego_pose(scene_dir, real_frame, view, n_views)
    if ego is None:
        return None
    c2w = np.asarray(ego["camera_extrinsics_world"], dtype=np.float64)
    if c2w.shape == (3, 4):
        c2w = np.vstack([c2w, np.array([[0.0, 0.0, 0.0, 1.0]])])
    K = np.asarray(ego["camera_intrinsics"], dtype=np.float64)
    W = int(ego["camera"]["width"])
    H = int(ego["camera"]["height"])
    return c2w, K, W, H


def training_cameras(scene_dir: str) -> List[np.ndarray]:
    """所有训练相机位姿（多视角场景里就是全部 flat 帧的相机）——用于 pose novelty。"""
    out: List[np.ndarray] = []
    ego = os.path.join(scene_dir, "ego_pose")
    if not os.path.isdir(ego):
        return out
    for f in sorted(os.listdir(ego)):
        if not f.endswith("_ego.json"):
            continue
        try:
            j = json.load(open(os.path.join(ego, f), "r"))
            m = np.asarray(j["camera_extrinsics_world"], dtype=np.float64)
            if m.shape == (3, 4):
                m = np.vstack([m, np.array([[0.0, 0.0, 0.0, 1.0]])])
            out.append(m)
        except Exception:  # noqa: BLE001
            continue
    return out


def yaw_of(c2w: np.ndarray) -> float:
    P = np.asarray(c2w, dtype=np.float64)
    return float(np.arctan2(P[0, 2], P[2, 2]))
