from __future__ import annotations

import torch


def make_translation(tx: float = 0.0, ty: float = 0.0, tz: float = 0.0, device: str | torch.device = "cpu") -> torch.Tensor:
    pose = torch.eye(4, device=device, dtype=torch.float32)
    pose[0, 3] = float(tx)
    pose[1, 3] = float(ty)
    pose[2, 3] = float(tz)
    return pose


def make_yaw_rotation(yaw_deg: float, device: str | torch.device = "cpu") -> torch.Tensor:
    yaw = torch.tensor(float(yaw_deg) * 3.1415926 / 180.0, device=device)
    c = torch.cos(yaw)
    s = torch.sin(yaw)
    rot = torch.eye(4, device=device, dtype=torch.float32)
    rot[0, 0] = c
    rot[0, 1] = -s
    rot[1, 0] = s
    rot[1, 1] = c
    return rot


def to_pose_matrix(pose_like, device: str | torch.device = "cpu") -> torch.Tensor:
    if isinstance(pose_like, torch.Tensor):
        return pose_like.to(device=device, dtype=torch.float32)
    return torch.tensor(pose_like, device=device, dtype=torch.float32)
