from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import torch

from .geometry import to_pose_matrix


@dataclass
class AssetInstance:
    asset_id: str
    source_type: str
    source_path: str
    default_pose: List[List[float]]
    metadata: Dict[str, Any] = field(default_factory=dict)


class SceneObjectAssetBank:
    def __init__(self, bank_path: str | Path):
        self.bank_path = Path(bank_path)
        self.assets: Dict[str, AssetInstance] = {}
        if self.bank_path.exists():
            self._load()

    def _load(self) -> None:
        with self.bank_path.open("r", encoding="utf-8") as f:
            raw = json.load(f)
        for item in raw.get("assets", []):
            asset = AssetInstance(
                asset_id=item["asset_id"],
                source_type=item.get("source_type", "dynamic_ply"),
                source_path=item["source_path"],
                default_pose=item.get("default_pose", _identity_pose_list()),
                metadata=item.get("metadata", {}) or {},
            )
            self.assets[asset.asset_id] = asset

    def get(self, asset_id: str) -> AssetInstance:
        if asset_id not in self.assets:
            raise KeyError(f"asset_id not found in bank: {asset_id}")
        return self.assets[asset_id]

    def resolve_pose(self, asset_id: str, pose_like=None) -> torch.Tensor:
        if pose_like is None:
            pose_like = self.get(asset_id).default_pose
        return to_pose_matrix(pose_like)

    def available_assets(self) -> List[str]:
        return sorted(self.assets.keys())


def _identity_pose_list() -> List[List[float]]:
    return [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
