from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional


EditActionType = Literal[
    "translate_track",
    "insert_object",
    "delete_object",
    "replace_object",
    "swap_tracks",
    "interpolate_track",
    "clip_track",
    "velocity_perturb",
    "heading_perturb",
    "time_shift",
    "stop_track",
    "collision_course",
]


@dataclass
class SceneTarget:
    track_id: Optional[int] = None
    object_id: Optional[int] = None
    frame_idx: Optional[int] = None


@dataclass
class EditAction:
    type: EditActionType
    target: SceneTarget = field(default_factory=SceneTarget)
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SceneEditSpec:
    scene_path: str
    output_dir: str
    num_frames: int
    start_idx: int = 0
    draw_bboxes: bool = False
    draw_ids: bool = False
    save_video: bool = False
    video_name: str = "rendered_video.mp4"
    video_fps: int = 10
    load_sky: bool = True
    static_only: bool = False
    actions: List[EditAction] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
