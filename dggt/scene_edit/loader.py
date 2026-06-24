from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from .specs import EditAction, SceneEditSpec, SceneTarget


def _load_actions(raw_actions: List[Dict[str, Any]]) -> List[EditAction]:
    actions: List[EditAction] = []
    for item in raw_actions:
        target_raw = item.get("target", {}) or {}
        target = SceneTarget(
            track_id=target_raw.get("track_id"),
            object_id=target_raw.get("object_id"),
            frame_idx=target_raw.get("frame_idx"),
        )
        actions.append(
            EditAction(
                type=item["type"],
                target=target,
                params=item.get("params", {}) or {},
            )
        )
    return actions


def load_scene_edit_spec(spec_path: str | Path) -> SceneEditSpec:
    path = Path(spec_path)
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    return SceneEditSpec(
        scene_path=raw["scene_path"],
        output_dir=raw["output_dir"],
        num_frames=int(raw["num_frames"]),
        start_idx=int(raw.get("start_idx", 0)),
        draw_bboxes=bool(raw.get("draw_bboxes", False)),
        draw_ids=bool(raw.get("draw_ids", False)),
        save_video=bool(raw.get("save_video", False)),
        video_name=raw.get("video_name", "rendered_video.mp4"),
        video_fps=int(raw.get("video_fps", 10)),
        load_sky=bool(raw.get("load_sky", True)),
        static_only=bool(raw.get("static_only", False)),
        actions=_load_actions(raw.get("actions", [])),
        metadata=raw.get("metadata", {}) or {},
    )


def load_scene_edit_specs(spec_dir: str | Path) -> List[SceneEditSpec]:
    directory = Path(spec_dir)
    return [load_scene_edit_spec(path) for path in sorted(directory.glob("*.json"))]


def dump_scene_edit_spec(spec: SceneEditSpec, path: str | Path) -> None:
    path = Path(path)
    payload = {
        "scene_path": spec.scene_path,
        "output_dir": spec.output_dir,
        "num_frames": spec.num_frames,
        "start_idx": spec.start_idx,
        "draw_bboxes": spec.draw_bboxes,
        "draw_ids": spec.draw_ids,
        "save_video": spec.save_video,
        "video_name": spec.video_name,
        "video_fps": spec.video_fps,
        "load_sky": spec.load_sky,
        "static_only": spec.static_only,
        "actions": [
            {
                "type": action.type,
                "target": {
                    "track_id": action.target.track_id,
                    "object_id": action.target.object_id,
                    "frame_idx": action.target.frame_idx,
                },
                "params": action.params,
            }
            for action in spec.actions
        ],
        "metadata": spec.metadata,
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
