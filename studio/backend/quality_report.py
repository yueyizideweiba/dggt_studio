"""
Corner case quality gate for DGGT Studio.

This module turns generated corner-case metadata into a machine-checkable
quality report. It intentionally depends only on TrackManager pose/bbox data in
its first version; map and rendered-visibility checks are reported as unknown
until those subsystems are wired in.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from dggt.scene_edit.collision_physics import check_collision


DEFAULT_THRESHOLDS = {
    "ttc_min_s": 0.2,
    "ttc_max_s": 3.0,
    "near_miss_distance_m": 2.0,
    "max_speed_mps": 45.0,
    "max_accel_mps2": 12.0,
    "max_jerk_mps3": 40.0,
    "max_yaw_rate_radps": 1.2,
    "max_step_distance_m": 12.0,
    "max_bbox_penetration_m": 1.0,
    "min_pose_coverage": 0.85,
}


def build_quality_report(tm, corner_case_result: Dict[str, Any], fps: float = 10.0,
                         thresholds: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
    """Build a quality report for a generated corner case.

    The report is safe to return directly from the API: it contains plain JSON
    scalars/lists/dicts and a list of gate-level checks with pass/fail status.
    """
    cfg = dict(DEFAULT_THRESHOLDS)
    if thresholds:
        cfg.update(thresholds)

    start_frame = int(corner_case_result.get("start_frame", 0))
    num_frames = int(corner_case_result.get("num_frames", 0))
    frames = list(range(start_frame, start_frame + num_frames + 1))
    affected_tracks = _unique_ints(corner_case_result.get("affected_tracks", []))
    collision_tracks = _unique_ints(corner_case_result.get("collision_tracks", []))
    if len(collision_tracks) < 2:
        collision_tracks = affected_tracks[:2]

    track_metrics = {
        str(tid): _track_motion_metrics(tm, tid, frames, fps)
        for tid in affected_tracks
    }
    pair_metrics = _pair_collision_metrics(tm, collision_tracks, frames)

    collision_analysis = corner_case_result.get("collision_analysis") or {}
    collision_frame = _first_non_none(
        corner_case_result.get("collision_frame"),
        collision_analysis.get("collision_frame"),
        pair_metrics.get("collision_frame"),
    )
    critical_frame = _first_non_none(
        corner_case_result.get("critical_frame"),
        collision_analysis.get("critical_frame"),
    )

    ttc_at_critical = None
    if critical_frame is not None and collision_frame is not None:
        ttc_at_critical = max(0.0, (int(collision_frame) - int(critical_frame)) / float(fps))
    elif collision_analysis.get("time_to_collision") is not None:
        ttc_at_critical = float(collision_analysis["time_to_collision"])

    event_present = bool(collision_frame is not None or pair_metrics.get("min_distance", math.inf) <= cfg["near_miss_distance_m"])
    annotation_consistency = _annotation_consistency(tm, affected_tracks, frames, cfg)

    checks = []
    _add_check(checks, "event_present", event_present,
               "collision or near-miss exists",
               {"collision_frame": collision_frame, "min_distance": pair_metrics.get("min_distance")})
    _add_check(checks, "ttc_range", _ttc_ok(ttc_at_critical, collision_frame, cfg),
               "TTC at critical frame is within the configured range, or this is a near-miss",
               {"ttc_at_critical": ttc_at_critical, "range": [cfg["ttc_min_s"], cfg["ttc_max_s"]]})
    _add_check(checks, "motion_physical", _motion_ok(track_metrics, cfg),
               "speed, acceleration, jerk, yaw-rate, and step distance are below thresholds",
               _motion_summary(track_metrics))
    _add_check(checks, "bbox_penetration", pair_metrics.get("max_penetration", 0.0) <= cfg["max_bbox_penetration_m"],
               "OBB penetration depth is not excessive",
               {"max_penetration": pair_metrics.get("max_penetration"), "threshold": cfg["max_bbox_penetration_m"]})
    _add_check(checks, "annotation_consistency", annotation_consistency["ok"],
               "affected tracks have usable poses and bbox dimensions across the generated frame window",
               annotation_consistency)
    _add_check(checks, "offroad", None,
               "map/drivable-area validation is not wired in yet",
               {"status": "unknown"})
    _add_check(checks, "visibility", None,
               "rendered visibility validation is not wired in yet",
               {"status": "unknown"})

    blocking = [c for c in checks if c["passed"] is False]
    report = {
        "valid": len(blocking) == 0,
        "case_type": corner_case_result.get("scenario_type"),
        "collision_frame": _json_int_or_none(collision_frame),
        "critical_frame": _json_int_or_none(critical_frame),
        "ttc_at_critical": _json_float_or_none(ttc_at_critical),
        "min_distance": _json_float_or_none(pair_metrics.get("min_distance")),
        "max_accel": _json_float_or_none(max((m.get("max_accel", 0.0) for m in track_metrics.values()), default=0.0)),
        "max_yaw_rate": _json_float_or_none(max((m.get("max_yaw_rate", 0.0) for m in track_metrics.values()), default=0.0)),
        "offroad_ratio": None,
        "bbox_penetration": _json_float_or_none(pair_metrics.get("max_penetration")),
        "visibility_score": None,
        "annotation_consistency": annotation_consistency["ok"],
        "checks": checks,
        "track_metrics": track_metrics,
        "pair_metrics": pair_metrics,
        "thresholds": cfg,
    }
    return _to_jsonable(report)


def _track_motion_metrics(tm, track_id: int, frames: List[int], fps: float) -> Dict[str, Any]:
    samples = []
    for f in frames:
        pose = tm.get_track_pose(track_id, f)
        if pose is None:
            continue
        pose = np.asarray(pose, dtype=np.float32)
        center = pose[:3, 3].astype(np.float32)
        yaw = _yaw_from_pose(pose)
        samples.append((int(f), center, yaw))

    if len(samples) < 2:
        return {
            "num_samples": len(samples),
            "pose_coverage": len(samples) / float(max(1, len(frames))),
            "max_speed": 0.0,
            "max_accel": 0.0,
            "max_jerk": 0.0,
            "max_yaw_rate": 0.0,
            "max_step_distance": 0.0,
            "has_pose_gap": len(samples) != len(frames),
        }

    speeds, yaws, step_distances = [], [], []
    for (f0, c0, y0), (f1, c1, y1) in zip(samples[:-1], samples[1:]):
        dt = max(1, f1 - f0) / float(fps)
        delta = c1 - c0
        delta[1] = 0.0
        dist = float(np.linalg.norm(delta))
        step_distances.append(dist)
        speeds.append(dist / dt)
        yaws.append(abs(_angle_diff(y1, y0)) / dt)

    accels = []
    for i in range(len(speeds) - 1):
        f0 = samples[i][0]
        f1 = samples[i + 2][0]
        dt = max(1, f1 - f0) / float(fps)
        accels.append(abs(speeds[i + 1] - speeds[i]) / dt)

    jerks = []
    for i in range(len(accels) - 1):
        f0 = samples[i][0]
        f1 = samples[i + 3][0]
        dt = max(1, f1 - f0) / float(fps)
        jerks.append(abs(accels[i + 1] - accels[i]) / dt)

    return {
        "num_samples": len(samples),
        "pose_coverage": len(samples) / float(max(1, len(frames))),
        "max_speed": max(speeds) if speeds else 0.0,
        "max_accel": max(accels) if accels else 0.0,
        "max_jerk": max(jerks) if jerks else 0.0,
        "max_yaw_rate": max(yaws) if yaws else 0.0,
        "max_step_distance": max(step_distances) if step_distances else 0.0,
        "has_pose_gap": len(samples) != len(frames),
    }


def _pair_collision_metrics(tm, track_ids: List[int], frames: List[int]) -> Dict[str, Any]:
    if len(track_ids) < 2:
        return {
            "track_pair": [],
            "min_distance": None,
            "min_distance_frame": None,
            "collision_frame": None,
            "max_penetration": 0.0,
        }

    a, b = int(track_ids[0]), int(track_ids[1])
    dims_a = _dims_or_default(tm.get_track_dimensions(a))
    dims_b = _dims_or_default(tm.get_track_dimensions(b))
    min_distance = math.inf
    min_frame = None
    collision_frame = None
    max_penetration = 0.0

    for f in frames:
        pa = tm.get_track_pose(a, f)
        pb = tm.get_track_pose(b, f)
        if pa is None or pb is None:
            continue
        pa = np.asarray(pa, dtype=np.float32)
        pb = np.asarray(pb, dtype=np.float32)
        ca = pa[:3, 3]
        cb = pb[:3, 3]
        d = float(np.linalg.norm(ca - cb))
        if d < min_distance:
            min_distance = d
            min_frame = int(f)
        hit, penetration = check_collision(pa, dims_a, pb, dims_b)
        if hit and collision_frame is None:
            collision_frame = int(f)
        max_penetration = max(max_penetration, float(penetration))

    return {
        "track_pair": [a, b],
        "min_distance": None if math.isinf(min_distance) else min_distance,
        "min_distance_frame": min_frame,
        "collision_frame": collision_frame,
        "max_penetration": max_penetration,
    }


def _annotation_consistency(tm, track_ids: Iterable[int], frames: List[int], cfg: Dict[str, float]) -> Dict[str, Any]:
    missing = {}
    bad_dims = []
    coverage = {}
    for tid in track_ids:
        tid = int(tid)
        dims = tm.get_track_dimensions(tid)
        if not dims or len(dims) < 3 or any(float(abs(x)) <= 1e-4 for x in dims[:3]):
            bad_dims.append(tid)
        count = 0
        missing_frames = []
        for f in frames:
            if tm.get_track_pose(tid, f) is None:
                missing_frames.append(int(f))
            else:
                count += 1
        cov = count / float(max(1, len(frames)))
        coverage[str(tid)] = cov
        if missing_frames:
            missing[str(tid)] = missing_frames[:10]

    ok = not bad_dims and all(v >= cfg["min_pose_coverage"] for v in coverage.values())
    return {
        "ok": ok,
        "coverage": coverage,
        "min_required_coverage": cfg["min_pose_coverage"],
        "tracks_with_bad_dimensions": bad_dims,
        "missing_pose_frames_preview": missing,
    }


def _motion_ok(track_metrics: Dict[str, Dict[str, Any]], cfg: Dict[str, float]) -> bool:
    if not track_metrics:
        return False
    for metrics in track_metrics.values():
        if metrics.get("max_speed", 0.0) > cfg["max_speed_mps"]:
            return False
        if metrics.get("max_accel", 0.0) > cfg["max_accel_mps2"]:
            return False
        if metrics.get("max_jerk", 0.0) > cfg["max_jerk_mps3"]:
            return False
        if metrics.get("max_yaw_rate", 0.0) > cfg["max_yaw_rate_radps"]:
            return False
        if metrics.get("max_step_distance", 0.0) > cfg["max_step_distance_m"]:
            return False
    return True


def _motion_summary(track_metrics: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
    return {
        "max_speed": max((m.get("max_speed", 0.0) for m in track_metrics.values()), default=0.0),
        "max_accel": max((m.get("max_accel", 0.0) for m in track_metrics.values()), default=0.0),
        "max_jerk": max((m.get("max_jerk", 0.0) for m in track_metrics.values()), default=0.0),
        "max_yaw_rate": max((m.get("max_yaw_rate", 0.0) for m in track_metrics.values()), default=0.0),
        "max_step_distance": max((m.get("max_step_distance", 0.0) for m in track_metrics.values()), default=0.0),
    }


def _ttc_ok(ttc: Optional[float], collision_frame: Optional[int], cfg: Dict[str, float]) -> bool:
    if collision_frame is None:
        return True
    if ttc is None:
        return False
    return cfg["ttc_min_s"] <= float(ttc) <= cfg["ttc_max_s"]


def _add_check(checks: List[Dict[str, Any]], name: str, passed: Optional[bool], desc: str, details: Dict[str, Any]) -> None:
    checks.append({
        "name": name,
        "passed": passed,
        "description": desc,
        "details": _to_jsonable(details),
    })


def _yaw_from_pose(pose: np.ndarray) -> float:
    forward = np.asarray(pose[:3, 2], dtype=np.float32)
    return float(math.atan2(forward[0], forward[2]))


def _angle_diff(a: float, b: float) -> float:
    return (a - b + math.pi) % (2.0 * math.pi) - math.pi


def _dims_or_default(dims: Any) -> List[float]:
    if dims and len(dims) >= 3:
        return [float(abs(dims[0])), float(abs(dims[1])), float(abs(dims[2]))]
    return [4.5, 2.0, 1.6]


def _unique_ints(values: Iterable[Any]) -> List[int]:
    out = []
    for value in values or []:
        try:
            iv = int(value)
        except (TypeError, ValueError):
            continue
        if iv not in out:
            out.append(iv)
    return out


def _first_non_none(*values):
    for value in values:
        if value is not None:
            return value
    return None


def _json_int_or_none(value):
    return None if value is None else int(value)


def _json_float_or_none(value):
    if value is None:
        return None
    value = float(value)
    return None if not math.isfinite(value) else value


def _to_jsonable(value):
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return _to_jsonable(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
        return None if not math.isfinite(value) else value
    if isinstance(value, float):
        return None if not math.isfinite(value) else value
    return value
