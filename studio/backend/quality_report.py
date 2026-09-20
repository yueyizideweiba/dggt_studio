from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from dggt.scene_edit.collision_physics import check_collision


DEFAULT_THRESHOLDS = {
    "ttc_min_s": 0.1,
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

# 这些场景的本质是"目标突然出现"，反应窗口本就极小 → 不做通用 TTC 区间要求
SUDDEN_APPEARANCE_SCENARIOS = {"pedestrian-crossing", "cut-out-reveal"}
# 这些场景本身不产生"碰撞事件"（急刹是单车风险；前车闪开是露出障碍，不撞）
NO_COLLISION_EVENT_SCENARIOS = {"hard-brake", "cut-out-reveal"}


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

    collision_analysis = corner_case_result.get("collision_analysis") or {}
    pair_metrics = _pair_collision_metrics(tm, collision_tracks, frames)
    collision_frame = _first_non_none(
        corner_case_result.get("collision_frame"),
        collision_analysis.get("collision_frame"),
        pair_metrics.get("collision_frame"),
    )
    # 碰撞/最近距离附近的帧是"撞击瞬间"：允许出现较大的瞬时减速，不纳入机动性阈值
    impact_frames = set()
    if collision_frame is not None:
        impact_frames |= {int(collision_frame) - 1, int(collision_frame), int(collision_frame) + 1}
    mdf = pair_metrics.get("min_distance_frame")
    if mdf is not None:
        impact_frames |= {int(mdf) - 1, int(mdf), int(mdf) + 1}
    # 连环事故会有多次碰撞（rear→middle、middle→lead），这里把所有"受影响物体两两之间"
    # 的碰撞帧都豁免掉（物体数 ≤3，代价可忽略）。
    try:
        dims_map = {int(t): _dims_or_default(tm.get_track_dimensions(t)) for t in affected_tracks}
        for ii in range(len(affected_tracks)):
            for jj in range(ii + 1, len(affected_tracks)):
                ta, tb = int(affected_tracks[ii]), int(affected_tracks[jj])
                for f in frames:
                    pa = tm.get_track_pose(ta, f)
                    pb = tm.get_track_pose(tb, f)
                    if pa is None or pb is None:
                        continue
                    hit, _pen = check_collision(np.asarray(pa, dtype=np.float32), dims_map[ta],
                                                np.asarray(pb, dtype=np.float32), dims_map[tb])
                    if hit:
                        impact_frames |= {int(f) - 1, int(f), int(f) + 1}
                        break
    except Exception:  # noqa: BLE001
        pass
    # 接触距离豁免：两个物体中心距低于"OBB 沿连线半宽之和 + 0.6m"时，说明已处于接触/挤压状态，
    # 这一帧的剧烈减速是撞击本身，不该按"机动加速度"去卡（连环事故常有多次碰撞）。
    try:
        for ii in range(len(affected_tracks)):
            for jj in range(ii + 1, len(affected_tracks)):
                ta, tb = int(affected_tracks[ii]), int(affected_tracks[jj])
                da, db = dims_map[ta], dims_map[tb]
                ra = 0.5 * math.sqrt(da[0] * da[0] + da[2] * da[2])
                rb = 0.5 * math.sqrt(db[0] * db[0] + db[2] * db[2])
                for f in frames:
                    pa = tm.get_track_pose(ta, f)
                    pb = tm.get_track_pose(tb, f)
                    if pa is None or pb is None:
                        continue
                    ca = np.asarray(pa, dtype=np.float32)[:3, 3]
                    cb = np.asarray(pb, dtype=np.float32)[:3, 3]
                    if float(np.linalg.norm(ca - cb)) < (ra + rb + 0.6):
                        impact_frames |= {int(f) - 1, int(f), int(f) + 1}
    except Exception:  # noqa: BLE001
        pass

    track_metrics = {
        str(tid): _track_motion_metrics(tm, tid, frames, fps, impact_frames=impact_frames)
        for tid in affected_tracks
    }
    critical_frame = _first_non_none(
        corner_case_result.get("critical_frame"),
        collision_analysis.get("critical_frame"),
    )

    ttc_at_critical = None
    if critical_frame is not None and collision_frame is not None:
        ttc_at_critical = max(0.0, (int(collision_frame) - int(critical_frame)) / float(fps))
    elif collision_analysis.get("time_to_collision") is not None:
        ttc_at_critical = float(collision_analysis["time_to_collision"])

    min_dist = pair_metrics.get("min_distance")
    has_pair = len(collision_tracks) >= 2
    scene_type = corner_case_result.get("scenario_type")
    if scene_type in NO_COLLISION_EVENT_SCENARIOS:
        event_present = None      # 该类场景按定义没有碰撞事件，不适用
    elif has_pair:
        event_present = bool(collision_frame is not None
                             or (min_dist is not None and min_dist <= cfg["near_miss_distance_m"]))
    else:
        # 单物体场景（如紧急刹车）本身没有"碰撞事件"，该项不适用
        event_present = None
    annotation_consistency = _annotation_consistency(tm, affected_tracks, frames, cfg)

    checks = []
    _add_check(checks, "event_present", event_present,
               "collision or near-miss exists",
               {"collision_frame": collision_frame, "min_distance": pair_metrics.get("min_distance")})
    # 行人横穿/前车闪开这类事故的本质就是"突然出现"，反应窗口本就极小，
    # 用通用 TTC 区间门去要求它反而不合理 → 判为"不适用"。
    ttc_ok = _ttc_ok(ttc_at_critical, collision_frame, cfg)
    if corner_case_result.get("scenario_type") in SUDDEN_APPEARANCE_SCENARIOS and collision_frame is not None:
        ttc_ok = None
    _add_check(checks, "ttc_range", ttc_ok,
               "TTC at critical frame is within the configured range, or this is a near-miss",
               {"ttc_at_critical": ttc_at_critical, "range": [cfg["ttc_min_s"], cfg["ttc_max_s"]]})
    _add_check(checks, "motion_physical", _motion_ok(track_metrics, cfg),
               "speed, acceleration, jerk, yaw-rate, and step distance are below thresholds",
               _motion_summary(track_metrics))
    # 碰撞双方里含小体积物体（行人等）→ 穿透门不适用（撞到人本来就会互穿）
    small_body = any(_is_small_body(tm, t) for t in collision_tracks) if has_pair else False
    penetration_ok = None if small_body else (
        pair_metrics.get("max_penetration", 0.0) <= cfg["max_bbox_penetration_m"])
    _add_check(checks, "bbox_penetration", penetration_ok,
               "OBB penetration depth is not excessive (N/A when a pedestrian-class body is involved)",
               {"max_penetration": pair_metrics.get("max_penetration"),
                "threshold": cfg["max_bbox_penetration_m"],
                "small_body_involved": bool(small_body)})
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


def _track_motion_metrics(tm, track_id: int, frames: List[int], fps: float,
                          impact_frames: Optional[set] = None) -> Dict[str, Any]:
    exempt = impact_frames or set()
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
        if samples[i + 1][0] in exempt:     # 撞击瞬间不查机动加速度
            continue
        f0 = samples[i][0]
        f1 = samples[i + 2][0]
        dt = max(1, f1 - f0) / float(fps)
        accels.append(abs(speeds[i + 1] - speeds[i]) / dt)

    jerks = []
    for i in range(len(accels) - 1):
        if samples[i + 2][0] in exempt:
            continue
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


def _is_small_body(tm, track_id) -> bool:
    """是否"小体积参与者"（行人/自行车/摩托等）。

    这类物体被车撞到时本来就会与车体互穿——那就是事故本身，用车辆之间
    "不许穿模"的穿透门去卡它没有意义（会把所有行人事故判成不通过）。
    """
    try:
        d = _dims_or_default(tm.get_track_dimensions(track_id))
        return float(abs(d[0]) * abs(d[1]) * abs(d[2])) < 2.0
    except Exception:  # noqa: BLE001
        return False


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
