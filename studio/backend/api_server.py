import os
import sys

# gsplat 首次渲染会 JIT 编译 CUDA kernel：需要 PATH 上能找到 ninja 和 nvcc。
# 用 `setsid ... /abs/path/python api_server.py` 这种后台方式启动时不会带上 conda env / cuda 的 bin，
# 会报 "Ninja is required to load C++ extensions" —— 这里统一补上（用户自己激活环境跑也没影响）。
_EXTRA_PATH = [os.path.dirname(sys.executable), "/usr/local/cuda/bin",
               os.path.join(os.environ.get("CUDA_HOME", ""), "bin")]
os.environ["PATH"] = os.pathsep.join([p for p in _EXTRA_PATH if p and os.path.isdir(p)]
                                     + [os.environ.get("PATH", "")])
import json
import math
import subprocess
import asyncio
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path
import base64
import io

import numpy as np
import cv2
import torch
# gsplat 首次渲染会 JIT 编译 CUDA kernel；只编当前 GPU 架构，避免"编译所有架构"长时间卡死/占满内存
if torch.cuda.is_available():
    _cap = torch.cuda.get_device_capability()
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", f"{_cap[0]}.{_cap[1]}")
from scipy.spatial.transform import Rotation as R
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, BackgroundTasks, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from dggt_engine import DGGTRenderer, TrajectoryController, CornerCaseGenerator

sys.path.insert(0, str(Path(__file__).parent))
from track_manager import TrackManager
import corner_case
import quality_report
import sam3d_client
import sam_segment
import scene_graph
import trust_runtime
import diffusion_refine
import sim_runtime
import sensor_rig
import scenario_export

# 创建FastAPI应用
app = FastAPI(title="DGGT Studio API V2", version="2.0.0")

# CORS配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局状态
studio_state = {
    "scenes": {},  # 场景ID -> DGGTRenderer实例
    "track_managers": {},  # 场景ID -> TrackManager实例
    "active_scene_id": None,
    "render_cache": {},  # 渲染缓存
    "edit_history": {},  # 编辑历史
    "preview_tasks": {}  # 预览任务
}


# ==================== 数据模型 ====================

class SceneLoadRequest(BaseModel):
    scene_path: str
    scene_id: Optional[str] = None
    load_sky: bool = True
    static_only: bool = False


class ObjectPose(BaseModel):
    frame_idx: int
    object_id: int
    pose_matrix: List[List[float]]


class TrajectoryKeyframe(BaseModel):
    frame_idx: int
    pose_world: List[List[float]]


class TrajectoryEdit(BaseModel):
    object_id: int
    keyframes: List[TrajectoryKeyframe]


class ObjectDragRequest(BaseModel):
    object_id: int
    frame_idx: int
    delta_x: float
    delta_y: float
    delta_z: float = 0.0


class RenderRequest(BaseModel):
    scene_id: str
    frame_idx: int
    width: Optional[int] = 800
    height: Optional[int] = 600
    draw_bboxes: bool = True
    draw_ids: bool = True
    draw_trajectories: bool = False
    trajectory_length: int = 30  # 显示前后多少帧的轨迹
    auto_extend: bool = True      # 请求的帧超出原场景帧数时，按需延长时间轴


class TimelineExtendRequest(BaseModel):
    """按需延长可渲染帧数：静态场景与帧无关，动态物体复用末次外观 + 轨迹给位姿，
    相机按自车轨迹外推——所以"原场景多少帧"不再是上限。"""
    scene_id: str
    extra_frames: Optional[int] = None    # 增量（如 +100）
    total_frames: Optional[int] = None    # 目标总数（优先于增量）
    mode: str = "extrapolate"             # extrapolate=沿各自轨迹惯性外推；hold=冻结在末帧
    ego_mode: Optional[str] = None        # 自车单独指定（默认同 mode）


class SequenceRenderRequest(BaseModel):
    scene_id: str
    start_frame: int
    num_frames: int
    draw_bboxes: bool = True
    draw_ids: bool = True
    draw_trajectories: bool = False
    fps: int = 10


class CornerCaseRequest(BaseModel):
    scene_id: str
    scenario_type: str
    selected_objects: List[int]
    start_frame: int
    num_frames: int
    metadata: Optional[Dict[str, Any]] = None


class ExportRequest(BaseModel):
    scene_id: str
    output_path: str
    format: str = "json"  # json, ply, video


# ==================== 辅助函数 ====================

def get_scene_or_404(scene_id: str) -> DGGTRenderer:
    """获取场景实例或返回404错误"""
    if scene_id not in studio_state["scenes"]:
        raise HTTPException(status_code=404, detail=f"Scene {scene_id} not found")
    return studio_state["scenes"][scene_id]


def _ensure_timeline(tm, frame_idx=None, num_frames=None):
    """按需把时间轴延长到覆盖请求的帧范围（超出原场景帧数就自动外推）。

    返回 None（无需延长）或延长结果 dict。只在真正超范围时才动，不改变已有行为。
    """
    if tm is None:
        return None
    try:
        return tm.ensure_frames(frame_idx=frame_idx, num_frames=num_frames)
    except Exception as e:  # noqa: BLE001
        print(f"[timeline] 按需延长失败（继续按原帧数渲染）: {e}")
        return None


def _timeline_info(renderer, tm=None):
    """时间轴信息：数据帧（磁盘上真有相机/动态 PLY 的帧）与可渲染帧（可延长）。"""
    try:
        data = int(renderer.data_frame_count())
    except Exception:  # noqa: BLE001
        data = int(renderer.frame_count())
    total = int(getattr(renderer, "total_frames", data))
    if tm is not None:
        total = int(getattr(tm, "num_frames", total))
    return {"data_frames": data, "total_frames": total,
            "extended_frames": max(0, total - data),
            "extended_from_frame": getattr(tm, "extended_from_frame", None) if tm is not None else None}


def get_track_manager_or_404(scene_id: str) -> TrackManager:
    """获取场景的 TrackManager（按需构建）。"""
    if scene_id not in studio_state["scenes"]:
        raise HTTPException(status_code=404, detail=f"Scene {scene_id} not found")
    tm = studio_state["track_managers"].get(scene_id)
    if tm is None:
        tm = TrackManager(studio_state["scenes"][scene_id])
        studio_state["track_managers"][scene_id] = tm
    return tm


def project_point_to_image(point_3d: np.ndarray, viewmat: torch.Tensor, K: torch.Tensor, device: str = "cuda") -> Optional[tuple]:
    """将3D点投影到2D图像坐标"""
    try:
        point_cam = viewmat @ torch.tensor([*point_3d, 1.0], device=device, dtype=torch.float32)
        xyz = point_cam[:3]
        
        if xyz[2] < 0.1:  # 在相机后面
            return None
        
        uv_w = K @ xyz
        u = uv_w[0].item() / (uv_w[2].item() + 1e-6)
        v = uv_w[1].item() / (uv_w[2].item() + 1e-6)
        
        return (u, v)
    except:
        return None


def draw_trajectory_on_image(
    image: np.ndarray,
    trajectory_points: List[np.ndarray],
    viewmat: torch.Tensor,
    K: torch.Tensor,
    color: tuple = (255, 100, 100),
    thickness: int = 2,
    device: str = "cuda"
) -> np.ndarray:
    """在图像上绘制轨迹曲线"""
    if len(trajectory_points) < 2:
        return image
    
    # 投影所有点到2D
    points_2d = []
    for point_3d in trajectory_points:
        uv = project_point_to_image(point_3d, viewmat, K, device)
        if uv is not None:
            points_2d.append(uv)
    
    if len(points_2d) < 2:
        return image
    
    # 绘制轨迹曲线
    pts = np.array(points_2d, dtype=np.int32)
    
    # 绘制平滑曲线
    for i in range(len(pts) - 1):
        # 渐变颜色（越旧越淡）
        alpha = (i + 1) / len(pts)
        current_color = tuple(int(c * alpha) for c in color)
        
        cv2.line(image, tuple(pts[i]), tuple(pts[i+1]), current_color, thickness, cv2.LINE_AA)
    
    # 绘制轨迹点
    for i, pt in enumerate(pts):
        alpha = (i + 1) / len(pts)
        radius = max(2, int(4 * alpha))
        cv2.circle(image, tuple(pt), radius, color, -1, cv2.LINE_AA)
    
    return image


def draw_all_trajectories(
    image: np.ndarray,
    renderer: DGGTRenderer,
    frame_idx: int,
    ego_data: dict,
    trajectory_length: int = 30
) -> np.ndarray:
    """绘制所有物体的轨迹"""
    if not ego_data:
        return image
    
    # 准备相机参数
    c2w = torch.tensor(ego_data["camera_extrinsics_world"], device=renderer.device).float()
    if c2w.shape == (3, 4):
        c2w = torch.cat([c2w, torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=renderer.device, dtype=c2w.dtype)], dim=0)
    viewmat = torch.inverse(c2w)
    K = torch.tensor(ego_data["camera_intrinsics"], device=renderer.device).float()
    
    # 颜色映射
    colors = [
        (255, 100, 100),  # 红
        (100, 255, 100),  # 绿
        (100, 100, 255),  # 蓝
        (255, 255, 100),  # 黄
        (255, 100, 255),  # 品红
        (100, 255, 255),  # 青
    ]
    
    # 收集所有物体ID
    object_ids = set()
    for offset in range(-trajectory_length, trajectory_length + 1):
        check_frame = frame_idx + offset
        if check_frame < 0:
            continue
        
        try:
            _cf = renderer.flat_index(int(check_frame), 0) if check_frame >= 0 else int(check_frame)
        except Exception:  # noqa: BLE001
            _cf = int(check_frame)
        obj_meta_path = os.path.join(renderer.meta_dir, f"frame_{_cf:04d}_objects.json")
        if os.path.exists(obj_meta_path):
            with open(obj_meta_path, "r") as f:
                objects = json.load(f)
                for obj in objects:
                    object_ids.add(obj["object_id"])
    
    # 为每个物体绘制轨迹
    for obj_idx, obj_id in enumerate(sorted(object_ids)):
        trajectory_points = []
        color = colors[obj_idx % len(colors)]
        
        for offset in range(-trajectory_length, trajectory_length + 1):
            check_frame = frame_idx + offset
            if check_frame < 0:
                continue
            
            # 获取物体位姿（优先使用编辑后的）
            pose = renderer.get_real_frame_object_pose(check_frame, obj_id)
            
            if pose is None:
                # 从原始数据加载
                try:
                    _cf = renderer.flat_index(int(check_frame), 0)
                except Exception:  # noqa: BLE001
                    _cf = int(check_frame)
                obj_meta_path = os.path.join(renderer.meta_dir, f"frame_{_cf:04d}_objects.json")
                if os.path.exists(obj_meta_path):
                    with open(obj_meta_path, "r") as f:
                        objects = json.load(f)
                        for obj in objects:
                            if obj["object_id"] == obj_id:
                                pose = np.array(obj["pose_world"])
                                break
            
            if pose is not None:
                # 提取位置（平移部分）
                if isinstance(pose, torch.Tensor):
                    center = pose[:3, 3].cpu().numpy()
                else:
                    center = np.array(pose)[:3, 3]
                trajectory_points.append(center)
        
        if len(trajectory_points) > 1:
            image = draw_trajectory_on_image(image, trajectory_points, viewmat, K, color, device=renderer.device)
    
    return image


def _max_metric(report: Dict[str, Any], key: str):
    """从 quality_report 的 track_metrics 里取某个运动学指标的最大值。"""
    tm_metrics = (report or {}).get("track_metrics") or {}
    vals = [m.get(key) for m in tm_metrics.values() if isinstance(m, dict) and isinstance(m.get(key), (int, float))]
    return max(vals) if vals else None


def _failing_metrics(report: Dict[str, Any]) -> Dict[str, Any]:
    """把"未通过的质量门"翻译成"哪个指标超了多少"，方便直接定位问题。"""
    cfg = (report or {}).get("thresholds") or {}
    out: Dict[str, Any] = {}
    for c in (report or {}).get("checks") or []:
        if c.get("passed") is not False:
            continue
        name = c.get("name")
        d = c.get("details") or {}
        if name == "motion_physical":
            for key, thr_key in (("max_speed", "max_speed_mps"), ("max_accel", "max_accel_mps2"),
                                 ("max_jerk", "max_jerk_mps3"), ("max_yaw_rate", "max_yaw_rate_radps"),
                                 ("max_step_distance", "max_step_distance_m")):
                v, t = d.get(key), cfg.get(thr_key)
                if isinstance(v, (int, float)) and isinstance(t, (int, float)) and v > t:
                    out[key] = {"value": float(v), "threshold": float(t)}
        elif name == "bbox_penetration":
            out["bbox_penetration"] = {"value": d.get("max_penetration"), "threshold": d.get("threshold")}
        elif name == "event_present":
            out["event_present"] = {"min_distance": d.get("min_distance"), "collision_frame": d.get("collision_frame")}
        elif name == "ttc_range":
            out["ttc_range"] = d
        elif name == "annotation_consistency":
            out["annotation_consistency"] = {"coverage": d.get("coverage"),
                                             "tracks_with_bad_dimensions": d.get("tracks_with_bad_dimensions")}
    return out


def render_ego_frame(renderer, tm, frame_idx: int, draw_bboxes: bool = True,
                     draw_ids: bool = True, draw_trajectories: bool = False,
                     highlight_track_id=None, camera_override=None):
    """渲染一帧"自车视角"图像（含 TrackManager 的编辑/合成物体 + 可选叠加），返回 RGB ndarray。

    与 `/api/render/frame` 走**同一条渲染路径**（同样的 object_overrides / extra_objects /
    包围盒 / 轨迹绘制），因此逐帧渲染出来的视频与前端 2D 播放看到的画面一致。
    """
    if tm is not None:
        object_overrides = tm.build_object_overrides(frame_idx)
        # 自车视角下不渲染"主车实体"（否则车体会糊在镜头上）
        extra_objects = tm.build_extra_objects(frame_idx, include_ego=False)
    else:
        object_overrides = {}
        extra_objects = []
        for oid in renderer.trajectory_controller.get_object_ids():
            pose = renderer.get_real_frame_object_pose(frame_idx, oid)
            if pose is not None:
                object_overrides[oid] = pose

    cam_kw = {}
    if camera_override is not None:
        c2w_ov = np.asarray(camera_override, dtype=np.float32)
        if c2w_ov.shape == (3, 4):
            c2w_ov = np.vstack([c2w_ov, np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32)])
        cam_kw["c2w_override"] = torch.tensor(c2w_ov, device=renderer.device, dtype=torch.float32)

    image = renderer._render_frame_with_object_overrides(
        frame_idx, object_overrides, extra_objects=extra_objects, **cam_kw)

    if tm is None or not (draw_bboxes or draw_ids or draw_trajectories):
        return image

    cam_flat = renderer.flat_index(frame_idx, 0)
    ego_path = os.path.join(renderer.ego_dir, f"frame_{cam_flat:04d}_ego.json")
    if not os.path.exists(ego_path):
        return image
    with open(ego_path, "r") as f:
        ego_data = json.load(f)
    if camera_override is not None:
        c2w = torch.tensor(np.asarray(camera_override, dtype=np.float32), device=renderer.device).float()
    else:
        c2w = torch.tensor(ego_data["camera_extrinsics_world"], device=renderer.device).float()
    if c2w.shape == (3, 4):
        c2w = torch.cat([c2w, torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=renderer.device, dtype=c2w.dtype)], dim=0)
    viewmat = torch.inverse(c2w)
    K = torch.tensor(ego_data["camera_intrinsics"], device=renderer.device).float()
    W, H = ego_data["camera"]["width"], ego_data["camera"]["height"]

    objects_data = [o for o in tm.get_frame_objects(frame_idx) if not o.get("ego")]
    if draw_bboxes or draw_ids:
        for obj in objects_data:
            tid = obj["track_id"]
            pose = torch.tensor(obj["pose_world"], device=renderer.device).float()
            corners_2d = renderer._get_bbox_corners_2d(pose, obj["dimensions"], viewmat, K, W, H)
            if corners_2d is None:
                continue
            if highlight_track_id is not None and tid == highlight_track_id:
                color = (0, 200, 255)
            else:
                color = (0, 165, 255) if obj.get("edited") else (0, 255, 0)
            label = f"T:{tid}" if draw_ids else None
            image = renderer._draw_bbox(image, corners_2d, color=color, label=label)

    if draw_trajectories:
        for obj in objects_data:
            traj = tm.get_track_trajectory(obj["track_id"], use_edits=True)
            pts = [np.asarray(p["center"], dtype=np.float32) for p in traj]
            if len(pts) >= 2:
                image = draw_trajectory_on_image(image, pts, viewmat, K, device=renderer.device)
    return image


def render_sequence_video(renderer, tm, start_frame: int, num_frames: int, fps: float,
                          out_path: str, draw_bboxes: bool = True, draw_ids: bool = True,
                          draw_trajectories: bool = False, highlight_track_id=None,
                          use_edited_ego: bool = False):
    """把 [start_frame, start_frame+num_frames) 按自车视角逐帧渲染并编码为可播放的 mp4。

    优先 H.264(libx264)（浏览器直接可播），失败退回 mp4v。
    """
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    frames = []
    follow_ego = bool(use_edited_ego and tm is not None
                      and getattr(tm, "ego_track_id", None) is not None
                      and (tm.ego_is_edited()
                           or getattr(tm, "ego_source_track", None) is not None))
    for i in range(int(max(1, num_frames))):
        try:
            f_i = int(start_frame) + i
            cam_ov = tm.get_ego_camera(f_i) if follow_ego else None
            frames.append(render_ego_frame(renderer, tm, f_i,
                                           draw_bboxes, draw_ids, draw_trajectories,
                                           highlight_track_id, camera_override=cam_ov))
        except Exception as e:  # noqa: BLE001
            # 该帧不可渲染（如超出场景帧范围）→ 跳过，保证仍能出视频
            print(f"[video] 跳过帧 {int(start_frame) + i}: {e}")
            continue
    if not frames:
        return None
    h, w = frames[0].shape[:2]
    try:
        import imageio
        with imageio.get_writer(out_path, fps=float(fps), codec="libx264", format="FFMPEG") as writer:
            for fr in frames:
                writer.append_data(fr)
    except Exception:  # noqa: BLE001
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        vw = cv2.VideoWriter(out_path, fourcc, float(fps), (w, h))
        for fr in frames:
            vw.write(cv2.cvtColor(fr, cv2.COLOR_RGB2BGR))
        vw.release()
    return out_path


BEV_PALETTE = [(230, 140, 70), (110, 200, 90), (70, 190, 235), (200, 110, 230),
               (150, 110, 240), (190, 190, 80), (80, 150, 220), (170, 170, 170)]
BEV_COLLISION_COLOR = (70, 70, 235)   # BGR 红
BEV_EGO_COLOR = (0, 215, 255)         # 主车：亮黄（与自车箭头同色）


def _bev_world_bounds(tm, start_frame: int, num_frames: int, margin: float = 8.0):
    """取整段窗口内所有物体的世界 XZ 范围（视频全程固定，避免画面抖动）。"""
    xs, zs = [], []
    for f in range(int(start_frame), int(start_frame) + int(num_frames)):
        try:
            objs = tm.get_frame_objects(f)
        except Exception:  # noqa: BLE001
            continue
        for o in objs:
            try:
                c = np.asarray(o["pose_world"], dtype=np.float32)[:3, 3]
            except Exception:  # noqa: BLE001
                continue
            xs.append(float(c[0]))
            zs.append(float(c[2]))
    if not xs:
        return (-20.0, 20.0, -20.0, 20.0)
    # 少量远处离群轨迹会把视角拉得过远：用 2%~98% 分位裁剪，保证主体（自车+参与者）看得清
    if len(xs) >= 8:
        lo_x, hi_x = np.percentile(xs, [2.0, 98.0])
        lo_z, hi_z = np.percentile(zs, [2.0, 98.0])
        xs = list(xs) + [float(lo_x), float(hi_x)]
        zs = list(zs) + [float(lo_z), float(hi_z)]
    return (min(xs) - margin, max(xs) + margin, min(zs) - margin, max(zs) + margin)


def render_bev_frame(tm, renderer, frame_idx: int, bounds, size: int = 560,
                     highlight_tracks=None, collision_frame=None, draw_ids: bool = False):
    """渲染一帧俯视（BEV）示意图：物体=填充矩形（按朝向），历史轨迹=折线，自车=黄色箭头。

    这是"上帝视角"，用于补充主车视角看不到的场景关系（谁在谁前面、从哪来）。
    """
    W = H = int(size)
    PAD = 28
    img = np.full((H, W, 3), 16, dtype=np.uint8)
    min_x, max_x, min_z, max_z = bounds
    span_x = max(1e-6, max_x - min_x)
    span_z = max(1e-6, max_z - min_z)
    s = min((W - 2 * PAD) / span_x, (H - 2 * PAD) / span_z)
    off_x = 0.5 * (W - span_x * s)      # 居中：世界范围等比缩放后居中，不留偏心空白
    off_y = 0.5 * (H - span_z * s)

    def to_px(x, z):
        return (off_x + (x - min_x) * s, H - off_y - (z - min_z) * s)

    # 10m 网格（铺满整张画布，避免出现"画布边界"）
    step = 10.0
    vis_x0 = min_x - off_x / s
    vis_x1 = vis_x0 + W / s
    vis_z1 = max_z + off_y / s
    vis_z0 = vis_z1 - H / s
    gx = math.floor(vis_x0 / step) * step
    while gx <= vis_x1:
        p1, p2 = to_px(gx, vis_z0), to_px(gx, vis_z1)
        cv2.line(img, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), (34, 42, 52), 1)
        gx += step
    gz = math.floor(vis_z0 / step) * step
    while gz <= vis_z1:
        p1, p2 = to_px(vis_x0, gz), to_px(vis_x1, gz)
        cv2.line(img, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), (34, 42, 52), 1)
        gz += step

    try:
        objs = tm.get_frame_objects(frame_idx)
    except Exception:  # noqa: BLE001
        objs = []
    highlight = {int(t) for t in (highlight_tracks or [])}

    # 先画轨迹（在物体下方）
    for o in objs:
        tid = int(o["track_id"])
        if o.get("ego") or o.get("viewer"):
            col = BEV_EGO_COLOR
        else:
            col = BEV_COLLISION_COLOR if tid in highlight else BEV_PALETTE[tid % len(BEV_PALETTE)]
        try:
            traj = tm.get_track_trajectory(tid, use_edits=True)
        except Exception:  # noqa: BLE001
            continue
        pts = [to_px(float(p["center"][0]), float(p["center"][2]))
               for p in traj if int(p["frame_idx"]) <= int(frame_idx)]
        for a, b in zip(pts[:-1], pts[1:]):
            cv2.line(img, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])), col, 1, cv2.LINE_AA)

    # 再画物体（填充多边形）+ 速度箭头
    for o in objs:
        tid = int(o["track_id"])
        if o.get("ego") or o.get("viewer"):
            col = BEV_EGO_COLOR
        else:
            col = BEV_COLLISION_COLOR if tid in highlight else BEV_PALETTE[tid % len(BEV_PALETTE)]
        try:
            pose = np.asarray(o["pose_world"], dtype=np.float32)
            c = pose[:3, 3]
            R = pose[:3, :3]
            dims = o.get("dimensions") or [2.0, 1.6, 4.5]
            w, l = abs(float(dims[0])), abs(float(dims[2]))
        except Exception:  # noqa: BLE001
            continue
        yaw = math.atan2(float(R[0, 2]), float(R[2, 2]))
        cy, sy = math.cos(yaw), math.sin(yaw)
        corners = []
        for sx, sz in ((-w / 2, -l / 2), (w / 2, -l / 2), (w / 2, l / 2), (-w / 2, l / 2)):
            wx = c[0] + sx * cy + sz * sy
            wz = c[2] - sx * sy + sz * cy
            corners.append(to_px(wx, wz))
        poly = np.array([[int(x), int(y)] for x, y in corners], dtype=np.int32)
        cv2.fillPoly(img, [poly], col)
        cv2.polylines(img, [poly], True, (238, 242, 248), 1, cv2.LINE_AA)
        if draw_ids:
            p = to_px(c[0], c[2])
            cv2.putText(img, f"T{tid}", (int(p[0]) + 7, int(p[1]) - 7),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (232, 237, 243), 1, cv2.LINE_AA)

    # 自车（相机）位置 + 朝向：主车已实体化 / 已指定其它视角来源时，用该物体的足迹表示
    has_ego_obj = any(o.get("ego") or o.get("viewer") for o in objs)
    try:
        cam = _frame_camera(renderer, frame_idx) if not has_ego_obj else None
        if cam:
            c2w = cam[0]
            ex, ez = float(c2w[0, 3]), float(c2w[2, 3])
            fx, fz = float(c2w[0, 2]), float(c2w[2, 2])
            p = to_px(ex, ez)
            q = to_px(ex + fx * 4.0, ez + fz * 4.0)
            cv2.arrowedLine(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])),
                            (0, 215, 255), 2, cv2.LINE_AA, tipLength=0.45)
            cv2.circle(img, (int(p[0]), int(p[1])), 3, (0, 215, 255), -1)
    except Exception:  # noqa: BLE001
        pass

    # 比例尺（10 m）+ 上方朝向提示
    try:
        import math as _m
        bar = 10.0 * s
        if 24 <= bar <= W - 40:
            x0, y0 = 14, H - 14
            cv2.line(img, (int(x0), int(y0)), (int(x0 + bar), int(y0)), (170, 180, 190), 2, cv2.LINE_AA)
            cv2.putText(img, "10 m", (int(x0), int(y0) - 6), cv2.FONT_HERSHEY_SIMPLEX,
                        0.42, (170, 180, 190), 1, cv2.LINE_AA)
    except Exception:  # noqa: BLE001
        pass
    cv2.putText(img, f"BEV  f{int(frame_idx)}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (200, 208, 216), 1, cv2.LINE_AA)
    if collision_frame is not None and abs(int(frame_idx) - int(collision_frame)) <= 1:
        cv2.putText(img, "COLLISION", (W - 118, 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (80, 80, 245), 2, cv2.LINE_AA)
    return img


def _project_points(pts_world, c2w, K, W, H):
    """把世界点投影到像素（返回 Nx2 像素 + 是否全部在相机前方）。"""
    pts = np.asarray(pts_world, dtype=np.float64)
    R = np.asarray(c2w, dtype=np.float64)[:3, :3]
    t = np.asarray(c2w, dtype=np.float64)[:3, 3]
    cam = (pts - t) @ R            # R^T (p - t)
    z = cam[:, 2]
    if np.any(z <= 0.05):
        return None, False
    K = np.asarray(K, dtype=np.float64)
    u = K[0, 0] * cam[:, 0] / z + K[0, 2]
    v = K[1, 1] * cam[:, 1] / z + K[1, 2]
    return np.stack([u, v], axis=1), True


def _object_top_face_corners(pose, dims):
    """物体包围盒**顶面**的 4 个角点（世界坐标）。"""
    R = np.asarray(pose, dtype=np.float64)[:3, :3]
    c = np.asarray(pose, dtype=np.float64)[:3, 3]
    hw, hh, hl = (abs(float(dims[0])) / 2.0, abs(float(dims[1])) / 2.0, abs(float(dims[2])) / 2.0)
    top = np.array([[hw, hh, hl], [-hw, hh, hl], [-hw, hh, -hl], [hw, hh, -hl]], dtype=np.float64)
    return (R @ top.T).T + c


# 俯视视频里给动态物体用的"实体替身"配色（BGR）：常规车=深灰车身+浅灰车顶，主车=亮黄
TOP_PROXY_SIDE = (46, 48, 52)
TOP_PROXY_BODY = (96, 100, 108)
TOP_PROXY_EDGE = (225, 230, 236)
TOP_PROXY_EGO = (0, 205, 245)
TOP_PROXY_EGO_EDGE = (255, 255, 255)


def _sane_dims(dims):
    """把包围盒尺寸夹到"车/人"的合理范围；明显异常（0 尺寸/巨大）返回 None 不画。"""
    try:
        d = [abs(float(v)) for v in dims[:3]]
    except Exception:  # noqa: BLE001
        return None
    if len(d) < 3:
        return None
    if not (0.15 <= d[0] <= 14.0 and 0.15 <= d[1] <= 5.0 and 0.15 <= d[2] <= 14.0):
        return None
    return d


def _draw_topdown_object_proxies(image, tm, frame_idx: int, cam, highlight_ego_track=None,
                                 skip_raw_ids=None):
    """在俯视真渲染画面上，为**每个动态物体**叠加一块"实体"（包围盒顶面 + 描边）。

    为什么需要：单目 4DGS 重建出的动态物体高斯是"贴片式"的（只在源相机视角附近有形），
    从正上方看会退化成细长条纹甚至消失；而静态道路/背景从上方渲染完全正常。
    所以俯视视频 = 真实渲染的静态场景 + 各动态物体的实体替身（位置/朝向/尺寸都来自真实位姿），
    这样"所有动态物体"在俯视视频里都看得见、位置也准确。
    """
    if tm is None or image is None:
        return image
    c2w, K, W, H = cam
    img = np.ascontiguousarray(image)
    try:
        objs = tm.get_frame_objects(frame_idx)
    except Exception:  # noqa: BLE001
        return img
    for o in objs:
        try:
            if o.get("ego") or o.get("synthetic"):
                continue          # 主车/合成物体用真模型渲染，不需要替身
            if skip_raw_ids and o.get("raw_object_id") is not None \
                    and int(o["raw_object_id"]) in skip_raw_ids:
                continue          # 已经有 actor 资产（真模型）渲染过
            dims = _sane_dims(o.get("dimensions") or [])
            if dims is None:
                continue
            pose = np.asarray(o["pose_world"], dtype=np.float64)
            is_ego = (highlight_ego_track is not None
                      and int(o["track_id"]) == int(highlight_ego_track))
            R = pose[:3, :3]
            c = pose[:3, 3]
            hw, hh, hl = dims[0] / 2.0, dims[1] / 2.0, dims[2] / 2.0
            # 8 个角点 → 先用凸包填"侧影"（深色），再填顶面（亮色），像一辆有体积的车
            local = np.array([[sx * hw, sy * hh, sz * hl]
                              for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], dtype=np.float64)
            corners = (R @ local.T).T + c
            pts, ok = _project_points(corners, c2w, K, W, H)
            if not ok or pts is None:
                continue
            if np.abs(pts).max() > 40000:
                continue
            hull = cv2.convexHull(np.round(pts).astype(np.int32))
            cv2.fillPoly(img, [hull], TOP_PROXY_EGO if is_ego else TOP_PROXY_SIDE, cv2.LINE_AA)
            top, ok2 = _project_points(_object_top_face_corners(pose, dims), c2w, K, W, H)
            if not ok2 or top is None:
                continue
            poly = np.round(top).astype(np.int32)
            cv2.fillPoly(img, [poly], TOP_PROXY_EGO if is_ego else TOP_PROXY_BODY, cv2.LINE_AA)
            cv2.polylines(img, [poly], True, TOP_PROXY_EGO_EDGE if is_ego else TOP_PROXY_EDGE,
                          2 if is_ego else 1, cv2.LINE_AA)
        except Exception:  # noqa: BLE001
            continue
    return img


_ACTOR_BANK_CACHE: Dict[str, Any] = {}


def load_actor_bank(scene_dir: str) -> Optional[Dict[str, Any]]:
    """找该场景对应的 actor 资产库（bank.json）。

    查找顺序：`<scene>/actor_bank.json` → `<repo>/output/actor_assets/<scene 名>/bank.json`。
    找到且含可用资产就返回（带缓存），否则 None（此时俯视渲染退回"方框替身"）。
    """
    cands = [os.path.join(scene_dir, "actor_bank.json")]
    repo_root = Path(__file__).resolve().parents[2]
    base = os.path.basename(os.path.normpath(scene_dir))
    cands.append(str(repo_root / "output" / "actor_assets" / base / "bank.json"))
    for p in cands:
        if not os.path.exists(p):
            continue
        key = f"{p}:{os.path.getmtime(p)}"
        if key in _ACTOR_BANK_CACHE:
            return _ACTOR_BANK_CACHE[key]
        try:
            bank = json.load(open(p, encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        usable = [a for a in bank.get("assets", []) if (a.get("metadata") or {}).get("usable", True)]
        if not usable:
            continue
        bank["_path"] = p
        # (flat_frame, cluster_id) → asset；同时准备 real_frame → [(cluster_id, asset)]
        by_key: Dict[Any, Any] = {}
        by_real: Dict[int, List[Any]] = {}
        n_views = int((usable[0].get("metadata") or {}).get("num_views") or 1) or 1
        for a in usable:
            md = a.get("metadata") or {}
            for fr, oid in (md.get("object_ids_per_frame") or []):
                by_key[(int(fr), int(oid))] = a
            for fr, oid in zip(md.get("frames") or [], md.get("object_ids_per_frame") or []):
                real = int(fr) // n_views
                by_real.setdefault(real, []).append((int(oid[1]), a))
        bank["_by_key"] = by_key
        bank["_by_real"] = by_real
        bank["_num_views"] = n_views
        _ACTOR_BANK_CACHE[key] = bank
        print(f"[topdown] 已加载 actor 资产库 {p}（可用资产 {len(usable)}）")
        return bank
    return None


def _assets_for_frame(tm, renderer, bank: Dict[str, Any], frame_idx: int):
    """把该帧的物体尽量用"资产几何 + TrackManager 当前位姿"渲染。

    Returns: (extras, covered_raw_ids)
      extras: 引擎的 extra_objects（资产 ply + 位姿）
      covered_raw_ids: 已由资产渲染的 raw_object_id 集合（俯视时不再画方框替身）
    """
    extras: List[Dict[str, Any]] = []
    covered: set = set()
    if tm is None or not bank:
        return extras, covered
    n_views = int(bank.get("_num_views") or 1)
    flat0 = int(frame_idx) * n_views
    try:
        objs = tm.get_frame_objects(frame_idx)
    except Exception:  # noqa: BLE001
        return extras, covered
    for o in objs:
        if o.get("synthetic"):
            continue                      # 合成物体（SAM3D/行人/主车）引擎已经原生渲染
        raw = o.get("raw_object_id")
        if raw is None:
            continue
        asset = bank["_by_key"].get((flat0, int(raw)))
        if asset is None:
            # 退化情况：该物体只在别的视角出现 → 用世界位置兜底匹配
            pose_now = np.asarray(o["pose_world"], dtype=np.float64)[:3, 3]
            best, bd = None, 2.0
            for cid, a in bank["_by_real"].get(int(frame_idx), []):
                md = a.get("metadata") or {}
                for fr, p in zip(md.get("frames") or [], md.get("poses_per_frame") or []):
                    if int(fr) // n_views != int(frame_idx) or p is None:
                        continue
                    P = np.asarray(p, dtype=np.float64)
                    d = float(np.linalg.norm((P[:3, 3] if P.ndim == 2 else P) - pose_now))
                    if d < bd:
                        best, bd = a, d
            asset = best
        if asset is None:
            continue
        pose = np.asarray(o["pose_world"], dtype=np.float32)
        extras.append({"ply_path": asset["source_path"], "transform": pose,
                       "synth_track_id": -1 - len(extras), "asset_id": asset.get("asset_id")})
        covered.add(int(raw))
    return extras, covered


def render_topdown_frame(renderer, tm, frame_idx: int, cam):
    """从固定俯视相机（正上方朝下）渲染一帧，返回 RGB ndarray。

    静态场景（道路/建筑/植被）用 gsplat **真渲染**；
    动态物体因为单目重建的高斯从上方会退化，这里改用"实体替身"（顶面多边形）保证
    **所有动态物体都可见且位置准确**；主车用真模型渲染（它是完整的 3D 高斯）。
    """
    c2w, K, W, H = cam
    device = renderer.device
    ego_track = getattr(tm, "ego_track_id", None) if tm is not None else None
    egos = []
    if tm is not None and ego_track is not None:
        egos = [e for e in tm.build_extra_objects(int(frame_idx)) if e.get("synth_track_id") == ego_track]
    synth_extras = []
    if tm is not None:
        # 合成物体（SAM3D 行人/主车、事故合成的参与者）也一并真渲染
        synth_extras = [e for e in tm.build_extra_objects(int(frame_idx))
                        if e.get("synth_track_id") != ego_track]
    bank = load_actor_bank(getattr(renderer, "scene_path", "") or "")
    asset_extras, covered = _assets_for_frame(tm, renderer, bank, int(frame_idx)) if bank else ([], set())
    image = renderer._render_frame_with_object_overrides(
        int(frame_idx), {},
        c2w_override=torch.tensor(c2w, device=device, dtype=torch.float32),
        K_override=torch.tensor(K, device=device, dtype=torch.float32),
        width_override=int(W), height_override=int(H),
        extra_objects=egos + synth_extras + asset_extras, include_dynamic=False)
    viewer = None
    if tm is not None:
        viewer = getattr(tm, "ego_source_track", None)
        if viewer is None:
            viewer = getattr(tm, "ego_track_id", None)
    # 只有"没有可用资产"的物体才退回方框替身（有资产的已经用真实模型渲染了）
    image = _draw_topdown_object_proxies(image, tm, int(frame_idx), cam,
                                         highlight_ego_track=viewer, skip_raw_ids=covered)
    return image


def _topdown_bounds(tm, renderer, start_frame: int, num_frames: int, focus_tracks=None):
    """俯视相机的取景范围。

    有参与者（focus_tracks）时**只框参与者**（事故才是要看的东西，自车可能很远）；
    车身只有 4.5m、行人 0.6m，跨度一大就只剩十几个像素，看不出是"真模型"。
    """
    xs, zs = [], []
    only_focus = bool(focus_tracks)
    for tid in (focus_tracks or []):
        for f in range(int(start_frame), int(start_frame) + int(num_frames) + 1):
            try:
                p = tm.get_track_pose(int(tid), f)
            except Exception:  # noqa: BLE001
                p = None
            if p is None:
                continue
            c = np.asarray(p, dtype=np.float32)[:3, 3]
            xs.append(float(c[0]))
            zs.append(float(c[2]))
    if not only_focus:
        for f in range(int(start_frame), int(start_frame) + int(num_frames)):
            cam = _frame_camera(renderer, f)
            if cam is None:
                continue
            c2w = cam[0]
            xs.append(float(c2w[0, 3]))
            zs.append(float(c2w[2, 3]))
    if not xs:
        return _bev_world_bounds(tm, start_frame, num_frames, margin=10.0)
    # 最小视野：横向 ≥16m、纵向 ≥30m，避免车少时贴得太近
    # 取景：以"参与者为主"并**限制最大跨度**，否则车在画面里只有十几个像素、看不出是模型。
    cx, cz = 0.5 * (min(xs) + max(xs)), 0.5 * (min(zs) + max(zs))
    span_x = max(10.0, min(20.0, max(xs) - min(xs) + 8.0))
    span_z = max(16.0, min(28.0, max(zs) - min(zs) + 8.0))
    # 若限制后仍装不下全部点（有远距离参与者），把中心偏向**近处主体**，保证主车+近处参与者清晰
    if max(xs) - min(xs) + 8.0 > span_x or max(zs) - min(zs) + 8.0 > span_z:
        cx = 0.5 * (float(np.percentile(xs, 10)) + float(np.percentile(xs, 90)))
        cz = 0.5 * (float(np.percentile(zs, 10)) + float(np.percentile(zs, 90)))
    return (cx - span_x / 2.0, cx + span_x / 2.0, cz - span_z / 2.0, cz + span_z / 2.0)


def _topdown_camera(tm, renderer, start_frame: int, num_frames: int, size: int = 640,
                    fov_y_deg: float = 60.0, focus_tracks=None, span_override: float = 0.0):
    """构造一个"正上方俯视"的相机（c2w, K, W, H）。

    朝向约定：世界 +Z 在画面里朝上（与 BEV 示意图一致），世界 -X 朝画面右，
    相机沿世界 -Y 向下看，因此画面是一张"竖着走"的地图。
    分辨率按取景范围的长宽比自适应（道路场景通常又长又窄，避免大片黑边）。
    """
    min_x, max_x, min_z, max_z = _topdown_bounds(tm, renderer, start_frame, num_frames, focus_tracks)
    span_x = max(6.0, max_x - min_x)
    span_z = max(6.0, max_z - min_z)
    if span_override and float(span_override) > 1.0:
        # 手工指定纵向视野（米）：想看清车/行人就调小（例如 20）
        span_z = float(span_override)
        span_x = span_z * 0.62
        cx0, cz0 = 0.5 * (min_x + max_x), 0.5 * (min_z + max_z)
        min_x, max_x = cx0 - span_x / 2, cx0 + span_x / 2
        min_z, max_z = cz0 - span_z / 2, cz0 + span_z / 2
    fov = math.radians(float(fov_y_deg))
    aspect = max(0.55, min(1.8, span_x / span_z))       # W/H
    long_side = int(max(480, min(960, size)))
    if aspect >= 1.0:
        W = long_side
        H = int(round(long_side / aspect))
    else:
        H = long_side
        W = int(round(long_side * aspect))
    W = int(max(320, W))
    H = int(max(320, H))
    fov_x = 2.0 * math.atan(math.tan(fov / 2.0) * W / float(H))
    height = max(8.0, min(400.0,
                          max(0.5 * span_z / math.tan(fov / 2.0),
                              0.5 * span_x / math.tan(fov_x / 2.0)) * 1.12 + 3.0))
    ys = []
    for f in range(int(start_frame), int(start_frame) + int(num_frames)):
        try:
            for o in tm.get_frame_objects(f):
                ys.append(float(np.asarray(o["pose_world"], dtype=np.float32)[1, 3]))
        except Exception:  # noqa: BLE001
            continue
    ground_y = float(np.median(ys)) if ys else 0.0
    cx = 0.5 * (min_x + max_x)
    cz = 0.5 * (min_z + max_z)
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, 0] = (-1.0, 0.0, 0.0)     # 画面右 = 世界 -X
    c2w[:3, 1] = (0.0, 0.0, -1.0)     # 画面下 = 世界 -Z（即 +Z 朝上）
    c2w[:3, 2] = (0.0, -1.0, 0.0)     # 相机朝向 = 世界 -Y（俯视）
    c2w[:3, 3] = (cx, ground_y + height, cz)
    K = _fov_to_intrinsics(float(fov_y_deg), W, H, renderer.device)
    K = K.detach().cpu().numpy().astype(np.float32)
    return c2w, K, W, H


def render_topdown_sequence_video(renderer, tm, start_frame: int, num_frames: int, fps: float,
                                  out_path: str, size: int = 640, fov_y_deg: float = 60.0,
                                  focus_tracks=None, span_override: float = 0.0):
    """把 [start, start+num) 逐帧用**真渲染的俯视相机**渲染并编码成 mp4。"""
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    cam = _topdown_camera(tm, renderer, start_frame, num_frames, size=size,
                          fov_y_deg=fov_y_deg, focus_tracks=focus_tracks,
                          span_override=span_override)
    frames = []
    for i in range(int(max(1, num_frames))):
        try:
            frames.append(render_topdown_frame(renderer, tm, int(start_frame) + i, cam))
        except Exception as e:  # noqa: BLE001
            print(f"[topdown] 跳过帧 {int(start_frame) + i}: {e}")
    if not frames:
        return None
    h, w = frames[0].shape[:2]
    try:
        import imageio
        with imageio.get_writer(out_path, fps=float(fps), codec="libx264", format="FFMPEG") as writer:
            for fr in frames:
                writer.append_data(fr if fr.dtype == np.uint8 else np.clip(fr, 0, 255).astype(np.uint8))
    except Exception:  # noqa: BLE001
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        vw = cv2.VideoWriter(out_path, fourcc, float(fps), (w, h))
        for fr in frames:
            vw.write(cv2.cvtColor(np.clip(fr, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
        vw.release()
    return out_path


def render_bev_sequence_video(tm, renderer, start_frame: int, num_frames: int, fps: float,
                              out_path: str, size: int = 560, highlight_tracks=None,
                              collision_frame=None, draw_ids: bool = False):
    """把俯视（BEV）逐帧画出来并编码成 mp4。"""
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    bounds = _bev_world_bounds(tm, start_frame, num_frames)
    frames = []
    for i in range(int(max(1, num_frames))):
        try:
            frames.append(render_bev_frame(tm, renderer, int(start_frame) + i, bounds, size,
                                           highlight_tracks, collision_frame, draw_ids))
        except Exception as e:  # noqa: BLE001
            print(f"[bev] 跳过帧 {int(start_frame) + i}: {e}")
    if not frames:
        return None
    h, w = frames[0].shape[:2]
    try:
        import imageio
        with imageio.get_writer(out_path, fps=float(fps), codec="libx264", format="FFMPEG") as writer:
            for fr in frames:
                writer.append_data(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
    except Exception:  # noqa: BLE001
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        vw = cv2.VideoWriter(out_path, fourcc, float(fps), (w, h))
        for fr in frames:
            vw.write(fr)
        vw.release()
    return out_path


# ==================== API路由 ====================

@app.get("/")
async def root():
    """API根路径"""
    return {
        "message": "DGGT Studio API V2",
        "version": "2.0.0",
        "features": [
            "轨迹可视化",
            "物体选择与编辑",
            "Corner Case生成",
            "实时渲染预览"
        ],
        "endpoints": {
            "scenes": "/api/scenes",
            "objects": "/api/objects/{scene_id}",
            "render": "/api/render",
            "edit": "/api/edit",
            "corner_case": "/api/corner_case"
        }
    }


@app.post("/api/scenes/load")
async def load_scene(request: SceneLoadRequest):
    """加载场景"""
    scene_id = request.scene_id or str(uuid.uuid4())
    
    try:
        renderer = DGGTRenderer(
            scene_path=request.scene_path,
            device="cuda",
            load_sky=request.load_sky,
            static_only=request.static_only
        )
        
        studio_state["scenes"][scene_id] = renderer
        studio_state["active_scene_id"] = scene_id
        studio_state["edit_history"][scene_id] = []

        # 构建 track 映射（跨帧追踪，供 3D 编辑/轨迹使用）
        try:
            studio_state["track_managers"][scene_id] = TrackManager(renderer)
        except Exception as e:
            print(f"[Studio] 构建 TrackManager 失败: {e}")
            studio_state["track_managers"][scene_id] = None
        
        # 获取场景信息
        scene_info = {
            "scene_id": scene_id,
            "scene_path": request.scene_path,
            "loaded_at": datetime.now().isoformat(),
            "objects": [],
            "num_frames": 0
        }
        
        # 扫描场景帧数（多视角数据已合并为真实帧数）
        scene_info["num_frames"] = renderer.frame_count()
        scene_info["data_frames"] = renderer.data_frame_count()   # 磁盘上真实存在的帧数
        scene_info["num_views"] = renderer.num_views
        
        return {"success": True, "scene": scene_info}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load scene: {str(e)}")


@app.get("/api/scenes")
async def list_scenes():
    """列出所有已加载的场景"""
    scenes = []
    for scene_id, renderer in studio_state["scenes"].items():
        scenes.append({
            "scene_id": scene_id,
            "scene_path": renderer.scene_path,
            "loaded_at": datetime.now().isoformat()
        })
    return {"scenes": scenes}


@app.delete("/api/scenes/{scene_id}")
async def unload_scene(scene_id: str):
    """卸载场景"""
    if scene_id in studio_state["scenes"]:
        del studio_state["scenes"][scene_id]
        studio_state["track_managers"].pop(scene_id, None)
        if scene_id in studio_state["edit_history"]:
            del studio_state["edit_history"][scene_id]
        return {"success": True}
    raise HTTPException(status_code=404, detail="Scene not found")


@app.get("/api/objects/{scene_id}")
async def get_objects(scene_id: str, frame_idx: int = 0):
    """获取指定帧的所有动态物体"""
    renderer = get_scene_or_404(scene_id)
    
    try:
        # 多视角场景的 meta 是 flat 排列的：真实帧 t 要取 flat_index(t, 0)
        try:
            _flat = renderer.flat_index(int(frame_idx), 0)
        except Exception:  # noqa: BLE001
            _flat = int(frame_idx)
        obj_meta_path = os.path.join(renderer.meta_dir, f"frame_{_flat:04d}_objects.json")
        if not os.path.exists(obj_meta_path):
            # 兜底：真实帧索引就是文件索引（单视角场景）
            obj_meta_path = os.path.join(renderer.meta_dir, f"frame_{int(frame_idx):04d}_objects.json")
        if not os.path.exists(obj_meta_path):
            return {"objects": [], "frame_idx": frame_idx}
        
        with open(obj_meta_path, "r") as f:
            objects = json.load(f)
        
        # 添加轨迹控制器中编辑过的物体位姿
        for obj in objects:
            obj_id = obj["object_id"]
            edited_pose = renderer.get_real_frame_object_pose(frame_idx, obj_id)
            if edited_pose is not None:
                obj["pose_world"] = edited_pose.cpu().numpy().tolist()
                obj["edited"] = True
            else:
                obj["edited"] = False
        
        return {
            "objects": objects,
            "frame_idx": frame_idx,
            "scene_id": scene_id
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/objects/{scene_id}/{object_id}/trajectory")
async def get_object_trajectory(scene_id: str, object_id: int):
    """获取物体的完整轨迹"""
    renderer = get_scene_or_404(scene_id)
    
    # 从轨迹控制器获取
    if object_id in renderer.trajectory_controller.plans:
        keyframes = []
        for frame_idx, pose in sorted(renderer.trajectory_controller.plans[object_id].items()):
            keyframes.append({
                "frame_idx": int(frame_idx),
                "pose_world": pose.tolist()
            })
        return {
            "object_id": object_id,
            "keyframes": keyframes,
            "source": "edited"
        }
    
    # 从原始数据获取
    trajectory = []
    for _t in range(int(renderer.frame_count())):   # 按真实帧扫描（多视角自动换算 flat）
        frame_idx = _t
        try:
            _flat = renderer.flat_index(_t, 0)
        except Exception:  # noqa: BLE001
            _flat = _t
        obj_meta_path = os.path.join(renderer.meta_dir, f"frame_{_flat:04d}_objects.json")
        if not os.path.exists(obj_meta_path):
            continue
        
        with open(obj_meta_path, "r") as f:
            objects = json.load(f)
        
        for obj in objects:
            if obj["object_id"] == object_id:
                trajectory.append({
                    "frame_idx": frame_idx,
                    "pose_world": obj["pose_world"],
                    "dimensions": obj.get("dimensions", [])
                })
                break
    
    return {
        "object_id": object_id,
        "trajectory": trajectory,
        "source": "original"
    }


@app.get("/api/timeline/{scene_id}")
async def get_timeline(scene_id: str):
    """当前时间轴：磁盘数据帧数 / 可渲染帧数（可延长）/ 从第几帧起是外推的。"""
    renderer = get_scene_or_404(scene_id)
    tm = None if getattr(renderer, "static_only", False) else get_track_manager_or_404(scene_id)
    info = _timeline_info(renderer, tm)
    info["success"] = True
    info["fps"] = float(getattr(tm, "fps", 10.0)) if tm is not None else 10.0
    info["max_frames"] = int(os.environ.get("DGGT_MAX_FRAMES", "4000"))
    return info


@app.post("/api/timeline/extend")
async def extend_timeline(request: TimelineExtendRequest):
    """按需延长可渲染帧数（超出原场景帧数）。

    延长做的事：自车/物体按各自轨迹把位姿外推出去，静态场景本来就是单份高斯、
    与帧无关；动态物体在超范围帧复用"最后一次出现"的外观。所以之后可以正常渲染、
    可以继续编辑轨迹、可以生成 corner case，而不是被原场景的帧数卡住。
    """
    renderer = get_scene_or_404(request.scene_id)
    tm = get_track_manager_or_404(request.scene_id)
    if tm is None:
        raise HTTPException(status_code=400, detail="该场景没有 TrackManager（static_only）")
    try:
        res = tm.extend_timeline(extra_frames=request.extra_frames,
                                 total_frames=request.total_frames,
                                 mode=(request.mode or "extrapolate"),
                                 ego_mode=request.ego_mode)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"延长时间轴失败: {e}")
    return {"success": True, **res, **_timeline_info(renderer, tm)}


@app.post("/api/render/frame")
async def render_frame(request: RenderRequest):
    """渲染单帧（增强版，支持轨迹可视化）"""
    renderer = get_scene_or_404(request.scene_id)
    
    try:
        # 多视角数据：物体覆盖 / 合成参与者统一走 TrackManager（基于 track 的编辑）
        tm = None if renderer.static_only else get_track_manager_or_404(request.scene_id)
        # 请求的帧超出原场景帧数时，按需把时间轴延长（物体/自车轨迹外推）
        if tm is not None and getattr(request, "auto_extend", True):
            _ensure_timeline(tm, frame_idx=request.frame_idx)
        if tm is not None:
            object_overrides = tm.build_object_overrides(request.frame_idx)
            # 自车视角：主车实体在镜头内部，必须排除，否则画面被车体糊住
            extra_objects = tm.build_extra_objects(request.frame_idx, include_ego=False)
        else:
            object_overrides = {}
            extra_objects = []
            # 兼容旧轨迹控制器编辑
            for oid in renderer.trajectory_controller.get_object_ids():
                pose = renderer.get_real_frame_object_pose(request.frame_idx, oid)
                if pose is not None:
                    object_overrides[oid] = pose

        # 若全局把"主车视角"切成了别的物体，2D 实时视图也跟着换到那个视角
        cam_ov = None
        if tm is not None and getattr(tm, "ego_source_track", None) is not None:
            try:
                cam_ov = tm.get_ego_camera(request.frame_idx)
            except Exception:  # noqa: BLE001
                cam_ov = None

        # 渲染（2D 自车视角，多视角下取该真实帧 view0 的自车相机）
        render_kw = {}
        if cam_ov is not None:
            render_kw["c2w_override"] = torch.tensor(np.asarray(cam_ov, dtype=np.float32),
                                                     device=renderer.device, dtype=torch.float32)
        image = renderer._render_frame_with_object_overrides(
            request.frame_idx,
            object_overrides,
            extra_objects=extra_objects,
            **render_kw,
        )

        # 相机参数（view0 自车相机）：数据帧读盘，超范围帧由引擎按自车轨迹合成
        try:
            ego_data = renderer._ego_data(request.frame_idx)
        except Exception:  # noqa: BLE001
            ego_data = None

        # 该真实帧下合并所有视角的动态物体（带 track_id / 编辑状态）
        objects_data = []
        if tm is not None:
            objects_data = tm.get_frame_objects(request.frame_idx)

        # 绘制边界框和ID
        if (request.draw_bboxes or request.draw_ids) and ego_data and objects_data:
            base_c2w = cam_ov if cam_ov is not None else ego_data["camera_extrinsics_world"]
            c2w = torch.tensor(np.asarray(base_c2w, dtype=np.float32), device=renderer.device).float()
            if c2w.shape == (3, 4):
                c2w = torch.cat([c2w, torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=renderer.device, dtype=c2w.dtype)], dim=0)
            viewmat = torch.inverse(c2w)
            K = torch.tensor(ego_data["camera_intrinsics"], device=renderer.device).float()
            W, H = ego_data["camera"]["width"], ego_data["camera"]["height"]

            for obj in objects_data:
                if obj.get("ego"):
                    # 主车：自车视角下相机就在车体里，画框会糊满画面 → 只保留在物体列表里
                    obj["object_id"] = obj["track_id"]
                    continue
                tid = obj["track_id"]
                pose = torch.tensor(obj["pose_world"], device=renderer.device).float()

                # 计算3D边界框
                corners_2d = renderer._get_bbox_corners_2d(
                    pose, obj["dimensions"], viewmat, K, W, H
                )

                if corners_2d is not None:
                    is_edited = bool(obj.get("edited"))
                    color = (0, 165, 255) if is_edited else (0, 255, 0)  # 橙色表示已编辑
                    label = f"T:{tid}" if request.draw_ids else None
                    image = renderer._draw_bbox(image, corners_2d, color=color, label=label)

                    # 添加2D边界框信息到物体数据（前端 overlay 用 object_id=track_id）
                    x_coords = corners_2d[:, 0]
                    y_coords = corners_2d[:, 1]
                    obj["object_id"] = tid
                    obj["bbox_2d"] = {
                        "x_min": float(np.min(x_coords)),
                        "x_max": float(np.max(x_coords)),
                        "y_min": float(np.min(y_coords)),
                        "y_max": float(np.max(y_coords))
                    }
        else:
            for obj in objects_data:
                obj["object_id"] = obj["track_id"]

        # 绘制轨迹
        if request.draw_trajectories and ego_data and tm is not None:
            viewmat = torch.inverse(
                torch.tensor(ego_data["camera_extrinsics_world"], device=renderer.device).float()
                if len(ego_data["camera_extrinsics_world"]) == 4
                else torch.cat([
                    torch.tensor(ego_data["camera_extrinsics_world"], device=renderer.device).float(),
                    torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=renderer.device),
                ], dim=0)
            )
            K = torch.tensor(ego_data["camera_intrinsics"], device=renderer.device).float()
            for obj in objects_data:
                traj = tm.get_track_trajectory(obj["track_id"], use_edits=True)
                pts = [np.asarray(p["center"], dtype=np.float32) for p in traj]
                if len(pts) >= 2:
                    image = draw_trajectory_on_image(
                        image, pts, viewmat, K, device=renderer.device
                    )

        # 编码为base64
        _, buffer = cv2.imencode('.png', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return {
            "success": True,
            "image": img_base64,
            "frame_idx": request.frame_idx,
            "width": image.shape[1],
            "height": image.shape[0],
            "objects": objects_data,  # 返回物体信息，包含2D边界框
            "timeline": _timeline_info(renderer, tm),
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/render/sequence")
async def render_sequence(request: SequenceRenderRequest, background_tasks: BackgroundTasks):
    """渲染视频序列（后台任务）"""
    renderer = get_scene_or_404(request.scene_id)
    try:
        _tm0 = None if renderer.static_only else get_track_manager_or_404(request.scene_id)
        _ensure_timeline(_tm0, frame_idx=int(request.start_frame) + int(request.num_frames) - 1)
    except Exception:  # noqa: BLE001
        pass
    
    task_id = str(uuid.uuid4())
    output_dir = f"/tmp/dggt_renders/{task_id}"
    os.makedirs(output_dir, exist_ok=True)
    
    # 添加后台任务
    background_tasks.add_task(
        renderer.render_sequence,
        request.num_frames,
        output_dir,
        request.draw_bboxes,
        request.draw_ids,
        request.start_frame,
        True,  # save_video
        "rendered_video.mp4",
        request.fps
    )
    
    return {
        "task_id": task_id,
        "output_dir": output_dir,
        "status": "processing"
    }


@app.post("/api/edit/object/pose")
async def edit_object_pose(scene_id: str, request: ObjectPose):
    """编辑单帧物体位姿"""
    renderer = get_scene_or_404(scene_id)
    
    try:
        pose_matrix = np.array(request.pose_matrix, dtype=np.float32)
        renderer.set_object_pose(request.frame_idx, request.object_id, pose_matrix)
        
        # 记录编辑历史
        history_entry = {
            "type": "pose_edit",
            "object_id": request.object_id,
            "frame_idx": request.frame_idx,
            "timestamp": datetime.now().isoformat()
        }
        studio_state["edit_history"][scene_id].append(history_entry)
        
        return {"success": True, "history_entry": history_entry}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/edit/trajectory")
async def edit_trajectory(scene_id: str, request: TrajectoryEdit):
    """编辑物体完整轨迹"""
    renderer = get_scene_or_404(scene_id)
    
    try:
        # 清除旧轨迹
        if request.object_id in renderer.trajectory_controller.plans:
            del renderer.trajectory_controller.plans[request.object_id]
        
        # 设置新轨迹
        keyframes = [(kf.frame_idx, np.array(kf.pose_world, dtype=np.float32)) for kf in request.keyframes]
        renderer.trajectory_controller.set_trajectory(request.object_id, keyframes)
        
        # 记录历史
        history_entry = {
            "type": "trajectory_edit",
            "object_id": request.object_id,
            "num_keyframes": len(request.keyframes),
            "timestamp": datetime.now().isoformat()
        }
        studio_state["edit_history"][scene_id].append(history_entry)
        
        return {
            "success": True,
            "object_id": request.object_id,
            "num_keyframes": len(request.keyframes),
            "history_entry": history_entry
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/edit/object/offset")
async def offset_object(scene_id: str, request: ObjectDragRequest):
    """移动物体（偏移操作）"""
    renderer = get_scene_or_404(scene_id)
    
    try:
        # 获取当前位姿
        current_pose = renderer.get_real_frame_object_pose(request.frame_idx, request.object_id)
        if current_pose is None:
            # 从原始数据加载
            try:
                _flat = renderer.flat_index(int(request.frame_idx), 0)
            except Exception:  # noqa: BLE001
                _flat = int(request.frame_idx)
            obj_meta_path = os.path.join(renderer.meta_dir, f"frame_{_flat:04d}_objects.json")
            with open(obj_meta_path, "r") as f:
                objects = json.load(f)
            for obj in objects:
                if obj["object_id"] == request.object_id:
                    current_pose = torch.tensor(obj["pose_world"], device=renderer.device).float()
                    break
        
        if current_pose is None:
            raise HTTPException(status_code=404, detail="Object not found in frame")
        
        # 应用偏移
        new_pose = current_pose.clone()
        new_pose[0, 3] += request.delta_x
        new_pose[1, 3] += request.delta_y
        new_pose[2, 3] += request.delta_z
        
        # 更新轨迹
        renderer.trajectory_controller.add_keyframe(
            request.object_id, 
            request.frame_idx, 
            new_pose.cpu().numpy()
        )
        
        return {
            "success": True,
            "object_id": request.object_id,
            "new_pose": new_pose.cpu().numpy().tolist()
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/edit/object/{scene_id}/{object_id}")
async def delete_object(scene_id: str, object_id: int, frame_idx: Optional[int] = None):
    """删除物体（设置到很远处或标记为删除）"""
    renderer = get_scene_or_404(scene_id)
    
    try:
        if frame_idx is not None:
            # 删除单帧中的物体
            renderer.set_object_pose(frame_idx, object_id, None)
        else:
            # 删除整个轨迹
            if object_id in renderer.trajectory_controller.plans:
                del renderer.trajectory_controller.plans[object_id]
        
        # 记录历史
        history_entry = {
            "type": "object_delete",
            "object_id": object_id,
            "frame_idx": frame_idx,
            "timestamp": datetime.now().isoformat()
        }
        studio_state["edit_history"][scene_id].append(history_entry)
        
        return {"success": True, "object_id": object_id}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class CornerCaseGenRequest(BaseModel):
    scene_id: str
    scenario_type: str
    roles: Dict[str, int]      # {role_key: track_id}
    start_frame: int = 0
    num_frames: int = 20
    intensity: float = 1.0
    enable_physics: bool = True   # 基于包围盒的物理碰撞规则
    fps: float = 10.0             # 帧率（用于碰撞时间/关键帧计算）
    sampling_params: Dict[str, Any] = Field(default_factory=dict)
    sampling_seed: Optional[int] = None


class EgoOffsetRequest(BaseModel):
    scene_id: str
    lateral: float = 0.0        # 车体右向为正
    longitudinal: float = 0.0   # 车头方向为正
    vertical: float = 0.0       # 世界 Y 向上为正


class EgoUpdateRequest(BaseModel):
    scene_id: str
    visible: Optional[bool] = None            # 主车实体是否渲染
    reset_trajectory: bool = False            # 还原为原始相机轨迹
    source_track_id: Optional[int] = None     # 用哪个物体的视角（None/0 = 真实主车）
    set_source: bool = False                  # 是否本次要更新视角来源
    scale: Optional[float] = None             # 模型缩放
    dimensions: Optional[List[float]] = None  # [宽, 高, 长]


class EgoGroundRequest(BaseModel):
    scene_id: str
    auto: bool = True                 # True=按当前场景地面重新贴地；False=还原为贴地前轨迹
    clearance: Optional[float] = None  # 车底离地余量（米），None=用默认
    radius: Optional[float] = None     # 地面估计的水平邻域半径（米）


@app.get("/api/ego/{scene_id}")
async def get_ego(scene_id: str):
    """主车（EGO）实体化信息：物体 track_id / 尺寸 / 是否被编辑 / 轨迹帧数。"""
    tm = get_track_manager_or_404(scene_id)
    if getattr(tm, "ego_track_id", None) is None:
        return {"success": True, "available": False,
                "reason": "未找到主车模型或自车相机外参（可用 DGGT_EGO_PLY 指定模型）"}
    s = tm.synthetic_tracks.get(tm.ego_track_id) or {}
    return {
        "success": True,
        "available": True,
        "track_id": int(tm.ego_track_id),
        **tm.get_ego_config(),
        "ply_path": s.get("ply_path"),
        "type": s.get("type"),
        "dimensions": s.get("dimensions"),
        "scale": s.get("scale"),
        "visible": bool(s.get("visible", True)),
        "edited": bool(tm.ego_is_edited()),
        "num_frames": len(s.get("poses") or {}),
        "frames": sorted(int(f) for f in (s.get("poses") or {}).keys())[:2000],
        "auto_ground": bool(s.get("auto_ground", False)),
        "ground": s.get("ground"),
        "cam_offset": None if s.get("cam_rel") is None else np.asarray(s["cam_rel"]).tolist(),
    }


@app.post("/api/ego/offset")
async def offset_ego(request: EgoOffsetRequest):
    """把整条主车轨迹整体平移（车体局部坐标系），用于快速改"主车走位"。"""
    tm = get_track_manager_or_404(request.scene_id)
    if getattr(tm, "ego_track_id", None) is None:
        raise HTTPException(status_code=404, detail="主车实体不存在")
    s = tm.synthetic_tracks[tm.ego_track_id]
    try:
        tm.push_history()
        for f, pose in list(s["poses"].items()):
            R = np.asarray(pose, dtype=np.float32)[:3, :3]
            right = R[:, 0]
            fwd = R[:, 2]
            delta = (right * float(request.lateral) + fwd * float(request.longitudinal)
                     + np.array([0.0, 1.0, 0.0], dtype=np.float32) * float(request.vertical))
            p = np.asarray(pose, dtype=np.float32).copy()
            p[:3, 3] = p[:3, 3] + delta
            s["poses"][f] = p
        tm._invalidate_heading_cache(tm.ego_track_id)
        return {"success": True, "track_id": int(tm.ego_track_id), "edited": bool(tm.ego_is_edited())}
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"主车平移失败: {e}")


@app.post("/api/ego")
async def update_ego(request: EgoUpdateRequest):
    """更新主车实体：显示/隐藏、缩放、尺寸、轨迹重置。

    编辑主车轨迹本身请走通用编辑接口（`/api/edit/track/point|trajectory|rotation` 等），
    track_id 用 `GET /api/ego/{scene_id}` 返回的 id。
    """
    tm = get_track_manager_or_404(request.scene_id)
    if getattr(tm, "ego_track_id", None) is None:
        raise HTTPException(status_code=404, detail="主车实体不存在")
    s = tm.synthetic_tracks[tm.ego_track_id]
    try:
        if request.visible is not None:
            tm.set_ego_visible(request.visible)
        if request.scale is not None:
            s["scale"] = float(max(0.05, min(50.0, request.scale)))
        if request.dimensions is not None and len(request.dimensions) == 3:
            s["dimensions"] = [float(max(0.2, min(20.0, v))) for v in request.dimensions]
        if request.set_source:
            tm.push_history()
            tm.set_ego_source(request.source_track_id)
        if request.reset_trajectory:
            tm.push_history()
            tm.reset_ego_track()
        return {"success": True, "track_id": int(tm.ego_track_id),
                **tm.get_ego_config(),
                "dimensions": s.get("dimensions"), "scale": s.get("scale"),
                "visible": bool(s.get("visible", True)),
                "edited": bool(tm.ego_is_edited())}
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"更新主车失败: {e}")


@app.post("/api/ego/ground")
async def ego_ground(request: EgoGroundRequest):
    """按当前 4DGS 场景地面重新对齐主车高度（逐场景自适应，避免车体沉进路面）。

    auto=true  : 从静态场景估计路面高度并重新贴地（可带 clearance/radius）
    auto=false : 还原为自动贴地之前的相机轨迹
    """
    tm = get_track_manager_or_404(request.scene_id)
    if getattr(tm, "ego_track_id", None) is None:
        raise HTTPException(status_code=404, detail="主车实体不存在")
    try:
        if request.auto:
            info = tm.refit_ego_ground(clearance=request.clearance, radius=request.radius)
        else:
            info = tm.reset_ego_ground()
        s = tm.synthetic_tracks.get(tm.ego_track_id) or {}
        return {"success": True, "track_id": int(tm.ego_track_id),
                "auto_ground": bool(s.get("auto_ground", False)),
                "ground": s.get("ground"), "info": info,
                "dimensions": s.get("dimensions"), "scale": s.get("scale"),
                "edited": bool(tm.ego_is_edited())}
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"主车贴地失败: {e}")


# ==================== 可信域（P1：novel-view trust / 越界截断 / 覆盖率报告） ====================


def _scene_path(scene_id: str) -> str:
    """scene_id → 重建场景目录（studio_state["scenes"] 存的是 renderer 实例）。"""
    r = get_scene_or_404(scene_id)
    p = getattr(r, "scene_path", None)
    if not p:
        raise HTTPException(status_code=404, detail=f"场景 {scene_id} 没有路径信息")
    return str(p)


def _trust_reports_auto() -> List[str]:
    """自动收集 output/trust 下已有的 trust_report.json（用于标定 trust 模型）。"""
    root = Path(__file__).resolve().parents[2] / "output" / "trust"
    if not root.is_dir():
        return []
    return sorted(str(p) for p in root.glob("trust_*.json"))


class TrustModelRequest(BaseModel):
    report_paths: Optional[List[str]] = None   # 不传则自动找 output/trust/trust_*.json
    metric: str = "ssim"


class TrustEnvelopeRequest(BaseModel):
    scene_id: str
    frame: int = 0
    lateral_offsets: List[float] = Field(default_factory=lambda: [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0])
    min_trust: float = 0.55
    report_paths: Optional[List[str]] = None


class TrustTrajectoryRequest(BaseModel):
    scene_id: str
    frame: int = 0
    num_frames: int = 20
    lateral: float = 0.0          # 整条轨迹横向平移（模拟自车偏离重建轨迹）
    track_id: Optional[int] = None  # 或以某个物体的轨迹作为"自车视角"
    min_trust: float = 0.55
    patience: int = 2
    report_paths: Optional[List[str]] = None


def _trust_model(report_paths: Optional[List[str]]) -> Dict[str, Any]:
    paths = report_paths or _trust_reports_auto()
    rows = trust_runtime.load_trust_rows(paths)
    return {"reports": paths, "num_rows": len(rows), "model": trust_runtime.fit_trust_model(rows)}


def _scene_camera(scene_id: str, frame: int):
    renderer = get_scene_or_404(scene_id)
    tm = None if renderer.static_only else get_track_manager_or_404(scene_id)
    cam = _frame_camera(renderer, frame)
    if cam is None:
        raise HTTPException(status_code=400, detail=f"帧 {frame} 没有自车相机")
    c2w, K, W, H = cam
    return renderer, tm, np.asarray(c2w, np.float64), np.asarray(K, np.float64), int(W), int(H)


@app.post("/api/trust/model")
async def trust_model(request: TrustModelRequest):
    """标定 trust 模型：用留出视角的真实误差 + coverage 拟合"在线指标 → 预期画质"。

    返回模型参数与 R²（以及"只用 coverage"的 R² 做对比），可复现、可审计。
    """
    try:
        return {"success": True, **_trust_model(request.report_paths)}
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"trust 模型拟合失败: {e}")


@app.post("/api/trust/envelope")
async def trust_envelope(request: TrustEnvelopeRequest):
    """可信域 envelope：把自车轨迹横向平移 N 米，给每档的 coverage / 预期 SSIM / 是否可信。

    `max_safe_lateral_m` = 仍然可信的最大横移 —— 闭环 rollout 允许的偏离范围。
    """
    try:
        _, _, c2w, K, W, H = _scene_camera(request.scene_id, request.frame)
        m = _trust_model(request.report_paths)
        rep = trust_runtime.lateral_sweep(
            _scene_path(request.scene_id), request.lateral_offsets, K, W, H,
            m["model"], min_trust=request.min_trust, frame=request.frame)
        return {"success": True, "model": m["model"], "num_rows": m["num_rows"], "envelope": rep}
    except HTTPException:
        raise
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"可信域评估失败: {e}")


@app.post("/api/trust/trajectory")
async def trust_trajectory(request: TrustTrajectoryRequest):
    """给一条（可平移/可换成某物体视角的）轨迹逐帧打 trust，并给出**越界截断点**与覆盖率报告。

    这是闭环仿真的运行时接口：仿真里自车每走一步，都能问"这一帧还可信吗、还能继续多久"。
    """
    try:
        renderer, tm, c2w0, K, W, H = _scene_camera(request.scene_id, request.frame)
        m = _trust_model(request.report_paths)
        poses = []
        n = int(max(1, request.num_frames))
        if request.track_id is not None and tm is not None:
            for i in range(n):
                f = request.frame + i
                p = tm.get_track_pose(int(request.track_id), f)
                if p is None:
                    continue
                cam = np.asarray(p, np.float64) @ np.asarray(
                    tm.synthetic_tracks[tm.ego_track_id]["cam_rel"], np.float64)
                poses.append((f, cam))
        else:
            for i in range(n):
                f = request.frame + i
                cam = _frame_camera(renderer, f)
                if cam is None:
                    continue
                P = np.asarray(cam[0], np.float64)
                if request.lateral:
                    P = P.copy()
                    P[:3, 3] = P[:3, 3] + P[:3, 0] * float(request.lateral)
                poses.append((f, P))
        rep = trust_runtime.trust_of_trajectory(
            _scene_path(request.scene_id), poses, K, W, H, m["model"],
            min_trust=request.min_trust, patience=request.patience)
        return {"success": True, "model": m["model"], "num_rows": m["num_rows"],
                "lateral": request.lateral, "report": rep}
    except HTTPException:
        raise
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"轨迹可信度评估失败: {e}")


# ==================== 闭环仿真工具链（资产库 / 多视角 / 扩散精修） ====================


class ActorBankRequest(BaseModel):
    scene_id: str
    out_dir: Optional[str] = None
    solidify: bool = True          # 体素化成体积代理（任意视角都能看到实体）
    min_frames: int = 3


@app.post("/api/actor_assets/build")
async def actor_assets_build(request: ActorBankRequest):
    """为当前场景构建 actor 资产库（对象中心 3D 资产；供俯视 3D 与可摆位渲染使用）。"""
    import actor_assets as _aa
    scene_dir = _scene_path(request.scene_id)
    out = request.out_dir or str(Path(__file__).resolve().parents[2] / "output" / "actor_assets"
                                / os.path.basename(os.path.normpath(scene_dir)))
    try:
        bank = _aa.build_scene_asset_bank(scene_dir, out, solidify=request.solidify,
                                          min_frames=request.min_frames, verbose=False)
        return {"success": True, "out_dir": out, "bank_path": os.path.join(out, "bank.json"),
                "num_tracks": bank["num_tracks"], "num_usable": bank["num_usable"],
                "assets": [{"asset_id": a["asset_id"],
                            "track_id": a["metadata"]["track_id"],
                            "num_gaussians": a["metadata"]["num_gaussians"],
                            "dimensions": a["metadata"]["dimensions"],
                            "usable": a["metadata"]["usable"],
                            "mean_speed_mps": a["metadata"]["mean_speed_mps"],
                            "num_frames": a["metadata"]["num_frames"]}
                           for a in bank["assets"]]}
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"构建 actor 资产库失败: {e}")


@app.get("/api/actor_assets/{scene_id}")
async def actor_assets_info(scene_id: str):
    """查看当前场景是否已有可用的 actor 资产库（俯视 3D 会自动用它渲染真模型）。"""
    scene_dir = _scene_path(scene_id)
    cands = [os.path.join(scene_dir, "actor_bank.json"),
             str(Path(__file__).resolve().parents[2] / "output" / "actor_assets"
                 / os.path.basename(os.path.normpath(scene_dir)) / "bank.json")]
    for p in cands:
        if os.path.exists(p):
            try:
                b = json.load(open(p, encoding="utf-8"))
                return {"success": True, "available": True, "path": p,
                        "num_tracks": b.get("num_tracks"), "num_usable": b.get("num_usable"),
                        "solidified": b.get("solidified"),
                        "assets": [{"asset_id": a["asset_id"],
                                    "track_id": (a.get("metadata") or {}).get("track_id"),
                                    "usable": (a.get("metadata") or {}).get("usable")}
                                   for a in b.get("assets", [])]}
            except Exception:  # noqa: BLE001
                continue
    return {"success": True, "available": False, "candidates": cands}


class DiffusionRefineRequest(BaseModel):
    scene_id: str
    frame_idx: int = 0
    width: int = 800
    height: int = 600


@app.post("/api/diffusion/refine_frame")
async def diffusion_refine_frame(request: DiffusionRefineRequest):
    """对当前场景的某一帧做扩散精修（Difix），返回「精修前 / 精修后」两张图。

    独立可用：不需要先跑「一键自检」。
    """
    st = diffusion_refine.status()
    if not st.get("ready"):
        raise HTTPException(status_code=503,
                            detail=f"扩散精修未就绪（缺依赖）：{st.get('missing') or st}")
    r = await render_frame(RenderRequest(scene_id=request.scene_id, frame_idx=request.frame_idx,
                                         width=request.width, height=request.height,
                                         draw_bboxes=False, draw_ids=False))
    buf = base64.b64decode(r["image"])
    bgr = cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    import time as _time
    t0 = _time.time()
    out = diffusion_refine.refine_image_rgb(rgb)
    dt = _time.time() - t0
    repo = Path(__file__).resolve().parents[2]
    out_dir = repo / "output" / "diffusion_preview"
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{request.scene_id[:8]}_f{request.frame_idx:04d}"
    before = out_dir / f"{stem}_before.png"
    after = out_dir / f"{stem}_after.png"
    cv2.imwrite(str(before), bgr)
    cv2.imwrite(str(after), cv2.cvtColor((out * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
    return {"success": True, "seconds": round(dt, 2), "frame_idx": request.frame_idx,
            "before": str(before), "after": str(after)}


@app.get("/api/diffusion/preview")
async def diffusion_preview(path: str):
    """受限于 output/ 的图片访问入口（扩散精修前后对比图）。"""
    repo = Path(__file__).resolve().parents[2]
    allowed = (repo / "output").resolve()
    p = Path(path).resolve()
    if not str(p).startswith(str(allowed)) or not p.exists():
        raise HTTPException(status_code=404, detail="图片不存在或不在 output/ 内")
    return FileResponse(str(p), media_type="image/png")


class DiffusionSeqRequest(BaseModel):
    scene_id: str
    start_frame: int = 0
    num_frames: int = 0            # 0 = 从 start_frame 精修到场景结尾
    width: int = 800
    height: int = 600
    make_video: bool = True
    fps: int = 10


_diffusion_jobs: Dict[str, Dict[str, Any]] = {}


def _run_refine_sequence(job_id: str, req: Dict[str, Any]) -> None:
    """后台把一段的所有帧依次精修，最后用 ffmpeg 合成"精修后/原始"两段 mp4。"""
    import asyncio
    import shutil as _sh
    import subprocess as _sp
    job = _diffusion_jobs[job_id]
    try:
        renderer = get_scene_or_404(req["scene_id"])
        total = int(renderer.frame_count())
        start = max(0, min(int(req.get("start_frame") or 0), max(0, total - 1)))
        num = int(req.get("num_frames") or 0) or (total - start)
        num = max(1, min(num, total - start))
        job["total"] = num
        repo = Path(__file__).resolve().parents[2]
        out_dir = (repo / "output" / "diffusion_seq" /
                   f"{req['scene_id'][:8]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        before_dir = out_dir / "before"
        before_dir.mkdir(parents=True, exist_ok=True)
        job["out_dir"] = str(out_dir)
        loop = asyncio.new_event_loop()
        try:
            for i in range(num):
                f = start + i
                r = loop.run_until_complete(render_frame(RenderRequest(
                    scene_id=req["scene_id"], frame_idx=f,
                    width=int(req.get("width") or 800), height=int(req.get("height") or 600),
                    draw_bboxes=False, draw_ids=False)))
                bgr = cv2.imdecode(np.frombuffer(base64.b64decode(r["image"]), np.uint8),
                                   cv2.IMREAD_COLOR)
                out = diffusion_refine.refine_image_rgb(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
                cv2.imwrite(str(before_dir / f"f{i:04d}.png"), bgr)
                cv2.imwrite(str(out_dir / f"f{i:04d}.png"),
                            cv2.cvtColor((out * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
                job["done"] = i + 1
                job["last_frame"] = f
                job["files"] = [str(out_dir / f"f{job['done'] - 1:04d}.png")]
        finally:
            loop.close()
        if req.get("make_video", True) and _sh.which("ffmpeg"):
            fps = str(int(req.get("fps") or 10))
            for src_dir, key in ((out_dir, "video"), (before_dir, "video_before")):
                dst = out_dir / ("refined.mp4" if key == "video" else "original.mp4")
                cmd = ["ffmpeg", "-y", "-loglevel", "error", "-framerate", fps,
                       "-i", str(src_dir / "f%04d.png"), "-c:v", "libx264", "-crf", "20",
                       "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(dst)]
                rr = _sp.run(cmd, capture_output=True, timeout=1800)
                if rr.returncode == 0 and dst.exists():
                    job[key] = str(dst)
        job["status"] = "done"
    except Exception as e:  # noqa: BLE001
        job["status"] = "failed"
        job["error"] = f"{type(e).__name__}: {e}"
    finally:
        job["ended"] = datetime.now().timestamp()


@app.post("/api/diffusion/refine_sequence")
async def diffusion_refine_sequence(request: DiffusionSeqRequest):
    """一次精修"所有帧"（后台作业），完成后给出精修后/原始两段视频。"""
    import threading
    st = diffusion_refine.status()
    if not st.get("ready"):
        raise HTTPException(status_code=503, detail=f"扩散精修未就绪：{st.get('missing') or st}")
    get_scene_or_404(request.scene_id)
    job_id = uuid.uuid4().hex[:10]
    _diffusion_jobs[job_id] = {"id": job_id, "scene_id": request.scene_id, "status": "running",
                               "done": 0, "total": 0, "files": [], "video": None,
                               "video_before": None, "out_dir": None, "error": None,
                               "started": datetime.now().timestamp(), "ended": None}
    threading.Thread(target=_run_refine_sequence, args=(job_id, request.dict()), daemon=True).start()
    return {"success": True, "job_id": job_id}


@app.get("/api/diffusion/refine_job/{job_id}")
async def diffusion_refine_job(job_id: str):
    j = _diffusion_jobs.get(job_id)
    if not j:
        raise HTTPException(status_code=404, detail="作业不存在（后端重启过？）")
    return j


@app.get("/api/diffusion/status")
async def diffusion_status():
    """扩散精修（Difix）依赖自检：权重是否就位、sd-turbo 是否齐全（缺就给出修复命令）。"""
    try:
        return {"success": True, **diffusion_refine.status()}
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"扩散精修自检失败: {e}")


class MultiViewPlanRequest(BaseModel):
    scene: str                                  # 预处理场景名（如 001）
    cameras: List[int] = Field(default_factory=lambda: [0, 1, 2])
    frames: int = 8
    holdout: List[int] = Field(default_factory=list)
    start_idx: int = 0
    run_name: str = ""
    do_eval: bool = True
    do_bank: bool = False
    max_images: int = 24


@app.post("/api/multiview/plan")
async def multiview_plan(request: MultiViewPlanRequest):
    """多视角重建的**预算检查 + 可复制命令**（不执行）。

    实测：24 张图（3 视角×8 帧）可行；40 张（5 视角×8 帧）在 24GB 卡上 OOM。
    """
    n_images = len(request.cameras) * int(request.frames)
    run_name = request.run_name or f"scene{request.scene}_cam{''.join(str(c) for c in request.cameras)}_f{request.frames}"
    parts = [sys.executable, "-u", str(Path(__file__).resolve().parent / "run_multiview.py"),
             "--scene", str(request.scene), "--cameras", ",".join(str(c) for c in request.cameras),
             "--frames", str(request.frames), "--start_idx", str(request.start_idx),
             "--run_name", run_name, "--max_images", str(request.max_images)]
    if request.holdout:
        parts += ["--holdout", ",".join(str(c) for c in request.holdout)]
    if request.do_eval:
        parts.append("--eval")
    if request.do_bank:
        parts.append("--bank")
    return {"success": True, "num_images": n_images, "max_images": int(request.max_images),
            "fits_budget": n_images <= int(request.max_images), "run_name": run_name,
            "cmd": " ".join(parts),
            "note": "预算超了会 OOM：减少 frames 或 cameras（或换更大显存的卡）"}


class MultiViewRunRequest(MultiViewPlanRequest):
    confirm: bool = False


@app.post("/api/multiview/run")
async def multiview_run(request: MultiViewRunRequest):
    """后台启动一次多视角重建+评测（返回日志路径，用 /api/multiview/log 查看进度）。"""
    if not request.confirm:
        raise HTTPException(status_code=400, detail="需要 confirm=true 才会真正启动（会占用 GPU）")
    plan = await multiview_plan(MultiViewPlanRequest(**request.model_dump()))
    if not plan["fits_budget"]:
        raise HTTPException(status_code=400, detail=f"预算超限：{plan['num_images']} 张图 > {plan['max_images']}")
    out_dir = Path(__file__).resolve().parents[2] / "output" / "trust" / plan["run_name"]
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "run.log"
    try:
        import shlex
        with open(log_path, "a", encoding="utf-8") as log:
            log.write(f"\n=== launched {datetime.now().isoformat()} ===\n$ {plan['cmd']}\n")
        p = subprocess.Popen(shlex.split(plan["cmd"]), cwd=str(Path(__file__).resolve().parents[2]),
                             stdout=open(log_path, "a"), stderr=subprocess.STDOUT,
                             start_new_session=True)
        return {"success": True, "pid": p.pid, "log_path": str(log_path),
                "out_dir": str(out_dir), "cmd": plan["cmd"]}
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"启动失败: {e}")


@app.get("/api/multiview/log")
async def multiview_log(path: str, tail: int = 80):
    """读取多视角运行的日志尾部（限制在仓库 output/ 下）。"""
    root = (Path(__file__).resolve().parents[2] / "output").resolve()
    p = Path(path).resolve()
    if not str(p).startswith(str(root)) or not p.exists():
        raise HTTPException(status_code=400, detail="路径不合法或不存在")
    lines = p.read_text(encoding="utf-8", errors="ignore").splitlines()
    return {"success": True, "path": str(p), "tail": lines[-int(max(1, min(2000, tail))):]}


# ==================== P2/P3：相机 rig 传感器渲染 + 闭环 rollout ====================


class SensorRigRequest(BaseModel):
    scene_id: str
    segment: str = "001"                     # 标定来源的预处理场景名
    frame: int = 0
    cameras: List[int] = Field(default_factory=lambda: [0, 1, 2, 3, 4])
    ref_camera: int = 0
    width: int = 960
    height: int = 640
    distort: bool = False                    # 把镜头畸变加回渲染图
    save_dir: Optional[str] = None


@app.post("/api/sensor/rig")
async def sensor_rig_render(request: SensorRigRequest):
    """渲染**整套相机 rig**（Waymo 5 相机标定）→ 返回每台相机的 base64 PNG + 几何校验。

    这是闭环仿真的"传感器"：规划器拿到的不是单张自车视角，而是一整套相机观测。
    """
    renderer = get_scene_or_404(request.scene_id)
    scene_dir = _scene_path(request.scene_id)
    data_root = str(Path(__file__).resolve().parents[2] / "data" / "waymo" / "processed" / "validation")
    try:
        rig = sensor_rig.load_rig(data_root, request.segment, request.cameras)
        imgs = sensor_rig.render_rig_frame(
            renderer, int(request.frame), rig, request.ref_camera, request.cameras,
            size=(int(request.width), int(request.height)), distort=bool(request.distort))
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"rig 渲染失败: {e}")
    out = []
    import cv2
    import base64
    for cam, img in imgs.items():
        T = sensor_rig.camera_to_ref(rig, request.ref_camera, int(cam))
        ang = math.degrees(math.acos(max(-1.0, min(1.0, float(T[2, 2])))))
        p = None
        if request.save_dir:
            d = Path(request.save_dir)
            if str(d.resolve()).startswith(str((Path(__file__).resolve().parents[2] / "output").resolve())):
                d.mkdir(parents=True, exist_ok=True)
                p = str(d / f"frame{int(request.frame):04d}_cam{int(cam)}.png")
                cv2.imwrite(p, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        _, buf = cv2.imencode(".png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        out.append({"camera": int(cam), "view_angle_vs_ref_deg": round(float(ang), 2),
                    "size": [int(img.shape[1]), int(img.shape[0])], "file": p,
                    "image": "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode()})
    return {"success": True, "scene_dir": scene_dir, "frame": int(request.frame),
            "cameras": out, "distort": bool(request.distort)}


class SimRolloutRequest(BaseModel):
    scene_id: str
    steps: int = 12
    speed: float = 8.0
    steer: float = 0.0
    actions: Optional[List[Dict[str, float]]] = None   # 也可逐帧给动作
    min_trust: float = 0.55
    min_trust_frac: Optional[float] = None             # 相对阈值（× 起始帧 pred_ssim）
    patience: int = 2
    cameras: List[int] = Field(default_factory=lambda: [0])
    width: int = 960
    height: int = 640
    keep_frames: int = 0                               # >0 时返回最后 N 帧观测图（base64）


@app.post("/api/sim/rollout")
async def sim_rollout(request: SimRolloutRequest):
    """跑一段**闭环脚本 rollout**：自车按动作运动 → 逐帧观测 + trust → 越界/碰撞即终止。

    返回逐步日志与 episode 级覆盖率报告（`trusted_ratio` / `min_coverage` / 截断原因 / 碰撞）。
    """
    scene_dir = _scene_path(request.scene_id)
    try:
        sim = sim_runtime.ScenarioSim(
            scene_dir, cameras=request.cameras, width=int(request.width), height=int(request.height),
            min_trust=float(request.min_trust), min_trust_frac=request.min_trust_frac,
            patience=int(request.patience), max_steps=int(request.steps))
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"初始化仿真失败: {e}")
    try:
        obs = sim.reset()
        log = [{"step": 0, "frame": obs.get("frame_idx"), "trust": obs.get("trust"),
                "near_m": None, "collision": []}]
        actions = request.actions or [{"speed": request.speed, "steer": request.steer}
                                      for _ in range(int(request.steps))]
        frames_b64 = []
        import cv2
        import base64
        for i, act in enumerate(actions):
            obs, r, done, info = sim.step(dict(act))
            log.append({"step": info["step"], "frame": info["frame"], "trust": obs.get("trust"),
                        "near_m": info["min_actor_distance_m"], "collision": info["collision_tracks"],
                        "reward": round(float(r), 3),
                        "truncated": info["truncated"], "truncate_reason": info["truncate_reason"]})
            if request.keep_frames > 0:
                for name, img in (obs.get("images") or {}).items():
                    _, buf = cv2.imencode(".png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                    frames_b64.append({"step": info["step"], "camera": name,
                                       "image": "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode()})
                frames_b64 = frames_b64[-int(request.keep_frames) * max(1, len(request.cameras)):]
            if done:
                break
        rep = sim.coverage_report()
        return {"success": True, "report": {k: v for k, v in rep.items() if k != "history"},
                "log": log, "frames": frames_b64}
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"rollout 失败: {e}")


# ==================== P4：导出 OpenSCENARIO / CommonRoad ====================


class ScenarioExportRequest(BaseModel):
    scene_id: str
    scenario_type: Optional[str] = None      # 给了就先生成该事故再导出（不改动场景状态）
    roles: Optional[Dict[str, int]] = None
    seed: int = 7
    start_frame: int = 0
    num_frames: int = 20
    fps: float = 10.0
    name: Optional[str] = None
    out_dir: Optional[str] = None
    all_tracks: bool = False                 # 默认只导 ego + 事故参与者 + 合成参与者
    map_file: str = "placeholder.xodr"       # 占位地图（预处理数据里没有 roadgraph）


@app.post("/api/export/scenario")
async def export_scenario(request: ScenarioExportRequest):
    """把（当前或新生成的）事故场景导出成 **OpenSCENARIO 1.2 + CommonRoad + 世界坐标 JSON**。

    注意：预处理数据里没有 roadgraph，所以道路/车道是**占位**（文件里 source/map_source 已标明）；
    轨迹、尺寸、速度都是真实完整的。导出后会**解析回来逐帧比对**（round-trip 校验）。
    """
    tm = get_track_manager_or_404(request.scene_id)
    frames = list(range(request.start_frame, request.start_frame + int(request.num_frames) + 1))
    fps = float(request.fps)
    tm.fps = fps
    affected: List[int] = []
    synthesized: List[int] = []
    pushed = False
    try:
        if request.scenario_type:
            roles = request.roles
            if not roles:
                g = scene_graph.SceneGraph(tm, fps=fps)
                g.build(frame_idx=request.start_frame, window=request.frame_window if hasattr(request, "frame_window") else 10)
                cands = g.propose_participants(request.scenario_type, max_candidates=2)
                if not cands:
                    raise HTTPException(status_code=400, detail="关系图里没有该类型的候选参与者")
                roles = cands[0]
            tm.push_history()
            pushed = True
            res = corner_case.generate(tm, request.scenario_type, roles, request.start_frame,
                                       int(request.num_frames), enable_physics=True, fps=fps,
                                       sampling_seed=request.seed)
            affected = [int(x) for x in (res.get("affected_tracks") or [])]
            synthesized = [int(x) for x in (res.get("synthesized_tracks") or [])]
        else:
            affected = [int(o["track_id"]) for o in tm.get_frame_objects(request.start_frame)]
        ego_tid = getattr(tm, "ego_track_id", None)
        keep = None if request.all_tracks else (
            ([ego_tid] if ego_tid is not None else []) + affected + synthesized)
        tracks = scenario_export.collect_tracks(tm, frames, ego_track=ego_tid, only_tracks=keep)
        if not tracks:
            raise HTTPException(status_code=400, detail="没有可导出的轨迹")
        name = request.name or f"{request.scenario_type or 'scene'}_s{request.seed}"
        repo = Path(__file__).resolve().parents[2]
        out_dir = Path(request.out_dir) if request.out_dir else (repo / "output" / "export" / name)
        if not str(out_dir.resolve()).startswith(str((repo / "output").resolve())):
            raise HTTPException(status_code=400, detail="out_dir 必须在仓库 output/ 下")
        osc = scenario_export.export_openscenario(
            tracks, str(out_dir / f"{name}.xosc"), dt=1.0 / fps, name=name, map_file=request.map_file)
        cr = scenario_export.export_commonroad(
            tracks, str(out_dir / f"{name}.cr.xml"), dt=1.0 / fps, name=name)
        wj = scenario_export.export_world_json(
            tracks, str(out_dir / f"{name}.world.json"), dt=1.0 / fps)
        val = scenario_export.validate_roundtrip(osc, cr, tracks, dt=1.0 / fps)
        return {"success": True, "name": name, "out_dir": str(out_dir),
                "files": {"openscenario": osc, "commonroad": cr, "world_json": wj},
                "num_tracks": len(tracks),
                "tracks": [{"track_id": t["track_id"], "role": t["role"],
                            "is_ego": t["is_ego"], "speed_mps": t["speed_mps"],
                            "dimensions": t["dimensions"], "num_frames": len(t["poses"])}
                           for t in tracks],
                "validation": val,
                "caveat": "道路/车道为占位（预处理数据无 roadgraph）；轨迹/尺寸/速度真实"}
    finally:
        if pushed:
            tm.undo()          # 导出不应改动场景状态


# ==================== 一键自检 / 扩散 A/B（后台任务 + 报告读取） ====================


class DemoRunRequest(BaseModel):
    scene_id: str
    segment: str = "001"
    scenario: str = "rear-end"
    seed: int = 7
    frames: int = 8
    steps: int = 5
    min_trust_frac: float = 0.8
    out_dir: Optional[str] = None
    confirm: bool = True


@app.post("/api/demo/plan")
async def demo_plan(request: DemoRunRequest):
    """闭环能力一键自检的**命令预览**（不执行）。"""
    scene_dir = _scene_path(request.scene_id)
    out_dir = request.out_dir or str(Path(__file__).resolve().parents[2] / "output" / "demo")
    parts = [sys.executable, "-u", str(Path(__file__).resolve().parent / "run_closedloop_demo.py"),
             "--scene", scene_dir, "--segment", request.segment, "--scenario", request.scenario,
             "--seed", str(request.seed), "--frames", str(request.frames), "--steps", str(request.steps),
             "--min_trust_frac", str(request.min_trust_frac), "--out_dir", out_dir]
    return {"success": True, "cmd": " ".join(parts), "out_dir": out_dir,
            "report_path": os.path.join(out_dir, os.path.basename(os.path.normpath(scene_dir)),
                                        "closedloop_demo_report.json")}


@app.post("/api/demo/run")
async def demo_run(request: DemoRunRequest):
    """后台跑一键自检（trust 模型 → 可信域 → 资产库 → rig → rollout → 导出 → 扩散状态）。"""
    if not request.confirm:
        raise HTTPException(status_code=400, detail="需要 confirm=true 才会真正启动（会占用 GPU）")
    plan = await demo_plan(request)
    log_path = os.path.join(plan["out_dir"], "demo_run.log")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    try:
        import shlex
        with open(log_path, "a", encoding="utf-8") as log:
            log.write(f"\n=== launched {datetime.now().isoformat()} ===\n$ {plan['cmd']}\n")
        p = subprocess.Popen(shlex.split(plan["cmd"]), cwd=str(Path(__file__).resolve().parents[2]),
                             stdout=open(log_path, "a"), stderr=subprocess.STDOUT, start_new_session=True)
        return {"success": True, "pid": p.pid, "log_path": log_path,
                "report_path": plan["report_path"], "cmd": plan["cmd"]}
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"启动失败: {e}")


class TrustABRequest(BaseModel):
    scene_id: str
    segment: str = "001"
    num_frames: int = 4
    heldout: List[int] = Field(default_factory=lambda: [3, 4])
    ref_camera: int = 0
    refine: str = "difix"            # none | stub | difix
    out_name: str = ""
    confirm: bool = True


@app.post("/api/trust/ab/plan")
async def trust_ab_plan(request: TrustABRequest):
    """渲染精修 A/B 的命令预览 + 依赖自检（difix 未就绪时直接说明缺什么）。"""
    scene_dir = _scene_path(request.scene_id)
    name = request.out_name or f"trust_ab_{request.refine}"
    out = str(Path(__file__).resolve().parents[2] / "output" / "trust" / f"{name}.json")
    parts = [sys.executable, "-u", str(Path(__file__).resolve().parent / "novelview_trust.py"),
             "--scene_dir", scene_dir, "--segment", request.segment,
             "--num_frames", str(request.num_frames), "--ref_camera", str(request.ref_camera),
             "--heldout", ",".join(str(c) for c in request.heldout), "--refine", request.refine,
             "--out", out]
    dep = diffusion_refine.status()
    return {"success": True, "cmd": " ".join(parts), "report_path": out,
            "refine": request.refine, "diffusion_ready": dep["ready"],
            "diffusion_missing": (dep.get("sd_turbo") or {}).get("missing")}


@app.post("/api/trust/ab/run")
async def trust_ab_run(request: TrustABRequest):
    """后台跑渲染精修 A/B（difix 需要 sd-turbo 权重就绪，否则用 stub 验证管道）。"""
    if not request.confirm:
        raise HTTPException(status_code=400, detail="需要 confirm=true 才会真正启动")
    plan = await trust_ab_plan(request)
    if request.refine == "difix" and not plan["diffusion_ready"]:
        raise HTTPException(status_code=400,
                            detail=f"Difix 未就绪，缺 {plan['diffusion_missing']}；"
                                   f"可先用 refine=stub 验证管道")
    log_path = plan["report_path"].replace(".json", ".log")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    try:
        import shlex
        with open(log_path, "a", encoding="utf-8") as log:
            log.write(f"\n=== launched {datetime.now().isoformat()} ===\n$ {plan['cmd']}\n")
        p = subprocess.Popen(shlex.split(plan["cmd"]), cwd=str(Path(__file__).resolve().parents[2]),
                             stdout=open(log_path, "a"), stderr=subprocess.STDOUT, start_new_session=True)
        return {"success": True, "pid": p.pid, "log_path": log_path, "report_path": plan["report_path"]}
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"启动失败: {e}")


@app.get("/api/report")
async def read_report(path: str):
    """读取仓库 output/ 下的 JSON 报告（一键自检 / A/B 结果）。"""
    root = (Path(__file__).resolve().parents[2] / "output").resolve()
    p = Path(path).resolve()
    if not str(p).startswith(str(root)) or not p.exists():
        raise HTTPException(status_code=400, detail="路径不合法或不存在")
    try:
        return {"success": True, "path": str(p), "report": json.load(open(p, encoding="utf-8"))}
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"读取失败: {e}")


@app.post("/api/corner_case/generate")
async def generate_corner_case(request: CornerCaseGenRequest):
    """基于轨迹编辑生成 corner case 交通事故场景。

    底层逻辑：修改相关 track 的轨迹（写入 TrackManager），与渲染/编辑管线统一。
    启用物理时基于包围盒计算接触距离，避免穿模，并返回碰撞关键帧分析。
    """
    tm = get_track_manager_or_404(request.scene_id)
    tm.fps = float(request.fps or 10.0)   # 朝向限速按帧率换算
    # 允许在原场景帧数之外生成：先把时间轴按需延长到覆盖请求范围
    _ensure_timeline(tm, frame_idx=int(request.start_frame) + int(request.num_frames) - 1)

    try:
        tm.push_history()
        # 备份持久合成物体（SAM3D 替换物体）当前轨迹：清除时只撤销事故改动
        tm.snapshot_synthetic_poses()
        result = corner_case.generate(
            tm,
            request.scenario_type,
            request.roles,
            request.start_frame,
            request.num_frames,
            request.intensity,
            enable_physics=request.enable_physics,
            fps=request.fps,
            sampling_params=request.sampling_params,
            sampling_seed=request.sampling_seed,
        )
        result["quality_report"] = quality_report.build_quality_report(
            tm,
            result,
            fps=request.fps,
        )
        if request.scene_id in studio_state["edit_history"]:
            studio_state["edit_history"][request.scene_id].append({
                "type": "corner_case",
                "scenario_type": request.scenario_type,
                "roles": request.roles,
                "timestamp": datetime.now().isoformat()
            })
        return {"success": True, **result}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class CornerCaseClearRequest(BaseModel):
    scene_id: str
    track_ids: List[int]


@app.post("/api/corner_case/clear")
async def clear_corner_case(request: CornerCaseClearRequest):
    """清除指定 track 的生成轨迹（恢复原始轨迹）。

    - 真实 track：清除位姿编辑，回到原始轨迹；
    - **SAM3D 替换物体等持久合成物体**：只把轨迹还原到替换时刻，**不删除物体本身**；
    - 事故自动合成的临时参与者（克隆外观）：整个移除。
    """
    tm = get_track_manager_or_404(request.scene_id)
    tm.push_history()
    cleared, removed_synth, restored = [], [], []
    for tid in request.track_ids:
        if tm.is_synthetic(tid):
            if tm.restore_synthetic_from_corner_backup(tid):
                # 持久物体（如被替换进场景的 SAM3D 物体）：保留模型，只还原轨迹
                restored.append(int(tid))
            else:
                tm.remove_synthetic_track(tid)
                removed_synth.append(int(tid))
        else:
            tm.clear_track_edits(tid)
            cleared.append(int(tid))
    return {"success": True, "cleared": cleared, "removed_synthetic": removed_synth,
            "restored": restored}


@app.get("/api/corner_case/types")
async def list_corner_case_types():
    """列出支持的 corner case 类型及其角色定义。"""
    return {"success": True, "types": corner_case.list_scenarios()}


class CollisionAnalysisRequest(BaseModel):
    scene_id: str
    track_a: int
    track_b: int
    start_frame: int = 0
    num_frames: int = 30
    fps: float = 10.0
    safety_margin: float = 1.5


@app.post("/api/corner_case/analyze_collision")
async def analyze_collision_endpoint(request: CollisionAnalysisRequest):
    """分析两个 track 之间的碰撞，识别**最晚反应关键帧**。

    用于自动驾驶系统评估：在 critical_frame 之前必须采取规避动作，
    否则在 collision_frame 将发生碰撞。返回逐帧距离曲线与关键帧信息。
    """
    tm = get_track_manager_or_404(request.scene_id)
    frames = list(range(request.start_frame, request.start_frame + request.num_frames + 1))

    info = corner_case.analyze_collision(
        tm, request.track_a, request.track_b, frames,
        fps=request.fps, safety_margin=request.safety_margin,
    )

    # 逐帧距离曲线（供前端绘制 + 关键帧高亮）
    distance_curve = []
    import numpy as _np
    for f in frames:
        pa = tm.get_track_pose(request.track_a, f)
        pb = tm.get_track_pose(request.track_b, f)
        if pa is None or pb is None:
            continue
        ca = _np.asarray(pa, dtype=_np.float32)[:3, 3]
        cb = _np.asarray(pb, dtype=_np.float32)[:3, 3]
        distance_curve.append({
            "frame_idx": int(f),
            "distance": float(_np.linalg.norm(ca - cb)),
        })

    return {
        "success": True,
        "track_a": request.track_a,
        "track_b": request.track_b,
        "collision_analysis": info,
        "distance_curve": distance_curve,
    }


@app.get("/api/corner_case/graph/{scene_id}")
async def get_scene_graph(scene_id: str, frame_idx: int = 0, window: int = 10):
    """返回动态物体关系图（节点 + 关系边 + 特征 + 冲突关键度 + 各事故类型的参与者提案）。

    这是"理解物体关系 → 生成多样事故"的基础：前端/批量生成都可以用它来挑选参与者。
    """
    tm = get_track_manager_or_404(scene_id)
    graph = scene_graph.SceneGraph(tm, fps=getattr(tm.renderer, "fps", 10.0) or 10.0)
    graph.build(frame_idx=frame_idx, window=window)
    proposals = {}
    for st in ("rear-end", "head-on", "intersection-tbone", "lane-change-cutin",
               "pedestrian-crossing", "hard-brake"):
        proposals[st] = graph.propose_participants(st, max_candidates=8)
    return {"success": True, "frame_idx": frame_idx, "graph": graph.to_dict(),
            "proposals": proposals}


class CornerCaseBatchRequest(BaseModel):
    scene_id: str
    scenario_types: List[str]
    num_per_type: int = 4
    seed: int = 0
    start_frame: int = 0
    num_frames: int = 30
    fps: float = 10.0
    intensity: float = 1.0
    enable_physics: bool = True
    frame_window: int = 10
    max_candidates_per_type: int = 8
    keep_only_valid: bool = False
    save: bool = True                       # 是否把实例落盘成数据集
    output_dir: Optional[str] = None        # 落盘根目录（默认 <repo>/output/corner_cases/<scene_id>/<run_name>）
    run_name: Optional[str] = None          # 本次批量运行名（默认时间戳）
    # —— 为每个实例渲染"与播放一致"的多帧视频（默认干净画面，不叠加包围盒/ID）——
    render_videos: bool = False
    video_draw_bboxes: bool = False
    video_draw_ids: bool = False
    video_draw_trajectories: bool = False
    video_fps: Optional[float] = None       # 默认与 fps 相同
    render_bev: bool = True                 # 额外输出俯视(BEV)示意图视频
    bev_size: int = 560
    bev_draw_ids: bool = False
    render_topdown: bool = True             # 额外输出"真渲染"的俯视视频
    topdown_size: int = 640
    topdown_span: float = 0.0              # >1 时手工指定俯视纵向视野（米）：想看清车/行人就调小
    use_edited_ego: bool = False            # 主车视角：沿"编辑后的主车轨迹"渲染
    annotate_trust: bool = True             # 给每条实例标注 per-frame 可信度（coverage / 预测 SSIM / 可信帧比例）
    trust_min: float = 0.55                 # 可信阈值：pred_ssim 低于它即记为首个不可信帧


@app.post("/api/corner_case/batch")
async def generate_corner_case_batch(request: CornerCaseBatchRequest):
    """批量生成多样化 Corner Case。

    流程（每个实例）：
      1. 用关系图按事故类型挑选候选参与者（多样性来源①：换参与者/换位置）；
      2. 用不同 seed 采样该类型的速度/车距/横向偏置/时序等参数（多样性来源②：换参数）；
      3. `corner_case.generate` 生成轨迹 → `quality_report` 打分；
      4. `undo()` 还原场景，保证批量之间互不污染，同时把每个实例的轨迹快照/质量写入清单。

    返回 manifest：按类型汇总，每个实例含 roles / 采样参数 / 关键帧 / 质量指标 / 是否通过。
    """
    tm = get_track_manager_or_404(request.scene_id)
    import random as _random
    fps = float(request.fps)
    tm.fps = fps
    # 批量生成必须从「干净场景」开始：否则之前手动生成/编辑的事故会残留下来，
    # 各事故类型都从同一个被污染的状态出发 → 生成出来的实例/视频看起来一模一样。
    _renderer_for_reset = studio_state["scenes"].get(request.scene_id)
    _had_edits = bool(getattr(tm, "track_edits", {})) or bool(getattr(tm, "synthetic_tracks", {}))
    if _renderer_for_reset is not None and _had_edits:
        tm = TrackManager(_renderer_for_reset)
        tm.fps = fps
        studio_state["track_managers"][request.scene_id] = tm
        print("[batch] 场景里有未清除的生成/编辑结果，已重置为干净场景后再批量生成")
    graph = scene_graph.SceneGraph(tm, fps=fps)
    graph.build(frame_idx=request.start_frame, window=request.frame_window)

    rng = _random.Random(request.seed)
    manifest: List[Dict[str, Any]] = []
    by_type: Dict[str, Dict[str, Any]] = {}

    # 落盘目录（把批量结果固化成可复现数据集；需要出视频时也要有目录）
    out_dir: Optional[Path] = None
    if request.save or request.render_videos or request.render_bev or request.render_topdown:
        repo_root = Path(__file__).resolve().parents[2]
        run_name = request.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
        base = Path(request.output_dir) if request.output_dir else (repo_root / "output" / "corner_cases" / request.scene_id)
        out_dir = base / run_name
        os.makedirs(out_dir, exist_ok=True)
    if (request.render_videos or request.render_bev or request.render_topdown) and out_dir is not None:
        os.makedirs(out_dir / "videos", exist_ok=True)
    renderer = get_scene_or_404(request.scene_id)

    # 生成器指纹：把"生成逻辑源码"的哈希写进数据里，保证数据可溯源（同一份代码 → 同一个指纹）
    def _generator_fingerprint() -> str:
        import hashlib
        h = hashlib.sha256()
        here = Path(__file__).resolve().parent
        for name in ("corner_case.py", "scenario_engine.py", "quality_report.py", "track_manager.py"):
            try:
                h.update(Path(here / name).read_bytes())
            except Exception:  # noqa: BLE001
                continue
        return h.hexdigest()[:16]

    generator_fp = _generator_fingerprint()

    # 可信度标注准备：trust 模型（用 output/trust 下已有的留出视角报告标定）+ 自车相机内参
    trust_model = None
    trust_cam = None
    if request.annotate_trust:
        try:
            _rows = trust_runtime.load_trust_rows(_trust_reports_auto())
            trust_model = trust_runtime.fit_trust_model(_rows)
            _cam = _frame_camera(renderer, request.start_frame)
            if _cam is not None:
                trust_cam = (np.asarray(_cam[1], dtype=np.float64), int(_cam[2]), int(_cam[3]))
            print(f"[batch] trust 模型：n={trust_model.get('n')} r2={trust_model.get('r2')} "
                  f"（仅 coverage 的 r2={trust_model.get('r2_coverage_only')}）")
        except Exception as e:  # noqa: BLE001
            print(f"[batch] trust 模型不可用：{e}")
            trust_model, trust_cam = None, None

    # 场景级可信域（一次横移 sweep，标注进每条实例：说明"这个场景允许自车偏离多少仍可信"）
    scene_envelope = None
    if trust_model is not None and trust_cam is not None:
        try:
            K_, W_, H_ = trust_cam
            scene_envelope = trust_runtime.lateral_sweep(
                _scene_path(request.scene_id), [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 6.0],
                K_, W_, H_, trust_model, min_trust=request.trust_min, frame=request.start_frame)
            scene_envelope = {"frame": scene_envelope["frame"],
                              "min_trust": scene_envelope["min_trust"],
                              "max_safe_lateral_m": scene_envelope["max_safe_lateral_m"],
                              "rows": [{"lateral_m": r["lateral_m"], "coverage": r["coverage"],
                                        "pred_ssim": r["pred_ssim"], "trusted": r["trusted"]}
                                       for r in scene_envelope["rows"]]}
            print(f"[batch] 场景可信域：max_safe_lateral={scene_envelope['max_safe_lateral_m']}m")
        except Exception as e:  # noqa: BLE001
            print(f"[batch] 场景可信域计算失败：{e}")
            scene_envelope = None

    for st in request.scenario_types:
        candidates = graph.propose_participants(st, max_candidates=request.max_candidates_per_type)
        type_instances = []
        if not candidates:
            by_type[st] = {"num_candidates": 0, "num_generated": 0, "num_valid": 0,
                           "reason": "关系图中没有匹配该事故类型的参与者对"}
            continue
        for k in range(request.num_per_type):
            roles = candidates[k % len(candidates)]
            seed_i = rng.randint(0, 10**9)
            tm.push_history()
            try:
                result = corner_case.generate(
                    tm, st, roles, request.start_frame, request.num_frames,
                    intensity=request.intensity, enable_physics=request.enable_physics,
                    fps=fps, sampling_seed=seed_i,
                )
                q = quality_report.build_quality_report(tm, result, fps=fps)
            except Exception as e:  # noqa: BLE001
                tm.undo()
                type_instances.append({"scenario_type": st, "roles": roles,
                                       "seed": seed_i, "error": str(e), "valid": False})
                continue

            # 这条实例「所有参与者」的世界坐标轨迹（含生成出来的新参与者）→
            # 单独落一份 world.json：④ CARLA / 其它工具可以直接吃它，完全不依赖内存里的场景状态
            inst_tracks = None
            try:
                _frames_i = list(range(request.start_frame, request.start_frame + request.num_frames + 1))
                _keep_i = ([tm.ego_track_id] if getattr(tm, "ego_track_id", None) is not None else []) + \
                          [int(x) for x in (result.get("affected_tracks") or [])] + \
                          [int(x) for x in (result.get("synthesized_tracks") or [])]
                inst_tracks = scenario_export.collect_tracks(
                    tm, _frames_i, ego_track=getattr(tm, "ego_track_id", None), only_tracks=_keep_i)
            except Exception as e:  # noqa: BLE001
                print(f"[batch] 收集实例轨迹失败 {st}/{k}: {e}")

            # 轨迹快照（生成后的位姿），供后续导出 spec/渲染使用
            snap = {}
            for tid in result.get("affected_tracks", []):
                snap[str(tid)] = {
                    str(f): np.asarray(tm.get_track_pose(tid, f), dtype=np.float32).tolist()
                    for f in range(request.start_frame, request.start_frame + request.num_frames + 1)
                    if tm.get_track_pose(tid, f) is not None
                }
            # 关系图特征（碰撞对），供后续 GNN 语料使用
            graph_feat = None
            ct = result.get("collision_tracks") or [int(rv) for rv in roles.values()]
            if len(ct) >= 2:
                pair = {int(x) for x in ct[:2]}
                for e in graph.edges:
                    if {int(x) for x in e["track_pair"]} == pair:
                        graph_feat = {
                            "relation": e["relation"],
                            "criticality": e["criticality"],
                            "ttc_s": e["ttc_s"],
                            "features": e["features"],
                        }
                        break
            # —— 可信度标注（在 undo 之前；用"事故生成后"的自车视角逐帧算 coverage/预测 SSIM）——
            trust_ann = None
            if trust_model is not None and trust_cam is not None:
                try:
                    K_, W_, H_ = trust_cam
                    poses = []
                    for i in range(int(request.num_frames)):
                        f = request.start_frame + i
                        P = None
                        if request.use_edited_ego and tm is not None and tm.ego_is_edited():
                            P = tm.get_ego_camera(f)
                        if P is None:
                            cam = _frame_camera(renderer, f)
                            P = None if cam is None else np.asarray(cam[0], dtype=np.float64)
                        if P is not None:
                            poses.append((f, np.asarray(P, dtype=np.float64)))
                    if poses:
                        trust_ann = trust_runtime.annotate_camera_track(
                            _scene_path(request.scene_id), poses, K_, W_, H_,
                            trust_model, min_trust=request.trust_min)
                        trust_ann["model_r2"] = trust_model.get("r2")
                except Exception as e:  # noqa: BLE001
                    print(f"[batch] trust 标注失败 {st}-{k}: {e}")
                    trust_ann = None

            # 为该校例渲染视频（必须在 undo 之前）：主车视角 + 俯视(BEV)视角
            video_path = None
            bev_path = None
            top_path = None
            if request.render_videos and out_dir is not None:
                iid = f"{st}-{k:03d}-{seed_i}"
                try:
                    video_path = str(out_dir / "videos" / f"{iid}.mp4")
                    render_sequence_video(
                        renderer, tm, request.start_frame, request.num_frames,
                        float(request.video_fps or fps), video_path,
                        request.video_draw_bboxes, request.video_draw_ids,
                        request.video_draw_trajectories,
                        use_edited_ego=request.use_edited_ego)
                except Exception as e:  # noqa: BLE001
                    print(f"[batch] 主视角视频失败 {iid}: {e}")
                    video_path = None
                if request.render_bev:
                    try:
                        bev_path = str(out_dir / "videos" / f"{iid}_bev.mp4")
                        render_bev_sequence_video(
                            tm, renderer, request.start_frame, request.num_frames,
                            float(request.video_fps or fps), bev_path,
                            size=request.bev_size,
                            highlight_tracks=result.get("collision_tracks"),
                            collision_frame=result.get("collision_frame"),
                            draw_ids=request.bev_draw_ids)
                    except Exception as e:  # noqa: BLE001
                        print(f"[batch] 俯视视频失败 {iid}: {e}")
                        bev_path = None
                if request.render_topdown:
                    try:
                        top_path = str(out_dir / "videos" / f"{iid}_top.mp4")
                        render_topdown_sequence_video(
                            renderer, tm, request.start_frame, request.num_frames,
                            float(request.video_fps or fps), top_path,
                            size=request.topdown_size,
                            focus_tracks=(result.get("affected_tracks")
                                          or ([] if not roles else list(roles.values()))),
                            span_override=float(getattr(request, "topdown_span", 0.0) or 0.0))
                    except Exception as e:  # noqa: BLE001
                        print(f"[batch] 俯视渲染视频失败 {iid}: {e}")
                        top_path = None

            inst = {
                "instance_id": f"{st}-{k:03d}-{seed_i}",
                "scenario_type": st,
                "roles": {str(rk): int(rv) for rk, rv in roles.items()},
                "seed": seed_i,
                "sampling_params": result.get("sampling_params") or {},
                "collision_frame": result.get("collision_frame"),
                "critical_frame": result.get("critical_frame"),
                "valid": bool(q.get("valid")),
                "quality": {
                    "min_distance": q.get("min_distance"),
                    "ttc_at_critical": q.get("ttc_at_critical"),
                    "max_speed": _max_metric(q, "max_speed"),
                    "max_accel": q.get("max_accel"),
                    "max_jerk": _max_metric(q, "max_jerk"),
                    "max_yaw_rate": q.get("max_yaw_rate"),
                    "max_step_distance": _max_metric(q, "max_step_distance"),
                    "bbox_penetration": q.get("bbox_penetration"),
                    "blocking_checks": [c["name"] for c in q.get("checks", []) if c.get("passed") is False],
                    "failing_metrics": _failing_metrics(q),
                    "thresholds": q.get("thresholds"),
                },
                "graph_features": graph_feat,
                "trust": trust_ann,
                "trust_envelope": scene_envelope,
                "generator_fingerprint": generator_fp,
                "video_path": video_path,
                "bev_video_path": bev_path,
                "topdown_video_path": top_path,
                "trajectory_snapshot": snap,
                # 下面这几个让「一条实例」可以独立于内存场景被复用（③ 闭环 / ④ CARLA 都用得上）
                "start_frame": request.start_frame,
                "num_frames": request.num_frames,
                "fps": fps,
                "num_tracks": len(inst_tracks) if inst_tracks else 0,
                "world_json": None,
            }
            if inst_tracks and out_dir is not None:
                try:
                    _wj = out_dir / f"{inst['instance_id']}.world.json"
                    scenario_export.export_world_json(inst_tracks, str(_wj), dt=1.0 / fps)
                    inst["world_json"] = str(_wj)
                except Exception as e:  # noqa: BLE001
                    print(f"[batch] 实例 world.json 落盘失败 {inst['instance_id']}: {e}")
            # 内容哈希：spec（场景/类型/角色/seed/参数/帧窗）+ 生成后的逐帧位姿
            # → 下游可以据此校验"这份数据是不是用同一份输入算出来的"（可复现/可审计）
            try:
                import hashlib
                h = hashlib.sha256()
                h.update(json.dumps({
                    "scene_id": request.scene_id, "scenario_type": st, "roles": inst["roles"],
                    "seed": seed_i, "sampling_params": inst["sampling_params"],
                    "start_frame": request.start_frame, "num_frames": request.num_frames,
                    "fps": fps, "generator": generator_fp,
                }, sort_keys=True, ensure_ascii=False).encode("utf-8"))
                h.update(json.dumps(inst["trajectory_snapshot"], sort_keys=True).encode("utf-8"))
                inst["spec_hash"] = h.hexdigest()[:32]
            except Exception:  # noqa: BLE001
                inst["spec_hash"] = None

            # 落盘（keep_only_valid 时只存通过质量门的实例）
            if out_dir is not None and (not request.keep_only_valid or inst.get("valid")):
                fname = f"{inst['instance_id']}.json"
                with open(out_dir / fname, "w", encoding="utf-8") as fp:
                    json.dump(inst, fp, ensure_ascii=False, indent=2)
                inst["file"] = fname
            type_instances.append(inst)
            tm.undo()   # 还原，保持场景不因批量而堆叠

        manifest.extend(type_instances)
        by_type[st] = {
            "num_candidates": len(candidates),
            "num_generated": len(type_instances),
            "num_valid": sum(1 for i in type_instances if i.get("valid")),
            "participants": "pair" if any(len(c) >= 2 for c in candidates) else "anchor+auto",
        }

    resp: Dict[str, Any] = {
        "success": True,
        "total": len(manifest),
        "num_valid": sum(1 for i in manifest if i.get("valid")),
        "by_type": by_type,
        "manifest": manifest,
        # 批量前是否把场景重置成了干净状态（之前有生成/编辑残留时才会为 True）
        "reset_scene": bool(_had_edits),
    }
    if out_dir is not None:
        manifest_path = out_dir / "manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as fp:
            json.dump({
                "scene_id": request.scene_id,
                "run_name": run_name,
                "seed": request.seed,
                "start_frame": request.start_frame,
                "num_frames": request.num_frames,
                "fps": fps,
                "scenario_types": request.scenario_types,
                "generator_fingerprint": generator_fp,
            "scene_trust_envelope": scene_envelope,
            "summary": {"total": len(manifest),
                            "num_valid": sum(1 for i in manifest if i.get("valid")),
                            "by_type": by_type},
                "instances": manifest,
            }, fp, ensure_ascii=False, indent=2)
        resp["output_dir"] = str(out_dir)
        resp["manifest_path"] = str(manifest_path)
        resp["num_saved"] = sum(1 for i in manifest if i.get("file"))
        resp["generator_fingerprint"] = generator_fp
        if scene_envelope is not None:
            resp["scene_trust_envelope"] = {k: scene_envelope[k] for k in ("frame", "min_trust", "max_safe_lateral_m")}
        _tr = [i.get("trust") or {} for i in manifest]
        _tr = [t for t in _tr if t.get("available")]
        if _tr:
            resp["trust_summary"] = {
                "num_annotated": len(_tr),
                "coverage_mean": round(float(np.mean([t["coverage_mean"] for t in _tr])), 4),
                "pred_ssim_mean": round(float(np.mean([t["pred_ssim_mean"] for t in _tr])), 4),
                "untrusted_instances": sum(1 for t in _tr if (t.get("trusted_ratio") or 1.0) < 1.0),
                "model_r2": trust_model.get("r2") if trust_model else None,
            }
        n_vid = sum(1 for i in manifest if i.get("video_path"))
        n_bev = sum(1 for i in manifest if i.get("bev_video_path"))
        n_top = sum(1 for i in manifest if i.get("topdown_video_path"))
        resp["num_videos"] = n_vid
        resp["num_bev_videos"] = n_bev
        resp["num_topdown_videos"] = n_top
        if n_vid or n_bev or n_top:
            resp["video_dir"] = str(out_dir / "videos")
    return resp


class CornerCaseVideoRequest(BaseModel):
    scene_id: str
    start_frame: int = 0
    num_frames: int = 30
    fps: float = 10.0
    draw_bboxes: bool = False
    draw_ids: bool = False
    draw_trajectories: bool = False
    output_dir: Optional[str] = None
    name: Optional[str] = None
    highlight_track_id: Optional[int] = None
    render_bev: bool = True
    bev_size: int = 560
    bev_draw_ids: bool = False
    render_topdown: bool = True
    topdown_size: int = 640
    topdown_span: float = 0.0              # >1 时手工指定俯视纵向视野（米）：想看清车/行人就调小
    use_edited_ego: bool = True             # 主车视角默认跟随"编辑后的主车轨迹"（未编辑则同原相机）


@app.post("/api/corner_case/video")
async def corner_case_video(request: CornerCaseVideoRequest):
    """把**当前场景状态**（含已生成/编辑的事故轨迹、合成参与者）渲染成一段可播放的 mp4。

    渲染路径与 2D 播放完全一致（自车视角 + 包围盒/ID/轨迹叠加）。
    """
    renderer = get_scene_or_404(request.scene_id)
    tm = None if renderer.static_only else get_track_manager_or_404(request.scene_id)
    try:
        # 把帧范围钳制到场景真实帧数内
        total = int(renderer.frame_count())
        start = int(max(0, min(request.start_frame, max(0, total - 1))))
        num = int(max(1, min(request.num_frames, total - start)))
        repo_root = Path(__file__).resolve().parents[2]
        out_dir = Path(request.output_dir) if request.output_dir else (
            repo_root / "output" / "corner_cases" / request.scene_id / "videos")
        name = request.name or f"case_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        out_path = out_dir / f"{name}.mp4"
        path = render_sequence_video(
            renderer, tm, start, num, request.fps,
            str(out_path), request.draw_bboxes, request.draw_ids,
            request.draw_trajectories, request.highlight_track_id,
            use_edited_ego=request.use_edited_ego)
        if not path:
            raise RuntimeError("没有可渲染的帧")
        bev_path = None
        if request.render_bev:
            try:
                bev_path = str(out_dir / f"{name}_bev.mp4")
                render_bev_sequence_video(
                    tm, renderer, start, num, request.fps, bev_path,
                    size=request.bev_size, draw_ids=request.bev_draw_ids)
            except Exception as e:  # noqa: BLE001
                print(f"[video] 俯视示意图失败: {e}")
        top_path = None
        if request.render_topdown:
            try:
                top_path = str(out_dir / f"{name}_top.mp4")
                focus = None
                if tm is not None:
                    try:
                        objs = tm.get_frame_objects(start)
                        # 优先框"事故参与者"（合成/被编辑的物体），否则退回全部物体
                        focus = [int(o["track_id"]) for o in objs
                                 if o.get("synthetic") or o.get("edited")]
                        if not focus:
                            focus = [int(o["track_id"]) for o in objs]
                        if len(focus) > 8:
                            focus = focus[:8]
                    except Exception:  # noqa: BLE001
                        focus = None
                render_topdown_sequence_video(
                    renderer, tm, start, num, request.fps, top_path,
                    size=request.topdown_size, focus_tracks=focus,
                    span_override=float(getattr(request, "topdown_span", 0.0) or 0.0))
            except Exception as e:  # noqa: BLE001
                print(f"[video] 俯视渲染失败: {e}")
        return {"success": True, "video_path": str(path), "bev_video_path": bev_path,
                "topdown_video_path": top_path,
                "start_frame": start, "num_frames": num, "fps": float(request.fps)}
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"渲染视频失败: {e}")


@app.get("/api/corner_case/video_file")
async def corner_case_video_file(path: str, request: Request):
    """受限于 output/ 目录的 mp4 访问入口，供前端 <video> 直接播放（支持 Range，可拖动进度条）。"""
    repo_root = Path(__file__).resolve().parents[2]
    allowed = (repo_root / "output").resolve()
    p = Path(path).resolve()
    if not str(p).startswith(str(allowed)) or not p.exists() or p.suffix.lower() != ".mp4":
        raise HTTPException(status_code=404, detail="视频不存在或不在允许的目录内")
    file_size = p.stat().st_size
    rng = request.headers.get("range") or request.headers.get("Range")
    if rng and rng.startswith("bytes="):
        try:
            spec = rng.split("=", 1)[1].split(",")[0].strip()
            start_s, _, end_s = spec.partition("-")
            start = int(start_s) if start_s else 0
            end = int(end_s) if end_s else file_size - 1
        except ValueError:
            start, end = 0, file_size - 1
        start = max(0, min(start, max(0, file_size - 1)))
        end = max(start, min(end, max(0, file_size - 1)))
        length = end - start + 1
        with open(p, "rb") as f:
            f.seek(start)
            data = f.read(length)
        return Response(content=data, status_code=206, media_type="video/mp4",
                        headers={"Content-Range": f"bytes {start}-{end}/{file_size}",
                                 "Accept-Ranges": "bytes", "Content-Length": str(length)})
    return FileResponse(str(p), media_type="video/mp4", filename=p.name,
                        headers={"Accept-Ranges": "bytes"})


class GNNCollectRequest(BaseModel):
    scene_id: str
    scenario_types: List[str]
    num_per_type: int = 40
    seed: int = 0
    start_frame: int = 0
    num_frames: int = 30
    fps: float = 10.0
    output_path: Optional[str] = None   # 保存 .npz 的路径（默认 output/gnn_dataset/<scene>_<n>.npz）


_SEV_STR = {"near_miss": 1, "low": 1, "medium": 2, "high": 3}


def _gnn_sample_features(graph, roles, result):
    """从关系图 + 生成结果拼一个定长样本特征向量（22 维），布局见 gnn_model.py。"""
    tids = sorted(int(v) for v in roles.values())
    if len(tids) != 2:
        return None
    node_map = {int(n["track_id"]): n for n in graph.nodes}
    if tids[0] not in node_map or tids[1] not in node_map:
        return None
    edge_feat = None
    for e in graph.edges:
        if {int(x) for x in e["track_pair"]} == set(tids):
            edge_feat = e["features"]
            break
    if edge_feat is None:
        return None

    def node_feat(tid):
        n = node_map[tid]
        dims = n.get("dimensions") or [1.0, 1.0, 1.0]
        try:
            vol = abs(float(dims[0]) * float(dims[1]) * float(dims[2]))
        except (TypeError, ValueError):
            vol = 1.0
        cls = n.get("class") or "other"
        return [
            float(n.get("speed") or 0.0),
            float(math.log1p(vol)),
            1.0 if cls in ("vehicle", "large_vehicle") else 0.0,
            1.0 if cls == "pedestrian" else 0.0,
        ]

    sp = result.get("sampling_params") or {}
    params = [
        float(sp.get("relative_speed_mps", sp.get("pedestrian_speed_mps", 0.0)) or 0.0),
        float(sp.get("initial_gap_m", 0.0) or 0.0),
        float(sp.get("lateral_offset_m", 0.0) or 0.0),
        float(sp.get("reaction_delay_s", 0.0) or 0.0),
    ]
    feat = node_feat(tids[0]) + node_feat(tids[1]) + [float(v) for v in edge_feat] + params
    return feat


@app.post("/api/gnn/collect")
async def gnn_collect(request: GNNCollectRequest):
    """生成带标签的 GNN 训练语料：关系图特征 → (是否碰撞, 严重度)。

    对每个候选参与者对，用不同 seed / 不同 collision_severity 采样物理参数生成事故，
    记录"生成前的关系图特征"（输入）与"生成后的碰撞结果/严重度"（标签），
    存成 .npz，供 `train_gnn.py` 离线训练。
    """
    tm = get_track_manager_or_404(request.scene_id)
    import random as _random
    fps = float(request.fps)
    graph = scene_graph.SceneGraph(tm, fps=fps)
    graph.build(frame_idx=request.start_frame, window=request.frame_window if hasattr(request, "frame_window") else 10)

    rng = _random.Random(request.seed)
    X, y_col, y_sev, metas = [], [], [], []
    for st in request.scenario_types:
        candidates = graph.propose_participants(st, max_candidates=10)
        if not candidates:
            continue
        for k in range(request.num_per_type):
            roles = candidates[k % len(candidates)]
            sev_dial = rng.choice(["near_miss", "minor", "severe"])
            seed_i = rng.randint(0, 10**9)
            tm.push_history()
            try:
                result = corner_case.generate(
                    tm, st, roles, request.start_frame, request.num_frames,
                    intensity=1.0, enable_physics=True, fps=fps,
                    sampling_params={"collision_severity": sev_dial}, sampling_seed=seed_i,
                )
            except Exception:  # noqa: BLE001
                tm.undo()
                continue
            tm.undo()

            feat = _gnn_sample_features(graph, roles, result)
            if feat is None:
                continue
            collided = 1 if result.get("collision_frame") is not None else 0
            sev_str = (result.get("collision_analysis") or {}).get("collision_severity", "low")
            sev = 0 if not collided else _SEV_STR.get(sev_str, 1)
            X.append(feat)
            y_col.append(collided)
            y_sev.append(sev)
            metas.append({"scenario_type": st, "roles": roles, "seed": seed_i,
                          "severity_dial": sev_dial})

    if not X:
        return {"success": False, "detail": "没有生成出任何样本（该场景/类型无可用的参与者对）"}

    X = np.asarray(X, dtype=np.float32)
    y_col = np.asarray(y_col, dtype=np.int64)
    y_sev = np.asarray(y_sev, dtype=np.int64)

    out_path = request.output_path
    if not out_path:
        repo_root = Path(__file__).resolve().parents[2]
        d = repo_root / "output" / "gnn_dataset"
        os.makedirs(d, exist_ok=True)
        out_path = str(d / f"{request.scene_id}_{len(X)}.npz")
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    np.savez(out_path, X=X, y_col=y_col, y_sev=y_sev)

    # 特征均值/标准差（供归一化复现）
    return {
        "success": True,
        "num_samples": int(len(X)),
        "collision_rate": float(y_col.mean()),
        "severity_dist": {int(s): int((y_sev == s).sum()) for s in sorted(set(y_sev.tolist()))},
        "feature_dim": int(X.shape[1]),
        "output_path": out_path,
        "sample_metas": metas[:10],
    }


@app.post("/api/export/trajectory")
async def export_trajectory(scene_id: str, output_path: str):
    """导出轨迹规格"""
    renderer = get_scene_or_404(scene_id)
    
    try:
        spec = renderer.export_trajectory_controller_spec(output_path)
        return {"success": True, "output_path": output_path, "spec": spec}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/export/scene_spec")
async def export_scene_spec(request: ExportRequest):
    """导出场景编辑规格"""
    renderer = get_scene_or_404(request.scene_id)
    
    try:
        output_path = renderer.export_scene_edit_spec(
            request.output_path,
            name=f"scene_{request.scene_id}",
            scenario_type="custom",
            actors=renderer.trajectory_controller.get_object_ids(),
            start_frame=0,
            duration=100
        )
        return {"success": True, "output_path": request.output_path}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/history/{scene_id}")
async def get_edit_history(scene_id: str, limit: int = 50):
    """获取编辑历史"""
    if scene_id not in studio_state["edit_history"]:
        return {"history": []}
    
    history = studio_state["edit_history"][scene_id][-limit:]
    return {"history": history, "scene_id": scene_id}


# ==================== 3D 视图 API ====================

def _matrix_to_list(pose):
    """将 torch/np 的 4x4 矩阵转换为嵌套 list。"""
    if isinstance(pose, torch.Tensor):
        return pose.detach().cpu().numpy().astype(float).tolist()
    return np.asarray(pose, dtype=float).tolist()


@app.get("/api/scene3d/{scene_id}/frame/{frame_idx}")
async def get_scene3d(scene_id: str, frame_idx: int):
    """获取某一帧的 3D 场景元信息（用于自由视角浏览/编辑）。

    采用 **服务端自由视角渲染** 方案：浏览器只用轨道相机控制视角，
    实际画面由后端用 gsplat 按该视角渲染。

    物体一律以 **track_id** 标识（跨帧稳定），避免不同帧 raw object_id 不连续
    导致"换帧后编辑到别的物体"。
    """
    renderer = get_scene_or_404(scene_id)

    objects = []
    if not renderer.static_only:
        tm = get_track_manager_or_404(scene_id)
        for obj in tm.get_frame_objects(frame_idx):
            objects.append({
                "track_id": obj["track_id"],
                "object_id": obj["track_id"],   # 兼容旧前端字段
                "raw_object_id": obj["raw_object_id"],
                "type": obj["type"],
                "pose_world": obj["pose_world"],
                "center": obj["center"],
                "dimensions": obj["dimensions"],
                "edited": obj["edited"],
                "synthetic": obj.get("synthetic", False),
            })

    # 相机参数
    camera = None
    scene_center = [0.0, 0.0, 0.0]
    cam_flat = renderer.flat_index(frame_idx, 0)
    ego_path = os.path.join(renderer.ego_dir, f"frame_{cam_flat:04d}_ego.json")
    if os.path.exists(ego_path):
        with open(ego_path, "r") as f:
            ego_data = json.load(f)
        camera = {
            "extrinsics_world": ego_data.get("camera_extrinsics_world"),
            "intrinsics": ego_data.get("camera_intrinsics"),
            "width": ego_data.get("camera", {}).get("width"),
            "height": ego_data.get("camera", {}).get("height"),
        }
        ext = ego_data.get("camera_extrinsics_world")
        if ext is not None:
            ext = np.asarray(ext, dtype=float)
            scene_center = [float(ext[0][3]), float(ext[1][3]), float(ext[2][3])]

    if objects:
        c = np.mean([o["center"] for o in objects], axis=0)
        scene_center = [float(c[0]), float(c[1]), float(c[2])]

    return {
        "success": True,
        "scene_id": scene_id,
        "frame_idx": frame_idx,
        "objects": objects,
        "camera": camera,
        "scene_center": scene_center,
    }


@app.get("/api/tracks/{scene_id}")
async def list_tracks(scene_id: str):
    """列出场景内所有动态物体 track 概要。"""
    tm = get_track_manager_or_404(scene_id)
    return {"success": True, "tracks": tm.list_tracks()}


@app.get("/api/tracks/{scene_id}/{track_id}/trajectory")
async def get_track_trajectory(scene_id: str, track_id: int):
    """获取某 track 的逐帧运动轨迹（世界坐标中心点序列，含编辑结果）。"""
    tm = get_track_manager_or_404(scene_id)
    traj = tm.get_track_trajectory(track_id, use_edits=True)
    raw_traj = tm.get_track_trajectory(track_id, use_edits=False)
    return {
        "success": True,
        "track_id": int(track_id),
        "trajectory": traj,
        "original_trajectory": raw_traj,
        "edited": tm.is_track_edited(track_id),
    }


class FreeViewRenderRequest(BaseModel):
    scene_id: str
    frame_idx: int
    # 相机：用 c2w (camera-to-world) 4x4 矩阵指定自由视角
    c2w: List[List[float]]
    fov_y: Optional[float] = None      # 垂直 FOV（度）。给定则据此计算内参
    width: int = 960
    height: int = 540
    draw_bboxes: bool = False
    draw_ids: bool = False
    highlight_track_id: Optional[int] = None
    draw_trajectories: bool = False
    # 三维极端天气：不是二维贴图；服务端在相机视锥内生成3D粒子并投影渲染
    weather: Optional[Dict[str, Any]] = None
    # 拖动编辑过程中的临时位姿（未提交），用于实时预览：{track_id, pose_matrix}
    live_track_id: Optional[int] = None
    live_pose_matrix: Optional[List[List[float]]] = None


def _fov_to_intrinsics(fov_y_deg: float, width: int, height: int, device: str):
    """由垂直 FOV 生成针孔相机内参 K (3x3)。"""
    fov_y = np.deg2rad(fov_y_deg)
    fy = (height / 2.0) / np.tan(fov_y / 2.0)
    fx = fy  # 方形像素
    cx, cy = width / 2.0, height / 2.0
    K = torch.tensor([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
                     device=device, dtype=torch.float32)
    return K


def _apply_atmospheric_extinction(image: np.ndarray, weather: Dict[str, Any]) -> np.ndarray:
    """基于能见度的体积雾/空气散射近似。不是贴图，而是对整幅渲染做物理雾化。"""
    kind = weather.get("type", "clear")
    intensity = float(weather.get("intensity", 0.0))
    if kind == "clear" or intensity <= 0.0:
        return image
    visibility = float(weather.get("visibility", 80.0))
    fog_alpha = float(np.clip(intensity * (1.0 - np.exp(-65.0 / max(1.0, visibility))), 0.0, 0.9))
    if kind in ("snow", "blizzard"):
        fog_color = np.array([224, 230, 234], dtype=np.float32)
    elif kind in ("rain", "storm"):
        fog_color = np.array([96, 106, 116], dtype=np.float32)
    else:
        fog_color = np.array([166, 170, 174], dtype=np.float32)

    img = image.astype(np.float32)
    out = img * (1.0 - fog_alpha) + fog_color * fog_alpha

    # 垂直方向的空气透视：远处/天空更雾、更低对比；不是贴图，是物理能见度近似。
    h = image.shape[0]
    y = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None, None]
    sky_haze = (1.0 - y) ** 1.6 * fog_alpha * (0.45 if kind in ("snow", "blizzard", "fog") else 0.28)
    out = out * (1.0 - sky_haze) + fog_color * sky_haze

    if kind in ("storm", "rain"):
        out *= (1.0 - (0.18 if kind == "rain" else 0.32) * intensity)  # 暴雨/暴风雨低照度
        # 微弱蓝灰色调，更接近暴雨天气的白平衡
        out[..., 2] *= 1.04
        out[..., 0] *= 0.96
    return np.clip(out, 0, 255).astype(np.uint8)


def _stable_weather_seed(kind: str) -> int:
    seeds = {"rain": 1103, "storm": 2207, "snow": 3301, "blizzard": 4409, "fog": 5501}
    return seeds.get(kind, 9109)


def _render_3d_weather_particles(image: np.ndarray, c2w: torch.Tensor, K: torch.Tensor,
                                 frame_idx: int, weather: Dict[str, Any]) -> np.ndarray:
    """持久3D粒子天气场：粒子在相机视锥内循环包裹，避免播放时忽有忽无。"""
    kind = weather.get("type", "clear")
    intensity = float(weather.get("intensity", 0.0))
    if kind == "clear" or intensity <= 0.0:
        return image
    h, w = image.shape[:2]
    scale = w * h / float(960 * 540)
    base_count = {"rain": 1800, "storm": 3200, "snow": 1400, "blizzard": 2800, "fog": 0}.get(kind, 0)
    n = int(base_count * np.clip(intensity, 0.05, 2.0) * scale)
    if n <= 0:
        return image

    rng = np.random.default_rng(_stable_weather_seed(kind) + int(w) * 3 + int(h) * 5)
    fx, fy = float(K[0, 0].item()), float(K[1, 1].item())
    z_near, z_far = 0.8, float(weather.get("depth", 85.0))

    # 使用对数深度：近景粒子更多，远景形成密度层；全部粒子跨帧持久，仅位置包裹循环。
    z = np.exp(rng.uniform(np.log(z_near), np.log(z_far), n)).astype(np.float32)
    margin = 96 if kind in ("rain", "storm") else 64
    u0 = rng.uniform(-margin, w + margin, n).astype(np.float32)
    v0 = rng.uniform(-margin, h + margin, n).astype(np.float32)
    phase = rng.uniform(0.0, 1.0, n).astype(np.float32)

    wind = weather.get("wind", [0.0, 0.0]) or [0.0, 0.0]
    wx = float(wind[0]) if len(wind) > 0 else 0.0
    wy = float(wind[1]) if len(wind) > 1 else 0.0
    t = float(frame_idx)
    depth_speed = np.clip((z_far / z) ** 0.35, 0.55, 3.5)

    if kind in ("rain", "storm"):
        fall_px = (38.0 if kind == "rain" else 62.0) * (0.65 + intensity) * depth_speed
        side_px = (wx * 0.9 + 8.0 * np.sin(phase * 6.283)) * depth_speed
        u = ((u0 + t * side_px * 0.35 + margin) % (w + 2 * margin)) - margin
        v = ((v0 + t * fall_px + wy * t * 0.25 + margin) % (h + 2 * margin)) - margin
    else:
        fall_px = (4.0 if kind == "snow" else 9.0) * (0.6 + intensity) * depth_speed
        swirl = np.sin(t * 0.18 + phase * 6.283) * (10.0 if kind == "snow" else 24.0)
        u = ((u0 + wx * t * 0.18 * depth_speed + swirl + margin) % (w + 2 * margin)) - margin
        v = ((v0 + t * fall_px + wy * t * 0.18 + margin) % (h + 2 * margin)) - margin

    valid = (u >= -margin) & (u < w + margin) & (v >= -margin) & (v < h + margin)
    u, v, z = u[valid], v[valid], z[valid]
    near = np.clip(1.0 - z / z_far, 0.08, 1.0)
    order = np.argsort(z)[::-1]  # 远到近绘制，近景粒子覆盖远景
    u, v, near, z = u[order], v[order], near[order], z[order]

    canvas = image.copy()
    overlay = np.zeros_like(canvas)

    if kind in ("rain", "storm"):
        color_far = np.array([115, 130, 145], dtype=np.float32)
        color_near = np.array([225, 235, 245], dtype=np.float32)
        for ui, vi, a, zi in zip(u.astype(np.int32), v.astype(np.int32), near, z):
            streak = int(np.clip((18 if kind == "rain" else 30) * (0.5 + intensity) * (0.35 + a), 6, 58))
            dx = int(np.clip(wx * (0.45 + a), -28, 28))
            thickness = 1 if zi > 8 else 2
            col = tuple(np.clip(color_far * (1 - a) + color_near * a, 0, 255).astype(np.uint8).tolist())
            cv2.line(overlay, (ui - dx // 3, vi - streak // 2), (ui + dx, vi + streak), col, thickness, cv2.LINE_AA)
        alpha = float(np.clip(0.32 + 0.20 * intensity, 0.28, 0.72))
        canvas = cv2.addWeighted(overlay, alpha, canvas, 1.0, 0)
        # 近景水汽/水膜轻微模糊，降低锐利度，更接近暴雨镜头。
        if kind == "storm" and intensity > 0.8:
            blur = cv2.GaussianBlur(canvas, (0, 0), sigmaX=0.45 + 0.25 * intensity)
            canvas = cv2.addWeighted(canvas, 0.82, blur, 0.18, 0)
    elif kind in ("snow", "blizzard"):
        for ui, vi, a, zi in zip(u.astype(np.int32), v.astype(np.int32), near, z):
            r = int(np.clip((1.2 if kind == "snow" else 1.8) + a * (3.2 if kind == "snow" else 5.0) * intensity, 1, 7))
            shade = int(np.clip(190 + 60 * a, 190, 250))
            cv2.circle(overlay, (ui, vi), r, (shade, shade, shade), -1, cv2.LINE_AA)
            if kind == "blizzard" and r >= 3:
                cv2.line(overlay, (ui - int(wx * 0.25), vi - 1), (ui + int(wx * 0.45), vi + 1), (shade, shade, shade), 1, cv2.LINE_AA)
        alpha = float(np.clip(0.38 + 0.18 * intensity, 0.35, 0.78))
        canvas = cv2.addWeighted(overlay, alpha, canvas, 1.0, 0)
        if kind == "blizzard" and intensity > 0.9:
            blur = cv2.GaussianBlur(canvas, (0, 0), sigmaX=0.25 + 0.18 * intensity)
            canvas = cv2.addWeighted(canvas, 0.88, blur, 0.12, 0)
    return np.clip(canvas, 0, 255).astype(np.uint8)


@app.post("/api/render/freeview")
async def render_freeview(request: FreeViewRenderRequest):
    """从任意自由视角渲染 4DGS 当前帧（服务端 gsplat 渲染）。

    返回 base64 PNG。相机由 c2w 给定，内参可由 fov_y 推导或使用自车内参缩放。
    """
    renderer = get_scene_or_404(request.scene_id)

    try:
        device = renderer.device
        try:
            _tm0 = None if renderer.static_only else get_track_manager_or_404(request.scene_id)
            _ensure_timeline(_tm0, frame_idx=request.frame_idx)
        except Exception:  # noqa: BLE001
            pass

        c2w = torch.tensor(request.c2w, device=device, dtype=torch.float32)
        if c2w.shape == (3, 4):
            c2w = torch.cat([c2w, torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=device, dtype=c2w.dtype)], dim=0)

        # 内参：优先用 fov_y；否则把自车内参按分辨率缩放
        if request.fov_y is not None:
            K = _fov_to_intrinsics(request.fov_y, request.width, request.height, device)
        else:
            ego_path = os.path.join(renderer.ego_dir, f"frame_{renderer.flat_index(request.frame_idx, 0):04d}_ego.json")
            with open(ego_path, "r") as f:
                ego_data = json.load(f)
            K0 = np.asarray(ego_data["camera_intrinsics"], dtype=np.float32)
            W0 = ego_data["camera"]["width"]
            H0 = ego_data["camera"]["height"]
            sx = request.width / float(W0)
            sy = request.height / float(H0)
            K0[0, 0] *= sx; K0[0, 2] *= sx
            K0[1, 1] *= sy; K0[1, 2] *= sy
            K = torch.tensor(K0, device=device, dtype=torch.float32)

        # 物体覆盖（基于 track 的编辑结果 + 删除）
        tm = None if renderer.static_only else get_track_manager_or_404(request.scene_id)
        object_overrides = tm.build_object_overrides(request.frame_idx) if tm else {}
        # 合成参与者（自动生成的事故参与者，克隆已有物体的高斯）
        extra_objects = tm.build_extra_objects(request.frame_idx) if tm else []

        # 实时预览：拖动中的临时位姿（未提交）覆盖该 track 在本帧的 raw object
        if tm and request.live_track_id is not None and request.live_pose_matrix is not None:
            raw_id = tm.get_raw_object_id(request.live_track_id, request.frame_idx)
            if raw_id is not None:
                live_pose = torch.tensor(request.live_pose_matrix, device=device).float()
                object_overrides[raw_id] = live_pose
            elif tm.is_synthetic(request.live_track_id):
                # 拖动合成参与者：动态物体高斯为局部坐标，直接使用预览目标位姿。
                live_pose = np.asarray(request.live_pose_matrix, dtype=np.float32)
                for ex in extra_objects:
                    if ex.get("synth_track_id") == request.live_track_id:
                        ex["transform"] = torch.tensor(live_pose, device=device).float()

        image = renderer._render_frame_with_object_overrides(
            request.frame_idx,
            object_overrides,
            c2w_override=c2w,
            K_override=K,
            width_override=request.width,
            height_override=request.height,
            extra_objects=extra_objects,
        )

        # 可选：绘制包围盒/高亮（按 track_id）
        if tm and request.draw_bboxes:
            viewmat = torch.inverse(c2w)
            for obj in tm.get_frame_objects(request.frame_idx):
                tid = obj["track_id"]
                # 若该 track 正在拖动预览，用临时位姿画框
                if tid == request.live_track_id and request.live_pose_matrix is not None:
                    pose = torch.tensor(request.live_pose_matrix, device=device).float()
                else:
                    pose = torch.tensor(obj["pose_world"], device=device).float()
                corners_2d = renderer._get_bbox_corners_2d(
                    pose, obj["dimensions"], viewmat, K, request.width, request.height
                )
                if corners_2d is not None:
                    if tid == request.highlight_track_id:
                        color = (0, 200, 255)
                    elif obj["edited"]:
                        color = (0, 165, 255)
                    else:
                        color = (0, 255, 0)
                    label = f"T:{tid}" if request.draw_ids else None
                    image = renderer._draw_bbox(image, corners_2d, color=color, label=label)

        # 可选：绘制选中 track 的运动轨迹
        if tm and request.draw_trajectories and request.highlight_track_id is not None:
            viewmat = torch.inverse(c2w)
            traj = tm.get_track_trajectory(request.highlight_track_id, use_edits=True)
            pts = [np.asarray(p["center"], dtype=np.float32) for p in traj]
            if len(pts) >= 2:
                image = draw_trajectory_on_image(
                    image, pts, viewmat, K, color=(255, 200, 0), thickness=2, device=device
                )

        # 三维极端天气：先做体积雾/能见度衰减，再渲染相机视锥内3D雨雪粒子
        if request.weather:
            image = _apply_atmospheric_extinction(image, request.weather)
            image = _render_3d_weather_particles(image, c2w, K, request.frame_idx, request.weather)

        _, buffer = cv2.imencode(".png", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        image_b64 = base64.b64encode(buffer).decode("utf-8")

        return {
            "success": True,
            "frame_idx": request.frame_idx,
            "image": f"data:image/png;base64,{image_b64}",
            "width": request.width,
            "height": request.height,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class TrackMatrixEdit(BaseModel):
    scene_id: str
    track_id: int
    frame_idx: int
    pose_matrix: List[List[float]]  # 4x4 世界变换矩阵


@app.post("/api/edit/object/matrix")
async def edit_object_matrix(request: TrackMatrixEdit):
    """用 4x4 世界变换矩阵设置某 track 在指定帧的位姿（3D 拖拽编辑）。

    基于 track_id，保证跨帧编辑的是同一物理物体。
    """
    tm = get_track_manager_or_404(request.scene_id)
    _reject_if_replaced(tm, request.track_id)
    try:
        pose = np.asarray(request.pose_matrix, dtype=np.float32)
        if pose.shape != (4, 4):
            raise HTTPException(status_code=400, detail="pose_matrix must be 4x4")

        tm.push_history()
        tm.set_track_pose(request.track_id, request.frame_idx, pose)

        history_entry = {
            "type": "matrix_edit",
            "track_id": request.track_id,
            "frame_idx": request.frame_idx,
            "timestamp": datetime.now().isoformat()
        }
        if request.scene_id in studio_state["edit_history"]:
            studio_state["edit_history"][request.scene_id].append(history_entry)

        return {
            "success": True,
            "track_id": request.track_id,
            "frame_idx": request.frame_idx,
            "pose_matrix": pose.tolist(),
            "history_entry": history_entry,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def _euler_ypr_to_matrix(yaw_deg, pitch_deg, roll_deg):
    """yaw(绕Y) / pitch(绕X) / roll(绕Z)，度 -> 3x3 旋转矩阵。"""
    return R.from_euler("YXZ", [yaw_deg, pitch_deg, roll_deg], degrees=True).as_matrix().astype(np.float32)


def _matrix_to_euler_ypr(Rm):
    return [float(v) for v in R.from_matrix(np.asarray(Rm, dtype=np.float64)).as_euler("YXZ", degrees=True)]


def _reject_if_replaced(tm, track_id):
    """对已被 SAM3D 替换掉的原始物体做编辑时给出明确提示（避免"改了半天没反应/冒出两个模型"）。"""
    try:
        if tm.is_track_replaced(int(track_id)):
            raise HTTPException(
                status_code=409,
                detail=(f"物体 T:{int(track_id)} 已被 SAM3D 替换，不能再编辑它；"
                        f"请选择替换后的新物体（物体列表里的 1000xx 号）"),
            )
    except HTTPException:
        raise


class ObjectRotationRequest(BaseModel):
    scene_id: str
    track_id: Optional[int] = None
    object_id: Optional[int] = None
    yaw: Optional[float] = None
    pitch: Optional[float] = None
    roll: Optional[float] = None
    delta_yaw: Optional[float] = None
    delta_pitch: Optional[float] = None
    delta_roll: Optional[float] = None
    reset: bool = False


@app.post("/api/edit/object/rotation")
async def edit_object_rotation(request: ObjectRotationRequest):
    """设置/叠加某物体（track）的全局 360° 旋转偏移，作用于所有帧（播放时保持）。

    - reset=true: 清除旋转
    - 给 delta_*: 在当前旋转上叠加增量
    - 给 yaw/pitch/roll: 设为绝对角度（度）
    """
    tm = get_track_manager_or_404(request.scene_id)
    tid = request.track_id if request.track_id is not None else request.object_id
    if tid is None:
        raise HTTPException(status_code=400, detail="需要 track_id 或 object_id")
    tid = int(tid)
    try:
        tm.push_history()
        if request.reset:
            tm.set_track_rotation(tid, np.eye(3, dtype=np.float32))
        elif any(v is not None for v in (request.delta_yaw, request.delta_pitch, request.delta_roll)):
            dR = _euler_ypr_to_matrix(request.delta_yaw or 0.0, request.delta_pitch or 0.0, request.delta_roll or 0.0)
            tm.accumulate_track_rotation(tid, dR)
        else:
            Rm = _euler_ypr_to_matrix(request.yaw or 0.0, request.pitch or 0.0, request.roll or 0.0)
            tm.set_track_rotation(tid, Rm)

        R_cur = tm.get_track_rotation(tid)
        return {
            "success": True,
            "track_id": tid,
            "rotation_matrix": np.asarray(R_cur).tolist(),
            "yaw_pitch_roll": _matrix_to_euler_ypr(R_cur),
        }
    except HTTPException:
        raise
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"旋转失败: {e}")


@app.get("/api/edit/object/rotation/{scene_id}/{track_id}")
async def get_object_rotation(scene_id: str, track_id: int):
    """读取某物体的当前全局旋转（yaw/pitch/roll）。"""
    tm = get_track_manager_or_404(scene_id)
    R_cur = tm.get_track_rotation(track_id)
    return {
        "success": True,
        "track_id": int(track_id),
        "rotation_matrix": np.asarray(R_cur).tolist(),
        "yaw_pitch_roll": _matrix_to_euler_ypr(R_cur),
    }


class AutoHeadingRequest(BaseModel):
    scene_id: str
    enabled: Optional[bool] = None
    smoothing: Optional[float] = None       # 0~1，越大越平滑
    window: Optional[int] = None            # 方向估计 ±帧窗口
    max_yaw_rate: Optional[float] = None    # 朝向变化率上限(rad/s)，默认 1.0
    fps: Optional[float] = None             # 帧率（决定"每帧最多转多少度"）
    track_ids: Optional[List[int]] = None   # 只对这些 track 生效
    all_tracks: bool = False                # True 表示对所有动态物体生效
    reset: bool = False                     # 关闭并清空


@app.post("/api/edit/auto_heading")
async def edit_auto_heading(request: AutoHeadingRequest):
    """让动态物体的车头自动朝向其运动方向（逐帧、平滑过渡）。

    实现：`TrackManager.get_track_pose` 在返回位姿前，把偏航角替换为
    由轨迹估算出的（平滑）运动方向；俯仰/滚转保留，用户手动旋转偏移仍然叠加。
    """
    tm = get_track_manager_or_404(request.scene_id)
    try:
        if request.reset:
            cfg = tm.set_auto_heading(enabled=False, track_ids=[], all_tracks=True)
        else:
            cfg = tm.set_auto_heading(
                enabled=request.enabled,
                track_ids=request.track_ids,
                smoothing=request.smoothing,
                window=request.window,
                max_yaw_rate=request.max_yaw_rate,
                fps=request.fps,
                all_tracks=request.all_tracks or request.track_ids is None,
            )
        return {"success": True, **cfg}
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"设置自动朝向失败: {e}")


@app.get("/api/edit/auto_heading/{scene_id}")
async def get_auto_heading(scene_id: str, frame_idx: int = 0):
    """读取自动朝向配置 + 各动态物体当前的目标朝向。"""
    tm = get_track_manager_or_404(scene_id)
    return {
        "success": True,
        **tm.get_auto_heading_config(),
        "headings": tm.list_motion_headings(frame_idx),
    }


class TrackDeleteRequest(BaseModel):
    scene_id: str
    track_id: int


@app.post("/api/edit/track/delete")
async def delete_track(request: TrackDeleteRequest):
    """删除某 track（整段不渲染）。"""
    tm = get_track_manager_or_404(request.scene_id)
    tm.push_history()
    tm.delete_track(request.track_id)
    if request.scene_id in studio_state["edit_history"]:
        studio_state["edit_history"][request.scene_id].append({
            "type": "delete_track",
            "track_id": request.track_id,
            "timestamp": datetime.now().isoformat()
        })
    return {"success": True, "track_id": request.track_id}


@app.post("/api/edit/track/restore")
async def restore_track(request: TrackDeleteRequest):
    """恢复被删除/清除编辑的 track（合成物体则还原到创建时的轨迹）。"""
    tm = get_track_manager_or_404(request.scene_id)
    _reject_if_replaced(tm, request.track_id)
    tm.clear_track_edits(request.track_id)
    return {"success": True, "track_id": request.track_id}


class TrajectoryKeyframeItem(BaseModel):
    frame_idx: int
    pose_matrix: List[List[float]]


class TrackTrajectoryEdit(BaseModel):
    scene_id: str
    track_id: int
    keyframes: List[TrajectoryKeyframeItem]


@app.post("/api/edit/track/trajectory")
async def edit_track_trajectory(request: TrackTrajectoryEdit):
    """用关键帧列表设置某 track 的整条编辑轨迹（供轨迹线编辑使用）。"""
    tm = get_track_manager_or_404(request.scene_id)
    _reject_if_replaced(tm, request.track_id)
    try:
        kfs = [(k.frame_idx, np.asarray(k.pose_matrix, dtype=np.float32)) for k in request.keyframes]
        tm.push_history()
        tm.set_track_trajectory(request.track_id, kfs)
        if request.scene_id in studio_state["edit_history"]:
            studio_state["edit_history"][request.scene_id].append({
                "type": "trajectory_edit",
                "track_id": request.track_id,
                "num_keyframes": len(kfs),
                "timestamp": datetime.now().isoformat()
            })
        return {"success": True, "track_id": request.track_id, "num_keyframes": len(kfs)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class TrackPointEdit(BaseModel):
    scene_id: str
    track_id: int
    frame_idx: int
    center: List[float]  # 新的世界坐标中心 [x,y,z]


@app.post("/api/edit/track/point")
async def edit_track_point(request: TrackPointEdit):
    """编辑轨迹上某一帧的点位置（只移动中心，保持朝向/尺寸）。"""
    tm = get_track_manager_or_404(request.scene_id)
    _reject_if_replaced(tm, request.track_id)
    try:
        pose = tm.get_track_pose(request.track_id, request.frame_idx)
        if pose is None:
            raise HTTPException(status_code=404, detail="该帧无此 track")
        pose = np.asarray(pose, dtype=np.float32).copy()
        pose[0, 3] = float(request.center[0])
        pose[1, 3] = float(request.center[1])
        pose[2, 3] = float(request.center[2])
        tm.push_history()
        tm.set_track_pose(request.track_id, request.frame_idx, pose)
        return {"success": True, "track_id": request.track_id, "frame_idx": request.frame_idx}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class TrackPointAdaptiveEdit(BaseModel):
    scene_id: str
    track_id: int
    frame_idx: int
    center: List[float]          # 新的世界坐标中心 [x,y,z]
    influence: int = 6           # 影响半径（前后多少帧自适应跟随）
    falloff: str = "smooth"      # smooth | linear | gaussian


@app.post("/api/edit/track/point_adaptive")
async def edit_track_point_adaptive(request: TrackPointAdaptiveEdit):
    """智能轨迹编辑：拖动某一帧的点，相邻帧按影响范围自适应平滑跟随。

    解决"需要一个个手动调整节点"的问题——拖动一个节点，整条路径自适应调整。
    """
    tm = get_track_manager_or_404(request.scene_id)
    _reject_if_replaced(tm, request.track_id)
    try:
        tm.push_history()
        edited = tm.drag_point_adaptive(
            request.track_id, request.frame_idx, request.center,
            influence=request.influence, falloff=request.falloff,
        )
        if not edited:
            raise HTTPException(status_code=404, detail="该帧无此 track 或无法编辑")
        return {
            "success": True,
            "track_id": request.track_id,
            "frame_idx": request.frame_idx,
            "edited_frames": edited,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class TrackSmoothRequest(BaseModel):
    scene_id: str
    track_id: int
    smoothness: float = 0.5
    keep_endpoints: bool = True


@app.post("/api/edit/track/smooth")
async def smooth_track_endpoint(request: TrackSmoothRequest):
    """对整条轨迹做平滑，消除手动编辑产生的抖动。"""
    tm = get_track_manager_or_404(request.scene_id)
    _reject_if_replaced(tm, request.track_id)
    try:
        tm.push_history()
        edited = tm.smooth_track(
            request.track_id, smoothness=request.smoothness,
            keep_endpoints=request.keep_endpoints,
        )
        return {"success": True, "track_id": request.track_id, "edited_frames": edited}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==================== 撤销 / 重做 ====================

@app.post("/api/undo/{scene_id}")
async def undo_edit(scene_id: str):
    """撤销最近一次轨迹/物体编辑（基于 TrackManager 快照）。"""
    tm = get_track_manager_or_404(scene_id)
    ok = tm.undo()
    return {"success": ok, "can_undo": tm.can_undo(), "can_redo": tm.can_redo()}


@app.post("/api/redo/{scene_id}")
async def redo_edit(scene_id: str):
    """重做最近一次被撤销的编辑。"""
    tm = get_track_manager_or_404(scene_id)
    ok = tm.redo()
    return {"success": ok, "can_undo": tm.can_undo(), "can_redo": tm.can_redo()}


# ==================== SAM 3D ====================
# SAM 3D Objects 运行在独立 conda 环境（端口 8001 微服务），这里通过 HTTP 转发。
# 服务端渲染仍走 DGGT gsplat：把重建出的 .ply 挂到 renderer.sam3d_objects 上，
# 后续 /api/render/freeview、/api/render/frame 会自动把它渲染进 3D 场景。


class SAM3DImportRequest(BaseModel):
    scene_id: str
    # 方式一：直接指定已重建好的 .ply 路径（推荐，先 /reconstruct 再 /import）
    ply_path: Optional[str] = None
    # 方式二：直接传图片(+掩码) base64，后端先重建再导入
    image_base64: Optional[str] = None
    mask_base64: Optional[str] = None
    seed: Optional[int] = None
    # 放置参数
    pose_world: Optional[List[List[float]]] = None   # 4x4，缺省单位阵（原点）
    scale: float = 1.0
    object_id: Optional[int] = None
    meta: Optional[Dict[str, Any]] = None


@app.get("/api/sam3d/health")
async def sam3d_health():
    """探测 SAM 3D 微服务健康状态（不触发模型加载）。"""
    try:
        client = sam3d_client.get_sam3d_client()
        info = await client.health()
        return {"success": True, "service": info}
    except Exception as e:  # noqa: BLE001
        return {"success": False, "service": None, "error": str(e)}


@app.post("/api/sam3d/reconstruct")
async def sam3d_reconstruct(
    image: UploadFile = File(...),
    mask: UploadFile = File(None),
    seed: Optional[int] = Form(None),
    format: str = Form("ply"),  # "ply" | "ply+glb"
):
    """上传图片(+掩码)，调用 SAM 3D 微服务重建 3D 对象，返回 .ply（可选 .glb）路径。"""
    image_bytes = await image.read()
    mask_bytes = await mask.read() if mask is not None else None
    client = sam3d_client.get_sam3d_client()
    try:
        result = await client.reconstruct(
            image_bytes, mask_bytes, seed=seed, format=format,
            image_filename=image.filename or "image.png",
            mask_filename=mask.filename if mask else "mask.png",
        )
        return {"success": True, **result}
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"SAM 3D 重建失败: {e}")


@app.post("/api/sam3d/import")
async def sam3d_import(request: SAM3DImportRequest):
    """把 SAM 3D 重建的 .ply 导入已加载场景，作为可渲染物体。"""
    renderer = get_scene_or_404(request.scene_id)
    try:
        ply_path = request.ply_path
        if not ply_path:
            if not request.image_base64:
                raise HTTPException(status_code=400, detail="需要提供 ply_path 或 image_base64")
            image_bytes = base64.b64decode(request.image_base64)
            mask_bytes = base64.b64decode(request.mask_base64) if request.mask_base64 else None
            client = sam3d_client.get_sam3d_client()
            result = await client.reconstruct(image_bytes, mask_bytes, seed=request.seed, format="ply")
            ply_path = result.get("local_ply_path") or result.get("ply_path")

        if not ply_path or not os.path.exists(ply_path):
            raise HTTPException(status_code=400, detail=f"ply 文件不存在: {ply_path}")

        pose = request.pose_world
        if pose is None:
            pose = np.eye(4, dtype=np.float32).tolist()
        obj_id = renderer.add_sam3d_object(
            ply_path, pose, scale=request.scale,
            object_id=request.object_id, meta=request.meta,
        )
        return {
            "success": True,
            "object_id": obj_id,
            "ply_path": str(ply_path),
            "objects": renderer.list_sam3d_objects(),
        }
    except HTTPException:
        raise
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"SAM 3D 导入失败: {e}")


def _all_sam3d_objects(renderer, tm):
    """合并 renderer.sam3d_objects（独立导入）与 TrackManager 的 SAM3D track（替换）。"""
    objs = list(renderer.list_sam3d_objects())
    objs += tm.list_sam3d_tracks()
    return objs


def _find_sam3d_ply(renderer, tm, object_id):
    for obj in _all_sam3d_objects(renderer, tm):
        if obj["object_id"] == int(object_id):
            return obj.get("ply_path")
    return None


@app.get("/api/sam3d/objects/{scene_id}")
async def list_sam3d_objects(scene_id: str):
    """列出当前场景内已导入/替换的 SAM 3D 物体。"""
    renderer = get_scene_or_404(scene_id)
    tm = get_track_manager_or_404(scene_id)
    return {"success": True, "objects": _all_sam3d_objects(renderer, tm)}


@app.delete("/api/sam3d/objects/{scene_id}/{object_id}")
async def remove_sam3d_object(scene_id: str, object_id: int):
    """移除已导入/替换的 SAM 3D 物体。"""
    renderer = get_scene_or_404(scene_id)
    tm = get_track_manager_or_404(scene_id)
    removed = renderer.remove_sam3d_object(object_id)
    if not removed and tm.is_sam3d_track(object_id):
        tm.push_history()
        tm.remove_synthetic_track(object_id)
        removed = True
    if not removed:
        raise HTTPException(status_code=404, detail=f"未找到 SAM 3D 物体 {object_id}")
    return {"success": True, "object_id": object_id, "objects": _all_sam3d_objects(renderer, tm)}


@app.get("/api/sam3d/objects/{scene_id}/{object_id}/download")
async def download_sam3d_object(scene_id: str, object_id: int):
    """导出某个已导入/替换物体的 .ply 文件。"""
    renderer = get_scene_or_404(scene_id)
    tm = get_track_manager_or_404(scene_id)
    path = _find_sam3d_ply(renderer, tm, object_id)
    if not path or not os.path.exists(path):
        raise HTTPException(status_code=404, detail=f"未找到 SAM 3D 物体 {object_id} 或 ply 文件")
    return FileResponse(path, filename=os.path.basename(path), media_type="application/octet-stream")


class SAM3DPreviewRequest(BaseModel):
    scene_id: str
    object_id: int
    num_frames: int = 36
    size: int = 512


@app.post("/api/sam3d/preview")
async def sam3d_preview(request: SAM3DPreviewRequest):
    """渲染某个 SAM3D 物体的旋转预览（turntable），返回多帧 base64 PNG。"""
    renderer = get_scene_or_404(request.scene_id)
    tm = get_track_manager_or_404(request.scene_id)
    try:
        ply_path = _find_sam3d_ply(renderer, tm, request.object_id)
        if not ply_path:
            raise HTTPException(status_code=404, detail=f"未找到 SAM 3D 物体 {request.object_id}")
        frames = renderer.render_ply_preview(
            ply_path, num_frames=request.num_frames, size=request.size
        )
        images = []
        for f in frames:
            _, buf = cv2.imencode(".png", cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
            images.append(base64.b64encode(buf).decode("ascii"))
        return {
            "success": True,
            "frames": images,
            "num_frames": len(images),
            "size": request.size,
        }
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"预览渲染失败: {e}")


# ==================== SAM 交互式分割 + 替换 ====================


def _render_source_rgb(renderer, scene_id: str, frame_idx: int, max_dim: int = 1536) -> np.ndarray:
    """渲染用于 SAM 分割 / SAM 3D 重建的高清源帧（干净、无包围盒/ID）。

    原生相机分辨率通常偏低（如 518×350），这里按长边 max_dim 等比放大分辨率与内参，
    得到更高清的源图，显著改善分割与重建质量。
    """
    tm = None if renderer.static_only else get_track_manager_or_404(scene_id)
    object_overrides = tm.build_object_overrides(frame_idx) if tm else {}
    extra_objects = tm.build_extra_objects(frame_idx, include_ego=False) if tm else []

    cam_flat = renderer.flat_index(frame_idx, 0)
    ego_path = os.path.join(renderer.ego_dir, f"frame_{cam_flat:04d}_ego.json")
    with open(ego_path, "r") as f:
        ego = json.load(f)
    W0, H0 = ego["camera"]["width"], ego["camera"]["height"]
    K0 = np.asarray(ego["camera_intrinsics"], dtype=np.float32).copy()

    scale = max_dim / max(W0, H0)
    W = max(1, int(round(W0 * scale)))
    H = max(1, int(round(H0 * scale)))
    sx, sy = W / W0, H / H0
    K0[0, 0] *= sx; K0[0, 2] *= sx
    K0[1, 1] *= sy; K0[1, 2] *= sy
    K = torch.tensor(K0, device=renderer.device, dtype=torch.float32)

    return renderer._render_frame_with_object_overrides(
        frame_idx, object_overrides, extra_objects=extra_objects,
        K_override=K, width_override=W, height_override=H,
    )


def _rgb_to_png(image: np.ndarray) -> bytes:
    _, buf = cv2.imencode(".png", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    return buf.tobytes()


def _frame_camera(renderer, frame_idx: int):
    """返回某真实帧 view0 自车相机 (c2w 4x4, K, W, H)，无则 None。"""
    try:
        cam_flat = renderer.flat_index(int(frame_idx), 0)
    except Exception:  # noqa: BLE001
        return None
    ego_path = os.path.join(renderer.ego_dir, f"frame_{cam_flat:04d}_ego.json")
    if not os.path.exists(ego_path):
        return None
    with open(ego_path, "r") as f:
        ego = json.load(f)
    c2w = np.asarray(ego["camera_extrinsics_world"], dtype=np.float32)
    if c2w.shape == (3, 4):
        c2w = np.vstack([c2w, np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32)])
    K = np.asarray(ego["camera_intrinsics"], dtype=np.float32)
    W = int(ego["camera"]["width"])
    H = int(ego["camera"]["height"])
    return c2w, K, W, H


class SamFrameCandidatesRequest(BaseModel):
    scene_id: str
    target_object_id: int
    max_candidates: int = 8
    stride: int = 1                      # 采样步长（长序列可加大以加速）
    frame_range: Optional[List[int]] = None   # [start, end] 限定范围


@app.post("/api/sam/frame_candidates")
async def sam_frame_candidates(request: SamFrameCandidatesRequest):
    """为“多帧选源图”打分：找出目标物体**遮挡最少、可见面积最大**的帧。

    评分：`score = (bbox 面积 / 全帧最大面积) * (1 - 遮挡比例)`；
    遮挡 = 比目标更靠近相机、且 2D 包围框与目标重叠的其它物体所占比例。
    单帧重建无法处理遮挡，因此建议在这些推荐帧上做分割与重建。
    """
    renderer = get_scene_or_404(request.scene_id)
    tm = get_track_manager_or_404(request.scene_id)
    try:
        target = int(request.target_object_id)
        frames = tm.get_track_frames(target)
        if not frames:
            raise HTTPException(status_code=404, detail=f"未找到物体 {target}")
        if request.frame_range and len(request.frame_range) >= 2:
            lo, hi = int(request.frame_range[0]), int(request.frame_range[1])
            frames = [f for f in frames if lo <= int(f) <= hi]
        stride = max(1, int(request.stride))
        frames = [int(f) for f in frames][::stride]

        rows = []
        dev = renderer.device
        for f in frames:
            cam = _frame_camera(renderer, f)
            if cam is None:
                continue
            c2w, K, W, H = cam
            viewmat = np.linalg.inv(c2w)
            viewmat_t = torch.tensor(viewmat, device=dev, dtype=torch.float32)
            K_t = torch.tensor(K, device=dev, dtype=torch.float32)
            objs = tm.get_frame_objects(f)
            tgt = next((o for o in objs if int(o["track_id"]) == target), None)
            if tgt is None:
                continue
            tgt_pose = np.asarray(tgt["pose_world"], dtype=np.float32)
            tgt_corners = renderer._get_bbox_corners_2d(
                torch.tensor(tgt_pose, device=dev, dtype=torch.float32),
                tgt["dimensions"], viewmat_t, K_t, W, H)
            if tgt_corners is None:
                continue
            x0, y0 = tgt_corners[:, 0].min(), tgt_corners[:, 1].min()
            x1, y1 = tgt_corners[:, 0].max(), tgt_corners[:, 1].max()
            area = float(max(0.0, x1 - x0) * max(0.0, y1 - y0))
            tgt_z = float((viewmat @ np.append(tgt_pose[:3, 3], 1.0))[2])

            occluded = 0.0  # 被遮挡的像素面积（用包围框近似）
            for o in objs:
                if int(o["track_id"]) == target:
                    continue
                pose = np.asarray(o["pose_world"], dtype=np.float32)
                try:
                    z = float((viewmat @ np.append(pose[:3, 3], 1.0))[2])
                except Exception:  # noqa: BLE001
                    continue
                if z >= tgt_z - 1e-3:      # 不比目标更近 → 不会遮挡目标
                    continue
                c = renderer._get_bbox_corners_2d(
                    torch.tensor(pose, device=dev, dtype=torch.float32),
                    o["dimensions"], viewmat_t, K_t, W, H)
                if c is None:
                    continue
                ox0, oy0 = c[:, 0].min(), c[:, 1].min()
                ox1, oy1 = c[:, 0].max(), c[:, 1].max()
                iw = max(0.0, min(x1, ox1) - max(x0, ox0))
                ih = max(0.0, min(y1, oy1) - max(y0, oy0))
                occluded += iw * ih
            occ_ratio = float(min(1.0, occluded / area)) if area > 1e-6 else 1.0
            rows.append({
                "frame_idx": int(f),
                "bbox": [float(x0), float(y0), float(x1), float(y1)],
                "bbox_area": area,
                "occlusion": occ_ratio,
                "depth": tgt_z,
            })

        if not rows:
            return {"success": True, "candidates": [], "scores": []}

        max_area = max(r["bbox_area"] for r in rows) or 1.0
        for r in rows:
            r["score"] = float((r["bbox_area"] / max_area) * (1.0 - r["occlusion"]) ** 2)

        ranked = sorted(rows, key=lambda r: -r["score"])
        return {
            "success": True,
            "target_object_id": target,
            "candidates": ranked[: max(1, int(request.max_candidates))],
            "scores": sorted(rows, key=lambda r: r["frame_idx"]),
            "num_frames": len(rows),
        }
    except HTTPException:
        raise
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"帧评分失败: {e}")


def _ply_bbox_and_orientation(ply_path: str, target_dims=None, sigma_k=None):
    """读 .ply 的 xyz，返回 (质心, 三轴尺寸, 朝向修正 3x3 旋转矩阵)。

    与早期版本的区别（解决"替换后比原模型大一圈"）：
    - 不再用"高斯中心的 min/max bbox"，而是用**可见轮廓**：每个高斯按其可见半径
      （sigma_k × 高斯尺度，尺度用 99.5% 分位截断防离群）向外扩张后的包络。
      早期只按中心 bbox 对齐，会忽略 SAM3D 输出的"虚边"，替换后看起来偏大。
    `sigma_k` 默认取环境变量 DGGT_SAM3D_SIGMA_K（默认 2.0，约 2σ 可见范围）。

    朝向修正：SAM3D canonical(上=-Z, 前=-Y, 右=+X) → DGGT 物体局部系(X=宽, Y=高, Z=长)，
    R = Ry(180°)·Rx(+90°) = [[-1,0,0],[0,0,-1],[0,-1,0]]（纯旋转，det=+1）。
    """
    from plyfile import PlyData
    v = PlyData.read(ply_path)["vertex"]
    xyz = np.stack([np.asarray(v["x"]), np.asarray(v["y"]), np.asarray(v["z"])], axis=-1).astype(np.float32)
    # 每个高斯的可见半径：scale 是未激活的 log 值，exp 还原；用 99.5% 分位截断，
    # 避免个别超大高斯把尺寸撑飞。
    if sigma_k is None:
        sigma_k = float(os.environ.get("DGGT_SAM3D_SIGMA_K", "2.0"))
    try:
        sc = np.stack([np.asarray(v["scale_0"]), np.asarray(v["scale_1"]),
                       np.asarray(v["scale_2"])], axis=-1).astype(np.float32)
        sc = np.exp(np.clip(sc, -30.0, 2.0))
        sc = np.minimum(sc, np.percentile(sc, 99.5, axis=0).astype(np.float32))
        rad = float(sigma_k) * sc
        # 可见轮廓 = 高斯中心 ± 可见半径 的包络（比只看中心的 bbox 更贴近渲染结果，
        # 否则替换后的模型会看起来"大一圈"）
        lo = (xyz - rad).min(axis=0)
        hi = (xyz + rad).max(axis=0)
        ext = hi - lo
        # 质心通常不等于包围盒中心（高斯在前脸密、车尾疏），而渲染时是"减去 center"把
        # 它挪到位姿原点、贴地又按"位姿中心 ± 高度/2"推算的。用质心会让模型相对位姿
        # 偏上/偏下 → 看起来悬空或沉一半。这里统一用**可见轮廓包围盒中心**。
        center = (lo + hi) / 2.0
    except Exception:  # noqa: BLE001
        p_lo = np.percentile(xyz, 0.5, axis=0)
        p_hi = np.percentile(xyz, 99.5, axis=0)
        ext = np.asarray(p_hi - p_lo, dtype=np.float32)
        center = (p_lo + p_hi) / 2.0
    R = np.array([[-1, 0, 0], [0, 0, -1], [0, -1, 0]], dtype=np.float32)
    return center, np.asarray(ext, dtype=np.float32), R


def _tight_model_dims(recon_ext, scale=None, scale_vec=None, corr=None, floor=0.10):
    """把重建模型的三轴尺寸换算成**物体局部系**里紧贴模型的包围盒 [宽, 高, 长]。

    为什么需要：SAM3D 重建出来的车和 4DGS 那个（偏扁偏宽、甚至悬浮的）包围盒并不是一回事。
    如果继续沿用"原物体包围盒尺寸"，就会出现**模型其实没碰到、过大的框先碰到了**，
    碰撞判定（corner case / 语言轨迹编辑都走 OBB-SAT）直接判成事故。

    换算依据（与 dggt_engine 里的渲染顺序严格一致）：
      1. `.ply` canonical 轴：x=右(宽)、y=前(长)、z=上(高)；
      2. 渲染时先做 `model_corr`（SAM3D_MODEL_CORR，把 canonical 旋到物体局部系
         X=宽 / Y=高 / Z=长），再在**局部系**里乘缩放；
      3. 所以局部包围盒 = (|corr| @ ext) * 缩放。（`|corr|` 让 canonical 的
         x/z/y 分别落到局部 X/Y/Z。）

    `floor` 是每一轴的下限（默认 10cm），避免个别退化重建（例如只有几个 cm 厚）
    让碰撞体退化成"纸片"而永远算不出碰撞。
    """
    ext = np.abs(np.asarray(recon_ext, dtype=np.float64).reshape(3))
    if corr is None:
        corr = np.array([[-1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, -1.0, 0.0]], dtype=np.float64)
    corr = np.abs(np.asarray(corr, dtype=np.float64).reshape(3, 3))
    dims = corr @ ext
    if scale_vec is not None:
        sv = np.abs(np.asarray(scale_vec, dtype=np.float64).reshape(3))
        if scale is not None:
            sv = sv * float(scale)
    elif scale is not None:
        sv = np.full(3, abs(float(scale)), dtype=np.float64)
    else:
        sv = np.ones(3, dtype=np.float64)
    dims = dims * sv
    return [round(float(max(float(floor), float(d))), 3) for d in dims]


def _fit_scale_and_dims(recon_ext, target_dims, fit=1.0, corr=None, flat_ratio_min=0.15,
                        floor=0.10, max_ratio=1.0):
    """由"重建尺寸 + 目标尺寸"决定 (缩放, 包围盒, 诊断信息) —— 替换/插入共用。

    **两条硬约束**（都是踩坑后定的）：

    1. **包围盒必须等于"渲染出来的模型"**：`dims` 一律用紧贴模型的尺寸，绝不回退到别的值。
       之前"扁片重建"时包围盒回退成目标尺寸、模型却按 2.5 倍渲染，两边不一致 →
       看起来**比原车大一圈**，而且贴地是按包围盒高度算的 → 模型**悬空/沉地**。
    2. **模型不超过目标包围盒**：缩放取 `min(按高度, 塞进目标)`：
       - 按高度：`目标高 / 重建高`（用户要求"只对齐高度、等比、不逐轴拉伸"）；
       - 塞进目标：`min_i(目标_i / 重建_i)`，保证任何一轴都不超过目标（不再"大一圈"）。
       两者取小 → 既保持比例，又不会比原物体大。`fit`（面板滑块 / `DGGT_SAM3D_FIT`）
       在此基础上再整体缩，因为包围盒跟着模型走，所以缩小也不会悬空。

    `suspect`（最短轴/最长轴 < `flat_ratio_min`，SAM3D 偶发的"扁片"重建）只作为**诊断信息**
    回报，用于提示"换更清晰的目标或上传参考图"，不再改变尺寸逻辑。
    """
    ext = np.abs(np.asarray(recon_ext, dtype=np.float64).reshape(3))
    corr_m = np.abs(np.asarray(
        corr if corr is not None else np.array([[-1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, -1.0, 0.0]]),
        dtype=np.float64).reshape(3, 3))
    mapped = corr_m @ ext                       # 物体局部 [宽, 高, 长]
    tgt = [float(x) for x in target_dims[:3]]
    if len(tgt) < 3:
        tgt = [4.5, 1.6, 2.0]

    by_height = float(tgt[1]) / max(1e-6, float(mapped[1]))          # 按高度
    fit_inside = min(float(tgt[i]) / max(1e-6, float(mapped[i])) for i in range(3))
    fit_inside *= float(max_ratio)                                    # max_ratio=1 → 不超目标
    base = max(1e-3, min(1000.0, min(by_height, fit_inside)))
    scale = max(1e-3, min(1000.0, base * float(fit)))

    tight = [float(x) for x in _tight_model_dims(ext, scale=scale, corr=corr, floor=floor)]
    model_h = float((corr_m @ (ext * scale))[1])                      # 渲染后的模型高度（局部 Y）
    flat = float(ext.min()) / max(1e-6, float(ext.max()))
    info = {
        "recon_flat_ratio": round(flat, 4),
        "scale_by_height": round(by_height, 4),
        "scale_fit_inside": round(fit_inside, 4),
        "scale": round(scale, 4),
        "model_height_m": round(model_h, 4),
        "dims_model": tight,
        "bbox_mode": "tight_to_model",
        "suspect": bool(flat < float(flat_ratio_min)),
    }
    if info["suspect"]:
        info["reason"] = (
            f"重建模型近似扁片（最短轴/最长轴={flat:.3f}）：已按『等比塞进目标包围盒』处理"
            f"（缩放 {scale:.2f}），包围盒仍与渲染一致，不会悬空；建议换更清晰的目标或上传参考图重做")
    return scale, tight, info


# 通用车模型兜底：SAM3D 偶尔会给出"扁片"（最短轴/最长轴 < 0.15），
# 这种结果插进场景就是个纸片车。与其插垃圾，不如换成这份通用车模型（按目标尺寸等比缩），
# 并在回执里说明。想换成别的车模用 DGGT_CAR_FALLBACK_PLY 指定。
FALLBACK_VEHICLE_PLY = os.environ.get(
    "DGGT_CAR_FALLBACK_PLY", "/root/autodl-fs/dggt-main/sam-3d-objects/ego.ply")


def _is_vehicle_cat(cat) -> bool:
    """按 VLM 给的类别名判断是不是车（复用 nl_entity 的规则）。"""
    try:
        import nl_entity as _nl
        return bool(_nl.is_vehicle_cat(cat))
    except Exception:  # noqa: BLE001
        return False


def _looks_like_vehicle(dims) -> bool:
    """按尺寸粗判是不是"车"（用于决定要不要套用车模兜底 / 夹尺寸）。"""
    try:
        w, h, l = (float(dims[0]), float(dims[1]), float(dims[2]))
    except Exception:  # noqa: BLE001
        return False
    return l >= 2.5 and 1.0 <= w <= 2.9 and 0.6 <= h <= 2.6 and l > w


def _measure_vehicle(ply_path, target_dims, fit=1.0, is_vehicle=True,
                     flat_ratio_min=0.15):
    """量重建模型 →（必要时用通用车模兜底）→ 给出缩放/包围盒/诊断。

    Returns: (ply_path, center, ext, corr, scale, dims, info)
    替换路径与插入路径共用，保证两边的尺寸/贴地逻辑一致。
    """
    center, ext, corr = _ply_bbox_and_orientation(ply_path, target_dims)
    ext = np.asarray(ext, dtype=np.float64).reshape(3)
    scale, dims, info = _fit_scale_and_dims(ext, target_dims, fit=fit, corr=corr,
                                            flat_ratio_min=flat_ratio_min)
    if not info.get("suspect") or not is_vehicle:
        return ply_path, center, ext, corr, scale, dims, info
    fb = FALLBACK_VEHICLE_PLY
    if not fb or not os.path.exists(fb) or os.path.abspath(fb) == os.path.abspath(str(ply_path)):
        return ply_path, center, ext, corr, scale, dims, info
    try:
        c2, e2, r2 = _ply_bbox_and_orientation(fb, target_dims)
        e2 = np.asarray(e2, dtype=np.float64).reshape(3)
        s2, d2, i2 = _fit_scale_and_dims(e2, target_dims, fit=fit, corr=r2,
                                         flat_ratio_min=flat_ratio_min)
        i2 = dict(i2)
        i2["fallback_ply"] = str(fb)
        # 保留"原始重建退化了"这件事：兜底后 suspect 描述的是 ego.ply（正常车），
        # 但用户需要知道"这次 SAM3D 输出是扁片、已经换过模型"。
        i2["suspect_original"] = bool(info.get("suspect"))
        i2["recon_flat_ratio_original"] = info.get("recon_flat_ratio")
        i2["reason"] = (str(info.get("reason") or "") +
                        "；已改用通用车模型兜底（DGGT_CAR_FALLBACK_PLY，默认 sam-3d-objects/ego.ply）")
        return str(fb), c2, e2, r2, s2, d2, i2
    except Exception:  # noqa: BLE001
        return ply_path, center, ext, corr, scale, dims, info


class SAMSegmentRequest(BaseModel):
    scene_id: str
    frame_idx: int
    points: List[List[float]]          # 归一化坐标 [[nx, ny], ...]  (0~1)
    point_labels: List[int]            # 与 points 等长：1=前景, 0=背景


class SAM3DReplaceRequest(BaseModel):
    scene_id: str
    frame_idx: int
    target_object_id: int              # 要被替换的动态物体 track_id
    points: List[List[float]]          # 归一化前景/背景点
    point_labels: List[int]
    seed: Optional[int] = None
    fit: Optional[float] = None        # 替换尺寸微调系数（<1 更小），None=用默认 DGGT_SAM3D_FIT


class SyntheticScaleRequest(BaseModel):
    scene_id: str
    track_id: int
    factor: float                      # 相对"基准尺寸"的倍数（1.0=替换时自动算的尺寸）


class SyntheticShadowRequest(BaseModel):
    scene_id: str
    track_id: int
    enabled: bool = True               # 是否给该物体补"接触阴影"


@app.post("/api/edit/synthetic/shadow")
async def set_synthetic_shadow(request: SyntheticShadowRequest):
    """开关某个合成物体（含 SAM3D 重建/替换物体、主车）的接触阴影。

    SAM3D 重建的 .ply 高斯没有烘焙阴影，插进 4DGS 场景会显得"发飘"，
    所以默认在物体底部补一圈暗高斯（见 dggt_engine._contact_shadow）。
    """
    tm = get_track_manager_or_404(request.scene_id)
    tid = int(request.track_id)
    s = tm.synthetic_tracks.get(tid)
    if s is None:
        raise HTTPException(status_code=404, detail=f"T:{tid} 不是合成物体")
    tm.push_history()
    s["shadow"] = bool(request.enabled)
    tm._invalidate_heading_cache(tid)
    return {"success": True, "track_id": tid, "shadow": bool(request.enabled)}


class SAMSourceImageRequest(BaseModel):
    scene_id: str
    frame_idx: int
    max_dim: int = 1536


@app.post("/api/sam/source_image")
async def sam_source_image(request: SAMSourceImageRequest):
    """渲染高清源帧（供前端交互分割显示，1:1 坐标映射）。"""
    renderer = get_scene_or_404(request.scene_id)
    try:
        image = _render_source_rgb(renderer, request.scene_id, request.frame_idx, request.max_dim)
        _, buf = cv2.imencode(".png", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        return {
            "success": True,
            "image": base64.b64encode(buf).decode("ascii"),
            "width": int(image.shape[1]),
            "height": int(image.shape[0]),
        }
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"源图渲染失败: {e}")


class SAMSetModelRequest(BaseModel):
    model_type: str


@app.get("/api/sam/models")
async def sam_models():
    """列出可用的 SAM 分割模型及当前选中的模型。"""
    return {
        "success": True,
        "current": sam_segment.current_model(),
        "models": sam_segment.available_models(),
    }


@app.post("/api/sam/model")
async def sam_set_model(request: SAMSetModelRequest):
    """切换 SAM 分割模型（vit_b / vit_l / vit_h）。"""
    try:
        m = sam_segment.set_model(request.model_type)
        return {"success": True, "current": m}
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/sam/segment")
async def sam_segment_endpoint(request: SAMSegmentRequest):
    """交互式分割：渲染指定帧，用 SAM 点提示生成掩码（供前端预览）。"""
    renderer = get_scene_or_404(request.scene_id)
    try:
        image = _render_source_rgb(renderer, request.scene_id, request.frame_idx)
        H, W = image.shape[:2]
        pts = [[float(p[0]) * W, float(p[1]) * H] for p in request.points]
        mask, score = sam_segment.segment(image, pts, request.point_labels)
        return {
            "success": True,
            "score": score,
            "mask": sam_segment.mask_to_base64(mask),
            "width": W,
            "height": H,
            "num_fg_pixels": int(mask.sum()),
        }
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"分割失败: {e}")


@app.post("/api/sam3d/replace")
async def sam3d_replace(request: SAM3DReplaceRequest):
    """交互式重建并替换：分割 -> SAM 3D 重建 -> 按目标物体位姿/尺寸对齐 -> 删除原物体。"""
    renderer = get_scene_or_404(request.scene_id)
    tm = get_track_manager_or_404(request.scene_id)
    try:
        if len(request.points) != len(request.point_labels) or not request.points:
            raise HTTPException(status_code=400, detail="points 与 point_labels 需非空且等长")

        # 1. 渲染干净帧 + SAM 分割
        image = _render_source_rgb(renderer, request.scene_id, request.frame_idx)
        H, W = image.shape[:2]
        pts = [[float(p[0]) * W, float(p[1]) * H] for p in request.points]
        mask, score = sam_segment.segment(image, pts, request.point_labels)

        # 2. 目标物体的完整轨迹 + 尺寸（替换后沿原轨迹运动）
        target_frames = tm.get_track_frames(request.target_object_id)
        if not target_frames:
            raise HTTPException(status_code=404, detail=f"未找到物体 {request.target_object_id}")
        target_dims = tm.get_track_dimensions(request.target_object_id) or [4.5, 2.0, 1.6]
        original_poses = {}
        for f in target_frames:
            p = tm.get_track_pose(request.target_object_id, f)
            if p is not None:
                original_poses[int(f)] = np.asarray(p, dtype=np.float32)

        # 3. 释放 SAM 显存，给 SAM 3D 让路
        sam_segment.release_memory()

        # 4. SAM 3D 重建
        client = sam3d_client.get_sam3d_client()
        result = await client.reconstruct(
            _rgb_to_png(image), sam_segment.mask_to_png_bytes(mask),
            seed=request.seed, format="ply",
        )
        ply_path = result.get("local_ply_path") or result.get("ply_path")
        if not ply_path or not os.path.exists(ply_path):
            raise RuntimeError("重建未返回有效的 ply 文件")

        # 5. 对齐：质心 + **只按高度等比缩放** + 朝向修正
        #    只对齐高度、保持重建模型原来的长宽高比例；逐轴对齐会把它拉伸变形（看起来很奇怪）。
        #    尺寸仍按"可见轮廓"（含高斯半径，见 _ply_bbox_and_orientation）测量。
        tgt = [float(d) for d in target_dims[:3]]
        if len(tgt) < 3:
            tgt = [4.5, 1.6, 2.0]
        if request.fit is not None:
            fit = float(request.fit)
        else:
            fit = float(os.environ.get("DGGT_SAM3D_FIT", "1.0"))   # 1.0=按目标高度等比套入（不额外缩小）
        fit = max(0.05, min(5.0, fit))
        # 量尺寸 + 定缩放/包围盒（重建退化成"扁片"时自动换通用车模兜底）：
        # 替换与插入共用同一个 helper，保证包围盒恒等于渲染出来的模型、且不超目标。
        ply_path, recon_center, recon_ext, r_corr, scale, dims_tight, bbox_info = _measure_vehicle(
            ply_path, tgt, fit=fit, is_vehicle=_looks_like_vehicle(tgt))
        ex = np.maximum(recon_ext, 1e-4)
        base_scale = float(bbox_info.get("scale_by_height") or scale)
        scale_vec = None

        # 6. 删除原物体；加入重建物体（带原轨迹，成为一等公民物体）；
        #    自动朝向修正作为"初始全局旋转偏移"，用户可用旋转 UI 继续调整（会持久化）
        tm.push_history()
        tm.delete_track(request.target_object_id)
        tm.track_replaced.add(int(request.target_object_id))
        # 包围盒紧贴模型（而不是沿用原物体那个偏大的框），否则会出现
        # "模型没碰到、过大的框碰到了"的假碰撞。
        obj_id = tm.add_sam3d_track(
            ply_path, original_poses, dims_tight,
            scale=scale, center=recon_center, model_corr=r_corr,
            type_name=f"SAM3D替换(原#{request.target_object_id})",
            scale_vec=scale_vec,
        )
        # 记录自动算出的基准尺寸（滑块 factor=1.0 即替换后的尺寸）与诊断信息
        try:
            tspec = tm.synthetic_tracks[obj_id]
            tspec["base_scale"] = float(scale)
            tspec["auto_fit"] = fit
            tspec["recon_extent"] = [float(x) for x in recon_ext]
            # 留档：紧贴模型的框 vs 原物体的框（便于对比/排错）
            tspec["dims_model"] = [float(x) for x in bbox_info["dims_model"]]
            tspec["dims_target_reference"] = [float(x) for x in tgt]
            tspec["bbox_mode"] = "tight_to_model"
            tspec["recon_quality"] = bbox_info
            tspec["recon_degenerate"] = bool(bbox_info.get("suspect") or bbox_info.get("suspect_original"))
        except Exception:  # noqa: BLE001
            pass

        # 重建完成，卸载 SAM 3D 模型释放显存（下次 /reconstruct 会懒重载）
        try:
            await client.unload()
        except Exception:
            pass

        return {
            "success": True,
            "object_id": obj_id,
            "replaced_track_id": request.target_object_id,
            "ply_path": str(ply_path),
            "scale": scale,
            "base_scale": base_scale,
            "scale_vec": None,
            "fit": fit,
            "recon_extent": [float(x) for x in recon_ext],
            "dimensions": [float(x) for x in dims_tight],
            "bbox_mode": "fallback_target" if bbox_info["suspect"] else "tight_to_model",
            "recon_quality": bbox_info,
            "target_dims": [float(x) for x in tgt],
            "score": score,
        }
    except HTTPException:
        raise
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"SAM 3D 替换失败: {e}")


@app.post("/api/edit/synthetic/scale")
async def edit_synthetic_scale(request: SyntheticScaleRequest):
    """调整 SAM3D 等合成物体（自带 .ply）的整体尺寸。

    factor 是相对"替换时后端自动算出的基准尺寸"的倍数：<1 更小（解决"比原模型大一圈"）。
    """
    tm = get_track_manager_or_404(request.scene_id)
    sid = int(request.track_id)
    s = tm.synthetic_tracks.get(sid)
    if not s or not s.get("ply_path"):
        raise HTTPException(status_code=404, detail=f"合成物体 {sid} 不存在或不是可缩放模型")
    try:
        base = float(s.get("base_scale", s.get("scale", 1.0)) or 1.0)
        factor = max(0.2, min(5.0, float(request.factor)))
        tm.push_history()
        base_vec = s.get("base_scale_vec")
        if base_vec is not None:
            bv = np.asarray(base_vec, dtype=np.float32).reshape(3)
            s["scale_vec"] = (bv * factor).astype(np.float32)
            s["scale"] = float(np.cbrt(max(1e-9, float(s["scale_vec"].prod()))))
        else:
            s["scale"] = base * factor
        s["fit"] = factor
        # 包围盒尺寸也按同一倍数等比缩放（长宽高比不变）
        base_dims = s.get("base_dimensions") or s.get("dimensions") or []
        if len(base_dims) == 3:
            s["dimensions"] = [round(max(0.05, float(d) * factor), 3) for d in base_dims]
        tm._invalidate_heading_cache(sid)
        return {"success": True, "track_id": sid, "scale": s["scale"],
                "dimensions": s.get("dimensions"),
                "scale_vec": None if s.get("scale_vec") is None else [float(x) for x in s["scale_vec"]],
                "factor": factor, "base_scale": base}
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"调整合成物体尺寸失败: {e}")


# ==================== 文本 → 实体（LLaDA-Image + VLM + SAM3D） ====================

class Text2EntityRequest(BaseModel):
    scene_id: str
    prompt: str
    frame_idx: int = 0
    num_frames: int = 20
    mode: str = "ahead"            # ahead | oncoming | roadside
    distance: float = 12.0
    lateral: float = 0.0
    speed: float = 0.0             # m/s（0=静止）
    fit: Optional[float] = None    # 尺寸系数；None=用默认 DGGT_SAM3D_FIT(1.0)
    seed: int = 0
    use_vlm: bool = True
    steps: int = 4
    reference_image: Optional[str] = None   # 可选：直接给参考图 base64，跳过 LLaDA
    max_attempts: int = 3                   # VLM 判定"图和描述不一致"时最多重生成几次


class Text2EntityReplanRequest(BaseModel):
    scene_id: str
    track_id: int
    mode: str = "ahead"            # ahead | oncoming | roadside
    distance: float = 12.0
    lateral: float = 0.0
    speed: float = 0.0
    num_frames: int = 20
    start_frame: int = 0
    fit: Optional[float] = None    # 顺带调整整体尺寸（相对创建时的基准）


@app.get("/api/text2entity/health")
async def text2entity_health():
    """文本→实体微服务（LLaDA-Image / Qwen-VL）状态。"""
    try:
        import nl_entity as nl
        return {"success": True, **nl.health(), "local_text2image": nl.local_text2image_available()}
    except Exception as e:  # noqa: BLE001
        return {"success": False, "status": "error", "detail": str(e)}


def _insert_warnings(plan: Dict[str, Any]) -> List[str]:
    """把"轨迹跑到 4DGS 静态场景覆盖范围之外"这件事显式告诉用户。

    静态点只覆盖相机看过的地方（本场景 z≥3.9m），生成物一旦跑到覆盖外，路面估计会退化、
    背景也会变空。不是错误，但用户需要知道，否则会以为"位置又错了"。
    """
    out: List[str] = []
    try:
        n = int(plan.get("ground_sparse_total") or plan.get("ground_fallback_frames") or 0)
        total = len(plan.get("poses") or {}) or 1
        if n > 0:
            out.append(
                f"生成轨迹里有 {n}/{total} 帧超出了 4DGS 静态场景的重建范围"
                "（自车起点后方没有路面点）：这几帧改用自车车底高度/远邻域推算贴地，"
                "背景也会偏空。想看完整的行驶过程，建议减少帧数/降低速度，"
                "或把起始帧往后挪。")
    except Exception:  # noqa: BLE001
        pass
    return out


@app.post("/api/text2entity/generate")
async def text2entity_generate(request: Text2EntityRequest):
    """一句话添加实体：LLaDA-Image 出参考图 → 自动掩码 → VLM 定尺寸/类别 →
    SAM3D 重建 → 沿车道生成无冲突轨迹并插入场景（三个大模型串行加载，避免显存叠加）。
    """
    import nl_entity as nl
    renderer = get_scene_or_404(request.scene_id)
    tm = get_track_manager_or_404(request.scene_id)
    try:
        # 0) 预清理显存：SAM3D 微服务做完一次重建会常驻约 9~18GB，直到 /unload。
        #    如果上一次请求被中断（或刚做过替换），它还在占着显存，接着加载 LLaDA（约 14GB）
        #    就会 CUDA OOM —— 表现是"文本生成实体失败: ... 生成失败: CUDA out of memory"。
        #    这里在**真的要文生图时**先把它卸掉（用参考图跳过文生图就不用折腾）。
        if not request.reference_image:
            try:
                await sam3d_client.get_sam3d_client().unload()
                await asyncio.sleep(1.0)
            except Exception:  # noqa: BLE001
                pass

        # 1) 先一次性生成 N 张候选图（LLaDA 只加载一次），再用 VLM 逐张判断选最合适的一张。
        #    这样整条链路只有一次 LLaDA↔VLM 切换，避免反复换模型把内存/显存搞爆。
        vlm: Dict[str, Any] = {}
        dims: Optional[List[float]] = None
        cat = "object"
        if request.reference_image:
            b64 = request.reference_image
            if "," in b64[:64]:
                b64 = b64.split(",", 1)[1]
            candidates = [base64.b64decode(b64 + "=" * (-len(b64) % 4))]
        else:
            n = max(1, min(4, int(request.max_attempts))) if request.use_vlm else 1
            candidates = []
            for i in range(n):
                candidates.append(await asyncio.to_thread(
                    nl.generate_reference, request.prompt,
                    steps=int(request.steps), seed=int(request.seed) + i * 100003))
            nl.release_local_text2image()   # 本地 sd-turbo 兜底若被用到就释放
            await asyncio.sleep(1.0)

        image_bytes = candidates[0]
        if request.use_vlm:
            best = None
            for i, img in enumerate(candidates):
                try:
                    v = await asyncio.to_thread(nl.vlm_analyze, img, request.prompt)
                    v["attempts"] = i + 1
                    d_, c_ = nl.estimate_dimensions(v)
                except Exception as e:  # noqa: BLE001
                    v, d_, c_ = {"error": str(e), "attempts": i + 1}, None, "object"
                if best is None:
                    best = (img, v, d_, c_)
                if d_ is not None and nl.vlm_matches(v):
                    best = (img, v, d_, c_)
                    break
            if best is not None:
                image_bytes, vlm, dims, cat = best
        if dims is None:
            dims, cat = nl.rule_dimensions(request.prompt)
            vlm["fallback"] = "rule_dimensions"
        # 车辆类：和"场景已有车辆 / 常识"融合，避免生成的车明显偏大
        dims = nl.refine_vehicle_dims(dims, cat, tm)
        vlm["refined_dims"] = [float(x) for x in dims]

        # 2) 掩码：优先 SAM 点提示（生成图常有渐变底/阴影），失败退回白底阈值
        mask_bytes = await asyncio.to_thread(nl.mask_reference, image_bytes)

        # 3) VLM 用完，释放微服务显存，给 SAM3D 让路
        nl.unload_service()
        await asyncio.sleep(1.0)

        # 4) SAM3D 重建
        client = sam3d_client.get_sam3d_client()
        result = await client.reconstruct(image_bytes, mask_bytes, seed=request.seed, format="ply")
        ply_path = result.get("local_ply_path") or result.get("ply_path")
        if not ply_path or not os.path.exists(ply_path):
            raise RuntimeError("SAM3D 未返回有效的 ply 文件")

        # 4.5) 轨迹帧数：按自然语言解析出来的值（面板/指令里的"多少帧 / 几秒"），
        #      并夹到合理范围，避免生成出"2 帧看不见"或"几千帧跑不完"的轨迹。
        max_ins = int(os.environ.get("DGGT_MAX_INSERT_FRAMES", "600"))
        num_frames = int(max(2, min(max_ins, int(request.num_frames or 20))))

        # 5) 只按"高度"等比缩放，保持重建模型原本的长宽高比例（逐轴对齐会拉伸变形）
        #    注意：这里量出来的三轴尺寸变量叫 `ext`（替换路径里叫 `recon_ext`），
        #    别混用——混用会直接 NameError（曾导致"文本生成实体失败: name 'recon_ext' is not defined"）。
        tgt = [float(dims[0]), float(dims[1]), float(dims[2])]
        if request.fit is not None:
            fit = float(request.fit)
        else:
            fit = float(os.environ.get("DGGT_SAM3D_FIT", "1.0"))   # 1.0=按目标高度等比套入（不额外缩小）
        fit = max(0.05, min(5.0, fit))
        # 量尺寸 + 定缩放/包围盒；车辆且重建退化时换成通用车模兜底（替换/插入共用）
        ply_path, center, ext, r_corr, scale, dims_tight, bbox_info = _measure_vehicle(
            ply_path, tgt, fit=fit, is_vehicle=_looks_like_vehicle(tgt) or _is_vehicle_cat(cat))

        # 6) 车道轨迹 + 冲突规避（VLM 若判定"模型朝后"则翻转 180°）
        facing = nl.vlm_facing(vlm)
        # 用"紧贴模型"的尺寸做无冲突放置/判定，与最终插入的包围盒保持一致
        plan = await asyncio.to_thread(
            nl.plan_collision_free, tm, dims_tight,
            mode=request.mode, start_frame=int(request.frame_idx),
            num_frames=int(num_frames), distance=float(request.distance),
            lateral=float(request.lateral), speed=float(request.speed),
            flip=(facing == "back"))
        if not plan.get("ok"):
            try:
                await client.unload()
            except Exception:  # noqa: BLE001
                pass
            raise HTTPException(status_code=409,
                                detail=f"无法无冲突地放置该物体：{plan.get('reason')}")

        # 7) 插入为一等公民合成物体
        tm.push_history()
        obj_id = tm.add_sam3d_track(
            ply_path, plan["poses"], dims_tight, center=center, model_corr=r_corr,
            scale=scale, type_name=f"文本生成:{cat}")
        try:
            spec = tm.synthetic_tracks[obj_id]
            spec["base_scale_vec"] = None
            spec["base_scale"] = float(scale)
            spec["source_prompt"] = request.prompt
            spec["vlm"] = vlm
            spec["num_frames"] = int(num_frames)
            spec["dims_model"] = [float(x) for x in bbox_info["dims_model"]]
            spec["dims_target_reference"] = [float(x) for x in dims]
            spec["bbox_mode"] = "tight_to_model"
            spec["recon_quality"] = bbox_info
        except Exception:  # noqa: BLE001
            pass

        try:
            await client.unload()
        except Exception:  # noqa: BLE001
            pass

        # 8) 预览：插入后渲染当前帧
        preview = None
        try:
            img = _render_source_rgb(renderer, request.scene_id, int(request.frame_idx), 960)
            ok, buf = cv2.imencode(".png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            if ok:
                preview = base64.b64encode(buf.tobytes()).decode("ascii")
        except Exception:  # noqa: BLE001
            pass

        return {
            "success": True,
            "object_id": int(obj_id),
            "category": cat,
            # 注意：这里要回"真正写进场景的包围盒"（紧贴模型的 dims_tight），
            # 而不是 VLM 估出来的 dims——否则前端显示的框和实际框不是一个东西。
            "dimensions": [float(x) for x in dims_tight],
            "dims_target": [float(x) for x in dims],
            "bbox_mode": "tight_to_model",
            "recon_degenerate": bool(bbox_info.get("suspect") or bbox_info.get("suspect_original")),
            "recon_quality": bbox_info,
            "scale": float(scale),
            "scale_vec": None,
            "vlm": vlm,
            "facing": facing,
            "ply_path": str(ply_path),
            "num_frames": int(num_frames),
            "placement": {
                "mode": plan.get("mode"), "lateral": plan.get("lateral"),
                "distance": plan.get("distance"), "speed": plan.get("speed"),
                "num_frames": len(plan.get("poses") or {}), "up_sign": plan.get("up_sign"),
                "tried": plan.get("tried"),
                "ground_fallback_frames": plan.get("ground_fallback_frames"),
                "ground_y_min": plan.get("ground_y_min"),
                "ground_y_max": plan.get("ground_y_max"),
                "beyond_scene_coverage": bool(plan.get("ground_fallback_frames")),
            },
            "warnings": _insert_warnings(plan),
            "reference_image": base64.b64encode(image_bytes).decode("ascii"),
            "preview": preview,
        }
    except HTTPException:
        raise
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"文本生成实体失败: {e}")


@app.post("/api/text2entity/replan")
async def text2entity_replan(request: Text2EntityReplanRequest):
    """把已有的 SAM3D/合成物体**重排**成"物理合理 + 无冲突"的轨迹（保留模型，只改轨迹）。

    即"修改实体"路径：选中一个已插入的物体 → 沿车道重新规划（可换车道/速度/尺寸）→
    逐帧 OBB 冲突检测（忽略它自己）→ 无冲突才落盘。
    """
    import nl_entity as nl
    tm = get_track_manager_or_404(request.scene_id)
    sid = int(request.track_id)
    s = tm.synthetic_tracks.get(sid)
    if not s or not s.get("ply_path"):
        raise HTTPException(status_code=404, detail=f"物体 {sid} 不是可重排的 SAM3D/合成模型")
    try:
        dims = [float(x) for x in (s.get("dimensions") or [1.85, 1.5, 4.6])][:3]
        if len(dims) < 3:
            dims = [1.85, 1.5, 4.6]
        fit = 1.0
        if request.fit is not None:
            fit = max(0.05, min(5.0, float(request.fit)))
            base_vec = s.get("base_scale_vec")
            if base_vec is not None:
                s["scale_vec"] = (np.asarray(base_vec, dtype=np.float32).reshape(3) * fit).astype(np.float32)
                s["scale"] = float(np.cbrt(max(1e-9, float(s["scale_vec"].prod()))))
            else:
                s["scale"] = float(s.get("base_scale", s.get("scale", 1.0)) or 1.0) * fit
        check_dims = [d * fit for d in dims]
        plan = await asyncio.to_thread(
            nl.plan_collision_free, tm, check_dims, sid,
            mode=request.mode, start_frame=int(request.start_frame),
            num_frames=int(request.num_frames), distance=float(request.distance),
            lateral=float(request.lateral), speed=float(request.speed))
        if not plan.get("ok"):
            raise HTTPException(status_code=409,
                                detail=f"无法无冲突地重排：{plan.get('reason')}")
        tm.push_history()
        s["poses"] = {int(f): np.asarray(p, dtype=np.float32) for f, p in plan["poses"].items()}
        s["base_poses"] = {f: p.copy() for f, p in s["poses"].items()}
        tm._invalidate_heading_cache(sid)
        return {
            "success": True, "track_id": sid, "num_frames": len(s["poses"]),
            "dimensions": dims, "fit": fit, "scale": float(s.get("scale", 1.0)),
            "placement": {"mode": plan.get("mode"), "lateral": plan.get("lateral"),
                          "distance": plan.get("distance"), "speed": plan.get("speed"),
                          "num_frames": len(plan.get("poses") or {}), "tried": plan.get("tried")},
        }
    except HTTPException:
        raise
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"重排失败: {e}")


# ==================== 语言驱动的轨迹编辑 ====================

class TrajEditRequest(BaseModel):
    scene_id: str
    instruction: str
    frame_idx: int = 0
    num_frames: int = 30
    # 新建的车在相撞里演谁：auto=按指令措辞判断（判不准会提示）| attacker=新车去撞 | victim=对方撞新车
    new_role: str = "auto"


class TrajEditApplyRequest(TrajEditRequest):
    instruction: str = ""                        # 直接给 ops 时可省略
    ops: Optional[List[Dict[str, Any]]] = None   # 可选：直接给结构化操作，跳过 LLM 解析
    reference_image: Optional[str] = None        # 可选：指令里要生成新物体时的参考图（base64）


@app.post("/api/traj_edit/plan")
async def traj_edit_plan(request: TrajEditRequest):
    """自然语言 → 结构化轨迹编辑操作（只规划、不落盘）。"""
    tm = get_track_manager_or_404(request.scene_id)
    try:
        import traj_llm
        res = await asyncio.to_thread(traj_llm.plan, tm, request.instruction,
                                      int(request.frame_idx), int(request.num_frames),
                                      request.new_role, float(getattr(tm, "fps", 10.0)))
        return {"success": True, **res,
                "summary": traj_llm.scene_summary(tm, int(request.frame_idx))}
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"轨迹编辑规划失败: {e}")


@app.post("/api/traj_edit/apply")
async def traj_edit_apply(request: TrajEditApplyRequest):
    """规划（或直接用给定 ops）并作用到场景：速度/变道/转向/删除/相撞，insert 走生成车辆流程。"""
    tm = get_track_manager_or_404(request.scene_id)
    try:
        # 允许在延长出来的帧上编辑：先按需把时间轴延长到覆盖请求范围
        _ensure_timeline(tm, frame_idx=int(request.frame_idx) + int(request.num_frames) - 1)
        import traj_llm
        if request.ops:
            plan = {"source": "provided", "ops": request.ops, "raw": ""}
        else:
            plan = await asyncio.to_thread(traj_llm.plan, tm, request.instruction,
                                           int(request.frame_idx), int(request.num_frames),
                                           request.new_role, float(getattr(tm, "fps", 10.0)))
        ops = list(plan.get("ops") or [])

        # 规划完成后，按 ops 里真正需要的帧数再确保时间轴够长
        # （例如"生成一辆车跑 100 帧"，原场景只有 25 帧，得先延长）
        need = int(request.frame_idx) + int(request.num_frames)
        for op in ops:
            try:
                need = max(need, int(request.frame_idx) + int(op.get("num_frames") or 0))
            except (TypeError, ValueError):
                pass
        _ensure_timeline(tm, num_frames=need)

        # 先执行 insert（要 LLaDA 出图 + SAM3D 重建，最重）
        inserted: List[int] = []
        insert_details: List[Dict[str, Any]] = []
        insert_warnings: List[str] = []
        for op in ops:
            if str(op.get("op")) == "insert":
                req = Text2EntityRequest(
                    scene_id=request.scene_id,
                    prompt=str(op.get("prompt") or request.instruction),
                    frame_idx=int(request.frame_idx),
                    num_frames=int(op.get("num_frames") or min(20, request.num_frames)),
                    mode=str(op.get("mode") or "ahead"),
                    distance=float(op.get("distance") or 14.0),
                    lateral=float(op.get("lateral") or 0.0),
                    speed=float(op.get("speed") or 0.0),
                    use_vlm=True, max_attempts=2,
                    # 语言编辑里也能贴参考图：给了就直接用它做 SAM3D（跳过文生图）
                    reference_image=request.reference_image or None,
                )
                r = await text2entity_generate(req)
                if r.get("object_id") is not None:
                    inserted.append(int(r["object_id"]))
                    for w in (r.get("warnings") or []):
                        if w not in insert_warnings:
                            insert_warnings.append(str(w))
                    # 把"实际用于重建的参考图"和插入预览回给前端
                    insert_details.append({
                        "object_id": int(r["object_id"]),
                        "prompt": req.prompt,
                        "mode": req.mode,
                        "num_frames": r.get("num_frames") or req.num_frames,
                        "dimensions": r.get("dimensions"),
                        "dims_target": r.get("dims_target"),
                        "bbox_mode": r.get("bbox_mode"),
                        "recon_degenerate": r.get("recon_degenerate"),
                        "reference_image": r.get("reference_image"),
                        "preview": r.get("preview"),
                    })

        # 占位符（null / -1）指向"刚生成的新车"：collide 的 a/b 与动作的 track 都回填。
        # 注意 0 不做占位符——0 在部分场景是合法 track id（如 output/waymo_eval/001 的 T:0）。
        new_id = inserted[0] if inserted else None
        PH = (None, -1)
        for op in ops:
            if str(op.get("op")) == "insert":
                continue
            if new_id is None:
                continue
            for k in ("a", "b", "track"):
                if k in op and op[k] in PH:
                    op[k] = new_id

        tm.push_history()
        report = await asyncio.to_thread(traj_llm.apply_ops, tm, ops, 10.0,
                                          int(request.num_frames))
        return {"success": True, "source": plan.get("source"), "ops": ops,
                "inserted": inserted, "inserts": insert_details, "report": report,
                "raw": plan.get("raw"),
                "warnings": list(plan.get("warnings") or []) + insert_warnings}
    except HTTPException:
        raise
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"轨迹编辑应用失败: {e}")


# ==================== CARLA 仿真 API（可选模块） ====================
# 见 carla_api.py：CARLA 服务启停 / 场景列表 / 参数化渲染作业 / 视频与事件回放 /
# 实时 MJPEG 代理 / 对齐版 xosc 导出 / ScenarioRunner。挂载失败不影响其它功能。
try:
    import carla_api
    # 依赖注入：不要把 api_server 再 import 一遍（脚本运行时它是 __main__，再 import 会产生空状态的影子模块）
    carla_api.init(studio_state,
                   export_scenario=export_scenario,
                   export_request_cls=ScenarioExportRequest,
                   get_track_manager=get_track_manager_or_404,
                   get_scene=get_scene_or_404,
                   track_manager_cls=TrackManager)
    app.include_router(carla_api.router)
    print(f"[carla] CARLA API 已挂载 /api/carla/*  (bridge={carla_api.BRIDGE})")
except Exception as _carla_err:  # noqa: BLE001
    print(f"[carla] CARLA API 挂载失败（其它功能不受影响）：{_carla_err}")


# ==================== 前端静态托管（和其它功能同一个端口，省得再开静态服务器）====================
# 打开 http://<host>:8000/studio/ 就是 studio 前端（原来的 frontend/ 目录）
try:
    _FE_DIR = Path(__file__).resolve().parents[1] / "frontend"
    if _FE_DIR.exists():
        app.mount("/studio", StaticFiles(directory=str(_FE_DIR), html=True), name="studio")
        print(f"[studio] 前端已挂到 /studio/  ({_FE_DIR})")
except Exception as _fe_err:  # noqa: BLE001
    print(f"[studio] 前端挂载失败：{_fe_err}")


# ==================== 启动配置 ====================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
