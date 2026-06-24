"""
DGGT Studio Backend API Server V2
增强版API，支持轨迹可视化和更完善的编辑功能
"""

import os
import json
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
from scipy.spatial.transform import Rotation as R
from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from dggt_engine import DGGTRenderer, TrajectoryController, CornerCaseGenerator

sys.path.insert(0, str(Path(__file__).parent))
from track_manager import TrackManager
import corner_case

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
    pose_matrix: List[List[float]]  # 4x4 变换矩阵


class TrajectoryKeyframe(BaseModel):
    frame_idx: int
    pose_world: List[List[float]]  # 4x4 变换矩阵


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
        
        obj_meta_path = os.path.join(renderer.meta_dir, f"frame_{check_frame:04d}_objects.json")
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
            pose = renderer._get_frame_object_pose(check_frame, obj_id)
            
            if pose is None:
                # 从原始数据加载
                obj_meta_path = os.path.join(renderer.meta_dir, f"frame_{check_frame:04d}_objects.json")
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
        
        # 扫描场景帧数
        ego_dir = os.path.join(request.scene_path, "ego_pose")
        if os.path.exists(ego_dir):
            frame_files = [f for f in os.listdir(ego_dir) if f.startswith("frame_") and f.endswith("_ego.json")]
            scene_info["num_frames"] = len(frame_files)
        
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
        obj_meta_path = os.path.join(
            renderer.meta_dir, 
            f"frame_{frame_idx:04d}_objects.json"
        )
        
        if not os.path.exists(obj_meta_path):
            return {"objects": [], "frame_idx": frame_idx}
        
        with open(obj_meta_path, "r") as f:
            objects = json.load(f)
        
        # 添加轨迹控制器中编辑过的物体位姿
        for obj in objects:
            obj_id = obj["object_id"]
            edited_pose = renderer._get_frame_object_pose(frame_idx, obj_id)
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
    for frame_idx in range(100):  # 扫描前100帧
        obj_meta_path = os.path.join(
            renderer.meta_dir, 
            f"frame_{frame_idx:04d}_objects.json"
        )
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


@app.post("/api/render/frame")
async def render_frame(request: RenderRequest):
    """渲染单帧（增强版，支持轨迹可视化）"""
    renderer = get_scene_or_404(request.scene_id)
    
    try:
        # 构建物体覆盖
        object_overrides = {}
        for oid in renderer.trajectory_controller.get_object_ids():
            pose = renderer._get_frame_object_pose(request.frame_idx, oid)
            if pose is not None:
                object_overrides[oid] = pose
        
        # 渲染
        image = renderer._render_frame_with_object_overrides(
            request.frame_idx, 
            object_overrides
        )
        
        # 加载物体元数据和相机参数
        obj_meta_path = os.path.join(
            renderer.meta_dir, 
            f"frame_{request.frame_idx:04d}_objects.json"
        )
        ego_path = os.path.join(
            renderer.ego_dir, 
            f"frame_{request.frame_idx:04d}_ego.json"
        )
        
        objects_data = []
        if os.path.exists(obj_meta_path):
            with open(obj_meta_path, "r") as f:
                objects_data = json.load(f)
        
        ego_data = None
        if os.path.exists(ego_path):
            with open(ego_path, "r") as f:
                ego_data = json.load(f)
        
        # 绘制边界框和ID
        if (request.draw_bboxes or request.draw_ids) and ego_data:
            c2w = torch.tensor(ego_data["camera_extrinsics_world"], device=renderer.device).float()
            if c2w.shape == (3, 4):
                c2w = torch.cat([c2w, torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=renderer.device, dtype=c2w.dtype)], dim=0)
            viewmat = torch.inverse(c2w)
            K = torch.tensor(ego_data["camera_intrinsics"], device=renderer.device).float()
            W, H = ego_data["camera"]["width"], ego_data["camera"]["height"]
            
            for obj in objects_data:
                obj_id = obj["object_id"]
                
                # 获取物体位姿（考虑编辑）
                pose = renderer._get_frame_object_pose(request.frame_idx, obj_id)
                if pose is None:
                    pose = torch.tensor(obj["pose_world"], device=renderer.device).float()
                
                # 计算3D边界框
                corners_2d = renderer._get_bbox_corners_2d(
                    pose, obj["dimensions"], viewmat, K, W, H
                )
                
                if corners_2d is not None:
                    # 判断是否被编辑过
                    is_edited = obj_id in renderer.trajectory_controller.plans
                    color = (0, 165, 255) if is_edited else (0, 255, 0)  # 橙色表示已编辑
                    label = f"ID:{obj_id}" if request.draw_ids else None
                    image = renderer._draw_bbox(image, corners_2d, color=color, label=label)
                    
                    # 添加2D边界框信息到物体数据
                    x_coords = corners_2d[:, 0]
                    y_coords = corners_2d[:, 1]
                    obj["bbox_2d"] = {
                        "x_min": float(np.min(x_coords)),
                        "x_max": float(np.max(x_coords)),
                        "y_min": float(np.min(y_coords)),
                        "y_max": float(np.max(y_coords))
                    }
        
        # 绘制轨迹
        if request.draw_trajectories and ego_data:
            image = draw_all_trajectories(
                image, 
                renderer, 
                request.frame_idx, 
                ego_data,
                request.trajectory_length
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
            "objects": objects_data  # 返回物体信息，包含2D边界框
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/render/sequence")
async def render_sequence(request: SequenceRenderRequest, background_tasks: BackgroundTasks):
    """渲染视频序列（后台任务）"""
    renderer = get_scene_or_404(request.scene_id)
    
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
        current_pose = renderer._get_frame_object_pose(request.frame_idx, request.object_id)
        if current_pose is None:
            # 从原始数据加载
            obj_meta_path = os.path.join(
                renderer.meta_dir, 
                f"frame_{request.frame_idx:04d}_objects.json"
            )
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


@app.post("/api/corner_case/generate")
async def generate_corner_case(request: CornerCaseGenRequest):
    """基于轨迹编辑生成 corner case 交通事故场景。

    底层逻辑：修改相关 track 的轨迹（写入 TrackManager），与渲染/编辑管线统一。
    """
    tm = get_track_manager_or_404(request.scene_id)

    try:
        tm.push_history()
        result = corner_case.generate(
            tm,
            request.scenario_type,
            request.roles,
            request.start_frame,
            request.num_frames,
            request.intensity,
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
    """清除指定 track 的生成轨迹（恢复原始轨迹）。"""
    tm = get_track_manager_or_404(request.scene_id)
    for tid in request.track_ids:
        tm.clear_track_edits(tid)
    return {"success": True, "cleared": request.track_ids}


@app.get("/api/corner_case/types")
async def list_corner_case_types():
    """列出支持的 corner case 类型及其角色定义。"""
    return {"success": True, "types": corner_case.list_scenarios()}


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
            })

    # 相机参数
    camera = None
    scene_center = [0.0, 0.0, 0.0]
    ego_path = os.path.join(renderer.ego_dir, f"frame_{frame_idx:04d}_ego.json")
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


@app.post("/api/render/freeview")
async def render_freeview(request: FreeViewRenderRequest):
    """从任意自由视角渲染 4DGS 当前帧（服务端 gsplat 渲染）。

    返回 base64 PNG。相机由 c2w 给定，内参可由 fov_y 推导或使用自车内参缩放。
    """
    renderer = get_scene_or_404(request.scene_id)

    try:
        device = renderer.device

        c2w = torch.tensor(request.c2w, device=device, dtype=torch.float32)
        if c2w.shape == (3, 4):
            c2w = torch.cat([c2w, torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=device, dtype=c2w.dtype)], dim=0)

        # 内参：优先用 fov_y；否则把自车内参按分辨率缩放
        if request.fov_y is not None:
            K = _fov_to_intrinsics(request.fov_y, request.width, request.height, device)
        else:
            ego_path = os.path.join(renderer.ego_dir, f"frame_{request.frame_idx:04d}_ego.json")
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

        # 实时预览：拖动中的临时位姿（未提交）覆盖该 track 在本帧的 raw object
        if tm and request.live_track_id is not None and request.live_pose_matrix is not None:
            raw_id = tm.get_raw_object_id(request.live_track_id, request.frame_idx)
            if raw_id is not None:
                live_pose = torch.tensor(request.live_pose_matrix, device=device).float()
                object_overrides[raw_id] = live_pose

        image = renderer._render_frame_with_object_overrides(
            request.frame_idx,
            object_overrides,
            c2w_override=c2w,
            K_override=K,
            width_override=request.width,
            height_override=request.height,
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
    """恢复被删除/清除编辑的 track。"""
    tm = get_track_manager_or_404(request.scene_id)
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


# ==================== 启动配置 ====================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
