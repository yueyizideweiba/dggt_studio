# Copyright (c) Meta Platforms, Inc. and affiliates.
"""
SAM 3D Objects 推理微服务。

用法（在独立的 sam3d-objects conda 环境中）:
    conda activate /root/autodl-tmp/conda_envs/sam3d-objects   # 或你的环境名
    cd /root/autodl-fs/dggt-main/sam-3d-objects
    python sam3d_service.py

默认监听 0.0.0.0:8001，供 DGGT 后端 (dggt 环境, 端口 8000) 通过 HTTP 调用。

端点:
    GET  /health        -> 服务与模型/GPU 状态
    POST /reconstruct   -> multipart: image(必填) + mask(可选) + seed + format
                           返回重建出的 3D Gaussian Splat (.ply)，可选 mesh (.glb)

设计要点:
    - 模型懒加载（首次 /reconstruct 时加载约 13GB 权重并常驻内存）。
    - /health 无需加载模型即可返回，便于后端探测服务是否在线。
    - 输出写入 SAM3D_OUTPUT_DIR，文件名带时间戳与随机后缀，避免并发冲突。
"""
import io
import os
import sys
import json
import time
import uuid
import threading
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

# ---- 路径与导入: 保证不依赖 pip install -e 也能运行 ----
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
if str(HERE / "notebook") not in sys.path:
    sys.path.insert(0, str(HERE / "notebook"))

# notebook/inference.py 顶部依赖 CONDA_PREFIX 设置 CUDA_HOME
if "CUDA_HOME" not in os.environ and "CONDA_PREFIX" in os.environ:
    os.environ["CUDA_HOME"] = os.environ["CONDA_PREFIX"]

# ---- 配置 ----
CONFIG_PATH = os.environ.get("SAM3D_CONFIG_PATH", "checkpoints/pipeline.yaml")
OUTPUT_DIR = Path(os.environ.get("SAM3D_OUTPUT_DIR", str(HERE / "outputs")))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
HOST = os.environ.get("SAM3D_HOST", "0.0.0.0")
PORT = int(os.environ.get("SAM3D_PORT", "8001"))

app = FastAPI(title="SAM 3D Objects Service", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- 全局模型状态（懒加载 + 线程安全） ----
_model = None
_model_lock = threading.Lock()
_model_error: str = None
_model_loaded = False


def _load_inference_class():
    """延迟导入 Inference，避免服务启动即加载重依赖，/health 因此始终可用。"""
    from inference import Inference
    return Inference


def get_model():
    """获取（必要时加载）SAM 3D 推理器。"""
    global _model, _model_error, _model_loaded
    with _model_lock:
        if _model is not None:
            return _model
        if _model_error is not None:
            raise RuntimeError(f"SAM 3D 模型加载失败: {_model_error}")
        if not torch.cuda.is_available():
            msg = "未检测到可用 GPU，SAM 3D 需要 NVIDIA GPU（建议 >= 32GB 显存）"
            _model_error = msg
            raise RuntimeError(msg)
        try:
            Inference = _load_inference_class()
            _model = Inference(CONFIG_PATH, compile=False)
            _model_loaded = True
        except Exception as e:  # noqa: BLE001
            _model_error = str(e)
            raise RuntimeError(f"SAM 3D 模型加载失败: {e}") from e
        return _model


def unload_model():
    """卸载已加载的模型并释放显存（供后端在重建完成后调用）。

    说明：spconv / warp / torch 各持有自己的 CUDA 内存池，单次 empty_cache 未必全部
    归还；这里做多次 gc + synchronize + empty_cache，尽量回收。
    """
    global _model, _model_loaded
    with _model_lock:
        _model = None
        _model_loaded = False
    import gc
    for _ in range(3):
        gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        except Exception:  # noqa: BLE001
            pass
    return True


def _read_image(bytes_: bytes, mode: str) -> np.ndarray:
    return np.asarray(Image.open(io.BytesIO(bytes_)).convert(mode))


def _prepare_image_and_mask(image_bytes: bytes, mask_bytes) -> tuple:
    """返回 (rgb uint8 ndarray [H,W,3], mask bool ndarray [H,W])。

    支持两种输入:
      1) 单独上传 mask 文件（灰度图，>127 视为前景）。
      2) 不传 mask 时，使用 RGBA 图片的 alpha 通道作为 mask。
    """
    if mask_bytes is not None:
        img = _read_image(image_bytes, "RGB")
        m = _read_image(mask_bytes, "L")
        # 掩码尺寸需与图片一致（Inference 内部会按 alpha 通道拼接）
        if m.shape[:2] != img.shape[:2]:
            m = np.asarray(
                Image.fromarray(m).resize((img.shape[1], img.shape[0]), Image.NEAREST)
            )
        mask = m > 127
        return img, mask

    rgba = np.asarray(Image.open(io.BytesIO(image_bytes)).convert("RGBA"))
    alpha = rgba[..., 3]
    if int(alpha.max()) == 0 or int(alpha.max()) == int(alpha.min()):
        raise ValueError(
            "未提供 mask 且图片没有有效的 alpha 通道，请上传独立的 mask 文件或带 alpha 的 RGBA 图片"
        )
    img = rgba[..., :3]
    mask = alpha > 127
    return img, mask


@app.get("/health")
async def health():
    """健康检查：不触发模型加载。"""
    global _model_error, _model_loaded
    return {
        "status": "ok",
        "service": "sam-3d-objects",
        "gpu_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "model_loaded": _model_loaded,
        "model_error": _model_error,
        "config_path": CONFIG_PATH,
        "output_dir": str(OUTPUT_DIR),
    }


@app.post("/unload")
async def unload_endpoint():
    """卸载模型并退出进程，让 run_sam3d_service.sh 监督循环重启。

    进程退出可彻底归还 GPU 显存（spconv/warp/torch 的 CUDA 内存池靠 empty_cache
    无法完全释放），重启后模型懒加载、空闲不占显存。
    """
    unload_model()

    def _exit_later():
        import os as _os
        import time as _time
        _time.sleep(2.0)  # 留时间把 HTTP 响应写完
        _os._exit(0)

    threading.Thread(target=_exit_later, daemon=True).start()
    return {"success": True, "model_loaded": _model_loaded, "restarting": True}


@app.post("/reconstruct")
async def reconstruct(
    image: UploadFile = File(...),
    mask: UploadFile = File(None),
    seed: int = Form(None),
    format: str = Form("ply"),  # "ply" | "ply+glb"
):
    """从单张图片 + mask 重建 3D Gaussian Splat（可选 mesh）。

    返回 JSON:
        {
          "success": true,
          "ply_path": "<绝对路径>",
          "glb_path": "<绝对路径或 null>",
          "pose": {"rotation": [...], "translation": [...], "scale": [...]},
          "num_points": int,
          "seed": int
        }
    """
    try:
        image_bytes = await image.read()
        mask_bytes = await mask.read() if mask is not None else None

        rgb, mask_arr = _prepare_image_and_mask(image_bytes, mask_bytes)

        inference = get_model()

        # 同步执行（服务端单 GPU 模型，串行更安全；如需并发可自行加队列）
        output = inference(rgb, mask_arr, seed=seed)

        gs = output.get("gs")
        if gs is None:
            raise RuntimeError("重建结果中缺少 Gaussian Splat（'gs'）")

        job_id = time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
        ply_path = OUTPUT_DIR / f"sam3d_{job_id}.ply"
        gs.save_ply(str(ply_path))

        glb_path = None
        if format == "ply+glb":
            glb = output.get("glb")
            if glb is not None:
                glb_path = OUTPUT_DIR / f"sam3d_{job_id}.glb"
                # trimesh.Trimesh.export
                glb.export(str(glb_path))

        # 提取位姿（列表/张量转 float list，安全序列化）
        def _to_list(v, n=None):
            if v is None:
                return None
            if isinstance(v, torch.Tensor):
                v = v.detach().cpu().numpy()
            arr = np.asarray(v).reshape(-1).tolist()
            return arr

        pose = {
            "rotation": _to_list(output.get("rotation")),
            "translation": _to_list(output.get("translation")),
            "scale": _to_list(output.get("scale")),
        }
        num_points = int(gs.get_xyz.shape[0]) if hasattr(gs.get_xyz, "shape") else None

        return {
            "success": True,
            "ply_path": str(ply_path),
            "glb_path": str(glb_path) if glb_path else None,
            "pose": pose,
            "num_points": num_points,
            "seed": seed,
        }
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/outputs/{filename}")
async def download_output(filename: str):
    """按文件名下载已生成的重建结果（.ply / .glb）。"""
    path = OUTPUT_DIR / Path(filename).name
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"未找到输出文件 {filename}")
    return FileResponse(path, filename=path.name)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=HOST, port=PORT)
