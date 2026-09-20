"""SAM (Segment Anything) 交互式分割：点提示 -> 二值掩码。

支持在 ViT-B / ViT-L / ViT-H 之间自由切换（前端可选），权重放在 SAM_CHECKPOINT_DIR。
运行在 dggt 后端环境，模型懒加载、可切换。
"""
import os
import io
import threading
import base64

import numpy as np
import torch
from PIL import Image

SAM_CHECKPOINT_DIR = os.environ.get(
    "SAM_CHECKPOINT_DIR", "/root/autodl-fs/dggt-main/sam-3d-objects/SAM_checkpoints"
)
# 模型类型 -> 权重文件名（segment-anything 官方命名）
_MODEL_FILES = {
    "vit_b": "sam_vit_b_01ec64.pth",
    "vit_l": "sam_vit_l_0b3195.pth",
    "vit_h": "sam_vit_h_4b8939.pth",
}
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_predictor = None
_current_model = os.environ.get("SAM_MODEL_TYPE", "vit_b")
_lock = threading.Lock()
_load_error = None


def available_models():
    """返回可用模型列表 [{type, path, exists, size_mb}]。"""
    out = []
    for mtype, fname in _MODEL_FILES.items():
        path = os.path.join(SAM_CHECKPOINT_DIR, fname)
        size_mb = round(os.path.getsize(path) / 1e6, 0) if os.path.exists(path) else None
        out.append({
            "type": mtype,
            "path": path,
            "exists": os.path.exists(path),
            "size_mb": size_mb,
        })
    return out


def current_model():
    return _current_model


def set_model(model_type):
    """切换分割模型（释放旧模型，下次 segment 懒加载新模型）。"""
    global _current_model, _predictor, _load_error
    if model_type not in _MODEL_FILES:
        raise ValueError(f"不支持的模型类型: {model_type}，可选 {list(_MODEL_FILES)}")
    with _lock:
        if _predictor is not None:
            del _predictor
            _predictor = None
        _load_error = None
        _current_model = model_type
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return _current_model


def _checkpoint_path(model_type=None):
    model_type = model_type or _current_model
    return os.path.join(SAM_CHECKPOINT_DIR, _MODEL_FILES[model_type])


def _load_predictor():
    global _predictor, _load_error
    if _predictor is not None or _load_error is not None:
        return
    ckpt = _checkpoint_path()
    if not os.path.exists(ckpt):
        _load_error = f"SAM 权重不存在: {ckpt}"
        return
    from segment_anything import sam_model_registry, SamPredictor

    sam = sam_model_registry[_current_model](checkpoint=ckpt)
    sam.to(DEVICE).eval()
    _predictor = SamPredictor(sam)


def get_predictor():
    with _lock:
        _load_predictor()
        if _predictor is None:
            raise RuntimeError(f"SAM 分割器未就绪: {_load_error}")
        return _predictor


def segment(
    image_rgb: np.ndarray,
    points,
    point_labels,
) -> tuple:
    """对 RGB 图做点提示分割。

    Args:
        image_rgb: HxWx3 uint8 RGB
        points: 像素坐标列表 [[x, y], ...]
        point_labels: 与 points 等长，1=前景, 0=背景

    Returns:
        (mask: HxW bool, score: float)
    """
    predictor = get_predictor()
    predictor.set_image(image_rgb)
    coords = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    labels = np.asarray(point_labels, dtype=np.int64).reshape(-1)
    if coords.shape[0] != labels.shape[0]:
        raise ValueError("points 与 point_labels 数量不一致")
    masks, scores, _ = predictor.predict(
        point_coords=coords,
        point_labels=labels,
        multimask_output=False,
    )
    return masks[0].astype(bool), float(scores[0])


def release_memory():
    """重建前彻底释放 SAM 模型显存，给 SAM 3D 让路（下次 segment 会懒重载）。"""
    global _predictor
    if _predictor is not None:
        del _predictor
        _predictor = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def mask_to_png_bytes(mask: np.ndarray) -> bytes:
    """bool 掩码 -> PNG 字节（白=前景）。"""
    img = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def mask_to_base64(mask: np.ndarray) -> str:
    return base64.b64encode(mask_to_png_bytes(mask)).decode("ascii")
