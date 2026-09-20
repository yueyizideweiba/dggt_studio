"""文本 → 实体：LLaDA-Image 生成参考图 → 自动掩码 → SAM3D 重建 → VLM 定尺寸/朝向
→ 沿道路生成物理合理且无空间冲突的轨迹并插入场景。

设计要点（24G 单卡）：
  * LLaDA / VLM / SAM3D 三个模型**串行**加载：每步用完就让对应微服务 `/unload`，
    再进入下一步，避免显存叠加 OOM。
  * 道路几何直接用"自车轨迹"作为车道中心线（这条线一定在可重建的路面上），
    横向偏移得到相邻/对向车道；朝向取中心线切向，保证车沿合理路径运动。
  * 插入前用 2D OBB(SAT) 对所有已有物体逐帧做冲突检测；有冲突就把候选位置
    沿横向/纵向小步挪动，直到无冲突（或返回失败原因）。
"""
from __future__ import annotations

import base64
import io
import json
import math
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

TEXT2ENTITY_URL = os.environ.get("TEXT2ENTITY_URL", "http://127.0.0.1:8002")


# ==================== 微服务客户端 ====================

def _svc_post(path: str, body: Dict[str, Any], timeout: float = 600.0, retries: int = 2) -> Dict[str, Any]:
    """POST 微服务；遇到传输错误（例如微服务正在重启/被杀）先等它就绪再重试。"""
    import time as _time
    import httpx
    last = None
    for i in range(retries + 1):
        try:
            with httpx.Client(timeout=timeout) as c:
                r = c.post(f"{TEXT2ENTITY_URL.rstrip('/')}{path}", json=body)
            if r.status_code != 200:
                raise RuntimeError(f"text2entity 服务 {path} 失败 (HTTP {r.status_code}): {r.text[:400]}")
            return r.json()
        except RuntimeError:
            raise
        except Exception as e:  # noqa: BLE001  (httpx.TransportError / RemoteProtocolError ...)
            last = e
            if i >= retries:
                break
            try:
                wait_ready(timeout=180.0)
            except Exception:  # noqa: BLE001
                pass
            _time.sleep(2.0)
    raise RuntimeError(f"text2entity 服务 {path} 连接失败: {last}")


def health() -> Dict[str, Any]:
    import httpx
    try:
        with httpx.Client(timeout=10.0) as c:
            return c.get(f"{TEXT2ENTITY_URL.rstrip('/')}/health").json()
    except Exception as e:  # noqa: BLE001
        return {"status": "down", "error": str(e)}


def unload_service() -> None:
    """让微服务释放显存（进程退出，由 supervisor 重启）。"""
    import httpx
    try:
        with httpx.Client(timeout=60.0) as c:
            c.post(f"{TEXT2ENTITY_URL.rstrip('/')}/unload")
    except Exception:  # noqa: BLE001
        pass


def wait_ready(timeout: float = 180.0, interval: float = 2.0) -> bool:
    import time
    deadline = time.time() + timeout
    while time.time() < deadline:
        if health().get("status") == "ok":
            return True
        time.sleep(interval)
    return False


# ==================== 参考图 / 掩码 ====================

REFERENCE_SUFFIX = (", a single isolated object, fully visible and centered, pure white seamless background, "
                    "no shadow, no other objects, no text, professional product photograph, 3/4 front view, "
                    "high detail, sharp focus")

# ---- 本地兜底：sd-turbo（DGGT 仓库里已经有 Difix 用的 sd-turbo 权重，免下载） ----
SD_TURBO_DIR = os.environ.get("DGGT_SD_TURBO_DIR", "/root/autodl-tmp/hf_home/sd-turbo")
_sd_pipe = None


def local_text2image_available() -> bool:
    return os.path.isdir(SD_TURBO_DIR) and os.path.exists(os.path.join(SD_TURBO_DIR, "unet"))


def _local_text2image(prompt: str, *, width: int = 512, height: int = 512,
                      steps: int = 4, guidance: float = 0.0, seed: int = 0) -> bytes:
    """用本地 sd-turbo 文生图（网络不可用时兜底；LLaDA 才是主模型）。"""
    global _sd_pipe
    import torch
    # dggt 环境的 transformers 5.x 删掉了 diffusers 0.30 需要的几个旧符号，先补齐
    try:
        import transformers.utils as tu
        for n, v in (("FLAX_WEIGHTS_NAME", "flax_model.msgpack"),
                     ("TF2_WEIGHTS_NAME", "tf_model.h5"),
                     ("SAFETENSORS_WEIGHTS_NAME", "model.safetensors")):
            if not hasattr(tu, n):
                setattr(tu, n, v)
    except Exception:  # noqa: BLE001
        pass
    from diffusers import StableDiffusionPipeline
    if _sd_pipe is None:
        _sd_pipe = StableDiffusionPipeline.from_pretrained(
            SD_TURBO_DIR, torch_dtype=torch.float16, variant="fp16",
            safety_checker=None, requires_safety_checker=False)
        _sd_pipe.to("cuda")
    img = _sd_pipe(prompt, num_inference_steps=int(steps) or 4,
                   guidance_scale=float(guidance), height=int(height), width=int(width),
                   generator=torch.Generator("cuda").manual_seed(int(seed))).images[0]
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def release_local_text2image() -> None:
    """释放本地 sd-turbo 显存（插入前要腾给 SAM3D）。"""
    global _sd_pipe
    if _sd_pipe is not None:
        _sd_pipe = None
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass


def generate_reference(prompt: str, *, width: int = 1024, height: int = 1024,
                       steps: int = 4, guidance: float = 1.0, seed: int = 0,
                       negative_prompt: str = "") -> bytes:
    """文本 → 白底参考图：优先 LLaDA-Image 微服务，不可用时退回本地 sd-turbo。"""
    full = prompt + REFERENCE_SUFFIX
    h = health()
    if h.get("status") == "ok" and h.get("prompt2image"):
        wait_ready()
        body = {"prompt": full, "negative_prompt": negative_prompt,
                "width": width, "height": height, "steps": steps,
                "guidance_scale": guidance, "seed": int(seed)}
        r = _svc_post("/generate", body, timeout=900.0)
        if not r.get("success"):
            raise RuntimeError(r.get("detail") or "生成参考图失败")
        return base64.b64decode(r["image"])
    if not local_text2image_available():
        raise RuntimeError("LLaDA 微服务未就绪，且本地 sd-turbo 也不可用（检查 DGGT_SD_TURBO_DIR）")
    return _local_text2image(full, width=512, height=512, steps=max(1, min(6, int(steps))),
                             guidance=0.0, seed=int(seed))


def auto_mask(image_bytes: bytes, bg_thresh: int = 240, min_ratio: float = 0.002) -> bytes:
    """白底参考图 → 前景掩码（PNG L 通道，>127 为前景）。"""
    import cv2
    arr = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    if arr is None:
        raise RuntimeError("参考图解码失败")
    hsv = cv2.cvtColor(arr, cv2.COLOR_BGR2HSV)
    # 背景近似纯白：低饱和 + 高亮度；其余为前景
    fg = ((hsv[..., 1] > 28) | (hsv[..., 2] < bg_thresh)).astype(np.uint8) * 255
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(fg, 8)
    if n > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]
        keep = np.zeros_like(fg)
        total = fg.shape[0] * fg.shape[1]
        for i, a in enumerate(areas, start=1):
            if a >= max(total * min_ratio, 200):
                keep[lab == i] = 255
        if keep.any():
            fg = keep
    # 填充内部空洞（否则 SAM3D 会把车顶/车窗当背景）
    cnts, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(fg)
    for c in cnts:
        cv2.drawContours(filled, [c], -1, 255, -1)
    ok, buf = cv2.imencode(".png", filled)
    if not ok:
        raise RuntimeError("掩码编码失败")
    return buf.tobytes()


def _clean_mask(mask: np.ndarray) -> np.ndarray:
    """填洞 + 只保留最大连通域（去掉 SAM 的零碎小片）。"""
    import cv2
    m = (mask.astype(np.uint8)) * 255
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(m)
    if cnts:
        c = max(cnts, key=cv2.contourArea)
        cv2.drawContours(filled, [c], -1, 255, -1)
    return filled > 127


def mask_reference(image_bytes: bytes) -> bytes:
    """给生成的参考图打掩码：优先用 SAM 点提示（生成图常有渐变底/阴影），失败退回白底阈值。

    注意：要选**面积最大的显著目标**（背景通常 >80% 被排除），而不是 score 最高的那个
    ——SAM 常给出一个 score 很高但只有 1% 面积的小碎片。
    用完会释放 SAM 显存（紧接着要加载 SAM3D）。
    """
    import cv2
    arr = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    if arr is not None:
        H, W = arr.shape[:2]
        try:
            import sam_segment
            rgb = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
            cands = []
            pts = [(0.5, 0.55), (0.5, 0.5), (0.5, 0.62), (0.45, 0.55), (0.55, 0.55),
                   (0.5, 0.42), (0.42, 0.5), (0.58, 0.5)]
            for fx, fy in pts:
                try:
                    m, s = sam_segment.segment(rgb, [[fx * W, fy * H]], [1])
                except Exception:  # noqa: BLE001
                    continue
                area = float(m.mean())
                if 0.03 <= area <= 0.80:          # 排除背景(>0.8) 和碎片(<0.03)
                    cands.append((area, float(s), m))
            sam_segment.release_memory()
            if cands:
                cands.sort(key=lambda t: (t[0], t[1]), reverse=True)
                return sam_segment.mask_to_png_bytes(_clean_mask(cands[0][2]))
        except Exception:  # noqa: BLE001
            pass
    return auto_mask(image_bytes)


# ==================== VLM 尺寸/类别推理 ====================

_VLM_PROMPT = (
    "你在为自动驾驶仿真场景挑选一个 3D 物体。用户的文字要求是：「{prompt}」。\n"
    "下面这张图是该文字生成的参考图。请只输出一个 JSON（不要 markdown、不要解释）：\n"
    '{{"matches_prompt": true/false, "short_desc": "图里物体的一句话中文描述", '
    '"category": "简短英文类别", "is_vehicle": true/false, '
    '"length_m": 数字, "width_m": 数字, "height_m": 数字, '
    '"facing": "front|back|left|right|unknown", "confidence": 0~1}}\n'
    "matches_prompt：图里的物体与文字要求是否一致（一致=true，是别的东西/看不出= false）。\n"
    "尺寸用真实世界米数估计（车辆按典型轿车/卡车/公交/摩托；行人/自行车/锥桶等按常识）。"
)


def _parse_json(text: str) -> Optional[dict]:
    if not text:
        return None
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:  # noqa: BLE001
        return None


def vlm_analyze(image_bytes: bytes, prompt_text: str = "", timeout: float = 300.0) -> Dict[str, Any]:
    b64 = base64.b64encode(image_bytes).decode("ascii")
    prompt = _VLM_PROMPT.format(prompt=prompt_text or "(未提供)")
    r = _svc_post("/vlm", {"prompt": prompt, "images": [b64]}, timeout=timeout)
    text = r.get("text") or ""
    data = _parse_json(text) or {}
    return {"raw": text, **data}


def vlm_matches(vlm: Dict[str, Any]) -> bool:
    """VLM 判断"生成图与文字要求是否一致"。没给判断时不拦（返回 True）。"""
    if not vlm:
        return True
    if "matches_prompt" not in vlm:
        return True
    return bool(vlm.get("matches_prompt"))


# 典型物体先验尺寸 [长, 宽, 高]（米），VLM 不可信时兜底
_PRIORS = {
    "car": (4.6, 1.85, 1.5), "sedan": (4.7, 1.85, 1.47), "suv": (4.8, 1.95, 1.75),
    "truck": (8.5, 2.5, 3.2), "van": (5.4, 2.0, 2.2), "bus": (11.0, 2.55, 3.2),
    "motorcycle": (2.1, 0.8, 1.2), "bicycle": (1.8, 0.6, 1.6),
    "pedestrian": (0.6, 0.6, 1.75), "person": (0.6, 0.6, 1.75),
    "cone": (0.4, 0.4, 0.7), "barrier": (1.5, 0.5, 1.0), "box": (1.0, 1.0, 1.0),
}


def estimate_dimensions(vlm: Dict[str, Any], fallback=(4.6, 1.85, 1.5)) -> Tuple[List[float], str]:
    """返回 (DGGT 尺寸 [宽,高,长], 类别)。把 VLM 的 [长,宽,高] 转成 [宽,高,长]。"""
    cat = str(vlm.get("category") or "").strip().lower()
    prior = fallback
    for k, v in _PRIORS.items():
        if k in cat:
            prior = v
            break
    if not cat:
        cat = "object"
    try:
        L = float(vlm.get("length_m") or prior[0])
        W = float(vlm.get("width_m") or prior[1])
        H = float(vlm.get("height_m") or prior[2])
    except Exception:  # noqa: BLE001
        L, W, H = prior
    # 合理性夹取：偏离先验太多就用先验（3B VLM 的米数估计经常离谱）
    def _clamp(x, ref, lo=0.45, hi=2.2):
        return ref if not (ref * lo <= x <= ref * hi) else x
    L, W, H = _clamp(L, prior[0]), _clamp(W, prior[1]), _clamp(H, prior[2])
    return [round(W, 3), round(H, 3), round(L, 3)], cat


# 中文/英文关键词 → 类别（VLM 不可用时的兜底，先用先验尺寸）
_KEYWORDS = [
    ("bus", ["公交", "巴士", "客车", "bus"]),
    ("truck", ["卡车", "货车", "泥头车", "truck", "lorry"]),
    ("van", ["面包车", "厢式", "van"]),
    ("suv", ["suv", "越野"]),
    ("motorcycle", ["摩托", "机车", "motorcycle", "motorbike"]),
    ("bicycle", ["自行车", "单车", "bicycle", "bike"]),
    ("pedestrian", ["行人", "路人", "pedestrian", "person", "人"]),
    ("cone", ["锥桶", "路锥", "锥", "cone"]),
    ("barrier", ["护栏", "栏杆", "围栏", "barrier", "fence"]),
    ("sedan", ["轿车", "小汽车", "汽车", "小车", "轿", "sedan", "car", "车"]),
]


def rule_dimensions(prompt: str) -> Tuple[List[float], str]:
    """不靠 VLM 的兜底：关键词 → 类别 → 先验尺寸，返回 ([宽,高,长], 类别)。"""
    p = (prompt or "").lower()
    for cat, kws in _KEYWORDS:
        if any(k.lower() in p for k in kws):
            L, W, H = _PRIORS.get(cat, _PRIORS["car"])
            return [round(W, 3), round(H, 3), round(L, 3)], cat
    return [round(_PRIORS["car"][1], 3), round(_PRIORS["car"][2], 3), round(_PRIORS["car"][0], 3)], "object"


def vlm_facing(vlm: Dict[str, Any]) -> str:
    """VLM 判定的朝向：front/back/left/right/unknown。back 时插入要翻 180°。"""
    f = str((vlm or {}).get("facing") or "").strip().lower()
    return f if f in ("front", "back", "left", "right") else "unknown"


# ==================== 2D OBB / SAT ====================

def _obb_corners_2d(pose: np.ndarray, dims) -> np.ndarray:
    """返回 4x2 的 XZ 角点。dims=[宽,高,长]，局部 X=宽、Z=长。"""
    pose = np.asarray(pose, dtype=np.float64)
    R = pose[:3, :3]
    c = pose[:3, 3]
    ex = R[:, 0]
    ez = R[:, 2]
    hw, hl = float(dims[0]) / 2.0, float(dims[2]) / 2.0
    pts = [c + ex * hw + ez * hl, c - ex * hw + ez * hl,
           c - ex * hw - ez * hl, c + ex * hw - ez * hl]
    return np.array([[p[0], p[2]] for p in pts], dtype=np.float64)


def _obb_overlap_2d(a: np.ndarray, b: np.ndarray, margin: float = 0.15) -> bool:
    """分离轴定理：两个 2D 矩形是否重叠（含 margin 外扩）。"""
    for poly in (a, b):
        for i in range(4):
            e = poly[(i + 1) % 4] - poly[i]
            axis = np.array([-e[1], e[0]])
            n = np.linalg.norm(axis)
            if n < 1e-9:
                continue
            axis = axis / n
            pa, pb = a @ axis, b @ axis
            if pa.max() + margin <= pb.min() or pb.max() + margin <= pa.min():
                return False
    return True


# ==================== 道路几何（用自车轨迹当车道中心线） ====================

def _ego_polyline(tm) -> List[Tuple[int, np.ndarray]]:
    pts = []
    if getattr(tm, "ego_track_id", None) is not None:
        for f in tm.get_track_frames(tm.ego_track_id):
            p = tm.get_track_pose(tm.ego_track_id, f)
            if p is not None:
                pts.append((int(f), np.asarray(p[:3, 3], dtype=np.float64)))
    if len(pts) < 2:
        # 退路：用原始自车相机位置
        cams = getattr(tm, "_ego_cams", {})
        pts = [(int(f), np.asarray(c[:3, 3], dtype=np.float64)) for f, c in sorted(cams.items())]
    return pts


class _Lane:
    """由折线构建的可按弧长取点的中心线。"""

    def __init__(self, pts: List[Tuple[int, np.ndarray]]):
        self.frames = [f for f, _ in pts]
        self.xyz = np.array([p for _, p in pts], dtype=np.float64)
        self.s = np.zeros(len(self.xyz))
        for i in range(1, len(self.xyz)):
            self.s[i] = self.s[i - 1] + float(np.linalg.norm(self.xyz[i, [0, 2]] - self.xyz[i - 1, [0, 2]]))

    def point_tangent(self, s: float):
        """按弧长取 (世界点, 水平切向)；**允许超出端点外推**（否则前进的物体会被钉在路径尽头）。"""
        if len(self.xyz) == 0:
            return None, None
        if len(self.xyz) == 1:
            return self.xyz[0], np.array([1.0, 0.0])
        s = float(s)
        if s <= 0.0:
            d = self.xyz[1] - self.xyz[0]
            t = np.array([d[0], d[2]], dtype=np.float64)
            n = np.linalg.norm(t)
            t = np.array([1.0, 0.0]) if n < 1e-9 else t / n
            p = self.xyz[0] + np.array([t[0], 0.0, t[1]]) * s
            return p, t
        if s >= self.s[-1]:
            d = self.xyz[-1] - self.xyz[-2]
            t = np.array([d[0], d[2]], dtype=np.float64)
            n = np.linalg.norm(t)
            t = np.array([1.0, 0.0]) if n < 1e-9 else t / n
            p = self.xyz[-1] + np.array([t[0], 0.0, t[1]]) * (s - self.s[-1])
            return p, t
        i = int(np.searchsorted(self.s, s, side="right") - 1)
        i = max(0, min(len(self.xyz) - 2, i))
        seg = self.s[i + 1] - self.s[i]
        a = 0.0 if seg < 1e-6 else (s - self.s[i]) / seg
        p = self.xyz[i] + (self.xyz[i + 1] - self.xyz[i]) * a
        d = self.xyz[i + 1] - self.xyz[i]
        t = np.array([d[0], d[2]], dtype=np.float64)
        n = np.linalg.norm(t)
        t = np.array([1.0, 0.0]) if n < 1e-9 else t / n
        return p, t

    def arc_at_frame(self, frame: int) -> float:
        if not self.frames:
            return 0.0
        fr = np.asarray(self.frames)
        i = int(np.argmin(np.abs(fr - int(frame))))
        return float(self.s[i])

    def mean_speed(self, fps: float = 10.0) -> float:
        """自车沿中心线的平均速度（m/s），用作"跟车同速"的默认速度。"""
        if len(self.frames) < 2:
            return 8.0
        dt = (self.frames[-1] - self.frames[0]) / max(1e-6, float(fps))
        v = (self.s[-1] - self.s[0]) / max(1e-6, dt)
        return float(np.clip(v, 1.0, 30.0))


def _yaw_pose(center: np.ndarray, yaw: float) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    P = np.eye(4, dtype=np.float32)
    P[:3, 0] = (c, 0.0, -s)
    P[:3, 1] = (0.0, 1.0, 0.0)
    P[:3, 2] = (s, 0.0, c)
    P[:3, 3] = center
    return P


def _ground_setup(tm):
    """返回 (静态点, up_sign) 供贴地使用。"""
    from track_manager import _scene_points_cpu, _ground_height_near  # noqa: F401
    pts = _scene_points_cpu(tm.renderer)
    up_sign = -1.0
    cams = getattr(tm, "_ego_cams", {})
    if cams:
        down_ys = [float(np.asarray(c, dtype=np.float32)[1, 1]) for c in cams.values()]
        up_sign = -1.0 if float(np.median(down_ys)) > 0 else 1.0
    return pts, up_sign


# ==================== 轨迹规划 + 冲突规避 ====================

def scene_reference_dims(tm) -> Optional[List[float]]:
    """场景里已有"车辆类"物体的中位尺寸 [宽,高,长]，用于把自车/生成车的尺寸拉回合理范围。"""
    rows = []
    for _tid, meta in (getattr(tm, "track_meta", {}) or {}).items():
        d = meta.get("dimensions") or []
        if len(d) != 3:
            continue
        w, h, l = float(d[0]), float(d[1]), float(d[2])
        if 2.5 <= l <= 6.5 and 1.0 <= w <= 2.8 and 0.7 <= h <= 2.5 and l > w:
            rows.append((w, h, l))
    if len(rows) < 3:
        return None
    return [float(x) for x in np.median(np.array(rows), axis=0)]


COMMON_CAR_DIMS = [1.85, 1.45, 4.6]        # [宽, 高, 长] 常识兜底（轿车）


def is_vehicle_cat(cat) -> bool:
    """按类别名粗判是不是"车"（轿车/SUV/面包车…；卡车/巴士/公交不含在内）。"""
    c = str(cat or "").lower()
    if not c:
        return False
    if any(k in c for k in ("truck", "bus", "卡车", "货车", "巴士", "公交")):
        return False
    return any(k in c for k in ("car", "sedan", "suv", "van", "vehicle", "汽车", "轿车", "面包车"))


def refine_vehicle_dims(dims, cat, tm, blend: float = 0.25,
                        lo_ratio: float = 0.75, hi_ratio: float = 1.12) -> List[float]:
    """车辆类：把尺寸和"场景已有车辆"对齐，避免生成的车明显偏大。

    为什么不能只用 VLM/常识的绝对值：4DGS 场景里车辆的**包围盒**是按可见高斯拟合出来的
    （普遍比真车小，例如本场景中位 [1.27, 1.01, 3.51]），而 SAM3D 重建出的是"真车比例"。
    只要目标高度给到 1.45m，渲染出来的车就会比场景里的车高一大截 —— 用户看到的就是
    "生成的车过大 / 和主车一样过大"。所以车辆类的目标尺寸以**场景中位车辆**为基准，
    VLM 的估计只保留少量权重（`blend`）并夹在 `[lo_ratio, hi_ratio]` 倍以内。

    非车辆（行人/锥桶等）原样返回；卡车/巴士/公交不被轿车中位数拉小。
    """
    c = str(cat or "").lower()
    vehicle = any(k in c for k in ("car", "sedan", "suv", "van", "vehicle",
                                   "汽车", "轿车", "车"))
    if not vehicle or any(k in c for k in ("truck", "bus", "卡车", "货车", "巴士", "公交")):
        return [round(float(x), 3) for x in dims]
    ref = scene_reference_dims(tm) or COMMON_CAR_DIMS
    # 类别系数：SUV/面包车略大一点，轿车 1.0（仍然夹在合理范围内，不会"大一圈"）
    if "suv" in c or "越野" in c:
        factor = 1.06
    elif "van" in c or "面包" in c or "mpv" in c:
        factor = 1.10
    else:
        factor = 1.00
    out = []
    for i in range(3):
        base = float(ref[i]) * factor
        v = (1.0 - blend) * base + blend * float(dims[i])
        lo, hi = float(ref[i]) * float(lo_ratio), float(ref[i]) * float(hi_ratio)
        out.append(round(max(lo, min(hi, v)), 3))
    return out


def plan_trajectory(tm, dims, *, mode: str = "ahead", start_frame: int = 0,
                    num_frames: int = 20, distance: float = 12.0, lateral: float = 0.0,
                    speed: float = 0.0, direction_along_ego: bool = True,
                    flip: bool = False, fps: float = 10.0) -> Dict[str, Any]:
    """在自车车道上规划一条轨迹：返回 poses / 冲突 / 参数。

    `flip=True`：绕世界 Y 轴再叠加 180°（用于 VLM 判定"模型朝向反了"时纠正）。
    """
    from track_manager import _ground_height_near, _smooth_series
    lane = _Lane(_ego_polyline(tm))
    if len(lane.xyz) < 2:
        return {"ok": False, "reason": "没有可用的自车轨迹"}
    pts, up_sign = _ground_setup(tm)
    # 全局路面参考高度（只作为"离群值夹取"的基准，不直接当地面用）
    ref_gy = None
    if pts is not None and len(pts):
        ref_gy = float(np.percentile(pts[:, 1], 55.0 if up_sign < 0 else 45.0))
    # 自车车体高度：静态点覆盖不到的远端，用"自车车体**底面**"当路面
    # （`p[1]` 是自车车体**中心**高度，直接当地面会让物体浮空半个车高）
    ego_h = None
    try:
        d_ego = tm.get_track_dimensions(900000)
        if d_ego and len(d_ego) == 3:
            ego_h = float(d_ego[1])
    except Exception:  # noqa: BLE001
        ego_h = None
    if not ego_h:
        ego_h = float(COMMON_CAR_DIMS[1])
    s0 = lane.arc_at_frame(start_frame)
    lat = float(lateral)
    if mode == "oncoming":
        lat = lat - 3.5 if lateral == 0.0 else lat   # 左侧对向车道
        direction_along_ego = False
    if mode == "roadside":
        lat = lat + 2.6 if lateral == 0.0 else lat   # 靠右路肩
        speed = 0.0
    # 前进/对向：没给速度时自动给一个（否则物体静止、看不到运动轨迹）。
    # 同向默认取 0.8×自车车速：这样在自车视角里会"逐渐靠近"，轨迹看得出来；
    # 若手动填成和自车同速，则相对静止（物理上正确）。
    if mode in ("ahead", "oncoming") and float(speed) <= 0.0:
        speed = max(2.0, lane.mean_speed(fps) * 0.8) if mode == "ahead" else 8.0
    sign = 1.0 if direction_along_ego else -1.0
    frames = list(range(int(start_frame), int(start_frame) + int(num_frames)))
    centers: Dict[int, np.ndarray] = {}
    yaws: Dict[int, float] = {}
    gys: Dict[int, float] = {}
    n_fallback = 0
    n_sparse = 0
    for i, f in enumerate(frames):
        s = s0 + float(distance) + sign * float(speed) * (i / max(1e-6, fps))
        p, t = lane.point_tangent(s)
        if p is None:
            continue
        yaw = math.atan2(t[0], t[1])
        if not direction_along_ego:
            yaw += math.pi
        if flip:
            yaw += math.pi
        right = np.array([math.cos(yaw), 0.0, -math.sin(yaw)])
        c = np.array([p[0], 0.0, p[2]], dtype=np.float64) + right * lat
        # 先用"固定 6m 邻域"探一下稠密程度：这一级失败说明轨迹已经离开静态点密集区
        # （4DGS 静态点只覆盖相机看过的地方），用于给用户提示"这一段背景会偏空"。
        gy = _ground_height_near(pts, c, 6.0, up_sign, max_tries=1) if pts is not None else None
        if gy is None:
            gy = _ground_height_near(pts, c, 6.0, up_sign) if pts is not None else None
            if gy is None:
                # 连放大到 96m 都没有点 → 真·场景外：用自车车体**底面**高度当路面
                # （而不是车体中心！车体中心比路面高半个车高，会让物体浮空）
                gy = float(p[1]) - up_sign * (float(ego_h) / 2.0)
                n_fallback += 1
            else:
                n_sparse += 1
        if ref_gy is not None:
            gy = max(ref_gy - 1.2, min(ref_gy + 1.2, float(gy)))
        centers[int(f)] = c
        yaws[int(f)] = float(yaw)
        gys[int(f)] = float(gy)
    # 逐帧地面高度做滑动平均，避免"路面估计在相邻帧跳 ±5cm"导致车体抖动
    if gys:
        keys = sorted(gys)
        sm = _smooth_series([gys[k] for k in keys], window=5)
        for k, v in zip(keys, sm):
            gys[k] = float(v)
    poses: Dict[int, np.ndarray] = {}
    for f in sorted(centers):
        c = centers[f].copy()
        c[1] = float(gys[f]) + up_sign * (float(dims[1]) / 2.0)
        poses[int(f)] = _yaw_pose(c.astype(np.float32), yaws[f])
    return {"ok": bool(poses), "poses": poses, "lane_s0": s0, "lateral": lat,
            "speed": speed, "direction_along_ego": direction_along_ego,
            "mode": mode, "distance": float(distance), "up_sign": up_sign,
            "ground_ref": ref_gy, "ground_fallback_frames": int(n_fallback),
            "ground_sparse_frames": int(n_sparse),
            "ground_sparse_total": int(n_fallback + n_sparse),
            "ground_y_min": (min(gys.values()) if gys else None),
            "ground_y_max": (max(gys.values()) if gys else None)}


def find_conflicts(tm, poses: Dict[int, np.ndarray], dims, *, margin: float = 0.15,
                   ignore_id: Optional[int] = None) -> List[Dict[str, Any]]:
    out = []
    for f, P in poses.items():
        cn = _obb_corners_2d(P, dims)
        for o in tm.get_frame_objects(int(f)):
            tid = int(o["track_id"])
            if ignore_id is not None and tid == int(ignore_id):
                continue
            do = o.get("dimensions") or [2.0, 1.6, 4.5]
            if _obb_overlap_2d(cn, _obb_corners_2d(np.asarray(o["pose_world"], dtype=np.float64), do), margin):
                out.append({"frame": int(f), "track_id": tid,
                            "type": o.get("type"), "dimensions": do})
    return out


def plan_collision_free(tm, dims, ignore_id: Optional[int] = None, **kw) -> Dict[str, Any]:
    """尝试若干候选（横向挪开 / 前后挪开），返回第一个无冲突的轨迹。

    `ignore_id`：重排某个已存在物体时忽略它自己，否则会一直和它的旧轨迹冲突。
    """
    base_lat = float(kw.get("lateral", 0.0))
    base_dist = float(kw.get("distance", 12.0))
    attempts = []
    for dlat in (0.0, 1.2, -1.2, 2.4, -2.4):
        for ddist in (0.0, 4.0, 8.0, -4.0, 16.0):
            attempts.append((base_lat + dlat, base_dist + ddist))
    tried = []
    for lat, dist in attempts:
        kw2 = dict(kw)
        kw2["lateral"] = lat
        kw2["distance"] = dist
        plan = plan_trajectory(tm, dims, **kw2)
        if not plan.get("ok"):
            continue
        conf = find_conflicts(tm, plan["poses"], dims, ignore_id=ignore_id)
        tried.append({"lateral": lat, "distance": dist, "n_conflicts": len(conf)})
        if not conf:
            plan["conflicts"] = []
            plan["tried"] = tried
            return plan
    return {"ok": False, "reason": "找不到无冲突的位置", "tried": tried}
