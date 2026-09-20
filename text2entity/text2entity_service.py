# Copyright (c) DGGT Studio.
"""
"文本 → 实体" 微服务：LLaDA-Image-Turbo (文生图) + Qwen2.5-VL-3B (视觉推理)。

单独进程/单独 conda 环境（llada-image, torch 2.8 + diffusers），因为与 dggt 后端
(torch 2.4.1) 依赖冲突；24G 单卡上两个大模型不能同时驻留，所以：
  * LLaDA 与 VLM 互斥（加载一个会先卸载另一个）；
  * 提供 /unload 主动释放显存（dggt 后端在调用 SAM3D 之前会调它）。

端口默认 8002。端点：
  GET  /health                 → 服务/模型/显存状态
  POST /generate               → {prompt,...} → base64 PNG（参考图）
  POST /vlm                    → {prompt, images:[base64]} → {text}
  POST /unload                 → 卸载模型（进程退出，由 run_service.sh 监督重启）
"""
import base64
import io
import os
import sys
import gc
import time
import threading
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional

HERE = Path(__file__).resolve().parent
CODE_DIR = os.environ.get("LLADA_CODE_DIR", "/root/autodl-tmp/code/LLaDA-Image")
LLADA_MODEL = os.environ.get("LLADA_MODEL_DIR", "/autodl-fs/data/models/LLaDA-Image-Turbo-FP8")
VLM_MODEL = os.environ.get("VLM_MODEL_DIR", "/autodl-fs/data/models/Qwen2.5-VL-3B-Instruct")
PORT = int(os.environ.get("TEXT2ENTITY_PORT", "8002"))

app = FastAPI(title="DGGT Text2Entity Service", version="1.0.0")

_lock = threading.RLock()   # 可重入：get_pipe/get_vlm 持锁时会调用 unload_all()（也加锁）
_pipe = None
_vlm = None          # (model, processor)
_error = None


def _free():
    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        except Exception:  # noqa: BLE001
            pass
    # glibc 释放的大块内存不一定立刻还给内核，RSS 会虚高；显式 trim，
    # 否则在受限 cgroup（本机约 43GB）里很容易被 OOM Killer 干掉。
    try:
        import ctypes
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:  # noqa: BLE001
        pass


def _drop_caches():
    """丢弃可回收的 page cache，给 LLaDA 腾出内存余量。

    LLaDA-Image-Turbo-FP8 加载后常驻 RSS 约 28GB，而本机 cgroup 上限约 43GB，
    叠加其它常驻进程与 page cache 后会顶到上限被 SIGKILL。page cache 是干净的
    文件页，丢掉不会丢数据（只是后续读盘慢一点），只在本进程加载大模型前做一次。
    """
    try:
        if os.geteuid() != 0:
            return
        with open("/proc/sys/vm/drop_caches", "w") as f:
            f.write("3\n")
    except Exception:  # noqa: BLE001
        pass


def _ensure_dir(p: str):
    if not os.path.isdir(p):
        raise RuntimeError(f"模型目录不存在: {p}（请先下载权重）")


def unload_all():
    global _pipe, _vlm
    with _lock:
        _pipe = None
        _vlm = None
    _free()


# ---------------- LLaDA-Image ----------------
def _disable_scaled_mm_if_unsupported():
    """sm<8.9（如 3090）不支持 torch._scaled_mm 的 FP8 GEMM。

    LLaDA 的 FP8 text_encoder 用 `hasattr(torch,'_scaled_mm')` 选择快路径，
    这里在 Ampere 上删掉该属性，强制走"逐专家反量化"的兜底实现（权重仍是 FP8，显存不变）。
    """
    try:
        if torch.cuda.is_available() and hasattr(torch, "_scaled_mm"):
            if torch.cuda.get_device_capability(0) < (8, 9):
                del torch._scaled_mm
                print("[llada] sm<8.9: 禁用 torch._scaled_mm，改用 FP8 反量化兜底", flush=True)
    except Exception:  # noqa: BLE001
        pass


def _patch_fp8_moe_for_ampere(pipe):
    """sm<8.9 上 torch 对 FP8 tensor 不支持 index_select，且 _scaled_mm 不可用。

    打补丁：先把整段激活反量化成模型 dtype，再按专家 index_select + F.linear，
    语义与官方"逐专家反量化"兜底一致，避免 FP8 index_select 报错。
    """
    import torch.nn.functional as F

    def _fp8_ref(self, hidden_states, routing_weights, selected_experts):
        xq_all, x_scale_all = self._quantize_token_fp8(hidden_states)
        x_all = (xq_all.float() * x_scale_all).to(hidden_states.dtype)
        selected_experts = selected_experts.reshape(-1, selected_experts.shape[-1])
        routing_weights = routing_weights.reshape(-1, routing_weights.shape[-1])
        token_ids = torch.arange(hidden_states.shape[0], device=hidden_states.device)[:, None].expand_as(selected_experts)
        flat_tokens = token_ids.reshape(-1)
        flat_experts = selected_experts.reshape(-1)
        flat_weights = routing_weights.reshape(-1)
        out = torch.zeros_like(hidden_states)
        for e in torch.unique(flat_experts, sorted=False):
            eid = int(e.item())
            mask = flat_experts == e
            ids = flat_tokens[mask]
            x = x_all.index_select(0, ids)
            gw = self._dequant_expert(self.gate_proj, self.gate_proj_scale, eid, hidden_states.dtype)
            uw = self._dequant_expert(self.up_proj, self.up_proj_scale, eid, hidden_states.dtype)
            gate = F.linear(x, gw)
            up = F.linear(x, uw)
            del gw, uw
            mid = F.silu(gate) * up
            dw = self._dequant_expert(self.down_proj, self.down_proj_scale, eid, hidden_states.dtype)
            y = F.linear(mid, dw)
            del dw, mid
            y = y * flat_weights[mask, None].to(y.dtype)
            out.index_add_(0, ids, y)
        return out

    patched = 0
    try:
        for mod in pipe.text_encoder.modules():
            cls = type(mod)
            if hasattr(cls, "_fp8_reference_forward") and cls.__dict__.get("_fp8_reference_forward") is not _fp8_ref:
                cls._fp8_reference_forward = _fp8_ref
                patched += 1
    except Exception as e:  # noqa: BLE001
        print(f"[llada] FP8 MoE 补丁失败: {e}", flush=True)
    print(f"[llada] Ampere FP8 补丁应用于 {patched} 个专家模块", flush=True)


def get_pipe():
    global _pipe, _error
    with _lock:
        if _pipe is not None:
            return _pipe
        if _vlm is not None:
            unload_all()
        _ensure_dir(LLADA_MODEL)
        try:
            if CODE_DIR not in sys.path:
                sys.path.insert(0, CODE_DIR)
            from src import LLaDAImagePipeline  # type: ignore
            _disable_scaled_mm_if_unsupported()
            _drop_caches()   # 先腾出 page cache，给 28GB 常驻权重留余量
            t0 = time.time()
            # 24G 单卡放不下整模型（约 27.6GB），所以先整体加载到 CPU，
            # 生成时再分段搬到 GPU（见 _staged_generate）。FP8 权重必须保持原 dtype。
            _pipe = LLaDAImagePipeline.from_pretrained(LLADA_MODEL, torch_dtype=None, device=None)
            _patch_fp8_moe_for_ampere(_pipe)
            print(f"[llada] loaded on CPU in {time.time()-t0:.1f}s", flush=True)
        except Exception as e:  # noqa: BLE001
            _error = f"LLaDA 加载失败: {e}"
            raise RuntimeError(_error) from e
        return _pipe


def _mods(pipe, names):
    return [getattr(pipe, n) for n in names if getattr(pipe, n, None) is not None]


def _staged_generate(pipe, prompt, negative_prompt, h, w, steps, guidance, seed):
    """分段上 GPU：text_encoder(17GB) 编码完就下 CPU，再上 transformer/sigvq/vae(约10GB)。

    24G 卡上这样峰值 ~18GB，能跑；否则整模型 27.6GB 直接 OOM。
    """
    dev = "cuda"
    enc_mods = _mods(pipe, ("text_encoder", "queryformer", "text_projection"))
    for m in enc_mods:
        m.to(dev)
    try:
        with torch.no_grad(), torch.inference_mode():
            embeds, mask, neg_e, neg_m = pipe.encode_prompt(
                prompt, negative_prompt or None, False, 1,
                None, None, None, None, 2048, dev,
            )
    finally:
        for m in enc_mods:
            m.to("cpu")
        _free()
    for m in _mods(pipe, ("transformer", "sigvq", "vae")):
        m.to(dev)
    try:
        with torch.no_grad(), torch.inference_mode():
            out = pipe(
                prompt=None,                   # 必须二选一：给了 prompt_embeds 就不能再给 prompt
                prompt_embeds=embeds,
                prompt_attention_mask=mask,
                negative_prompt_embeds=neg_e,
                negative_prompt_attention_mask=neg_m,
                generation_mode="text",
                height=h, width=w,
                num_inference_steps=int(steps),
                guidance_scale=float(guidance),
                generator=torch.Generator("cuda").manual_seed(int(seed)),
            )
    finally:
        for m in _mods(pipe, ("transformer", "sigvq", "vae")):
            m.to("cpu")
        _free()
    return out.images[0]


# ---------------- Qwen2.5-VL ----------------
def get_vlm():
    global _vlm, _error
    with _lock:
        if _vlm is not None:
            return _vlm
        if _pipe is not None:
            unload_all()
        _ensure_dir(VLM_MODEL)
        try:
            from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
            t0 = time.time()
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                VLM_MODEL, torch_dtype=torch.bfloat16, device_map="cuda")
            processor = AutoProcessor.from_pretrained(VLM_MODEL)
            _vlm = (model, processor)
            print(f"[vlm] loaded in {time.time()-t0:.1f}s", flush=True)
        except Exception as e:  # noqa: BLE001
            _error = f"VLM 加载失败: {e}"
            raise RuntimeError(_error) from e
        return _vlm


class GenRequest(BaseModel):
    prompt: str
    negative_prompt: str = ""
    width: int = 1024
    height: int = 1024
    steps: int = 4
    guidance_scale: float = 1.0
    seed: int = 0


class VlmRequest(BaseModel):
    prompt: str
    images: List[str] = []      # base64（可带 dataURL 前缀）
    max_new_tokens: int = 512


@app.get("/health")
def health():
    return {
        "status": "ok",
        "prompt2image": os.path.isdir(LLADA_MODEL),
        "vlm": os.path.isdir(VLM_MODEL),
        "llada_hot": _pipe is not None,
        "vlm_hot": _vlm is not None,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "error": _error,
        "model_dir": LLADA_MODEL,
        "vlm_dir": VLM_MODEL,
    }


@app.post("/generate")
def generate(req: GenRequest):
    try:
        pipe = get_pipe()
        w = int(req.width) - int(req.width) % 16
        h = int(req.height) - int(req.height) % 16
        img = _staged_generate(pipe, req.prompt, req.negative_prompt or "",
                               h, w, int(req.steps), float(req.guidance_scale), int(req.seed))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return {"success": True, "image": base64.b64encode(buf.getvalue()).decode("ascii"),
                "width": img.width, "height": img.height}
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"生成失败: {e}")


def _decode_image(b64: str) -> Image.Image:
    if "," in b64[:64]:
        b64 = b64.split(",", 1)[1]
    return Image.open(io.BytesIO(base64.b64decode(b64 + "=" * (-len(b64) % 4)))).convert("RGB")


class LlmRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 768


@app.post("/llm")
def llm(req: LlmRequest):
    """纯文本 LLM 推理（复用加载的 Qwen；用于把自然语言解析成轨迹编辑动作）。"""
    try:
        model, processor = get_vlm()
        messages = [{"role": "user", "content": [{"type": "text", "text": req.prompt}]}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        # VL processor 在"纯文本"时偶尔会造出空的 pixel_values 导致 generate 报错，
        # 失败就退回 tokenizer 直出 input_ids。
        try:
            inputs = processor(text=[text], return_tensors="pt").to("cuda")
        except Exception as pe:  # noqa: BLE001
            print(f"[llm] processor 文本编码失败({pe})，改用 tokenizer", flush=True)
            inputs = processor.tokenizer([text], return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=int(req.max_new_tokens))
        trimmed = out[0][inputs["input_ids"].shape[1]:]
        return {"success": True, "text": processor.decode(trimmed, skip_special_tokens=True)}
    except Exception as e:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"LLM 推理失败: {e}")


@app.post("/vlm")
def vlm(req: VlmRequest):
    try:
        model, processor = get_vlm()
        content = [{"type": "image", "image": _decode_image(b)} for b in req.images]
        content.append({"type": "text", "text": req.prompt})
        messages = [{"role": "user", "content": content}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        # qwen_vl_utils 优先；没有就直接传 PIL
        try:
            from qwen_vl_utils import process_vision_info
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(text=[text], images=image_inputs, videos=video_inputs,
                               padding=True, return_tensors="pt").to("cuda")
        except Exception:  # noqa: BLE001
            imgs = [_decode_image(b) for b in req.images]
            inputs = processor(text=[text], images=imgs, padding=True, return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=int(req.max_new_tokens))
        trimmed = out[0][inputs["input_ids"].shape[1]:]
        ans = processor.decode(trimmed, skip_special_tokens=True)
        return {"success": True, "text": ans}
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"VLM 推理失败: {e}")


@app.post("/unload")
def unload():
    unload_all()

    def _exit():
        time.sleep(1.5)
        os._exit(0)

    threading.Thread(target=_exit, daemon=True).start()
    return {"success": True, "restarting": True}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)
