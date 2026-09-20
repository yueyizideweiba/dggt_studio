"""DGGT 扩散渲染精修（Difix）接入。

论文里 "diffusion-based rendering refinement" 就是这一步：把 4DGS 渲染出来的帧再喂给
一个 1-step 扩散模型（Difix，基于 sd-turbo）去涂抹 artifacts，显著提升画质与 novel-view 稳定性。

本模块解决三件工程问题：
1. **单例缓存**：原始 `third_party/difix/infer.process_images_with_difix()` 每调用一次就
   `Difix(...)` 重新加载 5GB 权重 + sd-turbo 子模型 —— 逐帧用会慢到不可用。这里只加载一次。
2. **依赖自检**：Difix 需要 `stabilityai/sd-turbo` 的 tokenizer/text_encoder/vae/unet
   （HF 下载）。缺依赖时给出**明确报错与修复命令**，而不是崩在半路。
3. **A/B 指标**：提供 `refine_image` / `refine_dir` 与 CLI，可直接对"留出视角渲染"做
   精修前后 PSNR/SSIM/LPIPS 对比（配合 novelview_trust 的留出相机评测）。

用法：
    python diffusion_refine.py --status
    python diffusion_refine.py --dir <场景目录> --pattern "view_*.png" --out <输出目录>
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DIFIX_DEFAULT_CKPTS = ["pretrained/model_difix.pkl", "pretrained/diffusion_model.pth"]

_MODEL = None          # 单例


def _ckpt_path(explicit: Optional[str] = None) -> Optional[str]:
    for p in ([explicit] if explicit else []) + [str(ROOT / c) for c in DIFIX_DEFAULT_CKPTS]:
        if p and os.path.exists(p):
            return p
    return None


def sd_turbo_available() -> Dict[str, Any]:
    """检查 sd-turbo **权重**是否真的在本地（Difix 依赖它；只看 config 会误判）。

    会在所有候选 cache root 里找（`$HF_HOME` / `~/.cache/huggingface` / `/root/autodl-tmp/hf_home`），
    逐个快照按"权重体积"打分，取最完整的那个；fp32 与 fp16 命名都认。
    """
    fix = ("export HF_ENDPOINT=https://hf-mirror.com && python -c \"from huggingface_hub import "
           "snapshot_download as d; print(d('stabilityai/sd-turbo'))\"   "
           "# 或在有网的机器上下载后把 models--stabilityai--sd-turbo 目录拷到 $HF_HOME/hub/")
    try:
        from huggingface_hub import snapshot_download
    except Exception as e:  # noqa: BLE001
        return {"available": False, "reason": f"huggingface_hub 不可用: {e}", "fix": fix}

    roots = [os.environ.get("HF_HOME"), os.path.expanduser("~/.cache/huggingface"),
             "/root/autodl-tmp/hf_home"]
    cands: List[str] = []
    try:
        cands.append(snapshot_download("stabilityai/sd-turbo", local_files_only=True))
    except Exception:  # noqa: BLE001
        pass
    for r in roots:
        if not r:
            continue
        base = os.path.join(r, "hub", "models--stabilityai--sd-turbo", "snapshots")
        if os.path.isdir(base):
            cands += [os.path.join(base, s) for s in sorted(os.listdir(base))]

    # 阈值按"接近完整体积"来定（下载中的半成品必须判为缺失，否则会误报就绪）
    need = {
        "unet": [("unet/diffusion_pytorch_model.safetensors", 3.0e9),        # fp32 ≈3.46GB
                 ("unet/diffusion_pytorch_model.fp16.safetensors", 1.5e9)],  # fp16 ≈1.73GB
        "vae": [("vae/diffusion_pytorch_model.safetensors", 3.0e8),          # fp32 ≈334MB
                ("vae/diffusion_pytorch_model.fp16.safetensors", 1.5e8)],    # fp16 ≈167MB
        "text_encoder": [("text_encoder/model.safetensors", 1.2e9),          # fp32 ≈1.36GB
                         ("text_encoder/model.fp16.safetensors", 6.0e8)],    # fp16 ≈681MB
    }

    def score(path: str) -> Dict[str, Any]:
        sizes, variants, missing = {}, {}, []
        for comp, cl in need.items():
            hit, sz = None, 0
            for rel, minsize in cl:
                fp = os.path.join(path, rel)
                if os.path.exists(fp) and os.path.getsize(fp) >= minsize:
                    hit, sz = rel, os.path.getsize(fp)
                    variants[comp] = "fp16" if rel.endswith("fp16.safetensors") else "fp32"
                    break
            sizes[comp] = sz
            if hit is None:
                missing.append(" | ".join(r for r, _ in cl))
        only_fp16 = bool(variants) and all(v == "fp16" for v in variants.values())
        return {"path": path, "missing": missing, "variants": variants,
                "prefer_variant": ("fp16" if only_fp16 else None),
                "weight_bytes": int(sum(sizes.values()))}

    # 也支持"普通本地目录"（例如从 ModelScope 直接下到 <HF_HOME>/sd-turbo）
    for d in (os.environ.get("DIFIX_SD_TURBO_DIR"),
              os.path.join(os.environ.get("HF_HOME", "") or "", "sd-turbo"),
              "/root/autodl-tmp/hf_home/sd-turbo"):
        if d and os.path.isdir(os.path.join(d, "unet")):
            cands.append(os.path.abspath(d))
    if not cands:
        return {"available": False, "missing": ["(整个模型)"],
                "reason": f"没有找到任何 sd-turbo 快照（查过 {[r for r in roots if r]}）",
                "fix": fix, "searched_roots": [r for r in roots if r]}
    best = max((score(c) for c in cands), key=lambda d: d["weight_bytes"])
    best["available"] = not best["missing"]
    # 深度校验：文件大小够了也可能是"坏文件"（例如 curl -C - 在 git-LFS 指针文件后追加真实数据，
    # 结果 size 正确但开头仍是 "version https://..."）。这里真的去读一次 safetensors 头。
    if best["available"]:
        bad = []
        for comp, cl in need.items():
            for rel, _ in cl:
                fp = os.path.join(best["path"], rel)
                if os.path.exists(fp) and os.path.getsize(fp) >= 1000:
                    try:
                        from safetensors import safe_open
                        with safe_open(fp, framework="pt") as f:
                            _ = list(f.keys())[:1]
                        break
                    except Exception as e:  # noqa: BLE001
                        bad.append(f"{rel}: {type(e).__name__} {e}")
        if bad:
            best["available"] = False
            best["missing"] = best["missing"] + [f"(文件损坏) {b}" for b in bad]
            best["weights_verified"] = False
        else:
            best["weights_verified"] = True
    best["all_candidates"] = [{"path": c, "weight_bytes": score(c)["weight_bytes"]} for c in cands]
    best["fix"] = fix
    return best


def status() -> Dict[str, Any]:
    ck = _ckpt_path()
    st = sd_turbo_available()
    return {"difix_ckpt": ck, "difix_ckpt_ok": bool(ck),
            "difix_ckpt_candidates": [str(ROOT / c) for c in DIFIX_DEFAULT_CKPTS],
            "sd_turbo": st, "ready": bool(ck) and bool(st.get("available")),
            "note": "Difix 需要 sd-turbo 子模型；缺失时请按 sd_turbo.fix 下载"}


def _load_model(ckpt: Optional[str] = None):
    """加载并缓存 Difix（整个进程只加载一次）。"""
    global _MODEL
    if _MODEL is not None:
        return _MODEL
    path = _ckpt_path(ckpt)
    if not path:
        raise FileNotFoundError("找不到 Difix 权重（pretrained/model_difix.pkl 或 diffusion_model.pth）")
    st = sd_turbo_available()
    if not st.get("available"):
        raise RuntimeError(f"Difix 依赖 stabilityai/sd-turbo，本地缺失：{st.get('missing')}\n修复：{st.get('fix')}")
    sys.path.insert(0, str(ROOT))
    # --- 让 Difix 能直接用"普通本地目录"里的 sd-turbo（把 repo id 换成目录）---
    # Difix 内部（third_party/difix/src/model.py）会直接用 repo id 加载：
    #   AutoTokenizer/CLIPTextModel(subfolder=tokenizer/text_encoder)、
    #   DDPMScheduler(subfolder=scheduler)、AutoencoderKL(subfolder=vae)、
    #   UNet2DConditionModel(subfolder=unet)
    # 没网时这些调用会抛 "Cannot send a request, as the client has been closed"，
    # 所以逐个把 repo id 换成本地目录（按类精确重定向，避免扫描模块时踩到懒加载的坑）。
    local_dir = str(st.get("path") or "")
    if local_dir and "/snapshots/" not in local_dir and os.path.isdir(os.path.join(local_dir, "unet")):
        try:
            import diffusers
            import transformers
            targets = []
            for nm in ("AutoTokenizer", "CLIPTokenizer", "CLIPTextModel", "PreTrainedTokenizerFast"):
                c = getattr(transformers, nm, None)
                if isinstance(c, type):
                    targets.append(c)
            for nm in ("DDPMScheduler", "DDIMScheduler", "LCMScheduler", "EulerDiscreteScheduler",
                       "PNDMScheduler", "AutoencoderKL", "UNet2DConditionModel"):
                c = getattr(diffusers, nm, None)
                if isinstance(c, type):
                    targets.append(c)
            patched = 0
            for cls in targets:
                try:
                    orig = cls.from_pretrained.__func__
                except AttributeError:
                    continue

                def _wrap(orig=orig, local_dir=local_dir):
                    def _f(cls_, *a, **kw):                     # noqa: N807
                        if a and isinstance(a[0], str) and a[0] == "stabilityai/sd-turbo":
                            a = (local_dir,) + tuple(a[1:])
                        return orig(cls_, *a, **kw)
                    return classmethod(_f)
                try:
                    cls.from_pretrained = _wrap()
                    patched += 1
                except Exception:  # noqa: BLE001
                    pass
            print(f"[difix] sd-turbo 用本地目录 {local_dir}（重定向 {patched} 个类）")
        except Exception as e:  # noqa: BLE001
            print(f"[difix] 本地目录注入失败（回退 HF 缓存）：{type(e).__name__}: {e}")
    # --- 只有 fp16 权重时，diffusers 需要显式 variant/torch_dtype ---
    unet_fp32 = os.path.exists(os.path.join(local_dir, "unet", "diffusion_pytorch_model.safetensors")) \
        if local_dir else True
    if st.get("prefer_variant") == "fp16" or not unet_fp32:
        try:
            import torch
            from diffusers import UNet2DConditionModel, AutoencoderKL
            from transformers import CLIPTextModel
            for cls in (UNet2DConditionModel, AutoencoderKL, CLIPTextModel):
                orig = cls.from_pretrained.__func__

                def _wrap(orig=orig):
                    def _f(cls_, *a, **kw):
                        # 用 fp16 的**文件名**（本地只有 fp16 变体），但权重升到 fp32：
                        # Difix 会在 VAE 上挂 fp32 的新卷积 + 从 ckpt 载入 fp32 权重，
                        # 混 dtype 会在推理时报 "Input type (float) and bias type (c10::Half)"。
                        kw.setdefault("variant", "fp16")
                        kw.setdefault("torch_dtype", torch.float32)
                        return orig(cls_, *a, **kw)
                    return classmethod(_f)
                cls.from_pretrained = _wrap()
            print("[difix] 使用 fp16 权重文件（variant=fp16, torch_dtype=float32 避免混 dtype）")
        except Exception as e:  # noqa: BLE001
            print(f"[difix] fp16 注入失败：{e}")
    # --- 版本适配：本机 diffusers 0.30.3 的 AutoencoderKL 没继承 PeftAdapterMixin，
    #     而 Difix 要用 vae.add_adapter(...) 挂 LoRA（官方在更新版 diffusers 里是有的）。
    #     这里把 PeftAdapterMixin 的方法补到 AutoencoderKL 上，避免动环境。
    try:
        from diffusers import AutoencoderKL
        from diffusers.loaders import PeftAdapterMixin
        if not hasattr(AutoencoderKL, "add_adapter"):
            # 连私有的类属性也要补：peft 的 add_adapter() 会读/写
            # self._hf_peft_config_loaded（类属性，不是实例属性）
            for name in dir(PeftAdapterMixin):
                if name.startswith("__") and name.endswith("__"):
                    continue
                if not hasattr(AutoencoderKL, name):
                    try:
                        setattr(AutoencoderKL, name, getattr(PeftAdapterMixin, name))
                    except Exception:  # noqa: BLE001
                        pass
            print("[difix] 兼容补丁：给 AutoencoderKL 补上 peft 的 add_adapter 等方法")
    except Exception as e:  # noqa: BLE001
        print(f"[difix] VAE peft 兼容补丁失败：{e}")
    from third_party.difix.src.model import Difix
    model = Difix(pretrained_path=path, timestep=199, mv_unet=False)
    model.set_eval()
    _MODEL = model
    return model


def refine_tensor(img, ckpt: Optional[str] = None):
    """精修单张 (3,H,W) tensor[0,1]（只加载一次模型）。"""
    import torch
    import torchvision.transforms as transforms
    model = _load_model(ckpt)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, h, w = img.shape
    nh, nw = ((h + 7) // 8) * 8, ((w + 7) // 8) * 8
    if (nh, nw) != (h, w):
        pil = transforms.ToPILImage()(img.clamp(0, 1).cpu())
        from PIL import Image as _I
        img = transforms.ToTensor()(pil.resize((nw, nh), _I.Resampling.LANCZOS))
    img = img.to(device)
    model.sched.set_timesteps(1, device=device)
    model.sched.timesteps = torch.tensor([model.timesteps.item()], device=device)
    out = model.sample(transforms.ToPILImage()(img.cpu()), width=nw, height=nh,
                       prompt="remove degradation")
    out = transforms.ToTensor()(out)
    if (nh, nw) != (h, w):
        out = torch.nn.functional.interpolate(out.unsqueeze(0), size=(h, w),
                                             mode="bilinear", align_corners=False).squeeze(0)
    return out


def refine_image_rgb(img_rgb: np.ndarray, ckpt: Optional[str] = None) -> np.ndarray:
    """精修 (H,W,3) uint8/float[0,1] 图像，返回 float[0,1]。"""
    import torch
    a = np.asarray(img_rgb)
    a = a.astype(np.float32) / 255.0 if a.dtype == np.uint8 else np.clip(a.astype(np.float32), 0, 1)
    t = torch.from_numpy(a.transpose(2, 0, 1))
    out = refine_tensor(t, ckpt=ckpt)
    return np.clip(out.detach().cpu().numpy().transpose(1, 2, 0), 0, 1)


def refine_dir(scene_dir: str, out_dir: str, pattern: str = "view_*.png",
               ckpt: Optional[str] = None, verbose: bool = True) -> Dict[str, Any]:
    """把一个目录里的渲染帧批量精修（保持同名）。"""
    import cv2
    os.makedirs(out_dir, exist_ok=True)
    files = sorted(glob.glob(os.path.join(scene_dir, pattern)))
    done = []
    for f in files:
        img = cv2.imread(f)
        if img is None:
            continue
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        out = refine_image_rgb(rgb, ckpt=ckpt)
        cv2.imwrite(os.path.join(out_dir, os.path.basename(f)),
                    cv2.cvtColor((out * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
        done.append(os.path.basename(f))
        if verbose:
            print(f"[difix] {os.path.basename(f)} 已精修")
    return {"num": len(done), "files": done, "out_dir": out_dir}


def main():
    ap = argparse.ArgumentParser(description="DGGT 扩散渲染精修（Difix）")
    ap.add_argument("--status", action="store_true", help="打印依赖自检结果")
    ap.add_argument("--dir", default="", help="待精修的渲染帧目录")
    ap.add_argument("--pattern", default="view_*.png")
    ap.add_argument("--out", default="", help="输出目录")
    ap.add_argument("--ckpt", default=None)
    args = ap.parse_args()
    if args.status or not args.dir:
        print(json.dumps(status(), ensure_ascii=False, indent=2))
        return 0
    out = args.out or (args.dir.rstrip("/") + "_difix")
    rep = refine_dir(args.dir, out, args.pattern, ckpt=args.ckpt)
    print(json.dumps(rep, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
