#!/usr/bin/env python3
"""把 LLaDA-Image-Turbo-FP8 的 transformer 修成 diffusers 0.39 能直接加载的 bf16。

做两件事：
  1) 反量化：FP8(e4m3) `weight` + 128x128 block 的 `weight_scale_inv` → bf16
     （`W = W_fp8.float() * repeat_interleave(scale_inv,128)`）；
     并删掉 config.json 里的 `quantization_config`（vLLM 格式，diffusers 不认识）。
  2) 拆分 fused 权重：仓库 `src` 的模型是 split 版，checkpoint 是 fused 版
     - `attention.to_qkv.weight` → `to_q/to_k/to_v`（dim=0 三等分）
     - `feed_forward.w13.weight`  → `w1/w3`（dim=0 二等分）
  最后重建 safetensors index。

用法: python fix_transformer_fp8.py [transformer_dir]
"""
import glob
import json
import os
import sys

import torch
from safetensors.torch import load_file, save_file

D = sys.argv[1] if len(sys.argv) > 1 else '/autodl-fs/data/models/LLaDA-Image-Turbo-FP8/transformer'
BS = 128


def expand(scale, N, K):
    return scale.repeat_interleave(BS, 0).repeat_interleave(BS, 1)[:N, :K]


def main():
    shards = sorted(glob.glob(f"{D}/diffusion_pytorch_model-*.safetensors"))
    weight_map, total = {}, 0
    for sh in shards:
        name = os.path.basename(sh)
        sd = load_file(sh)
        out = {}
        for k, v in sd.items():
            if k.endswith(".weight_scale_inv"):
                continue
            if v.dtype == torch.float8_e4m3fn:                       # 反量化
                sc = sd.get(k.replace(".weight", ".weight_scale_inv"))
                if sc is None:
                    raise RuntimeError(f"缺少 scale: {k}")
                N, K = v.shape
                v = (v.float() * expand(sc, N, K)).to(torch.bfloat16)
            if k.endswith("attention.to_qkv.weight"):                 # 拆 QKV
                q, kk, vv = torch.chunk(v, 3, dim=0)
                base = k[: -len("to_qkv.weight")]
                out[base + "to_q.weight"], out[base + "to_k.weight"], out[base + "to_v.weight"] = (
                    q.contiguous(), kk.contiguous(), vv.contiguous())
            elif k.endswith("feed_forward.w13.weight"):               # 拆 w1/w3
                w1, w3 = torch.chunk(v, 2, dim=0)
                base = k[: -len("w13.weight")]
                out[base + "w1.weight"], out[base + "w3.weight"] = w1.contiguous(), w3.contiguous()
            else:
                out[k] = v
        save_file(out, sh)
        for k, v in out.items():
            weight_map[k] = name
            total += v.numel() * v.element_size()
        print(f"fixed {name}: {len(out)} tensors", flush=True)
        del sd, out
    json.dump({"metadata": {"total_size": int(total)}, "weight_map": weight_map},
              open(f"{D}/diffusion_pytorch_model.safetensors.index.json", "w"), indent=2)
    cfg_path = f"{D}/config.json"
    cfg = json.load(open(cfg_path))
    cfg.pop("quantization_config", None)
    cfg.pop("dtype", None)
    json.dump(cfg, open(cfg_path, "w"), indent=2)
    print(f"FIX_DONE {len(weight_map)} tensors {total/1e9:.2f} GB")


if __name__ == "__main__":
    main()
