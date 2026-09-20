#!/usr/bin/env python3
"""给 LLaDA-Image 的 pipeline 打补丁：

`LLaDAImagePipeline.from_pretrained` 原本把同一个 `torch_dtype` 传给所有组件。
但 text_encoder(LLaDA2 MoE) 必须保持 `dtype=None` 才能保留 FP8（转 bf16 会到 ~34GB），
而 transformer/sigvq/vae/queryformer/text_projection 用 bf16 才放得下 24G 卡。

用法: python patch_llada_pipeline.py /root/autodl-tmp/code/LLaDA-Image
"""
import sys
from pathlib import Path

CODE = Path(sys.argv[1] if len(sys.argv) > 1 else "/root/autodl-tmp/code/LLaDA-Image")
P = CODE / "src/pipelines/pipeline_llada_image.py"
s = P.read_text(encoding="utf-8")

if "_cd = torch_dtype" in s:
    print("already patched")
    sys.exit(0)

old = '        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(model_path / "scheduler")\n'
new = (old +
       '        # DGGT patch: 非 text_encoder 组件默认 bf16；text_encoder 保持 dtype=None 以保留 FP8\n'
       '        _cd = torch_dtype if torch_dtype is not None else torch.bfloat16\n')
assert old in s, "pipeline 结构不匹配"
s = s.replace(old, new, 1)
s = s.replace('AutoencoderKLFlux2.from_pretrained(model_path / "vae", torch_dtype=torch_dtype)',
              'AutoencoderKLFlux2.from_pretrained(model_path / "vae", torch_dtype=_cd)')
s = s.replace('LLaDAImageQueryFormerModel.from_pretrained(model_path / "queryformer", torch_dtype=torch_dtype)',
              'LLaDAImageQueryFormerModel.from_pretrained(model_path / "queryformer", torch_dtype=_cd)')
s = s.replace('model_path / "text_projection", torch_dtype=torch_dtype',
              'model_path / "text_projection", torch_dtype=_cd')
s = s.replace('LLaDAImageSigVQModel.from_pretrained(model_path / "sigvq", torch_dtype=torch_dtype)',
              'LLaDAImageSigVQModel.from_pretrained(model_path / "sigvq", torch_dtype=_cd)')
s = s.replace('LLaDAImageTransformer2DModel.from_pretrained(model_path / "transformer", torch_dtype=torch_dtype)',
              'LLaDAImageTransformer2DModel.from_pretrained(model_path / "transformer", torch_dtype=_cd)')
P.write_text(s, encoding="utf-8")
print(f"patched {P}")
