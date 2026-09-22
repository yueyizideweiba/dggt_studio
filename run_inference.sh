#!/bin/bash
# 用正确的 PATH 跑 DGGT 推理（gsplat 首次渲染要 JIT 编译 CUDA kernel，需要 ninja + nvcc）
set -euo pipefail
cd /autodl-fs/data/dggt-main
export PATH="/root/autodl-tmp/conda_envs/dggt/bin:/usr/local/cuda/bin:$PATH"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.6}"
# 24G 卡上同时跑 studio（会常驻 1~2G）时容易碎片化 OOM，开可扩展段更稳
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
exec /root/autodl-tmp/conda_envs/dggt/bin/python inference.py "$@"
