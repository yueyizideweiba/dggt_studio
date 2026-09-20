#!/usr/bin/env bash
# 启动 "文本 → 实体" 微服务（LLaDA-Image-Turbo + Qwen2.5-VL-3B，端口 8002）。
#
#   bash text2entity/run_service.sh            # 前台监督循环
#   nohup bash text2entity/run_service.sh > /tmp/text2entity.log 2>&1 &
#
# 环境变量：
#   LLADA_ENV_DIR   默认 /root/autodl-tmp/llada-venv25
#                   （复用 sam3d-objects 的 torch 2.5.1 + 安装 diffusers0.39/transformers4.57；
#                    本机新建的 conda env 需从 PyPI 下 torch 2.8，但下载很慢，故用此 venv）
#   LLADA_MODEL_DIR 默认 /autodl-fs/data/models/LLaDA-Image-Turbo-FP8
#   VLM_MODEL_DIR   默认 /autodl-fs/data/models/Qwen2.5-VL-3B-Instruct
#   LLADA_CODE_DIR  默认 /root/autodl-tmp/code/LLaDA-Image
#   TEXT2ENTITY_PORT 默认 8002
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_DIR="${LLADA_ENV_DIR:-/root/autodl-tmp/llada-venv25}"

if [ ! -x "${ENV_DIR}/bin/python" ]; then
  echo "错误: 未找到环境 ${ENV_DIR}，请先运行 text2entity/setup_env.sh" >&2
  exit 1
fi

# Qwen2.5-VL 的 flash-attn 可选；这里走默认 sdpa，避免额外依赖。
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HF_HUB_DISABLE_XET=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

cd "${SCRIPT_DIR}"
# /unload 会让进程退出以彻底归还显存，这里监督重启；模型懒加载，空闲不占显存。
while true; do
    "${ENV_DIR}/bin/python" text2entity_service.py || true
    sleep 1
done
