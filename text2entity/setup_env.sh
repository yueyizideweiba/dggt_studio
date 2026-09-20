#!/usr/bin/env bash
# 一次性安装 "文本 → 实体" 的运行环境与权重。
#
#   bash text2entity/setup_env.sh
#
# 设计（本机实测踩坑后定的）：
#   * /autodl-fs/data 是 200G 网络盘但 inode 很少 → 权重（少量大文件）放这里；
#   * PyPI 下载 torch 2.8 很慢（~0.7MB/s），所以**复用 sam3d-objects 里已有的 torch 2.5.1**，
#     用 venv --system-site-packages 叠加 diffusers0.39/transformers4.57（实测 LLaDA 可正常 import）；
#   * ModelScope 下载很快（10+MB/s），所以权重用 text2entity/download_models.py 直接拉。
set -x
exec 2>&1
BASE_PY="${BASE_PY:-/root/autodl-tmp/conda_envs/sam3d-objects/bin/python}"   # 提供 torch 2.5.1
VENV="${LLADA_ENV_DIR:-/root/autodl-tmp/llada-venv25}"
CODE="${LLADA_CODE_DIR:-/root/autodl-tmp/code/LLaDA-Image}"
ALI=https://mirrors.aliyun.com/pypi/simple/
export PIP_CONFIG_FILE=/dev/null PIP_NO_CACHE_DIR=1 TMPDIR=/root/autodl-tmp/tmp
mkdir -p "$TMPDIR" "$(dirname "$CODE")"

echo "=== [1/4] clone LLaDA-Image (走学术加速) ==="
source /etc/network_turbo || true
[ -d "$CODE" ] || git clone --depth 1 https://github.com/inclusionAI/LLaDA-Image.git "$CODE" || exit 11

echo "=== [2/4] venv (复用 torch 2.5.1) ==="
rm -rf "$VENV"
"$BASE_PY" -m venv --system-site-packages "$VENV" || exit 12

echo "=== [3/4] diffusers/transformers/... ==="
"$VENV/bin/python" -m pip install \
  "diffusers==0.39.0" "transformers==4.57.6" "peft>=0.17.0" accelerate safetensors timm \
  qwen-vl-utils fastapi uvicorn python-multipart -i "$ALI" || exit 13
"$VENV/bin/python" -c "import torch,diffusers,transformers;print('env ok',torch.__version__,diffusers.__version__,transformers.__version__)" || exit 14

echo "=== [4/5] 下载权重（ModelScope，可重入） ==="
DIR="$(dirname "$(readlink -f "$0")")"
"$VENV/bin/python" "$DIR/download_models.py" || exit 15

echo "=== [5/5] 修 transformer(FP8→bf16 + 拆 fused QKV/w13) + 给 pipeline 打补丁 ==="
"$VENV/bin/python" "$DIR/fix_transformer_fp8.py" || exit 16
"$VENV/bin/python" "$DIR/patch_llada_pipeline.py" "$CODE" || exit 17

echo "=== ALL DONE ==="
echo "启动： nohup bash text2entity/run_service.sh > /tmp/text2entity.log 2>&1 &"
