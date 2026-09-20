#!/usr/bin/env bash
# =============================================================================
# 启动 SAM 3D Objects 推理微服务（端口 8001）
#
# 用法:
#   bash sam-3d-objects/run_sam3d_service.sh
#
# 可选环境变量:
#   SAM3D_ENV_DIR     SAM 3D conda 环境路径（默认 /root/autodl-tmp/conda_envs/sam3d-objects）
#   SAM3D_CONFIG_PATH 模型 pipeline 配置（默认 checkpoints/pipeline.yaml）
#   SAM3D_PORT        监听端口（默认 8001）
#   SAM3D_OUTPUT_DIR  重建结果输出目录
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_DIR="${SAM3D_ENV_DIR:-/root/autodl-tmp/conda_envs/sam3d-objects}"

if [ ! -x "${ENV_DIR}/bin/python" ]; then
  echo "错误: 未找到 SAM 3D 环境 ${ENV_DIR}，请先运行 setup_sam3d_env.sh" >&2
  exit 1
fi

cd "${SCRIPT_DIR}"
export CONDA_PREFIX="${ENV_DIR}"
export CUDA_HOME="${ENV_DIR}"
export LD_LIBRARY_PATH="${ENV_DIR}/targets/x86_64-linux/lib:${ENV_DIR}/lib:${LD_LIBRARY_PATH:-}"

# torch.hub 需要通过 github.com 拉取 facebookresearch/dinov2（DINO 条件嵌入器）代码，
# 国内直连会被远端直接断开（RemoteDisconnected）。必须走 AutoDL 学术加速代理。
if [ -f /etc/network_turbo ]; then
  # shellcheck disable=SC1091
  source /etc/network_turbo || true
  # DINO 权重源直连很快，排除在代理外
  export no_proxy="localhost,127.0.0.1,modelscope.com,aliyuncs.com,tencentyun.com,wisemodel.cn,dl.fbaipublicfiles.com"
fi

# MoGe 深度模型权重从 HuggingFace 拉取时，新版 huggingface_hub 默认走 Xet 后端
# (cas-server.xethub.hf.co)，会返回 401 Unauthorized。禁用 Xet 并优先用本地缓存。
# 权重已预下载到 /root/.cache/huggingface/hub/models--Ruicheng--moge-vitl。
export HF_HUB_DISABLE_XET=1
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

# 减少显存碎片：24G 卡上"场景(~1.5G) + SAM3D(~23G)"很紧，expandable_segments
# 可显著降低峰值占用、避免显存碎片导致的 OOM。
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# 监督循环：/unload 会让进程退出以彻底归还 GPU 显存（spconv/warp 内存池无法被
# empty_cache 释放），这里自动重启；模型懒加载，重启后空闲不占显存。
while true; do
    "${ENV_DIR}/bin/python" sam3d_service.py || true
    sleep 1
done
