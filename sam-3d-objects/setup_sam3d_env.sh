#!/usr/bin/env bash
# =============================================================================
# 构建 SAM 3D Objects 独立运行环境（与 dggt 后端环境隔离，避免 torch/gsplat 冲突）
#
# 已验证可在 RTX 3090 (24G) + AutoDL 上跑通完整重建。
#
# 本脚本固化了以下踩坑修复：
#   1. 不依赖 `conda activate`（脚本环境里可能不可用），直接用绝对路径 python -m pip。
#   2. 必须 source /etc/network_turbo：GitHub / HuggingFace 直连不通。
#   3. PyPI 主源换华为云（阿里云实测仅 ~0.6MB/s，华为云 ~10MB/s）。
#   4. 显式先装 torch/torchvision/torchaudio 2.5.1+cu121：官方 requirements.txt 未 pin torch，
#      否则 pip 会拿最新版（可能 CPU 版）并与 torchaudio/xformers 反复回溯。
#   5. pytorch3d 编译需 CPATH 指向 conda CUDA 头文件目录 targets/x86_64-linux/include，
#      否则报 "cuda_runtime_api.h: No such file or directory"。
#   6. flash_attn 用官方预编译轮子，省 20-30 分钟源码编译。
#   7. 预下载 DINO ViT-L 权重（torch.hub 运行时用）+ MoGe 权重（禁用 HF Xet 后端，
#      否则 cas-server.xethub.hf.co 返回 401）。
#
# 用法:
#   bash sam-3d-objects/setup_sam3d_env.sh
# 可覆盖: SAM3D_ENV_DIR / CONDA_BIN / SAM3D_TMP
# =============================================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_DIR="${SAM3D_ENV_DIR:-/root/autodl-tmp/conda_envs/sam3d-objects}"
CONDA_BIN="${CONDA_BIN:-/root/miniconda3/bin/conda}"
WORK_TMP="${SAM3D_TMP:-/root/autodl-tmp/sam3d_tmp}"
mkdir -p "${WORK_TMP}"

echo "==> 环境位置: ${ENV_DIR}"

# ---- 0. 学术加速 + 源 ----
if [ -f /etc/network_turbo ]; then
  # shellcheck disable=SC1091
  source /etc/network_turbo || true
  echo "==> 已开启学术加速代理 (GitHub / HuggingFace)"
fi
export no_proxy="localhost,127.0.0.1,modelscope.com,aliyuncs.com,tencentyun.com,wisemodel.cn,mirrors.aliyun.com,download.pytorch.org,download-r2.pytorch.org,repo.huaweicloud.com,pypi.tuna.tsinghua.edu.cn,dl.fbaipublicfiles.com,nvidia-kaolin.s3.us-east-2.amazonaws.com"

# ---- 1. 创建 conda 环境（仅首次）----
export CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS:-/root/autodl-tmp/conda_pkgs}"
if [ -x "${ENV_DIR}/bin/python" ]; then
  echo "==> 环境已存在，跳过创建: ${ENV_DIR}"
else
  echo "==> 从 environments/default.yml 创建环境（含 CUDA 12.1 工具链）"
  "${CONDA_BIN}" env create -f "${SCRIPT_DIR}/environments/default.yml" -p "${ENV_DIR}" --solver libmamba
fi

PY="${ENV_DIR}/bin/python"
# ---- 编译环境（关键：CPATH 指向 conda CUDA 头文件）----
export CONDA_PREFIX="${ENV_DIR}"
export CUDA_HOME="${ENV_DIR}"
export CPATH="${ENV_DIR}/targets/x86_64-linux/include:${CPATH:-}"
export CPLUS_INCLUDE_PATH="${ENV_DIR}/targets/x86_64-linux/include:${CPLUS_INCLUDE_PATH:-}"
export LD_LIBRARY_PATH="${ENV_DIR}/targets/x86_64-linux/lib:${ENV_DIR}/lib:${LD_LIBRARY_PATH:-}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.6}"   # RTX 3090 = sm_86
export MAX_JOBS="${MAX_JOBS:-8}"
export FORCE_CUDA=1
export TMPDIR="${WORK_TMP}"
export PIP_NO_CACHE_DIR=1
export PIP_DISABLE_PIP_VERSION_CHECK=1
export PIP_INDEX_URL="https://repo.huaweicloud.com/repository/pypi/simple"
export PIP_EXTRA_INDEX_URL="https://download.pytorch.org/whl/cu121"
export PIP_FIND_LINKS="https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.5.1_cu121.html"

cd "${SCRIPT_DIR}" || exit 1

echo "==> [1/6] PyTorch 2.5.1 + CUDA 12.1（torch / torchvision / torchaudio）"
"${PY}" -m pip install \
  "torch==2.5.1+cu121" "torchvision==0.20.1+cu121" "torchaudio==2.5.1+cu121" \
  --index-url https://download.pytorch.org/whl/cu121 --extra-index-url "${PIP_INDEX_URL}"

echo "==> [2/6] sam3d_objects 核心依赖 (.[dev])"
"${PY}" -m pip install -e '.[dev]'

echo "==> [3/6] pytorch3d（源码编译，需 CPATH 指 CUDA 头文件）"
"${PY}" -m pip install --no-build-isolation \
  "pytorch3d @ git+https://github.com/facebookresearch/pytorch3d.git@75ebeeaea0908c5527e7b1e305fbc7681382db47"

echo "==> [4/6] flash_attn（官方预编译轮子，避免长编译）"
"${PY}" -m pip install --no-build-isolation \
  "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3%2Bcu12torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl" \
  || "${PY}" -m pip install flash_attn==2.8.3

echo "==> [5/6] 推理依赖 kaolin / gsplat / seaborn / gradio (.[inference])"
"${PY}" -m pip install -e '.[inference]'

echo "==> 打 hydra 补丁"
[ -f "${SCRIPT_DIR}/patching/hydra" ] && (bash "${SCRIPT_DIR}/patching/hydra" || echo "   (hydra 补丁失败，可忽略)")

echo "==> [6/6] 预下载运行时权重（DINO ViT-L + MoGe）"
export HF_HUB_DISABLE_XET=1
# 6.1 DINO ViT-L/14：torch.hub 从 facebookresearch/dinov2 加载
DINO="${HOME}/.cache/torch/hub/checkpoints/dinov2_vitl14_reg4_pretrain.pth"
mkdir -p "$(dirname "${DINO}")"
[ -s "${DINO}" ] || curl -C - -sL -o "${DINO}" \
  "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_reg4_pretrain.pth"
# 6.2 MoGe 深度模型（禁用 Xet，否则 401）
"${PY}" -c "from huggingface_hub import hf_hub_download; print('MoGe:', hf_hub_download('Ruicheng/moge-vitl','model.pt'))" || \
  echo "   (MoGe 下载失败，运行前请手动放到 HF 缓存)"

echo ""
echo "=============================================================="
echo " 环境构建完成: ${ENV_DIR}"
echo " 启动服务:  bash ${SCRIPT_DIR}/run_sam3d_service.sh"
echo " 冒烟测试:  bash ${SCRIPT_DIR}/test_sam3d_service.sh"
echo "=============================================================="
