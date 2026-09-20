#!/usr/bin/env bash
# =============================================================================
# 修复 SAM 3D 环境剩余依赖并验证
#   - pytorch3d 编译失败根因: conda CUDA 头文件在 $ENV/targets/x86_64-linux/include，
#     不在 $ENV/include，需通过 CPATH 暴露给 g++。
#   - flash_attn 直接用官方预编译轮子（省去 ~20-30 分钟源码编译）。
#   - 最后重启服务并跑官方 demo 图冒烟重建。
# 日志: /root/autodl-fs/dggt-main/sam3d-fix.log
# =============================================================================
set -uo pipefail

ROOT=/root/autodl-fs/dggt-main
ENV=/root/autodl-tmp/conda_envs/sam3d-objects
LOG="${ROOT}/sam3d-fix.log"
exec > "${LOG}" 2>&1
echo "=== fix start $(date) ==="

# ---- 代理 + 源 ----
if [ -f /etc/network_turbo ]; then
  # shellcheck disable=SC1091
  source /etc/network_turbo || true
  export no_proxy="localhost,127.0.0.1,modelscope.com,aliyuncs.com,tencentyun.com,wisemodel.cn,mirrors.aliyun.com,download.pytorch.org,download-r2.pytorch.org,repo.huaweicloud.com,pypi.tuna.tsinghua.edu.cn,nvidia-kaolin.s3.us-east-2.amazonaws.com"
fi

# ---- 编译环境（关键：CPATH 指向 conda CUDA targets 头文件）----
export CONDA_PREFIX="${ENV}"
export CUDA_HOME="${ENV}"
export CPATH="${ENV}/targets/x86_64-linux/include:${CPATH:-}"
export CPLUS_INCLUDE_PATH="${ENV}/targets/x86_64-linux/include:${CPLUS_INCLUDE_PATH:-}"
export LD_LIBRARY_PATH="${ENV}/targets/x86_64-linux/lib:${ENV}/lib:${LD_LIBRARY_PATH:-}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.6}"   # RTX 3090 = sm_86
export MAX_JOBS="${MAX_JOBS:-8}"
export FORCE_CUDA=1
export TMPDIR=/root/autodl-tmp/sam3d_tmp
export PIP_NO_CACHE_DIR=1
export PIP_DISABLE_PIP_VERSION_CHECK=1
export PIP_INDEX_URL="https://repo.huaweicloud.com/repository/pypi/simple"
export PIP_EXTRA_INDEX_URL="https://download.pytorch.org/whl/cu121"
export PIP_FIND_LINKS="https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.5.1_cu121.html"
mkdir -p "${TMPDIR}"

PY="${ENV}/bin/python"
cd "${ROOT}/sam-3d-objects" || exit 1

echo "=== [A] pytorch3d（源码编译）==="
"${PY}" -m pip install --no-build-isolation \
  "pytorch3d @ git+https://github.com/facebookresearch/pytorch3d.git@75ebeeaea0908c5527e7b1e305fbc7681382db47"
echo "pytorch3d rc=$?"

echo "=== [B] flash_attn（预编译轮子）==="
"${PY}" -m pip install --no-build-isolation \
  "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3%2Bcu12torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl"
echo "flash_attn rc=$?"

echo "=== [C] 推理依赖 kaolin/gsplat/seaborn/gradio (.[inference]) ==="
"${PY}" -m pip install -e '.[inference]'
echo "inference rc=$?"

echo "=== [D] 导入校验 ==="
"${PY}" - <<'PYEOF'
import importlib, torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "-")
for m in ["torchvision","torchaudio","numpy","pytorch3d","kaolin","spconv","moge",
          "utils3d","hydra","omegaconf","seaborn","gradio","gsplat","xformers",
          "flash_attn","sam3d_objects"]:
    try:
        mod = importlib.import_module(m)
        print(m, "OK", getattr(mod, "__version__", "?"))
    except Exception as e:
        print(m, "FAIL", type(e).__name__, str(e)[:140])
import sys
sys.path.insert(0, "notebook")
try:
    from inference import Inference  # noqa: F401
    print("inference import OK")
except Exception as e:
    print("inference import FAIL", type(e).__name__, str(e)[:200])
PYEOF

echo "=== [E] 重启服务 + 冒烟重建 ==="
pkill -f "sam3d[_]service.py" 2>/dev/null
sleep 2
nohup "${PY}" sam3d_service.py > "${ROOT}/sam3d-service.log" 2>&1 &
for _ in $(seq 1 24); do
  curl -s --max-time 5 http://127.0.0.1:8001/health >/dev/null 2>&1 && break
  sleep 5
done
echo "health: $(curl -s --max-time 10 http://127.0.0.1:8001/health)"

IMG="${ROOT}/sam-3d-objects/notebook/images/shutterstock_stylish_kidsroom_1640806567"
echo "=== reconstruct (首次加载 ~13GB 权重) ==="
curl -sS --max-time 2400 -X POST http://127.0.0.1:8001/reconstruct \
  -F "image=@${IMG}/image.png" \
  -F "mask=@${IMG}/14.png" \
  -F "seed=42"
echo ""
echo "=== 服务日志尾部 ==="
tail -30 "${ROOT}/sam3d-service.log"
echo "=== ALL DONE $(date) ==="
