#!/usr/bin/env bash
# =============================================================================
# 安装收尾：等 setup 完成后自动做导入校验 -> 启动服务 -> 冒烟重建
# 结果写入 /root/autodl-fs/dggt-main/sam3d-verify.log
# =============================================================================
set -uo pipefail

ROOT=/root/autodl-fs/dggt-main
ENV=/root/autodl-tmp/conda_envs/sam3d-objects
LOG="${ROOT}/sam3d-verify.log"
SVC_LOG="${ROOT}/sam3d-service.log"

exec >>"${LOG}" 2>&1
echo "=== verify watcher start $(date) ==="

# ---- 等待安装结束 ----
for _ in $(seq 1 360); do
  if ! pgrep -f "setup_sam3d[_]env.sh" >/dev/null 2>&1 && \
     ! pgrep -f "sam3d[-]objects/bin/python -m pip" >/dev/null 2>&1; then
    break
  fi
  sleep 20
done
echo "=== setup finished at $(date) ==="

cd "${ROOT}/sam-3d-objects" || exit 1
export CONDA_PREFIX="${ENV}"
export CUDA_HOME="${ENV}"

echo "--- 导入校验 ---"
"${ENV}/bin/python" - <<'PY'
import importlib, torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "-")
for m in ["torchvision","torchaudio","numpy","pytorch3d","kaolin","spconv","moge",
          "utils3d","hydra","omegaconf","seaborn","gradio","gsplat","sam3d_objects"]:
    try:
        mod = importlib.import_module(m)
        print(m, "OK", getattr(mod, "__version__", "?"))
    except Exception as e:
        print(m, "FAIL", type(e).__name__, str(e)[:140])
try:
    import sys; sys.path.insert(0, "notebook")
    from inference import Inference
    print("inference import OK")
except Exception as e:
    print("inference import FAIL", type(e).__name__, str(e)[:200])
PY

echo "--- 启动服务 ---"
nohup "${ENV}/bin/python" sam3d_service.py > "${SVC_LOG}" 2>&1 &
SVC_PID=$!
echo "service pid=${SVC_PID}"

echo "--- 等待 /health ---"
for _ in $(seq 1 24); do
  if curl -s --max-time 5 http://127.0.0.1:8001/health >/dev/null 2>&1; then break; fi
  sleep 5
done
echo "health: $(curl -s --max-time 10 http://127.0.0.1:8001/health)"

echo "--- 冒烟重建（首次加载约13GB权重，请耐心） ---"
IMG="${ROOT}/sam-3d-objects/notebook/images/shutterstock_stylish_kidsroom_1640806567"
curl -sS --max-time 2400 -X POST http://127.0.0.1:8001/reconstruct \
  -F "image=@${IMG}/image.png" \
  -F "mask=@${IMG}/14.png" \
  -F "seed=42" \
  -F "format=ply"
echo ""
echo "--- 服务日志尾部 ---"
tail -40 "${SVC_LOG}"
echo "=== verify watcher done $(date) ==="
