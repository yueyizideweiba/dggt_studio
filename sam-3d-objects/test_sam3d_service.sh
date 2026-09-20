#!/usr/bin/env bash
# =============================================================================
# SAM 3D 微服务冒烟测试：用官方 demo 图片 + 掩码验证重建链路
#
# 用法:
#   bash sam-3d-objects/test_sam3d_service.sh [图片目录]
# 默认图片目录: notebook/images/shutterstock_stylish_kidsroom_1640806567
#   其中 image.png 为原图，14.png 为第 14 号掩码（官方 single-object demo 用的就是这个）
# =============================================================================
set -euo pipefail

HOST="${SAM3D_URL:-http://127.0.0.1:8001}"
IMG_DIR="${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/notebook/images/shutterstock_stylish_kidsroom_1640806567}"

echo "==> [1/2] /health"
curl -sS --max-time 15 "$HOST/health" | head -c 1000; echo; echo

if [ ! -f "${IMG_DIR}/image.png" ]; then
  echo "找不到测试图片: ${IMG_DIR}/image.png" >&2
  exit 1
fi
# 找一张掩码
MASK="${IMG_DIR}/14.png"
[ -f "$MASK" ] || MASK="${IMG_DIR}/0.png"

echo "==> [2/2] /reconstruct  image=${IMG_DIR}/image.png  mask=${MASK}"
echo "    （首次会加载 ~13GB 权重，需耐心等待）"
curl -sS --max-time 1200 -X POST "$HOST/reconstruct" \
  -F "image=@${IMG_DIR}/image.png" \
  -F "mask=@${MASK}" \
  -F "seed=42" \
  -F "format=ply" | head -c 1500
echo
