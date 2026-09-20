#!/usr/bin/env bash
# 修复/检查 AutoDL 容器里残缺的 NVIDIA 图形用户态（CUDA 正常但 Vulkan/GL 起不来）
#
# 背景（本机实测）：
#   容器里 /usr/lib/x86_64-linux-gnu 下 580.142 / 570.x 的驱动库都是 0 字节占位文件，
#   只有少数 580.76.05 的库被 bind mount 进来，而且 libnvidia-gpucomp.so.580.76.05
#   等一批库整个缺失 -> libGLX_nvidia（即 Vulkan ICD）初始化失败：
#     vk_icdNegotiateLoaderICDInterfaceVersion 返回 -3，vulkaninfo 报
#     "Could not get 'vkCreateInstance' via 'vk_icdGetInstanceProcAddr'"。
#   把官方同版本驱动包里的用户态库补齐 + ldconfig 之后，RTX 3090 就能正常出 Vulkan 设备。
set -euo pipefail

LIBDIR=/usr/lib/x86_64-linux-gnu
DRV=$(grep -oE '[0-9]{3}\.[0-9]+\.[0-9]+' /proc/driver/nvidia/version 2>/dev/null | head -1)
echo "[nvidia] kernel driver = ${DRV:-unknown}"

command -v vulkaninfo >/dev/null 2>&1 || {
  echo "[nvidia] 装 vulkan-tools ..."
  apt-get install -y --no-install-recommends vulkan-tools >/tmp/apt_vulkan_tools.log 2>&1 || true
}

check() {
  VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json \
    timeout 90 vulkaninfo --summary 2>/dev/null | grep -q 'deviceName.*NVIDIA'
}

# 0) ICD json 得是合法 JSON（本机原来是 0 字节）
if [ ! -s /etc/vulkan/icd.d/nvidia_icd.json ] || ! grep -q file_format_version /etc/vulkan/icd.d/nvidia_icd.json 2>/dev/null; then
  cat > /etc/vulkan/icd.d/nvidia_icd.json <<'EOF'
{
    "file_format_version": "1.0.0",
    "ICD": {
        "library_path": "libGLX_nvidia.so.0",
        "api_version": "1.3.194"
    }
}
EOF
  echo "[nvidia] 写好了 /etc/vulkan/icd.d/nvidia_icd.json"
fi

if check; then
  echo "[nvidia] ✅ Vulkan 已能用 NVIDIA GPU，无需修复"
  VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json timeout 90 vulkaninfo --summary 2>/dev/null | grep -E 'deviceName|driverInfo'
  exit 0
fi

echo "[nvidia] Vulkan 用不了，开始补驱动用户态库 ..."
[ -n "$DRV" ] || { echo "读不到驱动版本，退出" >&2; exit 1; }

RUN=/autodl-fs/data/nv_driver/NVIDIA-Linux-x86_64-${DRV}.run
EXT=/autodl-fs/data/nv_driver/extract
mkdir -p /autodl-fs/data/nv_driver

if [ ! -f "$RUN" ]; then
  echo "[nvidia] 下载官方驱动包（约 390MB）：download.nvidia.com ..."
  curl -L --retry 20 --retry-delay 5 -o "$RUN" \
    "https://download.nvidia.com/XFree86/Linux-x86_64/${DRV}/NVIDIA-Linux-x86_64-${DRV}.run"
fi

if [ ! -d "$EXT" ]; then
  echo "[nvidia] 解包（--extract-only，不会安装驱动）..."
  sh "$RUN" --extract-only --target "$EXT"
fi

echo "[nvidia] 拷贝 $DRV 的用户态库到 $LIBDIR ..."
cp -f "$EXT"/lib*.so."$DRV" "$LIBDIR"/ 2>/dev/null || true
# 32 位版本按需忽略；顺便把 glvnd 的 EGL 配置也放好
[ -f "$EXT/10_nvidia.json" ] && mkdir -p /usr/share/glvnd/egl_vendor.d && \
  cp -f "$EXT/10_nvidia.json" /usr/share/glvnd/egl_vendor.d/10_nvidia.json
ldconfig
echo "[nvidia] ldconfig 完成"

if check; then
  echo "[nvidia] ✅ 修复成功"
  VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json timeout 90 vulkaninfo --summary 2>/dev/null | grep -E 'deviceName|driverInfo'
else
  echo "[nvidia] ❌ 仍然不行：容器对 GPU 图形接口的支持可能确实不完整。" >&2
  echo "         退路：SOFTWARE_RENDER=1 ./start_carla_server.sh（Mesa 软件渲染，很慢）" >&2
  exit 1
fi
