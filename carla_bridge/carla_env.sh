#!/usr/bin/env bash
# 公共环境变量（被 start_carla_*.sh / run_bridge.sh / demo_*.sh source）
#
# 目录约定（CARLA 0.9.15 的 tar.gz 解压出来是"平铺"的，没有 CARLA_0.9.15/ 外壳）：
#   /autodl-fs/data/carla/        CARLA 发行包（CarlaUE4.sh / PythonAPI / CarlaUE4/...）
#   /autodl-fs/data/carla/py37/   Python 3.7 环境（CARLA 0.9.15 Linux 只提供 cp37 wheel）
#   /autodl-fs/data/carla/logs/   server 日志
#
# 渲染：默认走 NVIDIA GPU 的 Vulkan（本机 RTX 3090，实测 30-50 fps）。
# 如果哪天 GPU Vulkan 又坏了，可以退化到 Mesa 软件渲染：SOFTWARE_RENDER=1（很慢）。

export CARLA_HOME=${CARLA_HOME:-/autodl-fs/data/carla}
export CARLA_ROOT="$CARLA_HOME"
export CARLA_PY=${CARLA_PY:-/autodl-fs/data/carla/py37/bin/python}
export SOFTWARE_RENDER=${SOFTWARE_RENDER:-0}

if [ "$SOFTWARE_RENDER" = "1" ]; then
  export VK_ICD_FILENAMES=${VK_ICD_FILENAMES:-/usr/share/vulkan/icd.d/lvp_icd.x86_64.json}
  export LIBGL_ALWAYS_SOFTWARE=1
  export GALLIUM_DRIVER=llvmpipe
  export MESA_LOADER_DRIVER_OVERRIDE=llvmpipe
  export LP_NUM_THREADS=${LP_NUM_THREADS:-32}
else
  export VK_ICD_FILENAMES=${VK_ICD_FILENAMES:-/etc/vulkan/icd.d/nvidia_icd.json}
fi

export SDL_AUDIODRIVER=${SDL_AUDIODRIVER:-dummy}
