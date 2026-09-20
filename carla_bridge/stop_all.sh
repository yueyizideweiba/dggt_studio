#!/usr/bin/env bash
# 停掉 CARLA server / 直播桥接 / noVNC
set -uo pipefail
printf '%s\n' '#!/bin/bash' 'pkill -9 -f "carla_scenario_bridge" 2>/dev/null' \
  'pkill -9 -f "CarlaUE4-Linux-Shipping" 2>/dev/null' \
  'pkill -f "websockify --web" 2>/dev/null' \
  'pkill -f "x11vnc -display" 2>/dev/null' \
  'sleep 2; echo "已停止 CARLA / 桥接 / VNC（SIGKILL，UE4 会忽略 SIGTERM）"' > /tmp/_stop_carla.sh
bash /tmp/_stop_carla.sh
nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader 2>/dev/null | sed 's/^/显存 used,free = /' || true
