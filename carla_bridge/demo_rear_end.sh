#!/usr/bin/env bash
# 一键 demo：起 CARLA(离屏, GPU) -> 跑 rear_end 事故场景 -> 出 MP4 / 事件 JSON
#
#   ./demo_rear_end.sh                                     # 默认 rear_end_s7b 场景
#   ./demo_rear_end.sh ../output/export/rear_end_s7/rear-end_s7.world.json
#   LIVE=1 ./demo_rear_end.sh                              # 改成无限循环 + 网页直播(8090)
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "$HERE/.." && pwd)
. "$HERE/carla_env.sh"

SCEN=${1:-$ROOT/output/export/rear_end_s7b/rear-end_s7.world.json}
NAME=$(basename "${SCEN%.world.json}")
OUT=${2:-$ROOT/output/carla/$NAME}
CARLA_PORT=${CARLA_PORT:-2000}
QUALITY=${QUALITY:-High}
MAP=${MAP:-Town10HD_Opt}
FPS=${FPS:-20}
CAMS=${CAMS:-chase,birdseye}
CAMSIZE=${CAMSIZE:-960x540}
FOCUS=${FOCUS:-crash}

echo "== 1) 启动 CARLA server（离屏, 画质=$QUALITY, GPU Vulkan）=="
"$HERE/fix_nvidia_vulkan.sh" || true
CARLA_PORT=$CARLA_PORT QUALITY=$QUALITY "$HERE/start_carla_server.sh" --daemon

echo "== 2) 等 RPC 端口 $CARLA_PORT =="
for i in $(seq 1 60); do
  if timeout 15 "$CARLA_PY" -c "import carla;c=carla.Client('127.0.0.1',$CARLA_PORT);c.set_timeout(10);print('UP')" 2>/dev/null | grep -q UP; then
    echo "   ready (${i}x8s)"; break
  fi
  sleep 8
done

if [ "${LIVE:-0}" = "1" ]; then
  echo "== 3) 无限循环 + MJPEG 直播 http://127.0.0.1:8090/ =="
  exec "$HERE/run_bridge.sh" --scenario "$SCEN" --map "$MAP" --fps "$FPS" \
    --cameras "$CAMS" --cam-size "$CAMSIZE" --focus "$FOCUS" \
    --loop 0 --hold-end 3 --live-http "${LIVE_PORT:-8090}"
fi

echo "== 3) 跑场景 -> $OUT =="
"$HERE/run_bridge.sh" \
  --scenario "$SCEN" --out-dir "$OUT" --map "$MAP" --fps "$FPS" \
  --cameras "$CAMS" --cam-size "$CAMSIZE" --focus "$FOCUS" --every 20

echo
echo "== 4) 产物 =="
ls -la "$OUT" || true
echo
echo "在 VSCode 里右键 MP4 -> Download 到本地播放；"
echo "或者在 PORTS 面板转发 8090 后浏览器打开 http://127.0.0.1:8090/（LIVE=1 时）。"
