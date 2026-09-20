#!/usr/bin/env bash
# CARLA server 启动器
#   RENDER_MODE=offscreen  无头离屏渲染（默认，最稳）
#   RENDER_MODE=xvfb       画到 Xvfb :99（配合 start_carla_vnc.sh 在浏览器里看）
#   RENDER_MODE=x11        画到已有 $DISPLAY
#
# UE4 拒绝以 root 运行，所以如果当前是 root，会自动降权到 RUN_AS 用户（默认 carla）。
#
# 用法：
#   ./start_carla_server.sh                # 前台
#   ./start_carla_server.sh --daemon       # 后台（日志 logs/carla_server.log）
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=carla_env.sh
. "$HERE/carla_env.sh"

CARLA_PORT=${CARLA_PORT:-2000}
QUALITY=${QUALITY:-High}
RENDER_MODE=${RENDER_MODE:-offscreen}
DISPLAY_NUM=${DISPLAY_NUM:-99}
GPU=${GPU:-0}
RESX=${RESX:-1280}
RESY=${RESY:-720}
LOGDIR=${LOGDIR:-/autodl-fs/data/carla/logs}
RUN_AS=${RUN_AS:-carla}

mkdir -p "$LOGDIR"
LOG="$LOGDIR/carla_server.log"

if [ ! -x "$CARLA_HOME/CarlaUE4.sh" ]; then
  echo "找不到 $CARLA_HOME/CarlaUE4.sh —— 先解压 CARLA 发行包（见 README_CARLA.md）" >&2
  exit 1
fi

# UE4 不允许 root 运行 -> 降权
LAUNCH=()
if [ "$(id -u)" = "0" ]; then
  if ! id "$RUN_AS" >/dev/null 2>&1; then
    useradd -m -s /bin/bash "$RUN_AS"
  fi
  mkdir -p "$CARLA_HOME/CarlaUE4/Saved" "$CARLA_HOME/CarlaUE4/Config"
  chmod -R 777 "$CARLA_HOME/CarlaUE4/Saved" "$LOGDIR" 2>/dev/null || true
  if command -v setpriv >/dev/null 2>&1; then
    LAUNCH=(env "HOME=/home/$RUN_AS" setpriv "--reuid=$RUN_AS" "--regid=$RUN_AS" --init-groups)
  else
    LAUNCH=(env "HOME=/home/$RUN_AS" runuser -u "$RUN_AS" --)
  fi
fi

ARGS=(-nosound "-carla-rpc-port=$CARLA_PORT" "-quality-level=$QUALITY" "-graphicsadapter=$GPU")
# 软件渲染(llvmpipe/lavapipe)下 shader 编译很慢，UE4 的 60s "GameThread timed out waiting for
# RenderThread" 看门狗会直接把 server 打死；那种情况要 ONETHREAD=1。GPU 正常时用 0。
ONETHREAD=${ONETHREAD:-0}
if [ "$ONETHREAD" = "1" ]; then ARGS+=(-onethread); fi
case "$RENDER_MODE" in
  offscreen) ARGS+=(-RenderOffScreen) ;;
  xvfb)      ARGS+=(-windowed "-ResX=$RESX" "-ResY=$RESY"); export DISPLAY=":$DISPLAY_NUM" ;;
  x11)       ARGS+=(-windowed "-ResX=$RESX" "-ResY=$RESY") ;;
  *) echo "未知 RENDER_MODE=$RENDER_MODE" >&2; exit 1 ;;
esac

echo "[carla] home=$CARLA_HOME mode=$RENDER_MODE port=$CARLA_PORT quality=$QUALITY software=$SOFTWARE_RENDER"
echo "[carla] run_as=${RUN_AS:-$(id -un)}   log -> $LOG"

if [ "${1:-}" = "--daemon" ]; then
  if pgrep -f "carla-rpc-port=$CARLA_PORT" >/dev/null; then
    echo "[carla] 已在运行 (port $CARLA_PORT)"; exit 0
  fi
  nohup "${LAUNCH[@]}" "$CARLA_HOME/CarlaUE4.sh" "${ARGS[@]}" >"$LOG" 2>&1 &
  echo "[carla] pid=$!  （tail -f $LOG 看日志）"
  sleep 12
  tail -6 "$LOG" || true
else
  exec "${LAUNCH[@]}" "$CARLA_HOME/CarlaUE4.sh" "${ARGS[@]}"
fi
