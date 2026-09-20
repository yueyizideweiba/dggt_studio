#!/usr/bin/env bash
# 在浏览器里实时看 CARLA：Xvfb 虚拟屏 + x11vnc + noVNC + CARLA 窗口
# 之后在 VSCode 里把 6080 端口转发到本地，浏览器打开：
#     http://127.0.0.1:6080/vnc.html?autoconnect=1&resize=scale
set -euo pipefail

DISPLAY_NUM=${DISPLAY_NUM:-99}
VNC_PORT=${VNC_PORT:-5900}
NOVNC_PORT=${NOVNC_PORT:-6080}
GEOMETRY=${GEOMETRY:-1920x1080x24}
CARLA_PORT=${CARLA_PORT:-2000}
QUALITY=${QUALITY:-Low}
LOGDIR=${LOGDIR:-/autodl-fs/data/carla/logs}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=carla_env.sh
. "$HERE/carla_env.sh"
mkdir -p "$LOGDIR"

start() { # name, check-pattern, command...
  local pat="$1"; shift
  if pgrep -f "$pat" >/dev/null; then
    echo "[vnc] 已在运行: $pat"
    return 0
  fi
  "$@" &
  echo "[vnc] 启动: $pat (pid $!)"
}

start "Xvfb :$DISPLAY_NUM" Xvfb ":$DISPLAY_NUM" -screen 0 "$GEOMETRY" -ac +extension GLX +render -noreset
sleep 2
if ! xdpyinfo -display ":$DISPLAY_NUM" >/dev/null 2>&1; then
  echo "!! Xvfb :$DISPLAY_NUM 起不来，检查 $LOGDIR/xvfb.log" >&2
  exit 1
fi

start "x11vnc.*:$DISPLAY_NUM" x11vnc -display ":$DISPLAY_NUM" -forever -shared -nopw \
  -rfbport "$VNC_PORT" -o "$LOGDIR/x11vnc.log"
sleep 1
start "websockify.*$NOVNC_PORT" websockify --web=/usr/share/novnc "$NOVNC_PORT" "localhost:$VNC_PORT"
sleep 1

export DISPLAY=":$DISPLAY_NUM"

if pgrep -f "carla-rpc-port=$CARLA_PORT" >/dev/null; then
  echo "[vnc] CARLA 已在运行 (port $CARLA_PORT)"
else
  nohup "$CARLA_HOME/CarlaUE4.sh" -windowed -ResX=1280 -ResY=720 -nosound \
    "-carla-rpc-port=$CARLA_PORT" "-quality-level=$QUALITY" \
    > "$LOGDIR/carla_vnc.log" 2>&1 &
  echo "[vnc] CARLA 窗口已启动 (pid $!)  log=$LOGDIR/carla_vnc.log"
fi

cat <<EOF

  ================ CARLA 远程桌面已就绪 ================
  noVNC  : http://127.0.0.1:${NOVNC_PORT}/vnc.html?autoconnect=1&resize=scale
  VNC    : 127.0.0.1:${VNC_PORT}   (无密码)
  CARLA  : RPC 端口 ${CARLA_PORT}

  在 VSCode 里（本机 kt）：
    1) 打开已连接的远程窗口 -> 端口(PORTS) 面板 -> 转发 6080
    2) 浏览器打开上面那个 noVNC 地址
  或者直接一条隧道：
    ssh -N -L 6080:127.0.0.1:${NOVNC_PORT} -p 38924 root@region-41.seetacloud.com
  =====================================================
EOF
