#!/usr/bin/env bash
# 启动 studio（前端 + API + CARLA）。
#
#   ./start_studio.sh            # 只起 studio 后端（8000）—— 不碰 GPU，CARLA 页面上按需再开
#   ./start_studio.sh --carla    # 顺便把 CARLA server 也起起来（占 ~5-6GB 显存）
#   ./start_studio.sh --live     # 起 CARLA + 实时直播（前端 ④ 页面直接能看到画面）
#
# 起完之后：
#   * 前端    http://127.0.0.1:8000/studio/   ← 一个端口搞定（VSCode 只需转发 8000）
#   * 接口文档 http://127.0.0.1:8000/docs
#   * CARLA   RPC 127.0.0.1:2000（TrafficManager 用 8010，避开 studio 的 8000）
#
# 显存提示：CARLA 一开 ~5-6GB，跑完约 1 分钟没人用会自动释放；
#          也可以在前端点「🧹 释放显存」，或直接 ./stop_all.sh。
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$HERE/.." && pwd)
BACKEND="$REPO/studio/backend"
LOGDIR=${LOGDIR:-/autodl-fs/data/carla/logs}
PY=${STUDIO_PY:-/root/autodl-tmp/conda_envs/dggt/bin/python}
PORT=${STUDIO_PORT:-8000}
mkdir -p "$LOGDIR"

WANT_CARLA=0
[ "${1:-}" = "--carla" ] && WANT_CARLA=1
[ "${1:-}" = "--live" ] && WANT_CARLA=1

echo "== 1) 起 studio 后端（前端也挂在同一个端口）=="
if curl -s -o /dev/null --max-time 3 "http://127.0.0.1:${PORT}/api/carla/status"; then
  echo "   已在 $PORT 跑着"
else
  cd "$BACKEND"
  setsid nohup "$PY" api_server.py > "$LOGDIR/studio_backend.log" 2>&1 < /dev/null &
  echo "   backend pid=$!"
  for i in $(seq 1 30); do
    sleep 5
    curl -s -o /dev/null --max-time 5 "http://127.0.0.1:${PORT}/api/carla/status" && { echo "   就绪 t=+$((i*5))s"; break; }
  done
fi

if [ "$WANT_CARLA" = "1" ]; then
  echo "== 2) 检查/修复 NVIDIA Vulkan =="
  bash "$HERE/fix_nvidia_vulkan.sh" | tail -2 || true
  echo "== 3) 起 CARLA server（会占 ~5-6GB 显存）=="
  curl -s --max-time 180 -X POST "http://127.0.0.1:${PORT}/api/carla/server" \
    -H 'Content-Type: application/json' -d '{"action":"start","quality":"High","map":"Town10HD_Opt"}' \
    | head -c 300; echo
else
  echo "== 2) 跳过 CARLA（省显存）。要开就在前端 ④ 页面点「启动 CARLA」，"
  echo "      或者重跑：$0 --carla"
fi

if [ "${1:-}" = "--live" ]; then
  echo "== 4) 开实时直播（默认用最典型的 rear-end 场景）=="
  SCEN=${SCEN:-output/export/rear_end_s7b/rear-end_s7.world.json}
  curl -s --max-time 240 -X POST "http://127.0.0.1:${PORT}/api/carla/live" \
    -H 'Content-Type: application/json' \
    -d "{\"action\":\"start\",\"scenario\":\"$SCEN\",\"map\":\"Town10HD_Opt\",\"focus\":\"crash\",\"weather\":\"ClearNoon\"}" \
    | head -c 300; echo
fi

nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader 2>/dev/null | sed 's/^/显存 used,free = /' || true

cat <<EOF

==================== 好了 ====================
前端：    http://127.0.0.1:${PORT}/studio/
接口文档： http://127.0.0.1:${PORT}/docs
VSCode 只需转发端口 ${PORT}，浏览器打开上面的地址；
左侧「实验室」是一条流程：① 挑事故 → ② 批量生成 → ③ 闭环评估 → ④ CARLA 验证 → ⑤/⑥。

释放显存： 前端 ④ 页面点「🧹 释放显存」  or  $HERE/stop_all.sh
停 studio： pkill -f api_server.py
==============================================
EOF
