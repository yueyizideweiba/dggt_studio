#!/usr/bin/env bash
# 用 CARLA 官方 ScenarioRunner 跑「CARLA 对齐版」OpenSCENARIO（标准逻辑路径）
#
# 说明：
#   * 输入要用 carla_bridge 导出的 <名字>_carla.xosc（默认场景自带 LogicFile=地图名、
#     FileHeader.description 以 "CARLA:" 开头、Polyline 的 Vertex 带 <Position><WorldPosition>）；
#   * ScenarioRunner 会把 Polyline 的每个顶点变成 waypoint，用 ChangeActorWaypoints
#     让车自己开过去（不是逐帧硬播放）；
#   * 它是逻辑/评测运行器，跑得很快且默认不渲染 —— 想看画面请用 run_bridge.sh。
#
# 用法：
#   ./run_scenario_runner.sh <xxx_carla.xosc> [额外参数...]
#   TIMEOUT=200 ./run_scenario_runner.sh out/rear-end_s7_carla.xosc --output
#
# 依赖 CARLA server 已启动且已加载同一张地图（run_bridge.sh --export-xosc 会把地图 load 好）。
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=carla_env.sh
. "$HERE/carla_env.sh"

SR=${SCENARIO_RUNNER:-/autodl-fs/data/carla/scenario_runner}
XOSC=${1:?用法: ./run_scenario_runner.sh <xxx_carla.xosc> [scenario_runner 参数...]}; shift || true

if [ ! -f "$XOSC" ]; then
  echo "找不到 xosc: $XOSC" >&2
  exit 1
fi
if [ ! -d "$SR" ]; then
  echo "找不到 ScenarioRunner: $SR" >&2
  echo "克隆：git clone --depth 1 -b v0.9.15 https://github.com/carla-simulator/scenario_runner.git $SR" >&2
  exit 1
fi

export CARLA_ROOT="$CARLA_HOME"
export PYTHONPATH="$CARLA_HOME/PythonAPI/carla:$SR:${PYTHONPATH:-}"

cd "$SR"
echo "[SR] $(basename "$XOSC")  ->  scenario_runner.py --sync"
exec "$CARLA_PY" -u scenario_runner.py --openscenario "$XOSC" --sync \
  --host 127.0.0.1 --port "${CARLA_PORT:-2000}" --timeout "${TIMEOUT:-120}" "$@"
