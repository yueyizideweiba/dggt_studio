#!/usr/bin/env bash
# 跑 DGGT->CARLA 桥接（自动用 py37 环境里的 carla PythonAPI）
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=carla_env.sh
. "$HERE/carla_env.sh"

export PYTHONPATH="$HERE:${CARLA_HOME}/PythonAPI/carla:${PYTHONPATH:-}"

PY="$CARLA_PY"
[ -x "$PY" ] || PY=${PY_FALLBACK:-/root/autodl-tmp/conda_envs/dggt/bin/python}

exec "$PY" "$HERE/carla_scenario_bridge.py" "$@"
