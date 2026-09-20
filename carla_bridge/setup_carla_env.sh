#!/usr/bin/env bash
# 一次性环境准备：Python 3.7 环境 + CARLA PythonAPI(+numpy,opencv)
#
# 为什么是 3.7：CARLA 0.9.15 的 Linux 发行包只带 cp27 / cp37 的 wheel
# （PythonAPI/carla/dist/carla-0.9.15-cp37-cp37m-manylinux_2_27_x86_64.whl）。
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
. "$HERE/carla_env.sh"

PY37_PREFIX=${PY37_PREFIX:-/autodl-fs/data/carla/py37}
CONDA=${CONDA:-/root/miniconda3/bin/conda}
PIP_INDEX=${PIP_INDEX:-https://pypi.tuna.tsinghua.edu.cn/simple}

if [ ! -f "$CARLA_HOME/PythonAPI/carla/dist/carla-0.9.15-cp37-cp37m-manylinux_2_27_x86_64.whl" ]; then
  echo "!! 找不到 $CARLA_HOME/PythonAPI/carla/dist —— CARLA 还没解压？" >&2
  exit 1
fi

echo "== 1) 建 Python3.7 环境：$PY37_PREFIX =="
if [ ! -x "$PY37_PREFIX/bin/python" ]; then
  "$CONDA" create -y -p "$PY37_PREFIX" python=3.7 pip
fi
"$PY37_PREFIX/bin/python" -V

echo "== 2) numpy / opencv（py3.7 能装的版本）=="
"$PY37_PREFIX/bin/pip" install -q --index-url "$PIP_INDEX" --no-cache-dir \
  "numpy<1.22" "opencv-python==4.5.5.64" 2>/dev/null || true

echo "== 3) CARLA PythonAPI =="
"$PY37_PREFIX/bin/pip" install -q --no-deps \
  "$CARLA_HOME/PythonAPI/carla/dist/carla-0.9.15-cp37-cp37m-manylinux_2_27_x86_64.whl"
"$PY37_PREFIX/bin/python" -c "import carla, numpy, cv2; \
print('  carla API ->', carla.__file__); \
print('  numpy', numpy.__version__, '/ cv2', cv2.__version__)"

echo
echo "OK。下一步："
echo "  cd $HERE"
echo "  ./start_carla_server.sh --daemon"
echo "  ./run_bridge.sh --scenario <场景>.world.json --out-dir ../output/carla/demo"
