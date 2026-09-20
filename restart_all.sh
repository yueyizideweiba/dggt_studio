#!/usr/bin/env bash
# =============================================================================
# 重启本项目的所有服务（先 stop_all.sh 再 start_all.sh）
#
#   bash restart_all.sh                 # 重启 studio + SAM3D + text2entity
#   bash restart_all.sh --studio-only   # 只重启 studio（改后端代码后最常用）
#   bash restart_all.sh --no-llm        # studio + SAM3D
#   bash restart_all.sh --carla         # 连 CARLA 一起重启
#
# 参数原样透传给 stop_all.sh / start_all.sh，所以两边的用法是一致的。
# 典型用途：改完 studio/backend/*.py 之后 `bash restart_all.sh --studio-only`，
# 前端静态文件改了不用重启（硬刷新即可，注意带上 ?v= 版本号）。
# =============================================================================
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO"

for a in "$@"; do
    case "$a" in
        -h|--help) sed -n '2,14p' "$0"; exit 0 ;;
    esac
done

ARGS=("$@")

echo "############ 1/2 停止 ############"
bash "$REPO/stop_all.sh" "${ARGS[@]}"

echo
echo "############ 2/2 启动 ############"
bash "$REPO/start_all.sh" "${ARGS[@]}"

echo
echo "重启完成（参数：${ARGS[*]:-无}）"
