#!/usr/bin/env bash
# =============================================================================
# 关掉本项目的所有服务（studio / SAM3D / text2entity [+ CARLA]）
#
#   bash stop_all.sh                 # 停 studio + SAM3D + text2entity
#   bash stop_all.sh --carla         # 连 CARLA / 直播 / VNC 一起停（等同 --all）
#   bash stop_all.sh --studio-only   # 只停 studio
#   bash stop_all.sh --no-llm        # 不停 text2entity
#   bash stop_all.sh --no-sam3d      # 不停 SAM3D
#
# 注意：SAM3D / text2entity 都是"监督循环 + python 服务"两层，**必须先杀循环**，
# 否则循环会立刻把 python 重新拉起来（这是常见的"杀了又活"的原因）。
# 顺序：text2entity → SAM3D → studio → (CARLA)。
#
# 停完会检查端口是否真的释放；服务被停掉后显存也会一起归还。
# =============================================================================
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO"

STOP_STUDIO=1
STOP_SAM3D=1
STOP_T2E=1
STOP_CARLA=0

for a in "$@"; do
    case "$a" in
        --studio-only) STOP_SAM3D=0; STOP_T2E=0 ;;
        --no-llm)      STOP_T2E=0 ;;
        --no-sam3d)    STOP_SAM3D=0 ;;
        --carla|--all) STOP_CARLA=1 ;;
        -h|--help)     sed -n '2,20p' "$0"; exit 0 ;;
        *) echo "未知参数: $a（-h 看用法）"; exit 2 ;;
    esac
done

port_open() { (echo >"/dev/tcp/127.0.0.1/$1") >/dev/null 2>&1; }

# 按 pattern 杀进程；排除本脚本自己与调用它的 shell，避免把自己干掉
kill_pat() {
    local pat="$1" desc="$2" pids alive p
    pids="$(pgrep -f "$pat" 2>/dev/null || true)"
    pids="$(printf '%s\n' $pids | grep -v -x -e "$$" -e "$PPID" | grep -v '^$' | tr '\n' ' ' || true)"
    if [ -z "${pids// /}" ]; then
        echo "   $desc: 未在运行"
        return 0
    fi
    echo "   $desc: 停止 $(printf '%s\n' $pids | wc -l) 个进程（$(printf '%s\n' $pids | tr '\n' ' ')）"
    kill $pids 2>/dev/null || true
    sleep 1
    alive=""
    for p in $pids; do kill -0 "$p" 2>/dev/null && alive="$alive $p"; done
    if [ -n "${alive// /}" ]; then
        echo "   $desc:$alive 未退出 → SIGKILL"
        kill -9 $alive 2>/dev/null || true
    fi
}

echo "== 停止服务 =="

if [ "$STOP_T2E" = "1" ]; then
    echo " -- text2entity (8002)"
    kill_pat "text2entity/supervise.sh" "supervise.sh（看门狗）"
    kill_pat "text2entity/run_service.sh" "run_service.sh（监督循环）"
    kill_pat "text2entity_service.py" "text2entity_service.py"
fi

if [ "$STOP_SAM3D" = "1" ]; then
    echo " -- SAM3D (8001)"
    kill_pat "run_sam3d_service.sh" "run_sam3d_service.sh（监督循环）"
    kill_pat "sam3d_service.py" "sam3d_service.py"
fi

if [ "$STOP_STUDIO" = "1" ]; then
    echo " -- studio (8000)"
    kill_pat "api_server.py" "api_server.py"
fi

if [ "$STOP_CARLA" = "1" ]; then
    echo " -- CARLA / 直播 / VNC"
    if [ -x carla_bridge/stop_all.sh ]; then
        bash carla_bridge/stop_all.sh || true
    else
        kill_pat "carla_scenario_bridge" "carla_scenario_bridge"
        kill_pat "CarlaUE4-Linux-Shipping" "CARLA server"
        kill_pat "websockify --web" "websockify"
        kill_pat "x11vnc -display" "x11vnc"
    fi
fi

# 等端口/进程真正释放（监督循环被 SIGKILL 后，子进程可能还在收尾）
sleep 2
echo
echo "==================== 状态 ===================="
for spec in "studio:8000" "SAM3D:8001" "text2entity:8002" "CARLA:2000"; do
    name="${spec%%:*}"; port="${spec##*:}"
    if port_open "$port"; then
        printf '  %-12s :%s  仍在监听（可能有别的东西占着，或没杀干净）\n' "$name" "$port"
    else
        printf '  %-12s :%s  已停止 ✓\n' "$name" "$port"
    fi
done
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader 2>/dev/null | sed 's/^/  显存 used,total = /' || true
