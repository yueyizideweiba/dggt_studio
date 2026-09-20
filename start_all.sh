#!/usr/bin/env bash
# =============================================================================
# 一键起服务（幂等：已在跑的就跳过，只补没起来的）
#
#   bash start_all.sh                # studio(8000) + SAM3D(8001) + text2entity(8002)
#   bash start_all.sh --studio-only  # 只 studio
#   bash start_all.sh --no-llm       # studio + SAM3D（不做文生图/LLM 解析）
#   bash start_all.sh --no-sam3d     # studio + text2entity
#   bash start_all.sh --carla        # 再带上 CARLA server（占 ~5-6GB 显存）
#
# 各服务的分工（按需开，别在 24G 卡上全塞满）：
#   studio      8000  编辑器/渲染/接口/前端，**其它一切都靠它**
#   SAM3D       8001  单图重建 3D 高斯 —— "SAM3D 替换物体"、"生成/插入新物体"要用
#   text2entity 8002  LLaDA 文生图 + Qwen-VL/LLM —— "文本添加实体"、语言编辑轨迹的
#                     **自然语言解析**要用（不在时会退化成关键词规则解析）
#   CARLA       2000  事故回放 / 实时仿真（前端 ④ 页面也可以按需起）
#
# 停：pkill -f api_server.py / pkill -f run_sam3d_service.sh / pkill -f supervise.sh
#     释放显存：curl -X POST http://127.0.0.1:8001/unload （或 8002/unload）
# =============================================================================
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO"

STUDIO_PY="${STUDIO_PY:-/root/autodl-tmp/conda_envs/dggt/bin/python}"
STUDIO_PORT="${STUDIO_PORT:-8000}"
SAM3D_PORT="${SAM3D_PORT:-8001}"
T2E_PORT="${T2E_PORT:-8002}"
LOGDIR="${LOGDIR:-/autodl-fs/data/carla/logs}"
mkdir -p "$LOGDIR"

WANT_SAM3D=1
WANT_T2E=1
WANT_CARLA=0
for a in "$@"; do
    case "$a" in
        --studio-only) WANT_SAM3D=0; WANT_T2E=0 ;;
        --no-llm)      WANT_T2E=0 ;;
        --no-sam3d)    WANT_SAM3D=0 ;;
        --carla)       WANT_CARLA=1 ;;
        -h|--help)     sed -n '2,25p' "$0"; exit 0 ;;
        *) echo "未知参数: $a（-h 看用法）"; exit 2 ;;
    esac
done

port_open() { (echo >"/dev/tcp/127.0.0.1/$1") >/dev/null 2>&1; }
wait_port() {  # $1=port $2=最多等几秒
    local i=0
    while [ "$i" -lt "$2" ]; do
        port_open "$1" && return 0
        sleep 1; i=$((i+1))
    done
    return 1
}

# ---------- 1) studio ----------
echo "== studio ($STUDIO_PORT) =="
if port_open "$STUDIO_PORT"; then
    echo "   已在跑，跳过"
else
    ( cd studio/backend && setsid nohup "$STUDIO_PY" api_server.py \
        > "$LOGDIR/studio_backend.log" 2>&1 </dev/null & )
    if wait_port "$STUDIO_PORT" 40; then echo "   已启动 (pid $(pgrep -f 'api_server.py' | head -1))";
    else echo "   ✗ 启动失败，看 $LOGDIR/studio_backend.log"; fi
fi

# ---------- 2) SAM3D ----------
if [ "$WANT_SAM3D" = "1" ]; then
    echo "== SAM3D ($SAM3D_PORT) =="
    if port_open "$SAM3D_PORT"; then
        echo "   已在跑，跳过"
    else
        setsid nohup bash sam-3d-objects/run_sam3d_service.sh \
            >> /tmp/sam3d_service.log 2>&1 </dev/null &
        if wait_port "$SAM3D_PORT" 90; then echo "   已启动"
        else echo "   ✗ 启动失败，看 /tmp/sam3d_service.log（首次会加载模型，慢一点）"; fi
    fi
fi

# ---------- 3) text2entity ----------
if [ "$WANT_T2E" = "1" ]; then
    echo "== text2entity ($T2E_PORT) =="
    if port_open "$T2E_PORT"; then
        echo "   已在跑，跳过"
    else
        setsid nohup bash text2entity/supervise.sh </dev/null >/dev/null 2>&1 &
        if wait_port "$T2E_PORT" 90; then echo "   已启动（模型懒加载，第一次调用才吃显存）"
        else echo "   ✗ 启动失败，看 /tmp/text2entity.log"; fi
    fi
fi

# ---------- 4) CARLA（可选） ----------
if [ "$WANT_CARLA" = "1" ]; then
    echo "== CARLA =="
    bash carla_bridge/start_studio.sh --carla || true
fi

# ---------- 汇总 ----------
echo
echo "==================== 状态 ===================="
for spec in "studio:$STUDIO_PORT" "SAM3D:$SAM3D_PORT" "text2entity:$T2E_PORT" "CARLA:2000"; do
    name="${spec%%:*}"; port="${spec##*:}"
    if port_open "$port"; then printf '  %-12s :%s  监听中 ✓\n' "$name" "$port"
    else printf '  %-12s :%s  未监听\n' "$name" "$port"; fi
done
echo "  前端     http://127.0.0.1:$STUDIO_PORT/studio/"
echo "  接口文档 http://127.0.0.1:$STUDIO_PORT/docs"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader 2>/dev/null | sed 's/^/  显存 used,total = /' || true
