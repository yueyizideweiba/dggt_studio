#!/usr/bin/env bash
# text2entity 微服务的"防掉线"守护：把 run_service.sh 再包一层，崩溃/被信号打断后自动拉起。
#
# 为什么需要：run_service.sh 本身是个 `while true` 循环（/unload 会让 python 退出，
# 靠它重启）。但它偶尔会整条被信号带走（例如显存 OOM 时连同子进程一起被清理），
# 结果 8002 就没人在监听了，前端表现为"微服务未启动 / Connection refused"。
# 这里再加一层 setsid + 循环，并且把 pid 记下来方便排查。
#
#   setsid nohup bash text2entity/supervise.sh </dev/null >/dev/null 2>&1 &
#
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG="${TEXT2ENTITY_LOG:-/tmp/text2entity.log}"
PIDFILE="${TEXT2ENTITY_PIDFILE:-/tmp/text2entity.supervisor.pid}"

echo $$ > "$PIDFILE"
while true; do
    bash "${SCRIPT_DIR}/run_service.sh" >> "$LOG" 2>&1
    echo "[supervise] run_service.sh 退出，2s 后重启 ($(date '+%F %T'))" >> "$LOG"
    sleep 2
done
