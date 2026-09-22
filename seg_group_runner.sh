#!/bin/bash
# 一个独立的 SegFormer 分组 worker：先延时（错开 CUDA 初始化），再跑 extract_masks，
# 最后写一个 sentinel。刻意做成"独立进程"—— 不要有共同的父脚本，避免父脚本被回收时
# 把子进程一起带走。
#   用法: seg_group_runner.sh <组号> <延时秒> -- <scene_ids...>
set -uo pipefail
G="$1"; shift
DELAY="$1"; shift
[ "${1:-}" = "--" ] && shift
cd /autodl-fs/data/dggt-main
export PYTHONPATH="/autodl-fs/data/SegFormer:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
LOG="/tmp/fill_masks/seg_group${G}.log"
mkdir -p /tmp/fill_masks
rm -f "/tmp/fill_masks/seg_done_${G}"
echo "[group $G] 延时 ${DELAY}s 后开始，场景: $*" > "$LOG"
sleep "$DELAY"
echo "[group $G] 开始 $(date +%T)" >> "$LOG"
/root/autodl-tmp/conda_envs/segformer/bin/python datasets/tools/extract_masks.py \
    --data_root data/waymo14/processed/validation --scene_ids "$@" \
    --segformer_path /autodl-fs/data/SegFormer --device cuda:0 --ignore_existing \
    >> "$LOG" 2>&1
rc=$?
echo "[group $G] 结束 rc=$rc $(date +%T)" >> "$LOG"
[ "$rc" -eq 0 ] && echo "OK" > "/tmp/fill_masks/seg_done_${G}" || echo "FAIL $rc" > "/tmp/fill_masks/seg_done_${G}"
