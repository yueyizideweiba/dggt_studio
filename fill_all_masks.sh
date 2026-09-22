#!/usr/bin/env bash
# 给 processed 下**所有**场景补齐标记图，使每个场景都具备和 005/017/018 一样的内容：
#     sky_masks/                      SegFormer 语义 → 天空
#     custom_masks/                   SegFormer 语义 → 类别图
#     dynamic_masks/{vehicle,human}   动态掩码（本流程用语义掩码本身当"粗掩码"）
#     fine_dynamic_masks/{all,human,vehicle}
#
# SegFormer 部分按场景分 GROUP 份**并行**跑（每份一个进程，B5 推理约 3~4GB 显存）；
# 跑完再统一做一次派生（多进程 + 硬链接，省 inode）。
#
# 用法：
#   bash fill_all_masks.sh                 # 只补缺的
#   bash fill_all_masks.sh --overwrite     # 全部重做
#   GROUPS=2 bash fill_all_masks.sh        # 并行份数
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

DATA="${DATA:-data/waymo14/processed/validation}"
SEG_PY="${SEG_PY:-/root/autodl-tmp/conda_envs/segformer/bin/python}"
DGGT_PY="${DGGT_PY:-/root/autodl-tmp/conda_envs/dggt/bin/python}"
SEGFORMER="${SEGFORMER:-/autodl-fs/data/SegFormer}"
# 注意：不能叫 GROUPS —— 那是 bash 的内置只读变量（当前用户的组 ID 列表），
# 赋值会静默失败，导致分组循环一次都不执行，而 fail=0 又把它伪装成"成功"。
N_GROUPS="${N_GROUPS:-3}"
# 缺省只补缺口（extract_masks 的 --ignore_existing = 存在就跳过）；--overwrite 时全部重做。
if [ "${1:-}" = "--overwrite" ]; then
    OVERWRITE=""
else
    OVERWRITE="--ignore_existing"
fi

LOGDIR=/tmp/fill_masks
mkdir -p "$LOGDIR"
# 3 个进程**同时**初始化 CUDA 会互相卡死（实测：都卡在 init_segmentor，显存不涨、无产出）；
# 错开启动就正常共存。这里每起一组等一会儿。
STAGGER="${STAGGER:-60}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

# ---- 找出还没齐的场景 ----
need=()
for d in "$DATA"/*/; do
    s=$(basename "$d")
    n=$(ls "$d/images" 2>/dev/null | wc -l)
    sky=$(ls "$d/sky_masks" 2>/dev/null | wc -l)
    cus=$(ls "$d/custom_masks" 2>/dev/null | wc -l)
    if [ "$n" -eq 0 ]; then continue; fi
    if [ "$sky" -lt "$n" ] || [ "$cus" -lt "$n" ]; then need+=("$((10#$s))"); fi
done
echo "[fill] 需要补语义掩码的场景 ${#need[@]} 个: ${need[*]:-无}"

if [ ${#need[@]} -gt 0 ]; then
    # ---- 分成 N_GROUPS 组，并行 ----
    pids=()
    for g in $(seq 0 $((N_GROUPS-1))); do
        grp=()
        for i in "${!need[@]}"; do
            if [ $((i % N_GROUPS)) -eq "$g" ]; then grp+=("${need[$i]}"); fi
        done
        [ ${#grp[@]} -eq 0 ] && continue
        echo "[fill] 组 $g (${#grp[@]} 个): ${grp[*]}"
        PYTHONPATH="$SEGFORMER:${PYTHONPATH:-}" "$SEG_PY" datasets/tools/extract_masks.py \
            --data_root "$DATA" --scene_ids "${grp[@]}" \
            --segformer_path "$SEGFORMER" --device cuda:0 $OVERWRITE \
            > "$LOGDIR/seg_group$g.log" 2>&1 &
        pids+=($!)
        # 错开启动，避免多个进程同时初始化 CUDA 上下文时互相卡死
        if [ "$g" -lt "$((N_GROUPS-1))" ]; then sleep "$STAGGER"; fi
    done
    # 守卫：一组都没起来的话 fail=0 会骗人（曾经被 bash 内置 GROUPS 变量坑过）
    if [ ${#pids[@]} -eq 0 ]; then
        echo "[fill] 致命：一組 SegFormer 都没启动（检查 N_GROUPS / seq）"
        echo "NO_JOB_STARTED" > /tmp/fill_masks_done
        exit 1
    fi
    fail=0
    for p in "${pids[@]}"; do wait "$p" || fail=1; done
    echo "[fill] SegFormer 结束（启动了 ${#pids[@]} 组）fail=$fail"
    [ "$fail" -ne 0 ] && { echo "[fill] 有分组失败，看 $LOGDIR/seg_group*.log"; echo "SEG_FAIL" > /tmp/fill_masks_done; exit 1; }
fi

# ---- 派生动态掩码（全部场景，保证 016 那种只剩一半的也补齐）----
echo "[fill] 派生动态掩码……"
"$DGGT_PY" datasets/tools/derive_dynamic_masks.py --data_root "$DATA" --workers 8 \
    > "$LOGDIR/derive.log" 2>&1 || { echo "[fill] 派生失败"; tail -20 "$LOGDIR/derive.log"; echo "DERIVE_FAIL" > /tmp/fill_masks_done; exit 1; }

echo "ALL_MASKS_DONE" > /tmp/fill_masks_done
echo "[fill] 完成"
