#!/bin/bash
# 等 3 个 SegFormer 分组 worker 都结束后，统一派生动态掩码（全部场景），并写完成标记。
# 独立进程，不依赖任何父脚本。
set -uo pipefail
cd /autodl-fs/data/dggt-main
rm -f /tmp/fill_masks_done
# 等 sentinel（最多 8 小时）
for i in $(seq 1 960); do
    ok=0
    for g in 0 1 2; do
        [ -f "/tmp/fill_masks/seg_done_${g}" ] && ok=$((ok+1))
    done
    [ "$ok" -eq 3 ] && break
    sleep 30
done
echo "[finalize] sentinels: $(cat /tmp/fill_masks/seg_done_* 2>/dev/null | tr '\n' ' ')" > /tmp/fill_masks/finalize.log
if grep -q FAIL /tmp/fill_masks/seg_done_* 2>/dev/null; then
    echo "SEG_FAIL" > /tmp/fill_masks_done
    exit 1
fi
echo "[finalize] 派生动态掩码（--minimal：只写 fine_dynamic_masks/all）$(date +%T)" >> /tmp/fill_masks/finalize.log
/root/autodl-tmp/conda_envs/dggt/bin/python datasets/tools/derive_dynamic_masks.py \
    --data_root data/waymo14/processed/validation --workers 8 --minimal >> /tmp/fill_masks/derive.log 2>&1
rc=$?
echo "[finalize] 派生 rc=$rc $(date +%T)" >> /tmp/fill_masks/finalize.log
if [ "$rc" -ne 0 ]; then echo "DERIVE_FAIL" > /tmp/fill_masks_done; exit 1; fi
echo "ALL_MASKS_DONE" > /tmp/fill_masks_done
echo "[finalize] 完成 $(date +%T)" >> /tmp/fill_masks/finalize.log
