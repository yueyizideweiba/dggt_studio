#!/bin/bash
# 等标记图任务结束（/tmp/fill_masks_done 出现），然后把各场景齐备情况写成
# output/mask_status.json —— 这样本地可以经 /api/report 轮询它，不用一直连远端。
set -uo pipefail
cd /autodl-fs/data/dggt-main
for i in $(seq 1 1200); do
    [ -f /tmp/fill_masks_done ] && break
    sleep 30
done
/root/autodl-tmp/conda_envs/dggt/bin/python tools/mask_status.py \
    --out output/mask_status.json >> /tmp/fill_masks/status_writer.log 2>&1
echo "STATUS_WRITTEN $(date +%T)" >> /tmp/fill_masks/status_writer.log
