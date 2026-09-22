#!/usr/bin/env bash
# 给某个 waymo14 processed 场景补生成 DGGT 需要的标记图：
#   sky_masks/ + custom_masks/（SegFormer 语义）×动态掩码（从语义派生）
#
#   用法: bash tools_make_masks.sh 005 [005 007 ...]
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

SEG_PY=/root/autodl-tmp/conda_envs/segformer/bin/python
DGGT_PY=/root/autodl-tmp/conda_envs/dggt/bin/python
SEGFORMER=/autodl-fs/data/SegFormer
DATA_ROOT=data/waymo14/processed/validation
IDS=("$@")
if [ ${#IDS[@]} -eq 0 ]; then echo "用法: bash tools_make_masks.sh 005"; exit 2; fi
# 去掉前导 0（extract_masks 内部会 zfill(3)）
NUM=()
for i in "${IDS[@]}"; do NUM+=("$((10#$i))"); done

echo "== 1/2 SegFormer 语义掩码 (scene_ids: ${NUM[*]}) =="
PYTHONPATH="$SEGFORMER:${PYTHONPATH:-}" "$SEG_PY" datasets/tools/extract_masks.py \
    --data_root "$DATA_ROOT" --scene_ids "${NUM[@]}" \
    --segformer_path "$SEGFORMER" --device cuda:0 || exit 1

echo "== 2/2 派生动态掩码 =="
"$DGGT_PY" datasets/tools/derive_dynamic_masks.py \
    --data_root "$DATA_ROOT" --scene_ids "${NUM[@]}" || exit 1
echo "MASKS_DONE"
