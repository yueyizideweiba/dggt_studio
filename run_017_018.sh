#!/usr/bin/env bash
# 等 017/018 的标记图齐了，自动把两个场景重建出来。
# 用法: bash run_017_018.sh
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

DATA=data/waymo14/processed/validation
STAMP=/tmp/inf_017_018_done
rm -f "$STAMP"

need_ready() {
    for s in 017 018; do
        local a b
        a=$(ls "$DATA/$s/sky_masks" 2>/dev/null | wc -l)
        b=$(ls "$DATA/$s/fine_dynamic_masks/all" 2>/dev/null | wc -l)
        if [ "$a" -lt 990 ] || [ "$b" -lt 990 ]; then return 1; fi
    done
    return 0
}

echo "[watch] 等标记图……"
for i in $(seq 1 240); do            # 最多等 2 小时
    if need_ready; then echo "[watch] 标记图齐了"; break; fi
    sleep 30
done
if ! need_ready; then echo "[watch] 超时，标记图仍未齐"; echo "TIMEOUT" > "$STAMP"; exit 1; fi

for s in 17 18; do
    sid=$(printf '%03d' "$s")
    echo "================ 重建场景 $sid ================"
    ./run_inference.sh \
        --image_dir "$DATA" --scene_names "$s" --input_views 1 \
        --sequence_length 25 --start_idx 0 --mode 2 \
        --ckpt_path pretrained/model_latest_waymo.pt \
        --output_path "output/waymo_eval_14/$sid" -images -depth -metrics
    echo "[watch] 场景 $sid 退出码=$?"
done
echo "ALL_DONE" > "$STAMP"
echo "[watch] 全部完成"
