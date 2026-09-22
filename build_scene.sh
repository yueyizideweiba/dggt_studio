#!/usr/bin/env bash
# 一个场景从"processed 已就绪"到"能在 studio 里用"的一条命令：
#   ① 检查/补标记图（缺图会抛很难懂的 IndexError，这里提前查）
#   ② 跑重建（脚本自己把 ninja/nvcc 放进 PATH，否则首次渲染会 JIT 编译失败）
#   ③ 打印地图对齐验收指标（尺度/rms/自车到车道距离/质量闸门）
#
# 用法：
#   bash build_scene.sh 017                       # 默认 START_IDX=0 / SEQ=25
#   START_IDX=100 bash build_scene.sh 017         # 换起始帧
#   SEQ=25 START_IDX=150 bash build_scene.sh 013  # 换窗口
#   bash build_scene.sh 017 018 005               # 多个场景依次建
#
# 注意：processed（`data/waymo14/processed/validation/<NNN>/`）要先存在。
#       没有的话先跑 datasets/waymo14_preprocess.py。
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

DATA=data/waymo14/processed/validation
SEQ="${SEQ:-25}"
CKPT="${CKPT:-pretrained/model_latest_waymo.pt}"
DGGT_PY=/root/autodl-tmp/conda_envs/dggt/bin/python
SEG_PY=/root/autodl-tmp/conda_envs/segformer/bin/python
SEGFORMER="${SEGFORMER:-/autodl-fs/data/SegFormer}"

if [ $# -lt 1 ]; then sed -n '2,22p' "$0"; exit 2; fi

n_masks_missing=0
for arg in "$@"; do
    sid=$(printf '%03d' "$((10#$arg))")
    if [ ! -d "$DATA/$sid" ]; then
        echo "!! $DATA/$sid 不存在：先跑 datasets/waymo14_preprocess.py 把 tfrecord 解析出来"; exit 2
    fi
    sky=$(ls "$DATA/$sid/sky_masks" 2>/dev/null | wc -l)
    fine=$(ls "$DATA/$sid/fine_dynamic_masks/all" 2>/dev/null | wc -l)
    if [ "$sky" -lt 1 ] || [ "$fine" -lt 1 ]; then
        echo "[1/3] $sid 缺标记图（sky=$sky fine=$fine）→ 生成中（约 10 分钟/场景）"
        bash tools_make_masks.sh "$sid" || { echo "!! 补图失败"; exit 1; }
    else
        echo "[1/3] $sid 标记图齐全（sky=$sky fine=$fine）"
    fi
done

for arg in "$@"; do
    sid=$(printf '%03d' "$((10#$arg))")
    start="${START_IDX:-0}"
    echo
    echo "================ 重建场景 $sid (start_idx=$start seq=$SEQ) ================"
    ./run_inference.sh \
        --image_dir "$DATA" --scene_names "$((10#$arg))" --input_views 1 \
        --sequence_length "$SEQ" --start_idx "$start" --mode 2 \
        --ckpt_path "$CKPT" \
        --output_path "output/waymo_eval_14/$sid" -images
    rc=$?
    if [ "$rc" -ne 0 ]; then echo "!! 重建失败（退出码 $rc），看上面日志"; continue; fi

    echo "---------------- 地图对齐验收 ----------------"
    "$DGGT_PY" studio/backend/waymo_map.py \
        --scene "output/waymo_eval_14/$sid/$sid" --processed "$DATA/$sid" || true
done
echo
echo "BUILD_SCENE_DONE"
