"""从 SegFormer 的 `custom_masks/` 派生 DGGT 需要的那几套标记图。

`datasets/tools/extract_masks.py` 在 `--process_dynamic_mask` 时要求额外的
`dynamic_masks/{human,vehicle}` 粗掩码（来自另一套 2D 检测器）。我们没有那一步，
但 SegFormer 的语义结果里本来就有 Vehicle / Person / Cyclist 类 —— 直接用它当"粗掩码"
就等价于 `valid = semantic ∧ rough`（rough 就是 semantic 本身）。

于是：
    dynamic_masks/vehicle|human              粗掩码（= 语义掩码）
    fine_dynamic_masks/vehicle|human|all      精细掩码（DGGT 的 dataset.py 只读 all）

用法：
    python datasets/tools/derive_dynamic_masks.py --data_root data/waymo14/processed/validation \
        --scene_ids 5 6 7
"""
from __future__ import annotations

import argparse
import os
import glob
import shutil

import numpy as np

try:
    import cv2
except Exception:  # noqa: BLE001
    cv2 = None
try:
    import imageio.v2 as imageio
except Exception:  # noqa: BLE001
    imageio = None

VEHICLE_VALUES = (40,)
HUMAN_VALUES = (50, 60)


def _read_mask(p):
    if imageio is not None:
        return np.asarray(imageio.imread(p))
    if cv2 is not None:
        return cv2.imread(p, cv2.IMREAD_GRAYSCALE)
    raise RuntimeError('需要 imageio 或 cv2 来读 png')


def _write_mask(p, m):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    arr = (np.asarray(m) > 0).astype(np.uint8) * 255
    if imageio is not None:
        imageio.imwrite(p, arr)
    else:
        cv2.imwrite(p, arr)


def main():
    ap = argparse.ArgumentParser(description='custom_masks → dynamic_masks/fine_dynamic_masks')
    ap.add_argument('--data_root', default='data/waymo14/processed/validation')
    ap.add_argument('--scene_ids', type=int, nargs='+', required=True)
    ap.add_argument('--copy_custom', action='store_true',
                    help='额外把 custom_masks 也复制成 fine_dynamic_masks/all（不推荐）')
    args = ap.parse_args()

    for sid in args.scene_ids:
        scene = os.path.join(args.data_root, str(sid).zfill(3))
        cm_dir = os.path.join(scene, 'custom_masks')
        if not os.path.isdir(cm_dir):
            print('!! %s 没有 custom_masks，跳过' % scene)
            continue
        rv = os.path.join(scene, 'dynamic_masks', 'vehicle')
        rh = os.path.join(scene, 'dynamic_masks', 'human')
        fv = os.path.join(scene, 'fine_dynamic_masks', 'vehicle')
        fh = os.path.join(scene, 'fine_dynamic_masks', 'human')
        fa = os.path.join(scene, 'fine_dynamic_masks', 'all')
        for d in (rv, rh, fv, fh, fa):
            os.makedirs(d, exist_ok=True)
        files = sorted(glob.glob(os.path.join(cm_dir, '*.png')))
        n = 0
        for f in files:
            base = os.path.splitext(os.path.basename(f))[0]
            m = _read_mask(f)
            veh = np.isin(m, VEHICLE_VALUES)
            hum = np.isin(m, HUMAN_VALUES)
            allm = np.logical_or(veh, hum)
            _write_mask(os.path.join(rv, base + '.png'), veh)
            _write_mask(os.path.join(rh, base + '.png'), hum)
            _write_mask(os.path.join(fv, base + '.png'), veh)
            _write_mask(os.path.join(fh, base + '.png'), hum)
            _write_mask(os.path.join(fa, base + '.png'), allm)
            n += 1
        nz = 0
        for f in sorted(glob.glob(os.path.join(fa, '*.png')))[:20]:
            if _read_mask(f).any():
                nz += 1
        print('scene %s: %d 帧 → fine_dynamic_masks/all（抽查 %d/20 帧有动态目标）'
              % (str(sid).zfill(3), n, nz), flush=True)
    print('DERIVE_DONE')


if __name__ == '__main__':
    main()
