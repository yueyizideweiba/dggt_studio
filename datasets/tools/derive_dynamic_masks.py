"""从 SegFormer 的 `custom_masks/` 派生 DGGT 需要的那几套标记图。

`datasets/tools/extract_masks.py` 在 `--process_dynamic_mask` 时要求额外的
`dynamic_masks/{human,vehicle}` 粗掩码（来自另一套 2D 检测器）。我们没有那一步，
但 SegFormer 的语义结果里本来就有 Vehicle / Person / Cyclist 类 —— 直接用语义掩码
当"粗掩码"就等价于 `valid = semantic ∧ rough`（rough 就是 semantic 本身）。

产出（与 005/017/018 一致的一套）：
    dynamic_masks/vehicle|human              粗掩码
    fine_dynamic_masks/vehicle|human          精细掩码（内容与粗掩码相同 → 用**硬链接**）
    fine_dynamic_masks/all                    两者之和（DGGT 的 dataset.py 只读这个）

inode 说明：`/autodl-fs/data` 有 20 万 inode 上限。粗/精细掩码内容完全一样，
所以 `fine_dynamic_masks/{vehicle,human}` 用硬链接指向 `dynamic_masks/` 里的文件，
每个场景省下约 1980 个 inode（25 个场景 ≈ 49,500 个）。

用法：
    python datasets/tools/derive_dynamic_masks.py --data_root data/waymo14/processed/validation \
        --scene_ids 5 6 7 [--workers 8] [--overwrite]
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

try:
    import cv2
except Exception:  # noqa: BLE001
    cv2 = None

VEHICLE_VALUES = (40,)
HUMAN_VALUES = (50, 60)


def _read_mask(p):
    if cv2 is not None:
        m = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if m is not None:
            return m
    try:
        import imageio.v2 as imageio
        return np.asarray(imageio.imread(p))
    except Exception:  # noqa: BLE001
        raise RuntimeError('需要 cv2 或 imageio 来读 png')


def _write_mask(p, m):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    arr = (np.asarray(m) > 0).astype(np.uint8) * 255
    if cv2 is not None:
        cv2.imwrite(p, arr)
    else:
        import imageio.v2 as imageio
        imageio.imwrite(p, arr)


def _link_or_copy(src: str, dst: str):
    """优先硬链接（同内容、0 额外 inode）；不行就退回复制。"""
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.exists(dst):
        try:
            if os.path.samefile(src, dst):
                return
        except OSError:
            pass
        os.remove(dst)
    try:
        os.link(src, dst)
    except OSError:
        import shutil
        shutil.copyfile(src, dst)


def _one(args):
    """处理单张语义掩码 → 写 3 个真实文件 + 2 个硬链接。"""
    f, rv, rh, fa, fv, fh, overwrite = args
    base = os.path.splitext(os.path.basename(f))[0]
    p_rv, p_rh, p_fa = (os.path.join(rv, base + '.png'), os.path.join(rh, base + '.png'),
                        os.path.join(fa, base + '.png'))
    if not overwrite and all(os.path.exists(p) for p in (p_rv, p_rh, p_fa)):
        return 0
    m = _read_mask(f)
    veh = np.isin(m, VEHICLE_VALUES)
    hum = np.isin(m, HUMAN_VALUES)
    _write_mask(p_rv, veh)
    _write_mask(p_rh, hum)
    _write_mask(p_fa, np.logical_or(veh, hum))
    _link_or_copy(p_rv, os.path.join(fv, base + '.png'))
    _link_or_copy(p_rh, os.path.join(fh, base + '.png'))
    return 1


def main():
    ap = argparse.ArgumentParser(description='custom_masks → dynamic_masks/fine_dynamic_masks')
    ap.add_argument('--data_root', default='data/waymo14/processed/validation')
    ap.add_argument('--scene_ids', type=int, nargs='*', default=None,
                    help='不填就是对 data_root 下所有场景')
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--overwrite', action='store_true')
    args = ap.parse_args()

    root = args.data_root
    scenes = ([str(s).zfill(3) for s in args.scene_ids] if args.scene_ids
              else sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))))

    for sid in scenes:
        scene = os.path.join(root, sid)
        cm_dir = os.path.join(scene, 'custom_masks')
        if not os.path.isdir(cm_dir):
            print('!! %s 没有 custom_masks，跳过（先跑 tools_make_masks.sh）' % sid, flush=True)
            continue
        paths = {
            'rv': os.path.join(scene, 'dynamic_masks', 'vehicle'),
            'rh': os.path.join(scene, 'dynamic_masks', 'human'),
            'fv': os.path.join(scene, 'fine_dynamic_masks', 'vehicle'),
            'fh': os.path.join(scene, 'fine_dynamic_masks', 'human'),
            'fa': os.path.join(scene, 'fine_dynamic_masks', 'all'),
        }
        for d in paths.values():
            os.makedirs(d, exist_ok=True)
        files = sorted(glob.glob(os.path.join(cm_dir, '*.png')))
        jobs = [(f, paths['rv'], paths['rh'], paths['fa'], paths['fv'], paths['fh'],
                 args.overwrite) for f in files]
        n = 0
        if args.workers > 1 and len(jobs) > 64:
            with ProcessPoolExecutor(max_workers=args.workers) as ex:
                for r in ex.map(_one, jobs, chunksize=16):
                    n += r
        else:
            for j in jobs:
                n += _one(j)
        nz = 0
        for f in sorted(glob.glob(os.path.join(paths['fa'], '*.png')))[:20]:
            if _read_mask(f).any():
                nz += 1
        print('scene %s: %d 帧（新写 %d）→ fine_dynamic_masks/all（抽查 %d/20 帧有动态目标）'
              % (sid, len(files), n, nz), flush=True)
    print('DERIVE_DONE')


if __name__ == '__main__':
    main()
