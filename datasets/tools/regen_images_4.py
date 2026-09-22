"""重新生成 processed 目录里的 `images_4/`（1/4 下采样图）。

为什么会有这个脚本：`images_4/` 在 DGGT 的推理/训练里**没有被任何代码读取**
（`datasets/dataset.py` 只读 `images/`），而它占掉每个场景约 990 个 inode。
`/autodl-fs/data` 有 20 万 inode 上限，为了给 25 个场景都补上标记图，
这些 inode 被腾出来了。需要它的时候用本脚本按需重建。

用法：
    python datasets/tools/regen_images_4.py                       # 全量补（缺的才补）
    python datasets/tools/regen_images_4.py --scenes 005 017      # 指定场景
    python datasets/tools/regen_images_4.py --overwrite           # 覆盖重建
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np

try:
    import cv2
except Exception:  # noqa: BLE001
    cv2 = None


def _resize(img, scale=4):
    h, w = img.shape[:2]
    return cv2.resize(img, (max(1, w // scale), max(1, h // scale)), interpolation=cv2.INTER_AREA)


def main():
    ap = argparse.ArgumentParser(description='重建 images_4/（1/4 下采样）')
    ap.add_argument('--data_root', default='data/waymo14/processed/validation')
    ap.add_argument('--scenes', nargs='*', default=None)
    ap.add_argument('--overwrite', action='store_true')
    args = ap.parse_args()
    if cv2 is None:
        print('需要 cv2：pip install opencv-python'); return 1

    root = args.data_root
    scenes = ([str(s).zfill(3) for s in args.scenes] if args.scenes
              else sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))))
    total = 0
    for s in scenes:
        src = os.path.join(root, s, 'images')
        dst = os.path.join(root, s, 'images_4')
        if not os.path.isdir(src):
            print('%s 没有 images/，跳过' % s); continue
        os.makedirs(dst, exist_ok=True)
        n = 0
        for f in sorted(glob.glob(os.path.join(src, '*.jpg'))):
            out = os.path.join(dst, os.path.basename(f))
            if os.path.exists(out) and not args.overwrite:
                continue
            img = cv2.imread(f)
            if img is None:
                continue
            cv2.imwrite(out, _resize(img, 4), [cv2.IMWRITE_JPEG_QUALITY, 90])
            n += 1
        total += n
        print('scene %s: 生成 %d 张' % (s, n), flush=True)
    print('完成，共 %d 张' % total)
    return 0


if __name__ == '__main__':
    sys.exit(main())
