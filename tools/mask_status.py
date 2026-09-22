"""汇总所有场景的标记图齐备情况，写成 JSON（给 `/api/report` 轮询用）。

用法：
    python tools/mask_status.py                 # 打印
    python tools/mask_status.py --out output/mask_status.json   # 落盘
"""
from __future__ import annotations

import argparse
import json
import os
import sys


def count(d):
    try:
        return len([f for f in os.listdir(d) if not f.startswith('.')])
    except Exception:  # noqa: BLE001
        return 0


def collect(root):
    subs = ['sky_masks', 'custom_masks',
            os.path.join('dynamic_masks', 'vehicle'), os.path.join('dynamic_masks', 'human'),
            os.path.join('fine_dynamic_masks', 'all'),
            os.path.join('fine_dynamic_masks', 'vehicle'),
            os.path.join('fine_dynamic_masks', 'human')]
    out = {}
    for s in sorted(os.listdir(root)):
        p = os.path.join(root, s)
        if not os.path.isdir(p) or not os.path.isdir(os.path.join(p, 'images')):
            continue
        n = count(os.path.join(p, 'images'))
        c = {k: count(os.path.join(p, k)) for k in subs}
        out[s] = {'images': n, **c, 'ok': all(v == n for v in c.values())}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_root', default='data/waymo14/processed/validation')
    ap.add_argument('--out', default=None)
    ap.add_argument('--stamp', default='/tmp/fill_masks_done')
    args = ap.parse_args()
    stamp = ''
    try:
        stamp = open(args.stamp).read().strip()
    except Exception:  # noqa: BLE001
        stamp = 'RUNNING'
    scenes = collect(args.data_root)
    incomplete = [k for k, v in scenes.items() if not v['ok']]
    doc = {'stamp': stamp, 'n_scenes': len(scenes), 'n_incomplete': len(incomplete),
           'incomplete': incomplete, 'scenes': scenes}
    txt = json.dumps(doc, ensure_ascii=False, indent=1)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)) or '.', exist_ok=True)
        tmp = args.out + '.tmp'
        with open(tmp, 'w', encoding='utf-8') as f:
            f.write(txt)
        os.replace(tmp, args.out)
    print(txt if not args.out else 'wrote %s（不完整场景 %d/%d）'
          % (args.out, len(incomplete), len(scenes)))
    return 0


if __name__ == '__main__':
    sys.exit(main())
