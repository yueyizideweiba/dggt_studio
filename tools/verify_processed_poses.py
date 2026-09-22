"""直接从 waymo14 tfrecord 读 `frame.pose`，和 processed/ego_pose 逐位对比，
确认 `datasets/waymo14_preprocess.py` 解析出来的位姿是对的（数据正确性回归测试）。

用法： python tools/verify_processed_poses.py [场景号，默认 000] [segment 名，可选]
"""
import glob
import os
import sys

import numpy as np


def _repo_root() -> str:
    d = os.path.dirname(os.path.abspath(__file__))
    for _ in range(5):
        if os.path.exists(os.path.join(d, 'datasets', 'waymo14_preprocess.py')):
            return d
        d = os.path.dirname(d)
    return os.getcwd()


ROOT = _repo_root()
RAW = os.environ.get('WAYMO14_RAW', os.path.join(ROOT, 'waymo', 'waymo14'))
PROC = os.environ.get('WAYMO14_PROC', os.path.join(ROOT, 'data/waymo14/processed/validation'))


def main():
    import tensorflow as tf   # 只在真正跑的时候才 import（避免没装 TF 时连 --help 都跑不了）
    idx = sys.argv[1] if len(sys.argv) > 1 else '000'
    seg = sys.argv[2] if len(sys.argv) > 2 else None
    if seg is None:
        import json
        seg = json.load(open(os.path.join(PROC, idx, 'map', 'meta.json')))['segment']
    path = None
    for f in os.listdir(RAW):
        if seg in f:
            path = os.path.join(RAW, f)
            break
    print('idx=%s segment=%s\nfile=%s' % (idx, seg, path))
    raw = []
    ds = tf.data.TFRecordDataset([path], compression_type='')
    from waymo_open_dataset import dataset_pb2 as wod
    for i, rec in enumerate(ds):
        fr = wod.Frame()
        fr.ParseFromString(rec.numpy())
        raw.append((fr.timestamp_micros, np.asarray(fr.pose.transform, dtype=np.float64).reshape(4, 4)[:3, 3],
                    len(fr.map_features)))
    print('raw frames =', len(raw))
    R = np.asarray([r[1] for r in raw])
    st = np.linalg.norm(np.diff(R, axis=0), axis=1)
    print('raw   path=%.1f med=%.3f zero=%d' % (st.sum(), np.median(st), (st < 0.01).sum()))
    print('raw   first 12 steps:', np.round(st[:12], 3))
    P = np.asarray([np.loadtxt(f)[:3, 3] for f in sorted(glob.glob(os.path.join(PROC, idx, 'ego_pose', '*.txt')))])
    st2 = np.linalg.norm(np.diff(P, axis=0), axis=1)
    print('proc  path=%.1f med=%.3f zero=%d  n=%d' % (st2.sum(), np.median(st2), (st2 < 0.01).sum(), len(P)))
    print('proc  first 12 steps:', np.round(st2[:12], 3))
    n = min(len(R), len(P))
    print('max |raw-proc| = %.3e' % np.abs(R[:n] - P[:n]).max())
    ts = np.asarray([r[0] for r in raw], dtype=np.float64) / 1e6
    print('raw dt: median %.5f min %.5f max %.5f' % (np.median(np.diff(ts)), np.diff(ts).min(), np.diff(ts).max()))
    print('map_features per frame (first 6):', [r[2] for r in raw[:6]])
    print('DIAG_DONE')


if __name__ == '__main__':
    main()
