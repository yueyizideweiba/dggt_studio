"""验证 `inference.py::_batch_scene_name`：从 image_paths / dataset.scenes 反推真实场景名。

为什么重要：上游用 `str(scene_idx).zfill(3)`（第几个 batch）当场景名去读 GT 位姿算
`scale_factor`，传入的场景名不是 `001` 时会用**别的段落**的位姿 —— 整个场景的米制尺度
全错，高精地图也就对不上。这个测试保证反推出来的名字是对的。

用法： python tools/verify_scene_name.py [场景号，默认 016]
"""
import os
import sys


def _repo_root() -> str:
    d = os.path.dirname(os.path.abspath(__file__))
    for _ in range(5):
        if os.path.exists(os.path.join(d, 'inference.py')):
            return d
        d = os.path.dirname(d)
    return os.getcwd()


ROOT = _repo_root()
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'studio', 'backend'))

from torch.utils.data import DataLoader  # noqa: E402

from inference import _batch_scene_name  # noqa: E402
from datasets.dataset import WaymoOpenDataset  # noqa: E402

names = [str(sys.argv[1] if len(sys.argv) > 1 else '016').zfill(3)]
ds = WaymoOpenDataset(os.path.join(ROOT, 'data/waymo14/processed/validation'), scene_names=names,
                      sequence_length=25, start_idx=0, mode=2, views=1, camera_ids=[0])
print('dataset.scenes =', ds.scenes, 'len =', len(ds))
dl = DataLoader(ds, batch_size=1, shuffle=False)
for k, batch in enumerate(dl):
    got = _batch_scene_name(batch, ds, k + 1)
    ip = batch['image_paths']
    while isinstance(ip, (list, tuple)):
        ip = ip[0]
    print('batch %d: image_paths[0]=%s -> scene_name=%r' % (k, ip, got))
    assert got == names[k], 'FAIL: %r != %r' % (got, names[k])
print('NAME_TEST PASS')
