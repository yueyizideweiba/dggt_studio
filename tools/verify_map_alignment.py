"""waymo_map.align_scene 的合成自检。

做法：拿一个**真场景**（只读，保留模型自己的相机朝向约定 —— 这是 `scene_up` 判据的来源），
人为构造一个"全局系"：把场景位置按某个已知相似变换 `W: 全局→场景` 的逆推回全局，
写成一个假的 processed 目录。然后要求 `align_scene` 把 `W` 还原出来。

只对位置做变换是刻意的：DGGT 导出的是模型自己预测的相机位姿，它的相机轴向约定和 WOD
的 `extrinsics` 不同（同一个相机的旋转差一个固定常数阵），所以对齐**只用位置**，
旋转只当诊断 —— 这个测试正好覆盖这一点。
"""
import json
import math
import os
import shutil
import sys
import tempfile

import numpy as np

def _repo_root() -> str:
    """向上找到含 studio/backend/waymo_map.py 的目录（脚本挪位置也不用改）。"""
    d = os.path.dirname(os.path.abspath(__file__))
    for _ in range(5):
        if os.path.exists(os.path.join(d, 'studio', 'backend', 'waymo_map.py')):
            return d
        d = os.path.dirname(d)
    return os.getcwd()


ROOT = _repo_root()
sys.path.insert(0, os.path.join(ROOT, 'studio/backend'))
import waymo_map as wm  # noqa: E402

SCENE = os.environ.get('WM_TEST_SCENE', os.path.join(ROOT, 'output/waymo_eval_14/016/016'))


def rot_y(deg):
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)


def scene_up_basis():
    """全局(上=+z) → 场景(上=-y)：e_z ↦ -e_y, e_x ↦ e_x, e_y ↦ e_z。"""
    return np.array([[1.0, 0.0, 0.0],
                     [0.0, 0.0, -1.0],
                     [0.0, 1.0, 0.0]], dtype=np.float64)


def main():
    ts, fids, mats, gf = wm._scene_cam_poses(SCENE)
    Ps = np.asarray([m[:3, 3] for m in mats], dtype=np.float64)
    n = len(Ps)
    print('真场景 =%s  帧数=%d' % (SCENE, n))

    # 已知的 全局→场景 相似变换 W（含 37° 绕场景"上"轴旋转）
    s_true = 1.0
    R_true = rot_y(37.0) @ scene_up_basis()
    t_true = np.array([12.0, -3.0, -40.0])

    # 由场景位置反推"全局"位置：Pg = W^{-1}(Ps)
    Pg = (np.linalg.inv(R_true) @ (Ps - t_true).T).T / s_true

    # 写假的 processed：extrinsics=I，ego_pose 直接放合成位姿
    tmp = tempfile.mkdtemp(prefix='wmselftest_')
    proc = os.path.join(tmp, 'proc')
    os.makedirs(os.path.join(proc, 'ego_pose'))
    os.makedirs(os.path.join(proc, 'extrinsics'))
    np.savetxt(os.path.join(proc, 'extrinsics', '0.txt'), np.eye(4), fmt='%.18e')
    for i in range(n):
        M = np.eye(4)
        M[:3, 3] = Pg[i]
        # 旋转随便给（对齐不该用到它）；用单位阵即可
        with open(os.path.join(proc, 'ego_pose', '%03d.txt' % i), 'w') as f:
            f.write('\n'.join(' '.join('%.18e' % v for v in row) for row in M) + '\n')

    al = wm.align_scene(SCENE, proc, cam=0)
    R_est = np.asarray(al['R'])
    ang = math.degrees(math.acos(float(np.clip((np.trace(R_true.T @ R_est) - 1) / 2, -1, 1))))
    print('source=%s  帧对=%d' % (al['source'], al['n_pairs']))
    print('s   真/估 = %.8f / %.8f' % (s_true, al['s']))
    print('rms       = %.3e m' % al['rms'])
    print('R 夹角    = %.3e°' % ang)
    print('t   真    = %s' % np.round(t_true, 6))
    print('t   估    = %s' % np.round(np.asarray(al['t']), 6))
    print('up_dot    = %+.6f (应=+1)' % al['up_dot'])
    ok = (abs(al['s'] - s_true) < 1e-6 and al['rms'] < 1e-5 and ang < 1e-4
          and np.allclose(np.asarray(al['t']), t_true, atol=1e-5)
          and al['up_dot'] > 0.9999)
    shutil.rmtree(tmp)
    print('SELFTEST', 'PASS' if ok else 'FAIL')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
