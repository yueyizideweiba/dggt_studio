"""把 Waymo v1.4 地图（`data/waymo14/processed/validation/<NNN>/map/`）接进 studio 场景。

## 为什么能精确对齐

DGGT 重建出来的场景世界系（"场景系"）是模型内部归一化坐标乘 `scale_factor` 的产物，
它和 Waymo 全局系之间只差一个**固定的 3D 相似变换** `(s, R, t)`：`p_scene = s·R·p_global + t`。
两边都有同一个刚体（自车相机）的观测，所以可以闭式解出来，不需要人工标定：

  processed 侧（米制、全局系）：`T_g_i = ego_pose[frame_ids[i]] @ extrinsics[cam]`
  scene     侧（米制、场景系）：`T_s_i = camera_extrinsics_world`（inference.py 导出）

  · 旋转：`R = orth(mean_i R_s_i · R_g_i^T)`（跨帧应当一致，散度就是质量指标）
  · 尺度/平移：`p_s = s·R·p_g + t` 最小二乘
  · 残差 rms 直接反映对齐质量；两边都已米制，所以 `s` 应当 ≈ 1

`scene_meta.json`（inference.py 导出）给出 `frame_ids`，因此帧对应关系是**确定**的；
老场景没有 meta 时才退化成搜索 `(start, stride)`。

## 对外接口

  load_scene_map(scene_dir, proc_dir=None) -> dict          # 带缓存
      {'align': {...}, 'lanes': {id: {...}}, 'polylines': [...], 'anchors': [...],
       'junctions': [...], 'counts': {...}, 'meta': {...}}
  to_scene(map, global_pts) -> Nx3                # 全局系 → 场景系
  nearest_lane(map, xz) -> (lane_id, dist_m)
  lane_heading(map, lane_id, xz) -> (dx, dz)      # 车道在该点处的前进方向（场景系）
  successors(map, lane_id) / predecessors(map, lane_id)
  lane_polyline(map, lane_id) -> Nx3
  route_between(map, from_lane, to_lane, max_hops) -> [lane_id] 或 None
  on_road(map, xz, tol) -> dict                    # 道路约束检查
  anchors_near(map, xz, radius) -> [...]

命令行（对齐诊断 + 俯视叠图）：
    python studio/backend/waymo_map.py --scene output/waymo_eval_14/016 \
        --processed data/waymo14/processed/validation/016 --overlay /tmp/map016.png
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_PROC = os.path.join(ROOT, 'data/waymo14/processed/validation')

# 车道中心线左右各留多少米算"在路上"（Waymo 车道宽约 3.5m）
LANE_HALF_WIDTH = 1.9
# 道路约束给轨迹点留的横向容差
ROAD_TOL_M = 3.5

_CACHE: Dict[str, Dict[str, Any]] = {}


# ---------------------------------------------------------------- 基础 IO

def _load_json(p: str):
    with open(p) as f:
        return json.load(f)


def _load_matrix(p: str) -> np.ndarray:
    return np.loadtxt(p).reshape(4, 4)


def clear_cache():
    _CACHE.clear()


# ---------------------------------------------------------------- 对齐

def resolve_processed_dir(scene_dir: str, proc_dir: Optional[str] = None) -> Optional[str]:
    """找出场景对应的 processed 目录（优先 scene_meta.json，再按目录名猜）。"""
    if proc_dir:
        return proc_dir if os.path.isdir(proc_dir) else None
    meta_p = os.path.join(scene_dir, 'scene_meta.json')
    if os.path.exists(meta_p):
        try:
            mt = _load_json(meta_p)
            img_dir = mt.get('image_dir')
            name = mt.get('scene_name')
            if img_dir and name:
                cand = os.path.join(img_dir, str(name))
                if os.path.isdir(cand):
                    return cand
        except Exception:  # noqa: BLE001
            pass
    base = os.path.basename(os.path.normpath(scene_dir))
    parent = os.path.basename(os.path.dirname(os.path.normpath(scene_dir)))
    for c in (base, parent):
        if c and os.path.isdir(os.path.join(DEFAULT_PROC, c)):
            return os.path.join(DEFAULT_PROC, c)
    return None


def _scene_cam_poses(scene_dir: str) -> Tuple[List[int], List[int], List[np.ndarray], List[int]]:
    """返回 (scene 帧序号, frame_id, camera_extrinsics_world 列表, global_frame 列表(可能全 -1))。"""
    ts, fids, mats, gfids = [], [], [], []
    for j in sorted(glob.glob(os.path.join(scene_dir, 'ego_pose', '*_ego.json'))):
        try:
            d = _load_json(j)
        except Exception:  # noqa: BLE001
            continue
        cam = d.get('camera_extrinsics_world')
        if cam is None:
            continue
        cam = np.asarray(cam, dtype=np.float64)
        if cam.shape == (3, 4):
            cam = np.vstack([cam, np.array([[0.0, 0.0, 0.0, 1.0]])])
        ts.append(len(ts))
        fids.append(int(d.get('frame_id', len(ts) - 1)))
        gfids.append(int(d['global_frame']) if d.get('global_frame') is not None else -1)
        mats.append(cam)
    return ts, fids, mats, gfids


def _proc_cam_pose(proc_dir: str, frame_idx: int, cam: int = 0) -> Optional[np.ndarray]:
    """全局系下的相机位姿 = ego_pose(frame) @ extrinsics(cam)。"""
    ep = os.path.join(proc_dir, 'ego_pose', '%03d.txt' % int(frame_idx))
    epp = os.path.join(proc_dir, 'extrinsics', '%d.txt' % int(cam))
    if not os.path.exists(ep) or not os.path.exists(epp):
        return None
    try:
        return _load_matrix(ep) @ _load_matrix(epp)
    except Exception:  # noqa: BLE001
        return None


def _orth(R: np.ndarray) -> np.ndarray:
    U, _, Vt = np.linalg.svd(R)
    D = np.eye(3)
    D[-1, -1] = np.sign(np.linalg.det(U @ Vt))
    return U @ D @ Vt


def _kabsch(P: np.ndarray, Q: np.ndarray):
    """已知对应点求相似变换 `q = s·R·p + t`（R 为真旋转）。返回 (s, R, t, rms)。"""
    Pc, Qc = P.mean(0), Q.mean(0)
    P0, Q0 = P - Pc, Q - Qc
    H = P0.T @ Q0
    U, sv, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1
        R = Vt.T @ U.T
    s = float(sv.sum() / max(1e-12, (P0 ** 2).sum()))
    t = Qc - s * (R @ Pc)
    pred = s * (R @ P.T).T + t
    rms = float(np.sqrt(np.mean(np.sum((pred - Q) ** 2, axis=1))))
    return s, R, t, rms


def scene_up_vector(mats, up_sign: Optional[float] = None):
    """场景系的"上"方向。

    DGGT 场景里世界 +y 可能是"下"（`up_sign=-1`）。判据：相机图像"下"方向
    （c2w 的第 2 列）在场景 y 上的分量 >0 → +y 向下。
    """
    if up_sign is None:
        ys = [float(np.asarray(m, dtype=np.float64)[1, 1]) for m in mats]
        med = float(np.median(ys)) if ys else -1.0
        up_sign = -1.0 if med > 0 else 1.0
    up = np.array([0.0, -1.0, 0.0]) if up_sign < 0 else np.array([0.0, 1.0, 0.0])
    return up, float(up_sign)


def _frame_mapping(scene_dir: str, proc_dir: str, cam: int = 0):
    """确定场景帧 ↔ processed 帧的对应关系。

    返回 (global_frame_ids, source)。source ∈ {'scene_meta','json','identity','search'}
    """
    ts, fids, mats, gfids = _scene_cam_poses(scene_dir)
    n = len(ts)
    meta_p = os.path.join(scene_dir, 'scene_meta.json')
    if os.path.exists(meta_p):
        try:
            mt = _load_json(meta_p)
            fids_meta = mt.get('frame_ids')
            if fids_meta and len(fids_meta) >= n:
                if all(os.path.exists(os.path.join(proc_dir, 'ego_pose', '%03d.txt' % int(f)))
                       for f in fids_meta[:n]):
                    return [int(f) for f in fids_meta[:n]], 'scene_meta'
        except Exception:  # noqa: BLE001
            pass
    if all(g >= 0 for g in gfids):
        return list(gfids), 'json'
    n_proc = len([f for f in os.listdir(os.path.join(proc_dir, 'ego_pose')) if f.endswith('.txt')])
    # 老场景：frame_id 是场景内序号，先试"从 0 起逐帧"
    if fids == list(range(n)) and n <= n_proc:
        return list(range(n)), 'identity'
    best, best_rms = None, None
    for stride in (1, 2, 4, 5, 8, 10):
        for start in range(0, min(stride, 8)):
            idx = [start + i * stride for i in range(n)]
            if idx[-1] >= n_proc:
                continue
            r = _fit_alignment(scene_dir, proc_dir, idx, cam)
            if r is None:
                continue
            if best_rms is None or r['rms'] < best_rms:
                best, best_rms = idx, r['rms']
    if best is None:
        raise RuntimeError('无法确定 %s ↔ %s 的帧对应关系' % (scene_dir, proc_dir))
    return best, 'search'


def _fit_alignment(scene_dir: str, proc_dir: str, frame_ids: Sequence[int],
                   cam: int = 0, lane_probe: Optional[List[np.ndarray]] = None,
                   verbose: bool = False) -> Optional[Dict[str, Any]]:
    """用已知帧对应关系闭式解相似变换 `p_scene = s·R·p_global + t`。

    **只对位置做 Procrustes**：场景里相机的"位置"是准的（轨迹形状与 GT 逐点吻合，
    残差厘米级），但模型的相机"朝向"预测不可靠 —— 我们用实测验证过，用旋转去定 R 会
    偏 100° 以上，所以旋转只当诊断指标，不参与求解。

    轨迹近似共面（地面上的一条曲线），所以 Procrustes 会给出两个残差几乎一样的候选
    （互为镜像）。用**场景系的"上"方向**消歧：全局 +z(上) 必须映到场景的"上"。
    再用"自车应当贴着车道中心线走"做交叉验证（`lane_probe`，全局系车道折线）。
    """
    ts, fids, mats, gfids = _scene_cam_poses(scene_dir)
    n = min(len(ts), len(frame_ids))
    if n < 2:
        return None
    Ps, Pg, Rg, Rs, used = [], [], [], [], []
    for i in range(n):
        Tg = _proc_cam_pose(proc_dir, int(frame_ids[i]), cam)
        if Tg is None:
            continue
        Ps.append(mats[i][:3, 3])
        Pg.append(Tg[:3, 3])
        Rs.append(mats[i][:3, :3])
        Rg.append(Tg[:3, :3])
        used.append(int(frame_ids[i]))
    if len(Ps) < 2:
        return None
    Ps = np.asarray(Ps, dtype=np.float64)
    Pg = np.asarray(Pg, dtype=np.float64)
    scene_up, up_sign = scene_up_vector(mats)
    up_y = float(scene_up[1])          # 全局"上"(+z) 映到场景的 y 分量：-1(下为+y 时) 或 +1

    # ------------------------------------------------------------------
    # 分两块解，避免"轨迹近似共线 → 绕轨迹轴的旋转解不出来"的镜像歧义：
    #   水平面：全局 (x,y) → 场景 (x,z)，2D 相似变换（唯一）
    #   竖直：  全局 +z(上) → 场景的"上"方向，斜率固定为 ±s（各向同性缩放）
    # 手性由场景"上"方向唯一确定：要让整体 det(R)=+1，2D 那块的 det 必须 = -up_y。
    # ------------------------------------------------------------------
    P2, Q2 = Pg[:, [0, 1]], Ps[:, [0, 2]]
    Pc, Qc = P2.mean(0), Q2.mean(0)
    P0, Q0 = P2 - Pc, Q2 - Qc
    H = P0.T @ Q0
    U, sv, Vt = np.linalg.svd(H)
    M = Vt.T @ U.T
    if np.linalg.det(M) * (-up_y) < 0:     # 强制 det(M) = -up_y
        Vt[-1] *= -1
        M = Vt.T @ U.T
    s = float(sv.sum() / max(1e-12, (P0 ** 2).sum()))
    t2 = Qc - s * (M @ Pc)
    c_y = float(np.mean(Ps[:, 1]) - s * up_y * np.mean(Pg[:, 2]))
    R = np.array([[M[0, 0], M[0, 1], 0.0],
                  [0.0, 0.0, up_y],
                  [M[1, 0], M[1, 1], 0.0]], dtype=np.float64)
    t = np.array([t2[0], c_y, t2[1]], dtype=np.float64)
    pred = s * (R @ Pg.T).T + t
    rms = float(np.sqrt(np.mean(np.sum((pred - Ps) ** 2, axis=1))))
    u = R @ np.array([0.0, 0.0, 1.0])
    best = {'flip': float(np.sign(np.linalg.det(M))), 's': s, 'R': R, 't': t, 'rms': rms,
            'up_dot': float(u @ scene_up), 'up_vec': u.tolist()}

    # 候选（仅诊断/兜底）：水平手性反过来 + 竖直反过来，保持 det(R)=+1
    cands = [dict(best)]
    M2 = M @ np.array([[1.0, 0.0], [0.0, -1.0]])
    s2 = float(sv.sum() / max(1e-12, (P0 ** 2).sum()))
    t22 = Qc - s2 * (M2 @ Pc)
    R2 = np.array([[M2[0, 0], M2[0, 1], 0.0],
                   [0.0, 0.0, -up_y],          # 手性反过来 → 竖直也反过来才能保持 det=+1
                   [M2[1, 0], M2[1, 1], 0.0]], dtype=np.float64)
    t2b = np.array([t22[0], float(np.mean(Ps[:, 1]) + s2 * up_y * np.mean(Pg[:, 2])), t22[1]])
    pred2 = s2 * (R2 @ Pg.T).T + t2b
    rms2 = float(np.sqrt(np.mean(np.sum((pred2 - Ps) ** 2, axis=1))))
    u2 = R2 @ np.array([0.0, 0.0, 1.0])
    cands.append({'flip': float(np.sign(np.linalg.det(M2))), 's': s2, 'R': R2, 't': t2b, 'rms': rms2,
                  'up_dot': float(u2 @ scene_up), 'up_vec': u2.tolist()})

    # 自车→车道交叉验证（两边都先变换到场景系再比距离）
    if lane_probe:
        for c in cands:
            lanes_sc = [c['s'] * (c['R'] @ P.T).T + c['t'] for P in lane_probe]
            step = max(1, len(Pg) // 8)
            c['ego_lane_med_m'] = float(np.median(
                [_scene_point_lane_min_dist(c['s'] * (c['R'] @ p) + c['t'], lanes_sc)
                 for p in Pg[::step]]))

    warn = None
    if best['up_dot'] < 0.5:
        warn = '场景"上"方向与地图不一致（相机外参可能异常），对齐可能上下颠倒'
    if lane_probe and best.get('ego_lane_med_m', 0.0) > 2.0:
        warn = ((warn + '；') if warn else '') + '自车离最近车道中心线 %.2fm，地图对齐可疑' % best['ego_lane_med_m']

    # 旋转一致性只做诊断
    Rrel = np.einsum('nij,nkj->nik', np.asarray(Rs), np.asarray(Rg))
    Rm = _orth(Rrel.mean(axis=0))
    spread = []
    for Ri in Rrel:
        cc = (np.trace(Ri.T @ Rm) - 1.0) / 2.0
        spread.append(math.degrees(math.acos(float(np.clip(cc, -1.0, 1.0)))))
    out = {
        's': float(best['s']), 'R': best['R'].tolist(), 't': best['t'].tolist(),
        'rms': float(best['rms']), 'flip': float(best['flip']),
        'up_dot': float(best['up_dot']), 'up_sign': float(up_sign),
        'ego_lane_med_m': best.get('ego_lane_med_m'),
        'mirror_ambiguous': bool(len(cands) == 2 and abs(cands[0]['rms'] - cands[1]['rms']) < 0.15),
        'n_pairs': len(used), 'frame_ids': used, 'cam': int(cam),
        # 诊断：模型相机朝向的跨帧一致性（不参与求解）
        'rot_spread_deg': float(max(spread)) if spread else None,
        'rot_spread_med_deg': float(np.median(spread)) if spread else None,
        'candidates': [{k: (float(v) if v is not None and not isinstance(v, list) else None)
                        for k, v in c.items() if k in ('flip', 'rms', 'up_dot', 'ego_lane_med_m')}
                       for c in cands],
        'warning': warn,
    }
    if verbose:
        for c in cands:
            print('  候选 flip=%+.0f rms=%.3fm s=%.4f up_dot=%+.3f ego_lane=%.2fm' % (
                c['flip'], c['rms'], c['s'], c['up_dot'], c.get('ego_lane_med_m', -1)))
    return out


def _scene_point_lane_min_dist(p_scene, lanes_scene) -> float:
    """场景系点 → 最近车道中心线的水平距离。场景系水平面是 (x, z)。"""
    q = np.asarray(p_scene, dtype=np.float64).reshape(3)
    best = float('inf')
    for P in lanes_scene:
        d = float(np.min(np.linalg.norm(P[:, [0, 2]] - q[[0, 2]], axis=1)))
        if d < best:
            best = d
    return best


def _apply_sim(p_g, s, R, t, flip: float) -> np.ndarray:
    """全局系点 → 场景系：`q = s·R·(p ⊙ [1,flip,1]) + t`。"""
    p = np.asarray(p_g, dtype=np.float64).reshape(-1, 3) * np.array([1.0, flip, 1.0])
    return s * (R @ p.T).T + np.asarray(t, dtype=np.float64).reshape(3)


def align_scene(scene_dir: str, proc_dir: str, cam: int = 0,
                lane_probe: Optional[List[np.ndarray]] = None) -> Dict[str, Any]:
    scene_dir = os.path.abspath(scene_dir)
    frame_ids, source = _frame_mapping(scene_dir, proc_dir, cam)
    al = _fit_alignment(scene_dir, proc_dir, frame_ids, cam, lane_probe=lane_probe)
    if al is None:
        raise RuntimeError('对齐失败：%s ↔ %s' % (scene_dir, proc_dir))
    al['source'] = source
    al['processed_dir'] = proc_dir
    al['scale_ok'] = bool(0.85 <= al['s'] <= 1.18)
    Rm = np.asarray(al['R'], dtype=np.float64)
    fwd = Rm @ np.array([1.0, 0.0, 0.0])   # 全局 x 轴在场景系的像
    al['global_x_in_scene'] = fwd.tolist()
    al['theta_deg'] = float(math.degrees(math.atan2(fwd[0], fwd[2])))
    return al


def to_scene(align: Dict[str, Any], pts) -> np.ndarray:
    """全局系点 → 场景系：`q = s·R·p + t`（R 已包含"上"方向和手性）。"""
    p = np.asarray(pts, dtype=np.float64).reshape(-1, 3)
    R = np.asarray(align['R'], dtype=np.float64).reshape(3, 3)
    t = np.asarray(align['t'], dtype=np.float64).reshape(3)
    return float(align['s']) * (R @ p.T).T + t


# ---------------------------------------------------------------- 装载地图

def _polyline_of(f: Dict[str, Any]):
    for k in ('polyline', 'polygon'):
        if f.get(k):
            return k, f[k]
    return None, None


def load_scene_map(scene_dir: str, proc_dir: Optional[str] = None,
                   cam: int = 0, use_cache: bool = True) -> Dict[str, Any]:
    """读 processed 地图 → 变换到场景系；返回带查询用索引的结构。"""
    scene_dir = os.path.abspath(scene_dir)
    proc = resolve_processed_dir(scene_dir, proc_dir)
    if not proc:
        raise RuntimeError('找不到场景对应的 processed 目录（scene=%s）' % scene_dir)
    # 用哪台相机做对齐：inference 导出时写进 scene_meta 的 camera_ids
    if cam == 0:
        try:
            sm = _load_json(os.path.join(scene_dir, 'scene_meta.json'))
            cids = sm.get('camera_ids') or []
            if cids:
                cam = int(cids[0])
        except Exception:  # noqa: BLE001
            pass
    key = '%s|%s|%d' % (scene_dir, proc, cam)
    if use_cache and key in _CACHE:
        return _CACHE[key]

    mf = os.path.join(proc, 'map', 'map_features.json')
    if not os.path.exists(mf):
        raise RuntimeError('该 processed 目录没有地图：%s' % mf)
    map_d = _load_json(mf)
    meta = {}
    mp = os.path.join(proc, 'map', 'meta.json')
    if os.path.exists(mp):
        meta = _load_json(mp)
    lg = {}
    lp = os.path.join(proc, 'map', 'lane_graph.json')
    if os.path.exists(lp):
        lg = _load_json(lp)

    # 消歧探针：全局系的车道中心线
    lane_probe = [np.asarray(f['polyline'], dtype=np.float64)
                  for f in map_d['features']
                  if f['type'] == 'lane' and f.get('polyline') and len(f['polyline']) >= 2]

    align = align_scene(scene_dir, proc, cam, lane_probe=lane_probe)

    polylines: List[Dict[str, Any]] = []
    lanes: Dict[str, Dict[str, Any]] = {}
    for f in map_d['features']:
        kind = f['type']
        if kind == 'stop_sign':
            P = to_scene(align, [f['position']])
        else:
            k, pts = _polyline_of(f)
            if k is None or len(pts) < 2:
                continue
            P = to_scene(align, pts)
        item = {'id': f.get('id'), 'type': kind, 'pts': P.tolist()}
        if kind == 'lane':
            item['lane_type'] = f.get('lane_type')
            item['speed_limit_mph'] = f.get('speed_limit_mph')
            item['entry_lanes'] = f.get('entry_lanes') or []
            item['exit_lanes'] = f.get('exit_lanes') or []
            item['left_neighbors'] = f.get('left_neighbors') or []
            item['right_neighbors'] = f.get('right_neighbors') or []
            lanes[str(f['id'])] = item
            item['pts'] = P.tolist()
        polylines.append(item)

    # 锚点：路口进口（出口 >=2）、停车标志、斑马线
    junctions, anchors = [], []
    for f in map_d['features']:
        if f['type'] == 'lane' and len(f.get('exit_lanes') or []) >= 2 and f.get('polyline'):
            end = to_scene(align, [f['polyline'][-1]])[0]
            junctions.append({'kind': 'lane_fork', 'lane': f['id'],
                              'n_exits': len(f['exit_lanes']),
                              'exits': list(f['exit_lanes']), 'at': end.tolist()})
    for f in map_d['features']:
        if f['type'] == 'stop_sign':
            pos = to_scene(align, [f['position']])[0]
            anchors.append({'kind': 'stop_sign', 'lanes': f.get('lanes') or [], 'at': pos.tolist()})
        elif f['type'] == 'crosswalk' and f.get('polygon'):
            c = to_scene(align, f['polygon']).mean(0)
            anchors.append({'kind': 'crosswalk', 'at': c.tolist()})
        elif f['type'] == 'speed_bump' and f.get('polygon'):
            c = to_scene(align, f['polygon']).mean(0)
            anchors.append({'kind': 'speed_bump', 'at': c.tolist()})

    out = {
        'scene_dir': scene_dir, 'processed_dir': proc,
        'segment': meta.get('segment'), 'meta': meta,
        'align': align, 'polylines': polylines, 'lanes': lanes,
        'junctions': junctions, 'anchors': anchors,
        'counts': map_d.get('counts', {}),
        'lane_graph_info': {k: v for k, v in lg.items() if k != 'nodes'},
    }
    if use_cache:
        _CACHE[key] = out
    return out


# ---------------------------------------------------------------- 查询

def _lane_pts(map_d, lane_id) -> Optional[np.ndarray]:
    l = map_d['lanes'].get(str(lane_id))
    if not l:
        return None
    return np.asarray(l['pts'], dtype=np.float64)


def lane_polyline(map_d: Dict[str, Any], lane_id) -> Optional[np.ndarray]:
    return _lane_pts(map_d, lane_id)


def nearest_lane(map_d: Dict[str, Any], xz,
                 lane_ids: Optional[Iterable] = None) -> Tuple[Optional[str], float, int]:
    """场景系 xz 到最近车道中心线的距离 → (lane_id, dist_m, 最近点序号)。"""
    q = np.asarray(xz, dtype=np.float64)[:2]
    ids = list(map_d['lanes'].keys()) if lane_ids is None else [str(i) for i in lane_ids]
    best, bd, bi = None, float('inf'), -1
    for lid in ids:
        P = _lane_pts(map_d, lid)
        if P is None or len(P) == 0:
            continue
        d = np.linalg.norm(P[:, [0, 2]] - q, axis=1)
        j = int(np.argmin(d))
        if d[j] < bd:
            best, bd, bi = lid, float(d[j]), j
    return best, bd, bi


def lane_heading(map_d: Dict[str, Any], lane_id, xz) -> Optional[np.ndarray]:
    """车道中心线在离 xz 最近处的单位前进方向（场景系 xz）。"""
    P = _lane_pts(map_d, lane_id)
    if P is None or len(P) < 2:
        return None
    q = np.asarray(xz, dtype=np.float64)[:2]
    j = int(np.argmin(np.linalg.norm(P[:, [0, 2]] - q, axis=1)))
    a = max(0, j - 1)
    b = min(len(P) - 1, j + 1)
    v = P[b, [0, 2]] - P[a, [0, 2]]
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else None


def lane_length(map_d, lane_id) -> float:
    P = _lane_pts(map_d, lane_id)
    if P is None or len(P) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(P[:, [0, 2]], axis=0), axis=1).sum())


def successors(map_d, lane_id) -> List[str]:
    l = map_d['lanes'].get(str(lane_id))
    if not l:
        return []
    out = [str(x) for x in l.get('exit_lanes') or [] if str(x) in map_d['lanes']]
    out += [str(x) for x in (l.get('left_neighbors') or []) + (l.get('right_neighbors') or [])
            if str(x) in map_d['lanes'] and str(x) not in out]
    return out


def _pure_successors(map_d, lane_id) -> List[str]:
    l = map_d['lanes'].get(str(lane_id))
    if not l:
        return []
    return [str(x) for x in l.get('exit_lanes') or [] if str(x) in map_d['lanes']]


def predecessors(map_d, lane_id) -> List[str]:
    lid = str(lane_id)
    return [k for k, v in map_d['lanes'].items() if lid in [str(x) for x in v.get('exit_lanes') or []]]


def route_between(map_d, from_lane, to_lane, max_hops: int = 12) -> Optional[List[str]]:
    """车道图 BFS：只在 successor/相邻边上游走。"""
    a, b = str(from_lane), str(to_lane)
    if a == b:
        return [a]
    if a not in map_d['lanes'] or b not in map_d['lanes']:
        return None
    seen = {a}
    queue = [[a]]
    while queue:
        path = queue.pop(0)
        if len(path) > max_hops:
            continue
        for nxt in successors(map_d, path[-1]):
            if nxt in seen:
                continue
            if nxt == b:
                return path + [nxt]
            seen.add(nxt)
            queue.append(path + [nxt])
    return None


def anchors_near(map_d, xz, radius: float = 25.0) -> List[Dict[str, Any]]:
    q = np.asarray(xz, dtype=np.float64)[:2]
    out = []
    for j in map_d['junctions']:
        p = np.asarray(j['at'], dtype=np.float64)
        d = float(np.linalg.norm(p[[0, 2]] - q))
        if d <= radius:
            out.append({**j, 'dist_m': d})
    for a in map_d['anchors']:
        p = np.asarray(a['at'], dtype=np.float64)
        d = float(np.linalg.norm(p[[0, 2]] - q))
        if d <= radius:
            out.append({**a, 'dist_m': d})
    out.sort(key=lambda x: x['dist_m'])
    return out


def on_road(map_d, xz, tol: float = ROAD_TOL_M) -> Dict[str, Any]:
    """道路约束检查：点是否落在车道走廊内。"""
    lid, d, idx = nearest_lane(map_d, xz)
    return {'on_road': bool(d <= tol), 'lane': lid, 'dist_m': d,
            'tol_m': tol, 'corridor_m': LANE_HALF_WIDTH}


# ------------------------------------------------------- 道路约束下的路径规划

def plan_path_through(map_d, start_xyz, end_xyz=None, lane_id=None,
                      max_hops: int = 12, sample_step: float = 2.0) -> Dict[str, Any]:
    """沿车道中心线生成一条"合法"路径：起点吸附到车道 → 沿车道图走到目标车道。

    返回 {'pts': Nx3(场景系), 'lane_ids': [...], 'start_lane':..., 'end_lane':..., 'reason':...}
    """
    start = np.asarray(start_xyz, dtype=np.float64).reshape(3)
    if lane_id is None:
        lane_id, d, _ = nearest_lane(map_d, start[[0, 2]])
    if lane_id is None:
        return {'pts': np.asarray([start]), 'lane_ids': [], 'reason': 'no_lane'}
    end_lane = None
    if end_xyz is not None:
        end_lane, _, _ = nearest_lane(map_d, np.asarray(end_xyz, dtype=np.float64)[:2])
    route = None
    if end_lane and str(end_lane) != str(lane_id):
        route = route_between(map_d, lane_id, end_lane, max_hops)
    if route is None:
        # 没给终点 / 走不通：就沿当前车道的后继一直往前走
        route = [str(lane_id)]
        cur = str(lane_id)
        for _ in range(max_hops):
            nxt = _pure_successors(map_d, cur)
            if len(nxt) != 1:
                break
            route.append(nxt[0])
            cur = nxt[0]
    pts: List[np.ndarray] = []
    for lid in route:
        P = _lane_pts(map_d, lid)
        if P is None:
            continue
        if pts and np.linalg.norm(P[0, [0, 2]] - pts[-1][[0, 2]]) > 1e-6:
            # 车道之间常有小间隙，插值补上
            n = max(1, int(np.linalg.norm(P[0, [0, 2]] - pts[-1][[0, 2]]) / sample_step))
            for k in range(1, n + 1):
                pts.append(pts[-1] + (P[0] - pts[-1]) * (k / n))
        pts.extend(list(P))
    # 重采样，避免点过密
    P = np.asarray(pts, dtype=np.float64)
    return {'pts': P, 'lane_ids': route, 'start_lane': str(lane_id),
            'end_lane': str(end_lane) if end_lane else None, 'reason': 'lane_graph'}


def route_following_lane(map_d, lane_id, forward_m: float = 60.0,
                         backward_m: float = 12.0) -> Optional[np.ndarray]:
    """以某条车道为主轴，取一段"前后"中心线（场景系），用于铺轨迹。"""
    P = _lane_pts(map_d, lane_id)
    if P is None or len(P) < 2:
        return None
    seg = [P]
    cur, acc = str(lane_id), 0.0
    while acc < forward_m:
        nxt = _pure_successors(map_d, cur)
        if len(nxt) != 1:
            break
        Q = _lane_pts(map_d, nxt[0])
        if Q is None:
            break
        acc += float(np.linalg.norm(np.diff(Q[:, [0, 2]], axis=0), axis=1).sum())
        seg.append(Q)
        cur = nxt[0]
    out = np.concatenate(seg, axis=0)
    return out


# ---------------------------------------------------------------- 叠图（诊断）

def _overlay(map_d, scene_dir, out_png, size=1500, scale_px=6.0, cam=0):
    import cv2
    img = np.zeros((size, size, 3), np.uint8)
    COL = {'lane': (0, 220, 0), 'road_line': (0, 200, 255), 'road_edge': (255, 160, 0),
           'crosswalk': (255, 0, 255), 'stop_sign': (0, 0, 255), 'speed_bump': (0, 140, 255),
           'driveway': (120, 120, 120)}

    def to_px(p):
        return int(round(p[0] * scale_px + size / 2)), int(round(p[2] * scale_px + size / 2))

    for it in map_d['polylines']:
        P = np.asarray(it['pts'], dtype=np.float64)
        col = COL.get(it['type'], (200, 200, 200))
        for a, b in zip(P[:-1], P[1:]):
            cv2.line(img, to_px(a), to_px(b), col, 2, cv2.LINE_AA)
    ts, fids, mats, gfids = _scene_cam_poses(scene_dir)
    for M in mats:
        cv2.circle(img, to_px(M[:3, 3]), 2, (255, 255, 255), -1)
    od = os.path.join(scene_dir, 'dynamic_objects')
    fs = sorted(glob.glob(os.path.join(od, 'frame_0000_objects.json')))
    if fs:
        for o in _load_json(fs[0]):
            m = np.asarray(o['pose_world'], dtype=np.float64)
            cv2.circle(img, to_px(m[:3, 3]), 6, (0, 255, 255), -1)
    cv2.imwrite(out_png, img)
    return out_png


def _ego_to_lane_stats(map_d, scene_dir, cam=0) -> Dict[str, float]:
    ts, fids, mats, gfids = _scene_cam_poses(scene_dir)
    ds = []
    for M in mats:
        _, d, _ = nearest_lane(map_d, M[:3, 3][[0, 2]])
        ds.append(d)
    ds = np.asarray(ds)
    return {'median_m': float(np.median(ds)), 'max_m': float(np.max(ds)), 'n': int(len(ds))}


def main():
    ap = argparse.ArgumentParser(description='把 waymo14 地图接进 studio 场景（对齐 + 查询 + 叠图）')
    ap.add_argument('--scene', required=True, help='如 output/waymo_eval_14/016')
    ap.add_argument('--processed', default=None)
    ap.add_argument('--cam', type=int, default=0)
    ap.add_argument('--overlay', default='')
    ap.add_argument('--at', default='', help='查这个点（场景系 "x,z"）到最近车道/路口的距离')
    args = ap.parse_args()
    m = load_scene_map(args.scene, args.processed, cam=args.cam)
    al = m['align']
    print('scene    = %s' % m['scene_dir'])
    print('processed= %s' % m['processed_dir'])
    print('segment  = %s' % m['segment'])
    print('帧对应   = %s（%d 对，相机 cam=%s）' % (al['source'], al['n_pairs'], al.get('cam')))
    print('对齐     = scale=%.4f  rms=%.3fm  镜像消歧: up_dot=%+.3f flip=%+.0f  scale_ok=%s' % (
        al['s'], al['rms'], al.get('up_dot', 0.0), al.get('flip', 0.0), al['scale_ok']))
    for c in al.get('candidates', []):
        print('   候选 flip=%+.0f rms=%.3fm up_dot=%+.3f 自车→车道=%.2fm' % (
            c.get('flip', 0), c.get('rms', 0), c.get('up_dot', 0), c.get('ego_lane_med_m') or -1))
    print('模型相机朝向跨帧一致性(诊断,不参与求解) = %.2f°（中位）' % (al.get('rot_spread_med_deg') or 0))
    if al.get('warning'):
        print('   警告: %s' % al['warning'])
    print('地图     = %s' % json.dumps(m['counts'], ensure_ascii=False))
    st = _ego_to_lane_stats(m, m['scene_dir'], cam=args.cam)
    print('自车到最近车道中心线: 中位 %.2fm 最大 %.2fm (n=%d)' % (st['median_m'], st['max_m'], st['n']))
    print('车道=%d 折线=%d 路口进口=%d 锚点=%d' % (
        len(m['lanes']), len(m['polylines']), len(m['junctions']), len(m['anchors'])))
    if args.at:
        x, z = [float(v) for v in args.at.split(',')]
        lid, d, i = nearest_lane(m, (x, z))
        print('点(%s) 最近车道 #%s 距离 %.2fm；附近锚点: %s' % (
            args.at, lid, d, json.dumps(anchors_near(m, (x, z))[:3], ensure_ascii=False)))
    if args.overlay:
        _overlay(m, m['scene_dir'], args.overlay, cam=args.cam)
        print('叠图已保存: %s（绿=车道 黄=标线 橙=边界 洋红=斑马线 红=停车标志 灰=出入口 白=自车 青=物体）'
              % args.overlay)
    print('WAYMO_MAP_DONE')


if __name__ == '__main__':
    main()
