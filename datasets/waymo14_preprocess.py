"""把 Waymo Open Dataset v1.4.x（带地图）的 tfrecord 解析成 DGGT 的 processed 目录格式。

背景：`data/waymo/processed/validation/<NNN>/` 是 DGGT 训练/推理直接吃的格式：
    ego_pose/{i:03d}.txt      4x4 车辆→全局（WOD frame.pose）
    extrinsics/{cam}.txt      4x4 相机→车辆（WOD camera extrinsic）
    intrinsics/{cam}.txt      9 个数 fx,fy,cx,cy,k1,k2,p1,p2,k3
    images/{i:03d}_{cam}.jpg  原分辨率（1920x1280 / 1920x886）
    images_4/{i:03d}_{cam}.jpg 同图下采样 1/4
    （depth_flows_4/ dynamic_masks/ ground_label_4/ lidar/ 这些由后续阶段填，这里建空目录保持一致）

旧数据（waymo12）**没有地图**；v1.4.x 的 tfrecord 第一帧里有 `frame.map_features`
（车道中心线/标线/道路边界/停车标志/斑马线/减速带/出入口）与 `map_pose_offset`。
本脚本除上述标准文件外，额外写出：
    map/map_features.json        地图要素（与 ego_pose 同一全局坐标系）
    map/lane_graph.json          车道图（节点=车道 id，边=entry/exit/neighbors）
    map/meta.json                segment/帧数/相机映射/map_pose_offset
并在 <dst> 下输出 scene_index.csv（NNN ↔ segment 名）与 scene_names.txt。

用法：
    python datasets/waymo14_preprocess.py --dst data/waymo14 \
        --index_from data/waymo_val_list.txt \
        --only data/waymo_mytest_list.txt          # 只转这 25 个（按 val_list 的编号落位）
    # 常用开关：--no-images（只写位姿+地图，快） --compact（从 000 起重新编号）
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_SRC = os.path.join(ROOT, 'waymo/waymo14')


def _read_list(path: str):
    out = []
    if not path:
        return out
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split(',')
            seg = parts[1] if len(parts) > 1 else parts[0]
            out.append(seg.replace('segment-', '').replace('_with_camera_labels', ''))
    return out


def _write_matrix(path: str, m: np.ndarray):
    np.savetxt(path, np.asarray(m, dtype=np.float64), fmt='%.18e')


def map_to_json(frame) -> dict:
    """MapFeature → 可直接建图/画图的 json（保持与 ego_pose 相同的全局坐标系）。"""
    feats = []
    for mf in frame.map_features:
        kind = mf.WhichOneof('feature_data')
        if kind == 'lane':
            ln = mf.lane
            feats.append({
                'id': int(mf.id), 'type': 'lane',
                'polyline': [[float(q.x), float(q.y), float(q.z)] for q in ln.polyline],
                'lane_type': ln.LaneType.Name(ln.type),
                'speed_limit_mph': float(ln.speed_limit_mph) if ln.HasField('speed_limit_mph') else None,
                'interpolating': bool(ln.interpolating),
                'entry_lanes': [int(x) for x in ln.entry_lanes],
                'exit_lanes': [int(x) for x in ln.exit_lanes],
                'left_neighbors': [int(n.feature_id) for n in ln.left_neighbors],
                'right_neighbors': [int(n.feature_id) for n in ln.right_neighbors],
                'left_boundaries': [int(b.boundary_feature_id) for b in ln.left_boundaries],
                'right_boundaries': [int(b.boundary_feature_id) for b in ln.right_boundaries],
            })
        elif kind == 'road_line':
            feats.append({'id': int(mf.id), 'type': 'road_line',
                          'road_line_type': mf.road_line.RoadLineType.Name(mf.road_line.type),
                          'polyline': [[float(q.x), float(q.y), float(q.z)]
                                       for q in mf.road_line.polyline]})
        elif kind == 'road_edge':
            feats.append({'id': int(mf.id), 'type': 'road_edge',
                          'road_edge_type': mf.road_edge.RoadEdgeType.Name(mf.road_edge.type),
                          'polyline': [[float(q.x), float(q.y), float(q.z)]
                                       for q in mf.road_edge.polyline]})
        elif kind == 'stop_sign':
            feats.append({'id': int(mf.id), 'type': 'stop_sign',
                          'lanes': [int(x) for x in mf.stop_sign.lane],
                          'position': [float(mf.stop_sign.position.x), float(mf.stop_sign.position.y),
                                       float(mf.stop_sign.position.z)]})
        elif kind == 'crosswalk':
            feats.append({'id': int(mf.id), 'type': 'crosswalk',
                          'polygon': [[float(q.x), float(q.y), float(q.z)]
                                      for q in mf.crosswalk.polygon]})
        elif kind == 'speed_bump':
            feats.append({'id': int(mf.id), 'type': 'speed_bump',
                          'polygon': [[float(q.x), float(q.y), float(q.z)]
                                      for q in mf.speed_bump.polygon]})
        elif kind == 'driveway':
            feats.append({'id': int(mf.id), 'type': 'driveway',
                          'polygon': [[float(q.x), float(q.y), float(q.z)]
                                      for q in mf.driveway.polygon]})
    return {
        'features': feats,
        'counts': {t: sum(1 for f in feats if f['type'] == t)
                   for t in sorted({f['type'] for f in feats})},
    }


def lane_graph(map_json: dict) -> dict:
    """车道图：节点=车道（中心线/限速/类型），边=entry/exit 连接 + 左右邻居。"""
    lanes = {f['id']: f for f in map_json['features'] if f['type'] == 'lane'}
    nodes = {}
    for lid, f in lanes.items():
        pl = np.asarray(f['polyline'], dtype=float)
        nodes[str(lid)] = {
            'lane_type': f['lane_type'],
            'speed_limit_mph': f['speed_limit_mph'],
            'centerline': f['polyline'],
            'start': pl[0].tolist() if len(pl) else None,
            'end': pl[-1].tolist() if len(pl) else None,
            'length_m': float(np.linalg.norm(np.diff(pl[:, :2], axis=0), axis=1).sum()) if len(pl) > 1 else 0.0,
        }
    edges = []
    for lid, f in lanes.items():
        for nxt in f['exit_lanes']:
            if nxt in lanes:
                edges.append({'from': lid, 'to': int(nxt), 'kind': 'successor'})
        for nb in f['right_neighbors']:
            if nb in lanes and nb > lid:
                edges.append({'from': lid, 'to': int(nb), 'kind': 'neighbor'})
    # 路口进口：出口 >= 2 条的车道
    junctions = [{'lane': lid, 'n_exits': len(f['exit_lanes']), 'n_entries': len(f['entry_lanes']),
                  'at': (np.asarray(f['polyline'], dtype=float)[-1].tolist() if f['polyline'] else None)}
                 for lid, f in lanes.items() if len(f['exit_lanes']) >= 2]
    return {'nodes': nodes, 'edges': edges, 'junction_entries': junctions,
            'n_lanes': len(lanes), 'n_edges': len(edges), 'n_junction_entries': len(junctions)}


def main():
    ap = argparse.ArgumentParser(description='Waymo v1.4.x(带地图) tfrecord → DGGT processed 目录')
    ap.add_argument('--src', default=DEFAULT_SRC, help='tfrecord 所在目录')
    ap.add_argument('--dst', default=os.path.join(ROOT, 'data/waymo14'))
    ap.add_argument('--split', default='validation')
    ap.add_argument('--index_from', default=os.path.join(ROOT, 'data/waymo_mytest_list.txt'),
                    help='用它的行号当输出编号 NNN（默认 mytest 的 000..024，与旧 data/waymo/processed 一致）')
    ap.add_argument('--only', default=os.path.join(ROOT, 'data/waymo_mytest_list.txt'),
                    help='只转这个列表里的 segment（默认就是 mytest 那 25 个）')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--compact', action='store_true', help='从 000 起重新编号（忽略 --index_from 的行号）')
    ap.add_argument('--no-images', action='store_true', help='只写位姿/内外参/地图（不写图片）')
    ap.add_argument('--skip-map', action='store_true', help='不写地图（只做标准格式）')
    ap.add_argument('--skip-empties', action='store_true', help='不建 depth_flows_4 等空目录')
    ap.add_argument('--estimate', action='store_true',
                    help='只估算需要的字节/inode，不写文件（先看磁盘够不够）')
    args = ap.parse_args()

    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
    import tensorflow as tf
    from waymo_open_dataset import dataset_pb2
    try:
        import cv2
    except Exception:  # noqa: BLE001
        cv2 = None

    idx_list = _read_list(args.index_from)
    idx_of = {seg: i for i, seg in enumerate(idx_list)}
    want = _read_list(args.only) if args.only else list(idx_list)
    files = {}
    for p in sorted(glob.glob(os.path.join(args.src, '*.tfrecord'))):
        name = os.path.basename(p)
        for a, b in (('individual_files_validation_', ''), ('individual_files_training_', ''),
                     ('individual_files_test_', ''), ('segment-', ''),
                     ('_with_camera_labels.tfrecord', ''), ('.tfrecord', '')):
            name = name.replace(a, b)
        files[name] = p
    print('src=%s  tfrecord=%d  目标 segment=%d' % (args.src, len(files), len(want)), flush=True)

    out_root = os.path.join(args.dst, 'processed', args.split)
    os.makedirs(out_root, exist_ok=True)
    if args.estimate:
        # 估算：每段 ~199 帧 × 5 相机 ×（images + images_4）+ 每帧 1 个 ego_pose + 内外参 + 地图
        n_seg = len([s for s in want if s in files])
        frames = 199
        n_txt = frames + 10 + 5          # ego_pose + 5 intrinsics + 5 extrinsics
        n_img = 0 if args.no_images else frames * 5 * (2 if cv2 is not None else 0)
        per = n_txt + n_img + 6          # + 地图/索引文件
        import shutil
        st = shutil.disk_usage(args.dst if os.path.exists(args.dst) else ROOT)
        try:
            sti = os.statvfs(args.dst if os.path.exists(args.dst) else ROOT)
            free_inodes = sti.f_ffree
        except Exception:  # noqa: BLE001
            free_inodes = -1
        print('=== 估算（%d 段，%s）===' % (n_seg, '含图片' if not args.no_images else '仅位姿+地图'))
        print('  inode 需求 ≈ %d（%.1f 个/段）' % (n_seg * per, per))
        print('  字节 需求 ≈ %.1f GB（旧 processed 约 0.37 GB/段）' % (n_seg * 0.37))
        print('  当前可用: 字节 %.1f GB, inode %s' % (st.free / 1e9, free_inodes))
        if free_inodes >= 0 and free_inodes < n_seg * per:
            print('  ⚠ inode 不够：需要先删掉一些文件（例如不再用的 data/waymo/processed 或 waymo/raw）')
        return
    rows, done = [], 0
    for seg in want:
        if seg not in files:
            print('  [跳过] 本地没有 %s' % seg, flush=True)
            continue
        if args.limit and done >= args.limit:
            break
        out_idx = done if args.compact else idx_of.get(seg, done)
        out_dir = os.path.join(out_root, '%03d' % out_idx)
        for sub in (('ego_pose', 'extrinsics', 'intrinsics', 'images', 'images_4') +
                    (() if args.skip_empties else ('depth_flows_4', 'dynamic_masks',
                                                   'ground_label_4', 'lidar'))):
            os.makedirs(os.path.join(out_dir, sub), exist_ok=True)
        t0 = time.time()
        n_frames = 0
        ts = []
        map_json, meta = None, None
        ds = tf.data.TFRecordDataset(files[seg], compression_type='')
        for i, raw in enumerate(ds):
            fr = dataset_pb2.Frame.FromString(bytearray(raw.numpy()))
            if i == 0:
                for c in fr.context.camera_calibrations:
                    cam = int(c.name) - 1                      # WOD 1..5 → 0..4（0=前视）
                    _write_matrix(os.path.join(out_dir, 'intrinsics', '%d.txt' % cam),
                                  np.asarray(c.intrinsic, dtype=np.float64).reshape(-1, 1))
                    _write_matrix(os.path.join(out_dir, 'extrinsics', '%d.txt' % cam),
                                  np.asarray(c.extrinsic.transform, dtype=np.float64).reshape(4, 4))
                if not args.skip_map:
                    map_json = map_to_json(fr)
                meta = {
                    'segment': seg, 'split': args.split, 'index': int(out_idx),
                    'fps': 10.0,
                    'cameras': {int(c.name) - 1: {'name': int(c.name), 'width': int(c.width),
                                                  'height': int(c.height)} for c in fr.context.camera_calibrations},
                    'map_pose_offsets': [],
                }
            _write_matrix(os.path.join(out_dir, 'ego_pose', '%03d.txt' % i),
                          np.asarray(fr.pose.transform, dtype=np.float64).reshape(4, 4))
            ts.append(int(fr.timestamp_micros))
            if meta is not None:
                mo = fr.map_pose_offset
                meta['map_pose_offsets'].append([float(mo.x), float(mo.y), float(mo.z)])
            if not args.no_images:
                for im in fr.images:
                    cam = int(im.name) - 1
                    arr = np.frombuffer(im.image, dtype=np.uint8)
                    with open(os.path.join(out_dir, 'images', '%03d_%d.jpg' % (i, cam)), 'wb') as fh:
                        fh.write(arr.tobytes())
                    if cv2 is not None:
                        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                        if img is not None:
                            small = cv2.resize(img, (img.shape[1] // 4, img.shape[0] // 4),
                                               interpolation=cv2.INTER_AREA)
                            cv2.imwrite(os.path.join(out_dir, 'images_4', '%03d_%d.jpg' % (i, cam)),
                                        small, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            n_frames += 1
        with open(os.path.join(out_dir, 'timestamp.txt'), 'w') as fh:
            fh.write('\n'.join(str(t) for t in ts))
        if map_json is not None:
            os.makedirs(os.path.join(out_dir, 'map'), exist_ok=True)
            lg = lane_graph(map_json)
            with open(os.path.join(out_dir, 'map', 'map_features.json'), 'w') as fh:
                json.dump(map_json, fh, ensure_ascii=False)
            with open(os.path.join(out_dir, 'map', 'lane_graph.json'), 'w') as fh:
                json.dump(lg, fh, ensure_ascii=False)
            meta['map'] = {'counts': map_json['counts'], 'n_lanes': lg['n_lanes'],
                           'n_edges': lg['n_edges'], 'n_junction_entries': lg['n_junction_entries']}
            with open(os.path.join(out_dir, 'map', 'meta.json'), 'w') as fh:
                json.dump(meta, fh, ensure_ascii=False)
        elif meta is not None:
            with open(os.path.join(out_dir, 'meta.json'), 'w') as fh:
                json.dump(meta, fh, ensure_ascii=False)
        mm = meta['map'] if (meta and meta.get('map')) else {}
        print('  [%03d] %-52s frames=%-4d %.1fs  地图:%s' % (
            out_idx, seg[:52], n_frames, time.time() - t0, mm), flush=True)
        rows.append({'index': out_idx, 'segment': seg, 'num_frames': n_frames,
                     'n_lanes': mm.get('n_lanes', ''), 'n_junction_entries': mm.get('n_junction_entries', ''),
                     'counts': json.dumps(mm.get('counts', {}), ensure_ascii=False)})
        done += 1

    with open(os.path.join(args.dst, 'scene_index.csv'), 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=['index', 'segment', 'num_frames', 'n_lanes',
                                           'n_junction_entries', 'counts'])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    with open(os.path.join(args.dst, 'scene_names.txt'), 'w') as fh:
        fh.write('\n'.join('%03d %s' % (r['index'], r['segment']) for r in rows) + '\n')
    print('\n完成 %d 个场景 → %s' % (len(rows), out_root), flush=True)
    print('索引表: %s' % os.path.join(args.dst, 'scene_index.csv'), flush=True)
    print('WAYMO14_PREP_DONE', flush=True)


if __name__ == '__main__':
    main()
