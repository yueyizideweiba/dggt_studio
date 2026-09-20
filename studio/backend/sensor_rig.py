"""多相机 rig 传感器仿真（P2）：用 Waymo 5 相机标定 + 重建出的自车位姿，输出多相机观测。

闭环仿真需要的"传感器"不是单张自车视角，而是一整套相机 rig：
- 相机内参来自 `data/waymo/processed/<split>/<segment>/intrinsics/{c}.txt`（fx, fy, cx, cy, k1, k2, p1, p2, k3）；
- 相机相对自车的位姿来自 `extrinsics/{c}.txt`（x=前 / y=左 / z=上 的相机系，需共轭到 OpenCV 系）；
- 自车位姿来自重建结果 `ego_pose/frame_XXXX_ego.json`（参考相机 = 重建时喂进去的那台，默认 0）。

于是任意相机 C 的世界位姿：`c2w_C = c2w_ref @ inv(M_ref) @ M_C`（相对位姿与全局坐标系无关）。
可选 `--distort` 把径向/切向畸变加回渲染图，让观测更像真实相机。

用法：
    python sensor_rig.py --scene output/trust/scene001_cam0/001 --segment 001 \
        --frame 0 --cameras 0,1,2,3,4 --out output/sensor_rig/scene001
"""
from __future__ import annotations

import argparse
import scene_frames as _sf
import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

# 标定文件里的相机系是 x=前, y=左, z=上；DGGT/渲染用 OpenCV 系 x=右, y=下, z=前
_CAL_TO_OPENCV = np.array([[0.0, -1.0, 0.0],
                           [0.0, 0.0, -1.0],
                           [1.0, 0.0, 0.0]], dtype=np.float64)


def _conj(T: np.ndarray) -> np.ndarray:
    B = np.eye(4)
    B[:3, :3] = _CAL_TO_OPENCV
    return B @ np.asarray(T, dtype=np.float64) @ np.linalg.inv(B)


def load_camera_calib(data_root: str, segment: str, camera: int) -> Dict:
    """读一台相机的标定（外参 4x4 + 内参/畸变）。"""
    ext = np.loadtxt(os.path.join(data_root, segment, "extrinsics", f"{camera}.txt"))
    if ext.shape == (3, 4):
        ext = np.vstack([ext, np.array([[0.0, 0.0, 0.0, 1.0]])])
    v = np.atleast_1d(np.loadtxt(os.path.join(data_root, segment, "intrinsics", f"{camera}.txt"))).ravel()
    fx, fy, cx, cy = [float(x) for x in v[:4]]
    dist = [float(x) for x in v[4:9]] if len(v) >= 5 else []
    return {"camera": int(camera), "M": ext.astype(np.float64),
            "K": np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64),
            "dist": dist}


def load_rig(data_root: str, segment: str, cameras: Sequence[int]) -> Dict[int, Dict]:
    return {int(c): load_camera_calib(data_root, segment, int(c)) for c in cameras}


def camera_to_ref(rig: Dict[int, Dict], ref_camera: int, camera: int) -> np.ndarray:
    """ref 相机 → 目标相机 的相对位姿（OpenCV 系，可直接左乘 c2w）。"""
    return _conj(np.linalg.inv(rig[int(ref_camera)]["M"]) @ rig[int(camera)]["M"])


def scaled_intrinsics(rig: Dict[int, Dict], camera: int, width: int, height: int,
                      native_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """把标定内参缩放到目标分辨率（native_size 缺省用 Waymo 原始 1920x1280/886）。"""
    K = np.asarray(rig[int(camera)]["K"], dtype=np.float64).copy()
    w0, h0 = native_size or (1920, 1280 if int(camera) in (0, 1, 2) else 886)
    K[0, 0] *= width / float(w0)
    K[0, 2] *= width / float(w0)
    K[1, 1] *= height / float(h0)
    K[1, 2] *= height / float(h0)
    return K


def apply_distortion(img: np.ndarray, dist: Sequence[float]) -> np.ndarray:
    """把畸变"加回"渲染图（用 inverse map：对每个无畸变像素找它在有畸变图上的位置）。

    只有 fx/fy/cx/cy + k1,k2,p1,p2 时用 `cv2.undistortPoints` 的逆用法；这里直接用
    畸变模型前向公式构造 remap，速度够快且不依赖额外标定。
    """
    if not dist or len(dist) < 4:
        return img
    import cv2
    h, w = img.shape[:2]
    k1, k2, p1, p2 = [float(x) for x in dist[:4]]
    fx = fy = 1.0
    cx, cy = w / 2.0, h / 2.0
    xs, ys = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))
    x = (xs - cx) / fx
    y = (ys - cy) / fy
    r2 = x * x + y * y
    radial = 1.0 + k1 * r2 + k2 * r2 * r2
    xd = x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
    yd = y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y
    map_x = (xd * fx + cx).astype(np.float32)
    map_y = (yd * fy + cy).astype(np.float32)
    return cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)


def render_rig_frame(renderer, frame_idx: int, rig: Dict[int, Dict], ref_camera: int,
                     cameras: Sequence[int], ego_c2w: Optional[np.ndarray] = None,
                     size: Tuple[int, int] = (960, 640), distort: bool = False,
                     object_overrides=None, extra_objects=None) -> Dict[int, np.ndarray]:
    """渲染一帧的整套相机 rig。返回 {camera: uint8 (H,W,3) RGB}。"""
    import torch
    if ego_c2w is None:
        _ego = _sf.load_ego_pose(renderer.scene_path, int(frame_idx), 0)
        if _ego is None:
            raise FileNotFoundError(f"找不到真实帧 {frame_idx} 的自车位姿")
        d = _ego
        ego_c2w = np.asarray(d["camera_extrinsics_world"], dtype=np.float64)
        if ego_c2w.shape == (3, 4):
            ego_c2w = np.vstack([ego_c2w, np.array([[0.0, 0.0, 0.0, 1.0]])])
    W, H = int(size[0]), int(size[1])
    out: Dict[int, np.ndarray] = {}
    for cam in cameras:
        T = camera_to_ref(rig, ref_camera, int(cam))
        c2w = np.asarray(ego_c2w, dtype=np.float64) @ T
        K = scaled_intrinsics(rig, int(cam), W, H)
        img = renderer._render_frame_with_object_overrides(
            int(frame_idx), object_overrides or {},
            c2w_override=torch.tensor(c2w, dtype=torch.float32, device=renderer.device),
            K_override=torch.tensor(K, dtype=torch.float32, device=renderer.device),
            width_override=W, height_override=H,
            extra_objects=list(extra_objects or []))
        arr = np.asarray(img, dtype=np.uint8)
        if distort:
            arr = apply_distortion(arr, rig[int(cam)]["dist"])
        out[int(cam)] = arr
    return out


def main():
    ap = argparse.ArgumentParser(description="多相机 rig 传感器仿真（渲染整套相机观测）")
    ap.add_argument("--scene", required=True, help="重建场景目录（含 ego_pose/gaussians）")
    ap.add_argument("--segment", required=True, help="标定来源的预处理场景名（如 001）")
    ap.add_argument("--data_root", default=str(ROOT / "data/waymo/processed/validation"))
    ap.add_argument("--frame", type=int, default=0)
    ap.add_argument("--cameras", default="0,1,2,3,4")
    ap.add_argument("--ref_camera", type=int, default=0, help="重建时喂进去的那台相机")
    ap.add_argument("--out", default="")
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--height", type=int, default=640)
    ap.add_argument("--distort", action="store_true", help="把镜头畸变加回渲染图")
    args = ap.parse_args()

    import cv2
    from dggt_engine import DGGTRenderer
    cams = [int(x) for x in args.cameras.split(",") if x.strip() != ""]
    rig = load_rig(args.data_root, args.segment, cams)
    renderer = DGGTRenderer(scene_path=args.scene, device="cuda", load_sky=True, verbose=False)
    imgs = render_rig_frame(renderer, args.frame, rig, args.ref_camera, cams,
                            size=(args.width, args.height), distort=args.distort)
    out_dir = args.out or str(ROOT / "output" / "sensor_rig" / os.path.basename(os.path.normpath(args.scene)))
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    for cam, img in imgs.items():
        p = os.path.join(out_dir, f"frame{args.frame:04d}_cam{cam}.png")
        cv2.imwrite(p, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        T = camera_to_ref(rig, args.ref_camera, cam)
        ang = math.degrees(math.acos(max(-1.0, min(1.0, float(T[2, 2])))))
        rows.append({"camera": cam, "view_angle_vs_ref_deg": round(ang, 2),
                     "t_in_ref_cam": [round(float(x), 3) for x in T[:3, 3]], "file": p})
    # 拼一张总览
    hs = min(i.shape[0] for i in imgs.values())
    montage = np.hstack([cv2.resize(i, (int(i.shape[1] * hs / i.shape[0]), hs)) for i in imgs.values()])
    mp = os.path.join(out_dir, f"frame{args.frame:04d}_rig_montage.png")
    cv2.imwrite(mp, cv2.cvtColor(montage, cv2.COLOR_RGB2BGR))
    print(json.dumps({"scene": args.scene, "frame": args.frame, "cameras": rows,
                      "montage": mp, "out_dir": out_dir}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
