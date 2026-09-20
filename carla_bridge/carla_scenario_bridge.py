#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DGGT corner-case 场景 -> CARLA 闭环仿真 / 可视化桥接。

输入（二选一，推荐 .world.json，信息最全）：
    --scenario output/export/rear_end_s7b/rear-end_s7.world.json
    --scenario output/export/rear_end_s7b/rear-end_s7.xosc

做的事：
  1. 解析 DGGT 导出的世界轨迹（y-up：x 右 / y 上 / z 前，米，10 Hz）；
  2. 把整段场景刚体对齐到 CARLA 内置地图上的一条**直路**（ego 首位姿 == 道路 spawn 点），
     DGGT 世界系 -> CARLA 左手系：loc_carla = (z, x, y)，yaw_carla = yaw_dggt；
  3. 按尺寸给每个交通参与者挑 CARLA 蓝图（车 / 行人），按轨迹回放；
     ego 也可以交给 CARLA 自己开（--ego-mode autopilot，真正的闭环）；
  4. 多机位离屏渲染 -> MP4，同时写每帧位姿 / 事件（碰撞、最小距离、TTC）JSON；
  5. 可选：把 CARLA 自带 spectator 跟着 ego（配合 x11vnc/noVNC 看实时窗口），
     以及内置 MJPEG 网页直播（不需要 X，浏览器直接看）。

典型用法：
    # 只解析，不连 CARLA（自检）
    python carla_scenario_bridge.py --scenario xx.world.json --dry-run

    # 连本机 CARLA server，录制 MP4
    python carla_scenario_bridge.py --scenario xx.world.json \
        --out-dir output/carla/rear_end_s7b --map Town10HD_Opt --fps 20 --mkv

    # 网页实时看（浏览器开 http://127.0.0.1:8090/）
    python carla_scenario_bridge.py --scenario xx.world.json --live-http 8090 --loop 0
"""
from __future__ import annotations

import argparse
import json
import math
import os
import queue
import sys
import threading
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import carla
except Exception:                                    # noqa: BLE001
    carla = None

try:
    import cv2
except Exception:                                    # noqa: BLE001
    cv2 = None


# --------------------------------------------------------------------------- #
#  数据模型
# --------------------------------------------------------------------------- #
@dataclass
class Track:
    """一条交通参与者轨迹（DGGT 世界系，y-up）。"""
    tid: int
    role: str
    is_ego: bool
    length: float
    width: float
    height: float
    times: np.ndarray        # (N,) 秒（全场景统一 0 基准）
    locs: np.ndarray         # (N,3) DGGT 世界系 x右/y上/z前
    yaws: np.ndarray         # (N,) rad，绕 +y
    speed: float = 0.0

    @property
    def kind(self) -> str:
        L, W, H = self.length, self.width, self.height
        if 0.30 <= L <= 1.0 and 0.40 <= W <= 1.1 and 1.1 <= H <= 2.2:
            return "pedestrian"
        if 1.2 <= W <= 3.4 and 1.4 <= L <= 14.0 and 0.7 <= H <= 4.8:
            return "vehicle"
        return "other"

    @property
    def t_end(self) -> float:
        return float(self.times[-1])


def _yaw_from_matrix(m: np.ndarray) -> float:
    return math.atan2(float(m[0, 2]), float(m[2, 2]))


def load_world_json(path: str) -> Dict[str, Any]:
    d = json.load(open(path, "r", encoding="utf-8"))
    dt = float(d.get("frame", {}).get("dt", 0.1))
    raw = d["tracks"]
    f0 = min(int(f) for t in raw for f in t["poses"])
    tracks: List[Track] = []
    for t in raw:
        L, W, H = [float(x) for x in t.get("dimensions_lwh", [4.5, 2.0, 1.6])]
        frames = sorted(int(k) for k in t["poses"])
        times, locs, yaws = [], [], []
        for f in frames:
            m = np.asarray(t["poses"][str(f)], dtype=np.float64).reshape(4, 4)
            times.append((f - f0) * dt)
            locs.append([m[0, 3], m[1, 3], m[2, 3]])
            yaws.append(_yaw_from_matrix(m))
        tracks.append(Track(tid=int(t["track_id"]), role=str(t.get("role", "actor")),
                            is_ego=bool(t.get("is_ego", False)), length=L, width=W, height=H,
                            times=np.asarray(times, float), locs=np.asarray(locs, float),
                            yaws=np.asarray(yaws, float), speed=float(t.get("speed_mps", 0.0))))
    return {"name": os.path.splitext(os.path.basename(path))[0], "dt": dt, "tracks": tracks,
            "source": path}


def load_xosc(path: str) -> Dict[str, Any]:
    """解析本工程 scenario_export.py 导出的 OpenSCENARIO 1.2。

    OSC 世界坐标映射（见导出脚本 FileHeader 说明）：
        osc.x = dggt.x(右)   osc.y = dggt.z(前)   osc.z = dggt.y(上)，h = dggt yaw
    于是 CARLA 坐标: loc = (osc.y, osc.x, osc.z)，yaw = h。
    """
    root = ET.parse(path).getroot()
    dims: Dict[str, Tuple[float, float, float]] = {}
    for so in root.iter("ScenarioObject"):
        nm = so.get("name")
        for tag in ("Vehicle", "vehicle", "Pedestrian", "pedestrian"):
            el = so.find(tag)
            if el is None:
                continue
            bb = el.find("BoundingBox/Dimensions")
            if bb is not None:
                dims[nm] = (float(bb.get("length", 4.5)), float(bb.get("width", 2.0)),
                            float(bb.get("height", 1.6)))
    tracks: List[Track] = []
    for traj in root.iter("Trajectory"):
        nm = (traj.get("name") or "").replace("traj_", "")
        pl = traj.find("Polyline")
        if pl is None:
            continue
        times, locs, yaws = [], [], []
        for v in pl.findall("Vertex"):
            times.append(float(v.get("time", 0.0)))
            locs.append([float(v.get("x", 0.0)), float(v.get("z", 0.0)), float(v.get("y", 0.0))])
            yaws.append(float(v.get("h", 0.0)))
        if not times:
            continue
        is_ego = nm == "ego"
        tid = 900000 if is_ego else int(nm.split("_")[-1]) if nm.startswith("actor_") else hash(nm) % 100000
        L, W, H = dims.get(nm, (4.5, 2.0, 1.6))
        spd = 0.0
        if len(times) >= 2:
            spd = float(np.linalg.norm(np.asarray(locs[-1]) - np.asarray(locs[0]))
                        / max(1e-3, times[-1] - times[0]))
        tracks.append(Track(tid=tid, role="ego" if is_ego else "actor", is_ego=is_ego,
                            length=L, width=W, height=H,
                            times=np.asarray(times, float), locs=np.asarray(locs, float),
                            yaws=np.asarray(yaws, float), speed=spd))
    if not tracks:
        raise RuntimeError(f"没有从 {path} 解析到任何轨迹")
    return {"name": os.path.splitext(os.path.basename(path))[0], "dt": 0.1, "tracks": tracks,
            "source": path}


def load_scenario(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    if path.endswith(".json"):
        return load_world_json(path)
    return load_xosc(path)


def _is_ego_track(t: Track) -> bool:
    """世界 json 里有时 ego 没标 is_ego（只有 tid==900000 / role=='ego'），统一判断。"""
    return bool(t.is_ego) or int(t.tid) == 900000 or t.role == "ego"


def select_tracks(tracks: Sequence[Track], mode: str, min_poses: int, max_actors: int) -> List[Track]:
    core = {t.tid for t in tracks if _is_ego_track(t) or t.role in ("ego", "victim", "attacker")}
    out: List[Track] = []
    for t in tracks:
        keep = False
        if mode == "raw":
            keep = True
        elif mode == "all":
            keep = t.kind in ("vehicle", "pedestrian")
        elif mode == "dynamic":
            keep = t.tid in core or (t.kind in ("vehicle", "pedestrian") and len(t.times) >= min_poses)
        else:                                          # core
            keep = t.tid in core
        if t.tid in core:
            keep = True
        if keep:
            out.append(t)
    # ego 一定要排最前面先 spawn，否则它那格会被别的车占了导致生成失败
    out.sort(key=lambda t: (not _is_ego_track(t), not (t.role in ("victim", "attacker")), -t.t_end))
    if max_actors > 0:
        keep_ids = {t.tid for t in out[:max_actors]}
        keep_ids |= core
        out = [t for t in out if t.tid in keep_ids]
    return out


# --------------------------------------------------------------------------- #
#  采样 / 对齐
# --------------------------------------------------------------------------- #
def sample_track(t: Track, tnow: float) -> Optional[Tuple[np.ndarray, float]]:
    """线性插值采样；tnow 早于轨迹起点返回 None。"""
    if tnow < t.times[0] - 1e-9:
        return None
    if tnow >= t.times[-1]:
        return t.locs[-1].copy(), float(t.yaws[-1])
    i = int(np.searchsorted(t.times, tnow, side="right")) - 1
    i = max(0, min(i, len(t.times) - 2))
    t0, t1 = float(t.times[i]), float(t.times[i + 1])
    a = 0.0 if t1 <= t0 else (tnow - t0) / (t1 - t0)
    loc = t.locs[i] * (1 - a) + t.locs[i + 1] * a
    y0, y1 = float(t.yaws[i]), float(t.yaws[i + 1])
    dy = (y1 - y0 + math.pi) % (2 * math.pi) - math.pi
    return loc, y0 + a * dy


def dggt_to_carla(loc: np.ndarray, yaw: float) -> Tuple[np.ndarray, float]:
    """DGGT(y-up, x右/y上/z前) -> CARLA(x前/y右/z上)。"""
    return np.array([loc[2], loc[0], loc[1]], dtype=np.float64), float(yaw)


def estimate_ground_offset(tracks: Sequence[Track], ego: Optional[Track]) -> float:
    """垂直方向参考：默认用 ego 首帧高度，其余车据此平移到同一地面。"""
    if ego is not None:
        return float(ego.locs[0][1])
    ys = [float(t.locs[:, 1].min()) for t in tracks if len(t.times)]
    return float(np.median(ys)) if ys else 0.0


class Aligner:
    """把整段场景刚体对齐到地图上的 anchor（位置 + yaw）。"""

    def __init__(self, anchor_loc: np.ndarray, anchor_yaw: float,
                 ego_world_loc: np.ndarray, ego_world_yaw: float, vertical: str = "flat",
                 ground_ref: float = 0.0):
        self.anchor = np.asarray(anchor_loc, float)
        self.dyaw = float(anchor_yaw) - float(ego_world_yaw)
        self.ca, self.sa = math.cos(self.dyaw), math.sin(self.dyaw)
        self.p0 = np.asarray(ego_world_loc, float)
        self.vertical = vertical
        self.ground_ref = float(ground_ref)

    def __call__(self, loc: np.ndarray, yaw: float) -> Tuple[np.ndarray, float]:
        rel = np.asarray(loc, float) - self.p0
        x = rel[0] * self.ca - rel[1] * self.sa
        y = rel[0] * self.sa + rel[1] * self.ca
        if self.vertical == "flat":
            dz = 0.0
        elif self.vertical == "relative":
            dz = rel[1]
        else:                                            # center
            dz = rel[1]
        return self.anchor + np.array([x, y, dz]), float(yaw) + self.dyaw


def wrap_pi(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def _load_world_guarded(client, name: str, timeout: float = 150.0):
    """load_world 带硬超时：CARLA server 中途被杀时这个调用会永远卡住（作业也就永远 running）。"""
    box: Dict[str, Any] = {}

    def _work():
        try:
            box["w"] = client.load_world(name)
        except Exception as e:                            # noqa: BLE001
            box["e"] = e

    th = threading.Thread(target=_work, daemon=True)
    th.start()
    th.join(timeout)
    if th.is_alive():
        raise TimeoutError(f"load_world('{name}') 超过 {timeout:.0f}s 没返回（CARLA server 卡住/被杀？）")
    if "e" in box:
        raise box["e"]
    return box.get("w")


def pick_free_port(start: int, tries: int = 25) -> int:
    """挑一个没人监听的端口。

    CARLA 的 TrafficManager 是**在客户端进程里**起一个监听线程的：端口被占时会直接
    C++ abort（进程 rc=-6 / SIGABRT）。所以 autopilot 用的 TM 端口必须动态选。
    """
    import socket as _socket
    for p in range(start, start + tries):
        try:
            with _socket.socket() as s:
                s.settimeout(0.15)
                if s.connect_ex(("127.0.0.1", p)) != 0:
                    return p
        except Exception:                                # noqa: BLE001
            return p
    return start


def snap_to_road(world, loc: np.ndarray, mode: str, max_dist: float = 0.0) -> np.ndarray:
    """把 actor 贴回路面，解决"车飘在空中 / 不在车道里"。

    mode:
      none  —— 不动
      z     —— 只按最近道路的高度抬到路面（保守，不改平面位置；任何路面类型都行）
      lane  —— 连 x/y 一起吸到最近的**行车道**中心线（看起来规规矩矩在路上，但会改动事故几何）

    注意：lane 模式用 LaneType.Driving（不是 Any）——用 Any 会吸到人行道/停车带，
    车看着仍然"不在车道里"（实测差 2~8m）。lane 的 max_dist 也放宽到 30m，
    否则偏得远的车干脆不被处理、继续飘在路外。
    """
    if mode == "none":
        return loc
    try:
        lane_type = carla.LaneType.Driving if mode == "lane" else carla.LaneType.Any
        if max_dist <= 0:
            max_dist = 30.0 if mode == "lane" else 14.0
        mp = world.get_map()
        q = carla.Location(float(loc[0]), float(loc[1]), float(loc[2]))
        wp = mp.get_waypoint(q, project_to_road=True, lane_type=lane_type)
        if wp is None:
            return loc
        w = wp.transform.location
        d = math.hypot(w.x - loc[0], w.y - loc[1])
        if d > max_dist:
            return loc
        if mode == "lane":
            return np.array([w.x, w.y, w.z], dtype=np.float64)
        return np.array([loc[0], loc[1], w.z], dtype=np.float64)
    except Exception:                                    # noqa: BLE001
        return loc


def _indent_xml(elem, level: int = 0) -> None:
    """给 py3.7 用的 ET.indent 替代（3.9+ 有内置的）。"""
    pad = "\n" + "  " * level
    if len(elem):
        if not (elem.text or "").strip():
            elem.text = pad + "  "
        for child in elem:
            _indent_xml(child, level + 1)
        if not (elem.tail or "").strip():
            elem.tail = pad
        if not (elem[-1].tail or "").strip():
            elem[-1].tail = pad
    elif level and not (elem.tail or "").strip():
        elem.tail = pad


def speed_at(t: Track, tnow: float) -> float:
    """由轨迹（而不是物理速度）估算 tnow 时刻的速度，物理关掉时也准。"""
    if len(t.times) < 2:
        return 0.0
    if tnow <= t.times[0]:
        i = 0
    elif tnow >= t.times[-1]:
        i = len(t.times) - 2
    else:
        i = int(np.searchsorted(t.times, tnow, side="right")) - 1
    dt = float(t.times[i + 1] - t.times[i])
    if dt <= 1e-6:
        return 0.0
    return float(np.linalg.norm(t.locs[i + 1] - t.locs[i]) / dt)


# --------------------------------------------------------------------------- #
#  CARLA 蓝图选择
# --------------------------------------------------------------------------- #
_VEH_CANDIDATES = {
    "truck": ["vehicle.carlamotors.firetruck", "vehicle.mitsubishi.fusorosa",
              "vehicle.ford.ambulance", "vehicle.carlamotors.carlacola"],
    "van": ["vehicle.volkswagen.t2", "vehicle.ford.crown", "vehicle.nissan.patrol",
            "vehicle.tesla.cybertruck", "vehicle.jeep.wrangler_rubicon"],
    "suv": ["vehicle.nissan.patrol", "vehicle.tesla.model3", "vehicle.audi.etron",
            "vehicle.bmw.grandtourer", "vehicle.jeep.wrangler_rubicon"],
    "sedan": ["vehicle.tesla.model3", "vehicle.audi.a2", "vehicle.dodge.charger_2020",
              "vehicle.lincoln.mkz_2020", "vehicle.seat.leon"],
    "compact": ["vehicle.audi.a2", "vehicle.mini.cooper_s", "vehicle.smart",
                "vehicle.nissan.micra", "vehicle.micro.microlino"],
    "bike": ["vehicle.bh.crossbike", "vehicle.kawasaki.ninja", "vehicle.yamaha.yzf",
             "vehicle.harley-davidson.low_rider", "vehicle.vespa.zx125"],
}
_VEH_FALLBACK = ["vehicle.tesla.model3", "vehicle.audi.a2", "vehicle.nissan.micra"]


def vehicle_class(L: float, W: float, H: float) -> str:
    if W < 1.1 and L < 2.6:
        return "bike"
    if L >= 6.5 or (W >= 2.2 and L >= 5.0):
        return "truck"
    if L >= 5.0:
        return "van"
    if L >= 4.2 or W >= 1.85:
        return "suv"
    if L >= 3.2:
        return "sedan"
    return "compact"


class BlueprintPicker:
    def __init__(self, world):
        self.lib = world.get_blueprint_library()
        self._cache: Dict[str, Any] = {}
        self._walkers = list(self.lib.filter("walker.pedestrian.*"))
        self._wheels = {b.id: b for b in self.lib.filter("vehicle.*")}

    def _find(self, name: str):
        if name in self._cache:
            return self._cache[name]
        bp = None
        try:
            bp = self.lib.find(name)
            if bp is not None and not bp.id.startswith(("vehicle.", "walker.")):
                bp = None
        except Exception:                                # noqa: BLE001
            bp = None
        self._cache[name] = bp
        return bp

    def pick(self, t: Track):
        if t.kind == "pedestrian" and self._walkers:
            return self._walkers[t.tid % len(self._walkers)]
        order = _VEH_CANDIDATES.get(vehicle_class(t.length, t.width, t.height), [])
        for name in list(order) + _VEH_FALLBACK:
            bp = self._find(name)
            if bp is not None:
                return bp
        if self._wheels:
            return sorted(self._wheels.values(), key=lambda b: b.id)[0]
        return None


# --------------------------------------------------------------------------- #
#  找一条直路做 anchor
# --------------------------------------------------------------------------- #
def _straight_score(wp, steps: int = 10, step: float = 4.0) -> Tuple[float, float]:
    """沿车道往前 steps 步，返回 (最大 yaw 偏差 deg, 总长度 m)。"""
    yaw0 = math.radians(wp.transform.rotation.yaw)
    cur = wp
    total = 0.0
    worst = 0.0
    for _ in range(steps):
        nxt = cur.next(step)
        if not nxt:
            break
        cur = nxt[0]
        total += step
        worst = max(worst, abs(wrap_pi(math.radians(cur.transform.rotation.yaw) - yaw0)))
    return math.degrees(worst), total


def find_straight_anchor(world, want_len: float = 30.0, index: int = 0,
                         max_dev_deg: float = 4.0) -> Tuple[np.ndarray, float, Dict[str, Any]]:
    mp = world.get_map()
    cands = []
    for sp in mp.get_spawn_points():
        wp = mp.get_waypoint(sp.location, project_to_road=False)
        if wp is None:
            continue
        if wp.is_junction:
            continue
        dev, length = _straight_score(wp)
        if length >= want_len and dev <= max_dev_deg:
            cands.append((dev, -length, sp, wp))
    cands.sort(key=lambda c: (c[0], c[1]))
    if not cands:
        sps = mp.get_spawn_points()
        if not sps:
            raise RuntimeError("地图没有 spawn point")
        sp = sps[0]
        return np.array([sp.location.x, sp.location.y, sp.location.z]), float(sp.rotation.yaw), \
            {"fallback": True}
    dev, neglen, sp, wp = cands[min(index, len(cands) - 1)]
    info = {"deviation_deg": round(dev, 2), "straight_len_m": round(-neglen, 1),
            "candidates": len(cands), "road_id": wp.road_id, "lane_id": wp.lane_id}
    return (np.array([sp.location.x, sp.location.y, sp.location.z]),
            float(sp.rotation.yaw), info)


# --------------------------------------------------------------------------- #
#  摄像机 / 视频 / MJPEG
# --------------------------------------------------------------------------- #
class MjpegServer(threading.Thread):
    """极简 MJPEG 网页直播（不需要 X11，浏览器直接看）。"""

    def __init__(self, port: int, quality: int = 80):
        super().__init__(daemon=True)
        self.port = port
        self.quality = quality
        self.latest: Optional[bytes] = None
        self.lock = threading.Lock()
        self.count = 0
        self._httpd = None

    def push(self, bgr: np.ndarray) -> None:
        if cv2 is None:
            return
        ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), self.quality])
        if ok:
            with self.lock:
                self.latest = buf.tobytes()
                self.count += 1

    def run(self) -> None:                              # pragma: no cover
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

        outer = self

        class H(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.0"

            def log_message(self, *a):                  # noqa: D102
                pass

            def do_GET(self):                           # noqa: N802
                if self.path.startswith("/stream"):
                    self.send_response(200)
                    self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                    self.end_headers()
                    last = -1
                    while True:
                        with outer.lock:
                            data, n = outer.latest, outer.count
                        if data is None or n == last:
                            time.sleep(0.01)
                            continue
                        last = n
                        try:
                            self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                             + str(len(data)).encode() + b"\r\n\r\n" + data + b"\r\n")
                        except Exception:               # noqa: BLE001
                            return
                elif self.path in ("/", "/index.html"):
                    html = ("<html><head><title>DGGT x CARLA live</title></head>"
                            "<body style='margin:0;background:#111'>"
                            "<img src='/stream' style='width:100%;height:auto'></body></html>")
                    b = html.encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(b)))
                    self.end_headers()
                    self.wfile.write(b)
                elif self.path == "/health":
                    b = b"ok"
                    self.send_response(200)
                    self.send_header("Content-Length", "2")
                    self.end_headers()
                    self.wfile.write(b)
                else:
                    self.send_error(404)

        self._httpd = ThreadingHTTPServer(("0.0.0.0", self.port), H)
        self._httpd.serve_forever()


def look_at(cam_loc: np.ndarray, target: np.ndarray):
    d = np.asarray(target, float) - np.asarray(cam_loc, float)
    yaw = math.degrees(math.atan2(d[1], d[0]))
    pitch = math.degrees(math.atan2(d[2], math.hypot(d[0], d[1])))
    return carla.Transform(carla.Location(float(cam_loc[0]), float(cam_loc[1]), float(cam_loc[2])),
                           carla.Rotation(pitch=pitch, yaw=yaw, roll=0.0))


# --------------------------------------------------------------------------- #
#  主流程
# --------------------------------------------------------------------------- #
class Bridge:
    def __init__(self, args):
        self.a = args
        self.sc = load_scenario(args.scenario)
        self.tracks: List[Track] = []
        self.ego: Optional[Track] = None
        self.actors: Dict[int, Any] = {}
        self.collisions: List[Dict[str, Any]] = []
        self.frames_log: List[Dict[str, Any]] = []
        self.sensors: List[Any] = []
        self.cameras: Dict[str, Any] = {}
        self.cam_queues: Dict[str, queue.Queue] = {}
        self.mjpeg = None
        self.world = None
        self.client = None
        self.tm = None
        self.writer = None
        self.n_frames = 0
        self.video_path = None
        self.cam_tf: Dict[str, Any] = {}
        self.cam_fov: Dict[str, float] = {}
        self.track_by_id: Dict[int, Track] = {}
        self.focus_ids: List[int] = []

    # ---------------- 准备 ----------------
    def prepare_tracks(self) -> None:
        a = self.a
        if getattr(a, "name", None):
            self.sc["name"] = str(a.name)
        self.tracks = select_tracks(self.sc["tracks"], a.actors, a.min_poses, a.max_actors)
        egos = [t for t in self.tracks if _is_ego_track(t)]
        if not egos:
            egos = sorted(self.tracks, key=lambda t: -t.t_end)[:1]
        self.ego = egos[0] if egos else None
        if self.ego is None:
            raise RuntimeError("场景里找不到 ego")
        self.t_start = min(t.times[0] for t in self.tracks)
        self.t_end = max(t.t_end for t in self.tracks)
        self.dur = max(0.1, self.t_end - self.t_start)
        # 相机跟随目标：默认"事故中心"（victim/attacker），否则 ego
        self.track_by_id = {t.tid: t for t in self.tracks}
        crash_roles = [t for t in self.tracks if t.role in ("victim", "attacker")]
        if a.focus == "crash" and crash_roles:
            self.focus_ids = [t.tid for t in crash_roles]
        elif a.focus == "all":
            self.focus_ids = [t.tid for t in self.tracks]
        else:
            self.focus_ids = [self.ego.tid]

    def describe(self) -> None:
        print(f"[scenario] {self.sc['name']}  来源={self.sc['source']}")
        print(f"           原始 track {len(self.sc['tracks'])} 条，选用 {len(self.tracks)} 条，"
              f"时长 {self.dur:.2f}s @ {1.0 / self.sc['dt']:.0f}Hz")
        print(f"           ego = track {self.ego.tid} (role={self.ego.role})")
        print(f"           相机跟随 = {self.a.focus} -> tracks {self.focus_ids}")
        print(f"           {'tid':>7} {'role':>8} {'kind':>10} {'L x W x H':>20} {'帧数':>5} {'速度':>7}")
        for t in self.tracks:
            print(f"           {t.tid:>7} {t.role:>8} {t.kind:>10} "
                  f"{t.length:>6.2f} x {t.width:>5.2f} x {t.height:>5.2f} "
                  f"{len(t.times):>5} {t.speed:>6.2f}m/s")

    # ---------------- CARLA ----------------
    def connect(self) -> None:
        a = self.a
        self.client = carla.Client(a.host, a.port)
        self.client.set_timeout(a.timeout)
        if not a.map:
            self.world = self.client.get_world()
            print(f"[carla] 用当前地图 {self.world.get_map().name}")
            return
        # --reload-world auto：已经在该地图上就别再 load_world（批处理能省 20s/场景）
        cur = ""
        if a.reload_world == "auto":
            try:
                cur = self.client.get_world().get_map().name
            except Exception:                            # noqa: BLE001
                cur = ""
            if a.map.lower() in cur.lower():
                self.world = self.client.get_world()
                print(f"[carla] 已在地图 {cur}，跳过 load_world")
                return
        print(f"[carla] load_world('{a.map}') ...")
        last = None
        for attempt in range(max(1, a.load_retries)):
            try:
                self.world = _load_world_guarded(self.client, a.map)
                last = None
                break
            except Exception as e:                       # noqa: BLE001
                last = e
                print(f"[carla] load_world 第 {attempt + 1} 次失败：{e}")
                time.sleep(5)
        if last is not None:
            raise RuntimeError(f"load_world('{a.map}') 失败：{last}")
        print(f"[carla] 地图 = {self.world.get_map().name}")

    def setup_world(self) -> None:
        a = self.a
        s = self.world.get_settings()
        self._old_settings = s
        s.synchronous_mode = True
        s.fixed_delta_seconds = 1.0 / a.fps
        s.no_rendering_mode = False
        self.world.apply_settings(s)
        if a.tm_port > 0:
            port = pick_free_port(a.tm_port)
            if port != a.tm_port:
                print(f"[carla] TM 端口 {a.tm_port} 被占用，改用 {port}")
            self.tm_port_used = port
            self.tm = self.client.get_trafficmanager(port)
            self.tm.set_synchronous_mode(True)
            self.tm.set_global_distance_to_leading_vehicle(2.5)
        if a.start_recorder:
            self.client.start_recorder(a.start_recorder, True)
            print(f"[carla] recorder -> {a.start_recorder}")
        # 固定天气（默认正午晴天，画面亮、清楚；否则不同地图/次运行可能是夜里）
        if a.weather and a.weather.lower() not in ("none", "keep", ""):
            try:
                params = getattr(carla.WeatherParameters, a.weather, None) \
                    or carla.WeatherParameters.ClearNoon
                self.world.set_weather(params)
                print(f"[carla] weather = {a.weather}")
            except Exception as e:                       # noqa: BLE001
                print(f"[carla] 设置天气失败：{e}")

    def compute_alignment(self) -> None:
        a = self.a
        if a.anchor:
            vals = [float(x) for x in a.anchor.split(",")]
            if len(vals) != 4:
                raise ValueError("--anchor 需要 x,y,z,yaw")
            anchor_loc = np.array(vals[:3]); anchor_yaw = vals[3]
            info = {"manual": True}
        else:
            anchor_loc, anchor_yaw, info = find_straight_anchor(self.world, index=a.anchor_index)
        ego_loc, ego_yaw = dggt_to_carla(self.ego.locs[0], self.ego.yaws[0])
        gref = estimate_ground_offset(self.tracks, self.ego)
        self.aligner = Aligner(anchor_loc, anchor_yaw, ego_loc, ego_yaw,
                               vertical=a.vertical, ground_ref=gref)
        print(f"[align] anchor loc={np.round(anchor_loc, 2).tolist()} yaw={anchor_yaw:.2f}  {info}")
        print(f"[align] ego 起点 -> {np.round(anchor_loc, 2).tolist()}, "
              f"整场景旋转 {math.degrees(self.aligner.dyaw):.2f}°，垂直模式={a.vertical}")

    def spawn_actors(self) -> None:
        a = self.a
        picker = BlueprintPicker(self.world)
        self.ego_actor = None
        for t in self.tracks:
            s = sample_track(t, self.t_start)
            if s is None:
                continue
            loc, yaw = self.aligner(*dggt_to_carla(s[0], s[1]))
            if getattr(a, "road_snap", "z") != "none":
                loc = snap_to_road(self.world, loc, a.road_snap)
            bp = picker.pick(t)
            if bp is None:
                continue
            try:
                if bp.has_attribute("role_name"):
                    bp.set_attribute("role_name", f"dggt_{t.role}_{t.tid}")
            except Exception:                            # noqa: BLE001
                pass
            tf = carla.Transform(carla.Location(float(loc[0]), float(loc[1]), float(loc[2]) + 0.15),
                                 carla.Rotation(yaw=math.degrees(yaw)))
            actor = None
            cands = [(0.0, 0.0), (0.3, 0.0), (0.8, 0.0), (1.5, 0.0), (2.5, 0.0), (4.0, 0.0)]
            if not _is_ego_track(t):
                # 非 ego 允许稍微挪一点位置，避免和已生成的车重叠
                cands += [(0.3, 0.8), (0.3, -0.8), (1.0, 1.6), (1.0, -1.6),
                          (1.2, 3.0), (1.2, -3.0)]
            last_err = ""
            for dz, lat in cands:
                tf.location.z = float(loc[2]) + 0.15 + dz
                if lat:
                    tf.location.x = float(loc[0]) + lat * math.sin(yaw)
                    tf.location.y = float(loc[1]) - lat * math.cos(yaw)
                try:
                    actor = self.world.spawn_actor(bp, tf)
                    if dz or lat:
                        print(f"        [~] tid={t.tid} 原地 spawn 撞车，已挪 高度{dz:+.1f}m/侧向{lat:+.1f}m")
                    break
                except RuntimeError as e:
                    last_err = str(e)
                    continue
            if actor is None:
                print(f"        [!] spawn 失败 tid={t.tid} kind={t.kind} role={t.role}：{last_err}")
                continue
            self.actors[t.tid] = actor
            if _is_ego_track(t):
                self.ego_actor = actor
            # autopilot 时 ego 必须保留物理，否则 TrafficManager 驱动一个"无物理"的车会崩
            _ego_autopilot = _is_ego_track(t) and a.ego_mode == "autopilot"
            if a.physics_off and not _ego_autopilot and hasattr(actor, "set_simulate_physics"):
                try:
                    actor.set_simulate_physics(False)
                except Exception:                        # noqa: BLE001
                    pass
            self._attach_collision(actor, t)
        print(f"[carla] 已生成 {len(self.actors)} 个 actor"
              f"（ego={getattr(self.ego_actor, 'id', None)}）")

    def _attach_collision(self, actor, t: Track) -> None:
        if t.kind == "pedestrian" or self.a.no_collision_sensors:
            return
        bp = self.world.get_blueprint_library().find("sensor.other.collision")
        try:
            sen = self.world.spawn_actor(bp, carla.Transform(), attach_to=actor)
        except Exception:                                # noqa: BLE001
            return
        tid = t.tid

        def cb(ev, tid=tid):
            other = ev.other_actor
            self.collisions.append({
                "frame": self.n_frames, "t": round(self._t_now, 3), "actor": tid,
                "other_actor": getattr(other, "id", None),
                "other_type": getattr(other, "type_id", None),
                "impulse": round(float(np.linalg.norm(
                    [ev.normal_impulse.x, ev.normal_impulse.y, ev.normal_impulse.z])), 2)})
        sen.listen(cb)
        self.sensors.append(sen)

    def setup_cameras(self) -> None:
        a = self.a
        self.cam_specs = [c for c in a.cameras.split(",") if c.strip()]
        w, h = [int(x) for x in a.cam_size.lower().split("x")]
        self.cam_w, self.cam_h = w, h
        for name in self.cam_specs:
            bp = self.world.get_blueprint_library().find("sensor.camera.rgb")
            fov = 70.0 if name == "birdseye" else 100.0
            try:
                bp.set_attribute("image_size_x", str(w))
                bp.set_attribute("image_size_y", str(h))
                bp.set_attribute("fov", str(fov))
                bp.set_attribute("sensor_tick", "0.0")
            except Exception:                            # noqa: BLE001
                pass
            self.cam_fov[name] = fov
            sen = self.world.spawn_actor(bp, carla.Transform(
                carla.Location(0, 0, 50)), attach_to=None)
            q: queue.Queue = queue.Queue(maxsize=4)
            sen.listen(lambda img, q=q: self._on_image(q, img))
            self.cameras[name] = sen
            self.cam_queues[name] = q
            self.sensors.append(sen)
        print(f"[carla] 相机 = {self.cam_specs}  ({w}x{h})")

    def _on_image(self, q: queue.Queue, img) -> None:
        try:
            q.put_nowait(img)
        except queue.Full:
            try:
                q.get_nowait()
                q.put_nowait(img)
            except Exception:                            # noqa: BLE001
                pass

    def write_carla_xosc(self, out_path: str) -> str:
        """导出「CARLA 对齐版」OpenSCENARIO 1.0，给 ScenarioRunner / 其它标准工具用。

        和 studio/backend 原始导出的区别：
          1. 坐标已经过落图对齐，并且是 CARLA 坐标系（x 前/y 右/z 上）；
          2. FileHeader.description 以 "CARLA:" 开头 -> ScenarioRunner 会用 CARLA 坐标系
             （否则它会把 y 和 yaw 取反）；
          3. LogicFile 写 CARLA 地图名（配合 scenario_runner.py --reloadWorld 能直接加载地图）；
          4. Polyline 的 Vertex 里带 <Position><WorldPosition/>，这是 ScenarioRunner
             parse_trajectory() 要求的结构（原导出把 x/y/z/h 直接写在 Vertex 属性上，
             ScenarioRunner 认不出来）；
          5. 带上 Performance / Axles / Properties，符合 OSC 1.0 的 Vehicle 结构。
        """
        picker = BlueprintPicker(self.world)
        name = f"{os.path.splitext(os.path.basename(self.sc['name']))[0]}_carla"
        out_path = out_path or os.path.join(self.a.out_dir or ".", name + ".xosc")
        os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)

        root = ET.Element("OpenSCENARIO")
        ET.SubElement(root, "FileHeader", {
            "revMajor": "1", "revMinor": "0", "date": "2026-01-01T00:00:00",
            "description": f"CARLA:{name}", "author": "DGGT studio / carla_bridge"})
        ET.SubElement(root, "ParameterDeclarations")
        ET.SubElement(root, "CatalogLocations")
        rn = ET.SubElement(root, "RoadNetwork")
        ET.SubElement(rn, "LogicFile", {"filepath": self.a.map or "Town10HD_Opt"})
        ET.SubElement(rn, "SceneGraphFile", {"filepath": ""})

        def aligned(t: Track, tnow: float):
            s = sample_track(t, tnow)
            if s is None:
                return None
            loc, yaw = self.aligner(*dggt_to_carla(s[0], s[1]))
            return loc, yaw

        # ---- Entities ----
        ents = ET.SubElement(root, "Entities")
        ent_names = {}
        for t in self.tracks:
            nm = "ego" if _is_ego_track(t) else f"actor_{t.tid}"
            ent_names[t.tid] = nm
            bp = picker.pick(t)
            bp_id = bp.id if bp is not None else ("vehicle.tesla.model3" if t.kind != "pedestrian"
                                                  else "walker.pedestrian.0001")
            so = ET.SubElement(ents, "ScenarioObject", {"name": nm})
            if t.kind == "pedestrian":
                veh = ET.SubElement(so, "Pedestrian", {
                    "name": bp_id, "model": "walker", "pedestrianCategory": "pedestrian",
                    "mass": "80"})
            else:
                veh = ET.SubElement(so, "Vehicle", {
                    "name": bp_id, "vehicleCategory": "car"})
            ET.SubElement(veh, "ParameterDeclarations")
            if t.kind != "pedestrian":
                ET.SubElement(veh, "Performance", {"maxSpeed": "69.444", "maxAcceleration": "200",
                                                   "maxDeceleration": "10.0"})
            bbox = ET.SubElement(veh, "BoundingBox")
            ET.SubElement(bbox, "Center", {"x": "0.0", "y": "0.0", "z": str(round(t.height / 2, 3))})
            ET.SubElement(bbox, "Dimensions", {"width": str(round(t.width, 3)),
                                               "length": str(round(t.length, 3)),
                                               "height": str(round(t.height, 3))})
            if t.kind != "pedestrian":
                ax = ET.SubElement(veh, "Axles")
                ET.SubElement(ax, "FrontAxle", {"maxSteering": "0.5", "wheelDiameter": "0.6",
                                                "trackWidth": "1.8",
                                                "positionX": str(round(t.length * 0.35, 2)),
                                                "positionZ": "0.3"})
                ET.SubElement(ax, "RearAxle", {"maxSteering": "0.0", "wheelDiameter": "0.6",
                                               "trackWidth": "1.8", "positionX": "0.0",
                                               "positionZ": "0.3"})
            props = ET.SubElement(veh, "Properties")
            ET.SubElement(props, "Property", {
                "name": "type",
                "value": "ego_vehicle" if _is_ego_track(t) else "simulation"})
            ET.SubElement(props, "Property", {"name": "color", "value":
                                              "0,0,255" if _is_ego_track(t) else "255,0,0"})

        # ---- Init: 每个 actor 的初始位姿 ----
        sb = ET.SubElement(root, "Storyboard")
        init = ET.SubElement(sb, "Init")
        iacts = ET.SubElement(init, "Actions")
        for t in self.tracks:
            r = aligned(t, t.times[0])
            if r is None:
                continue
            loc, yaw = r
            priv = ET.SubElement(iacts, "Private", {"entityRef": ent_names[t.tid]})
            pa = ET.SubElement(priv, "PrivateAction")
            ta = ET.SubElement(pa, "TeleportAction")
            pos = ET.SubElement(ta, "Position")
            ET.SubElement(pos, "WorldPosition", {
                "x": str(round(float(loc[0]), 4)), "y": str(round(float(loc[1]), 4)),
                "z": str(round(float(loc[2]), 4)), "h": str(round(float(yaw), 6)),
                "p": "0.0", "r": "0.0"})

        # ---- Story: 每个 actor 一条 FollowTrajectory ----
        story = ET.SubElement(sb, "Story", {"name": "dggt_trajectories"})
        act = ET.SubElement(story, "Act", {"name": "main"})
        mg = ET.SubElement(act, "ManeuverGroup", {"maximumExecutionCount": "1", "name": "all"})
        actors_el = ET.SubElement(mg, "Actors", {"selectTriggeringEntities": "false"})
        for t in self.tracks:
            ET.SubElement(actors_el, "EntityRef", {"entityRef": ent_names[t.tid]})
        man = ET.SubElement(mg, "Maneuver", {"name": "follow"})
        for t in self.tracks:
            nm = ent_names[t.tid]
            ev = ET.SubElement(man, "Event", {"name": f"follow_{nm}", "priority": "overwrite"})
            ac = ET.SubElement(ev, "Action", {"name": f"follow_{nm}"})
            pa = ET.SubElement(ac, "PrivateAction")
            # OSC 1.0 的 XSD 里 FollowTrajectoryAction 必须包在 RoutingAction 里
            ra = ET.SubElement(pa, "RoutingAction")
            fta = ET.SubElement(ra, "FollowTrajectoryAction")
            traj = ET.SubElement(fta, "Trajectory", {"name": f"traj_{nm}", "closed": "false"})
            ET.SubElement(traj, "ParameterDeclarations")
            shape = ET.SubElement(traj, "Shape")
            poly = ET.SubElement(shape, "Polyline")
            t0 = float(t.times[0])
            for k in range(len(t.times)):
                r = aligned(t, float(t.times[k]))
                if r is None:
                    continue
                loc, yaw = r
                vx = ET.SubElement(poly, "Vertex", {"time": str(round(float(t.times[k]) - t0, 3))})
                vpos = ET.SubElement(vx, "Position")
                ET.SubElement(vpos, "WorldPosition", {
                    "x": str(round(float(loc[0]), 4)), "y": str(round(float(loc[1]), 4)),
                    "z": str(round(float(loc[2]), 4)), "h": str(round(float(yaw), 6)),
                    "p": "0.0", "r": "0.0"})
            tref = ET.SubElement(fta, "TimeReference")
            ET.SubElement(tref, "Timing", {"domainAbsoluteRelative": "absolute", "offset": "0",
                                           "scale": "1"})
            ET.SubElement(fta, "TrajectoryFollowingMode", {"followingMode": "position"})
            st = ET.SubElement(ev, "StartTrigger")
            cg = ET.SubElement(st, "ConditionGroup")
            cond = ET.SubElement(cg, "Condition", {"name": "start", "delay": "0",
                                                   "conditionEdge": "rising"})
            bv = ET.SubElement(cond, "ByValueCondition")
            ET.SubElement(bv, "SimulationTimeCondition", {"value": "0", "rule": "greaterThan"})

        # Act 级 StartTrigger（OSC 1.0 的 XSD 里 Act 必须有：ManeuverGroup+ / StartTrigger / StopTrigger?）
        st_act = ET.SubElement(act, "StartTrigger")
        cg_act = ET.SubElement(st_act, "ConditionGroup")
        cond_act = ET.SubElement(cg_act, "Condition", {"name": "act_start", "delay": "0",
                                                       "conditionEdge": "rising"})
        bv_act = ET.SubElement(cond_act, "ByValueCondition")
        ET.SubElement(bv_act, "SimulationTimeCondition", {"value": "0", "rule": "greaterThan"})

        stop = ET.SubElement(sb, "StopTrigger")
        cgs = ET.SubElement(stop, "ConditionGroup")
        cs = ET.SubElement(cgs, "Condition", {"name": "end", "delay": "0",
                                              "conditionEdge": "rising"})
        bvs = ET.SubElement(cs, "ByValueCondition")
        ET.SubElement(bvs, "SimulationTimeCondition",
                      {"value": str(round(self.dur + 0.5, 3)), "rule": "greaterThan"})

        try:
            ET.indent(root)                                  # py>=3.9
        except Exception:                                    # noqa: BLE001
            _indent_xml(root)
        ET.ElementTree(root).write(out_path, encoding="utf-8", xml_declaration=True)
        return out_path

    # ---------------- 动作 ----------------
    def apply_playback(self, tnow: float) -> None:
        """把所有 actor 摆到 tnow 时刻的轨迹位置（线性插值）。"""
        for t in self.tracks:
            actor = self.actors.get(t.tid)
            if actor is None:
                continue
            if _is_ego_track(t) and self.a.ego_mode == "autopilot":
                continue
            s = sample_track(t, tnow)
            if s is None:
                continue
            loc, yaw = self.aligner(*dggt_to_carla(s[0], s[1]))
            if getattr(self.a, "road_snap", "z") != "none":
                loc = snap_to_road(self.world, loc, self.a.road_snap)
            if os.environ.get("DGGT_DEBUG_LOOP"):
                try:
                    alive = actor.is_alive
                except Exception:                            # noqa: BLE001
                    alive = "?"
                print(f"[dbg]   set_transform tid={t.tid} alive={alive} "
                      f"loc={[round(float(x), 2) for x in loc]}", flush=True)
            actor.set_transform(carla.Transform(
                carla.Location(float(loc[0]), float(loc[1]), float(loc[2]) + 0.15),
                carla.Rotation(yaw=math.degrees(yaw))))

    def ego_state(self) -> Optional[Dict[str, Any]]:
        if not getattr(self, "ego_actor", None):
            return None
        tf = self.ego_actor.get_transform()
        v = self.ego_actor.get_velocity()
        return {"loc": [round(tf.location.x, 3), round(tf.location.y, 3), round(tf.location.z, 3)],
                "yaw": round(tf.rotation.yaw, 2),
                "speed": round(math.sqrt(v.x ** 2 + v.y ** 2 + v.z ** 2), 3)}

    def focus_state(self, tnow: float) -> Optional[Tuple[np.ndarray, float, float]]:
        """返回 (关注点中心, 朝向 yaw, 目标离散度) —— 用来摆相机。"""
        pts, yaws, speeds = [], [], []
        for tid in self.focus_ids:
            t = self.track_by_id.get(tid)
            if t is None:
                continue
            s = sample_track(t, tnow)
            if s is None:
                continue
            loc, yaw = self.aligner(*dggt_to_carla(s[0], s[1]))
            pts.append(loc)
            yaws.append(yaw)
            speeds.append(speed_at(t, tnow))
        if not pts:
            if not getattr(self, "ego_actor", None):
                return None
            tf = self.ego_actor.get_transform()
            return (np.array([tf.location.x, tf.location.y, tf.location.z]),
                    math.radians(tf.rotation.yaw), 0.0)
        pts = np.asarray(pts)
        c = pts.mean(axis=0)
        spread = float(np.linalg.norm(pts[:, :2] - c[:2], axis=1).max()) if len(pts) > 1 else 0.0
        yaw = yaws[int(np.argmax(speeds))] if speeds else yaws[0]
        return c, float(yaw), spread

    def update_cameras(self) -> None:
        fs = self.focus_state(self._t_now) if hasattr(self, "_t_now") else None
        if fs is None:
            return
        center, yaw, spread = fs
        fwd = np.array([math.cos(yaw), math.sin(yaw), 0.0])
        dist = self.a.chase_dist + min(18.0, spread * 1.1)
        height = self.a.chase_height + min(10.0, spread * 0.5)
        bird_h = max(self.a.bird_height, 26.0 + spread * 1.8)
        for name, sen in self.cameras.items():
            if name == "chase":
                cam = center - fwd * dist + np.array([0, 0, height])
                tf = look_at(cam, center + np.array([0, 0, 0.8]))
            elif name == "birdseye":
                cam = center + np.array([0, 0, bird_h])
                tf = look_at(cam, center + fwd * 2.0)
            elif name == "front":
                cam = center + fwd * 1.4 + np.array([0, 0, 1.6])
                tf = look_at(cam, center + fwd * 30.0)
            elif name == "side":
                right = np.array([-math.sin(yaw), math.cos(yaw), 0.0])
                cam = center - fwd * 2.0 + right * (9.0 + spread) + np.array([0, 0, 3.5])
                tf = look_at(cam, center)
            else:
                continue
            sen.set_transform(tf)
            self.cam_tf[name] = tf
        if self.a.spectator and "chase" in self.cameras:
            self.world.get_spectator().set_transform(self.cameras["chase"].get_transform())

    # ---------- 画标注 ----------
    def _project(self, name: str, p: np.ndarray) -> Optional[Tuple[int, int]]:
        tf = self.cam_tf.get(name)
        if tf is None:
            return None
        loc = np.array([tf.location.x, tf.location.y, tf.location.z])
        rot = tf.rotation
        y, p_, r = math.radians(rot.yaw), math.radians(rot.pitch), math.radians(rot.roll)
        fwd = np.array([math.cos(y) * math.cos(p_), math.sin(y) * math.cos(p_), math.sin(p_)])
        right = np.array([-math.sin(y) * math.cos(r) + math.cos(y) * math.sin(p_) * math.sin(r),
                          math.cos(y) * math.cos(r) + math.sin(y) * math.sin(p_) * math.sin(r),
                          -math.cos(p_) * math.sin(r)])
        up = np.cross(right, fwd)
        d = np.asarray(p, float) - loc
        xc, yc, zc = float(d @ fwd), float(d @ right), float(d @ up)
        if xc <= 0.6:
            return None
        f = (self.cam_w / 2.0) / math.tan(math.radians(self.cam_fov.get(name, 90.0)) / 2.0)
        u = int(self.cam_w / 2.0 + f * yc / xc)
        v = int(self.cam_h / 2.0 - f * zc / xc)
        if not (-self.cam_w < u < 2 * self.cam_w and -self.cam_h < v < 2 * self.cam_h):
            return None
        return u, v

    def annotate(self, name: str, img: np.ndarray) -> np.ndarray:
        if cv2 is None:
            return img
        out = img.copy()
        for t in self.tracks:
            s = sample_track(t, self._t_now)
            if s is None:
                continue
            loc, _ = self.aligner(*dggt_to_carla(s[0], s[1]))
            uv = self._project(name, loc + np.array([0, 0, 1.4]))
            if uv is None:
                continue
            u, v = uv
            color = (0, 215, 255) if _is_ego_track(t) else ((0, 0, 255) if t.role == "attacker"
                                                   else ((0, 200, 0) if t.role == "victim" else (255, 200, 0)))
            cv2.circle(out, (u, v), 5, color, -1)
            cv2.circle(out, (u, v), 12, color, 1)
            spd = speed_at(t, self._t_now)
            label = f"{t.role}#{t.tid} {spd:.1f}m/s"
            cv2.putText(out, label, (u + 14, v - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(out, label, (u + 14, v - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
        m = self.metrics()
        hud = (f"t={self._t_now:5.2f}s  minDist={m.get('min_dist')}m  "
               f"minTTC={m.get('min_ttc_s')}s  collisions={len(self.collisions)}")
        cv2.rectangle(out, (0, 0), (self.cam_w, 24), (0, 0, 0), -1)
        cv2.putText(out, hud, (8, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        return out

    def grab_images(self, frame: Optional[int]) -> Dict[str, np.ndarray]:
        out: Dict[str, np.ndarray] = {}
        for name, q in self.cam_queues.items():
            img = None
            deadline = time.time() + self.a.cam_timeout
            while time.time() < deadline:
                try:
                    cand = q.get(timeout=1.0)
                except queue.Empty:
                    break
                img = cand
                if frame is None or cand.frame >= frame:
                    break
            if img is None:
                continue
            arr = np.frombuffer(img.raw_data, dtype=np.uint8).reshape((img.height, img.width, 4))
            out[name] = arr[:, :, :3][:, :, ::-1].copy()      # BGRA -> BGR
        return out

    def metrics(self) -> Dict[str, Any]:
        """ego 与其它 actor 的最小距离 / 最小 TTC（速度由轨迹估计）"""
        m: Dict[str, Any] = {}
        if not getattr(self, "ego_actor", None):
            return m
        e = self.ego_actor.get_transform()
        tnow = getattr(self, "_t_now", self.t_start)
        espeed = speed_at(self.ego, tnow)
        best_gap, best_ttc, best_id = None, None, None
        fx = math.cos(math.radians(e.rotation.yaw))
        fy = math.sin(math.radians(e.rotation.yaw))
        rx, ry = -fy, fx
        for tid, actor in self.actors.items():
            if actor is self.ego_actor:
                continue
            o = actor.get_transform()
            d = math.sqrt((e.location.x - o.location.x) ** 2 +
                          (e.location.y - o.location.y) ** 2 +
                          (e.location.z - o.location.z) ** 2)
            if best_gap is None or d < best_gap:
                best_gap, best_id = d, tid
            dx, dy = o.location.x - e.location.x, o.location.y - e.location.y
            lon, lat = dx * fx + dy * fy, dx * rx + dy * ry
            if lon > 0 and abs(lat) < 1.9:
                tr = self.track_by_id.get(tid)
                ospeed = speed_at(tr, tnow) if tr is not None else 0.0
                rel = espeed - ospeed
                if rel > 0.3:
                    ttc = max(0.0, lon - 2.0) / rel
                    if best_ttc is None or ttc < best_ttc:
                        best_ttc = ttc
        m["min_dist"] = round(best_gap, 3) if best_gap is not None else None
        m["min_dist_actor"] = best_id
        m["min_ttc_s"] = round(best_ttc, 2) if best_ttc is not None else None
        m["ego_speed"] = round(espeed, 2)
        return m

    # ---------------- 录制 ----------------
    def open_video(self) -> None:
        if not self.a.out_dir:
            return
        os.makedirs(self.a.out_dir, exist_ok=True)
        if self.a.mkv:
            self.video_path = os.path.join(self.a.out_dir, f"{self.sc['name']}.mkv")
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        else:
            self.video_path = os.path.join(self.a.out_dir, f"{self.sc['name']}.mp4")
            fourcc = cv2.VideoWriter_fourcc(*self.a.fourcc)
        n = max(1, len(self.cameras))
        self.writer = cv2.VideoWriter(self.video_path, fourcc, self.a.fps,
                                      (self.cam_w * n, self.cam_h))
        if not self.writer.isOpened():
            raise RuntimeError(f"cv2.VideoWriter 打不开 {self.video_path}（fourcc={self.a.fourcc}）")
        print(f"[video] -> {self.video_path}")

    def compose(self, imgs: Dict[str, np.ndarray]) -> Optional[np.ndarray]:
        """按相机顺序横向拼接（带标注）；缺帧的用黑图补位，保证每帧尺寸一致。"""
        if not imgs:
            return None
        panels = []
        for k in self.cam_specs:
            img = imgs.get(k)
            if img is None:
                img = np.zeros((self.cam_h, self.cam_w, 3), np.uint8)
            panel = img[:self.cam_h, :self.cam_w]
            if getattr(self, "_t_now", None) is not None:
                panel = self.annotate(k, panel)
            panels.append(panel)
        if len(panels) == 1:
            return panels[0]
        return np.hstack(panels)

    # ---------------- 主循环 ----------------
    def run(self) -> int:
        a = self.a
        self.prepare_tracks()
        self.describe()
        if a.dry_run:
            print("[dry-run] 只解析，不连 CARLA")
            return 0
        self.connect()
        self.setup_world()
        self.compute_alignment()
        if self.a.export_xosc is not None:
            try:
                xp = self.write_carla_xosc(self.a.export_xosc)
                print(f"[xosc] CARLA 对齐版 OpenSCENARIO -> {xp}")
            except Exception as e:                       # noqa: BLE001
                print(f"[xosc] 导出失败：{type(e).__name__}: {e}")
        if self.a.export_only:
            print("[carla] --export-only：只导出，不渲染，直接退出")
            try:
                self.world.apply_settings(self._old_settings)
            except Exception:                            # noqa: BLE001
                pass
            return 0
        self.spawn_actors()
        if not self.actors:
            raise RuntimeError("一个 actor 都没生成")
        self.setup_cameras()
        if self.a.live_http > 0:
            self.mjpeg = MjpegServer(self.a.live_http)
            self.mjpeg.start()
            print(f"[live] MJPEG 网页直播: http://127.0.0.1:{self.a.live_http}/")
        if self.a.out_dir:
            self.open_video()

        if a.ego_mode == "autopilot" and self.ego_actor is not None:
            try:
                self.ego_actor.set_autopilot(True, getattr(self, "tm_port_used", a.tm_port))
                if self.tm is not None:
                    try:
                        self.tm.set_desired_speed(self.ego_actor, max(3.0, self.ego.speed))
                    except Exception:                    # noqa: BLE001
                        pass
                print(f"[carla] ego 交给 CARLA autopilot（目标速度 {max(3.0, self.ego.speed):.1f} m/s）")
            except Exception as e:                       # noqa: BLE001
                print(f"[carla] autopilot 启动失败（退回回放）：{type(e).__name__}: {e}")
                self.a.ego_mode = "playback"

        t_begin = time.time()
        if a.warmup > 0 and self.cameras:
            print(f"[carla] 预热 {a.warmup} 帧...")
            self._t_now = self.t_start
            for _ in range(a.warmup):
                self.update_cameras()
                self.world.tick()
                self.grab_images(None)
            print("[carla] 预热完成")
        loops = 0
        try:
            while True:
                loops += 1
                self._play_once()
                if a.loop != 0 and loops >= a.loop:
                    break
                if a.loop == 0:
                    # 无限循环（直播模式）：回到 t_start 重放
                    self._reset_to_start()
                    print(f"[live] 重放第 {loops + 1} 遍 ...")
        except KeyboardInterrupt:
            print("\n[info] 用户中断")
        finally:
            self.cleanup()
        print(f"[done] 输出帧 {self.n_frames}，用时 {time.time() - t_begin:.1f}s，"
              f"碰撞事件 {len(self.collisions)}")
        return 0

    def _reset_to_start(self) -> None:
        for t in self.tracks:
            actor = self.actors.get(t.tid)
            if actor is None:
                continue
            s = sample_track(t, self.t_start)
            if s is None:
                continue
            loc, yaw = self.aligner(*dggt_to_carla(s[0], s[1]))
            if getattr(self.a, "road_snap", "z") != "none":
                loc = snap_to_road(self.world, loc, self.a.road_snap)
            if os.environ.get("DGGT_DEBUG_LOOP"):
                try:
                    alive = actor.is_alive
                except Exception:                            # noqa: BLE001
                    alive = "?"
                print(f"[dbg]   set_transform tid={t.tid} alive={alive} "
                      f"loc={[round(float(x), 2) for x in loc]}", flush=True)
            actor.set_transform(carla.Transform(
                carla.Location(float(loc[0]), float(loc[1]), float(loc[2]) + 0.15),
                carla.Rotation(yaw=math.degrees(yaw))))

    def _play_once(self) -> None:
        a = self.a
        dbg = bool(os.environ.get("DGGT_DEBUG_LOOP"))
        t0 = self.t_start - a.pre_roll
        t1 = self.t_end + a.post_roll
        n = max(1, int(round((t1 - t0) * a.fps)))
        for i in range(n):
            tnow = t0 + i / a.fps
            self._t_now = tnow
            if dbg:
                print(f"[dbg] i={i} apply t={tnow:.2f}", flush=True)
            self.apply_playback(min(max(tnow, self.t_start), self.t_end))
            if dbg:
                print(f"[dbg] i={i} cameras", flush=True)
            self.update_cameras()
            if dbg:
                print(f"[dbg] i={i} tick", flush=True)
            tick_ret = self.world.tick()
            frame = tick_ret if isinstance(tick_ret, int) else self.world.get_snapshot().frame
            if dbg:
                print(f"[dbg] i={i} grab frame={frame}", flush=True)
            imgs = self.grab_images(frame)
            panel = self.compose(imgs)
            if panel is not None and self.writer is not None:
                self.writer.write(panel)
            if self.mjpeg is not None and panel is not None:
                self.mjpeg.push(panel)
            if dbg:
                print(f"[dbg] i={i} metrics", flush=True)
            self.frames_log.append({"frame": self.n_frames, "t": round(tnow, 3),
                                    "ego": self.ego_state(), "metrics": self.metrics()})
            self.n_frames += 1
            if a.every and self.n_frames % a.every == 0:
                print(f"        frame {self.n_frames}  t={tnow:.2f}s  {self.metrics()}")
        # 场景结束后再等一会，方便直播观察
        if a.hold_end > 0:
            for _ in range(int(a.hold_end * a.fps)):
                self.world.tick()
                self.update_cameras()
                imgs = self.grab_images(self.world.get_snapshot().frame)
                panel = self.compose(imgs)
                if panel is not None and self.mjpeg is not None:
                    self.mjpeg.push(panel)

    # ---------------- 收尾 ----------------
    def cleanup(self) -> None:
        a = self.a
        # 顺序很重要：先让 ego 退出 autopilot、再把 TM 的同步模式关掉，**最后**才销毁 actor。
        # 反过来做的话，TM 的内部线程会去操作已经销毁的车，
        # 抛出 std::runtime_error → 进程直接 abort（rc=-6），而且 Python 侧 try/except 拦不住。
        if getattr(self, "ego_actor", None) is not None and a.ego_mode == "autopilot":
            try:
                self.ego_actor.set_autopilot(False, getattr(self, "tm_port_used", a.tm_port))
            except Exception:                            # noqa: BLE001
                pass
        if self.tm is not None:
            try:
                self.tm.set_synchronous_mode(False)
            except Exception:                            # noqa: BLE001
                pass
        try:
            if self.writer is not None:
                self.writer.release()
        except Exception:                                # noqa: BLE001
            pass
        if self.n_frames:
            self._write_reports()
        if a.start_recorder:
            try:
                self.client.stop_recorder()
            except Exception:                            # noqa: BLE001
                pass
        for s in self.sensors:
            try:
                s.stop()
            except Exception:                            # noqa: BLE001
                pass
        for s in self.sensors:
            try:
                s.destroy()
            except Exception:                            # noqa: BLE001
                pass
        for actor in list(self.actors.values()):
            try:
                actor.destroy()
            except Exception:                            # noqa: BLE001
                pass
        try:
            self.world.apply_settings(self._old_settings)
        except Exception:                                # noqa: BLE001
            pass

    def _write_reports(self) -> None:
        if not self.a.out_dir:
            return
        os.makedirs(self.a.out_dir, exist_ok=True)
        base = os.path.join(self.a.out_dir, self.sc["name"])
        gaps = [f["metrics"].get("min_dist") for f in self.frames_log
                if (f.get("metrics") or {}).get("min_dist") is not None]
        ttcs = [f["metrics"].get("min_ttc_s") for f in self.frames_log
                if (f.get("metrics") or {}).get("min_ttc_s") is not None]
        events = {
            "scenario": self.sc["source"], "map": self.world.get_map().name,
            "fps": self.a.fps, "frames": self.n_frames, "actors": len(self.actors),
            "focus": self.a.focus, "focus_ids": self.focus_ids,
            "ego_actor_id": getattr(getattr(self, "ego_actor", None), "id", None),
            "tracks": [{"tid": t.tid, "role": t.role, "kind": t.kind, "is_ego": _is_ego_track(t),
                        "dims_lwh": [t.length, t.width, t.height], "num_poses": len(t.times),
                        "spawned": t.tid in self.actors}
                       for t in self.tracks],
            "collisions": self.collisions,
            "min_dist_overall": round(min(gaps), 3) if gaps else None,
            "min_ttc_overall": round(min(ttcs), 2) if ttcs else None,
        }
        json.dump(events, open(base + "_events.json", "w", encoding="utf-8"),
                  ensure_ascii=False, indent=2)
        json.dump(self.frames_log, open(base + "_frames.json", "w", encoding="utf-8"),
                  ensure_ascii=False, indent=2)
        print(f"[report] {base}_events.json / {base}_frames.json")


# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="DGGT corner-case -> CARLA 闭环仿真/可视化",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--scenario", required=True, help="*.world.json 或 *.xosc")
    p.add_argument("--out-dir", default=None, help="输出目录（MP4 + JSON）")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=2000)
    p.add_argument("--timeout", type=float, default=120.0)
    p.add_argument("--load-retries", type=int, default=3)
    p.add_argument("--reload-world", default="auto", choices=["auto", "always"],
                   help="auto=已在该地图就不重新加载（批处理更快）")
    p.add_argument("--map", default="Town10HD_Opt", help="CARLA 地图（空串=用当前地图）")
    p.add_argument("--fps", type=float, default=20.0, help="输出/仿真帧率")
    p.add_argument("--fourcc", default="mp4v", help="MP4 编码 fourcc")
    p.add_argument("--mkv", action="store_true", help="输出 .mkv(MJPG)，兼容性更好")
    p.add_argument("--pre-roll", type=float, default=1.0)
    p.add_argument("--post-roll", type=float, default=1.5)
    p.add_argument("--hold-end", type=float, default=2.0, help="结束后保持时长（直播用）")
    p.add_argument("--loop", type=int, default=1, help="重放次数（0=无限，直播用）")
    p.add_argument("--actors", default="dynamic", choices=["core", "dynamic", "all", "raw"])
    p.add_argument("--focus", default="crash", choices=["crash", "ego", "all"],
                   help="相机跟随：crash=事故双方(victim/attacker)，ego=自车，all=全部")
    p.add_argument("--min-poses", type=int, default=2)
    p.add_argument("--max-actors", type=int, default=40)
    p.add_argument("--ego-mode", default="playback", choices=["playback", "autopilot"])
    p.add_argument("--vertical", default="flat", choices=["flat", "relative", "center"],
                   help="垂直方向处理：flat=全部贴路面(推荐)")
    p.add_argument("--anchor", default=None, help="手动 anchor: x,y,z,yaw（度）")
    p.add_argument("--anchor-index", type=int, default=0, help="直路候选序号")
    p.add_argument("--physics-off", action="store_true", default=True,
                   help="回放 actor 关物理，轨迹更贴合")
    p.add_argument("--keep-physics", dest="physics_off", action="store_false")
    p.add_argument("--cameras", default="chase,birdseye", help="chase,birdseye,front,side")
    p.add_argument("--cam-size", default="960x540")
    p.add_argument("--chase-dist", type=float, default=8.0)
    p.add_argument("--chase-height", type=float, default=3.0)
    p.add_argument("--bird-height", type=float, default=42.0)
    p.add_argument("--spectator", action="store_true", help="让 CARLA spectator 跟随 ego（配 VNC 看）")
    p.add_argument("--live-http", type=int, default=0, help="MJPEG 网页直播端口（0=关）")
    p.add_argument("--tm-port", type=int, default=8010,
                   help="CARLA TrafficManager 起始端口（会自动挑空闲的；避开 studio 后端的 8000）")
    p.add_argument("--road-snap", default="z", choices=["none", "z", "lane"],
                   help="把车贴回路面：z=只按最近道路修高度(默认)，lane=连 x/y 吸到车道中心，none=不动")
    p.add_argument("--name", default=None, help="输出名（覆盖视频/报告的文件名，方便区分哪次仿真）")
    p.add_argument("--weather", default="ClearNoon",
                   help="CARLA 预设天气（ClearNoon/CloudyNoon/ClearSunset/ClearNight...），none=不动")
    p.add_argument("--start-recorder", default=None, help="CARLA recorder 输出路径（*.log）")
    p.add_argument("--no-collision-sensors", action="store_true")
    p.add_argument("--every", type=int, default=20, help="每 N 帧打一次日志")
    p.add_argument("--warmup", type=int, default=3, help="正式录制前空跑几帧（编译 shader）")
    p.add_argument("--cam-timeout", type=float, default=30.0, help="每帧每相机等待超时(秒)")
    p.add_argument("--export-xosc", default=None, nargs="?", const="",
                   help="额外导出「CARLA 对齐版」OpenSCENARIO（给 ScenarioRunner 用）；"
                        "不带值则写到 out-dir/<名字>_carla.xosc")
    p.add_argument("--export-only", action="store_true",
                   help="只做落图对齐 + 导出 xosc，不生成 actor/不渲染（供 UI 用）")
    p.add_argument("--dry-run", action="store_true", help="只解析场景，不连 CARLA")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if carla is None and not args.dry_run:
        print("!! 没装 carla PythonAPI：请先 source 环境（见 README_CARLA.md）", file=sys.stderr)
        return 2
    if cv2 is None and not args.dry_run:
        print("!! 没装 opencv-python：无法写视频", file=sys.stderr)
        return 2
    return Bridge(args).run()


if __name__ == "__main__":
    sys.exit(main())
