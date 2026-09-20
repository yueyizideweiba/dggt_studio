#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CARLA 事故仿真引擎（可编辑 + 可播放的常驻会话）。

和 `carla_scenario_bridge.py`（一次性渲染出 mp4）不同，这个是**常驻的长会话**：
  * 在 CARLA 里把场景生成出来；
  * 后台线程按 fps 推进世界 → 可以暂停/单步/拖时间轴/调速；
  * 可以**编辑**：选中某个 actor 平移/旋转/删除、新增车辆、切相机/天气；
  * 实时画面（JPEG）给前端当"视频"看；
  * 随时把编辑后的场景**另存为 world.json**（回到 ①/④ 流程继续用）；
  * 自带一个小 HTTP 控制接口（默认 127.0.0.1:8110），studio 后端负责代理。

用法：
    python carla_engine.py --scenario xx.world.json --map Town10HD_Opt --port 8110
"""
from __future__ import annotations

import argparse
import json
import math
import os
import queue
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

import carla
import carla_scenario_bridge as B          # 复用解析/落图对齐/采样/蓝图挑选


def _load_world_watchdog(client, name: str, timeout: float = 150.0):
    """load_world 卡住就硬退出。

    踩过的坑：CARLA server 在 load_world 期间被杀掉，客户端会**永远**卡在这一句，
    后端只能等到超时再报 503，而且那个进程还挂着占显存。这里给它一个硬超时。
    """
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
        print(f"[engine] load_world('{name}') 超过 {timeout:.0f}s 没返回，直接退出"
              f"（CARLA server 可能被杀/卡住了）", flush=True)
        os._exit(3)
    if "e" in box:
        raise box["e"]
    return box.get("w")


def _lane_distance(world, loc, max_d: float = 60.0) -> float:
    """actor 离最近车道中心线多远（m）。用来判断"车到底在不在车道里"。"""
    try:
        q = carla.Location(float(loc[0]), float(loc[1]), float(loc[2]))
        wp = world.get_map().get_waypoint(q, project_to_road=True, lane_type=carla.LaneType.Driving)
        if wp is None:
            return -1.0
        w = wp.transform.location
        return round(min(math.hypot(w.x - loc[0], w.y - loc[1]), 999.0), 2)
    except Exception:                                    # noqa: BLE001
        return -1.0


# --------------------------------------------------------------------------- #
class Engine:
    def __init__(self, args):
        self.a = args
        self.lock = threading.RLock()
        self.sc = B.load_scenario(args.scenario)
        self.tracks: List[B.Track] = []
        self.ego: Optional[B.Track] = None
        self.actors: Dict[int, Any] = {}
        self.offsets: Dict[int, Dict[str, float]] = {}      # tid -> {dx,dy,dz,dyaw}
        self.deleted: set = set()
        self.extra: List[Dict[str, Any]] = []               # 用户新增的车辆
        self.next_extra_id = 900001
        self.cameras: Dict[str, Any] = {}
        self.cam_queues: Dict[str, queue.Queue] = {}
        self.cam_name = args.camera
        self.latest_jpg: Optional[bytes] = None
        self.play = True
        self.frame = 0.0
        self.n_frames = 0
        self.fps = float(args.fps)
        self.weather = args.weather
        self.stopped = False
        self.client = None
        self.world = None
        self._old_settings = None
        self._picker = None
        self._last_err = ""

    # ---------------- 准备 ----------------
    def prepare(self):
        a = self.a
        self.tracks = B.select_tracks(self.sc["tracks"], a.actors, 2, a.max_actors)
        egos = [t for t in self.tracks if B._is_ego_track(t)]
        if not egos:
            egos = sorted(self.tracks, key=lambda t: -t.t_end)[:1]
        self.ego = egos[0] if egos else None
        if self.ego is None:
            raise RuntimeError("场景里没有 ego")
        self.t_start = min(t.times[0] for t in self.tracks)
        self.t_end = max(t.t_end for t in self.tracks)
        self.n_frames = max(2, int(round((self.t_end - self.t_start) * self.fps)) + 1)

    def connect(self):
        a = self.a
        self.client = carla.Client(a.host, a.port)
        self.client.set_timeout(120.0)
        cur = ""
        try:
            cur = self.client.get_world().get_map().name
        except Exception:                                    # noqa: BLE001
            cur = ""
        if a.map and a.map.lower() not in cur.lower():
            print(f"[engine] load_world('{a.map}') …", flush=True)
            self.world = _load_world_watchdog(self.client, a.map)
        else:
            self.world = self.client.get_world()
        print(f"[engine] map = {self.world.get_map().name}", flush=True)
        self._picker = B.BlueprintPicker(self.world)

    def setup_world(self):
        s = self.world.get_settings()
        self._old_settings = s
        s.synchronous_mode = True
        s.fixed_delta_seconds = 1.0 / self.fps
        s.no_rendering_mode = False
        self.world.apply_settings(s)
        if self.weather and self.weather.lower() not in ("none", "keep", ""):
            try:
                params = getattr(carla.WeatherParameters, self.weather, None) or \
                    carla.WeatherParameters.ClearNoon
                self.world.set_weather(params)
            except Exception:                                # noqa: BLE001
                pass

    def compute_alignment(self):
        a = self.a
        if a.anchor:
            vals = [float(x) for x in a.anchor.split(",")]
            anchor_loc, anchor_yaw = np.array(vals[:3]), vals[3]
        else:
            anchor_loc, anchor_yaw, info = B.find_straight_anchor(self.world, index=a.anchor_index)
            print(f"[engine] anchor {np.round(anchor_loc, 2).tolist()} yaw={anchor_yaw:.2f} {info}",
                  flush=True)
        ego_loc, ego_yaw = B.dggt_to_carla(self.ego.locs[0], self.ego.yaws[0])
        self.aligner = B.Aligner(anchor_loc, anchor_yaw, ego_loc, ego_yaw, vertical=a.vertical)

    def spawn_actors(self):
        for t in self.tracks:
            s = B.sample_track(t, self.t_start)
            if s is None:
                continue
            loc, yaw = self.aligner(*B.dggt_to_carla(s[0], s[1]))
            bp = self._picker.pick(t)
            if bp is None:
                continue
            try:
                if bp.has_attribute("role_name"):
                    bp.set_attribute("role_name", f"{t.role}_{t.tid}")
            except Exception:                                # noqa: BLE001
                pass
            tf = carla.Transform(carla.Location(float(loc[0]), float(loc[1]), float(loc[2]) + 0.15),
                                 carla.Rotation(yaw=math.degrees(yaw)))
            actor = None
            _rs = getattr(self.a, "road_snap", "none")
            loc_snap = B.snap_to_road(self.world, loc, _rs) if _rs != "none" else loc
            # 候选位置：先原地，再抬一点，再往侧面挪一点。头碰头这类场景里对向车有时
            # 正好压在另一辆车上，spawn_actor 会因为"位置重叠"直接报错（以前这里就少生成车）。
            cands = [(0.0, 0.0), (0.0, 0.5), (0.0, 1.2), (0.0, 2.5),
                     (0.9, 0.0), (-0.9, 0.0), (1.8, 0.3), (-1.8, 0.3),
                     (2.8, 0.6), (-2.8, 0.6), (4.0, 0.6), (-4.0, 0.6)]
            last_err = ""
            for lat, dz in cands:
                tf.location.z = float(loc_snap[2]) + 0.15 + dz
                if lat:
                    tf.location.x = float(loc_snap[0]) + lat * math.sin(yaw)
                    tf.location.y = float(loc_snap[1]) - lat * math.cos(yaw)
                try:
                    actor = self.world.spawn_actor(bp, tf)
                    if lat or dz:
                        print(f"[engine] tid={t.tid} 原地 spawn 撞车，已挪 侧向{lat:+.1f}m/高度{dz:+.1f}m",
                              flush=True)
                    break
                except RuntimeError as e:                    # noqa: BLE001
                    last_err = str(e)
                    continue
            if actor is None:
                print(f"[engine] spawn 失败 tid={t.tid} role={t.role} loc={loc_snap.round(2).tolist()}：{last_err}",
                      flush=True)
                continue
            if hasattr(actor, "set_simulate_physics"):
                try:
                    actor.set_simulate_physics(False)
                except Exception:                            # noqa: BLE001
                    pass
            self.actors[t.tid] = actor
        print(f"[engine] 生成 {len(self.actors)} 个 actor", flush=True)

    def setup_cameras(self):
        w, h = [int(x) for x in self.a.cam_size.lower().split("x")]
        self.cam_w, self.cam_h = w, h
        for name in ("chase", "birdseye", "front", "side"):
            bp = self.world.get_blueprint_library().find("sensor.camera.rgb")
            fov = 70.0 if name == "birdseye" else 100.0
            try:
                bp.set_attribute("image_size_x", str(w))
                bp.set_attribute("image_size_y", str(h))
                bp.set_attribute("fov", str(fov))
                bp.set_attribute("sensor_tick", "0.0")
            except Exception:                                # noqa: BLE001
                pass
            self.cam_fov = getattr(self, "cam_fov", {})
            self.cam_fov[name] = fov
            sen = self.world.spawn_actor(bp, carla.Transform(carla.Location(0, 0, 60)))
            q: queue.Queue = queue.Queue(maxsize=3)
            sen.listen(lambda img, q=q: self._on_image(q, img))
            self.cameras[name] = sen
            self.cam_queues[name] = q

    @staticmethod
    def _on_image(q: queue.Queue, img):
        try:
            q.put_nowait(img)
        except queue.Full:
            try:
                q.get_nowait()
                q.put_nowait(img)
            except Exception:                                # noqa: BLE001
                pass

    # ---------------- 位姿计算 ----------------
    def pose_of(self, t: B.Track) -> Optional[Tuple[np.ndarray, float]]:
        tnow = self.t_start + self.frame / self.fps
        tnow = min(max(tnow, self.t_start), self.t_end)
        s = B.sample_track(t, tnow)
        if s is None:
            return None
        loc, yaw = self.aligner(*B.dggt_to_carla(s[0], s[1]))
        o = self.offsets.get(t.tid) or {}
        if o:
            ca, sa = math.cos(math.radians(o.get("dyaw", 0.0))), math.sin(math.radians(o.get("dyaw", 0.0)))
            dx, dy = o.get("dx", 0.0), o.get("dy", 0.0)
            rx = dx * ca - dy * sa
            ry = dx * sa + dy * ca
            loc = loc + np.array([rx, ry, o.get("dz", 0.0)])
            yaw = yaw + math.radians(o.get("dyaw", 0.0))
        return loc, yaw

    def subject_state(self):
        ids = [t.tid for t in self.tracks if t.role in ("victim", "attacker")] or \
              ([self.ego.tid] if self.ego else [])
        pts, yaws, spds = [], [], []
        tnow = min(max(self.t_start + self.frame / self.fps, self.t_start), self.t_end)
        for tid in ids:
            t = next((x for x in self.tracks if x.tid == tid), None)
            if t is None:
                continue
            p = self.pose_of(t)
            if p is None:
                continue
            pts.append(p[0]); yaws.append(p[1]); spds.append(B.speed_at(t, tnow))
        if not pts:
            return np.array([0.0, 0.0, 1.0]), 0.0, 0.0
        pts = np.asarray(pts)
        c = pts.mean(axis=0)
        spread = float(np.linalg.norm(pts[:, :2] - c[:2], axis=1).max()) if len(pts) > 1 else 0.0
        yaw = yaws[int(np.argmax(spds))] if spds else yaws[0]
        return c, float(yaw), spread

    def update_cameras(self):
        center, yaw, spread = self.subject_state()
        fwd = np.array([math.cos(yaw), math.sin(yaw), 0.0])
        dist = 9.0 + min(18.0, spread * 1.1)
        height = 3.2 + min(10.0, spread * 0.5)
        bird = max(30.0, 26.0 + spread * 1.8)
        cam = None
        if self.cam_name == "chase":
            cam = center - fwd * dist + np.array([0, 0, height])
            tf = B.look_at(cam, center + np.array([0, 0, 0.8]))
        elif self.cam_name == "birdseye":
            cam = center + np.array([0, 0, bird])
            tf = B.look_at(cam, center + fwd * 3.0)
        elif self.cam_name == "front":
            cam = center + fwd * 1.5 + np.array([0, 0, 1.6])
            tf = B.look_at(cam, center + fwd * 30.0)
        else:
            right = np.array([-math.sin(yaw), math.cos(yaw), 0.0])
            cam = center - fwd * 2.0 + right * (10.0 + spread) + np.array([0, 0, 3.5])
            tf = B.look_at(cam, center)
        for name, sen in self.cameras.items():
            if name == self.cam_name:
                sen.set_transform(tf)

    # ---------------- 主循环 ----------------
    def loop(self):
        dt = 1.0 / self.fps
        last = time.time()
        while not self.stopped:
            try:
                with self.lock:
                    self.update_cameras()
                    self._apply_playback()
                    frame_id = self.world.tick()
                    self._grab(frame_id)
                    if self.play:
                        self.frame = min(self.frame + 1, self.n_frames - 1)
                        if self.frame >= self.n_frames - 1:
                            time.sleep(self.a.hold_end)      # 结尾停一下再从头来
                            self.frame = 0
                            self._reset_extras()
            except Exception as e:                           # noqa: BLE001
                self._last_err = f"{type(e).__name__}: {e}"
                print("[engine] loop 异常:", self._last_err, flush=True)
            # 控制真实速度
            dt_wait = dt - (time.time() - last)
            if dt_wait > 0:
                time.sleep(dt_wait)
            last = time.time()

    def _apply_playback(self):
        for t in self.tracks:
            actor = self.actors.get(t.tid)
            if actor is None or t.tid in self.deleted:
                continue
            p = self.pose_of(t)
            if p is None:
                continue
            loc, yaw = p
            if self.a.road_snap != "none":
                loc = B.snap_to_road(self.world, loc, self.a.road_snap)
            actor.set_transform(carla.Transform(
                carla.Location(float(loc[0]), float(loc[1]), float(loc[2]) + 0.15),
                carla.Rotation(yaw=math.degrees(yaw))))
        for e in self.extra:
            e["loc"] = e["loc"] + np.array([math.cos(e["yaw"]), math.sin(e["yaw"]), 0.0]) * \
                (e["speed"] / self.fps)
            e["actor"].set_transform(carla.Transform(
                carla.Location(float(e["loc"][0]), float(e["loc"][1]), float(e["loc"][2]) + 0.15),
                carla.Rotation(yaw=math.degrees(e["yaw"]))))

    def _reset_extras(self):
        for e in self.extra:
            e["loc"] = e["loc0"].copy()

    def _grab(self, frame_id):
        q = self.cam_queues.get(self.cam_name)
        if q is None:
            return
        img = None
        deadline = time.time() + 3.0
        while time.time() < deadline:
            try:
                cand = q.get(timeout=0.5)
            except queue.Empty:
                break
            img = cand
            if cand.frame >= frame_id:
                break
        if img is None:
            return
        arr = np.frombuffer(img.raw_data, dtype=np.uint8).reshape((img.height, img.width, 4))
        # 注意：[:, :, :3] 只是非连续视图，cv2 画图会报 "Layout of the output array img is
        # incompatible with cv::Mat"，而且原 buffer 出回调后就会失效 —— 必须拷成连续内存。
        bgr = np.ascontiguousarray(arr[:, :, :3])
        # 简单 HUD
        cv2.rectangle(bgr, (0, 0), (self.cam_w, 22), (0, 0, 0), -1)
        txt = (f"{self.a.scenario.split('/')[-1]}  {self.cam_name}  frame {int(self.frame)}/{self.n_frames - 1}"
               f"  t={self.frame / self.fps:.2f}s  {'▶' if self.play else '⏸'}")
        cv2.putText(bgr, txt, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 78])
        if ok:
            self.latest_jpg = buf.tobytes()
        self.latest_bgr = bgr
        self.frames_got = getattr(self, "frames_got", 0) + 1

    # ---------------- 编辑操作 ----------------
    def state(self) -> Dict[str, Any]:
        with self.lock:
            acts = []
            for t in self.tracks:
                if t.tid in self.deleted:
                    continue
                p = self.pose_of(t)
                # 真正落地的位置：如果 actor 已经生成，读它的 transform（贴地/吸车道之后的真实值），
                # 不然 UI 上显示的会是没贴地的原始轨迹，看起来像"还在飘"。
                act = self.actors.get(t.tid)
                if act is not None:
                    try:
                        tf = act.get_transform()
                        p = (np.array([tf.location.x, tf.location.y, tf.location.z], dtype=np.float64),
                             math.radians(float(tf.rotation.yaw)))
                    except Exception:                        # noqa: BLE001
                        pass
                o = self.offsets.get(t.tid) or {}
                acts.append({
                    "tid": t.tid, "role": t.role, "kind": t.kind,
                    "is_ego": B._is_ego_track(t),
                    "size": [round(t.length, 2), round(t.width, 2), round(t.height, 2)],
                    "spawned": t.tid in self.actors,
                    "loc": [round(float(p[0][0]), 2), round(float(p[0][1]), 2), round(float(p[0][2]), 2)] if p else None,
                    "yaw": round(math.degrees(p[1]), 1) if p else None,
                    "lane_d": _lane_distance(self.world, p[0]) if p else None,
                    "offset": {k: round(v, 2) for k, v in o.items()} or None,
                })
            for e in self.extra:
                acts.append({"tid": e["tid"], "role": "added", "kind": "vehicle", "is_ego": False,
                             "size": [4.5, 2.0, 1.5], "spawned": True, "extra": True,
                             "loc": [round(float(x), 2) for x in e["loc"]],
                             "yaw": round(math.degrees(e["yaw"]), 1),
                             "speed": e["speed"], "offset": None})
            return {
                "scenario": self.sc["source"], "map": self.world.get_map().name if self.world else None,
                "frame": int(self.frame), "n_frames": self.n_frames, "fps": self.fps,
                "play": self.play, "camera": self.cam_name, "weather": self.weather,
                "actors": acts, "error": self._last_err or None,
                "frames_got": getattr(self, "frames_got", 0),
                "cameras": list(self.cameras.keys()),
            }

    def edit_actor(self, tid: int, dx=0.0, dy=0.0, dz=0.0, dyaw=0.0, absolute=None):
        with self.lock:
            e = next((x for x in self.extra if x["tid"] == tid), None)
            if e is not None:
                if absolute:
                    e["loc"] = np.array([float(absolute[0]), float(absolute[1]), float(absolute[2])])
                    e["yaw"] = math.radians(float(absolute[3]))
                e["loc"] = e["loc"] + np.array([float(dx), float(dy), float(dz)])
                e["yaw"] = e["yaw"] + math.radians(float(dyaw))
                return {"success": True, "extra": True}
            o = self.offsets.setdefault(int(tid), {})
            o["dx"] = o.get("dx", 0.0) + float(dx)
            o["dy"] = o.get("dy", 0.0) + float(dy)
            o["dz"] = o.get("dz", 0.0) + float(dz)
            o["dyaw"] = o.get("dyaw", 0.0) + float(dyaw)
            return {"success": True, "offset": o}

    def add_actor(self, x, y, z=0.6, yaw_deg=0.0, speed=6.0, kind="vehicle"):
        with self.lock:
            bp = None
            for name in ("vehicle.tesla.model3", "vehicle.audi.a2", "vehicle.nissan.micra"):
                try:
                    cand = self.world.get_blueprint_library().find(name)
                    if cand is not None:
                        bp = cand
                        break
                except Exception:                            # noqa: BLE001
                    continue
            if bp is None:
                return {"success": False, "detail": "找不到可用的车辆蓝图"}
            try:
                if bp.has_attribute("role_name"):
                    bp.set_attribute("role_name", "added")
            except Exception:                                # noqa: BLE001
                pass
            actor = self.world.spawn_actor(bp, carla.Transform(
                carla.Location(float(x), float(y), float(z)),
                carla.Rotation(yaw=float(yaw_deg))))
            if hasattr(actor, "set_simulate_physics"):
                try:
                    actor.set_simulate_physics(False)
                except Exception:                            # noqa: BLE001
                    pass
            self.next_extra_id += 1
            self.extra.append({"tid": self.next_extra_id, "actor": actor,
                               "loc0": np.array([float(x), float(y), float(z)]),
                               "loc": np.array([float(x), float(y), float(z)]),
                               "yaw": math.radians(float(yaw_deg)), "speed": float(speed)})
            return {"success": True, "tid": self.next_extra_id}

    def delete_actor(self, tid: int):
        with self.lock:
            if tid in self.actors:
                self.deleted.add(tid)
                try:
                    self.actors[tid].set_transform(carla.Transform(
                        carla.Location(0, 0, -50)))          # 藏到地下（保留对象，便于恢复）
                except Exception:                            # noqa: BLE001
                    pass
                return {"success": True, "hidden": True}
            for i, e in enumerate(self.extra):
                if e["tid"] == tid:
                    try:
                        e["actor"].destroy()
                    except Exception:                        # noqa: BLE001
                        pass
                    self.extra.pop(i)
                    return {"success": True, "removed": True}
            return {"success": False, "detail": "找不到该 actor"}

    def restore_actor(self, tid: int):
        with self.lock:
            self.deleted.discard(int(tid))
            self.offsets.pop(int(tid), None)
            return {"success": True}

    def reset_edits(self):
        with self.lock:
            self.offsets.clear()
            self.deleted.clear()
            self._reset_extras()
            return {"success": True}

    # ---------------- 导出编辑后的场景 ----------------
    def save_world_json(self, out_path: str) -> str:
        """把当前（含编辑）状态导出成 world.json（DGGT y-up 约定），可被 ①/④ 继续使用。"""
        with self.lock:
            fps = self.fps
            tracks = []
            for t in self.tracks:
                if t.tid in self.deleted:
                    continue
                poses = {}
                for i in range(self.n_frames):
                    self.frame, old = i, self.frame
                    p = self.pose_of(t)
                    if p is None:
                        continue
                    loc, yaw = p
                    poses[str(i)] = self._to_dggt_matrix(loc, yaw)
                tracks.append({"track_id": t.tid, "role": t.role, "is_ego": B._is_ego_track(t),
                               "dimensions_lwh": [t.length, t.width, t.height],
                               "speed_mps": t.speed, "poses": poses})
            for e in self.extra:
                poses = {}
                loc0 = e["loc0"].copy()
                for i in range(self.n_frames):
                    loc = loc0 + np.array([math.cos(e["yaw"]), math.sin(e["yaw"]), 0.0]) * (e["speed"] * i / fps)
                    poses[str(i)] = self._to_dggt_matrix(loc, e["yaw"])
                tracks.append({"track_id": e["tid"], "role": "added", "is_ego": False,
                               "dimensions_lwh": [4.5, 2.0, 1.5], "speed_mps": e["speed"],
                               "poses": poses})
            payload = {"frame": {"convention": "world y-up (x right, y up, z forward), meters",
                                 "dt": 1.0 / fps,
                                 "note": "edited in CARLA engine (world coords round-tripped)"},
                       "tracks": tracks}
            os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            return out_path

    @staticmethod
    def _to_dggt_matrix(loc, yaw) -> List[List[float]]:
        """CARLA (x前/y右/z上) -> DGGT (x右/y上/z前)，yaw 不变。"""
        c, s = math.cos(yaw), math.sin(yaw)
        x_r, y_u, z_f = float(loc[1]), float(loc[2]), float(loc[0])
        return [[c, 0.0, s, x_r],
                [0.0, 1.0, 0.0, y_u],
                [-s, 0.0, c, z_f],
                [0.0, 0.0, 0.0, 1.0]]

    # ---------------- 收尾 ----------------
    def close(self):
        self.stopped = True
        time.sleep(0.3)
        with self.lock:
            for e in self.extra:
                try:
                    e["actor"].destroy()
                except Exception:                            # noqa: BLE001
                    pass
            for s in self.cameras.values():
                try:
                    s.stop()
                    s.destroy()
                except Exception:                            # noqa: BLE001
                    pass
            for a_ in self.actors.values():
                try:
                    a_.destroy()
                except Exception:                            # noqa: BLE001
                    pass
            try:
                if self._old_settings is not None:
                    self.world.apply_settings(self._old_settings)
            except Exception:                                # noqa: BLE001
                pass


# --------------------------------------------------------------------------- #
#  控制 HTTP（只监听本机，由 studio 后端代理）
# --------------------------------------------------------------------------- #
def make_handler(eng: Engine):
    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):                           # noqa: D102
            pass

        def _json(self, obj, code=200):
            body = json.dumps(obj, ensure_ascii=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _body(self):
            n = int(self.headers.get("content-length") or 0)
            if not n:
                return {}
            try:
                return json.loads(self.rfile.read(n).decode("utf-8"))
            except Exception:                                # noqa: BLE001
                return {}

        def do_GET(self):                                # noqa: N802
            path = self.path.split("?")[0]
            if path == "/state":
                return self._json(eng.state())
            if path == "/frame.jpg":
                with eng.lock:
                    data = eng.latest_jpg
                if not data:
                    self.send_response(204)
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return
            if path == "/ping":
                return self._json({"ok": True})
            self.send_error(404)

        def do_POST(self):                               # noqa: N802
            path = self.path.split("?")[0]
            b = self._body()
            try:
                if path == "/play":
                    with eng.lock:
                        eng.play = bool(b.get("play", True))
                    return self._json({"success": True, "play": eng.play})
                if path == "/frame":
                    with eng.lock:
                        eng.frame = max(0.0, min(float(b.get("index", 0)), eng.n_frames - 1))
                    return self._json({"success": True, "frame": int(eng.frame)})
                if path == "/fps":
                    with eng.lock:
                        eng.fps = max(1.0, min(float(b.get("fps", 20)), 60.0))
                        try:
                            eng.world.get_settings().fixed_delta_seconds = 1.0 / eng.fps
                            s = eng.world.get_settings()
                            s.fixed_delta_seconds = 1.0 / eng.fps
                            eng.world.apply_settings(s)
                        except Exception:                    # noqa: BLE001
                            pass
                    return self._json({"success": True, "fps": eng.fps})
                if path == "/camera":
                    with eng.lock:
                        if b.get("camera") in eng.cameras:
                            eng.cam_name = b["camera"]
                    return self._json({"success": True, "camera": eng.cam_name})
                if path == "/weather":
                    name = str(b.get("weather") or "")
                    with eng.lock:
                        params = getattr(carla.WeatherParameters, name, None)
                        if params is not None:
                            eng.world.set_weather(params)
                            eng.weather = name
                    return self._json({"success": True, "weather": eng.weather})
                if path == "/actor":
                    return self._json(eng.edit_actor(int(b["tid"]), b.get("dx", 0), b.get("dy", 0),
                                                     b.get("dz", 0), b.get("dyaw", 0), b.get("absolute")))
                if path == "/actor/add":
                    return self._json(eng.add_actor(b.get("x", 0), b.get("y", 0), b.get("z", 0.6),
                                                    b.get("yaw", 0), b.get("speed", 6.0)))
                if path == "/actor/delete":
                    return self._json(eng.delete_actor(int(b["tid"])))
                if path == "/actor/restore":
                    return self._json(eng.restore_actor(int(b["tid"])))
                if path == "/reset":
                    return self._json(eng.reset_edits())
                if path == "/save":
                    out = b.get("path") or "/tmp/carla_engine_edited.world.json"
                    return self._json({"success": True, "path": eng.save_world_json(out)})
                if path == "/screenshot":
                    out = b.get("path") or "/tmp/carla_engine.png"
                    with eng.lock:
                        img = getattr(eng, "latest_bgr", None)
                    if img is None:
                        return self._json({"success": False, "detail": "还没有画面"}, 503)
                    os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
                    cv2.imwrite(out, img)
                    return self._json({"success": True, "path": out})
                if path == "/quit":
                    threading.Thread(target=lambda: (time.sleep(0.2), eng.close()), daemon=True).start()
                    return self._json({"success": True})
            except Exception as e:                           # noqa: BLE001
                traceback.print_exc()
                return self._json({"success": False, "detail": f"{type(e).__name__}: {e}"}, 500)
            self.send_error(404)

    return H


def main():
    ap = argparse.ArgumentParser(description="CARLA 事故仿真引擎（可编辑 + 可播放）")
    ap.add_argument("--scenario", required=True)
    ap.add_argument("--map", default="Town10HD_Opt")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=2000, help="CARLA RPC 端口")
    ap.add_argument("--control-port", type=int, default=8110)
    ap.add_argument("--fps", type=float, default=20.0)
    ap.add_argument("--cameras", default="chase,birdseye,front,side")
    ap.add_argument("--camera", default="chase")
    ap.add_argument("--cam-size", default="960x540")
    ap.add_argument("--weather", default="ClearNoon")
    ap.add_argument("--actors", default="dynamic", choices=["core", "dynamic", "all", "raw"])
    ap.add_argument("--max-actors", type=int, default=40)
    ap.add_argument("--vertical", default="flat")
    ap.add_argument("--road-snap", default="z", choices=["none", "z", "lane"],
                    help="把车贴回路面（z=只修高度，lane=吸到车道中心）")
    ap.add_argument("--anchor", default=None)
    ap.add_argument("--anchor-index", type=int, default=0)
    ap.add_argument("--hold-end", type=float, default=0.6)
    a = ap.parse_args()

    eng = Engine(a)
    eng.prepare()
    eng.connect()
    eng.setup_world()
    eng.compute_alignment()
    eng.spawn_actors()
    eng.setup_cameras()

    httpd = ThreadingHTTPServer((a.host, a.control_port), make_handler(eng))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    print(f"[engine] 控制接口 http://{a.host}:{a.control_port}/  （/state /frame.jpg …）", flush=True)
    try:
        eng.loop()
    except KeyboardInterrupt:
        pass
    finally:
        try:
            httpd.shutdown()
        except Exception:                                    # noqa: BLE001
            pass
        eng.close()
        print("[engine] 已退出", flush=True)


if __name__ == "__main__":
    main()
