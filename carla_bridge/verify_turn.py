#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""验证「CARLA 里车会拐弯 + 在车道里 + 不飘」：起引擎逐帧采所有 actor 的位姿。

用法：
  python carla_bridge/verify_turn.py <world.json> [road_snap ...]
  # 不传 road_snap 时依次测 lane / z / none
"""
import json
import math
import sys
import time
import urllib.request

API = "http://127.0.0.1:8000/api/carla"


def call(path, data=None, timeout=180):
    body = json.dumps(data).encode() if data is not None else None
    req = urllib.request.Request(f"{API}{path}", data=body,
                                 headers={"Content-Type": "application/json"} if body else {})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def span_of(ys):
    ys = [y % 360 for y in ys]
    d = max(ys) - min(ys)
    return d if d <= 180 else 360 - d


def run(scen, snap):
    print(f"\n{'='*78}\n[verify] road_snap={snap}  scenario={scen}")
    st = call("/engine/start", {"scenario": scen, "map": "Town10HD_Opt", "road_snap": snap,
                               "actors": "raw", "fps": 20.0})
    s = st.get("state") or {}
    n = int(s.get("n_frames") or 0)
    print(f"[verify] n_frames={n} actors={len(s.get('actors', []))}")
    if not n:
        print("!! 引擎没给出帧数")
        return False
    data = {}
    for i in range(n):
        call("/engine/frame", {"index": i}, timeout=30)
        time.sleep(0.2)                                  # 让引擎循环把这一帧真正应用到 actor 上
        cur = call("/engine/state", timeout=30)
        for a in cur.get("actors", []):
            if not a.get("loc"):
                continue
            d = data.setdefault(a["tid"], {"role": a.get("role"), "ego": a.get("is_ego"),
                                           "y": [], "z": [], "lane": []})
            d["y"].append(a["yaw"] or 0.0)
            d["z"].append(a["loc"][2])
            if a.get("lane_d") is not None and a["lane_d"] >= 0:
                d["lane"].append(a["lane_d"])
    ok = True
    print(f"  {'tid':>8} {'role':>8} {'ego':>5} {'朝向跨度':>9} {'z落差':>7} {'离车道中心':>12}")
    for tid, d in sorted(data.items()):
        sp = span_of(d["y"])
        zr = max(d["z"]) - min(d["z"])
        la = (f"{min(d['lane']):.2f}~{max(d['lane']):.2f} m" if d["lane"] else "n/a")
        turn = "✓" if sp >= 20 else "—"
        flat = "✓" if zr <= 1.5 else "✗"
        lane = "✓" if d["lane"] and max(d["lane"]) <= 3.5 else ("✗" if d["lane"] else "?")
        if d["ego"]:
            ok = ok and zr <= 1.5
        print(f"  {tid:>8} {str(d['role']):>8} {str(d['ego']):>5} {sp:7.1f}°{turn} {zr:6.2f}m{flat}  {la:>12} {lane}")
    try:
        call("/engine/stop", {}, timeout=30)
    except Exception:                                    # noqa: BLE001
        pass
    return ok


def main():
    scen = sys.argv[1]
    modes = sys.argv[2:] or ["lane", "z"]
    for m in modes:
        run(scen, m)
    return 0


if __name__ == "__main__":
    sys.exit(main())
