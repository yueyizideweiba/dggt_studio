#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把一批 DGGT 导出场景批量渲染成 CARLA 视频。

用法：
    python batch_render.py                          # 默认 output/export/*/*.world.json
    python batch_render.py --pattern 'output/export/*/*.xosc'
    python batch_render.py --include-demo           # 连 output/demo* 也一起
    python batch_render.py --cam-size 640x360 --fps 20

每个场景输出到 output/carla/<父目录>_<场景名>/，并汇总成 output/carla/_batch_report.json。

注意：CARLA server 必须先在跑（./start_carla_server.sh --daemon）。
同一张地图只 load_world 一次（桥接的 --reload-world auto），所以批处理很快。
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def find_scenarios(pattern: str, include_demo: bool) -> list:
    pats = [pattern]
    if include_demo:
        pats += ["output/demo/*/*.world.json", "output/demo_final/*/*.world.json",
                 "output/demo_t3v/*/*.world.json"]
    out = []
    for p in pats:
        full = p if os.path.isabs(p) else os.path.join(ROOT, p)
        out += [x for x in glob.glob(full) if os.path.isfile(x)]
    # 去重、排序；优先 .world.json（信息最全）
    seen, res = set(), []
    for f in sorted(set(out)):
        base = os.path.basename(f)
        key = os.path.splitext(base)[0]
        parent = os.path.basename(os.path.dirname(f))
        k2 = f"{parent}/{key}"
        if k2 in seen:
            continue
        seen.add(k2)
        res.append(f)
    return res


def main() -> int:
    ap = argparse.ArgumentParser(description="批量渲染 DGGT 场景到 CARLA 视频")
    ap.add_argument("--pattern", default="output/export/*/*.world.json")
    ap.add_argument("--include-demo", action="store_true", help="连带 output/demo* 里的场景")
    ap.add_argument("--out-root", default=os.path.join(ROOT, "output", "carla"))
    ap.add_argument("--map", default="Town10HD_Opt")
    ap.add_argument("--fps", default="20")
    ap.add_argument("--cameras", default="chase,birdseye")
    ap.add_argument("--cam-size", default="640x360")
    ap.add_argument("--focus", default="crash")
    ap.add_argument("--actors", default="dynamic")
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 个（调试用）")
    args = ap.parse_args()

    scens = find_scenarios(args.pattern, args.include_demo)
    if args.limit:
        scens = scens[:args.limit]
    if not scens:
        print("没找到场景，检查 --pattern"); return 2

    print(f"共 {len(scens)} 个场景：")
    for s in scens:
        print("   ", os.path.relpath(s, ROOT))

    report = []
    t_all = time.time()
    for i, s in enumerate(scens, 1):
        parent = os.path.basename(os.path.dirname(s))
        name = os.path.splitext(os.path.basename(s))[0]
        tag = f"{parent}_{name}" if parent else name
        out_dir = os.path.join(args.out_root, tag)
        print(f"\n================ [{i}/{len(scens)}] {tag}")
        t0 = time.time()
        cmd = [os.path.join(HERE, "run_bridge.sh"),
               "--scenario", s, "--out-dir", out_dir,
               "--map", args.map, "--fps", args.fps,
               "--cameras", args.cameras, "--cam-size", args.cam_size,
               "--focus", args.focus, "--actors", args.actors,
               "--reload-world", "auto", "--every", "0"]
        r = subprocess.run(cmd, cwd=HERE, capture_output=True, text=True)
        tail = (r.stdout or "").strip().splitlines()[-14:]
        print("\n".join("    " + x for x in tail))
        if r.returncode != 0:
            print("    !! stderr:", "\n    ".join((r.stderr or "").strip().splitlines()[-6:]))
        ev = os.path.join(out_dir, f"{name}_events.json")
        info = {"scenario": os.path.relpath(s, ROOT), "out_dir": os.path.relpath(out_dir, ROOT),
                "seconds": round(time.time() - t0, 1), "ok": r.returncode == 0}
        if os.path.exists(ev):
            try:
                e = json.load(open(ev, encoding="utf-8"))
                info.update({"frames": e.get("frames"), "actors": e.get("actors"),
                             "collisions": len(e.get("collisions") or []),
                             "min_dist": e.get("min_dist_overall"),
                             "min_ttc": e.get("min_ttc_overall")})
            except Exception:                            # noqa: BLE001
                pass
        report.append(info)

    os.makedirs(args.out_root, exist_ok=True)
    rp = os.path.join(args.out_root, "_batch_report.json")
    json.dump({"total": len(report), "seconds": round(time.time() - t_all, 1), "items": report},
              open(rp, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    print(f"\n===== 汇总（{time.time() - t_all:.0f}s）=====")
    print(f"{'scenario':<40}{'frames':>7}{'actors':>7}{'coll':>6}{'minDist':>9}{'minTTC':>8}")

    def _v(x):
        return "-" if x is None else (f"{x:g}" if isinstance(x, (int, float)) else str(x))
    for r in report:
        print(f"{r['scenario'][-40:]:<40}{_v(r.get('frames')):>7}{_v(r.get('actors')):>7}"
              f"{_v(r.get('collisions')):>6}{_v(r.get('min_dist')):>9}{_v(r.get('min_ttc')):>8}")
    print(f"\n报告 -> {rp}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
