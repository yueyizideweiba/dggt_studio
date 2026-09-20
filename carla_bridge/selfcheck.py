#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""carla_bridge 静态自检：抓「方法里用了裸变量 a」这类只在运行时才炸的低级错误。

背景（真踩过的坑）：`apply_playback()` 里写成了 `snap_to_road(self.world, loc, a.road_snap)`，
方法内并没有 `a`（应该是 `self.a`）→ 运行时 NameError；而这个 NameError 又在 cleanup 之前
抛出，cleanup 里销毁 TM 控制中的 actor 会触发 CARLA 的 C++ abort，
于是界面上只看到 `rc=-6`（SIGABRT），真正的报错被完全掩盖，排查了很久。

用法：
  python3 carla_bridge/selfcheck.py            # 检查 carla_bridge/*.py
  python3 carla_bridge/selfcheck.py path1.py   # 检查指定文件
退出码 0 = 干净，1 = 有问题。
"""
import ast
import pathlib
import sys

TARGETS = ["carla_scenario_bridge.py", "carla_engine.py"]


def check(path: pathlib.Path):
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except SyntaxError as e:
        return [f"{path}: 语法错误 {e}"]
    problems = []
    for fn in ast.walk(tree):
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        bound = {a.arg for a in fn.args.args} | {a.arg for a in fn.args.kwonlyargs}
        bound |= {n.id for n in ast.walk(fn) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}
        for n in ast.walk(fn):
            if (isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)
                    and n.id == "a" and n.id not in bound):
                problems.append(f"{path}:{n.lineno} 方法 {fn.name}() 里用了未定义的 `a`"
                                f"（大概想写 self.a）")
    return problems


def main(argv):
    base = pathlib.Path(__file__).resolve().parent
    files = [pathlib.Path(p) for p in argv[1:]] or [base / t for t in TARGETS]
    bad = []
    for f in files:
        if not f.exists():
            bad.append(f"{f}: 文件不存在")
            continue
        problems = check(f)
        bad.extend(problems)
        print(("✗" if problems else "✓"), f.name, "" if not problems else f"({len(problems)} 处)")
    for b in bad:
        print("  -", b)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
