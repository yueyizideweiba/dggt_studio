#!/usr/bin/env python3
"""静态自检：找出"函数里引用了未定义名字"的 bug。

为什么需要它：重构时把变量名写错（例如插入路径把 `ext` 写成替换路径的 `recon_ext`）
在语法检查（py_compile）里是**过不了关的**——它只在运行到那一行时才 NameError，
于是表现成"点了按钮才失败：name 'recon_ext' is not defined"。这类问题用 `symtable`
可以静态发现：某个名字在函数作用域里被当作**全局**引用，而模块顶层既没赋值也没导入，
那它要么是内置名，要么就是笔误。

用法：
    python check_undefined_names.py                # 检查默认的一组文件
    python check_undefined_names.py a.py b.py      # 检查指定文件
退出码非 0 表示发现问题（可直接挂到 CI / 提交前钩子）。
"""
from __future__ import annotations

import builtins
import io
import os
import symtable
import sys
from typing import List, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))

DEFAULT_FILES = [
    os.path.join(REPO, "dggt_engine.py"),
    os.path.join(REPO, "dggt", "scene_edit", "collision_physics.py"),
    os.path.join(HERE, "api_server.py"),
    os.path.join(HERE, "track_manager.py"),
    os.path.join(HERE, "traj_llm.py"),
    os.path.join(HERE, "corner_case.py"),
    os.path.join(HERE, "scenario_engine.py"),
    os.path.join(HERE, "nl_entity.py"),
    os.path.join(HERE, "carla_api.py"),
]


def check_file(path: str) -> List[Tuple[str, str]]:
    """返回 [(函数名, 可疑名字), ...]。"""
    src = io.open(path, encoding="utf-8").read()
    top = symtable.symtable(src, path, "exec")
    known = {s.get_name() for s in top.get_symbols() if s.is_assigned() or s.is_imported()}
    known |= {"__name__", "__file__", "__doc__", "__package__", "__spec__", "__loader__"}
    builtin_names = set(dir(builtins))

    bad: List[Tuple[str, str]] = []

    def walk(tbl, fn: str = "<module>") -> None:
        if tbl.get_type() == "function":
            for s in tbl.get_symbols():
                name = s.get_name()
                if s.is_global() and name not in known and name not in builtin_names:
                    bad.append((fn, name))
        for child in tbl.get_children():
            walk(child, child.get_name())

    walk(top)
    return bad


def main(argv: List[str]) -> int:
    files = argv[1:] or [f for f in DEFAULT_FILES if os.path.exists(f)]
    if not files:
        print("没有可检查的文件")
        return 1
    total = 0
    for f in files:
        try:
            bad = check_file(f)
        except SyntaxError as e:
            print(f"  ✗ {f} 语法错误: {e}")
            total += 1
            continue
        rel = os.path.relpath(f, REPO)
        if bad:
            total += len(bad)
            print(f"  ✗ {rel}")
            for fn, name in bad:
                print(f"      {fn}() 引用了未定义的名字: {name}")
        else:
            print(f"  ✓ {rel}")
    print()
    if total:
        print(f"发现 {total} 处可疑引用（很可能是变量名写错/漏 import）")
        return 1
    print("未发现\"引用了未定义的名字\"")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
