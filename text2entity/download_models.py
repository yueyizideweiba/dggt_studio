#!/usr/bin/env python3
"""从 ModelScope 递归下载模型（断点续传/重试）。无需 modelscope SDK。

用法:
    python download_models.py                      # 默认下 LLaDA-Image-Turbo-FP8 + Qwen2.5-VL-3B
    python download_models.py --repo X --dest Y
"""
import argparse
import json
import os
import sys
import time
import urllib.parse
import urllib.request

API = "https://www.modelscope.cn/api/v1/models/{repo}/repo/files?Revision=master&Recursive=true"
FILE = "https://www.modelscope.cn/api/v1/models/{repo}/repo?Revision=master&FilePath={path}"
UA = {"User-Agent": "Mozilla/5.0"}


def list_files(repo):
    req = urllib.request.Request(API.format(repo=repo), headers=UA)
    with urllib.request.urlopen(req, timeout=60) as r:
        d = json.load(r)
    return [(f["Path"], int(f["Size"])) for f in d.get("Data", {}).get("Files", [])
            if f.get("Path") and (f.get("Size") or 0) > 0]


def download(repo, dest):
    files = list_files(repo)
    total = sum(s for _, s in files)
    print(f"[{repo}] {len(files)} files, {total/1e9:.1f} GB -> {dest}", flush=True)
    for i, (path, size) in enumerate(files):
        out = os.path.join(dest, path)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        have = os.path.getsize(out) if os.path.exists(out) else 0
        if have == size:
            continue
        url = FILE.format(repo=repo, path=urllib.parse.quote(path))
        for attempt in range(8):
            have = os.path.getsize(out) if os.path.exists(out) else 0
            if have >= size:
                break
            headers = {**UA, "Range": f"bytes={have}-"} if have else UA
            try:
                t0 = time.time()
                req = urllib.request.Request(url, headers=headers)
                with urllib.request.urlopen(req, timeout=120) as r, open(out, "ab" if have else "wb") as fh:
                    while True:
                        chunk = r.read(1 << 20)
                        if not chunk:
                            break
                        fh.write(chunk)
                new = os.path.getsize(out)
                print(f"  [{i+1}/{len(files)}] {path} {new/1e6:.1f}MB "
                      f"(+{(new-have)/max(1e-3, time.time()-t0)/1e6:.1f}MB/s)", flush=True)
                if new >= size:
                    break
            except Exception as e:  # noqa: BLE001
                print(f"  retry {path} #{attempt+1}: {e}", flush=True)
                time.sleep(3)
    print(f"[{repo}] done", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", action="append", default=[])
    ap.add_argument("--dest", action="append", default=[])
    args = ap.parse_args()
    targets = list(zip(args.repo, args.dest)) or [
        ("inclusionAI/LLaDA-Image-Turbo-FP8", "/autodl-fs/data/models/LLaDA-Image-Turbo-FP8"),
        ("Qwen/Qwen2.5-VL-3B-Instruct", "/autodl-fs/data/models/Qwen2.5-VL-3B-Instruct"),
    ]
    for repo, dest in targets:
        try:
            download(repo, dest)
        except Exception as e:  # noqa: BLE001
            print(f"[{repo}] FAILED: {e}", flush=True)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
