"""CARLA 仿真 API（给 studio 前端用）。

设计要点
--------
1. studio 后端跑在 dggt 的 **python3.10** 环境里，**不能 import carla**
   （CARLA 0.9.15 的 Linux PythonAPI 只有 cp37 wheel）。所以这里所有与 CARLA 的交互
   都通过 subprocess 调 `carla_bridge/*.sh`（它们内部用 /autodl-fs/data/carla/py37）。
2. 渲染/导出这类几十秒的任务做成**后台作业 + 进度轮询**，前端能实时看到日志尾巴，
   不会因为一个 HTTP 长请求超时而"卡死"。
3. 视频走 `<video>` 播放，接口支持 HTTP Range（可拖动进度条）。
4. 「用当前编辑场景跑 CARLA」直接复用 api_server 的 `/api/export/scenario`
   （延迟导入，避免循环依赖），保证导出口径与手工导出完全一致。

挂载方式（在 api_server.py 末尾）：
    import carla_api
    carla_api.init(studio_state)
    app.include_router(carla_api.router)
"""
from __future__ import annotations

import asyncio
import json
import os
import shutil
import socket
import subprocess
import sys
import threading
import time
import uuid
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

# --------------------------------------------------------------------------- #
#  路径 / 常量
# --------------------------------------------------------------------------- #
BACKEND_DIR = Path(__file__).resolve().parent
REPO = BACKEND_DIR.parents[1]                       # .../dggt-main
BRIDGE = REPO / "carla_bridge"
OUTPUT = REPO / "output"
CARLA_HOME = Path(os.environ.get("CARLA_HOME", "/autodl-fs/data/carla"))
CARLA_PY = CARLA_HOME / "py37" / "bin" / "python"
CARLA_PORT = int(os.environ.get("CARLA_PORT", "2000"))
LIVE_PORT = int(os.environ.get("CARLA_LIVE_PORT", "8090"))

SH_SERVER = BRIDGE / "start_carla_server.sh"
SH_STOP = BRIDGE / "stop_all.sh"
SH_BRIDGE = BRIDGE / "run_bridge.sh"
SH_FIX = BRIDGE / "fix_nvidia_vulkan.sh"
SH_SR = BRIDGE / "run_scenario_runner.sh"

MAPS = ["Town10HD_Opt", "Town10HD", "Town03_Opt", "Town03", "Town05_Opt", "Town05",
        "Town01_Opt", "Town01", "Town02_Opt", "Town02", "Town04_Opt", "Town04"]

# CARLA 一开就占 ~5-6GB 显存；跑完没人用就自动放掉，别让它长期霸占（否则 SAM3D 会 OOM）
AUTO_RELEASE_SECONDS = float(os.environ.get("CARLA_AUTO_RELEASE_SECONDS", "60"))
CARLA_MIN_FREE_MB = int(os.environ.get("CARLA_MIN_FREE_MB", "7000"))

router = APIRouter(prefix="/api/carla", tags=["carla"])

_STATE: Dict[str, Any] = {"studio_state": None}
_JOBS: Dict[str, "Job"] = {}
_JOBS_LOCK = threading.Lock()
_IDLE_LOCK = threading.Lock()
_IDLE: Dict[str, Any] = {"timer": None, "gen": 0, "busy": 0, "last_start": 0.0}


def _busy_begin() -> None:
    """标记「正在起 CARLA/引擎/作业」：这段时间绝不能被空闲释放打断。

    踩过的坑：引擎启动时先起 CARLA server、再让客户端 load_world，这中间如果被
    空闲释放把 server 杀掉，客户端就会卡死在 load_world（表现为引擎起不来 / rc=-6）。
    """
    _cancel_idle_release()
    with _IDLE_LOCK:
        _IDLE["busy"] = int(_IDLE.get("busy") or 0) + 1


def _busy_end() -> None:
    with _IDLE_LOCK:
        _IDLE["busy"] = max(0, int(_IDLE.get("busy") or 0) - 1)


def init(studio_state: Dict[str, Any], *, export_scenario=None, export_request_cls=None,
         get_track_manager=None, get_scene=None, track_manager_cls=None) -> None:
    """由 api_server 注入依赖。

    注意：**不要**在 carla_api 里 `import api_server`。api_server 是用 `python api_server.py`
    直接跑的，它的模块名是 `__main__`；再 `import api_server` 会重新执行一遍模块、生成一个
    `studio_state` 为空的**影子模块**，于是"用当前编辑的场景"永远找不到场景
    （表现为 `Scene xxx not found`）。所以这里改成依赖注入。
    """
    _STATE.update({
        "studio_state": studio_state,
        "export_scenario": export_scenario,
        "export_request_cls": export_request_cls,
        "get_track_manager": get_track_manager,
        "get_scene": get_scene,
        "track_manager_cls": track_manager_cls,
    })


# --------------------------------------------------------------------------- #
#  小工具
# --------------------------------------------------------------------------- #
def _port_open(port: int, host: str = "127.0.0.1", timeout: float = 0.4) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except Exception:                                    # noqa: BLE001
        return False


def _pgrep(pattern: str) -> bool:
    try:
        r = subprocess.run(["pgrep", "-f", pattern], capture_output=True, text=True, timeout=5)
        return r.returncode == 0 and bool(r.stdout.strip())
    except Exception:                                    # noqa: BLE001
        return False


def _http_ok(url: str, timeout: float = 2.0) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.status == 200
    except Exception:                                    # noqa: BLE001
        return False


def _tail(path: Path, n: int = 40) -> List[str]:
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return f.read().splitlines()[-n:]
    except Exception:                                    # noqa: BLE001
        return []


def _rel(p) -> str:
    """转成相对仓库的路径，前端好展示；不在仓库内就返回绝对路径。"""
    try:
        return str(Path(p).resolve().relative_to(REPO))
    except Exception:                                    # noqa: BLE001
        return str(p)


def _safe_output_path(path: str) -> Path:
    p = Path(path)
    if not p.is_absolute():
        p = REPO / p
    p = p.resolve()
    if not str(p).startswith(str(OUTPUT.resolve())):
        raise HTTPException(status_code=400, detail="只允许访问仓库 output/ 下的文件")
    if not p.exists():
        raise HTTPException(status_code=404, detail=f"文件不存在：{_rel(p)}")
    return p


# --------------------------------------------------------------------------- #
#  显存 / 进程
# --------------------------------------------------------------------------- #
def _gpu_info() -> Optional[Dict[str, Any]]:
    """显存总量/已用/空闲 + 占用 GPU 的进程（用来解释"为什么 OOM"）。"""
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total,memory.used,memory.free",
             "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=15)
        parts = [int(x) for x in (r.stdout or "").strip().splitlines()[0].split(",")]
        info = {"total_mb": parts[0], "used_mb": parts[1], "free_mb": parts[2], "procs": []}
    except Exception:                                    # noqa: BLE001
        return None
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
             "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=15)
        for line in (r.stdout or "").strip().splitlines():
            bits = [b.strip() for b in line.split(",")]
            if len(bits) >= 3:
                name = bits[1].split("/")[-1]
                info["procs"].append({"pid": bits[0], "name": name,
                                      "used_mb": int(bits[2]) if bits[2].isdigit() else 0})
        info["procs"].sort(key=lambda x: -x["used_mb"])
    except Exception:                                    # noqa: BLE001
        pass
    return info


def _stop_carla_processes(stop_live: bool = True) -> None:
    """SIGKILL 掉 CARLA server（UE4 会忽略 SIGTERM）、仿真引擎与直播桥接。"""
    targets = ["CarlaUE4-Linux-Shipping", "carla_engine"]
    if stop_live:
        targets.append("carla_scenario_bridge")
    for t in targets:
        try:
            subprocess.run(["pkill", "-9", "-f", t], capture_output=True, timeout=20)
        except Exception:                                # noqa: BLE001
            pass


# --------------------------------------------------------------------------- #
#  视频：浏览器放不了 mp4v(MPEG-4 Part 2)，用 ffmpeg 转 H.264（缓存）
# --------------------------------------------------------------------------- #
def _has_browser_codec(p: Path) -> bool:
    try:
        with open(p, "rb") as f:
            head = f.read(16384)
    except OSError:
        return True
    return any(tag in head for tag in (b"avc1", b"hvc1", b"hev1", b"vp09", b"av01"))


def _ensure_browser_mp4(p: Path) -> Path:
    """把 mp4v/mjpeg 之类的 mp4 转成 H.264，返回可以直接丢给 <video> 的路径。"""
    if p.suffix.lower() != ".mp4" or _has_browser_codec(p):
        return p
    out = p.with_name(p.stem + ".h264.mp4")
    try:
        if out.exists() and out.stat().st_mtime >= p.stat().st_mtime and out.stat().st_size > 1024:
            return out
    except OSError:
        pass
    if not shutil.which("ffmpeg"):
        return p
    tmp = p.with_name(p.stem + ".h264.part.mp4")
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(p),
           "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
           "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(tmp)]
    try:
        r = subprocess.run(cmd, capture_output=True, timeout=1800)
        if r.returncode == 0 and tmp.exists() and tmp.stat().st_size > 1024:
            tmp.replace(out)
            print(f"[carla] 已把 {p.name} 转成浏览器可播的 H.264（{out.stat().st_size // 1024} KB）")
            return out
    except Exception as e:                               # noqa: BLE001
        print(f"[carla] 转码失败（返回原文件）：{e}")
    finally:
        try:
            if tmp.exists():
                tmp.unlink()
        except OSError:
            pass
    return p


def _cancel_idle_release() -> None:
    with _IDLE_LOCK:
        _IDLE["gen"] += 1
        if _IDLE["timer"] is not None:
            try:
                _IDLE["timer"].cancel()
            except Exception:                            # noqa: BLE001
                pass
            _IDLE["timer"] = None


def _schedule_idle_release(seconds: Optional[float] = None) -> None:
    """跑完一段时间没人用就自动释放显存；期间有作业/直播就顺延。"""
    seconds = AUTO_RELEASE_SECONDS if seconds is None else seconds
    _cancel_idle_release()
    if seconds <= 0:
        return
    with _IDLE_LOCK:
        gen = _IDLE["gen"]

    def _fire():
        with _IDLE_LOCK:
            if _IDLE["gen"] != gen:
                return
            busy = int(_IDLE.get("busy") or 0)
            last_start = float(_IDLE.get("last_start") or 0.0)
        if busy:
            _schedule_idle_release(20)
            return
        if time.time() - last_start < 120:                # 刚起过 CARLA/引擎，先别放
            _schedule_idle_release(30)
            return
        if any(j.status == "running" for j in _JOBS.values()):
            _schedule_idle_release(20)
            return
        if _engine_alive():
            _schedule_idle_release(60)                   # 仿真引擎在跑，别停 CARLA
            return
        if _http_ok(f"http://127.0.0.1:{LIVE_PORT}/health", timeout=1.0):
            _schedule_idle_release(30)
            return
        if _pgrep("CarlaUE4-Linux-Shipping"):
            _stop_carla_processes(stop_live=False)
            print("[carla] 空闲自动释放显存：已停 CARLA server")

    with _IDLE_LOCK:
        t = threading.Timer(seconds, _fire)
        t.daemon = True
        _IDLE["timer"] = t
        t.start()


def _on_job_finished(job: "Job") -> None:
    if job.kind in ("render", "export_xosc", "scenario_runner") and AUTO_RELEASE_SECONDS > 0:
        _schedule_idle_release()


# --------------------------------------------------------------------------- #
#  作业（后台跑 run_bridge.sh / scenario_runner）
# --------------------------------------------------------------------------- #
class Job:
    """一个后台 CARLA 作业：子进程 + 环形日志缓冲 + 产物清单。"""

    def __init__(self, kind: str, cmd: List[str], out_dir: Optional[Path] = None,
                 name: str = "job", title: str = ""):
        self.id = uuid.uuid4().hex[:12]
        self.kind = kind
        self.cmd = cmd
        self.out_dir = Path(out_dir) if out_dir else None
        self.name = name
        self.title = title or kind
        self.status = "running"           # running | done | failed | cancelled
        self.rc: Optional[int] = None
        self.started = time.time()
        self.ended: Optional[float] = None
        self.lines: List[str] = []
        self._lock = threading.Lock()
        self._proc: Optional[subprocess.Popen] = None

    # -- 生命周期 --
    def start(self) -> "Job":
        with _JOBS_LOCK:
            _JOBS[self.id] = self
        _cancel_idle_release()               # 有活干，别自动停 CARLA
        threading.Thread(target=self._run, daemon=True).start()
        return self

    def _run(self) -> None:
        env = dict(os.environ)
        env.setdefault("PYTHONUNBUFFERED", "1")
        try:
            self._proc = subprocess.Popen(
                self.cmd, cwd=str(BRIDGE), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1, env=env, start_new_session=True)
            assert self._proc.stdout is not None
            for line in self._proc.stdout:
                self._append(line.rstrip("\n"))
            self.rc = self._proc.wait()
        except Exception as e:                           # noqa: BLE001
            self._append(f"[job] 启动失败：{type(e).__name__}: {e}")
            self.rc = -1
        self.status = "done" if self.rc == 0 else "failed"
        self.ended = time.time()
        self._append(f"[job] 结束 rc={self.rc} 用时 {self.ended - self.started:.1f}s")
        try:
            _on_job_finished(self)                   # 跑完约 1 分钟后自动放显存
        except Exception:                            # noqa: BLE001
            pass

    def _append(self, line: str) -> None:
        with self._lock:
            self.lines.append(line)
            if len(self.lines) > 4000:
                del self.lines[:1000]

    def cancel(self) -> None:
        try:
            if self._proc and self._proc.poll() is None:
                self._proc.terminate()
                self.status = "cancelled"
        except Exception:                                # noqa: BLE001
            pass

    # -- 视图 --
    def info(self, tail: int = 60) -> Dict[str, Any]:
        with self._lock:
            lines = self.lines[-tail:] if tail else list(self.lines)
        return {"id": self.id, "kind": self.kind, "title": self.title, "status": self.status,
                "rc": self.rc, "started": self.started, "ended": self.ended,
                "seconds": round((self.ended or time.time()) - self.started, 1),
                "out_dir": _rel(self.out_dir) if self.out_dir else None,
                "log_tail": lines, "artifacts": self.artifacts()}

    def artifacts(self) -> Dict[str, Any]:
        """扫 out_dir 里**本次作业**产出的文件（按 mtime 过滤掉上一次跑剩下的）。"""
        out: Dict[str, Any] = {}
        if not self.out_dir or not self.out_dir.exists():
            return out
        t0 = self.started - 1.0
        for f in sorted(self.out_dir.rglob("*")):
            if not f.is_file():
                continue
            try:
                if f.stat().st_mtime < t0:
                    continue                     # 上一次跑的产物，不算本次
            except OSError:
                continue
            key = f.suffix.lower().lstrip(".")
            if f.name.endswith("_events.json"):
                out["events_json"] = _rel(f)
            elif f.name.endswith("_frames.json"):
                out["frames_json"] = _rel(f)
            elif key in ("mp4", "mkv"):
                out.setdefault("videos", []).append(_rel(f))
            elif key == "xosc":
                out["xosc"] = _rel(f)
        return out


# --------------------------------------------------------------------------- #
#  状态
# --------------------------------------------------------------------------- #
def _server_status(deep: bool = False) -> Dict[str, Any]:
    running = _pgrep("CarlaUE4-Linux-Shipping")
    port = _port_open(CARLA_PORT)
    info: Dict[str, Any] = {
        "running": bool(running), "port_open": bool(port), "port": CARLA_PORT,
        "ready": bool(running and port),
        "live": _http_ok(f"http://127.0.0.1:{LIVE_PORT}/health"),
        "live_port": LIVE_PORT,
        "maps": MAPS,
        "carla_home": str(CARLA_HOME),
        "server_log": _tail(CARLA_HOME / "logs" / "carla_server.log", 12),
        "gpu": _gpu_info(),
        "auto_release_seconds": AUTO_RELEASE_SECONDS,
        "engine": {"running": _engine_alive(), "port": ENGINE_PORT,
                   "log": _engine_log_tail(8)},
    }
    if deep and info["ready"]:
        try:
            r = subprocess.run(
                [str(CARLA_PY), "-c",
                 "import carla;c=carla.Client('127.0.0.1',%d);c.set_timeout(20);"
                 "w=c.get_world();print(w.get_map().name, len(w.get_actors()))" % CARLA_PORT],
                capture_output=True, text=True, timeout=60, cwd=str(CARLA_HOME))
            out = (r.stdout or "").strip().splitlines()
            if out:
                parts = out[-1].split()
                info["map"] = parts[0]
                info["actors"] = int(parts[1]) if len(parts) > 1 else None
        except Exception as e:                           # noqa: BLE001
            info["deep_error"] = f"{type(e).__name__}: {e}"
    return info


@router.get("/status")
async def carla_status(deep: int = 0):
    """CARLA 服务/直播/作业总览。deep=1 时额外连一次 RPC 拿当前地图。"""
    st = _server_status(deep=bool(deep))
    with _JOBS_LOCK:
        jobs = [j.info(tail=0) for j in sorted(_JOBS.values(), key=lambda x: -x.started)]
    st["jobs"] = jobs[:20]
    return st


class ServerRequest(BaseModel):
    action: str = "start"                 # start | stop
    quality: str = "High"
    map: str = "Town10HD_Opt"
    force: bool = False                   # 显存不够时也强行启动


def _vram_guard() -> Optional[str]:
    """启动前检查显存；不够就返回一句人能看懂的说明。"""
    gpu = _gpu_info()
    if not gpu:
        return None
    if gpu["free_mb"] >= CARLA_MIN_FREE_MB:
        return None
    top = "、".join(f"{p['name']}({p['used_mb']}MB)" for p in gpu["procs"][:3]) or "无"
    return (f"显存只剩 {gpu['free_mb']}MB，CARLA 需要约 5-6GB。"
            f"当前占用：{top}。请先释放（SAM3D/其它任务）再启动 CARLA。")


@router.post("/release")
async def carla_release():
    """释放显存：停 CARLA server + 直播桥接（UE4 会忽略 SIGTERM，这里直接 SIGKILL）。"""
    _cancel_idle_release()
    before = _gpu_info() or {}
    await asyncio.to_thread(_stop_carla_processes, stop_live=True)
    await asyncio.sleep(3)
    after = _gpu_info() or {}
    freed = (before.get("free_mb", 0) or 0) and (after.get("free_mb", 0) - before.get("free_mb", 0))
    return {"success": True, "freed_mb": freed, "gpu": after, "status": _server_status()}


@router.post("/server")
async def carla_server(req: ServerRequest):
    """启动/停止 CARLA server（启动后自动等它就绪，最多 ~3 分钟）。"""
    if req.action == "stop":
        return {"success": True, "status": (await carla_release())["status"]}
    # start
    if _server_status()["ready"]:
        with _IDLE_LOCK:
            _IDLE["last_start"] = time.time()
        return {"success": True, "already": True, "status": _server_status(deep=True)}
    warn = _vram_guard()
    if warn and not req.force:
        return {"success": False, "detail": warn, "gpu": _gpu_info()}
    env = dict(os.environ)
    env["QUALITY"] = req.quality
    env["CARLA_PORT"] = str(CARLA_PORT)
    _busy_begin()
    with _IDLE_LOCK:
        _IDLE["last_start"] = time.time()
    try:
        # 用线程跑：起 CARLA 最长要一两分钟，绝不能把事件循环堵住（否则整个 studio 页面转圈）
        await asyncio.to_thread(
            subprocess.run, ["bash", str(SH_SERVER), "--daemon"],
            capture_output=True, text=True, timeout=120, env=env, cwd=str(BRIDGE))
    finally:
        _busy_end()
    for _ in range(30):
        if _server_status()["ready"]:
            with _IDLE_LOCK:
                _IDLE["last_start"] = time.time()
            return {"success": True, "status": _server_status(deep=True), "warning": warn}
        await asyncio.sleep(6)
    return {"success": False, "detail": "CARLA 起不来（看 server_log）",
            "status": _server_status(deep=True)}


@router.post("/fix_vulkan")
async def fix_vulkan():
    """自检/修复容器里残缺的 NVIDIA 图形用户态（跑一次约 10s~2min）。"""
    r = await asyncio.to_thread(
        subprocess.run, ["bash", str(SH_FIX)], capture_output=True, text=True, timeout=900)
    return {"success": r.returncode == 0, "rc": r.returncode,
            "output": (r.stdout or "")[-4000:] + (r.stderr or "")[-1000:]}


# --------------------------------------------------------------------------- #
#  场景列表
# --------------------------------------------------------------------------- #
def _world_summary(path: Path, world: Optional[Dict[str, Any]] = None,
                   err: Optional[str] = None) -> Dict[str, Any]:
    d: Dict[str, Any] = {"path": _rel(path), "name": path.stem, "folder": path.parent.name,
                         "size_kb": round(path.stat().st_size / 1024, 1),
                         "mtime": datetime.fromtimestamp(path.stat().st_mtime).strftime("%m-%d %H:%M")}
    if world:
        tracks = world.get("tracks") or []
        d.update({"num_tracks": len(tracks),
                  "num_dynamic": sum(1 for t in tracks if len(t.get("poses") or {}) >= 3),
                  "roles": sorted({str(t.get("role")) for t in tracks}),
                  "dt": (world.get("frame") or {}).get("dt")})
    if err:
        d["error"] = err
    return d


@router.get("/scenarios")
async def list_scenarios(limit: int = 200):
    """列出可跑的导出场景（output/export、output/demo* 下的 .world.json/.xosc）。"""
    pats = ["export/*/*.world.json", "export/*/*.xosc", "demo*/*/*.world.json",
            "demo*/*/*.xosc", "corner_cases/*/videos/*.mp4"]
    found: List[Path] = []
    for pat in pats:
        found += sorted(OUTPUT.glob(pat))
    items: List[Dict[str, Any]] = []
    for p in found[:limit]:
        if p.suffix.lower() == ".mp4":
            items.append({"path": _rel(p), "name": p.stem, "folder": p.parent.name,
                          "kind": "video", "size_kb": round(p.stat().st_size / 1024, 1),
                          "mtime": datetime.fromtimestamp(p.stat().st_mtime).strftime("%m-%d %H:%M")})
            continue
        w = None
        err = None
        with_same = p.with_suffix(".world.json")
        if with_same.exists():
            try:
                w = json.loads(with_same.read_text(encoding="utf-8"))
            except Exception as e:                       # noqa: BLE001
                err = f"world.json 解析失败：{e}"
        it = _world_summary(p, w, err)
        it["kind"] = p.suffix.lower().lstrip(".")
        # 已经渲染过的产物
        carla_dir = OUTPUT / "carla" / f"{p.parent.name}_{p.stem}"
        if carla_dir.exists():
            vids = sorted(carla_dir.glob("*.mp4")) + sorted(carla_dir.glob("*.mkv"))
            if vids:
                it["rendered_video"] = _rel(vids[0])
        items.append(it)
    items.sort(key=lambda x: x.get("mtime", ""), reverse=True)
    return {"count": len(items), "items": items,
            "carla_outputs": sorted([_rel(p) for p in (OUTPUT / "carla").glob("*")])[:40]
            if (OUTPUT / "carla").exists() else []}


# --------------------------------------------------------------------------- #
#  渲染作业
# --------------------------------------------------------------------------- #
class RenderOptions(BaseModel):
    scenario: Optional[str] = None          # 直接给 world.json/.xosc 路径（优先）
    # 或者：用「当前编辑场景」现场导出
    scene_id: Optional[str] = None
    scenario_type: Optional[str] = None     # 空=导出当前场景（含已编辑轨迹）
    roles: Optional[Dict[str, int]] = None
    seed: int = 7
    start_frame: int = 0
    num_frames: int = 20
    export_name: Optional[str] = None
    all_tracks: bool = False
    export_fps: float = 10.0                # 数据采样率（重建场景是 10Hz，别和下面的播放 fps 混）
    # 渲染参数
    map: str = "Town10HD_Opt"
    fps: float = 20.0
    cameras: str = "chase,birdseye"
    cam_size: str = "960x540"
    focus: str = "crash"
    actors: str = "dynamic"
    weather: str = "ClearNoon"
    ego_mode: str = "playback"
    vertical: str = "flat"
    road_snap: str = "z"                    # 把车贴回路面：z（默认，保几何）/ lane / none
    pre_roll: float = 1.0
    post_roll: float = 1.5
    export_xosc: bool = False
    auto_server: bool = True
    out_dir: Optional[str] = None


async def _export_current_scene(req: RenderOptions) -> str:
    """把「当前场景状态」落成 world.json（直接调用 api_server 注入进来的导出端点）。"""
    export_fn = _STATE.get("export_scenario")
    req_cls = _STATE.get("export_request_cls")
    if export_fn is None or req_cls is None:
        raise HTTPException(status_code=503, detail="导出接口没注入成功，请重启 studio 后端")
    name = req.export_name or f"carla_ui_{req.scene_id}_{datetime.now().strftime('%H%M%S')}"
    body = req_cls(
        scene_id=req.scene_id, scenario_type=req.scenario_type or None, roles=req.roles,
        seed=req.seed, start_frame=req.start_frame, num_frames=req.num_frames,
        fps=req.export_fps, name=name, out_dir=None, all_tracks=req.all_tracks)
    try:
        res = await export_fn(body)
    except HTTPException as e:
        if e.status_code == 404:
            loaded = list((_STATE.get("studio_state") or {}).get("scenes", {}).keys())
            raise HTTPException(status_code=409, detail=(
                f"后端内存里找不到场景 {req.scene_id}（studio 后端重启过 → 内存里的场景丢了）。"
                f"请在左侧重新「加载场景」再试；当前已加载：{loaded or '（无）'}。"
                f"  —— 或者改用「② 批量生成」表格里每行的「CARLA」：那条路径直接吃实例的 "
                f"world.json，不依赖内存里的场景。"))
        raise
    return res["files"]["world_json"]


class InstanceApplyRequest(BaseModel):
    scene_id: str
    instance: Dict[str, Any]                   # 直接给「② 批量生成」manifest 里的那一条
    reset_scene: bool = True                   # 先把场景重置成干净状态（推荐）
    fps: Optional[float] = None


@router.post("/apply_instance")
async def carla_apply_instance(req: InstanceApplyRequest):
    """把批量生产里的某一条实例「还原」进当前场景（确定性重放：type + roles + seed）。

    这样就能在 ③ 闭环评估里逐帧看它、继续编辑，或者用「用当前编辑的场景」送 ④ CARLA。
    批量生成本身是带 seed 的确定性过程，所以同样的输入能重放出同一条实例。
    """
    import corner_case
    inst = req.instance or {}
    st = str(inst.get("scenario_type") or "").strip()
    # roles 的 key 是角色名（"ego"/"attacker"/"victim"...），value 才是 track_id
    roles = {str(k): int(v) for k, v in (inst.get("roles") or {}).items()}
    if not st or not roles:
        raise HTTPException(status_code=400, detail="实例缺少 scenario_type / roles，无法还原")
    get_scene = _STATE.get("get_scene")
    get_tm = _STATE.get("get_track_manager")
    tm_cls = _STATE.get("track_manager_cls")
    state = _STATE.get("studio_state") or {}
    if not (get_scene and get_tm and tm_cls):
        raise HTTPException(status_code=503, detail="场景接口没注入成功，请重启 studio 后端")
    renderer = get_scene(req.scene_id)
    tm = get_tm(req.scene_id)
    fps = float(req.fps or inst.get("fps") or 10.0)
    start = int(inst.get("start_frame") or 0)
    frames = int(inst.get("num_frames") or 20)
    if req.reset_scene:
        tm = tm_cls(renderer)
        state.setdefault("track_managers", {})[req.scene_id] = tm
    tm.fps = fps
    tm.push_history()
    try:
        res = corner_case.generate(tm, st, roles, start, frames, enable_physics=True,
                                   fps=fps, sampling_seed=int(inst.get("seed") or 0))
    except Exception as e:                                   # noqa: BLE001
        tm.undo()
        raise HTTPException(status_code=500, detail=f"还原实例失败：{type(e).__name__}: {e}")
    return {"success": True, "scenario_type": st, "roles": roles, "seed": inst.get("seed"),
            "instance_id": inst.get("instance_id"), "start_frame": start, "num_frames": frames,
            "fps": fps, "collision_frame": res.get("collision_frame"),
            "affected_tracks": res.get("affected_tracks"),
            "synthesized_tracks": res.get("synthesized_tracks"),
            "reset_scene": bool(req.reset_scene),
            "hint": "场景里现在已经出现这条事故轨迹：可以继续编辑，或直接「用当前编辑的场景」送 ④ CARLA"}


@router.post("/render")
async def carla_render(req: RenderOptions):
    """起一个 CARLA 渲染作业（后台跑，前端轮询 /jobs/{id} 看进度和产物）。"""
    if req.auto_server and not _server_status()["ready"]:
        r = await carla_server(ServerRequest(action="start", map=req.map))
        if not r.get("success"):
            raise HTTPException(status_code=503,
                                detail=r.get("detail") or "CARLA server 起不来，请先看 /api/carla/status")
    scen = req.scenario
    if not scen:
        if not req.scene_id:
            raise HTTPException(status_code=400, detail="要么给 scenario 路径，要么给 scene_id")
        scen = await _export_current_scene(req)
    sp = Path(scen)
    if not sp.is_absolute():
        sp = REPO / sp
    if not sp.exists():
        raise HTTPException(status_code=404, detail=f"场景不存在：{scen}")

    name = sp.stem
    tag = req.export_name or (name if not req.scenario else f"{sp.parent.name}_{name}")
    out_dir = Path(req.out_dir) if req.out_dir else (OUTPUT / "carla" / tag)
    cmd = ["bash", str(SH_BRIDGE), "--scenario", str(sp), "--out-dir", str(out_dir),
           "--map", req.map, "--fps", str(req.fps), "--cameras", req.cameras,
           "--cam-size", req.cam_size, "--focus", req.focus, "--actors", req.actors,
           "--weather", req.weather, "--ego-mode", req.ego_mode, "--vertical", req.vertical,
           "--road-snap", req.road_snap,
           "--pre-roll", str(req.pre_roll), "--post-roll", str(req.post_roll),
           "--reload-world", "auto", "--every", "10"]
    if req.export_name:
        cmd += ["--name", req.export_name]
    if req.export_xosc:
        cmd += ["--export-xosc"]
    job = Job("render", cmd, out_dir=out_dir, name=name,
              title=f"CARLA 渲染 {name}（{req.map}）").start()
    return {"success": True, "job": job.info()}


class ExportXoscRequest(BaseModel):
    scenario: str
    map: str = "Town10HD_Opt"
    out_dir: Optional[str] = None
    auto_server: bool = True


@router.post("/export_xosc")
async def carla_export_xosc(req: ExportXoscRequest):
    """只导出「CARLA 对齐版」OpenSCENARIO（不渲染），给 ScenarioRunner 等标准工具用。"""
    if req.auto_server and not _server_status()["ready"]:
        r = await carla_server(ServerRequest(action="start", map=req.map))
        if not r.get("success"):
            raise HTTPException(status_code=503, detail=r.get("detail") or "CARLA server 起不来")
    sp = Path(req.scenario)
    if not sp.is_absolute():
        sp = REPO / sp
    if not sp.exists():
        raise HTTPException(status_code=404, detail=f"场景不存在：{req.scenario}")
    out_dir = Path(req.out_dir) if req.out_dir else (OUTPUT / "carla" / f"{sp.parent.name}_{sp.stem}")
    cmd = ["bash", str(SH_BRIDGE), "--scenario", str(sp), "--out-dir", str(out_dir),
           "--map", req.map, "--export-only", "--export-xosc", "--reload-world", "auto"]
    job = Job("export_xosc", cmd, out_dir=out_dir, name=sp.stem,
              title=f"导出 CARLA 对齐版 xosc：{sp.stem}").start()
    return {"success": True, "job": job.info()}


class ScenarioRunnerRequest(BaseModel):
    xosc: str
    timeout: int = 120
    extra: str = ""


@router.post("/scenario_runner")
async def carla_scenario_runner(req: ScenarioRunnerRequest):
    """用 CARLA 官方 ScenarioRunner 跑对齐版 xosc（逻辑/评测路径，不渲染画面）。"""
    p = Path(req.xosc)
    if not p.is_absolute():
        p = REPO / p
    if not p.exists():
        raise HTTPException(status_code=404, detail=f"xosc 不存在：{req.xosc}")
    if not _server_status()["ready"]:
        r = await carla_server(ServerRequest(action="start"))
        if not r.get("success"):
            raise HTTPException(status_code=503, detail="CARLA server 起不来")
    cmd = ["bash", str(SH_SR), str(p), "--timeout", str(req.timeout)]
    if req.extra:
        cmd += req.extra.split()
    job = Job("scenario_runner", cmd, out_dir=p.parent, name=p.stem,
              title=f"ScenarioRunner：{p.stem}").start()
    return {"success": True, "job": job.info()}


@router.get("/jobs")
async def list_jobs(limit: int = 30):
    with _JOBS_LOCK:
        js = sorted(_JOBS.values(), key=lambda j: -j.started)[:limit]
    return {"jobs": [j.info(tail=0) for j in js]}


@router.get("/jobs/{job_id}")
async def get_job(job_id: str, tail: int = 80):
    j = _JOBS.get(job_id)
    if not j:
        raise HTTPException(status_code=404, detail="作业不存在")
    return j.info(tail=tail)


@router.post("/jobs/{job_id}/cancel")
async def cancel_job(job_id: str):
    j = _JOBS.get(job_id)
    if not j:
        raise HTTPException(status_code=404, detail="作业不存在")
    j.cancel()
    return {"success": True, "job": j.info(tail=0)}


# --------------------------------------------------------------------------- #
#  文件访问（视频/JSON/xosc，支持 Range 拖动进度条）
# --------------------------------------------------------------------------- #
@router.get("/file")
async def carla_file(path: str, request: Request, download: int = 0, raw: int = 0):
    p = _safe_output_path(path)
    # 浏览器只能放 H.264/HEVC/VP9/AV1；CARLA 桥接默认写的是 mp4v，这里按需转码（结果缓存）
    if not raw and p.suffix.lower() in (".mp4", ".mkv"):
        try:
            import asyncio
            p = await asyncio.get_event_loop().run_in_executor(None, _ensure_browser_mp4, p)
        except Exception:                                # noqa: BLE001
            pass
    ctype = {".mp4": "video/mp4", ".mkv": "video/x-matroska", ".json": "application/json",
             ".xosc": "application/xml", ".xml": "application/xml", ".png": "image/png",
             ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}.get(p.suffix.lower(),
                                                             "application/octet-stream")
    if download:
        return FileResponse(str(p), media_type=ctype, filename=p.name)
    if p.suffix.lower() not in (".mp4", ".mkv"):
        return FileResponse(str(p), media_type=ctype,
                            headers={"Accept-Ranges": "bytes", "Cache-Control": "no-store"})
    size = p.stat().st_size
    rng = request.headers.get("range") or request.headers.get("Range")
    if rng and rng.startswith("bytes="):
        try:
            spec = rng.split("=", 1)[1].split(",")[0].strip()
            s, _, e = spec.partition("-")
            start = int(s) if s else 0
            end = int(e) if e else size - 1
        except ValueError:
            start, end = 0, size - 1
        start = max(0, min(start, max(0, size - 1)))
        end = max(start, min(end, max(0, size - 1)))
        length = end - start + 1
        with open(p, "rb") as f:
            f.seek(start)
            data = f.read(length)
        return Response(content=data, status_code=206, media_type=ctype,
                        headers={"Content-Range": f"bytes {start}-{end}/{size}",
                                 "Accept-Ranges": "bytes", "Content-Length": str(length)})
    return FileResponse(str(p), media_type=ctype, headers={"Accept-Ranges": "bytes"})


# --------------------------------------------------------------------------- #
#  实时直播（MJPEG，后端代理 → 前端 <img> 直接看，不用再转发 8090）
# --------------------------------------------------------------------------- #
class LiveRequest(BaseModel):
    action: str = "start"                 # start | stop
    scenario: Optional[str] = None
    scene_id: Optional[str] = None
    scenario_type: Optional[str] = None
    seed: int = 7
    start_frame: int = 0
    num_frames: int = 20
    all_tracks: bool = False
    export_fps: float = 10.0
    map: str = "Town10HD_Opt"
    fps: float = 20.0
    cameras: str = "chase,birdseye"
    cam_size: str = "960x540"
    focus: str = "crash"
    actors: str = "dynamic"
    weather: str = "ClearNoon"
    hold_end: float = 3.0
    road_snap: str = "z"
    auto_server: bool = True


_live_proc: Optional[subprocess.Popen] = None
_live_lock = threading.Lock()


@router.post("/live")
async def carla_live(req: LiveRequest):
    """开/关无限循环的 CARLA 直播（内部 MJPEG :8090，前端通过 /api/carla/live/mjpeg 看）。"""
    global _live_proc
    if req.action == "stop":
        with _live_lock:
            try:
                subprocess.run(["pkill", "-9", "-f", "carla_scenario_bridge"],
                               capture_output=True, timeout=20)
            except Exception:                            # noqa: BLE001
                pass
            _live_proc = None
        _schedule_idle_release(30)                       # 直播停了，过一会把 CARLA 也放掉
        return {"success": True, "live": False, "gpu": _gpu_info()}
    if req.auto_server and not _server_status()["ready"]:
        r = await carla_server(ServerRequest(action="start", map=req.map))
        if not r.get("success"):
            raise HTTPException(status_code=503, detail=r.get("detail") or "CARLA server 起不来")
    _cancel_idle_release()
    scen = req.scenario
    if not scen:
        if not req.scene_id:
            raise HTTPException(status_code=400, detail="要么给 scenario，要么给 scene_id")
        scen = await _export_current_scene(RenderOptions(
            scene_id=req.scene_id, scenario_type=req.scenario_type, seed=req.seed,
            start_frame=req.start_frame, num_frames=req.num_frames, fps=req.fps,
            export_fps=req.export_fps, all_tracks=req.all_tracks))
    sp = Path(scen)
    if not sp.is_absolute():
        sp = REPO / sp
    if not sp.exists():
        raise HTTPException(status_code=404, detail=f"场景不存在：{scen}")
    with _live_lock:
        try:
            subprocess.run(["pkill", "-9", "-f", "carla_scenario_bridge"],
                           capture_output=True, timeout=20)
        except Exception:                                # noqa: BLE001
            pass
        time.sleep(1)
        cmd = ["bash", str(SH_BRIDGE), "--scenario", str(sp), "--map", req.map,
               "--fps", str(req.fps), "--cameras", req.cameras, "--cam-size", req.cam_size,
               "--focus", req.focus, "--actors", req.actors, "--weather", req.weather,
               "--road-snap", req.road_snap,
               "--loop", "0", "--hold-end", str(req.hold_end),
               "--live-http", str(LIVE_PORT), "--reload-world", "auto", "--every", "500"]
        log = CARLA_HOME / "logs" / "live_bridge.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        fh = open(log, "w")
        _live_proc = subprocess.Popen(cmd, cwd=str(BRIDGE), stdout=fh, stderr=subprocess.STDOUT,
                                      start_new_session=True)
    for _ in range(30):
        if _http_ok(f"http://127.0.0.1:{LIVE_PORT}/health"):
            return {"success": True, "live": True, "url": "/api/carla/live/mjpeg",
                    "log": _tail(CARLA_HOME / "logs" / "live_bridge.log", 12)}
        time.sleep(2)
    return {"success": False, "detail": "直播没起来", "log": _tail(CARLA_HOME / "logs" / "live_bridge.log", 20)}


@router.get("/live/mjpeg")
async def carla_live_mjpeg():
    """把内部 8090 的 MJPEG 流转发给前端（<img src=".../api/carla/live/mjpeg">）。"""
    if not _http_ok(f"http://127.0.0.1:{LIVE_PORT}/health"):
        raise HTTPException(status_code=503, detail="直播未启动")

    def gen():
        try:
            r = urllib.request.urlopen(f"http://127.0.0.1:{LIVE_PORT}/stream", timeout=15)
            while True:
                chunk = r.read(32768)
                if not chunk:
                    break
                yield chunk
            r.close()
        except Exception:                                # noqa: BLE001
            return
    return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame",
                             headers={"Cache-Control": "no-store"})


@router.get("/live/log")
async def carla_live_log(tail: int = 40):
    return {"running": _http_ok(f"http://127.0.0.1:{LIVE_PORT}/health"),
            "log": _tail(CARLA_HOME / "logs" / "live_bridge.log", tail)}


# --------------------------------------------------------------------------- #
#  🎮 CARLA 仿真引擎（可播放 + 可编辑的常驻会话）
# --------------------------------------------------------------------------- #
ENGINE_PORT = int(os.environ.get("CARLA_ENGINE_PORT", "8110"))
_engine_proc: Optional[subprocess.Popen] = None
_engine_lock = threading.Lock()


def _engine_http(path: str, method: str = "GET", body: Optional[Dict[str, Any]] = None,
                 timeout: float = 60.0):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(f"http://127.0.0.1:{ENGINE_PORT}{path}", data=data, method=method,
                                 headers={"Content-Type": "application/json"} if data else {})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        raw = r.read()
    return json.loads(raw.decode("utf-8")) if raw else {}


def _engine_alive() -> bool:
    try:
        _engine_http("/ping", timeout=2.0)
        return True
    except Exception:                                    # noqa: BLE001
        return False


def _engine_log_tail(n: int = 25) -> List[str]:
    return _tail(CARLA_HOME / "logs" / "carla_engine.log", n)


def _engine_stop() -> None:
    try:
        _engine_http("/quit", "POST", {}, timeout=5)
    except Exception:                                    # noqa: BLE001
        pass
    time.sleep(1.5)
    try:
        subprocess.run(["pkill", "-9", "-f", "carla_engine"], capture_output=True, timeout=15)
    except Exception:                                    # noqa: BLE001
        pass


class EngineStartRequest(BaseModel):
    scenario: Optional[str] = None
    scene_id: Optional[str] = None
    scenario_type: Optional[str] = None
    seed: int = 7
    start_frame: int = 0
    num_frames: int = 20
    all_tracks: bool = False
    export_fps: float = 10.0
    map: str = "Town10HD_Opt"
    fps: float = 20.0
    cam_size: str = "960x540"
    weather: str = "ClearNoon"
    actors: str = "dynamic"
    road_snap: str = "z"
    auto_server: bool = True


@router.post("/engine/start")
async def carla_engine_start(req: EngineStartRequest):
    """起一个常驻的 CARLA 仿真引擎会话：能播放、拖时间轴、编辑 actor、存成新场景。"""
    global _engine_proc
    if req.auto_server and not _server_status()["ready"]:
        r = await carla_server(ServerRequest(action="start", map=req.map))
        if not r.get("success"):
            raise HTTPException(status_code=503, detail=r.get("detail") or "CARLA server 起不来")
    scen = req.scenario
    if not scen:
        if not req.scene_id:
            raise HTTPException(status_code=400, detail="要么给 scenario，要么给 scene_id")
        scen = await _export_current_scene(RenderOptions(
            scene_id=req.scene_id, scenario_type=req.scenario_type, seed=req.seed,
            start_frame=req.start_frame, num_frames=req.num_frames, fps=req.fps,
            export_fps=req.export_fps, all_tracks=req.all_tracks))
    sp = Path(scen)
    if not sp.is_absolute():
        sp = REPO / sp
    if not sp.exists():
        raise HTTPException(status_code=404, detail=f"场景不存在：{scen}")
    with _engine_lock:
        _engine_stop()
        cmd = [str(CARLA_PY), str(BRIDGE / "carla_engine.py"), "--scenario", str(sp),
               "--map", req.map, "--fps", str(req.fps), "--cam-size", req.cam_size,
               "--weather", req.weather, "--actors", req.actors,
               "--road-snap", req.road_snap,
               "--control-port", str(ENGINE_PORT)]
        log = CARLA_HOME / "logs" / "carla_engine.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        fh = open(log, "w")
        env = dict(os.environ)
        env.setdefault("VK_ICD_FILENAMES", "/etc/vulkan/icd.d/nvidia_icd.json")
        _engine_proc = subprocess.Popen(cmd, cwd=str(BRIDGE), stdout=fh, stderr=subprocess.STDOUT,
                                        env=env, start_new_session=True)
    _cancel_idle_release()
    with _IDLE_LOCK:
        _IDLE["last_start"] = time.time()                 # 引擎刚起，别被空闲释放打断
    for _ in range(45):
        if _engine_alive():
            try:
                return {"success": True, "state": _engine_http("/state", timeout=25)}
            except Exception:                            # noqa: BLE001
                pass
        if _engine_proc is not None and _engine_proc.poll() is not None:
            break
        await asyncio.sleep(3)
    _engine_stop()                                        # 起不来就把卡住的引擎清掉，别留着占显存
    raise HTTPException(status_code=503,
                        detail="仿真引擎没起来：" + " | ".join(_engine_log_tail(12)))


@router.post("/engine/stop")
async def carla_engine_stop():
    _engine_stop()
    _schedule_idle_release(30)
    return {"success": True, "gpu": _gpu_info()}


@router.get("/engine/state")
async def carla_engine_state():
    try:
        return _engine_http("/state", timeout=25)
    except Exception as e:                               # noqa: BLE001
        raise HTTPException(status_code=503,
                            detail=f"仿真引擎未运行：{type(e).__name__}: {e}")


@router.get("/engine/mjpeg")
async def carla_engine_mjpeg():
    """把引擎的实时画面转成 MJPEG 流给 <img> 看（约跟引擎 fps 同步）。"""
    if not _engine_alive():
        raise HTTPException(status_code=503, detail="仿真引擎未运行")

    def gen():
        while True:
            try:
                with urllib.request.urlopen(
                        f"http://127.0.0.1:{ENGINE_PORT}/frame.jpg", timeout=10) as r:
                    data = r.read()
            except Exception:                            # noqa: BLE001
                time.sleep(0.4)
                continue
            if not data:
                time.sleep(0.05)
                continue
            yield (b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                   + str(len(data)).encode() + b"\r\n\r\n" + data + b"\r\n")
    return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame",
                             headers={"Cache-Control": "no-store"})

    return None


@router.post("/engine/{action}")
async def carla_engine_action(action: str, request: Request):
    """转发到引擎：play / frame / fps / camera / weather / actor / actor/add /
    actor/delete / actor/restore / reset / save / screenshot"""
    if action in ("start", "stop"):
        raise HTTPException(status_code=400, detail="请用 /api/carla/engine/start 或 /stop")
    body: Dict[str, Any] = {}
    try:
        raw = await request.body()
        if raw:
            body = json.loads(raw.decode("utf-8"))
    except Exception:                                    # noqa: BLE001
        body = {}
    if action == "save" and not body.get("path"):
        body["path"] = str(OUTPUT / "carla" / "engine" /
                           f"edited_{datetime.now().strftime('%H%M%S')}.world.json")
    try:
        return _engine_http(f"/{action}", "POST", body, timeout=300)
    except Exception as e:                               # noqa: BLE001
        raise HTTPException(status_code=503,
                            detail=f"引擎调用失败（{action}）：{type(e).__name__}: {e}")


# --------------------------------------------------------------------------- #
#  环境自检
# --------------------------------------------------------------------------- #
@router.get("/env")
async def carla_env():
    """给前端一个"能不能用"的总览：CARLA 装了没、py37 环境、桥接脚本、GPU Vulkan。"""
    vk = None
    try:
        r = subprocess.run(["bash", "-lc",
                            "VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json "
                            "timeout 60 vulkaninfo --summary 2>/dev/null | grep -E 'deviceName|driverInfo'"],
                           capture_output=True, text=True, timeout=90)
        vk = [l.strip() for l in (r.stdout or "").splitlines() if l.strip()]
    except Exception as e:                               # noqa: BLE001
        vk = [f"检测失败：{e}"]
    return {
        "carla_home": str(CARLA_HOME),
        "carla_installed": (CARLA_HOME / "CarlaUE4.sh").exists(),
        "python37": str(CARLA_PY),
        "python37_ok": CARLA_PY.exists(),
        "bridge_dir": str(BRIDGE),
        "scripts": {s.name: s.exists() for s in (SH_SERVER, SH_STOP, SH_BRIDGE, SH_FIX, SH_SR)},
        "vulkan": vk,
        "scenario_runner": (CARLA_HOME / "scenario_runner" / "scenario_runner.py").exists(),
        "disk_free_gb": round(shutil.disk_usage(str(CARLA_HOME)).free / 1e9, 1),
    }
