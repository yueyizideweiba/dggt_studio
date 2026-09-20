"""语言驱动的轨迹编辑：LLM 解析自然语言 → 结构化操作 → 作用到 TrackManager。

说明：没有现成的"语言驱动轨迹编辑"训练模型，这里用 LLM(Qwen，微服务 /llm) 做**意图解析**，
再用确定性几何规则执行（速度重定时 / 变道 / 转向 / 删除 / 碰撞编排），
LLM 不可用时退化为关键词规则解析。生成新车的 `insert` 由上层复用 LLaDA+SAM3D 流程。
"""
from __future__ import annotations

import json
import math
import re
import os
from typing import Any, Dict, List, Optional

import numpy as np

import nl_entity

TEXT2ENTITY_URL = os.environ.get("TEXT2ENTITY_URL", "http://127.0.0.1:8002")


# ==================== 场景摘要 ====================

def scene_summary(tm, frame_idx: int = 0) -> List[Dict[str, Any]]:
    rows = []
    for o in tm.get_frame_objects(int(frame_idx)):
        tid = int(o["track_id"])
        p = np.asarray(o["pose_world"], dtype=np.float64)
        row = {
            "id": tid,
            "type": o.get("type") or "未知",
            "x": round(float(p[0, 3]), 2),
            "z": round(float(p[2, 3]), 2),
            "dims": [round(float(d), 2) for d in (o.get("dimensions") or [])],
        }
        if tm.is_ego_track(tid):
            # 把主车也告诉 LLM，否则它不知道"主车/自车"该用哪个 id
            row["type"] = "主车(自车/ego)"
            rows.append(row)
            continue
        rows.append(row)
    return rows


# ==================== LLM 规划 ====================

PLAN_PROMPT = """你是自动驾驶场景编辑助手。当前场景里的车辆（JSON，帧 {frame}）：
{scene}

用户要求：{instruction}

请把它拆成可执行操作，**只输出一个 JSON**（不要 markdown、不要解释）：
{{"ops": [
 {{"op":"speed","track":<id>,"speed_mps":<数>,"start_frame":<帧>,"end_frame":<帧>}},
 {{"op":"lane_change","track":<id>,"lateral":<米，车体右侧为正>,"start_frame":<帧>,"duration":<帧数>}},
 {{"op":"turn","track":<id>,"yaw_deg":<相对角度，正=左转>,"start_frame":<帧>,"duration":<帧数>}},
 {{"op":"remove","track":<id>}},
 {{"op":"collide","a":<attacker id>,"b":<victim id>,"frame":<碰撞帧>}},
 {{"op":"insert","prompt":"<英文外观描述>","track":null,"mode":"ahead","distance":<米>,"speed":<m/s>,"num_frames":<帧数>}}
（insert 的 num_frames 就是"新物体生成多少帧轨迹"：用户说了帧数/秒数就按它填，
  1 秒 ≈ {fps} 帧，例如"持续 4 秒"→40、"跑 60 帧"→60；没说就填 20。
  帧数不要超过 {max_frames}。
  insert 的 mode 只能填三个字面量之一：ahead（同向、在自车前方）/oncoming（对向）/
  roadside（路肩），**不要把 "ahead|oncoming|roadside" 原样抄下来**。
  collide 的 frame 是"撞上的那一帧"，要放在整段轨迹的中间偏后（大约 60% 处），
  这样 40 帧里才有"撞前 → 撞上 → 撞后"三段；不要填在最后一帧。）
]}}
规则：track 用上面 JSON 里的 id（**主车/自车的 id 是 900000**，"主车/自车/本车/ego"都指它）；
帧号 0~{num_frames_minus1}；"撞/相撞/追尾/别车/加塞"用 collide
（同一辆车可同时给 lane_change/speed 让它更真实）；"加/生成/放入一辆车"用 insert；
① 只有用户明确要求"新增/生成/添加一辆新车"时才输出 insert，只是让已有车相撞/变道时**不要** insert；
② 若肇事车就是本条指令要**新建**的那辆，collide 的 "a" 必须写 null（会绑定到新生成的车上），
"b" 写被撞的那辆已有车的 id；③ 只输出 JSON，不要 markdown、不要解释。"""


def _llm_text(prompt: str, max_new_tokens: int = 768) -> str:
    """调用微服务的 /llm（Qwen 纯文本推理）。失败时重试并打印原因，返回空串触发规则兜底。"""
    import time as _t

    import httpx
    last = ""
    for attempt in range(3):
        try:
            with httpx.Client(timeout=300.0) as c:
                r = c.post(f"{TEXT2ENTITY_URL.rstrip('/')}/llm",
                           json={"prompt": prompt, "max_new_tokens": max_new_tokens})
            if r.status_code == 200:
                return r.json().get("text") or ""
            last = f"HTTP {r.status_code}: {r.text[:200]}"
        except Exception as e:  # noqa: BLE001
            last = repr(e)
        _t.sleep(2.0 * (attempt + 1))
    print(f"[traj_llm] /llm 调用失败（已退化为关键词规则解析）: {last}", flush=True)
    return ""


EGO_TRACK_ID = 900000

# 明确写成"已有车去撞新车"的情况（用于把肇事/被撞角色反过来）：
#   例："让 T:1 撞上这辆新车" / "T2 撞它" / "T1 撞向生成的车"
_EXISTING_HITS_NEW_RE = re.compile(
    r"T\s*[:：]?\s*\d+[^，。;；]{0,10}撞(?:上|到|向|了)?[^，。;；]{0,8}"
    r"(?:该|这|那|它|新|生成|红色轿车|蓝色轿车|轿车|车|一?辆)")

# 明确写成"新车去撞已有车"（"新车撞上 T:1" / "让它撞 T1" / "这辆生成的车撞向 T:2"）
_NEW_HITS_EXISTING_RE = re.compile(
    r"(?:新车|新生成的车|生成的车|该车|这辆车|这辆|它)[^，。;；]{0,10}撞(?:上|到|向|了)?"
    r"[^，。;；]{0,10}T\s*[:：]?\s*\d+")

# "新建车角色"的可选覆盖值
NEW_ROLE_AUTO = "auto"
NEW_ROLE_ATTACKER = "attacker"
NEW_ROLE_VICTIM = "victim"


def decide_new_role(text: str, has_victim: bool, new_role: str = NEW_ROLE_AUTO):
    """决定"新车 vs 已有车"里谁是肇事车。

    **不再默认新车一定是肇事车**。优先级：
      1. 调用方显式指定（界面上的「新建车角色」）；
      2. 指令里的明确写法：
         - "T:1 撞上这辆新车 / T2 撞它" → 已有车是肇事车；
         - "新车/它 撞上 T:1"          → 新车是肇事车；
      3. 都没有（例如"T1 和该红色轿车相撞"这种**对称说法**）→ 取一个默认（新车去撞），
         但会在 `warnings` 里说明"这是默认，不是你说的"，并告诉你怎么反过来。

    Returns: (role, reason)；role ∈ {"attacker","victim"}
    """
    if new_role in (NEW_ROLE_ATTACKER, NEW_ROLE_VICTIM):
        return new_role, "explicit-param"
    if _EXISTING_HITS_NEW_RE.search(text):
        return NEW_ROLE_VICTIM, "explicit-text"      # 已有车撞新车 → 新车是被撞方
    if _NEW_HITS_EXISTING_RE.search(text):
        return NEW_ROLE_ATTACKER, "explicit-text"
    # "…撞上 T:1"：撞字之后才出现已有车、之前没有 → 新车去撞它（也是明确写法）
    m_crash = _KEY_CRASH_RE.search(text)
    if m_crash:
        before, after = text[:m_crash.start()], text[m_crash.start():]
        if _TID_RE.search(after) and not _TID_RE.search(before):
            return NEW_ROLE_ATTACKER, "explicit-text"
    if not has_victim:
        return NEW_ROLE_ATTACKER, "no-victim"
    return NEW_ROLE_ATTACKER, "ambiguous"


KEY_LANE = ("变道", "换道", "别车", "加塞", "靠边", "lane")
KEY_SPEED = ("加速", "减速", "慢", "快", "速度", "刹车", "stop", "speed")
KEY_TURN = ("转向", "转弯", "左转", "右转", "掉头", "turn")
KEY_REMOVE = ("删", "移除", "去掉", "清除", "消除", "消失", "remove", "delete")
KEY_CRASH = ("撞", "相撞", "追尾", "碰", "collide", "crash")
KEY_INSERT = ("生成", "增加", "添加", "放入", "加一辆", "insert", "add")


_ACTION_CUT = re.compile(
    r"(在?\s*T\s*[:：]?\s*\d+|编号\s*\d+|相撞|撞|变道|换道|转弯|转向|掉头|减速|加速|刹车|停下|停车|驶入|插入"
    r"|持续|时长|多少帧|跑|行驶|运动|车速|帧|秒)")


def parse_num_frames(text: str, fps: float = 10.0):
    """从自然语言里解析"生成的轨迹要多少帧"。

    支持：`40帧` / `40 帧` / `4秒`（1 秒 ≈ fps 帧，默认 10） / `持续 40`（无单位按帧）。
    先认"秒"，再认"帧"，最后认"持续/时长/跑 N"——否则"持续 2 秒"会被当成 2 帧。

    Returns: (帧数 或 None, 说明)
    """
    t = str(text or "")
    m = re.search(r"(\d+(?:\.\d+)?)\s*(?:秒|秒钟|secs?|seconds?)(?![a-zA-Z])", t)
    if m:
        n = int(round(float(m.group(1)) * float(fps)))
        return max(1, n), "%s 秒 × %.0f fps" % (m.group(1), fps)
    m = re.search(r"(\d+)\s*(?:帧|frames?)", t)
    if m:
        return int(m.group(1)), "帧"
    m = re.search(r"(?:持续|时长|跑|运动|轨迹)\s*(\d+)", t)
    if m:
        return int(m.group(1)), "按帧（无单位）"
    return None, None


def rule_insert_prompt(text: str) -> str:
    """规则兜底时给文生图的提示词：只留"物体描述"，砍掉动作子句与开头的动词。

    否则整句"生成一辆对向来的红色轿车在T100000转弯时相撞"会被原样丢给
    LLaDA-Image，出图必然跑偏。
    """
    t = str(text or "")
    m = _ACTION_CUT.search(t)
    if m and m.start() > 0:
        t = t[:m.start()]
    for pre in ("帮我", "给我", "请", "生成", "添加", "增加", "放入", "加一辆", "加个", "来"):
        if t.startswith(pre):
            t = t[len(pre):]
    t = t.strip(" ，,。.、:：")
    return t or str(text or "").strip()


def rule_plan(instruction: str, summary: List[Dict[str, Any]], num_frames: int = 30,
              fps: float = 10.0) -> List[Dict[str, Any]]:
    """LLM 不可用时的兜底：关键词 → 操作；默认作用到场景里第一辆"车"。"""
    cars = [r for r in summary if r.get("dims") and len(r["dims"]) == 3 and 2.0 <= r["dims"][2] <= 8.0]
    tgt = (cars[0]["id"] if cars else (summary[0]["id"] if summary else None))
    other = (cars[1]["id"] if len(cars) > 1 else None)
    ops: List[Dict[str, Any]] = []
    text = instruction or ""
    inserted = False
    if any(k in text for k in KEY_INSERT):
        mode = "oncoming" if ("对向" in text or "迎面" in text or "逆行" in text) else "ahead"
        nf, _why = parse_num_frames(text, fps=fps)
        ops.append({"op": "insert", "prompt": rule_insert_prompt(text), "mode": mode,
                    "distance": 14.0, "speed": 0.0,
                    "num_frames": int(nf) if nf else min(20, num_frames)})
        inserted = True

    # 要生成新车时**不要再去猜别的车的动作**：规则兜底原来会把"变道/相撞"挂到
    # 随便两辆既有车上（实测出现"莫名其妙 T0 和 T1 相撞"）。事故应该由新车引发：
    # 涉及新车的 op 用 track=None / a=None 占位，上层拿到新 id 后回填；
    # 被撞的那辆已有车就取指令里写出来的 id。
    if inserted and any(k in text for k in KEY_CRASH):
        m = _TID_RE.search(text)
        victim = int(m.group(1) or m.group(2)) if m else None
        if any(k in text for k in KEY_LANE):
            ops.append({"op": "lane_change", "track": None, "lateral": 3.5,
                        "start_frame": 0, "duration": max(4, num_frames // 3)})
        if victim is not None:
            ops.append({"op": "collide", "a": None, "b": victim,
                        "frame": max(3, num_frames - 3)})
        return ops

    if any(k in text for k in KEY_CRASH):
        if tgt is not None and other is not None:
            ops.append({"op": "lane_change", "track": other, "lateral": 3.5,
                        "start_frame": 0, "duration": max(4, num_frames // 3)})
            ops.append({"op": "collide", "a": other, "b": tgt, "frame": max(3, num_frames - 3)})
    elif any(k in text for k in KEY_LANE) and tgt is not None:
        ops.append({"op": "lane_change", "track": tgt, "lateral": 3.5,
                    "start_frame": 0, "duration": max(4, num_frames // 3)})
    if any(k in text for k in KEY_SPEED) and tgt is not None:
        ops.append({"op": "speed", "track": tgt, "speed_mps": 8.0, "start_frame": 0, "end_frame": num_frames})
    if any(k in text for k in KEY_TURN) and tgt is not None:
        ops.append({"op": "turn", "track": tgt, "yaw_deg": 25.0, "start_frame": 0, "duration": 10})
    if any(k in text for k in KEY_REMOVE) and tgt is not None:
        ops.append({"op": "remove", "track": tgt})
    return ops


def fill_insert_frames(ops: List[Dict[str, Any]], instruction: str = "",
                       num_frames: int = 30, fps: float = 10.0,
                       max_frames: int = 600, base_frame: int = 0) -> List[Dict[str, Any]]:
    """统一把"生成多少帧"落到 `insert.num_frames`，并把 collide 的帧号夹进该范围。

    LLM 经常忘把指令里的"持续 40 帧 / 4 秒"写进 op（实测就有）；不补的话生成物体只会
    跑默认的 20 帧，用户明确要 40 帧却"变短了"。优先级：
    `op 里已有的` > `指令里解析出来的（40帧/4秒/持续40）` > `调用参数 num_frames`。
    """
    parsed = None
    try:
        _p = parse_num_frames(instruction, fps)
        parsed = int(_p[0]) if _p and _p[0] else None
    except Exception:  # noqa: BLE001
        parsed = None
    try:
        cap = int(os.environ.get("DGGT_MAX_INSERT_FRAMES", str(max_frames)))
    except Exception:  # noqa: BLE001
        cap = int(max_frames)
    cap = max(2, min(int(max_frames), cap))
    ins_last = None
    ins_start = int(base_frame or 0)
    ins_nf = 0
    for o in (ops or []):
        if not isinstance(o, dict) or str(o.get("op")) != "insert":
            continue
        try:
            nf = int(o.get("num_frames"))
        except (TypeError, ValueError):
            nf = 0
        if nf <= 0:
            nf = int(parsed or num_frames or 20)
        nf = max(2, min(cap, int(nf)))
        o["num_frames"] = nf
        try:
            start = int(o.get("start_frame") or base_frame or 0)
        except (TypeError, ValueError):
            start = int(base_frame or 0)
        ins_last = start + nf - 1
        ins_start, ins_nf = start, nf
    # 相撞帧号夹到生成物体的帧范围里（"第 40 帧相撞"但只生成 40 帧 0..39 时会越界）；
    # 并且**最晚只放到窗口的 75%**：40 帧的事故要是撞在第 40 帧，就完全没有"撞后"阶段了。
    if ins_last is not None:
        for o in (ops or []):
            if not isinstance(o, dict) or str(o.get("op")) != "collide":
                continue
            try:
                fr = int(o.get("frame"))
            except (TypeError, ValueError):
                continue
            if ins_nf >= 3:
                fr = min(fr, ins_start + max(1, int(round(0.75 * (ins_nf - 1)))))
            if fr > ins_last:
                fr = int(ins_last)
            o["frame"] = int(fr)
    return ops


def plan(tm, instruction: str, frame_idx: int = 0, num_frames: int = 30,
         new_role: str = NEW_ROLE_AUTO, fps: float = 10.0,
         max_frames: int = 600) -> Dict[str, Any]:
    summary = scene_summary(tm, frame_idx)
    txt = _llm_text(PLAN_PROMPT.format(scene=json.dumps(summary, ensure_ascii=False),
                                       instruction=instruction, frame=frame_idx,
                                       num_frames_minus1=max(0, num_frames - 1),
                                       fps=int(round(float(fps or 10.0))),
                                       max_frames=int(max_frames)))
    data = nl_entity._parse_json(txt) if txt else None
    if data and isinstance(data.get("ops"), list) and data["ops"]:
        warn: List[str] = []
        ops = sanitize_ops(tm, data["ops"], instruction, warnings=warn, new_role=new_role)
        if ops:
            ops = fill_insert_frames(ops, instruction, num_frames, fps=fps,
                                     max_frames=max_frames, base_frame=frame_idx)
            return {"source": "llm", "ops": ops, "raw": txt[:1200], "warnings": warn}
    warn = []
    ops = sanitize_ops(tm, rule_plan(instruction, summary, num_frames, fps=fps), instruction,
                       warnings=warn, new_role=new_role)
    ops = fill_insert_frames(ops, instruction, num_frames, fps=fps,
                             max_frames=max_frames, base_frame=frame_idx)
    if not txt:
        warn.insert(0, "LLM 解析不可用（微服务未启动/无响应），已退化为关键词规则解析，"
                       "结果可能不准；建议先把 text2entity 微服务起起来。")
    return {"source": "rule", "ops": ops, "raw": txt[:600], "warnings": warn}


# ==================== 执行 ====================

def _smoothstep(t: float) -> float:
    t = max(0.0, min(1.0, float(t)))
    return t * t * (3.0 - 2.0 * t)


def _path_frames(tm, tid: int):
    frames = sorted(tm.get_track_frames(int(tid)))
    poses = {f: np.asarray(tm.get_track_pose(int(tid), f), dtype=np.float32) for f in frames}
    return frames, poses


def _track_speed(tm, tid: int, fps: float = 10.0) -> float:
    frames, poses = _path_frames(tm, tid)
    if len(frames) < 2:
        return 0.0
    p0 = poses[frames[0]][:3, 3]
    p1 = poses[frames[-1]][:3, 3]
    dt = (frames[-1] - frames[0]) / max(1e-6, fps)
    return float(np.linalg.norm(p1[[0, 2]] - p0[[0, 2]]) / max(1e-6, dt))


_TID_RE = re.compile(r"T\s*[:：]?\s*(\d+)|(?:编号|id|ID)\s*[:：]?\s*(\d+)")
_KEY_CRASH_RE = re.compile("|".join(KEY_CRASH))
_EGO_RE = re.compile(r"主车|自车|本车|自动驾驶车|ego", re.IGNORECASE)


def _mentioned_with_pos(instruction: str) -> List[tuple]:
    """按出现顺序提取指令里显式提到的 track id 及其首次出现位置。"""
    out: List[tuple] = []
    seen = set()
    for m in _TID_RE.finditer(str(instruction or "")):
        v = m.group(1) or m.group(2)
        if v is None:
            continue
        i = int(v)
        if i not in seen:
            seen.add(i)
            out.append((i, m.start()))
    return out


def instruction_track_ids(instruction: str) -> List[int]:
    """按出现顺序提取指令里显式提到的 track id（兼容 `T:2` / `T2` / `编号2`）。"""
    return [i for i, _ in _mentioned_with_pos(instruction)]


def sanitize_ops(tm, ops: List[Dict[str, Any]], instruction: str = "",
                 warnings: Optional[List[str]] = None,
                 new_role: str = NEW_ROLE_AUTO) -> List[Dict[str, Any]]:
    """对 LLM 输出的 ops 做确定性纠错，避免误解析把场景改坏。

    LLM 有时会自作主张或指错车：明明只是"让 A 撞 B"，它却顺手 `insert` 一辆新车、
    把被撞的车 `remove` 掉、或者把 a/b 搞反。这里用指令里用户**显式写的 T:id** 兜底修正：
      1. 指令里没有"生成/添加/新增/放入"等词时，丢掉 `insert`；
      2. 指令里显式提到的 id 优先：collide 的 a/b 按"先提到的撞后提到的"重排；
         若整条指令的 op 都没指向被提到的车，则把它们挂到第一辆被提到的车上；
      3. 指令说"撞"但没有 collide，就补一条（用前两个被提到的 id）；
      4. 同时被其它 op 引用的车，其 `remove` 视为 LLM 幻觉，丢掉；`remove` 统一挪到最后；
      5. 丢掉引用"不存在 / 本轮已删"车辆的 op；collide 的 a、b 不能是同一辆车。
    """
    text = str(instruction or "")
    ops = [dict(o) for o in (ops or []) if isinstance(o, dict)]
    mentioned = instruction_track_ids(text)

    def _known(tid) -> bool:
        try:
            return tid is not None and len(tm.get_track_frames(int(tid))) > 0
        except Exception:  # noqa: BLE001
            return False

    # 1. 指令没让生成新车，就不要 insert
    if not any(k in text for k in KEY_INSERT):
        ops = [o for o in ops if str(o.get("op")) != "insert"]

    # 1.5 主车不能被动"顺手"删掉：LLM 偶尔会给出 remove 900000。一旦执行，
    #     主车没了、相机也没了，整个场景就没法看了。只有指令**明确**提到
    #     "主车/自车/本车/ego" 时才允许删它。
    if not _EGO_RE.search(text):
        kept = []
        for o in ops:
            if str(o.get("op")) == "remove":
                try:
                    t = int(o.get("track"))
                except (TypeError, ValueError):
                    t = None
                if t == EGO_TRACK_ID:
                    if warnings is not None:
                        warnings.append("模型想删除主车（T:900000），但指令里没有提到主车，已忽略该删除。")
                    continue
            kept.append(o)
        ops = kept

    # 2. 显式 id / "主车" 优先：按在句子里出现的先后定角色（先=肇事、后=被撞）
    pos_pairs = list(_mentioned_with_pos(text))
    ego_m = _EGO_RE.search(text)
    if ego_m:
        pos_pairs.append((EGO_TRACK_ID, ego_m.start()))
    pos_pairs.sort(key=lambda t: t[1])
    cands = [i for i, _ in pos_pairs if _known(i)]
    has_insert = any(str(o.get("op")) == "insert" for o in ops)

    # 2.0 安全兜底：指令要"生成一辆车"、却没点名任何**存在**的车（例如
    #     "生成一辆对向来的红色轿车在 T100000 转弯时相撞"，T100000 并不存在），
    #     规则兜底会把变道/相撞/转向挂到随便一辆车上——那会改坏别人的轨迹。
    #     这种"无目标"的猜测一律丢掉，只保留 insert，并回一条 warning。
    if has_insert and not cands:
        dropped = []
        keep = []
        for o in ops:
            if str(o.get("op")) in ("speed", "lane_change", "turn", "collide"):
                dropped.append(str(o.get("op")))
                continue
            keep.append(o)
        if dropped:
            ops = keep
            if warnings is not None:
                if mentioned:
                    warnings.append(
                        "指令里提到的车辆 id（%s）在场景里不存在，无法确定要撞/要变道的是谁；"
                        "已只保留“生成车辆”，没有改动任何已有车辆。请在指令里写上真实存在的 T:id。"
                        % ", ".join("T:%d" % i for i in mentioned))
                else:
                    warnings.append(
                        "指令里没有指名任何已有车辆，规则兜底推测出来的变道/相撞/转向已丢弃，"
                        "只保留“生成车辆”，避免误改别的车。需要事故就写上 T:id。")
    if cands:
        crash_m = _KEY_CRASH_RE.search(text)
        # "有新车 + 要撞车"时**跳过**这一步猜测：这一步会把没提到的车统统改挂到
        # 第一辆被提到的车上（而那辆往往是被撞方），结果变道动作用到了被撞车上。
        # 正确的归属由下面 insert+crash 分支统一处理（这些动作属于新车）。
        if not (has_insert and crash_m):
            for o in ops:
                if str(o.get("op")) == "collide":
                    continue
                t = o.get("track")
                if t is not None and t not in cands:
                    o["track"] = cands[0]
        collide_ops = [o for o in ops if str(o.get("op")) == "collide"]
        if crash_m and has_insert:
            # 有新车参与的事故：**默认新车是肇事车**——它才是我们能控制的那辆；
            # 指令里写出来的已有车是被撞方。这样"…T1转弯时和该红色轿车相撞"
            # 也只会让"新车去撞 T1"，而不会变成"T1 撞别人"。
            # 只有指令明确写成"已有车去撞新车"（T:1 撞这辆/该车…）才反过来。
            role, why = decide_new_role(text, bool(cands), new_role)
            new_is_attacker = (role == NEW_ROLE_ATTACKER)
            op_role = "attacker" if new_is_attacker else "victim"
            if cands:
                for o in collide_ops:
                    if new_is_attacker:
                        o["a"], o["b"] = None, cands[0]     # 新车撞那辆已有车
                    else:
                        o["a"], o["b"] = cands[0], None     # 那辆已有车撞新车
                    o["new_object_role"] = op_role
                if not collide_ops:
                    ops.append({"op": "collide",
                                "a": None if new_is_attacker else cands[0],
                                "b": cands[0] if new_is_attacker else None,
                                "new_object_role": op_role, "frame": 20})
                if warnings is not None and why == "ambiguous":
                    warnings.append(
                        "指令里『…和…相撞』是对称说法，没有指明谁撞谁；已默认让**新建的车**去撞 T:%d。"
                        "想反过来就把指令写成『让 T:%d 撞这辆新车』，或用面板上的「新建车角色」选『被撞车』。"
                        % (cands[0], cands[0]))
            else:
                for o in collide_ops:
                    o["a"], o["b"] = (None, None) if new_is_attacker else (None, None)
                    o["new_object_role"] = op_role
            # 规则兜底把变道/加速/转向挂到了"指令里没提过的车"上 → 改挂到新车
            mentioned_set = set(mentioned)
            for o in ops:
                if str(o.get("op")) in ("speed", "lane_change", "turn"):
                    t = o.get("track")
                    if t is not None and t not in mentioned_set:
                        o["track"] = None
        elif crash_m and len(cands) >= 2:
            for o in collide_ops:
                o["a"], o["b"] = cands[0], cands[1]
            if not collide_ops:
                ops.append({"op": "collide", "a": cands[0], "b": cands[1], "frame": 20})
        elif crash_m and len(cands) == 1:
            tid = cands[0]
            if pos_pairs[0][1] < crash_m.start():
                # 只提到肇事车 → 被撞的是新车（或交给 LLM）
                for o in collide_ops:
                    o["a"] = tid
            else:
                # 只提到被撞车："生成一辆车…撞上 T:x" → 肇事车就是新车
                for o in collide_ops:
                    o["b"] = tid
                if has_insert:
                    for o in collide_ops:
                        o["a"] = None          # 肇事车就是新生成的车
                    for o in ops:
                        if str(o.get("op")) in ("speed", "lane_change", "turn") and o.get("track") == tid:
                            o["track"] = None      # 这些动作其实是新车做的，由上层回填
                    if not collide_ops:
                        ops.append({"op": "collide", "a": None, "b": tid, "frame": 20})
        # 有 insert 时，collide 里"查无此车"的 a/b 一定是 LLM 给新车瞎编的 id → 改成占位
        if has_insert:
            for o in ops:
                if str(o.get("op")) != "collide":
                    continue
                for k in ("a", "b"):
                    if o.get(k) is not None and not _known(o.get(k)):
                        o[k] = None

    # 3. 被别的 op 引用的车不允许 remove（多为 LLM 幻觉），remove 一律排到最后
    referenced = set()
    for o in ops:
        if str(o.get("op")) == "remove":
            continue
        for k in ("track", "a", "b"):
            v = o.get(k)
            if v is not None:
                try:
                    referenced.add(int(v))
                except Exception:  # noqa: BLE001
                    pass
    removes = [o for o in ops if str(o.get("op")) == "remove"
               and o.get("track") is not None and int(o["track"]) not in referenced]
    rest = [o for o in ops if str(o.get("op")) != "remove"]
    ops = rest + removes
    removed = {int(o["track"]) for o in removes}

    cleaned: List[Dict[str, Any]] = []
    for o in ops:
        kind = str(o.get("op"))
        if kind == "insert":
            # LLM 会把提示词里的 "ahead|oncoming|roadside" 原样抄下来（实测）→ 归一化
            mo = str(o.get("mode") or "").strip().lower()
            if mo not in ("ahead", "oncoming", "roadside"):
                if mo and mo not in ("", "none", "null"):
                    if warnings is not None:
                        warnings.append(
                            f"insert 的 mode『{o.get('mode')}』不是合法取值，已按 ahead（同向前方）处理；"
                            "合法值只有 ahead / oncoming / roadside。")
                o["mode"] = "ahead"
            cleaned.append(o)
            continue
        if kind == "remove":
            tid = o.get("track")
            # 已在上一步按"是否被别的 op 引用"筛过；这里不能再查 removed（它自己就在里面）
            if tid is not None and _known(tid) and int(tid) not in referenced:
                cleaned.append(o)
            continue
        if kind == "collide":
            a, b = o.get("a"), o.get("b")
            if a is None and b is None:
                continue           # 双方都没确定（例如指令里没写被撞的车）→ 无意义
            # a/b 允许为 None（代表"刚生成的新车"，由上层回填）
            if a is not None and (a in removed or not _known(a)):
                continue
            if b is not None and (b in removed or not _known(b)):
                continue
            if a is not None and b is not None and int(a) == int(b):
                continue
            cleaned.append(o)
            continue
        tid = o.get("track")
        if tid is None:
            # 指向"新车"的占位（None）：等上层 insert 拿到 id 再回填
            if kind in ("speed", "lane_change", "turn") and \
                    any(str(x.get("op")) == "insert" for x in ops):
                cleaned.append(o)
            continue
        if tid in removed or not _known(tid):
            continue
        cleaned.append(o)
    return cleaned


def fuse_collide_ops(ops: List[Dict[str, Any]], num_frames: int = 30) -> Dict[int, Dict[str, Any]]:
    """把同一条指令里"撞车"相关的 op 融合成一次事故仿真。

    一条指令常常同时给出 `lane_change` + `collide`（"向左变道后撞上 T:x"）。如果按顺序
    各做一次，**追击仿真会把之前写好的横移整段覆盖掉**（corner case 里同样的坑，
    见 scenario_engine.plan_lane_change 的注释）。所以这里把横移量作为仿真的
    `attacker_lateral` / `victim_lateral` 传进去，一次仿真同时产生"变道 + 撞击 + 撞后"，
    并把被融合掉的 op 标记为 `_fused` 跳过执行，保证两条路径先后编辑也不会互相打架。

    Returns: {attacker_track_id: {"attacker_lateral":..,"victim_lateral":..,"lateral_ramp":(a,b),"consumed":[op,...]}}
    """
    out: Dict[int, Dict[str, Any]] = {}
    for op in ops or []:
        if str(op.get("op")) != "collide":
            continue
        a = op.get("a")
        if a is None:
            continue
        a = int(a)
        b = op.get("b")
        b = int(b) if b is not None else None
        n = max(2, int(num_frames or 30))
        fused = {"attacker_lateral": 0.0, "victim_lateral": 0.0,
                 "lateral_ramp": (0.0, 1.0), "consumed": []}
        # 横移（变道）→ 作为仿真内部叠加的横向偏移
        for o in ops:
            if str(o.get("op")) != "lane_change":
                continue
            t = o.get("track")
            if t is None:
                continue
            try:
                t = int(t)
            except Exception:  # noqa: BLE001
                continue
            if t not in (a, b):
                continue
            lat = float(o.get("lateral") or 0.0)
            f0 = int(o.get("start_frame") or 0)
            dur = max(1, int(o.get("duration") or 0))
            lo = max(0.0, min(1.0, f0 / float(n - 1)))
            hi = max(0.0, min(1.0, (f0 + dur) / float(n - 1)))
            if hi <= lo:
                hi = min(1.0, lo + 0.2)
            if t == a:
                fused["attacker_lateral"] += lat
            else:
                fused["victim_lateral"] += lat
            fused["lateral_ramp"] = (min(fused["lateral_ramp"][0], lo), max(fused["lateral_ramp"][1], hi))
            fused["consumed"].append(o)
        # 肇事车的 speed op：事故仿真的追击速度由它自己算，避免"先定时速再被覆盖"
        for o in ops:
            if str(o.get("op")) != "speed":
                continue
            try:
                t = int(o.get("track")) if o.get("track") is not None else None
            except Exception:  # noqa: BLE001
                t = None
            if t == a:
                fused["consumed"].append(o)
        out[a] = fused
    for o in out.values():
        for c in o["consumed"]:
            c["_fused"] = True
    return out


def apply_ops(tm, ops: List[Dict[str, Any]], fps: float = 10.0,
              num_frames: int = 30) -> List[Dict[str, Any]]:
    """执行除 insert 以外的操作；返回每步的执行结果。

    collide 与它的 lane_change/speed 会先被 `fuse_collide_ops` 融合成一次事故仿真，
    被融合的 op 只在回执里留一条 `fused` 记录，不再单独执行。
    """
    ops = [dict(o) for o in (ops or []) if isinstance(o, dict)]
    fused_map = fuse_collide_ops(ops, num_frames=num_frames)
    report = []
    for op in ops:
        kind = str(op.get("op") or "")
        try:
            if kind == "collide":
                a_key = op.get("a")
                fused = fused_map.get(int(a_key)) if a_key is not None else None
                res = _op_collide(tm, op, fused=fused)
                for c in (fused or {}).get("consumed", []):
                    res.setdefault("fused_ops", []).append(
                        {"op": c.get("op"), "track": c.get("track")})
                report.append(res)
                continue
            if op.get("_fused"):
                report.append({"op": kind, "track": op.get("track"), "ok": True,
                               "note": "已融合进 collide 的同一次仿真，不再单独执行"})
                continue
            if kind == "speed":
                report.append(_op_speed(tm, op, fps))
            elif kind == "lane_change":
                report.append(_op_lane_change(tm, op))
            elif kind == "turn":
                report.append(_op_turn(tm, op))
            elif kind == "remove":
                tid = int(op["track"])
                tm.delete_track(tid)
                report.append({"op": kind, "track": tid, "ok": True})
            elif kind == "insert":
                report.append({"op": kind, "ok": True, "note": "由上层生成车辆处理"})
            else:
                report.append({"op": kind, "ok": False, "error": "未知操作"})
        except Exception as e:  # noqa: BLE001
            report.append({"op": kind, "ok": False, "error": str(e)})
    return report


def _op_speed(tm, op, fps):
    tid = int(op["track"])
    target = float(op.get("speed_mps") or op.get("speed") or 0.0)
    f0 = int(op.get("start_frame") or 0)
    frames, poses = _path_frames(tm, tid)
    if not frames:
        return {"op": "speed", "ok": False, "error": "没有轨迹"}
    old = _track_speed(tm, tid, fps)
    if old < 0.3:
        return {"op": "speed", "ok": False, "error": f"原速度过小({old:.2f}m/s)"}
    k = max(0.2, min(5.0, target / old))
    # 以 f0 为锚点，把"沿路走过的距离"按 k 缩放
    anchor_f = f0 if f0 in poses else min(frames, key=lambda f: abs(f - f0))
    A = poses[anchor_f][:3, 3].copy()
    for f in frames:
        if f == anchor_f:
            P = poses[f].copy()
        else:
            P = poses[f].copy()
            P[:3, 3] = A + (poses[f][:3, 3] - A) * k
        tm.set_track_pose(tid, f, P)
    return {"op": "speed", "track": tid, "ok": True,
            "old_mps": round(old, 2), "target_mps": round(target, 2), "k": round(k, 3)}


def _op_lane_change(tm, op):
    tid = int(op["track"])
    lat = float(op.get("lateral") or 0.0)
    f0 = int(op.get("start_frame") or 0)
    dur = max(1, int(op.get("duration") or 15))
    frames, poses = _path_frames(tm, tid)
    if not frames:
        return {"op": "lane_change", "ok": False, "error": "没有轨迹"}
    for f in frames:
        a = _smoothstep((f - f0) / float(dur))
        if a <= 0:
            continue
        P = poses[f].copy()
        right = P[:3, 0]
        P[:3, 3] = P[:3, 3] + right * (lat * a)
        tm.set_track_pose(tid, f, P)
    return {"op": "lane_change", "track": tid, "ok": True,
            "lateral": lat, "start_frame": f0, "duration": dur}


def _op_turn(tm, op):
    tid = int(op["track"])
    yaw = math.radians(float(op.get("yaw_deg") or 0.0))
    f0 = int(op.get("start_frame") or 0)
    dur = max(1, int(op.get("duration") or 10))
    frames, poses = _path_frames(tm, tid)
    if not frames:
        return {"op": "turn", "ok": False, "error": "没有轨迹"}

    def Ry(a):
        c, s = math.cos(a), math.sin(a)
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)

    for f in frames:
        a = _smoothstep((f - f0) / float(dur)) * yaw
        if abs(a) < 1e-6:
            continue
        P = poses[f].copy()
        P[:3, :3] = Ry(a) @ P[:3, :3]
        tm.set_track_pose(tid, f, P)
    return {"op": "turn", "track": tid, "ok": True,
            "yaw_deg": round(math.degrees(yaw), 1), "start_frame": f0, "duration": dur}


def _yaw_of(P) -> float:
    """从世界系姿态矩阵取航向角（forward = 第 2 列）。"""
    f = P[:3, 2]
    return math.atan2(float(f[0]), float(f[2]))


def _wrap_pi(a: float) -> float:
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a


def _horizontal_normal_from(pa, pv, axis=None, fallback_center=(0.0, 0.0, 1.0)):
    """取一条水平的分离方向：优先用 SAT 最小平移轴的水平投影，退化时用两心水平连线。"""
    if axis is not None:
        n = np.array([axis[0], 0.0, axis[2]], dtype=np.float64)
        ln = float(np.linalg.norm(n))
        if ln >= 0.05:
            return n / ln
    d = np.asarray(pv, dtype=np.float64)[:3, 3] - np.asarray(pa, dtype=np.float64)[:3, 3]
    d[1] = 0.0
    ln = float(np.linalg.norm(d))
    if ln < 1e-6:
        n = np.asarray(fallback_center, dtype=np.float64)
        n[1] = 0.0
        return n / max(1e-6, float(np.linalg.norm(n)))
    return d / ln


def _ground_center_y(tm, x, z, height, fallback=None, radius=None, strict=False):
    """(x,z) 处的路面高度 → 车体中心 y（`up = -y` 或 `+y` 由 TrackManager 给出）。

    `strict=True`：只在邻域静态点足够密（= 真的覆盖到这里）时才贴地；覆盖外返回
    `fallback`（保持原高度）——因为那种地方"逐级放大邻域"估出来的高度可能是几十米外的
    路面，硬贴会把车抬到空中（实测 z≈99 处估出 1.4m 的偏差）。
    静态点覆盖不到的远端返回 `fallback`（通常是原来的 y），**绝不**用包围盒底面反推
    地面（那正是"浮空"的来源）。
    """
    gy = None
    if strict:
        gy = _dense_ground_y(tm, x, z, radius=radius)
    else:
        try:
            gy = tm.ground_y_at(float(x), float(z), radius=radius)
        except Exception:  # noqa: BLE001
            gy = None
    if gy is None:
        return fallback
    try:
        up = float(tm.world_up_sign())
    except Exception:  # noqa: BLE001
        up = -1.0
    return float(gy) + up * (float(height) / 2.0)


def _dense_ground_y(tm, x, z, radius=None, min_pts=30, max_radius=12.0):
    """只在 (x,z) 邻域"够密"时才给路面高度，否则 None（= 场景覆盖外，宁可不贴）。

    与 `ground_y_at` 的区别：不走"逐级放大到几十米"那套（那套用于**规划**时给出一个
    可用的地面估计），而是要求 6~12m 邻域里就有 ≥`min_pts` 个静态点 —— 否则这个估计
    很可能是几十米外的路面，贴上去反而浮空。
    """
    try:
        pts = getattr(tm, "_ground_points", None)
        if pts is None:
            from track_manager import _scene_points_cpu, EGO_GROUND_MAX_POINTS
            pts = _scene_points_cpu(tm.renderer, EGO_GROUND_MAX_POINTS)
            tm._ground_points = pts
        if pts is None or len(pts) == 0:
            return None
        from track_manager import _ground_height_near
        r = float(radius or 6.0)
        dx = pts[:, 0] - float(x)
        dz = pts[:, 2] - float(z)
        max_r = max(r, float(max_radius))
        m = (dx * dx + dz * dz) <= max_r * max_r
        if int(m.sum()) < int(min_pts):
            return None
        return _ground_height_near(pts, (float(x), 0.0, float(z)), r,
                                   tm.world_up_sign(), max_tries=1)
    except Exception:  # noqa: BLE001
        return None


def _reground_track(tm, tid, frames=None, dims=None):
    """把某条轨迹逐帧"贴回路面"：只改 y，保留 x/z 与朝向。

    事故仿真/兜底摆放沿用了旧的"锁定首帧高度"写法，在路面起伏或（更常见的）轨迹跑到
    静态点覆盖边缘时会整体偏一个车高，肉眼就是"车浮在空中"。这里在所有改动之后统一
    兜底：只要该帧（x,z）能估出路面，就把中心放到"路面 + up×半车高"。
    返回修好的帧数。
    """
    if dims is None:
        try:
            dims = tm.get_track_dimensions(int(tid))
        except Exception:  # noqa: BLE001
            dims = None
    if not dims or len(dims) < 2:
        return 0
    h = float(dims[1])
    if h <= 1e-3:
        return 0
    if frames is None:
        try:
            frames = sorted(tm.get_track_frames(int(tid)))
        except Exception:  # noqa: BLE001
            return 0
    fixed = 0
    for f in frames:
        p = tm.get_track_pose(int(tid), int(f))
        if p is None:
            continue
        p = np.asarray(p, dtype=np.float32).copy()
        x, z, y = float(p[0, 3]), float(p[2, 3]), float(p[1, 3])
        # strict：场景覆盖外不改（保持原高度），避免按"几十米外的路面"把车抬到空中
        ny = _ground_center_y(tm, x, z, h, fallback=None, strict=True)
        if ny is None:
            continue
        if abs(ny - y) < 1e-4:
            continue
        p[1, 3] = float(ny)
        try:
            tm.set_track_pose(int(tid), int(f), p)
            fixed += 1
        except Exception:  # noqa: BLE001
            break
    return fixed


def _smooth_track_heights(tm, tid, frames=None, window=5):
    """把（合成）物体逐帧中心 y 做滑动平均：只动高度，不动 x/z/朝向。

    逐帧路面估计本身有 ±5cm 的抖动（邻域点数变化会让分位数跳一下），直接贴地会让车
    在路上"点头"。这里在贴地之后把高度序列抹平，视觉上更贴地、更稳。
    """
    from track_manager import _smooth_series
    if frames is None:
        try:
            frames = sorted(tm.get_track_frames(int(tid)))
        except Exception:  # noqa: BLE001
            return 0
    ys = {}
    for f in frames:
        p = tm.get_track_pose(int(tid), int(f))
        if p is None:
            continue
        ys[int(f)] = float(np.asarray(p, dtype=np.float32)[1, 3])
    if len(ys) < 3:
        return 0
    keys = sorted(ys)
    sm = _smooth_series([ys[k] for k in keys], window=int(window))
    fixed = 0
    for k, v in zip(keys, sm):
        p = tm.get_track_pose(int(tid), int(k))
        if p is None:
            continue
        P = np.asarray(p, dtype=np.float32).copy()
        if abs(float(P[1, 3]) - float(v)) < 1e-4:
            continue
        P[1, 3] = float(v)
        try:
            tm.set_track_pose(int(tid), int(k), P)
            fixed += 1
        except Exception:  # noqa: BLE001
            break
    return fixed


def _contact_place(tm, a, b, frames, frame, dims_a, dims_v, contact_margin=0.12):
    """兜底：把 a 平滑推到"与 b 刚好接触"（第 `frame` 帧接触），之后贴着推、直到窗口结束。

    仅当追击仿真在该窗口内没能自然撞上时使用（例如受害车在窗口内跑掉了）。
    三条硬要求：
    1) **绝不重叠**：用 OBB 支撑距离算接触间距，最后还会再跑一次 `separate_pair`；
    2) **不改写受害车 b 的轨迹**：这里只移动 a（转弯的车继续按它自己的轨迹转弯）；
    3) 有"撞前 → 撞上 → 撞后"：`frame` 之前是平滑接近，`frame` 时刚好接触，
       `frame` 之后把间距收到 2cm（看起来是"顶上去"而不是隔空并排），随 b 一起走。
    """
    import corner_case
    n = _horizontal_normal_from(
        tm.get_track_pose(a, frames[0]), tm.get_track_pose(b, frame))
    a0 = frames[0]
    A0 = np.asarray(tm.get_track_pose(a, a0), dtype=np.float64)[:3, 3].copy()
    fixed = 0
    for f in frames:
        pa = tm.get_track_pose(a, f)
        pb = tm.get_track_pose(b, f)
        if pa is None or pb is None:
            continue
        pa = np.asarray(pa, dtype=np.float32).copy()
        pb = np.asarray(pb, dtype=np.float32).copy()
        if f == a0:
            tm.set_track_pose(a, f, pa)
            continue
        t = _smoothstep((f - a0) / float(max(1, frame - a0)))
        # 目标 = 受害车中心 - 支撑距离 - 余量（沿 n 方向刚好贴住）
        margin = float(contact_margin) if f < frame else 0.02
        need = (corner_case._obb_radius_along(pb, dims_v, n)
                + corner_case._obb_radius_along(pa, dims_a, n) + float(margin))
        target = np.asarray(pb[:3, 3], dtype=np.float64) - n * need
        center = A0 * (1.0 - t) + target * t
        # 高度：逐帧贴地（旧实现写死 A0[1]，轨迹跑到静态点覆盖边缘时会整体浮空）；
        # strict：覆盖外就沿用原高度，而不是按远邻域估计把车抬起来
        ny = _ground_center_y(tm, float(center[0]), float(center[2]), float(dims_a[1]),
                              fallback=float(A0[1]), strict=True)
        center[1] = float(A0[1]) if ny is None else float(ny)
        pa[:3, 3] = center.astype(np.float32)
        # 朝向对准目标（只改水平航向）
        d = target - center
        if float(np.linalg.norm(d[[0, 2]])) > 0.15:
            yaw = math.atan2(float(d[0]), float(d[2]))
            c, s = math.cos(yaw), math.sin(yaw)
            pa[:3, 0] = (c, 0.0, -s)
            pa[:3, 1] = (0.0, 1.0, 0.0)
            pa[:3, 2] = (s, 0.0, c)
        tm.set_track_pose(a, f, pa)
        fixed += 1
    return fixed


def func_len(x):
    try:
        return len(x)
    except TypeError:
        return 0


def _accident_snaps(tm):
    """Per-scene 事故重放快照（用于让重复应用同一条 collide 是幂等的）。"""
    snaps = getattr(tm, "_accident_snaps", None)
    if snaps is None:
        snaps = {}
        try:
            setattr(tm, "_accident_snaps", snaps)
        except Exception:  # noqa: BLE001
            pass
    return snaps


def _dump_poses(tm, tid, frames):
    out = {}
    for f in frames:
        p = tm.get_track_pose(tid, f)
        if p is not None:
            out[int(f)] = np.asarray(p, dtype=np.float32).copy()
    return out


def _max_center_delta(A, B):
    """两组逐帧位姿的**最大中心差**（用于诊断/判定"有没有被动过"）。"""
    if not A or set(A.keys()) != set(B.keys()):
        return float("inf")
    m = 0.0
    for f in A:
        m = max(m, float(np.abs(np.asarray(A[f], np.float32)[:3, 3]
                                - np.asarray(B[f], np.float32)[:3, 3]).max()))
    return m


def _poses_close(A, B, tol=2e-3):
    """比较两组逐帧位姿是否"没被动过"。

    只看**平移**（中心）：`get_track_pose` 返回的朝向里含有"车头自动朝向运动方向"的结果，
    而事故仿真会改变运动方向，于是朝向会跟着微调——拿整个 4x4 比会因为这点朝向差
    判定成"没动过=否"，白白错过幂等重放。位置才是我们要判定的东西。
    """
    if set(A.keys()) != set(B.keys()) or not A:
        return False
    for f in A:
        if float(np.abs(A[f][:3, 3] - B[f][:3, 3]).max()) > tol:
            return False
    return True


def _op_collide(tm, op, fused=None):
    """让 a 撞上 b —— **直接复用 corner case 的同一套事故仿真**。

    与"生成 corner case"共享 `corner_case.simulate_pair_collision`：
    限加速度追击 → OBB-SAT 接触判定（用各车**当前**包围盒尺寸）→ 非弹性动量冲量
    （受害车被撞开、肇事车减速）→ 碰撞后摩擦减速 → **逐帧防穿模分离**。
    因此两条路径的碰撞判定与撞后表现完全一致，先后用两种功能编辑同一段轨迹不会
    出现"一个说撞了、一个说没撞"或互相覆盖的冲突。

    `fused`（由 `fuse_collide_ops` 给出）会把同一条指令里的变道横移量作为仿真内部的
    `attacker_lateral`/`victim_lateral` 传进去，实现"变道 + 撞"一次成型。
    """
    import corner_case

    a = int(op.get("a"))
    b = int(op.get("b"))
    fps = float(op.get("fps") or 10.0)
    want_frame = op.get("frame") or op.get("collision_frame")
    # 事故窗口要包含"发生前 → 发生时 → 发生后"三个阶段：撞击帧最晚只能放在窗口的 ~75%，
    # 否则撞在最后一帧就看不到撞后（追尾/被撞开/滑停）。用户/LLM 没给期望帧时按 60% 放。
    want_impact = op.get("impact_frame") or want_frame
    fused = fused or {}
    fa = sorted(tm.get_track_frames(a))
    fb = sorted(tm.get_track_frames(b))
    if not fa or not fb:
        return {"op": "collide", "ok": False, "error": "缺少轨迹"}

    # 仿真窗口必须是**连续帧**：追击动力学按固定 dt=1/fps 逐步推进，如果喂进去的
    # frames 是"交集"这种稀疏列表（真实轨迹常在若干帧没有位姿，例如 T:1 只有
    # 0,3,11..19,25..39），仿真就会把"连续的物理时间"写到"跳跃的帧号"上，而没被写到的
    # 帧仍保留旧位姿 —— 于是同一条轨迹出现几段互不相接、来回跳的残影
    # （用户看到的就是"位置莫名其妙""撞的不是该撞的那辆"）。
    lo_f, hi_f = int(max(fa[0], fb[0])), int(min(fa[-1], fb[-1]))
    frames = list(range(lo_f, hi_f + 1))
    if len(frames) < 3:
        frames = sorted(set(fa) | set(fb))
    if len(frames) < 3:
        return {"op": "collide", "ok": False,
                "error": f"可用帧太少（T:{a}×T:{b} 共 {len(frames)} 帧），无法做事故仿真"}

    dims_a = list(tm.get_track_dimensions(a))
    dims_v = list(tm.get_track_dimensions(b))

    # 受害车轨迹"可信度"预检：稀疏/误检的轨迹相邻帧会跳十几米（实测 T:13 单帧跳 32m）。
    # 拿这样的轨迹当撞击目标，追击速度会被推到 45m/s 上限、冲量也会爆掉 —— 生成出来的
    # 是一条"飞出去"的轨迹。这种宁可**如实拒绝**（此时两支轨迹都还没被动过）。
    try:
        vmax = 0.0
        prev_f, prev_c = None, None
        for f in frames:
            p = tm.get_track_pose(b, f)
            if p is None:
                prev_f, prev_c = None, None
                continue
            c = np.asarray(p, dtype=np.float64)[:3, 3]
            if prev_c is not None:
                vmax = max(vmax, float(np.linalg.norm(c - prev_c))
                           / max(1e-6, float(f - prev_f) / max(1e-6, fps)))
            prev_f, prev_c = int(f), c
        vmax_cap = float(os.environ.get("DGGT_MAX_TRACK_SPEED", "60"))
    except Exception:  # noqa: BLE001
        vmax, vmax_cap = 0.0, 60.0
    if vmax_cap > 0 and vmax > vmax_cap:
        msg = (f"T:{b} 在这一段里轨迹不连续（相邻帧最高约 {vmax:.0f} m/s，疑似误检测/稀疏），"
               f"拿它做撞击目标会算出不可信的碰撞；已保留原轨迹、不做碰撞。"
               "请换一辆轨迹连续的车，或先修好它的轨迹。")
        return {"op": "collide", "a": a, "b": b, "ok": False, "error": msg, "warn": msg,
                "victim_max_speed_mps": round(float(vmax), 1),
                "max_track_speed_mps": float(vmax_cap)}

    # 撞击帧：默认放在窗口 60% 处，最晚 75%（留出"撞后"阶段）；夹在窗口内
    n_win = len(frames)
    last_ok = frames[0] + max(2, int(round(0.75 * (n_win - 1))))
    if want_impact is None:
        want_impact = frames[0] + int(round(0.6 * (n_win - 1)))
    try:
        impact_frame = int(want_impact)
    except (TypeError, ValueError):
        impact_frame = frames[0] + int(round(0.6 * (n_win - 1)))
    impact_frame = int(max(frames[0] + 2, min(impact_frame, last_ok)))
    if impact_frame > frames[-1]:
        impact_frame = frames[-1]

    # ---- 幂等：重复应用同一条 collide 不能把车越撞越远 ----
    # 如果两条轨迹自上次事故仿真之后没有被动过（用户没有手动改过），先把它们还原到
    # "事故前"，再重放一次仿真，结果与第一次完全一致。若中间被手动编辑过（或换了 pair），
    # 就直接从当前状态仿真——那才是"用新状态再撞一次"的正确语义。
    snaps = _accident_snaps(tm)
    key = (a, b)
    rec = snaps.get(key)
    replayed = False
    before_a = _dump_poses(tm, a, frames)
    before_b = _dump_poses(tm, b, frames)
    replay_miss = None
    if rec is not None:
        da = _max_center_delta(rec.get("after_a", {}), before_a)
        db = _max_center_delta(rec.get("after_b", {}), before_b)
        if max(da, db) > 2e-3:
            replay_miss = {"max_delta_m": round(float(max(da, db)), 4),
                           "frames_snapshot": len(rec.get("after_a", {})),
                           "frames_now": len(before_a)}
    if rec is not None and _poses_close(rec.get("after_a", {}), before_a) \
            and _poses_close(rec.get("after_b", {}), before_b):
        for f, p in rec.get("before_a", {}).items():
            tm.set_track_pose(a, f, p)
        for f, p in rec.get("before_b", {}).items():
            tm.set_track_pose(b, f, p)
        before_a = _dump_poses(tm, a, frames)
        before_b = _dump_poses(tm, b, frames)
        replayed = True

    try:
        # 撞击帧要可控：先按"期望帧"反推速度跑一次；实测撞击帧偏差 >2 帧时，
        # 用**实测的接近速度**按比例修正肇事车目标速度再跑（最多 3 次）。
        # 只改肇事车的目标速度，不动碰撞判定/防穿模那一套共用逻辑。
        sim = None
        target_ts = None
        best = None          # (absorb_err, target_speed, sim)
        attempts = 0
        for attempt in range(3):
            attempts = attempt + 1
            for f, p in before_a.items():
                tm.set_track_pose(a, f, p)
            for f, p in before_b.items():
                tm.set_track_pose(b, f, p)
            sim = corner_case.simulate_pair_collision(
                tm, a, b, frames, fps=fps, intensity=float(op.get("intensity") or 1.0),
                enable_physics=True,
                attacker_lateral=float(fused.get("attacker_lateral") or 0.0),
                victim_lateral=float(fused.get("victim_lateral") or 0.0),
                lateral_ramp=tuple(fused.get("lateral_ramp") or (0.0, 1.0)),
                # 语言编辑：受害车**沿它自己的轨迹**走（转弯的车继续转弯），冲量只当额外位移；
                # 并按"期望撞击帧"反推追击速度，让事故窗口包含前/中/后三段。
                victim_follow_track=True,
                impact_frame=(None if target_ts is not None else int(impact_frame)),
                attacker_target_speed=target_ts,
            )
            fc = sim.get("collision_frame")
            if not sim.get("collided") or fc is None:
                break
            err = float(fc) - float(impact_frame)
            if best is None or abs(err) < best[0]:
                best = (abs(err), target_ts, sim)
            if abs(err) <= 2.0 or attempt >= 2:
                break
            used = float(sim.get("target_speed_a") or 0.0)
            if used <= 1e-3:
                break
            t_meas = max(0.2, (float(fc) - float(frames[0])) / max(1e-6, float(fps)))
            t_want = max(0.2, (float(impact_frame) - float(frames[0])) / max(1e-6, float(fps)))
            target_ts = float(np.clip(used * (t_meas / t_want), 0.5, 45.0))
        # 迭代可能"改过头"（有一次撞上了、下一次反而撞不上）→ 回放**最接近期望帧且真的撞上**
        # 的那一次，别把好结果丢了。
        if best is not None:
            cur_fc = (sim or {}).get("collision_frame")
            cur_err = abs(float(cur_fc) - float(impact_frame)) if cur_fc is not None else None
            if (cur_err is None or best[0] < cur_err - 1e-6) and best[2] is not sim:
                for f, p in before_a.items():
                    tm.set_track_pose(a, f, p)
                for f, p in before_b.items():
                    tm.set_track_pose(b, f, p)
                sim = corner_case.simulate_pair_collision(
                    tm, a, b, frames, fps=fps, intensity=float(op.get("intensity") or 1.0),
                    enable_physics=True,
                    attacker_lateral=float(fused.get("attacker_lateral") or 0.0),
                    victim_lateral=float(fused.get("victim_lateral") or 0.0),
                    lateral_ramp=tuple(fused.get("lateral_ramp") or (0.0, 1.0)),
                    victim_follow_track=True,
                    impact_frame=None,
                    attacker_target_speed=best[1],
                )
                attempts += 1
        sim["impact_attempts"] = int(attempts)
        # 撞得离期望帧太远（超过窗口的 15%，且至少 3 帧）就**不硬用**仿真结果：
        # 回滚，改走确定性的兜底摆放（它把"刚好接触"精确放在期望帧上，受害车同样一动不动）。
        tol = max(3.0, 0.15 * float(max(1, frames[-1] - frames[0])))
        if best is not None and best[0] > tol and sim.get("collided"):
            for f, p in before_a.items():
                tm.set_track_pose(a, f, p)
            for f, p in before_b.items():
                tm.set_track_pose(b, f, p)
            sim["collided"] = False
            sim["collision_frame"] = None
            sim["skipped_far_impact"] = round(float(best[0]), 1)
    except Exception as e:  # noqa: BLE001
        return {"op": "collide", "ok": False, "error": f"事故仿真失败: {e}"}

    sep = sim.get("separation") or {}

    def _save_snapshot():
        """记录"事故前/事故后"。必须在**所有**改动做完之后调用（包括兜底的
        contact_place），否则下一次进来会以为轨迹被动过而放弃幂等重放。"""
        try:
            snaps[key] = {
                "before_a": before_a, "before_b": before_b,
                "after_a": _dump_poses(tm, a, frames), "after_b": _dump_poses(tm, b, frames),
            }
        except Exception:  # noqa: BLE001
            pass

    out = {
        "op": "collide", "a": a, "b": b, "ok": bool(sim.get("collided")),
        "frame": int(sim["collision_frame"]) if sim.get("collision_frame") is not None else None,
        "impact_frame_target": int(impact_frame),
        "impact_attempts": int(sim.get("impact_attempts") or 0),
        "attacker_target_speed": (round(float(sim.get("target_speed_a")), 2)
                                  if sim.get("target_speed_a") is not None else None),
        "window": [int(frames[0]), int(frames[-1])],
        "post_impact_frames": int(max(0, frames[-1] - impact_frame)),
        "victim_keeps_own_track": True,
        "attacker_dims": [round(float(x), 3) for x in dims_a],
        "victim_dims": [round(float(x), 3) for x in dims_v],
        "max_penetration_before": sep.get("max_penetration_before"),
        "max_penetration_after": sep.get("max_penetration_after"),
        "separation_frames_fixed": sep.get("frames_fixed"),
        "engine": "corner_case.simulate_pair_collision",
    }
    if replayed:
        out["idempotent_replay"] = True
    elif replay_miss is not None:
        out["replay_miss"] = replay_miss
    if sim.get("collided"):
        out["note"] = "与 corner case 共用同一套追击/碰撞/防穿模仿真"
        # 贴地兜底：仿真沿用了"锁定首帧高度"的写法，轨迹一旦跑到静态点覆盖边缘就会
        # 整体浮空；这里只改 y（保留 x/z 与朝向），把车按（x,z）处的路面贴回去。
        # **只对合成物体做**：真实轨迹的 y 是数据本身，强行贴地会把它按估计误差上下拽。
        try:
            synth = getattr(tm, "synthetic_tracks", {}) or {}
            # 肇事车的轨迹是仿真**重新合成**的，它的"原来的 y"已经没有意义了 → 一律贴地；
            # 受害车跟随自己的轨迹（victim_follow_track），它的 y 还是数据本身 → 只在它
            # 也是合成物体时才贴，避免把真实轨迹按估计误差上下拽。
            fa2 = _reground_track(tm, a, frames, dims=dims_a)
            # 受害车 = 它**自己**的轨迹（含高度）→ 这里不贴地、不平滑、不动它；
            # 只有防穿模分离在必要时会把它推开一点点。
            fv2 = 0
            _smooth_track_heights(tm, a, frames)
            if fa2 or fv2:
                out["regrounded_frames"] = {"attacker": int(fa2), "victim": int(fv2)}
            # 贴地/平滑会改变"高度方向"的相对关系：原本靠高度错开（SAT 判不重叠）的两车，
            # 贴回同一路面后**可能**在 XZ 上真的重叠了。所以贴完必须重新做一次
            # 逐帧防穿模分离，保证"贴地"不会引入新的穿模。
            sep2 = corner_case.separate_pair(tm, a, b, frames,
                                            dims_a=dims_a, dims_v=dims_v,
                                            from_frame=frames[0])
            out["separation_after_reground"] = {
                "frames_fixed": sep2.get("frames_fixed"),
                "max_penetration_before": sep2.get("max_penetration_before"),
                "max_penetration_after": sep2.get("max_penetration_after"),
                "still_overlapping": bool(float(sep2.get("max_penetration_after") or 0.0) > 0.01),
            }
        except Exception as e:  # noqa: BLE001
            out["reground_error"] = str(e)
        _save_snapshot()
        return out

    # 仿真窗口内没撞上（或撞得离期望帧太远）→ 看"两车**原本**的轨迹"最近能到多近：
    #   很近（差几米）→ 兜底把它们平滑推到"刚好接触"，合理；
    #   很远（十几米以上）→ 这段窗口里两条轨迹根本不会相遇，硬挪过去只会造出一条
    #   莫名其妙的长途追车轨迹（用户看到的就是"撞的不是该撞的那辆/位置乱跳"）。
    #   这种情况**回滚仿真改动**、如实报告"撞不上"，保留原轨迹。
    try:
        reach = 0.5 * (max(float(dims_a[0]), float(dims_a[2]))
                       + max(float(dims_v[0]), float(dims_v[2]))) + 12.0
    except Exception:  # noqa: BLE001
        reach = 15.0
    closest = float("inf")
    for f in frames:
        pa0, pb0 = before_a.get(f), before_b.get(f)
        if pa0 is None or pb0 is None:
            continue
        dd = (np.asarray(pa0, dtype=np.float64)[:3, 3]
              - np.asarray(pb0, dtype=np.float64)[:3, 3])
        closest = min(closest, float(np.linalg.norm(dd)))
    if closest > reach:
        for f, p in before_a.items():
            tm.set_track_pose(a, f, p)
        for f, p in before_b.items():
            tm.set_track_pose(b, f, p)
        msg = (f"窗口内撞不上 T:{b}：两车原本的轨迹最近也有 {closest:.1f}m"
               f"（需要 ≤{reach:.1f}m 才可能接触），它可能已经跑掉了/离得太远。"
               "已保留原轨迹，不做碰撞。")
        out["ok"] = False
        out["error"] = msg
        out["warn"] = msg
        out["closest_approach_m"] = round(float(closest), 2)
        out["reach_m"] = round(float(reach), 2)
        _save_snapshot()
        return out

    cf = int(impact_frame)
    cf = max(frames[0] + 1, min(frames[-1], cf))
    _contact_place(tm, a, b, frames, cf, dims_a, dims_v)
    try:
        if int(a) in (getattr(tm, "synthetic_tracks", {}) or {}):
            _smooth_track_heights(tm, a, frames)
    except Exception:  # noqa: BLE001
        pass
    out["ok"] = True
    out["frame"] = cf
    out["post_impact_frames"] = int(max(0, frames[-1] - cf))
    out["fallback"] = "contact_place"
    out["warn"] = ("追击仿真的窗口内两车没能自然撞上（对方可能跑掉了），"
                   "已改为把肇事车平滑推到‘第 %d 帧刚好接触’，之后贴着推；"
                   "受害车自己的轨迹不变，包围盒不会重叠" % int(cf))
    # 兜底摆放是"先算间距、再对朝向"的，朝向变化会改变包围盒footprint，可能仍然压在一起；
    # 这里统一再跑一次逐帧防穿模分离（与 corner case 同一套），保证"不穿模"这个硬要求。
    try:
        sep2 = corner_case.separate_pair(tm, a, b, frames, dims_a=dims_a, dims_v=dims_v,
                                        from_frame=frames[0])
        out["separation_after_fallback"] = {
            "frames_fixed": sep2.get("frames_fixed"),
            "max_penetration_after": sep2.get("max_penetration_after"),
        }
    except Exception as e:  # noqa: BLE001
        out["separation_error"] = str(e)
    try:
        pa = tm.get_track_pose(a, cf)
        pb = tm.get_track_pose(b, cf)
        from dggt.scene_edit.collision_physics import check_collision as _cc
        hit, _ = _cc(pa, dims_a, pb, dims_v)
        out["contact_overlap"] = bool(hit)
    except Exception:  # noqa: BLE001
        pass
    _save_snapshot()
    return out
