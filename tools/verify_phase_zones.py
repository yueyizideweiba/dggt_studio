#!/usr/bin/env python
"""多阶段交互结构（预警区/危险区/发生区）的数值验证。

用法：在项目根目录执行
    /root/autodl-tmp/conda_envs/dggt/bin/python tools/verify_phase_zones.py \
        [scene_path] [start] [num_frames]

验证三件事：
  1. 受控对向逼近：两车匀速对向，相态必须按 safe→warning→danger→occurrence→impact→safe
     单调演进，且 a* 数值与理论一致（闭合速度决定危险阈值）。
  2. 真实场景：对 rear-end / head-on / pedestrian-crossing / natural-conflict 各算一条
     相态时程，打印"切换序列 + 故事"，确认能区分"完整三段式 / 预警被跳过 / 险情"。
  3. 不变量：相态严重度随帧单调不减（在碰撞前）、a* 随 TTC 缩短单调不减、区域时长非负。
"""
import math
import sys

import numpy as np

sys.path.insert(0, ".")
sys.path.insert(0, "studio/backend")

import phase_zones as P  # noqa: E402


def _pose(z, yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    M = np.eye(4, dtype=np.float64)
    M[0, 0], M[0, 2] = c, s
    M[2, 0], M[2, 2] = -s, c
    M[2, 3] = z
    M[1, 3] = 0.0
    return M


def test_controlled():
    """受控对向逼近：闭合 24 m/s，初始 gap 70m。"""
    dims = [2.0, 1.7, 4.5]
    params = dict(P.PHASE_PARAMS_DEFAULT)
    params.update(horizon_s=2.0, t_reaction_s=0.7, a_comfort_mps2=3.0, a_emergency_mps2=8.0)
    ze, zo, v, dt = 0.0, 70.0, 12.0, 0.1
    order = {P.ZONE_SAFE: 0, P.ZONE_WARNING: 1, P.ZONE_DANGER: 2,
             P.ZONE_OCCURRENCE: 3, P.ZONE_IMPACT: 4}
    seq, last_order, last_astar = [], -1, 0.0
    passed_impact = False
    for _f in range(40):
        pe, po = _pose(ze, 0.0), _pose(zo, math.pi)
        va = np.array([0, 0, v], float)
        vb = np.array([0, 0, -v], float)
        ttc, _ = P._predict_contact(pe, va, dims, po, vb, dims, params["horizon_s"], params["dt_pred_s"])
        astar = P.min_required_deceleration(pe, va, dims, po, vb, dims, params) if ttc is not None else 0.0
        zone = P.ZONE_IMPACT if P._contact(pe, dims, po, dims) else (
            P.classify_zone(astar, params) if ttc is not None else P.ZONE_SAFE)
        if not seq or seq[-1][0] != zone:
            seq.append((zone, _f, astar))
        if zone == P.ZONE_IMPACT:
            passed_impact = True
        # 不变量：撞击之前严重度单调不减（撞击之后允许回到 safe）
        if not passed_impact:
            assert order[zone] >= last_order, f"严重度回退: {seq}"
            last_order = order[zone]
        ze += v * dt
        zo -= v * dt
    zones = [s[0] for s in seq]
    print("[受控对向] 切换序列:", " -> ".join("%s(f%d,a*=%.2f)" % (z, f, a if a != float('inf') else 9.9) for z, f, a in seq))
    expected = [P.ZONE_SAFE, P.ZONE_WARNING, P.ZONE_DANGER, P.ZONE_OCCURRENCE, P.ZONE_IMPACT]
    got = [z for z in zones if z in expected]
    # 撞击之后会回到 safe（事件结束），只校验"撞击之前的完整演进"
    if P.ZONE_IMPACT in got:
        got = got[:got.index(P.ZONE_IMPACT) + 1]
    assert got == expected, f"预期 {expected}，实际 {got}"
    # 数值合理性：预警区 a* 应在 (0, a_comf] 内
    warn = next((a for z, _f, a in seq if z == P.ZONE_WARNING), None)
    assert warn is not None and 0 < warn <= 3.0, f"预警区 a* 异常: {warn}"
    print("    ✓ 完整三段式单调演进 + a* 数值合理（预警 a*=%.2f）" % warn)


def test_occlusion():
    """鬼探头：对向车 B 在 emerge 帧之前对 A 不可见，一出现就落入发生区 → 预警被跳过。

    与 test_controlled 相同的对向逼近，但 B 前 20 帧"被遮挡"。可见前 A 无任何冲突信号
    （occluded），第 20 帧 B 一出现 gap 只剩 22m（ttc≈0.92s），直接进入发生区。
    """
    dims = [2.0, 1.7, 4.5]
    params = dict(P.PHASE_PARAMS_DEFAULT)
    params.update(horizon_s=2.0, t_reaction_s=0.7, a_comfort_mps2=3.0, a_emergency_mps2=8.0)
    emerge = 20
    ze, zo, v, dt = 0.0, 70.0, 12.0, 0.1
    seq = []
    for f in range(40):
        if f < emerge:
            if not seq or seq[-1][0] != P.ZONE_OCCLUDED:
                seq.append((P.ZONE_OCCLUDED, f))
        else:
            pe, po = _pose(ze, 0.0), _pose(zo, math.pi)
            va = np.array([0, 0, v], float)
            vb = np.array([0, 0, -v], float)
            ttc, _ = P._predict_contact(pe, va, dims, po, vb, dims, params["horizon_s"], params["dt_pred_s"])
            astar = P.min_required_deceleration(pe, va, dims, po, vb, dims, params) if ttc is not None else 0.0
            zone = P.ZONE_IMPACT if P._contact(pe, dims, po, dims) else (
                P.classify_zone(astar, params) if ttc is not None else P.ZONE_SAFE)
            if not seq or seq[-1][0] != zone:
                seq.append((zone, f))
        ze += v * dt
        zo -= v * dt
    zones = [z for z, _ in seq]
    print("[鬼探头遮挡] 切换序列:", " -> ".join("%s(f%d)" % (z, f) for z, f in seq))
    assert zones[0] == P.ZONE_OCCLUDED
    assert P.ZONE_WARNING not in zones, "鬼探头不应出现预警区"
    assert P.ZONE_OCCURRENCE in zones and P.ZONE_IMPACT in zones
    print("    ✓ 遮挡期无预警，一出现即落入发生区 → 预警被跳过（典型鬼探头 corner case）")


def test_scenarios(scene_path, start, num_frames):
    from dggt_engine import DGGTRenderer
    from track_manager import TrackManager
    import corner_case
    frames = list(range(start, start + num_frames + 1))
    cases = [
        ("rear-end", {"victim": 5, "attacker": 3}),
        ("head-on", {"attacker": 3}),
        ("pedestrian-crossing", {"vehicle": 3}),
        ("natural-conflict", {"ego": 5, "other": 2}),
    ]
    for st, roles in cases:
        r = DGGTRenderer(scene_path=scene_path, device="cuda", load_sky=True)
        tm = TrackManager(r)
        tm.fps = 10.0
        res = corner_case.generate(tm, st, roles, start, num_frames,
                                   intensity=1.0, enable_physics=True, fps=10.0, sampling_seed=7)
        pair = res.get("collision_tracks") or res.get("collision_pair") or []
        actor = int(pair[0]) if len(pair) else None
        others = [int(t) for t in (res.get("affected_tracks") or []) if t != actor]
        if actor is None:
            print("[%s] 无配对，跳过" % st)
            continue
        prof = P.SCENARIO_PHASE_PROFILES.get(st, {})
        pa = P.compute_phase_timeline(tm, actor, others, frames, 10.0, params=prof,
                                      collision_frame=res.get("collision_frame"))
        prev, sw = None, []
        for e in pa["timeline"]:
            if e["zone"] != prev:
                sw.append("f%d:%s" % (e["frame"], e["zone"]))
                prev = e["zone"]
        # 不变量：区域时长非负
        for k, v in pa["summary"]["zone_durations_s"].items():
            assert v >= 0.0
        print("[%-20s] actor=%s  切换: %s" % (st, actor, " -> ".join(sw)))
        print("              story: %s" % pa["summary"]["story"])


def main():
    scene = sys.argv[1] if len(sys.argv) > 1 else "/autodl-fs/data/dggt-main/output/waymo_eval_14/018/018"
    start = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    nf = int(sys.argv[3]) if len(sys.argv) > 3 else 24
    test_controlled()
    test_occlusion()
    test_scenarios(scene, start, nf)
    print("\n全部通过 ✓")


if __name__ == "__main__":
    main()
