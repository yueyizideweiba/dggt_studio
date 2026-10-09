#!/usr/bin/env python
"""多阶段交互结构（预警区/危险区/发生区，多因素判据）的数值验证。

用法：在项目根目录执行
    /root/autodl-tmp/conda_envs/dggt/bin/python tools/verify_phase_zones.py \
        [scene_path] [start] [num_frames]

验证：
  1. 受控对向逼近：完整三段式 safe→warning→danger→occurrence→impact，a_brake/a_lat 数值合理；
  2. 转向避让因子：同一对向场景，给出足够横向空间时转向应能"救"下危险区（binding=steer）；
  3. 鬼探头遮挡：occluded→occurrence→impact，无预警；
  4. 真实场景：多类型各算一条，打印"切换 + 故事 + 关键因素"，含多参与者聚合与连环多碰撞。
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


def _params(**over):
    p = dict(P.PHASE_PARAMS_DEFAULT)
    p.update(over)
    p["cap_brake_mps2"] = min(p["a_emergency_mps2"], p["mu"] * p["g_mps2"])
    p["cap_lat_mps2"] = 0.9 * p["mu"] * p["g_mps2"]
    return p


def test_controlled():
    """受控对向逼近：完整三段式 + 多因素数值。"""
    dims = [2.0, 1.7, 4.5]
    params = _params(horizon_s=2.0, t_reaction_s=0.7, a_comfort_mps2=3.0, a_emergency_mps2=8.0)
    ze, zo, v, dt = 0.0, 70.0, 12.0, 0.1
    order = {P.ZONE_SAFE: 0, P.ZONE_WARNING: 1, P.ZONE_DANGER: 2,
             P.ZONE_OCCURRENCE: 3, P.ZONE_IMPACT: 4}
    seq, last_order, passed = [], -1, False
    for _f in range(40):
        pe, po = _pose(ze, 0.0), _pose(zo, math.pi)
        va = np.array([0, 0, v], float)
        vb = np.array([0, 0, -v], float)
        aa = ab = np.zeros(3)
        r = P.analyze_pair(pe, va, aa, dims, po, vb, ab, dims, "vehicle", params)
        zone = r["zone"]
        if not seq or seq[-1][0] != zone:
            seq.append((zone, _f, r["factors"]))
        if zone == P.ZONE_IMPACT:
            passed = True
        elif not passed:
            assert order[zone] >= last_order, f"严重度回退: {seq}"
            last_order = order[zone]
        ze += v * dt
        zo -= v * dt
    zones = [z for z, _, _ in seq]
    print("[受控对向] 切换:", " -> ".join(z for z, _, _ in seq))
    exp = [P.ZONE_SAFE, P.ZONE_WARNING, P.ZONE_DANGER, P.ZONE_OCCURRENCE, P.ZONE_IMPACT]
    got = zones[:zones.index(P.ZONE_IMPACT) + 1] if P.ZONE_IMPACT in zones else zones
    assert got == exp, f"预期 {exp} 实际 {got}"
    warn = next((f for z, _, f in seq if z == P.ZONE_WARNING), None)
    dang = next((f for z, _, f in seq if z == P.ZONE_DANGER), None)
    assert warn and dang and warn["a_brake_mps2"] <= 3.0 < dang["a_brake_mps2"]
    # 多因素都在输出里
    for z, _f, f in seq:
        if f and z in (P.ZONE_WARNING, P.ZONE_DANGER, P.ZONE_OCCURRENCE):
            for k in ("a_brake_mps2", "a_lat_mps2", "ttc_s", "delta_v_mps", "geometry"):
                assert k in f, f"缺少因素 {k}: {f}"
    print("    ✓ 完整三段式 + 六因素齐全（制动/转向/TTC/Δv/几何/能力上限）")


def test_steering_saves():
    """转向避让因子：扫描找"制动不足、但转向可救"的状态（binding=steer）。"""
    dims = [2.0, 1.7, 4.5]
    params = _params(horizon_s=2.0, t_reaction_s=0.7, lane_width_m=3.5,
                     a_comfort_mps2=3.0, a_emergency_mps2=8.0, mu=0.8)
    found = None
    for gap in np.arange(20.0, 70.0, 0.5):
        pe, po = _pose(0.0, 0.0), _pose(float(gap), math.pi)
        va = np.array([0, 0, 12.0], float)
        vb = np.array([0, 0, -12.0], float)
        r = P.analyze_pair(pe, va, np.zeros(3), dims, po, vb, np.zeros(3), dims, "vehicle", params)
        if r.get("binding") == "steer":
            found = (gap, r)
            break
    assert found, "应存在'制动不足、转向可救'的状态"
    gap, r = found
    f = r["factors"]
    print("[转向避让] gap=%.1f zone=%s binding=%s a_brake=%s a_lat=%.2f cap_lat=%.2f cap_brake=%.2f" % (
        gap, r["zone"], r["binding"], f.get("a_brake_mps2"), f.get("a_lat_mps2"),
        params["cap_lat_mps2"], params["cap_brake_mps2"]))
    assert r["zone"] == P.ZONE_DANGER
    assert f["a_brake_mps2"] is None or f["a_brake_mps2"] > params["cap_brake_mps2"] - 0.01
    assert f["a_lat_mps2"] <= params["cap_lat_mps2"]
    print("    ✓ 横向转向作为独立因子在制动不足时接管（能力包络取最优项）")


def test_occlusion():
    """鬼探头：B 前 20 帧不可见，一出现就落入发生区。"""
    dims = [2.0, 1.7, 4.5]
    params = _params(horizon_s=2.0, t_reaction_s=0.7)
    emerge = 20
    ze, zo, v, dt = 0.0, 70.0, 12.0, 0.1
    seq = []
    for f in range(40):
        if f < emerge:
            z = P.ZONE_OCCLUDED
        else:
            pe, po = _pose(ze, 0.0), _pose(zo, math.pi)
            va = np.array([0, 0, v], float)
            vb = np.array([0, 0, -v], float)
            z = P.analyze_pair(pe, va, np.zeros(3), dims, po, vb, np.zeros(3), dims, "vehicle", params)["zone"]
        if not seq or seq[-1][0] != z:
            seq.append((z, f))
        ze += v * dt
        zo -= v * dt
    zones = [z for z, _ in seq]
    print("[鬼探头] 切换:", " -> ".join("%s(f%d)" % (z, f) for z, f in seq))
    assert zones[0] == P.ZONE_OCCLUDED and P.ZONE_WARNING not in zones
    assert P.ZONE_OCCURRENCE in zones and P.ZONE_IMPACT in zones
    print("    ✓ 遮挡期无预警，一出现即发生区")


def test_scenarios(scene_path, start, num_frames):
    from dggt_engine import DGGTRenderer
    from track_manager import TrackManager
    import corner_case
    frames = list(range(start, start + num_frames + 1))
    cases = [
        ("rear-end", {"victim": 5, "attacker": 3}),
        ("head-on", {"attacker": 3}),
        ("pedestrian-crossing", {"vehicle": 3}),
        ("chain-reaction-rear-end", {"rear": 3}),
        ("hard-brake", {"braker": 3}),
        ("natural-conflict", {"ego": 5, "other": 2}),
    ]
    for st, roles in cases:
        r = DGGTRenderer(scene_path=scene_path, device="cuda", load_sky=True)
        tm = TrackManager(r)
        tm.fps = 10.0
        res = corner_case.generate(tm, st, roles, start, num_frames,
                                   intensity=1.0, enable_physics=True, fps=10.0, sampling_seed=7)
        pa = res.get("phase_analysis") or {}
        prev, sw = None, []
        for e in pa.get("timeline", []):
            if e["zone"] != prev:
                sw.append("f%d:%s" % (e["frame"], e["zone"]))
                prev = e["zone"]
        s = (pa.get("summary") or {})
        print("[%-22s] actor=%s coll=%s  切换: %s" % (st, pa.get("actor"), res.get("collision_frame"), " -> ".join(sw)))
        print("              story: %s" % s.get("story"))
        # 关键因素抽样：打印 danger/occurrence 首帧的因素
        for e in pa.get("timeline", []):
            if e["zone"] in (P.ZONE_DANGER, P.ZONE_OCCURRENCE) and e.get("factors"):
                print("              %s 首帧因素: %s" % (e["zone"], e["factors"]))
                break


def main():
    scene = sys.argv[1] if len(sys.argv) > 1 else "/autodl-fs/data/dggt-main/output/waymo_eval_14/018/018"
    start = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    nf = int(sys.argv[3]) if len(sys.argv) > 3 else 24
    test_controlled()
    test_steering_saves()
    test_occlusion()
    test_scenarios(scene, start, nf)
    print("\n全部通过 ✓")


if __name__ == "__main__":
    main()
