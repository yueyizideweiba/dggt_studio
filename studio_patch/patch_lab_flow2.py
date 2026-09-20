#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""流程改造第二版：批量生成放到最前，之后"逐条实例"送入 ③/④；并写清 ③ 与 ④ 的关系。

    python patch_lab_flow2.py [--base DIR] [--dry-run]
"""
import argparse
import shutil
import sys
from pathlib import Path

DEFAULT_BASE = Path("/autodl-fs/data/dggt-main/studio/frontend")

# ---- 1) 标签页：批量生成放最前 -------------------------------------------------
TABS_OLD = '''                <button class="lab-tab active" data-tab="graph">① 挑事故（关系图）</button>
                <button class="lab-tab" data-tab="batch">② 批量生成（含视频）</button>'''
TABS_NEW = '''                <button class="lab-tab active" data-tab="batch">① 批量生成（起点·含视频）</button>
                <button class="lab-tab" data-tab="graph">② 挑事故（关系图·可选）</button>'''

PANEL_GRAPH_OLD = '''                <div class="lab-panel active" data-panel="graph">'''
PANEL_GRAPH_NEW = '''                <div class="lab-panel" data-panel="graph">'''
PANEL_BATCH_OLD = '''                <div class="lab-panel" data-panel="batch">'''
PANEL_BATCH_NEW = '''                <div class="lab-panel active" data-panel="batch">'''

# ---- 2) ③ 闭环评估 顶部说明 ---------------------------------------------------
SIM_NOTE_OLD = '''                    <div class="lab-subtitle">可信域（novel-view trust）：标定 → 扫描 → 逐帧判定</div>'''
SIM_NOTE_NEW = '''                    <div class="cc-status" style="margin-bottom:8px">
                        <b>③ 闭环评估</b>（用你们自己的 4DGS 重建渲染）和 <b>④ CARLA 验证</b>（第三方物理仿真器）是
                        <b>两条独立的验证通道</b>，都吃同一份事故轨迹：③ 回答"这个视角渲染可不可信、自车按动作推进会怎样"，
                        ④ 回答"在真仿真器里会不会撞、撞得有多严重"。<b>不必</b>先 ③ 再 ④；只关心物理效果可以直接 ④。
                        推荐用法：③ 快速筛（便宜）→ ④ 复核（有物理/碰撞事件）。<br>
                        下面这些按钮都是对 <b>当前场景状态</b> 做评估；要对"批量生成的某一条"评估，回
                        ① 批量生成，点那一行的「▶ 载入场景」。
                    </div>
                    <div class="lab-subtitle">可信域（novel-view trust）：标定 → 扫描 → 逐帧判定</div>'''

CARLA_NOTE_OLD = '''                    <div class="lab-subtitle">CARLA 闭环仿真：corner case 场景 → CARLA 里跑（含碰撞事件）→ 出视频 / 看实时画面 / 导出标准 OpenSCENARIO</div>'''
CARLA_NOTE_NEW = '''                    <div class="lab-subtitle">CARLA 闭环仿真：corner case 场景 → CARLA 里跑（含碰撞事件）→ 出视频 / 看实时画面 / 导出标准 OpenSCENARIO</div>
                    <div class="cc-status" style="margin-bottom:6px">
                        这是 <b>④ CARLA 验证</b>（③ 闭环评估是另一条独立通道，不必须先做 ③）。
                        两种送法：<b>用当前编辑的场景</b>（含刚生成/编辑的轨迹，现场导出）或
                        <b>从①批量生成里点「🎬 CARLA 这条」</b>（直接吃该实例的 world.json，不依赖内存场景，最稳）。
                    </div>'''

# ---- 3) 批量表格：按钮改成"逐条送后续" ----------------------------------------
ROW_BTN_OLD = '''                <td>${vid}</td>
                <td><button class="btn btn-secondary btn-small lab-batch-next" data-i="${this._escapeHtml(m.instance_id || '')}" data-j="sim">→ 闭环</button>
                    <button class="btn btn-primary btn-small lab-batch-next" data-i="${this._escapeHtml(m.instance_id || '')}" data-j="carla">→ CARLA</button></td>'''
ROW_BTN_NEW = '''                <td>${vid}</td>
                <td><button class="btn btn-secondary btn-small lab-batch-next" data-i="${this._escapeHtml(m.instance_id || '')}" data-j="apply" ${m.world_json ? '' : 'disabled'}>▶ 载入场景</button>
                    <button class="btn btn-primary btn-small lab-batch-next" data-i="${this._escapeHtml(m.instance_id || '')}" data-j="carla" ${m.world_json ? '' : 'disabled'}>🎬 CARLA 这条</button></td>'''

HANDLER_OLD = '''        // 每一行都能直接"送去下一步"（带案例上下文）
        box.querySelectorAll('.lab-batch-next').forEach(btn => {
            btn.addEventListener('click', () => {
                const man = this._lastBatchManifest || [];
                const m = man.find(x => (x.instance_id || '') === btn.dataset.i) || {};
                const ctx = this._lastBatchCtx || { start_frame: 0, num_frames: 20 };
                this._labSetCase({
                    scenario_type: m.scenario_type || '', seed: m.seed, roles: m.roles || null,
                    start_frame: ctx.start_frame, num_frames: ctx.num_frames,
                    quality: m.valid ? '通过' : '未通过'
                }, `批量生产 ${m.instance_id || ''}`);
                if (btn.dataset.j === 'carla') {
                    this._labSwitchTab('carla');
                    this._labPrefillCarla();
                } else {
                    this._labSwitchTab('sim');
                }
            });
        });'''

HANDLER_NEW = r'''        // 每一行 = 一条实例：可以"载入场景继续编辑/评估"，也可以"直接送 CARLA"
        box.querySelectorAll('.lab-batch-next').forEach(btn => {
            btn.addEventListener('click', () => {
                const man = this._lastBatchManifest || [];
                const m = man.find(x => (x.instance_id || '') === btn.dataset.i) || {};
                if (btn.dataset.j === 'carla') this._labSendInstanceToCarla(m);
                else this._labApplyInstance(m);
            });
        });'''

# ---- 4) 两个新动作 -------------------------------------------------------------
NEW_ACTIONS_OLD = '''    // ③ 闭环评估 → ④ CARLA 验证
    _labGoCarla() {'''
NEW_ACTIONS_NEW = r'''    // 把某条批量实例"载入场景"（确定性重放），之后可以在 ③ 里逐帧评估、继续编辑
    async _labApplyInstance(m) {
        if (!this.state.sceneId) { this._carlaStatus('请先加载场景，再载入实例', true); return; }
        if (!m || !m.instance_id) { this._carlaStatus('这条实例没有可用信息', true); return; }
        this._carlaStatus(`正在把实例 ${m.instance_id} 还原进场景（会先把场景重置成干净状态，之后可继续编辑）…`);
        try {
            const r = await fetch(`${API_BASE}/carla/apply_instance`, {
                method: 'POST', headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ scene_id: this.state.sceneId, instance: m, reset_scene: true })
            });
            const d = await r.json();
            if (!d.success) throw new Error(d.detail || '还原失败');
            this._labSetCase({
                scenario_type: d.scenario_type, seed: d.seed, roles: d.roles || null,
                start_frame: d.start_frame, num_frames: d.num_frames, instance_id: d.instance_id,
                quality: m.valid ? '通过' : '未通过', world_json: m.world_json || null
            }, `实例 ${d.instance_id}`);
            this._labSwitchTab('sim');
            this._simStatus('simP23Status',
                `已把 ${d.instance_id} 还原进场景（碰撞帧 ${d.collision_frame ?? '—'}）——`
                + '可以直接「跑闭环 rollout」，或点右上「下一步：④ CARLA 验证」');
        } catch (e) { this._carlaStatus('还原实例失败: ' + e.message, true); }
    }

    // 把某条批量实例直接送 ④ CARLA（用实例自带的 world.json，不依赖内存里的场景）
    _labSendInstanceToCarla(m) {
        this._labSetCase({
            scenario_type: m.scenario_type || '', seed: m.seed, roles: m.roles || null,
            start_frame: m.start_frame, num_frames: m.num_frames, instance_id: m.instance_id,
            quality: m.valid ? '通过' : '未通过', world_json: m.world_json || null
        }, `实例 ${m.instance_id || ''}`);
        this._labSwitchTab('carla');
        const chk = document.getElementById('carlaUseCurrent');
        const sel = document.getElementById('carlaSceneSel');
        if (m.world_json && sel) {
            if (chk) chk.checked = false;            // 走"实例 world.json"这条路，最稳
            if (![...sel.options].some(o => o.value === m.world_json)) {
                const o = document.createElement('option');
                o.value = m.world_json;
                o.textContent = `【批量实例】${m.instance_id} · ${m.world_json}`;
                sel.insertBefore(o, sel.firstChild);
            }
            sel.value = m.world_json;
        } else if (chk) {
            chk.checked = true;
        }
        this._labPrefillCarla();
        this._carlaStatus(`已选中批量实例 ${m.instance_id}：点「▶ 开始 CARLA 仿真并出视频」即可`
            + (m.world_json ? '（直接吃实例的 world.json，不依赖内存里的场景）' : '（会现场导出当前场景）'));
    }

    // ③ 闭环评估 → ④ CARLA 验证
    _labGoCarla() {'''

# ---- 5) 提示文案（html 与 js 各一处）------------------------------------------
HINT_OLD = '''还没有案例 —— 先在 ① 挑事故、② 批量生成里做出一个事故场景'''
HINT_NEW = '''还没有案例 —— 先在 ① 批量生成里跑一批，再点某一行右侧「▶ 载入场景」或「🎬 CARLA 这条」'''
HINT_OLD_JS = '''            : '还没有案例 —— 先在 ① 挑事故、② 批量生成里做出一个事故场景';'''
HINT_NEW_JS = '''            : '还没有案例 —— 先在 ① 批量生成里跑一批，再点某一行右侧「▶ 载入场景」或「🎬 CARLA 这条」';'''

# ---- 6) 批量完成后提示"是否重置了场景" ----------------------------------------
RESET_NOTICE_OLD = '''        if (data.video_dir) html += `<span>视频目录: <b>${this._escapeHtml(data.video_dir)}</b></span>`;'''
RESET_NOTICE_NEW = '''        if (data.reset_scene) {
            html += `<span style="color:#f59e0b">已自动清空批量前的生成/编辑残留（不清的话不同类型会生成出一样的视频）</span>`;
        }
        if (data.video_dir) html += `<span>视频目录: <b>${this._escapeHtml(data.video_dir)}</b></span>`;'''


def patch(path: Path, pairs, dry: bool):
    if not path.exists():
        print(f"!! 找不到 {path}")
        return False
    txt = path.read_text(encoding="utf-8")
    for i, (old, new) in enumerate(pairs, 1):
        n = txt.count(old)
        if n != 1:
            print(f"!! {path.name} 第 {i} 处锚点出现 {n} 次：{old[:70]!r}")
            return False
        if not dry:
            txt = txt.replace(old, new, 1)
    if not dry:
        if not Path(str(path) + ".bak3").exists():
            shutil.copy2(path, str(path) + ".bak3")
        path.write_text(txt, encoding="utf-8")
    print(f"{'✓ 校验' if dry else '✅ 已补丁'} {path.name}（{len(pairs)} 处）")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=str(DEFAULT_BASE))
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    base = Path(a.base)
    ok = patch(base / "index.html", [
        (TABS_OLD, TABS_NEW),
        (PANEL_GRAPH_OLD, PANEL_GRAPH_NEW),
        (PANEL_BATCH_OLD, PANEL_BATCH_NEW),
        (SIM_NOTE_OLD, SIM_NOTE_NEW),
        (CARLA_NOTE_OLD, CARLA_NOTE_NEW),
        (HINT_OLD, HINT_NEW),
    ], a.dry_run)
    ok &= patch(base / "app.js", [
        (ROW_BTN_OLD, ROW_BTN_NEW),
        (HANDLER_OLD, HANDLER_NEW),
        (NEW_ACTIONS_OLD, NEW_ACTIONS_NEW),
        (HINT_OLD_JS, HINT_NEW_JS),
        (RESET_NOTICE_OLD, RESET_NOTICE_NEW),
    ], a.dry_run)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
