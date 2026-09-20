#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把 Corner Case 工作台改造成"一条流水线" + 给 CARLA 面板加显存显示/释放。

用法：
    python patch_lab_pipeline.py                 # 打到默认（服务器）目录
    python patch_lab_pipeline.py --base <dir>    # 打到指定目录
    python patch_lab_pipeline.py --dry-run       # 只检查锚点能不能匹配（不改文件）

做的事：
  1) 六个标签页重排+改名，明确成一条流程：① 挑事故 → ② 批量生成 → ③ 闭环评估 → ④ CARLA 验证 → ⑤ 训练语料 → ⑥ 自检
  2) 标签页下面加"当前案例条 + 上一步/下一步"（跨板块携带 事故类型/seed/参与者/帧窗/质量门）
  3) 批量生产表格每行加「→ 闭环」「→ CARLA」两个跳转按钮（带案例上下文）
  4) ③ 闭环评估里加「→ 送去 CARLA 验证」
  5) ④ CARLA 面板：显示显存占用 + 「🧹 释放显存」按钮 + 显存警告
"""
import argparse
import shutil
import sys
from pathlib import Path

DEFAULT_BASE = Path("/autodl-fs/data/dggt-main/studio/frontend")

# --------------------------------------------------------------------------- #
HTML_TABS_OLD = '''            <div class="lab-tabs">
                <button class="lab-tab active" data-tab="graph">① 场景关系图</button>
                <button class="lab-tab" data-tab="batch">② 批量生产（含视频）</button>
                <button class="lab-tab" data-tab="gnn">③ 模型训练</button>
                <button class="lab-tab" data-tab="selftest">④ 自检诊断</button>
                <button class="lab-tab" data-tab="sim">⑤ 闭环仿真工具</button>
                <button class="lab-tab" data-tab="carla">⑥ CARLA 仿真</button>
            </div>'''

HTML_TABS_NEW = '''            <div class="lab-tabs">
                <button class="lab-tab active" data-tab="graph">① 挑事故（关系图）</button>
                <button class="lab-tab" data-tab="batch">② 批量生成（含视频）</button>
                <button class="lab-tab" data-tab="sim">③ 闭环评估（可信域/资产库）</button>
                <button class="lab-tab" data-tab="carla">④ CARLA 验证（视频/实时）</button>
                <button class="lab-tab" data-tab="gnn">⑤ 训练语料</button>
                <button class="lab-tab" data-tab="selftest">⑥ 自检诊断</button>
            </div>
            <div id="labFlowBar" class="lab-flow">
                <span class="lab-flow-step">流程</span>
                <span id="labCaseCtx" class="cc-status" style="flex:1">还没有案例 —— 先在 ① 挑事故、② 批量生成里做出一个事故场景</span>
                <button id="labPrevStepBtn" class="btn btn-secondary btn-small">← 上一步</button>
                <button id="labNextStepBtn" class="btn btn-primary btn-small">下一步 →</button>
            </div>'''

HTML_SIM_OLD = '''                        <button id="simExpBtn" class="btn btn-primary btn-small">导出 .xosc + .cr.xml</button>
                    </div>
                    <div id="simExpStatus" class="cc-status">'''

HTML_SIM_NEW = '''                        <button id="simExpBtn" class="btn btn-primary btn-small">导出 .xosc + .cr.xml</button>
                        <button id="simToCarlaBtn" class="btn btn-primary btn-small">→ 送去 CARLA 验证</button>
                    </div>
                    <div id="simExpStatus" class="cc-status">'''

HTML_VRAM_OLD = '''                        <button id="carlaStopBtn" class="btn btn-secondary btn-small">停止</button>
                        <button id="carlaEnvBtn" class="btn btn-secondary btn-small">环境自检</button>
                    </div>'''

HTML_VRAM_NEW = '''                        <button id="carlaStopBtn" class="btn btn-secondary btn-small">停止</button>
                        <span id="carlaVram" class="carla-vram" title="鼠标悬停看是谁在占">显存 —</span>
                        <button id="carlaReleaseBtn" class="btn btn-secondary btn-small">🧹 释放显存(停 CARLA/直播)</button>
                        <button id="carlaEnvBtn" class="btn btn-secondary btn-small">环境自检</button>
                    </div>'''

HTML_NOTE_OLD = '''                        ② 视频里每个交通参与者都标了 <b>角色 + 速度</b>，顶部 HUD 有 <code>minDist / minTTC / collisions</code>。<br>'''

HTML_NOTE_NEW = '''                        <b style="color:#f59e0b">⚠️ CARLA 一开就占 ~5-6GB 显存</b>：跑完约 1 分钟没人用会自动释放，也可以随时点「🧹 释放显存」。
                        要用 SAM3D / 加载大模型之前，<b>务必先释放</b>，否则会 OOM。<br>
                        ② 视频里每个交通参与者都标了 <b>角色 + 速度</b>，顶部 HUD 有 <code>minDist / minTTC / collisions</code>。<br>'''

# --------------------------------------------------------------------------- #
JS_BIND_OLD = """        bind('carlaQuickBtn', () => { this.openLab(); this._labSwitchTab('carla'); });"""
JS_BIND_NEW = r'''        bind('carlaQuickBtn', () => { this.openLab(); this._labSwitchTab('carla'); });
        // 流程串联：上一步/下一步 + ③→④ 直达 + 释放显存
        bind('labNextStepBtn', () => this._labStep(1));
        bind('labPrevStepBtn', () => this._labStep(-1));
        bind('simToCarlaBtn', () => this._labGoCarla());
        bind('carlaReleaseBtn', () => this.carlaRelease());'''

JS_METHODS_OLD = '''    _labSwitchTab(name) {'''
JS_METHODS_NEW = r'''    // ==================== 工作台流程串联（① 挑事故 → ② 批量生成 → ③ 闭环评估 → ④ CARLA 验证 → ⑤/⑥）====================
    _labSteps() { return ['graph', 'batch', 'sim', 'carla', 'gnn', 'selftest']; }
    _labStepNames() {
        return { graph: '① 挑事故', batch: '② 批量生成', sim: '③ 闭环评估',
                 carla: '④ CARLA 验证', gnn: '⑤ 训练语料', selftest: '⑥ 自检诊断' };
    }

    _labCase() {
        if (!this._caseCtx) {
            this._caseCtx = { scenario_type: '', seed: null, roles: null,
                              start_frame: 0, num_frames: 20, quality: null, source: '' };
        }
        return this._caseCtx;
    }

    // 各步骤产出案例后都调它，把上下文带到后面所有步骤
    _labSetCase(patch, source) {
        const c = this._labCase();
        Object.assign(c, patch || {});
        if (source) c.source = source;
        this._labRenderFlow();
        return c;
    }

    _labRenderFlow() {
        const el = document.getElementById('labCaseCtx');
        const cur = document.querySelector('.lab-tab.active')?.dataset.tab || 'graph';
        const names = this._labStepNames();
        const nxt = this._labStepName(cur, 1);
        const btn = document.getElementById('labNextStepBtn');
        if (btn) btn.textContent = nxt ? `下一步：${nxt} →` : '已经是最后一步';
        if (!el) return;
        const c = this._labCase();
        const bits = [];
        if (c.scenario_type) bits.push(`事故类型 <b>${this._escapeHtml(c.scenario_type)}</b>`);
        if (c.seed !== null && c.seed !== undefined) bits.push(`seed <b>${c.seed}</b>`);
        if (c.roles && Object.keys(c.roles).length) bits.push(`参与者 ${this._escapeHtml(JSON.stringify(c.roles))}`);
        if (c.num_frames) bits.push(`帧窗 <b>${c.start_frame || 0}–${(c.start_frame || 0) + c.num_frames}</b>`);
        if (c.quality) bits.push(`质量门 <b>${c.quality}</b>`);
        el.innerHTML = bits.length
            ? `当前案例：${bits.join(' · ')}${c.source ? ` <i>（来自 ${this._escapeHtml(c.source)}）</i>` : ''}`
            : '还没有案例 —— 先在 ① 挑事故、② 批量生成里做出一个事故场景';
    }

    _labStepName(cur, delta) {
        const steps = this._labSteps();
        const names = this._labStepNames();
        const i = steps.indexOf(cur);
        if (i < 0) return '';
        const j = i + delta;
        return (j >= 0 && j < steps.length) ? names[steps[j]] : '';
    }

    _labStep(delta) {
        const cur = document.querySelector('.lab-tab.active')?.dataset.tab || 'graph';
        const steps = this._labSteps();
        const i = steps.indexOf(cur);
        const j = Math.min(steps.length - 1, Math.max(0, i + delta));
        if (j === i) return;
        this._labSwitchTab(steps[j]);
    }

    // 把"当前案例"的上下文填进 CARLA 面板
    _labPrefillCarla() {
        const c = this._labCase();
        const set = (id, v) => {
            const el = document.getElementById(id);
            if (el && v !== undefined && v !== null && v !== '') el.value = v;
        };
        set('carlaScenarioType', c.scenario_type);
        set('carlaSeed', c.seed);
        set('carlaStart', c.start_frame);
        set('carlaFrames', c.num_frames);
    }

    // ③ 闭环评估 → ④ CARLA 验证
    _labGoCarla() {
        const c = this._labCase();
        if (!c.scenario_type) {
            this._carlaStatus('提示：还没选定具体事故案例，CARLA 会直接跑当前场景里已有的轨迹（也可以先回 ② 生成一个）');
        }
        this._labSwitchTab('carla');
        this._labPrefillCarla();
    }

    _labSwitchTab(name) {'''

JS_SWITCH_OLD = '''        if (name === 'carla') { this.carlaRefreshStatus(1); this.carlaLoadScenarios(); this.carlaLoadJobs(); }'''
JS_SWITCH_NEW = '''        if (name === 'carla') { this.carlaRefreshStatus(1); this.carlaLoadScenarios(); this.carlaLoadJobs(); }
        // ④ 页面里每 10 秒刷一次状态（顺带盯显存），切走就停
        if (this._carlaTimer) { clearInterval(this._carlaTimer); this._carlaTimer = null; }
        if (name === 'carla') this._carlaTimer = setInterval(() => this.carlaRefreshStatus(0), 10000);
        this._labRenderFlow();'''

JS_BATCH_SAVE_OLD = '''            this._labRenderBatchSummary(data);
            this._labRenderBatchTable(data);'''
JS_BATCH_SAVE_NEW = '''            this._lastBatchManifest = data.manifest || [];
            this._lastBatchCtx = { start_frame: body.start_frame, num_frames: body.num_frames };
            // 批量结果顺手变成"当前案例"，后面 ③④ 步直接用
            const firstValid = (data.manifest || []).find(m => m.valid) || (data.manifest || [])[0];
            if (firstValid) {
                this._labSetCase({
                    scenario_type: firstValid.scenario_type || '', seed: firstValid.seed,
                    roles: firstValid.roles || null, start_frame: body.start_frame,
                    num_frames: body.num_frames, quality: firstValid.valid ? '通过' : '未通过'
                }, '批量生产');
            }
            this._labRenderBatchSummary(data);
            this._labRenderBatchTable(data);'''

JS_BATCH_HEAD_OLD = '''            <thead><tr><th>实例</th><th>类型</th><th>参与者</th><th>碰撞帧</th><th>反应帧</th><th>TTC</th><th>加速度</th><th>质量门</th><th>未通过项</th><th>视频</th></tr></thead>
            <tbody>${rows || '<tr><td colspan="10">无实例</td></tr>'}</tbody>'''
JS_BATCH_HEAD_NEW = '''            <thead><tr><th>实例</th><th>类型</th><th>参与者</th><th>碰撞帧</th><th>反应帧</th><th>TTC</th><th>加速度</th><th>质量门</th><th>未通过项</th><th>视频</th><th>下一步</th></tr></thead>
            <tbody>${rows || '<tr><td colspan="11">无实例</td></tr>'}</tbody>'''

JS_BATCH_ROW_OLD = '''                <td>${vid}</td>
            </tr>`;'''
JS_BATCH_ROW_NEW = '''                <td>${vid}</td>
                <td><button class="btn btn-secondary btn-small lab-batch-next" data-i="${this._escapeHtml(m.instance_id || '')}" data-j="sim">→ 闭环</button>
                    <button class="btn btn-primary btn-small lab-batch-next" data-i="${this._escapeHtml(m.instance_id || '')}" data-j="carla">→ CARLA</button></td>
            </tr>`;'''

JS_BATCH_BIND_OLD = '''        box.querySelectorAll('.lab-play-video').forEach(btn => {
            btn.addEventListener('click', () => this._labShowVideo(
                { ego: btn.dataset.path || null, bev: btn.dataset.bev || null, top: btn.dataset.top || null },
                btn.dataset.label, btn.dataset.kind || 'ego'));
        });'''
JS_BATCH_BIND_NEW = '''        box.querySelectorAll('.lab-play-video').forEach(btn => {
            btn.addEventListener('click', () => this._labShowVideo(
                { ego: btn.dataset.path || null, bev: btn.dataset.bev || null, top: btn.dataset.top || null },
                btn.dataset.label, btn.dataset.kind || 'ego'));
        });
        // 每一行都能直接"送去下一步"（带案例上下文）
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

JS_VRAM_OLD = '''            this._carlaFillMaps(d.maps);
            return d;'''
JS_VRAM_NEW = '''            const v = document.getElementById('carlaVram');
            if (v && d.gpu) {
                const g = d.gpu;
                v.textContent = `显存 ${(g.used_mb / 1024).toFixed(1)}/${(g.total_mb / 1024).toFixed(1)} GB`;
                v.style.color = g.free_mb < 7000 ? '#ef4444' : '#9ca3af';
                v.title = '空闲 ' + g.free_mb + ' MB\\n' +
                    (g.procs || []).map(p => `${p.name}  ${p.used_mb} MB`).join('\\n');
            }
            this._carlaFillMaps(d.maps);
            return d;'''

JS_RELEASE_OLD = '''    async carlaCancelJob() {'''
JS_RELEASE_NEW = r'''    async carlaRelease() {
        this._carlaStatus('正在释放显存（停 CARLA + 直播）…');
        try {
            const r = await fetch(`${API_BASE}/carla/release`, { method: 'POST' });
            const d = await r.json();
            const img = document.getElementById('carlaLiveImg');
            if (img) { img.removeAttribute('src'); img.alt = '直播已停止'; }
            await this.carlaRefreshStatus(1);
            const g = d.gpu || {};
            this._carlaStatus(`已释放 ✅ 现在空闲显存 ${g.free_mb ?? '?'} MB / ${g.total_mb ?? '?'} MB`
                + '（可以放心去用 SAM3D 了）');
        } catch (e) { this._carlaStatus('释放失败: ' + e.message, true); }
    }

    async carlaCancelJob() {'''

CSS_ADD = '''
/* ==================== 工作台流程条 / 显存 ==================== */
.lab-flow{display:flex;align-items:center;gap:8px;padding:6px 12px;margin:0 0 6px 0;
    background:rgba(255,255,255,.04);border-bottom:1px solid rgba(255,255,255,.07);flex-wrap:wrap}
.lab-flow .lab-flow-step{font-size:12px;font-weight:600;color:#9ca3af;background:rgba(255,255,255,.06);
    border-radius:999px;padding:2px 10px;flex:0 0 auto}
.lab-flow #labCaseCtx{font-size:12px;line-height:1.6}
.carla-vram{font-size:12px;color:#9ca3af;background:rgba(255,255,255,.05);border-radius:6px;
    padding:2px 8px;flex:0 0 auto;cursor:help;white-space:nowrap}
'''


def patch(path: Path, pairs, dry: bool):
    if not path.exists():
        print(f"!! 找不到 {path}")
        return False
    txt = path.read_text(encoding="utf-8")
    for i, (old, new) in enumerate(pairs, 1):
        n = txt.count(old)
        if n != 1:
            print(f"!! {path.name} 第 {i} 处锚点出现 {n} 次（期望 1）：{old[:70]!r}")
            return False
        if not dry:
            txt = txt.replace(old, new, 1)
    if not dry:
        if not Path(str(path) + ".bak2").exists():
            shutil.copy2(path, str(path) + ".bak2")
        path.write_text(txt, encoding="utf-8")
    print(f"{'✓ 校验' if dry else '✅ 已补丁'} {path.name}（{len(pairs)} 处）")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=str(DEFAULT_BASE))
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    base = Path(a.base)
    ok = True
    ok &= patch(base / "index.html", [
        (HTML_TABS_OLD, HTML_TABS_NEW),
        (HTML_SIM_OLD, HTML_SIM_NEW),
        (HTML_VRAM_OLD, HTML_VRAM_NEW),
        (HTML_NOTE_OLD, HTML_NOTE_NEW),
    ], a.dry_run)
    ok &= patch(base / "app.js", [
        (JS_BIND_OLD, JS_BIND_NEW),
        (JS_METHODS_OLD, JS_METHODS_NEW),
        (JS_SWITCH_OLD, JS_SWITCH_NEW),
        (JS_BATCH_SAVE_OLD, JS_BATCH_SAVE_NEW),
        (JS_BATCH_HEAD_OLD, JS_BATCH_HEAD_NEW),
        (JS_BATCH_ROW_OLD, JS_BATCH_ROW_NEW),
        (JS_BATCH_BIND_OLD, JS_BATCH_BIND_NEW),
        (JS_VRAM_OLD, JS_VRAM_NEW),
        (JS_RELEASE_OLD, JS_RELEASE_NEW),
    ], a.dry_run)
    css = base / "styles.css"
    if css.exists():
        c = css.read_text(encoding="utf-8")
        if ".lab-flow" in c:
            print("· styles.css 已有流程条样式，跳过")
        elif a.dry_run:
            print("✓ 校验 styles.css（待追加）")
        else:
            if not Path(str(css) + ".bak2").exists():
                shutil.copy2(css, str(css) + ".bak2")
            css.write_text(c + CSS_ADD, encoding="utf-8")
            print("✅ 已补丁 styles.css")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
