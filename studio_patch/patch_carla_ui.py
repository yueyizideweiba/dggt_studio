#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""给 studio 前端打上「⑥ CARLA 仿真」标签页（index.html / app.js / styles.css）。

每个替换都校验"只出现一次"，避免误改；改前留 .bak。
"""
import shutil
import sys
from pathlib import Path

FE = Path("/autodl-fs/data/dggt-main/studio/frontend")
HTML = FE / "index.html"
APP = FE / "app.js"
CSS = FE / "styles.css"

TAB_OLD = '''                <button class="lab-tab" data-tab="sim">⑤ 闭环仿真工具</button>'''
TAB_NEW = '''                <button class="lab-tab" data-tab="sim">⑤ 闭环仿真工具</button>
                <button class="lab-tab" data-tab="carla">⑥ CARLA 仿真</button>'''

PANEL_OLD = '''                    <div id="simDifixOut" class="lab-table-wrap"></div>
                </div>

                <div class="lab-panel" data-panel="selftest">'''

PANEL_NEW = r'''                    <div id="simDifixOut" class="lab-table-wrap"></div>
                </div>

                <!-- ⑥ CARLA 仿真：把 corner case 送进 CARLA 跑，出视频 / 看实时 / 导出标准格式 -->
                <div class="lab-panel" data-panel="carla">
                    <div class="lab-subtitle">CARLA 闭环仿真：corner case 场景 → CARLA 里跑（含碰撞事件）→ 出视频 / 看实时画面 / 导出标准 OpenSCENARIO</div>

                    <div class="lab-row">
                        <span id="carlaSvcDot" class="carla-dot off"></span>
                        <span id="carlaSvcText" class="cc-status" style="flex:1;min-width:220px">检查中…</span>
                        <label>画质</label>
                        <select id="carlaQuality" class="cc-select" style="width:90px">
                            <option>High</option><option>Low</option><option>Epic</option>
                        </select>
                        <label>地图</label>
                        <select id="carlaMap" class="cc-select" style="width:150px"></select>
                        <button id="carlaStatusBtn" class="btn btn-secondary btn-small">刷新状态</button>
                        <button id="carlaStartBtn" class="btn btn-primary btn-small">启动 CARLA</button>
                        <button id="carlaStopBtn" class="btn btn-secondary btn-small">停止</button>
                        <button id="carlaEnvBtn" class="btn btn-secondary btn-small">环境自检</button>
                    </div>

                    <div class="lab-subtitle" style="margin-top:10px;">1) 选场景</div>
                    <div class="lab-row">
                        <label class="cc-checkbox"><input type="checkbox" id="carlaUseCurrent" checked>用当前编辑的场景（含刚生成/编辑的事故轨迹）</label>
                        <select id="carlaSceneSel" class="cc-select" style="flex:1;min-width:280px"></select>
                        <button id="carlaSceneReloadBtn" class="btn btn-secondary btn-small">刷新场景列表</button>
                    </div>
                    <div class="lab-row">
                        <label>事故类型</label>
                        <input type="text" id="carlaScenarioType" placeholder="留空=直接导出当前场景" style="width:190px">
                        <label>seed</label><input type="number" id="carlaSeed" value="7" style="width:60px">
                        <label>起始帧</label><input type="number" id="carlaStart" value="0" style="width:60px">
                        <label>帧数</label><input type="number" id="carlaFrames" value="20" style="width:60px">
                        <label>fps</label><input type="number" id="carlaFps" value="20" style="width:60px">
                    </div>

                    <div class="lab-subtitle" style="margin-top:10px;">2) 仿真/相机参数</div>
                    <div class="lab-row">
                        <label>相机</label><input type="text" id="carlaCameras" value="chase,birdseye" style="width:140px">
                        <label>分辨率</label><input type="text" id="carlaCamSize" value="960x540" style="width:80px">
                        <label>跟随</label>
                        <select id="carlaFocus" class="cc-select" style="width:100px">
                            <option value="crash">事故双方</option><option value="ego">自车</option><option value="all">全部</option>
                        </select>
                        <label>参与者</label>
                        <select id="carlaActors" class="cc-select" style="width:110px">
                            <option value="dynamic">动态参与者</option><option value="core">仅核心(ego/双方)</option>
                            <option value="all">全部车/人</option><option value="raw">原始全部</option>
                        </select>
                        <label>天气</label>
                        <select id="carlaWeather" class="cc-select" style="width:120px">
                            <option>ClearNoon</option><option>CloudyNoon</option><option>ClearSunset</option>
                            <option>ClearNight</option><option>MidRainyNoon</option>
                        </select>
                        <label>ego</label>
                        <select id="carlaEgoMode" class="cc-select" style="width:120px">
                            <option value="playback">按轨迹回放</option><option value="autopilot">CARLA 自己开(闭环)</option>
                        </select>
                        <label class="cc-checkbox"><input type="checkbox" id="carlaExportXosc">顺带导出对齐版 xosc</label>
                    </div>

                    <div class="lab-row">
                        <button id="carlaRenderBtn" class="btn btn-primary btn-small">▶ 开始 CARLA 仿真并出视频</button>
                        <button id="carlaLiveBtn" class="btn btn-primary btn-small">🔴 开实时直播</button>
                        <button id="carlaLiveStopBtn" class="btn btn-secondary btn-small">停直播</button>
                        <button id="carlaXoscBtn" class="btn btn-secondary btn-small">只导出对齐版 xosc</button>
                        <button id="carlaSrBtn" class="btn btn-secondary btn-small">用 ScenarioRunner 跑(逻辑)</button>
                        <button id="carlaCancelBtn" class="btn btn-secondary btn-small">取消作业</button>
                    </div>
                    <div id="carlaStatusLine" class="cc-status">—</div>

                    <div class="lab-split">
                        <div class="lab-split-left">
                            <div class="lab-subtitle">实时画面（点「开实时直播」后出现，循环播放）</div>
                            <img id="carlaLiveImg" class="carla-live" alt="直播未启动">
                        </div>
                        <div class="lab-split-right">
                            <div class="lab-subtitle">作业日志（实时刷新）</div>
                            <pre id="carlaJobLog" class="lab-code carla-log">—</pre>
                        </div>
                    </div>

                    <div class="lab-subtitle" style="margin-top:10px;">3) 结果（最近一次作业）</div>
                    <div id="carlaResultInfo" class="lab-summary"></div>
                    <video id="carlaVideo" class="carla-video" controls preload="metadata" style="display:none"></video>
                    <div class="lab-row" id="carlaDownloadRow" style="margin-top:6px;"></div>
                    <div id="carlaMetrics" class="lab-table-wrap"></div>

                    <div class="lab-subtitle" style="margin-top:10px;">历史作业</div>
                    <div id="carlaJobs" class="lab-table-wrap"></div>

                    <div class="lab-subtitle" style="margin-top:10px;">说明</div>
                    <div class="cc-status">
                        ① 「用当前编辑的场景」= 把此刻场景里（含刚生成/编辑过的轨迹）导出成世界坐标 json 再送进 CARLA，
                        所以改完轨迹点「开始 CARLA 仿真」就能立刻看新效果，形成"编辑 → 仿真 → 再看 → 再编辑"的循环。<br>
                        ② 视频里每个交通参与者都标了 <b>角色 + 速度</b>，顶部 HUD 有 <code>minDist / minTTC / collisions</code>。<br>
                        ③ 实时直播是 CARLA 里正在跑的画面（后端已代理，不需要额外转发端口）。<br>
                        ④ 「用 ScenarioRunner 跑」= 先导出 CARLA 对齐版 <code>.xosc</code>，再用 CARLA 官方 ScenarioRunner
                        执行（逻辑/评测路径，不出画面）。
                    </div>
                </div>

                <div class="lab-panel" data-panel="selftest">'''

BIND_OLD = '''        bind('simAbRunBtn', () => this.simAbRun());
        bind('simAbReportBtn', () => this.simAbReport());'''
BIND_NEW = r'''        bind('simAbRunBtn', () => this.simAbRun());
        bind('simAbReportBtn', () => this.simAbReport());

        // ⑥ CARLA 仿真
        bind('carlaStatusBtn', () => this.carlaRefreshStatus(1));
        bind('carlaStartBtn', () => this.carlaServer('start'));
        bind('carlaStopBtn', () => this.carlaServer('stop'));
        bind('carlaEnvBtn', () => this.carlaEnvCheck());
        bind('carlaSceneReloadBtn', () => this.carlaLoadScenarios());
        bind('carlaRenderBtn', () => this.carlaRender());
        bind('carlaLiveBtn', () => this.carlaLive('start'));
        bind('carlaLiveStopBtn', () => this.carlaLive('stop'));
        bind('carlaXoscBtn', () => this.carlaExportXosc());
        bind('carlaSrBtn', () => this.carlaScenarioRunner());
        bind('carlaCancelBtn', () => this.carlaCancelJob());'''

SWITCH_OLD = '''        if (name === 'graph' && this.state.sceneId) this.labLoadGraph();'''
SWITCH_NEW = '''        if (name === 'graph' && this.state.sceneId) this.labLoadGraph();
        if (name === 'carla') { this.carlaRefreshStatus(1); this.carlaLoadScenarios(); this.carlaLoadJobs(); }'''

METHODS_OLD = '''    _labSwitchTab(name) {'''
METHODS_NEW = r'''    // ==================== ⑥ CARLA 仿真（视频 / 实时直播 / 标准导出） ====================
    _carlaState() {
        if (!this._carla) this._carla = { jobId: null, timer: null, live: false, status: null, afterJob: null };
        return this._carla;
    }

    _carlaStatus(msg, err) {
        const el = document.getElementById('carlaStatusLine');
        if (el) { el.textContent = msg; el.style.color = err ? '#e5484d' : 'inherit'; }
    }

    _carlaVal(id) { const el = document.getElementById(id); return el ? el.value : ''; }
    _carlaChk(id) { const el = document.getElementById(id); return el ? el.checked : false; }

    _carlaParams() {
        const useCurrent = this._carlaChk('carlaUseCurrent') || !this._carlaVal('carlaSceneSel');
        const out = {
            map: this._carlaVal('carlaMap') || 'Town10HD_Opt',
            fps: parseFloat(this._carlaVal('carlaFps') || '20'),
            cameras: this._carlaVal('carlaCameras') || 'chase,birdseye',
            cam_size: this._carlaVal('carlaCamSize') || '960x540',
            focus: this._carlaVal('carlaFocus') || 'crash',
            actors: this._carlaVal('carlaActors') || 'dynamic',
            weather: this._carlaVal('carlaWeather') || 'ClearNoon',
            ego_mode: this._carlaVal('carlaEgoMode') || 'playback',
            seed: parseInt(this._carlaVal('carlaSeed') || '7', 10),
            start_frame: parseInt(this._carlaVal('carlaStart') || '0', 10),
            num_frames: parseInt(this._carlaVal('carlaFrames') || '20', 10),
            export_xosc: this._carlaChk('carlaExportXosc'),
        };
        const stype = (this._carlaVal('carlaScenarioType') || '').trim();
        if (stype) out.scenario_type = stype;
        if (useCurrent) {
            if (!this.state.sceneId) {
                throw new Error('没有已加载的场景：先「加载场景」，或取消勾选「用当前编辑的场景」再从下拉里选一个已导出场景');
            }
            out.scene_id = this.state.sceneId;
        } else {
            out.scenario = this._carlaVal('carlaSceneSel');
        }
        return out;
    }

    _carlaFillMaps(maps) {
        const sel = document.getElementById('carlaMap');
        if (!sel || sel.dataset.filled === '1' || !maps || !maps.length) return;
        sel.innerHTML = maps.map(m => `<option value="${m}"${m === 'Town10HD_Opt' ? ' selected' : ''}>${m}</option>`).join('');
        sel.dataset.filled = '1';
    }

    async carlaRefreshStatus(deep) {
        try {
            const r = await fetch(`${API_BASE}/carla/status?deep=${deep ? 1 : 0}`);
            const d = await r.json();
            this._carlaState().status = d;
            const dot = document.getElementById('carlaSvcDot');
            if (dot) dot.className = 'carla-dot ' + (d.ready ? 'on' : 'off');
            const txt = document.getElementById('carlaSvcText');
            if (txt) {
                txt.textContent = d.ready
                    ? `CARLA 运行中（RPC ${d.port}${d.map ? '，地图 ' + d.map : ''}${d.live ? '，直播中' : ''}）`
                    : (d.running ? 'CARLA 进程在但端口没通（可能还在加载，或已经卡住 → 点「停止」再启动）'
                                 : 'CARLA 未启动（点左边「启动 CARLA」）');
            }
            this._carlaFillMaps(d.maps);
            return d;
        } catch (e) {
            this._carlaStatus('状态查询失败: ' + e.message, true);
            return null;
        }
    }

    async carlaServer(action) {
        const body = { action, quality: this._carlaVal('carlaQuality') || 'High',
                       map: this._carlaVal('carlaMap') || 'Town10HD_Opt' };
        this._carlaStatus(action === 'start' ? '正在启动 CARLA（首次约 20-60 秒）…' : '正在停止…');
        try {
            const r = await fetch(`${API_BASE}/carla/server`, {
                method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body)
            });
            const d = await r.json();
            await this.carlaRefreshStatus(1);
            const ok = d.success !== false;
            this._carlaStatus(ok ? (action === 'start' ? 'CARLA 已就绪 ✅' : '已停止') :
                ('失败：' + (d.detail || '请看服务日志')), !ok);
        } catch (e) { this._carlaStatus('操作失败: ' + e.message, true); }
    }

    async carlaEnvCheck() {
        this._carlaStatus('环境自检中…');
        try {
            const r = await fetch(`${API_BASE}/carla/env`);
            const d = await r.json();
            const ok = d.carla_installed && d.python37_ok;
            const lines = [
                `CARLA 安装：${d.carla_installed ? '✅' : '❌'}  ${d.carla_home}`,
                `Python3.7 API：${d.python37_ok ? '✅' : '❌'}  ${d.python37}`,
                `脚本：${Object.entries(d.scripts || {}).map(([k, v]) => k + (v ? '✅' : '❌')).join('  ')}`,
                `ScenarioRunner：${d.scenario_runner ? '✅' : '❌'}    磁盘剩余 ${d.disk_free_gb} GB`,
                `Vulkan：${(d.vulkan || []).join(' | ') || '未检测到'}`,
            ];
            const box = document.getElementById('carlaMetrics');
            if (box) box.innerHTML = `<table class="lab-table"><tbody>${lines.map(l => `<tr><td>${l}</td></tr>`).join('')}</tbody></table>`;
            this._carlaStatus(ok ? '环境自检通过 ✅' : '环境不完整（见下表）', !ok);
        } catch (e) { this._carlaStatus('自检失败: ' + e.message, true); }
    }

    async carlaLoadScenarios() {
        try {
            const r = await fetch(`${API_BASE}/carla/scenarios`);
            const d = await r.json();
            const sel = document.getElementById('carlaSceneSel');
            if (sel) {
                const items = (d.items || []).filter(i => i.kind !== 'video');
                sel.innerHTML = items.map(i =>
                    `<option value="${i.path}">${i.path}  ·  ${i.num_tracks || '?'} 物体 / ${i.num_dynamic || 0} 动态`
                    + `${(i.roles || []).length ? ' · ' + i.roles.join(',') : ''}`
                    + `${i.rendered_video ? ' · 已出片' : ''}</option>`).join('');
            }
            this._carlaStatus(`可跑场景 ${d.count} 个`);
        } catch (e) { this._carlaStatus('场景列表失败: ' + e.message, true); }
    }

    async carlaRender() {
        let p;
        try { p = this._carlaParams(); } catch (e) { this._carlaStatus(e.message, true); return; }
        this._carlaStatus('提交 CARLA 渲染作业…');
        try {
            const r = await fetch(`${API_BASE}/carla/render`, {
                method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(p)
            });
            const d = await r.json();
            if (d.detail) throw new Error(d.detail);
            this._carlaState().jobId = d.job.id;
            this._carlaStatus(`作业 ${d.job.id} 已开始：${d.job.title}`);
            this._carlaPoll();
        } catch (e) { this._carlaStatus('提交失败: ' + e.message, true); }
    }

    _carlaPoll() {
        const s = this._carlaState();
        if (s.timer) clearTimeout(s.timer);
        if (!s.jobId) return;
        const tick = async () => {
            let j;
            try {
                const r = await fetch(`${API_BASE}/carla/jobs/${s.jobId}?tail=150`);
                j = await r.json();
            } catch (e) { this._carlaStatus('轮询失败: ' + e.message, true); return; }
            const log = document.getElementById('carlaJobLog');
            if (log) { log.textContent = (j.log_tail || []).join('\n'); log.scrollTop = log.scrollHeight; }
            if (j.status === 'running') {
                this._carlaStatus(`作业跑中… ${j.seconds}s（日志在右边实时刷）`);
                s.timer = setTimeout(tick, 2000);
                return;
            }
            const ok = j.status === 'done';
            this._carlaStatus(ok ? `作业完成 ✅（${j.seconds}s）` : `作业 ${j.status} ❌ rc=${j.rc}`, !ok);
            this._carlaShowResult(j);
            if (s.afterJob === 'sr' && j.artifacts && j.artifacts.xosc) {
                s.afterJob = null;
                this._carlaRunSr(j.artifacts.xosc);
            }
            this.carlaRefreshStatus();
        };
        tick();
    }

    _carlaFileUrl(p) { return `${API_BASE}/carla/file?path=${encodeURIComponent(p)}`; }
    _carlaDownloadUrl(p) { return `${API_BASE}/carla/file?path=${encodeURIComponent(p)}&download=1`; }

    async _carlaShowResult(j) {
        const vids = ((j.artifacts || {}).videos) || [];
        const vid = document.getElementById('carlaVideo');
        if (vid && vids.length) { vid.src = this._carlaFileUrl(vids[0]); vid.style.display = ''; }
        const dl = document.getElementById('carlaDownloadRow');
        if (dl) {
            const items = [];
            vids.forEach(v => items.push(`<a class="btn btn-secondary btn-small" href="${this._carlaDownloadUrl(v)}">⬇ ${v.split('/').pop()}</a>`));
            ['events_json', 'frames_json', 'xosc'].forEach(k => {
                if ((j.artifacts || {})[k]) {
                    items.push(`<a class="btn btn-secondary btn-small" href="${this._carlaDownloadUrl(j.artifacts[k])}">⬇ ${j.artifacts[k].split('/').pop()}</a>`);
                }
            });
            dl.innerHTML = items.join(' ');
        }
        const box = document.getElementById('carlaResultInfo');
        if (box) box.innerHTML = `<div class="cc-status">产物目录：<code>${j.out_dir || '-'}</code></div>`;
        const mbox = document.getElementById('carlaMetrics');
        const ev = (j.artifacts || {}).events_json;
        if (ev && mbox) {
            try {
                const r = await fetch(this._carlaFileUrl(ev));
                const e = await r.json();
                const rows = [
                    ['帧数', e.frames], ['actor 数', e.actors], ['地图', e.map],
                    ['碰撞事件', (e.collisions || []).length],
                    ['全程最小距离(m)', e.min_dist_overall], ['最小 TTC(s)', e.min_ttc_overall],
                    ['相机跟随', `${e.focus} → ${JSON.stringify(e.focus_ids)}`],
                ];
                let html = `<table class="lab-table"><thead><tr><th>指标</th><th>值</th></tr></thead><tbody>`
                    + rows.map(([k, v]) => `<tr><td>${k}</td><td>${v === null || v === undefined ? '-' : v}</td></tr>`).join('')
                    + `</tbody></table>`;
                const cs = e.collisions || [];
                if (cs.length) {
                    html += `<table class="lab-table"><thead><tr><th>碰撞帧</th><th>t(s)</th><th>actor</th><th>对方</th><th>冲量</th></tr></thead><tbody>`
                        + cs.slice(0, 12).map(c => `<tr><td>${c.frame}</td><td>${c.t}</td><td>${c.actor}</td>`
                            + `<td>${c.other_type || c.other_actor || '-'}</td><td>${c.impulse}</td></tr>`).join('')
                        + `</tbody></table>`;
                }
                mbox.innerHTML = html;
            } catch (e) { mbox.innerHTML = `<div class="cc-status">指标读取失败：${e.message}</div>`; }
        }
        this.carlaLoadJobs();
    }

    async carlaLoadJobs() {
        try {
            const r = await fetch(`${API_BASE}/carla/jobs`);
            const d = await r.json();
            const box = document.getElementById('carlaJobs');
            if (!box) return;
            const rows = (d.jobs || []).map(j => {
                const v = ((j.artifacts || {}).videos || [])[0];
                return [j.kind, j.title, j.status, j.seconds + 's',
                        v ? `<a href="${this._carlaFileUrl(v)}" target="_blank">看视频</a>` : '-'];
            });
            box.innerHTML = this._simRows(['类型', '说明', '状态', '用时', '产物'], rows);
        } catch (e) { /* 忽略 */ }
    }

    async carlaLive(action) {
        if (action === 'stop') {
            try {
                await fetch(`${API_BASE}/carla/live`, {
                    method: 'POST', headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ action: 'stop' })
                });
                const img = document.getElementById('carlaLiveImg');
                if (img) { img.removeAttribute('src'); img.alt = '直播已停止'; }
                this._carlaState().live = false;
                this._carlaStatus('直播已停止');
            } catch (e) { this._carlaStatus('停止失败: ' + e.message, true); }
            return;
        }
        let p;
        try { p = this._carlaParams(); } catch (e) { this._carlaStatus(e.message, true); return; }
        this._carlaStatus('正在启动直播（要等地图加载 + 预热，约 30-60 秒）…');
        try {
            const r = await fetch(`${API_BASE}/carla/live`, {
                method: 'POST', headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(Object.assign({ action: 'start' }, p))
            });
            const d = await r.json();
            if (!d.success) throw new Error(d.detail || '启动失败');
            const img = document.getElementById('carlaLiveImg');
            if (img) { img.alt = 'CARLA 实时画面'; img.src = `${API_BASE}/carla/live/mjpeg?t=${Date.now()}`; }
            this._carlaState().live = true;
            this._carlaStatus('直播中：左边就是 CARLA 里正在跑的场景（循环播放）');
        } catch (e) { this._carlaStatus('直播启动失败: ' + e.message, true); }
    }

    async carlaExportXosc() {
        let p;
        try { p = this._carlaParams(); } catch (e) { this._carlaStatus(e.message, true); return; }
        const body = { map: p.map };
        if (p.scenario) body.scenario = p.scenario;
        else if (p.scene_id) {
            this._carlaStatus('导出对齐版 xosc 需要一个已落盘的场景：请取消勾选「用当前编辑的场景」，从下拉里选一个（或先「开始 CARLA 仿真」会自动导出）', true);
            return;
        }
        this._carlaStatus('导出 CARLA 对齐版 xosc…');
        this._carlaState().afterJob = null;
        try {
            const r = await fetch(`${API_BASE}/carla/export_xosc`, {
                method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body)
            });
            const d = await r.json();
            if (d.detail) throw new Error(d.detail);
            this._carlaState().jobId = d.job.id;
            this._carlaPoll();
        } catch (e) { this._carlaStatus('导出失败: ' + e.message, true); }
    }

    // 先导出对齐版 xosc，再拿它的产物路径去跑 ScenarioRunner
    async carlaScenarioRunner() {
        const p = this._carlaParams();
        if (p.scene_id) {
            this._carlaStatus('ScenarioRunner 需要一个已落盘的场景：请取消勾选「用当前编辑的场景」并从下拉里选一个', true);
            return;
        }
        this._carlaStatus('Step1/2：先导出 CARLA 对齐版 xosc…');
        this._carlaState().afterJob = 'sr';
        await this.carlaExportXosc();
    }

    async _carlaRunSr(xosc) {
        this._carlaStatus('Step2/2：用 ScenarioRunner 跑 ' + xosc + ' …');
        try {
            const r = await fetch(`${API_BASE}/carla/scenario_runner`, {
                method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ xosc })
            });
            const d = await r.json();
            if (d.detail) throw new Error(d.detail);
            this._carlaState().jobId = d.job.id;
            this._carlaPoll();
        } catch (e) { this._carlaStatus('ScenarioRunner 失败: ' + e.message, true); }
    }

    async carlaCancelJob() {
        const s = this._carlaState();
        if (!s.jobId) { this._carlaStatus('当前没有跟踪中的作业'); return; }
        try {
            await fetch(`${API_BASE}/carla/jobs/${s.jobId}/cancel`, { method: 'POST' });
            this._carlaStatus('已请求取消作业 ' + s.jobId);
        } catch (e) { this._carlaStatus('取消失败: ' + e.message, true); }
    }

    _labSwitchTab(name) {'''

CSS_ADD = '''
/* ==================== ⑥ CARLA 仿真 ==================== */
.carla-dot{width:10px;height:10px;border-radius:50%;display:inline-block;flex:0 0 auto;
    background:#6b7280;box-shadow:0 0 0 rgba(0,0,0,0)}
.carla-dot.on{background:#22c55e;box-shadow:0 0 8px rgba(34,197,94,.8)}
.carla-dot.off{background:#ef4444;box-shadow:0 0 8px rgba(239,68,68,.5)}
.carla-live{width:100%;max-width:560px;min-height:150px;border-radius:8px;background:#0b0f14;
    border:1px solid rgba(255,255,255,.08);object-fit:contain}
.carla-video{width:100%;max-width:1000px;margin-top:6px;border-radius:8px;background:#000}
.carla-log{max-height:300px;overflow:auto;white-space:pre-wrap;word-break:break-all;
    background:#0b0f14;border-radius:8px;padding:8px;font-size:12px;line-height:1.45}
'''


def patch(path: Path, pairs):
    txt = path.read_text(encoding="utf-8")
    for old, new in pairs:
        n = txt.count(old)
        if n != 1:
            print(f"!! {path.name}: 锚点出现 {n} 次（期望 1），中止。片段={old[:60]!r}")
            return False
        txt = txt.replace(old, new, 1)
    shutil.copy2(path, str(path) + ".bak")
    path.write_text(txt, encoding="utf-8")
    print(f"✅ {path.name} 已打补丁（备份 .bak）")
    return True


ok = True
ok &= patch(HTML, [(TAB_OLD, TAB_NEW), (PANEL_OLD, PANEL_NEW)])
ok &= patch(APP, [(BIND_OLD, BIND_NEW), (SWITCH_OLD, SWITCH_NEW), (METHODS_OLD, METHODS_NEW)])
css = CSS.read_text(encoding="utf-8")
if ".carla-live" not in css:
    shutil.copy2(CSS, str(CSS) + ".bak")
    CSS.write_text(css + CSS_ADD, encoding="utf-8")
    print("✅ styles.css 已追加 CARLA 样式")
else:
    print("· styles.css 已有 CARLA 样式，跳过")
sys.exit(0 if ok else 1)
