#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""第三批前端改造：
  1) 删掉 ⑤ 训练语料 / ⑥ 自检诊断
  2) ③ 改名「渲染可信域评估」、④ 改名「CARLA 闭环仿真」并写清两者关系
  3) 扩散精修（Difix）挪到 ③ 面板最前面 + 加"对当前帧精修"按钮（不用跑一键自检也能用）
  4) 「实时直播」整块换成「🎮 CARLA 仿真引擎」：播放 / 时间轴 / 选车编辑 / 加车 / 存成场景
用法: python patch_flow3.py [--base DIR] [--dry-run]
"""
import argparse
import shutil
import sys
from pathlib import Path

DEFAULT_BASE = Path("/autodl-fs/data/dggt-main/studio/frontend")

# =========================== index.html ===========================
H1_OLD = '''                <button class="lab-tab" data-tab="sim">③ 闭环评估（可信域/资产库）</button>
                <button class="lab-tab" data-tab="carla">④ CARLA 验证（视频/实时）</button>
                <button class="lab-tab" data-tab="gnn">⑤ 训练语料</button>
                <button class="lab-tab" data-tab="selftest">⑥ 自检诊断</button>'''
H1_NEW = '''                <button class="lab-tab" data-tab="sim">③ 渲染可信域评估（4DGS）</button>
                <button class="lab-tab" data-tab="carla">④ CARLA 闭环仿真（可编辑/播放）</button>'''

H2_OLD = '''                <!-- ③ 模型训练 -->
                <div class="lab-panel" data-panel="gnn">
                    <div class="lab-subtitle">收集"生成前关系图特征 → 生成后(是否碰撞, 严重度)"的带标签样本</div>
                    <div class="lab-row">
                        <label>每类样本数</label><input type="number" id="labGnnNum" value="20" min="1">
                        <label>随机种子</label><input type="number" id="labGnnSeed" value="0">
                        <button id="labGnnCollectBtn" class="btn btn-primary btn-small">收集训练语料</button>
                    </div>
                    <div id="labGnnStatus" class="cc-status">样本存成 .npz，供离线训练碰撞预测 GNN。</div>
                    <div id="labGnnSummary" class="lab-summary"></div>
                    <div class="lab-subtitle">离线训练命令（在服务器终端执行）</div>
                    <pre id="labGnnCmd" class="lab-code">—</pre>
                </div>

'''
H2_NEW = ''''''

H3_OLD = '''                <div class="lab-panel" data-panel="selftest">
                    <div class="lab-row">
                        <button id="labSelfTestBtn" class="btn btn-primary">运行自检</button>
                        <span class="cc-status">依次验证：关系图 / 生成质量门 / 批量生产 / 训练语料</span>
                    </div>
                    <div id="labSelfTestOut" class="lab-selftest">
                        <div class="empty-state"><p>点「运行自检」开始</p></div>
                    </div>
                </div>
'''
H3_NEW = ''''''

# Difix：从尾部删掉
H4_OLD = '''                    <div class="lab-subtitle" style="margin-top:12px;">扩散渲染精修（Difix）+ 俯视 3D 取景</div>
                    <div class="lab-row">
                        <button id="simDifixStatusBtn" class="btn btn-secondary btn-small">检查扩散模型依赖</button>
                        <label>俯视视野(m)</label><input type="number" id="simTdSpan" value="0" min="0" step="2" style="width:70px">
                        <span class="detail">0=自动；调小（如 20）能把车/行人放大看清</span>
                    </div>
                    <div id="simDifixOut" class="lab-table-wrap"></div>
'''
H4_NEW = ''''''

# Difix：插到 ③ 面板最前面（"可信域"小节之前）
H5_OLD = '''                    <div class="lab-subtitle">可信域（novel-view trust）：标定 → 扫描 → 逐帧判定</div>'''
H5_NEW = '''                    <div class="lab-subtitle">★ 扩散渲染精修（Difix）：把渲染画面精修一遍（独立可用，不必先跑一键自检）</div>
                    <div class="lab-row">
                        <button id="simDifixStatusBtn" class="btn btn-secondary btn-small">检查扩散模型依赖</button>
                        <label>帧</label><input type="number" id="simDifixFrame" value="0" min="0" style="width:70px">
                        <button id="simDifixRefineBtn" class="btn btn-primary btn-small">✨ 对当前帧做扩散精修</button>
                        <label>俯视视野(m)</label><input type="number" id="simTdSpan" value="0" min="0" step="2" style="width:70px">
                        <span class="detail">精修前 / 精修后 并排显示；首次会加载扩散模型（较慢）</span>
                    </div>
                    <div id="simDifixImages" class="lab-row" style="gap:12px;align-items:flex-start"></div>
                    <div id="simDifixOut" class="lab-table-wrap"></div>

                    <div class="lab-subtitle" style="margin-top:6px;">可信域（novel-view trust）：标定 → 扫描 → 逐帧判定</div>'''

H6_OLD = '''                        <button id="carlaRenderBtn" class="btn btn-primary btn-small">▶ 开始 CARLA 仿真并出视频</button>
                        <button id="carlaLiveBtn" class="btn btn-primary btn-small">🔴 开实时直播</button>
                        <button id="carlaLiveStopBtn" class="btn btn-secondary btn-small">停直播</button>
'''
H6_NEW = '''                        <button id="carlaRenderBtn" class="btn btn-primary btn-small">▶ 开始 CARLA 仿真并出视频</button>
'''

H7_OLD = '''                    <div class="lab-split">
                        <div class="lab-split-left">
                            <div class="lab-subtitle">实时画面（点「开实时直播」后出现，循环播放）</div>
                            <img id="carlaLiveImg" class="carla-live" alt="直播未启动">
                        </div>
                        <div class="lab-split-right">
                            <div class="lab-subtitle">作业日志（实时刷新）</div>
                            <pre id="carlaJobLog" class="lab-code carla-log">—</pre>
                        </div>
                    </div>
'''
H7_NEW = '''                    <div class="lab-subtitle" style="margin-top:10px;">🎮 CARLA 仿真引擎（事故仿真把手：可播放 + 可编辑）</div>
                    <div class="lab-row">
                        <button id="engStartBtn" class="btn btn-primary btn-small">🎮 启动仿真引擎</button>
                        <button id="engStopBtn" class="btn btn-secondary btn-small">停止引擎</button>
                        <span id="engStatus" class="cc-status" style="flex:1">未启动：点左边按钮，会用「1) 选的场景」在 CARLA 里开一个可交互会话（约 40-90 秒）</span>
                    </div>
                    <div class="lab-split">
                        <div class="lab-split-left">
                            <img id="engImg" class="carla-live" alt="引擎未启动">
                            <div class="lab-row">
                                <button id="engPlayBtn" class="btn btn-primary btn-small">⏸ 暂停</button>
                                <button id="engPrevBtn" class="btn btn-secondary btn-small">⟲ 上一帧</button>
                                <button id="engNextBtn" class="btn btn-secondary btn-small">⟳ 下一帧</button>
                                <label>速度</label>
                                <select id="engFps" class="cc-select" style="width:70px">
                                    <option>10</option><option selected>20</option><option>30</option>
                                </select>
                                <label>相机</label>
                                <select id="engCam" class="cc-select" style="width:100px">
                                    <option value="chase">追逐</option><option value="birdseye">俯视</option>
                                    <option value="front">车头</option><option value="side">侧视</option>
                                </select>
                                <label>天气</label>
                                <select id="engWeather" class="cc-select" style="width:110px">
                                    <option>ClearNoon</option><option>CloudyNoon</option><option>ClearSunset</option>
                                    <option>ClearNight</option><option>MidRainyNoon</option>
                                </select>
                            </div>
                            <div class="lab-row">
                                <label>时间轴</label>
                                <input type="range" id="engSlider" min="0" max="0" value="0" style="flex:1;min-width:220px">
                                <span id="engFrameLabel" class="detail">0 / 0</span>
                            </div>
                        </div>
                        <div class="lab-split-right">
                            <div class="lab-subtitle">场景编辑（用下面按钮改选中行的位置/朝向）</div>
                            <div class="lab-row">
                                <label>步长(m)</label>
                                <select id="engStep" class="cc-select" style="width:70px">
                                    <option>0.5</option><option selected>1</option><option>2</option><option>5</option>
                                </select>
                                <button id="engResetBtn" class="btn btn-secondary btn-small">重置全部编辑</button>
                                <button id="engSaveBtn" class="btn btn-primary btn-small">💾 存成场景</button>
                                <button id="engShotBtn" class="btn btn-secondary btn-small">📷 截图</button>
                            </div>
                            <div id="engActors" class="lab-table-wrap"></div>
                            <div class="lab-row">
                                <label>新增车辆</label>
                                <input type="number" id="engAddX" placeholder="x" style="width:78px">
                                <input type="number" id="engAddY" placeholder="y" style="width:78px">
                                <input type="number" id="engAddYaw" placeholder="yaw°" value="0" style="width:66px">
                                <input type="number" id="engAddSpeed" placeholder="m/s" value="6" style="width:62px">
                                <button id="engAddBtn" class="btn btn-primary btn-small">+ 加车</button>
                            </div>
                            <div id="engSaved" class="cc-status"></div>
                        </div>
                    </div>
                    <div class="lab-subtitle" style="margin-top:8px;">作业日志（渲染 / 导出作业）</div>
                    <pre id="carlaJobLog" class="lab-code carla-log">—</pre>
'''

H8_OLD = '''                    <div class="cc-status" style="margin-bottom:6px">
                        这是 <b>④ CARLA 验证</b>（③ 闭环评估是另一条独立通道，不必须先做 ③）。
                        两种送法：<b>用当前编辑的场景</b>（含刚生成/编辑的轨迹，现场导出）或
                        <b>从①批量生成里点「🎬 CARLA 这条」</b>（直接吃该实例的 world.json，不依赖内存场景，最稳）。
                    </div>'''
H8_NEW = '''                    <div class="cc-status" style="margin-bottom:6px">
                        <b>④ CARLA 闭环仿真</b>：CARLA 本身就是物理闭环仿真器（自车在环、有物理和碰撞），
                        "闭环"说的就是它；③ 是<b>渲染层的可信度评估</b>（你们 4DGS 渲染出来的画面可不可信），
                        <b>不是仿真器</b>，两者互相独立、不分先后。<br>
                        送场景两种方式：<b>用当前编辑的场景</b>（现场导出）或
                        <b>从 ① 批量生成里点「🎬 CARLA 这条」</b>（直接吃该实例的 world.json，不依赖内存场景，最稳）。<br>
                        想边看边改就用下面的 <b>🎮 仿真引擎</b>：播放/暂停、拖时间轴、选中车辆平移旋转删除、加车、存成新场景。
                    </div>'''

# =========================== app.js ===========================
J1_OLD = '''    _labSteps() { return ['graph', 'batch', 'sim', 'carla', 'gnn', 'selftest']; }
    _labStepNames() {
        return { graph: '① 挑事故', batch: '② 批量生成', sim: '③ 闭环评估',
                 carla: '④ CARLA 验证', gnn: '⑤ 训练语料', selftest: '⑥ 自检诊断' };
    }'''
J1_NEW = '''    _labSteps() { return ['batch', 'graph', 'sim', 'carla']; }
    _labStepNames() {
        return { batch: '① 批量生成', graph: '② 挑事故', sim: '③ 渲染可信域评估',
                 carla: '④ CARLA 闭环仿真' };
    }'''

J2_OLD = '''        bind('carlaLiveBtn', () => this.carlaLive('start'));
        bind('carlaLiveStopBtn', () => this.carlaLive('stop'));'''
J2_NEW = '''        bind('engStartBtn', () => this.carlaEngineStart());
        bind('engStopBtn', () => this.carlaEngineStop());
        bind('engPlayBtn', () => this.carlaEnginePlay());
        bind('engPrevBtn', () => this.carlaEngineFrame(-1));
        bind('engNextBtn', () => this.carlaEngineFrame(1));
        bind('engResetBtn', () => this.carlaEngine('reset', {}).then(() => this.carlaEngineRefresh(false)));
        bind('engSaveBtn', () => this.carlaEngineSave());
        bind('engShotBtn', () => this.carlaEngineShot());
        bind('engAddBtn', () => {
            const g = id => parseFloat(document.getElementById(id)?.value || '0');
            this.carlaEngine('actor/add', { x: g('engAddX'), y: g('engAddY'),
                                            yaw: g('engAddYaw'), speed: g('engAddSpeed') })
                .then(() => this.carlaEngineRefresh(false));
        });
        const _engCam = document.getElementById('engCam');
        if (_engCam) _engCam.addEventListener('change', () => this.carlaEngine('camera', { camera: _engCam.value }));
        const _engW = document.getElementById('engWeather');
        if (_engW) _engW.addEventListener('change', () => this.carlaEngine('weather', { weather: _engW.value }));
        const _engF = document.getElementById('engFps');
        if (_engF) _engF.addEventListener('change', () => this.carlaEngine('fps', { fps: parseFloat(_engF.value) }));
        const _engS = document.getElementById('engSlider');
        if (_engS) _engS.addEventListener('change', () => this.carlaEngine('frame', { index: parseInt(_engS.value, 10) })
            .then(() => this.carlaEngineRefresh(false)));
        bind('simDifixRefineBtn', () => this.simDifixRefine());'''

J3_OLD = '''    async carlaLive(action) {
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
    }'''

J3_NEW = r'''    // ==================== 🎮 CARLA 仿真引擎（可播放 + 可编辑）====================
    _eng() {
        if (!this._engine) this._engine = { timer: null, st: null };
        return this._engine;
    }

    _engStatus(msg, err) {
        const el = document.getElementById('engStatus');
        if (el) { el.textContent = msg; el.style.color = err ? '#e5484d' : 'inherit'; }
    }

    async carlaEngine(action, body) {
        const r = await fetch(`${API_BASE}/carla/engine/${action}`, {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body || {})
        });
        const d = await r.json();
        if (d && d.detail && d.success === undefined) throw new Error(d.detail);
        return d;
    }

    async carlaEngineStart() {
        let p;
        try { p = this._carlaParams(); } catch (e) { this._engStatus(e.message, true); return; }
        this._engStatus('正在启动仿真引擎（起 CARLA + 生成场景 + 预热，约 40-90 秒）…');
        try {
            const d = await this.carlaEngine('start', p);
            if (!d.success) throw new Error(d.detail || '启动失败');
            this._engStatus('引擎已就绪：可播放 / 拖时间轴 / 选中车辆编辑 / 存成场景');
            const img = document.getElementById('engImg');
            if (img) img.src = `${API_BASE}/carla/engine/mjpeg?t=${Date.now()}`;
            await this.carlaEngineRefresh(true);
            const s = this._eng();
            if (s.timer) clearInterval(s.timer);
            s.timer = setInterval(() => this.carlaEngineRefresh(false), 700);
        } catch (e) { this._engStatus('启动失败: ' + e.message, true); }
    }

    async carlaEngineStop() {
        try { await this.carlaEngine('stop', {}); } catch (e) { /* 忽略 */ }
        const s = this._eng();
        if (s.timer) { clearInterval(s.timer); s.timer = null; }
        const img = document.getElementById('engImg');
        if (img) img.removeAttribute('src');
        this._engStatus('引擎已停止（显存约 1 分钟后自动释放，也可点「🧹 释放显存」）');
    }

    async carlaEngineRefresh(force) {
        try {
            const r = await fetch(`${API_BASE}/carla/engine/state`);
            const st = await r.json();
            if (st.detail) throw new Error(st.detail);
            this._eng().st = st;
            this._engRender(st);
        } catch (e) {
            if (force) this._engStatus('引擎未在运行: ' + e.message, true);
        }
    }

    _engRender(st) {
        const sl = document.getElementById('engSlider');
        const lb = document.getElementById('engFrameLabel');
        if (sl) { sl.max = Math.max(0, st.n_frames - 1); sl.value = st.frame; }
        if (lb) lb.textContent = `${st.frame} / ${st.n_frames - 1}  ·  t=${(st.frame / st.fps).toFixed(2)}s`;
        const pb = document.getElementById('engPlayBtn');
        if (pb) pb.textContent = st.play ? '⏸ 暂停' : '▶ 播放';
        const cam = document.getElementById('engCam');
        if (cam && st.camera) cam.value = st.camera;
        const w = document.getElementById('engWeather');
        if (w && st.weather) w.value = st.weather;
        const box = document.getElementById('engActors');
        if (!box) return;
        const rows = (st.actors || []).map(a => {
            const off = a.offset ? `Δ ${a.offset.dx || 0} / ${a.offset.dy || 0} m / ${a.offset.dyaw || 0}°` : '';
            const loc = a.loc ? `${a.loc[0]}, ${a.loc[1]}` : '-';
            return `<tr>
                <td>${a.is_ego ? '🚗 ' : ''}${this._escapeHtml(String(a.role))}#${a.tid}${a.extra ? ' <b>(新增)</b>' : ''}</td>
                <td>${a.kind}</td><td>${loc}</td><td>${a.yaw ?? '-'}°</td><td>${off}</td>
                <td>
                  <button class="btn btn-secondary btn-small eng-nudge" data-tid="${a.tid}" data-dx="1">◀</button>
                  <button class="btn btn-secondary btn-small eng-nudge" data-tid="${a.tid}" data-dx="-1">▶</button>
                  <button class="btn btn-secondary btn-small eng-nudge" data-tid="${a.tid}" data-dy="1">▲</button>
                  <button class="btn btn-secondary btn-small eng-nudge" data-tid="${a.tid}" data-dy="-1">▼</button>
                  <button class="btn btn-secondary btn-small eng-nudge" data-tid="${a.tid}" data-dyaw="15">↺</button>
                  <button class="btn btn-secondary btn-small eng-nudge" data-tid="${a.tid}" data-dyaw="-15">↻</button>
                  <button class="btn btn-secondary btn-small eng-del" data-tid="${a.tid}">🗑</button>
                </td></tr>`;
        }).join('');
        box.innerHTML = `<table class="lab-table"><thead><tr><th>参与者</th><th>类型</th><th>位置(x,y)</th>`
            + `<th>朝向</th><th>编辑量</th><th>调整（步行=◀▶▲▼，转角=↺↻）</th></tr></thead>`
            + `<tbody>${rows || '<tr><td colspan="6">（还没有 actor）</td></tr>'}</tbody></table>`;
        box.querySelectorAll('.eng-nudge').forEach(b => b.addEventListener('click', () => {
            const k = parseFloat(document.getElementById('engStep')?.value || '1');
            const body = { tid: parseInt(b.dataset.tid, 10) };
            if (b.dataset.dx) body.dx = parseFloat(b.dataset.dx) * k;
            if (b.dataset.dy) body.dy = parseFloat(b.dataset.dy) * k;
            if (b.dataset.dyaw) body.dyaw = parseFloat(b.dataset.dyaw);
            this.carlaEngine('actor', body).then(() => this.carlaEngineRefresh(false));
        }));
        box.querySelectorAll('.eng-del').forEach(b => b.addEventListener('click', () => {
            this.carlaEngine('actor/delete', { tid: parseInt(b.dataset.tid, 10) })
                .then(() => this.carlaEngineRefresh(false));
        }));
    }

    async carlaEnginePlay() {
        const st = this._eng().st || {};
        await this.carlaEngine('play', { play: !st.play });
        this.carlaEngineRefresh(false);
    }

    async carlaEngineFrame(delta) {
        const st = this._eng().st || { frame: 0, n_frames: 1 };
        const idx = Math.max(0, Math.min((st.frame || 0) + delta, (st.n_frames || 1) - 1));
        await this.carlaEngine('frame', { index: idx });
        this.carlaEngineRefresh(false);
    }

    async carlaEngineSave() {
        this._engStatus('正在把编辑后的场景存成 world.json …');
        try {
            const d = await this.carlaEngine('save', {});
            const el = document.getElementById('engSaved');
            if (el) el.innerHTML = `已保存：<code>${this._escapeHtml(d.path)}</code><br>`
                + '可以把它填到上面「1) 选场景」的输入里，或者直接「▶ 开始 CARLA 仿真并出视频」。';
            this._engStatus('已保存 ✅（编辑过的轨迹都在里面）');
        } catch (e) { this._engStatus('保存失败: ' + e.message, true); }
    }

    async carlaEngineShot() {
        try {
            const d = await this.carlaEngine('screenshot', {});
            const el = document.getElementById('engSaved');
            if (el) el.innerHTML += `<br>截图已存：<code>${this._escapeHtml(d.path)}</code>`;
        } catch (e) { this._engStatus('截图失败: ' + e.message, true); }
    }

    // 扩散精修：独立可用（不用跑一键自检）
    async simDifixRefine() {
        if (!this.state.sceneId) { this._simStatus('simDifixOut', '请先加载场景', true); return; }
        const f = parseInt(document.getElementById('simDifixFrame')?.value || '0', 10);
        this._simStatus('simDifixOut', `正在精修第 ${f} 帧……首次会加载扩散模型（可能 1-3 分钟）`);
        const box = document.getElementById('simDifixImages');
        if (box) box.innerHTML = '';
        try {
            const r = await fetch(`${API_BASE}/diffusion/refine_frame`, {
                method: 'POST', headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ scene_id: this.state.sceneId, frame_idx: f, width: 800, height: 600 })
            });
            const d = await r.json();
            if (!d.success) throw new Error(d.detail || '精修失败');
            const url = p => `${API_BASE}/diffusion/preview?path=${encodeURIComponent(p)}`;
            if (box) {
                box.innerHTML =
                    `<div><div class="lab-subtitle">精修前</div><img class="difix-img" src="${url(d.before)}"></div>`
                    + `<div><div class="lab-subtitle">精修后（Difix）</div><img class="difix-img" src="${url(d.after)}"></div>`;
            }
            this._simStatus('simDifixOut', `✅ 第 ${d.frame_idx} 帧精修完成，用时 ${d.seconds}s`);
        } catch (e) { this._simStatus('simDifixOut', '精修失败: ' + e.message, true); }
    }'''

CSS_ADD = '''
/* Difix 前后对比图 */
.difix-img{max-width:100%;width:420px;border-radius:8px;background:#000;border:1px solid rgba(255,255,255,.08)}
'''

PAIRS_HTML = [
    ("H1", H1_OLD, H1_NEW), ("H2", H2_OLD, H2_NEW), ("H3", H3_OLD, H3_NEW),
    ("H4", H4_OLD, H4_NEW), ("H5", H5_OLD, H5_NEW), ("H6", H6_OLD, H6_NEW),
    ("H7", H7_OLD, H7_NEW), ("H8", H8_OLD, H8_NEW),
]
PAIRS_JS = [("J1", J1_OLD, J1_NEW), ("J2", J2_OLD, J2_NEW), ("J3", J3_OLD, J3_NEW)]


def patch(path: Path, pairs, dry: bool):
    if not path.exists():
        print(f"!! 找不到 {path}")
        return False
    txt = path.read_text(encoding="utf-8")
    for tag, old, new in pairs:
        n = txt.count(old)
        if n != 1:
            print(f"!! {path.name} [{tag}] 锚点出现 {n} 次（期望 1）：{old[:70]!r}")
            return False
        if not dry:
            txt = txt.replace(old, new, 1)
    if not dry:
        if not Path(str(path) + ".bak4").exists():
            shutil.copy2(path, str(path) + ".bak4")
        path.write_text(txt, encoding="utf-8")
    print(f"{'✓ 校验' if dry else '✅ 已补丁'} {path.name}（{len(pairs)} 处）")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=str(DEFAULT_BASE))
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    base = Path(a.base)
    ok = patch(base / "index.html", PAIRS_HTML, a.dry_run)
    ok &= patch(base / "app.js", PAIRS_JS, a.dry_run)
    css = base / "styles.css"
    if css.exists():
        c = css.read_text(encoding="utf-8")
        if ".difix-img" in c:
            print("· styles.css 已有样式，跳过")
        elif a.dry_run:
            print("✓ 校验 styles.css（待追加）")
        else:
            shutil.copy2(css, str(css) + ".bak4")
            css.write_text(c + CSS_ADD, encoding="utf-8")
            print("✅ 已补丁 styles.css")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
