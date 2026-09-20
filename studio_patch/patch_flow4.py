#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""第四批前端改造：
  A) 修「🎬 CARLA 这条 / 送去 CARLA 验证」不生效：场景列表异步刷新会把选中的场景冲掉
  B) ④ 面板加「输出名」+「贴地」+「将跑：…」明示本次要跑什么
  C) ③ 扩散精修加「精修所有帧（出一段视频）」
用法: python patch_flow4.py [--base DIR] [--dry-run]
"""
import argparse
import shutil
import sys
from pathlib import Path

DEFAULT_BASE = Path("/autodl-fs/data/dggt-main/studio/frontend")

# ---------------- index.html ----------------
H1_OLD = '''                        <label class="cc-checkbox"><input type="checkbox" id="carlaExportXosc">顺带导出对齐版 xosc</label>
                    </div>
'''
H1_NEW = '''                        <label class="cc-checkbox"><input type="checkbox" id="carlaExportXosc">顺带导出对齐版 xosc</label>
                    </div>
                    <div class="lab-row">
                        <label>输出名</label>
                        <input type="text" id="carlaRunName" placeholder="给这次仿真起个名字（留空=按场景名）" style="width:220px">
                        <label>贴地</label>
                        <select id="carlaRoadSnap" class="cc-select" style="width:150px">
                            <option value="z" selected>贴合路面（推荐）</option>
                            <option value="lane">吸到车道中心</option>
                            <option value="none">不处理</option>
                        </select>
                        <span class="detail">车飘在空中就保持"贴合路面"；想都规规矩矩在车道里选"吸到车道中心"</span>
                    </div>
                    <div id="carlaWillRun" class="cc-status" style="color:#93c5fd">将跑：—</div>
'''

H2_OLD = '''                        <button id="simDifixRefineBtn" class="btn btn-primary btn-small">✨ 对当前帧做扩散精修</button>'''
H2_NEW = '''                        <button id="simDifixRefineBtn" class="btn btn-primary btn-small">✨ 对当前帧做扩散精修</button>
                        <button id="simDifixSeqBtn" class="btn btn-primary btn-small">🎞 精修所有帧（出一段视频）</button>'''

# ---------------- app.js ----------------
J1_OLD = '''            export_xosc: this._carlaChk('carlaExportXosc'),
        };'''
J1_NEW = '''            export_xosc: this._carlaChk('carlaExportXosc'),
            road_snap: this._carlaVal('carlaRoadSnap') || 'z',
        };
        const _rn = (this._carlaVal('carlaRunName') || '').trim();
        if (_rn) out.export_name = _rn;'''

J2_OLD = '''    async carlaLoadScenarios() {
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
    }'''
J2_NEW = '''    async carlaLoadScenarios() {
        const sel = document.getElementById('carlaSceneSel');
        // 关键：这个刷新是异步的，会把「批量实例/上一步」刚选好的场景冲掉（表现为永远跑默认场景）。
        // 所以先记下要保留的选择，重建列表后再选回去；如果它不在列表里（比如批量实例的 world.json）就补一条。
        const keep = this._carlaPendingScenario || (sel ? sel.value : '');
        try {
            const r = await fetch(`${API_BASE}/carla/scenarios`);
            const d = await r.json();
            if (sel) {
                const items = (d.items || []).filter(i => i.kind !== 'video');
                sel.innerHTML = items.map(i =>
                    `<option value="${i.path}">${i.path}  ·  ${i.num_tracks || '?'} 物体 / ${i.num_dynamic || 0} 动态`
                    + `${(i.roles || []).length ? ' · ' + i.roles.join(',') : ''}`
                    + `${i.rendered_video ? ' · 已出片' : ''}</option>`).join('');
                if (keep) {
                    if (![...sel.options].some(o => o.value === keep)) {
                        const o = document.createElement('option');
                        o.value = keep;
                        o.textContent = `【指定】${keep}`;
                        sel.insertBefore(o, sel.firstChild);
                    }
                    sel.value = keep;
                }
                if (!sel.dataset.bound) {
                    sel.addEventListener('change', () => {
                        this._carlaPendingScenario = null;   // 用户手动选了，就不再粘住旧的
                        this._carlaWillRun();
                    });
                    sel.dataset.bound = '1';
                }
            }
            this._carlaWillRun();
            this._carlaStatus(`可跑场景 ${d.count} 个`);
        } catch (e) { this._carlaStatus('场景列表失败: ' + e.message, true); }
    }

    // 明示"这次会跑什么场景"，避免又跑到默认场景
    _carlaWillRun() {
        const el = document.getElementById('carlaWillRun');
        if (!el) return;
        const useCurrent = this._carlaChk('carlaUseCurrent') || !this._carlaVal('carlaSceneSel');
        const rn = (this._carlaVal('carlaRunName') || '').trim();
        const t = (this._carlaVal('carlaScenarioType') || '').trim();
        const what = useCurrent
            ? `当前编辑的场景（scene_id=<code>${this._escapeHtml(this.state.sceneId || '未加载')}</code>`
              + `${t ? '，事故类型 ' + this._escapeHtml(t) : ''}${this._carlaVal('carlaSeed') ? '，seed ' + this._escapeHtml(this._carlaVal('carlaSeed')) : ''}）`
            : `<code>${this._escapeHtml(this._carlaVal('carlaSceneSel') || '（未选）')}</code>`;
        el.innerHTML = '将跑：' + what
            + (rn ? `　输出名 <code>${this._escapeHtml(rn)}</code>` : '')
            + `　贴地=${this._escapeHtml(this._carlaVal('carlaRoadSnap') || 'z')}`
            + `　ego=${this._escapeHtml(this._carlaVal('carlaEgoMode') || 'playback')}`;
    }'''

J3_OLD = '''        }, `实例 ${m.instance_id || ''}`);
        this._labSwitchTab('carla');'''
J3_NEW = '''        }, `实例 ${m.instance_id || ''}`);
        this._carlaPendingScenario = m.world_json || null;   // 让场景列表刷新后仍然选中它
        this._labSwitchTab('carla');'''

J4_OLD = '''    _labGoCarla() {
        const c = this._labCase();
        if (!c.scenario_type) {
            this._carlaStatus('提示：还没选定具体事故案例，CARLA 会直接跑当前场景里已有的轨迹（也可以先回 ② 生成一个）');
        }
        this._labSwitchTab('carla');
        this._labPrefillCarla();
    }'''
J4_NEW = '''    _labGoCarla() {
        const c = this._labCase();
        if (c.instance_id && c.world_json) {
            // 第三步刚"载入场景"过的实例：④ 也用它的 world.json 最稳（不依赖内存场景）
            this._carlaPendingScenario = c.world_json;
        }
        if (!c.scenario_type) {
            this._carlaStatus('提示：还没选定具体事故案例，CARLA 会直接跑当前场景里已有的轨迹（也可以先回 ① 生成一个）');
        } else {
            this._carlaStatus(`已带入案例 ${c.scenario_type}（seed ${c.seed ?? '-'}）`
                + (c.world_json ? '，并把它的 world.json 选中为本次场景' : ''));
        }
        this._labSwitchTab('carla');
        this._labPrefillCarla();
        this._carlaWillRun();
    }'''

J5_OLD = '''    async carlaRelease() {'''
J5_NEW = r'''    // 扩散精修：一次精修所有帧 -> 出一段视频
    async simDifixSequence() {
        if (!this.state.sceneId) { this._simStatus('simDifixOut', '请先加载场景', true); return; }
        const f = parseInt(document.getElementById('simDifixFrame')?.value || '0', 10);
        this._simStatus('simDifixOut', `已提交"精修所有帧"作业（从第 ${f} 帧到结尾），正在跑第 1 帧…`);
        try {
            const r = await fetch(`${API_BASE}/diffusion/refine_sequence`, {
                method: 'POST', headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ scene_id: this.state.sceneId, start_frame: f, num_frames: 0,
                                       make_video: true, fps: 10 })
            });
            const d = await r.json();
            if (!d.success) throw new Error(d.detail || '提交失败');
            this._difixJob = d.job_id;
            this._simDifixPoll();
        } catch (e) { this._simStatus('simDifixOut', '提交失败: ' + e.message, true); }
    }

    _simDifixPoll() {
        if (this._difixTimer) clearInterval(this._difixTimer);
        const url = p => `${API_BASE}/diffusion/preview?path=${encodeURIComponent(p)}`;
        const vurl = p => `${API_BASE}/carla/file?path=${encodeURIComponent(p)}`;
        this._difixTimer = setInterval(async () => {
            try {
                const r = await fetch(`${API_BASE}/diffusion/refine_job/${this._difixJob}`);
                const j = await r.json();
                if (j.status === 'running') {
                    this._simStatus('simDifixOut', `精修中 ${j.done}/${j.total} 帧…`
                        + `（首帧要加载模型较慢，之后每帧约几秒~几十秒）`);
                    return;
                }
                clearInterval(this._difixTimer); this._difixTimer = null;
                const box = document.getElementById('simDifixImages');
                if (j.status === 'done') {
                    this._simStatus('simDifixOut', `✅ 全部 ${j.total} 帧精修完成，产物目录 ${j.out_dir}`);
                    if (box) {
                        box.innerHTML = (j.video
                            ? `<div><div class="lab-subtitle">精修后视频</div>`
                              + `<video class="difix-img" controls preload="metadata" src="${vurl(j.video)}"></video></div>`
                            : '')
                          + (j.video_before
                            ? `<div><div class="lab-subtitle">原始视频（对比）</div>`
                              + `<video class="difix-img" controls preload="metadata" src="${vurl(j.video_before)}"></video></div>`
                            : '');
                    }
                } else {
                    this._simStatus('simDifixOut', '精修失败: ' + (j.error || '未知错误'), true);
                }
            } catch (e) { /* 忽略轮询错误 */ }
        }, 3000);
    }

    async carlaRelease() {'''

J6_OLD = '''        bind('simDifixRefineBtn', () => this.simDifixRefine());'''
J6_NEW = '''        bind('simDifixRefineBtn', () => this.simDifixRefine());
        bind('simDifixSeqBtn', () => this.simDifixSequence());
        const _rnEl = document.getElementById('carlaRunName');
        if (_rnEl) _rnEl.addEventListener('input', () => this._carlaWillRun());
        const _rsEl = document.getElementById('carlaRoadSnap');
        if (_rsEl) _rsEl.addEventListener('change', () => this._carlaWillRun());
        const _ucEl = document.getElementById('carlaUseCurrent');
        if (_ucEl) _ucEl.addEventListener('change', () => this._carlaWillRun());'''

J7_OLD = '''        if (name === 'carla') { this.carlaRefreshStatus(1); this.carlaLoadScenarios(); this.carlaLoadJobs(); }'''
J7_NEW = '''        if (name === 'carla') { this.carlaRefreshStatus(1); this.carlaLoadScenarios(); this.carlaLoadJobs(); this._carlaWillRun(); }'''

PAIRS_HTML = [("H1", H1_OLD, H1_NEW), ("H2", H2_OLD, H2_NEW)]
PAIRS_JS = [("J1", J1_OLD, J1_NEW), ("J2", J2_OLD, J2_NEW), ("J3", J3_OLD, J3_NEW),
            ("J4", J4_OLD, J4_NEW), ("J5", J5_OLD, J5_NEW), ("J6", J6_OLD, J6_NEW),
            ("J7", J7_OLD, J7_NEW)]


def patch(path: Path, pairs, dry: bool):
    if not path.exists():
        print(f"!! 找不到 {path}")
        return False
    txt = path.read_text(encoding="utf-8")
    for tag, old, new in pairs:
        n = txt.count(old)
        if n != 1:
            print(f"!! {path.name} [{tag}] 锚点出现 {n} 次：{old[:70]!r}")
            return False
        if not dry:
            txt = txt.replace(old, new, 1)
    if not dry:
        if not Path(str(path) + ".bak5").exists():
            shutil.copy2(path, str(path) + ".bak5")
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
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
