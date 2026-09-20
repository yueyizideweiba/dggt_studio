/* ============================================================================
 * 可折叠面板（手风琴）—— ui-panels.js
 *
 * 背景：左侧 8 个面板 + 右侧 2 个面板原来是**一路平铺**的 `.panel-section`，
 * 一屏看下去全是标题和控件，容易看混、也不实用。
 *
 * 这个脚本把它们变成"可收起的分组卡片"：
 *   - 点击标题栏（或按 Enter/Space）收起 / 展开，带箭头指示与过渡动画；
 *   - 按功能分成若干组（场景与物体 / 事故与轨迹 / 模型生成 / 主车视角）；
 *   - 顶部工具条：搜索面板、全部展开、全部收起、单开 / 多开切换；
 *   - 展开状态记在 localStorage，刷新后保持；
 *   - 收起时标题栏会显示一句"里面有什么"，避免收起后找不着；
 *   - 标题栏里原有的按钮/输入框（如"刷新""搜索物体"）照常点，不会误触发折叠。
 *
 * 实现上**只包裹既有 DOM**（把 header 之后的节点搬进 .panel-body），不移动、不改 id、
 * 不改顺序，所以 app.js 里所有 getElementById / querySelector 都不受影响。
 *
 * 对外 API（也可在控制台用）：
 *   window.DGGTPanels.open('语言编辑轨迹')     // 按标题（模糊）或 key 展开
 *   window.DGGTPanels.close('...')
 *   window.DGGTPanels.toggle('...')
 *   window.DGGTPanels.expandAll() / collapseAll()
 * ========================================================================== */
(function () {
    'use strict';

    var STORAGE_KEY = 'dggt.panels.v1';
    var TOOLBAR_ID = 'panelToolbar';

    /* 面板元信息：分组 + 收起时的一句说明（按标题正则匹配）。不用图标，保持简洁。 */
    var META = [
        { re: /场景信息/,              group: '场景与物体', hint: '路径 / 帧数 / 刷新' },
        { re: /动态物体列表/,          group: '场景与物体', hint: '搜索并选中物体' },
        { re: /生成\s*Corner\s*Case/i, group: '事故与轨迹', hint: '追尾 / 变道 / 侧碰 / 行人' },
        { re: /语言编辑轨迹/,          group: '事故与轨迹', hint: '自然语言改轨迹（LLM）' },
        { re: /车头朝向/,              group: '事故与轨迹', hint: '朝向自动跟随运动方向' },
        { re: /SAM\s*3D.*重建/i,       group: '模型生成',   hint: '分割 → 重建 → 替换' },
        { re: /文本添加实体/,          group: '模型生成',   hint: 'LLaDA 出图 + SAM3D' },
        { re: /主车视角/,              group: '主车视角',   hint: '切换视角 / 自动贴地' },
        { re: /选中物体/,              group: '当前选择',   hint: '位姿 / 尺寸 / 轨迹' },
        { re: /编辑历史/,              group: '当前选择',   hint: '撤销 / 重做' }
    ];

    /* 线框图标（内联 SVG，单色、跟随文字颜色）——不用 emoji，避免"AI 味" */
    var SVG = {
        chevron: '<svg viewBox="0 0 16 16" width="12" height="12" aria-hidden="true">'
            + '<path d="M4.5 6.5 8 10l3.5-3.5" fill="none" stroke="currentColor" '
            + 'stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/></svg>',
        expand: '<svg viewBox="0 0 16 16" width="13" height="13" aria-hidden="true">'
            + '<path d="M6.5 2.5h-4v4M9.5 2.5h4v4M6.5 13.5h-4v-4M9.5 13.5h4v-4" fill="none" '
            + 'stroke="currentColor" stroke-width="1.4" stroke-linecap="round" stroke-linejoin="round"/></svg>',
        collapse: '<svg viewBox="0 0 16 16" width="13" height="13" aria-hidden="true">'
            + '<path d="M2.5 6.5h4v-4M13.5 6.5h-4v-4M2.5 9.5h4v4M13.5 9.5h-4v4" fill="none" '
            + 'stroke="currentColor" stroke-width="1.4" stroke-linecap="round" stroke-linejoin="round"/></svg>',
        single: '<svg viewBox="0 0 16 16" width="13" height="13" aria-hidden="true">'
            + '<rect x="2.5" y="3" width="11" height="3.6" rx="1.1" fill="currentColor" opacity="0.9"/>'
            + '<rect x="2.5" y="9.4" width="11" height="3.6" rx="1.1" fill="none" '
            + 'stroke="currentColor" stroke-width="1.2"/></svg>',
        multi: '<svg viewBox="0 0 16 16" width="13" height="13" aria-hidden="true">'
            + '<rect x="2.5" y="3" width="11" height="3.6" rx="1.1" fill="currentColor" opacity="0.9"/>'
            + '<rect x="2.5" y="9.4" width="11" height="3.6" rx="1.1" fill="currentColor" opacity="0.9"/></svg>'
    };

    /* 首次打开时默认展开哪些（其余收起；之后以 localStorage 记录为准） */
    var DEFAULT_OPEN = [/场景信息/, /动态物体列表/, /选中物体/, /编辑历史/];

    var state = loadState();
    var searchSnapshot = null;   // 搜索前的展开状态，清空搜索后还原

    function loadState() {
        try {
            var raw = JSON.parse(localStorage.getItem(STORAGE_KEY) || '{}');
            return {
                open: (raw && typeof raw.open === 'object' && raw.open) || {},
                single: raw && raw.single !== undefined ? !!raw.single : true
            };
        } catch (e) {
            return { open: {}, single: true };
        }
    }

    function saveState() {
        try {
            localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
        } catch (e) { /* 隐私模式等：忽略 */ }
    }

    function slug(s) {
        return String(s || '').toLowerCase().replace(/[^a-z0-9\u4e00-\u9fa5]+/g, '-').replace(/^-|-$/g, '');
    }

    function metaFor(title) {
        for (var i = 0; i < META.length; i++) {
            if (META[i].re.test(title)) return META[i];
        }
        return { group: '', hint: '' };
    }

    function defaultOpen(title) {
        for (var i = 0; i < DEFAULT_OPEN.length; i++) {
            if (DEFAULT_OPEN[i].test(title)) return true;
        }
        return false;
    }

    /* ------------------------------------------------------------------ */
    /* 把一段平铺的 .panel-section 改造成可折叠卡片                        */
    /* ------------------------------------------------------------------ */

    function buildPanel(section, index) {
        if (section.dataset.panelKey) return null;          // 幂等：重复调用不重复包裹
        var header = section.querySelector(':scope > .panel-header');
        if (!header) return null;

        var h3 = header.querySelector('h3');
        var title = (h3 && h3.textContent ? h3.textContent : '').trim() || ('面板 ' + (index + 1));
        var meta = metaFor(title);
        var key = slug(title) || ('panel-' + index);

        // 1) 把 header 之后的兄弟节点全部搬进 .panel-body（先搬再挂 wrap，顺序不能反）
        var body = document.createElement('div');
        body.className = 'panel-body';
        var wrap = document.createElement('div');
        wrap.className = 'panel-body-wrap';
        wrap.id = 'panel-body-' + key;
        wrap.appendChild(body);
        var node = header.nextSibling;
        while (node) {
            var next = node.nextSibling;
            if (node !== wrap) body.appendChild(node);
            node = next;
        }
        section.appendChild(wrap);

        // 2) 标题区 = 图标 + 原标题；末尾再加折叠箭头
        var titleWrap = document.createElement('div');
        titleWrap.className = 'panel-title';
        header.insertBefore(titleWrap, h3);
        titleWrap.appendChild(h3);

        var hint = document.createElement('span');
        hint.className = 'panel-hint';
        hint.textContent = meta.hint;
        titleWrap.appendChild(hint);

        var chev = document.createElement('span');
        chev.className = 'panel-chevron';
        chev.innerHTML = SVG.chevron;
        chev.setAttribute('aria-hidden', 'true');
        header.appendChild(chev);

        // 3) 无障碍 + 交互
        header.setAttribute('role', 'button');
        header.setAttribute('tabindex', '0');
        header.setAttribute('aria-controls', wrap.id);
        header.setAttribute('title', '点击展开/收起：' + title);
        section.dataset.panelKey = key;
        section.dataset.panelTitle = title;
        section.dataset.panelGroup = meta.group || '';

        header.addEventListener('click', function (e) {
            // 标题栏里的按钮/输入框照常用，不触发折叠
            if (e.target.closest('button, input, select, textarea, a, label')) return;
            toggle(section);
        });
        header.addEventListener('keydown', function (e) {
            if (e.key === 'Enter' || e.key === ' ') {
                e.preventDefault();
                toggle(section);
            }
        });

        applyOpen(section, state.open[key] !== undefined ? state.open[key] : defaultOpen(title), false);
        return key;
    }

    function applyOpen(section, open, persist) {
        var key = section.dataset.panelKey;
        var header = section.querySelector(':scope > .panel-header');
        section.classList.toggle('collapsed', !open);
        if (header) header.setAttribute('aria-expanded', open ? 'true' : 'false');
        if (persist !== false) {
            state.open[key] = !!open;
            saveState();
        }
    }

    function isOpen(section) {
        return !section.classList.contains('collapsed');
    }

    function toggle(section, opts) {
        opts = opts || {};
        var willOpen = !isOpen(section);
        if (willOpen && state.single && !opts.noSingle) {
            // 单开模式：同侧其它面板收起，避免又变成"一路平铺"
            var root = section.parentElement;
            Array.prototype.forEach.call(root.querySelectorAll(':scope > .panel-section'), function (other) {
                if (other !== section && isOpen(other)) applyOpen(other, false, opts.persist);
            });
        }
        applyOpen(section, willOpen, opts.persist);
        if (willOpen) {
            try {
                section.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
            } catch (e) { /* 老浏览器 */ }
            document.dispatchEvent(new CustomEvent('dggt:panel-open', { detail: { key: section.dataset.panelKey } }));
        }
    }

    /* ------------------------------------------------------------------ */
    /* 分组标题 + 顶部工具条                                               */
    /* ------------------------------------------------------------------ */

    function insertGroupTitles(root, panels) {
        var lastGroup = null;
        panels.forEach(function (section) {
            var group = section.dataset.panelGroup || '';
            if (!group || group === lastGroup) { lastGroup = group; return; }
            var el = document.createElement('div');
            el.className = 'panel-group-title';
            el.dataset.group = group;
            el.textContent = group;
            root.insertBefore(el, section);
            lastGroup = group;
        });
    }

    function buildToolbar(root) {
        if (document.getElementById(TOOLBAR_ID)) return null;
        var bar = document.createElement('div');
        bar.className = 'panel-toolbar';
        bar.id = TOOLBAR_ID;

        var search = document.createElement('input');
        search.type = 'text';
        search.className = 'panel-search';
        search.id = 'panelSearch';
        search.placeholder = '搜索面板…';
        search.setAttribute('aria-label', '搜索面板');

        var btnExpand = mini(SVG.expand, '全部展开', 'expand');
        var btnCollapse = mini(SVG.collapse, '全部收起', 'collapse');
        var btnSingle = mini(state.single ? SVG.single : SVG.multi,
                             state.single ? '单开（点击切换为多开）' : '多开（点击切换为单开）',
                             'single');
        if (state.single) btnSingle.classList.add('active');

        bar.appendChild(search);
        bar.appendChild(btnExpand);
        bar.appendChild(btnCollapse);
        bar.appendChild(btnSingle);
        root.insertBefore(bar, root.firstChild);

        function mini(html, title, act) {
            var b = document.createElement('button');
            b.type = 'button';
            b.className = 'panel-mini';
            b.innerHTML = html;
            b.title = title;
            b.setAttribute('aria-label', title);
            b.dataset.act = act;
            return b;
        }

        bar.addEventListener('click', function (e) {
            var b = e.target.closest('.panel-mini');
            if (!b) return;
            var act = b.dataset.act;
            if (act === 'expand') expandAll(root, true);
            else if (act === 'collapse') expandAll(root, false);
            else if (act === 'single') {
                state.single = !state.single;
                saveState();
                b.classList.toggle('active', state.single);
                b.innerHTML = state.single ? SVG.single : SVG.multi;
                b.title = state.single ? '单开（点击切换为多开）' : '多开（点击切换为单开）';
                b.setAttribute('aria-label', b.title);
            }
        });

        search.addEventListener('input', function () { applyFilter(root, search.value); });
        search.addEventListener('keydown', function (e) {
            if (e.key === 'Escape') { search.value = ''; applyFilter(root, ''); }
        });
        return bar;
    }

    function expandAll(root, open, persist) {
        var panels = Array.prototype.slice.call(root.querySelectorAll(':scope > .panel-section'));
        // 先还原搜索状态，避免"全部展开"被搜索的临时状态覆盖
        if (searchSnapshot) { restoreSearchSnapshot(root); var s = document.getElementById('panelSearch'); if (s) s.value = ''; }
        panels.forEach(function (p) { applyOpen(p, open, persist); });
    }

    function applyFilter(root, query) {
        var q = String(query || '').trim().toLowerCase();
        var panels = Array.prototype.slice.call(root.querySelectorAll(':scope > .panel-section'));
        if (!q) {
            restoreSearchSnapshot(root);
            return;
        }
        if (!searchSnapshot) {
            searchSnapshot = {};
            panels.forEach(function (p) { searchSnapshot[p.dataset.panelKey] = isOpen(p); });
        }
        panels.forEach(function (p) {
            var title = (p.dataset.panelTitle || '').toLowerCase();
            var group = (p.dataset.panelGroup || '').toLowerCase();
            var hint = '';
            var hintEl = p.querySelector('.panel-hint');
            if (hintEl) hint = (hintEl.textContent || '').toLowerCase();
            var hit = title.indexOf(q) >= 0 || group.indexOf(q) >= 0 || hint.indexOf(q) >= 0;
            p.classList.toggle('filtered-out', !hit);
            // 搜索时把命中的直接展开，省一次点击；不写 localStorage
            if (hit) applyOpen(p, true, false);
        });
        // 整组都被过滤掉时，组标题也藏起来
        Array.prototype.forEach.call(root.querySelectorAll(':scope > .panel-group-title'), function (gt) {
            var group = gt.dataset.group;
            var any = panels.some(function (p) {
                return p.dataset.panelGroup === group && !p.classList.contains('filtered-out');
            });
            gt.classList.toggle('filtered-out', !any);
        });
    }

    function restoreSearchSnapshot(root) {
        if (!searchSnapshot) return;
        var snap = searchSnapshot;
        searchSnapshot = null;
        Array.prototype.slice.call(root.querySelectorAll(':scope > .panel-section')).forEach(function (p) {
            p.classList.remove('filtered-out');
            if (snap[p.dataset.panelKey] !== undefined) applyOpen(p, snap[p.dataset.panelKey], false);
        });
        Array.prototype.forEach.call(root.querySelectorAll(':scope > .panel-group-title'), function (gt) {
            gt.classList.remove('filtered-out');
        });
    }

    /* ------------------------------------------------------------------ */
    /* 初始化                                                              */
    /* ------------------------------------------------------------------ */

    function init() {
        var roots = [document.querySelector('.left-panel')].filter(Boolean);
        // 右侧面板也用同一套折叠体验（只是不加搜索工具条）
        var right = document.querySelector('.right-panel');
        if (right) roots.push(right);
        if (!roots.length) return;

        roots.forEach(function (root, ri) {
            var panels = Array.prototype.slice.call(root.querySelectorAll(':scope > .panel-section'));
            if (!panels.length) return;
            panels.forEach(function (s, i) { buildPanel(s, i); });
            insertGroupTitles(root, panels);
            if (ri === 0) buildToolbar(root);
        });
        watchSelection();
    }

    /* 选中物体时自动把右侧「选中物体」面板展开——否则点了车却看不到属性，很别扭 */
    function watchSelection() {
        if (!window.MutationObserver) return;
        var target = document.getElementById('selectedObject');
        var section = find('选中物体');
        if (!target || !section) return;
        var timer = null;
        new MutationObserver(function () {
            // app.js 更新选中物体信息时把面板打开（合并短时间内的多次改动）
            if (isOpen(section)) return;
            if (timer) clearTimeout(timer);
            timer = setTimeout(function () { applyOpen(section, true, true); }, 60);
        }).observe(target, { childList: true, subtree: true, characterData: true });
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }

    /* 对外 API */
    window.DGGTPanels = {
        open: function (idOrTitle) { return act(idOrTitle, true); },
        close: function (idOrTitle) { return act(idOrTitle, false); },
        toggle: function (idOrTitle) {
            var s = find(idOrTitle);
            if (s) toggle(s);
            return !!s;
        },
        expandAll: function () { var r = document.querySelector('.left-panel'); if (r) expandAll(r, true); },
        collapseAll: function () { var r = document.querySelector('.left-panel'); if (r) expandAll(r, false); },
        list: function () {
            return Array.prototype.map.call(
                document.querySelectorAll('.left-panel > .panel-section, .right-panel > .panel-section'),
                function (s) { return { key: s.dataset.panelKey, title: s.dataset.panelTitle, open: isOpen(s) }; });
        }
    };

    function find(idOrTitle) {
        var all = Array.prototype.slice.call(
            document.querySelectorAll('.left-panel > .panel-section, .right-panel > .panel-section'));
        var q = String(idOrTitle || '').trim();
        var exact = all.filter(function (s) { return s.dataset.panelKey === q; });
        if (exact.length) return exact[0];
        var hit = all.filter(function (s) { return (s.dataset.panelTitle || '').indexOf(q) >= 0; });
        return hit[0] || null;
    }

    function act(idOrTitle, open) {
        var s = find(idOrTitle);
        if (!s) return false;
        if (open && state.single) {
            var root = s.parentElement;
            Array.prototype.forEach.call(root.querySelectorAll(':scope > .panel-section'), function (o) {
                if (o !== s && isOpen(o)) applyOpen(o, false, true);
            });
        }
        applyOpen(s, open, true);
        if (open) {
            try { s.scrollIntoView({ block: 'nearest', behavior: 'smooth' }); } catch (e) { /* ignore */ }
        }
        return true;
    }
})();
