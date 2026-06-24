/**
 * DGGT Studio - 3D 自由视角视图模块（服务端渲染 + track 追踪）
 *
 * 目标：从任意视角查看 4DGS（普通渲染只能锁自车视角），并在三维空间中
 * 选中/拖动/删除动态物体、查看与编辑其运动轨迹。
 *
 * 关键决策：
 * - 画面由后端 gsplat 按相机 c2w 渲染并返回 PNG（浏览器零 GPU 负担，画质一致）。
 * - 所有物体一律用 **track_id** 标识（跨帧稳定），避免不同帧 raw object_id 不连续。
 * - 播放流畅性：保留上一帧图像直到新图就绪（不清屏→不黑屏）；按 track 预取相邻帧。
 * - 编辑时锁定场景：选中物体进入移动/旋转模式后，禁用相机轨道/平移，
 *   只允许拖动被选物体或其轨迹点。
 *
 * 相机：OpenCV 约定（+Z 前向，+Y 向下；世界“上方” = -Y）。
 */

class DGGTViewer3D {
    constructor(container, options = {}) {
        this.container = container;
        this.apiBase = options.apiBase || 'http://localhost:8000/api';
        this.onObjectSelected = options.onObjectSelected || (() => {});
        this.onObjectEdited = options.onObjectEdited || (() => {});

        this.renderHeight = options.renderHeight || 540;
        this.renderWidth = options.renderWidth || 960;
        this.fovY = options.fovY || 50;

        this.sceneId = null;
        this.frameIdx = 0;
        this.objects = [];          // 当前帧物体（含 track_id）
        this.sceneCenter = [0, 0, 0];
        this.cameraMeta = null;
        this.selectedTrackId = null;
        this.showBoxes = true;
        this.showTrajectory = true;
        this.lockHeight = true;     // 轨迹/移动编辑时锁定高度（仅在地面 XZ 平面移动）
        this.trajectory = null;     // 选中 track 的轨迹 {trajectory:[{frame_idx,center,edited}], ...}

        // 轨道相机
        this.target = [0, 0, 0];
        this.radius = 30;
        this.phi = Math.PI / 2;
        this.theta = 0;

        // 交互
        this.mode = 'select';       // 'select' | 'translate' | 'rotate'
        this.trajectoryEditMode = false;  // 轨迹编辑状态（独立开关）
        this.isOrbiting = false;
        this.isPanning = false;
        this.isDraggingObject = false;
        this.draggingTrajPoint = null;  // 正在拖动的轨迹点 frame_idx
        this.prevMouse = { x: 0, y: 0 };

        // 渲染节流
        this.pendingRender = false;
        this.renderQueued = false;
        this.lastC2W = null;

        // 帧图像缓存（用于播放流畅 & 不黑屏）: key=frameIdx -> Image
        this.frameCache = new Map();
        this.cacheLimit = 60;
        this.cacheCameraKey = null;  // 相机不变时缓存才有效

        this.active = false;
        this.initialized = false;
    }

    // ==================== 初始化 ====================

    init() {
        if (this.initialized) return;

        this.canvas = document.createElement('canvas');
        this.canvas.className = 'view3d-canvas';
        this.canvas.style.width = '100%';
        this.canvas.style.height = '100%';
        this.canvas.style.display = 'block';
        this.canvas.style.cursor = 'grab';
        this.container.appendChild(this.canvas);
        this.ctx = this.canvas.getContext('2d');

        this.overlay = document.createElement('canvas');
        this.overlay.style.position = 'absolute';
        this.overlay.style.inset = '0';
        this.overlay.style.width = '100%';
        this.overlay.style.height = '100%';
        this.overlay.style.pointerEvents = 'none';
        this.container.appendChild(this.overlay);
        this.octx = this.overlay.getContext('2d');

        this.hint = document.createElement('div');
        this.hint.className = 'view3d-hint';
        this.container.appendChild(this.hint);
        this._updateHint();

        this._setupEvents();
        this._resizeCanvas();
        this.initialized = true;
    }

    _updateHint() {
        if (!this.hint) return;
        if (this.trajectoryEditMode) {
            this.hint.textContent = '轨迹编辑：拖动轨迹上的黄色顶点修改轨迹（场景已锁定）· 再次点击轨迹按钮退出';
        } else if (this.mode === 'select') {
            this.hint.textContent = '左键旋转 · 右键平移 · 滚轮缩放 · 点击选中物体';
        } else if (this.mode === 'translate') {
            this.hint.textContent = '移动模式：拖动选中物体移动（场景已锁定）· 滚轮缩放';
        } else if (this.mode === 'rotate') {
            this.hint.textContent = '旋转模式：左右拖动旋转选中物体（场景已锁定）· 滚轮缩放';
        }
    }

    _resizeCanvas() {
        const w = this.container.clientWidth || 1280;
        const h = this.container.clientHeight || 720;
        this.canvas.width = this.renderWidth;
        this.canvas.height = this.renderHeight;
        this.overlay.width = w;
        this.overlay.height = h;
    }

    show() {
        if (!this.initialized) this.init();
        this.active = true;
        this.container.style.display = 'block';
        this._resizeCanvas();
    }

    hide() {
        this.active = false;
        this.container.style.display = 'none';
    }

    // ==================== 加载帧 ====================

    async loadFrame(sceneId, frameIdx, preserveCamera = false) {
        if (!this.initialized) this.init();
        const sceneChanged = this.sceneId !== sceneId;
        this.sceneId = sceneId;
        this.frameIdx = frameIdx;

        try {
            const resp = await fetch(`${this.apiBase}/scene3d/${sceneId}/frame/${frameIdx}`);
            const data = await resp.json();
            if (data && data.success) {
                this.objects = data.objects || [];
                this.sceneCenter = data.scene_center || [0, 0, 0];
                this.cameraMeta = data.camera || null;
                this._applyCameraMeta();
                if (!preserveCamera || sceneChanged) {
                    this._resetCameraToEgo();
                }
            }
        } catch (e) {
            console.error('[Viewer3D] 获取场景信息失败:', e);
        }

        // 若选中了 track，刷新它在本帧的轨迹（轨迹本身跨帧不变，只在选中变化时取）
        if (this.selectedTrackId !== null && !this.trajectory) {
            this._fetchTrajectory(this.selectedTrackId);
        }

        this.requestRender();
    }

    _applyCameraMeta() {
        const cm = this.cameraMeta;
        if (!cm) return;
        if (cm.width && cm.height) {
            const aspect = cm.width / cm.height;
            this.renderWidth = Math.round(this.renderHeight * aspect);
        }
        if (cm.intrinsics && cm.height) {
            const fy = cm.intrinsics[1][1];
            this.fovY = 2 * Math.atan((cm.height / 2) / fy) * 180 / Math.PI;
        }
        this._resizeCanvas();
    }

    _resetCameraToEgo() {
        const ext = this.cameraMeta && this.cameraMeta.extrinsics_world;
        if (!ext) { this._resetCameraToScene(); return; }
        const eye = [ext[0][3], ext[1][3], ext[2][3]];
        const fwd = this._normalize([ext[0][2], ext[1][2], ext[2][2]]);
        let R = 25;
        if (this.objects.length) {
            const c = this.sceneCenter;
            R = Math.max(8, this._len([c[0] - eye[0], c[1] - eye[1], c[2] - eye[2]]));
        }
        this.radius = R;
        this.target = [eye[0] + fwd[0] * R, eye[1] + fwd[1] * R, eye[2] + fwd[2] * R];
        const dir = this._normalize([eye[0] - this.target[0], eye[1] - this.target[1], eye[2] - this.target[2]]);
        this.phi = Math.acos(Math.max(-1, Math.min(1, -dir[1])));
        this.theta = Math.atan2(dir[2], dir[0]);
        this._invalidateCache();
    }

    _resetCameraToScene() {
        this.target = this.sceneCenter.slice();
        this.radius = 30;
        this.phi = Math.PI / 2;
        this.theta = -Math.PI / 2;
        this._invalidateCache();
    }

    resetView() { this._resetCameraToEgo(); this.requestRender(); }

    focusSelected() {
        const obj = this.objects.find(o => o.track_id === this.selectedTrackId);
        if (obj && obj.center) {
            this.target = obj.center.slice();
            this.radius = 12;
            this._invalidateCache();
            this.requestRender();
        }
    }

    // ==================== 相机数学（OpenCV：+Z 前，+Y 下） ====================

    _visualUp() { return [0, -1, 0]; }

    _orbitDir() {
        const sp = Math.sin(this.phi), cp = Math.cos(this.phi);
        return [sp * Math.cos(this.theta), -cp, sp * Math.sin(this.theta)];
    }

    _cameraPosition() {
        const dir = this._orbitDir();
        return [
            this.target[0] + this.radius * dir[0],
            this.target[1] + this.radius * dir[1],
            this.target[2] + this.radius * dir[2],
        ];
    }

    _computeC2W() {
        const eye = this._cameraPosition();
        const z = this._normalize([
            this.target[0] - eye[0], this.target[1] - eye[1], this.target[2] - eye[2]
        ]);
        const visualUp = this._visualUp();
        let yCam = [-visualUp[0], -visualUp[1], -visualUp[2]];
        let x = this._cross(yCam, z);
        if (this._len(x) < 1e-6) x = [1, 0, 0];
        x = this._normalize(x);
        yCam = this._normalize(this._cross(z, x));
        return [
            [x[0], yCam[0], z[0], eye[0]],
            [x[1], yCam[1], z[1], eye[1]],
            [x[2], yCam[2], z[2], eye[2]],
            [0, 0, 0, 1],
        ];
    }

    _cross(a, b) { return [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]]; }
    _len(a) { return Math.sqrt(a[0]*a[0]+a[1]*a[1]+a[2]*a[2]); }
    _normalize(a) { const l = this._len(a) || 1; return [a[0]/l, a[1]/l, a[2]/l]; }

    _worldToScreen(world, c2w) {
        const R = [
            [c2w[0][0], c2w[1][0], c2w[2][0]],
            [c2w[0][1], c2w[1][1], c2w[2][1]],
            [c2w[0][2], c2w[1][2], c2w[2][2]],
        ];
        const t = [c2w[0][3], c2w[1][3], c2w[2][3]];
        const d = [world[0]-t[0], world[1]-t[1], world[2]-t[2]];
        const cam = [
            R[0][0]*d[0]+R[0][1]*d[1]+R[0][2]*d[2],
            R[1][0]*d[0]+R[1][1]*d[1]+R[1][2]*d[2],
            R[2][0]*d[0]+R[2][1]*d[1]+R[2][2]*d[2],
        ];
        const z = cam[2];
        if (z <= 0.05) return null;
        const fy = (this.renderHeight / 2.0) / Math.tan(this._deg2rad(this.fovY) / 2.0);
        const fx = fy;
        return {
            x: (cam[0] / z) * fx + this.renderWidth / 2.0,
            y: (cam[1] / z) * fy + this.renderHeight / 2.0,
            depth: z,
        };
    }

    _deg2rad(d) { return d * Math.PI / 180; }

    // ==================== 缓存 ====================

    _cameraKey() {
        const c = this._computeC2W();
        return c.map(r => r.map(v => v.toFixed(3)).join(',')).join(';')
            + `|${this.showBoxes}|${this.showTrajectory}|${this.selectedTrackId}`;
    }

    _invalidateCache() {
        this.frameCache.clear();
        this.cacheCameraKey = null;
    }

    // ==================== 渲染请求（节流 + 缓存） ====================

    requestRender() {
        if (!this.active || !this.sceneId) return;

        // 拖动编辑预览时，始终重新渲染（位姿是临时的，不能用缓存）
        if (this._liveEdit()) {
            if (this.pendingRender) { this.renderQueued = true; return; }
            this._doRender();
            return;
        }

        // 缓存命中：相机/显示状态未变且该帧已渲染
        const camKey = this._cameraKey();
        if (this.cacheCameraKey === camKey && this.frameCache.has(this.frameIdx)) {
            this._blit(this.frameCache.get(this.frameIdx));
            this._drawOverlay();
            return;
        }
        if (this.cacheCameraKey !== camKey) {
            this.frameCache.clear();
            this.cacheCameraKey = camKey;
        }

        if (this.pendingRender) { this.renderQueued = true; return; }
        this._doRender();
    }

    async _doRender() {
        this.pendingRender = true;
        const camKey = this._cameraKey();
        const c2w = this._computeC2W();
        this.lastC2W = c2w;
        const frameAtRequest = this.frameIdx;
        const live = this._liveEdit();  // 拖动中的临时位姿

        try {
            const resp = await fetch(`${this.apiBase}/render/freeview`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    scene_id: this.sceneId,
                    frame_idx: frameAtRequest,
                    c2w: c2w,
                    fov_y: this.fovY,
                    width: this.renderWidth,
                    height: this.renderHeight,
                    draw_bboxes: this.showBoxes,
                    draw_ids: this.showBoxes,
                    draw_trajectories: this.showTrajectory,
                    highlight_track_id: this.selectedTrackId,
                    live_track_id: live ? live.track_id : null,
                    live_pose_matrix: live ? live.pose : null
                })
            });
            const data = await resp.json();
            if (data && data.success && data.image) {
                const img = await this._loadImage(data.image);
                // 拖动预览的图像不写缓存（位姿是临时的）
                if (!live && this.cacheCameraKey === camKey) {
                    this.frameCache.set(frameAtRequest, img);
                    if (this.frameCache.size > this.cacheLimit) {
                        const first = this.frameCache.keys().next().value;
                        this.frameCache.delete(first);
                    }
                }
                if (this.frameIdx === frameAtRequest) {
                    this._blit(img);
                    this._drawOverlay();
                }
            }
        } catch (e) {
            console.error('[Viewer3D] 渲染失败:', e);
        } finally {
            this.pendingRender = false;
            if (this.renderQueued) {
                this.renderQueued = false;
                this._doRender();
            }
        }
    }

    // 当前是否处于某物体/轨迹点的拖动预览，返回 {track_id, pose}
    _liveEdit() {
        if (this.isDraggingObject && this.selectedTrackId !== null) {
            const obj = this.objects.find(o => o.track_id === this.selectedTrackId);
            if (obj) return { track_id: obj.track_id, pose: obj.pose_world };
        }
        if (this.draggingTrajPoint === this.frameIdx && this.selectedTrackId !== null) {
            const obj = this.objects.find(o => o.track_id === this.selectedTrackId);
            if (obj) return { track_id: obj.track_id, pose: obj.pose_world };
        }
        return null;
    }

    _loadImage(dataUrl) {
        return new Promise((resolve) => {
            const img = new Image();
            img.onload = () => resolve(img);
            img.onerror = () => resolve(null);
            img.src = dataUrl;
        });
    }

    _blit(img) {
        if (!img) return;
        // 不清屏直接覆盖，避免黑屏闪烁
        this.ctx.drawImage(img, 0, 0, this.canvas.width, this.canvas.height);
    }

    // 播放时预取：直接用已有相机渲染指定帧（异步填充缓存）
    async prefetchFrame(frameIdx) {
        if (!this.active || !this.sceneId) return;
        if (this.frameCache.has(frameIdx)) return;
        const camKey = this._cameraKey();
        if (this.cacheCameraKey !== camKey) return; // 相机变了不预取
        try {
            const resp = await fetch(`${this.apiBase}/render/freeview`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    scene_id: this.sceneId, frame_idx: frameIdx, c2w: this.lastC2W,
                    fov_y: this.fovY, width: this.renderWidth, height: this.renderHeight,
                    draw_bboxes: this.showBoxes, draw_ids: this.showBoxes,
                    draw_trajectories: this.showTrajectory, highlight_track_id: this.selectedTrackId
                })
            });
            const data = await resp.json();
            if (data && data.success && data.image && this.cacheCameraKey === camKey) {
                const img = await this._loadImage(data.image);
                this.frameCache.set(frameIdx, img);
            }
        } catch (e) { /* ignore */ }
    }

    // ==================== 叠加层 ====================

    _drawOverlay() {
        if (!this.octx) return;
        const ow = this.overlay.width, oh = this.overlay.height;
        this.octx.clearRect(0, 0, ow, oh);
        if (!this.lastC2W) return;
        const sx = ow / this.renderWidth;
        const sy = oh / this.renderHeight;

        // 选中物体高亮圈
        const sel = this.objects.find(o => o.track_id === this.selectedTrackId);
        if (sel) {
            const sp = this._worldToScreen(sel.center, this.lastC2W);
            if (sp) {
                const x = sp.x * sx, y = sp.y * sy;
                this.octx.beginPath();
                this.octx.arc(x, y, 11, 0, Math.PI * 2);
                this.octx.strokeStyle = '#00c8ff';
                this.octx.lineWidth = 3;
                this.octx.stroke();
            }
        }

        // 选中 track 的轨迹（可编辑点）
        if (this.showTrajectory && this.trajectory && this.trajectory.trajectory) {
            const pts = this.trajectory.trajectory;
            this.octx.strokeStyle = 'rgba(255,200,0,0.9)';
            this.octx.lineWidth = 2;
            this.octx.beginPath();
            let started = false;
            this._trajScreenPts = [];
            pts.forEach(p => {
                const sp = this._worldToScreen(p.center, this.lastC2W);
                if (!sp) { this._trajScreenPts.push(null); return; }
                const x = sp.x * sx, y = sp.y * sy;
                this._trajScreenPts.push({ x, y, frame_idx: p.frame_idx, depth: sp.depth });
                if (!started) { this.octx.moveTo(x, y); started = true; }
                else this.octx.lineTo(x, y);
            });
            this.octx.stroke();

            // 关键点（当前帧高亮，可拖动）
            this._trajScreenPts.forEach(sp => {
                if (!sp) return;
                const isCur = sp.frame_idx === this.frameIdx;
                this.octx.beginPath();
                this.octx.arc(sp.x, sp.y, isCur ? 6 : 3.5, 0, Math.PI * 2);
                this.octx.fillStyle = isCur ? '#ffd000' : 'rgba(255,200,0,0.7)';
                this.octx.fill();
                if (isCur) {
                    this.octx.strokeStyle = '#fff';
                    this.octx.lineWidth = 2;
                    this.octx.stroke();
                }
            });
        } else {
            this._trajScreenPts = null;
        }
    }

    // ==================== 事件 ====================

    _setupEvents() {
        this.canvas.addEventListener('mousedown', (e) => this._onMouseDown(e));
        window.addEventListener('mousemove', (e) => this._onMouseMove(e));
        window.addEventListener('mouseup', () => this._onMouseUp());
        this.canvas.addEventListener('wheel', (e) => this._onWheel(e), { passive: false });
        this.canvas.addEventListener('contextmenu', (e) => e.preventDefault());
        window.addEventListener('resize', () => {
            if (this.active) { this._resizeCanvas(); this._drawOverlay(); }
        });
    }

    _canvasMouse(e) {
        const rect = this.canvas.getBoundingClientRect();
        return {
            x: (e.clientX - rect.left) / rect.width * this.renderWidth,
            y: (e.clientY - rect.top) / rect.height * this.renderHeight,
        };
    }

    _overlayMouse(e) {
        const rect = this.canvas.getBoundingClientRect();
        return {
            x: (e.clientX - rect.left) / rect.width * this.overlay.width,
            y: (e.clientY - rect.top) / rect.height * this.overlay.height,
        };
    }

    // 是否处于锁定场景的编辑状态（移动/旋转物体 或 轨迹编辑）
    _editing() { return this.mode === 'translate' || this.mode === 'rotate'; }
    _sceneLocked() { return this._editing() || this.trajectoryEditMode; }

    _onMouseDown(e) {
        if (!this.active) return;
        this.prevMouse = { x: e.clientX, y: e.clientY };

        if (e.button === 0) {
            // 轨迹编辑状态：只允许拖动轨迹顶点
            if (this.trajectoryEditMode && this.selectedTrackId !== null) {
                const tp = this._pickTrajPoint(e);
                if (tp !== null) {
                    this.draggingTrajPoint = tp;
                    this.canvas.style.cursor = 'grabbing';
                }
                return;  // 锁定场景，不旋转
            }

            // 移动/旋转模式：锁定场景，操作选中物体
            if (this._editing()) {
                // 若已有选中物体，直接开始拖动它（无需精确点中）
                if (this.selectedTrackId !== null &&
                    this.objects.some(o => o.track_id === this.selectedTrackId)) {
                    // 若点到了另一个物体则切换选择
                    const picked = this._pickObject(e);
                    if (picked && picked.track_id !== this.selectedTrackId) {
                        this.selectTrack(picked.track_id);
                    }
                    this.isDraggingObject = true;
                    this.canvas.style.cursor = 'grabbing';
                    return;
                }
                // 尚无选中：尝试选中点中的物体
                const picked = this._pickObject(e);
                if (picked) {
                    this.selectTrack(picked.track_id);
                    this.isDraggingObject = true;
                    this.canvas.style.cursor = 'grabbing';
                }
                return;  // 场景锁定，不旋转
            }

            // 选择模式：点中则选中，否则旋转视角
            const picked = this._pickObject(e);
            if (picked) {
                this.selectTrack(picked.track_id);
            }
            this.isOrbiting = true;
            this.canvas.style.cursor = 'grabbing';
        } else if (e.button === 2 || e.button === 1) {
            if (this._sceneLocked()) return;  // 编辑/轨迹编辑时锁定平移
            this.isPanning = true;
            this.canvas.style.cursor = 'move';
        }
    }

    _onMouseMove(e) {
        if (!this.active) return;
        const dx = e.clientX - this.prevMouse.x;
        const dy = e.clientY - this.prevMouse.y;

        if (this.draggingTrajPoint !== null) {
            this._dragTrajPoint(e);
            this.prevMouse = { x: e.clientX, y: e.clientY };
            return;
        }
        if (this.isDraggingObject && this.selectedTrackId !== null) {
            this._dragObject(dx, dy);
            this.prevMouse = { x: e.clientX, y: e.clientY };
            return;
        }
        if (this.isOrbiting) {
            this.theta += dx * 0.005;
            this.phi = Math.max(0.05, Math.min(Math.PI - 0.05, this.phi - dy * 0.005));
            this.prevMouse = { x: e.clientX, y: e.clientY };
            this._invalidateCache();
            this.requestRender();
        } else if (this.isPanning) {
            this._panCamera(dx, dy);
            this.prevMouse = { x: e.clientX, y: e.clientY };
            this._invalidateCache();
            this.requestRender();
        }
    }

    _onMouseUp() {
        if (this.draggingTrajPoint !== null) {
            this._commitTrajPoint(this.draggingTrajPoint);
            this.draggingTrajPoint = null;
        }
        if (this.isDraggingObject) {
            this._commitObjectEdit();
        }
        this.isOrbiting = false;
        this.isPanning = false;
        this.isDraggingObject = false;
        this.canvas.style.cursor = this._sceneLocked() ? 'crosshair' : 'grab';
    }

    _onWheel(e) {
        if (!this.active) return;
        e.preventDefault();
        // 编辑模式仍允许缩放观察
        this.radius *= e.deltaY > 0 ? 1.08 : 0.926;
        this.radius = Math.max(1, Math.min(800, this.radius));
        this._invalidateCache();
        this.requestRender();
    }

    _panCamera(dx, dy) {
        const c2w = this._computeC2W();
        const right = [c2w[0][0], c2w[1][0], c2w[2][0]];
        const visualUp = this._visualUp();
        const speed = this.radius * 0.0012;
        for (let i = 0; i < 3; i++) {
            this.target[i] += -right[i] * dx * speed + visualUp[i] * dy * speed;
        }
    }

    // ==================== 选择 ====================

    _pickObject(e) {
        const c2w = this._computeC2W();
        const m = this._canvasMouse(e);
        let best = null, bestDist = Infinity;
        this.objects.forEach(obj => {
            const sp = this._worldToScreen(obj.center, c2w);
            if (!sp) return;
            const d = Math.hypot(sp.x - m.x, sp.y - m.y);
            const thresh = Math.max(20, 1500 / sp.depth);
            if (d < thresh && sp.depth < bestDist) { bestDist = sp.depth; best = obj; }
        });
        return best;
    }

    _pickTrajPoint(e) {
        if (!this._trajScreenPts) return null;
        const m = this._overlayMouse(e);
        let best = null, bestD = 14;
        this._trajScreenPts.forEach(sp => {
            if (!sp) return;
            const d = Math.hypot(sp.x - m.x, sp.y - m.y);
            if (d < bestD) { bestD = d; best = sp.frame_idx; }
        });
        return best;
    }

    selectTrack(trackId) {
        const changed = this.selectedTrackId !== trackId;
        this.selectedTrackId = trackId;
        const obj = this.objects.find(o => o.track_id === trackId);
        if (obj) this.onObjectSelected(trackId, obj);
        if (changed) {
            this.trajectory = null;
            this._fetchTrajectory(trackId);
            this._invalidateCache();  // 高亮变化需重渲染
        }
        this.requestRender();
    }

    selectByObjectId(trackId) { this.selectTrack(trackId); }

    deselect() {
        this.selectedTrackId = null;
        this.trajectory = null;
        this._invalidateCache();
        this.requestRender();
    }

    async _fetchTrajectory(trackId) {
        if (trackId === null || trackId === undefined) return;
        try {
            const resp = await fetch(`${this.apiBase}/tracks/${this.sceneId}/${trackId}/trajectory`);
            const data = await resp.json();
            if (data && data.success) {
                this.trajectory = data;
                this._drawOverlay();
            }
        } catch (e) { /* ignore */ }
    }

    setTransformMode(mode) {
        this.mode = mode || 'select';
        if (this._editing()) this.trajectoryEditMode = false;  // 互斥
        this._updateHint();
        this.canvas.style.cursor = this._sceneLocked() ? 'crosshair' : 'grab';
    }

    // 切换轨迹编辑状态（需先选中物体）。返回最新状态。
    toggleTrajectoryEdit(on) {
        const want = (on === undefined) ? !this.trajectoryEditMode : on;
        if (want && this.selectedTrackId === null) {
            return false;  // 未选中物体，无法进入
        }
        this.trajectoryEditMode = want;
        if (want) {
            this.mode = 'select';        // 退出移动/旋转
            this.showTrajectory = true;  // 强制显示轨迹
            if (!this.trajectory) this._fetchTrajectory(this.selectedTrackId);
            this._invalidateCache();
        }
        this._updateHint();
        this.canvas.style.cursor = this._sceneLocked() ? 'crosshair' : 'grab';
        this.requestRender();
        return this.trajectoryEditMode;
    }

    toggleBoxes(show) {
        this.showBoxes = (show === undefined) ? !this.showBoxes : show;
        this._invalidateCache();
        this.requestRender();
        return this.showBoxes;
    }

    toggleTrajectory(show) {
        this.showTrajectory = (show === undefined) ? !this.showTrajectory : show;
        this._invalidateCache();
        this.requestRender();
        return this.showTrajectory;
    }

    // ==================== 物体拖动编辑（按 track） ====================

    _dragObject(dx, dy) {
        const obj = this.objects.find(o => o.track_id === this.selectedTrackId);
        if (!obj) return;
        const c2w = this._computeC2W();

        if (this.mode === 'translate') {
            const sp = this._worldToScreen(obj.center, c2w);
            const depth = sp ? sp.depth : this.radius;
            const fy = (this.renderHeight / 2.0) / Math.tan(this._deg2rad(this.fovY) / 2.0);
            const worldPerPixel = depth / fy;
            const rect = this.canvas.getBoundingClientRect();
            const scaleX = this.renderWidth / rect.width;
            const scaleY = this.renderHeight / rect.height;
            const moveX = dx * scaleX * worldPerPixel;
            const moveY = dy * scaleY * worldPerPixel;
            const delta = this._screenDeltaToWorld(c2w, moveX, moveY);
            for (let i = 0; i < 3; i++) {
                obj.center[i] += delta[i];
                obj.pose_world[i][3] = obj.center[i];
            }
        } else if (this.mode === 'rotate') {
            const angle = dx * 0.02;
            this._rotatePoseY(obj.pose_world, angle);
        }
        // 拖动时本帧需重渲染（相机不变，但物体动了）
        this.frameCache.delete(this.frameIdx);
        this.requestRender();
    }

    // 把屏幕位移（右/下方向的世界量）转换为世界位移；默认锁定高度（地面 XZ 平面移动）
    _screenDeltaToWorld(c2w, moveX, moveY) {
        const right = [c2w[0][0], c2w[1][0], c2w[2][0]];
        const camDown = [c2w[0][1], c2w[1][1], c2w[2][1]];
        let delta = [
            right[0] * moveX + camDown[0] * moveY,
            right[1] * moveX + camDown[1] * moveY,
            right[2] * moveX + camDown[2] * moveY,
        ];
        if (this.lockHeight) {
            // 地面为世界 XZ 平面（Y 为竖直方向，车辆不起飞）
            // 把 delta 投影到地面：去掉竖直分量，并按需放大水平分量以匹配垂直拖拽幅度
            delta[1] = 0;
        }
        return delta;
    }

    _rotatePoseY(pose, angle) {
        const c = Math.cos(angle), s = Math.sin(angle);
        const Ry = [[c, 0, s], [0, 1, 0], [-s, 0, c]];
        for (let col = 0; col < 3; col++) {
            const a = pose[0][col], b = pose[1][col], d = pose[2][col];
            pose[0][col] = Ry[0][0]*a + Ry[0][1]*b + Ry[0][2]*d;
            pose[1][col] = Ry[1][0]*a + Ry[1][1]*b + Ry[1][2]*d;
            pose[2][col] = Ry[2][0]*a + Ry[2][1]*b + Ry[2][2]*d;
        }
    }

    async _commitObjectEdit() {
        const obj = this.objects.find(o => o.track_id === this.selectedTrackId);
        if (!obj) return;
        try {
            await fetch(`${this.apiBase}/edit/object/matrix`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    scene_id: this.sceneId,
                    track_id: obj.track_id,
                    frame_idx: this.frameIdx,
                    pose_matrix: obj.pose_world
                })
            });
            obj.edited = true;
            this._invalidateCache();              // 旧缓存失效
            this._fetchTrajectory(obj.track_id);  // 编辑后轨迹会变
            this.onObjectEdited(obj.track_id, obj.pose_world);
            this.requestRender();
        } catch (e) {
            console.error('[Viewer3D] 回写编辑失败:', e);
        }
    }

    // ==================== 轨迹点拖动编辑 ====================

    _dragTrajPoint(e) {
        if (!this.trajectory) return;
        const fi = this.draggingTrajPoint;
        const pt = this.trajectory.trajectory.find(p => p.frame_idx === fi);
        if (!pt) return;
        const c2w = this._computeC2W();
        const sp = this._worldToScreen(pt.center, c2w);
        const depth = sp ? sp.depth : this.radius;
        const fy = (this.renderHeight / 2.0) / Math.tan(this._deg2rad(this.fovY) / 2.0);
        const worldPerPixel = depth / fy;
        const dx = e.clientX - this.prevMouse.x;
        const dy = e.clientY - this.prevMouse.y;
        const rect = this.canvas.getBoundingClientRect();
        const scaleX = this.renderWidth / rect.width;
        const scaleY = this.renderHeight / rect.height;
        const moveX = dx * scaleX * worldPerPixel;
        const moveY = dy * scaleY * worldPerPixel;
        const delta = this._screenDeltaToWorld(c2w, moveX, moveY);
        for (let i = 0; i < 3; i++) {
            pt.center[i] += delta[i];
        }
        // 若拖的是当前帧的点，同步更新物体显示并重渲染
        if (fi === this.frameIdx) {
            const obj = this.objects.find(o => o.track_id === this.selectedTrackId);
            if (obj) {
                obj.center = pt.center.slice();
                for (let i = 0; i < 3; i++) obj.pose_world[i][3] = pt.center[i];
            }
            this.frameCache.delete(this.frameIdx);
            this.requestRender();
        } else {
            this._drawOverlay();
        }
    }

    async _commitTrajPoint(frameIdx) {
        if (!this.trajectory) return;
        const pt = this.trajectory.trajectory.find(p => p.frame_idx === frameIdx);
        if (!pt) return;
        try {
            await fetch(`${this.apiBase}/edit/track/point`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    scene_id: this.sceneId,
                    track_id: this.selectedTrackId,
                    frame_idx: frameIdx,
                    center: pt.center
                })
            });
            this.onObjectEdited(this.selectedTrackId, null);
            this._invalidateCache();
            this._fetchTrajectory(this.selectedTrackId);
        } catch (e) {
            console.error('[Viewer3D] 轨迹点回写失败:', e);
        }
    }

    // ==================== 删除 ====================

    async deleteSelected() {
        if (this.selectedTrackId === null) return;
        const tid = this.selectedTrackId;
        try {
            await fetch(`${this.apiBase}/edit/track/delete`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ scene_id: this.sceneId, track_id: tid })
            });
            this.objects = this.objects.filter(o => o.track_id !== tid);
            this.selectedTrackId = null;
            this.trajectory = null;
            this._invalidateCache();
            this.onObjectEdited(tid, null);
            this.requestRender();
        } catch (e) {
            console.error('[Viewer3D] 删除失败:', e);
        }
    }

    dispose() {
        this.active = false;
        [this.canvas, this.overlay, this.hint].forEach(el => {
            if (el && el.parentNode) el.parentNode.removeChild(el);
        });
        this.initialized = false;
    }
}

window.DGGTViewer3D = DGGTViewer3D;
