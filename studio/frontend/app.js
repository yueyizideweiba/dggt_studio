const API_BASE = 'http://localhost:8000/api';

// 实验台：10 种事故类型（值, 中文名）
const LAB_SCENARIOS = [
    ['rear-end', '追尾'], ['head-on', '对向碰撞'], ['intersection-tbone', '路口侧碰'],
    ['lane-change-cutin', '变道加塞'], ['hard-brake', '紧急刹车'], ['pedestrian-crossing', '行人横穿'],
    ['cut-out-reveal', '前车闪开'], ['chain-reaction-rear-end', '连环追尾'],
    ['cutin-brake-pileup', '加塞急刹连环'], ['occluded-pedestrian-pileup', '遮挡行人连环']
];

class DGGTStudio {
    constructor() {
        this.state = {
            sceneId: null,
            scenePath: null,
            totalFrames: 0,      // 可渲染帧数（可延长）
            dataFrames: 0,       // 磁盘上真实存在的帧数
            currentFrame: 0,
            objects: [],
            selectedObjects: [],
            hoveredObject: null,
            editingTrajectory: null,
            isPlaying: false,
            playInterval: null,
            zoom: 1.0,
            pan: { x: 0, y: 0 },
            tool: 'select',
            showTrajectories: true,
            showGrid: false,
            showBoundingBoxes: true,
            showObjectIds: true,
            editHistory: [],
            redoHistory: [],
            historyIndex: -1,
            renderCache: new Map(),
            cacheSize: 10,
            cameraParams: null,
            lastRenderTime: 0,
            frameCount: 0,
            fps: 0,
            viewMode: '2d',  // '2d' | '3d'
            lastQualityReport: null,
            lastSamplingParams: null
        };

        this.canvas = null;
        this.ctx = null;
        this.currentImage = null;
        this.viewer3d = null;  // 3D 视图实例（懒加载）

        // SAM 3D 交互重建状态
        this.samPoints = [];           // [{x, y, label:'fg'|'bg'}]（源图像素坐标）
        this.samPointType = 'fg';      // 当前点类型
        this.samMask = null;           // 掩码 base64
        this.samMaskImage = null;      // 掩码 Image
        this.samSrcImage = null;       // 高清源图 Image
        this.samSrcWidth = 0;
        this.samSrcHeight = 0;
        this._sam3dSegTimer = null;
        this.lastSamObjectId = null;   // 最近生成的物体 id
        this.samPreviewFrames = [];    // 预览帧 base64 列表
        this.samPreviewTimer = null;
        this.samPreviewIdx = 0;

        // 选中物体的全局旋转（持久 360°）
        this.objectRotation = { yaw: 0, pitch: 0, roll: 0 };
        this._rotationLoadedFor = null;
        this._rotationTimer = null;

        this.init();
    }

    init() {
        this.canvas = document.getElementById('mainCanvas');
        this.ctx = this.canvas.getContext('2d');
        
        this.setupEventListeners();
        this.setupCanvasInteractions();
        this.setupKeyboardShortcuts();
        this.setupToolButtons();
        this.setupViewModeToggle();
        this.setupSam3dControls();
        this.setupNlEntityControls();
        this.setupAutoHeadingControls();
        this.setupLabControls();
        
        this.updateStatus('就绪 - 请加载场景开始编辑');
        this.renderEmptyCanvas();
    }


    // ==================== 视图模式切换 (2D / 3D) ====================

    setupViewModeToggle() {
        const view2dBtn = document.getElementById('view2dBtn');
        const view3dBtn = document.getElementById('view3dBtn');
        if (view2dBtn) view2dBtn.addEventListener('click', () => this.setViewMode('2d'));
        if (view3dBtn) view3dBtn.addEventListener('click', () => this.setViewMode('3d'));

        // 3D 视图工具栏
        const v3dSelect = document.getElementById('v3dSelectBtn');
        const v3dTranslate = document.getElementById('v3dTranslateBtn');
        const v3dRotate = document.getElementById('v3dRotateBtn');
        const v3dResetView = document.getElementById('v3dResetViewBtn');
        const v3dFocus = document.getElementById('v3dFocusBtn');

        if (v3dSelect) v3dSelect.addEventListener('click', () => this.set3dTool('select'));
        if (v3dTranslate) v3dTranslate.addEventListener('click', () => this.set3dTool('translate'));
        if (v3dRotate) v3dRotate.addEventListener('click', () => this.set3dTool('rotate'));
        if (v3dResetView) v3dResetView.addEventListener('click', () => {
            if (this.viewer3d) this.viewer3d.resetView();
        });
        if (v3dFocus) v3dFocus.addEventListener('click', () => {
            if (this.viewer3d) this.viewer3d.focusSelected();
        });

        const v3dBoxes = document.getElementById('v3dBoxesBtn');
        if (v3dBoxes) v3dBoxes.addEventListener('click', () => {
            if (!this.viewer3d) return;
            const on = this.viewer3d.toggleBoxes();
            v3dBoxes.classList.toggle('active', on);
        });

        const v3dTraj = document.getElementById('v3dTrajBtn');
        if (v3dTraj) v3dTraj.addEventListener('click', () => {
            if (!this.viewer3d) return;
            const on = this.viewer3d.toggleTrajectory();
            v3dTraj.classList.toggle('active', on);
        });

        const v3dRoad = document.getElementById('v3dRoadBtn');
        if (v3dRoad) v3dRoad.addEventListener('click', () => {
            if (!this.viewer3d) return;
            if (!this.viewer3d.roadMap) {
                const why = (this.viewer3d.roadMapStatus && this.viewer3d.roadMapStatus.reason) || '本场景没有高精地图';
                this.updateStatus(`无法显示道路：${why}`);
                return;
            }
            const on = this.viewer3d.toggleRoadMap();
            v3dRoad.classList.toggle('active', on);
        });

        const v3dTrajEdit = document.getElementById('v3dTrajEditBtn');
        if (v3dTrajEdit) v3dTrajEdit.addEventListener('click', () => {
            if (!this.viewer3d) return;
            if (this.viewer3d.selectedTrackId === null && !this.viewer3d.trajectoryEditMode) {
                this.updateStatus('请先选中一个物体再编辑轨迹');
                return;
            }
            const on = this.viewer3d.toggleTrajectoryEdit();
            v3dTrajEdit.classList.toggle('active', on);
            // 进入轨迹编辑时，取消移动/旋转按钮高亮
            if (on) {
                ['v3dSelectBtn', 'v3dTranslateBtn', 'v3dRotateBtn'].forEach(id => {
                    const b = document.getElementById(id);
                    if (b) b.classList.remove('active');
                });
                document.getElementById('v3dSelectBtn')?.classList.add('active');
            }
        });

        const v3dAdaptive = document.getElementById('v3dAdaptiveBtn');
        if (v3dAdaptive) v3dAdaptive.addEventListener('click', () => {
            if (!this.viewer3d) return;
            const on = !this.viewer3d.adaptiveTrajEdit;
            this.viewer3d.setAdaptiveEdit(on);
            v3dAdaptive.classList.toggle('active', on);
            this.updateStatus(on ? '智能轨迹已开启：拖动节点整条路径自适应跟随' : '智能轨迹已关闭：仅调整单个节点');
        });

        const v3dSmooth = document.getElementById('v3dSmoothBtn');
        if (v3dSmooth) v3dSmooth.addEventListener('click', () => {
            if (!this.viewer3d) return;
            if (this.viewer3d.selectedTrackId === null) {
                this.updateStatus('请先选中一个物体再平滑轨迹');
                return;
            }
            this.viewer3d.smoothSelectedTrajectory(0.5);
            this.updateStatus('已平滑选中物体的轨迹');
        });

        const v3dDelete = document.getElementById('v3dDeleteBtn');
        if (v3dDelete) v3dDelete.addEventListener('click', () => {
            if (!this.viewer3d) return;
            if (this.viewer3d.selectedTrackId === null) {
                this.updateStatus('请先选中一个物体');
                return;
            }
            if (confirm('确定删除选中物体（整段轨迹）?')) {
                this.viewer3d.deleteSelected();
            }
        });

    }

    ensureViewer3d() {
        if (this.viewer3d) return this.viewer3d;
        const container = document.getElementById('view3dContainer');
        this.viewer3d = new DGGTViewer3D(container, {
            apiBase: API_BASE,
            onObjectSelected: (trackId, asset) => {
                // 若 Corner Case 面板正在等待为某角色指派物体，则优先指派
                this.onTrackSelectedForCorner(trackId);
                // 「替换目标物体」拾取模式：点击即选定替换目标
                this._sam3dFinishPickTarget(trackId);
                this.state.selectedObjects = [trackId];
                // 用 3D 物体信息更新属性面板（以 track_id 为标识）
                let obj = this.state.objects.find(o => o.track_id === trackId);
                if (!obj && asset) {
                    obj = {
                        track_id: trackId,
                        object_id: trackId,
                        raw_object_id: asset.raw_object_id,
                        pose_world: asset.pose_world,
                        dimensions: asset.dimensions,
                        type: asset.type,
                        edited: asset.edited
                    };
                    this.state.objects.push(obj);
                }
                this.updateSelectedObjectPanel();
                this.updateObjectList();
            },
            onObjectEdited: (trackId) => {
                // 编辑后清空 2D 渲染缓存，保证切回 2D 时显示最新结果
                this.state.renderCache.clear();
                // 强制重新读取该物体的全局旋转，刷新旋转滑杆数值
                this._rotationLoadedFor = null;
                this.updateSelectedObjectPanel();
                this.saveEditHistory('move', trackId);
            }
        });
        this.viewer3d.init();
        return this.viewer3d;
    }

    setViewMode(mode) {
        if (mode === this.state.viewMode) return;
        this.state.viewMode = mode;

        const view2dBtn = document.getElementById('view2dBtn');
        const view3dBtn = document.getElementById('view3dBtn');
        const container3d = document.getElementById('view3dContainer');
        const toolbar3d = document.getElementById('view3dToolbar');
        const canvas2d = document.querySelector('.canvas-wrapper #mainCanvas');
        const overlay2d = document.getElementById('objectOverlay');
        const trajectorySvg = document.getElementById('trajectorySvg');
        const canvasToolbar = document.querySelector('.canvas-toolbar');

        if (mode === '3d') {
            view2dBtn.classList.remove('active');
            view3dBtn.classList.add('active');
            if (canvas2d) canvas2d.style.display = 'none';
            if (overlay2d) overlay2d.style.display = 'none';
            if (trajectorySvg) trajectorySvg.style.display = 'none';
            if (canvasToolbar) canvasToolbar.style.display = 'none';
            if (container3d) container3d.style.display = 'block';
            if (toolbar3d) toolbar3d.style.display = 'flex';

            const viewer = this.ensureViewer3d();
            viewer.show();
            if (this.state.sceneId) {
                viewer.loadFrame(this.state.sceneId, this.state.currentFrame);
            }
            this.updateStatus('3D 视图 - 拖动旋转视角，点击选中物体');
        } else {
            view3dBtn.classList.remove('active');
            view2dBtn.classList.add('active');
            if (container3d) container3d.style.display = 'none';
            if (toolbar3d) toolbar3d.style.display = 'none';
            if (canvas2d) canvas2d.style.display = 'block';
            if (overlay2d) overlay2d.style.display = 'block';
            if (trajectorySvg) trajectorySvg.style.display = 'block';
            if (canvasToolbar) canvasToolbar.style.display = 'flex';
            if (this.viewer3d) this.viewer3d.hide();
            // 切回 2D 时刷新当前帧以显示最新编辑
            if (this.state.sceneId) {
                this.clearFrameCache(this.state.currentFrame);
                this.loadFrame(this.state.currentFrame);
            }
            this.updateStatus('2D 渲染视图');
        }
    }

    set3dTool(tool) {
        if (!this.viewer3d) return;
        const map = { select: 'v3dSelectBtn', translate: 'v3dTranslateBtn', rotate: 'v3dRotateBtn' };
        // 仅在 select/translate/rotate 这组按钮间切换 active
        Object.values(map).forEach(id => {
            const b = document.getElementById(id);
            if (b) b.classList.remove('active');
        });
        const btn = document.getElementById(map[tool]);
        if (btn) btn.classList.add('active');

        // 切换工具时退出轨迹编辑状态
        if (this.viewer3d.trajectoryEditMode) {
            this.viewer3d.toggleTrajectoryEdit(false);
            document.getElementById('v3dTrajEditBtn')?.classList.remove('active');
        }

        if (tool === 'select') {
            this.viewer3d.setTransformMode(null);
        } else {
            this.viewer3d.setTransformMode(tool);
        }
    }


    // ==================== 工具按钮设置 ====================

    setupToolButtons() {
        const toolButtonMap = {
            'selectToolBtn': 'select',
            'moveToolBtn': 'move',
            'rotateToolBtn': 'rotate',
            'trajectoryToolBtn': 'trajectory'
        };

        Object.entries(toolButtonMap).forEach(([btnId, tool]) => {
            const btn = document.getElementById(btnId);
            if (btn) {
                btn.addEventListener('click', () => this.setTool(tool));
            }
        });

        // 显示/隐藏轨迹按钮
        const drawTrajectoryBtn = document.getElementById('drawTrajectoryBtn');
        if (drawTrajectoryBtn) {
            drawTrajectoryBtn.addEventListener('click', () => this.toggleTrajectories());
            this.updateToggleButton(drawTrajectoryBtn, this.state.showTrajectories);
        }

        // 显示/隐藏网格按钮
        const gridBtn = document.getElementById('gridBtn');
        if (gridBtn) {
            gridBtn.addEventListener('click', () => this.toggleGrid());
            this.updateToggleButton(gridBtn, this.state.showGrid);
        }
    }

    updateToggleButton(btn, isActive) {
        if (isActive) {
            btn.classList.add('active');
        } else {
            btn.classList.remove('active');
        }
    }


    // ==================== 事件监听设置 ====================

    setupEventListeners() {
        document.getElementById('loadSceneBtn').addEventListener('click', () => this.showLoadSceneModal());
        document.getElementById('exportBtn').addEventListener('click', () => this.showExportModal());
        
        document.getElementById('prevFrame').addEventListener('click', () => this.prevFrame());
        document.getElementById('nextFrame').addEventListener('click', () => this.nextFrame());
        // 顶部帧数拖动条：拖动时即时更新帧号（渲染做节流，避免拖一次发几百个请求）
        const frameSlider = document.getElementById('frameSlider');
        if (frameSlider) {
            frameSlider.addEventListener('input', (e) => {
                this._scrubToFrame(parseInt(e.target.value, 10) || 0);
            });
            frameSlider.addEventListener('change', (e) => {
                this._scrubToFrame(parseInt(e.target.value, 10) || 0, true);
            });
        }
        document.getElementById('playBtn').addEventListener('click', () => this.togglePlay());
        document.getElementById('frameInput').addEventListener('change', (e) => {
            const frame = parseInt(e.target.value);
            if (frame >= 0 && frame < this.state.totalFrames) {
                this.jumpToFrame(frame);
            }
        });
        
        // 时间轴：按需延长（原场景帧数不再是上限）
        const _extBtn = document.getElementById('tlExtendBtn');
        if (_extBtn) _extBtn.addEventListener('click', () => {
            const v = parseInt(document.getElementById('tlExtendInput').value);
            this.extendTimeline(isNaN(v) ? 50 : v);
        });
        document.querySelectorAll('.timeline-extend [data-extra]').forEach(b => {
            b.addEventListener('click', () => this.extendTimeline(parseInt(b.dataset.extra)));
        });

        document.getElementById('zoomInBtn').addEventListener('click', () => this.zoomIn());
        document.getElementById('zoomOutBtn').addEventListener('click', () => this.zoomOut());
        document.getElementById('fitToViewBtn').addEventListener('click', () => this.fitToView());
        
        document.getElementById('undoBtn').addEventListener('click', () => this.undo());
        document.getElementById('redoBtn').addEventListener('click', () => this.redo());
        
        document.querySelectorAll('.modal-close, .modal-cancel').forEach(btn => {
            btn.addEventListener('click', (e) => {
                e.target.closest('.modal').classList.remove('active');
            });
        });
        
        document.getElementById('confirmLoadSceneBtn').addEventListener('click', () => this.loadScene());
        document.getElementById('confirmExportBtn').addEventListener('click', () => this.exportScene());
        
        document.getElementById('objectSearch').addEventListener('input', (e) => {
            this.filterObjects(e.target.value);
        });

        // 初始化 Corner Case 面板
        this.initCornerCasePanel();
    }


    // ==================== 画布交互 ====================

    setupCanvasInteractions() {
        let isPanning = false;
        let panStart = { x: 0, y: 0 };
        let isDraggingObject = false;
        let selectedObject = null;
        let lastDragPos = { x: 0, y: 0 };
        
        this.canvas.addEventListener('mousedown', (e) => {
            const rect = this.canvas.getBoundingClientRect();
            const canvasX = e.clientX - rect.left;
            const canvasY = e.clientY - rect.top;
            
            if (this.state.tool === 'select' || this.state.tool === 'move' || this.state.tool === 'rotate') {
                const imageCoords = this.canvasToImageCoords(canvasX, canvasY);
                const clickedObject = this.findObjectAtPosition(imageCoords.x, imageCoords.y);
                
                if (clickedObject) {
                    selectedObject = clickedObject;
                    isDraggingObject = true;
                    lastDragPos = { x: canvasX, y: canvasY };
                    // 「替换目标物体」拾取模式：点击即选定目标
                    this._sam3dFinishPickTarget(clickedObject.object_id);
                    
                    if (!e.shiftKey) {
                        this.state.selectedObjects = [clickedObject.object_id];
                    } else {
                        if (!this.state.selectedObjects.includes(clickedObject.object_id)) {
                            this.state.selectedObjects.push(clickedObject.object_id);
                        }
                    }
                    
                    this.updateSelectedObjectPanel();
                    this.renderOverlay();
                } else {
                    if (!e.shiftKey) {
                        this.state.selectedObjects = [];
                    }
                    this.updateSelectedObjectPanel();
                    this.renderOverlay();
                }
            } else if (this.state.tool === 'trajectory') {
                const imageCoords = this.canvasToImageCoords(canvasX, canvasY);
                const clickedObject = this.findObjectAtPosition(imageCoords.x, imageCoords.y);
                if (clickedObject) {
                    this.showTrajectoryEditor(clickedObject.object_id);
                }
            }
        });
        
        this.canvas.addEventListener('mousemove', (e) => {
            const rect = this.canvas.getBoundingClientRect();
            const canvasX = e.clientX - rect.left;
            const canvasY = e.clientY - rect.top;
            
            if (isDraggingObject && selectedObject) {
                if (this.state.tool === 'move') {
                    const dx = canvasX - lastDragPos.x;
                    const dy = canvasY - lastDragPos.y;
                    const worldOffset = this.screenToWorldOffset(dx, dy, selectedObject);
                    if (worldOffset) {
                        this.updateObjectPosition(selectedObject.object_id, worldOffset);
                    }
                    lastDragPos = { x: canvasX, y: canvasY };
                } else if (this.state.tool === 'rotate') {
                    const centerX = this.canvas.width / 2;
                    const centerY = this.canvas.height / 2;
                    const angle1 = Math.atan2(lastDragPos.y - centerY, lastDragPos.x - centerX);
                    const angle2 = Math.atan2(canvasY - centerY, canvasX - centerX);
                    const deltaAngle = (angle2 - angle1) * 180 / Math.PI;
                    this.updateObjectRotation(selectedObject.object_id, deltaAngle);
                    lastDragPos = { x: canvasX, y: canvasY };
                }
            } else {
                const imageCoords = this.canvasToImageCoords(canvasX, canvasY);
                const hoveredObj = this.findObjectAtPosition(imageCoords.x, imageCoords.y);
                if (hoveredObj !== this.state.hoveredObject) {
                    this.state.hoveredObject = hoveredObj;
                    this.renderOverlay();
                }
            }
        });
        
        this.canvas.addEventListener('mouseup', () => {
            if (isDraggingObject && selectedObject) {
                this.saveEditHistory(this.state.tool, selectedObject.object_id);
            }
            isDraggingObject = false;
            selectedObject = null;
            this.updateCursor();
        });
        
        this.canvas.addEventListener('wheel', (e) => {
            e.preventDefault();
            const delta = e.deltaY > 0 ? -0.1 : 0.1;
            this.state.zoom = Math.max(0.1, Math.min(5.0, this.state.zoom + delta));
            this.updateZoomDisplay();
            this.renderImage();
            this.renderOverlay();
        });
    }


    // ==================== 键盘快捷键 ====================

    setupKeyboardShortcuts() {
        document.addEventListener('keydown', (e) => {
            if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
            
            switch(e.key) {
                case 'Delete':
                case 'Backspace':
                    if (this.state.viewMode === '3d' && this.viewer3d && this.viewer3d.selectedTrackId !== null) {
                        e.preventDefault();
                        if (confirm('确定删除选中物体（整段轨迹）?')) this.viewer3d.deleteSelected();
                    } else if (this.state.selectedObjects.length > 0) {
                        e.preventDefault();
                        this.deleteSelectedObjects();
                    }
                    break;
                case 'ArrowLeft':
                    e.preventDefault();
                    this.prevFrame();
                    break;
                case 'ArrowRight':
                    e.preventDefault();
                    this.nextFrame();
                    break;
                case ' ':
                    e.preventDefault();
                    this.togglePlay();
                    break;
                case 'v':
                case 'V':
                    this.setTool('select');
                    break;
                case 'm':
                case 'M':
                    this.setTool('move');
                    break;
                case 'r':
                case 'R':
                    this.setTool('rotate');
                    break;
                case 't':
                case 'T':
                    this.setTool('trajectory');
                    break;
                case 'g':
                case 'G':
                    this.toggleGrid();
                    break;
                case 'b':
                case 'B':
                    this.toggleTrajectories();
                    break;
            }
        });
        
        // Ctrl+Z 撤销, Ctrl+Y 重做
        document.addEventListener('keydown', (e) => {
            if (e.ctrlKey || e.metaKey) {
                if (e.key === 'z') {
                    e.preventDefault();
                    this.undo();
                } else if (e.key === 'y') {
                    e.preventDefault();
                    this.redo();
                }
            }
        });
    }


    // ==================== 工具和视图控制 ====================

    setTool(tool) {
        this.state.tool = tool;
        document.querySelectorAll('.tool-btn').forEach(btn => btn.classList.remove('active'));
        const toolBtn = document.getElementById(`${tool}ToolBtn`);
        if (toolBtn) {
            toolBtn.classList.add('active');
        }
        this.updateCursor();
    }

    updateCursor() {
        switch (this.state.tool) {
            case 'select':
                this.canvas.style.cursor = 'default';
                break;
            case 'move':
                this.canvas.style.cursor = 'move';
                break;
            case 'rotate':
                this.canvas.style.cursor = 'crosshair';
                break;
            case 'trajectory':
                this.canvas.style.cursor = 'pointer';
                break;
            default:
                this.canvas.style.cursor = 'default';
        }
    }

    toggleTrajectories() {
        this.state.showTrajectories = !this.state.showTrajectories;
        const btn = document.getElementById('drawTrajectoryBtn');
        this.updateToggleButton(btn, this.state.showTrajectories);
        if (this.state.sceneId) {
            this.loadFrame(this.state.currentFrame);
        }
    }

    toggleGrid() {
        this.state.showGrid = !this.state.showGrid;
        const btn = document.getElementById('gridBtn');
        this.updateToggleButton(btn, this.state.showGrid);
        this.renderImage();
        this.renderOverlay();
    }

    zoomIn() {
        this.state.zoom = Math.min(5.0, this.state.zoom + 0.2);
        this.updateZoomDisplay();
        this.renderImage();
        this.renderOverlay();
    }

    zoomOut() {
        this.state.zoom = Math.max(0.1, this.state.zoom - 0.2);
        this.updateZoomDisplay();
        this.renderImage();
        this.renderOverlay();
    }

    fitToView() {
        this.state.zoom = 1.0;
        this.state.pan = { x: 0, y: 0 };
        this.updateZoomDisplay();
        this.renderImage();
        this.renderOverlay();
    }

    updateZoomDisplay() {
        document.getElementById('zoomLevel').textContent = `${Math.round(this.state.zoom * 100)}%`;
    }


    // ==================== 坐标转换 ====================

    canvasToImageCoords(canvasX, canvasY) {
        const imageX = (canvasX / this.state.zoom) - this.state.pan.x;
        const imageY = (canvasY / this.state.zoom) - this.state.pan.y;
        return { x: imageX, y: imageY };
    }

    imageToCanvasCoords(imageX, imageY) {
        const canvasX = (imageX + this.state.pan.x) * this.state.zoom;
        const canvasY = (imageY + this.state.pan.y) * this.state.zoom;
        return { x: canvasX, y: canvasY };
    }

    screenToWorldOffset(dx, dy, object) {
        if (!this.state.cameraParams) {
            const pixelsPerMeter = 50;
            return { 
                dx: dx / (pixelsPerMeter * this.state.zoom), 
                dy: 0, 
                dz: dy / (pixelsPerMeter * this.state.zoom) 
            };
        }
        
        const K = this.state.cameraParams.intrinsics;
        const fx = K[0][0];
        const fy = K[1][1];
        const depth = 30.0;
        
        const worldDx = (dx / fx) * depth / this.state.zoom;
        const worldDz = (dy / fy) * depth / this.state.zoom;
        
        return { dx: worldDx, dy: 0, dz: worldDz };
    }


    // ==================== 场景管理 ====================

    showLoadSceneModal() {
        document.getElementById('loadSceneModal').classList.add('active');
    }

    async loadScene() {
        const scenePath = document.getElementById('scenePathInput').value;
        const loadSky = document.getElementById('loadSkyCheckbox').checked;
        const staticOnly = document.getElementById('staticOnlyCheckbox').checked;
        
        if (!scenePath) {
            alert('请输入场景路径');
            return;
        }
        
        this.updateStatus('正在加载场景...');
        
        try {
            const response = await fetch(`${API_BASE}/scenes/load`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    scene_path: scenePath,
                    load_sky: loadSky,
                    static_only: staticOnly
                })
            });
            
            const data = await response.json();
            
            if (data.success) {
                this.state.sceneId = data.scene.scene_id;
                this.state.scenePath = data.scene.scene_path;
                this.state.totalFrames = data.scene.num_frames;
                this.state.currentFrame = 0;
                this.cornerAffectedTracks = [];
                this.cornerSynthesizedTracks = [];
                this.state.lastQualityReport = null;
                this.state.lastSamplingParams = null;
                this._updateQualityReportButton();
                this.state.renderCache.clear();
                
                document.getElementById('scenePath').textContent = this.state.scenePath;
                document.getElementById('sceneIdDisplay').textContent = this.state.sceneId;
                document.getElementById('sceneTotalFrames').textContent = this.state.totalFrames;
                document.getElementById('totalFrames').textContent = `/ ${this.state.totalFrames}`;
                this.state.dataFrames = data.scene.data_frames || data.scene.num_frames;
                this._refreshTimelineUI();
                
                document.getElementById('loadSceneModal').classList.remove('active');
                
                await this.loadFrame(this.state.currentFrame);
                this._autoHeadingLoad();
                this.loadEgoInfo();
                this.loadSceneMap();          // 高精地图（道路）图层
                this.updateStatus(`场景已加载: ${this.state.sceneId}`);
            }
        } catch (error) {
            console.error('加载场景失败:', error);
            this.updateStatus('加载场景失败');
            alert('加载场景失败: ' + error.message);
        }
    }


    // 高精地图（道路）：取回后交给 3D 视图叠加，并在场景信息里说明
    async loadSceneMap() {
        this.state.sceneMap = null;
        if (!this.state.sceneId) return;
        try {
            const r = await fetch(`${API_BASE}/scene/map/${this.state.sceneId}`);
            const d = await r.json();
            if (d.available) {
                this.state.sceneMap = d;
                if (this.viewer3d) this.viewer3d.setRoadMap(d);
                const n = d.counts || {};
                const rms = d.align && d.align.rms_m != null ? d.align.rms_m.toFixed(2) : '-';
                this._setMapStatus(
                    `已接入 ${n.lane || 0} 车道 / ${n.road_line || 0} 标线 / `
                    + `${n.road_edge || 0} 边界 / ${n.crosswalk || 0} 斑马线（对齐 ${rms} m）`, false);
            } else {
                if (this.viewer3d) this.viewer3d.setRoadMap({ available: false });
                this._setMapStatus(d.reason || '本场景没有高精地图', true);
            }
        } catch (e) {
            console.warn('加载高精地图失败', e);
            if (this.viewer3d) this.viewer3d.setRoadMap({ available: false });
            this._setMapStatus('加载失败（后端未响应）', true);
        }
        const btn = document.getElementById('v3dRoadBtn');
        if (btn) btn.classList.toggle('active', !!(this.viewer3d && this.viewer3d.showRoadMap && this.viewer3d.roadMap));
    }

    _setMapStatus(text, bad) {
        const el = document.getElementById('sceneMapStatus');
        if (!el) return;
        el.textContent = text;
        el.style.color = bad ? '#ff9a6b' : '#8fe38f';
    }

    async loadFrame(frameIdx) {
        if (!this.state.sceneId) return;
        // 实验台打开且停在"关系图"页时，切帧自动刷新关系图
        this._labMaybeAutoRefresh(frameIdx);

        // 3D 模式下把任务转给 viewer3d
        if (this.state.viewMode === '3d' && this.viewer3d) {
            this.viewer3d.frameIdx = frameIdx;
            await this.viewer3d.loadFrame(this.state.sceneId, frameIdx, true);
            // 用 viewer 已获取的物体列表更新左侧面板（以 track_id 标识）
            this.state.objects = (this.viewer3d.objects || []).map(o => ({
                track_id: o.track_id,
                object_id: o.track_id,
                raw_object_id: o.raw_object_id,
                pose_world: o.pose_world,
                dimensions: o.dimensions,
                type: o.type,
                edited: o.edited,
                bbox_2d: null
            }));
            this.updateObjectList();
            this.updateObjectCount();
            return;
        }
        
        // 检查缓存
        const cacheKey = `${frameIdx}_${this.state.showTrajectories}_${this.state.showBoundingBoxes}`;
        if (this.state.renderCache.has(cacheKey)) {
            const cached = this.state.renderCache.get(cacheKey);
            this.currentImage = cached.image;
            this.state.objects = cached.objects;
            this.state.cameraParams = cached.cameraParams;
            this.renderImage();
            this.renderOverlay();
            this.updateObjectList();
            this.updateObjectCount();
            return;
        }
        
        try {
            const renderResponse = await fetch(`${API_BASE}/render/frame`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    scene_id: this.state.sceneId,
                    frame_idx: frameIdx,
                    draw_bboxes: this.state.showBoundingBoxes,
                    draw_ids: this.state.showObjectIds,
                    draw_trajectories: this.state.showTrajectories,
                    trajectory_length: 15
                })
            });
            
            const renderData = await renderResponse.json();
            
            if (renderData.success) {
                this.currentImage = renderData.image;
                this.state.objects = renderData.objects || [];
                
                if (renderData.camera_params) {
                    this.state.cameraParams = renderData.camera_params;
                }
                
                // LRU缓存
                if (this.state.renderCache.size >= this.state.cacheSize) {
                    const firstKey = this.state.renderCache.keys().next().value;
                    this.state.renderCache.delete(firstKey);
                }
                this.state.renderCache.set(cacheKey, {
                    image: this.currentImage,
                    objects: this.state.objects,
                    cameraParams: this.state.cameraParams
                });
                
                this.renderImage();
                this.renderOverlay();
                this.updateObjectList();
                this.updateObjectCount();
                this.updateFPS();
            }
        } catch (error) {
            console.error('加载帧失败:', error);
            this.updateStatus('加载帧失败: ' + error.message);
        }
    }


    renderEmptyCanvas() {
        this.ctx.fillStyle = '#1a1a2e';
        this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
        
        if (this.state.showGrid) {
            this.drawGrid();
        }
        
        this.ctx.fillStyle = '#666';
        this.ctx.font = '16px Inter, sans-serif';
        this.ctx.textAlign = 'center';
        this.ctx.fillText('请加载场景开始编辑', this.canvas.width / 2, this.canvas.height / 2);
    }

    renderImage() {
        if (!this.currentImage) {
            this.renderEmptyCanvas();
            return;
        }
        
        const img = new Image();
        img.onload = () => {
            this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
            this.ctx.save();
            this.ctx.scale(this.state.zoom, this.state.zoom);
            this.ctx.translate(this.state.pan.x, this.state.pan.y);
            this.ctx.drawImage(img, 0, 0, this.canvas.width, this.canvas.height);
            this.ctx.restore();
            
            if (this.state.showGrid) {
                this.drawGrid();
            }
        };
        img.src = `data:image/png;base64,${this.currentImage}`;
    }

    renderOverlay() {
        const overlay = document.getElementById('objectOverlay');
        if (!overlay) return;
        
        overlay.innerHTML = '';
        
        if (!this.state.objects || this.state.objects.length === 0) return;
        
        this.state.objects.forEach(obj => {
            if (!obj.bbox_2d) return;
            
            const bbox = obj.bbox_2d;
            const isSelected = this.state.selectedObjects.includes(obj.object_id);
            const isHovered = this.state.hoveredObject && this.state.hoveredObject.object_id === obj.object_id;
            
            if (isSelected || isHovered) {
                const div = document.createElement('div');
                div.className = 'object-highlight';
                div.style.cssText = `
                    position: absolute;
                    left: ${bbox.x_min * this.state.zoom}px;
                    top: ${bbox.y_min * this.state.zoom}px;
                    width: ${(bbox.x_max - bbox.x_min) * this.state.zoom}px;
                    height: ${(bbox.y_max - bbox.y_min) * this.state.zoom}px;
                    border: 2px solid ${isSelected ? '#ff6b6b' : '#4ecdc4'};
                    background: transparent;
                    pointer-events: none;
                `;
                overlay.appendChild(div);
            }
        });
    }

    drawGrid() {
        const gridSize = 50;
        const numLines = 20;
        
        this.ctx.strokeStyle = 'rgba(255, 255, 255, 0.1)';
        this.ctx.lineWidth = 1;
        
        for (let i = -numLines; i <= numLines; i++) {
            const x = this.canvas.width / 2 + i * gridSize * this.state.zoom;
            this.ctx.beginPath();
            this.ctx.moveTo(x, 0);
            this.ctx.lineTo(x, this.canvas.height);
            this.ctx.stroke();
        }
        
        for (let i = -numLines; i <= numLines; i++) {
            const y = this.canvas.height / 2 + i * gridSize * this.state.zoom;
            this.ctx.beginPath();
            this.ctx.moveTo(0, y);
            this.ctx.lineTo(this.canvas.width, y);
            this.ctx.stroke();
        }
    }

    updateFPS() {
        const now = performance.now();
        this.frameCount++;
        
        if (now - this.state.lastRenderTime >= 1000) {
            this.state.fps = this.frameCount;
            this.frameCount = 0;
            this.state.lastRenderTime = now;
            document.getElementById('fpsDisplay').textContent = `FPS: ${this.state.fps}`;
        }
    }


    // ==================== 物体交互 ====================

    findObjectAtPosition(x, y) {
        // 按面积从小到大排序，优先选择小物体
        const sortedObjects = [...this.state.objects].sort((a, b) => {
            const areaA = a.bbox_2d ? (a.bbox_2d.x_max - a.bbox_2d.x_min) * (a.bbox_2d.y_max - a.bbox_2d.y_min) : Infinity;
            const areaB = b.bbox_2d ? (b.bbox_2d.x_max - b.bbox_2d.x_min) * (b.bbox_2d.y_max - b.bbox_2d.y_min) : Infinity;
            return areaA - areaB;
        });
        
        for (const obj of sortedObjects) {
            if (obj.bbox_2d) {
                const bbox = obj.bbox_2d;
                const margin = 5;
                if (x >= bbox.x_min - margin && x <= bbox.x_max + margin && 
                    y >= bbox.y_min - margin && y <= bbox.y_max + margin) {
                    return obj;
                }
            }
        }
        return null;
    }

    async updateObjectPosition(objectId, offset) {
        try {
            const response = await fetch(`${API_BASE}/edit/object/offset`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    scene_id: this.state.sceneId,
                    object_id: objectId,
                    frame_idx: this.state.currentFrame,
                    delta_x: offset.dx,
                    delta_y: offset.dy,
                    delta_z: offset.dz
                })
            });
            
            const data = await response.json();
            
            if (data.success) {
                this.clearFrameCache(this.state.currentFrame);
                await this.loadFrame(this.state.currentFrame);
            }
        } catch (error) {
            console.error('更新物体位置失败:', error);
        }
    }

    async updateObjectRotation(objectId, deltaAngle) {
        // 旋转工具：叠加 yaw 旋转（全局持久，播放时保持）
        try {
            const response = await fetch(`${API_BASE}/edit/object/rotation`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    scene_id: this.state.sceneId,
                    track_id: objectId,
                    delta_yaw: deltaAngle,
                }),
            });
            const data = await response.json();
            if (data && data.success) {
                this._syncRotationUI(data.yaw_pitch_roll || [0, 0, 0]);
                this._invalidateRenderCaches();
                await this.loadFrame(this.state.currentFrame);
            }
        } catch (error) {
            console.error('更新物体旋转失败:', error);
        }
    }

    _syncRotationUI(ypr) {
        this.objectRotation = { yaw: ypr[0], pitch: ypr[1], roll: ypr[2] };
        const ids = ['rotValYaw', 'rotValPitch', 'rotValRoll'];
        const inputs = document.querySelectorAll('.rotation-controls input[type=range]');
        ids.forEach((id, i) => {
            const el = document.getElementById(id);
            if (el) el.textContent = `${Math.round(ypr[i])}°`;
        });
        inputs.forEach((el, i) => { el.value = Math.round(ypr[i]); });
    }

    async _maybeLoadRotation(trackId) {
        if (this._rotationLoadedFor === trackId) return;
        this._rotationLoadedFor = trackId;
        try {
            const resp = await fetch(`${API_BASE}/edit/object/rotation/${this.state.sceneId}/${trackId}`);
            const d = await resp.json();
            if (d && d.success) this._syncRotationUI(d.yaw_pitch_roll || [0, 0, 0]);
        } catch (e) {
            console.error('加载旋转失败:', e);
        }
    }

    onRotationSlider(axis, value) {
        const v = parseFloat(value) || 0;
        this.objectRotation[axis] = v;
        const valId = { yaw: 'rotValYaw', pitch: 'rotValPitch', roll: 'rotValRoll' }[axis];
        const el = document.getElementById(valId);
        if (el) el.textContent = `${Math.round(v)}°`;
        if (this._rotationTimer) clearTimeout(this._rotationTimer);
        this._rotationTimer = setTimeout(() => this._applyRotation(), 60);
    }

    async _applyRotation() {
        const tid = this.state.selectedObjects[0];
        // track_id 0（T0）是合法物体，不能把 0 当成"没选中"。
        if (tid === undefined || tid === null) return;
        try {
            await fetch(`${API_BASE}/edit/object/rotation`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    scene_id: this.state.sceneId,
                    track_id: tid,
                    yaw: this.objectRotation.yaw,
                    pitch: this.objectRotation.pitch,
                    roll: this.objectRotation.roll,
                }),
            });
            this._invalidateRenderCaches();
            await this.loadFrame(this.state.currentFrame);
        } catch (e) {
            console.error('旋转失败:', e);
        }
    }

    // 一键把物体绕自身竖直轴翻转 180°（用于单张图重建无法判断前后的情况）
    async flipObjectRotation() {
        const tid = this.state.selectedObjects[0];
        // track_id 0（T0）是合法物体，不能把 0 当成"没选中"。
        if (tid === undefined || tid === null) return;
        try {
            const resp = await fetch(`${API_BASE}/edit/object/rotation`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ scene_id: this.state.sceneId, track_id: tid, delta_yaw: 180 }),
            });
            const data = await resp.json();
            if (data && data.success) {
                this._syncRotationUI(data.yaw_pitch_roll || [0, 0, 0]);
                this._invalidateRenderCaches();
                await this.loadFrame(this.state.currentFrame);
            }
        } catch (e) {
            console.error('翻转朝向失败:', e);
        }
    }

    async resetObjectRotation() {
        const tid = this.state.selectedObjects[0];
        // track_id 0（T0）是合法物体，不能把 0 当成"没选中"。
        if (tid === undefined || tid === null) return;
        if (this._rotationTimer) clearTimeout(this._rotationTimer);
        try {
            await fetch(`${API_BASE}/edit/object/rotation`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ scene_id: this.state.sceneId, track_id: tid, reset: true }),
            });
            this._syncRotationUI([0, 0, 0]);
            this._rotationLoadedFor = tid;
            this._invalidateRenderCaches();
            await this.loadFrame(this.state.currentFrame);
        } catch (e) {
            console.error('重置朝向失败:', e);
        }
    }

    async deleteSelectedObjects() {
        if (!confirm('确定要删除选中的物体吗?')) return;
        
        for (const objectId of this.state.selectedObjects) {
            try {
                await fetch(`${API_BASE}/edit/object/${this.state.sceneId}/${objectId}`, {
                    method: 'DELETE'
                });
            } catch (error) {
                console.error('删除物体失败:', error);
            }
        }
        
        this.state.selectedObjects = [];
        this.clearFrameCache(this.state.currentFrame);
        await this.loadFrame(this.state.currentFrame);
        this.updateSelectedObjectPanel();
    }

    clearFrameCache(frameIdx) {
        for (const key of this.state.renderCache.keys()) {
            if (key.startsWith(`${frameIdx}_`)) {
                this.state.renderCache.delete(key);
            }
        }
    }

    // 任何会改变渲染结果的编辑（旋转/位移/删除…）后必须调用：
    // 清空 2D 帧缓存 + 3D 视图帧缓存，否则会命中旧图像而"看起来没生效/自动回原位"。
    _invalidateRenderCaches() {
        this.state.renderCache.clear();
        if (this.viewer3d && typeof this.viewer3d.invalidateRenderCache === 'function') {
            this.viewer3d.invalidateRenderCache();
        }
    }


    // ==================== 播放控制 ====================

    prevFrame() {
        if (this.state.currentFrame > 0) {
            this.jumpToFrame(this.state.currentFrame - 1);
        }
    }

    nextFrame() {
        if (this.state.currentFrame < this.state.totalFrames - 1) {
            this.jumpToFrame(this.state.currentFrame + 1);
        }
    }

    // ==================== 时间轴（按需延长帧数） ====================

    _refreshTimelineUI() {
        const set = (id, val) => { const el = document.getElementById(id); if (el) el.textContent = val; };
        set('tlDataFrames', this.state.dataFrames || 0);
        set('tlTotalFrames', this.state.totalFrames || 0);
        const last = Math.max(0, (this.state.totalFrames || 1) - 1);
        const fi = document.getElementById('frameInput');
        if (fi) fi.max = last;
        const fs = document.getElementById('frameSlider');
        if (fs) {
            fs.max = last;
            fs.disabled = last <= 0;
            if (String(fs.value) !== String(Math.min(this.state.currentFrame || 0, last))) {
                fs.value = Math.min(this.state.currentFrame || 0, last);
            }
        }
        const tot = document.getElementById('totalFrames');
        if (tot) tot.textContent = `/ ${this.state.totalFrames}`;
    }

    _tlStatus(msg, isErr) {
        const el = document.getElementById('tlStatus');
        if (el) {
            el.textContent = msg;
            el.style.color = isErr ? 'var(--danger-color)' : '';
        }
    }

    async extendTimeline(extra) {
        if (!this.state.sceneId) { alert('请先加载场景'); return; }
        const n = Math.max(1, Math.min(2000, parseInt(extra) || 50));
        const holdCam = !!(document.getElementById('tlHoldCamera') || {}).checked;
        this._tlStatus(`正在延长 ${n} 帧（动态物体沿各自轨迹外推，`
            + (holdCam ? '相机固定在末帧' : '相机沿自车轨迹外推') + '）…');
        try {
            const resp = await fetch(`${API_BASE}/timeline/extend`, {
                method: 'POST', headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ scene_id: this.state.sceneId, extra_frames: n,
                                       mode: 'extrapolate',
                                       ego_mode: holdCam ? 'hold' : 'extrapolate' }),
            });
            const d = await resp.json();
            if (!resp.ok || !d.success) throw new Error(d.detail || ('HTTP ' + resp.status));
            this.state.dataFrames = d.data_frames;
            this.state.totalFrames = d.total_frames;
            this.state.renderCache.clear();
            this._refreshTimelineUI();
            const eg = (d.extended_tracks || 0);
            this._tlStatus(`已延长到 ${d.total_frames} 帧（数据帧 ${d.data_frames}，`
                + `外推 ${d.extended_frames} 帧，涉及 ${eg} 条轨迹）。可以直接跳到后面的帧继续编辑。`);
            this.updateStatus(`时间轴: ${d.data_frames} → ${d.total_frames} 帧`);
        } catch (e) {
            this._tlStatus('延长失败: ' + e.message, true);
        }
    }

    jumpToFrame(frameIdx) {
        const last = Math.max(0, (this.state.totalFrames || 1) - 1);
        const f = Math.max(0, Math.min(last, parseInt(frameIdx, 10) || 0));
        this.state.currentFrame = f;
        const fi = document.getElementById('frameInput');
        if (fi) fi.value = f;
        const fs = document.getElementById('frameSlider');
        if (fs && String(fs.value) !== String(f)) fs.value = f;
        if (this._scrubTimer) { clearTimeout(this._scrubTimer); this._scrubTimer = null; }
        this.loadFrame(f);
    }

    // 帧号变化时把顶部"数字输入 + 拖动条"同步一下（播放/跳帧都走这里）
    _syncFrameWidgets() {
        const f = this.state.currentFrame || 0;
        const fi = document.getElementById('frameInput');
        if (fi && String(fi.value) !== String(f)) fi.value = f;
        const fs = document.getElementById('frameSlider');
        if (fs && String(fs.value) !== String(f)) fs.value = f;
    }

    // 拖动顶部帧数条：帧号立刻更新，渲染节流（拖动过程中最多每 ~150ms 渲染一次）
    _scrubToFrame(frameIdx, immediate = false) {
        const last = Math.max(0, (this.state.totalFrames || 1) - 1);
        const f = Math.max(0, Math.min(last, parseInt(frameIdx, 10) || 0));
        this.state.currentFrame = f;
        const fi = document.getElementById('frameInput');
        if (fi) fi.value = f;
        const fs = document.getElementById('frameSlider');
        if (fs && String(fs.value) !== String(f)) fs.value = f;
        const now = Date.now();
        const run = () => {
            this._scrubTimer = null;
            this._scrubLastAt = Date.now();
            this.loadFrame(this.state.currentFrame);
        };
        if (immediate || now - (this._scrubLastAt || 0) > 150) {
            if (this._scrubTimer) { clearTimeout(this._scrubTimer); this._scrubTimer = null; }
            run();
        } else if (!this._scrubTimer) {
            this._scrubTimer = setTimeout(run, 150);
        }
    }

    togglePlay() {
        this.state.isPlaying = !this.state.isPlaying;
        
        const playIcon = document.getElementById('playIcon');
        const pauseIcon = document.getElementById('pauseIcon');
        
        if (this.state.isPlaying) {
            playIcon.style.display = 'none';
            pauseIcon.style.display = 'block';

            // 若已经在最后一帧，则从头开始；否则从当前帧继续
            if (this.state.currentFrame >= this.state.totalFrames - 1) {
                this.jumpToFrame(0);
            }

            if (this.state.viewMode === '3d' && this.viewer3d) {
                this._play3d();
            } else {
                this.state.playInterval = setInterval(() => {
                    if (this.state.currentFrame < this.state.totalFrames - 1) {
                        this.nextFrame();
                    } else {
                        this.pausePlayback();
                    }
                }, 100);
            }
        } else {
            this.pausePlayback();
        }
    }

    // 暂停播放（不改变 isPlaying 之外的状态，停在当前帧）
    pausePlayback() {
        this.state.isPlaying = false;
        const playIcon = document.getElementById('playIcon');
        const pauseIcon = document.getElementById('pauseIcon');
        if (playIcon) playIcon.style.display = 'block';
        if (pauseIcon) pauseIcon.style.display = 'none';
        if (this.state.playInterval) {
            clearInterval(this.state.playInterval);
            clearTimeout(this.state.playInterval);
            this.state.playInterval = null;
        }
    }

    // 3D 播放：逐帧用同一相机渲染（命中缓存则瞬时），并预取后续帧，避免黑屏
    async _play3d() {
        const viewer = this.viewer3d;
        const fps = 10;
        const interval = 1000 / fps;

        // 播放前先预加载前几帧，避免初始黑屏
        if (viewer.preloadForPlayback) {
            this.updateStatus('正在预加载帧...');
            await viewer.preloadForPlayback(this.state.currentFrame, 5);
            this.updateStatus('播放中...');
        }

        const step = async () => {
            if (!this.state.isPlaying || this.state.viewMode !== '3d') return;
            const t0 = performance.now();

            if (this.state.currentFrame < this.state.totalFrames - 1) {
                this.state.currentFrame += 1;
                this._syncFrameWidgets();
                
                // 使用优化的播放加载方法：保留上一帧避免黑屏，异步加载新帧，自动预取
                if (viewer.loadFrameForPlayback) {
                    await viewer.loadFrameForPlayback(this.state.sceneId, this.state.currentFrame);
                } else {
                    // 回退方案
                    viewer.frameIdx = this.state.currentFrame;
                    await viewer.loadFrame(this.state.sceneId, this.state.currentFrame, true);
                }
            } else {
                this.pausePlayback();  // 播到末尾自动停在最后一帧，不自动回到开头
                return;
            }

            const elapsed = performance.now() - t0;
            const wait = Math.max(0, interval - elapsed);
            this.state.playInterval = setTimeout(step, wait);
        };
        step();
    }

    // ==================== 属性编辑 ====================

    updateSelectedObjectPanel() {
        const panel = document.getElementById('selectedObjectPanel');
        if (!panel) return;
        if (this.state.selectedObjects.length !== 1) this._rotationLoadedFor = null;

        if (this.state.selectedObjects.length === 0) {
            panel.innerHTML = '<div class="empty-state"><p>点击画布中的物体进行选择</p></div>';
            return;
        }

        if (this.state.selectedObjects.length === 1) {
            const tid = this.state.selectedObjects[0];
            const obj = this.state.objects.find(o => (o.track_id ?? o.object_id) === tid);
            if (obj) {
                const pose = obj.pose_world;
                const pos = pose ? [pose[0][3], pose[1][3], pose[2][3]] : [0, 0, 0];
                const dims = obj.dimensions || [];
                const dimStr = dims.length === 3
                    ? `${dims[0].toFixed(1)} × ${dims[1].toFixed(1)} × ${dims[2].toFixed(1)}`
                    : '-';
                const r = this.objectRotation || { yaw: 0, pitch: 0, roll: 0 };
                const srcId = (this.state.ego || {}).source_track_id;
                const isSrc = srcId !== null && srcId !== undefined && Number(srcId) === Number(tid);
                const viewSourceBtn = isSrc
                    ? `<button class="btn btn-small btn-secondary" onclick="studio.egoSetSource('')">恢复真实主车视角</button>`
                    : `<button class="btn btn-small btn-primary" onclick="studio.egoSetSource(${tid})">以此物体视角渲染</button>`;
                panel.innerHTML = `
                    <div class="selected-object-info">
                        <div class="info-row"><span class="label">Track:</span><span class="value">T:${tid}</span></div>
                        <div class="info-row"><span class="label">类型:</span><span class="value">${obj.type || '未知'}</span></div>
                        <div class="info-row"><span class="label">位置:</span><span class="value">(${pos[0].toFixed(1)}, ${pos[1].toFixed(1)}, ${pos[2].toFixed(1)})</span></div>
                        <div class="info-row"><span class="label">尺寸:</span><span class="value">${dimStr}</span></div>
                        <div class="info-row"><span class="label">已编辑:</span><span class="value">${obj.edited ? '是' : '否'}</span></div>
                        <div class="rotation-controls">
                            <div class="rot-row"><label>朝向</label><input type="range" min="-180" max="180" step="1" value="${Math.round(r.yaw)}" oninput="studio.onRotationSlider('yaw', this.value)"><span class="rot-val" id="rotValYaw">${Math.round(r.yaw)}°</span></div>
                            <div class="rot-row"><label>俯仰</label><input type="range" min="-180" max="180" step="1" value="${Math.round(r.pitch)}" oninput="studio.onRotationSlider('pitch', this.value)"><span class="rot-val" id="rotValPitch">${Math.round(r.pitch)}°</span></div>
                            <div class="rot-row"><label>翻滚</label><input type="range" min="-180" max="180" step="1" value="${Math.round(r.roll)}" oninput="studio.onRotationSlider('roll', this.value)"><span class="rot-val" id="rotValRoll">${Math.round(r.roll)}°</span></div>
                            <div class="button-row">
                                <button class="btn btn-small btn-secondary" onclick="studio.flipObjectRotation()">翻转180°</button>
                                <button class="btn btn-small btn-secondary" onclick="studio.resetObjectRotation()">重置朝向</button>
                            </div>
                        </div>
                        <div class="button-row">
                            <button class="btn btn-small btn-secondary" onclick="studio.viewer3d && studio.viewer3d.focusSelected()">聚焦</button>
                            <button class="btn btn-small btn-danger" onclick="studio.delete3dSelected()">删除物体</button>
                        </div>
                        <div class="button-row">
                            ${viewSourceBtn}
                        </div>
                        ${(Number(tid) >= 100000 || Number(tid) === 900000) ? `
                        <div class="rotation-controls">
                            <div class="rot-row"><label>模型大小</label>
                                <input type="range" min="0.5" max="1.5" step="0.02"
                                    value="${Number((this.state.modelFit && this.state.modelFit[tid]) || 1).toFixed(2)}"
                                    oninput="studio.onModelSizeSlider(${tid}, this.value)">
                                <span class="rot-val" id="modelSizeVal_${tid}">×${Number((this.state.modelFit && this.state.modelFit[tid]) || 1).toFixed(2)}</span>
                            </div>
                            <div class="detail" style="margin-top:4px;">等比缩放模型与包围盒（长宽高比不变）。</div>
                        </div>` : ''}
                    </div>
                `;
                this._maybeLoadRotation(tid);
            }
        } else {
            panel.innerHTML = `
                <div class="multiple-objects">
                    <p>已选择 ${this.state.selectedObjects.length} 个物体</p>
                </div>
            `;
        }
    }

    // 选中物体的"模型大小"滑块：等比缩放模型 + 包围盒（长宽高比不变）
    async onModelSizeSlider(tid, value) {
        const factor = Math.max(0.2, Math.min(5.0, parseFloat(value) || 1));
        this.state.modelFit = this.state.modelFit || {};
        this.state.modelFit[tid] = factor;
        const el = document.getElementById('modelSizeVal_' + tid);
        if (el) el.textContent = '×' + factor.toFixed(2);
        if (this._modelSizeTimer) clearTimeout(this._modelSizeTimer);
        this._modelSizeTimer = setTimeout(async () => {
            try {
                const resp = await fetch(`${API_BASE}/edit/synthetic/scale`, {
                    method: 'POST', headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ scene_id: this.state.sceneId, track_id: tid, factor }),
                });
                const d = await resp.json();
                if (!resp.ok || !d.success) throw new Error(d.detail || ('HTTP ' + resp.status));
                const dims = (d.dimensions || []).map(v => Number(v).toFixed(2)).join(' × ');
                this.updateStatus(`T:${tid} 模型大小 ×${factor.toFixed(2)}（包围盒 ${dims} m）`);
                this._invalidateRenderCaches();
                await this.loadFrame(this.state.currentFrame);
                if (this.viewer3d && typeof this.viewer3d.loadFrame === 'function') {
                    await this.viewer3d.loadFrame(this.state.sceneId, this.state.currentFrame, true);
                }
                this.updateSelectedObjectPanel();
            } catch (e) {
                this.updateStatus('调整模型大小失败: ' + e.message);
            }
        }, 120);
    }

    delete3dSelected() {
        if (this.viewer3d && this.viewer3d.selectedTrackId !== null) {
            if (confirm('确定删除选中物体（整段轨迹）?')) this.viewer3d.deleteSelected();
        }
    }

    // ==================== 轨迹编辑 ====================

    async showTrajectoryEditor(objectId) {
        const modal = document.getElementById('keyframeEditorModal');
        modal.classList.add('active');
        
        try {
            const response = await fetch(`${API_BASE}/objects/${this.state.sceneId}/${objectId}/trajectory`);
            const data = await response.json();
            
            this.state.editingTrajectory = {
                objectId: objectId,
                keyframes: data.keyframes || data.trajectory || []
            };
            
            this.renderKeyframeTimeline();
        } catch (error) {
            console.error('加载轨迹失败:', error);
        }
    }

    renderKeyframeTimeline() {
        const timeline = document.getElementById('keyframeTimeline');
        
        if (!this.state.editingTrajectory || this.state.editingTrajectory.keyframes.length === 0) {
            timeline.innerHTML = '<div class="empty-state"><p>暂无关键帧</p></div>';
            return;
        }
        
        timeline.innerHTML = this.state.editingTrajectory.keyframes.map((kf, index) => {
            const pose = kf.pose_world;
            const pos = [pose[0][3], pose[1][3], pose[2][3]];
            
            return `
                <div class="keyframe-item" data-index="${index}">
                    <div class="keyframe-frame">帧 ${kf.frame_idx}</div>
                    <div class="keyframe-pos">
                        (${pos[0].toFixed(1)}, ${pos[1].toFixed(1)}, ${pos[2].toFixed(1)})
                    </div>
                </div>
            `;
        }).join('');
        
        timeline.querySelectorAll('.keyframe-item').forEach(item => {
            item.addEventListener('click', () => {
                const index = parseInt(item.dataset.index);
                this.selectKeyframe(index);
            });
        });
    }

    selectKeyframe(index) {
        document.querySelectorAll('.keyframe-item').forEach(item => item.classList.remove('selected'));
        const item = document.querySelector(`.keyframe-item[data-index="${index}"]`);
        if (item) {
            item.classList.add('selected');
        }
        
        const kf = this.state.editingTrajectory.keyframes[index];
        this.jumpToFrame(kf.frame_idx);
    }

    addKeyframe() {
        if (!this.state.editingTrajectory) return;
        
        const currentObj = this.state.objects.find(o => o.object_id === this.state.editingTrajectory.objectId);
        if (!currentObj) return;
        
        this.state.editingTrajectory.keyframes.push({
            frame_idx: this.state.currentFrame,
            pose_world: currentObj.pose_world
        });
        
        this.renderKeyframeTimeline();
    }

    async saveKeyframes() {
        if (!this.state.editingTrajectory) return;
        
        try {
            await fetch(`${API_BASE}/edit/trajectory`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    scene_id: this.state.sceneId,
                    object_id: this.state.editingTrajectory.objectId,
                    keyframes: this.state.editingTrajectory.keyframes
                })
            });
            
            document.getElementById('keyframeEditorModal').classList.remove('active');
            this.clearFrameCache(this.state.currentFrame);
            await this.loadFrame(this.state.currentFrame);
            this.saveEditHistory('trajectory', this.state.editingTrajectory.objectId);
        } catch (error) {
            console.error('保存轨迹失败:', error);
        }
    }


    // ==================== Corner Case生成 ====================

    // ==================== Corner Case 生成（基于轨迹编辑） ====================

    async initCornerCasePanel() {
        // 拉取场景类型定义
        try {
            const resp = await fetch(`${API_BASE}/corner_case/types`);
            const data = await resp.json();
            this.cornerScenarios = {};
            (data.types || []).forEach(s => { this.cornerScenarios[s.type] = s; });
        } catch (e) {
            this.cornerScenarios = {};
        }
        this.cornerRoleAssign = {};      // {role_key: track_id}
        this.cornerActiveRole = null;    // 当前等待指派的 role_key
        this.cornerAffectedTracks = [];  // 上次生成影响的 track（用于清除）
        this.cornerSynthesizedTracks = [];
        this._updateQualityReportButton();

        const sel = document.getElementById('cornerScenarioType');
        if (sel) {
            sel.addEventListener('change', () => this.renderCornerRoles());
            this.renderCornerRoles();
        }
        const intensity = document.getElementById('cornerIntensity');
        if (intensity) {
            intensity.addEventListener('input', () => {
                document.getElementById('cornerIntensityVal').textContent = parseFloat(intensity.value).toFixed(1);
            });
        }
        const genBtn = document.getElementById('generateCornerCaseBtn');
        if (genBtn) genBtn.addEventListener('click', () => this.generateCornerCase());
        const clearBtn = document.getElementById('clearCornerCaseBtn');
        if (clearBtn) clearBtn.addEventListener('click', () => this.clearCornerCase());
        const jumpBtn = document.getElementById('jumpCriticalFrameBtn');
        if (jumpBtn) jumpBtn.addEventListener('click', () => this.jumpToCriticalFrame());
        const qualityBtn = document.getElementById('viewQualityReportBtn');
        if (qualityBtn) qualityBtn.addEventListener('click', () => this.showQualityReportModal());
        
        // 场景参数折叠面板
        const paramsHeader = document.getElementById('scenarioParamsHeader');
        const paramsContent = document.getElementById('cornerSamplingParams');
        if (paramsHeader && paramsContent) {
            paramsHeader.addEventListener('click', () => {
                const isCollapsed = paramsHeader.classList.toggle('collapsed');
                paramsContent.style.display = isCollapsed ? 'none' : 'flex';
            });
        }
    }

    renderCornerRoles() {
        const sel = document.getElementById('cornerScenarioType');
        const rolesDiv = document.getElementById('cornerRoles');
        const samplingDiv = document.getElementById('cornerSamplingParams');
        const descEl = document.getElementById('cornerScenarioDesc');
        if (!sel || !rolesDiv) return;
        const scenario = this.cornerScenarios[sel.value];
        if (!scenario) {
            rolesDiv.innerHTML = '';
            if (samplingDiv) samplingDiv.innerHTML = '';
            return;
        }
        if (descEl) descEl.textContent = scenario.desc || '';

        // 切换场景时重置角色指派
        this.cornerRoleAssign = {};
        this.cornerActiveRole = null;
        if (this.viewer3d) this.viewer3d.clearCriticalFrames();

        rolesDiv.innerHTML = scenario.roles.map(r => {
            const auto = r.optional || r.auto || (r.label && r.label.includes('可选'));
            const hint = auto ? '<span class="cc-role-auto" title="不指定则自动生成">自动生成</span>' : '';
            return `
            <div class="cc-role" data-role="${r.key}">
                <span class="cc-role-label">${r.label}${hint}</span>
                <button class="btn btn-small cc-role-btn" data-role="${r.key}">指定</button>
                <span class="cc-role-value" data-role="${r.key}">${auto ? '自动' : '未指定'}</span>
            </div>`;
        }).join('');

        rolesDiv.querySelectorAll('.cc-role-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                const role = btn.dataset.role;
                // 若当前已选中物体，直接指派；否则进入"等待点选"状态
                if (this.viewer3d && this.viewer3d.selectedTrackId !== null) {
                    this.assignCornerRole(role, this.viewer3d.selectedTrackId);
                } else if (this.state.selectedObjects.length > 0) {
                    this.assignCornerRole(role, this.state.selectedObjects[0]);
                } else {
                    this.cornerActiveRole = role;
                    rolesDiv.querySelectorAll('.cc-role-btn').forEach(b => b.classList.remove('active'));
                    btn.classList.add('active');
                    this.updateStatus(`请在视图中点选一个物体作为「${this._cornerRoleLabel(role)}」`);
                }
            });
        });

        if (samplingDiv) {
            samplingDiv.innerHTML = this._renderSamplingControls(scenario.sampling_schema || {});
            samplingDiv.querySelectorAll('.cc-sampling-range input').forEach(input => {
                input.addEventListener('input', () => this._normalizeSamplingRange(input));
            });
        }
    }

    _renderSamplingControls(schema) {
        const entries = Object.entries(schema || {});
        if (entries.length === 0) {
            return '<div class="cc-sampling-empty">该场景暂无可采样参数</div>';
        }
        return entries.map(([key, spec]) => {
            const label = this._escapeHtml(spec.label || key);
            const unit = spec.unit ? `<span class="cc-param-unit">${this._escapeHtml(spec.unit)}</span>` : '';
            if (spec.type === 'enum') {
                const options = (spec.options || []).map(opt => {
                    const selected = opt === spec.default ? 'selected' : '';
                    return `<option value="${this._escapeHtml(opt)}" ${selected}>${this._samplingOptionLabel(opt)}</option>`;
                }).join('');
                return `<div class="cc-param" data-param="${this._escapeHtml(key)}" data-type="enum">
                    <label>${label}${unit}</label>
                    <select class="cc-select cc-param-value" data-param="${this._escapeHtml(key)}">${options}</select>
                </div>`;
            }
            const def = Array.isArray(spec.default) ? spec.default : [spec.min, spec.max];
            const lo = Number(def[0] ?? spec.min ?? 0);
            const hi = Number(def[1] ?? spec.max ?? lo);
            return `<div class="cc-param cc-sampling-range" data-param="${this._escapeHtml(key)}" data-type="range">
                <label>${label}${unit}</label>
                <div class="cc-param-range-row">
                    <input type="number" class="cc-param-min" data-param="${this._escapeHtml(key)}" value="${lo}" min="${spec.min}" max="${spec.max}" step="${spec.step || 0.1}">
                    <span>至</span>
                    <input type="number" class="cc-param-max" data-param="${this._escapeHtml(key)}" value="${hi}" min="${spec.min}" max="${spec.max}" step="${spec.step || 0.1}">
                </div>
            </div>`;
        }).join('');
    }

    _samplingOptionLabel(value) {
        const labels = {
            near_miss: 'near-miss',
            minor: '轻微',
            severe: '严重',
            clear: '晴朗',
            rain: '雨天',
            fog: '雾天',
            night: '夜间',
            straight: '直路',
            curve: '弯道',
            intersection: '路口',
            ramp: '匝道',
            none: '无遮挡',
            vehicle: '车辆遮挡',
            roadside: '路侧遮挡'
        };
        return this._escapeHtml(labels[value] || value);
    }

    _normalizeSamplingRange(input) {
        const row = input.closest('.cc-sampling-range');
        if (!row) return;
        const minInput = row.querySelector('.cc-param-min');
        const maxInput = row.querySelector('.cc-param-max');
        if (!minInput || !maxInput) return;
        const minVal = Number(minInput.value);
        const maxVal = Number(maxInput.value);
        if (Number.isFinite(minVal) && Number.isFinite(maxVal) && minVal > maxVal) {
            if (input === minInput) maxInput.value = minVal;
            else minInput.value = maxVal;
        }
    }

    _cornerRoleLabel(roleKey) {
        const sel = document.getElementById('cornerScenarioType');
        const scenario = this.cornerScenarios[sel.value];
        const r = scenario && scenario.roles.find(x => x.key === roleKey);
        return r ? r.label : roleKey;
    }

    _collectSamplingParams() {
        const params = {};
        const container = document.getElementById('cornerSamplingParams');
        if (!container) return params;
        container.querySelectorAll('.cc-param').forEach(row => {
            const key = row.dataset.param;
            const type = row.dataset.type;
            if (!key) return;
            if (type === 'enum') {
                const select = row.querySelector('.cc-param-value');
                if (select) params[key] = select.value;
            } else if (type === 'range') {
                const minInput = row.querySelector('.cc-param-min');
                const maxInput = row.querySelector('.cc-param-max');
                if (!minInput || !maxInput) return;
                const minVal = Number(minInput.value);
                const maxVal = Number(maxInput.value);
                if (Number.isFinite(minVal) && Number.isFinite(maxVal)) {
                    params[key] = [Math.min(minVal, maxVal), Math.max(minVal, maxVal)];
                }
            }
        });
        return params;
    }

    assignCornerRole(roleKey, trackId) {
        this.cornerRoleAssign[roleKey] = trackId;
        this.cornerActiveRole = null;
        const valEl = document.querySelector(`.cc-role-value[data-role="${roleKey}"]`);
        if (valEl) valEl.textContent = `T:${trackId}`;
        document.querySelectorAll('.cc-role-btn').forEach(b => b.classList.remove('active'));
        this.updateStatus(`已指定「${this._cornerRoleLabel(roleKey)}」= T:${trackId}`);
    }

    // 由选中事件调用：若正在等待指派角色，则把点选的 track 赋给该角色
    onTrackSelectedForCorner(trackId) {
        if (this.cornerActiveRole) {
            this.assignCornerRole(this.cornerActiveRole, trackId);
            return true;
        }
        return false;
    }

    async generateCornerCase() {
        const sel = document.getElementById('cornerScenarioType');
        const scenarioType = sel.value;
        const scenario = this.cornerScenarios[scenarioType];
        if (!scenario) { alert('请选择场景类型'); return; }

        // 校验必填角色（label 含"可选"的可不填）
        // 校验角色：optional/auto 的角色可不指定（后端会自动合成参与者）
        const roles = {};
        let assignedCount = 0;
        for (const r of scenario.roles) {
            const tid = this.cornerRoleAssign[r.key];
            if (tid === undefined || tid === null) {
                const skippable = r.optional || r.auto || (r.label && r.label.includes('可选'));
                if (!skippable) {
                    alert(`请指定角色：${r.label}`);
                    return;
                }
            } else {
                roles[r.key] = tid;
                assignedCount += 1;
            }
        }
        const allowAllAuto = scenario.roles.every(r => r.optional || r.auto || (r.label && r.label.includes('可选')));
        if (assignedCount === 0 && !allowAllAuto) {
            alert('请至少指定一个参与者，未指定的角色将自动生成');
            return;
        }

        const startFrame = parseInt(document.getElementById('cornerStartFrame').value) || 0;
        const numFrames = parseInt(document.getElementById('cornerNumFrames').value) || 20;
        const intensity = parseFloat(document.getElementById('cornerIntensity').value) || 1.0;
        const physicsEl = document.getElementById('cornerEnablePhysics');
        const enablePhysics = physicsEl ? physicsEl.checked : true;
        const samplingParams = this._collectSamplingParams();

        try {
            const response = await fetch(`${API_BASE}/corner_case/generate`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    scene_id: this.state.sceneId,
                    scenario_type: scenarioType,
                    roles: roles,
                    start_frame: startFrame,
                    num_frames: numFrames,
                    intensity: intensity,
                    enable_physics: enablePhysics,
                    sampling_params: samplingParams
                })
            });
            const data = await response.json();
            if (data.success) {
                this.cornerAffectedTracks = data.affected_tracks || [];
                this.cornerSynthesizedTracks = data.synthesized_tracks || [];
                this.state.lastQualityReport = data.quality_report || null;
                this.state.lastSamplingParams = data.sampling_params || null;
                this._updateQualityReportButton();
                this.state.renderCache.clear();
                if (this.viewer3d) {
                    this.viewer3d._invalidateCache();
                    this.viewer3d.trajectory = null;
                    if (this.viewer3d.selectedTrackId !== null) {
                        this.viewer3d._fetchTrajectory(this.viewer3d.selectedTrackId);
                    }
                    // 在轨迹上高亮碰撞关键帧（仅限碰撞涉及的 track）
                    if (data.collision_analysis) {
                        this.viewer3d.setCriticalFrames(data.collision_analysis, data.collision_tracks);
                    } else {
                        this.viewer3d.clearCriticalFrames();
                    }
                }
                await this.loadFrame(this.state.currentFrame);
                this.saveEditHistory('corner_case', scenarioType);
                const statusEl = document.getElementById('cornerCaseStatus');
                if (statusEl) {
                    let msg = `已生成「${scenario.name}」，影响 ${this.cornerAffectedTracks.length} 个物体的轨迹`;
                    const ca = data.collision_analysis;
                    if (ca) {
                        msg += this._formatCollisionInfo(ca);
                    }
                    if (data.sampling_params) {
                        msg += this._formatSamplingSummary(data.sampling_params);
                    }
                    if (data.quality_report) {
                        msg += this._formatQualityReport(data.quality_report);
                        msg += '<div class="cc-quality-note">可点击“查看质量报告”查看完整检查结果和原始 JSON。</div>';
                    }
                    statusEl.innerHTML = msg;
                }
                this.updateStatus(`已生成 Corner Case: ${scenario.name}`);
            } else {
                alert('生成失败: ' + (data.detail || '未知错误'));
            }
        } catch (error) {
            console.error('生成Corner Case失败:', error);
            alert('生成失败: ' + error.message);
        }
    }

    _formatCollisionInfo(ca) {
        if (!ca) return '';
        const sevMap = { high: '高危', medium: '中等', low: '较低', near_miss: '险情' };
        let html = '<div class="cc-collision-info">';
        if (ca.collision_frame !== null && ca.collision_frame !== undefined) {
            html += `<div>碰撞帧: <b>${ca.collision_frame}</b></div>`;
            html += `<div>最晚反应帧: <b style="color:#ff9500">${ca.critical_frame}</b></div>`;
            if (ca.time_to_collision !== null && ca.time_to_collision !== undefined) {
                html += `<div>反应窗口: ${ca.time_to_collision.toFixed(2)} 秒 (${ca.reaction_frames} 帧)</div>`;
            }
            if (ca.distance_at_critical !== undefined) {
                html += `<div>关键帧距离: ${ca.distance_at_critical.toFixed(2)} m</div>`;
            }
            html += `<div>严重程度: ${sevMap[ca.collision_severity] || ca.collision_severity}</div>`;
        } else if (ca.collision_severity === 'near_miss') {
            html += `<div>险情(未实际碰撞)，最接近帧: <b>${ca.critical_frame}</b></div>`;
            html += `<div>最近距离: ${ca.distance_at_critical.toFixed(2)} m</div>`;
        }
        html += '</div>';
        return html;
    }

    _formatSamplingSummary(params) {
        const entries = Object.entries(params || {});
        if (entries.length === 0) return '';
        let html = '<div class="cc-sampling-summary"><b>本次采样参数</b>';
        entries.slice(0, 8).forEach(([key, value]) => {
            const v = typeof value === 'number' ? value.toFixed(2) : value;
            html += `<div><span>${this._escapeHtml(key)}</span><em>${this._escapeHtml(v)}</em></div>`;
        });
        html += '</div>';
        return html;
    }

    _formatQualityReport(report) {
        if (!report) return '';
        const valid = report.valid === true;
        const statusText = valid ? '通过质量准入' : '未通过质量准入';
        const statusClass = valid ? 'pass' : 'fail';
        const fmt = (v, digits = 2, suffix = '') => {
            if (v === null || v === undefined || Number.isNaN(Number(v))) return '未知';
            return `${Number(v).toFixed(digits)}${suffix}`;
        };
        const failedChecks = (report.checks || []).filter(c => c.passed === false);
        const unknownChecks = (report.checks || []).filter(c => c.passed === null);
        let html = `<div class="cc-quality-info ${statusClass}">`;
        html += `<div class="cc-quality-header"><span>质量准入</span><b>${statusText}</b></div>`;
        html += '<div class="cc-quality-grid">';
        html += `<div><span>最近距离</span><b>${fmt(report.min_distance, 2, ' m')}</b></div>`;
        html += `<div><span>TTC</span><b>${fmt(report.ttc_at_critical, 2, ' s')}</b></div>`;
        html += `<div><span>最大加速度</span><b>${fmt(report.max_accel, 2, ' m/s²')}</b></div>`;
        html += `<div><span>最大穿透</span><b>${fmt(report.bbox_penetration, 2, ' m')}</b></div>`;
        html += '</div>';
        if (failedChecks.length > 0) {
            html += '<div class="cc-quality-list"><span>未通过项</span>';
            failedChecks.slice(0, 4).forEach(check => {
                html += `<div class="cc-quality-check fail">${this._qualityCheckLabel(check.name)}</div>`;
            });
            html += '</div>';
        }
        if (unknownChecks.length > 0) {
            const names = unknownChecks.map(c => this._qualityCheckLabel(c.name)).join('、');
            html += `<div class="cc-quality-note">待接入检查: ${names}</div>`;
        }
        html += '</div>';
        return html;
    }

    _qualityCheckLabel(name) {
        const labels = {
            event_present: '碰撞或 near-miss 事件',
            ttc_range: 'TTC 合理范围',
            motion_physical: '轨迹动力学合理性',
            bbox_penetration: '包围盒穿透深度',
            annotation_consistency: '标注一致性',
            offroad: '道路区域约束',
            visibility: '可见性'
        };
        return this._escapeHtml(labels[name] || name || '未知检查');
    }

    _updateQualityReportButton() {
        const btn = document.getElementById('viewQualityReportBtn');
        if (btn) btn.disabled = !this.state.lastQualityReport;
    }

    showQualityReportModal() {
        const modal = document.getElementById('qualityReportModal');
        const content = document.getElementById('qualityReportContent');
        const report = this.state.lastQualityReport;
        if (!modal || !content) return;
        if (!report) {
            content.innerHTML = '<div class="empty-state"><p>生成 Corner Case 后可查看质量报告</p></div>';
        } else {
            content.innerHTML = this._formatQualityReportDetail(report);
        }
        modal.classList.add('active');
    }

    _formatQualityReportDetail(report) {
        const valid = report.valid === true;
        const fmt = (v, digits = 2, suffix = '') => {
            if (v === null || v === undefined || Number.isNaN(Number(v))) return '未知';
            return `${Number(v).toFixed(digits)}${suffix}`;
        };
        const status = valid ? '通过质量准入' : '未通过质量准入';
        const checks = report.checks || [];
        let html = `<div class="quality-report-banner ${valid ? 'pass' : 'fail'}">`;
        html += `<span>${status}</span><b>${this._escapeHtml(report.case_type || '未知类型')}</b>`;
        html += '</div>';
        html += '<div class="quality-report-grid">';
        html += this._qualityMetricCell('碰撞帧', report.collision_frame ?? '无');
        html += this._qualityMetricCell('最晚反应帧', report.critical_frame ?? '无');
        html += this._qualityMetricCell('TTC', fmt(report.ttc_at_critical, 2, ' s'));
        html += this._qualityMetricCell('最近距离', fmt(report.min_distance, 2, ' m'));
        html += this._qualityMetricCell('最大加速度', fmt(report.max_accel, 2, ' m/s²'));
        html += this._qualityMetricCell('最大偏航率', fmt(report.max_yaw_rate, 2, ' rad/s'));
        html += this._qualityMetricCell('最大穿透', fmt(report.bbox_penetration, 2, ' m'));
        html += this._qualityMetricCell('标注一致性', report.annotation_consistency ? '通过' : '失败');
        html += '</div>';
        html += '<div class="quality-report-section"><h3>分项检查</h3>';
        html += '<div class="quality-check-table">';
        checks.forEach(check => {
            const state = check.passed === true ? 'pass' : (check.passed === false ? 'fail' : 'unknown');
            const stateText = check.passed === true ? '通过' : (check.passed === false ? '失败' : '待接入');
            html += `<div class="quality-check-row ${state}">`;
            html += `<span>${this._qualityCheckLabel(check.name)}</span>`;
            html += `<b>${stateText}</b>`;
            html += `<small>${this._escapeHtml(check.description || '')}</small>`;
            html += '</div>';
        });
        html += '</div></div>';
        html += this._formatSamplingDetailFromReport(report);
        html += '<div class="quality-report-section"><h3>原始 JSON</h3>';
        html += `<pre class="quality-json">${this._escapeHtml(JSON.stringify(report, null, 2))}</pre>`;
        html += '</div>';
        return html;
    }

    _formatSamplingDetailFromReport(report) {
        const params = this.state.lastSamplingParams || {};
        const entries = Object.entries(params);
        if (entries.length === 0) return '';
        let html = '<div class="quality-report-section"><h3>场景族采样参数</h3>';
        html += '<div class="quality-report-grid">';
        entries.forEach(([key, value]) => {
            const v = typeof value === 'number' ? value.toFixed(3) : value;
            html += this._qualityMetricCell(key, v);
        });
        html += '</div></div>';
        return html;
    }

    _qualityMetricCell(label, value) {
        return `<div><span>${this._escapeHtml(label)}</span><b>${this._escapeHtml(value)}</b></div>`;
    }

    _escapeHtml(value) {
        return String(value).replace(/[&<>'"]/g, ch => ({
            '&': '&amp;',
            '<': '&lt;',
            '>': '&gt;',
            "'": '&#39;',
            '"': '&quot;'
        }[ch]));
    }

    // 跳转到关键帧
    jumpToCriticalFrame() {
        const ca = this.viewer3d && this.viewer3d.criticalFrames;
        if (ca && ca.critical_frame !== null && ca.critical_frame !== undefined) {
            this.loadFrame(ca.critical_frame);
            this.state.currentFrame = ca.critical_frame;
            this._syncFrameWidgets();
            this.updateStatus(`已跳转到最晚反应关键帧: ${ca.critical_frame}`);
        }
    }

    async clearCornerCase() {
        const tracks = [...(this.cornerAffectedTracks || [])];
        // 合成参与者也要一并清除
        (this.cornerSynthesizedTracks || []).forEach(t => {
            if (!tracks.includes(t)) tracks.push(t);
        });
        // 也包含当前已指派的角色
        Object.values(this.cornerRoleAssign || {}).forEach(t => {
            if (!tracks.includes(t)) tracks.push(t);
        });
        if (tracks.length === 0) { this.updateStatus('没有需要清除的生成轨迹'); return; }
        try {
            await fetch(`${API_BASE}/corner_case/clear`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ scene_id: this.state.sceneId, track_ids: tracks })
            });
            this.cornerAffectedTracks = [];
            this.cornerSynthesizedTracks = [];
            this.state.lastQualityReport = null;
            this.state.lastSamplingParams = null;
            this._updateQualityReportButton();
            this.state.renderCache.clear();
            if (this.viewer3d) {
                this.viewer3d._invalidateCache();
                this.viewer3d.trajectory = null;
                this.viewer3d.clearCriticalFrames();
                if (this.viewer3d.selectedTrackId !== null) {
                    this.viewer3d._fetchTrajectory(this.viewer3d.selectedTrackId);
                }
            }
            await this.loadFrame(this.state.currentFrame);
            const statusEl = document.getElementById('cornerCaseStatus');
            if (statusEl) statusEl.textContent = '已清除生成的事故轨迹';
            this.updateStatus('已清除生成的事故轨迹');
        } catch (e) {
            console.error('清除失败:', e);
        }
    }

    // ==================== 导出 ====================

    showExportModal() {
        document.getElementById('exportModal').classList.add('active');
    }

    async exportScene() {
        const format = document.getElementById('exportFormat').value;
        const outputPath = document.getElementById('exportPathInput').value;
        
        if (!outputPath) {
            alert('请输入输出路径');
            return;
        }
        
        try {
            let endpoint = '';
            
            if (format === 'trajectory') {
                endpoint = `${API_BASE}/export/trajectory`;
            } else if (format === 'scene_spec') {
                endpoint = `${API_BASE}/export/scene_spec`;
            }
            
            const response = await fetch(`${endpoint}?scene_id=${this.state.sceneId}&output_path=${outputPath}`, {
                method: 'POST'
            });
            
            const data = await response.json();
            
            if (data.success) {
                document.getElementById('exportModal').classList.remove('active');
                this.updateStatus(`已导出到: ${outputPath}`);
            }
        } catch (error) {
            console.error('导出失败:', error);
        }
    }


    // ==================== 历史管理 ====================

    saveEditHistory(type, objectId) {
        this.state.editHistory.push({
            type: type,
            objectId: objectId,
            frame: this.state.currentFrame,
            timestamp: Date.now()
        });
        // 新操作会清空后端的 redo 栈，前端同步清空
        this.state.redoHistory = [];
        this.state.historyIndex = this.state.editHistory.length - 1;
        this.updateHistoryPanel();
    }

    updateHistoryPanel() {
        const historyList = document.getElementById('historyList');
        const undoBtn = document.getElementById('undoBtn');
        const redoBtn = document.getElementById('redoBtn');
        const hasRedo = this.state.redoHistory && this.state.redoHistory.length > 0;

        if (this.state.editHistory.length === 0) {
            historyList.innerHTML = '<div class="empty-state"><p>暂无编辑记录</p></div>';
            if (undoBtn) undoBtn.disabled = true;
            if (redoBtn) redoBtn.disabled = !hasRedo;
            return;
        }
        
        historyList.innerHTML = this.state.editHistory.slice(-10).reverse().map(entry => {
            const typeNames = {
                'move': '移动物体',
                'transform': '变换物体',
                'trajectory': '编辑轨迹',
                'delete': '删除物体',
                'corner_case': '生成事故场景'
            };
            const detail = entry.type === 'corner_case'
                ? (entry.objectId || '')
                : `T:${entry.objectId}`;
            return `
                <div class="history-item">
                    <span class="history-type">${typeNames[entry.type] || entry.type}</span>
                    <span class="history-detail">${detail}</span>
                </div>
            `;
        }).join('');
        
        if (undoBtn) undoBtn.disabled = false;
        if (redoBtn) redoBtn.disabled = !hasRedo;
    }

    async undo() {
        if (!this.state.sceneId) return;
        try {
            const r = await fetch(`${API_BASE}/undo/${this.state.sceneId}`, { method: 'POST' });
            const data = await r.json();
            if (data.success) {
                // 把最近一条历史移入 redo 栈
                if (this.state.editHistory.length > 0) {
                    const entry = this.state.editHistory.pop();
                    this.state.redoHistory = this.state.redoHistory || [];
                    this.state.redoHistory.push(entry);
                }
                await this._afterUndoRedo();
                this._syncHistoryButtons(data.can_undo, data.can_redo);
                this.updateStatus('已撤销');
            } else {
                this.updateStatus('没有可撤销的操作');
            }
        } catch (error) {
            console.error('撤销失败:', error);
        }
    }

    async redo() {
        if (!this.state.sceneId) return;
        try {
            const r = await fetch(`${API_BASE}/redo/${this.state.sceneId}`, { method: 'POST' });
            const data = await r.json();
            if (data.success) {
                if (this.state.redoHistory && this.state.redoHistory.length > 0) {
                    const entry = this.state.redoHistory.pop();
                    this.state.editHistory.push(entry);
                }
                await this._afterUndoRedo();
                this._syncHistoryButtons(data.can_undo, data.can_redo);
                this.updateStatus('已重做');
            } else {
                this.updateStatus('没有可重做的操作');
            }
        } catch (error) {
            console.error('重做失败:', error);
        }
    }

    _syncHistoryButtons(canUndo, canRedo) {
        const undoBtn = document.getElementById('undoBtn');
        const redoBtn = document.getElementById('redoBtn');
        if (undoBtn && canUndo !== undefined) undoBtn.disabled = !canUndo;
        if (redoBtn && canRedo !== undefined) redoBtn.disabled = !canRedo;
        this.updateHistoryPanel();
    }

    async _afterUndoRedo() {
        this.state.renderCache.clear();
        if (this.viewer3d) {
            this.viewer3d._invalidateCache();
            this.viewer3d.trajectory = null;
            if (this.viewer3d.selectedTrackId !== null) {
                this.viewer3d._fetchTrajectory(this.viewer3d.selectedTrackId);
            }
        }
        await this.loadFrame(this.state.currentFrame);
    }

    // ==================== UI更新 ====================

    updateObjectList() {
        const listContainer = document.getElementById('objectList');
        this._sam3dPopulateTargets();
        
        if (this.state.objects.length === 0) {
            listContainer.innerHTML = '<div class="empty-state"><p>当前帧无动态物体</p></div>';
            return;
        }
        
        listContainer.innerHTML = this.state.objects.map(obj => {
            const isSelected = this.state.selectedObjects.includes(obj.object_id);
            const isEdited = obj.edited;
            
            return `
                <div class="object-item ${isSelected ? 'selected' : ''} ${isEdited ? 'edited' : ''}"
                     data-object-id="${obj.object_id}"
                     onclick="studio.selectObject(${obj.object_id}, event)">
                    <div class="object-color" style="background-color: ${this.getObjectColor(obj.object_id)}"></div>
                    <div class="object-info">
                        <div class="object-name">物体 ${obj.object_id}</div>
                        <div class="object-type">${obj.type || '未知类型'}</div>
                    </div>
                    ${isEdited ? '<span class="edited-badge">已编辑</span>' : ''}
                </div>
            `;
        }).join('');
    }

    selectObject(objectId, event) {
        if (!event.shiftKey) {
            this.state.selectedObjects = [objectId];
        } else {
            const index = this.state.selectedObjects.indexOf(objectId);
            if (index >= 0) {
                this.state.selectedObjects.splice(index, 1);
            } else {
                this.state.selectedObjects.push(objectId);
            }
        }
        
        this.updateObjectList();
        this.updateSelectedObjectPanel();
        this.renderOverlay();
    }

    getObjectColor(objectId) {
        const colors = ['#ef4444', '#f59e0b', '#10b981', '#3b82f6', '#8b5cf6', '#ec4899'];
        return colors[objectId % colors.length];
    }

    filterObjects(query) {
        const items = document.querySelectorAll('.object-item');
        const lowerQuery = query.toLowerCase();
        
        items.forEach(item => {
            const name = item.querySelector('.object-name').textContent.toLowerCase();
            const type = item.querySelector('.object-type').textContent.toLowerCase();
            
            item.style.display = (name.includes(lowerQuery) || type.includes(lowerQuery)) ? '' : 'none';
        });
    }

    updateObjectCount() {
        document.getElementById('objectCount').textContent = `物体: ${this.state.objects.length}`;
        document.getElementById('selectedCount').textContent = `选中: ${this.state.selectedObjects.length}`;
    }

    updateStatus(message) {
        document.getElementById('statusText').textContent = message;
    }

    // ==================== SAM 3D 交互重建 ====================

    setupSam3dControls() {
        const toggle = document.getElementById('sam3dToggleBtn');
        const addBtn = document.getElementById('sam3dAddObjectBtn');
        if (toggle) toggle.addEventListener('click', () => this.openSamSourceModal());
        if (addBtn) addBtn.addEventListener('click', () => this._sam3dAddObject());

        const modelSel = document.getElementById('sam3dModelSelect');
        if (modelSel) modelSel.addEventListener('change', () => this._samSetModel(modelSel.value));
        this._samLoadModels();

        const previewBtn = document.getElementById('sam3dPreviewBtn');
        if (previewBtn) previewBtn.addEventListener('click', () => this.openSamPreview());

        const closeBtn = document.getElementById('samSourceCloseBtn');
        if (closeBtn) closeBtn.addEventListener('click', () => this.closeSamSourceModal());
        const bind = (id, fn) => { const el = document.getElementById(id); if (el) el.addEventListener('click', fn); };
        bind('sam3dPickTargetBtn', () => this._sam3dStartPickTarget());
        bind('samSrcPrevBtn', () => this._samSrcLoadFrame((this.samSrcFrameIdx ?? this.state.currentFrame) - 1, false));
        bind('samSrcNextBtn', () => this._samSrcLoadFrame((this.samSrcFrameIdx ?? this.state.currentFrame) + 1, false));
        bind('samSrcBestBtn', () => {
            const best = this.samSrcCandidates && this.samSrcCandidates[0];
            if (!best) { this._samSrcStatus('推荐帧还在计算中…', true); return; }
            this._samSrcLoadFrame(best.frame_idx, false);
            this._samSrcStatus(`已跳到推荐帧 #${best.frame_idx}（遮挡 ${Math.round((best.occlusion || 0) * 100)}%）`);
        });
        bind('samSrcFgBtn', () => this._samSrcSetType('fg'));
        bind('samSrcBgBtn', () => this._samSrcSetType('bg'));
        bind('samSrcUndoBtn', () => this._samSrcUndo());
        bind('samSrcClearBtn', () => this._samSrcClear());
        bind('samSrcGenerateBtn', () => this._samSrcGenerate());

        const canvas = document.getElementById('samSourceCanvas');
        if (canvas) {
            canvas.addEventListener('mousedown', (e) => this._samSrcHandleClick(e));
            canvas.addEventListener('contextmenu', (e) => { e.preventDefault(); this._samSrcUndo(); });
        }

        // 预览模态框
        bind('samPreviewCloseBtn', () => this.closeSamPreview());
        bind('samPreviewPlayBtn', () => this._samPreviewPlay());
        bind('samPreviewStopBtn', () => this._samPreviewStop());
        const slider = document.getElementById('samPreviewSlider');
        if (slider) slider.addEventListener('input', () => this._samPreviewSeek(parseInt(slider.value, 10)));

        // 替换后模型大小微调（相对后端自动算出的基准尺寸）
        const scaleRange = document.getElementById('sam3dScaleRange');
        if (scaleRange) {
            scaleRange.addEventListener('input', () => {
                const el = document.getElementById('sam3dScaleVal');
                if (el) el.textContent = Math.round((parseFloat(scaleRange.value) || 1) * 100) + '%';
            });
            scaleRange.addEventListener('change', () => {
                this._sam3dApplyScale(parseFloat(scaleRange.value) || 1.0);
            });
        }
    }

    // ==================== 文本添加实体（LLaDA-Image + SAM3D） ====================

    _nlEntityStatus(msg, isError = false) {
        const el = document.getElementById('nlEntityStatus');
        if (el) { el.textContent = msg; el.style.color = isError ? '#e5484d' : 'inherit'; }
        if (msg) this.updateStatus('文本实体: ' + msg);
    }

    setupNlEntityControls() {
        const btn = document.getElementById('nlEntityBtn');
        if (btn) btn.addEventListener('click', () => this.nlEntityGenerate());
        const rb = document.getElementById('nlEntityReplanBtn');
        if (rb) rb.addEventListener('click', () => this.nlEntityReplan());
        const tp = document.getElementById('trajEditPlanBtn');
        if (tp) tp.addEventListener('click', () => this.trajEdit('plan'));
        const ta = document.getElementById('trajEditApplyBtn');
        if (ta) ta.addEventListener('click', () => this.trajEdit('apply'));
        this.nlEntityHealth();
    }

    async nlEntityHealth() {
        try {
            const r = await fetch(`${API_BASE}/text2entity/health`);
            const d = await r.json();
            if (d && d.success && d.status === 'ok') {
                this._nlEntityStatus((d.prompt2image && d.vlm)
                    ? '服务就绪（LLaDA-Image + Qwen2.5-VL-3B）'
                    : '微服务在线，但模型目录缺失（请先下载权重）', !(d.prompt2image && d.vlm));
            } else if (d && d.local_text2image) {
                this._nlEntityStatus('LLaDA 微服务未启动，将用本地 sd-turbo 兜底出图（VLM 退回关键词先验）');
            } else {
                this._nlEntityStatus('微服务未启动：'
                    + ((d && (d.detail || d.error)) || '请运行 bash text2entity/run_service.sh'), true);
            }
        } catch (e) {
            this._nlEntityStatus('微服务未启动（' + e.message + '）：请运行 bash text2entity/run_service.sh', true);
        }
    }

    async nlEntityGenerate() {
        if (!this.state.sceneId) { alert('请先加载场景'); return; }
        const prompt = (document.getElementById('nlEntityPrompt')?.value || '').trim();
        if (!prompt) { alert('请先输入要添加的物体描述'); return; }
        // 可选：用户上传参考图 → 跳过文生图，直接用它做 SAM3D 重建（质量最可控）
        let referenceImage = null;
        const fileEl = document.getElementById('nlEntityRefFile');
        if (fileEl && fileEl.files && fileEl.files[0]) {
            referenceImage = await new Promise((resolve, reject) => {
                const fr = new FileReader();
                fr.onload = () => resolve(String(fr.result).split(',')[1]);
                fr.onerror = reject;
                fr.readAsDataURL(fileEl.files[0]);
            });
        }
        const body = {
            scene_id: this.state.sceneId,
            prompt,
            frame_idx: this.state.currentFrame,
            mode: document.getElementById('nlEntityMode')?.value || 'ahead',
            distance: parseFloat(document.getElementById('nlEntityDistance')?.value || '12') || 0,
            speed: parseFloat(document.getElementById('nlEntitySpeed')?.value || '0') || 0,
            num_frames: parseInt(document.getElementById('nlEntityFrames')?.value || '20', 10) || 20,
            use_vlm: true,
        };
        if (referenceImage) body.reference_image = referenceImage;
        const btn = document.getElementById('nlEntityBtn');
        if (btn) btn.disabled = true;
        this._nlEntityStatus('正在出图/重建（三个大模型串行，首次约 1~3 分钟，请耐心等待）…');
        const t0 = Date.now();
        try {
            const resp = await fetch(`${API_BASE}/text2entity/generate`, {
                method: 'POST', headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(body),
            });
            const d = await resp.json();
            if (!resp.ok || !d.success) throw new Error(d.detail || ('HTTP ' + resp.status));
            const cat = d.category || 'object';
            const dims = (d.dimensions || []).map(v => Number(v).toFixed(2)).join(' × ');
            const desc = (d.vlm && d.vlm.short_desc) ? `（${d.vlm.short_desc}）` : '';
            const att = (d.vlm && d.vlm.attempts) ? `，尝试 ${d.vlm.attempts} 次` : '';
            this._nlEntityStatus(`已插入「${cat}」#${d.object_id}${desc}（${dims} m）· ${(d.placement || {}).mode || ''}`
                + `${att} · 用时 ${((Date.now() - t0) / 1000).toFixed(0)}s`);
            if (d.reference_image) {
                const grp = document.getElementById('nlEntityPreviewGroup');
                const img = document.getElementById('nlEntityPreviewImg');
                if (img) img.src = 'data:image/png;base64,' + d.reference_image;
                if (grp) grp.style.display = 'block';
            }
            this.state.selectedObjects = [d.object_id];
            this._invalidateRenderCaches();
            await this.loadFrame(this.state.currentFrame);
            if (this.viewer3d && typeof this.viewer3d.loadFrame === 'function') {
                await this.viewer3d.loadFrame(this.state.sceneId, this.state.currentFrame, true);
            }
            this.updateSelectedObjectPanel();
        } catch (e) {
            this._nlEntityStatus('生成失败：' + e.message, true);
        } finally {
            if (btn) btn.disabled = false;
        }
    }

    // 重排选中物体：沿车道重新生成"物理合理 + 无冲突"的轨迹（保留模型）
    async nlEntityReplan() {
        if (!this.state.sceneId) { alert('请先加载场景'); return; }
        const tid = this.state.selectedObjects[0];
        if (tid === undefined || tid === null) { alert('请先在画布中选中一个已插入的物体'); return; }
        const body = {
            scene_id: this.state.sceneId,
            track_id: tid,
            mode: document.getElementById('nlEntityMode')?.value || 'ahead',
            distance: parseFloat(document.getElementById('nlEntityDistance')?.value || '12') || 0,
            speed: parseFloat(document.getElementById('nlEntitySpeed')?.value || '0') || 0,
            num_frames: parseInt(document.getElementById('nlEntityFrames')?.value || '20', 10) || 20,
            start_frame: this.state.currentFrame,
        };
        const btn = document.getElementById('nlEntityReplanBtn');
        if (btn) btn.disabled = true;
        this._nlEntityStatus(`正在为 #${tid} 重排无冲突轨迹…`);
        try {
            const resp = await fetch(`${API_BASE}/text2entity/replan`, {
                method: 'POST', headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(body),
            });
            const d = await resp.json();
            if (!resp.ok || !d.success) throw new Error(d.detail || ('HTTP ' + resp.status));
            const p = d.placement || {};
            this._nlEntityStatus(`已重排 #${d.track_id}：${p.mode || ''} 横向 ${Number(p.lateral || 0).toFixed(1)}m / `
                + `距离 ${Number(p.distance || 0).toFixed(1)}m，${d.num_frames} 帧，0 冲突`);
            this._invalidateRenderCaches();
            await this.loadFrame(this.state.currentFrame);
            if (this.viewer3d && typeof this.viewer3d.loadFrame === 'function') {
                await this.viewer3d.loadFrame(this.state.sceneId, this.state.currentFrame, true);
            }
            this.updateSelectedObjectPanel();
        } catch (e) {
            this._nlEntityStatus('重排失败：' + e.message, true);
        } finally {
            if (btn) btn.disabled = false;
        }
    }

    // ==================== 语言编辑轨迹（LLM） ====================

    _trajEditStatus(msg) {
        const el = document.getElementById('trajEditStatus');
        if (el) el.textContent = msg;
        if (msg) this.updateStatus('轨迹编辑: ' + msg);
    }

    _trajEditPreview(obj) {
        const el = document.getElementById('trajEditPreview');
        if (el) el.textContent = typeof obj === 'string' ? obj : JSON.stringify(obj, null, 1);
    }

    // 显示"生成新物体"时实际用的参考图 + 插入预览（用户要求：生成了新物体要能看到参考图）
    _trajEditShowInserts(inserts) {
        const box = document.getElementById('trajEditImages');
        if (!box) return;
        const list = inserts || [];
        const it = list[0];
        if (!it || (!it.reference_image && !it.preview)) { box.style.display = 'none'; return; }
        const refImg = document.getElementById('trajEditRefImg');
        const prevImg = document.getElementById('trajEditPrevImg');
        const label = document.getElementById('trajEditRefLabel');
        if (refImg) refImg.src = it.reference_image ? ('data:image/png;base64,' + it.reference_image) : '';
        if (prevImg) prevImg.src = it.preview ? ('data:image/png;base64,' + it.preview) : '';
        if (label) {
            const bits = [`参考图（实际用于重建） · T:${it.object_id}`];
            if (it.prompt) bits.push(it.prompt);
            if (it.dimensions) bits.push(`包围盒 ${it.dimensions.map(v => v.toFixed(2)).join('×')}m`);
            if (it.dims_target) bits.push(`目标 ${it.dims_target.map(v => v.toFixed(2)).join('×')}m`);
            if (it.num_frames) bits.push(`${it.num_frames} 帧`);
            if (it.bbox_mode) bits.push(it.bbox_mode === 'tight_to_model' ? '框紧贴模型' : it.bbox_mode);
            if (it.recon_degenerate) bits.push('重建退化→已换通用车模');
            label.textContent = bits.join(' · ');
        }
        box.style.display = 'block';
    }

    async trajEdit(mode) {
        if (!this.state.sceneId) { alert('请先加载场景'); return; }
        const instruction = (document.getElementById('trajEditInstruction')?.value || '').trim();
        if (!instruction) { alert('请先输入要做的轨迹编辑'); return; }
        const path = mode === 'apply' ? '/traj_edit/apply' : '/traj_edit/plan';
        const nfEl = document.getElementById('trajEditNumFrames');
        const panelFrames = Math.max(2, Math.min(600, parseInt(nfEl?.value || '30', 10) || 30));
        const body = {
            scene_id: this.state.sceneId, instruction,
            frame_idx: this.state.currentFrame, num_frames: panelFrames,
            new_role: (document.getElementById('trajEditNewRole')?.value) || 'auto',
        };
        // 可选参考图：指令里有"生成一辆车"时，直接用它做 SAM3D（跳过文生图）
        const refEl = document.getElementById('trajEditRefFile');
        if (mode === 'apply' && refEl && refEl.files && refEl.files[0]) {
            try {
                body.reference_image = await new Promise((resolve, reject) => {
                    const fr = new FileReader();
                    fr.onload = () => resolve(String(fr.result).split(',')[1]);
                    fr.onerror = reject;
                    fr.readAsDataURL(refEl.files[0]);
                });
            } catch (e) { /* 读图失败就当没给 */ }
        }
        this._trajEditStatus(mode === 'apply' ? '正在规划并应用（含 insert 时较慢）…' : '正在用 LLM 解析…');
        const t0 = Date.now();
        try {
            const resp = await fetch(`${API_BASE}${path}`, {
                method: 'POST', headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(body),
            });
            const d = await resp.json();
            if (!resp.ok || !d.success) throw new Error(d.detail || ('HTTP ' + resp.status));
            const warns = d.warnings || [];
            // 有操作没成功（例如"窗口内撞不上"）时，把原因也说清楚，别只显示"完成"
            const fails = (d.report || []).filter(r => r && r.ok === false && (r.error || r.warn));
            for (const r of fails) {
                const tag = r.op === 'collide' ? `相撞 T:${r.a}↔T:${r.b}` : String(r.op || '操作');
                warns.push(`${tag}未生效：${r.error || r.warn}`);
            }
            this._trajEditStatus(`完成（解析来源 ${d.source || '?'}，${(d.ops || []).length} 个操作，`
                + `${((Date.now() - t0) / 1000).toFixed(0)}s）`
                + (warns.length ? ` 注意：${warns[0]}` : ''));
            this._trajEditPreview({ source: d.source, warnings: warns, ops: d.ops,
                                    inserted: d.inserted, report: d.report });
            this._trajEditShowInserts(d.inserts);
            this._invalidateRenderCaches();
            await this.loadFrame(this.state.currentFrame);
            if (this.viewer3d && typeof this.viewer3d.loadFrame === 'function') {
                await this.viewer3d.loadFrame(this.state.sceneId, this.state.currentFrame, true);
            }
            this.updateSelectedObjectPanel();
        } catch (e) {
            this._trajEditStatus('失败：' + e.message);
        }
    }

    _sam3dShowScaleControl(fit = 1.0) {
        const grp = document.getElementById('sam3dScaleGroup');
        const rng = document.getElementById('sam3dScaleRange');
        const val = document.getElementById('sam3dScaleVal');
        if (grp) grp.style.display = 'block';
        if (rng) rng.value = String(fit);
        if (val) val.textContent = Math.round(fit * 100) + '%';
    }

    async _sam3dApplyScale(factor) {
        if (!this.state.sceneId || !this.lastSamObjectId) return;
        try {
            const resp = await fetch(`${API_BASE}/edit/synthetic/scale`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    scene_id: this.state.sceneId,
                    track_id: this.lastSamObjectId,
                    factor: Math.max(0.2, Math.min(5.0, factor)),
                }),
            });
            const d = await resp.json();
            if (!resp.ok || !d.success) throw new Error(d.detail || ('HTTP ' + resp.status));
            this._sam3dStatus(`替换模型大小已调整为基准的 ${Math.round((d.factor || factor) * 100)}%`);
            this._invalidateRenderCaches();
            await this.loadFrame(this.state.currentFrame);
            if (this.viewer3d && typeof this.viewer3d.loadFrame === 'function') {
                await this.viewer3d.loadFrame(this.state.sceneId, this.state.currentFrame, true);
            }
        } catch (e) {
            this._sam3dStatus('调整模型大小失败: ' + e.message, true);
        }
    }

    // ==================== 车头朝向运动方向 ====================

    setupAutoHeadingControls() {
        const sm = document.getElementById('autoHeadingSmooth');
        if (sm) {
            sm.addEventListener('input', () => {
                const v = parseFloat(sm.value) || 0;
                const el = document.getElementById('autoHeadingSmoothVal');
                if (el) el.textContent = v.toFixed(2);
                if (this._autoHeadingTimer) clearTimeout(this._autoHeadingTimer);
                this._autoHeadingTimer = setTimeout(() => this._autoHeadingApply({ smoothing: v }), 120);
            });
        }
    }

    async _autoHeadingLoad() {
        if (!this.state.sceneId) return;
        try {
            const resp = await fetch(`${API_BASE}/edit/auto_heading/${this.state.sceneId}?frame_idx=${this.state.currentFrame || 0}`);
            const d = await resp.json();
            if (!d || !d.success) return;
            // 该功能默认开启且作用于全部动态物体；若后端被关掉了则自动重新开启，保证"无需手动应用"
            if (!d.enabled || !d.all_tracks) {
                await this._autoHeadingApply({ enabled: true, all_tracks: true, smoothing: d.smoothing });
                return;
            }
            this._syncAutoHeadingUI(d);
        } catch (e) {
            /* 忽略 */
        }
    }

    _syncAutoHeadingUI(d) {
        if (d.smoothing !== undefined && d.smoothing !== null) {
            const sm = document.getElementById('autoHeadingSmooth');
            if (sm) sm.value = d.smoothing;
            const el = document.getElementById('autoHeadingSmoothVal');
            if (el) el.textContent = Number(d.smoothing).toFixed(2);
        }
        const scope = d.all_tracks ? '全部动态物体' : `选中的 ${(d.track_ids || []).length} 个物体`;
        this._autoHeadingStatus(d.enabled
            ? `已开启：${scope} 车头朝向运动方向（平滑 ${Number(d.smoothing || 0).toFixed(2)}）`
            : '未开启');
    }

    async _autoHeadingApply(payload) {
        if (!this.state.sceneId) { this._autoHeadingStatus('请先加载场景', true); return; }
        try {
            const resp = await fetch(`${API_BASE}/edit/auto_heading`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ scene_id: this.state.sceneId, ...payload }),
            });
            const d = await resp.json();
            if (!resp.ok || !d.success) throw new Error(d.detail || ('HTTP ' + resp.status));
            this._syncAutoHeadingUI(d);
            this._invalidateRenderCaches();
            await this.loadFrame(this.state.currentFrame);
        } catch (e) {
            this._autoHeadingStatus('设置自动朝向失败: ' + e.message, true);
        }
    }

    _autoHeadingStatus(msg, isError = false) {
        const el = document.getElementById('autoHeadingStatus');
        if (el) {
            el.textContent = msg;
            el.style.color = isError ? '#e5484d' : 'inherit';
        }
    }

    // ==================== 实验台（测试各改动） ====================

    // ==================== 主车（EGO）实体化 ====================
    async loadEgoInfo() {
        const box = document.getElementById('egoStatus');
        if (!this.state.sceneId) {
            if (box) box.textContent = '请先加载场景。';
            return null;
        }
        try {
            const resp = await fetch(`${API_BASE}/ego/${encodeURIComponent(this.state.sceneId)}`);
            const d = await resp.json();
            this.state.ego = d;
            const vis = document.getElementById('egoVisible');
            if (vis && d.available) vis.checked = !!d.visible;
            this._fillEgoSource(d);
            if (box) {
                if (!d.available) {
                    box.innerHTML = '未启用主车实体（' + (d.reason || '未知原因') + '）';
                } else {
                    const src = d.source_track_id === null || d.source_track_id === undefined
                        ? '真实主车（自车相机）' : `track ${d.source_track_id}（${d.source_type || ''}）`;
                    const g = d.ground || {};
                    const groundTxt = (d.auto_ground && g.applied)
                        ? ` · <span title="按场景地面自动抬升，避免穿模">已自动贴地(相机高≈${Number(g.cam_height_median || 0).toFixed(2)}m)</span>`
                        : '';
                    box.innerHTML = `当前视角：<b>${this._escapeHtml(src)}</b>${groundTxt}`;
                }
            }
            return d;
        } catch (e) {
            if (box) box.textContent = '主车信息获取失败: ' + e.message;
            return null;
        }
    }

    _fillEgoSource(d) {
        const sel = document.getElementById('egoSource');
        if (!sel) return;
        if (!d || !d.available) {
            sel.innerHTML = '<option value="">真实主车（自车相机）</option>';
            sel.disabled = true;
            return;
        }
        const cur = d.source_track_id === null || d.source_track_id === undefined ? '' : String(d.source_track_id);
        const opts = ['<option value="">真实主车（自车相机）</option>'];
        (d.options || []).forEach(o => {
            const label = `${o.track_id} · ${o.type || ''}（${o.num_frames || 0}帧）`;
            opts.push(`<option value="${o.track_id}">${this._escapeHtml(label)}</option>`);
        });
        sel.innerHTML = opts.join('');
        sel.value = cur;
        sel.disabled = false;
    }

    async egoSetSource(v) {
        const id = String(v || '').trim();
        const d = await this._egoPost('/ego', {
            set_source: true,
            source_track_id: id === '' ? null : parseInt(id, 10) });
        if (d) this.updateStatus(`主车视角已切换到：${d.source_type || '真实主车'}`);
    }

    async _egoPost(path, body) {
        if (!this.state.sceneId) { alert('请先加载场景'); return null; }
        try {
            const resp = await fetch(`${API_BASE}${path}`, {
                method: 'POST', headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(Object.assign({ scene_id: this.state.sceneId }, body || {}))
            });
            const d = await resp.json();
            if (!d.success) throw new Error(d.detail || '操作失败');
            this._invalidateRenderCaches();
            await this.loadFrame(this.state.currentFrame);
            if (this.viewer3d && typeof this.viewer3d.loadFrame === 'function') {
                await this.viewer3d.loadFrame(this.state.sceneId, this.state.currentFrame, true);
            }
            await this.loadEgoInfo();
            this.updateSelectedObjectPanel();
            return d;
        } catch (e) {
            alert('主车操作失败: ' + e.message);
            return null;
        }
    }

    async egoSetVisible(visible) {
        await this._egoPost('/ego', { visible: !!visible });
        this.updateStatus(visible ? '主车实体：已显示' : '主车实体：已隐藏');
    }

    // 自动贴地：让后端按当前 4DGS 场景的地面重新摆放主车（不同场景相机高度不同）
    async egoRefitGround() {
        if (!this.state.sceneId) { alert('请先加载场景'); return; }
        const d = await this._egoPost('/ego/ground', { auto: true });
        if (!d) return;
        const g = d.ground || d.info || {};
        if (g && g.applied) {
            this.updateStatus(`主车已自动贴地：场景地面 y≈${Number(g.ground_y_median || 0).toFixed(2)}，`
                + `相机高≈${Number(g.cam_height_median || 0).toFixed(2)}m`);
        } else {
            this.updateStatus('主车自动贴地未生效：' + ((g && g.reason) || '未知原因'));
        }
    }

    setupLabControls() {
        const openBtn = document.getElementById('labBtn');
        if (openBtn) openBtn.addEventListener('click', () => this.openLab());
        const closeBtn = document.getElementById('labCloseBtn');
        if (closeBtn) closeBtn.addEventListener('click', () => this.closeLab());

        document.querySelectorAll('.lab-tab').forEach(tab => {
            tab.addEventListener('click', () => this._labSwitchTab(tab.dataset.tab));
        });

        const graphRun = document.getElementById('labGraphRunBtn');
        if (graphRun) graphRun.addEventListener('click', () => this.labLoadGraph());
        const batchRun = document.getElementById('labBatchRunBtn');
        if (batchRun) batchRun.addEventListener('click', () => this.labRunBatch());
        const gnnRun = document.getElementById('labGnnCollectBtn');
        if (gnnRun) gnnRun.addEventListener('click', () => this.labCollectGnn());
        const stRun = document.getElementById('labSelfTestBtn');
        if (stRun) stRun.addEventListener('click', () => this.labSelfTest());

        // ⑤ 闭环仿真工具
        const bind = (id, fn) => {
            const el = document.getElementById(id);
            if (el) el.addEventListener('click', fn);
        };
        bind('simTrustModelBtn', () => this.simTrustModel());
        bind('simEnvelopeBtn', () => this.simEnvelope());
        bind('simTrajBtn', () => this.simTrajectory());
        bind('simBankBuildBtn', () => this.simBuildBank());
        bind('simBankInfoBtn', () => this.simBankInfo());
        bind('simMvPlanBtn', () => this.simMvPlan());
        bind('simMvRunBtn', () => this.simMvRun());
        bind('simMvLogBtn', () => this.simMvLog());
        bind('simDifixStatusBtn', () => this.simDifixStatus());
        bind('simRigBtn', () => this.simRenderRig());
        bind('simRunBtn', () => this.simRollout());
        bind('simExpBtn', () => this.simExportScenario());
        bind('simDemoRunBtn', () => this.simDemoRun());
        bind('simDemoReportBtn', () => this.simDemoReport());
        bind('simAbRunBtn', () => this.simAbRun());
        bind('simAbReportBtn', () => this.simAbReport());

        // ⑥ CARLA 仿真
        bind('carlaStatusBtn', () => this.carlaRefreshStatus(1));
        bind('carlaStartBtn', () => this.carlaServer('start'));
        bind('carlaStopBtn', () => this.carlaServer('stop'));
        bind('carlaEnvBtn', () => this.carlaEnvCheck());
        bind('carlaSceneReloadBtn', () => this.carlaLoadScenarios());
        bind('carlaRenderBtn', () => this.carlaRender());
        bind('engStartBtn', () => this.carlaEngineStart());
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
        bind('simDifixRefineBtn', () => this.simDifixRefine());
        bind('simDifixSeqBtn', () => this.simDifixSequence());
        const _rnEl = document.getElementById('carlaRunName');
        if (_rnEl) _rnEl.addEventListener('input', () => this._carlaWillRun());
        const _rsEl = document.getElementById('carlaRoadSnap');
        if (_rsEl) _rsEl.addEventListener('change', () => this._carlaWillRun());
        const _ucEl = document.getElementById('carlaUseCurrent');
        if (_ucEl) _ucEl.addEventListener('change', () => this._carlaWillRun());
        bind('carlaXoscBtn', () => this.carlaExportXosc());
        bind('carlaSrBtn', () => this.carlaScenarioRunner());
        bind('carlaCancelBtn', () => this.carlaCancelJob());
        // Corner Case 面板里的快捷入口：直接跳到 ⑥ CARLA 仿真
        bind('carlaQuickBtn', () => { this.openLab(); this._labSwitchTab('carla'); });
        // 流程串联：上一步/下一步 + ③→④ 直达 + 释放显存
        bind('labNextStepBtn', () => this._labStep(1));
        bind('labPrevStepBtn', () => this._labStep(-1));
        bind('simToCarlaBtn', () => this._labGoCarla());
        bind('carlaReleaseBtn', () => this.carlaRelease());

        const videoClose = document.getElementById('labVideoCloseBtn');
        if (videoClose) videoClose.addEventListener('click', () => this.closeVideoModal());
        const tabEgo = document.getElementById('labVideoTabEgo');
        if (tabEgo) tabEgo.addEventListener('click', () => this._labSwitchVideo('ego'));
        const tabBev = document.getElementById('labVideoTabBev');
        if (tabBev) tabBev.addEventListener('click', () => this._labSwitchVideo('bev'));
        const tabTop = document.getElementById('labVideoTabTop');
        if (tabTop) tabTop.addEventListener('click', () => this._labSwitchVideo('top'));
        const caseVideoBtn = document.getElementById('renderCaseVideoBtn');
        if (caseVideoBtn) caseVideoBtn.addEventListener('click', () => this.labRenderCaseVideo());

        // 主车（EGO）控件：显示开关 + "以谁的视角渲染" + 自动贴地
        const egoVis = document.getElementById('egoVisible');
        if (egoVis) egoVis.addEventListener('change', () => this.egoSetVisible(egoVis.checked));
        const egoSrc = document.getElementById('egoSource');
        if (egoSrc) egoSrc.addEventListener('change', () => this.egoSetSource(egoSrc.value));
        const egoGroundBtn = document.getElementById('egoGroundBtn');
        if (egoGroundBtn) egoGroundBtn.addEventListener('click', () => this.egoRefitGround());

        // 事故类型复选框
        const typesBox = document.getElementById('labBatchTypes');
        if (typesBox) {
            typesBox.innerHTML = LAB_SCENARIOS.map(([v, label]) => {
                const checked = ['rear-end', 'head-on', 'lane-change-cutin'].includes(v) ? 'checked' : '';
                return `<label class="cc-checkbox"><input type="checkbox" class="lab-batch-type" value="${v}" ${checked}>${label}</label>`;
            }).join('');
        }
    }

    openLab() {
        const modal = document.getElementById('labModal');
        if (modal) modal.classList.add('active');
        this._labRenderFlow();
        if (this.state.sceneId) {
            const frameEl = document.getElementById('labGraphFrame');
            if (frameEl && !frameEl.dataset.touched) frameEl.value = this.state.currentFrame || 0;
            this._labGnnCommand('');
        } else {
            this._labStatus('labGraphStatus', '请先加载场景（顶部「加载场景」）。');
        }
    }

    closeLab() {
        const modal = document.getElementById('labModal');
        if (modal) modal.classList.remove('active');
    }

    // ==================== ⑤ 闭环仿真工具 ====================
    _simStatus(id, msg, err) {
        const el = document.getElementById(id);
        if (el) { el.textContent = msg; el.style.color = err ? '#fca5a5' : ''; }
    }

    _simRows(headers, rows) {
        if (!rows.length) return '<div class="cc-status">（无结果）</div>';
        return `<table class="lab-table"><thead><tr>${headers.map(h => `<th>${h}</th>`).join('')}</tr></thead>`
            + `<tbody>${rows.map(r => `<tr>${r.map(c => `<td>${c}</td>`).join('')}</tr>`).join('')}</tbody></table>`;
    }

    async _simPost(url, body, statusId, label) {
        if (!this.state.sceneId && url.indexOf('/scene') === -1 && url.indexOf('multiview') === -1) {
            this._simStatus(statusId, '请先加载场景。', true); return null;
        }
        try {
            const resp = await fetch(`${API_BASE}${url}`, {
                method: 'POST', headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(body || {})
            });
            const d = await resp.json();
            if (d.success === false || d.detail) throw new Error(d.detail || '失败');
            return d;
        } catch (e) {
            this._simStatus(statusId, `${label}失败: ${e.message}`, true);
            return null;
        }
    }

    async simTrustModel() {
        this._simStatus('simTrustStatus', '正在标定 trust 模型…');
        const d = await this._simPost('/trust/model', {}, 'simTrustStatus', '标定');
        if (!d) return;
        const m = d.model || {};
        this._simStatus('simTrustStatus',
            `标定完成：样本 ${d.num_rows}，两特征 R²=${m.r2}（只用 coverage 时 ${m.r2_coverage_only}），`
            + `模型 SSIM ≈ ${m.y0} + ${m.y1}·cov^${m.p}·exp(-ang/${m.tau_deg}°)`);
        const el = document.getElementById('simTrustOut');
        if (el) el.innerHTML = this._simRows(['指标', '值'], [
            ['样本数', d.num_rows], ['R²（coverage+视角）', m.r2], ['R²（仅 coverage）', m.r2_coverage_only],
            ['p', m.p], ['tau(度)', m.tau_deg], ['报告数', (d.reports || []).length]
        ]);
    }

    async simEnvelope() {
        const minTrust = parseFloat(document.getElementById('simTrustMin')?.value || '0.55');
        const sweep = (document.getElementById('simSweep')?.value || '0,1,2,3,4')
            .split(',').map(v => parseFloat(v)).filter(v => !isNaN(v));
        this._simStatus('simTrustStatus', '正在扫描可信域…');
        const d = await this._simPost('/trust/envelope', {
            scene_id: this.state.sceneId, frame: this.state.currentFrame || 0,
            lateral_offsets: sweep, min_trust: minTrust }, 'simTrustStatus', '可信域扫描');
        if (!d) return;
        const env = d.envelope || {};
        this._simStatus('simTrustStatus', `可信域：最大安全横移 ${env.max_safe_lateral_m === null
            ? '无（当前视角已不可信）' : env.max_safe_lateral_m + ' m'}（阈值 ${minTrust}）`);
        const el = document.getElementById('simTrustOut');
        if (el) el.innerHTML = this._simRows(['横移(m)', 'coverage', '预测 SSIM', '可信'],
            (env.rows || []).map(r => [r.lateral_m, r.coverage, r.pred_ssim,
                r.trusted ? '<span class="pass">是</span>' : '<span class="fail">否</span>']));
    }

    async simTrajectory() {
        const minTrust = parseFloat(document.getElementById('simTrustMin')?.value || '0.55');
        const num = parseInt(document.getElementById('cornerNumFrames')?.value) || 20;
        this._simStatus('simTrustStatus', '正在逐帧判定…');
        const d = await this._simPost('/trust/trajectory', {
            scene_id: this.state.sceneId, frame: this.state.currentFrame || 0, num_frames: num,
            min_trust: minTrust }, 'simTrustStatus', '轨迹 trust');
        if (!d) return;
        const rep = d.report || {};
        this._simStatus('simTrustStatus', `信任 ${rep.trusted_frames}/${rep.num_frames} 帧`
            + (rep.truncate_frame !== null && rep.truncate_frame !== undefined
                ? `；越界截断帧 = ${rep.truncate_frame}（${rep.truncate_reason || ''}）` : '；未越界'));
        const el = document.getElementById('simTrustOut');
        if (el) el.innerHTML = this._simRows(['帧', 'coverage', '预测 SSIM', '可信', '离训练视角(m)', '夹角(°)'],
            (rep.frames || []).map(f => [f.frame, f.coverage, f.pred_ssim,
                f.trusted ? '是' : '否', f.d_trans_m, f.d_view_angle_deg]));
    }

    async simBuildBank() {
        const solidify = !!document.getElementById('simBankSolidify')?.checked;
        this._simStatus('simBankStatus', '正在构建 actor 资产库…（含跨帧关联，几秒）');
        const d = await this._simPost('/actor_assets/build',
            { scene_id: this.state.sceneId, solidify }, 'simBankStatus', '构建资产库');
        if (!d) return;
        this._simStatus('simBankStatus', `完成：${d.num_tracks} 条 track，可用 ${d.num_usable}；`
            + `bank.json = ${d.bank_path}`);
        const el = document.getElementById('simBankOut');
        if (el) el.innerHTML = this._simRows(['资产', 'track', '高斯数', '尺寸(宽高长)', '速度(m/s)', '可用'],
            (d.assets || []).map(a => [this._escapeHtml(a.asset_id), a.track_id, a.num_gaussians,
                (a.dimensions || []).map(v => Number(v).toFixed(1)).join('×'),
                a.mean_speed_mps, a.usable ? '是' : '否']));
    }

    async simBankInfo() {
        try {
            const r = await fetch(`${API_BASE}/actor_assets/${encodeURIComponent(this.state.sceneId)}`);
            const d = await r.json();
            this._simStatus('simBankStatus', d.available
                ? `已有资产库：${d.num_usable}/${d.num_tracks} 可用（${d.path}）`
                : '当前场景还没有资产库 → 点「构建 actor 资产库」；构建后俯视 3D 会用真模型渲染。');
        } catch (e) {
            this._simStatus('simBankStatus', '查询失败: ' + e.message, true);
        }
    }

    _simMvBody() {
        const cameras = (document.getElementById('simMvCameras')?.value || '0,1,2')
            .split(',').map(v => parseInt(v)).filter(v => !isNaN(v));
        const holdout = (document.getElementById('simMvHoldout')?.value || '')
            .split(',').map(v => parseInt(v)).filter(v => !isNaN(v));
        return {
            scene: (document.getElementById('simMvScene')?.value || '001').trim(),
            cameras, frames: parseInt(document.getElementById('simMvFrames')?.value) || 8,
            holdout, do_eval: !!document.getElementById('simMvEval')?.checked,
            do_bank: !!document.getElementById('simMvBank')?.checked
        };
    }

    async simMvPlan() {
        const d = await this._simPost('/multiview/plan', this._simMvBody(), 'simMvStatus', '预算检查');
        if (!d) return;
        this._simStatus('simMvStatus', `预算：${d.num_images} 张图 / 上限 ${d.max_images} → `
            + (d.fits_budget ? '可以跑' : '会 OOM，请减少帧数或相机'));
        const el = document.getElementById('simMvOut');
        if (el) el.innerHTML = `<div class="cc-status">命令：<code>${this._escapeHtml(d.cmd)}</code></div>`;
    }

    async simMvRun() {
        const body = Object.assign(this._simMvBody(), { confirm: true });
        const d = await this._simPost('/multiview/run', body, 'simMvStatus', '启动');
        if (!d) return;
        this._simStatus('simMvStatus', `已在后台启动（pid ${d.pid}）；日志：${d.log_path}`);
        this._simMvLastLog = d.log_path;
        const lp = document.getElementById('simMvLogPath');
        if (lp) lp.value = d.log_path;
        const el = document.getElementById('simMvOut');
        if (el) el.innerHTML = `<div class="cc-status">${this._escapeHtml(d.cmd)}</div>`;
    }

    async simMvLog() {
        const out = document.getElementById('simMvOut');
        const path = (this._simMvLastLog || '');
        if (!path) { this._simStatus('simMvStatus', '先点「后台启动」，或把日志路径填进下面输入框。'); }
        const el = document.getElementById('simMvLogPath');
        const logPath = (el && el.value) || path;
        if (!logPath) return;
        this._simMvLastLog = logPath;
        try {
            const r = await fetch(`${API_BASE}/multiview/log?path=${encodeURIComponent(logPath)}&tail=60`);
            const d = await r.json();
            if (!d.success) throw new Error(d.detail || '读取失败');
            if (out) out.innerHTML = `<pre class="lab-code">${this._escapeHtml((d.tail || []).join('\n'))}</pre>`;
        } catch (e) {
            this._simStatus('simMvStatus', '日志读取失败: ' + e.message, true);
        }
    }

    async simDifixStatus() {
        try {
            const r = await fetch(`${API_BASE}/diffusion/status`);
            const d = await r.json();
            const ok = d.ready;
            this._simStatus('simTrustStatus', ok ? '扩散精修可用（Difix + sd-turbo 就绪）'
                : '扩散精修暂时不可用：' + JSON.stringify(d.sd_turbo && d.sd_turbo.missing || []));
            const el = document.getElementById('simDifixOut');
            if (el) el.innerHTML = `<div class="cc-status">权重：${this._escapeHtml(d.difix_ckpt || '未找到')}<br>`
                + `sd-turbo：${d.sd_turbo && d.sd_turbo.available ? '就绪' : '缺失（' + this._escapeHtml(JSON.stringify(d.sd_turbo && d.sd_turbo.missing || [])) + '）'}<br>`
                + `修复：<code>${this._escapeHtml((d.sd_turbo && d.sd_turbo.fix) || '')}</code></div>`;
        } catch (e) {
            this._simStatus('simTrustStatus', '扩散自检失败: ' + e.message, true);
        }
    }

    async simRenderRig() {
        const seg = (document.getElementById('simRigSeg')?.value || '001').trim();
        const cams = (document.getElementById('simRigCams')?.value || '0,1,2,3,4')
            .split(',').map(v => parseInt(v)).filter(v => !isNaN(v));
        const distort = !!document.getElementById('simRigDistort')?.checked;
        this._simStatus('simP23Status', '正在渲染相机 rig…');
        const d = await this._simPost('/sensor/rig', {
            scene_id: this.state.sceneId, segment: seg, frame: this.state.currentFrame || 0,
            cameras: cams, distort }, 'simP23Status', 'rig 渲染');
        if (!d) return;
        const camsOut = d.cameras || [];
        this._simStatus('simP23Status', `已渲染 ${camsOut.length} 台相机`
            + `（视角偏离：${camsOut.map(c => c.view_angle_vs_ref_deg + '°').join(' / ')}）`
            + (distort ? '，含镜头畸变' : ''));
        const el = document.getElementById('simRigOut');
        if (el) {
            el.innerHTML = '<div class="lab-row">' + camsOut.map(c =>
                `<div style="flex:1;min-width:180px;"><div class="detail">cam${c.camera} · ${c.view_angle_vs_ref_deg}°</div>`
                + `<img src="${c.image}" style="width:100%;border-radius:4px;"></div>`).join('') + '</div>';
        }
    }

    async simRollout() {
        const body = {
            scene_id: this.state.sceneId,
            steps: parseInt(document.getElementById('simSteps')?.value) || 8,
            speed: parseFloat(document.getElementById('simSpeed')?.value) || 0,
            steer: parseFloat(document.getElementById('simSteer')?.value) || 0,
            min_trust_frac: parseFloat(document.getElementById('simFrac')?.value) || undefined
        };
        this._simStatus('simP23Status', '正在跑闭环 rollout…（逐帧渲染 + trust 判定）');
        const d = await this._simPost('/sim/rollout', body, 'simP23Status', 'rollout');
        if (!d) return;
        const rep = d.report || {};
        this._simStatus('simP23Status', `rollout 完成：${rep.steps} 步，可信 ${rep.trusted_steps} 步`
            + `（比例 ${rep.trusted_ratio}），最小 coverage ${rep.min_coverage}，`
            + `${rep.collided ? '发生碰撞' : '无碰撞'}`);
        const el = document.getElementById('simRollOut');
        if (el) el.innerHTML = this._simRows(['步', '帧', 'coverage', '预测 SSIM', '阈值', '最近物体(m)', '碰撞', '终止原因'],
            (d.log || []).map(e => {
                const t = e.trust || {};
                return [e.step, e.frame, t.coverage, t.pred_ssim, t.threshold, e.near_m,
                    (e.collision || []).join(',') || '—', this._escapeHtml(e.truncate_reason || '—')];
            }));
    }

    async simExportScenario() {
        const body = {
            scene_id: this.state.sceneId,
            scenario_type: (document.getElementById('simExpType')?.value || '').trim() || null,
            seed: parseInt(document.getElementById('simExpSeed')?.value) || 7,
            num_frames: parseInt(document.getElementById('simExpFrames')?.value) || 20,
            start_frame: this.state.currentFrame || 0,
            all_tracks: !!document.getElementById('simExpAll')?.checked
        };
        this._simStatus('simExpStatus', '正在导出 OpenSCENARIO + CommonRoad…（会先生成事故再导出，导出后不改动场景）');
        const d = await this._simPost('/export/scenario', body, 'simExpStatus', '导出');
        if (!d) return;
        const v = d.validation || {};
        this._simStatus('simExpStatus', `导出完成：${d.num_tracks} 条轨迹；round-trip 校验 `
            + (v.ok ? '通过 ✓' : '失败 ✗'));
        const el = document.getElementById('simExpOut');
        if (el) {
            let html = this._simRows(['文件', '路径'], [
                ['OpenSCENARIO', this._escapeHtml((d.files || {}).openscenario || '')],
                ['CommonRoad', this._escapeHtml((d.files || {}).commonroad || '')],
                ['世界坐标 JSON', this._escapeHtml((d.files || {}).world_json || '')]
            ]);
            html += this._simRows(['track', '角色', 'ego', '速度(m/s)', '尺寸(w×l×h)', '帧数'],
                (d.tracks || []).map(t => [t.track_id, t.role, t.is_ego ? '是' : '否', t.speed_mps,
                    (t.dimensions || []).map(x => Number(x).toFixed(2)).join('×'), t.num_frames]));
            html += `<div class="cc-status">${this._escapeHtml(d.caveat || '')}</div>`;
            el.innerHTML = html;
        }
    }

    _simDemoBody() {
        return {
            scene_id: this.state.sceneId,
            segment: (document.getElementById('simDemoSeg')?.value || '001').trim(),
            scenario: (document.getElementById('simDemoScenario')?.value || 'rear-end').trim(),
            frames: parseInt(document.getElementById('simDemoFrames')?.value) || 8,
            steps: parseInt(document.getElementById('simDemoSteps')?.value) || 5,
            confirm: true
        };
    }

    async simDemoRun() {
        this._simStatus('simDemoStatus', '已提交一键自检（后台跑，约 1-3 分钟）…');
        const d = await this._simPost('/demo/run', this._simDemoBody(), 'simDemoStatus', '自检');
        if (!d) return;
        this._simDemoReport = d.report_path;
        this._simStatus('simDemoStatus', `自检已在后台启动（pid ${d.pid}）。完成后点「看报告」：${d.report_path}`);
        const el = document.getElementById('simDemoOut');
        if (el) el.innerHTML = `<div class="cc-status">日志：<code>${this._escapeHtml(d.log_path)}</code></div>`;
    }

    async simDemoReport() {
        const path = this._simDemoReport;
        if (!path) { this._simStatus('simDemoStatus', '还没有报告路径：先点「跑一键自检」，或启动后稍等再点。'); return; }
        try {
            const r = await fetch(`${API_BASE}/report?path=${encodeURIComponent(path)}`);
            const d = await r.json();
            if (!d.success) throw new Error(d.detail || '读取失败');
            const rep = d.report || {};
            const st = rep.steps || {};
            const rows = [];
            if (st.trust_model) rows.push(['trust 模型', `样本 ${st.trust_model.num_rows}，R²=${st.trust_model.model.r2}`]);
            if (st.trust_envelope) rows.push(['可信域', `max_safe_lateral=${st.trust_envelope.max_safe_lateral_m} m`]);
            if (st.actor_bank) rows.push(['actor 资产库', `${st.actor_bank.num_tracks} track / 可用 ${st.actor_bank.num_usable}`]);
            if (st.sensor_rig) rows.push(['相机 rig', `${st.sensor_rig.cameras.length} 台相机已渲染`]);
            if (st.rollout) Object.entries(st.rollout.runs || {}).forEach(([k, v]) =>
                rows.push([`rollout:${k}`, `${v.report.steps} 步，可信比例 ${v.report.trusted_ratio}`]));
            if (st.export) rows.push(['导出', `${st.export.num_tracks} 条轨迹，round-trip ${st.export.validation && st.export.validation.ok ? '通过' : '失败'}`]);
            if (st.diffusion) rows.push(['扩散精修', st.diffusion.ready ? '就绪' : `缺 ${JSON.stringify(st.diffusion.sd_turbo_missing)}`]);
            this._simStatus('simDemoStatus', '自检报告已读取 ✓');
            const el = document.getElementById('simDemoOut');
            if (el) el.innerHTML = this._simRows(['环节', '结果'], rows);
        } catch (e) {
            this._simStatus('simDemoStatus', '报告还没生成或读取失败: ' + e.message, true);
        }
    }

    async simAbRun() {
        const refine = document.getElementById('simAbRefine')?.value || 'stub';
        const held = (document.getElementById('simAbHeldout')?.value || '3,4')
            .split(',').map(v => parseInt(v)).filter(v => !isNaN(v));
        const body = { scene_id: this.state.sceneId, segment: (document.getElementById('simDemoSeg')?.value || '001').trim(),
                       num_frames: parseInt(document.getElementById('simDemoFrames')?.value) || 4,
                       heldout: held, refine, confirm: true };
        const plan = await this._simPost('/trust/ab/plan', Object.assign({}, body, { confirm: false }),
                                         'simDemoStatus', 'A/B 预检');
        if (!plan) return;
        if (refine === 'difix' && !plan.diffusion_ready) {
            this._simStatus('simDemoStatus', `Difix 未就绪，缺 ${JSON.stringify(plan.diffusion_missing)}；可先用 stub 验证管道。`, true);
            return;
        }
        const d = await this._simPost('/trust/ab/run', body, 'simDemoStatus', 'A/B');
        if (!d) return;
        this._simAbReport = d.report_path;
        this._simStatus('simDemoStatus', `A/B 已在后台启动（pid ${d.pid}，refine=${refine}）。完成后可读报告：${d.report_path}`);
        const el = document.getElementById('simDemoOut');
        if (el) el.innerHTML = `<div class="cc-status">日志：<code>${this._escapeHtml(d.log_path)}</code></div>`;
    }

    async simAbReport() {
        const path = this._simAbReport;
        if (!path) { this._simStatus('simDemoStatus', '还没有 A/B 报告：先点「跑精修 A/B」，等它跑完再点这里。'); return; }
        try {
            const r = await fetch(`${API_BASE}/report?path=${encodeURIComponent(path)}`);
            const d = await r.json();
            if (!d.success) throw new Error(d.detail || '读取失败');
            const rep = d.report || {};
            const rows = (rep.per_view || []).map(v => {
                const rf = v.refined || {};
                const f = (x, n) => (x === undefined || x === null) ? '—' : Number(x).toFixed(n);
                return [v.camera, f(v.mean_psnr, 2), f(v.mean_ssim, 4), f(v.mean_lpips, 4),
                        f(rf.mean_psnr, 2), f(rf.mean_ssim, 4), f(rf.mean_lpips, 4),
                        f(rf.delta_psnr, 3), f(rf.delta_ssim, 4), f(rf.delta_lpips, 4)];
            });
            this._simStatus('simDemoStatus', `A/B 报告（精修方式 ${rep.refine || 'none'}）：`
                + ((rep.per_view || []).some(v => v.refined) ? '含精修前后对比 ✓' : '这次没有精修结果'));
            const el = document.getElementById('simDemoOut');
            if (el) el.innerHTML = this._simRows(
                ['相机', 'PSNR', 'SSIM', 'LPIPS', '精修 PSNR', '精修 SSIM', '精修 LPIPS',
                 'ΔPSNR', 'ΔSSIM', 'ΔLPIPS'], rows);
        } catch (e) {
            this._simStatus('simDemoStatus', 'A/B 报告还没生成或读取失败: ' + e.message, true);
        }
    }

    // ==================== ⑥ CARLA 仿真（视频 / 实时直播 / 标准导出） ====================
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
            road_snap: this._carlaVal('carlaRoadSnap') || 'z',
        };
        const _rn = (this._carlaVal('carlaRunName') || '').trim();
        if (_rn) out.export_name = _rn;
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
            const v = document.getElementById('carlaVram');
            if (v && d.gpu) {
                const g = d.gpu;
                v.textContent = `显存 ${(g.used_mb / 1024).toFixed(1)}/${(g.total_mb / 1024).toFixed(1)} GB`;
                v.style.color = g.free_mb < 7000 ? '#ef4444' : '#9ca3af';
                v.title = '空闲 ' + g.free_mb + ' MB\n' +
                    (g.procs || []).map(p => `${p.name}  ${p.used_mb} MB`).join('\n');
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
            this._carlaStatus(ok ? (action === 'start' ? 'CARLA 已就绪 ✓' : '已停止') :
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
                `CARLA 安装：${d.carla_installed ? '✓' : '✗'}  ${d.carla_home}`,
                `Python3.7 API：${d.python37_ok ? '✓' : '✗'}  ${d.python37}`,
                `脚本：${Object.entries(d.scripts || {}).map(([k, v]) => k + (v ? '✓' : '✗')).join('  ')}`,
                `ScenarioRunner：${d.scenario_runner ? '✓' : '✗'}    磁盘剩余 ${d.disk_free_gb} GB`,
                `Vulkan：${(d.vulkan || []).join(' | ') || '未检测到'}`,
            ];
            const box = document.getElementById('carlaMetrics');
            if (box) box.innerHTML = `<table class="lab-table"><tbody>${lines.map(l => `<tr><td>${l}</td></tr>`).join('')}</tbody></table>`;
            this._carlaStatus(ok ? '环境自检通过 ✓' : '环境不完整（见下表）', !ok);
        } catch (e) { this._carlaStatus('自检失败: ' + e.message, true); }
    }

    async carlaLoadScenarios() {
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
            this._carlaStatus(ok ? `作业完成 ✓（${j.seconds}s）` : `作业 ${j.status} ✗ rc=${j.rc}`, !ok);
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
            vids.forEach(v => items.push(`<a class="btn btn-secondary btn-small" href="${this._carlaDownloadUrl(v)}">下载 ${v.split('/').pop()}</a>`));
            ['events_json', 'frames_json', 'xosc'].forEach(k => {
                if ((j.artifacts || {})[k]) {
                    items.push(`<a class="btn btn-secondary btn-small" href="${this._carlaDownloadUrl(j.artifacts[k])}">下载 ${j.artifacts[k].split('/').pop()}</a>`);
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

    // ==================== CARLA 仿真引擎（可播放 + 可编辑）====================
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
        this._engStatus('引擎已停止（显存约 1 分钟后自动释放，也可点「释放显存」）');
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
        if (pb) pb.textContent = st.play ? '暂停' : '播放';
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
                <td>${a.is_ego ? '主车 ' : ''}${this._escapeHtml(String(a.role))}#${a.tid}${a.extra ? ' <b>(新增)</b>' : ''}</td>
                <td>${a.kind}</td><td>${loc}</td><td>${a.yaw ?? '-'}°</td><td>${off}</td>
                <td>
                  <button class="btn btn-secondary btn-small eng-nudge" data-tid="${a.tid}" data-dx="1">左</button>
                  <button class="btn btn-secondary btn-small eng-nudge" data-tid="${a.tid}" data-dx="-1">右</button>
                  <button class="btn btn-secondary btn-small eng-nudge" data-tid="${a.tid}" data-dy="1">前</button>
                  <button class="btn btn-secondary btn-small eng-nudge" data-tid="${a.tid}" data-dy="-1">后</button>
                  <button class="btn btn-secondary btn-small eng-nudge" data-tid="${a.tid}" data-dyaw="15">逆</button>
                  <button class="btn btn-secondary btn-small eng-nudge" data-tid="${a.tid}" data-dyaw="-15">顺</button>
                  <button class="btn btn-secondary btn-small eng-del" data-tid="${a.tid}">删</button>
                </td></tr>`;
        }).join('');
        box.innerHTML = `<table class="lab-table"><thead><tr><th>参与者</th><th>类型</th><th>位置(x,y)</th>`
            + `<th>朝向</th><th>编辑量</th><th>调整（步行=左/右/前/后，转角=逆/顺）</th></tr></thead>`
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
                + '可以把它填到上面「1) 选场景」的输入里，或者直接「开始 CARLA 仿真并出视频」。';
            this._engStatus('已保存 ✓（编辑过的轨迹都在里面）');
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
            this._simStatus('simDifixOut', `第 ${d.frame_idx} 帧精修完成，用时 ${d.seconds}s`);
        } catch (e) { this._simStatus('simDifixOut', '精修失败: ' + e.message, true); }
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

    // 扩散精修：一次精修所有帧 -> 出一段视频
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
                    const tot = j.total || 0;
                    this._simStatus('simDifixOut', tot
                        ? `精修中 ${j.done}/${tot} 帧…（首帧要加载模型较慢，之后每帧约几秒~几十秒）`
                        : '正在准备第 1 帧（加载模型）…');
                    return;
                }
                clearInterval(this._difixTimer); this._difixTimer = null;
                const box = document.getElementById('simDifixImages');
                if (j.status === 'done') {
                    this._simStatus('simDifixOut', `全部 ${j.total} 帧精修完成，产物目录 ${j.out_dir}`);
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

    async carlaRelease() {
        this._carlaStatus('正在释放显存（停 CARLA + 直播）…');
        try {
            const r = await fetch(`${API_BASE}/carla/release`, { method: 'POST' });
            const d = await r.json();
            const img = document.getElementById('carlaLiveImg');
            if (img) { img.removeAttribute('src'); img.alt = '直播已停止'; }
            await this.carlaRefreshStatus(1);
            const g = d.gpu || {};
            this._carlaStatus(`已释放，现在空闲显存 ${g.free_mb ?? '?'} MB / ${g.total_mb ?? '?'} MB`
                + '（可以放心去用 SAM3D 了）');
        } catch (e) { this._carlaStatus('释放失败: ' + e.message, true); }
    }

    async carlaCancelJob() {
        const s = this._carlaState();
        if (!s.jobId) { this._carlaStatus('当前没有跟踪中的作业'); return; }
        try {
            await fetch(`${API_BASE}/carla/jobs/${s.jobId}/cancel`, { method: 'POST' });
            this._carlaStatus('已请求取消作业 ' + s.jobId);
        } catch (e) { this._carlaStatus('取消失败: ' + e.message, true); }
    }

    // ==================== 工作台流程串联（① 挑事故 → ② 批量生成 → ③ 闭环评估 → ④ CARLA 验证 → ⑤/⑥）====================
    _labSteps() { return ['batch', 'graph', 'sim', 'carla']; }
    _labStepNames() {
        return { batch: '① 批量生成', graph: '② 挑事故', sim: '③ 渲染可信域评估',
                 carla: '④ CARLA 闭环仿真' };
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
            : '还没有案例 —— 先在 ① 批量生成里跑一批，再点某一行右侧「载入场景」或「CARLA 这条」';
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

    // 把某条批量实例"载入场景"（确定性重放），之后可以在 ③ 里逐帧评估、继续编辑
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
        this._carlaPendingScenario = m.world_json || null;   // 让场景列表刷新后仍然选中它
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
        this._carlaStatus(`已选中批量实例 ${m.instance_id}：点「开始 CARLA 仿真并出视频」即可`
            + (m.world_json ? '（直接吃实例的 world.json，不依赖内存里的场景）' : '（会现场导出当前场景）'));
    }

    // ③ 闭环评估 → ④ CARLA 验证
    _labGoCarla() {
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
    }

    _labSwitchTab(name) {
        document.querySelectorAll('.lab-tab').forEach(t => t.classList.toggle('active', t.dataset.tab === name));
        document.querySelectorAll('.lab-panel').forEach(p => p.classList.toggle('active', p.dataset.panel === name));
        if (name === 'graph' && this.state.sceneId) this.labLoadGraph();
        if (name === 'carla') { this.carlaRefreshStatus(1); this.carlaLoadScenarios(); this.carlaLoadJobs(); this._carlaWillRun(); }
        // ④ 页面里每 10 秒刷一次状态（顺带盯显存），切走就停
        if (this._carlaTimer) { clearInterval(this._carlaTimer); this._carlaTimer = null; }
        if (name === 'carla') this._carlaTimer = setInterval(() => this.carlaRefreshStatus(0), 10000);
        this._labRenderFlow();
    }

    _labStatus(id, msg, isError = false) {
        const el = document.getElementById(id);
        if (el) {
            el.textContent = msg;
            el.style.color = isError ? '#e5484d' : 'inherit';
        }
    }

    // 帧切换时（实验台打开 + 关系图页 + 勾选自动刷新）防抖刷新
    _labMaybeAutoRefresh(frameIdx) {
        const modal = document.getElementById('labModal');
        if (!modal || !modal.classList.contains('active')) return;
        const graphTabActive = document.querySelector('.lab-tab.active')?.dataset.tab === 'graph';
        const autoEl = document.getElementById('labGraphAuto');
        if (!graphTabActive || !autoEl || !autoEl.checked || !this.state.sceneId) return;
        const frameEl = document.getElementById('labGraphFrame');
        if (frameEl) frameEl.value = frameIdx;
        if (this._labTimer) clearTimeout(this._labTimer);
        this._labTimer = setTimeout(() => this.labLoadGraph(frameIdx), 150);
    }

    async labLoadGraph(frameIdx) {
        if (!this.state.sceneId) { this._labStatus('labGraphStatus', '请先加载场景。', true); return; }
        const f = (frameIdx !== undefined && frameIdx !== null)
            ? frameIdx
            : (parseInt(document.getElementById('labGraphFrame')?.value) || 0);
        const win = parseInt(document.getElementById('labGraphWindow')?.value) || 10;
        this._labStatus('labGraphStatus', `正在读取第 ${f} 帧关系图…`);
        try {
            const resp = await fetch(`${API_BASE}/corner_case/graph/${encodeURIComponent(this.state.sceneId)}?frame_idx=${f}&window=${win}`);
            const data = await resp.json();
            if (!data.success) throw new Error(data.detail || '读取失败');
            const g = data.graph || {};
            this._labGraph = g;
            this._labProposals = data.proposals || {};
            this._labRenderGraphSummary(g, data.proposals || {}, f);
            this._labRenderGraphSvg(g);
            this._labRenderGraphEdges(g);
            this._labRenderProposals(data.proposals || {});
            this._labStatus('labGraphStatus', `第 ${f} 帧：${g.num_nodes || 0} 个物体 / ${g.num_edges || 0} 条关系。`);
        } catch (e) {
            this._labStatus('labGraphStatus', '读取关系图失败: ' + e.message, true);
        }
    }

    _labRenderGraphSummary(g, proposals, frameIdx) {
        const el = document.getElementById('labGraphSummary');
        if (!el) return;
        const nodes = g.nodes || [];
        const edges = g.edges || [];
        const crit = edges.filter(e => (e.criticality || 0) > 0.6).length;
        const onCourse = edges.filter(e => e.on_collision_course).length;
        const classes = {};
        nodes.forEach(n => { classes[n.class] = (classes[n.class] || 0) + 1; });
        let html = `<span>帧 <b>${frameIdx}</b></span>`;
        html += `<span>物体 <b>${nodes.length}</b>（车辆 ${classes.vehicle || 0} / 行人 ${classes.pedestrian || 0} / 大车 ${classes.large_vehicle || 0}）</span>`;
        html += `<span>关系 <b>${edges.length}</b></span>`;
        html += `<span>高危关系(>0.6) <b class="${crit ? 'bad' : ''}">${crit}</b></span>`;
        html += `<span>碰撞航向 <b class="${onCourse ? 'bad' : ''}">${onCourse}</b></span>`;
        const nProp = Object.values(proposals).reduce((a, c) => a + (c ? c.length : 0), 0);
        html += `<span>参与者提案 <b>${nProp}</b></span>`;
        el.innerHTML = html;
    }

    _labRenderGraphSvg(g) {
        const svg = document.getElementById('labGraphSvg');
        if (!svg) return;
        const nodes = g.nodes || [];
        if (!nodes.length) { svg.innerHTML = '<text x="20" y="30" fill="#6b7686" font-size="12">该帧附近没有动态物体</text>'; return; }
        const xs = nodes.map(n => n.center[0]), zs = nodes.map(n => n.center[2]);
        const minX = Math.min(...xs), maxX = Math.max(...xs), minZ = Math.min(...zs), maxZ = Math.max(...zs);
        const W = 520, H = 380, pad = 40;
        const sx = x => pad + (maxX - minX < 1e-6 ? (W - 2 * pad) / 2 : (x - minX) / (maxX - minX) * (W - 2 * pad));
        const sz = z => H - pad - (maxZ - minZ < 1e-6 ? (H - 2 * pad) / 2 : (z - minZ) / (maxZ - minZ) * (H - 2 * pad));
        const parts = [];
        // 边
        (g.edges || []).forEach(e => {
            const a = nodes.find(n => n.track_id === e.track_pair[0]);
            const b = nodes.find(n => n.track_id === e.track_pair[1]);
            if (!a || !b) return;
            const c = e.criticality || 0;
            const stroke = c > 0.6 ? '#ef4444' : (c > 0.3 ? '#f59e0b' : '#3b82f6');
            parts.push(`<line x1="${sx(a.center[0]).toFixed(1)}" y1="${sz(a.center[2]).toFixed(1)}" x2="${sx(b.center[0]).toFixed(1)}" y2="${sz(b.center[2]).toFixed(1)}" stroke="${stroke}" stroke-width="${(1 + c * 3).toFixed(1)}" opacity="0.6"/>`);
        });
        // 节点 + 速度箭头
        nodes.forEach(n => {
            const cx = sx(n.center[0]), cy = sz(n.center[2]);
            const fill = n.class === 'pedestrian' ? '#22c55e' : (n.class === 'large_vehicle' ? '#a855f7' : '#38bdf8');
            parts.push(`<circle cx="${cx.toFixed(1)}" cy="${cy.toFixed(1)}" r="9" fill="${fill}" stroke="#0b1220" stroke-width="1.5"/>`);
            parts.push(`<text x="${cx.toFixed(1)}" y="${(cy - 13).toFixed(1)}" fill="#e5e7eb" font-size="11" text-anchor="middle">T${n.track_id}</text>`);
            const v = n.velocity || [0, 0, 0];
            const vn = Math.hypot(v[0], v[2]);
            if (vn > 0.5) {
                const scale = 26 / Math.max(1, vn);
                const ex = cx + v[0] * scale;
                const ey = cy - v[2] * scale;   // z 增大 = 屏幕向上
                parts.push(`<line x1="${cx.toFixed(1)}" y1="${cy.toFixed(1)}" x2="${ex.toFixed(1)}" y2="${ey.toFixed(1)}" stroke="${fill}" stroke-width="2" marker-end="url(#labArrow)"/>`);
            }
        });
        parts.unshift('<defs><marker id="labArrow" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path d="M0,0 L7,3.5 L0,7 z" fill="#94a3b8"/></marker></defs>');
        svg.innerHTML = parts.join('');
    }

    _labRenderGraphEdges(g) {
        const box = document.getElementById('labGraphEdges');
        if (!box) return;
        const relLabel = {
            following: '同向跟车', adjacent_parallel: '并列同向', oncoming: '对向',
            crossing: '交叉', stationary: '含静止', receding: '远离', unrelated: '无关'
        };
        const rows = (g.edges || []).slice(0, 40).map(e => {
            const c = e.criticality || 0;
            const cls = c > 0.6 ? 'crit-high' : (c > 0.3 ? 'crit-mid' : '');
            const ttc = e.ttc_s === null || e.ttc_s === undefined ? '—' : e.ttc_s.toFixed(2);
            return `<tr>
                <td>${e.track_pair[0]}–${e.track_pair[1]}</td>
                <td>${relLabel[e.relation] || e.relation}</td>
                <td>${e.distance.toFixed(1)}</td>
                <td>${e.relative_speed.toFixed(1)}</td>
                <td>${e.approach_rate.toFixed(1)}</td>
                <td>${ttc}</td>
                <td class="${cls}">${c.toFixed(2)}</td>
                <td>${e.on_collision_course ? '是' : ''}</td>
            </tr>`;
        }).join('');
        box.innerHTML = `<table class="lab-table">
            <thead><tr><th>对</th><th>关系</th><th>距离</th><th>相对速度</th><th>接近速率</th><th>TTC</th><th>关键度</th><th>碰撞航向</th></tr></thead>
            <tbody>${rows || '<tr><td colspan="8">无关系边</td></tr>'}</tbody>
        </table>`;
    }

    _labRenderProposals(proposals) {
        const box = document.getElementById('labGraphProposals');
        if (!box) return;
        const nameMap = {};
        LAB_SCENARIOS.forEach(([v, label]) => { nameMap[v] = label; });
        let html = '';
        Object.entries(proposals).forEach(([st, cands]) => {
            if (!cands || !cands.length) {
                html += `<div class="lab-proposal-row"><span class="p-type">${nameMap[st] || st}</span><span class="detail">无可候选组合</span></div>`;
                return;
            }
            cands.slice(0, 3).forEach((roles, i) => {
                const chips = Object.entries(roles).map(([k, v]) => `<span class="lab-chip">${k}=T${v}</span>`).join('');
                const rolesJson = JSON.stringify(roles);
                html += `<div class="lab-proposal-row">
                    <span class="p-type">${nameMap[st] || st} #${i + 1}</span>
                    ${chips}
                    <button class="btn btn-small btn-secondary lab-use-proposal" data-type="${st}" data-roles='${this._escapeHtml(rolesJson)}'>用此组合</button>
                </div>`;
            });
        });
        box.innerHTML = html || '<div class="empty-state"><p>暂无提案</p></div>';
        box.querySelectorAll('.lab-use-proposal').forEach(btn => {
            btn.addEventListener('click', () => {
                let roles = {};
                try { roles = JSON.parse(btn.dataset.roles); } catch (e) { return; }
                this.labUseProposal(btn.dataset.type, roles);
            });
        });
    }

    labUseProposal(scenarioType, roles) {
        const sel = document.getElementById('cornerScenarioType');
        if (!sel) return;
        sel.value = scenarioType;
        this.renderCornerRoles();          // 会重置角色指派
        Object.entries(roles).forEach(([k, v]) => this.assignCornerRole(k, Number(v)));
        this.closeLab();
        this._switchToCornerPanel();
        const st = this.cornerScenarios[scenarioType];
        this.updateStatus(`已按关系图提案填入「${st ? st.name : scenarioType}」的参与者，可直接点「生成事故轨迹」`);
    }

    _switchToCornerPanel() {
        // 展开折叠的场景参数并滚动到生成面板
        const panel = document.getElementById('cornerCasePanel');
        if (panel) panel.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }

    _labSelectedTypes() {
        return Array.from(document.querySelectorAll('.lab-batch-type:checked')).map(c => c.value);
    }

    async labRunBatch() {
        if (!this.state.sceneId) { this._labStatus('labBatchStatus', '请先加载场景。', true); return; }
        const types = this._labSelectedTypes();
        if (!types.length) { this._labStatus('labBatchStatus', '请至少勾选一个事故类型。', true); return; }
        const body = {
            scene_id: this.state.sceneId,
            scenario_types: types,
            num_per_type: parseInt(document.getElementById('labBatchNum')?.value) || 3,
            seed: parseInt(document.getElementById('labBatchSeed')?.value) || 0,
            start_frame: parseInt(document.getElementById('labBatchStart')?.value) || 0,
            num_frames: parseInt(document.getElementById('labBatchFrames')?.value) || 20,
            save: !!document.getElementById('labBatchSave')?.checked,
            keep_only_valid: !!document.getElementById('labBatchOnlyValid')?.checked,
            // 注意：元素缺失时用 ?? 兜底（旧缓存页面里没有这些控件）——
            // 否则 render_videos 会变成 false，导致"视频全是无"。
            render_videos: document.getElementById('labBatchRenderVideo')?.checked ?? true,
            // 真实渲染视频不叠加包围盒/ID/轨迹（用户要求干净画面）
            video_draw_bboxes: false,
            video_draw_ids: false,
            video_draw_trajectories: false,
            render_bev: document.getElementById('labBatchVidBev')?.checked ?? true,
            render_topdown: document.getElementById('labBatchVidTop')?.checked ?? true,
            use_edited_ego: document.getElementById('labBatchFollowEgo')?.checked ?? false
        };
        const runName = document.getElementById('labBatchRunName')?.value?.trim();
        if (runName) body.run_name = runName;
        this._labStatus('labBatchStatus', '批量生产中…（每条：图选参与者 → 采样参数 → 生成 → 质量打分' + (body.render_videos ? ' → 渲染视频' : '') + ' → 还原）');
        const t0 = performance.now();
        try {
            const resp = await fetch(`${API_BASE}/corner_case/batch`, {
                method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body)
            });
            const data = await resp.json();
            if (!data.success) throw new Error(data.detail || '批量失败');
            const dt = ((performance.now() - t0) / 1000).toFixed(1);
            this._labStatus('labBatchStatus', `完成：${data.total} 条，通过质量门 ${data.num_valid} 条`
                + (data.num_videos ? `，主车视角视频 ${data.num_videos} 个` : '')
                + (data.num_bev_videos ? `，俯视BEV ${data.num_bev_videos} 个` : '')
                + (data.num_topdown_videos ? `，俯视3D渲染 ${data.num_topdown_videos} 个` : '') + `（耗时 ${dt}s）。`);
            this._lastBatchManifest = data.manifest || [];
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
            this._labRenderBatchTable(data);
        } catch (e) {
            this._labStatus('labBatchStatus', '批量生成失败: ' + e.message, true);
        }
    }

    _labRenderBatchSummary(data) {
        const el = document.getElementById('labBatchSummary');
        if (!el) return;
        let html = `<span>实例 <b>${data.total}</b></span>`;
        html += `<span>通过质量门 <b class="${data.num_valid ? 'ok' : 'bad'}">${data.num_valid}</b></span>`;
        if (data.num_saved !== undefined) html += `<span>已落盘 <b>${data.num_saved}</b></span>`;
        if (data.num_videos !== undefined) html += `<span>主车视角视频 <b class="${data.num_videos ? 'ok' : ''}">${data.num_videos}</b></span>`;
        if (data.num_bev_videos !== undefined) html += `<span>俯视BEV <b class="${data.num_bev_videos ? 'ok' : ''}">${data.num_bev_videos}</b></span>`;
        if (data.num_topdown_videos !== undefined) html += `<span>俯视3D渲染 <b class="${data.num_topdown_videos ? 'ok' : ''}">${data.num_topdown_videos}</b></span>`;
        const bt = data.by_type || {};
        Object.entries(bt).forEach(([st, info]) => {
            html += `<span>${st}: 候选 ${info.num_candidates ?? 0} / 生成 ${info.num_generated ?? 0} / 通过 ${info.num_valid ?? 0}</span>`;
        });
        if (data.reset_scene) {
            html += `<span style="color:#f59e0b">已自动清空批量前的生成/编辑残留（不清的话不同类型会生成出一样的视频）</span>`;
        }
        if (data.video_dir) html += `<span>视频目录: <b>${this._escapeHtml(data.video_dir)}</b></span>`;
        if (data.manifest_path) html += `<span>manifest: <b>${this._escapeHtml(data.manifest_path)}</b></span>`;
        el.innerHTML = html;
    }

    _labRenderBatchTable(data) {
        const box = document.getElementById('labBatchTable');
        if (!box) return;
        const rows = (data.manifest || []).map(m => {
            const q = m.quality || {};
            const blk = (q.blocking_checks || []).map(n => this._qualityCheckLabel(n)).join('、');
            const ok = m.valid ? '<span class="pass">通过</span>' : '<span class="fail">未通过</span>';
            const ego = m.video_path || null;
            const bev = m.bev_video_path || null;
            const top = m.topdown_video_path || null;
            const lbl = this._escapeHtml(m.instance_id || '');
            let vid = '<span class="video-btn disabled">无</span>';
            if (ego || bev || top) {
                const attrs = `data-path="${ego ? this._escapeHtml(ego) : ''}" data-bev="${bev ? this._escapeHtml(bev) : ''}" data-top="${top ? this._escapeHtml(top) : ''}" data-label="${lbl}"`;
                vid = '';
                if (ego) vid += `<button class="video-btn lab-play-video" data-kind="ego" ${attrs}>主车</button>`;
                if (bev) vid += `<button class="video-btn video-btn-bev lab-play-video" data-kind="bev" ${attrs}>俯视BEV</button>`;
                if (top) vid += `<button class="video-btn video-btn-top lab-play-video" data-kind="top" ${attrs}>俯视3D</button>`;
            }
            return `<tr>
                <td>${this._escapeHtml(m.instance_id || '')}</td>
                <td>${this._escapeHtml(m.scenario_type || '')}</td>
                <td>${this._escapeHtml(JSON.stringify(m.roles || {}))}</td>
                <td>${m.collision_frame ?? '—'}</td>
                <td>${m.critical_frame ?? '—'}</td>
                <td>${q.ttc_at_critical === null || q.ttc_at_critical === undefined ? '—' : Number(q.ttc_at_critical).toFixed(2)}</td>
                <td>${q.max_accel === null || q.max_accel === undefined ? '—' : Number(q.max_accel).toFixed(1)}</td>
                <td>${ok}</td>
                <td>${this._escapeHtml(blk)}</td>
                <td>${vid}</td>
                <td><button class="btn btn-secondary btn-small lab-batch-next" data-i="${this._escapeHtml(m.instance_id || '')}" data-j="apply" ${m.world_json ? '' : 'disabled'}>载入场景</button>
                    <button class="btn btn-primary btn-small lab-batch-next" data-i="${this._escapeHtml(m.instance_id || '')}" data-j="carla" ${m.world_json ? '' : 'disabled'}>CARLA 这条</button></td>
            </tr>`;
        }).join('');
        box.innerHTML = `<table class="lab-table">
            <thead><tr><th>实例</th><th>类型</th><th>参与者</th><th>碰撞帧</th><th>反应帧</th><th>TTC</th><th>加速度</th><th>质量门</th><th>未通过项</th><th>视频</th><th>下一步</th></tr></thead>
            <tbody>${rows || '<tr><td colspan="11">无实例</td></tr>'}</tbody>
        </table>`;
        box.querySelectorAll('.lab-play-video').forEach(btn => {
            btn.addEventListener('click', () => this._labShowVideo(
                { ego: btn.dataset.path || null, bev: btn.dataset.bev || null, top: btn.dataset.top || null },
                btn.dataset.label, btn.dataset.kind || 'ego'));
        });
        // 每一行 = 一条实例：可以"载入场景继续编辑/评估"，也可以"直接送 CARLA"
        box.querySelectorAll('.lab-batch-next').forEach(btn => {
            btn.addEventListener('click', () => {
                const man = this._lastBatchManifest || [];
                const m = man.find(x => (x.instance_id || '') === btn.dataset.i) || {};
                if (btn.dataset.j === 'carla') this._labSendInstanceToCarla(m);
                else this._labApplyInstance(m);
            });
        });
        box.scrollTop = 0;
    }

    async labCollectGnn() {
        if (!this.state.sceneId) { this._labStatus('labGnnStatus', '请先加载场景。', true); return; }
        const types = this._labSelectedTypes();
        const scenarioTypes = types.length ? types : ['rear-end', 'head-on', 'lane-change-cutin'];
        const body = {
            scene_id: this.state.sceneId,
            scenario_types: scenarioTypes,
            num_per_type: parseInt(document.getElementById('labGnnNum')?.value) || 20,
            seed: parseInt(document.getElementById('labGnnSeed')?.value) || 0,
            start_frame: 0,
            num_frames: 20
        };
        this._labStatus('labGnnStatus', '正在收集样本…');
        try {
            const resp = await fetch(`${API_BASE}/gnn/collect`, {
                method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body)
            });
            const data = await resp.json();
            if (!data.success) throw new Error(data.detail || '收集失败');
            this._labStatus('labGnnStatus', `完成：${data.num_samples} 个样本（特征维 ${data.feature_dim}）。`);
            const el = document.getElementById('labGnnSummary');
            if (el) {
                const dist = Object.entries(data.severity_dist || {}).map(([k, v]) => `${k}:${v}`).join(' ');
                el.innerHTML = `<span>样本 <b>${data.num_samples}</b></span>`
                    + `<span>特征维 <b>${data.feature_dim}</b></span>`
                    + `<span>碰撞率 <b>${(data.collision_rate * 100).toFixed(1)}%</b></span>`
                    + `<span>严重度分布 <b>${this._escapeHtml(dist || '—')}</b></span>`
                    + `<span>文件 <b>${this._escapeHtml(data.output_path || '')}</b></span>`;
            }
            this._labGnnCommand(data.output_path);
        } catch (e) {
            this._labStatus('labGnnStatus', '收集失败: ' + e.message, true);
        }
    }

    _labGnnCommand(dataPath) {
        const el = document.getElementById('labGnnCmd');
        if (!el) return;
        const p = dataPath || '<收集后生成的 .npz 路径>';
        el.textContent = [
            'cd /root/autodl-fs/dggt-main/studio/backend',
            '/root/autodl-tmp/conda_envs/dggt/bin/python train_gnn.py \\',
            `  --data ${p} \\`,
            '  --epochs 300 --out gnn_model.pt'
        ].join('\n');
    }

    // ---------- 视频回放 ----------
    _videoUrl(path) {
        return `${API_BASE}/corner_case/video_file?path=${encodeURIComponent(path)}`;
    }

    // videos: {ego, bev}；kind: 'ego' | 'bev'
    _labShowVideo(videos, label, kind) {
        this._labVideos = Object.assign({ ego: null, bev: null, top: null }, videos || {});
        this._labVideoLabel = label || '';
        const modal = document.getElementById('labVideoModal');
        const tabs = { ego: document.getElementById('labVideoTabEgo'),
                       bev: document.getElementById('labVideoTabBev'),
                       top: document.getElementById('labVideoTabTop') };
        Object.entries(tabs).forEach(([k, el]) => { if (el) el.style.display = this._labVideos[k] ? '' : 'none'; });
        if (!this._labVideos.ego && !this._labVideos.bev && !this._labVideos.top) return;
        this._labSwitchVideo(kind || (this._labVideos.ego ? 'ego' : (this._labVideos.bev ? 'bev' : 'top')));
        if (modal) modal.classList.add('active');
    }

    _labSwitchVideo(kind) {
        const path = this._labVideos ? this._labVideos[kind] : null;
        if (!path) return;
        this._labVideoKind = kind;
        const player = document.getElementById('labVideoPlayer');
        const info = document.getElementById('labVideoInfo');
        const link = document.getElementById('labVideoOpenLink');
        const pathEl = document.getElementById('labVideoPath');
        const url = this._videoUrl(path);
        if (player) { player.src = url; player.load(); player.play().catch(() => {}); }
        const views = { ego: '主车视角', bev: '俯视(BEV)示意图', top: '俯视(3D渲染)' };
        if (info) info.textContent = (this._labVideoLabel ? `${this._labVideoLabel} · ` : '') + (views[kind] || '');
        if (link) link.href = url;
        if (pathEl) pathEl.textContent = path;
        ['ego', 'bev', 'top'].forEach(k => {
            const el = document.getElementById('labVideoTab' + k.charAt(0).toUpperCase() + k.slice(1));
            if (el) el.className = 'btn btn-small ' + (k === kind ? 'btn-primary' : 'btn-secondary');
        });
    }

    _labPlayVideo(path, label) {
        if (!path) return;
        this._labShowVideo({ ego: path }, label, 'ego');
    }

    closeVideoModal() {
        const modal = document.getElementById('labVideoModal');
        const player = document.getElementById('labVideoPlayer');
        if (player) { player.pause(); player.removeAttribute('src'); player.load(); }
        if (modal) modal.classList.remove('active');
    }

    // 把"当前场景状态"（已生成/编辑的事故轨迹）录制成可播放视频
    async labRenderCaseVideo() {
        if (!this.state.sceneId) { alert('请先加载场景'); return; }
        const start = parseInt(document.getElementById('cornerStartFrame')?.value) || 0;
        const num = parseInt(document.getElementById('cornerNumFrames')?.value) || 20;
        const btn = document.getElementById('renderCaseVideoBtn');
        const oldText = btn ? btn.textContent : '';
        if (btn) { btn.disabled = true; btn.textContent = '录制中…（逐帧渲染）'; }
        this.updateStatus('正在渲染案例视频（与播放效果一致）…');
        try {
            const resp = await fetch(`${API_BASE}/corner_case/video`, {
                method: 'POST', headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    scene_id: this.state.sceneId, start_frame: start, num_frames: num, fps: 10.0,
                    draw_bboxes: false, draw_ids: false, draw_trajectories: false,
                    render_bev: true, render_topdown: true,
                    // 始终跟随"当前主车视角"（真实主车轨迹 / 全局选定的物体）
                    use_edited_ego: true,
                    name: `case_${Date.now()}`
                })
            });
            const data = await resp.json();
            if (!data.success) throw new Error(data.detail || '渲染失败');
            this.updateStatus(`案例视频已生成：${data.video_path}`);
            this._labShowVideo(
                { ego: data.video_path || null, bev: data.bev_video_path || null,
                  top: data.topdown_video_path || null },
                `当前案例（帧 ${data.start_frame}~${data.start_frame + data.num_frames - 1}）`, 'ego');
        } catch (e) {
            alert('渲染视频失败: ' + e.message);
        } finally {
            if (btn) { btn.disabled = false; btn.textContent = oldText; }
        }
    }

    // ---------- 一键自检 ----------
    _labCheck(icon, cls, title, detail) {
        return `<div class="lab-check ${cls}"><span class="icon">${icon}</span><div><div>${this._escapeHtml(title)}</div><div class="detail">${this._escapeHtml(detail || '')}</div></div></div>`;
    }

    async labSelfTest() {
        const out = document.getElementById('labSelfTestOut');
        if (!out) return;
        if (!this.state.sceneId) { out.innerHTML = this._labCheck('✗', 'fail', '未加载场景', '请先加载一个场景'); return; }
        out.innerHTML = this._labCheck('…', 'warn', '开始自检…', '');
        const logs = [];
        const push = (icon, cls, title, detail) => {
            logs.push(this._labCheck(icon, cls, title, detail));
            out.innerHTML = logs.join('');
            out.scrollTop = out.scrollHeight;
        };

        // ① 关系图
        let graph = null;
        try {
            const r = await fetch(`${API_BASE}/corner_case/graph/${encodeURIComponent(this.state.sceneId)}?frame_idx=0&window=10`);
            const d = await r.json();
            graph = d.graph;
            const ok = d.success && graph && graph.num_nodes > 0;
            push(ok ? '✓' : '✗', ok ? 'pass' : 'fail', '① 关系图（scene graph）',
                `节点 ${graph?.num_nodes ?? 0}，关系 ${graph?.num_edges ?? 0}`);
        } catch (e) { push('✗', 'fail', '① 关系图', e.message); }

        // 选一个探测物体（优先有速度的车辆）
        let probe = null;
        if (graph && graph.nodes) {
            const veh = graph.nodes.filter(n => n.class !== 'pedestrian' && n.speed > 1.0);
            probe = (veh[0] || graph.nodes[0] || {}).track_id;
        }

        // ② 统一引擎 + 质量门（干净合成参与者：只给一个角色，另一个自动合成）
        if (probe !== null && probe !== undefined) {
            for (const [st, roleKey] of [['rear-end', 'attacker'], ['head-on', 'attacker'], ['intersection-tbone', 'attacker']]) {
                try {
                    const resp = await fetch(`${API_BASE}/corner_case/generate`, {
                        method: 'POST', headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                            scene_id: this.state.sceneId, scenario_type: st, roles: { [roleKey]: probe },
                            start_frame: 0, num_frames: 30, intensity: 1.0, enable_physics: true, fps: 10.0, sampling_seed: 123
                        })
                    });
                    const d = await resp.json();
                    if (!d.success) { push('✗', 'fail', `② 生成 ${st}`, d.detail || '生成失败'); continue; }
                    const q = d.quality_report || {};
                    const ca = d.collision_analysis || {};
                    const blocking = (q.checks || []).filter(c => c.passed === false).map(c => c.name);
                    const ok = q.valid === true;
                    push(ok ? '✓' : '!', ok ? 'pass' : 'warn', `② 统一引擎生成 · ${st}`,
                        `碰撞帧 ${d.collision_frame ?? '无'}，反应帧 ${d.critical_frame ?? '无'}，TTC ${ca.time_to_collision ?? '—'}s，`
                        + `最大加速度 ${(q.max_accel ?? 0).toFixed(1)} m/s²，质量门 ${ok ? '通过' : '未通过(' + blocking.join(',') + ')'}`);
                    await fetch(`${API_BASE}/undo/${encodeURIComponent(this.state.sceneId)}`, { method: 'POST' });
                } catch (e) { push('✗', 'fail', `② 生成 ${st}`, e.message); }
            }
        } else {
            push('!', 'warn', '② 统一引擎生成', '未找到可用的探测物体，跳过');
        }

        // ③ 批量生成
        try {
            const resp = await fetch(`${API_BASE}/corner_case/batch`, {
                method: 'POST', headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    scene_id: this.state.sceneId, scenario_types: ['rear-end', 'head-on'],
                    num_per_type: 2, seed: 7, start_frame: 0, num_frames: 20, save: false
                })
            });
            const d = await resp.json();
            const ok = d.success && d.total > 0;
            push(ok ? '✓' : '✗', ok ? 'pass' : 'fail', '③ 批量生成 + 质量评分',
                `实例 ${d.total ?? 0}，通过 ${d.num_valid ?? 0}；多样性由"图选参与者 + 参数采样"提供`);
        } catch (e) { push('✗', 'fail', '③ 批量生成', e.message); }

        // ④ GNN 语料
        try {
            const resp = await fetch(`${API_BASE}/gnn/collect`, {
                method: 'POST', headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    scene_id: this.state.sceneId, scenario_types: ['rear-end', 'head-on'],
                    num_per_type: 5, seed: 0, start_frame: 0, num_frames: 20
                })
            });
            const d = await resp.json();
            const ok = d.success && d.num_samples > 0;
            push(ok ? '✓' : '✗', ok ? 'pass' : 'fail', '④ GNN 语料收集',
                ok ? `样本 ${d.num_samples}，特征维 ${d.feature_dim}，碰撞率 ${(d.collision_rate * 100).toFixed(0)}%，文件 ${d.output_path}`
                   : (d.detail || '收集失败'));
            if (ok) this._labGnnCommand(d.output_path);
        } catch (e) { push('✗', 'fail', '④ GNN 语料', e.message); }

        push('注', 'pass', '自检结束', '注：② 用"一个真实物体 + 一个自动合成参与者"，结果反映引擎本身；若用真实噪声轨迹（批量）可能因数据质量被质量门拦下。');
    }

    async _samLoadModels() {
        try {
            const resp = await fetch(`${API_BASE}/sam/models`);
            const data = await resp.json();
            if (!data.success) return;
            const sel = document.getElementById('sam3dModelSelect');
            if (!sel) return;
            const cur = data.current;
            sel.innerHTML = data.models
                .filter(m => m.exists)
                .map(m => `<option value="${m.type}">${m.type}${m.type === cur ? '（当前）' : ''} · ${m.size_mb}MB</option>`)
                .join('');
            if (cur) sel.value = cur;
        } catch (e) {
            console.error('加载 SAM 模型列表失败:', e);
        }
    }

    async _samSetModel(modelType) {
        try {
            const resp = await fetch(`${API_BASE}/sam/model`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ model_type: modelType }),
            });
            const data = await resp.json();
            if (!resp.ok || !data.success) throw new Error(data.detail || ('HTTP ' + resp.status));
            this._sam3dStatus(`已切换到分割模型 ${data.current}`);
        } catch (e) {
            this._sam3dStatus('切换模型失败: ' + e.message, true);
            this._samLoadModels();
        }
    }

    _sam3dStatus(msg, isError = false) {
        const el = document.getElementById('sam3dStatus');
        if (el) {
            el.textContent = msg;
            el.style.color = isError ? '#e5484d' : 'inherit';
        }
        if (msg) this.updateStatus('SAM 3D: ' + msg);
    }

    _samSrcStatus(msg, isError = false) {
        const el = document.getElementById('samSourceStatus');
        if (el) { el.textContent = msg; el.style.color = isError ? '#e5484d' : 'inherit'; }
        this._sam3dStatus(msg, isError);
    }

    // 读取「替换目标物体」下拉框的 track_id。
    // 注意：track_id 允许为 0（T0 是合法物体），不能用 `!targetId` 判断——
    // parseInt('0') === 0 是 falsy，会把 T0 误判成"没选物体"。
    _sam3dSelectedTargetId() {
        const sel = document.getElementById('sam3dTargetObject');
        if (!sel) return null;
        const raw = String(sel.value ?? '').trim();
        if (raw === '') return null;
        const n = parseInt(raw, 10);
        return Number.isFinite(n) ? n : null;
    }

    openSamSourceModal() {
        if (!this.state.sceneId) { alert('请先加载场景'); return; }
        const targetId = this._sam3dSelectedTargetId();
        if (targetId === null) { alert('请先在左侧选择要替换的目标物体'); return; }
        document.getElementById('samSourceModal').classList.add('active');
        this.samPoints = [];
        this.samMask = null;
        this.samMaskImage = null;
        this._samSrcLoadCandidates(targetId);
        this._samSrcLoadFrame(this.state.currentFrame, false);
    }

    // 多帧源图：切换到某一帧重新做高清渲染（遮挡少、看得全的帧更适合单图重建）
    async _samSrcLoadFrame(frameIdx, keepPoints = false) {
        const total = Math.max(1, this.state.totalFrames || 1);
        const f = Math.max(0, Math.min(total - 1, parseInt(frameIdx, 10) || 0));
        if (!keepPoints) {
            this.samPoints = [];
            this.samMask = null;
            this.samMaskImage = null;
        }
        this.samSrcFrameIdx = f;
        const label = document.getElementById('samSrcFrameLabel');
        if (label) label.textContent = `帧 ${f} / ${total - 1}`;
        this._samSrcRenderCandidates();
        this._samSrcStatus(`正在渲染第 ${f} 帧高清源图...`);
        try {
            const resp = await fetch(`${API_BASE}/sam/source_image`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ scene_id: this.state.sceneId, frame_idx: f, max_dim: 1536 }),
            });
            const data = await resp.json();
            if (!resp.ok || !data.success) throw new Error(data.detail || '渲染失败');
            const img = new Image();
            img.onload = () => {
                this.samSrcWidth = data.width;
                this.samSrcHeight = data.height;
                this.samSrcImage = img;
                const c = document.getElementById('samSourceCanvas');
                if (c) { c.width = this.samSrcWidth; c.height = this.samSrcHeight; }
                this._samSrcRedraw();
                this._samSrcStatus(`第 ${f} 帧源图 ${data.width}×${data.height}：左键加点，右键撤销`);
            };
            img.src = 'data:image/png;base64,' + data.image;
        } catch (e) {
            this._samSrcStatus('源图渲染失败: ' + e.message, true);
        }
    }

    async _samSrcLoadCandidates(targetId) {
        this.samSrcCandidates = null;
        this._samSrcRenderCandidates();
        try {
            const resp = await fetch(`${API_BASE}/sam/frame_candidates`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ scene_id: this.state.sceneId, target_object_id: targetId, max_candidates: 8 }),
            });
            const data = await resp.json();
            if (resp.ok && data.success) {
                this.samSrcCandidates = data.candidates || [];
                this._samSrcRenderCandidates();
            }
        } catch (e) {
            console.error('帧评分失败:', e);
        }
    }

    _samSrcRenderCandidates() {
        const box = document.getElementById('samSrcCandidates');
        if (!box) return;
        const cands = this.samSrcCandidates;
        if (!cands || !cands.length) {
            box.innerHTML = '<span class="cand-hint">正按遮挡程度为各帧打分…（遮挡越少、越完整越靠前）</span>';
            return;
        }
        box.innerHTML = '<span class="cand-hint">推荐帧（遮挡少→多）：</span>' + cands.map((c, i) => {
            const occ = Math.round((c.occlusion || 0) * 100);
            const active = (c.frame_idx === this.samSrcFrameIdx) ? ' active' : '';
            return `<button class="cand-chip${active}" data-frame="${c.frame_idx}">#${c.frame_idx}${i === 0 ? '首选' : ''} 遮挡${occ}%</button>`;
        }).join('');
        box.querySelectorAll('.cand-chip').forEach(btn => {
            btn.addEventListener('click', () => this._samSrcLoadFrame(parseInt(btn.dataset.frame, 10), false));
        });
    }

    closeSamSourceModal() {
        document.getElementById('samSourceModal').classList.remove('active');
    }

    _samSrcSetType(type) {
        this.samPointType = type;
        const fg = document.getElementById('samSrcFgBtn');
        const bg = document.getElementById('samSrcBgBtn');
        if (fg) fg.classList.toggle('active', type === 'fg');
        if (bg) bg.classList.toggle('active', type === 'bg');
    }

    _samSrcUndo() {
        if (!this.samPoints.length) return;
        this.samPoints.pop();
        this._samSrcRedraw();
        this._samSrcSegmentDebounced();
    }

    _samSrcClear() {
        this.samPoints = [];
        this.samMask = null;
        this._samSrcRedraw();
        this._samSrcStatus('已清除所有点');
    }

    _sam3dAddObject() {
        this.samPoints = [];
        this.samMask = null;
        this.samMaskImage = null;
        this.samSrcFrameIdx = this.state.currentFrame;
        this.samSrcCandidates = null;
        this._sam3dPopulateTargets();
        const sel = document.getElementById('sam3dTargetObject');
        if (sel) sel.value = '';
        this._sam3dStatus('请选择下一个要替换的目标物体，然后点「开始交互分割」');
    }

    _sam3dPopulateTargets() {
        const sel = document.getElementById('sam3dTargetObject');
        if (!sel) return;
        const cur = sel.value;
        const objs = this.state.objects || [];
        sel.innerHTML = '<option value="">（选择要替换的物体）</option>' + objs.map(o => {
            const tid = (o.track_id !== undefined && o.track_id !== null) ? o.track_id : o.object_id;
            return `<option value="${tid}">T:${tid}${o.type ? ' · ' + o.type : ''}</option>`;
        }).join('');
        if (cur) sel.value = cur;
    }

    // ==================== 视图点击选择「替换目标物体」 ====================

    _sam3dStartPickTarget() {
        if (!this.state.sceneId) { alert('请先加载场景'); return; }
        this._sam3dPickingTarget = true;
        const btn = document.getElementById('sam3dPickTargetBtn');
        if (btn) btn.classList.add('active');
        this._sam3dStatus('请在视图（2D 画布或 3D 视图）中点击要替换的物体…');
    }

    _sam3dFinishPickTarget(trackId) {
        if (!this._sam3dPickingTarget) return false;
        const tid = parseInt(trackId, 10);
        if (!Number.isFinite(tid)) return false;
        this._sam3dPickingTarget = false;
        const btn = document.getElementById('sam3dPickTargetBtn');
        if (btn) btn.classList.remove('active');
        this._sam3dPopulateTargets();
        const sel = document.getElementById('sam3dTargetObject');
        if (sel) {
            if (![...sel.options].some(o => parseInt(o.value, 10) === tid)) {
                const opt = document.createElement('option');
                opt.value = String(tid);
                opt.textContent = `T:${tid}`;
                sel.appendChild(opt);
            }
            sel.value = String(tid);
        }
        const label = document.getElementById('sam3dSrcFrameLabel');
        if (label) label.textContent = `帧 ${this.state.currentFrame}`;
        this._sam3dStatus(`已选择替换目标 T:${tid}，点「开始交互分割」继续`);
        return true;
    }

    _samSrcCanvasPoint(e) {
        const c = e.currentTarget;
        const rect = c.getBoundingClientRect();
        return {
            x: (e.clientX - rect.left) / rect.width * c.width,
            y: (e.clientY - rect.top) / rect.height * c.height,
        };
    }

    _samSrcHandleClick(e) {
        if (e.button === 2) return;
        if (!this.samSrcWidth) return;
        const p = this._samSrcCanvasPoint(e);
        this.samPoints.push({ x: p.x, y: p.y, label: this.samPointType });
        this._samSrcRedraw();
        this._samSrcSegmentDebounced();
    }

    _samSrcRedraw() {
        const c = document.getElementById('samSourceCanvas');
        if (!c) return;
        const ctx = c.getContext('2d');
        ctx.clearRect(0, 0, c.width, c.height);
        if (this.samSrcImage) ctx.drawImage(this.samSrcImage, 0, 0, c.width, c.height);
        if (this.samMask && this.samMaskImage) {
            ctx.save(); ctx.globalAlpha = 0.4;
            ctx.drawImage(this.samMaskImage, 0, 0, c.width, c.height);
            ctx.restore();
        }
        const r = Math.max(3, c.width / 300);
        for (const p of this.samPoints) {
            ctx.beginPath();
            ctx.arc(p.x, p.y, r, 0, Math.PI * 2);
            ctx.fillStyle = p.label === 'fg' ? '#22c55e' : '#ef4444';
            ctx.fill();
            ctx.strokeStyle = '#fff';
            ctx.lineWidth = 1.5;
            ctx.stroke();
        }
    }

    _samSrcSegmentDebounced() {
        if (this._sam3dSegTimer) clearTimeout(this._sam3dSegTimer);
        this._sam3dSegTimer = setTimeout(() => this._samSrcSegment(), 300);
    }

    async _samSrcSegment() {
        if (!this.state.sceneId || !this.samPoints.length || !this.samSrcWidth) return;
        try {
            const resp = await fetch(`${API_BASE}/sam/segment`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    scene_id: this.state.sceneId,
                    frame_idx: (this.samSrcFrameIdx !== undefined ? this.samSrcFrameIdx : this.state.currentFrame),
                    points: this.samPoints.map(p => [p.x / this.samSrcWidth, p.y / this.samSrcHeight]),
                    point_labels: this.samPoints.map(p => p.label === 'fg' ? 1 : 0),
                }),
            });
            const data = await resp.json();
            if (!resp.ok || !data.success) throw new Error(data.detail || ('HTTP ' + resp.status));
            this.samMask = data.mask;
            if (!this.samMaskImage) this.samMaskImage = new Image();
            this.samMaskImage.src = `data:image/png;base64,${this.samMask}`;
            this._samSrcRedraw();
        } catch (e) {
            this._samSrcStatus('分割失败: ' + e.message, true);
        }
    }

    async _samSrcGenerate() {
        if (!this.state.sceneId) { alert('请先加载场景'); return; }
        const targetId = this._sam3dSelectedTargetId();
        if (targetId === null) { alert('请先选择要替换的目标物体'); return; }
        if (!this.samPoints.length) { alert('请先在源图上点选前景/背景点'); return; }

        this._samSrcStatus(`正在重建并替换物体 #${targetId}（首次约需数分钟）...`);
        try {
            const resp = await fetch(`${API_BASE}/sam3d/replace`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    scene_id: this.state.sceneId,
                    frame_idx: (this.samSrcFrameIdx !== undefined ? this.samSrcFrameIdx : this.state.currentFrame),
                    target_object_id: targetId,
                    points: this.samPoints.map(p => [p.x / this.samSrcWidth, p.y / this.samSrcHeight]),
                    point_labels: this.samPoints.map(p => p.label === 'fg' ? 1 : 0),
                }),
            });
            const data = await resp.json();
            if (!resp.ok || !data.success) throw new Error(data.detail || ('HTTP ' + resp.status));
            this._samSrcStatus(`已替换物体 #${targetId} → 新物体 #${data.object_id}（scale=${Number(data.scale).toFixed(3)}）`);
            this.lastSamObjectId = data.object_id;
            this._sam3dShowScaleControl(1.0);
            // 选中替换后的新物体：后续轨迹/位姿编辑都作用于它，
            // 避免仍选中被替换掉的旧物体（会导致旧模型被"复活"）
            this.state.selectedObjects = [data.object_id];
            if (this.viewer3d) {
                await this.viewer3d.loadFrame(this.state.sceneId, this.state.currentFrame, true);
                if (typeof this.viewer3d.selectTrack === 'function') {
                    this.viewer3d.selectTrack(data.object_id);
                }
            } else {
                this.updateSelectedObjectPanel();
            }
            // 显示导出 .ply 的下载链接 + 预览按钮
            const dl = document.getElementById('sam3dDownloadBtn');
            if (dl) {
                dl.href = `${API_BASE}/sam3d/objects/${this.state.sceneId}/${data.object_id}/download`;
                dl.style.display = 'block';
            }
            const pv = document.getElementById('sam3dPreviewBtn');
            if (pv) pv.style.display = 'block';
            this.samPoints = [];
            this.samMask = null;
            this._samSrcRedraw();
            this.closeSamSourceModal();
            await this.loadFrame(this.state.currentFrame);
            this._sam3dPopulateTargets();
        } catch (e) {
            console.error('替换失败:', e);
            this._samSrcStatus('替换失败: ' + e.message, true);
        }
    }

    async openSamPreview() {
        if (!this.state.sceneId || !this.lastSamObjectId) { alert('请先生成一个物体'); return; }
        this._sam3dStatus('正在渲染旋转预览...');
        try {
            const resp = await fetch(`${API_BASE}/sam3d/preview`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ scene_id: this.state.sceneId, object_id: this.lastSamObjectId, num_frames: 36, size: 512 }),
            });
            const data = await resp.json();
            if (!resp.ok || !data.success) throw new Error(data.detail || ('HTTP ' + resp.status));
            this.samPreviewFrames = data.frames || [];
            this.samPreviewIdx = 0;
            const slider = document.getElementById('samPreviewSlider');
            if (slider) { slider.max = Math.max(0, this.samPreviewFrames.length - 1); slider.value = 0; }
            document.getElementById('samPreviewModal').classList.add('active');
            this._samPreviewShow(0);
            this._sam3dStatus('预览已就绪');
            this._samPreviewPlay();
        } catch (e) {
            this._sam3dStatus('预览失败: ' + e.message, true);
        }
    }

    closeSamPreview() {
        this._samPreviewStop();
        document.getElementById('samPreviewModal').classList.remove('active');
    }

    _samPreviewShow(idx) {
        const img = document.getElementById('samPreviewImg');
        const label = document.getElementById('samPreviewFrameLabel');
        if (!this.samPreviewFrames.length) return;
        this.samPreviewIdx = (idx + this.samPreviewFrames.length) % this.samPreviewFrames.length;
        if (img) img.src = 'data:image/png;base64,' + this.samPreviewFrames[this.samPreviewIdx];
        if (label) label.textContent = `${this.samPreviewIdx + 1} / ${this.samPreviewFrames.length}`;
        const slider = document.getElementById('samPreviewSlider');
        if (slider) slider.value = this.samPreviewIdx;
    }

    _samPreviewPlay() {
        this._samPreviewStop();
        this.samPreviewTimer = setInterval(() => this._samPreviewShow(this.samPreviewIdx + 1), 100);
    }

    _samPreviewStop() {
        if (this.samPreviewTimer) { clearInterval(this.samPreviewTimer); this.samPreviewTimer = null; }
    }

    _samPreviewSeek(idx) {
        this._samPreviewStop();
        this._samPreviewShow(idx);
    }

    render() {
        this.renderImage();
        this.renderOverlay();
        this.updateObjectList();
    }
}

// 初始化应用
const studio = new DGGTStudio();

// 添加关键帧按钮事件
document.getElementById('addKeyframeBtn').addEventListener('click', () => studio.addKeyframe());
document.getElementById('saveKeyframesBtn').addEventListener('click', () => studio.saveKeyframes());
