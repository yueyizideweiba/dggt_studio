/**
 * DGGT Studio - 4DGS动态物体编辑画布前端应用 (修复版)
 * 修复内容：
 * - 完善轨迹/网格显示切换功能
 * - 修复工具选择和切换逻辑
 * - 优化物体选择和坐标转换
 * - 添加渲染缓存和性能优化
 */

const API_BASE = 'http://localhost:8000/api';

class DGGTStudio {
    constructor() {
        this.state = {
            sceneId: null,
            scenePath: null,
            totalFrames: 0,
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
            viewMode: '2d'  // '2d' | '3d'
        };

        this.canvas = null;
        this.ctx = null;
        this.currentImage = null;
        this.viewer3d = null;  // 3D 视图实例（懒加载）

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
        document.getElementById('playBtn').addEventListener('click', () => this.togglePlay());
        document.getElementById('frameInput').addEventListener('change', (e) => {
            const frame = parseInt(e.target.value);
            if (frame >= 0 && frame < this.state.totalFrames) {
                this.jumpToFrame(frame);
            }
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
                this.state.renderCache.clear();
                
                document.getElementById('scenePath').textContent = this.state.scenePath;
                document.getElementById('sceneIdDisplay').textContent = this.state.sceneId;
                document.getElementById('sceneTotalFrames').textContent = this.state.totalFrames;
                document.getElementById('totalFrames').textContent = `/ ${this.state.totalFrames}`;
                
                document.getElementById('loadSceneModal').classList.remove('active');
                
                await this.loadFrame(this.state.currentFrame);
                this.updateStatus(`场景已加载: ${this.state.sceneId}`);
            }
        } catch (error) {
            console.error('加载场景失败:', error);
            this.updateStatus('加载场景失败');
            alert('加载场景失败: ' + error.message);
        }
    }


    async loadFrame(frameIdx) {
        if (!this.state.sceneId) return;

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
        try {
            const response = await fetch(`${API_BASE}/edit/object/rotation`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    scene_id: this.state.sceneId,
                    object_id: objectId,
                    frame_idx: this.state.currentFrame,
                    delta_yaw: deltaAngle
                })
            });
            
            if (response.ok) {
                this.clearFrameCache(this.state.currentFrame);
                await this.loadFrame(this.state.currentFrame);
            }
        } catch (error) {
            console.error('更新物体旋转失败:', error);
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

    jumpToFrame(frameIdx) {
        this.state.currentFrame = frameIdx;
        document.getElementById('frameInput').value = frameIdx;
        this.loadFrame(frameIdx);
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

        const step = async () => {
            if (!this.state.isPlaying || this.state.viewMode !== '3d') return;
            const t0 = performance.now();

            // 预取后续几帧到缓存
            for (let k = 1; k <= 3; k++) {
                const f = this.state.currentFrame + k;
                if (f < this.state.totalFrames) viewer.prefetchFrame(f);
            }

            if (this.state.currentFrame < this.state.totalFrames - 1) {
                this.state.currentFrame += 1;
                document.getElementById('frameInput').value = this.state.currentFrame;
                viewer.frameIdx = this.state.currentFrame;
                await viewer.loadFrame(this.state.sceneId, this.state.currentFrame, true);
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
                panel.innerHTML = `
                    <div class="selected-object-info">
                        <div class="info-row"><span class="label">Track:</span><span class="value">T:${tid}</span></div>
                        <div class="info-row"><span class="label">类型:</span><span class="value">${obj.type || '未知'}</span></div>
                        <div class="info-row"><span class="label">位置:</span><span class="value">(${pos[0].toFixed(1)}, ${pos[1].toFixed(1)}, ${pos[2].toFixed(1)})</span></div>
                        <div class="info-row"><span class="label">尺寸:</span><span class="value">${dimStr}</span></div>
                        <div class="info-row"><span class="label">已编辑:</span><span class="value">${obj.edited ? '是' : '否'}</span></div>
                        <div class="button-row">
                            <button class="btn btn-small btn-secondary" onclick="studio.viewer3d && studio.viewer3d.focusSelected()">聚焦</button>
                            <button class="btn btn-small btn-danger" onclick="studio.delete3dSelected()">删除物体</button>
                        </div>
                    </div>
                `;
            }
        } else {
            panel.innerHTML = `
                <div class="multiple-objects">
                    <p>已选择 ${this.state.selectedObjects.length} 个物体</p>
                </div>
            `;
        }
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
    }

    renderCornerRoles() {
        const sel = document.getElementById('cornerScenarioType');
        const rolesDiv = document.getElementById('cornerRoles');
        const descEl = document.getElementById('cornerScenarioDesc');
        if (!sel || !rolesDiv) return;
        const scenario = this.cornerScenarios[sel.value];
        if (!scenario) { rolesDiv.innerHTML = ''; return; }
        if (descEl) descEl.textContent = scenario.desc || '';

        // 切换场景时重置角色指派
        this.cornerRoleAssign = {};
        this.cornerActiveRole = null;

        rolesDiv.innerHTML = scenario.roles.map(r => `
            <div class="cc-role" data-role="${r.key}">
                <span class="cc-role-label">${r.label}</span>
                <button class="btn btn-small cc-role-btn" data-role="${r.key}">指定</button>
                <span class="cc-role-value" data-role="${r.key}">未指定</span>
            </div>
        `).join('');

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
    }

    _cornerRoleLabel(roleKey) {
        const sel = document.getElementById('cornerScenarioType');
        const scenario = this.cornerScenarios[sel.value];
        const r = scenario && scenario.roles.find(x => x.key === roleKey);
        return r ? r.label : roleKey;
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
        const roles = {};
        for (const r of scenario.roles) {
            const tid = this.cornerRoleAssign[r.key];
            if (tid === undefined || tid === null) {
                if (!r.label.includes('可选')) {
                    alert(`请指定角色：${r.label}`);
                    return;
                }
            } else {
                roles[r.key] = tid;
            }
        }

        const startFrame = parseInt(document.getElementById('cornerStartFrame').value) || 0;
        const numFrames = parseInt(document.getElementById('cornerNumFrames').value) || 20;
        const intensity = parseFloat(document.getElementById('cornerIntensity').value) || 1.0;

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
                    intensity: intensity
                })
            });
            const data = await response.json();
            if (data.success) {
                this.cornerAffectedTracks = data.affected_tracks || [];
                this.state.renderCache.clear();
                if (this.viewer3d) {
                    this.viewer3d._invalidateCache();
                    this.viewer3d.trajectory = null;
                    if (this.viewer3d.selectedTrackId !== null) {
                        this.viewer3d._fetchTrajectory(this.viewer3d.selectedTrackId);
                    }
                }
                await this.loadFrame(this.state.currentFrame);
                this.saveEditHistory('corner_case', scenarioType);
                const statusEl = document.getElementById('cornerCaseStatus');
                if (statusEl) statusEl.textContent = `已生成「${scenario.name}」，影响 ${this.cornerAffectedTracks.length} 个物体的轨迹`;
                this.updateStatus(`已生成 Corner Case: ${scenario.name}`);
            } else {
                alert('生成失败: ' + (data.detail || '未知错误'));
            }
        } catch (error) {
            console.error('生成Corner Case失败:', error);
            alert('生成失败: ' + error.message);
        }
    }

    async clearCornerCase() {
        const tracks = [...(this.cornerAffectedTracks || [])];
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
            this.state.renderCache.clear();
            if (this.viewer3d) {
                this.viewer3d._invalidateCache();
                this.viewer3d.trajectory = null;
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
