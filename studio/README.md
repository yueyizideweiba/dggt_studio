# DGGT Studio - 4DGS动态物体编辑画布

一个基于Web的4D Gaussian Splatting (4DGS)动态场景编辑工具，支持灵活编辑和生成corner cases交通事故场景。

## 功能特性

### 🎯 核心功能

- **动态物体选择与编辑**
  - 点击选中画布中的动态物体
  - 支持多选 (Shift + 点击)
  - 实时高亮显示选中物体

- **拖拽与移动**
  - 直接拖拽物体移动位置
  - 支持X/Y/Z三轴精确位置输入
  - 旋转和缩放编辑

- **轨迹可视化与编辑**
  - 显示物体运动轨迹曲线
  - 关键帧编辑器
  - 添加/删除/修改关键帧
  - 实时预览轨迹变化

- **Corner Case生成**
  - 紧急刹车场景
  - 追尾预碰撞场景
  - 变道加塞场景
  - 路口侧碰 (T-Bone) 场景
  - 对向碰撞场景
  - 行人横穿（鬼探头）场景
  - 前车闪开露出障碍（幽灵障碍）场景
  - 支持场景族参数采样：初始距离、相对速度、制动强度、反应延迟、横向偏置、碰撞强度、天气/道路上下文等

- **物理碰撞模拟** ✨ 新增
  - 基于定向包围盒 (OBB) 的碰撞检测（分离轴定理 SAT）
  - 事故生成时按包围盒尺寸计算接触距离，避免车辆互相穿模
  - 动量守恒的碰撞响应
  - 碰撞时间预测

- **碰撞关键帧识别** ✨ 新增
  - 自动识别每个事故的"最晚反应关键帧"
  - 在轨迹上高亮标记碰撞帧（红色）和最晚反应帧（橙色菱形）
  - 计算反应时间窗口、关键帧距离、严重程度
  - 帮助自动驾驶系统评估最晚何时必须采取规避动作

- **质量准入报告** ✨ 新增
  - 事故生成后可在前端点击"查看质量报告"
  - 展示碰撞/near-miss、TTC、轨迹动力学、包围盒穿透和标注一致性检查
  - 支持查看完整原始 JSON，便于调参和批量生产筛选

- **三维极端天气 Corner Case** ✨ 新增
  - 3D 视图支持大雨、暴风雨、大雪、暴风雪、浓雾
  - 雨滴/雪花不是二维贴图，而是在相机视锥中生成的 3D 粒子，具备深度、风偏、时间连续性和投影缩放
  - 支持体积雾化/能见度衰减，模拟极端天气下传感器可视性降低
  - 支持横风、垂直风、强度和能见度参数调节

- **智能轨迹编辑** ✨ 增强
  - 拖动一个节点时，整条路径按平滑衰减自适应跟随（无需逐个调整节点）
  - 可调影响范围与衰减方式（smooth / linear / gaussian）
  - 一键平滑整条轨迹，消除手动编辑抖动

### 🎬 渲染与预览

- 实时渲染预览
- 帧级别导航
- 视频播放控制
- 多种显示模式（边界框、ID、轨迹）

### 💾 数据管理

- 场景加载与卸载
- 编辑历史记录
- 撤销/重做操作
- 多种格式导出（JSON、视频）

## 项目结构

```
studio/
├── backend/
│   ├── api_server.py      # FastAPI后端服务
│   └── requirements.txt    # Python依赖
├── frontend/
│   ├── index.html         # 主页面
│   ├── styles.css         # 样式表
│   └── app.js             # 前端应用逻辑
└── README.md              # 本文档
```

## 快速开始

### 1. 安装依赖

```bash
cd studio/backend
pip install -r requirements.txt
```

### 2. 启动后端服务

```bash
python api_server.py
```

服务将在 `http://localhost:8000` 启动。

### 3. 访问前端

直接在浏览器中打开 `frontend/index.html` 文件，或使用任何静态文件服务器：

```bash
cd studio/frontend
python -m http.server 3000
```

然后访问 `http://localhost:3000`。

## 使用指南

### 加载场景

1. 点击顶部工具栏的"加载场景"按钮
2. 输入场景路径（包含gaussians和dynamic_objects目录的根路径）
3. 选择是否加载天空盒和仅静态场景
4. 点击"加载"按钮

### 编辑物体

#### 选择物体
- 使用选择工具（V键）点击画布中的物体
- 按住Shift键多选

#### 移动物体
- 切换到移动工具（M键）
- 拖拽选中的物体
- 或在右侧面板输入精确坐标

#### 编辑轨迹
1. 选中物体后点击"编辑轨迹"按钮
2. 在关键帧编辑器中添加/修改关键帧
3. 点击"保存"应用更改

### 生成Corner Case

1. 在画布中选择要参与的物体（至少1-2个）
2. 点击"生成Corner Case"按钮
3. 选择场景类型：
   - 🛑 紧急刹车
   - 🚗💥 追尾预碰撞
   - ↔️ 变道
   - ➕ 路口交叉
4. 配置参数（起始帧、持续帧数）
5. 点击"生成场景"

### 导出结果

支持三种导出格式：
- **轨迹规格 (JSON)**: 导出物体运动轨迹
- **场景编辑规格**: 导出完整的场景编辑信息
- **渲染视频**: 导出渲染后的视频文件

## 键盘快捷键

| 快捷键 | 功能 |
|--------|------|
| `V` | 选择工具 |
| `M` | 移动工具 |
| `R` | 旋转工具 |
| `T` | 轨迹编辑工具 |
| `←` / `→` | 上一帧/下一帧 |
| `Space` | 播放/暂停 |
| `Delete` | 删除选中物体 |
| `Ctrl+Z` | 撤销 |
| `Ctrl+Y` | 重做 |
| `+` / `-` | 放大/缩小 |

## API文档

后端API提供以下主要端点：

### 场景管理
- `POST /api/scenes/load` - 加载场景
- `GET /api/scenes` - 列出场景
- `DELETE /api/scenes/{scene_id}` - 卸载场景

### 物体操作
- `GET /api/objects/{scene_id}` - 获取物体列表
- `GET /api/objects/{scene_id}/{object_id}/trajectory` - 获取物体轨迹

### 渲染
- `POST /api/render/frame` - 渲染单帧
- `POST /api/render/freeview` - 服务端自由视角渲染，支持 `weather` 字段生成三维极端天气粒子
- `POST /api/render/sequence` - 渲染视频序列

### 编辑
- `POST /api/edit/object/pose` - 编辑物体位姿
- `POST /api/edit/trajectory` - 编辑轨迹
- `POST /api/edit/object/offset` - 移动物体
- `DELETE /api/edit/object/{scene_id}/{object_id}` - 删除物体

### Corner Case
- `POST /api/corner_case/generate` - 生成corner case（支持 `enable_physics` 物理碰撞规则，返回碰撞关键帧分析与 `quality_report` 质量准入报告）
- `GET /api/corner_case/types` - 获取支持的类型
- `POST /api/corner_case/analyze_collision` - 分析两个 track 的碰撞，识别最晚反应关键帧，返回逐帧距离曲线
- `POST /api/corner_case/clear` - 清除生成的事故轨迹

### 智能轨迹编辑
- `POST /api/edit/track/point` - 编辑单帧轨迹点
- `POST /api/edit/track/point_adaptive` - 智能自适应编辑（拖动一个节点整条路径跟随）
- `POST /api/edit/track/smooth` - 平滑整条轨迹
- `POST /api/edit/track/trajectory` - 用关键帧列表设置整条轨迹

### 导出
- `POST /api/export/trajectory` - 导出轨迹
- `POST /api/export/scene_spec` - 导出场景规格

## 技术架构

### 后端
- **FastAPI**: 高性能Python Web框架
- **PyTorch**: 深度学习框架
- **gsplat**: 4DGS渲染引擎
- **NumPy/SciPy**: 数值计算

### 前端
- **原生JavaScript**: 无框架依赖
- **Canvas API**: 图像渲染
- **Fetch API**: HTTP请求

## 扩展开发

### 添加新的Corner Case类型

1. 在 `api_server.py` 的 `CornerCaseGenerator` 类中添加新的生成方法
2. 在前端 `index.html` 的场景类型列表中添加新卡片
3. 在 `app.js` 中添加对应的参数配置UI

### 自定义轨迹插值

修改 `dggt_engine.py` 中的 `_interp_pose` 方法实现不同的插值算法。

## 注意事项

1. 确保已安装CUDA和PyTorch GPU版本
2. 场景数据需要包含以下文件：
   - `gaussians/static_scene.ply`
   - `gaussians/frame_xxxx_dynamic.ply`
   - `dynamic_objects/frame_xxxx_objects.json`
   - `ego_pose/frame_xxxx_ego.json`
3. 大场景渲染可能需要较多GPU内存

## 许可证

本项目基于DGGT项目开发，遵循相应的开源协议。
