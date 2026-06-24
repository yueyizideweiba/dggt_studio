# DGGT Studio

基于 4D Gaussian Splatting (4DGS) 的动态场景编辑工具。支持在三维自由视角下查看
4DGS 场景、选中并编辑动态物体的位姿与运动轨迹，并通过修改轨迹快速生成交通事故
（Corner Case）场景。

## 功能特性

- **3D 自由视角浏览**：普通 4DGS 渲染只能锁定自车视角，本工具用服务端 gsplat
  按任意相机位姿渲染当前帧并回传图像，浏览器零 GPU 负担即可 360° 环绕查看。
- **基于 track 的物体编辑**：每帧动态物体的 id 不连续，工具用跨帧 track_id 稳定
  追踪同一物理物体，保证编辑、轨迹、删除在整段序列中作用于同一对象。
- **三维拖拽编辑**：选中物体后在地面平面内拖动移动、旋转，或直接拖动轨迹线上的
  顶点编辑运动轨迹（编辑后物体沿新轨迹运动）。
- **Corner Case 生成**：底层通过修改相关 track 的轨迹来制造事故效果，支持追尾、
  紧急刹车、变道加塞、路口侧碰、对向碰撞等，可分别为事故中的不同角色指派物体。
- **撤销 / 重做**：基于轨迹编辑快照栈，覆盖移动、旋转、删除、轨迹编辑、事故生成。

## 项目结构

```
dggt_studio/
├── README.md
├── requirements.txt
├── dggt_engine.py            # 4DGS 渲染引擎（gsplat 光栅化 + 轨迹控制）
├── dggt/
│   ├── __init__.py
│   └── scene_edit/           # 场景编辑规格解析/执行（asset_bank, executor,
│                             #   geometry, loader, specs）
└── studio/
    ├── README.md
    ├── backend/              # FastAPI 后端
    │   ├── api_server.py      # API 服务（渲染 / 编辑 / track / corner case）
    │   ├── track_manager.py   # 跨帧 track 追踪与编辑/撤销管理
    │   ├── corner_case.py     # 基于轨迹的事故场景生成
    │   └── requirements.txt
    └── frontend/             # 原生 JS 前端
        ├── index.html
        ├── app.js
        ├── viewer3d.js        # 3D 自由视角视图（服务端渲染客户端）
        └── styles.css
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

> `torch` 与 `gsplat` 需与本机 CUDA 版本匹配，建议参考各自官方说明安装。

### 2. 启动后端

```bash
cd studio/backend
python api_server.py
```

服务在 `http://localhost:8000` 启动。

### 3. 打开前端

```bash
cd studio/frontend
python -m http.server 3000
```

浏览器访问 `http://localhost:3000`。

## 场景数据格式

场景目录需包含：

```
<scene>/
├── gaussians/
│   ├── static_scene.ply           # 静态场景（世界坐标）
│   ├── sky_scene.ply              # 天空盒（可选）
│   └── frame_XXXX_dynamic.ply     # 各帧动态高斯（物体局部坐标 + object_id）
├── dynamic_objects/
│   └── frame_XXXX_objects.json    # 各帧动态物体位姿与尺寸
└── ego_pose/
    └── frame_XXXX_ego.json        # 各帧自车相机内外参
```

## 许可证

本项目基于 DGGT 项目开发，遵循相应开源协议。
