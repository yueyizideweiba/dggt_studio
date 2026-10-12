# DGGT Studio — 4DGS 动态场景编辑画布与 Corner Case 生成

一个基于 **4D Gaussian Splatting (4DGS)** 的自动驾驶动态场景**编辑 + 事故（corner case）生成**平台。
把重建出来的 4D 驾驶场景当成一块可编辑画布：选中/移动/旋转任意交通参与者、改轨迹、加天气、
生成可控参数的事故，并用一套**可解释的多阶段交互模型**（预警区 / 危险区 / 发生区）标注事故的演化过程。

> 本仓库是**我们自己的工程项目**（`dggt_studio`）。底层的 4DGS 重建能力基于开源的 DGGT
> （Xiaomi Research，见文末「上游依赖」），本仓库在其之上做了数据管线、编辑引擎、事故生成、
> 多阶段可解释判据、Web 编辑器与仿真桥接。本 README 只描述**本仓库**的内容。

---

## 项目定位

**通过 4DGS + 交通参与对象运动参数，批量生成 Corner Case。** 核心是**参数控制**：
事故"是否发生、何时发生、多难规避"都由参数决定；研究目的是**避免事故**，所以重点放在
**最晚反应点 / 回避条件**，而不是撞完之后怎么飞。

## 主要功能

### 场景编辑（`studio/`）

- **动态物体选择与编辑**：点击选中、Shift 多选、实时高亮；拖拽 / 三轴精确输入 / 旋转 / 缩放。
- **轨迹可视化与编辑**：轨迹曲线 + 关键帧编辑（增删改）+ 自适应平滑（拖动一个点整条路径按衰减跟随）。
- **智能轨迹重定向**：视图里拾取起点/终点，沿高精地图车道**自动算路**（严格贴车道、必要时才变道、
  平滑转向），并可直接在该路线上放置一辆沿路线行驶的车辆。
- **三维极端天气**：大雨 / 暴风雨 / 大雪 / 暴风雪 / 浓雾，粒子在相机视锥内生成（有深度、风偏、
  时间连续性），并支持体积雾化 / 能见度衰减。
- **物体生成与替换**：SAM3D 单图重建 3D 高斯替换物体；文本（LLM）添加实体、自然语言改轨迹。

### Corner Case 生成（`studio/backend/`）

- **多种事故类型**：紧急刹车、追尾、变道加塞、路口 T-Bone、对向碰撞、行人横穿（鬼探头）、
  前车闪开露出障碍、连环追尾、自然冲突（保轨迹调时序）等。
- **参数化采样**：初始距离、相对速度、制动强度、反应延迟、横向偏置、碰撞强度、天气/道路上下文……
  `sampling_seed` + `intensity` 决定"事故是否发生 / 何时发生"，完全可复现。
- **物理约束与质量门**：OBB（SAT）碰撞检测、动量守恒碰撞响应、防穿模分离；
  `quality_report` 显式检查速度/加速度/jerk/偏航率/单帧位移/包围盒穿透/事件存在/标注一致性。
- **碰撞关键帧**：自动识别"最晚反应关键帧"（在轨迹上标碰撞帧与最晚反应帧）。

### 多阶段交互模型：预警区 / 危险区 / 发生区（`studio/backend/phase_zones.py`）

把"是否会发生事故"的二值判断，推广为**"在哪个阶段、该做什么反应、还来不来得及"**的多维可解释判据：

| 区域 | 判据 | 系统该做什么 |
| --- | --- | --- |
| 安全 `safe` | 预测轨迹无冲突 | 正常行驶 |
| **预警区** `warning` | 0 < a\* ≤ a_comfort | 监测 + 准备，舒适减速，评估变道空间 |
| **危险区** `danger` | a_comfort < a\* ≤ a_emergency | 立即紧急制动（AEB）或按规则变道避让 |
| **发生区** `occurrence` | a\* > a_emergency（制动且转向都不可解） | 最小化损失 |
| 撞击 `impact` | 已接触 | 稳方向、持续制动、双闪、最小风险停车 |
| 遮挡 `occluded` | 参与者不可见（鬼探头） | 降低车速 / 遮挡推理 |

- **多因素判据**（逐帧可追溯）：纵向制动所需减速度 a\*（二分精确解 + 一维解析上界）、
  横向转向所需加速度 a_lat（OBB 感知横向净空）、撞击严重度（Δv·几何·VRU）、路面附着 μ、
  TTC / 反应裕度、车头时距 THW、机动余量 μ\*。
- **二维风险**：紧急度（区域）× 严重度（Δv·几何·类型）→ `critical / severe / high / elevated / low`，
  并输出 `binding`（是哪个因素决定了该帧的区域）与完整**决策链路**。
- **最晚反应点（LRP）**：`danger → occurrence` 的分界帧，附该点的 TTC 与机动余量。
- **Corner case 度量**：预警区被**压缩 / 跳过**，给出"有多难规避"的可比较语言。
- 形式化定义 / 与 ISO 15623、RSS、ICS、THW/TTC 的对应 / 伪代码 / 超参数表：
  见 [`docs/multiphase_interaction_model.md`](docs/multiphase_interaction_model.md)。

### CARLA 闭环（`carla_bridge/`，可选）

「编辑 → 仿真 → 再看 → 再编辑」闭环：把当前编辑好的场景导出到 CARLA 回放 / 自车闭环，
出 MP4、碰撞事件表、最小距离 / 最小 TTC，并支持实时 MJPEG 直播。

## 快速开始

```bash
cd /autodl-fs/data/dggt-main

# 常用三件套（studio + SAM3D + text2entity）
bash start_all.sh
# 只要 studio（纯轨迹编辑 / 渲染够用）
bash start_all.sh --studio-only
# 带上 CARLA
bash start_all.sh --carla
```

打开 **<http://localhost:8000/studio/>**（后端已把前端挂在 `/studio/`，只需放行 8000 一个端口）。

| 服务 | 端口 | 干什么 |
| --- | --- | --- |
| **studio** | 8000 | 编辑器 / 渲染 / 接口 / 前端（`/studio/`） |
| SAM3D | 8001 | 单图重建 3D 高斯：替换物体、生成新物体 |
| text2entity | 8002 | LLaDA 文生图 + LLM：文本添加实体、语言改轨迹 |
| CARLA | 2000 | 事故回放 / 实时仿真（可选） |

停止 / 重启：`bash stop_all.sh`、`bash restart_all.sh --studio-only`（改完后端代码后最常用）。

## 项目结构

```
dggt-main/
├── studio/                 # 编辑器（我们的主战场）
│   ├── backend/            # FastAPI 后端：api_server / track_manager / corner_case /
│   │                       #   scenario_engine / phase_zones / route_entity / road_rules ...
│   ├── frontend/           # 原生 JS 前端（index.html / app.js / viewer3d.js / ui-panels.js）
│   └── README.md           # ★ studio 的详细文档（功能、修复记录、多阶段模型说明）
├── docs/                   # 方法论 / 形式化规范
│   └── multiphase_interaction_model.md
├── tools/                  # 数值验证脚本（verify_phase_zones / verify_map_alignment ...）
├── dggt_engine.py          # 4DGS 渲染引擎（场景加载 / 逐帧渲染 / 物体覆盖 / 外观缓存）
├── inference.py, train.py  # 上游 DGGT 的推理 / 训练入口
├── start_all.sh / stop_all.sh / restart_all.sh
├── sam-3d-objects/         # SAM3D 单图重建服务
├── text2entity/            # 文生图 + LLM 服务
├── carla_bridge/           # CARLA 仿真桥接
└── output/                 # 重建好的场景（`output/waymo_eval_14/<scene>/<scene>/`）
```

## 场景数据格式

一个重建好的场景目录（如 `output/waymo_eval_14/018/018/`）大致包含：

```
scene_meta.json                 # 场景元信息（帧数、相机、间隔…）
gaussians/
  static_scene.ply              # 静态背景高斯
  sky_scene.ply                 # 天空高斯（可选）
  frame_XXXX_dynamic.ply        # 每帧动态物体高斯（带 object_id 属性）
dynamic_objects/
  frame_XXXX_objects.json       # 每帧动态物体位姿 / 尺寸元数据
ego_pose/                       # 自车相机内外参
```

## 数值验证

```bash
# 多阶段交互模型（受控对向三段式 / 转向接管 / 鬼探头 / 真实场景）
/root/autodl-tmp/conda_envs/dggt/bin/python tools/verify_phase_zones.py \
    output/waymo_eval_14/018/018 0 24
```

验证全部是**数值化**的（不靠看图）：相态单调切换、a\* 与阈值的关系、六因素齐全、
遮挡期无预警、真实场景的事件与关键因素。

## 上游依赖

本仓库的 4DGS 重建内核来自开源项目 **DGGT**（Xiaomi Research，
[arXiv:2512.03004](https://arxiv.org/abs/2512.03004) /
[project page](https://xiaomi-research.github.io/dggt/)）。上游的模型定义、预训练权重、
数据处理与训练/推理入口（`dggt/`、`inference.py`、`train.py`、`datasets/`、`pretrained/` 等）
保留原样，使用请遵循其许可证（Apache-2.0，部分文件来自 VGGT，见 `NOTICE`）。

本仓库新增/维护的是：`studio/`（编辑器后端+前端）、`docs/`、`tools/`、
`dggt_engine.py` 的编辑相关扩展、`carla_bridge/`、`sam-3d-objects/`、`text2entity/`
以及各服务启停脚本。

## 详细文档

- 编辑器功能、修复记录、多阶段模型工程说明：[`studio/README.md`](studio/README.md)
- 多阶段交互模型的形式化规范（可写论文级别）：[`docs/multiphase_interaction_model.md`](docs/multiphase_interaction_model.md)
- CARLA 桥接：[`carla_bridge/README_CARLA.md`](carla_bridge/README_CARLA.md)
