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

## 服务清单 / 一键启动

这一套是**四个服务**，默认只有第一个是必须的；其余按你要用的功能开（24G 卡别全塞满）：

| 服务 | 端口 | 干什么 | 启动命令 | 健康检查 |
| --- | --- | --- | --- | --- |
| **studio** | 8000 | 编辑器 / 渲染 / 接口 / 前端（前端就挂在 `/studio/`） | `bash start_all.sh --studio-only` 或 `cd carla_bridge && ./start_studio.sh` | `curl -s localhost:8000/api/carla/status` |
| **SAM3D** | 8001 | 单图重建 3D 高斯：**SAM3D 替换物体**、**生成/插入新物体**要用 | `bash start_all.sh --no-llm` | `curl -s localhost:8001/health` |
| **text2entity** | 8002 | LLaDA 文生图 + Qwen-VL/LLM：**文本添加实体**、语言编辑轨迹的**自然语言解析** | `bash start_all.sh` | `curl -s localhost:8002/health` |
| CARLA（可选） | 2000 | 事故回放 / 实时仿真 | `bash start_all.sh --carla` | 前端 ④ 页面 / `stop_all.sh` 释放 |

**一键起（幂等，已在跑的就跳过）：**

```bash
cd /autodl-fs/data/dggt-main
bash start_all.sh                # studio + SAM3D + text2entity（常用三件套）
bash start_all.sh --studio-only  # 只 studio（纯轨迹编辑/渲染够用）
bash start_all.sh --no-llm       # studio + SAM3D（不做文生图，但语言解析会退化成关键词规则）
bash start_all.sh --carla        # 再带上 CARLA
```

**单独起（等价的手工命令）：**

```bash
# studio
cd /autodl-fs/data/dggt-main && setsid nohup /root/autodl-tmp/conda_envs/dggt/bin/python     studio/backend/api_server.py > /autodl-fs/data/carla/logs/studio_backend.log 2>&1 &
# SAM3D
cd /autodl-fs/data/dggt-main && setsid nohup bash sam-3d-objects/run_sam3d_service.sh     >> /tmp/sam3d_service.log 2>&1 &
# text2entity（supervise.sh 是带看门狗的包装，比 run_service.sh 更不容易整条掉线）
cd /autodl-fs/data/dggt-main && setsid nohup bash text2entity/supervise.sh </dev/null >/dev/null 2>&1 &
```

**关 / 重启（脚本）：**

```bash
cd /autodl-fs/data/dggt-main
bash stop_all.sh                 # 停 studio + SAM3D + text2entity
bash stop_all.sh --carla         # 连 CARLA / 直播 / VNC 一起停
bash stop_all.sh --studio-only   # 只停 studio

bash restart_all.sh              # 先停再起（一套都重启）
bash restart_all.sh --studio-only  # 改完后端代码后最常用
bash restart_all.sh --no-llm     # 只重启 studio + SAM3D
```

两个脚本都**幂等**：已在跑的会跳过、没起来的会补起来，结束时打印端口与显存状态。
参数在 `start_all.sh` / `stop_all.sh` / `restart_all.sh` 之间是**一致**的
（`--studio-only` / `--no-llm` / `--no-sam3d` / `--carla`）。

> 注意 SAM3D / text2entity 都是"监督循环 + python 服务"两层，`stop_all.sh` 会**先杀循环**
> 再杀服务，否则循环会立刻把服务重新拉起来（这就是"杀了又活"的原因）。

**只放显存（不停服务）：**

```bash
curl -X POST http://127.0.0.1:8001/unload   # SAM3D（常驻 9~18GB）
curl -X POST http://127.0.0.1:8002/unload   # text2entity（常驻 14~28GB）
```

> 显存提醒：SAM3D 与 text2entity 都很大，24G 卡上同时加载会 OOM。
> 现在 `text2entity_generate` 在文生图前会**自动先卸掉 SAM3D**，用完再把它按需拉起。

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

**推荐（一个端口搞定）**：后端已经把前端挂到了 `/studio/`，直接访问

```
http://localhost:8000/studio/
```

这样只要转发/放行 8000 一个端口即可（VSCode 里转发 8000 → 浏览器打开上面地址）。

也可以像以前那样单独起静态服务器：

```bash
cd studio/frontend
python -m http.server 3000
```

然后访问 `http://localhost:3000`（前端里的 API 地址固定指向 `http://localhost:8000/api`）。

### 4. 一键启动（含 CARLA）

```bash
cd carla_bridge
./start_studio.sh            # 起 studio 后端 + CARLA server + 自检 Vulkan
./start_studio.sh --live     # 顺便开 CARLA 实时直播，前端 ⑥ 页面直接能看到画面
```

## CARLA 仿真（实验室 → ⑥ CARLA 仿真）

前端「实验室」里新增了第 ⑥ 个标签页，把 CARLA 的整套能力都接进来了：

| 区块 | 能做什么 |
| --- | --- |
| 服务状态 | 显示 CARLA 是否就绪（绿点）、当前地图；「启动 / 停止 / 刷新 / 环境自检」 |
| 1) 选场景 | 勾选**用当前编辑的场景**（含刚生成/编辑的事故轨迹，现场导出成世界坐标 json 再送 CARLA），或从下拉里选一个已导出的场景 |
| 2) 参数 | 地图、fps、相机（chase/birdseye/side/front）、分辨率、相机跟随（事故双方/自车/全部）、参与者筛选、天气、ego 模式（按轨迹回放 / CARLA 自己开=闭环） |
| 动作 | ▶ 开始 CARLA 仿真并出视频、🔴 开实时直播、只导出对齐版 xosc、用 ScenarioRunner 跑（逻辑）、取消作业 |
| 结果 | MP4 播放器（可拖进度条）、碰撞事件表、`帧数/actor 数/最小距离/最小 TTC` 指标、一键下载 mp4/events/frames/xosc |
| 实时 | 后端代理 CARLA 的 MJPEG 流，页面里直接看循环播放的仿真画面（不需要再转发 8090） |
| 历史 | 最近作业列表，可直接回看视频 |

**典型用法（"编辑 → 仿真 → 再看 → 再编辑"闭环）**：

1. 左侧生成 / 编辑事故轨迹（生成 Corner Case、拖动、改轨迹点…）；
2. 点 Corner Case 面板里的 **🎬 送进 CARLA 仿真**（或直接开「实验室 → ⑥ CARLA 仿真」）；
3. 保持勾选「用当前编辑的场景」，点 **▶ 开始 CARLA 仿真并出视频**；
4. 视频出来后看碰撞事件 / 最小距离 / TTC，不满意就回去继续改轨迹，再点一次。

> 依赖：需要 `carla_bridge/` 那套（CARLA 0.9.15 + py37 环境），见 `carla_bridge/README_CARLA.md`。
> 后端接口都在 `/api/carla/*`（`/docs` 里有交互式文档）。

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

### 面板（可折叠 / 分组 / 搜索）

左右两侧原来是**一路平铺**的 `.panel-section`，一屏全是标题和控件、容易看混。现在改成了
可折叠面板（`frontend/ui-panels.js` + `styles.css` 末尾一段）：

- **点击标题栏**（或聚焦后按 Enter/Space）收起 / 展开，右侧箭头指示状态，带高度过渡动画；
- 标题栏是 **sticky** 的：面板内容长了以后滚动时标题一直吸顶，不会"滚丢了"；
- **按功能分组**（左侧顺序也按组重排过）：

  | 分组 | 面板 |
  | --- | --- |
  | 场景与物体 | 场景信息、动态物体列表 |
  | 事故与轨迹 | 生成 Corner Case、语言编辑轨迹（LLM）、车头朝向 |
  | 模型生成 | SAM 3D 交互重建、文本添加实体 |
  | 主车视角 | 主车视角 |
  | 当前选择（右侧） | 选中物体、编辑历史 |

- **顶部工具条**：`搜索面板…`（输入即过滤，命中的面板自动展开）、`全部展开`、`全部收起`、
  **单开 ⇄ 多开**切换。默认**单开**（展开一个就收起同侧其它），这样一次只专注一个面板；
  想同时对照多个面板就点那个按钮切成多开。
- **收起时标题栏显示一句"里面有什么"**（如「语言编辑轨迹 · 自然语言改轨迹（LLM）」），
  收起后也不会找不着；鼠标悬停还有完整标题的 tooltip。
- 界面**不用 emoji**：折叠箭头与工具条按钮都是内联单色 SVG，其它按钮/状态文字一律纯文本。
- 展开状态记在 `localStorage`（键 `dggt.panels.v1`），刷新后保持。
- 选中物体时右侧「选中物体」会**自动展开**。
- 标题栏里原有的控件（"场景信息"的刷新、"动态物体列表"的搜索框）照常点，不会误触发折叠。

实现上**只包裹既有 DOM**（把 header 之后的节点搬进 `.panel-body`），不移动节点、不改 id，
所以 `app.js` 里所有 `getElementById` / `querySelector` 都不受影响；右侧面板复用同一套逻辑
（只是没有工具条）。想用代码控制面板也可以：

```js
DGGTPanels.open('语言编辑轨迹');   // 按标题模糊匹配或 panel key
DGGTPanels.close('场景信息');
DGGTPanels.toggle('生成 Corner Case');
DGGTPanels.expandAll(); DGGTPanels.collapseAll();
DGGTPanels.list();                 // 当前所有面板及展开状态
```

## 语言编辑轨迹 / 文本生成实体的几个修复（按报错）

### 1) `name 'recon_ext' is not defined`（已修）

文本生成实体（插入）路径量出来的三轴尺寸变量叫 **`ext`**，而替换路径叫 `recon_ext`；
重构"包围盒紧贴模型"时在插入路径误用了 `recon_ext`，于是走插入就 NameError：

```
失败：文本生成实体失败: name 'recon_ext' is not defined
```

已改成 `ext` 并删掉被 helper 覆盖的死代码。为防止同类问题复发，加了一个基于 `symtable`
的静态自检：检查每个函数里"被当成全局引用、但模块里并不存在"的名字——
本次就是这样把 `corner_case._create_pedestrian_track` 里**早就存在**的 `os` 未导入
（行人场景的错误分支会 NameError，把真正原因盖掉）也一起找出来修了。

### 2) 「生成一辆车去撞某车」——事故必须由**新车**引发

原来规则兜底会把"变道/相撞"挂到随便两辆既有车上，所以
`生成一辆对向来的红色轿车，T1转弯时和该红色轿车相撞` 会莫名其妙变成 **T0 和 T1 相撞**，
新生成的车反而跟事故无关（直线走、也没撞上）。

现在按下面的规则定角色：

- **有 `insert` + "撞/相撞"时，默认新车是肇事车**（它才是我们要控制的那辆），
  指令里写出来的那辆已有车是被撞方 → `collide(a=null, b=<那辆车>)`；
  新车的动作同样用占位（`track=null`），上层拿到新 id 后回填。
- 只有指令明确写成"已有车去撞新车"（`T:1 撞上这辆新车` / `T2 撞它`）才把角色反过来。
- **规则兜底不再去猜别的车的动作**：要生成新车时，变道/加速/转向一律挂在**新车**上，
  不会再动到没在指令里出现的车。
- 另外修正：`insert+crash` 时跳过"把没提到的车统统改挂到第一辆被提到的车"这一步——
  正是它把变道动作挂到了**被撞车**上。

> 注意 `T100000` 这类 id 是**存在**的：用 SAM3D 替换某个物体后，新物体从 10 万号开始
> （替换 T:1 → T:100000）。`_known()` 查的就是"这个 track 有没有轨迹"，
> 合成物体同样算数。

### 2.5) 生成车辆的尺寸 / 贴地 / 轨迹帧数

三个问题一起修（前两个是同一个根因）：

**根因**：`_fit_scale_and_dims` 在"扁片重建"分支里返回的**包围盒是目标尺寸**，而模型却按
最多 2.5 倍渲染 —— 两边不一致，于是模型看起来**比原车大一圈**；而贴地是按包围盒高度算的
（`位姿中心 = 地面 + 包围盒高/2`），模型和包围盒不等高 → **悬空/沉地**。

现在两条硬约束：

1. **包围盒恒等于渲染出来的模型**（`dims` 一律用紧贴模型的尺寸，绝不回退到别的值）；
2. **模型不超过目标包围盒**：缩放取 `min(按高度, 等比塞进目标)`：
   - `按高度 = 目标高/重建高`（你要的"只对齐高度、等比、不逐轴拉伸"）；
   - `等比塞进目标 = min_i(目标_i/重建_i)`，保证任何一轴都不超过原物体。
   `fit`（面板滑块 / `DGGT_SAM3D_FIT`）在此基础上再整体缩——因为包围盒跟着模型走，
   **缩小也不会悬空**（这也顺便治好了之前"0.62 就浮空"的那类问题）。

顺带两处也修了：

- `_ply_bbox_and_orientation` 原来返回**质心**当 center，而渲染/贴地都按"包围盒中心"推算；
  前脸高斯密、车尾疏时质心≠包围盒中心 → 模型整体偏上/偏下。现在统一用**可见轮廓包围盒中心**。
- `refine_vehicle_dims` 现在**以场景中位车辆为基准**（`blend` 只留 25% 给 VLM，并夹在
  0.75~1.12 倍以内；SUV/面包车给 1.06/1.10 的类别系数；卡车/巴士/公交不夹）。
  为什么改成"以场景为基准"而不是"和常识融合"：4DGS 场景里车辆的包围盒是按可见高斯拟合的
  （本场景中位 `[1.81, 1.13, 3.70]`），比真车小；只要目标高度给到常识的 1.45m，渲染出来就
  比场景里的车高一大截 —— 用户看到的就是"生成的车过大"。目标贴着场景车之后，生成车
  （本例紧贴模型 `[1.64, 1.22, 3.80]`）和场景车、"主车"就一个量级了。

**主车（ego）也一样"过大"**：`_init_ego_track` 以前**只按车长**缩放（`scale = 场景中位车长 / 模型长`），
模型的三维比例和 4DGS 拟合出来的包围盒比例不一样，按车长对齐后自车会**又高又宽**。现在改成
**等比塞进场景中位车的包围盒**（`scale = min_i(场景_i / 模型_i) × DGGT_EGO_FIT`），长宽高都
不超过场景车、比例不变（不变形）。本场景自车从 `[?, ?, 3.70]` 变成 `[1.53, 1.13, 3.60]`。

**贴地（悬空）**：

- 根因在 `plan_trajectory` 的兜底：静态点覆盖不到的远端 `_ground_height_near` 会返回 `None`，
  旧代码退化成 `gy = 自车轨迹点的 y` —— 而那是**自车车体中心**的高度（比路面高半个车高），
  于是生成的车**浮空约 0.53m**。现在：
  1. `_ground_height_near` **逐级放大邻域**（6→12→24→48→96m），只要视野里还有路面点就给出估计；
  2. 真·覆盖外时，兜底改成 `自车车底高度`（`p[1] - up×自车高/2`），而不是车体中心；
  3. 逐帧地面高度再夹在"全局路面中位 ±1.2m"内，并做 5 帧滑动平均，去掉 ±5cm 抖动。
- 事故仿真/兜底摆放沿用了"锁定首帧高度"的写法，也会整体浮空 → `_op_collide` 在**所有改动之后**
  统一贴地（`_reground_track`，只改 y、保留 x/z 与朝向）+ 高度平滑（`_smooth_track_heights`），
  并且**只对合成物体**这么做（真实轨迹的 y 是数据本身，强行贴地会按估计误差上下拽）。
- 贴地会改变"高度方向"的相对关系（原本靠高度错开的两车贴回同一路面后可能在 XZ 上重叠），
  所以贴完**必再跑一次** `separate_pair` 逐帧防穿模。

**轨迹连续性（"位置莫名其妙"）**：

- `_op_collide` 原来把 `frames = 攻守双方的帧交集`（真实轨迹常常缺帧，例如 T:1 只有
  `0,3,11..19,25..39`）喂给"按固定 dt 逐步推进"的追击仿真 —— 连续的物理时间被写到跳跃的
  帧号上、没写到的帧保留旧位姿，同一条轨迹出现**几段互不相接、来回跳的残影**。
  现在窗口一律取**连续帧** `[max(lo), min(hi)]`。
- 追击瞄点从"受害车**当前**位置"改成"**拦截点**"（迭代求它接下来会到的地方）：对向/侧向
  碰撞时朝当前位置追会让肇事车先掉头（转向率上限 60°/s → 掉头要 3s），窗口内撞不上。
- 兜底 `_contact_place` 是"先算间距、再对朝向"的，朝向变化会改变包围盒 footprint，可能仍然
  压在一起（实测 `contact_overlap=true`）→ 兜底之后也跑一次 `separate_pair`。
- **撞不上的时候要如实说，而不是硬造一条轨迹**：仿真没撞上时，先量"两车**原本**的轨迹在窗口内
  最近能到多近"。很近（≤ `0.5×(车长和)+12m`）→ 兜底推到"刚好接触"；很远 → **回滚仿真改动**、
  保留原轨迹，并回 `ok=false` + 原因（"窗口内撞不上 T:X：两车原本的轨迹最近也有 65m…"）。
  之前这种情况会硬把肇事车挪到受害车旁边，造出一条几十米的长途追车轨迹（"位置莫名其妙"）。
  前端状态行现在也会显示这类"操作未生效"的原因。

> 主车尺寸可用环境变量回退：`DGGT_EGO_FIT_MODE=length`（旧行为：只按车长对齐）、
> `DGGT_EGO_FIT=1.1`（整体再放大 10%）；生成车辆的尺寸/贴地相关的还有
> `DGGT_SAM3D_FIT`、`DGGT_CAR_FALLBACK_PLY`、`DGGT_MAX_INSERT_FRAMES`。

**"轨迹跑到场景外"会明确提示**：静态点只覆盖相机看过的地方（本场景 `z ≥ 3.9m`），
生成物跑到自车起点后方就既没有路面点也没有背景。`plan_trajectory` 会统计这类帧数
（`ground_fallback_frames`），响应里给 `warnings`（前端状态行会显示）：
"生成轨迹里有 N/M 帧超出了 4DGS 静态场景的重建范围…建议减少帧数/降低速度"。

**自然语言指定"生成多少帧"**：

- LLM 提示词里明确要求把"多少帧 / 几秒"写到 `insert.num_frames`（1 秒 ≈ 10 帧）；
- 但 LLM 经常漏写（实测有 `insert` 里根本没有 `num_frames` 的情况）→ 新增
  `fill_insert_frames()`：规划完成后统一补 `insert.num_frames`，优先级
  `op 里已有的` > `指令里解析出来的` > `面板/请求的 num_frames`；同时把 `collide.frame`
  夹进"生成物体实际存在的帧范围"（避免"第 40 帧相撞"但物体只到第 39 帧）；
- 规则兜底也有解析器 `parse_num_frames()`：`40帧` / `4秒`（→40） / `持续 40`（无单位按帧）；
- 生成时夹到 `2 ~ DGGT_MAX_INSERT_FRAMES`（默认 600）帧；
- 如果帧数超出原场景长度，`traj_edit/apply` 会**先按需延长时间轴**再生成，所以
  "生成一辆车跑 100 帧"在只有 25 帧的场景里也能跑；
- 回执里有 `num_frames`（新物体的轨迹帧数），面板「语言编辑轨迹」里可以直接填默认帧数
  （`#trajEditNumFrames`，指令里写了帧数则以指令为准），插入回执里还会显示
  `包围盒 / 目标 / 帧数 / 框紧贴模型 / 重建退化→已换通用车模`。

### 2.6) 事故窗口三段式 / 不动受害车轨迹 / 顶部帧数拖动条

**① 40 帧的事故 = 撞前 + 撞上 + 撞后**

- 撞击帧不再听 LLM 随便给的那个数：`collide.frame` 会被夹到**窗口 75% 以内**
  （`fill_insert_frames` 里就夹，规划回执里看到的就是修正后的值），
  没给期望帧时默认放 **60%**（40 帧 → 第 24 帧撞上，后 16 帧是撞后）。
- 追击速度按"期望撞击帧"反推：`need = 中心距 / 期望时长 + 受害车速度`，并允许肇事车为了
  "晚一点撞"而松油门/刹车（旧写法速度下界是初始速度，永远降不下来）。**不要**在这里减
  "两车半长"——斜碰时支撑距离远小于半长之和，减了会把"其实还没贴上"的两车变成 0.3m/s 的爬行。
- 速度会**迭代**：跑一次看实测撞击帧，偏差 >2 帧就按 `实测接近时间/期望接近时间` 比例修正
  再跑（最多 3 次，并回放"最接近期望帧且真的撞上"的那一次，避免改过头）。
- 如果仿真撞上的帧离期望超过**窗口的 15%**（且至少 3 帧），就**不硬用**仿真结果：回滚，
  改走确定性的兜底摆放（把"刚好接触"精确放在期望帧上，受害车同样一动不动）。
- 回执新增 `impact_frame_target` / `post_impact_frames` / `window` / `impact_attempts` / `fallback`。
  实测：期望第 20 帧 → 实际第 20 帧（`fallback=contact_place`，`post_impact_frames=19`）。
- 受害车轨迹"可信度"预检：相邻帧最高速度超过 `DGGT_MAX_TRACK_SPEED`（默认 60m/s，
  本场景 T:16/T:13/T:9 高达 300+ m/s，是误检测）就**如实拒绝**碰撞并说明原因，
  而不是用一条"瞬移"的轨迹算出一场飞车事故。
- 提示词里把 `mode` 的示例从 `"ahead|oncoming|roadside"` 改成 `"ahead"`（LLM 会把这个
  占位串**原样抄下来**，实测出现过 `mode:"ahead|oncoming|roadside"`），并在 `sanitize_ops`
  里把非法 mode 归一化成 `ahead` + 一条 warning。

**② 受害车不再被拉直（转弯的车继续转弯）**

旧的事故仿真用"受害车首帧的朝向 + 速度"重新积分它的运动 → 一辆正在转弯的车被撞，
它的整条轨迹会被拉成直线（用户看到的就是"路线突然变直，很奇怪"）。

现在 `simulate_pair_collision(..., victim_follow_track=True)`（语言编辑轨迹固定用这个模式，
corner case 的参数化场景仍可用 False）：

- 受害车**逐帧沿它自己的轨迹**走（含转弯、含它自己的高度），仿真只在撞上的那一刻把
  **冲量**叠加成"额外位移"（摩擦衰减），撞后它继续按自己的路线开；
- 实测：受害车在撞前逐帧位移 = **0.00m**（bit 级不动），撞后只多出 0.4m 的冲量位移，
  朝向继续按自己的转角变化（2.1°→20.7° 一路右转）；
- 肇事车撞后摩擦减速到停（实测第 22 帧撞上，第 36 帧停住），受害车继续走 → 三段齐全；
- 兜底摆放（`contact_place`）也遵守这条：只移动肇事车，`frame` 之后把间距收到 2cm
  "顶上去"，受害车一动不动（只可能被防穿模分离推开几厘米）。
- 受害车的"自身速度"做了 **3 帧中值滤波 + ≤30m/s 限幅**，冲量也限幅（肇事车单次撞击
  速度增量 ≤8m/s、受害车 ≤12m/s）：真实轨迹经常某一帧位置跳一下（实测 T:13 单帧跳
  6.8m = 68m/s），不限制的话冲量项会爆掉（把肇事车打飞、甚至甩到天上）。
- 贴地改成分级：`_dense_ground_y` 只在 6~12m 邻域里**有 ≥30 个静态点**（= 真的覆盖到
  这里）时才贴，覆盖外保持原高度 —— 之前逐级放大到几十米会拿"远处路面"来贴，
  轨迹跑到场景外时车会被抬到空中 1m+。

**③ 顶部帧数改成可拖动条**

工具栏新增 `#frameSlider`（`frame-controls` 里，数字输入右边）：拖动即时更新帧号，
渲染**节流到 ~150ms 一次**（拖一次不会发几百个渲染请求，松手立即渲染一帧）；
和数字输入、上一帧/下一帧、播放、跳关键帧全部双向同步。

> 注意：`collide` 现在**不会**再去改写受害车的轨迹，所以"谁被撞"只能靠肇事车去够；
> 如果两条轨迹在这个窗口里最近也有十几米（对方跑掉了），会**如实回 `ok=false`**
> 并保留原轨迹，而不是硬造一条长途追车轨迹（详见上面的连续性一节）。

### 2.7) 自然语言总控：一句话驱动 编辑 / 生成 / 事故 / 时间轴

入口还是「自然语言总控」面板（`POST /api/traj_edit/plan|apply`），但能识别的操作远不止轨迹编辑：

| 操作 | 示例 | 复用的既有功能 |
|---|---|---|
| speed / lane_change / turn / remove / collide | "让 T:4 减速到 4m/s，然后和 T:2 相撞" | `traj_llm` 里同一批 `_op_*` + `fuse_collide_ops` |
| insert（生成车辆） | "生成一辆与 T:12 同向行驶的白色货车变道撞击 T:12，持续 30 帧" | LLaDA 出图 + SAM3D 重建 + 相对轨迹规划 |
| scale | "把 T:100000 放大到 1.3 倍" | `/api/edit/synthetic/scale` 的同一套逻辑 |
| shadow | "给 T:100000 去掉阴影" | 接触阴影开关 |
| replan | "把 T:100000 的轨迹重排到前方 16m" | `/api/text2entity/replan` 的同一套逻辑 |
| corner_case | "生成一个 T:5 追尾 T:2 的事故，20 帧" | `corner_case.generate`（同一套事故引擎） |
| extend | "把时间轴延长 50 帧" | `tm.extend_timeline` |
| undo / redo | "撤销" / "重做" | `tm.undo/redo` |

**"与 T<id> 同向/对向"不再搞反（这次的 bug）**：

1. `scene_summary` 现在带 `yaw`（世界运动方向）和 `vs_ego`（相对主车同向/对向）——
   旧摘要只有 id/type/x/z/dims，**没有任何朝向信息**，LLM 只能瞎猜（实测就把"与 T100000 同向"
   生成成了对向车）；
2. 提示词里写明"方向必须看 yaw"，并要求填 `direction_ref / same_dir / lane_ref / lane_offset`；
3. 规划完之后 `_inject_insert_hints()` 用**指令原文**兜底：正则解析"与 T<id> 同向/对向"、
   "在 T<id> 左边/右侧"、以及"变道撞击"该配的 `lane_change`。LLM 填错（实测把
   `direction_ref` 填成了主车 900000）或漏填都会被按指令改回来并回一条 warning；
4. 轨迹不再靠"相对自车车道"的 ahead/oncoming 猜：新增
   `nl_entity.plan_relative_trajectory()` —— **以参照车自己的轨迹为基准**铺路线：
   同向 = 跟在它后面把间距从 `distance` 收到"刚好接触"，对向 = 从它前方迎上来；
   横向 = 参照车右向量 × `lane_offset`（±3.5 就是旁边一条车道）。方向/车道/转弯几何自动一致；
5. 参照车轨迹抖得厉害时（原始折线长度 ÷ 平滑路线长度 > 1.6，实测误检轨迹能到 3+），先用
   `_smooth_route()` 整理成"单调、平滑的路线"再铺（否则新物体位置会来回摆十几米），
   并在回执里给 warning 说明；
6. "变道撞击" = `insert(lane_offset=±3.5)` + 自动补的
   `lane_change(track=null, lateral=-lane_offset)`，后者会被 `fuse_collide_ops` 融进同一次事故仿真，
   即"一边切进来一边撞"，不会先变道再撞两段式地互相覆盖；
7. 实测（干净参照车 = 主车）：同向 → 新物体朝向 -2.7°（参照 -2.6°），车距 8.95m→6.33m；
   对向 → 朝向 -179.1°（正好相反），车距 29.4m→**0.76m（第 18 帧精确接触）**。

其它：
- `replace`（SAM3D 替换）仍需要**点选/框选目标**，自然语言里会明确回
  "这类操作需要在对应面板里做"；`render/export` 同理（需要参数）。
- 顺手修了一个会让替换失败的坑：`/api/sam3d/replace` 在重建前先让 SAM3D 微服务
  `/unload`（上一次重建会常驻 9~18GB，实测直接再重建会 `CUDA OOM: Tried to allocate 1.31 GiB`）。
- 事故场景名从 `corner_case.SCENARIOS` 现取后写进提示词（含每个场景需要的 roles key），
  不再手写（手写会写错名字，例如把 `head-on` 写成 `oncoming-collision`）。

### 3) 语言编辑里生成新物体时，能看到 / 指定参考图

界面「自然语言总控」面板（原「语言编辑轨迹（LLM）」）新增：

- **参考图（可选）**：选一张图就直接用它做 SAM3D 重建（跳过文生图），
  和「文本添加实体」面板一致；不选则按描述由 LLaDA 出图。
- 生成完成后，面板里会显示**实际用于重建的参考图**和**插入效果预览**，
  以及新物体 id / 尺寸——避免"生成出来一个木头集装箱还不知道为什么"。

后端：`POST /api/traj_edit/apply` 新增可选 `reference_image`，
响应新增 `inserts: [{object_id, prompt, mode, dimensions, bbox_mode, reference_image, preview}]`。

### 4) 没点名任何存在的车时，不再乱改别人的轨迹

`生成一辆对向来的红色轿车在T100000转弯时相撞` 里 **T100000 并不存在**。规则兜底会把
"变道/相撞/转向"挂到随便一辆车上（实测挂到了 T:1 / T:2），等于顺手改了别的车。

现在：**指令要生成新车、却没有指名任何"存在的"车时，把 speed/lane_change/turn/collide
全部丢掉，只保留 insert**，并在回执里给 `warnings` 说明原因；前端在状态行与预览里都会显示
（`_trajEditStatus` / `_trajEditPreview` 现在带 `warnings`）。

### 5) 主车不能被动删掉

LLM 偶尔会给出 `remove 900000`（删主车）。一旦执行主车/相机都没了。现在只有当指令**明确**
提到"主车/自车/本车/ego"时才允许删它，否则忽略并记一条 warning。

### 6) 规则兜底的出图提示词

以前整句指令会被原样丢给文生图（"生成一辆对向来的红色轿车在T100000转弯时相撞"），
出图必然跑偏。现在 `rule_insert_prompt()` 只保留物体描述（→"一辆对向来的红色轿车"），
并按"对向/迎面/逆行"自动选 `mode=oncoming`。

### 7) 排查这类问题时注意 GPU 占用

`/llm` 报 `VLM 加载失败: CUDA out of memory`、插入跑到一半失败，通常是**显存被别的服务占着**：

- SAM3D 微服务做完一次重建会常驻约 **9~18GB**，直到 `/unload`（studio 正常流程会在结束时调，
  但请求被中断就会一直占着）→ `curl -X POST http://127.0.0.1:8001/unload`；
  **已加自动处理**：`text2entity_generate` 在真的要文生图之前会先给 SAM3D 发 `/unload`
  腾出显存（用参考图跳过文生图时不折腾）。
- text2entity 微服务常驻 LLaDA 约 17–28GB → `POST http://127.0.0.1:8002/unload`；
- studio 自己：**每 `POST /api/scenes/load` 一次就常驻一份 4DGS 场景**，
  反复加载会把显存吃光。批量/脚本测试请用完 `DELETE /api/scenes/{scene_id}` 卸载。

## 可扩展时间轴（原场景帧数不再是上限）

4DGS 场景里**静态场景是单份高斯**（`gaussians/static_scene.ply`，与帧无关），只有两部分是
"逐帧"的：`ego_pose/frame_XXXX_ego.json`（相机）与 `gaussians/frame_XXXX_dynamic.ply`
（动态物体外观）。所以原场景多少帧**不该**是上限——把这三者拆开之后就能按需要多少帧给多少帧：

| 组成 | 超范围帧怎么办 |
| --- | --- |
| 静态场景 | 单份高斯，任何帧都能渲 ✓ |
| 动态物体**外观** | 复用"该物体最后一次出现"的那份高斯（`_obj_appearance` 缓存） |
| 动态物体**位姿** | 由轨迹给：延长时按各自末尾速度/角速度外推（`track_extended`），之后可继续编辑 |
| 相机 | 由自车轨迹外推（`camera_provider` → `get_ego_camera`）；或固定在末帧 |

接口：

- `GET /api/timeline/{scene_id}` → `{data_frames, total_frames, extended_frames, extended_from_frame, fps, max_frames}`
- `POST /api/timeline/extend` `{scene_id, extra_frames|total_frames, mode, ego_mode}`
  - `mode`：`extrapolate`（默认，沿各自轨迹惯性外推）/ `hold`（冻结在末帧）
  - `ego_mode`：单独控制相机（`hold` = 相机固定，物体继续动）
- **按需自动延长**：`/api/render/frame`、`/api/render/sequence`、`/api/render/freeview`、
  `/api/traj_edit/apply`、`/api/corner_case/generate` 在请求的范围超出原帧数时会自动延长
  （渲染类请求可用 `auto_extend: false` 关掉）。上限 `DGGT_MAX_FRAMES`（默认 4000）。

前端在「场景信息」面板里：显示 `数据帧 / 可渲染`，一个「延长」输入 + `+50` / `+200`
快捷按钮，以及「相机固定在末帧（物体继续动）」勾选项。

### 外推有多平滑 / 画质怎么取舍

实测（`waymo_eval/001`，10fps，数据 25 帧延长到 90 帧）：

- **平滑性**：自车数据段末尾步长 1.06/0.98/1.02 m，外推段为**恒定 1.057 m/帧**，
  接缝处没有跳变；各 track 外推段步长抖动 **0.0000**（常速度延续）。
- **帧数不再受限**：在原场景之外做语言轨迹编辑（70 帧窗口）能自然撞上
  （collision_frame=30，最大穿透 0.0 m）；`corner_case.generate` 用
  `start_frame=50` 也能正常生成（碰撞帧 59）；轨迹帧范围变成 0..89。
- **画质与视角有关**：相机跟着自车往前外推时，走到原轨迹覆盖之外会逐渐出现
  4DGS 的拉伸/涂抹（这是重建本身的覆盖范围问题，不是延长导致的；实测末段数据帧
  本身清晰度就已下降）。把相机固定在末帧（`ego_mode=hold`）时背景保持清晰：
  实测外推 40 帧内清晰度 134/131/128，甚至高于末段数据帧的 110，而且物体照常继续运动。
  → **要"车继续往前开"就跟着外推；要背景稳、只看物体动作就勾"相机固定在末帧"。**

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

## 实验室流水线里的几个关键约定（都是踩坑后定的）

### ① 挑事故 → ② 批量生成 → ③ 闭环评估 → ④ CARLA 验证

**批量生成必须是第一步。** 批量会先把 TrackManager 重置成干净场景（如果场景里有手动生成/编辑的残留），
否则每个事故类型都从同一个被污染的状态出发，生成出来的实例/视频会**一模一样**。

批量把每个实例落成 `<output>/corner_cases/<scene_id>/<run_name>/<type>-<idx>-<tid>.world.json`，
并记录 `start_frame / num_frames / fps / num_tracks`。后面 ③/④ 都用**这个 json**（不依赖内存里的场景），
所以后端重启也不影响复现。

### ③ 与 ④ 的关系

③ 是"闭环评估"（用生成的事故跑到碰撞/指标），④ 是"CARLA 验证"（把同一条事故放进 CARLA 里出画面/视频）。
两者是**并列的两种验证**：③ 不必须先跑，④ 也不依赖 ③ 的产物；只要 ② 里有实例（或当前场景有事故轨迹），
④ 就能跑。界面上从 ③ 点"送去 CARLA 验证"只是把该实例的 `world.json` **预选**到 ④ 并跳到 ④ 页。

④ 页顶部有一行 **"将跑：…"**，会明确写出这次到底跑哪个场景 / 输出名 / 贴地模式 / ego 模式——
以前这个不显示，加上场景列表是异步刷新的，会出现"点了某条实例，结果还是跑默认场景"的假象
（现在会把选中的场景粘住：`_carlaPendingScenario`）。

### json 还是 xosc？

CARLA 渲染/仿真实际吃的是 **`.world.json`**（轨迹 + 位姿）。`.xosc` 只有勾选
"顺带导出对齐版 xosc"（或调用 `/api/carla/export_xosc`）时才生成，是喂给
ScenarioRunner 等标准工具做 OpenSCENARIO 校验用的，**CARLA 渲染不用它**。

### 扩散精修：单帧 vs 整段

* `✨ 对当前帧做扩散精修` → `/api/diffusion/refine_frame`，出一张前后对比图。
* `🎞 精修所有帧（出一段视频）` → `/api/diffusion/refine_sequence`，后台逐帧精修，
  用 `/api/diffusion/refine_job/{id}` 轮询进度，完成后返回 `refined.mp4`（精修后）和
  `original.mp4`（原始，对比用），产物在 `output/diffusion_seq/<scene_id 前 8 位>_<时间戳>/`。
  首帧要把 difix + sd-turbo 加载进显存（约 1~3 分钟），之后每帧几十秒。

### 贴地模式（把车放回路面）

④ 页"贴地"选择：`z`（默认，只改高度 → 不飘，且几何完全忠于原轨迹，**做评测用这个**）/
`lane`（吸到行车道中心 → 画面规整好看，但会改动事故相对位置）/ `none`。
对应桥接的 `--road-snap`，细节与实测数据见 `carla_bridge/README_CARLA.md` 第 8 节。

### 显存策略

CARLA 一开约 5~6 GB，studio 后端按需加载 difix（约 5~7 GB）。为了不把用户的 SAM3D 挤爆：
CARLA **不会自动启动**，空闲 `CARLA_AUTO_RELEASE_SECONDS`（默认 60 s）后自动释放；
"启动 CARLA / 起引擎 / 提交作业"期间会被标记为 **busy**（`_busy_begin/_busy_end`）并给 120 s 宽限期，
不会出现"正起引擎时 CARLA 被释放"的竞态（这个竞态会让 `load_world` 永远卡住）。
④ 页有显存徽标和「🧹 释放显存」按钮。

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

## 主车自动贴地（逐场景）

不同 4DGS 场景里自车相机的安装高度差别很大（实测约 1.1m ~ 1.9m），而"相机→车体中心"
的固定偏移（`EGO_CAM_OFFSET` 的 y=1.15）是按某一种相机高度定的，于是同一套偏移会让主车在
部分场景里**沉进路面**（实测最深约 1.2m）。

现在 `TrackManager._init_ego_track` 会**自动按场景地面贴地**：

- 从静态场景高斯估计自车路径下方的路面高度（水平邻域半径 `DGGT_EGO_GROUND_RADIUS`，默认 6m；
  取 y 分位数，自动识别世界 +y 是"上"还是"下"）；
- 令车体中心 = 地面 + 半个车高，逐帧放到地面上（对世界 y 平滑，相机外参偶发跳变也不会带偏）；
- 结果是车底与地面误差 < 3cm（旧偏移下平均穿模 0.3~1.2m）。
- 面板「主车视角」新增 **「自动贴地」** 按钮，可随时重算；接口 `POST /api/ego/ground`
  （`auto:true` 重算 / `auto:false` 还原为贴地前轨迹）。
- 可用环境变量调整/关闭：`DGGT_EGO_AUTO_GROUND=0`、`DGGT_EGO_GROUND_CLEARANCE`（车底离地余量，米，默认 0）、
  `DGGT_EGO_GROUND_RADIUS`。

`GET /api/ego/{scene_id}` 会额外返回 `auto_ground` 与 `ground`（估计出的地面高度、相机高度等）。

## 文本添加实体（LLaDA-Image + SAM3D + VLM）

左栏新增「**文本添加实体**」面板：输入一句描述（如"一辆红色轿车"），选放置方式
（自车前方同向 / 对向车道 / 路边停放）、起始距离、速度、帧数，点「生成并插入」即可。

流水线：**LLaDA-Image-Turbo** 按文本出 N 张白底参考图 → 自动掩码 → **Qwen2.5-VL-3B**
用 `matches_prompt` 选图并推理物体类别与真实尺寸（长/宽/高）→ **SAM3D** 单图重建 .ply →
沿"自车轨迹 = 车道中心线"生成轨迹，并用 2D OBB(SAT) 对所有已有物体逐帧做冲突检测，
有冲突自动挪位，最后作为一等公民合成物体插入（可选中、可编辑、可参与事故）。

- 独立微服务 + 独立环境（见 `text2entity/README.md`）：`bash text2entity/setup_env.sh`
  安装，`nohup bash text2entity/run_service.sh &` 启动（端口 8002）。
- 24G 单卡上三个大模型**串行**加载（每步用完 `/unload`），避免显存叠加 OOM。
- 接口：`GET /api/text2entity/health`、`POST /api/text2entity/generate`、`POST /api/traj_edit/{plan,apply}`。
- 面板里若显示"微服务未启动"，先跑一次 `run_service.sh`。

### 尺寸基准与逐物体缩放

4DGS 场景里重建出的原始包围盒普遍比真实车辆**更扁平更宽**（尤其高度被压扁），
直接把 SAM3D 模型逐个轴向对齐会把车**拉伸变形**。因此：

- 只对齐**高度**、且**等比缩放**（不逐轴拉伸），保留重建模型正常的长宽高比例。
- 基准比例 `DGGT_SAM3D_FIT`，默认 **1.0**：`scale = 目标高 / 重建高`，即"按高度套进去"，
  不再额外缩小。
  > 曾经把它默认成 0.62（当时觉得更协调），但反馈是**车明显变小并浮在空中**——缩小后
  > 模型底面离开包围盒底面，视觉上就飘起来了，所以已回退到 1.0。要更小就调环境变量
  > `DGGT_SAM3D_FIT`，或选中物体后用「模型大小」滑块（等比，连带包围盒）。
- 同理自车尺寸改为**参照场景内已有车辆**（`_scene_vehicle_length` × `DGGT_EGO_FIT`，
  默认 1.0，夹在 1.5~5.2m），不再用固定偏大的包围盒。
- 选中任意物体后，面板「模型大小」滑块（0.5×~1.5×）可**等比**放大缩小该物体
  及其包围盒，走 `POST /api/edit/synthetic/scale`。

### 接触阴影

SAM3D 重建的 `.ply` 高斯**没有烘焙阴影**，插进场景后会显得"发飘"。`dggt_engine.py` 的
`_contact_shadow()` 会在物体底部用确定性采样（固定种子，避免逐帧闪烁）生成一圈扁平暗高斯，
拼成椭圆接触阴影；合成物体、SAM3D 替换物体与自车都会自动带上。

- 逐物体开关：`POST /api/edit/synthetic/shadow` `{scene_id, track_id, enabled}`。
- 全局关闭：给 extra object 传 `"shadow": false`（或在该 track 的 spec 里设 `shadow=False`），
  默认为 `True`。
- **阴影贴的是"估计出的真实路面"，不是包围盒底面**：4DGS 重建的包围盒在高度方向普遍不准
  （偏扁、还可能整体悬浮），照包围盒底面画会飘起来。`TrackManager.ground_y_at(x, z)`
  复用主车贴地那套静态高斯分位数估计，把 `ground_y` 传给 `_contact_shadow()`。
- 还踩过一个坑：`build_extra_objects` 传进去的是 **CUDA tensor**，而 `_contact_shadow`
  里用 `np.asarray(pose)` 会直接 `TypeError`，异常又被外层 `except` 悄悄吞掉——结果是
  "阴影从来就没画出来而且毫无提示"。现在 tensor/numpy 都支持，异常也会打印。

### 包围盒紧贴模型（碰撞判定才可信）

替换/插入 SAM3D 模型后，包围盒**不再沿用原物体那个 4DGS 包围盒**，而是按重建模型自己的
长宽高算（`_tight_model_dims`，`bbox_mode: tight_to_model`）。

为什么必须这样：4DGS 的原始包围盒在高度方向不可靠（偏扁、还可能整体悬浮），而且和 SAM3D
重建出来的车根本不是一回事。沿用旧尺寸会出现**"模型其实没碰到、过大的框先碰到了"**
——corner case 与语言轨迹编辑都走 OBB-SAT，于是被判成发生了事故。

换算与渲染顺序严格一致（`dggt_engine` 里是先 `model_corr`、再在物体局部系里乘缩放）：

```
局部包围盒[宽,高,长] = (|model_corr| @ canonical_ext) * 缩放
```

`.ply` 的 canonical 轴是 x=右(宽)、y=前(长)、z=上(高)，`model_corr`（`SAM3D_MODEL_CORR`）
把它旋到物体局部系 X=宽 / Y=高 / Z=长，所以典型情况下就是
`[ext_x, ext_z, ext_y] × scale`。每轴有 10cm 下限，避免个别退化重建（例如只有几 cm 厚）
让碰撞体退化成"纸片"而永远算不出碰撞。响应里同时给 `dimensions`（紧贴模型）与
`target_dims`（原物体，仅作参考）。

**扁片重建保护**：SAM3D 对正面/背面视角、或画面里太小的目标，偶尔会输出一个近似**扁片**
的结果。实测 `recon_ext=[1.024, 1.016, 0.067]`（最短轴/最长轴 = 0.065），按"目标高/重建高"
会放大 **15.6 倍**，变成 16m × 1m × 15.8m 的薄饼。`_fit_scale_and_dims()` 会：

- 给缩放置顶（最长轴不超过目标最长轴的 2.5 倍）——实测 15.6 倍被限到 4.4 倍；
- 包围盒回退到原物体尺寸（`bbox_mode: fallback_target`），因为按扁片算"紧贴"没有意义；
- 在 `recon_quality` 里返回 `suspect: true` + `reason` + `recon_flat_ratio`，
  提示"换更清晰的目标或上传参考图重做"。

正常重建仍走紧贴模型分支（`bbox_mode: tight_to_model`）。

顺带修掉一个会让判定**误报**的 SAT bug：`BoundingBox.get_axes()` 返回的是旋转矩阵，
而调用方 `axes.extend(...)` 会把它按**行**展开——但盒子的三个主轴是旋转矩阵的**列**
（角点是 `R @ corners_local.T`）。在 4000 组随机位姿上实测，按行展开会有约 **0.5%**
的假碰撞（真实已分离却判成碰撞），按列则与参考实现完全一致。

## 语言驱动轨迹编辑（LLM → 结构化操作）

左栏「**语言编辑轨迹（LLM）**」面板：直接用一句自然语言描述想怎么改，
例如「让 T:1 减速到 4m/s，让 T:2 向左变道，删掉 T:3」「生成一辆红色轿车向左变道撞上 T:2」。
点「解析」只做规划（`/api/traj_edit/plan`），点「应用」会真正落到场景（`/api/traj_edit/apply`）。

> 说明：目前**没有现成的"语言驱动轨迹编辑"训练模型**，所以这里用
> **LLM 做意图解析 + 确定性几何规则执行**的方案：`traj_llm.py` 把自然语言交给微服务的
> `POST /llm`（Qwen2.5-VL-3B）转成结构化 ops JSON；LLM 不可用时退化为关键词规则解析。

支持的 op：

| op | 作用 | 关键参数 |
| --- | --- | --- |
| `speed` | 变速（保持原路径形态，按 k 重定时） | `track`、`speed_mps` |
| `lane_change` | 平滑变道 | `track`、`lateral`（车体右侧为正）、`duration` |
| `turn` | 转向（有高精地图时**沿车道几何在路口真转弯**） | `track`、`direction`（`left`/`right`/`straight`）、`yaw_deg`（正=左转，作为兜底）、`duration`、`forward_m` |
| `remove` | 删除车辆 | `track` |
| `collide` | 碰撞编排 | `a`（肇事车）、`b`（被撞车）、`frame` |
| `insert` | 生成新车（复用 LLaDA+SAM3D） | `prompt`、`mode`、`distance`、`speed`、`num_frames` |

**碰撞编排与"生成 corner case"共用同一套逻辑**：`collide` 不再自己"把车拽到目标点"，
而是直接调用 corner case 用的那个仿真入口
`corner_case.simulate_pair_collision()`（corner case 的 `prim_pursue` 走的是同一个
`_simulate_pursuit_collision`）。一次调用完成：

1. **限加速度追击**：`_step_toward_2d` 把肇事车的速度按 ≤(4+3×intensity) m/s² 加速、
   ≤8 m/s² 制动、≤60°/s 转向地逼近受害车（速度剖面连续，不会瞬移）；
2. **接触判定**：逐帧用 `dggt.scene_edit.collision_physics.check_collision`（15 轴 OBB-SAT）
   判断是否真的碰上，**用的就是各车当前的包围盒**；
3. **非弹性碰撞响应**：按动量守恒分配冲量（restitution 0.15），受害车被撞开、肇事车减速；
4. **撞后摩擦减速**：双方按 7 m/s² 逐渐停下；
5. **逐帧防穿模**：`separate_pair()` 在碰撞帧之后**每一帧**都做一次分离——
   沿 SAT 的最小平移向量（MTV）把两车推开到刚好接触（按质量分配位移，重的挪得少），
   只平移不改朝向。首次碰撞之后肇事车仍可能比受害车快、再次贴上，所以这一步是逐帧的，
   不是只处理第一次接触。

两条路径因此**共用同一套碰撞判定与撞后行为**：先用"生成 corner case"造事故、再用自然语言
改同一条轨迹（或反过来）不会出现"一个说撞了、一个说没撞"或互相覆盖的问题。

**重复应用是幂等的**：`_op_collide` 会记下这对车"事故前/事故后"的逐帧位姿。再点一次"应用"
时，如果两车轨迹还是上次事故后的样子（说明用户没手动改过），就先还原到事故前再重放，
结果与第一次**完全一致**（实测逐帧差 0.000000 m）；如果中间被手动编辑过，则从当前状态
重新仿真（那才是"用新状态再撞一次"的正确语义）。

**变道 + 撞一次成型**：一条指令常同时给 `lane_change` 和 `collide`（"向左变道后撞上 T:x"）。
如果按顺序各做一次，追击仿真会把刚写好的横移整段覆盖掉（corner case 里同样的坑，见
`scenario_engine.plan_lane_change`）。所以 `fuse_collide_ops()` 会把横移量作为仿真内部的
`attacker_lateral` / `victim_lateral` 传进去，并把被融合的 op 标记为"已融合、不再单独执行"。

**窗口内撞不上时**（例如受害车跑掉了、窗口太短）不会硬造重叠：改为把肇事车平滑推到
"**刚好接触**"的位置（用 OBB 支撑距离算间距），回执里给 `fallback: contact_place`
与 `contact_overlap`（实测为 `false`，即包围盒不重叠）。

**LLM 误解析兜底（`sanitize_ops`）**：LLM 偶尔会自作主张或指错车（比如只是"让 A 撞 B"，
它却顺手生成一辆新车、或把被撞的车删掉、或把 a/b 写反）。因为用户自己写的 `T:id` 最可信，
`sanitize_ops` 会用指令里的**显式 id** 做确定性纠正：

1. 指令里没有"生成/添加/新增/放入"等词 → 丢掉 `insert`；被其它 op 引用的车不允许 `remove`
   （多为幻觉），且 `remove` 一律排到最后；
2. 按"`撞`字之前提到的车是肇事车、之后提到的是被撞车"重排 `collide.a/b`；
   `主车/自车/本车/ego` 映射到 id **900000**（也会写进给 LLM 的场景 JSON）；
3. 「生成一辆车…撞上 T:x」→ 补出 `collide(a=null, b=x)`，并把该指令里的
   变速/变道/转向改挂到**新车**上（`track=null`），由 `apply` 在拿到新 id 后回填；
4. 丢掉引用"不存在/已删"车辆的 op。

**一句话生成"肇事车"**：`insert` 与 `collide` 可以写在同一条指令里。`collide` 的
`a`/`b` 与动作的 `track` 写 `null`（或 `-1`）即代表"刚生成的那辆新车"，`apply` 会先跑完
生成拿到新 track id 再替换占位符，于是「生成一辆红色轿车，向左变道后撞上 T:1」能一步到位地
生成车辆**并**生成它引发事故的轨迹。

## 高精地图（Waymo v1.4 map）——道路约束

`data/waymo14/processed/validation/<NNN>/map/` 里存着 Waymo 官方地图（车道中心线、
车道标线、道路边界、斑马线、停车标志、减速带、出入口），和 `ego_pose` 同一全局坐标系。
studio 把它**当成一套道路约束规则**接进了渲染、路径编辑和事故生成。

### 1. 地图怎么对齐到重建出来的场景

DGGT 场景的世界系是模型内部归一化坐标乘 `scale_factor` 的产物，它和 Waymo 全局系之间
只差一个固定的 3D 相似变换 `p_scene = s·R·p_global + t`。两边都有同一台相机的观测，所以
闭式解、**不需要人工标定**（`studio/backend/waymo_map.py`）：

```
processed 侧（米制、全局系）：T_g_i = ego_pose[frame_ids[i]] @ extrinsics[cam]
scene     侧（米制、场景系）：T_s_i = 相机位姿 (camera_extrinsics_world)
```

1. **帧对应关系**由 `scene_meta.json` 给出（inference.py 导出：`start_idx`/`interval`/
   `frame_ids`/`scale_factor`），所以是确定的，不用猜；
2. 水平面（全局 `x,y` → 场景 `x,z`）做 2D 相似变换，竖直方向由**场景的"上"方向**唯一
   确定（相机图像"下"方向就是场景 +y ⇒ 场景"上"是 -y），两者一起给出一个 `det(R)=+1`
   的完整旋转；
3. 只用**相机位置**做 Procrustes，**不用相机朝向**：实测模型的相机朝向预测与 WOD 的
   相机轴向约定差一个固定常数阵（同一场景里用旋转去解 R 会偏 100° 以上），旋转只当
   诊断指标；位置拟合的残差是厘米级。

实测（25 帧窗口，两次独立重建）：

| 场景 | 内容 | 尺度 `s` | 位置残差 rms | 自车到最近车道中心线（中位/最大） | `up_dot` | 镜像候选 rms |
| --- | --- | --- | --- | --- | --- | --- |
| 016 | 四向环岛路口 | 1.0011 | 0.062 m | 0.15 m / 0.25 m | +1.000 | 5.82 m |
| 005 | 城市主干道 × 两个十字路口 | 0.9986 | 0.144 m | 0.48 m / 0.92 m | +1.000 | 9.05 m |

> 一个必须修的坑：上游 `inference.py` 用 `str(scene_idx).zfill(3)`（**第几个 batch**）
> 当场景名去读 GT 位姿算 `scale_factor`，只要传入的场景名不是 `001`，scale 就会用**别的
> 段落**的位姿去算 —— 整个场景的米制尺度全错，地图自然对不上。现已改为按
> `batch['image_paths']` 反推真实场景名（并写进 `scene_meta.json`）。

**质量闸门**：对齐不可信时宁可不用，也不画一张错的底图。下面任一条不满足就返回
`available:false` 并给出原因（老场景没 `scene_meta.json` 时帧对应关系是猜的，很容易踩到）：
拟合尺度 `s` 不在 [0.85, 1.18]、位置残差 rms > 1m、自车到最近车道中位 > 3.5m、
变换后的地图跨度 > 2km。例：`output/waymo_eval/000/test1`（旧数据、无 meta）会得到
`s=23557 / rms=12.1m / 地图跨度 5.6e6m` → 直接判不通过。

### 2. 在视图里可视化

* **3D 视图叠加**（工具栏最右的"道路"按钮）：`viewer3d.js` 把车道中心线（绿）、车道标线
  （黄）、道路边界（橙）、斑马线（洋红）、停车标志/路口锚点（红/蓝点）投影到叠加上，
  只画相机 140m 内的段；
* **俯视 BEV 示意图**：`render_bev_frame()` 把地图当底图画在网格之上、物体之下；
* **场景信息栏**显示"已接入 N 车道 / N 标线 / … （对齐 x.xx m）"，没地图的场景会说明原因；
* 接口：`GET /api/scene/map/{scene_id}?kinds=lane,road_edge&max_points=…`（场景世界系、
  单位米、折线保留 2 位小数），不可用时返回 `available:false` 和 `reason`。

命令行诊断（对齐指标 + 俯视叠图）：

```bash
python studio/backend/waymo_map.py \
    --scene output/waymo_eval_14/016/016 \
    --processed data/waymo14/processed/validation/016 \
    --overlay /tmp/map016.png
```

### 3. 道路约束规则（`studio/backend/road_rules.py`）

| 规则 | 含义 | 落地方式 |
| --- | --- | --- |
| R1 车道走廊 | 轨迹点应落在某条 LaneCenter 走廊内（默认 ±3.5m） | `snap_centers()`：**有界**横向吸附（单帧最多 2.6m，超了就只报警不动），返回吸附前后体检对照 |
| R2 航向一致 | 车头方向应与所在车道方向一致，反向即逆行 | `classify()` → `wrong_way` || R3 车道连通 | 跨车道只能走 entry/exit/左右邻居边 | `successors()` / `plan_turn()` 的 BFS |
| R4 路口转向 | 左/右转只发生在路口，且要沿出口支路几何转弯 | `plan_turn()` / `turn_for_track()` |
| R5 限速 | 车速上限取车道 `speed_limit_mph` | `speed_limit_mps()`；转弯还会再压到限速的一半 |
| R6 语义锚点 | 停车标志 / 斑马线 / 路口进口 | `anchors_near()` / `junction_ahead()` |

**"最近车道"必须带上方向过滤**（`lane_at(..., heading=)`）：路口附近不同方向的车道中心线
会挨得极近 —— 实测 scene 005 的自车在路口处到"最近车道中心线"只有 0.2m，但那条是**横穿**的
车道（方向差 88°）。纯按距离选会把正常行驶的车判成逆行（实测误报 6 帧、航向偏差中位 33°），
横向吸附时还可能把车吸到对面车道上。加上 `heading` 过滤后：航向偏差中位 **33° → 2°**、
误报逆行 **6 帧 → 0**，而本来就正常的 scene 016（0.7°）不受影响。

**R4 为什么要在车道图上"多搜几跳"**：Waymo 的路口结构是"进口车道先扇出成若干条平行车道，
再各自接到不同去向"（实测 scene 016：进口 49 → {40,52,53,54} → 分别接 39 / 58(左, +111°)
/ 50 / …）。只看一跳看不出转向，所以 `plan_turn()` 从当前车道出发做有界 BFS，用**路径走
`forward_m` 之后的净航向变化**去匹配目标方向（`turn_side() > 0` = 左转）。

接入点：

* **插入实体**（`nl_entity.plan_relative_trajectory`）：生成的中心点先做一次**逐帧**
  最近车道吸附（不是整条轨道吸附到单一车道 —— 这样"变道撞击"的变道结构不会被压平），
  吸附结果写进回执的 `road_constraint`，`/api/text2entity/*` 会把它转成前端 warning；
* **语言转向**（`traj_llm._op_turn`）：有地图就 `map_lane_route`（位置沿车道弧长前进、
  航向取车道方向、速度压到限速一半），没有地图才退回"原地把朝向转过去"并附 `note`；
  如果这个位置/方向地图里根本没有可用的转向支路（搜到的路线航向几乎没变），
  回执里会带 `warn` 说明"车只是沿当前车道往前开了一段，没有真的转过去"——不假装成功；
* **事故生成**（`corner_case.simulate_pair_collision`）：撞击**之前**的行驶段做轻度吸附
  （≤1.0m），撞后不动（撞车时压线/冲出路面是合理的）；结果里附 `road_report`
  （双方 `on_road_ratio` / `dist_med_m` / `wrong_way` / `issues`）。

### 4. 用任意 waymo14 段落重建场景

一条命令（processed 已就绪时；缺标记图会自动补，最后打印对齐验收）：

```bash
bash build_scene.sh 017                  # 默认 START_IDX=0 / SEQ=25
START_IDX=100 bash build_scene.sh 017    # 换起始帧
bash build_scene.sh 017 018 005          # 多个场景依次建
```

拆开看就是三步：

```bash
# 1) 解析 tfrecord（位姿 + 相机内参外参 + 图片 + 地图）
python datasets/waymo14_preprocess.py --dst data/waymo14 \
    --index_from data/waymo_mytest_list.txt --only data/waymo_mytest_list.txt

# 2) 补 DGGT 需要的标记图（SegFormer 语义 → sky_masks/custom_masks + 派生动态掩码）
bash tools_make_masks.sh 005

# 3) 重建（`run_inference.sh` 已把 ninja/nvcc 放进 PATH —— gsplat 首次渲染要 JIT 编译 CUDA kernel）
bash run_inference.sh --image_dir data/waymo14/processed/validation --scene_names 5 \
    --input_views 1 --sequence_length 25 --start_idx 0 --mode 2 \
    --ckpt_path pretrained/model_latest_waymo.pt --output_path output/waymo_eval_14/005 -images

# 4) 核对地图对齐
python studio/backend/waymo_map.py --scene output/waymo_eval_14/005/005 \
    --processed data/waymo14/processed/validation/005 --overlay /tmp/map005.png
```

几个容易踩的坑（现在都会**提前报清楚**，不再抛晦涩的 `IndexError`）：

* **缺 `sky_masks/` 或 `fine_dynamic_masks/all`**：mode 2 会无条件读天空掩码，缺图时只会抛
  `IndexError: list index out of range`，指向一行看起来毫不相干的代码。现在 `inference.py`
  在**加载模型之前**就 `preflight_check()`，直接列出哪个场景缺哪个目录 + 给出补图命令；
* **`-depth` 不需要 GT 深度**：它存的是渲染出的预测深度。只有 `comparison.mp4` 需要
  `depth_flows_4`（GT 深度），没有时会明说跳过，而不是抛 `KeyError('gt_depth')`；
* **`--intervals` 在 mode 2 下无效**（dataset 内部固定 interval=1），想要别的窗口用
  `--start_idx` / `--sequence_length`；
* **输出目录名 = 真实输入场景名**（`output/waymo_eval_14/017/017`）。上游用"第几个 batch"
  当名字，既让目录名对不上，也会让 scale recovery 去读别的段落 —— 已修；
* 选窗口看自车速度：`data/waymo14/processed/validation/<NNN>/ego_pose/` 相邻帧间距的中位数
  就是每帧位移。拿一段停着不动的帧建出来的场景没有动态物体可用。

第 2 步的 `datasets/tools/derive_dynamic_masks.py` 解决的是：上游
`extract_masks.py --process_dynamic_mask` 需要另一套 2D 检测器产出的
`dynamic_masks/{human,vehicle}` 粗掩码；没有那一步时直接用 SegFormer 的
Vehicle/Person/Cyclist 类当粗掩码，等价于 `valid = semantic ∧ rough`。
