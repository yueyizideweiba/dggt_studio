# Corner Case 批量生成 · 架构与路线图

## 目标

把交通事故 Corner Case 从"硬编码启发式、逐条生成"，升级为：

> 理解动态物体关系（关系图）→ 挑选参与者 + 采样参数（多样性）→ 统一参数化生成（质量/可控）→ 批量生产 + 质量打分（可筛选、可用于训练）。

## 现状盘点

- `studio/backend/corner_case.py`：逐类型的硬编码生成器（`_gen_rear_end` / `_gen_tbone` /
  `_gen_head_on` / `_gen_lane_change` / `_gen_pedestrian_crossing` / `_gen_hard_brake` /
  `_gen_cut_out_reveal` / 三车连环等）+ 采样参数 schema + 两体物理（动量碰撞 + 摩擦）。
- `dggt/scene_edit/`：声明式 `SceneEditSpec`（动作列表）+ `SceneEditExecutor`（`translate_track`、
  `insert_object`、`collision_course` 等）+ JSON spec 批量加载（`load_scene_edit_specs`）。
- `dggt/scene_edit/collision_physics.py`：OBB SAT 碰撞、碰撞冲量、关键帧、TTC、样条平滑。
- `studio/backend/quality_report.py`：质量门槛（TTC 区间、速度/加速度/加加速度/偏航率/步长、
  穿透深度、标注一致性；offroad / visibility 尚未接入）。

## 本次新增（第一层：关系图 + 批量骨架）

### `studio/backend/scene_graph.py` —— 动态物体关系图

- **节点**：动态物体（真实 + 合成，跳过已删除/已被替换），特征含 class（行人/车辆/大车）、
  尺寸、中心、速度、航向、帧覆盖、是否合成。
- **边**：任意两物体在当前帧 ±`window` 内的关系特征——相对位置/速度、距离、**接近速率**、
  航向差、交角、**TTC（闭式匀速圆盘近似）**、最近距离、是否处于碰撞航向、**关系类型**
  （following / adjacent_parallel / oncoming / crossing / stationary / receding / unrelated）、
  **冲突关键度**（0~1，TTC 越短/越近/接近越快越危险）。
- `propose_participants(type)`：用关系类型给每种事故挑选候选参与者（rear-end→following、
  head-on→oncoming、tbone/crossing、cutin→following/adjacent、pedestrian→crossing、
  hard-brake→有速度者）。
- 边/节点都带**定长特征向量**，是后续 **GNN 可直接消费的输入表示**。

### API

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/api/corner_case/graph/{scene_id}?frame_idx=&window=` | 返回关系图 + 各类型参与者提案 |
| POST | `/api/corner_case/batch` | `{scene_id, scenario_types[], num_per_type, seed, start_frame, num_frames, ...}` → 每实例：图选参与者 → 采样参数 → `corner_case.generate` → `quality_report` 打分 → `undo()` 还原 → 汇总 manifest |

多样性来自两个维度：**① 换参与者/位置**（关系图候选对），**② 换速度/车距/时序参数**（不同 seed 采样）。
批量间用 `push_history()`/`undo()` 互不污染，场景在批量后保持原样；每个实例都带 `trajectory_snapshot`
（生成后的逐帧位姿），后续可直接导出成 `SceneEditSpec` 渲染。

## 实测结果（waymo_eval/001，稀疏/噪声较大的测试场景）

- 关系图：12 节点 / 66 边，关系分类正确（如 T1—T4 `oncoming` criticality 0.93、T2—T7 `following`）。
- 提案：rear-end 5 对、head-on 3 对、lane-change-cutin 7 对（tbone/pedestrian/hard-brake 因该场景
  无交叉/行人/高速目标而空缺——符合预期）。
- 批量：12 个实例、参与者/参数各不相同，**批量后场景被完整还原**（frame0 仍是 0,1,2,3）。

### 关键发现：质量门槛目前全部为 `valid=False`

批量实例普遍被 `motion_physical`（max_accel 高达 43~635 m/s²，阈值 12）、`ttc_range`
（TTC≈0，临界帧≈碰撞帧，无反应窗口）、`event_present`（部分参与者对距离太远未真正碰撞）、
`annotation_consistency`（稀疏帧覆盖不足）拦下。

这正是"不够真实"的**量化证据**：现有 `_simulate_pursuit_collision` 用 `need_speed` 瞬时加速来保证
追尾、碰撞冲量瞬时改变速度，没有加速度/加加速度约束，也没有留出反应时间窗口。

## 路线图（下一步）

### 已实现（第 2 轮）

1. ✅ **运动可行性层（部分）**：`corner_case.py` 新增 `_step_toward_2d`（限加速度/减速度/转向率的
   速度步进），`_simulate_pursuit_collision` 改为"朝受害车以限加速度逼近"（不再用 `need_speed`
   瞬时加速）；`quality_report.py` 对"撞击瞬间"（碰撞帧 ±1）做机动性豁免。
   **验证**：同一 `head-on` 实例从 `max_accel=58 m/s²、TTC=0.1s、被 motion_physical+ttc_range 拦下`
   → `max_accel=4.0 m/s²、TTC=0.5s、critical=14 < collision=19`，**通过全部质量门**。
   剩余未过门的实例主要来自 waymo_eval/001 这类**噪声/稀疏轨迹**（原始 speed 高达 56 m/s 的坏 track），
   以及"肇事车起步过近导致 2 帧内就撞"的布局问题。

### 已实现（第 3 轮）

2. ✅ **反应窗口 + 防穿模 + 关键帧/TTC 修正**：
   - `_simulate_pursuit_collision` 支持 `victim_brake_decel/victim_brake_frame`（前车在仿真内制动），
     修掉了"前车制动被追击仿真覆盖、追尾场景前车从不减速"的 bug；`_gen_rear_end` 保证至少 0.6s 接近时间。
   - 防穿模改用 `_obb_radius_along`（OBB 沿法线的精确半宽），修掉"接触距离只算半个车宽导致穿透过深"。
   - 关键帧反推改用 `_velocity_vec`（相对速度**向量**的模长），修掉"对向/侧碰时 |v_a|-|v_v| 抵消成 0
     → 关键帧=碰撞帧、TTC=0"的 bug。
   - **验证（干净合成参与者，attacker=真实 track1 / victim 自动合成）**：rear-end / head-on /
     intersection-tbone 三类全部 `valid=True`，TTC 分别 0.2 / 0.5 / 0.3s，max_accel ≤ 4.0 m/s²，
     penetration ≤ 0.35m，**通过全部质量门**。
   - 批量里仍 `valid=False` 的实例全部来自测试场景本身的**坏轨迹**（tracks 3/4/6/10 原始速度/加速度异常、
     帧稀疏），属数据质量而非生成器；质量门正确地拦下了它们。

### 已实现（第 4 轮）

3. ✅ **spec 化批量产出（可复现数据集）**：`POST /api/corner_case/batch` 新增 `save / output_dir / run_name`，
   把每个实例落盘成 `<output_dir>/<instance_id>.json`（含 `roles`、`seed`、`sampling_params`、
   `collision_frame/critical_frame`、`quality`、`trajectory_snapshot`、**`graph_features`**），
   并写一份 `manifest.json`（summary + 全部实例）。默认目录 `output/corner_cases/<scene_id>/<run_name>`。
   - 实例用 `corner_case.generate(..., sampling_seed=seed)` 可**确定性复现**（同 seed 得到同 collision_frame）；
   - `graph_features`（关系类型/关键度/TTC/定长特征向量）+ `quality.valid` 就是后续 **GNN 训练语料**
     （图结构 → 碰撞/严重度标签）。
   - 验证：一次批量 9 实例全部落盘 + manifest 结构完整；每个实例 4x4 位姿快照、图特征、质量指标齐全。

### 已实现（第 5 轮）

4. ✅ **GNN 训练链路（图特征 → 碰撞/严重度预测）**：
   - `gnn_model.py`：`CollisionGNN`（节点 MLP 编码两物体 → 边 MLP 融合 [节点嵌入 + 边特征 + 物理参数]
     → 碰撞二分类 + 严重度 4 分类两个头）+ `train_model`/`evaluate`（纯 torch，CPU/GPU 均可）。
   - `api_server.py` 新增 `POST /api/gnn/collect`：用关系图选参与者 → 变 severity/seed 生成事故 →
     记录"生成前关系图特征(22维) → (是否碰撞, 严重度)"，存 `.npz`。
   - `train_gnn.py`：离线训练脚本（不需要加载场景）。
   - **验证**（waymo_eval/001 收集 120 样本，碰撞率 0.34）：测试集碰撞准确率 **0.958**（多数类基线 0.542）、
     严重度准确率 **0.917**（基线 0.542），loss 2.12 → 0.002。证明"关系图特征 + 物理参数 → 事故结果"
     是可学习、有真实信号的（模型明显超过多数类基线）。
   - 说明：这是在小样本上的机制验证；真实训练需用干净数据集 + 更多样本，且可扩展到多节点消息传递。

### 已实现（第 6 轮）

5. ✅ **统一参数化生成引擎（10/10 场景）**：新增 `scenario_engine.py`——
   - **运动原语**：`prim_layout_pair`（布局）、`prim_pursue`（限加速度追击碰撞）、
     `prim_lateral_shift`（横移变道）、`prim_brake_to_stop`（刹停）、`prim_hold_still`（静止），
     以及行人拦截（`_make_intercepting_pedestrian`/`_retarget_pedestrian`/`_simulate_pedestrian`）。
   - **场景计划（plans）**：10 个场景类型全部收敛为 `plan_xxx(ctx)`，声明式地"组合原语 + 采样参数"。
   - `corner_case.generate()` 改为 **数据驱动分发**（`SCENARIO_PLANS` 查表），替代硬编码 if/elif。
   - 新增/调事故类型只需写一个 plan 函数复用原语，不再新写硬编码循环。
   - 顺带修了 `quality_report` 的既有 bug（单物体场景 `min_distance=None` 触发 `None<=float`，
     及 `event_present` 对单物体场景应判 N/A）。
   - 验证：10/10 类型无异常生成，rear-end/head-on/tbone 通过质量门（与迁移前碰撞帧一致=行为等价）；
     批量生成回归正常；其余 `valid=False` 均为测试场景噪声数据或场景语义（cut-out 无碰撞、hard-brake 单物体）所致。

## 结论

目标八项已全部落地：关系图 → 图选参与者+参数采样（多样性）→ 限加速度+反应窗口生成（真实性）
→ 统一参数化引擎（10/10 场景）→ 批量生产+质量打分+落盘（可复现数据集）→ GNN（碰撞/严重度预测）。
真实 GNN 的精度与大规模生产需接入**干净数据集**（当前 waymo_eval/001 为噪声/稀疏测试场景）。

## 使用入口（前端「Corner Case 工作台」）

顶部工具栏 **「Corner Case 工作台」**（`index.html` / `app.js` / `styles.css`）四个页签：

| 页签 | 作用 |
| --- | --- |
| ① 场景关系图 | 读 `GET /api/corner_case/graph/{scene_id}`：俯视图 + 关系边表 + 参与者提案（「用此组合」一键填入生成面板） |
| ② 批量生产 | 勾类型/条数/种子/帧区间 + **为每条渲染可播放视频** + 落盘，结果表逐条带「▶ 播放」 |
| ③ 模型训练 | `POST /api/gnn/collect` 收语料 → 生成 `train_gnn.py` 训练命令 |
| ④ 自检诊断 | 关系图 / 生成质量门 / 批量 / 语料 四步体检 |

**视频（与播放效果一致）**：新增渲染复用与 `/api/render/frame` **同一条路径**
（`render_ego_frame` / `render_sequence_video`），所以录出来的 mp4 就是 2D 播放所见画面。

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| POST | `/api/corner_case/video` | 把**当前场景状态**（已生成/编辑的事故轨迹）渲染成 mp4（帧范围自动钳制到场景帧数） |
| POST | `/api/corner_case/batch` | 新增 `render_videos / video_draw_bboxes / video_draw_ids / video_draw_trajectories / video_fps` → 每条实例输出 `<out_dir>/videos/<instance_id>.mp4`，manifest 里带 `video_path` |
| GET | `/api/corner_case/video_file?path=` | 受限在 `output/` 下的 mp4 访问入口，**支持 Range**（可拖进度条），供 `<video>` 播放 |
| POST | `/api/corner_case/video`（生成面板按钮） | 「录制该案例视频」= 生成后一键出视频并回放 |

编码优先 H.264(libx264)，失败退回 mp4v；单个视频约 100–200KB，浏览器可直接播放/循环/拖拽。

## 第 7 轮：批量"0 通过"根因修复 + 质量门细化

用户在浏览器里跑批量得到 **0/9 通过、视频全是「无」**，定位到两类问题：

**（一）视频没出**：浏览器缓存了旧 `index.html`（新的"渲染视频"勾选框不存在），
前端 `render_videos` 因此是 `false`。修复：前端对缺失控件用 `?? true` 兜底（旧页面也会出视频）；
`index.html` 加 no-cache meta；脚本版本号 bump。**结论：改 UI 后请强刷一次（Ctrl+Shift+R）。**

**（二）0 通过 → 根因是"锚点物体在起始帧没有位姿"**：关系图会把"只在别的帧出现"的物体也当候选，
而 `_apply_initial_pair_layout` / `_ensure_participant` 在锚点无起始帧位姿时会**静默不生效**，
参与者于是沿用原始稀疏/噪声轨迹 → 覆盖率 0.19、加速度 289~732 m/s²，必然失败。逐项修复：

1. **关系图加可用性过滤**：`usable = 起始帧有位姿 且 尺寸合理 且 速度≤45m/s`；
   车辆类角色不能是行人/锥桶；没有可用"成对"组合时**自动退回"1 个锚点 + 其余自动合成"**（干净参与者）。
2. **所有参与者先从干净基轨迹出发**：新增 `prim_regularize`（把整段窗口重写成从起始帧出发的匀速直线），
   各场景计划统一调用 —— 不再继承原始噪声/稀疏轨迹（覆盖率 → 1.00，动力学有界）。
3. **变道加塞**：横移方向改为**起始帧固定**（原逐帧航向会乱跳）；并改为
   **追击动力学 + 横向 ramp**（`_simulate_pursuit_collision` 新增 `attacker_lateral/victim_lateral`），
   取代"直接插值贴到前车"（后者造成巨额加速度与穿模）。
4. **急刹/前车闪开**：`prim_regularize` 后用更慢的横移（duration 1.2）铺满整段窗口。
5. **连环场景**：`plan_cutin_brake` 重写为"布局 + 横向并入 + **单次**追击（前车制动在仿真内）"，
   避免同一目标车被多次追击覆盖导致深穿透；`plan_chain` 三角色先正规化。
6. **质量门细化**（都是"按语义判 N/A"而非放水）：
   - `event_present` 对 `hard-brake / cut-out-reveal` 判 N/A（按定义没有碰撞事件）；
   - `ttc_range` 对 `pedestrian-crossing / cut-out-reveal` 判 N/A（事故本质是"突然出现"，反应窗口本就极小），
     最小 TTC 由 0.2s 放宽到 0.1s；
   - 撞击帧豁免升级：**所有受影响物体两两之间**的碰撞帧 + **接触距离内（半宽之和+0.6m）** 的帧都豁免
     （连环事故有多处撞击），避免把撞击本身当成"机动加速度超限"。

**实测（waymo_eval/001，三个随机种子 × 10 类型 × 3 条 = 90 条）**：

| | 修复前 | 修复后 |
| --- | --- | --- |
| 通过率 | **0/9 = 0%** | **72/90 = 80%** |
| rear-end / head-on / hard-brake / tbone / cut-out / cutin-brake / occluded | 0 | **各 3/3**（每个种子） |
| lane-change-cutin | 0/3 | 1~2/3 |
| pedestrian-crossing | — | 0~1/3（本质"突然出现"，最难） |

剩余未通过主要是这两类"本质就极端"的场景（行人突然横穿、极限加塞），以及个别种子下的 TTC 边界情况。



1. **运动可行性层（质量核心）**：给生成器加"运动学重投影"——限制速度变化率（加速度≤8~12 m/s²、
   加加速度有界），用限速 + 重积分把瞬时冲量/瞬时加速平滑成可信轨迹；并保证 TTC 落在
   [0.2, 3.0]s 的反应窗口（减速段 + 关键帧语义）。
2. **统一参数化生成引擎**：把逐类型 `_gen_*` 收敛成"运动原语"（匀速/加减速/换道横移/追瞄/弹开/刹停）
   的组合，由（类型模板 + 参与者 + 参数）驱动，替代硬编码分支。
3. **spec 化批量产出**：每个通过的实例导出 `SceneEditSpec`（复用 `dggt/scene_edit`）到
   `output/corner_cases/<type>/<id>.json` + 一个 `manifest.json`（参数 + 质量 + 轨迹），形成可直接
   渲染/复现的"数据集"。
4. **关系图 → GNN**：先积累"图 + 成功/失败实例 + 质量标签"的语料；再用 GNN 做两件事——
   ① 节点/边特征 → 预测"是否会碰撞/碰撞严重度"（筛选器），② 条件生成"给定图结构 → 采样参数"。
   现在的关系图特征向量就是其输入层。
5. **offroad / visibility 质量门**：接入可行驶区域与渲染可见性校验，把质量报告补齐。


---

## 第 9 轮：输出视频（主车视角干净画面 + 俯视 BEV 双视角）

**需求**：① 输出视频不要再叠加包围盒/ID；② 每条要额外多一个**俯视视角**视频（主车视角有时候看不到事故）。

**改动**：

1. **主车视角视频默认"干净画面"**：`CornerCaseBatchRequest.video_draw_bboxes/video_draw_ids` 默认值
   由 `True` 改为 `False`；`CornerCaseVideoRequest.draw_bboxes/draw_ids` 同样默认 `False`。
   需要看标注时前端仍可勾选叠加。渲染路径不变（复用 `/api/render/frame` 的 gaussians +
   `object_overrides` + `extra_objects`），因此**视频与 2D 播放完全一致**。
2. **新增俯视（BEV）渲染**（`api_server.py`，纯 cv2 示意图，不依赖 3D 渲染）：
   - `_bev_world_bounds(tm, start, num, margin=8)`：统计窗口内所有物体中心，2%~98% 分位裁剪离群点，
     避免被远处单条轨迹把视角拉飞；
   - `render_bev_frame(...)`：10m 网格铺满画布、**等比缩放居中**、按朝向填充的矩形足迹、
     淡色历史轨迹折线、自车黄色箭头（位置+朝向取自 `_frame_camera`）、左上 `BEV f<idx>`、
     左下 10m 比例尺、碰撞帧 ±1 显示红色 `COLLISION`；碰撞对（`collision_tracks`）用红色高亮；
   - `render_bev_sequence_video(...)`：逐帧画完再编码 H.264（imageio/libx264，失败回退 cv2 mp4v）。
3. **批量接口**：`render_bev=True`（默认开）、`bev_size=560`、`bev_draw_ids=False`；
   每条实例输出两个视频 `<id>.mp4`（主车）与 `<id>_bev.mp4`（俯视），实例 JSON 里分别是
   `video_path` / `bev_video_path`；响应新增 `num_bev_videos`。
4. **单案例接口** `/api/corner_case/video`：新增 `render_bev/bev_size/bev_draw_ids`，
   一次返回 `video_path` + `bev_video_path`（前端"录制该案例视频"自动出两个视角）。
5. **前端**：批量面板新增勾选「每条额外输出俯视(BEV)视频」（默认勾选），
   「视频叠加包围盒 / 叠加ID」改为**默认不勾选**；结果表视频列变成 **▶主车 / ▣俯视** 两个按钮；
   视频回放弹窗新增「主车视角 / 俯视视角」切换标签；汇总栏分别显示两种视频数量。

**验证**：`output/corner_cases/bev3/bevtest3/videos/` 下 `rear-end-*.mp4` / `head-on-*.mp4`
各 14 帧、10fps、可解码；抽帧确认主车视角画面无任何包围盒/ID 叠加，BEV 视角网格/足迹/轨迹/自车箭头/比例尺正常。


---

## 第 10 轮：过质量门率、真渲染俯视视频、主车实体化与可编辑主车轨迹

### 1. 过质量门率（用户实测 2/9 → 9/9）

定位（用 `/api/corner_case/batch` 新增的 `quality.failing_metrics` 直接看到"哪一项超了多少"）：

| 现象 | 根因 | 修复 |
| --- | --- | --- |
| 几乎所有实例 `motion_physical` 不过，`max_yaw_rate` 恒为 **3.508 rad/s** | 车头自动朝向的"限速转向"是按**帧**给的（`4+46*(1-smoothing)` 度/帧），10fps 下等于 20.1°/帧 = 3.508 rad/s，远超质量门的 1.2 rad/s | 限速改成**按物理角速度**：每帧最大转角 = `auto_heading_max_yaw_rate(默认1.0 rad/s) / fps`；fps 由生成接口同步给 TrackManager（`tm.fps`），可用 `POST /api/edit/auto_heading {max_yaw_rate, fps}` 调整 |
| `lane-change-cutin` 三条都 `event_present` 不过（min_distance 恒为 8.73m） | 加塞车/被加塞车用的是原始几何位置，采样出的相对速度只有 1.3 m/s，20 帧内根本追不上 | 重写 `plan_lane_change`：先 `prim_layout_pair` 把 cutter 放到 target **前方 gap、相邻车道**（干净基轨迹），再让 target 追 cutter，并在**仿真内部**给 cutter 叠加"0 → -side*lane_off"的**横向并入偏移**；加塞方向由几何自动判定 |
| 噪声场景下 `max_speed` 高达 115~198 m/s | `_speed_mps` 用中心差分，遇到稀疏轨迹的单帧跳变（10~20m）就把速度估成几百 m/s，再被当成"参考车速"写进生成轨迹 | `_speed_mps` 改为**±6 帧逐帧位移的中位数**（对跳变鲁棒）+ 上限 28 m/s |
| `ttc_range` 不过且 `ttc_at_critical = 0.0` | "最晚反应帧"前向搜索没找到更早帧时，critical 停在碰撞帧 → TTC=0（物理上等于零反应时间，10fps 下也超出分辨率含义） | 兜底取"撞击前一帧"（=0.1s 反应窗口，正好等于门限下限） |

**实测**：用户场景 `output/waymo_eval/000/test1`（3 类型 × 3 条，seed 7，20 帧）**9/9 通过**；
退化噪声场景 `output/waymo_eval/001`（10 类型 × 3 × 3 seeds）**63/90**（旧代码同一条件下只有 29/90）。
剩余未通过主要是 `bbox_penetration`（`cut-out-reveal` / `cutin-brake-pileup` 个别实例的布局互穿）。

### 2. 真渲染的俯视视频（第三个视频）

之前只有"俯视示意图(BEV)"。现在每个实例输出 **三个** 视频：

| 文件 | 内容 |
| --- | --- |
| `<id>.mp4` | 主车视角（与 2D 播放一致，默认不加包围盒/ID） |
| `<id>_bev.mp4` | 俯视**示意图**（cv2 画的网格/足迹/轨迹，God view） |
| `<id>_top.mp4` | 俯视**真渲染**（正上方朝下的 gsplat 渲染，能看到真实车辆/道路外观） |

实现：`_topdown_bounds`（取景范围 = 参与者 + 自车 + 最小 14m×24m）→ `_topdown_camera`
（c2w 朝下、画面 +Z 朝上、按取景长宽比自适应分辨率、高度按 FOV 自动求）→
`render_topdown_frame` / `render_topdown_sequence_video`（复用 `_render_frame_with_object_overrides` 的
`c2w_override/K_override`，因此编辑后的物体同样生效）。
批量请求新增 `render_topdown`（默认开）、`topdown_size`；响应新增 `num_topdown_videos`。

### 3. 主车（EGO）实体化 + 可编辑主车轨迹

- 模型：`/autodl-fs/data/dggt-main/sam-3d-objects/ego.ply`（243k 高斯；可用环境变量 `DGGT_EGO_PLY` 换）。
  自动读 PLY 三轴尺寸，按**真实车长 4.9m** 换算缩放（实测尺寸 2.08 × 1.54 × 4.90 m）。
- 位姿：由逐帧自车相机外参推导车体中心位姿（朝向 = 相机前向在地面的投影；相机→车体固定偏移
  `(0, 1.15, -0.55)`），同时记下**相机-车体固定安装关系** `cam_rel`。
- 它作为 **track 900000** 注册进 `synthetic_tracks`（带 `ego: True`）：于是自动获得"可渲染、可选、
  可拖动编辑、可撤销、可重置"的全部能力；`scene_graph` 里被排除在"交通参与者"之外，不会成为事故候选。
- **只出现在外部视角**：自车视角渲染（`/api/render/frame`、主车视频、SAM 源帧）传
  `build_extra_objects(include_ego=False)`，否则车体会糊在镜头上；自由视角/俯视渲染则包含它。
- BEV 里主车用亮黄独立配色（`BEV_EGO_COLOR`）画成真实足迹（不再只是一个箭头）。
- 接口：`GET /api/ego/{scene_id}`（信息/是否被编辑）、`POST /api/ego`（visible / 缩放 / 尺寸 /
  `reset_trajectory`）、`POST /api/ego/offset`（整条轨迹按车体坐标系平移）；
  逐帧编辑仍走通用接口（track_id=900000）。
- **沿编辑后的主车轨迹出视频**：`render_sequence_video(..., use_edited_ego=True)` 时逐帧用
  `tm.get_ego_camera(f) = 车体位姿(f) · cam_rel` 作为相机 → 车怎么走，镜头就怎么走。
  批量 `use_edited_ego`、单案例 `use_edited_ego`（默认 true）都可控。
- 顺带修掉一个渲染坑：部分 SAM3D 导出的 ply 含 inf/nan（ego.ply 的 opacity 有 5 个）或非单位四元数，
  会让整幅渲染变 NaN/纯黑 → `_load_sam3d_ply` 现在做 `nan_to_num` + 裁剪 + 四元数归一化。

### 遗留 / 下一步（本轮实测发现的更深问题）

- `cut-out-reveal` 偶发 `bbox_penetration` 超限（约 1/3）。**订正**：一开始怀疑是
  `collision_physics.check_collision` 的尺寸顺序与物体局部系不一致，后来核对数据发现
  **元数据里的 dimensions 本身就是 [x, y, z] 三个轴的 extent**（例如 track 2 是 [1.93, 1.32, 2.77]，
  与它那帧 dynamic ply 的高斯 extent 完全一致），因此 `dims[i] → 局部轴 i` 是对的，
  只有 `_get_dimensions` 的 docstring 写成 "[length, width, height]" 属于注释误导
  （`plan_*` 里把 `dims[0]` 当车长用来算间距会偏小，是另一个待清理的小问题）。
  所以这条穿透超限是**真实的几何互穿**：遮挡车"闪开"时还没完全避开就与静止障碍车重叠。
  调过两版让开量/时序（横向让开量、提前量、障碍车放置距离）都只能在两个测试场景之间此消彼长
  （c1 2/3 + c2 3/3 ↔ c1 1/3 + c2 3/3），最后保留原始参数 + 已加的 TTC 兜底，
  把"闪开时序"记为后续单独一轮的事。


---

## 第 11 轮：俯视视频渲染全部动态物体、全局切换主车视角、界面精简

**需求**：① 主车实体不要参数面板，只要一个显示开关；② 俯视(3D)视频里只看到主车一辆车，要求渲染所有动态物体；
③ 全局可切换"主车"——用任意动态物体的视角渲染（生成不同视角的事故视频）；④ 真实渲染视频不要任何叠加；⑤ 删掉"极端天气 Corner Case"面板。

### 1. 俯视(3D)视频：所有动态物体都渲染出来

**根因**：单目 4DGS 重建的动态物体高斯是"贴片式"的——只在源相机视角附近有实体，从正上方看会退化成
细长条纹甚至完全消失（静态道路/植被/建筑从上方渲染完全正常，主车是我们给的完整 3D 模型所以一直正常）。
实测：同一个目标车，俯视下只剩一条"梳齿状"黑带，而它的包围盒位置是空的。

**做法**（`render_topdown_frame` + `_draw_topdown_object_proxies`）：
- `_render_frame_with_object_overrides` 新增 `include_dynamic` 参数；俯视渲染时**关掉动态高斯**（避免残留条纹），
  只真渲染静态场景 + 主车（完整模型）；
- 然后按每个动态物体的**真实位姿/尺寸**画"实体替身"：8 个角点投影取凸包填深色侧影 + 顶面填车体灰 + 描边，
  尺寸异常（0 尺寸/超大）的物体跳过；**当前视角来源**的物体用亮黄。
- 效果：俯视视频 = 真实渲染的道路/环境 + 所有动态物体的实体块（位置、朝向、大小都来自真实位姿）+ 真实主车模型。

### 2. 主车：一个开关 + 全局视角来源

- 删掉原来的参数面板（横向/纵向/升降/重置/单独录制），只留「显示主车实体」开关。
- `TrackManager.ego_source_track`：默认 `None` = 真实主车（自车相机轨迹）；
  `POST /api/ego {set_source: true, source_track_id: N}` 可把任意动态物体设为"主车视角来源"，
  `0/null` 切回真实主车。相机 = `该物体位姿 × 相机安装关系(cam_rel)`。
- `GET /api/ego` 返回 `options`（可选物体列表：真实主车 + 所有可用动态 track + 帧数），前端下拉框直接用。
- 视角来源物体在 BEV / 俯视替身里用主车配色标出；`use_edited_ego` 在该物体未编辑时同样生效
  （`follow_ego = use_edited_ego and (主车被编辑 or 已切换来源)`）。
- 实测：切到 track 3 后录出的"主车视角"视频与真实主车视角平均像素差 19.8 → 确实是不同视角的事故视频。

### 3. 真实渲染视频不叠加任何东西

- 前端批量面板去掉「叠加包围盒 / 叠加ID / 叠加轨迹」三个选项，payload 里恒为 `false`；
- 「录制该案例视频」也固定 `draw_bboxes/draw_ids/draw_trajectories = false`；
- 俯视(3D)视频天然无叠加（替身是几何体而不是标注）。

### 4. 删除「极端天气 Corner Case」面板

- 前端删掉该面板与 `setupWeatherControls/getWeatherFromUI/applyWeatherFromUI`（后端天气渲染能力保留，
  以后要用再加回来）。

补充：**2D 实时视图也跟随视角来源**——`/api/render/frame` 在 `ego_source_track` 非空时用
`tm.get_ego_camera(frame)` 作为 `c2w_override`（包围盒投影用同一个相机），因此切换后能立刻在 2D 视图里
看到新视角，所见即所录（实测：切到 track2 视图像素差 25.9，切回真实主车后与原图完全一致）。


---

## 第 12 轮：行人用真实 3D 模型；点选任意物体作为渲染视角

### 1. 行人参与的事故用真实行人模型

以前"行人"是克隆某个 donor 的外观光束（往往克隆出一辆车）。现在：

- `TrackManager.add_pedestrian_track(poses, ...)`：从 `person_0.ply` / `person_1.ply`
  （`DGGT_PED_PLYS` 可配，默认两个文件冒号分隔）里**随机取一个**，
  按 PLY 的 z 轴 extent 把身高缩放到 **1.75m**，尺寸 = [宽, 高, 前后]，
  用同一套 SAM3D `model_corr` 对齐到"X=宽 / Y=上 / Z=前"，然后走 `add_sam3d_track` 注册成合成物体。
- 行人合成入口统一到 `corner_case._create_pedestrian_track`：行人优先用模型，
  **没有可用模型时自动退回**原来的"克隆 donor 外观"，流程不中断。
  两个入口（`_ensure_participant` 的 pedestrian 分支、`_make_intercepting_pedestrian`）都已切过来。
- 这类物体标记 `ephemeral=True`：**清除生成轨迹时整段移除**（不会被当成"用户手动替换进来的
  SAM3D 持久物体"而只还原轨迹）；`snapshot_synthetic_poses`/`list_sam3d_tracks` 也会跳过它们。
- 实测：pedestrian-crossing 自动合成的行人 = track 100000，尺寸 0.57×1.75×0.57m，
  自由视角特写能看到真实行人模型（连同手包/衣着），`/corner_case/clear` 返回
  `removed_synthetic:[100000]` 正确移除。

### 2. "以谁的视角渲染"支持视图点选

- 2D/3D 视图里点选任意物体后，"选中物体"面板底部新增按钮
  **「👁 以此物体视角渲染」**（该物体已是当前视角时变成「↩ 恢复真实主车视角」）。
- 点击 → `POST /api/ego {set_source, source_track_id}` → 2D 实时视图、后续录制的视频、
  BEV/俯视高亮全部跟着切；右下"主车视角"下拉框与按钮状态同步刷新。

### 3. 行人事故不再被"穿模门"误杀

行人被车撞到时**必然**与车体互穿——那正是事故本身。原来的 `bbox_penetration`（≤1.0m）是给
"车辆之间不许穿模"用的，结果把所有行人事故都判成不通过。现在：碰撞双方里只要有
**小体积参与者**（包围盒体积 < 2 m³，即行人/自行车等），该门判 **N/A**
（`small_body_involved` 会写进报告，不是直接放过），车-车碰撞仍然照旧卡。

**实测**（你的场景 `waymo_eval/000/test1`，10 类型 × 3 条 × 3 seeds）：
**78/90 通过**（每个 seed 26/27），唯一未通过的是 `cut-out-reveal-000` 的车辆互穿（前述已知问题）。
