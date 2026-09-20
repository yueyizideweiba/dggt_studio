# 从"能渲染"到"可用于闭环仿真"——路线与进展

> 目标（用户 2026-09 决策）：让基于真实数据重建的场景，能生成**更接近真实世界、可信且稳定**的仿真画面，
> 支撑大规模训练与评测，并**能用于自动驾驶闭环仿真**。

## 一、已确认的三个方向性决策

| 决策点 | 选择 | 影响 |
| --- | --- | --- |
| 下游格式/互操作 | **CommonRoad + OpenSCENARIO(+OpenDRIVE)（标准中立）** | 导出层要做 lanelet/roadgraph（需从 tfrecord 抽地图）、信号灯/障碍物映射 |
| 第一里程碑 | **P1 视觉可信度**：5 相机重训/重建 + novel-view 可信域 + actor 资产库 | 先解决"自车一偏离轨迹画面就崩"，并能量化"偏多少还可信" |
| rollout 时长策略 | **限制在可信域内，越界即截断并记录覆盖率** | 运行时要带 per-frame trust + coverage 截断，不做生成式背景幻想 |

## 二、现状盘点（仓库里已有的资产）

- **原始数据**：`waymo/raw/validation` 25 个 tfrecord（24 GB）
- **预处理数据**：`data/waymo/processed/validation/` 202 个场景，每个含
  `images/`（199 帧 × 5 相机，1920×1280/886）、`images_4/`、`intrinsics/{0..4}.txt`、
  `extrinsics/{0..4}.txt`（相机 rig）、`ego_pose/NNN.txt`（自车外参）、`dynamic_masks/`、
  `ground_label_4/`、`lidar/`、`depth_flows_4/`；其中 **24 个场景**额外有 `sky_masks/`、
  `fine_dynamic_masks/`、`custom_masks/`（可直接跑推理）
- **可渲染产物**：`output/waymo_eval` 26 个（单视角/3 视角/夜间/雨天/转弯变体）
- **模型**：`pretrained/model_latest_waymo.pt`（5.4 GB，Waymo 前置模型）、`model.pt`、
  `tracking_model.pth`；**缺** `pretrained/diffusion_model.pth`（论文里的 diffusion 渲染精修，
  官方 HuggingFace `xiaomi-research/dggt/model_difix.pkl`，是"更真实/更稳的仿真画面"最直接的抓手）
- **管线**：`inference.py`（mode 1/2/3 = 训练/重建/插值，内置 PSNR/SSIM/LPIPS）、
  `train.py` / `my_train.py`、`datasets/dataset.py`（已支持 `views` 与新增的 `camera_ids`）
- **编辑/资产框架**：`dggt/scene_edit/`（`SceneEditSpec`/`EditAction`/`SceneEditExecutor`/`SceneObjectAssetBank`）
- **工具**：`lpips`/`skimage`/`trimesh`/`open3d`/`diffusers`/`av`/`rerun-sdk` 已在依赖中

### 已知的关键限制

1. **单目/少视角重建的视角相关性**：动态物体高斯只在源视角附近有形，从正上方/大角度看会退化成条纹；
   静态背景也只在记录轨迹附近可靠 → **自车偏离轨迹后画面不可信**（这正是闭环仿真的前提问题）。
2. **可信域没有度量**：不知道该在什么偏离范围内使用画面。
3. **actor 不可控**：动态物体是逐帧视角相关高斯，换位/换向就崩；只有主车/行人用了 SAM3D 资产。
4. 预处理数据里的 `ego_pose` 有部分场景是**退化的（整段复制同一帧）**，用时需要检测；
   `data/waymo/processed` 的数值约定是 **z-up（x 前, y 左, z 上）**，而 DGGT 重建场景是
   **y-up 的 OpenCV 相机系**，跨域换算必须显式处理。
5. Windows 之外：`gsplat` 需要 **ninja**（conda env 里有，但必须在 PATH 上，否则 JIT 编译失败）。

## 三、路线（P0→P4）

**P0 地基（确定性 + 运行时骨架）**
`ScenarioSpec`（场景/参与者初始态/参数/seed/资产与生成器版本/内容哈希）→ `reset()/step()/observe()`；
全链路 seed 化；导出兼容 `dggt/scene_edit` 的 `SceneEditSpec`。

**P1 视觉可信度（当前里程碑）**
① novel-view 可信域度量（留出相机/视角的真实误差 + 无 GT 的覆盖度/稳定性）
② 逐帧 trust + 越界截断 + 覆盖率报告
③ actor 资产库（逐帧动态高斯 → 对象中心资产；主车/行人 SAM3D 资产纳入同一库）
④ 5 相机多视角重建（feedforward，无需长训练）→ 扩大可信域
⑤ diffusion 渲染精修（补 `diffusion_model.pth`）→ 更真实、更稳
⑥ 阴影/光照一致性与去物体背景补全（`diffusers`）

**P2 传感器仿真**：可配置相机 rig（内参/外参/时间戳/畸变）、从**仿真自车状态**渲染、LiDAR 仿真
（`lidar/` 真值 + 射线投射）、运动模糊/滚动快门

**P3 闭环运行时 + 评测**：gym 式 step、NPC 控制器、episode 截断（trust/coverage）、
闭环指标（碰撞/near-miss/TTC/offroad/舒适/进度）、多进程批量 rollout

**P4 互操作与规模**：CommonRoad / OpenSCENARIO(+OpenDRIVE) 导出（含 roadgraph）、
分片数据集 + 内容哈希 + 版本化、吞吐优化

## 四、本轮已落地

1. **`datasets/dataset.py` + `inference.py` 支持任意视角数与"留出相机"**
   - `views=N` 通用化（原写死 1/3），`--camera_ids 0,2,4` 显式指定相机；
   - 用 `camera_ids` 就能做"重建时只喂部分相机 → 用留出相机做 novel-view 评测"。
2. **`studio/backend/scene_matcher.py`**：把 `output/waymo_eval` 的重建场景匹配回原始 segment
   （自车系逐帧位移序列滑窗匹配 + 裁掉退化段；横向符号两种约定都试）。
   实测 `test1` 最佳候选 scale≈0.946（≈1，说明重建是米制）、score 0.0147；
   但该场景是直线行驶，运动签名区分度不足 → 需要图像内容二次校验（下一步）。
3. **`studio/backend/novelview_trust.py`**（P1 核心评测器）
   - rig：从 `extrinsics/{0..4}.txt` 推 `相机0→相机C` 相对位姿（与坐标系约定无关）。
     **实测校验通过**：cam1=44.9°、cam2=45.0°、cam3=90.0°、cam4=90.7°，平移 ±0.05m/±0.1m，
     与 Waymo 5 相机 rig（前视 + 前左/前右 ±45° + 侧左/侧右 ±90°）完全吻合；
   - 留出相机 GT 误差：按预处理同规格（宽 518、保比例、过高中心裁剪）加载真实图像，
     用**标定内参**渲染留出相机并算 PSNR/SSIM/LPIPS；
   - 无 GT 在线指标：`coverage`（高斯中心投影的网格命中率）+ `sensitivity`（位姿扰动后重渲染差异）；
   - 偏差度量改成**约定无关**（位移三分量 + 相对旋转角 + 视线夹角），并输出"模型内参 vs 标定内参"诊断。
4. **首次测量（场景 001，8 帧，仅喂前视相机）**
   - 模型预测 fx=529.4 vs 标定 fx=558.7 → **-5.2% 标定/尺度偏差**（与 scene_matcher 得到的 0.946 一致），
     说明重建存在系统性尺度误差，会直接影响几何可信度；
   - **覆盖度随相机横移单调下降**：0m → 0.583、2m → 0.463、4m → 0.353、8m → 0.210
     —— 这就是"可信域截断"要用的在线信号（下一步与真实 novel-view 误差做相关性标定）。

## 五、P1 首批量化结果（场景 `data/waymo/processed/validation/001`，前 8 帧）

**方法**：用 `inference.py` 只喂**部分相机**重建 → 用 `novelview_trust.py` 在**留出相机**上渲染，
与该相机真实图像（按 DGGT 同规格预处理）比 PSNR/SSIM/LPIPS，并算无 GT 的覆盖度。
（rig 已用标定文件校验：cam1/2 = 44.9°/45.0°，cam3/4 = 90.0°/90.7°，平移 9~18cm，与 Waymo 5 相机 rig 一致）

### A. 只喂前视相机（cam0）→ 留出视角的真实误差

| 留出相机 | 视角偏离 | PSNR | SSIM | LPIPS | coverage |
| --- | --- | --- | --- | --- | --- |
| cam1 前左 | 44.9° | 7.21 | 0.141 | 0.812 | 0.232 |
| cam2 前右 | 45.0° | 6.74 | 0.173 | 0.775 | 0.206 |
| cam3 侧左 | 90.0° | 8.27 | 0.003 | 0.923 | 0.047 |
| cam4 侧右 | 90.7° | 5.86 | 0.001 | 1.015 | 0.001 |

### B. 喂三个相机（cam0/1/2）→ 同一批视角

| 视角 | 是否输入 | PSNR | SSIM | LPIPS | coverage |
| --- | --- | --- | --- | --- | --- |
| cam1 前左 44.9° | 输入 | 15.04 | 0.462 | 0.551 | 0.430 |
| cam2 前右 45.0° | 输入 | 14.50 | 0.442 | 0.550 | 0.456 |
| **cam3 侧左 90.0°** | **留出** | 10.93 | **0.206** | 0.779 | 0.328 |
| **cam4 侧右 90.7°** | **留出** | 9.36 | **0.162** | 0.773 | 0.356 |

### 结论（可直接支撑决策）

1. **单目重建几乎没有"视角外可信度"**：偏 45° 时 SSIM 只有 0.14、LPIPS 0.81；
   偏 90° 时 SSIM ≈ 0.001~0.003（等于毫无关系）。这定量解释了之前"自车一偏离轨迹画面就崩"。
2. **多视角重建能显著扩大可信域**：多喂两个 45° 相机后，**从未输入过的 90° 侧向视角**
   SSIM 从 0.003 → 0.206（约 70 倍），coverage 0.047 → 0.328。
   → "5 相机/多视角重建"是把画面变得可信的最直接手段（且 feedforward，不需要长训练）。
3. **coverage 与真实误差单调相关**（0.001→SSIM 0.001；0.047→0.003；0.206→0.17；0.33→0.21；0.43→0.46）
   → 可作为**在线截断信号**：`coverage < τ` 就截断 episode 并记录覆盖率（符合已定的 rollout 策略）。
4. **PSNR 在宽幅侧向相机上会误导**（侧向图像大片天空/路面，PSNR 反而偏高），
   评估应以 **SSIM/LPIPS + coverage** 为主。
5. **模型内参有系统性偏差**：预测 fx=529.4 vs 标定 558.7（-5.2%），与 motion-matching 得到的
   尺度比 0.946 一致 → 重建存在约 5% 的尺度/焦距误差，是"可信度"的另一个来源，值得在校准层面修。

## 六、第 2 轮（goal round 1）进展：actor 资产库落地 + 3 个坑

### A. 关键发现：重建输出的 `object_id` 是"每帧重排的簇号"，不是持续 track

实测（场景 001，8 帧）：逐帧簇号 39 个，但同一个 `object_id` 在相邻帧的世界位置会**跳 10~45m**
（10Hz 下等于 100~450 m/s 的"瞬移"）——其实是**换号**，即每帧重新聚类。
这也解释了之前 studio 里"轨迹只有 4 帧、单帧跳 15m"的现象。

→ 新增 `associate_tracks()`（SORT 式：匀速预测 + 最近邻 + 尺寸相容），把簇号关联成持续 track。
实测：**39 个簇 → 6 条持续 track**，速度 1.3 / 9.5 / 7.9 / 9.7 / 9.3 / 7.2 m/s（全部物理合理），
局部形状波动 `ext_cv` 0.09~0.39，**5/6 资产标记 usable**。

### B. 资产库 `studio/backend/actor_assets.py`

- 逐帧读 `gaussians/frame_XXXX_dynamic.ply`（局部坐标、质心≈0）+ 每帧 pose/dims；
- 多视角感知：`flat = real*num_views + view`，时间维只走 view 0，其它视角按世界位置（<1.5m）并进来；
- bbox 裁剪 + 体素去重 + 可选**体素化（solidify）**：把单目 surfel 变成体积代理，任意视角都是实体；
- 写标准 3DGS ply（原始值 logit/log）→ 直接走渲染器 `_load_sam3d_ply`，与 SAM3D 资产同 schema；
- 输出 `bank.json`（兼容 `SceneObjectAssetBank`），每个资产带 `usable/consistent/extent_cv/max_jump/速度` 等质量字段。

### C. 验证（场景 001，帧 0，对比"逐帧动态高斯"）

| 指标 | 逐帧高斯 | 资产（原始 surfel） | 资产（体素化） |
| --- | --- | --- | --- |
| 同视角还原度（vs 原渲染） | — | PSNR 34.8 / SSIM 0.989 | PSNR 28.9 / SSIM 0.963 |
| 自车视角"动态物体像素占比" | 1.21% | 1.90% | **5.13%** |
| 俯视视角"动态物体像素占比" | 0.06% | 0.29% | **0.28%** |
| 俯视可见性 | 几乎不可见（退化成条纹） | 仍偏弱（surfel 边缘朝向） | 可见（实体） |

**可摆位演示**（`asset_repose.py`）：把 track4 的车摆到自车前方 11m/右侧 1.8m，
自车视角下出现一辆**落在路面上的完整车辆**（与静态背景差异 1.18/255 ≈ 0.5% 像素），
即"按剧本把 actor 摆到指定位置并可信渲染" ✓（这正是闭环仿真需要的可控性）。

### D. 本轮踩到并修掉的坑（都很隐蔽）

1. **资产 ply 用相对路径 → 渲染器静默跳过**：`_load_sam3d_ply` 对不存在文件返回 `None`，
   整个资产"消失"但没有任何报错。→ bank 里写绝对路径 + `asset_verify` 显式告警。
2. **引擎 `extra_objects` 的 `transform` 只接受 torch.Tensor**，传 numpy 会在
   `_transform_gaussians` 里 `ndarray @ Tensor` 报错。→ 引擎侧统一 `torch.as_tensor`。
3. **gsplat 需要 ninja 做 JIT**：conda env 里装了 `ninja` 但必须在 PATH 上
   （`export PATH=/root/autodl-tmp/conda_envs/dggt/bin:$PATH`），否则 `inference.py` 直接崩。
4. 5 相机 × 8 帧 = 40 张图会 **CUDA OOM**（23.5GB 卡）→ 需按"帧×视角"预算控制（3 视角×8 帧可行）。

### E. 结论 / 下一步

- 资产库解决了**身份、位姿、可控摆位**；但**外观的视角一致性**受限于单目 surfel：
  体素化只是"能用"的体积代理。要真正好看且一致，正路是
  ①多视角重建（已验证 3 视角能把 90° 留出视角 SSIM 从 0.003 提到 0.206）；
  ②**每个 actor 走 SAM3D 重建成真 3D 资产**（主车/行人已经证明这条路可行，画质最好）；
  ③把 trust/coverage 接进 studio API 与运行时截断。

## 七、第 3 轮（goal round 2）：可信域运行时 + 多视角可复现管线 + SAM3D 资产入库

### A. 可信域运行时 `studio/backend/trust_runtime.py`（① 的运行时部分）

- **在线指标（无需 GT，任意位姿都能算）**：`coverage_map`（高斯中心投影命中网格率）
  + `pose_novelty`（相对最近训练相机：横向/纵向/高度位移 + 视线夹角）。
- **trust 模型（用留出视角真实误差标定）**：
  `SSIM ≈ y0 + y1 · coverage^p · exp(-angle/τ)`，参数由 88 个（coverage, 视角偏离, SSIM）样本拟合：

  | 模型 | R² |
  | --- | --- |
  | 仅 coverage | 0.740 |
  | **coverage + 视角偏离（两特征）** | **0.909** |

  拟合结果：`SSIM = 0.0238 + 1.4706 · cov^1.25 · exp(-ang/90°)` —— 单调、有物理含义、可审计。
  样本来自 3 条重建（1 视角 / 3 视角 / 4 视角 × 各自留出相机 × 8 或 4 帧）+ 各自参考视角锚点。
- **轨迹级判定 + 越界截断**：`trust_of_trajectory()` 逐帧给 coverage/pred_ssim/是否可信，
  连续 `patience` 帧跌破阈值就截断，输出 `coverage_report`（信任帧比例、截断帧、截断时刻的位姿偏离）。
- **可信域 envelope**：`lateral_sweep()` 把自车轨迹横向平移 N 米 → 给出 `max_safe_lateral_m`。

**API（已接入 `api_server.py`，实测通过）**：
`POST /api/trust/model`（标定，返回 R²）、`POST /api/trust/envelope`（可信域）、
`POST /api/trust/trajectory`（逐帧 trust + 截断 + 覆盖率报告；支持整体横移或"以某物体视角"）。

### B. **多视角把可信域做大（这是本轮最重要的量化结论）**

同一条自车轨迹横向平移，按 min_trust=0.55 判定"仍可信"的最大横移：

| 重建输入 | 留出相机(cam4 @90.7°) SSIM | coverage | **max_safe_lateral** |
| --- | --- | --- | --- |
| 1 视角 (cam0) | 0.0006 | 0.001 | **2.0 m** |
| 3 视角 (cam0,1,2) | 0.1620 | 0.356 | **6.0 m** |
| 4 视角 (cam0..3, 4 帧) | 0.1865 | 0.373 | **6.0 m** |

→ 多视角重建把可信域从 **2m 扩到 ≥6m（≥3×）**，同时把 90° 留出视角的 SSIM 从 0.0006 提到 0.19。
这直接支撑"用多视角重建换取闭环可信域"的路线选择。

### C. 可复现的多视角管线 `studio/backend/run_multiview.py`（③）

一条命令完成：**显存预算检查 → 多视角重建 → 留出相机 novel-view 评测 → actor 资产库**，
并写出 `run_manifest.json`（命令、退出码、耗时、产物路径、指标）。

- 预算：实测 **24 张图可行、40 张（5 视角×8 帧）在 24GB 卡上 OOM**；脚本默认卡在 24 张以内并提前报错；
- 实测：`--cameras 0,1,2,3 --holdout 4 --frames 4`（16 张）三步全部 exit 0，
  `run_manifest.json` 里 trust = {0(ref): ssim 0.614, 4: ssim 0.187}；
- 顺带修掉 `inference.py` 的多视角对比视频崩溃（`make_comparison_video_quad` 只支持 1/3 视角布局 →
  改为容错 + 其它视角数跳过，主产物不受影响）。

### D. 统一资产库（② 收尾）

`register_sam3d_assets()`：把已有的 SAM3D **真 3D 资产**登记进同一份 `SceneObjectAssetBank` schema
（`sam3d_ego` 车长 0.996 单位→4.9m、`sam3d_person_0/1` 身高 0.993 单位→1.75m，含 `model_corr`），
与 `canonical_actor` 资产（多帧拼接 + 体素化）在同一张表里，下游用同一套摆放/渲染逻辑。

## 八、第 4 轮（goal round 3）：确定性与"带可信度标注的数据"

### A. 确定性（之前唯一的反例已修）

- 行人模型选择原本是 `random.choice`（同 seed 重跑外观会变）→ 改为**首帧位姿的稳定哈希**（`zlib.crc32`），
  同一个 seed/同一条轨迹必然取到同一个 `person_*.ply`；
- 新增 `studio/backend/determinism_check.py`：跑两次生成并逐帧比较
  （roles / 采样参数 / 碰撞帧 / 受影响 track / 合成物体外观 / 逐帧位姿最大差异）。实测：

  | 场景 | 结果 |
  | --- | --- |
  | rear-end | `deterministic: true`，diffs=[] |
  | pedestrian-crossing | `deterministic: true` |
  | occluded-pedestrian-pileup | `deterministic: true`，合成行人 = `person_0.ply`（两次一致） |
  | cutin-brake-pileup | `deterministic: true` |

### B. 批量数据里的可信度标注（`annotate_trust`，默认开）

每条实例落盘的 JSON 与 manifest 现在都带：

- `trust`：`coverage_mean/min`、`pred_ssim_mean/min`、`trusted_ratio`、`first_untrusted_frame`、
  `mean_novelty_trans_m / mean_view_angle_deg`（离训练视角多远）、`model_r2`；
- `trust_envelope`：**场景级可信域**（`max_safe_lateral_m` + 各档 coverage/pred_ssim/是否可信）；
- `generator_fingerprint`：生成逻辑源码的 sha256 前 16 位（数据可溯源：同一份代码 → 同一指纹）；
- `spec_hash`：sha256(spec + 生成后逐帧位姿) 前 32 位 → **同 seed 重跑哈希完全一致**（已实测）。

批量响应新增 `trust_summary`（标注条数、平均 coverage、平均预测 SSIM、不可信实例数、模型 R²）
与 `scene_trust_envelope`。实测（1 视角重建场景）：`trust_summary = {num_annotated: 4,
coverage_mean: 0.582, pred_ssim_mean: 0.759, untrusted_instances: 0, model_r2: 0.9087}`，
`scene_trust_envelope.max_safe_lateral_m = 2.0`，两次同 seed 的 `spec_hash` 完全一致。

### C. 本阶段（P1）产出清单

| 能力 | 文件 | 验证 |
| --- | --- | --- |
| novel-view 真实误差评测（留出相机 + 标定 rig） | `novelview_trust.py` | 1/3/4 视角对照表 |
| 可信域运行时（trust 模型 R²0.909 / envelope / 截断 / 覆盖率报告） | `trust_runtime.py` + `/api/trust/*` | 实测 API 通过 |
| actor 资产库（跟踪 + 多视角并入 + 体素化 + SAM3D 统一入库） | `actor_assets.py` | 39簇→6 track；同视角 SSIM 0.96~0.99；动态像素 1.2%→5.1% |
| 资产验证/摆位演示 | `asset_verify.py` / `asset_repose.py` / `asset_turntable.py` | 摆位车辆落在路面上 |
| 多视角可复现管线（预算检查 + 重建 + 评测 + 建库 + manifest） | `run_multiview.py` | 3/4 视角跑通，5 视角 OOM 预算已记录 |
| 确定性 | `determinism_check.py` | 4 个场景 `deterministic: true` |
| 场景↔segment 匹配（拿 GT 用） | `scene_matcher.py` | scale≈1 候选 |

## 九、下一步（P2/P3，不在本阶段 goal 范围）

- **P2 传感器仿真**：可配置多相机 rig（Waymo 5 相机现成）、从仿真自车状态渲染、LiDAR 仿真（数据里有 `lidar/`）。
- **P3 闭环运行时**：gym 式 `reset/step`、NPC 控制器、episode 截断接入 trust envelope、闭环指标。
- **P4 互操作**：CommonRoad / OpenSCENARIO(+OpenDRIVE) 导出（需补 roadgraph）、分片数据集。
- 质量提升项：补 `pretrained/diffusion_model.pth` 做扩散渲染精修；每个 actor 走 SAM3D 真 3D 重建。


1. 等 Run A（`--camera_ids 0`）与 Run C（`--camera_ids 0,1,2`）跑完 → 用 `novelview_trust.py`
   测**留出相机 1/2/3/4** 的 PSNR/SSIM/LPIPS，得到"误差 ~ 视角偏离(45°/90°)"曲线；
2. 用"喂更多相机"的重建做对比 → 验证"5 相机重建提升 novel-view 可信度"；
3. 把 `coverage` 与真实 novel-view 误差做相关性标定 → 定出 trust 阈值与截断规则；
4. 补 `pretrained/diffusion_model.pth` 并对比 `-diffusion` 前后的指标；
5. actor 资产库（对象中心化动态高斯）。


---

## 第 5 轮（goal round 4）：俯视 3D 真模型 / 行人逻辑清理 / 扩散精修接入 / 前端测试面板

### ① 俯视 3D 改用真实 actor 模型

- `api_server.load_actor_bank()`：自动找该场景的 `bank.json`（`<scene>/actor_bank.json` 或
  `output/actor_assets/<scene 名>/bank.json`，带 mtime 缓存）；找到后 `render_topdown_frame` 用
  **"资产几何 + TrackManager 当前位姿"** 渲染每个物体（`extra_objects`），`include_dynamic=False`；
- 只有**没有资产**的物体才退回方框替身（`_draw_topdown_object_proxies(skip_raw_ids=...)`）；
  合成物体（SAM3D 行人/主车、事故参与者）本来就走真模型渲染；
- 取景改进：`_topdown_bounds` 有参与者时**只框参与者**，跨度上限收紧到 20m×28m；
  新增 `topdown_span`（>1 时手工指定纵向视野，米）—— 想看清车/行人就调小（实测 span=20 时主车清晰）；
- 资产与物体的对应用 `object_ids_per_frame`（(flat_frame, 簇号) → 资产），退化时按世界位置兜底匹配。

### ② 删除"用场景已有动态物体冒充行人"的旧逻辑

`corner_case._create_pedestrian_track` 不再 `_pick_donor(prefer_pedestrian=True)` +
`create_synthetic_track` 克隆外观，**只用 person_*.ply**（首帧位姿稳定哈希选模型）；没有模型直接
报错并给出 `DGGT_PED_PLYS` 提示。`_make_intercepting_pedestrian` 里为行人估尺寸的 donor 逻辑一并删除。
实测：`occluded-pedestrian-pileup` 自动合成的行人 = `person_0.ply`，尺寸 0.572×1.75×0.575 m；两次重跑
模型一致（确定性）。

### ③ 扩散渲染精修（Difix）接入 + 依赖阻塞

- `studio/backend/diffusion_refine.py`：**单例缓存**（原 `process_images_with_difix` 每帧重新加载 5GB
  权重，逐帧用不可行）、`refine_image_rgb/refine_dir`、`--status` 自检、CLI；
- `GET /api/diffusion/status` 已接入前端。
- **阻塞（环境）**：Difix 需要 `stabilityai/sd-turbo` 的 unet/vae/text_encoder 权重；
  本机 HF 缓存里只有 1.6MB 配置（unet≈1.7GB / vae≈160MB / text_encoder≈500MB 都缺），
  通过学术代理与 hf-mirror 下载都卡在 ~66MB 不动。`status` 现在按**权重体积**判定，缺就明确报出来
  （早期只看 config 会误判为可用）。修复命令已写进 status：
  `export HF_ENDPOINT=https://hf-mirror.com && python -c "from huggingface_hub import snapshot_download as d; d('stabilityai/sd-turbo')"`
  （或在有网机器下载后整目录拷到 `~/.cache/huggingface/hub/`）。

### ④ 前端：新增「⑤ 闭环仿真工具」面板

在 Corner Case 工作台里加了第 5 个 tab，可直接测试本阶段全部新能力：

- **可信域**：标定 trust 模型（显示样本数/R²/参数）、可信域扫描（逐档横移 + `max_safe_lateral_m`）、
  轨迹逐帧 trust（信任帧比例 + 越界截断帧 + 原因）；
- **actor 资产库**：一键构建（可选体素化）+ 查看状态 + 资产表（高斯数/尺寸/速度/usable）；
- **多视角重建**：填场景/相机/留出相机/帧数 → 「预算/命令」给出张数与可复制命令 →
  「后台启动」真正跑（返回 pid + 日志路径）→ 「看日志」读取日志尾部；
- **扩散精修 + 俯视取景**：依赖自检（显示缺失项与修复命令）、`topdown_span` 输入。

新增后端接口：`POST /api/actor_assets/build`、`GET /api/actor_assets/{scene_id}`、
`GET /api/diffusion/status`、`POST /api/multiview/plan|run`、`GET /api/multiview/log`。

### ⑤ 多视角显存

用户确认多视角重建已支持，只是显存可能爆 —— 现状与实测一致：`run_multiview.py` 有
`frames × views ≤ max_images(默认 24)` 的前置检查，24 张可行、40 张（5 视角×8 帧）OOM；
前端「预算/命令」按钮会先算张数再决定是否启动。


---

## 第 6 轮（goal round 5）：P2 相机 rig 传感器仿真 + P3 闭环 rollout（含越界截断）

### A. `studio/backend/sensor_rig.py`（P2：多相机 rig）

- 读 Waymo 标定：`intrinsics/{c}.txt`（fx,fy,cx,cy + k1,k2,p1,p2,k3）、`extrinsics/{c}.txt`
  （x前/y左/z上 相机系，共轭到 OpenCV 系）；
- 任意相机世界位姿：`c2w_C = c2w_ref @ inv(M_ref) @ M_C`（相对位姿与全局系无关）；
- 内参按目标分辨率缩放；`--distort` 用畸变模型构造 remap 把镜头畸变**加回**渲染图（更像真实相机）；
- CLI + `POST /api/sensor/rig`（返回每台相机的 base64 PNG + 视角偏离）。

**实测**（`scene001_cam012`，4 视角重建）：5 台相机一次渲染成功，视角偏离
**0° / 44.9° / 44.97° / 90.03° / 90.73°**（与 Waymo rig 完全一致），输出 960×640；
单目重建（`scene001_cam0`）时 90° 侧向相机基本是黑的（该重建没有那部分覆盖）——与 trust 结论一致。

### B. `studio/backend/sim_runtime.py`（P3：gym 风格 reset/step）

- `ScenarioSim.reset()/step(action)`：动作 `{speed, steer, dt}`（简易运动学自行车模型，作用在自车位姿上）
  或直接给 `{pose}`（接规划器输出）；`obs` 含**多相机图像**（复用 sensor_rig）、参与者列表（含速度/尺寸）、
  per-frame trust（coverage + 相对最近训练视角的偏离 → 预测 SSIM）；
- **终止条件**：① trust 连续 `patience` 步低于阈值（**越界**）；② 超出记录帧窗口（**不做时间外推**，
  明确截断而不是编造画面）；③ 与任一参与者 OBB 碰撞（`dggt.scene_edit.collision_physics`）；
- **相对可信阈值** `min_trust_frac`：`trusted ⟺ pred_ssim ≥ frac × 起始帧 pred_ssim`。
  动机：不同重建（1/3/4 视角）绝对画质差别很大，绝对阈值会让"质量本来就一般"的场景一开局就判越界；
- `coverage_report()`：episode 级报告（步数、信任步比例、最小/平均 coverage、最小 pred_ssim、是否碰撞、逐步历史）。
- CLI + `POST /api/sim/rollout`（脚本化 rollout，返回逐步日志 + 报告 + 可选最后 N 帧观测图）。

**实测**（4 视角重建，相对阈值 0.8）：
- 直行 `speed=8, steer=0`：21 步全部可信（trusted_ratio **1.0**，min coverage 0.563，min pred_ssim 0.721），
  到第 9 步因"超出记录帧窗口（8 帧）"正常终止；
- 大转向 `speed=12, steer=0.3`：第 3 步起 pred_ssim 掉到 0.572→0.516（阈值 0.596），**连续 2 步越界即截断**，
  报告 `trusted_ratio=0.5`，理由写清"pred_ssim < 阈值"。
  这正是闭环要的契约：环境会告诉规划器"这一步还可不可信、还能走多久"。

### C. 前端

「⑤ 闭环仿真工具」面板新增：**渲染相机 rig**（标定场景/相机/畸变开关 → 直接显示 5 台相机图像）
与 **跑闭环 rollout**（步数/速度/转向/相对阈值 → 逐步 trust 表 + 终止原因）。

### D. 下一步

- **P4 导出**：CommonRoad / OpenSCENARIO(+OpenDRIVE)。现状：预处理数据里没有 roadgraph，
  需要从 raw tfrecord 抽车道，或者先导出"最小可用"结构（动态障碍 + ego 轨迹 + 一条线性 road）
  并明确标注"地图为占位"。下一轮做。
- 扩散精修仍等 `stabilityai/sd-turbo` 权重（HF 下载在本机被限速，见第 5 轮记录）。


### E. 环境提醒：根盘曾经 97% 满（已处理）

本轮发现 **根盘 `/` (30GB) 用到 97%、只剩 1.1GB**，而 HF 缓存占 14GB（VGGT-1B 4.7GB +
sd-turbo 3GB + moge 1.2GB 等）。当时 sd-turbo 还差 ~2.5GB，继续下就会把根盘写满。

处理：
1. 删掉 sd-turbo 的**未完成分片**（2.4GB，`.incomplete` 无任何用处），根盘回到 3.4GB 空闲；
2. 把后续 HF 下载重定向到数据盘：`export HF_HOME=/root/autodl-tmp/hf_home`
   （`/dev/md0` 50GB，空闲 7GB），sd-turbo 已在该目录继续下载（`/root/autodl-tmp/hf_home/dl.log`）；
3. `diffusion_refine.sd_turbo_available()` 现在会依次查找 `$HF_HOME` / `~/.cache/huggingface` /
   `/root/autodl-tmp/hf_home` 三个位置 —— 以后缓存放在哪都能识别。

**建议**：这台机器根盘偏小，长期跑 HF 模型（VGGT/moge/sd-turbo）请统一用
`HF_HOME=/root/autodl-tmp/hf_home`，并定期清 `~/.cache/huggingface` 里不用的模型。


---

## 第 7 轮（goal round 6）：P4 标准格式导出（OpenSCENARIO + CommonRoad）

### `studio/backend/scenario_export.py`

- **OpenSCENARIO 1.2**（`.xosc`）：`FileHeader` → `RoadNetwork/LogicFile`（占位）→ `Entities`
  （每个物体一个 `ScenarioObject` + `BoundingBox`，尺寸用我们的 [宽,高,长] 正确映射到
  OSC 的 width/length/height）→ `Storyboard/Init`（`WorldPosition` 起始位姿）→
  `Story/Act/ManeuverGroup/Maneuver/Event/Action`，每个物体一条
  `FollowTrajectoryAction`（`Polyline` 顶点带 time/x/y/z/h）→ `StopTrigger`（SimulationTime 条件）；
- **CommonRoad**（`.cr.xml`）：`commonRoad` 根 + 必备标量（timeStepSize/date/author…）+
  `<dynamicObstacles>` 里每个物体 `<obstacle>`（type/shape/initialState/trajectory 的
  `<state time, position, orientation, velocity, acceleration>`）+ 一条由所有轨迹 PCA 拟合的
  **线性 lanelet 占位路网**（leftBound/rightBound/centerLine）；
- **额外**导出 `*.world.json`（世界坐标轨迹，y-up、米、含尺寸/速度/逐帧 4x4 位姿）——换地图/换工具信息不丢；
- **round-trip 校验**：把两份 XML 解析回来，逐帧比对轨迹（容差 1e-3）与实际轨迹，
  报告 `{"openscenario": {...}, "commonroad": {...}, "ok": true}`。

**坐标系与地图的诚实说明**（写进文件 `description`/`source`）：重建世界系是 y-up（x 右/y 上/z 前、米）；
OpenSCENARIO 直接写世界 x/y/z + yaw；CommonRoad 做二维映射 `X=x, Y=-z`（保持右手系），
`yaw = atan2(-fwd_z, fwd_x)`。**道路/车道是占位**（预处理数据里没有 roadgraph，要从 raw tfrecord 抽），
下游可直接替换 road/lanelet 段。

### 接口与前端

- `POST /api/export/scenario`：可选先按 `scenario_type/roles/seed` 生成事故（导出后 `undo()`，**不改动场景状态**），
  默认只导出 **ego + 参与者 + 合成参与者**（`all_tracks=false`），写 `<repo>/output/export/<name>/`；
- 「⑤ 闭环仿真工具」面板新增：事故类型 / seed / 帧数 / 是否导出全部物体 + **导出按钮**，
  结果区显示三个文件路径、每条轨迹的 role/速度/尺寸/帧数、round-trip 校验结果与"地图为占位"的说明。

### 实测

`rear-end`（3 视角重建场景，seed 7，12 帧）：导出 3 条轨迹（ego 900000、victim 1、attacker 5），
`.xosc` 11.9KB / `.cr.xml` 12.3KB / `.world.json` 20.5KB；
**round-trip 校验两份都通过**（`num_trajectories=3`、`num_obstacles=3`、`problems: []`、`ok: true`）。
CLI 也支持（`python scenario_export.py --scene ... --scenario rear-end ...`，默认只导参与者，
`--all_tracks` 导全部）。

### 扩散精修（③）状态

sd-turbo 权重仍在下载（已重定向到数据盘 `HF_HOME=/root/autodl-tmp/hf_home`，当前 3.1GB / 6-22 文件，
日志 `/root/autodl-tmp/hf_home/dl.log`）；`/api/diffusion/status` 会按**权重体积**判定并报出缺哪些文件。
下载完成后即可跑"留出视角上 精修前/后 PSNR/SSIM/LPIPS"的 A/B。


---

## 第 8 轮（goal round 7）：扩散精修的 A/B 评测管道（③ 的验证部分）

`novelview_trust.py` 新增 `--refine {none|difix|stub}`：
- 每个**留出相机**每帧渲染后，可选做一次精修，然后**同时**算 raw / refined 的 PSNR/SSIM/LPIPS，
  在报告里给出 `refined: {num, mean_psnr/ssim/lpips, delta_psnr/ssim/lpips}` —— 精修到底有没有用，用数字说话；
- `difix` 走 `diffusion_refine.refine_image_rgb`（**单例缓存**，不会逐帧重载 5GB 权重），
  未就绪时直接报错并列出缺哪些权重；
- `stub` 是一个轻量去噪+锐化，只用来**验证 A/B 管道本身**（也能作为"精修无效"的下界对照）。

**实测（stub，4 帧，留出 cam3/cam4）**：
- cam3：raw SSIM 0.2173 → refined 0.2222（**+0.0049**），PSNR +0.003，LPIPS -0.0047；
- cam4：raw SSIM 0.1204 → 0.1245（**+0.0041**）。
数值变化很小且方向合理 —— 说明 A/B 管道工作正常，接下来把 `stub` 换成 `difix` 即可得到扩散精修的真实增益。

**③ 现状（环境阻塞，已尽量推进）**：`pretrained/model_difix.pkl` 就位 ✓、单例缓存 ✓、依赖自检 ✓、
A/B 管道 ✓；唯一缺的是 `stabilityai/sd-turbo` 的 **unet 权重**（text_encoder/vae 的 fp16 已下好，
unet 仍在慢速下载中，本机下载速率 ~50MB/min 且数据盘只剩几 GB）。等权重落地后一条命令即可出真实 A/B：
`python novelview_trust.py --scene_dir <场景> --segment 001 --heldout 3,4 --refine difix`。


### 附：sd-turbo 下载的处置（重要，避免继续写满数据盘）

- 本轮尝试用 `hf_hub_download` 单独拉 `unet/diffusion_pytorch_model.fp16.safetensors`（唯一缺的大文件），
  下载速率只有 ~50MB/min，且 `HF_HOME` 所在数据盘一度到 **95%**；
- 为避免写满影响环境，**已停止下载**并清理了全部 `.incomplete` 分片（数据盘回到 88% / 6.4GB 空闲）。
  当前 `sd-turbo` 快照里只保留了 `model_index.json / scheduler / tokenizer / text_encoder(fp16) / vae(fp16)`。
- 恢复方式（任选）：
  1. 在磁盘/网络更好的机器上 `HF_ENDPOINT=https://hf-mirror.com python -c "from huggingface_hub import snapshot_download as d; print(d('stabilityai/sd-turbo'))"`，
     再把整个 `models--stabilityai--sd-turbo` 目录拷到 `$HF_HOME/hub/`；
  2. 或先腾出 ≥6GB 空间再在本机重跑同一命令；
  3. 之后 `python studio/backend/diffusion_refine.py --status` 应显示 `ready: true`，
     再跑 `novelview_trust.py --refine difix ...` 得到真实 A/B。


---

## 第 9 轮（goal round 8）：一键自检 + 前端可测（④ 收尾）

### `studio/backend/run_closedloop_demo.py`（一条命令验证整条链路）

按顺序跑：**trust 模型标定 → 可信域 envelope → actor 资产库 → 相机 rig → 闭环 rollout（直行/转向）→
导出 OpenSCENARIO+CommonRoad+world.json → 扩散精修依赖状态**，把每一步的耗时与结果写进
`<out_dir>/<场景名>/closedloop_demo_report.json`。

**实测报告（3 视角重建场景）**：

| 环节 | 结果 |
| --- | --- |
| trust 模型 | 100 样本，R²=**0.9118**（仅 coverage 0.7341） |
| 可信域 | `max_safe_lateral` = **6.0 m** |
| actor 资产库 | 7 条 track，可用 **5** |
| 相机 rig | 3 台相机，视角 0° / 44.9° / 44.97° |
| rollout 直行 (8m/s,0rad) | 6 步，可信比例 **1.0**，无截断 |
| rollout 大转向 (12m/s,0.3rad) | 4 步，可信比例 **0.5**，截断理由"预测画质 0.516 < 阈值 0.5988 连续 2 步（越界）" |
| 导出 | 3 条轨迹，OpenSCENARIO/CommonRoad **round-trip 通过** |
| 扩散精修 | 未就绪（缺 unet 权重） |

### API + 前端

- `POST /api/demo/plan|run`（一键自检的命令预览 / 后台启动）、`GET /api/report?path=`（读 output 下任意 JSON 报告）、
  `POST /api/trust/ab/plan|run`（渲染精修 A/B 的预览 / 后台启动；`refine=difix` 在依赖未就绪时**直接拦下并说明缺哪个文件**）；
- 「⑤ 闭环仿真工具」面板新增：**一键自检**（跑 + 看报告，报告以表格列出每个环节结论）与
  **渲染精修 A/B**（stub/difix/none 选择 + 留出相机 + 启动）。

### 工程注意（踩到的坑）

`gsplat` 的 CUDA 扩展首次使用会 JIT 编译（需要 `ninja` 在 PATH 上）；**两个 python 进程同时首次编译**
会在 `~/.cache/torch_extensions` 的构建锁上互相等待（表现为 GPU 0% 占用、进程卡住无输出）。
所以：后台并行跑多 GPU 任务前，先让一个进程把扩展编译缓存好（或串行启动）。


---

## 第 10 轮（goal round 9）：俯视 3D「真模型 vs 方框替身」决定性对比 + A/B 报告前端

### ① 的证据补强：`studio/backend/td3d_ab.py`

之前用整场景俯视视频对比时，车只有十几个像素，肉眼难判断。这个脚本把**资产摆到自车前方 9m**、
俯视视野收到 **15m**，再分别渲染「方框替身」和「真模型」，并排输出：

- 实测（`output/asset_verify/td3d_ab/model_vs_box.png`）：左边只有一个灰色矩形描边 +
  半透明填充（替身），右边是**真渲染出来的车身**（有车顶/侧面板/车尾结构，
  与地面的接触和阴影都正确），两者画面差异 0.0022（整图，可见差异集中在车体像素）。
- 结论：俯视 3D 里的动态物体已经是**真实模型**渲染；方框只在"该物体没有可用资产"时才出现。

### ④ 收尾：前端看 A/B 报告

「⑤ 闭环仿真工具」面板的渲染精修 A/B 区新增 **看 A/B 报告**：读 `/api/report` 后按相机列出
`PSNR/SSIM/LPIPS` 与 `精修后 PSNR/SSIM/LPIPS` 及三个 Δ，一眼看出精修有没有用。

### 又一个 gsplat 坑（重要）

`~/.cache/torch_extensions/py310_cu121/gsplat_cuda/` 里一旦留下 **`lock` 文件**（比如 JIT 编译时
进程被 kill），后续所有进程都会卡在"等待构建锁"（GPU 0%、无输出）。
处理：`rm -f ~/.cache/torch_extensions/*/gsplat_cuda/lock`；
另外设 `TORCH_CUDA_ARCH_LIST=8.6`（本机 3090）可以把首次编译从"编译所有架构"缩短到只编一个。

### ③ 再确认：仍缺 unet 权重

本轮又试了 `hf_hub_download` 单文件与 `curl -C -` 直连 hf-mirror：**连接能建立但数据不流动**
（0 字节），说明这台机器对 HF 大文件的下载被限速/阻断。已有结论：`model_difix.pkl` ✓、单例缓存 ✓、
依赖自检 ✓、fp16 加载补丁 ✓、A/B 管道 ✓（stub 实测可用），**只差 `unet/*.safetensors` 一个文件**。


---

## 第 11 轮：sd-turbo 权重到位（ModelScope 路线）+ 两个新坑

### 能用的下载路线（HF/hf-mirror 在这台机器上大文件 0 字节，ModelScope 可以）

```bash
cd /root/autodl-tmp/hf_home/sd-turbo     # 之前 git clone 下来的指针文件目录
# 关键：先把 git-LFS 指针文件删掉再下，否则会把真实数据"追加"在指针后面
for f in unet/diffusion_pytorch_model.fp16.safetensors \
         vae/diffusion_pytorch_model.fp16.safetensors \
         text_encoder/model.fp16.safetensors; do
  rm -f "$f"
  curl -L --retry 3 -o "$f" \
    "https://www.modelscope.cn/api/v1/models/stabilityai/sd-turbo/repo?Revision=master&FilePath=$f"
done
```
实测 ~12MB/s，3 个文件（1.73GB + 167MB + 681MB）约 4 分钟下完。

### 坑 1：git-LFS 指针 + `-C -` 会把文件"接"坏

`git clone`（本例走 ModelScope）只拿到 **135 字节的 LFS 指针**（`version https://...` + oid + size）。
此时用 `curl -C -` 续传，curl 会从指针的第 135 字节继续写 → 文件**大小正确**、但开头仍是指针文本，
`safetensors` 报 `Error while deserializing header: header too large`。
→ 正确做法：**先删指针再下**；`diffusion_refine.status()` 现在会真的用 `safe_open` 读一次头部
（`weights_verified`），因此"大小够了但是坏文件"不会再被误判为 ready。

### 坑 2：diffusers 0.30.3 的 `AutoencoderKL` 没有 `add_adapter`

Difix 构造时要给 VAE 挂 LoRA（`vae.add_adapter(...)`，peft API）。本机 diffusers 0.30.3 的
`AutoencoderKL` MRO 里没有 `PeftAdapterMixin`（`UNet2DConditionModel` 有），于是报
`'AutoencoderKL' object has no attribute 'add_adapter'`。
→ `diffusion_refine._load_model()` 里加了兼容补丁：把 `PeftAdapterMixin` 的公开方法补到
`AutoencoderKL` 上（不动环境、不动 `third_party`）。

### 现状

`diffusion_refine.status()` → **ready: True, weights_verified: True**（unet fp16 686 tensors、
vae fp16 248 tensors、text_encoder fp16 372 tensors，共 2.58GB，放在 `/root/autodl-tmp/hf_home/sd-turbo`）。


### ③ 真实扩散精修 A/B —— 结果（关键证据）

`novelview_trust.py --refine difix`，场景 `scene001_cam012/001`（3 视角重建），留出相机 cam3/cam4（均 ~90°），2 帧：

| 相机 | raw PSNR / SSIM / LPIPS | 扩散精修后 | Δ |
| --- | --- | --- | --- |
| **0 (参考, 重建上界锚点)** | 22.28 / 0.7298 / 0.3272 | 22.6 / 0.733 / 0.30（每帧 26.52→26.78、18.04→18.27） | ≈ **中性** |
| **3（留出，90.0°）** | 8.77 / 0.0567 / 0.8929 | **11.65 / 0.3864 / 0.7847** | **+2.88 / +0.3297 / −0.1082** |
| **4（留出，90.7°）** | 10.07 / 0.2052 / 0.7252 | **11.80 / 0.3446 / 0.6832** | **+1.73 / +0.1394 / −0.0420** |

结论：扩散精修在**覆盖很差的留出视角**上把 SSIM 从 0.057 提到 0.386（**6.8×**）、LPIPS 显著下降；
而在重建很好的参考视角上几乎不动（不破坏已有画质）——正是"补 artifact、不重画"的期望行为。
显存实测峰值 ~7.8GB（sd-turbo fp32 权重 + difix 头），24GB 卡余量充足；单次精修 0 失败。

工程适配（都写进 `diffusion_refine.py`，不动环境/不改 third_party）：
本地目录注入（把 `stabilityai/sd-turbo` 换成 `/root/autodl-tmp/hf_home/sd-turbo`）、
`variant=fp16` 但 `torch_dtype=float32`（避免 VAE 新增卷积与 fp16 混 dtype 的
`Input type (float) and bias type (c10::Half)`）、给 `AutoencoderKL` 补 peft 的
`add_adapter` 等方法与 `_hf_peft_config_loaded` 类属性（本机 diffusers 0.30.3 缺）。


---

## 第 12 轮：多视角场景"真实帧 ↔ flat 帧"索引修复（用户报"加载不了多视角数据"）

### 约定（多视角重建目录）

`ego_pose/`、`gaussians/`、`dynamic_objects/` 都是**逐视角展开（flat）**的：

    flat = real_frame * num_views + view      （view 0 = 重建时喂进去的第一台相机 = studio 的"自车视角"）

`view_XXXX.png` 是**每个真实帧一张**预览图，`view_XXXX_<view>.npy` 是每视角一份，
所以 `num_views = #view_*.npy / #view_*.png`（实测 `test3view`：24/8 = 3 ✓）。

**直接用真实帧号去读 `ego_pose` 的后果**（实测 `test3view` 自车相机朝向）：

| 读法 | 前 8 帧朝向 | 最大帧间跳变 |
| --- | --- | --- |
| 直接用真实帧号（错） | 0°, −41.8°, +44.0°, 0°, −41.8°, +44.0°, 0°, −42° | **85.8°** |
| `flat_index(real, 0)`（对） | 0°, −0°, −0°, −0°, −0°, −0°, −0.1°, −0.1° | 0.1° |

即"每帧在三台相机之间跳"，看起来就是"多视角数据加载不对"。

### 引擎本身是对的，错的是外层调用

- `DGGTRenderer._render_frame_with_object_overrides(t, ...)`：内部 `flat_index(t, 0)` ✓、
  逐视角 `flat_index(t, view)` 载入动态高斯 ✓ → **真实帧语义** ✓
- `DGGTRenderer.load_real_frame_objects(real)` ✓、`/api/scene3d` ✓ 都是对的
- 而下面这些地方把**真实帧号**直接当 flat 用（多视角必错）：
  `studio/backend/api_server.py` 的 `/api/objects`（meta 读取 + 位姿查询）、轨迹叠加绘制、渲染/编辑里的
  `_get_frame_object_pose(request.frame_idx, ...)`；以及我新写的
  `sensor_rig / sim_runtime / trust_runtime / novelview_trust / asset_verify / asset_repose / scene_matcher`。

### 修复

1. 新增 `studio/backend/scene_frames.py`（统一入口）：`num_views / num_real_frames / flat_index /
   ego_pose_path / load_ego_pose / load_ego_cam / training_cameras`；
2. 上面列的新模块全部改用它（`coverage_map` 也改成按 `flat = real * V` 取该真实帧 view0 的动态高斯/meta）；
3. 引擎新增 **`get_real_frame_object_pose(real_frame, object_id)`**（对外真实帧语义：先查按真实帧存的编辑轨迹，
   再按 view0 的 flat 帧查原始位姿），`api_server` 里 5 处调用全部换过来；轨迹叠加处也改 flat；
   老的 `_get_frame_object_pose` 保留为**内部 flat 语义**并补了文档。
4. 新增 `studio/backend/check_view_mapping.py`：用"渲染 vs GT 图"证明哪种读法对
   （`test3view`：正确读法各帧 SSIM 稳定 0.288~0.296；错误读法在 0.149~0.380 之间乱跳）。

验证（`test3view`）：`/api/objects` 与 `/api/scene3d` 现在**逐帧一致**（此前 frame1 会差 16m）；
单视角 `test1` 无回归（两者同样一致）；`sensor_rig` 在该 3 视角场景渲染出 0°/44.1°/44.8° 三台相机；
`sim_runtime` rollout 6 步全部可信（coverage ≈0.58）。

### 另一个坑：旧进程还在服务旧代码

排查中发现 **8090 上还挂着一个更早启动的 uvicorn**（新起的进程因 `address already in use` 静默失败），
于是一直在跑打补丁前的代码，表现为"代码明明改了、行为没变"。
以后改完后端代码记得确认：`pgrep -af "uvicorn api_server"` 只应有一个，且启动时间晚于文件 mtime
（`kill -9 <python 的 pid>`，不要 kill bash 包装进程）。
