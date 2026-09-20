# 文本 → 实体（LLaDA-Image + SAM3D + Qwen2.5-VL）

一句话描述要加的物体，自动把它作为可渲染、可编辑、可参与事故的"一等公民物体"插入到
4DGS 场景里，并保证它沿物理合理的路径运动、不与已有物体空间冲突。

```
文本 prompt
   │
   ▼  ①  LLaDA-Image-Turbo-FP8（文生图，白底参考图）
参考图 PNG
   │
   ▼  ②  自动掩码（HSV 去白底 + 连通域 + 填洞）
参考图 + mask
   │
   ├─▶③  Qwen2.5-VL-3B：判断类别 + 真实尺寸(长/宽/高)（VLM 推理过程会回传前端）
   │
   ▼  ④  SAM 3D Objects：单图重建 3D 高斯 .ply
  .ply
   │
   ▼  ⑤  逐轴套进 VLM 给的尺寸（宽/高/长分别对齐）
   │
   ▼  ⑥  沿"自车轨迹=车道中心线"生成轨迹（纵向距离/横向车道/速度），
   │      用 2D OBB(SAT) 对所有已有物体逐帧做冲突检测，有冲突就自动挪位
   ▼
插入 TrackManager（合成 track，自带 .ply 高斯）
```

## 为什么三个模型要"串行"

24G 单卡上：SAM3D 重建峰值约 23G，LLaDA-Image 约 7G，Qwen2.5-VL-3B 约 7G。
三个同时驻留会 OOM，所以后端每用完一个就调该微服务 `/unload`（进程退出、supervisor 重启），
再加载下一个。整体一次插入大约：

| 阶段 | 首次（含加载） | 之后 |
| --- | --- | --- |
| LLaDA 出图（4 步） | ~1-2 min | 数秒~十几秒 |
| Qwen2.5-VL 推理 | ~30-60 s | ~10-20 s |
| SAM3D 重建 | ~2-3 min | ~1-2 min |
| 规划 + 插入 | <1 s | <1 s |

## 安装（一次性）

```bash
bash text2entity/setup_env.sh
```

> 目录约定：`/autodl-fs/data` 是 200G 网络盘但 **inode 很少**（本机只剩几百个），
> 所以 **conda 环境** 放 `/root/autodl-tmp/conda_envs/llada-image`（会先清掉 conda 包缓存腾空间），
> **模型权重** 放 `/autodl-fs/data/models`（少量大文件，空间充足）。

## 启动

```bash
# 微服务（端口 8002，LLaDA-Image + Qwen2.5-VL）
nohup bash text2entity/run_service.sh > /tmp/text2entity.log 2>&1 &

# studio 后端照常
cd studio/backend && python api_server.py
```

## API

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/api/text2entity/health` | 微服务/模型目录状态 |
| POST | `/api/text2entity/generate` | 文本 → 出图 → 重建 → 规划 → 插入 |
| POST | `/api/text2entity/replan` | **修改**：把已有合成物体沿车道重排成无冲突轨迹（保留模型） |

`/api/text2entity/generate` 请求体：

```jsonc
{
  "scene_id": "...",
  "prompt": "一辆红色轿车",
  "frame_idx": 0,          // 从哪一帧开始出现
  "num_frames": 20,        // 存在多少帧
  "mode": "ahead",         // ahead(同向) | oncoming(对向) | roadside(路边停放)
  "distance": 12.0,        // 起始纵向距离(m)
  "lateral": 0.0,          // 横向偏移(m)；0 = 用 mode 的默认车道
  "speed": 0.0,            // m/s（0=静止）
  "fit": 1.0,              // 尺寸整体微调
  "use_vlm": true,
  "reference_image": null  // 可选：直接给参考图 base64，跳过 LLaDA
}
```

返回里有 `object_id`、`category`、`dimensions`、`scale_vec`、`vlm`（VLM 原始推理文本）、
`placement`（最终车道/距离/尝试记录）、`reference_image`、`preview`。

## 前端

左栏新增「文本添加实体（LLaDA-Image + SAM3D）」面板：输入描述 → 选放置方式/距离/速度/帧数
→ 「生成并插入」，成功后自动选中新物体并显示参考图。

「**重排选中物体（避碰 & 物理合理）**」按钮：对当前选中的合成物体按同样的车道/避碰逻辑
重新规划轨迹（走的 `/api/text2entity/replan`），用于"修改"已有实体而不重建模型。

## 当前状态（重要）

- **全链路已跑通**：`文本 → 出图 → 掩码 → SAM3D → 逐轴尺寸 → 无冲突轨迹 → 插入`
  已在真实场景 `output/waymo_eval/001` 上端到端验证（约 90~100s/次）：
  输入 "a red sedan car" → 前方车道插入一辆红色轿车，朝向与自车同向，0 冲突。
- **文生图后端是"可插拔"的**：
  - 首选 **LLaDA-Image-Turbo-FP8**（微服务 `/generate`）；
  - 若微服务不可用/权重缺失，自动退回**本地 sd-turbo**（`/root/autodl-tmp/hf_home/sd-turbo`，
    DGGT 仓库里 Difix 用的那份，免下载）。`nl_entity.generate_reference` 自动选择。
- **VLM 是"可选增强"**：Qwen2.5-VL 可用时给出类别/尺寸；不可用时退回
  `rule_dimensions(prompt)`（中英文关键词 → 先验尺寸），返回体里 `vlm.fallback == "rule_dimensions"`。
- **本机网络当前无法下载大模型**（实测 PyPI/ModelScope/HF 只有几~十几 KB/s，连接还会挂起），
  所以 LLaDA-Image 与 Qwen2.5-VL 的权重尚未落地；等网络恢复后跑一次
  `bash text2entity/setup_env.sh`（环境脚本已就绪）并启动 `run_service.sh` 即可自动启用。
- 掩码：生成图常有渐变底/阴影，所以优先用 **SAM 点提示**（取面积最大的显著目标、排除背景、
  填洞+最大连通域），失败才退回白底阈值。

## 实测可用方案（RTX 3090 24G，已验证）

**端到端实测**：输入「a red sedan car」→ 355s 后场景里出现一辆红色轿车（前方 14m、同向、0 冲突）。
分阶段耗时：LLaDA 出图 183s（含加载）、SAM 掩码 ~20s、Qwen2.5-VL 34s、SAM3D ~90s。

关键点（每个都是踩过的坑）：

1. **环境**：复用 `sam3d-objects` 的 torch 2.5.1 建 venv，再装
   `diffusers==0.39.0 + transformers==4.57.6 + peft>=0.17 + qwen-vl-utils + fastapi/uvicorn`。
   （PyPI 下 torch 2.8 太慢，不要用。）
2. **transformer 是 vLLM 风格 FP8**（`quant_method: fp8` + 128×128 `weight_scale_inv`），
   diffusers 0.39/0.40 都不认识 → 运行 `fix_transformer_fp8.py`：
   反量化成 bf16（13.08GB）并删掉 `quantization_config`。
3. **checkpoint 是 fused 版**（`to_qkv`/`w13`），而仓库代码是 split 版（`to_q/k/v`、`w1/w3`）
   → 同一个脚本在 dim=0 上等分拆分，并重建 safetensors index。
4. **text_encoder 必须保持 FP8**（19.39GB：float8 15.3G + fp32 尺度 4.09G）。
   仓库 pipeline 默认把同一个 dtype 传给所有组件 → `patch_llada_pipeline.py` 改成
   "DiT 组件 bf16、text_encoder dtype=None"。
5. **Ampere(3090, sm_86) 两条 FP8 限制**（服务里已处理）：
   - `torch._scaled_mm` 不支持 → `del torch._scaled_mm`；
   - FP8 tensor 不支持 `index_select` → 打了 `_patch_fp8_moe_for_ampere`（先整体反量化再取专家）。
6. **显存**：整模型 27.6GB 放不进 24G → 分段上下卡（text_encoder 编码完立刻下 CPU，
   再上 transformer/sigvq/vae）；且前向必须套 `no_grad/inference_mode`，否则计算图会多占 ~3.9GB。
7. **VLM 定尺寸/朝向**：Qwen2.5-VL-3B 输出
   `{"category":"汽车","length_m":4.5,"width_m":1.7,"height_m":1.2,"facing":"front","confidence":0.98}`；
   尺寸用于逐轴缩放，`facing=back` 时插入会翻 180°。

## 2026-09-19 修复（按反馈）

1. **替换/插入只按"高度"等比缩放，不再逐轴对齐**：逐轴把模型套进目标包围盒会把它拉伸变形；
   现在 `scale = 目标高 / 重建高`（`recon_ext[2]` 是几何高度方向），长宽比保持模型原样。
   两个入口都已改：`/api/sam3d/replace` 与 `/api/text2entity/generate`。
2. **插入物体默认有运动轨迹**：`ahead` 未给速度时自动用**自车车速**（实测 3090 上约 11 m/s），
   `oncoming` 默认 8 m/s，`roadside` 静止；面板速度填 0 即"自动跟车"。
3. **生成质量门（避免"要黄车却生成石头"）**：
   - 参考图 prompt 后缀加强（单物体、纯白底、无其它物体/文字、产品照、3/4 前视）；
   - 先一次性生成 N 张候选图（LLaDA 只加载一次），再用 Qwen-VL 逐张判断
     `matches_prompt`，选第一张匹配的（都不匹配就用第一张）；
   - 前端新增 **「参考图（可选：上传后跳过文生图）」**：直接上传一张图做 SAM3D，质量最可控。
   - 微服务调用加了传输错误重试（进程被 OOM/重启时自动等就绪再试）。

> 提示：LLaDA-Image-Turbo 是 4 步蒸馏模型，中文短提示偶尔会跑偏；要稳定结果建议
> ①上传参考图，或 ②用更具体的英文描述（如 "a yellow sedan car"）。N 越大越稳但越慢
> （每张候选约 1~2 分钟）。

## 2026-09-19 二次修复（按反馈）

1. **"自车前方(同向)"填了速度却不动**：`_Lane.point_tangent` 一开始就把弧长 `s` 夹到自车轨迹长度内，
   导致"末尾外推"是死代码——前进的物体一超出轨迹尽头就被钉住不动（对向是往回走，所以在范围内，看起来正常）。
   现已允许**两端外推**：实测 `ahead` 从第 0 帧走 20 帧位移 10.6m、从第 18 帧起也有 11.4m。
2. **重排永远失败"找不到无冲突的位置"**：同一个夹取 bug 让所有候选距离在轨迹尽头**塌缩成同一个位置**，
   于是每个候选都冲突。修复外推后，`plan_collision_free(..., ignore_id)` 实测一次即成功（`ok=True`）。
3. **自车/生成模型偏大 + 需要按模型调节尺寸**：
   - **自车**：车长改为参照"场景里已有车辆的中位车长"（clamp 4.2~5.2m），实测由 `2.08×1.54×4.9m`
     变为 `1.79×1.32×4.2m`。
   - **文本生成的车辆**：VLM 尺寸与"场景车辆中位数 / 常识(轿车 4.6m)"按 50% 融合
     （`refine_vehicle_dims`，卡车/巴士不参与），避免偏大。
   - **每个模型可等比缩放（模型 + 包围盒）**：选中任意合成物体（含主车 900000）后，
     物体面板新增「模型大小」滑块（0.5×~1.5×），走 `/api/edit/synthetic/scale`；
     该接口现在同时把 `dimensions` 按同倍数缩放，长宽高比不变。
     实测 `factor=1.2` → 包围盒 `1.79×1.32×4.2` → `2.14×1.59×5.04`。
     （原来只有 SAM3D 面板上一个作用于"最近替换物体"的滑块，现在任意物体都能调。）

## 2026-09-19 三次修复（按反馈）

1. **尺寸基准回退到 1.0**：`/api/sam3d/replace` 与 `/api/text2entity/generate` 的等比系数
   用环境变量 `DGGT_SAM3D_FIT` 控制，默认 **1.0**，即 `scale = 目标高 / 重建高`
   （**只对高度、等比缩**，长宽高比例不变、不拉伸）。
   曾把默认值试成 0.62，但反馈是**车变小并浮在空中**：模型被缩小后底面对不上包围盒底面，
   就飘了，所以回退到 1.0。要更小就调 `DGGT_SAM3D_FIT`，或选中物体后用「模型大小」
   滑块（0.5×~1.5×，等比缩放模型 + 包围盒）。
2. **SAM3D 模型没有阴影 → 加接触阴影**：重建的 `.ply` 高斯没有烘焙阴影，插进场景会"发飘"。
   `dggt_engine.py::_contact_shadow()` 在物体底部用确定性采样（固定种子 → 逐帧不闪）生成
   一圈扁平暗高斯拼成椭圆接触阴影，合成物体 / SAM3D 替换物体 / 自车都会自动带上；
   extra object 传 `"shadow": false` 可关闭。
3. **内存（cgroup ~43GB）导致的 SIGKILL**：LLaDA 常驻 RSS 约 28GB，叠加其它进程与
   page cache 会顶到 cgroup 上限被 `Killed`（日志里 `run_service.sh: line 37: ... Killed`），
   表现就是"生成一半 500 / 连接被断"。已加两道缓解：
   - `_drop_caches()`：加载 LLaDA 前丢弃可回收 page cache（约 9GB）腾余量；
   - `_free()` 里 `malloc_trim(0)`：把 glibc 已释放但未归还内核的内存交回去。
   另外 `nl_entity._svc_post` 对进程重启做了带退避的重试。

## 语言驱动轨迹编辑（`/api/traj_edit/{plan,apply}`）

- `POST /api/traj_edit/plan`：`{scene_id, instruction, frame_idx, num_frames}` →
  `{source:"llm"|"rule", ops:[...], raw, summary}`（只解析、不改场景）。
- `POST /api/traj_edit/apply`：同上，另可传 `ops` 直接给结构化操作（跳过 LLM）→
  `{source, ops, inserted, report, raw}`，每步 op 都有 `ok/error` 回执。
- `POST /llm`（微服务 8002）：纯文本推理，复用常驻的 Qwen2.5-VL-3B。

`traj_llm.py` 负责解析与执行；LLM 不可用时退化为 `rule_plan` 关键词兜底。
`insert` 复用本文件的 LLaDA+SAM3D 流水线：`api_server` 先跑完 `insert` 拿到新 track id，
再把 `collide` 里写成 `null/0/-1` 的占位替换成它，所以
「生成一辆红色轿车变道撞上 T:2」可以一句话同时生成车辆**和**它引发事故的轨迹。

### 只允许跑一个 supervisor（重要）

`run_service.sh` 是一个 `while true` 监督循环（`/unload` 让进程退出后自动重启）。
如果不小心启动了多个（例如重复执行 `nohup bash run_service.sh &`），会有两个后果：

- 多个 python 抢 8002 端口 → 日志出现 `[Errno 98] address already in use`，服务反复起停；
- 更糟的是两个进程同时加载 LLaDA，**内存翻倍**，直接顶爆 cgroup 被 OOM Killer 干掉
  （表现为"生成到一半连接被断 / 500"）。

正确做法：启动前先确认只有一个。

```bash
pgrep -af run_service.sh          # 应该只有 1 个
pkill -f run_service.sh           # 多了就全清掉
pkill -f text2entity_service.py
cd /autodl-fs/data/dggt-main && nohup bash text2entity/run_service.sh > /tmp/text2entity.log 2>&1 &
```

另外 `/llm` 现在会在 processor 文本编码失败时回退到 tokenizer，并把 traceback 打到日志；
`traj_llm._llm_text` 失败会重试 3 次并打印原因，最终退回关键词规则解析（仍然可用）。

## 2026-09-19 四次修复（按反馈）

### 1) 语言轨迹编辑与"生成 corner case"共用同一套事故逻辑

以前两边各写了一套：corner case 走 `scenario_engine` + `corner_case._simulate_pursuit_collision`
（限加速度追击 + OBB-SAT 碰撞 + 动量冲量），而语言轨迹编辑的 `collide` 只是"把肇事车
平滑插值到受害车位置"——结果碰撞帧两车中心重合、**明显穿模**，而且和 corner case 的判定
不一致，先后用两种功能改同一段轨迹就会互相打架。

现在 `collide` 直接调 `corner_case.simulate_pair_collision()`（corner case 的 `prim_pursue`
走的是同一个 `_simulate_pursuit_collision`），一次完成：限加速度追击 → 逐帧 OBB-SAT 接触判定
（用各车**当前**包围盒）→ 非弹性冲量（受害车被撞开）→ 撞后摩擦减速 → **逐帧防穿模分离**。

`separate_pair()` 是新增的共用防穿模：沿 SAT 的**最小平移向量（MTV）**把两车推到刚好接触，
按质量分配位移、只平移不改朝向。关键细节：

- 必须按 **MTV 穿透深度**推，不能用支撑距离（`_obb_radius_along`）。斜碰时支撑距离会
  **低估**，推开后仍然重叠——实测残留 7cm；改用 MTV 深度后 645 组重叠用例**全部消除**
  （最大残余 0.00000 m）。
- 逐帧而不是只处理第一次接触：首次碰撞之后肇事车仍可能比受害车快、再次贴上。
- 竖直方向不算穿模，轴投影到 XZ；投影后退化就用两心水平连线。

**幂等**：`_op_collide` 记录这对车"事故前/事故后"的逐帧位姿；再点一次"应用"时若轨迹还是
上次事故后的样子，就先还原再重放，结果与第一次**逐帧完全一致**（0.000000 m）；若中间被手动
改过，则从当前状态重新仿真（"用新状态再撞一次"）。

**变道 + 撞一次成型**：`fuse_collide_ops()` 把同一条指令里的 `lane_change` 横移量作为仿真的
`attacker_lateral`/`victim_lateral` 传进去，并把被融合的 op 标记为不再单独执行——否则追击
仿真会把刚写好的横移整段覆盖掉（corner case 里同样的坑）。

实测（场景 `waymo_eval/001`，T:1/T:2 为两辆真实车）：

| 步骤 | 结果 |
| --- | --- |
| corner case `rear-end` | collision_frame=8，两车最大穿透 **0.0000 m** |
| 接着用语言 `collide` | `engine=corner_case.simulate_pair_collision`，穿透 **0.0000 m** |
| `lane_change` + `collide` 融合 | 横移被融合，穿透 **0.0000 m** |

### 2) 替换后包围盒紧贴模型（碰撞判定才可信）

新增 `_tight_model_dims()`：按重建模型自己的长宽高算包围盒，`bbox_mode: tight_to_model`。
换算与渲染顺序严格一致（`dggt_engine` 是先 `model_corr`、再在**物体局部系**里乘缩放）：

```
局部包围盒[宽,高,长] = (|model_corr| @ canonical_ext) * 缩放
```

`.ply` canonical 轴为 x=右(宽) / y=前(长) / z=上(高)，`model_corr` 把它旋到局部
X=宽 / Y=高 / Z=长，所以典型情况就是 `[ext_x, ext_z, ext_y] × scale`。每轴 10cm 下限，
避免个别退化重建（例如只有几 cm 厚）让碰撞体变成"纸片"而永远算不出碰撞。
响应里同时给 `dimensions`（紧贴模型）与 `target_dims`（原物体，仅参考），
`/api/text2entity/generate` 的无冲突放置也改用紧贴尺寸。

### 3) 顺带修掉一个会造成"误判碰撞"的 SAT bug

`BoundingBox.get_axes()` 返回旋转矩阵，而调用方 `axes.extend(...)` 会把它按**行**展开；
但盒子的三个主轴是旋转矩阵的**列**（角点是 `R @ corners_local.T`）。4000 组随机位姿实测：
按行展开约 **0.5%** 的假碰撞（真实已分离却判成碰撞），按列则与参考实现完全一致。
已同时新增 `separating_axis()`（返回最小平移轴）供防穿模使用。
