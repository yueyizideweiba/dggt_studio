# SAM 3D 接入说明

DGGT 后端已接入 **SAM 3D Objects**（单图 + 掩码 → 3D Gaussian Splat `.ply`）。
采用**独立 conda 环境 + 独立 HTTP 微服务**，避免与 dggt 后端（torch 2.4.1 / gsplat 1.5.3）冲突；
并在 dggt 后端加入 **SAM（Segment Anything）交互式分割**，形成「点选前景/背景点 → 分割 →
SAM 3D 重建 → 替换已有动态物体」的闭环。

```
┌────────────────────────────────────────────┐   HTTP(8001)   ┌──────────────────────────┐
│  DGGT 后端 (dggt 环境) :8000               │ ─────────────▶ │  SAM 3D 微服务 :8001     │
│  ├ api_server.py  渲染 + 接口              │  image+mask     │  sam3d_service.py        │
│  ├ sam_segment.py  SAM 点提示分割          │ ◀─────────────  │  sam3d-objects 环境       │
│  ├ sam3d_client.py HTTP 客户端             │  .ply 路径      │  懒加载 ~13GB 权重         │
│  └ dggt_engine.py 渲染导入的 .ply          │                 │  /unload 释放显存         │
└────────────────────────────────────────────┘                 └──────────────────────────┘
```

## 已验证状态（RTX 3090 24G / AutoDL）

- ✅ 环境：torch 2.5.1+cu121、pytorch3d 0.7.8、flash-attn 2.8.3、kaolin 0.17.0、gsplat 1.5.3、moge、spconv 全可导入。
- ✅ SAM 交互分割：`/api/sam/segment` 点提示 → 掩码（score≈0.91）。
- ✅ SAM 3D 重建：demo 图 → `num_points≈3e5` 的 `.ply`，显存峰值 ~23.4G。
- ✅ 替换：`/api/sam3d/replace` 完成「分割 → 重建 → 按目标物体尺寸缩放(scale=3.18) → 删除原物体 → 挂载新物体 → 卸载模型」。
- ✅ 渲染验证：替换后物体列表只剩新物体，位姿/尺寸与目标物体对齐。

## 安装（一次性）

```bash
cd /root/autodl-fs/dggt-main
bash sam-3d-objects/setup_sam3d_env.sh   # SAM 3D 独立环境 + 预下载 DINO/MoGe 权重
```

SAM 交互分割（dggt 后端）依赖：

```bash
# segment-anything + ViT-B 权重（~375MB）
/root/autodl-tmp/conda_envs/dggt/bin/python -m pip install \
  -i https://repo.huaweicloud.com/repository/pypi/simple segment-anything
curl -sL -o /root/autodl-tmp/sam_vit_b_01ec64.pth \
  https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
```

## 启动

```bash
# 1) SAM 3D 微服务（端口 8001）
bash sam-3d-objects/run_sam3d_service.sh

# 2) dggt 后端（照常，需激活环境使 ninja 在 PATH 中）
conda activate /root/autodl-tmp/conda_envs/dggt
cd studio/backend && python api_server.py
```

## 前端使用（交互式逐物体重建）

1. 加载场景后，在「SAM 3D 交互重建」面板点 **「开始交互分割」**。
2. 在「替换目标物体」下拉框里选择要替换的已有动态物体。
3. 在画布上点选：**左键 = 当前类型点**（前景绿/背景红，用「前景点/背景点」按钮切换），
   **右键 = 撤销上一点**；掩码实时预览。
4. 点 **「生成（替换目标物体）」**：后端分割 → SAM 3D 重建 → 按目标物体大小/位置对齐替换。
5. 点 **「添加物体（下一个）」**：清空点，选择下一个目标物体，重复 3-4。

## API

### 交互分割 / 替换（dggt 后端）

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| POST | `/api/sam/source_image` | `{scene_id, frame_idx, max_dim}` → 高清源帧（长边 1536）base64 |
| GET | `/api/sam/models` | 列出可用分割模型（vit_b/vit_l/vit_h）+ 当前模型 |
| POST | `/api/sam/model` | `{model_type}` → 切换分割模型 |
| POST | `/api/sam/segment` | `{scene_id, frame_idx, points[[nx,ny]], point_labels[0/1]}` → `{mask(base64), score}` |
| POST | `/api/sam/frame_candidates` | `{scene_id, target_object_id, max_candidates, stride, frame_range?}` → 各帧“遮挡最少/可见面积最大”评分（多帧选源图用） |
| POST | `/api/sam3d/replace` | `{scene_id, frame_idx, target_object_id, points, point_labels}` → 分割+重建+替换，返回 `{object_id, scale, score}` |
| POST | `/api/sam3d/preview` | `{scene_id, object_id, num_frames, size}` → 物体旋转预览多帧 base64 |
| GET | `/api/sam3d/objects/{scene_id}/{object_id}/download` | 导出 .ply |

### 车头自动朝向运动方向（更真实的交通场景，**默认开启**）

该功能**默认开启并作用于全部动态物体**，前端不需要"点击应用"这一步；前端只有在读回配置发现
被关掉时才会自动重新开启。面板只剩一个「平滑」滑杆 + 状态文字。

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| POST | `/api/edit/auto_heading` | `{scene_id, enabled, smoothing(0~1), window, sustain, track_ids?, all_tracks?, reset?}` |
| GET | `/api/edit/auto_heading/{scene_id}?frame_idx=` | 读取配置 + 各动态物体当前目标朝向与推断出的车头轴 |

实现要点：

- 顺序为 **基础轨迹 → 车头朝向运动方向 → 用户旋转偏移（360°）**，所以真实高斯物体、
  SAM3D 替换物体、事故合成参与者一律生效，用户手动旋转仍叠加在上面。
- 方向估计：±`window` 帧（默认 6 ≈ 0.6s）窗口内对轨迹中心做 **PCA 主轴**拟合（比端点差稳健，
  抗单帧跳变），主轴散布 < 0.4m 视为不可信 → 保持上一朝向；整段位移 < 0.5m 的物体
  （杆/静止车/锥桶）不做自动朝向。
- 平滑两步：先对方向**向量**在 ±`round(smoothing*6)` 帧内平均（去抖、天然处理角度环绕），
  再做**限速转向**——每帧最多转 `4+46*(1-smoothing)` 度（平滑 0.6 → 约 22°/帧），
  且目标方向与当前朝向相差 >100° 时必须连续 `sustain` 帧（默认 4）都指向新方向才接受，
  避免轨迹噪声导致来回掉头。只替换偏航角，保留俯仰/滚转。
- **车头是 +Z 还是 -Z 逐物体不同**（数据集里所有 `pose_world` 旋转都是单位阵，4D 重建把每个
  物体 canonical 化到世界轴，正负号随机）。逐个推断：SAM3D 重建物体按 `model_corr` 认 +Z；
  克隆合成参与者沿用 donor；真实物体用**不可变的原始位姿**——净位移在原始 +Z 轴上的投影符号。
- 个别物体方向仍相反时，用物体面板的 **「翻转180°」** 单独修正（走全局旋转偏移，可持久化）。

### 合成物体的轨迹编辑（修复"拖了又复原/冒出两个模型"）

`TrackManager.set_track_pose` / `set_track_trajectory` 现在按类型分流：

- **合成物体**（SAM3D 替换物体、事故合成参与者）：直接写入它自己的 `synthetic_tracks[sid]["poses"]`。
  以前统一写 `track_edits`，而合成物体的 `_get_track_pose_base` 只读 `poses`，
  所以编辑被静默忽略 → 表现为"拖拽轨迹节点后自动复原"。
- 真实 track：仍写 `track_edits` 位姿关键帧。
- 两者写入前都会**去掉用户全局旋转偏移**（`_strip_user_rotation`），读时重新叠加，
  避免反复编辑把旋转累加进去。
- `drag_point_adaptive` 允许在**插值帧**上新增轨迹点（合成物体只在目标轨迹的稀疏帧上有位姿，
  但渲染时插值出现在所有帧）。
- 被 SAM3D 替换掉的原始物体记入 `track_replaced`：`build_object_overrides`/`get_frame_objects`
  一律隐藏它，编辑它不会被 `track_deleted.discard` 复活，编辑接口直接返回 **409** 并提示改用替换后的新物体。
  前端在替换成功后会**自动选中替换后的新物体**，避免后续操作落在旧物体上。

### 清除事故轨迹不会删掉被替换的物体

`POST /api/corner_case/clear` 对每个 track 分三类处理：

| track 类型 | 处理 |
| --- | --- |
| 真实 track | `clear_track_edits` → 回到原始轨迹 |
| **SAM3D 替换物体（持久合成物体）** | `reset_synthetic_to_base` → **保留模型**，轨迹还原到替换时刻 |
| 事故自动合成的临时参与者（克隆外观） | `remove_synthetic_track` → 整个移除 |

`add_sam3d_track` 会额外存一份 `base_poses`（替换时刻的原始轨迹），`reset_synthetic_to_base`
用它还原；`_copy_track` 也会深拷贝 `base_poses`，所以撤销/重做同样安全。

另外 `/api/corner_case/generate` 会先调用 `snapshot_synthetic_poses()` 备份持久合成物体的**当前**轨迹，
`clear` 优先用这份备份还原（`restore_synthetic_from_corner_backup`），因此
**"清除事故轨迹"只撤销事故造成的轨迹改动，不会把用户之前手动调好的轨迹一起清掉**。

### 动态物体 360° 旋转（持久，播放时保持）

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| POST | `/api/edit/object/rotation` | `{scene_id, track_id, yaw/pitch/roll}`（绝对）或 `delta_yaw/...`（叠加）或 `reset:true` → 设置全局旋转偏移 |
| GET | `/api/edit/object/rotation/{scene_id}/{track_id}` | 读取当前 yaw/pitch/roll |

实现：TrackManager 为每个 track 保存一个 3x3 **局部旋转偏移** `track_rotations[track_id]`，
`get_track_pose` 在返回前应用 `pose[:3,:3] = pose[:3,:3] @ R_offset`。因为是全局偏移，
**作用于该物体所有帧**，所以播放时不会回退；且对真实高斯物体和 SAM3D 替换物体同样生效。
前端选中物体后面板有 **朝向/俯仰/翻滚 滑杆 + 翻转180° + 重置朝向**，以及画布旋转工具。

⚠️ **前端缓存**：2D 的 `state.renderCache` 与 3D 视图的 `frameCache` 都是"按帧缓存已渲染图像"，
任何改变渲染结果的编辑（旋转/位移/删除）之后**必须**调用 `studio._invalidateRenderCaches()`
（内部清空 2D 缓存并调用 `viewer3d.invalidateRenderCache()`），否则会直接命中旧图像，
表现为"设置了旋转但画面自动回到原状态/完全没反应"。
3D 视图的画布旋转工具提交时也走 `/api/edit/object/rotation`（**而不是**逐帧的 `/api/edit/object/matrix`），
这样旋转才是全局持久的（逐帧编辑只改当前帧，一播放就回到原朝向）。

### 基础 SAM 3D（dggt 后端，前缀 `/api/sam3d`）

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/api/sam3d/health` | 探测微服务健康状态 |
| POST | `/api/sam3d/reconstruct` | multipart `image`(+`mask`/`seed`/`format`) → `ply_path` + `local_ply_path` |
| POST | `/api/sam3d/import` | JSON `scene_id` + `ply_path` + `pose_world` + `scale` → 导入场景 |
| GET | `/api/sam3d/objects/{scene_id}` | 列出已导入物体 |
| DELETE | `/api/sam3d/objects/{scene_id}/{object_id}` | 移除已导入物体 |

### 微服务（端口 8001）

`GET /health`、`POST /reconstruct`、`POST /unload`、`GET /outputs/{filename}`。

## 前端交互要点（本次新增）

- **替换目标物体支持视图点选**：SAM 面板点「在视图中点击选择目标」后，在 2D 画布或 3D 视图中
  点击任意物体即把它设为替换目标（`viewer3d.onObjectSelected` / 2D `mousedown` 里调用
  `_sam3dFinishPickTarget`）。
- **源图支持多帧选择**：源图模态框顶部有 `← 上一帧 / 帧 N / 下一帧 →` 与「跳到遮挡最少的帧」，
  下面一排 chip 是 `/api/sam/frame_candidates` 给出的推荐帧（★ = 最优，chip 上显示遮挡百分比）。
  切换帧会重新高清渲染该帧原图并清空已点选的点；**分割与重建都用所选帧**（`this.samSrcFrameIdx`），
  不再固定用当前帧。单图重建无法处理遮挡，所以建议挑遮挡少、面积大的帧。
- **车头朝向面板**：左栏「车头朝向（自动跟随运动方向）」有开关、平滑滑杆（0~1）、
  「应用到全部物体 / 仅选中物体」。平滑 0.6 时每帧最多转约 22°，越大越慢越平滑；
  0 表示不限制（直接跟随，可能突然掉头）。
- **SAM3D 连续重建**：`/unload` 会让微服务进程退出并由 supervisor 重启，所以客户端
  `reconstruct` 前会先 `wait_ready()` 轮询 `/health` 并带重试，连续替换两个物体不会再出现
  "Server disconnected"。


## 关键实现点

- **高清源图**：`/api/sam/source_image` 把相机原生帧（如 518×350）等比放大到长边 1536（内参同步放大），
  分割与重建都用这张高清图，显著改善质量（score 0.91→0.94）。
- **坐标 1:1**：前端在独立模态框里按源图原始分辨率显示，点击坐标 `像素=(click/显示宽)*源图宽` 直接映射，无拉伸/缩放偏差。
- **格式转换（黑块修复）**：SAM 3D `save_ply` 存的是**未激活值**（opacity=logit、scale=log、f_dc=未激活 SH），
  而 DGGT `_load_ply` + gsplat(`sh_degree=None`) 期望**激活值**。`_load_sam3d_ply` 里做
  `sigmoid(opacity)`、`exp(scale)`、`sigmoid(f_dc)` 转换，否则会渲染成黑块。
- **替换对齐**：`_ply_bbox_and_orientation` 计算重建物体质心 + 最大边长 + 朝向修正；
  `scale = 目标物体最大边长 / 重建物体最大边长`，质心归零后缩放/摆位。
- **朝向修正（模型修正 model_corr）**：SAM3D canonical 相机系里物体"上"(图像上方) = -Z、
  朝向(远离相机的前方) = -Y、右 = +X；DGGT 物体局部系是 X=宽、Y=上、Z=前。
  修正矩阵为 `R = Ry(180°)·Rx(+90°) = [[-1,0,0],[0,0,-1],[0,-1,0]]`（纯旋转，det=+1）：
  先用 `Rx(+90°)` 把 -Z(上)→+Y(上) 摆正上下，再叠加 180° yaw 把 -Y(前)→+Z(前) 摆正前后。
  **该修正只作用于 gaussians**（`model_corr`），不影响包围盒（包围盒用目标物体
  局部尺寸，避免被二次旋转）。单张图无法区分车头/车尾，若某个物体仍反了，
  点面板里的 **「翻转180°」** 一键翻转，或用滑杆手动微调（会持久化）。
- **一等公民物体**：替换后物体注册为 TrackManager 的合成 track（自带 .ply 高斯），
  因此自动拥有**包围盒、可选中、运动轨迹（沿原物体轨迹）、可编辑（位姿/轨迹/删除）**，
  与普通动态物体完全一致。
- **显存管理**：分割用 SAM ViT-B（后端，重建前 `release_memory` 释放）；重建后后端调微服务 `/unload`，
  微服务卸载后**退出进程、由启动脚本监督循环重启**，把 spconv/warp 的 CUDA 内存池彻底归还。
- **gsplat 编译**：`api_server.py` 顶部按当前 GPU 自动设置 `TORCH_CUDA_ARCH_LIST`，首次渲染只编本机架构（~30-60s），避免编译所有架构卡死。

## 踩坑记录（脚本已修复，勿重复踩）

| 现象 | 根因 | 修复 |
| --- | --- | --- |
| `ModuleNotFoundError: numpy` | 原脚本 `conda activate` 中途退出 | 改用绝对路径 `${ENV}/bin/python -m pip` |
| 阿里云源 0.6MB/s | 镜像限速 | 主源换华为云 |
| pip 下载 `torch-2.14.0`(疑 CPU 版) | requirements 未 pin torch | 先显式装 `torch==2.5.1+cu121` 三件套 |
| `cuda_runtime_api.h: No such file` | conda CUDA 头在 `targets/x86_64-linux/include` | `export CPATH=$ENV/targets/x86_64-linux/include` |
| MoGe `401 ... cas-server.xethub.hf.co` | huggingface_hub 默认 Xet 后端 | `HF_HUB_DISABLE_XET=1` + 预下载权重 |
| DINO `RemoteDisconnected` | torch.hub 需 github.com | run 脚本内置 `source /etc/network_turbo` |
| dggt 渲染 `Ninja is required` | gsplat JIT 首次需编译，PATH 缺 ninja | `conda activate` 使 `${ENV}/bin` 在 PATH |
| 替换成黑块 | SAM3D ply 是未激活值、gsplat 期望激活值 | `_load_sam3d_ply` 做 sigmoid/exp 转换 |
| 替换成一坨巨大黑球 | scale 已转成线性后，渲染仍用 `+log(scale)` 做对数缩放 | 改为线性 `× scale` |
| 首次渲染卡死/占满内存 | gsplat 编译所有 CUDA 架构 | 自动设 `TORCH_CUDA_ARCH_LIST` 为当前 GPU 架构 |
| 点选位置偏移 | 画布拉伸 + zoom/pan 未计入 | 独立源图模态框 1:1 像素映射 |

## 已知限制

- SAM 3D 重建峰值 ~23.4G；与后端渲染共享 24G GPU。重建时后端已释放 SAM、微服务重建后卸载模型并重启，
  GPU 内存可彻底归还（实测 19.7G→3 MiB）。
- 重建结果无真实世界尺度，但替换流程会按目标物体尺寸自动等比缩放，无需手工调。
- 首次 `/reconstruct` 加载 ~13GB 权重约 2-3 分钟；首次后端渲染会编译 gsplat kernel（约 30-60s）。
- 当前分割用 SAM ViT-B（快、省显存）；如需更高精度可换 ViT-H/L（约 1.2~2.4GB 权重）。

## 本次修复（替换尺寸 / T0 目标）

### 替换后比原模型"大一圈"
- 旧实现用**高斯中心的 min/max bbox** 作为重建物体的尺寸，并且只按**最长轴等比缩放**。
  源 4DGS 物体的包围盒比真实车型"扁/窄"得多（例如 3.3m 长的车只有 0.85m 高），
  等比对齐最长轴后，重建模型在宽/高上会大 20%~30% → 看起来"大一圈"。
- 现在改为**逐轴对齐**：`scale_vec = [目标宽/重建宽, 目标高/重建高, 目标长/重建长]`
  （在 `model_corr` 之后的物体局部系里作用），重建模型正好套进目标物体的包围盒。
- 尺寸测量本身也从"中心 bbox"改成**可见轮廓**（中心 ± `sigma_k`×高斯尺度，尺度用 99.5%
  分位截断防离群）；`sigma_k` 可用 `DGGT_SAM3D_SIGMA_K`（默认 2.0）调整。
- 面板新增 **「替换模型大小」滑块**（替换成功后出现）：调 `/api/edit/synthetic/scale`
  的 `factor`（相对自动尺寸，1.0=自动值，<1 更小），实时生效。
- 也可用 `DGGT_SAM3D_FIT`（默认 1.0）给**所有**替换一个整体缩放系数；
  `/api/sam3d/replace` 支持请求体里的 `fit` 覆盖它。

### T0（track_id=0）选不了
- 前端用 `parseInt(...); if (!targetId)` 判断是否选了目标，`parseInt('0')===0` 是 falsy，
  于是 T0 被误判成"没选物体"，弹出"请先在左侧选择要替换的目标物体"。
- 已改为 `_sam3dSelectedTargetId()`：空串/NaN 才算没选，**0 是合法 track_id**。
  同一处 falsy 判断（旋转/翻转/重置朝向）也一并修正。
- ⚠️ 前端 HTML 里 `app.js?v=` 的版本号已更新；如果仍看到旧行为，请**强制刷新**（Ctrl+F5）。


## 更正（2026-09-19）：不再逐轴对齐，改为"只对齐高度"

逐轴把重建模型套进目标包围盒虽然能让三轴尺寸都吻合，但会**拉伸变形**（看起来很奇怪）。
按用户反馈已改为：**只按高度等比缩放**（`scale = 目标高 / 重建高`），保持模型原本的长宽高比例。
`/api/sam3d/replace` 与 `/api/text2entity/generate` 均已如此；面板「替换模型大小」滑块照旧可用。
