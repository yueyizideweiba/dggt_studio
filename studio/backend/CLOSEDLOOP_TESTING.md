# 闭环仿真功能测试指南（DGGT studio）

> 面向"我要亲手验证这些能力"的场景：先看**前端面板**（可视化、点按钮），再看**命令行**（可脚本化、可复现）。
> 括号里的数字都是本机实测值，可作为对照。测试数据用重建场景 `output/trust/scene001_cam012/001`（3 视角重建，8 帧）。
> 标定文件来自 `data/waymo/processed/validation/001`（Waymo 5 相机）。

---

## 0. 启动服务

```bash
# 后端（API）。生产你用 8000；测试用 8090 互不影响
cd /root/autodl-fs/dggt-main/studio/backend
export PATH=/root/autodl-tmp/conda_envs/dggt/bin:$PATH
python -m uvicorn api_server:app --host 0.0.0.0 --port 8000

# 前端静态页（已有一套跑在 3000）
cd /root/autodl-fs/dggt-main/studio/frontend
python -m http.server 3000
```

浏览器打开前端 → 加载一个重建场景 → 打开 **Corner Case 工作台** → 切到 **「⑤ 闭环仿真工具」** 标签
（前端有改动时记得 **Ctrl+Shift+R 硬刷新**）。

> 注意：后端是单进程单卡，**不要同时**在浏览器点两个 GPU 重任务（如"后台启动多视角"+"一键自检"）；
> 另外 `gsplat` 首次使用会 JIT 编译，若残留 `~/.cache/torch_extensions/*/gsplat_cuda/lock`
> 会让进程卡住（表现：GPU 0%、无输出）→ `rm -f ~/.cache/torch_extensions/*/gsplat_cuda/lock`。

---

## 1. 前端面板：逐项怎么点、该看到什么

### 1.1 可信域（novel-view trust）
| 操作 | 期望结果（实测） |
| --- | --- |
| 点「标定 trust 模型」 | 显示样本数与 R²；实测 100 样本、**R²=0.9118**（只用 coverage 时 0.7341），并给出模型式 `SSIM ≈ y0 + y1·cov^p·exp(-ang/τ)` |
| 「扫描可信域」 | 给出每档横移的 coverage/预测 SSIM/是否可信，以及 **max_safe_lateral_m**；3 视角重建实测 **6.0 m**（1 视角只有 2.0 m） |
| 「轨迹逐帧 trust」 | 逐帧表 + 信任帧比例；越界时给出**截断帧与原因** |

意义：这是"仿真画面还可不可信"的量化判据；横移档位越大越接近重建覆盖的边界。

### 1.2 actor 资产库（俯视 3D 用真模型的前提）
| 操作 | 期望结果（实测） |
| --- | --- |
| 勾"体素化成体积代理" + 「构建 actor 资产库」 | 返回 track 数/可用数/资产表；实测 7 条 track、**5 条可用**（高斯数、尺寸、速度、usable） |
| 「查看状态」 | 显示该场景是否已有 `bank.json`（俯视 3D 会自动用它渲染真模型） |

构建完成后，**俯视 3D 视频里的动态物体就是真模型**（缺资产的才退回方框）；
想看清车/行人，把「俯视视野(m)」调到 15~20 再录。

### 1.3 多视角重建（显存预算 → 后台跑 → 看日志）
| 操作 | 期望结果（实测） |
| --- | --- |
| 填相机 `0,1,2`、留出 `3,4`、帧数 8 → 「预算/命令」 | 算出 `24 张图 / 上限 24 → 可以跑`（3 视角×8 帧）；给出一条完整命令 |
| 「后台启动」 | 返回 pid + 日志路径；「看日志」可看进度 |
| 帧数改 8、相机 `0,1,2,3,4` | 40 张 > 24 → 会提示会 OOM（本机 24GB 实测 40 张必 OOM） |

> 取舍：视角越多 → 留出相机画质越好、可信域越大（90° 留出 SSIM 0.0006→0.162→0.187；可信域 2m→6m），
> 但显存吃紧；`frames × views ≤ 24` 是 24GB 卡的安全线。

### 1.4 相机 rig 传感器 + 闭环 rollout
| 操作 | 期望结果（实测） |
| --- | --- |
| 「渲染相机 rig」（相机 `0,1,2,3,4`，可选"加镜头畸变"） | 直接显示多台相机图像 + 视角偏离；实测 **0° / 44.9° / 44.97° / 90.03° / 90.73°** |
| rollout：速度 8、转向 0、相对阈值 0.8 | 逐步表 + 报告：**6 步全可信（比例 1.0）**，到"超出记录帧窗口"正常结束 |
| rollout：速度 12、转向 0.3 | 第 3~4 步预测画质跌破阈值 → **截断**并写明原因（比例 0.5） |

`相对阈值` = × 起始帧的预测 SSIM；因为不同重建（1/3/4 视角）绝对画质差别很大，用绝对阈值会把"质量本来就一般"的场景一开局就判越界。

### 1.5 导出标准格式（P4）
| 操作 | 期望结果（实测） |
| --- | --- |
| 事故类型 `rear-end`、seed 7、帧数 20 → 「导出 .xosc + .cr.xml」 | 3 条轨迹（ego + 攻击车 + 被撞车），**round-trip 校验通过**；给出三个文件路径 |

文件：`output/export/<name>/<name>.xosc`（OpenSCENARIO 1.2）、`.cr.xml`（CommonRoad）、`.world.json`（世界坐标轨迹）。
**注意**：预处理数据里没有 roadgraph，所以导出里的道路/车道是**占位**（文件 `source`/`description` 已标明）；轨迹/尺寸/速度是真实的。

### 1.6 一键自检 + 精修 A/B
| 操作 | 期望结果（实测） |
| --- | --- |
| 「跑一键自检」→ 等 1~3 分钟 → 「看报告」 | 报告表逐行给出：trust 模型 R²、可信域、资产库、rig、两组 rollout、导出 round-trip、扩散状态；实测 `ok: true` |
| 精修方式选 `stub` → 「跑精修 A/B」→ 「看 A/B 报告」 | 每台留出相机的 `PSNR/SSIM/LPIPS` 与精修后数值 + Δ；实测 cam3 SSIM 0.2173→0.2222 |
| 精修方式选 `difix` | 依赖就绪时正常跑；未就绪时**直接拦下并说明缺哪个文件** |

---

## 2. 命令行等价测试（可脚本化）

```bash
cd /root/autodl-fs/dggt-main
export PATH=/root/autodl-tmp/conda_envs/dggt/bin:$PATH
SCENE=output/trust/scene001_cam012/001

# ① 一条命令跑完整链路（trust→可信域→资产库→rig→rollout→导出→扩散状态）
python studio/backend/run_closedloop_demo.py --scene $SCENE --segment 001 --frames 8 --steps 5 \
    --out_dir output/demo
#   → output/demo/<场景标识>/closedloop_demo_report.json

# ② 相机 rig 传感器（多相机观测）
python studio/backend/sensor_rig.py --scene $SCENE --segment 001 --frame 0 \
    --cameras 0,1,2,3,4 --out output/sensor_rig/demo --distort

# ③ gym 风格闭环 rollout（可换动作/阈值）
python studio/backend/sim_runtime.py --scene $SCENE --steps 20 --speed 8 --steer 0 \
    --min_trust_frac 0.8 --out_dir output/sim

# ④ 导出 OpenSCENARIO + CommonRoad（默认只导参与者，--all_tracks 导全部）
python studio/backend/scenario_export.py --scene $SCENE --scenario rear-end --seed 7 \
    --frames 12 --out output/export/demo

# ⑤ 留出视角画质 + 精修 A/B（difix 需要 sd-turbo 权重就绪）
python studio/backend/novelview_trust.py --scene_dir $SCENE --segment 001 --num_frames 4 \
    --heldout 3,4 --refine difix --out output/trust/trust_ab_difix.json

# ⑥ 确定性自检（同 seed 重跑是否一致）
python studio/backend/determinism_check.py --scene $SCENE --scenario rear-end --seed 7 --runs 2

# ⑦ 俯视 3D「真模型 vs 方框」对比图
python studio/backend/td3d_ab.py --scene $SCENE --ahead 9 --span 15
#   → output/asset_verify/td3d_ab/model_vs_box.png（左=方框，右=真模型）
```

---

## 3. 产物在哪、怎么用

| 产物 | 路径 | 用途 |
| --- | --- | --- |
| 总自检报告 | `output/demo*/<tag>/closedloop_demo_report.json` | 每个环节的结论/耗时，可当回归基线 |
| 可信域报告 | `output/trust/trust_*.json` | 拟合 trust 模型的原始样本（含留出相机 PSNR/SSIM/LPIPS） |
| 资产库 | `output/actor_assets/<tag>/bank.json` + 每资产 `*.ply` | 俯视 3D 与"任意摆位渲染"的输入 |
| 相机观测 | `output/sensor_rig/*/frame*_cam*.png` | 多相机传感器数据 |
| rollout | `output/sim/sim_*.json` | 逐帧 coverage/预测 SSIM/最近物体/碰撞/截断 |
| 标准格式 | `output/export/<name>/*.xosc|*.cr.xml|*.world.json` | 给规划器/仿真器（CARLA、esmini、CommonRoad 工具链） |
| 视频 | `output/corner_cases/<scene>/videos/*.mp4` | 主车视角 / BEV / 俯视 3D |

---

## 4. 判读要点（避免误读）

1. **可信域大小取决于重建质量**：1 视角 → ~2m，3/4 视角 → ~6m（横移）。想大范围偏离原轨迹，先做多视角重建。
2. **绝对画质**：3 视角重建下，留出 45° 相机 SSIM ≈0.2，90° 侧向 ≈0.12~0.19；参考视角（重建时喂过的）≈0.86。
   所以"闭环可信"≠"画面像真照片"，而是"在给定阈值内可用"，阈值应随重建质量设置（用相对阈值）。
3. **不做时间外推**：超出记录帧窗口就截断并说明，不会编造画面。
4. **地图是占位**：导出里的道路/车道需要你自己的高精地图替换（轨迹本身是真实的）。
5. **扩散精修（Difix）**：`pretrained/model_difix.pkl` 是精修头；还需要 `stabilityai/sd-turbo` 的
   unet/vae/text_encoder 权重。放在 `/root/autodl-tmp/hf_home/sd-turbo` 或 HF 缓存里均可，
   `python studio/backend/diffusion_refine.py --status` 会做**深度校验**（真读 safetensors 头）。
   本机能用的下载路线是 ModelScope（HF/hf-mirror 对大文件不传数据）：

   ```bash
   cd /root/autodl-tmp/hf_home/sd-turbo
   for f in unet/diffusion_pytorch_model.fp16.safetensors \
            vae/diffusion_pytorch_model.fp16.safetensors \
            text_encoder/model.fp16.safetensors; do
     rm -f "$f"   # 必须先删 git-LFS 指针文件，否则下完也读不了
     curl -L --retry 3 -o "$f" \
       "https://www.modelscope.cn/api/v1/models/stabilityai/sd-turbo/repo?Revision=master&FilePath=$f"
   done
   ```
