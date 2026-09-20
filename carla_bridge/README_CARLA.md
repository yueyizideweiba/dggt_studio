# DGGT corner-case → CARLA 闭环仿真 / 可视化

把 `studio/backend` 导出的事故场景（`.world.json` / `.xosc`）接入 **CARLA 0.9.15**，
在 CARLA 内置地图上回放 / 闭环跑，产出 **MP4 视频 + 事件 JSON + 浏览器网页实时画面**。

---

## 0. TL;DR（在服务器上）

```bash
cd /autodl-fs/data/dggt-main/carla_bridge

./demo_rear_end.sh                    # 一键：起 CARLA + 跑 rear_end 事故场景 + 出 MP4
python batch_render.py                # 批量：output/export 下所有场景都出一遍视频（约 45s）
python batch_render.py --include-demo  # 连 output/demo* 里的场景一起

# 看效果
#  A. MP4：VSCode 里右键 output/carla/<场景>/<场景>.mp4 -> Download
#  B. 网页实时：LIVE=1 ./demo_rear_end.sh  然后转发 8090，浏览器开 http://127.0.0.1:8090/
#  C. 标准链路：./run_bridge.sh ... --export-xosc 后 ./run_scenario_runner.sh <xxx_carla.xosc>
```

已实测跑通的产物：

```
output/carla/rear_end_s7b_high/
  rear-end_s7.world.mp4          74 帧，3 机位（追逐/俯视/侧视）+ 标注
  rear-end_s7.world_events.json  地图、actor 清单、碰撞事件、全程最小距离/TTC
  rear-end_s7.world_frames.json  每帧 ego 位姿/速度/指标
```

---

## 1. 装在哪

| 东西 | 路径 |
| --- | --- |
| CARLA 0.9.15 发行包 | `/autodl-fs/data/carla/`（tar.gz 解压出来是**平铺的**，`CarlaUE4.sh` 直接在根） |
| Python 3.7 环境（CARLA API） | `/autodl-fs/data/carla/py37/` |
| 服务器日志 | `/autodl-fs/data/carla/logs/` |
| 桥接脚本 | `/autodl-fs/data/dggt-main/carla_bridge/` |
| 输出 | `/autodl-fs/data/dggt-main/output/carla/` |
| NVIDIA 驱动包（修复用） | `/autodl-fs/data/nv_driver/` |

> 为什么装 `/autodl-fs/data`：容器根分区 `/` 只剩 ~3 GB，`/autodl-fs/data` 有 100+ GB。

---

## 2. ⚠️ 容器里 NVIDIA 图形用户态原本是坏的（已修复，记录一下）

**症状**：CUDA 完全正常（`torch.cuda.is_available()==True`），但 Vulkan/GL 起不来：

```
ERROR: Could not get 'vkCreateInstance' via 'vk_icdGetInstanceProcAddr' for ICD libGLX_nvidia.so.0
Cannot create Vulkan instance ... ERROR_INCOMPATIBLE_DRIVER
```

**根因**：容器里 `/usr/lib/x86_64-linux-gnu` 下 580.142 / 570.x 的驱动库全是 **0 字节占位文件**，
只有一部分 580.76.05 的库从宿主机 bind mount 进来，而 `libnvidia-gpucomp.so.580.76.05`
（还有 libEGL.so.580.76.05 等）**整个缺失** → `libGLX_nvidia`（Vulkan ICD）初始化失败，
`vk_icdNegotiateLoaderICDInterfaceVersion` 直接返回 -3。
另外 `/etc/vulkan/icd.d/nvidia_icd.json` 也是 0 字节。

**修复**：`./fix_nvidia_vulkan.sh`（可重复执行，幂等）：

1. 把 `nvidia_icd.json` 写成合法 JSON；
2. 从 NVIDIA 官方 `download.nvidia.com` 下 **与内核模块同版本** 的 `.run`（580.76.05，394 MB），
   `--extract-only` 解包（不安装、不碰内核）；
3. 把 `lib*.so.580.76.05` 全部拷进 `/usr/lib/x86_64-linux-gnu/` 并 `ldconfig`；
4. 用 `vulkaninfo --summary` 验证能看到 `NVIDIA GeForce RTX 3090 / 580.76.05`。

修复后实测：CARLA 离屏渲染 **30–50 fps**，一张 960×540 相机帧 ≈ 3 ms。

> 退路：如果哪天又坏了，`SOFTWARE_RENDER=1 ONETHREAD=1 ./start_carla_server.sh` 可以退化到
> Mesa 软件渲染（lavapipe），但很慢（本机实测 UE4 起不来/极慢）。

---

## 3. 脚本一览

| 脚本 | 作用 |
| --- | --- |
| `carla_env.sh` | 公共环境变量（CARLA_HOME / python / 渲染后端） |
| `fix_nvidia_vulkan.sh` | 修 NVIDIA Vulkan（见上） |
| `start_carla_server.sh` | 起 CARLA server；`--daemon` 后台；`RENDER_MODE=offscreen/xvfb/x11` |
| `stop_all.sh` | 停 CARLA / 桥接 / VNC |
| `carla_scenario_bridge.py` | 主程序：场景 → CARLA → MP4/JSON/直播 |
| `run_bridge.sh` | 用 py37 环境跑主程序 |
| `batch_render.py` | 批量把 `output/export` 下所有场景渲染成视频 + 汇总报告 |
| `run_scenario_runner.sh` | 用 CARLA 官方 ScenarioRunner 跑「CARLA 对齐版」.xosc（标准逻辑路径） |
| `demo_rear_end.sh` | 一键 demo（`LIVE=1` 走网页直播） |
| `smoke_test.py` | 连通性 + 渲染 + 帧率冒烟测试 |
| `setup_carla_env.sh` | 重建 py37 环境时用 |
| `start_carla_vnc.sh` | Xvfb + x11vnc + noVNC（本容器里 CARLA 窗口模式会 segfault，见第 6 节） |
| `SSH_SETUP.md` | VSCode 免密 SSH 说明（kt 已配好 + Windows 步骤） |
| `README_CARLA.md` | 本文 |

---

## 4. 三种看效果的方式

### 4.1 MP4（最稳，默认）
```bash
./run_bridge.sh \
  --scenario ../output/export/rear_end_s7b/rear-end_s7.world.json \
  --out-dir ../output/carla/rear_end_s7b \
  --map Town10HD_Opt --fps 20 --cameras chase,birdseye --focus crash
```
- 画面里每个交通参与者都带 **role + 速度** 标注（`victim#1 1.5m/s` / `attacker#5 13.2m/s` / `ego#900000 14.0m/s`）
- 顶部 HUD 实时显示 `t / minDist / minTTC / collisions`
- 默认 `--weather ClearNoon`（正午晴天，画面亮好认）；要别的用 `--weather ClearSunset|CloudyNoon|ClearNight|...`，`--weather none` 保持地图默认
- `--focus crash|ego|all` 决定相机跟谁：`crash`=事故双方(victim/attacker)、`ego`=自车
- 加 `--start-recorder <file>.log` 还能出 CARLA recorder 文件，之后可 `client.replay_file()` 回放

### 4.2 浏览器网页实时（推荐，不需要 X11）
```bash
LIVE=1 ./demo_rear_end.sh          # 等价于 run_bridge.sh ... --loop 0 --live-http 8090
```
在 VSCode（本机 kt）里：远程窗口 → **端口 / PORTS** 面板 → 转发 **8090** →
浏览器打开 `http://127.0.0.1:8090/`。就是 4.1 的画面，但是**实时循环播放**的 MJPEG。
（也可以本地起隧道：`ssh -N -L 8090:127.0.0.1:8090 -p 38924 root@region-41.seetacloud.com`）

### 4.3 CARLA 原生窗口 / noVNC（本容器里不行，仅供参考）
```bash
./start_carla_vnc.sh    # Xvfb:99 + x11vnc:5900 + noVNC:6080 + CARLA -windowed
```
**注意**：本容器里 CARLA 的 `-windowed`（X11）模式会 segfault（离屏完全正常），
所以这条路目前看不了画面。要么等以后换容器，要么直接用 4.2。

---

## 5. 坐标 / 落图 / 闭环模式

**坐标系**：DGGT 重建世界是 **y-up**（x 右、y 上、z 前，米，10 Hz）；CARLA 是 **z-up 左手系**（x 前、y 右、z 上）。

```
loc_carla = (z_dggt, x_dggt, y_dggt)     yaw_carla = yaw_dggt = atan2(R[0,2], R[2,2])
.xosc 里导出脚本已重排轴序，所以：loc_carla = (osc.y, osc.x, osc.z)，yaw = h
```

**落图**：导出里没有真实路网（`LogicFile=placeholder.xodr`），所以桥接会

1. 在 CARLA 地图里找一条**直路**（沿车道走 40 m 航向偏差 < 4°，且非路口，Town10HD_Opt 里有 73 个候选）；
2. 把整段场景**刚体**平移+旋转，使 **ego 首帧位姿 == 该道路锚点**；
3. 其它参与者保持相对关系；垂直方向默认 `--vertical flat`（全部贴路面，避免重建高度不准导致悬空）。

**ego 控制模式**：

| 模式 | 说明 |
| --- | --- |
| `--ego-mode playback`（默认） | 所有参与者按记录轨迹回放 → 忠实重放你闭环仿真的结果 |
| `--ego-mode autopilot` | ego 交给 CARLA TrafficManager 自己开（目标速度=场景首帧速度），其余车仍按轨迹跑 → 真正在 CARLA 里闭环 |

**筛选**：`--actors dynamic`（默认）会自动挑出 ego/victim/attacker + 尺寸像车/行人的动态目标，
把电线杆、路牌、噪声点（L/W/H 明显不合理的 track）过滤掉（`--actors all/raw/core` 可选）。

---

## 6. 已知问题 / 踩坑

| 现象 | 说明 / 处理 |
| --- | --- |
| `Refusing to run with the root privileges` | UE4 不允许 root 跑；脚本自动降权到 `carla` 用户（`RUN_AS`） |
| `GameThread timed out waiting for RenderThread after 60.00 secs` | 软件渲染下 shader 编译超过 UE4 看门狗；GPU 正常时不会出现，必要时 `ONETHREAD=1` 绕开 |
| `-quality-level=Epic` + `Town10HD` 会 segfault | 用默认 `High` + `Town10HD_Opt`（已实测稳定） |
| 路面有青色网格线 | `_Opt` 地图材质自带的可视化网格，纯视觉、不影响数据；非 Opt 的 HD 地图在本容器里加载会崩 |
| `load_world` 偶发 60 s 超时 | 服务器刚起时首帧较慢；桥接现在默认 120 s 超时并重试 3 次 |
| CARLA 窗口模式 segfault | 见 4.3，用网页直播替代 |

---

## 7. 和你的流水线对接

### 7.1 一条命令：导出后自动跑 CARLA

`studio/backend/run_closedloop_demo.py` 已加好可选开关（默认关闭，不影响原有行为）：

```bash
python run_closedloop_demo.py --scene <重建场景> --segment 001 \
    --carla                       # 导出后自动送进 CARLA 渲染回放视频
    # 可选：--carla_out <目录>  --carla_map Town10HD_Opt  --carla_focus crash|ego|all
```

跑完会在 `output/carla/<tag>/` 出 MP4 + events/frames JSON，并把这些信息写进
`closedloop_demo_report.json` 的 `steps.carla` 里（CARLA 失败不会影响原有流程）。

### 7.2 批量：把所有已导出场景都渲染一遍

```bash
python batch_render.py                       # 默认 output/export/*/*.world.json
python batch_render.py --include-demo        # 连 output/demo* 一起
```
同地图只 `load_world` 一次（`--reload-world auto`），4 个场景约 45 秒；
汇总在 `output/carla/_batch_report.json`（帧数 / actor 数 / 碰撞数 / 最小距离 / 最小 TTC）。

### 7.3 ScenarioRunner 标准路径

桥接可以额外导出一份「CARLA 对齐版」OpenSCENARIO 1.0：

```bash
./run_bridge.sh --scenario ../output/export/rear_end_s7b/rear-end_s7.world.json \
  --out-dir ../output/carla/xosc_test --export-xosc
# -> ../output/carla/xosc_test/rear-end_s7_carla.xosc
./run_scenario_runner.sh ../output/carla/xosc_test/rear-end_s7_carla.xosc
```

这份文件针对 ScenarioRunner 做了 4 处适配（原始导出直接用会被 XSD 拒绝）：

| 项 | 原始导出 | 对齐版 |
| --- | --- | --- |
| 坐标 | DGGT 世界系（y-up） | CARLA 坐标系（x 前/y 右/z 上），已落图对齐 |
| 坐标系开关 | 无 | `FileHeader description="CARLA:..."`（SR 据此不做 y/yaw 取反） |
| LogicFile | `placeholder.xodr` | CARLA 地图名（`Town10HD_Opt`），`--reloadWorld` 可直接加载 |
| Polyline | `<Vertex x y z h time/>` | `<Vertex time><Position><WorldPosition/></Position></Vertex>`（SR 的 `get_trajectory()` 只认这种） |
| FollowTrajectoryAction | 直接在 `PrivateAction` 下（XSD 非法） | 包在 `RoutingAction` 里（OSC 1.0 要求） |

实测：SR 的 xmlschema 校验通过、`Preparing scenario / Running scenario` 正常跑完，
并且最后通过 `ChangeActorWaypoints` 把 Polyline 的每个顶点变成 waypoint 让车开过去。
注意 SR 是**逻辑/评测**运行器（跑得快、没有 criteria 就会报 not successful），
**要看事故画面还是用桥接出的 MP4 / 网页直播**。

### 7.4 也可以直接吃 .xosc

桥接两种输入都支持（`--scenario xx.xosc`），只是 `.xosc` 里没有 victim/attacker
角色信息（会退化成 `actor_*`）；要角色信息就用 `.world.json`。

## 8. 让车「不飘 / 在车道里 / 会拐弯」——`--road-snap`

DGGT 的轨迹是 y-up（x 右 / y 上 / z 前），落图对齐后如果直接拿原始高度，
车会悬在空中或飞出去；平面位置也可能压在路肩/人行道上。桥接和常驻引擎都支持：

| 值 | 行为 | 什么时候用 |
| --- | --- | --- |
| `z`（默认） | 只把高度改成最近路面的高度，平面位置不动 | **评测/复现事故**：几何完全忠于原轨迹，只是不再悬空/乱飞 |
| `lane` | 把每辆车吸到最近 **行车道中心线**（`LaneType.Driving`），高度=路面高度 | 演示/出片：画面规整好看，但**事故相对位置会被改动** |
| `none` | 完全不处理 | 调试原始数据 |

```bash
./run_bridge.sh --scenario ../output/export/head_on/x.world.json --road-snap lane
./run_bridge.sh --scenario ../output/export/head_on/x.world.json --road-snap z
```

注意：`lane` 会**改动事故几何**（车被挪到车道中心），所以**默认用 `z`**：
实测同一条 lane-change-cutin 实例，`z` 时两车最近距离 19.1 m、`lane` 时 12.6 m —— 已经被挪过位置了。
早期版本 `lane` 用的是 `LaneType.Any`，会吸到人行道/停车带，实测仍偏 2~8 m，
已经改成 `Driving` 并把吸附半径放宽到 30 m。

另外：`spawn_actor` 在**位置重叠**时会直接报错（头碰头场景里对向车常常正好压在另一辆车上，
以前是静默跳过 → 画面里少一辆车）。现在会依次尝试"抬高 → 侧向挪 0.9/1.8/2.8/4.0 m"，
并在日志里打印挪了多少，保证车不会凭空消失。

## 9. ego 交给 CARLA 自己开（`--ego-mode autopilot`）

```bash
./run_bridge.sh --scenario ../output/export/head_on/x.world.json \
  --ego-mode autopilot --road-snap lane
```

* ego 保持**物理开启**（TM 驱动一台没有物理的车会直接崩），其余车仍按轨迹回放。
* TrafficManager 端口会**自动挑空闲端口**（`--tm-port` 只是起始值）：
  TM 是在客户端进程里起监听线程的，端口被占（比如 studio 后端的 8000）时会 C++ abort（rc=-6）。
* 退出顺序很关键：先 `set_autopilot(False)`、再关 TM 同步模式，**最后**才销毁 actor。
  顺序反了 TM 的内部线程会去操作已销毁的车，抛 `std::runtime_error` → 进程 abort，
  而且这个异常在 Python 侧 `try/except` 拦不住（排查时只看到 rc=-6）。

## 10. 回归自检 / 验证脚本

```bash
# 静态自检：抓「方法里用了裸变量 a」这类只在运行时才炸的错误（曾经导致 rc=-6）
python3 selfcheck.py

# 动态验证：起常驻引擎，逐帧采所有 actor 的朝向/高度/离车道中心距离
python3 verify_turn.py ../output/corner_cases/<uuid>/<时间戳>/head-on-000-xxx.world.json lane z
```

`verify_turn.py` 输出形如：

```
       tid     role   ego      朝向跨度     z落差        离车道中心
         3    actor False     2.7°—   0.00m✓   0.00~0.06 m ✓
    100000    actor False    28.5°✓   0.00m✓   0.00~0.00 m ✓
    900000      ego  True     0.1°—   0.00m✓   0.00~0.11 m ✓
```

判断标准：`z落差` 接近 0 = 不飘；`离车道中心` ≤ 3.5 m = 在车道里；
`朝向跨度` ≥ 20° = 真的在转弯。**注意**：很多事故类型里 ego 本来就是直行
（转向的是对向车/加塞车），所以别只看 ego 那一行。
