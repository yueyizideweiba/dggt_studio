# 多阶段交通交互模型：预警区 / 危险区 / 发生区

> 可引用 / 可评审的方法论规范。本文把 `studio/backend/phase_zones.py` 中的工程实现，
> 提升为一套**形式化、可解释、与实际应用对齐**的多阶段交互判据模型。
> 核心贡献一句话：**把"是否会发生事故"的二值判断，推广为"在哪个阶段、该做什么反应、
> 还来不来得及"的分级、多维、可解释判据。**

---

## 1. 问题定义与动机

自动驾驶系统（ADS）对一次潜在交互，不应只回答"撞 / 不撞"，而应回答三个连续的问题：

1. **该不该反应**（是否进入危险交互）；
2. **该怎么反应**（制动、转向，还是只能最小化损失）；
3. **还来不来得及**（反应窗口还剩多少）。

据此，把一次交互按**运动学可达性**划分为若干阶段（区域），并给每个区域配一个可执行的反应策略。
对 ADS 而言，这就是"分级响应"的判据；对 corner case 生成而言，这是"事故可不可被规避"的可度量语言。

## 2. 符号约定

| 符号 | 含义 |
| --- | --- |
| A, B | 反应主体（ego）与其它参与者 |
| $\mathbf{p}_A, \mathbf{p}_B$ | 世界坐标中心（XZ 地面平面，Y 为高） |
| $\mathbf{v}_A, \mathbf{v}_B$ | 速度向量 |
| $\mathbf{a}_A, \mathbf{a}_B$ | 加速度向量（轨迹差分估计，预测时饱和到 ±3 m/s²） |
| $dims_A, dims_B$ | 包围盒尺寸（宽/高/长），OBB |
| $t_r$ | 反应时间（感知 + 决策 + 执行延迟） |
| $a_{comf}, a_{emerg}$ | 舒适 / 紧急减速度阈值 |
| $\mu$ | 路面附着系数 |
| $H$ | 预测时域 |
| $TTC$ | 碰撞时间（Time To Collision） |
| $THW$ | 车头时距（Time Headway） |

## 3. 多因素判据（每个因素单独可解释、逐帧可追溯）

判据不是单一标量，而是**一组物理因素**，最终由"能力包络"合成一个区域。

### 3.1 纵向制动所需减速度 $a^*$（制动能力）

$$
a^* \;=\; \min\Big\{ a \ge 0 \;\Big|\; \text{在 } t_r \text{ 反应之后以恒定 } a \text{ 制动，预测时域 } H \text{ 内 } \text{OBB}_A \cap \text{OBB}_B = \varnothing \Big\}
$$

- 预测采用**二阶外推**：$B$ 以饱和后的实测加速度运动；$A$ 在反应期保持原速、之后沿当前航向减速。
- $a^*=0$：本无接触；$a^*\to\infty$：直到紧急上限仍无法避免（**不可避免碰撞态**）。

**解析闭式（brake-to-stop 上界）**：一维闭合场景（对向/追尾，$B$ 匀速或静止，转化为 $B$ 参考系），
设有效间隙 $d_0 = gap - v_{rel}\,t_r$（$gap$ 已扣 OBB 半长），则把相对速度降到 0 的最小减速度为

$$
a^*_{stop} \;=\; \frac{v_{rel}^2}{2\,d_0} \;=\; \frac{v_{rel}^2}{2\,(gap - v_{rel}\,t_r)}
$$

这是"恒定减速度刹停"的标准运动学结果，也是 RSS / ISO 15623 制动判据的连续形式。它要求**完全刹停**，
故是数值解 $a^*$（只需在时域 $H$ 内避免接触）的**解析上界**：当 $H$ 不足以完全刹停时 $a^*_{stop}\ge a^*$。
一般二阶加速度 / 横穿无闭式，数值解用**二分法**精确求根（对一维闭合，判据"以 $a$ 制动是否避免接触"
关于 $a$ 单调，二分收敛到阈值）。

### 3.2 横向转向所需加速度 $a_{lat}$（转向能力）

要在 $(TTC - t_r)$ 秒内横向避让所需的**横向位移** $W_{clr}$，所需最小横向加速度：

$$
a_{lat} \;=\; \frac{2\,W_{clr}}{(TTC - t_r)^2}
$$

$W_{clr}$ 为 **OBB 感知的横向净空**（而非固定一个车道宽）：

$$
W_{clr} \;=\; \tfrac{1}{2}(w_A + w_B) + \delta
$$

（$w_A,w_B$ 为两车宽度，$\delta$ 为近距余量）。两辆 2 m 宽的车只需 ~2 m 横移即可错开，而非整整一个
车道（3.5 m），故转向更真实地"更可行"，也不会在横向净空不足时低估转向难度。

### 3.3 能力包络（受路面附着约束）

$$
a_{cap}^{brake} = \min(a_{emerg},\; \mu g), \qquad a_{cap}^{lat} = 0.9\,\mu g
$$

湿滑 / 冰雪路面（$\mu$ 小）直接压缩能力包络 → 危险区与发生区变宽——这是"路面状态如何决定事故是否可规避"的显式建模。

### 3.4 撞击严重度（能量维度）$\Delta v$ 与接触几何

- $\Delta v = \|\mathbf{v}_A - \mathbf{v}_B\|$（预测接触时的相对速度）。
- 几何分类：对向（航向夹角 ≥135°）/ 侧面（45°–135°）/ 追尾（<45°）/ 静止。
- 弱势道路使用者（VRU，行人/非机动车）给更保守的余量与严重度加权。

### 3.5 时间紧迫度：$TTC$ 与反应裕度

- $TTC$：二阶外推下最早 OBB 接触时刻。
- 反应裕度 $= TTC - t_r$：反应之后还剩多少时间。

### 3.6 车头时距 $THW$（补充判据，覆盖"贴车不撞"）

当无预测接触、但两车**同向**（航向夹角 <45°）且前方有车时：

$$
THW = \frac{\text{纵向间距}}{|\mathbf{v}_A|}
$$

它捕获"跟车过近"这类**尚未发生冲突、但已不安全**的状态，是 ISO 15623 前向碰撞预警中
"headway warning"的同类量。

### 3.7 机动余量（maneuver margin，可解释的"还剩多少能力"）

对每个可行机动定义余量 $\mu = 1 - \tfrac{\text{需求}}{\text{能力}}$（1 = 零负担，0 = 用满能力，<0 = 能力不足）：

$$
\mu_{brake} = 1 - \frac{a^*}{a_{cap}^{brake}}, \qquad
\mu_{lat} = 1 - \frac{a_{lat}}{a_{cap}^{lat}}
$$

最优机动余量 $\mu^* = \max(\mu_{brake}, \mu_{lat})$ 直接刻画"离不可行还差多少"，是分级响应
（档位）的**连续化、可比较**量化指标，逐帧可追溯。

## 4. 能力包络 → 区域划分（形式化）

对制动、转向各评一个**能力档位**：

| 档位 | 制动 | 转向 |
| --- | --- | --- |
| 0 无需求 | $a^* = 0$ | — |
| 1 舒适 | $a^* \le a_{comf}$ | $a_{lat} \le a_{lat}^{comf}$ |
| 2 紧急 | $a^* \le a_{cap}^{brake}$ | $a_{lat} \le a_{cap}^{lat}$ |
| 3 不足 | $a^* > a_{cap}^{brake}$ | $a_{lat} > a_{cap}^{lat}$ |

主车取**最好可行动作**（能力包络边界）：

$$
tier = \min(tier_{brake}, tier_{steer})
$$

区域定义：

$$
zone = \begin{cases}
\text{safe}       & \text{无预测接触 且 } THW \ge thw_{warn} \\[2pt]
\text{warning}    & tier = 1,\ \text{或}\ THW < thw_{warn} \\[2pt]
\text{danger}     & tier = 2,\ \text{或}\ THW < thw_{danger} \\[2pt]
\text{occurrence} & tier = 3\ (\text{制动且转向都不可解 = ICS}) \\[2pt]
\text{impact}     & \text{OBB 已接触} \\[2pt]
\text{occluded}   & B \text{ 不可见（鬼探头）}
\end{cases}
$$

**关键结论（最晚反应点 LRP）**：`danger` 与 `occurrence` 的分界，就是"即使采取最激进的规避动作也无法
避免"的时刻——即**最晚反应点（Latest Reaction Point, LRP）**。形式化：

$$
t_{LRP} \;=\; \max\big\{\, t \;\big|\; \text{zone}(t)=\text{danger} \,\big\}
$$

（若无危险区则退化为最后一个预警区）。这是"如何避免事故"研究最关心的量：在 $t_{LRP}$ 之前反应可规避，
之后只能最小化损失。输出中 $t_{LRP}$ 附带上该点的 $TTC$ 与机动余量 $\mu^*$，把 LRP 从"一个帧号"升级为
"还剩多少时间 + 还剩多少能力"。

## 5. 二维风险：紧急度 × 严重度

区域回答"**多紧急**"（运动学可达性），严重度回答"**多严重**"（能量后果）。二者合成风险等级：

| | 高严重度 | 中严重度 | 低严重度 |
| --- | --- | --- | --- |
| occurrence | **critical** | severe | severe |
| danger | high | high | high |
| warning | elevated | elevated | elevated |
| safe | low | low | low |

输出中 `binding ∈ {brake, steer, none, impact, near_miss, headway, occluded, self_brake}`
记录**是哪一个因素决定了该帧的区域**——这是"可解释"的关键：每一帧都能回答"为什么是这个区域、
我该做什么"。

## 6. 与既有安全模型的对应关系

| 本模型概念 | 对应既有工作 |
| --- | --- |
| $a^*$（所需减速度）与两级阈值 | ISO 15623（FCW/AEB 的分级制动）、RSS 的"proper response" |
| occurrence = 制动且转向都不可解 | **Inevitable Collision States (ICS)**，Fraichard & Asama 2004 |
| $THW$ 补充判据 | 交通冲突技术中的 headway 代理指标（Vogel 2003 等） |
| $TTC$ + 反应裕度 | TTC 代理（Hayward 1972） |
| 分级响应（舒适→紧急→最小化损失） | 驾驶任务分级 / 责任敏感安全的分级干预 |

本模型的增量贡献：把上述**单因素阈值**统一进一个**多维能力包络 + 二维风险**的框架，并显式给出
**最晚反应点**与**corner case 的可度量签名**（见下节）。

## 7. Corner case 的度量：预警区被压缩 / 跳过

设一次交互的预警区、危险区、发生区时长分别为 $\tau_w, \tau_d, \tau_o$，则：

- **完整演进**：$\tau_w > 0,\ \tau_d > 0,\ \tau_o > 0$（正常可预警事故）；
- **预警被压缩**：$0 < \tau_w \le \delta$（$\delta$ 为压缩阈值，如 0.3 s）；
- **预警被跳过**：$\tau_w = 0$ 且 $\tau_d>0$ 或 $\tau_o>0$（鬼探头等）；
- **险情（可避免）**：$\tau_o = 0$ 且 $\tau_w>0$（只有预警、无接触）。

由此，corner case 的"严重程度"有了**可度量、可比较**的语言：预警区越短、越被跳过，越"难规避"。
这为"生成—评估—筛选 corner case"提供了量化指标。

## 8. 算法（伪代码）

```
对每个参与者对 (A, B) 在帧 t：
  1. 从轨迹差分估计 v_A, a_A, v_B, a_B
  2. 若 OBB 已接触 → impact
  3. 二阶外推预测 TTC 与最近距离 d_min
  4. 若 TTC 存在：
       a* = 最小避免接触减速度（二分法精确求根；一维闭合有解析上界 a*_stop）
       a_lat = 2·W_clr/(TTC-t_r)^2，W_clr = 半宽之和 + 余量
       tier = min(tier_brake(a*), tier_steer(a_lat))
       zone = {1:warning, 2:danger, 3:occurrence}[tier]
       余量 μ_brake/μ_lat = 1 - 需求/能力；严重度 = f(Δv, 几何, VRU)
  5. 否则（无接触）：
       THW = 同向且前方时的纵向间距/速度
       zone = THW<thw_danger ? danger : THW<thw_warn ? warning : (近距擦过? warning : safe)
  6. 记录 binding、各因素、risk_level
全局：取所有对中"最危险"区域作为该帧全局状态
```

## 9. 模型超参数（默认值 + 依据）

| 参数 | 默认 | 依据 |
| --- | --- | --- |
| $t_r$ | 0.7 s | 典型驾驶员反应时间（行人场景 0.9–1.0 s） |
| $a_{comf}$ | 3.0 m/s² | 乘员舒适减速度上界 |
| $a_{emerg}$ | 8.0 m/s² | 干燥沥青紧急制动能力（≈$\mu g$, $\mu$=0.8） |
| $\mu$ | 0.8 | 干沥青；湿/冰可降到 0.4/0.2 |
| $W$ | 3.5 m | 单车道宽度（转向避让横移量） |
| $H$ | 4.0 s | 预测时域 |
| $thw_{warn}$ | 1.0 s | 车头时距预警阈值（ISO 头车距分级） |
| $thw_{danger}$ | 0.5 s | 车头时距危险阈值 |

所有参数均可覆盖（`/api/corner_case/pairwise` / `phases` 的 `params`），实现"同一场景、不同参数 →
不同区域划分"的可控实验。

## 10. 评估指标

1. **区域时长分布**：$\tau_w,\tau_d,\tau_o$ 的统计（刻画事故的预警充分性）。
2. **规避成功率**：对 warning/danger 阶段施加对应动作，验证"窗口内是否不再相撞"（`apply_evasion`）。
3. **最晚反应点估计**：danger→occurrence 的分界帧，与 $TTC$ 对照。
4. **压缩/跳过率**：corner case 数据集中 `compressed_warning / skipped_warning` 的比例。

## 11. 局限与未来工作

1. 制动是"沿当前航向刹停"原语；转向避让为独立因子，尚未做"制动+转向联合优化"。
2. 预测用二阶（常加速度饱和）外推；高阶/交互式预测（考虑对方也会反应）是扩展方向。
3. 单参与者急刹（hard-brake）用主车自身制动强度判相态，尚未显式建模"后车追尾风险传递"。
4. 参数默认值为通用值；面向特定 ODD（高速/城市/雨雪）可做标定。

---

**实现位置**：`studio/backend/phase_zones.py`（`analyze_pair` / `compute_phase_timeline` /
`compute_pairwise_phases` / `apply_evasion`）；验证：`tools/verify_phase_zones.py`。
