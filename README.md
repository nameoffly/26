# 2026 MCM Problem C: Dancing with the Stars 排名预测与淘汰一致性分析

## 项目概述

本项目针对**2026年MCM数学建模竞赛C题**，建立了一套完整的数学模型和算法框架，用于分析舞蹈竞技节目 *Dancing with the Stars* 中的观众投票排名反演问题。核心目标是在已知评委打分排名和淘汰结果的条件下，推断未知的观众排名，并进行不确定性量化分析。

---

## 问题背景

### 竞赛规则
在 *Dancing with the Stars* 节目中，每周的淘汰规则为：
- **评委排名** $r^J_{i,w}$：根据评委打分得出（已知）
- **观众排名** $r^F_{i,w}$：根据观众投票得出（未知）
- **综合排名** $R_{i,w} = r^J_{i,w} + r^F_{i,w}$

每周综合排名最差（数值最大）的选手被淘汰。

### 核心挑战
观众投票数据不公开，仅知道淘汰结果。本模型通过约束优化反演观众排名。

---

## 数学建模

### 1. 符号定义

设第 $w$ 周仍在场选手集合为 $S_w$，人数 $n_w = |S_w|$。

| 符号 | 含义 |
|------|------|
| $r^J_{i,w}$ | 选手 $i$ 在第 $w$ 周的评委排名（已知） |
| $r^F_{i,w}$ | 选手 $i$ 在第 $w$ 周的观众排名（待求） |
| $R_{i,w}$ | 综合排名，$R_{i,w} = r^J_{i,w} + r^F_{i,w}$ |
| $e(w)$ | 第 $w$ 周实际被淘汰的选手 |
| $x_{i,k,w}$ | 二值变量，表示选手 $i$ 第 $w$ 周观众排名为 $k$ |
| $\delta_{j,w}$ | 松弛变量，允许淘汰约束轻微违反 |

### 2. 决策变量

引入二值指派变量建模排名的排列结构：

$$
x_{i,k,w} \in \{0,1\}, \quad i \in S_w,\ k = 1, \ldots, n_w
$$

观众排名由指派变量确定：

$$
r^F_{i,w} = \sum_{k=1}^{n_w} k \cdot x_{i,k,w}
$$

### 3. 约束条件

#### 约束1：排列约束（Assignment Constraints）

确保观众排名构成一个有效排列：

**每个选手恰好一个排名：**
$$
\sum_{k=1}^{n_w} x_{i,k,w} = 1, \quad \forall i \in S_w
$$

**每个排名恰好给一个选手：**
$$
\sum_{i \in S_w} x_{i,k,w} = 1, \quad \forall k = 1, \ldots, n_w
$$

#### 约束2：淘汰约束（带松弛变量）

淘汰者的综合排名应为最差：

$$
R_{e(w),w} + \delta_{j,w} \geq R_{j,w} + 1, \quad \forall j \in S_w \setminus \{e(w)\}
$$

$$
\delta_{j,w} \geq 0
$$

其中 $+1$ 确保严格最差（避免并列）。松弛变量 $\delta_{j,w}$ 允许在无法完全满足淘汰约束时付出代价。

### 4. 目标函数

目标函数由三个加权项组成：

#### (A) 评委贴近项（Judge Term）

观众排名应与评委排名相近：

$$
J_{term} = \sum_w \sum_{i \in S_w} \left( r^F_{i,w} - r^J_{i,w} \right)^2
$$

**数学直觉**：假设观众和评委对选手表现的评价具有相关性。

#### (B) 跨周平滑项（Smoothing Term）

同一选手相邻周的排名变化应平滑：

$$
S_{term} = \sum_{w \geq 2} \sum_{i \in S_w \cap S_{w-1}} \left( r^F_{i,w} - r^F_{i,w-1} \right)^2
$$

**数学直觉**：选手的观众支持度不会剧烈波动。

#### (C) 松弛惩罚项（Slack Penalty）

尽量满足淘汰约束：

$$
\delta_{term} = \sum_w \sum_{j \in S_w \setminus \{e(w)\}} \delta_{j,w}
$$

#### 总目标函数

$$
\min\ \alpha \cdot J_{term} + \beta \cdot S_{term} + \gamma \cdot \delta_{term}
$$

其中 $\alpha, \beta, \gamma > 0$ 为权重参数。

---

## 算法实现

### 1. 约束规划求解器（CP-SAT）

采用 **Google OR-Tools CP-SAT** 求解器进行精确优化。

#### 模型构建流程

```
输入: 赛季数据 {season, week, contestants, judge_rank, eliminated_ids}
输出: 最优观众排名 rF_vars, 目标函数值

1. 创建 CP-SAT 模型
2. 对每周 w:
   2.1 创建二值变量 x_{i,k,w}
   2.2 创建整数变量 rF_{i,w} ∈ [1, n_w]
   2.3 添加排列约束
   2.4 链接 rF = Σ k·x_{i,k,w}
   2.5 计算评委贴近项 (rF - rJ)²
   2.6 若有淘汰:
       - 创建松弛变量 δ_{j,w}
       - 添加淘汰约束
3. 计算跨周平滑项 (rF_w - rF_{w-1})²
4. 构造加权目标函数
5. 调用 CP-SAT 求解
6. 提取最优解
```

#### 平方项处理

CP-SAT 通过 `AddMultiplicationEquality` 处理平方项：

```python
d_var = model.NewIntVar(-n_w, n_w, "d")  # d = rF - rJ
d_sq = model.NewIntVar(0, n_w*n_w, "d_sq")
model.Add(d_var == rF_var - rJ)
model.AddMultiplicationEquality(d_sq, [d_var, d_var])  # d_sq = d²
```

#### 权重缩放

为避免浮点精度问题，权重经过整数缩放：

$$
\text{scale} = 10^{\max(\text{decimal\_places})}
$$

例如 $\alpha=1.0, \beta=0.05, \gamma=10.0$ 缩放后为 $\alpha=100, \beta=5, \gamma=1000$（scale=100）。

### 2. 参数网格搜索

系统搜索最优权重组合：

| 参数 | 搜索空间 |
|------|----------|
| $\alpha$ | {1.0} |
| $\beta$ | {0.05, 0.1, 0.2} |
| $\gamma$ | {10.0, 50.0, 100.0, 200.0} |

**最优参数**：$\alpha=1.0, \beta=0.05, \gamma=10.0$（目标值=2710.75）

### 3. 淘汰预测

给定观众排名解 $r^F$，预测淘汰选手：

```
1. 计算综合排名 R_i = r^J_i + r^F_i
2. 按 R_i 降序排列
3. 取 top-k 作为预测淘汰（k = 实际淘汰人数）
```

---

## 不确定性分析

### 1. 近最优区间分析（Near-Optimal Interval）

#### 方法

对于给定容忍度 $\varepsilon \in \{0.01, 0.05, 0.1\}$，在近最优可行域内求排名区间：

$$
\mathcal{F}_\varepsilon = \{x : f(x) \leq (1+\varepsilon) \cdot f^*\}
$$

对每个选手 $(i, w)$，分别求解：

$$
r^F_{i,w}^{\min} = \min_{x \in \mathcal{F}_\varepsilon} r^F_{i,w}
$$

$$
r^F_{i,w}^{\max} = \max_{x \in \mathcal{F}_\varepsilon} r^F_{i,w}
$$

#### 淘汰稳定性判定

基于综合排名区间 $[R_i^{\min}, R_i^{\max}]$ 判定：

| 状态 | 条件 |
|------|------|
| `always_safe` | $|\{j \neq i : R_j^{\min} > R_i^{\max}\}| \geq k$ |
| `always_eliminated` | $|\{j \neq i : R_j^{\max} > R_i^{\min}\}| \leq k-1$ |
| `uncertain` | 其他情况 |

### 2. 输入扰动分析（Input Perturbation）

#### 扰动模型

对评委分数添加高斯噪声：

$$
\tilde{s}_{i,w} = s_{i,w} + \varepsilon_{i,w}, \quad \varepsilon_{i,w} \sim \mathcal{N}(0, \sigma^2)
$$

默认参数：$\sigma = 0.7$，样本数 $N = 500$。

#### 统计量计算

对扰动样本计算：

- **排名统计**：$\text{mean}(r^F), \text{std}(r^F), P_{05}, P_{95}$
- **淘汰概率**：$P(\text{eliminated}) = \frac{1}{N}\sum_{n=1}^N \mathbb{1}[\text{contestant eliminated in sample } n]$
- **秩相关**：与基准解的 Spearman $\rho$ 和 Kendall $\tau$

#### 秩相关系数

**Spearman 秩相关**：

$$
\rho = \frac{\text{Cov}(\text{rank}_X, \text{rank}_Y)}{\sigma_{\text{rank}_X} \cdot \sigma_{\text{rank}_Y}}
$$

**Kendall tau-a**：

$$
\tau_a = \frac{C - D}{\binom{n}{2}}
$$

其中 $C$ 为一致对数，$D$ 为不一致对数。

### 3. 交替最优化扰动（Alternating Optimization）

#### 动机

CP-SAT 精确求解较慢，不适合大量扰动样本。交替最优化提供快速启发式。

#### 算法流程

```
输入: weeks, 权重 (α, β, γ), sweeps 次数
输出: 近似最优观众排名 rF

1. 初始化 rF（使用基准解或评委排名）
2. for sweep in 1..sweeps:
   2.1 for week in weeks:
       - 固定相邻周排名 rF_{w-1}, rF_{w+1}
       - 构造成本矩阵 C[i,k]
       - 求解指派问题（匈牙利算法）
       - 更新 rF_w
3. 返回 rF
```

#### 成本矩阵构造

$$
C(i, k) = \alpha (k - r^J_i)^2 + \beta (k - r^F_{i,w-1})^2 + \beta (k - r^F_{i,w+1})^2 + \gamma_{\text{elim}} \cdot (n_w - k) \cdot \mathbb{1}[i = e(w)]
$$

最后一项鼓励被淘汰者获得差排名（$k$ 大时 $n_w - k$ 小）。

#### 匈牙利算法

每周的指派问题可在 $O(n_w^3)$ 内精确求解：

$$
\min_{\pi} \sum_{i \in S_w} C(i, \pi(i))
$$

其中 $\pi$ 为排列。

---

## 投票百分比建模（Zipf分布）

### Zipf 分布模型

观众投票通常服从幂律分布。给定观众排名 $r^F_i$，建模投票份额：

$$
z_i = \frac{1}{(r^F_i + \beta)^\alpha}
$$

$$
p_i = \frac{z_i}{\sum_j z_j} \times 100\%
$$

### 参数选择

| 参数 | 默认值 | 含义 |
|------|--------|------|
| $\alpha$ | 0.9 | Zipf 指数（控制头部集中度） |
| $\beta$ | 1.0 | 平滑参数（避免 $r^F=0$ 时除零） |

**数学依据**：经验表明社交媒体投票、网络流量等服从 Zipf 分布（$\alpha \approx 1$）。

---

## 项目结构

```
/home/hisheep/d/MCM/26/
├── 2026_MCM_Problem_C.pdf          # 原始问题文档
├── Data_4.xlsx                      # 输入数据（9个赛季）
├── prompt1-4.md                     # 数学模型 LaTeX 公式
├── README.md                        # 本文件
└── task1-4/                         # 核心代码与输出
    ├── data_processing.py           # 数据预处理模块
    ├── model_rank.py                # CP-SAT 模型构建
    ├── solve_rank.py                # 基准求解主程序
    ├── grid_search.py               # 参数网格搜索
    ├── uncertainty_analysis.py      # CP-SAT 不确定性分析
    ├── uncertainty_altopt.py        # 交替最优化扰动
    ├── zipf_vote_percent.py         # Zipf 投票百分比
    ├── requirements.txt             # Python 依赖
    ├── outputs/                     # 基准输出
    │   ├── grid_search_summary.csv  # 网格搜索汇总
    │   └── grid_a1p0_b0p05_g10p0/   # 最优参数结果
    ├── outputs_uncertainty/         # CP-SAT 不确定性结果
    └── outputs_uncertainty_altopt/  # 交替最优化结果
```

---

## 核心代码模块说明

### data_processing.py

**功能**：从 Excel 数据构建周级输入结构

**关键函数**：
- `get_week_columns(df)`：提取 `week{N}_judge_score` 格式列
- `parse_elimination_week(value)`：解析 "Eliminated Week X" 字符串
- `build_season_weeks(df, season, week_cols)`：构建赛季数据结构

**排名计算**：采用 competition ranking（并列同名次，跳过后续）

$$
\text{rank}(-\text{score}, \text{method}=\text{'min'})
$$

### model_rank.py

**功能**：构建 CP-SAT 优化模型

**核心函数**：
- `build_rank_model(weeks, weights_scaled)`：构建模型变量与约束
- `solve_season(weeks, weights_scaled, weight_scale, time_limit, num_workers)`：求解并返回结果

**变量类型**：
- `BoolVar`：二值指派变量 $x_{i,k,w}$
- `IntVar`：观众排名 $r^F_{i,w}$、松弛 $\delta_{j,w}$

### solve_rank.py

**功能**：基准求解主程序

**输出文件**：
- `weekly_predictions.csv`：周级预测结果
- `consistency_summary.csv`：淘汰一致性统计
- `weekly_penalty.csv`：目标函数分解
- `optimization_info.json`：求解元信息

### uncertainty_analysis.py

**功能**：基于 CP-SAT 的不确定性量化

**分析类型**：
1. 近最优区间（$\varepsilon = 0.01, 0.05, 0.1$）
2. 输入扰动（500次蒙特卡洛）

**并行化**：使用 `ProcessPoolExecutor` 多进程加速

### uncertainty_altopt.py

**功能**：交替最优化快速扰动分析

**算法特点**：
- 使用匈牙利算法求解指派子问题
- 多轮 sweep 迭代收敛
- 速度快，适合大量样本

---

## 实验结果

### 基准求解结果

| 赛季 | 状态 | 目标值 | 淘汰一致性 |
|------|------|--------|----------|
| 1 | OPTIMAL | 44.10 | 100% |
| 2 | OPTIMAL | 71.40 | 100% |
| 28 | OPTIMAL | 354.80 | 85.71% |
| 29 | OPTIMAL | 358.75 | 100% |
| 30 | OPTIMAL | 643.40 | 75.00% |
| 31 | OPTIMAL | 185.55 | 100% |
| 32 | OPTIMAL | 384.40 | 66.67% |
| 33 | OPTIMAL | 280.00 | 83.33% |
| 34 | OPTIMAL | 388.35 | 87.50% |
| **总计** | - | **2710.75** | **88.58%** |

### 参数敏感性分析

| $\alpha$ | $\beta$ | $\gamma$ | 目标值 | 说明 |
|----------|---------|----------|--------|------|
| 1.0 | 0.05 | 10.0 | 2710.75 | **最优** |
| 1.0 | 0.10 | 10.0 | 2907.30 | 次优 |
| 1.0 | 0.20 | 10.0 | 3269.80 | - |
| 1.0 | 0.05 | 50.0 | 3936.95 | - |
| 1.0 | 0.05 | 200.0 | 7236.95 | 最差 |

**结论**：
- $\beta$ 越小（平滑权重低），目标值越优
- $\gamma$ 越大（淘汰惩罚高），目标值增加但一致性提升

---

## 使用方法

### 环境配置

```bash
cd /home/hisheep/d/MCM/26/task1-4
pip install -r requirements.txt
```

**依赖**：`ortools`, `pandas`, `numpy`, `openpyxl`

### 基准求解

```bash
python solve_rank.py \
    --alpha 1.0 --beta 0.05 --gamma 10.0 \
    --time-limit 60 --workers 8
```

### 参数网格搜索

```bash
python grid_search.py \
    --alphas 1.0 \
    --betas 0.05,0.1,0.2 \
    --gammas 10,50,100,200 \
    --time-limit 120
```

### 不确定性分析（CP-SAT）

```bash
python uncertainty_analysis.py \
    --epsilons 0.01,0.05,0.1 \
    --n-samples 500 \
    --noise-std 0.7 \
    --processes 12
```

### 交替最优化扰动

```bash
python uncertainty_altopt.py \
    --sweeps 5 \
    --n-samples 500 \
    --processes 12
```

### Zipf 投票百分比

```bash
python zipf_vote_percent.py \
    --alpha 0.9 --beta 1.0
```

---

## 模型创新点

### 1. 排列约束的精确建模

使用二值指派变量 $x_{i,k,w}$ 精确表示排名的组合结构，避免了连续松弛导致的非整数解。

### 2. 软约束与硬约束结合

通过松弛变量 $\delta_{j,w}$ 将淘汰约束软化，在目标函数中惩罚违反，实现了约束可行性与模型灵活性的平衡。

### 3. 多目标权衡

三项目标（评委贴近、跨周平滑、淘汰一致）的加权组合，通过网格搜索找到最优权衡点。

### 4. 双层不确定性量化

- **近最优分析**：量化模型参数的敏感性
- **输入扰动**：量化数据噪声的影响

### 5. 精确+启发式互补

CP-SAT 用于基准求解和严格不确定性分析，交替最优化用于大规模扰动的快速评估。

---

## 论文撰写建议

### 问题建模部分

1. 清晰定义符号表
2. 从物理意义解释约束（排列、淘汰）
3. 论证目标函数各项的合理性

### 算法设计部分

1. CP-SAT 模型的完整数学表述
2. 平方项的线性化/乘法约束处理
3. 权重缩放的技术细节

### 不确定性分析部分

1. 近最优可行域的定义与求解
2. 蒙特卡洛扰动的统计设计
3. 秩相关系数的选择依据

### 结果分析部分

1. 淘汰一致性的赛季差异分析
2. 参数敏感性的物理解释
3. 不确定区间的决策意义

---

## 参考文献格式建议

- Google OR-Tools CP-SAT 求解器
- Zipf 分布在投票模型中的应用
- Spearman/Kendall 秩相关系数
- 约束规划与组合优化

---

## 作者信息

2026 MCM Problem C 项目

---

## 版本历史

| 日期 | 版本 | 说明 |
|------|------|------|
| 2026-01-31 | v1.0 | 初始版本 |
