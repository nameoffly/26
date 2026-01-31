# task1-4 名次排序模型（Rank）

本目录用于赛季 1、2、28–34 的观众名次（`rF`）推断与淘汰一致性分析，并提供
不确定性评估（近最优区间 + 输入扰动）。数据源为 `/home/hisheep/d/MCM/26/Data_4.xlsx`。

## 建模思路

1) **名次变量（排列约束）**
- 对每周设置二值变量 `x_{i,k,w}`，表示选手 `i` 在第 `w` 周处于名次 `k`。
- 约束每位选手恰好一个名次、每个名次恰好一位选手，形成一个排列。
- 由此得到观众名次 `rF_{i,w} = sum_k k * x_{i,k,w}`。

2) **评委名次**
- 使用当周裁判总分降序排名（competition ranking / `rank(method="min")`）。
- 记为 `rJ`，同分时使用最小名次。

3) **综合名次与淘汰约束**
- 综合名次：`R = rJ + rF`（值越大越差）。
- 对实际淘汰选手 `e`，要求其综合名次不优于任一非淘汰选手：
  `rF_e + rJ_e + delta >= rF_j + rJ_j + 1`
- `delta >= 0` 为松弛变量，允许违反并计入惩罚（对应一致性不足）。

4) **目标函数**
- 贴近评委：`alpha * sum (rF - rJ)^2`
- 跨周平滑：`beta * sum (rF_w - rF_{w-1})^2`（同一选手跨周）
- 淘汰一致性松弛：`gamma * sum delta`

求解器使用 OR-Tools CP-SAT，平方项通过乘法等式精确表示。

## 数据处理规则

- 只保留当周 `judge_score > 0` 且非 NA 的选手。
- 仅解析 `results` 中的 `"Eliminated Week X"`；`Withdrew` 或无淘汰周不加淘汰约束。
- 解析 `results` 中的名次（如 `1st/2nd/3rd/4th/5th Place`），得到决赛名次 `final_place`。
- `judge_rank` 使用 competition ranking（并列同名次，下一名次跳过）。
- **决赛周约束**：每季最后一周视为决赛周，仅对 `results` 中含名次的决赛选手生效。
  若 A 的 Place 更高（数值更小），则强制 `R_A <= R_B`，**允许并列**。

## 程序结构与作用

- `data_processing.py`  
  构建周级输入：筛选有效选手、计算 `judge_rank`、解析淘汰周。

- `model_rank.py`  
  - `build_rank_model(...)`：构建 CP-SAT 模型并返回变量/目标表达式。  
  - `solve_season(...)`：完整求解并输出 `rF`、周惩罚、松弛等结果。
  - **决赛约束**：在决赛周加入 `AllDifferent(R_iw)`，并用 `R_iw = rJ_i + rF_i` 约束综合名次。

- `solve_rank.py`  
  单次求解主脚本：按 season 运行模型并输出预测结果与一致性统计。

- `grid_search.py`  
  参数网格搜索，输出每组参数的 objective 汇总，用于挑选最优权重。

- `uncertainty_analysis.py`  
  不确定性分析（近最优区间 + 输入扰动），并行执行、汇总输出。

## 运行方式

安装依赖：

```
pip install -r requirements.txt
```

### 基准求解

```
python solve_rank.py
```

自定义示例：

```
python solve_rank.py --seasons 1 2 28 29 30 31 32 33 34 --alpha 1 --beta 0.05 --gamma 10 --time-limit 60
```

### 网格搜索

```
python grid_search.py --time-limit 120
```

汇总结果：`task1-4/outputs/grid_search_summary.csv`

### 不确定性分析（推荐）

默认配置即为：ε=1%/5%/10%，噪声 N(0,0.7)，500 次扰动，CPU 并行。

```
python uncertainty_analysis.py
```

并行与参数控制示例：

```
python uncertainty_analysis.py \
  --processes 12 \
  --week-group-size 3 \
  --n-samples 500 \
  --noise-std 0.7 \
  --seed 42
```

参数说明（节选）：
- `--baseline-dir`：包含 `weekly_predictions.csv` 与 `optimization_info.json` 的目录
- `--week-group-size`：近最优任务的周分组大小（越小越细，但任务更多）
- `--batch-size`：扰动批大小（0 表示自动均分）
- `--near-opt-time-limit` / `--perturb-time-limit`：单次求解时限
- `--processes`：并行进程数（默认 `min(cpu_count, 12)`，每进程单线程求解）

### 交替最优化扰动分析（新脚本，速度优先）

该模式使用交替最优化（逐周指派 + 多轮 sweep）替代 CP-SAT，仅做输入扰动分析，速度更快。
脚本：`task1-4/uncertainty_altopt.py`，默认输出目录：
`task1-4/outputs_uncertainty_altopt/perturb`。

运行默认配置（4-6 轮 sweep，基于基准解初始化）：

```
python uncertainty_altopt.py
```

典型参数示例：

```
python uncertainty_altopt.py \
  --processes 12 \
  --n-samples 300 \
  --noise-std 0.7 \
  --sweeps 5 \
  --init baseline
```

核心参数说明：
- `--sweeps`：每个扰动样本的交替最优化轮数（建议 4-6）。
- `--init`：初始化方式（`baseline` 使用基准 `fan_rank`；`judge` 使用评委名次）。
- `--elim-weight`：淘汰倾向权重（默认跟随 `gamma`）。
- `--cost-scale`：指派问题成本缩放因子（整数化用，默认 1000）。

输出文件（与 CP-SAT 扰动格式一致，便于对比）：
- `perturb_rF_stats.csv`：扰动下 `fan_rank` 的均值/标准差/P05/P95。
- `perturb_elim_prob.csv`：扰动下被预测淘汰的概率。
- `perturb_rank_stability.csv`：与基准解 Spearman/Kendall 稳定性统计。

注意事项：
- 该脚本是启发式求解，不保证最优；适合大量扰动快速评估。
- 若需更高精度，请使用 `uncertainty_analysis.py` 的 CP-SAT 扰动。

### 可视化（PNG）

脚本：`task1-4/plot_results.py`，输出目录：`task1-4/outputs_images`。

运行默认配置（所有季度合并输出）：

```
python plot_results.py
```

常用参数：
- `--output-dir`：图片输出目录（默认 `task1-4/outputs_images`）
- `--baseline-dir`：基准结果目录（默认 `outputs/grid_a1p0_b0p05_g10p0`）
- `--perturb-dir`：扰动不确定性目录（默认 `outputs_uncertainty_altopt/perturb`）
- `--seasons`：指定赛季（默认全部赛季）
- `--top-k`：轨迹与误差条图的 Top-K（默认 6）
- `--vote-week`：指定投票分布周（默认每季最后一周）
- `--zipf-beta`：Zipf log-log 里的 beta（默认 1.0）
- `--mode`：绘图模式（`all` 合并所有季度；`per-season` 每季一组；`both` 两者都有）

输出图示例（合并所有季度）：
- `all_rank_distribution.png`
- `all_judge_vs_fan.png`
- `all_consistency_heatmap.png`
- `all_penalty_mean.png`
- `all_vote_distribution.png`
- `all_zipf_loglog.png`
- `all_fan_rank_std.png`
- `all_elim_prob_heatmap.png`
- `all_rank_stability.png`

### 数学说明（交替最优化扰动）

扰动分析对评委分数加入噪声后，使用交替最优化近似最小化目标函数：

- 评委贴近：\u03b1\u22c5( r^F_{i,w} - r^J_{i,w} )^2
- 跨周平滑：\u03b2\u22c5( r^F_{i,w} - r^F_{i,w-1} )^2 + \u03b2\u22c5( r^F_{i,w} - r^F_{i,w+1} )^2
- 淘汰倾向：elim_weight\u22c5P(i,k)（对淘汰者在更差名次给予额外代价）

在固定相邻周名次的条件下，第 w 周可以转化为指派问题：为每位选手 i 选择名次 k=1..n_w，
最小化

```
C(i,k) = \u03b1 (k-r^J_{i,w})^2 + \u03b2 (k-r^F_{i,w-1})^2 + \u03b2 (k-r^F_{i,w+1})^2 + elim_weight * P(i,k)
```

这里 P(i,k) 默认可取 (n_w - k)（让淘汰者倾向于更差名次）。每个 sweep 按周解一次指派（匈牙利算法），
更新 r^F，再重复 4-6 轮以逼近局部最优。

### 数学说明（Spearman/Kendall 稳定性）

稳定性统计基于“基准综合名次”与“扰动综合名次”的秩相关：

- Spearman \u03c1：对“秩序排名”的 Pearson 相关

```
rho = corr( rank(x), rank(y) )
```

若存在并列，可用 average rank（平均秩）处理。

- Kendall tau-a：基于一致/不一致对数

```
tau = (C - D) / (n(n-1)/2)
```

其中 C 为一致对数，D 为不一致对数。若并列较多，可考虑 tau-b（对并列修正）。

### 数学说明（近最优区间分析）

近最优分析基于 CP-SAT 的全季模型。先求得原问题最优目标值 Opt，然后对每个
\u03b5 \u2208 {0.01, 0.05, 0.1} 加入近最优约束：

```
Objective <= (1 + \u03b5) * Opt
```

在该约束下，对每个选手 i 与周 w，分别解两个优化问题：

- rF_min(i,w) = min rF_{i,w}
- rF_max(i,w) = max rF_{i,w}  (等价于 min -rF_{i,w})

由此得到综合名次区间：

```
R_min(i,w) = rJ_{i,w} + rF_min(i,w)
R_max(i,w) = rJ_{i,w} + rF_max(i,w)
```

由于需要为大量 (i,w) 解两次优化，程序会按 week_group 分块并行执行；分块只影响
求解任务的分发，不改变模型约束。

淘汰稳定性基于“更差名次”的区间比较。设当周真实淘汰人数为 k：

- 更差的确定人数：
  worse_definite(i) = #{ j != i | R_min(j) > R_max(i) }
- 更差的可能人数：
  worse_possible(i) = #{ j != i | R_max(j) > R_min(i) }

判定规则：

- 若 worse_definite(i) >= k，则 i 一定安全（always_safe）
- 若 worse_possible(i) <= k-1，则 i 一定淘汰（always_eliminated）
- 否则为不确定（uncertain）

该规则与“综合名次越大越差”的淘汰逻辑一致，并考虑了近最优区间的不确定性。

## 结果解读

### 基准输出（`task1-4/outputs/...`）
- `weekly_predictions.csv`  
  每周每位选手的评委名次、观众名次、综合名次与淘汰预测。
- `consistency_summary.csv`  
  每周淘汰一致性（是否覆盖真实淘汰者）与命中率。
- `weekly_penalty.csv`  
  三类目标项（贴近评委/平滑/淘汰松弛）的周级分解。
- `optimization_info.json`  
  权重、求解状态、耗时与各 season 的目标值。
- `final_rank`（在 `weekly_predictions.csv` 中）：  
  决赛周（每季最后一周）综合名次的排序结果，按 `combined_rank = rJ + rF` 升序得到；仅在决赛周行填值，其余周为空。

### 近最优区间输出（`outputs_uncertainty/near_opt`）
- `near_opt_interval.csv`  
  在 `objective <= (1+ε)*opt` 约束下的 `rF` 与 `R` 区间（min/max）。
- `near_opt_elim_certainty.csv`  
  基于区间的淘汰稳定性：`always_safe / always_eliminated / uncertain`。

### 交替最优化近最优区间（无求解器）

脚本：`task1-4/near_opt_altopt.py`  
输出目录：`task1-4/outputs_uncertainty_altopt/near_opt_altopt`

思路：用交替最优化生成多组候选解，并用基准 Opt 过滤近最优解。

1) 读取基准 `optimization_info.json` 中的 `Opt`  
2) 用交替最优化（baseline 初始化，多轮 sweep）生成 N 个候选 `rF`  
3) 用近似目标值计算：

```
Objective = alpha * Jterm + beta * Smooth + gamma * Slack
```

4) 保留满足 `Objective <= (1+ε)*Opt` 的候选解  
5) 对保留解计算 `rF_min/rF_max` 与 `R_min/R_max`，并导出稳定性判断

常用参数：
- `--n-samples`：候选解数量（默认 10）
- `--sweeps`：交替最优化轮数（默认 5）
- `--epsilons`：近最优比例（默认 0.01,0.05,0.1）

输出文件：
- `near_opt_interval.csv`
- `near_opt_elim_certainty.csv`
- `near_opt_summary.json`（记录参数）

### 输入扰动输出（`outputs_uncertainty/perturb`）
- `perturb_rF_stats.csv`  
  扰动下 `rF` 的均值、标准差、P05/P95。
- `perturb_elim_prob.csv`  
  扰动下被预测淘汰的概率。
- `perturb_rank_stability.csv`  
  与基准解的 Spearman（按秩）与 Kendall（tau-a）稳定性统计。

## 并行说明与注意事项

- 并行采用多进程：`ProcessPoolExecutor(max_workers=processes)`。
- 每个进程内固定 `num_search_workers=1`，避免 CPU 过度竞争。
- 建议先跑近最优，再跑扰动，避免两类重任务同时占用资源。

