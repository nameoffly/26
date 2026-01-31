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
- `judge_rank` 使用 competition ranking（并列同名次，下一名次跳过）。

## 程序结构与作用

- `data_processing.py`  
  构建周级输入：筛选有效选手、计算 `judge_rank`、解析淘汰周。

- `model_rank.py`  
  - `build_rank_model(...)`：构建 CP-SAT 模型并返回变量/目标表达式。  
  - `solve_season(...)`：完整求解并输出 `rF`、周惩罚、松弛等结果。

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

### 近最优区间输出（`outputs_uncertainty/near_opt`）
- `near_opt_interval.csv`  
  在 `objective <= (1+ε)*opt` 约束下的 `rF` 与 `R` 区间（min/max）。
- `near_opt_elim_certainty.csv`  
  基于区间的淘汰稳定性：`always_safe / always_eliminated / uncertain`。

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

