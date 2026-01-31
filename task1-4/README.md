# task1-4 名次排序模型（Rank）

本目录基于 `prompt1-4.md` 的名次排序框架，使用 `Data_4.xlsx` 对赛季 1、2、28–34
估计每周观众名次，并尽量满足淘汰结果一致性。

## 模型概述

- 观众名次采用指派变量 `x_{i,k,w}`，保证每周名次是一个排列。
- 评委名次 `r^J` 由当周评委总分降序排名得到（并列用 competition ranking）。
- 综合名次 `R = r^J + r^F`。
- 淘汰约束：当周淘汰者综合名次应为最差，允许用松弛变量 `delta` 违约并计入惩罚。
- 目标函数包含三部分：贴近评委（同周）、跨周平滑（同选手相邻周）、松弛惩罚。

## 数据与处理规则

- 数据源：`/home/hisheep/d/MCM/26/Data_4.xlsx`
- 赛季范围：1、2、28–34
- 当周选手集合 `S_w`：该周评委总分 `> 0` 且非 NA
- 淘汰周解析：仅使用 `results` 中的 `"Eliminated Week X"`。若为 `Withdrew` 或无淘汰，跳过该周淘汰约束。

## 运行方式

安装依赖：

```
pip install -r requirements.txt
```

运行默认赛季：

```
python solve_rank.py
```

自定义参数示例：

```
python solve_rank.py --seasons 1 2 28 29 30 31 32 33 34 --alpha 1 --beta 0.1 --gamma 10 --time-limit 60
```

## 网格搜索（选择最小 objective）

默认网格：

- alpha = [1]
- beta = [0.05, 0.1, 0.2]
- gamma = [10, 50, 100, 200]

运行网格搜索（每个组合输出到独立目录，并汇总 objective）：

```
python grid_search.py --time-limit 120
```

汇总结果输出在：

- `task1-4/outputs/grid_search_summary.csv`

自定义网格示例（逗号分隔）：

```
python grid_search.py --alphas 1 --betas 0.05,0.1,0.2 --gammas 10,50,100 --time-limit 120
```

若需要重新覆盖已有输出，加 `--rerun`。

## 不确定性分析（近最优区间 + 输入扰动，CPU 并行）

本目录提供 `uncertainty_analysis.py`，按以下配置执行：

- 近最优区间：ε = 1% / 5% / 10%（相对比例）
- 输入扰动：评委分数加噪声 `N(0, 0.7)`，重复 500 次
- 并行策略：多进程并行，每个进程内部使用单线程求解
- 默认最优参数：`alpha=1.0, beta=0.05, gamma=10.0`
- 默认基准输出：`task1-4/outputs/grid_a1p0_b0p05_g10p0`

运行默认配置：

```
python uncertainty_analysis.py
```

常用参数示例：

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
- `--output-root`：不确定性分析输出根目录
- `--week-group-size`：近最优任务的周分组大小（0 表示整季一次任务）
- `--batch-size`：扰动批大小（0 表示自动按进程数均分）
- `--near-opt-time-limit` / `--perturb-time-limit`：每次求解时限

## 输出文件

输出目录：`/home/hisheep/d/MCM/26/task1-4/outputs`

- `weekly_predictions.csv`：每周每位选手的评委名次、观众名次、综合名次及淘汰预测
- `consistency_summary.csv`：按周/按季淘汰一致性统计
- `weekly_penalty.csv`：目标函数三部分的按周分解（未加权）
- `optimization_info.json`：权重、求解状态、耗时等信息

不确定性输出目录（默认）：`/home/hisheep/d/MCM/26/task1-4/outputs_uncertainty`

- `near_opt/near_opt_interval.csv`：近最优区间（`rF` 与 `R` 的最小/最大）
- `near_opt/near_opt_elim_certainty.csv`：区间推导的淘汰稳定性（`always_safe/always_eliminated/uncertain`）
- `perturb/perturb_rF_stats.csv`：扰动下 `rF` 的均值/方差/分位数
- `perturb/perturb_elim_prob.csv`：扰动下淘汰概率
- `perturb/perturb_rank_stability.csv`：与基准解的 Spearman/Kendall 稳定性统计

## 说明

模型使用 OR-Tools CP-SAT 求解，目标中的平方项通过乘法等式精确表示。
若出现求解器未在时限内达到最优，将输出可行解并在 `optimization_info.json` 中标记状态。
