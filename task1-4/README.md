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

## 输出文件

输出目录：`/home/hisheep/d/MCM/26/task1-4/outputs`

- `weekly_predictions.csv`：每周每位选手的评委名次、观众名次、综合名次及淘汰预测
- `consistency_summary.csv`：按周/按季淘汰一致性统计
- `weekly_penalty.csv`：目标函数三部分的按周分解（未加权）
- `optimization_info.json`：权重、求解状态、耗时等信息

## 说明

模型使用 OR-Tools CP-SAT 求解，目标中的平方项通过乘法等式精确表示。
若出现求解器未在时限内达到最优，将输出可行解并在 `optimization_info.json` 中标记状态。
