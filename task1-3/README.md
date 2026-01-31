# 任务1-3：百分比法观众投票权重估计

本目录包含基于 `Data_4.xlsx` 的百分比法（赛季 3–27a，对应季号 3–27）观众投票权重反推脚本与结果。实现遵循 `prompt2.md` 的 softmax + 排序损失（pairwise hinge）目标函数，并将脚本与输出保持在 `task1-3`。

## 关键假设与规则

- 使用题面“百分比法”赛季 3–27；赛季过滤以季号 3–27 实现。
- 每周样本仅包含当周有裁判分的选手（`weekX_judge_score > 0`）。
- `results == "Withdrew"` 的选手不作为淘汰者参与约束，但仍保留其当周得分进入总分计算。
- 若某周无淘汰者，则该周不进入目标函数；双淘汰周按每名淘汰者分别计罚并累加。
- 裁判百分比由当周裁判总分归一化计算（与数据中的 `X_percent` 一致）。
- 特征范围：裁判分（标准化总分 + 百分比，可关闭总分）、年龄、职业（one-hot）、国家/地区（one-hot）。
- 低频类别合并为 `Other`（默认阈值 `min_category_count=3`）。
- 排序损失采用 `max(0, margin + S_elim - S_survive)` 的 pairwise hinge 形式。
- 赛季效应默认通过“赛季 × 特征”交互项进入权重（如 `season_5__judge_score_std`），允许不同赛季对特征权重产生偏移。

## 运行方式

```bash
python /home/hisheep/d/MCM/26/task1-3/solve_percent.py
```

可选参数：

- `--margin`：惩罚边界（默认 0.01）
- `--l2`：L2 正则系数（默认 0.0）
- `--maxiter`：最大迭代次数（默认 500）
- `--min-category-count`：类别保留阈值（默认 3）
- `--no-season-interactions`：关闭赛季交互特征，仅使用基础特征
- `--no-judge-score`：关闭 `judge_score_std` 特征，仅保留裁判百分比与其他特征

## 输出说明（`task1-3/outputs`）

- `weights_summary.csv`：特征权重（含赛季交互项，若关闭则仅基础特征）
- `weekly_predictions.csv`：每周每位选手的裁判百分比、预测观众百分比、总分与淘汰对比
- `weekly_penalty.csv`：每周惩罚值与命中情况
- `consistency_summary.csv`：按赛季与总体一致性统计
- `optimization_info.json`：优化收敛信息
