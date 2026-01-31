**任务目标：**

编写一个基于 PyTorch 的机器学习模型，用于解决“反向排名问题”。已知“裁判打分”和“淘汰结果”，需要反向学习“观众投票”的逻辑。

**适用规则：** 百分比制（Percentage Rule），即 `总分 = 裁判分占比 + 观众分占比`，总分最低者被淘汰。

#### **1. 数据结构 (Data Structure)**

请构建一个 `Dataset` 类，每一条数据代表**“一周的比赛”**。

- **输入特征 (`features`)**: 张量形状 `(N, D)`。
	- `N`: 当周参赛选手数量（不同周 N 可能不同，需处理变长输入或使用 Mask）。
	- `D`: 每个选手的特征维度（如：归一化的裁判分、年龄、上周是否濒临淘汰、社交媒体热度等）。
- **裁判分数 (`judge_raw_scores`)**: 张量形状 `(N,)`，记录当周每个选手的裁判原始总分。
- **真实淘汰者索引 (`eliminated_idx`)**: 标量 `int`，指示哪位选手在当周被淘汰了（Ground Truth）。

#### **2. 模型架构 (Model Architecture)**

请定义一个 `AudiencePreferenceModel` 类：

- **层结构**: 一个简单的多层感知机 (MLP)。
	- Input: `D` (特征维度)
	- Hidden: 比如 64 -> 32 (带 ReLU 激活)
	- Output: 1 (输出每个选手的“原始人气值” `raw_popularity`)
- **注意**: 不需要最后的激活函数（如 Sigmoid），因为我们在后续步骤会做归一化。

#### **3. 前向传播与分数计算逻辑 (Forward Pass & Scoring Logic)**

在 `forward` 函数中实现以下“百分比制”规则：

1. **预测观众分**:
	- 将 `features` 输入模型，得到 `raw_popularity`，形状 `(N, 1)`。
	- **关键转换**: 使用 **Softmax** 函数将 `raw_popularity` 转换为**观众投票占比 (`fan_percent`)**。
	- 公式: `fan_percent = softmax(raw_popularity, dim=0)`。这保证了所有选手观众票占比之和为 100%。
2. **计算裁判分占比**:
	- 输入: `judge_raw_scores`。
	- 计算: `judge_percent = judge_raw_scores / sum(judge_raw_scores)`。
3. **计算总分**:
	- `total_score = judge_percent + fan_percent`。
4. **返回**: `total_score`, `fan_percent`。

#### **4. 损失函数设计 (Custom Ranking Loss)**

请实现一个自定义损失函数 `PercentageEliminationLoss`。

**逻辑目标**: 让“真实淘汰者”的总分比“所有晋级者”的总分都要**低**。

- **输入**:

	- `total_scores`: 模型计算出的当周所有选手总分，形状 `(N,)`。
	- `target_idx`: 真实被淘汰选手的索引 `k`。
	- `margin`: 超参数，例如 `0.01`（强制要求淘汰者比晋级者低出的安全距离）。

- **计算步骤**:

	1. 提取淘汰者的分数: `score_eliminated = total_scores[target_idx]`。

	2. 提取所有晋级者的分数: `scores_survivors` (通过掩码 Mask 去除 `target_idx` 对应的值)。

	3. **成对损失 (Pairwise Loss)**:

		对于每一个晋级者 $j$，如果 `score_eliminated > score_survivors[j] - margin`，则产生损失。

		公式: `loss = sum(ReLU(score_eliminated - scores_survivors + margin))`。

	4. **辅助正则化 (Optional)**: 可以加一个 L2 正则项防止 `raw_popularity` 数值爆炸。

#### **5. 训练循环与评估 (Training Loop & Evaluation)**

- **优化器**: Adam, learning rate 建议 0.001。
- **Evaluation Metrics (评估指标)**:
	1. **一致性 (Consistency/Accuracy)**:
		- 计算 `predicted_eliminated_idx = argmin(total_scores)`。
		- 判断 `predicted_eliminated_idx == target_idx` 是否成立。
		- 返回 `Accuracy = Correct_Weeks / Total_Weeks`。
	2. **确定性 (Certainty)**:
		- 对于真实淘汰者，计算其与“倒数第二名”（最危险的晋级者）的分数差值。
		- `Gap = min(scores_survivors) - score_eliminated`。
		- `Gap` 越大，确定性越高；`Gap` 为负数说明预测错误。