#### **Prompt 指令：使用 Scipy 优化求解观众投票权重**

**任务目标：**

使用传统的**线性模型**和**数学优化方法 (`scipy.optimize`)** 来反推观众投票逻辑。不使用 PyTorch 或神经网络。

**1. 数据准备：**

- 从 CSV 文件  中读取数据。

	

	

- 提取特征 $X$（矩阵）：包括归一化的裁判分、年龄、是否是歌手/演员（One-hot编码）等。

- 提取目标 $Y$：每一周被真实淘汰的选手索引。

**2. 定义模型函数：**

定义一个函数 `predict_fan_scores(weights, features)`：

- `weights`: 待优化的参数数组 (1D array)。
- `features`: 选手特征矩阵。
- 逻辑: `raw_score = dot(features, weights)`。
- 转换: 使用 `softmax` 将 `raw_score` 转换为百分比 `fan_percent`。

**3. 定义目标函数 (Loss Function)：**

定义 `objective_function(weights)`：

- 遍历每一周的数据。
- 计算当周所有选手的 `total_score = judge_percent + predict_fan_scores(...)`。
- 找到真实淘汰者 $k$ 的分数 `S_k`。
- 找到所有晋级者的分数集合 $\{S_j\}$。
- **计算惩罚**：对于每一个晋级者 $j$，如果 `S_k > S_j` （预测错误，淘汰者分反而高），则累加惩罚 `penalty += (S_k - S_j) + margin`。
- 返回总 `penalty`。

**4. 求解与结果：**

- 使用 `scipy.optimize.minimize`，方法选择 `'L-BFGS-B'` 或 `'SLSQP'`。
- 初始权重 `initial_weights` 可以全设为随机小数。
- 输出最优权重 `optimized.x`。

**5. 分析输出：**

- 打印出每个特征对应的权重值（例如：裁判分权重是多少，年龄权重是多少），并解释其物理意义。
- 计算**一致性 (Consistency)**：用优化后的权重重新跑一遍历史数据，看有多少周预测出的淘汰者和真实一致。