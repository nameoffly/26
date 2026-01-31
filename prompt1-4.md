# 变量与已知量

对某一赛季，设第 $w$ 周仍在场选手集合为 $S_w$，人数 $n_w = \lvert S_w \rvert$。

- 已知：评委名次 $r^J_{i,w} \in \{1,\ldots,n_w\}$（你已算好）。
- 未知（观众名次的排列）：用指派变量

$$
x_{i,k,w} \in \{0,1\},\quad i \in S_w,\ k=1,\ldots,n_w
$$

表示“选手 $i$ 在第 $w$ 周的观众名次为 $k$”。

由此定义观众名次：

$$
r^F_{i,w} = \sum_{k=1}^{n_w} k\,x_{i,k,w}
$$

综合名次和（Rank 规则只用名次和）：

$$
R_{i,w} = r^J_{i,w} + r^F_{i,w}
$$

# 约束 1：观众名次必须是一个排列

每个选手恰好一个名次：

$$
\sum_{k=1}^{n_w} x_{i,k,w} = 1,\quad \forall i \in S_w
$$

每个名次恰好给一个选手：

$$
\sum_{i\in S_w} x_{i,k,w} = 1,\quad \forall k=1,\ldots,n_w
$$

# 约束 2：淘汰规则 + 松弛变量（你要用的）

设第 $w$ 周真实淘汰者为 $e(w)$。

原本硬约束是：淘汰者综合最差

$$
R_{e(w),w} \ge R_{j,w} + 1,\quad \forall j \in S_w \setminus \{e(w)\}
$$

加入松弛变量 $\delta_{j,w} \ge 0$（允许少量违反，但要付代价）：

$$
R_{e(w),w} + \delta_{j,w} \ge R_{j,w} + 1,\quad \forall j \ne e(w)
$$

$$
\delta_{j,w} \ge 0
$$

注：这里的“+1”为了避免并列最差；如果你允许并列最差，再靠目标函数决定，可以把 +1 改成 +0。

# 目标函数：评委贴近 + 跨周平滑 + 松弛惩罚（加权）

你要求的形式可以写成：

(A) 贴近评委（同周）

$$
Jterm = \sum_w \sum_{i\in S_w} \left(r^F_{i,w} - r^J_{i,w}\right)^2
$$

(B) 跨周平滑（同一选手相邻周）

只对仍同时出现在两周的选手求：

$$
Smooth = \sum_{w\ge 2} \sum_{i\in S_w \cap S_{w-1}} \left(r^F_{i,w} - r^F_{i,w-1}\right)^2
$$

(C) 松弛惩罚（尽量别违反淘汰规则）

$$
Slack = \sum_w \sum_{j\in S_w \setminus \{e(w)\}} \delta_{j,w}
$$

（你也可以用平方 $\sum \delta^2$ 更强烈压大违规；但线性更常见，也更稳。）

总目标（加权）

$$
\min\ \alpha \cdot Jterm + \beta \cdot Smooth + \gamma \cdot Slack
$$
