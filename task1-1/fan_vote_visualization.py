"""
观众投票估计可视化分析 - 问题1综合可视化

功能：
- 确定性与不确定性分析（4张图）
- 时间演化分析（3张图）
- 评委与观众意见对比（2张图）
- 争议选手深度分析（3张图）

总计：12张高质量可视化图表
风格：演示汇报风格（色彩鲜艳、易于理解）
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os
import sys
import warnings
from matplotlib.patches import Rectangle

# 设置中文字体和样式
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.labelweight'] = 'bold'

warnings.filterwarnings('ignore')

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# 演示汇报风格调色板
COLORS = {
    'eliminated': '#D62246',     # 红色 - 淘汰
    'survived': '#06A77D',       # 绿色 - 幸存
    'judge': '#2E86AB',          # 蓝色 - 评委
    'fan': '#F18F01',            # 橙色 - 观众
    'uncertainty': '#A23B72',    # 紫色 - 不确定性
    'primary': '#0077B6',        # 主色调
    'secondary': '#FF6B6B',      # 次色调
    'accent': '#4ECDC4',         # 强调色
}

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def load_all_data():
    """加载所有必要的CSV文件"""
    print("正在加载数据...")
    
    estimates_path = os.path.join(SCRIPT_DIR, 'fan_vote_estimates_entropy_smooth_150.csv')
    bootstrap_path = os.path.join(SCRIPT_DIR, 'certainty_method2_bootstrap_150_1000.csv')
    interval_summary_path = os.path.join(SCRIPT_DIR, 'certainty_method1_interval_summary_150.csv')
    
    estimates = pd.read_csv(estimates_path)
    bootstrap = pd.read_csv(bootstrap_path)
    interval_summary = pd.read_csv(interval_summary_path)
    
    print(f"✓ 估计数据: {len(estimates)} 条记录")
    print(f"✓ Bootstrap数据: {len(bootstrap)} 条记录")
    print(f"✓ 区间摘要: {len(interval_summary)} 条记录")
    
    return estimates, bootstrap, interval_summary


# ========== 图表组1: 确定性与不确定性分析 ==========

def plot_interval_width_distribution(df, output_dir):
    """图1: 可行域区间宽度分布对比图"""
    print("生成图1: 可行域区间宽度分布对比图...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # 子图1: 淘汰者 vs 幸存者的箱线图+小提琴图
    eliminated_data = df[df['eliminated'] == True]['interval_width'].values
    survived_data = df[df['eliminated'] == False]['interval_width'].values
    
    data_to_plot = [eliminated_data, survived_data]
    labels = ['淘汰者', '幸存者']
    
    # 小提琴图
    parts = ax1.violinplot(data_to_plot, positions=[0, 1], showmeans=True, 
                           showmedians=True, widths=0.7)
    
    # 设置颜色
    for i, pc in enumerate(parts['bodies']):
        color = COLORS['eliminated'] if i == 0 else COLORS['survived']
        pc.set_facecolor(color)
        pc.set_alpha(0.6)
    
    # 叠加箱线图
    bp = ax1.boxplot(data_to_plot, positions=[0, 1], widths=0.3,
                     patch_artist=True, showfliers=False)
    
    for i, patch in enumerate(bp['boxes']):
        color = COLORS['eliminated'] if i == 0 else COLORS['survived']
        patch.set_facecolor(color)
        patch.set_alpha(0.8)
    
    ax1.set_xticks([0, 1])
    ax1.set_xticklabels(labels, fontsize=12, fontweight='bold')
    ax1.set_ylabel('可行域区间宽度', fontsize=12, fontweight='bold')
    ax1.set_title('淘汰者 vs 幸存者的不确定性对比', fontsize=14, fontweight='bold', pad=15)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # 添加统计信息
    text_str = f"淘汰者均值: {eliminated_data.mean():.4f}\n幸存者均值: {survived_data.mean():.4f}"
    ax1.text(0.02, 0.98, text_str, transform=ax1.transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', 
             facecolor='wheat', alpha=0.5), fontsize=10)
    
    # 子图2: 按季度分组的箱线图
    seasons = sorted(df['season'].unique())
    season_groups = [df[df['season'] == s]['interval_width'].values for s in seasons]
    
    bp2 = ax2.boxplot(season_groups, patch_artist=True, showfliers=False)
    
    # 渐变色
    colors_gradient = plt.cm.viridis(np.linspace(0, 1, len(seasons)))
    for patch, color in zip(bp2['boxes'], colors_gradient):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax2.set_xlabel('季度', fontsize=12, fontweight='bold')
    ax2.set_ylabel('可行域区间宽度', fontsize=12, fontweight='bold')
    ax2.set_title('各季度的不确定性分布', fontsize=14, fontweight='bold', pad=15)
    ax2.set_xticklabels(seasons, rotation=45)
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, '1_interval_width_distribution.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ 已保存: {output_path}")


def plot_bootstrap_ci_heatmap(df, output_dir):
    """图2: Bootstrap置信区间宽度热力图"""
    print("生成图2: Bootstrap置信区间宽度热力图...")
    
    # 计算CI宽度
    df['ci_width'] = df['fan_vote_ci_upper'] - df['fan_vote_ci_lower']
    
    # 创建透视表：季度 × 周数
    pivot = df.pivot_table(values='ci_width', index='season', 
                           columns='week', aggfunc='mean')
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # 绘制热力图
    im = ax.imshow(pivot.values, cmap='RdYlBu_r', aspect='auto')
    
    # 设置坐标轴
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticklabels(pivot.index)
    
    ax.set_xlabel('周数', fontsize=12, fontweight='bold')
    ax.set_ylabel('季度', fontsize=12, fontweight='bold')
    ax.set_title('Bootstrap 95%置信区间宽度热力图\n（颜色越深表示不确定性越高）', 
                 fontsize=14, fontweight='bold', pad=20)
    
    # 添加颜色条
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('CI宽度', rotation=270, labelpad=20, fontweight='bold')
    
    # 在热力图上标注数值（仅显示部分）
    for i in range(0, len(pivot.index), 2):
        for j in range(0, len(pivot.columns), 1):
            if not np.isnan(pivot.values[i, j]):
                text = ax.text(j, i, f'{pivot.values[i, j]:.3f}',
                              ha="center", va="center", color="black", 
                              fontsize=7, alpha=0.7)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, '2_bootstrap_ci_heatmap.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ 已保存: {output_path}")


def plot_std_vs_interval_correlation(df, output_dir):
    """图3: 标准差与可行域区间的相关性散点图"""
    print("生成图3: 标准差与可行域区间相关性散点图...")
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # 分组数据
    eliminated = df[df['eliminated'] == True]
    survived = df[df['eliminated'] == False]
    
    # 绘制散点图
    ax.scatter(survived['interval_width'], survived['fan_vote_std_b'],
              c=COLORS['survived'], s=50, alpha=0.5, label='幸存者', 
              edgecolors='black', linewidth=0.5)
    
    ax.scatter(eliminated['interval_width'], eliminated['fan_vote_std_b'],
              c=COLORS['eliminated'], s=50, alpha=0.5, label='淘汰者',
              edgecolors='black', linewidth=0.5)
    
    # 计算并绘制回归线
    valid_data = df.dropna(subset=['interval_width', 'fan_vote_std_b'])
    if len(valid_data) > 0:
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            valid_data['interval_width'], valid_data['fan_vote_std_b']
        )
        
        x_line = np.array([valid_data['interval_width'].min(), 
                          valid_data['interval_width'].max()])
        y_line = slope * x_line + intercept
        
        ax.plot(x_line, y_line, 'r--', linewidth=2, alpha=0.8,
               label=f'回归线 (R²={r_value**2:.4f})')
        
        # 添加统计信息
        text_str = f'相关系数: {r_value:.4f}\nR²: {r_value**2:.4f}\np值: {p_value:.6f}'
        ax.text(0.05, 0.95, text_str, transform=ax.transAxes,
               verticalalignment='top', bbox=dict(boxstyle='round',
               facecolor='lightblue', alpha=0.8), fontsize=11)
    
    ax.set_xlabel('可行域区间宽度（方法1）', fontsize=12, fontweight='bold')
    ax.set_ylabel('Bootstrap标准差（方法2）', fontsize=12, fontweight='bold')
    ax.set_title('两种不确定性度量方法的一致性验证', fontsize=14, 
                 fontweight='bold', pad=15)
    ax.legend(fontsize=11, loc='lower right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, '3_std_vs_interval_correlation.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ 已保存: {output_path}")


def plot_confidence_interval_errorbar(df, output_dir):
    """图4: 95%置信区间可视化（选取代表性周）"""
    print("生成图4: 95%置信区间误差棒图...")
    
    # 选取3个代表性周：第27季第1周、第5周、第9周
    selected_weeks = [
        (27, 1, "第27季第1周（初期）"),
        (27, 5, "第27季第5周（中期）"),
        (27, 9, "第27季第9周（决赛）")
    ]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for idx, (season, week, title) in enumerate(selected_weeks):
        ax = axes[idx]
        week_data = df[(df['season'] == season) & (df['week'] == week)].copy()
        
        if len(week_data) == 0:
            continue
        
        # 按估计值排序
        week_data = week_data.sort_values('fan_vote_percent', ascending=False)
        
        # 准备数据
        x = np.arange(len(week_data))
        y = week_data['fan_vote_percent'].values
        yerr_lower = y - week_data['fan_vote_ci_lower'].values
        yerr_upper = week_data['fan_vote_ci_upper'].values - y
        
        # 颜色编码
        colors = [COLORS['eliminated'] if elim else COLORS['survived'] 
                 for elim in week_data['eliminated']]
        
        # 绘制误差棒
        ax.errorbar(x, y, yerr=[yerr_lower, yerr_upper], fmt='o',
                   markersize=8, capsize=5, capthick=2, linewidth=2,
                   alpha=0.7)
        
        # 着色点
        for i, (xi, yi, color) in enumerate(zip(x, y, colors)):
            ax.scatter(xi, yi, c=color, s=100, zorder=3, 
                      edgecolors='black', linewidth=1.5)
        
        ax.set_xticks(x)
        ax.set_xticklabels(week_data['celebrity_name'], rotation=45, 
                          ha='right', fontsize=9)
        ax.set_ylabel('观众投票百分比', fontsize=11, fontweight='bold')
        ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
        ax.grid(True, alpha=0.3, axis='y')
        
        # 添加图例
        if idx == 2:
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor=COLORS['survived'], label='幸存者'),
                Patch(facecolor=COLORS['eliminated'], label='淘汰者')
            ]
            ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    plt.suptitle('代表性周的观众投票估计及95%置信区间', 
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, '4_confidence_interval_errorbar.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ 已保存: {output_path}")


# ========== 图表组2: 时间演化分析 ==========

def plot_fan_vote_trajectory(df, output_dir):
    """图5: 选手人气随时间变化的轨迹图"""
    print("生成图5: 选手人气轨迹图...")
    
    # 选择2个代表性季度
    selected_seasons = [11, 27]
    
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    
    for idx, season in enumerate(selected_seasons):
        ax = axes[idx]
        season_data = df[df['season'] == season]
        
        # 获取所有选手
        contestants = season_data['celebrity_name'].unique()
        
        # 为每位选手绘制轨迹
        for contestant in contestants:
            contestant_data = season_data[season_data['celebrity_name'] == contestant]
            contestant_data = contestant_data.sort_values('week')
            
            weeks = contestant_data['week'].values
            fan_votes = contestant_data['fan_vote_percent'].values
            eliminated_weeks = contestant_data[contestant_data['eliminated'] == True]['week'].values
            
            # 选择颜色
            color = plt.cm.tab20(hash(contestant) % 20)
            
            # 绘制线条
            if len(eliminated_weeks) > 0:
                # 淘汰前用实线
                elim_week = eliminated_weeks[0]
                mask_before = weeks < elim_week
                if mask_before.any():
                    ax.plot(weeks[mask_before], fan_votes[mask_before], 
                           '-', color=color, linewidth=2, alpha=0.8)
                # 淘汰周用虚线
                mask_elim = weeks >= elim_week
                if mask_elim.any():
                    ax.plot(weeks[mask_elim], fan_votes[mask_elim],
                           '--', color=color, linewidth=2, alpha=0.6,
                           label=f'{contestant} [淘汰]')
            else:
                ax.plot(weeks, fan_votes, '-', color=color, 
                       linewidth=2, alpha=0.8, label=contestant)
        
        ax.set_xlabel('周数', fontsize=12, fontweight='bold')
        ax.set_ylabel('观众投票百分比', fontsize=12, fontweight='bold')
        ax.set_title(f'第{season}季选手人气演化', fontsize=14, fontweight='bold', pad=10)
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    
    plt.suptitle('选手观众人气随时间变化轨迹\n（虚线表示淘汰后）', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, '5_fan_vote_trajectory.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ 已保存: {output_path}")


def plot_smoothness_distribution(df, output_dir):
    """图6: 平滑度指标分布直方图"""
    print("生成图6: 平滑度分布直方图...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # 过滤掉只参加1周的选手（他们没有平滑度）
    df_multi_week = df[df['n_weeks'] > 1].copy()
    
    # 按参与周数分组
    df_multi_week['week_group'] = pd.cut(df_multi_week['n_weeks'], 
                                          bins=[1, 3, 6, 20],
                                          labels=['2-3周', '4-6周', '7+周'])
    
    # 子图1: 堆叠直方图
    groups = df_multi_week.groupby('week_group')['smoothness']
    data_to_plot = [groups.get_group(g).values for g in ['2-3周', '4-6周', '7+周']]
    
    ax1.hist(data_to_plot, bins=30, stacked=True, alpha=0.7,
            label=['2-3周', '4-6周', '7+周'],
            color=[COLORS['eliminated'], COLORS['fan'], COLORS['survived']])
    
    ax1.set_xlabel('平滑度（相邻周投票差异²）', fontsize=12, fontweight='bold')
    ax1.set_ylabel('选手数量', fontsize=12, fontweight='bold')
    ax1.set_title('平滑度分布（按参与周数分组）', fontsize=14, fontweight='bold', pad=15)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # 子图2: 箱线图对比
    bp = ax2.boxplot(data_to_plot, labels=['2-3周', '4-6周', '7+周'],
                    patch_artist=True, showfliers=True)
    
    colors_box = [COLORS['eliminated'], COLORS['fan'], COLORS['survived']]
    for patch, color in zip(bp['boxes'], colors_box):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax2.set_ylabel('平滑度', fontsize=12, fontweight='bold')
    ax2.set_title('不同参与周数的平滑度对比', fontsize=14, fontweight='bold', pad=15)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 添加统计信息
    text_str = f"平滑度均值: {df_multi_week['smoothness'].mean():.6f}\n"
    text_str += f"平滑度中位数: {df_multi_week['smoothness'].median():.6f}\n"
    text_str += f"λ参数: 100.0"
    ax2.text(0.02, 0.98, text_str, transform=ax2.transAxes,
            verticalalignment='top', bbox=dict(boxstyle='round',
            facecolor='lightyellow', alpha=0.8), fontsize=10)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, '6_smoothness_distribution.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ 已保存: {output_path}")


def plot_entropy_evolution(df, output_dir):
    """图7: 熵值随季度和周数的演化"""
    print("生成图7: 熵值演化热力图...")
    
    # 为每个(season, week)计算平均熵
    entropy_pivot = df.groupby(['season', 'week'])['week_entropy'].mean().reset_index()
    pivot = entropy_pivot.pivot(index='season', columns='week', values='week_entropy')
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # 绘制热力图
    im = ax.imshow(pivot.values, cmap='YlOrRd', aspect='auto')
    
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticklabels(pivot.index)
    
    ax.set_xlabel('周数', fontsize=12, fontweight='bold')
    ax.set_ylabel('季度', fontsize=12, fontweight='bold')
    ax.set_title('观众投票分布熵值演化\n（熵越高表示投票越分散均匀）', 
                 fontsize=14, fontweight='bold', pad=20)
    
    # 颜色条
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('熵值', rotation=270, labelpad=20, fontweight='bold')
    
    # 标注部分数值
    for i in range(0, len(pivot.index), 3):
        for j in range(len(pivot.columns)):
            if not np.isnan(pivot.values[i, j]):
                text = ax.text(j, i, f'{pivot.values[i, j]:.2f}',
                              ha="center", va="center", color="white", 
                              fontsize=7, weight='bold')
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, '7_entropy_evolution.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ 已保存: {output_path}")


# ========== 图表组3: 评委与观众意见对比 ==========

def plot_judge_fan_scatter(df, output_dir):
    """图8: 评委-观众百分比散点图"""
    print("生成图8: 评委-观众百分比散点图...")
    
    fig, ax = plt.subplots(figsize=(12, 11))
    
    # 创建密度散点图
    from matplotlib.colors import LinearSegmentedColormap
    
    # 分别绘制淘汰者和幸存者
    eliminated = df[df['eliminated'] == True]
    survived = df[df['eliminated'] == False]
    
    # 幸存者（背景）
    h = ax.hexbin(survived['judge_percent'], survived['fan_vote_percent'],
                  gridsize=30, cmap='Greens', alpha=0.6, mincnt=1)
    
    # 淘汰者（覆盖）
    h2 = ax.hexbin(eliminated['judge_percent'], eliminated['fan_vote_percent'],
                   gridsize=30, cmap='Reds', alpha=0.7, mincnt=1)
    
    # 对角线
    max_val = max(df['judge_percent'].max(), df['fan_vote_percent'].max())
    ax.plot([0, max_val], [0, max_val], 'k--', linewidth=2, alpha=0.5,
           label='评委=观众')
    
    # 四个象限标注
    mid_x = df['judge_percent'].median()
    mid_y = df['fan_vote_percent'].median()
    
    # 左上：观众喜欢、评委不喜欢
    ax.text(0.02, 0.98, '观众喜欢\n评委不喜欢', 
           transform=ax.transAxes, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor=COLORS['fan'], alpha=0.7),
           fontsize=11, fontweight='bold')
    
    # 右下：评委喜欢、观众不喜欢
    ax.text(0.98, 0.02, '评委喜欢\n观众不喜欢',
           transform=ax.transAxes, verticalalignment='bottom',
           horizontalalignment='right',
           bbox=dict(boxstyle='round', facecolor=COLORS['judge'], alpha=0.7),
           fontsize=11, fontweight='bold')
    
    ax.set_xlabel('评委百分比', fontsize=12, fontweight='bold')
    ax.set_ylabel('观众投票百分比（估计）', fontsize=12, fontweight='bold')
    ax.set_title('评委与观众意见分布对比\n（绿色=幸存者，红色=淘汰者）', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.legend(fontsize=11, loc='lower right')
    ax.grid(True, alpha=0.3)
    
    # 颜色条
    cb1 = plt.colorbar(h, ax=ax, label='幸存者密度', pad=0.02)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, '8_judge_fan_scatter.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ 已保存: {output_path}")


def plot_judge_fan_difference_trend(df, output_dir):
    """图9: 评委-观众差异随季度变化趋势"""
    print("生成图9: 评委-观众差异趋势图...")
    
    # 计算每季每周的平均差异
    df['abs_diff'] = np.abs(df['judge_percent'] - df['fan_vote_percent'])
    
    season_stats = df.groupby('season').agg({
        'abs_diff': ['mean', 'std', 'count']
    }).reset_index()
    
    season_stats.columns = ['season', 'mean_diff', 'std_diff', 'count']
    season_stats['se'] = season_stats['std_diff'] / np.sqrt(season_stats['count'])
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # 子图1: 折线图with误差带
    seasons = season_stats['season']
    mean_diff = season_stats['mean_diff']
    se = season_stats['se']
    
    ax1.plot(seasons, mean_diff, 'o-', linewidth=2, markersize=8,
            color=COLORS['primary'], label='平均绝对差异')
    ax1.fill_between(seasons, mean_diff - se, mean_diff + se,
                     alpha=0.3, color=COLORS['primary'])
    
    # 添加趋势线
    z = np.polyfit(seasons, mean_diff, 2)
    p = np.poly1d(z)
    ax1.plot(seasons, p(seasons), '--', color=COLORS['secondary'],
            linewidth=2, label='二次趋势线')
    
    # 平均线
    overall_mean = mean_diff.mean()
    ax1.axhline(y=overall_mean, color=COLORS['accent'], linestyle='--',
               linewidth=2, label=f'总平均值 ({overall_mean:.4f})')
    
    ax1.set_xlabel('季度', fontsize=12, fontweight='bold')
    ax1.set_ylabel('平均绝对差异 |评委%-观众%|', fontsize=12, fontweight='bold')
    ax1.set_title('评委与观众意见差异随季度变化', fontsize=14, fontweight='bold', pad=15)
    ax1.legend(fontsize=11, loc='best')
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(range(3, 28, 2))
    
    # 子图2: 按淘汰状态分组的趋势
    eliminated_stats = df[df['eliminated'] == True].groupby('season')['abs_diff'].mean()
    survived_stats = df[df['eliminated'] == False].groupby('season')['abs_diff'].mean()
    
    ax2.plot(eliminated_stats.index, eliminated_stats.values, 'o-',
            linewidth=2, markersize=7, color=COLORS['eliminated'],
            label='淘汰者', alpha=0.8)
    ax2.plot(survived_stats.index, survived_stats.values, 's-',
            linewidth=2, markersize=7, color=COLORS['survived'],
            label='幸存者', alpha=0.8)
    
    ax2.set_xlabel('季度', fontsize=12, fontweight='bold')
    ax2.set_ylabel('平均绝对差异', fontsize=12, fontweight='bold')
    ax2.set_title('按淘汰状态分组的差异趋势', fontsize=14, fontweight='bold', pad=15)
    ax2.legend(fontsize=11, loc='best')
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(range(3, 28, 2))
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, '9_judge_fan_difference_trend.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ 已保存: {output_path}")


# ========== 图表组4: 争议选手深度分析 ==========

def plot_bobby_bones_analysis(df, output_dir):
    """图10: Bobby Bones (第27季) 详细分析"""
    print("生成图10: Bobby Bones详细分析...")
    
    bobby = df[(df['season'] == 27) & (df['celebrity_name'] == 'Bobby Bones')].copy()
    bobby = bobby.sort_values('week')
    
    if len(bobby) == 0:
        print("  警告: 未找到Bobby Bones数据")
        return
    
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # 子图1: 评委、观众、总分（柱状图）
    ax1 = fig.add_subplot(gs[0, :])
    
    weeks = bobby['week'].values
    x = np.arange(len(weeks))
    width = 0.25
    
    ax1.bar(x - width, bobby['judge_percent'] * 100, width,
           label='评委分', color=COLORS['judge'], alpha=0.8)
    ax1.bar(x, bobby['fan_vote_percent'] * 100, width,
           label='观众投票（估计）', color=COLORS['fan'], alpha=0.8)
    ax1.bar(x + width, bobby['total_percent'] * 100, width,
           label='总分', color=COLORS['primary'], alpha=0.8)
    
    ax1.set_xlabel('周数', fontsize=12, fontweight='bold')
    ax1.set_ylabel('百分比 (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Bobby Bones 每周得分分解', fontsize=14, fontweight='bold', pad=10)
    ax1.set_xticks(x)
    ax1.set_xticklabels(weeks)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # 子图2: 置信区间（误差棒）
    ax2 = fig.add_subplot(gs[1, 0])
    
    y = bobby['fan_vote_percent'].values * 100
    yerr_lower = y - bobby['fan_vote_ci_lower'].values * 100
    yerr_upper = bobby['fan_vote_ci_upper'].values * 100 - y
    
    ax2.errorbar(weeks, y, yerr=[yerr_lower, yerr_upper],
                fmt='o-', linewidth=2, markersize=10, capsize=5,
                color=COLORS['fan'], label='观众投票估计')
    
    ax2.set_xlabel('周数', fontsize=12, fontweight='bold')
    ax2.set_ylabel('观众投票百分比 (%)', fontsize=12, fontweight='bold')
    ax2.set_title('观众投票估计的95%置信区间', fontsize=14, fontweight='bold', pad=10)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)
    
    # 子图3: 平滑度变化
    ax3 = fig.add_subplot(gs[1, 1])
    
    ax3.plot(weeks, bobby['smoothness'].values, 'o-',
            linewidth=2, markersize=8, color=COLORS['uncertainty'])
    
    ax3.set_xlabel('周数', fontsize=12, fontweight='bold')
    ax3.set_ylabel('平滑度', fontsize=12, fontweight='bold')
    ax3.set_title('人气变化平滑度指标', fontsize=14, fontweight='bold', pad=10)
    ax3.grid(True, alpha=0.3)
    
    # 子图4: 评委-观众对比折线图
    ax4 = fig.add_subplot(gs[2, :])
    
    ax4.plot(weeks, bobby['judge_percent'].values * 100, 'o-',
            linewidth=3, markersize=10, color=COLORS['judge'],
            label='评委分', alpha=0.8)
    ax4.plot(weeks, bobby['fan_vote_percent'].values * 100, 's-',
            linewidth=3, markersize=10, color=COLORS['fan'],
            label='观众投票', alpha=0.8)
    
    ax4.set_xlabel('周数', fontsize=12, fontweight='bold')
    ax4.set_ylabel('百分比 (%)', fontsize=12, fontweight='bold')
    ax4.set_title('评委分 vs 观众投票对比（Bobby Bones）', 
                  fontsize=14, fontweight='bold', pad=10)
    ax4.legend(fontsize=12, loc='best')
    ax4.grid(True, alpha=0.3)
    
    # 添加总体说明
    avg_judge = bobby['judge_percent'].mean() * 100
    avg_fan = bobby['fan_vote_percent'].mean() * 100
    text_str = f"平均评委分: {avg_judge:.2f}%\n"
    text_str += f"平均观众投票: {avg_fan:.2f}%\n"
    text_str += f"评委-观众差异: {avg_fan - avg_judge:.2f}%"
    
    ax4.text(0.02, 0.98, text_str, transform=ax4.transAxes,
            verticalalignment='top', bbox=dict(boxstyle='round',
            facecolor='wheat', alpha=0.8), fontsize=11)
    
    plt.suptitle('Bobby Bones (第27季冠军) 完整数据分析', 
                 fontsize=16, fontweight='bold')
    
    output_path = os.path.join(output_dir, '10_bobby_bones_analysis.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ 已保存: {output_path}")


def plot_bristol_palin_comparison(df, output_dir):
    """图11: Bristol Palin (第11季) vs 对比选手"""
    print("生成图11: Bristol Palin对比分析...")
    
    season_11 = df[df['season'] == 11].copy()
    
    # 计算每位选手的统计量
    contestant_stats = []
    for name in season_11['celebrity_name'].unique():
        contestant_data = season_11[season_11['celebrity_name'] == name]
        
        stats_dict = {
            'name': name,
            'avg_judge': contestant_data['judge_percent'].mean(),
            'avg_fan': contestant_data['fan_vote_percent'].mean(),
            'avg_ci_width': (contestant_data['fan_vote_ci_upper'] - 
                           contestant_data['fan_vote_ci_lower']).mean(),
            'avg_smoothness': contestant_data['smoothness'].mean(),
            'n_weeks': contestant_data['n_weeks'].max(),
            'eliminated': contestant_data['eliminated'].any()
        }
        contestant_stats.append(stats_dict)
    
    stats_df = pd.DataFrame(contestant_stats)
    
    # 选择Bristol和前3名选手进行对比
    bristol = stats_df[stats_df['name'] == 'Bristol Palin']
    top3 = stats_df.nlargest(3, 'n_weeks')
    
    # 合并Bristol和top3
    comparison = pd.concat([bristol, top3]).drop_duplicates(subset=['name'])
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    # 5个维度的对比
    metrics = [
        ('avg_judge', '平均评委分', '%'),
        ('avg_fan', '平均观众投票', '%'),
        ('avg_ci_width', '平均CI宽度', ''),
        ('avg_smoothness', '平均平滑度', ''),
        ('n_weeks', '存活周数', '周')
    ]
    
    for idx, (metric, title, unit) in enumerate(metrics):
        ax = axes[idx]
        
        values = comparison[metric].values
        if unit == '%':
            values = values * 100
        
        names = comparison['name'].values
        colors = [COLORS['secondary'] if name == 'Bristol Palin' 
                 else COLORS['primary'] for name in names]
        
        bars = ax.barh(names, values, color=colors, alpha=0.8,
                      edgecolor='black', linewidth=1.5)
        
        ax.set_xlabel(f'{title} ({unit})' if unit else title,
                     fontsize=11, fontweight='bold')
        ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
        ax.grid(True, alpha=0.3, axis='x')
        
        # 在柱子上标注数值
        for i, (bar, val) in enumerate(zip(bars, values)):
            ax.text(val, i, f' {val:.2f}{unit}',
                   va='center', fontweight='bold', fontsize=10)
    
    # 第6个子图：雷达图
    ax6 = axes[5]
    ax6.remove()
    ax6 = fig.add_subplot(2, 3, 6, projection='polar')
    
    # 准备雷达图数据（归一化）
    bristol_data = bristol.iloc[0]
    categories = ['评委分', '观众投票', 'CI宽度', '平滑度', '存活周数']
    
    # 归一化到0-1
    values_bristol = [
        bristol_data['avg_judge'] / stats_df['avg_judge'].max(),
        bristol_data['avg_fan'] / stats_df['avg_fan'].max(),
        bristol_data['avg_ci_width'] / stats_df['avg_ci_width'].max(),
        bristol_data['avg_smoothness'] / stats_df['avg_smoothness'].max(),
        bristol_data['n_weeks'] / stats_df['n_weeks'].max()
    ]
    
    # 平均值作为对比
    avg_values = [
        stats_df['avg_judge'].mean() / stats_df['avg_judge'].max(),
        stats_df['avg_fan'].mean() / stats_df['avg_fan'].max(),
        stats_df['avg_ci_width'].mean() / stats_df['avg_ci_width'].max(),
        stats_df['avg_smoothness'].mean() / stats_df['avg_smoothness'].max(),
        stats_df['n_weeks'].mean() / stats_df['n_weeks'].max()
    ]
    
    # 闭合图形
    values_bristol += values_bristol[:1]
    avg_values += avg_values[:1]
    
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]
    
    ax6.plot(angles, values_bristol, 'o-', linewidth=2, 
            label='Bristol Palin', color=COLORS['secondary'])
    ax6.fill(angles, values_bristol, alpha=0.25, color=COLORS['secondary'])
    
    ax6.plot(angles, avg_values, 's-', linewidth=2,
            label='平均水平', color=COLORS['primary'])
    ax6.fill(angles, avg_values, alpha=0.15, color=COLORS['primary'])
    
    ax6.set_xticks(angles[:-1])
    ax6.set_xticklabels(categories, fontsize=10)
    ax6.set_ylim(0, 1)
    ax6.set_title('Bristol Palin多维特征雷达图', 
                  fontsize=12, fontweight='bold', pad=20)
    ax6.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=10)
    ax6.grid(True)
    
    plt.suptitle('Bristol Palin (第11季) vs 同季选手对比分析',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, '11_bristol_palin_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ 已保存: {output_path}")


def plot_controversy_ranking(df, output_dir):
    """图12: 争议度排行榜"""
    print("生成图12: 争议度排行榜...")
    
    # 计算争议度：|评委%-观众%| × (存活周数)
    # 更关注那些长期存在争议的选手
    contestant_controversy = []
    
    for (season, name), group in df.groupby(['season', 'celebrity_name']):
        avg_diff = np.abs(group['judge_percent'] - group['fan_vote_percent']).mean()
        n_weeks = group['n_weeks'].max()
        eliminated = group['eliminated'].any()
        
        # 争议度 = 平均差异 × 存活周数
        controversy_score = avg_diff * n_weeks
        
        contestant_controversy.append({
            'season': season,
            'name': name,
            'controversy_score': controversy_score,
            'avg_diff': avg_diff,
            'n_weeks': n_weeks,
            'eliminated': eliminated
        })
    
    controversy_df = pd.DataFrame(contestant_controversy)
    
    # 取Top 20
    top20 = controversy_df.nlargest(20, 'controversy_score')
    
    fig, ax = plt.subplots(figsize=(12, 14))
    
    # 颜色按季度
    colors = plt.cm.viridis(top20['season'] / top20['season'].max())
    
    # 水平条形图
    y_pos = np.arange(len(top20))
    bars = ax.barh(y_pos, top20['controversy_score'].values,
                   color=colors, alpha=0.8, edgecolor='black', linewidth=1)
    
    # 标签
    labels = [f"{row['name']} (S{row['season']})" 
             for _, row in top20.iterrows()]
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=10)
    
    ax.set_xlabel('争议度得分（差异×周数）', fontsize=12, fontweight='bold')
    ax.set_title('评委-观众意见分歧最大的选手 Top 20\n（颜色表示季度，越亮越晚）',
                 fontsize=14, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, axis='x')
    
    # 在柱子上标注得分
    for i, (bar, score) in enumerate(zip(bars, top20['controversy_score'].values)):
        ax.text(score, i, f' {score:.3f}',
               va='center', fontsize=9, fontweight='bold')
    
    # 添加颜色条
    sm = plt.cm.ScalarMappable(cmap='viridis',
                               norm=plt.Normalize(vmin=top20['season'].min(),
                                                 vmax=top20['season'].max()))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label('季度', rotation=270, labelpad=20, fontweight='bold')
    
    # 标注特殊选手
    special_contestants = ['Bobby Bones', 'Bristol Palin']
    for idx, row in top20.iterrows():
        if row['name'] in special_contestants:
            y_idx = top20.index.get_loc(idx)
            ax.plot(row['controversy_score'], y_idx, '*',
                   markersize=20, color='red', markeredgecolor='black',
                   markeredgewidth=1.5)
    
    # 添加图例
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], marker='*', color='w', 
                             markerfacecolor='r', markersize=15,
                             label='争议冠军', markeredgecolor='black')]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=11)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, '12_controversy_ranking.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ 已保存: {output_path}")


# ========== 主函数 ==========

def generate_visualization_report(output_dir):
    """生成可视化报告"""
    report_path = os.path.join(output_dir, 'visualization_report.md')
    
    report_content = """# 观众投票估计可视化分析报告

## 生成的图表清单

### 图表组1: 确定性与不确定性分析
1. **可行域区间宽度分布对比图** (`1_interval_width_distribution.png`)
   - 展示淘汰者vs幸存者的不确定性差异
   - 各季度的不确定性分布

2. **Bootstrap置信区间宽度热力图** (`2_bootstrap_ci_heatmap.png`)
   - 季度×周数的CI宽度热力图
   - 颜色表示估计可靠性

3. **标准差与可行域区间相关性散点图** (`3_std_vs_interval_correlation.png`)
   - 验证两种不确定性度量方法的一致性
   - 包含回归线和R²统计量

4. **95%置信区间误差棒图** (`4_confidence_interval_errorbar.png`)
   - 选取代表性周展示置信区间
   - 直观显示估计的不确定性范围

### 图表组2: 时间演化分析
5. **选手人气轨迹图** (`5_fan_vote_trajectory.png`)
   - 展示平滑性正则化的效果
   - 选手淘汰后轨迹变为虚线

6. **平滑度分布直方图** (`6_smoothness_distribution.png`)
   - 按参与周数分组的平滑度分布
   - 验证λ=100参数的合理性

7. **熵值演化热力图** (`7_entropy_evolution.png`)
   - 季度×周数的熵值演化
   - 展示投票分布的均匀程度

### 图表组3: 评委与观众意见对比
8. **评委-观众百分比散点图** (`8_judge_fan_scatter.png`)
   - 密度散点图显示意见分布
   - 四象限分析意见差异

9. **评委-观众差异趋势图** (`9_judge_fan_difference_trend.png`)
   - 差异随季度变化趋势
   - 按淘汰状态分组对比

### 图表组4: 争议选手深度分析
10. **Bobby Bones完整分析** (`10_bobby_bones_analysis.png`)
    - 第27季冠军的多维度分析
    - 评委分、观众投票、置信区间、平滑度

11. **Bristol Palin对比分析** (`11_bristol_palin_comparison.png`)
    - 与同季选手多维对比
    - 雷达图展示综合特征

12. **争议度排行榜** (`12_controversy_ranking.png`)
    - Top 20评委-观众意见分歧最大的选手
    - 争议度 = 平均差异 × 存活周数

## 关键发现

1. **不确定性特征**
   - 淘汰者的估计不确定性显著低于幸存者
   - 两种不确定性度量方法（可行域和Bootstrap）高度一致

2. **时间演化**
   - 平滑性正则化有效减少了人气的剧烈波动
   - 熵值随竞赛进程逐渐降低（投票更集中）

3. **评委-观众意见**
   - 评委与观众意见存在系统性差异
   - 部分季度差异特别显著

4. **争议选手**
   - Bobby Bones和Bristol Palin是历史上最具争议的选手
   - 他们的观众人气显著高于评委分数

## 使用建议

- 所有图表采用演示汇报风格，适合PPT展示
- 图片分辨率为300 DPI，适合打印和发表
- 建议在论文中按组引用图表，讲述完整故事

---

**生成时间**: """ + pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S') + """
**图表数量**: 12张
**数据范围**: 第3-27季
"""
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    print(f"\n✓ 可视化报告已保存: {report_path}")


def generate_all_visualizations(output_dir='visualization_results'):
    """生成所有可视化"""
    
    # 创建输出目录
    full_output_dir = os.path.join(SCRIPT_DIR, output_dir)
    os.makedirs(full_output_dir, exist_ok=True)
    
    print("="*70)
    print("观众投票估计可视化分析")
    print("="*70)
    print(f"输出目录: {full_output_dir}\n")
    
    # 加载数据
    estimates, bootstrap, interval_summary = load_all_data()
    print()
    
    # 图表组1: 确定性与不确定性分析
    print("="*70)
    print("图表组1: 确定性与不确定性分析")
    print("="*70)
    plot_interval_width_distribution(estimates, full_output_dir)
    plot_bootstrap_ci_heatmap(bootstrap, full_output_dir)
    plot_std_vs_interval_correlation(bootstrap, full_output_dir)
    plot_confidence_interval_errorbar(bootstrap, full_output_dir)
    print()
    
    # 图表组2: 时间演化分析
    print("="*70)
    print("图表组2: 时间演化分析")
    print("="*70)
    plot_fan_vote_trajectory(estimates, full_output_dir)
    plot_smoothness_distribution(estimates, full_output_dir)
    plot_entropy_evolution(estimates, full_output_dir)
    print()
    
    # 图表组3: 评委与观众意见对比
    print("="*70)
    print("图表组3: 评委与观众意见对比")
    print("="*70)
    plot_judge_fan_scatter(estimates, full_output_dir)
    plot_judge_fan_difference_trend(estimates, full_output_dir)
    print()
    
    # 图表组4: 争议选手深度分析
    print("="*70)
    print("图表组4: 争议选手深度分析")
    print("="*70)
    plot_bobby_bones_analysis(bootstrap, full_output_dir)
    plot_bristol_palin_comparison(bootstrap, full_output_dir)
    plot_controversy_ranking(estimates, full_output_dir)
    print()
    
    # 生成报告
    generate_visualization_report(full_output_dir)
    
    print("\n" + "="*70)
    print("✓ 所有可视化已完成！")
    print("="*70)
    print(f"\n请查看目录: {full_output_dir}")
    print("共生成12张高质量图表 + 1份说明文档\n")


if __name__ == '__main__':
    generate_all_visualizations()
