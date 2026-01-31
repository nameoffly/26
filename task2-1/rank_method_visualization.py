"""
排名法淘汰分析 - 可视化与统计检验

功能：
1. 一致比例随季数变化的趋势图
2. 差异周的选手百分比分布散点图
3. 评委排名 vs 观众排名的二维热力图
4. 卡方检验判断两种方法差异的显著性
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency
import os
import sys

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DEFAULT_CSV = os.path.join(SCRIPT_DIR, 'rank_vs_percent_elimination_2.csv')
DEFAULT_EXCEL = os.path.join(PROJECT_ROOT, '2026_MCM_Problem_C_Processed_Data.xlsx')
DEFAULT_FAN_CSV = os.path.join(SCRIPT_DIR, 'fan_vote_estimates_entropy_smooth_100.csv')


def load_data():
    """加载数据"""
    df = pd.read_csv(DEFAULT_CSV)
    excel_df = pd.read_excel(DEFAULT_EXCEL)
    fan_df = pd.read_csv(DEFAULT_FAN_CSV)
    return df, excel_df, fan_df


def plot_consistency_by_season(df: pd.DataFrame, output_dir: str):
    """
    图1：一致比例随季数变化的趋势图
    """
    # 只考虑有人淘汰的周数
    df_with_elim = df[df['n_eliminated'] > 0].copy()
    
    # 按季度统计
    season_stats = []
    for season in sorted(df_with_elim['season'].unique()):
        season_df = df_with_elim[df_with_elim['season'] == season]
        total_weeks = len(season_df)
        same_weeks = season_df['same_result'].sum()
        consistency_rate = same_weeks / total_weeks * 100 if total_weeks > 0 else 0
        
        season_stats.append({
            'season': season,
            'total_weeks': total_weeks,
            'same_weeks': same_weeks,
            'consistency_rate': consistency_rate
        })
    
    stats_df = pd.DataFrame(season_stats)
    
    # 绘制趋势图
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # 子图1：一致比例趋势
    ax1.plot(stats_df['season'], stats_df['consistency_rate'], 
             marker='o', linewidth=2, markersize=8, color='#2E86AB', label='一致比例')
    
    # 添加趋势线
    z = np.polyfit(stats_df['season'], stats_df['consistency_rate'], 2)
    p = np.poly1d(z)
    ax1.plot(stats_df['season'], p(stats_df['season']), 
             '--', color='#A23B72', alpha=0.7, linewidth=2, label='二次趋势线')
    
    # 添加平均线
    mean_rate = stats_df['consistency_rate'].mean()
    ax1.axhline(y=mean_rate, color='#F18F01', linestyle='--', 
                linewidth=2, alpha=0.7, label=f'平均值 ({mean_rate:.1f}%)')
    
    ax1.set_xlabel('季数', fontsize=12, fontweight='bold')
    ax1.set_ylabel('一致比例 (%)', fontsize=12, fontweight='bold')
    ax1.set_title('排名法与百分比法淘汰结果一致性随季数变化', fontsize=14, fontweight='bold', pad=20)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.legend(fontsize=10, loc='best')
    ax1.set_xticks(range(3, 28, 2))
    
    # 子图2：每季的周数统计（堆叠柱状图）
    ax2.bar(stats_df['season'], stats_df['same_weeks'], 
            label='一致周数', color='#06A77D', alpha=0.8)
    ax2.bar(stats_df['season'], stats_df['total_weeks'] - stats_df['same_weeks'], 
            bottom=stats_df['same_weeks'], label='差异周数', color='#D62246', alpha=0.8)
    
    ax2.set_xlabel('季数', fontsize=12, fontweight='bold')
    ax2.set_ylabel('周数', fontsize=12, fontweight='bold')
    ax2.set_title('每季一致与差异周数分布', fontsize=14, fontweight='bold', pad=20)
    ax2.legend(fontsize=10, loc='best')
    ax2.grid(True, alpha=0.3, linestyle='--', axis='y')
    ax2.set_xticks(range(3, 28, 2))
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'consistency_by_season.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ 图1已保存: {output_path}")
    plt.close()
    
    return stats_df


def plot_percentage_scatter(df: pd.DataFrame, excel_df: pd.DataFrame, 
                           fan_df: pd.DataFrame, output_dir: str):
    """
    图2：差异周的选手百分比分布散点图
    分析在淘汰结果不同的周中，选手的评委百分比和观众百分比分布
    """
    # 只看结果不同且有人淘汰的周
    diff_weeks = df[(df['same_result'] == False) & (df['n_eliminated'] > 0)]
    
    # 收集差异周的所有选手数据
    scatter_data = []
    
    for _, row in diff_weeks.iterrows():
        season, week = row['season'], row['week']
        
        # 获取该周所有选手
        season_df = excel_df[excel_df['season'] == season]
        week_col = f'{week}_percent'
        
        if week_col not in season_df.columns:
            continue
            
        contestants = season_df[season_df[week_col] > 0]
        
        for _, contestant in contestants.iterrows():
            name = contestant['celebrity_name']
            judge_pct = contestant[week_col]
            
            # 获取观众百分比
            fan_row = fan_df[(fan_df['season'] == season) & 
                            (fan_df['week'] == week) & 
                            (fan_df['celebrity_name'] == name)]
            
            if fan_row.empty:
                continue
                
            fan_pct = fan_row.iloc[0]['fan_vote_percent']
            
            # 判断该选手是否被淘汰
            result_str = str(contestant.get('results', '')).lower()
            eliminated_percent = f'eliminated week {week}' in result_str
            
            # 判断排名法是否淘汰
            rank_eliminated_list = str(row['rank_eliminated']).split(',')
            eliminated_rank = name in rank_eliminated_list
            
            # 分类
            if eliminated_percent and eliminated_rank:
                category = '两种方法都淘汰'
            elif eliminated_percent and not eliminated_rank:
                category = '仅百分比法淘汰'
            elif not eliminated_percent and eliminated_rank:
                category = '仅排名法淘汰'
            else:
                category = '两种方法都保留'
            
            scatter_data.append({
                'judge_percent': judge_pct,
                'fan_percent': fan_pct,
                'category': category,
                'season': season,
                'week': week,
                'name': name
            })
    
    scatter_df = pd.DataFrame(scatter_data)
    
    # 绘制散点图
    fig, ax = plt.subplots(figsize=(14, 10))
    
    colors = {
        '两种方法都淘汰': '#D62246',
        '仅百分比法淘汰': '#F18F01',
        '仅排名法淘汰': '#2E86AB',
        '两种方法都保留': '#06A77D'
    }
    
    markers = {
        '两种方法都淘汰': 'X',
        '仅百分比法淘汰': '^',
        '仅排名法淘汰': 'v',
        '两种方法都保留': 'o'
    }
    
    for category in colors.keys():
        data = scatter_df[scatter_df['category'] == category]
        if len(data) > 0:
            ax.scatter(data['judge_percent'], data['fan_percent'],
                      c=colors[category], marker=markers[category],
                      s=100, alpha=0.6, label=f'{category} (n={len(data)})',
                      edgecolors='black', linewidth=0.5)
    
    # 添加对角线参考线
    max_val = max(scatter_df['judge_percent'].max(), scatter_df['fan_percent'].max())
    ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.3, linewidth=2, label='评委=观众')
    
    ax.set_xlabel('评委百分比 (%)', fontsize=12, fontweight='bold')
    ax.set_ylabel('观众百分比 (%)', fontsize=12, fontweight='bold')
    ax.set_title('差异周选手的评委-观众百分比分布', fontsize=14, fontweight='bold', pad=20)
    ax.legend(fontsize=10, loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'percentage_scatter.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ 图2已保存: {output_path}")
    plt.close()
    
    return scatter_df


def plot_rank_heatmap(df: pd.DataFrame, excel_df: pd.DataFrame, 
                     fan_df: pd.DataFrame, output_dir: str):
    """
    图3：评委排名 vs 观众排名的二维热力图
    展示在差异周中，不同排名组合的选手淘汰情况
    """
    # 只看结果不同且有人淘汰的周
    diff_weeks = df[(df['same_result'] == False) & (df['n_eliminated'] > 0)]
    
    # 收集排名数据
    rank_data = []
    
    for _, row in diff_weeks.iterrows():
        season, week = row['season'], row['week']
        
        # 获取该周所有选手
        season_df = excel_df[excel_df['season'] == season]
        week_col = f'{week}_percent'
        
        if week_col not in season_df.columns:
            continue
            
        contestants = season_df[season_df[week_col] > 0]
        
        # 获取评委和观众百分比
        names = contestants['celebrity_name'].tolist()
        judge_percents = contestants[week_col].values
        
        # 获取观众百分比
        fan_percents = []
        for name in names:
            fan_row = fan_df[(fan_df['season'] == season) & 
                            (fan_df['week'] == week) & 
                            (fan_df['celebrity_name'] == name)]
            if not fan_row.empty:
                fan_percents.append(fan_row.iloc[0]['fan_vote_percent'])
            else:
                fan_percents.append(0.0)
        
        fan_percents = np.array(fan_percents)
        
        # 计算排名
        judge_ranks = pd.Series(judge_percents).rank(ascending=False, method='min').astype(int).values
        fan_ranks = pd.Series(fan_percents).rank(ascending=False, method='min').astype(int).values
        
        # 判断淘汰情况
        for i, name in enumerate(names):
            result_str = str(contestants.iloc[i].get('results', '')).lower()
            eliminated_percent = f'eliminated week {week}' in result_str
            
            rank_eliminated_list = str(row['rank_eliminated']).split(',')
            eliminated_rank = name in rank_eliminated_list
            
            rank_data.append({
                'judge_rank': judge_ranks[i],
                'fan_rank': fan_ranks[i],
                'eliminated_percent': 1 if eliminated_percent else 0,
                'eliminated_rank': 1 if eliminated_rank else 0,
                'both_eliminated': 1 if (eliminated_percent and eliminated_rank) else 0,
                'only_percent': 1 if (eliminated_percent and not eliminated_rank) else 0,
                'only_rank': 1 if (not eliminated_percent and eliminated_rank) else 0,
                'neither': 1 if (not eliminated_percent and not eliminated_rank) else 0
            })
    
    rank_df = pd.DataFrame(rank_data)
    
    # 创建热力图（使用网格聚合）
    max_rank = 10  # 只看前10名
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    
    # 四种情况的热力图
    categories = [
        ('both_eliminated', '两种方法都淘汰', 'Reds'),
        ('only_percent', '仅百分比法淘汰', 'Oranges'),
        ('only_rank', '仅排名法淘汰', 'Blues'),
        ('neither', '两种方法都保留', 'Greens')
    ]
    
    for idx, (col, title, cmap) in enumerate(categories):
        ax = axes[idx // 2, idx % 2]
        
        # 创建热力图矩阵
        heatmap_data = np.zeros((max_rank, max_rank))
        
        for j_rank in range(1, max_rank + 1):
            for f_rank in range(1, max_rank + 1):
                count = rank_df[(rank_df['judge_rank'] == j_rank) & 
                               (rank_df['fan_rank'] == f_rank)][col].sum()
                heatmap_data[j_rank - 1, f_rank - 1] = count
        
        # 绘制热力图
        sns.heatmap(heatmap_data, annot=True, fmt='.0f', cmap=cmap,
                   ax=ax, cbar_kws={'label': '选手数量'},
                   xticklabels=range(1, max_rank + 1),
                   yticklabels=range(1, max_rank + 1))
        
        ax.set_xlabel('观众排名', fontsize=11, fontweight='bold')
        ax.set_ylabel('评委排名', fontsize=11, fontweight='bold')
        ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
        
        # 添加对角线
        ax.plot([0, max_rank], [0, max_rank], 'k--', alpha=0.5, linewidth=2)
    
    plt.suptitle('差异周选手的评委排名-观众排名分布热力图', 
                fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'rank_heatmap.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ 图3已保存: {output_path}")
    plt.close()
    
    return rank_df


def perform_chi_square_test(df: pd.DataFrame):
    """
    卡方检验：判断两种方法差异的显著性
    
    原假设 H0: 两种方法的淘汰结果独立（无系统性差异）
    备择假设 H1: 两种方法的淘汰结果不独立（存在系统性差异）
    """
    print("\n" + "=" * 70)
    print("卡方检验：两种方法差异显著性分析")
    print("=" * 70)
    
    # 只考虑有人淘汰的周数
    df_with_elim = df[df['n_eliminated'] > 0].copy()
    
    # 构建列联表
    # 行: 是否一致 (一致/不一致)
    # 列: 观察值/期望值（如果完全随机）
    
    same_count = df_with_elim['same_result'].sum()
    diff_count = len(df_with_elim) - same_count
    total = len(df_with_elim)
    
    print(f"\n观测数据：")
    print(f"  总周数（有淘汰）: {total}")
    print(f"  一致周数: {same_count} ({same_count/total*100:.2f}%)")
    print(f"  差异周数: {diff_count} ({diff_count/total*100:.2f}%)")
    
    # 方法1: 单样本卡方检验（goodness of fit）
    # 如果两种方法完全随机且独立，期望一致比例应该很低
    # 但实际上，我们需要基于选手数和淘汰数来计算期望一致概率
    
    print(f"\n方法1: 基于随机期望的卡方检验")
    print("-" * 70)
    
    # 计算每周的期望一致概率（基于组合数学）
    expected_consistencies = []
    
    for _, row in df_with_elim.iterrows():
        n = row['n_contestants']
        k = row['n_eliminated']
        
        if k == 0 or k >= n:
            continue
        
        # 期望一致概率 = C(n,k) / C(n,k) = 1 / C(n,k)
        # 但实际上，如果两种方法完全独立随机选择k人，
        # 两次选择完全一致的概率是很低的
        from scipy.special import comb
        total_combinations = comb(n, k, exact=True)
        # 完全一致的概率（两次独立选择选中同样的k个人）
        prob_exact_match = 1.0 / total_combinations if total_combinations > 0 else 0
        expected_consistencies.append(prob_exact_match)
    
    # 期望一致周数
    expected_same = sum(expected_consistencies)
    expected_diff = len(expected_consistencies) - expected_same
    
    print(f"  如果两种方法完全独立随机:")
    print(f"    期望一致周数: {expected_same:.2f}")
    print(f"    期望差异周数: {expected_diff:.2f}")
    print(f"    期望一致比例: {expected_same/len(expected_consistencies)*100:.4f}%")
    
    # 卡方统计量
    observed = np.array([same_count, diff_count])
    expected = np.array([expected_same, expected_diff])
    
    chi2_stat = np.sum((observed - expected) ** 2 / expected)
    dof = 1  # 自由度
    p_value = 1 - np.sum([np.exp(-chi2_stat/2) * (chi2_stat/2)**i / np.math.factorial(i) 
                          for i in range(20)])  # 近似计算
    
    from scipy.stats import chi2
    p_value = 1 - chi2.cdf(chi2_stat, dof)
    
    print(f"\n  卡方统计量: χ² = {chi2_stat:.2f}")
    print(f"  自由度: df = {dof}")
    print(f"  p值: p = {p_value:.6f}")
    
    if p_value < 0.001:
        print(f"  结论: p < 0.001，极其显著 (***)")
        print(f"       强烈拒绝原假设，两种方法存在极显著的系统性差异")
    elif p_value < 0.01:
        print(f"  结论: p < 0.01，非常显著 (**)")
        print(f"       拒绝原假设，两种方法存在显著差异")
    elif p_value < 0.05:
        print(f"  结论: p < 0.05，显著 (*)")
        print(f"       拒绝原假设，两种方法存在差异")
    else:
        print(f"  结论: p ≥ 0.05，不显著")
        print(f"       无法拒绝原假设")
    
    # 方法2: 按季度分组的列联表检验
    print(f"\n\n方法2: 按季度分组的列联表卡方检验")
    print("-" * 70)
    
    # 将季度分为三组: 早期(3-10), 中期(11-19), 晚期(20-27)
    df_with_elim['period'] = pd.cut(df_with_elim['season'], 
                                     bins=[2, 10, 19, 28],
                                     labels=['早期(S3-10)', '中期(S11-19)', '晚期(S20-27)'])
    
    # 构建列联表
    contingency_table = pd.crosstab(df_with_elim['period'], df_with_elim['same_result'])
    contingency_table.columns = ['不一致', '一致']
    
    print("\n列联表（按时期）:")
    print(contingency_table)
    print("\n比例:")
    print(contingency_table.div(contingency_table.sum(axis=1), axis=0).round(3))
    
    # 卡方检验
    chi2_stat2, p_value2, dof2, expected_freq = chi2_contingency(contingency_table)
    
    print(f"\n期望频数:")
    expected_df = pd.DataFrame(expected_freq, 
                              index=contingency_table.index,
                              columns=contingency_table.columns)
    print(expected_df.round(2))
    
    print(f"\n卡方统计量: χ² = {chi2_stat2:.4f}")
    print(f"自由度: df = {dof2}")
    print(f"p值: p = {p_value2:.6f}")
    
    if p_value2 < 0.05:
        print(f"结论: p < 0.05，拒绝原假设")
        print(f"     不同时期的一致性存在显著差异")
    else:
        print(f"结论: p ≥ 0.05，不能拒绝原假设")
        print(f"     不同时期的一致性无显著差异")
    
    # 方法3: 效应量计算（Cramér's V）
    print(f"\n\n效应量分析:")
    print("-" * 70)
    
    # Cramér's V
    n_total = contingency_table.sum().sum()
    cramers_v = np.sqrt(chi2_stat2 / (n_total * min(contingency_table.shape[0] - 1, 
                                                      contingency_table.shape[1] - 1)))
    
    print(f"Cramér's V = {cramers_v:.4f}")
    
    if cramers_v < 0.1:
        print("效应量: 可忽略")
    elif cramers_v < 0.3:
        print("效应量: 小")
    elif cramers_v < 0.5:
        print("效应量: 中等")
    else:
        print("效应量: 大")
    
    print("\n" + "=" * 70)
    
    return {
        'chi2_stat': chi2_stat,
        'p_value': p_value,
        'chi2_stat2': chi2_stat2,
        'p_value2': p_value2,
        'cramers_v': cramers_v,
        'contingency_table': contingency_table
    }


def generate_summary_report(df: pd.DataFrame, stats_df: pd.DataFrame, 
                           test_results: dict, output_dir: str):
    """生成分析总结报告"""
    
    df_with_elim = df[df['n_eliminated'] > 0]
    
    report = f"""
# 排名法淘汰分析 - 可视化与统计检验报告

## 一、基本统计摘要

### 整体统计
- **总周数**: {len(df)} 周
- **有人淘汰周数**: {len(df_with_elim)} 周
- **无人淘汰周数**: {len(df) - len(df_with_elim)} 周

### 一致性统计（仅统计有淘汰周数）
- **一致周数**: {df_with_elim['same_result'].sum()} 周
- **差异周数**: {len(df_with_elim) - df_with_elim['same_result'].sum()} 周
- **一致比例**: {df_with_elim['same_result'].sum() / len(df_with_elim) * 100:.2f}%

### 按季度统计
- **平均一致比例**: {stats_df['consistency_rate'].mean():.2f}%
- **最高一致比例**: {stats_df['consistency_rate'].max():.2f}% (第{stats_df.loc[stats_df['consistency_rate'].idxmax(), 'season']:.0f}季)
- **最低一致比例**: {stats_df['consistency_rate'].min():.2f}% (第{stats_df.loc[stats_df['consistency_rate'].idxmin(), 'season']:.0f}季)
- **标准差**: {stats_df['consistency_rate'].std():.2f}%

## 二、可视化分析结果

### 图1: 一致比例随季数变化趋势
- **文件**: `consistency_by_season.png`
- **主要发现**:
  - 一致比例在不同季度间存在波动
  - 趋势线显示了长期变化模式
  - 部分季度一致性显著高于/低于平均水平

### 图2: 差异周选手百分比分布散点图
- **文件**: `percentage_scatter.png`
- **主要发现**:
  - 展示了在淘汰结果不同的周中，选手的评委-观众百分比分布
  - 不同颜色表示不同的淘汰类别
  - 可观察到两种方法在处理不同百分比组合时的差异

### 图3: 评委排名-观众排名热力图
- **文件**: `rank_heatmap.png`
- **主要发现**:
  - 四个子图分别展示不同淘汰类别的排名分布
  - 热力图颜色深度表示该排名组合的选手数量
  - 对角线表示评委排名=观众排名的情况

## 三、统计检验结果

### 卡方检验1: 基于随机期望
- **卡方统计量**: χ² = {test_results['chi2_stat']:.2f}
- **p值**: p = {test_results['p_value']:.6f}
- **结论**: {"极其显著，两种方法存在系统性差异" if test_results['p_value'] < 0.001 else "显著差异" if test_results['p_value'] < 0.05 else "无显著差异"}

### 卡方检验2: 按时期分组
- **卡方统计量**: χ² = {test_results['chi2_stat2']:.4f}
- **p值**: p = {test_results['p_value2']:.6f}
- **结论**: {"不同时期的一致性存在显著差异" if test_results['p_value2'] < 0.05 else "不同时期的一致性无显著差异"}

### 效应量
- **Cramér's V**: {test_results['cramers_v']:.4f}
- **效应大小**: {"可忽略" if test_results['cramers_v'] < 0.1 else "小" if test_results['cramers_v'] < 0.3 else "中等" if test_results['cramers_v'] < 0.5 else "大"}

## 四、结论与讨论

### 主要发现
1. 排名法与百分比法的淘汰结果在 {df_with_elim['same_result'].sum() / len(df_with_elim) * 100:.1f}% 的周数中一致
2. 卡方检验表明两种方法存在{"极其显著" if test_results['p_value'] < 0.001 else "显著" if test_results['p_value'] < 0.05 else "不显著"}的系统性差异
3. 一致性在不同季度间存在波动，可能与节目规则调整、选手特征等因素有关

### 实际意义
- 两种方法在大多数情况下产生相似结果，但在约 {(1 - df_with_elim['same_result'].sum() / len(df_with_elim)) * 100:.1f}% 的周数中存在差异
- 这些差异可能影响选手的命运，值得进一步研究其产生原因
- 建议结合具体案例分析差异产生的机制

---
**生成时间**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
    
    output_path = os.path.join(output_dir, '可视化与统计分析报告.md')
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"✓ 分析报告已保存: {output_path}")


def main():
    print("=" * 70)
    print("排名法淘汰分析 - 可视化与统计检验")
    print("=" * 70)
    
    # 创建输出目录
    output_dir = os.path.join(SCRIPT_DIR, 'visualization_results')
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n输出目录: {output_dir}\n")
    
    # 加载数据
    print("正在加载数据...")
    df, excel_df, fan_df = load_data()
    print(f"✓ 数据加载完成: {len(df)} 周记录\n")
    
    # 1. 一致比例趋势图
    print("生成图1: 一致比例随季数变化趋势...")
    stats_df = plot_consistency_by_season(df, output_dir)
    
    # 2. 百分比散点图
    print("生成图2: 差异周选手百分比分布散点图...")
    scatter_df = plot_percentage_scatter(df, excel_df, fan_df, output_dir)
    
    # 3. 排名热力图
    print("生成图3: 评委排名-观众排名热力图...")
    rank_df = plot_rank_heatmap(df, excel_df, fan_df, output_dir)
    
    # 4. 卡方检验
    print("\n进行统计检验...")
    test_results = perform_chi_square_test(df)
    
    # 5. 生成报告
    print("\n生成分析报告...")
    generate_summary_report(df, stats_df, test_results, output_dir)
    
    print("\n" + "=" * 70)
    print("所有分析完成！")
    print("=" * 70)
    print(f"\n请查看 {output_dir} 目录下的结果文件")


if __name__ == '__main__':
    main()
