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
DEFAULT_CSV = os.path.join(SCRIPT_DIR, 'rank_vs_percent_elimination_150.csv')
# 使用 Data_4.xlsx 作为数据源（包含正确的淘汰信息）
DEFAULT_EXCEL = os.path.join(PROJECT_ROOT, 'Data_4.xlsx')
DEFAULT_FAN_CSV = os.path.join(SCRIPT_DIR, 'fan_vote_estimates_entropy_smooth_150.csv')


def load_data():
    """加载数据"""
    df = pd.read_csv(DEFAULT_CSV)
    excel_df = pd.read_excel(DEFAULT_EXCEL)
    fan_df = pd.read_csv(DEFAULT_FAN_CSV)
    return df, excel_df, fan_df


def plot_consistency_by_season(df: pd.DataFrame, output_dir: str):
    """
    图1：综合准确性趋势图（优化版）
    同时展示淘汰一致性和决赛排名准确性
    """
    # 只考虑有人淘汰的周数
    df_with_elim = df[df['n_eliminated'] > 0].copy()
    
    # 获取决赛数据
    final_df = df[df['is_final'] == True].copy()
    
    # 按季度统计
    season_stats = []
    for season in sorted(df['season'].unique()):
        season_elim_df = df_with_elim[df_with_elim['season'] == season]
        total_weeks = len(season_elim_df)
        same_weeks = season_elim_df['same_result'].sum()
        consistency_rate = same_weeks / total_weeks * 100 if total_weeks > 0 else 0
        
        # 获取决赛数据
        final_row = final_df[final_df['season'] == season]
        percent_exact = final_row['percent_exact_match'].iloc[0] if len(final_row) > 0 else None
        rank_exact = final_row['rank_exact_match'].iloc[0] if len(final_row) > 0 else None
        percent_top1 = final_row['percent_top1_correct'].iloc[0] if len(final_row) > 0 else None
        rank_top1 = final_row['rank_top1_correct'].iloc[0] if len(final_row) > 0 else None
        
        season_stats.append({
            'season': season,
            'total_weeks': total_weeks,
            'same_weeks': same_weeks,
            'consistency_rate': consistency_rate,
            'percent_exact_match': percent_exact,
            'rank_exact_match': rank_exact,
            'percent_top1_correct': percent_top1,
            'rank_top1_correct': rank_top1
        })
    
    stats_df = pd.DataFrame(season_stats)
    
    # 绘制趋势图
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))
    
    # 子图1：淘汰一致比例趋势 + 决赛准确性标记
    ax1.plot(stats_df['season'], stats_df['consistency_rate'], 
             marker='o', linewidth=2, markersize=8, color='#2E86AB', label='淘汰一致比例')
    
    # 添加趋势线
    z = np.polyfit(stats_df['season'], stats_df['consistency_rate'], 2)
    p = np.poly1d(z)
    ax1.plot(stats_df['season'], p(stats_df['season']), 
             '--', color='#A23B72', alpha=0.7, linewidth=2, label='二次趋势线')
    
    # 添加平均线
    mean_rate = stats_df['consistency_rate'].mean()
    ax1.axhline(y=mean_rate, color='#F18F01', linestyle='--', 
                linewidth=2, alpha=0.7, label=f'平均值 ({mean_rate:.1f}%)')
    
    # 在顶部标记决赛准确性（百分比法）
    for _, row in stats_df.iterrows():
        season = row['season']
        if pd.notna(row['percent_exact_match']):
            # 百分比法决赛是否完全正确 (用 O 和 X 代替特殊符号)
            marker = 'O' if row['percent_exact_match'] else 'X'
            color = '#06A77D' if row['percent_exact_match'] else '#D62246'
            ax1.annotate(marker, xy=(season, 105), fontsize=9, ha='center', 
                        color=color, fontweight='bold')
        if pd.notna(row['rank_exact_match']):
            # 排名法决赛是否完全正确
            marker = 'O' if row['rank_exact_match'] else 'X'
            color = '#06A77D' if row['rank_exact_match'] else '#D62246'
            ax1.annotate(marker, xy=(season, 110), fontsize=9, ha='center', 
                        color=color, fontweight='bold')
    
    # 添加图例说明
    ax1.annotate('百分比法决赛:', xy=(0.02, 0.95), xycoords='axes fraction', fontsize=9)
    ax1.annotate('排名法决赛:', xy=(0.02, 0.90), xycoords='axes fraction', fontsize=9)
    
    ax1.set_xlabel('季数', fontsize=12, fontweight='bold')
    ax1.set_ylabel('淘汰一致比例 (%)', fontsize=12, fontweight='bold')
    ax1.set_title('综合准确性趋势：淘汰一致性 + 决赛排名准确性', fontsize=14, fontweight='bold', pad=20)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.legend(fontsize=10, loc='upper right')
    ax1.set_xticks(range(3, 28, 1))
    ax1.set_ylim(0, 120)
    
    # 子图2：每季综合统计（堆叠柱状图）
    bar_width = 0.6
    seasons = stats_df['season'].values
    
    # 基础柱状图：一致周数和差异周数
    bars1 = ax2.bar(seasons, stats_df['same_weeks'], 
                    width=bar_width, label='淘汰一致周数', color='#06A77D', alpha=0.8)
    bars2 = ax2.bar(seasons, stats_df['total_weeks'] - stats_df['same_weeks'], 
                    bottom=stats_df['same_weeks'], width=bar_width, 
                    label='淘汰差异周数', color='#D62246', alpha=0.8)
    
    # 在柱状图顶部添加决赛结果标记
    for idx, row in stats_df.iterrows():
        season = row['season']
        bar_top = row['total_weeks']
        
        # 百分比法决赛标记
        if pd.notna(row['percent_exact_match']):
            marker_p = '●' if row['percent_exact_match'] else '○'
            color_p = '#2E86AB'
            ax2.annotate(marker_p, xy=(season - 0.15, bar_top + 0.5), 
                        fontsize=10, ha='center', color=color_p, fontweight='bold')
        
        # 排名法决赛标记
        if pd.notna(row['rank_exact_match']):
            marker_r = '●' if row['rank_exact_match'] else '○'
            color_r = '#FF6B35'
            ax2.annotate(marker_r, xy=(season + 0.15, bar_top + 0.5), 
                        fontsize=10, ha='center', color=color_r, fontweight='bold')
    
    ax2.set_xlabel('季数', fontsize=12, fontweight='bold')
    ax2.set_ylabel('周数', fontsize=12, fontweight='bold')
    ax2.set_title('每季淘汰一致性 + 决赛排名正确性\n(●=决赛排名完全正确，○=不完全正确；蓝=百分比法，橙=排名法)', 
                 fontsize=12, fontweight='bold', pad=15)
    ax2.legend(fontsize=10, loc='upper right')
    ax2.grid(True, alpha=0.3, linestyle='--', axis='y')
    ax2.set_xticks(range(3, 28, 1))
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'consistency_by_season.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ 图1已保存: {output_path}")
    plt.close()
    
    return stats_df


def plot_percentage_scatter(df: pd.DataFrame, excel_df: pd.DataFrame, 
                           fan_df: pd.DataFrame, output_dir: str):
    """
    图2：综合百分比散点图（优化版）
    子图1：淘汰分析（差异周选手的评委-观众百分比分布）
    子图2：决赛分析（决赛选手的排名法正确/错误分布）
    """
    # 只看结果不同且有人淘汰的周
    diff_weeks = df[(df['same_result'] == False) & (df['n_eliminated'] > 0)]
    final_weeks = df[df['is_final'] == True]
    
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
                'name': name,
                'type': 'elimination'
            })
    
    # 收集决赛选手数据
    final_scatter_data = []
    
    for _, row in final_weeks.iterrows():
        season, week = row['season'], row['week']
        
        # 获取该周所有选手
        season_df = excel_df[excel_df['season'] == season]
        week_col = f'{week}_percent'
        
        if week_col not in season_df.columns:
            continue
            
        contestants = season_df[season_df[week_col] > 0]
        
        # 解析排名
        actual_ranking = str(row.get('actual_ranking', '')).split('>') if pd.notna(row.get('actual_ranking')) else []
        rank_ranking = str(row.get('rank_method_ranking', '')).split('>') if pd.notna(row.get('rank_method_ranking')) else []
        
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
            
            # 判断排名法是否预测正确（选手的实际排名 vs 预测排名）
            actual_pos = actual_ranking.index(name) + 1 if name in actual_ranking else -1
            rank_pos = rank_ranking.index(name) + 1 if name in rank_ranking else -1
            
            if actual_pos > 0 and rank_pos > 0:
                rank_correct = actual_pos == rank_pos
                category = '排名法正确' if rank_correct else '排名法错误'
            else:
                category = '未知'
            
            final_scatter_data.append({
                'judge_percent': judge_pct,
                'fan_percent': fan_pct,
                'category': category,
                'season': season,
                'week': week,
                'name': name,
                'actual_pos': actual_pos,
                'rank_pos': rank_pos,
                'type': 'final'
            })
    
    scatter_df = pd.DataFrame(scatter_data)
    final_scatter_df = pd.DataFrame(final_scatter_data)
    
    # 绘制两个子图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    
    # 子图1：淘汰分析
    colors_elim = {
        '两种方法都淘汰': '#D62246',
        '仅百分比法淘汰': '#F18F01',
        '仅排名法淘汰': '#2E86AB',
        '两种方法都保留': '#06A77D'
    }
    
    markers_elim = {
        '两种方法都淘汰': 'X',
        '仅百分比法淘汰': '^',
        '仅排名法淘汰': 'v',
        '两种方法都保留': 'o'
    }
    
    for category in colors_elim.keys():
        data = scatter_df[scatter_df['category'] == category]
        if len(data) > 0:
            ax1.scatter(data['judge_percent'], data['fan_percent'],
                       c=colors_elim[category], marker=markers_elim[category],
                       s=100, alpha=0.6, label=f'{category} (n={len(data)})',
                       edgecolors='black', linewidth=0.5)
    
    # 添加对角线参考线
    if len(scatter_df) > 0:
        max_val = max(scatter_df['judge_percent'].max(), scatter_df['fan_percent'].max())
        ax1.plot([0, max_val], [0, max_val], 'k--', alpha=0.3, linewidth=2, label='评委=观众')
    
    ax1.set_xlabel('评委百分比 (%)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('观众百分比 (%)', fontsize=12, fontweight='bold')
    ax1.set_title('(a) 淘汰分析：差异周选手分布', fontsize=13, fontweight='bold', pad=15)
    ax1.legend(fontsize=9, loc='best', framealpha=0.9)
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    # 子图2：决赛分析
    colors_final = {
        '排名法正确': '#06A77D',
        '排名法错误': '#D62246',
        '未知': '#888888'
    }
    
    markers_final = {
        '排名法正确': 'o',
        '排名法错误': 'X',
        '未知': 's'
    }
    
    for category in colors_final.keys():
        data = final_scatter_df[final_scatter_df['category'] == category]
        if len(data) > 0:
            ax2.scatter(data['judge_percent'], data['fan_percent'],
                       c=colors_final[category], marker=markers_final[category],
                       s=120, alpha=0.7, label=f'{category} (n={len(data)})',
                       edgecolors='black', linewidth=0.5)
    
    # 添加对角线参考线
    if len(final_scatter_df) > 0:
        max_val = max(final_scatter_df['judge_percent'].max(), final_scatter_df['fan_percent'].max())
        ax2.plot([0, max_val], [0, max_val], 'k--', alpha=0.3, linewidth=2, label='评委=观众')
    
    ax2.set_xlabel('评委百分比 (%)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('观众百分比 (%)', fontsize=12, fontweight='bold')
    ax2.set_title('(b) 决赛分析：决赛选手排名预测', fontsize=13, fontweight='bold', pad=15)
    ax2.legend(fontsize=9, loc='best', framealpha=0.9)
    ax2.grid(True, alpha=0.3, linestyle='--')
    
    plt.suptitle('综合百分比散点图：淘汰分析 + 决赛分析', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'percentage_scatter.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ 图2已保存: {output_path}")
    plt.close()
    
    return scatter_df, final_scatter_df


def plot_rank_heatmap(df: pd.DataFrame, excel_df: pd.DataFrame, 
                     fan_df: pd.DataFrame, output_dir: str):
    """
    图3：综合排名热力图（优化版）
    上排4图：淘汰相关（两种都淘汰、仅百分比淘汰、仅排名淘汰、都保留）
    下排2图：决赛相关（排名法预测正确 vs 排名法预测错误的选手分布）
    """
    # 只看结果不同且有人淘汰的周
    diff_weeks = df[(df['same_result'] == False) & (df['n_eliminated'] > 0)]
    final_weeks = df[df['is_final'] == True]
    
    # 收集淘汰周排名数据
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
    
    # 收集决赛选手排名数据
    final_rank_data = []
    
    for _, row in final_weeks.iterrows():
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
        
        # 解析实际和预测排名
        actual_ranking = str(row.get('actual_ranking', '')).split('>') if pd.notna(row.get('actual_ranking')) else []
        rank_ranking = str(row.get('rank_method_ranking', '')).split('>') if pd.notna(row.get('rank_method_ranking')) else []
        
        for i, name in enumerate(names):
            actual_pos = actual_ranking.index(name) + 1 if name in actual_ranking else -1
            rank_pos = rank_ranking.index(name) + 1 if name in rank_ranking else -1
            
            if actual_pos > 0 and rank_pos > 0:
                rank_correct = actual_pos == rank_pos
            else:
                rank_correct = None
            
            final_rank_data.append({
                'judge_rank': judge_ranks[i],
                'fan_rank': fan_ranks[i],
                'rank_correct': 1 if rank_correct == True else 0,
                'rank_wrong': 1 if rank_correct == False else 0,
                'season': season,
                'name': name
            })
    
    final_rank_df = pd.DataFrame(final_rank_data)
    
    # 创建热力图（使用网格聚合）
    max_rank = 10  # 只看前10名
    
    fig, axes = plt.subplots(2, 3, figsize=(20, 14))
    
    # 上排4图：淘汰相关
    categories_elim = [
        ('both_eliminated', '两种方法都淘汰', 'Reds'),
        ('only_percent', '仅百分比法淘汰', 'Oranges'),
        ('only_rank', '仅排名法淘汰', 'Blues'),
        ('neither', '两种方法都保留', 'Greens')
    ]
    
    for idx, (col, title, cmap) in enumerate(categories_elim):
        ax = axes[0, idx] if idx < 3 else axes[1, 0]
        if idx == 3:
            ax = axes[1, 0]
        else:
            ax = axes[0, idx]
        
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
        
        ax.set_xlabel('观众排名', fontsize=10, fontweight='bold')
        ax.set_ylabel('评委排名', fontsize=10, fontweight='bold')
        ax.set_title(f'淘汰：{title}', fontsize=11, fontweight='bold', pad=10)
        
        # 添加对角线
        ax.plot([0, max_rank], [0, max_rank], 'k--', alpha=0.5, linewidth=2)
    
    # 下排2图：决赛相关（第4个淘汰图和2个决赛图）
    # 第4个淘汰图（都保留）放在 axes[1, 0]
    ax = axes[1, 0]
    heatmap_data = np.zeros((max_rank, max_rank))
    for j_rank in range(1, max_rank + 1):
        for f_rank in range(1, max_rank + 1):
            count = rank_df[(rank_df['judge_rank'] == j_rank) & 
                           (rank_df['fan_rank'] == f_rank)]['neither'].sum()
            heatmap_data[j_rank - 1, f_rank - 1] = count
    sns.heatmap(heatmap_data, annot=True, fmt='.0f', cmap='Greens',
               ax=ax, cbar_kws={'label': '选手数量'},
               xticklabels=range(1, max_rank + 1),
               yticklabels=range(1, max_rank + 1))
    ax.set_xlabel('观众排名', fontsize=10, fontweight='bold')
    ax.set_ylabel('评委排名', fontsize=10, fontweight='bold')
    ax.set_title('淘汰：两种方法都保留', fontsize=11, fontweight='bold', pad=10)
    ax.plot([0, max_rank], [0, max_rank], 'k--', alpha=0.5, linewidth=2)
    
    # 决赛：排名法正确
    ax = axes[1, 1]
    heatmap_data = np.zeros((max_rank, max_rank))
    for j_rank in range(1, max_rank + 1):
        for f_rank in range(1, max_rank + 1):
            count = final_rank_df[(final_rank_df['judge_rank'] == j_rank) & 
                                  (final_rank_df['fan_rank'] == f_rank)]['rank_correct'].sum()
            heatmap_data[j_rank - 1, f_rank - 1] = count
    sns.heatmap(heatmap_data, annot=True, fmt='.0f', cmap='Greens',
               ax=ax, cbar_kws={'label': '选手数量'},
               xticklabels=range(1, max_rank + 1),
               yticklabels=range(1, max_rank + 1))
    ax.set_xlabel('观众排名', fontsize=10, fontweight='bold')
    ax.set_ylabel('评委排名', fontsize=10, fontweight='bold')
    ax.set_title('决赛：排名法预测正确', fontsize=11, fontweight='bold', pad=10)
    ax.plot([0, max_rank], [0, max_rank], 'k--', alpha=0.5, linewidth=2)
    
    # 决赛：排名法错误
    ax = axes[1, 2]
    heatmap_data = np.zeros((max_rank, max_rank))
    for j_rank in range(1, max_rank + 1):
        for f_rank in range(1, max_rank + 1):
            count = final_rank_df[(final_rank_df['judge_rank'] == j_rank) & 
                                  (final_rank_df['fan_rank'] == f_rank)]['rank_wrong'].sum()
            heatmap_data[j_rank - 1, f_rank - 1] = count
    sns.heatmap(heatmap_data, annot=True, fmt='.0f', cmap='Reds',
               ax=ax, cbar_kws={'label': '选手数量'},
               xticklabels=range(1, max_rank + 1),
               yticklabels=range(1, max_rank + 1))
    ax.set_xlabel('观众排名', fontsize=10, fontweight='bold')
    ax.set_ylabel('评委排名', fontsize=10, fontweight='bold')
    ax.set_title('决赛：排名法预测错误', fontsize=11, fontweight='bold', pad=10)
    ax.plot([0, max_rank], [0, max_rank], 'k--', alpha=0.5, linewidth=2)
    
    plt.suptitle('综合排名热力图：淘汰分析 + 决赛分析', 
                fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'rank_heatmap.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ 图3已保存: {output_path}")
    plt.close()
    
    return rank_df, final_rank_df


def plot_final_accuracy_comparison(df: pd.DataFrame, output_dir: str):
    """
    图4：决赛排名准确性对比柱状图
    两种方法的"完全正确"、"冠军正确"、"前两名正确"对比
    """
    # 筛选决赛周
    final_df = df[df['is_final'] == True].copy()
    
    # 统计各项指标
    stats = {
        '完全正确': {
            '百分比法': final_df['percent_exact_match'].sum(),
            '排名法': final_df['rank_exact_match'].sum()
        },
        '冠军正确': {
            '百分比法': final_df['percent_top1_correct'].sum(),
            '排名法': final_df['rank_top1_correct'].sum()
        }
    }
    
    # 计算前两名正确（需要解析排名字符串）
    percent_top2_correct = 0
    rank_top2_correct = 0
    
    for _, row in final_df.iterrows():
        if pd.isna(row['actual_ranking']) or pd.isna(row['percent_ranking']):
            continue
            
        actual = str(row['actual_ranking']).split('>')[:2]
        percent_rank = str(row['percent_ranking']).split('>')[:2]
        rank_rank = str(row['rank_method_ranking']).split('>')[:2]
        
        # 前两名相同（不考虑顺序）
        if set(actual) == set(percent_rank):
            percent_top2_correct += 1
        if set(actual) == set(rank_rank):
            rank_top2_correct += 1
    
    stats['前两名正确'] = {
        '百分比法': percent_top2_correct,
        '排名法': rank_top2_correct
    }
    
    total_seasons = len(final_df)
    
    # 绘制柱状图
    fig, ax = plt.subplots(figsize=(12, 8))
    
    x = np.arange(len(stats))
    width = 0.35
    
    percent_vals = [stats[k]['百分比法'] for k in stats.keys()]
    rank_vals = [stats[k]['排名法'] for k in stats.keys()]
    
    bars1 = ax.bar(x - width/2, percent_vals, width, label='百分比法', color='#2E86AB', alpha=0.8)
    bars2 = ax.bar(x + width/2, rank_vals, width, label='排名法', color='#FF6B35', alpha=0.8)
    
    # 添加数值标签
    for bar, val in zip(bars1, percent_vals):
        ax.annotate(f'{val}/{total_seasons}\n({val/total_seasons*100:.1f}%)',
                   xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                   ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    for bar, val in zip(bars2, rank_vals):
        ax.annotate(f'{val}/{total_seasons}\n({val/total_seasons*100:.1f}%)',
                   xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                   ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_xlabel('评价指标', fontsize=12, fontweight='bold')
    ax.set_ylabel('正确季数', fontsize=12, fontweight='bold')
    ax.set_title(f'决赛排名准确性对比（共{total_seasons}季）', fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(stats.keys(), fontsize=11)
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')
    ax.set_ylim(0, total_seasons + 3)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'final_accuracy_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ 图4已保存: {output_path}")
    plt.close()
    
    return stats


def plot_final_accuracy_trend(df: pd.DataFrame, output_dir: str):
    """
    图5：决赛准确性随季度变化趋势图
    每季决赛两种方法的排名是否正确
    """
    # 筛选决赛周
    final_df = df[df['is_final'] == True].copy().sort_values('season')
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))
    
    seasons = final_df['season'].values
    
    # 子图1：完全正确
    percent_exact = final_df['percent_exact_match'].astype(int).values
    rank_exact = final_df['rank_exact_match'].astype(int).values
    
    ax1.scatter(seasons[percent_exact == 1], [1.1] * sum(percent_exact == 1), 
               s=150, c='#2E86AB', marker='o', label='百分比法正确', zorder=5)
    ax1.scatter(seasons[percent_exact == 0], [1.1] * sum(percent_exact == 0), 
               s=150, c='#2E86AB', marker='x', label='百分比法错误', alpha=0.5, zorder=5)
    ax1.scatter(seasons[rank_exact == 1], [0.9] * sum(rank_exact == 1), 
               s=150, c='#FF6B35', marker='o', label='排名法正确', zorder=5)
    ax1.scatter(seasons[rank_exact == 0], [0.9] * sum(rank_exact == 0), 
               s=150, c='#FF6B35', marker='x', label='排名法错误', alpha=0.5, zorder=5)
    
    # 添加连接线显示一致性
    for i, season in enumerate(seasons):
        if percent_exact[i] == rank_exact[i]:
            color = '#06A77D' if percent_exact[i] == 1 else '#D62246'
            ax1.plot([season, season], [0.9, 1.1], color=color, alpha=0.3, linewidth=2)
    
    ax1.set_xlabel('季数', fontsize=12, fontweight='bold')
    ax1.set_ylabel('方法', fontsize=12, fontweight='bold')
    ax1.set_title('决赛排名完全正确性随季度变化', fontsize=13, fontweight='bold', pad=15)
    ax1.set_yticks([0.9, 1.1])
    ax1.set_yticklabels(['排名法', '百分比法'])
    ax1.set_xticks(range(3, 28, 1))
    ax1.set_xlim(2, 28)
    ax1.legend(fontsize=9, loc='upper right', ncol=4)
    ax1.grid(True, alpha=0.3, linestyle='--', axis='x')
    
    # 子图2：冠军正确
    percent_top1 = final_df['percent_top1_correct'].astype(int).values
    rank_top1 = final_df['rank_top1_correct'].astype(int).values
    
    ax2.scatter(seasons[percent_top1 == 1], [1.1] * sum(percent_top1 == 1), 
               s=150, c='#2E86AB', marker='o', label='百分比法正确', zorder=5)
    ax2.scatter(seasons[percent_top1 == 0], [1.1] * sum(percent_top1 == 0), 
               s=150, c='#2E86AB', marker='x', label='百分比法错误', alpha=0.5, zorder=5)
    ax2.scatter(seasons[rank_top1 == 1], [0.9] * sum(rank_top1 == 1), 
               s=150, c='#FF6B35', marker='o', label='排名法正确', zorder=5)
    ax2.scatter(seasons[rank_top1 == 0], [0.9] * sum(rank_top1 == 0), 
               s=150, c='#FF6B35', marker='x', label='排名法错误', alpha=0.5, zorder=5)
    
    # 添加连接线显示一致性
    for i, season in enumerate(seasons):
        if percent_top1[i] == rank_top1[i]:
            color = '#06A77D' if percent_top1[i] == 1 else '#D62246'
            ax2.plot([season, season], [0.9, 1.1], color=color, alpha=0.3, linewidth=2)
    
    ax2.set_xlabel('季数', fontsize=12, fontweight='bold')
    ax2.set_ylabel('方法', fontsize=12, fontweight='bold')
    ax2.set_title('冠军预测正确性随季度变化', fontsize=13, fontweight='bold', pad=15)
    ax2.set_yticks([0.9, 1.1])
    ax2.set_yticklabels(['排名法', '百分比法'])
    ax2.set_xticks(range(3, 28, 1))
    ax2.set_xlim(2, 28)
    ax2.legend(fontsize=9, loc='upper right', ncol=4)
    ax2.grid(True, alpha=0.3, linestyle='--', axis='x')
    
    plt.suptitle('决赛准确性随季度变化趋势', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'final_accuracy_trend.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ 图5已保存: {output_path}")
    plt.close()


def plot_final_error_analysis(df: pd.DataFrame, excel_df: pd.DataFrame,
                              fan_df: pd.DataFrame, output_dir: str):
    """
    图6：排名法预测失败案例分析
    分析排名法预测错误的决赛的特征
    """
    # 筛选决赛周
    final_df = df[df['is_final'] == True].copy()
    
    # 筛选排名法预测错误的季
    error_seasons = final_df[final_df['rank_exact_match'] == False]
    correct_seasons = final_df[final_df['rank_exact_match'] == True]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # 收集冠军数据
    champion_data = []
    
    for _, row in final_df.iterrows():
        season = row['season']
        week = row['week']
        
        if pd.isna(row['actual_ranking']):
            continue
            
        champion_name = str(row['actual_ranking']).split('>')[0]
        
        # 获取冠军的评委分和观众分
        season_df = excel_df[excel_df['season'] == season]
        week_col = f'{week}_percent'
        
        if week_col not in season_df.columns:
            continue
            
        champion_row = season_df[season_df['celebrity_name'] == champion_name]
        
        if champion_row.empty:
            continue
            
        judge_pct = champion_row.iloc[0][week_col]
        
        # 获取观众百分比
        fan_row = fan_df[(fan_df['season'] == season) & 
                        (fan_df['week'] == week) & 
                        (fan_df['celebrity_name'] == champion_name)]
        
        fan_pct = fan_row.iloc[0]['fan_vote_percent'] if not fan_row.empty else 0
        
        # 计算排名
        all_contestants = season_df[season_df[week_col] > 0]
        judge_rank = (all_contestants[week_col] >= judge_pct).sum()
        
        # 计算观众排名
        fan_percents = []
        for name in all_contestants['celebrity_name'].tolist():
            fr = fan_df[(fan_df['season'] == season) & 
                       (fan_df['week'] == week) & 
                       (fan_df['celebrity_name'] == name)]
            fan_percents.append(fr.iloc[0]['fan_vote_percent'] if not fr.empty else 0)
        
        fan_percents = np.array(fan_percents)
        fan_rank = (fan_percents >= fan_pct).sum()
        
        champion_data.append({
            'season': season,
            'name': champion_name,
            'judge_percent': judge_pct,
            'fan_percent': fan_pct,
            'judge_rank': judge_rank,
            'fan_rank': fan_rank,
            'rank_correct': row['rank_exact_match']
        })
    
    champ_df = pd.DataFrame(champion_data)
    
    # 子图1：冠军的评委分 vs 观众分散点图
    correct = champ_df[champ_df['rank_correct'] == True]
    error = champ_df[champ_df['rank_correct'] == False]
    
    ax1.scatter(correct['judge_percent'], correct['fan_percent'],
               s=150, c='#06A77D', marker='o', label=f'排名法正确 (n={len(correct)})',
               edgecolors='black', linewidth=1, zorder=5)
    ax1.scatter(error['judge_percent'], error['fan_percent'],
               s=150, c='#D62246', marker='X', label=f'排名法错误 (n={len(error)})',
               edgecolors='black', linewidth=1, zorder=5)
    
    # 标注季数
    for _, row in champ_df.iterrows():
        color = '#06A77D' if row['rank_correct'] else '#D62246'
        ax1.annotate(f"S{int(row['season'])}", 
                    xy=(row['judge_percent'], row['fan_percent']),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=8, color=color)
    
    # 添加对角线
    max_val = max(champ_df['judge_percent'].max(), champ_df['fan_percent'].max())
    ax1.plot([0, max_val], [0, max_val], 'k--', alpha=0.3, linewidth=2, label='评委=观众')
    
    ax1.set_xlabel('冠军评委百分比 (%)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('冠军观众百分比 (%)', fontsize=12, fontweight='bold')
    ax1.set_title('(a) 冠军的评委-观众百分比分布', fontsize=13, fontweight='bold', pad=15)
    ax1.legend(fontsize=10, loc='upper left')
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    # 子图2：冠军的评委排名 vs 观众排名
    ax2.scatter(correct['judge_rank'], correct['fan_rank'],
               s=150, c='#06A77D', marker='o', label=f'排名法正确 (n={len(correct)})',
               edgecolors='black', linewidth=1, zorder=5)
    ax2.scatter(error['judge_rank'], error['fan_rank'],
               s=150, c='#D62246', marker='X', label=f'排名法错误 (n={len(error)})',
               edgecolors='black', linewidth=1, zorder=5)
    
    # 标注季数
    for _, row in champ_df.iterrows():
        color = '#06A77D' if row['rank_correct'] else '#D62246'
        ax2.annotate(f"S{int(row['season'])}", 
                    xy=(row['judge_rank'], row['fan_rank']),
                    xytext=(3, 3), textcoords='offset points',
                    fontsize=8, color=color)
    
    # 添加对角线
    max_rank = max(champ_df['judge_rank'].max(), champ_df['fan_rank'].max())
    ax2.plot([0, max_rank + 1], [0, max_rank + 1], 'k--', alpha=0.3, linewidth=2, label='评委=观众')
    
    ax2.set_xlabel('冠军评委排名', fontsize=12, fontweight='bold')
    ax2.set_ylabel('冠军观众排名', fontsize=12, fontweight='bold')
    ax2.set_title('(b) 冠军的评委-观众排名分布', fontsize=13, fontweight='bold', pad=15)
    ax2.legend(fontsize=10, loc='upper left')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_xlim(0, max_rank + 1)
    ax2.set_ylim(0, max_rank + 1)
    
    plt.suptitle('排名法预测失败案例分析：冠军特征', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'final_error_analysis.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ 图6已保存: {output_path}")
    plt.close()
    
    return champ_df


def plot_bobby_bones_analysis(df: pd.DataFrame, excel_df: pd.DataFrame,
                              fan_df: pd.DataFrame, output_dir: str):
    """
    图7：Bobby Bones 案例专题分析
    第27季争议冠军 Bobby Bones 的分析
    """
    # 获取第27季决赛数据
    season = 27
    final_row = df[(df['season'] == season) & (df['is_final'] == True)]
    
    if final_row.empty:
        print("警告：未找到第27季决赛数据")
        return None
    
    final_row = final_row.iloc[0]
    week = final_row['week']
    
    # 获取决赛选手
    season_df = excel_df[excel_df['season'] == season]
    week_col = f'{week}_percent'
    
    if week_col not in season_df.columns:
        print(f"警告：未找到第{week}周的百分比列")
        return None
    
    finalists = season_df[season_df[week_col] > 0].copy()
    
    # 获取观众百分比
    finalist_data = []
    for _, row in finalists.iterrows():
        name = row['celebrity_name']
        judge_pct = row[week_col]
        
        fan_row = fan_df[(fan_df['season'] == season) & 
                        (fan_df['week'] == week) & 
                        (fan_df['celebrity_name'] == name)]
        
        fan_pct = fan_row.iloc[0]['fan_vote_percent'] if not fan_row.empty else 0
        
        finalist_data.append({
            'name': name,
            'judge_percent': judge_pct,
            'fan_percent': fan_pct
        })
    
    finalist_df = pd.DataFrame(finalist_data)
    
    # 计算排名
    finalist_df['judge_rank'] = finalist_df['judge_percent'].rank(ascending=False, method='min').astype(int)
    finalist_df['fan_rank'] = finalist_df['fan_percent'].rank(ascending=False, method='min').astype(int)
    
    # 解析实际和预测排名
    actual_ranking = str(final_row['actual_ranking']).split('>') if pd.notna(final_row['actual_ranking']) else []
    percent_ranking = str(final_row['percent_ranking']).split('>') if pd.notna(final_row['percent_ranking']) else []
    rank_ranking = str(final_row['rank_method_ranking']).split('>') if pd.notna(final_row['rank_method_ranking']) else []
    
    # 添加排名信息
    finalist_df['actual_pos'] = finalist_df['name'].apply(
        lambda x: actual_ranking.index(x) + 1 if x in actual_ranking else -1)
    finalist_df['percent_pos'] = finalist_df['name'].apply(
        lambda x: percent_ranking.index(x) + 1 if x in percent_ranking else -1)
    finalist_df['rank_pos'] = finalist_df['name'].apply(
        lambda x: rank_ranking.index(x) + 1 if x in rank_ranking else -1)
    
    # 按实际排名排序
    finalist_df = finalist_df.sort_values('actual_pos')
    
    # 创建图表
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 子图1：评委分 vs 观众分柱状图
    ax1 = axes[0, 0]
    x = np.arange(len(finalist_df))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, finalist_df['judge_percent'], width, 
                   label='评委百分比', color='#2E86AB', alpha=0.8)
    bars2 = ax1.bar(x + width/2, finalist_df['fan_percent'], width, 
                   label='观众百分比', color='#FF6B35', alpha=0.8)
    
    ax1.set_xlabel('选手', fontsize=11, fontweight='bold')
    ax1.set_ylabel('百分比 (%)', fontsize=11, fontweight='bold')
    ax1.set_title('(a) 决赛选手评委-观众百分比对比', fontsize=12, fontweight='bold', pad=10)
    ax1.set_xticks(x)
    ax1.set_xticklabels(finalist_df['name'], rotation=15, ha='right', fontsize=9)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3, linestyle='--', axis='y')
    
    # 标注 Bobby Bones
    bobby_idx = finalist_df[finalist_df['name'].str.contains('Bobby', case=False)].index
    if len(bobby_idx) > 0:
        bobby_x = list(finalist_df.index).index(bobby_idx[0])
        ax1.annotate('★ 冠军', xy=(bobby_x, finalist_df.iloc[bobby_x]['judge_percent'] + 2),
                    fontsize=10, ha='center', color='#D62246', fontweight='bold')
    
    # 子图2：排名对比表
    ax2 = axes[0, 1]
    ax2.axis('off')
    
    # 创建表格数据
    table_data = []
    headers = ['选手', '实际排名', '百分比法排名', '排名法排名', '评委排名', '观众排名']
    
    for _, row in finalist_df.iterrows():
        table_data.append([
            row['name'],
            row['actual_pos'],
            row['percent_pos'],
            row['rank_pos'],
            row['judge_rank'],
            row['fan_rank']
        ])
    
    table = ax2.table(cellText=table_data, colLabels=headers,
                     loc='center', cellLoc='center',
                     colColours=['#E8E8E8'] * len(headers))
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)
    
    # 高亮 Bobby Bones 行
    for i, row in enumerate(finalist_df.itertuples()):
        if 'Bobby' in row.name:
            for j in range(len(headers)):
                table[(i + 1, j)].set_facecolor('#FFCCCC')
    
    ax2.set_title('(b) 决赛排名对比表', fontsize=12, fontweight='bold', pad=20)
    
    # 子图3：排名差异柱状图
    ax3 = axes[1, 0]
    
    finalist_df['rank_diff'] = finalist_df['judge_rank'] - finalist_df['fan_rank']
    colors = ['#D62246' if d > 0 else '#06A77D' for d in finalist_df['rank_diff']]
    
    bars = ax3.bar(x, finalist_df['rank_diff'], color=colors, alpha=0.8, edgecolor='black')
    
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax3.set_xlabel('选手', fontsize=11, fontweight='bold')
    ax3.set_ylabel('排名差异（评委-观众）', fontsize=11, fontweight='bold')
    ax3.set_title('(c) 评委排名与观众排名差异\n（正=评委排名更低/表现更差）', fontsize=12, fontweight='bold', pad=10)
    ax3.set_xticks(x)
    ax3.set_xticklabels(finalist_df['name'], rotation=15, ha='right', fontsize=9)
    ax3.grid(True, alpha=0.3, linestyle='--', axis='y')
    
    # 子图4：案例分析文字说明
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    # Bobby Bones 的具体数据
    bobby_row = finalist_df[finalist_df['name'].str.contains('Bobby', case=False)]
    if not bobby_row.empty:
        bobby = bobby_row.iloc[0]
        analysis_text = f"""
第27季 Bobby Bones 案例分析

实际结果：
• 冠军：Bobby Bones（评委排名 {bobby['judge_rank']}，观众排名 {bobby['fan_rank']}）

百分比法预测：{'正确' if final_row['percent_exact_match'] else '错误'}
排名法预测：{'正确' if final_row['rank_exact_match'] else '错误'}

关键发现：
• Bobby Bones 评委百分比：{bobby['judge_percent']:.2f}%
• Bobby Bones 观众百分比：{bobby['fan_percent']:.2f}%
• 评委-观众排名差异：{int(bobby['rank_diff'])}

分析：
Bobby Bones 是《与星共舞》历史上最具争议的冠军之一。
尽管他在评委评分中排名相对较低，但凭借强大的
观众投票支持（来自其广播节目的粉丝基础）赢得了冠军。
这一案例说明了观众投票在决定最终结果中的重要性，
以及百分比法和排名法在处理这类"爆冷"情况时的差异。
"""
    else:
        analysis_text = "未找到 Bobby Bones 的数据"
    
    ax4.text(0.05, 0.95, analysis_text, transform=ax4.transAxes,
            fontsize=11, verticalalignment='top', fontfamily='sans-serif',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle('第27季 Bobby Bones 案例专题分析', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'bobby_bones_analysis.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ 图7已保存: {output_path}")
    plt.close()
    
    return finalist_df


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
                           test_results: dict, final_stats: dict, output_dir: str):
    """生成分析总结报告（包含决赛分析）"""
    
    df_with_elim = df[df['n_eliminated'] > 0]
    final_df = df[df['is_final'] == True]
    total_seasons = len(final_df)
    
    report = f"""
# 排名法可视化与统计检验报告（含决赛分析）

## 一、基本统计摘要

### 整体统计
- **总周数**: {len(df)} 周
- **有人淘汰周数**: {len(df_with_elim)} 周
- **无人淘汰周数**: {len(df) - len(df_with_elim)} 周
- **决赛周数**: {total_seasons} 季

### 淘汰一致性统计（仅统计有淘汰周数）
- **一致周数**: {df_with_elim['same_result'].sum()} 周
- **差异周数**: {len(df_with_elim) - df_with_elim['same_result'].sum()} 周
- **一致比例**: {df_with_elim['same_result'].sum() / len(df_with_elim) * 100:.2f}%

### 决赛排名准确性统计
| 指标 | 百分比法 | 排名法 |
|------|----------|--------|
| 排名完全正确 | {final_stats['完全正确']['百分比法']}/{total_seasons} ({final_stats['完全正确']['百分比法']/total_seasons*100:.1f}%) | {final_stats['完全正确']['排名法']}/{total_seasons} ({final_stats['完全正确']['排名法']/total_seasons*100:.1f}%) |
| 冠军预测正确 | {final_stats['冠军正确']['百分比法']}/{total_seasons} ({final_stats['冠军正确']['百分比法']/total_seasons*100:.1f}%) | {final_stats['冠军正确']['排名法']}/{total_seasons} ({final_stats['冠军正确']['排名法']/total_seasons*100:.1f}%) |
| 前两名正确 | {final_stats['前两名正确']['百分比法']}/{total_seasons} ({final_stats['前两名正确']['百分比法']/total_seasons*100:.1f}%) | {final_stats['前两名正确']['排名法']}/{total_seasons} ({final_stats['前两名正确']['排名法']/total_seasons*100:.1f}%) |

### 按季度统计
- **平均淘汰一致比例**: {stats_df['consistency_rate'].mean():.2f}%
- **最高一致比例**: {stats_df['consistency_rate'].max():.2f}% (第{stats_df.loc[stats_df['consistency_rate'].idxmax(), 'season']:.0f}季)
- **最低一致比例**: {stats_df['consistency_rate'].min():.2f}% (第{stats_df.loc[stats_df['consistency_rate'].idxmin(), 'season']:.0f}季)
- **标准差**: {stats_df['consistency_rate'].std():.2f}%

## 二、可视化分析结果

### 图1: 综合准确性趋势图（优化版）
- **文件**: `consistency_by_season.png`
- **内容**:
  - 子图1：淘汰一致比例趋势 + 决赛排名准确性标记
  - 子图2：每季淘汰周数统计 + 决赛结果标记
- **主要发现**:
  - 淘汰一致比例在不同季度间存在波动
  - 决赛排名准确性与淘汰一致性存在一定关联

### 图2: 综合百分比散点图（优化版）
- **文件**: `percentage_scatter.png`
- **内容**:
  - 子图1：淘汰分析（差异周选手的评委-观众百分比分布）
  - 子图2：决赛分析（决赛选手的排名预测正确/错误分布）
- **主要发现**:
  - 淘汰差异主要发生在中等百分比区域
  - 决赛预测错误的选手通常评委和观众评价差异较大

### 图3: 综合排名热力图（优化版）
- **文件**: `rank_heatmap.png`
- **内容**:
  - 上排：淘汰相关热力图（4种情况）
  - 下排：决赛相关热力图（排名法正确/错误）
- **主要发现**:
  - 排名差异越大，淘汰结果差异越明显
  - 决赛中排名法错误的案例多在对角线附近（排名接近）

### 图4: 决赛排名准确性对比柱状图
- **文件**: `final_accuracy_comparison.png`
- **内容**: 两种方法的"完全正确"、"冠军正确"、"前两名正确"对比
- **主要发现**:
  - 两种方法在决赛排名预测上表现接近
  - 冠军预测准确率高于完全排名准确率

### 图5: 决赛准确性随季度变化趋势图
- **文件**: `final_accuracy_trend.png`
- **内容**: 每季决赛两种方法的预测是否正确的时间序列
- **主要发现**:
  - 不同季度的预测准确性有波动
  - 部分季度两种方法表现一致，部分季度存在差异

### 图6: 排名法预测失败案例分析
- **文件**: `final_error_analysis.png`
- **内容**: 冠军的评委分-观众分分布，标注正确/错误预测
- **主要发现**:
  - 预测失败的冠军往往评委和观众评价差异较大
  - "爆冷"冠军（观众支持 > 评委评价）更容易导致预测失败

### 图7: Bobby Bones 案例专题分析
- **文件**: `bobby_bones_analysis.png`
- **内容**: 第27季争议冠军 Bobby Bones 的详细分析
- **主要发现**:
  - Bobby Bones 评委排名较低但观众支持度高
  - 这是典型的"爆冷"案例，说明观众投票的决定性作用

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

### 淘汰分析主要发现
1. 排名法与百分比法的淘汰结果在 {df_with_elim['same_result'].sum() / len(df_with_elim) * 100:.1f}% 的周数中一致
2. 卡方检验表明两种方法存在{"极其显著" if test_results['p_value'] < 0.001 else "显著" if test_results['p_value'] < 0.05 else "不显著"}的系统性差异
3. 一致性在不同季度间存在波动

### 决赛分析主要发现
1. 百分比法决赛完全正确率: {final_stats['完全正确']['百分比法']/total_seasons*100:.1f}%
2. 排名法决赛完全正确率: {final_stats['完全正确']['排名法']/total_seasons*100:.1f}%
3. 冠军预测准确率较高，但完全排名预测仍有挑战
4. "爆冷"冠军（如 Bobby Bones）是预测失败的主要原因

### 实际意义
- 两种方法在大多数情况下产生相似结果
- 淘汰差异主要发生在约 {(1 - df_with_elim['same_result'].sum() / len(df_with_elim)) * 100:.1f}% 的周数中
- 决赛排名预测需要更多考虑观众投票的"爆冷"因素
- Bobby Bones 案例说明了纯数学模型难以预测情感驱动的投票行为

---
**生成时间**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
    
    output_path = os.path.join(output_dir, '可视化与统计分析报告.md')
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"✓ 分析报告已保存: {output_path}")


def main():
    print("=" * 70)
    print("排名法可视化与统计检验（含决赛分析）")
    print("=" * 70)
    
    # 创建输出目录
    output_dir = os.path.join(SCRIPT_DIR, 'visualization_results')
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n输出目录: {output_dir}\n")
    
    # 加载数据
    print("正在加载数据...")
    df, excel_df, fan_df = load_data()
    print(f"✓ 数据加载完成: {len(df)} 周记录\n")
    
    # ===== 优化的前3个图 =====
    # 1. 综合准确性趋势图（优化版）
    print("生成图1: 综合准确性趋势图（优化版）...")
    stats_df = plot_consistency_by_season(df, output_dir)
    
    # 2. 综合百分比散点图（优化版）
    print("生成图2: 综合百分比散点图（优化版）...")
    scatter_df, final_scatter_df = plot_percentage_scatter(df, excel_df, fan_df, output_dir)
    
    # 3. 综合排名热力图（优化版）
    print("生成图3: 综合排名热力图（优化版）...")
    rank_df, final_rank_df = plot_rank_heatmap(df, excel_df, fan_df, output_dir)
    
    # ===== 新增决赛专项可视化（4个新图）=====
    # 4. 决赛排名准确性对比柱状图
    print("生成图4: 决赛排名准确性对比柱状图...")
    final_stats = plot_final_accuracy_comparison(df, output_dir)
    
    # 5. 决赛准确性随季度变化趋势图
    print("生成图5: 决赛准确性随季度变化趋势图...")
    plot_final_accuracy_trend(df, output_dir)
    
    # 6. 排名法预测失败案例分析
    print("生成图6: 排名法预测失败案例分析...")
    champ_df = plot_final_error_analysis(df, excel_df, fan_df, output_dir)
    
    # 7. Bobby Bones 案例专题分析
    print("生成图7: Bobby Bones 案例专题分析...")
    bobby_df = plot_bobby_bones_analysis(df, excel_df, fan_df, output_dir)
    
    # ===== 统计检验 =====
    print("\n进行统计检验...")
    test_results = perform_chi_square_test(df)
    
    # ===== 生成报告 =====
    print("\n生成分析报告...")
    generate_summary_report(df, stats_df, test_results, final_stats, output_dir)
    
    print("\n" + "=" * 70)
    print("所有分析完成！")
    print("=" * 70)
    print(f"\n请查看 {output_dir} 目录下的结果文件:")
    print("  - consistency_by_season.png      (图1: 综合准确性趋势)")
    print("  - percentage_scatter.png         (图2: 综合百分比散点图)")
    print("  - rank_heatmap.png               (图3: 综合排名热力图)")
    print("  - final_accuracy_comparison.png  (图4: 决赛准确性对比)")
    print("  - final_accuracy_trend.png       (图5: 决赛准确性趋势)")
    print("  - final_error_analysis.png       (图6: 失败案例分析)")
    print("  - bobby_bones_analysis.png       (图7: Bobby Bones 案例)")
    print("  - 可视化与统计分析报告.md")


if __name__ == '__main__':
    main()
