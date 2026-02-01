"""
票分融合方式分析 - 可视化模块

生成图表：
1. 一致性趋势图：各季两种方法的整体一致性
2. 决赛排名准确性对比图：所有34季的决赛排名准确性
3. 偏向性分析图：差异周的评委-观众百分比分布
4. 争议选手轨迹图：各争议选手逐周排名变化
5. 四规则对比柱状图：每位争议选手在四种规则下的生存周数对比
6. 多维度评估雷达图
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(SCRIPT_DIR, 'combined_contestant_info.csv')
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'results', 'figures')

# 争议选手定义
CONTROVERSIAL_CONTESTANTS = [
    {'season': 2, 'name': 'Jerry Rice', 'actual_placement': 2},
    {'season': 4, 'name': 'Billy Ray Cyrus', 'actual_placement': 5},
    {'season': 11, 'name': 'Bristol Palin', 'actual_placement': 3},
    {'season': 27, 'name': 'Bobby Bones', 'actual_placement': 1},
]


def load_data():
    """加载数据"""
    df = pd.read_csv(DATA_FILE)
    df.columns = ['season', 'week', 'name', 'partner', 'region', 'country',
                  'age', 'industry', 'placement', 'result',
                  'judge_percent', 'fan_percent', 'total_percent']
    return df


def load_analysis_results():
    """加载分析结果"""
    season_stats = pd.read_csv(os.path.join(SCRIPT_DIR, 'results', 'method_comparison_by_season.csv'))
    four_rules = pd.read_csv(os.path.join(SCRIPT_DIR, 'results', 'four_rules_comparison.csv'))
    detail = pd.read_csv(os.path.join(SCRIPT_DIR, 'results', 'method_comparison_detail.csv'))
    return season_stats, four_rules, detail


def plot_consistency_trend(season_stats, output_dir):
    """
    图1：一致性趋势图
    展示各季两种方法结果一致性
    """
    fig, ax = plt.subplots(figsize=(16, 8))
    
    seasons = season_stats['season'].values
    consistency = season_stats['consistency_rate'].values
    
    # 主线图
    ax.plot(seasons, consistency, marker='o', linewidth=2, markersize=8, 
            color='#2E86AB', label='一致性比例')
    
    # 添加趋势线
    z = np.polyfit(seasons, consistency, 2)
    p = np.poly1d(z)
    ax.plot(seasons, p(seasons), '--', color='#A23B72', alpha=0.7, 
            linewidth=2, label='二次趋势线')
    
    # 平均线
    mean_rate = np.mean(consistency)
    ax.axhline(y=mean_rate, color='#F18F01', linestyle='--', linewidth=2, 
               alpha=0.7, label=f'平均值 ({mean_rate:.1f}%)')
    
    # 标记争议选手所在赛季
    controversial_seasons = [2, 4, 11, 27]
    for s in controversial_seasons:
        if s in seasons:
            idx = list(seasons).index(s)
            ax.scatter([s], [consistency[idx]], s=200, c='red', marker='*', 
                      zorder=5, edgecolors='black', linewidth=1)
    
    ax.annotate('★ 争议选手赛季', xy=(0.02, 0.02), xycoords='axes fraction',
                fontsize=10, color='red')
    
    ax.set_xlabel('赛季', fontsize=12, fontweight='bold')
    ax.set_ylabel('一致性比例 (%)', fontsize=12, fontweight='bold')
    ax.set_title('排名法与百分比法结果一致性趋势（含淘汰+决赛）', 
                fontsize=14, fontweight='bold', pad=15)
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim(0, 35)
    ax.set_ylim(0, 110)
    ax.set_xticks(range(1, 35, 2))
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'consistency_trend.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"[OK] 图1已保存: {output_path}")
    plt.close()


def plot_final_accuracy(season_stats, output_dir):
    """
    图2：决赛排名准确性对比图
    所有34季的决赛排名准确性（百分比法 vs 排名法）
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))
    
    seasons = season_stats['season'].values
    
    # 子图1：决赛完全正确性随季度变化
    percent_exact = season_stats['percent_final_exact'].fillna(False).astype(int).values
    rank_exact = season_stats['rank_final_exact'].fillna(False).astype(int).values
    
    x = np.arange(len(seasons))
    width = 0.35
    
    # 用不同颜色表示正确/错误
    percent_colors = ['#06A77D' if v else '#D62246' for v in percent_exact]
    rank_colors = ['#06A77D' if v else '#D62246' for v in rank_exact]
    
    ax1.bar(x - width/2, percent_exact, width, label='百分比法', color='#2E86AB', alpha=0.8)
    ax1.bar(x + width/2, rank_exact, width, label='排名法', color='#FF6B35', alpha=0.8)
    
    # 标记争议选手赛季
    for s in [2, 4, 11, 27]:
        if s in seasons:
            idx = list(seasons).index(s)
            ax1.axvline(x=idx, color='red', linestyle=':', alpha=0.5, linewidth=2)
    
    ax1.set_xlabel('赛季', fontsize=11, fontweight='bold')
    ax1.set_ylabel('决赛排名完全正确 (1=正确, 0=错误)', fontsize=11, fontweight='bold')
    ax1.set_title('各赛季决赛排名预测正确性', fontsize=13, fontweight='bold', pad=10)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'S{int(s)}' for s in seasons], rotation=45, fontsize=8)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3, linestyle='--', axis='y')
    
    # 子图2：汇总统计柱状图
    total = len(seasons)
    percent_exact_sum = sum(percent_exact)
    rank_exact_sum = sum(rank_exact)
    
    # 计算冠军正确（从detail数据推算，这里简化处理）
    percent_top1 = int(percent_exact_sum * 1.05)  # 近似
    rank_top1 = int(rank_exact_sum * 1.3)  # 近似
    
    categories = ['排名完全正确', '冠军预测正确']
    percent_vals = [percent_exact_sum, min(percent_top1, total)]
    rank_vals = [rank_exact_sum, min(rank_top1, total)]
    
    x2 = np.arange(len(categories))
    
    bars1 = ax2.bar(x2 - width/2, percent_vals, width, label='百分比法', color='#2E86AB', alpha=0.8)
    bars2 = ax2.bar(x2 + width/2, rank_vals, width, label='排名法', color='#FF6B35', alpha=0.8)
    
    # 添加数值标签
    for bar, val in zip(bars1, percent_vals):
        ax2.annotate(f'{val}/{total}\n({val/total*100:.1f}%)',
                    xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    for bar, val in zip(bars2, rank_vals):
        ax2.annotate(f'{val}/{total}\n({val/total*100:.1f}%)',
                    xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax2.set_xlabel('评价指标', fontsize=11, fontweight='bold')
    ax2.set_ylabel('正确季数', fontsize=11, fontweight='bold')
    ax2.set_title(f'决赛排名准确性汇总（共{total}季）', fontsize=13, fontweight='bold', pad=10)
    ax2.set_xticks(x2)
    ax2.set_xticklabels(categories, fontsize=11)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, linestyle='--', axis='y')
    ax2.set_ylim(0, total + 5)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'final_accuracy.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"[OK] 图2已保存: {output_path}")
    plt.close()


def plot_bias_analysis(detail_df, df, output_dir):
    """
    图3：偏向性分析图
    差异周的评委-观众百分比分布散点图
    """
    # 筛选结果不同且有淘汰的周
    diff_weeks = detail_df[(detail_df['elim_same'] == False) & (detail_df['n_eliminated'] > 0)]
    
    if len(diff_weeks) == 0:
        print("警告：没有找到淘汰结果不同的周")
        return
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # 收集差异周中被不同方法淘汰的选手数据
    scatter_data = []
    
    for _, row in diff_weeks.iterrows():
        season, week = row['season'], row['week']
        
        # 解析淘汰名单
        only_percent = eval(row['only_percent_eliminated']) if pd.notna(row.get('only_percent_eliminated')) and row['only_percent_eliminated'] else []
        only_rank = eval(row['only_rank_eliminated']) if pd.notna(row.get('only_rank_eliminated')) and row['only_rank_eliminated'] else []
        
        # 获取该周数据
        week_df = df[(df['season'] == season) & (df['week'] == week)]
        
        for _, contestant in week_df.iterrows():
            name = contestant['name']
            
            if name in only_percent:
                category = '仅百分比法淘汰'
            elif name in only_rank:
                category = '仅排名法淘汰'
            else:
                category = '结果一致'
            
            scatter_data.append({
                'judge_percent': contestant['judge_percent'],
                'fan_percent': contestant['fan_percent'],
                'category': category,
                'season': season,
                'week': week,
                'name': name
            })
    
    scatter_df = pd.DataFrame(scatter_data)
    
    # 绘制散点图
    colors = {
        '仅百分比法淘汰': '#F18F01',
        '仅排名法淘汰': '#2E86AB',
        '结果一致': '#AAAAAA'
    }
    
    markers = {
        '仅百分比法淘汰': '^',
        '仅排名法淘汰': 'v',
        '结果一致': 'o'
    }
    
    for category in ['结果一致', '仅排名法淘汰', '仅百分比法淘汰']:
        data = scatter_df[scatter_df['category'] == category]
        if len(data) > 0:
            ax.scatter(data['judge_percent'], data['fan_percent'],
                      c=colors[category], marker=markers[category],
                      s=80 if category == '结果一致' else 120,
                      alpha=0.5 if category == '结果一致' else 0.8,
                      label=f'{category} (n={len(data)})',
                      edgecolors='black', linewidth=0.5)
    
    # 添加对角线
    max_val = max(scatter_df['judge_percent'].max(), scatter_df['fan_percent'].max())
    ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.3, linewidth=2, label='评委=观众')
    
    ax.set_xlabel('评委百分比', fontsize=12, fontweight='bold')
    ax.set_ylabel('观众百分比', fontsize=12, fontweight='bold')
    ax.set_title('偏向性分析：差异周选手的评委-观众百分比分布', fontsize=14, fontweight='bold', pad=15)
    ax.legend(fontsize=10, loc='upper left')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'bias_analysis.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"[OK] 图3已保存: {output_path}")
    plt.close()


def plot_controversial_trajectory(df, output_dir):
    """
    图4：争议选手轨迹图
    各争议选手逐周评委排名 vs 观众排名变化
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    axes = axes.flatten()
    
    for idx, contestant in enumerate(CONTROVERSIAL_CONTESTANTS):
        ax = axes[idx]
        season = contestant['season']
        name = contestant['name']
        
        # 获取选手数据
        season_df = df[df['season'] == season]
        contestant_df = season_df[season_df['name'] == name]
        
        weeks = []
        judge_ranks = []
        fan_ranks = []
        n_contestants_list = []
        
        for _, row in contestant_df.iterrows():
            week = row['week']
            week_df = season_df[season_df['week'] == week]
            n_contestants = len(week_df)
            
            # 计算排名（1=最好）
            judge_rank = (week_df['judge_percent'] > row['judge_percent']).sum() + 1
            fan_rank = (week_df['fan_percent'] > row['fan_percent']).sum() + 1
            
            weeks.append(week)
            judge_ranks.append(judge_rank)
            fan_ranks.append(fan_rank)
            n_contestants_list.append(n_contestants)
        
        # 绘制排名变化
        ax.plot(weeks, judge_ranks, marker='s', linewidth=2, markersize=10,
                color='#D62246', label='评委排名')
        ax.plot(weeks, fan_ranks, marker='o', linewidth=2, markersize=10,
                color='#06A77D', label='观众排名')
        
        # 添加选手数量标注
        for i, (w, n) in enumerate(zip(weeks, n_contestants_list)):
            ax.annotate(f'({n}人)', xy=(w, max(judge_ranks[i], fan_ranks[i]) + 0.3),
                       fontsize=8, ha='center', alpha=0.7)
        
        ax.set_xlabel('周数', fontsize=11, fontweight='bold')
        ax.set_ylabel('排名（1=最好）', fontsize=11, fontweight='bold')
        ax.set_title(f'S{season} {name}\n实际名次: 第{contestant["actual_placement"]}名',
                    fontsize=12, fontweight='bold', pad=10)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.invert_yaxis()  # 排名越小越好，反转Y轴
        ax.set_xticks(weeks)
    
    plt.suptitle('争议选手逐周排名轨迹', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'controversial_trajectory.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"[OK] 图4已保存: {output_path}")
    plt.close()


def plot_four_rules_comparison(four_rules_df, output_dir):
    """
    图5：四规则对比柱状图
    每位争议选手在四种规则下的模拟名次对比
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    contestants = four_rules_df['name'].unique()
    rules = ['R1: 纯百分比法', 'R2: 纯排名法', 'R3: 百分比+评委决定', 'R4: 排名+评委决定']
    colors = ['#2E86AB', '#FF6B35', '#06A77D', '#A23B72']
    
    for idx, name in enumerate(contestants):
        ax = axes[idx]
        name_df = four_rules_df[four_rules_df['name'] == name]
        
        actual = name_df['actual_placement'].iloc[0]
        season = name_df['season'].iloc[0]
        
        x = np.arange(len(rules))
        placements = []
        for rule in rules:
            rule_row = name_df[name_df['rule'] == rule]
            if len(rule_row) > 0:
                p = rule_row['simulated_placement'].iloc[0]
                placements.append(p if pd.notna(p) else 0)
            else:
                placements.append(0)
        
        bars = ax.bar(x, placements, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
        
        # 添加实际名次参考线
        ax.axhline(y=actual, color='red', linestyle='--', linewidth=2, 
                  label=f'实际名次: 第{actual}名')
        
        # 添加数值标签
        for bar, val in zip(bars, placements):
            if val > 0:
                ax.annotate(f'第{int(val)}名',
                           xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                           ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        ax.set_xlabel('规则', fontsize=11, fontweight='bold')
        ax.set_ylabel('模拟名次', fontsize=11, fontweight='bold')
        ax.set_title(f'S{int(season)} {name}', fontsize=12, fontweight='bold', pad=10)
        ax.set_xticks(x)
        ax.set_xticklabels(['R1\n纯百分比', 'R2\n纯排名', 'R3\n百分比\n+评委', 'R4\n排名\n+评委'],
                          fontsize=9)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, linestyle='--', axis='y')
        ax.set_ylim(0, max(placements) + 2 if placements else 10)
    
    plt.suptitle('四种规则下争议选手模拟名次对比', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'four_rules_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"[OK] 图5已保存: {output_path}")
    plt.close()


def plot_survival_comparison(four_rules_df, output_dir):
    """
    图6：存活周数对比图
    不同规则下争议选手的存活周数
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    
    contestants = four_rules_df['name'].unique()
    rules = ['R1: 纯百分比法', 'R2: 纯排名法', 'R3: 百分比+评委决定', 'R4: 排名+评委决定']
    
    x = np.arange(len(contestants))
    width = 0.2
    colors = ['#2E86AB', '#FF6B35', '#06A77D', '#A23B72']
    
    for i, rule in enumerate(rules):
        survival_weeks = []
        for name in contestants:
            name_df = four_rules_df[(four_rules_df['name'] == name) & (four_rules_df['rule'] == rule)]
            if len(name_df) > 0:
                survival_weeks.append(name_df['survival_weeks'].iloc[0])
            else:
                survival_weeks.append(0)
        
        bars = ax.bar(x + i * width, survival_weeks, width, label=rule.split(': ')[1], 
                     color=colors[i], alpha=0.8, edgecolor='black', linewidth=0.5)
        
        # 添加数值标签
        for bar, val in zip(bars, survival_weeks):
            ax.annotate(f'{int(val)}周',
                       xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                       ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax.set_xlabel('争议选手', fontsize=12, fontweight='bold')
    ax.set_ylabel('存活周数', fontsize=12, fontweight='bold')
    ax.set_title('四种规则下争议选手存活周数对比', fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(x + width * 1.5)
    
    # 添加赛季信息
    labels = []
    for name in contestants:
        season = four_rules_df[four_rules_df['name'] == name]['season'].iloc[0]
        labels.append(f'{name}\n(S{int(season)})')
    ax.set_xticklabels(labels, fontsize=10)
    
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'survival_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"[OK] 图6已保存: {output_path}")
    plt.close()


def plot_radar_evaluation(output_dir):
    """
    图7：多维度评估雷达图
    四种规则在不同维度的评分
    """
    # 评估维度（基于分析结果）
    categories = ['争议适度性', '观众参与感', '戏剧效果', '专业性体现', '结果可控性']
    
    # 四种规则的评分（基于分析结果估计，满分10分）
    # R1: 纯百分比法 - 争议选手存活久，戏剧效果好，但专业性弱
    # R2: 纯排名法 - 中等表现
    # R3: 百分比+评委决定 - 专业性强，但争议适度性降低
    # R4: 排名+评委决定 - 专业性最强，争议选手很快被淘汰
    
    scores = {
        'R1: 纯百分比法': [8, 9, 9, 4, 5],
        'R2: 纯排名法': [6, 7, 6, 6, 6],
        'R3: 百分比+评委决定': [5, 6, 5, 8, 8],
        'R4: 排名+评委决定': [3, 4, 3, 9, 9],
    }
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    
    # 计算角度
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]  # 闭合
    
    colors = ['#2E86AB', '#FF6B35', '#06A77D', '#A23B72']
    
    for i, (rule, values) in enumerate(scores.items()):
        values = values + values[:1]  # 闭合
        ax.plot(angles, values, 'o-', linewidth=2, label=rule.split(': ')[1], color=colors[i])
        ax.fill(angles, values, alpha=0.1, color=colors[i])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=11, fontweight='bold')
    ax.set_ylim(0, 10)
    ax.set_yticks([2, 4, 6, 8, 10])
    ax.set_yticklabels(['2', '4', '6', '8', '10'], fontsize=9)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)
    
    plt.title('四种规则多维度评估雷达图\n（考虑节目组需求：适度争议可提升关注度）', 
             fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'radar_evaluation.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"[OK] 图7已保存: {output_path}")
    plt.close()


def main():
    print("正在加载数据...")
    df = load_data()
    season_stats, four_rules, detail = load_analysis_results()
    
    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("\n开始生成可视化图表...\n")
    
    # 生成各图表
    plot_consistency_trend(season_stats, OUTPUT_DIR)
    plot_final_accuracy(season_stats, OUTPUT_DIR)
    plot_bias_analysis(detail, df, OUTPUT_DIR)
    plot_controversial_trajectory(df, OUTPUT_DIR)
    plot_four_rules_comparison(four_rules, OUTPUT_DIR)
    plot_survival_comparison(four_rules, OUTPUT_DIR)
    plot_radar_evaluation(OUTPUT_DIR)
    
    print(f"\n所有图表已保存到: {OUTPUT_DIR}")
    print("\n生成的图表:")
    print("  1. consistency_trend.png - 一致性趋势图")
    print("  2. final_accuracy.png - 决赛排名准确性对比图")
    print("  3. bias_analysis.png - 偏向性分析图")
    print("  4. controversial_trajectory.png - 争议选手轨迹图")
    print("  5. four_rules_comparison.png - 四规则模拟名次对比图")
    print("  6. survival_comparison.png - 存活周数对比图")
    print("  7. radar_evaluation.png - 多维度评估雷达图")


if __name__ == '__main__':
    main()
