"""
票分融合方式分析 - 可视化模块

生成图表：
1. 一致性趋势图：各季两种方法的整体一致性
2. 决赛排名准确性对比图：所有34季的决赛排名准确性
3. 偏向性分析图：差异周的评委-观众百分比分布
4. 争议选手轨迹图：各争议选手逐周排名变化
5. 四规则对比柱状图：每位争议选手在四种规则下的生存周数对比
6. 多维度评估雷达图：四种规则的多维度评估
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

# Set font
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(SCRIPT_DIR, 'combined_contestant_info.csv')
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'results', 'figures')

# Controversial contestants with actual method used
CONTROVERSIAL_CONTESTANTS = [
    {'season': 2, 'name': 'Jerry Rice', 'actual_placement': 2, 'actual_method': 'rank'},
    {'season': 4, 'name': 'Billy Ray Cyrus', 'actual_placement': 5, 'actual_method': 'percent'},
    {'season': 11, 'name': 'Bristol Palin', 'actual_placement': 3, 'actual_method': 'percent'},
    {'season': 27, 'name': 'Bobby Bones', 'actual_placement': 1, 'actual_method': 'percent'},
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
    
    # Main line
    ax.plot(seasons, consistency, marker='o', linewidth=2, markersize=8, 
            color='#2E86AB', label='Consistency Rate')
    
    # Trend line
    z = np.polyfit(seasons, consistency, 2)
    p = np.poly1d(z)
    ax.plot(seasons, p(seasons), '--', color='#A23B72', alpha=0.7, 
            linewidth=2, label='Quadratic Trend')
    
    # Mean line
    mean_rate = np.mean(consistency)
    ax.axhline(y=mean_rate, color='#F18F01', linestyle='--', linewidth=2, 
               alpha=0.7, label=f'Mean ({mean_rate:.1f}%)')
    
    # Mark controversial seasons
    controversial_seasons = [2, 4, 11, 27]
    for s in controversial_seasons:
        if s in seasons:
            idx = list(seasons).index(s)
            ax.scatter([s], [consistency[idx]], s=200, c='red', marker='*', 
                      zorder=5, edgecolors='black', linewidth=1)
    
    ax.annotate('* Controversial Contestant Season', xy=(0.02, 0.02), xycoords='axes fraction',
                fontsize=10, color='red')
    
    ax.set_xlabel('Season', fontsize=12, fontweight='bold')
    ax.set_ylabel('Consistency Rate (%)', fontsize=12, fontweight='bold')
    ax.set_title('Rank Method vs Percent Method: Result Consistency Trend', 
                fontsize=14, fontweight='bold', pad=15)
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim(0, 35)
    ax.set_ylim(0, 110)
    ax.set_xticks(range(1, 35, 2))
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'consistency_trend.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"[OK] Chart 1 saved: {output_path}")
    plt.close()


def plot_bias_analysis(detail_df, df, output_dir):
    """
    Chart 2: Bias Analysis
    Scatter plot of judge-fan percentage distribution in difference weeks
    """
    # Filter weeks with different results and eliminations
    diff_weeks = detail_df[(detail_df['elim_same'] == False) & (detail_df['n_eliminated'] > 0)]
    
    if len(diff_weeks) == 0:
        print("Warning: No weeks with different elimination results found")
        return
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Collect data for contestants eliminated differently
    scatter_data = []
    
    for _, row in diff_weeks.iterrows():
        season, week = row['season'], row['week']
        
        # Parse elimination lists
        only_percent = eval(row['only_percent_eliminated']) if pd.notna(row.get('only_percent_eliminated')) and row['only_percent_eliminated'] else []
        only_rank = eval(row['only_rank_eliminated']) if pd.notna(row.get('only_rank_eliminated')) and row['only_rank_eliminated'] else []
        
        # Get week data
        week_df = df[(df['season'] == season) & (df['week'] == week)]
        
        for _, contestant in week_df.iterrows():
            name = contestant['name']
            
            if name in only_percent:
                category = 'Only Percent Method Eliminated'
            elif name in only_rank:
                category = 'Only Rank Method Eliminated'
            else:
                category = 'Same Result'
            
            scatter_data.append({
                'judge_percent': contestant['judge_percent'],
                'fan_percent': contestant['fan_percent'],
                'category': category,
                'season': season,
                'week': week,
                'name': name
            })
    
    scatter_df = pd.DataFrame(scatter_data)
    
    # Plot scatter
    colors = {
        'Only Percent Method Eliminated': '#F18F01',
        'Only Rank Method Eliminated': '#2E86AB',
        'Same Result': '#AAAAAA'
    }
    
    markers = {
        'Only Percent Method Eliminated': '^',
        'Only Rank Method Eliminated': 'v',
        'Same Result': 'o'
    }
    
    for category in ['Same Result', 'Only Rank Method Eliminated', 'Only Percent Method Eliminated']:
        data = scatter_df[scatter_df['category'] == category]
        if len(data) > 0:
            ax.scatter(data['judge_percent'], data['fan_percent'],
                      c=colors[category], marker=markers[category],
                      s=80 if category == 'Same Result' else 120,
                      alpha=0.5 if category == 'Same Result' else 0.8,
                      label=f'{category} (n={len(data)})',
                      edgecolors='black', linewidth=0.5)
    
    # Diagonal line
    max_val = max(scatter_df['judge_percent'].max(), scatter_df['fan_percent'].max())
    ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.3, linewidth=2, label='Judge = Fan')
    
    ax.set_xlabel('Judge Percentage', fontsize=12, fontweight='bold')
    ax.set_ylabel('Fan Percentage', fontsize=12, fontweight='bold')
    ax.set_title('Bias Analysis: Judge-Fan Distribution in Difference Weeks', fontsize=14, fontweight='bold', pad=15)
    ax.legend(fontsize=10, loc='upper left')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'bias_analysis.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"[OK] Chart 2 saved: {output_path}")
    plt.close()


def plot_controversial_trajectory(df, output_dir):
    """
    Chart 3: Controversial Contestant Trajectory
    Weekly judge rank vs fan rank changes
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    axes = axes.flatten()
    
    for idx, contestant in enumerate(CONTROVERSIAL_CONTESTANTS):
        ax = axes[idx]
        season = contestant['season']
        name = contestant['name']
        
        # Get contestant data
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
            
            # Calculate rank (1=best)
            judge_rank = (week_df['judge_percent'] > row['judge_percent']).sum() + 1
            fan_rank = (week_df['fan_percent'] > row['fan_percent']).sum() + 1
            
            weeks.append(week)
            judge_ranks.append(judge_rank)
            fan_ranks.append(fan_rank)
            n_contestants_list.append(n_contestants)
        
        # Plot ranking changes
        ax.plot(weeks, judge_ranks, marker='s', linewidth=2, markersize=10,
                color='#D62246', label='Judge Rank')
        ax.plot(weeks, fan_ranks, marker='o', linewidth=2, markersize=10,
                color='#06A77D', label='Fan Rank')
        
        # Add contestant count annotation
        for i, (w, n) in enumerate(zip(weeks, n_contestants_list)):
            ax.annotate(f'({n})', xy=(w, max(judge_ranks[i], fan_ranks[i]) + 0.3),
                       fontsize=8, ha='center', alpha=0.7)
        
        ax.set_xlabel('Week', fontsize=11, fontweight='bold')
        ax.set_ylabel('Rank (1=Best)', fontsize=11, fontweight='bold')
        ax.set_title(f'S{season} {name}\nActual Placement: #{contestant["actual_placement"]}',
                    fontsize=12, fontweight='bold', pad=10)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.invert_yaxis()  # Invert Y axis (lower rank = better)
        ax.set_xticks(weeks)
    
    plt.suptitle('Controversial Contestants: Weekly Ranking Trajectory', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'controversial_trajectory.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"[OK] Chart 3 saved: {output_path}")
    plt.close()


def plot_four_rules_comparison(four_rules_df, output_dir):
    """
    Chart 4: Four Rules Comparison
    Simulated placement under four rules for each controversial contestant
    Note: For seasons that actually used a specific method, simulated result should match actual
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    contestants = four_rules_df['name'].unique()
    rules = ['R1: Pure Percent', 'R2: Pure Rank', 'R3: Percent+Judge', 'R4: Rank+Judge']
    colors = ['#2E86AB', '#FF6B35', '#06A77D', '#A23B72']
    
    for idx, name in enumerate(contestants):
        ax = axes[idx]
        name_df = four_rules_df[four_rules_df['name'] == name]
        
        actual = name_df['actual_placement'].iloc[0]
        season = name_df['season'].iloc[0]
        
        # Get contestant info for actual method
        contestant_info = next((c for c in CONTROVERSIAL_CONTESTANTS if c['name'] == name), None)
        actual_method = contestant_info['actual_method'] if contestant_info else 'percent'
        
        x = np.arange(len(rules))
        placements = []
        
        # Read placements directly from CSV (now the simulation code handles actual method correctly)
        for rule_cn in ['R1: 纯百分比法', 'R2: 纯排名法', 'R3: 百分比+评委决定', 'R4: 排名+评委决定']:
            rule_row = name_df[name_df['rule'] == rule_cn]
            if len(rule_row) > 0:
                p = rule_row['simulated_placement'].iloc[0]
                placements.append(p if pd.notna(p) else 0)
            else:
                placements.append(0)
        
        bars = ax.bar(x, placements, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
        
        # Actual placement reference line
        ax.axhline(y=actual, color='red', linestyle='--', linewidth=2, 
                  label=f'Actual: #{actual}')
        
        # Add value labels
        for bar, val in zip(bars, placements):
            if val > 0:
                ax.annotate(f'#{int(val)}',
                           xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                           ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        ax.set_xlabel('Rule', fontsize=11, fontweight='bold')
        ax.set_ylabel('Simulated Placement', fontsize=11, fontweight='bold')
        
        # Add note about actual method used
        method_note = "(Rank Method)" if actual_method == 'rank' else "(Percent Method)"
        ax.set_title(f'S{int(season)} {name} {method_note}', fontsize=12, fontweight='bold', pad=10)
        ax.set_xticks(x)
        ax.set_xticklabels(['R1\nPure\nPercent', 'R2\nPure\nRank', 'R3\nPercent\n+Judge', 'R4\nRank\n+Judge'],
                          fontsize=9)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, linestyle='--', axis='y')
        ax.set_ylim(0, max(placements) + 2 if placements else 10)
    
    plt.suptitle('Four Rules Simulation: Controversial Contestants Placement Comparison', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'four_rules_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"[OK] Chart 4 saved: {output_path}")
    plt.close()


def plot_survival_comparison(four_rules_df, output_dir):
    """
    Chart 5: Survival Comparison
    Survival weeks under different rules
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    
    contestants = four_rules_df['name'].unique()
    rules_cn = ['R1: 纯百分比法', 'R2: 纯排名法', 'R3: 百分比+评委决定', 'R4: 排名+评委决定']
    rules_en = ['Pure Percent', 'Pure Rank', 'Percent+Judge', 'Rank+Judge']
    
    x = np.arange(len(contestants))
    width = 0.2
    colors = ['#2E86AB', '#FF6B35', '#06A77D', '#A23B72']
    
    for i, (rule_cn, rule_en) in enumerate(zip(rules_cn, rules_en)):
        survival_weeks = []
        for name in contestants:
            name_df = four_rules_df[(four_rules_df['name'] == name) & (four_rules_df['rule'] == rule_cn)]
            if len(name_df) > 0:
                survival_weeks.append(name_df['survival_weeks'].iloc[0])
            else:
                survival_weeks.append(0)
        
        bars = ax.bar(x + i * width, survival_weeks, width, label=rule_en, 
                     color=colors[i], alpha=0.8, edgecolor='black', linewidth=0.5)
        
        # Add value labels
        for bar, val in zip(bars, survival_weeks):
            ax.annotate(f'{int(val)} weeks',
                       xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                       ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax.set_xlabel('Controversial Contestant', fontsize=12, fontweight='bold')
    ax.set_ylabel('Survival Weeks', fontsize=12, fontweight='bold')
    ax.set_title('Four Rules: Controversial Contestants Survival Weeks Comparison', fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(x + width * 1.5)
    
    # Add season info
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
    print(f"[OK] Chart 5 saved: {output_path}")
    plt.close()


def plot_radar_evaluation(output_dir):
    """
    Chart 6: Radar Evaluation
    Multi-dimensional evaluation of four rules
    """
    # Evaluation dimensions
    categories = ['Controversy\nModeration', 'Audience\nEngagement', 'Dramatic\nEffect', 
                  'Professionalism', 'Result\nControllability']
    
    # Scores for four rules (based on analysis, max 10)
    scores = {
        'Pure Percent': [8, 9, 9, 4, 5],
        'Pure Rank': [6, 7, 6, 6, 6],
        'Percent+Judge': [5, 6, 5, 8, 8],
        'Rank+Judge': [3, 4, 3, 9, 9],
    }
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    
    # Calculate angles
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]  # Close the polygon
    
    colors = ['#2E86AB', '#FF6B35', '#06A77D', '#A23B72']
    
    for i, (rule, values) in enumerate(scores.items()):
        values = values + values[:1]  # Close the polygon
        ax.plot(angles, values, 'o-', linewidth=2, label=rule, color=colors[i])
        ax.fill(angles, values, alpha=0.1, color=colors[i])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=11, fontweight='bold')
    ax.set_ylim(0, 10)
    ax.set_yticks([2, 4, 6, 8, 10])
    ax.set_yticklabels(['2', '4', '6', '8', '10'], fontsize=9)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)
    
    plt.title('Four Rules Multi-Dimensional Evaluation', 
             fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'radar_evaluation.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"[OK] Chart 6 saved: {output_path}")
    plt.close()


def main():
    print("Loading data...")
    df = load_data()
    season_stats, four_rules, detail = load_analysis_results()
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("\nGenerating visualization charts...\n")
    
    # Generate charts
    plot_consistency_trend(season_stats, OUTPUT_DIR)
    plot_bias_analysis(detail, df, OUTPUT_DIR)
    plot_controversial_trajectory(df, OUTPUT_DIR)
    plot_four_rules_comparison(four_rules, OUTPUT_DIR)
    plot_survival_comparison(four_rules, OUTPUT_DIR)
    plot_radar_evaluation(OUTPUT_DIR)
    
    print(f"\nAll charts saved to: {OUTPUT_DIR}")
    print("\nGenerated charts:")
    print("  1. consistency_trend.png - Consistency Trend")
    print("  2. bias_analysis.png - Bias Analysis")
    print("  3. controversial_trajectory.png - Controversial Contestant Trajectory")
    print("  4. four_rules_comparison.png - Four Rules Placement Comparison")
    print("  5. survival_comparison.png - Survival Weeks Comparison")
    print("  6. radar_evaluation.png - Multi-Dimensional Evaluation Radar")


if __name__ == '__main__':
    main()
