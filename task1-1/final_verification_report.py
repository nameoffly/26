"""
最终验证报告：决赛周识别和排名约束
"""
import pandas as pd
import sys
sys.stdout.reconfigure(encoding='utf-8') if hasattr(sys.stdout, 'reconfigure') else None

# 加载估计结果
results_df = pd.read_csv(r'd:\Users\13016\Desktop\26MCM\2026_C\task1-1\fan_vote_estimates_entropy_smooth_150.csv')
data_df = pd.read_excel(r'd:\Users\13016\Desktop\26MCM\2026_C\Data_4.xlsx')

print("="*80)
print("决赛周识别和排名约束验证报告")
print("="*80)

final_weeks = []
all_satisfied = True

for season in range(3, 28):
    season_orig = data_df[data_df['season'] == season]
    
    # 找出最大淘汰周数
    max_elim_week = 0
    for week in range(1, 12):
        elim = season_orig[season_orig['results'].str.contains(f'Eliminated Week {week}', na=False)]
        if len(elim) > 0:
            max_elim_week = week
    
    final_week = max_elim_week + 1
    
    # 找出决赛选手
    finalists = season_orig[
        season_orig['results'].str.contains('Place', na=False) & 
        ~season_orig['results'].str.contains('Eliminated', na=False)
    ]
    
    if len(finalists) < 2:
        continue
    
    # 获取决赛周的估计结果
    week_results = results_df[(results_df['season'] == season) & (results_df['week'] == final_week)]
    
    # 构建排名映射
    rankings = {}
    for _, row in finalists.iterrows():
        rankings[row['celebrity_name']] = row['placement']
    
    # 验证约束
    sorted_rankings = sorted(rankings.items(), key=lambda x: x[1])
    
    violations = 0
    prev_total = None
    min_margin = float('inf')
    
    for name, rank in sorted_rankings:
        contestant_row = week_results[week_results['celebrity_name'] == name]
        if len(contestant_row) > 0:
            total = contestant_row['total_percent'].values[0]
            
            if prev_total is not None:
                margin = prev_total - total
                if margin < 0:  # 违反约束
                    violations += 1
                if margin < min_margin:
                    min_margin = margin
            
            prev_total = total
    
    status = "✓" if violations == 0 else "✗"
    
    final_weeks.append({
        'season': season,
        'final_week': final_week,
        'n_finalists': len(finalists),
        'violations': violations,
        'min_margin': min_margin * 100 if min_margin != float('inf') else 0,
        'status': status
    })
    
    if violations > 0:
        all_satisfied = False
        print(f"Season {season:2d}: Week {final_week:2d}, {len(finalists)} 决赛选手, {violations} 违反 {status}")

print(f"\n{'='*80}")
print("汇总统计:")
print(f"{'='*80}")

fw_df = pd.DataFrame(final_weeks)
print(f"\n总季度数: {len(fw_df)}")
print(f"决赛周正确识别: {len(fw_df)} / {len(fw_df)}")
print(f"约束满足的季度: {(fw_df['violations'] == 0).sum()}")
print(f"约束违反的季度: {(fw_df['violations'] > 0).sum()}")
print(f"约束满足率: {(fw_df['violations'] == 0).sum() / len(fw_df) * 100:.2f}%")

print(f"\n决赛选手人数分布:")
print(fw_df['n_finalists'].value_counts().sort_index())

print(f"\n决赛周分布:")
print(fw_df['final_week'].value_counts().sort_index())

if all_satisfied:
    print(f"\n{'='*80}")
    print("✅ 所有季度的决赛排名约束都满足！")
    print(f"{'='*80}")
else:
    print(f"\n{'='*80}")
    print("⚠️ 存在约束违反的季度")
    print(f"{'='*80}")

# 统计最小边际
print(f"\n排名边际统计 (排名靠前选手总分 - 排名靠后选手总分):")
print(f"  平均边际: {fw_df['min_margin'].mean():.4f}%")
print(f"  最小边际: {fw_df['min_margin'].min():.4f}%")
print(f"  最大边际: {fw_df['min_margin'].max():.4f}%")
