"""
票分融合方式对比分析 - 主分析脚本

任务一：对比分析排名法与百分比法在各季应用后的比赛结果
- 淘汰一致性分析
- 决赛排名准确性分析
- 偏向性分析：哪种方法更偏向观众投票

数据源：combined_contestant_info.csv（包含预测的粉丝投票百分比）
"""

import pandas as pd
import numpy as np
import os
import sys

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(SCRIPT_DIR, 'combined_contestant_info.csv')
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'results')


def load_data():
    """加载数据"""
    df = pd.read_csv(DATA_FILE)
    # 重命名列以便处理
    df.columns = ['season', 'week', 'name', 'partner', 'region', 'country', 
                  'age', 'industry', 'placement', 'result', 
                  'judge_percent', 'fan_percent', 'total_percent']
    return df


def get_elimination_week(result_str):
    """从结果字符串中提取淘汰周数"""
    result_str = str(result_str).lower()
    if 'eliminated week' in result_str:
        try:
            week = int(result_str.split('eliminated week')[1].strip().split()[0])
            return week
        except:
            return None
    return None


def get_season_max_week(season_df):
    """获取该季最大周数"""
    return season_df['week'].max()


def get_final_week(season_df):
    """获取决赛周（最大淘汰周+1或最后一周）"""
    max_elim_week = 0
    for _, row in season_df.drop_duplicates('name').iterrows():
        elim_week = get_elimination_week(row['result'])
        if elim_week:
            max_elim_week = max(max_elim_week, elim_week)
    
    max_week = get_season_max_week(season_df)
    final_week = max_elim_week + 1 if max_elim_week > 0 else max_week
    return min(final_week, max_week)


def apply_percent_method(week_df, n_eliminate):
    """
    百分比法淘汰
    综合得分 = 评委% + 观众%，淘汰得分最低的n_eliminate人
    返回：淘汰名单
    """
    if n_eliminate <= 0:
        return []
    
    # 按总百分比升序排序（最低的被淘汰）
    sorted_df = week_df.sort_values('total_percent', ascending=True)
    eliminated = sorted_df.head(n_eliminate)['name'].tolist()
    return eliminated


def apply_rank_method(week_df, n_eliminate):
    """
    排名法淘汰
    综合排名 = 评委排名 + 观众排名（1=最好）
    淘汰综合排名最大的n_eliminate人
    同分时评委排名差的优先淘汰
    返回：淘汰名单
    """
    if n_eliminate <= 0:
        return []
    
    df = week_df.copy()
    # 评委排名：百分比越高排名越小（越好）
    df['judge_rank'] = df['judge_percent'].rank(ascending=False, method='min').astype(int)
    # 观众排名：百分比越高排名越小（越好）
    df['fan_rank'] = df['fan_percent'].rank(ascending=False, method='min').astype(int)
    # 综合排名
    df['sum_rank'] = df['judge_rank'] + df['fan_rank']
    
    # 按综合排名降序（越大越差），同分时评委排名大的（更差）优先
    df = df.sort_values(['sum_rank', 'judge_rank'], ascending=[False, False])
    eliminated = df.head(n_eliminate)['name'].tolist()
    return eliminated


def compute_final_ranking_percent(week_df):
    """百分比法计算决赛排名（总分越高排名越前）"""
    df = week_df.copy()
    df = df.sort_values('total_percent', ascending=False)
    rankings = {row['name']: rank + 1 for rank, (_, row) in enumerate(df.iterrows())}
    return rankings


def compute_final_ranking_rank_method(week_df):
    """排名法计算决赛排名（综合排名越小越好）"""
    df = week_df.copy()
    df['judge_rank'] = df['judge_percent'].rank(ascending=False, method='min').astype(int)
    df['fan_rank'] = df['fan_percent'].rank(ascending=False, method='min').astype(int)
    df['sum_rank'] = df['judge_rank'] + df['fan_rank']
    
    # 按综合排名升序（越小越好），同分时评委排名小的在前
    df = df.sort_values(['sum_rank', 'judge_rank'], ascending=[True, True])
    rankings = {row['name']: rank + 1 for rank, (_, row) in enumerate(df.iterrows())}
    return rankings


def get_actual_rankings(week_df):
    """获取实际决赛排名"""
    rankings = {}
    for _, row in week_df.iterrows():
        result = str(row['result']).lower()
        if 'place' in result and 'eliminated' not in result:
            rankings[row['name']] = int(row['placement'])
    return rankings


def compare_rankings(actual, predicted):
    """比较排名准确性"""
    if not actual or not predicted:
        return {'exact_match': False, 'top1_correct': False, 'top2_correct': False}
    
    common = set(actual.keys()) & set(predicted.keys())
    if len(common) < 2:
        return {'exact_match': False, 'top1_correct': False, 'top2_correct': False}
    
    actual_sorted = sorted(common, key=lambda x: actual[x])
    pred_sorted = sorted(common, key=lambda x: predicted[x])
    
    exact_match = actual_sorted == pred_sorted
    top1_correct = actual_sorted[0] == pred_sorted[0] if len(actual_sorted) > 0 else False
    top2_correct = actual_sorted[:2] == pred_sorted[:2] if len(actual_sorted) >= 2 else False
    
    return {
        'exact_match': exact_match,
        'top1_correct': top1_correct,
        'top2_correct': top2_correct
    }


def analyze_season(season_df, season_num):
    """分析单个赛季"""
    results = []
    max_week = get_season_max_week(season_df)
    final_week = get_final_week(season_df)
    
    for week in range(1, max_week + 1):
        week_df = season_df[season_df['week'] == week].copy()
        if week_df.empty:
            continue
        
        # 获取本周实际淘汰的人
        actual_eliminated = []
        for _, row in week_df.iterrows():
            elim_week = get_elimination_week(row['result'])
            if elim_week == week:
                actual_eliminated.append(row['name'])
        
        n_eliminate = len(actual_eliminated)
        
        # 应用两种方法
        percent_eliminated = apply_percent_method(week_df, n_eliminate)
        rank_eliminated = apply_rank_method(week_df, n_eliminate)
        
        # 判断淘汰结果是否一致
        elim_same = set(percent_eliminated) == set(rank_eliminated)
        
        # 决赛排名分析
        is_final = (week == final_week)
        final_info = {}
        
        if is_final:
            actual_rankings = get_actual_rankings(week_df)
            if actual_rankings:
                percent_rankings = compute_final_ranking_percent(week_df)
                rank_rankings = compute_final_ranking_rank_method(week_df)
                
                percent_accuracy = compare_rankings(actual_rankings, percent_rankings)
                rank_accuracy = compare_rankings(actual_rankings, rank_rankings)
                
                final_info = {
                    'actual_rankings': actual_rankings,
                    'percent_rankings': percent_rankings,
                    'rank_rankings': rank_rankings,
                    'percent_exact': percent_accuracy['exact_match'],
                    'percent_top1': percent_accuracy['top1_correct'],
                    'rank_exact': rank_accuracy['exact_match'],
                    'rank_top1': rank_accuracy['top1_correct'],
                }
        
        # 偏向性分析（当淘汰结果不同时）
        bias_info = {}
        if not elim_same and n_eliminate > 0:
            only_percent = set(percent_eliminated) - set(rank_eliminated)
            only_rank = set(rank_eliminated) - set(percent_eliminated)
            
            # 分析被淘汰者的观众投票占比
            mean_fan = week_df['fan_percent'].mean()
            
            # 仅百分比法淘汰的人，如果观众投票高于平均，说明排名法保留了高人气选手
            percent_favor_audience = 0
            rank_favor_audience = 0
            
            for name in only_percent:
                fan_pct = week_df[week_df['name'] == name]['fan_percent'].values[0]
                if fan_pct > mean_fan:
                    rank_favor_audience += 1  # 排名法保留了高人气选手
            
            for name in only_rank:
                fan_pct = week_df[week_df['name'] == name]['fan_percent'].values[0]
                if fan_pct < mean_fan:
                    rank_favor_audience += 1  # 排名法淘汰了低人气选手
            
            bias_info = {
                'only_percent_eliminated': list(only_percent),
                'only_rank_eliminated': list(only_rank),
                'rank_favor_audience': rank_favor_audience,
                'percent_favor_audience': len(only_percent) + len(only_rank) - rank_favor_audience
            }
        
        results.append({
            'season': season_num,
            'week': week,
            'n_contestants': len(week_df),
            'n_eliminated': n_eliminate,
            'actual_eliminated': actual_eliminated,
            'percent_eliminated': percent_eliminated,
            'rank_eliminated': rank_eliminated,
            'elim_same': elim_same,
            'is_final': is_final,
            **final_info,
            **bias_info
        })
    
    return results


def run_full_analysis(df):
    """运行完整分析"""
    all_results = []
    
    for season in sorted(df['season'].unique()):
        season_df = df[df['season'] == season]
        season_results = analyze_season(season_df, season)
        all_results.extend(season_results)
    
    return all_results


def generate_statistics(results):
    """生成统计摘要"""
    df = pd.DataFrame(results)
    
    # 只统计有淘汰的周
    df_elim = df[df['n_eliminated'] > 0]
    
    # 整体一致性统计
    total_weeks = len(df_elim)
    same_weeks = df_elim['elim_same'].sum()
    consistency_rate = same_weeks / total_weeks * 100 if total_weeks > 0 else 0
    
    # 决赛统计
    df_final = df[df['is_final'] == True]
    df_final_valid = df_final[df_final['percent_exact'].notna()]
    
    total_finals = len(df_final_valid)
    percent_exact_count = df_final_valid['percent_exact'].sum() if total_finals > 0 else 0
    percent_top1_count = df_final_valid['percent_top1'].sum() if total_finals > 0 else 0
    rank_exact_count = df_final_valid['rank_exact'].sum() if total_finals > 0 else 0
    rank_top1_count = df_final_valid['rank_top1'].sum() if total_finals > 0 else 0
    
    # 偏向性统计
    rank_favor_total = df_elim['rank_favor_audience'].fillna(0).sum()
    percent_favor_total = df_elim['percent_favor_audience'].fillna(0).sum()
    
    # 按季度统计
    season_stats = []
    for season in df['season'].unique():
        season_df = df[df['season'] == season]
        season_elim = season_df[season_df['n_eliminated'] > 0]
        
        season_total = len(season_elim)
        season_same = season_elim['elim_same'].sum()
        season_rate = season_same / season_total * 100 if season_total > 0 else 0
        
        # 决赛信息
        final_row = season_df[season_df['is_final'] == True]
        percent_exact = final_row['percent_exact'].iloc[0] if len(final_row) > 0 and 'percent_exact' in final_row.columns else None
        rank_exact = final_row['rank_exact'].iloc[0] if len(final_row) > 0 and 'rank_exact' in final_row.columns else None
        
        season_stats.append({
            'season': season,
            'total_weeks': season_total,
            'same_weeks': season_same,
            'consistency_rate': season_rate,
            'percent_final_exact': percent_exact,
            'rank_final_exact': rank_exact
        })
    
    return {
        'overall': {
            'total_weeks': total_weeks,
            'same_weeks': same_weeks,
            'consistency_rate': consistency_rate,
            'total_finals': total_finals,
            'percent_exact_count': percent_exact_count,
            'percent_top1_count': percent_top1_count,
            'rank_exact_count': rank_exact_count,
            'rank_top1_count': rank_top1_count,
            'rank_favor_audience': rank_favor_total,
            'percent_favor_audience': percent_favor_total,
        },
        'by_season': pd.DataFrame(season_stats)
    }


def print_report(stats, results):
    """打印分析报告"""
    overall = stats['overall']
    season_df = stats['by_season']
    
    print("=" * 80)
    print("票分融合方式对比分析报告")
    print("=" * 80)
    
    print("\n【一、整体一致性统计】")
    print("-" * 50)
    print(f"有淘汰的总周数: {overall['total_weeks']}")
    print(f"淘汰结果一致周数: {overall['same_weeks']}")
    print(f"淘汰结果不一致周数: {overall['total_weeks'] - overall['same_weeks']}")
    print(f"一致性比例: {overall['consistency_rate']:.2f}%")
    
    print("\n【二、决赛排名准确性统计】")
    print("-" * 50)
    print(f"有效决赛季数: {overall['total_finals']}")
    print(f"\n{'指标':^20} | {'百分比法':^15} | {'排名法':^15}")
    print("-" * 55)
    print(f"{'排名完全正确':^20} | {overall['percent_exact_count']:^6}/{overall['total_finals']} ({overall['percent_exact_count']/overall['total_finals']*100 if overall['total_finals'] > 0 else 0:.1f}%) | {overall['rank_exact_count']:^6}/{overall['total_finals']} ({overall['rank_exact_count']/overall['total_finals']*100 if overall['total_finals'] > 0 else 0:.1f}%)")
    print(f"{'冠军预测正确':^20} | {overall['percent_top1_count']:^6}/{overall['total_finals']} ({overall['percent_top1_count']/overall['total_finals']*100 if overall['total_finals'] > 0 else 0:.1f}%) | {overall['rank_top1_count']:^6}/{overall['total_finals']} ({overall['rank_top1_count']/overall['total_finals']*100 if overall['total_finals'] > 0 else 0:.1f}%)")
    
    print("\n【三、偏向性分析】")
    print("-" * 50)
    print(f"排名法更偏向观众的次数: {int(overall['rank_favor_audience'])}")
    print(f"百分比法更偏向观众的次数: {int(overall['percent_favor_audience'])}")
    
    if overall['rank_favor_audience'] > overall['percent_favor_audience']:
        print("\n结论: 当淘汰结果不同时，排名法更偏向观众意见。")
    elif overall['percent_favor_audience'] > overall['rank_favor_audience']:
        print("\n结论: 当淘汰结果不同时，百分比法更偏向观众意见。")
    else:
        print("\n结论: 两种方法对观众意见的偏向程度相当。")
    
    print("\n【四、按季度一致性统计（部分）】")
    print("-" * 70)
    print(f"{'季数':^6} | {'总周数':^8} | {'一致周数':^10} | {'一致率':^10} | {'百分比法决赛':^12} | {'排名法决赛':^12}")
    print("-" * 70)
    
    for _, row in season_df.head(20).iterrows():
        p_exact = '√' if row['percent_final_exact'] == True else ('×' if row['percent_final_exact'] == False else '-')
        r_exact = '√' if row['rank_final_exact'] == True else ('×' if row['rank_final_exact'] == False else '-')
        print(f"{int(row['season']):^6} | {int(row['total_weeks']):^8} | {int(row['same_weeks']):^10} | {row['consistency_rate']:^9.1f}% | {p_exact:^12} | {r_exact:^12}")
    
    if len(season_df) > 20:
        print(f"... 共 {len(season_df)} 季")
    
    print("=" * 80)


def save_results(results, stats):
    """保存结果到CSV"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 保存详细结果
    df_results = pd.DataFrame(results)
    # 转换列表为字符串以便保存
    for col in ['actual_eliminated', 'percent_eliminated', 'rank_eliminated', 
                'only_percent_eliminated', 'only_rank_eliminated',
                'actual_rankings', 'percent_rankings', 'rank_rankings']:
        if col in df_results.columns:
            df_results[col] = df_results[col].apply(lambda x: str(x) if x else '')
    
    df_results.to_csv(os.path.join(OUTPUT_DIR, 'method_comparison_detail.csv'), 
                      index=False, encoding='utf-8-sig')
    
    # 保存季度统计
    stats['by_season'].to_csv(os.path.join(OUTPUT_DIR, 'method_comparison_by_season.csv'),
                              index=False, encoding='utf-8-sig')
    
    print(f"\n结果已保存到 {OUTPUT_DIR}/")


def main():
    print("正在加载数据...")
    df = load_data()
    print(f"加载完成: {len(df)} 条记录, {df['season'].nunique()} 个赛季")
    
    print("\n正在分析...")
    results = run_full_analysis(df)
    
    print("\n正在生成统计...")
    stats = generate_statistics(results)
    
    print_report(stats, results)
    save_results(results, stats)
    
    return results, stats


if __name__ == '__main__':
    results, stats = main()
