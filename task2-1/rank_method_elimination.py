"""
问题二：对第3-27季应用排名法进行淘汰，与百分比法淘汰结果比较

排名法规则（见 Problem C Appendix）：
- 每周对评委评分排名（1=最好），对观众投票排名（1=最好）
- 综合排名 = 评委排名 + 观众排名（和越小越好）
- 淘汰：综合排名和最大的 k 名选手被淘汰（k = 该周实际淘汰人数，与百分比法一致）

使用估计的观众投票百分比 fan_vote_estimates_entropy_smooth_150.csv 得到每周观众排名。
评委排名由 Data_4.xlsx 中该周评委百分比得到。

功能：
1. 每周淘汰结果对比（排名法 vs 百分比法）
2. 决赛排名准确性分析（两种方法预测的排名与实际排名对比）
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
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
# 使用 Data_4.xlsx 作为数据源（包含正确的淘汰信息）
DEFAULT_EXCEL = os.path.join(PROJECT_ROOT, 'Data_4.xlsx')
DEFAULT_FAN_CSV = os.path.join(SCRIPT_DIR, 'fan_vote_estimates_entropy_smooth_150.csv')


def load_processed_data(excel_path: str) -> pd.DataFrame:
    return pd.read_excel(excel_path)


def load_fan_estimates(csv_path: str) -> pd.DataFrame:
    return pd.read_csv(csv_path)


def get_season_weeks(season_df: pd.DataFrame) -> int:
    max_week = 1
    for w in range(1, 12):
        col = f'{w}_percent'
        if col in season_df.columns and (season_df[col] > 0).any():
            max_week = w
    return max_week


def get_week_contestants_and_judge(season_df: pd.DataFrame, week: int):
    col = f'{week}_percent'
    if col not in season_df.columns:
        return [], np.array([])
    mask = season_df[col] > 0
    df = season_df[mask]
    names = df['celebrity_name'].tolist()
    judge_percents = df[col].values
    return names, judge_percents


def get_eliminated_this_week(season_df: pd.DataFrame, week: int) -> list:
    eliminated = []
    for _, row in season_df.iterrows():
        r = str(row.get('results', '')).lower()
        if f'eliminated week {week}' in r:
            eliminated.append(row['celebrity_name'])
    return eliminated


def get_max_elimination_week(season_df: pd.DataFrame) -> int:
    """获取该季最大淘汰周数"""
    max_elim_week = 0
    for w in range(1, 12):
        for _, row in season_df.iterrows():
            r = str(row.get('results', '')).lower()
            if f'eliminated week {w}' in r:
                max_elim_week = w
                break
    return max_elim_week


def get_final_week(season_df: pd.DataFrame) -> int:
    """获取决赛周数（最大淘汰周+1）"""
    max_elim_week = get_max_elimination_week(season_df)
    return max_elim_week + 1 if max_elim_week > 0 else 0


def get_final_rankings(season_df: pd.DataFrame, week: int) -> dict:
    """
    获取决赛周的实际排名信息
    
    Returns:
        dict: {celebrity_name: placement}, 如果不是决赛周返回空字典
    """
    rankings = {}
    col = f'{week}_percent'
    
    if col not in season_df.columns:
        return rankings
    
    # 确认是决赛周
    expected_final_week = get_final_week(season_df)
    if week != expected_final_week:
        return rankings
    
    # 遍历本周有评分的选手
    week_contestants = season_df[season_df[col] > 0]
    
    for _, row in week_contestants.iterrows():
        result = str(row.get('results', '')).lower()
        # 检查是否包含排名信息(1st place, 2nd place等)且不是被淘汰
        if 'place' in result and 'eliminated' not in result:
            rankings[row['celebrity_name']] = int(row['placement'])
    
    # 只有当有多个选手有排名信息时才返回(确认是决赛)
    if len(rankings) >= 2:
        return rankings
    else:
        return {}


def compute_final_ranking_by_percent(
    names: list,
    judge_percents: np.ndarray,
    fan_percents: np.ndarray,
) -> dict:
    """
    根据百分比法计算决赛排名
    总分 = 评委% + 观众%，总分越高排名越前
    
    Returns:
        dict: {celebrity_name: rank} (1=第一名)
    """
    total = judge_percents + fan_percents
    # 按总分降序排序，排名越小越好
    order = np.argsort(-total)
    rankings = {}
    for rank, idx in enumerate(order, start=1):
        rankings[names[idx]] = rank
    return rankings


def compute_final_ranking_by_rank_method(
    names: list,
    judge_percents: np.ndarray,
    fan_percents: np.ndarray,
) -> dict:
    """
    根据排名法计算决赛排名
    综合排名 = 评委排名 + 观众排名，综合排名越小越好
    同分时评委排名小的（更好）排在前面
    
    Returns:
        dict: {celebrity_name: rank} (1=第一名)
    """
    # 评委排名：降序，最高百分比=1
    judge_rank = pd.Series(judge_percents).rank(ascending=False, method='min').astype(int).values
    # 观众排名：同上
    fan_rank = pd.Series(fan_percents).rank(ascending=False, method='min').astype(int).values
    sum_rank = judge_rank + fan_rank
    
    # 按综合排名升序（越小越好），同分时评委排名小的在前
    idx = np.lexsort((judge_rank, sum_rank))
    
    rankings = {}
    for rank, i in enumerate(idx, start=1):
        rankings[names[i]] = rank
    return rankings


def compare_ranking_accuracy(
    actual_rankings: dict,
    method_rankings: dict,
) -> dict:
    """
    比较预测排名与实际排名的准确性
    
    Returns:
        dict: {
            'exact_match': bool,  # 完全一致
            'top1_correct': bool,  # 第一名正确
            'top2_correct': bool,  # 前两名正确（顺序也对）
            'kendall_tau': float,  # Kendall tau相关系数
            'displacement_sum': int,  # 排名位移总和
        }
    """
    if not actual_rankings or not method_rankings:
        return {}
    
    # 取交集的选手
    common_names = set(actual_rankings.keys()) & set(method_rankings.keys())
    if len(common_names) < 2:
        return {}
    
    # 按实际排名排序
    sorted_by_actual = sorted(common_names, key=lambda x: actual_rankings[x])
    sorted_by_method = sorted(common_names, key=lambda x: method_rankings[x])
    
    # 完全一致
    exact_match = sorted_by_actual == sorted_by_method
    
    # 第一名正确
    top1_correct = sorted_by_actual[0] == sorted_by_method[0]
    
    # 前两名正确（顺序也对）
    top2_correct = sorted_by_actual[:2] == sorted_by_method[:2]
    
    # 排名位移总和
    displacement_sum = 0
    for name in common_names:
        displacement_sum += abs(actual_rankings[name] - method_rankings[name])
    
    # Kendall tau（简化计算：统计concordant和discordant对）
    n = len(common_names)
    concordant = 0
    discordant = 0
    for i, name_i in enumerate(sorted_by_actual):
        for name_j in sorted_by_actual[i+1:]:
            # 在actual中 name_i 排在 name_j 前面
            # 检查在method中是否也是
            if method_rankings[name_i] < method_rankings[name_j]:
                concordant += 1
            elif method_rankings[name_i] > method_rankings[name_j]:
                discordant += 1
            # 相等时不计
    
    total_pairs = n * (n - 1) / 2
    kendall_tau = (concordant - discordant) / total_pairs if total_pairs > 0 else 0
    
    return {
        'exact_match': exact_match,
        'top1_correct': top1_correct,
        'top2_correct': top2_correct,
        'kendall_tau': kendall_tau,
        'displacement_sum': displacement_sum,
        'n_contestants': n,
    }


def apply_rank_method_one_week(
    names: list,
    judge_percents: np.ndarray,
    fan_percents: np.ndarray,
    n_eliminate: int,
) -> list:
    """
    对该周选手按排名法决定淘汰名单。
    评委排名：1=评委百分比最高（最好）；观众排名：1=观众百分比最高（最好）。
    综合 = 评委排名 + 观众排名，和越大越差。淘汰综合和最大的 n_eliminate 人。
    同分时先淘汰评委排名更差（数字更大）的，即更偏观众的人先淘汰。

    排名规则：同一百分比的选手排名相同，下一名顺延。例如两人并列第一，则下一名为第三名。
    使用 pandas rank(ascending=False, method='min') 实现：同分取该组最小名次，后续名次顺延。
    """
    n = len(names)
    if n_eliminate <= 0:
        return []
    if n_eliminate >= n:
        return list(names)

    # 评委排名：降序，最高百分比=1；同分同排名、下一名顺延（method='min'）
    judge_rank = pd.Series(judge_percents).rank(ascending=False, method='min').astype(int).values
    # 观众排名：同上
    fan_rank = pd.Series(fan_percents).rank(ascending=False, method='min').astype(int).values
    sum_rank = judge_rank + fan_rank

    # 按综合和从大到小排，同分时评委排名大的（更差）在前，先淘汰
    idx = np.lexsort((-judge_rank, -sum_rank))  # 先按 -sum_rank 升序即 sum_rank 降序，再按 -judge_rank
    eliminated_idx = idx[:n_eliminate]
    return [names[i] for i in eliminated_idx]


def run_rank_method_all_seasons(
    excel_path: str,
    fan_csv_path: str,
    seasons: tuple = (3, 28),
) -> tuple:
    """
    对指定季度范围（默认3-27）每周应用排名法，得到淘汰名单，并与百分比法实际淘汰比较。
    返回：(weekly_results, differences_summary, final_rankings_results)
    """
    raw = load_processed_data(excel_path)
    fan_df = load_fan_estimates(fan_csv_path)

    rows = []
    differences = []
    final_results = []  # 决赛排名结果

    for season in range(seasons[0], seasons[1]):
        season_df = raw[raw['season'] == season]
        if season_df.empty:
            continue
        max_week = get_season_weeks(season_df)
        fan_season = fan_df[fan_df['season'] == season]
        final_week = get_final_week(season_df)

        for week in range(1, max_week + 1):
            names, judge_percents = get_week_contestants_and_judge(season_df, week)
            if not names:
                continue

            # 该周实际淘汰（百分比法）
            actual_eliminated = get_eliminated_this_week(season_df, week)
            n_eliminate = len(actual_eliminated)

            # 观众百分比：从估计 CSV 取该周该季的 (celebrity_name, fan_vote_percent)
            week_fan = fan_season[fan_season['week'] == week]
            name_to_fan = dict(zip(week_fan['celebrity_name'], week_fan['fan_vote_percent']))
            fan_percents = np.array([name_to_fan.get(n, 0.0) for n in names])

            # 排名法淘汰
            rank_eliminated = apply_rank_method_one_week(
                names, judge_percents, fan_percents, n_eliminate
            )

            set_actual = set(actual_eliminated)
            set_rank = set(rank_eliminated)
            same = set_actual == set_rank

            # 基本行数据
            row_data = {
                'season': season,
                'week': week,
                'n_contestants': len(names),
                'n_eliminated': n_eliminate,
                'percent_eliminated': ','.join(sorted(actual_eliminated)) if actual_eliminated else '',
                'rank_eliminated': ','.join(sorted(rank_eliminated)) if rank_eliminated else '',
                'same_result': same,
                'is_final': False,
                'actual_ranking': '',
                'percent_ranking': '',
                'rank_method_ranking': '',
                'percent_exact_match': '',
                'rank_exact_match': '',
                'percent_top1_correct': '',
                'rank_top1_correct': '',
            }

            # 决赛排名分析
            if week == final_week:
                actual_rankings = get_final_rankings(season_df, week)
                if actual_rankings:
                    # 计算两种方法的排名
                    percent_rankings = compute_final_ranking_by_percent(
                        names, judge_percents, fan_percents
                    )
                    rank_method_rankings = compute_final_ranking_by_rank_method(
                        names, judge_percents, fan_percents
                    )
                    
                    # 比较准确性
                    percent_accuracy = compare_ranking_accuracy(actual_rankings, percent_rankings)
                    rank_accuracy = compare_ranking_accuracy(actual_rankings, rank_method_rankings)
                    
                    # 格式化排名字符串（按排名顺序）
                    def format_ranking(rankings):
                        return '>'.join([name for name, _ in sorted(rankings.items(), key=lambda x: x[1])])
                    
                    row_data['is_final'] = True
                    row_data['actual_ranking'] = format_ranking(actual_rankings)
                    row_data['percent_ranking'] = format_ranking(percent_rankings)
                    row_data['rank_method_ranking'] = format_ranking(rank_method_rankings)
                    row_data['percent_exact_match'] = percent_accuracy.get('exact_match', False)
                    row_data['rank_exact_match'] = rank_accuracy.get('exact_match', False)
                    row_data['percent_top1_correct'] = percent_accuracy.get('top1_correct', False)
                    row_data['rank_top1_correct'] = rank_accuracy.get('top1_correct', False)
                    
                    final_results.append({
                        'season': season,
                        'week': week,
                        'n_finalists': len(actual_rankings),
                        'actual_rankings': actual_rankings,
                        'percent_rankings': percent_rankings,
                        'rank_method_rankings': rank_method_rankings,
                        'percent_accuracy': percent_accuracy,
                        'rank_accuracy': rank_accuracy,
                    })

            rows.append(row_data)

            if not same:
                only_percent = set_actual - set_rank  # 仅百分比法淘汰的
                only_rank_set = set_rank - set_actual  # 仅排名法淘汰的
                differences.append({
                    'season': season,
                    'week': week,
                    'only_percent_eliminated': list(only_percent),
                    'only_rank_eliminated': list(only_rank_set),
                    'actual_eliminated': actual_eliminated,
                    'rank_eliminated': rank_eliminated,
                })

    return pd.DataFrame(rows), differences, final_results


def analyze_which_favors_audience(differences: list, excel_path: str, fan_csv_path: str) -> str:
    """
    当两种方法淘汰结果不同时，分析哪种方法更偏向观众意见。
    若某人在百分比法下被淘汰、在排名法下未淘汰：说明百分比法下他“总分”低；可能评委分低但观众分高，则排名法救了他 → 排名法更偏观众。
    若某人在排名法下被淘汰、在百分比法下未淘汰：说明排名法下他“名次和”大；可能观众排名差，则百分比法救了他 → 百分比法更偏观众。
    综合：看多数差异案例是“谁被谁救”，给出结论。
    """
    if not differences:
        return "两种方法每周淘汰结果一致，无法区分谁更偏观众。"

    raw = load_processed_data(excel_path)
    fan_df = load_fan_estimates(fan_csv_path)

    # 对每个差异周：看 only_rank_eliminated（仅排名法淘汰）的人观众%是否普遍较低；only_percent_eliminated（仅百分比法淘汰）的人观众%是否较高
    rank_eliminated_tend_to_low_fan = 0
    percent_eliminated_tend_to_high_fan = 0
    cases = 0

    for d in differences:
        s, w = d['season'], d['week']
        only_percent = d['only_percent_eliminated']  # 仅百分比法淘汰 → 排名法下没淘汰
        only_rank = d['only_rank_eliminated']       # 仅排名法淘汰 → 百分比法下没淘汰

        week_fan = fan_df[(fan_df['season'] == s) & (fan_df['week'] == w)]
        if week_fan.empty:
            continue
        all_fan = week_fan.set_index('celebrity_name')['fan_vote_percent']
        mean_fan = all_fan.mean()

        # 仅排名法淘汰的人：若其观众%多低于当周平均，说明排名法淘汰的是“观众人气较低”的 → 排名法更偏评委
        for name in only_rank:
            if name in all_fan.index:
                if all_fan[name] < mean_fan:
                    rank_eliminated_tend_to_low_fan += 1
                cases += 1
        # 仅百分比法淘汰的人：若其观众%多高于当周平均，说明百分比法淘汰了“观众人气较高”的 → 百分比法更偏评委（排名法更偏观众）
        for name in only_percent:
            if name in all_fan.index:
                if all_fan[name] > mean_fan:
                    percent_eliminated_tend_to_high_fan += 1
                cases += 1

    # 综合判断
    # 仅百分比法淘汰且观众%高 → 排名法救了高观众人气 → 排名法更偏观众
    # 仅排名法淘汰且观众%低 → 排名法淘汰了低观众人气、百分比法救了他们 → 排名法更偏观众（百分比法更偏评委）
    favor_rank_audience = percent_eliminated_tend_to_high_fan + rank_eliminated_tend_to_low_fan
    favor_percent_audience = cases - favor_rank_audience  # 其余情况可视为百分比法更偏观众

    print(f"favor_rank_audience: {favor_rank_audience},这是排名法更偏观众的次数")
    print(f"favor_percent_audience: {favor_percent_audience},这是百分比法更偏观众的次数")


    if favor_rank_audience > favor_percent_audience:
        return "当淘汰结果不同时，多数情况下：（1）「仅百分比法淘汰」的选手观众投票占比较高，即排名法下他们被保留；或（2）「仅排名法淘汰」的选手观众投票占比偏低，即排名法淘汰了观众人气较低者、百分比法则因其评委分高而保留他们。综合可知 **排名法更偏向观众意见**（观众人气高的人更容易在排名法下被保留，观众人气低的人更容易在排名法下被淘汰）。"
    elif favor_percent_audience > favor_rank_audience:
        return "当淘汰结果不同时，多数情况下「仅百分比法淘汰」的选手观众投票占比偏低，即百分比法淘汰了观众人气较低者、排名法则因其评委排名尚可而保留他们。因此在这些周中 **百分比法更偏向观众意见**（在百分比法下观众人气对总分影响更大，观众人气高的人更容易被保留）。"
    else:
        return "当淘汰结果不同时，两种方法下被淘汰选手的观众人气分布没有明显倾向，无法明确判断哪种方法更偏向观众意见。"


def format_ranking_list(rankings: dict) -> str:
    """将排名字典格式化为字符串，按排名顺序显示"""
    if not rankings:
        return "无"
    sorted_items = sorted(rankings.items(), key=lambda x: x[1])
    return ' > '.join([f"{name}({rank})" for name, rank in sorted_items])


def analyze_final_rankings(final_results: list) -> None:
    """分析决赛排名准确性"""
    if not final_results:
        print("无决赛排名数据")
        return
    
    print("\n" + "=" * 80)
    print("决赛排名准确性分析")
    print("=" * 80)
    
    # 统计汇总
    percent_exact = 0
    percent_top1 = 0
    percent_top2 = 0
    rank_exact = 0
    rank_top1 = 0
    rank_top2 = 0
    total_seasons = len(final_results)
    
    percent_kendall_sum = 0
    rank_kendall_sum = 0
    percent_displacement_sum = 0
    rank_displacement_sum = 0
    
    print(f"\n共 {total_seasons} 季有决赛排名数据\n")
    
    # 详细显示每季决赛排名
    for fr in final_results:
        season = fr['season']
        week = fr['week']
        n_finalists = fr['n_finalists']
        actual = fr['actual_rankings']
        percent_r = fr['percent_rankings']
        rank_r = fr['rank_method_rankings']
        pa = fr['percent_accuracy']
        ra = fr['rank_accuracy']
        
        print("-" * 80)
        print(f"【第 {season} 季 第 {week} 周决赛】 ({n_finalists} 位选手)")
        print(f"  实际排名:   {format_ranking_list(actual)}")
        print(f"  百分比法:   {format_ranking_list(percent_r)}", end="")
        if pa.get('exact_match'):
            print(" ✓完全正确")
        elif pa.get('top1_correct'):
            print(" ✓冠军正确")
        else:
            print(" ✗")
        print(f"  排名法:     {format_ranking_list(rank_r)}", end="")
        if ra.get('exact_match'):
            print(" ✓完全正确")
        elif ra.get('top1_correct'):
            print(" ✓冠军正确")
        else:
            print(" ✗")
        
        # 累计统计
        if pa.get('exact_match'):
            percent_exact += 1
        if pa.get('top1_correct'):
            percent_top1 += 1
        if pa.get('top2_correct'):
            percent_top2 += 1
        if ra.get('exact_match'):
            rank_exact += 1
        if ra.get('top1_correct'):
            rank_top1 += 1
        if ra.get('top2_correct'):
            rank_top2 += 1
        
        percent_kendall_sum += pa.get('kendall_tau', 0)
        rank_kendall_sum += ra.get('kendall_tau', 0)
        percent_displacement_sum += pa.get('displacement_sum', 0)
        rank_displacement_sum += ra.get('displacement_sum', 0)
    
    print("-" * 80)
    
    # 汇总统计
    print(f"\n【汇总统计】({total_seasons} 季决赛)")
    print("-" * 50)
    print(f"{'指标':^20} | {'百分比法':^12} | {'排名法':^12}")
    print("-" * 50)
    print(f"{'完全正确':^20} | {percent_exact:^6} ({percent_exact/total_seasons*100:5.1f}%) | {rank_exact:^6} ({rank_exact/total_seasons*100:5.1f}%)")
    print(f"{'第一名正确':^20} | {percent_top1:^6} ({percent_top1/total_seasons*100:5.1f}%) | {rank_top1:^6} ({rank_top1/total_seasons*100:5.1f}%)")
    print(f"{'前两名正确':^20} | {percent_top2:^6} ({percent_top2/total_seasons*100:5.1f}%) | {rank_top2:^6} ({rank_top2/total_seasons*100:5.1f}%)")
    print(f"{'平均Kendall τ':^20} | {percent_kendall_sum/total_seasons:^12.4f} | {rank_kendall_sum/total_seasons:^12.4f}")
    print(f"{'总排名位移':^20} | {percent_displacement_sum:^12} | {rank_displacement_sum:^12}")
    print("-" * 50)
    
    # 结论
    print("\n【决赛排名准确性结论】")
    if percent_top1 > rank_top1:
        print(f"  百分比法在预测冠军方面更准确 ({percent_top1} vs {rank_top1})")
    elif rank_top1 > percent_top1:
        print(f"  排名法在预测冠军方面更准确 ({rank_top1} vs {percent_top1})")
    else:
        print(f"  两种方法预测冠军准确率相同 ({percent_top1})")
    
    if percent_exact > rank_exact:
        print(f"  百分比法在预测完整排名方面更准确 ({percent_exact} vs {rank_exact})")
    elif rank_exact > percent_exact:
        print(f"  排名法在预测完整排名方面更准确 ({rank_exact} vs {percent_exact})")
    else:
        print(f"  两种方法预测完整排名准确率相同 ({percent_exact})")


def main():
    excel_path = DEFAULT_EXCEL
    fan_csv_path = DEFAULT_FAN_CSV

    print("=" * 60)
    print("问题二：排名法 vs 百分比法 淘汰结果比较（第3-27季）")
    print("=" * 60)
    print(f"评委数据: {excel_path}")
    print(f"观众估计: {fan_csv_path}")
    print()

    df, differences, final_results = run_rank_method_all_seasons(excel_path, fan_csv_path)

    # 汇总（排除没有人淘汰的周数）
    total_weeks = len(df)
    # same_weeks = df['same_result'].sum()
    # diff_weeks = total_weeks - same_weeks
    df_with_elimination = df[df['n_eliminated'] > 0]  # 只考虑有人淘汰的周数
    weeks_with_elimination = len(df_with_elimination)
    same_weeks = df_with_elimination['same_result'].sum()
    diff_weeks = weeks_with_elimination - same_weeks
    weeks_no_elimination = total_weeks - weeks_with_elimination
    
    print(f"总周数（有选手参与）: {total_weeks}")
    print(f"  其中无人淘汰周数: {weeks_no_elimination}")
    print(f"  其中有人淘汰周数: {weeks_with_elimination}")
    print(f"淘汰结果一致周数: {same_weeks}")
    print(f"淘汰结果不同周数: {diff_weeks}")
    if weeks_with_elimination > 0:
        print(f"一致比例（仅统计有淘汰的周数）: {same_weeks/weeks_with_elimination*100:.1f}%")

    # if total_weeks > 0:
    #     print(f"一致比例: {same_weeks/total_weeks*100:.1f}%")
    print()

    # 保存每周对比
    out_csv = os.path.join(SCRIPT_DIR, 'rank_vs_percent_elimination_150.csv')
    df.to_csv(out_csv, index=False, encoding='utf-8-sig')
    print(f"每周对比已保存: {out_csv}")
    print()

    # 列出部分差异周
    if differences:
        print("淘汰结果不同的部分周（前20个）：")
        print("-" * 60)
        for i, d in enumerate(differences[:20]):
            print(f"  第{d['season']}季 第{d['week']}周:")
            print(f"    百分比法淘汰: {d['actual_eliminated']}")
            print(f"    排名法淘汰:   {d['rank_eliminated']}")
            print(f"    仅百分比法淘汰: {d['only_percent_eliminated']}")
            print(f"    仅排名法淘汰:   {d['only_rank_eliminated']}")
            print()
        if len(differences) > 20:
            print(f"  ... 共 {len(differences)} 周不同")
        print()
        conclusion = analyze_which_favors_audience(differences, excel_path, fan_csv_path)
        print("结论（哪种方法更偏向观众意见）：")
        print(conclusion)
    else:
        print("所有周淘汰结果一致，无需判断谁更偏观众。")

    # 决赛排名分析
    analyze_final_rankings(final_results)

    return df, differences, final_results


if __name__ == '__main__':
    main()
