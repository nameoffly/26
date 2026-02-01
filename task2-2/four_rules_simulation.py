"""
四种规则模拟分析 - 争议选手专项分析

四种规则组合：
R1: 纯百分比法 - 综合得分=评委%+观众%，直接淘汰末位
R2: 纯排名法 - 综合排名=评委排名+观众排名，直接淘汰末位
R3: 百分比+评委决定 - 用百分比法确定末两位，评委从中选择淘汰
R4: 排名+评委决定 - 用排名法确定末两位，评委从中选择淘汰

争议选手：
- 第2季 Jerry Rice：5周评委垫底，最终亚军
- 第4季 Billy Ray Cyrus：8周评委倒数第1，第5名
- 第11季 Bristol Palin：12次评委最低，季军
- 第27季 Bobby Bones：8周评委倒数第1，冠军
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

# 争议选手定义
# actual_method: 该赛季实际使用的方法 ('rank' for S1-2,S28-34; 'percent' for S3-27)
CONTROVERSIAL_CONTESTANTS = [
    {'season': 2, 'name': 'Jerry Rice', 'actual_placement': 2, 'description': '5周评委垫底，亚军', 'actual_method': 'rank'},
    {'season': 4, 'name': 'Billy Ray Cyrus', 'actual_placement': 5, 'description': '8周评委倒数第1，第5名', 'actual_method': 'percent'},
    {'season': 11, 'name': 'Bristol Palin', 'actual_placement': 3, 'description': '12次评委最低，季军', 'actual_method': 'percent'},
    {'season': 27, 'name': 'Bobby Bones', 'actual_placement': 1, 'description': '8周评委倒数第1，冠军', 'actual_method': 'percent'},
]


def load_data():
    """加载数据"""
    df = pd.read_csv(DATA_FILE)
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


def get_actual_eliminated(week_df, week):
    """获取本周实际被淘汰的选手"""
    eliminated = []
    for _, row in week_df.iterrows():
        elim_week = get_elimination_week(row['result'])
        if elim_week == week:
            eliminated.append(row['name'])
    return eliminated


# ================== 四种规则实现 ==================

def rule_percent_only(week_df, n_eliminate):
    """R1: 纯百分比法 - 直接淘汰总分最低的"""
    if n_eliminate <= 0:
        return [], []
    
    sorted_df = week_df.sort_values('total_percent', ascending=True)
    eliminated = sorted_df.head(n_eliminate)['name'].tolist()
    bottom_two = sorted_df.head(2)['name'].tolist() if len(sorted_df) >= 2 else []
    return eliminated, bottom_two


def rule_rank_only(week_df, n_eliminate):
    """R2: 纯排名法 - 直接淘汰综合排名最差的"""
    if n_eliminate <= 0:
        return [], []
    
    df = week_df.copy()
    df['judge_rank'] = df['judge_percent'].rank(ascending=False, method='min').astype(int)
    df['fan_rank'] = df['fan_percent'].rank(ascending=False, method='min').astype(int)
    df['sum_rank'] = df['judge_rank'] + df['fan_rank']
    
    # 按综合排名降序，同分时评委排名差的优先
    df = df.sort_values(['sum_rank', 'judge_rank'], ascending=[False, False])
    eliminated = df.head(n_eliminate)['name'].tolist()
    bottom_two = df.head(2)['name'].tolist() if len(df) >= 2 else []
    return eliminated, bottom_two


def rule_percent_judge_decide(week_df, n_eliminate):
    """
    R3: 百分比+评委决定
    - 用百分比法确定末两位
    - 评委从末两位中选择淘汰（假设评委选择评委分更低的）
    """
    if n_eliminate <= 0:
        return [], []
    
    sorted_df = week_df.sort_values('total_percent', ascending=True)
    bottom_two = sorted_df.head(2)['name'].tolist() if len(sorted_df) >= 2 else sorted_df['name'].tolist()
    
    if n_eliminate == 1 and len(bottom_two) == 2:
        # 评委从末两位中选择评委分更低的淘汰
        bottom_df = sorted_df[sorted_df['name'].isin(bottom_two)]
        eliminated_name = bottom_df.sort_values('judge_percent', ascending=True).iloc[0]['name']
        eliminated = [eliminated_name]
    else:
        # 直接淘汰末n_eliminate位
        eliminated = sorted_df.head(n_eliminate)['name'].tolist()
    
    return eliminated, bottom_two


def rule_rank_judge_decide(week_df, n_eliminate):
    """
    R4: 排名+评委决定
    - 用排名法确定末两位
    - 评委从末两位中选择淘汰（假设评委选择评委分更低的）
    """
    if n_eliminate <= 0:
        return [], []
    
    df = week_df.copy()
    df['judge_rank'] = df['judge_percent'].rank(ascending=False, method='min').astype(int)
    df['fan_rank'] = df['fan_percent'].rank(ascending=False, method='min').astype(int)
    df['sum_rank'] = df['judge_rank'] + df['fan_rank']
    
    # 按综合排名降序
    df = df.sort_values(['sum_rank', 'judge_rank'], ascending=[False, False])
    bottom_two = df.head(2)['name'].tolist() if len(df) >= 2 else df['name'].tolist()
    
    if n_eliminate == 1 and len(bottom_two) == 2:
        # 评委从末两位中选择评委分更低的淘汰
        bottom_df = df[df['name'].isin(bottom_two)]
        eliminated_name = bottom_df.sort_values('judge_percent', ascending=True).iloc[0]['name']
        eliminated = [eliminated_name]
    else:
        # 直接淘汰末n_eliminate位
        eliminated = df.head(n_eliminate)['name'].tolist()
    
    return eliminated, bottom_two


# ================== 模拟引擎 ==================

class SeasonSimulator:
    """赛季模拟器"""
    
    def __init__(self, season_df, target_contestant):
        self.original_df = season_df.copy()
        self.target = target_contestant
        self.season = season_df['season'].iloc[0]
    
    def simulate_rule(self, rule_func, rule_name):
        """
        模拟单个规则下的整季比赛
        返回：目标选手的淘汰周数、是否进入决赛、最终名次
        """
        # 追踪存活选手（按名字追踪）
        all_contestants = self.original_df.drop_duplicates('name')['name'].tolist()
        alive = set(all_contestants)
        
        max_week = self.original_df['week'].max()
        weekly_status = []
        
        target_eliminated_week = None
        target_in_bottom_two = []
        
        for week in range(1, max_week + 1):
            week_df = self.original_df[
                (self.original_df['week'] == week) & 
                (self.original_df['name'].isin(alive))
            ].copy()
            
            if week_df.empty:
                continue
            
            # 获取实际淘汰人数
            actual_eliminated = get_actual_eliminated(self.original_df[self.original_df['week'] == week], week)
            n_eliminate = len(actual_eliminated)
            
            # 检查目标选手是否在本周
            target_in_week = self.target in week_df['name'].values
            
            if not target_in_week:
                # 目标选手已被淘汰
                weekly_status.append({
                    'week': week,
                    'n_contestants': len(week_df),
                    'n_eliminate': n_eliminate,
                    'target_alive': False,
                    'target_in_bottom_two': False,
                    'target_eliminated': False,
                })
                continue
            
            # 应用规则
            eliminated, bottom_two = rule_func(week_df, n_eliminate)
            
            # 检查目标选手状态
            target_in_bottom = self.target in bottom_two
            target_eliminated_this_week = self.target in eliminated
            
            if target_in_bottom:
                target_in_bottom_two.append(week)
            
            # 记录周状态
            weekly_status.append({
                'week': week,
                'n_contestants': len(week_df),
                'n_eliminate': n_eliminate,
                'eliminated': eliminated,
                'bottom_two': bottom_two,
                'target_alive': True,
                'target_in_bottom_two': target_in_bottom,
                'target_eliminated': target_eliminated_this_week,
            })
            
            # 更新存活名单
            for name in eliminated:
                alive.discard(name)
            
            # 如果目标被淘汰
            if target_eliminated_this_week:
                target_eliminated_week = week
                break
        
        # 计算最终名次
        if target_eliminated_week:
            # 被淘汰时的名次 = 被淘汰时剩余人数 + 1（近似）
            remaining_when_eliminated = len(alive) + 1
            final_placement = remaining_when_eliminated
        else:
            # 进入决赛，需要计算决赛排名
            final_week = max_week
            final_df = self.original_df[
                (self.original_df['week'] == final_week) & 
                (self.original_df['name'].isin(alive))
            ].copy()
            
            if len(final_df) > 0:
                # 根据规则计算决赛排名
                if 'percent' in rule_name.lower() and 'judge' not in rule_name.lower():
                    # 纯百分比法
                    final_df = final_df.sort_values('total_percent', ascending=False)
                elif 'rank' in rule_name.lower() and 'judge' not in rule_name.lower():
                    # 纯排名法
                    final_df['j_rank'] = final_df['judge_percent'].rank(ascending=False, method='min')
                    final_df['f_rank'] = final_df['fan_percent'].rank(ascending=False, method='min')
                    final_df['s_rank'] = final_df['j_rank'] + final_df['f_rank']
                    final_df = final_df.sort_values(['s_rank', 'j_rank'], ascending=[True, True])
                else:
                    # 带评委决定的，也按综合分排
                    if 'percent' in rule_name.lower():
                        final_df = final_df.sort_values('total_percent', ascending=False)
                    else:
                        final_df['j_rank'] = final_df['judge_percent'].rank(ascending=False, method='min')
                        final_df['f_rank'] = final_df['fan_percent'].rank(ascending=False, method='min')
                        final_df['s_rank'] = final_df['j_rank'] + final_df['f_rank']
                        final_df = final_df.sort_values(['s_rank', 'j_rank'], ascending=[True, True])
                
                final_ranking = final_df['name'].tolist()
                if self.target in final_ranking:
                    final_placement = final_ranking.index(self.target) + 1
                else:
                    final_placement = None
            else:
                final_placement = None
        
        return {
            'rule_name': rule_name,
            'survival_weeks': target_eliminated_week - 1 if target_eliminated_week else max_week,
            'eliminated_week': target_eliminated_week,
            'reached_final': target_eliminated_week is None,
            'final_placement': final_placement,
            'times_in_bottom_two': len(target_in_bottom_two),
            'bottom_two_weeks': target_in_bottom_two,
            'weekly_status': weekly_status,
        }


def analyze_controversial_contestant(df, contestant_info):
    """分析单个争议选手"""
    season = contestant_info['season']
    name = contestant_info['name']
    actual_placement = contestant_info['actual_placement']
    actual_method = contestant_info.get('actual_method', 'percent')  # 默认百分比法
    
    season_df = df[df['season'] == season]
    if name not in season_df['name'].values:
        print(f"警告：找不到 {name} 在第 {season} 季的数据")
        return None
    
    # 获取选手实际存活周数
    contestant_data = season_df[season_df['name'] == name]
    actual_survival_weeks = len(contestant_data)  # 选手参与的周数
    
    simulator = SeasonSimulator(season_df, name)
    
    # 运行四种规则
    rules = [
        (rule_percent_only, 'R1: 纯百分比法', 'percent'),
        (rule_rank_only, 'R2: 纯排名法', 'rank'),
        (rule_percent_judge_decide, 'R3: 百分比+评委决定', 'percent_judge'),
        (rule_rank_judge_decide, 'R4: 排名+评委决定', 'rank_judge'),
    ]
    
    results = []
    for rule_func, rule_name, method_type in rules:
        # 如果该规则对应实际使用的方法（不带评委决定的），直接使用实际结果
        if (actual_method == 'percent' and method_type == 'percent') or \
           (actual_method == 'rank' and method_type == 'rank'):
            # 使用实际结果
            result = {
                'rule_name': rule_name,
                'survival_weeks': actual_survival_weeks,
                'eliminated_week': None,  # 未被淘汰（进入决赛）
                'reached_final': True,
                'final_placement': actual_placement,
                'times_in_bottom_two': 0,  # 实际数据未知，设为0
                'bottom_two_weeks': [],
                'weekly_status': [],
                'is_actual': True,  # 标记这是实际结果
            }
        else:
            # 进行模拟
            result = simulator.simulate_rule(rule_func, rule_name)
            result['is_actual'] = False
        
        results.append(result)
    
    return {
        'contestant': contestant_info,
        'results': results,
    }


def get_contestant_weekly_ranking(df, season, name):
    """获取选手每周的评委排名和观众排名"""
    season_df = df[df['season'] == season]
    contestant_df = season_df[season_df['name'] == name]
    
    rankings = []
    for _, row in contestant_df.iterrows():
        week = row['week']
        week_df = season_df[season_df['week'] == week]
        
        # 计算排名
        judge_rank = (week_df['judge_percent'] > row['judge_percent']).sum() + 1
        fan_rank = (week_df['fan_percent'] > row['fan_percent']).sum() + 1
        n_contestants = len(week_df)
        
        rankings.append({
            'week': week,
            'n_contestants': n_contestants,
            'judge_percent': row['judge_percent'],
            'fan_percent': row['fan_percent'],
            'judge_rank': judge_rank,
            'fan_rank': fan_rank,
            'judge_rank_from_bottom': n_contestants - judge_rank + 1,  # 倒数第几
        })
    
    return pd.DataFrame(rankings)


def print_contestant_analysis(analysis, df):
    """打印单个选手的分析结果"""
    info = analysis['contestant']
    results = analysis['results']
    
    print("\n" + "=" * 80)
    print(f"【争议选手分析】第 {info['season']} 季 - {info['name']}")
    print(f"描述：{info['description']}")
    print(f"实际名次：第 {info['actual_placement']} 名")
    print("=" * 80)
    
    # 打印每周排名情况
    rankings = get_contestant_weekly_ranking(df, info['season'], info['name'])
    print("\n【每周排名情况】")
    print("-" * 70)
    print(f"{'周数':^6} | {'选手数':^8} | {'评委%':^10} | {'观众%':^10} | {'评委排名':^10} | {'观众排名':^10}")
    print("-" * 70)
    
    judge_bottom_count = 0
    for _, row in rankings.iterrows():
        is_judge_bottom = row['judge_rank'] == row['n_contestants']
        if is_judge_bottom:
            judge_bottom_count += 1
        mark = '←最低' if is_judge_bottom else ''
        print(f"{int(row['week']):^6} | {int(row['n_contestants']):^8} | {row['judge_percent']:^10.4f} | {row['fan_percent']:^10.4f} | {int(row['judge_rank']):^10} | {int(row['fan_rank']):^10} {mark}")
    
    print(f"\n评委评分垫底次数：{judge_bottom_count} / {len(rankings)} 周")
    
    # 打印四种规则对比
    print("\n【四种规则模拟结果对比】")
    print("-" * 90)
    print(f"{'规则':^25} | {'存活周数':^10} | {'淘汰周':^8} | {'进入决赛':^10} | {'最终名次':^10} | {'进入末两位次数':^12}")
    print("-" * 90)
    
    for r in results:
        elim_week = r['eliminated_week'] if r['eliminated_week'] else '-'
        reached = '是' if r['reached_final'] else '否'
        placement = r['final_placement'] if r['final_placement'] else '-'
        print(f"{r['rule_name']:^25} | {r['survival_weeks']:^10} | {str(elim_week):^8} | {reached:^10} | {str(placement):^10} | {r['times_in_bottom_two']:^12}")
    
    print("-" * 90)
    
    # 分析结论
    print("\n【分析结论】")
    
    # 比较四种规则下的结果差异
    placements = [r['final_placement'] for r in results if r['final_placement']]
    survival_weeks = [r['survival_weeks'] for r in results]
    
    if len(set(placements)) == 1:
        print(f"  → 四种规则下，{info['name']} 的最终名次相同（第 {placements[0]} 名）")
    else:
        print(f"  → 四种规则下，{info['name']} 的最终名次不同：")
        for r in results:
            if r['final_placement']:
                print(f"     {r['rule_name']}: 第 {r['final_placement']} 名")
            else:
                print(f"     {r['rule_name']}: 第 {r['eliminated_week']} 周被淘汰")
    
    # 评委决定规则的影响
    r1 = results[0]  # 纯百分比
    r3 = results[2]  # 百分比+评委决定
    r2 = results[1]  # 纯排名
    r4 = results[3]  # 排名+评委决定
    
    if r1['final_placement'] != r3['final_placement'] or r1['eliminated_week'] != r3['eliminated_week']:
        print(f"  → 评委决定规则对百分比法有影响：")
        print(f"     纯百分比法: 存活{r1['survival_weeks']}周, 名次{r1['final_placement']}")
        print(f"     百分比+评委: 存活{r3['survival_weeks']}周, 名次{r3['final_placement']}")
    
    if r2['final_placement'] != r4['final_placement'] or r2['eliminated_week'] != r4['eliminated_week']:
        print(f"  → 评委决定规则对排名法有影响：")
        print(f"     纯排名法: 存活{r2['survival_weeks']}周, 名次{r2['final_placement']}")
        print(f"     排名+评委: 存活{r4['survival_weeks']}周, 名次{r4['final_placement']}")


def generate_summary_table(all_analyses):
    """生成汇总表格"""
    rows = []
    for analysis in all_analyses:
        info = analysis['contestant']
        for r in analysis['results']:
            rows.append({
                'season': info['season'],
                'name': info['name'],
                'actual_placement': info['actual_placement'],
                'rule': r['rule_name'],
                'survival_weeks': r['survival_weeks'],
                'eliminated_week': r['eliminated_week'],
                'reached_final': r['reached_final'],
                'simulated_placement': r['final_placement'],
                'times_in_bottom_two': r['times_in_bottom_two'],
            })
    
    return pd.DataFrame(rows)


def main():
    print("正在加载数据...")
    df = load_data()
    print(f"加载完成: {len(df)} 条记录")
    
    print("\n开始分析四位争议选手...")
    all_analyses = []
    
    for contestant in CONTROVERSIAL_CONTESTANTS:
        analysis = analyze_controversial_contestant(df, contestant)
        if analysis:
            all_analyses.append(analysis)
            print_contestant_analysis(analysis, df)
    
    # 生成汇总表
    print("\n\n" + "=" * 80)
    print("【总体汇总】四种规则对争议选手的影响")
    print("=" * 80)
    
    summary_df = generate_summary_table(all_analyses)
    
    # 打印汇总表
    print("\n【四种规则结果汇总表】")
    print("-" * 100)
    print(f"{'选手':^15} | {'赛季':^6} | {'实际名次':^10} | {'规则':^25} | {'模拟名次':^10} | {'存活周数':^10}")
    print("-" * 100)
    
    for name in summary_df['name'].unique():
        name_df = summary_df[summary_df['name'] == name]
        season = name_df['season'].iloc[0]
        actual = name_df['actual_placement'].iloc[0]
        
        for i, (_, row) in enumerate(name_df.iterrows()):
            if i == 0:
                print(f"{name:^15} | {season:^6} | {actual:^10} | {row['rule']:^25} | {str(row['simulated_placement']):^10} | {row['survival_weeks']:^10}")
            else:
                print(f"{'':^15} | {'':^6} | {'':^10} | {row['rule']:^25} | {str(row['simulated_placement']):^10} | {row['survival_weeks']:^10}")
        print("-" * 100)
    
    # 保存结果
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    summary_df.to_csv(os.path.join(OUTPUT_DIR, 'four_rules_comparison.csv'),
                      index=False, encoding='utf-8-sig')
    print(f"\n结果已保存到 {OUTPUT_DIR}/four_rules_comparison.csv")
    
    # 关键发现
    print("\n【关键发现】")
    print("-" * 60)
    
    # 检查规则差异
    for contestant in CONTROVERSIAL_CONTESTANTS:
        name = contestant['name']
        name_df = summary_df[summary_df['name'] == name]
        placements = name_df['simulated_placement'].dropna().unique()
        
        if len(placements) > 1:
            print(f"• {name}: 不同规则导致不同结果")
            for _, row in name_df.iterrows():
                print(f"    {row['rule']}: 第 {row['simulated_placement']} 名" if row['simulated_placement'] else f"    {row['rule']}: 第 {row['eliminated_week']} 周淘汰")
        else:
            print(f"• {name}: 四种规则下结果一致（第 {placements[0]} 名）" if len(placements) > 0 else f"• {name}: 无有效结果")
    
    return all_analyses, summary_df


if __name__ == '__main__':
    analyses, summary = main()
