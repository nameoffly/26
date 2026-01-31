"""
观众投票估计模型
使用线性约束优化+最小方差正则化方法
针对Dancing with the Stars第3-27季

模型描述：
- 决策变量：每位选手的观众投票百分比 v_i
- 约束条件：
  1. 归一化约束：sum(v_i) = 1
  2. 非负性约束：v_i >= 0
  3. 淘汰者约束：被淘汰者的总百分比（评委+观众）应低于幸存者
- 目标函数：最小化投票百分比的方差（最小方差正则化）
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
import warnings
import sys

# 设置UTF-8编码
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except:
        pass

warnings.filterwarnings('ignore')

# 尝试导入cvxpy，如果失败则使用scipy
try:
    import cvxpy as cp
    USE_CVXPY = True
except ImportError:
    from scipy.optimize import minimize, Bounds
    USE_CVXPY = False
    print("Warning: cvxpy not found, using scipy.optimize instead")


class FanVoteEstimator:
    """观众投票估计器类"""
    
    def __init__(self, epsilon: float = 0.001):
        """
        初始化估计器
        
        Args:
            epsilon: 严格不等式的小量，用于处理 T_e < T_s 约束
        """
        self.epsilon = epsilon
        self.results = {}
    
    def estimate_fan_votes_cvxpy(
        self, 
        judge_percents: np.ndarray, 
        eliminated_indices: List[int],
        is_finale: bool = False,
        final_rankings: Optional[List[int]] = None
    ) -> Tuple[Optional[np.ndarray], str]:
        """
        使用cvxpy求解观众投票百分比
        
        Args:
            judge_percents: 各选手评委评分百分比数组
            eliminated_indices: 被淘汰选手的索引列表
            is_finale: 是否为决赛周
            final_rankings: 决赛排名（索引列表，从冠军到末位）
            
        Returns:
            (fan_vote_percents, status): 估计的观众投票百分比和状态
        """
        n = len(judge_percents)
        
        if n == 0:
            return None, "no_contestants"
        
        # 决策变量
        v = cp.Variable(n)
        
        # 目标函数：最小化二次范数（等价于最小化方差）
        objective = cp.Minimize(cp.sum_squares(v))
        
        # 约束条件
        constraints = [
            cp.sum(v) == 1,  # 归一化
            v >= 0,          # 非负
        ]
        
        # 淘汰约束
        survived_indices = [i for i in range(n) if i not in eliminated_indices]
        
        for e in eliminated_indices:
            for s in survived_indices:
                # v_s - v_e >= j_e - j_s + epsilon
                # 即 T_s > T_e (总分：幸存者 > 淘汰者)
                diff = judge_percents[e] - judge_percents[s] + self.epsilon
                constraints.append(v[s] - v[e] >= diff)
        
        # 决赛排名约束
        if is_finale and final_rankings is not None and len(final_rankings) > 1:
            for i in range(len(final_rankings) - 1):
                higher = final_rankings[i]   # 排名更高的选手
                lower = final_rankings[i+1]  # 排名更低的选手
                # T_higher > T_lower
                diff = judge_percents[lower] - judge_percents[higher] + self.epsilon
                constraints.append(v[higher] - v[lower] >= diff)
        
        # 求解
        prob = cp.Problem(objective, constraints)
        try:
            prob.solve(solver=cp.OSQP, verbose=False)
            
            if prob.status == cp.OPTIMAL or prob.status == cp.OPTIMAL_INACCURATE:
                return v.value, "optimal"
            else:
                # 尝试其他求解器
                prob.solve(solver=cp.SCS, verbose=False)
                if prob.status == cp.OPTIMAL or prob.status == cp.OPTIMAL_INACCURATE:
                    return v.value, "optimal"
                return None, prob.status
        except Exception as e:
            return None, str(e)
    
    def estimate_fan_votes_scipy(
        self, 
        judge_percents: np.ndarray, 
        eliminated_indices: List[int]
    ) -> Tuple[Optional[np.ndarray], str]:
        """
        使用scipy求解观众投票百分比（备用方法）
        
        Args:
            judge_percents: 各选手评委评分百分比数组
            eliminated_indices: 被淘汰选手的索引列表
            
        Returns:
            (fan_vote_percents, status): 估计的观众投票百分比和状态
        """
        n = len(judge_percents)
        
        if n == 0:
            return None, "no_contestants"
        
        # 目标函数：最小化方差
        def objective(v):
            return np.sum(v**2)
        
        def objective_grad(v):
            return 2 * v
        
        # 约束条件列表
        constraints = []
        
        # 等式约束：投票之和为1
        constraints.append({
            'type': 'eq', 
            'fun': lambda v: np.sum(v) - 1
        })
        
        # 不等式约束：淘汰者总分 < 幸存者总分
        survived_indices = [i for i in range(n) if i not in eliminated_indices]
        
        for e in eliminated_indices:
            for s in survived_indices:
                diff = judge_percents[e] - judge_percents[s] + self.epsilon
                # 创建闭包时需要捕获当前值
                def make_constraint(s_idx, e_idx, d):
                    def constraint(v):
                        return v[s_idx] - v[e_idx] - d
                    return constraint
                constraints.append({
                    'type': 'ineq',
                    'fun': make_constraint(s, e, diff)
                })
        
        # 边界：非负且不超过1
        bounds = Bounds(lb=np.zeros(n), ub=np.ones(n))
        
        # 初始值：均匀分布
        v0 = np.ones(n) / n
        
        try:
            result = minimize(
                objective, v0, 
                method='SLSQP',
                jac=objective_grad,
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 1000}
            )
            
            if result.success:
                return result.x, "optimal"
            else:
                return None, result.message
        except Exception as e:
            return None, str(e)
    
    def estimate(
        self, 
        judge_percents: np.ndarray, 
        eliminated_indices: List[int],
        is_finale: bool = False,
        final_rankings: Optional[List[int]] = None
    ) -> Tuple[Optional[np.ndarray], str]:
        """
        估计观众投票百分比（自动选择求解器）
        
        Args:
            judge_percents: 各选手评委评分百分比数组
            eliminated_indices: 被淘汰选手的索引列表
            is_finale: 是否为决赛周
            final_rankings: 决赛排名
            
        Returns:
            (fan_vote_percents, status): 估计的观众投票百分比和状态
        """
        if USE_CVXPY:
            return self.estimate_fan_votes_cvxpy(
                judge_percents, eliminated_indices, is_finale, final_rankings
            )
        else:
            return self.estimate_fan_votes_scipy(
                judge_percents, eliminated_indices
            )


class DWTSProcessedDataProcessor:
    """Dancing with the Stars 处理后数据的处理器"""
    
    def __init__(self, excel_path: str):
        """
        初始化数据处理器
        
        Args:
            excel_path: 处理后的Excel数据文件路径
        """
        self.excel_path = excel_path
        self.df = None
        self.load_data()
    
    def load_data(self):
        """加载Excel数据"""
        self.df = pd.read_excel(self.excel_path)
        print(f"数据加载完成：{len(self.df)} 位选手")
    
    def get_season_data(self, season: int) -> pd.DataFrame:
        """获取指定季度的数据"""
        return self.df[self.df['season'] == season].copy()
    
    def get_max_week(self, season_df: pd.DataFrame) -> int:
        """获取该季度的最大周数"""
        max_week = 1
        for week in range(1, 12):  # 最多11周
            col = f'{week}_percent'
            if col in season_df.columns:
                # 检查是否有有效百分比（大于0）
                valid = (season_df[col] > 0).any()
                if valid:
                    max_week = week
        return max_week
    
    def get_week_contestants(self, season_df: pd.DataFrame, week: int) -> Tuple[List[str], np.ndarray]:
        """
        获取指定周还在比赛的选手及其评委百分比
        
        Args:
            season_df: 季度数据
            week: 周数
            
        Returns:
            (选手名列表, 评委百分比数组)
        """
        col = f'{week}_percent'
        if col not in season_df.columns:
            return [], np.array([])
        
        # 筛选该周有有效百分比的选手（大于0）
        mask = season_df[col] > 0
        contestants_df = season_df[mask]
        
        names = contestants_df['celebrity_name'].tolist()
        percents = contestants_df[col].values
        
        return names, percents
    
    def get_eliminated_this_week(self, season_df: pd.DataFrame, week: int) -> List[str]:
        """
        获取本周被淘汰的选手
        
        Args:
            season_df: 季度数据
            week: 周数
            
        Returns:
            被淘汰选手名字列表
        """
        eliminated = []
        
        for _, row in season_df.iterrows():
            result = str(row['results']).lower()
            
            # 检查是否在本周被淘汰
            if f'eliminated week {week}' in result:
                eliminated.append(row['celebrity_name'])
        
        return eliminated
    
    def get_finale_rankings(self, season_df: pd.DataFrame, week: int) -> List[str]:
        """
        获取决赛排名（如果是决赛周）
        
        Args:
            season_df: 季度数据
            week: 周数
            
        Returns:
            按名次排列的选手名字列表（冠军在前）
        """
        # 获取本周选手
        names, _ = self.get_week_contestants(season_df, week)
        if not names:
            return []
        
        # 筛选本周选手并按placement排序
        contestants = season_df[season_df['celebrity_name'].isin(names)]
        contestants = contestants.sort_values('placement')
        
        return contestants['celebrity_name'].tolist()
    
    def is_finale_week(self, season_df: pd.DataFrame, week: int) -> bool:
        """判断是否为决赛周"""
        names, _ = self.get_week_contestants(season_df, week)
        
        # 通常决赛有2-4位选手
        if 2 <= len(names) <= 4:
            # 检查下一周是否还有比赛
            next_names, _ = self.get_week_contestants(season_df, week + 1)
            if len(next_names) == 0:
                return True
        
        return False
    
    def process_week(self, season: int, week: int) -> Optional[Dict]:
        """
        处理指定季度和周的数据
        
        Args:
            season: 季度
            week: 周数
            
        Returns:
            包含选手信息、评委百分比和淘汰信息的字典
        """
        season_df = self.get_season_data(season)
        contestant_names, judge_percents = self.get_week_contestants(season_df, week)
        
        if len(contestant_names) == 0:
            return None
        
        eliminated = self.get_eliminated_this_week(season_df, week)
        is_finale = self.is_finale_week(season_df, week)
        finale_rankings = self.get_finale_rankings(season_df, week) if is_finale else None
        
        # 按选手名创建索引映射
        name_to_idx = {name: i for i, name in enumerate(contestant_names)}
        
        # 获取淘汰者索引
        eliminated_indices = [name_to_idx[name] for name in eliminated if name in name_to_idx]
        
        return {
            'season': season,
            'week': week,
            'contestant_names': contestant_names,
            'judge_percents': judge_percents,
            'eliminated': eliminated,
            'eliminated_indices': eliminated_indices,
            'is_finale': is_finale,
            'finale_rankings': finale_rankings,
            'n_contestants': len(contestant_names)
        }


def run_estimation(excel_path: str, output_path: str = None):
    """
    运行完整的观众投票估计
    
    Args:
        excel_path: 处理后的Excel数据路径
        output_path: 结果输出路径（可选）
    """
    # 初始化
    processor = DWTSProcessedDataProcessor(excel_path)
    estimator = FanVoteEstimator(epsilon=0.001)
    
    all_results = []
    
    # 遍历第3-27季（使用百分比方法的季度）
    for season in range(3, 28):
        print(f"\n{'='*50}")
        print(f"处理第 {season} 季...")
        
        season_df = processor.get_season_data(season)
        if len(season_df) == 0:
            print(f"  第 {season} 季没有数据")
            continue
        
        max_week = processor.get_max_week(season_df)
        print(f"  该季共 {max_week} 周")
        
        for week in range(1, max_week + 1):
            week_data = processor.process_week(season, week)
            
            if week_data is None or week_data['n_contestants'] == 0:
                continue
            
            print(f"\n  第 {week} 周: {week_data['n_contestants']} 位选手", end="")
            if week_data['eliminated']:
                print(f", 淘汰: {week_data['eliminated']}", end="")
            if week_data['is_finale']:
                print(" [决赛]", end="")
            print()
            
            # 决赛排名索引
            finale_ranking_indices = None
            if week_data['is_finale'] and week_data['finale_rankings']:
                name_to_idx = {name: i for i, name in enumerate(week_data['contestant_names'])}
                finale_ranking_indices = [
                    name_to_idx[name] 
                    for name in week_data['finale_rankings'] 
                    if name in name_to_idx
                ]
            
            # 估计观众投票
            fan_votes, status = estimator.estimate(
                week_data['judge_percents'],
                week_data['eliminated_indices'],
                week_data['is_finale'],
                finale_ranking_indices
            )
            
            if fan_votes is not None:
                print(f"    求解状态: {status}")
                
                # 计算总百分比
                total_percents = week_data['judge_percents'] + fan_votes
                
                # 显示结果
                for i, name in enumerate(week_data['contestant_names']):
                    judge_pct = week_data['judge_percents'][i] * 100
                    fan_pct = fan_votes[i] * 100
                    total_pct = total_percents[i] * 100
                    elim_mark = " [淘汰]" if name in week_data['eliminated'] else ""
                    print(f"      {name}: 评委{judge_pct:.2f}% + 观众{fan_pct:.2f}% = {total_pct:.2f}%{elim_mark}")
                
                # 保存结果
                for i, name in enumerate(week_data['contestant_names']):
                    all_results.append({
                        'season': season,
                        'week': week,
                        'celebrity_name': name,
                        'judge_percent': week_data['judge_percents'][i],
                        'fan_vote_percent': fan_votes[i],
                        'total_percent': total_percents[i],
                        'eliminated': name in week_data['eliminated'],
                        'is_finale': week_data['is_finale'],
                        'solve_status': status
                    })
            else:
                print(f"    求解失败: {status}")
    
    # 保存结果
    if output_path and all_results:
        results_df = pd.DataFrame(all_results)
        results_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"\n结果已保存到: {output_path}")
    
    return all_results


def validate_results(results: List[Dict]) -> Dict:
    """
    验证估计结果的一致性
    
    Args:
        results: 估计结果列表
        
    Returns:
        验证统计信息
    """
    if not results:
        return {'error': 'no results'}
    
    df = pd.DataFrame(results)
    
    # 统计信息
    total_weeks = df.groupby(['season', 'week']).ngroups
    
    # 验证淘汰者约束
    violations = 0
    total_checks = 0
    
    for (season, week), group in df.groupby(['season', 'week']):
        eliminated = group[group['eliminated'] == True]
        survived = group[group['eliminated'] == False]
        
        if len(eliminated) > 0 and len(survived) > 0:
            max_elim_total = eliminated['total_percent'].max()
            min_surv_total = survived['total_percent'].min()
            
            if max_elim_total >= min_surv_total:
                violations += 1
            total_checks += 1
    
    validation = {
        'total_seasons': df['season'].nunique(),
        'total_weeks': total_weeks,
        'total_contestants_estimates': len(df),
        'constraint_checks': total_checks,
        'constraint_violations': violations,
        'constraint_satisfaction_rate': 1 - violations/total_checks if total_checks > 0 else 1.0
    }
    
    return validation


def calculate_confidence(results: List[Dict]) -> pd.DataFrame:
    """
    计算每个估计的置信度
    
    置信度基于约束的松紧程度：
    - 如果淘汰者和幸存者的总分差距大，置信度高
    - 如果差距小（接近边界），置信度低
    
    Args:
        results: 估计结果列表
        
    Returns:
        带有置信度的DataFrame
    """
    if not results:
        return pd.DataFrame()
    
    df = pd.DataFrame(results)
    
    confidence_scores = []
    
    for (season, week), group in df.groupby(['season', 'week']):
        eliminated = group[group['eliminated'] == True]
        survived = group[group['eliminated'] == False]
        
        if len(eliminated) > 0 and len(survived) > 0:
            max_elim = eliminated['total_percent'].max()
            min_surv = survived['total_percent'].min()
            margin = min_surv - max_elim
            
            # 置信度：基于margin的大小，margin越大置信度越高
            # 使用sigmoid函数映射到[0,1]区间
            confidence = 1 / (1 + np.exp(-margin * 50))  # 缩放因子50
        else:
            # 无淘汰或无幸存者时，置信度较低（解不唯一）
            confidence = 0.5
        
        for idx in group.index:
            confidence_scores.append({
                'index': idx,
                'confidence': confidence
            })
    
    conf_df = pd.DataFrame(confidence_scores).set_index('index')
    df = df.join(conf_df)
    
    return df


if __name__ == "__main__":
    # 设置路径 - 使用处理后的Excel文件
    excel_path = r"d:\Users\13016\Desktop\26MCM\2026_C\2026_MCM_Problem_C_Processed_Data.xlsx"
    output_path = r"d:\Users\13016\Desktop\26MCM\2026_C\fan_vote_estimates.csv"
    
    # 运行估计
    results = run_estimation(excel_path, output_path)
    
    # 验证结果
    if results:
        validation = validate_results(results)
        print(f"\n{'='*50}")
        print("验证结果:")
        for key, value in validation.items():
            print(f"  {key}: {value}")
        
        # 计算置信度
        results_with_conf = calculate_confidence(results)
        if len(results_with_conf) > 0:
            # 保存带置信度的结果
            output_with_conf = r"d:\Users\13016\Desktop\26MCM\2026_C\fan_vote_estimates_with_confidence.csv"
            results_with_conf.to_csv(output_with_conf, index=False, encoding='utf-8-sig')
            print(f"\n带置信度的结果已保存到: {output_with_conf}")
            
            # 显示置信度统计
            print(f"\n置信度统计:")
            print(f"  平均置信度: {results_with_conf['confidence'].mean():.4f}")
            print(f"  最低置信度: {results_with_conf['confidence'].min():.4f}")
            print(f"  最高置信度: {results_with_conf['confidence'].max():.4f}")
