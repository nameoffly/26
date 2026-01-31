"""
观众投票估计模型 - 最大熵+平滑性联合正则化版本

目标函数：
    max ∑_{s,t} H_{s,t}(v) - λ ∑_{s,t,i} (v_{s,i,t} - v_{s,i,t-1})²

其中：
    - H_{s,t}(v) = -∑_{i} v_{i,t} log(v_{i,t}) 是每周的熵
    - λ ≥ 0 是平滑性权重参数

等价于最小化：
    min ∑_{s,t} ∑_{i} v_{i,t} log(v_{i,t}) + λ ∑_{t,i} (v_{i,t} - v_{i,t-1})²

约束条件：
    1. ∑_i v_{i,t} = 1  (每周归一化)
    2. v_{i,t} ≥ 0      (非负)
    3. T_e < T_s        (淘汰者总分低于幸存者)

这种方法：
    - 最大熵：让投票分布尽可能均匀，避免极端解
    - 平滑性：让同一选手的投票份额在相邻周之间变化尽量小
    - 联合优化：避免每周独立求解导致的人气剧烈抖动
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

from scipy.optimize import minimize, linprog
from scipy.sparse import csr_matrix


def _compute_feasible_interval_one(
    judge_percents: np.ndarray,
    eliminated_indices: List[int],
    target_index: int,
    epsilon: float = 0.001,
) -> Tuple[float, float]:
    """
    计算单名选手在单周的观众投票可行区间 [v_min, v_max]（仅用 scipy）。
    """
    n = len(judge_percents)
    if n == 0:
        return 0.0, 0.0
    survived_indices = [i for i in range(n) if i not in eliminated_indices]
    A_eq = np.ones((1, n))
    b_eq = np.array([1.0])
    A_ub_list = []
    b_ub_list = []
    for e in eliminated_indices:
        for s in survived_indices:
            row = np.zeros(n)
            row[e], row[s] = 1, -1
            A_ub_list.append(row)
            b_ub_list.append(-(judge_percents[e] - judge_percents[s] + epsilon))
    A_ub = np.array(A_ub_list) if A_ub_list else None
    b_ub = np.array(b_ub_list) if b_ub_list else None
    bounds = [(0, 1)] * n
    v_min, v_max = 0.0, 1.0
    methods = ['highs', 'interior-point', 'simplex']
    for method in methods:
        try:
            r_min = linprog(
                np.array([1.0 if i == target_index else 0.0 for i in range(n)]),
                A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method=method
            )
            if r_min.success:
                v_min = float(r_min.x[target_index])
                break
        except (ValueError, TypeError):
            continue
    for method in methods:
        try:
            c_max = np.zeros(n)
            c_max[target_index] = -1
            r_max = linprog(
                c_max, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method=method
            )
            if r_max.success:
                v_max = float(r_max.x[target_index])
                break
        except (ValueError, TypeError):
            continue
    return max(0.0, v_min), min(1.0, v_max)


def _compute_all_intervals(
    judge_percents: np.ndarray,
    eliminated_indices: List[int],
    epsilon: float = 0.001,
) -> List[Tuple[float, float]]:
    """计算该周所有选手的可行区间。"""
    n = len(judge_percents)
    return [
        _compute_feasible_interval_one(judge_percents, eliminated_indices, i, epsilon)
        for i in range(n)
    ]


class EntropySmoothFanVoteEstimator:
    """基于最大熵+平滑性联合正则化的观众投票估计器"""
    
    def __init__(self, epsilon: float = 0.001, lambda_smooth: float = 1.0):
        """
        初始化估计器
        
        Args:
            epsilon: 严格不等式的小量
            lambda_smooth: 平滑性惩罚的权重 (λ)
        """
        self.epsilon = epsilon
        self.lambda_smooth = lambda_smooth
    
    def _negative_entropy(self, v: np.ndarray) -> float:
        """
        计算负熵 ∑ v_i * log(v_i)
        
        注意：为了数值稳定性，对于v_i接近0的情况，使用v_i*log(v_i+eps)
        """
        eps = 1e-12
        v_safe = np.maximum(v, eps)
        return np.sum(v_safe * np.log(v_safe))
    
    def _negative_entropy_grad(self, v: np.ndarray) -> np.ndarray:
        """
        负熵的梯度: 1 + log(v_i)
        """
        eps = 1e-12
        v_safe = np.maximum(v, eps)
        return 1 + np.log(v_safe)
    
    def estimate_season(self, season_data: List[Dict]) -> List[Dict]:
        """
        对一个季度的所有周进行联合优化
        
        使用最大熵+平滑性联合正则化
        
        Args:
            season_data: 每周数据的列表，每个元素包含:
                - week: 周数
                - contestant_names: 选手名列表
                - judge_percents: 评委百分比数组
                - eliminated_indices: 淘汰者索引
                
        Returns:
            每周结果的列表
        """
        if not season_data:
            return []
        
        # 收集所有变量信息
        week_var_info = []
        total_vars = 0
        all_contestants = set()
        
        for t, week_data in enumerate(season_data):
            n_this_week = len(week_data['contestant_names'])
            week_var_info.append({
                'start_idx': total_vars,
                'n_vars': n_this_week,
                'names': week_data['contestant_names'],
                'judge_percents': week_data['judge_percents'],
                'eliminated_indices': week_data['eliminated_indices'],
                'week': week_data['week']
            })
            total_vars += n_this_week
            all_contestants.update(week_data['contestant_names'])
        
        all_contestants = sorted(list(all_contestants))
        
        # 构建选手在各周的变量索引映射
        contestant_var_map = {name: [] for name in all_contestants}
        for t, info in enumerate(week_var_info):
            for local_idx, name in enumerate(info['names']):
                global_idx = info['start_idx'] + local_idx
                contestant_var_map[name].append((info['week'], global_idx))
        
        # 对每个选手按周排序
        for name in contestant_var_map:
            contestant_var_map[name].sort(key=lambda x: x[0])
        
        # 定义目标函数：负熵 + λ * 平滑性惩罚
        def objective(v):
            # 负熵项
            neg_entropy = 0
            for info in week_var_info:
                start = info['start_idx']
                n = info['n_vars']
                v_week = v[start:start+n]
                neg_entropy += self._negative_entropy(v_week)
            
            # 平滑性项
            smoothness = 0
            for name, indices in contestant_var_map.items():
                if len(indices) > 1:
                    for i in range(len(indices) - 1):
                        idx1 = indices[i][1]
                        idx2 = indices[i+1][1]
                        smoothness += (v[idx2] - v[idx1]) ** 2
            
            return neg_entropy + self.lambda_smooth * smoothness
        
        # 定义目标函数的梯度
        def objective_grad(v):
            grad = np.zeros(total_vars)
            
            # 负熵项的梯度
            for info in week_var_info:
                start = info['start_idx']
                n = info['n_vars']
                v_week = v[start:start+n]
                grad[start:start+n] = self._negative_entropy_grad(v_week)
            
            # 平滑性项的梯度
            for name, indices in contestant_var_map.items():
                if len(indices) > 1:
                    for i in range(len(indices) - 1):
                        idx1 = indices[i][1]
                        idx2 = indices[i+1][1]
                        diff = v[idx2] - v[idx1]
                        grad[idx1] += -2 * self.lambda_smooth * diff
                        grad[idx2] += 2 * self.lambda_smooth * diff
            
            return grad
        
        # 构建约束
        constraints = []
        
        # 等式约束: 每周sum(v) = 1
        for info in week_var_info:
            start = info['start_idx']
            n = info['n_vars']
            
            def eq_constraint(v, start=start, n=n):
                return np.sum(v[start:start+n]) - 1.0
            
            def eq_constraint_jac(v, start=start, n=n):
                jac = np.zeros(total_vars)
                jac[start:start+n] = 1.0
                return jac
            
            constraints.append({
                'type': 'eq',
                'fun': eq_constraint,
                'jac': eq_constraint_jac
            })
        
        # 不等式约束: 淘汰约束 v_s - v_e >= j_e - j_s + epsilon
        # 转换为 -(v_s - v_e) + (j_e - j_s + epsilon) <= 0
        for info in week_var_info:
            start = info['start_idx']
            n = info['n_vars']
            judge_percents = info['judge_percents']
            eliminated = info['eliminated_indices']
            survived = [i for i in range(n) if i not in eliminated]
            
            for e in eliminated:
                for s in survived:
                    diff = judge_percents[e] - judge_percents[s] + self.epsilon
                    
                    def ineq_constraint(v, start=start, e=e, s=s, diff=diff):
                        # v_s - v_e >= diff
                        # 转换为 -(v_s - v_e) + diff <= 0
                        return -(v[start+s] - v[start+e]) + diff
                    
                    def ineq_constraint_jac(v, start=start, e=e, s=s):
                        jac = np.zeros(total_vars)
                        jac[start+s] = -1
                        jac[start+e] = 1
                        return jac
                    
                    constraints.append({
                        'type': 'ineq',
                        'fun': lambda v, f=ineq_constraint: -f(v),  # scipy的ineq是>=0
                        'jac': lambda v, j=ineq_constraint_jac: -j(v)
                    })
        
        # 变量边界: v_i >= 0
        bounds = [(1e-10, 1.0) for _ in range(total_vars)]  # 下界用小正数避免log(0)
        
        # 初始值: 均匀分布
        v0 = []
        for info in week_var_info:
            v0.extend([1.0 / info['n_vars']] * info['n_vars'])
        v0 = np.array(v0)
        
        # 求解优化问题
        try:
            result = minimize(
                objective, v0,
                method='SLSQP',
                jac=objective_grad,
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 3000, 'ftol': 1e-9}
            )
            solve_status = "optimal" if result.success else result.message
            fan_votes = result.x
        except Exception as e:
            solve_status = str(e)
            fan_votes = v0
        
        # 格式化结果
        return self._format_results(week_var_info, fan_votes, solve_status, 
                                   contestant_var_map, all_contestants)
    
    def _format_results(self, week_var_info, fan_votes, solve_status, 
                       contestant_var_map, all_contestants):
        """格式化结果"""
        results = []
        
        # 计算每个选手的平均熵贡献和平滑度
        contestant_stats = {}
        for name in all_contestants:
            indices = contestant_var_map[name]
            if len(indices) > 1:
                smoothness = 0
                for i in range(len(indices) - 1):
                    idx1 = indices[i][1]
                    idx2 = indices[i+1][1]
                    smoothness += (fan_votes[idx2] - fan_votes[idx1]) ** 2
                contestant_stats[name] = {
                    'smoothness': smoothness / (len(indices) - 1),
                    'n_weeks': len(indices)
                }
            else:
                contestant_stats[name] = {
                    'smoothness': 0,
                    'n_weeks': 1
                }
        
        for info in week_var_info:
            start = info['start_idx']
            n = info['n_vars']
            names = info['names']
            judge_percents = info['judge_percents']
            eliminated = info['eliminated_indices']
            
            week_fan_votes = fan_votes[start:start+n]
            
            # 计算该周每位选手的可行区间
            intervals = _compute_all_intervals(judge_percents, eliminated, self.epsilon)
            
            # 计算该周的熵
            eps = 1e-12
            v_safe = np.maximum(week_fan_votes, eps)
            week_entropy = -np.sum(v_safe * np.log(v_safe))
            
            week_results = {
                'week': info['week'],
                'solve_status': solve_status,
                'week_entropy': week_entropy,
                'contestants': []
            }
            
            for i, name in enumerate(names):
                fan_vote = week_fan_votes[i]
                v_min, v_max = intervals[i]
                interval_width = v_max - v_min
                
                week_results['contestants'].append({
                    'name': name,
                    'judge_percent': judge_percents[i],
                    'fan_vote_percent': float(fan_vote),
                    'total_percent': float(judge_percents[i] + fan_vote),
                    'fan_vote_min': v_min,
                    'fan_vote_max': v_max,
                    'interval_width': interval_width,
                    'smoothness': contestant_stats[name]['smoothness'],
                    'n_weeks': contestant_stats[name]['n_weeks'],
                    'eliminated': i in eliminated
                })
            
            results.append(week_results)
        
        return results


class DWTSProcessedDataProcessor:
    """Dancing with the Stars 处理后数据的处理器"""
    
    def __init__(self, excel_path: str):
        self.excel_path = excel_path
        self.df = None
        self.load_data()
    
    def load_data(self):
        self.df = pd.read_excel(self.excel_path)
        print(f"数据加载完成：{len(self.df)} 位选手")
    
    def get_season_data(self, season: int) -> pd.DataFrame:
        return self.df[self.df['season'] == season].copy()
    
    def get_max_week(self, season_df: pd.DataFrame) -> int:
        max_week = 1
        for week in range(1, 12):
            col = f'{week}_percent'
            if col in season_df.columns:
                if (season_df[col] > 0).any():
                    max_week = week
        return max_week
    
    def get_week_contestants(self, season_df: pd.DataFrame, week: int) -> Tuple[List[str], np.ndarray]:
        col = f'{week}_percent'
        if col not in season_df.columns:
            return [], np.array([])
        
        mask = season_df[col] > 0
        contestants_df = season_df[mask]
        
        names = contestants_df['celebrity_name'].tolist()
        percents = contestants_df[col].values
        
        return names, percents
    
    def get_eliminated_this_week(self, season_df: pd.DataFrame, week: int) -> List[str]:
        eliminated = []
        for _, row in season_df.iterrows():
            result = str(row['results']).lower()
            if f'eliminated week {week}' in result:
                eliminated.append(row['celebrity_name'])
        return eliminated
    
    def process_season(self, season: int) -> List[Dict]:
        """处理一个季度的所有周数据"""
        season_df = self.get_season_data(season)
        if len(season_df) == 0:
            return []
        
        max_week = self.get_max_week(season_df)
        season_data = []
        
        for week in range(1, max_week + 1):
            contestant_names, judge_percents = self.get_week_contestants(season_df, week)
            
            if len(contestant_names) == 0:
                continue
            
            eliminated = self.get_eliminated_this_week(season_df, week)
            name_to_idx = {name: i for i, name in enumerate(contestant_names)}
            eliminated_indices = [name_to_idx[name] for name in eliminated if name in name_to_idx]
            
            season_data.append({
                'week': week,
                'contestant_names': contestant_names,
                'judge_percents': judge_percents,
                'eliminated': eliminated,
                'eliminated_indices': eliminated_indices,
                'n_contestants': len(contestant_names)
            })
        
        return season_data


def run_estimation(excel_path: str, output_path: str = None, lambda_smooth: float = 1.0):
    """
    运行完整的观众投票估计（最大熵+平滑性联合正则化版本）
    
    Args:
        excel_path: 数据文件路径
        output_path: 输出文件路径
        lambda_smooth: 平滑性权重参数
    """
    processor = DWTSProcessedDataProcessor(excel_path)
    estimator = EntropySmoothFanVoteEstimator(epsilon=0.001, lambda_smooth=lambda_smooth)
    
    all_results = []
    
    print(f"\n使用参数: epsilon=0.001, lambda_smooth={lambda_smooth}")
    
    # 遍历第3-27季
    for season in range(3, 28):
        print(f"\n{'='*60}")
        print(f"处理第 {season} 季...")
        
        season_data = processor.process_season(season)
        if not season_data:
            print(f"  第 {season} 季没有数据")
            continue
        
        print(f"  该季共 {len(season_data)} 周")
        
        # 使用联合优化
        week_results = estimator.estimate_season(season_data)
        
        for week_result in week_results:
            week = week_result['week']
            status = week_result['solve_status']
            entropy = week_result['week_entropy']
            contestants = week_result['contestants']
            
            eliminated_names = [c['name'] for c in contestants if c['eliminated']]
            
            print(f"\n  第 {week} 周: {len(contestants)} 位选手, 熵={entropy:.4f}", end="")
            if eliminated_names:
                print(f", 淘汰: {eliminated_names}", end="")
            print()
            print(f"    求解状态: {status}")
            
            for c in contestants:
                elim_mark = " [淘汰]" if c['eliminated'] else ""
                smooth_info = f"平滑度={c['smoothness']:.6f}" if c['n_weeks'] > 1 else "首周"
                print(f"      {c['name']}: 评委{c['judge_percent']*100:.2f}% + "
                      f"观众{c['fan_vote_percent']*100:.2f}% = {c['total_percent']*100:.2f}% "
                      f"({smooth_info}){elim_mark}")
                
                all_results.append({
                    'season': season,
                    'week': week,
                    'celebrity_name': c['name'],
                    'judge_percent': c['judge_percent'],
                    'fan_vote_percent': c['fan_vote_percent'],
                    'total_percent': c['total_percent'],
                    'fan_vote_min': c['fan_vote_min'],
                    'fan_vote_max': c['fan_vote_max'],
                    'interval_width': c['interval_width'],
                    'smoothness': c['smoothness'],
                    'n_weeks': c['n_weeks'],
                    'week_entropy': entropy,
                    'eliminated': c['eliminated'],
                    'solve_status': status
                })
    
    # 保存结果
    if output_path and all_results:
        results_df = pd.DataFrame(all_results)
        results_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"\n结果已保存到: {output_path}")
    
    return all_results


def validate_results(results: List[Dict]) -> Dict:
    """验证估计结果"""
    if not results:
        return {'error': 'no results'}
    
    df = pd.DataFrame(results)
    
    total_weeks = df.groupby(['season', 'week']).ngroups
    
    # 验证淘汰者约束
    violations = 0
    total_checks = 0
    margins = []
    
    for (season, week), group in df.groupby(['season', 'week']):
        eliminated = group[group['eliminated'] == True]
        survived = group[group['eliminated'] == False]
        
        if len(eliminated) > 0 and len(survived) > 0:
            max_elim_total = eliminated['total_percent'].max()
            min_surv_total = survived['total_percent'].min()
            
            margin = min_surv_total - max_elim_total
            margins.append(margin)
            
            if max_elim_total >= min_surv_total:
                violations += 1
            total_checks += 1
    
    # 计算平滑性指标
    smoothness_scores = []
    for (season, name), group in df.groupby(['season', 'celebrity_name']):
        if len(group) > 1:
            group_sorted = group.sort_values('week')
            diffs = np.diff(group_sorted['fan_vote_percent'].values)
            smoothness_scores.append(np.mean(np.abs(diffs)))
    
    validation = {
        'total_seasons': df['season'].nunique(),
        'total_weeks': total_weeks,
        'total_contestants_estimates': len(df),
        'constraint_checks': total_checks,
        'constraint_violations': violations,
        'constraint_satisfaction_rate': 1 - violations/total_checks if total_checks > 0 else 1.0,
        'avg_margin': np.mean(margins) if margins else 0,
        'min_margin': np.min(margins) if margins else 0,
        'avg_entropy': df['week_entropy'].mean(),
        'avg_smoothness': np.mean(smoothness_scores) if smoothness_scores else 0,
    }
    
    return validation


def analyze_controversies(results: List[Dict]):
    """分析争议性选手"""
    if not results:
        return
    
    df = pd.DataFrame(results)
    
    print("\n" + "="*60)
    print("争议性选手分析 (最大熵+平滑性联合正则化)")
    print("="*60)
    
    # Bobby Bones (Season 27)
    print("\n【第27季 - Bobby Bones】")
    s27 = df[df['season'] == 27]
    bobby = s27[s27['celebrity_name'] == 'Bobby Bones'].sort_values('week')
    if len(bobby) > 0:
        print("周次 | 评委% | 观众%(估计) | 平滑度 | 总%")
        print("-" * 55)
        for _, row in bobby.iterrows():
            smooth = f"{row['smoothness']:.6f}" if row['n_weeks'] > 1 else "首周"
            print(f"  {int(row['week'])}  | {row['judge_percent']*100:5.2f}% | "
                  f"{row['fan_vote_percent']*100:5.2f}% | "
                  f"{smooth:>10s} | "
                  f"{row['total_percent']*100:5.2f}%")
        
        # 计算平均观众投票
        avg_fan = bobby['fan_vote_percent'].mean()
        std_fan = bobby['fan_vote_percent'].std()
        print(f"  平均观众投票: {avg_fan*100:.2f}% ± {std_fan*100:.2f}%")
    
    # Bristol Palin (Season 11)
    print("\n【第11季 - Bristol Palin】")
    s11 = df[df['season'] == 11]
    bristol = s11[s11['celebrity_name'] == 'Bristol Palin'].sort_values('week')
    if len(bristol) > 0:
        print("周次 | 评委% | 观众%(估计) | 平滑度 | 总%")
        print("-" * 55)
        for _, row in bristol.iterrows():
            elim = " [淘汰]" if row['eliminated'] else ""
            smooth = f"{row['smoothness']:.6f}" if row['n_weeks'] > 1 else "首周"
            print(f"  {int(row['week'])}  | {row['judge_percent']*100:5.2f}% | "
                  f"{row['fan_vote_percent']*100:5.2f}% | "
                  f"{smooth:>10s} | "
                  f"{row['total_percent']*100:5.2f}%{elim}")
        
        avg_fan = bristol['fan_vote_percent'].mean()
        std_fan = bristol['fan_vote_percent'].std()
        print(f"  平均观众投票: {avg_fan*100:.2f}% ± {std_fan*100:.2f}%")


def compare_lambda_values(excel_path: str, lambda_values: List[float] = [0.1, 1.0, 10.0]):
    """比较不同λ值的效果"""
    print("\n" + "="*60)
    print("不同λ值的效果比较")
    print("="*60)
    
    for lam in lambda_values:
        print(f"\n--- λ = {lam} ---")
        results = run_estimation(excel_path, output_path=None, lambda_smooth=lam)
        
        if results:
            validation = validate_results(results)
            print(f"\n验证结果 (λ={lam}):")
            print(f"  约束满足率: {validation['constraint_satisfaction_rate']*100:.2f}%")
            print(f"  平均边际余量: {validation['avg_margin']*100:.4f}%")
            print(f"  平均熵: {validation['avg_entropy']:.4f}")
            print(f"  平均平滑度: {validation['avg_smoothness']:.6f}")


if __name__ == "__main__":
    excel_path = r"d:\Users\13016\Desktop\26MCM\2026_C\2026_MCM_Problem_C_Processed_Data.xlsx"
    output_path = r"d:\Users\13016\Desktop\26MCM\2026_C\task1-1\fan_vote_estimates_entropy_smooth_100.csv"
    
    # 默认λ=10.0，更强调时间连续性
    lambda_smooth = 100.0
    
    # 运行估计
    results = run_estimation(excel_path, output_path, lambda_smooth=lambda_smooth)
    
    # 验证结果
    if results:
        validation = validate_results(results)
        print(f"\n{'='*60}")
        print("验证结果:")
        for key, value in validation.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.6f}")
            else:
                print(f"  {key}: {value}")
        
        # 分析争议性选手
        analyze_controversies(results)
