"""
观众投票估计模型 - 平滑性正则化版本

方法：
1. 首先求解每位选手每周观众投票百分比的可行区间 [v_min, v_max]
2. 然后使用平滑性原则：最小化选手在相邻周之间投票百分比的变化

平滑性假设：选手的粉丝基础在相邻周之间变化较小
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
    print("使用 cvxpy 求解器")
except ImportError:
    USE_CVXPY = False
    print("cvxpy 未安装，使用 scipy 求解器")

from scipy.optimize import minimize, linprog, Bounds, LinearConstraint


class FanVoteIntervalEstimator:
    """观众投票可行区间估计器"""
    
    def __init__(self, epsilon: float = 0.001):
        """
        初始化估计器
        
        Args:
            epsilon: 严格不等式的小量
        """
        self.epsilon = epsilon
    
    def compute_feasible_interval_cvxpy(
        self, 
        judge_percents: np.ndarray, 
        eliminated_indices: List[int],
        target_index: int
    ) -> Tuple[float, float]:
        """使用cvxpy计算可行区间"""
        n = len(judge_percents)
        
        if n == 0:
            return 0, 0
        
        survived_indices = [i for i in range(n) if i not in eliminated_indices]
        
        # 求最小值
        v = cp.Variable(n)
        objective_min = cp.Minimize(v[target_index])
        
        constraints = [
            cp.sum(v) == 1,
            v >= 0,
        ]
        
        for e in eliminated_indices:
            for s in survived_indices:
                diff = judge_percents[e] - judge_percents[s] + self.epsilon
                constraints.append(v[s] - v[e] >= diff)
        
        prob_min = cp.Problem(objective_min, constraints)
        try:
            prob_min.solve(solver=cp.OSQP, verbose=False)
            v_min = v[target_index].value if prob_min.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE] else 0
        except:
            v_min = 0
        
        # 求最大值
        v = cp.Variable(n)
        objective_max = cp.Maximize(v[target_index])
        
        constraints = [
            cp.sum(v) == 1,
            v >= 0,
        ]
        
        for e in eliminated_indices:
            for s in survived_indices:
                diff = judge_percents[e] - judge_percents[s] + self.epsilon
                constraints.append(v[s] - v[e] >= diff)
        
        prob_max = cp.Problem(objective_max, constraints)
        try:
            prob_max.solve(solver=cp.OSQP, verbose=False)
            v_max = v[target_index].value if prob_max.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE] else 1
        except:
            v_max = 1
        
        if v_min is None:
            v_min = 0
        if v_max is None:
            v_max = 1
        
        return max(0, float(v_min)), min(1, float(v_max))
    
    def compute_feasible_interval_scipy(
        self, 
        judge_percents: np.ndarray, 
        eliminated_indices: List[int],
        target_index: int
    ) -> Tuple[float, float]:
        """使用scipy计算可行区间"""
        n = len(judge_percents)
        
        if n == 0:
            return 0, 0
        
        survived_indices = [i for i in range(n) if i not in eliminated_indices]
        
        # 构建约束
        # 等式约束: sum(v) = 1
        A_eq = np.ones((1, n))
        b_eq = np.array([1.0])
        
        # 不等式约束: v_s - v_e >= j_e - j_s + epsilon
        # 转换为 -v_s + v_e <= -(j_e - j_s + epsilon)
        A_ub_list = []
        b_ub_list = []
        
        for e in eliminated_indices:
            for s in survived_indices:
                row = np.zeros(n)
                row[e] = 1
                row[s] = -1
                A_ub_list.append(row)
                b_ub_list.append(-(judge_percents[e] - judge_percents[s] + self.epsilon))
        
        if A_ub_list:
            A_ub = np.array(A_ub_list)
            b_ub = np.array(b_ub_list)
        else:
            A_ub = None
            b_ub = None
        
        bounds = [(0, 1) for _ in range(n)]
        
        # 求最小值
        c_min = np.zeros(n)
        c_min[target_index] = 1
        
        # 尝试不同的方法以兼容不同版本的scipy
        methods = ['highs', 'interior-point', 'simplex']
        v_min = 0
        
        for method in methods:
            try:
                result_min = linprog(c_min, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, 
                                    bounds=bounds, method=method)
                if result_min.success:
                    v_min = result_min.x[target_index]
                    break
            except (ValueError, TypeError):
                continue
        
        # 求最大值 (最小化负值)
        c_max = np.zeros(n)
        c_max[target_index] = -1
        v_max = 1
        
        for method in methods:
            try:
                result_max = linprog(c_max, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, 
                                    bounds=bounds, method=method)
                if result_max.success:
                    v_max = result_max.x[target_index]
                    break
            except (ValueError, TypeError):
                continue
        
        return max(0, float(v_min)), min(1, float(v_max))
    
    def compute_feasible_interval(
        self, 
        judge_percents: np.ndarray, 
        eliminated_indices: List[int],
        target_index: int
    ) -> Tuple[float, float]:
        """计算指定选手的观众投票百分比可行区间"""
        if USE_CVXPY:
            return self.compute_feasible_interval_cvxpy(
                judge_percents, eliminated_indices, target_index
            )
        else:
            return self.compute_feasible_interval_scipy(
                judge_percents, eliminated_indices, target_index
            )
    
    def compute_all_intervals(
        self, 
        judge_percents: np.ndarray, 
        eliminated_indices: List[int]
    ) -> List[Tuple[float, float]]:
        """计算所有选手的可行区间"""
        n = len(judge_percents)
        intervals = []
        
        for i in range(n):
            v_min, v_max = self.compute_feasible_interval(
                judge_percents, eliminated_indices, i
            )
            intervals.append((v_min, v_max))
        
        return intervals


class SmoothFanVoteEstimator:
    """基于平滑性原则的观众投票估计器"""
    
    def __init__(self, epsilon: float = 0.001, smoothness_weight: float = 1.0):
        """
        初始化估计器
        
        Args:
            epsilon: 严格不等式的小量
            smoothness_weight: 平滑性惩罚的权重
        """
        self.epsilon = epsilon
        self.smoothness_weight = smoothness_weight
        self.interval_estimator = FanVoteIntervalEstimator(epsilon)
    
    def estimate_season(
        self, 
        season_data: List[Dict]
    ) -> Dict[str, List[Tuple[int, float, float, float]]]:
        """
        对一个季度的所有周进行联合优化
        
        使用平滑性正则化：最小化选手在相邻周之间投票百分比的变化
        
        Args:
            season_data: 每周数据的列表，每个元素包含:
                - week: 周数
                - contestant_names: 选手名列表
                - judge_percents: 评委百分比数组
                - eliminated_indices: 淘汰者索引
                
        Returns:
            {选手名: [(周, 评委%, 观众%, 区间宽度), ...]}
        """
        if not season_data:
            return {}
        
        # 收集所有选手
        all_contestants = set()
        for week_data in season_data:
            all_contestants.update(week_data['contestant_names'])
        all_contestants = sorted(list(all_contestants))
        
        # 为每位选手收集其参赛周的数据
        contestant_weeks = {name: [] for name in all_contestants}
        
        for week_data in season_data:
            week = week_data['week']
            names = week_data['contestant_names']
            judge_percents = week_data['judge_percents']
            eliminated = week_data['eliminated_indices']
            
            # 计算该周所有选手的可行区间
            intervals = self.interval_estimator.compute_all_intervals(
                judge_percents, eliminated
            )
            
            for i, name in enumerate(names):
                contestant_weeks[name].append({
                    'week': week,
                    'index_in_week': i,
                    'judge_percent': judge_percents[i],
                    'interval': intervals[i],
                    'week_data': week_data
                })
        
        # 对每位选手单独进行平滑优化
        results = {}
        
        for name in all_contestants:
            weeks_info = contestant_weeks[name]
            if not weeks_info:
                continue
            
            # 按周排序
            weeks_info = sorted(weeks_info, key=lambda x: x['week'])
            n_weeks = len(weeks_info)
            
            if n_weeks == 1:
                # 只有一周，取区间中点
                info = weeks_info[0]
                v_min, v_max = info['interval']
                v_mid = (v_min + v_max) / 2
                results[name] = [(info['week'], info['judge_percent'], v_mid, v_max - v_min)]
                continue
            
            # 多周联合优化：最小化相邻周变化
            v = cp.Variable(n_weeks)
            
            # 平滑性目标：最小化相邻周的差异
            smoothness_term = cp.sum_squares(cp.diff(v))
            
            # 约束：每周的值在可行区间内
            constraints = []
            for t, info in enumerate(weeks_info):
                v_min, v_max = info['interval']
                constraints.append(v[t] >= v_min)
                constraints.append(v[t] <= v_max)
            
            # 求解
            objective = cp.Minimize(smoothness_term)
            prob = cp.Problem(objective, constraints)
            
            try:
                prob.solve(solver=cp.OSQP, verbose=False)
                
                if prob.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
                    fan_votes = v.value
                else:
                    # 如果优化失败，使用区间中点
                    fan_votes = np.array([(info['interval'][0] + info['interval'][1]) / 2 
                                          for info in weeks_info])
            except:
                fan_votes = np.array([(info['interval'][0] + info['interval'][1]) / 2 
                                      for info in weeks_info])
            
            # 收集结果
            results[name] = []
            for t, info in enumerate(weeks_info):
                v_min, v_max = info['interval']
                fan_vote = fan_votes[t] if fan_votes is not None else (v_min + v_max) / 2
                fan_vote = max(v_min, min(v_max, fan_vote))  # 确保在区间内
                results[name].append((
                    info['week'], 
                    info['judge_percent'], 
                    float(fan_vote),
                    v_max - v_min  # 区间宽度作为不确定性度量
                ))
        
        return results
    
    def estimate_with_weekly_constraints_cvxpy(
        self, 
        season_data: List[Dict]
    ) -> List[Dict]:
        """使用cvxpy进行联合优化"""
        if not season_data:
            return []
        
        all_contestants = set()
        for week_data in season_data:
            all_contestants.update(week_data['contestant_names'])
        all_contestants = sorted(list(all_contestants))
        
        week_var_info = []
        total_vars = 0
        
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
        
        v = cp.Variable(total_vars)
        constraints = []
        
        for t, info in enumerate(week_var_info):
            start = info['start_idx']
            n = info['n_vars']
            judge_percents = info['judge_percents']
            eliminated = info['eliminated_indices']
            
            v_week = v[start:start+n]
            constraints.append(cp.sum(v_week) == 1)
            constraints.append(v_week >= 0)
            
            survived = [i for i in range(n) if i not in eliminated]
            for e in eliminated:
                for s in survived:
                    diff = judge_percents[e] - judge_percents[s] + self.epsilon
                    constraints.append(v_week[s] - v_week[e] >= diff)
        
        smoothness_terms = []
        
        for name in all_contestants:
            contestant_var_indices = []
            for t, info in enumerate(week_var_info):
                if name in info['names']:
                    local_idx = info['names'].index(name)
                    global_var_idx = info['start_idx'] + local_idx
                    contestant_var_indices.append((info['week'], global_var_idx))
            
            contestant_var_indices.sort(key=lambda x: x[0])
            
            if len(contestant_var_indices) > 1:
                for i in range(len(contestant_var_indices) - 1):
                    idx1 = contestant_var_indices[i][1]
                    idx2 = contestant_var_indices[i+1][1]
                    smoothness_terms.append(cp.square(v[idx2] - v[idx1]))
        
        if smoothness_terms:
            objective = cp.Minimize(cp.sum(smoothness_terms))
        else:
            objective = cp.Minimize(cp.sum_squares(v))
        
        prob = cp.Problem(objective, constraints)
        
        try:
            prob.solve(solver=cp.OSQP, verbose=False)
            if prob.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
                prob.solve(solver=cp.SCS, verbose=False)
            solve_status = prob.status
            fan_votes = v.value
        except Exception as e:
            solve_status = str(e)
            fan_votes = None
        
        return self._format_results(week_var_info, fan_votes, solve_status)
    
    def estimate_with_weekly_constraints_scipy(
        self, 
        season_data: List[Dict]
    ) -> List[Dict]:
        """使用scipy进行联合优化"""
        if not season_data:
            return []
        
        all_contestants = set()
        for week_data in season_data:
            all_contestants.update(week_data['contestant_names'])
        all_contestants = sorted(list(all_contestants))
        
        week_var_info = []
        total_vars = 0
        
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
        
        # 构建约束
        # 等式约束: 每周sum(v) = 1
        A_eq_list = []
        b_eq_list = []
        
        for info in week_var_info:
            row = np.zeros(total_vars)
            row[info['start_idx']:info['start_idx']+info['n_vars']] = 1
            A_eq_list.append(row)
            b_eq_list.append(1.0)
        
        A_eq = np.array(A_eq_list)
        b_eq = np.array(b_eq_list)
        
        # 不等式约束: 淘汰约束
        A_ub_list = []
        b_ub_list = []
        
        for info in week_var_info:
            start = info['start_idx']
            n = info['n_vars']
            judge_percents = info['judge_percents']
            eliminated = info['eliminated_indices']
            survived = [i for i in range(n) if i not in eliminated]
            
            for e in eliminated:
                for s in survived:
                    row = np.zeros(total_vars)
                    row[start + e] = 1
                    row[start + s] = -1
                    A_ub_list.append(row)
                    b_ub_list.append(-(judge_percents[e] - judge_percents[s] + self.epsilon))
        
        if A_ub_list:
            A_ub = np.array(A_ub_list)
            b_ub = np.array(b_ub_list)
        else:
            A_ub = None
            b_ub = None
        
        # 构建平滑性矩阵
        smoothness_pairs = []
        for name in all_contestants:
            contestant_var_indices = []
            for info in week_var_info:
                if name in info['names']:
                    local_idx = info['names'].index(name)
                    global_var_idx = info['start_idx'] + local_idx
                    contestant_var_indices.append((info['week'], global_var_idx))
            
            contestant_var_indices.sort(key=lambda x: x[0])
            
            if len(contestant_var_indices) > 1:
                for i in range(len(contestant_var_indices) - 1):
                    idx1 = contestant_var_indices[i][1]
                    idx2 = contestant_var_indices[i+1][1]
                    smoothness_pairs.append((idx1, idx2))
        
        # 目标函数: 最小化平滑性 sum((v[idx2] - v[idx1])^2)
        # = sum(v[idx2]^2 - 2*v[idx1]*v[idx2] + v[idx1]^2)
        # 转换为二次规划: 0.5 * x'Qx
        Q = np.zeros((total_vars, total_vars))
        for idx1, idx2 in smoothness_pairs:
            Q[idx1, idx1] += 2
            Q[idx2, idx2] += 2
            Q[idx1, idx2] -= 2
            Q[idx2, idx1] -= 2
        
        if np.sum(np.abs(Q)) == 0:
            # 没有平滑性约束，使用最小方差
            Q = 2 * np.eye(total_vars)
        
        def objective(v):
            return 0.5 * v @ Q @ v
        
        def objective_grad(v):
            return Q @ v
        
        # 约束列表
        constraints = []
        
        # 等式约束
        constraints.append(LinearConstraint(A_eq, b_eq, b_eq))
        
        # 不等式约束
        if A_ub is not None:
            constraints.append(LinearConstraint(A_ub, -np.inf, b_ub))
        
        bounds = Bounds(np.zeros(total_vars), np.ones(total_vars))
        
        # 初始值
        v0 = []
        for info in week_var_info:
            v0.extend([1.0 / info['n_vars']] * info['n_vars'])
        v0 = np.array(v0)
        
        try:
            result = minimize(
                objective, v0,
                method='SLSQP',
                jac=objective_grad,
                bounds=bounds,
                constraints=[{'type': 'eq', 'fun': lambda v: A_eq @ v - b_eq}] + 
                           ([{'type': 'ineq', 'fun': lambda v: b_ub - A_ub @ v}] if A_ub is not None else []),
                options={'maxiter': 2000}
            )
            solve_status = "optimal" if result.success else result.message
            fan_votes = result.x
        except Exception as e:
            solve_status = str(e)
            fan_votes = v0
        
        return self._format_results(week_var_info, fan_votes, solve_status)
    
    def _format_results(self, week_var_info, fan_votes, solve_status):
        """格式化结果"""
        results = []
        
        for info in week_var_info:
            start = info['start_idx']
            n = info['n_vars']
            names = info['names']
            judge_percents = info['judge_percents']
            eliminated = info['eliminated_indices']
            
            if fan_votes is not None:
                week_fan_votes = fan_votes[start:start+n]
            else:
                week_fan_votes = np.ones(n) / n
            
            intervals = self.interval_estimator.compute_all_intervals(
                judge_percents, eliminated
            )
            
            week_results = {
                'week': info['week'],
                'solve_status': solve_status,
                'contestants': []
            }
            
            for i, name in enumerate(names):
                v_min, v_max = intervals[i]
                fan_vote = week_fan_votes[i]
                
                week_results['contestants'].append({
                    'name': name,
                    'judge_percent': judge_percents[i],
                    'fan_vote_percent': float(fan_vote),
                    'total_percent': float(judge_percents[i] + fan_vote),
                    'v_min': v_min,
                    'v_max': v_max,
                    'interval_width': v_max - v_min,
                    'eliminated': i in eliminated
                })
            
            results.append(week_results)
        
        return results
    
    def estimate_with_weekly_constraints(
        self, 
        season_data: List[Dict]
    ) -> List[Dict]:
        """
        对一个季度进行联合优化，同时满足每周的约束条件
        
        Args:
            season_data: 每周数据的列表
                
        Returns:
            每周结果的列表
        """
        if USE_CVXPY:
            return self.estimate_with_weekly_constraints_cvxpy(season_data)
        else:
            return self.estimate_with_weekly_constraints_scipy(season_data)


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


def run_estimation(excel_path: str, output_path: str = None):
    """
    运行完整的观众投票估计（平滑性正则化版本）
    """
    processor = DWTSProcessedDataProcessor(excel_path)
    estimator = SmoothFanVoteEstimator(epsilon=0.001)
    
    all_results = []
    
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
        week_results = estimator.estimate_with_weekly_constraints(season_data)
        
        for week_result in week_results:
            week = week_result['week']
            status = week_result['solve_status']
            contestants = week_result['contestants']
            
            eliminated_names = [c['name'] for c in contestants if c['eliminated']]
            
            print(f"\n  第 {week} 周: {len(contestants)} 位选手", end="")
            if eliminated_names:
                print(f", 淘汰: {eliminated_names}", end="")
            print()
            print(f"    求解状态: {status}")
            
            for c in contestants:
                elim_mark = " [淘汰]" if c['eliminated'] else ""
                interval_info = f"[{c['v_min']*100:.2f}%, {c['v_max']*100:.2f}%]"
                print(f"      {c['name']}: 评委{c['judge_percent']*100:.2f}% + "
                      f"观众{c['fan_vote_percent']*100:.2f}% = {c['total_percent']*100:.2f}% "
                      f"区间{interval_info}{elim_mark}")
                
                all_results.append({
                    'season': season,
                    'week': week,
                    'celebrity_name': c['name'],
                    'judge_percent': c['judge_percent'],
                    'fan_vote_percent': c['fan_vote_percent'],
                    'total_percent': c['total_percent'],
                    'fan_vote_min': c['v_min'],
                    'fan_vote_max': c['v_max'],
                    'interval_width': c['interval_width'],
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
    
    for (season, week), group in df.groupby(['season', 'week']):
        eliminated = group[group['eliminated'] == True]
        survived = group[group['eliminated'] == False]
        
        if len(eliminated) > 0 and len(survived) > 0:
            max_elim_total = eliminated['total_percent'].max()
            min_surv_total = survived['total_percent'].min()
            
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
        'avg_interval_width': df['interval_width'].mean(),
        'avg_smoothness': np.mean(smoothness_scores) if smoothness_scores else 0,
    }
    
    return validation


def analyze_controversies(results: List[Dict]):
    """分析争议性选手"""
    if not results:
        return
    
    df = pd.DataFrame(results)
    
    print("\n" + "="*60)
    print("争议性选手分析")
    print("="*60)
    
    # Bobby Bones (Season 27)
    print("\n【第27季 - Bobby Bones】")
    s27 = df[df['season'] == 27]
    bobby = s27[s27['celebrity_name'] == 'Bobby Bones'].sort_values('week')
    if len(bobby) > 0:
        print("周次 | 评委% | 观众%(估计) | 观众区间 | 总%")
        print("-" * 55)
        for _, row in bobby.iterrows():
            print(f"  {int(row['week'])}  | {row['judge_percent']*100:5.2f}% | "
                  f"{row['fan_vote_percent']*100:5.2f}% | "
                  f"[{row['fan_vote_min']*100:.1f}%, {row['fan_vote_max']*100:.1f}%] | "
                  f"{row['total_percent']*100:5.2f}%")
    
    # Bristol Palin (Season 11)
    print("\n【第11季 - Bristol Palin】")
    s11 = df[df['season'] == 11]
    bristol = s11[s11['celebrity_name'] == 'Bristol Palin'].sort_values('week')
    if len(bristol) > 0:
        print("周次 | 评委% | 观众%(估计) | 观众区间 | 总%")
        print("-" * 55)
        for _, row in bristol.iterrows():
            elim = " [淘汰]" if row['eliminated'] else ""
            print(f"  {int(row['week'])}  | {row['judge_percent']*100:5.2f}% | "
                  f"{row['fan_vote_percent']*100:5.2f}% | "
                  f"[{row['fan_vote_min']*100:.1f}%, {row['fan_vote_max']*100:.1f}%] | "
                  f"{row['total_percent']*100:5.2f}%{elim}")


if __name__ == "__main__":
    excel_path = r"d:\Users\13016\Desktop\26MCM\2026_C\2026_MCM_Problem_C_Processed_Data.xlsx"
    output_path = r"d:\Users\13016\Desktop\26MCM\2026_C\task1-1\fan_vote_estimates_smooth.csv"
    
    # 运行估计
    results = run_estimation(excel_path, output_path)
    
    # 验证结果
    if results:
        validation = validate_results(results)
        print(f"\n{'='*60}")
        print("验证结果:")
        for key, value in validation.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")
        
        # 分析争议性选手
        analyze_controversies(results)
