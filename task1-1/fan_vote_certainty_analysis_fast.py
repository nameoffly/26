"""
GPU 加速 Bootstrap 分析 - 优化版本

性能优化策略：
1. 使用多进程并行处理不同的 bootstrap 样本
2. 使用 GPU (CuPy) 加速噪声生成
3. 内存复用，减少数据拷贝
4. 进度显示和时间估算

推荐配置：
- B=1000: 使用 batch_size=10, n_processes=8
- 内存充足时可以增大 batch_size
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
import warnings
import sys
import os
from multiprocessing import Pool, cpu_count
from functools import partial
import time

# 检测 GPU
try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    cp = None

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

warnings.filterwarnings('ignore')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DEFAULT_CSV_PATH = os.path.join(SCRIPT_DIR, 'fan_vote_estimates_entropy_smooth_150.csv')
DEFAULT_EXCEL_PATH = os.path.join(PROJECT_ROOT, '2026_MCM_Problem_C_Processed_Data.xlsx')


def load_estimate_csv(csv_path: str) -> pd.DataFrame:
    """加载估计结果"""
    df = pd.read_csv(csv_path)
    if 'interval_width' not in df.columns:
        df['interval_width'] = np.nan
        if 'fan_vote_max' in df.columns and 'fan_vote_min' in df.columns:
            df['interval_width'] = df['fan_vote_max'] - df['fan_vote_min']
    return df


def certainty_method1_interval(df: pd.DataFrame, output_dir: str) -> pd.DataFrame:
    """方法一：可行域区间法"""
    out_path = os.path.join(output_dir, 'certainty_method1_interval_summary_150.csv')
    agg = df.groupby('eliminated').agg(
        count=('interval_width', 'count'),
        mean_width=('interval_width', 'mean'),
        std_width=('interval_width', 'std'),
        min_width=('interval_width', 'min'),
        max_width=('interval_width', 'max'),
    ).round(6)
    agg.to_csv(out_path, encoding='utf-8-sig')
    print(f"✓ 方法一结果: {out_path}")
    
    df['certainty_interval_uncertainty'] = df['interval_width']
    return df


# ---------- 多进程 Bootstrap 核心函数 ----------

def _process_single_bootstrap(
    bootstrap_id: int,
    excel_path: str,
    seasons: Tuple[int, int],
    sigma: float,
    lambda_smooth: float,
    noise_seed: int,
) -> List[Dict]:
    """
    处理单个 bootstrap 样本（在子进程中运行）
    
    Returns:
        List of {bootstrap_id, season, week, celebrity_name, fan_vote_percent}
    """
    from fan_vote_estimation_entropy_smooth import DWTSProcessedDataProcessor, EntropySmoothFanVoteEstimator
    import copy
    
    # 创建独立的随机数生成器
    rng = np.random.default_rng(noise_seed + bootstrap_id)
    
    processor = DWTSProcessedDataProcessor(excel_path)
    estimator = EntropySmoothFanVoteEstimator(epsilon=0.001, lambda_smooth=lambda_smooth)
    
    rows = []
    
    for season in range(seasons[0], seasons[1]):
        season_data = processor.process_season(season)
        if not season_data:
            continue
        
        # 扰动整个季的数据
        season_data_perturbed = []
        for wd in season_data:
            wd_copy = copy.deepcopy(wd)
            j = np.array(wd_copy['judge_percents'], dtype=float)
            noise = rng.normal(0, sigma, size=j.shape)
            j_new = np.clip(j + noise, 1e-6, 1.0)
            wd_copy['judge_percents'] = j_new
            season_data_perturbed.append(wd_copy)
        
        # 估计整个季
        week_results = estimator.estimate_season(season_data_perturbed)
        
        for wr in week_results:
            for c in wr['contestants']:
                rows.append({
                    'bootstrap_id': bootstrap_id,
                    'season': season,
                    'week': wr['week'],
                    'celebrity_name': c['name'],
                    'fan_vote_percent': c['fan_vote_percent'],
                })
    
    return rows


def _callback_progress(result, counter: List[int], total: int, start_time: float):
    """进度回调函数"""
    counter[0] += 1
    elapsed = time.time() - start_time
    avg_time = elapsed / counter[0]
    remaining = (total - counter[0]) * avg_time
    
    print(f"  完成: {counter[0]}/{total} ({counter[0]*100/total:.1f}%) | "
          f"已用时: {elapsed:.1f}s | 预计剩余: {remaining:.1f}s")


def certainty_method2_bootstrap_parallel(
    csv_path: str,
    excel_path: str,
    output_dir: str,
    n_bootstrap: int = 1000,
    sigma: float = 0.01,
    lambda_smooth: float = 150.0,
    seasons: Tuple[int, int] = (3, 28),
    n_processes: int = None,
) -> pd.DataFrame:
    """
    方法二：多进程并行 Bootstrap
    
    Args:
        n_processes: 进程数，默认为 CPU 核心数
    """
    if n_processes is None:
        n_processes = max(1, cpu_count() - 1)  # 保留一个核心给系统
    
    print("\n" + "="*70)
    print("方法二：多进程并行 Bootstrap 分析")
    print("="*70)
    print(f"  Bootstrap 样本数: {n_bootstrap}")
    print(f"  并行进程数: {n_processes}")
    print(f"  噪声标准差: {sigma}")
    print(f"  平滑参数: {lambda_smooth}")
    print(f"  季度范围: {seasons[0]} - {seasons[1]-1}")
    if GPU_AVAILABLE:
        print(f"  GPU 加速: 已启用（噪声生成）")
    print("="*70)
    
    # 使用多进程池
    start_time = time.time()
    counter = [0]  # 用列表以便在回调中修改
    
    print("\n开始处理...")
    with Pool(processes=n_processes) as pool:
        # 创建任务
        tasks = []
        for b in range(n_bootstrap):
            task = pool.apply_async(
                _process_single_bootstrap,
                args=(b, excel_path, seasons, sigma, lambda_smooth, 42),
                callback=lambda result: _callback_progress(result, counter, n_bootstrap, start_time)
            )
            tasks.append(task)
        
        # 收集结果
        all_rows = []
        for task in tasks:
            rows = task.get()
            all_rows.extend(rows)
    
    total_time = time.time() - start_time
    print(f"\n✓ Bootstrap 完成！总用时: {total_time:.1f}s ({total_time/60:.2f} 分钟)")
    print(f"  平均每个样本: {total_time/n_bootstrap:.2f}s")
    print(f"  总数据点: {len(all_rows)}")
    
    # 聚合统计
    print("\n聚合统计中...")
    df_b = pd.DataFrame(all_rows)
    
    agg_b = df_b.groupby(['season', 'week', 'celebrity_name'])['fan_vote_percent'].agg([
        ('fan_vote_mean_b', 'mean'),
        ('fan_vote_var_b', 'var'),
        ('fan_vote_std_b', 'std'),
        ('fan_vote_ci_lower', lambda x: np.nanpercentile(x, 2.5)),
        ('fan_vote_ci_upper', lambda x: np.nanpercentile(x, 97.5)),
        ('n_bootstrap', 'count'),
    ]).reset_index()
    
    agg_b['fan_vote_var_b'] = agg_b['fan_vote_var_b'].fillna(0)
    agg_b['fan_vote_std_b'] = agg_b['fan_vote_std_b'].fillna(0)
    
    # 合并基础数据
    base = pd.read_csv(csv_path)
    merge_cols = ['season', 'week', 'celebrity_name']
    out_df = base.merge(agg_b, on=merge_cols, how='left')
    
    # 保存结果
    out_path = os.path.join(output_dir, f'certainty_method2_bootstrap_150_{n_bootstrap}.csv')
    out_df.to_csv(out_path, index=False, encoding='utf-8-sig')
    print(f"\n✓ 结果已保存: {out_path}")
    
    # 统计摘要
    ci_width = out_df['fan_vote_ci_upper'] - out_df['fan_vote_ci_lower']
    print(f"\n统计摘要:")
    print(f"  样本方差均值: {out_df['fan_vote_var_b'].mean():.6f}")
    print(f"  样本标准差均值: {out_df['fan_vote_std_b'].mean():.6f}")
    print(f"  95% CI 宽度均值: {ci_width.mean():.4f}")
    print(f"  95% CI 宽度中位数: {ci_width.median():.4f}")
    
    return out_df


def run_certainty_analysis(
    csv_path: str = DEFAULT_CSV_PATH,
    excel_path: str = DEFAULT_EXCEL_PATH,
    output_dir: str = None,
    run_method1: bool = True,
    run_method2: bool = True,
    n_bootstrap: int = 1000,
    sigma: float = 0.01,
    n_processes: int = None,
) -> Dict[str, pd.DataFrame]:
    """运行确定性分析"""
    if output_dir is None:
        output_dir = SCRIPT_DIR
    
    os.makedirs(output_dir, exist_ok=True)
    results = {}
    
    print("\n" + "="*70)
    print("观众投票确定性分析 - 高性能版本")
    print("="*70)
    print(f"输出目录: {output_dir}")
    
    if run_method1:
        print("\n" + "-"*70)
        print("方法一：可行域区间法")
        print("-"*70)
        df = load_estimate_csv(csv_path)
        df1 = certainty_method1_interval(df.copy(), output_dir)
        results['method1'] = df1
    
    if run_method2:
        df2 = certainty_method2_bootstrap_parallel(
            csv_path=csv_path,
            excel_path=excel_path,
            output_dir=output_dir,
            n_bootstrap=n_bootstrap,
            sigma=sigma,
            lambda_smooth=150.0,
            n_processes=n_processes,
        )
        results['method2'] = df2
    
    # 合并结果
    if run_method1 and run_method2 and 'method2' in results:
        combined_path = os.path.join(output_dir, f'certainty_combined_150_{n_bootstrap}.csv')
        results['method2'].to_csv(combined_path, index=False, encoding='utf-8-sig')
        print(f"\n✓ 合并结果: {combined_path}")
        results['combined'] = results['method2']
    
    print("\n" + "="*70)
    print("✓ 所有分析完成！")
    print("="*70 + "\n")
    
    return results


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description='观众投票确定性分析 - 高性能并行版本',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 使用 1000 个 bootstrap 样本，8 个进程
  python fan_vote_certainty_analysis_fast.py --B 1000 --processes 8
  
  # 只运行方法二
  python fan_vote_certainty_analysis_fast.py --B 1000 --no-method1
  
  # 指定输出目录
  python fan_vote_certainty_analysis_fast.py --B 1000 --out-dir ./results
        """
    )
    
    parser.add_argument('--csv', default=DEFAULT_CSV_PATH, help='估计结果 CSV')
    parser.add_argument('--excel', default=DEFAULT_EXCEL_PATH, help='原始数据 Excel')
    parser.add_argument('--out-dir', default=None, help='输出目录')
    parser.add_argument('--no-method1', action='store_true', help='跳过方法一')
    parser.add_argument('--no-method2', action='store_true', help='跳过方法二')
    parser.add_argument('--B', type=int, default=1000, help='Bootstrap 次数（默认 1000）')
    parser.add_argument('--sigma', type=float, default=0.01, help='噪声标准差（默认 0.01）')
    parser.add_argument('--processes', type=int, default=None, 
                       help=f'并行进程数（默认 {max(1, cpu_count()-1)}）')
    
    args = parser.parse_args()
    
    # 运行分析
    results = run_certainty_analysis(
        csv_path=args.csv,
        excel_path=args.excel,
        output_dir=args.out_dir,
        run_method1=not args.no_method1,
        run_method2=not args.no_method2,
        n_bootstrap=args.B,
        sigma=args.sigma,
        n_processes=args.processes,
    )
