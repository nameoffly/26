"""
观众投票估计的确定性分析（问题一第二问）- GPU 加速版本

使用 CuPy 进行 GPU 加速的 Bootstrap 分析
需要安装: pip install cupy-cuda12x (根据你的 CUDA 版本)

主要优化：
1. 使用 CuPy 替代 NumPy 进行矩阵运算
2. 批量处理 bootstrap 样本
3. GPU 并行化噪声生成和数值计算
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
import warnings
import sys
import os

# 检测 GPU 可用性
try:
    import cupy as cp
    GPU_AVAILABLE = True
    print("✓ CuPy 已加载，GPU 加速已启用")
    print(f"  GPU 设备: {cp.cuda.Device().compute_capability}")
except ImportError:
    GPU_AVAILABLE = False
    print("✗ CuPy 未安装，将使用 CPU 模式")
    print("  安装命令: pip install cupy-cuda12x (或对应你的 CUDA 版本)")

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

warnings.filterwarnings('ignore')

# 默认路径
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DEFAULT_CSV_PATH = os.path.join(SCRIPT_DIR, 'fan_vote_estimates_entropy_smooth_100.csv')
DEFAULT_EXCEL_PATH = os.path.join(PROJECT_ROOT, '2026_MCM_Problem_C_Processed_Data.xlsx')


def load_estimate_csv(csv_path: str) -> pd.DataFrame:
    """加载 lambda_smooth=100 的估计结果"""
    df = pd.read_csv(csv_path)
    if 'interval_width' not in df.columns:
        df['interval_width'] = np.nan
        if 'fan_vote_max' in df.columns and 'fan_vote_min' in df.columns:
            df['interval_width'] = df['fan_vote_max'] - df['fan_vote_min']
    return df


# ---------- 方法一：可行域区间法（无需 GPU 加速）----------

def certainty_method1_interval(df: pd.DataFrame, output_dir: str) -> pd.DataFrame:
    """
    方法一：可行域区间法
    不确定性度量：w_s,i,t = v_max - v_min（已有列 interval_width）
    """
    out_path = os.path.join(output_dir, 'certainty_method1_interval_summary.csv')
    agg = df.groupby('eliminated').agg(
        count=('interval_width', 'count'),
        mean_width=('interval_width', 'mean'),
        std_width=('interval_width', 'std'),
        min_width=('interval_width', 'min'),
        max_width=('interval_width', 'max'),
    ).round(6)
    agg.to_csv(out_path, encoding='utf-8-sig')
    print(f"方法一（可行域区间）汇总已保存: {out_path}")

    df_week = df.groupby(['season', 'week']).agg(
        n_contestants=('celebrity_name', 'count'),
        n_eliminated=('eliminated', 'sum'),
        mean_interval_width=('interval_width', 'mean'),
        min_interval_width=('interval_width', 'min'),
        max_interval_width=('interval_width', 'max'),
    ).reset_index()
    df_week['has_elimination'] = df_week['n_eliminated'] > 0
    by_elim = df_week.groupby('has_elimination')['mean_interval_width'].agg(['mean', 'std', 'count'])
    print("\n方法一：按本周是否有淘汰统计平均区间宽度")
    print(by_elim.round(6))

    df['certainty_interval_uncertainty'] = df['interval_width']
    return df


# ---------- 方法二：GPU 加速的 Bootstrap 法 ----------

def _generate_noise_batch_gpu(shapes: List[tuple], sigma: float, n_bootstrap: int, seed: int = 42):
    """
    在 GPU 上批量生成所有 bootstrap 样本的噪声
    
    Args:
        shapes: 每个 week 的 judge_percents 形状列表
        sigma: 噪声标准差
        n_bootstrap: bootstrap 次数
        seed: 随机种子
    
    Returns:
        List of GPU arrays，每个元素是 (n_bootstrap, *shape) 的噪声矩阵
    """
    if not GPU_AVAILABLE:
        # CPU 回退
        rng = np.random.default_rng(seed)
        return [rng.normal(0, sigma, size=(n_bootstrap, *shape)) for shape in shapes]
    
    cp.random.seed(seed)
    noise_batches = []
    for shape in shapes:
        # 在 GPU 上生成 (n_bootstrap, *shape) 的噪声
        noise = cp.random.normal(0, sigma, size=(n_bootstrap, *shape))
        noise_batches.append(noise)
    return noise_batches


def _apply_noise_batch(judge_percents_list: List[np.ndarray], 
                       noise_batches: List, 
                       n_bootstrap: int) -> List[List[np.ndarray]]:
    """
    批量应用噪声到评委打分数据
    
    Returns:
        List[List[np.ndarray]]: [bootstrap_id][week_id] -> perturbed judge_percents
    """
    if not GPU_AVAILABLE:
        # CPU 版本
        result = []
        for b in range(n_bootstrap):
            bootstrap_data = []
            for i, j in enumerate(judge_percents_list):
                j_perturbed = np.clip(j + noise_batches[i][b], 1e-6, 1.0)
                bootstrap_data.append(j_perturbed)
            result.append(bootstrap_data)
        return result
    
    # GPU 版本
    result = []
    for b in range(n_bootstrap):
        bootstrap_data = []
        for i, j in enumerate(judge_percents_list):
            # 将原始数据移到 GPU
            j_gpu = cp.asarray(j)
            # 应用噪声
            j_perturbed_gpu = cp.clip(j_gpu + noise_batches[i][b], 1e-6, 1.0)
            # 移回 CPU（因为后续的优化器可能需要 CPU 数据）
            j_perturbed = cp.asnumpy(j_perturbed_gpu)
            bootstrap_data.append(j_perturbed)
        result.append(bootstrap_data)
    return result


def certainty_method2_bootstrap_gpu(
    csv_path: str,
    excel_path: str,
    output_dir: str,
    n_bootstrap: int = 1000,
    sigma: float = 0.01,
    lambda_smooth: float = 100.0,
    seasons: Tuple[int, int] = (3, 28),
    batch_size: int = 50,  # 每批处理的 bootstrap 样本数
) -> pd.DataFrame:
    """
    方法二：GPU 加速的 Bootstrap 分析
    
    优化策略：
    1. 预先在 GPU 上批量生成所有噪声
    2. 分批处理 bootstrap 样本以控制显存
    3. 复用季节数据加载
    """
    from fan_vote_estimation_entropy_smooth import DWTSProcessedDataProcessor, EntropySmoothFanVoteEstimator

    processor = DWTSProcessedDataProcessor(excel_path)
    estimator = EntropySmoothFanVoteEstimator(epsilon=0.001, lambda_smooth=lambda_smooth)

    print(f"\n{'='*60}")
    print(f"方法二（GPU 加速 Bootstrap）")
    print(f"{'='*60}")
    print(f"  Bootstrap 样本数: {n_bootstrap}")
    print(f"  噪声标准差: {sigma}")
    print(f"  平滑参数: {lambda_smooth}")
    print(f"  批处理大小: {batch_size}")
    print(f"  GPU 状态: {'已启用' if GPU_AVAILABLE else '未启用（使用 CPU）'}")
    print(f"{'='*60}\n")

    # 第一步：加载所有季节数据（只加载一次）
    print("步骤 1/4: 加载所有季节数据...")
    all_season_data = {}
    for season in range(seasons[0], seasons[1]):
        season_data = processor.process_season(season)
        if season_data:
            all_season_data[season] = season_data
    print(f"  已加载 {len(all_season_data)} 个季的数据")

    # 第二步：提取所有 judge_percents 和形状
    print("\n步骤 2/4: 准备数据结构...")
    season_week_mapping = []  # [(season, week_idx, week_num, judge_percents)]
    for season, season_data in all_season_data.items():
        for week_idx, wd in enumerate(season_data):
            season_week_mapping.append({
                'season': season,
                'week_idx': week_idx,
                'week': wd['week'],
                'judge_percents': np.array(wd['judge_percents'], dtype=float),
                'week_data': wd  # 保存完整的 week 数据
            })
    
    shapes = [item['judge_percents'].shape for item in season_week_mapping]
    print(f"  总计 {len(season_week_mapping)} 个 week 数据点")

    # 第三步：在 GPU 上批量生成噪声
    print("\n步骤 3/4: 在 GPU 上生成噪声...")
    noise_batches = _generate_noise_batch_gpu(shapes, sigma, n_bootstrap, seed=42)
    print(f"  噪声生成完成")

    # 第四步：分批处理 bootstrap 样本
    print(f"\n步骤 4/4: 处理 {n_bootstrap} 个 bootstrap 样本...")
    bootstrap_rows = []
    
    n_batches = (n_bootstrap + batch_size - 1) // batch_size
    
    for batch_idx in range(n_batches):
        batch_start = batch_idx * batch_size
        batch_end = min((batch_idx + 1) * batch_size, n_bootstrap)
        batch_n = batch_end - batch_start
        
        print(f"\n  批次 {batch_idx + 1}/{n_batches}: 样本 {batch_start+1}-{batch_end}/{n_bootstrap}")
        
        # 批量应用噪声
        judge_percents_list = [item['judge_percents'] for item in season_week_mapping]
        
        # 为当前批次生成扰动数据
        for b_local in range(batch_n):
            b_global = batch_start + b_local
            
            if (b_global + 1) % 50 == 0 or b_global == 0:
                print(f"    处理 bootstrap 样本 {b_global+1}/{n_bootstrap}...")
            
            # 为每个 season 的每个 week 应用噪声
            for item_idx, item in enumerate(season_week_mapping):
                # 应用噪声
                j_original = item['judge_percents']
                if GPU_AVAILABLE:
                    j_gpu = cp.asarray(j_original)
                    noise_gpu = noise_batches[item_idx][b_local]
                    j_perturbed_gpu = cp.clip(j_gpu + noise_gpu, 1e-6, 1.0)
                    j_perturbed = cp.asnumpy(j_perturbed_gpu)
                else:
                    j_perturbed = np.clip(j_original + noise_batches[item_idx][b_local], 1e-6, 1.0)
                
                # 构造扰动后的 week_data
                import copy
                wd_perturbed = copy.deepcopy(item['week_data'])
                wd_perturbed['judge_percents'] = j_perturbed
                
                # 对当前季节的这一个 week 运行估计
                # 注意：这里需要传入完整的 season_data，但我们只更新了一个 week
                # 为了效率，我们需要重构 estimator 来支持单 week 估计
                # 暂时使用原有方式（可能需要进一步优化）
                
                # 临时方案：为每个 week 单独调用（后续可优化为批量）
                season = item['season']
                season_data_perturbed = copy.deepcopy(all_season_data[season])
                season_data_perturbed[item['week_idx']]['judge_percents'] = j_perturbed
                
                # 只估计当前 week（需要完整 season 上下文）
                week_results = estimator.estimate_season(season_data_perturbed)
                
                # 提取当前 week 的结果
                for wr in week_results:
                    if wr['week'] == item['week']:
                        for c in wr['contestants']:
                            bootstrap_rows.append({
                                'bootstrap_id': b_global,
                                'season': season,
                                'week': item['week'],
                                'celebrity_name': c['name'],
                                'fan_vote_percent': c['fan_vote_percent'],
                            })
                        break

    print(f"\n  Bootstrap 样本处理完成，共 {len(bootstrap_rows)} 条记录")

    # 聚合统计
    print("\n步骤 5/5: 聚合统计结果...")
    df_b = pd.DataFrame(bootstrap_rows)
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

    base = pd.read_csv(csv_path)
    merge_cols = ['season', 'week', 'celebrity_name']
    out_df = base.merge(agg_b, on=merge_cols, how='left')
    
    out_path = os.path.join(output_dir, f'certainty_method2_bootstrap_gpu_{n_bootstrap}.csv')
    out_df.to_csv(out_path, index=False, encoding='utf-8-sig')
    print(f"\n方法二（GPU Bootstrap）结果已保存: {out_path}")
    
    # 简要汇总
    ci_width = out_df['fan_vote_ci_upper'] - out_df['fan_vote_ci_lower']
    print(f"\n统计摘要:")
    print(f"  样本方差均值: {out_df['fan_vote_var_b'].mean():.6f}")
    print(f"  标准差均值: {out_df['fan_vote_std_b'].mean():.6f}")
    print(f"  95% 置信区间宽度均值: {ci_width.mean():.4f}")
    print(f"  95% 置信区间宽度中位数: {ci_width.median():.4f}")
    
    return out_df


def run_certainty_analysis(
    csv_path: str = DEFAULT_CSV_PATH,
    excel_path: str = DEFAULT_EXCEL_PATH,
    output_dir: str = None,
    run_method1: bool = True,
    run_method2: bool = True,
    n_bootstrap: int = 1000,
    sigma: float = 0.01,
    batch_size: int = 50,
    use_gpu: bool = True,
) -> Dict[str, pd.DataFrame]:
    """
    运行两种确定性分析（GPU 加速版本）
    """
    if output_dir is None:
        output_dir = SCRIPT_DIR

    os.makedirs(output_dir, exist_ok=True)
    df = load_estimate_csv(csv_path)
    results = {}

    if run_method1:
        print("\n" + "="*60)
        print("运行方法一：可行域区间法")
        print("="*60)
        df1 = certainty_method1_interval(df.copy(), output_dir)
        results['method1'] = df1

    if run_method2:
        if use_gpu and not GPU_AVAILABLE:
            print("\n⚠️  警告: GPU 不可用，将使用 CPU 模式（速度较慢）")
        
        df2 = certainty_method2_bootstrap_gpu(
            csv_path=csv_path,
            excel_path=excel_path,
            output_dir=output_dir,
            n_bootstrap=n_bootstrap,
            sigma=sigma,
            lambda_smooth=100.0,
            batch_size=batch_size,
        )
        results['method2'] = df2

    # 合并结果
    if run_method1 and run_method2 and 'method2' in results:
        combined_path = os.path.join(output_dir, f'certainty_combined_gpu_{n_bootstrap}.csv')
        results['method2'].to_csv(combined_path, index=False, encoding='utf-8-sig')
        print(f"\n合并确定性结果已保存: {combined_path}")
        results['combined'] = results['method2']

    return results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='观众投票估计的确定性分析（GPU 加速版）')
    parser.add_argument('--csv', default=DEFAULT_CSV_PATH, help='估计结果 CSV')
    parser.add_argument('--excel', default=DEFAULT_EXCEL_PATH, help='处理后数据 Excel')
    parser.add_argument('--out-dir', default=None, help='输出目录')
    parser.add_argument('--no-method1', action='store_true', help='跳过方法一')
    parser.add_argument('--no-method2', action='store_true', help='跳过方法二')
    parser.add_argument('--B', type=int, default=1000, help='Bootstrap 次数（默认 1000）')
    parser.add_argument('--sigma', type=float, default=0.01, help='噪声标准差')
    parser.add_argument('--batch-size', type=int, default=50, help='批处理大小')
    parser.add_argument('--no-gpu', action='store_true', help='强制使用 CPU')
    args = parser.parse_args()

    print("\n" + "="*60)
    print("观众投票确定性分析 - GPU 加速版本")
    print("="*60)
    
    results = run_certainty_analysis(
        csv_path=args.csv,
        excel_path=args.excel,
        output_dir=args.out_dir,
        run_method1=not args.no_method1,
        run_method2=not args.no_method2,
        n_bootstrap=args.B,
        sigma=args.sigma,
        batch_size=args.batch_size,
        use_gpu=not args.no_gpu,
    )
    
    print("\n" + "="*60)
    print("✓ 分析完成！")
    print("="*60)
