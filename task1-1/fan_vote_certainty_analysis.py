"""
观众投票估计的确定性分析（问题一第二问）

两种方式量化观众投票百分比的确定性：
1. 可行域区间法：使用已有 fan_vote_min, fan_vote_max，区间宽度 w = v_max - v_min 作为不确定性度量
2. 扰动-重建(Bootstrap)法：对评委打分加噪声 J' = J + η，重新求解得到 v^(b)，用样本方差和分位数区间表达确定性

输入：fan_vote_estimates_entropy_smooth_100.csv（lambda_smooth=100 的估计结果）
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
import warnings
import sys
import os

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

warnings.filterwarnings('ignore')

# 默认路径：与 task1-1 同级的 Excel
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DEFAULT_CSV_PATH = os.path.join(SCRIPT_DIR, 'fan_vote_estimates_entropy_smooth_100.csv')
DEFAULT_EXCEL_PATH = os.path.join(PROJECT_ROOT, '2026_MCM_Problem_C_Processed_Data.xlsx')


def load_estimate_csv(csv_path: str) -> pd.DataFrame:
    """加载 lambda_smooth=100 的估计结果"""
    df = pd.read_csv(csv_path)
    # 确保有可行区间列（若旧版 CSV 无则需后续只做 Method 2 或先补算）
    if 'interval_width' not in df.columns:
        df['interval_width'] = np.nan
        if 'fan_vote_max' in df.columns and 'fan_vote_min' in df.columns:
            df['interval_width'] = df['fan_vote_max'] - df['fan_vote_min']
    return df


# ---------- 方法一：可行域区间法 ----------

def certainty_method1_interval(df: pd.DataFrame, output_dir: str) -> pd.DataFrame:
    """
    方法一：可行域区间法
    不确定性度量：w_s,i,t = v_max - v_min（已有列 interval_width）
    """
    out_path = os.path.join(output_dir, 'certainty_method1_interval_summary.csv')
    # 按是否淘汰分组统计
    agg = df.groupby('eliminated').agg(
        count=('interval_width', 'count'),
        mean_width=('interval_width', 'mean'),
        std_width=('interval_width', 'std'),
        min_width=('interval_width', 'min'),
        max_width=('interval_width', 'max'),
    ).round(6)
    agg.to_csv(out_path, encoding='utf-8-sig')
    print(f"方法一（可行域区间）汇总已保存: {out_path}")

    # 按 (season, week) 有淘汰 vs 无淘汰 的周统计
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

    # 全表保留 interval_width 作为确定性指标（宽度越大越不确定）
    df['certainty_interval_uncertainty'] = df['interval_width']
    return df


# ---------- 方法二：扰动-重建 (Bootstrap) 法 ----------

def _perturb_season_data(season_data: List[Dict], sigma: float, rng: np.random.Generator) -> List[Dict]:
    """对单季每周的 judge_percents 加噪声，深拷贝后返回"""
    import copy
    out = []
    for wd in season_data:
        wd_copy = copy.deepcopy(wd)
        j = np.array(wd_copy['judge_percents'], dtype=float)
        noise = rng.normal(0, sigma, size=j.shape)
        j_new = np.clip(j + noise, 1e-6, 1.0)
        wd_copy['judge_percents'] = j_new
        out.append(wd_copy)
    return out


def certainty_method2_bootstrap(
    csv_path: str,
    excel_path: str,
    output_dir: str,
    n_bootstrap: int = 15,
    sigma: float = 0.01,
    lambda_smooth: float = 100.0,
    seasons: Tuple[int, int] = (3, 28),
) -> pd.DataFrame:
    """
    方法二：对评委打分加噪声 J' = J + η，重新求解得 v^(b)，计算样本方差与 95% 分位数区间
    """
    from fan_vote_estimation_entropy_smooth import DWTSProcessedDataProcessor, EntropySmoothFanVoteEstimator

    processor = DWTSProcessedDataProcessor(excel_path)
    estimator = EntropySmoothFanVoteEstimator(epsilon=0.001, lambda_smooth=lambda_smooth)
    rng = np.random.default_rng(42)

    bootstrap_rows = []  # list of list of dicts (season, week, name, fan_vote_percent)
    print(f"方法二（Bootstrap）：B={n_bootstrap}, sigma={sigma}, lambda_smooth={lambda_smooth}")

    for b in range(n_bootstrap):
        if (b + 1) % 5 == 0 or b == 0:
            print(f"  Bootstrap 样本 {b+1}/{n_bootstrap} ...")
        for season in range(seasons[0], seasons[1]):
            season_data = processor.process_season(season)
            if not season_data:
                continue
            season_data_perturbed = _perturb_season_data(season_data, sigma, rng)
            week_results = estimator.estimate_season(season_data_perturbed)
            for wr in week_results:
                for c in wr['contestants']:
                    bootstrap_rows.append({
                        'bootstrap_id': b,
                        'season': season,
                        'week': wr['week'],
                        'celebrity_name': c['name'],
                        'fan_vote_percent': c['fan_vote_percent'],
                    })

    df_b = pd.DataFrame(bootstrap_rows)
    # 按 (season, week, celebrity_name) 聚合
    agg_b = df_b.groupby(['season', 'week', 'celebrity_name'])['fan_vote_percent'].agg([
        ('fan_vote_mean_b', 'mean'),
        ('fan_vote_var_b', 'var'),
        ('fan_vote_std_b', 'std'),
        ('fan_vote_ci_lower', lambda x: np.nanpercentile(x, 2.5)),
        ('fan_vote_ci_upper', lambda x: np.nanpercentile(x, 97.5)),
        ('n_bootstrap', 'count'),
    ]).reset_index()
    # 单样本时 var 为 NaN，用 0 填
    agg_b['fan_vote_var_b'] = agg_b['fan_vote_var_b'].fillna(0)
    agg_b['fan_vote_std_b'] = agg_b['fan_vote_std_b'].fillna(0)

    base = pd.read_csv(csv_path)
    merge_cols = ['season', 'week', 'celebrity_name']
    out_df = base.merge(agg_b, on=merge_cols, how='left')
    out_path = os.path.join(output_dir, 'certainty_method2_bootstrap_10000.csv')
    out_df.to_csv(out_path, index=False, encoding='utf-8-sig')
    print(f"方法二（Bootstrap）结果已保存: {out_path}")
    # 简要汇总
    ci_width = out_df['fan_vote_ci_upper'] - out_df['fan_vote_ci_lower']
    print(f"  样本方差均值: {out_df['fan_vote_var_b'].mean():.6f}")
    print(f"  95% 置信区间宽度均值: {ci_width.mean():.4f}")
    return out_df


def run_certainty_analysis(
    csv_path: str = DEFAULT_CSV_PATH,
    excel_path: str = DEFAULT_EXCEL_PATH,
    output_dir: str = None,
    run_method1: bool = True,
    run_method2: bool = True,
    n_bootstrap: int = 15,
    sigma: float = 0.01,
) -> Dict[str, pd.DataFrame]:
    """
    运行两种确定性分析，写入 output_dir。
    """
    if output_dir is None:
        output_dir = SCRIPT_DIR

    os.makedirs(output_dir, exist_ok=True)
    df = load_estimate_csv(csv_path)
    results = {}

    if run_method1:
        df1 = certainty_method1_interval(df.copy(), output_dir)
        results['method1'] = df1

    if run_method2:
        df2 = certainty_method2_bootstrap(
            csv_path=csv_path,
            excel_path=excel_path,
            output_dir=output_dir,
            n_bootstrap=n_bootstrap,
            sigma=sigma,
            lambda_smooth=100.0,
        )
        results['method2'] = df2

    # 合并：method2 结果已包含 base 的 interval_width，直接另存为 combined
    if run_method1 and run_method2 and 'method2' in results:
        combined_path = os.path.join(output_dir, 'certainty_combined.csv')
        results['method2'].to_csv(combined_path, index=False, encoding='utf-8-sig')
        print(f"合并确定性结果已保存: {combined_path}")
        results['combined'] = results['method2']

    return results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='观众投票估计的确定性分析')
    parser.add_argument('--csv', default=DEFAULT_CSV_PATH, help='估计结果 CSV（如 entropy_smooth_100）')
    parser.add_argument('--excel', default=DEFAULT_EXCEL_PATH, help='处理后数据 Excel')
    parser.add_argument('--out-dir', default=None, help='输出目录，默认与 CSV 同目录')
    parser.add_argument('--no-method1', action='store_true', help='跳过方法一')
    parser.add_argument('--no-method2', action='store_true', help='跳过方法二')
    parser.add_argument('--B', type=int, default=15, help='Bootstrap 次数')
    parser.add_argument('--sigma', type=float, default=0.01, help='评委打分噪声标准差')
    args = parser.parse_args()

    run_certainty_analysis(
        csv_path=args.csv,
        excel_path=args.excel,
        output_dir=args.out_dir,
        run_method1=not args.no_method1,
        run_method2=not args.no_method2,
        n_bootstrap=args.B,
        sigma=args.sigma,
    )
