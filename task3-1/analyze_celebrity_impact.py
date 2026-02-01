#!/usr/bin/env python3
"""
MCM Problem C 第三问：分析名人属性与比赛结果的关系

分析内容：
1. 描述性统计：各行业的平均名次、获奖率
2. 假设检验：卡方检验、ANOVA、Pearson相关
3. 回归分析：预测最终名次、是否获奖
4. 舞伴分析：舞伴"带飞"能力排名
5. 评委vs粉丝差异分析
"""

import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'task3-1/combined_contestant_info.csv')
OUTPUT_DIR = os.path.join(BASE_DIR, 'task3-1/outputs')
FIGURE_DIR = os.path.join(BASE_DIR, 'task3-1/figures')

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FIGURE_DIR, exist_ok=True)


def load_data():
    """加载数据并创建选手级别汇总"""
    df = pd.read_csv(DATA_PATH)
    print(f"周级别数据: {len(df)} 行")

    # 选手级别汇总
    contestant = df.groupby(['赛季', '人名']).agg({
        '粉丝投票百分比': 'mean',
        '评委评分百分比': 'mean',
        '周数': 'max',
        '伴侣名': 'first',
        '地区': 'first',
        '国家': 'first',
        '年龄': 'first',
        '行业': 'first',
        '最终名次': 'first',
        '比赛结果': 'first'
    }).reset_index()

    contestant.columns = ['赛季', '人名', '平均粉丝投票', '平均评委评分', '存活周数',
                          '伴侣名', '地区', '国家', '年龄', '行业', '最终名次', '比赛结果']

    contestant['是否获奖'] = contestant['最终名次'] <= 3
    print(f"选手级别数据: {len(contestant)} 人")
    return df, contestant


def descriptive_stats(contestant):
    """描述性统计分析"""
    print("\n" + "="*60)
    print("描述性统计分析")
    print("="*60)

    # 各行业统计
    industry_stats = contestant.groupby('行业').agg({
        '最终名次': ['mean', 'std', 'count'],
        '是否获奖': ['sum', 'mean'],
        '年龄': 'mean',
        '平均粉丝投票': 'mean',
        '平均评委评分': 'mean'
    }).round(3)
    industry_stats.columns = ['平均名次', '名次标准差', '人数', '获奖人数', '获奖率',
                               '平均年龄', '平均粉丝投票', '平均评委评分']
    industry_stats = industry_stats.sort_values('平均名次')

    # 保存
    industry_stats.to_csv(os.path.join(OUTPUT_DIR, 'descriptive_stats.csv'), encoding='utf-8-sig')
    print("\n各行业统计（按平均名次排序）:")
    print(industry_stats[['平均名次', '人数', '获奖率', '平均年龄']].head(10))

    # 年龄统计
    print(f"\n年龄统计:")
    print(f"  平均年龄: {contestant['年龄'].mean():.1f}")
    print(f"  年龄范围: {contestant['年龄'].min()} - {contestant['年龄'].max()}")
    print(f"  中位数: {contestant['年龄'].median():.0f}")

    # 国家统计（美国 vs 非美国）
    contestant['是否美国'] = contestant['国家'] == 'United States'
    country_stats = contestant.groupby('是否美国').agg({
        '最终名次': ['mean', 'count'],
        '是否获奖': ['sum', 'mean']
    }).round(3)
    country_stats.columns = ['平均名次', '人数', '获奖人数', '获奖率']
    country_stats.index = ['Non-US', 'US']
    country_stats.to_csv(os.path.join(OUTPUT_DIR, 'country_stats.csv'), encoding='utf-8-sig')
    print(f"\n美国 vs 非美国统计:")
    print(country_stats)

    # 地区统计（仅美国选手）
    us_contestants = contestant[contestant['国家'] == 'United States']
    region_stats = us_contestants.groupby('地区').agg({
        '最终名次': ['mean', 'count'],
        '是否获奖': ['sum', 'mean']
    }).round(3)
    region_stats.columns = ['平均名次', '人数', '获奖人数', '获奖率']
    region_stats = region_stats.sort_values('人数', ascending=False)
    region_stats.to_csv(os.path.join(OUTPUT_DIR, 'region_stats.csv'), encoding='utf-8-sig')
    print(f"\n美国各州统计（按人数排序）:")
    print(region_stats.head(10))

    return industry_stats, country_stats, region_stats


def hypothesis_tests(contestant):
    """假设检验"""
    print("\n" + "="*60)
    print("假设检验")
    print("="*60)

    results = []

    # 1. 卡方检验：行业与获奖独立性
    contingency = pd.crosstab(contestant['行业'], contestant['是否获奖'])
    chi2, p_chi2, dof, expected = stats.chi2_contingency(contingency)
    results.append({
        '检验': '卡方检验',
        '假设': '行业与获奖独立',
        '统计量': round(chi2, 4),
        'p值': round(p_chi2, 4),
        '自由度': dof,
        '结论': '拒绝H0' if p_chi2 < 0.05 else '不拒绝H0'
    })
    print(f"\n1. 卡方检验（行业与获奖独立性）:")
    print(f"   χ² = {chi2:.4f}, p = {p_chi2:.4f}, df = {dof}")
    print(f"   结论: {'行业与获奖显著相关' if p_chi2 < 0.05 else '行业与获奖无显著关系'}")

    # 2. Pearson相关：年龄与名次
    r, p_corr = stats.pearsonr(contestant['年龄'], contestant['最终名次'])
    results.append({
        '检验': 'Pearson相关',
        '假设': '年龄与名次无关',
        '统计量': round(r, 4),
        'p值': round(p_corr, 4),
        '自由度': len(contestant)-2,
        '结论': '拒绝H0' if p_corr < 0.05 else '不拒绝H0'
    })
    print(f"\n2. Pearson相关（年龄与名次）:")
    print(f"   r = {r:.4f}, p = {p_corr:.4f}")
    print(f"   结论: {'年龄与名次显著正相关（年龄越大名次越差）' if p_corr < 0.05 and r > 0 else '年龄与名次无显著关系'}")

    # 3. ANOVA：不同行业名次差异
    # 只选择人数>=10的行业
    major_industries = contestant.groupby('行业').filter(lambda x: len(x) >= 10)
    model = smf.ols('最终名次 ~ C(行业)', data=major_industries).fit()
    anova = sm.stats.anova_lm(model, typ=2)
    f_stat = anova['F'].iloc[0]
    p_anova = anova['PR(>F)'].iloc[0]
    results.append({
        '检验': 'ANOVA',
        '假设': '各行业名次相同',
        '统计量': round(f_stat, 4),
        'p值': round(p_anova, 4),
        '自由度': f"{anova['df'].iloc[0]:.0f},{anova['df'].iloc[1]:.0f}",
        '结论': '拒绝H0' if p_anova < 0.05 else '不拒绝H0'
    })
    print(f"\n3. ANOVA（不同行业名次差异）:")
    print(f"   F = {f_stat:.4f}, p = {p_anova:.4f}")
    print(f"   结论: {'不同行业的名次存在显著差异' if p_anova < 0.05 else '不同行业的名次无显著差异'}")

    # 4. 年龄与粉丝投票相关
    r_fan, p_fan = stats.pearsonr(contestant['年龄'], contestant['平均粉丝投票'])
    results.append({
        '检验': 'Pearson相关',
        '假设': '年龄与粉丝投票无关',
        '统计量': round(r_fan, 4),
        'p值': round(p_fan, 4),
        '自由度': len(contestant)-2,
        '结论': '拒绝H0' if p_fan < 0.05 else '不拒绝H0'
    })
    print(f"\n4. Pearson相关（年龄与粉丝投票）:")
    print(f"   r = {r_fan:.4f}, p = {p_fan:.4f}")

    # 5. 年龄与评委评分相关
    r_judge, p_judge = stats.pearsonr(contestant['年龄'], contestant['平均评委评分'])
    results.append({
        '检验': 'Pearson相关',
        '假设': '年龄与评委评分无关',
        '统计量': round(r_judge, 4),
        'p值': round(p_judge, 4),
        '自由度': len(contestant)-2,
        '结论': '拒绝H0' if p_judge < 0.05 else '不拒绝H0'
    })
    print(f"\n5. Pearson相关（年龄与评委评分）:")
    print(f"   r = {r_judge:.4f}, p = {p_judge:.4f}")

    # 6. 卡方检验：美国vs非美国与获奖独立性
    contestant['是否美国'] = contestant['国家'] == 'United States'
    contingency_country = pd.crosstab(contestant['是否美国'], contestant['是否获奖'])
    chi2_country, p_country, dof_country, _ = stats.chi2_contingency(contingency_country)
    results.append({
        '检验': '卡方检验',
        '假设': '美国vs非美国与获奖独立',
        '统计量': round(chi2_country, 4),
        'p值': round(p_country, 4),
        '自由度': dof_country,
        '结论': '拒绝H0' if p_country < 0.05 else '不拒绝H0'
    })
    print(f"\n6. 卡方检验（美国vs非美国与获奖独立性）:")
    print(f"   χ² = {chi2_country:.4f}, p = {p_country:.4f}, df = {dof_country}")
    print(f"   结论: {'美国vs非美国与获奖显著相关' if p_country < 0.05 else '美国vs非美国与获奖无显著关系'}")

    # 7. 卡方检验：地区与获奖独立性（仅美国选手）
    us_contestants = contestant[contestant['国家'] == 'United States']
    if len(us_contestants) > 50:
        contingency_region = pd.crosstab(us_contestants['地区'], us_contestants['是否获奖'])
        # 过滤掉样本量太小的地区
        valid_regions = contingency_region[contingency_region.sum(axis=1) >= 5]
        if len(valid_regions) >= 2:
            chi2_region, p_region, dof_region, _ = stats.chi2_contingency(valid_regions)
            results.append({
                '检验': '卡方检验',
                '假设': '地区与获奖独立(美国)',
                '统计量': round(chi2_region, 4),
                'p值': round(p_region, 4),
                '自由度': dof_region,
                '结论': '拒绝H0' if p_region < 0.05 else '不拒绝H0'
            })
            print(f"\n7. 卡方检验（地区与获奖独立性，仅美国选手）:")
            print(f"   χ² = {chi2_region:.4f}, p = {p_region:.4f}, df = {dof_region}")
            print(f"   结论: {'地区与获奖显著相关' if p_region < 0.05 else '地区与获奖无显著关系'}")

    # 保存结果
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(OUTPUT_DIR, 'hypothesis_tests.csv'), index=False, encoding='utf-8-sig')

    return results_df


def regression_analysis(contestant):
    """回归分析"""
    print("\n" + "="*60)
    print("回归分析")
    print("="*60)

    # 只选择人数>=5的行业和舞伴
    major_industries = contestant['行业'].value_counts()
    major_industries = major_industries[major_industries >= 5].index.tolist()
    major_partners = contestant['伴侣名'].value_counts()
    major_partners = major_partners[major_partners >= 3].index.tolist()

    data = contestant[contestant['行业'].isin(major_industries) &
                      contestant['伴侣名'].isin(major_partners)].copy()
    print(f"筛选后样本量: {len(data)}")

    # 模型1：预测最终名次（OLS回归）
    print("\n模型1：预测最终名次 (OLS)")
    model1 = smf.ols('最终名次 ~ 年龄 + C(行业)', data=data).fit()
    print(f"  R² = {model1.rsquared:.4f}")
    print(f"  年龄系数 = {model1.params['年龄']:.4f} (p = {model1.pvalues['年龄']:.4f})")

    # 模型2：预测是否获奖（Logistic回归）
    print("\n模型2：预测是否获奖 (Logit)")
    try:
        data['是否获奖_int'] = data['是否获奖'].astype(int)
        model2 = smf.logit('是否获奖_int ~ 年龄 + C(行业)', data=data).fit(disp=0)
        print(f"  Pseudo R² = {model2.prsquared:.4f}")
        print(f"  年龄系数 = {model2.params['年龄']:.4f} (p = {model2.pvalues['年龄']:.4f})")
    except Exception as e:
        print(f"  Logit模型拟合失败: {e}")
        model2 = None

    # 模型3：评委评分影响因素
    print("\n模型3：评委评分影响因素 (OLS)")
    model_judge = smf.ols('平均评委评分 ~ 年龄 + C(行业)', data=data).fit()
    print(f"  R² = {model_judge.rsquared:.4f}")
    print(f"  年龄系数 = {model_judge.params['年龄']:.6f} (p = {model_judge.pvalues['年龄']:.4f})")

    # 模型4：粉丝投票影响因素
    print("\n模型4：粉丝投票影响因素 (OLS)")
    model_fan = smf.ols('平均粉丝投票 ~ 年龄 + C(行业)', data=data).fit()
    print(f"  R² = {model_fan.rsquared:.4f}")
    print(f"  年龄系数 = {model_fan.params['年龄']:.6f} (p = {model_fan.pvalues['年龄']:.4f})")

    # 保存回归结果
    with open(os.path.join(OUTPUT_DIR, 'regression_results.txt'), 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write("模型1：预测最终名次 (OLS)\n")
        f.write("="*60 + "\n")
        f.write(model1.summary().as_text())
        f.write("\n\n")
        f.write("="*60 + "\n")
        f.write("模型3：评委评分影响因素 (OLS)\n")
        f.write("="*60 + "\n")
        f.write(model_judge.summary().as_text())
        f.write("\n\n")
        f.write("="*60 + "\n")
        f.write("模型4：粉丝投票影响因素 (OLS)\n")
        f.write("="*60 + "\n")
        f.write(model_fan.summary().as_text())

    # 系数对比
    print("\n评委vs粉丝：年龄影响对比")
    print(f"  年龄对评委评分的影响: {model_judge.params['年龄']:.6f}")
    print(f"  年龄对粉丝投票的影响: {model_fan.params['年龄']:.6f}")

    return model1, model2, model_judge, model_fan


def partner_analysis(contestant):
    """舞伴详细分析"""
    print("\n" + "="*60)
    print("舞伴分析")
    print("="*60)

    # 舞伴统计
    partner_stats = contestant.groupby('伴侣名').agg({
        '最终名次': 'mean',
        '是否获奖': ['sum', 'mean'],
        '人名': 'count',
        '平均粉丝投票': 'mean',
        '平均评委评分': 'mean'
    }).round(3)
    partner_stats.columns = ['平均名次', '获奖次数', '获奖率', '参赛次数', '平均粉丝投票', '平均评委评分']

    # 只保留参赛次数>=3的舞伴
    partner_stats = partner_stats[partner_stats['参赛次数'] >= 3]
    partner_stats = partner_stats.sort_values('平均名次')

    # 保存
    partner_stats.to_csv(os.path.join(OUTPUT_DIR, 'partner_ranking.csv'), encoding='utf-8-sig')

    print(f"\n舞伴排名（参赛>=3次，按平均名次排序）:")
    print(partner_stats[['平均名次', '获奖次数', '获奖率', '参赛次数']].head(15))

    print(f"\n最佳舞伴 Top 5:")
    for i, (name, row) in enumerate(partner_stats.head(5).iterrows(), 1):
        print(f"  {i}. {name}: 平均名次={row['平均名次']:.1f}, 获奖率={row['获奖率']:.1%}, 参赛{int(row['参赛次数'])}次")

    return partner_stats


def create_visualizations(contestant, partner_stats, industry_stats, country_stats, region_stats):
    """生成可视化图表"""
    print("\n" + "="*60)
    print("生成可视化图表")
    print("="*60)

    # 1. 行业-名次箱线图
    fig, ax = plt.subplots(figsize=(14, 6))
    # 只显示人数>=10的行业
    major_industries = contestant.groupby('行业').filter(lambda x: len(x) >= 10)
    order = major_industries.groupby('行业')['最终名次'].mean().sort_values().index
    sns.boxplot(x='行业', y='最终名次', data=major_industries, order=order, ax=ax)
    ax.set_title('Industry vs Final Placement Distribution', fontsize=14)
    ax.set_xlabel('Industry', fontsize=12)
    ax.set_ylabel('Final Placement (lower is better)', fontsize=12)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, 'industry_placement_boxplot.png'), dpi=300)
    plt.close()
    print("  1. industry_placement_boxplot.png")

    # 2. 年龄-名次散点图
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.regplot(x='年龄', y='最终名次', data=contestant, ax=ax,
                scatter_kws={'alpha': 0.5}, line_kws={'color': 'red'})
    r, p = stats.pearsonr(contestant['年龄'], contestant['最终名次'])
    ax.set_title(f'Age vs Final Placement (r={r:.3f}, p={p:.4f})', fontsize=14)
    ax.set_xlabel('Age', fontsize=12)
    ax.set_ylabel('Final Placement (lower is better)', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, 'age_placement_scatter.png'), dpi=300)
    plt.close()
    print("  2. age_placement_scatter.png")

    # 3. 舞伴排名条形图
    fig, ax = plt.subplots(figsize=(12, 10))
    top_partners = partner_stats.head(20)
    colors = ['green' if x <= 5 else 'steelblue' for x in top_partners['平均名次']]
    top_partners['平均名次'].plot(kind='barh', ax=ax, color=colors)
    ax.set_title('Pro Dancer Ranking by Average Placement (Top 20)', fontsize=14)
    ax.set_xlabel('Average Placement (lower is better)', fontsize=12)
    ax.set_ylabel('Pro Dancer', fontsize=12)
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, 'partner_ranking_bar.png'), dpi=300)
    plt.close()
    print("  3. partner_ranking_bar.png")

    # 4. 行业获奖率条形图
    fig, ax = plt.subplots(figsize=(12, 6))
    # 只显示人数>=10的行业
    major_stats = industry_stats[industry_stats['人数'] >= 10].copy()
    major_stats = major_stats.sort_values('获奖率', ascending=True)
    colors = ['gold' if x >= 0.3 else 'steelblue' for x in major_stats['获奖率']]
    major_stats['获奖率'].plot(kind='barh', ax=ax, color=colors)
    ax.set_title('Win Rate by Industry (Top 3 Placement)', fontsize=14)
    ax.set_xlabel('Win Rate', fontsize=12)
    ax.set_ylabel('Industry', fontsize=12)
    ax.axvline(x=major_stats['获奖率'].mean(), color='red', linestyle='--', label='Average')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, 'industry_winrate_bar.png'), dpi=300)
    plt.close()
    print("  4. industry_winrate_bar.png")

    # 5. 年龄分布与获奖关系
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 5a. 获奖vs未获奖的年龄分布
    ax = axes[0]
    contestant[contestant['是否获奖']]['年龄'].hist(ax=ax, bins=20, alpha=0.7, label='Top 3', color='gold')
    contestant[~contestant['是否获奖']]['年龄'].hist(ax=ax, bins=20, alpha=0.7, label='Others', color='steelblue')
    ax.set_title('Age Distribution: Winners vs Others', fontsize=12)
    ax.set_xlabel('Age', fontsize=10)
    ax.set_ylabel('Count', fontsize=10)
    ax.legend()

    # 5b. 年龄组的获奖率
    ax = axes[1]
    contestant['年龄组'] = pd.cut(contestant['年龄'], bins=[0, 25, 35, 45, 55, 100],
                                  labels=['<25', '25-35', '35-45', '45-55', '55+'])
    age_winrate = contestant.groupby('年龄组')['是否获奖'].mean()
    age_winrate.plot(kind='bar', ax=ax, color='steelblue')
    ax.set_title('Win Rate by Age Group', fontsize=12)
    ax.set_xlabel('Age Group', fontsize=10)
    ax.set_ylabel('Win Rate', fontsize=10)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    ax.axhline(y=contestant['是否获奖'].mean(), color='red', linestyle='--', label='Average')
    ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, 'age_analysis.png'), dpi=300)
    plt.close()
    print("  5. age_analysis.png")

    # 6. 评委vs粉丝对比
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 6a. 年龄与评委评分
    ax = axes[0]
    sns.regplot(x='年龄', y='平均评委评分', data=contestant, ax=ax,
                scatter_kws={'alpha': 0.5}, line_kws={'color': 'red'})
    r, p = stats.pearsonr(contestant['年龄'], contestant['平均评委评分'])
    ax.set_title(f'Age vs Judge Score (r={r:.3f})', fontsize=12)
    ax.set_xlabel('Age', fontsize=10)
    ax.set_ylabel('Average Judge Score %', fontsize=10)

    # 6b. 年龄与粉丝投票
    ax = axes[1]
    sns.regplot(x='年龄', y='平均粉丝投票', data=contestant, ax=ax,
                scatter_kws={'alpha': 0.5}, line_kws={'color': 'red'})
    r, p = stats.pearsonr(contestant['年龄'], contestant['平均粉丝投票'])
    ax.set_title(f'Age vs Fan Vote (r={r:.3f})', fontsize=12)
    ax.set_xlabel('Age', fontsize=10)
    ax.set_ylabel('Average Fan Vote %', fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, 'judge_vs_fan_age.png'), dpi=300)
    plt.close()
    print("  6. judge_vs_fan_age.png")

    # 7. 美国 vs 非美国分析
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 7a. 人数和获奖人数对比
    ax = axes[0]
    contestant['是否美国'] = contestant['国家'] == 'United States'
    us_stats = contestant.groupby('是否美国').agg({
        '人名': 'count',
        '是否获奖': 'sum'
    })
    us_stats.index = ['Non-US', 'US']
    x = range(len(us_stats))
    width = 0.35
    ax.bar([i - width/2 for i in x], us_stats['人名'], width, label='Total', color='steelblue')
    ax.bar([i + width/2 for i in x], us_stats['是否获奖'], width, label='Winners', color='gold')
    ax.set_xticks(x)
    ax.set_xticklabels(us_stats.index)
    ax.set_title('US vs Non-US: Contestants Count', fontsize=12)
    ax.set_ylabel('Count', fontsize=10)
    ax.legend()

    # 7b. 获奖率对比
    ax = axes[1]
    winrate = contestant.groupby('是否美国')['是否获奖'].mean()
    winrate.index = ['Non-US', 'US']
    colors = ['gold' if x >= 0.3 else 'steelblue' for x in winrate]
    winrate.plot(kind='bar', ax=ax, color=colors)
    ax.set_title('US vs Non-US: Win Rate', fontsize=12)
    ax.set_ylabel('Win Rate', fontsize=10)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    ax.axhline(y=contestant['是否获奖'].mean(), color='red', linestyle='--', label='Average')
    ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, 'country_analysis.png'), dpi=300)
    plt.close()
    print("  7. country_analysis.png")

    # 8. 美国各州分析
    fig, ax = plt.subplots(figsize=(14, 8))
    top_regions = region_stats[region_stats['人数'] >= 5].sort_values('获奖率', ascending=True)
    if len(top_regions) > 0:
        colors = ['gold' if x >= 0.3 else 'steelblue' for x in top_regions['获奖率']]
        top_regions['获奖率'].plot(kind='barh', ax=ax, color=colors)
        ax.set_title('Win Rate by US State (n>=5)', fontsize=14)
        ax.set_xlabel('Win Rate', fontsize=12)
        ax.set_ylabel('State', fontsize=12)
        ax.axvline(x=contestant['是否获奖'].mean(), color='red', linestyle='--', label='Average')
        ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, 'region_analysis.png'), dpi=300)
    plt.close()
    print("  8. region_analysis.png")

    print(f"\n图表已保存到: {FIGURE_DIR}")


def main():
    print("="*60)
    print("MCM Problem C 第三问：名人属性与比赛结果分析")
    print("="*60)

    # 加载数据
    df, contestant = load_data()

    # 描述性统计
    industry_stats, country_stats, region_stats = descriptive_stats(contestant)

    # 假设检验
    tests = hypothesis_tests(contestant)

    # 回归分析
    models = regression_analysis(contestant)

    # 舞伴分析
    partner_stats = partner_analysis(contestant)

    # 可视化
    create_visualizations(contestant, partner_stats, industry_stats, country_stats, region_stats)

    print("\n" + "="*60)
    print("分析完成！")
    print("="*60)
    print(f"\n输出文件:")
    print(f"  - {OUTPUT_DIR}/descriptive_stats.csv")
    print(f"  - {OUTPUT_DIR}/country_stats.csv")
    print(f"  - {OUTPUT_DIR}/region_stats.csv")
    print(f"  - {OUTPUT_DIR}/hypothesis_tests.csv")
    print(f"  - {OUTPUT_DIR}/regression_results.txt")
    print(f"  - {OUTPUT_DIR}/partner_ranking.csv")
    print(f"  - {FIGURE_DIR}/*.png")


if __name__ == '__main__':
    main()
