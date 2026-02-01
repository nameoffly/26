#!/usr/bin/env python3
"""
合并数据并追加赛季3-27的数据
"""

import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA4_PATH = os.path.join(BASE_DIR, 'Data_4.xlsx')
FAN_VOTE_PATH_1 = os.path.join(BASE_DIR, 'task1-4/final_outputs/baseline/grid_a1p0_b0p05_g10p0/fan_vote_percent.csv')
FAN_VOTE_PATH_2 = os.path.join(BASE_DIR, 'fan_vote_estimates_entropy_smooth_150.csv')
OUTPUT_PATH = os.path.join(BASE_DIR, 'task3-1/combined_contestant_info.csv')


def process_seasons_1_2_28_34(data4):
    """处理赛季1, 2, 28-34的数据"""
    fan_vote = pd.read_csv(FAN_VOTE_PATH_1)

    base_cols = ['celebrity_name', 'ballroom_partner', 'celebrity_homestate',
                 'celebrity_homecountry/region', 'celebrity_age_during_season', 'season']
    percent_cols = [f'{i}_percent' for i in range(1, 12)]

    data4_long = data4.melt(
        id_vars=base_cols,
        value_vars=percent_cols,
        var_name='week_str',
        value_name='judge_percent'
    )
    data4_long['week'] = data4_long['week_str'].str.extract(r'(\d+)').astype(int)

    merged = pd.merge(
        fan_vote[['season', 'week', 'celebrity_name', 'ballroom_partner', 'fan_vote_percent']],
        data4_long[['season', 'week', 'celebrity_name', 'celebrity_homestate',
                    'celebrity_homecountry/region', 'celebrity_age_during_season', 'judge_percent']],
        on=['season', 'week', 'celebrity_name'],
        how='left'
    )

    merged['total_percent'] = merged['judge_percent'] + merged['fan_vote_percent']
    return merged


def process_seasons_3_27(data4):
    """处理赛季3-27的数据"""
    fan_vote = pd.read_csv(FAN_VOTE_PATH_2)

    # 从 Data_4.xlsx 获取选手基本信息
    data4_info = data4[['celebrity_name', 'season', 'ballroom_partner',
                        'celebrity_homestate', 'celebrity_homecountry/region',
                        'celebrity_age_during_season']].drop_duplicates()

    # 合并数据
    merged = pd.merge(
        fan_vote[['season', 'week', 'celebrity_name', 'judge_percent', 'fan_vote_percent']],
        data4_info,
        on=['season', 'celebrity_name'],
        how='left'
    )

    merged['total_percent'] = merged['judge_percent'] + merged['fan_vote_percent']
    return merged


def main():
    data4 = pd.read_excel(DATA4_PATH)

    # 处理两部分数据
    df1 = process_seasons_1_2_28_34(data4)
    df2 = process_seasons_3_27(data4)

    # 合并两部分
    combined = pd.concat([df1, df2], ignore_index=True)

    # 合并行业和名次字段
    industry_placement = data4[['season', 'celebrity_name', 'celebrity_industry',
                                 'results', 'placement']].drop_duplicates()
    combined = pd.merge(
        combined,
        industry_placement,
        on=['season', 'celebrity_name'],
        how='left'
    )

    # 重命名列
    result = combined.rename(columns={
        'season': '赛季',
        'week': '周数',
        'celebrity_name': '人名',
        'ballroom_partner': '伴侣名',
        'celebrity_homestate': '地区',
        'celebrity_homecountry/region': '国家',
        'celebrity_age_during_season': '年龄',
        'judge_percent': '评委评分百分比',
        'fan_vote_percent': '粉丝投票百分比',
        'total_percent': '总百分比',
        'celebrity_industry': '行业',
        'results': '比赛结果',
        'placement': '最终名次'
    })

    # 选择并排序列
    result = result[['赛季', '周数', '人名', '伴侣名', '地区', '国家', '年龄',
                     '行业', '最终名次', '比赛结果',
                     '评委评分百分比', '粉丝投票百分比', '总百分比']]

    # 填充地区空值为 others
    result['地区'] = result['地区'].fillna('others')

    # 按赛季和周数排序
    result = result.sort_values(['赛季', '周数', '人名']).reset_index(drop=True)

    # 输出
    result.to_csv(OUTPUT_PATH, index=False, encoding='utf-8-sig')
    print(f"输出文件已保存到: {OUTPUT_PATH}")
    print(f"总行数: {len(result)}")
    print(f"赛季分布:")
    print(result['赛季'].value_counts().sort_index())


if __name__ == '__main__':
    main()
