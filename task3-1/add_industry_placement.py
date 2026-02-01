#!/usr/bin/env python3
"""
为 combined_contestant_info.csv 添加行业和名次字段
"""

import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA4_PATH = os.path.join(BASE_DIR, 'Data_4.xlsx')
CSV_PATH = os.path.join(BASE_DIR, 'task3-1/combined_contestant_info.csv')


def main():
    # 读取现有数据
    df = pd.read_csv(CSV_PATH)
    print(f"原始数据行数: {len(df)}")
    print(f"原始列: {list(df.columns)}")

    # 读取 Data_4.xlsx 获取行业和名次信息
    data4 = pd.read_excel(DATA4_PATH)
    industry_placement = data4[['season', 'celebrity_name', 'celebrity_industry',
                                 'results', 'placement']].drop_duplicates()

    # 合并数据
    df = pd.merge(
        df,
        industry_placement,
        left_on=['赛季', '人名'],
        right_on=['season', 'celebrity_name'],
        how='left'
    )

    # 删除重复的合并键
    df = df.drop(columns=['season', 'celebrity_name'])

    # 重命名新列
    df = df.rename(columns={
        'celebrity_industry': '行业',
        'results': '比赛结果',
        'placement': '最终名次'
    })

    # 重新排列列顺序
    cols = ['赛季', '周数', '人名', '伴侣名', '地区', '国家', '年龄',
            '行业', '最终名次', '比赛结果',
            '评委评分百分比', '粉丝投票百分比', '总百分比']
    df = df[cols]

    # 保存
    df.to_csv(CSV_PATH, index=False, encoding='utf-8-sig')
    print(f"\n更新后数据行数: {len(df)}")
    print(f"更新后列: {list(df.columns)}")
    print(f"\n行业分布:")
    print(df['行业'].value_counts())
    print(f"\n名次分布:")
    print(df['最终名次'].value_counts().sort_index().head(10))


if __name__ == '__main__':
    main()
