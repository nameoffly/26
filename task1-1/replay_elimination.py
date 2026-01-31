import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import data_processing


def predict_eliminated_names(week_df, k):
    if k <= 0:
        return []
    df = week_df.copy()
    df["combined"] = df["fan_mean"].astype(float) + df["judge_pct"].astype(float)
    df = df.sort_values(["combined", "celebrity_name"], ascending=[True, True])
    return df["celebrity_name"].head(k).tolist()


def evaluate_week(week_df, actual_eliminated):
    k = len(actual_eliminated)
    if k == 0:
        return [], "NA"
    predicted = predict_eliminated_names(week_df, k)
    match = set(predicted) == set(actual_eliminated)
    return predicted, bool(match)


def build_actual_elimination_map(data_path, seasons, weeks_by_season):
    df = pd.read_excel(data_path)
    actual = {}
    for season in seasons:
        season_df = df[df["season"] == season].copy()
        if season_df.empty:
            continue
        last_week = data_processing.compute_last_week(season_df)
        if last_week == 0:
            continue
        target_weeks = weeks_by_season.get(season, set())
        for week in range(1, last_week + 1):
            if target_weeks and week not in target_weeks:
                continue
            _, _, eliminated_names, _ = data_processing.week_slice(
                season_df, week=week, last_week=last_week
            )
            actual[(season, week)] = eliminated_names
    return actual


def build_week_results(contestant_df, actual_map):
    rows = []
    for (season, week), week_df in contestant_df.groupby(["season", "week"]):
        actual_elim = actual_map.get((season, week), [])
        predicted, match = evaluate_week(week_df, actual_elim)
        rows.append(
            {
                "season": season,
                "week": week,
                "predicted_eliminated_names": ";".join(predicted),
                "actual_eliminated_names": ";".join(actual_elim),
                "match": match,
            }
        )
    return pd.DataFrame(rows)


def compute_consistency_stats(results_df):
    valid = results_df[results_df["match"].isin([True, False])]
    overall_total = len(valid)
    overall_matched = int(valid["match"].sum())
    overall_acc = overall_matched / overall_total if overall_total else np.nan

    rows = [
        {
            "scope": "overall",
            "season": "",
            "matched": overall_matched,
            "total": overall_total,
            "accuracy": overall_acc,
        }
    ]

    for season, g in valid.groupby("season"):
        total = len(g)
        matched = int(g["match"].sum())
        acc = matched / total if total else np.nan
        rows.append(
            {
                "scope": "season",
                "season": season,
                "matched": matched,
                "total": total,
                "accuracy": acc,
            }
        )

    return pd.DataFrame(rows)


def run_replay(
    data_path,
    contestant_summary_path,
    weekly_summary_path,
    output_dir,
):
    contestant_df = pd.read_csv(contestant_summary_path)
    weeks_by_season = (
        contestant_df.groupby("season")["week"].apply(lambda s: set(s)).to_dict()
    )
    seasons = sorted(weeks_by_season.keys())
    actual_map = build_actual_elimination_map(data_path, seasons, weeks_by_season)

    results_df = build_week_results(contestant_df, actual_map)

    weekly_df = pd.read_csv(weekly_summary_path)
    merged = weekly_df.merge(results_df, on=["season", "week"], how="left")
    merged.to_csv(weekly_summary_path, index=False)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stats_df = compute_consistency_stats(results_df)
    stats_df.to_csv(output_dir / "replay_consistency_summary.csv", index=False)

    mismatches = results_df[results_df["match"] == False]  # noqa: E712
    mismatches.to_csv(output_dir / "replay_mismatches.csv", index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        default="/home/hisheep/d/MCM/26/Data_4.xlsx",
        help="原始数据文件路径",
    )
    parser.add_argument(
        "--contestant-summary",
        default="/home/hisheep/d/MCM/26/task1-1/outputs/contestant_fan_vote_summary.csv",
    )
    parser.add_argument(
        "--weekly-summary",
        default="/home/hisheep/d/MCM/26/task1-1/outputs/weekly_summary.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="/home/hisheep/d/MCM/26/task1-1/outputs",
    )
    args = parser.parse_args()

    run_replay(
        args.data, args.contestant_summary, args.weekly_summary, args.output_dir
    )


if __name__ == "__main__":
    main()
