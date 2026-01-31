import re
import pandas as pd
import numpy as np


_WEEK_AGG_SCORE_RE = re.compile(r"^week(\d+)_judge_score$")
_WEEK_JUDGE_SCORE_RE = re.compile(r"^week(\d+)_judge\d+_score$")


def _extract_week_numbers(columns):
    weeks = set()
    for col in columns:
        m = _WEEK_AGG_SCORE_RE.match(col)
        if m:
            weeks.add(int(m.group(1)))
            continue
        m = _WEEK_JUDGE_SCORE_RE.match(col)
        if m:
            weeks.add(int(m.group(1)))
    return sorted(weeks)


def _judge_score_columns(columns, week):
    agg_col = f"week{week}_judge_score"
    if agg_col in columns:
        return [agg_col]

    judge_cols = [
        col
        for col in columns
        if _WEEK_JUDGE_SCORE_RE.match(col) and col.startswith(f"week{week}_")
    ]
    if judge_cols:
        return judge_cols

    fallback = [
        col
        for col in columns
        if col.startswith(f"week{week}_") and col.endswith("_score")
    ]
    return fallback


def compute_week_raw_scores(season_df, week):
    cols = _judge_score_columns(season_df.columns, week)
    if not cols:
        return pd.Series(np.zeros(len(season_df)), index=season_df.index)
    return season_df[cols].fillna(0).astype(float).sum(axis=1)


def compute_last_week(season_df):
    last_week = 0
    weeks = _extract_week_numbers(season_df.columns)
    for week in weeks:
        raw_scores = compute_week_raw_scores(season_df, week)
        if raw_scores.fillna(0).gt(0).any():
            last_week = max(last_week, week)
    return last_week


def week_slice(season_df, week, last_week):
    raw_scores = compute_week_raw_scores(season_df, week)
    active_mask = raw_scores.fillna(0).gt(0)
    active = season_df.loc[active_mask]
    names = active["celebrity_name"].tolist()
    judge_raw_scores = raw_scores[active_mask].astype(float).to_numpy()

    eliminated_names = []
    eliminated_idx = []
    if week < last_week:
        next_raw = compute_week_raw_scores(season_df, week + 1)
        next_active = next_raw.fillna(0).gt(0)
        eliminated_mask = active_mask & (~next_active)
        eliminated_names = season_df.loc[eliminated_mask, "celebrity_name"].tolist()
        eliminated_idx = [names.index(n) for n in eliminated_names if n in names]

    return names, judge_raw_scores, eliminated_names, eliminated_idx
