import re
import numpy as np


_WEEK_SCORE_RE = re.compile(r"week(\d+)_judge_score$")


def compute_last_week(season_df):
    week_nums = []
    for col in season_df.columns:
        m = _WEEK_SCORE_RE.match(col)
        if not m:
            continue
        week = int(m.group(1))
        if season_df[col].fillna(0).gt(0).any():
            week_nums.append(week)
    return max(week_nums) if week_nums else 0


def week_slice(season_df, week, last_week):
    score_col = f"week{week}_judge_score"
    pct_col = f"{week}_percent"

    active_mask = season_df[score_col].fillna(0).gt(0)
    active = season_df.loc[active_mask]
    names = active["celebrity_name"].tolist()
    judge_pct = active[pct_col].astype(float).to_numpy()

    eliminated_names = []
    eliminated_idx = []
    if week < last_week:
        next_col = f"week{week + 1}_judge_score"
        if next_col in season_df.columns:
            active_next = season_df[next_col].fillna(0).gt(0)
            eliminated_mask = active_mask & (~active_next)
            eliminated_names = season_df.loc[eliminated_mask, "celebrity_name"].tolist()
            eliminated_idx = [names.index(n) for n in eliminated_names if n in names]

    return names, judge_pct, eliminated_names, eliminated_idx
