from __future__ import annotations

import re
from typing import Dict, List, Optional

import pandas as pd

WEEK_COL_RE = re.compile(r"week(\d+)_judge_score", re.IGNORECASE)
ELIM_RE = re.compile(r"Eliminated\s+Week\s+(\d+)", re.IGNORECASE)
FINAL_RE = re.compile(r"(\d+)(st|nd|rd|th)\s+Place", re.IGNORECASE)


def get_week_columns(df: pd.DataFrame) -> List[str]:
    week_cols = []
    for col in df.columns:
        match = WEEK_COL_RE.match(str(col))
        if match:
            week_cols.append((int(match.group(1)), col))
    week_cols.sort(key=lambda x: x[0])
    return [col for _, col in week_cols]


def parse_elimination_week(result_value: object) -> Optional[int]:
    if result_value is None or (isinstance(result_value, float) and pd.isna(result_value)):
        return None
    result_str = str(result_value)
    match = ELIM_RE.search(result_str)
    if match:
        return int(match.group(1))
    return None


def parse_final_place(result_value: object) -> Optional[int]:
    if result_value is None or (isinstance(result_value, float) and pd.isna(result_value)):
        return None
    result_str = str(result_value)
    match = FINAL_RE.search(result_str)
    if match:
        return int(match.group(1))
    return None


def build_season_weeks(
    df: pd.DataFrame, season: int, week_cols: List[str]
) -> List[Dict]:
    season_df = df[df["season"] == season].copy()
    if season_df.empty:
        return []

    season_df = season_df.reset_index(drop=True)
    season_df["contestant_id"] = season_df.index.astype(int)
    season_df["elim_week"] = season_df["results"].apply(parse_elimination_week)
    season_df["final_place"] = season_df["results"].apply(parse_final_place)

    weeks = []
    for idx, col in enumerate(week_cols, start=1):
        scores = season_df[col]
        active_mask = scores.notna() & (scores > 0)
        if active_mask.sum() == 0:
            continue

        week_df = season_df.loc[
            active_mask,
            [
                "contestant_id",
                "celebrity_name",
                "ballroom_partner",
                "results",
                "elim_week",
                "final_place",
                col,
            ],
        ].copy()

        week_df = week_df.rename(columns={col: "judge_score"})
        week_df["judge_rank"] = (
            (-week_df["judge_score"]).rank(method="min").astype(int)
        )

        eliminated_ids = week_df.loc[
            week_df["elim_week"] == idx, "contestant_id"
        ].tolist()
        final_places = week_df.dropna(subset=["final_place"])
        final_place_map = {
            int(row.contestant_id): int(row.final_place)
            for row in final_places.itertuples(index=False)
        }

        weeks.append(
            {
                "season": season,
                "week": idx,
                "contestants": week_df["contestant_id"].tolist(),
                "judge_score": dict(
                    zip(week_df["contestant_id"], week_df["judge_score"])
                ),
                "judge_rank": dict(
                    zip(week_df["contestant_id"], week_df["judge_rank"])
                ),
                "names": dict(
                    zip(
                        week_df["contestant_id"],
                        week_df["celebrity_name"],
                    )
                ),
                "partners": dict(
                    zip(
                        week_df["contestant_id"],
                        week_df["ballroom_partner"],
                    )
                ),
                "eliminated_ids": eliminated_ids,
                "final_places": final_place_map,
            }
        )

    return weeks
