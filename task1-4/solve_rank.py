from __future__ import annotations

import argparse
import json
import os
from decimal import Decimal
from typing import Dict, List

import pandas as pd

from data_processing import build_season_weeks, get_week_columns
from model_rank import solve_season


DEFAULT_SEASONS = [1, 2, 28, 29, 30, 31, 32, 33, 34]


def parse_weights(alpha: str, beta: str, gamma: str) -> Dict[str, float]:
    return {
        "alpha": float(Decimal(alpha)),
        "beta": float(Decimal(beta)),
        "gamma": float(Decimal(gamma)),
    }


def scale_weights(alpha: str, beta: str, gamma: str) -> Dict[str, int]:
    weights = [Decimal(alpha), Decimal(beta), Decimal(gamma)]
    max_decimals = max(0, max(-w.as_tuple().exponent for w in weights))
    scale = 10 ** max_decimals
    scaled = [int((w * scale).to_integral_value()) for w in weights]
    return {
        "scaled": {"alpha": scaled[0], "beta": scaled[1], "gamma": scaled[2]},
        "scale": scale,
    }


def predicted_set(week: Dict, rF: Dict) -> List[int]:
    week_num = week["week"]
    contestants = week["contestants"]
    scores = []
    for i in contestants:
        rJ = week["judge_rank"][i]
        rF_val = rF[(week_num, i)]
        scores.append((i, rJ + rF_val))
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        default="/home/hisheep/d/MCM/26/Data_4.xlsx",
        help="Path to Data_4.xlsx",
    )
    parser.add_argument(
        "--output-dir",
        default="/home/hisheep/d/MCM/26/task1-4/outputs",
        help="Output directory",
    )
    parser.add_argument(
        "--seasons",
        nargs="*",
        type=int,
        default=DEFAULT_SEASONS,
        help="Seasons to solve",
    )
    parser.add_argument("--alpha", default="1")
    parser.add_argument("--beta", default="0.1")
    parser.add_argument("--gamma", default="10")
    parser.add_argument("--time-limit", type=float, default=60.0)
    parser.add_argument("--workers", type=int, default=8)

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    df = pd.read_excel(args.data)
    week_cols = get_week_columns(df)

    weights = parse_weights(args.alpha, args.beta, args.gamma)
    scaled_info = scale_weights(args.alpha, args.beta, args.gamma)
    weights_scaled = scaled_info["scaled"]
    weight_scale = scaled_info["scale"]

    weekly_rows = []
    consistency_rows = []
    penalty_rows = []
    season_info = {}

    for season in args.seasons:
        weeks = build_season_weeks(df, season, week_cols)
        if not weeks:
            season_info[season] = {"status": "NO_DATA"}
            continue

        result = solve_season(
            weeks=weeks,
            weights_scaled=weights_scaled,
            weight_scale=weight_scale,
            time_limit=args.time_limit,
            num_workers=args.workers,
        )

        season_info[season] = {
            "status": result["status"],
            "objective_scaled": result["objective_scaled"],
            "objective": result["objective"],
        }

        if result["status"] not in ("OPTIMAL", "FEASIBLE"):
            continue

        rF = result["rF"]

        week_map = {w["week"]: w for w in weeks}
        week_numbers = sorted(week_map.keys())
        week_consistent = []

        for week_num in week_numbers:
            week = week_map[week_num]
            eliminated = week["eliminated_ids"]
            k = len(eliminated)

            scores = predicted_set(week, rF)
            predicted_ids = set()
            if k > 0 and scores:
                threshold = scores[min(k, len(scores)) - 1][1]
                predicted_ids = {i for i, R in scores if R >= threshold}

            for i in week["contestants"]:
                row = {
                    "season": season,
                    "week": week_num,
                    "contestant_id": i,
                    "celebrity_name": week["names"][i],
                    "ballroom_partner": week["partners"][i],
                    "judge_score": float(week["judge_score"][i]),
                    "judge_rank": int(week["judge_rank"][i]),
                    "fan_rank": int(rF[(week_num, i)]),
                    "combined_rank": int(
                        week["judge_rank"][i] + rF[(week_num, i)]
                    ),
                    "actual_eliminated": int(i in eliminated),
                    "predicted_eliminated": int(i in predicted_ids)
                    if k > 0
                    else 0,
                    "slack_sum": int(
                        result["slack_by_week_contestant"].get((week_num, i), 0)
                    ),
                }
                weekly_rows.append(row)

            if k > 0:
                actual_set = set(eliminated)
                consistent = int(actual_set.issubset(predicted_ids))
                hit_rate = (
                    len(actual_set.intersection(predicted_ids)) / k
                    if k > 0
                    else 0.0
                )
                week_consistent.append(consistent)

                consistency_rows.append(
                    {
                        "season": season,
                        "week": week_num,
                        "k_actual": k,
                        "predicted_size": len(predicted_ids),
                        "consistent": consistent,
                        "hit_rate": round(hit_rate, 4),
                        "actual_eliminated": ";".join(
                            week["names"][i] for i in eliminated
                        ),
                        "predicted_eliminated": ";".join(
                            week["names"][i] for i in sorted(predicted_ids)
                        ),
                    }
                )

            penalty = result["week_terms"].get(
                week_num, {"jterm": 0, "smooth": 0, "slack": 0}
            )
            penalty_rows.append(
                {
                    "season": season,
                    "week": week_num,
                    "jterm": penalty["jterm"],
                    "smooth": penalty["smooth"],
                    "slack": penalty["slack"],
                }
            )

        if week_consistent:
            consistency_rows.append(
                {
                    "season": season,
                    "week": "ALL",
                    "k_actual": "",
                    "predicted_size": "",
                    "consistent": round(sum(week_consistent) / len(week_consistent), 4),
                    "hit_rate": "",
                    "actual_eliminated": "",
                    "predicted_eliminated": "",
                }
            )

    pd.DataFrame(weekly_rows).to_csv(
        os.path.join(args.output_dir, "weekly_predictions.csv"), index=False
    )
    pd.DataFrame(consistency_rows).to_csv(
        os.path.join(args.output_dir, "consistency_summary.csv"), index=False
    )
    pd.DataFrame(penalty_rows).to_csv(
        os.path.join(args.output_dir, "weekly_penalty.csv"), index=False
    )

    info = {
        "weights": weights,
        "weights_scaled": weights_scaled,
        "weight_scale": weight_scale,
        "time_limit": args.time_limit,
        "workers": args.workers,
        "seasons": args.seasons,
        "season_info": season_info,
        "notes": [
            "Elimination constraints only applied when results contains 'Eliminated Week X'.",
            "Weeks with no active contestants are skipped.",
        ],
    }

    with open(
        os.path.join(args.output_dir, "optimization_info.json"),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(info, f, indent=2, ensure_ascii=True)


if __name__ == "__main__":
    main()
