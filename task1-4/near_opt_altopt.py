from __future__ import annotations

import argparse
import csv
import json
import math
import os
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from ortools.graph.python import linear_sum_assignment

from data_processing import build_season_weeks, get_week_columns


DEFAULT_SEASONS = [1, 2, 28, 29, 30, 31, 32, 33, 34]


def _write_csv(path: str, fieldnames: List[str], rows: Iterable[Dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _solve_week_assignment(cost_matrix: np.ndarray) -> List[int]:
    n = cost_matrix.shape[0]
    assignment = linear_sum_assignment.SimpleLinearSumAssignment()
    for i in range(n):
        for j in range(n):
            assignment.add_arc_with_cost(
                i, j, int(round(cost_matrix[i, j]))
            )
    status = assignment.solve()
    if status != assignment.OPTIMAL:
        raise RuntimeError("Assignment solver failed")
    ranks = [0] * n
    for i in range(n):
        ranks[i] = assignment.right_mate(i) + 1
    return ranks


def _altopt_solve(
    weeks: List[Dict],
    alpha: float,
    beta: float,
    gamma: float,
    sweeps: int,
    cost_scale: int,
    baseline_fan_rank: Dict[Tuple[int, int], Dict[int, int]],
) -> Dict[Tuple[int, int], int]:
    week_map = {w["week"]: w for w in weeks}
    week_numbers = sorted(week_map.keys())
    season = weeks[0]["season"]
    rF: Dict[Tuple[int, int], int] = {}

    for week_num in week_numbers:
        week = week_map[week_num]
        base_week = baseline_fan_rank.get((season, week_num), {})
        for contestant_id in week["contestants"]:
            init_rank = base_week.get(contestant_id)
            if init_rank is None:
                init_rank = int(week["judge_rank"][contestant_id])
            rF[(week_num, contestant_id)] = int(init_rank)

    for _ in range(sweeps):
        for week_num in week_numbers:
            week = week_map[week_num]
            contestants = week["contestants"]
            n_w = len(contestants)
            if n_w == 0:
                continue
            cost = np.zeros((n_w, n_w), dtype=float)
            for row, contestant_id in enumerate(contestants):
                rJ = int(week["judge_rank"][contestant_id])
                r_prev = rF.get((week_num - 1, contestant_id))
                r_next = rF.get((week_num + 1, contestant_id))
                is_elim = contestant_id in week["eliminated_ids"]
                for col in range(n_w):
                    k = col + 1
                    c = alpha * (k - rJ) ** 2
                    if r_prev is not None:
                        c += beta * (k - r_prev) ** 2
                    if r_next is not None:
                        c += beta * (k - r_next) ** 2
                    if is_elim:
                        c += gamma * (n_w - k)
                    cost[row, col] = c * cost_scale
            ranks = _solve_week_assignment(cost)
            for row, contestant_id in enumerate(contestants):
                rF[(week_num, contestant_id)] = ranks[row]

    return rF


def _objective_value(
    weeks: List[Dict],
    rF: Dict[Tuple[int, int], int],
    alpha: float,
    beta: float,
    gamma: float,
) -> float:
    week_map = {w["week"]: w for w in weeks}
    week_numbers = sorted(week_map.keys())

    jterm = 0.0
    smooth = 0.0
    slack = 0.0

    for week_num in week_numbers:
        week = week_map[week_num]
        contestants = week["contestants"]
        for i in contestants:
            rJ = int(week["judge_rank"][i])
            rF_val = rF[(week_num, i)]
            jterm += (rF_val - rJ) ** 2

        eliminated = week["eliminated_ids"]
        if eliminated:
            for e in eliminated:
                rJ_e = int(week["judge_rank"][e])
                R_e = rJ_e + rF[(week_num, e)]
                for j in contestants:
                    if j == e:
                        continue
                    rJ_j = int(week["judge_rank"][j])
                    R_j = rJ_j + rF[(week_num, j)]
                    slack += max(0.0, (R_j + 1) - R_e)

    for week_num in week_numbers:
        prev_week = week_num - 1
        if prev_week not in week_map:
            continue
        week = week_map[week_num]
        prev = week_map[prev_week]
        shared = set(week["contestants"]).intersection(prev["contestants"])
        for i in shared:
            smooth += (rF[(week_num, i)] - rF[(prev_week, i)]) ** 2

    return alpha * jterm + beta * smooth + gamma * slack


def _compute_interval_certainty(df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict] = []
    group_keys = ["season", "week", "epsilon"]
    for (season, week, epsilon), group in df.groupby(group_keys):
        k_actual = int(group["actual_eliminated"].sum())
        has_nan = (
            group["combined_rank_min"].isna().any()
            or group["combined_rank_max"].isna().any()
        )
        r_min = group["combined_rank_min"].to_numpy()
        r_max = group["combined_rank_max"].to_numpy()

        for idx in range(len(group)):
            r_min_i = r_min[idx]
            r_max_i = r_max[idx]
            if pd.isna(r_min_i) or pd.isna(r_max_i):
                status = "unsolved"
                worse_def = None
                worse_pos = None
            elif has_nan:
                status = "uncertain"
                worse_def = None
                worse_pos = None
            elif k_actual == 0:
                status = "no_elimination"
                worse_def = None
                worse_pos = None
            else:
                worse_def = int(
                    sum(
                        1
                        for j in range(len(group))
                        if j != idx and r_min[j] > r_max_i
                    )
                )
                worse_pos = int(
                    sum(
                        1
                        for j in range(len(group))
                        if j != idx and r_max[j] > r_min_i
                    )
                )
                if worse_def >= k_actual:
                    status = "always_safe"
                elif worse_pos <= k_actual - 1:
                    status = "always_eliminated"
                else:
                    status = "uncertain"

            row = group.iloc[idx].to_dict()
            row.update(
                {
                    "k_actual": k_actual,
                    "status": status,
                    "worse_definite": worse_def,
                    "worse_possible": worse_pos,
                }
            )
            rows.append(row)

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        default="/home/hisheep/d/MCM/26/Data_4.xlsx",
        help="Path to Data_4.xlsx",
    )
    parser.add_argument(
        "--baseline-dir",
        default="/home/hisheep/d/MCM/26/task1-4/outputs/grid_a1p0_b0p05_g10p0",
        help="Directory with baseline weekly_predictions.csv and optimization_info.json",
    )
    parser.add_argument(
        "--output-dir",
        default="/home/hisheep/d/MCM/26/task1-4/outputs_uncertainty_altopt/near_opt_altopt",
        help="Output directory for altopt near-opt analysis",
    )
    parser.add_argument(
        "--seasons",
        nargs="*",
        type=int,
        default=DEFAULT_SEASONS,
        help="Seasons to analyze",
    )
    parser.add_argument("--alpha", default="1.0")
    parser.add_argument("--beta", default="0.05")
    parser.add_argument("--gamma", default="10.0")
    parser.add_argument(
        "--epsilons",
        default="0.01,0.05,0.1",
        help="Comma-separated epsilon list",
    )
    parser.add_argument(
        "--sweeps",
        type=int,
        default=5,
        help="Number of alternating optimization sweeps",
    )
    parser.add_argument(
        "--cost-scale",
        type=int,
        default=1000,
        help="Scale factor for assignment costs (int)",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=10,
        help="Number of candidate solutions per season",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--rerun",
        action="store_true",
        help="Recompute even if output exists",
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    interval_path = os.path.join(args.output_dir, "near_opt_interval.csv")
    certainty_path = os.path.join(
        args.output_dir, "near_opt_elim_certainty.csv"
    )
    if (
        not args.rerun
        and os.path.exists(interval_path)
        and os.path.exists(certainty_path)
    ):
        return

    epsilons = [
        float(part.strip())
        for part in args.epsilons.split(",")
        if part.strip()
    ]

    baseline_pred_path = os.path.join(args.baseline_dir, "weekly_predictions.csv")
    baseline_opt_path = os.path.join(args.baseline_dir, "optimization_info.json")
    baseline_pred = pd.read_csv(baseline_pred_path)
    baseline_opt = json.load(open(baseline_opt_path, "r", encoding="utf-8"))

    baseline_fan_rank: Dict[Tuple[int, int], Dict[int, int]] = {}
    for row in baseline_pred.itertuples(index=False):
        key = (int(row.season), int(row.week))
        baseline_fan_rank.setdefault(key, {})[int(row.contestant_id)] = int(
            row.fan_rank
        )

    df = pd.read_excel(args.data)
    week_cols = get_week_columns(df)

    rows: List[Dict] = []
    for season in args.seasons:
        weeks = build_season_weeks(df, season, week_cols)
        if not weeks:
            continue

        opt = baseline_opt.get("season_info", {}).get(str(season), {}).get(
            "objective"
        )
        if opt is None:
            continue

        rng = np.random.default_rng(args.seed + season)
        candidates: List[Tuple[float, Dict[Tuple[int, int], int]]] = []
        for _ in range(args.n_samples):
            rF = _altopt_solve(
                weeks,
                alpha=float(args.alpha),
                beta=float(args.beta),
                gamma=float(args.gamma),
                sweeps=args.sweeps,
                cost_scale=args.cost_scale,
                baseline_fan_rank=baseline_fan_rank,
            )
            obj = _objective_value(
                weeks,
                rF,
                alpha=float(args.alpha),
                beta=float(args.beta),
                gamma=float(args.gamma),
            )
            candidates.append((obj, rF))

            if args.n_samples > 1:
                jitter = rng.integers(0, 2)
                if jitter:
                    for key in rF:
                        rF[key] = max(1, rF[key] + int(rng.integers(-1, 2)))

        for eps in epsilons:
            bound = opt * (1.0 + eps)
            selected = [rF for obj, rF in candidates if obj <= bound]
            if not selected:
                continue

            week_map = {w["week"]: w for w in weeks}
            for week_num, week in week_map.items():
                contestants = week["contestants"]
                for contestant_id in contestants:
                    rF_vals = [
                        rF[(week_num, contestant_id)] for rF in selected
                    ]
                    rJ = int(week["judge_rank"][contestant_id])
                    rF_min = min(rF_vals)
                    rF_max = max(rF_vals)
                    rows.append(
                        {
                            "season": season,
                            "week": week_num,
                            "contestant_id": contestant_id,
                            "celebrity_name": week["names"][contestant_id],
                            "ballroom_partner": week["partners"][contestant_id],
                            "judge_rank": rJ,
                            "fan_rank_min": rF_min,
                            "fan_rank_max": rF_max,
                            "combined_rank_min": rJ + rF_min,
                            "combined_rank_max": rJ + rF_max,
                            "actual_eliminated": int(
                                contestant_id in week["eliminated_ids"]
                            ),
                            "epsilon": eps,
                        }
                    )

    interval_df = pd.DataFrame(rows)
    interval_df.to_csv(interval_path, index=False)

    certainty_df = _compute_interval_certainty(interval_df)
    certainty_df.to_csv(certainty_path, index=False)

    summary_path = os.path.join(args.output_dir, "near_opt_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "baseline_dir": args.baseline_dir,
                "epsilons": epsilons,
                "n_samples": args.n_samples,
                "sweeps": args.sweeps,
                "alpha": float(args.alpha),
                "beta": float(args.beta),
                "gamma": float(args.gamma),
            },
            f,
            indent=2,
        )


if __name__ == "__main__":
    main()
