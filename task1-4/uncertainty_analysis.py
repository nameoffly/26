from __future__ import annotations

import argparse
import csv
import json
import math
import os
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from ortools.sat.python import cp_model

from data_processing import build_season_weeks, get_week_columns, parse_elimination_week
from model_rank import build_rank_model, solve_season
from solve_rank import DEFAULT_SEASONS, scale_weights


NEAR_DF: Optional[pd.DataFrame] = None
NEAR_WEEK_COLS: Optional[List[str]] = None
NEAR_SEASON_CACHE: Dict[int, List[Dict]] = {}
NEAR_WEIGHTS_SCALED: Optional[Dict[str, int]] = None

PERTURB_DF: Optional[pd.DataFrame] = None
PERTURB_WEEK_COLS: Optional[List[str]] = None
PERTURB_WEIGHTS_SCALED: Optional[Dict[str, int]] = None
PERTURB_WEIGHT_SCALE: Optional[int] = None
PERTURB_BASELINE_BY_WEEK: Dict[Tuple[int, int], Dict[int, int]] = {}
PERTURB_BASELINE_META: Dict[Tuple[int, int], Dict[str, str]] = {}


def _format_float(value: float) -> str:
    text = f"{value}".replace(".", "p")
    return text


def _ensure_near_data(data_path: str, weights_scaled: Dict[str, int]) -> None:
    global NEAR_DF, NEAR_WEEK_COLS, NEAR_SEASON_CACHE, NEAR_WEIGHTS_SCALED
    if NEAR_DF is not None:
        return
    NEAR_DF = pd.read_excel(data_path)
    NEAR_WEEK_COLS = get_week_columns(NEAR_DF)
    NEAR_SEASON_CACHE = {}
    NEAR_WEIGHTS_SCALED = dict(weights_scaled)


def _get_near_weeks(season: int) -> List[Dict]:
    if season not in NEAR_SEASON_CACHE:
        NEAR_SEASON_CACHE[season] = build_season_weeks(
            NEAR_DF, season, NEAR_WEEK_COLS
        )
    return NEAR_SEASON_CACHE[season]


def _init_near_worker(data_path: str, weights_scaled: Dict[str, int]) -> None:
    _ensure_near_data(data_path, weights_scaled)


def _write_csv(path: str, fieldnames: List[str], rows: Iterable[Dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _run_near_opt_task(task: Dict) -> str:
    season = task["season"]
    epsilon = task["epsilon"]
    bound = task["bound"]
    week_group = task["week_group"]
    output_path = task["output_path"]
    time_limit = task["time_limit"]

    weeks = _get_near_weeks(season)
    if not weeks:
        _write_csv(
            output_path,
            [
                "season",
                "week",
                "contestant_id",
                "celebrity_name",
                "ballroom_partner",
                "judge_rank",
                "fan_rank_min",
                "fan_rank_max",
                "combined_rank_min",
                "combined_rank_max",
                "actual_eliminated",
                "epsilon",
            ],
            [],
        )
        return output_path

    build = build_rank_model(weeks, NEAR_WEIGHTS_SCALED)
    model = build["model"]
    rF_vars = build["rF_vars"]
    week_map = build["week_map"]
    objective_expr = build["objective_expr"]

    if objective_expr is not None:
        model.Add(objective_expr <= int(bound))

    rows: List[Dict] = []
    for week_num in week_group:
        if week_num not in week_map:
            continue
        week = week_map[week_num]
        contestants = week["contestants"]
        for contestant_id in contestants:
            rF_var = rF_vars[(week_num, contestant_id)]

            model.Minimize(rF_var)
            solver = cp_model.CpSolver()
            solver.parameters.max_time_in_seconds = float(time_limit)
            solver.parameters.num_search_workers = 1
            status_min = solver.Solve(model)
            rF_min = (
                int(solver.Value(rF_var))
                if status_min in (cp_model.OPTIMAL, cp_model.FEASIBLE)
                else None
            )

            model.Minimize(-rF_var)
            solver = cp_model.CpSolver()
            solver.parameters.max_time_in_seconds = float(time_limit)
            solver.parameters.num_search_workers = 1
            status_max = solver.Solve(model)
            rF_max = (
                int(solver.Value(rF_var))
                if status_max in (cp_model.OPTIMAL, cp_model.FEASIBLE)
                else None
            )

            rJ = int(week["judge_rank"][contestant_id])
            combined_min = rJ + rF_min if rF_min is not None else None
            combined_max = rJ + rF_max if rF_max is not None else None
            actual_elim = int(contestant_id in week["eliminated_ids"])

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
                    "combined_rank_min": combined_min,
                    "combined_rank_max": combined_max,
                    "actual_eliminated": actual_elim,
                    "epsilon": epsilon,
                }
            )

    _write_csv(
        output_path,
        [
            "season",
            "week",
            "contestant_id",
            "celebrity_name",
            "ballroom_partner",
            "judge_rank",
            "fan_rank_min",
            "fan_rank_max",
            "combined_rank_min",
            "combined_rank_max",
            "actual_eliminated",
            "epsilon",
        ],
        rows,
    )
    return output_path


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


def _load_baseline_predictions(baseline_dir: str) -> Tuple[
    Dict[Tuple[int, int], Dict[int, int]],
    Dict[Tuple[int, int], Dict[str, str]],
]:
    path = os.path.join(baseline_dir, "weekly_predictions.csv")
    df = pd.read_csv(path)
    by_week: Dict[Tuple[int, int], Dict[int, int]] = {}
    meta: Dict[Tuple[int, int], Dict[str, str]] = {}

    for row in df.itertuples(index=False):
        key = (int(row.season), int(row.week))
        contestant_id = int(row.contestant_id)
        by_week.setdefault(key, {})[contestant_id] = int(row.combined_rank)
        meta[(int(row.season), contestant_id)] = {
            "celebrity_name": row.celebrity_name,
            "ballroom_partner": row.ballroom_partner,
        }

    return by_week, meta


def _ensure_perturb_data(
    data_path: str,
    baseline_dir: str,
    weights_scaled: Dict[str, int],
    weight_scale: int,
) -> None:
    global PERTURB_DF, PERTURB_WEEK_COLS, PERTURB_WEIGHTS_SCALED
    global PERTURB_WEIGHT_SCALE, PERTURB_BASELINE_BY_WEEK, PERTURB_BASELINE_META
    if PERTURB_DF is not None:
        return
    PERTURB_DF = pd.read_excel(data_path)
    PERTURB_WEEK_COLS = get_week_columns(PERTURB_DF)
    PERTURB_WEIGHTS_SCALED = dict(weights_scaled)
    PERTURB_WEIGHT_SCALE = int(weight_scale)
    (
        PERTURB_BASELINE_BY_WEEK,
        PERTURB_BASELINE_META,
    ) = _load_baseline_predictions(baseline_dir)


def _init_perturb_worker(
    data_path: str,
    baseline_dir: str,
    weights_scaled: Dict[str, int],
    weight_scale: int,
) -> None:
    _ensure_perturb_data(data_path, baseline_dir, weights_scaled, weight_scale)


def _build_season_weeks_with_noise(
    df: pd.DataFrame,
    season: int,
    week_cols: List[str],
    rng: np.random.Generator,
    noise_std: float,
    min_score: float,
) -> List[Dict]:
    season_df = df[df["season"] == season].copy()
    if season_df.empty:
        return []
    season_df = season_df.reset_index(drop=True)
    season_df["contestant_id"] = season_df.index.astype(int)
    season_df["elim_week"] = season_df["results"].apply(parse_elimination_week)

    weeks = []
    for idx, col in enumerate(week_cols, start=1):
        scores = season_df[col]
        active_mask = scores.notna() & (scores > 0)
        if active_mask.sum() == 0:
            continue

        base_scores = scores.loc[active_mask].astype(float).to_numpy()
        noise = rng.normal(0.0, noise_std, size=base_scores.shape)
        pert_scores = np.maximum(base_scores + noise, min_score)

        week_df = season_df.loc[
            active_mask,
            [
                "contestant_id",
                "celebrity_name",
                "ballroom_partner",
                "results",
                "elim_week",
            ],
        ].copy()
        week_df["judge_score"] = pert_scores
        week_df["judge_rank"] = (
            (-week_df["judge_score"]).rank(method="min").astype(int)
        )

        eliminated_ids = week_df.loc[
            week_df["elim_week"] == idx, "contestant_id"
        ].tolist()

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
            }
        )

    return weeks


def _predict_eliminated(week: Dict, rF: Dict) -> Tuple[set, Dict[int, int]]:
    week_num = week["week"]
    contestants = week["contestants"]
    scores = []
    for i in contestants:
        rJ = week["judge_rank"][i]
        rF_val = rF[(week_num, i)]
        scores.append((i, rJ + rF_val))
    scores.sort(key=lambda x: x[1], reverse=True)

    eliminated = set()
    if week["eliminated_ids"]:
        k = len(week["eliminated_ids"])
        threshold = scores[min(k, len(scores)) - 1][1]
        eliminated = {i for i, R in scores if R >= threshold}

    combined = {i: r for i, r in scores}
    return eliminated, combined


def _spearman_rho(x: List[int], y: List[int]) -> Optional[float]:
    if len(x) < 2:
        return None
    arr_x = np.asarray(x, dtype=float)
    arr_y = np.asarray(y, dtype=float)
    if np.std(arr_x) == 0 or np.std(arr_y) == 0:
        return None
    return float(np.corrcoef(arr_x, arr_y)[0, 1])


def _kendall_tau_a(x: List[int], y: List[int]) -> Optional[float]:
    n = len(x)
    if n < 2:
        return None
    concordant = 0
    discordant = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            prod = dx * dy
            if prod > 0:
                concordant += 1
            elif prod < 0:
                discordant += 1
    denom = n * (n - 1) / 2
    if denom == 0:
        return None
    return (concordant - discordant) / denom


def _run_perturb_batch(task: Dict) -> Tuple[str, str]:
    batch_id = task["batch_id"]
    n_samples = task["n_samples"]
    seasons = task["seasons"]
    noise_std = task["noise_std"]
    min_score = task["min_score"]
    seed = task["seed"]
    time_limit = task["time_limit"]
    output_dir = task["output_dir"]

    rng = np.random.default_rng(seed + batch_id)
    sample_rows: List[Dict] = []
    corr_rows: List[Dict] = []

    for iteration in range(n_samples):
        for season in seasons:
            weeks = _build_season_weeks_with_noise(
                PERTURB_DF,
                season,
                PERTURB_WEEK_COLS,
                rng,
                noise_std,
                min_score,
            )
            if not weeks:
                continue

            result = solve_season(
                weeks=weeks,
                weights_scaled=PERTURB_WEIGHTS_SCALED,
                weight_scale=PERTURB_WEIGHT_SCALE,
                time_limit=time_limit,
                num_workers=1,
            )
            if result["status"] not in ("OPTIMAL", "FEASIBLE"):
                continue

            rF = result["rF"]
            week_map = {w["week"]: w for w in weeks}
            for week_num, week in week_map.items():
                predicted, combined = _predict_eliminated(week, rF)
                baseline_key = (season, week_num)
                baseline = PERTURB_BASELINE_BY_WEEK.get(baseline_key, {})
                base_ranks = []
                pert_ranks = []

                for contestant_id in week["contestants"]:
                    rF_val = rF[(week_num, contestant_id)]
                    combined_rank = combined[contestant_id]
                    sample_rows.append(
                        {
                            "season": season,
                            "week": week_num,
                            "contestant_id": contestant_id,
                            "celebrity_name": week["names"][contestant_id],
                            "ballroom_partner": week["partners"][contestant_id],
                            "fan_rank": rF_val,
                            "combined_rank": combined_rank,
                            "predicted_eliminated": int(
                                contestant_id in predicted
                            ),
                            "iteration": iteration,
                        }
                    )

                    base_rank = baseline.get(contestant_id)
                    if base_rank is not None:
                        base_ranks.append(base_rank)
                        pert_ranks.append(combined_rank)

                if base_ranks and pert_ranks:
                    corr_rows.append(
                        {
                            "season": season,
                            "week": week_num,
                            "iteration": iteration,
                            "spearman": _spearman_rho(
                                base_ranks, pert_ranks
                            ),
                            "kendall": _kendall_tau_a(
                                base_ranks, pert_ranks
                            ),
                        }
                    )

    sample_path = os.path.join(
        output_dir, f"perturb_samples_batch_{batch_id}.csv"
    )
    corr_path = os.path.join(
        output_dir, f"perturb_rank_corr_batch_{batch_id}.csv"
    )

    if sample_rows:
        _write_csv(
            sample_path,
            [
                "season",
                "week",
                "contestant_id",
                "celebrity_name",
                "ballroom_partner",
                "fan_rank",
                "combined_rank",
                "predicted_eliminated",
                "iteration",
            ],
            sample_rows,
        )
    else:
        _write_csv(
            sample_path,
            [
                "season",
                "week",
                "contestant_id",
                "celebrity_name",
                "ballroom_partner",
                "fan_rank",
                "combined_rank",
                "predicted_eliminated",
                "iteration",
            ],
            [],
        )

    if corr_rows:
        _write_csv(
            corr_path,
            ["season", "week", "iteration", "spearman", "kendall"],
            corr_rows,
        )
    else:
        _write_csv(
            corr_path,
            ["season", "week", "iteration", "spearman", "kendall"],
            [],
        )

    return sample_path, corr_path


def _aggregate_near_opt(output_dir: str) -> Tuple[str, str]:
    parts_dir = os.path.join(output_dir, "parts")
    part_files = [
        os.path.join(parts_dir, name)
        for name in os.listdir(parts_dir)
        if name.endswith(".csv")
    ]
    if not part_files:
        return "", ""

    df = pd.concat(
        [pd.read_csv(path) for path in part_files], ignore_index=True
    )

    interval_path = os.path.join(output_dir, "near_opt_interval.csv")
    df.to_csv(interval_path, index=False)

    certainty_df = _compute_interval_certainty(df)
    certainty_path = os.path.join(output_dir, "near_opt_elim_certainty.csv")
    certainty_df.to_csv(certainty_path, index=False)

    return interval_path, certainty_path


def _aggregate_perturb(output_dir: str) -> Tuple[str, str, str]:
    part_files = [
        os.path.join(output_dir, name)
        for name in os.listdir(output_dir)
        if name.startswith("perturb_samples_batch_") and name.endswith(".csv")
    ]
    if not part_files:
        return "", "", ""

    df = pd.concat(
        [pd.read_csv(path) for path in part_files], ignore_index=True
    )
    group_keys = [
        "season",
        "week",
        "contestant_id",
        "celebrity_name",
        "ballroom_partner",
    ]

    group = df.groupby(group_keys)["fan_rank"]
    stats = group.agg(["mean", "std"]).rename(
        columns={"mean": "fan_rank_mean", "std": "fan_rank_std"}
    )
    p05 = group.quantile(0.05).rename("fan_rank_p05")
    p95 = group.quantile(0.95).rename("fan_rank_p95")
    stats = stats.join(p05).join(p95).reset_index()

    stats_path = os.path.join(output_dir, "perturb_rF_stats.csv")
    stats.to_csv(stats_path, index=False)

    elim_prob = (
        df.groupby(group_keys)["predicted_eliminated"]
        .mean()
        .rename("predicted_elim_prob")
        .reset_index()
    )
    elim_path = os.path.join(output_dir, "perturb_elim_prob.csv")
    elim_prob.to_csv(elim_path, index=False)

    corr_files = [
        os.path.join(output_dir, name)
        for name in os.listdir(output_dir)
        if name.startswith("perturb_rank_corr_batch_") and name.endswith(".csv")
    ]
    if corr_files:
        corr_df = pd.concat(
            [pd.read_csv(path) for path in corr_files], ignore_index=True
        )
        corr_group = corr_df.groupby(["season", "week"])
        corr_stats = corr_group.agg(
            spearman_mean=("spearman", "mean"),
            spearman_std=("spearman", "std"),
            spearman_p05=("spearman", lambda x: x.quantile(0.05)),
            spearman_p95=("spearman", lambda x: x.quantile(0.95)),
            kendall_mean=("kendall", "mean"),
            kendall_std=("kendall", "std"),
            kendall_p05=("kendall", lambda x: x.quantile(0.05)),
            kendall_p95=("kendall", lambda x: x.quantile(0.95)),
        ).reset_index()
        corr_path = os.path.join(output_dir, "perturb_rank_stability.csv")
        corr_stats.to_csv(corr_path, index=False)
    else:
        corr_path = ""

    return stats_path, elim_path, corr_path


def _make_week_groups(weeks: List[int], group_size: int) -> List[List[int]]:
    if group_size <= 0 or group_size >= len(weeks):
        return [weeks]
    return [
        weeks[i : i + group_size] for i in range(0, len(weeks), group_size)
    ]


def _load_opt_bounds(
    opt_info_path: str, epsilons: List[float], seasons: List[int]
) -> Dict[Tuple[int, float], int]:
    with open(opt_info_path, "r", encoding="utf-8") as f:
        info = json.load(f)
    season_info = info.get("season_info", {})

    bounds: Dict[Tuple[int, float], int] = {}
    for season in seasons:
        season_str = str(season)
        data = season_info.get(season_str)
        if not data:
            continue
        opt_scaled = data.get("objective_scaled")
        if opt_scaled is None:
            continue
        for eps in epsilons:
            bound = int(math.ceil(opt_scaled * (1.0 + eps)))
            bounds[(season, eps)] = bound
    return bounds


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        default="/home/hisheep/d/MCM/26/Data_4.xlsx",
        help="Path to Data_4.xlsx",
    )
    parser.add_argument(
        "--baseline-dir",
        default=(
            "/home/hisheep/d/MCM/26/task1-4/outputs/"
            "grid_a1p0_b0p05_g10p0"
        ),
        help="Directory with baseline weekly_predictions.csv and optimization_info.json",
    )
    parser.add_argument(
        "--output-root",
        default="/home/hisheep/d/MCM/26/task1-4/outputs_uncertainty",
        help="Root output directory for uncertainty analysis",
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
        help="Comma-separated epsilon list for near-opt bounds",
    )
    parser.add_argument(
        "--near-opt-time-limit",
        type=float,
        default=60.0,
        help="Time limit per near-opt solve (seconds)",
    )
    parser.add_argument(
        "--week-group-size",
        type=int,
        default=3,
        help="Group size for weeks in near-opt tasks (0 = all weeks)",
    )
    parser.add_argument(
        "--noise-std",
        type=float,
        default=0.7,
        help="Standard deviation for judge_score perturbation",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=500,
        help="Total perturbation samples",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=0,
        help="Samples per perturbation batch (0 = auto)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--processes",
        type=int,
        default=0,
        help="Number of worker processes (0 = auto)",
    )
    parser.add_argument(
        "--perturb-time-limit",
        type=float,
        default=60.0,
        help="Time limit per perturbation solve (seconds)",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=1e-6,
        help="Minimum judge score after perturbation",
    )
    parser.add_argument(
        "--near-opt-only",
        action="store_true",
        help="Run only near-opt interval analysis",
    )
    parser.add_argument(
        "--perturb-only",
        action="store_true",
        help="Run only perturbation analysis",
    )
    parser.add_argument(
        "--rerun",
        action="store_true",
        help="Recompute even if part files already exist",
    )

    args = parser.parse_args()

    epsilons = [
        float(part.strip())
        for part in args.epsilons.split(",")
        if part.strip()
    ]
    weight_info = scale_weights(args.alpha, args.beta, args.gamma)
    weights_scaled = weight_info["scaled"]
    weight_scale = weight_info["scale"]

    processes = args.processes
    if processes <= 0:
        cpu_total = os.cpu_count() or 1
        processes = min(cpu_total, 12)

    os.makedirs(args.output_root, exist_ok=True)

    do_near_opt = not args.perturb_only
    do_perturb = not args.near_opt_only

    if do_near_opt:
        opt_info_path = os.path.join(
            args.baseline_dir, "optimization_info.json"
        )
        bounds = _load_opt_bounds(opt_info_path, epsilons, args.seasons)

        df = pd.read_excel(args.data)
        week_cols = get_week_columns(df)
        season_week_numbers: Dict[int, List[int]] = {}
        for season in args.seasons:
            weeks = build_season_weeks(df, season, week_cols)
            season_week_numbers[season] = sorted(
                {week["week"] for week in weeks}
            )

        near_dir = os.path.join(args.output_root, "near_opt")
        parts_dir = os.path.join(near_dir, "parts")
        os.makedirs(parts_dir, exist_ok=True)

        tasks: List[Dict] = []
        for season in args.seasons:
            weeks = season_week_numbers.get(season, [])
            if not weeks:
                continue
            week_groups = _make_week_groups(
                weeks, args.week_group_size
            )
            for epsilon in epsilons:
                bound = bounds.get((season, epsilon))
                if bound is None:
                    continue
                for idx, week_group in enumerate(week_groups):
                    eps_label = _format_float(epsilon)
                    part_path = os.path.join(
                        parts_dir,
                        f"near_opt_s{season}_e{eps_label}_g{idx}.csv",
                    )
                    if os.path.exists(part_path) and not args.rerun:
                        continue
                    tasks.append(
                        {
                            "season": season,
                            "epsilon": epsilon,
                            "bound": bound,
                            "week_group": week_group,
                            "output_path": part_path,
                            "time_limit": args.near_opt_time_limit,
                        }
                    )

        if tasks:
            with ProcessPoolExecutor(
                max_workers=processes,
                initializer=_init_near_worker,
                initargs=(args.data, weights_scaled),
            ) as executor:
                for _ in executor.map(_run_near_opt_task, tasks):
                    pass

        _aggregate_near_opt(near_dir)

    if do_perturb:
        perturb_dir = os.path.join(args.output_root, "perturb")
        os.makedirs(perturb_dir, exist_ok=True)

        batch_size = args.batch_size
        if batch_size <= 0:
            batch_size = int(math.ceil(args.n_samples / processes))

        n_batches = int(math.ceil(args.n_samples / batch_size))
        tasks = []
        for batch_id in range(n_batches):
            start = batch_id * batch_size
            count = min(batch_size, args.n_samples - start)
            sample_path = os.path.join(
                perturb_dir, f"perturb_samples_batch_{batch_id}.csv"
            )
            corr_path = os.path.join(
                perturb_dir, f"perturb_rank_corr_batch_{batch_id}.csv"
            )
            if (
                not args.rerun
                and os.path.exists(sample_path)
                and os.path.exists(corr_path)
            ):
                continue
            tasks.append(
                {
                    "batch_id": batch_id,
                    "n_samples": count,
                    "seasons": args.seasons,
                    "noise_std": args.noise_std,
                    "min_score": args.min_score,
                    "seed": args.seed,
                    "time_limit": args.perturb_time_limit,
                    "output_dir": perturb_dir,
                }
            )

        if tasks:
            with ProcessPoolExecutor(
                max_workers=processes,
                initializer=_init_perturb_worker,
                initargs=(
                    args.data,
                    args.baseline_dir,
                    weights_scaled,
                    weight_scale,
                ),
            ) as executor:
                for _ in executor.map(_run_perturb_batch, tasks):
                    pass

        _aggregate_perturb(perturb_dir)


if __name__ == "__main__":
    main()
