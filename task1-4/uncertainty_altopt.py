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
from ortools.graph.python import linear_sum_assignment


DEFAULT_SEASONS = [1, 2, 28, 29, 30, 31, 32, 33, 34]
DEFAULT_BASELINE_DIR = (
    "/home/hisheep/d/MCM/26/task1-4/outputs/grid_a1p0_b0p05_g10p0"
)
DEFAULT_OUTPUT_ROOT = "/home/hisheep/d/MCM/26/task1-4/outputs_uncertainty_altopt"

WORK_DF: Optional[pd.DataFrame] = None
WORK_WEEK_COLS: Optional[List[str]] = None
BASELINE_COMBINED: Dict[Tuple[int, int], Dict[int, int]] = {}
BASELINE_FAN_RANK: Dict[Tuple[int, int], Dict[int, int]] = {}
BASELINE_META: Dict[Tuple[int, int], Dict[str, str]] = {}

ALT_ALPHA: float = 1.0
ALT_BETA: float = 0.05
ALT_GAMMA: float = 10.0
ALT_SWEEPS: int = 5
ALT_INIT: str = "baseline"
ALT_COST_SCALE: int = 1000
ALT_ELIM_WEIGHT: Optional[float] = None


def _write_csv(path: str, fieldnames: List[str], rows: Iterable[Dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _get_week_columns(df: pd.DataFrame) -> List[str]:
    week_cols = []
    for col in df.columns:
        name = str(col)
        if name.startswith("week") and name.endswith("_judge_score"):
            idx = int(name.split("_")[0].replace("week", ""))
            week_cols.append((idx, col))
    week_cols.sort(key=lambda x: x[0])
    return [col for _, col in week_cols]


def _parse_elimination_week(value: object) -> Optional[int]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value)
    if "Eliminated Week" not in text:
        return None
    parts = text.split()
    if len(parts) >= 3 and parts[-1].isdigit():
        return int(parts[-1])
    return None


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
    season_df["elim_week"] = season_df["results"].apply(_parse_elimination_week)

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


def _load_baseline_predictions(
    baseline_dir: str,
) -> Tuple[
    Dict[Tuple[int, int], Dict[int, int]],
    Dict[Tuple[int, int], Dict[int, int]],
    Dict[Tuple[int, int], Dict[str, str]],
]:
    path = os.path.join(baseline_dir, "weekly_predictions.csv")
    df = pd.read_csv(path)
    combined: Dict[Tuple[int, int], Dict[int, int]] = {}
    fan_rank: Dict[Tuple[int, int], Dict[int, int]] = {}
    meta: Dict[Tuple[int, int], Dict[str, str]] = {}

    for row in df.itertuples(index=False):
        key = (int(row.season), int(row.week))
        contestant_id = int(row.contestant_id)
        combined.setdefault(key, {})[contestant_id] = int(row.combined_rank)
        fan_rank.setdefault(key, {})[contestant_id] = int(row.fan_rank)
        meta[(int(row.season), contestant_id)] = {
            "celebrity_name": row.celebrity_name,
            "ballroom_partner": row.ballroom_partner,
        }

    return combined, fan_rank, meta


def _ensure_worker_data(
    data_path: str,
    baseline_dir: str,
) -> None:
    global WORK_DF, WORK_WEEK_COLS, BASELINE_COMBINED, BASELINE_FAN_RANK
    global BASELINE_META
    if WORK_DF is not None:
        return
    WORK_DF = pd.read_excel(data_path)
    WORK_WEEK_COLS = _get_week_columns(WORK_DF)
    (
        BASELINE_COMBINED,
        BASELINE_FAN_RANK,
        BASELINE_META,
    ) = _load_baseline_predictions(baseline_dir)


def _init_worker(
    data_path: str,
    baseline_dir: str,
    alpha: float,
    beta: float,
    gamma: float,
    sweeps: int,
    init_method: str,
    cost_scale: int,
    elim_weight: Optional[float],
) -> None:
    _ensure_worker_data(data_path, baseline_dir)
    global ALT_ALPHA, ALT_BETA, ALT_GAMMA, ALT_SWEEPS, ALT_INIT
    global ALT_COST_SCALE, ALT_ELIM_WEIGHT
    ALT_ALPHA = alpha
    ALT_BETA = beta
    ALT_GAMMA = gamma
    ALT_SWEEPS = sweeps
    ALT_INIT = init_method
    ALT_COST_SCALE = cost_scale
    ALT_ELIM_WEIGHT = elim_weight


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
    init_method: str,
    cost_scale: int,
    elim_weight: Optional[float],
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
            if init_method == "baseline":
                init_rank = base_week.get(contestant_id)
                if init_rank is None:
                    init_rank = int(week["judge_rank"][contestant_id])
            else:
                init_rank = int(week["judge_rank"][contestant_id])
            rF[(week_num, contestant_id)] = int(init_rank)

    elim_weight_val = (
        float(elim_weight) if elim_weight is not None else float(gamma)
    )

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
                        c += elim_weight_val * (n_w - k)
                    cost[row, col] = c * cost_scale
            ranks = _solve_week_assignment(cost)
            for row, contestant_id in enumerate(contestants):
                rF[(week_num, contestant_id)] = ranks[row]

    return rF


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
    rank_x = pd.Series(x).rank(method="average").to_numpy(dtype=float)
    rank_y = pd.Series(y).rank(method="average").to_numpy(dtype=float)
    if np.std(rank_x) == 0 or np.std(rank_y) == 0:
        return None
    return float(np.corrcoef(rank_x, rank_y)[0, 1])


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


def _run_perturb_batch(task: Dict) -> Tuple[List[Dict], List[Dict]]:
    batch_id = task["batch_id"]
    n_samples = task["n_samples"]
    seasons = task["seasons"]
    noise_std = task["noise_std"]
    min_score = task["min_score"]
    seed = task["seed"]
    output_dir = task["output_dir"]

    rng = np.random.default_rng(seed + batch_id)
    sample_rows: List[Dict] = []
    corr_rows: List[Dict] = []

    for iteration in range(n_samples):
        for season in seasons:
            weeks = _build_season_weeks_with_noise(
                WORK_DF,
                season,
                WORK_WEEK_COLS,
                rng,
                noise_std,
                min_score,
            )
            if not weeks:
                continue

            rF = _altopt_solve(
                weeks,
                alpha=ALT_ALPHA,
                beta=ALT_BETA,
                gamma=ALT_GAMMA,
                sweeps=ALT_SWEEPS,
                init_method=ALT_INIT,
                cost_scale=ALT_COST_SCALE,
                elim_weight=ALT_ELIM_WEIGHT,
                baseline_fan_rank=BASELINE_FAN_RANK,
            )

            week_map = {w["week"]: w for w in weeks}
            for week_num, week in week_map.items():
                predicted, combined = _predict_eliminated(week, rF)
                baseline_key = (season, week_num)
                baseline = BASELINE_COMBINED.get(baseline_key, {})
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

    return sample_rows, corr_rows


def _aggregate_perturb_rows(
    sample_rows: List[Dict],
    corr_rows: List[Dict],
    output_dir: str,
) -> Tuple[str, str, str]:
    if not sample_rows:
        return "", "", ""

    df = pd.DataFrame(sample_rows)
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

    if corr_rows:
        corr_df = pd.DataFrame(corr_rows)
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        default="/home/hisheep/d/MCM/26/Data_4.xlsx",
        help="Path to Data_4.xlsx",
    )
    parser.add_argument(
        "--baseline-dir",
        default=DEFAULT_BASELINE_DIR,
        help="Directory with baseline weekly_predictions.csv",
    )
    parser.add_argument(
        "--output-root",
        default=DEFAULT_OUTPUT_ROOT,
        help="Root output directory for altopt uncertainty",
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
        "--sweeps",
        type=int,
        default=5,
        help="Number of alternating optimization sweeps per sample",
    )
    parser.add_argument(
        "--init",
        default="baseline",
        choices=["baseline", "judge"],
        help="Initialization for fan_rank",
    )
    parser.add_argument(
        "--elim-weight",
        type=float,
        default=None,
        help="Elimination penalty weight (default: use gamma)",
    )
    parser.add_argument(
        "--cost-scale",
        type=int,
        default=1000,
        help="Scale factor for assignment costs (int)",
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
        "--min-score",
        type=float,
        default=1e-6,
        help="Minimum judge score after perturbation",
    )
    parser.add_argument(
        "--rerun",
        action="store_true",
        help="Recompute even if part files already exist",
    )

    args = parser.parse_args()

    alpha = float(args.alpha)
    beta = float(args.beta)
    gamma = float(args.gamma)

    processes = args.processes
    if processes <= 0:
        cpu_total = os.cpu_count() or 1
        processes = min(cpu_total, 12)

    output_dir = os.path.join(args.output_root, "perturb")
    os.makedirs(output_dir, exist_ok=True)

    stats_path = os.path.join(output_dir, "perturb_rF_stats.csv")
    elim_path = os.path.join(output_dir, "perturb_elim_prob.csv")
    corr_path = os.path.join(output_dir, "perturb_rank_stability.csv")
    if (
        not args.rerun
        and os.path.exists(stats_path)
        and os.path.exists(elim_path)
        and os.path.exists(corr_path)
    ):
        return

    batch_size = args.batch_size
    if batch_size <= 0:
        batch_size = int(math.ceil(args.n_samples / processes))

    n_batches = int(math.ceil(args.n_samples / batch_size))
    tasks = []
    for batch_id in range(n_batches):
        start = batch_id * batch_size
        count = min(batch_size, args.n_samples - start)
        tasks.append(
            {
                "batch_id": batch_id,
                "n_samples": count,
                "seasons": args.seasons,
                "noise_std": args.noise_std,
                "min_score": args.min_score,
                "seed": args.seed,
                "output_dir": output_dir,
            }
        )

    all_samples: List[Dict] = []
    all_corrs: List[Dict] = []
    if tasks:
        with ProcessPoolExecutor(
            max_workers=processes,
            initializer=_init_worker,
            initargs=(
                args.data,
                args.baseline_dir,
                alpha,
                beta,
                gamma,
                args.sweeps,
                args.init,
                args.cost_scale,
                args.elim_weight,
            ),
        ) as executor:
            for sample_rows, corr_rows in executor.map(
                _run_perturb_batch, tasks
            ):
                all_samples.extend(sample_rows)
                all_corrs.extend(corr_rows)

    _aggregate_perturb_rows(all_samples, all_corrs, output_dir)


if __name__ == "__main__":
    main()
