import argparse
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize


ELIM_PATTERN = re.compile(r"Eliminated Week (\d+)")


@dataclass
class DataBundle:
    long_df: pd.DataFrame
    feature_cols: List[str]
    feature_matrix: np.ndarray
    numeric_means: pd.Series
    numeric_stds: pd.Series
    seasons: List[int]


def get_week_columns(df: pd.DataFrame) -> Tuple[List[int], List[str]]:
    week_cols = [c for c in df.columns if c.startswith("week") and c.endswith("_judge_score")]
    week_nums = [int(c.replace("week", "").replace("_judge_score", "")) for c in week_cols]
    week_cols = [c for _, c in sorted(zip(week_nums, week_cols))]
    week_nums = sorted(week_nums)
    return week_nums, week_cols


def parse_elimination_week(results: str) -> int | None:
    if not isinstance(results, str):
        return None
    match = ELIM_PATTERN.search(results)
    if match:
        return int(match.group(1))
    return None


def reduce_categories(series: pd.Series, min_count: int) -> pd.Series:
    series = series.fillna("Unknown").astype(str)
    counts = series.value_counts()
    keep = counts[counts >= min_count].index
    return series.where(series.isin(keep), "Other")


def build_long_df(df: pd.DataFrame, season_min: int, season_max: int) -> pd.DataFrame:
    df = df[(df["season"] >= season_min) & (df["season"] <= season_max)].copy()
    week_nums, week_cols = get_week_columns(df)

    df["elim_week"] = df["results"].map(parse_elimination_week)
    df["withdrew"] = df["results"].eq("Withdrew")

    records: List[Dict] = []
    for _, row in df.iterrows():
        season = int(row["season"])
        for week, col in zip(week_nums, week_cols):
            score = row[col]
            if pd.isna(score) or score <= 0:
                continue
            records.append(
                {
                    "season": season,
                    "week": week,
                    "celebrity_name": row["celebrity_name"],
                    "judge_score": float(score),
                    "age": row["celebrity_age_during_season"],
                    "industry": row["celebrity_industry"],
                    "country": row["celebrity_homecountry/region"],
                    "results": row["results"],
                    "withdrew": bool(row["withdrew"]),
                    "elim_week": row["elim_week"],
                }
            )

    long_df = pd.DataFrame(records)
    if long_df.empty:
        raise ValueError("No valid weekly samples after filtering. Check input data.")

    # Judge percent per week
    long_df["judge_percent"] = long_df.groupby(["season", "week"])["judge_score"].transform(
        lambda s: s / s.sum()
    )

    # Elimination flags (withdrawals are not treated as eliminations)
    long_df["eliminated"] = long_df["elim_week"].eq(long_df["week"]) & (~long_df["withdrew"])

    return long_df


def prepare_features(
    long_df: pd.DataFrame,
    min_count: int,
    season_interactions: bool,
    include_judge_score: bool,
) -> DataBundle:
    long_df = long_df.copy()

    long_df["industry"] = reduce_categories(long_df["industry"], min_count)
    long_df["country"] = reduce_categories(long_df["country"], min_count)

    numeric = long_df[["judge_score", "judge_percent", "age"]].copy()
    numeric["age"] = numeric["age"].fillna(numeric["age"].mean())
    means = numeric.mean()
    stds = numeric.std(ddof=0).replace(0, 1)
    numeric_std = (numeric - means) / stds
    numeric_std = numeric_std.rename(
        columns={
            "judge_score": "judge_score_std",
            "judge_percent": "judge_percent_std",
            "age": "age_std",
        }
    )
    if not include_judge_score:
        numeric_std = numeric_std.drop(columns=["judge_score_std"])

    industry_dummies = pd.get_dummies(long_df["industry"], prefix="industry")
    country_dummies = pd.get_dummies(long_df["country"], prefix="country")

    base_features = pd.concat([numeric_std, industry_dummies, country_dummies], axis=1)
    base_cols = base_features.columns.tolist()
    base_matrix = base_features.to_numpy(dtype=float)

    seasons = sorted(long_df["season"].unique().tolist())
    if season_interactions:
        season_array = long_df["season"].to_numpy()
        season_cols: List[str] = []
        season_matrices = []
        for season in seasons:
            mask = (season_array == season).astype(float)[:, None]
            season_matrices.append(base_matrix * mask)
            season_cols.extend([f"season_{season}__{col}" for col in base_cols])

        if season_matrices:
            season_matrix = np.concatenate(season_matrices, axis=1)
            feature_matrix = np.concatenate([base_matrix, season_matrix], axis=1)
            feature_cols = base_cols + season_cols
        else:
            feature_matrix = base_matrix
            feature_cols = base_cols
    else:
        feature_matrix = base_matrix
        feature_cols = base_cols

    return DataBundle(
        long_df=long_df,
        feature_cols=feature_cols,
        feature_matrix=feature_matrix,
        numeric_means=means,
        numeric_stds=stds,
        seasons=seasons,
    )


def softmax_stable(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits)
    exps = np.exp(shifted)
    return exps / np.sum(exps)


def pairwise_ranking_loss(
    elim_scores: np.ndarray, survive_scores: np.ndarray, margin: float
) -> float:
    diffs = elim_scores[:, None] - survive_scores[None, :]
    loss = np.maximum(0.0, margin + diffs)
    return float(loss.sum())


def build_groups(long_df: pd.DataFrame) -> List[Dict]:
    groups = []
    for (season, week), group in long_df.groupby(["season", "week"]):
        group = group.copy()
        elim_count = int(group["eliminated"].sum())
        groups.append(
            {
                "season": int(season),
                "week": int(week),
                "indices": group.index.to_numpy(),
                "elim_count": elim_count,
            }
        )
    return groups


def objective_factory(
    feature_matrix: np.ndarray,
    judge_percent: np.ndarray,
    eliminated: np.ndarray,
    groups: List[Dict],
    margin: float,
    l2: float,
):
    def objective(params: np.ndarray) -> float:
        num_features = feature_matrix.shape[1]
        weights = params[:num_features]
        total_penalty = 0.0

        for group in groups:
            idx = group["indices"]
            elim_count = group["elim_count"]
            if elim_count == 0 or elim_count >= len(idx):
                continue
            logits = feature_matrix[idx] @ weights
            fan_percent = softmax_stable(logits)
            total_score = judge_percent[idx] + fan_percent
            elim_mask = eliminated[idx]

            elim_scores = total_score[elim_mask]
            survive_scores = total_score[~elim_mask]
            total_penalty += pairwise_ranking_loss(elim_scores, survive_scores, margin)

        if l2 > 0:
            total_penalty += l2 * np.sum(weights**2)
        return float(total_penalty)

    return objective


def optimize_weights(
    data: DataBundle,
    margin: float,
    l2: float,
    seed: int,
    maxiter: int,
) -> Tuple[np.ndarray, Dict]:
    long_df = data.long_df
    feature_matrix = data.feature_matrix
    judge_percent = long_df["judge_percent"].to_numpy()
    eliminated = long_df["eliminated"].to_numpy()

    groups = build_groups(long_df)
    num_features = feature_matrix.shape[1]

    rng = np.random.default_rng(seed)
    init_weights = rng.normal(0, 0.01, size=num_features)
    init_params = init_weights

    objective = objective_factory(feature_matrix, judge_percent, eliminated, groups, margin, l2)
    result = minimize(objective, init_params, method="L-BFGS-B", options={"maxiter": maxiter})
    best = result
    best_method = "L-BFGS-B"

    if not result.success:
        alt = minimize(objective, init_params, method="SLSQP", options={"maxiter": maxiter})
        if alt.fun < best.fun or (not best.success and alt.success):
            best = alt
            best_method = "SLSQP"

    return best.x[:num_features], {
        "success": bool(best.success),
        "status": int(best.status),
        "message": str(best.message),
        "fun": float(best.fun),
        "nit": int(best.nit),
        "method": best_method,
    }


def evaluate_predictions(
    data: DataBundle,
    weights: np.ndarray,
    margin: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    long_df = data.long_df.copy()

    fan_percent = np.zeros(len(long_df))
    total_score = np.zeros(len(long_df))
    predicted_elim = np.zeros(len(long_df), dtype=bool)

    weekly_rows = []

    for (season, week), group in long_df.groupby(["season", "week"]):
        idx = group.index.to_numpy()
        logits = data.feature_matrix[idx] @ weights
        fan = softmax_stable(logits)
        total = group["judge_percent"].to_numpy() + fan

        fan_percent[idx] = fan
        total_score[idx] = total

        elim_count = int(group["eliminated"].sum())
        if elim_count > 0 and elim_count < len(group):
            order = np.argsort(total)
            bottom_idx = idx[order[:elim_count]]
            predicted_elim[bottom_idx] = True

            elim_mask = group["eliminated"].to_numpy()
            elim_scores = total[elim_mask]
            survive_scores = total[~elim_mask]
            penalty = pairwise_ranking_loss(elim_scores, survive_scores, margin)
            correct = bool(set(group.index[elim_mask]).issubset(set(bottom_idx)))
        else:
            penalty = 0.0
            correct = False

        weekly_rows.append(
            {
                "season": int(season),
                "week": int(week),
                "elim_count": elim_count,
                "week_correct": int(correct) if elim_count > 0 else 0,
                "penalty": penalty,
            }
        )

    long_df["fan_percent"] = fan_percent
    long_df["total_score"] = total_score
    long_df["predicted_elim"] = predicted_elim

    weekly_summary = pd.DataFrame(weekly_rows)
    weekly_summary = weekly_summary.sort_values(["season", "week"])

    season_summary = (
        weekly_summary[weekly_summary["elim_count"] > 0]
        .groupby("season")
        .agg(weeks_total=("week_correct", "count"), weeks_correct=("week_correct", "sum"), penalty=("penalty", "sum"))
        .reset_index()
    )
    season_summary["consistency"] = season_summary["weeks_correct"] / season_summary["weeks_total"]

    overall = pd.DataFrame(
        {
            "season": ["Overall"],
            "weeks_total": [int(season_summary["weeks_total"].sum())],
            "weeks_correct": [int(season_summary["weeks_correct"].sum())],
            "penalty": [float(season_summary["penalty"].sum())],
        }
    )
    overall["consistency"] = overall["weeks_correct"] / overall["weeks_total"]

    consistency_summary = pd.concat([season_summary, overall], ignore_index=True)

    return long_df, weekly_summary, consistency_summary


def write_outputs(
    output_dir: str,
    data: DataBundle,
    weights: np.ndarray,
    optimization_info: Dict,
    long_df: pd.DataFrame,
    weekly_summary: pd.DataFrame,
    consistency_summary: pd.DataFrame,
):
    output_dir = output_dir.rstrip("/")
    weights_rows = []
    for name, weight in zip(data.feature_cols, weights):
        weights_rows.append(
            {
                "name": name,
                "weight": weight,
                "feature_mean": data.numeric_means.get(name.replace("_std", ""), np.nan),
                "feature_std": data.numeric_stds.get(name.replace("_std", ""), np.nan),
                "type": "feature",
            }
        )
    weights_df = pd.DataFrame(weights_rows)
    weights_df.to_csv(f"{output_dir}/weights_summary.csv", index=False)

    cols = [
        "season",
        "week",
        "celebrity_name",
        "judge_score",
        "judge_percent",
        "fan_percent",
        "total_score",
        "eliminated",
        "predicted_elim",
        "age",
        "industry",
        "country",
        "results",
        "withdrew",
    ]
    long_df[cols].to_csv(f"{output_dir}/weekly_predictions.csv", index=False)

    weekly_summary.to_csv(f"{output_dir}/weekly_penalty.csv", index=False)
    consistency_summary.to_csv(f"{output_dir}/consistency_summary.csv", index=False)

    opt_path = f"{output_dir}/optimization_info.json"
    pd.Series(optimization_info).to_json(opt_path, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Percent-method fan vote estimation (seasons 3-27a)")
    parser.add_argument("--data-path", default="/home/hisheep/d/MCM/26/Data_4.xlsx", help="Path to Data_4.xlsx")
    parser.add_argument("--output-dir", default="/home/hisheep/d/MCM/26/task1-3/outputs", help="Output directory")
    parser.add_argument("--season-min", type=int, default=3, help="Minimum season (inclusive)")
    parser.add_argument("--season-max", type=int, default=27, help="Maximum season (inclusive)")
    parser.add_argument("--min-category-count", type=int, default=3, help="Min count to keep category (else Other)")
    parser.add_argument("--no-season-interactions", action="store_true", help="Disable season-feature interactions")
    parser.add_argument("--no-judge-score", action="store_true", help="Disable judge_score_std feature")
    parser.add_argument("--margin", type=float, default=0.01, help="Penalty margin")
    parser.add_argument("--l2", type=float, default=0.0, help="L2 regularization strength")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--maxiter", type=int, default=500, help="Max iterations for optimizer")
    args = parser.parse_args()

    raw_df = pd.read_excel(args.data_path)
    long_df = build_long_df(raw_df, args.season_min, args.season_max)
    data = prepare_features(
        long_df,
        args.min_category_count,
        not args.no_season_interactions,
        not args.no_judge_score,
    )

    weights, optimization_info = optimize_weights(
        data, margin=args.margin, l2=args.l2, seed=args.seed, maxiter=args.maxiter
    )
    pred_df, weekly_summary, consistency_summary = evaluate_predictions(
        data, weights, margin=args.margin
    )

    write_outputs(
        args.output_dir,
        data,
        weights,
        optimization_info,
        pred_df,
        weekly_summary,
        consistency_summary,
    )

    print("Optimization:", optimization_info)
    print("Outputs written to:", args.output_dir)


if __name__ == "__main__":
    main()
