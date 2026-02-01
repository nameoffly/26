from __future__ import annotations

import argparse
import os
from decimal import Decimal
from typing import Tuple

import pandas as pd


DEFAULT_INPUT = (
    "/home/hisheep/d/MCM/26/task1-4/final_outputs/baseline/"
    "grid_a1p0_b0p05_g10p0/weekly_predictions.csv"
)
DEFAULT_OUTPUT = (
    "/home/hisheep/d/MCM/26/task1-4/final_outputs/baseline/"
    "grid_a1p0_b0p05_g10p0/fan_vote_percent.csv"
)


def parse_float(value: str) -> float:
    return float(Decimal(value))


def validate_columns(df: pd.DataFrame) -> None:
    required = {"season", "week", "fan_rank"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")


def compute_group(
    group: pd.DataFrame, alpha: float, beta: float, percent_sum: float
) -> pd.DataFrame:
    fan_rank = group["fan_rank"].astype(float)
    if (fan_rank <= 0).any():
        raise ValueError("fan_rank must be positive for all rows.")

    zipf_score = 1.0 / ((fan_rank + beta) ** alpha)
    total = zipf_score.sum()
    if total <= 0:
        raise ValueError("Zipf score sum is non-positive for a group.")

    percent = zipf_score / total * percent_sum
    group = group.copy()
    group["zipf_score"] = zipf_score
    group["fan_vote_percent"] = percent
    return group


def run(
    input_path: str, output_path: str, alpha: float, beta: float, percent_sum: float
) -> Tuple[int, int]:
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input not found: {input_path}")

    df = pd.read_csv(input_path)
    validate_columns(df)

    grouped = df.groupby(["season", "week"], group_keys=False)
    result = grouped.apply(
        lambda g: compute_group(g, alpha=alpha, beta=beta, percent_sum=percent_sum)
    )

    result.to_csv(output_path, index=False)
    return len(result), result.groupby(["season", "week"]).ngroups


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--alpha", default="0.9")
    parser.add_argument("--beta", default="1.0")
    parser.add_argument("--percent-sum", default="1")

    args = parser.parse_args()

    alpha = parse_float(args.alpha)
    beta = parse_float(args.beta)
    percent_sum = parse_float(args.percent_sum)

    run(args.input, args.output, alpha, beta, percent_sum)


if __name__ == "__main__":
    main()
