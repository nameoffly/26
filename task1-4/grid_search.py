from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from decimal import Decimal
from typing import Dict, List, Optional

import pandas as pd


DEFAULT_ALPHA = [1.0]
DEFAULT_BETA = [0.05, 0.1, 0.2]
DEFAULT_GAMMA = [10.0, 50.0, 100.0, 200.0]


def parse_float_list(value: Optional[str], default: List[float]) -> List[float]:
    if value is None:
        return list(default)
    parts = [v.strip() for v in value.split(",") if v.strip()]
    return [float(Decimal(v)) for v in parts]


def format_val(value: float) -> str:
    text = str(value)
    return text.replace(".", "p")


def build_dir_name(alpha: float, beta: float, gamma: float) -> str:
    return (
        f"grid_a{format_val(alpha)}_b{format_val(beta)}_g{format_val(gamma)}"
    )


def load_objective(path: str) -> Dict:
    if not os.path.exists(path):
        return {
            "objective_sum": None,
            "objective_avg": None,
            "valid_seasons": 0,
            "season_status": "",
        }

    with open(path, "r", encoding="utf-8") as f:
        info = json.load(f)

    season_info = info.get("season_info", {})
    objectives = []
    status_pairs = []
    for season_key in sorted(season_info.keys(), key=lambda x: int(x)):
        data = season_info[season_key]
        status = data.get("status")
        status_pairs.append(f"{season_key}:{status}")
        obj = data.get("objective")
        if status in ("OPTIMAL", "FEASIBLE") and obj is not None:
            objectives.append(float(obj))

    objective_sum = sum(objectives) if objectives else None
    objective_avg = (
        objective_sum / len(objectives) if objectives else None
    )

    return {
        "objective_sum": objective_sum,
        "objective_avg": objective_avg,
        "valid_seasons": len(objectives),
        "season_status": ";".join(status_pairs),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root",
        default="/home/hisheep/d/MCM/26/task1-4/outputs",
        help="Root output directory for grid runs",
    )
    parser.add_argument(
        "--summary-file",
        default="grid_search_summary.csv",
        help="Summary CSV file name",
    )
    parser.add_argument(
        "--alphas",
        default=None,
        help="Comma-separated alpha list, e.g. 1,2",
    )
    parser.add_argument(
        "--betas",
        default=None,
        help="Comma-separated beta list, e.g. 0.05,0.1,0.2",
    )
    parser.add_argument(
        "--gammas",
        default=None,
        help="Comma-separated gamma list, e.g. 10,50,100",
    )
    parser.add_argument(
        "--seasons",
        nargs="*",
        type=int,
        default=None,
        help="Seasons to solve (optional)",
    )
    parser.add_argument("--time-limit", type=float, default=120.0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--rerun",
        action="store_true",
        help="Rerun even if output exists",
    )

    args = parser.parse_args()

    alphas = parse_float_list(args.alphas, DEFAULT_ALPHA)
    betas = parse_float_list(args.betas, DEFAULT_BETA)
    gammas = parse_float_list(args.gammas, DEFAULT_GAMMA)

    solve_rank_path = os.path.join(
        os.path.dirname(__file__), "solve_rank.py"
    )

    os.makedirs(args.output_root, exist_ok=True)
    rows = []

    for alpha in alphas:
        for beta in betas:
            for gamma in gammas:
                dir_name = build_dir_name(alpha, beta, gamma)
                output_dir = os.path.join(args.output_root, dir_name)
                os.makedirs(output_dir, exist_ok=True)

                info_path = os.path.join(output_dir, "optimization_info.json")
                should_run = args.rerun or not os.path.exists(info_path)

                if should_run:
                    cmd = [
                        sys.executable,
                        solve_rank_path,
                        "--alpha",
                        str(alpha),
                        "--beta",
                        str(beta),
                        "--gamma",
                        str(gamma),
                        "--output-dir",
                        output_dir,
                        "--time-limit",
                        str(args.time_limit),
                        "--workers",
                        str(args.workers),
                    ]
                    if args.seasons:
                        cmd.append("--seasons")
                        cmd.extend(str(s) for s in args.seasons)

                    subprocess.run(cmd, check=True)

                stats = load_objective(info_path)
                rows.append(
                    {
                        "alpha": alpha,
                        "beta": beta,
                        "gamma": gamma,
                        "objective_sum": stats["objective_sum"],
                        "objective_avg": stats["objective_avg"],
                        "valid_seasons": stats["valid_seasons"],
                        "season_status": stats["season_status"],
                        "output_dir": output_dir,
                    }
                )

    df = pd.DataFrame(rows)
    df_sorted = df.sort_values(
        by=["objective_sum"], ascending=True, na_position="last"
    )
    summary_path = os.path.join(args.output_root, args.summary_file)
    df_sorted.to_csv(summary_path, index=False)


if __name__ == "__main__":
    main()
