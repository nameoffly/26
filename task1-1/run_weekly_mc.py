import argparse
from pathlib import Path
import numpy as np
import pandas as pd

import constraints
import data_processing
import sampler
import replay_elimination


def summarize_samples(samples, judge_pct, eliminated_idx):
    summary = {}
    if samples is None or len(samples) == 0:
        summary["margin_mean"] = None
        summary["margin_p05"] = None
        summary["margin_p50"] = None
        return summary

    if not eliminated_idx:
        summary["margin_mean"] = None
        summary["margin_p05"] = None
        summary["margin_p50"] = None
        return summary

    n = samples.shape[1]
    elim_mask = np.zeros(n, dtype=bool)
    elim_mask[eliminated_idx] = True
    non_elim_mask = ~elim_mask

    combined = samples + judge_pct[None, :]
    max_elim = combined[:, elim_mask].max(axis=1)
    min_non = combined[:, non_elim_mask].min(axis=1)
    margin = min_non - max_elim

    summary["margin_mean"] = float(np.mean(margin))
    summary["margin_p05"] = float(np.percentile(margin, 5))
    summary["margin_p50"] = float(np.percentile(margin, 50))
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="/home/hisheep/d/MCM/26/Data_4.xlsx",
        help="输入Excel路径",
    )
    parser.add_argument("--season-start", type=int, default=3)
    parser.add_argument("--season-end", type=int, default=27)
    parser.add_argument("--n-samples", type=int, default=2000)
    parser.add_argument("--burn-in", type=int, default=500)
    parser.add_argument("--thin", type=int, default=1)
    parser.add_argument("--epsilon", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--lp-ensemble",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="使用LP顶点混合生成内部起点",
    )
    parser.add_argument("--lp-m", type=int, default=30)
    parser.add_argument("--lp-seed", type=int, default=None)
    parser.add_argument(
        "--output-dir",
        default="/home/hisheep/d/MCM/26/task1-1/outputs",
    )
    parser.add_argument(
        "--replay-check",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="根据 fan_mean + judge_pct 反代淘汰并写入一致性结果",
    )
    args = parser.parse_args()

    df = pd.read_excel(args.input)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    week_summaries = []
    contestant_summaries = []
    rng = np.random.default_rng(args.seed)

    for season in range(args.season_start, args.season_end + 1):
        season_df = df[df["season"] == season].copy()
        if season_df.empty:
            continue

        last_week = data_processing.compute_last_week(season_df)
        if last_week == 0:
            continue

        for week in range(1, last_week + 1):
            names, judge_pct, eliminated_names, eliminated_idx = data_processing.week_slice(
                season_df, week=week, last_week=last_week
            )
            n_active = len(names)
            if n_active == 0:
                continue

            cons = constraints.build_constraints(
                judge_pct, eliminated_idx, epsilon=args.epsilon
            )
            x0 = None
            if args.lp_ensemble:
                lp_seed = args.lp_seed
                if lp_seed is None:
                    lp_seed = int(rng.integers(0, 1_000_000_000))
                x0 = constraints.find_interior_point_lp_ensemble(
                    cons["A_ub"],
                    cons["b_ub"],
                    cons["A_eq"],
                    cons["b_eq"],
                    m=args.lp_m,
                    seed=lp_seed,
                )
            if x0 is None:
                x0 = constraints.find_feasible_point(
                    cons["A_ub"], cons["b_ub"], cons["A_eq"], cons["b_eq"]
                )

            feasible = x0 is not None
            samples = None
            if feasible:
                samples = sampler.hit_and_run(
                    x0,
                    cons["A_ub"],
                    cons["b_ub"],
                    cons["A_eq"],
                    cons["b_eq"],
                    n_samples=args.n_samples,
                    burn_in=args.burn_in,
                    thin=args.thin,
                    seed=int(rng.integers(0, 1_000_000_000)),
                )

            summary = summarize_samples(samples, judge_pct, eliminated_idx)
            week_summaries.append(
                {
                    "season": season,
                    "week": week,
                    "n_active": n_active,
                    "n_eliminated": len(eliminated_idx),
                    "eliminated_names": ";".join(eliminated_names),
                    "feasible": feasible,
                    "n_samples": 0 if samples is None else len(samples),
                    "margin_mean": summary["margin_mean"],
                    "margin_p05": summary["margin_p05"],
                    "margin_p50": summary["margin_p50"],
                }
            )

            if samples is None:
                continue

            stats = {
                "mean": np.mean(samples, axis=0),
                "median": np.median(samples, axis=0),
                "std": np.std(samples, axis=0, ddof=1),
                "p025": np.percentile(samples, 2.5, axis=0),
                "p975": np.percentile(samples, 97.5, axis=0),
            }

            eliminated_set = set(eliminated_names)
            for i, name in enumerate(names):
                contestant_summaries.append(
                    {
                        "season": season,
                        "week": week,
                        "celebrity_name": name,
                        "judge_pct": float(judge_pct[i]),
                        "fan_mean": float(stats["mean"][i]),
                        "fan_median": float(stats["median"][i]),
                        "fan_std": float(stats["std"][i]),
                        "fan_p025": float(stats["p025"][i]),
                        "fan_p975": float(stats["p975"][i]),
                        "eliminated": name in eliminated_set,
                        "n_active": n_active,
                    }
                )

    week_df = pd.DataFrame(week_summaries)
    contestant_df = pd.DataFrame(contestant_summaries)

    week_df.to_csv(out_dir / "weekly_summary.csv", index=False)
    contestant_df.to_csv(out_dir / "contestant_fan_vote_summary.csv", index=False)

    if args.replay_check:
        replay_elimination.run_replay(
            args.input,
            out_dir / "contestant_fan_vote_summary.csv",
            out_dir / "weekly_summary.csv",
            out_dir,
        )


if __name__ == "__main__":
    main()
