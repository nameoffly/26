#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from math import ceil
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PERTURB_DIR = os.path.join(
    BASE_DIR, "outputs_uncertainty_altopt", "perturb"
)
DEFAULT_OUTPUT_DIR = os.path.join(BASE_DIR, "outputs_images_uncertainty")

plt.style.use("seaborn-v0_8-whitegrid")
PALETTE = [
    "#2E86AB",
    "#A23B72",
    "#F18F01",
    "#3A7D44",
    "#8338EC",
    "#FF6B6B",
    "#4ECDC4",
    "#45B7D1",
    "#6C757D",
]


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _load_inputs(perturb_dir: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rF_path = os.path.join(perturb_dir, "perturb_rF_stats.csv")
    elim_path = os.path.join(perturb_dir, "perturb_elim_prob.csv")
    corr_path = os.path.join(perturb_dir, "perturb_rank_stability.csv")
    for path in (rF_path, elim_path, corr_path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing input file: {path}")
    rF_df = pd.read_csv(rF_path)
    elim_df = pd.read_csv(elim_path)
    corr_df = pd.read_csv(corr_path)
    return rF_df, elim_df, corr_df


def _save(fig: plt.Figure, output_dir: str, name: str) -> None:
    path = os.path.join(output_dir, name)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_stability_trend(corr_df: pd.DataFrame, output_dir: str) -> None:
    weeks = sorted(corr_df["week"].unique())
    agg = corr_df.groupby("week").agg(
        spearman_mean=("spearman_mean", "mean"),
        spearman_p05=("spearman_mean", lambda x: x.quantile(0.05)),
        spearman_p95=("spearman_mean", lambda x: x.quantile(0.95)),
        kendall_mean=("kendall_mean", "mean"),
        kendall_p05=("kendall_mean", lambda x: x.quantile(0.05)),
        kendall_p95=("kendall_mean", lambda x: x.quantile(0.95)),
    )
    agg = agg.reindex(weeks)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(weeks, agg["spearman_mean"], color=PALETTE[0], marker="o")
    axes[0].fill_between(
        weeks, agg["spearman_p05"], agg["spearman_p95"], color=PALETTE[0], alpha=0.2
    )
    axes[0].set_title("Spearman Stability (All Seasons)")
    axes[0].set_xlabel("Week")
    axes[0].set_ylabel("Spearman ρ")

    axes[1].plot(weeks, agg["kendall_mean"], color=PALETTE[1], marker="o")
    axes[1].fill_between(
        weeks, agg["kendall_p05"], agg["kendall_p95"], color=PALETTE[1], alpha=0.2
    )
    axes[1].set_title("Kendall Stability (All Seasons)")
    axes[1].set_xlabel("Week")
    axes[1].set_ylabel("Kendall τ")

    _save(fig, output_dir, "01_stability_trend.png")


def plot_stability_scatter(corr_df: pd.DataFrame, output_dir: str) -> None:
    seasons = sorted(corr_df["season"].unique())
    fig, ax = plt.subplots(figsize=(6, 6))
    for idx, season in enumerate(seasons):
        data = corr_df[corr_df["season"] == season]
        ax.scatter(
            data["spearman_mean"],
            data["kendall_mean"],
            color=PALETTE[idx % len(PALETTE)],
            label=f"Season {season}",
            alpha=0.8,
        )
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1)
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Spearman ρ")
    ax.set_ylabel("Kendall τ")
    ax.set_title("Spearman vs Kendall (All Seasons)")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8)
    _save(fig, output_dir, "02_stability_scatter.png")


def plot_stability_heatmap(corr_df: pd.DataFrame, output_dir: str) -> None:
    pivot = corr_df.pivot(index="season", columns="week", values="spearman_mean")
    seasons = pivot.index.tolist()
    weeks = pivot.columns.tolist()
    data = pivot.to_numpy()

    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(data, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(weeks)))
    ax.set_xticklabels(weeks)
    ax.set_yticks(range(len(seasons)))
    ax.set_yticklabels(seasons)
    ax.set_xlabel("Week")
    ax.set_ylabel("Season")
    ax.set_title("Spearman Stability Heatmap")
    fig.colorbar(im, ax=ax, label="Spearman ρ")
    _save(fig, output_dir, "03_stability_heatmap.png")


def plot_fan_rank_std_boxplot(rF_df: pd.DataFrame, output_dir: str) -> None:
    weeks = sorted(rF_df["week"].unique())
    data = [
        rF_df[rF_df["week"] == w]["fan_rank_std"].dropna().values for w in weeks
    ]
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.boxplot(data, tick_labels=[str(w) for w in weeks], showfliers=False)
    ax.set_xlabel("Week")
    ax.set_ylabel("fan_rank std")
    ax.set_title("Fan Rank Variability by Week (All Seasons)")
    _save(fig, output_dir, "04_fan_rank_std_boxplot.png")


def plot_final_week_errorbar(
    rF_df: pd.DataFrame, output_dir: str, season: int
) -> None:
    season_df = rF_df[rF_df["season"] == season].copy()
    if season_df.empty:
        return
    final_week = int(season_df["week"].max())
    week_df = season_df[season_df["week"] == final_week].copy()
    if week_df.empty:
        return

    week_df = week_df.sort_values("fan_rank_mean")
    y = week_df["fan_rank_mean"].to_numpy()
    p05 = week_df["fan_rank_p05"].to_numpy()
    p95 = week_df["fan_rank_p95"].to_numpy()
    yerr = np.vstack([np.maximum(0.0, y - p05), np.maximum(0.0, p95 - y)])

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.errorbar(
        range(len(week_df)),
        y,
        yerr=yerr,
        fmt="o",
        color=PALETTE[0],
        ecolor=PALETTE[0],
        capsize=3,
    )
    ax.set_xticks(range(len(week_df)))
    ax.set_xticklabels(week_df["celebrity_name"], rotation=45, ha="right")
    ax.set_ylabel("fan_rank")
    ax.set_title(f"Season {season} Final Week Fan Rank (p05–p95)")
    _save(fig, output_dir, "05_final_week_errorbar.png")


def plot_uncertainty_heatmap(
    rF_df: pd.DataFrame, output_dir: str, season: int
) -> None:
    season_df = rF_df[rF_df["season"] == season].copy()
    if season_df.empty:
        return
    season_df["interval_width"] = (
        season_df["fan_rank_p95"] - season_df["fan_rank_p05"]
    )
    pivot = season_df.pivot(
        index="celebrity_name", columns="week", values="interval_width"
    )
    data = pivot.to_numpy()
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(data, aspect="auto", cmap="magma")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel("Week")
    ax.set_ylabel("Contestant")
    ax.set_title(f"Season {season} Fan Rank Uncertainty Width")
    fig.colorbar(im, ax=ax, label="p95 - p05")
    _save(fig, output_dir, "06_fan_rank_uncertainty_heatmap.png")


def plot_elim_prob_heatmap(
    elim_df: pd.DataFrame, output_dir: str, season: int
) -> None:
    season_df = elim_df[elim_df["season"] == season].copy()
    if season_df.empty:
        return
    pivot = season_df.pivot(
        index="celebrity_name", columns="week", values="predicted_elim_prob"
    )
    data = pivot.to_numpy()
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(data, aspect="auto", cmap="Reds")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel("Week")
    ax.set_ylabel("Contestant")
    ax.set_title(f"Season {season} Elimination Probability Heatmap")
    fig.colorbar(im, ax=ax, label="Elimination Probability")
    _save(fig, output_dir, "07_elim_prob_heatmap.png")


def plot_elim_prob_topk(
    elim_df: pd.DataFrame, output_dir: str, season: int, top_k: int
) -> None:
    season_df = elim_df[elim_df["season"] == season].copy()
    if season_df.empty:
        return
    weeks = sorted(season_df["week"].unique())
    ncols = 3
    nrows = ceil(len(weeks) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 3 * nrows))
    axes = np.atleast_1d(axes).flatten()
    for idx, week in enumerate(weeks):
        ax = axes[idx]
        week_df = season_df[season_df["week"] == week].sort_values(
            "predicted_elim_prob", ascending=False
        )
        week_df = week_df.head(top_k)
        ax.barh(
            week_df["celebrity_name"],
            week_df["predicted_elim_prob"],
            color=PALETTE[2],
        )
        ax.set_title(f"Week {week} Top {top_k}")
        ax.set_xlim(0, 1)
        ax.invert_yaxis()
    for j in range(len(weeks), len(axes)):
        axes[j].axis("off")
    fig.suptitle(
        f"Season {season} Top-{top_k} Elimination Risk by Week", y=1.02
    )
    _save(fig, output_dir, "08_elim_prob_topk.png")


def plot_elim_risk_trajectory(
    elim_df: pd.DataFrame, output_dir: str, season: int, top_m: int
) -> None:
    season_df = elim_df[elim_df["season"] == season].copy()
    if season_df.empty:
        return
    max_risk = (
        season_df.groupby("celebrity_name")["predicted_elim_prob"]
        .max()
        .sort_values(ascending=False)
    )
    top_names = max_risk.head(top_m).index.tolist()
    fig, ax = plt.subplots(figsize=(10, 4))
    for idx, name in enumerate(top_names):
        data = season_df[season_df["celebrity_name"] == name].sort_values("week")
        ax.plot(
            data["week"],
            data["predicted_elim_prob"],
            marker="o",
            label=name,
            color=PALETTE[idx % len(PALETTE)],
        )
    ax.set_xlabel("Week")
    ax.set_ylabel("Elimination Probability")
    ax.set_title(f"Season {season} Risk Trajectories (Top {top_m})")
    ax.set_ylim(0, 1)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8)
    _save(fig, output_dir, "09_elim_risk_trajectory.png")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--perturb-dir",
        default=DEFAULT_PERTURB_DIR,
        help="Directory with perturb_*.csv outputs",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to save uncertainty plots",
    )
    parser.add_argument(
        "--season",
        type=int,
        default=29,
        help="Typical season for single-season plots",
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--top-m", type=int, default=6)
    args = parser.parse_args()

    _ensure_dir(args.output_dir)
    rF_df, elim_df, corr_df = _load_inputs(args.perturb_dir)

    plot_stability_trend(corr_df, args.output_dir)
    plot_stability_scatter(corr_df, args.output_dir)
    plot_stability_heatmap(corr_df, args.output_dir)
    plot_fan_rank_std_boxplot(rF_df, args.output_dir)
    plot_final_week_errorbar(rF_df, args.output_dir, args.season)
    plot_uncertainty_heatmap(rF_df, args.output_dir, args.season)
    plot_elim_prob_heatmap(elim_df, args.output_dir, args.season)
    plot_elim_prob_topk(elim_df, args.output_dir, args.season, args.top_k)
    plot_elim_risk_trajectory(elim_df, args.output_dir, args.season, args.top_m)


if __name__ == "__main__":
    main()
