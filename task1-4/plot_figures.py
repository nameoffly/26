#!/usr/bin/env python3
"""
Visualization script for MCM 2026 Problem C
Generates publication-quality figures for the mathematical modeling paper.
Includes both baseline analysis (7 figures) and perturbation uncertainty analysis (15 figures).
"""

from __future__ import annotations

import os
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

# ============================================================================
# Configuration
# ============================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs_image")
DATA_DIR = os.path.join(BASE_DIR, "outputs")
BEST_PARAMS_DIR = os.path.join(DATA_DIR, "grid_a1p0_b0p05_g10p0")
PERTURB_DIR = os.path.join(BASE_DIR, "outputs_uncertainty_altopt", "perturb")

# Style settings
plt.style.use("seaborn-v0_8-whitegrid")
COLORS = {
    "primary": "#2E86AB",
    "secondary": "#A23B72",
    "tertiary": "#F18F01",
    "quaternary": "#C73E1D",
    "success": "#3A7D44",
    "neutral": "#6C757D",
}
PALETTE = ["#2E86AB", "#A23B72", "#F18F01", "#C73E1D", "#3A7D44", "#8338EC", "#FF6B6B", "#4ECDC4", "#45B7D1"]
DPI = 300
FIGSIZE_WIDE = (12, 6)
FIGSIZE_SQUARE = (8, 8)
FIGSIZE_TALL = (10, 8)


def ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================================
# Figure 1: Parameter Heatmap
# ============================================================================

def plot_param_heatmap():
    """Plot heatmap of objective values across beta-gamma parameter space."""
    df = pd.read_csv(os.path.join(DATA_DIR, "grid_search_summary.csv"))

    # Pivot for heatmap
    pivot = df.pivot(index="gamma", columns="beta", values="objective_sum")

    fig, ax = plt.subplots(figsize=(10, 7))

    # Create heatmap with custom colormap
    cmap = sns.color_palette("YlOrRd", as_cmap=True)
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".1f",
        cmap=cmap,
        linewidths=2,
        linecolor="white",
        cbar_kws={"label": "Total Objective Value", "shrink": 0.8},
        ax=ax,
        annot_kws={"size": 12, "weight": "bold"},
    )

    ax.set_xlabel(r"Smoothing Weight $\beta$", fontsize=14, fontweight="bold")
    ax.set_ylabel(r"Slack Penalty Weight $\gamma$", fontsize=14, fontweight="bold")
    ax.set_title("Parameter Grid Search: Objective Value Landscape", fontsize=16, fontweight="bold", pad=20)

    # Highlight optimal cell
    opt_row = df.loc[df["objective_sum"].idxmin()]
    opt_beta_idx = list(pivot.columns).index(opt_row["beta"])
    opt_gamma_idx = list(pivot.index).index(opt_row["gamma"])
    ax.add_patch(plt.Rectangle((opt_beta_idx, opt_gamma_idx), 1, 1, fill=False, edgecolor="lime", lw=4))

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "01_param_heatmap.png"), dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close()
    print("  [1/7] Parameter heatmap saved.")


# ============================================================================
# Figure 2: Bump Chart (Ranking Trajectories)
# ============================================================================

def plot_bump_chart():
    """Plot bump chart showing contestant ranking trajectories over weeks."""
    df = pd.read_csv(os.path.join(BEST_PARAMS_DIR, "weekly_predictions.csv"))

    # Select Season 2 (has more weeks and contestants)
    season_df = df[df["season"] == 2].copy()

    fig, ax = plt.subplots(figsize=(14, 8))

    # Get unique contestants
    contestants = season_df["celebrity_name"].unique()
    color_map = {name: PALETTE[i % len(PALETTE)] for i, name in enumerate(contestants)}

    for name in contestants:
        contestant_data = season_df[season_df["celebrity_name"] == name].sort_values("week")
        weeks = contestant_data["week"].values
        ranks = contestant_data["combined_rank"].values
        eliminated = contestant_data["actual_eliminated"].values

        # Plot line
        ax.plot(weeks, ranks, marker="o", markersize=10, linewidth=3,
                color=color_map[name], label=name, alpha=0.85)

        # Mark elimination point
        if eliminated.any():
            elim_idx = np.where(eliminated == 1)[0][0]
            ax.scatter(weeks[elim_idx], ranks[elim_idx], s=300, c="red",
                      marker="X", zorder=10, edgecolors="darkred", linewidths=2)

    # Invert y-axis (rank 1 at top)
    ax.invert_yaxis()

    ax.set_xlabel("Week", fontsize=14, fontweight="bold")
    ax.set_ylabel("Combined Rank (Lower is Better)", fontsize=14, fontweight="bold")
    ax.set_title("Season 2: Contestant Ranking Trajectories", fontsize=16, fontweight="bold", pad=20)

    # Legend outside
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=10, frameon=True, fancybox=True)

    # Grid styling
    ax.set_xticks(sorted(season_df["week"].unique()))
    ax.grid(True, alpha=0.3, linestyle="--")

    # Add elimination marker to legend
    handles, labels = ax.get_legend_handles_labels()
    handles.append(plt.Line2D([0], [0], marker="X", color="red", linestyle="None",
                              markersize=12, markeredgecolor="darkred", markeredgewidth=2))
    labels.append("Eliminated")
    ax.legend(handles, labels, loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=10, frameon=True)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "02_bump_chart.png"), dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close()
    print("  [2/7] Bump chart saved.")


# ============================================================================
# Figure 3: Objective Function Decomposition (Stacked Area)
# ============================================================================

def plot_objective_decomposition():
    """Plot stacked area chart of objective function components over weeks."""
    df = pd.read_csv(os.path.join(BEST_PARAMS_DIR, "weekly_penalty.csv"))

    # Select a season with interesting patterns (Season 30)
    seasons_to_plot = [29, 30]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    colors = [COLORS["primary"], COLORS["tertiary"], COLORS["quaternary"]]

    for idx, season in enumerate(seasons_to_plot):
        ax = axes[idx]
        season_df = df[df["season"] == season].sort_values("week")

        weeks = season_df["week"].values
        jterm = season_df["jterm"].values
        smooth = season_df["smooth"].values
        slack = season_df["slack"].values

        # Stacked area
        ax.stackplot(weeks, jterm, smooth, slack,
                    labels=[r"$J_{term}$ (Judge Proximity)",
                           r"$S_{term}$ (Smoothness)",
                           r"$\delta_{term}$ (Slack)"],
                    colors=colors, alpha=0.8)

        ax.set_xlabel("Week", fontsize=12, fontweight="bold")
        ax.set_ylabel("Penalty Value", fontsize=12, fontweight="bold")
        ax.set_title(f"Season {season}: Objective Decomposition", fontsize=14, fontweight="bold")
        ax.set_xticks(weeks)
        ax.legend(loc="upper right", fontsize=9, frameon=True)
        ax.grid(True, alpha=0.3, linestyle="--")

    plt.suptitle("Multi-Objective Function Components by Week", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "03_objective_decomposition.png"), dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close()
    print("  [3/7] Objective decomposition saved.")


# ============================================================================
# Figure 4: Elimination Consistency Bar Chart
# ============================================================================

def plot_consistency_bar():
    """Plot bar chart comparing elimination prediction consistency across seasons."""
    df = pd.read_csv(os.path.join(BEST_PARAMS_DIR, "consistency_summary.csv"))

    # Filter for season-level summaries (week == "ALL")
    season_summary = df[df["week"] == "ALL"].copy()
    season_summary["season"] = season_summary["season"].astype(int)
    season_summary = season_summary.sort_values("season")

    fig, ax = plt.subplots(figsize=(12, 6))

    seasons = season_summary["season"].values
    consistency = season_summary["consistent"].values * 100  # Convert to percentage

    # Create gradient colors based on consistency
    colors = [COLORS["success"] if c >= 90 else COLORS["tertiary"] if c >= 75 else COLORS["quaternary"]
              for c in consistency]

    bars = ax.bar(range(len(seasons)), consistency, color=colors, edgecolor="white", linewidth=2, width=0.7)

    # Add value labels on bars
    for bar, val in zip(bars, consistency):
        height = bar.get_height()
        ax.annotate(f"{val:.1f}%",
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 5),
                   textcoords="offset points",
                   ha="center", va="bottom",
                   fontsize=12, fontweight="bold")

    ax.set_xticks(range(len(seasons)))
    ax.set_xticklabels([f"S{s}" for s in seasons], fontsize=12)
    ax.set_xlabel("Season", fontsize=14, fontweight="bold")
    ax.set_ylabel("Prediction Consistency (%)", fontsize=14, fontweight="bold")
    ax.set_title("Elimination Prediction Consistency by Season", fontsize=16, fontweight="bold", pad=20)
    ax.set_ylim(0, 110)

    # Add reference line
    ax.axhline(y=100, color=COLORS["success"], linestyle="--", linewidth=2, alpha=0.7, label="Perfect Consistency")
    ax.axhline(y=np.mean(consistency), color=COLORS["neutral"], linestyle=":", linewidth=2, alpha=0.7,
               label=f"Average ({np.mean(consistency):.1f}%)")

    # Legend
    legend_elements = [
        Patch(facecolor=COLORS["success"], edgecolor="white", label="High (>=90%)"),
        Patch(facecolor=COLORS["tertiary"], edgecolor="white", label="Medium (75-90%)"),
        Patch(facecolor=COLORS["quaternary"], edgecolor="white", label="Low (<75%)"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=10, frameon=True)

    ax.grid(True, axis="y", alpha=0.3, linestyle="--")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "04_consistency_bar.png"), dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close()
    print("  [4/7] Consistency bar chart saved.")


# ============================================================================
# Figure 5: Judge vs Fan Rank Scatter Plot
# ============================================================================

def plot_rank_scatter():
    """Plot scatter plot comparing judge rank vs fan rank with regression."""
    df = pd.read_csv(os.path.join(BEST_PARAMS_DIR, "weekly_predictions.csv"))

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: All data scatter with density
    ax1 = axes[0]

    # Add jitter for visibility
    jitter_strength = 0.15
    x = df["judge_rank"] + np.random.uniform(-jitter_strength, jitter_strength, len(df))
    y = df["fan_rank"] + np.random.uniform(-jitter_strength, jitter_strength, len(df))

    scatter = ax1.scatter(x, y, c=df["season"], cmap="viridis", alpha=0.6, s=50, edgecolors="white", linewidths=0.5)

    # Perfect correlation line
    max_rank = max(df["judge_rank"].max(), df["fan_rank"].max())
    ax1.plot([1, max_rank], [1, max_rank], "r--", linewidth=2, label="Perfect Correlation", alpha=0.8)

    # Regression line
    z = np.polyfit(df["judge_rank"], df["fan_rank"], 1)
    p = np.poly1d(z)
    ax1.plot(range(1, max_rank + 1), p(range(1, max_rank + 1)),
            color=COLORS["primary"], linewidth=2, label=f"Linear Fit (slope={z[0]:.2f})")

    ax1.set_xlabel("Judge Rank", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Fan Rank", fontsize=12, fontweight="bold")
    ax1.set_title("Judge Rank vs Fan Rank (All Seasons)", fontsize=14, fontweight="bold")
    ax1.legend(loc="upper left", fontsize=10)

    cbar = plt.colorbar(scatter, ax=ax1, shrink=0.8)
    cbar.set_label("Season", fontsize=10)

    # Right: Correlation by season
    ax2 = axes[1]

    correlations = []
    seasons = sorted(df["season"].unique())
    for season in seasons:
        season_df = df[df["season"] == season]
        corr = season_df["judge_rank"].corr(season_df["fan_rank"])
        correlations.append(corr)

    bars = ax2.bar(range(len(seasons)), correlations, color=PALETTE[:len(seasons)],
                   edgecolor="white", linewidth=2, width=0.7)

    for bar, val in zip(bars, correlations):
        height = bar.get_height()
        ax2.annotate(f"{val:.3f}",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 5),
                    textcoords="offset points",
                    ha="center", va="bottom",
                    fontsize=10, fontweight="bold")

    ax2.set_xticks(range(len(seasons)))
    ax2.set_xticklabels([f"S{s}" for s in seasons], fontsize=10)
    ax2.set_xlabel("Season", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Pearson Correlation", fontsize=12, fontweight="bold")
    ax2.set_title("Judge-Fan Rank Correlation by Season", fontsize=14, fontweight="bold")
    ax2.set_ylim(0, 1.1)
    ax2.axhline(y=np.mean(correlations), color=COLORS["quaternary"], linestyle="--",
               linewidth=2, label=f"Mean ({np.mean(correlations):.3f})")
    ax2.legend(loc="lower right", fontsize=10)
    ax2.grid(True, axis="y", alpha=0.3, linestyle="--")

    plt.suptitle("Relationship Between Judge and Fan Rankings", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "05_rank_scatter.png"), dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close()
    print("  [5/7] Rank scatter plot saved.")


# ============================================================================
# Figure 6: Zipf Distribution
# ============================================================================

def plot_zipf_distribution():
    """Plot Zipf distribution of fan vote percentages."""
    df = pd.read_csv(os.path.join(BEST_PARAMS_DIR, "fan_vote_percent.csv"))

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Vote percentage vs rank (log-log)
    ax1 = axes[0]

    # Aggregate by rank
    rank_avg = df.groupby("fan_rank")["fan_vote_percent"].mean().reset_index()
    rank_avg = rank_avg.sort_values("fan_rank")

    ax1.scatter(rank_avg["fan_rank"], rank_avg["fan_vote_percent"] * 100,
               s=100, c=COLORS["primary"], alpha=0.8, edgecolors="white", linewidths=2, label="Observed")

    # Theoretical Zipf curve
    ranks = np.arange(1, rank_avg["fan_rank"].max() + 1)
    alpha, beta = 0.9, 1.0
    zipf_scores = 1 / (ranks + beta) ** alpha
    zipf_percent = zipf_scores / zipf_scores.sum() * 100

    ax1.plot(ranks, zipf_percent, color=COLORS["secondary"], linewidth=3,
            label=rf"Zipf Fit ($\alpha$={alpha}, $\beta$={beta})", linestyle="--")

    ax1.set_xlabel("Fan Rank", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Vote Share (%)", fontsize=12, fontweight="bold")
    ax1.set_title("Fan Vote Distribution (Linear Scale)", fontsize=14, fontweight="bold")
    ax1.legend(loc="upper right", fontsize=10)
    ax1.grid(True, alpha=0.3, linestyle="--")

    # Right: Log-log plot
    ax2 = axes[1]

    ax2.scatter(rank_avg["fan_rank"], rank_avg["fan_vote_percent"] * 100,
               s=100, c=COLORS["primary"], alpha=0.8, edgecolors="white", linewidths=2, label="Observed")
    ax2.plot(ranks, zipf_percent, color=COLORS["secondary"], linewidth=3,
            label=rf"Zipf: $p(r) \propto (r+\beta)^{{-\alpha}}$", linestyle="--")

    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlabel("Fan Rank (log scale)", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Vote Share % (log scale)", fontsize=12, fontweight="bold")
    ax2.set_title("Fan Vote Distribution (Log-Log Scale)", fontsize=14, fontweight="bold")
    ax2.legend(loc="upper right", fontsize=10)
    ax2.grid(True, alpha=0.3, linestyle="--", which="both")

    plt.suptitle("Zipf Distribution of Fan Voting Behavior", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "06_zipf_distribution.png"), dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close()
    print("  [6/7] Zipf distribution saved.")


# ============================================================================
# Figure 7: Radar Chart (Multi-Objective Trade-off)
# ============================================================================

def plot_radar_chart():
    """Plot radar chart showing multi-objective trade-offs across seasons."""
    df = pd.read_csv(os.path.join(BEST_PARAMS_DIR, "weekly_penalty.csv"))

    # Aggregate by season
    season_agg = df.groupby("season")[["jterm", "smooth", "slack"]].sum().reset_index()

    # Normalize each metric (0-1 scale for comparison)
    for col in ["jterm", "smooth", "slack"]:
        max_val = season_agg[col].max()
        if max_val > 0:
            season_agg[col + "_norm"] = season_agg[col] / max_val
        else:
            season_agg[col + "_norm"] = 0

    # Radar chart
    categories = ["Judge Proximity\n(Jterm)", "Smoothness\n(Smooth)", "Constraint Slack\n(Slack)"]
    n_cats = len(categories)

    # Compute angle for each category
    angles = [n / float(n_cats) * 2 * np.pi for n in range(n_cats)]
    angles += angles[:1]  # Complete the loop

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

    # Plot each season
    seasons = season_agg["season"].values
    for idx, season in enumerate(seasons):
        row = season_agg[season_agg["season"] == season].iloc[0]
        values = [row["jterm_norm"], row["smooth_norm"], row["slack_norm"]]
        values += values[:1]

        color = PALETTE[idx % len(PALETTE)]
        ax.plot(angles, values, "o-", linewidth=2, label=f"Season {season}", color=color, markersize=8)
        ax.fill(angles, values, alpha=0.15, color=color)

    # Customize
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=12, fontweight="bold")
    ax.set_ylim(0, 1.1)

    # Add y-axis labels
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["25%", "50%", "75%", "100%"], fontsize=9, color="gray")

    ax.set_title("Multi-Objective Trade-off by Season\n(Normalized Penalty Components)",
                fontsize=14, fontweight="bold", pad=30)

    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.0), fontsize=10, frameon=True)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "07_radar_chart.png"), dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close()
    print("  [7/22] Radar chart saved.")


# ============================================================================
# PERTURBATION UNCERTAINTY ANALYSIS FIGURES (08-22)
# ============================================================================

def load_perturb_data():
    """Load all perturbation analysis data."""
    stats = pd.read_csv(os.path.join(PERTURB_DIR, "perturb_rF_stats.csv"))
    elim = pd.read_csv(os.path.join(PERTURB_DIR, "perturb_elim_prob.csv"))
    corr = pd.read_csv(os.path.join(PERTURB_DIR, "perturb_rank_stability.csv"))
    baseline = pd.read_csv(os.path.join(BEST_PARAMS_DIR, "weekly_predictions.csv"))
    return stats, elim, corr, baseline


# ============================================================================
# Figure 8: Rank Uncertainty Error Bar Plot
# ============================================================================

def plot_rank_errorbar():
    """Plot error bars showing rank uncertainty with 95% CI."""
    stats, elim, corr, baseline = load_perturb_data()

    # Merge baseline with stats
    merged = pd.merge(
        baseline[["season", "week", "contestant_id", "celebrity_name", "fan_rank", "actual_eliminated"]],
        stats[["season", "week", "contestant_id", "fan_rank_mean", "fan_rank_p05", "fan_rank_p95"]],
        on=["season", "week", "contestant_id"]
    )

    # Select two seasons with interesting patterns
    seasons_to_plot = [29, 32]

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    for idx, season in enumerate(seasons_to_plot):
        ax = axes[idx]
        season_data = merged[merged["season"] == season]

        # Get week 3 data (early competition)
        week_data = season_data[season_data["week"] == 3].sort_values("fan_rank")

        contestants = week_data["celebrity_name"].str[:15].values  # Truncate names
        baseline_rank = week_data["fan_rank"].values
        mean_rank = week_data["fan_rank_mean"].values
        p05 = week_data["fan_rank_p05"].values
        p95 = week_data["fan_rank_p95"].values
        eliminated = week_data["actual_eliminated"].values

        y_pos = np.arange(len(contestants))

        # Compute error bounds (ensure non-negative)
        err_low = np.maximum(mean_rank - p05, 0)
        err_high = np.maximum(p95 - mean_rank, 0)

        # Plot error bars for perturbed ranks
        ax.errorbar(mean_rank, y_pos, xerr=[err_low, err_high],
                   fmt='o', color=COLORS["primary"], capsize=5, capthick=2,
                   markersize=10, label="Perturbed Mean (95% CI)", elinewidth=2)

        # Plot baseline ranks
        ax.scatter(baseline_rank, y_pos, marker='D', s=100, color=COLORS["quaternary"],
                  zorder=5, label="Baseline Rank", edgecolors="white", linewidths=1.5)

        # Highlight eliminated
        for i, elim in enumerate(eliminated):
            if elim:
                ax.scatter(baseline_rank[i], y_pos[i], marker='X', s=200, color="red",
                          zorder=10, edgecolors="darkred", linewidths=2)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(contestants, fontsize=9)
        ax.set_xlabel("Fan Rank", fontsize=12, fontweight="bold")
        ax.set_ylabel("Contestant", fontsize=12, fontweight="bold")
        ax.set_title(f"Season {season}, Week 3: Rank Uncertainty", fontsize=14, fontweight="bold")
        ax.legend(loc="lower right", fontsize=9)
        ax.grid(True, axis="x", alpha=0.3, linestyle="--")
        ax.invert_yaxis()

    plt.suptitle("Perturbation Analysis: Fan Rank Uncertainty", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "08_rank_errorbar.png"), dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close()
    print("  [8/22] Rank error bar plot saved.")


# ============================================================================
# Figure 9: Rank Ribbon/Band Plot
# ============================================================================

def plot_rank_ribbon():
    """Plot ribbon chart showing rank evolution with uncertainty bands."""
    stats, elim, corr, baseline = load_perturb_data()

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    seasons_to_plot = [2, 31]

    for idx, season in enumerate(seasons_to_plot):
        ax = axes[idx]
        season_stats = stats[stats["season"] == season]

        # Get top 3 contestants (by average rank)
        avg_rank = season_stats.groupby("contestant_id")["fan_rank_mean"].mean()
        top_contestants = avg_rank.nsmallest(4).index.tolist()

        for i, contestant_id in enumerate(top_contestants):
            contestant_data = season_stats[season_stats["contestant_id"] == contestant_id].sort_values("week")
            if len(contestant_data) < 2:
                continue

            weeks = contestant_data["week"].values
            mean_rank = contestant_data["fan_rank_mean"].values
            p05 = contestant_data["fan_rank_p05"].values
            p95 = contestant_data["fan_rank_p95"].values
            name = contestant_data["celebrity_name"].iloc[0][:12]

            color = PALETTE[i % len(PALETTE)]
            ax.plot(weeks, mean_rank, "o-", linewidth=2.5, color=color, label=name, markersize=8)
            ax.fill_between(weeks, p05, p95, alpha=0.2, color=color)

        ax.invert_yaxis()
        ax.set_xlabel("Week", fontsize=12, fontweight="bold")
        ax.set_ylabel("Fan Rank (with 95% CI)", fontsize=12, fontweight="bold")
        ax.set_title(f"Season {season}: Rank Evolution with Uncertainty", fontsize=14, fontweight="bold")
        ax.legend(loc="upper right", fontsize=9, frameon=True)
        ax.grid(True, alpha=0.3, linestyle="--")

    plt.suptitle("Rank Trajectory with Perturbation Uncertainty Bands", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "09_rank_ribbon.png"), dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close()
    print("  [9/22] Rank ribbon plot saved.")


# ============================================================================
# Figure 10: Rank Std Box Plot by Season
# ============================================================================

def plot_rank_std_boxplot():
    """Plot boxplot of rank standard deviations by season."""
    stats, _, _, _ = load_perturb_data()

    fig, ax = plt.subplots(figsize=(12, 7))

    # Create boxplot
    seasons = sorted(stats["season"].unique())
    data_by_season = [stats[stats["season"] == s]["fan_rank_std"].dropna().values for s in seasons]

    bp = ax.boxplot(data_by_season, patch_artist=True, labels=[f"S{s}" for s in seasons])

    # Color the boxes
    for i, (box, median) in enumerate(zip(bp["boxes"], bp["medians"])):
        box.set_facecolor(PALETTE[i % len(PALETTE)])
        box.set_alpha(0.7)
        median.set_color("black")
        median.set_linewidth(2)

    # Style whiskers and caps
    for whisker in bp["whiskers"]:
        whisker.set_color(COLORS["neutral"])
        whisker.set_linewidth(1.5)
    for cap in bp["caps"]:
        cap.set_color(COLORS["neutral"])
        cap.set_linewidth(1.5)
    for flier in bp["fliers"]:
        flier.set_markerfacecolor(COLORS["quaternary"])
        flier.set_markeredgecolor(COLORS["quaternary"])
        flier.set_markersize(5)

    # Add mean markers
    means = [np.mean(d) for d in data_by_season]
    ax.scatter(range(1, len(seasons) + 1), means, marker="D", s=80, color=COLORS["quaternary"],
              zorder=5, label="Mean", edgecolors="white", linewidths=1.5)

    ax.set_xlabel("Season", fontsize=14, fontweight="bold")
    ax.set_ylabel("Fan Rank Standard Deviation", fontsize=14, fontweight="bold")
    ax.set_title("Distribution of Rank Uncertainty by Season", fontsize=16, fontweight="bold", pad=20)
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, axis="y", alpha=0.3, linestyle="--")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "10_rank_std_boxplot.png"), dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close()
    print("  [10/22] Rank std boxplot saved.")


# ============================================================================
# Figure 11: Baseline vs Perturbed Scatter
# ============================================================================

def plot_baseline_vs_perturbed():
    """Plot scatter comparing baseline ranks to perturbed mean ranks."""
    stats, _, _, baseline = load_perturb_data()

    merged = pd.merge(
        baseline[["season", "week", "contestant_id", "fan_rank"]],
        stats[["season", "week", "contestant_id", "fan_rank_mean", "fan_rank_std"]],
        on=["season", "week", "contestant_id"]
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Scatter plot
    ax1 = axes[0]

    scatter = ax1.scatter(merged["fan_rank"], merged["fan_rank_mean"],
                         c=merged["fan_rank_std"], cmap="YlOrRd", alpha=0.7,
                         s=50, edgecolors="white", linewidths=0.5)

    # 1:1 line
    max_rank = max(merged["fan_rank"].max(), merged["fan_rank_mean"].max())
    ax1.plot([0, max_rank + 1], [0, max_rank + 1], "k--", linewidth=2, label="Perfect Agreement")

    # Regression
    z = np.polyfit(merged["fan_rank"], merged["fan_rank_mean"], 1)
    p = np.poly1d(z)
    x_line = np.linspace(1, max_rank, 100)
    ax1.plot(x_line, p(x_line), color=COLORS["primary"], linewidth=2,
            label=f"Fit (slope={z[0]:.3f})")

    ax1.set_xlabel("Baseline Fan Rank", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Perturbed Mean Fan Rank", fontsize=12, fontweight="bold")
    ax1.set_title("Baseline vs Perturbed Rankings", fontsize=14, fontweight="bold")
    ax1.legend(loc="upper left", fontsize=10)
    ax1.set_xlim(0, max_rank + 1)
    ax1.set_ylim(0, max_rank + 1)

    cbar = plt.colorbar(scatter, ax=ax1, shrink=0.8)
    cbar.set_label("Rank Std Dev", fontsize=10)

    # Right: Difference histogram
    ax2 = axes[1]
    diff = merged["fan_rank_mean"] - merged["fan_rank"]

    ax2.hist(diff, bins=30, color=COLORS["primary"], alpha=0.7, edgecolor="white", linewidth=1.5)
    ax2.axvline(x=0, color=COLORS["quaternary"], linewidth=2, linestyle="--", label="No Change")
    ax2.axvline(x=diff.mean(), color=COLORS["success"], linewidth=2, linestyle="-",
               label=f"Mean ({diff.mean():.3f})")

    ax2.set_xlabel("Rank Difference (Perturbed - Baseline)", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Frequency", fontsize=12, fontweight="bold")
    ax2.set_title("Distribution of Rank Changes", fontsize=14, fontweight="bold")
    ax2.legend(loc="upper right", fontsize=10)
    ax2.grid(True, axis="y", alpha=0.3, linestyle="--")

    plt.suptitle("Impact of Score Perturbation on Fan Rankings", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "11_baseline_vs_perturbed.png"), dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close()
    print("  [11/22] Baseline vs perturbed scatter saved.")


# ============================================================================
# Figure 12: Elimination Probability Heatmap
# ============================================================================

def plot_elim_prob_heatmap():
    """Plot heatmap of elimination probabilities."""
    _, elim, _, baseline = load_perturb_data()

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    seasons_to_plot = [29, 32]

    for idx, season in enumerate(seasons_to_plot):
        ax = axes[idx]

        season_elim = elim[elim["season"] == season].copy()
        season_base = baseline[baseline["season"] == season].copy()

        # Create pivot table
        pivot = season_elim.pivot(index="celebrity_name", columns="week", values="predicted_elim_prob")

        # Get actual elimination info
        actual_elim = season_base[season_base["actual_eliminated"] == 1][["week", "celebrity_name"]]

        # Sort by average elimination probability
        row_order = pivot.mean(axis=1).sort_values(ascending=False).index
        pivot = pivot.reindex(row_order)

        # Truncate names
        pivot.index = [name[:18] for name in pivot.index]

        sns.heatmap(pivot, cmap="YlOrRd", annot=True, fmt=".2f", linewidths=0.5,
                   ax=ax, cbar_kws={"label": "Elimination Prob.", "shrink": 0.8},
                   annot_kws={"size": 8}, vmin=0, vmax=1)

        ax.set_xlabel("Week", fontsize=12, fontweight="bold")
        ax.set_ylabel("Contestant", fontsize=12, fontweight="bold")
        ax.set_title(f"Season {season}: Elimination Probability Matrix", fontsize=14, fontweight="bold")

    plt.suptitle("Perturbation-Based Elimination Risk Assessment", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "12_elim_prob_heatmap.png"), dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close()
    print("  [12/22] Elimination probability heatmap saved.")


# ============================================================================
# Figure 13: Elimination Probability Bar Chart
# ============================================================================

def plot_elim_prob_bar():
    """Plot grouped bar chart of elimination probabilities."""
    _, elim, _, baseline = load_perturb_data()

    # Select Season 30, week with elimination
    season = 30
    week = 5

    fig, ax = plt.subplots(figsize=(14, 7))

    week_elim = elim[(elim["season"] == season) & (elim["week"] == week)].copy()
    week_base = baseline[(baseline["season"] == season) & (baseline["week"] == week)].copy()

    merged = pd.merge(week_elim, week_base[["contestant_id", "actual_eliminated"]], on="contestant_id")
    merged = merged.sort_values("predicted_elim_prob", ascending=False)

    contestants = merged["celebrity_name"].str[:15].values
    probs = merged["predicted_elim_prob"].values
    eliminated = merged["actual_eliminated"].values

    colors = [COLORS["quaternary"] if e else COLORS["primary"] for e in eliminated]

    bars = ax.barh(range(len(contestants)), probs, color=colors, edgecolor="white", linewidth=2)

    # Add value labels
    for i, (bar, prob) in enumerate(zip(bars, probs)):
        width = bar.get_width()
        ax.text(width + 0.02, bar.get_y() + bar.get_height()/2,
               f"{prob:.2f}", va="center", fontsize=10, fontweight="bold")

    ax.set_yticks(range(len(contestants)))
    ax.set_yticklabels(contestants, fontsize=10)
    ax.set_xlabel("Elimination Probability", fontsize=14, fontweight="bold")
    ax.set_ylabel("Contestant", fontsize=14, fontweight="bold")
    ax.set_title(f"Season {season}, Week {week}: Elimination Risk Under Perturbation", fontsize=16, fontweight="bold")
    ax.set_xlim(0, 1.15)
    ax.axvline(x=0.5, color=COLORS["neutral"], linestyle="--", linewidth=2, alpha=0.7, label="50% Threshold")

    # Legend
    legend_elements = [
        Patch(facecolor=COLORS["quaternary"], label="Actually Eliminated"),
        Patch(facecolor=COLORS["primary"], label="Survived"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=10, frameon=True)
    ax.grid(True, axis="x", alpha=0.3, linestyle="--")
    ax.invert_yaxis()

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "13_elim_prob_bar.png"), dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close()
    print("  [13/22] Elimination probability bar chart saved.")


# ============================================================================
# Figure 14: Elimination Probability Bubble Plot
# ============================================================================

def plot_elim_prob_bubble():
    """Plot bubble chart of elimination probabilities."""
    _, elim, _, baseline = load_perturb_data()

    fig, ax = plt.subplots(figsize=(14, 10))

    # Select Season 29
    season = 29
    season_elim = elim[elim["season"] == season].copy()
    season_base = baseline[baseline["season"] == season].copy()

    merged = pd.merge(season_elim, season_base[["season", "week", "contestant_id", "actual_eliminated"]],
                     on=["season", "week", "contestant_id"])

    # Create contestant index
    contestants = merged.groupby("contestant_id")["celebrity_name"].first()
    contestant_map = {cid: i for i, cid in enumerate(contestants.index)}
    merged["y_pos"] = merged["contestant_id"].map(contestant_map)

    # Filter to show only non-zero probabilities or actual eliminations
    plot_data = merged[(merged["predicted_elim_prob"] > 0.01) | (merged["actual_eliminated"] == 1)]

    # Size based on probability
    sizes = plot_data["predicted_elim_prob"] * 500 + 20

    # Color based on actual elimination
    colors = [COLORS["quaternary"] if e else COLORS["primary"] for e in plot_data["actual_eliminated"]]

    scatter = ax.scatter(plot_data["week"], plot_data["y_pos"], s=sizes, c=colors,
                        alpha=0.7, edgecolors="white", linewidths=1.5)

    ax.set_yticks(range(len(contestants)))
    ax.set_yticklabels([name[:15] for name in contestants.values], fontsize=9)
    ax.set_xticks(sorted(merged["week"].unique()))
    ax.set_xlabel("Week", fontsize=14, fontweight="bold")
    ax.set_ylabel("Contestant", fontsize=14, fontweight="bold")
    ax.set_title(f"Season {season}: Elimination Risk Bubble Chart\n(Bubble Size = Elimination Probability)",
                fontsize=16, fontweight="bold")

    # Legend
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS["quaternary"],
               markersize=15, label="Eliminated"),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS["primary"],
               markersize=15, label="At Risk"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=10, frameon=True)
    ax.grid(True, alpha=0.3, linestyle="--")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "14_elim_prob_bubble.png"), dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close()
    print("  [14/22] Elimination probability bubble plot saved.")


# ============================================================================
# Figure 15: Elimination Risk Distribution Donut Chart
# ============================================================================

def plot_elim_risk_donut():
    """Plot donut chart of elimination risk distribution."""
    _, elim, _, _ = load_perturb_data()

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Categorize probabilities
    def categorize(prob):
        if prob >= 0.7:
            return "High Risk (>=70%)"
        elif prob >= 0.3:
            return "Medium Risk (30-70%)"
        elif prob > 0:
            return "Low Risk (>0-30%)"
        else:
            return "Safe (0%)"

    elim["risk_category"] = elim["predicted_elim_prob"].apply(categorize)

    colors_risk = {
        "High Risk (>=70%)": COLORS["quaternary"],
        "Medium Risk (30-70%)": COLORS["tertiary"],
        "Low Risk (>0-30%)": COLORS["primary"],
        "Safe (0%)": COLORS["success"],
    }

    # Overall distribution
    ax1 = axes[0]
    counts = elim["risk_category"].value_counts()
    colors_list = [colors_risk[cat] for cat in counts.index]

    wedges, texts, autotexts = ax1.pie(counts.values, labels=counts.index, autopct="%1.1f%%",
                                        colors=colors_list, pctdistance=0.75,
                                        wedgeprops=dict(width=0.5, edgecolor="white"))
    ax1.set_title("Overall Risk Distribution", fontsize=14, fontweight="bold")

    # Early weeks (1-4)
    ax2 = axes[1]
    early = elim[elim["week"] <= 4]
    counts_early = early["risk_category"].value_counts()
    colors_early = [colors_risk[cat] for cat in counts_early.index]

    wedges2, texts2, autotexts2 = ax2.pie(counts_early.values, labels=counts_early.index, autopct="%1.1f%%",
                                           colors=colors_early, pctdistance=0.75,
                                           wedgeprops=dict(width=0.5, edgecolor="white"))
    ax2.set_title("Early Weeks (1-4)", fontsize=14, fontweight="bold")

    # Late weeks (7+)
    ax3 = axes[2]
    late = elim[elim["week"] >= 7]
    counts_late = late["risk_category"].value_counts()
    colors_late = [colors_risk[cat] for cat in counts_late.index]

    wedges3, texts3, autotexts3 = ax3.pie(counts_late.values, labels=counts_late.index, autopct="%1.1f%%",
                                           colors=colors_late, pctdistance=0.75,
                                           wedgeprops=dict(width=0.5, edgecolor="white"))
    ax3.set_title("Late Weeks (7+)", fontsize=14, fontweight="bold")

    for autotext in autotexts + autotexts2 + autotexts3:
        autotext.set_fontsize(9)
        autotext.set_fontweight("bold")

    plt.suptitle("Elimination Risk Category Distribution Under Perturbation", fontsize=16, fontweight="bold", y=1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "15_elim_risk_donut.png"), dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close()
    print("  [15/22] Elimination risk donut chart saved.")


# ============================================================================
# Figure 16: Rank Correlation Line Plot with CI Band
# ============================================================================

def plot_corr_line():
    """Plot line chart of rank correlations with confidence bands."""
    _, _, corr, _ = load_perturb_data()

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Left: Spearman correlation
    ax1 = axes[0]

    for season in sorted(corr["season"].unique()):
        season_data = corr[corr["season"] == season].sort_values("week")
        color = PALETTE[list(corr["season"].unique()).index(season) % len(PALETTE)]

        ax1.plot(season_data["week"], season_data["spearman_mean"], "o-",
                color=color, linewidth=2, markersize=6, label=f"S{season}", alpha=0.8)
        ax1.fill_between(season_data["week"], season_data["spearman_p05"], season_data["spearman_p95"],
                        color=color, alpha=0.15)

    ax1.set_xlabel("Week", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Spearman Correlation", fontsize=12, fontweight="bold")
    ax1.set_title("Spearman Rank Correlation with Baseline", fontsize=14, fontweight="bold")
    ax1.legend(loc="lower left", fontsize=8, ncol=3, frameon=True)
    ax1.set_ylim(0, 1.05)
    ax1.axhline(y=0.8, color=COLORS["neutral"], linestyle="--", linewidth=1.5, alpha=0.7)
    ax1.grid(True, alpha=0.3, linestyle="--")

    # Right: Kendall correlation
    ax2 = axes[1]

    for season in sorted(corr["season"].unique()):
        season_data = corr[corr["season"] == season].sort_values("week")
        color = PALETTE[list(corr["season"].unique()).index(season) % len(PALETTE)]

        ax2.plot(season_data["week"], season_data["kendall_mean"], "o-",
                color=color, linewidth=2, markersize=6, label=f"S{season}", alpha=0.8)
        ax2.fill_between(season_data["week"], season_data["kendall_p05"], season_data["kendall_p95"],
                        color=color, alpha=0.15)

    ax2.set_xlabel("Week", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Kendall Tau Correlation", fontsize=12, fontweight="bold")
    ax2.set_title("Kendall Tau Correlation with Baseline", fontsize=14, fontweight="bold")
    ax2.legend(loc="lower left", fontsize=8, ncol=3, frameon=True)
    ax2.set_ylim(0, 1.05)
    ax2.axhline(y=0.8, color=COLORS["neutral"], linestyle="--", linewidth=1.5, alpha=0.7)
    ax2.grid(True, alpha=0.3, linestyle="--")

    plt.suptitle("Rank Stability Under Score Perturbation", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "16_corr_line.png"), dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close()
    print("  [16/22] Correlation line plot saved.")


# ============================================================================
# Figure 17: Rank Correlation Heatmap
# ============================================================================

def plot_corr_heatmap():
    """Plot heatmap of rank correlations by season and week."""
    _, _, corr, _ = load_perturb_data()

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Left: Spearman
    ax1 = axes[0]
    pivot_spearman = corr.pivot(index="season", columns="week", values="spearman_mean")

    sns.heatmap(pivot_spearman, cmap="RdYlGn", annot=True, fmt=".2f", linewidths=0.5,
               ax=ax1, cbar_kws={"label": "Spearman Corr.", "shrink": 0.8},
               annot_kws={"size": 8}, vmin=0.3, vmax=1.0, center=0.7)

    ax1.set_xlabel("Week", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Season", fontsize=12, fontweight="bold")
    ax1.set_title("Spearman Correlation Matrix", fontsize=14, fontweight="bold")

    # Right: Kendall
    ax2 = axes[1]
    pivot_kendall = corr.pivot(index="season", columns="week", values="kendall_mean")

    sns.heatmap(pivot_kendall, cmap="RdYlGn", annot=True, fmt=".2f", linewidths=0.5,
               ax=ax2, cbar_kws={"label": "Kendall Tau", "shrink": 0.8},
               annot_kws={"size": 8}, vmin=0.3, vmax=1.0, center=0.7)

    ax2.set_xlabel("Week", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Season", fontsize=12, fontweight="bold")
    ax2.set_title("Kendall Tau Correlation Matrix", fontsize=14, fontweight="bold")

    plt.suptitle("Rank Correlation Stability Across Seasons and Weeks", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "17_corr_heatmap.png"), dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close()
    print("  [17/22] Correlation heatmap saved.")


# ============================================================================
# Figure 18: Correlation Distribution Histogram
# ============================================================================

def plot_corr_histogram():
    """Plot histogram comparing Spearman and Kendall correlation distributions."""
    _, _, corr, _ = load_perturb_data()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Side-by-side histogram
    ax1 = axes[0]

    bins = np.linspace(0, 1, 25)
    ax1.hist(corr["spearman_mean"], bins=bins, alpha=0.7, color=COLORS["primary"],
            label="Spearman", edgecolor="white", linewidth=1.5)
    ax1.hist(corr["kendall_mean"], bins=bins, alpha=0.7, color=COLORS["secondary"],
            label="Kendall", edgecolor="white", linewidth=1.5)

    ax1.axvline(x=corr["spearman_mean"].mean(), color=COLORS["primary"], linestyle="--",
               linewidth=2, label=f"Spearman Mean ({corr['spearman_mean'].mean():.3f})")
    ax1.axvline(x=corr["kendall_mean"].mean(), color=COLORS["secondary"], linestyle="--",
               linewidth=2, label=f"Kendall Mean ({corr['kendall_mean'].mean():.3f})")

    ax1.set_xlabel("Correlation Coefficient", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Frequency", fontsize=12, fontweight="bold")
    ax1.set_title("Distribution of Rank Correlations", fontsize=14, fontweight="bold")
    ax1.legend(loc="upper left", fontsize=9)
    ax1.grid(True, axis="y", alpha=0.3, linestyle="--")

    # Right: KDE plot
    ax2 = axes[1]

    sns.kdeplot(data=corr, x="spearman_mean", ax=ax2, color=COLORS["primary"],
               linewidth=3, label="Spearman", fill=True, alpha=0.3)
    sns.kdeplot(data=corr, x="kendall_mean", ax=ax2, color=COLORS["secondary"],
               linewidth=3, label="Kendall", fill=True, alpha=0.3)

    ax2.set_xlabel("Correlation Coefficient", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Density", fontsize=12, fontweight="bold")
    ax2.set_title("Kernel Density Estimation", fontsize=14, fontweight="bold")
    ax2.legend(loc="upper left", fontsize=10)
    ax2.grid(True, alpha=0.3, linestyle="--")

    plt.suptitle("Spearman vs Kendall Correlation Comparison", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "18_corr_histogram.png"), dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close()
    print("  [18/22] Correlation histogram saved.")


# ============================================================================
# Figure 19: Correlation Boxplot by Season
# ============================================================================

def plot_corr_boxplot():
    """Plot boxplot of correlations by season."""
    _, _, corr, _ = load_perturb_data()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    seasons = sorted(corr["season"].unique())

    # Left: Spearman
    ax1 = axes[0]
    data_spearman = [corr[corr["season"] == s]["spearman_mean"].values for s in seasons]
    bp1 = ax1.boxplot(data_spearman, patch_artist=True, labels=[f"S{s}" for s in seasons])

    for i, box in enumerate(bp1["boxes"]):
        box.set_facecolor(PALETTE[i % len(PALETTE)])
        box.set_alpha(0.7)
    for median in bp1["medians"]:
        median.set_color("black")
        median.set_linewidth(2)

    ax1.set_xlabel("Season", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Spearman Correlation", fontsize=12, fontweight="bold")
    ax1.set_title("Spearman Correlation by Season", fontsize=14, fontweight="bold")
    ax1.axhline(y=0.8, color=COLORS["neutral"], linestyle="--", linewidth=1.5, alpha=0.7, label="0.8 Threshold")
    ax1.legend(loc="lower left", fontsize=9)
    ax1.grid(True, axis="y", alpha=0.3, linestyle="--")
    ax1.set_ylim(0, 1.05)

    # Right: Kendall
    ax2 = axes[1]
    data_kendall = [corr[corr["season"] == s]["kendall_mean"].values for s in seasons]
    bp2 = ax2.boxplot(data_kendall, patch_artist=True, labels=[f"S{s}" for s in seasons])

    for i, box in enumerate(bp2["boxes"]):
        box.set_facecolor(PALETTE[i % len(PALETTE)])
        box.set_alpha(0.7)
    for median in bp2["medians"]:
        median.set_color("black")
        median.set_linewidth(2)

    ax2.set_xlabel("Season", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Kendall Tau Correlation", fontsize=12, fontweight="bold")
    ax2.set_title("Kendall Tau by Season", fontsize=14, fontweight="bold")
    ax2.axhline(y=0.8, color=COLORS["neutral"], linestyle="--", linewidth=1.5, alpha=0.7, label="0.8 Threshold")
    ax2.legend(loc="lower left", fontsize=9)
    ax2.grid(True, axis="y", alpha=0.3, linestyle="--")
    ax2.set_ylim(0, 1.05)

    plt.suptitle("Rank Stability Distribution by Season", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "19_corr_boxplot.png"), dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close()
    print("  [19/22] Correlation boxplot saved.")


# ============================================================================
# Figure 20: Faceted Violin Plot of Rank Std
# ============================================================================

def plot_rank_violin():
    """Plot faceted violin plot of rank standard deviations."""
    stats, _, _, _ = load_perturb_data()

    fig, ax = plt.subplots(figsize=(14, 7))

    # Prepare data
    seasons = sorted(stats["season"].unique())

    parts = ax.violinplot([stats[stats["season"] == s]["fan_rank_std"].dropna().values for s in seasons],
                          positions=range(len(seasons)), showmeans=True, showmedians=True)

    # Color the violins
    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(PALETTE[i % len(PALETTE)])
        pc.set_edgecolor("black")
        pc.set_alpha(0.7)

    parts["cmeans"].set_color(COLORS["quaternary"])
    parts["cmeans"].set_linewidth(2)
    parts["cmedians"].set_color("black")
    parts["cmedians"].set_linewidth(2)

    ax.set_xticks(range(len(seasons)))
    ax.set_xticklabels([f"S{s}" for s in seasons], fontsize=12)
    ax.set_xlabel("Season", fontsize=14, fontweight="bold")
    ax.set_ylabel("Fan Rank Standard Deviation", fontsize=14, fontweight="bold")
    ax.set_title("Distribution Shape of Rank Uncertainty by Season", fontsize=16, fontweight="bold", pad=20)

    # Add legend
    legend_elements = [
        Line2D([0], [0], color=COLORS["quaternary"], linewidth=2, label="Mean"),
        Line2D([0], [0], color="black", linewidth=2, label="Median"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=10)
    ax.grid(True, axis="y", alpha=0.3, linestyle="--")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "20_rank_violin.png"), dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close()
    print("  [20/22] Rank violin plot saved.")


# ============================================================================
# Figure 21: Uncertainty Summary Bar Chart
# ============================================================================

def plot_uncertainty_summary():
    """Plot summary bar chart of uncertainty metrics by season."""
    stats, elim, corr, _ = load_perturb_data()

    seasons = sorted(stats["season"].unique())

    # Compute metrics per season
    avg_std = [stats[stats["season"] == s]["fan_rank_std"].mean() for s in seasons]
    avg_spearman = [corr[corr["season"] == s]["spearman_mean"].mean() for s in seasons]

    # Compute elimination entropy
    def entropy(probs):
        probs = probs[(probs > 0) & (probs < 1)]
        if len(probs) == 0:
            return 0
        return -np.sum(probs * np.log2(probs) + (1 - probs) * np.log2(1 - probs)) / len(probs)

    avg_entropy = [entropy(elim[elim["season"] == s]["predicted_elim_prob"].values) for s in seasons]

    # Normalize for comparison
    avg_std_norm = np.array(avg_std) / max(avg_std)
    avg_entropy_norm = np.array(avg_entropy) / max(avg_entropy) if max(avg_entropy) > 0 else np.zeros(len(avg_entropy))

    fig, ax = plt.subplots(figsize=(14, 7))

    x = np.arange(len(seasons))
    width = 0.25

    bars1 = ax.bar(x - width, avg_std_norm, width, label="Rank Uncertainty (Normalized Std)",
                   color=COLORS["primary"], edgecolor="white", linewidth=1.5)
    bars2 = ax.bar(x, 1 - np.array(avg_spearman), width, label="Rank Instability (1 - Spearman)",
                   color=COLORS["secondary"], edgecolor="white", linewidth=1.5)
    bars3 = ax.bar(x + width, avg_entropy_norm, width, label="Elimination Uncertainty (Normalized Entropy)",
                   color=COLORS["tertiary"], edgecolor="white", linewidth=1.5)

    ax.set_xticks(x)
    ax.set_xticklabels([f"S{s}" for s in seasons], fontsize=12)
    ax.set_xlabel("Season", fontsize=14, fontweight="bold")
    ax.set_ylabel("Normalized Uncertainty Index", fontsize=14, fontweight="bold")
    ax.set_title("Multi-Dimensional Uncertainty Summary by Season", fontsize=16, fontweight="bold", pad=20)
    ax.legend(loc="upper right", fontsize=10, frameon=True)
    ax.grid(True, axis="y", alpha=0.3, linestyle="--")
    ax.set_ylim(0, 1.1)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "21_uncertainty_summary.png"), dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close()
    print("  [21/22] Uncertainty summary bar chart saved.")


# ============================================================================
# Figure 22: Rank Change Waterfall Chart
# ============================================================================

def plot_rank_waterfall():
    """Plot waterfall chart showing rank changes from baseline to perturbed."""
    stats, _, _, baseline = load_perturb_data()

    merged = pd.merge(
        baseline[["season", "week", "contestant_id", "celebrity_name", "fan_rank"]],
        stats[["season", "week", "contestant_id", "fan_rank_mean"]],
        on=["season", "week", "contestant_id"]
    )

    merged["rank_change"] = merged["fan_rank_mean"] - merged["fan_rank"]

    # Select Season 32, Week 4
    season, week = 32, 4
    week_data = merged[(merged["season"] == season) & (merged["week"] == week)].copy()
    week_data = week_data.sort_values("rank_change")

    fig, ax = plt.subplots(figsize=(14, 7))

    contestants = week_data["celebrity_name"].str[:15].values
    baseline_ranks = week_data["fan_rank"].values
    changes = week_data["rank_change"].values

    colors = [COLORS["success"] if c < 0 else COLORS["quaternary"] if c > 0 else COLORS["neutral"]
              for c in changes]

    # Create waterfall-style bars
    y_pos = np.arange(len(contestants))

    # Plot baseline as starting point
    ax.barh(y_pos, baseline_ranks, height=0.4, color=COLORS["primary"], alpha=0.5,
           label="Baseline Rank", edgecolor="white")

    # Plot changes
    for i, (base, change) in enumerate(zip(baseline_ranks, changes)):
        if change != 0:
            ax.arrow(base, i, change, 0, head_width=0.2, head_length=0.1,
                    fc=colors[i], ec=colors[i], linewidth=2)

    # Plot final positions
    ax.scatter(baseline_ranks + changes, y_pos, s=100, c=colors, zorder=5,
              edgecolors="white", linewidths=2, marker="D")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(contestants, fontsize=10)
    ax.set_xlabel("Fan Rank", fontsize=14, fontweight="bold")
    ax.set_ylabel("Contestant", fontsize=14, fontweight="bold")
    ax.set_title(f"Season {season}, Week {week}: Rank Changes Under Perturbation\n(Arrow: Baseline → Perturbed Mean)",
                fontsize=16, fontweight="bold")

    # Legend
    legend_elements = [
        Patch(facecolor=COLORS["success"], label="Improved (Lower Rank)"),
        Patch(facecolor=COLORS["quaternary"], label="Worsened (Higher Rank)"),
        Patch(facecolor=COLORS["primary"], alpha=0.5, label="Baseline Rank"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=10, frameon=True)
    ax.grid(True, axis="x", alpha=0.3, linestyle="--")
    ax.invert_yaxis()

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "22_rank_waterfall.png"), dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close()
    print("  [22/22] Rank waterfall chart saved.")


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 60)
    print("MCM 2026 Problem C - Visualization Script")
    print("=" * 60)

    ensure_output_dir()
    print(f"\nOutput directory: {OUTPUT_DIR}\n")

    print("Generating baseline figures (1-7)...")
    plot_param_heatmap()
    plot_bump_chart()
    plot_objective_decomposition()
    plot_consistency_bar()
    plot_rank_scatter()
    plot_zipf_distribution()
    plot_radar_chart()

    print("\nGenerating perturbation uncertainty figures (8-22)...")
    plot_rank_errorbar()
    plot_rank_ribbon()
    plot_rank_std_boxplot()
    plot_baseline_vs_perturbed()
    plot_elim_prob_heatmap()
    plot_elim_prob_bar()
    plot_elim_prob_bubble()
    plot_elim_risk_donut()
    plot_corr_line()
    plot_corr_heatmap()
    plot_corr_histogram()
    plot_corr_boxplot()
    plot_rank_violin()
    plot_uncertainty_summary()
    plot_rank_waterfall()

    print("\n" + "=" * 60)
    print("All 22 figures generated successfully!")
    print(f"Output location: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
