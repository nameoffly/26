#!/usr/bin/env python3
"""
Perturbation Analysis Visualization Script
For MCM 2026 Problem C - Kendall Correlation Visualizations

Input: outputs_uncertainty_altopt/perturb/
Output: outputs_image_perturb/
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

# ============================================================================
# Configuration
# ============================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PERTURB_DIR = os.path.join(BASE_DIR, "outputs_uncertainty_altopt", "perturb")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs_image_perturb")

# Style settings
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.family"] = "DejaVu Sans"

# Color palette
PALETTE = [
    "#2E86AB", "#A23B72", "#F18F01", "#C73E1D", "#3A7D44",
    "#8338EC", "#FF6B6B", "#4ECDC4", "#45B7D1"
]
COLORS = {
    "primary": "#2E86AB",
    "secondary": "#A23B72",
    "tertiary": "#F18F01",
    "quaternary": "#C73E1D",
    "success": "#3A7D44",
    "neutral": "#6C757D",
}

DPI = 300


def ensure_output_dir():
    """Create output directory"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_corr_data() -> pd.DataFrame:
    """Load correlation data"""
    return pd.read_csv(os.path.join(PERTURB_DIR, "perturb_rank_stability.csv"))


def load_stats_data() -> pd.DataFrame:
    """Load rank statistics data"""
    return pd.read_csv(os.path.join(PERTURB_DIR, "perturb_rF_stats.csv"))


# ============================================================================
# Figure 1: Kendall Correlation Heatmap
# ============================================================================

def plot_kendall_heatmap():
    """Season x Week heatmap of Kendall tau correlation with distinct colors"""
    corr = load_corr_data()

    fig, ax = plt.subplots(figsize=(14, 8))

    # Create pivot table
    pivot_kendall = corr.pivot(index="season", columns="week", values="kendall_mean")

    # Create custom colormap with more distinct colors
    # Red (low) -> Yellow (medium) -> Green (high)
    colors_list = [
        "#D32F2F",  # Strong Red (very low: < 0.4)
        "#F44336",  # Red (low: 0.4-0.5)
        "#FF9800",  # Orange (medium-low: 0.5-0.6)
        "#FFC107",  # Amber (medium: 0.6-0.7)
        "#CDDC39",  # Lime (medium-high: 0.7-0.8)
        "#8BC34A",  # Light Green (high: 0.8-0.9)
        "#4CAF50",  # Green (very high: 0.9-1.0)
        "#2E7D32",  # Dark Green (excellent: > 0.95)
    ]
    custom_cmap = LinearSegmentedColormap.from_list("distinct", colors_list, N=256)

    # Plot heatmap
    sns.heatmap(
        pivot_kendall,
        cmap=custom_cmap,
        annot=True,
        fmt=".2f",
        linewidths=2,
        linecolor="white",
        ax=ax,
        cbar_kws={
            "label": "Kendall Tau Correlation",
            "shrink": 0.85,
            "aspect": 30
        },
        annot_kws={"size": 10, "weight": "bold"},
        vmin=0.3,
        vmax=1.0,
        square=False
    )

    # Style adjustments
    ax.set_xlabel("Week", fontsize=15, fontweight="bold", labelpad=10)
    ax.set_ylabel("Season", fontsize=15, fontweight="bold", labelpad=10)
    ax.set_title("Rank Stability Under Perturbation: Kendall Tau Correlation",
                fontsize=18, fontweight="bold", pad=20)

    # Adjust tick labels
    ax.tick_params(axis='both', labelsize=12)

    # Colorbar styling
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=11)
    cbar.set_label("Kendall Tau Correlation", fontsize=13, fontweight="bold")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "kendall_heatmap.png"),
                dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close()
    print("[1/5] Kendall Correlation Heatmap saved")


# ============================================================================
# Figure 2: Kendall Boxplot by Season
# ============================================================================

def plot_kendall_boxplot_by_season():
    """Boxplot comparing Kendall tau across different seasons"""
    corr = load_corr_data()

    fig, ax = plt.subplots(figsize=(14, 7))

    seasons = sorted(corr["season"].unique())

    # Collect kendall_mean values for each season (across all weeks)
    data_by_season = [
        corr[corr["season"] == s]["kendall_mean"].dropna().values
        for s in seasons
    ]

    # Create boxplot
    bp = ax.boxplot(
        data_by_season,
        patch_artist=True,
        labels=[f"S{s}" for s in seasons],
        widths=0.6
    )

    # Style boxes
    for i, (box, median) in enumerate(zip(bp["boxes"], bp["medians"])):
        box.set_facecolor(PALETTE[i % len(PALETTE)])
        box.set_alpha(0.8)
        box.set_edgecolor("black")
        box.set_linewidth(1.5)
        median.set_color("white")
        median.set_linewidth(2.5)

    for whisker in bp["whiskers"]:
        whisker.set_color(COLORS["neutral"])
        whisker.set_linewidth(1.5)
        whisker.set_linestyle("--")
    for cap in bp["caps"]:
        cap.set_color(COLORS["neutral"])
        cap.set_linewidth(2)
    for flier in bp["fliers"]:
        flier.set_markerfacecolor(COLORS["quaternary"])
        flier.set_markeredgecolor("white")
        flier.set_markersize(8)
        flier.set_alpha(0.8)

    # Add mean markers
    means = [np.mean(d) for d in data_by_season]
    ax.scatter(
        range(1, len(seasons) + 1), means,
        marker="D", s=100, color=COLORS["tertiary"],
        zorder=5, label="Mean", edgecolors="white", linewidths=2
    )

    # Add reference line
    ax.axhline(y=0.8, color=COLORS["success"], linestyle="--",
               linewidth=2, alpha=0.8, label="High Stability (0.8)")

    ax.set_xlabel("Season", fontsize=14, fontweight="bold")
    ax.set_ylabel("Kendall Tau Correlation", fontsize=14, fontweight="bold")
    ax.set_title("Kendall Tau Distribution by Season (Averaged Across Weeks)",
                fontsize=16, fontweight="bold", pad=20)
    ax.legend(loc="lower right", fontsize=11, frameon=True, fancybox=True)
    ax.grid(True, axis="y", alpha=0.4, linestyle="-")
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "kendall_boxplot_by_season.png"),
                dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close()
    print("[2/5] Kendall Boxplot by Season saved")


# ============================================================================
# Figure 3: Kendall Boxplot for Season 29 by Week
# ============================================================================

def plot_kendall_boxplot_s29_by_week():
    """Boxplot comparing Kendall tau across weeks for Season 29"""
    corr = load_corr_data()

    # Filter Season 29
    s29 = corr[corr["season"] == 29].sort_values("week")

    fig, ax = plt.subplots(figsize=(14, 7))

    weeks = s29["week"].values

    # For each week, use [p05, mean, p95] to represent the distribution
    # We'll create box-like visualization using error bars
    kendall_mean = s29["kendall_mean"].values
    kendall_p05 = s29["kendall_p05"].values
    kendall_p95 = s29["kendall_p95"].values

    x_pos = np.arange(len(weeks))

    # Plot bars for mean values
    bars = ax.bar(x_pos, kendall_mean, width=0.6, color=PALETTE[:len(weeks)],
                  edgecolor="black", linewidth=1.5, alpha=0.8)

    # Add error bars for p05-p95 range
    err_low = kendall_mean - kendall_p05
    err_high = kendall_p95 - kendall_mean
    ax.errorbar(x_pos, kendall_mean, yerr=[err_low, err_high],
               fmt='none', color='black', capsize=8, capthick=2.5,
               elinewidth=2.5, label="95% CI (P05-P95)")

    # Add mean value labels on top of bars
    for i, (x, m) in enumerate(zip(x_pos, kendall_mean)):
        ax.annotate(f'{m:.2f}',
                   xy=(x, m + err_high[i] + 0.02),
                   ha='center', va='bottom',
                   fontsize=10, fontweight='bold')

    # Add reference line
    ax.axhline(y=0.8, color=COLORS["success"], linestyle="--",
               linewidth=2, alpha=0.8, label="High Stability (0.8)")

    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"Week {w}" for w in weeks], fontsize=11, fontweight="bold")
    ax.set_xlabel("Week", fontsize=14, fontweight="bold")
    ax.set_ylabel("Kendall Tau Correlation", fontsize=14, fontweight="bold")
    ax.set_title("Season 29: Kendall Tau by Week (Mean with 95% CI)",
                fontsize=16, fontweight="bold", pad=20)
    ax.legend(loc="lower right", fontsize=11, frameon=True, fancybox=True)
    ax.grid(True, axis="y", alpha=0.4, linestyle="-")
    ax.set_ylim(0, 1.15)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "kendall_boxplot_s29_by_week.png"),
                dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close()
    print("[3/5] Kendall Boxplot for Season 29 by Week saved")


# ============================================================================
# Figure 4: Rank Volatility (fan_rank_std) Boxplot by Season
# ============================================================================

def plot_rank_std_boxplot_by_season():
    """Boxplot comparing fan rank std (volatility) across different seasons"""
    stats = load_stats_data()

    fig, ax = plt.subplots(figsize=(14, 7))

    seasons = sorted(stats["season"].unique())

    # Collect fan_rank_std values for each season (across all weeks and contestants)
    data_by_season = [
        stats[stats["season"] == s]["fan_rank_std"].dropna().values
        for s in seasons
    ]

    # Create boxplot
    bp = ax.boxplot(
        data_by_season,
        patch_artist=True,
        labels=[f"S{s}" for s in seasons],
        widths=0.6
    )

    # Style boxes
    for i, (box, median) in enumerate(zip(bp["boxes"], bp["medians"])):
        box.set_facecolor(PALETTE[i % len(PALETTE)])
        box.set_alpha(0.8)
        box.set_edgecolor("black")
        box.set_linewidth(1.5)
        median.set_color("white")
        median.set_linewidth(2.5)

    for whisker in bp["whiskers"]:
        whisker.set_color(COLORS["neutral"])
        whisker.set_linewidth(1.5)
        whisker.set_linestyle("--")
    for cap in bp["caps"]:
        cap.set_color(COLORS["neutral"])
        cap.set_linewidth(2)
    for flier in bp["fliers"]:
        flier.set_markerfacecolor(COLORS["quaternary"])
        flier.set_markeredgecolor("white")
        flier.set_markersize(6)
        flier.set_alpha(0.7)

    # Add mean markers
    means = [np.mean(d) for d in data_by_season]
    ax.scatter(
        range(1, len(seasons) + 1), means,
        marker="D", s=100, color=COLORS["tertiary"],
        zorder=5, label="Mean", edgecolors="white", linewidths=2
    )

    # Add overall mean reference line
    overall_mean = stats["fan_rank_std"].mean()
    ax.axhline(y=overall_mean, color=COLORS["quaternary"], linestyle="--",
               linewidth=2, alpha=0.8, label=f"Overall Mean ({overall_mean:.2f})")

    # Add sample size annotations
    for i, (s, data) in enumerate(zip(seasons, data_by_season)):
        ax.annotate(f'n={len(data)}',
                   xy=(i + 1, -0.08),
                   ha='center', va='top',
                   fontsize=9, color=COLORS["neutral"])

    ax.set_xlabel("Season", fontsize=14, fontweight="bold")
    ax.set_ylabel("Fan Rank Std Dev (Volatility)", fontsize=14, fontweight="bold")
    ax.set_title("Rank Volatility Distribution by Season Under Perturbation",
                fontsize=16, fontweight="bold", pad=20)
    ax.legend(loc="upper right", fontsize=11, frameon=True, fancybox=True)
    ax.grid(True, axis="y", alpha=0.4, linestyle="-")
    ax.set_ylim(-0.15, max([max(d) for d in data_by_season]) * 1.1)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "rank_std_boxplot_by_season.png"),
                dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close()
    print("[4/5] Rank Volatility Boxplot by Season saved")


# ============================================================================
# Figure 5: Rank Volatility (fan_rank_std) Boxplot by Week (Across All Seasons)
# ============================================================================

def plot_rank_std_boxplot_by_week():
    """Boxplot comparing fan rank std (volatility) across different weeks (all seasons combined)"""
    stats = load_stats_data()

    fig, ax = plt.subplots(figsize=(14, 7))

    weeks = sorted(stats["week"].unique())

    # Collect fan_rank_std values for each week (across all seasons and contestants)
    data_by_week = [
        stats[stats["week"] == w]["fan_rank_std"].dropna().values
        for w in weeks
    ]

    # Create boxplot
    bp = ax.boxplot(
        data_by_week,
        patch_artist=True,
        labels=[f"W{w}" for w in weeks],
        widths=0.6
    )

    # Style boxes - use gradient color from light to dark
    n_weeks = len(weeks)
    for i, (box, median) in enumerate(zip(bp["boxes"], bp["medians"])):
        # Gradient from blue to red as weeks progress
        ratio = i / (n_weeks - 1) if n_weeks > 1 else 0
        r = int(46 + ratio * (199 - 46))
        g = int(134 + ratio * (59 - 134))
        b = int(171 + ratio * (114 - 171))
        box.set_facecolor(f"#{r:02x}{g:02x}{b:02x}")
        box.set_alpha(0.8)
        box.set_edgecolor("black")
        box.set_linewidth(1.5)
        median.set_color("white")
        median.set_linewidth(2.5)

    for whisker in bp["whiskers"]:
        whisker.set_color(COLORS["neutral"])
        whisker.set_linewidth(1.5)
        whisker.set_linestyle("--")
    for cap in bp["caps"]:
        cap.set_color(COLORS["neutral"])
        cap.set_linewidth(2)
    for flier in bp["fliers"]:
        flier.set_markerfacecolor(COLORS["quaternary"])
        flier.set_markeredgecolor("white")
        flier.set_markersize(6)
        flier.set_alpha(0.7)

    # Add mean markers and connect with line
    means = [np.mean(d) for d in data_by_week]
    ax.scatter(
        range(1, len(weeks) + 1), means,
        marker="D", s=80, color=COLORS["tertiary"],
        zorder=5, label="Mean", edgecolors="white", linewidths=2
    )
    ax.plot(range(1, len(weeks) + 1), means, color=COLORS["tertiary"],
            linewidth=2, linestyle="-", alpha=0.6)

    # Add overall mean reference line
    overall_mean = stats["fan_rank_std"].mean()
    ax.axhline(y=overall_mean, color=COLORS["quaternary"], linestyle="--",
               linewidth=2, alpha=0.8, label=f"Overall Mean ({overall_mean:.2f})")

    # Add sample size annotations
    for i, (w, data) in enumerate(zip(weeks, data_by_week)):
        ax.annotate(f'n={len(data)}',
                   xy=(i + 1, -0.08),
                   ha='center', va='top',
                   fontsize=8, color=COLORS["neutral"])

    ax.set_xlabel("Week", fontsize=14, fontweight="bold")
    ax.set_ylabel("Fan Rank Std Dev (Volatility)", fontsize=14, fontweight="bold")
    ax.set_title("Rank Volatility Distribution by Week (All Seasons Combined)",
                fontsize=16, fontweight="bold", pad=20)
    ax.legend(loc="upper right", fontsize=11, frameon=True, fancybox=True)
    ax.grid(True, axis="y", alpha=0.4, linestyle="-")
    ax.set_ylim(-0.15, max([max(d) for d in data_by_week if len(d) > 0]) * 1.1)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "rank_std_boxplot_by_week.png"),
                dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close()
    print("[5/5] Rank Volatility Boxplot by Week saved")


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 50)
    print("Generating Perturbation Analysis Visualizations")
    print("=" * 50)

    ensure_output_dir()

    plot_kendall_heatmap()
    plot_kendall_boxplot_by_season()
    plot_kendall_boxplot_s29_by_week()
    plot_rank_std_boxplot_by_season()
    plot_rank_std_boxplot_by_week()

    print(f"\nOutput directory: {OUTPUT_DIR}")
    print("=" * 50)


if __name__ == "__main__":
    main()
