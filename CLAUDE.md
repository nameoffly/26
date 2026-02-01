# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a 2026 MCM (Mathematical Contest in Modeling) Problem C solution that analyzes "Dancing with the Stars" TV show data. The project infers unknown audience voting rankings from judge scores and elimination results using constraint programming optimization.

## Commands

### Installation
```bash
cd task1-4
pip install -r requirements.txt
```

### Run Baseline Solver
```bash
python task1-4/solve_rank.py
python task1-4/solve_rank.py --alpha 1.0 --beta 0.05 --gamma 10.0 --time-limit 60 --workers 8
```

### Parameter Grid Search
```bash
python task1-4/grid_search.py --time-limit 120
```

### Uncertainty Analysis
```bash
# CP-SAT based (slower, more accurate)
python task1-4/uncertainty_analysis.py --processes 12 --n-samples 500 --noise-std 0.7 --epsilons 0.01,0.05,0.1

# Alternating optimization (faster heuristic)
python task1-4/uncertainty_altopt.py --processes 12 --n-samples 500 --sweeps 5 --init baseline
```

### Generate Visualizations
```bash
python task1-4/plot_figures.py --mode all --top-k 6
```

## Architecture

### Data Flow
```
Data_4.xlsx → data_processing.py → model_rank.py → solve_rank.py → outputs/
                                                         ↓
                                    uncertainty_analysis.py / uncertainty_altopt.py
                                                         ↓
                                                  plot_figures.py → outputs_image/
```

### Core Modules (task1-4/)

- **data_processing.py**: Extracts weekly contestant data from Excel, computes judge rankings, parses elimination weeks
- **model_rank.py**: Builds CP-SAT constraint programming model with binary assignment variables, elimination constraints, and weighted objective function
- **solve_rank.py**: Main entry point - runs optimization and outputs predictions to `outputs/`
- **grid_search.py**: Grid search over weight parameters (α, β, γ)
- **uncertainty_analysis.py**: CP-SAT based uncertainty quantification with near-optimal interval analysis and Monte Carlo perturbation
- **uncertainty_altopt.py**: Fast heuristic alternative using Hungarian algorithm with alternating optimization
- **zipf_vote_percent.py**: Models audience voting percentages using Zipf distribution
- **plot_figures.py**: Comprehensive visualization module (8+ plot types)
- **near_opt_altopt.py**: Fast near-optimal interval analysis using alternating optimization
- **plot_perturb.py**: Perturbation analysis visualizations (Kendall correlation heatmaps)
- **plot_uncertainty_altopt.py**: Uncertainty visualization for alternating optimization results

### Mathematical Model

**Decision Variables:**
- Binary assignment: x_{i,k,w} ∈ {0,1} (contestant i gets rank k in week w)
- Audience rankings: rF_{i,w} = Σ k·x_{i,k,w}
- Slack variables: δ_{j,w} ≥ 0 for constraint violations

**Objective Function:**
```
min α·(Judge Proximity) + β·(Smoothing) + γ·(Slack Penalty)
```
- Judge Proximity = Σ(rF - rJ)²
- Smoothing = Σ(rF_w - rF_{w-1})²
- Slack Penalty = Σδ

**Optimal Parameters:** α=1.0, β=0.05, γ=10.0

### Output Directories

- `task1-4/outputs/`: Baseline solver results (predictions, consistency, penalties)
- `task1-4/outputs_uncertainty/`: CP-SAT uncertainty analysis results
- `task1-4/outputs_uncertainty_altopt/`: Alternating optimization results
- `task1-4/outputs_image/`: Generated visualization plots from plot_figures.py
- `task1-4/outputs_image_perturb/`: Perturbation analysis plots from plot_perturb.py
- `task1-4/outputs_images_uncertainty/`: Uncertainty visualization plots
- `task1-4/final_outputs/`: Final consolidated outputs for paper submission

## Key Implementation Notes

- CP-SAT uses integer arithmetic internally; weight scaling handles decimal precision
- Uncertainty analysis uses ProcessPoolExecutor with single-threaded solving per process
- The alternating optimization approach is significantly faster than CP-SAT for large perturbation studies
- Rank correlation metrics: Spearman ρ and Kendall τ-a for stability assessment
- Plotting scripts require seaborn (not in requirements.txt): `pip install seaborn`
- Default seasons analyzed: 1, 2, 28, 29, 30, 31, 32, 33, 34
