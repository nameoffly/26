# Stability Metrics: Formulas and Interpretation

This note documents the exact stability metrics produced by the two analysis modes:

- Near-opt interval analysis (CP-SAT): `outputs_uncertainty/near_opt/*`
- Input perturbation analysis (CP-SAT or alt-opt): `outputs_uncertainty/perturb/*` or
  `outputs_uncertainty_altopt/perturb/*`

All notation below is per season and per week unless stated otherwise.

## Notation

- i: contestant index
- w: week index
- rJ_{i,w}: judge rank (lower is better)
- rF_{i,w}: fan rank (lower is better)
- R_{i,w} = rJ_{i,w} + rF_{i,w}: combined rank
- Opt: optimal objective value for the season
- epsilon: near-opt tolerance (e.g., 0.01, 0.05, 0.1)
- k_actual: number of eliminated contestants in week w

## 1) Near-opt interval stability (near_opt_interval.csv)

We restrict the objective by a near-opt bound:

```
Objective <= (1 + epsilon) * Opt
```

For each contestant i in week w, we solve two optimization problems:

```
rF_min(i,w) = min rF_{i,w}  subject to all constraints + near-opt bound
rF_max(i,w) = max rF_{i,w}  subject to all constraints + near-opt bound
```

Then we derive combined-rank bounds:

```
R_min(i,w) = rJ_{i,w} + rF_min(i,w)
R_max(i,w) = rJ_{i,w} + rF_max(i,w)
```

Interpretation:

- Interval width: delta_rF = rF_max - rF_min
  Smaller width => more stable ranking under near-opt solutions.

## 2) Elimination certainty from intervals (near_opt_elim_certainty.csv)

Using R_min / R_max, define counts of "definitely worse" and "possibly worse":

```
worse_definite(i) = #{ j != i | R_min(j) > R_max(i) }
worse_possible(i) = #{ j != i | R_max(j) > R_min(i) }
```

Decision rules:

```
if worse_definite(i) >= k_actual:
    status = "always_safe"
elif worse_possible(i) <= k_actual - 1:
    status = "always_eliminated"
else:
    status = "uncertain"
```

If any R_min/R_max is missing, status is "unsolved". If k_actual == 0, status is
"no_elimination".

## 3) Input perturbation stability

### 3.1 Score perturbation model

For each sample:

```
judge_score' = judge_score + N(0, sigma)
judge_score' = max(judge_score', min_score)
```

Then the model is re-solved to obtain rF_{i,w} and R_{i,w}.

### 3.2 Fan-rank statistics (perturb_rF_stats.csv)

Across N samples for each (i,w):

```
fan_rank_mean = (1/N) * sum rF_{i,w}^{(s)}
fan_rank_std  = sqrt( (1/(N-1)) * sum (rF_{i,w}^{(s)} - mean)^2 )
fan_rank_p05  = 5th percentile of { rF_{i,w}^{(s)} }
fan_rank_p95  = 95th percentile of { rF_{i,w}^{(s)} }
```

Smaller std and tighter [p05, p95] => more stable ranking under noise.

### 3.3 Elimination probability (perturb_elim_prob.csv)

For each (i,w):

```
predicted_elim_prob = (1/N) * sum I[ i is eliminated in sample s ]
```

Values close to 0 or 1 indicate stable elimination outcomes.

### 3.4 Rank stability correlations (perturb_rank_stability.csv)

Let x be the baseline combined ranks for week w, and y the perturbed combined
ranks for the same contestants. We compute:

Spearman rho (rank correlation):

```
rho = corr( rank(x), rank(y) )
```

Kendall tau-a:

```
tau = (C - D) / (n(n-1)/2)
```

where C is the number of concordant pairs and D is the number of discordant pairs.

Notes:

- rank(.) uses average ranks for ties.
- In alt-opt perturbation, Spearman uses rank-transformed values.
- If you keep the original CP-SAT perturb script unchanged, the reported
  "spearman" is Pearson on raw combined-rank values (not rank-transformed).

Higher rho/tau => more stable ranking under perturbations.
