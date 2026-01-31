import numpy as np
from scipy.optimize import linprog


def build_constraints(judge_pct, eliminated_idx, epsilon=0.0):
    judge_pct = np.asarray(judge_pct, dtype=float)
    n = len(judge_pct)

    A_ub = []
    b_ub = []

    # Non-negativity: -I x <= 0
    A_ub.extend(-np.eye(n))
    b_ub.extend(np.zeros(n))

    eliminated_set = set(eliminated_idx)
    non_elim = [i for i in range(n) if i not in eliminated_set]

    for e in eliminated_idx:
        for i in non_elim:
            row = np.zeros(n)
            row[e] = 1.0
            row[i] = -1.0
            A_ub.append(row)
            b_ub.append(judge_pct[i] - judge_pct[e] - float(epsilon))

    A_ub = np.asarray(A_ub, dtype=float)
    b_ub = np.asarray(b_ub, dtype=float)

    A_eq = np.ones((1, n), dtype=float)
    b_eq = np.array([1.0], dtype=float)

    return {"A_ub": A_ub, "b_ub": b_ub, "A_eq": A_eq, "b_eq": b_eq}


def find_feasible_point(A_ub, b_ub, A_eq, b_eq):
    n = A_eq.shape[1]
    c = np.zeros(n, dtype=float)
    bounds = [(0.0, 1.0)] * n

    res = linprog(
        c,
        A_ub=A_ub,
        b_ub=b_ub,
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=bounds,
        method="highs",
    )

    if not res.success:
        return None
    return res.x


def find_interior_point_lp_ensemble(A_ub, b_ub, A_eq, b_eq, m=30, seed=None):
    n = A_eq.shape[1]
    bounds = [(0.0, 1.0)] * n
    rng = np.random.default_rng(seed)
    vertices = []

    for _ in range(m):
        c = rng.normal(size=n)
        res = linprog(
            -c,
            A_ub=A_ub,
            b_ub=b_ub,
            A_eq=A_eq,
            b_eq=b_eq,
            bounds=bounds,
            method="highs",
        )
        if res.success:
            vertices.append(res.x)

    if len(vertices) < 2:
        return None

    w = rng.dirichlet(np.ones(len(vertices)))
    x0 = np.zeros(n, dtype=float)
    for wi, vi in zip(w, vertices):
        x0 += wi * vi
    return x0
