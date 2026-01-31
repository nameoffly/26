import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

import constraints  # noqa: E402


def _satisfies(A_ub, b_ub, A_eq, b_eq, x, tol=1e-8):
    if A_ub is not None and len(A_ub):
        if np.any(A_ub @ x - b_ub > tol):
            return False
    if A_eq is not None and len(A_eq):
        if np.any(np.abs(A_eq @ x - b_eq) > tol):
            return False
    return True


def test_build_constraints_basic():
    judge_pct = np.array([0.6, 0.4])
    eliminated_idx = [1]
    cons = constraints.build_constraints(judge_pct, eliminated_idx, epsilon=0.0)

    A_ub = cons["A_ub"]
    b_ub = cons["b_ub"]
    A_eq = cons["A_eq"]
    b_eq = cons["b_eq"]

    assert A_eq.shape == (1, 2)
    assert np.allclose(A_eq, np.array([[1.0, 1.0]]))
    assert np.allclose(b_eq, np.array([1.0]))

    # Expect non-negativity + one elimination constraint
    # Non-negativity encoded as -I x <= 0
    assert A_ub.shape[0] == 3
    assert np.allclose(A_ub[:2], -np.eye(2))
    # Elimination: x_e - x_i <= judge_i - judge_e
    # Here: x2 - x1 <= 0.2
    assert np.allclose(A_ub[2], np.array([-1.0, 1.0]))
    assert np.allclose(b_ub[2], 0.2)


def test_find_feasible_point():
    judge_pct = np.array([0.55, 0.45])
    eliminated_idx = [1]
    cons = constraints.build_constraints(judge_pct, eliminated_idx, epsilon=0.0)

    x0 = constraints.find_feasible_point(
        cons["A_ub"], cons["b_ub"], cons["A_eq"], cons["b_eq"]
    )
    assert x0 is not None
    assert _satisfies(cons["A_ub"], cons["b_ub"], cons["A_eq"], cons["b_eq"], x0)


def test_find_interior_point_lp_ensemble():
    judge_pct = np.array([0.5, 0.3, 0.2])
    eliminated_idx = []
    cons = constraints.build_constraints(judge_pct, eliminated_idx, epsilon=0.0)
    x0 = constraints.find_interior_point_lp_ensemble(
        cons["A_ub"], cons["b_ub"], cons["A_eq"], cons["b_eq"], m=10, seed=123
    )
    assert x0 is not None
    assert _satisfies(cons["A_ub"], cons["b_ub"], cons["A_eq"], cons["b_eq"], x0)
    assert np.all(x0 >= 0.0)
