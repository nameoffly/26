import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

import constraints  # noqa: E402
import sampler  # noqa: E402


def _satisfies(A_ub, b_ub, A_eq, b_eq, x, tol=1e-8):
    if A_ub is not None and len(A_ub):
        if np.any(A_ub @ x - b_ub > tol):
            return False
    if A_eq is not None and len(A_eq):
        if np.any(np.abs(A_eq @ x - b_eq) > tol):
            return False
    return True


def test_hit_and_run_simplex_constraints():
    judge_pct = np.array([0.5, 0.5, 0.0])
    eliminated_idx = []
    cons = constraints.build_constraints(judge_pct, eliminated_idx, epsilon=0.0)
    x0 = constraints.find_feasible_point(
        cons["A_ub"], cons["b_ub"], cons["A_eq"], cons["b_eq"]
    )
    samples = sampler.hit_and_run(
        x0,
        cons["A_ub"],
        cons["b_ub"],
        cons["A_eq"],
        cons["b_eq"],
        n_samples=200,
        burn_in=50,
        thin=1,
        seed=123,
    )

    assert samples.shape == (200, 3)
    assert np.all(samples >= -1e-10)
    assert np.allclose(samples.sum(axis=1), 1.0, atol=1e-6)


def test_hit_and_run_elimination_constraints():
    judge_pct = np.array([0.6, 0.3, 0.1])
    eliminated_idx = [2]
    cons = constraints.build_constraints(judge_pct, eliminated_idx, epsilon=0.0)
    x0 = constraints.find_feasible_point(
        cons["A_ub"], cons["b_ub"], cons["A_eq"], cons["b_eq"]
    )
    samples = sampler.hit_and_run(
        x0,
        cons["A_ub"],
        cons["b_ub"],
        cons["A_eq"],
        cons["b_eq"],
        n_samples=100,
        burn_in=50,
        thin=1,
        seed=7,
    )

    for x in samples:
        assert _satisfies(cons["A_ub"], cons["b_ub"], cons["A_eq"], cons["b_eq"], x)


def test_step_bounds_tolerance():
    A_ub = np.array([[1.0], [-1.0]])
    b_ub = np.array([1.0, -1.0 - 1e-13])
    x = np.array([1.0])
    d = np.array([1.0])

    bounds = sampler._step_bounds(A_ub, b_ub, x, d, tol=1e-12)
    assert bounds is not None
    t_min, t_max = bounds
    assert np.isclose(t_min, t_max)
