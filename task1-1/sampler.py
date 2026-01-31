import numpy as np


def _nullspace_basis(A_eq, tol=1e-12):
    if A_eq is None or len(A_eq) == 0:
        return None
    u, s, vh = np.linalg.svd(A_eq, full_matrices=True)
    rank = np.sum(s > tol)
    if rank >= vh.shape[0]:
        return np.zeros((A_eq.shape[1], 0))
    return vh[rank:].T


def _step_bounds(A_ub, b_ub, x, d, tol=1e-12):
    t_min = -np.inf
    t_max = np.inf

    Ad = A_ub @ d
    Ax = A_ub @ x
    for adi, axi, bi in zip(Ad, Ax, b_ub):
        if abs(adi) < 1e-14:
            if axi > bi + 1e-10:
                return None
            continue
        t = (bi - axi) / adi
        if adi > 0:
            t_max = min(t_max, t)
        else:
            t_min = max(t_min, t)

    if t_min > t_max:
        if t_min - t_max <= tol:
            t_mid = 0.5 * (t_min + t_max)
            return t_mid, t_mid
        return None

    return t_min, t_max


def hit_and_run(
    x0,
    A_ub,
    b_ub,
    A_eq=None,
    b_eq=None,
    n_samples=1000,
    burn_in=200,
    thin=1,
    seed=None,
):
    x = np.asarray(x0, dtype=float).copy()
    n = x.shape[0]
    rng = np.random.default_rng(seed)

    if A_ub is None:
        A_ub = np.zeros((0, n), dtype=float)
        b_ub = np.zeros(0, dtype=float)

    basis = _nullspace_basis(A_eq)
    if basis is not None and basis.shape[1] == 0:
        return np.repeat(x[None, :], n_samples, axis=0)

    samples = []
    total_steps = burn_in + n_samples * thin
    for step in range(total_steps):
        attempts = 0
        while True:
            if basis is None:
                d = rng.normal(size=n)
            else:
                z = rng.normal(size=basis.shape[1])
                d = basis @ z

            if np.allclose(d, 0.0):
                attempts += 1
                if attempts >= 50:
                    raise ValueError("无可行步长区间")
                continue

            bounds = _step_bounds(A_ub, b_ub, x, d, tol=1e-12)
            if bounds is None:
                attempts += 1
                if attempts >= 50:
                    raise ValueError("无可行步长区间")
                continue

            t_min, t_max = bounds
            t = rng.uniform(t_min, t_max)
            x = x + t * d
            break

        if step >= burn_in and (step - burn_in) % thin == 0:
            samples.append(x.copy())

    return np.asarray(samples)
