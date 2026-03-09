from __future__ import annotations

import numpy as np

from ._helpers import (
    as_float_array,
    cols,
    is_sparse,
    max_abs_diagonal,
    node_dof_indices,
    solve_linear_system,
)

try:
    import scipy.sparse as sp
except ImportError:  # pragma: no cover
    sp = None


def setbc(K, p, C, dof: int = 1):
    constraints = as_float_array(C)
    if constraints.size == 0:
        return K, p, 0.0
    stiffness_scale = 1.0e6 * max_abs_diagonal(K)
    if stiffness_scale == 0.0:
        stiffness_scale = 1.0
    for row in constraints:
        if dof == 1:
            index = int(row[0]) - 1
        else:
            index = (int(row[0]) - 1) * dof + int(row[1]) - 1
        K[index, index] = K[index, index] + stiffness_scale
        p[index, 0] = p[index, 0] + stiffness_scale * row[-1]
    return K, p, stiffness_scale


def solve_lag(K, p, C=None, dof: int = 1):
    if C is None:
        return solve_linear_system(K, p)
    constraints = as_float_array(C)
    if constraints.size == 0:
        return solve_linear_system(K, p)

    n_constraints = constraints.shape[0]
    system_size = K.shape[0]
    stiffness_scale = 1.0e-2 * max_abs_diagonal(K)
    if stiffness_scale == 0.0:
        stiffness_scale = 1.0

    G = np.zeros((n_constraints, system_size), dtype=float)
    Q = (stiffness_scale * constraints[:, -1]).reshape(-1, 1)
    for i, row in enumerate(constraints):
        if dof == 1:
            index = int(row[0]) - 1
        else:
            index = (int(row[0]) - 1) * dof + int(row[1]) - 1
        G[i, index] = stiffness_scale

    if is_sparse(K) and sp is not None:
        Kbar = sp.bmat([[K, sp.csr_matrix(G.T)], [sp.csr_matrix(G), None]], format="csr")
    else:
        Kbar = np.block([[as_float_array(K), G.T], [G, np.zeros((n_constraints, n_constraints), dtype=float)]])
    pbar = np.vstack([as_float_array(p).reshape(-1, 1), Q])
    ubar = solve_linear_system(Kbar, pbar)
    return ubar[:system_size]


def rnorm(f, C, dof: int):
    force = as_float_array(f).reshape(-1)
    constraints = as_float_array(C)
    fixed = np.zeros(force.shape[0], dtype=bool)
    for row in constraints:
        index = (int(row[0]) - 1) * dof + int(row[1]) - 1
        fixed[index] = True
    return float(np.linalg.norm(force[~fixed]))


__all__ = ["rnorm", "setbc", "solve_lag"]
