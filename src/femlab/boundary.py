from __future__ import annotations

import numpy as np

from ._helpers import (
    as_float_array,
    is_sparse,
    max_abs_diagonal,
    solve_linear_system,
)

try:
    import scipy.sparse as sp
except ImportError:  # pragma: no cover
    sp = None


def solve_lag_general(
    K,
    p,
    G,
    Q=None,
    *,
    scale: float | None = None,
    return_lagrange: bool = False,
):
    """Solve a linear system with general linear constraints ``G u = Q``.

    The augmented system is scaled to keep the constraint rows numerically
    compatible with the stiffness matrix, matching the legacy toolbox pattern.
    """

    constraint_matrix = as_float_array(G)
    if constraint_matrix.size == 0:
        solution = solve_linear_system(K, p)
        if return_lagrange:
            return solution, np.zeros((0, 1), dtype=float)
        return solution

    if constraint_matrix.ndim == 1:
        constraint_matrix = constraint_matrix.reshape(1, -1)

    system_size = K.shape[0]
    if constraint_matrix.shape[1] != system_size:
        raise ValueError(
            "Constraint matrix width must match the number of system DOFs."
        )

    if Q is None:
        constraint_rhs = np.zeros((constraint_matrix.shape[0], 1), dtype=float)
    else:
        constraint_rhs = as_float_array(Q).reshape(-1, 1)
        if constraint_rhs.shape[0] != constraint_matrix.shape[0]:
            raise ValueError(
                "Constraint RHS height must match the number of constraint rows."
            )

    if scale is None:
        scale = 1.0e-2 * max_abs_diagonal(K)
        if scale == 0.0:
            scale = 1.0

    Gbar = scale * constraint_matrix
    Qbar = scale * constraint_rhs

    if is_sparse(K) and sp is not None:
        Kbar = sp.bmat(
            [
                [K, sp.csr_matrix(Gbar.T)],
                [sp.csr_matrix(Gbar), None],
            ],
            format="csr",
        )
    else:
        Kbar = np.block(
            [
                [as_float_array(K), Gbar.T],
                [
                    Gbar,
                    np.zeros(
                        (constraint_matrix.shape[0], constraint_matrix.shape[0]),
                        dtype=float,
                    ),
                ],
            ]
        )

    pbar = np.vstack([as_float_array(p).reshape(-1, 1), Qbar])
    augmented = solve_linear_system(Kbar, pbar)
    solution = augmented[:system_size]

    if return_lagrange:
        lagrange = augmented[system_size:] * scale
        return solution, lagrange
    return solution


def setbc(K, p, C, dof: int = 1):
    constraints = as_float_array(C)
    if constraints.size == 0:
        return K, p, 0.0
    stiffness_scale = 1.0e6 * max_abs_diagonal(K)
    if stiffness_scale == 0.0:
        stiffness_scale = 1.0
    if dof == 1:
        indices = constraints[:, 0].astype(int) - 1
    else:
        indices = (constraints[:, 0].astype(int) - 1) * dof + constraints[
            :, 1
        ].astype(int) - 1
    K[indices, indices] = K[indices, indices] + stiffness_scale
    p[indices, 0] = p[indices, 0] + stiffness_scale * constraints[:, -1]
    return K, p, stiffness_scale


def solve_lag(K, p, C=None, dof: int = 1, *, return_lagrange: bool = False):
    if C is None:
        solution = solve_linear_system(K, p)
        if return_lagrange:
            return solution, np.zeros((0, 1), dtype=float)
        return solution
    constraints = as_float_array(C)
    if constraints.size == 0:
        solution = solve_linear_system(K, p)
        if return_lagrange:
            return solution, np.zeros((0, 1), dtype=float)
        return solution

    n_constraints = constraints.shape[0]
    system_size = K.shape[0]

    G = np.zeros((n_constraints, system_size), dtype=float)
    Q = constraints[:, -1].reshape(-1, 1)
    if dof == 1:
        indices = constraints[:, 0].astype(int) - 1
    else:
        indices = (constraints[:, 0].astype(int) - 1) * dof + constraints[
            :, 1
        ].astype(int) - 1
    G[np.arange(n_constraints), indices] = 1.0

    return solve_lag_general(K, p, G, Q, return_lagrange=return_lagrange)


def rnorm(f, C, dof: int):
    force = as_float_array(f).reshape(-1)
    constraints = as_float_array(C)
    fixed = np.zeros(force.shape[0], dtype=bool)
    indices = (constraints[:, 0].astype(int) - 1) * dof + constraints[:, 1].astype(
        int
    ) - 1
    fixed[indices] = True
    return float(np.linalg.norm(force[~fixed]))

__all__ = ["rnorm", "setbc", "solve_lag", "solve_lag_general"]
