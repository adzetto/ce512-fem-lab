from __future__ import annotations

import numpy as np
import pytest

from femlabpy import addload, rnorm, setbc, setload, solve_lag, solve_lag_general


@pytest.mark.parametrize("dof", [1, 2, 3, 4])
def test_setload_assigns_components(dof):
    p = np.zeros((12, 1), dtype=float)
    loads = np.column_stack(
        [
            np.array([1, 3], dtype=float),
            np.arange(1, 2 * dof + 1, dtype=float).reshape(2, dof),
        ]
    )
    updated = setload(p.copy(), loads)
    indices = ((loads[:, [0]].astype(int) - 1) * dof + np.arange(dof)).reshape(-1)
    expected = np.zeros_like(p)
    expected[indices, 0] = loads[:, 1:].reshape(-1)
    assert np.allclose(updated, expected)


@pytest.mark.parametrize("dof", [1, 2, 3, 4])
def test_addload_accumulates_components(dof):
    p = np.full((12, 1), 0.5, dtype=float)
    loads = np.column_stack(
        [
            np.array([2, 3], dtype=float),
            np.arange(1, 2 * dof + 1, dtype=float).reshape(2, dof),
        ]
    )
    updated = addload(p.copy(), loads)
    expected = p.copy()
    indices = ((loads[:, [0]].astype(int) - 1) * dof + np.arange(dof)).reshape(-1)
    np.add.at(expected[:, 0], indices, loads[:, 1:].reshape(-1))
    assert np.allclose(updated, expected)


@pytest.mark.parametrize(
    ("dof", "constraints", "expected_indices", "expected_values"),
    [
        (1, np.array([[1, 10.0], [3, -5.0]]), np.array([0, 2]), np.array([10.0, -5.0])),
        (
            2,
            np.array([[1, 1, 10.0], [2, 2, -5.0]]),
            np.array([0, 3]),
            np.array([10.0, -5.0]),
        ),
        (
            3,
            np.array([[1, 2, 7.0], [2, 3, -4.0]]),
            np.array([1, 5]),
            np.array([7.0, -4.0]),
        ),
    ],
)
def test_setbc_constrains_specified_dofs(dof, constraints, expected_indices, expected_values):
    K = np.eye(6, dtype=float)
    p = np.zeros((6, 1), dtype=float)
    updated_K, updated_p, ks = setbc(K.copy(), p.copy(), constraints, dof=dof)
    assert ks > 0.0
    # Direct elimination: diagonal is ks (row/column zeroed)
    assert np.allclose(updated_K[expected_indices, expected_indices], ks)
    # Off-diagonal entries in constrained rows/columns must be zero
    for idx in expected_indices:
        row_copy = np.array(updated_K[idx, :]).ravel().copy()
        row_copy[idx] = 0.0
        assert np.allclose(row_copy, 0.0)
    assert np.allclose(updated_p[expected_indices, 0], ks * expected_values)


@pytest.mark.parametrize(
    ("K", "p", "G", "Q"),
    [
        (
            np.diag([2.0, 3.0]),
            np.array([[2.0], [6.0]]),
            np.array([[1.0, 0.0]]),
            np.array([[0.0]]),
        ),
        (
            np.diag([4.0, 5.0]),
            np.array([[8.0], [10.0]]),
            np.array([[0.0, 1.0]]),
            np.array([[1.0]]),
        ),
        (
            np.diag([3.0, 7.0, 11.0]),
            np.array([[3.0], [14.0], [22.0]]),
            np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
            np.array([[0.0], [1.0]]),
        ),
        (
            np.array([[5.0, 1.0], [1.0, 4.0]]),
            np.array([[1.0], [2.0]]),
            np.array([[1.0, -1.0]]),
            np.array([[0.0]]),
        ),
        (
            np.array([[6.0, 2.0, 0.0], [2.0, 5.0, 1.0], [0.0, 1.0, 4.0]]),
            np.array([[3.0], [1.0], [2.0]]),
            np.array([[1.0, 0.0, -1.0]]),
            np.array([[0.5]]),
        ),
        (
            np.diag([10.0, 20.0, 30.0]),
            np.array([[5.0], [10.0], [15.0]]),
            np.array([[1.0, 1.0, 0.0], [0.0, 1.0, 1.0]]),
            np.array([[1.0], [2.0]]),
        ),
    ],
)
def test_solve_lag_general_satisfies_constraints(K, p, G, Q):
    solution, lagrange = solve_lag_general(K, p, G, Q, return_lagrange=True)
    augmented = np.block([[K, G.T], [G, np.zeros((G.shape[0], G.shape[0]))]])
    rhs = np.vstack([p, Q])
    recovered = np.vstack([solution, lagrange])
    assert np.allclose(G @ solution, Q, atol=1.0e-10)
    assert np.allclose(augmented @ recovered, rhs, atol=1.0e-8)


@pytest.mark.parametrize(
    ("K", "p", "C", "dof"),
    [
        (np.diag([2.0, 3.0]), np.array([[2.0], [6.0]]), np.array([[1.0, 0.0]]), 1),
        (np.diag([4.0, 5.0]), np.array([[8.0], [10.0]]), np.array([[2.0, 1.0]]), 1),
        (
            np.diag([3.0, 7.0, 11.0, 13.0]),
            np.array([[3.0], [14.0], [22.0], [26.0]]),
            np.array([[1.0, 1.0, 0.0], [2.0, 2.0, 1.0]]),
            2,
        ),
        (
            np.array([[6.0, 2.0], [2.0, 5.0]]),
            np.array([[3.0], [1.0]]),
            np.array([[1.0, 0.5]]),
            1,
        ),
        (
            np.diag([10.0, 20.0, 30.0, 40.0]),
            np.array([[5.0], [10.0], [15.0], [20.0]]),
            np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 0.0]]),
            2,
        ),
        (
            np.diag([8.0, 9.0, 10.0]),
            np.array([[4.0], [9.0], [10.0]]),
            np.array([[3.0, -1.0]]),
            1,
        ),
    ],
)
def test_solve_lag_matches_explicit_general_system(K, p, C, dof):
    solution = solve_lag(K, p, C, dof=dof)
    if dof == 1:
        indices = C[:, 0].astype(int) - 1
    else:
        indices = (C[:, 0].astype(int) - 1) * dof + C[:, 1].astype(int) - 1
    G = np.zeros((C.shape[0], K.shape[0]), dtype=float)
    G[np.arange(C.shape[0]), indices] = 1.0
    Q = C[:, -1].reshape(-1, 1)
    expected = solve_lag_general(K, p, G, Q)
    assert np.allclose(solution, expected)


@pytest.mark.parametrize(
    ("force", "constraints", "dof", "expected"),
    [
        (np.array([[3.0], [4.0], [0.0], [0.0]]), np.array([[1.0, 1.0, 0.0]]), 2, 4.0),
        (
            np.array([[3.0], [4.0], [12.0], [5.0]]),
            np.array([[1.0, 1.0, 0.0], [2.0, 2.0, 0.0]]),
            2,
            np.sqrt(4.0**2 + 12.0**2),
        ),
        (
            np.array([[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]]),
            np.array([[1.0, 2.0, 0.0], [2.0, 3.0, 0.0]]),
            3,
            np.linalg.norm([1.0, 3.0, 4.0, 5.0]),
        ),
    ],
)
def test_rnorm_masks_fixed_components(force, constraints, dof, expected):
    assert np.isclose(rnorm(force, constraints, dof), expected)
