"""
Damping models for structural dynamics.

Provides Rayleigh (proportional) damping and modal damping construction.
"""

from __future__ import annotations

import numpy as np

from ._helpers import as_float_array, is_sparse

try:
    import scipy.sparse as sp
except ImportError:  # pragma: no cover
    sp = None


def rayleigh_coefficients(
    omega1: float, omega2: float, zeta1: float, zeta2: float
) -> tuple[float, float]:
    """
    Compute Rayleigh damping coefficients from two target frequencies.

    Solves the 2x2 system:
        zeta_i = alpha / (2 * omega_i) + beta * omega_i / 2

    Parameters
    ----------
    omega1, omega2 : float
        Two target natural frequencies (rad/s).
    zeta1, zeta2 : float
        Desired damping ratios at those frequencies.

    Returns
    -------
    alpha : float
        Mass-proportional coefficient.
    beta : float
        Stiffness-proportional coefficient.

    Examples
    --------
    >>> alpha, beta = rayleigh_coefficients(10.0, 50.0, 0.05, 0.05)
    """
    A = np.array(
        [
            [1.0 / (2.0 * omega1), omega1 / 2.0],
            [1.0 / (2.0 * omega2), omega2 / 2.0],
        ],
        dtype=float,
    )
    b = np.array([zeta1, zeta2], dtype=float)
    coeffs = np.linalg.solve(A, b)
    return float(coeffs[0]), float(coeffs[1])


def rayleigh_damping(M, K, alpha: float, beta: float):
    """
    Build the Rayleigh (proportional) damping matrix C = alpha*M + beta*K.

    Parameters
    ----------
    M : ndarray or sparse matrix
        Global mass matrix.
    K : ndarray or sparse matrix
        Global stiffness matrix.
    alpha : float
        Mass-proportional damping coefficient.
    beta : float
        Stiffness-proportional damping coefficient.

    Returns
    -------
    C : ndarray or sparse matrix
        Damping matrix with the same storage format as `K`.

    Examples
    --------
    >>> C = rayleigh_damping(M, K, alpha=0.5, beta=0.001)
    """
    if is_sparse(M) or is_sparse(K):
        M_s = M.tocsr() if is_sparse(M) else sp.csr_matrix(M)
        K_s = K.tocsr() if is_sparse(K) else sp.csr_matrix(K)
        return (alpha * M_s + beta * K_s).tolil()
    return alpha * as_float_array(M) + beta * as_float_array(K)


def modal_damping(M, omega, phi, zeta):
    """
    Construct a damping matrix from modal damping ratios.

    C = M * Phi * diag(2 * zeta_i * omega_i) * Phi^T * M

    where Phi columns are mass-normalized mode shapes.

    Parameters
    ----------
    M : ndarray
        Global mass matrix (dense).
    omega : array_like
        Natural frequencies (rad/s) for each mode.
    phi : ndarray, shape (ndof, n_modes)
        Mass-normalized mode shape matrix.
    zeta : array_like
        Damping ratios for each mode.

    Returns
    -------
    C : ndarray
        Damping matrix, shape (ndof, ndof).
    """
    omega = as_float_array(omega).ravel()
    zeta = as_float_array(zeta).ravel()
    phi = as_float_array(phi)
    M_arr = as_float_array(M)

    diag_vals = 2.0 * zeta * omega
    # C = M @ Phi @ diag(2*zeta*omega) @ Phi^T @ M
    M_phi = M_arr @ phi
    C = M_phi @ np.diag(diag_vals) @ M_phi.T
    return C


__all__ = [
    "modal_damping",
    "rayleigh_coefficients",
    "rayleigh_damping",
]
