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
    Compute Rayleigh damping mass and stiffness proportional multipliers.

    Rayleigh damping constructs the damping matrix as a linear combination
    of the mass and stiffness matrices:
        C = alpha * M + beta * K

    This function computes the coefficients `alpha` and `beta` such that
    the specified modal damping ratios (`zeta1`, `zeta2`) are achieved at
    the given circular natural frequencies (`omega1`, `omega2`).

    Mathematical Formulation
    ------------------------
    The modal damping ratio zeta_n for mode n with circular frequency omega_n is:
        zeta_n = (alpha / (2 * omega_n)) + (beta * omega_n / 2)

    Given two frequencies and desired damping ratios, this forms a 2x2 linear system.
    Solving this system yields the exact mass and stiffness multipliers needed to
    anchor the damping curve. Note that frequencies between omega1 and omega2 will
    have slightly less damping than the target, and frequencies outside this range
    will have significantly higher damping.

    Parameters
    ----------
    omega1, omega2 : float
        Two target circular natural frequencies (rad/s). If you have frequencies
        in Hz, multiply by 2*pi before passing them to this function.
    zeta1, zeta2 : float
        Desired critical damping ratios at those frequencies (e.g., 0.05 for 5%).

    Returns
    -------
    alpha : float
        Mass-proportional coefficient (s^-1).
    beta : float
        Stiffness-proportional coefficient (s).

    Examples
    --------
    >>> # Compute coefficients for 5% damping at 10 rad/s and 50 rad/s
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

    This formulation creates a classical damping matrix that preserves the
    normal modes of the undamped system. Because it is a linear combination
    of the mass and stiffness matrices, the resulting damping matrix `C`
    will automatically inherit their sparsity patterns, making it highly
    efficient for implicit time-history analysis using direct solvers.

    Mathematical Formulation
    ------------------------
    C = alpha * M + beta * K

    The damping ratio for mode n with circular frequency omega_n is:
    zeta_n = alpha / (2 * omega_n) + beta * omega_n / 2

    Parameters
    ----------
    M : ndarray or scipy.sparse matrix, shape (ndof, ndof)
        Global mass matrix of the structure.
    K : ndarray or scipy.sparse matrix, shape (ndof, ndof)
        Global stiffness matrix of the structure.
    alpha : float
        Mass-proportional damping multiplier (s^-1). Controls damping at
        low frequencies.
    beta : float
        Stiffness-proportional damping multiplier (s). Controls numerical
        damping at high frequencies.

    Returns
    -------
    C : ndarray or scipy.sparse.lil_matrix, shape (ndof, ndof)
        The explicit global damping matrix. If either `M` or `K` is sparse,
        the returned `C` will be a sparse LIL matrix suitable for assembly
        or conversion to CSR/CSC formats.

    Examples
    --------
    >>> alpha, beta = rayleigh_coefficients(10.0, 50.0, 0.05, 0.05)
    >>> C = rayleigh_damping(M, K, alpha, beta)
    """
    if is_sparse(M) or is_sparse(K):
        M_s = M.tocsr() if is_sparse(M) else sp.csr_matrix(M)
        K_s = K.tocsr() if is_sparse(K) else sp.csr_matrix(K)
        return (alpha * M_s + beta * K_s).tolil()
    return alpha * as_float_array(M) + beta * as_float_array(K)


def modal_damping(M, omega, phi, zeta):
    """
    Construct a dense Caughey damping matrix from explicit modal damping ratios.

    Unlike Rayleigh damping which only fits two points, modal damping allows
    you to specify the exact critical damping ratio for an arbitrary number of
    individual modes. The resulting matrix will preserve the classical normal
    modes of the system.

    Mathematical Formulation
    ------------------------
    Using the orthogonality of mass-normalized mode shapes (Phi^T M Phi = I),
    the modal damping matrix can be constructed directly in physical space as:

        C = M * Phi * diag(2 * zeta_i * omega_i) * Phi^T * M

    Note: This explicitly constructs a fully dense matrix, which is highly
    inefficient for large FEM models. It is typically used only for reduced
    systems or theoretical validation on small degrees-of-freedom problems.
    For large systems, use `rayleigh_damping` instead.

    Parameters
    ----------
    M : ndarray, shape (ndof, ndof)
        Global mass matrix (dense).
    omega : array_like, shape (n_modes,)
        Natural circular frequencies (rad/s) for each mode.
    phi : ndarray, shape (ndof, n_modes)
        Mass-normalized mode shape matrix (eigenvectors).
        Must satisfy phi^T @ M @ phi = I.
    zeta : array_like, shape (n_modes,)
        Target critical damping ratios for each individual mode.

    Returns
    -------
    C : ndarray, shape (ndof, ndof)
        The explicit dense damping matrix.
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
