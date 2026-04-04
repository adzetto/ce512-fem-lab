"""
Modal analysis — eigenvalue solver for natural frequencies and mode shapes.

Solves the generalized eigenvalue problem  K phi = omega^2 M phi
using SciPy's sparse eigensolvers (shift-invert for smallest modes).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ._helpers import as_float_array, is_sparse

try:
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla
except ImportError:  # pragma: no cover
    sp = None
    spla = None


@dataclass
class ModalResult:
    """
    Container for modal analysis results.

    Attributes
    ----------
    eigenvalues : ndarray, shape (n_modes,)
        Eigenvalues omega_i^2 (square of natural frequencies).
    omega : ndarray, shape (n_modes,)
        Natural frequencies in rad/s.
    freq_hz : ndarray, shape (n_modes,)
        Natural frequencies in Hz.
    period : ndarray, shape (n_modes,)
        Natural periods in seconds.
    mode_shapes : ndarray, shape (ndof, n_modes)
        Mode shape vectors (columns), mass-normalized.
    participation : ndarray, shape (n_modes, ndim) or None
        Modal participation factors per direction.
    effective_mass : ndarray, shape (n_modes, ndim) or None
        Effective modal mass per direction.
    """

    eigenvalues: np.ndarray
    omega: np.ndarray
    freq_hz: np.ndarray
    period: np.ndarray
    mode_shapes: np.ndarray
    participation: np.ndarray | None = field(default=None)
    effective_mass: np.ndarray | None = field(default=None)


def _get_free_dofs(C, dof, ndof):
    """Return a boolean mask of unconstrained (free) DOFs."""
    if C is None:
        return np.ones(ndof, dtype=bool)
    constraints = as_float_array(C)
    if constraints.size == 0:
        return np.ones(ndof, dtype=bool)
    free = np.ones(ndof, dtype=bool)
    if dof == 1:
        indices = constraints[:, 0].astype(int) - 1
    else:
        indices = (
            (constraints[:, 0].astype(int) - 1) * dof
            + constraints[:, 1].astype(int)
            - 1
        )
    free[indices.astype(int)] = False
    return free


def _reduce_system(K, M, free_mask):
    """Extract the free-DOF sub-system from K and M."""
    free_idx = np.where(free_mask)[0]
    if is_sparse(K):
        K_d = np.asarray(K.toarray(), dtype=float)
    else:
        K_d = as_float_array(K)
    if is_sparse(M):
        M_d = np.asarray(M.toarray(), dtype=float)
    else:
        M_d = as_float_array(M)
    K_red = K_d[np.ix_(free_idx, free_idx)]
    M_red = M_d[np.ix_(free_idx, free_idx)]
    return K_red, M_red, free_idx


def _modal_participation(M, phi, dof, ndof):
    """
    Compute modal participation factors and effective modal mass.

    Mathematical Formulation
    ------------------------
    For a base excitation in direction 'j', the influence vector r_j
    contains 1s at all DOFs acting in direction 'j' and 0s elsewhere.

    The participation factor Gamma_{n,j} for mode n in direction j is:
        Gamma_{n,j} = (phi_n^T * M * r_j) / (phi_n^T * M * phi_n)

    If the mode shapes `phi` are already mass-normalized, the denominator is 1.

    The effective modal mass m_{eff,n,j} is defined as:
        m_{eff,n,j} = Gamma_{n,j}^2 * (phi_n^T * M * phi_n) = Gamma_{n,j}^2

    Parameters
    ----------
    M : ndarray, shape (ndof, ndof)
        Global mass matrix.
    phi : ndarray, shape (ndof, n_modes)
        Mass-normalized mode shape matrix in the full DOF space.
    dof : int
        Number of DOFs per node.
    ndof : int
        Total number of DOFs in the system.

    Returns
    -------
    participation : ndarray, shape (n_modes, dof)
        Participation factor for each mode and spatial direction.
    effective_mass : ndarray, shape (n_modes, dof)
        Effective modal mass for each mode and spatial direction.

    """
    M_arr = as_float_array(M) if not is_sparse(M) else np.asarray(M.toarray())
    n_modes = phi.shape[1]
    nn = ndof // dof

    participation = np.zeros((n_modes, dof), dtype=float)
    effective_mass = np.zeros((n_modes, dof), dtype=float)

    for d in range(dof):
        # Unit influence vector for direction d
        r = np.zeros(ndof, dtype=float)
        r[d::dof] = 1.0
        for i in range(n_modes):
            phi_i = phi[:, i]
            gamma = phi_i @ M_arr @ r
            m_phi = phi_i @ M_arr @ phi_i
            participation[i, d] = gamma
            effective_mass[i, d] = gamma**2 / m_phi if m_phi > 0 else 0.0

    return participation, effective_mass


def solve_modal(
    K,
    M,
    n_modes: int = 10,
    *,
    C_bc=None,
    dof: int = 2,
    sigma: float = 0.0,
) -> ModalResult:
    """
    Compute the lowest natural frequencies and mode shapes.

    Mathematical Formulation
    ------------------------
    This function solves the generalized structural eigenvalue problem
    for undamped free vibration:

        K * phi = omega^2 * M * phi

    where:
    - K is the global stiffness matrix.
    - M is the global mass matrix.
    - omega^2 are the eigenvalues (square of circular natural frequencies).
    - phi are the eigenvectors (mode shapes).

    The modes are extracted after statically condensing out all constrained
    degrees of freedom defined in `C_bc`. The returned mode shapes `phi`
    are mass-normalized such that:
        phi^T * M * phi = I
    and expanded back to the full structural DOF size (zeros at fixed DOFs).

    Parameters
    ----------
    K : ndarray or scipy.sparse matrix, shape (ndof, ndof)
        Global stiffness matrix.
    M : ndarray or scipy.sparse matrix, shape (ndof, ndof)
        Global mass matrix.
    n_modes : int, default 10
        Number of lowest modes to compute.
    C_bc : array_like, optional
        Boundary condition constraint table (same format as ``setbc`` C argument).
        Constrained DOFs are eliminated before the eigensolve.
    dof : int, default 2
        Degrees of freedom per node. Determines directionality for effective mass.
    sigma : float, default 0.0
        Shift for the shift-and-invert solver (Lanczos). Use 0.0 to find the
        fundamental (lowest) modes.

    Returns
    -------
    ModalResult
        Dataclass containing eigenvalues, circular frequencies (rad/s),
        frequencies (Hz), periods (s), mass-normalized mode shapes (ndof, n_modes),
        participation factors, and effective modal mass vectors.

    Examples
    --------
    >>> from femlabpy import init, kq4e, solve_modal
    >>> result = solve_modal(K, M, n_modes=5, C_bc=C, dof=2)
    >>> print(result.freq_hz[:3])
    """
    ndof = K.shape[0]
    free_mask = _get_free_dofs(C_bc, dof, ndof)
    K_red, M_red, free_idx = _reduce_system(K, M, free_mask)
    n_free = K_red.shape[0]

    # Clamp n_modes to feasible range
    if n_free < 1:
        return ModalResult(
            eigenvalues=np.array([]),
            omega=np.array([]),
            freq_hz=np.array([]),
            period=np.array([]),
            mode_shapes=np.zeros((ndof, 0)),
        )

    # Use dense eigensolver for small systems, sparse for large
    if n_free <= 200:
        from scipy.linalg import eigh

        # Dense eigh returns all n_free eigenvalues; we select the lowest n_modes
        n_modes = min(n_modes, n_free)
        eigenvalues, eigenvectors = eigh(K_red, M_red)
        idx = np.argsort(eigenvalues)[:n_modes]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
    else:
        if spla is None:
            raise RuntimeError("scipy is required for sparse eigenvalue problems")
        # eigsh requires k < n; clamp to n_free - 1 (at least 1)
        n_modes = min(n_modes, max(n_free - 1, 1))
        K_sp = sp.csc_matrix(K_red) if not sp.issparse(K_red) else K_red.tocsc()
        M_sp = sp.csc_matrix(M_red) if not sp.issparse(M_red) else M_red.tocsc()
        eigenvalues, eigenvectors = spla.eigsh(
            K_sp, k=n_modes, M=M_sp, sigma=sigma, which="LM"
        )
        idx = np.argsort(eigenvalues)
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

    # Remove any negative eigenvalues (numerical noise for rigid body modes)
    eigenvalues = np.maximum(eigenvalues, 0.0)

    # Mass-normalize mode shapes
    for i in range(eigenvectors.shape[1]):
        phi_i = eigenvectors[:, i]
        m_norm = phi_i @ M_red @ phi_i
        if m_norm > 0:
            eigenvectors[:, i] = phi_i / np.sqrt(m_norm)

    # Expand to full DOF space
    mode_shapes_full = np.zeros((ndof, n_modes), dtype=float)
    mode_shapes_full[free_idx, :] = eigenvectors

    omega = np.sqrt(eigenvalues)
    freq_hz = omega / (2.0 * np.pi)
    period = np.where(freq_hz > 0, 1.0 / freq_hz, np.inf)

    # Compute participation factors using the full-size matrices/modes
    M_full = as_float_array(M) if not is_sparse(M) else np.asarray(M.toarray())
    participation, effective_mass = _modal_participation(
        M_full, mode_shapes_full, dof, ndof
    )

    return ModalResult(
        eigenvalues=eigenvalues,
        omega=omega,
        freq_hz=freq_hz,
        period=period,
        mode_shapes=mode_shapes_full,
        participation=participation,
        effective_mass=effective_mass,
    )


def plot_modes(T, X, phi, dof, mode_indices=None, *, scale=1.0):
    """
    Plot mode shapes as deformed meshes.

    Parameters
    ----------
    T : array_like
        Element topology table.
    X : array_like
        Nodal coordinates, shape (nn, ndim).
    phi : ndarray, shape (ndof, n_modes)
        Mode shape matrix.
    dof : int
        DOFs per node.
    mode_indices : list of int, optional
        Which mode indices to plot (0-based). Defaults to first 4.
    scale : float
        Displacement amplification factor.
    """
    import matplotlib.pyplot as plt

    phi = as_float_array(phi)
    X = as_float_array(X)
    T = as_float_array(T)

    if mode_indices is None:
        mode_indices = list(range(min(4, phi.shape[1])))

    n_plots = len(mode_indices)
    ncols = min(n_plots, 4)
    nrows = (n_plots + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    if n_plots == 1:
        axes = [axes]
    else:
        axes = np.asarray(axes).ravel()

    for ax_idx, mode_idx in enumerate(mode_indices):
        ax = axes[ax_idx]
        mode = phi[:, mode_idx].reshape(-1, dof)[:, : X.shape[1]]
        X_def = X + scale * mode

        # Plot undeformed
        nnodes_per_elem = T.shape[1] - 1
        for row in T:
            nodes = row[:nnodes_per_elem].astype(int) - 1
            poly_orig = np.vstack([X[nodes], X[nodes[0]]])
            poly_def = np.vstack([X_def[nodes], X_def[nodes[0]]])
            ax.plot(poly_orig[:, 0], poly_orig[:, 1], "k--", linewidth=0.5, alpha=0.4)
            ax.plot(poly_def[:, 0], poly_def[:, 1], "b-", linewidth=1.0)

        ax.set_aspect("equal")
        ax.set_title(f"Mode {mode_idx + 1}")

    for ax_idx in range(len(mode_indices), len(axes)):
        axes[ax_idx].set_visible(False)

    plt.tight_layout()
    return fig


__all__ = [
    "ModalResult",
    "plot_modes",
    "solve_modal",
]
