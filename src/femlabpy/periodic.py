"""
Periodic boundary conditions for RVE/unit cell analysis.

Provides node pairing, constraint matrix construction, macro strain
application, volume averaging, and computational homogenization.
"""

from __future__ import annotations

import numpy as np

from ._helpers import (
    as_float_array,
    element_dof_indices,
    is_sparse,
    solve_linear_system,
)
from .boundary import solve_lag_general

try:
    from scipy.spatial import cKDTree
except ImportError:  # pragma: no cover
    cKDTree = None


# ---------------------------------------------------------------------------
# Node pair identification
# ---------------------------------------------------------------------------


def find_periodic_pairs(X, axis: int, tol: float = 1e-6):
    """
    Match nodes on opposite faces of a domain along a given axis.

    Parameters
    ----------
    X : array_like, shape (nn, ndim)
        Nodal coordinates.
    axis : int
        Axis index (0=x, 1=y, 2=z) along which to identify periodicity.
    tol : float
        Matching tolerance relative to domain size.

    Returns
    -------
    pairs : ndarray, shape (n_pairs, 2)
        Each row ``[left_node, right_node]`` with **1-based** node numbers.
    """
    X = as_float_array(X)
    ndim = X.shape[1]
    x_min = float(X[:, axis].min())
    x_max = float(X[:, axis].max())
    domain_size = x_max - x_min

    abs_tol = tol * domain_size if domain_size > 0 else tol

    left_mask = np.abs(X[:, axis] - x_min) < abs_tol
    right_mask = np.abs(X[:, axis] - x_max) < abs_tol

    left_nodes = np.where(left_mask)[0]
    right_nodes = np.where(right_mask)[0]

    if len(left_nodes) != len(right_nodes):
        raise ValueError(
            f"Periodic mesh mismatch along axis {axis}: "
            f"{len(left_nodes)} left nodes vs {len(right_nodes)} right nodes."
        )

    # Build coordinates for matching (excluding the periodic axis)
    other_axes = [i for i in range(ndim) if i != axis]
    if len(other_axes) == 0:
        # 1D case
        pairs = np.column_stack([left_nodes + 1, right_nodes + 1])
        return pairs

    left_coords = X[left_nodes][:, other_axes]
    right_coords = X[right_nodes][:, other_axes]

    # Match using KD-tree or brute force
    if cKDTree is not None and len(left_nodes) > 20:
        tree = cKDTree(right_coords)
        distances, indices = tree.query(left_coords)
        if np.any(distances > abs_tol):
            bad = np.sum(distances > abs_tol)
            raise ValueError(
                f"Could not match {bad} nodes on periodic boundaries. "
                f"Check mesh periodicity."
            )
        matched_right = right_nodes[indices]
    else:
        matched_right = np.zeros(len(left_nodes), dtype=int)
        for i, lc in enumerate(left_coords):
            dists = np.linalg.norm(right_coords - lc, axis=1)
            best = np.argmin(dists)
            if dists[best] > abs_tol:
                raise ValueError(
                    f"No matching right node for left node {left_nodes[i]} "
                    f"(min distance = {dists[best]:.6e})."
                )
            matched_right[i] = right_nodes[best]

    # Return 1-based node pairs
    pairs = np.column_stack([left_nodes + 1, matched_right + 1])
    return pairs


def find_all_periodic_pairs(X, periodic_axes, tol: float = 1e-6) -> dict:
    """
    Find periodic node pairs along multiple axes.

    Parameters
    ----------
    X : array_like, shape (nn, ndim)
        Nodal coordinates.
    periodic_axes : list of int
        Axes along which periodicity is enforced (e.g. [0, 1] for 2D).
    tol : float
        Matching tolerance.

    Returns
    -------
    dict
        ``{axis: pairs_array}`` for each periodic axis.
    """
    return {axis: find_periodic_pairs(X, axis, tol) for axis in periodic_axes}


# ---------------------------------------------------------------------------
# Constraint matrix construction
# ---------------------------------------------------------------------------


def periodic_constraints(X, pairs, dof: int, *, eps_macro=None):
    """
    Build the constraint matrix G and RHS Q for periodic BCs.

    Constraints: u_right - u_left = eps_macro * (x_right - x_left)

    Parameters
    ----------
    X : array_like, shape (nn, ndim)
        Nodal coordinates.
    pairs : ndarray, shape (n_pairs, 2)
        Node pairs (1-based): [[left, right], ...].
    dof : int
        DOFs per node.
    eps_macro : array_like, optional
        Macro strain tensor in Voigt notation: [exx, eyy, gxy] for 2D
        or [exx, eyy, ezz, gxy, gyz, gxz] for 3D.
        If None, purely periodic (zero fluctuation).

    Returns
    -------
    G : ndarray, shape (n_constraints, ndof_total)
        Constraint matrix.
    Q : ndarray, shape (n_constraints, 1)
        Constraint RHS.
    """
    X = as_float_array(X)
    pairs = np.asarray(pairs, dtype=int)
    n_pairs = pairs.shape[0]
    nn = X.shape[0]
    ndim = X.shape[1]
    ndof_total = nn * dof
    n_constraints = n_pairs * dof

    G = np.zeros((n_constraints, ndof_total), dtype=float)
    Q = np.zeros((n_constraints, 1), dtype=float)

    if eps_macro is not None:
        eps_tensor = _voigt_to_tensor(eps_macro, ndim)
    else:
        eps_tensor = np.zeros((ndim, ndim), dtype=float)

    for i, (left, right) in enumerate(pairs):
        left_0 = left - 1  # to 0-based
        right_0 = right - 1
        dx = X[right_0] - X[left_0]  # coordinate difference

        for d in range(dof):
            row = i * dof + d
            left_dof = left_0 * dof + d
            right_dof = right_0 * dof + d
            G[row, right_dof] = 1.0
            G[row, left_dof] = -1.0

            # RHS from macro strain
            if d < ndim:
                Q[row, 0] = float(eps_tensor[d, :] @ dx)

    return G, Q


# ---------------------------------------------------------------------------
# Macro strain application
# ---------------------------------------------------------------------------


def apply_macro_strain(X, pairs, eps_macro, dof: int):
    """
    Compute the RHS vector Q for periodic BCs with imposed macro strain.

    Parameters
    ----------
    X : array_like
        Nodal coordinates.
    pairs : ndarray
        Periodic node pairs (1-based).
    eps_macro : array_like
        Macro strain in Voigt form.
    dof : int
        DOFs per node.

    Returns
    -------
    Q : ndarray, shape (n_constraints, 1)
    """
    _, Q = periodic_constraints(X, pairs, dof, eps_macro=eps_macro)
    return Q


# ---------------------------------------------------------------------------
# Periodic solver
# ---------------------------------------------------------------------------


def solve_periodic(K, p, X, pairs, dof: int, *, eps_macro=None, return_lagrange=False):
    """
    Solve a periodic boundary value problem using Lagrange multipliers.

    Parameters
    ----------
    K : ndarray or sparse
        Global stiffness matrix.
    p : ndarray
        Load vector.
    X : array_like
        Nodal coordinates.
    pairs : ndarray
        Periodic node pairs (1-based).
    dof : int
        DOFs per node.
    eps_macro : array_like, optional
        Imposed macro strain in Voigt form.
    return_lagrange : bool
        If True, also return Lagrange multipliers.

    Returns
    -------
    u : ndarray, shape (ndof, 1)
        Displacement solution.
    lam : ndarray (only if return_lagrange=True)
        Lagrange multiplier values.
    """
    G, Q = periodic_constraints(X, pairs, dof, eps_macro=eps_macro)
    return solve_lag_general(K, p, G, Q, return_lagrange=return_lagrange)


# ---------------------------------------------------------------------------
# Volume averaging
# ---------------------------------------------------------------------------


def volume_average_stress(T, X, G_mat, u, dof: int, *, element_type: str = "q4"):
    """
    Compute volume-averaged stress over all elements.

    sigma_avg = (1/V_total) * sum(sigma_e * V_e)

    Parameters
    ----------
    T : array_like
        Element topology table.
    X : array_like
        Nodal coordinates.
    G_mat : array_like
        Material property table.
    u : array_like
        Displacement solution.
    dof : int
        DOFs per node.
    element_type : str
        Element type: ``'t3'``, ``'q4'``, ``'t4'``, ``'h8'``.

    Returns
    -------
    sigma_avg : ndarray
        Volume-averaged stress in Voigt notation.
    """
    from .elements.triangles import qt3e, _triangle_batch_geometry
    from .elements.quads import qq4e

    T = as_float_array(T)
    X = as_float_array(X)
    nn = X.shape[0]

    if element_type == "t3":
        q_dummy = np.zeros((nn * dof, 1), dtype=float)
        _, S, _ = qt3e(q_dummy, T, X, G_mat, u)
        nodes = T[:, :3].astype(int) - 1
        _, areas = _triangle_batch_geometry(X[nodes])
        total_vol = float(areas.sum())
        sigma_avg = (S * areas[:, None]).sum(axis=0) / total_vol
    elif element_type == "q4":
        q_dummy = np.zeros((nn * dof, 1), dtype=float)
        _, S, _ = qq4e(q_dummy, T, X, G_mat, u)
        # S has shape (nel, 12) -> reshape to (nel, 4, 3) -> average over GPs
        S_avg = S.reshape(-1, 4, 3).mean(axis=1)
        # Approximate element areas from node coords
        nodes = T[:, :4].astype(int) - 1
        nel = T.shape[0]
        areas = np.zeros(nel, dtype=float)
        for e in range(nel):
            xe = X[nodes[e]]
            # Shoelace formula for quadrilateral
            x_ = xe[:, 0]
            y_ = xe[:, 1]
            areas[e] = 0.5 * abs(
                (x_[0] * y_[1] - x_[1] * y_[0])
                + (x_[1] * y_[2] - x_[2] * y_[1])
                + (x_[2] * y_[3] - x_[3] * y_[2])
                + (x_[3] * y_[0] - x_[0] * y_[3])
            )
        total_vol = float(areas.sum())
        sigma_avg = (S_avg * areas[:, None]).sum(axis=0) / total_vol
    else:
        raise NotImplementedError(
            f"volume_average_stress not implemented for '{element_type}'"
        )

    return sigma_avg


def volume_average_strain(T, X, G_mat, u, dof: int, *, element_type: str = "q4"):
    """
    Compute volume-averaged strain over all elements.

    Parameters are the same as volume_average_stress.

    Returns
    -------
    eps_avg : ndarray
        Volume-averaged strain in Voigt notation.
    """
    from .elements.triangles import qt3e, _triangle_batch_geometry
    from .elements.quads import qq4e

    T = as_float_array(T)
    X = as_float_array(X)
    nn = X.shape[0]

    if element_type == "t3":
        q_dummy = np.zeros((nn * dof, 1), dtype=float)
        _, _, E = qt3e(q_dummy, T, X, G_mat, u)
        nodes = T[:, :3].astype(int) - 1
        _, areas = _triangle_batch_geometry(X[nodes])
        total_vol = float(areas.sum())
        eps_avg = (E * areas[:, None]).sum(axis=0) / total_vol
    elif element_type == "q4":
        q_dummy = np.zeros((nn * dof, 1), dtype=float)
        _, _, E = qq4e(q_dummy, T, X, G_mat, u)
        E_avg = E.reshape(-1, 4, 3).mean(axis=1)
        nodes = T[:, :4].astype(int) - 1
        nel = T.shape[0]
        areas = np.zeros(nel, dtype=float)
        for e in range(nel):
            xe = X[nodes[e]]
            x_ = xe[:, 0]
            y_ = xe[:, 1]
            areas[e] = 0.5 * abs(
                (x_[0] * y_[1] - x_[1] * y_[0])
                + (x_[1] * y_[2] - x_[2] * y_[1])
                + (x_[2] * y_[3] - x_[3] * y_[2])
                + (x_[3] * y_[0] - x_[0] * y_[3])
            )
        total_vol = float(areas.sum())
        eps_avg = (E_avg * areas[:, None]).sum(axis=0) / total_vol
    else:
        raise NotImplementedError(
            f"volume_average_strain not implemented for '{element_type}'"
        )

    return eps_avg


# ---------------------------------------------------------------------------
# Homogenization
# ---------------------------------------------------------------------------


def homogenize(K, T, X, G_mat, pairs, dof: int, *, element_type: str = "q4"):
    """
    Compute the effective (homogenized) stiffness tensor of an RVE.

    Applies canonical unit strains and extracts the effective constitutive
    matrix from volume-averaged stress-strain relationships.

    Parameters
    ----------
    K : ndarray or sparse
        Global stiffness matrix.
    T : array_like
        Element topology.
    X : array_like
        Nodal coordinates.
    G_mat : array_like
        Material property table.
    pairs : ndarray
        Periodic node pairs.
    dof : int
        DOFs per node.
    element_type : str
        ``'t3'`` or ``'q4'``.

    Returns
    -------
    C_eff : ndarray, shape (n_voigt, n_voigt)
        Effective stiffness matrix.
    """
    X = as_float_array(X)
    ndim = X.shape[1]

    if ndim == 2:
        n_voigt = 3
        # Unit strains: exx, eyy, gxy
        unit_strains = np.eye(n_voigt, dtype=float)
    elif ndim == 3:
        n_voigt = 6
        unit_strains = np.eye(n_voigt, dtype=float)
    else:
        raise ValueError(f"Unsupported dimension {ndim}")

    nn = X.shape[0]
    p_zero = np.zeros((nn * dof, 1), dtype=float)

    C_eff = np.zeros((n_voigt, n_voigt), dtype=float)
    for i in range(n_voigt):
        eps_macro = unit_strains[i]
        u_i = solve_periodic(K, p_zero, X, pairs, dof, eps_macro=eps_macro)
        sigma_avg = volume_average_stress(
            T, X, G_mat, u_i, dof, element_type=element_type
        )
        C_eff[:, i] = sigma_avg[:n_voigt]

    return C_eff


# ---------------------------------------------------------------------------
# Mesh validation
# ---------------------------------------------------------------------------


def fix_corner(X, C_existing, dof: int):
    """
    Add zero-displacement constraints at the corner node to remove rigid body modes.

    Parameters
    ----------
    X : array_like
        Nodal coordinates.
    C_existing : array_like or None
        Existing constraint table.
    dof : int
        DOFs per node.

    Returns
    -------
    C_extended : ndarray
        Extended constraint table.
    """
    X = as_float_array(X)
    ndim = X.shape[1]

    # Find the node closest to the minimum corner
    corner = X.min(axis=0)
    dists = np.linalg.norm(X - corner, axis=1)
    corner_node = int(np.argmin(dists)) + 1  # 1-based

    # Create constraints for all DOFs at the corner
    new_rows = []
    for d in range(min(dof, ndim)):
        new_rows.append([corner_node, d + 1, 0.0])
    new_constraints = np.array(new_rows, dtype=float)

    if C_existing is not None and as_float_array(C_existing).size > 0:
        C_existing = as_float_array(C_existing)
        return np.vstack([C_existing, new_constraints])
    return new_constraints


def check_periodic_mesh(X, axis: int, tol: float = 1e-6) -> dict:
    """
    Validate that a mesh is periodic along a given axis.

    Parameters
    ----------
    X : array_like
        Nodal coordinates.
    axis : int
        Axis to check.
    tol : float
        Tolerance.

    Returns
    -------
    report : dict
        Keys: ``'valid'``, ``'n_left'``, ``'n_right'``, ``'max_mismatch'``,
        ``'message'``.
    """
    X = as_float_array(X)
    x_min = float(X[:, axis].min())
    x_max = float(X[:, axis].max())
    domain_size = x_max - x_min
    abs_tol = tol * domain_size if domain_size > 0 else tol

    left_mask = np.abs(X[:, axis] - x_min) < abs_tol
    right_mask = np.abs(X[:, axis] - x_max) < abs_tol
    n_left = int(np.sum(left_mask))
    n_right = int(np.sum(right_mask))

    report = {
        "valid": True,
        "n_left": n_left,
        "n_right": n_right,
        "max_mismatch": 0.0,
        "message": "Mesh appears periodic.",
    }

    if n_left != n_right:
        report["valid"] = False
        report["message"] = (
            f"Node count mismatch: {n_left} left vs {n_right} right. "
            "Re-mesh with structured elements."
        )
        return report

    # Try to pair nodes
    try:
        pairs = find_periodic_pairs(X, axis, tol)
        report["message"] = f"Mesh is periodic: {len(pairs)} pairs found."
    except ValueError as e:
        report["valid"] = False
        report["message"] = str(e)

    return report


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _voigt_to_tensor(eps_voigt, ndim):
    """Convert Voigt strain to tensor form."""
    eps = as_float_array(eps_voigt).ravel()
    if ndim == 2:
        # [exx, eyy, gxy] -> [[exx, gxy/2], [gxy/2, eyy]]
        return np.array([[eps[0], eps[2] / 2.0], [eps[2] / 2.0, eps[1]]], dtype=float)
    elif ndim == 3:
        # [exx, eyy, ezz, gxy, gyz, gxz]
        return np.array(
            [
                [eps[0], eps[3] / 2.0, eps[5] / 2.0],
                [eps[3] / 2.0, eps[1], eps[4] / 2.0],
                [eps[5] / 2.0, eps[4] / 2.0, eps[2]],
            ],
            dtype=float,
        )
    raise ValueError(f"Unsupported ndim={ndim}")


__all__ = [
    "apply_macro_strain",
    "check_periodic_mesh",
    "find_all_periodic_pairs",
    "find_periodic_pairs",
    "fix_corner",
    "homogenize",
    "periodic_constraints",
    "solve_periodic",
    "volume_average_strain",
    "volume_average_stress",
]
