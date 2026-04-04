"""
2D RVE homogenization — periodic boundary conditions example.

Demonstrates:
  - Building a structured Q4 mesh on a unit square
  - Identifying periodic node pairs on opposite faces
  - Applying canonical unit strains via periodic BCs
  - Computing volume-averaged stress to extract effective stiffness C_eff
  - Verifying symmetry and positive definiteness of C_eff
  - For a homogeneous material, C_eff should match the constitutive matrix exactly

System: Unit cell [0,1] x [0,1], plane stress, Q4 elements.
"""

from __future__ import annotations

import numpy as np

from ..elements.quads import kq4e
from ..periodic import (
    check_periodic_mesh,
    find_periodic_pairs,
    fix_corner,
    homogenize,
    solve_periodic,
    volume_average_strain,
    volume_average_stress,
)


def _unit_square_mesh(nx=4, ny=4):
    """Create a structured Q4 mesh on [0,1] x [0,1].

    Returns
    -------
    T : ndarray, shape (nel, 5)
        Topology (1-based nodes + material group).
    X : ndarray, shape (nn, 2)
        Nodal coordinates.
    """
    nn_x = nx + 1
    nn_y = ny + 1
    nn = nn_x * nn_y
    X = np.zeros((nn, 2), dtype=float)
    for j in range(nn_y):
        for i in range(nn_x):
            node = j * nn_x + i
            X[node, 0] = i / nx
            X[node, 1] = j / ny

    nel = nx * ny
    T = np.zeros((nel, 5), dtype=float)
    for j in range(ny):
        for i in range(nx):
            e = j * nx + i
            n0 = j * nn_x + i
            T[e, 0] = n0 + 1
            T[e, 1] = n0 + 2
            T[e, 2] = n0 + nn_x + 2
            T[e, 3] = n0 + nn_x + 1
            T[e, 4] = 1
    return T, X


def _plane_stress_D(E, nu):
    """Analytical plane-stress constitutive matrix."""
    c = E / (1.0 - nu**2)
    return c * np.array(
        [
            [1.0, nu, 0.0],
            [nu, 1.0, 0.0],
            [0.0, 0.0, (1.0 - nu) / 2.0],
        ]
    )


def periodic_rve_data(
    nx: int = 6,
    ny: int = 6,
    E: float = 100.0,
    nu: float = 0.3,
    thickness: float = 1.0,
):
    """Build an RVE (unit-cell) input deck.

    Parameters
    ----------
    nx, ny : int
        Elements in x and y directions.
    E : float
        Young's modulus.
    nu : float
        Poisson's ratio.
    thickness : float
        Out-of-plane thickness.

    Returns
    -------
    dict
        Keys: ``T``, ``X``, ``G``, ``K``, ``pairs_x``, ``pairs_y``,
        ``pairs``, ``nn``, ``dof``, ``E``, ``nu``, ``D_exact``.
    """
    T, X = _unit_square_mesh(nx, ny)
    nn = X.shape[0]
    dof = 2
    ndof = nn * dof

    G = np.array([[E, nu, 1.0, thickness]])

    K = np.zeros((ndof, ndof), dtype=float)
    K = kq4e(K, T, X, G)

    pairs_x = find_periodic_pairs(X, axis=0)
    pairs_y = find_periodic_pairs(X, axis=1)

    # Combine and deduplicate
    all_pairs = np.vstack([pairs_x, pairs_y])
    seen = set()
    unique = []
    for row in all_pairs:
        key = (int(row[0]), int(row[1]))
        if key not in seen:
            seen.add(key)
            unique.append(row)
    pairs = np.array(unique)

    D_exact = _plane_stress_D(E, nu)

    return {
        "T": T,
        "X": X,
        "G": G,
        "K": K,
        "pairs_x": pairs_x,
        "pairs_y": pairs_y,
        "pairs": pairs,
        "nn": nn,
        "dof": dof,
        "E": E,
        "nu": nu,
        "D_exact": D_exact,
        "thickness": thickness,
    }


def run_periodic_rve(*, plot: bool = False):
    """Homogenize a homogeneous unit cell and verify against exact C.

    Returns
    -------
    dict
        Keys: ``C_eff``, ``D_exact``, ``max_error``, ``is_symmetric``,
        ``is_positive_definite``, ``mesh_report``, ``data``, ``figures``.
    """
    data = periodic_rve_data()
    T, X, G, K = data["T"], data["X"], data["G"], data["K"]
    pairs = data["pairs"]
    dof = data["dof"]
    D_exact = data["D_exact"]

    # Mesh validation
    report_x = check_periodic_mesh(X, axis=0)
    report_y = check_periodic_mesh(X, axis=1)

    # Homogenize
    C_eff = homogenize(K, T, X, G, pairs, dof, element_type="q4")

    # Verify properties
    max_error = float(np.max(np.abs(C_eff - D_exact)))
    is_symmetric = float(np.max(np.abs(C_eff - C_eff.T))) < 1e-10
    eigvals = np.linalg.eigvalsh(C_eff)
    is_positive_definite = bool(np.all(eigvals > 0))

    figures = []
    if plot:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # C_eff as heatmap
        ax1 = axes[0]
        im = ax1.imshow(C_eff, cmap="RdBu_r", aspect="equal")
        ax1.set_title("Effective Stiffness C_eff")
        ax1.set_xticks(range(3))
        ax1.set_xticklabels(["xx", "yy", "xy"])
        ax1.set_yticks(range(3))
        ax1.set_yticklabels(["xx", "yy", "xy"])
        for i in range(3):
            for j in range(3):
                ax1.text(
                    j, i, f"{C_eff[i, j]:.2f}", ha="center", va="center", fontsize=9
                )
        plt.colorbar(im, ax=ax1, shrink=0.8)

        # Error comparison
        ax2 = axes[1]
        err_matrix = C_eff - D_exact
        im2 = ax2.imshow(err_matrix, cmap="RdBu_r", aspect="equal")
        ax2.set_title(f"Error (C_eff - D_exact), max = {max_error:.2e}")
        ax2.set_xticks(range(3))
        ax2.set_xticklabels(["xx", "yy", "xy"])
        ax2.set_yticks(range(3))
        ax2.set_yticklabels(["xx", "yy", "xy"])
        for i in range(3):
            for j in range(3):
                ax2.text(
                    j,
                    i,
                    f"{err_matrix[i, j]:.2e}",
                    ha="center",
                    va="center",
                    fontsize=8,
                )
        plt.colorbar(im2, ax=ax2, shrink=0.8)

        plt.tight_layout()
        figures.append(fig)

    return {
        "C_eff": C_eff,
        "D_exact": D_exact,
        "max_error": max_error,
        "is_symmetric": is_symmetric,
        "is_positive_definite": is_positive_definite,
        "mesh_report": {"x": report_x, "y": report_y},
        "data": data,
        "figures": figures,
    }
