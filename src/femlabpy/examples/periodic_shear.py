"""
Homogeneous unit cell under macro shear — periodic BC validation.

Demonstrates:
  - Applying a macro shear strain eps_xy via periodic BCs
  - Verifying uniform shear stress throughout a homogeneous material
  - Volume-averaged stress matches G * gamma_xy exactly
  - Displacement field is affine (linear in coordinates)

System: Unit cell [0,1] x [0,1], plane stress, Q4 elements.
Applied macro strain: [exx=0, eyy=0, gamma_xy=0.01]
Expected: sigma_xy = G * gamma_xy = E / (2*(1+nu)) * 0.01
"""

from __future__ import annotations

import numpy as np

from ..elements.quads import kq4e
from ..periodic import (
    find_periodic_pairs,
    solve_periodic,
    volume_average_strain,
    volume_average_stress,
)


def _unit_square_mesh(nx=4, ny=4):
    """Create a structured Q4 mesh on [0,1] x [0,1]."""
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


def periodic_shear_data(
    nx: int = 6,
    ny: int = 6,
    E: float = 100.0,
    nu: float = 0.3,
    gamma_xy: float = 0.01,
):
    """Build the shear test input deck.

    Returns
    -------
    dict
        Keys: ``T``, ``X``, ``G``, ``K``, ``pairs``, ``eps_macro``,
        ``nn``, ``dof``, ``E``, ``nu``, ``gamma_xy``, ``sigma_xy_exact``.
    """
    T, X = _unit_square_mesh(nx, ny)
    nn = X.shape[0]
    dof = 2
    ndof = nn * dof

    G_mat = np.array([[E, nu, 1.0, 1.0]])  # plane stress, t=1

    K = np.zeros((ndof, ndof), dtype=float)
    K = kq4e(K, T, X, G_mat)

    pairs_x = find_periodic_pairs(X, axis=0)
    pairs_y = find_periodic_pairs(X, axis=1)
    all_pairs = np.vstack([pairs_x, pairs_y])
    seen = set()
    unique = []
    for row in all_pairs:
        key = (int(row[0]), int(row[1]))
        if key not in seen:
            seen.add(key)
            unique.append(row)
    pairs = np.array(unique)

    eps_macro = np.array([0.0, 0.0, gamma_xy])

    # Exact shear stress: sigma_xy = G * gamma_xy
    G_shear = E / (2.0 * (1.0 + nu))
    sigma_xy_exact = G_shear * gamma_xy

    return {
        "T": T,
        "X": X,
        "G": G_mat,
        "K": K,
        "pairs": pairs,
        "eps_macro": eps_macro,
        "nn": nn,
        "dof": dof,
        "E": E,
        "nu": nu,
        "gamma_xy": gamma_xy,
        "sigma_xy_exact": sigma_xy_exact,
    }


def run_periodic_shear(*, plot: bool = False):
    """Apply macro shear and verify uniform stress response.

    Returns
    -------
    dict
        Keys: ``u``, ``sigma_avg``, ``eps_avg``, ``sigma_xy_exact``,
        ``sigma_error``, ``eps_error``, ``data``, ``figures``.
    """
    data = periodic_shear_data()
    T, X, G_mat, K = data["T"], data["X"], data["G"], data["K"]
    pairs = data["pairs"]
    dof = data["dof"]
    nn = data["nn"]
    ndof = nn * dof
    eps_macro = data["eps_macro"]

    p = np.zeros((ndof, 1), dtype=float)
    u = solve_periodic(K, p, X, pairs, dof, eps_macro=eps_macro)

    sigma_avg = volume_average_stress(T, X, G_mat, u, dof, element_type="q4")
    eps_avg = volume_average_strain(T, X, G_mat, u, dof, element_type="q4")

    sigma_error = abs(sigma_avg[2] - data["sigma_xy_exact"])
    eps_error = abs(eps_avg[2] - data["gamma_xy"])

    figures = []
    if plot:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Deformed mesh
        ax1 = axes[0]
        U = u.reshape(-1, 2)
        scale = 50.0  # amplify for visibility
        nodes = data["T"][:, :4].astype(int) - 1
        for row in nodes:
            poly_orig = np.vstack([X[row], X[row[0]]])
            poly_def = np.vstack(
                [X[row] + scale * U[row], X[row[0]] + scale * U[row[0]]]
            )
            ax1.plot(poly_orig[:, 0], poly_orig[:, 1], "k--", lw=0.5, alpha=0.4)
            ax1.plot(poly_def[:, 0], poly_def[:, 1], "b-", lw=1.0)
        ax1.set_aspect("equal")
        ax1.set_title(f"Deformed Mesh (scale={scale}x)")
        ax1.grid(True, alpha=0.3)

        # Stress bar chart
        ax2 = axes[1]
        labels = ["sigma_xx", "sigma_yy", "sigma_xy"]
        exact_vals = [0.0, 0.0, data["sigma_xy_exact"]]
        x_pos = np.arange(3)
        ax2.bar(
            x_pos - 0.15, sigma_avg[:3], 0.3, label="FEM (vol avg)", color="steelblue"
        )
        ax2.bar(x_pos + 0.15, exact_vals, 0.3, label="Exact", color="coral")
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(labels)
        ax2.set_ylabel("Stress")
        ax2.set_title("Volume-Averaged Stress")
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        figures.append(fig)

    return {
        "u": u,
        "sigma_avg": sigma_avg,
        "eps_avg": eps_avg,
        "sigma_xy_exact": data["sigma_xy_exact"],
        "sigma_error": sigma_error,
        "eps_error": eps_error,
        "data": data,
        "figures": figures,
    }
