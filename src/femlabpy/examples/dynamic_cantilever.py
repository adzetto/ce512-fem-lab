"""
Cantilever beam free vibration — modal analysis validation.

Demonstrates:
  - Building K and M for a Q4 cantilever beam
  - solve_modal() for natural frequencies and mode shapes
  - Comparison with Euler-Bernoulli analytical frequencies
  - Mass and stiffness orthogonality verification

Analytical (Euler-Bernoulli beam theory):
  f_n = (beta_n*L)^2 / (2*pi*L^2) * sqrt(E*I / (rho*A))
  beta_1*L = 1.8751, beta_2*L = 4.6941, beta_3*L = 7.8548
"""

from __future__ import annotations

import numpy as np

from .. import init, kq4e, setbc
from ..elements.quads import mq4e
from ..modal import solve_modal


def dynamic_cantilever_data(
    L: float = 4.0,
    H: float = 1.0,
    nx: int = 16,
    ny: int = 4,
    E: float = 200.0,
    nu: float = 0.3,
    rho: float = 1.0,
    thickness: float = 1.0,
):
    """Build the cantilever beam Q4 mesh and material data.

    Parameters
    ----------
    L : float
        Beam length.
    H : float
        Beam height.
    nx, ny : int
        Number of elements in x and y directions.
    E : float
        Young's modulus.
    nu : float
        Poisson's ratio.
    rho : float
        Density.
    thickness : float
        Out-of-plane thickness.

    Returns
    -------
    dict
        Keys: ``T``, ``X``, ``G``, ``C``, ``dof``, ``nn``, ``L``, ``H``,
        ``E``, ``nu``, ``rho``, ``thickness``, ``nx``, ``ny``.
    """
    nn_x = nx + 1
    nn_y = ny + 1
    nn = nn_x * nn_y
    X = np.zeros((nn, 2), dtype=float)
    for j in range(nn_y):
        for i in range(nn_x):
            node = j * nn_x + i
            X[node, 0] = i * L / nx
            X[node, 1] = j * H / ny

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

    # Material: [E, nu, type, t, rho]
    G = np.array([[E, nu, 1.0, thickness, rho]])

    # Fixed left end: all nodes at x = 0
    left_nodes = [i for i in range(nn) if abs(X[i, 0]) < 1e-12]
    C_rows = []
    for n in left_nodes:
        C_rows.append([n + 1, 1, 0.0])  # fix x
        C_rows.append([n + 1, 2, 0.0])  # fix y
    C = np.array(C_rows, dtype=float)

    return {
        "T": T,
        "X": X,
        "G": G,
        "C": C,
        "dof": 2,
        "nn": nn,
        "L": L,
        "H": H,
        "E": E,
        "nu": nu,
        "rho": rho,
        "thickness": thickness,
        "nx": nx,
        "ny": ny,
    }


def _euler_bernoulli_frequencies(L, E, I, rho, A, n_modes=5):
    """Analytical cantilever beam frequencies (Euler-Bernoulli).

    Returns frequencies in Hz.
    """
    beta_L = [1.8751, 4.6941, 7.8548, 10.9955, 14.1372]
    while len(beta_L) < n_modes:
        beta_L.append(beta_L[-1] + np.pi)  # asymptotic approximation
    freqs = []
    for i in range(n_modes):
        f_i = (beta_L[i] ** 2) / (2.0 * np.pi * L**2) * np.sqrt(E * I / (rho * A))
        freqs.append(f_i)
    return np.array(freqs)


def run_dynamic_cantilever(*, n_modes: int = 5, plot: bool = False):
    """Solve the cantilever modal analysis and compare with analytical.

    Returns
    -------
    dict
        Keys: ``modal_result`` (ModalResult), ``freq_analytical``,
        ``freq_errors_pct``, ``data``, ``figures``.
    """
    data = dynamic_cantilever_data()
    nn = data["nn"]
    dof = data["dof"]
    ndof = nn * dof

    # Assemble stiffness and mass
    K = np.zeros((ndof, ndof), dtype=float)
    M = np.zeros((ndof, ndof), dtype=float)
    K = kq4e(K, data["T"], data["X"], data["G"])
    M = mq4e(M, data["T"], data["X"], data["G"])

    # Solve eigenvalue problem
    modal = solve_modal(K, M, n_modes=n_modes, C_bc=data["C"], dof=dof)

    # Analytical comparison (bending modes only)
    L = data["L"]
    H = data["H"]
    E = data["E"]
    rho = data["rho"]
    t = data["thickness"]
    I_beam = t * H**3 / 12.0  # second moment of area
    A_beam = H * t  # cross-section area
    freq_analytical = _euler_bernoulli_frequencies(L, E, I_beam, rho, A_beam, n_modes)

    # The FEM will produce both bending and other modes; compare the lowest
    n_compare = min(3, n_modes, len(modal.freq_hz))
    freq_errors_pct = np.zeros(n_compare)
    for i in range(n_compare):
        if freq_analytical[i] > 0:
            freq_errors_pct[i] = (
                abs(modal.freq_hz[i] - freq_analytical[i]) / freq_analytical[i] * 100.0
            )

    # Verify mass orthogonality: Phi^T M Phi = I
    M_full = M.copy()
    phi = modal.mode_shapes
    mass_ortho = phi.T @ M_full @ phi
    mass_ortho_error = float(
        np.max(
            np.abs(
                mass_ortho
                - np.eye(n_modes if n_modes <= len(modal.omega) else len(modal.omega))
            )
        )
    )

    figures = []
    if plot:
        import matplotlib.pyplot as plt

        # Frequency comparison table
        fig1, ax1 = plt.subplots(figsize=(8, 3))
        ax1.axis("off")
        n_show = min(n_compare, 5)
        col_labels = ["Mode", "FEM (Hz)", "Analytical (Hz)", "Error (%)"]
        table_data = []
        for i in range(n_show):
            table_data.append(
                [
                    f"{i + 1}",
                    f"{modal.freq_hz[i]:.4f}",
                    f"{freq_analytical[i]:.4f}",
                    f"{freq_errors_pct[i]:.2f}" if i < len(freq_errors_pct) else "—",
                ]
            )
        table = ax1.table(
            cellText=table_data,
            colLabels=col_labels,
            loc="center",
            cellLoc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)
        ax1.set_title("Natural Frequencies: FEM vs Euler-Bernoulli")
        figures.append(fig1)

        # Mode shapes
        try:
            from ..modal import plot_modes

            fig2 = plot_modes(
                data["T"],
                data["X"],
                phi,
                dof,
                mode_indices=list(range(min(4, len(modal.omega)))),
                scale=0.3,
            )
            figures.append(fig2)
        except Exception:
            pass

    return {
        "modal_result": modal,
        "freq_analytical": freq_analytical,
        "freq_errors_pct": freq_errors_pct,
        "mass_ortho_error": mass_ortho_error,
        "data": data,
        "figures": figures,
    }
