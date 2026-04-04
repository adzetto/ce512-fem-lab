"""
Bar wave propagation — explicit central difference validation.

Demonstrates:
  - Building a 1D bar mesh (truss elements)
  - Lumped mass matrix assembly
  - solve_central_diff() for explicit time integration
  - Wave front velocity verification: c = sqrt(E/rho)
  - Critical time step computation

System: Long bar of length L, fixed at left, impulse at right.
The compressive wave should travel at c = sqrt(E/rho) and arrive
at the fixed end at t = L/c.
"""

from __future__ import annotations

import numpy as np

from ..dynamics import constant_load, critical_timestep, solve_central_diff
from ..elements.bars import kebar, mebar


def dynamic_wave_data(
    L: float = 10.0,
    nel: int = 50,
    E: float = 100.0,
    A: float = 1.0,
    rho: float = 1.0,
):
    """Build a 1D bar mesh for wave propagation.

    Parameters
    ----------
    L : float
        Bar length.
    nel : int
        Number of elements.
    E : float
        Young's modulus.
    A : float
        Cross-section area.
    rho : float
        Density.

    Returns
    -------
    dict
        Keys: ``T``, ``X``, ``G``, ``nn``, ``dof``, ``nel``, ``L``,
        ``E``, ``A``, ``rho``, ``c_wave``, ``t_arrival``.
    """
    nn = nel + 1
    dof = 1  # 1D bar
    X = np.zeros((nn, 1), dtype=float)
    for i in range(nn):
        X[i, 0] = i * L / nel

    T = np.zeros((nel, 3), dtype=float)
    for e in range(nel):
        T[e, 0] = e + 1
        T[e, 1] = e + 2
        T[e, 2] = 1  # material group

    G = np.array([[A, E, rho]])

    c_wave = np.sqrt(E / rho)
    t_arrival = L / c_wave

    return {
        "T": T,
        "X": X,
        "G": G,
        "nn": nn,
        "dof": dof,
        "nel": nel,
        "L": L,
        "E": E,
        "A": A,
        "rho": rho,
        "c_wave": c_wave,
        "t_arrival": t_arrival,
    }


def _assemble_bar_system(data):
    """Assemble K and lumped M for the 1D bar."""
    T = data["T"]
    X = data["X"]
    G = data["G"]
    nn = data["nn"]
    dof = data["dof"]
    ndof = nn * dof

    K = np.zeros((ndof, ndof), dtype=float)
    M = np.zeros((ndof, ndof), dtype=float)

    nel = data["nel"]
    for e in range(nel):
        nodes = T[e, :2].astype(int) - 1
        Xe = X[nodes]  # shape (2, 1)
        Ge = G[0]
        ke = kebar(Xe, Ge)
        me = mebar(Xe, Ge, lumped=True)

        # Scatter
        edof = nodes  # 1 DOF per node
        ix = np.ix_(edof, edof)
        K[ix] += ke
        M[ix] += me

    return K, M


def run_dynamic_wave(*, dt_factor: float = 0.8, plot: bool = False):
    """Simulate wave propagation in a bar and verify wave speed.

    Parameters
    ----------
    dt_factor : float
        Fraction of the critical time step to use (< 1 for stability).
    plot : bool
        If True, produce displacement waterfall plot.

    Returns
    -------
    dict
        Keys: ``result`` (TimeHistory), ``c_wave_exact``, ``c_wave_measured``,
        ``dt_cr``, ``dt_used``, ``data``, ``figures``.
    """
    data = dynamic_wave_data()
    K, M = _assemble_bar_system(data)
    nn = data["nn"]
    dof = data["dof"]
    ndof = nn * dof

    # Critical time step
    dt_cr = critical_timestep(K, M)
    dt = dt_factor * dt_cr

    # Apply impulse at the right end (last node)
    P0 = 1.0
    impulse_duration = 5.0 * dt  # short pulse

    def p_func(t):
        p = np.zeros((ndof, 1))
        if t <= impulse_duration:
            p[-1, 0] = P0
        return p

    # Fixed left end: node 0
    C_bc = np.array([[1, 1, 0.0]])

    u0 = np.zeros((ndof, 1))
    v0 = np.zeros((ndof, 1))

    # Run for enough time for the wave to travel across the bar
    t_arrival = data["t_arrival"]
    t_end = 2.0 * t_arrival
    nsteps = int(t_end / dt)

    M_lumped = np.diag(M).copy()
    C_damp = np.zeros((ndof, ndof))

    result = solve_central_diff(
        M_lumped,
        C_damp,
        K,
        p_func,
        u0,
        v0,
        dt,
        nsteps,
        C_bc=C_bc,
        dof=dof,
    )

    # Measure wave speed: find when the midpoint node first responds
    mid_node = nn // 2
    mid_disp = result.u[:, mid_node]
    threshold = (
        0.01 * np.max(np.abs(mid_disp)) if np.max(np.abs(mid_disp)) > 0 else 1e-15
    )
    arrival_idx = np.argmax(np.abs(mid_disp) > threshold)
    t_mid_arrival = result.t[arrival_idx] if arrival_idx > 0 else np.inf

    x_mid = data["X"][mid_node, 0]
    dist_from_right = data["L"] - x_mid
    c_wave_measured = dist_from_right / t_mid_arrival if t_mid_arrival > 0 else np.inf

    figures = []
    if plot:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Waterfall plot of displacement at selected times
        ax1 = axes[0]
        x_coords = data["X"][:, 0]
        n_snapshots = 8
        step_interval = max(1, nsteps // n_snapshots)
        for i in range(0, nsteps + 1, step_interval):
            t_snap = result.t[i]
            u_snap = result.u[i, :]
            ax1.plot(
                x_coords,
                u_snap + 0.05 * (i // step_interval),
                label=f"t = {t_snap:.3f}",
            )
        ax1.set_xlabel("x")
        ax1.set_ylabel("Displacement (offset)")
        ax1.set_title("Wave Propagation Snapshots")
        ax1.legend(fontsize=7)
        ax1.grid(True, alpha=0.3)

        # Time history at midpoint
        ax2 = axes[1]
        ax2.plot(result.t, result.u[:, mid_node], "b-", linewidth=1.0)
        ax2.axvline(
            dist_from_right / data["c_wave"],
            color="r",
            ls="--",
            label=f"Expected arrival (t = {dist_from_right / data['c_wave']:.3f})",
        )
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Displacement")
        ax2.set_title(f"Midpoint Response (node {mid_node})")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        figures.append(fig)

    return {
        "result": result,
        "c_wave_exact": data["c_wave"],
        "c_wave_measured": c_wave_measured,
        "dt_cr": dt_cr,
        "dt_used": dt,
        "data": data,
        "figures": figures,
    }
