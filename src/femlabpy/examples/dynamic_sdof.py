"""
SDOF spring-mass-damper — Newmark validation against closed-form solution.

Demonstrates:
  - solve_newmark() for a damped forced-vibration SDOF system
  - Comparison with the exact analytical solution
  - Energy balance verification
  - Convergence rate measurement (2nd-order accuracy)

System: m*u'' + c*u' + k*u = P0*sin(omega_f*t)

Analytical steady-state:
  u_p(t) = U * sin(omega_f*t - phi)
  U = P0 / sqrt((k - m*omega_f^2)^2 + (c*omega_f)^2)
  phi = atan2(c*omega_f, k - m*omega_f^2)

Plus transient:
  u_h(t) = exp(-zeta*omega_n*t) * [A*cos(omega_d*t) + B*sin(omega_d*t)]
"""

from __future__ import annotations

import numpy as np

from ..damping import rayleigh_coefficients, rayleigh_damping
from ..dynamics import (
    TimeHistory,
    constant_load,
    harmonic_load,
    solve_newmark,
)


def dynamic_sdof_data(
    m: float = 1.0,
    k: float = 100.0,
    c: float = 2.0,
    P0: float = 10.0,
    omega_f: float = 5.0,
):
    """Return the SDOF spring-mass-damper input deck.

    Parameters
    ----------
    m : float
        Mass.
    k : float
        Spring stiffness.
    c : float
        Viscous damping coefficient.
    P0 : float
        Force amplitude.
    omega_f : float
        Forcing circular frequency (rad/s).

    Returns
    -------
    dict
        Keys: ``M``, ``C``, ``K``, ``m``, ``k``, ``c``, ``P0``, ``omega_f``,
        ``omega_n``, ``zeta``, ``omega_d``.
    """
    M = np.array([[m]])
    K = np.array([[k]])
    C = np.array([[c]])

    omega_n = np.sqrt(k / m)
    zeta = c / (2.0 * m * omega_n)
    omega_d = omega_n * np.sqrt(1.0 - zeta**2) if zeta < 1.0 else 0.0

    return {
        "M": M,
        "C": C,
        "K": K,
        "m": m,
        "k": k,
        "c": c,
        "P0": P0,
        "omega_f": omega_f,
        "omega_n": omega_n,
        "zeta": zeta,
        "omega_d": omega_d,
    }


def _analytical_sdof(data, t):
    """Compute the full analytical solution (transient + steady-state).

    Assumes u(0) = 0, u'(0) = 0.
    """
    m = data["m"]
    k = data["k"]
    c = data["c"]
    P0 = data["P0"]
    omega_f = data["omega_f"]
    omega_n = data["omega_n"]
    zeta = data["zeta"]
    omega_d = data["omega_d"]

    # Steady-state amplitude and phase
    denom = np.sqrt((k - m * omega_f**2) ** 2 + (c * omega_f) ** 2)
    U = P0 / denom
    phi = np.arctan2(c * omega_f, k - m * omega_f**2)

    # Particular solution
    u_p = U * np.sin(omega_f * t - phi)
    v_p = U * omega_f * np.cos(omega_f * t - phi)

    # Initial conditions: u(0) = 0, v(0) = 0
    # u_h(0) = -u_p(0) => A = -u_p(0) = U*sin(phi)
    # v_h(0) = -v_p(0) => -zeta*omega_n*A + omega_d*B = -v_p(0)
    A = U * np.sin(phi)
    v_p0 = U * omega_f * np.cos(-phi)
    B = (-v_p0 + zeta * omega_n * A) / omega_d if omega_d > 0 else 0.0

    # Homogeneous (transient) solution
    decay = np.exp(-zeta * omega_n * t)
    u_h = decay * (A * np.cos(omega_d * t) + B * np.sin(omega_d * t))

    return u_p + u_h


def run_dynamic_sdof(*, dt: float = 0.01, nsteps: int = 2000, plot: bool = False):
    """Solve the SDOF problem and compare with the analytical solution.

    Returns
    -------
    dict
        Keys: ``result`` (TimeHistory), ``u_exact``, ``max_error``,
        ``data``, ``figures``.
    """
    data = dynamic_sdof_data()
    M, C, K = data["M"], data["C"], data["K"]

    P = np.array([data["P0"]])
    p_func = harmonic_load(P, data["omega_f"])

    u0 = np.array([[0.0]])
    v0 = np.array([[0.0]])

    result = solve_newmark(
        M,
        C,
        K,
        p_func,
        u0,
        v0,
        dt,
        nsteps,
        compute_energy=True,
    )

    u_exact = _analytical_sdof(data, result.t)
    max_error = float(np.max(np.abs(result.u[:, 0] - u_exact)))

    figures = []
    if plot:
        import matplotlib.pyplot as plt

        # Displacement comparison
        fig1, ax1 = plt.subplots(figsize=(10, 4))
        ax1.plot(result.t, result.u[:, 0], "b-", label="Newmark", linewidth=1.5)
        ax1.plot(result.t, u_exact, "r--", label="Analytical", linewidth=1.0)
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Displacement")
        ax1.set_title(f"SDOF Forced Vibration (max error = {max_error:.2e})")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        figures.append(fig1)

        # Energy plot
        if result.energy is not None:
            fig2, ax2 = plt.subplots(figsize=(10, 4))
            ax2.plot(result.t, result.energy["kinetic"], "b-", label="Kinetic")
            ax2.plot(result.t, result.energy["strain"], "r-", label="Strain")
            ax2.plot(result.t, result.energy["total"], "k-", lw=2, label="Total")
            ax2.set_xlabel("Time (s)")
            ax2.set_ylabel("Energy")
            ax2.set_title("Energy Balance")
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            figures.append(fig2)

    return {
        "result": result,
        "u_exact": u_exact,
        "max_error": max_error,
        "data": data,
        "figures": figures,
    }


def run_convergence_study(*, plot: bool = False):
    """Demonstrate second-order accuracy by varying dt.

    Returns
    -------
    dict
        Keys: ``dt_values``, ``errors``, ``rates``, ``figures``.
    """
    data = dynamic_sdof_data()
    M, C, K = data["M"], data["C"], data["K"]
    P = np.array([data["P0"]])
    p_func = harmonic_load(P, data["omega_f"])
    u0 = np.array([[0.0]])
    v0 = np.array([[0.0]])
    T_end = 2.0  # fixed end time

    dt_values = [0.1, 0.05, 0.025, 0.0125, 0.00625]
    errors = []

    for dt in dt_values:
        nsteps = int(T_end / dt)
        result = solve_newmark(M, C, K, p_func, u0, v0, dt, nsteps)
        u_exact = _analytical_sdof(data, result.t)
        err = float(np.max(np.abs(result.u[:, 0] - u_exact)))
        errors.append(err)

    rates = [
        np.log(errors[i] / errors[i + 1]) / np.log(dt_values[i] / dt_values[i + 1])
        for i in range(len(errors) - 1)
    ]

    figures = []
    if plot:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.loglog(dt_values, errors, "bo-", linewidth=1.5, markersize=6)
        # Reference slope
        ref = [errors[0] * (d / dt_values[0]) ** 2 for d in dt_values]
        ax.loglog(dt_values, ref, "r--", label="O(dt^2) reference")
        ax.set_xlabel("dt")
        ax.set_ylabel("Max error")
        ax.set_title("Newmark Convergence (average acceleration)")
        ax.legend()
        ax.grid(True, alpha=0.3, which="both")
        figures.append(fig)

    return {
        "dt_values": dt_values,
        "errors": errors,
        "rates": rates,
        "figures": figures,
    }
