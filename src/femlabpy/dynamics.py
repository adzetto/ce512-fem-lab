"""
Time integration solvers for structural dynamics.

Provides Newmark-beta (implicit), central difference (explicit),
HHT-alpha, and nonlinear Newmark solvers with load function builders.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from ._helpers import as_float_array, is_sparse, solve_linear_system

try:
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla
except ImportError:  # pragma: no cover
    sp = None
    spla = None


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class NewmarkParams:
    """
    Newmark-beta integration parameters.

    Attributes
    ----------
    beta : float
        Newmark beta parameter (default 0.25, average acceleration).
    gamma : float
        Newmark gamma parameter (default 0.5).
    """

    beta: float = 0.25
    gamma: float = 0.5

    @classmethod
    def average_acceleration(cls) -> NewmarkParams:
        """Unconditionally stable, no numerical dissipation, 2nd order."""
        return cls(0.25, 0.5)

    @classmethod
    def linear_acceleration(cls) -> NewmarkParams:
        """Conditionally stable (dt < 0.551*T_min), 2nd order."""
        return cls(1.0 / 6.0, 0.5)

    @classmethod
    def central_difference(cls) -> NewmarkParams:
        """Explicit, conditionally stable (dt < T_min/pi), 2nd order."""
        return cls(0.0, 0.5)

    @classmethod
    def fox_goodwin(cls) -> NewmarkParams:
        """Conditionally stable, 4th order for undamped SDOF."""
        return cls(1.0 / 12.0, 0.5)


@dataclass
class TimeHistory:
    """
    Container for time integration results.

    Attributes
    ----------
    t : ndarray, shape (nsteps+1,)
        Time values at each step.
    u : ndarray, shape (nsteps+1, ndof)
        Displacement history.
    v : ndarray, shape (nsteps+1, ndof)
        Velocity history.
    a : ndarray, shape (nsteps+1, ndof)
        Acceleration history.
    dt : float
        Time step size.
    nsteps : int
        Number of time steps.
    energy : dict or None
        Energy history: kinetic, strain, dissipated, external, total.
    """

    t: np.ndarray
    u: np.ndarray
    v: np.ndarray
    a: np.ndarray
    dt: float
    nsteps: int
    energy: dict | None = field(default=None)


# ---------------------------------------------------------------------------
# Load function builders
# ---------------------------------------------------------------------------


def constant_load(P) -> Callable:
    """Return a load function that is constant in time: p(t) = P."""
    P = as_float_array(P).reshape(-1, 1).copy()
    return lambda t: P


def ramp_load(P, t_ramp: float) -> Callable:
    """Linear ramp: p(t) = P * min(t / t_ramp, 1.0)."""
    P = as_float_array(P).reshape(-1, 1).copy()
    return lambda t: P * min(t / t_ramp, 1.0)


def harmonic_load(P, omega: float, phase: float = 0.0) -> Callable:
    """Sinusoidal: p(t) = P * sin(omega * t + phase)."""
    P = as_float_array(P).reshape(-1, 1).copy()
    return lambda t: P * np.sin(omega * t + phase)


def pulse_load(P, t_start: float, t_duration: float) -> Callable:
    """Rectangular pulse: P for t_start <= t <= t_start+t_duration, else 0."""
    P = as_float_array(P).reshape(-1, 1).copy()

    def _pulse(t):
        if t_start <= t <= t_start + t_duration:
            return P.copy()
        return np.zeros_like(P)

    return _pulse


def tabulated_load(P, time_table, value_table) -> Callable:
    """
    Interpolated load history: p(t) = P * interp(time_table, value_table)(t).

    Parameters
    ----------
    P : array_like
        Spatial load pattern (ndof, 1).
    time_table : array_like
        Time sample points.
    value_table : array_like
        Scalar multiplier values at each time sample.
    """
    P = as_float_array(P).reshape(-1, 1).copy()
    tt = as_float_array(time_table).ravel()
    vt = as_float_array(value_table).ravel()

    def _tabulated(t):
        val = float(np.interp(t, tt, vt, left=0.0, right=0.0))
        return P * val

    return _tabulated


def seismic_load(M, direction, accel_record, dt_record: float) -> Callable:
    """
    Construct a time-dependent effective force vector for uniform base excitation.

    Mathematical Formulation
    ------------------------
    When a structure undergoes uniform ground acceleration a_g(t), the equation
    of motion is usually solved for the relative displacement u_r(t):
        M * ü_r(t) + C * u̇_r(t) + K * u_r(t) = -M * r * a_g(t)

    where `r` is the influence vector mapping ground motion to structural DOFs.
    This function computes and returns a callable representing the effective
    load vector p_{eff}(t):
        p_{eff}(t) = -M * r * a_g(t)

    This callable dynamically interpolates the discrete ground acceleration record
    to exactly match the continuous time requests `t` during time integration.

    Parameters
    ----------
    M : ndarray or scipy.sparse matrix, shape (ndof, ndof)
        Global mass matrix of the structure.
    direction : array_like, shape (ndof,)
        Unit influence vector `r`. This vector has a value of `1.0` for DOFs
        that act parallel to the direction of ground excitation, and `0.0`
        for orthogonal or unexcited DOFs.
    accel_record : array_like, shape (N,)
        Discrete time series of ground acceleration. Ensure the units match
        the mass matrix (e.g., m/s^2, not g's, unless you pre-scale it).
    dt_record : float
        Time step (seconds) between samples in `accel_record`.

    Returns
    -------
    Callable
        A function `p_eff(t)` that returns the effective seismic load vector
        of shape `(ndof, 1)` at any time `t`.
    """
    direction = as_float_array(direction).reshape(-1)
    if is_sparse(M):
        M_r = np.asarray(M @ direction, dtype=float).reshape(-1, 1)
    else:
        M_r = (as_float_array(M) @ direction).reshape(-1, 1)
    accel = as_float_array(accel_record).ravel()
    t_rec = np.arange(len(accel)) * dt_record

    def _seismic(t):
        ag = float(np.interp(t, t_rec, accel, left=0.0, right=0.0))
        return -M_r * ag

    return _seismic


# ---------------------------------------------------------------------------
# Stability utilities
# ---------------------------------------------------------------------------


def critical_timestep(K, M, *, method: str = "power", n_iter: int = 50) -> float:
    """
    Estimate the critical time step for conditionally stable schemes.

    Parameters
    ----------
    K, M : ndarray or sparse
        Stiffness and mass matrices.
    method : str
        ``'power'`` for power iteration to approximate omega_max.
    n_iter : int
        Number of power iterations.

    Returns
    -------
    dt_cr : float
        Critical time step: dt_cr = 2 / omega_max.
    """
    ndof = K.shape[0]
    # Power iteration on M^{-1} K to find largest eigenvalue
    x = np.random.RandomState(42).randn(ndof)
    x /= np.linalg.norm(x)

    K_arr = K if is_sparse(K) else as_float_array(K)
    M_arr = M if is_sparse(M) else as_float_array(M)

    for _ in range(n_iter):
        y = K_arr @ x if is_sparse(K_arr) else K_arr @ x
        # Solve M z = y
        if is_sparse(M_arr):
            z = spla.spsolve(M_arr.tocsc(), y)
        else:
            z = np.linalg.solve(M_arr, y)
        lam = float(z @ x)
        norm_z = np.linalg.norm(z)
        if norm_z == 0:
            break
        x = z / norm_z

    omega_max = np.sqrt(max(lam, 0.0))
    if omega_max == 0:
        return np.inf
    return 2.0 / omega_max


def _check_stability(dt, dt_cr, beta, gamma):
    """Warn if dt exceeds critical time step for conditionally stable schemes."""
    unconditional = beta >= gamma / 2.0 >= 0.25
    if not unconditional and dt > dt_cr:
        warnings.warn(
            f"Time step dt={dt:.6e} exceeds critical dt_cr={dt_cr:.6e}. "
            f"The scheme (beta={beta}, gamma={gamma}) is conditionally stable "
            f"and may diverge.",
            stacklevel=3,
        )


# ---------------------------------------------------------------------------
# Energy computation
# ---------------------------------------------------------------------------


def _compute_energy(M, C, K, u, v, p_ext, dt):
    """Compute energy components at a single time step."""
    M_arr = M if is_sparse(M) else as_float_array(M)
    K_arr = K if is_sparse(K) else as_float_array(K)
    u_flat = u.ravel()
    v_flat = v.ravel()

    kin = 0.5 * float(v_flat @ (M_arr @ v_flat))
    strain = 0.5 * float(u_flat @ (K_arr @ u_flat))
    return {"kinetic": kin, "strain": strain}


# ---------------------------------------------------------------------------
# Newmark-beta implicit solver
# ---------------------------------------------------------------------------


def solve_newmark(
    M,
    C,
    K,
    p_func: Callable,
    u0,
    v0,
    dt: float,
    nsteps: int,
    *,
    beta: float = 0.25,
    gamma: float = 0.5,
    C_bc=None,
    dof: int = 2,
    compute_energy: bool = False,
) -> TimeHistory:
    """
    Implicit Newmark-beta time integration for linear structural dynamics.

    Solves the transient equations of motion:
        M * a(t) + C * v(t) + K * u(t) = p(t)

    Mathematical Formulation
    ------------------------
    The Newmark method approximates the displacement and velocity at t+dt as:
        u(t+dt) = u(t) + dt * v(t) + dt^2 * [ (0.5 - beta) * a(t) + beta * a(t+dt) ]
        v(t+dt) = v(t) + dt * [ (1 - gamma) * a(t) + gamma * a(t+dt) ]

    This leads to an effective static linear system solved at each time step:
        K_eff * u(t+dt) = p_eff(t+dt)

    where:
        K_eff = K + a0 * M + a1 * C
        p_eff(t+dt) = p(t+dt) + M * (a0*u(t) + a2*v(t) + a3*a(t))
                              + C * (a1*u(t) + a4*v(t) + a5*a(t))
    and a0, a1, ... are Newmark constants depending on dt, beta, and gamma.

    Parameters
    ----------
    M : ndarray or scipy.sparse matrix, shape (ndof, ndof)
        Global mass matrix.
    C : ndarray or scipy.sparse matrix, shape (ndof, ndof)
        Global damping matrix.
    K : ndarray or scipy.sparse matrix, shape (ndof, ndof)
        Global stiffness matrix.
    p_func : callable
        Load function returning the force vector at time t.
        Signature: `p_func(t) -> ndarray of shape (ndof, 1)`.
    u0 : array_like, shape (ndof, 1)
        Initial displacement vector.
    v0 : array_like, shape (ndof, 1)
        Initial velocity vector.
    dt : float
        Time step size (seconds).
    nsteps : int
        Number of time steps to compute. Total time = dt * nsteps.
    beta : float, default 0.25
        Newmark beta parameter. Controls variation of acceleration over the step.
    gamma : float, default 0.5
        Newmark gamma parameter. Controls artificial numerical damping.
    C_bc : array_like, optional
        Boundary constraint table. Constrained DOFs are perfectly fixed (u=v=a=0)
        during the dynamic response.
    dof : int, default 2
        DOFs per node (used to interpret `C_bc`).
    compute_energy : bool, default False
        If True, compute kinetic and strain energy at each step.

    Returns
    -------
    TimeHistory
        Dataclass containing the full time history of displacements `u`,
        velocities `v`, accelerations `a`, and optionally energy.

    Notes
    -----
    Default parameters (`beta=0.25`, `gamma=0.5`) use the constant average
    acceleration method (trapezoidal rule). This scheme is unconditionally stable
    for linear systems and produces no artificial numerical dissipation.
    """
    ndof = K.shape[0]
    u0 = as_float_array(u0).reshape(-1, 1)
    v0 = as_float_array(v0).reshape(-1, 1)

    # Determine constrained DOFs
    bc_dofs = _get_bc_dofs(C_bc, dof, ndof)

    # Newmark integration constants
    a0 = 1.0 / (beta * dt**2)
    a1 = gamma / (beta * dt)
    a2 = 1.0 / (beta * dt)
    a3 = 1.0 / (2.0 * beta) - 1.0
    a4 = gamma / beta - 1.0
    a5 = dt * (gamma / (2.0 * beta) - 1.0)

    # Form effective stiffness: K_eff = K + a0*M + a1*C
    K_eff = _form_effective_stiffness(K, M, C, a0, a1)

    # Apply BCs to effective stiffness
    K_eff, ks = _apply_bc_to_matrix(K_eff, bc_dofs)

    # Factorize K_eff if possible for repeated solves
    K_factor = _factorize(K_eff)

    # Initial acceleration: a0 = M^{-1} (p(0) - C v0 - K u0)
    p0 = as_float_array(p_func(0.0)).reshape(-1, 1)
    rhs_init = p0 - _matvec(C, v0) - _matvec(K, u0)
    a_init = _solve_mass(M, rhs_init, bc_dofs)

    # Allocate history
    t_hist = np.linspace(0, nsteps * dt, nsteps + 1)
    u_hist = np.zeros((nsteps + 1, ndof), dtype=float)
    v_hist = np.zeros((nsteps + 1, ndof), dtype=float)
    a_hist = np.zeros((nsteps + 1, ndof), dtype=float)
    u_hist[0] = u0.ravel()
    v_hist[0] = v0.ravel()
    a_hist[0] = a_init.ravel()

    energy_hist = None
    if compute_energy:
        energy_hist = {
            "kinetic": np.zeros(nsteps + 1),
            "strain": np.zeros(nsteps + 1),
            "total": np.zeros(nsteps + 1),
        }
        e = _compute_energy(M, C, K, u0, v0, p0, dt)
        energy_hist["kinetic"][0] = e["kinetic"]
        energy_hist["strain"][0] = e["strain"]
        energy_hist["total"][0] = e["kinetic"] + e["strain"]

    # Time stepping
    u_n = u0.copy()
    v_n = v0.copy()
    a_n = a_init.copy()

    for n in range(nsteps):
        t_next = (n + 1) * dt
        p_next = as_float_array(p_func(t_next)).reshape(-1, 1)

        # Effective load
        p_eff = (
            p_next
            + _matvec(M, a0 * u_n + a2 * v_n + a3 * a_n)
            + _matvec(C, a1 * u_n + a4 * v_n + a5 * a_n)
        )

        # Apply BCs to effective load
        p_eff[bc_dofs] = 0.0

        # Solve
        u_next = _solve_factored(K_factor, K_eff, p_eff)
        u_next[bc_dofs] = 0.0

        # Update acceleration and velocity
        a_next = a0 * (u_next - u_n) - a2 * v_n - a3 * a_n
        v_next = v_n + dt * ((1.0 - gamma) * a_n + gamma * a_next)

        # Zero constrained DOFs
        a_next[bc_dofs] = 0.0
        v_next[bc_dofs] = 0.0

        # Store
        u_hist[n + 1] = u_next.ravel()
        v_hist[n + 1] = v_next.ravel()
        a_hist[n + 1] = a_next.ravel()

        if compute_energy:
            e = _compute_energy(M, C, K, u_next, v_next, p_next, dt)
            energy_hist["kinetic"][n + 1] = e["kinetic"]
            energy_hist["strain"][n + 1] = e["strain"]
            energy_hist["total"][n + 1] = e["kinetic"] + e["strain"]

        u_n = u_next
        v_n = v_next
        a_n = a_next

    return TimeHistory(
        t=t_hist,
        u=u_hist,
        v=v_hist,
        a=a_hist,
        dt=dt,
        nsteps=nsteps,
        energy=energy_hist,
    )


# ---------------------------------------------------------------------------
# Central difference (explicit) solver
# ---------------------------------------------------------------------------


def solve_central_diff(
    M_lumped,
    C,
    K,
    p_func: Callable,
    u0,
    v0,
    dt: float,
    nsteps: int,
    *,
    C_bc=None,
    dof: int = 2,
    compute_energy: bool = False,
) -> TimeHistory:
    """
    Explicit central difference time integration.

    Requires a diagonal (lumped) mass matrix.

    Parameters
    ----------
    M_lumped : ndarray, shape (ndof,) or (ndof, ndof) diagonal
        Lumped (diagonal) mass matrix. Can be a 1D array of diagonal entries
        or a full diagonal matrix.
    C : ndarray or sparse, shape (ndof, ndof)
        Damping matrix. For pure explicit, use C=0.
    K : ndarray or sparse, shape (ndof, ndof)
        Stiffness matrix.
    p_func : callable
        Load function: p_func(t) -> ndarray (ndof, 1).
    u0, v0 : array_like
        Initial displacement and velocity.
    dt : float
        Time step size (must be < dt_critical).
    nsteps : int
        Number of time steps.
    C_bc : array_like, optional
        Boundary constraint table.
    dof : int, default 2
        DOFs per node.
    compute_energy : bool
        If True, compute energy at each step.

    Returns
    -------
    TimeHistory
        Full time history of u, v, a.

    Raises
    ------
    ValueError
        If the mass matrix appears non-diagonal.
    """
    ndof = K.shape[0]
    u0 = as_float_array(u0).reshape(-1, 1)
    v0 = as_float_array(v0).reshape(-1, 1)

    # Extract diagonal mass
    m_diag = _get_lumped_diagonal(M_lumped, ndof)

    bc_dofs = _get_bc_dofs(C_bc, dof, ndof)

    # Initial acceleration
    p0 = as_float_array(p_func(0.0)).reshape(-1, 1)
    r0 = p0 - _matvec(K, u0)
    if C is not None and not _is_zero_matrix(C):
        r0 -= _matvec(C, v0)
    a0 = np.zeros_like(u0)
    free = m_diag > 0
    a0[free] = r0[free] / m_diag[free]
    a0[bc_dofs] = 0.0

    # Backward extrapolation for u_{-1}
    u_prev = u0 - dt * v0 + 0.5 * dt**2 * a0

    # Allocate
    t_hist = np.linspace(0, nsteps * dt, nsteps + 1)
    u_hist = np.zeros((nsteps + 1, ndof), dtype=float)
    v_hist = np.zeros((nsteps + 1, ndof), dtype=float)
    a_hist = np.zeros((nsteps + 1, ndof), dtype=float)
    u_hist[0] = u0.ravel()
    v_hist[0] = v0.ravel()
    a_hist[0] = a0.ravel()

    energy_hist = None
    if compute_energy:
        energy_hist = {
            "kinetic": np.zeros(nsteps + 1),
            "strain": np.zeros(nsteps + 1),
            "total": np.zeros(nsteps + 1),
        }
        M_diag_mat = np.diag(m_diag.ravel())
        e = _compute_energy(M_diag_mat, C, K, u0, v0, p0, dt)
        energy_hist["kinetic"][0] = e["kinetic"]
        energy_hist["strain"][0] = e["strain"]
        energy_hist["total"][0] = e["kinetic"] + e["strain"]

    u_n = u0.copy()
    u_nm1 = u_prev.copy()

    # Effective diagonal mass with damping: M_eff = M/dt^2 + C/(2*dt)
    has_C = C is not None and not _is_zero_matrix(C)

    for n in range(nsteps):
        t_n = n * dt
        p_n = as_float_array(p_func(t_n)).reshape(-1, 1)

        # p_eff = p_n - (K - 2*M/dt^2)*u_n - (M/dt^2 - C/(2*dt))*u_{n-1}
        p_eff = (
            p_n - _matvec(K, u_n) + 2.0 * m_diag * u_n / dt**2 - m_diag * u_nm1 / dt**2
        )
        if has_C:
            p_eff += _matvec(C, u_nm1) / (2.0 * dt)

        # M_eff * u_{n+1} = p_eff
        m_eff = m_diag / dt**2
        if has_C:
            c_diag = _get_damping_diagonal(C, ndof)
            m_eff = m_eff + c_diag / (2.0 * dt)

        u_next = np.zeros_like(u_n)
        active = m_eff.ravel() > 0
        u_next[active] = p_eff[active] / m_eff[active]
        u_next[bc_dofs] = 0.0

        # Velocity and acceleration
        v_n = (u_next - u_nm1) / (2.0 * dt)
        a_n_new = (u_next - 2.0 * u_n + u_nm1) / dt**2
        v_n[bc_dofs] = 0.0
        a_n_new[bc_dofs] = 0.0

        u_hist[n + 1] = u_next.ravel()
        v_hist[n + 1] = v_n.ravel()
        a_hist[n + 1] = a_n_new.ravel()

        if compute_energy:
            M_diag_mat = np.diag(m_diag.ravel())
            e = _compute_energy(M_diag_mat, C, K, u_next, v_n, p_n, dt)
            energy_hist["kinetic"][n + 1] = e["kinetic"]
            energy_hist["strain"][n + 1] = e["strain"]
            energy_hist["total"][n + 1] = e["kinetic"] + e["strain"]

        u_nm1 = u_n
        u_n = u_next

    return TimeHistory(
        t=t_hist,
        u=u_hist,
        v=v_hist,
        a=a_hist,
        dt=dt,
        nsteps=nsteps,
        energy=energy_hist,
    )


# ---------------------------------------------------------------------------
# HHT-alpha solver
# ---------------------------------------------------------------------------


def solve_hht(
    M,
    C,
    K,
    p_func: Callable,
    u0,
    v0,
    dt: float,
    nsteps: int,
    *,
    alpha: float = -0.05,
    C_bc=None,
    dof: int = 2,
    compute_energy: bool = False,
) -> TimeHistory:
    """
    HHT-alpha (Hilber-Hughes-Taylor) time integration.

    A generalization of Newmark-beta that provides controllable numerical
    dissipation of high-frequency modes while maintaining second-order accuracy.

    Parameters
    ----------
    M, C, K : ndarray or sparse
        Mass, damping, stiffness matrices.
    p_func : callable
        Load function.
    u0, v0 : array_like
        Initial conditions.
    dt : float
        Time step.
    nsteps : int
        Number of time steps.
    alpha : float, default -0.05
        HHT parameter in [-1/3, 0]. alpha=0 recovers standard Newmark.
        More negative = more high-frequency dissipation.
    C_bc, dof, compute_energy : see solve_newmark

    Returns
    -------
    TimeHistory
    """
    if alpha < -1.0 / 3.0 or alpha > 0.0:
        raise ValueError(f"HHT alpha must be in [-1/3, 0], got {alpha}")

    # Derived Newmark parameters from alpha
    beta = (1.0 - alpha) ** 2 / 4.0
    gamma = 0.5 - alpha

    ndof = K.shape[0]
    u0 = as_float_array(u0).reshape(-1, 1)
    v0 = as_float_array(v0).reshape(-1, 1)

    bc_dofs = _get_bc_dofs(C_bc, dof, ndof)

    a0 = 1.0 / (beta * dt**2)
    a1 = gamma / (beta * dt)
    a2 = 1.0 / (beta * dt)
    a3 = 1.0 / (2.0 * beta) - 1.0
    a4 = gamma / beta - 1.0
    a5 = dt * (gamma / (2.0 * beta) - 1.0)

    # HHT effective stiffness includes (1+alpha) factors
    K_eff = _form_hht_effective_stiffness(K, M, C, a0, a1, alpha)
    K_eff, ks = _apply_bc_to_matrix(K_eff, bc_dofs)
    K_factor = _factorize(K_eff)

    # Initial acceleration
    p0 = as_float_array(p_func(0.0)).reshape(-1, 1)
    rhs_init = p0 - _matvec(C, v0) - _matvec(K, u0)
    a_init = _solve_mass(M, rhs_init, bc_dofs)

    t_hist = np.linspace(0, nsteps * dt, nsteps + 1)
    u_hist = np.zeros((nsteps + 1, ndof), dtype=float)
    v_hist = np.zeros((nsteps + 1, ndof), dtype=float)
    a_hist = np.zeros((nsteps + 1, ndof), dtype=float)
    u_hist[0] = u0.ravel()
    v_hist[0] = v0.ravel()
    a_hist[0] = a_init.ravel()

    energy_hist = None
    if compute_energy:
        energy_hist = {
            "kinetic": np.zeros(nsteps + 1),
            "strain": np.zeros(nsteps + 1),
            "total": np.zeros(nsteps + 1),
        }
        e = _compute_energy(M, C, K, u0, v0, p0, dt)
        energy_hist["kinetic"][0] = e["kinetic"]
        energy_hist["strain"][0] = e["strain"]
        energy_hist["total"][0] = e["kinetic"] + e["strain"]

    u_n = u0.copy()
    v_n = v0.copy()
    a_n = a_init.copy()

    for n in range(nsteps):
        t_n = n * dt
        t_next = (n + 1) * dt
        p_n = as_float_array(p_func(t_n)).reshape(-1, 1)
        p_next = as_float_array(p_func(t_next)).reshape(-1, 1)

        # HHT effective load
        p_eff = (
            (1.0 + alpha) * p_next
            - alpha * p_n
            + _matvec(M, a0 * u_n + a2 * v_n + a3 * a_n)
            + (1.0 + alpha) * _matvec(C, a1 * u_n + a4 * v_n + a5 * a_n)
            - alpha * _matvec(C, v_n)
            - alpha * _matvec(K, u_n)
        )
        p_eff[bc_dofs] = 0.0

        u_next = _solve_factored(K_factor, K_eff, p_eff)
        u_next[bc_dofs] = 0.0

        a_next = a0 * (u_next - u_n) - a2 * v_n - a3 * a_n
        v_next = v_n + dt * ((1.0 - gamma) * a_n + gamma * a_next)
        a_next[bc_dofs] = 0.0
        v_next[bc_dofs] = 0.0

        u_hist[n + 1] = u_next.ravel()
        v_hist[n + 1] = v_next.ravel()
        a_hist[n + 1] = a_next.ravel()

        if compute_energy:
            e = _compute_energy(M, C, K, u_next, v_next, p_next, dt)
            energy_hist["kinetic"][n + 1] = e["kinetic"]
            energy_hist["strain"][n + 1] = e["strain"]
            energy_hist["total"][n + 1] = e["kinetic"] + e["strain"]

        u_n = u_next
        v_n = v_next
        a_n = a_next

    return TimeHistory(
        t=t_hist,
        u=u_hist,
        v=v_hist,
        a=a_hist,
        dt=dt,
        nsteps=nsteps,
        energy=energy_hist,
    )


# ---------------------------------------------------------------------------
# Nonlinear Newmark solver
# ---------------------------------------------------------------------------


def solve_newmark_nl(
    M,
    C,
    tangent_func: Callable,
    internal_force_func: Callable,
    p_func: Callable,
    u0,
    v0,
    dt: float,
    nsteps: int,
    *,
    beta: float = 0.25,
    gamma: float = 0.5,
    tol: float = 1e-6,
    max_iter: int = 20,
    C_bc=None,
    dof: int = 2,
) -> TimeHistory:
    """
    Nonlinear Newmark-beta with Newton-Raphson iteration at each step.

    Parameters
    ----------
    M, C : ndarray or sparse
        Mass and damping matrices.
    tangent_func : callable
        tangent_func(u, state) -> K_t (tangent stiffness).
    internal_force_func : callable
        internal_force_func(u, state) -> (q, state_new).
    p_func : callable
        External load function: p_func(t) -> p.
    u0, v0 : array_like
        Initial conditions.
    dt, nsteps : float, int
        Time step and number of steps.
    beta, gamma : float
        Newmark parameters.
    tol : float
        Newton convergence tolerance.
    max_iter : int
        Max Newton iterations per step.
    C_bc : array_like, optional
        Boundary constraint table.
    dof : int
        DOFs per node.

    Returns
    -------
    TimeHistory
    """
    ndof = M.shape[0]
    u0 = as_float_array(u0).reshape(-1, 1)
    v0 = as_float_array(v0).reshape(-1, 1)
    bc_dofs = _get_bc_dofs(C_bc, dof, ndof)

    a0_c = 1.0 / (beta * dt**2)
    a1_c = gamma / (beta * dt)
    a2_c = 1.0 / (beta * dt)
    a3_c = 1.0 / (2.0 * beta) - 1.0

    # Initial acceleration
    p0 = as_float_array(p_func(0.0)).reshape(-1, 1)
    q0, state = internal_force_func(u0, None)
    q0 = as_float_array(q0).reshape(-1, 1)
    rhs_init = p0 - q0 - _matvec(C, v0)
    a_init = _solve_mass(M, rhs_init, bc_dofs)

    t_hist = np.linspace(0, nsteps * dt, nsteps + 1)
    u_hist = np.zeros((nsteps + 1, ndof), dtype=float)
    v_hist = np.zeros((nsteps + 1, ndof), dtype=float)
    a_hist = np.zeros((nsteps + 1, ndof), dtype=float)
    u_hist[0] = u0.ravel()
    v_hist[0] = v0.ravel()
    a_hist[0] = a_init.ravel()

    u_n = u0.copy()
    v_n = v0.copy()
    a_n = a_init.copy()

    for n in range(nsteps):
        t_next = (n + 1) * dt
        p_next = as_float_array(p_func(t_next)).reshape(-1, 1)

        # Predictor
        u_k = u_n + dt * v_n + 0.5 * dt**2 * (1.0 - 2.0 * beta) * a_n
        v_k = v_n + dt * (1.0 - gamma) * a_n
        a_k = np.zeros_like(a_n)

        for _it in range(max_iter):
            K_t = tangent_func(u_k, state)
            q_k, state_trial = internal_force_func(u_k, state)
            q_k = as_float_array(q_k).reshape(-1, 1)

            # Residual
            R = p_next - _matvec(M, a_k) - _matvec(C, v_k) - q_k
            R[bc_dofs] = 0.0

            # Check convergence
            r_norm = float(np.linalg.norm(R))
            p_norm = float(np.linalg.norm(p_next))
            if r_norm < tol * max(p_norm, 1.0):
                state = state_trial
                break

            # Effective tangent
            K_eff_nl = _form_effective_stiffness(K_t, M, C, a0_c, a1_c)
            K_eff_nl, _ = _apply_bc_to_matrix(K_eff_nl, bc_dofs)

            # Newton correction
            du = solve_linear_system(K_eff_nl, R)
            du[bc_dofs] = 0.0

            u_k = u_k + du
            a_k = a0_c * (u_k - u_n) - a2_c * v_n - a3_c * a_n
            v_k = v_n + dt * ((1.0 - gamma) * a_n + gamma * a_k)
            u_k[bc_dofs] = 0.0
            a_k[bc_dofs] = 0.0
            v_k[bc_dofs] = 0.0

        u_hist[n + 1] = u_k.ravel()
        v_hist[n + 1] = v_k.ravel()
        a_hist[n + 1] = a_k.ravel()

        u_n = u_k
        v_n = v_k
        a_n = a_k

    return TimeHistory(
        t=t_hist,
        u=u_hist,
        v=v_hist,
        a=a_hist,
        dt=dt,
        nsteps=nsteps,
        energy=None,
    )


# ---------------------------------------------------------------------------
# Dynamic post-processing plots
# ---------------------------------------------------------------------------


def plot_time_history(
    result: TimeHistory, dof_index, *, quantity="displacement", ax=None
):
    """
    Plot displacement, velocity, or acceleration vs time at specified DOFs.

    Parameters
    ----------
    result : TimeHistory
        Output from any time integrator.
    dof_index : int or list of int
        DOF index (0-based) or list of DOF indices.
    quantity : str
        ``'displacement'``, ``'velocity'``, or ``'acceleration'``.
    ax : matplotlib Axes, optional
        If provided, plot on this axes.

    Returns
    -------
    matplotlib Axes
    """
    import matplotlib.pyplot as plt

    if isinstance(dof_index, int):
        dof_index = [dof_index]

    data_map = {
        "displacement": result.u,
        "velocity": result.v,
        "acceleration": result.a,
    }
    data = data_map[quantity]

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))

    for di in dof_index:
        ax.plot(result.t, data[:, di], label=f"DOF {di}")

    ax.set_xlabel("Time (s)")
    ax.set_ylabel(quantity.capitalize())
    ax.legend()
    ax.grid(True, alpha=0.3)
    return ax


def plot_energy(result: TimeHistory, *, ax=None):
    """
    Plot energy components vs time.

    Parameters
    ----------
    result : TimeHistory
        Must have compute_energy=True in the solver call.
    ax : matplotlib Axes, optional

    Returns
    -------
    matplotlib Axes
    """
    import matplotlib.pyplot as plt

    if result.energy is None:
        raise ValueError("No energy data. Run solver with compute_energy=True.")

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))

    ax.plot(result.t, result.energy["kinetic"], "b-", label="Kinetic")
    ax.plot(result.t, result.energy["strain"], "r-", label="Strain")
    ax.plot(result.t, result.energy["total"], "k-", linewidth=2, label="Total")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Energy")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return ax


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _get_bc_dofs(C_bc, dof, ndof):
    """Return array of constrained DOF indices from BC table."""
    if C_bc is None:
        return np.array([], dtype=int)
    constraints = as_float_array(C_bc)
    if constraints.size == 0:
        return np.array([], dtype=int)
    if dof == 1:
        return constraints[:, 0].astype(int) - 1
    return (constraints[:, 0].astype(int) - 1) * dof + constraints[:, 1].astype(int) - 1


def _form_effective_stiffness(K, M, C, a0, a1):
    """K_eff = K + a0*M + a1*C."""
    if is_sparse(K) or is_sparse(M):
        K_s = K.tocsr() if is_sparse(K) else sp.csr_matrix(as_float_array(K))
        M_s = M.tocsr() if is_sparse(M) else sp.csr_matrix(as_float_array(M))
        C_s = C.tocsr() if is_sparse(C) else sp.csr_matrix(as_float_array(C))
        return (K_s + a0 * M_s + a1 * C_s).toarray()
    return as_float_array(K) + a0 * as_float_array(M) + a1 * as_float_array(C)


def _form_hht_effective_stiffness(K, M, C, a0, a1, alpha):
    """K_eff = (1+alpha)*K + a0*M + (1+alpha)*a1*C."""
    if is_sparse(K) or is_sparse(M):
        K_s = K.tocsr() if is_sparse(K) else sp.csr_matrix(as_float_array(K))
        M_s = M.tocsr() if is_sparse(M) else sp.csr_matrix(as_float_array(M))
        C_s = C.tocsr() if is_sparse(C) else sp.csr_matrix(as_float_array(C))
        return ((1.0 + alpha) * K_s + a0 * M_s + (1.0 + alpha) * a1 * C_s).toarray()
    return (
        (1.0 + alpha) * as_float_array(K)
        + a0 * as_float_array(M)
        + (1.0 + alpha) * a1 * as_float_array(C)
    )


def _apply_bc_to_matrix(K_eff, bc_dofs):
    """Zero rows/cols at BC DOFs and put 1 on diagonal."""
    K_eff = as_float_array(K_eff).copy()
    ks = 0.1 * float(np.max(np.abs(np.diag(K_eff)))) if K_eff.size > 0 else 1.0
    if ks == 0:
        ks = 1.0
    for j in bc_dofs:
        K_eff[j, :] = 0.0
        K_eff[:, j] = 0.0
        K_eff[j, j] = ks
    return K_eff, ks


def _factorize(K_eff):
    """Attempt LU factorization for repeated solves."""
    try:
        from scipy.linalg import lu_factor

        return lu_factor(K_eff)
    except Exception:
        return None


def _solve_factored(factor, K_eff, rhs):
    """Solve using pre-computed LU or fallback to direct solve."""
    rhs_flat = rhs.reshape(-1)
    if factor is not None:
        try:
            from scipy.linalg import lu_solve

            return lu_solve(factor, rhs_flat).reshape(-1, 1)
        except Exception:
            pass
    return np.linalg.solve(K_eff, rhs_flat).reshape(-1, 1)


def _matvec(mat, vec):
    """Matrix-vector product handling sparse and dense."""
    v = vec.reshape(-1) if vec.ndim > 1 else vec
    if is_sparse(mat):
        return np.asarray(mat @ v, dtype=float).reshape(-1, 1)
    return (as_float_array(mat) @ v).reshape(-1, 1)


def _solve_mass(M, rhs, bc_dofs):
    """Solve M a = rhs with zero BCs."""
    result = solve_linear_system(M, rhs)
    result[bc_dofs] = 0.0
    return result


def _get_lumped_diagonal(M, ndof):
    """Extract diagonal from a lumped mass matrix."""
    M_arr = as_float_array(M) if not is_sparse(M) else np.asarray(M.toarray())
    if M_arr.ndim == 1:
        if M_arr.size != ndof:
            raise ValueError(f"Lumped mass vector size {M_arr.size} != ndof {ndof}")
        return M_arr.reshape(-1, 1)
    if M_arr.shape != (ndof, ndof):
        raise ValueError(f"Mass matrix shape {M_arr.shape} != ({ndof}, {ndof})")
    # Check it's actually diagonal
    off_diag_norm = np.linalg.norm(M_arr - np.diag(np.diag(M_arr)))
    if off_diag_norm > 1e-12 * np.linalg.norm(M_arr):
        raise ValueError(
            "Central difference requires a lumped (diagonal) mass matrix. "
            "Use lumped=True when computing element mass matrices."
        )
    return np.diag(M_arr).reshape(-1, 1)


def _get_damping_diagonal(C, ndof):
    """Extract diagonal of the damping matrix."""
    if is_sparse(C):
        return np.asarray(C.diagonal()).reshape(-1, 1)
    return np.diag(as_float_array(C)).reshape(-1, 1)


def _is_zero_matrix(mat):
    """Check if a matrix is identically zero."""
    if mat is None:
        return True
    if is_sparse(mat):
        return mat.nnz == 0
    return float(np.linalg.norm(as_float_array(mat))) == 0.0


# ---------------------------------------------------------------------------
# Frequency Response Function (FRF)
# ---------------------------------------------------------------------------


def compute_frf(M, C, K, input_dof, output_dof, freq_range, *, n_points=500):
    """
    Compute the Frequency Response Function (FRF), H(omega), for harmonic excitation.

    Mathematical Formulation
    ------------------------
    The equation of motion for a structural system subjected to a complex
    harmonic excitation F(t) = P * e^{i * omega * t} has the steady-state
    harmonic response U(t) = U * e^{i * omega * t}.

    Substituting these into M * U'' + C * U' + K * U = P yields:
        [-omega^2 * M + i * omega * C + K] * U = P

    The complex term in brackets is the Dynamic Stiffness Matrix, Z(omega).
    The FRF matrix is H(omega) = Z(omega)^-1.

    This function computes the scalar transfer function H(omega)_{j,k} where
    an excitation is applied at DOF `k` (`input_dof`) and the response is
    measured at DOF `j` (`output_dof`). The magnitude |H| gives the amplification,
    and the phase angle gives the lag relative to the excitation.

    Parameters
    ----------
    M : ndarray or scipy.sparse matrix, shape (ndof, ndof)
        Global mass matrix.
    C : ndarray or scipy.sparse matrix, shape (ndof, ndof)
        Global damping matrix.
    K : ndarray or scipy.sparse matrix, shape (ndof, ndof)
        Global stiffness matrix.
    input_dof : int
        The 0-based global degree-of-freedom index where the unit harmonic
        load P is applied.
    output_dof : int
        The 0-based global degree-of-freedom index where the complex
        displacement response U is measured.
    freq_range : tuple of floats
        The frequency spectrum window (f_min, f_max) to evaluate, in Hertz (Hz).
    n_points : int, default 500
        Number of discrete frequency sampling points within the `freq_range`.

    Returns
    -------
    freq_hz : ndarray, shape (n_points,)
        Array of frequencies evaluated (Hz).
    H : ndarray, shape (n_points,), dtype=complex
        Array of complex scalar frequency response values H(omega).
        Use `numpy.abs(H)` for magnitude and `numpy.angle(H)` for phase.

        Frequency values in Hz.
    H : ndarray, shape (n_points,), complex
        Complex FRF values.

    Examples
    --------
    >>> freq, H = compute_frf(M, C, K, input_dof=0, output_dof=0,
    ...                       freq_range=(0.1, 50.0), n_points=1000)
    >>> magnitude = np.abs(H)
    """
    f_min, f_max = freq_range
    freq_hz = np.linspace(f_min, f_max, n_points)
    omega = 2.0 * np.pi * freq_hz

    ndof = K.shape[0]
    K_arr = as_float_array(K) if not is_sparse(K) else np.asarray(K.toarray())
    M_arr = as_float_array(M) if not is_sparse(M) else np.asarray(M.toarray())
    C_arr = as_float_array(C) if not is_sparse(C) else np.asarray(C.toarray())

    # Unit force vector at input DOF
    f_unit = np.zeros(ndof, dtype=complex)
    f_unit[input_dof] = 1.0

    H = np.zeros(n_points, dtype=complex)
    for i, w in enumerate(omega):
        # Dynamic stiffness matrix: Z = K - w^2*M + i*w*C
        Z = K_arr - w**2 * M_arr + 1j * w * C_arr
        try:
            u = np.linalg.solve(Z, f_unit)
            H[i] = u[output_dof]
        except np.linalg.LinAlgError:
            H[i] = np.nan + 1j * np.nan

    return freq_hz, H


def plot_frf(freq_hz, H, *, ax=None, log_scale=True, mark_peaks=True):
    """
    Plot magnitude and phase of a Frequency Response Function.

    Parameters
    ----------
    freq_hz : ndarray
        Frequency values in Hz.
    H : ndarray, complex
        FRF values.
    ax : matplotlib Axes or None
        If None, creates a new 2-subplot figure.
    log_scale : bool
        If True, use log scale for magnitude.
    mark_peaks : bool
        If True, mark resonance peaks.

    Returns
    -------
    fig : matplotlib Figure
    """
    import matplotlib.pyplot as plt

    magnitude = np.abs(H)
    phase = np.angle(H, deg=True)

    if ax is None:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    else:
        fig = ax.figure
        ax1 = ax
        ax2 = fig.add_subplot(212)

    # Magnitude
    if log_scale:
        ax1.semilogy(freq_hz, magnitude, "b-", linewidth=1.0)
    else:
        ax1.plot(freq_hz, magnitude, "b-", linewidth=1.0)
    ax1.set_ylabel("Magnitude |H(f)|")
    ax1.set_title("Frequency Response Function")
    ax1.grid(True, alpha=0.3)

    if mark_peaks and len(magnitude) > 2:
        # Simple peak detection
        from scipy.signal import find_peaks as _find_peaks

        peaks, _ = _find_peaks(magnitude, prominence=0.1 * np.max(magnitude))
        for pk in peaks:
            ax1.axvline(freq_hz[pk], color="r", ls="--", alpha=0.5)
            ax1.annotate(
                f"{freq_hz[pk]:.2f} Hz",
                xy=(freq_hz[pk], magnitude[pk]),
                fontsize=8,
                color="r",
            )

    # Phase
    ax2.plot(freq_hz, phase, "r-", linewidth=1.0)
    ax2.set_xlabel("Frequency (Hz)")
    ax2.set_ylabel("Phase (deg)")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


__all__ = [
    "NewmarkParams",
    "TimeHistory",
    "compute_frf",
    "constant_load",
    "critical_timestep",
    "harmonic_load",
    "plot_energy",
    "plot_frf",
    "plot_time_history",
    "pulse_load",
    "ramp_load",
    "seismic_load",
    "solve_central_diff",
    "solve_hht",
    "solve_newmark",
    "solve_newmark_nl",
    "tabulated_load",
]
