"""
Tests for femlabpy.dynamics — Newmark-beta implicit solver.

Validates against closed-form SDOF solutions, energy conservation,
second-order accuracy, boundary condition handling, and load builders.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from femlabpy.dynamics import (
    NewmarkParams,
    TimeHistory,
    constant_load,
    critical_timestep,
    harmonic_load,
    pulse_load,
    ramp_load,
    solve_newmark,
    tabulated_load,
)


# ---------------------------------------------------------------------------
# Helpers — SDOF spring-mass system
# ---------------------------------------------------------------------------


def _sdof_system(m=1.0, k=100.0):
    """Return M, C=0, K for a 1-DOF spring-mass system."""
    M = np.array([[m]])
    K = np.array([[k]])
    C = np.zeros((1, 1))
    return M, C, K


def _sdof_omega(m=1.0, k=100.0):
    return np.sqrt(k / m)


# ---------------------------------------------------------------------------
# NewmarkParams dataclass
# ---------------------------------------------------------------------------


class TestNewmarkParams:
    def test_average_acceleration(self):
        p = NewmarkParams.average_acceleration()
        assert p.beta == 0.25
        assert p.gamma == 0.5

    def test_linear_acceleration(self):
        p = NewmarkParams.linear_acceleration()
        assert_allclose(p.beta, 1.0 / 6.0)
        assert p.gamma == 0.5

    def test_central_difference(self):
        p = NewmarkParams.central_difference()
        assert p.beta == 0.0
        assert p.gamma == 0.5

    def test_fox_goodwin(self):
        p = NewmarkParams.fox_goodwin()
        assert_allclose(p.beta, 1.0 / 12.0)


# ---------------------------------------------------------------------------
# Load builders
# ---------------------------------------------------------------------------


class TestLoadBuilders:
    def test_constant_load(self):
        P = np.array([10.0])
        f = constant_load(P)
        assert_allclose(f(0.0).ravel(), [10.0])
        assert_allclose(f(999.0).ravel(), [10.0])

    def test_ramp_load(self):
        P = np.array([20.0])
        f = ramp_load(P, t_ramp=2.0)
        assert_allclose(f(0.0).ravel(), [0.0])
        assert_allclose(f(1.0).ravel(), [10.0])
        assert_allclose(f(2.0).ravel(), [20.0])
        assert_allclose(f(5.0).ravel(), [20.0])  # capped

    def test_harmonic_load(self):
        P = np.array([1.0])
        omega = 10.0
        f = harmonic_load(P, omega)
        assert_allclose(f(0.0).ravel(), [0.0], atol=1e-15)
        assert_allclose(f(np.pi / (2 * omega)).ravel(), [1.0], atol=1e-14)

    def test_pulse_load(self):
        P = np.array([5.0])
        f = pulse_load(P, t_start=1.0, t_duration=0.5)
        assert_allclose(f(0.5).ravel(), [0.0])
        assert_allclose(f(1.0).ravel(), [5.0])
        assert_allclose(f(1.25).ravel(), [5.0])
        assert_allclose(f(1.5).ravel(), [5.0])
        assert_allclose(f(2.0).ravel(), [0.0])

    def test_tabulated_load(self):
        P = np.array([1.0])
        tt = [0, 1, 2, 3]
        vt = [0, 1, 1, 0]
        f = tabulated_load(P, tt, vt)
        assert_allclose(f(0.5).ravel(), [0.5], atol=1e-14)
        assert_allclose(f(1.5).ravel(), [1.0], atol=1e-14)
        assert_allclose(f(2.5).ravel(), [0.5], atol=1e-14)
        assert_allclose(f(5.0).ravel(), [0.0], atol=1e-14)  # outside range


# ---------------------------------------------------------------------------
# SDOF undamped free vibration
# ---------------------------------------------------------------------------


class TestNewmarkSDOFFreeVibration:
    """SDOF undamped: u(t) = u0 * cos(omega*t)."""

    def test_cosine_response(self):
        m, k = 1.0, 100.0
        omega = _sdof_omega(m, k)
        M, C, K = _sdof_system(m, k)

        u0 = np.array([[1.0]])
        v0 = np.array([[0.0]])
        T_period = 2 * np.pi / omega
        dt = T_period / 100  # 100 steps per period
        nsteps = 500  # 5 periods

        result = solve_newmark(M, C, K, constant_load(np.zeros(1)), u0, v0, dt, nsteps)

        # Analytical solution
        u_exact = np.cos(omega * result.t)
        # For average acceleration (beta=0.25, gamma=0.5), 2nd order accuracy.
        # Phase error accumulates over multiple periods: ~O(dt^2 * n_periods).
        # With dt/T=0.01 over 5 periods, max error ~0.01 is expected.
        max_err = np.max(np.abs(result.u[:, 0] - u_exact))
        assert max_err < 0.02, f"max error = {max_err}"

    def test_initial_velocity(self):
        """u(t) = (v0/omega)*sin(omega*t)."""
        m, k = 1.0, 100.0
        omega = _sdof_omega(m, k)
        M, C, K = _sdof_system(m, k)

        u0 = np.array([[0.0]])
        v0 = np.array([[5.0]])
        T_period = 2 * np.pi / omega
        dt = T_period / 100
        nsteps = 200

        result = solve_newmark(M, C, K, constant_load(np.zeros(1)), u0, v0, dt, nsteps)
        u_exact = (5.0 / omega) * np.sin(omega * result.t)
        max_err = np.max(np.abs(result.u[:, 0] - u_exact))
        # Phase error accumulates over 2 periods; ~0.002 is expected
        assert max_err < 0.01, f"max error = {max_err}"

    def test_result_type(self):
        M, C, K = _sdof_system()
        u0 = np.array([[1.0]])
        v0 = np.array([[0.0]])
        result = solve_newmark(M, C, K, constant_load(np.zeros(1)), u0, v0, 0.01, 10)
        assert isinstance(result, TimeHistory)
        assert result.u.shape == (11, 1)
        assert result.v.shape == (11, 1)
        assert result.a.shape == (11, 1)
        assert result.t.shape == (11,)
        assert result.dt == 0.01
        assert result.nsteps == 10


# ---------------------------------------------------------------------------
# Energy conservation
# ---------------------------------------------------------------------------


class TestNewmarkEnergy:
    """Undamped system should conserve total energy."""

    def test_energy_conservation(self):
        m, k = 1.0, 100.0
        M, C, K = _sdof_system(m, k)
        u0 = np.array([[1.0]])
        v0 = np.array([[0.0]])
        omega = _sdof_omega(m, k)
        dt = (2 * np.pi / omega) / 50
        nsteps = 200

        result = solve_newmark(
            M,
            C,
            K,
            constant_load(np.zeros(1)),
            u0,
            v0,
            dt,
            nsteps,
            compute_energy=True,
        )
        assert result.energy is not None
        total = result.energy["total"]
        # Total energy should be nearly constant (E0 = 0.5*k*u0^2 = 50)
        assert_allclose(total, total[0], atol=1e-10)

    def test_energy_equals_analytic(self):
        """E_total = 0.5 * k * u0^2."""
        m, k = 2.0, 50.0
        M, C, K = _sdof_system(m, k)
        u0 = np.array([[3.0]])
        v0 = np.array([[0.0]])
        E0 = 0.5 * k * 3.0**2  # = 225
        omega = _sdof_omega(m, k)
        dt = (2 * np.pi / omega) / 50
        result = solve_newmark(
            M,
            C,
            K,
            constant_load(np.zeros(1)),
            u0,
            v0,
            dt,
            100,
            compute_energy=True,
        )
        assert_allclose(result.energy["total"][0], E0, rtol=1e-12)


# ---------------------------------------------------------------------------
# Second-order accuracy
# ---------------------------------------------------------------------------


class TestNewmarkAccuracy:
    """Halving dt should reduce error by ~4x (2nd order)."""

    def test_second_order_convergence(self):
        m, k = 1.0, 100.0
        omega = _sdof_omega(m, k)
        M, C, K = _sdof_system(m, k)
        u0 = np.array([[1.0]])
        v0 = np.array([[0.0]])
        T_end = 2 * np.pi / omega  # one period

        errors = []
        for n in [50, 100, 200]:
            dt = T_end / n
            nsteps = n
            result = solve_newmark(
                M, C, K, constant_load(np.zeros(1)), u0, v0, dt, nsteps
            )
            u_exact = np.cos(omega * result.t)
            err = np.max(np.abs(result.u[:, 0] - u_exact))
            errors.append(err)

        # Convergence rate: err[i]/err[i+1] should be ~4 for 2nd order
        rate1 = errors[0] / errors[1]
        rate2 = errors[1] / errors[2]
        assert rate1 > 3.5, f"rate1 = {rate1}"
        assert rate2 > 3.5, f"rate2 = {rate2}"


# ---------------------------------------------------------------------------
# Boundary conditions
# ---------------------------------------------------------------------------


class TestNewmarkBCs:
    """Constrained DOFs should remain zero."""

    def test_constrained_dofs_stay_zero(self):
        """2-DOF system with DOF 1 fixed."""
        M = np.array([[2.0, 0.0], [0.0, 1.0]])
        K = np.array([[6.0, -2.0], [-2.0, 4.0]])
        C = np.zeros((2, 2))

        # Constrain node 1, dof 1 (0-based DOF 0)
        C_bc = np.array([[1, 1, 0.0]])

        u0 = np.array([[0.0], [1.0]])
        v0 = np.array([[0.0], [0.0]])

        result = solve_newmark(
            M,
            C,
            K,
            constant_load(np.zeros(2)),
            u0,
            v0,
            dt=0.01,
            nsteps=100,
            C_bc=C_bc,
            dof=1,
        )
        # DOF 0 should remain zero throughout
        assert_allclose(result.u[:, 0], 0.0, atol=1e-14)
        assert_allclose(result.v[:, 0], 0.0, atol=1e-14)
        assert_allclose(result.a[:, 0], 0.0, atol=1e-14)


# ---------------------------------------------------------------------------
# Critical timestep
# ---------------------------------------------------------------------------


class TestCriticalTimestep:
    def test_sdof(self):
        """dt_cr = 2/omega for SDOF."""
        m, k = 1.0, 100.0
        M, C, K = _sdof_system(m, k)
        omega = _sdof_omega(m, k)
        dt_cr = critical_timestep(K, M)
        expected = 2.0 / omega
        assert_allclose(dt_cr, expected, rtol=0.05)

    def test_2dof(self):
        """2-DOF system: dt_cr = 2/omega_max."""
        M = np.diag([1.0, 1.0])
        K = np.array([[3.0, -1.0], [-1.0, 2.0]])
        from scipy.linalg import eigh

        eigvals, _ = eigh(K, M)
        omega_max = np.sqrt(eigvals.max())
        dt_cr = critical_timestep(K, M)
        expected = 2.0 / omega_max
        assert_allclose(dt_cr, expected, rtol=0.05)


# ---------------------------------------------------------------------------
# Forced vibration (harmonic)
# ---------------------------------------------------------------------------


class TestNewmarkForcedVibration:
    """Harmonic forcing of undamped SDOF: resonance and steady state."""

    def test_static_load_gives_static_displacement(self):
        """Constant load with small damping should reach u_static = P/K."""
        m, k = 1.0, 100.0
        M, C_mat, K = _sdof_system(m, k)
        # Add artificial damping to reach steady state
        C_mat = np.array([[2.0]])  # overdamped

        P = np.array([10.0])
        u0 = np.array([[0.0]])
        v0 = np.array([[0.0]])
        omega = _sdof_omega(m, k)
        T = 2 * np.pi / omega
        dt = T / 50
        nsteps = 2000  # many periods to damp out

        result = solve_newmark(M, C_mat, K, constant_load(P), u0, v0, dt, nsteps)
        u_static = 10.0 / 100.0  # P/k = 0.1
        # Last few percent of displacement history should be near u_static
        assert_allclose(result.u[-1, 0], u_static, rtol=0.01)
