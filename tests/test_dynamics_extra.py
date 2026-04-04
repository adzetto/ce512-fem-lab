"""
Tests for femlabpy.dynamics — Central difference (explicit) and HHT-alpha solvers.

Validates central difference accuracy, conditional stability, lumped mass
requirement, HHT alpha=0 equivalence to Newmark, alpha range validation,
and high-frequency numerical dissipation.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from femlabpy.dynamics import (
    constant_load,
    solve_central_diff,
    solve_hht,
    solve_newmark,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sdof_system(m=1.0, k=100.0):
    M = np.array([[m]])
    K = np.array([[k]])
    C = np.zeros((1, 1))
    return M, C, K


def _sdof_omega(m=1.0, k=100.0):
    return np.sqrt(k / m)


# ---------------------------------------------------------------------------
# Central difference — accuracy
# ---------------------------------------------------------------------------


class TestCentralDiffAccuracy:
    """SDOF undamped free vibration with central difference."""

    def test_cosine_response(self):
        """u(t) = u0 * cos(omega*t) for undamped SDOF."""
        m, k = 1.0, 100.0
        omega = _sdof_omega(m, k)
        T_period = 2 * np.pi / omega

        M_lumped = np.array([[m]])  # diagonal
        K = np.array([[k]])
        C = np.zeros((1, 1))

        u0 = np.array([[1.0]])
        v0 = np.array([[0.0]])
        # Central diff is conditionally stable: dt < 2/omega = 0.2
        dt = T_period / 200  # well within stability
        nsteps = 400  # 2 periods

        result = solve_central_diff(
            M_lumped, C, K, constant_load(np.zeros(1)), u0, v0, dt, nsteps
        )
        u_exact = np.cos(omega * result.t)
        max_err = np.max(np.abs(result.u[:, 0] - u_exact))
        assert max_err < 0.01, f"max error = {max_err}"

    def test_1d_mass_vector(self):
        """Accept 1D array as lumped mass."""
        m, k = 1.0, 100.0
        omega = _sdof_omega(m, k)
        T_period = 2 * np.pi / omega

        M_vec = np.array([m])
        K = np.array([[k]])
        C = np.zeros((1, 1))

        u0 = np.array([[1.0]])
        v0 = np.array([[0.0]])
        dt = T_period / 200
        nsteps = 100

        result = solve_central_diff(
            M_vec, C, K, constant_load(np.zeros(1)), u0, v0, dt, nsteps
        )
        u_exact = np.cos(omega * result.t)
        max_err = np.max(np.abs(result.u[:, 0] - u_exact))
        assert max_err < 0.01, f"max error = {max_err}"


# ---------------------------------------------------------------------------
# Central difference — conditional stability
# ---------------------------------------------------------------------------


class TestCentralDiffStability:
    """Central difference should blow up for dt > dt_critical."""

    def test_stable_within_limit(self):
        """dt = 0.8 * dt_cr should remain bounded."""
        m, k = 1.0, 100.0
        omega = _sdof_omega(m, k)
        dt_cr = 2.0 / omega  # = 0.2
        dt = 0.8 * dt_cr

        M = np.array([[m]])
        K = np.array([[k]])
        C = np.zeros((1, 1))
        u0 = np.array([[1.0]])
        v0 = np.array([[0.0]])

        result = solve_central_diff(
            M, C, K, constant_load(np.zeros(1)), u0, v0, dt, 200
        )
        # Solution should stay bounded (amplitude ~ 1)
        assert np.max(np.abs(result.u)) < 2.0

    def test_unstable_past_limit(self):
        """dt = 1.1 * dt_cr should diverge."""
        m, k = 1.0, 100.0
        omega = _sdof_omega(m, k)
        dt_cr = 2.0 / omega
        dt = 1.1 * dt_cr

        M = np.array([[m]])
        K = np.array([[k]])
        C = np.zeros((1, 1))
        u0 = np.array([[1.0]])
        v0 = np.array([[0.0]])

        result = solve_central_diff(
            M, C, K, constant_load(np.zeros(1)), u0, v0, dt, 500
        )
        # Solution should have blown up
        assert np.max(np.abs(result.u)) > 100.0


# ---------------------------------------------------------------------------
# Central difference — requires diagonal mass
# ---------------------------------------------------------------------------


class TestCentralDiffLumpedMass:
    """Non-diagonal mass should raise ValueError."""

    def test_rejects_consistent_mass(self):
        M_consistent = np.array([[2.0, 1.0], [1.0, 2.0]])
        K = np.array([[4.0, -2.0], [-2.0, 4.0]])
        C = np.zeros((2, 2))
        u0 = np.zeros((2, 1))
        v0 = np.zeros((2, 1))

        with pytest.raises(ValueError, match="lumped.*diagonal"):
            solve_central_diff(
                M_consistent, C, K, constant_load(np.zeros(2)), u0, v0, 0.01, 10
            )


# ---------------------------------------------------------------------------
# Central difference — energy
# ---------------------------------------------------------------------------


class TestCentralDiffEnergy:
    """Energy computation for explicit solver."""

    def test_energy_approximately_conserved(self):
        m, k = 1.0, 100.0
        omega = _sdof_omega(m, k)
        T_period = 2 * np.pi / omega
        dt = T_period / 200

        M = np.array([[m]])
        K = np.array([[k]])
        C = np.zeros((1, 1))
        u0 = np.array([[1.0]])
        v0 = np.array([[0.0]])

        result = solve_central_diff(
            M,
            C,
            K,
            constant_load(np.zeros(1)),
            u0,
            v0,
            dt,
            200,
            compute_energy=True,
        )
        total = result.energy["total"]
        E0 = 0.5 * k * 1.0**2  # = 50
        # Central difference computes velocity at staggered (midpoint) times,
        # so the discrete energy oscillates around E0.  With dt/T ~ 0.005
        # the oscillation amplitude is ~3%.  Check mean is close and no drift.
        assert_allclose(np.mean(total), E0, rtol=0.02)
        # No secular drift: first-half mean ≈ second-half mean
        half = len(total) // 2
        assert_allclose(np.mean(total[:half]), np.mean(total[half:]), rtol=0.02)


# ---------------------------------------------------------------------------
# HHT — alpha=0 matches Newmark
# ---------------------------------------------------------------------------


class TestHHTAlphaZero:
    """HHT with alpha=0 should reproduce standard Newmark (beta=0.25, gamma=0.5)."""

    def test_matches_newmark(self):
        m, k = 1.0, 100.0
        M, C, K = _sdof_system(m, k)
        u0 = np.array([[1.0]])
        v0 = np.array([[0.0]])
        omega = _sdof_omega(m, k)
        dt = (2 * np.pi / omega) / 50
        nsteps = 100

        p = constant_load(np.zeros(1))

        result_nm = solve_newmark(M, C, K, p, u0, v0, dt, nsteps)
        result_hht = solve_hht(M, C, K, p, u0, v0, dt, nsteps, alpha=0.0)

        assert_allclose(result_hht.u, result_nm.u, atol=1e-12)
        assert_allclose(result_hht.v, result_nm.v, atol=1e-12)
        assert_allclose(result_hht.a, result_nm.a, atol=1e-12)


# ---------------------------------------------------------------------------
# HHT — alpha range validation
# ---------------------------------------------------------------------------


class TestHHTAlphaRange:
    """HHT alpha must be in [-1/3, 0]."""

    def test_rejects_positive_alpha(self):
        M, C, K = _sdof_system()
        u0 = np.array([[0.0]])
        v0 = np.array([[0.0]])
        with pytest.raises(ValueError, match="alpha"):
            solve_hht(
                M,
                C,
                K,
                constant_load(np.zeros(1)),
                u0,
                v0,
                dt=0.01,
                nsteps=1,
                alpha=0.1,
            )

    def test_rejects_too_negative_alpha(self):
        M, C, K = _sdof_system()
        u0 = np.array([[0.0]])
        v0 = np.array([[0.0]])
        with pytest.raises(ValueError, match="alpha"):
            solve_hht(
                M,
                C,
                K,
                constant_load(np.zeros(1)),
                u0,
                v0,
                dt=0.01,
                nsteps=1,
                alpha=-0.5,
            )

    def test_accepts_valid_alpha(self):
        """alpha = -1/3 (boundary) should work."""
        M, C, K = _sdof_system()
        u0 = np.array([[0.0]])
        v0 = np.array([[0.0]])
        result = solve_hht(
            M,
            C,
            K,
            constant_load(np.zeros(1)),
            u0,
            v0,
            dt=0.01,
            nsteps=5,
            alpha=-1.0 / 3.0,
        )
        assert result.u.shape == (6, 1)


# ---------------------------------------------------------------------------
# HHT — high-frequency dissipation
# ---------------------------------------------------------------------------


class TestHHTDissipation:
    """Negative alpha should dissipate high-frequency content."""

    def test_dissipation_reduces_amplitude(self):
        """Compare alpha=0 (no dissipation) vs alpha=-0.1 over many periods."""
        m, k = 1.0, 100.0
        M, C, K = _sdof_system(m, k)
        u0 = np.array([[1.0]])
        v0 = np.array([[0.0]])
        omega = _sdof_omega(m, k)
        dt = (2 * np.pi / omega) / 50
        nsteps = 500

        p = constant_load(np.zeros(1))
        result_0 = solve_hht(M, C, K, p, u0, v0, dt, nsteps, alpha=0.0)
        result_d = solve_hht(M, C, K, p, u0, v0, dt, nsteps, alpha=-0.1)

        # With alpha=0 the amplitude should be preserved (~1.0)
        # With alpha=-0.1 the amplitude should decay
        amp_final_0 = np.max(np.abs(result_0.u[-50:, 0]))
        amp_final_d = np.max(np.abs(result_d.u[-50:, 0]))

        # alpha=0: amplitude stays near 1
        assert amp_final_0 > 0.95
        # alpha=-0.1: amplitude should be noticeably reduced
        assert amp_final_d < amp_final_0

    def test_hht_energy(self):
        """HHT with alpha < 0 should show decreasing total energy."""
        m, k = 1.0, 100.0
        M, C, K = _sdof_system(m, k)
        u0 = np.array([[1.0]])
        v0 = np.array([[0.0]])
        omega = _sdof_omega(m, k)
        dt = (2 * np.pi / omega) / 50
        nsteps = 200

        p = constant_load(np.zeros(1))
        result = solve_hht(
            M,
            C,
            K,
            p,
            u0,
            v0,
            dt,
            nsteps,
            alpha=-0.1,
            compute_energy=True,
        )
        # Total energy at end should be less than at start
        assert result.energy["total"][-1] < result.energy["total"][0]
