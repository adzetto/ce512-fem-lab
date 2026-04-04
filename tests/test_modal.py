"""
Tests for femlabpy.modal — eigenvalue solver for natural frequencies/mode shapes.

Validates against exact SDOF/2-DOF eigenvalues, mass-orthogonality,
stiffness-orthogonality, BC elimination, and participation factors.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from femlabpy.modal import ModalResult, solve_modal


# ---------------------------------------------------------------------------
# SDOF exact frequency
# ---------------------------------------------------------------------------


class TestModalSDOF:
    """Single DOF: omega = sqrt(k/m)."""

    def test_sdof_frequency(self):
        m, k = 2.0, 200.0
        M = np.array([[m]])
        K = np.array([[k]])
        result = solve_modal(K, M, n_modes=1, dof=1)
        omega_exact = np.sqrt(k / m)
        assert_allclose(result.omega[0], omega_exact, rtol=1e-12)

    def test_sdof_freq_hz(self):
        m, k = 1.0, 100.0
        M = np.array([[m]])
        K = np.array([[k]])
        result = solve_modal(K, M, n_modes=1, dof=1)
        expected_hz = np.sqrt(k / m) / (2 * np.pi)
        assert_allclose(result.freq_hz[0], expected_hz, rtol=1e-12)

    def test_sdof_period(self):
        m, k = 1.0, 100.0
        M = np.array([[m]])
        K = np.array([[k]])
        result = solve_modal(K, M, n_modes=1, dof=1)
        expected_T = 2 * np.pi / np.sqrt(k / m)
        assert_allclose(result.period[0], expected_T, rtol=1e-12)


# ---------------------------------------------------------------------------
# 2-DOF exact frequencies
# ---------------------------------------------------------------------------


class TestModal2DOF:
    """2-DOF system with analytical eigenvalues."""

    @pytest.fixture()
    def system_2dof(self):
        """Two identical masses, three identical springs in series."""
        m = 1.0
        k = 100.0
        M = np.diag([m, m])
        K = np.array([[2 * k, -k], [-k, 2 * k]])
        # Exact eigenvalues: k and 3k
        # omega1 = sqrt(k/m) = 10, omega2 = sqrt(3k/m) = 17.32...
        return M, K, m, k

    def test_frequencies(self, system_2dof):
        M, K, m, k = system_2dof
        result = solve_modal(K, M, n_modes=2, dof=1)
        omega_exact = np.array([np.sqrt(k / m), np.sqrt(3 * k / m)])
        assert_allclose(result.omega, omega_exact, rtol=1e-10)

    def test_eigenvalues(self, system_2dof):
        M, K, m, k = system_2dof
        result = solve_modal(K, M, n_modes=2, dof=1)
        lam_exact = np.array([k / m, 3 * k / m])
        assert_allclose(result.eigenvalues, lam_exact, rtol=1e-10)


# ---------------------------------------------------------------------------
# Mass and stiffness orthogonality
# ---------------------------------------------------------------------------


class TestModalOrthogonality:
    """Mode shapes must satisfy orthogonality conditions."""

    @pytest.fixture()
    def modal_result(self):
        M = np.diag([2.0, 1.0, 1.5])
        K = np.array(
            [
                [10.0, -3.0, 0.0],
                [-3.0, 8.0, -2.0],
                [0.0, -2.0, 6.0],
            ]
        )
        return solve_modal(K, M, n_modes=3, dof=1), M, K

    def test_mass_orthogonality(self, modal_result):
        """Phi^T M Phi = I for mass-normalized modes."""
        result, M, K = modal_result
        phi = result.mode_shapes
        MtM = phi.T @ M @ phi
        assert_allclose(MtM, np.eye(3), atol=1e-10)

    def test_stiffness_orthogonality(self, modal_result):
        """Phi^T K Phi = diag(omega^2)."""
        result, M, K = modal_result
        phi = result.mode_shapes
        KtK = phi.T @ K @ phi
        expected = np.diag(result.eigenvalues)
        assert_allclose(KtK, expected, atol=1e-10)


# ---------------------------------------------------------------------------
# Boundary condition elimination
# ---------------------------------------------------------------------------


class TestModalBCs:
    """Constrained DOFs should be zero in mode shapes."""

    def test_fixed_dof_is_zero(self):
        """3-DOF system with DOF 0 fixed: only 2 modes remain."""
        M = np.diag([1.0, 2.0, 1.0])
        K = np.array(
            [
                [4.0, -2.0, 0.0],
                [-2.0, 6.0, -2.0],
                [0.0, -2.0, 3.0],
            ]
        )
        # Fix node 1, dof 1 (1-based): DOF index 0
        C_bc = np.array([[1, 1, 0.0]])
        result = solve_modal(K, M, n_modes=2, C_bc=C_bc, dof=1)

        # DOF 0 should be zero in all mode shapes
        assert_allclose(result.mode_shapes[0, :], 0.0, atol=1e-14)
        # Should still have 2 modes
        assert len(result.omega) == 2
        assert all(result.omega > 0)


# ---------------------------------------------------------------------------
# Participation factors
# ---------------------------------------------------------------------------


class TestParticipation:
    """Modal participation factors and effective mass."""

    def test_effective_mass_sum(self):
        """Sum of effective mass = total mass (for unconstrained system)."""
        M = np.diag([1.0, 2.0, 3.0])
        K = np.array(
            [
                [6.0, -2.0, 0.0],
                [-2.0, 5.0, -1.0],
                [0.0, -1.0, 4.0],
            ]
        )
        result = solve_modal(K, M, n_modes=3, dof=1)
        total_mass = np.trace(M)  # 6.0
        eff_mass_sum = result.effective_mass[:, 0].sum()
        assert_allclose(eff_mass_sum, total_mass, rtol=1e-10)

    def test_participation_shape(self):
        M = np.diag([1.0, 1.0])
        K = np.array([[3.0, -1.0], [-1.0, 2.0]])
        result = solve_modal(K, M, n_modes=2, dof=1)
        assert result.participation.shape == (2, 1)
        assert result.effective_mass.shape == (2, 1)


# ---------------------------------------------------------------------------
# ModalResult dataclass
# ---------------------------------------------------------------------------


class TestModalResult:
    def test_fields(self):
        M = np.array([[1.0]])
        K = np.array([[100.0]])
        result = solve_modal(K, M, n_modes=1, dof=1)
        assert isinstance(result, ModalResult)
        assert hasattr(result, "eigenvalues")
        assert hasattr(result, "omega")
        assert hasattr(result, "freq_hz")
        assert hasattr(result, "period")
        assert hasattr(result, "mode_shapes")
        assert hasattr(result, "participation")
        assert hasattr(result, "effective_mass")

    def test_mode_shape_dimension(self):
        n = 5
        M = np.eye(n)
        K = np.diag([1, 2, 3, 4, 5.0])
        result = solve_modal(K, M, n_modes=3, dof=1)
        assert result.mode_shapes.shape == (n, 3)
        assert result.omega.shape == (3,)
