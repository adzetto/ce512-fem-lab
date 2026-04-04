"""
Tests for femlabpy.damping — Rayleigh and modal damping.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from femlabpy.damping import modal_damping, rayleigh_coefficients, rayleigh_damping


# ---------------------------------------------------------------------------
# Rayleigh coefficients
# ---------------------------------------------------------------------------


class TestRayleighCoefficients:
    """Verify the 2x2 system solve for alpha, beta."""

    def test_equal_damping_ratios(self):
        """Standard case: zeta1 = zeta2 = 0.05 at omega1=10, omega2=50."""
        alpha, beta = rayleigh_coefficients(10.0, 50.0, 0.05, 0.05)
        # Verify the defining relation at both frequencies
        zeta1_check = alpha / (2 * 10) + beta * 10 / 2
        zeta2_check = alpha / (2 * 50) + beta * 50 / 2
        assert_allclose(zeta1_check, 0.05, atol=1e-14)
        assert_allclose(zeta2_check, 0.05, atol=1e-14)

    def test_unequal_damping_ratios(self):
        """Different damping ratios at two frequencies."""
        alpha, beta = rayleigh_coefficients(5.0, 100.0, 0.02, 0.10)
        zeta1 = alpha / (2 * 5) + beta * 5 / 2
        zeta2 = alpha / (2 * 100) + beta * 100 / 2
        assert_allclose(zeta1, 0.02, atol=1e-14)
        assert_allclose(zeta2, 0.10, atol=1e-14)

    def test_positive_coefficients_for_typical_input(self):
        """For equal zeta at two well-separated frequencies, both coeffs > 0."""
        alpha, beta = rayleigh_coefficients(10.0, 100.0, 0.05, 0.05)
        assert alpha > 0
        assert beta > 0

    def test_symmetry(self):
        """Swapping (omega1,zeta1) with (omega2,zeta2) gives same result."""
        a1, b1 = rayleigh_coefficients(10.0, 50.0, 0.03, 0.07)
        a2, b2 = rayleigh_coefficients(50.0, 10.0, 0.07, 0.03)
        assert_allclose(a1, a2, atol=1e-14)
        assert_allclose(b1, b2, atol=1e-14)


# ---------------------------------------------------------------------------
# Rayleigh damping matrix
# ---------------------------------------------------------------------------


class TestRayleighDamping:
    """Verify C = alpha*M + beta*K construction."""

    @pytest.fixture()
    def mk_pair(self):
        """Simple 2x2 system."""
        M = np.array([[2.0, 0.0], [0.0, 1.0]])
        K = np.array([[6.0, -2.0], [-2.0, 4.0]])
        return M, K

    def test_formula(self, mk_pair):
        M, K = mk_pair
        alpha, beta = 0.5, 0.001
        C = rayleigh_damping(M, K, alpha, beta)
        expected = alpha * M + beta * K
        assert_allclose(C, expected, atol=1e-15)

    def test_symmetry(self, mk_pair):
        M, K = mk_pair
        C = rayleigh_damping(M, K, 1.2, 0.003)
        assert_allclose(C, C.T, atol=1e-15)

    def test_sparse_input(self, mk_pair):
        """Sparse M/K should produce the same result (returned as lil)."""
        pytest.importorskip("scipy")
        import scipy.sparse as sp

        M, K = mk_pair
        M_sp = sp.csr_matrix(M)
        K_sp = sp.csr_matrix(K)
        alpha, beta = 0.5, 0.001
        C_sp = rayleigh_damping(M_sp, K_sp, alpha, beta)
        C_dense = rayleigh_damping(M, K, alpha, beta)
        assert_allclose(C_sp.toarray(), C_dense, atol=1e-14)

    def test_zero_coefficients(self, mk_pair):
        M, K = mk_pair
        C = rayleigh_damping(M, K, 0.0, 0.0)
        assert_allclose(C, np.zeros_like(M), atol=1e-15)


# ---------------------------------------------------------------------------
# Modal damping
# ---------------------------------------------------------------------------


class TestModalDamping:
    """Verify modal damping matrix reconstruction."""

    def _make_2dof(self):
        """2-DOF system with known eigenproperties."""
        M = np.array([[2.0, 0.0], [0.0, 1.0]])
        K = np.array([[6.0, -2.0], [-2.0, 4.0]])
        # Solve eigenvalue problem
        from scipy.linalg import eigh

        eigvals, eigvecs = eigh(K, M)
        omega = np.sqrt(eigvals)
        # Mass-normalize
        for i in range(eigvecs.shape[1]):
            m_norm = eigvecs[:, i] @ M @ eigvecs[:, i]
            eigvecs[:, i] /= np.sqrt(m_norm)
        return M, K, omega, eigvecs

    def test_modal_mass_orthogonality(self):
        """Modal C should diagonalize in modal coordinates."""
        M, K, omega, phi = self._make_2dof()
        zeta = np.array([0.05, 0.10])
        C = modal_damping(M, omega, phi, zeta)

        # Transform: phi^T C phi should = diag(2*zeta*omega)
        C_modal = phi.T @ C @ phi
        expected_diag = 2.0 * zeta * omega
        assert_allclose(np.diag(C_modal), expected_diag, atol=1e-12)
        # Off-diagonals should be near zero
        C_modal_offdiag = C_modal - np.diag(np.diag(C_modal))
        assert_allclose(C_modal_offdiag, 0.0, atol=1e-12)

    def test_symmetry(self):
        M, K, omega, phi = self._make_2dof()
        C = modal_damping(M, omega, phi, [0.02, 0.05])
        assert_allclose(C, C.T, atol=1e-15)

    def test_zero_damping(self):
        """Zero damping ratios should give zero C."""
        M, K, omega, phi = self._make_2dof()
        C = modal_damping(M, omega, phi, [0.0, 0.0])
        assert_allclose(C, np.zeros_like(M), atol=1e-15)

    def test_shape(self):
        M, K, omega, phi = self._make_2dof()
        C = modal_damping(M, omega, phi, [0.05, 0.05])
        assert C.shape == (2, 2)
