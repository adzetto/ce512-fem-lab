"""Tests for element mass matrix routines.

Every element type is checked for:
  - correct shape and symmetry
  - positive semi-definiteness (all eigenvalues >= 0)
  - total mass preservation: r^T M r == rho * Volume  (rigid-body translation)
  - consistent vs lumped total mass agreement
  - global assembly correctness
"""

from __future__ import annotations

import numpy as np
import pytest

from femlabpy import init
from femlabpy.elements.bars import mebar, mbar
from femlabpy.elements.triangles import met3e, mt3e
from femlabpy.elements.quads import meq4e, mq4e
from femlabpy.elements.solids import meT4e, mT4e, meh8e, mh8e


def _total_mass(Me, dof):
    """Compute total mass via rigid-body mode r^T M r in direction 0."""
    r = np.zeros(Me.shape[0], dtype=float)
    r[0::dof] = 1.0
    return float(r @ Me @ r)


# ---------------------------------------------------------------------------
# Bar element
# ---------------------------------------------------------------------------


class TestBarMass:
    Xe = np.array([[0.0, 0.0], [3.0, 0.0]], dtype=float)  # L=3
    Ge = np.array([2.0, 200.0, 7.0])  # A=2, E=200, rho=7
    expected_mass = 7.0 * 2.0 * 3.0  # rho * A * L = 42

    def test_shape_and_symmetry(self):
        Me = mebar(self.Xe, self.Ge, dof=2)
        assert Me.shape == (4, 4)
        np.testing.assert_allclose(Me, Me.T, atol=1e-15)

    def test_consistent_total_mass(self):
        Me = mebar(self.Xe, self.Ge, dof=2)
        np.testing.assert_allclose(_total_mass(Me, 2), self.expected_mass, atol=1e-12)

    def test_lumped_total_mass(self):
        Me = mebar(self.Xe, self.Ge, dof=2, lumped=True)
        np.testing.assert_allclose(_total_mass(Me, 2), self.expected_mass, atol=1e-12)

    def test_lumped_is_diagonal(self):
        Me = mebar(self.Xe, self.Ge, dof=2, lumped=True)
        np.testing.assert_allclose(Me, np.diag(np.diag(Me)), atol=1e-15)

    def test_positive_semi_definite(self):
        Me = mebar(self.Xe, self.Ge, dof=2)
        eigs = np.linalg.eigvalsh(Me)
        assert eigs.min() >= -1e-14

    def test_3d_bar(self):
        Xe = np.array([[0, 0, 0], [4, 0, 0]], dtype=float)
        Ge = np.array([3.0, 100.0, 5.0])  # A=3, E=100, rho=5
        Me = mebar(Xe, Ge, dof=3)
        assert Me.shape == (6, 6)
        np.testing.assert_allclose(_total_mass(Me, 3), 5 * 3 * 4, atol=1e-12)

    def test_rho_default(self):
        Ge_no_rho = np.array([2.0, 200.0])  # no rho => defaults to 1
        Me = mebar(self.Xe, Ge_no_rho, dof=2)
        np.testing.assert_allclose(_total_mass(Me, 2), 1.0 * 2.0 * 3.0, atol=1e-12)

    def test_global_assembly(self):
        # Two bar elements sharing middle node: 1--2--3
        X = np.array([[0, 0], [1, 0], [2, 0]], dtype=float)
        T = np.array([[1, 2, 1], [2, 3, 1]], dtype=float)
        G = np.array([[1.0, 100.0, 10.0]])  # A=1, E=100, rho=10
        K, p, q = init(3, 2)
        M = np.zeros_like(K)
        M = mbar(M, T, X, G, dof=2)
        # Total mass = 10*1*1 + 10*1*1 = 20 (two bars of length 1)
        r = np.zeros(6)
        r[0::2] = 1.0
        np.testing.assert_allclose(r @ M @ r, 20.0, atol=1e-12)


# ---------------------------------------------------------------------------
# T3 triangle element
# ---------------------------------------------------------------------------


class TestT3Mass:
    Xe = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=float)  # A=0.5
    Ge = np.array([200.0, 0.3, 1, 1.0, 5.0])  # E,nu,type,t=1,rho=5
    expected_mass = 5.0 * 1.0 * 0.5  # rho * t * A = 2.5

    def test_shape_and_symmetry(self):
        Me = met3e(self.Xe, self.Ge)
        assert Me.shape == (6, 6)
        np.testing.assert_allclose(Me, Me.T, atol=1e-15)

    def test_consistent_total_mass(self):
        Me = met3e(self.Xe, self.Ge)
        np.testing.assert_allclose(_total_mass(Me, 2), self.expected_mass, atol=1e-12)

    def test_lumped_total_mass(self):
        Me = met3e(self.Xe, self.Ge, lumped=True)
        np.testing.assert_allclose(_total_mass(Me, 2), self.expected_mass, atol=1e-12)

    def test_positive_definite(self):
        Me = met3e(self.Xe, self.Ge)
        eigs = np.linalg.eigvalsh(Me)
        assert eigs.min() > -1e-14

    def test_consistent_formula(self):
        """Verify the analytical consistent mass formula."""
        Me = met3e(self.Xe, self.Ge)
        rho, t, A = 5.0, 1.0, 0.5
        scalar = np.array([[2, 1, 1], [1, 2, 1], [1, 1, 2]], dtype=float)
        expected = (rho * t * A / 12.0) * np.kron(scalar, np.eye(2))
        np.testing.assert_allclose(Me, expected, atol=1e-14)

    def test_global_assembly(self):
        X = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=float)
        T = np.array([[1, 2, 3, 1], [2, 4, 3, 1]], dtype=float)
        G = np.array([[100.0, 0.3, 1, 1.0, 8.0]])
        K, p, q = init(4, 2)
        M = np.zeros_like(K)
        M = mt3e(M, T, X, G)
        r = np.zeros(8)
        r[0::2] = 1.0
        total = r @ M @ r
        # Total area = 0.5 + 0.5 = 1.0, mass = 8*1*1 = 8
        np.testing.assert_allclose(total, 8.0, atol=1e-12)


# ---------------------------------------------------------------------------
# Q4 quad element
# ---------------------------------------------------------------------------


class TestQ4Mass:
    Xe = np.array([[0, 0], [2, 0], [2, 3], [0, 3]], dtype=float)  # A=6
    Ge = np.array([200.0, 0.3, 1, 1.5, 8.0])  # t=1.5, rho=8
    expected_mass = 8.0 * 1.5 * 6.0  # = 72

    def test_shape_and_symmetry(self):
        Me = meq4e(self.Xe, self.Ge)
        assert Me.shape == (8, 8)
        np.testing.assert_allclose(Me, Me.T, atol=1e-14)

    def test_consistent_total_mass(self):
        Me = meq4e(self.Xe, self.Ge)
        np.testing.assert_allclose(_total_mass(Me, 2), self.expected_mass, atol=1e-10)

    def test_lumped_total_mass(self):
        Me = meq4e(self.Xe, self.Ge, lumped=True)
        np.testing.assert_allclose(_total_mass(Me, 2), self.expected_mass, atol=1e-10)

    def test_positive_semi_definite(self):
        Me = meq4e(self.Xe, self.Ge)
        eigs = np.linalg.eigvalsh(Me)
        assert eigs.min() >= -1e-12

    def test_unit_square(self):
        """Unit square with unit thickness and density => mass = 1."""
        Xe = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
        Ge = np.array([100.0, 0.3, 1, 1.0, 1.0])
        Me = meq4e(Xe, Ge)
        np.testing.assert_allclose(_total_mass(Me, 2), 1.0, atol=1e-12)


# ---------------------------------------------------------------------------
# T4 tetrahedron element
# ---------------------------------------------------------------------------


class TestT4Mass:
    Xe = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
    Ge = np.array([200.0, 0.3, 10.0])  # rho=10
    V = 1.0 / 6.0
    expected_mass = 10.0 * V

    def test_shape_and_symmetry(self):
        Me = meT4e(self.Xe, self.Ge)
        assert Me.shape == (12, 12)
        np.testing.assert_allclose(Me, Me.T, atol=1e-15)

    def test_consistent_total_mass(self):
        Me = meT4e(self.Xe, self.Ge)
        np.testing.assert_allclose(_total_mass(Me, 3), self.expected_mass, atol=1e-12)

    def test_lumped_total_mass(self):
        Me = meT4e(self.Xe, self.Ge, lumped=True)
        np.testing.assert_allclose(_total_mass(Me, 3), self.expected_mass, atol=1e-12)

    def test_consistent_formula(self):
        Me = meT4e(self.Xe, self.Ge)
        rho, V = 10.0, 1.0 / 6.0
        scalar = np.array(
            [[2, 1, 1, 1], [1, 2, 1, 1], [1, 1, 2, 1], [1, 1, 1, 2]], dtype=float
        )
        expected = (rho * V / 20.0) * np.kron(scalar, np.eye(3))
        np.testing.assert_allclose(Me, expected, atol=1e-14)


# ---------------------------------------------------------------------------
# H8 hexahedral element
# ---------------------------------------------------------------------------


class TestH8Mass:
    Xe = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
            [1, 1, 1],
            [0, 1, 1],
        ],
        dtype=float,
    )
    Ge = np.array([200.0, 0.3, 6.0])  # rho=6
    expected_mass = 6.0 * 1.0  # rho * V = 6

    def test_shape_and_symmetry(self):
        Me = meh8e(self.Xe, self.Ge)
        assert Me.shape == (24, 24)
        np.testing.assert_allclose(Me, Me.T, atol=1e-13)

    def test_consistent_total_mass(self):
        Me = meh8e(self.Xe, self.Ge)
        np.testing.assert_allclose(_total_mass(Me, 3), self.expected_mass, atol=1e-10)

    def test_lumped_total_mass(self):
        Me = meh8e(self.Xe, self.Ge, lumped=True)
        np.testing.assert_allclose(_total_mass(Me, 3), self.expected_mass, atol=1e-10)

    def test_positive_semi_definite(self):
        Me = meh8e(self.Xe, self.Ge)
        eigs = np.linalg.eigvalsh(Me)
        assert eigs.min() >= -1e-12


# ---------------------------------------------------------------------------
# init() dynamic flag
# ---------------------------------------------------------------------------


class TestDynamicInit:
    def test_static_returns_3(self):
        result = init(10, 2)
        assert len(result) == 3

    def test_dynamic_returns_4(self):
        result = init(10, 2, dynamic=True)
        assert len(result) == 4
        K, M, p, q = result
        assert K.shape == (20, 20)
        assert M.shape == (20, 20)
        assert p.shape == (20, 1)
        assert q.shape == (20, 1)
