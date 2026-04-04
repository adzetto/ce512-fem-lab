"""
Tests for femlabpy.periodic — Periodic boundary conditions and homogenization.

Validates node pair detection, constraint matrix construction, periodic solver,
homogenization of homogeneous material, mesh validation, and corner fixing.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from femlabpy.periodic import (
    check_periodic_mesh,
    find_all_periodic_pairs,
    find_periodic_pairs,
    fix_corner,
    periodic_constraints,
    solve_periodic,
)


# ---------------------------------------------------------------------------
# Helpers — structured quad mesh on unit square
# ---------------------------------------------------------------------------


def _unit_square_mesh(nx=4, ny=4):
    """
    Create a structured Q4 mesh on [0,1]x[0,1].

    Returns
    -------
    T : ndarray, shape (nel, 5)
        Topology table (4 nodes + material group), 1-based.
    X : ndarray, shape (nn, 2)
        Nodal coordinates.
    """
    nn = (nx + 1) * (ny + 1)
    X = np.zeros((nn, 2), dtype=float)
    for j in range(ny + 1):
        for i in range(nx + 1):
            node = j * (nx + 1) + i
            X[node, 0] = i / nx
            X[node, 1] = j / ny

    nel = nx * ny
    T = np.zeros((nel, 5), dtype=float)
    for j in range(ny):
        for i in range(nx):
            e = j * nx + i
            n0 = j * (nx + 1) + i
            T[e, 0] = n0 + 1  # 1-based
            T[e, 1] = n0 + 2
            T[e, 2] = n0 + (nx + 1) + 2
            T[e, 3] = n0 + (nx + 1) + 1
            T[e, 4] = 1  # material group

    return T, X


def _unit_square_stiffness(T, X, E=1.0, nu=0.3, plane_type=1, thickness=1.0):
    """Build K for a unit-square Q4 mesh with plane stress."""
    from femlabpy.elements.quads import kq4e

    nn = X.shape[0]
    ndof = nn * 2
    # Material: [E, nu, type, t]
    G = np.array([[E, nu, plane_type, thickness]])
    K = np.zeros((ndof, ndof), dtype=float)
    K = kq4e(K, T, X, G)
    return K, G


# ---------------------------------------------------------------------------
# Periodic pair detection
# ---------------------------------------------------------------------------


class TestFindPeriodicPairs:
    """Test node pairing on opposite faces."""

    def test_x_axis_pairs(self):
        """Left-right pairs along x-axis on 4x4 mesh."""
        T, X = _unit_square_mesh(4, 4)
        pairs = find_periodic_pairs(X, axis=0)
        # 5 nodes on each vertical edge (ny+1=5)
        assert pairs.shape[0] == 5
        # All pairs should be 1-based
        assert pairs.min() >= 1
        assert pairs.max() <= X.shape[0]

    def test_y_axis_pairs(self):
        """Bottom-top pairs along y-axis."""
        T, X = _unit_square_mesh(4, 4)
        pairs = find_periodic_pairs(X, axis=1)
        # 5 nodes on each horizontal edge (nx+1=5)
        assert pairs.shape[0] == 5

    def test_coordinate_consistency(self):
        """Paired nodes should have same transverse coordinates."""
        T, X = _unit_square_mesh(4, 4)
        pairs = find_periodic_pairs(X, axis=0)
        for left, right in pairs:
            left_0 = left - 1
            right_0 = right - 1
            # y-coordinates should match
            assert_allclose(X[left_0, 1], X[right_0, 1], atol=1e-12)
            # left should be at x=0, right at x=1
            assert_allclose(X[left_0, 0], 0.0, atol=1e-12)
            assert_allclose(X[right_0, 0], 1.0, atol=1e-12)

    def test_mismatched_mesh_raises(self):
        """Non-periodic mesh should raise ValueError."""
        # Create mesh with different number of nodes on left vs right
        X = np.array(
            [
                [0.0, 0.0],
                [0.0, 0.5],
                [0.0, 1.0],
                [1.0, 0.0],
                [1.0, 1.0],  # only 2 right nodes vs 3 left
            ]
        )
        with pytest.raises(ValueError, match="mismatch"):
            find_periodic_pairs(X, axis=0)


class TestFindAllPeriodicPairs:
    def test_both_axes(self):
        T, X = _unit_square_mesh(3, 3)
        all_pairs = find_all_periodic_pairs(X, periodic_axes=[0, 1])
        assert 0 in all_pairs
        assert 1 in all_pairs
        assert all_pairs[0].shape[0] == 4  # ny+1 = 4
        assert all_pairs[1].shape[0] == 4  # nx+1 = 4


# ---------------------------------------------------------------------------
# Constraint matrix
# ---------------------------------------------------------------------------


class TestPeriodicConstraints:
    """Test constraint matrix G and RHS Q construction."""

    def test_constraint_shape(self):
        T, X = _unit_square_mesh(3, 3)
        pairs = find_periodic_pairs(X, axis=0)
        nn = X.shape[0]
        dof = 2
        G, Q = periodic_constraints(X, pairs, dof)
        n_pairs = pairs.shape[0]
        assert G.shape == (n_pairs * dof, nn * dof)
        assert Q.shape == (n_pairs * dof, 1)

    def test_zero_strain_gives_zero_rhs(self):
        """No macro strain => Q = 0."""
        T, X = _unit_square_mesh(3, 3)
        pairs = find_periodic_pairs(X, axis=0)
        G, Q = periodic_constraints(X, pairs, dof=2)
        assert_allclose(Q, 0.0, atol=1e-15)

    def test_unit_strain_rhs(self):
        """eps_macro = [1,0,0] (uniaxial x) => Q_x = dx for each pair."""
        T, X = _unit_square_mesh(3, 3)
        pairs = find_periodic_pairs(X, axis=0)
        eps_macro = np.array([1.0, 0.0, 0.0])
        G, Q = periodic_constraints(X, pairs, dof=2, eps_macro=eps_macro)

        for i, (left, right) in enumerate(pairs):
            dx = X[right - 1, 0] - X[left - 1, 0]  # should be 1.0
            # Row for x-dof
            row_x = i * 2
            assert_allclose(Q[row_x, 0], dx, atol=1e-12)
            # Row for y-dof: eps_xy contribution from tensor[1,:] @ dx_vec
            # For pure exx=1, tensor = [[1,0],[0,0]], so y-row RHS = 0
            row_y = i * 2 + 1
            assert_allclose(Q[row_y, 0], 0.0, atol=1e-12)

    def test_constraint_rank(self):
        """G should have full row rank (all constraints independent)."""
        T, X = _unit_square_mesh(3, 3)
        pairs = find_periodic_pairs(X, axis=0)
        G, Q = periodic_constraints(X, pairs, dof=2)
        rank = np.linalg.matrix_rank(G, tol=1e-10)
        assert rank == G.shape[0]


# ---------------------------------------------------------------------------
# Periodic solver — homogeneous material
# ---------------------------------------------------------------------------


class TestSolvePeriodicHomogeneous:
    """Homogeneous material under unit strain should give exact C_eff."""

    def _solve_unit_strain(self, eps_macro, nx=4, ny=4, E=100.0, nu=0.3):
        """Apply macro strain and return displacement."""
        T, X = _unit_square_mesh(nx, ny)
        K, G_mat = _unit_square_stiffness(T, X, E=E, nu=nu)
        nn = X.shape[0]
        dof = 2
        ndof = nn * dof

        # Combine pairs from both axes
        pairs_x = find_periodic_pairs(X, axis=0)
        pairs_y = find_periodic_pairs(X, axis=1)
        # Remove duplicate corner pairs — just stack them
        pairs = np.vstack([pairs_x, pairs_y])
        # Remove duplicates
        pairs_set = set()
        unique_pairs = []
        for row in pairs:
            key = (int(row[0]), int(row[1]))
            if key not in pairs_set:
                pairs_set.add(key)
                unique_pairs.append(row)
        pairs = np.array(unique_pairs)

        p = np.zeros((ndof, 1), dtype=float)
        u = solve_periodic(K, p, X, pairs, dof, eps_macro=eps_macro)
        return u, T, X, G_mat

    def test_uniaxial_strain_displacement_linear(self):
        """Under eps_xx=1, the x-displacement should vary linearly with x."""
        eps = np.array([1.0, 0.0, 0.0])
        u, T, X, _ = self._solve_unit_strain(eps)
        nn = X.shape[0]
        # u_x at each node should be approximately x (plus a constant)
        u_x = u[0::2, 0]
        x_coords = X[:, 0]
        # Remove rigid body: u_x - u_x[corner] should equal x - x[corner]
        corner = np.argmin(np.linalg.norm(X, axis=1))
        u_x_rel = u_x - u_x[corner]
        x_rel = x_coords - x_coords[corner]
        assert_allclose(u_x_rel, x_rel, atol=0.05)


# ---------------------------------------------------------------------------
# Mesh validation
# ---------------------------------------------------------------------------


class TestCheckPeriodicMesh:
    def test_valid_mesh(self):
        T, X = _unit_square_mesh(4, 4)
        report = check_periodic_mesh(X, axis=0)
        assert report["valid"] is True
        assert report["n_left"] == 5
        assert report["n_right"] == 5

    def test_invalid_mesh(self):
        X = np.array(
            [
                [0.0, 0.0],
                [0.0, 0.5],
                [0.0, 1.0],
                [1.0, 0.0],
                [1.0, 1.0],
            ]
        )
        report = check_periodic_mesh(X, axis=0)
        assert report["valid"] is False


# ---------------------------------------------------------------------------
# Corner fixing
# ---------------------------------------------------------------------------


class TestFixCorner:
    def test_adds_constraints(self):
        T, X = _unit_square_mesh(3, 3)
        C_bc = fix_corner(X, None, dof=2)
        # Should have 2 constraints (x and y at corner node)
        assert C_bc.shape[0] == 2
        assert C_bc.shape[1] == 3
        # Values should be zero
        assert_allclose(C_bc[:, 2], 0.0)

    def test_appends_to_existing(self):
        T, X = _unit_square_mesh(3, 3)
        existing = np.array([[5, 1, 0.0]])
        C_bc = fix_corner(X, existing, dof=2)
        assert C_bc.shape[0] == 3  # 1 existing + 2 new

    def test_corner_node_at_origin(self):
        """Corner node should be the one closest to min(X)."""
        T, X = _unit_square_mesh(3, 3)
        C_bc = fix_corner(X, None, dof=2)
        corner_node = int(C_bc[0, 0])
        # Should be node 1 (0-based index 0) at origin
        assert_allclose(X[corner_node - 1], [0.0, 0.0], atol=1e-12)
