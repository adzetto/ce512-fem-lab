from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from femlabpy import load_gmsh, load_gmsh2, reaction


REPO = Path(__file__).resolve().parents[1]
MESH_PATH = REPO / "src" / "femlabpy" / "data" / "meshes" / "deneme.msh"


def test_load_gmsh_exposes_legacy_aliases_and_physical_tags():
    mesh = load_gmsh(MESH_PATH)
    assert np.allclose(mesh.POS, mesh.positions)
    assert np.allclose(mesh.MIN, mesh.bounds_min)
    assert np.allclose(mesh.MAX, mesh.bounds_max)
    assert np.array_equal(mesh.TRIANGLES, mesh.triangles)
    assert mesh.nbTriangles == mesh.triangles.shape[0]
    assert mesh.nbType[1] == mesh.triangles.shape[0]
    if mesh.triangles.shape[0] > 0:
        triangle_rows = np.flatnonzero(mesh.element_infos[:, 1] == 2)
        assert np.array_equal(mesh.triangles[:, -1], mesh.element_tags[triangle_rows, 0])


def test_load_gmsh2_minus_one_skips_explicit_type_arrays():
    mesh = load_gmsh2(MESH_PATH, -1)
    assert mesh.nbElm == mesh.element_infos.shape[0]
    assert mesh.nbType.sum() > 0
    assert mesh.triangles.size == 0
    with pytest.raises(AttributeError):
        _ = mesh.TRIANGLES


def test_reaction_component_matches_legacy_constraint_row_indexing():
    q = np.array([[10.0], [20.0], [30.0], [40.0]], dtype=float)
    C = np.array([[1, 1, 0.0], [1, 2, 0.0], [2, 2, 0.0]], dtype=float)
    R = reaction(q, C, 2, comp=2)
    np.testing.assert_allclose(R, np.array([[2.0, 20.0], [3.0, 40.0]]))
