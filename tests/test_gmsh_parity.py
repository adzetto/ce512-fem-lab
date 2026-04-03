from __future__ import annotations

import tempfile
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
        assert np.array_equal(
            mesh.triangles[:, -1], mesh.element_tags[triangle_rows, 0]
        )


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


def test_load_gmsh_accepts_modern_v4_meshes_when_sdk_is_available():
    gmsh = pytest.importorskip("gmsh")
    temp_root = REPO / ".pytest_tmp"
    temp_root.mkdir(exist_ok=True)

    with tempfile.TemporaryDirectory(dir=temp_root) as temp_dir:
        gmsh.initialize(readConfigFiles=False)
        try:
            gmsh.option.setNumber("General.Terminal", 0)
            gmsh.model.add("unit_square")
            p1 = gmsh.model.geo.addPoint(0.0, 0.0, 0.0, 0.2)
            p2 = gmsh.model.geo.addPoint(1.0, 0.0, 0.0, 0.2)
            p3 = gmsh.model.geo.addPoint(1.0, 1.0, 0.0, 0.2)
            p4 = gmsh.model.geo.addPoint(0.0, 1.0, 0.0, 0.2)
            l1 = gmsh.model.geo.addLine(p1, p2)
            l2 = gmsh.model.geo.addLine(p2, p3)
            l3 = gmsh.model.geo.addLine(p3, p4)
            l4 = gmsh.model.geo.addLine(p4, p1)
            curve_loop = gmsh.model.geo.addCurveLoop([l1, l2, l3, l4])
            surface = gmsh.model.geo.addPlaneSurface([curve_loop])
            gmsh.model.geo.synchronize()
            gmsh.model.addPhysicalGroup(2, [surface], tag=1, name="domain")
            gmsh.model.mesh.generate(2)

            mesh_path = Path(temp_dir) / "unit_square_v41.msh"
            gmsh.option.setNumber("Mesh.Binary", 0)
            gmsh.option.setNumber("Mesh.MshFileVersion", 4.1)
            gmsh.write(str(mesh_path))
            gmsh.clear()
        finally:
            if bool(gmsh.isInitialized()):
                gmsh.finalize()

        mesh = load_gmsh(mesh_path)
        assert mesh.positions.shape[1] == 3
        assert mesh.triangles.shape[0] > 0
        assert np.all(mesh.triangles[:, -1] == 1)
