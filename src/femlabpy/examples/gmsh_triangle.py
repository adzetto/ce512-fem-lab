from __future__ import annotations

from importlib.resources import as_file, files
from pathlib import Path

import numpy as np

from .. import (
    init,
    kt3e,
    load_gmsh,
    plotbc,
    plotelem,
    plotforces,
    plott3,
    qt3e,
    reaction,
    setload,
    solve_lag,
)


def default_mesh_path() -> Path:
    """Return the repository path to the packaged classroom Gmsh mesh."""
    return (
        Path(__file__).resolve().parents[3]
        / "src"
        / "femlabpy"
        / "data"
        / "meshes"
        / "deneme.msh"
    )


def gmsh_triangle_data(mesh_path: str | Path | None = None):
    """
    Return the packaged Gmsh triangle benchmark in FemLab array form.

    Parameters
    ----------
    mesh_path:
        Optional override for the `.msh` file. When omitted, the mesh bundled
        with `femlabpy` is used so that the example remains functional after
        installation from a wheel.
    """

    if mesh_path is None:
        packaged_mesh = files("femlabpy.data").joinpath("meshes").joinpath("deneme.msh")
        with as_file(packaged_mesh) as resolved_mesh:
            mesh = load_gmsh(resolved_mesh)
    else:
        mesh = load_gmsh(mesh_path)
    T = mesh.triangles.astype(int)
    X = mesh.positions[:, :2]
    G = np.array([[2.0e8, 0.3, 1.0], [0.7e8, 0.23, 1.0]], dtype=float)
    C = np.array(
        [
            [5, 1, 0],
            [7, 1, 0],
            [8, 1, 0],
            [8, 2, 0],
            [9, 2, 0],
            [10, 2, 0],
            [9, 1, 0],
            [10, 1, 0],
            [11, 1, 0],
            [6, 1, 0],
        ],
        dtype=float,
    )
    P = np.array([[25, 0, -0.05], [24, 0, -0.1], [22, 0, -0.05]], dtype=float)
    return {"mesh": mesh, "T": T, "X": X, "G": G, "C": C, "P": P, "dof": 2}


def run_gmsh_triangle(
    mesh_path: str | Path | None = None,
    *,
    displacement_scale: float = 1000.0,
    plot: bool = False,
):
    """
    Solve the bundled triangular Gmsh example with CST elements.

    Parameters
    ----------
    mesh_path:
        Optional path to a custom Gmsh mesh.
    displacement_scale:
        Visual scale factor used only for the optional deformed-geometry plot.
    plot:
        When `True`, return Matplotlib figures in `result["figures"]`.
    """

    data = gmsh_triangle_data(mesh_path)
    K, p, q = init(data["X"].shape[0], data["dof"], use_sparse=False)
    K = kt3e(K, data["T"], data["X"], data["G"])
    p = setload(p, data["P"])
    u = solve_lag(K, p, data["C"], data["dof"])
    q, S, E = qt3e(q, data["T"], data["X"], data["G"], u)
    R = reaction(q, data["C"], data["dof"])

    figures = []
    if plot:
        from matplotlib import pyplot as plt

        fig1, ax1 = plt.subplots()
        plotelem(data["T"], data["X"], ax=ax1)
        plotforces(data["T"], data["X"], data["P"], ax=ax1)
        plotbc(data["T"], data["X"], data["C"], ax=ax1)
        U = u.reshape(data["X"].shape)
        plotelem(
            data["T"], data["X"] + displacement_scale * U, line_style="c--", ax=ax1
        )
        figures.append(fig1)

        fig2, ax2 = plt.subplots()
        plott3(data["T"], data["X"], S, 1, ax=ax2)
        figures.append(fig2)

    return {"u": u, "q": q, "S": S, "E": E, "R": R, "data": data, "figures": figures}
