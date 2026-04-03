from __future__ import annotations

from importlib.resources import files

import numpy as np

from ..plotting import plotelem, plotq4
from ..solvers import solve_nlbar, solve_plastic


def _case_directory(case_name: str):
    return files("femlabpy.data").joinpath("cases", case_name)


def _load_case(case_name: str) -> dict[str, np.ndarray | int]:
    """Load a packaged legacy FemLab case exported as TSV matrices."""

    case_dir = _case_directory(case_name)
    data: dict[str, np.ndarray | int] = {}
    for entry in case_dir.iterdir():
        if entry.is_file() and entry.name.endswith(".tsv"):
            with entry.open("r", encoding="utf-8") as handle:
                data[entry.name[:-4]] = np.loadtxt(handle, dtype=float, ndmin=2)
    if not data:
        raise FileNotFoundError(f"No packaged case data was found for '{case_name}'.")
    if "dof" in data:
        data["dof"] = int(float(np.asarray(data["dof"]).reshape(-1)[0]))
    else:
        data["dof"] = int(np.asarray(data["X"]).shape[1])
    return data


def bar01_data():
    """
    Return the original `bar01.m` nonlinear truss benchmark data.

    The packaged matrices include the geometry, material table, boundary
    conditions, loads, and all nonlinear stepping controls used by the legacy
    MATLAB driver.
    """

    return _load_case("bar01_nlbar")


def bar02_data():
    """
    Return the original `bar02.m` nonlinear truss benchmark data.

    This benchmark is the three-dimensional companion to :func:`bar01_data`.
    """

    return _load_case("bar02_nlbar")


def bar03_data():
    """
    Return the original `bar03.m` 12-bar truss benchmark data.

    The case is retained for parity and regression testing even though the
    original continuation strategy can diverge on this benchmark.
    """

    return _load_case("bar03_nlbar")


def square_data(*, plane_strain: bool = False):
    """
    Return the packaged `square.m` data for plane stress or plane strain.

    Besides the standard FemLab arrays, the packaged deck also contains the
    legacy plotting-control vectors and nonlinear iteration settings.
    """

    case_name = "square_plastpe" if plane_strain else "square_plastps"
    return _load_case(case_name)


def hole_data(*, plane_strain: bool = False):
    """
    Return the packaged `hole.m` data for plane stress or plane strain.

    Besides the standard FemLab arrays, the packaged deck also contains the
    legacy plotting-control vectors and nonlinear iteration settings.
    """

    case_name = "hole_plastpe" if plane_strain else "hole_plastps"
    return _load_case(case_name)


def _plot_bar_solution(data: dict[str, np.ndarray | int], result: dict[str, np.ndarray]):
    from matplotlib import pyplot as plt

    figures = []
    X = np.asarray(data["X"], dtype=float)
    T = np.asarray(data["T"], dtype=int)
    U = result["u"].reshape(X.shape)

    if X.shape[1] == 3:
        fig_geom = plt.figure()
        ax_geom = fig_geom.add_subplot(111, projection="3d")
    else:
        fig_geom, ax_geom = plt.subplots()
    plotelem(T, X, ax=ax_geom)
    plotelem(T, X + U, line_style="c--", ax=ax_geom)
    ax_geom.set_title("Undeformed and deformed truss")
    if "elaxis" in data:
        flat = np.asarray(data["elaxis"], dtype=float).reshape(-1)
        if flat.size == 4:
            ax_geom.axis(flat)
        elif flat.size == 6 and hasattr(ax_geom, "set_zlim"):
            ax_geom.set_xlim(flat[0], flat[1])
            ax_geom.set_ylim(flat[2], flat[3])
            ax_geom.set_zlim(flat[4], flat[5])
    figures.append(fig_geom)

    fig_path, ax_path = plt.subplots()
    ax_path.plot(result["U_path"][:, 0], result["F_path"][:, 0], "k-o", markersize=3)
    ax_path.set_xlabel("Displacement")
    ax_path.set_ylabel("Load")
    ax_path.set_title("Load-displacement path")
    if "plotaxis" in data:
        flat = np.asarray(data["plotaxis"], dtype=float).reshape(-1)
        if flat.size == 4:
            ax_path.axis(flat)
    figures.append(fig_path)
    return figures


def _plot_plastic_solution(
    data: dict[str, np.ndarray | int],
    result: dict[str, np.ndarray],
    *,
    plane_strain: bool,
):
    from matplotlib import pyplot as plt

    figures = []
    X = np.asarray(data["X"], dtype=float)
    T = np.asarray(data["T"], dtype=int)
    U = result["u"].reshape(X.shape)

    fig_field, ax_field = plt.subplots(figsize=(7, 5))
    plotq4(T, X + U, result["E"], 5 if plane_strain else 4, ax=ax_field)
    ax_field.set_title("Equivalent plastic strain")
    if "elaxis" in data:
        flat = np.asarray(data["elaxis"], dtype=float).reshape(-1)
        if flat.size == 4:
            ax_field.axis(flat)
    if "strainaxis" in data:
        flat = np.asarray(data["strainaxis"], dtype=float).reshape(-1)
        if flat.size == 2 and ax_field.collections:
            ax_field.collections[0].set_clim(flat[0], flat[1])
    figures.append(fig_field)

    fig_path, ax_path = plt.subplots()
    ax_path.plot(result["U_path"][:, 0], result["F_path"][:, 0], "k-o", markersize=3)
    ax_path.set_xlabel("Displacement")
    ax_path.set_ylabel("Load")
    ax_path.set_title("Load-displacement path")
    figures.append(fig_path)
    return figures


def _bundle_result(data: dict[str, np.ndarray | int], result: dict[str, np.ndarray], figures):
    return {**result, "data": data, "figures": figures}


def run_bar01_nlbar(*, plot: bool = False):
    """
    Solve the original `bar01.m` example through the legacy `nlbar` driver.

    Parameters
    ----------
    plot:
        When ``True``, return the load-displacement path and deformed truss
        figures in ``result["figures"]``.
    """

    data = bar01_data()
    result = solve_nlbar(
        data["X"],
        data["T"],
        data["G"],
        data["C"],
        data["P"],
        no_loadsteps=int(data["no_loadsteps"][0, 0]),
        i_max=int(data["i_max"][0, 0]),
        i_d=int(data["i_d"][0, 0]),
        tol=float(data["TOL"][0, 0]),
        plotdof=int(data["plotdof"][0, 0]),
    )
    figures = _plot_bar_solution(data, result) if plot else []
    return _bundle_result(data, result, figures)


def run_bar02_nlbar(*, plot: bool = False):
    """
    Solve the original `bar02.m` example through the legacy `nlbar` driver.

    Parameters
    ----------
    plot:
        When ``True``, return the load-displacement path and deformed truss
        figures in ``result["figures"]``.
    """

    data = bar02_data()
    result = solve_nlbar(
        data["X"],
        data["T"],
        data["G"],
        data["C"],
        data["P"],
        no_loadsteps=int(data["no_loadsteps"][0, 0]),
        i_max=int(data["i_max"][0, 0]),
        i_d=int(data["i_d"][0, 0]),
        tol=float(data["TOL"][0, 0]),
        plotdof=int(data["plotdof"][0, 0]),
    )
    figures = _plot_bar_solution(data, result) if plot else []
    return _bundle_result(data, result, figures)


def run_bar03_nlbar(*, plot: bool = False):
    """
    Solve the original `bar03.m` example through the legacy `nlbar` driver.

    Notes
    -----
    This 12-bar benchmark is numerically difficult and may exceed the restart
    guard, matching the behavior observed in the legacy solver comparisons.
    """

    data = bar03_data()
    result = solve_nlbar(
        data["X"],
        data["T"],
        data["G"],
        data["C"],
        data["P"],
        no_loadsteps=int(data["no_loadsteps"][0, 0]),
        i_max=int(data["i_max"][0, 0]),
        i_d=int(data["i_d"][0, 0]),
        tol=float(data["TOL"][0, 0]),
        plotdof=int(data["plotdof"][0, 0]),
    )
    figures = _plot_bar_solution(data, result) if plot else []
    return _bundle_result(data, result, figures)


def run_square_plastps(*, plot: bool = False):
    """
    Solve the plane-stress `square.m` elastoplastic benchmark.

    Parameters
    ----------
    plot:
        When ``True``, return the deformed equivalent-plastic-strain field and
        load-displacement path figures in ``result["figures"]``.
    """

    data = square_data(plane_strain=False)
    result = solve_plastic(
        data["X"],
        data["T"],
        data["G"],
        data["C"],
        data["P"],
        no_loadsteps=int(data["no_loadsteps"][0, 0]),
        i_max=int(data["i_max"][0, 0]),
        i_d=int(data["i_d"][0, 0]),
        tol=float(data["TOL"][0, 0]),
        plotdof=int(data["plotdof"][0, 0]),
        plane_strain=False,
    )
    figures = _plot_plastic_solution(data, result, plane_strain=False) if plot else []
    return _bundle_result(data, result, figures)


def run_square_plastpe(*, plot: bool = False):
    """
    Solve the plane-strain `square.m` elastoplastic benchmark.

    Parameters
    ----------
    plot:
        When ``True``, return the deformed equivalent-plastic-strain field and
        load-displacement path figures in ``result["figures"]``.
    """

    data = square_data(plane_strain=True)
    result = solve_plastic(
        data["X"],
        data["T"],
        data["G"],
        data["C"],
        data["P"],
        no_loadsteps=int(data["no_loadsteps"][0, 0]),
        i_max=int(data["i_max"][0, 0]),
        i_d=int(data["i_d"][0, 0]),
        tol=float(data["TOL"][0, 0]),
        plotdof=int(data["plotdof"][0, 0]),
        plane_strain=True,
    )
    figures = _plot_plastic_solution(data, result, plane_strain=True) if plot else []
    return _bundle_result(data, result, figures)


def run_hole_plastps(*, plot: bool = False):
    """
    Solve the plane-stress `hole.m` elastoplastic benchmark.

    Parameters
    ----------
    plot:
        When ``True``, return the deformed equivalent-plastic-strain field and
        load-displacement path figures in ``result["figures"]``.
    """

    data = hole_data(plane_strain=False)
    result = solve_plastic(
        data["X"],
        data["T"],
        data["G"],
        data["C"],
        data["P"],
        no_loadsteps=int(data["no_loadsteps"][0, 0]),
        i_max=int(data["i_max"][0, 0]),
        i_d=int(data["i_d"][0, 0]),
        tol=float(data["TOL"][0, 0]),
        plotdof=int(data["plotdof"][0, 0]),
        plane_strain=False,
    )
    figures = _plot_plastic_solution(data, result, plane_strain=False) if plot else []
    return _bundle_result(data, result, figures)


def run_hole_plastpe(*, plot: bool = False):
    """
    Solve the plane-strain `hole.m` elastoplastic benchmark.

    Parameters
    ----------
    plot:
        When ``True``, return the deformed equivalent-plastic-strain field and
        load-displacement path figures in ``result["figures"]``.
    """

    data = hole_data(plane_strain=True)
    result = solve_plastic(
        data["X"],
        data["T"],
        data["G"],
        data["C"],
        data["P"],
        no_loadsteps=int(data["no_loadsteps"][0, 0]),
        i_max=int(data["i_max"][0, 0]),
        i_d=int(data["i_d"][0, 0]),
        tol=float(data["TOL"][0, 0]),
        plotdof=int(data["plotdof"][0, 0]),
        plane_strain=True,
    )
    figures = _plot_plastic_solution(data, result, plane_strain=True) if plot else []
    return _bundle_result(data, result, figures)


__all__ = [
    "bar01_data",
    "bar02_data",
    "bar03_data",
    "hole_data",
    "run_bar01_nlbar",
    "run_bar02_nlbar",
    "run_bar03_nlbar",
    "run_hole_plastpe",
    "run_hole_plastps",
    "run_square_plastpe",
    "run_square_plastps",
    "square_data",
]
