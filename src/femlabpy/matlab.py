from __future__ import annotations

from typing import Any

import numpy as np

from ._helpers import as_float_array


def _legacy_bundle(
    T,
    X,
    G,
    C,
    P=None,
    *,
    dof: int | None = None,
) -> dict[str, np.ndarray | int]:
    """Normalize a legacy FemLab input deck into a typed Python dictionary."""

    coords = as_float_array(X)
    resolved_dof = int(coords.shape[1] if dof is None else dof)
    if P is None:
        loads = np.zeros((0, resolved_dof + 1), dtype=float)
    else:
        loads = as_float_array(P)
    return {
        "T": as_float_array(T).astype(int),
        "X": coords,
        "G": as_float_array(G),
        "C": as_float_array(C),
        "P": loads,
        "dof": resolved_dof,
    }


def _set_axis(ax, limits) -> None:
    """Apply a MATLAB-style axis vector when one is supplied."""

    if limits is None:
        return
    flat = np.asarray(limits, dtype=float).reshape(-1)
    if flat.size == 4:
        ax.axis(flat)
    elif flat.size == 6:
        ax.set_xlim(flat[0], flat[1])
        ax.set_ylim(flat[2], flat[3])
        if hasattr(ax, "set_zlim"):
            ax.set_zlim(flat[4], flat[5])


def canti() -> dict[str, np.ndarray | int]:
    """
    Return the original ``canti.m`` cantilever benchmark input deck.

    Returns
    -------
    dict
        Dictionary with the legacy FemLab fields ``T``, ``X``, ``G``, ``C``,
        ``P``, and ``dof``.
    """

    from .examples.cantilever import cantilever_data

    return cantilever_data()


def flow() -> dict[str, np.ndarray | int]:
    """
    Return the original ``flow.m`` potential-flow benchmark data.

    The returned dictionary contains both the Q4 topology ``T1`` and the T3
    topology ``T2`` used by the legacy ``flowq4.m`` and ``flowt3.m`` drivers.
    """

    from .examples.flow import flow_data

    return flow_data()


def bar01() -> dict[str, np.ndarray | int]:
    """
    Return the packaged input deck corresponding to ``bar01.m``.

    The returned dictionary contains the original truss geometry, material
    table, constraints, loads, and nonlinear load-stepping controls exported
    from the classroom MATLAB benchmark.
    """

    from .examples.legacy_cases import bar01_data

    return bar01_data()


def bar02() -> dict[str, np.ndarray | int]:
    """
    Return the packaged input deck corresponding to ``bar02.m``.

    This is the spatial three-dimensional nonlinear truss benchmark used by
    the legacy ``nlbar.m`` teaching script.
    """

    from .examples.legacy_cases import bar02_data

    return bar02_data()


def bar03() -> dict[str, np.ndarray | int]:
    """
    Return the packaged input deck corresponding to ``bar03.m``.

    Notes
    -----
    ``bar03.m`` is a historically difficult 12-bar benchmark. It is exposed for
    compatibility and regression work even though the legacy load-stepping
    method can exceed its restart guard on this case.
    """

    from .examples.legacy_cases import bar03_data

    return bar03_data()


def square(*, plane_strain: bool = False) -> dict[str, np.ndarray | int]:
    """
    Return the packaged input deck corresponding to ``square.m``.

    Parameters
    ----------
    plane_strain:
        When ``False`` the plane-stress deck used by ``plastps.m`` is returned.
        When ``True`` the plane-strain deck used by ``plastpe.m`` is returned.

    Returns
    -------
    dict
        Legacy FemLab matrices together with the nonlinear stepping controls
        expected by :func:`plastps` and :func:`plastpe`.
    """

    from .examples.legacy_cases import square_data

    return square_data(plane_strain=plane_strain)


def hole(*, plane_strain: bool = False) -> dict[str, np.ndarray | int]:
    """
    Return the packaged input deck corresponding to ``hole.m``.

    Parameters
    ----------
    plane_strain:
        When ``False`` the plane-stress deck used by ``plastps.m`` is returned.
        When ``True`` the plane-strain deck used by ``plastpe.m`` is returned.

    Returns
    -------
    dict
        Legacy FemLab matrices together with the nonlinear stepping controls
        expected by :func:`plastps` and :func:`plastpe`.
    """

    from .examples.legacy_cases import hole_data

    return hole_data(plane_strain=plane_strain)


def elastic(
    T,
    X,
    G,
    C,
    P,
    *,
    dof: int | None = None,
    plot: bool = False,
    scale: float = 5.0,
) -> dict[str, Any]:
    """
    Solve a linear Q4 elasticity problem following the original ``elastic.m`` workflow.

    Parameters
    ----------
    T, X, G, C, P:
        Legacy FemLab topology, coordinates, material data, prescribed
        displacements, and nodal loads.
    dof:
        Degrees of freedom per node. Defaults to ``X.shape[1]``.
    plot:
        When ``True``, return Matplotlib figures for the geometry and the first
        stress component.
    scale:
        Displacement magnification factor used only for the deformed plot.

    Returns
    -------
    dict
        Result dictionary containing ``u``, ``q``, ``S``, ``E``, ``R``, the
        normalized ``data`` dictionary, and an optional ``figures`` list.
    """

    from .boundary import setbc
    from .core import init
    from .elements import kq4e, qq4e
    from .loads import setload
    from .plotting import plotbc, plotelem, plotforces, plotq4
    from .postprocess import reaction

    data = _legacy_bundle(T, X, G, C, P, dof=dof)
    K, p, q = init(data["X"].shape[0], int(data["dof"]), use_sparse=False)
    K = kq4e(K, data["T"], data["X"], data["G"])
    p = setload(p, data["P"])
    K, p, _ = setbc(K, p, data["C"], int(data["dof"]))
    u = np.linalg.solve(K, p)
    q, S, E = qq4e(q, data["T"], data["X"], data["G"], u)
    R = reaction(q, data["C"], int(data["dof"]))

    figures = []
    if plot:
        from matplotlib import pyplot as plt

        fig_geom, ax_geom = plt.subplots()
        plotelem(data["T"], data["X"], ax=ax_geom)
        plotforces(data["T"], data["X"], data["P"], ax=ax_geom)
        plotbc(data["T"], data["X"], data["C"], ax=ax_geom)
        U = u.reshape(data["X"].shape)
        plotelem(data["T"], data["X"] + float(scale) * U, line_style="k--", ax=ax_geom)
        ax_geom.set_title("Linear elastic Q4 response")
        figures.append(fig_geom)

        fig_stress, ax_stress = plt.subplots()
        plotq4(data["T"], data["X"], S, 1, ax=ax_stress)
        ax_stress.set_title("Stress component 1")
        figures.append(fig_stress)

    return {"u": u, "q": q, "S": S, "E": E, "R": R, "data": data, "figures": figures}


def flowq4(
    T1=None,
    X=None,
    G=None,
    C=None,
    P=None,
    *,
    dof: int = 1,
    plot: bool = False,
) -> dict[str, Any]:
    """
    Solve a Q4 potential problem following the original ``flowq4.m`` driver.

    Parameters
    ----------
    T1, X, G, C, P:
        Legacy FemLab topology, coordinates, material data, prescribed
        potentials, and optional nodal loads. When omitted, the packaged
        ``flow.m`` benchmark data is used.
    dof:
        Degrees of freedom per node. The legacy flow examples use ``dof=1``.
    plot:
        When ``True``, return a single four-panel Matplotlib figure matching the
        original classroom driver.
    """

    if T1 is None or X is None or G is None or C is None:
        data = flow()
        T1 = data["T1"]
        X = data["X"]
        G = data["G"]
        C = data["C"]
        P = data.get("P")
        dof = int(data["dof"])
    data = _legacy_bundle(T1, X, G, C, P, dof=dof)
    result = _flow_driver(
        data["T"],
        data["X"],
        data["G"],
        data["C"],
        data["P"],
        dof=int(data["dof"]),
        plot=plot,
        triangle=False,
    )
    return {**result, "data": {"T1": data["T"], "X": data["X"], "G": data["G"], "C": data["C"], "P": data["P"], "dof": data["dof"]}}


def flowt3(
    T2=None,
    X=None,
    G=None,
    C=None,
    P=None,
    *,
    dof: int = 1,
    plot: bool = False,
) -> dict[str, Any]:
    """
    Solve a T3 potential problem following the original ``flowt3.m`` driver.

    Parameters
    ----------
    T2, X, G, C, P:
        Legacy FemLab topology, coordinates, material data, prescribed
        potentials, and optional nodal loads. When omitted, the packaged
        ``flow.m`` benchmark data is used.
    dof:
        Degrees of freedom per node. The legacy flow examples use ``dof=1``.
    plot:
        When ``True``, return a single four-panel Matplotlib figure matching the
        original classroom driver.
    """

    if T2 is None or X is None or G is None or C is None:
        data = flow()
        T2 = data["T2"]
        X = data["X"]
        G = data["G"]
        C = data["C"]
        P = data.get("P")
        dof = int(data["dof"])
    data = _legacy_bundle(T2, X, G, C, P, dof=dof)
    result = _flow_driver(
        data["T"],
        data["X"],
        data["G"],
        data["C"],
        data["P"],
        dof=int(data["dof"]),
        plot=plot,
        triangle=True,
    )
    return {**result, "data": {"T2": data["T"], "X": data["X"], "G": data["G"], "C": data["C"], "P": data["P"], "dof": data["dof"]}}


def _flow_driver(T, X, G, C, P, *, dof: int, plot: bool, triangle: bool) -> dict[str, Any]:
    """Internal helper implementing the classroom `flowq4.m` and `flowt3.m` drivers."""

    from .boundary import setbc
    from .core import init
    from .elements import kq4p, kt3p, qq4p, qt3p
    from .loads import setload
    from .plotting import plotelem, plotq4, plott3, plotu

    data = _legacy_bundle(T, X, G, C, P, dof=dof)
    K, p, q = init(data["X"].shape[0], int(data["dof"]), use_sparse=False)
    if triangle:
        K = kt3p(K, data["T"], data["X"], data["G"])
    else:
        K = kq4p(K, data["T"], data["X"], data["G"])
    if data["P"].size:
        p = setload(p, data["P"])
    K, p, _ = setbc(K, p, data["C"], int(data["dof"]))
    u = np.linalg.solve(K, p)
    if triangle:
        q, S, E = qt3p(q, data["T"], data["X"], data["G"], u)
    else:
        q, S, E = qq4p(q, data["T"], data["X"], data["G"], u)

    figures = []
    if plot:
        from matplotlib import pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        plotelem(data["T"], data["X"], ax=axes[0, 0])
        axes[0, 0].set_title("Element mesh")
        plotu(data["T"], data["X"], u, ax=axes[0, 1])
        axes[0, 1].set_title("Potentials")
        if triangle:
            plott3(data["T"], data["X"], E, 1, ax=axes[1, 0])
            plott3(data["T"], data["X"], E, 2, ax=axes[1, 1])
        else:
            plotq4(data["T"], data["X"], E, 1, ax=axes[1, 0])
            plotq4(data["T"], data["X"], E, 2, ax=axes[1, 1])
        axes[1, 0].set_title("Ex")
        axes[1, 1].set_title("Ey")
        figures.append(fig)

    return {"u": u, "q": q, "S": S, "E": E, "figures": figures}


def nlbar(
    T,
    X,
    G,
    C,
    P,
    *,
    no_loadsteps: int,
    i_max: int,
    i_d: int,
    plotdof: int,
    tol: float = 1.0e-8,
    plot: bool = False,
    plotaxis=None,
    elaxis=None,
) -> dict[str, Any]:
    """
    Solve a nonlinear truss problem through the legacy ``nlbar.m`` driver logic.

    Parameters
    ----------
    T, X, G, C, P:
        Legacy FemLab topology, coordinates, material data, displacement
        constraints, and nodal loads.
    no_loadsteps, i_max, i_d, plotdof:
        Legacy load-stepping controls used by the orthogonal residual method.
        ``plotdof`` is interpreted as the original 1-based degree-of-freedom
        index.
    tol:
        Convergence tolerance used by :func:`femlabpy.solve_nlbar`.
    plot:
        When ``True``, return a load-displacement figure and a deformed-geometry
        figure.
    plotaxis, elaxis:
        Optional MATLAB-style axis vectors kept for API compatibility.
    """

    from .plotting import plotelem
    from .solvers import solve_nlbar

    data = _legacy_bundle(T, X, G, C, P)
    result = solve_nlbar(
        data["X"],
        data["T"],
        data["G"],
        data["C"],
        data["P"],
        no_loadsteps=no_loadsteps,
        i_max=i_max,
        i_d=i_d,
        tol=tol,
        plotdof=plotdof,
    )

    figures = []
    if plot:
        from matplotlib import pyplot as plt

        fig_path, ax_path = plt.subplots()
        ax_path.plot(-result["U_path"][:, 0], -result["F_path"][:, 0], "k-o", markersize=3)
        ax_path.set_xlabel("Displacement")
        ax_path.set_ylabel("Load")
        ax_path.set_title("Nonlinear bar load-displacement path")
        _set_axis(ax_path, plotaxis)
        figures.append(fig_path)

        fig_geom, ax_geom = plt.subplots()
        plotelem(data["T"], data["X"], ax=ax_geom)
        U = result["u"].reshape(data["X"].shape)
        plotelem(data["T"], data["X"] + U, line_style="k--", ax=ax_geom)
        ax_geom.set_title("Nonlinear bar geometry")
        _set_axis(ax_geom, elaxis)
        figures.append(fig_geom)

    return {**result, "data": data, "figures": figures}


def plastps(
    T,
    X,
    G,
    C,
    P,
    *,
    no_loadsteps: int,
    i_max: int,
    i_d: int,
    plotdof: int,
    tol: float = 1.0e-8,
    mattype: int = 1,
    plot: bool = False,
    strainaxis=None,
    elaxis=None,
) -> dict[str, Any]:
    """
    Solve a plane-stress elastoplastic Q4 problem following ``plastps.m``.

    Parameters
    ----------
    T, X, G, C, P:
        Legacy FemLab topology, coordinates, material data, displacement
        constraints, and nodal loads.
    no_loadsteps, i_max, i_d, plotdof:
        Orthogonal-residual load-stepping controls. ``plotdof`` is interpreted
        in the original 1-based numbering.
    tol:
        Convergence tolerance passed to :func:`femlabpy.solve_plastic`.
    mattype:
        ``1`` for von Mises and ``2`` for Drucker-Prager, matching FemLab.
    plot:
        When ``True``, return a load-displacement figure and an equivalent
        plastic-strain contour plot in the deformed geometry.
    strainaxis, elaxis:
        Optional MATLAB-style axis vectors kept for API compatibility.
    """

    return _plast_driver(
        T,
        X,
        G,
        C,
        P,
        no_loadsteps=no_loadsteps,
        i_max=i_max,
        i_d=i_d,
        plotdof=plotdof,
        tol=tol,
        mattype=mattype,
        plane_strain=False,
        plot=plot,
        strainaxis=strainaxis,
        elaxis=elaxis,
    )


def plastpe(
    T,
    X,
    G,
    C,
    P,
    *,
    no_loadsteps: int,
    i_max: int,
    i_d: int,
    plotdof: int,
    tol: float = 1.0e-8,
    mattype: int = 1,
    plot: bool = False,
    strainaxis=None,
    elaxis=None,
) -> dict[str, Any]:
    """
    Solve a plane-strain elastoplastic Q4 problem following ``plastpe.m``.

    Parameters
    ----------
    T, X, G, C, P:
        Legacy FemLab topology, coordinates, material data, displacement
        constraints, and nodal loads.
    no_loadsteps, i_max, i_d, plotdof:
        Orthogonal-residual load-stepping controls. ``plotdof`` is interpreted
        in the original 1-based numbering.
    tol:
        Convergence tolerance passed to :func:`femlabpy.solve_plastic`.
    mattype:
        ``1`` for von Mises and ``2`` for Drucker-Prager, matching FemLab.
    plot:
        When ``True``, return a load-displacement figure and an equivalent
        plastic-strain contour plot in the deformed geometry.
    strainaxis, elaxis:
        Optional MATLAB-style axis vectors kept for API compatibility.
    """

    return _plast_driver(
        T,
        X,
        G,
        C,
        P,
        no_loadsteps=no_loadsteps,
        i_max=i_max,
        i_d=i_d,
        plotdof=plotdof,
        tol=tol,
        mattype=mattype,
        plane_strain=True,
        plot=plot,
        strainaxis=strainaxis,
        elaxis=elaxis,
    )


def _plast_driver(
    T,
    X,
    G,
    C,
    P,
    *,
    no_loadsteps: int,
    i_max: int,
    i_d: int,
    plotdof: int,
    tol: float,
    mattype: int,
    plane_strain: bool,
    plot: bool,
    strainaxis,
    elaxis,
) -> dict[str, Any]:
    """Internal helper shared by the legacy ``plastps`` and ``plastpe`` wrappers."""

    from .plotting import plotq4
    from .solvers import solve_plastic

    data = _legacy_bundle(T, X, G, C, P)
    result = solve_plastic(
        data["X"],
        data["T"],
        data["G"],
        data["C"],
        data["P"],
        no_loadsteps=no_loadsteps,
        i_max=i_max,
        i_d=i_d,
        tol=tol,
        plotdof=plotdof,
        plane_strain=plane_strain,
        material_type=mattype,
    )

    figures = []
    if plot:
        from matplotlib import pyplot as plt

        fig_path, ax_path = plt.subplots()
        ax_path.plot(result["U_path"][:, 0], result["F_path"][:, 0], "k-o", markersize=3)
        ax_path.set_xlabel("Displacement")
        ax_path.set_ylabel("Load")
        ax_path.set_title("Elastoplastic load-displacement path")
        figures.append(fig_path)

        fig_field, ax_field = plt.subplots(figsize=(7, 5))
        U = result["u"].reshape(data["X"].shape)
        comp = 5 if plane_strain else 4
        plotq4(data["T"], data["X"] + U, result["E"], comp, ax=ax_field)
        ax_field.set_title("Equivalent plastic strain")
        _set_axis(ax_field, elaxis)
        if strainaxis is not None:
            clim = np.asarray(strainaxis, dtype=float).reshape(-1)
            if clim.size == 2 and ax_field.collections:
                ax_field.collections[0].set_clim(clim[0], clim[1])
        figures.append(fig_field)

    return {**result, "data": data, "figures": figures}


__all__ = [
    "bar01",
    "bar02",
    "bar03",
    "canti",
    "elastic",
    "flow",
    "flowq4",
    "flowt3",
    "hole",
    "nlbar",
    "plastpe",
    "plastps",
    "square",
]
