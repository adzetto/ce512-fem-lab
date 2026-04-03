from __future__ import annotations

import numpy as np

from ._helpers import (
    as_float_array,
    solve_legacy_symmetric_system,
    solve_linear_system,
)
from .boundary import rnorm, setbc
from .elements import kbar, kq4epe, kq4eps, qbar, qq4epe, qq4eps
from .loads import setload
from .postprocess import reaction

try:
    import scipy.sparse as sp
except ImportError:  # pragma: no cover
    sp = None


def _scalar(value: object) -> float:
    return float(as_float_array(value).reshape(-1)[0])


def _column(values: list[float]) -> np.ndarray:
    return np.asarray(values, dtype=float).reshape(-1, 1)


def _solve_plastic_system(matrix, rhs, *, plane_strain: bool):
    if plane_strain:
        return solve_legacy_symmetric_system(matrix, rhs)
    return solve_linear_system(matrix, rhs)


def solve_nlbar(
    X,
    T,
    G,
    C,
    P,
    *,
    no_loadsteps: int,
    i_max: int,
    i_d: int,
    tol: float,
    plotdof: int,
):
    """
    Solve the legacy nonlinear bar examples with the orthogonal residual method.

    The implementation follows the original ``nlbar.m`` control flow while using
    vectorized element kernels for bar stiffness and internal-force assembly.

    Parameters
    ----------
    X, T, G, C, P:
        Legacy FemLab node coordinates, topology, material table, prescribed
        displacements, and nodal loads.
    no_loadsteps, i_max, i_d:
        Legacy stepping parameters controlling the outer load increments and the
        inner equilibrium iterations.
    tol:
        Relative residual tolerance used in the orthogonal residual test.
    plotdof:
        One-based response degree of freedom stored in the returned
        load-displacement path.

    Returns
    -------
    dict
        Dictionary containing the converged displacement vector, internal force
        vector, stresses, strains, reactions, final external load vector, and
        the full load-displacement history.
    """

    coords = as_float_array(X)
    topology = as_float_array(T).astype(int)
    materials = as_float_array(G)
    constraints = as_float_array(C)
    loads = as_float_array(P)

    dof = coords.shape[1]
    ndof = coords.shape[0] * dof
    response_dof = int(plotdof) - 1

    u = np.zeros((ndof, 1), dtype=float)
    du = np.zeros((ndof, 1), dtype=float)
    f = np.zeros((ndof, 1), dtype=float)
    df = setload(np.zeros((ndof, 1), dtype=float), loads)

    U_path = [0.0]
    F_path = [0.0]

    n = 1
    i = int(i_d)
    restarts = 0
    max_restarts = max(1, int(no_loadsteps) * int(i_max))

    while n <= int(no_loadsteps):
        if restarts > max_restarts:
            raise RuntimeError("Nonlinear bar solver exceeded the restart guard.")

        if i < int(i_max):
            K = np.zeros((ndof, ndof), dtype=float)
            K = kbar(K, topology, coords, materials, u)
            Kt, df, _ = setbc(K.copy(), df, constraints, dof)
            du0 = np.linalg.solve(Kt, df)
            if not np.all(np.isfinite(du0)):
                raise RuntimeError("NaN/Inf encountered in the nonlinear bar solver.")

            if float((du.T @ du0).item()) < 0.0:
                df = -df
                du0 = -du0

            if n == 1:
                l0 = float(np.linalg.norm(du0))
                l = l0
                l_max = 2.0 * l0
            else:
                l = float(np.linalg.norm(du))
                l0 = float(np.linalg.norm(du0))

        if i_d <= i < int(i_max):
            du = min(l / l0, l_max / l0) * du0
        elif i < i_d:
            du = min(2.0 * l / l0, l_max / l0) * du0
        else:
            du0 = 0.5 * du0
            du = du0.copy()
            restarts += 1

        xi = 0.0
        for i in range(1, int(i_max) + 1):
            q = np.zeros((ndof, 1), dtype=float)
            q, S, E = qbar(q, topology, coords, materials, u + du)
            if not np.all(np.isfinite(q)):
                raise RuntimeError("NaN/Inf encountered in the nonlinear bar solver.")

            dq = q - f
            xi = float(((dq.T @ du) / (df.T @ du)).item())
            residual = -dq + xi * df
            if rnorm(residual, constraints, dof) < float(tol) * rnorm(df, constraints, dof):
                break

            Kt, residual_bc, _ = setbc(K.copy(), residual, constraints, dof)
            delta_u = np.linalg.solve(Kt, residual_bc)
            du = du + delta_u

        if i >= int(i_max):
            continue

        f = f + xi * df
        u = u + du
        U_path.append(float(u[response_dof, 0]))
        F_path.append(float(f[response_dof, 0]))
        n += 1

    q = np.zeros((ndof, 1), dtype=float)
    q, S, E = qbar(q, topology, coords, materials, u)
    R = reaction(q, constraints, dof)
    return {
        "u": u,
        "q": q,
        "S": S,
        "E": E,
        "R": R,
        "f": f,
        "U_path": _column(U_path),
        "F_path": _column(F_path),
    }


def solve_plastic(
    X,
    T,
    G,
    C,
    P,
    *,
    no_loadsteps: int,
    i_max: int,
    i_d: int,
    tol: float,
    plotdof: int,
    plane_strain: bool,
    material_type: int = 1,
):
    """
    Solve the legacy Q4 elastoplastic examples with orthogonal residual iterations.

    Parameters follow the original FemLab classroom drivers ``plastps.m`` and
    ``plastpe.m``. The Gauss-point constitutive updates remain in the element
    routines; this function orchestrates the load stepping and equilibrium loop.

    Parameters
    ----------
    X, T, G, C, P:
        Legacy FemLab node coordinates, topology, material table, prescribed
        displacements, and nodal loads.
    no_loadsteps, i_max, i_d:
        Legacy stepping parameters controlling the outer load increments and the
        inner equilibrium iterations.
    tol:
        Relative residual tolerance used in the orthogonal residual test.
    plotdof:
        One-based response degree of freedom stored in the returned
        load-displacement path.
    plane_strain:
        Selects the plane-strain constitutive update when ``True`` and the
        plane-stress update when ``False``. For the small legacy plane-strain
        classroom systems, the linear solves use a symmetry-aware dense
        fallback that better reproduces MATLAB's historical ``\\`` behavior.
    material_type:
        ``1`` for von Mises and ``2`` for Drucker-Prager, matching the original
        FemLab element routines.

    Returns
    -------
    dict
        Dictionary containing the converged displacement vector, internal force
        vector, element stress and plastic-strain state, reactions, final
        external load vector, and the full load-displacement history.
    """

    coords = as_float_array(X)
    topology = as_float_array(T).astype(int)
    materials = as_float_array(G)
    constraints = as_float_array(C)
    loads = as_float_array(P)

    dof = coords.shape[1]
    ndof = coords.shape[0] * dof
    nelem = topology.shape[0]
    response_dof = int(plotdof) - 1

    f = np.zeros((ndof, 1), dtype=float)
    df = setload(np.zeros((ndof, 1), dtype=float), loads)
    u = np.zeros((ndof, 1), dtype=float)
    du = np.zeros((ndof, 1), dtype=float)
    S = np.zeros((nelem, 1), dtype=float)
    E = np.zeros((nelem, 1), dtype=float)

    U_path = [0.0]
    F_path = [0.0]

    n = 1
    i = int(i_d)
    restarts = 0
    max_restarts = max(1, int(no_loadsteps) * int(i_max))

    while n <= int(no_loadsteps):
        if restarts > max_restarts:
            raise RuntimeError("Plastic solver exceeded the restart guard.")

        if i < int(i_max):
            K = np.zeros((ndof, ndof), dtype=float)
            if plane_strain:
                K = kq4epe(K, topology, coords, materials, S, E, material_type)
            else:
                K = kq4eps(K, topology, coords, materials, S, E, material_type)

            system_matrix = sp.csc_matrix(K) if sp is not None else K
            Kt, df, _ = setbc(system_matrix.copy(), df, constraints, dof)
            du0 = _solve_plastic_system(Kt, df, plane_strain=plane_strain)
            if not np.all(np.isfinite(du0)):
                raise RuntimeError("NaN/Inf encountered in the plastic solver.")

            if float((du.T @ du0).item()) < 0.0:
                df = -df
                du0 = -du0

            if n == 1:
                l0 = float(np.linalg.norm(du0))
                l = l0
                l_max = 2.0 * l0
            else:
                l = float(np.linalg.norm(du))
                l0 = float(np.linalg.norm(du0))

        if i_d <= i < int(i_max):
            du = min(l / l0, l_max / l0) * du0
        elif i < i_d:
            du = min(2.0 * l / l0, l_max / l0) * du0
        else:
            du0 = 0.5 * du0
            du = du0.copy()
            restarts += 1

        xi = 0.0
        for i in range(1, int(i_max) + 1):
            q = np.zeros((ndof, 1), dtype=float)
            if plane_strain:
                q, Sn, En = qq4epe(q, topology, coords, materials, u + du, S, E, material_type)
            else:
                q, Sn, En = qq4eps(q, topology, coords, materials, u + du, S, E, material_type)

            if not np.all(np.isfinite(q)):
                raise RuntimeError("NaN/Inf encountered in the plastic solver.")

            dq = q - f
            xi = float(((dq.T @ du) / (df.T @ du)).item())
            residual = -dq + xi * df
            if rnorm(residual, constraints, dof) < float(tol) * rnorm(df, constraints, dof):
                break

            Kt, residual_bc, _ = setbc(system_matrix.copy(), residual, constraints, dof)
            delta_u = _solve_plastic_system(Kt, residual_bc, plane_strain=plane_strain)
            du = du + delta_u

        if i >= int(i_max):
            continue

        f = f + xi * df
        u = u + du
        S = Sn
        E = En
        U_path.append(float(u[response_dof, 0]))
        F_path.append(float(f[response_dof, 0]))
        n += 1

    q = np.zeros((ndof, 1), dtype=float)
    if plane_strain:
        q, S, E = qq4epe(q, topology, coords, materials, u, S, E, material_type)
    else:
        q, S, E = qq4eps(q, topology, coords, materials, u, S, E, material_type)

    R = reaction(q, constraints, dof)
    return {
        "u": u,
        "q": q,
        "S": S,
        "E": E,
        "R": R,
        "f": f,
        "U_path": _column(U_path),
        "F_path": _column(F_path),
    }


__all__ = ["solve_nlbar", "solve_plastic"]
