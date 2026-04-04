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
    r"""Return the first scalar entry from an arbitrary array-like object.

    Mathematical Formulation
    ------------------------
    Extracts a scalar from a tensor :math:`\mathbf{T}`:
    .. math:: t = \mathbf{T}_{0, 0, \dots, 0}

    Algorithm
    ---------
    1. Flatten the array-like object to 1D.
    2. Return the element at index 0 as float.
    """
    return float(as_float_array(value).reshape(-1)[0])


def _column(values: list[float]) -> np.ndarray:
    r"""Return a Python list as a floating-point column vector.

    Mathematical Formulation
    ------------------------
    Constructs a column vector :math:`\mathbf{v} \in \mathbb{R}^{n \times 1}`:
    .. math:: \mathbf{v} = [v_1, v_2, \dots, v_n]^T

    Algorithm
    ---------
    1. Convert the input list to a numpy array.
    2. Reshape the array to :math:`(n, 1)`.
    """
    return np.asarray(values, dtype=float).reshape(-1, 1)


def _solve_plastic_system(matrix, rhs, *, plane_strain: bool):
    r"""Select the dense legacy fallback only for plane-strain plastic solves.

    Mathematical Formulation
    ------------------------
    Solves the linear system of equations:
    .. math:: \mathbf{K} \mathbf{u} = \mathbf{f}

    Algorithm
    ---------
    1. Check if the system is plane-strain.
    2. If true, use the legacy symmetric dense solver.
    3. If false, use the standard linear system solver.
    """
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
    r"""
    Solve the legacy nonlinear bar examples with the orthogonal residual method.

    The implementation follows the original ``nlbar.m`` control flow while using
    vectorized element kernels for bar stiffness and internal-force assembly.

    Mathematical Formulation
    ------------------------
    Tracks the load-displacement path of a snap-through structure using the
    Orthogonal Residual Method. The fundamental constraint requires that the
    iterative displacement updates remain orthogonal to the previous increment:

    .. math:: \Delta u_i^T \Delta u_{i+1} = 0

    The external force vector is scaled by a load parameter :math:`\lambda`,
    where :math:`f_{ext} = \lambda P`. The residual at iteration :math:`k` is:

    .. math:: r^{(k)} = \lambda^{(k)} P - f_{int}(u^{(k)})

    Algorithm
    ---------
    1. Initialize nodal displacements :math:`u = 0`, load scaling :math:`\lambda = 0`.
    2. Compute the tangent stiffness matrix :math:`K_T(u)`.
    3. Solve for the reference displacement increment :math:`\Delta u_0 = K_T^{-1} P`.
    4. Loop over load steps:
       a. Predictor: :math:`\Delta u = \alpha \Delta u_0` (scaled by arc-length).
       b. Corrector loop (until convergence):
          i. Compute internal forces :math:`q(u + \Delta u)`.
          ii. Compute load increment :math:`\xi = \frac{(q - f)^T \Delta u}{P^T \Delta u}`.
          iii. Compute residual :math:`r = -(q - f) + \xi P`.
          iv. Solve :math:`K_T \delta u = r`.
          v. Update :math:`\Delta u \leftarrow \Delta u + \delta u`.
       c. Update total displacements and forces.

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
    iteration_count = int(i_d)
    restarts = 0
    max_restarts = max(1, int(no_loadsteps) * int(i_max))

    while n <= int(no_loadsteps):
        if restarts > max_restarts:
            raise RuntimeError("Nonlinear bar solver exceeded the restart guard.")

        if iteration_count < int(i_max):
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
                step_length = l0
                l_max = 2.0 * l0
            else:
                step_length = float(np.linalg.norm(du))
                l0 = float(np.linalg.norm(du0))

        if i_d <= iteration_count < int(i_max):
            du = min(step_length / l0, l_max / l0) * du0
        elif iteration_count < i_d:
            du = min(2.0 * step_length / l0, l_max / l0) * du0
        else:
            du0 = 0.5 * du0
            du = du0.copy()
            restarts += 1

        xi = 0.0
        for _iteration_count in range(1, int(i_max) + 1):
            q = np.zeros((ndof, 1), dtype=float)
            q, S, E = qbar(q, topology, coords, materials, u + du)
            if not np.all(np.isfinite(q)):
                raise RuntimeError("NaN/Inf encountered in the nonlinear bar solver.")

            dq = q - f
            xi = float(((dq.T @ du) / (df.T @ du)).item())
            residual = -dq + xi * df
            if rnorm(residual, constraints, dof) < float(tol) * rnorm(
                df, constraints, dof
            ):
                break

            Kt, residual_bc, _ = setbc(K.copy(), residual, constraints, dof)
            delta_u = np.linalg.solve(Kt, residual_bc)
            du = du + delta_u
        iteration_count = _iteration_count

        if iteration_count >= int(i_max):
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
    r"""
    Solve the legacy Q4 elastoplastic examples with orthogonal residual iterations.

    Parameters follow the original FemLab classroom drivers ``plastps.m`` and
    ``plastpe.m``. The Gauss-point constitutive updates remain in the element
    routines; this function orchestrates the load stepping and equilibrium loop.

    Mathematical Formulation
    ------------------------
    Tracks the elastoplastic load-displacement path using the Orthogonal Residual
    Method. The equilibrium equation balances internal and external forces:

    .. math:: \int_V B^T \sigma(u) \, dV = \lambda P

    The iterative displacement increment :math:`\Delta u` is constrained by:

    .. math:: \Delta u_i^T \Delta u_{i+1} = 0

    Algorithm
    ---------
    1. Initialize displacements :math:`u = 0` and plastic internal history variables.
    2. Compute the tangent stiffness matrix :math:`K_T` based on current state.
    3. Solve for reference step :math:`\Delta u_0 = K_T^{-1} P`.
    4. Loop over load increments:
       a. Predictor step based on arc-length scaling.
       b. Corrector iterations:
          i. Integrate element stresses and update internal variables.
          ii. Assemble internal force vector :math:`q`.
          iii. Evaluate load increment :math:`\xi = \frac{(q - f)^T \Delta u}{P^T \Delta u}`.
          iv. Compute residual :math:`r = -q + f + \xi P`.
          v. Solve :math:`K_T \delta u = r` and update :math:`\Delta u`.
       c. Commit internal history variables (stress :math:`S`, strain :math:`E`).
       d. Update total load and displacement fields.

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
    iteration_count = int(i_d)
    restarts = 0
    max_restarts = max(1, int(no_loadsteps) * int(i_max))

    while n <= int(no_loadsteps):
        if restarts > max_restarts:
            raise RuntimeError("Plastic solver exceeded the restart guard.")

        if iteration_count < int(i_max):
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
                step_length = l0
                l_max = 2.0 * l0
            else:
                step_length = float(np.linalg.norm(du))
                l0 = float(np.linalg.norm(du0))

        if i_d <= iteration_count < int(i_max):
            du = min(step_length / l0, l_max / l0) * du0
        elif iteration_count < i_d:
            du = min(2.0 * step_length / l0, l_max / l0) * du0
        else:
            du0 = 0.5 * du0
            du = du0.copy()
            restarts += 1

        xi = 0.0
        for _iteration_count in range(1, int(i_max) + 1):
            q = np.zeros((ndof, 1), dtype=float)
            if plane_strain:
                q, Sn, En = qq4epe(
                    q, topology, coords, materials, u + du, S, E, material_type
                )
            else:
                q, Sn, En = qq4eps(
                    q, topology, coords, materials, u + du, S, E, material_type
                )

            if not np.all(np.isfinite(q)):
                raise RuntimeError("NaN/Inf encountered in the plastic solver.")

            dq = q - f
            xi = float(((dq.T @ du) / (df.T @ du)).item())
            residual = -dq + xi * df
            if rnorm(residual, constraints, dof) < float(tol) * rnorm(
                df, constraints, dof
            ):
                break

            Kt, residual_bc, _ = setbc(system_matrix.copy(), residual, constraints, dof)
            delta_u = _solve_plastic_system(Kt, residual_bc, plane_strain=plane_strain)
            du = du + delta_u
        iteration_count = _iteration_count

        if iteration_count >= int(i_max):
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
