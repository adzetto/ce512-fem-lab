from __future__ import annotations

from typing import Iterable

import numpy as np

try:
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla
    import scipy.linalg as sla
except ImportError:  # pragma: no cover - optional at import time
    sp = None
    spla = None
    sla = None

FloatArray = np.ndarray
IntArray = np.ndarray


def as_float_array(data: object, *, copy: bool = False) -> FloatArray:
    array = np.asarray(data, dtype=float)
    return array.copy() if copy else array


def as_int_array(data: object, *, copy: bool = False) -> IntArray:
    array = np.asarray(data, dtype=int)
    return array.copy() if copy else array


def as_column(data: object, *, copy: bool = False) -> FloatArray:
    array = as_float_array(data, copy=copy).reshape(-1, 1)
    return array


def rows(data: object) -> int:
    """Return the number of rows in an array-like object."""
    return int(np.asarray(data).shape[0])


def cols(data: object) -> int:
    """Return the number of columns in an array-like object."""
    array = np.asarray(data)
    if array.ndim == 1:
        return int(array.shape[0])
    return int(array.shape[1])


def is_sparse(matrix: object) -> bool:
    return bool(sp is not None and sp.issparse(matrix))


def max_abs_diagonal(matrix: object) -> float:
    if is_sparse(matrix):
        diagonal = np.asarray(matrix.diagonal()).ravel()
    else:
        diagonal = np.diag(np.asarray(matrix, dtype=float))
    if diagonal.size == 0:
        return 0.0
    return float(np.max(np.abs(diagonal)))


def node_dof_indices(node_numbers: Iterable[int], dof: int) -> IntArray:
    return element_dof_indices(node_numbers, dof).reshape(-1)


def element_dof_indices(
    node_numbers: object, dof: int, *, one_based: bool = True
) -> IntArray:
    nodes = as_int_array(node_numbers)
    zero_based = nodes - 1 if one_based else nodes
    offsets = np.arange(dof, dtype=int)
    return (zero_based[..., None] * dof + offsets).reshape(*nodes.shape[:-1], -1)


def topology_nodes(topology_row: object) -> IntArray:
    topology_row = as_int_array(topology_row).ravel()
    return topology_row[:-1]


def topology_property(topology_row: object) -> int:
    topology_row = as_int_array(topology_row).ravel()
    return int(topology_row[-1])


def material_row(materials: object, property_number: int) -> FloatArray:
    materials = as_float_array(materials)
    if materials.ndim == 1:
        materials = materials.reshape(1, -1)
    return materials[property_number - 1]


def zeros_matrix(size: int, *, use_sparse: bool = False) -> object:
    if use_sparse and sp is not None:
        return sp.lil_matrix((size, size), dtype=float)
    return np.zeros((size, size), dtype=float)


def zeros_vector(size: int) -> FloatArray:
    return np.zeros((size, 1), dtype=float)


def solve_linear_system(matrix: object, rhs: object) -> FloatArray:
    rhs_array = as_float_array(rhs).reshape(-1)
    if is_sparse(matrix):
        if spla is None:  # pragma: no cover
            raise RuntimeError("scipy is required to solve sparse systems")
        solution = spla.spsolve(matrix.tocsc(), rhs_array)
    else:
        solution = np.linalg.solve(as_float_array(matrix), rhs_array)
    return np.asarray(solution, dtype=float).reshape(-1, 1)


def solve_legacy_symmetric_system(
    matrix: object,
    rhs: object,
    *,
    dense_size_limit: int = 256,
    condition_threshold: float = 2.0e8,
) -> FloatArray:
    """Solve small symmetric legacy systems with a MATLAB-like dense fallback.

    The legacy FemLab classroom examples use very small symmetric systems in
    the elastoplastic plane-strain driver. MATLAB's sparse ``\\`` path tends to
    switch between Cholesky-like and symmetric-indefinite strategies depending
    on conditioning, while SciPy's default sparse solve uses a generic LU path.
    For these tiny systems, densifying and selecting between positive-definite
    and symmetric solves gives closer parity with the historical MATLAB traces.

    Parameters
    ----------
    matrix, rhs:
        Linear system ``A x = b``.
    dense_size_limit:
        Maximum system size for the dense legacy fallback.
    condition_threshold:
        Use a positive-definite solve below this condition number and a generic
        symmetric solve above it.
    """

    if sla is None:
        return solve_linear_system(matrix, rhs)

    size = int(np.shape(matrix)[0])
    if size > dense_size_limit:
        return solve_linear_system(matrix, rhs)

    dense_matrix = (
        np.asarray(matrix.toarray(), dtype=float)
        if is_sparse(matrix)
        else as_float_array(matrix)
    )
    if dense_matrix.ndim != 2 or dense_matrix.shape[0] != dense_matrix.shape[1]:
        return solve_linear_system(matrix, rhs)

    symmetry_error = np.linalg.norm(dense_matrix - dense_matrix.T, ord=np.inf)
    scale = max(1.0, np.linalg.norm(dense_matrix, ord=np.inf))
    if symmetry_error > 1.0e-10 * scale:
        return solve_linear_system(matrix, rhs)

    rhs_array = as_float_array(rhs).reshape(-1)
    try:
        condition = float(np.linalg.cond(dense_matrix))
        assume = "pos" if condition < condition_threshold else "sym"
        solution = sla.solve(
            dense_matrix,
            rhs_array,
            assume_a=assume,
            check_finite=False,
            overwrite_a=False,
            overwrite_b=False,
        )
    except Exception:
        return solve_linear_system(matrix, rhs)
    return np.asarray(solution, dtype=float).reshape(-1, 1)
