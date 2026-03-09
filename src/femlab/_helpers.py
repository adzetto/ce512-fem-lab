from __future__ import annotations

from typing import Iterable

import numpy as np

try:
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla
except ImportError:  # pragma: no cover - optional at import time
    sp = None
    spla = None

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
    return int(np.asarray(data).shape[0])


def cols(data: object) -> int:
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
    node_numbers = as_int_array(list(node_numbers)).ravel()
    indices = []
    for node in node_numbers:
        start = (int(node) - 1) * dof
        indices.extend(range(start, start + dof))
    return np.asarray(indices, dtype=int)


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
        solution = spla.spsolve(matrix.tocsr(), rhs_array)
    else:
        solution = np.linalg.solve(as_float_array(matrix), rhs_array)
    return np.asarray(solution, dtype=float).reshape(-1, 1)
