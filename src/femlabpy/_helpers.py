from __future__ import annotations

from collections.abc import Iterable
from typing import Any, cast

import numpy as np

try:
    import scipy.linalg as sla
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla
except ImportError:  # pragma: no cover - optional at import time
    sp = None
    spla = None
    sla = None


FloatArray = np.ndarray
IntArray = np.ndarray


def as_float_array(data: object, *, copy: bool = False) -> FloatArray:
    """
    Coerce arbitrary array-like input to a floating-point NumPy array.

    Parameters
    ----------
    data:
        Scalar, sequence, or array-like object accepted by :func:`numpy.asarray`.
    copy:
        When ``True``, return a defensive copy even if ``data`` is already a
        floating-point array.

    Returns
    -------
    ndarray
        Floating-point view of the input data.

    Examples
    --------
    >>> as_float_array([[1, 2], [3, 4]]).dtype.kind
    'f'
    """
    array = np.asarray(data, dtype=float)
    return array.copy() if copy else array


def as_int_array(data: object, *, copy: bool = False) -> IntArray:
    """
    Coerce arbitrary array-like input to an integer NumPy array.

    Parameters
    ----------
    data:
        Scalar, sequence, or array-like object accepted by :func:`numpy.asarray`.
    copy:
        When ``True``, return a defensive copy even if ``data`` is already an
        integer array.

    Returns
    -------
    ndarray
        Integer view of the input data.
    """
    array = np.asarray(data, dtype=int)
    return array.copy() if copy else array


def as_column(data: object, *, copy: bool = False) -> FloatArray:
    """
    Return input data as a two-dimensional column vector.

    Parameters
    ----------
    data:
        Scalar or array-like values to reshape into ``(n, 1)`` form.
    copy:
        Forwarded to :func:`as_float_array`.

    Returns
    -------
    ndarray
        Floating-point column vector.
    """
    return as_float_array(data, copy=copy).reshape(-1, 1)


def rows(data: object) -> int:
    """
    Return the leading dimension of an array-like object.

    Parameters
    ----------
    data:
        Array-like input.

    Returns
    -------
    int
        Number of rows in the array representation.
    """
    return int(np.asarray(data).shape[0])


def cols(data: object) -> int:
    """
    Return the column count of an array-like object.

    One-dimensional inputs follow the MATLAB classroom convention where a row
    vector reports its full length as the column count.

    Parameters
    ----------
    data:
        Array-like input.

    Returns
    -------
    int
        Number of columns in the array representation.
    """
    array = np.asarray(data)
    if array.ndim == 1:
        return int(array.shape[0])
    return int(array.shape[1])


def is_sparse(matrix: object) -> bool:
    """
    Return ``True`` when ``matrix`` is a SciPy sparse matrix instance.

    Parameters
    ----------
    matrix:
        Candidate matrix object.

    Returns
    -------
    bool
        ``True`` if SciPy is available and ``matrix`` is sparse.
    """
    return bool(sp is not None and sp.issparse(matrix))


def max_abs_diagonal(matrix: object) -> float:
    """
    Return the largest absolute diagonal entry of a matrix-like object.

    Parameters
    ----------
    matrix:
        Dense or sparse square matrix.

    Returns
    -------
    float
        Maximum absolute value along the main diagonal, or ``0.0`` for an empty
        diagonal.
    """
    if is_sparse(matrix):
        diagonal = np.asarray(cast(Any, matrix).diagonal()).ravel()
    else:
        diagonal = np.diag(as_float_array(matrix))
    if diagonal.size == 0:
        return 0.0
    return float(np.max(np.abs(diagonal)))


def node_dof_indices(node_numbers: Iterable[int], dof: int) -> IntArray:
    """
    Expand one-based node numbers into flattened degree-of-freedom indices.

    Parameters
    ----------
    node_numbers:
        Iterable of one-based node identifiers.
    dof:
        Degrees of freedom per node.

    Returns
    -------
    ndarray
        Flattened zero-based degree-of-freedom indices.
    """
    return element_dof_indices(node_numbers, dof).reshape(-1)


def element_dof_indices(
    node_numbers: object, dof: int, *, one_based: bool = True
) -> IntArray:
    """
    Expand node connectivity into zero-based degree-of-freedom indices.

    Parameters
    ----------
    node_numbers:
        Array-like connectivity table or node list.
    dof:
        Degrees of freedom per node.
    one_based:
        When ``True``, interpret ``node_numbers`` using the MATLAB/Scilab
        one-based convention and convert to zero-based Python indices.

    Returns
    -------
    ndarray
        Degree-of-freedom index array with the same leading dimensions as
        ``node_numbers`` and a flattened trailing DOF dimension.
    """
    nodes = as_int_array(node_numbers)
    zero_based = nodes - 1 if one_based else nodes
    offsets = np.arange(dof, dtype=int)
    return (zero_based[..., None] * dof + offsets).reshape(*nodes.shape[:-1], -1)


def topology_nodes(topology_row: object) -> IntArray:
    """
    Extract the node-number portion of a legacy FemLab topology row.

    Parameters
    ----------
    topology_row:
        One topology row whose last entry stores the material or property id.

    Returns
    -------
    ndarray
        One-based node numbers for the element.
    """
    row = as_int_array(topology_row).ravel()
    return row[:-1]


def topology_property(topology_row: object) -> int:
    """
    Return the property number stored in a legacy FemLab topology row.

    Parameters
    ----------
    topology_row:
        One topology row whose last entry stores the material or property id.

    Returns
    -------
    int
        One-based material or property number.
    """
    row = as_int_array(topology_row).ravel()
    return int(row[-1])


def material_row(materials: object, property_number: int) -> FloatArray:
    """
    Return one material-property row using FemLab's one-based numbering.

    Parameters
    ----------
    materials:
        Material table where each row is one material definition.
    property_number:
        One-based row number into ``materials``.

    Returns
    -------
    ndarray
        Selected material-property row.
    """
    material_table = as_float_array(materials)
    if material_table.ndim == 1:
        material_table = material_table.reshape(1, -1)
    return material_table[property_number - 1]


def zeros_matrix(size: int, *, use_sparse: bool = False) -> object:
    """
    Create a zero square matrix in dense or sparse storage.

    Parameters
    ----------
    size:
        Number of rows and columns.
    use_sparse:
        When ``True`` and SciPy is available, return an ``lil_matrix`` for
        efficient incremental assembly.

    Returns
    -------
    ndarray or sparse matrix
        Zero-initialized square matrix.
    """
    if use_sparse and sp is not None:
        return sp.lil_matrix((size, size), dtype=float)
    return np.zeros((size, size), dtype=float)


def zeros_vector(size: int) -> FloatArray:
    """
    Create a zero column vector of length ``size``.

    Parameters
    ----------
    size:
        Vector length.

    Returns
    -------
    ndarray
        Zero-initialized column vector with shape ``(size, 1)``.
    """
    return np.zeros((size, 1), dtype=float)


def solve_linear_system(matrix: object, rhs: object) -> FloatArray:
    """
    Solve a dense or sparse linear system and return a column vector solution.

    Parameters
    ----------
    matrix:
        Dense or sparse coefficient matrix ``A``.
    rhs:
        Right-hand side ``b``.

    Returns
    -------
    ndarray
        Solution vector ``x`` with shape ``(n, 1)``.
    """
    rhs_array = as_float_array(rhs).reshape(-1)
    if is_sparse(matrix):
        if spla is None:  # pragma: no cover
            raise RuntimeError("scipy is required to solve sparse systems")
        solution = spla.spsolve(cast(Any, matrix).tocsc(), rhs_array)
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
    """
    Solve a small symmetric system with a MATLAB-like dense fallback path.

    The legacy plane-strain classroom examples use tiny symmetric tangent
    systems. MATLAB often switches automatically between positive-definite and
    symmetric-indefinite dense kernels for such problems, while SciPy's sparse
    default tends to use a more generic LU route. Densifying these small
    systems and selecting a solve strategy based on the estimated condition
    number reproduces the historical MATLAB traces more closely.

    Parameters
    ----------
    matrix, rhs:
        Linear system ``A x = b``.
    dense_size_limit:
        Maximum system size eligible for the dense fallback.
    condition_threshold:
        Use a positive-definite solver below this condition number and a generic
        symmetric solver above it.

    Returns
    -------
    ndarray
        Solution vector ``x`` with shape ``(n, 1)``.
    """
    if sla is None:
        return solve_linear_system(matrix, rhs)

    dense_matrix = (
        np.asarray(cast(Any, matrix).toarray(), dtype=float)
        if is_sparse(matrix)
        else as_float_array(matrix)
    )
    if dense_matrix.ndim != 2 or dense_matrix.shape[0] != dense_matrix.shape[1]:
        return solve_linear_system(matrix, rhs)

    size = int(dense_matrix.shape[0])
    if size > dense_size_limit:
        return solve_linear_system(matrix, rhs)

    symmetry_error = float(np.linalg.norm(dense_matrix - dense_matrix.T, ord=np.inf))
    scale = max(1.0, float(np.linalg.norm(dense_matrix, ord=np.inf)))
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
