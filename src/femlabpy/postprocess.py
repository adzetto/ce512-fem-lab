from __future__ import annotations

import numpy as np

from ._helpers import as_float_array


def reaction(q, C, dof: int, comp: int | None = None):
    """Extract support reactions at constrained degrees of freedom."""
    force = as_float_array(q).reshape(-1, 1)
    constraints = as_float_array(C)
    if comp is not None:
        row_indices = np.flatnonzero(np.asarray(constraints[:, 1] == comp).ravel())
        constraints = constraints[row_indices]
    if constraints.size == 0:
        width = 2 if comp is not None else 3
        return np.zeros((0, width), dtype=float)
    indices = ((constraints[:, 0] - 1) * dof + constraints[:, 1] - 1).astype(int)
    reactions = force[indices, 0]
    if comp is not None:
        # MATLAB's `reaction(..., comp)` returns the filtered constraint-row index.
        return np.column_stack([row_indices + 1, reactions])
    return np.column_stack([constraints[:, :2], reactions])


__all__ = ["reaction"]
