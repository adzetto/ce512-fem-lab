from __future__ import annotations

import numpy as np

from ._helpers import as_float_array


def reaction(q, C, dof: int, comp: int | None = None):
    force = as_float_array(q).reshape(-1, 1)
    constraints = as_float_array(C)
    if comp is not None:
        constraints = constraints[np.asarray(constraints[:, 1] == comp).ravel()]
    if constraints.size == 0:
        width = 2 if comp is not None else 3
        return np.zeros((0, width), dtype=float)
    indices = ((constraints[:, 0] - 1) * dof + constraints[:, 1] - 1).astype(int)
    reactions = force[indices, 0]
    if comp is not None:
        return np.column_stack([constraints[:, 0], reactions])
    return np.column_stack([constraints[:, :2], reactions])


__all__ = ["reaction"]
