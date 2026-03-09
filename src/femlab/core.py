from __future__ import annotations

from ._helpers import cols, rows, zeros_matrix, zeros_vector


def init(nn: int, dof: int, *, use_sparse: bool | None = None):
    total_dofs = int(nn) * int(dof)
    if use_sparse is None:
        use_sparse = nn >= 1000
    stiffness = zeros_matrix(total_dofs, use_sparse=use_sparse)
    load = zeros_vector(total_dofs)
    internal = zeros_vector(total_dofs)
    return stiffness, load, internal


__all__ = ["cols", "init", "rows"]
