from __future__ import annotations

from ._helpers import as_float_array


def setload(p, P):
    p = as_float_array(p)
    loads = as_float_array(P)
    if loads.size == 0:
        return p
    dof = loads.shape[1] - 1
    for row in loads:
        offset = (int(row[0]) - 1) * dof
        p[offset : offset + dof, 0] = row[1 : 1 + dof]
    return p


def addload(p, P):
    p = as_float_array(p)
    loads = as_float_array(P)
    if loads.size == 0:
        return p
    dof = loads.shape[1] - 1
    for row in loads:
        offset = (int(row[0]) - 1) * dof
        p[offset : offset + dof, 0] += row[1 : 1 + dof]
    return p


__all__ = ["addload", "setload"]
