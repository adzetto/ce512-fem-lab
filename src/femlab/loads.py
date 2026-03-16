from __future__ import annotations

import numpy as np

from ._helpers import as_float_array


def setload(p, P):
    p = as_float_array(p)
    loads = as_float_array(P)
    if loads.size == 0:
        return p
    dof = loads.shape[1] - 1
    indices = ((loads[:, [0]].astype(int) - 1) * dof + np.arange(dof)).reshape(-1)
    p[indices, 0] = loads[:, 1 : 1 + dof].reshape(-1)
    return p


def addload(p, P):
    p = as_float_array(p)
    loads = as_float_array(P)
    if loads.size == 0:
        return p
    dof = loads.shape[1] - 1
    indices = ((loads[:, [0]].astype(int) - 1) * dof + np.arange(dof)).reshape(-1)
    np.add.at(p[:, 0], indices, loads[:, 1 : 1 + dof].reshape(-1))
    return p


__all__ = ["addload", "setload"]
