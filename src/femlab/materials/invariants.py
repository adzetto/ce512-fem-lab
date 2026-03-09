from __future__ import annotations

import numpy as np

from .._helpers import as_float_array


def devstress(S):
    stress = as_float_array(S, copy=True).reshape(-1, 1)
    if stress.shape[0] == 3:
        mean = float((stress[0, 0] + stress[1, 0]) / 3.0)
        stress[0:2, 0] -= mean
    else:
        mean = float((stress[0, 0] + stress[1, 0] + stress[2, 0]) / 3.0)
        stress[0:3, 0] -= mean
    return stress, mean


def eqstress(S) -> float:
    stress = as_float_array(S).reshape(-1)
    if stress.shape[0] == 3:
        value = stress[0] ** 2 + stress[1] ** 2 - stress[0] * stress[1] + 3.0 * stress[2] ** 2
    else:
        value = 0.5 * (
            (stress[0] - stress[1]) ** 2
            + (stress[1] - stress[2]) ** 2
            + (stress[2] - stress[0]) ** 2
        )
        value += np.sum(3.0 * stress[3:] ** 2)
    return float(np.sqrt(value))
