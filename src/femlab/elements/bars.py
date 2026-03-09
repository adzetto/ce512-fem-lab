from __future__ import annotations

import numpy as np

from .._helpers import as_float_array, cols, material_row, rows, topology_nodes, topology_property
from ..assembly import assmk, assmq


def kebar(Xe0, Xe1, Ge):
    initial = as_float_array(Xe0)
    current = as_float_array(Xe1)
    props = as_float_array(Ge).reshape(-1)

    a0 = (initial[1] - initial[0]).reshape(-1, 1)
    l0 = float(np.sqrt(a0.T @ a0))
    a1 = (current[1] - current[0]).reshape(-1, 1)
    l1 = float(np.sqrt(a1.T @ a1))

    A = props[0]
    E = props[1] if props.size > 1 else 1.0
    strain = 0.5 * (l1**2 - l0**2) / l0**2
    normal_force = A * E * strain
    identity = np.eye(a0.shape[0], dtype=float)
    return (E * A / l0**3) * np.block([[a1 @ a1.T, -a1 @ a1.T], [-a1 @ a1.T, a1 @ a1.T]]) + (
        normal_force / l0
    ) * np.block([[identity, -identity], [-identity, identity]])


def qebar(Xe0, Xe1, Ge):
    initial = as_float_array(Xe0)
    current = as_float_array(Xe1)
    props = as_float_array(Ge).reshape(-1)

    a0 = (initial[1] - initial[0]).reshape(-1, 1)
    l0 = float(np.sqrt(a0.T @ a0))
    a1 = (current[1] - current[0]).reshape(-1, 1)
    l1 = float(np.sqrt(a1.T @ a1))

    A = props[0]
    E = props[1] if props.size > 1 else 1.0
    strain = 0.5 * (l1**2 - l0**2) / l0**2
    stress = E * strain
    qe = (A * stress / l0) * np.vstack([-a1, a1])
    return qe, float(stress), float(strain)


def kbar(K, T, X, G, u=None):
    X = as_float_array(X)
    topology = as_float_array(T)
    if u is None:
        current = X
    else:
        current = X + as_float_array(u).reshape(rows(X), cols(X))

    for row in topology:
        nodes = topology_nodes(row)
        prop = topology_property(row)
        Ke = kebar(X[nodes - 1], current[nodes - 1], material_row(G, prop))
        K = assmk(K, Ke, row, cols(X))
    return K


def qbar(q, T, X, G, u=None):
    X = as_float_array(X)
    topology = as_float_array(T)
    if u is None:
        current = X
    else:
        current = X + as_float_array(u).reshape(rows(X), cols(X))

    S = np.zeros((topology.shape[0], 1), dtype=float)
    E = np.zeros((topology.shape[0], 1), dtype=float)
    for i, row in enumerate(topology):
        nodes = topology_nodes(row)
        prop = topology_property(row)
        qe, Se, Ee = qebar(X[nodes - 1], current[nodes - 1], material_row(G, prop))
        q = assmq(q, qe, row, cols(X))
        S[i, 0] = Se
        E[i, 0] = Ee
    return q, S, E


__all__ = ["kbar", "kebar", "qbar", "qebar"]
