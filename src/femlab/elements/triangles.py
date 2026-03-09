from __future__ import annotations

import numpy as np

from .._helpers import as_float_array, material_row, topology_nodes, topology_property
from ..assembly import assmk, assmq


def _triangle_geometry(Xe):
    Xe = as_float_array(Xe)
    a = np.vstack([Xe[2] - Xe[1], Xe[0] - Xe[2], Xe[1] - Xe[0]])
    area = 0.5 * abs(np.linalg.det(a[0:2, 0:2]))
    return a, area


def _elastic_matrix(Ge, *, plane_strain: bool = False):
    material = as_float_array(Ge).reshape(-1)
    E = material[0]
    nu = material[1]
    if not plane_strain:
        return E / (1.0 - nu**2) * np.array(
            [[1.0, nu, 0.0], [nu, 1.0, 0.0], [0.0, 0.0, (1.0 - nu) / 2.0]],
            dtype=float,
        )
    return E / ((1.0 + nu) * (1.0 - 2.0 * nu)) * np.array(
        [[1.0 - nu, nu, 0.0], [nu, 1.0 - nu, 0.0], [0.0, 0.0, (1.0 - 2.0 * nu) / 2.0]],
        dtype=float,
    )


def ket3e(Xe, Ge):
    a, area = _triangle_geometry(Xe)
    dN = (1.0 / (2.0 * area)) * np.column_stack([-a[:, 1], a[:, 0]]).T
    B = np.array(
        [
            [dN[0, 0], 0.0, dN[0, 1], 0.0, dN[0, 2], 0.0],
            [0.0, dN[1, 0], 0.0, dN[1, 1], 0.0, dN[1, 2]],
            [dN[1, 0], dN[0, 0], dN[1, 1], dN[0, 1], dN[1, 2], dN[0, 2]],
        ],
        dtype=float,
    )
    props = as_float_array(Ge).reshape(-1)
    plane_strain = props.size > 2 and int(props[2]) == 2
    D = _elastic_matrix(props, plane_strain=plane_strain)
    return (B.T @ D @ B) * area


def qet3e(Xe, Ge, Ue):
    a, area = _triangle_geometry(Xe)
    dN = (1.0 / (2.0 * area)) * np.column_stack([-a[:, 1], a[:, 0]])
    B = np.array(
        [
            [dN[0, 0], 0.0, dN[1, 0], 0.0, dN[2, 0], 0.0],
            [0.0, dN[0, 1], 0.0, dN[1, 1], 0.0, dN[2, 1]],
            [dN[0, 1], dN[0, 0], dN[1, 1], dN[1, 0], dN[2, 1], dN[2, 0]],
        ],
        dtype=float,
    )
    props = as_float_array(Ge).reshape(-1)
    plane_strain = props.size > 2 and int(props[2]) == 2
    D = _elastic_matrix(props, plane_strain=plane_strain)
    Ue = as_float_array(Ue).reshape(-1, 1)
    Ee = (B @ Ue).reshape(1, -1)
    Se = Ee @ D
    qe = (B.T @ Se.T) * area
    return qe, Se.reshape(-1), Ee.reshape(-1)


def kt3e(K, T, X, G):
    topology = as_float_array(T)
    coordinates = as_float_array(X)
    for row in topology:
        nodes = topology_nodes(row)
        prop = topology_property(row)
        K = assmk(K, ket3e(coordinates[nodes - 1], material_row(G, prop)), row, 2)
    return K


def qt3e(q, T, X, G, u):
    topology = as_float_array(T)
    coordinates = as_float_array(X)
    U = as_float_array(u).reshape(coordinates.shape[0], coordinates.shape[1])
    S = np.zeros((topology.shape[0], 3), dtype=float)
    E = np.zeros((topology.shape[0], 3), dtype=float)
    for i, row in enumerate(topology):
        nodes = topology_nodes(row)
        prop = topology_property(row)
        Ue = U[nodes - 1].reshape(-1, 1, order="C")
        qe, Se, Ee = qet3e(coordinates[nodes - 1], material_row(G, prop), Ue)
        q = assmq(q, qe, row, coordinates.shape[1])
        S[i] = Se
        E[i] = Ee
    return q, S, E


def ket3p(Xe, Ge):
    a, area = _triangle_geometry(Xe)
    props = as_float_array(Ge).reshape(-1)
    conductivity = props[0]
    D = np.eye(2, dtype=float) * conductivity
    B = (1.0 / (2.0 * area)) * np.column_stack([-a[:, 1], a[:, 0]]).T
    Ke = area * B.T @ D @ B
    if props.size > 1:
        b = props[1]
        Ke = Ke + (b * area / 12.0) * np.array([[2.0, 1.0, 1.0], [1.0, 2.0, 1.0], [1.0, 1.0, 2.0]])
    return Ke


def qet3p(Xe, Ge, Ue):
    a, area = _triangle_geometry(Xe)
    B = (1.0 / (2.0 * area)) * np.column_stack([-a[:, 1], a[:, 0]]).T
    conductivity = as_float_array(Ge).reshape(-1)[0]
    D = np.eye(2, dtype=float) * conductivity
    Ue = as_float_array(Ue).reshape(-1, 1)
    Ee = (B @ Ue).reshape(1, -1)
    Se = Ee @ D
    qe = (B.T @ Se.T) * area
    return qe, Se.reshape(-1), Ee.reshape(-1)


def kt3p(K, T, X, G):
    topology = as_float_array(T)
    coordinates = as_float_array(X)
    for row in topology:
        nodes = topology_nodes(row)
        prop = topology_property(row)
        K = assmk(K, ket3p(coordinates[nodes - 1], material_row(G, prop)), row, 1)
    return K


def qt3p(q, T, X, G, u):
    topology = as_float_array(T)
    coordinates = as_float_array(X)
    potentials = as_float_array(u).reshape(-1, 1)
    S = np.zeros((topology.shape[0], 2), dtype=float)
    E = np.zeros((topology.shape[0], 2), dtype=float)
    for i, row in enumerate(topology):
        nodes = topology_nodes(row)
        prop = topology_property(row)
        qe, Se, Ee = qet3p(coordinates[nodes - 1], material_row(G, prop), potentials[nodes - 1])
        q = assmq(q, qe, row, 1)
        S[i] = Se
        E[i] = Ee
    return q, S, E


__all__ = ["ket3e", "ket3p", "kt3e", "kt3p", "qet3e", "qet3p", "qt3e", "qt3p"]
