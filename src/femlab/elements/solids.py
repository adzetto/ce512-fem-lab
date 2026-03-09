from __future__ import annotations

import numpy as np

from .._helpers import as_float_array, material_row, topology_nodes, topology_property
from ..assembly import assmk, assmq


def _elastic3d_matrix(Ge):
    props = as_float_array(Ge).reshape(-1)
    E = props[0]
    nu = props[1]
    return E / ((1.0 + nu) * (1.0 - 2.0 * nu)) * np.array(
        [
            [1.0 - nu, nu, nu, 0.0, 0.0, 0.0],
            [nu, 1.0 - nu, nu, 0.0, 0.0, 0.0],
            [nu, nu, 1.0 - nu, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, (1.0 - 2.0 * nu) / 2.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, (1.0 - 2.0 * nu) / 2.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, (1.0 - 2.0 * nu) / 2.0],
        ],
        dtype=float,
    )


def _solid_B(dN):
    nnodes = dN.shape[1]
    B = np.zeros((6, nnodes * 3), dtype=float)
    for n in range(nnodes):
        col = slice(3 * n, 3 * (n + 1))
        B[:, col] = np.array(
            [
                [dN[0, n], 0.0, 0.0],
                [0.0, dN[1, n], 0.0],
                [0.0, 0.0, dN[2, n]],
                [dN[1, n], dN[0, n], 0.0],
                [0.0, dN[2, n], dN[1, n]],
                [dN[2, n], 0.0, dN[0, n]],
            ],
            dtype=float,
        )
    return B


def keT4e(Xe, Ge):
    Xe = as_float_array(Xe)
    dN = np.array([[1.0, 0.0, 0.0, -1.0], [0.0, 1.0, 0.0, -1.0], [0.0, 0.0, 1.0, -1.0]], dtype=float)
    J = dN @ Xe
    dN = np.linalg.solve(J, dN)
    B = _solid_B(dN)
    D = _elastic3d_matrix(Ge)
    return 2.0 * (B.T @ D @ B) * np.linalg.det(J)


def qeT4e(Xe, Ge, Ue):
    Xe = as_float_array(Xe)
    Ue = as_float_array(Ue).reshape(-1, 1)
    dN = np.array([[1.0, 0.0, 0.0, -1.0], [0.0, 1.0, 0.0, -1.0], [0.0, 0.0, 1.0, -1.0]], dtype=float)
    J = dN @ Xe
    dN = np.linalg.solve(J, dN)
    B = _solid_B(dN)
    D = _elastic3d_matrix(Ge)
    Ee = (B @ Ue).reshape(-1)
    Se = Ee @ D
    qe = (B.T @ Se.reshape(-1, 1)) * np.linalg.det(J)
    return qe, Se, Ee


def kT4e(K, T, X, G):
    topology = as_float_array(T)
    coordinates = as_float_array(X)
    for row in topology:
        nodes = topology_nodes(row)
        prop = topology_property(row)
        K = assmk(K, keT4e(coordinates[nodes - 1], material_row(G, prop)), row, 3)
    return K


def qT4e(q, T, X, G, u):
    topology = as_float_array(T)
    coordinates = as_float_array(X)
    U = as_float_array(u).reshape(coordinates.shape[0], coordinates.shape[1])
    S = np.zeros((topology.shape[0], 6), dtype=float)
    E = np.zeros((topology.shape[0], 6), dtype=float)
    for i, row in enumerate(topology):
        nodes = topology_nodes(row)
        prop = topology_property(row)
        qe, Se, Ee = qeT4e(coordinates[nodes - 1], material_row(G, prop), U[nodes - 1].reshape(-1, 1))
        q = assmq(q, qe, row, coordinates.shape[1])
        S[i] = Se
        E[i] = Ee
    return q, S, E


def _hexa_dN(r_i: float, r_j: float, r_k: float):
    dNi = np.array(
        [
            -(1.0 - r_j) * (1.0 - r_k),
            (1.0 - r_j) * (1.0 - r_k),
            (1.0 + r_j) * (1.0 - r_k),
            -(1.0 + r_j) * (1.0 - r_k),
            -(1.0 - r_j) * (1.0 + r_k),
            (1.0 - r_j) * (1.0 + r_k),
            (1.0 + r_j) * (1.0 + r_k),
            -(1.0 + r_j) * (1.0 + r_k),
        ],
        dtype=float,
    ) / 8.0
    dNj = np.array(
        [
            -(1.0 - r_i) * (1.0 - r_k),
            -(1.0 + r_i) * (1.0 - r_k),
            (1.0 + r_i) * (1.0 - r_k),
            (1.0 - r_i) * (1.0 - r_k),
            -(1.0 - r_i) * (1.0 + r_k),
            -(1.0 + r_i) * (1.0 + r_k),
            (1.0 + r_i) * (1.0 + r_k),
            (1.0 - r_i) * (1.0 + r_k),
        ],
        dtype=float,
    ) / 8.0
    dNk = np.array(
        [
            -(1.0 - r_i) * (1.0 - r_j),
            -(1.0 + r_i) * (1.0 - r_j),
            -(1.0 + r_i) * (1.0 + r_j),
            -(1.0 - r_i) * (1.0 + r_j),
            (1.0 - r_i) * (1.0 - r_j),
            (1.0 + r_i) * (1.0 - r_j),
            (1.0 + r_i) * (1.0 + r_j),
            (1.0 - r_i) * (1.0 + r_j),
        ],
        dtype=float,
    ) / 8.0
    return np.vstack([dNi, dNj, dNk])


def keh8e(Xe, Ge):
    Xe = as_float_array(Xe)
    if Xe.shape[0] == 20:
        raise NotImplementedError("20-node hexahedra are not supported yet, matching the Scilab limitation.")
    D = _elastic3d_matrix(Ge)
    r = np.array([-1.0, 1.0], dtype=float) / np.sqrt(3.0)
    w = np.array([1.0, 1.0], dtype=float)
    Ke = np.zeros((24, 24), dtype=float)
    for i in range(2):
        for j in range(2):
            for k in range(2):
                dN = _hexa_dN(r[i], r[j], r[k])
                Jt = dN @ Xe
                dN_global = np.linalg.solve(Jt, dN)
                B = _solid_B(dN_global)
                Ke += w[i] * w[j] * w[k] * (B.T @ D @ B) * np.linalg.det(Jt)
    return Ke


def qeh8e(Xe, Ge, Ue):
    Xe = as_float_array(Xe)
    if Xe.shape[0] == 20:
        raise NotImplementedError("20-node hexahedra are not supported yet, matching the Scilab limitation.")
    D = _elastic3d_matrix(Ge)
    Ue = as_float_array(Ue).reshape(-1, 1)
    r = np.array([-1.0, 1.0], dtype=float) / np.sqrt(3.0)
    w = np.array([1.0, 1.0], dtype=float)
    qe = np.zeros((24, 1), dtype=float)
    Se = np.zeros((8, 6), dtype=float)
    Ee = np.zeros((8, 6), dtype=float)
    gp = 0
    for i in range(2):
        for j in range(2):
            for k in range(2):
                dN = _hexa_dN(r[i], r[j], r[k])
                Jt = dN @ Xe
                dN_global = np.linalg.solve(Jt, dN)
                B = _solid_B(dN_global)
                Ee[gp] = (B @ Ue).ravel()
                Se[gp] = Ee[gp] @ D
                qe += w[i] * w[j] * w[k] * (B.T @ Se[gp].reshape(-1, 1)) * np.linalg.det(Jt)
                gp += 1
    return qe, Se, Ee


def kh8e(K, T, X, G):
    topology = as_float_array(T)
    coordinates = as_float_array(X)
    for row in topology:
        nodes = topology_nodes(row)
        prop = topology_property(row)
        K = assmk(K, keh8e(coordinates[nodes - 1], material_row(G, prop)), row, 3)
    return K


def qh8e(q, T, X, G, u):
    topology = as_float_array(T)
    coordinates = as_float_array(X)
    U = as_float_array(u).reshape(coordinates.shape[0], coordinates.shape[1])
    S = np.zeros((topology.shape[0], 48), dtype=float)
    E = np.zeros((topology.shape[0], 48), dtype=float)
    for i, row in enumerate(topology):
        nodes = topology_nodes(row)
        prop = topology_property(row)
        qe, Se, Ee = qeh8e(coordinates[nodes - 1], material_row(G, prop), U[nodes - 1].reshape(-1, 1))
        q = assmq(q, qe, row, coordinates.shape[1])
        S[i] = Se.reshape(1, 48)
        E[i] = Ee.reshape(1, 48)
    return q, S, E


__all__ = ["keT4e", "keh8e", "kT4e", "kh8e", "qeT4e", "qeh8e", "qT4e", "qh8e"]
