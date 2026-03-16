from __future__ import annotations

import numpy as np

from .._helpers import as_float_array, material_row, topology_nodes, topology_property
from ..assembly import assmk, assmq
from ..materials import devstress, eqstress, stressdp, stressvm


def _plane_elastic_matrix(Ge, *, plane_strain: bool = False):
    props = as_float_array(Ge).reshape(-1)
    E = props[0]
    nu = props[1]
    if not plane_strain:
        return (
            E
            / (1.0 - nu**2)
            * np.array(
                [[1.0, nu, 0.0], [nu, 1.0, 0.0], [0.0, 0.0, (1.0 - nu) / 2.0]],
                dtype=float,
            )
        )
    return (
        E
        / ((1.0 + nu) * (1.0 - 2.0 * nu))
        * np.array(
            [
                [1.0 - nu, nu, nu, 0.0],
                [nu, 1.0 - nu, nu, 0.0],
                [nu, nu, 1.0 - nu, 0.0],
                [0.0, 0.0, 0.0, (1.0 - 2.0 * nu) / 2.0],
            ],
            dtype=float,
        )
    )


def _plane_elastic_matrix_2d(Ge, *, plane_strain: bool = False):
    props = as_float_array(Ge).reshape(-1)
    E = props[0]
    nu = props[1]
    if not plane_strain:
        return (
            E
            / (1.0 - nu**2)
            * np.array(
                [[1.0, nu, 0.0], [nu, 1.0, 0.0], [0.0, 0.0, (1.0 - nu) / 2.0]],
                dtype=float,
            )
        )
    return (
        E
        / ((1.0 + nu) * (1.0 - 2.0 * nu))
        * np.array(
            [
                [1.0 - nu, nu, 0.0],
                [nu, 1.0 - nu, 0.0],
                [0.0, 0.0, (1.0 - 2.0 * nu) / 2.0],
            ],
            dtype=float,
        )
    )


def _q4_dN(r_i: float, r_j: float, nnodes: int):
    dN = (
        np.array(
            [
                [-(1.0 - r_j), 1.0 - r_j, 1.0 + r_j, -(1.0 + r_j)],
                [-(1.0 - r_i), -(1.0 + r_i), 1.0 + r_i, 1.0 - r_i],
            ],
            dtype=float,
        )
        / 4.0
    )
    if nnodes != 8:
        return dN

    dN8 = np.array(
        [
            [
                -r_i * (1.0 - r_j),
                0.5 * (1.0 - r_j**2),
                -r_i * (1.0 + r_j),
                -0.5 * (1.0 - r_j**2),
            ],
            [
                -0.5 * (1.0 - r_i**2),
                -r_j * (1.0 + r_i),
                0.5 * (1.0 - r_i**2),
                -r_j * (1.0 - r_i),
            ],
        ],
        dtype=float,
    )
    dN[:, 0] = dN[:, 0] - 0.5 * dN8[:, 0] - 0.5 * dN8[:, 3]
    dN[:, 1] = dN[:, 1] - 0.5 * dN8[:, 1] - 0.5 * dN8[:, 0]
    dN[:, 2] = dN[:, 2] - 0.5 * dN8[:, 2] - 0.5 * dN8[:, 1]
    dN[:, 3] = dN[:, 3] - 0.5 * dN8[:, 3] - 0.5 * dN8[:, 2]
    return np.hstack([dN, dN8])


def _q4_B(dN):
    nnodes = dN.shape[1]
    B = np.zeros((3, 2 * nnodes), dtype=float)
    for k in range(nnodes):
        B[0, 2 * k] = dN[0, k]
        B[1, 2 * k + 1] = dN[1, k]
        B[2, 2 * k] = dN[1, k]
        B[2, 2 * k + 1] = dN[0, k]
    return B


def _q4_gauss_points():
    r = np.array([-1.0, 1.0], dtype=float) / np.sqrt(3.0)
    w = np.array([1.0, 1.0], dtype=float)
    return r, w


def _q4_gp_index(i: int, j: int) -> int:
    return i + 3 * j - 2 * i * j


def keq4e(Xe, Ge):
    Xe = as_float_array(Xe)
    props = as_float_array(Ge).reshape(-1)
    plane_strain = props.size > 2 and int(props[2]) == 2
    D = _plane_elastic_matrix_2d(props, plane_strain=plane_strain)
    nnodes = Xe.shape[0]
    r, w = _q4_gauss_points()
    Ke = np.zeros((2 * nnodes, 2 * nnodes), dtype=float)
    for i in range(2):
        for j in range(2):
            dN = _q4_dN(r[i], r[j], nnodes)
            Jt = dN @ Xe
            dN_global = np.linalg.solve(Jt, dN)
            B = _q4_B(dN_global)
            Ke += w[i] * w[j] * (B.T @ D @ B) * np.linalg.det(Jt)
    return Ke


def qeq4e(Xe, Ge, Ue):
    Xe = as_float_array(Xe)
    props = as_float_array(Ge).reshape(-1)
    plane_strain = props.size > 2 and int(props[2]) == 2
    D = _plane_elastic_matrix_2d(props, plane_strain=plane_strain)
    nnodes = Xe.shape[0]
    Ue = as_float_array(Ue).reshape(-1, 1)
    qe = np.zeros((2 * nnodes, 1), dtype=float)
    Se = np.zeros((4, 3), dtype=float)
    Ee = np.zeros((4, 3), dtype=float)
    r, w = _q4_gauss_points()
    for i in range(2):
        for j in range(2):
            gp = _q4_gp_index(i, j)
            dN = _q4_dN(r[i], r[j], nnodes)
            Jt = dN @ Xe
            dN_global = np.linalg.solve(Jt, dN)
            B = _q4_B(dN_global)
            Ee[gp] = (B @ Ue).ravel()
            Se[gp] = Ee[gp] @ D
            qe += w[i] * w[j] * (B.T @ Se[gp].reshape(-1, 1)) * np.linalg.det(Jt)
    return qe, Se, Ee


def kq4e(K, T, X, G):
    topology = as_float_array(T)
    coordinates = as_float_array(X)
    for row in topology:
        nodes = topology_nodes(row)
        prop = topology_property(row)
        K = assmk(K, keq4e(coordinates[nodes - 1], material_row(G, prop)), row, 2)
    return K


def qq4e(q, T, X, G, u):
    topology = as_float_array(T)
    coordinates = as_float_array(X)
    U = as_float_array(u).reshape(coordinates.shape[0], coordinates.shape[1])
    S = np.zeros((topology.shape[0], 12), dtype=float)
    E = np.zeros((topology.shape[0], 12), dtype=float)
    for i, row in enumerate(topology):
        nodes = topology_nodes(row)
        prop = topology_property(row)
        Ue = U[nodes - 1].reshape(-1, 1)
        qe, Se, Ee = qeq4e(coordinates[nodes - 1], material_row(G, prop), Ue)
        q = assmq(q, qe, row, coordinates.shape[1])
        S[i] = Se.reshape(1, 12)
        E[i] = Ee.reshape(1, 12)
    return q, S, E


def keq4p(Xe, Ge):
    Xe = as_float_array(Xe)
    props = as_float_array(Ge).reshape(-1)
    k = props[0]
    b = props[1] if props.size > 1 else 0.0
    D = np.eye(2, dtype=float) * k
    Ke = np.zeros((4, 4), dtype=float)
    r, w = _q4_gauss_points()
    for i in range(2):
        for j in range(2):
            N = (
                np.array(
                    [
                        (1.0 - r[i]) * (1.0 - r[j]),
                        (1.0 + r[i]) * (1.0 - r[j]),
                        (1.0 + r[i]) * (1.0 + r[j]),
                        (1.0 - r[i]) * (1.0 + r[j]),
                    ],
                    dtype=float,
                ).reshape(1, -1)
                / 4.0
            )
            dN = _q4_dN(r[i], r[j], 4)
            Jt = dN @ Xe
            B = np.linalg.solve(Jt, dN)
            Ke += w[i] * w[j] * (B.T @ D @ B + N.T @ (b * N)) * np.linalg.det(Jt)
    return Ke


def qeq4p(Xe, Ge, Ue):
    Xe = as_float_array(Xe)
    props = as_float_array(Ge).reshape(-1)
    k = props[0]
    D = np.eye(2, dtype=float) * k
    Ue = as_float_array(Ue).reshape(-1, 1)
    qe = np.zeros((4, 1), dtype=float)
    Se = np.zeros((4, 2), dtype=float)
    Ee = np.zeros((4, 2), dtype=float)
    r, w = _q4_gauss_points()
    for i in range(2):
        for j in range(2):
            gp = _q4_gp_index(i, j)
            dN = _q4_dN(r[i], r[j], 4)
            Jt = dN @ Xe
            B = np.linalg.solve(Jt, dN)
            Ee[gp] = (B @ Ue).ravel()
            Se[gp] = Ee[gp] @ D
            qe += w[i] * w[j] * (B.T @ Se[gp].reshape(-1, 1)) * np.linalg.det(Jt)
    return qe, Se, Ee


def kq4p(K, T, X, G):
    topology = as_float_array(T)
    coordinates = as_float_array(X)
    for row in topology:
        nodes = topology_nodes(row)
        prop = topology_property(row)
        K = assmk(K, keq4p(coordinates[nodes - 1], material_row(G, prop)), row, 1)
    return K


def qq4p(q, T, X, G, u):
    topology = as_float_array(T)
    coordinates = as_float_array(X)
    potentials = as_float_array(u).reshape(-1, 1)
    S = np.zeros((topology.shape[0], 8), dtype=float)
    E = np.zeros((topology.shape[0], 8), dtype=float)
    for i, row in enumerate(topology):
        nodes = topology_nodes(row)
        prop = topology_property(row)
        qe, Se, Ee = qeq4p(
            coordinates[nodes - 1], material_row(G, prop), potentials[nodes - 1]
        )
        q = assmq(q, qe, row, 1)
        S[i] = Se.reshape(1, 8)
        E[i] = Ee.reshape(1, 8)
    return q, S, E


def _ensure_state_width(state, element_count: int, width: int):
    state = as_float_array(state)
    if state.size == 0:
        return np.zeros((element_count, width), dtype=float)
    if state.ndim == 1:
        state = state.reshape(1, -1)
    if state.shape[1] < width:
        expanded = np.zeros((state.shape[0], width), dtype=float)
        expanded[:, : state.shape[1]] = state
        return expanded
    return state


def keq4eps(Xe, Ge, Se, Ee, mtype: int = 1):
    _VonMises = 1  # noqa: F841
    DruckerPrager = 2
    Xe = as_float_array(Xe)
    props = as_float_array(Ge).reshape(-1)
    Se = as_float_array(Se)
    Ee = as_float_array(Ee)

    r, w = _q4_gauss_points()
    E = props[0]
    nu = props[1]
    S0 = props[2]
    H = props[3]
    phi = props[4] if mtype == DruckerPrager and props.size == 5 else 0.0
    D = (
        E
        / (1.0 - nu**2)
        * np.array([[1.0, nu, 0.0], [nu, 1.0, 0.0], [0.0, 0.0, (1.0 - nu) / 2.0]])
    )
    mu = E / (2.0 * (1.0 + nu))
    k = E / (3.0 * (1.0 - nu))
    Ke = np.zeros((8, 8), dtype=float)

    for i in range(2):
        for j in range(2):
            gp = _q4_gp_index(i, j)
            dN = _q4_dN(r[i], r[j], 4)
            Jt = dN @ Xe
            dN_global = np.linalg.solve(Jt, dN)
            B = _q4_B(dN_global)

            Ep = Ee[gp, 3]
            Sy = S0 + Ep * H
            Sd, Sm = devstress(Se[gp, 0:3].reshape(-1, 1))
            Seq = Se[gp, 3]
            f = Seq + phi * Sm - Sy
            if f < 0.0:
                Dep = D
            else:
                a = np.array(
                    [
                        Sd[0, 0] + nu * Sd[1, 0],
                        nu * Sd[0, 0] + Sd[1, 0],
                        (1.0 - nu) * Sd[2, 0],
                    ]
                ).reshape(-1, 1)
                a = (3.0 * E / (2.0 * Seq * (1.0 - nu**2))) * a
                if phi == 0.0:
                    factor = (
                        3.0
                        * mu
                        * (
                            1.0
                            - (1.0 - 2.0 * nu)
                            * (3.0 * Sm) ** 2
                            / (6.0 * (1.0 - nu) * Seq**2)
                        )
                    )
                else:
                    a = a + k * phi * np.array([[1.0], [1.0], [0.0]])
                    b = (3.0 / (2.0 * Seq)) * np.array(
                        [[Sd[0, 0]], [Sd[1, 0]], [2.0 * Sd[2, 0]]]
                    ) + phi / 3.0 * np.array([[1.0], [1.0], [0.0]])
                    factor = float(b.T @ a)
                Dep = D - (a @ a.T) / (H + factor)
            Ke += w[i] * w[j] * (B.T @ Dep @ B) * np.linalg.det(Jt)
    return Ke


def qeq4eps(Xe, Ge, Ue, Se, Ee, mtype: int = 1):
    _VonMises = 1  # noqa: F841
    DruckerPrager = 2
    Xe = as_float_array(Xe)
    props = as_float_array(Ge).reshape(-1)
    Ue = as_float_array(Ue).reshape(-1, 1)
    Se = as_float_array(Se, copy=True)
    Ee = as_float_array(Ee, copy=True)
    r, w = _q4_gauss_points()

    E = props[0]
    nu = props[1]
    S0 = props[2]
    H = props[3]
    phi = props[4] if mtype == DruckerPrager and props.size == 5 else 0.0
    D = (
        E
        / (1.0 - nu**2)
        * np.array([[1.0, nu, 0.0], [nu, 1.0, 0.0], [0.0, 0.0, (1.0 - nu) / 2.0]])
    )
    qe = np.zeros((8, 1), dtype=float)

    for i in range(2):
        for j in range(2):
            gp = _q4_gp_index(i, j)
            dN = _q4_dN(r[i], r[j], 4)
            Jt = dN @ Xe
            dN_global = np.linalg.solve(Jt, dN)
            B = _q4_B(dN_global)

            e = B @ Ue
            dE = e - Ee[gp, 0:3].reshape(-1, 1)
            Ee[gp, 0:3] = e.ravel()
            dS = D @ dE
            S = Se[gp, 0:3].reshape(-1, 1) + dS
            Ep = Ee[gp, 3]
            Sy = S0 + Ep * H
            _Sd, Sm = devstress(S)
            Seq = eqstress(S)
            f = Seq + phi * Sm - Sy

            dL = 0.0
            if f >= 0.0:
                if phi == 0.0:
                    S, dL = stressvm(S, props, Sy)
                else:
                    S, dL = stressdp(S, props, Sy, dE, dS)

            Se[gp, 0:3] = S.ravel()
            Se[gp, 3] = eqstress(S)
            Ee[gp, 3] += dL
            qe += w[i] * w[j] * (B.T @ S) * np.linalg.det(Jt)

    return qe, Se, Ee


def kq4eps(K, T, X, G, S, E, mtype: int = 1):
    topology = as_float_array(T)
    coordinates = as_float_array(X)
    S = _ensure_state_width(S, topology.shape[0], 16)
    E = _ensure_state_width(E, topology.shape[0], 16)
    for i, row in enumerate(topology):
        nodes = topology_nodes(row)
        prop = topology_property(row)
        Se = S[i].reshape(4, 4)
        Ee = E[i].reshape(4, 4)
        K = assmk(
            K,
            keq4eps(coordinates[nodes - 1], material_row(G, prop), Se, Ee, mtype),
            row,
            2,
        )
    return K


def qq4eps(q, T, X, G, u, S, E, mtype: int = 1):
    topology = as_float_array(T)
    coordinates = as_float_array(X)
    U = as_float_array(u).reshape(coordinates.shape[0], coordinates.shape[1])
    S = _ensure_state_width(S, topology.shape[0], 16)
    E = _ensure_state_width(E, topology.shape[0], 16)
    Sn = np.zeros_like(S)
    En = np.zeros_like(E)
    for i, row in enumerate(topology):
        nodes = topology_nodes(row)
        prop = topology_property(row)
        Ue = U[nodes - 1].reshape(-1, 1)
        qe, Sen, Een = qeq4eps(
            coordinates[nodes - 1],
            material_row(G, prop),
            Ue,
            S[i].reshape(4, 4),
            E[i].reshape(4, 4),
            mtype,
        )
        q = assmq(q, qe, row, coordinates.shape[1])
        Sn[i] = Sen.reshape(1, 16)
        En[i] = Een.reshape(1, 16)
    return q, Sn, En


def keq4epe(Xe, Ge, Se, Ee, mtype: int = 1):
    DruckerPrager = 2
    Xe = as_float_array(Xe)
    props = as_float_array(Ge).reshape(-1)
    Se = as_float_array(Se)
    Ee = as_float_array(Ee)

    E = props[0]
    nu = props[1]
    S0 = props[2]
    H = props[3]
    phi = props[4] if mtype == DruckerPrager and props.size == 5 else 0.0
    D = _plane_elastic_matrix(props, plane_strain=True)
    mu = E / (2.0 * (1.0 + nu))
    k = E / (3.0 * (1.0 - nu))

    dN0 = np.array([[-1.0, 1.0, 1.0, -1.0], [-1.0, -1.0, 1.0, 1.0]], dtype=float) / 4.0
    Jt0 = dN0 @ Xe
    dN0 = np.linalg.solve(Jt0, dN0)
    W = np.array(
        [
            dN0[0, 0],
            dN0[1, 0],
            dN0[0, 1],
            dN0[1, 1],
            dN0[0, 2],
            dN0[1, 2],
            dN0[0, 3],
            dN0[1, 3],
        ]
    )
    m = np.array([[1.0], [1.0], [0.0], [0.0]])
    I4 = np.eye(4, dtype=float)

    Ke = np.zeros((8, 8), dtype=float)
    r, w = _q4_gauss_points()
    for i in range(2):
        for j in range(2):
            gp = _q4_gp_index(i, j)
            dN = _q4_dN(r[i], r[j], 4)
            Jt = dN @ Xe
            dN_global = np.linalg.solve(Jt, dN)
            B = np.array(
                [
                    [
                        dN_global[0, 0],
                        0.0,
                        dN_global[0, 1],
                        0.0,
                        dN_global[0, 2],
                        0.0,
                        dN_global[0, 3],
                        0.0,
                    ],
                    [
                        0.0,
                        dN_global[1, 0],
                        0.0,
                        dN_global[1, 1],
                        0.0,
                        dN_global[1, 2],
                        0.0,
                        dN_global[1, 3],
                    ],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [
                        dN_global[1, 0],
                        dN_global[0, 0],
                        dN_global[1, 1],
                        dN_global[0, 1],
                        dN_global[1, 2],
                        dN_global[0, 2],
                        dN_global[1, 3],
                        dN_global[0, 3],
                    ],
                ],
                dtype=float,
            )
            B = (I4 - 0.5 * (m @ m.T)) @ B + 0.5 * (m @ W.reshape(1, -1))

            Ep = Ee[gp, 4]
            Sy = S0 + Ep * H
            Sd, Sm = devstress(Se[gp, 0:4].reshape(-1, 1))
            Seq = Se[gp, 4]
            f = Seq + phi * Sm - Sy
            if f < 0.0:
                Dep = D
            else:
                Dp = 9.0 * mu**2 * (Sd @ Sd.T) / (Seq**2)
                if phi != 0.0:
                    mp = np.array([[1.0], [1.0], [1.0], [0.0]])
                    Dp = Dp + (k * phi) ** 2 * (mp @ mp.T)
                factor = 3.0 * mu + k * phi**2
                Dep = D - Dp / (H + factor)
            Ke += w[i] * w[j] * (B.T @ Dep @ B) * np.linalg.det(Jt)
    return Ke


def qeq4epe(Xe, Ge, Ue, Se, Ee, mtype: int = 1):
    DruckerPrager = 2
    Xe = as_float_array(Xe)
    props = as_float_array(Ge).reshape(-1)
    Ue = as_float_array(Ue).reshape(-1, 1)
    Se = as_float_array(Se, copy=True)
    Ee = as_float_array(Ee, copy=True)

    E = props[0]
    nu = props[1]
    S0 = props[2]
    H = props[3]
    phi = props[4] if mtype == DruckerPrager and props.size == 5 else 0.0
    D = _plane_elastic_matrix(props, plane_strain=True)
    mu = E / (2.0 * (1.0 + nu))
    k = E / (3.0 * (1.0 - 2.0 * nu))

    dN0 = np.array([[-1.0, 1.0, 1.0, -1.0], [-1.0, -1.0, 1.0, 1.0]], dtype=float) / 4.0
    Jt0 = dN0 @ Xe
    dN0 = np.linalg.solve(Jt0, dN0)
    W = np.array(
        [
            dN0[0, 0],
            dN0[1, 0],
            dN0[0, 1],
            dN0[1, 1],
            dN0[0, 2],
            dN0[1, 2],
            dN0[0, 3],
            dN0[1, 3],
        ]
    )
    m = np.array([[1.0], [1.0], [0.0], [0.0]])
    I4 = np.eye(4, dtype=float)

    qe = np.zeros((8, 1), dtype=float)
    r, w = _q4_gauss_points()
    for i in range(2):
        for j in range(2):
            gp = _q4_gp_index(i, j)
            dN = _q4_dN(r[i], r[j], 4)
            Jt = dN @ Xe
            dN_global = np.linalg.solve(Jt, dN)
            B = np.array(
                [
                    [
                        dN_global[0, 0],
                        0.0,
                        dN_global[0, 1],
                        0.0,
                        dN_global[0, 2],
                        0.0,
                        dN_global[0, 3],
                        0.0,
                    ],
                    [
                        0.0,
                        dN_global[1, 0],
                        0.0,
                        dN_global[1, 1],
                        0.0,
                        dN_global[1, 2],
                        0.0,
                        dN_global[1, 3],
                    ],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [
                        dN_global[1, 0],
                        dN_global[0, 0],
                        dN_global[1, 1],
                        dN_global[0, 1],
                        dN_global[1, 2],
                        dN_global[0, 2],
                        dN_global[1, 3],
                        dN_global[0, 3],
                    ],
                ],
                dtype=float,
            )
            B = (I4 - 0.5 * (m @ m.T)) @ B + 0.5 * (m @ W.reshape(1, -1))

            e = B @ Ue
            dE = e - Ee[gp, 0:4].reshape(-1, 1)
            Ee[gp, 0:4] = e.ravel()
            dS = D @ dE
            S = Se[gp, 0:4].reshape(-1, 1) + dS
            Ep = Ee[gp, 4]
            Sy = S0 + Ep * H
            Sd, Sm = devstress(S)
            Seq = eqstress(S)
            f = Seq + phi * Sm - Sy

            dL = 0.0
            if f >= 0.0:
                dL = f / (H + 3.0 * mu + k * phi**2)
                mp = np.array([[1.0], [1.0], [1.0], [0.0]])
                S = S - dL * (3.0 * mu * Sd / Seq + k * phi * mp)

            Se[gp, 0:4] = S.ravel()
            Se[gp, 4] = eqstress(S)
            Ee[gp, 4] += dL
            qe += w[i] * w[j] * (B.T @ S) * np.linalg.det(Jt)
    return qe, Se, Ee


def kq4epe(K, T, X, G, S, E, mtype: int = 1):
    topology = as_float_array(T)
    coordinates = as_float_array(X)
    S = _ensure_state_width(S, topology.shape[0], 20)
    E = _ensure_state_width(E, topology.shape[0], 20)
    for i, row in enumerate(topology):
        nodes = topology_nodes(row)
        prop = topology_property(row)
        K = assmk(
            K,
            keq4epe(
                coordinates[nodes - 1],
                material_row(G, prop),
                S[i].reshape(4, 5),
                E[i].reshape(4, 5),
                mtype,
            ),
            row,
            2,
        )
    return K


def qq4epe(q, T, X, G, u, S, E, mtype: int = 1):
    topology = as_float_array(T)
    coordinates = as_float_array(X)
    U = as_float_array(u).reshape(coordinates.shape[0], coordinates.shape[1])
    S = _ensure_state_width(S, topology.shape[0], 20)
    E = _ensure_state_width(E, topology.shape[0], 20)
    Sn = np.zeros_like(S)
    En = np.zeros_like(E)
    for i, row in enumerate(topology):
        nodes = topology_nodes(row)
        prop = topology_property(row)
        Ue = U[nodes - 1].reshape(-1, 1)
        qe, Sen, Een = qeq4epe(
            coordinates[nodes - 1],
            material_row(G, prop),
            Ue,
            S[i].reshape(4, 5),
            E[i].reshape(4, 5),
            mtype,
        )
        q = assmq(q, qe, row, coordinates.shape[1])
        Sn[i] = Sen.reshape(1, 20)
        En[i] = Een.reshape(1, 20)
    return q, Sn, En


__all__ = [
    "keq4e",
    "keq4epe",
    "keq4eps",
    "keq4p",
    "kq4e",
    "kq4epe",
    "kq4eps",
    "kq4p",
    "qeq4e",
    "qeq4epe",
    "qeq4eps",
    "qeq4p",
    "qq4e",
    "qq4epe",
    "qq4eps",
    "qq4p",
]
