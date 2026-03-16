from __future__ import annotations

import numpy as np

from .._helpers import as_float_array, element_dof_indices, is_sparse

try:
    import scipy.sparse as sp
except ImportError:  # pragma: no cover
    sp = None


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


def _elastic_matrix_batch(materials, plane_strain):
    materials = as_float_array(materials)
    if materials.ndim == 1:
        materials = materials.reshape(1, -1)
    plane_strain = np.asarray(plane_strain, dtype=bool)
    if plane_strain.ndim == 0:
        plane_strain = np.full(materials.shape[0], bool(plane_strain), dtype=bool)
    modulus = materials[:, 0]
    nu = materials[:, 1]

    plane_stress_matrix = np.zeros((materials.shape[0], 3, 3), dtype=float)
    plane_stress_matrix[:, 0, 0] = 1.0
    plane_stress_matrix[:, 0, 1] = nu
    plane_stress_matrix[:, 1, 0] = nu
    plane_stress_matrix[:, 1, 1] = 1.0
    plane_stress_matrix[:, 2, 2] = (1.0 - nu) / 2.0
    plane_stress_matrix *= (modulus / (1.0 - nu**2))[:, None, None]

    plane_strain_matrix = np.zeros((materials.shape[0], 3, 3), dtype=float)
    plane_strain_matrix[:, 0, 0] = 1.0 - nu
    plane_strain_matrix[:, 0, 1] = nu
    plane_strain_matrix[:, 1, 0] = nu
    plane_strain_matrix[:, 1, 1] = 1.0 - nu
    plane_strain_matrix[:, 2, 2] = (1.0 - 2.0 * nu) / 2.0
    plane_strain_matrix *= (
        modulus / ((1.0 + nu) * (1.0 - 2.0 * nu))
    )[:, None, None]

    return np.where(
        plane_strain[:, None, None], plane_strain_matrix, plane_stress_matrix
    )


def _triangle_batch_geometry(Xe):
    Xe = as_float_array(Xe)
    edges = np.stack([Xe[:, 2] - Xe[:, 1], Xe[:, 0] - Xe[:, 2], Xe[:, 1] - Xe[:, 0]], axis=1)
    area = 0.5 * np.abs(
        (Xe[:, 1, 0] - Xe[:, 0, 0]) * (Xe[:, 2, 1] - Xe[:, 0, 1])
        - (Xe[:, 2, 0] - Xe[:, 0, 0]) * (Xe[:, 1, 1] - Xe[:, 0, 1])
    )
    return edges, area


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
    nodes = topology[:, :3].astype(int) - 1
    materials = as_float_array(G)[topology[:, -1].astype(int) - 1]
    edges, area = _triangle_batch_geometry(coordinates[nodes])
    dN = np.stack([-edges[:, :, 1], edges[:, :, 0]], axis=1) / (2.0 * area)[
        :, None, None
    ]
    B = np.zeros((topology.shape[0], 3, 6), dtype=float)
    B[:, 0, 0::2] = dN[:, 0, :]
    B[:, 1, 1::2] = dN[:, 1, :]
    B[:, 2, 0::2] = dN[:, 1, :]
    B[:, 2, 1::2] = dN[:, 0, :]
    plane_strain = (
        materials[:, 2].astype(int) == 2
        if materials.shape[1] > 2
        else np.zeros(topology.shape[0], dtype=bool)
    )
    D = _elastic_matrix_batch(materials, plane_strain)
    element_matrices = area[:, None, None] * np.einsum(
        "eik,ekl,elj->eij", B.transpose(0, 2, 1), D, B
    )
    indices = element_dof_indices(nodes, 2, one_based=False)
    if is_sparse(K) and sp is not None:
        scatter_rows = np.broadcast_to(indices[:, :, None], element_matrices.shape).reshape(-1)
        scatter_cols = np.broadcast_to(indices[:, None, :], element_matrices.shape).reshape(-1)
        delta = sp.coo_matrix(
            (element_matrices.reshape(-1), (scatter_rows, scatter_cols)),
            shape=K.shape,
            dtype=float,
        )
        return (K.tocsr() + delta.tocsr()).tolil()
    np.add.at(K, (indices[:, :, None], indices[:, None, :]), element_matrices)
    return K


def qt3e(q, T, X, G, u):
    topology = as_float_array(T)
    coordinates = as_float_array(X)
    U = as_float_array(u).reshape(coordinates.shape[0], coordinates.shape[1])
    nodes = topology[:, :3].astype(int) - 1
    materials = as_float_array(G)[topology[:, -1].astype(int) - 1]
    edges, area = _triangle_batch_geometry(coordinates[nodes])
    dN = np.stack([-edges[:, :, 1], edges[:, :, 0]], axis=1) / (2.0 * area)[
        :, None, None
    ]
    B = np.zeros((topology.shape[0], 3, 6), dtype=float)
    B[:, 0, 0::2] = dN[:, 0, :]
    B[:, 1, 1::2] = dN[:, 1, :]
    B[:, 2, 0::2] = dN[:, 1, :]
    B[:, 2, 1::2] = dN[:, 0, :]
    plane_strain = (
        materials[:, 2].astype(int) == 2
        if materials.shape[1] > 2
        else np.zeros(topology.shape[0], dtype=bool)
    )
    D = _elastic_matrix_batch(materials, plane_strain)
    element_displacements = U[nodes].reshape(topology.shape[0], -1)
    E = np.einsum("eij,ej->ei", B, element_displacements)
    S = np.einsum("ei,eij->ej", E, D)
    element_vectors = area[:, None] * np.einsum(
        "eij,ej->ei", B.transpose(0, 2, 1), S
    )
    indices = element_dof_indices(nodes, coordinates.shape[1], one_based=False)
    np.add.at(q[:, 0], indices.reshape(-1), element_vectors.reshape(-1))
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
        Ke = Ke + (b * area / 12.0) * np.array(
            [[2.0, 1.0, 1.0], [1.0, 2.0, 1.0], [1.0, 1.0, 2.0]]
        )
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
    nodes = topology[:, :3].astype(int) - 1
    materials = as_float_array(G)[topology[:, -1].astype(int) - 1]
    edges, area = _triangle_batch_geometry(coordinates[nodes])
    B = np.stack([-edges[:, :, 1], edges[:, :, 0]], axis=1) / (2.0 * area)[
        :, None, None
    ]
    conductivity = materials[:, 0]
    element_matrices = area[:, None, None] * conductivity[:, None, None] * np.einsum(
        "eik,ekj->eij", B.transpose(0, 2, 1), B
    )
    if materials.shape[1] > 1:
        element_matrices = element_matrices + (
            materials[:, 1] * area / 12.0
        )[:, None, None] * np.array(
            [[2.0, 1.0, 1.0], [1.0, 2.0, 1.0], [1.0, 1.0, 2.0]], dtype=float
        )
    indices = element_dof_indices(nodes, 1, one_based=False)
    if is_sparse(K) and sp is not None:
        scatter_rows = np.broadcast_to(indices[:, :, None], element_matrices.shape).reshape(-1)
        scatter_cols = np.broadcast_to(indices[:, None, :], element_matrices.shape).reshape(-1)
        delta = sp.coo_matrix(
            (element_matrices.reshape(-1), (scatter_rows, scatter_cols)),
            shape=K.shape,
            dtype=float,
        )
        return (K.tocsr() + delta.tocsr()).tolil()
    np.add.at(K, (indices[:, :, None], indices[:, None, :]), element_matrices)
    return K


def qt3p(q, T, X, G, u):
    topology = as_float_array(T)
    coordinates = as_float_array(X)
    potentials = as_float_array(u).reshape(-1, 1)
    nodes = topology[:, :3].astype(int) - 1
    materials = as_float_array(G)[topology[:, -1].astype(int) - 1]
    edges, area = _triangle_batch_geometry(coordinates[nodes])
    B = np.stack([-edges[:, :, 1], edges[:, :, 0]], axis=1) / (2.0 * area)[
        :, None, None
    ]
    conductivity = materials[:, 0]
    element_potentials = potentials[nodes, 0]
    E = np.einsum("eij,ej->ei", B, element_potentials)
    S = conductivity[:, None] * E
    element_vectors = area[:, None] * np.einsum(
        "eij,ej->ei", B.transpose(0, 2, 1), S
    )
    indices = element_dof_indices(nodes, 1, one_based=False)
    np.add.at(q[:, 0], indices.reshape(-1), element_vectors.reshape(-1))
    return q, S, E


__all__ = ["ket3e", "ket3p", "kt3e", "kt3p", "qet3e", "qet3p", "qt3e", "qt3p"]
