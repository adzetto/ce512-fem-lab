from __future__ import annotations

import numpy as np

from .._helpers import as_float_array, element_dof_indices, is_sparse

try:
    import scipy.sparse as sp
except ImportError:  # pragma: no cover
    sp = None


def _elastic3d_matrix(Ge):
    props = as_float_array(Ge).reshape(-1)
    E = props[0]
    nu = props[1]
    return (
        E
        / ((1.0 + nu) * (1.0 - 2.0 * nu))
        * np.array(
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
    )


def _elastic3d_matrix_batch(Ge):
    props = as_float_array(Ge)
    if props.ndim == 1:
        props = props.reshape(1, -1)
    E = props[:, 0]
    nu = props[:, 1]
    D = np.zeros((props.shape[0], 6, 6), dtype=float)
    D[:, 0, 0] = 1.0 - nu
    D[:, 0, 1] = nu
    D[:, 0, 2] = nu
    D[:, 1, 0] = nu
    D[:, 1, 1] = 1.0 - nu
    D[:, 1, 2] = nu
    D[:, 2, 0] = nu
    D[:, 2, 1] = nu
    D[:, 2, 2] = 1.0 - nu
    D[:, 3, 3] = (1.0 - 2.0 * nu) / 2.0
    D[:, 4, 4] = (1.0 - 2.0 * nu) / 2.0
    D[:, 5, 5] = (1.0 - 2.0 * nu) / 2.0
    return D * (E / ((1.0 + nu) * (1.0 - 2.0 * nu)))[:, None, None]


def _solid_B(dN):
    nnodes = dN.shape[1]
    B = np.zeros((6, nnodes * 3), dtype=float)
    B[0, 0::3] = dN[0]
    B[1, 1::3] = dN[1]
    B[2, 2::3] = dN[2]
    B[3, 0::3] = dN[1]
    B[3, 1::3] = dN[0]
    B[4, 1::3] = dN[2]
    B[4, 2::3] = dN[1]
    B[5, 0::3] = dN[2]
    B[5, 2::3] = dN[0]
    return B


def _solid_B_batch(dN):
    nnodes = dN.shape[-1]
    B = np.zeros(dN.shape[:-2] + (6, nnodes * 3), dtype=float)
    B[..., 0, 0::3] = dN[..., 0, :]
    B[..., 1, 1::3] = dN[..., 1, :]
    B[..., 2, 2::3] = dN[..., 2, :]
    B[..., 3, 0::3] = dN[..., 1, :]
    B[..., 3, 1::3] = dN[..., 0, :]
    B[..., 4, 1::3] = dN[..., 2, :]
    B[..., 4, 2::3] = dN[..., 1, :]
    B[..., 5, 0::3] = dN[..., 2, :]
    B[..., 5, 2::3] = dN[..., 0, :]
    return B


def keT4e(Xe, Ge):
    """Compute the element stiffness matrix for a 4-node tetrahedral solid."""
    Xe = as_float_array(Xe)
    dN = np.array(
        [[1.0, 0.0, 0.0, -1.0], [0.0, 1.0, 0.0, -1.0], [0.0, 0.0, 1.0, -1.0]],
        dtype=float,
    )
    J = dN @ Xe
    dN = np.linalg.solve(J, dN)
    B = _solid_B(dN)
    D = _elastic3d_matrix(Ge)
    return 2.0 * (B.T @ D @ B) * np.linalg.det(J)


def qeT4e(Xe, Ge, Ue):
    """Compute stress and strain results for one tetrahedral solid element."""
    Xe = as_float_array(Xe)
    Ue = as_float_array(Ue).reshape(-1, 1)
    dN = np.array(
        [[1.0, 0.0, 0.0, -1.0], [0.0, 1.0, 0.0, -1.0], [0.0, 0.0, 1.0, -1.0]],
        dtype=float,
    )
    J = dN @ Xe
    dN = np.linalg.solve(J, dN)
    B = _solid_B(dN)
    D = _elastic3d_matrix(Ge)
    Ee = (B @ Ue).reshape(-1)
    Se = Ee @ D
    qe = (B.T @ Se.reshape(-1, 1)) * np.linalg.det(J)
    return qe, Se, Ee


def kT4e(K, T, X, G):
    """Assemble T4 solid element stiffness contributions into the global matrix."""
    topology = as_float_array(T)
    coordinates = as_float_array(X)
    nodes = topology[:, :4].astype(int) - 1
    materials = as_float_array(G)[topology[:, -1].astype(int) - 1]
    Xe = coordinates[nodes]
    dN = np.array(
        [[1.0, 0.0, 0.0, -1.0], [0.0, 1.0, 0.0, -1.0], [0.0, 0.0, 1.0, -1.0]],
        dtype=float,
    )
    J = np.einsum("ij,ejk->eik", dN, Xe)
    dN_global = np.linalg.solve(J, np.broadcast_to(dN, (topology.shape[0],) + dN.shape))
    B = _solid_B_batch(dN_global)
    D = _elastic3d_matrix_batch(materials)
    detJ = np.linalg.det(J)
    element_matrices = 2.0 * detJ[:, None, None] * np.einsum(
        "eik,ekl,elj->eij", B.transpose(0, 2, 1), D, B
    )
    indices = element_dof_indices(nodes, 3, one_based=False)
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


def qT4e(q, T, X, G, u):
    """Compute T4 solid stresses and assemble internal forces."""
    topology = as_float_array(T)
    coordinates = as_float_array(X)
    U = as_float_array(u).reshape(coordinates.shape[0], coordinates.shape[1])
    nodes = topology[:, :4].astype(int) - 1
    materials = as_float_array(G)[topology[:, -1].astype(int) - 1]
    Xe = coordinates[nodes]
    dN = np.array(
        [[1.0, 0.0, 0.0, -1.0], [0.0, 1.0, 0.0, -1.0], [0.0, 0.0, 1.0, -1.0]],
        dtype=float,
    )
    J = np.einsum("ij,ejk->eik", dN, Xe)
    dN_global = np.linalg.solve(J, np.broadcast_to(dN, (topology.shape[0],) + dN.shape))
    B = _solid_B_batch(dN_global)
    D = _elastic3d_matrix_batch(materials)
    detJ = np.linalg.det(J)
    element_displacements = U[nodes].reshape(topology.shape[0], -1)
    E = np.einsum("eij,ej->ei", B, element_displacements)
    S = np.einsum("ei,eij->ej", E, D)
    element_vectors = detJ[:, None] * np.einsum(
        "eij,ej->ei", B.transpose(0, 2, 1), S
    )
    indices = element_dof_indices(nodes, coordinates.shape[1], one_based=False)
    np.add.at(q[:, 0], indices.reshape(-1), element_vectors.reshape(-1))
    return q, S, E


def _hexa_dN(r_i: float, r_j: float, r_k: float):
    dNi = (
        np.array(
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
        )
        / 8.0
    )
    dNj = (
        np.array(
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
        )
        / 8.0
    )
    dNk = (
        np.array(
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
        )
        / 8.0
    )
    return np.vstack([dNi, dNj, dNk])


def _hexa_dN_batch(points):
    points = as_float_array(points)
    r_i = points[:, 0][:, None]
    r_j = points[:, 1][:, None]
    r_k = points[:, 2][:, None]
    dNi = np.hstack(
        [
            -(1.0 - r_j) * (1.0 - r_k),
            (1.0 - r_j) * (1.0 - r_k),
            (1.0 + r_j) * (1.0 - r_k),
            -(1.0 + r_j) * (1.0 - r_k),
            -(1.0 - r_j) * (1.0 + r_k),
            (1.0 - r_j) * (1.0 + r_k),
            (1.0 + r_j) * (1.0 + r_k),
            -(1.0 + r_j) * (1.0 + r_k),
        ]
    ) / 8.0
    dNj = np.hstack(
        [
            -(1.0 - r_i) * (1.0 - r_k),
            -(1.0 + r_i) * (1.0 - r_k),
            (1.0 + r_i) * (1.0 - r_k),
            (1.0 - r_i) * (1.0 - r_k),
            -(1.0 - r_i) * (1.0 + r_k),
            -(1.0 + r_i) * (1.0 + r_k),
            (1.0 + r_i) * (1.0 + r_k),
            (1.0 - r_i) * (1.0 + r_k),
        ]
    ) / 8.0
    dNk = np.hstack(
        [
            -(1.0 - r_i) * (1.0 - r_j),
            -(1.0 + r_i) * (1.0 - r_j),
            -(1.0 + r_i) * (1.0 + r_j),
            -(1.0 - r_i) * (1.0 + r_j),
            (1.0 - r_i) * (1.0 - r_j),
            (1.0 + r_i) * (1.0 - r_j),
            (1.0 + r_i) * (1.0 + r_j),
            (1.0 - r_i) * (1.0 + r_j),
        ]
    ) / 8.0
    return np.stack([dNi, dNj, dNk], axis=1)


def keh8e(Xe, Ge):
    """Compute the element stiffness matrix for an 8-node hexahedral solid."""
    Xe = as_float_array(Xe)
    if Xe.shape[0] == 20:
        raise NotImplementedError(
            "20-node hexahedra are not supported yet, matching the Scilab limitation."
        )
    D = _elastic3d_matrix(Ge)
    gauss_points = np.array(
        [
            [-1.0, -1.0, -1.0],
            [1.0, -1.0, -1.0],
            [1.0, 1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, -1.0, 1.0],
            [1.0, -1.0, 1.0],
            [1.0, 1.0, 1.0],
            [-1.0, 1.0, 1.0],
        ],
        dtype=float,
    ) / np.sqrt(3.0)
    dN = _hexa_dN_batch(gauss_points)
    Jt = np.einsum("gik,kj->gij", dN, Xe)
    dN_global = np.linalg.solve(Jt, dN)
    B = _solid_B_batch(dN_global)
    return np.einsum(
        "g,gik,kl,glj->ij",
        np.linalg.det(Jt),
        B.transpose(0, 2, 1),
        D,
        B,
    )


def qeh8e(Xe, Ge, Ue):
    """Compute stress and strain results for one H8 solid element."""
    Xe = as_float_array(Xe)
    if Xe.shape[0] == 20:
        raise NotImplementedError(
            "20-node hexahedra are not supported yet, matching the Scilab limitation."
        )
    D = _elastic3d_matrix(Ge)
    Ue = as_float_array(Ue).reshape(-1, 1)
    gauss_points = np.array(
        [
            [-1.0, -1.0, -1.0],
            [1.0, -1.0, -1.0],
            [1.0, 1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, -1.0, 1.0],
            [1.0, -1.0, 1.0],
            [1.0, 1.0, 1.0],
            [-1.0, 1.0, 1.0],
        ],
        dtype=float,
    ) / np.sqrt(3.0)
    dN = _hexa_dN_batch(gauss_points)
    Jt = np.einsum("gik,kj->gij", dN, Xe)
    dN_global = np.linalg.solve(Jt, dN)
    B = _solid_B_batch(dN_global)
    Ee = np.einsum("gij,jk->gi", B, Ue).reshape(8, 6)
    Se = np.einsum("gi,ij->gj", Ee, D)
    qe = np.einsum(
        "g,gij,gj->i",
        np.linalg.det(Jt),
        B.transpose(0, 2, 1),
        Se,
    ).reshape(-1, 1)
    return qe, Se, Ee


def kh8e(K, T, X, G):
    """Assemble H8 solid element stiffness contributions into the global matrix."""
    topology = as_float_array(T)
    coordinates = as_float_array(X)
    nodes = topology[:, :8].astype(int) - 1
    materials = as_float_array(G)[topology[:, -1].astype(int) - 1]
    Xe = coordinates[nodes]
    gauss_points = np.array(
        [
            [-1.0, -1.0, -1.0],
            [1.0, -1.0, -1.0],
            [1.0, 1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, -1.0, 1.0],
            [1.0, -1.0, 1.0],
            [1.0, 1.0, 1.0],
            [-1.0, 1.0, 1.0],
        ],
        dtype=float,
    ) / np.sqrt(3.0)
    dN = _hexa_dN_batch(gauss_points)
    Jt = np.einsum("gik,ekj->egij", dN, Xe)
    dN_global = np.linalg.solve(
        Jt, np.broadcast_to(dN, (topology.shape[0],) + dN.shape)
    )
    B = _solid_B_batch(dN_global)
    D = _elastic3d_matrix_batch(materials)
    detJ = np.linalg.det(Jt)
    element_matrices = np.einsum(
        "eg,egik,ekl,eglj->eij",
        detJ,
        B.transpose(0, 1, 3, 2),
        D,
        B,
    )
    indices = element_dof_indices(nodes, 3, one_based=False)
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


def qh8e(q, T, X, G, u):
    """Compute H8 solid stresses and assemble internal forces."""
    topology = as_float_array(T)
    coordinates = as_float_array(X)
    U = as_float_array(u).reshape(coordinates.shape[0], coordinates.shape[1])
    nodes = topology[:, :8].astype(int) - 1
    materials = as_float_array(G)[topology[:, -1].astype(int) - 1]
    Xe = coordinates[nodes]
    gauss_points = np.array(
        [
            [-1.0, -1.0, -1.0],
            [1.0, -1.0, -1.0],
            [1.0, 1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, -1.0, 1.0],
            [1.0, -1.0, 1.0],
            [1.0, 1.0, 1.0],
            [-1.0, 1.0, 1.0],
        ],
        dtype=float,
    ) / np.sqrt(3.0)
    dN = _hexa_dN_batch(gauss_points)
    Jt = np.einsum("gik,ekj->egij", dN, Xe)
    dN_global = np.linalg.solve(
        Jt, np.broadcast_to(dN, (topology.shape[0],) + dN.shape)
    )
    B = _solid_B_batch(dN_global)
    D = _elastic3d_matrix_batch(materials)
    detJ = np.linalg.det(Jt)
    element_displacements = U[nodes].reshape(topology.shape[0], -1)
    E = np.einsum("egij,ej->egi", B, element_displacements).reshape(topology.shape[0], -1)
    strain_gp = E.reshape(topology.shape[0], 8, 6)
    stress_gp = np.einsum("egi,eij->egj", strain_gp, D)
    element_vectors = np.einsum(
        "eg,egij,egj->ei",
        detJ,
        B.transpose(0, 1, 3, 2),
        stress_gp,
    )
    indices = element_dof_indices(nodes, coordinates.shape[1], one_based=False)
    np.add.at(q[:, 0], indices.reshape(-1), element_vectors.reshape(-1))
    return q, stress_gp.reshape(topology.shape[0], -1), E


__all__ = ["keT4e", "keh8e", "kT4e", "kh8e", "qeT4e", "qeh8e", "qT4e", "qh8e"]
