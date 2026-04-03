from __future__ import annotations

import numpy as np

from .._helpers import (
    as_float_array,
    cols,
    element_dof_indices,
    is_sparse,
    rows,
)

try:
    import scipy.sparse as sp
except ImportError:  # pragma: no cover
    sp = None


def kebar(Xe0, Xe1, Ge):
    """Compute the tangent stiffness matrix of a geometrically nonlinear bar element."""
    initial = as_float_array(Xe0)
    current = as_float_array(Xe1)
    props = as_float_array(Ge).reshape(-1)

    a0 = (initial[1] - initial[0]).reshape(-1, 1)
    l0 = float(np.linalg.norm(a0))
    a1 = (current[1] - current[0]).reshape(-1, 1)
    l1 = float(np.linalg.norm(a1))

    A = props[0]
    E = props[1] if props.size > 1 else 1.0
    strain = 0.5 * (l1**2 - l0**2) / l0**2
    normal_force = A * E * strain
    identity = np.eye(a0.shape[0], dtype=float)
    return (E * A / l0**3) * np.block(
        [[a1 @ a1.T, -a1 @ a1.T], [-a1 @ a1.T, a1 @ a1.T]]
    ) + (normal_force / l0) * np.block([[identity, -identity], [-identity, identity]])


def qebar(Xe0, Xe1, Ge):
    """Compute the internal-force response of a single geometrically nonlinear bar."""
    initial = as_float_array(Xe0)
    current = as_float_array(Xe1)
    props = as_float_array(Ge).reshape(-1)

    a0 = (initial[1] - initial[0]).reshape(-1, 1)
    l0 = float(np.linalg.norm(a0))
    a1 = (current[1] - current[0]).reshape(-1, 1)
    l1 = float(np.linalg.norm(a1))

    A = props[0]
    E = props[1] if props.size > 1 else 1.0
    strain = 0.5 * (l1**2 - l0**2) / l0**2
    stress = E * strain
    qe = (A * stress / l0) * np.vstack([-a1, a1])
    return qe, float(stress), float(strain)


def kbar(K, T, X, G, u=None):
    """Assemble bar or truss tangent stiffness contributions into the global matrix."""
    X = as_float_array(X)
    topology = as_float_array(T)
    if u is None:
        current = X
    else:
        current = X + as_float_array(u).reshape(rows(X), cols(X))
    element_nodes = topology[:, :-1].astype(int) - 1
    props = as_float_array(G)[topology[:, -1].astype(int) - 1]
    initial = X[element_nodes]
    current_nodes = current[element_nodes]
    a0 = initial[:, 1, :] - initial[:, 0, :]
    a1 = current_nodes[:, 1, :] - current_nodes[:, 0, :]
    l0 = np.linalg.norm(a0, axis=1)
    l1 = np.linalg.norm(a1, axis=1)
    area = props[:, 0]
    modulus = props[:, 1] if props.shape[1] > 1 else np.ones_like(area)
    strain = 0.5 * (l1**2 - l0**2) / l0**2
    normal_force = area * modulus * strain
    a1a1 = np.einsum("ei,ej->eij", a1, a1)
    identity = np.eye(cols(X), dtype=float)[None, :, :]
    axial = (modulus * area / l0**3)[:, None, None] * a1a1
    geometric = (normal_force / l0)[:, None, None] * identity
    upper = np.concatenate([axial + geometric, -(axial + geometric)], axis=2)
    lower = np.concatenate([-(axial + geometric), axial + geometric], axis=2)
    element_matrices = np.concatenate([upper, lower], axis=1)
    indices = element_dof_indices(element_nodes, cols(X), one_based=False)
    if is_sparse(K) and sp is not None:
        scatter_rows = np.broadcast_to(
            indices[:, :, None], element_matrices.shape
        ).reshape(-1)
        scatter_cols = np.broadcast_to(
            indices[:, None, :], element_matrices.shape
        ).reshape(-1)
        delta = sp.coo_matrix(
            (element_matrices.reshape(-1), (scatter_rows, scatter_cols)),
            shape=K.shape,
            dtype=float,
        )
        return (K.tocsr() + delta.tocsr()).tolil()
    np.add.at(K, (indices[:, :, None], indices[:, None, :]), element_matrices)
    return K


def qbar(q, T, X, G, u=None):
    """Assemble bar or truss internal forces and element output quantities."""
    X = as_float_array(X)
    topology = as_float_array(T)
    if u is None:
        current = X
    else:
        current = X + as_float_array(u).reshape(rows(X), cols(X))
    element_nodes = topology[:, :-1].astype(int) - 1
    props = as_float_array(G)[topology[:, -1].astype(int) - 1]
    initial = X[element_nodes]
    current_nodes = current[element_nodes]
    a0 = initial[:, 1, :] - initial[:, 0, :]
    a1 = current_nodes[:, 1, :] - current_nodes[:, 0, :]
    l0 = np.linalg.norm(a0, axis=1)
    l1 = np.linalg.norm(a1, axis=1)
    area = props[:, 0]
    modulus = props[:, 1] if props.shape[1] > 1 else np.ones_like(area)
    strain = (0.5 * (l1**2 - l0**2) / l0**2).reshape(-1, 1)
    stress = (modulus[:, None] * strain).reshape(-1, 1)
    element_vectors = (
        (area * stress[:, 0] / l0)[:, None] * np.concatenate([-a1, a1], axis=1)
    )
    indices = element_dof_indices(element_nodes, cols(X), one_based=False)
    np.add.at(q[:, 0], indices.reshape(-1), element_vectors.reshape(-1))
    return q, stress, strain


__all__ = ["kbar", "kebar", "qbar", "qebar"]
