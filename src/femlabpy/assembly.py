from __future__ import annotations

import numpy as np

from ._helpers import as_float_array, is_sparse, node_dof_indices, topology_nodes


def assmk(K, Ke, Te, dof: int = 1):
    """
    Assemble one element stiffness matrix into the global stiffness matrix.

    Parameters
    ----------
    K:
        Global stiffness matrix updated in place.
    Ke:
        Element stiffness matrix ordered by the element's local DOFs.
    Te:
        Legacy FemLab topology row whose last entry stores the property id.
    dof:
        Degrees of freedom per node.

    Returns
    -------
    ndarray or sparse matrix
        Updated global stiffness matrix.
    """
    element_nodes = topology_nodes(Te)
    indices = node_dof_indices(element_nodes, dof)
    element_matrix = as_float_array(Ke)
    if is_sparse(K):
        K[np.ix_(indices, indices)] = K[np.ix_(indices, indices)] + element_matrix
    else:
        K[np.ix_(indices, indices)] += element_matrix
    return K


def assmq(q, qe, Te, dof: int = 1):
    """
    Assemble one element force vector into the global internal-force vector.

    Parameters
    ----------
    q:
        Global internal-force vector updated in place.
    qe:
        Element internal-force vector.
    Te:
        Legacy FemLab topology row whose last entry stores the property id.
    dof:
        Degrees of freedom per node.

    Returns
    -------
    ndarray
        Updated global internal-force vector.
    """
    element_nodes = topology_nodes(Te)
    indices = node_dof_indices(element_nodes, dof)
    q = as_float_array(q)
    qe = as_float_array(qe).reshape(-1, 1)
    q[indices, 0] += qe[:, 0]
    return q


__all__ = ["assmk", "assmq"]
