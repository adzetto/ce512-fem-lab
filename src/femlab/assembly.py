from __future__ import annotations

import numpy as np

from ._helpers import as_float_array, is_sparse, node_dof_indices, topology_nodes


def assmk(K, Ke, Te, dof: int = 1):
    element_nodes = topology_nodes(Te)
    indices = node_dof_indices(element_nodes, dof)
    element_matrix = as_float_array(Ke)
    if is_sparse(K):
        K[np.ix_(indices, indices)] = K[np.ix_(indices, indices)] + element_matrix
    else:
        K[np.ix_(indices, indices)] += element_matrix
    return K


def assmq(q, qe, Te, dof: int = 1):
    element_nodes = topology_nodes(Te)
    indices = node_dof_indices(element_nodes, dof)
    q = as_float_array(q)
    qe = as_float_array(qe).reshape(-1, 1)
    q[indices, 0] += qe[:, 0]
    return q


__all__ = ["assmk", "assmq"]
