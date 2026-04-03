from __future__ import annotations

import numpy as np

from ._helpers import as_float_array


def setload(p, P):
    """
    Set nodal loads from a load matrix P.

    Replaces existing values in the load vector at specified nodes.

    Parameters
    ----------
    p : ndarray, shape (ndof, 1)
        Load vector (modified in place).

    P : array_like, shape (nloads, dof+1)
        Load matrix. Each row: [node, Fx, Fy] for 2D or [node, Fx, Fy, Fz] for 3D.
        Node indices are 1-based.

    Returns
    -------
    p : ndarray
        Updated load vector.

    Examples
    --------
    >>> from femlabpy import init, setload
    >>> K, p, q = init(nn=10, dof=2)
    >>> # Apply forces: node 5 gets Fx=-100, Fy=0; node 10 gets Fx=0, Fy=-200
    >>> P = np.array([
    ...     [5, -100, 0],
    ...     [10, 0, -200],
    ... ])
    >>> p = setload(p, P)
    """
    p = as_float_array(p)
    loads = as_float_array(P)
    if loads.size == 0:
        return p
    dof = loads.shape[1] - 1
    indices = ((loads[:, [0]].astype(int) - 1) * dof + np.arange(dof)).reshape(-1)
    p[indices, 0] = loads[:, 1 : 1 + dof].reshape(-1)
    return p


def addload(p, P):
    """
    Add nodal loads from a load matrix P (accumulates, doesn't replace).

    Parameters
    ----------
    p : ndarray, shape (ndof, 1)
        Load vector (modified in place).

    P : array_like, shape (nloads, dof+1)
        Load matrix. Each row: [node, Fx, Fy, ...].

    Returns
    -------
    p : ndarray
        Updated load vector.

    See Also
    --------
    setload : Set loads (replaces existing values).

    Examples
    --------
    >>> p = addload(p, P)  # Adds to existing loads
    """
    p = as_float_array(p)
    loads = as_float_array(P)
    if loads.size == 0:
        return p
    dof = loads.shape[1] - 1
    indices = ((loads[:, [0]].astype(int) - 1) * dof + np.arange(dof)).reshape(-1)
    np.add.at(p[:, 0], indices, loads[:, 1 : 1 + dof].reshape(-1))
    return p


__all__ = ["addload", "setload"]
