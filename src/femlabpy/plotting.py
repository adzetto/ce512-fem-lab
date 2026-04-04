from __future__ import annotations

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.collections import PolyCollection
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from ._helpers import as_float_array


def _axis(ax=None, ndim: int = 2):
    if ax is not None:
        return ax
    if ndim == 3:
        fig = plt.figure()
        return fig.add_subplot(111, projection="3d")
    _, axis = plt.subplots()
    return axis


def plotelem(
    T, X, line_style: str = "k-", nonum: bool = False, noelem: bool = False, ax=None
):
    """Plot the undeformed mesh and optionally annotate node or element numbers.

    Algorithm
    ---------
    1. Extract element nodal coordinates.
    2. Build a `matplotlib.collections.PolyCollection`.
    3. Map internal stresses to a colormap using `vmin` and `vmax`.
    """
    topology = as_float_array(T).astype(int)
    coords = as_float_array(X)
    ndim = coords.shape[1]
    ax = _axis(ax, ndim)
    if ndim == 1:
        coords = np.column_stack([coords[:, 0], np.zeros(coords.shape[0])])
        ndim = 2
    for idx, row in enumerate(topology, start=1):
        nodes = row[:-1] - 1
        order = list(nodes)
        if len(order) >= 3:
            order.append(order[0])
        points = coords[order]
        if ndim == 2:
            ax.plot(points[:, 0], points[:, 1], line_style)
            if nonum:
                for node in nodes:
                    ax.text(coords[node, 0], coords[node, 1], str(node + 1), fontsize=8)
            if noelem:
                center = coords[nodes].mean(axis=0)
                ax.text(center[0], center[1], str(idx), fontsize=8)
        else:
            ax.plot(points[:, 0], points[:, 1], points[:, 2], line_style)
    if ndim == 2:
        ax.set_aspect("equal", adjustable="box")
    return ax


def plotforces(T, X, P, ax=None):
    """Plot nodal loads as arrows on a 2D mesh view.

    Algorithm
    ---------
    1. Extract element nodal coordinates.
    2. Build a `matplotlib.collections.PolyCollection`.
    3. Map internal stresses to a colormap using `vmin` and `vmax`.
    """
    _ = T
    coords = as_float_array(X)
    loads = as_float_array(P)
    ax = _axis(ax, 2)
    if loads.size == 0:
        return ax
    max_force = max(np.max(np.abs(loads[:, 1:3])), 1.0)
    span = np.ptp(coords[:, :2], axis=0)
    scale = 0.5 * max(span.max(), 1.0)
    for row in loads:
        node = int(row[0]) - 1
        x0, y0 = coords[node, :2]
        dx = scale * row[1] / max_force
        dy = scale * row[2] / max_force
        ax.arrow(
            x0, y0, dx, dy, color="tab:green", width=0.002, length_includes_head=True
        )
    return ax


def plotbc(T, X, C, ax=None):
    """Plot prescribed boundary conditions on a 2D mesh view.

    Algorithm
    ---------
    1. Extract element nodal coordinates.
    2. Build a `matplotlib.collections.PolyCollection`.
    3. Map internal stresses to a colormap using `vmin` and `vmax`.
    """
    _ = T
    coords = as_float_array(X)
    constraints = as_float_array(C)
    ax = _axis(ax, 2)
    span = np.ptp(coords[:, :2], axis=0)
    scale = 0.1 * max(span.max(), 1.0)
    for row in constraints:
        node = int(row[0]) - 1
        dof = int(row[1])
        x0, y0 = coords[node, :2]
        value = row[-1]
        if value == 0:
            if dof == 1:
                ax.scatter([x0 - scale], [y0], color="black", marker="s", s=20)
            else:
                ax.scatter([x0], [y0 - scale], color="black", marker="s", s=20)
        else:
            dx = value if dof == 1 else 0.0
            dy = value if dof == 2 else 0.0
            ax.arrow(
                x0, y0, dx, dy, color="tab:red", width=0.002, length_includes_head=True
            )
    return ax


def _triangulate_quads(T):
    topology = as_float_array(T).astype(int)
    triangles = []
    for row in topology:
        nodes = row[:-1]
        triangles.append([nodes[0], nodes[1], nodes[2]])
        triangles.append([nodes[0], nodes[2], nodes[3]])
    return np.asarray(triangles, dtype=int)


def plotq4(T, X, S, scomp: int, ax=None):
    """Plot a contour field reconstructed from Q4 Gauss-point results.

    Algorithm
    ---------
    1. Extract element nodal coordinates.
    2. Build a `matplotlib.collections.PolyCollection`.
    3. Map internal stresses to a colormap using `vmin` and `vmax`.
    """
    topology = as_float_array(T).astype(int)
    coords = as_float_array(X)
    values = as_float_array(S)
    ncomp = values.shape[1] // 4
    if scomp > ncomp:
        raise ValueError(f"Requested component {scomp} is not available.")
    r = np.array([-1.0, 1.0], dtype=float) * np.sqrt(3.0)
    N = np.zeros((4, 4), dtype=float)
    for i in range(2):
        for j in range(2):
            gp = i + 3 * j - 2 * i * j
            N[gp] = (
                np.array(
                    [
                        (1.0 - r[i]) * (1.0 - r[j]),
                        (1.0 + r[i]) * (1.0 - r[j]),
                        (1.0 + r[i]) * (1.0 + r[j]),
                        (1.0 - r[i]) * (1.0 + r[j]),
                    ],
                    dtype=float,
                )
                / 4.0
            )
    nodal_values = np.zeros((coords.shape[0],), dtype=float)
    counts = np.zeros((coords.shape[0],), dtype=float)
    component_index = scomp - 1
    for element, row in enumerate(topology):
        nodes = row[:-1] - 1
        gauss_values = values[element, component_index::ncomp]
        node_values = N @ gauss_values.reshape(-1, 1)
        nodal_values[nodes] += node_values.ravel()
        counts[nodes] += 1.0
    nodal_values /= np.maximum(counts, 1.0)
    triangles = _triangulate_quads(topology) - 1
    ax = _axis(ax, 2)
    trip = ax.tripcolor(
        coords[:, 0], coords[:, 1], triangles, nodal_values, shading="gouraud"
    )
    ax.set_aspect("equal", adjustable="box")
    plt.colorbar(trip, ax=ax)
    return ax


def plott3(T, X, S, scomp: int, ax=None):
    """Plot a contour field from T3 element results.

    Algorithm
    ---------
    1. Extract element nodal coordinates.
    2. Build a `matplotlib.collections.PolyCollection`.
    3. Map internal stresses to a colormap using `vmin` and `vmax`.
    """
    topology = as_float_array(T).astype(int)
    coords = as_float_array(X)
    values = as_float_array(S)
    if scomp > values.shape[1]:
        raise ValueError(f"Requested component {scomp} is not available.")
    nodal_values = np.zeros((coords.shape[0],), dtype=float)
    counts = np.zeros((coords.shape[0],), dtype=float)
    for element, row in enumerate(topology):
        nodes = row[:-1] - 1
        nodal_values[nodes] += values[element, scomp - 1]
        counts[nodes] += 1.0
    nodal_values /= np.maximum(counts, 1.0)
    triangles = topology[:, :-1] - 1
    ax = _axis(ax, 2)
    trip = ax.tripcolor(
        coords[:, 0], coords[:, 1], triangles, nodal_values, shading="flat"
    )
    ax.set_aspect("equal", adjustable="box")
    plt.colorbar(trip, ax=ax)
    return ax


def plotu(T, X, u, ax=None):
    """Plot a scalar nodal field over a 2D or 3D mesh.

    Algorithm
    ---------
    1. Extract element nodal coordinates.
    2. Build a `matplotlib.collections.PolyCollection`.
    3. Map internal stresses to a colormap using `vmin` and `vmax`.
    """
    topology = as_float_array(T).astype(int)
    coords = as_float_array(X)
    values = as_float_array(u).reshape(-1)
    ndim = coords.shape[1]
    ax = _axis(ax, ndim)
    if ndim == 2:
        polygons = []
        colors = []
        for row in topology:
            nodes = row[:-1] - 1
            polygons.append(coords[nodes, :2])
            colors.append(values[nodes].mean())
        collection = PolyCollection(
            polygons, array=np.asarray(colors), cmap="viridis", edgecolors="k"
        )
        ax.add_collection(collection)
        ax.autoscale()
        ax.set_aspect("equal", adjustable="box")
        plt.colorbar(collection, ax=ax)
        return ax

    polygons3d = []
    colors = []
    for row in topology:
        nodes = row[:-1] - 1
        polygons3d.append(coords[nodes, :3])
        colors.append(values[nodes].mean())
    collection = Poly3DCollection(
        polygons3d, array=np.asarray(colors), cmap="viridis", edgecolors="k"
    )
    ax.add_collection3d(collection)
    ax.auto_scale_xyz(coords[:, 0], coords[:, 1], coords[:, 2])
    plt.colorbar(collection, ax=ax)
    return ax


__all__ = ["plotbc", "plotelem", "plotforces", "plotq4", "plott3", "plotu"]
