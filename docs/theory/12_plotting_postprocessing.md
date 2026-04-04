# Plotting and Post-Processing

In finite element analysis, visualizing the results and calculating reaction forces are crucial steps after solving the global system of equations. This document details the procedures used in our software.

## Mesh Plotting with `PolyCollection`

In `plotting.py`, rendering the finite element mesh element-by-element using standard plot commands is extremely slow for large meshes. Instead, it uses `matplotlib.collections.PolyCollection`.

A `PolyCollection` allows `matplotlib` to draw a large number of polygons in a single efficient step. The implementation works as follows:
1. The global coordinates of the nodes for each element are gathered into a 3D NumPy array of shape `(num_elements, num_nodes_per_element, 2)`.
2. This array is passed directly into the `PolyCollection` constructor.
3. The resulting collection is then added to the plot's `Axes` object. This allows for fast and seamless rendering of the entire mesh, as well as easy mapping of scalar arrays (like extrapolated nodal values) to a colormap for contour plotting.

## Stress Extrapolation in `plotq4`

In a displacement-based Q4 (bilinear quadrilateral) element formulation, stresses and strains are evaluated at the integration points (Gauss points) inside the element. For a 2x2 Gauss quadrature rule, there are 4 Gauss points.

However, continuous contour plotting (Gouraud shading) requires scalar values at the element nodes. In `plotq4`, this is achieved by extrapolating the Gauss point stresses to the corner nodes:
1. A local coordinate system is defined where the 4 Gauss points are treated as the "corners" of an auxiliary square ($\xi, \eta = \pm 1/\sqrt{3}$).
2. An extrapolation matrix $\mathbf{E}$ is constructed using standard bilinear shape functions evaluated at the actual node locations ($\xi, \eta = \pm 1$) relative to the auxiliary square.
3. The Gauss point stresses for each element, $\boldsymbol{\sigma}_{GP}$, are multiplied by this matrix to yield the nodal stresses: 
   $$ \boldsymbol{\sigma}_{node} = \mathbf{E} \boldsymbol{\sigma}_{GP} $$
4. Because multiple elements share the same node, the extrapolated stress values from all elements meeting at a node are averaged to produce a smooth, continuous field for contour plotting.

## Reaction Forces Calculation

To compute the reaction forces at the constrained degrees of freedom (the supports), `postprocess.py` applies the global equilibrium equation:

$$ \mathbf{R} = \mathbf{K} \mathbf{u} - \mathbf{P} $$

Where:
*   $\mathbf{K}$ is the global stiffness matrix.
*   $\mathbf{u}$ is the full global displacement vector.
*   $\mathbf{P}$ is the vector of applied external nodal forces.
*   $\mathbf{R}$ is the vector of reaction forces.

Since external forces and internal forces balance out at free degrees of freedom, $\mathbf{R}$ will be zero at the free nodes and will contain the actual support reactions at the constrained nodes.

The exact Python code used to calculate the reaction forces is:

```python
R = K @ u - P
```