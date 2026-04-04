# Chapter 2: Element Library

The element library in `femlabpy` provides the mathematical mapping from the continuous differential equations of solid mechanics to the discrete algebraic space. This chapter details the shape functions, integration schemes, and strain-displacement matrices ($\mathbf{B}$) used for our core elements. As a student of computational mechanics, it is paramount that you understand not only the theory, but the precise programmatic execution of these formulations. This chapter goes deep into the Python implementation.

## 2.1 1D Bar Element (`bar`)

The bar element supports axial tension and compression. In `femlabpy`, it is formulated to handle large deformations by utilizing the Green-Lagrange strain measure.

### Kinematics and Strain
For large deformations, the axial strain $\epsilon$ is defined as:
$$ \epsilon = \frac{du}{dx} + \frac{1}{2}\left(\frac{du}{dx}\right)^2 $$

The internal virtual work gives rise to two stiffness components: the standard linear elastic material stiffness $\mathbf{K}_m$ and the geometric stiffness $\mathbf{K}_g$, which accounts for the effect of the internal axial force $N$ on the transverse stiffness.

$$ \mathbf{K}_e = \mathbf{K}_m + \mathbf{K}_g $$
$$ \mathbf{K}_m = \frac{EA}{L} \begin{bmatrix} 1 & -1 \\ -1 & 1 \end{bmatrix}, \quad \mathbf{K}_g = \frac{N}{L} \begin{bmatrix} 1 & -1 \\ -1 & 1 \end{bmatrix} $$

## 2.2 Constant Strain Triangle (`t3`)

The 3-node triangular element (CST) is the simplest 2D element. The displacement field is interpolated linearly using area coordinates $L_1, L_2, L_3$.

$$ u(x,y) = N_1 u_1 + N_2 u_2 + N_3 u_3 $$

Since the shape functions $N_i$ are linear, their spatial derivatives are constant over the element. Consequently, the strain-displacement matrix $\mathbf{B}$ is constant:

$$ \mathbf{B} = \frac{1}{2A} \begin{bmatrix}
y_{23} & 0 & y_{31} & 0 & y_{12} & 0 \\
0 & x_{32} & 0 & x_{13} & 0 & x_{21} \\
x_{32} & y_{23} & x_{13} & y_{31} & x_{21} & y_{12}
\end{bmatrix} $$

where $x_{ij} = x_i - x_j$ and $y_{ij} = y_i - y_j$, and $A$ is the element area. The element stiffness matrix is derived exactly without numerical integration:
$$ \mathbf{K}_e = \mathbf{B}^T \mathbf{D} \mathbf{B} A t $$

### Implementation of `kt3e`

Let us examine how this is mapped into Python using `numpy`. The function `kt3e(ex, ey, D, t)` computes the element stiffness matrix for a T3 element.

```python
import numpy as np

def kt3e(ex, ey, D, t):
    # Calculate the cyclic coordinate differences
    x1, x2, x3 = ex
    y1, y2, y3 = ey
    
    y23 = y2 - y3
    y31 = y3 - y1
    y12 = y1 - y2
    
    x32 = x3 - x2
    x13 = x1 - x3
    x21 = x2 - x1
    
    # Calculate twice the area of the triangle using the determinant
    detJ = x13 * y23 - x32 * y31
    A = detJ / 2.0
    
    # Assemble the constant strain-displacement matrix B
    B = np.array([
        [y23,   0, y31,   0, y12,   0],
        [  0, x32,   0, x13,   0, x21],
        [x32, y23, x13, y31, x21, y12]
    ]) / detJ  # Note division by 2A (detJ)
    
    # Compute the stiffness matrix: B^T * D * B * A * t
    K = B.T @ D @ B * A * t
    return K
```
*Professor's Note:* Notice how `detJ` elegantly represents twice the area. The `@` operator in Python performs matrix multiplication. Because $B$ is constant, we avoid any quadrature loop, making this element remarkably cheap to compute but prone to locking in bending and overly stiff behavior.

## 2.3 Isoparametric Quadrilateral (`q4`)

The 4-node quadrilateral element uses a bilinear isoparametric formulation. The element geometry and displacements are mapped from a natural coordinate system $(\xi, \eta) \in [-1, 1]$ to the physical system $(x, y)$.

### Shape Functions
The bilinear shape functions are:
$$ N_i(\xi, \eta) = \frac{1}{4} (1 + \xi_i \xi) (1 + \eta_i \eta) $$

### The Jacobian Matrix
To compute spatial derivatives with respect to $x$ and $y$, we apply the chain rule via the Jacobian matrix $\mathbf{J}$:

$$ \mathbf{J} = \begin{bmatrix}
\frac{\partial x}{\partial \xi} & \frac{\partial y}{\partial \xi} \\
\frac{\partial x}{\partial \eta} & \frac{\partial y}{\partial \eta}
\end{bmatrix} = \sum_{i=1}^4 \begin{bmatrix}
\frac{\partial N_i}{\partial \xi} x_i & \frac{\partial N_i}{\partial \xi} y_i \\
\frac{\partial N_i}{\partial \eta} x_i & \frac{\partial N_i}{\partial \eta} y_i
\end{bmatrix} $$

The Cartesian shape function derivatives are then evaluated by inverting the Jacobian:
$$ \begin{Bmatrix} \frac{\partial N_i}{\partial x} \\ \frac{\partial N_i}{\partial y} \end{Bmatrix} = \mathbf{J}^{-1} \begin{Bmatrix} \frac{\partial N_i}{\partial \xi} \\ \frac{\partial N_i}{\partial \eta} \end{Bmatrix} $$

### Gauss Quadrature
The stiffness matrix requires integrating over the element area. We use $2 \times 2$ Gauss-Legendre quadrature:

$$ \mathbf{K}_e = \int_{-1}^{1} \int_{-1}^{1} \mathbf{B}^T \mathbf{D} \mathbf{B} |\mathbf{J}| t \, d\xi \, d\eta \approx \sum_{i=1}^{2} \sum_{j=1}^{2} w_i w_j \mathbf{B}^T(\xi_i, \eta_j) \mathbf{D} \mathbf{B}(\xi_i, \eta_j) |\mathbf{J}(\xi_i, \eta_j)| t $$

where the integration points are $\xi_i, \eta_j \in \pm \frac{1}{\sqrt{3}}$ and weights $w_i = 1.0$.

### Implementation of `kq4e`

The python implementation `kq4e(ex, ey, D, t)` represents a fundamental concept in FEA: the element integration loop.

```python
def kq4e(ex, ey, D, t):
    # Gauss points and weights for 2x2 quadrature
    gauss_pts = [-1/np.sqrt(3), 1/np.sqrt(3)]
    weights = [1.0, 1.0]
    
    # Initialize the 8x8 element stiffness matrix
    K = np.zeros((8, 8))
    
    # Nodal natural coordinates corresponding to (xi, eta)
    node_coords = np.array([
        [-1, -1],
        [ 1, -1],
        [ 1,  1],
        [-1,  1]
    ])
    
    # Assemble the element coordinates matrix (4x2)
    coords = np.column_stack((ex, ey))
    
    # 2x2 Gauss Quadrature loop
    for i in range(2):
        for j in range(2):
            xi = gauss_pts[i]
            eta = gauss_pts[j]
            w = weights[i] * weights[j]
            
            # Derivatives of shape functions with respect to xi and eta
            # dN_dxi: 2x4 matrix
            dN_dxi = np.zeros((2, 4))
            for k in range(4):
                xi_k = node_coords[k, 0]
                eta_k = node_coords[k, 1]
                dN_dxi[0, k] = 0.25 * xi_k * (1 + eta_k * eta)
                dN_dxi[1, k] = 0.25 * eta_k * (1 + xi_k * xi)
                
            # The Jacobian Matrix J (2x2) = dN_dxi * coords
            J = dN_dxi @ coords
            detJ = np.linalg.det(J)
            
            # Cartesian derivatives of shape functions dN_dx
            # J * dN_dx = dN_dxi  =>  dN_dx = J^-1 * dN_dxi
            dN_dx = np.linalg.solve(J, dN_dxi)
            
            # Construct the Strain-Displacement matrix B (3x8)
            B = np.zeros((3, 8))
            for k in range(4):
                B[0, 2*k]   = dN_dx[0, k]
                B[1, 2*k+1] = dN_dx[1, k]
                B[2, 2*k]   = dN_dx[1, k]
                B[2, 2*k+1] = dN_dx[0, k]
                
            # Accumulate the stiffness matrix at this Gauss point
            K += B.T @ D @ B * detJ * w * t
            
    return K
```
*Professor's Note:*
1. **Jacobian Computation (`J = dN_dxi @ coords`):** This maps the parametric derivatives to the physical coordinate derivatives.
2. **Jacobian Inversion (`np.linalg.solve(J, dN_dxi)`):** Instead of explicitly computing `J_inv = np.linalg.inv(J)` and multiplying, it is significantly more numerically stable and efficient to solve the linear system `J * dN_dx = dN_dxi`. This computes $\frac{\partial N_i}{\partial x}$ and $\frac{\partial N_i}{\partial y}$ for all nodes simultaneously.
3. **Integration Weight (`detJ * w * t`):** The differential area $dx dy$ maps to $detJ d\xi d\eta$. Since thickness $t$ is constant, we multiply it directly along with the Gauss weight $w$.

### Implementation of `qeq4e`

Computing the internal element forces, given a uniform distributed load $q$ on an edge, involves integrating the load against the element shape functions. This is known as calculating the equivalent nodal forces `qeq4e`.

```python
def qeq4e(ex, ey, t, edge, q):
    # edge can be 1 (nodes 1-2), 2 (2-3), 3 (3-4), or 4 (4-1)
    # We map the 1D edge to a parameter s in [-1, 1]
    gauss_pts = [-1/np.sqrt(3), 1/np.sqrt(3)]
    weights = [1.0, 1.0]
    
    f = np.zeros(8)
    
    # Map edges to node indices (0-based)
    edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
    n1, n2 = edges[edge - 1]
    
    # Calculate edge length and Jacobian of the edge (L/2)
    dx = ex[n2] - ex[n1]
    dy = ey[n2] - ey[n1]
    L = np.sqrt(dx**2 + dy**2)
    detJ_edge = L / 2.0
    
    # 1D Gauss quadrature along the edge
    for i in range(2):
        s = gauss_pts[i]
        w = weights[i]
        
        # 1D shape functions N1(s) = 0.5*(1-s), N2(s) = 0.5*(1+s)
        N_edge = np.array([0.5 * (1 - s), 0.5 * (1 + s)])
        
        # In a generic formulation, these map to the 2D shape functions
        # evaluated at the edge boundaries.
        # Add contributions to the corresponding nodal force components (x and y)
        f[2*n1]   += N_edge[0] * q[0] * detJ_edge * w * t
        f[2*n1+1] += N_edge[0] * q[1] * detJ_edge * w * t
        
        f[2*n2]   += N_edge[1] * q[0] * detJ_edge * w * t
        f[2*n2+1] += N_edge[1] * q[1] * detJ_edge * w * t
        
    return f
```
*Professor's Note:* The beauty of the consistent nodal load vector is that it honors the shape function interpolation. Notice how a uniform load vector $q = [q_x, q_y]^T$ is not simply divided by two; it is integrated rigorously along the boundary path length $L$, mapped via the 1D boundary Jacobian `detJ_edge = L / 2`. 

## 2.4 3D Elements (`t4`, `h8`)

The concepts extend naturally to three dimensions. The 8-node hexahedron (`h8`) utilizes trilinear shape functions $N_i(\xi, \eta, \zeta)$ and employs $2 \times 2 \times 2$ Gauss integration. The Jacobian becomes a $3 \times 3$ matrix, and the strain-displacement matrix $\mathbf{B}$ expands to compute the 6 components of the 3D strain tensor.
