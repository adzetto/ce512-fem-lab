# Chapter 8: Custom Element Development

The architecture of `femlabpy` makes it exceptionally straightforward to write and integrate your own finite elements. Unlike heavy, deeply-nested object-oriented frameworks, `femlabpy` requires exactly two mathematically pure functions per element type:

1. **Stiffness Routine (`ke...`)**: A function that returns the $n \times n$ elemental stiffness matrix.
2. **Force Routine (`qe...`)**: A function that returns the $n \times 1$ elemental internal force vector.

In this chapter, we will dive profoundly into the mechanics of element assembly. We will thoroughly explore how the `assmk` and `assmq` core functions operate, investigate the monumental performance implications of `is_sparse`, and walk step-by-step through the creation of a **Custom 2D Heat Transfer Element** and a **2D Beam Element**.

---

## 8.1 The Philosophy of Element Assembly

At the heart of the finite element method lies the assembly process—mapping local elemental contributions into the global system matrices. In `femlabpy`, this heavy lifting is performed by two highly optimized workhorses: `assmk` (Assemble Matrix K) and `assmq` (Assemble Vector Q).

### Deep Dive: `assmk` and `assmq`

The purpose of `assmk(K, Ke, nodes, dof)` is to take an $n_{el} \times n_{el}$ local stiffness matrix `Ke` and scatter-add it into the global $N_{total} \times N_{total}$ matrix `K`.

*   `K`: The global stiffness, mass, or damping matrix.
*   `Ke`: The elemental matrix (e.g., $3 \times 3$ for a 3-node scalar triangle, $6 \times 6$ for a 2-node beam).
*   `nodes`: A list or array of global node IDs connected to this element (1-based indexing in standard `femlabpy`).
*   `dof`: The number of degrees of freedom per node.

When you pass `dof`, `assmk` automatically calculates the correct block-strides in the global matrix. For example, if `dof=2` (solid mechanics), node `5` corresponds to global rows/cols `8` and `9` (since $(5-1) \times 2 = 8$). `assmk` seamlessly maps the local matrix indices to these global slots.

`assmq(Q, Qe, nodes, dof)` operates identically but for vectors, mapping an $n_{el} \times 1$ local force vector `Qe` into the global load vector `Q`.

### The Profound Impact of `is_sparse`

In computational mechanics, memory and performance are dominated by the sparsity of the global matrix. `femlabpy` detects the matrix format dynamically using the `is_sparse` utility.

```{warning}
A $10,000$ node 2D solid mechanics problem has $20,000$ DOFs. A dense float64 matrix of this size requires **3.2 Gigabytes** of RAM. A sparse representation requires barely **a few Megabytes**.
```

How does `is_sparse` affect the custom element assembly loop?

1. **Dense Assembly (`is_sparse(K) == False`)**: 
   If `K` is a standard NumPy `ndarray`, `assmk` executes a direct slicing and accumulation operation. It calculates the global indices and adds `Ke` using `K[ix, iy] += Ke`. This is exceedingly fast for small academic problems but scales at $O(N^2)$ in memory.
   
2. **Sparse Assembly (`is_sparse(K) == True`)**:
   If `K` is a `scipy.sparse` matrix (typically LIL or COO format during assembly), direct slicing is incredibly slow. When `assmk` detects a sparse matrix via `is_sparse`, it switches paradigms. It does not slice. Instead, it extracts the target row and column indices for the elemental block and appends the non-zero entries of `Ke` along with their `(row, col)` coordinates to internal lists (COO format generation).
   
By abstracting this inside `assmk`, your custom element driver loops remain perfectly clean, yet they scale seamlessly from 10 elements to 10,000,000 elements.

---

## 8.2 Writing a Custom 2D Heat Transfer Element (`ket3p`)

Let's design a 3-node triangular element for 2D steady-state heat conduction (a "Potential" problem). 

### The Mathematical Formulation

The governing equation (Poisson's equation) is:
$$ -\nabla \cdot (\mathbf{k} \nabla T) = Q $$

Where $T$ is temperature, $\mathbf{k}$ is the thermal conductivity matrix, and $Q$ is the heat generation. Using the Galerkin method, the weak form leads to the elemental stiffness matrix:

$$ \mathbf{K}_e = \int_{\Omega_e} \nabla \mathbf{N}^T \mathbf{k} \nabla \mathbf{N} \, dA $$

For a 3-node triangle, the shape functions $\mathbf{N} = [N_1, N_2, N_3]$ depend on the natural coordinates $(\xi, \eta)$. The gradient matrix $\mathbf{B}$ contains the spatial derivatives of the shape functions:

$$ \mathbf{B} = \begin{bmatrix} \frac{\partial N_1}{\partial x} & \frac{\partial N_2}{\partial x} & \frac{\partial N_3}{\partial x} \\ \frac{\partial N_1}{\partial y} & \frac{\partial N_2}{\partial y} & \frac{\partial N_3}{\partial y} \end{bmatrix} $$

### Gauss Integration: Theory to Python

To evaluate the integral over the element domain $\Omega_e$, we map the element to a reference master triangle and use numerical Gauss-Legendre quadrature. 

The mapping introduces the Jacobian matrix $\mathbf{J}$, which links the natural derivatives to the spatial derivatives. The integral transforms as:

$$ \int_{\Omega_e} \mathbf{B}^T \mathbf{k} \mathbf{B} \, dA = \sum_{i=1}^{n_{int}} w_i \mathbf{B}(\xi_i, \eta_i)^T \mathbf{k} \mathbf{B}(\xi_i, \eta_i) \det(\mathbf{J}) $$

For a linear triangle, the $\mathbf{B}$ matrix and Jacobian are constant, so a single integration point ($n_{int}=1$, $w_1 = 1/2$) is exact! The beautiful part of computational mechanics is how this complex continuous math reduces to a single, elegant line of linear algebra in Python:

```python
Ke += B.T @ D @ B * detJ * weight
```
*(where `D` is our conductivity matrix $\mathbf{k}$)*.

### Python Implementation of `ket3p`

Here is the exact implementation of the local element routine.

```python
import numpy as np

def ket3p(Xe, D, th):
    """
    Compute the 3x3 conductivity (stiffness) matrix for a 2D Heat Transfer Triangle.
    
    Parameters
    ----------
    Xe : ndarray
        3x2 array of node coordinates: [[x1, y1], [x2, y2], [x3, y3]]
    D : ndarray
        2x2 thermal conductivity matrix (k_xx, k_yy, etc.)
    th : float
        Thickness of the element
        
    Returns
    -------
    Ke : ndarray
        3x3 elemental stiffness matrix
    """
    # 1. Define Gauss integration point for linear triangle (1 point rule)
    r, s = 1.0/3.0, 1.0/3.0
    weight = 0.5  # Area of the reference triangle
    
    # 2. Derivatives of shape functions w.r.t natural coords (xi, eta)
    # N1 = 1 - r - s; N2 = r; N3 = s
    dN = np.array([
        [-1.0, 1.0, 0.0],
        [-1.0, 0.0, 1.0]
    ])
    
    # 3. Compute Jacobian
    J = dN @ Xe
    detJ = np.linalg.det(J)
    
    if detJ <= 0:
        raise ValueError("Element Jacobian is zero or negative. Check node numbering.")
        
    # 4. Map natural derivatives to spatial derivatives (B matrix)
    invJ = np.linalg.inv(J)
    B = invJ @ dN
    
    # 5. Gauss Integration
    # \int B^T * D * B * detJ * dA
    Ke = B.T @ D @ B * detJ * weight * th
    
    return Ke
```

### The Assembly Driver Loop

To assemble a mesh of 50,000 thermal triangles, we write a driver function `kt3p`. Note how heavily we rely on `assmk`. For a scalar potential problem, `dof = 1`.

```python
from femlabpy.assembly import assmk

def kt3p(K, T, X, G):
    """
    Global assembly driver for 2D Heat Transfer Triangles.
    """
    nel = T.shape[0]
    dof = 1  # 1 DOF per node (Temperature)
    
    # Loop over all elements in the mesh
    for e in range(nel):
        # 1-based node IDs from topology matrix
        n1, n2, n3 = int(T[e, 0]), int(T[e, 1]), int(T[e, 2])
        prop_id = int(T[e, 3]) - 1
        
        # Extract nodal coordinates
        Xe = np.vstack((X[n1-1], X[n2-1], X[n3-1]))
        
        # Extract material properties (Conductivity and thickness)
        kx = G[prop_id, 0]
        ky = G[prop_id, 1]
        D = np.array([[kx, 0], [0, ky]])
        th = G[prop_id, 2]
        
        # 1. Compute elemental matrix
        Ke = ket3p(Xe, D, th)
        
        # 2. Assemble into global matrix
        nodes = [n1, n2, n3]
        K = assmk(K, Ke, nodes, dof)
        
    return K
```

This design separates the finite element math (`ket3p`) from the graph-theory mesh topology (`kt3p`), yielding code that is both beautiful to read and fiercely performant.

---

## 8.3 Example: 2D Euler-Bernoulli Beam Element

For completeness, let us observe a structural element. A 2D beam element has two nodes, with 3 DOFs per node ($u_x, u_y, \theta_z$). 

The stiffness matrix relies on the standard beam equations and requires an angle transformation $\mathbf{R}^T \mathbf{K}_{loc} \mathbf{R}$.

```python
def kebeam2d(Xe, Ge):
    A, I, E = Ge[0], Ge[1], Ge[2]
    
    dx = Xe[1, 0] - Xe[0, 0]
    dy = Xe[1, 1] - Xe[0, 1]
    L = np.sqrt(dx**2 + dy**2)
    c, s = dx / L, dy / L
    
    k_axial = E * A / L
    k_shear = 12 * E * I / L**3
    k_mom1  = 6 * E * I / L**2
    k_mom2  = 4 * E * I / L
    k_mom3  = 2 * E * I / L
    
    K_loc = np.array([
        [ k_axial,  0,        0,       -k_axial,  0,        0      ],
        [ 0,        k_shear,  k_mom1,   0,       -k_shear,  k_mom1 ],
        [ 0,        k_mom1,   k_mom2,   0,       -k_mom1,   k_mom3 ],
        [-k_axial,  0,        0,        k_axial,  0,        0      ],
        [ 0,       -k_shear, -k_mom1,   0,        k_shear, -k_mom1 ],
        [ 0,        k_mom1,   k_mom3,   0,       -k_mom1,   k_mom2 ]
    ])
    
    R = np.array([
        [ c,  s,  0,  0,  0,  0],
        [-s,  c,  0,  0,  0,  0],
        [ 0,  0,  1,  0,  0,  0],
        [ 0,  0,  0,  c,  s,  0],
        [ 0,  0,  0, -s,  c,  0],
        [ 0,  0,  0,  0,  0,  1]
    ])
    
    return R.T @ K_loc @ R
```

The assembly driver `kbeam2d` operates exactly like `kt3p`, except we declare `dof = 3`. The core philosophy remains undisturbed: build local, assemble global.
