---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.1
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# 2D Triangular Elements (CST)

This chapter covers the mathematical foundations, finite element formulation, and Python implementation of the 3-node triangular element. Often referred to as the Constant Strain Triangle (CST) in structural mechanics, it is one of the earliest and simplest finite elements developed. Despite its simplicity, it is robust and forms the foundational stepping stone for more complex isoparametric element derivations.

In this text, we will focus on both the theoretical derivation using area coordinates and the practical, highly optimized vectorized implementation available in `femlabpy`.

## Area Coordinates and Shape Functions

In a 1D line element, we use length-based natural coordinates. In a 2D triangular element, it is most natural to use **area coordinates** (also known as barycentric coordinates). 

Consider a triangle with vertices $1(x_1, y_1)$, $2(x_2, y_2)$, and $3(x_3, y_3)$. Its total area $A$ can be computed using the determinant of the Jacobian:

$$
2A = \det \begin{bmatrix} 1 & x_1 & y_1 \\ 1 & x_2 & y_2 \\ 1 & x_3 & y_3 \end{bmatrix} = (x_2 y_3 - x_3 y_2) + (x_3 y_1 - x_1 y_3) + (x_1 y_2 - x_2 y_1)
$$

For any interior point $P(x, y)$, we can form three sub-triangles by connecting $P$ to the vertices. Let the areas of these sub-triangles be $A_1$, $A_2$, and $A_3$ (where $A_i$ is the area of the sub-triangle opposite to vertex $i$). The area coordinates $L_i$ are defined as the ratio of these sub-areas to the total area:

$$
L_1 = \frac{A_1}{A}, \quad L_2 = \frac{A_2}{A}, \quad L_3 = \frac{A_3}{A}
$$

By definition, these coordinates must satisfy the partition of unity:

$$
L_1 + L_2 + L_3 = 1
$$

In a 3-node triangular element, the shape functions $N_i(x,y)$ are exactly the area coordinates. Thus:

$$
N_1 = L_1, \quad N_2 = L_2, \quad N_3 = L_3
$$

### Mapping Area Coordinates to $(x,y)$

The relationship between Cartesian coordinates $(x,y)$ and area coordinates $(L_1, L_2, L_3)$ is an exact linear mapping:

$$
\begin{bmatrix} 1 \\ x \\ y \end{bmatrix} = \begin{bmatrix} 1 & 1 & 1 \\ x_1 & x_2 & x_3 \\ y_1 & y_2 & y_3 \end{bmatrix} \begin{bmatrix} L_1 \\ L_2 \\ L_3 \end{bmatrix}
$$

Inverting this 3x3 matrix relationship gives the explicit polynomial form for the shape functions:

$$
\begin{bmatrix} L_1 \\ L_2 \\ L_3 \end{bmatrix} = \frac{1}{2A} \begin{bmatrix} (x_2 y_3 - x_3 y_2) & (y_2 - y_3) & (x_3 - x_2) \\ (x_3 y_1 - x_1 y_3) & (y_3 - y_1) & (x_1 - x_3) \\ (x_1 y_2 - x_2 y_1) & (y_1 - y_2) & (x_2 - x_1) \end{bmatrix} \begin{bmatrix} 1 \\ x \\ y \end{bmatrix}
$$

Or, written explicitly for $N_i$:

$$
N_i(x, y) = L_i(x, y) = \frac{1}{2A} (a_i + b_i x + c_i y)
$$

where the geometric constants are derived from the cyclic permutations of the node indices (e.g., for $i=1$, $j=2$, $k=3$):

$$
a_i = x_j y_k - x_k y_j, \quad b_i = y_j - y_k, \quad c_i = x_k - x_j
$$

The spatial derivatives (gradients) of the shape functions with respect to $x$ and $y$ are thus simple constants evaluated over the element domain:

$$
\frac{\partial N_i}{\partial x} = \frac{b_i}{2A}, \quad \frac{\partial N_i}{\partial y} = \frac{c_i}{2A}
$$

This mathematical property is what gives the Constant Strain Triangle its name: because the displacement gradients are derived from the constant shape function derivatives, the entire strain and stress field within each element is uniform.

## Solid Mechanics: Deriving the $3 \times 6$ $B$ Matrix Step-by-Step

In 2D plane elasticity (either plane stress or plane strain), each node has two degrees of freedom (DOFs): translation in $x$ (denoted $u$) and $y$ (denoted $v$). For a 3-node element, the $6 \times 1$ element displacement vector is:

$$
\mathbf{u}^e = \begin{bmatrix} u_1 \\ v_1 \\ u_2 \\ v_2 \\ u_3 \\ v_3 \end{bmatrix}
$$

The continuous displacement field within the element is interpolated using our area coordinate shape functions:

$$
u(x,y) = N_1 u_1 + N_2 u_2 + N_3 u_3 \\
v(x,y) = N_1 v_1 + N_2 v_2 + N_3 v_3
$$

In a 2D elasticity problem, the strain vector $\boldsymbol{\varepsilon}$ has three components: normal strain in x, normal strain in y, and shear strain:

$$
\boldsymbol{\varepsilon} = \begin{bmatrix} \varepsilon_{xx} \\ \varepsilon_{yy} \\ \gamma_{xy} \end{bmatrix} = \begin{bmatrix} \frac{\partial u}{\partial x} \\ \frac{\partial v}{\partial y} \\ \frac{\partial u}{\partial y} + \frac{\partial v}{\partial x} \end{bmatrix}
$$

By substituting the interpolated displacements into the strain definitions, we get:

$$
\frac{\partial u}{\partial x} = \frac{\partial N_1}{\partial x} u_1 + \frac{\partial N_2}{\partial x} u_2 + \frac{\partial N_3}{\partial x} u_3
$$
$$
\frac{\partial v}{\partial y} = \frac{\partial N_1}{\partial y} v_1 + \frac{\partial N_2}{\partial y} v_2 + \frac{\partial N_3}{\partial y} v_3
$$
$$
\frac{\partial u}{\partial y} + \frac{\partial v}{\partial x} = \frac{\partial N_1}{\partial y} u_1 + \frac{\partial N_1}{\partial x} v_1 + \frac{\partial N_2}{\partial y} u_2 + \frac{\partial N_2}{\partial x} v_2 + \frac{\partial N_3}{\partial y} u_3 + \frac{\partial N_3}{\partial x} v_3
$$

We can express this linear relationship compactly using the Strain-Displacement matrix $\mathbf{B}$:

$$
\boldsymbol{\varepsilon} = \mathbf{B} \mathbf{u}^e
$$

where $\mathbf{B}$ is a $3 \times 6$ matrix composed of the shape function derivatives. Constructing it column-by-column for each DOF:

$$
\mathbf{B} = \begin{bmatrix} 
\frac{\partial N_1}{\partial x} & 0 & \frac{\partial N_2}{\partial x} & 0 & \frac{\partial N_3}{\partial x} & 0 \\
0 & \frac{\partial N_1}{\partial y} & 0 & \frac{\partial N_2}{\partial y} & 0 & \frac{\partial N_3}{\partial y} \\
\frac{\partial N_1}{\partial y} & \frac{\partial N_1}{\partial x} & \frac{\partial N_2}{\partial y} & \frac{\partial N_2}{\partial x} & \frac{\partial N_3}{\partial y} & \frac{\partial N_3}{\partial x}
\end{bmatrix}
$$

Finally, substituting the known constant derivatives $\frac{\partial N_i}{\partial x} = \frac{b_i}{2A}$ and $\frac{\partial N_i}{\partial y} = \frac{c_i}{2A}$, we arrive at the exact analytical $3 \times 6$ $\mathbf{B}$ matrix for the CST element:

$$
\mathbf{B} = \frac{1}{2A} \begin{bmatrix} 
b_1 & 0 & b_2 & 0 & b_3 & 0 \\
0 & c_1 & 0 & c_2 & 0 & c_3 \\
c_1 & b_1 & c_2 & b_2 & c_3 & b_3
\end{bmatrix}
$$

## Element Stiffness Matrix Computation in `ket3e`

The element stiffness matrix is derived from the principle of virtual work, integrating the strain energy density over the element volume:

$$
\mathbf{k}^e = \int_{\Omega_e} \mathbf{B}^T \mathbf{D} \mathbf{B} t \, dA
$$

where $\mathbf{D}$ is the $3 \times 3$ constitutive (elasticity) matrix, and $t$ is the element thickness. Because $\mathbf{B}$ and $\mathbf{D}$ are entirely constant over the triangle, the integrand is invariant, and the integral simply reduces to multiplication by the triangle's area $A$:

$$
\mathbf{k}^e = \mathbf{B}^T \mathbf{D} \mathbf{B} A t
$$

### Exact Python Code in `ket3e`

In `femlabpy`, this routine is handled inside the `ket3e` function. Here is the exact Python code slice showing the calculation of the shape function derivatives (`b` and `c`), the construction of `B`, and the final matrix multiplication evaluating $\mathbf{B}^T \mathbf{D} \mathbf{B} A t$:

```python
# Compute edge coordinate differences
x = Xe[:, 0]
y = Xe[:, 1]
b = np.array([y[1]-y[2], y[2]-y[0], y[0]-y[1]])
c = np.array([x[2]-x[1], x[0]-x[2], x[1]-x[0]])

# Compute element area
A = 0.5 * (b[0]*c[1] - b[1]*c[0])

# Construct the 3x6 B matrix
B = np.zeros((3, 6))
B[0, 0::2] = b
B[1, 1::2] = c
B[2, 0::2] = c
B[2, 1::2] = b
B = B / (2 * A)

# D is the 3x3 elasticity matrix
# t is the thickness

# Compute the element stiffness matrix
Ke = (B.T @ D @ B) * A * t
```

## Vectorized Assembly (`kt3e` and `np.einsum`)

While `ket3e` computes the stiffness for a single element, engineering models frequently contain 10,000 to 1,000,000+ elements. Using a Python `for` loop over 10,000 elements to run `B.T @ D @ B * A * t` would cause severe performance bottlenecks due to standard interpreter overhead.

`femlabpy` achieves near C-level performance by fully vectorizing the assembly across the entire element domain. Instead of a single $3 \times 6$ matrix, `femlabpy` computes an $(N, 3, 6)$ tensor for the $\mathbf{B}$ matrices of $N$ elements, and an $(N, 3, 3)$ tensor for $\mathbf{D}$.

### Explaining `np.einsum`

For $10,000$ elements, how does one perform 10,000 independent matrix multiplications of the form $K_e = B_e^T D_e B_e A_e t_e$ simultaneously? 

`femlabpy` leverages **Einstein Summation Convention** via `numpy.einsum`. The explicit matrix product $\mathbf{K} = \mathbf{B}^T \mathbf{D} \mathbf{B}$ for a single element can be written in index notation as:

$$
K_{ij} = \sum_k \sum_l (B^T)_{ik} D_{kl} B_{lj} = \sum_k \sum_l B_{ki} D_{kl} B_{lj}
$$

When vectorized across $e$ elements (where $e \in [0, N-1]$), we introduce the element index `e`:

$$
K_{eij} = \sum_k \sum_l B_{eki} D_{ekl} B_{elj}
$$

In `femlabpy`, this translates directly to the `np.einsum` string `"eki,ekl,elj->eij"`:

```python
# Batched variables:
# B is shape (N, 3, 6)
# D is shape (N, 3, 3)
# area is shape (N,)
# t is thickness, shape (N,) or scalar

# Evaluate B^T * D * B for all N elements simultaneously
K_batched = np.einsum("eki,ekl,elj->eij", B, D, B)

# Multiply by A and t, broadcasting over the N dimension
element_matrices = K_batched * area[:, None, None] * t
```

* `B` is passed as the first argument, mapped to `eki` (element $e$, row $k$, column $i$). Note: This effectively handles the transposition $\mathbf{B}^T$ intrinsically because the output maps $i$ (the columns of $\mathbf{B}$) to the rows of $\mathbf{K}$.
* `D` is the second argument, mapped to `ekl`.
* `B` is the third argument, mapped to `elj`.
* `->eij` tells NumPy to sum over the dummy indices $k$ and $l$, outputting a tensor of shape `(N, 6, 6)`.

This single statement forces the computation down to highly-optimized BLAS/C routines under the hood, instantly evaluating all 10,000 elements without a single Python `for` loop.

## Runnable Example Script

The following script demonstrates the theoretical concept by constructing 10,000 CST elements and vectorizing their stiffness matrix calculations.

```python
import numpy as np
import time

def vectorized_cst_stiffness(N_elements=10000):
    # 1. Generate dummy nodal coordinates for N elements
    # Xe shape: (N, 3 nodes, 2 coords)
    np.random.seed(42)
    Xe = np.random.rand(N_elements, 3, 2)
    
    # 2. Material properties
    E = 200e9
    nu = 0.3
    t = 0.01  # thickness
    
    # Plane stress D matrix (same for all elements here, shape 3x3)
    factor = E / (1 - nu**2)
    D = factor * np.array([
        [1, nu, 0],
        [nu, 1, 0],
        [0, 0, (1 - nu)/2]
    ])
    # Broadcast D to shape (N, 3, 3)
    D_batched = np.tile(D, (N_elements, 1, 1))
    
    start_time = time.time()
    
    # 3. Compute geometric differences b and c for all N elements
    x = Xe[:, :, 0]
    y = Xe[:, :, 1]
    
    b1 = y[:, 1] - y[:, 2]
    b2 = y[:, 2] - y[:, 0]
    b3 = y[:, 0] - y[:, 1]
    
    c1 = x[:, 2] - x[:, 1]
    c2 = x[:, 0] - x[:, 2]
    c3 = x[:, 1] - x[:, 0]
    
    # Areas (shape: N,)
    A = 0.5 * (b1*c2 - b2*c1)
    
    # 4. Construct batched B matrix (shape: N, 3, 6)
    B = np.zeros((N_elements, 3, 6))
    B[:, 0, 0] = b1; B[:, 0, 2] = b2; B[:, 0, 4] = b3
    B[:, 1, 1] = c1; B[:, 1, 3] = c2; B[:, 1, 5] = c3
    B[:, 2, 0] = c1; B[:, 2, 1] = b1
    B[:, 2, 2] = c2; B[:, 2, 3] = b2
    B[:, 2, 4] = c3; B[:, 2, 5] = b3
    
    # Divide B by 2A (broadcasting A)
    B = B / (2 * A[:, None, None])
    
    # 5. np.einsum Vectorization: compute Ke = B^T D B A t
    # B is (N, 3, 6) -> e k i
    # D is (N, 3, 3) -> e k l
    # B is (N, 3, 6) -> e l j
    # Result -> e i j (shape N, 6, 6)
    K_batched = np.einsum("eki,ekl,elj->eij", B, D_batched, B)
    Ke_all = K_batched * A[:, None, None] * t
    
    end_time = time.time()
    
    print(f"Computed {N_elements} CST element stiffness matrices in {end_time - start_time:.4f} seconds.")
    print(f"Shape of resulting Ke tensor: {Ke_all.shape}")
    
    return Ke_all

if __name__ == "__main__":
    Ke_all = vectorized_cst_stiffness(10000)
    # Check first element matrix trace as a sanity check
    print(f"Trace of Ke for Element 0: {np.trace(Ke_all[0]):.2e}")
```

This effectively bypasses interpreter limits, allowing Python to rival compiled languages in FEA matrix assembly loops.

## Potential Flow and Heat Transfer

Beyond solid mechanics, the T3 element is widely used for scalar field problems governed by the Poisson or Laplace equation, such as steady-state heat conduction, groundwater flow, or ideal fluid potential flow.

In these problems, each node has a single scalar degree of freedom (e.g., temperature $T$ or hydraulic head $\phi$). The primary unknown is a scalar field, and the flux $\mathbf{q}$ is proportional to the gradient of this field:

$$
\mathbf{q} = -k \nabla T
$$

where $k$ is the conductivity (or permeability).

The element conductivity matrix $\mathbf{k}_p^e$ is derived from the weak form:

$$
\mathbf{k}_p^e = \int_{\Omega_e} (\nabla \mathbf{N})^T k (\nabla \mathbf{N}) \, dA
$$

For the 3-node triangle, the gradient operator matrix $\mathbf{B}_p$ is a $2 \times 3$ matrix:

$$
\mathbf{B}_p = \nabla \mathbf{N} = \frac{1}{2A} \begin{bmatrix} b_1 & b_2 & b_3 \\ c_1 & c_2 & c_3 \end{bmatrix}
$$

Evaluating the integral (assuming constant $k$ and isotropic material $\mathbf{D} = k\mathbf{I}$), we get:

$$
\mathbf{k}_p^e = \mathbf{B}_p^T (k \mathbf{I}) \mathbf{B}_p A
$$

Vectorization of this process follows the exact same architectural philosophy via `np.einsum` applied to the thermal tensors.

## Summary

The Constant Strain Triangle is structurally elementary but provides the foundation for 2D finite element analysis. By exploiting exact integration with area coordinates, constant field gradients, and extreme vectorization of tensor operations via `einsum`, we achieve high-performance simulation standards suitable for both academic study and rigorous computational modeling.