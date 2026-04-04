---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.1
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# 3D Solid Elements: Tetrahedrons and Hexahedrons

Welcome, scholars, to our advanced discourse on three-dimensional finite element formulations. In computational mechanics and finite element analysis (FEA), the modeling of general volumetric bodies—where foundational simplifications such as plane stress, plane strain, or axisymmetry strictly do not apply—mandates the deployment of fully three-dimensional solid elements.

In this chapter, we will embark on a rigorous mathematical journey, rigorously deriving the theoretical foundations and exploring the precise computational implementation of two indispensable elements: the 4-node linear tetrahedron (T4) and the 8-node trilinear hexahedron (H8). We shall thoroughly dissect their shape functions, the formulation of their respective Jacobian matrices, the intricate construction of the strain-displacement matrix ($\mathbf{B}$), the algorithmic assembly of the stiffness matrix ($\mathbf{K}^e$), and the recovery of internal forces and stresses. 

Let us begin our rigorous theoretical exposition.

## 1. The 4-Node Tetrahedron Element (T4)

The 4-node tetrahedron, widely designated in literature as the Constant Strain Tetrahedron (CST) generalized to 3D, stands as the most fundamental three-dimensional continuum element. It possesses four vertex nodes, with each node endowed with three translational degrees of freedom ($u, v, w$). Consequently, a single T4 element possesses a total of $4 \times 3 = 12$ degrees of freedom.

### 1.1 Volume Coordinates and Shape Functions

For geometries defined by simplices such as the tetrahedron, Cartesian coordinates are computationally unwieldy. Instead, it is vastly more elegant to invoke **volume coordinates** (a 3D extension of barycentric or area coordinates), denoted as $L_1, L_2, L_3, L_4$. 

Consider an arbitrary interior point $P$ within a tetrahedron of total volume $V$. This point elegantly partitions the master tetrahedron into four constituent sub-tetrahedra, having volumes $V_1, V_2, V_3, V_4$. Each sub-volume $V_i$ is subtended by the point $P$ and the triangular face strictly opposite to node $i$. We define the volume coordinates rigorously as:

$$
L_i = \frac{V_i}{V}, \quad i = 1, 2, 3, 4
$$

By fundamental geometric conservation, these dimensionless coordinates must sum exactly to unity:
$$
L_1 + L_2 + L_3 + L_4 = 1
$$

In the paradigm of isoparametric mapping, we elect three independent natural coordinates $\xi, \eta, \zeta$ to span the parametric domain. We map them directly to the volume coordinates:
$$
\xi = L_1, \quad \eta = L_2, \quad \zeta = L_3 \implies L_4 = 1 - \xi - \eta - \zeta
$$

The interpolation or **shape functions** $N_i$ for the T4 element are remarkably pristine—they are identically equal to the volume coordinates themselves. This enforces a strictly linear displacement field within the element interior:
$$
\begin{align*}
N_1(\xi, \eta, \zeta) &= \xi \\
N_2(\xi, \eta, \zeta) &= \eta \\
N_3(\xi, \eta, \zeta) &= \zeta \\
N_4(\xi, \eta, \zeta) &= 1 - \xi - \eta - \zeta
\end{align*}
$$

Because the shape functions are strictly linear, their local derivatives with respect to the natural coordinates yield a remarkably simple constant matrix:
$$
\frac{\partial \mathbf{N}}{\partial \boldsymbol{\xi}} =
\begin{bmatrix}
\frac{\partial N_1}{\partial \xi} & \frac{\partial N_2}{\partial \xi} & \frac{\partial N_3}{\partial \xi} & \frac{\partial N_4}{\partial \xi} \\
\frac{\partial N_1}{\partial \eta} & \frac{\partial N_2}{\partial \eta} & \frac{\partial N_3}{\partial \eta} & \frac{\partial N_4}{\partial \eta} \\
\frac{\partial N_1}{\partial \zeta} & \frac{\partial N_2}{\partial \zeta} & \frac{\partial N_3}{\partial \zeta} & \frac{\partial N_4}{\partial \zeta}
\end{bmatrix}
=
\begin{bmatrix}
1 & 0 & 0 & -1 \\
0 & 1 & 0 & -1 \\
0 & 0 & 1 & -1
\end{bmatrix}
$$

### 1.2 The Jacobian and Construction of the 12x12 Stiffness Matrix

The Jacobian matrix $\mathbf{J}$ acts as the indispensable tensor mapping the natural coordinate space to the physical Cartesian space $(x, y, z)$. It is formulated as:
$$
\mathbf{J} = \frac{\partial \mathbf{N}}{\partial \boldsymbol{\xi}} \mathbf{X}_e
$$
where $\mathbf{X}_e$ represents the $4 \times 3$ matrix of physical nodal coordinates. Because the local shape function derivatives are spatially invariant (constant), the resulting Jacobian $\mathbf{J}$ is stringently constant throughout the entirety of the element's volume.

By taking the inverse of this constant Jacobian matrix, we execute a push-forward operation, mapping the shape function derivatives from the parametric domain to the global physical domain:
$$
\begin{bmatrix} \frac{\partial N_i}{\partial x} \\ \frac{\partial N_i}{\partial y} \\ \frac{\partial N_i}{\partial z} \end{bmatrix} = \mathbf{J}^{-1} \begin{bmatrix} \frac{\partial N_i}{\partial \xi} \\ \frac{\partial N_i}{\partial \eta} \\ \frac{\partial N_i}{\partial \zeta} \end{bmatrix}
$$

**The Strain-Displacement Matrix ($\mathbf{B}$)**

The strain-displacement matrix $\mathbf{B}$ mathematically bridges the discrete nodal displacement vector $\mathbf{u}_e$ to the continuum strain tensor. We express strains in Voigt notation as a 6-component vector: $\boldsymbol{\epsilon} = [\epsilon_{xx}, \epsilon_{yy}, \epsilon_{zz}, \gamma_{xy}, \gamma_{yz}, \gamma_{zx}]^T$. 

For the T4 element, with 12 total DOFs, the $\mathbf{B}$ matrix rigorously assumes a $6 \times 12$ dimension. We construct it by concatenating four $6 \times 3$ submatrices, one dedicated to each node $i \in \{1, 2, 3, 4\}$:
$$
\mathbf{B} = \begin{bmatrix} \mathbf{B}_1 & \mathbf{B}_2 & \mathbf{B}_3 & \mathbf{B}_4 \end{bmatrix}
$$
The exquisite structure of each submatrix $\mathbf{B}_i$, dictated by the symmetric gradient of the displacement field, is defined precisely as:
$$
\mathbf{B}_i = \begin{bmatrix}
N_{i,x} & 0 & 0 \\
0 & N_{i,y} & 0 \\
0 & 0 & N_{i,z} \\
N_{i,y} & N_{i,x} & 0 \\
0 & N_{i,z} & N_{i,y} \\
N_{i,z} & 0 & N_{i,x}
\end{bmatrix}
$$

**Stiffness Matrix Derivation**

Invoking the principle of virtual work for linear elastostatics, the element stiffness matrix $\mathbf{K}^e$ (which will be a dense $12 \times 12$ symmetric positive-definite matrix for properly supported bodies) is governed by the volume integral:
$$
\mathbf{K}^e = \int_{V} \mathbf{B}^T \mathbf{D} \mathbf{B} \, dV
$$
where $\mathbf{D}$ denotes the $6 \times 6$ constitutive material matrix for isotropic, linear elasticity.

Because both the mapping derivatives $\mathbf{B}$ and the constitutive matrix $\mathbf{D}$ are strictly spatially invariant over the element domain, the integration process simplifies drastically. The integrand is pulled entirely outside the integral, leaving only the integration of the differential volume element:
$$
\mathbf{K}^e = \left( \mathbf{B}^T \mathbf{D} \mathbf{B} \right) \int_V dV = \left( \mathbf{B}^T \mathbf{D} \mathbf{B} \right) V_e
$$
Here, the volume $V_e$ is algorithmically computed via the determinant of the transformation Jacobian: $V_e = \frac{1}{6} \det(\mathbf{J})$.

---

## 2. The 8-Node Hexahedron Element (H8)

We now graduate to the 8-node hexahedron, colloquially termed the "brick" element. This is arguably the most dominant element in the computational mechanic's arsenal due to its superior numerical accuracy and robustness against volumetric locking when compared to linear tetrahedra. Each of its 8 nodes possesses 3 translational DOFs, culminating in a computationally significant 24 degrees of freedom per element.

### 2.1 Trilinear Shape Functions

Unlike the simplex coordinates of the T4, the H8 element relies on a tensor-product formulation mapped from a bi-unit reference cube. The parent parametric coordinates reside strictly within the domain $\xi, \eta, \zeta \in [-1, 1]$. The eight master nodes are localized precisely at the logical extremities of this cube: $(\pm 1, \pm 1, \pm 1)$.

The canonical shape functions $N_i$ for an arbitrary node $i$ (whose parametric coordinates are denoted as $\xi_i, \eta_i, \zeta_i$) are elegantly synthesized through the product of one-dimensional linear Lagrange polynomials:
$$
N_i(\xi, \eta, \zeta) = \frac{1}{8} (1 + \xi_i \xi)(1 + \eta_i \eta)(1 + \zeta_i \zeta) \quad \text{for } i = 1, \dots, 8
$$

These functions are strictly $C^0$ continuous. While they guarantee intra-element displacement continuity, note well that strain continuity across element interfaces is absolutely not enforced—a hallmark of standard displacement-based finite elements.

### 2.2 Jacobian, Kinematics, and the 6x24 B-Matrix

Because the trilinear shape functions contain coupled quadratic and cubic parametric terms (e.g., $\xi\eta$, $\xi\eta\zeta$), their spatial derivatives are definitively *not* constant. They fluctuate dynamically throughout the element volume. For instance, computing the analytic derivative of $N_1$ (positioned at $\xi_1=-1, \eta_1=-1, \zeta_1=-1$) with respect to $\xi$ yields:
$$
\frac{\partial N_1}{\partial \xi} = \frac{1}{8}(\xi_1)(1 + \eta_1 \eta)(1 + \zeta_1 \zeta) = -\frac{1}{8} (1 - \eta)(1 - \zeta)
$$

The corresponding $3 \times 3$ Jacobian matrix $\mathbf{J}$ becomes a strictly local operator, dependent on the instantaneous parametric coordinates $\xi, \eta, \zeta$:
$$
\mathbf{J}(\xi, \eta, \zeta) = \sum_{i=1}^8 \begin{bmatrix} \frac{\partial N_i}{\partial \xi} \\ \frac{\partial N_i}{\partial \eta} \\ \frac{\partial N_i}{\partial \zeta} \end{bmatrix} \begin{bmatrix} x_i & y_i & z_i \end{bmatrix}
$$

**Constructing the $6 \times 24$ $\mathbf{B}$ Matrix**

To form the global strain-displacement matrix $\mathbf{B}$, we must evaluate it pointwise. The $\mathbf{B}$ matrix dictates the mapping from the 24 elemental degrees of freedom to the 6 physical strain components. It assumes a $6 \times 24$ topology, comprised of eight $6 \times 3$ nodal submatrices:
$$
\mathbf{B}(\xi, \eta, \zeta) = \begin{bmatrix} \mathbf{B}_1 & \mathbf{B}_2 & \dots & \mathbf{B}_8 \end{bmatrix}
$$
To construct each $\mathbf{B}_i$, we first map the parent shape function derivatives into the physical Cartesian frame precisely by operating on them with the inverse of the local Jacobian:
$$
\begin{bmatrix} N_{i,x} \\ N_{i,y} \\ N_{i,z} \end{bmatrix} = \mathbf{J}^{-1}(\xi, \eta, \zeta) \begin{bmatrix} N_{i,\xi} \\ N_{i,\eta} \\ N_{i,\zeta} \end{bmatrix}
$$
Once the global spatial derivatives are isolated, the matrix $\mathbf{B}_i$ is flawlessly populated matching the tensorial mechanics of the displacement gradients:
$$
\mathbf{B}_i = \begin{bmatrix}
N_{i,x} & 0 & 0 \\
0 & N_{i,y} & 0 \\
0 & 0 & N_{i,z} \\
N_{i,y} & N_{i,x} & 0 \\
0 & N_{i,z} & N_{i,y} \\
N_{i,z} & 0 & N_{i,x}
\end{bmatrix}
$$

### 2.3 Gauss Quadrature and Numerical Integration

The computation of the $24 \times 24$ element stiffness matrix necessitates numerical integration, as closed-form analytical integration of the rational polynomial expressions stemming from $\mathbf{J}^{-1}$ is mathematically intractable for distorted geometries:
$$
\mathbf{K}^e = \int_{-1}^1 \int_{-1}^1 \int_{-1}^1 \mathbf{B}^T(\xi, \eta, \zeta) \mathbf{D} \mathbf{B}(\xi, \eta, \zeta) \det(\mathbf{J}(\xi, \eta, \zeta)) \, d\xi \, d\eta \, d\zeta
$$

We appeal rigorously to Gauss-Legendre quadrature. For the H8 element, standard exactness criteria command the use of a $2 \times 2 \times 2$ integration scheme, which exactly integrates polynomials up to degree 3. 

The eight optimal Gauss points are situated at $\pm 1/\sqrt{3}$ along each local axis:
$$
r_k \in \left\{-\frac{1}{\sqrt{3}}, \frac{1}{\sqrt{3}}\right\}
$$
Each integration point commands an integration weight of $W = 1 \times 1 \times 1 = 1$. The integral is converted into a finite summation over the 8 Gauss points (indexed by $g$):
$$
\mathbf{K}^e = \sum_{g=1}^8 \mathbf{B}_g^T \mathbf{D} \mathbf{B}_g \det(\mathbf{J}_g) W_g = \sum_{g=1}^8 \mathbf{B}_g^T \mathbf{D} \mathbf{B}_g \det(\mathbf{J}_g)
$$

---

## 3. Implementation in Python: The Professor's Runnable Script

Below, I present a rigorous, completely self-contained, and highly optimized script. It synthesizes all the intricate tensorial operations and numerical integration schemes we've discussed into functional Python code using `numpy`. This script actively builds both the T4 ($12 \times 12$) and H8 ($24 \times 24$) local stiffness matrices. 

```python
import numpy as np

def _elastic3d_matrix(E, nu):
    """Constitutive matrix D for 3D isotropic linear elasticity."""
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu = E / (2 * (1 + nu))
    D = np.zeros((6, 6))
    D[0:3, 0:3] = lam
    np.fill_diagonal(D[0:3, 0:3], lam + 2 * mu)
    np.fill_diagonal(D[3:6, 3:6], mu)
    return D

def _solid_B(dN_global):
    """Assembles the B matrix from global shape function derivatives."""
    num_nodes = dN_global.shape[1]
    B = np.zeros((6, num_nodes * 3))
    for i in range(num_nodes):
        nx, ny, nz = dN_global[:, i]
        col = i * 3
        B[0, col]   = nx
        B[1, col+1] = ny
        B[2, col+2] = nz
        B[3, col:col+2] = [ny, nx]
        B[4, col+1:col+3] = [nz, ny]
        B[5, col::2] = [nz, nx]
    return B

def keT4e(Xe, E, nu):
    """Computes the 12x12 Stiffness Matrix for a T4 element."""
    # Master gradients
    dN = np.array([
        [1.0, 0.0, 0.0, -1.0], 
        [0.0, 1.0, 0.0, -1.0], 
        [0.0, 0.0, 1.0, -1.0]
    ], dtype=float)
    
    J = dN @ Xe
    detJ = np.linalg.det(J)
    dN_global = np.linalg.solve(J, dN)
    
    B = _solid_B(dN_global)
    D = _elastic3d_matrix(E, nu)
    
    # K = B^T * D * B * Volume
    # Volume of tetrahedron = (1/6) * det(J). However, natural coord integration domain 
    # for T4 yields an intrinsic 1/6 mapping factor when doing analytical int. 
    # True relation: Volume = 1/6 * det(J). 
    volume = np.abs(detJ) / 6.0
    
    K = B.T @ D @ B * volume
    return K

def keh8e(Xe, E, nu):
    """Computes the 24x24 Stiffness Matrix for an H8 element."""
    D = _elastic3d_matrix(E, nu)
    K = np.zeros((24, 24))
    
    # 2x2x2 Gauss Quadrature Points and Weights
    gpt = 1.0 / np.sqrt(3.0)
    gauss_points = np.array([
        [-gpt, -gpt, -gpt], [gpt, -gpt, -gpt], [gpt, gpt, -gpt], [-gpt, gpt, -gpt],
        [-gpt, -gpt, gpt],  [gpt, -gpt, gpt],  [gpt, gpt, gpt],  [-gpt, gpt, gpt]
    ])
    
    # Nodal parametric coordinates for shape function calculation
    nodes_xi = np.array([-1, 1, 1, -1, -1, 1, 1, -1])
    nodes_eta = np.array([-1, -1, 1, 1, -1, -1, 1, 1])
    nodes_zeta = np.array([-1, -1, -1, -1, 1, 1, 1, 1])

    # 2x2x2 Gauss Loop
    for gp in gauss_points:
        xi, eta, zeta = gp
        
        # Local shape function derivatives (3x8 matrix)
        dN_dxi = np.zeros((3, 8))
        for i in range(8):
            dN_dxi[0, i] = 0.125 * nodes_xi[i] * (1 + nodes_eta[i]*eta) * (1 + nodes_zeta[i]*zeta)
            dN_dxi[1, i] = 0.125 * nodes_eta[i] * (1 + nodes_xi[i]*xi) * (1 + nodes_zeta[i]*zeta)
            dN_dxi[2, i] = 0.125 * nodes_zeta[i] * (1 + nodes_xi[i]*xi) * (1 + nodes_eta[i]*eta)
            
        # Compute Jacobian, its determinant, and global derivatives
        J = dN_dxi @ Xe
        detJ = np.linalg.det(J)
        dN_global = np.linalg.solve(J, dN_dxi)
        
        # Build 6x24 B matrix
        B = _solid_B(dN_global)
        
        # Add to element stiffness matrix (Weight = 1.0 for each point)
        K += B.T @ D @ B * np.abs(detJ)
        
    return K

if __name__ == "__main__":
    print("Executing Professor's 3D Element Validation Script...\n")
    
    # Material properties (Steel)
    E = 200e9   # Pa
    nu = 0.3    # Poisson's ratio
    
    # --- T4 Element Assembly ---
    # Nodal coordinates for a simple right tetrahedron
    Xe_T4 = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]
    ])
    K_T4 = keT4e(Xe_T4, E, nu)
    print(f"T4 Stiffness Matrix Assembled. Shape: {K_T4.shape}")
    print(f"Norm of K_T4: {np.linalg.norm(K_T4):.2e}\n")
    
    # --- H8 Element Assembly ---
    # Nodal coordinates for a unit bi-unit cube mapped to physical (2x2x2 physical cube)
    Xe_H8 = np.array([
        [0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [2.0, 2.0, 0.0], [0.0, 2.0, 0.0],
        [0.0, 0.0, 2.0], [2.0, 0.0, 2.0], [2.0, 2.0, 2.0], [0.0, 2.0, 2.0]
    ])
    K_H8 = keh8e(Xe_H8, E, nu)
    print(f"H8 Stiffness Matrix Assembled. Shape: {K_H8.shape}")
    print(f"Norm of K_H8: {np.linalg.norm(K_H8):.2e}")
    print("\nValidation complete. Class dismissed!")
```

## Summary

In modern 3D continuum structural mechanics, the tetrahedral elements (T4) proffer exceptional, robust meshing flexibility—vital for aggressively complex CAD topologies where topological mapping functions fail. However, their mathematically rigid constant strain formulation subjects them perilously to artificial volumetric stiffening (the notorious "shear locking") and demands extraordinarily fine grid discretizations to adequately resolve flexural phenomena.

The Hexahedral elements (H8), fortified by their higher-order trilinear interpolation manifolds, effortlessly manifest superior kinematic accuracy alongside drastically minimized degree-of-freedom allocations. The foremost algorithmic tribulation with H8 deployment remains the geometrical meshing complexity—it is formidably difficult to automatically discretize completely arbitrary 3D domains into purely hexahedral primitives without topological degeneracy. Ultimately, a prudent scholar of mechanics utilizes a judicious superposition of both topologies to master computational resilience and extreme precision.