---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Chapter 5: 2D Isoparametric Quadrilateral Elements

The finite element method (FEM) heavily relies on the concept of isoparametric elements. These elements use the same shape functions to interpolate both the geometric coordinates and the field variables (such as displacements or temperatures) within the element. This powerful concept allows elements to take on distorted, non-rectangular shapes in the global Cartesian coordinate system while being mapped from a simple, standard square shape in a local, "parent" coordinate system.

In this comprehensive chapter, we delve into the theory, mathematics, and computational implementation of the 4-node bilinear isoparametric quadrilateral element (Q4 element). We will explore the formulation of its shape functions, the Jacobian matrix used for coordinate transformation, the strain-displacement matrix, and the application of Gauss-Legendre numerical integration. Finally, we will dissect its software implementation within `femlabpy`, covering both solid mechanics (elasticity) and scalar potential flow (e.g., heat conduction, seepage).

## 1. Bilinear Shape Functions

The Q4 element has four nodes, typically numbered counter-clockwise starting from the bottom-left corner. To facilitate integration and generic element formulation, we define a "parent" element in a local coordinate system with axes denoted by $\xi$ and $\eta$. The parent element is a square with its domain defined by $\xi \in [-1, 1]$ and $\eta \in [-1, 1]$. 

The nodes of the parent element are located at:
* Node 1: $(-1, -1)$
* Node 2: $(1, -1)$
* Node 3: $(1, 1)$
* Node 4: $(-1, 1)$

For an interpolation function to be valid, it must take the value of $1$ at its corresponding node and $0$ at all other nodes. For a 4-node element, the simplest polynomials that satisfy these conditions and represent a complete bi-linear field are the bilinear shape functions. These are constructed by taking the tensor product of 1D linear Lagrange polynomials.

The general expression for the shape function $N_i$ associated with node $i$ (having coordinates $\xi_i, \eta_i$) is given by:

$$
N_i(\xi, \eta) = \frac{1}{4} (1 + \xi_i \xi) (1 + \eta_i \eta) \quad \text{for } i = 1, 2, 3, 4
$$

Expanding this for each of the four nodes, we obtain the individual shape functions:

$$
\begin{aligned}
N_1(\xi, \eta) &= \frac{1}{4} (1 - \xi) (1 - \eta) \\
N_2(\xi, \eta) &= \frac{1}{4} (1 + \xi) (1 - \eta) \\
N_3(\xi, \eta) &= \frac{1}{4} (1 + \xi) (1 + \eta) \\
N_4(\xi, \eta) &= \frac{1}{4} (1 - \xi) (1 + \eta)
\end{aligned}
$$

These shape functions possess several important properties:
1.  **Kronecker delta property:** $N_i(\xi_j, \eta_j) = \delta_{ij}$.
2.  **Partition of unity:** $\sum_{i=1}^4 N_i(\xi, \eta) = 1$ for any point $(\xi, \eta)$ within the element. This ensures that a rigid body translation is exactly represented.
3.  **Linear representation:** They can exactly represent a linear field $C_1 + C_2 x + C_3 y$.

## 2. Isoparametric Mapping and the Jacobian Matrix

The word "isoparametric" stems from "iso" (same) and "parametric" (parameters). In this formulation, the global Cartesian coordinates $(x, y)$ of any point within the element are interpolated from the nodal coordinates $(x_i, y_i)$ using the identical shape functions $N_i$ that are used for the primary unknown fields:

$$
x(\xi, \eta) = \sum_{i=1}^4 N_i(\xi, \eta) x_i, \quad y(\xi, \eta) = \sum_{i=1}^4 N_i(\xi, \eta) y_i
$$

Similarly, the primary field variable—let's consider the displacement vector $\mathbf{u} = [u, v]^T$ for solid mechanics—is interpolated as:

$$
u(\xi, \eta) = \sum_{i=1}^4 N_i(\xi, \eta) u_i, \quad v(\xi, \eta) = \sum_{i=1}^4 N_i(\xi, \eta) v_i
$$

To compute physical quantities like strains or heat fluxes, we need the derivatives of the shape functions with respect to the global physical coordinates $x$ and $y$. However, our shape functions $N_i(\xi, \eta)$ are defined entirely in terms of the local parent coordinates $\xi$ and $\eta$. To bridge this gap, we must employ the multivariable chain rule of differentiation:

$$
\begin{aligned}
\frac{\partial N_i}{\partial \xi} &= \frac{\partial N_i}{\partial x} \frac{\partial x}{\partial \xi} + \frac{\partial N_i}{\partial y} \frac{\partial y}{\partial \xi} \\
\frac{\partial N_i}{\partial \eta} &= \frac{\partial N_i}{\partial x} \frac{\partial x}{\partial \eta} + \frac{\partial N_i}{\partial y} \frac{\partial y}{\partial \eta}
\end{aligned}
$$

This relationship linearly maps the derivatives in the physical domain to the derivatives in the parent domain. We can express this elegantly in matrix form:

$$
\begin{Bmatrix}
\frac{\partial N_i}{\partial \xi} \\
\frac{\partial N_i}{\partial \eta}
\end{Bmatrix}
=
\begin{bmatrix}
\frac{\partial x}{\partial \xi} & \frac{\partial y}{\partial \xi} \\
\frac{\partial x}{\partial \eta} & \frac{\partial y}{\partial \eta}
\end{bmatrix}
\begin{Bmatrix}
\frac{\partial N_i}{\partial x} \\
\frac{\partial N_i}{\partial y}
\end{Bmatrix}
= \mathbf{J}
\begin{Bmatrix}
\frac{\partial N_i}{\partial x} \\
\frac{\partial N_i}{\partial y}
\end{Bmatrix}
$$

Here, $\mathbf{J}$ is the **Jacobian matrix**. It acts as the mathematical translator between the reference element and the physical element. The components of the Jacobian matrix $\mathbf{J}$ are easily evaluated using the derivatives of the coordinate interpolation equations:

$$
\mathbf{J} = \begin{bmatrix}
\sum_{i=1}^4 \frac{\partial N_i}{\partial \xi} x_i & \sum_{i=1}^4 \frac{\partial N_i}{\partial \xi} y_i \\
\sum_{i=1}^4 \frac{\partial N_i}{\partial \eta} x_i & \sum_{i=1}^4 \frac{\partial N_i}{\partial \eta} y_i
\end{bmatrix}
$$

To find the desired Cartesian derivatives of the shape functions, we simply invert the Jacobian matrix:

$$
\begin{Bmatrix}
\frac{\partial N_i}{\partial x} \\
\frac{\partial N_i}{\partial y}
\end{Bmatrix}
= \mathbf{J}^{-1}
\begin{Bmatrix}
\frac{\partial N_i}{\partial \xi} \\
\frac{\partial N_i}{\partial \eta}
\end{Bmatrix}
$$

The determinant of the Jacobian, denoted as $|\mathbf{J}|$ or $\det(\mathbf{J})$, has a profound physical meaning: it provides the scale factor relating the infinitesimal area in the parent domain to the physical domain, i.e., $dx dy = |\mathbf{J}| d\xi d\eta$. 

For the mapping to be valid, bijective, and invertible, $|\mathbf{J}|$ must be strictly positive everywhere inside the element. If $|\mathbf{J}| \leq 0$ at any point, the physical element is unacceptably distorted (e.g., self-intersecting, has a re-entrant corner, or nodes are numbered clockwise instead of counter-clockwise).

## 3. The Strain-Displacement Matrix ($\mathbf{B}$)

In 2D plane elasticity (plane stress or plane strain), the strain vector in Voigt notation is given by $\boldsymbol{\varepsilon} = [\varepsilon_{xx}, \varepsilon_{yy}, \gamma_{xy}]^T$. Using the small-strain kinematic relations:

$$
\boldsymbol{\varepsilon} = \begin{Bmatrix}
\frac{\partial u}{\partial x} \\
\frac{\partial v}{\partial y} \\
\frac{\partial u}{\partial y} + \frac{\partial v}{\partial x}
\end{Bmatrix}
$$

Substituting the interpolated displacements into the strain definitions yields the linear relationship between the strains at any point and the nodal displacement vector $\mathbf{d}_e = [u_1, v_1, u_2, v_2, u_3, v_3, u_4, v_4]^T$:

$$
\boldsymbol{\varepsilon} = \mathbf{B} \mathbf{d}_e
$$

Where $\mathbf{B}$ is the strain-displacement matrix. For the Q4 element, $\mathbf{B}$ is a $3 \times 8$ matrix constructed from sub-matrices $\mathbf{B}_i$ for each node $i$:

$$
\mathbf{B} = \begin{bmatrix} \mathbf{B}_1 & \mathbf{B}_2 & \mathbf{B}_3 & \mathbf{B}_4 \end{bmatrix}
$$

Each nodal sub-matrix $\mathbf{B}_i$ is defined using the Cartesian derivatives of the shape functions we obtained via the Jacobian:

$$
\mathbf{B}_i = \begin{bmatrix}
\frac{\partial N_i}{\partial x} & 0 \\
0 & \frac{\partial N_i}{\partial y} \\
\frac{\partial N_i}{\partial y} & \frac{\partial N_i}{\partial x}
\end{bmatrix}
$$

## 4. Gauss-Legendre Numerical Integration

The element stiffness matrix $\mathbf{K}_e$ is defined by the integral of the strain energy over the element volume $V_e$:

$$
\mathbf{K}_e = \int_{V_e} \mathbf{B}^T \mathbf{D} \mathbf{B} \, dV
$$

Assuming a uniform element thickness $t$, this becomes an area integral over the physical domain $\Omega_e$:

$$
\mathbf{K}_e = t \int_{\Omega_e} \mathbf{B}^T \mathbf{D} \mathbf{B} \, dx dy
$$

Where $\mathbf{D}$ is the constitutive (material) matrix. Because the entries of $\mathbf{B}$ are rational functions (due to dividing by the Jacobian determinant during inversion), exact analytical integration is generally impossible for an arbitrary mapped quadrilateral. We must transform the integral to the parent domain and use numerical quadrature:

$$
\mathbf{K}_e = t \int_{-1}^{1} \int_{-1}^{1} \mathbf{B}^T(\xi, \eta) \mathbf{D} \mathbf{B}(\xi, \eta) |\mathbf{J}(\xi, \eta)| \, d\xi d\eta
$$

The standard approach for a full-integration Q4 is the $2 \times 2$ Gauss-Legendre quadrature. This rule uses four integration points (Gauss points) strategically located within the parent element. The integral is approximated as a weighted sum of the integrand evaluated at these points:

$$
\mathbf{K}_e \approx \sum_{i=1}^2 \sum_{j=1}^2 w_i w_j \left[ \mathbf{B}^T(\xi_i, \eta_j) \mathbf{D} \mathbf{B}(\xi_i, \eta_j) |\mathbf{J}(\xi_i, \eta_j)| t \right]
$$

For the $2 \times 2$ rule, the coordinates and weights are:
*   Points $\xi_i, \eta_j \in \left\{ -\frac{1}{\sqrt{3}}, \frac{1}{\sqrt{3}} \right\}$
*   Weights $w_i = w_j = 1.0$

This $2 \times 2$ rule perfectly evaluates the stiffness matrix and prevents zero-energy hourglass modes that plague reduced $1 \times 1$ integration.

## 5. Software Implementation in `femlabpy`

The `femlabpy` package provides robust and efficient implementations of Q4 elements. Let's look closely at how the math translates into the `keq4e` and `qeq4e` functions.

### 5.1 The `keq4e` Gauss Loop and `np.linalg.solve`

The `keq4e` function calculates the $8 \times 8$ element stiffness matrix for a linearly elastic plane stress or plane strain problem. A critical part of its implementation is the loop over the four Gauss points:

```python
    Ke = np.zeros((8, 8))
    # 2x2 Gauss integration points and weights
    a = 1.0 / np.sqrt(3.0)
    gauss_pts = np.array([[-a, -a], [a, -a], [a, a], [-a, a]])
    weights = np.array([1.0, 1.0, 1.0, 1.0])

    for i in range(4):
        xi, eta = gauss_pts[i]
        weight = weights[i]

        # 1. Local shape function derivatives
        # dN is 2x4: [ [dN1/dxi,   dN2/dxi,   ... ],
        #              [dN1/deta,  dN2/deta,  ... ] ]
        dN = _q4_dN(xi, eta)

        # 2. Jacobian transpose matrix (2x2) = dN * Xe
        Jt = dN @ Xe
        detJ = np.linalg.det(Jt)

        # 3. Global shape function derivatives
        # We solve Jt * dN_global = dN instead of explicitly inverting Jt
        dN_global = np.linalg.solve(Jt, dN)

        # 4. Construct B matrix (3x8)
        B = _q4_B(dN_global)

        # 5. Accumulate stiffness
        Ke += weight * (B.T @ D @ B) * detJ * t
```

**Understanding `dN_global = np.linalg.solve(Jt, dN)`**

Recall the mathematical relationship:
$$
\begin{Bmatrix} \frac{\partial N_i}{\partial \xi} \\ \frac{\partial N_i}{\partial \eta} \end{Bmatrix} = \mathbf{J} \begin{Bmatrix} \frac{\partial N_i}{\partial x} \\ \frac{\partial N_i}{\partial y} \end{Bmatrix}
$$
In code, our matrices are typically laid out with points/nodes in columns. By taking the transpose, we obtain a system of linear equations mapping the local derivatives `dN` to the global derivatives `dN_global`. 

Instead of computing the inverse explicitly with `np.linalg.inv(Jt)` and then multiplying by `dN`—which can be computationally inefficient and sensitive to floating-point inaccuracies when elements are distorted—`femlabpy` cleverly uses `np.linalg.solve(Jt, dN)`. This numerically solves the linear system $\mathbf{J}^T \nabla_{\mathbf{x}} \mathbf{N} = \nabla_{\boldsymbol{\xi}} \mathbf{N}$, directly yielding the highly accurate physical derivatives (`dN_global`) needed for the $\mathbf{B}$ matrix.

### 5.2 Internal Forces and Stresses: `qeq4e`

Once the global displacement vector is solved, we extract the local nodal displacements $\mathbf{d}_e$ and use it to compute the internal forces, stresses, and strains at the Gauss points using `qeq4e`.

Unlike simpler elements (like constant strain triangles), the stress $\boldsymbol{\sigma}$ and strain $\boldsymbol{\varepsilon}$ inside a Q4 element are *not* constant. They vary bilinearly over the element's area. Therefore, stresses are typically sampled at the optimal locations: the 4 Gauss integration points.

The `qeq4e` implementation performs a similar loop over the Gauss points as the stiffness matrix assembly:
1.  Calculates `dN_global` at the current Gauss point using `np.linalg.solve(Jt, dN)`.
2.  Constructs the $\mathbf{B}$ matrix.
3.  Evaluates the strains: $\boldsymbol{\varepsilon} = \mathbf{B} \mathbf{d}_e$.
4.  Evaluates the stresses: $\boldsymbol{\sigma} = \mathbf{D} \boldsymbol{\varepsilon}$.
5.  Accumulates the equivalent nodal forces: $\mathbf{q}_e \mathrel{+}= \mathbf{B}^T \boldsymbol{\sigma} |\mathbf{J}| t w_i$.

The output includes a matrix `Se` of shape $(4, 3)$, storing the $[\sigma_{xx}, \sigma_{yy}, \tau_{xy}]$ components at each of the four discrete Gauss points.

### 5.3 Runnable Script: End-to-End Q4 Element Evaluation

The following interactive script demonstrates assembling a single Q4 element, generating an artificial displacement field, and retrieving the stresses at the Gauss points.

```python
import numpy as np

# Mocking the core femlabpy helper functions for this standalone script
def _q4_dN(xi, eta):
    return np.array([
        [-0.25*(1-eta),  0.25*(1-eta), 0.25*(1+eta), -0.25*(1+eta)],
        [-0.25*(1-xi),  -0.25*(1+xi),  0.25*(1+xi),   0.25*(1-xi) ]
    ])

def _q4_B(dN_global):
    B = np.zeros((3, 8))
    for i in range(4):
        B[0, 2*i]   = dN_global[0, i]
        B[1, 2*i+1] = dN_global[1, i]
        B[2, 2*i]   = dN_global[1, i]
        B[2, 2*i+1] = dN_global[0, i]
    return B

# Material Matrix D (Plane Stress)
E = 200e9
nu = 0.3
t = 0.05
factor = E / (1.0 - nu**2)
D = factor * np.array([
    [1.0, nu,  0.0],
    [nu,  1.0, 0.0],
    [0.0, 0.0, (1.0 - nu) / 2.0]
])

# Define the nodal coordinates [x, y] for the 4 nodes
Xe = np.array([
    [0.0, 0.0],
    [2.0, 0.0],
    [2.0, 1.5],
    [0.0, 1.5]
])

# Assume some arbitrary computed displacements for the 8 DOFs
Ue = np.array([0.0, 0.0, 1e-4, -5e-5, 1.2e-4, -2e-5, 0.2e-4, 1e-5])

# --- Evaluate qeq4e ---
qe = np.zeros(8)
Se = np.zeros((4, 3))
Ee = np.zeros((4, 3))

a = 1.0 / np.sqrt(3.0)
gauss_pts = np.array([[-a, -a], [a, -a], [a, a], [-a, a]])
weights = np.array([1.0, 1.0, 1.0, 1.0])

for i in range(4):
    xi, eta = gauss_pts[i]
    weight = weights[i]
    
    dN = _q4_dN(xi, eta)
    Jt = dN @ Xe
    detJ = np.linalg.det(Jt)
    
    # Solve Jt * dN_global = dN
    dN_global = np.linalg.solve(Jt, dN)
    
    B = _q4_B(dN_global)
    
    # Strain and Stress
    eps = B @ Ue
    sig = D @ eps
    
    Ee[i, :] = eps
    Se[i, :] = sig
    
    # Internal forces
    qe += weight * (B.T @ sig) * detJ * t

print("--- Q4 Element Evaluation ---")
print("Internal Force Vector (qe):")
print(np.array2string(qe, precision=2, suppress_small=True))
print("\nStresses at 4 Gauss Points [Sxx, Syy, Sxy] (Pa):")
print(np.array2string(Se, precision=2, suppress_small=True))
```

## 6. Summary

The 2D Isoparametric Q4 element is a workhorse in computational mechanics. By mapping complex geometries to a simple parent domain, evaluating Jacobian matrices to relate local and global spaces, and employing Gauss-Legendre quadrature, it provides a robust numerical framework. The `femlabpy` library implements these steps efficiently—notably using `np.linalg.solve` for stable derivative mappings—providing a transparent and rigorous toolset for engineers simulating complex 2D structures.
