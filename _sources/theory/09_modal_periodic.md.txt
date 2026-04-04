# Chapter 9: Eigenvalue Problems and Periodic Boundaries

In computational mechanics, evaluating the dynamic characteristics and effective macroscopic properties of materials forms a cornerstone of modern engineering analysis. This chapter delves into two advanced topics fundamental to finite element analysis (FEA): Free Vibration Modal Analysis and the enforcement of Periodic Boundary Conditions (PBCs) for Computational Homogenization.

By examining the generalized eigenvalue problem, we will characterize the natural frequencies and mode shapes of a structure. Subsequently, we explore Representative Volume Elements (RVEs) and how microstructural properties are scaled up using periodic boundaries and Lagrange multipliers to extract the effective macroscopic matrix $\mathbf{C}_{eff}$.

---

## 9.1 Free Vibration Modal Analysis

The dynamic response of an undamped structural system without external forcing is governed by the homogeneous equation of motion:

$$
\mathbf{M} \ddot{\mathbf{u}}(t) + \mathbf{K} \mathbf{u}(t) = \mathbf{0}
$$

where $\mathbf{M}$ is the global mass matrix, $\mathbf{K}$ is the global stiffness matrix, and $\mathbf{u}(t)$ is the nodal displacement vector as a function of time. Assuming a harmonic solution of the form $\mathbf{u}(t) = \phi e^{i\omega t}$, taking the second time derivative yields $\ddot{\mathbf{u}}(t) = -\omega^2 \phi e^{i\omega t}$. Substituting this back into the equation of motion provides the classic generalized eigenvalue problem:

$$
\left( \mathbf{K} - \omega^2 \mathbf{M} \right) \phi = \mathbf{0}
$$

Rearranging this expression yields the canonical form for free vibration modal analysis:

$$
\mathbf{K}\phi = \omega^2 \mathbf{M}\phi
$$

In this formulation:
- $\omega_i^2$ represents the $i$-th eigenvalue, where $\omega_i$ is the natural angular frequency of the $i$-th mode in radians per second.
- $\phi_i$ is the $i$-th eigenvector, which physically represents the mode shape or spatial deformation pattern of the structure oscillating at $\omega_i$.

Because $\mathbf{K}$ and $\mathbf{M}$ are typically real, symmetric, and positive-definite (assuming proper rigid body constraints are applied), all eigenvalues $\omega_i^2$ are real and non-negative, and the corresponding eigenvectors are real-valued.

### 9.1.1 Solving the Modal Problem and Dropping Constrained DOFs

When solving the generalized eigenvalue problem, boundary conditions must be applied to prevent rigid body motions and ensure the stiffness matrix $\mathbf{K}$ is non-singular. Rather than using the penalty method (which artificially inflates eigenvalues), it is standard practice to partition the matrices and drop the constrained Degrees of Freedom (DOFs). 

In Python, the `np.ix_` function provides an elegant way to extract the "free-free" portion of the stiffness and mass matrices. Below is the implementation of `solve_modal`:

```python
import numpy as np
import scipy.linalg

def solve_modal(K, M, constrained_dofs, num_modes=10):
    """
    Solve K phi = omega^2 M phi for the lowest modes.
    constrained_dofs: list or array of DOF indices to fix.
    """
    n_dofs = K.shape[0]
    all_dofs = np.arange(n_dofs)
    
    # Identify free DOFs by removing constrained ones
    free_dofs = np.setdiff1d(all_dofs, constrained_dofs)
    
    # Extract the free-free block using np.ix_
    idx = np.ix_(free_dofs, free_dofs)
    K_ff = K[idx]
    M_ff = M[idx]
    
    # Solve generalized eigenvalue problem for lowest num_modes
    # eigh is optimized for symmetric matrices
    eigvals, eigvecs_ff = scipy.linalg.eigh(K_ff, M_ff, subset_by_index=[0, num_modes-1])
    
    # Reconstruct full eigenvectors including zero displacements at constrained DOFs
    eigvecs = np.zeros((n_dofs, num_modes))
    eigvecs[free_dofs, :] = eigvecs_ff
    
    # Natural frequencies
    omega = np.sqrt(np.abs(eigvals))
    
    return omega, eigvecs
```

---

## 9.2 Periodic Boundary Conditions (PBC)

In multiscale computational mechanics, the macroscopic behavior of heterogeneous materials (like composites, foams, or lattice structures) is derived by analyzing a microscopic Representative Volume Element (RVE). An RVE must be statistically representative of the bulk material.

To ensure that the RVE behaves as though it is embedded within an infinite periodic medium, Periodic Boundary Conditions (PBCs) are applied to its boundaries. This avoids artificial stiffening or softening at the edges, which would occur with standard Dirichlet or Neumann boundaries.

### 9.2.1 The Periodic Constraint Equation

Consider a 2D rectangular RVE with dimensions $\Delta x$ and $\Delta y$. The boundaries consist of paired sets of nodes: a right face (+) and a left face (-), as well as a top face (+) and a bottom face (-).

For the RVE to remain continuous with its periodic neighbors under a given macroscopic strain field $\bar{\boldsymbol{\epsilon}}$, the displacements of corresponding nodes on opposite boundary faces must satisfy:

$$
\mathbf{u}^+ - \mathbf{u}^- = \bar{\boldsymbol{\epsilon}} \Delta \mathbf{x}
$$

For a 2D continuum, the macroscopic strain tensor has three independent components: $\bar{\epsilon}_{xx}$, $\bar{\epsilon}_{yy}$, and $\bar{\gamma}_{xy}$. The constraint equations for the $x$-displacements ($u$) and $y$-displacements ($v$) between a node on the right face $(x^+, y)$ and a corresponding node on the left face $(x^-, y)$ become:

$$
u^+ - u^- = \bar{\epsilon}_{xx} \Delta x + \frac{1}{2} \bar{\gamma}_{xy} \Delta y
$$
$$
v^+ - v^- = \frac{1}{2} \bar{\gamma}_{xy} \Delta x + \bar{\epsilon}_{yy} \Delta y
$$

### 9.2.2 Enforcement via Lagrange Multipliers

In the finite element framework, these multipoint constraints are enforced using Lagrange multipliers. For each pair of periodic nodes, an additional algebraic constraint equation is appended to the global system. 

Let $\mathbf{C}_{pbc}$ be the boolean constraint matrix and $\mathbf{q}$ be the vector of prescribed relative displacements (derived from $\bar{\boldsymbol{\epsilon}} \Delta \mathbf{x}$). The extended system to solve becomes a saddle-point problem:

$$
\begin{bmatrix}
\mathbf{K} & \mathbf{C}_{pbc}^T \\
\mathbf{C}_{pbc} & \mathbf{0}
\end{bmatrix}
\begin{Bmatrix}
\mathbf{u} \\
\boldsymbol{\lambda}
\end{Bmatrix}
=
\begin{Bmatrix}
\mathbf{f}_{ext} \\
\mathbf{q}
\end{Bmatrix}
$$

Here, $\boldsymbol{\lambda}$ represents the Lagrange multipliers, which physically correspond to the unknown reaction forces maintaining periodicity. For homogenization, $\mathbf{f}_{ext} = \mathbf{0}$ for internal nodes, and the deformation is entirely driven by $\mathbf{q}$.

---

## 9.3 Computational Homogenization

The primary objective of analyzing an RVE under PBCs is to extract the effective macroscopic properties—specifically, the homogenized stiffness matrix (or compliance matrix) $\mathbf{C}_{eff}$. In 2D plane elasticity, $\mathbf{C}_{eff}$ is a $3 \times 3$ matrix relating the macroscopic stresses $\bar{\boldsymbol{\sigma}}$ to macroscopic strains $\bar{\boldsymbol{\epsilon}}$:

$$
\bar{\boldsymbol{\sigma}} = \mathbf{C}_{eff} \bar{\boldsymbol{\epsilon}}
$$

### 9.3.1 Driving Macro-Strain States

To fully populate the $\mathbf{C}_{eff}$ matrix, the `homogenize` routine systematically drives the RVE through three independent, unit macroscopic strain states. The three tests are:

1.  **Test 1: Pure X-Tension:**
    $$ \bar{\boldsymbol{\epsilon}}^{(1)} = [1, 0, 0]^T \implies \bar{\epsilon}_{xx} = 1, \bar{\epsilon}_{yy} = 0, \bar{\gamma}_{xy} = 0 $$
2.  **Test 2: Pure Y-Tension:**
    $$ \bar{\boldsymbol{\epsilon}}^{(2)} = [0, 1, 0]^T \implies \bar{\epsilon}_{xx} = 0, \bar{\epsilon}_{yy} = 1, \bar{\gamma}_{xy} = 0 $$
3.  **Test 3: Pure In-Plane Shear:**
    $$ \bar{\boldsymbol{\epsilon}}^{(3)} = [0, 0, 1]^T \implies \bar{\epsilon}_{xx} = 0, \bar{\epsilon}_{yy} = 0, \bar{\gamma}_{xy} = 1 $$

### 9.3.2 Extracting the Effective Stiffness

For each load case $k \in \{1, 2, 3\}$, the system of equations is solved for the nodal displacements $\mathbf{u}^{(k)}$ and the Lagrange multipliers $\boldsymbol{\lambda}^{(k)}$.

The macroscopic stress $\bar{\boldsymbol{\sigma}}^{(k)}$ corresponding to the imposed strain $\bar{\boldsymbol{\epsilon}}^{(k)}$ is calculated by averaging the stress over the volume $V$ of the RVE. Due to Hill's energy condition, this volume average can be computed using the boundary traction forces or by evaluating the total element stresses:

$$
\langle \sigma_{ij} \rangle = \frac{1}{V} \int_V \sigma_{ij} dV 
$$

Since the imposed strain vectors were chosen as unit basis vectors, the resulting macroscopic stress vector $\langle \boldsymbol{\sigma}^{(k)} \rangle = [\langle \sigma_{xx}^{(k)} \rangle, \langle \sigma_{yy}^{(k)} \rangle, \langle \sigma_{xy}^{(k)} \rangle]^T$ directly constitutes the $k$-th column of the homogenized matrix $\mathbf{C}_{eff}$.

$$
\mathbf{C}_{eff} = \left[ \langle \boldsymbol{\sigma}^{(1)} \rangle \quad \langle \boldsymbol{\sigma}^{(2)} \rangle \quad \langle \boldsymbol{\sigma}^{(3)} \rangle \right]
$$

### 9.3.3 Implementation Details

The following runnable Python script encapsulates the `homogenize` logic described above, demonstrating the application of the 3 macro-strain cases and averaging of stresses:

```python
import numpy as np

def homogenize(solve_rve_state, V_rve):
    """
    Computes the effective 3x3 stiffness matrix C_eff.
    
    solve_rve_state: function that takes a macro-strain vector [e11, e22, g12]
                     and returns the volume-averaged stress vector [s11, s22, s12].
    V_rve: Volume of the RVE.
    """
    C_eff = np.zeros((3, 3))
    
    # Define the 3 independent macro-strain cases
    strain_cases = [
        np.array([1.0, 0.0, 0.0]), # Case 1: Pure X-Tension
        np.array([0.0, 1.0, 0.0]), # Case 2: Pure Y-Tension
        np.array([0.0, 0.0, 1.0])  # Case 3: Pure Shear
    ]
    
    for i, macro_strain in enumerate(strain_cases):
        # Solve the boundary value problem for the RVE under PBCs
        # and compute the volume-averaged stress <sigma>
        avg_stress = solve_rve_state(macro_strain)
        
        # The averaged stress forms the i-th column of C_eff
        C_eff[:, i] = avg_stress
        
    return C_eff

# --- Runnable Demonstration ---
if __name__ == "__main__":
    # Mock RVE volume
    V = 1.0
    
    # Mock function representing the FEA solution of the RVE
    # Let's pretend the material has E=100, nu=0.3 (plane stress)
    E = 100.0
    nu = 0.3
    factor = E / (1 - nu**2)
    mock_C = factor * np.array([
        [1.0, nu, 0.0],
        [nu, 1.0, 0.0],
        [0.0, 0.0, (1 - nu) / 2]
    ])
    
    def mock_solve_rve(macro_strain):
        # In a real code, this would build the constraint matrix,
        # solve the saddle-point problem, integrate element stresses,
        # and divide by V. Here we just multiply by our mock C.
        return mock_C @ macro_strain
    
    # Run homogenization
    C_effective = homogenize(mock_solve_rve, V)
    
    print("Effective Stiffness Matrix C_eff:")
    print(np.round(C_effective, 3))
```

The resulting $\mathbf{C}_{eff}$ seamlessly replaces detailed microstructural models in macro-scale FE simulations, retaining the intricate geometric and material influences of the underlying architecture.