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

# Chapter 3: 1D Bar Elements (Linear & Nonlinear)

This chapter introduces the formulation of 1D bar elements (trusses), which are fundamental building blocks in computational mechanics. We will start with the basic linear assumption and extend the theory to encompass large deformations using the Green-Lagrange strain. 

By the end of this chapter, you will understand the underlying physics and mathematics, and see exactly how the tangent stiffness matrices and internal force vectors are implemented in Python.

## 3.1 Linear 1D Bar Formulation

The simplest finite element is the 2-node linear bar element. A bar can only transmit axial loads—meaning it has no bending or shear stiffness.

### 3.1.1 Kinematics and Strain
Consider a 1D bar of initial length $L$, cross-sectional area $A$, and Young's modulus $E$. Let the bar be aligned along the local $x$-axis. The bar is defined by two nodes, $1$ and $2$, with displacements $u_1$ and $u_2$.

The axial strain $\varepsilon$ under the assumption of small deformations is simply the change in length divided by the original length:

$$ \varepsilon = \frac{\Delta L}{L} = \frac{u_2 - u_1}{L} $$

Using the standard finite element approach, we define the element displacement vector $\mathbf{d}^e = [u_1, u_2]^T$. The strain-displacement relation can be written in matrix form:

$$ \varepsilon = \mathbf{B} \mathbf{d}^e = \begin{bmatrix} -\frac{1}{L} & \frac{1}{L} \end{bmatrix} \begin{bmatrix} u_1 \\ u_2 \end{bmatrix} $$

where $\mathbf{B}$ is the strain-displacement matrix.

### 3.1.2 The Linear Stiffness Matrix
The principle of virtual work or minimizing the total potential energy yields the element stiffness matrix $\mathbf{K}^e$:

$$ \mathbf{K}^e = \int_V \mathbf{B}^T E \mathbf{B} \, dV $$

Since the area $A$ and modulus $E$ are constant, the volume integral becomes $A \cdot \int_0^L dx$:

$$ \mathbf{K}^e = E A L \left( \begin{bmatrix} -\frac{1}{L} \\ \frac{1}{L} \end{bmatrix} \begin{bmatrix} -\frac{1}{L} & \frac{1}{L} \end{bmatrix} \right) = \frac{EA}{L} \begin{bmatrix} 1 & -1 \\ -1 & 1 \end{bmatrix} $$

This is the canonical local linear stiffness matrix for a 2-node bar element. 

## 3.2 Geometric Nonlinear Formulation

When displacements and rotations become large, the linear assumption ($\varepsilon = \Delta L / L$) is no longer valid. Even if the material remains linear-elastic, large rigid body rotations introduce significant geometric nonlinearity.

To capture large deformations, we must use an objective measure of strain, such as the Green-Lagrange strain.

### 3.2.1 Green-Lagrange Strain ($\epsilon$)
For 1D problems where displacement is given by $u(x)$, the true Green-Lagrange strain measures the stretch and includes higher-order deformation terms:

$$ \epsilon = \frac{du}{dx} + \frac{1}{2}\left(\frac{du}{dx}\right)^2 $$

For a bar with initial node coordinates $\mathbf{X}_1$ and $\mathbf{X}_2$ and a constant strain along its length, this translates to the squared change in lengths. Let the initial length vector be $\mathbf{a}_0 = \mathbf{X}_2 - \mathbf{X}_1$ with length $l_0$, and the current length vector be $\mathbf{a}_1 = \mathbf{x}_2 - \mathbf{x}_1$ with length $l_1$. The discrete Green-Lagrange strain evaluates equivalently to:

$$ \epsilon = \frac{l_1^2 - l_0^2}{2 l_0^2} $$

This strain measure is invariant under rigid body translations and rotations because it strictly depends on the squared lengths. 

### 3.2.2 Internal Force Response ($\mathbf{q}^e$)
The internal force vector is derived from the variation of the internal strain energy $U = \int_V \frac{1}{2} E \epsilon^2 dV$. The variation of the Green-Lagrange strain $\delta \epsilon$ is related to virtual displacements $\delta \mathbf{u}$:

$$ \delta \epsilon = \frac{1}{l_0^2} \mathbf{a}_1 \cdot \delta \mathbf{a}_1 = \frac{1}{l_0^2} \mathbf{a}_1^T (\delta \mathbf{u}_2 - \delta \mathbf{u}_1) $$

Defining the normal stress as $S = E \epsilon$ and the normal force as $N = A S = A E \epsilon$, the virtual work $\int_V S \delta \epsilon \, dV$ yields the internal force vector corresponding to the nodal displacements:

$$ \mathbf{q}^e = \frac{N}{l_0} \begin{bmatrix} -\mathbf{a}_1 \\ \mathbf{a}_1 \end{bmatrix} $$

#### Implementation: `qebar`
Here is the exact NumPy implementation for evaluating the internal force of a nonlinear bar.

```python
import numpy as np

def qebar(Xe0, Xe1, Ge):
    """Compute the internal-force response of a single geometrically nonlinear bar."""
    initial = np.array(Xe0, dtype=float)
    current = np.array(Xe1, dtype=float)
    props = np.array(Ge, dtype=float).reshape(-1)

    a0 = (initial[1] - initial[0]).reshape(-1, 1)
    l0 = float(np.linalg.norm(a0))
    a1 = (current[1] - current[0]).reshape(-1, 1)
    l1 = float(np.linalg.norm(a1))

    A = props[0]
    E = props[1] if props.size > 1 else 1.0
    
    # Green-Lagrange Strain
    strain = 0.5 * (l1**2 - l0**2) / l0**2
    stress = E * strain
    
    # Internal Force Vector
    qe = (A * stress / l0) * np.vstack([-a1, a1])
    return qe, float(stress), float(strain)
```

### 3.2.3 Tangent Stiffness Matrix ($\mathbf{K}^e_{tan}$)
For nonlinear solvers like the Newton-Raphson method, we require the tangent stiffness matrix, which is the derivative of the internal force vector with respect to the nodal displacements:

$$ \mathbf{K}_{tan}^e = \frac{\partial \mathbf{q}^e}{\partial \mathbf{d}^e} $$

Taking the derivative of the internal force vector with respect to displacement gives two distinct terms via the product rule, corresponding to $K_m$ and $K_g$:

$$ \mathbf{K}_{tan}^e = \mathbf{K}_m + \mathbf{K}_g $$

#### 1. Material Stiffness Matrix ($\mathbf{K}_m$)
The material stiffness arises from taking the variation of the normal force $N$ itself (which depends on the strain variation). It projects the axial stiffness into the current spatial configuration:

$$ \mathbf{K}_m = \frac{E A}{l_0^3} \begin{bmatrix} \mathbf{a}_1 \mathbf{a}_1^T & -\mathbf{a}_1 \mathbf{a}_1^T \\ -\mathbf{a}_1 \mathbf{a}_1^T & \mathbf{a}_1 \mathbf{a}_1^T \end{bmatrix} $$

#### 2. Geometric Stiffness Matrix ($\mathbf{K}_g$)
The geometric stiffness part comes from the derivative of the spatial vector $\mathbf{a}_1$ while treating the axial force $N$ as constant. It represents how the current tension/compression modifies the structural stiffness, regardless of material changes.

In a purely 1D local context, this evaluates strictly as:
$$ K_g = \frac{N}{L} \begin{bmatrix} 1 & -1 \\ -1 & 1 \end{bmatrix} $$

For 2D or 3D trusses, it generalizes to:
$$ \mathbf{K}_g = \frac{N}{l_0} \begin{bmatrix} \mathbf{I} & -\mathbf{I} \\ -\mathbf{I} & \mathbf{I} \end{bmatrix} $$

#### Implementation: `kebar`
Here is the exact NumPy implementation for generating the local nonlinear tangent stiffness matrix. Note how $K_g$ uses the Identity matrix to expand the local 1D $K_g$ equation across dimensions.

```python
def kebar(Xe0, Xe1, Ge):
    """Compute the tangent stiffness matrix of a geometrically nonlinear bar element."""
    initial = np.array(Xe0, dtype=float)
    current = np.array(Xe1, dtype=float)
    props = np.array(Ge, dtype=float).reshape(-1)

    a0 = (initial[1] - initial[0]).reshape(-1, 1)
    l0 = float(np.linalg.norm(a0))
    a1 = (current[1] - current[0]).reshape(-1, 1)
    l1 = float(np.linalg.norm(a1))

    A = props[0]
    E = props[1] if props.size > 1 else 1.0
    
    # Strain and Normal Force
    strain = 0.5 * (l1**2 - l0**2) / l0**2
    normal_force = A * E * strain
    
    # Matrix Components
    identity = np.eye(a0.shape[0], dtype=float)
    
    # K_m (Material Stiffness)
    K_m = (E * A / l0**3) * np.block([
        [ a1 @ a1.T, -a1 @ a1.T],
        [-a1 @ a1.T,  a1 @ a1.T]
    ])
    
    # K_g (Geometric Stiffness)
    K_g = (normal_force / l0) * np.block([
        [ identity, -identity],
        [-identity,  identity]
    ])
    
    return K_m + K_g
```

## 3.4 The Newton-Raphson Iteration Scheme & Example

To solve a nonlinear equilibrium path $\mathbf{q}(\mathbf{d}) = \mathbf{f}_{ext}$, the Newton-Raphson scheme iteratively finds the displacement increment $\Delta \mathbf{d}$.

Below is a complete, runnable Python script solving a 2-bar snap-through truss model using a manual Newton-Raphson loop.

``` python
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# Element Functions
# ---------------------------------------------------------
def qebar(Xe0, Xe1, Ge):
    a0 = (Xe0[1] - Xe0[0]).reshape(-1, 1)
    l0 = np.linalg.norm(a0)
    a1 = (Xe1[1] - Xe1[0]).reshape(-1, 1)
    l1 = np.linalg.norm(a1)
    A, E = Ge[0], Ge[1]
    
    strain = 0.5 * (l1**2 - l0**2) / l0**2
    stress = E * strain
    qe = (A * stress / l0) * np.vstack([-a1, a1])
    return qe

def kebar(Xe0, Xe1, Ge):
    a0 = (Xe0[1] - Xe0[0]).reshape(-1, 1)
    l0 = np.linalg.norm(a0)
    a1 = (Xe1[1] - Xe1[0]).reshape(-1, 1)
    l1 = np.linalg.norm(a1)
    A, E = Ge[0], Ge[1]
    
    strain = 0.5 * (l1**2 - l0**2) / l0**2
    normal_force = A * E * strain
    
    I = np.eye(2)
    K_m = (E * A / l0**3) * np.block([
        [ a1 @ a1.T, -a1 @ a1.T],
        [-a1 @ a1.T,  a1 @ a1.T]
    ])
    K_g = (normal_force / l0) * np.block([
        [ I, -I],
        [-I,  I]
    ])
    return K_m + K_g

# ---------------------------------------------------------
# Model Definition: 2-Bar Snap-Through Truss
# ---------------------------------------------------------
# Nodes: 0 (Left pin), 1 (Right pin), 2 (Center, loaded)
nodes0 = np.array([
    [-10.0, 0.0],
    [ 10.0, 0.0],
    [  0.0, 5.0]
])

# Two elements connecting nodes (0->2) and (1->2)
elems = [[0, 2], [1, 2]]
props = [1.0, 1000.0] # Area = 1.0, E = 1000.0

# Free DOFs: Node 2 x and y (Indices 4 and 5)
free_dofs = [4, 5]

# External Load at Node 2 (y-direction)
f_ext = np.zeros(6)
f_ext[5] = -250.0  # Push down

# Initial displacement
u = np.zeros(6)

# ---------------------------------------------------------
# Newton-Raphson Loop
# ---------------------------------------------------------
tol = 1e-6
max_iter = 20

print("Iter | Residual Norm")
print("--------------------")
for i in range(max_iter):
    # Current positions
    nodes1 = nodes0 + u.reshape(-1, 2)
    
    # Global Assembly
    K_global = np.zeros((6, 6))
    q_global = np.zeros(6)
    
    for el in elems:
        idx = [el[0]*2, el[0]*2+1, el[1]*2, el[1]*2+1]
        
        Xe0 = np.vstack([nodes0[el[0]], nodes0[el[1]]])
        Xe1 = np.vstack([nodes1[el[0]], nodes1[el[1]]])
        
        ke = kebar(Xe0, Xe1, props)
        qe = qebar(Xe0, Xe1, props)
        
        for r in range(4):
            q_global[idx[r]] += qe[r, 0]
            for c in range(4):
                K_global[idx[r], idx[c]] += ke[r, c]
                
    # Check Residual
    R = f_ext[free_dofs] - q_global[free_dofs]
    norm_R = np.linalg.norm(R)
    print(f"{i:4d} | {norm_R:.6e}")
    
    if norm_R < tol:
        print("Converged!")
        break
        
    # Tangent stiffness at free DOFs
    K_free = K_global[np.ix_(free_dofs, free_dofs)]
    
    # Solve for displacement increment
    du_free = np.linalg.solve(K_free, R)
    u[free_dofs] += du_free

print("\nFinal Displacements:")
print(f"Node 2 u_x = {u[4]:.6f}")
print(f"Node 2 u_y = {u[5]:.6f}")
```
