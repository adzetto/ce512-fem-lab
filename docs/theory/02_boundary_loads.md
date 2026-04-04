# Chapter 2: Boundary Conditions and Loads

In computational mechanics, the formulation of a finite element problem ultimately leads to a system of algebraic equations, typically expressed in the form of a global system:

$$ \mathbf{K} \mathbf{u} = \mathbf{P} $$

where $\mathbf{K}$ is the global stiffness matrix, $\mathbf{u}$ is the vector of unknown nodal displacements, and $\mathbf{P}$ is the global load vector. However, before this system can be solved, the equations are singular; they represent a body floating freely in space with rigid body modes. To render the system nonsingular and to reflect the physical reality of the problem, we must impose constraints (Boundary Conditions) and apply external forces (Loads).

This chapter discusses the mathematical theory and the computational implementation of boundary conditions and loads within the `femlabpy` framework, specifically focusing on `src/femlabpy/boundary.py` and `src/femlabpy/loads.py`.

---

## 2.1 Applying External Loads

The global load vector $\mathbf{P}$ accumulates all external forces applied to the structure. These forces can originate from nodal point loads, surface tractions (distributed loads), or body forces (like gravity).

In standard finite element assembly, the contributions from elements are transformed into equivalent nodal loads and added into the global vector. When dealing with point loads explicitly applied to specific degrees of freedom (DOFs), we directly manipulate the entries of $\mathbf{P}$.

### 2.1.1 Setting and Adding Loads

The `femlabpy` library provides two primary functions in `loads.py` for manipulating the global load vector: `setload` and `addload`.

*   **`setload(P, dofs, values)`**: Explicitly overrides the current value at the specified DOFs with new values.
*   **`addload(P, dofs, values)`**: Adds new load values to the existing values in the load vector.

```python
# Example: Applying loads in femlabpy
import numpy as np
from femlabpy.loads import setload, addload

P = np.zeros(6)
setload(P, [2], [50.0])
addload(P, [2, 4], [10.0, -25.0])
print("Global Load Vector:", P)
```

---

## 2.2 Dirichlet Boundary Conditions: The Penalty and Direct Elimination Methods

Dirichlet boundary conditions, also known as essential boundary conditions, specify the known values of the primary field variable (e.g., displacements) at certain boundaries:

$$ u_i = \bar{u}_i \quad \forall i \in \mathcal{C} $$

Enforcing this directly in the system $\mathbf{K} \mathbf{u} = \mathbf{P}$ requires modifying the equations. Two highly prominent alternatives are the Penalty Method (Large Spring) and the Direct Elimination Method (with a scaled diagonal). `femlabpy` utilizes a hybrid approach in its `setbc` function.

### 2.2.1 Mathematics of the Large Spring Method in Extreme Detail

The penalty method, fundamentally, attaches an extremely stiff spring to the constrained DOF and applies an enormous force such that the spring extends exactly by the desired displacement $\bar{u}_i$.

Often in textbooks, a penalty parameter $\alpha$ is defined to be significantly larger than the stiffness of the system, e.g., $\alpha = 10^6 \times \max(\mathbf{K})$. 
1. **Modify Stiffness:** Add $\alpha$ to the diagonal entry of $\mathbf{K}$: $K_{ii} \leftarrow K_{ii} + \alpha$
2. **Modify Load:** Set the corresponding load vector entry to $P_i \leftarrow \alpha \times \bar{u}_i$.

Since $\alpha$ is overwhelmingly large, the $i$-th equation effectively becomes $\alpha u_i = \alpha \bar{u}_i \implies u_i \approx \bar{u}_i$.

However, `femlabpy` uses a rigorous direct elimination variant matching the legacy Scilab FemLab implementation. Instead of adding a penalty to the existing stiffness, the entire row and column for DOF $i$ are zeroed out, and a specific "spring stiffness" $k_s$ is assigned to the diagonal, accompanied by transferring coupling forces. The scaling used in `femlabpy` is $0.1 \times \max(|K_{ii}|)$.

### 2.2.2 Exact Python Code for `setbc`

The `setbc` function in `src/femlabpy/boundary.py` implements this flawlessly. Here is the exact Python code responsible for zeroing the rows/columns, updating the RHS, and placing the diagonal stiffness:

```python
    ks = 0.1 * max_abs_diagonal(K)
    if ks == 0.0:
        ks = 1.0

    # ... (DOF parsing omitted for brevity) ...

    for k in range(len(cdofs)):
        j = int(cdofs[k])
        val = cvals[k]
        # Transfer coupling forces to RHS *before* zeroing the column.
        if val != 0.0:
            if sparse:
                col_j = np.asarray(K[:, j].toarray()).ravel()
            else:
                col_j = K[:, j].copy()
            p[:, 0] -= col_j * val
            
        # Zero row and column, set diagonal spring.
        K[j, :] = 0
        K[:, j] = 0
        K[j, j] = ks
        p[j, 0] = ks * val
```

This procedure ensures mathematical exactness. Before the row/column is zeroed, the off-diagonal terms $K_{ij}$ are multiplied by the prescribed displacement `val` and moved to the load vector `p`. Then, the $j$-th equation is entirely replaced with $k_s u_j = k_s \bar{u}_j$.

---

## 2.3 General Constraints: Lagrange Multipliers

While Dirichlet conditions fix individual DOFs, engineering problems often involve multi-point constraints (MPCs), where a linear combination of DOFs must satisfy a condition:

$$ \mathbf{G} \mathbf{u} = \mathbf{Q} $$

where $\mathbf{G}$ is an $m \times n$ constraint matrix and $\mathbf{Q}$ is an $m \times 1$ vector. We use the method of **Lagrange Multipliers**.

### 2.3.1 Building the Saddle-Point Matrix via `np.block`

We introduce a vector of Lagrange multipliers, $\boldsymbol{\lambda}$. The total potential energy functional is augmented to include the constraint, leading to the saddle-point equations:

$$ \begin{bmatrix} \mathbf{K} & \mathbf{G}^T \\ \mathbf{G} & \mathbf{0} \end{bmatrix} \begin{bmatrix} \mathbf{u} \\ \boldsymbol{\lambda} \end{bmatrix} = \begin{bmatrix} \mathbf{P} \\ \mathbf{Q} \end{bmatrix} $$

In `femlabpy`, `solve_lag_general` handles the construction of this augmented block matrix. A scaling factor is applied to $\mathbf{G}$ and $\mathbf{Q}$ to maintain numerical compatibility, then `numpy.block` is utilized to effortlessly stitch the submatrices together for dense arrays:

```python
    Gbar = scale * constraint_matrix
    Qbar = scale * constraint_rhs
    
    # Building the Saddle-Point Matrix using np.block
    Kbar = np.block(
        [
            [as_float_array(K), Gbar.T],
            [
                Gbar,
                np.zeros(
                    (constraint_matrix.shape[0], constraint_matrix.shape[0]),
                    dtype=float,
                ),
            ],
        ]
    )
```

This resulting `Kbar` is the indefinite augmented stiffness matrix, and the augmented load vector `pbar` is obtained via `np.vstack`.

### 2.3.2 Example: 2-Node Spring System

Here is a 50-line runnable Python script demonstrating `solve_lag_general` applied to a simple 2-node spring system where the two nodes are constrained to move exactly together (i.e., $u_1 - u_2 = 0$).

```python
import numpy as np

# A standalone mock-up of the solve_lag_general logic for demonstration
def solve_lag_general(K, p, G, Q):
    scale = 1.0e-2 * np.max(np.abs(np.diag(K)))
    Gbar = scale * G
    Qbar = scale * Q
    
    # 1. Build Saddle-Point Matrix
    Kbar = np.block([
        [K, Gbar.T],
        [Gbar, np.zeros((G.shape[0], G.shape[0]))]
    ])
    
    # 2. Build Augmented RHS
    pbar = np.vstack([p, Qbar])
    
    # 3. Solve System
    augmented = np.linalg.solve(Kbar, pbar)
    u = augmented[:K.shape[0]]
    lagrange = augmented[K.shape[0]:] * scale
    return u, lagrange

def main():
    # 2-node system: Spring 1 connects ground to Node 1, Spring 2 connects Node 1 to Node 2
    # k1 = 1000, k2 = 500
    K = np.array([
        [1500.0, -500.0],
        [-500.0,  500.0]
    ])
    
    # Load applied to Node 2
    p = np.array([[0.0], 
                  [100.0]])
                  
    # Constraint: u_1 - u_2 = 0
    G = np.array([[1.0, -1.0]])
    Q = np.array([[0.0]])
    
    # Solve using Lagrange Multipliers
    u, lagrange = solve_lag_general(K, p, G, Q)
    
    print("Displacements:")
    print(f"u_1 = {u[0,0]:.5f}")
    print(f"u_2 = {u[1,0]:.5f}")
    print("\nConstraint Force (Lagrange Multiplier):")
    print(f"lambda = {lagrange[0,0]:.5f}")

if __name__ == "__main__":
    main()
```

When run, both $u_1$ and $u_2$ evaluate to $0.1$ because the two nodes are constrained to displace by the same amount, making the equivalent stiffness acting on the 100 load equal to $1000$, thus $100 / 1000 = 0.1$. The Lagrange multiplier returns the constraint force transmitted between the nodes.

---

## Summary

Proper handling of boundary conditions dictates the stability and correctness of finite element solutions. `femlabpy` combines the computationally efficient direct elimination with a scaled diagonal for Dirichlet conditions, and the rigorous Lagrange Multiplier formulation for complex multi-point constraints utilizing `np.block` for saddle-point matrix assembly.