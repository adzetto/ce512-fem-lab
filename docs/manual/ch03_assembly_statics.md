# Chapter 3: Assembly & Constraints

This chapter details the numerical procedures used to assemble global systems of equations and enforce kinematic constraints.

## 3.1 Global Assembly Algorithm

In the finite element method, the global stiffness matrix $\mathbf{K}$ is constructed by summing the contributions of individual element stiffness matrices $\mathbf{K}_e$. Mathematically, this is expressed using Boolean connectivity matrices $\mathbf{L}_e$:

$$ \mathbf{K} = \sum_{e=1}^{nel} \mathbf{L}_e^T \mathbf{K}_e \mathbf{L}_e $$

In `femlabpy`, this assembly is achieved through the driver functions (e.g., `kq4e`, `kt3e`) which internally rely on indexing mechanisms. For example, for a 4-node quad element with 2 DOFs per node, the global DOF indices for the element are computed as:

```python
global_dofs = []
for n in element_nodes:
    global_dofs.extend([2*n-2, 2*n-1]) # Assuming 1-based indexing adjusted for 0-based
```

The $8 \times 8$ element matrix $\mathbf{K}_e$ is then added into the $N_{dof} \times N_{dof}$ global matrix at the intersection of `global_dofs` rows and columns. Similarly, internal element force vectors $\mathbf{q}_e$ are assembled into the global internal force vector $\mathbf{q}$.

### 3.1.1 Manual Element Iteration and `assmk` Assembly

While `femlabpy` abstracts many underlying operations, advanced users may want to build custom element loops. Let's look at exactly how a user might manually iterate through elements and call an assembly routine like `assmk`.

In a conventional FE code, you loop over all elements, retrieve their local topology (node numbers), form the local element stiffness matrix $\mathbf{K}_e$, and then scatter it into the global $\mathbf{K}$. `femlabpy` typically provides a utility `assmk` which performs this scattering. 

```python
import numpy as np

# Assume K_global is already allocated
# edof_matrix is a 2D array where each row represents an element
# and the columns are the global degrees of freedom for that element.

def manual_assembly(K_global, elements, coordinates, edof_matrix):
    """
    Manually iterate over elements and assemble the global stiffness matrix.
    """
    for e in range(len(elements)):
        # 1. Fetch element properties and nodes
        edof = edof_matrix[e, :]  # Global DOFs for this element
        
        # 2. Compute local element stiffness (example placeholder)
        # ke = my_custom_element_stiffness(coordinates, edof)
        ke = np.ones((len(edof), len(edof))) # Placeholder for K_e
        
        # 3. Assemble into the global matrix
        for i in range(len(edof)):
            row = edof[i]
            for j in range(len(edof)):
                col = edof[j]
                K_global[row, col] += ke[i, j]
                
    return K_global
```

### 3.1.2 Dense vs. Sparse Assembly (`scipy.sparse.lil_matrix`)

For small academic problems, creating a dense `numpy.ndarray` for the global stiffness matrix $\mathbf{K}$ is computationally acceptable. However, for industrial-scale models with thousands of DOFs, $\mathbf{K}$ is mostly composed of zeros. This sparsity must be exploited to avoid memory exhaustion and to drastically speed up matrix inversion/factorization. 

We can define a parameter `is_sparse` to decide between a dense matrix or a `scipy.sparse` matrix. The `scipy.sparse.lil_matrix` (List of Lists) format is highly recommended for building sparse matrices because it allows for fast insertion of individual elements.

```python
import scipy.sparse as sp

def create_global_matrix(ndof, is_sparse=True):
    """
    Initialize the global stiffness matrix based on the is_sparse flag.
    """
    if is_sparse:
        # lil_matrix is efficient for incremental construction
        return sp.lil_matrix((ndof, ndof), dtype=np.float64)
    else:
        # Standard dense matrix
        return np.zeros((ndof, ndof), dtype=np.float64)

# Assembly using the is_sparse check
def assmk_advanced(K, ke, edof, is_sparse):
    """
    Assemble element matrix `ke` into global `K` using `edof`.
    """
    # For sparse lil_matrix or dense numpy array, slicing/advanced indexing 
    # might differ slightly in performance, but conceptually it remains:
    if is_sparse:
        # We can use np.ix_ to create an open mesh from multiple boolean masks
        # or integer arrays.
        ix = np.ix_(edof, edof)
        K[ix] = K[ix] + ke
    else:
        # Dense assembly
        ix = np.ix_(edof, edof)
        K[ix] += ke
        
    return K
```

Once assembly is complete, a `lil_matrix` should be converted to a Compressed Sparse Column (`csc_matrix`) or Compressed Sparse Row (`csr_matrix`) format before solving the linear system using solvers like `scipy.sparse.linalg.spsolve`.

```python
# Convert to CSR for fast math operations and solving
K_csr = K_global.tocsr()
```

## 3.2 Dirichlet Boundary Conditions

To solve the equilibrium equations $\mathbf{K} \mathbf{u} = \mathbf{p}$, the global stiffness matrix must be rendered non-singular by preventing rigid body motions. This is achieved by prescribing known displacements (Dirichlet boundary conditions).

`femlabpy` utilizes a direct modification approach (often conceptually similar to the penalty method) to enforce $u_i = \bar{u}_i$. The `setbc` function applies this using a massive artificial stiffness:

1. A very large stiffness value $k_{bc}$ is determined based on the maximum diagonal entry of $\mathbf{K}$:
   $$ k_{bc} = 10^6 \times \max(\text{diag}(\mathbf{K})) $$
2. For each constrained degree of freedom $i$, the corresponding row and column in $\mathbf{K}$ are zeroed out.
3. The diagonal entry $\mathbf{K}_{ii}$ is replaced with $k_{bc}$.
4. The load vector entry $\mathbf{p}_i$ is modified to $k_{bc} \times \bar{u}_i$.

$$
\mathbf{K} = \begin{bmatrix}
\ddots & 0 & \dots \\
0 & k_{bc} & 0 \\
\vdots & 0 & \ddots
\end{bmatrix},
\quad
\mathbf{p} = \begin{bmatrix}
\vdots \\
k_{bc} \bar{u}_i \\
\vdots
\end{bmatrix}
$$

When the system is solved, the equation for row $i$ yields $k_{bc} u_i = k_{bc} \bar{u}_i$, strictly enforcing the constraint.

```python
def enforce_dirichlet(K, p, constrained_dofs, prescribed_values):
    """
    Enforce Dirichlet boundary conditions using the penalty approach.
    Works for both dense and sparse (LIL) matrices.
    """
    # Find the maximum diagonal element
    max_diag = K.diagonal().max()
    k_bc = 1e6 * max_diag

    for dof, val in zip(constrained_dofs, prescribed_values):
        # Zero out the row and column
        K[dof, :] = 0.0
        K[:, dof] = 0.0
        
        # Set diagonal to penalty parameter
        K[dof, dof] = k_bc
        
        # Modify the force vector
        p[dof] = k_bc * val
        
    return K, p
```

## 3.3 General Linear Constraints (Lagrange Multipliers)

For advanced constraints where multiple DOFs are coupled linearly (e.g., rigid links or periodic boundaries), `femlabpy` employs the method of Lagrange multipliers. The constraint equation is defined as:

$$ \mathbf{G} \mathbf{u} = \mathbf{Q} $$

The potential energy functional is augmented with Lagrange multipliers $\lambda$, representing the constraint forces. This leads to an expanded saddle-point system:

$$
\begin{bmatrix}
\mathbf{K} & \mathbf{G}^T \\
\mathbf{G} & \mathbf{0}
\end{bmatrix}
\begin{Bmatrix}
\mathbf{u} \\
\lambda
\end{Bmatrix}
=
\begin{Bmatrix}
\mathbf{p} \\
\mathbf{Q}
\end{Bmatrix}
$$

The `solve_lag_general` function solves this indefinite system directly, yielding both the constrained displacements $\mathbf{u}$ and the precise constraint forces $\lambda$.

### 3.3.1 Python Implementation of Saddle-Point Assembly

In Python, building this augmented matrix can be elegantly handled using NumPy's block matrix construction functions. Specifically, `np.block` allows you to assemble large matrices by explicitly stating their layout in a nested list.

Let's examine how a function like `solve_lag_general` might construct and solve this system:

```python
import numpy as np

def solve_lag_general(K, p, G, Q):
    """
    Solve the augmented system [K  G.T; G  0] * [u; lambda] = [p; Q].
    
    Parameters:
    - K: Global stiffness matrix (N x N)
    - p: Load vector (N)
    - G: Constraint matrix (M x N)
    - Q: Constraint right-hand side (M)
    
    Returns:
    - u: Displacements
    - lam: Lagrange multipliers (constraint forces)
    """
    N = K.shape[0]
    M = G.shape[0]
    
    # 1. Create the zero block for the bottom right
    zero_block = np.zeros((M, M))
    
    # 2. Assemble the block matrix for the augmented stiffness
    # Using np.block provides a visually clear way to concatenate arrays
    K_aug = np.block([
        [K, G.T],
        [G, zero_block]
    ])
    
    # 3. Assemble the augmented force vector
    p_aug = np.concatenate([p, Q])
    
    # 4. Solve the expanded system
    # Since the saddle point matrix is indefinite, standard Cholesky 
    # factorization fails. We rely on a general solver like np.linalg.solve.
    sol = np.linalg.solve(K_aug, p_aug)
    
    # 5. Extract results
    u = sol[:N]      # First N entries are displacements
    lam = sol[N:]    # Last M entries are Lagrange multipliers
    
    return u, lam
```

`np.block` dynamically calculates the required shapes and allocates memory for the massive `K_aug` block matrix. This approach shines when managing dense matrices. If working entirely in a sparse ecosystem, `scipy.sparse.bmat` acts as the direct sparse equivalent to `np.block`.

## 3.4 Full Runnable Example: 1D Spring Assembly

To tie all these concepts together, here is a complete, runnable Python script that simulates a 3-node, 2-element 1D spring system. We will manually assemble the system, apply boundary conditions via Lagrange multipliers (tying node 1 to a wall and coupling nodes 2 and 3), and solve.

```python
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

# --- Problem Definition ---
# Nodes:    (0) --- [k1] --- (1) --- [k2] --- (2)
# We want to constrain Node 0 to have 0 displacement (Wall).
# We apply a force F at Node 1.
# We also want to link Node 1 and Node 2 such that u1 = u2 (rigid link).

def run_spring_simulation():
    print("Starting 1D Spring Assembly Simulation...")
    
    # --- System Parameters ---
    ndof = 3                 # Three nodes, 1 DOF per node
    k1, k2 = 1000.0, 500.0   # Spring stiffness values
    F = 150.0                # Force applied at node 1
    
    # 1. Initialization
    is_sparse = True
    if is_sparse:
        K = sp.lil_matrix((ndof, ndof))
    else:
        K = np.zeros((ndof, ndof))
        
    p = np.zeros(ndof)
    p[1] = F  # Apply force to DOF 1
    
    # 2. Element Definitions
    # Element 1: connects DOF 0 and DOF 1
    ke1 = np.array([[ k1, -k1],
                    [-k1,  k1]])
    edof1 = [0, 1]
    
    # Element 2: connects DOF 1 and DOF 2
    ke2 = np.array([[ k2, -k2],
                    [-k2,  k2]])
    edof2 = [1, 2]
    
    # 3. Manual Assembly
    print("Assembling global matrix...")
    ix1 = np.ix_(edof1, edof1)
    K[ix1] = K[ix1] + ke1
    
    ix2 = np.ix_(edof2, edof2)
    K[ix2] = K[ix2] + ke2
    
    # Convert LIL to dense to use with our dense np.block solver below
    K_dense = K.toarray() if is_sparse else K
    
    # 4. Applying Constraints via Lagrange Multipliers
    # Constraint 1: u0 = 0
    # Constraint 2: u1 - u2 = 0
    print("Applying constraints (u0 = 0, u1 = u2)...")
    
    G = np.array([
        [1.0,  0.0,  0.0],  # 1*u0 + 0*u1 + 0*u2 = 0
        [0.0,  1.0, -1.0]   # 0*u0 + 1*u1 - 1*u2 = 0
    ])
    
    Q = np.array([0.0, 0.0]) # RHS of constraints
    
    # 5. Solve using np.block technique
    M = G.shape[0]
    zero_block = np.zeros((M, M))
    
    K_aug = np.block([
        [K_dense, G.T],
        [G, zero_block]
    ])
    
    p_aug = np.concatenate([p, Q])
    
    print("\nAugmented System Matrix [K_aug]:")
    print(K_aug)
    
    # 6. Solution Extraction
    sol = np.linalg.solve(K_aug, p_aug)
    u = sol[:ndof]
    lam = sol[ndof:]
    
    print("\nResults:")
    print(f"Displacements (u): {u}")
    print(f"Lagrange Multipliers (forces): {lam}")
    
    # Optional Verification
    print(f"Constraint verification (G*u - Q): {G @ u - Q}")

if __name__ == "__main__":
    run_spring_simulation()
```

This full script demonstrates matrix allocation, sparse-aware element insertion `ix_()`, saddle-point construction utilizing `np.block`, and the solver step, encapsulating all concepts presented in this chapter.