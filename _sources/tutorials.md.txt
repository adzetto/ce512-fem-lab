# Tutorials & Usage Manual

This manual explains how to build and solve finite element models using femlabpy. The workflow follows standard finite element procedures. We focus on how to define the model, assemble the matrices, and solve the equations.

## Core Variables

All model data in femlabpy is stored in standard arrays. To use the library effectively, you should understand these five core matrices:

*   **`X` (Node Coordinates):** Each row represents a node. For 2D problems, a row is `[x, y]`. The row index (1-based) is the global node number.
*   **`T` (Topology):** Each row represents an element. It contains the node numbers that make up the element, followed by a material property ID. For a 4-node quad, a row is `[node1, node2, node3, node4, prop_id]`.
*   **`G` (Material Properties):** Each row defines a set of material properties. The `prop_id` in the topology matrix points to a row in `G`.
*   **`C` (Constraints):** Defines prescribed displacements. A row is `[node_id, dof_index, prescribed_value]`.
*   **`P` (Loads):** Defines point loads. A row is `[node_id, dof_index, force_value]`.

---

## 1. Linear Static Analysis

We will model a 2D cantilever beam. The beam is fixed on the left and subjected to a downward point load on the right. 

### Model Generation

First, we generate the nodes `X` and elements `T`. We use the `gmsh` Python API to create a 4.0 by 0.5 meter rectangular mesh.

```python
import gmsh
import numpy as np
import femlabpy as fp

# 1. Create a structured mesh using Gmsh
gmsh.initialize()
gmsh.model.add("beam")
gmsh.model.occ.addRectangle(0.0, 0.0, 0.0, 4.0, 0.5)
gmsh.model.occ.synchronize()
gmsh.model.mesh.setRecombine(2, 1)  # Force quadrilateral elements
gmsh.model.mesh.generate(2)
gmsh.write("beam.msh")
gmsh.finalize()

# 2. Load the mesh into femlabpy
mesh = fp.load_gmsh2("beam.msh")
T = mesh.quads.astype(int)
X = mesh.positions[:, :2]

nn = X.shape[0]  # Total number of nodes
dof = 2          # Degrees of freedom per node (x and y)
```

### Material Properties and Assembly

Next, we define the material and build the global stiffness matrix `K`.

```python
# Material matrix G for 2D elements: [E, nu, type, thickness, density]
# type = 1 means plane stress.
G = np.array([[210e9, 0.3, 1.0, 0.1, 7850.0]])

# Initialize the global stiffness matrix K and load vector p.
# They are sized based on the number of nodes and degrees of freedom.
K, p = fp.init(nn, dof)

# Assemble the stiffness matrix using the 4-node quad driver (kq4e).
K = fp.kq4e(K, T, X, G)
```

### Boundary Conditions and Solving

We fix the left boundary and apply a load to the right boundary. The functions `setload` and `setbc` modify the global matrices in place.

```python
tol = 1e-6

# Find nodes on the left edge (x = 0)
left_nodes = np.where(X[:, 0] < tol)[0] + 1  # +1 for 1-based indexing

# Build the constraint matrix C
C = []
for n in left_nodes:
    C.append([n, 1, 0.0]) # Fix displacement in X (dof 1)
    C.append([n, 2, 0.0]) # Fix displacement in Y (dof 2)
C = np.array(C)

# Find the top-right node and apply a 1000 N downward load
right_nodes = np.where((X[:, 0] > 4.0 - tol) & (X[:, 1] > 0.5 - tol))[0] + 1
P = np.array([[right_nodes[0], 2, -1000.0]])

# Apply loads to vector p
p = fp.setload(p, P, dof)

# Apply boundary conditions to K and p
K_bc, p_bc, _ = fp.setbc(K, p, C, dof)

# Solve the linear equation system (K * u = p)
u = np.linalg.solve(K_bc, p_bc)

print(f"Max vertical displacement: {np.min(u):.6e} meters")
```

### Post-processing

To evaluate internal forces, stresses, and strains, we pass the displacement vector `u` back to the element driver `qq4e`.

```python
# Evaluate internal forces (q), element stresses (S), and strains (E)
q, S, E = fp.qq4e(np.zeros_like(p), T, X, G, u)

# Plot the deformed shape
fp.plotelem(T, X)
```

---

## 2. Dynamic Analysis

femlabpy supports free vibration (modal) and forced vibration (time-history) analysis.

### Modal Analysis

Modal analysis extracts the natural frequencies and mode shapes of the structure. It solves the generalized eigenvalue problem $K \phi = \omega^2 M \phi$.

```python
# Assemble the global mass matrix M using the quad driver (mq4e)
M = np.zeros_like(K)
M = fp.mq4e(M, T, X, G)

# Solve for the first 3 natural modes
# C_bc is required so the solver knows which DOFs are fixed
modal_result = fp.solve_modal(K, M, n_modes=3, C_bc=C, dof=2)

print("Natural frequencies (Hz):", modal_result.freq_hz)
```

### Time-History Analysis (Earthquake)

We can simulate an earthquake by applying a base acceleration to the structure. This is done using the implicit Newmark-beta time integrator.

```python
from femlabpy.dynamics import seismic_load, solve_newmark
from femlabpy.damping import rayleigh_coefficients, rayleigh_damping

# 1. Define the effective seismic load
# The influence vector tells the solver which direction the ground shakes.
# Here, we set 1.0 for all X degrees of freedom, and 0.0 for Y.
inf_vec = np.zeros(nn * dof)
inf_vec[0::2] = 1.0

# accel_data is a 1D numpy array of ground acceleration (m/s^2)
dt = 0.01 
p_eff = seismic_load(M, inf_vec, accel_data, dt)

# 2. Define Rayleigh Damping
# We anchor 5% critical damping to the first two natural frequencies.
f1, f2 = modal_result.freq_hz[0], modal_result.freq_hz[1]
alpha, beta = rayleigh_coefficients(f1, f2, zeta1=0.05, zeta2=0.05)
C_damp = rayleigh_damping(M, K, alpha, beta)

# 3. Set initial conditions (start from rest)
u0 = np.zeros((nn * dof, 1))
v0 = np.zeros((nn * dof, 1))

# 4. Solve the transient equations
nsteps = len(accel_data)
history = solve_newmark(M, C_damp, K, p_eff, u0, v0, dt, nsteps, C_bc=C)

# history.u is a 2D array of shape (nsteps+1, total_dofs)
print("Maximum dynamic roof displacement:", np.max(np.abs(history.u)))
```

---

## 3. Periodic Boundaries (Homogenization)

If you are modeling a repeating unit cell (RVE) of a material, you must enforce periodic boundary conditions. This ensures that the opposite edges of the cell deform in exactly the same way.

femlabpy provides tools to automatically find matching nodes on opposite boundaries and apply the necessary constraint equations.

```python
from femlabpy.periodic import find_periodic_pairs, homogenize

# 1. Identify the nodes on the boundary faces
left_nodes = np.where(X[:, 0] < tol)[0] + 1
right_nodes = np.where(X[:, 0] > 1.0 - tol)[0] + 1
bottom_nodes = np.where(X[:, 1] < tol)[0] + 1
top_nodes = np.where(X[:, 1] > 1.0 - tol)[0] + 1

# 2. Link the opposite boundaries
# This matches nodes by their Y-coordinate (for left/right) 
# and X-coordinate (for bottom/top)
lr_pairs = find_periodic_pairs(X, left_nodes, right_nodes)
bt_pairs = find_periodic_pairs(X, bottom_nodes, top_nodes)
all_pairs = lr_pairs + bt_pairs

# 3. Compute the effective homogenized stiffness matrix
# This function applies macro-strains to the unit cell and averages the resulting stress.
C_eff = homogenize(T, X, G, all_pairs, dof=2)

print("Effective 3x3 Modulus Matrix:")
print(C_eff)
```
