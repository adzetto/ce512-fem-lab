# Chapter 6: Periodic Boundaries & I/O

Advanced finite element analyses often require specialized boundary condition enforcement and integration with industrial meshing tools. In composite material modeling and computational micromechanics, these techniques are essential for capturing accurate physical responses across different scales.

## 6.1 Periodic Boundary Conditions (Homogenization)

When simulating the mechanical response of a heterogeneous or composite material, we typically analyze a Representative Volume Element (RVE). An RVE is a small sub-volume that statistically represents the entire macroscopic material. To ensure that the micro-scale deformations within this unit cell seamlessly represent a continuous macro-scale material without boundary effects, Periodic Boundary Conditions (PBCs) must be applied.

PBCs enforce that the deformation on opposite faces of the RVE are identical, offset only by the applied macroscopic strain tensor $\bar{\mathbf{\epsilon}}$. For two corresponding nodes $A^+$ (on the positive face) and $A^-$ (on the negative face), the displacement mapping is mathematically expressed as:

$$ \mathbf{u}^+ - \mathbf{u}^- = \bar{\mathbf{\epsilon}} \Delta \mathbf{x} $$

where $\Delta \mathbf{x} = \mathbf{x}^+ - \mathbf{x}^-$ is the physical distance vector between the paired nodes in the undeformed configuration. This relationship ensures that when RVEs are tiled infinitely in space, the material does not overlap or open gaps at the boundaries.

In a finite element framework, this is mathematically imposed as a set of linear constraint equations:

$$ \mathbf{G} \mathbf{u} = \mathbf{Q} $$

where $\mathbf{G}$ is the constraint matrix mapping the degrees of freedom, and $\mathbf{Q}$ is the constraint vector that introduces the macroscopic strain effect.

### Understanding the Python Logic in `femlabpy`

The implementation of PBCs in `femlabpy` relies on a few critical functions that handle the geometry matching, constraint generation, and solution process.

#### 1. Node Pairing Logic: `find_periodic_pairs`
The first step is identifying which nodes on opposite boundaries correspond to each other. The `find_periodic_pairs(X, left_nodes, right_nodes, tol)` function accomplishes this. 

**How it works under the hood:**
The function uses coordinate matching to pair nodes. If we are matching the Left boundary (where $X=0$) to the Right boundary (where $X=L$), the corresponding nodes must have the exact same Y-coordinate (and Z-coordinate in 3D). 

```python
# Conceptual logic inside find_periodic_pairs
pairs = []
for n_left in left_nodes:
    y_left = X[n_left - 1, 1]
    for n_right in right_nodes:
        y_right = X[n_right - 1, 1]
        
        # We use np.isclose instead of exact equality '==' 
        # to avoid floating-point precision errors
        if np.isclose(y_left, y_right, atol=tol):
            pairs.append((n_left, n_right))
            break
```

The use of `numpy.isclose` is critical here. Meshing software (like Gmsh) outputs coordinates as floating-point numbers. Due to machine precision, a node on the right face might have a Y-coordinate of `0.5000000000000001` while the left face has `0.4999999999999999`. A strict equality check `y_left == y_right` would fail to match these nodes, breaking the periodic boundary. By using `np.isclose` with a defined absolute tolerance (e.g., `tol=1e-5`), the function robustly identifies pairs even with slight numerical noise.

#### 2. Generating Constraints: `periodic_constraints`
Once the pairs are established, `periodic_constraints(X, all_pairs, macro_strain)` converts these node tuples into the global $\mathbf{G}$ matrix and $\mathbf{Q}$ vector. For every pair, it adds entries $+1$ and $-1$ in the appropriate columns of $\mathbf{G}$ for the paired DOFs, and computes $\bar{\mathbf{\epsilon}} \Delta \mathbf{x}$ to populate $\mathbf{Q}$.

#### 3. Homogenization Driver: `homogenize`
The `homogenize` function wraps these steps. It computes the effective stiffness $\mathbf{C}_{eff}$ by running three independent load cases (pure X-tension, pure Y-tension, and pure shear). For each case, it applies the strain, solves the saddle-point system via `solve_lag_general` (using Lagrange multipliers to enforce the $\mathbf{G}$ constraints), extracts the resulting stress field, and computes the volume-averaged stress $\langle \sigma \rangle$. 

---

### Complete Code: Homogenizing a Porous Unit Cell & Micro-Stress Extraction

This script generates a $1 \times 1$ square unit cell with a central hole, applies PBCs to the Left/Right and Top/Bottom faces, calculates the effective macroscopic stiffness matrix $\mathbf{C}_{eff}$, and finally extracts and plots the micro-stress field over the RVE.

```python
import gmsh
import numpy as np
import matplotlib.pyplot as plt
import femlabpy as fp
from femlabpy.periodic import find_periodic_pairs, homogenize

# ==========================================
# 1. Generate the RVE Mesh using Gmsh
# ==========================================
gmsh.initialize()
gmsh.model.add("unit_cell")

L = 1.0  # Side length
R = 0.2  # Hole radius

# Create the square (tag 1) and hole (tag 2)
gmsh.model.occ.addRectangle(0, 0, 0, L, L, tag=1)
gmsh.model.occ.addDisk(L/2, L/2, 0, R, R, tag=2)
gmsh.model.occ.cut([(2, 1)], [(2, 2)])
gmsh.model.occ.synchronize()

# Force structured meshing on boundaries to ensure perfectly aligned nodes
gmsh.model.mesh.setRecombine(2, 1)
gmsh.model.mesh.generate(2)
gmsh.write("rve.msh")
gmsh.finalize()

# ==========================================
# 2. Load Mesh into femlabpy
# ==========================================
mesh = fp.load_gmsh2("rve.msh")
T = mesh.quads.astype(int)
X = mesh.positions[:, :2]

# ==========================================
# 3. Material Properties (Base Material)
# ==========================================
E = 210e9     # Young's Modulus (Pa)
nu = 0.3      # Poisson's Ratio
t = 1.0       # Thickness
# G format: [E, nu, plane_stress(1)/strain(2), thickness]
G = np.array([[E, nu, 1, t]])

# ==========================================
# 4. Identify Boundary Nodes
# ==========================================
tol = 1e-5
left_nodes   = np.where(X[:, 0] < tol)[0] + 1
right_nodes  = np.where(X[:, 0] > L - tol)[0] + 1
bottom_nodes = np.where(X[:, 1] < tol)[0] + 1
top_nodes    = np.where(X[:, 1] > L - tol)[0] + 1

# ==========================================
# 5. Link Opposite Boundaries (Periodic Pairs)
# ==========================================
lr_pairs = find_periodic_pairs(X, left_nodes, right_nodes, tol=tol)
bt_pairs = find_periodic_pairs(X, bottom_nodes, top_nodes, tol=tol)
all_pairs = lr_pairs + bt_pairs

# ==========================================
# 6. Execute Homogenization
# ==========================================
C_eff = homogenize(T, X, G, all_pairs, dof=2)

print("--- EFFECTIVE HOMOGENIZED STIFFNESS MATRIX (Pa) ---")
print(np.array2string(C_eff, formatter={'float_kind':lambda x: f"{x:.2e}"}))

# ==========================================
# 7. Extracting & Plotting the Micro-Stress Field
# ==========================================
# While C_eff gives the macro response, we often want to inspect the 
# internal stress distribution under a specific loading state.

# Define a pure macroscopic X-tension strain state: [exx, eyy, exy]
macro_strain = np.array([0.01, 0.0, 0.0])

# Generate constraint matrices for this specific strain
G_mat, Q_vec = fp.periodic_constraints(X, all_pairs, macro_strain)

# Assemble global stiffness matrix
K = fp.kasm(T, X, G, dof=2)

# Solve the constrained system using Lagrange multipliers
F_ext = np.zeros(K.shape[0])
U, _ = fp.solve_lag_general(K, F_ext, G_mat, Q_vec)

# Extract element-level stresses using qq4e (Quadrilateral 4-node Stress/Strain)
stress, strain = fp.qq4e(T, X, U, G)

# Plot the sigma_xx component of the micro-stress field
plt.figure(figsize=(8, 6))
# stress[:, 0] corresponds to the XX stress component
fp.plot_trimesh(T, X, stress[:, 0], title="Micro-Stress Field ($\sigma_{xx}$) under X-Tension")
plt.show()
```

When you run this script, you not only get the homogenized macroscopic behavior of the porous unit cell, but you can visually inspect the stress concentrations around the central hole on the micro-scale!

---

## 6.2 Gmsh I/O Integration

`femlabpy` interfaces seamlessly with the open-source mesh generator [Gmsh](https://gmsh.info/). The `io.gmsh` module reads `.msh` files (versions 2.2 and 4.1) natively without requiring complex external dependencies. This acts as the crucial bridge between complex CAD geometry and raw finite element array operations.

### How `load_gmsh2` Extracts Geometry and Topology

The `load_gmsh2(filepath)` function is a dedicated parser for the ASCII `.msh` format. 

**1. Extracting Node Positions:**
The parser scans the file for the `$Nodes` block. Gmsh stores nodes with their global IDs and their X, Y, Z coordinates. `load_gmsh2` processes this block, strips out the IDs, and populates the `mesh.positions` array. This results in a dense $N \times 3$ NumPy array, where the row index corresponds to the node ID minus one.

**2. Extracting Elements and Mapping Tags:**
Next, it looks for the `$Elements` block. This is where element topologies (connectivity) and Physical Group tags reside.
- If you assigned a surface as a Physical Group in Gmsh (e.g., `gmsh.model.addPhysicalGroup(2, [surf1], tag=1)`), this `tag=1` is written into the element data.
- `load_gmsh2` sorts the elements by type (triangles vs. quadrilaterals). 
- Crucially, it appends the Gmsh Physical Tag as the **last column** of the output topology arrays. 

This means that if you have an $E \times 4$ array for `triangles` (CST elements), the first 3 columns are the node indices $[n_1, n_2, n_3]$, and the 4th column is the `prop_id` (the physical tag from Gmsh). For `quads` (Q4 elements), it is an $E \times 5$ array, with the 5th column holding the property ID.

### Multi-Material Workflows
Because the physical tags are inherently mapped to the last column of the topology matrix `T`, configuring multi-material composites is incredibly simple. You can mesh a domain with multiple regions in Gmsh, assign them different Physical Groups (e.g., tag 1 for matrix, tag 2 for fiber). In `femlabpy`, you simply define a material property matrix `G` with two rows. During assembly (like calling `fp.kasm(T, X, G)`), the solver automatically looks at that last column of `T` to fetch the corresponding material properties from `G` for each element!
