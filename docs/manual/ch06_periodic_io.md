# Chapter 6: Periodic Boundaries & I/O

Advanced finite element analyses often require specialized boundary condition enforcement and integration with industrial meshing tools.

## 6.1 Periodic Boundary Conditions (Homogenization)

When simulating the mechanical response of a composite material, we typically analyze a Representative Volume Element (RVE). To ensure that the micro-scale deformations represent a continuous macro-scale material, Periodic Boundary Conditions (PBCs) must be applied. 

PBCs enforce that the deformation on opposite faces of the RVE are identical, offset only by the applied macroscopic strain tensor $\bar{\mathbf{\epsilon}}$. For two corresponding nodes $A^+$ (on the positive face) and $A^-$ (on the negative face), the displacement mapping is:

$$ \mathbf{u}^+ - \mathbf{u}^- = \bar{\mathbf{\epsilon}} \Delta \mathbf{x} $$

where $\Delta \mathbf{x} = \mathbf{x}^+ - \mathbf{x}^-$ is the physical distance vector between the paired nodes. This is mathematically imposed as a set of linear constraint equations:

$$ \mathbf{G} \mathbf{u} = \mathbf{Q} $$

### Implementation in femlabpy

1. **Pairing Nodes:** The `find_periodic_pairs` function scans two arrays of node indices and links nodes that have matching coordinates along the boundary planes (within a numerical tolerance).
2. **Generating Constraints:** The `periodic_constraints` function converts these node pairs into linear constraint equations, assembling the $\mathbf{G}$ and $\mathbf{Q}$ matrices for the Lagrange multiplier solver.
3. **Homogenization Driver:** The `homogenize` function automates the entire process. It applies three independent macroscopic strain states (pure X-tension, pure Y-tension, and pure shear) to the RVE, solves the constrained equations using `solve_lag_general`, and computes the volume-averaged stress tensor $\langle \sigma \rangle$ for each state. The resultant vectors form the effective $3 \times 3$ elasticity matrix $\mathbf{C}_{eff}$.

### Complete Code: Homogenizing a Porous Unit Cell

This script generates a $1 \times 1$ square unit cell with a central hole, applies periodic boundary conditions to the Left/Right and Top/Bottom faces, and calculates the effective macroscopic stiffness matrix $\mathbf{C}_{eff}$.

```python
import gmsh
import numpy as np
import matplotlib.pyplot as plt
import femlabpy as fp
from femlabpy.periodic import find_periodic_pairs, homogenize

# 1. Generate the RVE Mesh using Gmsh (A square with a hole)
gmsh.initialize()
gmsh.model.add("unit_cell")

L = 1.0  # Side length
R = 0.2  # Hole radius

# Create the square
gmsh.model.occ.addRectangle(0, 0, 0, L, L, tag=1)
# Create the hole
gmsh.model.occ.addDisk(L/2, L/2, 0, R, R, tag=2)
# Cut the hole from the square
gmsh.model.occ.cut([(2, 1)], [(2, 2)])
gmsh.model.occ.synchronize()

# Force structured meshing on boundaries to ensure nodes align perfectly
gmsh.model.mesh.setRecombine(2, 1)
gmsh.model.mesh.generate(2)
gmsh.write("rve.msh")
gmsh.finalize()

# 2. Load Mesh into femlabpy
mesh = fp.load_gmsh2("rve.msh")
T = mesh.quads.astype(int)
X = mesh.positions[:, :2]

# 3. Material Properties (Base Material)
E = 210e9     # Young's Modulus (Pa)
nu = 0.3      # Poisson's Ratio
t = 1.0       # Thickness
# G = [E, nu, plane_stress(1)/strain(2), thickness]
G = np.array([[E, nu, 1, t]])

# 4. Identify Boundary Nodes
tol = 1e-5
left_nodes   = np.where(X[:, 0] < tol)[0] + 1
right_nodes  = np.where(X[:, 0] > L - tol)[0] + 1
bottom_nodes = np.where(X[:, 1] < tol)[0] + 1
top_nodes    = np.where(X[:, 1] > L - tol)[0] + 1

# 5. Link Opposite Boundaries (Periodic Pairs)
# This matches nodes by their Y-coordinate (for left/right) 
# and X-coordinate (for bottom/top)
lr_pairs = find_periodic_pairs(X, left_nodes, right_nodes, tol=tol)
bt_pairs = find_periodic_pairs(X, bottom_nodes, top_nodes, tol=tol)
all_pairs = lr_pairs + bt_pairs

# 6. Execute Homogenization
# The homogenize function automatically applies the 3 macro-strain cases,
# solves the saddle-point systems via Lagrange multipliers, and 
# calculates the volume-averaged stresses.
C_eff = homogenize(T, X, G, all_pairs, dof=2)

print("--- EFFECTIVE HOMOGENIZED STIFFNESS MATRIX (Pa) ---")
print(np.array2string(C_eff, formatter={'float_kind':lambda x: f"{x:.2e}"}))
```

When you print the matrix, you'll see a $3 \times 3$ orthotropic stiffness tensor representing the macroscopic behavior of the porous material!

## 6.2 Gmsh I/O Integration

`femlabpy` interfaces seamlessly with the open-source mesh generator [Gmsh](https://gmsh.info/). The `io.gmsh` module reads `.msh` files (versions 2.2 and 4.1) natively without requiring complex external dependencies.

The `load_gmsh2(filepath)` function parses the ASCII mesh block and returns a `GmshMesh` dataclass containing:
- `positions`: An $N \times 3$ array of spatial coordinates.
- `triangles`: An $E \times 4$ array of CST topologies.
- `quads`: An $E \times 5$ array of Q4 topologies.

This native array extraction bridges the gap between complex 3D CAD modeling and the flat-array `femlabpy` drivers instantly.

### Extracting Element Properties
You can assign Physical Groups in Gmsh (e.g., `gmsh.model.addPhysicalGroup(2, [surf1], tag=1)`), and `load_gmsh2` will automatically map these tags to the last column of the topology matrix `T` (`prop_id`). This means you can mesh a multi-material composite in Gmsh and `femlabpy` will naturally pull the correct rows from the `G` matrix during assembly.
