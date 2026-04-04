# Chapter 7: Advanced Workflows & Examples

This chapter provides complete, runnable Python scripts demonstrating how to solve highly nonlinear and dynamic problems using `femlabpy`. These examples highlight the raw array-based flexibility of the library.

## 7.1 Geometrically Nonlinear Truss (Snap-Through)

A classical problem in structural mechanics is the snap-through buckling of a shallow truss arch. This problem demonstrates large deformations where the stiffness matrix changes as the geometry updates. 

In `femlabpy`, the nonlinear bar element (`kebar` and `qebar`) calculates both the material and geometric stiffness matrices using the Green-Lagrange strain measure.

### The 2-Bar Dome Script

This script loads a pre-packaged 2-bar dome, sets up an orthogonal residual method (arc-length-like solver) via `nlbar`, and tracks the load-displacement path.

```python
import numpy as np
import matplotlib.pyplot as plt
import femlabpy as fp

# 1. Load the benchmark data
# The dome consists of a central node pulled downward, supported by two pinned bases.
case = fp.bar01()
T = case["T"]  # Topology: [n1, n2, prop_id]
X = case["X"]  # Node coordinates: [x, y]
G = case["G"]  # Properties: [Area, Young's Modulus]
C = case["C"]  # Boundary conditions (Pinned supports)
P = case["P"]  # Nominal load vector (Downward force)

# 2. Nonlinear Solver Parameters
no_loadsteps = 20    # Number of load increments
i_max = 50           # Max Newton iterations per step
tol = 1e-6           # Convergence tolerance for residual norm

# 3. Solve using Orthogonal Residual Method
# This method can track the post-buckling path (snap-through) where the load decreases.
result = fp.nlbar(T, X, G, C, P, no_loadsteps, i_max, tol)

# 4. Extract Load-Displacement Path
# U_path contains the vertical displacement of the central node.
# F_path contains the applied load factor at that step.
u_tip = result["U_path"].ravel()
f_load = result["F_path"].ravel() * P[0, 2] # Multiply factor by nominal load

# 5. Plot the Snap-Through Curve
plt.figure(figsize=(8, 5))
plt.plot(u_tip, f_load, 'b-o', linewidth=2, markersize=6)
plt.axhline(0, color='black', linewidth=1)
plt.title("2-Bar Truss Snap-Through Buckling")
plt.xlabel("Vertical Displacement at Tip (m)")
plt.ylabel("Applied Load (N)")
plt.grid(True, alpha=0.3)
plt.show()
```

## 7.2 Wave Propagation in a 1D Bar (Explicit Dynamics)

For high-speed impact or stress-wave propagation, implicit integrators like Newmark-$\beta$ smooth out the sharp shock fronts due to numerical dissipation. In such cases, the **Explicit Central Difference Method** combined with a lumped (diagonal) mass matrix is vastly superior.

This script models a long steel bar struck by a short rectangular force pulse at its left end, sending a compressive stress wave down the bar.

```python
import numpy as np
import matplotlib.pyplot as plt
import femlabpy as fp
from femlabpy.dynamics import solve_central_diff, pulse_load

# 1. Geometry and Mesh Parameters
L = 10.0           # Length of the bar (m)
nel = 200          # Number of elements
nn = nel + 1       # Number of nodes
dof = 1            # 1D problem (ux only)

# 2. Create nodes (X) and elements (T)
X = np.linspace(0, L, nn).reshape(-1, 1)
T = np.zeros((nel, 3), dtype=int)
for i in range(nel):
    T[i] = [i + 1, i + 2, 1]  # 1-based indexing for nodes, prop_id=1

# 3. Material properties (Steel)
E = 210e9          # Young's modulus (Pa)
rho = 7850.0       # Density (kg/m^3)
A = 0.01           # Cross-sectional area (m^2)
G = np.array([[A, E, rho]])

# Wave speed (c = sqrt(E/rho))
c_wave = np.sqrt(E / rho)
dt_critical = (L / nel) / c_wave  # Courant-Friedrichs-Lewy (CFL) limit

# 4. Assemble K and Lumped M
K, p = fp.init(nn, dof, use_sparse=True)
M = fp.init(nn, dof, use_sparse=True)[0]

# Important: Use lumped=True for explicit methods
K = fp.kbar(K, T, X, G, u=np.zeros(nn*dof))
M = fp.mbar(M, T, X, G, lumped=True)

# 5. Boundary Conditions (Free-Free bar, no constraints)
C_bc = np.array([], dtype=float).reshape(0, 3)

# 6. Apply a short 10 kN Impact Pulse at the left end (Node 1)
P_impact = np.zeros((nn * dof, 1))
P_impact[0] = -10000.0  # Compressive force

# The pulse lasts for exactly 0.5 milliseconds
impact_func = pulse_load(P_impact, t_start=0.0, t_duration=0.0005)

# 7. Time Integration using Central Difference
dt = dt_critical * 0.9  # Operate at 90% of the stability limit
nsteps = int(0.003 / dt) # Simulate 3 milliseconds total

u0 = np.zeros((nn * dof, 1))
v0 = np.zeros((nn * dof, 1))
C_damp = None # Pure undamped wave propagation

result = solve_central_diff(
    M, C_damp, K, impact_func, u0, v0, dt, nsteps, C_bc=C_bc, dof=dof
)

# 8. Post-processing: Plot the wave snapshot at T = 2.0 ms
# At 2ms, the wave should have traveled distance d = c * 2ms.
step_2ms = int(0.002 / dt)
displacements = result.u[step_2ms, :]

# Compute strains using qebar driver
q, axial_force, strain = fp.qbar(np.zeros_like(P_impact), T, X, G, displacements)

# Plotting the strain wave
element_centers = X[:-1] + (X[1] - X[0])/2.0
plt.figure(figsize=(8, 4))
plt.plot(element_centers, strain.ravel(), 'r-', linewidth=2)
plt.title(f"Compressive Strain Wave at t = {step_2ms * dt * 1000:.2f} ms")
plt.xlabel("Position along bar (m)")
plt.ylabel("Axial Strain")
plt.grid(True, alpha=0.3)
plt.show()
```

These two examples highlight the core strength of `femlabpy`: there is no "black box." You extract exactly the matrices you need (`K`, `M`), pass them directly into a mathematically transparent solver (`nlbar` or `solve_central_diff`), and plot the raw numerical arrays.