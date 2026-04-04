# Chapter 9: Advanced Dynamic Workflows

This chapter explores deeper, specialized workflows for structural dynamics, providing highly detailed line-by-line Python code breakdowns for extracting building periods, plotting mode shapes, generating Frequency Response Functions (FRF), and performing comprehensive Time History Analyses (THA) under seismic loading.

## 9.1 Modal Analysis: Finding Periods and Plotting Mode Shapes

The structural period $T$ (in seconds) is the time it takes for a building to complete one full cycle of free vibration in a particular mode shape. The period is simply the inverse of the natural frequency: $T = 1 / f_n$.

`femlabpy` abstracts the eigenvalue extraction process in the `solve_modal` function. Behind the scenes, it condenses out the fixed Boundary Conditions (using the exact `C` array) and calls `scipy.linalg.eigh` (for dense matrices) or `scipy.sparse.linalg.eigsh` (for sparse matrices using Shift-and-Invert Lanczos iterations).

### Deep Dive: How `solve_modal` Condenses Matrices using `np.ix_`

In structural dynamics, degrees of freedom (DOFs) that are fully restrained (fixed supports) have exactly zero displacement. Therefore, their corresponding rows and columns in the Stiffness ($K$) and Mass ($M$) matrices do not participate in the generalized eigenvalue problem $(K - \omega^2 M) \Phi = 0$. 

To prevent singular matrices and compute the true dynamic modes of the *unrestrained* structure, `solve_modal` drops these constrained rows and columns. It does this by:
1. Identifying all global DOFs.
2. Extracting the constrained DOFs from the `C_bc` array.
3. Finding the "free" DOFs using `np.setdiff1d(all_dofs, constrained_dofs)`.
4. Using NumPy's multi-dimensional meshgrid indexer, `np.ix_`, to extract the sub-matrices.

```python
# Internal mechanic of solve_modal
free_dofs = np.setdiff1d(np.arange(total_dof), constrained_dofs)
K_free = K[np.ix_(free_dofs, free_dofs)]
M_free = M[np.ix_(free_dofs, free_dofs)]
```
By utilizing `np.ix_(free_dofs, free_dofs)`, NumPy efficiently grabs the intersection of the "free" rows and "free" columns without needing a slow `for` loop. The eigenvalue solver then evaluates `K_free` and `M_free`. After solving, the eigenvectors are padded with zeros at the constrained DOFs to map back to the global system.

### Complete Code: 10-Story Shear Building Modal Analysis

Assume we have a simple column or building mesh. We want to find its fundamental periods and visualize its first 3 bending modes.

```{code-block} python
import numpy as np
import matplotlib.pyplot as plt
import femlabpy as fp
from femlabpy.modal import solve_modal

# 1. Geometry and Mesh Generation (10m high column, 1m wide)
L, H = 1.0, 10.0
nx, ny = 2, 20
nn = (nx + 1) * (ny + 1)
dof = 2

# Generate a basic structured Q4 mesh grid manually or via Gmsh
# (For brevity, we assume X and T are generated here)
# X = ... (shape: nn x 2)
# T = ... (shape: nel x 5)

# 2. Material (Concrete: E=30GPa, nu=0.2, t=1.0m, rho=2500kg/m3)
G = np.array([[30e9, 0.2, 1.0, 1.0, 2500.0]])

# 3. Assemble Stiffness and Mass Matrices
K, p = fp.init(nn, dof)
M = fp.init(nn, dof)[0]

K = fp.kq4e(K, T, X, G)
M = fp.mq4e(M, T, X, G, lumped=False) # Consistent mass matrix

# 4. Boundary Conditions (Fixed Base at y=0)
fixed_nodes = np.where(np.abs(X[:, 1]) < 1e-6)[0] + 1
C = []
for n in fixed_nodes:
    C.append([n, 1, 0.0]) # Fix Ux
    C.append([n, 2, 0.0]) # Fix Uy
C = np.array(C)

# 5. Execute Modal Solver
# Requesting the first 5 modes
result = solve_modal(K, M, n_modes=5, C_bc=C, dof=2)

print("--- MODAL ANALYSIS RESULTS ---")
for i in range(5):
    print(f"Mode {i+1}:")
    print(f"  Frequency = {result.freq_hz[i]:.3f} Hz")
    print(f"  Period    = {result.period[i]:.4f} s")
    print(f"  Eff. Mass (X) = {result.effective_mass[i, 0]:.1f} kg")

# 6. Plotting Mode Shapes
scale_factor = 2.0
fig, axes = plt.subplots(1, 3, figsize=(12, 6))
for i in range(3):
    ax = axes[i]
    phi_i = result.mode_shapes[:, i].reshape(-1, 2)
    X_def = X + phi_i * scale_factor
    for elem in T:
        n_idx = elem[:4] - 1
        poly = np.append(n_idx, n_idx[0])
        ax.plot(X[poly, 0], X[poly, 1], 'k-', alpha=0.1)
        ax.plot(X_def[poly, 0], X_def[poly, 1], 'b-', linewidth=1.5)
    ax.set_title(f"Mode {i+1}: {result.freq_hz[i]:.2f} Hz\n(T = {result.period[i]:.2f} s)")
    ax.axis('equal')
    ax.axis('off')

plt.tight_layout()
plt.show()
```

## 9.2 Frequency Response Functions (FRF)

The **Frequency Response Function**, $H(\omega)$, measures how the structure responds to a harmonic excitation $P \sin(\omega t)$ at varying frequencies $\omega$. Resonance occurs when the excitation frequency matches one of the natural frequencies, causing a spike in the FRF magnitude.

$$ H(\omega) = \left( -\omega^2 \mathbf{M} + i\omega \mathbf{C} + \mathbf{K} \right)^{-1} $$

`femlabpy` provides `compute_frf` and `plot_frf` to automate the calculation of the complex Dynamic Stiffness matrix and extract the steady-state transfer function between an input DOF and an output DOF.

## 9.3 Full Time-History Seismic Analysis

When given an earthquake record (e.g., an `.AT2` file containing acceleration samples), a standard Response Spectrum analysis is often insufficient for non-linear assessments. A complete Time-History Analysis (THA) is required.

In THA, the ground acceleration $\ddot{u}_g(t)$ is transformed into an effective dynamic point load applied to every mass in the structure:

$$ \mathbf{P}_{eff}(t) = -\mathbf{M} \mathbf{r} \ddot{u}_g(t) $$

where $\mathbf{r}$ is the influence vector (1 for DOFs in the shaking direction, 0 otherwise).

### Deep Dive: How `seismic_load` Handles Interpolation and Sparse Matrices

The `seismic_load` function inside `femlabpy` is a factory function. It takes your earthquake record array, your mass matrix, and your spatial influence vector, and returns a Python *callable* function `p(t)` that the Newmark-Beta solver can invoke at any arbitrary time $t$. 

1. **Pre-computing the Load Vector Profile:** To avoid multiplying the massive Mass matrix $\mathbf{M}$ at every single time step, `seismic_load` pre-computes the product $-\mathbf{M} \mathbf{r}$ once. If $\mathbf{M}$ is a `scipy.sparse` matrix (like CSR or CSC), it leverages sparse matrix-vector multiplication (`M.dot(r)`), ensuring memory and computational efficiency.
2. **The `p(t)` Callable:** The returned function `def p(t):` encapsulates the state.
3. **`np.interp` for Sub-stepping:** The Newmark-Beta solver might adapt its time step internally, evaluating forces at times $t$ that do not fall exactly on the earthquake record's discrete points (`dt_record`). To handle this, `p(t)` uses `np.interp(t, time_array, ag_ms2)`. This linearly interpolates the ground acceleration at any exact float time $t$ by finding the two nearest discrete points in the record, effectively producing a continuous acceleration curve.

### Step-by-Step Earthquake Integration

```{code-block} python
import numpy as np
import matplotlib.pyplot as plt
from femlabpy.dynamics import seismic_load, solve_newmark

# 1. Load an Earthquake Record
dt_record = 0.01
n_points = 1000
time_array = np.arange(n_points) * dt_record
ag_g = 0.5 * np.sin(2.0 * np.pi * 2.5 * time_array) * np.exp(-0.2 * time_array)
ag_ms2 = ag_g * 9.80665

# 2. Build the Influence Vector for Horizontal (X) Shaking
inf_vec = np.zeros(nn * dof)
inf_vec[0::2] = 1.0  # 1.0 for all Ux DOFs, 0.0 for Uy DOFs

# 3. Create the Time-Varying Load Function
# Returns a callable p(t) utilizing np.interp internally
p_func = seismic_load(M, inf_vec, ag_ms2, dt_record)

# 4. Set Initial Conditions
u0 = np.zeros((nn * dof, 1))
v0 = np.zeros((nn * dof, 1))

# 5. Solve using the Implicit Newmark-Beta Integrator
history = solve_newmark(
    M, C_damp, K, p_func, u0, v0, 
    dt=dt_record, 
    nsteps=n_points, 
    beta=0.25, gamma=0.5, 
    C_bc=C,
    compute_energy=True  # Enables calculating strain/kinetic histories
)
```

### Understanding the `result.u` History Array

The variable `history.u` returned by `solve_newmark` is a 2-dimensional NumPy array. It captures the spatial degrees of freedom across all time steps.

*   **Structure:** It has the shape `(N_steps, N_dof)`. The rows represent the passage of time (from $t=0$ to $t_{end}$), and the columns represent the individual Degrees of Freedom (Ux and Uy for all nodes).
*   **Slicing for Plotting:** To extract the history of a single node, we use 2D array slicing: `history.u[:, target_dof]`. The `:` means "give me all rows (all time steps)", and `target_dof` isolates the specific column.

```{code-block} python
# 6. Extract and Plot the Roof Displacement History
roof_dof_x = (np.argmax(X[:, 1])) * dof + 0

# Line-by-line slice: Grab all time steps (rows) for the single roof DOF column
u_roof = history.u[:, roof_dof_x]

plt.figure(figsize=(10, 4))
plt.plot(history.t, u_roof * 1000, 'b-', linewidth=1.5) # Convert m to mm
plt.title("Roof Relative Displacement History (X-Direction)")
plt.xlabel("Time (s)")
plt.ylabel("Displacement (mm)")
plt.grid(True, alpha=0.3)
plt.axhline(0, color='black', linewidth=0.8)
plt.fill_between(history.t, u_roof * 1000, 0, color='blue', alpha=0.1)
plt.show()
```

## 9.4 Global Energy Balance Extraction

To ensure numerical stability during the Time-History Analysis, engineers often plot the energy balance over time. The energy input into the system by the earthquake must equate to the sum of the system's kinetic energy, strain energy, and damped dissipation energy.

By passing `compute_energy=True` to `solve_newmark`, the solver computes these metrics line-by-line at every time step:
*   **Kinetic Energy:** $E_k = \frac{1}{2} \mathbf{\dot{u}}^T \mathbf{M} \mathbf{\dot{u}}$
*   **Strain Energy:** $E_s = \frac{1}{2} \mathbf{u}^T \mathbf{K} \mathbf{u}$
*   **Damped Energy:** Accumulated numerical integral of $\mathbf{\dot{u}}^T \mathbf{C} \mathbf{\dot{u}}$

You can directly access these arrays off the `history` object and visualize them:

```{code-block} python
# 7. Verification: Energy Conservation Plot
plt.figure(figsize=(10, 5))

plt.plot(history.t, history.kinetic_energy, label="Kinetic Energy", color='r', alpha=0.8)
plt.plot(history.t, history.strain_energy, label="Strain Energy", color='g', alpha=0.8)
plt.plot(history.t, history.damping_energy, label="Damped Dissipation", color='c', alpha=0.8)

# Compute the total internal energy sum
total_internal = history.kinetic_energy + history.strain_energy + history.damping_energy

plt.plot(history.t, total_internal, 'k--', label="Total Internal Energy", linewidth=2)
plt.plot(history.t, history.external_work, 'm:', label="Input External Work", linewidth=2)

plt.title("Seismic Energy Balance Verification")
plt.xlabel("Time (s)")
plt.ylabel("Energy (Joules)")
plt.legend(loc='upper right')
plt.grid(True, alpha=0.3)
plt.show()
```

When plotted, the Total Internal Energy curve should lay perfectly over the Input External Work curve. Any divergence indicates numerical instability or a time step (`dt`) that is too large.

These advanced workflows prove that `femlabpy` is not just a static matrix assembler, but a fully-fledged structural dynamics engine.