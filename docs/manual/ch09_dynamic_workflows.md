# Chapter 9: Advanced Dynamic Workflows

This chapter explores deeper, specialized workflows for structural dynamics, providing highly detailed code blocks for extracting building periods, plotting mode shapes, generating Frequency Response Functions (FRF), and performing comprehensive Time History Analyses (THA) under seismic loading.

## 9.1 Modal Analysis: Finding Periods and Plotting Mode Shapes

The structural period $T$ (in seconds) is the time it takes for a building to complete one full cycle of free vibration in a particular mode shape. The period is simply the inverse of the natural frequency: $T = 1 / f_n$.

`femlabpy` abstracts the eigenvalue extraction process in the `solve_modal` function. Behind the scenes, it condenses out the fixed Boundary Conditions (using the exact `C` array) and calls `scipy.linalg.eigh` (for dense matrices) or `scipy.sparse.linalg.eigsh` (for sparse matrices using Shift-and-Invert Lanczos iterations).

### Complete Code: 10-Story Shear Building Modal Analysis

Assume we have a simple column or building mesh. We want to find its fundamental periods and visualize its first 3 bending modes.

```python
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
    # Effective mass in X-direction (index 0)
    print(f"  Eff. Mass (X) = {result.effective_mass[i, 0]:.1f} kg")

# 6. Plotting Mode Shapes
# The mode shapes are mass-normalized eigenvectors. We scale them for visibility.
scale_factor = 2.0

fig, axes = plt.subplots(1, 3, figsize=(12, 6))
for i in range(3):
    ax = axes[i]
    # Extract the full 2D mode shape vector for mode i
    phi_i = result.mode_shapes[:, i].reshape(-1, 2)
    
    # Calculate deformed coordinates
    X_def = X + phi_i * scale_factor
    
    # Plot original mesh (light grey)
    for elem in T:
        n_idx = elem[:4] - 1  # 0-based indices
        poly = np.append(n_idx, n_idx[0]) # close the polygon
        ax.plot(X[poly, 0], X[poly, 1], 'k-', alpha=0.1)
        
        # Plot deformed mesh (blue)
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

### Plotting the FRF Curve

```python
from femlabpy.dynamics import compute_frf, plot_frf
from femlabpy.damping import rayleigh_damping, rayleigh_coefficients

# 1. Define Damping (5% Rayleigh damping anchored at Mode 1 and Mode 2)
a, b = rayleigh_coefficients(result.freq_hz[0], result.freq_hz[1], 0.05, 0.05)
C_damp = rayleigh_damping(M, K, a, b)

# 2. Define Input and Output DOFs
# Apply a unit harmonic force at the Roof (Top Right Node, X-direction)
roof_node = np.argmax(X[:, 1]) + 1
input_dof = (roof_node - 1) * dof + 0  # 0-based index for Ux

# Measure displacement at the same Roof DOF (Driving Point FRF)
output_dof = input_dof

# 3. Compute FRF from 0 to 200 Hz with 1000 resolution points
freqs, H_complex = compute_frf(
    M, C_damp, K, 
    input_dof=input_dof, 
    output_dof=output_dof, 
    freq_range=(0.1, 200.0), 
    n_points=1000
)

# 4. Plot Magnitude and Phase
fig = plot_frf(freqs, H_complex, log_scale=True, mark_peaks=True)
fig.suptitle("Roof Driving Point FRF (Receptance)")
plt.show()
```

In the resulting plot, you will see distinct magnitude peaks precisely at the frequencies calculated by `solve_modal`, heavily attenuated by the 5% Rayleigh damping.

## 9.3 Full Time-History Seismic Analysis

When given an earthquake record (e.g., an `.AT2` file containing acceleration samples), a standard Response Spectrum analysis is often insufficient for non-linear assessments. A complete Time-History Analysis (THA) is required.

In THA, the ground acceleration $\ddot{u}_g(t)$ is transformed into an effective dynamic point load applied to every mass in the structure:

$$ \mathbf{P}_{eff}(t) = -\mathbf{M} \mathbf{r} \ddot{u}_g(t) $$

where $\mathbf{r}$ is the influence vector (1 for DOFs in the shaking direction, 0 otherwise).

### Step-by-Step Earthquake Integration

```python
import numpy as np
import matplotlib.pyplot as plt
from femlabpy.dynamics import seismic_load, solve_newmark

# 1. Load an Earthquake Record (Assume we parsed a generic 1D array `ag_g`)
# `ag_g` is the acceleration in units of gravity (g)
# `dt_record` is the time step of the record (e.g., 0.01 seconds)
# Here we create a synthetic 10-second sweep for demonstration:
dt_record = 0.01
n_points = 1000
time_array = np.arange(n_points) * dt_record
ag_g = 0.5 * np.sin(2.0 * np.pi * 2.5 * time_array) * np.exp(-0.2 * time_array)

# Convert from g to m/s^2
ag_ms2 = ag_g * 9.80665

# 2. Build the Influence Vector for Horizontal (X) Shaking
inf_vec = np.zeros(nn * dof)
inf_vec[0::2] = 1.0  # 1.0 for all Ux DOFs, 0.0 for Uy DOFs

# 3. Create the Time-Varying Load Function
# `seismic_load` returns a Python callable `p(t)` that automatically 
# interpolates the discrete earthquake record at any integration time `t`.
p_func = seismic_load(M, inf_vec, ag_ms2, dt_record)

# 4. Set Initial Conditions
u0 = np.zeros((nn * dof, 1))
v0 = np.zeros((nn * dof, 1))

# 5. Solve using the Implicit Newmark-Beta Integrator
# We use the Unconditionally Stable Average Acceleration method (beta=0.25, gamma=0.5)
history = solve_newmark(
    M, C_damp, K, p_func, u0, v0, 
    dt=dt_record, 
    nsteps=n_points, 
    beta=0.25, gamma=0.5, 
    C_bc=C  # Extremely important: Pass the Boundary Constraints array!
)

# 6. Extract and Plot the Roof Displacement History
roof_dof_x = (np.argmax(X[:, 1])) * dof + 0
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

# 7. Verification: Energy Conservation
# You can also pass `compute_energy=True` to solve_newmark to track
# Strain Energy, Kinetic Energy, and Damped Dissipation over time!
```

These advanced workflows prove that `femlabpy` is not just a static matrix assembler, but a fully-fledged structural dynamics engine capable of replicating the exact procedures used by commercial tools like SAP2000 or ETABS.