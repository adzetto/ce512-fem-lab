# femlabpy — Dynamics, Periodic BCs & Advanced Solvers Implementation Plan

> **Author:** Auto-generated implementation plan  
> **Date:** 2026-04-04  
> **Target Version:** v0.6.0  
> **Total Tasks:** 82 across 12 phases  

---

## Current State Summary

femlabpy v0.5.0 is a Python FEM teaching library with:

| Category | What Exists |
|---|---|
| **Elements** | T3 (CST), Q4 (bilinear), T4 (tet), H8 (hex), Bar/Truss — structural + scalar/potential |
| **Solvers** | Linear elastic, nonlinear bar (arc-length/orthogonal residual), elastoplastic (Von Mises + Drucker-Prager) |
| **BCs** | Direct elimination (`setbc`), Lagrange multiplier (`solve_lag`, `solve_lag_general`) |
| **Materials** | Linear elastic (plane stress/strain/3D), Von Mises plasticity, Drucker-Prager plasticity |
| **Mesh I/O** | Gmsh `.msh` parser (v2.x ASCII + v4.x binary), 19 element types |
| **Post-process** | Reaction recovery, stress/strain at Gauss points, matplotlib contour plots |
| **Assembly** | Dense + scipy sparse (COO scatter), vectorized (T3/T4/H8/Bar) and loop-based (Q4) |

### What's Missing

- **No mass matrices** — none of the element routines compute element mass matrices
- **No damping** — no Rayleigh damping or any damping formulation
- **No time integration** — no Newmark-beta, central difference, HHT-alpha, or any transient solver
- **No modal analysis** — no eigenvalue solver for natural frequencies / mode shapes
- **No periodic BCs** — no multi-point constraint tying for periodic unit cells (RVE homogenization)
- **No consistent mass vs. lumped mass options**
- **No time-dependent loading** — no infrastructure for load functions `P(t)`

---

## Implementation Order

1. **Phase 1 + 10** → Mass matrices + new module infrastructure (foundation)
2. **Phase 6** → Modal analysis (validates mass matrices, needed for damping)
3. **Phase 2** → Damping models (required before dynamics)
4. **Phase 3** → Newmark-beta integrator (core dynamic solver)
5. **Phase 4** → Central difference / explicit solver
6. **Phase 5** → HHT-alpha method (refinement of Newmark)
7. **Phase 7** → Periodic boundary conditions (independent track)
8. **Phase 8** → Nonlinear dynamics
9. **Phase 9** → Dynamic post-processing & visualization
10. **Phase 11 + 12** → Examples & tests (validation throughout)

---

## Phase 1: Mass Matrices

> Add element-level mass matrix routines (consistent + lumped) for all element types.
> **Priority:** HIGH | **New files:** element modules get mass functions

### 1.1 Bar Mass Matrix
- [ ] **Function:** `mebar(Xe, Ge, *, lumped=False)`
- **Location:** `src/femlabpy/elements/bars.py`
- **Consistent (4×4 for 2D, 6×6 for 3D):**
  ```
  M = (rho * A * L / 6) * [2, 1; 1, 2] ⊗ I_dof
  ```
  where `rho` = density, `A` = cross-section area, `L` = bar length, `I_dof` = identity of size `dof`
- **Lumped (diagonal):**
  ```
  M = (rho * A * L / 2) * I_{2*dof}
  ```
- **Material table:** `G = [A, E, rho]` (extend from `[A, E]`)
- **Inputs:** `Xe` = nodal coordinates `(2, ndim)`, `Ge` = material row `[A, E, rho]`
- **Outputs:** `Me` = element mass matrix `(2*dof, 2*dof)`

### 1.2 T3 (CST Triangle) Mass Matrix
- [ ] **Function:** `met3e(Xe, Ge, *, lumped=False)`
- **Location:** `src/femlabpy/elements/triangles.py`
- **Consistent (6×6 for 2-DOF):**
  ```
  M = (rho * t * A / 12) * [2, 1, 1;
                             1, 2, 1;
                             1, 1, 2] ⊗ I_2
  ```
  where `A` = triangle area, `t` = thickness
- **Lumped (diagonal):**
  ```
  M = (rho * t * A / 3) * I_6
  ```
- **Material table:** `G = [E, nu, type, t, rho]` (extend from `[E, nu, type]` or `[E, nu]`)
- **Inputs:** `Xe` = nodal coordinates `(3, 2)`, `Ge` = material row
- **Outputs:** `Me` = `(6, 6)` mass matrix

### 1.3 Q4 (Bilinear Quad) Mass Matrix
- [ ] **Function:** `meq4e(Xe, Ge, *, lumped=False)`
- **Location:** `src/femlabpy/elements/quads.py`
- **Consistent (8×8) via 2×2 Gauss quadrature:**
  ```
  M = Σ_{gp} rho * t * N^T * N * det(J) * w
  ```
  where `N` = shape function row vector `[N1*I2, N2*I2, N3*I2, N4*I2]`
- **Lumped via row-sum:**
  ```
  M_L[i,i] = Σ_j M[i,j]    (then scale to preserve total mass)
  ```
- **Material table:** `G = [E, nu, type, t, rho]`
- **Inputs:** `Xe` = `(4, 2)`, `Ge` = material row
- **Outputs:** `Me` = `(8, 8)` mass matrix

### 1.4 T4 (Tetrahedron) Mass Matrix
- [ ] **Function:** `meT4e(Xe, Ge, *, lumped=False)`
- **Location:** `src/femlabpy/elements/solids.py`
- **Consistent (12×12) — analytical formula:**
  ```
  M = (rho * V / 20) * [2I, I, I, I;
                         I, 2I, I, I;
                         I, I, 2I, I;
                         I, I, I, 2I]
  ```
  where `V` = tet volume, `I` = 3×3 identity
- **Lumped (diagonal):**
  ```
  M = (rho * V / 4) * I_12
  ```
- **Material table:** `G = [E, nu, rho]` (extend from `[E, nu]`)
- **Inputs:** `Xe` = `(4, 3)`, `Ge` = material row
- **Outputs:** `Me` = `(12, 12)` mass matrix

### 1.5 H8 (Hexahedron) Mass Matrix
- [ ] **Function:** `meh8e(Xe, Ge, *, lumped=False)`
- **Location:** `src/femlabpy/elements/solids.py`
- **Consistent (24×24) via 2×2×2 Gauss quadrature:**
  ```
  M = Σ_{gp} rho * N^T * N * det(J) * w
  ```
  where `N` = `(3, 24)` shape function matrix with `N_i * I_3` blocks
- **Lumped via HRZ (Hinton-Rock-Zienkiewicz) method:**
  1. Compute consistent diagonal `d_i = M_c[i,i]`
  2. Scale: `M_L[i,i] = d_i * (total_mass / Σ d_i)`
- **Material table:** `G = [E, nu, rho]`
- **Inputs:** `Xe` = `(8, 3)`, `Ge` = material row
- **Outputs:** `Me` = `(24, 24)` mass matrix

### 1.6 Global Mass Assembly
- [ ] **Function:** `assmm(M, Me, Te, dof)`
- **Location:** `src/femlabpy/assembly.py`
- Identical scatter logic to `assmk()` but for mass matrix
- Handles both dense `np.ndarray` and `scipy.sparse.lil_matrix`
- Uses `np.ix_` indexing for scatter operations

### 1.7 Vectorized Mass Assembly
- [ ] **Functions:** `mt3e(M, T, X, G)`, `mT4e(M, T, X, G)`, `mh8e(M, T, X, G)`, `mbar(M, T, X, G, dof=2)`
- **Location:** Respective element modules
- Batch assembly using `np.einsum` matching existing vectorized k-routines
- Sparse COO scatter for large meshes
- **Pattern:** Same as `kt3e`, `kT4e`, `kh8e`, `kbar` but for mass

### 1.8 Material Table Extension
- [ ] Add `rho` (density) field to material table `G`
- **Backward compatibility:** If `rho` not present in G row, default to `rho=1.0` with a warning
- Updated conventions:
  - Bar: `[A, E]` → `[A, E, rho]`
  - 2D structural: `[E, nu]` or `[E, nu, type]` → `[E, nu, type, t, rho]`
  - 3D structural: `[E, nu]` → `[E, nu, rho]`
- Helper function: `_extract_density(Ge, element_type)` with smart defaults

### 1.9 Lumped Mass Option
- [ ] All mass routines accept `lumped: bool = False` keyword argument
- When `lumped=True`:
  - Bar, T3, T4: use analytical lumped formulas (equal distribution)
  - Q4, H8: use row-sum or HRZ method
- Document trade-offs: lumped = cheaper, no coupling, required for explicit methods; consistent = more accurate, better convergence for implicit

---

## Phase 2: Damping Models

> Implement Rayleigh and other damping formulations.
> **Priority:** HIGH | **New file:** `src/femlabpy/damping.py`

### 2.1 Rayleigh Damping
- [ ] **Function:** `rayleigh_damping(M, K, alpha, beta) -> C`
- **Formula:** `C = alpha * M + beta * K`
- **Convenience:** `rayleigh_coefficients(omega1, omega2, zeta1, zeta2) -> (alpha, beta)`
  ```
  [alpha]   1        [omega2, -omega1] [zeta1]
  [beta ] = ――――――――― [-1/omega2, 1/omega1] [zeta2]
            2
  ```
  Solves the 2×2 system for α, β given two target frequencies and damping ratios.
- Support both dense and sparse matrices
- **Location:** `src/femlabpy/damping.py`

### 2.2 Modal Damping
- [ ] **Function:** `modal_damping(M, K, zeta, omega, phi) -> C`
- **Formula:** `C = M * Φ * diag(2*ζᵢ*ωᵢ) * Φ^T * M`
  where `Φ` = mass-normalized mode shapes, `ζᵢ` = damping ratios, `ωᵢ` = natural frequencies
- Requires modal analysis results (Phase 6) as input
- **Priority:** MEDIUM

### 2.3 Caughey Damping Series
- [ ] **Function:** `caughey_damping(M, K, omega_targets, zeta_targets) -> C`
- **Formula:** `C = M * Σ aₖ * (M⁻¹K)^k` for k = 0, 1, ..., n-1
- Allows prescribing damping ratios at more than 2 frequencies
- Coefficients `aₖ` found by solving a Vandermonde-like system
- **Priority:** LOW

---

## Phase 3: Newmark-Beta Time Integrator

> Core transient dynamic solver — the backbone of structural dynamics.
> **Priority:** HIGH | **New file:** `src/femlabpy/dynamics.py`

### 3.1 Time Integration Data Structures
- [ ] **Dataclass:** `TimeHistory`
  ```python
  @dataclass
  class TimeHistory:
      t: np.ndarray          # (nsteps+1,) time values
      u: np.ndarray          # (nsteps+1, ndof) displacement history
      v: np.ndarray          # (nsteps+1, ndof) velocity history
      a: np.ndarray          # (nsteps+1, ndof) acceleration history
      dt: float              # time step size
      nsteps: int            # number of time steps
      energy: dict | None    # {kinetic, strain, dissipated, external, total}
  ```
- **Dataclass:** `NewmarkParams`
  ```python
  @dataclass
  class NewmarkParams:
      beta: float = 0.25     # Newmark beta parameter
      gamma: float = 0.5     # Newmark gamma parameter
      
      @classmethod
      def average_acceleration(cls): return cls(0.25, 0.5)
      
      @classmethod
      def linear_acceleration(cls): return cls(1/6, 0.5)
      
      @classmethod
      def central_difference(cls): return cls(0.0, 0.5)
  ```

### 3.2 Load Function Interface
- [ ] **Abstract interface:** `load_func(t) -> p_vector`
- **Built-in load functions:**
  ```python
  def constant_load(P):
      """Returns lambda t: P (constant in time)"""
  
  def ramp_load(P, t_ramp):
      """Linear ramp: P * min(t/t_ramp, 1.0)"""
  
  def harmonic_load(P, omega, phase=0.0):
      """Sinusoidal: P * sin(omega*t + phase)"""
  
  def pulse_load(P, t_start, t_duration):
      """Rectangular pulse: P for t_start <= t <= t_start+t_duration, else 0"""
  
  def tabulated_load(P, time_table, value_table):
      """Interpolated from table: P * interp1d(time_table, value_table)(t)"""
  
  def seismic_load(M, direction, accel_record, dt_record):
      """Ground motion: p(t) = -M @ direction * a_g(t)"""
  ```
- All return a callable `f(t) -> ndarray(ndof, 1)`

### 3.3 Newmark-Beta Solver (Main)
- [ ] **Function:** `solve_newmark(M, C, K, p_func, u0, v0, dt, nsteps, *, beta=0.25, gamma=0.5, C_bc=None, dof=2) -> TimeHistory`
- **Algorithm (implicit, average acceleration default):**
  ```
  1. Compute initial acceleration: a0 = M^{-1} * (p(0) - C*v0 - K*u0)
  2. Form effective stiffness: K_eff = K + M/(beta*dt^2) + gamma*C/(beta*dt)
  3. Factorize K_eff (once, since linear)
  4. For each step n = 0, 1, ..., nsteps-1:
     a. Compute effective load:
        p_eff = p(t_{n+1}) 
              + M * [u_n/(beta*dt^2) + v_n/(beta*dt) + (1/(2*beta)-1)*a_n]
              + C * [gamma*u_n/(beta*dt) + (gamma/beta-1)*v_n + dt*(gamma/(2*beta)-1)*a_n]
     b. Solve: K_eff * u_{n+1} = p_eff
     c. Update acceleration: a_{n+1} = (u_{n+1}-u_n)/(beta*dt^2) - v_n/(beta*dt) - (1/(2*beta)-1)*a_n
     d. Update velocity: v_{n+1} = v_n + dt*[(1-gamma)*a_n + gamma*a_{n+1}]
     e. Store in TimeHistory
  ```

### 3.4 Effective Stiffness Formation
- [ ] **Function:** `_newmark_effective_stiffness(K, M, C, dt, beta, gamma) -> K_eff`
- **Formula:**
  ```
  K_eff = K + (1 / (beta * dt^2)) * M + (gamma / (beta * dt)) * C
  ```
- Handles sparse and dense matrices
- Returns factorized form (`scipy.sparse.linalg.splu` or `scipy.linalg.lu_factor`) for repeated solves

### 3.5 Newmark Update Formulas
- [ ] **Function:** `_newmark_update(u_new, u_old, v_old, a_old, dt, beta, gamma) -> (a_new, v_new)`
- **Acceleration update:**
  ```
  a_{n+1} = (1/(beta*dt^2)) * (u_{n+1} - u_n) - (1/(beta*dt)) * v_n - (1/(2*beta) - 1) * a_n
  ```
- **Velocity update:**
  ```
  v_{n+1} = v_n + dt * [(1 - gamma) * a_n + gamma * a_{n+1}]
  ```

### 3.6 Newmark Parameter Presets
- [ ] Named presets as class methods on `NewmarkParams`:
  - `average_acceleration()` → `(beta=0.25, gamma=0.5)` — unconditionally stable, no numerical dissipation, 2nd order accurate
  - `linear_acceleration()` → `(beta=1/6, gamma=0.5)` — conditionally stable (`dt < 0.551 * T_min`), 2nd order accurate
  - `central_difference()` → `(beta=0, gamma=0.5)` — explicit, conditionally stable (`dt < T_min/π`), 2nd order accurate
  - `fox_goodwin()` → `(beta=1/12, gamma=0.5)` — conditionally stable, 4th order accurate for undamped SDOF
- Document stability regions and accuracy order for each

### 3.7 Adaptive Time Stepping
- [ ] **Function:** `solve_newmark_adaptive(M, C, K, p_func, u0, v0, dt0, t_end, *, tol=1e-4) -> TimeHistory`
- **Error estimator:** Run dual solutions with `beta=0.25` and `beta=1/6`, compare:
  ```
  err = ||u_025 - u_016|| / ||u_025||
  ```
- **Step adjustment:** `dt_new = dt * (tol / err)^(1/3)` with safety factor 0.9
- **Priority:** LOW (nice-to-have)

### 3.8 Boundary Conditions in Dynamic Context
- [ ] **Function:** `_apply_dynamic_bc(K_eff, p_eff, C_bc, dof, t, ks)`
- Handle three BC types:
  1. **Fixed zero:** Same as static `setbc` — zero row/col, penalty diagonal
  2. **Prescribed displacement history:** `u_bc(t)` — modify effective load:
     ```
     p_eff[free] -= K_eff[free, bc] * u_bc(t)
     ```
  3. **Velocity BC:** Convert to displacement BC via integration
- BC constraint matrix `C_bc` can include time-dependent values via callable
- Re-use existing `setbc` pattern where possible

### 3.9 Energy Balance Check
- [ ] **Function:** `_compute_energy(M, C, K, u, v, p_func, t, dt) -> dict`
- **Quantities at each step:**
  ```
  E_kinetic    = 0.5 * v^T * M * v
  E_strain     = 0.5 * u^T * K * u
  E_dissipated += 0.5 * (v_n + v_{n+1})^T * C * (v_n + v_{n+1}) * dt  (incremental)
  W_external   += 0.5 * (p_n + p_{n+1})^T * (u_{n+1} - u_n)           (incremental)
  E_total      = E_kinetic + E_strain + E_dissipated
  ```
- **Validation:** For undamped system (C=0), `E_total` should be constant
- Store in `TimeHistory.energy` dict

### 3.10 Stability Check
- [ ] **Function:** `_check_stability(K, M, dt, beta, gamma) -> (is_stable, dt_critical)`
- **For conditionally stable schemes (beta < 0.25 or gamma != 0.5):**
  ```
  omega_max = sqrt(max(eig(M^{-1} K)))      # maximum natural frequency
  dt_critical = 2 / omega_max               # for central difference
  ```
  For linear acceleration: `dt_critical = 0.551 * (2*pi / omega_max)`
- **Warning:** Emit `UserWarning` if `dt > dt_critical`
- For **unconditionally stable** schemes `(beta >= gamma/2 >= 0.25)`: no warning needed

---

## Phase 4: Central Difference (Explicit) Solver

> For wave propagation, impact, and blast loading.
> **Priority:** MEDIUM | **Location:** `src/femlabpy/dynamics.py`

### 4.1 Central Difference Solver
- [ ] **Function:** `solve_central_diff(M_lumped, C, K, p_func, u0, v0, dt, nsteps, *, dof=2) -> TimeHistory`
- **Requirements:** `M_lumped` must be diagonal (1D array or diagonal matrix)
- **Algorithm:**
  ```
  1. Initial: a0 = M_L^{-1} * (p(0) - C*v0 - K*u0)
  2. Compute: u_{-1} = u0 - dt*v0 + 0.5*dt^2*a0  (backward extrapolation)
  3. For each step n = 0, 1, ..., nsteps-1:
     a. p_eff = p(t_n) - (K - 2*M_L/dt^2)*u_n - (M_L/dt^2 - C/(2*dt))*u_{n-1}
     b. u_{n+1} = (M_L/dt^2 + C/(2*dt))^{-1} * p_eff
     c. Velocity: v_n = (u_{n+1} - u_{n-1}) / (2*dt)
     d. Acceleration: a_n = (u_{n+1} - 2*u_n + u_{n-1}) / dt^2
  ```
- **Key advantage:** No matrix factorization when M is diagonal and C is also diagonal (or zero)

### 4.2 Central Difference Algorithm Details
- [ ] Implement the three-level time marching scheme
- Handle the special initial step (u_{-1} computation)
- For diagonal M and diagonal C: pure element-by-element, O(ndof) per step
- For diagonal M and full C: still requires M/C combination but no full solve

### 4.3 Critical Time Step Computation
- [ ] **Function:** `critical_timestep(K, M, *, method='power') -> dt_cr`
- **Methods:**
  - `'power'`: Power iteration to find `omega_max = sqrt(lambda_max(M^{-1}K))`
    then `dt_cr = 2 / omega_max`
  - `'element'`: Estimate from smallest element size and wave speed
    `dt_cr = h_min / c` where `c = sqrt(E/rho)` and `h_min` = min element characteristic length
- **Usage:** Called automatically by `solve_central_diff` with warning if `dt > dt_cr`

### 4.4 Lumped Mass Validation
- [ ] Auto-check in `solve_central_diff`:
  - If M is not diagonal, raise `ValueError("Central difference requires lumped (diagonal) mass matrix")`
  - Provide helper: `is_lumped(M) -> bool` — checks if off-diagonal entries are zero (or negligible)
  - Suggest: `"Use lumped=True in mass matrix computation"`

---

## Phase 5: HHT-Alpha (Hilber-Hughes-Taylor) Method

> Numerical dissipation control for high-frequency noise.
> **Priority:** MEDIUM | **Location:** `src/femlabpy/dynamics.py`

### 5.1 HHT-Alpha Solver
- [ ] **Function:** `solve_hht(M, C, K, p_func, u0, v0, dt, nsteps, *, alpha=-0.05) -> TimeHistory`
- **Valid range:** `alpha ∈ [-1/3, 0]`
  - `alpha = 0` → standard Newmark (trapezoidal)
  - `alpha = -0.05` → mild dissipation (recommended default)
  - `alpha = -1/3` → maximum dissipation

### 5.2 HHT Parameters and Equilibrium
- [ ] **Derived Newmark parameters:**
  ```
  beta  = (1 - alpha)^2 / 4
  gamma = 0.5 - alpha
  ```
- **Modified equilibrium equation at step n+1:**
  ```
  M * a_{n+1} + (1+alpha) * C * v_{n+1} - alpha * C * v_n
               + (1+alpha) * K * u_{n+1} - alpha * K * u_n
               = (1+alpha) * p_{n+1} - alpha * p_n
  ```
- **Effective stiffness:**
  ```
  K_eff = (1+alpha)*K + M/(beta*dt^2) + (1+alpha)*gamma*C/(beta*dt)
  ```
- **Effective load includes alpha-weighted previous step terms**

### 5.3 Recovery & Validation
- [ ] Verify that `alpha=0` produces identical results to `solve_newmark(beta=0.25, gamma=0.5)`
- Demonstrate numerical dissipation: run high-frequency oscillation and show amplitude decay increases with |alpha|
- Show that low-frequency response is preserved (2nd order accuracy maintained)

---

## Phase 6: Modal Analysis (Eigenvalue Solver)

> Natural frequencies and mode shapes.
> **Priority:** HIGH | **New file:** `src/femlabpy/modal.py`

### 6.1 Eigenvalue Solver
- [ ] **Function:** `solve_modal(K, M, n_modes=10, *, C_bc=None, dof=2, sigma=0.0) -> ModalResult`
- **Dataclass:** `ModalResult`
  ```python
  @dataclass
  class ModalResult:
      eigenvalues: np.ndarray      # (n_modes,) omega^2 values
      omega: np.ndarray            # (n_modes,) natural frequencies in rad/s
      freq_hz: np.ndarray          # (n_modes,) frequencies in Hz
      period: np.ndarray           # (n_modes,) periods in seconds
      mode_shapes: np.ndarray      # (ndof, n_modes) eigenvectors (columns)
      participation: np.ndarray    # (n_modes, ndim) modal participation factors
      effective_mass: np.ndarray   # (n_modes, ndim) effective modal mass
  ```
- **Solver:** `scipy.sparse.linalg.eigsh(K, k=n_modes, M=M, sigma=sigma, which='SM')`
  (shift-invert mode for smallest eigenvalues)

### 6.2 Boundary Condition Handling in Eigenvalue Problem
- [ ] **Strategy: System Reduction**
  ```python
  def _reduce_system(K, M, C_bc, dof):
      """Eliminate constrained DOFs before eigensolve."""
      free_dofs = _get_free_dofs(C_bc, dof, ndof)
      K_red = K[np.ix_(free_dofs, free_dofs)]
      M_red = M[np.ix_(free_dofs, free_dofs)]
      return K_red, M_red, free_dofs
  ```
- After eigensolve, expand mode shapes back to full DOF space (zeros at constrained DOFs)
- **Alternative:** Large penalty on constrained DOFs: `K[j,j] += 1e20 * max(diag(K))` (simpler but less accurate)

### 6.3 Mode Shape Normalization
- [ ] **Mass normalization (default):**
  ```
  phi_i = phi_i / sqrt(phi_i^T * M * phi_i)
  ```
  so that `Phi^T * M * Phi = I`
- **Displacement normalization:**
  ```
  phi_i = phi_i / max(|phi_i|)
  ```
  so that maximum component is 1.0

### 6.4 Frequency Output
- [ ] Compute and store:
  ```
  omega_i = sqrt(lambda_i)         # rad/s
  f_i     = omega_i / (2 * pi)     # Hz
  T_i     = 1 / f_i                # seconds (period)
  ```
- Print formatted table:
  ```
  Mode  omega (rad/s)   f (Hz)    T (s)
  ----  ------------   --------  --------
    1      12.34        1.964     0.509
    2      48.67        7.747     0.129
    ...
  ```

### 6.5 Modal Participation Factors
- [ ] **Function:** `_modal_participation(M, phi, dof) -> (participation, effective_mass)`
- **Participation factor for mode i, direction d:**
  ```
  Gamma_{i,d} = phi_i^T * M * r_d
  ```
  where `r_d` = unit influence vector for direction d (1 at all DOFs in direction d, 0 elsewhere)
- **Effective modal mass:**
  ```
  M_eff_{i,d} = Gamma_{i,d}^2 / (phi_i^T * M * phi_i)
  ```
- Sum of effective masses for all modes = total mass (verification)

### 6.6 Mode Shape Plotting
- [ ] **Function:** `plot_modes(T, X, phi, dof, mode_indices=None, *, scale=1.0, rows=None, cols=None)`
- Subplot grid showing deformed mesh for each requested mode
- Overlay undeformed (dashed) and deformed (solid) mesh
- Title: "Mode {i}: f = {f_i:.3f} Hz"
- Optional: colormap by displacement magnitude

---

## Phase 7: Periodic Boundary Conditions

> For RVE/unit cell analysis and computational homogenization.
> **Priority:** HIGH | **New file:** `src/femlabpy/periodic.py`

### 7.1 Periodic Node Pair Identification
- [ ] **Function:** `find_periodic_pairs(X, axis, tol=1e-6) -> ndarray`
- **Algorithm:**
  ```
  1. Find bounding box: x_min, x_max along specified axis
  2. Select nodes on "left" face: X[:, axis] ≈ x_min
  3. Select nodes on "right" face: X[:, axis] ≈ x_max
  4. For each left node, find matching right node by remaining coordinates
  5. Return pairs as (n_pairs, 2) array: [[left_node, right_node], ...]
  ```
- **Matching:** KD-tree or brute-force distance for remaining coordinate dimensions
- **Tolerance:** `tol` relative to element size for floating-point robustness
- **Output:** `pairs = [[n_left_1, n_right_1], [n_left_2, n_right_2], ...]` (1-based node numbers)

### 7.2 Multi-Axis Periodicity
- [ ] **Function:** `find_all_periodic_pairs(X, periodic_axes, tol=1e-6) -> dict`
- **Support:**
  - 1D: `periodic_axes=[0]` → x-periodic only
  - 2D: `periodic_axes=[0, 1]` → x+y periodic
  - 3D: `periodic_axes=[0, 1, 2]` → fully periodic
- Returns dict: `{0: pairs_x, 1: pairs_y, 2: pairs_z}`
- Handle **corner nodes** that appear in multiple pair sets (avoid double-constraining)
- Handle **edge nodes** in 3D (on intersection of two periodic faces)

### 7.3 Periodic Constraint Matrix
- [ ] **Function:** `periodic_constraints(X, pairs, dof, *, eps_macro=None) -> (G, Q)`
- **Constraint equation for each pair (L, R):**
  ```
  u_R - u_L = eps_macro * (x_R - x_L)
  ```
  In matrix form: `G * u = Q`
  - `G[row, R_dof] = +1`, `G[row, L_dof] = -1`
  - `Q[row] = eps_macro * (x_R - x_L)` projected onto the DOF direction
- **If `eps_macro=None`:** Purely periodic (zero fluctuation): `Q = 0`, constraints become `u_R = u_L`
- **Size:** `G` is `(n_constraints, n_total_dof)`, `Q` is `(n_constraints, 1)`
- n_constraints = n_pairs * dof

### 7.4 Integration with Lagrange Multiplier Solver
- [ ] **Function:** `solve_periodic(K, p, X, T, G, dof, *, eps_macro=None) -> (u, lam)`
- Build `G, Q` from `periodic_constraints()`
- Pass to existing `solve_lag_general(K, p, G, Q)` to solve the augmented system:
  ```
  [K    G^T] [u]   [p]
  [G    0  ] [λ] = [Q]
  ```
- Return displacement field `u` and Lagrange multipliers `λ` (constraint forces = tractions)

### 7.5 MPC Alternative (Direct Elimination)
- [ ] **Function:** `apply_mpc(K, p, pairs, dof) -> (K_red, p_red, slave_dofs)`
- **Strategy:** For each pair, eliminate "right" (slave) DOF:
  ```
  u_slave = u_master + Q/G
  ```
  Condense K and p by substituting slave DOFs
- Advantage: Smaller system, no Lagrange multipliers
- Disadvantage: More complex implementation, harder to extract reactions
- **Priority:** MEDIUM

### 7.6 Macro Strain Application
- [ ] **Function:** `apply_macro_strain(X, pairs, eps_macro, dof) -> Q`
- **2D macro strain tensor (Voigt):** `eps_macro = [exx, eyy, gamma_xy]`
  ```
  eps_tensor = [[exx,       gamma_xy/2],
                [gamma_xy/2, eyy       ]]
  ```
- **3D macro strain tensor (Voigt):** `eps_macro = [exx, eyy, ezz, gamma_xy, gamma_yz, gamma_xz]`
- **RHS for each pair:**
  ```
  Q_pair = eps_tensor @ (x_R - x_L)   → vector of size dof
  ```
- Assembled into full Q vector matching constraint ordering

### 7.7 Computational Homogenization
- [ ] **Function:** `homogenize(K, T, X, G, pairs, dof) -> C_eff`
- **Algorithm:**
  ```
  1. For each canonical unit strain e_i (i = 1..n_voigt):
     a. Build G, Q_i = apply_macro_strain(X, pairs, e_i, dof)
     b. Solve: u_i = solve_periodic(K, p=0, X, T, G, dof, eps_macro=e_i)
     c. Compute: sigma_avg_i = volume_average_stress(T, X, G, u_i, dof)
  2. Assemble C_eff column-wise: C_eff[:, i] = sigma_avg_i
  ```
- **2D:** 3 load cases (exx, eyy, gamma_xy) → 3×3 `C_eff`
- **3D:** 6 load cases → 6×6 `C_eff`
- Verify: `C_eff` should be symmetric and positive definite

### 7.8 Volume Averaging Utilities
- [ ] **Function:** `volume_average_stress(T, X, G, u, dof, *, element_type='q4') -> sigma_avg`
  ```
  sigma_avg = (1/V_total) * Σ_e sigma_e * V_e
  ```
  where `sigma_e` = stress at element centroid (or Gauss point average), `V_e` = element volume/area
- [ ] **Function:** `volume_average_strain(T, X, G, u, dof, *, element_type='q4') -> eps_avg`
  ```
  eps_avg = (1/V_total) * Σ_e eps_e * V_e
  ```
- Support T3, Q4, T4, H8 elements
- For elements with multiple Gauss points (Q4, H8), average over Gauss points first

### 7.9 Corner Node Fixing
- [ ] **Function:** `fix_corner(X, C, dof) -> C_extended`
- Find the node at the minimum coordinates (bottom-left corner for 2D)
- Add zero displacement constraints for all DOFs at that node
- Removes rigid body translation while preserving periodicity
- For 2D: fix 2 DOFs (remove translation)
- For 3D: fix 3 DOFs at corner + additional rotational fix if needed

### 7.10 Periodic Mesh Validation
- [ ] **Function:** `check_periodic_mesh(X, T, axis, tol=1e-6) -> (is_valid, report)`
- **Checks:**
  1. Equal number of nodes on opposite faces
  2. Matching coordinate patterns (sorted by remaining coordinates)
  3. Compatible element topology on faces (matching edge patterns)
  4. No orphan nodes (not paired)
- **Report:** Human-readable dict with pass/fail for each check + suggested fixes
- Warn if mesh is non-periodic and suggest re-meshing with Gmsh structured mesh

---

## Phase 8: Nonlinear Dynamics

> Combine existing nonlinear solvers with time integration.
> **Priority:** MEDIUM | **Location:** `src/femlabpy/dynamics.py`

### 8.1 Nonlinear Newmark Solver
- [ ] **Function:** `solve_newmark_nl(M, C, tangent_func, internal_force_func, p_func, u0, v0, dt, nsteps, *, beta=0.25, gamma=0.5, tol=1e-6, max_iter=20) -> TimeHistory`
- **Arguments:**
  - `tangent_func(u, state) -> K_t` — computes tangent stiffness at current config
  - `internal_force_func(u, state) -> (q, state_new)` — computes internal forces + updates state
  - `state` — can carry plastic history `(S, E)` or geometric config
- **Algorithm at each time step:**
  ```
  1. Predictor: u_pred = u_n + dt*v_n + 0.5*dt^2*(1-2*beta)*a_n
                v_pred = v_n + dt*(1-gamma)*a_n
  2. Newton iteration k = 0, 1, ...:
     a. K_t = tangent_func(u^{k}, state)
     b. q = internal_force_func(u^{k}, state)
     c. K_eff = K_t + M/(beta*dt^2) + gamma*C/(beta*dt)
     d. R = p(t_{n+1}) - M*a^{k} - C*v^{k} - q
     e. Δu = K_eff^{-1} * R
     f. u^{k+1} = u^{k} + Δu
     g. Update a, v from Newmark formulas
     h. Check convergence: ||R|| / ||p|| < tol
  3. Accept: u_{n+1} = u^{k+1}, update state
  ```

### 8.2 Tangent-Based Iteration Details
- [ ] **Effective tangent:** `K_eff(u) = K_t(u) + a0*M + a1*C`
  where `a0 = 1/(beta*dt^2)`, `a1 = gamma/(beta*dt)`
- Option: `modified_newton=True` → re-use K_t from first iteration (cheaper, slower convergence)
- Option: `initial_stiffness=True` → use K_0 (elastic) throughout (simplest, worst convergence)

### 8.3 Convergence Criteria
- [ ] **Multiple criteria (configurable):**
  1. **Residual norm:** `||R||₂ / ||p(t_{n+1})||₂ < tol_r`
  2. **Energy norm:** `|ΔuᵀR| / |u_{n+1}ᵀp_{n+1}| < tol_e`
  3. **Displacement increment:** `||Δu||₂ / ||u_{n+1}||₂ < tol_u`
- Default: use residual norm with `tol_r = 1e-6`
- Log iteration history: `{step, iter, ||R||, ||Δu||, energy_error}`

### 8.4 Line Search and Arc-Length
- [ ] **Line search:** After Newton direction `Δu`, find optimal step `η ∈ (0, 1]`:
  ```
  min η: G(η) = Δu^T * R(u + η*Δu)
  ```
  Using bisection or cubic interpolation
- **Arc-length within time steps:** For highly nonlinear snap-through in dynamics
  - Constraint: `||Δu||^2 + (β*ψ*dt)^2*||Δλ*p||^2 = Δs^2`
  - Rarely needed but important for explosive/impact problems
- **Priority:** LOW

---

## Phase 9: Dynamic Post-Processing & Visualization

> **Priority:** MEDIUM | **Location:** `src/femlabpy/plotting.py` (extend)

### 9.1 Time History Plots
- [ ] **Function:** `plot_time_history(result: TimeHistory, dof_index: int | list, *, quantity='displacement', ax=None)`
- **Quantities:** `'displacement'`, `'velocity'`, `'acceleration'`
- Support single DOF (line plot) or multiple DOFs (overlaid or subplot)
- Auto-label axes: "Displacement (m)", "Time (s)", etc.
- Return matplotlib axes for further customization

### 9.2 Frequency Response Function (FRF)
- [ ] **Function:** `compute_frf(M, C, K, input_dof, output_dof, freq_range, *, n_points=500) -> (freq, H)`
- **Formula:** At each frequency `ω`:
  ```
  H(ω) = [(-ω^2*M + iω*C + K)^{-1}]_{output_dof, input_dof}
  ```
- **Plotting:** `plot_frf(freq, H, *, log_scale=True)` — magnitude and phase vs frequency
- Mark peaks (resonances) with vertical lines and frequency labels
- Support dB scale: `20*log10(|H|/|H_ref|)`

### 9.3 Dynamic Animation
- [ ] **Function:** `animate_dynamic(T, X, u_history, dof, *, dt=1.0, scale=1.0, interval=50, save_path=None)`
- Uses `matplotlib.animation.FuncAnimation`
- Each frame: deformed mesh at time step `n`
- Optional: color by displacement magnitude or stress
- Save to `.gif` or `.mp4` via `save_path`
- Show time counter in title

### 9.4 Energy Plots
- [ ] **Function:** `plot_energy(result: TimeHistory, *, ax=None)`
- Plot on same axes: E_kinetic (blue), E_strain (red), E_dissipated (green), W_external (black dashed), E_total (thick black)
- Verify: for undamped, E_total should be flat (horizontal line)
- Legend with max values

### 9.5 Phase Plots
- [ ] **Function:** `plot_phase(result: TimeHistory, dof_index: int, *, ax=None)`
- Plot velocity vs displacement at specified DOF
- For SDOF harmonic: should show ellipse
- For nonlinear: shows limit cycle or chaotic attractor
- **Priority:** LOW

### 9.6 FFT of Response
- [ ] **Function:** `fft_response(result: TimeHistory, dof_index: int, *, n_fft=None, window='hann') -> (freq, amplitude)`
- Apply window function, compute `np.fft.rfft`
- Plot amplitude spectrum with peaks labeled
- Compare peaks with modal frequencies (if available)
- **Priority:** LOW

---

## Phase 10: Infrastructure & Integration

> **Priority:** HIGH

### 10.1 Create `src/femlabpy/dynamics.py`
- [ ] New module housing:
  - `TimeHistory` and `NewmarkParams` dataclasses
  - `solve_newmark()` — implicit Newmark-beta solver
  - `solve_central_diff()` — explicit central difference
  - `solve_hht()` — HHT-alpha method
  - `solve_newmark_nl()` — nonlinear Newmark
  - Load function builders (constant, ramp, harmonic, pulse, tabulated, seismic)
  - `critical_timestep()` — critical dt computation
  - Internal helpers (`_newmark_effective_stiffness`, `_newmark_update`, `_apply_dynamic_bc`, `_compute_energy`, `_check_stability`)

### 10.2 Create `src/femlabpy/modal.py`
- [ ] New module housing:
  - `ModalResult` dataclass
  - `solve_modal()` — eigenvalue solver
  - `_reduce_system()` — BC elimination for eigenproblems
  - `_modal_participation()` — participation factors
  - `plot_modes()` — mode shape visualization

### 10.3 Create `src/femlabpy/periodic.py`
- [ ] New module housing:
  - `find_periodic_pairs()` — node pair identification
  - `find_all_periodic_pairs()` — multi-axis periodicity
  - `periodic_constraints()` — constraint matrix builder
  - `solve_periodic()` — Lagrange multiplier periodic solver
  - `apply_mpc()` — direct elimination alternative
  - `apply_macro_strain()` — RHS from macro strain
  - `homogenize()` — full homogenization pipeline
  - `volume_average_stress()` / `volume_average_strain()` — averaging
  - `fix_corner()` — rigid body removal
  - `check_periodic_mesh()` — mesh validation

### 10.4 Extend `init()` in `core.py`
- [ ] **Updated signature:** `init(nn, dof, *, dynamic=False, use_sparse=None)`
- When `dynamic=True`: return `(K, M, p, q)` — allocate mass matrix M in addition to K
- Damping matrix C is not pre-allocated (constructed from M, K via Rayleigh coefficients)
- Backward compatible: `dynamic=False` (default) returns `(K, p, q)` as before

### 10.5 Update `__init__.py` Exports
- [ ] Add all new public symbols to `__all__`:
  ```python
  # Mass matrices
  from .elements.bars import mebar, mbar
  from .elements.triangles import met3e, mt3e
  from .elements.quads import meq4e, mq4e
  from .elements.solids import meT4e, mT4e, meh8e, mh8e
  
  # Damping
  from .damping import rayleigh_damping, rayleigh_coefficients, modal_damping
  
  # Dynamics
  from .dynamics import (solve_newmark, solve_central_diff, solve_hht,
                         solve_newmark_nl, TimeHistory, NewmarkParams,
                         constant_load, ramp_load, harmonic_load, pulse_load,
                         tabulated_load, seismic_load, critical_timestep)
  
  # Modal
  from .modal import solve_modal, ModalResult, plot_modes
  
  # Periodic
  from .periodic import (find_periodic_pairs, find_all_periodic_pairs,
                         periodic_constraints, solve_periodic, apply_macro_strain,
                         homogenize, volume_average_stress, volume_average_strain,
                         check_periodic_mesh, fix_corner)
  ```

### 10.6 Update CLI `--info`
- [ ] Extend `__main__.py` to list:
  - Dynamic solvers: Newmark-beta, central difference, HHT-alpha, nonlinear Newmark
  - Modal analysis capabilities
  - Periodic BC support
  - Mass matrix element types
  - Damping models
- **Priority:** LOW

---

## Phase 11: Examples & Validation

> **Priority:** HIGH for core examples, MEDIUM for advanced

### 11.1 SDOF Spring-Mass-Damper (Newmark Validation)
- [ ] **File:** `src/femlabpy/examples/dynamic_sdof.py`
- **Setup:** Single DOF: `m=1, k=100, c=2, p(t) = 10*sin(5t)`
- **Analytical solution:** `u(t) = A*sin(ωt - φ) + transient`
  where `ω_n = sqrt(k/m)`, `ζ = c/(2*m*ω_n)`, `ω_d = ω_n*sqrt(1-ζ²)`
- **Verification:** Compare Newmark solution with analytical at all time steps
- **Convergence:** Show error decreases as `O(dt²)`

### 11.2 Cantilever Beam Free Vibration (Modal Validation)
- [ ] **File:** `src/femlabpy/examples/dynamic_cantilever.py`
- **Setup:** Cantilever beam (Q4 elements), fixed left end, free right end
- **Analytical frequencies (Euler-Bernoulli):**
  ```
  f_n = (beta_n * L)^2 / (2*pi*L^2) * sqrt(EI / (rho*A))
  ```
  where `beta_1*L = 1.875, beta_2*L = 4.694, beta_3*L = 7.855, ...`
- **Procedure:** Assemble K and M, run `solve_modal()`, compare first 3 frequencies
- **Expected accuracy:** Within 5% for coarse Q4 mesh, <1% for refined mesh

### 11.3 Bar Wave Propagation (Explicit Validation)
- [ ] **File:** `src/femlabpy/examples/dynamic_wave.py`
- **Setup:** Long bar, impulse at one end, free at other
- **Analytical:** Wave travels at `c = sqrt(E/rho)`, arrives at other end at `t = L/c`
- **Solver:** `solve_central_diff` with lumped mass
- **Verification:** Wave front position matches analytical speed
- **Priority:** MEDIUM

### 11.4 2D RVE with Circular Inclusion (Homogenization)
- [ ] **File:** `src/femlabpy/examples/periodic_rve.py`
- **Setup:** Square unit cell (Q4 mesh) with circular inclusion of different material
- **Materials:** Matrix (E1, nu1), Inclusion (E2, nu2), volume fraction `f`
- **Procedure:** `homogenize()` → C_eff
- **Validation:** Compare with Mori-Tanaka prediction:
  ```
  E_eff = E_m * (1 + f*(E_i/E_m - 1) / (1 + (1-f)*(E_i/E_m - 1)*S))
  ```
  and check that C_eff lies within Hashin-Shtrikman bounds

### 11.5 Periodic Unit Cell Under Shear
- [ ] **File:** `src/femlabpy/examples/periodic_shear.py`
- **Setup:** Homogeneous unit cell, apply macro shear strain `gamma_xy = 0.01`
- **Expected result:** Uniform shear stress throughout, sigma_xy = G * gamma_xy
- **Verification:** Volume-averaged stress matches analytical, displacement field is affine
- **Priority:** MEDIUM

### 11.6 Forced Vibration of Plate
- [ ] **File:** `src/femlabpy/examples/dynamic_plate.py`
- **Setup:** Simply-supported plate (Q4), harmonic point load at center
- **Sweep frequency** and compute response amplitude
- **Verification:** FRF peaks coincide with modal analysis frequencies
- **Priority:** MEDIUM

### 11.7 Seismic Ground Motion Response
- [ ] **File:** `src/femlabpy/examples/dynamic_seismic.py`
- **Setup:** Multi-story frame (bar/beam elements) subjected to El Centro earthquake
- **Load:** `p(t) = -M @ r * a_g(t)` where `a_g(t)` = ground acceleration record
- **Solver:** `solve_newmark` with Rayleigh damping (5% at modes 1 and 3)
- **Output:** Roof displacement time history, max inter-story drift
- **Priority:** MEDIUM

---

## Phase 12: Tests

> **Priority:** HIGH for core, MEDIUM for advanced

### 12.1 `tests/test_mass_matrices.py`
- [ ] **Tests:**
  - `test_bar_mass_consistent` — verify shape, symmetry, positive semi-definite
  - `test_bar_mass_lumped` — verify diagonal, total mass = rho*A*L
  - `test_t3_mass_consistent` — verify shape (6,6), symmetry, positive definite
  - `test_t3_mass_total` — sum of diagonal = total mass = rho*t*A
  - `test_q4_mass_consistent` — verify (8,8), symmetry, SPD
  - `test_q4_mass_lumped_preserves_mass` — row-sum lumped preserves total mass
  - `test_t4_mass_consistent` — verify (12,12), total mass = rho*V
  - `test_h8_mass_consistent` — verify (24,24), total mass = rho*V
  - `test_global_mass_assembly` — verify assembled M has correct total mass
  - `test_mass_sparse_dense_agree` — sparse and dense assembly give same result

### 12.2 `tests/test_newmark.py`
- [ ] **Tests:**
  - `test_sdof_undamped_free` — compare with `u(t) = u0*cos(omega_n*t)`, error < 1e-3
  - `test_sdof_damped_forced` — compare with analytical forced response
  - `test_energy_conservation_undamped` — E_total variation < 1e-6 over 1000 steps
  - `test_unconditional_stability` — large dt still converges (doesn't blow up)
  - `test_2nd_order_accuracy` — error ratio at dt vs dt/2 ≈ 4.0
  - `test_linear_acceleration_preset` — beta=1/6 produces correct results
  - `test_zero_initial_conditions` — static load converges to static solution
  - `test_bc_handling` — constrained DOFs remain zero throughout

### 12.3 `tests/test_central_diff.py`
- [ ] **Tests:**
  - `test_sdof_explicit_matches_implicit` — for small dt, same as Newmark
  - `test_conditional_stability_blowup` — dt > dt_cr causes amplitude growth
  - `test_critical_timestep_computation` — dt_cr matches analytical for SDOF
  - `test_requires_lumped_mass` — error raised for consistent mass input
  - `test_wave_speed` — wave front position in bar matches `c = sqrt(E/rho)`
  - **Priority:** MEDIUM

### 12.4 `tests/test_modal.py`
- [ ] **Tests:**
  - `test_sdof_frequency` — `omega = sqrt(k/m)` for single spring-mass
  - `test_bar_frequencies` — compare with analytical `f_n = n/(2L) * sqrt(E/rho)`
  - `test_mass_orthogonality` — `Phi^T M Phi = I` within tolerance
  - `test_stiffness_orthogonality` — `Phi^T K Phi = diag(omega_i^2)`
  - `test_bc_elimination` — constrained DOFs have zero in mode shapes
  - `test_participation_sum` — sum of effective masses = total mass (per direction)
  - `test_n_modes_parameter` — requesting n modes returns exactly n

### 12.5 `tests/test_periodic.py`
- [ ] **Tests:**
  - `test_pair_detection_simple` — 4x4 mesh, known pairs on boundaries
  - `test_pair_detection_symmetric_count` — left face nodes = right face nodes
  - `test_constraint_matrix_rank` — G has correct rank (n_pairs * dof)
  - `test_homogeneous_unit_strain` — uniform material under unit strain gives expected C
  - `test_C_eff_symmetry` — C_eff = C_eff^T within tolerance
  - `test_C_eff_positive_definite` — all eigenvalues of C_eff > 0
  - `test_periodic_vs_large_domain` — C_eff matches large domain (homogeneous case)
  - `test_corner_fixing` — solution doesn't have rigid body modes

### 12.6 `tests/test_hht.py`
- [ ] **Tests:**
  - `test_alpha_zero_is_newmark` — solve_hht(alpha=0) matches solve_newmark(beta=0.25, gamma=0.5)
  - `test_high_freq_dissipation` — high-frequency component amplitude decays with alpha < 0
  - `test_low_freq_preservation` — low-frequency response unaffected by alpha
  - `test_alpha_range_validation` — error for alpha outside [-1/3, 0]
  - **Priority:** MEDIUM

### 12.7 `tests/test_rayleigh_damping.py`
- [ ] **Tests:**
  - `test_rayleigh_formula` — C = alpha*M + beta*K exactly
  - `test_coefficients_at_target_frequencies` — damping ratios match at omega1, omega2
  - `test_sparse_dense_agree` — sparse and dense versions give same C
  - `test_zero_coefficients` — alpha=0, beta=0 gives C=0
  - **Priority:** MEDIUM

### 12.8 `tests/test_energy_balance.py`
- [ ] **Tests:**
  - `test_undamped_energy_constant` — E_k + E_s = const (within 1e-8)
  - `test_damped_energy_decreasing` — E_total strictly decreasing for free vibration with damping
  - `test_external_work_balance` — W_ext = ΔE_k + ΔE_s + E_dissipated
  - `test_energy_at_rest` — E_k = 0, E_s = 0.5*u^T*K*u when velocity = 0
  - **Priority:** MEDIUM

---

## Summary Statistics

| Phase | Tasks | Priority | New Files |
|---|---|---|---|
| 1. Mass Matrices | 9 | HIGH | — (extend existing) |
| 2. Damping Models | 3 | HIGH/MED/LOW | `damping.py` |
| 3. Newmark-Beta | 10 | HIGH | `dynamics.py` |
| 4. Central Difference | 4 | MEDIUM | (in `dynamics.py`) |
| 5. HHT-Alpha | 3 | MEDIUM | (in `dynamics.py`) |
| 6. Modal Analysis | 6 | HIGH | `modal.py` |
| 7. Periodic BCs | 10 | HIGH | `periodic.py` |
| 8. Nonlinear Dynamics | 4 | MEDIUM | (in `dynamics.py`) |
| 9. Post-Processing | 6 | MEDIUM/LOW | (in `plotting.py`) |
| 10. Infrastructure | 6 | HIGH | — (integration) |
| 11. Examples | 7 | HIGH/MED | `examples/dynamic_*.py`, `examples/periodic_*.py` |
| 12. Tests | 8 | HIGH/MED | `tests/test_*.py` |
| **TOTAL** | **76 subtasks** | | **4 new modules** |
