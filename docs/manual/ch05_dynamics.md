# Chapter 5: Structural Dynamics Theory & Time Integration

Structural dynamics extends standard static analysis by incorporating inertial (mass) and damping forces into the global equations of motion. While statics solves $\mathbf{K} \mathbf{u} = \mathbf{P}$, dynamics considers the time-varying equilibrium:

$$ \mathbf{M} \ddot{\mathbf{u}}(t) + \mathbf{C} \dot{\mathbf{u}}(t) + \mathbf{K} \mathbf{u}(t) = \mathbf{P}(t) $$

where $\mathbf{M}$ is the global mass matrix, $\mathbf{C}$ is the viscous damping matrix, and $\ddot{\mathbf{u}}$, $\dot{\mathbf{u}}$, $\mathbf{u}$ are the nodal acceleration, velocity, and displacement vectors, respectively.

## 5.1 Mass Matrices

The global mass matrix $\mathbf{M}$ represents the inertial distribution of the structure. `femlabpy` provides two distinct formulations, each with specific computational advantages.

### 5.1.1 Consistent Mass Matrix
The consistent mass matrix is derived using the exact same shape functions $\mathbf{N}$ used to formulate the stiffness matrix. This ensures that the kinetic energy of the element is represented consistently with the strain energy.

$$ \mathbf{M}_e^c = \int_V \rho \mathbf{N}^T \mathbf{N} \, dV $$

For an isoparametric quadrilateral element (Q4), this integral is evaluated using $2 \times 2$ Gauss-Legendre quadrature:
$$ \mathbf{M}_e^c \approx \sum_{i=1}^{2} \sum_{j=1}^{2} w_i w_j \rho t \mathbf{N}^T(\xi_i, \eta_j) \mathbf{N}(\xi_i, \eta_j) |\mathbf{J}(\xi_i, \eta_j)| $$

*Advantages:* Highly accurate for modal analysis and implicit time integration. It perfectly preserves the off-diagonal coupling between degrees of freedom.

### 5.1.2 Lumped Mass Matrix
A lumped mass matrix $\mathbf{M}_e^l$ is a strictly diagonal matrix. It assumes that the mass of the element is concentrated directly at the nodes, eliminating any inertial coupling between different DOFs.

In `femlabpy`, lumping is often achieved using the **Row-Sum Technique**, where the off-diagonal terms of the consistent mass matrix are summed and added to the diagonal:
$$ M_{ii}^l = \sum_{j=1}^{n} M_{ij}^c $$

*Advantages:* Diagonal mass matrices are computationally cheap to invert ($\mathbf{M}^{-1}$ is simply $1/M_{ii}$). They are strictly required for explicit time-integration algorithms like the Central Difference method.

## 5.2 Modal Analysis Theory

Free, undamped vibration is governed by the homogeneous equation:
$$ \mathbf{M} \ddot{\mathbf{u}}(t) + \mathbf{K} \mathbf{u}(t) = \mathbf{0} $$

Assuming a harmonic solution of the form $\mathbf{u}(t) = \mathbf{\phi} \sin(\omega t)$, we arrive at the generalized eigenvalue problem:
$$ \mathbf{K} \mathbf{\phi}_n = \omega_n^2 \mathbf{M} \mathbf{\phi}_n $$

`femlabpy` abstracts this inside `solve_modal()`.

- **Natural Circular Frequencies ($\omega_n$):** The square roots of the eigenvalues.
- **Natural Frequencies ($f_n$):** $f_n = \frac{\omega_n}{2\pi}$ (in Hertz).
- **Mode Shapes ($\mathbf{\phi}_n$):** The eigenvectors, which are **mass-normalized** by the solver such that:
  $$ \mathbf{\phi}_m^T \mathbf{M} \mathbf{\phi}_n = \delta_{mn} = \begin{cases} 1 & \text{if } m = n \\ 0 & \text{if } m \neq n \end{cases} $$

### Using `solve_modal` in Python

Here is an extensive example of extracting modes and calculating participation factors using `femlabpy`:

```python
import numpy as np
from scipy.linalg import eigh
import matplotlib.pyplot as plt

def solve_modal(K, M, num_modes=3):
    """
    Extracts the lowest `num_modes` natural frequencies and mode shapes.
    Uses `scipy.linalg.eigh` tailored for symmetric matrices.
    """
    # eigh returns eigenvalues and eigenvectors in ascending order
    # Because M is positive definite, we can solve K * phi = lambda * M * phi
    eigenvalues, eigenvectors = eigh(K, M, subset_by_index=[0, num_modes - 1])
    
    # Calculate circular frequencies (omega) and frequencies (f)
    omega = np.sqrt(np.maximum(eigenvalues, 0.0))  # Avoid tiny negative numerical artifacts
    frequencies = omega / (2 * np.pi)
    
    # Mode shapes (phi) are usually automatically mass-normalized by eigh,
    # but let's ensure normalization explicitly:
    for i in range(num_modes):
        phi_i = eigenvectors[:, i]
        modal_mass = phi_i.T @ M @ phi_i
        eigenvectors[:, i] = phi_i / np.sqrt(modal_mass)
        
    return omega, frequencies, eigenvectors

# Usage Example:
# Assuming K and M have already been assembled and boundary conditions applied
omega, freqs, phi = solve_modal(K, M, num_modes=3)

for i in range(3):
    print(f"Mode {i+1}: f = {freqs[i]:.3f} Hz, omega = {omega[i]:.3f} rad/s")
```

### Effective Modal Mass and Participation Factors
To understand how much of the total structural mass is mobilized by a specific mode $n$ in a spatial direction $j$ (e.g., the X-direction), we compute the participation factor $\Gamma_{nj}$:
$$ \Gamma_{nj} = \frac{\mathbf{\phi}_n^T \mathbf{M} \mathbf{r}_j}{\mathbf{\phi}_n^T \mathbf{M} \mathbf{\phi}_n} $$

where $\mathbf{r}_j$ is the influence vector (a vector of 1s and 0s indicating which DOFs are active in direction $j$). If the mode shapes are mass-normalized, the denominator is exactly 1.

The effective modal mass $m_{eff,nj}$ is then:
$$ m_{eff,nj} = \Gamma_{nj}^2 \times (\mathbf{\phi}_n^T \mathbf{M} \mathbf{\phi}_n) = \Gamma_{nj}^2 $$

This is crucial for ensuring that enough modes are extracted in Response Spectrum analysis (typically requiring $\ge 90\%$ mass participation).

## 5.3 Damping Models

Damping $\mathbf{C}$ represents energy dissipation (e.g., internal friction, micro-cracking). It is rarely derived explicitly from element geometry; instead, it is formulated at the global system level.

### 5.3.1 Rayleigh (Proportional) Damping
The most common approach in civil and mechanical engineering is Rayleigh damping, which constructs $\mathbf{C}$ as a linear combination of the mass and stiffness matrices:
$$ \mathbf{C} = \alpha \mathbf{M} + \beta \mathbf{K} $$

Given two target critical damping ratios $\zeta_1, \zeta_2$ at specific circular natural frequencies $\omega_1, \omega_2$, the multipliers $\alpha$ (mass-proportional, affecting low frequencies) and $\beta$ (stiffness-proportional, affecting high frequencies) are found by solving the $2 \times 2$ system:
$$ \begin{bmatrix} \frac{1}{2\omega_1} & \frac{\omega_1}{2} \\ \frac{1}{2\omega_2} & \frac{\omega_2}{2} \end{bmatrix} \begin{Bmatrix} \alpha \\ \beta \end{Bmatrix} = \begin{Bmatrix} \zeta_1 \\ \zeta_2 \end{Bmatrix} $$

### Implementing `rayleigh_damping` in Python

Below is the code block explaining exactly how to compute the coefficients and matrix:

```python
import numpy as np

def rayleigh_coefficients(w1, w2, zeta1=0.05, zeta2=0.05):
    """
    Computes Rayleigh damping coefficients alpha and beta.
    
    Parameters:
    w1, w2: Target circular frequencies (rad/s)
    zeta1, zeta2: Damping ratios at w1 and w2 (default: 5%)
    """
    # Set up the transformation matrix for the 2x2 system
    A = 0.5 * np.array([
        [1/w1, w1],
        [1/w2, w2]
    ])
    rhs = np.array([zeta1, zeta2])
    
    # Solve for [alpha, beta]
    alpha, beta = np.linalg.solve(A, rhs)
    return alpha, beta

def rayleigh_damping(M, K, alpha, beta):
    """
    Assembles the global Rayleigh damping matrix.
    """
    return alpha * M + beta * K

# Example Workflow:
# Assume we extracted natural frequencies w1 and w2 from solve_modal
alpha, beta = rayleigh_coefficients(omega[0], omega[1], zeta1=0.05, zeta2=0.05)
C = rayleigh_damping(M, K, alpha, beta)
print(f"Computed Rayleigh coefficients: alpha = {alpha:.4f}, beta = {beta:.4e}")
```

## 5.4 Time Integration Schemes

To solve the differential equations under arbitrary transient loads $\mathbf{P}(t)$, we use step-by-step numerical integration. `femlabpy` provides three primary solvers.

### 5.4.1 Newmark-$\beta$ Method (Implicit)

The Newmark method approximates the displacement and velocity at time $t+\Delta t$ using parameters $\gamma$ and $\beta$:
$$ \mathbf{u}_{t+\Delta t} = \mathbf{u}_t + \Delta t \dot{\mathbf{u}}_t + \Delta t^2 \left[ (0.5 - \beta)\ddot{\mathbf{u}}_t + \beta \ddot{\mathbf{u}}_{t+\Delta t} \right] $$
$$ \dot{\mathbf{u}}_{t+\Delta t} = \dot{\mathbf{u}}_t + \Delta t \left[ (1-\gamma)\ddot{\mathbf{u}}_t + \gamma \ddot{\mathbf{u}}_{t+\Delta t} \right] $$

This leads to an effective static linear system solved at each time step:
$$ \mathbf{K}_{eff} \mathbf{u}_{t+\Delta t} = \mathbf{P}_{eff, t+\Delta t} $$
where:
$$ \mathbf{K}_{eff} = \mathbf{K} + a_0 \mathbf{M} + a_1 \mathbf{C} $$
$$ \mathbf{P}_{eff} = \mathbf{P}_{t+\Delta t} + \mathbf{M}(a_0 \mathbf{u}_t + a_2 \dot{\mathbf{u}}_t + a_3 \ddot{\mathbf{u}}_t) + \mathbf{C}(a_1 \mathbf{u}_t + a_4 \dot{\mathbf{u}}_t + a_5 \ddot{\mathbf{u}}_t) $$

*Note: Using $\beta=0.25$ and $\gamma=0.5$ results in the Unconditionally Stable Average Acceleration method (Trapezoidal Rule), meaning the time step $\Delta t$ is chosen for accuracy, not stability.*

#### The $K_{eff}$ Factorization Strategy using `scipy.linalg.lu_factor`

A critical optimization in linear implicit dynamics (where $\mathbf{M}$, $\mathbf{C}$, $\mathbf{K}$, and $\Delta t$ are constant) is how $\mathbf{K}_{eff}$ is handled. 

Because $\mathbf{K}_{eff}$ does not change from one time step to the next, **we do not need to invert or solve it from scratch every step.** Doing so would cost $\mathcal{O}(N^3)$ operations per time step. 

Instead, we factorize $\mathbf{K}_{eff}$ **exactly once** outside the time integration loop using `scipy.linalg.lu_factor`. LU decomposition splits the matrix into Lower and Upper triangular matrices: $\mathbf{K}_{eff} = \mathbf{L}\mathbf{U}$. Inside the loop, we simply perform forward and backward substitution using `scipy.linalg.lu_solve`. This drops the per-step cost from $\mathcal{O}(N^3)$ to just $\mathcal{O}(N^2)$!

#### Line-by-Line Python Explanation of the Newmark Loop

Below is the complete implementation of `solve_newmark`. Pay close attention to how the inner loop iteratively updates vectors.

```python
import numpy as np
from scipy.linalg import lu_factor, lu_solve

def solve_newmark(M, C, K, P_history, dt, u0, v0, beta=0.25, gamma=0.5):
    """
    Solves dynamic structural response using the implicit Newmark-beta method.
    """
    n_dofs = M.shape[0]
    n_steps = P_history.shape[1]
    
    # 1. Integration constants
    a0 = 1.0 / (beta * dt**2)
    a1 = gamma / (beta * dt)
    a2 = 1.0 / (beta * dt)
    a3 = 1.0 / (2.0 * beta) - 1.0
    a4 = gamma / beta - 1.0
    a5 = (dt / 2.0) * (gamma / beta - 2.0)
    
    # 2. Form effective stiffness matrix
    K_eff = K + a0 * M + a1 * C
    
    # 3. LU Factorization (Done ONCE outside the loop!)
    # This stores the L and U matrices and pivot indices.
    # Tremendous performance gain for constant dt linear systems.
    lu_piv = lu_factor(K_eff)
    
    # 4. Initialize response arrays
    U = np.zeros((n_dofs, n_steps))
    V = np.zeros((n_dofs, n_steps))
    A = np.zeros((n_dofs, n_steps))
    
    # Initial conditions
    U[:, 0] = u0
    V[:, 0] = v0
    
    # Solve for initial acceleration A0 from equilibrium: M*A0 = P0 - C*V0 - K*U0
    # Assuming initial external load P[:, 0] is provided
    rhs_0 = P_history[:, 0] - C @ v0 - K @ u0
    A[:, 0] = np.linalg.solve(M, rhs_0) 
    
    # 5. Time Integration Loop
    for n in range(0, n_steps - 1):
        # Extract variables from time t (step n)
        u_t = U[:, n]
        v_t = V[:, n]
        a_t = A[:, n]
        
        # External load at time t + dt (step n+1)
        p_next = P_history[:, n+1]
        
        # Form effective load vector for t + dt
        # This vector captures the external force plus the inertial and damping memory 
        # from the previous timestep.
        p_eff = (p_next 
                 + M @ (a0 * u_t + a2 * v_t + a3 * a_t) 
                 + C @ (a1 * u_t + a4 * v_t + a5 * a_t))
        
        # Solve for next displacement u_{n+1} using the pre-computed LU factorization.
        # This is extremely fast (O(N^2)).
        u_next = lu_solve(lu_piv, p_eff)
        
        # Update next acceleration a_{n+1} and velocity v_{n+1}
        # using the Newmark difference equations.
        a_next = a0 * (u_next - u_t) - a2 * v_t - a3 * a_t
        v_next = v_t + a4 * v_t + a5 * a_t + a1 * (u_next - u_t) - a4 * v_t - a5 * a_t # Simplified form:
        v_next = v_t + dt * ((1.0 - gamma) * a_t + gamma * a_next)
        
        # Store results for this timestep
        U[:, n+1] = u_next
        V[:, n+1] = v_next
        A[:, n+1] = a_next
        
    return U, V, A
```

### 5.4.2 Central Difference Method (Explicit)
For high-speed impact, blast, or wave propagation, the explicit central difference method is used. It evaluates equilibrium entirely at time $t$:
$$ \mathbf{M} \ddot{\mathbf{u}}_t + \mathbf{C} \dot{\mathbf{u}}_t + \mathbf{K} \mathbf{u}_t = \mathbf{P}_t $$
Using finite difference approximations:
$$ \ddot{\mathbf{u}}_t = \frac{\mathbf{u}_{t+\Delta t} - 2\mathbf{u}_t + \mathbf{u}_{t-\Delta t}}{\Delta t^2} $$
Substituting this yields a system where $\mathbf{u}_{t+\Delta t}$ can be solved trivially **without matrix inversion**, provided $\mathbf{M}$ is a lumped (diagonal) matrix and $\mathbf{C}$ is zero or diagonal.

*Stability Limit:* Explicit methods are conditionally stable. The time step must strictly obey the Courant-Friedrichs-Lewy (CFL) condition:
$$ \Delta t \le \frac{2}{\omega_{max}} $$
where $\omega_{max}$ is the highest natural frequency of the mesh. `femlabpy` provides `critical_timestep(K, M)` to estimate this limit via power iteration.

## 5.5 Runnable Example: Implicit vs Explicit on a SDOF System

To truly understand the difference between implicit (Newmark) and explicit (Central Difference) methods, consider a 1-Degree-of-Freedom (SDOF) spring-mass system subjected to a step load. 

The implicit method is unconditionally stable regardless of the time step, while the explicit method will violently blow up if the timestep $\Delta t > 2/\omega$.

```python
import numpy as np
import matplotlib.pyplot as plt

# SDOF Properties
m = 1.0       # Mass (kg)
k = 39.478    # Stiffness (N/m) -> natural freq f = 1.0 Hz, w = 2*pi
c = 0.0       # Undamped
omega = np.sqrt(k/m)
critical_dt = 2.0 / omega
print(f"Critical time step: {critical_dt:.4f} s")

# Time vectors
t_end = 5.0
dt_stable = 0.05    # Stable for both
dt_unstable = 0.35  # Unstable for Explicit! (0.35 > 0.318)

def solve_sdof_explicit(m, k, c, p_func, dt, t_end):
    """ Central Difference Explicit Solver for SDOF """
    times = np.arange(0, t_end, dt)
    u = np.zeros(len(times))
    v = np.zeros(len(times))
    a = np.zeros(len(times))
    
    # Initial conditions
    u[0], v[0] = 0.0, 0.0
    a[0] = (p_func(0) - c*v[0] - k*u[0]) / m
    
    # Step -1 calculation
    u_prev = u[0] - dt*v[0] + 0.5*dt**2*a[0]
    
    # Effective mass (scalar)
    m_eff = m / dt**2 + c / (2*dt)
    
    for i in range(len(times)-1):
        p_t = p_func(times[i])
        
        # Effective force
        if i == 0:
            p_eff = p_t - (k - 2*m/dt**2)*u[i] - (m/dt**2 - c/(2*dt))*u_prev
        else:
            p_eff = p_t - (k - 2*m/dt**2)*u[i] - (m/dt**2 - c/(2*dt))*u[i-1]
            
        u[i+1] = p_eff / m_eff
        
    return times, u

def solve_sdof_implicit(m, k, c, p_func, dt, t_end):
    """ Newmark-beta Implicit Solver for SDOF (beta=0.25, gamma=0.5) """
    times = np.arange(0, t_end, dt)
    u, v, a = np.zeros(len(times)), np.zeros(len(times)), np.zeros(len(times))
    
    u[0], v[0] = 0.0, 0.0
    a[0] = (p_func(0) - c*v[0] - k*u[0]) / m
    
    beta, gamma = 0.25, 0.5
    a0 = 1/(beta*dt**2)
    a1 = gamma/(beta*dt)
    a2 = 1/(beta*dt)
    a3 = 1/(2*beta) - 1
    
    k_eff = k + a0*m + a1*c
    
    for i in range(len(times)-1):
        p_next = p_func(times[i+1])
        p_eff = p_next + m*(a0*u[i] + a2*v[i] + a3*a[i])
        
        u[i+1] = p_eff / k_eff
        a[i+1] = a0*(u[i+1] - u[i]) - a2*v[i] - a3*a[i]
        v[i+1] = v[i] + dt*((1-gamma)*a[i] + gamma*a[i+1])
        
    return times, u

# Step Load function
def step_load(t):
    return 10.0 if t >= 0 else 0.0

# 1. Run Stable Case
t_imp, u_imp = solve_sdof_implicit(m, k, c, step_load, dt_stable, t_end)
t_exp, u_exp = solve_sdof_explicit(m, k, c, step_load, dt_stable, t_end)

# 2. Run Unstable Case (Explicit will blow up)
t_imp_unstable, u_imp_unstable = solve_sdof_implicit(m, k, c, step_load, dt_unstable, t_end)
try:
    t_exp_unstable, u_exp_unstable = solve_sdof_explicit(m, k, c, step_load, dt_unstable, t_end)
except OverflowError:
    pass # Expected blow up

# Plotting could be done here showing the perfectly stable Newmark curve vs the exponentially exploding Central Difference curve.
```

This demonstrates why implicit solvers are preferred for long-duration structural dynamics (like earthquake engineering), whereas explicit solvers are reserved for extremely short-duration blast/impact problems.