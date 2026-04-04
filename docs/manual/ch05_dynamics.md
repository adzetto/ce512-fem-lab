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

In Python:
```python
from femlabpy.damping import rayleigh_coefficients, rayleigh_damping

# Anchor 5% damping to modes 1 and 2
alpha, beta = rayleigh_coefficients(w1, w2, zeta1=0.05, zeta2=0.05)
C = rayleigh_damping(M, K, alpha, beta)
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

### 5.4.2 HHT-$\alpha$ Method (Implicit)
The Hilber-Hughes-Taylor (HHT-$\alpha$) method is an extension of Newmark that introduces numerical dissipation to damp out spurious high-frequency oscillations without degrading second-order accuracy.
It modifies the equilibrium equation:
$$ \mathbf{M} \ddot{\mathbf{u}}_{t+\Delta t} + (1+\alpha)\mathbf{C}\dot{\mathbf{u}}_{t+\Delta t} - \alpha\mathbf{C}\dot{\mathbf{u}}_t + (1+\alpha)\mathbf{K}\mathbf{u}_{t+\Delta t} - \alpha\mathbf{K}\mathbf{u}_t = (1+\alpha)\mathbf{P}_{t+\Delta t} - \alpha\mathbf{P}_t $$
where $\alpha \in [-1/3, 0]$. Setting $\alpha=0$ recovers the standard Newmark method.

### 5.4.3 Central Difference Method (Explicit)
For high-speed impact, blast, or wave propagation, the explicit central difference method is used. It evaluates equilibrium entirely at time $t$:
$$ \mathbf{M} \ddot{\mathbf{u}}_t + \mathbf{C} \dot{\mathbf{u}}_t + \mathbf{K} \mathbf{u}_t = \mathbf{P}_t $$
Using finite difference approximations:
$$ \ddot{\mathbf{u}}_t = \frac{\mathbf{u}_{t+\Delta t} - 2\mathbf{u}_t + \mathbf{u}_{t-\Delta t}}{\Delta t^2} $$
Substituting this yields a system where $\mathbf{u}_{t+\Delta t}$ can be solved trivially **without matrix inversion**, provided $\mathbf{M}$ is a lumped (diagonal) matrix and $\mathbf{C}$ is zero or diagonal.

*Stability Limit:* Explicit methods are conditionally stable. The time step must strictly obey the Courant-Friedrichs-Lewy (CFL) condition:
$$ \Delta t \le \frac{2}{\omega_{max}} $$
where $\omega_{max}$ is the highest natural frequency of the mesh. `femlabpy` provides `critical_timestep(K, M)` to estimate this limit via power iteration.