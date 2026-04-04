---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Structural Dynamics and Time Integration

This chapter covers the numerical treatment of the equations of motion for structural systems in `femlabpy`. When inertial and viscous forces cannot be neglected, the standard static equilibrium equation $\mathbf{K}\mathbf{u} = \mathbf{f}$ extends to the semidiscrete equation of motion:

$$
\mathbf{M} \ddot{\mathbf{u}}(t) + \mathbf{C} \dot{\mathbf{u}}(t) + \mathbf{K} \mathbf{u}(t) = \mathbf{f}(t)
$$

where $\mathbf{M}$ is the global mass matrix, $\mathbf{C}$ is the global viscous damping matrix, $\mathbf{K}$ is the structural stiffness matrix, and $\mathbf{f}(t)$ is the time-varying external force vector. The vectors $\mathbf{u}$, $\dot{\mathbf{u}}$, and $\ddot{\mathbf{u}}$ represent the nodal displacements, velocities, and accelerations, respectively.

---

## 1. Mass Matrix Formulation

The inertial forces in the structure are governed by the mass matrix $\mathbf{M}$. There are two primary techniques for formulating this matrix in finite element analysis: the **consistent** mass matrix and the **lumped** mass matrix.

### Consistent Mass Matrix

The consistent mass matrix is derived directly from the principle of virtual work, utilizing the exact same shape functions $\mathbf{N}(\xi, \eta)$ that are used to formulate the stiffness matrix. This guarantees that the inertial properties are distributed in a way that is mathematically consistent with the displacement field.

For an element occupying volume $\Omega_e$ with mass density $\rho$, the consistent element mass matrix is given by:

$$
\mathbf{m}_e^{(cons)} = \int_{\Omega_e} \rho \mathbf{N}^\top \mathbf{N} \, d\Omega
$$

Consistent mass matrices are fully populated (non-diagonal) and provide an upper-bound overestimation of the system's natural frequencies. They are highly accurate but computationally expensive because the resulting global mass matrix $\mathbf{M}$ must be factorized alongside the stiffness matrix in dynamic solvers.

### Lumped Mass Matrix

The lumped mass matrix approach concentrates the element mass at the nodal points, resulting in a **diagonal** matrix. This dramatically reduces memory and computational requirements, especially for explicit time integration schemes where a diagonal mass matrix makes the system trivial to invert.

The mathematical formulation often uses a row-sum lumping technique applied to the consistent mass matrix:

$$
m_{ii}^{(lump)} = \sum_{j} m_{ij}^{(cons)}
$$

Alternatively, special integration rules (like nodal quadrature) can automatically yield a diagonal matrix. Lumped mass matrices typically underestimate natural frequencies.

```python
# femlabpy example: mass matrix assembly
from femlabpy.assembly import assemble_system

# 'lumped' or 'consistent' mass formulation
K, M = assemble_system(model, mass_formulation='lumped')
```

---

## 2. Rayleigh Proportional Damping

Modeling the energy dissipation (damping) of a complex structure strictly from physical principles is phenomenologically difficult. A mathematical convenience widely adopted in structural dynamics is **Rayleigh Proportional Damping**, where the damping matrix $\mathbf{C}$ is constructed as a linear combination of the mass and stiffness matrices:

$$
\mathbf{C} = \alpha \mathbf{M} + \beta \mathbf{K}
$$

The scalar coefficients $\alpha$ (mass-proportional) and $\beta$ (stiffness-proportional) are calibrated to yield specified critical damping ratios at specific frequencies. 

### Anchoring to Modal Frequencies

For an undamped $n$-degree-of-freedom system, the free vibration modes $\boldsymbol{\phi}_n$ and natural circular frequencies $\omega_n$ satisfy the eigenvalue problem $(\mathbf{K} - \omega_n^2 \mathbf{M})\boldsymbol{\phi}_n = \mathbf{0}$. If we apply proportional damping and assume the damping matrix is orthogonalized by the same modal matrix, the modal damping ratio $\zeta_n$ for mode $n$ becomes:

$$
\zeta_n = \frac{1}{2} \left( \frac{\alpha}{\omega_n} + \beta \omega_n \right)
$$

Given two target modes $i$ and $j$ with circular frequencies $\omega_i, \omega_j$ and desired damping ratios $\zeta_i, \zeta_j$, we construct a linear system:

$$
\frac{1}{2} \begin{bmatrix} 1/\omega_i & \omega_i \\ 1/\omega_j & \omega_j \end{bmatrix} \begin{Bmatrix} \alpha \\ \beta \end{Bmatrix} = \begin{Bmatrix} \zeta_i \\ \zeta_j \end{Bmatrix}
$$

Solving this yields the required coefficients $\alpha$ and $\beta$. `femlabpy` implements this exact derivation via `femlabpy.damping.rayleigh_coefficients`.

```python
from femlabpy.damping import rayleigh_coefficients

omega_1 = 2.5 * (2 * np.pi)  # rad/s
omega_2 = 10.0 * (2 * np.pi) # rad/s
zeta_1 = 0.05                # 5% damping
zeta_2 = 0.05                # 5% damping

alpha, beta = rayleigh_coefficients(omega_1, omega_2, zeta_1, zeta_2)
print(f"Alpha: {alpha:.4f}, Beta: {beta:.6f}")
```

---

## 3. Implicit Time Integration: The Newmark-$\beta$ Method

To solve the equations of motion in the time domain, `femlabpy` relies heavily on step-by-step numerical integration. The **Newmark-$\beta$** method is one of the most versatile implicit integration schemes in computational mechanics.

The algorithm approximates the velocities and displacements at time $t_{n+1} = t_n + \Delta t$ as:

$$
\dot{\mathbf{u}}_{n+1} = \dot{\mathbf{u}}_n + \Delta t \left[ (1 - \gamma) \ddot{\mathbf{u}}_n + \gamma \ddot{\mathbf{u}}_{n+1} \right]
$$

$$
\mathbf{u}_{n+1} = \mathbf{u}_n + \Delta t \dot{\mathbf{u}}_n + \Delta t^2 \left[ \left( \frac{1}{2} - \beta \right) \ddot{\mathbf{u}}_n + \beta \ddot{\mathbf{u}}_{n+1} \right]
$$

where $\beta$ and $\gamma$ are algorithmic parameters controlling stability and numerical dissipation. 

### Standard Parameter Sets

1. **Average Acceleration** ($\beta = 1/4$, $\gamma = 1/2$): Unconditionally stable, no numerical dissipation. Acceleration is assumed constant across $\Delta t$.
2. **Linear Acceleration** ($\beta = 1/6$, $\gamma = 1/2$): Conditionally stable, second-order accurate. Acceleration varies linearly across $\Delta t$.

### Effective Static System

By isolating $\ddot{\mathbf{u}}_{n+1}$ from the displacement update, we can substitute it into the dynamic equilibrium equation at time $t_{n+1}$:

$$
\mathbf{M} \ddot{\mathbf{u}}_{n+1} + \mathbf{C} \dot{\mathbf{u}}_{n+1} + \mathbf{K} \mathbf{u}_{n+1} = \mathbf{f}_{n+1}
$$

This produces an effective static system:

$$
\mathbf{K}_{eff} \mathbf{u}_{n+1} = \mathbf{f}_{eff, n+1}
$$

where the effective stiffness matrix $\mathbf{K}_{eff}$ incorporates inertial and viscous contributions:

$$
\mathbf{K}_{eff} = \mathbf{K} + a_0 \mathbf{M} + a_1 \mathbf{C}
$$

and the integration constants are defined as $a_0 = \frac{1}{\beta \Delta t^2}$ and $a_1 = \frac{\gamma}{\beta \Delta t}$. The effective load vector $\mathbf{f}_{eff, n+1}$ captures historical states:

$$
\mathbf{f}_{eff, n+1} = \mathbf{f}_{n+1} + \mathbf{M} \left( a_0 \mathbf{u}_n + a_2 \dot{\mathbf{u}}_n + a_3 \ddot{\mathbf{u}}_n \right) + \mathbf{C} \left( a_1 \mathbf{u}_n + a_4 \dot{\mathbf{u}}_n + a_5 \ddot{\mathbf{u}}_n \right)
$$

Because $\mathbf{K}_{eff}$ requires a matrix inversion (factorization), the Newmark method is **implicit** and more expensive per time step than explicit methods, but permits much larger time steps $\Delta t$.

### Python Implementation of Newmark-$\beta$

Here is the exact Python implementation showing how we construct $\mathbf{K}_{eff}$, factorize it once with `scipy.linalg.lu_factor`, and iteratively update displacements, velocities, and accelerations.

```python
import numpy as np
import scipy.linalg as la

def solve_newmark(M, C, K, F, dt, u0, v0, beta=0.25, gamma=0.5):
    """
    Solves dynamic equilibrium using the Newmark-beta method.
    F is an array of shape (ndof, num_steps) containing external forces over time.
    """
    ndof, num_steps = F.shape
    u = np.zeros((ndof, num_steps))
    v = np.zeros((ndof, num_steps))
    a = np.zeros((ndof, num_steps))
    
    # Step 1: Initialize current state
    u[:, 0] = u0
    v[:, 0] = v0
    
    # Evaluate initial acceleration: M * a0 = F0 - C * v0 - K * u0
    rhs0 = F[:, 0] - C @ v0 - K @ u0
    a[:, 0] = la.solve(M, rhs0)
    
    # Step 2: Calculate integration constants
    a0 = 1.0 / (beta * dt**2)
    a1 = gamma / (beta * dt)
    a2 = 1.0 / (beta * dt)
    a3 = 1.0 / (2.0 * beta) - 1.0
    a4 = gamma / beta - 1.0
    a5 = (dt / 2.0) * (gamma / beta - 2.0)
    
    # Step 3: Form effective stiffness matrix K_eff
    K_eff = K + a0 * M + a1 * C
    
    # Step 4: Factorize K_eff once using LU decomposition
    lu, piv = la.lu_factor(K_eff)
    
    # Step 5: Time integration loop
    for n in range(num_steps - 1):
        # Calculate effective force
        F_eff = F[:, n+1] \
              + M @ (a0 * u[:, n] + a2 * v[:, n] + a3 * a[:, n]) \
              + C @ (a1 * u[:, n] + a4 * v[:, n] + a5 * a[:, n])
                          
        # Solve for next displacement u_{n+1}
        u[:, n+1] = la.lu_solve((lu, piv), F_eff)
        
        # Update acceleration and velocity
        a[:, n+1] = a0 * (u[:, n+1] - u[:, n]) - a2 * v[:, n] - a3 * a[:, n]
        v[:, n+1] = v[:, n] + dt * ((1.0 - gamma) * a[:, n] + gamma * a[:, n+1])
        
    return u, v, a
```

---

## 4. Hilber-Hughes-Taylor (HHT-$\alpha$) Method

While Newmark average acceleration is unconditionally stable, its lack of numerical dissipation can lead to spurious high-frequency oscillations that pollute the finite element solution. 

The **HHT-$\alpha$** method generalizes the Newmark scheme by slightly modifying the equilibrium equation to introduce a controllable algorithmic damping that aggressively damps high frequencies while preserving second-order accuracy at low frequencies. 

The modified equilibrium equation at $t_{n+1}$ becomes:

$$
\mathbf{M} \ddot{\mathbf{u}}_{n+1} + (1 + \alpha_H) \mathbf{C} \dot{\mathbf{u}}_{n+1} - \alpha_H \mathbf{C} \dot{\mathbf{u}}_n + (1 + \alpha_H) \mathbf{K} \mathbf{u}_{n+1} - \alpha_H \mathbf{K} \mathbf{u}_n = \mathbf{f}(t_{n+1+\alpha_H})
$$

where the force is typically evaluated as $\mathbf{f}(t_{n+1+\alpha_H}) = (1+\alpha_H)\mathbf{f}_{n+1} - \alpha_H \mathbf{f}_n$.

To ensure unconditional stability, second-order accuracy, and optimal high-frequency dissipation, the Newmark parameters $\beta$ and $\gamma$ are linked to $\alpha_H$:

$$
\alpha_H \in \left[-\frac{1}{3}, 0\right] \quad \implies \quad \beta = \frac{(1 - \alpha_H)^2}{4}, \quad \gamma = \frac{1}{2} - \alpha_H
$$

When $\alpha_H = 0$, the method perfectly reduces to the classical Newmark Average Acceleration method.

### Python Implementation of HHT-$\alpha$

The implementation structure for HHT-$\alpha$ is quite similar to Newmark-$\beta$, but the effective stiffness matrix and effective load vector are updated to reflect the modified equilibrium equation.

```python
import numpy as np
import scipy.linalg as la

def solve_hht(M, C, K, F, dt, u0, v0, alpha=-0.1):
    """
    Solves dynamic equilibrium using the HHT-alpha method.
    The parameter alpha is typically chosen in the range [-1/3, 0].
    """
    # Relate beta and gamma to alpha for unconditional stability and second-order accuracy
    beta = (1 - alpha)**2 / 4.0
    gamma = 0.5 - alpha
    
    ndof, num_steps = F.shape
    u = np.zeros((ndof, num_steps))
    v = np.zeros((ndof, num_steps))
    a = np.zeros((ndof, num_steps))
    
    u[:, 0] = u0
    v[:, 0] = v0
    
    # Initial acceleration
    rhs0 = F[:, 0] - C @ v0 - K @ u0
    a[:, 0] = la.solve(M, rhs0)
    
    a0 = 1.0 / (beta * dt**2)
    a1 = gamma / (beta * dt)
    a2 = 1.0 / (beta * dt)
    a3 = 1.0 / (2.0 * beta) - 1.0
    a4 = gamma / beta - 1.0
    a5 = (dt / 2.0) * (gamma / beta - 2.0)
    
    # HHT-specific effective stiffness matrix K_eff
    K_eff = (1 + alpha) * K + a0 * M + (1 + alpha) * a1 * C
    
    # Factorize K_eff once
    lu, piv = la.lu_factor(K_eff)
    
    for n in range(num_steps - 1):
        # Interpolated external force
        F_interp = (1 + alpha) * F[:, n+1] - alpha * F[:, n]
        
        # Effective force incorporating historical stiffness, mass, and damping terms
        F_eff = F_interp \
              + M @ (a0 * u[:, n] + a2 * v[:, n] + a3 * a[:, n]) \
              + C @ ((1 + alpha) * (a1 * u[:, n] + a4 * v[:, n] + a5 * a[:, n]) + alpha * v[:, n]) \
              + alpha * K @ u[:, n]
              
        # Solve for u_{n+1}
        u[:, n+1] = la.lu_solve((lu, piv), F_eff)
        
        # Standard Newmark updates for acceleration and velocity kinematics
        a[:, n+1] = a0 * (u[:, n+1] - u[:, n]) - a2 * v[:, n] - a3 * a[:, n]
        v[:, n+1] = v[:, n] + dt * ((1.0 - gamma) * a[:, n] + gamma * a[:, n+1])
        
    return u, v, a

# ==========================================
# Runnable Example demonstrating HHT-alpha
# ==========================================
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Define a 1-DOF system
    m = 1.0     # Mass
    k = 100.0   # Stiffness
    c = 0.5     # Damping
    
    M = np.array([[m]])
    C = np.array([[c]])
    K = np.array([[k]])
    
    # Time discretization
    dt = 0.02
    t = np.arange(0, 5, dt)
    num_steps = len(t)
    
    # External load (a square pulse / impulse)
    F = np.zeros((1, num_steps))
    F[0, 10:15] = 100.0 
    
    # Initial conditions
    u0 = np.array([0.0])
    v0 = np.array([0.0])
    
    # Solve system using HHT-alpha algorithm
    u_hht, v_hht, a_hht = solve_hht(M, C, K, F, dt, u0, v0, alpha=-0.1)
    
    # Plot results
    plt.plot(t, u_hht[0], label=r'HHT-$\alpha$ ($\alpha=-0.1$)')
    plt.xlabel('Time [s]')
    plt.ylabel('Displacement [m]')
    plt.title(r'HHT-$\alpha$ Transient Response')
    plt.legend()
    plt.grid(True)
    plt.show()
```

---

## 5. Explicit Time Integration: Central Difference Method

Explicit methods calculate the state at $t_{n+1}$ based solely on historical information, entirely avoiding the matrix factorization of $\mathbf{K}_{eff}$. The most prominent explicit solver in structural dynamics is the **Central Difference** method.

Using standard central difference approximations for derivatives:

$$
\ddot{\mathbf{u}}_n = \frac{\mathbf{u}_{n+1} - 2\mathbf{u}_n + \mathbf{u}_{n-1}}{\Delta t^2}
$$
$$
\dot{\mathbf{u}}_n = \frac{\mathbf{u}_{n+1} - \mathbf{u}_{n-1}}{2\Delta t}
$$

Substituting these into the equilibrium equation at time $t_n$:

$$
\mathbf{M} \left( \frac{\mathbf{u}_{n+1} - 2\mathbf{u}_n + \mathbf{u}_{n-1}}{\Delta t^2} \right) + \mathbf{C} \left( \frac{\mathbf{u}_{n+1} - \mathbf{u}_{n-1}}{2\Delta t} \right) + \mathbf{K} \mathbf{u}_n = \mathbf{f}_n
$$

Solving for the unknown displacement $\mathbf{u}_{n+1}$:

$$
\left( \frac{1}{\Delta t^2} \mathbf{M} + \frac{1}{2\Delta t} \mathbf{C} \right) \mathbf{u}_{n+1} = \mathbf{f}_n - \left( \mathbf{K} - \frac{2}{\Delta t^2} \mathbf{M} \right) \mathbf{u}_n - \left( \frac{1}{\Delta t^2} \mathbf{M} - \frac{1}{2\Delta t} \mathbf{C} \right) \mathbf{u}_{n-1}
$$

### The Power of Lumped Masses

The critical advantage of explicit dynamics emerges when $\mathbf{M}$ and $\mathbf{C}$ are **diagonal matrices** (lumped). In such scenarios, the matrix $\left( \frac{1}{\Delta t^2} \mathbf{M} + \frac{1}{2\Delta t} \mathbf{C} \right)$ is entirely diagonal. Solving for $\mathbf{u}_{n+1}$ reduces to simple, independent scalar divisions per degree of freedom, requiring zero linear algebra factorization.

### Courant-Friedrichs-Lewy (CFL) Stability Limit

The Central Difference method is **conditionally stable**. If the time step $\Delta t$ is too large, the numerical solution will diverge exponentially. The stability criterion demands that the time step satisfy:

$$
\Delta t \le \Delta t_{cr} = \frac{2}{\omega_{max}}
$$

where $\omega_{max}$ is the highest natural circular frequency of the finite element mesh. This is heavily dependent on the smallest element size $L_e$ and the material wave speed $c = \sqrt{E/\rho}$. A common approximation for the critical time step is:

$$
\Delta t_{cr} \approx \min_{e} \left( \frac{L_e}{c} \right)
$$

Because $\omega_{max}$ can be exceedingly high in dense meshes, explicit methods often require thousands or millions of tiny time steps. However, since each step is incredibly cheap computationally (thanks to lumped mass matrices), explicit integration is typically faster than implicit integration for short-duration events like wave propagation, impact, or explosions.

```python
from femlabpy.dynamics import CentralDifferenceSolver

# Note: Always prefer lumped mass matrices for explicit solves!
solver_cd = CentralDifferenceSolver(M_lumped, C_lumped, K)

# Time step must satisfy CFL condition
dt_critical = 2.0 / omega_max
dt = 0.9 * dt_critical 

t_steps, u_history = solver_cd.solve(force_func, dt=dt, t_max=0.5)
```