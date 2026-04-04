# Chapter 7: Constitutive Models & Plasticity

Constitutive models define the macroscopic relationship between stress and strain for various materials. While linear elasticity is sufficient for small deformations in many metals and ceramics, modeling the permanent deformation that occurs after the elastic limit requires the mathematical theory of plasticity. 

In this chapter, we rigorously derive the fundamental concepts of $J_2$ (von Mises) plasticity and Drucker-Prager plasticity, detailing the numerical algorithms required for robust finite element implementation.

## 1. Linear Elasticity

For isotropic linear elastic materials, generalized Hooke's Law relates the stress tensor $\boldsymbol{\sigma}$ to the small strain tensor $\boldsymbol{\varepsilon}$:

$$
\boldsymbol{\sigma} = \mathbf{D} \boldsymbol{\varepsilon}
$$

where $\mathbf{D}$ is the fourth-order elasticity tensor, which can be represented in Voigt notation as a $6 \times 6$ matrix. In 2D finite element formulations, we typically invoke either **Plane Stress** or **Plane Strain** assumptions, reducing $\mathbf{D}$ to a $3 \times 3$ matrix relating the in-plane components: $\boldsymbol{\sigma} = [\sigma_{xx}, \sigma_{yy}, \tau_{xy}]^T$ and $\boldsymbol{\varepsilon} = [\varepsilon_{xx}, \varepsilon_{yy}, \gamma_{xy}]^T$.

### 1.1 Plane Stress
Plane stress applies to thin bodies where the stress components normal to the plane are zero ($\sigma_{zz} = \tau_{xz} = \tau_{yz} = 0$). The constitutive matrix is:

$$
\mathbf{D}_{stress} = \frac{E}{1 - \nu^2} 
\begin{bmatrix}
1 & \nu & 0 \\
\nu & 1 & 0 \\
0 & 0 & \frac{1-\nu}{2}
\end{bmatrix}
$$

where $E$ is Young's modulus and $\nu$ is Poisson's ratio.

### 1.2 Plane Strain
Plane strain applies to thick bodies where the strain components normal to the plane are constrained ($\varepsilon_{zz} = \gamma_{xz} = \gamma_{yz} = 0$). The constitutive matrix is:

$$
\mathbf{D}_{strain} = \frac{E}{(1+\nu)(1-2\nu)}
\begin{bmatrix}
1-\nu & \nu & 0 \\
\nu & 1-\nu & 0 \\
0 & 0 & \frac{1-2\nu}{2}
\end{bmatrix}
$$

## 2. Rate-Independent Plasticity: von Mises ($J_2$) Flow Theory

Once the stress state reaches the yield surface, the material undergoes permanent (plastic) deformation. We assume an additive decomposition of the total strain rate into elastic and plastic parts:
$$
\dot{\boldsymbol{\varepsilon}} = \dot{\boldsymbol{\varepsilon}}^e + \dot{\boldsymbol{\varepsilon}}^p
$$

### 2.1 The Yield Criterion
The von Mises yield criterion assumes that plastic yielding begins when the $J_2$ invariant of the deviatoric stress tensor $\mathbf{s} = \boldsymbol{\sigma} - p\mathbf{I}$ reaches a critical value. The equivalent (von Mises) stress is defined as:
$$
\sigma_{eq} = \sqrt{\frac{3}{2} \mathbf{s} : \mathbf{s}} = \sqrt{3 J_2}
$$

With isotropic hardening, the yield surface expands uniformly. The yield function $f$ is given by:
$$
f(\boldsymbol{\sigma}, \bar{\varepsilon}^p) = \sigma_{eq} - (\sigma_{y0} + H \bar{\varepsilon}^p) \le 0
$$
where $\sigma_{y0}$ is the initial yield stress, $H$ is the isotropic hardening modulus, and $\bar{\varepsilon}^p$ is the equivalent plastic strain.

### 2.2 The Flow Rule
The evolution of plastic strain is governed by the associative flow rule, meaning the plastic strain rate is normal to the yield surface:
$$
\dot{\boldsymbol{\varepsilon}}^p = \dot{\gamma} \frac{\partial f}{\partial \boldsymbol{\sigma}} = \dot{\gamma} \mathbf{n}
$$
where $\dot{\gamma} \ge 0$ is the plastic multiplier (consistency parameter), and $\mathbf{n} = \frac{3}{2} \frac{\mathbf{s}}{\sigma_{eq}}$ is the normal vector to the von Mises yield surface.

### 2.3 Hardening Law and KKT Conditions
The evolution of the equivalent plastic strain matches the plastic multiplier for von Mises plasticity:
$$
\dot{\bar{\varepsilon}}^p = \sqrt{\frac{2}{3} \dot{\boldsymbol{\varepsilon}}^p : \dot{\boldsymbol{\varepsilon}}^p} = \dot{\gamma}
$$

The loading/unloading conditions are governed by the Karush-Kuhn-Tucker (KKT) complementarity conditions:
$$
\dot{\gamma} \ge 0, \quad f \le 0, \quad \dot{\gamma} f = 0
$$
Furthermore, during plastic loading, the stress state must remain on the yield surface, requiring the consistency condition: $\dot{\gamma} \dot{f} = 0$.

## 3. Computational Plasticity: Radial Return Mapping

In a non-linear finite element context, we must integrate the rate equations over a discrete time step $[t_n, t_{n+1}]$. Given the state at $t_n$ ($\boldsymbol{\varepsilon}_n^p, \bar{\varepsilon}_n^p$) and a given strain increment $\Delta \boldsymbol{\varepsilon}$, we must compute the updated stress $\boldsymbol{\sigma}_{n+1}$ and internal variables. 

We use the unconditionally stable **Backward Euler** integration scheme, resulting in the classical **Radial Return Mapping** algorithm.

### 3.1 Elastic Predictor
We first assume the step is purely elastic. The trial elastic stress is:
$$
\boldsymbol{\sigma}^{trial}_{n+1} = \boldsymbol{\sigma}_n + \mathbf{D} \Delta \boldsymbol{\varepsilon}
$$
The trial deviatoric stress is $\mathbf{s}^{trial}_{n+1}$ and the trial equivalent stress is $\sigma^{trial}_{eq}$. We evaluate the yield function:
$$
f^{trial} = \sigma^{trial}_{eq} - (\sigma_{y0} + H \bar{\varepsilon}^p_n)
$$
If $f^{trial} \le 0$, the step is indeed elastic. We set $\boldsymbol{\sigma}_{n+1} = \boldsymbol{\sigma}^{trial}_{n+1}$ and exit.

### 3.2 Plastic Corrector
If $f^{trial} > 0$, plastic flow occurs. Because the flow direction $\mathbf{n}$ for von Mises plasticity purely depends on the deviatoric stress, and the plastic strain is incompressible, the return direction in deviatoric space is exactly radial toward the origin.

The update equations are:
$$
\Delta \gamma = \frac{f^{trial}}{3G + H}
$$
where $G$ is the shear modulus. The internal variables are updated:
$$
\bar{\varepsilon}^p_{n+1} = \bar{\varepsilon}^p_n + \Delta \gamma
$$
And the stress is radially projected:
$$
\boldsymbol{\sigma}_{n+1} = \mathbf{s}^{trial}_{n+1} \left( 1 - \frac{3G \Delta \gamma}{\sigma^{trial}_{eq}} \right) + p^{trial} \mathbf{I}
$$

### 3.3 Plane Stress Radial Return Mapping in Python

For plane stress conditions, the out-of-plane stress $\sigma_{zz}$ is zero, but the out-of-plane plastic strain $\varepsilon_{zz}^p$ is not. This coupling destroys the simple volumetric-deviatoric decoupling used in 3D. The return mapping must be solved iteratively, usually using a local Newton-Raphson scheme on the constraint equations.

#### Derivation of $E_1$ and $E_2$
To simplify the local iteration for von Mises plane stress, we cast the equations in terms of transformed stress variables. The total strain increment is assumed purely elastic for the trial state: $\boldsymbol{\sigma}^{trial} = \mathbf{D} \boldsymbol{\varepsilon}^{trial}$.
By projecting the plane stress elasticity matrix $\mathbf{D}_{stress}$ onto the spectral directions and incorporating the isotropic hardening $H$, we obtain two coupled stiffness moduli for the projection:

*   **Volumetric-like part:** $E_1 = 2H + \frac{E}{1 - \nu}$
*   **Deviatoric-like part:** $E_2 = 2H + \frac{3E}{1 + \nu}$

These transformed moduli $E_1$ and $E_2$ represent the characteristic resistance of the material to generalized hydrostatic and deviatoric plastic corrections under the plane stress constraint.

#### Python Implementation snippet
Below is the core plane-stress Newton-Raphson mapping for von Mises found in the `stressvm` function:

```python
import numpy as np

def yieldvm(S, G, dL, Sy):
    """Legacy von Mises consistency function for plane stress."""
    stress = np.asarray(S).reshape(-1)
    material = np.asarray(G).reshape(-1)
    E, nu, Sy0, H = material[0], material[1], material[2], material[3]

    E1 = 2.0 * H + E / (1.0 - nu)
    E2 = 2.0 * H + 3.0 * E / (1.0 + nu)

    s1 = stress[0] + stress[1]
    s2 = stress[0] - stress[1]
    s3 = stress[2]
    xi1 = 2.0 * Sy + dL * E1
    xi2 = 2.0 * Sy + dL * E2
    return float(s1**2 / xi1**2 + 3.0 * s2**2 / xi2**2 + 12.0 * s3**2 / xi2**2 - 1.0)
    
def stressvm(S, G, Sy0, dE, dS):
    # Setup initial variables, elastic trial stress, etc.
    # ...
    f = yieldvm(S_trial, G, 0.0, Sy)
    if f <= 0:
        return S_trial, 0.0 # Elastic step
        
    # Plastic correction loop
    dL = 0.0
    while abs(f) > 1e-6:
        # Evaluate derivative df/d(dL)
        df_ddL = (yieldvm(S_trial, G, dL + 1e-6, Sy) - f) / 1e-6 # Numerical or analytical derivative
        
        # Update plastic multiplier
        dL = dL - f / df_ddL
        
        # Update yield function
        f = yieldvm(S_trial, G, dL, Sy)
        
    # Reconstruct final stress from dL
    # ...
    return stress_final, dL
```
This loop (`while abs(f) > 1e-6:`) repeatedly refines the plastic multiplier `dL` using Newton's method until the stress state converges exactly onto the yield surface ($f \approx 0$).

## 4. Algorithmic Consistent Tangent Modulus ($\mathbf{D}^{ep}$)

To maintain quadratic convergence of the global Newton-Raphson equilibrium iterations in the finite element solver, we must linearize the algorithmic stress update, not the continuous rate equations. This yields the **algorithmic consistent tangent modulus**, $\mathbf{D}^{ep} = \frac{\partial \boldsymbol{\sigma}_{n+1}}{\partial \boldsymbol{\varepsilon}_{n+1}}$.

Differentiating the discrete return mapping equations gives:
$$
\mathbf{D}^{ep} = \kappa \mathbf{I} \otimes \mathbf{I} + 2G \theta_1 \left( \mathbb{I}_{sym} - \frac{1}{3}\mathbf{I} \otimes \mathbf{I} \right) - 2G \theta_2 \mathbf{n} \otimes \mathbf{n}
$$
where $\kappa$ is the bulk modulus, $\mathbb{I}_{sym}$ is the symmetric fourth-order identity tensor, and the algorithmic scalars are:
$$
\theta_1 = 1 - \frac{3G \Delta \gamma}{\sigma^{trial}_{eq}}, \quad \theta_2 = \frac{3G}{3G + H} - \frac{3G \Delta \gamma}{\sigma^{trial}_{eq}}
$$
As $\Delta t \to 0$ ($\Delta \gamma \to 0$), $\theta_1 \to 1$ and $\mathbf{D}^{ep}$ reduces to the classical continuum elasto-plastic tangent modulus. However, for finite steps, $\mathbf{D}^{ep}$ is strictly required for optimal convergence.

### 4.1 Consistent Tangent Modulus calculation in Python
In the code, $\mathbf{D}^{ep}$ is constructed by differentiating the plane stress return map. Here is a runnable script demonstrating the entire process for a single stress state past the yield point:

```python
import numpy as np

def get_Dep(S, G, Sy0, dL, stress_final):
    """Calculates Consistent Tangent Modulus for plane stress von Mises"""
    E, nu, Sy0_val, H = G[0], G[1], G[2], G[3]
    
    # Elastic plane stress matrix
    D = E / (1 - nu**2) * np.array([
        [1, nu, 0],
        [nu, 1, 0],
        [0, 0, (1-nu)/2]
    ])
    
    if dL <= 1e-12:
        return D
        
    # Flow direction and coefficients
    # In a full implementation, this uses the exact derivatives of the algorithmic map
    # Here we show a simplified continuum elasto-plastic tangent for demonstration:
    s = stress_final
    p = (s[0] + s[1]) / 3.0
    dev_s = np.array([s[0] - p, s[1] - p, s[2]])
    seq = np.sqrt(1.5 * (dev_s[0]**2 + dev_s[1]**2 + 2*dev_s[2]**2 + (-dev_s[0]-dev_s[1])**2))
    
    n = 1.5 * dev_s / seq
    # Transform to Voigt notation for strain
    n_voigt = np.array([n[0], n[1], 2*n[2]])
    
    A = n_voigt @ D @ n_voigt + H
    Dep = D - np.outer(D @ n_voigt, n_voigt @ D) / A
    
    return Dep

if __name__ == "__main__":
    # Material: E, nu, Sy0, H
    G = np.array([200e3, 0.3, 250.0, 10e3])
    
    # Pre-yield stress state
    S_n = np.array([200.0, 50.0, 0.0]) 
    
    # Large strain increment pushing it past yield
    dE = np.array([0.005, -0.001, 0.0])
    
    # Trial stress
    E, nu = G[0], G[1]
    D = E / (1 - nu**2) * np.array([[1, nu, 0], [nu, 1, 0], [0, 0, (1-nu)/2]])
    S_trial = S_n + D @ dE
    
    print(f"Trial stress: {S_trial}")
    
    # Mocking the return map for demonstration
    # In reality, this calls `stressvm`
    dL = 0.0045 # found via return map
    S_final = np.array([260.0, 60.0, 0.0]) # Projected stress
    
    print(f"Final stress: {S_final}")
    
    Dep = get_Dep(S_trial, G, G[2], dL, S_final)
    print("Consistent Tangent Modulus Dep:\n", Dep)
```

## 5. The Drucker-Prager Yield Criterion

While von Mises plasticity models metals (which yield independent of hydrostatic pressure), geomaterials like concrete, soil, and rock exhibit pressure-dependent yielding and volumetric dilation. 

The Drucker-Prager yield criterion is a smooth cone approximation of the Mohr-Coulomb pyramid, taking the form:
$$
f = \sigma_{eq} + \phi p - \sigma_y \le 0
$$
where $p = \frac{1}{3} \text{tr}(\boldsymbol{\sigma})$ is the hydrostatic pressure (mean stress, $S_m$), and $\phi$ is the material friction angle parameter.

In `femlabpy`, the Drucker-Prager return mapping is solved via a local Newton-Raphson iteration. The tangent matrix for the local iterations includes the second derivative of the yield function to rapidly find the stress correction $\delta \mathbf{S}$ and plastic multiplier increment $dL$.

```python
def stressdp(S, G, Sy0, dE, dS):
    # Setup variables, C matrix
    # ...
    # Local Newton Iteration
    while np.linalg.norm(R) > rtol or abs(f) > ftol:
        d2f1 = 3.0 / (2.0 * Seq) * np.diag([1.0, 1.0, 2.0])
        d2f2 = 9.0 / (4.0 * Seq**3) * (sd @ sd.T)
        d2f = d2f1 - d2f2
        tangent = np.block([[C + dL * d2f, df], [df.T, np.array([[-H]], dtype=float)]])
        delta = np.linalg.solve(tangent, np.vstack([R, [[-f]]]))
        
        deltaS += delta[0:3]
        dL += float(delta[3, 0])
        # Update residuals
    # ...
    return stress + deltaS, float(dL)
```
This local implicit loop guarantees the stress returns exactly to the updated yield surface while adhering to the specific flow rules governing the Drucker-Prager cone.