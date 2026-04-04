# Chapter 4: Material Models

`femlabpy` supports a variety of constitutive models, mapped through the material matrix $\mathbf{D}$ relating stress to strain: $\sigma = \mathbf{D} \epsilon$. This chapter outlines the theoretical foundations for elasticity and plasticity implementations.

## 4.1 Linear Elasticity

For an isotropic linear elastic material, the compliance tensor depends solely on Young's Modulus $E$ and Poisson's ratio $\nu$. In 2D formulations, we distinguish between two states:

### Plane Stress
Assume $\sigma_z = \tau_{xz} = \tau_{yz} = 0$. The material stiffness matrix is:
$$ \mathbf{D} = \frac{E}{1 - \nu^2} \begin{bmatrix}
1 & \nu & 0 \\
\nu & 1 & 0 \\
0 & 0 & \frac{1-\nu}{2}
\end{bmatrix} $$

### Plane Strain
Assume $\epsilon_z = \gamma_{xz} = \gamma_{yz} = 0$. The material stiffness matrix is:
$$ \mathbf{D} = \frac{E}{(1+\nu)(1-2\nu)} \begin{bmatrix}
1-\nu & \nu & 0 \\
\nu & 1-\nu & 0 \\
0 & 0 & \frac{1-2\nu}{2}
\end{bmatrix} $$

## 4.2 von Mises Elastoplasticity

For ductile metals, the von Mises yield criterion is employed with isotropic hardening.

### Yield Function
The yield function $f$ defines the elastic limit based on the deviatoric stress invariants:
$$ f(\sigma, \bar{\epsilon}_p) = \sigma_{eq} - (\sigma_y + H \bar{\epsilon}_p) \le 0 $$
where $\sigma_{eq} = \sqrt{\frac{3}{2} \mathbf{s}:\mathbf{s}}$ is the equivalent von Mises stress, $\mathbf{s}$ is the deviatoric stress tensor, $\sigma_y$ is the initial yield stress, $H$ is the hardening modulus, and $\bar{\epsilon}_p$ is the accumulated equivalent plastic strain.

### Radial Return Mapping Algorithm
When the elastic trial stress violates the yield condition ($f_{trial} > 0$), a backward-Euler integration scheme (Radial Return) is executed at the Gauss point level to return the stress to the yield surface. In `femlabpy`, the scalar nonlinear yield equation is solved using Newton-Raphson iterations. 

The exact Python logic for the von Mises radial return mapping loop involves a `while` loop computing the root of the consistency parameter $\Delta\gamma$ (`dL`):

```python
    dL = 0.0
    f = yieldvm(stress, material, dL, Sy)
    while abs(f) > 1.0e-6:
        df = dyieldvm(stress, material, dL, Sy)
        dL -= f / df
        f = yieldvm(stress, material, dL, Sy)
```

## 4.3 Drucker-Prager Plasticity

For soils and concrete, the yield strength depends heavily on hydrostatic pressure. The Drucker-Prager model approximates the Mohr-Coulomb failure surface using the invariants $I_1$ (pressure) and $J_2$ (deviatoric shear).

$$ f(\sigma) = \sqrt{J_2} + \alpha I_1 - k \le 0 $$

### Consistent Tangent Modulus
To preserve the quadratic convergence of the global Newton-Raphson iterations, the algorithmic consistent tangent stiffness $\mathbf{D}^{ep}$ must be computed, modifying the elastic $\mathbf{D}$ matrix based on the normal flow direction $\mathbf{n}$. 

In the Drucker-Prager local Newton loop, the generalized local system solves for both stress corrections and the plastic multiplier. The tangent matrix is assembled beautifully in Python using `np.block`:

```python
        d2f1 = 3.0 / (2.0 * Seq) * np.diag([1.0, 1.0, 2.0])
        d2f2 = 9.0 / (4.0 * Seq**3) * (sd @ sd.T)
        d2f = d2f1 - d2f2
        # Assembling the local elastoplastic tangent stiffness via block matrix
        tangent = np.block([[C + dL * d2f, df], [df.T, np.array([[-H]], dtype=float)]])
        delta = np.linalg.solve(tangent, np.vstack([R, [[-f]]]))
```

---

## 4.4 Tutorial: `src/femlabpy/materials/plasticity.py` Line-by-Line

To master computational plasticity implementations, we will break down the `femlabpy` plasticity module.

### 1. `yieldvm` and `dyieldvm`
```python
def yieldvm(S, G, dL, Sy):
    ...
    stress = as_float_array(S).reshape(-1)
    material = as_float_array(G).reshape(-1)
    E = material[0]
    nu = material[1]
    H = material[3]
```
These lines flatten the stress and material vectors and extract Young's Modulus ($E$), Poisson's ratio ($\nu$), and Hardening modulus ($H$).

```python
    E1 = 2.0 * H + E / (1.0 - nu)
    E2 = 2.0 * H + 3.0 * E / (1.0 + nu)
```
These are the effective algorithmic elastic moduli adapted for plane stress kinematics.

```python
    s1 = stress[0] + stress[1]
    s2 = stress[0] - stress[1]
    s3 = stress[2]
    xi1 = 2.0 * Sy + dL * E1
    xi2 = 2.0 * Sy + dL * E2
    return float(s1**2 / xi1**2 + 3.0 * s2**2 / xi2**2 + 12.0 * s3**2 / xi2**2 - 1.0)
```
The stress components are transformed. `xi1` and `xi2` represent denominators in the implicit return equation. Finally, the residual $f(\Delta \gamma)$ is returned. `dyieldvm` acts analogously but evaluates the explicit analytical derivative `df`.

### 2. `stressvm`
```python
def stressvm(S, G, Sy):
    ...
    dL = 0.0
    f = yieldvm(stress, material, dL, Sy)
    while abs(f) > 1.0e-6:
        df = dyieldvm(stress, material, dL, Sy)
        dL -= f / df
        f = yieldvm(stress, material, dL, Sy)
```
This is the core standard Newton-Raphson scheme! We initialize the plastic multiplier $\Delta\gamma = 0$. If $f \le 10^{-6}$, it's elastic. If $f > 0$, we deduct $f/f'$ from `dL` until the yield surface residual shrinks to near-zero.

```python
    Sy = Sy + H * dL
    E1 = E / (1.0 - nu)
    ...
    stress[0] = 0.5 * (s1 + s2)
```
Following the converged `dL`, the final yield stress is updated using linear isotropic hardening. The individual stress components are updated via the radial return update rule.

### 3. `stressdp`
```python
def stressdp(S, G, Sy0, dE, dS):
...
    C = (1.0 / E) * np.array([
            [1.0, -nu, 0.0],
            [-nu, 1.0, 0.0],
            [0.0, 0.0, 2.0 * (1.0 + nu)],
        ], dtype=float)
```
`C` is the purely elastic compliance matrix.

```python
    while np.linalg.norm(R) > rtol or abs(f) > ftol:
        d2f1 = ...
        d2f2 = ...
        d2f = d2f1 - d2f2
        tangent = np.block([[C + dL * d2f, df], [df.T, np.array([[-H]], dtype=float)]])
        delta = np.linalg.solve(tangent, np.vstack([R, [[-f]]]))
        deltaS += delta[0:3]
        dL += float(delta[3, 0])
```
Unlike von Mises where we only iterated a scalar equation, Drucker-Prager utilizes a multi-variable local Newton iteration. `R` represents the integration point strain residual, and `f` is the yield function residual. They are solved simultaneously for stress increments `deltaS` and multiplier `dL` by factoring the symmetric Jacobian matrix assembled via `np.block`.

---

## 4.5 Runnable Example: Loading a Single Integration Point to Yield

Here is a standalone script that loads an elastic integration point beyond the yield strength to showcase the `stressvm` return mapping logic:

```python
import numpy as np
from femlabpy.materials.plasticity import stressvm, yieldvm

# Material properties: [E, nu, Sy0, H]
# Young's Modulus = 200e3 MPa, Poisson = 0.3, Yield = 250 MPa, Hardening = 2e3 MPa
G = np.array([200000.0, 0.3, 250.0, 2000.0])
Sy = G[2]

# Trial stress (e.g. from an elastic predictor step): [sigma_xx, sigma_yy, tau_xy]
# Let's apply a uniaxial tension well past the 250 MPa yield limit.
S_trial = np.array([400.0, 0.0, 0.0])

# Check initial yield condition (with dL = 0)
f_initial = yieldvm(S_trial, G, 0.0, Sy)
print(f"Initial Yield Residual: {f_initial:.4f}") 
# If > 0, it means the state is illegal and outside the yield surface

if f_initial > 0:
    print("Trial stress exceeds yield surface. Performing return mapping...")
    
    # Perform radial return mapping
    S_returned, dL = stressvm(S_trial, G, Sy)
    
    print("\n--- Results ---")
    print(f"Converged plastic multiplier (dL): {dL:.6f}")
    print(f"New Yield Stress: {Sy + G[3] * dL:.2f} MPa")
    print(f"Returned Stress [Sxx, Syy, Sxy]:")
    print(np.round(S_returned.flatten(), 2))
    
    # Verification
    f_final = yieldvm(S_returned, G, 0.0, Sy + G[3] * dL)
    print(f"Final Yield Residual (should be ~0): {f_final:.2e}")
```
