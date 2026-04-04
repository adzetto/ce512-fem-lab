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

# Chapter 11: Nonlinear Solvers

Welcome to Chapter 11. As a computational mechanics professor, I will guide you through the intricate world of nonlinear solvers, focusing on two pivotal methods utilized in our `femlab-python` repository: the Orthogonal Residual Method (often referred to as Arc-Length) in `solve_nlbar`, and the standard Newton-Raphson scheme in `solve_plastic`.

## 11.1 The Orthogonal Residual Method (`solve_nlbar`)

When structures undergo severe nonlinearities, such as snap-through or snap-back buckling, load-control and displacement-control methods fail. We must allow both the load parameter, $\lambda$, and the displacement vector, $\mathbf{u}$, to vary simultaneously. This is where Arc-Length methods come into play.

In our `solve_nlbar` implementation, we employ the Orthogonal Residual Method. The equilibrium equation is:
$$ \mathbf{r}(\mathbf{u}, \lambda) = \mathbf{f}_{int}(\mathbf{u}) - \lambda \mathbf{f}_{ext} = \mathbf{0} $$

For a given iteration, the change in displacement is decomposed into two parts:
$$ \Delta \mathbf{u} = \Delta \mathbf{u}_r + \Delta \lambda \Delta \mathbf{u}_f $$
where:
* $\Delta \mathbf{u}_r = \mathbf{K}_T^{-1} (-\mathbf{r})$ is the residual displacement due to out-of-balance forces.
* $\Delta \mathbf{u}_f = \mathbf{K}_T^{-1} \mathbf{f}_{ext}$ is the forward displacement due to external loads.

### Enforcing Orthogonality

To solve for the unknown load increment $\Delta \lambda$, we constrain the iterative change in displacement to be orthogonal to the previous step displacement. That is, we enforce:
$$ \Delta \mathbf{u}_i^T \Delta \mathbf{u}_{i+1} = 0 $$

In Python, this constraint allows us to predict the load increment factor `dlambda` ($\Delta \lambda$) and the updated displacement `du` ($\Delta \mathbf{u}$):

```python
# Tangent stiffness matrix K_T, internal forces f_int, external load f_ext
# Compute residual
residual = f_int - lambda_current * f_ext

# Solve for the two displacement components
du_r = np.linalg.solve(K_T, -residual)
du_f = np.linalg.solve(K_T, f_ext)

# Predict dlambda by enforcing the orthogonality condition: du_prev.T @ du_new = 0
# where du_new = du_r + dlambda * du_f
# Therefore: du_prev.T @ (du_r + dlambda * du_f) = 0
dlambda = -(du_prev.T @ du_r) / (du_prev.T @ du_f)

# Predict updated displacement increment
du_new = du_r + dlambda * du_f

# Update totals
lambda_current += dlambda
u_current += du_new
```

## 11.2 The Standard Newton-Raphson Scheme (`solve_plastic`)

For material nonlinearities such as plasticity (without path-instabilities), the standard Newton-Raphson algorithm remains our workhorse, as seen in `solve_plastic`.

We seek to find $\mathbf{u}$ such that the residual vanishes:
$$ \mathbf{R}(\mathbf{u}) = \mathbf{F}_{int}(\mathbf{u}) - \mathbf{F}_{ext} = \mathbf{0} $$

Using a Taylor series expansion, the update equation is:
$$ \mathbf{K}_T \Delta \mathbf{u} = -\mathbf{R}(\mathbf{u}) $$

where $\mathbf{K}_T = \frac{\partial \mathbf{F}_{int}}{\partial \mathbf{u}}$ is the algorithmic tangent stiffness matrix.

### 1D Nonlinear Spring Example

To crystallize this concept, here is a runnable Python script of a 1D nonlinear spring with stiffness $k(x) = k_0 + \alpha x^2$. The internal force is $F_{int}(x) = k_0 x + \frac{1}{3}\alpha x^3$.

``` python
import numpy as np

def solve_1d_spring(F_ext, k0=10.0, alpha=5.0, tol=1e-6, max_iter=20):
    """Solves a 1D nonlinear spring using Newton-Raphson."""
    x = 0.0  # Initial guess
    
    print(f"Solving for F_ext = {F_ext}")
    print("-" * 30)
    
    for i in range(max_iter):
        # Internal force and Tangent stiffness
        F_int = k0 * x + (1.0/3.0) * alpha * x**3
        K_T = k0 + alpha * x**2
        
        # Residual
        R = F_int - F_ext
        
        print(f"Iter {i}: x = {x:.6f}, R = {R:.2e}")
        
        if abs(R) < tol:
            print(f"Converged in {i} iterations.\n")
            return x
            
        # Newton-Raphson update
        dx = -R / K_T
        x += dx
        
    raise RuntimeError("Newton-Raphson failed to converge.")

# Run the script
if __name__ == "__main__":
    final_x = solve_1d_spring(F_ext=25.0)
    print(f"Final displacement: x = {final_x:.6f}")
```
