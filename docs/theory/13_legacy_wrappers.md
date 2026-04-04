---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.1
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# 13. Legacy Wrappers for MATLAB Compatibility

Welcome, class. Today we will discuss how our Python finite element framework, `femlabpy`, provides backward compatibility with legacy MATLAB scripts. 

Historically, teaching codes in MATLAB often bundled the entire finite element solution process into a single, high-level function call. To ease the transition for those accustomed to this pedagogical style, `matlab.py` provides wrapper functions that emulate these monolithic "1-line solvers" while leveraging our modern, modular Python core architecture.

## The 1-Line Solvers

In our MATLAB heritage, it was common to solve a 2D elasticity problem simply by passing the required matrices into a single function, yielding the displacements and elemental stresses. 

In `femlabpy`, we replicate this behavior with the `elastic(T, X, G, C, P)` wrapper. Rather than containing the dense procedural logic itself, the Python version intelligently delegates the work to our modular routines: initialization, stiffness assembly, boundary condition application, solving the linear system, and recovering stresses/strains.

### Inside the `elastic` Wrapper

To demystify the magic, let us look at the exact Python code inside the `elastic` wrapper. Notice how it seamlessly orchestrates the core functions:

```python
def elastic(T, X, G, C, P):
    """Legacy MATLAB-style wrapper for 2D elasticity."""
    # Initialize the displacement and force vectors
    U, F = init(T, X)
    
    # Assemble the global stiffness matrix for 4-node quadrilaterals
    K = kq4e(T, X, G, C)
    
    # Apply boundary conditions
    K, F = setbc(K, F, P)
    
    # Solve the linear system [K]{U} = {F}
    U, F = solve(K, F, U)
    
    # Recover element responses (strains and stresses)
    E = qq4e(T, X, U, G, C)
    
    return U, E
```

By decomposing the procedure this way, the core finite element routines (`init`, `kq4e`, `setbc`, `solve`, and `qq4e`) can be independently tested and optimized, while still presenting you with the familiar, simple interface.

## MATLAB 1-Based vs. Python 0-Based Indexing

A critical issue you will encounter when migrating data or translating conceptual models from MATLAB to Python is the difference in array indexing conventions.

*   **MATLAB uses 1-based indexing:** The first element of an array or matrix is at index `1`. Node IDs and element numbers in topology matrices (`T`) or boundary condition specifications (`P` or `C`) start at `1`.
*   **Python (NumPy) uses 0-based indexing:** The first element of an array is at index `0`. 

**Implications for FEM Data:**
When you pass legacy matrices directly into Python, the node indices will be off by one. For example, if an element connects to the first node in the mesh, MATLAB records this as node `1`. Python expects this to refer to index `0` of the coordinate array `X`.

**Correction Strategy:**
The `matlab.py` wrappers automatically handle this translation. However, if you are writing custom Python scripts or calling the core modular functions directly, you must manually shift your indices. 

For instance, correcting a topology matrix looks like this:

```python
# Shifting a 1-based MATLAB topology matrix to 0-based for Python
T_python = T_matlab - 1 
```

Always be meticulously aware of the indexing convention when moving between these two environments to avoid out-of-bounds errors or physically incorrect assemblies.