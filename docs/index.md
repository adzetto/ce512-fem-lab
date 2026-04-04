# femlabpy

Welcome to the **femlabpy** documentation!

femlabpy is a pure-Python finite element library. It brings the array-based approach of MATLAB/Scilab FemLab into the Python scientific ecosystem.

```{toctree}
:maxdepth: 2
:hidden:

tutorials
manual/ch01_fundamentals
manual/ch02_elements
manual/ch03_assembly_statics
manual/ch04_materials
manual/ch05_dynamics
manual/ch06_periodic_io
manual/ch07_advanced_examples
manual/ch08_custom_elements
manual/ch09_dynamic_workflows
api
```

## Quick Installation

```bash
pip install femlabpy
```

## Features

- **Elements:** Truss, Plane Stress/Strain (T3, Q4), 3D Solids (T4, H8).
- **Material Models:** Elastic, von Mises Plasticity, Drucker-Prager.
- **Dynamic Analysis:** Modal extraction, Newmark-beta, HHT-alpha time-history integration.
- **Boundary Conditions:** Standard fixities and periodic boundaries for homogenization.
- **I/O:** `gmsh` support for loading 2D and 3D meshes.

Check out the [Tutorials](tutorials.md) or the [API Reference](api.rst) to get started.
