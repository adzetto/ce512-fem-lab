# femlabpy

Welcome to the **femlabpy** documentation!

femlabpy is a pure-Python finite element library. It brings the array-based approach of MATLAB/Scilab FemLab into the Python scientific ecosystem.

```{toctree}
:maxdepth: 2
:hidden:

tutorials
manual/index
theory/index
api
```

```{toctree}
:maxdepth: 2
:caption: Theory & Reference Manual
:hidden:

theory/01_core_assembly
theory/02_boundary_loads
theory/03_1d_bars
theory/04_2d_triangles
theory/05_2d_quads
theory/06_3d_solids
theory/07_plasticity
theory/08_dynamics
theory/09_modal_periodic
theory/10_io_mesh
```

```{toctree}
:maxdepth: 2
:caption: API Reference
:hidden:

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
