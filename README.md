# femlabpy

[![PyPI version](https://badge.fury.io/py/femlabpy.svg)](https://pypi.org/project/femlabpy/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/github/actions/workflow/status/adzetto/femlabpy/tests.yml?branch=main&label=tests)](https://github.com/adzetto/femlabpy/actions/workflows/tests.yml)
[![License](https://img.shields.io/github/license/adzetto/femlabpy)](LICENSE)

`femlabpy` is a Python finite element teaching library built from the legacy MATLAB FemLab toolbox and the Scilab FemLab wrapper used in CE 512. It keeps the original classroom workflow recognizable while exposing a clean `pip` package with tests, packaged benchmark data, and documented Python entry points.

The package currently covers:

- Linear bars, CST triangles, bilinear quads, tetrahedra, and hexahedra
- Potential / diffusion problems on T3 and Q4 meshes
- Nonlinear truss load stepping (`nlbar`)
- Plane-stress and plane-strain elastoplastic Q4 benchmarks
- Legacy-compatible wrappers such as `canti`, `flowq4`, `plastps`, and `plastpe`
- Gmsh mesh import and plotting helpers
- Packaged classroom examples under `femlabpy.examples`

## Repository

- Source: <https://github.com/adzetto/femlabpy>
- PyPI: <https://pypi.org/project/femlabpy/>
- Issues: <https://github.com/adzetto/femlabpy/issues>
- Code of Conduct: [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md)
- Contributing: [CONTRIBUTING.md](CONTRIBUTING.md)
- License: [LICENSE](LICENSE)

## Installation

```bash
pip install femlabpy
```

Development install:

```bash
python -m venv .venv
. .venv/Scripts/activate
python -m pip install -e .[dev]
pytest -q
```

Version check:

```bash
python -m femlabpy --version
python -c "import femlabpy; print(femlabpy.__version__)"
```

## Quick Start

### Packaged cantilever benchmark

```python
from femlabpy.examples import run_cantilever

result = run_cantilever(plot=False)
print(result["u"].shape)
print(result["S"].shape)
print(result["R"])
```

### Legacy-compatible workflow

```python
import femlabpy as fp

data = fp.canti()
result = fp.elastic(
    data["T"],
    data["X"],
    data["G"],
    data["C"],
    data["P"],
    dof=int(data["dof"]),
    plot=False,
)
print(result["u"][:6])
```

### Potential flow on Q4 and T3 meshes

```python
import femlabpy as fp

q4 = fp.flowq4(plot=False)
t3 = fp.flowt3(plot=False)

print(q4["u"].min(), q4["u"].max())
print(t3["u"].min(), t3["u"].max())
```

### Nonlinear truss benchmark

```python
import femlabpy as fp

case = fp.bar01()
result = fp.nlbar(
    case["T"],
    case["X"],
    case["G"],
    case["C"],
    case["P"],
    no_loadsteps=int(case["no_loadsteps"][0, 0]),
    i_max=int(case["i_max"][0, 0]),
    i_d=int(case["i_d"][0, 0]),
    plotdof=int(case["plotdof"][0, 0]),
    tol=float(case["TOL"][0, 0]),
)

print(result["U_path"].ravel())
print(result["F_path"].ravel())
```

### Elastoplastic benchmark

```python
from femlabpy.examples import run_square_plastpe

result = run_square_plastpe(plot=False)
print(result["u"].shape)
print(result["E"].shape)
```

### Gmsh import

```python
from femlabpy import load_gmsh2

mesh = load_gmsh2("src/femlabpy/data/meshes/deneme.msh")
print(mesh["positions"].shape)
print(mesh["triangles"].shape)
```

## Legacy FemLab Compatibility

`femlabpy` is intentionally organized around the original classroom naming:

- Data loaders such as `canti()`, `flow()`, `bar01()`, `square()`, and `hole()`
- Legacy solver wrappers such as `elastic()`, `flowq4()`, `flowt3()`, `nlbar()`, `plastps()`, and `plastpe()`
- Lower-level kernels such as `ket3e`, `kq4e`, `qq4e`, `qbar`, `stressvm`, and `solve_lag`

Packaged example data and reproducible drivers live in `femlabpy.examples`, while the preserved Scilab material remains under `legacy/scilab/`.

## Using `help()`

The README reference below is aligned with the actual docstrings exposed by the package. You can inspect the same information locally:

```bash
python -c "import femlabpy; help(femlabpy.ket3e)"
python -c "import femlabpy; help(femlabpy.plastpe)"
python -c "from femlabpy import examples; help(examples.run_cantilever)"
python -m pydoc femlabpy
```

## API Reference

<details>
<summary><strong>Compatibility</strong></summary>

| Function | `help()` summary |
| --- | --- |
| `canti` | Return the original `canti.m` cantilever benchmark input deck. |
| `flow` | Return the original `flow.m` potential-flow benchmark data. |
| `bar01` | Return the packaged input deck corresponding to `bar01.m`. |
| `bar02` | Return the packaged input deck corresponding to `bar02.m`. |
| `bar03` | Return the packaged input deck corresponding to `bar03.m`. |
| `square` | Return the packaged input deck corresponding to `square.m`. |
| `hole` | Return the packaged input deck corresponding to `hole.m`. |
| `elastic` | Solve a linear Q4 elasticity problem following the original `elastic.m` workflow. |
| `flowq4` | Solve a Q4 potential problem following the original `flowq4.m` driver. |
| `flowt3` | Solve a T3 potential problem following the original `flowt3.m` driver. |
| `nlbar` | Solve a nonlinear truss problem through the legacy `nlbar.m` driver logic. |
| `plastps` | Solve a plane-stress elastoplastic Q4 problem following `plastps.m`. |
| `plastpe` | Solve a plane-strain elastoplastic Q4 problem following `plastpe.m`. |
| `setpath` | Return canonical package paths for compatibility with legacy FemLab scripts. |

</details>

<details>
<summary><strong>Core And Assembly</strong></summary>

| Function | `help()` summary |
| --- | --- |
| `init` | Initialize FEM arrays for a problem with nn nodes and dof DOFs per node. |
| `rows` | Return the number of rows in an array-like object. |
| `cols` | Return the number of columns in an array-like object. |
| `assmk` | Assemble one element stiffness matrix into the global stiffness matrix. |
| `assmq` | Assemble one element force vector into the global internal-force vector. |
| `setload` | Set nodal loads from a load matrix P. |
| `addload` | Add nodal loads from a load matrix P (accumulates, doesn't replace). |
| `setbc` | Apply boundary conditions using direct elimination. |
| `solve_lag` | Solve a linear system with Dirichlet constraints via Lagrange multipliers. |
| `solve_lag_general` | Solve a linear system with general linear constraints `G u = Q`. |
| `reaction` | Extract support reactions at constrained degrees of freedom. |
| `rnorm` | Return the residual norm restricted to unconstrained degrees of freedom. |

</details>

<details>
<summary><strong>Bars And Solids</strong></summary>

| Function | `help()` summary |
| --- | --- |
| `kebar` | Compute the tangent stiffness matrix of a geometrically nonlinear bar element. |
| `kbar` | Assemble bar or truss tangent stiffness contributions into the global matrix. |
| `qebar` | Compute the internal-force response of a single geometrically nonlinear bar. |
| `qbar` | Assemble bar or truss internal forces and element output quantities. |
| `keT4e` | Compute the element stiffness matrix for a 4-node tetrahedral solid. |
| `kT4e` | Assemble T4 solid element stiffness contributions into the global matrix. |
| `qeT4e` | Compute stress and strain results for one tetrahedral solid element. |
| `qT4e` | Compute T4 solid stresses and assemble internal forces. |
| `keh8e` | Compute the element stiffness matrix for an 8-node hexahedral solid. |
| `kh8e` | Assemble H8 solid element stiffness contributions into the global matrix. |
| `qeh8e` | Compute stress and strain results for one H8 solid element. |
| `qh8e` | Compute H8 solid stresses and assemble internal forces. |

</details>

<details>
<summary><strong>Triangles</strong></summary>

| Function | `help()` summary |
| --- | --- |
| `ket3e` | Compute the element stiffness matrix for a 3-node triangular element (CST). |
| `kt3e` | Assemble T3 (CST) element stiffness matrices into global stiffness matrix. |
| `qet3e` | Compute stresses and strains for a single T3 element. |
| `qt3e` | Compute element stresses/strains for all T3 elements and assemble internal forces. |
| `ket3p` | Compute the element conductivity matrix for a 3-node potential triangle. |
| `kt3p` | Assemble T3 potential-element conductivities into the global matrix. |
| `qet3p` | Compute gradient and flux results for one 3-node potential triangle. |
| `qt3p` | Compute T3 potential-element fluxes and assemble nodal fluxes. |

</details>

<details>
<summary><strong>Quadrilaterals</strong></summary>

| Function | `help()` summary |
| --- | --- |
| `keq4e` | Compute element stiffness matrix for a 4-node quadrilateral (Q4) element. |
| `kq4e` | Assemble Q4 element stiffness matrices into global stiffness matrix. |
| `qeq4e` | Compute stresses and strains at Gauss points for a single Q4 element. |
| `qq4e` | Compute stresses/strains for all Q4 elements and assemble internal forces. |
| `keq4p` | Compute the element conductivity matrix for a 4-node potential quadrilateral. |
| `kq4p` | Assemble Q4 potential-element conductivities into the global matrix. |
| `qeq4p` | Compute gradient and flux results for one Q4 potential element. |
| `qq4p` | Compute Q4 potential-element fluxes and assemble nodal fluxes. |
| `keq4eps` | Compute the consistent tangent stiffness of a plane-stress plastic Q4 element. |
| `kq4eps` | Assemble plane-stress plastic Q4 tangent matrices into the global matrix. |
| `qeq4eps` | Update plane-stress plastic Q4 response at Gauss points. |
| `qq4eps` | Compute plane-stress plastic Q4 internal forces and updated history fields. |
| `keq4epe` | Compute the consistent tangent stiffness of a plane-strain plastic Q4 element. |
| `kq4epe` | Assemble plane-strain plastic Q4 tangent matrices into the global matrix. |
| `qeq4epe` | Update plane-strain plastic Q4 response at Gauss points. |
| `qq4epe` | Compute plane-strain plastic Q4 internal forces and updated history fields. |

</details>

<details>
<summary><strong>Materials, I/O, And Plotting</strong></summary>

| Function | `help()` summary |
| --- | --- |
| `devstress` | Return the deviatoric stress vector together with the mean stress. |
| `devstres` | Return the deviatoric stress vector together with the mean stress. |
| `eqstress` | Return the von Mises equivalent stress for 2D or 3D stress input. |
| `yieldvm` | Evaluate the legacy von Mises yield function with isotropic hardening. |
| `dyieldvm` | Differentiate the legacy von Mises yield function with respect to dL. |
| `stressvm` | Perform a legacy von Mises return-mapping update. |
| `stressdp` | Perform a Drucker-Prager stress update with Newton iterations. |
| `load_gmsh` | Read a Gmsh mesh using the legacy `load_gmsh.m` semantics. |
| `load_gmsh2` | Read a Gmsh mesh using the more flexible `load_gmsh2.m` semantics. |
| `plotelem` | Plot the undeformed mesh and optionally annotate node or element numbers. |
| `plotforces` | Plot nodal loads as arrows on a 2D mesh view. |
| `plotbc` | Plot prescribed boundary conditions on a 2D mesh view. |
| `plotq4` | Plot a contour field reconstructed from Q4 Gauss-point results. |
| `plott3` | Plot a contour field from T3 element results. |
| `plotu` | Plot a scalar nodal field over a 2D or 3D mesh. |
| `solve_nlbar` | Solve the legacy nonlinear bar examples with the orthogonal residual method. |
| `solve_plastic` | Solve the legacy Q4 elastoplastic examples with orthogonal residual iterations. |

</details>

<details>
<summary><strong>Examples</strong></summary>

| Function | `help()` summary |
| --- | --- |
| `femlabpy.examples.bar01_data` | Return the original `bar01.m` nonlinear truss benchmark data. |
| `femlabpy.examples.bar02_data` | Return the original `bar02.m` nonlinear truss benchmark data. |
| `femlabpy.examples.bar03_data` | Return the original `bar03.m` 12-bar truss benchmark data. |
| `femlabpy.examples.cantilever_data` | Return the packaged classroom cantilever benchmark input deck. |
| `femlabpy.examples.ex_lag_mult_data` | Return the packaged three-bar Lagrange-multiplier benchmark data. |
| `femlabpy.examples.flow_data` | Return the packaged potential-flow benchmark for both Q4 and T3 meshes. |
| `femlabpy.examples.gmsh_triangle_data` | Return the packaged Gmsh triangle benchmark in FemLab array form. |
| `femlabpy.examples.hole_data` | Return the packaged `hole.m` data for plane stress or plane strain. |
| `femlabpy.examples.run_bar01_nlbar` | Solve the original `bar01.m` example through the legacy `nlbar` driver. |
| `femlabpy.examples.run_bar02_nlbar` | Solve the original `bar02.m` example through the legacy `nlbar` driver. |
| `femlabpy.examples.run_bar03_nlbar` | Solve the original `bar03.m` example through the legacy `nlbar` driver. |
| `femlabpy.examples.run_cantilever` | Solve the packaged cantilever benchmark and optionally produce figures. |
| `femlabpy.examples.run_ex_lag_mult` | Solve the packaged three-bar truss with linear displacement constraints. |
| `femlabpy.examples.run_flow_q4` | Solve the packaged Q4 potential-flow benchmark and optionally plot it. |
| `femlabpy.examples.run_flow_t3` | Solve the packaged T3 potential-flow benchmark and optionally plot it. |
| `femlabpy.examples.run_gmsh_triangle` | Solve the bundled triangular Gmsh example with CST elements. |
| `femlabpy.examples.run_hole_plastpe` | Solve the plane-strain `hole.m` elastoplastic benchmark. |
| `femlabpy.examples.run_hole_plastps` | Solve the plane-stress `hole.m` elastoplastic benchmark. |
| `femlabpy.examples.run_square_plastpe` | Solve the plane-strain `square.m` elastoplastic benchmark. |
| `femlabpy.examples.run_square_plastps` | Solve the plane-stress `square.m` elastoplastic benchmark. |
| `femlabpy.examples.square_data` | Return the packaged `square.m` data for plane stress or plane strain. |

</details>

## Project Layout

```text
src/femlabpy/           Python package
tests/                  Regression and parity tests
legacy/scilab/          Preserved Scilab macros and examples
benchmarks/             Stored reference outputs and comparison tables
scripts/                Comparison, validation, and helper runners
docs/                   Notes, figures, and mapping documents
```

## Development And Validation

Run the local test suite:

```bash
pytest -q
```

Build the package:

```bash
python -m build
python -m twine check dist/*
```

Selected maintenance scripts:

- `python scripts/generate_solver_comparison.py --solver python --solver scilab`
- `python scripts/compare_ex_lag_mult.py`
- `python scripts/generate_parity_artifacts.py`

## Community Standards

This repository is meant to stay usable as both a teaching codebase and a publishable Python package.

- Contributor expectations: [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md)
- Change process and local setup: [CONTRIBUTING.md](CONTRIBUTING.md)
- Packaging and redistribution terms: [LICENSE](LICENSE)

Legacy MATLAB / Scilab source material is preserved for historical and compatibility purposes inside `legacy/`.
