# femlabpy

[![PyPI version](https://badge.fury.io/py/femlabpy.svg)](https://pypi.org/project/femlabpy/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/github/actions/workflow/status/adzetto/femlabpy/tests.yml?branch=main&label=tests)](https://github.com/adzetto/femlabpy/actions/workflows/tests.yml)
[![License](https://img.shields.io/github/license/adzetto/femlabpy)](LICENSE)

Python FEM library for teaching. Based on MATLAB/Scilab FemLab.

## Install

```bash
pip install femlabpy
```

Optional:
```bash
pip install "femlabpy[mesh]"  # Gmsh 4.x support
pip install "femlabpy[gui]"   # GUI tools
```

## Usage

**1. Import library.**
```python
import femlabpy as fp
```

**2. Load problem data.**
```python
data = fp.canti()
```

**3. Run solver.**
```python
result = fp.elastic(data["T"], data["X"], data["G"], data["C"], data["P"], dof=2)
```

**4. Get results.**
```python
u = result["u"]  # displacements
S = result["S"]  # stresses
```

## Problem Types

**Linear elastic analysis:**
```python
result = fp.elastic(T, X, G, C, P, dof=2)
```

**Potential flow (Q4):**
```python
result = fp.flowq4()
```

**Potential flow (T3):**
```python
result = fp.flowt3()
```

**Nonlinear truss:**
```python
result = fp.nlbar(T, X, G, C, P, no_loadsteps=20, i_max=50)
```

**Plane stress plasticity:**
```python
result = fp.plastps(...)
```

**Plane strain plasticity:**
```python
result = fp.plastpe(...)
```

**Load Gmsh mesh:**
```python
mesh = fp.load_gmsh2("mesh.msh")
```

## Examples

**Cantilever beam:**
```python
from femlabpy.examples import run_cantilever
result = run_cantilever(plot=False)
```

**Potential flow:**
```python
q4 = fp.flowq4(plot=False)
t3 = fp.flowt3(plot=False)
```

**Nonlinear truss:**
```python
case = fp.bar01()
result = fp.nlbar(case["T"], case["X"], case["G"], case["C"], case["P"],
                  no_loadsteps=20, i_max=50)
```

**Plasticity:**
```python
from femlabpy.examples import run_square_plastpe
result = run_square_plastpe(plot=False)
```

**Gmsh mesh:**
```python
mesh = fp.load_gmsh2("mesh.msh")
print(mesh.positions.shape)
print(mesh.triangles.shape)
```

## API Reference

<details>
<summary><strong>Data Loaders</strong></summary>

| Function | Description |
| --- | --- |
| `canti` | Cantilever benchmark data |
| `flow` | Potential flow data |
| `bar01`, `bar02`, `bar03` | Nonlinear truss data |
| `square`, `hole` | Plasticity data |

</details>

<details>
<summary><strong>Solvers</strong></summary>

| Function | Description |
| --- | --- |
| `elastic` | Linear Q4 elasticity |
| `flowq4`, `flowt3` | Potential flow |
| `nlbar` | Nonlinear truss |
| `plastps`, `plastpe` | Plane stress/strain plasticity |

</details>

<details>
<summary><strong>Elements</strong></summary>

| Function | Description |
| --- | --- |
| `ket3e`, `kt3e` | T3 triangle stiffness |
| `keq4e`, `kq4e` | Q4 quad stiffness |
| `kebar`, `kbar` | Bar stiffness |
| `keT4e`, `kT4e` | T4 tetrahedron stiffness |
| `keh8e`, `kh8e` | H8 hexahedron stiffness |

</details>

<details>
<summary><strong>Assembly</strong></summary>

| Function | Description |
| --- | --- |
| `init` | Initialize FEM arrays |
| `assmk` | Assemble stiffness |
| `assmq` | Assemble forces |
| `setbc` | Apply boundary conditions |
| `setload` | Set nodal loads |

</details>

<details>
<summary><strong>Materials</strong></summary>

| Function | Description |
| --- | --- |
| `stressvm` | Von Mises return mapping |
| `stressdp` | Drucker-Prager update |
| `eqstress` | Equivalent stress |
| `devstress` | Deviatoric stress |

</details>

<details>
<summary><strong>I/O</strong></summary>

| Function | Description |
| --- | --- |
| `load_gmsh` | Load Gmsh mesh |
| `load_gmsh2` | Load Gmsh mesh (flexible) |

</details>

## Development

```bash
pytest -q                    # run tests
python -m build              # build package
```

## Links

- Source: <https://github.com/adzetto/femlabpy>
- PyPI: <https://pypi.org/project/femlabpy/>
- Issues: <https://github.com/adzetto/femlabpy/issues>

## References

1. Bathe, K.J. (2014). *Finite Element Procedures*. Prentice Hall.
2. Zienkiewicz, Taylor, Zhu (2013). *The Finite Element Method*. Elsevier.
3. Hughes, T.J.R. (2000). *The Finite Element Method*. Dover.
4. Cook et al. (2002). *Concepts and Applications of FEA*. Wiley.
5. de Souza Neto et al. (2008). *Computational Methods for Plasticity*. Wiley.
6. Gmsh: <https://gmsh.info/>
