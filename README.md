# FemLab Python

Python port of the legacy Scilab FemLab wrapper prepared by G. Turan at IYTE, itself derived from the original MATLAB FemLab teaching toolbox by O. Hededal and S. Krenk at Aalborg University.

## What This Repository Contains

This repository currently preserves three layers of the project:

| Layer | Status | Notes |
| --- | --- | --- |
| Original Scilab wrapper | Preserved | Historical source tree kept in place for reference and parity checks |
| Original MATLAB toolbox reference | External reference | Compared during porting from `C:\Users\lenovo\Downloads\FemLab_matlab\FemLab_matlab` |
| New Python implementation | In progress | Modern, testable package replacing the fragile Scilab execution flow |

The legacy Scilab entry flow described in [README_first.txt](README_first.txt) is:

```text
exec loader.sce;
exec examples/canti.sce;
exec examples/elastic.sce;
```

That path works as a historical reference, but it has known issues:

- `examples/elastic.sce` uses `inv(K)*p` instead of a direct solve.
- `examples/canti_gmsh.sce` hard-codes an obsolete Linux path.
- `examples/a1.sce` expects quadrilateral connectivity from a triangular mesh.
- The Scilab elastoplastic path contains a likely naming regression around `devstres` vs `devstress`.

## Why The Python Port Exists

The Scilab wrapper is useful as a teaching artifact, but it is hard to validate, hard to package, and partially diverges from the MATLAB original. The Python port aims to:

- preserve the original FEM examples and formulas,
- recover behavior from the MATLAB toolbox where the Scilab wrapper regressed,
- provide a clean package structure, tests, and reproducible runs,
- keep the legacy Scilab files available for comparison.

## Legacy Structure

| Path | Role |
| --- | --- |
| `macros/` | Core FEM routines: assembly, element matrices, loads, constraints, stress recovery, plotting helpers |
| `examples/` | Hand-authored teaching examples |
| `mesh/` | Gmsh geometry, mesh assets, and a triangle demo |
| `help/` | Exported HTML reference pages for the legacy API |
| `doc/` | Inherited manual bundle from the MATLAB toolbox |

## Editable Targets

The high-value editable targets are documented in [docs/EDITABLE_TARGETS.md](docs/EDITABLE_TARGETS.md). That file separates:

- files worth editing directly,
- generated or inherited artifacts that should usually not be edited,
- the planned Python destination for each important legacy area.

## Porting Strategy

1. Keep the Scilab tree intact as a reference baseline.
2. Port the solver and element routines into a typed Python package.
3. Port the mesh import and example drivers.
4. Add tests against legacy example outputs and matrix shapes.
5. Replace the legacy first-run path with Python-first usage.

## Status

- Legacy Scilab source imported into Git history.
- GitHub repository created with `gh`.
- Python package now includes core assembly, Gmsh import, 1D/2D/3D element kernels, plotting helpers, and example drivers.

## Quick Start

```powershell
py -3 -m venv .venv
.\.venv\Scripts\python -m pip install -e .[dev]
.\.venv\Scripts\python -m pytest -q
```

Example usage:

```python
from femlab.examples import run_cantilever, run_flow_q4, run_gmsh_triangle

cantilever = run_cantilever(plot=False)
flow = run_flow_q4(plot=False)
triangle = run_gmsh_triangle(plot=False)
```

## Current Python Coverage

- Line/bar elements: `kbar`, `qbar`, `kebar`, `qebar`
- 2D elastic and potential elements: `Q4E`, `Q4P`, `T3E`, `T3P`
- 2D elastoplastic quadrilateral elements: `Q4EPS`, `Q4EPE`
- 3D elastic solid elements from the Scilab tree: `T4E`, `H8E`
- Mesh readers: `load_gmsh`, `load_gmsh2`
- Plot helpers: `plotelem`, `plotu`, `plotq4`, `plott3`, `plotbc`, `plotforces`
- Example drivers: cantilever beam, Gmsh triangle mesh, Q4/T3 potential flow

## License Note

The legacy source files contain copyright notices but no obvious standalone license file in the imported material. Until upstream licensing is clarified, treat the legacy MATLAB/Scilab content as historical reference material.
