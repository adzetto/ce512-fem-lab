# MATLAB to Python Mapping

This note maps the original MATLAB FemLab toolbox under `FemLab_matlab/M_Files/` to the current `femlabpy` package layout.

## Core Utilities

| MATLAB | Python |
| --- | --- |
| `init.m` | `femlabpy.core.init` |
| `rows.m`, `cols.m` | `femlabpy.core.rows`, `femlabpy.core.cols` |
| `assmk.m`, `assmq.m` | `femlabpy.assembly.assmk`, `femlabpy.assembly.assmq` |
| `setload.m` | `femlabpy.loads.setload` |
| `setbc.m` | `femlabpy.boundary.setbc` |
| `reaction.m` | `femlabpy.postprocess.reaction` |
| `rnorm.m` | `femlabpy.boundary.rnorm` |
| `setpath.m` | `femlabpy.compat.setpath` |

## Material Helpers

| MATLAB | Python |
| --- | --- |
| `devstres.m` / `devstress` | `femlabpy.materials.devstress` and alias `femlabpy.devstres` |
| `eqstress.m` | `femlabpy.materials.eqstress` |
| `yieldvm.m` | `femlabpy.materials.yieldvm` |
| `dyieldvm.m` | `femlabpy.materials.dyieldvm` |
| `stressvm.m` | `femlabpy.materials.stressvm` |
| `stressdp.m` | `femlabpy.materials.stressdp` |

## Element Kernels

| MATLAB | Python |
| --- | --- |
| `kebar.m`, `qebar.m`, `kbar.m`, `qbar.m` | `femlabpy.elements.bars` |
| `ket3e.m`, `qet3e.m`, `kt3e.m`, `qt3e.m` | `femlabpy.elements.triangles` |
| `ket3p.m`, `qet3p.m`, `kt3p.m`, `qt3p.m` | `femlabpy.elements.triangles` |
| `keq4e.m`, `qeq4e.m`, `kq4e.m`, `qq4e.m` | `femlabpy.elements.quads` |
| `keq4p.m`, `qeq4p.m`, `kq4p.m`, `qq4p.m` | `femlabpy.elements.quads` |
| `keq4eps.m`, `qeq4eps.m`, `kq4eps.m`, `qq4eps.m` | `femlabpy.elements.quads` |
| `keq4epe.m`, `qeq4epe.m`, `kq4epe.m`, `qq4epe.m` | `femlabpy.elements.quads` |

## Plotting and Mesh I/O

| MATLAB | Python |
| --- | --- |
| `plotelem.m` | `femlabpy.plotting.plotelem` |
| `plotq4.m` | `femlabpy.plotting.plotq4` |
| `plott3.m` | `femlabpy.plotting.plott3` |
| `plotu.m` | `femlabpy.plotting.plotu` |
| `load_gmsh.m`, `load_gmsh2.m` | `femlabpy.io.gmsh.load_gmsh`, `femlabpy.io.gmsh.load_gmsh2` |

The Gmsh loaders now expose both the normalized Python mesh fields (`positions`,
`triangles`, `element_infos`, ...) and the legacy MATLAB aliases (`POS`,
`TRIANGLES`, `ELE_INFOS`, `nbTriangles`, `MIN`, `MAX`, ...).

## Original Example Coverage

| MATLAB example | Python entry point |
| --- | --- |
| `canti.m` | `femlabpy.examples.run_cantilever` |
| `flow.m` + `flowq4.m` | `femlabpy.examples.run_flow_q4` |
| `flow.m` + `flowt3.m` | `femlabpy.examples.run_flow_t3` |
| `bar01.m` + `nlbar.m` | `femlabpy.examples.run_bar01_nlbar` |
| `bar02.m` + `nlbar.m` | `femlabpy.examples.run_bar02_nlbar` |
| `bar03.m` + `nlbar.m` | `femlabpy.examples.run_bar03_nlbar` |
| `square.m` + `plastps.m` | `femlabpy.examples.run_square_plastps` |
| `square.m` + `plastpe.m` | `femlabpy.examples.run_square_plastpe` |
| `hole.m` + `plastps.m` | `femlabpy.examples.run_hole_plastps` |
| `hole.m` + `plastpe.m` | `femlabpy.examples.run_hole_plastpe` |

The nonlinear and plastic drivers now live in `femlabpy.solvers`, and the example inputs needed for installed-package parity are packaged under `femlabpy.data.cases`.

## MATLAB Compatibility Aliases

The package also exposes the original classroom script names as documented Python wrappers with explicit arguments:

| MATLAB name | Python compatibility wrapper |
| --- | --- |
| `canti` | `femlabpy.canti()` |
| `flow` | `femlabpy.flow()` |
| `flowq4` | `femlabpy.flowq4()` or `femlabpy.flowq4(T1, X, G, C, P=None, dof=1)` |
| `flowt3` | `femlabpy.flowt3()` or `femlabpy.flowt3(T2, X, G, C, P=None, dof=1)` |
| `bar01`, `bar02`, `bar03` | `femlabpy.bar01()`, `femlabpy.bar02()`, `femlabpy.bar03()` |
| `square`, `hole` | `femlabpy.square()`, `femlabpy.hole()` |
| `elastic` | `femlabpy.elastic(...)` |
| `nlbar` | `femlabpy.nlbar(...)` |
| `plastps`, `plastpe` | `femlabpy.plastps(...)`, `femlabpy.plastpe(...)` |

These wrappers are designed for `help()` / docstring discovery and for users coming directly from the original MATLAB notes. Internally they call the vectorized `femlabpy` kernels and solvers.
