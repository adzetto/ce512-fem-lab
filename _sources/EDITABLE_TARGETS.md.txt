# Editable Targets

This table is the working inventory for the migration. "Python target" refers to the destination module or area in the new package.

| Legacy path | Purpose | Edit now | Python target | Notes |
| --- | --- | --- | --- | --- |
| `README_first.txt` | Original Scilab usage note | Yes | `README.md` | Historical instructions only |
| `builder.sce` | Broken Scilab build script | Rarely | none | Keep for provenance |
| `loader.sce` | Legacy macro loader | Rarely | package init / CLI | Reference only |
| `examples/canti.sce` | Main cantilever setup | Yes | `src/femlab/examples/cantilever.py` | Primary parity example |
| `examples/elastic.sce` | Linear elastic driver | Yes | `src/femlab/examples/linear_elastic.py` | Needs direct solve, not explicit inverse |
| `examples/canti_gmsh.sce` | Gmsh cantilever variant | Yes | `src/femlab/examples/gmsh_cantilever.py` | Contains stale path in legacy tree |
| `examples/a1.sce` | Legacy mesh experiment | No | none until fixed | Currently inconsistent with shipped mesh |
| `mesh/deneme.sce` | Triangle-element demo | Yes | `src/femlab/examples/triangle_gmsh.py` | Good second parity case |
| `mesh/deneme.geo` | Gmsh geometry source | Yes | `examples/assets/deneme.geo` | Editable source of mesh geometry |
| `mesh/deneme.msh` | Generated mesh file | No | `examples/assets/deneme.msh` | Data artifact, not first edit point |
| `mesh/deneme_inclass.geo` | Simpler teaching geometry | Yes | `examples/assets/deneme_inclass.geo` | Useful scratch input |
| `mesh/intro_gmsh.odt` | Editable tutorial source | Maybe | `docs/` | Prefer editing ODT, not PDF |
| `mesh/intro_gmsh.pdf` | Exported tutorial PDF | No | none | Generated document |
| `doc/Manual.pdf` | Inherited reference manual | No | `docs/legacy/` reference | Byte-identical to MATLAB bundle |
| `doc/manual.ps` | Inherited PostScript manual | No | `docs/legacy/` reference | Better text extraction than PDF |
| `help/*.htm` | Exported function reference | Rarely | API docs later | Secondary to source code |
| `macros/init.sci` | Global matrix/vector initialization | Yes | `src/femlab/core.py` | Straightforward port |
| `macros/assmk.sci` | Global matrix assembly | Yes | `src/femlab/assembly.py` | Fundamental utility |
| `macros/assmq.sci` | Global vector assembly | Yes | `src/femlab/assembly.py` | Fundamental utility |
| `macros/setload.sci` | Nodal load application | Yes | `src/femlab/loads.py` | Small but central |
| `macros/addload.sci` | Load accumulation helper | Yes | `src/femlab/loads.py` | Small helper |
| `macros/setbc.sci` | Constraint handling | Yes | `src/femlab/boundary.py` | Scilab behavior diverges from MATLAB |
| `macros/solve_lag.sci` | Lagrange-multiplier solver | Yes | `src/femlab/solvers.py` | Required for robust constrained solves |
| `macros/reaction.sci` | Reaction recovery | Yes | `src/femlab/postprocess.py` | Keep output compatible |
| `macros/load_gmsh.sci` | Gmsh reader | Yes | `src/femlab/io/gmsh.py` | Use MATLAB file as source of truth where needed |
| `macros/k*.sci` | Element stiffness routines | Yes | `src/femlab/elements/` | Group by physics and topology |
| `macros/q*.sci` | Internal force, stress, strain recovery | Yes | `src/femlab/elements/` | Pair with matching `k*` routines |
| `macros/stressdp.sci` | Drucker-Prager update | Yes | `src/femlab/materials/plasticity.py` | Needs naming fix during port |
| `macros/stressvm.sci` | Von Mises update | Yes | `src/femlab/materials/plasticity.py` | Pair with `yieldvm` |
| `macros/dyieldvm.sci` | Yield derivative | Yes | `src/femlab/materials/plasticity.py` | Small helper |
| `macros/yieldvm.sci` | Yield function | Yes | `src/femlab/materials/plasticity.py` | Small helper |
| `macros/devstres.sci` | Deviatoric stress helper | Yes | `src/femlab/materials/invariants.py` | Scilab naming mismatch matters |
| `macros/eqstress.sci` | Equivalent stress helper | Yes | `src/femlab/materials/invariants.py` | Small helper |
| `macros/plot*.sci` | Legacy plotting helpers | Yes | `src/femlab/plotting.py` | Python port should use Matplotlib |
| `macros/lib` | Generated Scilab library metadata | No | none | Generated artifact |
| `macros/names` | Generated Scilab names list | No | none | Generated artifact |

## MATLAB Reference Notes

The MATLAB directory at `C:\Users\lenovo\Downloads\FemLab_matlab\FemLab_matlab` contains a broader original toolbox than the Scilab wrapper. It is especially useful for:

- the original `setbc.m` spring-based boundary condition logic,
- `load_gmsh.m` and `load_gmsh2.m` behavior,
- additional examples such as `flow.m`, `hole.m`, `nlbar.m`, `plastpe.m`, and `plastps.m`,
- missing utilities like `plotu.m`.

The Python port uses the Scilab tree as the preserved baseline and the MATLAB tree as the authoritative fallback when the Scilab wrapper is stale or inconsistent.
