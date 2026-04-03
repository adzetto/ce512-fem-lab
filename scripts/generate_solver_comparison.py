from __future__ import annotations

import csv
import json
import math
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
OUTDIR = REPO / "benchmarks" / "matlab_regression"
INPUTS = OUTDIR / "inputs"
RAW = OUTDIR / "raw"
PLOTS = OUTDIR / "plots"
LOGS = OUTDIR / "logs"

SCILAB_CLI = Path(r"C:\Program Files\scilab-2025.0.0\bin\WScilex-cli.exe")
MATLAB_CLI = Path(r"C:\Program Files\MATLAB\R2025b\bin\matlab.exe")

sys.path.insert(0, str(REPO / "src"))

from femlabpy._helpers import solve_legacy_symmetric_system, solve_linear_system  # noqa: E402
from femlabpy import (  # noqa: E402
    init,
    kbar,
    kq4e,
    kq4epe,
    kq4eps,
    kq4p,
    kt3e,
    kt3p,
    qbar,
    qq4e,
    qq4epe,
    qq4eps,
    qq4p,
    qt3e,
    qt3p,
    reaction,
    rnorm,
    setbc,
    setload,
    solve_lag,
)

try:
    import scipy.sparse as sp
except ImportError:  # pragma: no cover
    sp = None


def solve_plastic_system(matrix, rhs, *, plane_strain: bool) -> np.ndarray:
    if plane_strain:
        return solve_legacy_symmetric_system(matrix, rhs)
    return solve_linear_system(matrix, rhs)


@dataclass(frozen=True)
class CaseSpec:
    name: str
    case_type: str
    description: str
    expected_statuses: dict[str, tuple[str, ...]] = field(default_factory=dict)


CASES: tuple[CaseSpec, ...] = (
    CaseSpec("cantilever_q4", "elastic_q4", "Q4 cantilever beam"),
    CaseSpec("gmsh_triangle_t3", "elastic_t3_lag", "Triangle mesh cantilever loaded from deneme.msh"),
    CaseSpec("flow_q4", "flow_q4", "Q4 potential flow"),
    CaseSpec("flow_t3", "flow_t3", "T3 potential flow"),
    CaseSpec("bar01_nlbar", "nlbar", "2D nonlinear bar truss"),
    CaseSpec("bar02_nlbar", "nlbar", "3D nonlinear bar truss"),
    CaseSpec(
        "bar03_nlbar",
        "nlbar",
        "12-bar space truss (expected unstable legacy benchmark)",
        expected_statuses={"matlab": ("failed",), "python": ("failed",), "scilab": ("failed", "timeout")},
    ),
    CaseSpec("square_plastps", "plastps", "Plane-stress square von Mises plasticity"),
    CaseSpec("square_plastpe", "plastpe", "Plane-strain square von Mises plasticity"),
    CaseSpec("hole_plastps", "plastps", "Plane-stress plate with hole"),
    CaseSpec("hole_plastpe", "plastpe", "Plane-strain plate with hole"),
)

SOLVERS = ("matlab", "python", "scilab")
PAIRWISE = (("python", "matlab"), ("scilab", "matlab"), ("python", "scilab"))


def _case_data_from_python(spec: CaseSpec) -> dict[str, Any] | None:
    """Return input data dict for a case using Python-side definitions."""
    if spec.case_type == "elastic_q4":
        from femlabpy.examples.cantilever import cantilever_data
        return cantilever_data()
    if spec.case_type == "elastic_t3_lag":
        from femlabpy.examples.gmsh_triangle import gmsh_triangle_data
        data = gmsh_triangle_data()
        data.pop("mesh", None)
        return data
    if spec.case_type in {"flow_q4", "flow_t3"}:
        from femlabpy.examples.flow import flow_data
        data = flow_data()
        key = "T1" if spec.case_type == "flow_q4" else "T2"
        C = data["C"]
        # Expand 2-column C to 3-column [node, dof_component, value] for Scilab compat
        if C.shape[1] == 2:
            C = np.column_stack([C[:, 0], np.ones(C.shape[0]), C[:, 1]])
        return {"X": data["X"], "T": data[key], "G": data["G"], "C": C, "dof": data["dof"]}
    return None  # nonlinear/plastic cases rely on MATLAB export


def prepare_inputs() -> None:
    """Generate input TSVs for all cases from Python-side definitions.

    For cases where Python-side data is not available (nlbar, plasticity),
    falls back to existing MATLAB-exported inputs.
    """
    for spec in CASES:
        base = case_input_dir(spec.name)
        base.mkdir(parents=True, exist_ok=True)
        data = _case_data_from_python(spec)
        if data is None:
            continue  # keep existing MATLAB-exported inputs
        for name, value in data.items():
            if isinstance(value, (int, float)):
                write_tsv(base / f"{name}.tsv", np.array([[value]]))
            elif isinstance(value, np.ndarray):
                # Preserve 2D shape: don't let write_tsv collapse row vectors
                arr = value if value.ndim >= 2 else value.reshape(-1, 1)
                write_tsv(base / f"{name}.tsv", arr)
            else:
                write_tsv(base / f"{name}.tsv", np.array([[value]]))


def ensure_dirs() -> None:
    for path in (OUTDIR, INPUTS, RAW, PLOTS, LOGS):
        path.mkdir(parents=True, exist_ok=True)
    for solver in SOLVERS:
        (RAW / solver).mkdir(parents=True, exist_ok=True)


def write_tsv(path: Path, array: Any) -> None:
    arr = np.asarray(array, dtype=float)
    path.parent.mkdir(parents=True, exist_ok=True)
    if arr.size == 0:
        path.write_text("", encoding="utf-8")
        return
    if arr.ndim == 0:
        arr = arr.reshape(1, 1)
    elif arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    # Use %.16g for Scilab compatibility (avoids scientific notation for integers)
    np.savetxt(path, arr, delimiter="\t", fmt="%.16g")


def read_tsv(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return np.zeros((0, 0), dtype=float)
    data = np.loadtxt(path, delimiter="\t", ndmin=2)
    arr = np.asarray(data, dtype=float)
    if arr.ndim == 0:
        return arr.reshape(1, 1)
    return arr


def scalar_input(inputs: dict[str, np.ndarray], name: str, default: float | None = None) -> float:
    value = inputs.get(name)
    if value is None:
        if default is None:
            raise KeyError(name)
        return default
    arr = np.asarray(value, dtype=float).reshape(-1)
    if arr.size == 0:
        if default is None:
            raise KeyError(name)
        return default
    return float(arr[0])


def column_vector(values: Any) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    return arr.reshape(-1, 1)


def case_input_dir(case_name: str) -> Path:
    return INPUTS / case_name


def case_raw_dir(solver: str, case_name: str) -> Path:
    return RAW / solver / case_name


def load_case_inputs(case_name: str) -> dict[str, np.ndarray]:
    base = case_input_dir(case_name)
    inputs: dict[str, np.ndarray] = {}
    for path in base.glob("*.tsv"):
        data = read_tsv(path)
        if data is not None:
            inputs[path.stem] = data
    return inputs


def export_case_outputs(base: Path, outputs: dict[str, Any]) -> None:
    for name, value in outputs.items():
        if value is None:
            continue
        if isinstance(value, str):
            (base / f"{name}.txt").write_text(value, encoding="utf-8")
            continue
        if name == "status":
            continue
        write_tsv(base / f"{name}.tsv", value)


def available_outputs(base: Path) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for name in ("u", "q", "S", "E", "R", "f", "U_path", "F_path", "U", "F", "path_u", "path_f"):
        data = read_tsv(base / f"{name}.tsv")
        if data is not None:
            out[name] = data
    if "U_path" not in out and "path_u" in out:
        out["U_path"] = out["path_u"]
    if "U_path" not in out and "U" in out:
        out["U_path"] = out["U"]
    if "F_path" not in out and "path_f" in out:
        out["F_path"] = out["path_f"]
    if "F_path" not in out and "F" in out:
        out["F_path"] = out["F"]
    return out


def constraint_reactions(q: np.ndarray, C: np.ndarray, dof: int) -> np.ndarray:
    constraints = np.asarray(C, dtype=float)
    if constraints.size == 0:
        return np.zeros((0, 3), dtype=float)
    if dof == 1 and constraints.shape[1] == 2:
        nodes = constraints[:, 0]
        comps = np.ones_like(nodes)
        values = constraints[:, 1]
        constraints = np.column_stack([nodes, comps, values])
    forces = reaction(column_vector(q), constraints, dof)
    if forces.shape[1] == 2:
        return np.column_stack([forces[:, 0], np.ones(forces.shape[0]), forces[:, 1]])
    return forces


def solve_elastic_q4(inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    X = inputs["X"]
    T = inputs["T"].astype(int)
    G = inputs["G"]
    C = inputs["C"]
    P = inputs.get("P", np.zeros((0, 3), dtype=float))
    dof = int(scalar_input(inputs, "dof"))
    K, p, q = init(X.shape[0], dof, use_sparse=False)
    K = kq4e(K, T, X, G)
    p = setload(p, P)
    K, p, _ = setbc(K, p, C, dof)
    u = np.linalg.solve(K, p)
    q, S, E = qq4e(q, T, X, G, u)
    R = constraint_reactions(q, C, dof)
    return {"u": u, "q": q, "S": S, "E": E, "R": R}


def solve_elastic_t3_lag(inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    X = inputs["X"]
    T = inputs["T"].astype(int)
    G = inputs["G"]
    C = inputs["C"]
    P = inputs["P"]
    dof = int(scalar_input(inputs, "dof"))
    K, p, q = init(X.shape[0], dof, use_sparse=False)
    K = kt3e(K, T, X, G)
    p = setload(p, P)
    u = solve_lag(K, p, C, dof)
    q, S, E = qt3e(q, T, X, G, u)
    R = constraint_reactions(q, C, dof)
    return {"u": u, "q": q, "S": S, "E": E, "R": R}


def solve_flow(inputs: dict[str, np.ndarray], *, triangular: bool) -> dict[str, np.ndarray]:
    X = inputs["X"]
    T = inputs["T"].astype(int)
    G = inputs["G"]
    C = inputs["C"]
    K, p, q = init(X.shape[0], 1, use_sparse=False)
    if triangular:
        K = kt3p(K, T, X, G)
        q, S, E = qt3p(q, T, X, G, np.zeros((X.shape[0], 1)))
    else:
        K = kq4p(K, T, X, G)
        q, S, E = qq4p(q, T, X, G, np.zeros((X.shape[0], 1)))
    K, p, _ = setbc(K, p, C, 1)
    u = np.linalg.solve(K, p)
    q = np.zeros_like(u)
    if triangular:
        q, S, E = qt3p(q, T, X, G, u)
    else:
        q, S, E = qq4p(q, T, X, G, u)
    R = constraint_reactions(q, C, 1)
    return {"u": u, "q": q, "S": S, "E": E, "R": R}


def solve_nlbar(inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    X = inputs["X"]
    T = inputs["T"].astype(int)
    G = inputs["G"]
    C = inputs["C"]
    P = inputs["P"]
    no_loadsteps = int(scalar_input(inputs, "no_loadsteps"))
    i_max = int(scalar_input(inputs, "i_max"))
    i_d = int(scalar_input(inputs, "i_d"))
    tol = scalar_input(inputs, "TOL")
    dof = X.shape[1]
    ndof = X.shape[0] * dof

    u = np.zeros((ndof, 1), dtype=float)
    du = np.zeros((ndof, 1), dtype=float)
    f = np.zeros((ndof, 1), dtype=float)
    df = setload(np.zeros((ndof, 1), dtype=float), P)
    U_path = [0.0]
    F_path = [0.0]
    plotdof = int(scalar_input(inputs, "plotdof")) - 1

    n = 1
    i = i_d
    restarts = 0
    max_restarts = no_loadsteps * i_max
    while n <= no_loadsteps:
        if restarts > max_restarts:
            raise RuntimeError("Nonlinear bar solver exceeded restart guard.")
        if i < i_max:
            K = np.zeros((ndof, ndof), dtype=float)
            K = kbar(K, T, X, G, u)
            Kt, df, _ = setbc(K.copy(), df, C, dof)
            du0 = np.linalg.solve(Kt, df)
            if not np.all(np.isfinite(du0)):
                raise RuntimeError("NaN/Inf in nlbar solver.")
            if float((du.T @ du0).item()) < 0.0:
                df = -df
                du0 = -du0
            if n == 1:
                l0 = float(np.linalg.norm(du0))
                l = l0
                l_max = 2.0 * l0
            else:
                l = float(np.linalg.norm(du))
                l0 = float(np.linalg.norm(du0))
        if i_d <= i < i_max:
            du = min(l / l0, l_max / l0) * du0
        elif i < i_d:
            du = min(2.0 * l / l0, l_max / l0) * du0
        else:
            du0 = 0.5 * du0
            du = du0.copy()
            restarts += 1

        xi = 0.0
        for i in range(1, i_max + 1):
            q = np.zeros((ndof, 1), dtype=float)
            q, S, E = qbar(q, T, X, G, u + du)
            if not np.all(np.isfinite(q)):
                raise RuntimeError("NaN/Inf in nlbar solver.")
            dq = q - f
            xi = float(((dq.T @ du) / (df.T @ du)).item())
            r = -dq + xi * df
            if rnorm(r, C, dof) < tol * rnorm(df, C, dof):
                break
            Kt, r_bc, _ = setbc(K.copy(), r, C, dof)
            delta_u = np.linalg.solve(Kt, r_bc)
            du = du + delta_u

        if i >= i_max:
            continue

        f = f + xi * df
        u = u + du
        U_path.append(float(u[plotdof, 0]))
        F_path.append(float(f[plotdof, 0]))
        n += 1

    q = np.zeros((ndof, 1), dtype=float)
    q, S, E = qbar(q, T, X, G, u)
    R = constraint_reactions(q, C, dof)
    return {"u": u, "q": q, "S": S, "E": E, "R": R, "f": f, "U_path": column_vector(U_path), "F_path": column_vector(F_path)}


def solve_plastic(inputs: dict[str, np.ndarray], *, plane_strain: bool) -> dict[str, np.ndarray]:
    X = inputs["X"]
    T = inputs["T"].astype(int)
    G = inputs["G"]
    C = inputs["C"]
    P = inputs["P"]
    no_loadsteps = int(scalar_input(inputs, "no_loadsteps"))
    i_max = int(scalar_input(inputs, "i_max"))
    i_d = int(scalar_input(inputs, "i_d"))
    tol = scalar_input(inputs, "TOL")
    plotdof = int(scalar_input(inputs, "plotdof")) - 1
    dof = X.shape[1]
    ndof = X.shape[0] * dof
    nelem = T.shape[0]

    f = np.zeros((ndof, 1), dtype=float)
    df = setload(np.zeros((ndof, 1), dtype=float), P)
    u = np.zeros((ndof, 1), dtype=float)
    du = np.zeros((ndof, 1), dtype=float)
    S = np.zeros((nelem, 1), dtype=float)
    E = np.zeros((nelem, 1), dtype=float)
    U_path = [0.0]
    F_path = [0.0]
    mattype = 1

    n = 1
    i = i_d
    restarts = 0
    max_restarts = no_loadsteps * i_max
    while n <= no_loadsteps:
        if restarts > max_restarts:
            raise RuntimeError("Plastic solver exceeded restart guard.")
        if i < i_max:
            K = np.zeros((ndof, ndof), dtype=float)
            if plane_strain:
                K = kq4epe(K, T, X, G, S, E, mattype)
            else:
                K = kq4eps(K, T, X, G, S, E, mattype)
            system_matrix = sp.csc_matrix(K) if sp is not None else K
            Kt, df, _ = setbc(system_matrix.copy(), df, C, dof)
            du0 = solve_plastic_system(Kt, df, plane_strain=plane_strain)
            if not np.all(np.isfinite(du0)):
                raise RuntimeError("NaN/Inf in plastic solver.")
            if float((du.T @ du0).item()) < 0.0:
                df = -df
                du0 = -du0
            if n == 1:
                l0 = float(np.linalg.norm(du0))
                l = l0
                l_max = 2.0 * l0
            else:
                l = float(np.linalg.norm(du))
                l0 = float(np.linalg.norm(du0))

        if i_d <= i < i_max:
            du = min(l / l0, l_max / l0) * du0
        elif i < i_d:
            du = min(2.0 * l / l0, l_max / l0) * du0
        else:
            du0 = 0.5 * du0
            du = du0.copy()
            restarts += 1

        xi = 0.0
        for i in range(1, i_max + 1):
            q = np.zeros((ndof, 1), dtype=float)
            if plane_strain:
                q, Sn, En = qq4epe(q, T, X, G, u + du, S, E, mattype)
            else:
                q, Sn, En = qq4eps(q, T, X, G, u + du, S, E, mattype)
            if not np.all(np.isfinite(q)):
                raise RuntimeError("NaN/Inf in plastic solver.")
            dq = q - f
            xi = float(((dq.T @ du) / (df.T @ du)).item())
            r = -dq + xi * df
            if rnorm(r, C, dof) < tol * rnorm(df, C, dof):
                break
            Kt, r_bc, _ = setbc(system_matrix.copy(), r, C, dof)
            delta_u = solve_plastic_system(Kt, r_bc, plane_strain=plane_strain)
            du = du + delta_u

        if i >= i_max:
            continue

        f = f + xi * df
        u = u + du
        S = Sn
        E = En
        U_path.append(float(u[plotdof, 0]))
        F_path.append(float(f[plotdof, 0]))
        n += 1

    q = np.zeros((ndof, 1), dtype=float)
    if plane_strain:
        q, S, E = qq4epe(q, T, X, G, u, S, E, mattype)
    else:
        q, S, E = qq4eps(q, T, X, G, u, S, E, mattype)
    R = constraint_reactions(q, C, dof)
    return {"u": u, "q": q, "S": S, "E": E, "R": R, "f": f, "U_path": column_vector(U_path), "F_path": column_vector(F_path)}


def solve_python_case(spec: CaseSpec) -> dict[str, Any]:
    inputs = load_case_inputs(spec.name)
    if not inputs:
        return {"status": "missing_inputs", "note": "MATLAB input export did not produce TSV inputs."}
    if spec.case_type == "elastic_q4":
        result = solve_elastic_q4(inputs)
    elif spec.case_type == "elastic_t3_lag":
        result = solve_elastic_t3_lag(inputs)
    elif spec.case_type == "flow_q4":
        result = solve_flow(inputs, triangular=False)
    elif spec.case_type == "flow_t3":
        result = solve_flow(inputs, triangular=True)
    elif spec.case_type == "nlbar":
        result = solve_nlbar(inputs)
    elif spec.case_type == "plastps":
        result = solve_plastic(inputs, plane_strain=False)
    elif spec.case_type == "plastpe":
        result = solve_plastic(inputs, plane_strain=True)
    else:
        return {"status": "unsupported", "note": f"Unsupported Python case type {spec.case_type}"}
    result["status"] = "ok"
    return result


def find_matlab() -> Path:
    if MATLAB_CLI.exists():
        return MATLAB_CLI
    candidates = sorted(Path(r"C:\Program Files\MATLAB").glob("R*/bin/matlab.exe"), reverse=True)
    if not candidates:
        raise FileNotFoundError("matlab.exe was not found under C:\\Program Files\\MATLAB.")
    return candidates[0]


def find_scilab() -> Path:
    if SCILAB_CLI.exists():
        return SCILAB_CLI
    candidates = sorted(Path(r"C:\Program Files").glob("scilab-*/bin/WScilex-cli.exe"), reverse=True)
    if not candidates:
        raise FileNotFoundError("WScilex-cli.exe was not found under C:\\Program Files.")
    return candidates[0]


def run_matlab_case(case_name: str) -> tuple[str, float, str]:
    matlab = find_matlab()
    log_path = LOGS / f"matlab_{case_name}.log"
    start = time.perf_counter()
    command = (
        f"addpath('{(REPO / 'scripts' / 'matlab').as_posix()}'); "
        f"run_solver_case('{case_name}', '{REPO.as_posix()}', '{OUTDIR.as_posix()}');"
    )
    start = time.perf_counter()
    try:
        completed = subprocess.run(
            [str(matlab), "-batch", command],
            cwd=REPO,
            capture_output=True,
            text=True,
            timeout=300,
        )
        elapsed = time.perf_counter() - start
        log_path.write_text((completed.stdout or "") + "\n" + (completed.stderr or ""), encoding="utf-8")
        status = "ok" if completed.returncode == 0 else "failed"
        note = "" if completed.returncode == 0 else f"Exit code {completed.returncode}"
    except subprocess.TimeoutExpired:
        elapsed = time.perf_counter() - start
        log_path.write_text("TIMEOUT after 300s\n", encoding="utf-8")
        status = "timeout"
        note = "MATLAB timed out after 300s"
    return status, elapsed, note


def scilab_common_prelude(case_name: str) -> str:
    input_dir = case_input_dir(case_name).as_posix()
    output_dir = case_raw_dir("scilab", case_name).as_posix()
    repo_dir = REPO.as_posix()
    macro_dir = (REPO / "legacy" / "scilab" / "macros").as_posix()
    return f"""
mode(0);
ieee(1);
repo = "{repo_dir}";
indir = "{input_dir}";
outdir = "{output_dir}";
macrodir = "{macro_dir}";
if ~isdir(outdir) then
    mkdir(outdir);
end
sep = ascii(9);

function data = read_tsv(path)
    if ~isfile(path) then
        data = [];
        return
    end
    data = csvRead(path, sep);
endfunction

function value = read_scalar(path, default_value)
    data = read_tsv(path);
    if size(data, "*") == 0 then
        value = default_value;
    else
        value = data(1, 1);
    end
endfunction

function write_tsv(path, data)
    if size(data, "*") == 0 then
        mputl("", path);
        return
    end
    csvWrite(data, path, sep);
endfunction

if ~isdir(macrodir) then
    error("Scilab macro directory not found: " + macrodir);
end

function R = reaction_table(q, C, dof)
    if size(C, "*") == 0 then
        R = [];
        return
    end
    if dof == 1 then
        if size(C, 2) == 2 then
            dof_col = ones(size(C, 1), 1);
            dof_no = C(:, 1);
            R = [C(:, 1), dof_col, q(dof_no)];
        else
            dof_no = (C(:, 1) - 1) * dof + C(:, 2);
            R = [C(:, 1:2), q(dof_no)];
        end
    else
        dof_no = (C(:, 1) - 1) * dof + C(:, 2);
        R = [C(:, 1:2), q(dof_no)];
    end
endfunction

"""


def scilab_case_body(spec: CaseSpec) -> str:
    if spec.case_type == "elastic_q4":
        return """
X = read_tsv(indir + "/X.tsv");
T = read_tsv(indir + "/T.tsv");
G = read_tsv(indir + "/G.tsv");
C = read_tsv(indir + "/C.tsv");
P = read_tsv(indir + "/P.tsv");
dof = int(read_scalar(indir + "/dof.tsv", 2));
T = int(T);
C = int(C);
getd(macrodir);
function [Sd, Sm] = devstress(S)
    [Sd, Sm] = devstres(S);
endfunction
[K, p, q] = init(size(X, 1), dof);
K = kq4e(K, T, X, G);
p = setload(p, P);
[K, p, ks] = setbc(K, p, C, dof);
u = K \\ p;
[q, S, E] = qq4e(q, T, X, G, u);
R = reaction_table(q, C, dof);
write_tsv(outdir + "/u.tsv", u);
write_tsv(outdir + "/q.tsv", q);
write_tsv(outdir + "/S.tsv", S);
write_tsv(outdir + "/E.tsv", E);
write_tsv(outdir + "/R.tsv", R);
"""
    if spec.case_type == "elastic_t3_lag":
        return """
X = read_tsv(indir + "/X.tsv");
T = read_tsv(indir + "/T.tsv");
G = read_tsv(indir + "/G.tsv");
C = read_tsv(indir + "/C.tsv");
P = read_tsv(indir + "/P.tsv");
dof = int(read_scalar(indir + "/dof.tsv", 2));
T = int(T);
C = int(C);
getd(macrodir);
function [Sd, Sm] = devstress(S)
    [Sd, Sm] = devstres(S);
endfunction
[K, p, q] = init(size(X, 1), dof);
K = kt3e(K, T, X, G);
p = setload(p, P);
u = solve_lag(K, p, C, dof);
[q, S, E] = qt3e(q, T, X, G, u);
R = reaction_table(q, C, dof);
write_tsv(outdir + "/u.tsv", u);
write_tsv(outdir + "/q.tsv", q);
write_tsv(outdir + "/S.tsv", S);
write_tsv(outdir + "/E.tsv", E);
write_tsv(outdir + "/R.tsv", R);
"""
    if spec.case_type in {"flow_q4", "flow_t3"}:
        assembler = "kq4p" if spec.case_type == "flow_q4" else "kt3p"
        post = "qq4p" if spec.case_type == "flow_q4" else "qt3p"
        return f"""
X = read_tsv(indir + "/X.tsv");
T = read_tsv(indir + "/T.tsv");
G = read_tsv(indir + "/G.tsv");
C = read_tsv(indir + "/C.tsv");
P = read_tsv(indir + "/P.tsv");
T = int(T);
C = int(C);
getd(macrodir);
function [Sd, Sm] = devstress(S)
    [Sd, Sm] = devstres(S);
endfunction
[K, p, q] = init(size(X, 1), 1);
K = {assembler}(K, T, X, G);
if size(P, "*") <> 0 then
    p = setload(p, P);
end
[K, p, ks] = setbc(K, p, C, 1);
u = K \\ p;
[q, S, E] = {post}(q, T, X, G, u);
R = reaction_table(q, C, 1);
write_tsv(outdir + "/u.tsv", u);
write_tsv(outdir + "/q.tsv", q);
write_tsv(outdir + "/S.tsv", S);
write_tsv(outdir + "/E.tsv", E);
write_tsv(outdir + "/R.tsv", R);
"""
    if spec.case_type == "nlbar":
        return """
X = read_tsv(indir + "/X.tsv");
T = read_tsv(indir + "/T.tsv");
G = read_tsv(indir + "/G.tsv");
C = read_tsv(indir + "/C.tsv");
P = read_tsv(indir + "/P.tsv");
no_loadsteps = int(read_scalar(indir + "/no_loadsteps.tsv", 1));
i_max = int(read_scalar(indir + "/i_max.tsv", 8));
i_d = int(read_scalar(indir + "/i_d.tsv", 3));
TOL = read_scalar(indir + "/TOL.tsv", 1.0d-3);
plotdof = int(read_scalar(indir + "/plotdof.tsv", 1));
T = int(T);
C = int(C);
getd(macrodir);
function [Sd, Sm] = devstress(S)
    [Sd, Sm] = devstres(S);
endfunction
dof = size(X, 2);
ndof = size(X, 1) * dof;
u = zeros(ndof, 1);
du = zeros(ndof, 1);
f = zeros(ndof, 1);
df = zeros(ndof, 1);
df = setload(df, P);
U = zeros(no_loadsteps + 1, 1);
F = zeros(no_loadsteps + 1, 1);
n = 1;
i = i_d;
restarts = 0;
max_restarts = no_loadsteps * i_max;
while n <= no_loadsteps
    if restarts > max_restarts then
        mprintf("Nonlinear solver exceeded restart guard.\\n");
        break
    end
    if i < i_max then
        K = zeros(ndof, ndof);
        K = kbar(K, T, X, G, u);
        [Kt, df, ks] = setbc(K, df, C, dof);
        du0 = Kt \\ df;
        if du' * du0 < 0 then
            df = -df;
            du0 = -du0;
        end
        if n == 1 then
            l0 = norm(du0);
            l = l0;
            l_max = 2 * l0;
        else
            l = norm(du);
            l0 = norm(du0);
        end
    end
    if i_d <= i & i < i_max then
        du = min(l / l0, l_max / l0) * du0;
    elseif i < i_d then
        du = min(2 * l / l0, l_max / l0) * du0;
    else
        du0 = 0.5 * du0;
        du = du0;
        restarts = restarts + 1;
    end
    for i = 1:i_max
        q = zeros(ndof, 1);
        [q, S, E] = qbar(q, T, X, G, u + du);
        dq = q - f;
        xi = (dq' * du) / (df' * du);
        r = -dq + xi * df;
        if rnorm(r, C, dof) < TOL * rnorm(df, C, dof) then
            break
        else
            [Kt, r, ks] = setbc(K, r, C, dof);
            delta_u = Kt \\ r;
            du = du + delta_u;
        end
    end
    if i < i_max then
        f = f + xi * df;
        u = u + du;
        U(n + 1) = u(plotdof);
        F(n + 1) = f(plotdof);
        n = n + 1;
    end
end
q = zeros(ndof, 1);
[q, S, E] = qbar(q, T, X, G, u);
R = reaction_table(q, C, dof);
write_tsv(outdir + "/u.tsv", u);
write_tsv(outdir + "/q.tsv", q);
write_tsv(outdir + "/S.tsv", S);
write_tsv(outdir + "/E.tsv", E);
write_tsv(outdir + "/R.tsv", R);
write_tsv(outdir + "/f.tsv", f);
write_tsv(outdir + "/path_u.tsv", U);
write_tsv(outdir + "/path_f.tsv", F);
"""
    if spec.case_type in {"plastps", "plastpe"}:
        stiffness = "kq4eps" if spec.case_type == "plastps" else "kq4epe"
        post = "qq4eps" if spec.case_type == "plastps" else "qq4epe"
        return f"""
X = read_tsv(indir + "/X.tsv");
T = read_tsv(indir + "/T.tsv");
G = read_tsv(indir + "/G.tsv");
C = read_tsv(indir + "/C.tsv");
P = read_tsv(indir + "/P.tsv");
no_loadsteps = int(read_scalar(indir + "/no_loadsteps.tsv", 1));
i_max = int(read_scalar(indir + "/i_max.tsv", 20));
i_d = int(read_scalar(indir + "/i_d.tsv", 8));
TOL = read_scalar(indir + "/TOL.tsv", 1.0d-2);
plotdof = int(read_scalar(indir + "/plotdof.tsv", 1));
mattype = 1;
T = int(T);
C = int(C);
getd(macrodir);
function [Sd, Sm] = devstress(S)
    [Sd, Sm] = devstres(S);
endfunction
dof = 2;
ndof = size(X, 1) * dof;
nelem = size(T, 1);
f = zeros(ndof, 1);
df = zeros(ndof, 1);
df = setload(df, P);
u = zeros(ndof, 1);
du = zeros(ndof, 1);
S = zeros(nelem, 1);
E = zeros(nelem, 1);
U = zeros(no_loadsteps + 1, 1);
F = zeros(no_loadsteps + 1, 1);
n = 1;
i = i_d;
restarts = 0;
max_restarts = no_loadsteps * i_max;
while n <= no_loadsteps
    if restarts > max_restarts then
        mprintf("Plastic solver exceeded restart guard.\\n");
        break
    end
    if i < i_max then
        K = zeros(ndof, ndof);
        K = {stiffness}(K, T, X, G, S, E, mattype);
        [Kt, df, ks] = setbc(K, df, C, dof);
        du0 = Kt \\ df;
        if du' * du0 < 0 then
            df = -df;
            du0 = -du0;
        end
        if n == 1 then
            l0 = norm(du0);
            l = l0;
            l_max = 2 * l0;
        else
            l = norm(du);
            l0 = norm(du0);
        end
    end
    if i_d <= i & i < i_max then
        du = min(l / l0, l_max / l0) * du0;
    elseif i < i_d then
        du = min(2 * l / l0, l_max / l0) * du0;
    else
        du0 = 0.5 * du0;
        du = du0;
        restarts = restarts + 1;
    end
    for i = 1:i_max
        q = zeros(ndof, 1);
        [q, Sn, En] = {post}(q, T, X, G, u + du, S, E, mattype);
        dq = q - f;
        xi = (dq' * du) / (df' * du);
        r = -dq + xi * df;
        if rnorm(r, C, dof) < TOL * rnorm(df, C, dof) then
            break
        else
            [Kt, r, ks] = setbc(K, r, C, dof);
            delta_u = Kt \\ r;
            du = du + delta_u;
        end
    end
    if i < i_max then
        f = f + xi * df;
        u = u + du;
        S = Sn;
        E = En;
        U(n + 1) = u(plotdof);
        F(n + 1) = f(plotdof);
        n = n + 1;
    end
end
q = zeros(ndof, 1);
[q, S, E] = {post}(q, T, X, G, u, S, E, mattype);
R = reaction_table(q, C, dof);
write_tsv(outdir + "/u.tsv", u);
write_tsv(outdir + "/q.tsv", q);
write_tsv(outdir + "/S.tsv", S);
write_tsv(outdir + "/E.tsv", E);
write_tsv(outdir + "/R.tsv", R);
write_tsv(outdir + "/f.tsv", f);
write_tsv(outdir + "/path_u.tsv", U);
write_tsv(outdir + "/path_f.tsv", F);
"""
    raise ValueError(f"Unsupported Scilab case type: {spec.case_type}")


def write_scilab_script(spec: CaseSpec) -> Path:
    tmp_dir = REPO / "tmp" / "solver_comparison"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    path = tmp_dir / f"{spec.name}.sce"
    content = scilab_common_prelude(spec.name) + "\n" + scilab_case_body(spec) + "\nexit;\n"
    path.write_text(content, encoding="utf-8")
    return path


def run_scilab_case(case_name: str) -> tuple[str, float, str]:
    scilab = find_scilab()
    log_path = LOGS / f"scilab_{case_name}.log"
    spec = next(item for item in CASES if item.name == case_name)
    runner = write_scilab_script(spec)
    start = time.perf_counter()
    try:
        completed = subprocess.run(
            [str(scilab), "-nb", "-f", str(runner)],
            cwd=REPO,
            capture_output=True,
            text=True,
            timeout=120,
        )
        elapsed = time.perf_counter() - start
        log_path.write_text((completed.stdout or "") + "\n" + (completed.stderr or ""), encoding="utf-8")
        status = "ok" if completed.returncode == 0 else "failed"
        note = "" if completed.returncode == 0 else f"Exit code {completed.returncode}"
    except subprocess.TimeoutExpired:
        elapsed = time.perf_counter() - start
        log_path.write_text("TIMEOUT after 120s\n", encoding="utf-8")
        status = "timeout"
        note = "Scilab timed out after 120s"
    return status, elapsed, note


def summarize_run(spec: CaseSpec, solver: str, status: str, runtime_sec: float, note: str) -> dict[str, Any]:
    base = case_raw_dir(solver, spec.name)
    outputs = available_outputs(base)
    inputs = load_case_inputs(spec.name)
    X = inputs.get("X")
    T = inputs.get("T")

    row: dict[str, Any] = {
        "case_name": spec.name,
        "description": spec.description,
        "case_type": spec.case_type,
        "solver": solver,
        "status": status,
        "runtime_sec": runtime_sec,
        "note": note,
        "node_count": "" if X is None else int(X.shape[0]),
        "element_count": "" if T is None else int(T.shape[0]),
        "u_size": "",
        "u_max_abs": "",
        "u_norm": "",
        "S_max_abs": "",
        "E_max_abs": "",
        "R_max_abs": "",
        "f_max_abs": "",
        "U_path_final": "",
        "F_path_final": "",
    }
    if status != "ok":
        return row

    for key, metric in (
        ("u", ("u_size", "u_max_abs", "u_norm")),
        ("S", ("S_max_abs",)),
        ("E", ("E_max_abs",)),
        ("R", ("R_max_abs",)),
        ("f", ("f_max_abs",)),
        ("U_path", ("U_path_final",)),
        ("F_path", ("F_path_final",)),
    ):
        data = outputs.get(key)
        if data is None or data.size == 0:
            continue
        flat = np.asarray(data, dtype=float).reshape(-1)
        if key == "u":
            row["u_size"] = int(flat.size)
            row["u_max_abs"] = float(np.max(np.abs(flat)))
            row["u_norm"] = float(np.linalg.norm(flat))
        elif key in ("S", "E", "R", "f"):
            row[metric[0]] = float(np.max(np.abs(flat)))
        else:
            row[metric[0]] = float(flat[-1])
    return row


def compare_outputs(
    spec: CaseSpec,
    solver_a: str,
    solver_b: str,
    run_statuses: dict[tuple[str, str], str] | None = None,
) -> dict[str, Any]:
    base_a = case_raw_dir(solver_a, spec.name)
    base_b = case_raw_dir(solver_b, spec.name)
    out_a = available_outputs(base_a)
    out_b = available_outputs(base_b)
    status_a = None if run_statuses is None else run_statuses.get((spec.name, solver_a))
    status_b = None if run_statuses is None else run_statuses.get((spec.name, solver_b))
    if status_a not in (None, "ok") or status_b not in (None, "ok"):
        pair_note = f"{solver_a}:{status_a or 'unknown'}; {solver_b}:{status_b or 'unknown'}"
        row: dict[str, Any] = {
            "case_name": spec.name,
            "description": spec.description,
            "case_type": spec.case_type,
            "solver_a": solver_a,
            "solver_b": solver_b,
            "status": "expected_failure" if "expected_failure" in {status_a, status_b} else "solver_failure",
            "note": pair_note,
            "u_max_abs_diff": "",
            "u_l2_diff": "",
            "S_max_abs_diff": "",
            "E_max_abs_diff": "",
            "R_max_abs_diff": "",
            "f_max_abs_diff": "",
            "U_path_final_diff": "",
            "F_path_final_diff": "",
        }
        return row
    missing = [name for name in ("u",) if name not in out_a or name not in out_b]
    status = "ok"
    note = ""
    if missing:
        status = "missing_output"
        note = ",".join(missing)

    row: dict[str, Any] = {
        "case_name": spec.name,
        "description": spec.description,
        "case_type": spec.case_type,
        "solver_a": solver_a,
        "solver_b": solver_b,
        "status": status,
        "note": note,
        "u_max_abs_diff": "",
        "u_l2_diff": "",
        "S_max_abs_diff": "",
        "E_max_abs_diff": "",
        "R_max_abs_diff": "",
        "f_max_abs_diff": "",
        "U_path_final_diff": "",
        "F_path_final_diff": "",
    }
    if status != "ok":
        return row

    def max_abs_diff(name: str) -> float | None:
        if name not in out_a or name not in out_b:
            return None
        a = np.asarray(out_a[name], dtype=float)
        b = np.asarray(out_b[name], dtype=float)
        if a.shape != b.shape:
            row["status"] = "shape_mismatch"
            row["note"] = f"{name}:{a.shape}!={b.shape}"
            return None
        return float(np.max(np.abs(a - b)))

    u_diff = max_abs_diff("u")
    if u_diff is not None:
        row["u_max_abs_diff"] = u_diff
        row["u_l2_diff"] = float(np.linalg.norm(out_a["u"] - out_b["u"]))
    for name, column in (
        ("S", "S_max_abs_diff"),
        ("E", "E_max_abs_diff"),
        ("R", "R_max_abs_diff"),
        ("f", "f_max_abs_diff"),
    ):
        value = max_abs_diff(name)
        if value is not None:
            row[column] = value
    for name, column in (("U_path", "U_path_final_diff"), ("F_path", "F_path_final_diff")):
        value = max_abs_diff(name)
        if value is not None:
            row[column] = float(
                abs(np.asarray(out_a[name], dtype=float).reshape(-1)[-1] - np.asarray(out_b[name], dtype=float).reshape(-1)[-1])
            )
    return row


def write_rows_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: "" if value is None else value for key, value in row.items()})


def latex_escape(text: str) -> str:
    return (
        text.replace("\\", "\\textbackslash{}")
        .replace("_", "\\_")
        .replace("&", "\\&")
        .replace("%", "\\%")
        .replace("#", "\\#")
    )


def write_plot_table(path: Path, header: list[str], rows: list[list[Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(header)
        writer.writerows(rows)


def generate_plot_inputs(summary_rows: list[dict[str, Any]], comparison_rows: list[dict[str, Any]]) -> None:
    runtime_rows: list[list[Any]] = []
    for row in summary_rows:
        runtime_rows.append([row["case_name"], row["solver"], row["runtime_sec"] if row["status"] == "ok" else math.nan])
    write_plot_table(PLOTS / "runtime_plot.tsv", ["case_name", "solver", "runtime_sec"], runtime_rows)

    u_rows: list[list[Any]] = []
    s_rows: list[list[Any]] = []
    for row in comparison_rows:
        pair = f"{row['solver_a']}-{row['solver_b']}"
        u_value = row["u_max_abs_diff"] if row["u_max_abs_diff"] != "" else math.nan
        s_value = row["S_max_abs_diff"] if row["S_max_abs_diff"] != "" else math.nan
        u_rows.append([row["case_name"], pair, u_value])
        s_rows.append([row["case_name"], pair, s_value])
    write_plot_table(PLOTS / "u_diff_plot.tsv", ["case_name", "pair", "u_max_abs_diff"], u_rows)
    write_plot_table(PLOTS / "stress_diff_plot.tsv", ["case_name", "pair", "S_max_abs_diff"], s_rows)


def write_plot_tex() -> list[Path]:
    plots = {
        "runtime_by_solver.tex": r"""
\documentclass[tikz,border=4pt]{standalone}
\usepackage{pgfplots}
\usepgfplotslibrary{groupplots}
\pgfplotsset{compat=1.18}
\begin{document}
\begin{tikzpicture}
\begin{axis}[
  width=16cm,
  height=8cm,
  ybar,
  bar width=8pt,
  ymin=0,
  ylabel={Runtime (s)},
  symbolic x coords={cantilever_q4,gmsh_triangle_t3,flow_q4,flow_t3,bar01_nlbar,bar02_nlbar,bar03_nlbar,square_plastps,square_plastpe,hole_plastps,hole_plastpe},
  xtick=data,
  xticklabel style={rotate=40,anchor=east,font=\scriptsize},
  legend style={at={(0.5,1.1)},anchor=south,legend columns=3},
  enlargelimits=0.03,
]
\addplot table[x=case_name,y=runtime_sec,col sep=tab,row predicate/.code={\ifnum\pdfstrcmp{\thisrow{solver}}{matlab}=0\else\pgfplotstableuserowfalse\fi}] {runtime_plot.tsv};
\addplot table[x=case_name,y=runtime_sec,col sep=tab,row predicate/.code={\ifnum\pdfstrcmp{\thisrow{solver}}{python}=0\else\pgfplotstableuserowfalse\fi}] {runtime_plot.tsv};
\addplot table[x=case_name,y=runtime_sec,col sep=tab,row predicate/.code={\ifnum\pdfstrcmp{\thisrow{solver}}{scilab}=0\else\pgfplotstableuserowfalse\fi}] {runtime_plot.tsv};
\legend{MATLAB,Python,Scilab}
\end{axis}
\end{tikzpicture}
\end{document}
""",
        "u_diff_vs_baseline.tex": r"""
\documentclass[tikz,border=4pt]{standalone}
\usepackage{pgfplots}
\pgfplotsset{compat=1.18}
\begin{document}
\begin{tikzpicture}
\begin{axis}[
  width=16cm,
  height=8cm,
  ybar,
  bar width=8pt,
  ymode=log,
  ymin=1e-18,
  ylabel={Max $\lvert \Delta u \rvert$},
  symbolic x coords={cantilever_q4,gmsh_triangle_t3,flow_q4,flow_t3,bar01_nlbar,bar02_nlbar,bar03_nlbar,square_plastps,square_plastpe,hole_plastps,hole_plastpe},
  xtick=data,
  xticklabel style={rotate=40,anchor=east,font=\scriptsize},
  legend style={at={(0.5,1.1)},anchor=south,legend columns=3},
  enlargelimits=0.03,
  unbounded coords=discard,
]
\addplot table[x=case_name,y=u_max_abs_diff,col sep=tab,row predicate/.code={\ifnum\pdfstrcmp{\thisrow{pair}}{python-matlab}=0\else\pgfplotstableuserowfalse\fi}] {u_diff_plot.tsv};
\addplot table[x=case_name,y=u_max_abs_diff,col sep=tab,row predicate/.code={\ifnum\pdfstrcmp{\thisrow{pair}}{scilab-matlab}=0\else\pgfplotstableuserowfalse\fi}] {u_diff_plot.tsv};
\addplot table[x=case_name,y=u_max_abs_diff,col sep=tab,row predicate/.code={\ifnum\pdfstrcmp{\thisrow{pair}}{python-scilab}=0\else\pgfplotstableuserowfalse\fi}] {u_diff_plot.tsv};
\legend{Python-MATLAB,Scilab-MATLAB,Python-Scilab}
\end{axis}
\end{tikzpicture}
\end{document}
""",
        "stress_diff_vs_baseline.tex": r"""
\documentclass[tikz,border=4pt]{standalone}
\usepackage{pgfplots}
\pgfplotsset{compat=1.18}
\begin{document}
\begin{tikzpicture}
\begin{axis}[
  width=16cm,
  height=8cm,
  ybar,
  bar width=8pt,
  ymode=log,
  ymin=1e-18,
  ylabel={Max $\lvert \Delta \sigma \rvert$},
  symbolic x coords={cantilever_q4,gmsh_triangle_t3,flow_q4,flow_t3,bar01_nlbar,bar02_nlbar,bar03_nlbar,square_plastps,square_plastpe,hole_plastps,hole_plastpe},
  xtick=data,
  xticklabel style={rotate=40,anchor=east,font=\scriptsize},
  legend style={at={(0.5,1.1)},anchor=south,legend columns=3},
  enlargelimits=0.03,
  unbounded coords=discard,
]
\addplot table[x=case_name,y=S_max_abs_diff,col sep=tab,row predicate/.code={\ifnum\pdfstrcmp{\thisrow{pair}}{python-matlab}=0\else\pgfplotstableuserowfalse\fi}] {stress_diff_plot.tsv};
\addplot table[x=case_name,y=S_max_abs_diff,col sep=tab,row predicate/.code={\ifnum\pdfstrcmp{\thisrow{pair}}{scilab-matlab}=0\else\pgfplotstableuserowfalse\fi}] {stress_diff_plot.tsv};
\addplot table[x=case_name,y=S_max_abs_diff,col sep=tab,row predicate/.code={\ifnum\pdfstrcmp{\thisrow{pair}}{python-scilab}=0\else\pgfplotstableuserowfalse\fi}] {stress_diff_plot.tsv};
\legend{Python-MATLAB,Scilab-MATLAB,Python-Scilab}
\end{axis}
\end{tikzpicture}
\end{document}
""",
    }
    written: list[Path] = []
    for name, content in plots.items():
        path = PLOTS / name
        path.write_text(content.strip() + "\n", encoding="utf-8")
        written.append(path)
    return written


def compile_plot(tex_path: Path) -> None:
    result = subprocess.run(
        ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", tex_path.name],
        cwd=tex_path.parent,
        check=False,
        timeout=180,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if result.returncode != 0:
        print(f"  pdflatex failed for {tex_path.name} (exit {result.returncode})")
        return
    pdf_path = tex_path.with_suffix(".pdf")
    png_path = tex_path.with_suffix(".png")
    result = subprocess.run(
        ["pdftocairo", "-png", "-singlefile", "-r", "180", str(pdf_path), str(png_path.with_suffix(""))],
        cwd=tex_path.parent,
        check=False,
        timeout=180,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if result.returncode != 0:
        print(f"  pdftocairo failed for {tex_path.name} (exit {result.returncode})")


def generate_readme_stub(summary_rows: list[dict[str, Any]], comparison_rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Solver Comparison",
        "",
        "Generated by `scripts/generate_solver_comparison.py`.",
        "",
        "This live regression covers all 11 classroom benchmark cases.",
        "",
        "The original `bar03.m` truss is included as an explicit expected-failure case "
        "because the legacy load-stepping method is known to diverge across implementations.",
        "",
        "For the small plane-strain plasticity benchmarks, the Python solver uses "
        "a symmetry-aware dense fallback to better match MATLAB's historical sparse "
        "backslash path.",
        "",
        "## Run Summary",
        "",
        "| Case | Solver | Status | Runtime (s) | `|u|_max` | `|S|_max` | Note |",
        "| --- | --- | --- | ---: | ---: | ---: | --- |",
    ]
    for row in summary_rows:
        lines.append(
            f"| `{row['case_name']}` | `{row['solver']}` | `{row['status']}` | "
            f"{row['runtime_sec']:.2f} | {row['u_max_abs'] or ''} | {row['S_max_abs'] or ''} | {row['note'] or ''} |"
        )
    lines.extend(
        [
            "",
            "## Pairwise Diff Summary",
            "",
            "| Case | Pair | Status | Max `|Δu|` | Max `|Δσ|` | Note |",
            "| --- | --- | --- | ---: | ---: | --- |",
        ]
    )
    for row in comparison_rows:
        lines.append(
            f"| `{row['case_name']}` | `{row['solver_a']}-{row['solver_b']}` | `{row['status']}` | "
            f"{row['u_max_abs_diff'] or ''} | {row['S_max_abs_diff'] or ''} | {row['note'] or ''} |"
        )
    (OUTDIR / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_python_case(spec: CaseSpec) -> tuple[str, float, str]:
    base = case_raw_dir("python", spec.name)
    base.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    try:
        outputs = solve_python_case(spec)
        status = outputs.get("status", "ok")
        note = outputs.get("note", "")
        if status == "ok":
            export_case_outputs(base, outputs)
    except Exception as exc:  # pragma: no cover
        status = "failed"
        note = str(exc)
    return status, time.perf_counter() - start, note


def classify_run_status(spec: CaseSpec, solver: str, status: str, note: str) -> tuple[str, str]:
    """Translate raw runner status into a report status with expected-failure handling."""

    expected = spec.expected_statuses.get(solver, ())
    if status in expected:
        detail = note.strip() if note else status
        return "expected_failure", f"Expected legacy divergence: {detail}"
    return status, note


def run_all(case_names: list[str] | None = None, solvers: list[str] | None = None) -> None:
    ensure_dirs()
    prepare_inputs()

    specs = list(CASES)
    if case_names:
        by_name = {s.name: s for s in CASES}
        specs = [by_name[n] for n in case_names if n in by_name]
        unknown = [n for n in case_names if n not in by_name]
        if unknown:
            print(f"Unknown cases ignored: {unknown}")
    run_solvers = solvers or list(SOLVERS)
    active_pairs = [pair for pair in PAIRWISE if pair[0] in run_solvers and pair[1] in run_solvers]

    summary_rows: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []
    manifest: dict[str, Any] = {"repo": str(REPO), "generated_at_epoch": time.time(), "cases": []}
    run_statuses: dict[tuple[str, str], str] = {}

    for idx, spec in enumerate(specs, 1):
        print(f"[{idx}/{len(specs)}] {spec.name}")

        if "matlab" in run_solvers:
            print(f"  MATLAB ...", end=" ", flush=True)
            matlab_status, matlab_runtime, matlab_note = run_matlab_case(spec.name)
            print(f"{matlab_status} ({matlab_runtime:.1f}s)")
            matlab_status, matlab_note = classify_run_status(spec, "matlab", matlab_status, matlab_note)
            run_statuses[(spec.name, "matlab")] = matlab_status
            summary_rows.append(summarize_run(spec, "matlab", matlab_status, matlab_runtime, matlab_note))

        if "python" in run_solvers:
            print(f"  Python ...", end=" ", flush=True)
            python_status, python_runtime, python_note = run_python_case(spec)
            print(f"{python_status} ({python_runtime:.1f}s)")
            python_status, python_note = classify_run_status(spec, "python", python_status, python_note)
            run_statuses[(spec.name, "python")] = python_status
            summary_rows.append(summarize_run(spec, "python", python_status, python_runtime, python_note))

        if "scilab" in run_solvers:
            print(f"  Scilab ...", end=" ", flush=True)
            scilab_status, scilab_runtime, scilab_note = run_scilab_case(spec.name)
            print(f"{scilab_status} ({scilab_runtime:.1f}s)")
            scilab_status, scilab_note = classify_run_status(spec, "scilab", scilab_status, scilab_note)
            run_statuses[(spec.name, "scilab")] = scilab_status
            summary_rows.append(summarize_run(spec, "scilab", scilab_status, scilab_runtime, scilab_note))

        for solver_a, solver_b in active_pairs:
            comparison_rows.append(compare_outputs(spec, solver_a, solver_b, run_statuses))

        manifest["cases"].append(
            {
                "name": spec.name,
                "case_type": spec.case_type,
                "description": spec.description,
                "summary": [row for row in summary_rows if row["case_name"] == spec.name],
            }
        )

    write_rows_tsv(OUTDIR / "summary_runs.tsv", summary_rows)
    write_rows_tsv(OUTDIR / "pairwise_comparison.tsv", comparison_rows)
    generate_plot_inputs(summary_rows, comparison_rows)
    write_plot_tex()
    generate_readme_stub(summary_rows, comparison_rows)
    (OUTDIR / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    import argparse

    all_case_names = [s.name for s in CASES]
    parser = argparse.ArgumentParser(description="Run FemLab solver comparison.")
    parser.add_argument("cases", nargs="*", default=None,
                        help=f"Case names to run (default: all). Choices: {all_case_names}")
    parser.add_argument("--solver", "-s", action="append", dest="solvers",
                        choices=list(SOLVERS),
                        help="Run only these solvers (repeatable, default: all)")
    parser.add_argument("--list", "-l", action="store_true",
                        help="List available case names and exit")
    args = parser.parse_args()
    if args.list:
        for s in CASES:
            print(f"  {s.name:25s}  ({s.case_type})  {s.description}")
        sys.exit(0)
    run_all(case_names=args.cases or None, solvers=args.solvers)
