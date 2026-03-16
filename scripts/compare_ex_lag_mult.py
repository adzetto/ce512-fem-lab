from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
OUTDIR = REPO / "tmp" / "ex_lag_mult_compare"
LOGDIR = OUTDIR / "logs"
FIELDS = (
    "U",
    "Lag",
    "R",
    "member_forces",
    "local_displacements",
    "constraint_residual",
)

MATLAB_CLI = Path(r"C:\Program Files\MATLAB\R2025b\bin\matlab.exe")
SCILAB_CLI = Path(r"C:\Program Files\scilab-2025.0.0\bin\WScilex-cli.exe")
JULIA_CLI = Path(r"C:\Users\lenovo\AppData\Local\Programs\Julia-1.12.5\bin\julia.exe")

sys.path.insert(0, str(REPO / "src"))

from femlab.examples import run_ex_lag_mult  # noqa: E402


def write_tsv(path: Path, data: Any) -> None:
    array = np.asarray(data, dtype=float)
    path.parent.mkdir(parents=True, exist_ok=True)
    if array.ndim == 0:
        array = array.reshape(1, 1)
    elif array.ndim == 1:
        array = array.reshape(-1, 1)
    np.savetxt(path, array, delimiter="\t", fmt="%.16g")


def read_tsv(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return np.zeros((0, 0), dtype=float)
    return np.loadtxt(path, delimiter="\t", ndmin=2)


def find_matlab() -> Path:
    if MATLAB_CLI.exists():
        return MATLAB_CLI
    candidates = sorted(
        Path(r"C:\Program Files\MATLAB").glob("R*/bin/matlab.exe"),
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError("matlab.exe was not found.")
    return candidates[0]


def find_scilab() -> Path:
    if SCILAB_CLI.exists():
        return SCILAB_CLI
    candidates = sorted(
        Path(r"C:\Program Files").glob("scilab-*/bin/WScilex-cli.exe"),
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError("WScilex-cli.exe was not found.")
    return candidates[0]


def find_julia() -> Path:
    if JULIA_CLI.exists():
        return JULIA_CLI
    candidates = sorted(
        Path.home().glob(r"AppData/Local/Programs/Julia-*/bin/julia.exe"),
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError("julia.exe was not found.")
    return candidates[0]


def export_python(outdir: Path) -> tuple[str, float, str]:
    start = time.perf_counter()
    result = run_ex_lag_mult()
    elapsed = time.perf_counter() - start
    for field in FIELDS:
        write_tsv(outdir / f"{field}.tsv", result[field])
    return "ok", elapsed, ""


def run_matlab(outdir: Path) -> tuple[str, float, str]:
    matlab = find_matlab()
    command = (
        f"output_dir = '{outdir.as_posix()}'; "
        f"run('{(REPO / 'scripts' / 'matlab' / 'ex_lag_mult.m').as_posix()}');"
    )
    start = time.perf_counter()
    try:
        completed = subprocess.run(
            [str(matlab), "-batch", command],
            cwd=REPO,
            capture_output=True,
            text=True,
            timeout=120,
        )
        elapsed = time.perf_counter() - start
        (LOGDIR / "matlab.log").write_text(
            (completed.stdout or "") + "\n" + (completed.stderr or ""),
            encoding="utf-8",
        )
        if completed.returncode != 0:
            return "failed", elapsed, f"Exit code {completed.returncode}"
        return "ok", elapsed, ""
    except Exception as exc:  # pragma: no cover - tool availability
        elapsed = time.perf_counter() - start
        return "failed", elapsed, str(exc)


def run_scilab(outdir: Path) -> tuple[str, float, str]:
    scilab = find_scilab()
    runner = REPO / "scripts" / "scilab" / "ex_lag_mult.sce"
    env = os.environ.copy()
    env["EX_LAG_MULT_OUTDIR"] = outdir.as_posix()
    start = time.perf_counter()
    try:
        completed = subprocess.run(
            [str(scilab), "-nb", "-f", str(runner)],
            cwd=REPO,
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
        )
        elapsed = time.perf_counter() - start
        (LOGDIR / "scilab.log").write_text(
            (completed.stdout or "") + "\n" + (completed.stderr or ""),
            encoding="utf-8",
        )
        if completed.returncode != 0:
            return "failed", elapsed, f"Exit code {completed.returncode}"
        return "ok", elapsed, ""
    except Exception as exc:  # pragma: no cover - tool availability
        elapsed = time.perf_counter() - start
        return "failed", elapsed, str(exc)


def run_julia(outdir: Path) -> tuple[str, float, str]:
    julia = find_julia()
    runner = REPO / "scripts" / "julia" / "ex_lag_mult.jl"
    start = time.perf_counter()
    try:
        completed = subprocess.run(
            [str(julia), str(runner), outdir.as_posix()],
            cwd=REPO,
            capture_output=True,
            text=True,
            timeout=120,
        )
        elapsed = time.perf_counter() - start
        (LOGDIR / "julia.log").write_text(
            (completed.stdout or "") + "\n" + (completed.stderr or ""),
            encoding="utf-8",
        )
        if completed.returncode != 0:
            return "failed", elapsed, f"Exit code {completed.returncode}"
        return "ok", elapsed, ""
    except Exception as exc:  # pragma: no cover - tool availability
        elapsed = time.perf_counter() - start
        return "failed", elapsed, str(exc)


def compare_arrays(a: np.ndarray | None, b: np.ndarray | None) -> dict[str, Any]:
    if a is None or b is None:
        return {"status": "missing"}
    if a.shape != b.shape:
        return {"status": "shape_mismatch", "shape_a": a.shape, "shape_b": b.shape}
    diff = a - b
    return {
        "status": "ok",
        "max_abs_diff": float(np.max(np.abs(diff))) if diff.size else 0.0,
        "l2_diff": float(np.linalg.norm(diff)),
    }


def main() -> int:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    LOGDIR.mkdir(parents=True, exist_ok=True)

    runners = {
        "python": export_python,
        "scilab": run_scilab,
        "matlab": run_matlab,
        "julia": run_julia,
    }

    runs: dict[str, dict[str, Any]] = {}
    for solver, runner in runners.items():
        solver_dir = OUTDIR / solver
        solver_dir.mkdir(parents=True, exist_ok=True)
        status, runtime_sec, note = runner(solver_dir)
        runs[solver] = {
            "status": status,
            "runtime_sec": runtime_sec,
            "note": note,
        }

    comparisons: dict[str, dict[str, Any]] = {}
    pairs = [
        ("python", "scilab"),
        ("python", "matlab"),
        ("python", "julia"),
        ("scilab", "matlab"),
        ("scilab", "julia"),
        ("matlab", "julia"),
    ]
    for left, right in pairs:
        pair_key = f"{left}_vs_{right}"
        field_metrics: dict[str, Any] = {}
        for field in FIELDS:
            arr_left = read_tsv(OUTDIR / left / f"{field}.tsv")
            arr_right = read_tsv(OUTDIR / right / f"{field}.tsv")
            field_metrics[field] = compare_arrays(arr_left, arr_right)
        comparisons[pair_key] = field_metrics

    summary = {"runs": runs, "comparisons": comparisons}
    summary_path = OUTDIR / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"\nSummary written to {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
