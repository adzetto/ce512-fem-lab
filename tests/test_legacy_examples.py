from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from femlabpy import devstres, devstress, eqstress, setpath, stressvm, yieldvm
from femlabpy.examples import (
    bar01_data,
    hole_data,
    run_bar01_nlbar,
    run_hole_plastpe,
    run_square_plastpe,
    run_square_plastps,
    square_data,
)

REPO = Path(__file__).resolve().parents[1]


def _read_tsv(path: Path) -> np.ndarray:
    return np.loadtxt(path, dtype=float, ndmin=2)


def test_top_level_legacy_helpers_are_exported():
    package_paths = setpath()
    assert "package" in package_paths
    assert "examples" in package_paths
    assert str(package_paths["examples"]) in sys.path
    assert np.allclose(devstres([3.0, 1.0, 2.0])[0], devstress([3.0, 1.0, 2.0])[0])
    assert np.isclose(eqstress([1.0, 0.5, 0.25]), eqstress(np.array([1.0, 0.5, 0.25])))
    assert np.isfinite(
        yieldvm(np.array([1.0, 0.2, 0.1]), np.array([100.0, 0.3, 1.0, 10.0]), 0.0, 1.0)
    )
    updated_stress, plastic_increment = stressvm(
        np.array([2.0, 0.5, 0.1]), np.array([100.0, 0.3, 1.0, 10.0]), 1.0
    )
    assert np.asarray(updated_stress, dtype=float).reshape(-1).shape == (3,)
    assert plastic_increment >= 0.0


def test_packaged_case_data_loaders_expose_original_examples():
    bar_data = bar01_data()
    square = square_data(plane_strain=False)
    hole = hole_data(plane_strain=True)
    assert bar_data["X"].shape == (3, 2)
    assert "plotaxis" in bar_data
    assert "elaxis" in bar_data
    assert square["T"].shape[1] == 5
    assert "strainaxis" in square
    assert "stressaxis" in square
    assert int(hole["dof"]) == 2
    assert "elaxis" in hole


def test_bar01_runner_matches_scilab_benchmark():
    result = run_bar01_nlbar(plot=False)
    baseline_dir = REPO / "benchmarks" / "raw" / "scilab" / "bar01_nlbar"
    u_expected = _read_tsv(baseline_dir / "u.tsv")
    f_expected = _read_tsv(baseline_dir / "path_f.tsv")
    assert np.allclose(result["u"], u_expected, atol=1.0e-12)
    assert np.allclose(result["F_path"], f_expected, atol=1.0e-12)


def test_square_plane_stress_runner_matches_matlab_benchmark():
    result = run_square_plastps(plot=False)
    baseline_dir = REPO / "benchmarks" / "raw" / "matlab" / "square_plastps"
    u_expected = _read_tsv(baseline_dir / "u.tsv")
    e_expected = _read_tsv(baseline_dir / "E.tsv")
    assert np.allclose(result["u"], u_expected, atol=1.0e-5)
    assert np.allclose(result["E"], e_expected, atol=1.0e-5)


def test_plane_strain_runners_stay_close_to_scilab_benchmarks():
    square_result = run_square_plastpe(plot=False)
    square_dir = REPO / "benchmarks" / "raw" / "scilab" / "square_plastpe"
    square_u = _read_tsv(square_dir / "u.tsv")
    square_s = _read_tsv(square_dir / "S.tsv")
    assert np.max(np.abs(square_result["u"] - square_u)) < 3.0e-3
    assert np.max(np.abs(square_result["S"] - square_s)) < 3.0e-2

    hole_result = run_hole_plastpe(plot=False)
    hole_dir = REPO / "benchmarks" / "raw" / "scilab" / "hole_plastpe"
    hole_u = _read_tsv(hole_dir / "u.tsv")
    hole_s = _read_tsv(hole_dir / "S.tsv")
    assert np.max(np.abs(hole_result["u"] - hole_u)) < 2.0e-4
    assert np.max(np.abs(hole_result["S"] - hole_s)) < 4.0e-3
