from __future__ import annotations

import numpy as np

from femlab.examples import run_cantilever, run_flow_q4, run_flow_t3, run_gmsh_triangle


def test_cantilever_runs_and_deflects_downward():
    result = run_cantilever(plot=False)
    assert result["u"].shape == (54, 1)
    assert result["S"].shape == (16, 12)
    assert result["u"][-1, 0] < 0.0


def test_gmsh_triangle_runs():
    result = run_gmsh_triangle(plot=False)
    assert result["u"].shape[0] == result["data"]["X"].shape[0] * 2
    assert result["S"].shape[0] == result["data"]["T"].shape[0]
    assert np.isfinite(result["u"]).all()


def test_flow_examples_stay_within_boundary_range():
    q4 = run_flow_q4(plot=False)
    t3 = run_flow_t3(plot=False)
    for result in (q4, t3):
        assert np.isfinite(result["u"]).all()
        assert result["u"].min() >= 19.9
        assert result["u"].max() <= 40.1
