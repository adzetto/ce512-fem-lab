from __future__ import annotations

import numpy as np

from femlabpy.examples import run_ex_lag_mult


def test_ex_lag_mult_runs_and_satisfies_constraints():
    result = run_ex_lag_mult()

    assert result["U"].shape == (6, 1)
    assert result["Lag"].shape == (3, 1)
    assert result["member_forces"].shape == (4, 3)
    assert np.allclose(result["constraint_residual"], 0.0, atol=1.0e-12)

    expected_u = np.array(
        [0.0, 0.0, -0.280386276499, -0.485643276499, 0.073955417288, -0.798143276499]
    )
    assert np.allclose(result["U"].ravel(), expected_u, atol=1.0e-10)
