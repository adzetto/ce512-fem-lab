from __future__ import annotations

import numpy as np

from femlab import keT4e, keh8e


def test_tetra_stiffness_shape_and_symmetry():
    Xe = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
    Ge = np.array([100.0, 0.25], dtype=float)
    Ke = keT4e(Xe, Ge)
    assert Ke.shape == (12, 12)
    assert np.allclose(Ke, Ke.T)


def test_hexa_stiffness_shape_and_symmetry():
    Xe = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
            [1, 1, 1],
            [0, 1, 1],
        ],
        dtype=float,
    )
    Ge = np.array([100.0, 0.25], dtype=float)
    Ke = keh8e(Xe, Ge)
    assert Ke.shape == (24, 24)
    assert np.allclose(Ke, Ke.T)
