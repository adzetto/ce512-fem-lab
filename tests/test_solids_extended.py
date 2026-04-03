from __future__ import annotations

import numpy as np
import pytest

from femlabpy import keT4e, keh8e, kT4e, kh8e, qeT4e, qeh8e, qT4e, qh8e


T4_CASES = [
    pytest.param(
        np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            dtype=float,
        ),
        np.array([100.0, 0.25], dtype=float),
        np.array(
            [
                [0.00, 0.00, 0.00],
                [0.01, 0.00, 0.00],
                [0.00, 0.02, 0.00],
                [0.00, 0.00, 0.03],
            ],
            dtype=float,
        ),
        id="unit-tet",
    ),
    pytest.param(
        np.array(
            [[0.2, 0.1, 0.0], [1.5, 0.2, 0.1], [0.3, 1.4, 0.2], [0.4, 0.5, 1.6]],
            dtype=float,
        ),
        np.array([150.0, 0.30], dtype=float),
        np.array(
            [
                [0.00, 0.00, 0.00],
                [0.02, 0.01, 0.00],
                [0.00, 0.03, 0.01],
                [0.01, 0.02, 0.04],
            ],
            dtype=float,
        ),
        id="skew-tet",
    ),
    pytest.param(
        np.array(
            [[1.0, 0.0, 0.0], [2.0, 0.1, 0.0], [1.2, 1.3, 0.2], [1.1, 0.4, 1.5]],
            dtype=float,
        ),
        np.array([180.0, 0.20], dtype=float),
        np.array(
            [
                [0.00, 0.00, 0.00],
                [0.01, 0.01, 0.00],
                [0.02, 0.03, 0.01],
                [0.01, 0.01, 0.05],
            ],
            dtype=float,
        ),
        id="offset-tet",
    ),
    pytest.param(
        np.array(
            [[0.0, 0.0, 0.0], [1.2, 0.0, 0.2], [0.1, 1.1, 0.1], [0.2, 0.3, 1.4]],
            dtype=float,
        ),
        np.array([120.0, 0.28], dtype=float),
        np.array(
            [
                [0.00, 0.00, 0.00],
                [0.03, 0.00, 0.01],
                [0.01, 0.02, 0.00],
                [0.01, 0.02, 0.03],
            ],
            dtype=float,
        ),
        id="sheared-tet",
    ),
]

T4_RIGID_CASES = [
    pytest.param(
        T4_CASES[0].values[0],
        T4_CASES[0].values[1],
        np.array([[1.0, 0.0, 0.0]] * 4, dtype=float),
        id="tet-tx",
    ),
    pytest.param(
        T4_CASES[0].values[0],
        T4_CASES[0].values[1],
        np.array([[0.0, 1.0, 0.0]] * 4, dtype=float),
        id="tet-ty",
    ),
    pytest.param(
        T4_CASES[0].values[0],
        T4_CASES[0].values[1],
        np.array([[0.0, 0.0, 1.0]] * 4, dtype=float),
        id="tet-tz",
    ),
    pytest.param(
        T4_CASES[1].values[0],
        T4_CASES[1].values[1],
        np.array([[1.0, 0.0, 0.0]] * 4, dtype=float),
        id="skew-tx",
    ),
    pytest.param(
        T4_CASES[1].values[0],
        T4_CASES[1].values[1],
        np.array([[0.0, 1.0, 0.0]] * 4, dtype=float),
        id="skew-ty",
    ),
    pytest.param(
        T4_CASES[1].values[0],
        T4_CASES[1].values[1],
        np.array([[0.0, 0.0, 1.0]] * 4, dtype=float),
        id="skew-tz",
    ),
]

H8_CASES = [
    pytest.param(
        np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 1.0],
                [1.0, 1.0, 1.0],
                [0.0, 1.0, 1.0],
            ],
            dtype=float,
        ),
        np.array([100.0, 0.25], dtype=float),
        np.array(
            [
                [0.00, 0.00, 0.00],
                [0.01, 0.00, 0.00],
                [0.02, 0.01, 0.00],
                [0.00, 0.02, 0.00],
                [0.00, 0.00, 0.02],
                [0.01, 0.00, 0.02],
                [0.02, 0.01, 0.03],
                [0.00, 0.02, 0.03],
            ],
            dtype=float,
        ),
        id="unit-hexa",
    ),
    pytest.param(
        np.array(
            [
                [0.0, 0.0, 0.0],
                [2.0, 0.0, 0.1],
                [2.2, 1.1, 0.0],
                [0.2, 1.0, -0.1],
                [0.1, 0.0, 1.2],
                [2.1, 0.1, 1.1],
                [2.3, 1.1, 1.2],
                [0.3, 1.0, 1.3],
            ],
            dtype=float,
        ),
        np.array([150.0, 0.30], dtype=float),
        np.array(
            [
                [0.00, 0.00, 0.00],
                [0.02, 0.00, 0.00],
                [0.03, 0.01, 0.01],
                [0.01, 0.02, 0.00],
                [0.00, 0.00, 0.03],
                [0.02, 0.00, 0.03],
                [0.03, 0.01, 0.04],
                [0.01, 0.02, 0.04],
            ],
            dtype=float,
        ),
        id="skew-hexa",
    ),
    pytest.param(
        np.array(
            [
                [1.0, 0.0, 0.0],
                [3.0, 0.1, 0.0],
                [3.1, 2.0, 0.1],
                [1.1, 1.9, 0.0],
                [1.0, 0.1, 1.5],
                [3.0, 0.2, 1.4],
                [3.2, 2.1, 1.6],
                [1.2, 2.0, 1.5],
            ],
            dtype=float,
        ),
        np.array([180.0, 0.20], dtype=float),
        np.array(
            [
                [0.00, 0.00, 0.00],
                [0.01, 0.00, 0.00],
                [0.03, 0.01, 0.01],
                [0.01, 0.02, 0.00],
                [0.00, 0.00, 0.04],
                [0.01, 0.00, 0.04],
                [0.03, 0.01, 0.05],
                [0.01, 0.02, 0.05],
            ],
            dtype=float,
        ),
        id="offset-hexa",
    ),
    pytest.param(
        np.array(
            [
                [0.0, 0.0, 0.0],
                [1.2, 0.0, 0.2],
                [1.4, 1.1, 0.1],
                [0.1, 1.0, 0.0],
                [0.1, 0.1, 1.3],
                [1.3, 0.1, 1.4],
                [1.5, 1.2, 1.5],
                [0.2, 1.1, 1.4],
            ],
            dtype=float,
        ),
        np.array([130.0, 0.28], dtype=float),
        np.array(
            [
                [0.00, 0.00, 0.00],
                [0.02, 0.00, 0.01],
                [0.03, 0.02, 0.01],
                [0.01, 0.02, 0.00],
                [0.00, 0.00, 0.03],
                [0.02, 0.00, 0.04],
                [0.03, 0.02, 0.04],
                [0.01, 0.02, 0.03],
            ],
            dtype=float,
        ),
        id="sheared-hexa",
    ),
]

H8_RIGID_CASES = [
    pytest.param(
        H8_CASES[0].values[0],
        H8_CASES[0].values[1],
        np.array([[1.0, 0.0, 0.0]] * 8, dtype=float),
        id="hexa-tx",
    ),
    pytest.param(
        H8_CASES[0].values[0],
        H8_CASES[0].values[1],
        np.array([[0.0, 1.0, 0.0]] * 8, dtype=float),
        id="hexa-ty",
    ),
    pytest.param(
        H8_CASES[0].values[0],
        H8_CASES[0].values[1],
        np.array([[0.0, 0.0, 1.0]] * 8, dtype=float),
        id="hexa-tz",
    ),
    pytest.param(
        H8_CASES[1].values[0],
        H8_CASES[1].values[1],
        np.array([[1.0, 0.0, 0.0]] * 8, dtype=float),
        id="skew-hexa-tx",
    ),
    pytest.param(
        H8_CASES[1].values[0],
        H8_CASES[1].values[1],
        np.array([[0.0, 1.0, 0.0]] * 8, dtype=float),
        id="skew-hexa-ty",
    ),
    pytest.param(
        H8_CASES[1].values[0],
        H8_CASES[1].values[1],
        np.array([[0.0, 0.0, 1.0]] * 8, dtype=float),
        id="skew-hexa-tz",
    ),
]


@pytest.mark.parametrize(("Xe", "Ge", "u"), T4_CASES)
def test_keT4e_is_symmetric(Xe, Ge, u):
    Ke = keT4e(Xe, Ge)
    assert Ke.shape == (12, 12)
    assert np.allclose(Ke, Ke.T)


@pytest.mark.parametrize(("Xe", "Ge", "u"), T4_RIGID_CASES)
def test_qeT4e_zero_for_rigid_body_translation(Xe, Ge, u):
    qe, stress, strain = qeT4e(Xe, Ge, u)
    assert np.allclose(qe, 0.0, atol=1.0e-12)
    assert np.allclose(stress, 0.0, atol=1.0e-12)
    assert np.allclose(strain, 0.0, atol=1.0e-12)


@pytest.mark.parametrize(("Xe", "Ge", "u"), T4_CASES)
def test_kT4e_single_element_matches_element_matrix(Xe, Ge, u):
    K = np.zeros((12, 12), dtype=float)
    T = np.array([[1.0, 2.0, 3.0, 4.0, 1.0]], dtype=float)
    G = Ge.reshape(1, -1)
    assembled = kT4e(K, T, Xe, G)
    expected = keT4e(Xe, Ge)
    assert np.allclose(assembled, expected)


@pytest.mark.parametrize(("Xe", "Ge", "u"), T4_CASES)
def test_qT4e_single_element_matches_element_response(Xe, Ge, u):
    q = np.zeros((12, 1), dtype=float)
    T = np.array([[1.0, 2.0, 3.0, 4.0, 1.0]], dtype=float)
    G = Ge.reshape(1, -1)
    assembled, stress, strain = qT4e(q, T, Xe, G, u)
    expected_q, expected_stress, expected_strain = qeT4e(Xe, Ge, u)
    assert np.allclose(assembled, expected_q)
    assert np.allclose(stress[0], expected_stress)
    assert np.allclose(strain[0], expected_strain)


@pytest.mark.parametrize(("Xe", "Ge", "u"), H8_CASES)
def test_keh8e_is_symmetric(Xe, Ge, u):
    Ke = keh8e(Xe, Ge)
    assert Ke.shape == (24, 24)
    assert np.allclose(Ke, Ke.T)


@pytest.mark.parametrize(("Xe", "Ge", "u"), H8_RIGID_CASES)
def test_qeh8e_zero_for_rigid_body_translation(Xe, Ge, u):
    qe, stress, strain = qeh8e(Xe, Ge, u)
    assert np.allclose(qe, 0.0, atol=1.0e-12)
    assert np.allclose(stress, 0.0, atol=1.0e-12)
    assert np.allclose(strain, 0.0, atol=1.0e-12)


@pytest.mark.parametrize(("Xe", "Ge", "u"), H8_CASES)
def test_kh8e_single_element_matches_element_matrix(Xe, Ge, u):
    K = np.zeros((24, 24), dtype=float)
    T = np.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 1.0]], dtype=float)
    G = Ge.reshape(1, -1)
    assembled = kh8e(K, T, Xe, G)
    expected = keh8e(Xe, Ge)
    assert np.allclose(assembled, expected)


@pytest.mark.parametrize(("Xe", "Ge", "u"), H8_CASES)
def test_qh8e_single_element_matches_element_response(Xe, Ge, u):
    q = np.zeros((24, 1), dtype=float)
    T = np.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 1.0]], dtype=float)
    G = Ge.reshape(1, -1)
    assembled, stress, strain = qh8e(q, T, Xe, G, u)
    expected_q, expected_stress, expected_strain = qeh8e(Xe, Ge, u)
    assert np.allclose(assembled, expected_q)
    assert np.allclose(stress[0].reshape(8, 6), expected_stress)
    assert np.allclose(strain[0].reshape(8, 6), expected_strain)


def test_keh8e_rejects_twenty_node_hexahedra():
    Xe = np.zeros((20, 3), dtype=float)
    Ge = np.array([100.0, 0.25], dtype=float)
    with pytest.raises(NotImplementedError):
        keh8e(Xe, Ge)
