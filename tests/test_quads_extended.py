from __future__ import annotations

import numpy as np
import pytest

from femlabpy import keq4e, keq4p, kq4e, kq4p, qeq4e, qeq4p, qq4e, qq4p


Q4_ELASTIC_CASES = [
    pytest.param(
        np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 1.0], [0.0, 1.0]], dtype=float),
        np.array([210.0, 0.30], dtype=float),
        np.array(
            [[0.00, 0.00], [0.02, 0.00], [0.03, 0.01], [0.00, 0.02]], dtype=float
        ),
        id="rectangle-plane-stress",
    ),
    pytest.param(
        np.array([[0.0, 0.0], [1.5, 0.2], [1.8, 1.3], [0.2, 1.0]], dtype=float),
        np.array([185.0, 0.25], dtype=float),
        np.array(
            [[0.00, 0.00], [0.01, -0.01], [0.04, 0.02], [0.01, 0.03]], dtype=float
        ),
        id="skew-plane-stress",
    ),
    pytest.param(
        np.array([[1.0, 0.0], [4.0, 0.0], [4.0, 2.0], [1.0, 2.0]], dtype=float),
        np.array([170.0, 0.22, 2.0], dtype=float),
        np.array(
            [[0.00, 0.00], [0.02, 0.01], [0.05, 0.03], [0.01, 0.04]], dtype=float
        ),
        id="rectangle-plane-strain",
    ),
    pytest.param(
        np.array([[0.0, 0.0], [2.0, 0.0], [3.0, 1.0], [1.0, 1.0]], dtype=float),
        np.array([160.0, 0.20, 2.0], dtype=float),
        np.array(
            [[0.00, 0.00], [0.01, 0.00], [0.04, 0.02], [0.01, 0.02]], dtype=float
        ),
        id="parallelogram-plane-strain",
    ),
]

Q4_RIGID_CASES = [
    pytest.param(
        np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 1.0], [0.0, 1.0]], dtype=float),
        np.array([210.0, 0.30], dtype=float),
        np.array([[1.0, 0.0]] * 4, dtype=float),
        id="tx-rectangle",
    ),
    pytest.param(
        np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 1.0], [0.0, 1.0]], dtype=float),
        np.array([210.0, 0.30], dtype=float),
        np.array([[0.0, 1.0]] * 4, dtype=float),
        id="ty-rectangle",
    ),
    pytest.param(
        np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 1.0], [0.0, 1.0]], dtype=float),
        np.array([210.0, 0.30], dtype=float),
        0.05
        * np.column_stack(
            [
                -np.array([0.0, 0.0, 1.0, 1.0], dtype=float),
                np.array([0.0, 2.0, 2.0, 0.0], dtype=float),
            ]
        ),
        id="rot-rectangle",
    ),
    pytest.param(
        np.array([[0.0, 0.0], [1.5, 0.2], [1.8, 1.3], [0.2, 1.0]], dtype=float),
        np.array([185.0, 0.25], dtype=float),
        np.array([[1.0, 0.0]] * 4, dtype=float),
        id="tx-skew",
    ),
    pytest.param(
        np.array([[0.0, 0.0], [1.5, 0.2], [1.8, 1.3], [0.2, 1.0]], dtype=float),
        np.array([185.0, 0.25], dtype=float),
        np.array([[0.0, 1.0]] * 4, dtype=float),
        id="ty-skew",
    ),
    pytest.param(
        np.array([[0.0, 0.0], [1.5, 0.2], [1.8, 1.3], [0.2, 1.0]], dtype=float),
        np.array([185.0, 0.25], dtype=float),
        0.05
        * np.column_stack(
            [
                -np.array([0.0, 0.2, 1.3, 1.0], dtype=float),
                np.array([0.0, 1.5, 1.8, 0.2], dtype=float),
            ]
        ),
        id="rot-skew",
    ),
]

Q4_POTENTIAL_CASES = [
    pytest.param(
        np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 1.0], [0.0, 1.0]], dtype=float),
        np.array([3.0], dtype=float),
        np.array([[0.0], [1.0], [2.0], [1.0]], dtype=float),
        id="rectangle-diffusion",
    ),
    pytest.param(
        np.array([[0.0, 0.0], [1.5, 0.2], [1.8, 1.3], [0.2, 1.0]], dtype=float),
        np.array([2.5, 0.2], dtype=float),
        np.array([[0.0], [2.0], [1.0], [0.5]], dtype=float),
        id="skew-reaction-diffusion",
    ),
    pytest.param(
        np.array([[1.0, 0.0], [4.0, 0.0], [4.0, 2.0], [1.0, 2.0]], dtype=float),
        np.array([1.8], dtype=float),
        np.array([[1.0], [0.0], [3.0], [2.0]], dtype=float),
        id="offset-diffusion",
    ),
    pytest.param(
        np.array([[0.0, 0.0], [2.0, 0.0], [3.0, 1.0], [1.0, 1.0]], dtype=float),
        np.array([4.0, 0.1], dtype=float),
        np.array([[2.0], [1.0], [0.0], [1.0]], dtype=float),
        id="parallelogram-reaction-diffusion",
    ),
]

ZERO_Q4_POTENTIAL_CASES = [(case.values[0], case.values[1]) for case in Q4_POTENTIAL_CASES]


@pytest.mark.parametrize(("Xe", "Ge", "u"), Q4_ELASTIC_CASES)
def test_keq4e_is_symmetric(Xe, Ge, u):
    Ke = keq4e(Xe, Ge)
    assert Ke.shape == (8, 8)
    assert np.allclose(Ke, Ke.T)


@pytest.mark.parametrize(("Xe", "Ge", "u"), Q4_RIGID_CASES)
def test_qeq4e_zero_for_rigid_body_modes(Xe, Ge, u):
    qe, stress, strain = qeq4e(Xe, Ge, u)
    assert np.allclose(qe, 0.0, atol=1.0e-12)
    assert np.allclose(stress, 0.0, atol=1.0e-12)
    assert np.allclose(strain, 0.0, atol=1.0e-12)


@pytest.mark.parametrize(("Xe", "Ge", "u"), Q4_ELASTIC_CASES)
def test_kq4e_single_element_matches_element_matrix(Xe, Ge, u):
    K = np.zeros((8, 8), dtype=float)
    T = np.array([[1.0, 2.0, 3.0, 4.0, 1.0]], dtype=float)
    G = Ge.reshape(1, -1)
    assembled = kq4e(K, T, Xe, G)
    expected = keq4e(Xe, Ge)
    assert np.allclose(assembled, expected)


@pytest.mark.parametrize(("Xe", "Ge", "u"), Q4_ELASTIC_CASES)
def test_qq4e_single_element_matches_element_response(Xe, Ge, u):
    q = np.zeros((8, 1), dtype=float)
    T = np.array([[1.0, 2.0, 3.0, 4.0, 1.0]], dtype=float)
    G = Ge.reshape(1, -1)
    assembled, stress, strain = qq4e(q, T, Xe, G, u)
    expected_q, expected_stress, expected_strain = qeq4e(Xe, Ge, u)
    assert np.allclose(assembled, expected_q)
    assert np.allclose(stress[0].reshape(4, 3), expected_stress)
    assert np.allclose(strain[0].reshape(4, 3), expected_strain)


@pytest.mark.parametrize(("Xe", "Ge", "u"), Q4_POTENTIAL_CASES)
def test_keq4p_is_symmetric(Xe, Ge, u):
    Ke = keq4p(Xe, Ge)
    assert Ke.shape == (4, 4)
    assert np.allclose(Ke, Ke.T)


@pytest.mark.parametrize(("Xe", "Ge"), ZERO_Q4_POTENTIAL_CASES)
def test_qeq4p_zero_for_constant_field(Xe, Ge):
    Ue = np.full((4, 1), 7.5, dtype=float)
    qe, flux, grad = qeq4p(Xe, Ge, Ue)
    assert np.allclose(qe, 0.0, atol=1.0e-12)
    assert np.allclose(flux, 0.0, atol=1.0e-12)
    assert np.allclose(grad, 0.0, atol=1.0e-12)


@pytest.mark.parametrize(("Xe", "Ge", "u"), Q4_POTENTIAL_CASES)
def test_kq4p_single_element_matches_element_matrix(Xe, Ge, u):
    K = np.zeros((4, 4), dtype=float)
    T = np.array([[1.0, 2.0, 3.0, 4.0, 1.0]], dtype=float)
    G = Ge.reshape(1, -1)
    assembled = kq4p(K, T, Xe, G)
    expected = keq4p(Xe, Ge)
    assert np.allclose(assembled, expected)


@pytest.mark.parametrize(("Xe", "Ge", "u"), Q4_POTENTIAL_CASES)
def test_qq4p_single_element_matches_element_response(Xe, Ge, u):
    q = np.zeros((4, 1), dtype=float)
    T = np.array([[1.0, 2.0, 3.0, 4.0, 1.0]], dtype=float)
    G = Ge.reshape(1, -1)
    assembled, flux, grad = qq4p(q, T, Xe, G, u)
    expected_q, expected_flux, expected_grad = qeq4p(Xe, Ge, u)
    assert np.allclose(assembled, expected_q)
    assert np.allclose(flux[0].reshape(4, 2), expected_flux)
    assert np.allclose(grad[0].reshape(4, 2), expected_grad)
