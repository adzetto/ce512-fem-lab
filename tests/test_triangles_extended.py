from __future__ import annotations

import numpy as np
import pytest

from femlabpy import ket3e, ket3p, kt3e, kt3p, qet3e, qet3p, qt3e, qt3p

TRI_ELASTIC_CASES = [
    pytest.param(
        np.array([[0.0, 0.0], [2.0, 0.0], [0.0, 1.0]], dtype=float),
        np.array([210.0, 0.30], dtype=float),
        np.array([[0.00, 0.00], [0.01, 0.00], [0.00, 0.02]], dtype=float),
        id="right-plane-stress",
    ),
    pytest.param(
        np.array([[0.0, 0.0], [1.5, 0.2], [0.3, 1.4]], dtype=float),
        np.array([180.0, 0.28], dtype=float),
        np.array([[0.00, 0.00], [0.02, -0.01], [0.01, 0.03]], dtype=float),
        id="skew-plane-stress",
    ),
    pytest.param(
        np.array([[1.0, 0.0], [3.0, 0.5], [1.4, 2.0]], dtype=float),
        np.array([160.0, 0.22], dtype=float),
        np.array([[0.00, 0.00], [0.03, 0.01], [0.02, 0.04]], dtype=float),
        id="offset-plane-stress",
    ),
    pytest.param(
        np.array([[0.0, 0.0], [2.0, 0.0], [0.0, 1.0]], dtype=float),
        np.array([210.0, 0.30, 2.0], dtype=float),
        np.array([[0.00, 0.00], [0.01, 0.00], [0.00, 0.02]], dtype=float),
        id="right-plane-strain",
    ),
    pytest.param(
        np.array([[0.0, 0.0], [1.5, 0.2], [0.3, 1.4]], dtype=float),
        np.array([185.0, 0.25, 2.0], dtype=float),
        np.array([[0.00, 0.00], [0.02, -0.02], [0.01, 0.03]], dtype=float),
        id="skew-plane-strain",
    ),
    pytest.param(
        np.array([[1.0, 0.0], [3.0, 0.5], [1.4, 2.0]], dtype=float),
        np.array([175.0, 0.20, 2.0], dtype=float),
        np.array([[0.00, 0.00], [0.02, 0.01], [0.03, 0.05]], dtype=float),
        id="offset-plane-strain",
    ),
]

TRI_RIGID_CASES = [
    pytest.param(
        np.array([[0.0, 0.0], [2.0, 0.0], [0.0, 1.0]], dtype=float),
        np.array([210.0, 0.30], dtype=float),
        np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]], dtype=float),
        id="tx-right",
    ),
    pytest.param(
        np.array([[0.0, 0.0], [2.0, 0.0], [0.0, 1.0]], dtype=float),
        np.array([210.0, 0.30], dtype=float),
        np.array([[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]], dtype=float),
        id="ty-right",
    ),
    pytest.param(
        np.array([[0.0, 0.0], [2.0, 0.0], [0.0, 1.0]], dtype=float),
        np.array([210.0, 0.30], dtype=float),
        0.05
        * np.column_stack(
            [
                -np.array([0.0, 0.0, 1.0], dtype=float),
                np.array([0.0, 2.0, 0.0], dtype=float),
            ]
        ),
        id="rot-right",
    ),
    pytest.param(
        np.array([[0.0, 0.0], [1.5, 0.2], [0.3, 1.4]], dtype=float),
        np.array([185.0, 0.25, 2.0], dtype=float),
        np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]], dtype=float),
        id="tx-skew",
    ),
    pytest.param(
        np.array([[0.0, 0.0], [1.5, 0.2], [0.3, 1.4]], dtype=float),
        np.array([185.0, 0.25, 2.0], dtype=float),
        np.array([[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]], dtype=float),
        id="ty-skew",
    ),
    pytest.param(
        np.array([[0.0, 0.0], [1.5, 0.2], [0.3, 1.4]], dtype=float),
        np.array([185.0, 0.25, 2.0], dtype=float),
        0.05
        * np.column_stack(
            [
                -np.array([0.0, 0.2, 1.4], dtype=float),
                np.array([0.0, 1.5, 0.3], dtype=float),
            ]
        ),
        id="rot-skew",
    ),
]

TRI_POTENTIAL_CASES = [
    pytest.param(
        np.array([[0.0, 0.0], [2.0, 0.0], [0.0, 1.0]], dtype=float),
        np.array([3.0], dtype=float),
        np.array([[0.0], [1.0], [2.0]], dtype=float),
        id="right-diffusion",
    ),
    pytest.param(
        np.array([[0.0, 0.0], [1.5, 0.2], [0.3, 1.4]], dtype=float),
        np.array([2.5, 0.2], dtype=float),
        np.array([[0.0], [2.0], [1.0]], dtype=float),
        id="skew-reaction-diffusion",
    ),
    pytest.param(
        np.array([[1.0, 0.0], [3.0, 0.5], [1.4, 2.0]], dtype=float),
        np.array([1.8], dtype=float),
        np.array([[1.0], [0.0], [3.0]], dtype=float),
        id="offset-diffusion",
    ),
    pytest.param(
        np.array([[0.0, 0.0], [2.2, 0.1], [0.4, 1.7]], dtype=float),
        np.array([4.0, 0.1], dtype=float),
        np.array([[2.0], [1.0], [0.0]], dtype=float),
        id="skew2-reaction-diffusion",
    ),
    pytest.param(
        np.array([[0.5, 0.2], [2.5, 0.2], [0.8, 1.5]], dtype=float),
        np.array([2.2], dtype=float),
        np.array([[0.5], [1.5], [2.5]], dtype=float),
        id="offset2-diffusion",
    ),
    pytest.param(
        np.array([[0.0, 0.0], [1.2, 0.4], [0.2, 1.1]], dtype=float),
        np.array([3.5, 0.3], dtype=float),
        np.array([[1.0], [1.5], [0.5]], dtype=float),
        id="small-reaction-diffusion",
    ),
]

ZERO_TRI_POTENTIAL_CASES = [
    (case.values[0], case.values[1]) for case in TRI_POTENTIAL_CASES
]


@pytest.mark.parametrize(("Xe", "Ge", "u"), TRI_ELASTIC_CASES)
def test_ket3e_is_symmetric(Xe, Ge, u):
    Ke = ket3e(Xe, Ge)
    assert Ke.shape == (6, 6)
    assert np.allclose(Ke, Ke.T)


@pytest.mark.parametrize(("Xe", "Ge", "u"), TRI_RIGID_CASES)
def test_qet3e_zero_for_rigid_body_modes(Xe, Ge, u):
    qe, stress, strain = qet3e(Xe, Ge, u)
    assert np.allclose(qe, 0.0, atol=1.0e-12)
    assert np.allclose(stress, 0.0, atol=1.0e-12)
    assert np.allclose(strain, 0.0, atol=1.0e-12)


@pytest.mark.parametrize(("Xe", "Ge", "u"), TRI_ELASTIC_CASES)
def test_kt3e_single_element_matches_element_matrix(Xe, Ge, u):
    K = np.zeros((6, 6), dtype=float)
    T = np.array([[1.0, 2.0, 3.0, 1.0]], dtype=float)
    G = Ge.reshape(1, -1)
    assembled = kt3e(K, T, Xe, G)
    expected = ket3e(Xe, Ge)
    assert np.allclose(assembled, expected)


@pytest.mark.parametrize(("Xe", "Ge", "u"), TRI_ELASTIC_CASES)
def test_qt3e_single_element_matches_element_response(Xe, Ge, u):
    q = np.zeros((6, 1), dtype=float)
    T = np.array([[1.0, 2.0, 3.0, 1.0]], dtype=float)
    G = Ge.reshape(1, -1)
    assembled, stress, strain = qt3e(q, T, Xe, G, u)
    expected_q, expected_stress, expected_strain = qet3e(Xe, Ge, u)
    assert np.allclose(assembled, expected_q)
    assert np.allclose(stress[0], expected_stress)
    assert np.allclose(strain[0], expected_strain)


@pytest.mark.parametrize(("Xe", "Ge", "u"), TRI_POTENTIAL_CASES)
def test_ket3p_is_symmetric(Xe, Ge, u):
    Ke = ket3p(Xe, Ge)
    assert Ke.shape == (3, 3)
    assert np.allclose(Ke, Ke.T)


@pytest.mark.parametrize(("Xe", "Ge"), ZERO_TRI_POTENTIAL_CASES)
def test_qet3p_zero_for_constant_field(Xe, Ge):
    Ue = np.full((3, 1), 5.0, dtype=float)
    qe, flux, grad = qet3p(Xe, Ge, Ue)
    assert np.allclose(qe, 0.0, atol=1.0e-12)
    assert np.allclose(flux, 0.0, atol=1.0e-12)
    assert np.allclose(grad, 0.0, atol=1.0e-12)


@pytest.mark.parametrize(("Xe", "Ge", "u"), TRI_POTENTIAL_CASES)
def test_kt3p_single_element_matches_element_matrix(Xe, Ge, u):
    K = np.zeros((3, 3), dtype=float)
    T = np.array([[1.0, 2.0, 3.0, 1.0]], dtype=float)
    G = Ge.reshape(1, -1)
    assembled = kt3p(K, T, Xe, G)
    expected = ket3p(Xe, Ge)
    assert np.allclose(assembled, expected)


@pytest.mark.parametrize(("Xe", "Ge", "u"), TRI_POTENTIAL_CASES)
def test_qt3p_single_element_matches_element_response(Xe, Ge, u):
    q = np.zeros((3, 1), dtype=float)
    T = np.array([[1.0, 2.0, 3.0, 1.0]], dtype=float)
    G = Ge.reshape(1, -1)
    assembled, flux, grad = qt3p(q, T, Xe, G, u)
    expected_q, expected_flux, expected_grad = qet3p(Xe, Ge, u)
    assert np.allclose(assembled, expected_q)
    assert np.allclose(flux[0], expected_flux)
    assert np.allclose(grad[0], expected_grad)
