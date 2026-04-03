from __future__ import annotations

import numpy as np
import pytest

from femlabpy import kbar, kebar, qbar, qebar


def _dofs(nodes: np.ndarray, dim: int) -> np.ndarray:
    return (nodes[:, None] * dim + np.arange(dim)).reshape(-1)


BAR_CASES = [
    pytest.param(
        np.array([[0.0, 0.0], [2.0, 0.0]], dtype=float),
        np.array([2.0, 100.0], dtype=float),
        np.array([[0.0, 0.0], [0.10, 0.0]], dtype=float),
        id="2d-axial-extension",
    ),
    pytest.param(
        np.array([[0.0, 0.0], [3.0, 4.0]], dtype=float),
        np.array([1.5, 210.0], dtype=float),
        np.array([[0.0, 0.0], [0.06, 0.08]], dtype=float),
        id="2d-diagonal-extension",
    ),
    pytest.param(
        np.array([[1.0, 1.0], [4.0, 1.0]], dtype=float),
        np.array([3.0, 150.0], dtype=float),
        np.array([[0.0, 0.0], [-0.05, 0.0]], dtype=float),
        id="2d-axial-compression",
    ),
    pytest.param(
        np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 2.0]], dtype=float),
        np.array([0.8, 70.0], dtype=float),
        np.array([[0.0, 0.0, 0.0], [0.03, 0.06, 0.06]], dtype=float),
        id="3d-diagonal-extension",
    ),
    pytest.param(
        np.array([[0.0, 1.0, 0.0], [2.0, 2.0, 1.0]], dtype=float),
        np.array([1.2, 95.0], dtype=float),
        np.array([[0.0, 0.0, 0.0], [0.02, 0.01, 0.01]], dtype=float),
        id="3d-skew-extension",
    ),
]

RIGID_TRANSLATION_CASES = [
    pytest.param(
        np.array([[0.0, 0.0], [2.0, 0.0]], dtype=float),
        np.array([2.0, 100.0], dtype=float),
        np.array([0.5, -0.2], dtype=float),
        id="2d-horizontal",
    ),
    pytest.param(
        np.array([[0.0, 0.0], [3.0, 4.0]], dtype=float),
        np.array([1.5, 210.0], dtype=float),
        np.array([-0.3, 0.4], dtype=float),
        id="2d-diagonal",
    ),
    pytest.param(
        np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 2.0]], dtype=float),
        np.array([0.8, 70.0], dtype=float),
        np.array([0.1, -0.2, 0.3], dtype=float),
        id="3d-diagonal",
    ),
    pytest.param(
        np.array([[0.0, 1.0, 0.0], [2.0, 2.0, 1.0]], dtype=float),
        np.array([1.2, 95.0], dtype=float),
        np.array([-0.4, 0.25, 0.15], dtype=float),
        id="3d-skew",
    ),
]

MULTI_BAR_CASES = [
    pytest.param(
        np.array([[0.0, 0.0], [2.0, 0.0], [3.0, 1.0]], dtype=float),
        np.array([[2.0, 100.0], [1.0, 120.0]], dtype=float),
        np.array([[1.0, 2.0, 1.0], [2.0, 3.0, 2.0]], dtype=float),
        np.array([[0.0, 0.0], [0.03, 0.01], [0.05, 0.04]], dtype=float),
        id="2d-two-bars",
    ),
    pytest.param(
        np.array([[0.0, 0.0], [1.5, 0.5], [3.0, 0.5]], dtype=float),
        np.array([[1.4, 90.0], [0.9, 140.0]], dtype=float),
        np.array([[1.0, 2.0, 1.0], [2.0, 3.0, 2.0]], dtype=float),
        np.array([[0.0, 0.0], [0.02, -0.01], [0.07, 0.00]], dtype=float),
        id="2d-chain",
    ),
    pytest.param(
        np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 0.0], [2.0, 2.0, 1.0]], dtype=float),
        np.array([[0.8, 75.0], [1.1, 110.0]], dtype=float),
        np.array([[1.0, 2.0, 1.0], [2.0, 3.0, 2.0]], dtype=float),
        np.array(
            [[0.0, 0.0, 0.0], [0.02, 0.04, 0.00], [0.05, 0.03, 0.02]], dtype=float
        ),
        id="3d-two-bars",
    ),
    pytest.param(
        np.array([[0.0, 1.0, 0.0], [2.0, 1.5, 0.5], [3.5, 2.0, 1.0]], dtype=float),
        np.array([[1.3, 80.0], [0.7, 130.0]], dtype=float),
        np.array([[1.0, 2.0, 1.0], [2.0, 3.0, 2.0]], dtype=float),
        np.array(
            [[0.0, 0.0, 0.0], [0.03, 0.01, 0.01], [0.06, 0.03, 0.02]], dtype=float
        ),
        id="3d-chain",
    ),
]

ZERO_BAR_CASES = [(case.values[0], case.values[1]) for case in RIGID_TRANSLATION_CASES]


@pytest.mark.parametrize(("Xe", "Ge", "u"), BAR_CASES)
def test_kebar_is_symmetric(Xe, Ge, u):
    Ke = kebar(Xe, Xe + u, Ge)
    assert Ke.shape == (Xe.size, Xe.size)
    assert np.allclose(Ke, Ke.T)


@pytest.mark.parametrize(("Xe", "Ge", "shift"), RIGID_TRANSLATION_CASES)
def test_qebar_zero_for_rigid_translation(Xe, Ge, shift):
    qe, stress, strain = qebar(Xe, Xe + shift, Ge)
    assert np.allclose(qe, 0.0, atol=1.0e-12)
    assert np.isclose(stress, 0.0, atol=1.0e-12)
    assert np.isclose(strain, 0.0, atol=1.0e-12)


@pytest.mark.parametrize(("Xe", "Ge", "u"), BAR_CASES)
def test_qebar_reports_stress_and_strain_consistently(Xe, Ge, u):
    qe, stress, strain = qebar(Xe, Xe + u, Ge)
    assert np.isclose(stress, Ge[1] * strain)
    assert np.allclose(qe[: Xe.shape[1]], -qe[Xe.shape[1] :], atol=1.0e-12)


@pytest.mark.parametrize(("Xe", "Ge", "u"), BAR_CASES)
def test_kbar_single_element_matches_local_element(Xe, Ge, u):
    topology = np.array([[1.0, 2.0, 1.0]], dtype=float)
    materials = Ge.reshape(1, -1)
    K = np.zeros((Xe.size, Xe.size), dtype=float)
    assembled = kbar(K, topology, Xe, materials, u.reshape(-1, 1))
    expected = kebar(Xe, Xe + u, Ge)
    assert np.allclose(assembled, expected)


@pytest.mark.parametrize(("Xe", "Ge", "u"), BAR_CASES)
def test_qbar_single_element_matches_local_response(Xe, Ge, u):
    topology = np.array([[1.0, 2.0, 1.0]], dtype=float)
    materials = Ge.reshape(1, -1)
    q = np.zeros((Xe.size, 1), dtype=float)
    assembled, stress, strain = qbar(q, topology, Xe, materials, u.reshape(-1, 1))
    expected_q, expected_stress, expected_strain = qebar(Xe, Xe + u, Ge)
    assert np.allclose(assembled, expected_q)
    assert np.allclose(stress.ravel(), [expected_stress])
    assert np.allclose(strain.ravel(), [expected_strain])


@pytest.mark.parametrize(("Xe", "Ge"), ZERO_BAR_CASES)
def test_qbar_zero_without_displacement(Xe, Ge):
    topology = np.array([[1.0, 2.0, 1.0]], dtype=float)
    materials = Ge.reshape(1, -1)
    q = np.zeros((Xe.size, 1), dtype=float)
    assembled, stress, strain = qbar(q, topology, Xe, materials)
    assert np.allclose(assembled, 0.0, atol=1.0e-12)
    assert np.allclose(stress, 0.0, atol=1.0e-12)
    assert np.allclose(strain, 0.0, atol=1.0e-12)


@pytest.mark.parametrize(("X", "G", "T", "u"), MULTI_BAR_CASES)
def test_kbar_multi_element_matches_manual_assembly(X, G, T, u):
    dim = X.shape[1]
    K = np.zeros((X.shape[0] * dim, X.shape[0] * dim), dtype=float)
    assembled = kbar(K.copy(), T, X, G, u.reshape(-1, 1))

    expected = np.zeros_like(assembled)
    current = X + u
    for row in T.astype(int):
        nodes = row[:2] - 1
        ke = kebar(X[nodes], current[nodes], G[row[-1] - 1])
        dofs = _dofs(nodes, dim)
        expected[np.ix_(dofs, dofs)] += ke

    assert np.allclose(assembled, expected)


@pytest.mark.parametrize(("X", "G", "T", "u"), MULTI_BAR_CASES)
def test_qbar_multi_element_matches_manual_assembly(X, G, T, u):
    dim = X.shape[1]
    q = np.zeros((X.shape[0] * dim, 1), dtype=float)
    assembled, stress, strain = qbar(q.copy(), T, X, G, u.reshape(-1, 1))

    expected = np.zeros_like(assembled)
    expected_stress = np.zeros((T.shape[0], 1), dtype=float)
    expected_strain = np.zeros((T.shape[0], 1), dtype=float)
    current = X + u
    for i, row in enumerate(T.astype(int)):
        nodes = row[:2] - 1
        qe, se, ee = qebar(X[nodes], current[nodes], G[row[-1] - 1])
        dofs = _dofs(nodes, dim)
        expected[dofs, 0] += qe[:, 0]
        expected_stress[i, 0] = se
        expected_strain[i, 0] = ee

    assert np.allclose(assembled, expected)
    assert np.allclose(stress, expected_stress)
    assert np.allclose(strain, expected_strain)
