from __future__ import annotations

import numpy as np

from .._helpers import as_float_array
from .invariants import devstress, eqstress


def yieldvm(S, G, dL, Sy):
    """
    Evaluate the legacy von Mises consistency function.

    Parameters
    ----------
    S:
        Current stress vector in plane form.
    G:
        Material row ``[E, nu, Sy0, H, ...]``.
    dL:
        Plastic-multiplier increment.
    Sy:
        Current yield stress including prior hardening.

    Returns
    -------
    float
        Residual of the scalar return-mapping equation.
    """
    stress = as_float_array(S).reshape(-1)
    material = as_float_array(G).reshape(-1)
    E = material[0]
    nu = material[1]
    H = material[3]

    E1 = 2.0 * H + E / (1.0 - nu)
    E2 = 2.0 * H + 3.0 * E / (1.0 + nu)

    s1 = stress[0] + stress[1]
    s2 = stress[0] - stress[1]
    s3 = stress[2]
    xi1 = 2.0 * Sy + dL * E1
    xi2 = 2.0 * Sy + dL * E2
    return float(s1**2 / xi1**2 + 3.0 * s2**2 / xi2**2 + 12.0 * s3**2 / xi2**2 - 1.0)


def dyieldvm(S, G, dL, Sy):
    """
    Differentiate :func:`yieldvm` with respect to the plastic multiplier.

    Parameters
    ----------
    S, G, dL, Sy:
        Same quantities passed to :func:`yieldvm`.

    Returns
    -------
    float
        Derivative of the consistency residual with respect to ``dL``.
    """
    stress = as_float_array(S).reshape(-1)
    material = as_float_array(G).reshape(-1)
    E = material[0]
    nu = material[1]
    H = material[3]

    E1 = 2.0 * H + E / (1.0 - nu)
    E2 = 2.0 * H + 3.0 * E / (1.0 + nu)
    xi1 = 2.0 * Sy + E1 * dL
    xi2 = 2.0 * Sy + E2 * dL

    s1 = stress[0] + stress[1]
    s2 = stress[0] - stress[1]
    s3 = stress[2]
    df1 = -2.0 * E1 * s1**2 / xi1**3
    df2 = -2.0 * E2 * (3.0 * s2**2 + 12.0 * s3**2) / xi2**3
    return float(df1 + df2)


def stressvm(S, G, Sy):
    """
    Perform the legacy plane-stress von Mises return mapping.

    Parameters
    ----------
    S:
        Trial stress vector.
    G:
        Material row ``[E, nu, Sy0, H, ...]``.
    Sy:
        Current yield stress including prior hardening.

    Returns
    -------
    tuple[ndarray, float]
        Updated stress vector and plastic-multiplier increment.
    """
    stress = as_float_array(S, copy=True).reshape(-1)
    material = as_float_array(G).reshape(-1)
    E = material[0]
    nu = material[1]
    H = material[3]

    dL = 0.0
    f = yieldvm(stress, material, dL, Sy)
    while abs(f) > 1.0e-6:
        df = dyieldvm(stress, material, dL, Sy)
        dL -= f / df
        f = yieldvm(stress, material, dL, Sy)

    Sy = Sy + H * dL
    E1 = E / (1.0 - nu)
    E2 = 3.0 * E / (1.0 + nu)
    s1 = (stress[0] + stress[1]) / (1.0 + 0.5 * dL * E1 / Sy)
    s2 = (stress[0] - stress[1]) / (1.0 + 0.5 * dL * E2 / Sy)

    stress[0] = 0.5 * (s1 + s2)
    stress[1] = 0.5 * (s1 - s2)
    stress[2] = stress[2] / (1.0 + 0.5 * dL * E2 / Sy)
    return stress.reshape(-1, 1), float(dL)


def stressdp(S, G, Sy0, dE, dS):
    """
    Perform a Drucker-Prager stress correction with Newton iterations.

    Parameters
    ----------
    S:
        Trial stress vector.
    G:
        Material row ``[E, nu, Sy0, H, phi]``.
    Sy0:
        Current yield stress before the increment.
    dE:
        Strain increment at the integration point.
    dS:
        Elastic trial stress increment.

    Returns
    -------
    tuple[ndarray, float]
        Corrected stress vector and plastic-multiplier increment.
    """
    stress = as_float_array(S, copy=True).reshape(-1, 1)
    material = as_float_array(G).reshape(-1)
    dE = as_float_array(dE).reshape(-1, 1)
    dS = as_float_array(dS).reshape(-1, 1)

    E = material[0]
    nu = material[1]
    H = material[3]
    phi = material[4]

    C = (1.0 / E) * np.array(
        [
            [1.0, -nu, 0.0],
            [-nu, 1.0, 0.0],
            [0.0, 0.0, 2.0 * (1.0 + nu)],
        ],
        dtype=float,
    )

    Sd, Sm = devstress(stress)
    Seq = eqstress(stress)
    f = Seq + phi * Sm - Sy0

    sd = np.array([Sd[0, 0], Sd[1, 0], 2.0 * Sd[2, 0]], dtype=float).reshape(-1, 1)
    mp = np.array([1.0, 1.0, 0.0], dtype=float).reshape(-1, 1)
    df = 3.0 / (2.0 * Seq) * sd + phi / 3.0 * mp

    R = np.zeros((3, 1), dtype=float)
    deltaS = np.zeros((3, 1), dtype=float)
    dL = 0.0

    ftol = 1.0e-6
    rtol = 1.0e-3 * np.linalg.norm(dE)
    while np.linalg.norm(R) > rtol or abs(f) > ftol:
        d2f1 = 3.0 / (2.0 * Seq) * np.diag([1.0, 1.0, 2.0])
        d2f2 = 9.0 / (4.0 * Seq**3) * (sd @ sd.T)
        d2f = d2f1 - d2f2
        tangent = np.block([[C + dL * d2f, df], [df.T, np.array([[-H]], dtype=float)]])
        delta = np.linalg.solve(tangent, np.vstack([R, [[-f]]]))
        deltaS += delta[0:3]
        dL += float(delta[3, 0])

        Sd, Sm = devstress(stress + deltaS)
        Seq = eqstress(stress + deltaS)
        Sy = Sy0 + dL * H
        f = Seq + phi * Sm - Sy

        sd = np.array([Sd[0, 0], Sd[1, 0], 2.0 * Sd[2, 0]], dtype=float).reshape(-1, 1)
        df = 3.0 / (2.0 * Seq) * sd + phi / 3.0 * mp
        R = dE - C @ (dS + deltaS) - dL * df

    return stress + deltaS, float(dL)
