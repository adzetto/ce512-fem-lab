"""Solver bridge — runs femlabpy solvers on the FEModel data."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .model import FEModel

log = logging.getLogger("femlabpy.gui.solver")


def solve_model(model: FEModel) -> dict:
    """Run the appropriate femlabpy solver on the model and populate results.

    Returns a dict with keys: u, q, S, R.
    """
    from .. import init, reaction, setbc, setload
    from ..elements.quads import kq4e, kq4p, qq4e, qq4p
    from ..elements.triangles import kt3e, kt3p, qt3e, qt3p

    data = model.to_dict()
    X = data["X"]
    T = data["T"]
    G = data["G"]
    C = data["C"]
    P = data["P"]
    dof = data["dof"]
    etype = model.element_type

    nn = X.shape[0]
    log.info(
        "solve_model: etype=%s dof=%d nn=%d ne=%d "
        "T.shape=%s X.shape=%s G.shape=%s C.shape=%s P.shape=%s",
        etype,
        dof,
        nn,
        T.shape[0],
        T.shape,
        X.shape,
        G.shape,
        C.shape,
        P.shape,
    )

    K, p, q = init(nn, dof)
    log.info("init done: K.shape=%s  p.shape=%s", K.shape, p.shape)

    # ---- Assembly (whole-model functions) ----
    if etype == "Q4":
        K = kq4e(K, T, X, G) if dof == 2 else kq4p(K, T, X, G)
    elif etype == "T3":
        K = kt3e(K, T, X, G) if dof == 2 else kt3p(K, T, X, G)
    else:
        raise ValueError(f"Unsupported element type: {etype}")
    log.info("assembly done")

    # ---- Loads ----
    if P.shape[0] > 0:
        p = setload(p, P)
        log.info("loads applied: %d load rows", P.shape[0])

    # ---- Boundary conditions ----
    K, p, scale = setbc(K, p, C, dof)
    log.info("BCs applied: %d constraints, scale=%.4g", C.shape[0], scale)

    # ---- Solve ----
    u = np.linalg.solve(K, p)
    log.info("solve done: max|u|=%.6g", np.max(np.abs(u)))

    # ---- Post-process stresses / gradients ----
    S = None
    if etype == "Q4":
        if dof == 2:
            q, S, _E = qq4e(q, T, X, G, u)
        else:
            q, S, _E = qq4p(q, T, X, G, u)
    elif etype == "T3":
        if dof == 2:
            q, S, _E = qt3e(q, T, X, G, u)
        else:
            q, S, _E = qt3p(q, T, X, G, u)

    if S is not None:
        log.info("stress recovery done: S.shape=%s", S.shape)

    # ---- Reactions ----
    R = None
    if C.shape[0] > 0 and dof > 1:
        R = reaction(q, C, dof)
        log.info("reactions: R.shape=%s", R.shape)

    model.u = u
    model.stresses = S
    model.reactions = R
    model.solved = True

    return {"u": u, "q": q, "S": S, "R": R}
