from __future__ import annotations

import numpy as np

from .. import init, kq4e, qq4e, reaction, setbc, setload
from ..plotting import plotbc, plotelem, plotforces, plotq4


def cantilever_data():
    T = np.array(
        [
            [1, 4, 5, 2, 1],
            [2, 5, 6, 3, 1],
            [4, 7, 8, 5, 1],
            [5, 8, 9, 6, 1],
            [7, 10, 11, 8, 1],
            [8, 11, 12, 9, 1],
            [10, 13, 14, 11, 1],
            [11, 14, 15, 12, 1],
            [13, 16, 17, 14, 1],
            [14, 17, 18, 15, 1],
            [16, 19, 20, 17, 1],
            [17, 20, 21, 18, 1],
            [19, 22, 23, 20, 1],
            [20, 23, 24, 21, 1],
            [22, 25, 26, 23, 1],
            [23, 26, 27, 24, 1],
        ],
        dtype=int,
    )
    X = np.array(
        [
            [0.0, 0.0],
            [0.0, 0.5],
            [0.0, 1.0],
            [0.5, 0.0],
            [0.5, 0.5],
            [0.5, 1.0],
            [1.0, 0.0],
            [1.0, 0.5],
            [1.0, 1.0],
            [1.5, 0.0],
            [1.5, 0.5],
            [1.5, 1.0],
            [2.0, 0.0],
            [2.0, 0.5],
            [2.0, 1.0],
            [2.5, 0.0],
            [2.5, 0.5],
            [2.5, 1.0],
            [3.0, 0.0],
            [3.0, 0.5],
            [3.0, 1.0],
            [3.5, 0.0],
            [3.5, 0.5],
            [3.5, 1.0],
            [4.0, 0.0],
            [4.0, 0.5],
            [4.0, 1.0],
        ],
        dtype=float,
    )
    G = np.array([[100.0, 0.3, 1.0]], dtype=float)
    C = np.array([[1, 1, 0.0], [2, 1, 0.0], [2, 2, 0.0], [3, 1, 0.0]], dtype=float)
    P = np.array([[25, 0.0, -0.05], [26, 0.0, -0.10], [27, 0.0, -0.05]], dtype=float)
    return {"T": T, "X": X, "G": G, "C": C, "P": P, "dof": 2}


def run_cantilever(*, plot: bool = False):
    data = cantilever_data()
    K, p, q = init(data["X"].shape[0], data["dof"], use_sparse=False)
    K = kq4e(K, data["T"], data["X"], data["G"])
    p = setload(p, data["P"])
    K, p, _ = setbc(K, p, data["C"], data["dof"])
    u = np.linalg.solve(K, p)
    q, S, E = qq4e(q, data["T"], data["X"], data["G"], u)
    R = reaction(q, data["C"], data["dof"])

    figures = []
    if plot:
        from matplotlib import pyplot as plt

        fig1, ax1 = plt.subplots()
        plotelem(data["T"], data["X"], ax=ax1)
        plotforces(data["T"], data["X"], data["P"], ax=ax1)
        plotbc(data["T"], data["X"], data["C"], ax=ax1)
        U = u.reshape(data["X"].shape)
        plotelem(data["T"], data["X"] + U, line_style="c--", ax=ax1)
        figures.append(fig1)

        fig2, ax2 = plt.subplots()
        plotq4(data["T"], data["X"], S, 1, ax=ax2)
        figures.append(fig2)

    return {"u": u, "q": q, "S": S, "E": E, "R": R, "data": data, "figures": figures}
