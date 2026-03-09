from __future__ import annotations

import numpy as np

from .. import init, kq4p, kt3p, plotq4, plotelem, plott3, plotu, qq4p, qt3p, setbc


def flow_data():
    X = np.array(
        [
            [0.0, 0.0],
            [2.0, 0.0],
            [5.5, 0.0],
            [12.0, 0.0],
            [0.0, 1.25],
            [1.75, 1.25],
            [4.0, 2.0],
            [7.5, 3.0],
            [12.0, 3.0],
            [0.0, 2.5],
            [1.5, 2.5],
            [2.75, 3.25],
            [5.25, 3.75],
            [6.0, 6.0],
            [8.25, 6.0],
            [12.0, 6.0],
            [3.5, 4.5],
            [4.0, 6.0],
            [0.0, 3.25],
            [1.0, 3.5],
            [1.75, 4.25],
            [2.25, 5.0],
            [2.5, 6.0],
            [0.0, 4.0],
            [0.5, 4.0],
            [0.75, 4.5],
            [1.0, 5.25],
            [1.125, 6.0],
            [0.0, 4.5],
            [0.0, 5.25],
            [0.0, 6.0],
        ],
        dtype=float,
    )
    T1 = np.array(
        [
            [1, 2, 6, 5, 1],
            [5, 6, 11, 10, 1],
            [10, 11, 20, 19, 1],
            [19, 20, 25, 24, 1],
            [24, 25, 26, 29, 1],
            [29, 26, 27, 30, 1],
            [30, 27, 28, 31, 1],
            [2, 3, 7, 6, 1],
            [6, 7, 12, 11, 1],
            [11, 12, 21, 20, 1],
            [20, 21, 26, 25, 1],
            [21, 22, 27, 26, 1],
            [22, 23, 28, 27, 1],
            [3, 4, 9, 8, 1],
            [3, 8, 13, 7, 1],
            [7, 13, 17, 12, 1],
            [12, 17, 22, 21, 1],
            [17, 18, 23, 22, 1],
            [17, 13, 14, 18, 1],
            [13, 8, 15, 14, 1],
            [8, 9, 16, 15, 1],
        ],
        dtype=int,
    )
    T2 = np.array(
        [
            [1, 2, 5, 1],
            [2, 6, 5, 1],
            [5, 6, 10, 1],
            [6, 11, 10, 1],
            [10, 11, 19, 1],
            [11, 20, 19, 1],
            [19, 20, 25, 1],
            [19, 25, 24, 1],
            [24, 25, 29, 1],
            [25, 26, 29, 1],
            [29, 26, 30, 1],
            [26, 27, 30, 1],
            [30, 27, 31, 1],
            [27, 28, 31, 1],
            [2, 3, 6, 1],
            [3, 7, 6, 1],
            [6, 7, 11, 1],
            [7, 12, 11, 1],
            [11, 12, 20, 1],
            [20, 12, 21, 1],
            [20, 21, 25, 1],
            [25, 21, 26, 1],
            [26, 21, 27, 1],
            [21, 22, 27, 1],
            [27, 22, 28, 1],
            [22, 23, 28, 1],
            [12, 17, 21, 1],
            [21, 17, 22, 1],
            [22, 17, 23, 1],
            [17, 18, 23, 1],
            [4, 9, 8, 1],
            [3, 4, 8, 1],
            [3, 8, 7, 1],
            [7, 8, 13, 1],
            [7, 13, 12, 1],
            [12, 13, 17, 1],
            [17, 13, 14, 1],
            [17, 14, 18, 1],
            [13, 8, 14, 1],
            [8, 15, 14, 1],
            [8, 9, 15, 1],
            [9, 16, 15, 1],
        ],
        dtype=int,
    )
    G = np.array([[1.0e-4]], dtype=float)
    C = np.array([[1, 20], [5, 20], [10, 20], [19, 20], [24, 20], [14, 40], [15, 40], [16, 40], [18, 40], [23, 40], [28, 40], [31, 40]], dtype=float)
    return {"X": X, "T1": T1, "T2": T2, "G": G, "C": C, "dof": 1}


def _solve_potential(T, X, G, C, assembler, postprocessor):
    K, p, q = init(X.shape[0], 1, use_sparse=False)
    K = assembler(K, T, X, G)
    K, p, _ = setbc(K, p, C, 1)
    u = np.linalg.solve(K, p)
    q, S, E = postprocessor(q, T, X, G, u)
    return {"u": u, "q": q, "S": S, "E": E}


def run_flow_q4(*, plot: bool = False):
    data = flow_data()
    result = _solve_potential(data["T1"], data["X"], data["G"], data["C"], kq4p, qq4p)
    result["data"] = data
    result["figures"] = []
    if plot:
        from matplotlib import pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        plotelem(data["T1"], data["X"], ax=axes[0, 0])
        plotu(data["T1"], data["X"], result["u"], ax=axes[0, 1])
        plotq4(data["T1"], data["X"], result["E"], 1, ax=axes[1, 0])
        plotq4(data["T1"], data["X"], result["E"], 2, ax=axes[1, 1])
        result["figures"].append(fig)
    return result


def run_flow_t3(*, plot: bool = False):
    data = flow_data()
    result = _solve_potential(data["T2"], data["X"], data["G"], data["C"], kt3p, qt3p)
    result["data"] = data
    result["figures"] = []
    if plot:
        from matplotlib import pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        plotelem(data["T2"], data["X"], ax=axes[0, 0])
        plotu(data["T2"], data["X"], result["u"], ax=axes[0, 1])
        plott3(data["T2"], data["X"], result["E"], 1, ax=axes[1, 0])
        plott3(data["T2"], data["X"], result["E"], 2, ax=axes[1, 1])
        result["figures"].append(fig)
    return result
