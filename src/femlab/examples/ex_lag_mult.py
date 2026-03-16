from __future__ import annotations

import math

import numpy as np


def ex_lag_mult_data():
    areas = np.array([1.0, 1.0, 1.0], dtype=float)
    moduli = np.array([64.0, 64.0, 64.0], dtype=float)
    lengths = np.array([4.0, 4.0, 6.0], dtype=float)
    angles = np.array([math.acos(3.0 / 4.0), -math.acos(3.0 / 4.0), 0.0], dtype=float)

    coordinates = np.array(
        [
            [0.0, 0.0],
            [6.0, 0.0],
            [3.0, math.sqrt(7.0)],
        ],
        dtype=float,
    )
    topology = np.array([[1, 3, 1], [3, 2, 2], [1, 2, 3]], dtype=int)
    dof_map = np.array([[1, 2, 5, 6], [5, 6, 3, 4], [1, 2, 3, 4]], dtype=int)
    materials = np.column_stack([areas, moduli])

    constraint_matrix = np.array(
        [
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [
                0.0,
                0.0,
                -math.sin(math.radians(60.0)),
                math.cos(math.radians(60.0)),
                0.0,
                0.0,
            ],
        ],
        dtype=float,
    )
    constraint_rhs = np.zeros((constraint_matrix.shape[0], 1), dtype=float)

    load = np.zeros((6, 1), dtype=float)
    load[5, 0] = -10.0

    return {
        "A": areas,
        "E_modulus": moduli,
        "L": lengths,
        "alpha": angles,
        "X": coordinates,
        "T": topology,
        "Dvec": dof_map,
        "G": materials,
        "constraint_matrix": constraint_matrix,
        "constraint_rhs": constraint_rhs,
        "P": load,
        "dof": 2,
    }


def run_ex_lag_mult():
    data = ex_lag_mult_data()
    dof_map = data["Dvec"] - 1
    selectors = np.eye(data["P"].shape[0], dtype=float)[dof_map]
    c = np.cos(data["alpha"])
    s = np.sin(data["alpha"])
    scale = data["A"] * data["E_modulus"] / data["L"]

    transforms = np.zeros((data["T"].shape[0], 4, 4), dtype=float)
    transforms[:, 0, 0] = c
    transforms[:, 0, 1] = s
    transforms[:, 1, 0] = -s
    transforms[:, 1, 1] = c
    transforms[:, 2, 2] = c
    transforms[:, 2, 3] = s
    transforms[:, 3, 2] = -s
    transforms[:, 3, 3] = c

    local_template = np.array(
        [
            [1.0, 0.0, -1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [-1.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ],
        dtype=float,
    )
    local_stiffness = scale[:, None, None] * local_template[None, :, :]
    global_stiffness = np.einsum(
        "eji,ejk,ekl->eil", transforms, local_stiffness, transforms
    )
    stiffness = np.einsum("eai,eab,ebj->ij", selectors, global_stiffness, selectors)

    a_g = float(np.max(stiffness))
    g_bar = a_g * data["constraint_matrix"]
    q_bar = a_g * data["constraint_rhs"]
    system_matrix = np.block(
        [
            [stiffness, g_bar.T],
            [g_bar, np.zeros((g_bar.shape[0], g_bar.shape[0]), dtype=float)],
        ]
    )
    system_rhs = np.vstack([data["P"], q_bar])
    solution = np.linalg.solve(system_matrix, system_rhs)

    displacement = solution[: data["P"].shape[0], :]
    lagrange = solution[data["P"].shape[0] :, :] * a_g
    global_displacements = displacement[:, 0][dof_map]
    local_displacements = np.einsum("eab,eb->ea", transforms, global_displacements).T
    member_forces = np.einsum(
        "eab,eb->ea", local_stiffness, local_displacements.T
    ).T
    global_internal = np.einsum("eab,eb->ea", global_stiffness, global_displacements)
    q = np.einsum("eai,ea->i", selectors, global_internal).reshape(-1, 1)

    initial_nodes = data["X"][data["T"][:, :2] - 1]
    current = data["X"] + displacement.reshape(-1, data["dof"])
    current_nodes = current[data["T"][:, :2] - 1]
    initial_vectors = initial_nodes[:, 1, :] - initial_nodes[:, 0, :]
    current_vectors = current_nodes[:, 1, :] - current_nodes[:, 0, :]
    initial_lengths = np.linalg.norm(initial_vectors, axis=1)
    current_lengths = np.linalg.norm(current_vectors, axis=1)
    strain = (
        0.5 * (current_lengths**2 - initial_lengths**2) / initial_lengths**2
    ).reshape(-1, 1)
    stress = (data["E_modulus"][:, None] * strain).reshape(-1, 1)

    reactions = stiffness[np.array([0, 1, 2, 3]), :] @ displacement
    constraint_residual = (
        data["constraint_matrix"] @ displacement - data["constraint_rhs"]
    )

    return {
        "U": displacement,
        "Lag": lagrange,
        "R": reactions,
        "member_forces": member_forces,
        "local_displacements": local_displacements,
        "constraint_residual": constraint_residual,
        "K": stiffness,
        "q": q,
        "S": stress,
        "E": strain,
        "data": data,
    }


__all__ = ["ex_lag_mult_data", "run_ex_lag_mult"]
