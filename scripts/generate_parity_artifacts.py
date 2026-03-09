from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
TMP = REPO / "tmp" / "parity"
SCILAB_INPUTS = TMP / "scilab_inputs"
SCILAB_OUTPUTS = TMP / "scilab_outputs"
ASSETS = REPO / "docs" / "assets" / "comparison"

sys.path.insert(0, str(REPO / "src"))

from femlab.examples import run_cantilever, run_gmsh_triangle  # noqa: E402


def find_scilab() -> Path:
    candidates = sorted(Path("C:/Program Files").glob("scilab-*/bin/WScilex-cli.exe"), reverse=True)
    if not candidates:
        raise FileNotFoundError("Scilab CLI was not found under C:/Program Files.")
    return candidates[0]


def ensure_dirs() -> None:
    SCILAB_INPUTS.mkdir(parents=True, exist_ok=True)
    SCILAB_OUTPUTS.mkdir(parents=True, exist_ok=True)
    ASSETS.mkdir(parents=True, exist_ok=True)


def save_csv(path: Path, array: np.ndarray, *, integer: bool = False) -> None:
    fmt = "%.0f" if integer else "%.18e"
    np.savetxt(path, np.asarray(array), delimiter=",", fmt=fmt)


def prepare_triangle_inputs() -> None:
    data = run_gmsh_triangle(plot=False)["data"]
    save_csv(SCILAB_INPUTS / "triangle_T.csv", data["T"], integer=True)
    save_csv(SCILAB_INPUTS / "triangle_X.csv", data["X"])
    save_csv(SCILAB_INPUTS / "triangle_G.csv", data["G"])
    save_csv(SCILAB_INPUTS / "triangle_C.csv", data["C"])
    save_csv(SCILAB_INPUTS / "triangle_P.csv", data["P"])


def run_scilab(script_name: str) -> None:
    scilab = find_scilab()
    script = REPO / "scripts" / "scilab" / script_name
    subprocess.run([str(scilab), "-nb", "-f", str(script)], cwd=REPO, check=True, timeout=180)


def load_csv(name: str) -> np.ndarray:
    return np.loadtxt(SCILAB_OUTPUTS / name, delimiter=",")


def quad_triangulation(T: np.ndarray) -> np.ndarray:
    tris = []
    for row in T.astype(int):
        n1, n2, n3, n4 = row[:4] - 1
        tris.append([n1, n2, n3])
        tris.append([n1, n3, n4])
    return np.asarray(tris, dtype=int)


def plot_mesh(ax, X: np.ndarray, T: np.ndarray, *, color: str, linewidth: float = 1.0, linestyle: str = "-") -> None:
    X = np.asarray(X)
    T = np.asarray(T).astype(int)
    for row in T:
        nodes = row[:-1] - 1
        order = list(nodes)
        if len(order) >= 3:
            order.append(order[0])
        pts = X[order]
        ax.plot(pts[:, 0], pts[:, 1], color=color, linewidth=linewidth, linestyle=linestyle)


def displacement_overlay(ax, X: np.ndarray, T: np.ndarray, u_a: np.ndarray, u_b: np.ndarray, *, scale: float, title: str) -> None:
    plot_mesh(ax, X, T, color="0.75", linewidth=0.8)
    plot_mesh(ax, X + scale * u_a.reshape(X.shape), T, color="#1f77b4", linewidth=1.0)
    plot_mesh(ax, X + scale * u_b.reshape(X.shape), T, color="#ff7f0e", linewidth=1.0, linestyle="--")
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")


def tripcolor_diff(ax, X: np.ndarray, triangles: np.ndarray, values: np.ndarray, title: str) -> None:
    trip = ax.tripcolor(X[:, 0], X[:, 1], triangles, values.reshape(-1), shading="gouraud", cmap="magma")
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    plt.colorbar(trip, ax=ax, fraction=0.046, pad=0.04)


def stress_scatter(ax, s_a: np.ndarray, s_b: np.ndarray, title: str) -> None:
    a = s_a.reshape(-1)
    b = s_b.reshape(-1)
    lo = min(a.min(), b.min())
    hi = max(a.max(), b.max())
    ax.scatter(a, b, s=14, alpha=0.75)
    ax.plot([lo, hi], [lo, hi], color="black", linewidth=1.0)
    ax.set_title(title)
    ax.set_xlabel("Scilab")
    ax.set_ylabel("Python")


def make_figure(example_name: str, X: np.ndarray, T: np.ndarray, u_s: np.ndarray, u_p: np.ndarray, s_s: np.ndarray, s_p: np.ndarray, *, scale: float, is_quad: bool) -> dict[str, float]:
    if is_quad:
        triangles = quad_triangulation(T)
    else:
        triangles = T[:, :3].astype(int) - 1

    diff_u = np.linalg.norm(u_s.reshape(X.shape) - u_p.reshape(X.shape), axis=1)
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    displacement_overlay(axes[0, 0], X, T, u_s, u_p, scale=scale, title=f"{example_name}: displaced mesh overlay")
    axes[0, 0].legend(["Undeformed", "Scilab", "Python"], loc="best")
    displacement_overlay(axes[0, 1], X, T, u_s, u_p, scale=0.0, title=f"{example_name}: undeformed mesh")
    tripcolor_diff(axes[1, 0], X, triangles, diff_u, title=f"{example_name}: |u_scilab - u_python|")
    stress_scatter(axes[1, 1], s_s, s_p, title=f"{example_name}: stress component parity")
    fig.tight_layout()
    outfile = ASSETS / f"{example_name.lower().replace(' ', '_')}_parity.png"
    fig.savefig(outfile, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return {
        "u_max_abs_diff": float(np.max(np.abs(u_s - u_p))),
        "u_l2_diff": float(np.linalg.norm(u_s - u_p)),
        "stress_max_abs_diff": float(np.max(np.abs(s_s - s_p))),
    }


def main() -> None:
    ensure_dirs()
    prepare_triangle_inputs()
    run_scilab("run_cantilever.sce")
    run_scilab("run_triangle_from_csv.sce")

    py_canti = run_cantilever(plot=False)
    py_tri = run_gmsh_triangle(plot=False)

    sc_canti = {
        "u": load_csv("cantilever_u.csv").reshape(-1, 1),
        "S": load_csv("cantilever_S.csv"),
        "R": load_csv("cantilever_R.csv"),
        "X": load_csv("cantilever_X.csv"),
        "T": load_csv("cantilever_T.csv"),
    }
    sc_tri = {
        "u": load_csv("triangle_u.csv").reshape(-1, 1),
        "S": load_csv("triangle_S.csv"),
        "R": load_csv("triangle_R.csv"),
        "X": load_csv("triangle_X.csv"),
        "T": load_csv("triangle_T.csv"),
    }

    metrics = {
        "cantilever": make_figure(
            "Cantilever",
            sc_canti["X"],
            sc_canti["T"],
            sc_canti["u"],
            py_canti["u"],
            sc_canti["S"][:, 0::3],
            py_canti["S"][:, 0::3],
            scale=1.0,
            is_quad=True,
        ),
        "triangle": make_figure(
            "Triangle",
            sc_tri["X"],
            sc_tri["T"],
            sc_tri["u"],
            py_tri["u"],
            sc_tri["S"][:, :1],
            py_tri["S"][:, :1],
            scale=1000.0,
            is_quad=False,
        ),
    }
    metrics["cantilever"]["reaction_max_abs_diff"] = float(np.max(np.abs(sc_canti["R"] - py_canti["R"])))
    metrics["triangle"]["reaction_max_abs_diff"] = float(np.max(np.abs(sc_tri["R"] - py_tri["R"])))
    metrics["notes"] = {
        "triangle_mesh_source": "Scilab 2025 hangs in the legacy load_gmsh.sci path, so the triangle parity run feeds Scilab the exact mesh connectivity extracted from mesh/deneme.msh before solving.",
        "scilab_cli": str(find_scilab()),
    }
    (ASSETS / "parity_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
