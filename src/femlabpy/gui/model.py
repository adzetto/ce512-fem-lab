"""FEM model data container — owns nodes, elements, materials, BCs, loads."""

from __future__ import annotations

import numpy as np


class Material:
    """A named set of material properties."""

    def __init__(self, name: str = "Default", props: list[float] | None = None):
        self.name = name
        self.props = props if props is not None else [1.0, 0.3, 1.0]

    def as_array(self) -> np.ndarray:
        return np.array(self.props, dtype=float)

    def __repr__(self) -> str:
        return f"Material({self.name!r}, {self.props})"


class FEModel:
    """Central data model for the FEM GUI session.

    Stores raw arrays compatible with the femlab solver functions.
    """

    def __init__(self) -> None:
        self.nodes: np.ndarray = np.empty((0, 2), dtype=float)
        self.elements: np.ndarray = np.empty((0, 5), dtype=int)
        self.materials: list[Material] = [Material()]
        self.bcs: np.ndarray = np.empty((0, 3), dtype=float)
        self.loads: np.ndarray = np.empty((0, 3), dtype=float)
        self.dof: int = 2
        self.element_type: str = "Q4"

        # Results (populated after solve)
        self.u: np.ndarray | None = None
        self.stresses: np.ndarray | None = None
        self.reactions: np.ndarray | None = None
        self.solved: bool = False

    # --- node helpers ---
    @property
    def n_nodes(self) -> int:
        return self.nodes.shape[0]

    @property
    def n_elements(self) -> int:
        return self.elements.shape[0]

    def add_node(self, x: float, y: float) -> int:
        """Add a node, return its 1-based index."""
        self.nodes = np.vstack([self.nodes, [x, y]])
        self._invalidate()
        return self.n_nodes

    def remove_node(self, idx: int) -> None:
        """Remove node at 0-based index."""
        if 0 <= idx < self.n_nodes:
            self.nodes = np.delete(self.nodes, idx, axis=0)
            self._invalidate()

    def move_node(self, idx: int, x: float, y: float) -> None:
        if 0 <= idx < self.n_nodes:
            self.nodes[idx] = [x, y]
            self._invalidate()

    # --- element helpers ---
    def add_element(self, connectivity: list[int], prop: int = 1) -> int:
        """Add an element. connectivity is list of 1-based node indices."""
        if self.element_type == "Q4":
            if len(connectivity) != 4:
                raise ValueError("Q4 element needs 4 nodes")
            row = connectivity + [prop]
        elif self.element_type == "T3":
            if len(connectivity) != 3:
                raise ValueError("T3 element needs 3 nodes")
            row = connectivity + [prop]
        else:
            row = connectivity + [prop]
        cols = (
            max(len(row), self.elements.shape[1]) if self.n_elements > 0 else len(row)
        )
        if self.n_elements == 0:
            self.elements = np.array([row], dtype=int)
        else:
            if len(row) < cols:
                row = row + [0] * (cols - len(row))
            self.elements = np.vstack([self.elements, row])
        self._invalidate()
        return self.n_elements

    def remove_element(self, idx: int) -> None:
        if 0 <= idx < self.n_elements:
            self.elements = np.delete(self.elements, idx, axis=0)
            self._invalidate()

    # --- BC helpers ---
    def add_bc(self, node: int, dof_comp: int, value: float = 0.0) -> None:
        """Add a boundary condition. node is 1-based."""
        self.bcs = np.vstack([self.bcs, [node, dof_comp, value]])
        self._invalidate()

    def remove_bc(self, idx: int) -> None:
        if 0 <= idx < self.bcs.shape[0]:
            self.bcs = np.delete(self.bcs, idx, axis=0)
            self._invalidate()

    def clear_bcs(self) -> None:
        self.bcs = np.empty((0, 3), dtype=float)
        self._invalidate()

    # --- load helpers ---
    def add_load(self, node: int, fx: float, fy: float) -> None:
        """Add a nodal load. node is 1-based."""
        self.loads = np.vstack([self.loads, [node, fx, fy]])
        self._invalidate()

    def remove_load(self, idx: int) -> None:
        if 0 <= idx < self.loads.shape[0]:
            self.loads = np.delete(self.loads, idx, axis=0)
            self._invalidate()

    def clear_loads(self) -> None:
        self.loads = np.empty((0, 3), dtype=float)
        self._invalidate()

    # --- serialization ---
    def to_dict(self) -> dict:
        # G must be 2D (nm, cols) — assembly functions expect rows of materials
        g_rows = [m.as_array() for m in self.materials]
        G = np.atleast_2d(np.array(g_rows, dtype=float))
        return {
            "X": self.nodes.copy(),
            "T": self.elements.copy(),
            "G": G,
            "C": self.bcs.copy(),
            "P": self.loads.copy(),
            "dof": self.dof,
        }

    def load_from_dict(self, data: dict) -> None:
        self.nodes = np.array(data["X"], dtype=float)
        self.elements = np.array(data["T"], dtype=int)
        if "G" in data:
            G = np.atleast_2d(np.array(data["G"], dtype=float))
            self.materials = [
                Material(f"Material {i + 1}", row.tolist()) for i, row in enumerate(G)
            ]
        if "C" in data:
            self.bcs = np.array(data["C"], dtype=float)
        if "P" in data:
            self.loads = np.array(data["P"], dtype=float)
        if "dof" in data:
            self.dof = int(data["dof"])
        ncols = self.elements.shape[1] if self.n_elements > 0 else 5
        if ncols == 5:
            self.element_type = "Q4"
        elif ncols == 4:
            self.element_type = "T3"
        self._invalidate()

    def load_example(self, name: str) -> None:
        if name == "cantilever":
            from ..examples.cantilever import cantilever_data

            self.load_from_dict(cantilever_data())
        elif name == "flow_q4":
            from ..examples.flow import flow_data

            data = flow_data()
            self.nodes = np.array(data["X"], dtype=float)
            self.elements = np.array(data["T1"], dtype=int)
            props = np.array(data["G"], dtype=float).ravel().tolist()
            self.materials = [Material("Default", props)]
            self.bcs = np.array(data["C"], dtype=float)
            self.loads = np.empty((0, 3), dtype=float)
            self.dof = data["dof"]
            self.element_type = "Q4"
            self._invalidate()
        elif name == "gmsh_triangle":
            from ..examples.gmsh_triangle import gmsh_triangle_data

            self.load_from_dict(gmsh_triangle_data())
            self.element_type = "T3"
            self._invalidate()

    def _invalidate(self) -> None:
        self.u = None
        self.stresses = None
        self.reactions = None
        self.solved = False
