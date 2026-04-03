"""PyVista-based 3D viewport widget embedded in Qt."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pyvista as pv
from pyvistaqt import QtInteractor

if TYPE_CHECKING:
    from .model import FEModel


class Viewport(QtInteractor):
    """3D FEM mesh viewer with node/element display and result contours."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._model: FEModel | None = None
        self._mesh_actor = None
        self._node_actor = None
        self._bc_actor = None
        self._load_actor = None

        self.set_background("white")
        self.add_axes()
        self.show_bounds(grid=False, location="outer")

        # Display options
        self._show_nodes = True
        self._show_node_labels = True
        self._show_element_labels = True
        self._show_edges = True
        self._result_mode: str | None = None  # None, "displacement", "stress"

    def set_model(self, model: FEModel) -> None:
        self._model = model

    def refresh(self) -> None:
        """Rebuild the scene from model data."""
        self.clear()
        self.add_axes()
        if self._model is None or self._model.n_nodes == 0:
            self.reset_camera()
            self.update()
            return

        self._draw_mesh()

        if self._show_nodes:
            self._draw_nodes()

        if self._model.bcs.shape[0] > 0:
            self._draw_bcs()

        if self._model.loads.shape[0] > 0:
            self._draw_loads()

        if self._result_mode and self._model.solved:
            self._draw_results()

        self.reset_camera()
        self.update()

    def _make_pv_mesh(self) -> pv.UnstructuredGrid | None:
        """Convert model arrays to a PyVista mesh."""
        m = self._model
        if m is None or m.n_nodes == 0 or m.n_elements == 0:
            return None

        X = m.nodes
        # Pad to 3D if 2D
        if X.shape[1] == 2:
            X3 = np.column_stack([X, np.zeros(X.shape[0])])
        else:
            X3 = X

        T = m.elements
        cells = []
        celltypes = []

        for row in T:
            if m.element_type == "Q4":
                nodes = row[:4] - 1  # 1-based to 0-based
                cells.append([4, *nodes.tolist()])
                celltypes.append(pv.CellType.QUAD)
            elif m.element_type == "T3":
                nodes = row[:3] - 1
                cells.append([3, *nodes.tolist()])
                celltypes.append(pv.CellType.TRIANGLE)

        cells_flat = []
        for c in cells:
            cells_flat.extend(c)

        grid = pv.UnstructuredGrid(
            np.array(cells_flat, dtype=np.int64),
            np.array(celltypes, dtype=np.uint8),
            X3.astype(np.float64),
        )
        return grid

    def _draw_mesh(self) -> None:
        grid = self._make_pv_mesh()
        if grid is None:
            return

        self.add_mesh(
            grid,
            show_edges=self._show_edges,
            color="lightblue",
            opacity=0.7,
            name="fem_mesh",
        )

        if self._show_element_labels and self._model is not None:
            centers = grid.cell_centers()
            labels = [str(i + 1) for i in range(grid.n_cells)]
            self.add_point_labels(
                centers,
                labels,
                font_size=10,
                text_color="blue",
                name="elem_labels",
                shape_opacity=0.3,
            )

    def _draw_nodes(self) -> None:
        m = self._model
        if m is None:
            return
        X = m.nodes
        if X.shape[1] == 2:
            X3 = np.column_stack([X, np.zeros(X.shape[0])])
        else:
            X3 = X

        pts = pv.PolyData(X3.astype(np.float64))
        self.add_mesh(
            pts,
            color="red",
            point_size=8,
            render_points_as_spheres=True,
            name="nodes",
        )

        if self._show_node_labels:
            labels = [str(i + 1) for i in range(X.shape[0])]
            self.add_point_labels(
                X3,
                labels,
                font_size=9,
                text_color="red",
                name="node_labels",
                shape_opacity=0.0,
            )

    def _draw_bcs(self) -> None:
        m = self._model
        if m is None:
            return
        C = m.bcs
        X = m.nodes
        bc_nodes = np.unique(C[:, 0].astype(int)) - 1  # 0-based
        bc_nodes = bc_nodes[bc_nodes < X.shape[0]]
        if len(bc_nodes) == 0:
            return

        pts_2d = X[bc_nodes]
        if pts_2d.shape[1] == 2:
            pts_3d = np.column_stack([pts_2d, np.zeros(len(bc_nodes))])
        else:
            pts_3d = pts_2d

        pts = pv.PolyData(pts_3d.astype(np.float64))
        self.add_mesh(
            pts,
            color="green",
            point_size=14,
            render_points_as_spheres=True,
            name="bcs",
        )

    def _draw_loads(self) -> None:
        m = self._model
        if m is None:
            return
        P = m.loads
        X = m.nodes
        if P.shape[0] == 0:
            return

        origins = []
        vectors = []
        for row in P:
            node_idx = int(row[0]) - 1
            if node_idx < 0 or node_idx >= X.shape[0]:
                continue
            pos = X[node_idx]
            if len(row) >= 3:
                fx, fy = row[1], row[2]
            else:
                fx, fy = row[1], 0.0
            if pos.shape[0] == 2:
                origins.append([pos[0], pos[1], 0.0])
                vectors.append([fx, fy, 0.0])
            else:
                origins.append(pos.tolist())
                vectors.append([fx, fy, 0.0])

        if not origins:
            return

        origins = np.array(origins, dtype=np.float64)
        vectors = np.array(vectors, dtype=np.float64)

        # Normalize for display; scale to ~10% of model size
        mag = np.linalg.norm(vectors, axis=1, keepdims=True)
        max_mag = mag.max()
        if max_mag > 0:
            bbox = X.max(axis=0) - X.min(axis=0)
            scale = 0.15 * max(bbox) / max_mag
            vectors = vectors * scale

        for i in range(len(origins)):
            direction = vectors[i]
            norm = np.linalg.norm(direction)
            if norm < 1e-30:
                continue
            arrow = pv.Arrow(
                start=origins[i].tolist(),
                direction=direction.tolist(),
                scale=norm,
                tip_length=0.3,
                shaft_radius=0.06,
            )
            self.add_mesh(arrow, color="orange", name=f"load_{i}")

    def _draw_results(self) -> None:
        m = self._model
        if m is None or not m.solved:
            return

        grid = self._make_pv_mesh()
        if grid is None:
            return

        if self._result_mode == "displacement" and m.u is not None:
            u = m.u.ravel()
            if m.dof == 2:
                u_mag = np.sqrt(u[0::2] ** 2 + u[1::2] ** 2)
            elif m.dof == 1:
                u_mag = np.abs(u)
            else:
                u_mag = np.abs(u)

            if len(u_mag) == grid.n_points:
                grid.point_data["Displacement"] = u_mag
                self.add_mesh(
                    grid,
                    scalars="Displacement",
                    show_edges=True,
                    cmap="jet",
                    name="result_mesh",
                    scalar_bar_args={"title": "Displacement Magnitude"},
                )

                # Draw deformed shape
                if m.dof == 2 and len(u) == 2 * grid.n_points:
                    disp3 = np.column_stack([u[0::2], u[1::2], np.zeros(grid.n_points)])
                    scale_factor = (
                        0.1
                        * max(m.nodes.max(axis=0) - m.nodes.min(axis=0))
                        / max(u_mag.max(), 1e-30)
                    )
                    deformed = grid.copy()
                    deformed.points += disp3 * scale_factor
                    self.add_mesh(
                        deformed,
                        show_edges=True,
                        style="wireframe",
                        color="darkred",
                        line_width=2,
                        name="deformed",
                    )

        elif self._result_mode == "stress" and m.stresses is not None:
            S = m.stresses
            vm = self._compute_von_mises(S, m.element_type, m.dof)
            if vm is not None and len(vm) == grid.n_cells:
                grid.cell_data["VonMises"] = vm
                self.add_mesh(
                    grid,
                    scalars="VonMises",
                    show_edges=True,
                    cmap="hot",
                    name="result_mesh",
                    scalar_bar_args={"title": "Von Mises Stress"},
                )

    @staticmethod
    def _compute_von_mises(S: np.ndarray, etype: str, dof: int) -> np.ndarray | None:
        """Compute von Mises (or magnitude for potential) per element."""
        if S is None or S.size == 0:
            return None
        ne = S.shape[0]
        if dof == 2:
            if etype == "T3":
                # S shape (ne, 3): [sxx, syy, txy] per element
                sxx, syy, sxy = S[:, 0], S[:, 1], S[:, 2]
            elif etype == "Q4":
                # S shape (ne, 12): 4 GPs x [sigma_x, sigma_y, tau_xy], flattened
                S4 = S.reshape(ne, 4, 3)
                sxx = S4[:, :, 0].mean(axis=1)
                syy = S4[:, :, 1].mean(axis=1)
                sxy = S4[:, :, 2].mean(axis=1)
            else:
                return None
            return np.sqrt(sxx**2 - sxx * syy + syy**2 + 3 * sxy**2)
        else:
            # Potential problem — gradient magnitude
            if etype == "T3":
                # S shape (ne, 2): [dT/dx, dT/dy]
                return np.sqrt(S[:, 0] ** 2 + S[:, 1] ** 2)
            elif etype == "Q4":
                # S shape (ne, 8): 4 GPs x [dT/dx, dT/dy]
                S4 = S.reshape(ne, 4, 2)
                gx = S4[:, :, 0].mean(axis=1)
                gy = S4[:, :, 1].mean(axis=1)
                return np.sqrt(gx**2 + gy**2)
        return None

    def show_displacement(self) -> None:
        self._result_mode = "displacement"
        self.refresh()

    def show_stress(self) -> None:
        self._result_mode = "stress"
        self.refresh()

    def show_mesh_only(self) -> None:
        self._result_mode = None
        self.refresh()

    def toggle_node_labels(self, on: bool) -> None:
        self._show_node_labels = on
        self.refresh()

    def toggle_element_labels(self, on: bool) -> None:
        self._show_element_labels = on
        self.refresh()

    def toggle_nodes(self, on: bool) -> None:
        self._show_nodes = on
        self.refresh()

    def toggle_edges(self, on: bool) -> None:
        self._show_edges = on
        self.refresh()
