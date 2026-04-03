"""FemLab GUI — main application window.

Launch with:
    python -m femlabpy.gui
"""

from __future__ import annotations

import logging
import sys
import traceback

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QMainWindow,
    QMessageBox,
    QSplitter,
    QStatusBar,
    QToolBar,
)

from .model import FEModel
from .panels import GmshImportDialog, ModelPanel, ResultsPanel
from .solver_bridge import solve_model
from .viewport import Viewport

log = logging.getLogger("femlabpy.gui")


class MainWindow(QMainWindow):
    """FemLab GUI main window."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("FemLab — FEM Pre/Post-Processor")
        self.resize(1400, 900)

        # Model
        self.model = FEModel()

        # --- Central layout: splitter with [panel | viewport | results] ---
        splitter = QSplitter(Qt.Orientation.Horizontal)
        self.setCentralWidget(splitter)

        # Left panel
        self.panel = ModelPanel()
        self.panel.set_model(self.model)
        splitter.addWidget(self.panel)

        # 3D viewport
        self.viewport = Viewport(self)
        self.viewport.set_model(self.model)
        splitter.addWidget(self.viewport)

        # Right panel
        self.results_panel = ResultsPanel()
        splitter.addWidget(self.results_panel)

        splitter.setSizes([320, 760, 280])

        # --- Menu bar ---
        self._build_menus()

        # --- Toolbar ---
        self._build_toolbar()

        # --- Status bar ---
        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self.status.showMessage("Ready — load an example or build a model")

        # --- Connections ---
        self.panel.model_changed.connect(self._on_model_changed)
        self.results_panel.show_displacement.connect(self.viewport.show_displacement)
        self.results_panel.show_stress.connect(self.viewport.show_stress)
        self.results_panel.show_mesh.connect(self.viewport.show_mesh_only)

        # Initial refresh
        self.viewport.refresh()

    # -----------------------------------------------------------------------
    # Menu bar
    # -----------------------------------------------------------------------
    def _build_menus(self) -> None:
        menubar = self.menuBar()

        # File
        file_menu = menubar.addMenu("&File")

        new_act = QAction("&New Model", self)
        new_act.setShortcut(QKeySequence.StandardKey.New)
        new_act.triggered.connect(self._new_model)
        file_menu.addAction(new_act)

        import_gmsh_act = QAction("Import &Gmsh Mesh…", self)
        import_gmsh_act.triggered.connect(self._import_gmsh)
        file_menu.addAction(import_gmsh_act)

        file_menu.addSeparator()

        export_act = QAction("&Export Results…", self)
        export_act.triggered.connect(self._export_results)
        file_menu.addAction(export_act)

        file_menu.addSeparator()

        quit_act = QAction("&Quit", self)
        quit_act.setShortcut(QKeySequence.StandardKey.Quit)
        quit_act.triggered.connect(self.close)
        file_menu.addAction(quit_act)

        # Examples
        example_menu = menubar.addMenu("&Examples")
        for name, label in [
            ("cantilever", "Cantilever Beam (Q4)"),
            ("flow_q4", "Potential Flow (Q4)"),
            ("gmsh_triangle", "Gmsh Triangle Mesh (T3)"),
        ]:
            act = QAction(label, self)
            act.triggered.connect(lambda checked, n=name: self._load_example(n))
            example_menu.addAction(act)

        # Solve
        solve_menu = menubar.addMenu("&Solve")
        solve_act = QAction("&Run Solver", self)
        solve_act.setShortcut(QKeySequence("F5"))
        solve_act.triggered.connect(self._run_solver)
        solve_menu.addAction(solve_act)

        # View
        view_menu = menubar.addMenu("&View")

        node_labels_act = QAction("Node Labels", self, checkable=True, checked=True)
        node_labels_act.triggered.connect(self.viewport.toggle_node_labels)
        view_menu.addAction(node_labels_act)

        elem_labels_act = QAction("Element Labels", self, checkable=True, checked=True)
        elem_labels_act.triggered.connect(self.viewport.toggle_element_labels)
        view_menu.addAction(elem_labels_act)

        nodes_act = QAction("Show Nodes", self, checkable=True, checked=True)
        nodes_act.triggered.connect(self.viewport.toggle_nodes)
        view_menu.addAction(nodes_act)

        edges_act = QAction("Show Edges", self, checkable=True, checked=True)
        edges_act.triggered.connect(self.viewport.toggle_edges)
        view_menu.addAction(edges_act)

        view_menu.addSeparator()
        reset_cam = QAction("Reset Camera", self)
        reset_cam.setShortcut(QKeySequence("R"))
        reset_cam.triggered.connect(self.viewport.reset_camera)
        view_menu.addAction(reset_cam)

        # Help
        help_menu = menubar.addMenu("&Help")
        about_act = QAction("&About FemLab", self)
        about_act.triggered.connect(self._show_about)
        help_menu.addAction(about_act)

    # -----------------------------------------------------------------------
    # Toolbar
    # -----------------------------------------------------------------------
    def _build_toolbar(self) -> None:
        tb = QToolBar("Main")
        tb.setMovable(False)
        self.addToolBar(tb)

        tb.addAction("New").triggered.connect(self._new_model)
        tb.addAction("Cantilever").triggered.connect(
            lambda: self._load_example("cantilever")
        )
        tb.addAction("Flow").triggered.connect(
            lambda: self._load_example("flow_q4")
        )
        tb.addAction("Triangle").triggered.connect(
            lambda: self._load_example("gmsh_triangle")
        )
        tb.addSeparator()
        solve_btn = tb.addAction("▶ Solve (F5)")
        solve_btn.triggered.connect(self._run_solver)
        tb.addSeparator()
        tb.addAction("Displacement").triggered.connect(
            self.viewport.show_displacement
        )
        tb.addAction("Stress").triggered.connect(self.viewport.show_stress)
        tb.addAction("Mesh").triggered.connect(self.viewport.show_mesh_only)

    # -----------------------------------------------------------------------
    # Slots
    # -----------------------------------------------------------------------
    def _on_model_changed(self) -> None:
        log.debug(
            "model_changed: etype=%s nodes=%d elems=%d dof=%d BCs=%d loads=%d",
            self.model.element_type, self.model.n_nodes,
            self.model.n_elements, self.model.dof,
            self.model.bcs.shape[0], self.model.loads.shape[0],
        )
        self.panel.refresh()
        self.viewport.refresh()
        self.results_panel.clear_results()
        self.status.showMessage(
            f"{self.model.element_type} | "
            f"{self.model.n_nodes} nodes | "
            f"{self.model.n_elements} elements"
        )

    def _new_model(self) -> None:
        log.info("new_model")
        self.model = FEModel()
        self.panel.set_model(self.model)
        self.viewport.set_model(self.model)
        self.viewport.refresh()
        self.results_panel.clear_results()
        self.status.showMessage("New model created")

    def _load_example(self, name: str) -> None:
        log.info("load_example: %s", name)
        try:
            self.model = FEModel()
            self.model.load_example(name)
            log.info(
                "example loaded: nodes=%d elems=%d etype=%s dof=%d "
                "BCs=%d loads=%d materials=%d",
                self.model.n_nodes, self.model.n_elements,
                self.model.element_type, self.model.dof,
                self.model.bcs.shape[0], self.model.loads.shape[0],
                len(self.model.materials),
            )
            self.panel.set_model(self.model)
            self.viewport.set_model(self.model)
            self.viewport.refresh()
            self.results_panel.clear_results()
            self.status.showMessage(f"Loaded example: {name}")
        except Exception as exc:
            log.exception("Failed to load example %s", name)
            QMessageBox.critical(self, "Error", f"Failed to load example:\n{exc}")

    def _import_gmsh(self) -> None:
        dlg = GmshImportDialog(self)
        if not dlg.exec():
            return
        path = dlg.path_edit.text().strip()
        if not path:
            return
        log.info("import_gmsh: %s", path)
        try:
            from .. import load_gmsh2

            mesh = load_gmsh2(path)
            self.model = FEModel()

            # Use triangles or quads from the mesh
            X = mesh.positions[:, :2]
            self.model.nodes = X

            if mesh.triangles.shape[0] > 0:
                self.model.elements = mesh.triangles
                self.model.element_type = "T3"
            elif mesh.quads.shape[0] > 0:
                self.model.elements = mesh.quads
                self.model.element_type = "Q4"

            self.panel.set_model(self.model)
            self.viewport.set_model(self.model)
            self.viewport.refresh()
            self.results_panel.clear_results()
            self.status.showMessage(f"Imported: {path}")
        except Exception as exc:
            QMessageBox.critical(
                self, "Import Error", f"Failed to import Gmsh file:\n{exc}"
            )

    def _run_solver(self) -> None:
        log.info("run_solver requested")
        if self.model.n_nodes == 0 or self.model.n_elements == 0:
            log.warning("cannot solve: no nodes/elements")
            QMessageBox.warning(
                self, "Cannot solve", "Model has no nodes or elements."
            )
            return
        if self.model.bcs.shape[0] == 0:
            log.warning("cannot solve: no BCs")
            QMessageBox.warning(
                self,
                "Cannot solve",
                "No boundary conditions defined. The system is unconstrained.",
            )
            return

        log.info(
            "solving: etype=%s dof=%d nodes=%d elems=%d BCs=%d loads=%d",
            self.model.element_type, self.model.dof,
            self.model.n_nodes, self.model.n_elements,
            self.model.bcs.shape[0], self.model.loads.shape[0],
        )
        self.status.showMessage("Solving…")
        QApplication.processEvents()
        try:
            results = solve_model(self.model)
            log.info(
                "solve complete: max|u|=%.6g  S=%s  R=%s",
                np.max(np.abs(results["u"])),
                results["S"].shape if results["S"] is not None else None,
                results["R"].shape if results["R"] is not None else None,
            )
            self.panel.refresh()
            self.results_panel.set_results(
                results["u"], results["S"], results["R"]
            )
            self.viewport.show_displacement()
            self.status.showMessage(
                f"Solved — max |u| = {np.max(np.abs(results['u'])):.6g}"
            )
        except Exception as exc:
            log.exception("Solver failed")
            tb = traceback.format_exc()
            QMessageBox.critical(
                self, "Solver Error", f"Solver failed:\n{exc}\n\n{tb}"
            )
            self.status.showMessage("Solver failed")

    def _export_results(self) -> None:
        if not self.model.solved or self.model.u is None:
            QMessageBox.information(self, "No results", "Solve the model first.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Displacements",
            "displacements.tsv",
            "TSV files (*.tsv);;All files (*)",
        )
        if not path:
            return
        np.savetxt(path, self.model.u, delimiter="\t", fmt="%.16g")
        self.status.showMessage(f"Exported to {path}")

    def _show_about(self) -> None:
        QMessageBox.about(
            self,
            "About FemLab",
            "FemLab — Finite Element Method Teaching Toolbox\n\n"
            "Python port with PySide6 + PyVista GUI.\n"
            "Original MATLAB toolbox by O. Hededal & S. Krenk.\n"
            "Scilab wrapper by G. Turan (IYTE).\n\n"
            "Elements: Bar, T3, Q4, T4, H8\n"
            "Analysis: Elastic, Potential, Elastoplastic",
        )

    def closeEvent(self, event) -> None:
        self.viewport.close()
        super().closeEvent(event)


def main() -> None:
    """Entry point for the FemLab GUI."""
    app = QApplication.instance() or QApplication(sys.argv)
    window = MainWindow()
    window.show()
    if not QApplication.instance() or not getattr(app, "_exec_called", False):
        app._exec_called = True  # type: ignore[union-attr]
        sys.exit(app.exec())
