"""Side panels — model tree, property editor, tables for nodes/elements/BCs/loads."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

if TYPE_CHECKING:
    from .model import FEModel


# ---------------------------------------------------------------------------
# Reusable numeric table
# ---------------------------------------------------------------------------
class DataTable(QTableWidget):
    """Read-only numeric data table with column headers."""

    def __init__(self, headers: list[str], parent=None):
        super().__init__(0, len(headers), parent)
        self.setHorizontalHeaderLabels(headers)
        self.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)

    def set_data(self, data: np.ndarray) -> None:
        self.setRowCount(0)
        if data.size == 0:
            return
        if data.ndim == 1:
            data = data.reshape(1, -1)
        self.setRowCount(data.shape[0])
        for i in range(data.shape[0]):
            for j in range(min(data.shape[1], self.columnCount())):
                val = data[i, j]
                txt = str(int(val)) if val == int(val) else f"{val:.6g}"
                item = QTableWidgetItem(txt)
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                self.setItem(i, j, item)


# ---------------------------------------------------------------------------
# Node editor dialog
# ---------------------------------------------------------------------------
class AddNodeDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Add Node")
        layout = QFormLayout(self)
        self.x_spin = QDoubleSpinBox()
        self.x_spin.setRange(-1e6, 1e6)
        self.x_spin.setDecimals(6)
        self.y_spin = QDoubleSpinBox()
        self.y_spin.setRange(-1e6, 1e6)
        self.y_spin.setDecimals(6)
        layout.addRow("X:", self.x_spin)
        layout.addRow("Y:", self.y_spin)
        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        layout.addRow(btns)


# ---------------------------------------------------------------------------
# Element editor dialog
# ---------------------------------------------------------------------------
class AddElementDialog(QDialog):
    def __init__(self, n_nodes: int, elem_type: str = "Q4", parent=None):
        super().__init__(parent)
        self.setWindowTitle("Add Element")
        layout = QFormLayout(self)
        n_per_elem = 4 if elem_type == "Q4" else 3
        self.node_spins: list[QSpinBox] = []
        for i in range(n_per_elem):
            spin = QSpinBox()
            spin.setRange(1, max(n_nodes, 1))
            self.node_spins.append(spin)
            layout.addRow(f"Node {i + 1}:", spin)
        self.prop_spin = QSpinBox()
        self.prop_spin.setRange(1, 100)
        self.prop_spin.setValue(1)
        layout.addRow("Property ID:", self.prop_spin)
        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        layout.addRow(btns)


# ---------------------------------------------------------------------------
# BC dialog
# ---------------------------------------------------------------------------
class AddBCDialog(QDialog):
    def __init__(self, n_nodes: int, dof: int = 2, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Add Boundary Condition")
        layout = QFormLayout(self)
        self.node_spin = QSpinBox()
        self.node_spin.setRange(1, max(n_nodes, 1))
        layout.addRow("Node:", self.node_spin)
        self.dof_spin = QSpinBox()
        self.dof_spin.setRange(1, dof)
        layout.addRow("DOF Component:", self.dof_spin)
        self.val_spin = QDoubleSpinBox()
        self.val_spin.setRange(-1e6, 1e6)
        self.val_spin.setDecimals(6)
        layout.addRow("Value:", self.val_spin)
        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        layout.addRow(btns)


# ---------------------------------------------------------------------------
# Load dialog
# ---------------------------------------------------------------------------
class AddLoadDialog(QDialog):
    def __init__(self, n_nodes: int, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Add Nodal Load")
        layout = QFormLayout(self)
        self.node_spin = QSpinBox()
        self.node_spin.setRange(1, max(n_nodes, 1))
        layout.addRow("Node:", self.node_spin)
        self.fx_spin = QDoubleSpinBox()
        self.fx_spin.setRange(-1e12, 1e12)
        self.fx_spin.setDecimals(6)
        layout.addRow("Fx:", self.fx_spin)
        self.fy_spin = QDoubleSpinBox()
        self.fy_spin.setRange(-1e12, 1e12)
        self.fy_spin.setDecimals(6)
        layout.addRow("Fy:", self.fy_spin)
        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        layout.addRow(btns)


# ---------------------------------------------------------------------------
# Material editor dialog
# ---------------------------------------------------------------------------
class MaterialDialog(QDialog):
    def __init__(self, props: list[float] | None = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Material Properties")
        layout = QFormLayout(self)
        props = props or [1.0, 0.3, 1.0]
        self.name_edit = QLineEdit("Default")
        layout.addRow("Name:", self.name_edit)

        self.e_spin = QDoubleSpinBox()
        self.e_spin.setRange(0, 1e15)
        self.e_spin.setDecimals(4)
        self.e_spin.setValue(props[0] if len(props) > 0 else 1.0)
        layout.addRow("E (Young's modulus):", self.e_spin)

        self.nu_spin = QDoubleSpinBox()
        self.nu_spin.setRange(0, 0.5)
        self.nu_spin.setDecimals(4)
        self.nu_spin.setValue(props[1] if len(props) > 1 else 0.3)
        layout.addRow("ν (Poisson ratio):", self.nu_spin)

        self.ptype_combo = QComboBox()
        self.ptype_combo.addItems(["1 — Plane stress", "2 — Plane strain"])
        if len(props) > 2:
            self.ptype_combo.setCurrentIndex(int(props[2]) - 1)
        layout.addRow("Plane type:", self.ptype_combo)

        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        layout.addRow(btns)


# ---------------------------------------------------------------------------
# Gmsh import dialog
# ---------------------------------------------------------------------------
class GmshImportDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Import Gmsh Mesh")
        layout = QFormLayout(self)
        self.path_edit = QLineEdit()
        layout.addRow("File path:", self.path_edit)
        browse = QPushButton("Browse…")
        browse.clicked.connect(self._browse)
        layout.addRow(browse)
        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        layout.addRow(btns)

    def _browse(self) -> None:
        from PySide6.QtWidgets import QFileDialog

        path, _ = QFileDialog.getOpenFileName(
            self, "Select Gmsh file", "", "Gmsh files (*.msh *.geo);;All files (*)"
        )
        if path:
            self.path_edit.setText(path)


# ---------------------------------------------------------------------------
# Model panel (left sidebar with tabs)
# ---------------------------------------------------------------------------
class ModelPanel(QWidget):
    """Left sidebar: tabbed panel showing nodes, elements, BCs, loads, materials."""

    model_changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._model: FEModel | None = None
        self.setMinimumWidth(320)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        # Info bar
        self.info_label = QLabel("No model loaded")
        layout.addWidget(self.info_label)

        # Tabs
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)

        # -- Nodes tab --
        nodes_widget = QWidget()
        nodes_layout = QVBoxLayout(nodes_widget)
        self.node_table = DataTable(["#", "X", "Y"])
        nodes_layout.addWidget(self.node_table)
        node_btns = QHBoxLayout()
        self.add_node_btn = QPushButton("Add")
        self.del_node_btn = QPushButton("Delete")
        node_btns.addWidget(self.add_node_btn)
        node_btns.addWidget(self.del_node_btn)
        nodes_layout.addLayout(node_btns)
        self.tabs.addTab(nodes_widget, "Nodes")

        # -- Elements tab --
        elems_widget = QWidget()
        elems_layout = QVBoxLayout(elems_widget)
        self.elem_table = DataTable(["#", "N1", "N2", "N3", "N4", "Prop"])
        elems_layout.addWidget(self.elem_table)
        elem_btns = QHBoxLayout()
        self.add_elem_btn = QPushButton("Add")
        self.del_elem_btn = QPushButton("Delete")
        elem_btns.addWidget(self.add_elem_btn)
        elem_btns.addWidget(self.del_elem_btn)
        elems_layout.addLayout(elem_btns)
        self.tabs.addTab(elems_widget, "Elements")

        # -- BCs tab --
        bcs_widget = QWidget()
        bcs_layout = QVBoxLayout(bcs_widget)
        self.bc_table = DataTable(["Node", "DOF", "Value"])
        bcs_layout.addWidget(self.bc_table)
        bc_btns = QHBoxLayout()
        self.add_bc_btn = QPushButton("Add")
        self.del_bc_btn = QPushButton("Delete")
        self.clear_bc_btn = QPushButton("Clear All")
        bc_btns.addWidget(self.add_bc_btn)
        bc_btns.addWidget(self.del_bc_btn)
        bc_btns.addWidget(self.clear_bc_btn)
        bcs_layout.addLayout(bc_btns)
        self.tabs.addTab(bcs_widget, "BCs")

        # -- Loads tab --
        loads_widget = QWidget()
        loads_layout = QVBoxLayout(loads_widget)
        self.load_table = DataTable(["Node", "Fx", "Fy"])
        loads_layout.addWidget(self.load_table)
        load_btns = QHBoxLayout()
        self.add_load_btn = QPushButton("Add")
        self.del_load_btn = QPushButton("Delete")
        self.clear_load_btn = QPushButton("Clear All")
        load_btns.addWidget(self.add_load_btn)
        load_btns.addWidget(self.del_load_btn)
        load_btns.addWidget(self.clear_load_btn)
        loads_layout.addLayout(load_btns)
        self.tabs.addTab(loads_widget, "Loads")

        # -- Material tab --
        mat_widget = QWidget()
        mat_layout = QVBoxLayout(mat_widget)
        self.mat_label = QLabel("E=1.0  ν=0.3  type=1")
        mat_layout.addWidget(self.mat_label)
        self.edit_mat_btn = QPushButton("Edit Material…")
        mat_layout.addWidget(self.edit_mat_btn)
        mat_layout.addStretch()

        # Element type selector
        type_group = QGroupBox("Element Type")
        type_layout = QHBoxLayout(type_group)
        self.type_combo = QComboBox()
        self.type_combo.addItems(["Q4 — Quadrilateral", "T3 — Triangle"])
        type_layout.addWidget(self.type_combo)
        mat_layout.addWidget(type_group)

        # DOF selector
        dof_group = QGroupBox("DOF per node")
        dof_layout = QHBoxLayout(dof_group)
        self.dof_combo = QComboBox()
        self.dof_combo.addItems(["1 — Scalar (potential/flow)", "2 — Vector (elastic)"])
        self.dof_combo.setCurrentIndex(1)
        dof_layout.addWidget(self.dof_combo)
        mat_layout.addWidget(dof_group)

        self.tabs.addTab(mat_widget, "Material")

        # Connections
        self.add_node_btn.clicked.connect(self._on_add_node)
        self.del_node_btn.clicked.connect(self._on_del_node)
        self.add_elem_btn.clicked.connect(self._on_add_element)
        self.del_elem_btn.clicked.connect(self._on_del_element)
        self.add_bc_btn.clicked.connect(self._on_add_bc)
        self.del_bc_btn.clicked.connect(self._on_del_bc)
        self.clear_bc_btn.clicked.connect(self._on_clear_bcs)
        self.add_load_btn.clicked.connect(self._on_add_load)
        self.del_load_btn.clicked.connect(self._on_del_load)
        self.clear_load_btn.clicked.connect(self._on_clear_loads)
        self.edit_mat_btn.clicked.connect(self._on_edit_material)
        self.type_combo.currentIndexChanged.connect(self._on_type_changed)
        self.dof_combo.currentIndexChanged.connect(self._on_dof_changed)

    def set_model(self, model: FEModel) -> None:
        self._model = model
        self.refresh()

    def refresh(self) -> None:
        m = self._model
        if m is None:
            self.info_label.setText("No model")
            return

        self.info_label.setText(
            f"{m.element_type} | {m.n_nodes} nodes | {m.n_elements} elems | "
            f"DOF={m.dof} | {'SOLVED' if m.solved else 'not solved'}"
        )

        # Nodes
        if m.n_nodes > 0:
            idx = np.arange(1, m.n_nodes + 1).reshape(-1, 1)
            self.node_table.set_data(np.hstack([idx, m.nodes]))
        else:
            self.node_table.setRowCount(0)

        # Elements
        if m.n_elements > 0:
            idx = np.arange(1, m.n_elements + 1).reshape(-1, 1)
            self.elem_table.set_data(np.hstack([idx, m.elements]).astype(float))
        else:
            self.elem_table.setRowCount(0)

        # BCs
        if m.bcs.shape[0] > 0:
            self.bc_table.set_data(m.bcs)
        else:
            self.bc_table.setRowCount(0)

        # Loads
        if m.loads.shape[0] > 0:
            self.load_table.set_data(m.loads)
        else:
            self.load_table.setRowCount(0)

        # Material
        if m.materials:
            mat = m.materials[0]
            p = mat.props
            self.mat_label.setText(
                f"E={p[0]:.4g}  ν={p[1] if len(p) > 1 else '?'}  "
                f"type={int(p[2]) if len(p) > 2 else '?'}"
            )

        # Sync combos
        self.type_combo.blockSignals(True)
        self.type_combo.setCurrentIndex(0 if m.element_type == "Q4" else 1)
        self.type_combo.blockSignals(False)
        self.dof_combo.blockSignals(True)
        self.dof_combo.setCurrentIndex(m.dof - 1)
        self.dof_combo.blockSignals(False)

    # --- slots ---
    def _on_add_node(self) -> None:
        dlg = AddNodeDialog(self)
        if dlg.exec() and self._model:
            self._model.add_node(dlg.x_spin.value(), dlg.y_spin.value())
            self.model_changed.emit()

    def _on_del_node(self) -> None:
        row = self.node_table.currentRow()
        if row >= 0 and self._model:
            self._model.remove_node(row)
            self.model_changed.emit()

    def _on_add_element(self) -> None:
        if not self._model or self._model.n_nodes == 0:
            QMessageBox.warning(self, "No nodes", "Add nodes first.")
            return
        dlg = AddElementDialog(self._model.n_nodes, self._model.element_type, self)
        if dlg.exec():
            nodes = [s.value() for s in dlg.node_spins]
            self._model.add_element(nodes, dlg.prop_spin.value())
            self.model_changed.emit()

    def _on_del_element(self) -> None:
        row = self.elem_table.currentRow()
        if row >= 0 and self._model:
            self._model.remove_element(row)
            self.model_changed.emit()

    def _on_add_bc(self) -> None:
        if not self._model or self._model.n_nodes == 0:
            return
        dlg = AddBCDialog(self._model.n_nodes, self._model.dof, self)
        if dlg.exec():
            self._model.add_bc(
                dlg.node_spin.value(), dlg.dof_spin.value(), dlg.val_spin.value()
            )
            self.model_changed.emit()

    def _on_del_bc(self) -> None:
        row = self.bc_table.currentRow()
        if row >= 0 and self._model:
            self._model.remove_bc(row)
            self.model_changed.emit()

    def _on_clear_bcs(self) -> None:
        if self._model:
            self._model.clear_bcs()
            self.model_changed.emit()

    def _on_add_load(self) -> None:
        if not self._model or self._model.n_nodes == 0:
            return
        dlg = AddLoadDialog(self._model.n_nodes, self)
        if dlg.exec():
            self._model.add_load(
                dlg.node_spin.value(), dlg.fx_spin.value(), dlg.fy_spin.value()
            )
            self.model_changed.emit()

    def _on_del_load(self) -> None:
        row = self.load_table.currentRow()
        if row >= 0 and self._model:
            self._model.remove_load(row)
            self.model_changed.emit()

    def _on_clear_loads(self) -> None:
        if self._model:
            self._model.clear_loads()
            self.model_changed.emit()

    def _on_edit_material(self) -> None:
        if not self._model:
            return
        props = self._model.materials[0].props if self._model.materials else None
        dlg = MaterialDialog(props, self)
        if dlg.exec():
            from .model import Material

            new_props = [
                dlg.e_spin.value(),
                dlg.nu_spin.value(),
                float(dlg.ptype_combo.currentIndex() + 1),
            ]
            self._model.materials = [Material(dlg.name_edit.text(), new_props)]
            self.model_changed.emit()

    def _on_type_changed(self, idx: int) -> None:
        if self._model:
            self._model.element_type = "Q4" if idx == 0 else "T3"
            self.model_changed.emit()

    def _on_dof_changed(self, idx: int) -> None:
        if self._model:
            self._model.dof = idx + 1
            self.model_changed.emit()


# ---------------------------------------------------------------------------
# Results panel (right sidebar or bottom)
# ---------------------------------------------------------------------------
class ResultsPanel(QWidget):
    """Shows solver output summary and result display controls."""

    show_displacement = Signal()
    show_stress = Signal()
    show_mesh = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumWidth(250)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        layout.addWidget(QLabel("Results"))
        self.summary_label = QLabel("Not solved yet")
        self.summary_label.setWordWrap(True)
        layout.addWidget(self.summary_label)

        self.disp_btn = QPushButton("Show Displacement")
        self.disp_btn.clicked.connect(self.show_displacement.emit)
        layout.addWidget(self.disp_btn)

        self.stress_btn = QPushButton("Show Stress")
        self.stress_btn.clicked.connect(self.show_stress.emit)
        layout.addWidget(self.stress_btn)

        self.mesh_btn = QPushButton("Show Mesh Only")
        self.mesh_btn.clicked.connect(self.show_mesh.emit)
        layout.addWidget(self.mesh_btn)

        layout.addStretch()

    def set_results(self, u, S, R) -> None:
        lines = []
        if u is not None:
            u_flat = u.ravel()
            lines.append(f"max |u| = {np.max(np.abs(u_flat)):.6g}")
            lines.append(f"‖u‖ = {np.linalg.norm(u_flat):.6g}")
        if S is not None:
            lines.append(f"max |σ| = {np.max(np.abs(S)):.6g}")
        if R is not None:
            lines.append(f"Reactions: {R.shape[0]} entries")
        self.summary_label.setText("\n".join(lines) if lines else "No results")

    def clear_results(self) -> None:
        self.summary_label.setText("Not solved yet")
