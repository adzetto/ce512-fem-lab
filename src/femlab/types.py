from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class GmshMesh:
    positions: np.ndarray
    element_infos: np.ndarray
    element_tags: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=int))
    element_nodes: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=int))
    nb_type: np.ndarray = field(default_factory=lambda: np.zeros(19, dtype=int))
    points: np.ndarray = field(default_factory=lambda: np.zeros((0, 2), dtype=int))
    lines: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), dtype=int))
    lines2: np.ndarray = field(default_factory=lambda: np.zeros((0, 4), dtype=int))
    triangles: np.ndarray = field(default_factory=lambda: np.zeros((0, 4), dtype=int))
    quads: np.ndarray = field(default_factory=lambda: np.zeros((0, 5), dtype=int))
    tets: np.ndarray = field(default_factory=lambda: np.zeros((0, 5), dtype=int))
    tets10: np.ndarray = field(default_factory=lambda: np.zeros((0, 11), dtype=int))
    hexa: np.ndarray = field(default_factory=lambda: np.zeros((0, 9), dtype=int))
    hexa20: np.ndarray = field(default_factory=lambda: np.zeros((0, 21), dtype=int))
    hexa27: np.ndarray = field(default_factory=lambda: np.zeros((0, 28), dtype=int))
    prism: np.ndarray = field(default_factory=lambda: np.zeros((0, 7), dtype=int))
    prism15: np.ndarray = field(default_factory=lambda: np.zeros((0, 16), dtype=int))
    prism18: np.ndarray = field(default_factory=lambda: np.zeros((0, 19), dtype=int))
    pyramid: np.ndarray = field(default_factory=lambda: np.zeros((0, 6), dtype=int))
    pyramid13: np.ndarray = field(default_factory=lambda: np.zeros((0, 14), dtype=int))
    pyramid14: np.ndarray = field(default_factory=lambda: np.zeros((0, 15), dtype=int))
    qtriangles: np.ndarray = field(default_factory=lambda: np.zeros((0, 7), dtype=int))
    quads8: np.ndarray = field(default_factory=lambda: np.zeros((0, 9), dtype=int))
    quads9: np.ndarray = field(default_factory=lambda: np.zeros((0, 10), dtype=int))
    bounds_min: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=float))
    bounds_max: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=float))

    def property_numbers(self, element_refs: np.ndarray, *, info_column: int = 4) -> np.ndarray:
        refs = np.asarray(element_refs, dtype=int).ravel()
        return self.element_infos[refs - 1, info_column]
