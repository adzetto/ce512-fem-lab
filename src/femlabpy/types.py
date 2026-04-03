from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np


@dataclass
class GmshMesh:
    """
    Parsed Gmsh mesh with both normalized Python fields and legacy FemLab aliases.

    Lowercase attributes such as ``positions``, ``element_infos``, and
    ``triangles`` form the stable Python API used internally by ``femlabpy``.
    Uppercase aliases such as ``POS``, ``ELE_INFOS``, ``TRIANGLES``,
    ``nbTriangles``, ``MIN``, and ``MAX`` are exposed through ``__getattr__``
    to mimic the original MATLAB loaders.
    """
    _LEGACY_FIELD_MAP: ClassVar[dict[str, tuple[str | None, int | None]]] = {
        "POS": ("positions", None),
        "ELE_INFOS": ("legacy_element_infos", None),
        "ELE_TAGS": ("legacy_element_tags", None),
        "ELE_NODES": ("element_nodes", None),
        "nbType": ("nb_type", None),
        "MIN": ("bounds_min", None),
        "MAX": ("bounds_max", None),
        "NODES_PER_TYPE_OF_ELEMENT": ("nodes_per_type_of_element", None),
        "POINTS": ("points", 15),
        "LINES": ("lines", 1),
        "TRIANGLES": ("triangles", 2),
        "QUADS": ("quads", 3),
        "TETS": ("tets", 4),
        "HEXAS": ("hexa", 5),
        "PRISMS": ("prism", 6),
        "PYRAMIDS": ("pyramid", 7),
        "LINES3": ("lines2", 8),
        "TRIANGLES6": ("qtriangles", 9),
        "QUADS9": ("quads9", 10),
        "TETS10": ("tets10", 11),
        "HEXAS27": ("hexa27", 12),
        "PRISMS18": ("prism18", 13),
        "PYRAMIDS14": ("pyramid14", 14),
        "QUADS8": ("quads8", 16),
        "HEXAS20": ("hexa20", 17),
        "PRISMS15": ("prism15", 18),
        "PYRAMIDS13": ("pyramid13", 19),
    }
    _LEGACY_COUNT_MAP: ClassVar[dict[str, tuple[str, int]]] = {
        "nbPoints": ("points", 15),
        "nbLines": ("lines", 1),
        "nbTriangles": ("triangles", 2),
        "nbQuads": ("quads", 3),
        "nbTets": ("tets", 4),
        "nbHexas": ("hexa", 5),
        "nbPrisms": ("prism", 6),
        "nbPyramids": ("pyramid", 7),
        "nbLines3": ("lines2", 8),
        "nbTriangles6": ("qtriangles", 9),
        "nbQuads9": ("quads9", 10),
        "nbTets10": ("tets10", 11),
        "nbHexas27": ("hexa27", 12),
        "nbPrisms18": ("prism18", 13),
        "nbPyramids14": ("pyramid14", 14),
        "nbQuads8": ("quads8", 16),
        "nbHexas20": ("hexa20", 17),
        "nbPrisms15": ("prism15", 18),
        "nbPyramids13": ("pyramid13", 19),
    }
    _LEGACY_TYPES: ClassVar[tuple[tuple[int, int, str, str], ...]] = (
        (2, 1, "LINES", "nbLines"),
        (3, 2, "TRIANGLES", "nbTriangles"),
        (4, 2, "QUADS", "nbQuads"),
        (4, 3, "TETS", "nbTets"),
        (8, 3, "HEXAS", "nbHexas"),
        (6, 3, "PRISMS", "nbPrisms"),
        (5, 3, "PYRAMIDS", "nbPyramids"),
        (3, 1, "LINES3", "nbLines3"),
        (6, 2, "TRIANGLES6", "nbTriangles6"),
        (9, 2, "QUADS9", "nbQuads9"),
        (10, 3, "TETS10", "nbTets10"),
        (27, 3, "HEXAS27", "nbHexas27"),
        (18, 3, "PRISMS18", "nbPrisms18"),
        (14, 3, "PYRAMIDS14", "nbPyramids14"),
        (1, 0, "POINTS", "nbPoints"),
        (8, 2, "QUADS8", "nbQuads8"),
        (20, 3, "HEXAS20", "nbHexas20"),
        (15, 3, "PRISMS15", "nbPrisms15"),
        (13, 3, "PYRAMIDS13", "nbPyramids13"),
    )

    positions: np.ndarray
    element_infos: np.ndarray
    element_tags: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 0), dtype=int)
    )
    element_nodes: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 0), dtype=int)
    )
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
    legacy_element_infos: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 0), dtype=int)
    )
    legacy_element_tags: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 0), dtype=int)
    )
    loader_name: str = "load_gmsh2"
    explicit_types: frozenset[int] = field(
        default_factory=lambda: frozenset(range(1, 20))
    )
    nodes_per_type_of_element: np.ndarray = field(
        default_factory=lambda: np.array(
            [2, 3, 4, 4, 8, 6, 5, 3, 6, 9, 10, 27, 18, 14, 1, 8, 20, 15, 13],
            dtype=int,
        )
    )

    def property_numbers(
        self, element_refs: np.ndarray, *, info_column: int | None = None
    ) -> np.ndarray:
        refs = np.asarray(element_refs, dtype=int).ravel()
        if refs.size == 0:
            return np.zeros((0,), dtype=int)
        if info_column is None:
            if self.element_tags.shape[0] == 0:
                return np.zeros(refs.shape, dtype=int)
            return self.element_tags[refs - 1, 0]
        return self.element_infos[refs - 1, info_column]

    @property
    def Types(self) -> tuple[tuple[int, int, str, str], ...]:
        return self._LEGACY_TYPES

    @property
    def nbNod(self) -> int:
        return int(self.positions.shape[0])

    @property
    def nbElm(self) -> int:
        return int(self.element_infos.shape[0])

    def __getattr__(self, name: str):
        if name in self._LEGACY_FIELD_MAP:
            field_name, element_type = self._LEGACY_FIELD_MAP[name]
            if element_type is not None and element_type not in self.explicit_types:
                raise AttributeError(
                    f"{name} is not available for this mesh instance. "
                    "Use load_gmsh2(..., which=<types>) to request explicit type arrays."
                )
            if field_name == "legacy_element_infos" and self.legacy_element_infos.size == 0:
                return self.element_infos
            if field_name == "legacy_element_tags" and self.legacy_element_tags.size == 0:
                return self.element_tags
            return getattr(self, field_name)
        if name in self._LEGACY_COUNT_MAP:
            field_name, element_type = self._LEGACY_COUNT_MAP[name]
            if element_type not in self.explicit_types:
                raise AttributeError(
                    f"{name} is not available for this mesh instance. "
                    "Use load_gmsh2(..., which=<types>) to request explicit type arrays."
                )
            return int(getattr(self, field_name).shape[0])
        raise AttributeError(f"{type(self).__name__!s} has no attribute {name!r}")
