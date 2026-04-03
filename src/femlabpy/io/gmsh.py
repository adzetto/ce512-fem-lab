from __future__ import annotations

from pathlib import Path

import numpy as np

from ..types import GmshMesh


TYPE_DIMENSIONS = {
    1: 1,
    2: 2,
    3: 2,
    4: 3,
    5: 3,
    6: 3,
    7: 3,
    8: 1,
    9: 2,
    10: 2,
    11: 3,
    12: 3,
    13: 3,
    14: 3,
    15: 0,
    16: 2,
    17: 3,
    18: 3,
    19: 3,
}

TYPE_LAYOUTS = {
    15: ("points", 2),
    1: ("lines", 3),
    2: ("triangles", 4),
    3: ("quads", 5),
    4: ("tets", 5),
    5: ("hexa", 9),
    6: ("prism", 7),
    7: ("pyramid", 6),
    8: ("lines2", 4),
    9: ("qtriangles", 7),
    10: ("quads9", 10),
    11: ("tets10", 11),
    12: ("hexa27", 28),
    13: ("prism18", 19),
    14: ("pyramid14", 15),
    16: ("quads8", 9),
    17: ("hexa20", 21),
    18: ("prism15", 16),
    19: ("pyramid13", 14),
}


def _empty_array(width: int) -> np.ndarray:
    return np.zeros((0, width), dtype=int)


def _padded(rows: list[list[int]], width: int) -> np.ndarray:
    if not rows:
        return np.zeros((0, width), dtype=int)
    array = np.zeros((len(rows), width), dtype=int)
    for i, row in enumerate(rows):
        array[i, : len(row)] = row
    return array


def _parse_gmsh_file(filename) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, list[dict[str, object]]]:
    lines = Path(filename).read_text(encoding="utf-8", errors="ignore").splitlines()
    index = 0
    file_format = 1
    node_id_map: dict[int, int] = {}
    positions = np.zeros((0, 3), dtype=float)
    bounds_min = np.zeros(3, dtype=float)
    bounds_max = np.zeros(3, dtype=float)
    elements: list[dict[str, object]] = []

    while index < len(lines):
        line = lines[index].strip()
        if line == "$MeshFormat":
            file_format = 2
            index += 3
            continue

        if line == "$PhysicalNames":
            count = int(lines[index + 1].strip())
            index += count + 3
            continue

        if line in {"$Nodes", "$NOD"}:
            node_count = int(lines[index + 1].strip())
            coordinates = np.zeros((node_count, 3), dtype=float)
            for i in range(node_count):
                parts = lines[index + 2 + i].split()
                node_id = int(parts[0])
                xyz = np.asarray(parts[1:4], dtype=float)
                node_id_map[node_id] = i + 1
                coordinates[i] = xyz
            positions = coordinates
            if coordinates.size:
                bounds_min = coordinates.min(axis=0)
                bounds_max = coordinates.max(axis=0)
            end_marker = "$EndNodes" if line == "$Nodes" else "$ENDNOD"
            while index < len(lines) and lines[index].strip() != end_marker:
                index += 1
            index += 1
            continue

        if line in {"$Elements", "$ELM"}:
            element_count = int(lines[index + 1].strip())
            start = index + 2
            for row_number in range(1, element_count + 1):
                parts = [int(value) for value in lines[start + row_number - 1].split()]
                if file_format == 2 and line == "$Elements":
                    element_id = parts[0]
                    element_type = parts[1]
                    num_tags = parts[2]
                    tags = parts[3 : 3 + num_tags]
                    node_ids = parts[3 + num_tags :]
                    load_gmsh_info = [element_id, element_type, num_tags, tags[0] if tags else 0]
                    load_gmsh_tags = tags[1:]
                else:
                    element_id, element_type, reg_phys, reg_elem, num_nodes = parts[:5]
                    tags = [reg_phys, reg_elem]
                    node_ids = parts[5:]
                    load_gmsh_info = [element_id, element_type, reg_phys, reg_elem, num_nodes]
                    load_gmsh_tags = []

                mapped_nodes = [node_id_map[node_id] for node_id in node_ids]
                elements.append(
                    {
                        "row_number": row_number,
                        "id": element_id,
                        "type": element_type,
                        "tags": tags,
                        "nodes": mapped_nodes,
                        "dimension": TYPE_DIMENSIONS.get(element_type, 0),
                        "load_gmsh_info": load_gmsh_info,
                        "load_gmsh_tags": load_gmsh_tags,
                    }
                )

            end_marker = "$EndElements" if line == "$Elements" else "$ENDELM"
            while index < len(lines) and lines[index].strip() != end_marker:
                index += 1
            index += 1
            continue

        index += 1

    return positions, bounds_min, bounds_max, file_format, elements


def _build_normalized_mesh(
    *,
    positions: np.ndarray,
    bounds_min: np.ndarray,
    bounds_max: np.ndarray,
    elements: list[dict[str, object]],
    explicit_types: set[int],
    loader_name: str,
    legacy_infos: np.ndarray,
    legacy_tags: np.ndarray,
) -> GmshMesh:
    element_infos = np.zeros((len(elements), 4), dtype=int)
    tag_width = max((len(element["tags"]) for element in elements), default=0)
    node_width = max((len(element["nodes"]) for element in elements), default=0)
    element_tags = np.zeros((len(elements), tag_width), dtype=int)
    element_nodes = np.zeros((len(elements), node_width), dtype=int)
    nb_type = np.zeros(19, dtype=int)
    explicit_rows = {field_name: [] for field_name, _ in TYPE_LAYOUTS.values()}

    for i, element in enumerate(elements):
        element_id = int(element["id"])
        element_type = int(element["type"])
        tags = list(element["tags"])
        nodes = list(element["nodes"])
        element_infos[i] = [element_id, element_type, len(tags), TYPE_DIMENSIONS.get(element_type, 0)]
        if tags:
            element_tags[i, : len(tags)] = tags
        if nodes:
            element_nodes[i, : len(nodes)] = nodes
        if 1 <= element_type <= 19:
            nb_type[element_type - 1] += 1
        if element_type in explicit_types and element_type in TYPE_LAYOUTS:
            field_name, _ = TYPE_LAYOUTS[element_type]
            explicit_rows[field_name].append([*nodes, tags[0] if tags else 0])

    explicit_arrays = {}
    for element_type, (field_name, width) in TYPE_LAYOUTS.items():
        rows = explicit_rows[field_name] if element_type in explicit_types else []
        explicit_arrays[field_name] = _padded(rows, width)

    return GmshMesh(
        positions=positions,
        element_infos=element_infos,
        element_tags=element_tags,
        element_nodes=element_nodes,
        nb_type=nb_type,
        points=explicit_arrays["points"],
        lines=explicit_arrays["lines"],
        lines2=explicit_arrays["lines2"],
        triangles=explicit_arrays["triangles"],
        quads=explicit_arrays["quads"],
        tets=explicit_arrays["tets"],
        tets10=explicit_arrays["tets10"],
        hexa=explicit_arrays["hexa"],
        hexa20=explicit_arrays["hexa20"],
        hexa27=explicit_arrays["hexa27"],
        prism=explicit_arrays["prism"],
        prism15=explicit_arrays["prism15"],
        prism18=explicit_arrays["prism18"],
        pyramid=explicit_arrays["pyramid"],
        pyramid13=explicit_arrays["pyramid13"],
        pyramid14=explicit_arrays["pyramid14"],
        qtriangles=explicit_arrays["qtriangles"],
        quads8=explicit_arrays["quads8"],
        quads9=explicit_arrays["quads9"],
        bounds_min=bounds_min,
        bounds_max=bounds_max,
        legacy_element_infos=legacy_infos,
        legacy_element_tags=legacy_tags,
        loader_name=loader_name,
        explicit_types=frozenset(explicit_types),
    )


def load_gmsh(filename) -> GmshMesh:
    """
    Read a Gmsh mesh using the legacy ``load_gmsh.m`` semantics.

    The returned :class:`~femlabpy.types.GmshMesh` exposes the normalized Python
    fields used by ``femlabpy`` and the MATLAB-style aliases expected by the
    classroom toolbox, including explicit type arrays whose last column stores
    the first element tag.
    """

    positions, bounds_min, bounds_max, _, elements = _parse_gmsh_file(filename)
    explicit_types = set(TYPE_LAYOUTS)
    legacy_info_width = max((len(element["load_gmsh_info"]) for element in elements), default=0)
    legacy_tag_width = max((len(element["load_gmsh_tags"]) for element in elements), default=0)
    legacy_infos = _padded([list(element["load_gmsh_info"]) for element in elements], legacy_info_width)
    legacy_tags = _padded([list(element["load_gmsh_tags"]) for element in elements], legacy_tag_width)
    return _build_normalized_mesh(
        positions=positions,
        bounds_min=bounds_min,
        bounds_max=bounds_max,
        elements=elements,
        explicit_types=explicit_types,
        loader_name="load_gmsh",
        legacy_infos=legacy_infos,
        legacy_tags=legacy_tags,
    )


def load_gmsh2(filename, which=None) -> GmshMesh:
    """
    Read a Gmsh mesh using the more flexible ``load_gmsh2.m`` semantics.

    Parameters
    ----------
    filename:
        Path to an ASCII Gmsh ``.msh`` file.
    which:
        Optional iterable of element-type ids whose explicit arrays should be
        materialized. ``None`` loads all explicit arrays, while ``-1`` or an
        empty iterable reproduces the MATLAB behavior of skipping them.
    """

    positions, bounds_min, bounds_max, _, elements = _parse_gmsh_file(filename)
    if which is None:
        explicit_types = set(TYPE_LAYOUTS)
    else:
        requested = np.asarray(which, dtype=int).ravel()
        if requested.size == 0 or (requested.size == 1 and int(requested[0]) == -1):
            explicit_types = set()
        else:
            explicit_types = {int(value) for value in requested}

    legacy_infos = _padded(
        [
            [
                int(element["id"]),
                int(element["type"]),
                len(element["tags"]),
                int(element["dimension"]),
            ]
            for element in elements
        ],
        4,
    )
    legacy_tag_width = max((len(element["tags"]) for element in elements), default=0)
    legacy_tags = _padded([list(element["tags"]) for element in elements], legacy_tag_width)

    return _build_normalized_mesh(
        positions=positions,
        bounds_min=bounds_min,
        bounds_max=bounds_max,
        elements=elements,
        explicit_types=explicit_types,
        loader_name="load_gmsh2",
        legacy_infos=legacy_infos,
        legacy_tags=legacy_tags,
    )
