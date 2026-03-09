from __future__ import annotations

from pathlib import Path

import numpy as np

from ..types import GmshMesh


NODES_PER_TYPE = {
    1: 2,
    2: 3,
    3: 4,
    4: 4,
    5: 8,
    6: 6,
    7: 5,
    8: 3,
    9: 6,
    10: 9,
    11: 10,
    12: 27,
    13: 18,
    14: 14,
    15: 1,
    16: 8,
    17: 20,
    18: 15,
    19: 13,
}


def _empty_array(width: int) -> np.ndarray:
    return np.zeros((0, width), dtype=int)


def load_gmsh(filename) -> GmshMesh:
    lines = Path(filename).read_text(encoding="utf-8", errors="ignore").splitlines()
    index = 0
    file_format = 1
    node_id_map: dict[int, int] = {}
    positions = np.zeros((0, 3), dtype=float)
    bounds_min = np.zeros(3, dtype=float)
    bounds_max = np.zeros(3, dtype=float)
    element_rows: list[list[int]] = []
    point_rows: list[list[int]] = []
    line_rows: list[list[int]] = []
    line2_rows: list[list[int]] = []
    triangle_rows: list[list[int]] = []
    quad_rows: list[list[int]] = []
    tet_rows: list[list[int]] = []
    tet10_rows: list[list[int]] = []
    hexa_rows: list[list[int]] = []
    hexa20_rows: list[list[int]] = []
    hexa27_rows: list[list[int]] = []
    prism_rows: list[list[int]] = []
    prism15_rows: list[list[int]] = []
    prism18_rows: list[list[int]] = []
    pyramid_rows: list[list[int]] = []
    pyramid13_rows: list[list[int]] = []
    pyramid14_rows: list[list[int]] = []
    qtriangle_rows: list[list[int]] = []
    quad8_rows: list[list[int]] = []
    quad9_rows: list[list[int]] = []

    while index < len(lines):
        line = lines[index].strip()
        if line == "$MeshFormat":
            file_format = 2
            index += 3
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
                    info_row = [element_id, element_type, num_tags, *tags]
                else:
                    element_id, element_type, reg_phys, reg_elem, num_nodes = parts[:5]
                    node_ids = parts[5:]
                    info_row = [element_id, element_type, reg_phys, reg_elem, num_nodes]
                element_rows.append(info_row)

                mapped_nodes = [node_id_map[node_id] for node_id in node_ids]
                record = [*mapped_nodes, row_number]
                if element_type == 15:
                    point_rows.append(record)
                elif element_type == 1:
                    line_rows.append(record)
                elif element_type == 2:
                    triangle_rows.append(record)
                elif element_type == 3:
                    quad_rows.append(record)
                elif element_type == 4:
                    tet_rows.append(record)
                elif element_type == 5:
                    hexa_rows.append(record)
                elif element_type == 6:
                    prism_rows.append(record)
                elif element_type == 7:
                    pyramid_rows.append(record)
                elif element_type == 8:
                    line2_rows.append(record)
                elif element_type == 9:
                    qtriangle_rows.append(record)
                elif element_type == 10:
                    quad9_rows.append(record)
                elif element_type == 11:
                    tet10_rows.append(record)
                elif element_type == 12:
                    hexa27_rows.append(record)
                elif element_type == 13:
                    prism18_rows.append(record)
                elif element_type == 14:
                    pyramid14_rows.append(record)
                elif element_type == 16:
                    quad8_rows.append(record)
                elif element_type == 17:
                    hexa20_rows.append(record)
                elif element_type == 18:
                    prism15_rows.append(record)
                elif element_type == 19:
                    pyramid13_rows.append(record)
            end_marker = "$EndElements" if line == "$Elements" else "$ENDELM"
            while index < len(lines) and lines[index].strip() != end_marker:
                index += 1
            index += 1
            continue
        index += 1

    width = max((len(row) for row in element_rows), default=0)
    element_infos = np.zeros((len(element_rows), width), dtype=int)
    for i, row in enumerate(element_rows):
        element_infos[i, : len(row)] = row

    return GmshMesh(
        positions=positions,
        element_infos=element_infos,
        points=np.asarray(point_rows, dtype=int) if point_rows else _empty_array(2),
        lines=np.asarray(line_rows, dtype=int) if line_rows else _empty_array(3),
        lines2=np.asarray(line2_rows, dtype=int) if line2_rows else _empty_array(4),
        triangles=np.asarray(triangle_rows, dtype=int) if triangle_rows else _empty_array(4),
        quads=np.asarray(quad_rows, dtype=int) if quad_rows else _empty_array(5),
        tets=np.asarray(tet_rows, dtype=int) if tet_rows else _empty_array(5),
        tets10=np.asarray(tet10_rows, dtype=int) if tet10_rows else _empty_array(11),
        hexa=np.asarray(hexa_rows, dtype=int) if hexa_rows else _empty_array(9),
        hexa20=np.asarray(hexa20_rows, dtype=int) if hexa20_rows else _empty_array(21),
        hexa27=np.asarray(hexa27_rows, dtype=int) if hexa27_rows else _empty_array(28),
        prism=np.asarray(prism_rows, dtype=int) if prism_rows else _empty_array(7),
        prism15=np.asarray(prism15_rows, dtype=int) if prism15_rows else _empty_array(16),
        prism18=np.asarray(prism18_rows, dtype=int) if prism18_rows else _empty_array(19),
        pyramid=np.asarray(pyramid_rows, dtype=int) if pyramid_rows else _empty_array(6),
        pyramid13=np.asarray(pyramid13_rows, dtype=int) if pyramid13_rows else _empty_array(14),
        pyramid14=np.asarray(pyramid14_rows, dtype=int) if pyramid14_rows else _empty_array(15),
        qtriangles=np.asarray(qtriangle_rows, dtype=int) if qtriangle_rows else _empty_array(7),
        quads8=np.asarray(quad8_rows, dtype=int) if quad8_rows else _empty_array(9),
        quads9=np.asarray(quad9_rows, dtype=int) if quad9_rows else _empty_array(10),
        bounds_min=bounds_min,
        bounds_max=bounds_max,
    )
