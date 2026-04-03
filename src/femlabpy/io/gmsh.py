from __future__ import annotations

from importlib import import_module
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, TypedDict

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


class ParsedElement(TypedDict):
    """Normalized intermediate representation for one Gmsh element row."""

    row_number: int
    id: int
    type: int
    tags: list[int]
    nodes: list[int]
    dimension: int
    load_gmsh_info: list[int]
    load_gmsh_tags: list[int]


def _padded(rows: list[list[int]], width: int) -> np.ndarray:
    """
    Convert ragged integer rows to a zero-padded 2D array.

    Parameters
    ----------
    rows:
        Integer rows to pack.
    width:
        Target number of columns.

    Returns
    -------
    ndarray
        Zero-padded array with shape ``(len(rows), width)``.
    """
    if not rows:
        return np.zeros((0, width), dtype=int)
    array = np.zeros((len(rows), width), dtype=int)
    for i, row in enumerate(rows):
        array[i, : len(row)] = row
    return array


def _mesh_format_version(filename: str | Path) -> float | None:
    """
    Read the declared Gmsh mesh-file version from the header when available.

    Parameters
    ----------
    filename:
        Path to a ``.msh`` file.

    Returns
    -------
    float or None
        Mesh-format version such as ``2.2`` or ``4.1``. Returns ``None`` when
        the header cannot be parsed.
    """
    path = Path(filename)
    try:
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except OSError:
        return None

    for index, line in enumerate(lines):
        if line.strip() == "$MeshFormat" and index + 1 < len(lines):
            try:
                return float(lines[index + 1].split()[0])
            except (IndexError, ValueError):
                return None
    return None


def _import_gmsh_sdk() -> Any | None:
    """
    Import the optional official Gmsh Python SDK module.

    Returns
    -------
    module or None
        Imported ``gmsh`` module when available.
    """
    try:
        return import_module("gmsh")
    except ImportError:
        return None


def _legacy_view_path(filename: str | Path) -> tuple[Path, Any | None]:
    """
    Return a path guaranteed to follow the legacy Gmsh 2.2 ASCII layout.

    Modern Gmsh 4.x meshes use block-based node and element sections that do
    not match the original FemLab ``load_gmsh.m`` parser. When the optional
    official ``gmsh`` SDK is installed, this helper re-emits the mesh as an
    ASCII 2.2 file and returns the temporary path.

    Parameters
    ----------
    filename:
        Input mesh path.

    Returns
    -------
    tuple[pathlib.Path, TemporaryDirectory | None]
        Path to a mesh file readable by the legacy parser and an optional live
        temporary-directory handle that must be kept until the file is read.

    Raises
    ------
    ValueError
        If a Gmsh 4.x mesh is detected but the optional ``gmsh`` package is not
        installed.
    """
    mesh_version = _mesh_format_version(filename)
    if mesh_version is None or mesh_version < 4.0:
        return Path(filename), None

    gmsh = _import_gmsh_sdk()
    if gmsh is None:
        raise ValueError(
            "This mesh uses the Gmsh 4.x format. Install the optional mesh "
            'extra with `pip install "femlabpy[mesh]"` to enable automatic '
            "conversion through the official Gmsh SDK."
        )

    temp_dir = TemporaryDirectory()
    legacy_path = Path(temp_dir.name) / "legacy_v22_ascii.msh"
    initialized_here = not bool(gmsh.isInitialized())

    try:
        if initialized_here:
            gmsh.initialize(readConfigFiles=False)
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.option.setNumber("Mesh.Binary", 0)
        gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
        gmsh.open(str(filename))
        gmsh.write(str(legacy_path))
        gmsh.clear()
    except Exception:
        temp_dir.cleanup()
        raise
    finally:
        if initialized_here and bool(gmsh.isInitialized()):
            gmsh.finalize()

    return legacy_path, temp_dir


def _parse_gmsh_file(
    filename: str | Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, list[ParsedElement]]:
    """
    Parse a legacy-layout Gmsh mesh into normalized element dictionaries.

    Parameters
    ----------
    filename:
        Path to an ASCII Gmsh mesh following the legacy 2.2-style node and
        element layout.

    Returns
    -------
    tuple
        Tuple containing nodal coordinates, minimum and maximum bounds, legacy
        file-format flag, and a list of parsed element dictionaries.
    """
    legacy_path, temp_dir = _legacy_view_path(filename)
    try:
        lines = legacy_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    finally:
        if temp_dir is not None:
            temp_dir.cleanup()
    index = 0
    file_format = 1
    node_id_map: dict[int, int] = {}
    positions = np.zeros((0, 3), dtype=float)
    bounds_min = np.zeros(3, dtype=float)
    bounds_max = np.zeros(3, dtype=float)
    elements: list[ParsedElement] = []

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
            for row_number in range(node_count):
                node_parts = lines[index + 2 + row_number].split()
                node_id = int(node_parts[0])
                xyz = np.asarray(node_parts[1:4], dtype=float)
                node_id_map[node_id] = row_number + 1
                coordinates[row_number] = xyz
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
                element_parts = [
                    int(value) for value in lines[start + row_number - 1].split()
                ]
                if file_format == 2 and line == "$Elements":
                    element_id = element_parts[0]
                    element_type = element_parts[1]
                    num_tags = element_parts[2]
                    tags = element_parts[3 : 3 + num_tags]
                    node_ids = element_parts[3 + num_tags :]
                    load_gmsh_info = [
                        element_id,
                        element_type,
                        num_tags,
                        tags[0] if tags else 0,
                    ]
                    load_gmsh_tags = tags[1:]
                else:
                    (
                        element_id,
                        element_type,
                        reg_phys,
                        reg_elem,
                        num_nodes,
                    ) = element_parts[:5]
                    tags = [reg_phys, reg_elem]
                    node_ids = element_parts[5 : 5 + num_nodes]
                    load_gmsh_info = [
                        element_id,
                        element_type,
                        reg_phys,
                        reg_elem,
                        num_nodes,
                    ]
                    load_gmsh_tags = []

                mapped_nodes = [node_id_map[node_id] for node_id in node_ids]
                elements.append(
                    ParsedElement(
                        row_number=row_number,
                        id=element_id,
                        type=element_type,
                        tags=tags,
                        nodes=mapped_nodes,
                        dimension=TYPE_DIMENSIONS.get(element_type, 0),
                        load_gmsh_info=load_gmsh_info,
                        load_gmsh_tags=load_gmsh_tags,
                    )
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
    elements: list[ParsedElement],
    explicit_types: set[int],
    loader_name: str,
    legacy_infos: np.ndarray,
    legacy_tags: np.ndarray,
) -> GmshMesh:
    """
    Build the normalized :class:`~femlabpy.types.GmshMesh` container.

    Parameters
    ----------
    positions, bounds_min, bounds_max:
        Parsed nodal coordinates and bounding box.
    elements:
        Parsed element dictionaries.
    explicit_types:
        Element-type ids whose explicit topology tables should be materialized.
    loader_name:
        Legacy loader name used for provenance.
    legacy_infos, legacy_tags:
        MATLAB-compatible element-info tables.

    Returns
    -------
    GmshMesh
        Normalized mesh object with both Pythonic fields and legacy aliases.
    """
    element_infos = np.zeros((len(elements), 4), dtype=int)
    tag_width = max((len(element["tags"]) for element in elements), default=0)
    node_width = max((len(element["nodes"]) for element in elements), default=0)
    element_tags = np.zeros((len(elements), tag_width), dtype=int)
    element_nodes = np.zeros((len(elements), node_width), dtype=int)
    nb_type = np.zeros(19, dtype=int)
    explicit_rows: dict[str, list[list[int]]] = {
        field_name: [] for field_name, _ in TYPE_LAYOUTS.values()
    }

    for i, element in enumerate(elements):
        element_id = int(element["id"])
        element_type = int(element["type"])
        tags = list(element["tags"])
        nodes = list(element["nodes"])
        element_infos[i] = [
            element_id,
            element_type,
            len(tags),
            TYPE_DIMENSIONS.get(element_type, 0),
        ]
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

    Parameters
    ----------
    filename:
        Path to a Gmsh ``.msh`` file. Legacy 2.x ASCII files are parsed
        directly. Modern 4.x files are converted through the optional official
        ``gmsh`` SDK when the ``mesh`` extra is installed.

    Returns
    -------
    GmshMesh
        Normalized mesh object exposing both Python fields and legacy MATLAB
        aliases.

    Notes
    -----
    The returned :class:`~femlabpy.types.GmshMesh` stores explicit type arrays
    whose last column contains the first element tag, matching the original
    classroom loaders.
    """
    positions, bounds_min, bounds_max, _, elements = _parse_gmsh_file(filename)
    explicit_types = set(TYPE_LAYOUTS)
    legacy_info_width = max(
        (len(element["load_gmsh_info"]) for element in elements), default=0
    )
    legacy_tag_width = max(
        (len(element["load_gmsh_tags"]) for element in elements), default=0
    )
    legacy_infos = _padded(
        [list(element["load_gmsh_info"]) for element in elements],
        legacy_info_width,
    )
    legacy_tags = _padded(
        [list(element["load_gmsh_tags"]) for element in elements],
        legacy_tag_width,
    )
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
        Path to a Gmsh ``.msh`` file.
    which:
        Optional iterable of element-type ids whose explicit arrays should be
        materialized. ``None`` loads all explicit arrays, while ``-1`` or an
        empty iterable reproduces the MATLAB behavior of skipping them.

    Returns
    -------
    GmshMesh
        Normalized mesh object with optional explicit topology tables.
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
    legacy_tags = _padded(
        [list(element["tags"]) for element in elements], legacy_tag_width
    )

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
