# Mesh I/O, Data Types, and Helpers

In computational mechanics, finite element analysis (FEA) heavily relies on robust and efficient representation of mesh data and sparse linear algebra operations. The `femlabpy` framework is designed to bridge the gap between traditional, legacy MATLAB implementations and modern, Pythonic object-oriented paradigms. 

This chapter provides a comprehensive overview of how mesh topologies are imported, the data structures used to store them, and the underlying mathematical and programmatic helpers that facilitate the assembly and solution of the resulting finite element systems.

---

## 1. The Gmsh File Format (`.msh`)

The open-source mesh generator **Gmsh** is the standard source of finite element meshes for `femlabpy`. Over time, Gmsh has evolved its `.msh` file format. `femlabpy` must accommodate both the legacy 2.2 format and the modern 4.1 format.

### 1.1 The Legacy v2.2 Format

The Gmsh v2.2 format is characterized by its simple ASCII block structure, which is easily parsed by line-by-line readers. The file is divided into distinct sections such as `$MeshFormat`, `$PhysicalNames`, `$Nodes`, and `$Elements`.

A typical v2.2 node section looks like this:
```text
$Nodes
number_of_nodes
node_number x-coord y-coord z-coord
...
$EndNodes
```
The coordinate vector for a node $i$ is denoted as $\mathbf{x}_i \in \mathbb{R}^3$.

The element section defines the topological connectivity:
```text
$Elements
number_of_elements
element_number element_type number_of_tags <tags> node_number_list
...
$EndElements
```
Here, the `element_type` corresponds to standardized integer IDs (e.g., `1` for lines, `2` for triangles, `3` for quadrilaterals, `4` for tetrahedrons). The `tags` usually encapsulate the physical and elementary regions, representing the material property IDs or boundary condition labels.

### 1.2 The Modern v4.1 Format

The modern Gmsh v4.1 format introduces block-based data storage, which significantly improves read/write performance for extremely large meshes. However, the block-based layout is incompatible with simple line-by-line procedural parsers written for v2.2.

To natively support v4.1 without rewriting the entire legacy logic, `femlabpy` employs a strategic workaround: if the Python `gmsh` SDK is installed, `femlabpy` loads the v4.1 mesh into the SDK's internal memory and temporarily re-emits it as a v2.2 legacy ASCII file. This guarantees backward compatibility and unified parsing behavior.

---

## 2. Parsing Meshes with `load_gmsh2`

The `load_gmsh2` parser in `femlabpy.io.gmsh` is responsible for translating the ASCII `.msh` text structure into structured numpy arrays. The parser systematically iterates over the `$Nodes` and `$Elements` blocks.

### 2.1 Extracting Coordinates

In the `$Nodes` block, the parser extracts $N \times 3$ coordinates by reading each line, converting the last three values into floats, and appending them to an array. The node IDs are used to ensure the array is correctly ordered.

Here is the exact Python code that extracts $N \times 3$ coordinates:

```python
with open(filename, 'r') as f:
    lines = f.readlines()

# ... find $Nodes index
n_nodes = int(lines[nodes_idx + 1])
positions = np.zeros((n_nodes, 3), dtype=np.float64)

for i in range(n_nodes):
    parts = lines[nodes_idx + 2 + i].split()
    node_id = int(parts[0]) - 1 # 0-indexed
    positions[node_id, 0] = float(parts[1])
    positions[node_id, 1] = float(parts[2])
    positions[node_id, 2] = float(parts[3])
```

### 2.2 Physical Groups and Topology Arrays

A critical step in FEA preprocessing is associating specific regions of the mesh with physical properties (e.g., Young's modulus, mass density). In Gmsh, these are defined via "Physical Groups".

When `load_gmsh2` processes an element row, it extracts the tags. In the v2.2 format, the first tag in the `<tags>` list corresponds to the Physical Group ID.

Here is the exact Python code that extracts the nodes and maps Physical Groups to the `prop_id` column in the `T` array:

```python
# During the element parsing loop for a specific element type (e.g. triangles)
T_list = []
for i in range(n_elements):
    parts = lines[elements_idx + 2 + i].split()
    elem_type = int(parts[1])
    num_tags = int(parts[2])
    
    if elem_type == 2:  # Triangle
        phys_grp_id = int(parts[3])  # First tag is Physical Group
        node_start = 3 + num_tags
        n1 = int(parts[node_start])
        n2 = int(parts[node_start + 1])
        n3 = int(parts[node_start + 2])
        # Append node IDs and the prop_id (Physical Group)
        T_list.append([n1, n2, n3, phys_grp_id])

T_triangles = np.array(T_list, dtype=np.int32)
```

This `prop_id` directly maps to the material property IDs in the topology array `T`. For example, in a simple 2D triangle element, `T` is a row matrix where the first 3 columns are the node numbers, and the last column is the material property ID:
$$ T_e = [n_1, n_2, n_3, \text{prop\_id}] $$

---

## 3. The `GmshMesh` Data Structure

To manage the parsed data, `femlabpy.types` implements the `GmshMesh` dataclass. This structure bridges the gap between modern Python attributes and the legacy MATLAB variables.

### 3.1 Dataclass Definition

The `GmshMesh` dataclass acts as a robust container for all the node coordinates and explicit topology matrices extracted from the `.msh` file.

```python
from dataclasses import dataclass
import numpy as np

@dataclass
class GmshMesh:
    """Dataclass holding finite element mesh arrays."""
    positions: np.ndarray          # Shape: (N, 3), node coordinates
    element_infos: np.ndarray      # General element information
    element_tags: np.ndarray       # Tag values
    element_nodes: np.ndarray      # Connectivity
    
    # Specific element topologies with prop_id
    points: np.ndarray = None      # 1-node
    lines: np.ndarray = None       # 2-node
    triangles: np.ndarray = None   # 3-node
    quads: np.ndarray = None       # 4-node
    tets: np.ndarray = None        # 4-node 3D
    hexa: np.ndarray = None        # 8-node 3D
```

### 3.2 Legacy Aliases

To maintain backward compatibility with scripts that expect legacy MATLAB variable names, `GmshMesh` implements custom property accessors. Thus, accessing `mesh.POS` returns `mesh.positions`, and `mesh.TRIANGLES` returns the specific connectivity array `mesh.triangles`.

---

## 4. End-to-End Example: Generating and Loading a Mesh

The following runnable script demonstrates how to generate a mesh programmatically using the `gmsh` Python SDK, save it in the v2.2 legacy format, and load it using `femlabpy`.

```python
import gmsh
import numpy as np
# Assume femlabpy is in path
# from femlabpy.io.gmsh import load_gmsh2

def generate_and_load_mesh():
    # 1. Initialize Gmsh
    gmsh.initialize()
    gmsh.model.add("Rectangle")

    # 2. Define Geometry
    L, H = 1.0, 0.5
    lc = 0.1 # Characteristic length
    gmsh.model.geo.addPoint(0, 0, 0, lc, 1)
    gmsh.model.geo.addPoint(L, 0, 0, lc, 2)
    gmsh.model.geo.addPoint(L, H, 0, lc, 3)
    gmsh.model.geo.addPoint(0, H, 0, lc, 4)

    gmsh.model.geo.addLine(1, 2, 1)
    gmsh.model.geo.addLine(2, 3, 2)
    gmsh.model.geo.addLine(3, 4, 3)
    gmsh.model.geo.addLine(4, 1, 4)

    gmsh.model.geo.addCurveLoop([1, 2, 3, 4], 1)
    gmsh.model.geo.addPlaneSurface([1], 1)

    # 3. Define Physical Groups (Material properties/BCs)
    # The physical group ID becomes the prop_id in the T array
    gmsh.model.geo.addPhysicalGroup(2, [1], 100) # Surface elements, tag=100
    gmsh.model.geo.synchronize()

    # 4. Generate and save mesh as v2.2
    gmsh.model.mesh.generate(2)
    gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
    filename = "test_mesh.msh"
    gmsh.write(filename)
    gmsh.finalize()

    # 5. Load with femlabpy (Simulated output for demonstration)
    print(f"Mesh generated and saved to {filename}")
    # mesh = load_gmsh2(filename)
    # print("Node coordinates shape:", mesh.positions.shape)
    # print("Triangles shape (nodes + prop_id):", mesh.triangles.shape)

if __name__ == "__main__":
    generate_and_load_mesh()
```

---

## 5. Summary

The infrastructure laid out in `femlabpy` encapsulates the raw strings of a `.msh` file into highly optimized, algebraically-ready array structures. By leveraging `GmshMesh` and its targeted parsers to map Physical Groups to `prop_id` columns, engineers can seamlessly transition from mesh generation to stiffness matrix assembly in a few lines of Python.
