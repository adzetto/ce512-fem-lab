from .cantilever import cantilever_data, run_cantilever
from .flow import flow_data, run_flow_q4, run_flow_t3
from .gmsh_triangle import gmsh_triangle_data, run_gmsh_triangle

__all__ = [
    "cantilever_data",
    "flow_data",
    "gmsh_triangle_data",
    "run_cantilever",
    "run_flow_q4",
    "run_flow_t3",
    "run_gmsh_triangle",
]
