from .cantilever import cantilever_data, run_cantilever
from .ex_lag_mult import ex_lag_mult_data, run_ex_lag_mult
from .flow import flow_data, run_flow_q4, run_flow_t3
from .gmsh_triangle import gmsh_triangle_data, run_gmsh_triangle
from .legacy_cases import (
    bar01_data,
    bar02_data,
    bar03_data,
    hole_data,
    run_bar01_nlbar,
    run_bar02_nlbar,
    run_bar03_nlbar,
    run_hole_plastpe,
    run_hole_plastps,
    run_square_plastpe,
    run_square_plastps,
    square_data,
)

__all__ = [
    "bar01_data",
    "bar02_data",
    "bar03_data",
    "cantilever_data",
    "ex_lag_mult_data",
    "flow_data",
    "gmsh_triangle_data",
    "hole_data",
    "run_bar01_nlbar",
    "run_bar02_nlbar",
    "run_bar03_nlbar",
    "run_cantilever",
    "run_ex_lag_mult",
    "run_flow_q4",
    "run_flow_t3",
    "run_gmsh_triangle",
    "run_hole_plastpe",
    "run_hole_plastps",
    "run_square_plastpe",
    "run_square_plastps",
    "square_data",
]
