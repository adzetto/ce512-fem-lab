"""
FemLab Python - Finite Element Method Library.

A Python port of the legacy Scilab FemLab wrapper, derived from the original
MATLAB FemLab teaching toolbox by O. Hededal and S. Krenk at Aalborg University.
"""

__version__ = "0.6.0"
__author__ = "Muhammet Yagcioglu"

from .assembly import assmk, assmq
from .boundary import rnorm, setbc, solve_lag, solve_lag_general
from .compat import setpath
from .core import cols, init, rows
from .damping import modal_damping, rayleigh_coefficients, rayleigh_damping
from .dynamics import (
    NewmarkParams,
    TimeHistory,
    compute_frf,
    constant_load,
    critical_timestep,
    harmonic_load,
    plot_energy,
    plot_frf,
    plot_time_history,
    pulse_load,
    ramp_load,
    seismic_load,
    solve_central_diff,
    solve_hht,
    solve_newmark,
    solve_newmark_nl,
    tabulated_load,
)
from .elements import (
    kbar,
    kebar,
    keh8e,
    keq4e,
    keq4epe,
    keq4eps,
    keq4p,
    ket3e,
    ket3p,
    keT4e,
    kh8e,
    kq4e,
    kq4epe,
    kq4eps,
    kq4p,
    kt3e,
    kt3p,
    kT4e,
    mbar,
    mebar,
    meh8e,
    meq4e,
    met3e,
    meT4e,
    mh8e,
    mq4e,
    mt3e,
    mT4e,
    qbar,
    qebar,
    qeh8e,
    qeq4e,
    qeq4epe,
    qeq4eps,
    qeq4p,
    qet3e,
    qet3p,
    qeT4e,
    qh8e,
    qq4e,
    qq4epe,
    qq4eps,
    qq4p,
    qt3e,
    qt3p,
    qT4e,
)
from .io import load_gmsh, load_gmsh2
from .loads import addload, setload
from .materials import (
    devstres,
    devstress,
    dyieldvm,
    eqstress,
    stressdp,
    stressvm,
    yieldvm,
)
from .matlab import (
    bar01,
    bar02,
    bar03,
    canti,
    elastic,
    flow,
    flowq4,
    flowt3,
    hole,
    nlbar,
    plastpe,
    plastps,
    square,
)
from .modal import ModalResult, plot_modes, solve_modal
from .periodic import (
    apply_macro_strain,
    check_periodic_mesh,
    find_all_periodic_pairs,
    find_periodic_pairs,
    fix_corner,
    homogenize,
    periodic_constraints,
    solve_periodic,
    volume_average_strain,
    volume_average_stress,
)
from .plotting import plotbc, plotelem, plotforces, plotq4, plott3, plotu
from .postprocess import reaction
from .solvers import solve_nlbar, solve_plastic

__all__ = [
    # Assembly
    "addload",
    "assmk",
    "assmq",
    # Examples / data loaders
    "bar01",
    "bar02",
    "bar03",
    "canti",
    "cols",
    # Damping
    "compute_frf",
    "constant_load",
    "critical_timestep",
    # Materials
    "devstress",
    "devstres",
    "dyieldvm",
    "elastic",
    "eqstress",
    # Periodic BCs
    "apply_macro_strain",
    "check_periodic_mesh",
    "find_all_periodic_pairs",
    "find_periodic_pairs",
    "fix_corner",
    "flow",
    "flowq4",
    "flowt3",
    # Dynamics
    "harmonic_load",
    "hole",
    "homogenize",
    "init",
    # Stiffness
    "kT4e",
    "kbar",
    "kebar",
    "keT4e",
    "keh8e",
    "keq4e",
    "keq4epe",
    "keq4eps",
    "keq4p",
    "ket3e",
    "ket3p",
    "kh8e",
    "kq4e",
    "kq4epe",
    "kq4eps",
    "kq4p",
    "kt3e",
    "kt3p",
    "load_gmsh",
    "load_gmsh2",
    # Mass matrices
    "mT4e",
    "mbar",
    "mebar",
    "meh8e",
    "meT4e",
    "meq4e",
    "met3e",
    "mh8e",
    "modal_damping",
    "ModalResult",
    "mq4e",
    "mt3e",
    # Dynamics classes
    "NewmarkParams",
    "nlbar",
    # Periodic
    "periodic_constraints",
    "plastpe",
    "plastps",
    "plot_energy",
    "plot_frf",
    "plot_modes",
    "plot_time_history",
    "plotbc",
    "plotelem",
    "plotforces",
    "plotq4",
    "plott3",
    "plotu",
    "pulse_load",
    # Internal forces
    "qT4e",
    "qbar",
    "qebar",
    "qeT4e",
    "qeh8e",
    "qeq4e",
    "qeq4epe",
    "qeq4eps",
    "qeq4p",
    "qh8e",
    "qet3e",
    "qet3p",
    "qq4e",
    "qq4epe",
    "qq4eps",
    "qq4p",
    "qt3e",
    "qt3p",
    "ramp_load",
    "rayleigh_coefficients",
    "rayleigh_damping",
    "reaction",
    "rnorm",
    "rows",
    "seismic_load",
    "setpath",
    "setbc",
    "setload",
    "solve_central_diff",
    "solve_hht",
    "solve_lag",
    "solve_lag_general",
    "solve_modal",
    "solve_newmark",
    "solve_newmark_nl",
    "solve_nlbar",
    "solve_periodic",
    "solve_plastic",
    "square",
    "stressdp",
    "stressvm",
    "tabulated_load",
    "TimeHistory",
    "volume_average_strain",
    "volume_average_stress",
    "__version__",
    "yieldvm",
]


def get_version():
    """
    Return the installed ``femlabpy`` version string.

    Returns
    -------
    str
        Package version following :pep:`440`.
    """
    return __version__
