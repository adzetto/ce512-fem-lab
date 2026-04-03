"""
FemLab Python - Finite Element Method Library.

A Python port of the legacy Scilab FemLab wrapper, derived from the original
MATLAB FemLab teaching toolbox by O. Hededal and S. Krenk at Aalborg University.
"""

__version__ = "0.5.0"
__author__ = "Muhammet Yagcioglu"

from .assembly import assmk, assmq
from .boundary import rnorm, setbc, solve_lag, solve_lag_general
from .compat import setpath
from .core import cols, init, rows
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
from .plotting import plotbc, plotelem, plotforces, plotq4, plott3, plotu
from .postprocess import reaction
from .solvers import solve_nlbar, solve_plastic

__all__ = [
    "addload",
    "assmk",
    "assmq",
    "bar01",
    "bar02",
    "bar03",
    "canti",
    "cols",
    "devstress",
    "devstres",
    "dyieldvm",
    "elastic",
    "eqstress",
    "flow",
    "flowq4",
    "flowt3",
    "hole",
    "init",
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
    "plotbc",
    "plotelem",
    "plotforces",
    "plotq4",
    "plott3",
    "plotu",
    "plastpe",
    "plastps",
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
    "reaction",
    "rnorm",
    "rows",
    "setpath",
    "setbc",
    "setload",
    "square",
    "solve_lag",
    "solve_lag_general",
    "nlbar",
    "solve_nlbar",
    "solve_plastic",
    "stressdp",
    "stressvm",
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
