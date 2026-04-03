"""
CLI entry point for femlabpy.

Usage:
    python -m femlabpy --version
    python -m femlabpy --info
"""

import argparse
import sys

from . import __version__


def main():
    """
    Run the lightweight command-line interface for package metadata display.

    Returns
    -------
    int
        Process exit status.
    """
    parser = argparse.ArgumentParser(
        prog="femlabpy",
        description="FemLab Python - Finite Element Method Library",
    )
    parser.add_argument(
        "-V",
        "--version",
        action="version",
        version=f"femlabpy {__version__}",
    )
    parser.add_argument(
        "--info",
        action="store_true",
        help="Show package information and available functions",
    )

    args = parser.parse_args()

    if args.info:
        print(f"femlabpy v{__version__}")
        print("=" * 50)
        print("\nFinite Element Method library for structural analysis.")
        print("\nAvailable element types:")
        print("  - Bar elements:      kbar, qbar (truss)")
        print("  - Q4 elements:       kq4e, qq4e (plane stress/strain)")
        print("  - T3 elements:       kt3e, qt3e (CST triangle)")
        print("  - H8 elements:       kh8e, qh8e (3D hexahedral)")
        print("  - T4 elements:       kT4e, qT4e (3D tetrahedral)")
        print("\nCore functions:")
        print("  - init(nn, nd)       Initialize arrays")
        print("  - setbc(K, p, C, v)  Apply boundary conditions")
        print("  - setload(p, P)      Apply nodal loads")
        print("  - solve_lag(...)     Lagrange multiplier solver")
        print("\nPostprocessing:")
        print("  - reaction(K, u, p)  Compute reactions")
        print("  - plotq4, plott3     Mesh visualization")
        print("  - plotu              Displacement plot")
        print("\nExample:")
        print("  from femlabpy import init, kq4e, qq4e, setbc, setload")
        print("  K, q, p, C, P, S = init(nn=27, nd=2)")
        print("  K = kq4e(K, T, X, G)")
        print("  u = np.linalg.solve(K, p)")
        return 0

    # If no arguments, show help
    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
