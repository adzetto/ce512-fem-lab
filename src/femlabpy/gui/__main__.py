"""Allow running the GUI with ``python -m femlabpy.gui``."""

from __future__ import annotations

import logging
import sys


def main() -> int:
    """
    Configure basic logging and delegate to the GUI application entry point.

    Returns
    -------
    int
        Process exit status returned by :mod:`femlabpy.gui.app`.
    """
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(name)-28s %(levelname)-5s %(message)s",
        datefmt="%H:%M:%S",
    )

    from .app import main as run_app

    return int(run_app())


if __name__ == "__main__":
    sys.exit(main())
