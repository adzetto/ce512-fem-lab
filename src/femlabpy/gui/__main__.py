"""Allow running the GUI with ``python -m femlabpy.gui``."""

import logging

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(name)-28s %(levelname)-5s %(message)s",
    datefmt="%H:%M:%S",
)

from .app import main

main()
