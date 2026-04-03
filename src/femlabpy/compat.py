from __future__ import annotations

import sys
from pathlib import Path


def setpath(*, append_examples: bool = True) -> dict[str, Path]:
    """
    Return canonical package paths for compatibility with legacy FemLab scripts.

    The original MATLAB helper added the ``examples`` directory to the MATLAB path.
    Python packages do not require that pattern, so this helper is intentionally
    conservative: it returns the resolved package and examples directories while
    appending the examples directory to ``sys.path`` by default, mirroring the
    classroom ``setpath.m`` helper as closely as makes sense in Python.
    """

    package_dir = Path(__file__).resolve().parent
    examples_dir = package_dir / "examples"
    if append_examples:
        example_path = str(examples_dir)
        if example_path not in sys.path:
            sys.path.append(example_path)
    return {"package": package_dir, "examples": examples_dir}


__all__ = ["setpath"]
