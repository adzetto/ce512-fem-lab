# Contributing

## Scope

This project is both a teaching codebase and a publishable Python package. New
contributions should preserve that balance:

- Keep the legacy MATLAB / Scilab naming recognizable where compatibility matters
- Prefer numerically verified changes over purely cosmetic refactors
- Update docstrings and README examples when the public API changes
- Add or update tests for every behavior change

## Local Setup

```bash
python -m venv .venv
. .venv/Scripts/activate
python -m pip install -e .[dev]
pytest -q
```

## Pull Request Checklist

- Make the smallest coherent change set you can defend
- Keep public function names and return shapes stable unless the change explicitly requires otherwise
- Add benchmark or regression coverage for solver behavior changes
- Update `README.md` when user-facing workflows or package metadata change
- Keep docstrings aligned with what `help()` should show to end users

## Coding Notes

- Prefer NumPy vectorization for element loops when it does not reduce clarity
- Avoid silent numerical behavior changes in plasticity and nonlinear solvers
- Preserve one-based topology semantics at the package boundary where legacy compatibility expects them
- Keep new files ASCII unless a file already uses Unicode intentionally

## Tests And Validation

Run the standard suite before opening a PR:

```bash
pytest -q
python -m build
python -m twine check dist/*
```

Useful comparison scripts:

```bash
python scripts/compare_ex_lag_mult.py
python scripts/generate_solver_comparison.py --solver python --solver scilab
```

## Reporting Problems

- Bugs: open a GitHub issue with a minimal reproducer, expected result, and actual result
- Numerical regressions: include the benchmark case name and the affected output files if possible
- Packaging problems: include Python version, platform, and install command

## Communication

By participating in this repository, you agree to follow the
[Code of Conduct](CODE_OF_CONDUCT.md).
