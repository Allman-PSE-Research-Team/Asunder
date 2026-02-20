"""Solver factory helpers."""

from __future__ import annotations

import os
from typing import Any

_DEFAULT_SOLVER = None


def create_solver(solver_name: str = "gurobi_direct", **solver_kwargs: Any):
    """Create a pyomo solver instance.

    If `solver_name` starts with `gurobi` and no explicit options are supplied,
    this function relies on the `GRB_LICENSE_FILE` environment variable.
    """
    try:
        from pyomo.opt import SolverFactory
    except Exception as exc:  # pragma: no cover - optional dependency
        raise ImportError("pyomo is required to create solver instances. Ensure base dependencies are installed.") from exc

    if solver_name.startswith("gurobi") and "options" not in solver_kwargs:
        _ = os.environ.get("GRB_LICENSE_FILE")
    return SolverFactory(solver_name, **solver_kwargs)


def set_default_solver(solver: Any) -> None:
    """Set the process-wide default solver instance used by Asunder."""
    global _DEFAULT_SOLVER
    _DEFAULT_SOLVER = solver


def get_default_solver():
    """Return the configured default solver, creating one lazily if needed."""
    global _DEFAULT_SOLVER
    if _DEFAULT_SOLVER is None:
        _DEFAULT_SOLVER = create_solver()
    return _DEFAULT_SOLVER
