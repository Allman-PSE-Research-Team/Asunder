"""Asunder: Constrained structure detection on undirected graphs."""

from asunder.column_generation.master import solve_master_problem
from asunder.column_generation.subproblem import solve_subproblem
from asunder.config import CSDDecompositionConfig
from asunder.orchestrator import CSDDecomposition, run_csd_decomposition
from asunder.solvers import create_solver
from asunder.types import DecompositionResult, IterationRecord


def run_evaluation(*args, **kwargs):
    """Run benchmark evaluations using :mod:`asunder.evaluation.runner`.

    This lazy import keeps optional evaluation dependencies out of import-time
    paths for users who only need decomposition APIs.
    """
    from asunder.evaluation.runner import run_evaluation as _run_evaluation

    return _run_evaluation(*args, **kwargs)


__all__ = [
    "CSDDecomposition",
    "CSDDecompositionConfig",
    "DecompositionResult",
    "IterationRecord",
    "create_solver",
    "run_csd_decomposition",
    "run_evaluation",
    "solve_master_problem",
    "solve_subproblem",
]
