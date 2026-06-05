"""Asunder: Constrained structure detection on undirected graphs."""

from asunder.base.column_generation.master import solve_master_problem
from asunder.base.column_generation.subproblem import solve_subproblem
from asunder.config import CSDDecompositionConfig
from asunder.orchestrator import CSDDecomposition, run_csd_decomposition
from asunder.solvers import create_solver
from asunder.types import DecompositionResult, IterationRecord

__version__ = "0.2.1"

def run_evaluation(*args, **kwargs):
    """
    Run benchmark evaluations using :mod:`asunder.nlbnp.case_studies.runner`.
    
    Parameters
    ----------
    *args : Any
        Additional positional arguments.
    **kwargs : Any
        Additional keyword arguments.
    
    Returns
    -------
    Any
        Computed result.
    """
    from asunder.nlbnp.case_studies.runner import run_evaluation as _run_evaluation

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
