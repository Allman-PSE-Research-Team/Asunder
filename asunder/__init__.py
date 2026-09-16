"""Asunder: Constrained structure detection on undirected graphs."""

from asunder.base.algorithms.modular_VFD import refine_partition_modular_vfd
from asunder.base.column_generation.master import solve_master_problem
from asunder.base.column_generation.subproblem import solve_subproblem
from asunder.config import CSDDecompositionConfig
from asunder.orchestrator import CSDDecomposition, run_csd_decomposition
from asunder.solvers import create_solver
from asunder.types import DecompositionResult, IterationRecord, MatrixLike

__version__ = "0.3.0"

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
    "MatrixLike",
    "create_solver",
    "run_csd_decomposition",
    "run_evaluation",
    "refine_partition_modular_vfd",
    "solve_master_problem",
    "solve_subproblem",
]
