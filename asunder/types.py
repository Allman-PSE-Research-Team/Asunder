"""Core types and protocols for Asunder."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol

import numpy as np


class MasterProblemFn(Protocol):
    """
    Protocol for master-problem solvers used in decomposition loops.

    Notes
    -----
    Implementations are expected to accept the current column pool and return
    either an integer-master solution tuple or an LP-with-duals tuple,
    depending on whether dual extraction is enabled.
    """

    def __call__(
        self,
        A: np.ndarray,
        a: np.ndarray,
        m: float,
        Z_star: List[np.ndarray],
        f_stars: List[float],
        **kwargs: Any,
    ) -> Any: ...


class SubproblemFn(Protocol):
    """
    Protocol for pricing/subproblem solvers used in decomposition loops.

    Notes
    -----
    Implementations are expected to solve or approximate the pricing problem
    and return ``(sub_obj_val, z_sol)``.
    """

    def __call__(
        self,
        A: np.ndarray,
        a: np.ndarray,
        m: float,
        duals: Dict[str, Any],
        **kwargs: Any,
    ) -> Any: ...


@dataclass
class IterationRecord:
    """
    Structured record for one decomposition iteration.
    
    Attributes
    ----------
    lambda_sol : list[float] or ndarray[float]
        A list/vector which sums to ``1`` that indicates what weight is assigned 
        to each column (and by implication, what columns are active).
    duals : Dict[str, ndarray[float] or float]
        The dual terms corresponding to a constraint in the relaxed master problem. 
        This could be a 2D array, 1D array or a float.
    master_obj_val : float
        The objective value of the master problem.
    z_sol : ndarray[int], shape (N, N)
        The most recently generated column from the subproblem.
    heuristic_col : ndarray[int], shape (N, N)
        Refined column generated using local search that is initialized 
        with ``z_sol`` or the convex combinations of all columns.
    sub_obj_val : float
        The reduced cost of the current column.
    columns : list[ndarray]
        All columns under consideration for the next iteration. The most recently 
        generated column is at index ``-1``.
    f_stars : list[float]
        List of objective values computed using each column in ``columns``.
    """

    lambda_sol: Optional[List[float]]
    duals: Dict[str, Any] = field(default_factory=dict)
    master_obj_val: Optional[float] = None
    z_sol: Optional[np.ndarray] = None
    heuristic_col: Optional[np.ndarray] = None
    sub_obj_val: Optional[float] = None
    columns: List[np.ndarray] = field(default_factory=list)
    f_stars: List[float] = field(default_factory=list)


@dataclass
class DecompositionResult:
    """
    Container for full decomposition output and summary metadata.
    
    Attributes
    ----------
    records : List[IterationRecord]
        Column generation iteration records.
    final_partition : Optional[np.ndarray]
        Partition matrix.
    final_master_obj : Optional[float]
        Final master problem objective.
    metadata : Dict[str, Any]
        Decomposition metadata.
    """

    records: List[IterationRecord]
    final_partition: Optional[np.ndarray]
    final_master_obj: Optional[float]
    metadata: Dict[str, Any] = field(default_factory=dict)
