"""Core types and protocols for Asunder."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol

import numpy as np


class MasterProblemFn(Protocol):
    """Protocol for master-problem solvers used in decomposition loops."""

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
    """Protocol for pricing/subproblem solvers used in decomposition loops."""

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
    """Structured record for one decomposition iteration."""

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
    """Container for full decomposition output and summary metadata."""

    records: List[IterationRecord]
    final_partition: Optional[np.ndarray]
    final_master_obj: Optional[float]
    metadata: Dict[str, Any] = field(default_factory=dict)
