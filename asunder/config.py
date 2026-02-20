"""Configuration dataclasses."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class CSDDecompositionConfig:
    """Configuration container for :class:`asunder.orchestrator.CSDDecomposition`.

    The fields mirror keyword arguments accepted by
    :func:`asunder.column_generation.decomposition.CSD_decomposition`.
    """

    columns: Optional[list] = None
    f_stars: Optional[list] = None
    must_link: list = field(default_factory=list)
    cannot_link: list = field(default_factory=list)
    additional_constraints: Dict[str, Any] = field(default_factory=dict)
    contract_graph: bool = False
    stopping_window: int = 5
    check_flat_pricing: bool = True
    algo: str = "louvain"
    package: str = "sknetwork"
    extract_dual: bool = False
    ifc_params: Dict[str, Any] = field(default_factory=dict)
    refine_in_subproblem: bool = False
    refine_params: Dict[str, Any] = field(default_factory=dict)
    use_refined_column: bool = False
    final_master_solve: bool = True
    max_iterations: Optional[int] = 1000
    tolerance: float = 1e-10
    verbosity: int = 1
