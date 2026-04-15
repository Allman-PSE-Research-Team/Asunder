"""Configuration dataclasses."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class CSDDecompositionConfig:
    """
    Configuration container for :class:`asunder.orchestrator.CSDDecomposition`.
    
    Attributes
    ----------
    columns : list[ndarray of int] or None
        Existing columns. This parameter is typically active during Branch and Price.
    f_stars : list[float] or None
        Objective values of the existing columns. 
        This parameter is typically active during Branch and Price.
    must_link : list[tuple[int, int]]
        List of node pairs that must be together.
    cannot_link : list[tuple[int, int]]
        List of node pairs that must not be together.
    additional_constraints : dict[str, Any]
        Constraints beyond must- and cannot-links. For example, worthy edges (edges that can connect communities), community size, and balance constraints.
    contract_graph : bool
        Boolean that determines whether must links are handled via graph contraction or not.
    stopping_window : int
        Maximum number of allowed stangnant CG iterations. After this, CG is terminated.
    check_flat_pricing : bool
        Boolean that determines whether to check for flat/stagnant pricing or not.
    algo : str
        Name of heuristic subproblem used to replace the ILP subproblem. Third-party algorithms combine adjacency and dual information intro a unified input while custom algorithms treat adjacency and duals as separate inputs. Supported third-party algorithms are listed under the ``package`` parameter.
        Available custom algorithm options include:

        ``"spectral"``:
            Modified iterative bisection algorithm based on Mark Newman's eigenvector-based method.
        ``"full_louvain"``:
            Modified but Louvain-like algorithm.
        ``"RCCS"``:
            This means Reduced Cost Community Search and is a greedy and local search heuristic for finding commiunities that maximize the reduced cost.
    package : str or None
        Package from which non-custom heuristic subproblem is selected. Package and algorithm options include:

        ``"networkx"``:
            ``"louvain"``, ``"greedy"``, ``"girvan_newman"``
        ``"sknetwork"``:
            ``"louvain"``, ``"leiden"``, ``"lpa"``
        ``"igraph"``:
            ``"leiden"``, ``"greedy"``, ``"infomap"``, ``"lpa"``, ``"multilevel"``, ``"voronoi"``, ``"walktrap"``
        ``leidenalg``:
            ``"leiden"``
        ``None``:
            ``"signed_louvain"``, ``"spinglass"``
    extract_dual : bool
        Boolean that determines whether we extract duals from the master problem or not.
    ifc_params : dict[str, callable or dict or int]
        Number of initial feasible columns (ifc), initial feasible column generator, and its corresponding arguments.
    refine_in_subproblem : bool
        Boolean value that determines whether a refinement operation is run in the subproblem.
    refine_params : dict[str, callable or dict]
        Refinement function and its corresponding arguments.
    use_refined_column : bool
        Boolean that determines whether refined columns are used in the main column generation loop or not.
    final_master_solve : bool
        Boolean that determines whether a final master solve is executed or not.
    max_iterations : int
        Maximum number of column generation iterations.
    tolerance : float
        Tolerance value for terminating column generation.
    verbose : int or bool
        Controls the level of detail in the printed output.
        ``-1``: No output
        ``False`` | ``0``: Minimal output
        ``True`` | ``1``: Detailed output
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
    verbose: int | bool = 1
