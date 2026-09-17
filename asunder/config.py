"""Configuration dataclasses."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Literal, Optional

from asunder.base.utils.matrix import (
    DEFAULT_MAX_DENSE_WORKING_BYTES,
    DEFAULT_SPARSE_COLUMN_DENSITY_THRESHOLD,
)


@dataclass
class CSDDecompositionConfig:
    """
    Configuration container for :class:`asunder.orchestrator.CSDDecomposition`.
    
    Attributes
    ----------
    columns : list[numpy.ndarray or scipy.sparse.csr_matrix] or None
        Existing binary co-association columns. This parameter is typically
        active during branch-and-price.
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
        Whether must-links are handled through graph contraction. Compatible
        cannot-links, initial-column constraints, warm starts, refinement
        constraints, and load-balance weights are mapped to contracted
        components automatically.
    stopping_window : int
        Maximum number of allowed stagnant CG iterations. After this, CG is terminated.
    check_flat_pricing : bool
        Boolean that determines whether to check for flat/stagnant pricing or not.
    algo : str
        Name of heuristic subproblem used to replace the ILP subproblem. Third-party algorithms combine adjacency and dual information into a unified input while custom algorithms treat adjacency and duals as separate inputs. Supported third-party algorithms are listed under the ``package`` parameter.
        Available custom algorithm options include:

        ``"spectral"``:
            Modified iterative bisection algorithm based on Mark Newman's eigenvector-based method.
        ``"full_louvain"``:
            Modified but Louvain-like algorithm.
        ``"RCCS"``:
            This means Reduced Cost Community Search and is a greedy and local search heuristic for finding communities that maximize the reduced cost.
    package : str or None
        Package from which non-custom heuristic subproblem is selected. Package and algorithm options include:

        ``"networkx"``:
            ``"louvain"``, ``"greedy"``, ``"girvan_newman"``
        ``"sknetwork"``:
            ``"louvain"``, ``"leiden"``, ``"lpa"``
        ``"igraph"``:
            ``"leiden"``, ``"greedy"``, ``"infomap"``, ``"lpa"``, ``"multilevel"``, ``"voronoi"``, ``"walktrap"``, ``"cpm_leiden"``
        ``"leidenalg"``:
            ``"leiden"``,  ``"signed_leiden"``, ``"cpm_leiden"``, ``"surprise_leiden"``, ``"signed_surprise_leiden"``
        ``None``:
            ``"signed_louvain"``, ``"spinglass"``

        Algorithms that start with ``"cpm"``, ``"signed"``, and ``"spinglass"`` are signed.
    resolution : float
        Modularity resolution parameter. Pricing algorithms that do not
        implement non-default resolution reject values other than ``1``.
    column_storage : {"auto", "dense", "csr"}
        Physical storage policy for binary co-association columns. Automatic mode
        preserves supplied dense/CSR representations, including mixed pools.
    sparse_column_density_threshold : float
        Maximum density for newly constructed automatic CSR columns; CSR must
        also have a smaller estimated footprint than dense Boolean storage.
    max_dense_working_bytes : int or None
        Maximum operation-specific estimate for package-created dense work
        arrays. Sparse-compatible operations retain CSR; dense-only boundaries
        raise ``MemoryError`` when the estimate is exceeded. ``None`` disables
        the guard, and existing dense caller input is not rejected merely
        because of its size.
    seed : int or None
        Random seed value.
    ifc_params : dict[str, callable or dict or int]
        Number of initial feasible columns (ifc), initial feasible column generator, and its corresponding arguments.
    refine_params : dict[str, callable or dict]
        Refinement function and its corresponding arguments.
    subproblem_params : dict[str, Any]
        Keyword arguments supplied only to the selected pricing/subproblem
        callable.
    use_refined_column : bool
        Boolean that determines whether refined columns are used in the main column generation loop or not.
    refine_post_loop : bool
        Boolean that determines whether post-loop refinement is run after column generation terminates.
    final_master_solve : bool
        Boolean that determines whether a final master solve is executed or not.
    max_iterations : int
        Maximum number of column generation iterations.
    disable_tqdm : bool
        Whether to disable the progress bar or not.
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
    algo: str = "signed_leiden"
    package: str = "leidenalg"
    seed: int | None = 42
    ifc_params: Dict[str, Any] = field(default_factory=dict)
    refine_params: Dict[str, Any] = field(default_factory=dict)
    subproblem_params: Dict[str, Any] = field(default_factory=dict)
    use_refined_column: bool = False
    refine_post_loop: bool = True
    final_master_solve: bool = True
    max_iterations: Optional[int] = 1000
    disable_tqdm: bool = False
    tolerance: float = 1e-10
    verbose: int | bool = 1
    resolution: float = 1.0
    column_storage: Literal["auto", "dense", "csr"] = "auto"
    sparse_column_density_threshold: float = DEFAULT_SPARSE_COLUMN_DENSITY_THRESHOLD
    max_dense_working_bytes: int | None = DEFAULT_MAX_DENSE_WORKING_BYTES
