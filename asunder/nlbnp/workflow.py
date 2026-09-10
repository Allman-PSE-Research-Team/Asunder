"""High-level nonlinear branch-and-price application workflows."""

from __future__ import annotations

import copy
import inspect
import time
from collections.abc import Hashable, Sequence
from typing import Any, Literal

import networkx as nx
import numpy as np

from asunder.base.algorithms.core_periphery import (
    CorePeripheryTarget,
    partition_periphery_components,
)
from asunder.base.column_generation.master import compute_f_star, solve_master_problem
from asunder.base.column_generation.subproblem import (
    custom_heuristic_subproblem,
    heuristic_subproblem,
)
from asunder.base.utils.graph import (
    expand_z_matrix,
    group_nodes_by_community,
    map_community_labels,
    normalize_node_pairs,
    partition_satisfies_pairwise_constraints,
    validate_partition_matrix,
)
from asunder.base.utils.partition_generation import make_partitions_random_links_only
from asunder.config import CSDDecompositionConfig
from asunder.nlbnp.algorithms.core_periphery import (
    _detect_core_periphery,
    _nlbnp_linear_only_mask,
)
from asunder.nlbnp.algorithms.linear_group import (
    LinearGroupFeasibility,
    compute_max_feasible_linear_group,
    linear_only_communities,
    partition_satisfies_edge_constraints,
    partition_satisfies_nlbnp_cardinality,
    unworthy_edges,
)
from asunder.nlbnp.algorithms.refinement import (
    refine_partition_linear_group,
    refine_partition_with_cp,
)
from asunder.orchestrator import run_csd_decomposition
from asunder.types import DecompositionResult, MasterProblemFn, SubproblemFn

_CUSTOM_HEURISTIC_ALGOS = {"spectral", "full_louvain", "RCCS"}
_CARDINALITY_METHODS = {"reformulated", "confidence", "core_periphery"}


def _items_or_empty(items: Sequence[Any] | None) -> list[Any]:
    return [] if items is None else list(items)


def _coerce_graph_input(graph: nx.Graph | np.ndarray) -> tuple[np.ndarray, list[Hashable], nx.Graph | None]:
    if isinstance(graph, nx.Graph):
        node_labels = list(graph.nodes())
        return nx.to_numpy_array(graph, nodelist=node_labels), node_labels, graph

    A = np.asarray(graph, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("graph must be a networkx.Graph or a square adjacency matrix.")
    return A, list(range(A.shape[0])), None


def _map_pairs(
    pairs: Sequence[tuple[Hashable, Hashable]] | None,
    label_node_map: dict[Hashable, int],
    *,
    name: str,
) -> list[tuple[int, int]]:
    if pairs is None:
        return []

    mapped = []
    for source, target in pairs:
        try:
            mapped.append((label_node_map[source], label_node_map[target]))
        except KeyError as exc:
            raise ValueError(f"{name} contains a node label that is not present in the graph: {exc.args[0]!r}") from exc
    return mapped


def _edge_pairs_from_attribute(
    graph: nx.Graph | None,
    edge_attr: str | None,
    edge_value: Any,
    *,
    name: str,
) -> list[tuple[Hashable, Hashable]]:
    if edge_attr is None:
        return []
    if graph is None:
        raise ValueError(f"{name} can only be used when graph is a networkx.Graph.")

    edges = []
    for source, target, attrs in graph.edges(data=True):
        attr_value = attrs.get(edge_attr)
        if (edge_value is None and attr_value) or (edge_value is not None and attr_value == edge_value):
            edges.append((source, target))
    return edges


def _nodes_from_attribute(
    graph: nx.Graph | None,
    node_attr: str | None,
    node_value: Any,
    *,
    name: str,
) -> list[Hashable]:
    if node_attr is None:
        return []
    if graph is None:
        raise ValueError(f"{name} can only be used when graph is a networkx.Graph.")

    nodes = []
    for node, attrs in graph.nodes(data=True):
        attr_value = attrs.get(node_attr)
        if (node_value is None and attr_value) or (node_value is not None and attr_value == node_value):
            nodes.append(node)
    return nodes


def _map_nodes(
    nodes: Sequence[Hashable] | None,
    label_node_map: dict[Hashable, int],
    *,
    name: str,
) -> list[int]:
    if nodes is None:
        return []

    mapped = []
    for node in nodes:
        try:
            mapped.append(label_node_map[node])
        except KeyError as exc:
            raise ValueError(f"{name} contains a node label that is not present in the graph: {exc.args[0]!r}") from exc
    return list(dict.fromkeys(mapped))


def _unique_pairs(pairs: Sequence[tuple[int, int]]) -> list[tuple[int, int]]:
    seen = set()
    unique = []
    for source, target in pairs:
        pair = (int(source), int(target))
        key = tuple(sorted(pair))
        if key in seen:
            continue
        seen.add(key)
        unique.append(pair)
    return unique


def _infeasible_result(reason: str, **metadata: Any) -> DecompositionResult:
    return DecompositionResult(
        records=[],
        final_partition=None,
        final_master_obj=None,
        metadata={"status": "infeasible", "infeasible_reason": reason, **metadata},
    )


def _expanded_candidate(
    partition: np.ndarray,
    *,
    n_nodes: int,
    node2comp: np.ndarray | None,
) -> np.ndarray | None:
    matrix = np.asarray(partition)
    if matrix.shape == (n_nodes, n_nodes):
        return matrix
    if node2comp is None:
        return None
    n_components = int(node2comp.max()) + 1 if node2comp.size else 0
    if matrix.shape != (n_components, n_components):
        return None
    return expand_z_matrix(matrix, node2comp)


def _select_hard_partition(
    result: DecompositionResult,
    A: np.ndarray,
    *,
    must_link: Sequence[tuple[int, int]],
    cannot_link: Sequence[tuple[int, int]],
    worthy_edges: Sequence[tuple[int, int]],
    resolution: float,
) -> tuple[np.ndarray | None, str | None, float | None]:
    """Select the best available integral column satisfying NLBNP rules.

    A partition selected by the final integer master is authoritative. Without
    one, stored column scores order the latest generated pool so validation can
    stop at the first feasible candidate. Scores are recomputed only when a
    record does not contain one score per column.

    Parameters
    ----------
    result : DecompositionResult
        Result returned by the Stage 1 decomposition.
    A : ndarray of float, shape (N, N)
        Original adjacency matrix.
    must_link, cannot_link : sequence of tuple of int
        Active pairwise constraints.
    worthy_edges : sequence of tuple of int
        Nonempty structural edges allowed to cross communities.
    resolution : float
        Modularity resolution used only when stored column scores are absent.

    Returns
    -------
    partition : ndarray of int, shape (N, N), or None
        Selected hard partition, or ``None`` when no candidate is feasible.
    source : str or None
        Description of how the partition was selected.
    score : float or None
        Modularity score associated with the selected partition.
    """

    n_nodes = A.shape[0]
    node2comp_value = result.metadata.get("node2comp")
    node2comp = (
        None
        if node2comp_value is None
        else np.asarray(node2comp_value, dtype=int)
    )

    def validated(partition: np.ndarray | None) -> np.ndarray | None:
        if partition is None:
            return None
        expanded = _expanded_candidate(
            partition,
            n_nodes=n_nodes,
            node2comp=node2comp,
        )
        if expanded is None:
            return None
        try:
            candidate = validate_partition_matrix(
                expanded,
                n_nodes,
                name="NLBNP final candidate",
            )
        except ValueError:
            return None
        if not partition_satisfies_pairwise_constraints(
            candidate,
            must_link=must_link,
            cannot_link=cannot_link,
        ):
            return None
        if not partition_satisfies_edge_constraints(A, candidate, worthy_edges):
            return None
        return candidate

    selected = validated(result.final_partition)
    selected_source = result.metadata.get("final_partition_source")
    if selected is not None and selected_source in {
        "integer_master",
        "contracted_trivial",
    }:
        return (
            selected,
            selected_source,
            compute_f_star(
                A,
                A.sum(axis=1),
                float(A.sum()),
                selected,
                gamma=resolution,
            ),
        )

    pool: list[np.ndarray] = []
    stored_scores: list[float] = []
    for record in reversed(result.records):
        if record.columns:
            pool = record.columns
            stored_scores = record.f_stars
            break
    if len(stored_scores) == len(pool) and all(
        np.isfinite(score) for score in stored_scores
    ):
        order = np.argsort(np.asarray(stored_scores, dtype=float))[::-1]
        for index in order:
            candidate = validated(pool[int(index)])
            if candidate is not None:
                return (
                    candidate,
                    "best_feasible_generated_column",
                    float(stored_scores[int(index)]),
                )
    else:
        best_partition = None
        best_score = -np.inf
        for column in pool:
            candidate = validated(column)
            if candidate is None:
                continue
            score = compute_f_star(
                A,
                A.sum(axis=1),
                float(A.sum()),
                candidate,
                gamma=resolution,
            )
            if score > best_score:
                best_partition = candidate
                best_score = score
        if best_partition is not None:
            return (
                best_partition,
                "best_feasible_generated_column",
                float(best_score),
            )

    if selected is None:
        return None, None, None
    return (
        selected,
        selected_source or "decomposition",
        compute_f_star(
            A,
            A.sum(axis=1),
            float(A.sum()),
            selected,
            gamma=resolution,
        ),
    )


def _result_identity_metadata(
    result: DecompositionResult,
    *,
    node_label_map: dict[int, Hashable],
) -> None:
    if result.final_partition is None:
        return
    community_map, communities = group_nodes_by_community(result.final_partition)
    result.metadata["community_map"] = community_map
    result.metadata["community_map_labels"] = map_community_labels(
        community_map,
        node_label_map,
    )
    result.metadata["communities"] = communities


def run_nonlinear_branch_and_price(
    graph: nx.Graph | np.ndarray,
    *,
    worthy_edges: Sequence[tuple[Hashable, Hashable]] | None = None,
    worthy_edge_attr: str | None = None,
    worthy_edge_value: Any = None,
    must_link: Sequence[tuple[Hashable, Hashable]] | None = None,
    cannot_link: Sequence[tuple[Hashable, Hashable]] | None = None,
    nonlinear_nodes: Sequence[Hashable] | None = None,
    nonlinear_node_attr: str | None = None,
    nonlinear_node_value: Any = None,
    cardinality_method: Literal[
        "reformulated", "confidence", "core_periphery"
    ] = "reformulated",
    cardinality_params: dict[str, Any] | None = None,
    algorithm: str = "signed_leiden",
    package: str | None = "leidenalg",
    resolution: float = 1.0,
    seed: int | None = 42,
    ifc_params: dict[str, Any] | None = None,
    refine_params: dict[str, Any] | None = None,
    prob_method: str = "threshold",
    use_refined_column: bool = False,
    refine_post_loop: bool = False,
    final_master_solve: bool = False,
    check_flat_pricing: bool = True,
    stopping_window: int = 5,
    contract_graph: bool | None = None,
    max_iterations: int | None = None,
    tolerance: float = 1e-8,
    disable_tqdm: bool = False,
    verbose: int | bool = -1,
    additional_constraints: dict[str, Any] | None = None,
    config: CSDDecompositionConfig | None = None,
    master_fn: MasterProblemFn = solve_master_problem,
    subproblem_fn: SubproblemFn | None = None,
    **overrides: Any,
) -> DecompositionResult:
    """
    Run a generic nonlinear branch-and-price decomposition workflow.

    Parameters
    ----------
    graph : networkx.Graph or ndarray
        Input graph or square adjacency matrix.
    worthy_edges : sequence of tuple, optional
        Nonempty edge pairs that should be treated as worthy edges. For ``networkx``
        inputs, pairs use graph node labels. For adjacency inputs, pairs use
        integer node indices. Every pair must be a nonzero structural edge in
        the graph. NLBNP requires at least one worthy edge for every
        cardinality method.
    worthy_edge_attr : str, optional
        Edge attribute used to derive worthy edges from a ``networkx.Graph``.
        When ``worthy_edge_value`` is ``None``, truthy attribute values are
        selected. Otherwise, the attribute must equal ``worthy_edge_value``.
    worthy_edge_value : Any, optional
        Attribute value selected by ``worthy_edge_attr``.
    must_link, cannot_link : sequence of tuple, optional
        Pairwise constraints using graph node labels or adjacency indices.
    nonlinear_nodes : sequence, optional
        Nodes governed by nonlinear constraints. A valid linear-only community
        contains none of these nodes. Required for every cardinality method.
    nonlinear_node_attr : str, optional
        NetworkX node attribute used to derive ``nonlinear_nodes``. Truthy
        values are selected when ``nonlinear_node_value`` is ``None``.
    nonlinear_node_value : Any, optional
        Attribute value selected by ``nonlinear_node_attr``.
    cardinality_method : {"reformulated", "confidence", "core_periphery"}
        How the one-linear-only-community rule is enforced.
        ``"reformulated"``:
            Computes the exact maximum eligible set and reduces it to pairwise
        constraints. 
        ``"confidence"``:
            Clusters assignment confidence scores.
        ``"core_periphery"``
            Detects the linear-only side structurally. The latter two refine 
            a hard feasible generated column after column generation and 
            are heuristic.
    cardinality_params : dict, optional
        Extra keyword arguments for the selected confidence or core-periphery
        refiner. Ignored by ``"reformulated"``.
    algorithm : str
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
    resolution : float, default=1.0
        Modularity resolution used by pricing and column scoring. Algorithms
        that cannot apply non-default resolution reject it.
    seed : int or None
        Random seed.
    ifc_params : dict or None
        Initial feasible column generator configuration. Defaults to
        one deterministic DSATUR-colored pairwise-feasible partition.
    refine_params : dict or None
        Optional Stage 1 column-refinement configuration passed unchanged to
        :func:`run_csd_decomposition`. It is independent of cardinality
        enforcement. Supply ``refine_func`` and optional ``kwargs``.
    prob_method : str
        Probability-to-label method passed to the default linear-group
        refinement function.
    use_refined_column : bool
        Whether Stage 1 adds columns produced by ``refine_params`` inside the
        column-generation loop. Requires a callable ``refine_func``.
    refine_post_loop : bool
        Whether Stage 1 applies the configured refiner to the final fractional
        co-association matrix. Requires a callable ``refine_func``. This is
        independent of the Stage 2 confidence and core-periphery refiners.
    final_master_solve : bool
        Whether to run a final integer master solve.
    check_flat_pricing : bool
        Whether to terminate after a window of stagnant reduced costs.
    stopping_window : int
        Number of reduced costs used by the flat-pricing test.
    contract_graph : bool or None
        Whether to contract must-link and unworthy-edge components before
        decomposition. ``None`` enables contraction automatically for the
        exact reformulated method and disables it for heuristic methods.
    max_iterations : int or None
        Maximum column-generation iterations.
    tolerance : float
        Reduced-cost stopping tolerance.
    disable_tqdm : bool
        Disable the progress bar.
    verbose : int or bool
        Verbosity passed to the decomposition loop.
    additional_constraints : dict or None
        Additional constraints passed through to the master problem.
    config : CSDDecompositionConfig or None
        Base decomposition configuration. Explicit wrapper arguments override
        matching fields.
    master_fn : callable
        Master problem callable.
    subproblem_fn : callable or None
        Pricing/subproblem callable. Defaults to the appropriate built-in
        heuristic callable for ``algorithm``.
    **overrides : Any
        Additional configuration overrides passed to ``run_csd_decomposition``.

    Returns
    -------
    DecompositionResult
        Structured decomposition result with label-aware metadata. Exact
        infeasibility and heuristic cardinality failure are reported with
        ``final_partition=None`` and a descriptive metadata status/reason.
    """
    A, node_labels, nx_graph = _coerce_graph_input(graph)
    label_node_map = {label: idx for idx, label in enumerate(node_labels)}
    node_label_map = {idx: label for idx, label in enumerate(node_labels)}
    if cardinality_method not in _CARDINALITY_METHODS:
        choices = ", ".join(sorted(_CARDINALITY_METHODS))
        raise ValueError(f"cardinality_method must be one of {choices}.")

    user_must_link = normalize_node_pairs(
        _map_pairs(must_link, label_node_map, name="must_link"),
        A.shape[0],
        relation_name="must-link",
    )
    user_cannot_link = normalize_node_pairs(
        _map_pairs(cannot_link, label_node_map, name="cannot_link"),
        A.shape[0],
        relation_name="cannot-link",
        reject_self=True,
    )
    attr_nodes = _nodes_from_attribute(
        nx_graph,
        nonlinear_node_attr,
        nonlinear_node_value,
        name="nonlinear_node_attr",
    )
    nonlinear_idx = _map_nodes(
        [*_items_or_empty(nonlinear_nodes), *attr_nodes],
        label_node_map,
        name="nonlinear_nodes",
    )
    if not nonlinear_idx:
        raise ValueError(
            "nonlinear_nodes (or nonlinear_node_attr) must identify at least "
            "one node so the linear-only cardinality constraint is defined."
        )

    attr_edges = _edge_pairs_from_attribute(
        nx_graph,
        worthy_edge_attr,
        worthy_edge_value,
        name="worthy_edge_attr",
    )
    explicit_worthy_edges = worthy_edges is not None or worthy_edge_attr is not None
    worthy_edge_idx = _unique_pairs(
        _map_pairs(
            [*_items_or_empty(worthy_edges), *attr_edges],
            label_node_map,
            name="worthy_edges",
        )
    )

    cfg = copy.deepcopy(config) if config is not None else CSDDecompositionConfig()
    constraints = copy.deepcopy(cfg.additional_constraints)
    if additional_constraints:
        constraints.update(copy.deepcopy(additional_constraints))
    if explicit_worthy_edges:
        constraints["worthy_edges"] = worthy_edge_idx
    if "worthy_edges" not in constraints or constraints["worthy_edges"] is None:
        raise ValueError(
            "NLBNP requires at least one worthy structural edge, supplied with "
            "worthy_edges, worthy_edge_attr, or additional_constraints."
        )
    effective_worthy_edges = normalize_node_pairs(
        constraints["worthy_edges"],
        A.shape[0],
        relation_name="worthy-edge",
        reject_self=True,
    )
    # Besides returning the complementary edge set, this verifies that the
    # worthy-edge specification is nonempty and contains only graph edges.
    active_unworthy_edges = unworthy_edges(A, effective_worthy_edges)
    constraints["worthy_edges"] = effective_worthy_edges

    feasibility: LinearGroupFeasibility | None = None
    active_must_link = list(user_must_link)
    active_cannot_link = list(user_cannot_link)
    if cardinality_method == "reformulated":
        feasibility = compute_max_feasible_linear_group(
            A,
            worthy_edges=effective_worthy_edges,
            nonlinear_nodes=nonlinear_idx,
            must_link=user_must_link,
            cannot_link=user_cannot_link,
        )
        feasibility_metadata = {
            "cardinality_method": cardinality_method,
            "node_label_map": node_label_map,
            "label_node_map": label_node_map,
            "nonlinear_nodes": nonlinear_idx,
            "worthy_edges": effective_worthy_edges,
            "K_max": feasibility.K_max,
            "eligible_nodes": list(feasibility.eligible_nodes),
            "y": feasibility.y.copy(),
            "required_together_components": feasibility.components,
            "derived_must_link": list(feasibility.derived_must_link),
            "derived_cannot_link": list(feasibility.derived_cannot_link),
        }
        if feasibility.K_max == 0:
            return _infeasible_result(
                "empty_maximum_eligible_set",
                **feasibility_metadata,
            )
        if feasibility.conflicting_cannot_link:
            eligible = set(feasibility.eligible_nodes)
            conflict_in_y = any(
                source in eligible and target in eligible
                for source, target in feasibility.conflicting_cannot_link
            )
            return _infeasible_result(
                (
                    "cannot_link_inside_maximum_eligible_set"
                    if conflict_in_y
                    else "cannot_link_inside_required_together_component"
                ),
                conflicting_cannot_link=list(feasibility.conflicting_cannot_link),
                **feasibility_metadata,
            )
        active_must_link = normalize_node_pairs(
            [*user_must_link, *feasibility.derived_must_link],
            A.shape[0],
            relation_name="must-link",
        )
        active_cannot_link = normalize_node_pairs(
            [*user_cannot_link, *feasibility.derived_cannot_link],
            A.shape[0],
            relation_name="cannot-link",
            reject_self=True,
        )

    required_links = list(user_must_link)
    required_links = normalize_node_pairs(
        [*required_links, *active_unworthy_edges],
        A.shape[0],
        relation_name="must-link",
    )
    initial_must_link = normalize_node_pairs(
        [*active_must_link, *required_links],
        A.shape[0],
        relation_name="initial-column must-link",
    )

    resolved_contract_graph = (
        cardinality_method == "reformulated"
        if contract_graph is None
        else bool(contract_graph)
    )
    automatic_contraction = contract_graph is None and resolved_contract_graph

    if ifc_params is None:
        resolved_ifc_params = {
            "generator": make_partitions_random_links_only,
            "num": 1,
            "args": {
                "N": A.shape[0],
                "must_link": initial_must_link,
                "cannot_link": active_cannot_link,
                "n_parts": 1,
            },
        }
    else:
        resolved_ifc_params = copy.deepcopy(ifc_params)

    cardinality_kwargs = copy.deepcopy(cardinality_params or {})
    reserved_cardinality_keys = {
        "cannot_link",
        "must_group",
        "must_link",
        "nonlinear_nodes",
        "worthy_edges",
    }
    invalid_cardinality_keys = reserved_cardinality_keys.intersection(
        cardinality_kwargs
    )
    if invalid_cardinality_keys:
        names = ", ".join(sorted(invalid_cardinality_keys))
        raise ValueError(
            f"cardinality_params cannot override workflow constraint inputs: {names}."
        )
    source_refine_params = cfg.refine_params if refine_params is None else refine_params
    resolved_refine_params = dict(source_refine_params or {})
    if "kwargs" in resolved_refine_params:
        resolved_refine_params["kwargs"] = dict(
            resolved_refine_params.get("kwargs") or {}
        )
    stage_one_refiner = resolved_refine_params.get("refine_func")
    if (use_refined_column or refine_post_loop) and not callable(stage_one_refiner):
        raise ValueError(
            "use_refined_column=True or refine_post_loop=True requires "
            "refine_params['refine_func']."
        )

    pricing_fn = subproblem_fn
    if pricing_fn is None:
        pricing_fn = (
            custom_heuristic_subproblem
            if algorithm in _CUSTOM_HEURISTIC_ALGOS
            else heuristic_subproblem
        )

    cfg.must_link = active_must_link
    cfg.cannot_link = active_cannot_link
    cfg.additional_constraints = constraints
    cfg.algo = algorithm
    cfg.package = package
    cfg.resolution = resolution
    cfg.seed = seed
    cfg.ifc_params = resolved_ifc_params
    cfg.refine_params = resolved_refine_params
    cfg.use_refined_column = use_refined_column
    cfg.refine_post_loop = refine_post_loop
    cfg.final_master_solve = final_master_solve
    cfg.check_flat_pricing = check_flat_pricing
    cfg.stopping_window = stopping_window
    cfg.contract_graph = resolved_contract_graph
    cfg.max_iterations = max_iterations
    cfg.disable_tqdm = disable_tqdm
    cfg.tolerance = tolerance
    cfg.verbose = verbose

    start = time.perf_counter()
    result = run_csd_decomposition(
        A,
        config=cfg,
        master_fn=master_fn,
        subproblem_fn=pricing_fn,
        **overrides,
    )
    elapsed = time.perf_counter() - start

    metadata = dict(result.metadata)
    metadata.update(
        {
            "algorithm": algorithm,
            "package": package,
            "resolution": float(resolution),
            "execution_time": elapsed,
            "node_label_map": node_label_map,
            "label_node_map": label_node_map,
            "cardinality_method": cardinality_method,
            "nonlinear_nodes": nonlinear_idx,
            "must_link": active_must_link,
            "cannot_link": active_cannot_link,
            "user_must_link": user_must_link,
            "user_cannot_link": user_cannot_link,
            "worthy_edges": effective_worthy_edges,
            "contract_graph": resolved_contract_graph,
            "automatic_contraction": automatic_contraction,
            "use_refined_column": bool(use_refined_column),
            "refine_post_loop": bool(refine_post_loop),
        }
    )
    if feasibility is not None:
        metadata.update(
            {
                "K_max": feasibility.K_max,
                "eligible_nodes": list(feasibility.eligible_nodes),
                "y": feasibility.y.copy(),
                "required_together_components": feasibility.components,
                "derived_must_link": list(feasibility.derived_must_link),
                "derived_cannot_link": list(feasibility.derived_cannot_link),
            }
        )
    result.metadata = metadata

    base_partition, source, score = _select_hard_partition(
        result,
        A,
        must_link=active_must_link,
        cannot_link=active_cannot_link,
        worthy_edges=effective_worthy_edges,
        resolution=resolution,
    )
    if base_partition is None:
        result.final_partition = None
        if result.metadata.get("status") != "infeasible":
            result.metadata["status"] = "no_integral_partition"
        result.metadata["final_partition_source"] = None
        return result

    final_partition = base_partition
    final_source = source
    uses_cardinality_refiner = cardinality_method in {
        "confidence",
        "core_periphery",
    }
    if uses_cardinality_refiner:
        try:
            if cardinality_method == "confidence":
                refiner_kwargs = {
                    "nonlinear_nodes": nonlinear_idx,
                    "worthy_edges": effective_worthy_edges,
                    "must_link": user_must_link,
                    "cannot_link": user_cannot_link,
                    "prob_method": prob_method,
                    "verbose": False,
                    "seed": seed,
                }
                refiner_kwargs.update(cardinality_kwargs)
                final_partition = refine_partition_linear_group(
                    A,
                    base_partition,
                    **refiner_kwargs,
                )
            else:
                refiner_kwargs = {
                    "nonlinear_nodes": nonlinear_idx,
                    "must_link": required_links,
                    "cannot_link": user_cannot_link,
                    "seed": seed,
                }
                refiner_kwargs.update(cardinality_kwargs)
                final_partition = refine_partition_with_cp(
                    A,
                    base_partition,
                    **refiner_kwargs,
                )
        except RuntimeError as exc:
            final_partition = None
            result.metadata["cardinality_refinement_error"] = str(exc)
        final_source = f"{cardinality_method}_cardinality_refinement"

    expected_nodes = feasibility.eligible_nodes if feasibility is not None else None
    try:
        final_partition = (
            None
            if final_partition is None
            else validate_partition_matrix(
                final_partition,
                A.shape[0],
                name="NLBNP final partition",
            )
        )
    except ValueError:
        final_partition = None
    final_is_valid = (
        final_partition is not None
        and partition_satisfies_pairwise_constraints(
            final_partition,
            must_link=active_must_link,
            cannot_link=active_cannot_link,
        )
        and partition_satisfies_edge_constraints(
            A,
            final_partition,
            effective_worthy_edges,
        )
        and partition_satisfies_nlbnp_cardinality(
            final_partition,
            nonlinear_idx,
            expected_nodes=expected_nodes,
        )
    )
    if not final_is_valid:
        result.final_partition = None
        result.metadata["status"] = (
            "cardinality_refinement_failed"
            if uses_cardinality_refiner
            else "cardinality_constraint_unsatisfied"
        )
        result.metadata["final_partition_source"] = final_source
        return result

    result.final_partition = final_partition
    result.metadata["status"] = "ok"
    result.metadata["final_partition_source"] = final_source
    result.metadata["final_partition_score"] = (
        score
        if final_partition is base_partition
        else compute_f_star(
            A,
            A.sum(axis=1),
            float(A.sum()),
            final_partition,
            gamma=resolution,
        )
    )
    linear_groups = linear_only_communities(final_partition, nonlinear_idx)
    linear_nodes = list(linear_groups[0])
    result.metadata["linear_only_node_indices"] = linear_nodes
    result.metadata["linear_only_nodes"] = [node_label_map[node] for node in linear_nodes]
    result.metadata["cardinality"] = len(linear_nodes)
    if feasibility is None:
        y = np.zeros(A.shape[0], dtype=bool)
        y[linear_nodes] = True
        result.metadata["y"] = y
        result.metadata["K_max"] = None
    _result_identity_metadata(result, node_label_map=node_label_map)
    return result


def CorePeripheryPartition(
    graph: nx.Graph | np.ndarray,
    *,
    must_link: Sequence[tuple[Hashable, Hashable]] | None = None,
    must_link_edge_attr: str | None = None,
    must_link_edge_value: Any = None,
    must_group: Sequence[Hashable] | None = None,
    must_group_node_attr: str | None = None,
    must_group_node_value: Any = None,
    cp_algorithm: str = "SPEC",
    target: CorePeripheryTarget = "contracted",
    spectral_rank: int = 1,
    prob_method: str = "gaussian_mixture",
    threshold: float = 0.8,
    seed: int | None = 42,
    verbose: bool = False,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Apply the NLBNP linear-only-group structural shortcut.

    In the intended NLBNP interpretation, grouping constraints collect the
    nonlinear nodes into one detection block on the core side. The complementary
    periphery is merged into the single linear-only community required by the
    cardinality constraint. That linear-only group is then excluded from the
    original input adjacency, not the contracted detection adjacency, and each
    remaining core-side connected component becomes an independent final
    community. Use :func:`run_nonlinear_branch_and_price` when this shortcut is
    not valid.

    Parameters
    ----------
    graph : networkx.Graph or ndarray
        Input graph or square adjacency matrix.
    must_link : sequence of tuple, optional
        Node pairs that must share a core-periphery block and final community.
    must_link_edge_attr : str, optional
        Edge attribute used to derive ``must_link`` pairs.
    must_link_edge_value : Any, optional
        Attribute value selected by ``must_link_edge_attr``.
    must_group : sequence, optional
        Designated nonlinear nodes merged into one detection block. They share
        the core side but are not forced into one final independent community.
    must_group_node_attr : str, optional
        Node attribute used to derive ``must_group`` nodes.
    must_group_node_value : Any, optional
        Attribute value selected by ``must_group_node_attr``.
    cp_algorithm : {"SPEC", "GA", "KL"}
        Core-periphery detection algorithm.
    target : {"contracted", "original"}
        Space whose core-periphery structure is optimized and reported as the
        primary fit. Contracted space is the default. The final connected
        components are always calculated from the original input adjacency.
    spectral_rank : {1, 2}
        Spectral approximation rank. Rank two uses a scaled adjacency spectral
        embedding and requires Gaussian-mixture conversion.
    prob_method : {"threshold", "gaussian_mixture", "DBSCAN"}
        Method used to convert continuous coreness values to discrete labels.
    threshold : float
        Threshold used when ``prob_method="threshold"``.
    seed : int or None
        Random seed.
    verbose : bool
        Controls probability conversion output.

    Returns
    -------
    community_labels : ndarray of int, shape (N,)
        Community labels where the merged linear-only periphery is community
        ``0`` and independent original-graph core components are communities
        ``1..K``.
    metadata : dict
        Core-periphery detection, component, and graph-label metadata.
    """
    start = time.perf_counter()
    A, node_labels, nx_graph = _coerce_graph_input(graph)
    label_node_map = {label: idx for idx, label in enumerate(node_labels)}
    node_label_map = {idx: label for idx, label in enumerate(node_labels)}

    attr_edges = _edge_pairs_from_attribute(
        nx_graph,
        must_link_edge_attr,
        must_link_edge_value,
        name="must_link_edge_attr",
    )
    must_link_idx = _unique_pairs(
        _map_pairs([*_items_or_empty(must_link), *attr_edges], label_node_map, name="must_link")
    )
    attr_nodes = _nodes_from_attribute(
        nx_graph,
        must_group_node_attr,
        must_group_node_value,
        name="must_group_node_attr",
    )
    must_group_idx = _map_nodes(
        [*_items_or_empty(must_group), *attr_nodes],
        label_node_map,
        name="must_group",
    )

    cp_result = _detect_core_periphery(
        A,
        must_link=must_link_idx,
        must_group=must_group_idx,
        algorithm=cp_algorithm,
        target=target,
        spectral_rank=spectral_rank,
        prob_method=prob_method,
        threshold=threshold,
        verbose=verbose,
        seed=seed,
    )
    if cp_result.node_labels is None:
        raise RuntimeError("Core-periphery detection did not return binary node labels.")
    core_labels = cp_result.node_labels
    linear_only_mask = _nlbnp_linear_only_mask(
        core_labels,
        nonlinear_nodes=must_group_idx,
    )
    community_labels, component_info = partition_periphery_components(
        A,
        linear_only_mask,
        must_link=must_link_idx,
    )
    component_info.update(
        {
            "n_core": int(np.count_nonzero(core_labels)),
            "n_periphery": int(np.count_nonzero(linear_only_mask)),
            "n_linear_only": int(np.count_nonzero(linear_only_mask)),
            "n_independent_nodes": int(np.count_nonzero(core_labels)),
            "linear_only_node_indices": np.flatnonzero(linear_only_mask),
            "independent_components": component_info["components"],
            "component_graph_space": "original",
        }
    )
    community_map = {idx: int(label) for idx, label in enumerate(community_labels)}
    communities = component_info["community_node_indices"]

    metadata = {
        **cp_result.to_metadata(),
        **component_info,
        "execution_time": time.perf_counter() - start,
        "node_label_map": node_label_map,
        "label_node_map": label_node_map,
        "must_link": must_link_idx,
        "must_group": must_group_idx,
        "core_labels": core_labels,
        "community_map": community_map,
        "community_map_labels": map_community_labels(community_map, node_label_map),
        "communities_labels": [
            [node_label_map[int(idx)] for idx in community]
            for community in communities
        ],
        "n_communities": int(np.unique(community_labels).size),
    }
    return community_labels, metadata


def NonlinearBranchAndPrice(*args: Any, **kwargs: Any) -> DecompositionResult:
    """
    LoadBalancer-style alias for :func:`run_nonlinear_branch_and_price`.
    """
    return run_nonlinear_branch_and_price(*args, **kwargs)


NonlinearBranchAndPrice.__signature__ = inspect.signature(run_nonlinear_branch_and_price)
