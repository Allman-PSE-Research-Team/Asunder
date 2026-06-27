"""Generic high-level nonlinear branch-and-price workflow."""

from __future__ import annotations

import copy
import inspect
import time
from collections.abc import Hashable, Sequence
from typing import Any

import networkx as nx
import numpy as np

from asunder.base.algorithms.core_periphery import partition_periphery_components
from asunder.base.column_generation.master import solve_master_problem
from asunder.base.column_generation.subproblem import (
    custom_heuristic_subproblem,
    heuristic_subproblem,
)
from asunder.base.utils.graph import group_nodes_by_community, map_community_labels
from asunder.base.utils.partition_generation import make_simple_partition
from asunder.config import CSDDecompositionConfig
from asunder.nlbnp.algorithms.core_periphery import _detect_core_periphery
from asunder.nlbnp.algorithms.refinement import refine_partition_linear_group
from asunder.orchestrator import run_csd_decomposition
from asunder.types import DecompositionResult, MasterProblemFn, SubproblemFn

_CUSTOM_HEURISTIC_ALGOS = {"spectral", "full_louvain", "RCCS"}


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
) -> list[Hashable]:
    if node_attr is None:
        return []
    if graph is None:
        raise ValueError("nonlinear_node_attr can only be used when graph is a networkx.Graph.")

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


def run_nonlinear_branch_and_price(
    graph: nx.Graph | np.ndarray,
    *,
    worthy_edges: Sequence[tuple[Hashable, Hashable]] | None = None,
    worthy_edge_attr: str | None = None,
    worthy_edge_value: Any = None,
    must_link: Sequence[tuple[Hashable, Hashable]] | None = None,
    cannot_link: Sequence[tuple[Hashable, Hashable]] | None = None,
    algorithm: str = "louvain",
    package: str | None = "networkx",
    seed: int | None = 42,
    ifc_params: dict[str, Any] | None = None,
    refine: bool = True,
    refine_params: dict[str, Any] | None = None,
    prob_method: str = "threshold",
    use_refined_column: bool = True,
    refine_post_loop: bool = True,
    final_master_solve: bool = False,
    extract_dual: bool = True,
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
        Edge pairs that should be treated as worthy edges. For ``networkx``
        inputs, pairs use graph node labels. For adjacency inputs, pairs use
        integer node indices.
    worthy_edge_attr : str, optional
        Edge attribute used to derive worthy edges from a ``networkx.Graph``.
        When ``worthy_edge_value`` is ``None``, truthy attribute values are
        selected. Otherwise, the attribute must equal ``worthy_edge_value``.
    worthy_edge_value : Any, optional
        Attribute value selected by ``worthy_edge_attr``.
    must_link, cannot_link : sequence of tuple, optional
        Pairwise constraints using graph node labels or adjacency indices.
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
    seed : int or None
        Random seed.
    ifc_params : dict or None
        Initial feasible column generator configuration. Defaults to
        ``make_simple_partition``.
    refine : bool
        Whether to configure the default linear-group refinement hook.
    refine_params : dict or None
        Explicit refinement configuration. Overrides the default refinement
        configuration when supplied.
    prob_method : str
        Probability-to-label method passed to the default linear-group
        refinement function.
    use_refined_column : bool
        Whether refined columns should be added to the column pool.
    refine_post_loop : bool
        Whether to run post-loop refinement after column generation terminates.
    final_master_solve : bool
        Whether to run a final integer master solve.
    extract_dual : bool
        Whether to solve the relaxed master with dual extraction.
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
        Structured decomposition result with label-aware metadata.
    """
    A, node_labels, nx_graph = _coerce_graph_input(graph)
    label_node_map = {label: idx for idx, label in enumerate(node_labels)}
    node_label_map = {idx: label for idx, label in enumerate(node_labels)}

    must_link_idx = _map_pairs(must_link, label_node_map, name="must_link")
    cannot_link_idx = _map_pairs(cannot_link, label_node_map, name="cannot_link")

    attr_edges = _edge_pairs_from_attribute(
        nx_graph,
        worthy_edge_attr,
        worthy_edge_value,
        name="worthy_edge_attr",
    )
    worthy_edge_idx = _unique_pairs(
        _map_pairs([*_items_or_empty(worthy_edges), *attr_edges], label_node_map, name="worthy_edges")
    )
    cfg = copy.deepcopy(config) if config is not None else CSDDecompositionConfig()
    constraints = copy.deepcopy(cfg.additional_constraints)
    if additional_constraints:
        constraints.update(copy.deepcopy(additional_constraints))
    if worthy_edges is not None or worthy_edge_attr is not None:
        constraints["worthy_edges"] = worthy_edge_idx

    if ifc_params is None:
        ifc_params = {
            "generator": make_simple_partition,
            "num": 1,
            "args": {"N": A.shape[0], "cannot_link": cannot_link_idx},
        }
    else:
        ifc_params = copy.deepcopy(ifc_params)

    if refine_params is None:
        refine_params = (
            {
                "refine_func": refine_partition_linear_group,
                "kwargs": {"prob_method": prob_method, "verbose": False},
            }
            if refine
            else {}
        )
    else:
        refine_params = copy.deepcopy(refine_params)

    pricing_fn = subproblem_fn
    if pricing_fn is None:
        pricing_fn = custom_heuristic_subproblem if algorithm in _CUSTOM_HEURISTIC_ALGOS else heuristic_subproblem

    cfg.must_link = must_link_idx
    cfg.cannot_link = cannot_link_idx
    cfg.additional_constraints = constraints
    cfg.algo = algorithm
    cfg.package = package
    cfg.seed = seed
    cfg.extract_dual = extract_dual
    cfg.ifc_params = ifc_params
    cfg.refine_params = refine_params
    cfg.use_refined_column = use_refined_column and refine
    cfg.refine_post_loop = refine_post_loop and refine
    cfg.final_master_solve = final_master_solve
    cfg.max_iterations = max_iterations
    cfg.disable_tqdm = disable_tqdm
    cfg.tolerance = tolerance
    cfg.verbose = verbose

    start = time.perf_counter()
    result = run_csd_decomposition(A, config=cfg, master_fn=master_fn, subproblem_fn=pricing_fn, **overrides)
    elapsed = time.perf_counter() - start

    metadata = dict(result.metadata)
    metadata.update(
        {
            "algorithm": algorithm,
            "package": package,
            "execution_time": elapsed,
            "node_label_map": node_label_map,
            "label_node_map": label_node_map,
            "must_link": must_link_idx,
            "cannot_link": cannot_link_idx,
            "worthy_edges": worthy_edge_idx if (worthy_edges is not None or worthy_edge_attr is not None) else constraints.get("worthy_edges"),
        }
    )
    if result.final_partition is not None:
        community_map, communities = group_nodes_by_community(result.final_partition)
        metadata["community_map"] = community_map
        metadata["community_map_labels"] = map_community_labels(community_map, node_label_map)
        metadata["communities"] = communities

    result.metadata = metadata
    return result


def CorePeripheryPartition(
    graph: nx.Graph | np.ndarray,
    *,
    unworthy_edges: Sequence[tuple[Hashable, Hashable]] | None = None,
    unworthy_edge_attr: str | None = None,
    unworthy_edge_value: Any = None,
    nonlinear_nodes: Sequence[Hashable] | None = None,
    nonlinear_node_attr: str | None = None,
    nonlinear_node_value: Any = None,
    cp_algorithm: str = "SPEC",
    prob_method: str = "gaussian_mixture",
    threshold: float = 0.8,
    seed: int | None = 42,
    verbose: bool = False,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Solve NLBNP by separating a core and partitioning periphery components.

    This path is appropriate when removing the detected core naturally leaves
    each connected periphery component as a complete final community. Use
    :func:`run_nonlinear_branch_and_price` when those components need further
    subdivision.

    Parameters
    ----------
    graph : networkx.Graph or ndarray
        Input graph or square adjacency matrix.
    unworthy_edges : sequence of tuple, optional
        Edge pairs that cannot connect separate communities.
    unworthy_edge_attr : str, optional
        Edge attribute used to derive unworthy edges from a ``networkx.Graph``.
    unworthy_edge_value : Any, optional
        Attribute value selected by ``unworthy_edge_attr``.
    nonlinear_nodes : sequence, optional
        Nodes that represent nonlinear constraints and should remain together.
    nonlinear_node_attr : str, optional
        Node attribute used to derive nonlinear nodes from a ``networkx.Graph``.
    nonlinear_node_value : Any, optional
        Attribute value selected by ``nonlinear_node_attr``.
    cp_algorithm : {"SPEC", "GA", "KL"}
        Core-periphery detection algorithm.
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
        Community labels where the core is community ``0`` and connected
        periphery components are communities ``1..K``.
    metadata : dict
        Core-periphery detection, component, and graph-label metadata.
    """
    start = time.perf_counter()
    A, node_labels, nx_graph = _coerce_graph_input(graph)
    label_node_map = {label: idx for idx, label in enumerate(node_labels)}
    node_label_map = {idx: label for idx, label in enumerate(node_labels)}

    attr_edges = _edge_pairs_from_attribute(
        nx_graph,
        unworthy_edge_attr,
        unworthy_edge_value,
        name="unworthy_edge_attr",
    )
    unworthy_edge_idx = _unique_pairs(
        _map_pairs([*_items_or_empty(unworthy_edges), *attr_edges], label_node_map, name="unworthy_edges")
    )
    attr_nodes = _nodes_from_attribute(nx_graph, nonlinear_node_attr, nonlinear_node_value)
    nonlinear_node_idx = _map_nodes(
        [*_items_or_empty(nonlinear_nodes), *attr_nodes],
        label_node_map,
        name="nonlinear_nodes",
    )

    core_labels, cp_metadata = _detect_core_periphery(
        A,
        unworthy_edges=unworthy_edge_idx,
        nonlinear_nodes=nonlinear_node_idx,
        algorithm=cp_algorithm,
        prob_method=prob_method,
        threshold=threshold,
        verbose=verbose,
        seed=seed,
    )
    community_labels, component_info = partition_periphery_components(A, core_labels)
    community_map = {idx: int(label) for idx, label in enumerate(community_labels)}
    communities = component_info["community_node_indices"]

    metadata = {
        **cp_metadata,
        **component_info,
        "execution_time": time.perf_counter() - start,
        "node_label_map": node_label_map,
        "label_node_map": label_node_map,
        "unworthy_edges": unworthy_edge_idx,
        "nonlinear_nodes": nonlinear_node_idx,
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
