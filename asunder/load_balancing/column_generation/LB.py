import time

import networkx as nx
import numpy as np

from asunder.base.column_generation.master import compute_f_star
from asunder.base.column_generation.subproblem import custom_heuristic_subproblem, heuristic_subproblem
from asunder.base.utils.graph import group_nodes_by_community, map_community_labels
from asunder.config import CSDDecompositionConfig
from asunder.load_balancing.algorithms.VFD import refine_partition
from asunder.load_balancing.column_generation.master import solve_master_problem
from asunder.load_balancing.utils.partition_generation import (
    make_partitions,
    make_partitions_random,
)
from asunder.orchestrator import run_csd_decomposition
from asunder.types import DecompositionResult

_CUSTOM_HEURISTIC_ALGOS = {"spectral", "full_louvain", "RCCS"}

def LoadBalancer(
    G, 
    R=1, 
    K=2, 
    R_bounds=None, 
    algorithm="greedy", 
    package="networkx", 
    ifc_generator="random", 
    seed=42, 
    must_link=[],
    cannot_link=[],
    refine_post_loop=True,
    max_iterations=None,
    disable_tqdm=False,
    verbose=-1
) -> DecompositionResult:
    """
    Solve the load-balanced structure detection problem using Asunder's column generation workflow.
    
    Parameters
    ----------
    G : nx.Graph
        NetworkX graph representing the relevant problem.
    R : int
        Width of the allowed cluster-size range. Also corresponds to the load balance tightness (smaller R implies tighter load balance).
        For a selected cluster count, the lower and upper bounds are computed from the corresponding
        balanced range rule.
    K : int
        Number of communities.
    R_bounds : tuple[int, int] | None
        Minimum and maximum number of nodes per community (community size constraint).
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
    ifc_generator : str
        ``"random"`` if the initial feasible column should be randomly generated (default).
        ``"ordered"`` if the initial feasible column should be generated with some structure-based ordering.
    seed : int, default=None
        Random seed.
    must_link : list[tuple[int, int]]
        List of node pairs that must be together.
    cannot_link : list[tuple[int, int]]
        List of node pairs that must not be together.
    refine_post_loop : bool
        Whether to run post-loop refinement after column generation terminates.
    max_iterations : int or None
        Maximum number of column-generation iterations. ``None`` runs until convergence.
    disable_tqdm : bool
        Whether to disable progress bar or not.
    verbose : int or bool
        Controls the level of detail in the printed output.
        ``-1``: No output
        ``False`` | ``0``: Minimal output
        ``True`` | ``1``: Detailed output
    
    Returns
    -------
    DecompositionResult
        Column generation result. The final co-clustering matrix is available
        as ``final_partition`` and load-balancing summaries are in ``metadata``.
    """
    A = nx.to_numpy_array(G)
    a = np.sum(A, axis=1)
    m = a.sum()
    node_label_map = {i: label for i, label in enumerate(list(G.nodes()))}
    label_node_map = {label: i for i, label in enumerate(list(G.nodes()))}

    # normalize constraint labels
    must_link = [(label_node_map[i], label_node_map[j]) for i, j in must_link]
    cannot_link = [(label_node_map[i], label_node_map[j]) for i, j in cannot_link]

    # preprocess R_bounds if available for load balancing usecases to handle different cases
    if R_bounds is not None:
        R_min, R_max = R_bounds
        if R_min is None and R_max is None:
            # no bounds
            R_bounds = None
        else:
            if R_min is None:
                R_min = 1
            if R_max is None:
                R_max = np.shape(A)[0]
            if R_min > R_max:
                raise ValueError("Cardinality bounds are improperly defined.")
            R_bounds = (R_min, R_max)

    ifc_params = {
        "num": 1, 
        "args": {"must_link": must_link, "cannot_link": cannot_link, "R_bounds": R_bounds}
    }
    if ifc_generator == "ordered":
        ifc_params["generator"] = make_partitions
        ifc_params["args"] = dict(G=G, K=K, R=R, n_cols=1, **ifc_params["args"])
    elif ifc_generator == "random":
        ifc_params["generator"] = make_partitions_random
        ifc_params["args"] = dict(N=A.shape[0], K=K, R=R,  **ifc_params["args"])
    else:
        raise ValueError("ifc_generator must be either 'random' or 'ordered'.")

    refine_params = {
        "refine_func": refine_partition,
        "kwargs": dict(
            a=a, m=m, K=K, R=R, R_bounds=R_bounds,
            must_link=must_link, cannot_link=cannot_link,
            clustering_seeds=(seed,), w_coassoc=0.0,
        )
    }

    additional_constraints={"LB": True, "R": R, "K": K, "R_bounds": R_bounds}

    start = time.perf_counter()

    config = CSDDecompositionConfig(
        must_link=must_link, cannot_link=cannot_link,
        additional_constraints=additional_constraints,
        algo=algorithm,
        package=package,
        disable_tqdm=disable_tqdm,
        seed=seed,
        extract_dual=True,
        check_flat_pricing=True,
        stopping_window=3,
        # initial feasible column generator
        ifc_params = ifc_params,
        # refinement
        refine_params=refine_params,
        use_refined_column=True,
        refine_post_loop=refine_post_loop,
        final_master_solve=False,
        max_iterations=max_iterations, tolerance=1e-8, verbose=verbose,
    )
    result = run_csd_decomposition(
        A, a=a, m=m,
        config=config,
        master_fn=solve_master_problem,
        subproblem_fn=custom_heuristic_subproblem if algorithm in _CUSTOM_HEURISTIC_ALGOS else heuristic_subproblem,
    )
    elapsed = time.perf_counter() - start
    if result.final_partition is None:
        raise RuntimeError("Column generation failed due to infeasbility (No feasible initial columns could be generated or RMP is infeasible).")

    z = result.final_partition
    community_map, _ = group_nodes_by_community(np.array(z))
    community_map_labels = map_community_labels(community_map, node_label_map)
    result.metadata.update({
        "community_map_labels": community_map_labels,
        "modularity": compute_f_star(A, a, m, z),
        "execution_time": elapsed
    })

    return result
