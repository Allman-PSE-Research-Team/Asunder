from collections import defaultdict
import time
import numpy as np
import networkx as nx

from asunder.base.column_generation.decomposition import CSD_decomposition
from asunder.base.column_generation.subproblem import heuristic_subproblem
from asunder.base.utils.graph import group_nodes_by_community, map_community_labels

from asunder.load_balancing.column_generation.master import solve_master_problem
from asunder.load_balancing.algorithms.VFD import refine_partition
from asunder.load_balancing.utils.partition_generation import make_partitions, make_partitions_random

def LoadBalancer(
    G, 
    R=1, 
    K=None, 
    R_bounds=None, 
    algorithm="greedy", 
    package="networkx", 
    ifc_generator="random", 
    seed=None, 
    must_link=[], 
    cannot_link=[], 
    disable_tqdm=False,
    verbose=-1
):
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
    K : int | None
        Number of communities.
    R_bounds : tuple[int, int] | None
        Minimum and maximum number of nodes per community (community size constraint).
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
    ifc_generator : str
        ``"random"`` if the initial feasible column should be randomly generated (default).
        ``"ordered"`` if the initial feasible column should be generated with some structure-based ordering.
    seed : int, default=None
        Random seed.
    must_link : list[tuple[int, int]] or None
        List of node pairs that must be together.
    cannot_link : list[tuple[int, int]] or None
        List of node pairs that must not be together.
    disable_tqdm : bool
        Whether to disable progress bar or not.
    verbose : int or bool
        Controls the level of detail in the printed output.
        ``-1``: No output
        ``False`` | ``0``: Minimal output
        ``True`` | ``1``: Detailed output
    
    Returns
    -------
    lambda_sol: list or ndarray of float
        A list/vector which sums to ``1`` that indicates what weight is assigned to each column (and by implication, what columns are active).
    duals: dict[str, ndarray or float]
        Dual values computed from the master problem. This could be a 1D array, 2D array or a float.
    master_obj_val: float
        The objective value of the master problem.
    """
    A = nx.to_numpy_array(G)
    a = np.sum(A, axis=1)
    m = a.sum()
    node_label_map = {i: label for i, label in enumerate(list(G.nodes()))}

    ifc_params = {"num": 1, "must_link": must_link, "cannot_link": cannot_link, "R_bounds": R_bounds}
    if ifc_generator == "ordered":
        ifc_params["generator"] = make_partitions
        ifc_params["args"] = dict(G=G, K=K, R=R, n_cols=1)
    else:
        ifc_params["generator"] = make_partitions_random
        ifc_params["args"] = dict(N=A.shape[0], K=K, R=R)

    refine_params = {
        "refine_func": refine_partition,
        "kwargs": dict(
            a=a, m=m, K=K, R=R, R_bounds=R_bounds,
            must_link=must_link, cannot_link=cannot_link,
            clustering_seeds=(seed,), w_coassoc=0.0,
        )
    }

    additional_constraints=defaultdict(
        lambda: None,
        {"LB": True, "R": R, "K": K, "R_bounds": R_bounds}
    )

    start = time.time()

    colgen_results = CSD_decomposition(
        A, a, m,
        solve_master_problem,
        heuristic_subproblem,
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
        final_master_solve=False,
        max_iterations=None, tolerance=1e-8, verbose=verbose,
    )
    elapsed = time.time() - start

    z = colgen_results[-1]['z_sol']
    community_map, _ = group_nodes_by_community(np.array(z))
    community_map_labels = map_community_labels(community_map, node_label_map)

    return z, community_map_labels, elapsed