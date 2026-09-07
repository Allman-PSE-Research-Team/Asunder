import time

import networkx as nx
import numpy as np

from asunder.base.column_generation.master import compute_f_star
from asunder.base.column_generation.subproblem import (
    custom_heuristic_subproblem,
    heuristic_subproblem,
)
from asunder.base.utils.graph import (
    expand_z_matrix,
    group_nodes_by_community,
    map_community_labels,
)
from asunder.config import CSDDecompositionConfig
from asunder.load_balancing.algorithms.projection import project_partition_ilp
from asunder.load_balancing.algorithms.qmetis import bundled_qmetis_release
from asunder.load_balancing.algorithms.VFD import refine_partition
from asunder.load_balancing.column_generation.master import solve_master_problem
from asunder.load_balancing.column_generation.subproblem import (
    qmetis_pricing_subproblem,
)
from asunder.load_balancing.utils.balance import resolve_balance_bounds
from asunder.load_balancing.utils.partition_generation import (
    make_partitions,
    make_partitions_random,
)
from asunder.orchestrator import run_csd_decomposition
from asunder.solvers import get_default_solver
from asunder.types import DecompositionResult

_CUSTOM_HEURISTIC_ALGOS = {"spectral", "full_louvain", "RCCS"}


def _projection_time_limit_option(solver):
    names = [
        getattr(solver, "type", ""),
        getattr(solver, "name", ""),
        solver.__class__.__name__,
    ]
    solver_name = " ".join(str(name).lower() for name in names)
    if "gurobi" in solver_name:
        return "TimeLimit"
    if "cplex" in solver_name:
        return "timelimit"
    if "highs" in solver_name:
        return "time_limit"
    return None


def _temporarily_apply_projection_time_limit(solver, projection_time_limit):
    option_name = _projection_time_limit_option(solver)
    if projection_time_limit is None or option_name is None:
        return lambda: None, False

    options = getattr(solver, "options", None)
    if options is None:
        return lambda: None, False

    had_option = option_name in options
    old_value = options.get(option_name)
    options[option_name] = float(projection_time_limit)

    def restore():
        if had_option:
            options[option_name] = old_value
        else:
            try:
                del options[option_name]
            except KeyError:
                pass

    return restore, True


def _post_loop_refinement_succeeded(result: DecompositionResult) -> bool:
    if not result.records:
        return False
    final_record = result.records[-1]
    return final_record.sub_obj_val is None and final_record.heuristic_col is not None


def _last_master_fractional_partition(result: DecompositionResult):
    for record in reversed(result.records):
        if record.lambda_sol is None or not record.columns:
            continue
        wz = np.zeros_like(record.columns[0], dtype=float)
        for lambda_, column in zip(record.lambda_sol, record.columns):
            wz += float(lambda_) * np.asarray(column, dtype=float)
        node2comp = result.metadata.get("node2comp")
        if node2comp is not None and wz.shape[0] != len(node2comp):
            wz = expand_z_matrix(wz, node2comp)
        return wz, record
    return None, None


def _project_after_failed_post_loop_refinement(
    *,
    result: DecompositionResult,
    A,
    a,
    m,
    K,
    R,
    R_bounds,
    must_link,
    cannot_link,
    balance_weights,
    seed,
    projection_time_limit,
):
    if result.metadata.get("final_partition_source") in {
        "contracted_trivial",
        "integer_master",
        "post_loop_refinement",
        "projection_repair",
    }:
        return None
    if _post_loop_refinement_succeeded(result):
        return None

    wz, _ = _last_master_fractional_partition(result)
    if wz is None:
        return None

    try:
        solver = get_default_solver()
    except Exception:
        return None

    restore_time_limit, time_limit_applied = _temporarily_apply_projection_time_limit(
        solver,
        projection_time_limit,
    )
    try:
        projected = project_partition_ilp(
            wz=wz,
            A=A,
            a=a,
            m=m,
            K=K,
            R=R,
            R_bounds=R_bounds,
            must_link=must_link,
            cannot_link=cannot_link,
            balance_weights=balance_weights,
            seed=seed,
            solver=solver,
        )
    finally:
        restore_time_limit()

    if projected is None:
        return None

    repaired_z, meta = projected
    meta["projection_time_limit"] = None if projection_time_limit is None else float(projection_time_limit)
    meta["projection_time_limit_applied"] = bool(time_limit_applied)
    return repaired_z, meta


def LoadBalancer(
    G,
    R=1,
    K=2,
    R_bounds=None,
    algorithm="signed_leiden",
    package="leidenalg",
    ifc_generator="random",
    seed=42,
    must_link=None,
    cannot_link=None,
    node_weight_attr=None,
    contract_graph=False,
    refine=True,
    refine_params=None,
    use_refined_column=True,
    refine_post_loop=True,
    final_master_solve=False,
    check_flat_pricing=True,
    stopping_window=3,
    projection_repair=False,
    projection_time_limit=15.0,
    max_iterations=None,
    disable_tqdm=False,
    verbose=-1,
    resolution=1.0,
) -> DecompositionResult:
    """
    Solve the load-balanced structure detection problem using Asunder's column generation workflow.

    Parameters
    ----------
    G : nx.Graph
        NetworkX graph representing the relevant problem.
    R : int
        Width of the allowed community-load range. Also corresponds to the load balance tightness (smaller R implies tighter load balance).
        For a selected cluster count, the lower and upper bounds are computed from the corresponding
        balanced range rule.
    K : int
        Number of communities.
    resolution : float, default=1.0
        Modularity resolution parameter used by pricing and column scoring.
        Algorithms that cannot apply non-default resolution reject it.
    R_bounds : tuple[int, int] | None
        Minimum and maximum total node weight per community.
    algo : str
        Name of heuristic subproblem used to replace the ILP subproblem. Third-party algorithms combine adjacency and dual information into a unified input while custom algorithms treat adjacency and duals as separate inputs. Supported third-party algorithms are listed under the ``package`` parameter.
        Available custom algorithm options include:

        ``"spectral"``:
            Modified iterative bisection algorithm based on Mark Newman's eigenvector-based method.
        ``"full_louvain"``:
            Modified but Louvain-like algorithm.
        ``"RCCS"``:
            This means Reduced Cost Community Search and is a greedy and local search heuristic for finding communities that maximize the reduced cost.
        ``"qmetis"``:
            Uses the bundled modularity QMETIS library as a load-balancing
            pricing heuristic. Fractional dual-adjusted weights are safely
            quantized for QMETIS and every candidate is rescored using
            Asunder's original floating-point reduced-cost objective.
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
    node_weight_attr : str or None
        Positive integer node attribute used for weighted load balancing.
        Missing values default to one. If omitted, every node has unit load.
    contract_graph : bool
        Contract must-link components before column generation while
        preserving their summed balance weights.
    refine : bool
        Master switch for LB VFD refinement.
    refine_params : dict or None
        Optional custom refinement hook configuration. By default the
        independent load-balancing VFD adapter is used.
    use_refined_column : bool
        Whether to run and add refinement columns inside the main loop.
    refine_post_loop : bool
        Whether to run post-loop refinement after column generation terminates.
    final_master_solve : bool
        Whether to solve the final integer restricted master.
    check_flat_pricing : bool
        Whether to stop when reduced costs remain flat.
    stopping_window : int
        Number of reduced costs used by the flat-pricing test.
    projection_repair : bool
        If True, project the refinement input to the nearest feasible
        load-balanced partition when VFD refinement returns ``None``.
    projection_time_limit : float or None
        Best-effort solver time limit in seconds for ``projection_repair``.
        Applied only to supported solver backends.
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
    nodes = list(G.nodes())
    A = nx.to_numpy_array(G, nodelist=nodes)
    a = np.sum(A, axis=1)
    m = a.sum()
    node_label_map = {i: label for i, label in enumerate(nodes)}
    label_node_map = {label: i for i, label in enumerate(nodes)}

    if node_weight_attr is None:
        balance_weights = np.ones(A.shape[0], dtype=int)
    else:
        raw_weights = np.asarray(
            [G.nodes[node].get(node_weight_attr, 1) for node in nodes]
        )
        if not np.all(np.isfinite(raw_weights)) or np.any(raw_weights <= 0):
            raise ValueError(
                f"Node attribute {node_weight_attr!r} must contain finite "
                "positive weights."
            )
        if not np.all(raw_weights == np.rint(raw_weights)):
            raise ValueError(
                f"Node attribute {node_weight_attr!r} must contain integer weights."
            )
        balance_weights = np.rint(raw_weights).astype(int)

    # normalize constraint labels
    must_link = [
        (label_node_map[i], label_node_map[j]) for i, j in (must_link or [])
    ]
    cannot_link = [
        (label_node_map[i], label_node_map[j]) for i, j in (cannot_link or [])
    ]

    if projection_time_limit is not None and float(projection_time_limit) < 0:
        raise ValueError("projection_time_limit must be nonnegative or None.")

    if R_bounds is not None:
        R_bounds = resolve_balance_bounds(
            int(balance_weights.sum()),
            K,
            R,
            R_bounds,
        )

    ifc_params = {
        "num": 1,
        "args": {
            "must_link": must_link,
            "cannot_link": cannot_link,
            "R_bounds": R_bounds,
            "node_weights": balance_weights,
            "max_K_increase": 0,
        }
    }
    if ifc_generator == "ordered":
        ifc_params["generator"] = make_partitions
        ifc_params["args"] = dict(G=G, K=K, R=R, n_cols=1, **ifc_params["args"])
    elif ifc_generator == "random":
        ifc_params["generator"] = make_partitions_random
        ifc_params["args"] = dict(N=A.shape[0], K=K, R=R,  **ifc_params["args"])
    else:
        raise ValueError("ifc_generator must be either 'random' or 'ordered'.")

    if refine_params is None:
        resolved_refine_params = {
            "refine_func": refine_partition,
            "kwargs": dict(
                K=K,
                R=R,
                R_bounds=R_bounds,
                balance_weights=balance_weights,
                must_link=must_link,
                cannot_link=cannot_link,
                gamma=resolution,
                clustering_seeds=(seed,),
                w_coassoc=0.0,
            ),
        }

        # Illustrative ModularVFD replacement (intentionally inactive).  This is
        # equivalent to the LB adapter above when both searches receive the same
        # seed/defaults and ModularVFD's optional balance constraint is enabled.
        # from asunder.base.algorithms.modular_VFD import refine_partition_modular_vfd
        # resolved_refine_params = {
        #     "refine_func": refine_partition_modular_vfd,
        #     "kwargs": dict(
        #         K=K,
        #         R=R,
        #         R_bounds=R_bounds,
        #         balance_weights=balance_weights,
        #         must_link=must_link,
        #         cannot_link=cannot_link,
        #         gamma=resolution,
        #         use_K_constraint=True,
        #         candidate_Ks=None,
        #         fingerprint_decimals=6,
        #         allow_block_splitting=True,
        #         max_K_increase=0,
        #         restarts=6,
        #         local_iters=60,
        #         w_coassoc=0.0,
        #         clustering_Ks=None,
        #         clustering_seeds=(seed,),
        #         clustering_methods=("kmeans", "gmm", "spectral"),
        #         wz_is_C_node=False,
        #         tabu_max_steps=60,
        #         shake_rounds=3,
        #         orbit_fallback=False,
        #     ),
        # }
    else:
        resolved_refine_params = dict(refine_params)
    if not refine:
        resolved_refine_params = {}

    additional_constraints={
        "LB": True,
        "R": R,
        "K": K,
        "R_bounds": R_bounds,
        "balance_weights": balance_weights,
    }

    start = time.perf_counter()

    subproblem_params = {}
    if algorithm == "qmetis":
        subproblem_params = {
            "K": K,
            "R": R,
            "R_bounds": R_bounds,
        }
        if node_weight_attr is not None:
            subproblem_params["balance_weights"] = balance_weights

    config = CSDDecompositionConfig(
        must_link=must_link, cannot_link=cannot_link,
        additional_constraints=additional_constraints,
        algo=algorithm,
        package=package,
        resolution=resolution,
        contract_graph=contract_graph,
        disable_tqdm=disable_tqdm,
        seed=seed,
        check_flat_pricing=check_flat_pricing,
        stopping_window=stopping_window,
        # initial feasible column generator
        ifc_params = ifc_params,
        # refinement
        refine_params=resolved_refine_params,
        subproblem_params=subproblem_params,
        use_refined_column=bool(refine and use_refined_column),
        refine_post_loop=bool(refine and refine_post_loop),
        final_master_solve=final_master_solve,
        max_iterations=max_iterations, tolerance=1e-8, verbose=verbose,
    )
    if algorithm == "qmetis":
        pricing_function = qmetis_pricing_subproblem
    elif algorithm in _CUSTOM_HEURISTIC_ALGOS:
        pricing_function = custom_heuristic_subproblem
    else:
        pricing_function = heuristic_subproblem

    result = run_csd_decomposition(
        A, a=a, m=m,
        config=config,
        master_fn=solve_master_problem,
        subproblem_fn=pricing_function,
    )
    projection_repair_meta = []
    if projection_repair and refine and refine_post_loop:
        if not disable_tqdm:
            print("Trying a projection into the feasible space...")
        projected = _project_after_failed_post_loop_refinement(
            result=result,
            A=A,
            a=a,
            m=m,
            K=K,
            R=R,
            R_bounds=R_bounds,
            must_link=must_link,
            cannot_link=cannot_link,
            balance_weights=balance_weights,
            seed=seed,
            projection_time_limit=projection_time_limit,
        )
        if projected is not None:
            repaired_z, meta = projected
            result.final_partition = repaired_z
            result.final_master_obj = None
            result.metadata["final_partition_source"] = "projection_repair"
            result.metadata["status"] = "ok"
            projection_repair_meta.append(meta)

    if result.final_partition is None:
        raise RuntimeError(
            "Column generation did not produce an integral feasible partition. "
            "Enable refinement, final_master_solve, or projection_repair."
        )

    elapsed = time.perf_counter() - start
    z = result.final_partition
    community_map, _ = group_nodes_by_community(np.array(z))
    community_map_labels = map_community_labels(community_map, node_label_map)
    community_balance_weights = {}
    for index, community in community_map.items():
        community_balance_weights.setdefault(community, 0)
        community_balance_weights[community] += int(balance_weights[index])
    result.metadata.update({
        "community_map_labels": community_map_labels,
        "modularity": compute_f_star(A, a, m, z, gamma=resolution),
        "resolution": float(resolution),
        "execution_time": elapsed,
        "node_weight_attr": node_weight_attr,
        "total_balance_weight": int(balance_weights.sum()),
        "community_balance_weights": community_balance_weights,
    })
    if algorithm == "qmetis":
        result.metadata["qmetis_release"] = bundled_qmetis_release()
    if projection_repair_meta:
        result.metadata["projection_repairs"] = projection_repair_meta

    return result
