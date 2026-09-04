"""Column generation decomposition orchestration."""
from __future__ import annotations

import copy
import inspect
from collections import deque

import numpy as np
from tqdm.auto import tqdm

from asunder.base.column_generation.master import compute_f_star
from asunder.base.utils.graph import (
    contract_adj_matrix_new,
    contract_node_pairs,
    contract_partition_matrix,
    expand_z_matrix,
    sufficiently_different,
)


def CSD_decomposition(
    A, a, m,
    mp_function,
    sp_function,
    columns=None, f_stars=None,
    must_link=[], cannot_link=[],
    additional_constraints=None,
    contract_graph=False,
    stopping_window=5,
    check_flat_pricing=True,
    algo="louvain",
    package="sknetwork",
    seed=42,
    # initial feasible column generator
    ifc_params: dict = {},
    # refinement
    refine_params: dict={},
    use_refined_column=False,
    refine_post_loop=True,
    final_master_solve=True,
    max_iterations=1000, disable_tqdm=False,
    tolerance=1e-10, verbose=False,
    subproblem_params: dict | None = None,
    resolution=1.0,
):
    """
    Function that does column generation (CG) and refinement given a master and subproblem function.
    
    Parameters
    ----------
    A : np.ndarray of int | float, shape (N, N)
        Adjacency / weight matrix.
    a : np.ndarray of int | float, shape (N,)
        Degree-like vector; defaults to row sums of the symmetrized adjacency.
    m : float
        Twice the total weight in the graph.
    mp_function : callable
        Master problem function (Handles ILP and LP versions).
    sp_function : callable
        Pricing subproblem which can be implemented as a ILP or a heuristic (custom / third-party) subproblem.
    columns : list[ndarray of int] or None
        Existing columns. This parameter is typically active during Branch and
        Price. With ``contract_graph=True``, original-dimension columns must
        respect every contracted component and are converted automatically.
    f_stars : list[float] or None
        Objective values of the existing columns. This parameter is typically
        active during Branch and Price. Scores are recomputed on the contracted
        graph when contraction is active.
    must_link : list[tuple[int, int]]
        List of node pairs that must be together.
    cannot_link : list[tuple[int, int]]
        List of node pairs that must not be together.
    additional_constraints : dict[str, Any]
        Constraints beyond must- and cannot-links. For example, worthy edges (edges that can connect communities), community size, and balance constraints.
    contract_graph : bool
        Whether must-links are handled by graph contraction. Cannot-links and
        initial-column constraints are mapped to component indices; a
        cannot-link internal to one component is rejected as infeasible.
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
    resolution : float, default=1.0
        Modularity resolution parameter. It is applied consistently when
        pricing and scoring columns.
    seed : int or None
        Random seed value.
    ifc_params : dict[str, callable or dict or int]
        Number of initial feasible columns (ifc), initial feasible column generator, and its corresponding arguments (excluding seed values).
    refine_params : dict[str, callable or dict]
        Refinement function and its corresponding arguments (excluding seed values).
    subproblem_params : dict[str, Any] or None
        Additional keyword arguments passed to the pricing/subproblem callable.
    use_refined_column : bool
        Boolean that determines whether refined columns are used in the main column generation loop or not.
    refine_post_loop : bool
        Boolean that determines whether a post-loop refinement column is generated after column generation terminates.
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

    Returns
    -------
    results : list[dict]
        Column-generation iteration records. Each dictionary may include
        ``lambda_sol``, dual terms, ``master_obj_val``, ``z_sol``,
        ``sub_obj_val``, ``columns``, ``f_stars``, and ``heuristic_col``.

        When graph contraction is active, public ``z_sol`` values are expanded
        to original-node dimensions. ``columns`` and ``heuristic_col`` remain
        in contracted component dimensions, and ``f_stars`` score those
        contracted columns. Each record also contains ``node2comp``, mapping
        original node indices to contracted component indices.
    """
    if max_iterations is not None and max_iterations < 1:
        raise ValueError("max_iterations must be at least 1 or None")
    ifc_params = copy.deepcopy(ifc_params)
    refine_params = copy.deepcopy(refine_params)
    resolution = float(resolution)
    if not np.isfinite(resolution) or resolution < 0:
        raise ValueError("resolution must be a finite nonnegative value.")
    must_link = list(must_link or [])
    cannot_link = list(cannot_link or [])
    if stopping_window < 1:
        raise ValueError("stopping_window must be at least 1")

    # normalize refinement configurations
    refine_func = refine_params.get("refine_func")
    refine_kwargs = dict(refine_params.get("kwargs") or {})

    if use_refined_column and not callable(refine_func):
        raise ValueError(
            "use_refined_column=True requires refine_params['refine_func']."
        )

    # Make generator arguments optional.
    ifc_params["args"] = dict(ifc_params.get("args") or {})

    # drop seed values from parameters if included
    ifc_params["args"].pop("seed", None)
    refine_kwargs.pop("seed", None)
    subproblem_params = (
        {} if subproblem_params is None else dict(subproblem_params)
    )

    # validate warm start when necessary
    columns = [] if columns is None else list(columns)
    f_stars = None if f_stars is None else list(f_stars)

    if columns and f_stars is None:
        raise ValueError(
            "Warm-start f_stars are required when columns are supplied."
        )

    if f_stars is not None and len(columns) != len(f_stars):
        raise ValueError(
            "Warm-start columns and f_stars must have the same length."
        )

    if not columns:
        missing = {"generator", "num"} - ifc_params.keys()
        if missing:
            raise ValueError(
                f"ifc_params is missing required keys: {sorted(missing)}"
            )

        if not callable(ifc_params["generator"]):
            raise ValueError("ifc_params['generator'] must be callable.")

        if ifc_params["num"] < 1:
            raise ValueError("ifc_params['num'] must be at least 1.")

    additional_constraints = (
        {} if additional_constraints is None else dict(additional_constraints)
    )

    if contract_graph and additional_constraints.get("LB"):
        # TODO: Support load-balancing contraction by carrying original
        # component sizes as vertex weights through pricing and refinement.
        raise ValueError(
            "Graph contraction is not supported for load-balancing "
            "decomposition until component-size vertex weights are implemented."
        )

    # contract graph if necessary
    if contract_graph and (must_link or additional_constraints.get("worthy_edges")):
        A, node2comp = contract_adj_matrix_new(A, additional_constraints.get("worthy_edges"), must_link)
        a = A.sum(axis=0)
        m = np.sum(a)
        additional_constraints["worthy_edges"] = None
        cannot_link = contract_node_pairs(
            cannot_link,
            node2comp,
            relation_name="cannot-link",
            reject_internal=True,
        )

        generator_args = ifc_params.get("args", {})
        if "N" in ifc_params.get("args", {}):
            generator_args["N"] = np.shape(A)[0]
        if "cannot_link" in generator_args:
            generator_args["cannot_link"] = contract_node_pairs(
                generator_args["cannot_link"],
                node2comp,
                relation_name="initial-column cannot-link",
                reject_internal=True,
            )
        if "must_link" in generator_args:
            generator_args["must_link"] = []

        if columns is not None and len(columns) > 0:
            columns = [
                contract_partition_matrix(column, node2comp)
                for column in columns
            ]
            # Recompute scores so the column pool and contracted objective
            # cannot retain mismatched original-graph bookkeeping.
            f_stars = [
                compute_f_star(A, a, m, column, gamma=resolution)
                for column in columns
            ]

        if A.shape == (1, 1):
            coarse_partition = np.ones((1, 1), dtype=int)
            score = (
                compute_f_star(A, a, m, coarse_partition, gamma=resolution)
                if m > 0
                else 0.0
            )
            return [{
                "lambda_sol": [1.0],
                "master_obj_val": score,
                "z_sol": expand_z_matrix(coarse_partition, node2comp),
                "heuristic_col": None,
                "sub_obj_val": None,
                "columns": [coarse_partition],
                "f_stars": [score],
                "node2comp": node2comp.copy(),
            }]
    else:
        node2comp = None

    # deque to track flat pricing for termination
    SUB_OBJS = deque(maxlen=stopping_window)

    if (columns is not None and f_stars is not None) and len(columns) > 0:
        # initialize from parameters
        Z_star = list(columns)
        if resolution != 1.0:
            # Caller-supplied scores do not record the resolution at which
            # they were computed, so non-default runs rescore their columns.
            f_stars = [
                compute_f_star(A, a, m, column, gamma=resolution)
                for column in Z_star
            ]
    else:
        # generate initial feasible columns
        ifc_generator = ifc_params["generator"]
        feasible_columns = ifc_generator(**ifc_params["args"], seed=seed)

        # without feasible columns, terminate
        if len(feasible_columns) == 0:
            if verbose != -1:
                print("A feasible initial partition cannot be generated.")
            return None

        # initialize columns and their scores
        if ifc_params["num"] == 1:
            initial_z = feasible_columns[0]
            f_star_initial = compute_f_star(
                A, a, m, initial_z, gamma=resolution
            )
            Z_star = [initial_z]
            f_stars = [f_star_initial]
        else:
            Z_star = feasible_columns[:ifc_params["num"]]
            f_stars = [
                compute_f_star(A, a, m, col, gamma=resolution)
                for col in Z_star
            ]

    results = []

    # main column generation loop
    # for iteration in tqdm(range(max_iterations)):
    with tqdm(total=max_iterations, disable=disable_tqdm) as pbar:
        iteration = 1

        while True:
            if verbose != -1:
                print("\nIteration:", iteration)

            # run relaxed master problem (RMP)
            (lambda_sol,
            duals,
            master_obj_val
            ) = mp_function(
                A, a, m,
                Z_star, f_stars,
                cannot_link=cannot_link,
                must_link=[] if contract_graph else must_link,
                **additional_constraints,
                verbose=verbose,
                extract_dual=True,
            )

            # call it a day if RMP is infeasible
            if master_obj_val is None:
                return None

            if verbose != -1:
                print(duals)

                print("Master Obj:", master_obj_val)
                print("lambda: ", lambda_sol)

            pricing_kwargs = {
                **subproblem_params,
                "algo": algo,
                "package": package,
                "gamma": resolution,
                "verbose": verbose,
                "seed": seed,
            }
            signature = inspect.signature(sp_function)
            accepts_extra = any(
                parameter.kind == inspect.Parameter.VAR_KEYWORD
                for parameter in signature.parameters.values()
            )
            if resolution != 1.0 and not accepts_extra and "gamma" not in signature.parameters:
                raise ValueError(
                    f"{getattr(sp_function, '__name__', 'The selected subproblem')} "
                    "does not support a non-default resolution."
                )
            if not accepts_extra:
                pricing_kwargs = {
                    name: value
                    for name, value in pricing_kwargs.items()
                    if name in signature.parameters
                }
            sub_obj_val, z_sol = sp_function(
                A,
                a,
                m,
                duals,
                **pricing_kwargs,
            )

            results.append({
                "lambda_sol": lambda_sol,
                **duals,
                "master_obj_val": master_obj_val,
                "z_sol": expand_z_matrix(z_sol, node2comp) if contract_graph else z_sol,
                "sub_obj_val": sub_obj_val, "columns": Z_star.copy(),
                "f_stars": f_stars.copy()
            })
                        
            # check if the pricing problem generated a column with positive reduced cost.
            if sub_obj_val > tolerance: # and iteration == 0:
                if verbose != -1:
                    print("New column generated with f* =", sub_obj_val)
                    print(z_sol)
            else:
                if verbose != -1:
                    print(f"No improving column found (reduced cost: {sub_obj_val:.2g}); stopping column generation.")
                break

            # add priced out column and its score to the column set
            Z_star.append(z_sol)
            f_stars.append(
                compute_f_star(A, a, m, z_sol, gamma=resolution)
            )
            SUB_OBJS.append(sub_obj_val)

            # refine z_sol and potentially add to column list
            if callable(refine_func) and (use_refined_column or final_master_solve):
                # refine z_sol if needed in column generation or final master solve
                in_loop_kwargs = copy.deepcopy(refine_kwargs)
                if "shake_rounds" in inspect.signature(refine_func).parameters:
                    in_loop_kwargs["shake_rounds"] = 0
                heuristic_col = refine_func(
                    A=A,
                    partition=z_sol,
                    **in_loop_kwargs,
                    seed=seed
                )
                if heuristic_col is not None:
                    results[-1]["heuristic_col"] = heuristic_col
                    # add heuristic column to list if it is needed in column generation and is sufficiently different
                    if use_refined_column and sufficiently_different(heuristic_col, Z_star, dist_min=0.01):
                        Z_star.append(heuristic_col)
                        f_stars.append(
                            compute_f_star(
                                A,
                                a,
                                m,
                                heuristic_col,
                                gamma=resolution,
                            )
                        )
            
            # if pricing is flat for `stopping_window` iterations, stop.
            if (
                check_flat_pricing
                and len(SUB_OBJS) == stopping_window
                and (max(SUB_OBJS) - min(SUB_OBJS)) <= tolerance
            ):
                if verbose != -1:
                    print(f"Pricing seems flat after {stopping_window} iterations; Stopping column generation...")
                break

            iteration += 1
            pbar.update(1)
            if max_iterations is not None:
                if iteration > max_iterations:
                    break
    # generate heuristic column using wz after CG terminates and add it to results
    if refine_post_loop and callable(refine_func):
        if not disable_tqdm:
            print("Running post-loop refinement...")
        wz = np.zeros_like(Z_star[0]).astype(np.float64)
        for lambda_, column in zip(lambda_sol, Z_star):
            wz += (lambda_ * column)
        
        heuristic_col = refine_func(
                A=A,
                partition=wz,
                **refine_kwargs,
                seed=seed
            )
        if heuristic_col is not None:
            Z_star.append(heuristic_col)
            f_stars.append(
                compute_f_star(
                    A,
                    a,
                    m,
                    heuristic_col,
                    gamma=resolution,
                )
            )
            empty_duals = {k:None for k,v in duals.items()}

            results.append({
                "lambda_sol": None,
                **empty_duals,
                "master_obj_val": None,
                "z_sol": expand_z_matrix(heuristic_col, node2comp) if contract_graph else heuristic_col,
                "heuristic_col": heuristic_col,
                "sub_obj_val": None,
                "columns": Z_star.copy(),
                "f_stars": f_stars.copy()
            })

    if final_master_solve:
        if verbose != -1:
            print("Final Integer Master Solve...")
        if not use_refined_column:
            for record in results:
                if record["sub_obj_val"] is None:
                    continue
                heuristic_col = record.get("heuristic_col")
                if heuristic_col is not None:
                    Z_star.append(heuristic_col)
                    f_stars.append(
                        compute_f_star(
                            A,
                            a,
                            m,
                            heuristic_col,
                            gamma=resolution,
                        )
                    )

        (lambda_sol, master_obj_val) = mp_function(
            A, a, m,
            Z_star, f_stars,
            cannot_link=cannot_link,
            must_link=[] if contract_graph else must_link,
            **additional_constraints,
            verbose=verbose,
            extract_dual=False,
        )
        if lambda_sol is not None:
            z_sol = Z_star[np.argmax(lambda_sol)]
        else:
            if verbose != -1:
                print("Final master solve is infeasible.")
            z_sol = None

        empty_duals = {k:None for k, _ in duals.items()}

        results.append({
            "lambda_sol": lambda_sol,
            **empty_duals,
            "master_obj_val": master_obj_val,
            "z_sol": expand_z_matrix(z_sol, node2comp) if contract_graph else z_sol,
            "heuristic_col": None,
            "sub_obj_val": None,
            "columns": Z_star.copy(),
            "f_stars": f_stars.copy()
        })
    if node2comp is not None:
        for record in results:
            record["node2comp"] = node2comp.copy()
    return results
