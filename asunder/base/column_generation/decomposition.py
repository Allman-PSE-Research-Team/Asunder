"""Column generation decomposition orchestration."""
from __future__ import annotations

from collections import defaultdict, deque

import numpy as np
from tqdm.auto import tqdm

from asunder.base.column_generation.master import compute_f_star
from asunder.base.utils.graph import (
    contract_adj_matrix_new,
    expand_z_matrix,
    sufficiently_different,
)


def CSD_decomposition(
    A, a, m,
    mp_function,
    sp_function,
    columns=None, f_stars=None,
    must_link=[], cannot_link=[],
    additional_constraints=defaultdict(lambda: None),
    contract_graph=False,
    stopping_window=5,
    check_flat_pricing=True,
    algo="louvain",
    package="sknetwork",
    extract_dual=False,
    # initial feasible column generator
    ifc_params: dict = {},
    # refinement
    refine_in_subproblem=False,
    refine_params: dict={},
    use_refined_column=False,
    final_master_solve=True,
    max_iterations=1000, tolerance=1e-10, verbose=False,
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

    Returns
    -------
    results : list[dict]
        Column-generation iteration records. Each dictionary may include
        ``lambda_sol``, dual terms, ``master_obj_val``, ``z_sol``,
        ``sub_obj_val``, ``columns``, ``f_stars``, and ``heuristic_col``.
    """

    # cold start
    if columns is None:
        columns = []

    # contract graph if necessary
    if contract_graph and (must_link or additional_constraints["worthy_edges"]):
        A, node2comp = contract_adj_matrix_new(A, additional_constraints["worthy_edges"], must_link)
        a = A.sum(axis=0)
        m = np.sum(a)
        additional_constraints["worthy_edges"] = None
        if "N" in ifc_params["args"]:
            ifc_params["args"]["N"] = np.shape(A)[0]
    else:
        node2comp = None

    # deque to track flat pricing for termination
    SUB_OBJS = deque(maxlen=stopping_window)

    if (columns is not None and f_stars is not None) and len(columns) > 0:
        # initialize from parameters
        Z_star = columns
        f_stars = f_stars
    else:
        # generate initial feasible columns
        ifc_generator = ifc_params["generator"]
        feasible_columns = ifc_generator(**ifc_params["args"])

        # without feasible columns, terminate
        if len(feasible_columns) == 0:
            if verbose != -1:
                print("A feasible initial partition cannot be generated.")
            return None, None, None

        # initialize columns and their scores
        if ifc_params["num"] == 1:
            initial_z = feasible_columns[0]
            f_star_initial = compute_f_star(A, a, m, initial_z)
            Z_star = [initial_z]
            f_stars = [f_star_initial]
        else:
            Z_star = feasible_columns[:ifc_params["num"]]
            f_stars = [
                compute_f_star(A, a, m, col) for col in Z_star
            ]

    results = []

    # main column generation loop
    # for iteration in tqdm(range(max_iterations)):
    with tqdm(total=max_iterations, disable=(verbose == -1)) as pbar:
        iteration = 1

        while True:
            if verbose != -1:
                print("\nIteration:", iteration + 1)

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
                extract_dual=extract_dual,
            )

            # call it a day if RMP is infeasible
            if master_obj_val is None:
                return None, None, None

            if verbose != -1:
                print(duals)

                print("Master Obj:", master_obj_val)
                print("lambda: ", lambda_sol)

            try:
                if algo in {"spectral", "full_louvain", "one_level_louvain"}:
                    # uses custom heuristic subproblem
                    sub_obj_val, z_sol = sp_function(
                        A, a, m, duals, algo=algo,
                        refine=refine_in_subproblem,
                        refine_params=refine_params,
                        verbose=verbose
                    )
                else:
                    # uses package based heuristic subproblem
                    sub_obj_val, z_sol = sp_function(
                        A, a, m, duals,
                        algo=algo, package=package,
                        refine=refine_in_subproblem,
                        refine_params=refine_params,
                        verbose=verbose
                    )
            except Exception:
                # uses ILP subproblem
                sub_obj_val, z_sol = sp_function(
                    A, a, m, duals, verbose=verbose
                )

            if verbose != -1:
                print(f"Subproblem obj: {sub_obj_val}")
                print(z_sol)

            results.append({
                "lambda_sol": lambda_sol,
                **duals,
                "master_obj_val": master_obj_val,
                "z_sol": expand_z_matrix(z_sol, node2comp) if contract_graph else z_sol,
                "sub_obj_val": sub_obj_val, "columns": Z_star.copy(),
                "f_stars": f_stars
            })

            # add priced out column and its score to the column set
            Z_star.append(z_sol)
            f_stars.append(compute_f_star(A, a, m, z_sol))
            SUB_OBJS.append(sub_obj_val)

            # refine z_sol and potentially add to column list
            try:
                if use_refined_column or final_master_solve:
                    # refine z_sol if needed in column generation or final master solve
                    heuristic_col = refine_params["refine_func"](
                        A=A,
                        partition=z_sol,
                        **refine_params["kwargs"]
                    )
                    if heuristic_col is not None:
                        results[-1]["heuristic_col"] = heuristic_col
                        # add heuristic column to list if it is needed in column generation and is sufficiently different
                        if use_refined_column and sufficiently_different(heuristic_col, Z_star, dist_min=0.01):
                            Z_star.append(heuristic_col)
                            f_stars.append(compute_f_star(A, a, m, heuristic_col))
            except KeyError as e:
                if verbose != -1:
                    print(f"Exception: {e}")

            # check if the pricing problem generates a column with positive reduced cost.
            if sub_obj_val > tolerance: # and iteration == 0:
                if verbose != -1:
                    print("New column generated with f* =", sub_obj_val)
            else:
                if verbose != -1:
                    print(f"No improving column found (reduced cost: {sub_obj_val:.2g}); stopping column generation.")
                break

            # if pricing is flat for `stopping_window` iterations, stop.
            if check_flat_pricing:
                try:
                    if (max(SUB_OBJS) - min(SUB_OBJS)) <= tolerance and iteration > stopping_window:
                        if verbose != -1:
                            print(f"Pricing seems flat after {stopping_window} iterations; Stopping column generation...")
                        break
                    else:
                        ...
                except Exception:
                    ...
            iteration += 1
            pbar.update(1)
            if max_iterations is not None:
                if iteration > max_iterations:
                    break
    # generate heuristic column using wz after CG terminates and add it to results
    if use_refined_column:
        wz = np.zeros_like(Z_star[0]).astype(np.float64)
        for lambda_, column in zip(lambda_sol, Z_star):
            wz += (lambda_ * column)
        heuristic_col = refine_params["refine_func"](
                A=A,
                partition=wz,
                **refine_params["kwargs"]
            )
        if heuristic_col is not None:
            Z_star.append(heuristic_col)
            f_stars.append(compute_f_star(A, a, m, heuristic_col))
            empty_duals = {k:None for k,v in duals.items()}

            results.append({
                "lambda_sol": None,
                **empty_duals,
                "master_obj_val": None,
                "z_sol": expand_z_matrix(heuristic_col, node2comp) if contract_graph else heuristic_col,
                "heuristic_col": heuristic_col,
                "sub_obj_val": None, "columns": Z_star.copy(),
                "f_stars": f_stars
            })

    if final_master_solve:
        if verbose != -1:
            print("Final Integer Master Solve...")
        if not use_refined_column:
            for iter in results:
                if iter["sub_obj_val"] is None:
                    continue
                try:
                    Z_star.append(iter["heuristic_col"])
                    f_stars.append(compute_f_star(A, a, m, iter["heuristic_col"]))
                except Exception:
                    pass

        (lambda_sol, master_obj_val) = mp_function(
            A, a, m,
            Z_star, f_stars,
            cannot_link=cannot_link,
            must_link=[] if contract_graph else must_link,
            **additional_constraints,
            verbose=verbose
        )

        z_sol = Z_star[np.argmax(lambda_sol)]

        if verbose != -1:
            print(z_sol)

        empty_duals = {k:None for k,v in duals.items()}

        results.append({
            "lambda_sol": lambda_sol,
            **empty_duals,
            "master_obj_val": master_obj_val,
            "z_sol": expand_z_matrix(z_sol, node2comp) if contract_graph else z_sol,
            "heuristic_col": None,
            "sub_obj_val": None, "columns": Z_star.copy(),
            "f_stars": f_stars
        })
    return results
