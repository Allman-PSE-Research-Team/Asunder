"""Column generation decomposition orchestration."""
from __future__ import annotations

from collections import defaultdict, deque

import numpy as np
from tqdm.auto import tqdm

from asunder.column_generation.master import compute_f_star
from asunder.utils.graph import contract_adj_matrix_new, expand_z_matrix, sufficiently_different


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
    Function that does column generation and refinement given a master and subproblem function.
    additional_constraints: defaulltdict(lambda: None)
    extract_dual        : relaxes master problem to become an LP if true
    ifc_params: dict with
        num             : number of initial feasible columns to use
        generator       : function that generates initial feasible partition
        args            : arguments to pass to initial feasible column generator
    refine_in_subproblem: refine partition using random walk confidence scores in heuristic subproblem
    refine_params       : parameters for refining heuristic subproblem result
    use_refined_column: refine partition/column as a part of the column generation process
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
            f_star_initial = compute_f_star(A, a, m, initial_z, algo=algo)
            Z_star = [initial_z]
            f_stars = [f_star_initial]
        else:
            Z_star = feasible_columns[:ifc_params["num"]]
            f_stars = [
                compute_f_star(A, a, m, col, algo=algo) for col in Z_star
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
            f_stars.append(compute_f_star(A, a, m, z_sol, algo=algo))
            SUB_OBJS.append(sub_obj_val)

            # refine z_sol and potentially add to column list
            try:
                # refine z_sol and potentially add to column list
                heuristic_col = refine_params["refine_func"](
                    A=A,
                    partition=z_sol,
                    **refine_params["kwargs"]
                )
                if heuristic_col is not None:
                    results[-1]["heuristic_col"] = heuristic_col
                    if use_refined_column and sufficiently_different(heuristic_col, Z_star, dist_min=0.01):
                        Z_star.append(heuristic_col)
                        f_stars.append(compute_f_star(A, a, m, heuristic_col, algo=algo))
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
            f_stars.append(compute_f_star(A, a, m, heuristic_col, algo=algo))
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
                    f_stars.append(compute_f_star(A, a, m, iter["heuristic_col"], algo=algo))
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
