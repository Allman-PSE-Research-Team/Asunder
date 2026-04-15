"""Subproblem routines for CSD decomposition."""

from __future__ import annotations

import numpy as np

from asunder.base.algorithms.community import (
    probability_to_integer_labels,
    run_igraph,
    run_igraph_spinglass,
    run_lpa,
    run_modularity,
    run_signed_louvain,
)
from asunder.base.algorithms.louvain_modified import ModifiedLouvain
from asunder.base.algorithms.RCCS import search_partition_by_reduced_cost
from asunder.base.algorithms.spectral import full_spectral_bisection
from asunder.base.column_generation.master import compute_f_star
from asunder.base.utils.graph import partition_vector_to_2d_matrix
from asunder.solvers import get_default_solver

try:
    from pyomo.environ import (
        Binary,
        ConcreteModel,
        Constraint,
        Objective,
        RangeSet,
        TerminationCondition,
        Var,
        maximize,
        value,
    )
except Exception:  # pragma: no cover - optional dependency
    ConcreteModel = None


def heuristic_subproblem(
    A,
    a,
    m,
    duals,
    algo="louvain",
    package="networkx",
    refine=False,
    refine_params=None,
    verbose=False,
):
    """
    Solve pricing heuristically via selected clustering backend.
    
    Parameters
    ----------
    A : np.ndarray of int | float, shape (N, N)
        Adjacency / weight matrix.
    a : np.ndarray of int | float, shape (N,)
        Degree-like vector; defaults to row sums of the symmetrized adjacency.
    m : float
        Twice the total weight in the graph.
    duals : Dict[str, np.ndarray or float]
        Dual terms used to modify the community detection objective. 2D, 1D and scalar dual values are supported.
        Missing entries are treated as zeros.
    algo : str
        Name of third-party heuristic subproblem used to replace the ILP subproblem.
    package : str
        Package from which third-party heuristic subproblem is selected. See ``CSD_decomposition`` for more detail.
    refine : bool
        Boolean value that determines whether a refinement operation is run in the subproblem.
    refine_params : dict[str, callable or dict]
        Refinement function and its corresponding arguments.
    verbose : int or bool
        Controls the level of detail in the printed output.
        ``-1``: No output
        ``False`` | ``0``: Minimal output
        ``True`` | ``1``: Detailed output
    
    Returns
    -------
    Any
        Computed result.
    """
    I = A.shape[0]
    dualW = np.zeros_like(A)
    constant_terms = 0
    for _, dual in duals.items():
        if isinstance(dual, np.ndarray):
            if dual.ndim == 1:
                temp_dual = np.zeros_like(A)
                for i in range(I):
                    for j in range(I):
                        temp_dual[i, j] = 0.5 * (dual[i] + dual[j])
                dualW += temp_dual
            elif dual.ndim == 2:
                dualW += dual if np.array_equal(dual, dual.T) else (dual + dual.T) / 2
        elif isinstance(dual, float):
            constant_terms += dual

    modA = A - (m * dualW)
    mod_a = modA.sum(axis=0)
    mod_m = np.sum(mod_a)
    modA_positive = modA.copy()
    modA_positive[modA_positive < 0] = 0

    if package == "igraph" and algo != "lpa":
        zii, metric = run_igraph(modA, algo=algo, resolution=1)
    elif algo == "spinglass":
        zii = run_igraph_spinglass(modA)
        metric = compute_f_star(modA, mod_a, mod_m, zii)
    elif algo == "signed_louvain":
        zii, metric = run_signed_louvain(modA)
    elif algo == "lpa":
        if package == "sknetwork":
            zii, metric = run_lpa(modA)
        elif package == "igraph":
            zii, metric = run_igraph(modA, algo=algo)
        else:
            raise NotImplementedError(f"Invalid package entered: {package}")
    else:
        zii, metric = run_modularity(
            modA_positive,
            algo=algo,
            package=package,
            refine=refine,
            refine_params=refine_params,
            resolution=1,
            verbose=verbose,
        )
        if metric is None:
            mod_a_p = modA_positive.sum(axis=0)
            mod_m_p = np.sum(mod_a_p)
            modB_p = (modA_positive / mod_m_p) - np.outer(mod_a_p, mod_a_p) / (mod_m_p**2)
            metric = np.sum(modB_p * zii)
    sub_obj_val = metric - constant_terms
    return sub_obj_val, zii


def solve_subproblem(A, a, m, duals, use_augmented_adjacency=False, verbose=False, solver=None):
    """
    Solve pricing exactly as a binary ILP with transitivity constraints.
    
    Parameters
    ----------
    A : np.ndarray of int | float, shape (N, N)
        Adjacency / weight matrix.
    a : np.ndarray of int | float, shape (N,)
        Degree-like vector; defaults to row sums of the symmetrized adjacency.
    m : float
        Twice the total weight in the graph.
    duals : Dict[str, np.ndarray or float]
        Dual terms used to modify the community detection objective. 2D, 1D and scalar dual values are supported.
        Missing entries are treated as zeros.
    use_augmented_adjacency : bool
        Determines whether augmented adjacency is used with the ILP or not. Defaults to ``False``.
    verbose : int or bool
        Controls the level of detail in the printed output.
        ``-1``: No output
        ``False`` | ``0``: Minimal output
        ``True`` | ``1``: Detailed output
    solver : Any
        Solver object.
    
    Returns
    -------
    Any
        Computed result.
    """
    if ConcreteModel is None:
        raise ImportError("pyomo is required for ILP subproblem. Ensure base dependencies are installed.")
    solver = get_default_solver() if solver is None else solver
    I = np.shape(A)[0]

    model = ConcreteModel()
    model.I = RangeSet(0, I - 1)
    model.z = Var(model.I, model.I, domain=Binary)
    model.DiagonalUnity = Constraint(model.I, rule=lambda mdl, i: mdl.z[i, i] == 1)

    def z_symmetry_rule(mdl, i, j):
        """
        Z symmetry rule.
        """
        if i < j:
            return mdl.z[i, j] == mdl.z[j, i]
        return Constraint.Skip

    model.ZSymmetry = Constraint(model.I, model.I, rule=z_symmetry_rule)

    def transitivity_rule(mdl, i, j, k):
        """
        Transitivity rule.
        """
        if len(set([i, j, k])) == 3:
            return mdl.z[i, j] + mdl.z[i, k] - mdl.z[j, k] <= 1
        return Constraint.Skip

    model.Transitivity = Constraint(model.I, model.I, model.I, rule=transitivity_rule)

    def sub_objective_rule(mdl):
        """
        Subproblem objective rule.
        """
        dualW = np.zeros_like(A)
        constant_terms = 0
        for _, dual in duals.items():
            if isinstance(dual, np.ndarray):
                if dual.ndim == 1:
                    temp_dual = np.zeros_like(A)
                    for i in range(I):
                        for j in range(I):
                            temp_dual[i, j] = 0.5 * (dual[i] + dual[j])
                    dualW += temp_dual
                elif dual.ndim == 2:
                    dualW += dual if np.array_equal(dual, dual.T) else (dual + dual.T) / 2
            elif isinstance(dual, float):
                constant_terms += dual

        if use_augmented_adjacency:
            modA = A - (m * dualW)
            mod_a = modA.sum(axis=1)
            mod_m = mod_a.sum()
            M = sum(
                ((modA[i, j] / mod_m) - ((mod_a[i] * mod_a[j]) / (mod_m**2))) * mdl.z[i, j]
                for i in mdl.I
                for j in mdl.I
            )
        else:
            M = sum(
                ((A[i, j] / m) - ((a[i] * a[j]) / (m**2)) - dualW[i, j]) * mdl.z[i, j]
                for i in mdl.I
                for j in mdl.I
            )
        return M - constant_terms

    model.OBJ = Objective(rule=sub_objective_rule, sense=maximize)
    res = solver.solve(model, tee=False)
    if verbose != -1 and res.solver.termination_condition != TerminationCondition.optimal:
        lb = getattr(res.problem, "lower_bound", None)
        ub = getattr(res.problem, "upper_bound", None)
        print(f"[Pricing] bounds: lower={lb}, upper={ub}")
    z_sol = np.array([[value(model.z[i, j]) for j in model.I] for i in model.I])
    return value(model.OBJ), z_sol


def custom_heuristic_subproblem(
    A,
    a,
    m,
    duals,
    algo="full_louvain",
    verbose=False,
    refine=False,
    refine_params=None,
    max_iterations=50,
    tolerance=1e-8,
):
    """
    Run in-package custom pricing heuristics (spectral/modified Louvain).
    
    Parameters
    ----------
    A : np.ndarray of int | float, shape (N, N)
        Adjacency / weight matrix.
    a : np.ndarray of int | float, shape (N,)
        Degree-like vector; defaults to row sums of the symmetrized adjacency.
    m : float
        Twice the total weight in the graph.
    duals : Dict[str, np.ndarray or float]
        Dual terms used to modify the community detection objective. 2D, 1D and scalar dual values are supported.
        Missing entries are treated as zeros.
    algo : str
        Name of custom heuristic subproblem used to replace the ILP subproblem.
    verbose : int or bool
        Controls the level of detail in the printed output.
        ``-1``: No output
        ``False`` | ``0``: Minimal output
        ``True`` | ``1``: Detailed output
    refine : bool
        Boolean value that determines whether a refinement operation is run in the subproblem.
    refine_params : dict[str, callable or dict]
        Refinement function and its corresponding arguments.
    max_iterations : int
        Maximum number of iterations.
    tolerance : float
        Tolerance value.
    
    Returns
    -------
    Any
        Computed result.
    """
    assert algo in {"spectral", "full_louvain", "one_level_louvain", "RCCS"}
    I = A.shape[0]
    constant_terms = 0
    dualW = np.zeros_like(A)
    for _, dual in duals.items():
        if isinstance(dual, np.ndarray):
            if dual.ndim == 1:
                temp_dual = np.zeros_like(A)
                for i in range(I):
                    for j in range(I):
                        temp_dual[i, j] = 0.5 * (dual[i] + dual[j])
                dualW += temp_dual
            elif dual.ndim == 2:
                dualW += dual if np.array_equal(dual, dual.T) else (dual + dual.T) / 2
        elif isinstance(dual, float):
            constant_terms += dual

    if "louvain" in algo:
        louvain_model = ModifiedLouvain(random_state=None)
        if algo.startswith("full_"):
            louvain_model.fit(A, duals)
        else:
            louvain_model.fit_modified_one_level(A, duals, max_iter=max_iterations, tol=tolerance)
        if refine:
            ilabels = probability_to_integer_labels(louvain_model.predict_proba(), "gaussian_mixture")
            z_sol = partition_vector_to_2d_matrix(ilabels)
            metric = None
        else:
            z_sol = partition_vector_to_2d_matrix(louvain_model.labels_)
            metric = louvain_model.obj_val_
    else:
        if algo == "RCCS":
            res = search_partition_by_reduced_cost(adjacency=A, duals=duals)#, random_seed=seed)
            best_labels = res["best_labels"]
            z_sol = partition_vector_to_2d_matrix(best_labels)
            metric = res["best_reduced_cost"] + constant_terms # for normalization sake
        else:
            z_sol, metric = full_spectral_bisection(A, a, m, dualW, refinement=False, verbose=verbose)
        if refine and refine_params is not None:
            z_refine = refine_params["refine_func"](A=A, partition=z_sol, **refine_params["kwargs"])
            if z_refine is not None:
                z_sol = z_refine
                metric = None
    if metric is None:
        modularity_contribution = compute_f_star(A, a, m, z_sol)
        dual_contribution = (dualW * z_sol).sum()
        metric = modularity_contribution - dual_contribution
    sub_obj_val = metric - constant_terms
    return sub_obj_val, z_sol
