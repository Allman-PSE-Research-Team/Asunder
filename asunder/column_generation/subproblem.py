"""Subproblem routines for CSD decomposition."""

from __future__ import annotations

import numpy as np

from asunder.algorithms.community import (
    probability_to_integer_labels,
    run_igraph,
    run_igraph_spinglass,
    run_lpa,
    run_modularity,
    run_signed_louvain,
)
from asunder.algorithms.louvain_modified import ModifiedLouvain
from asunder.algorithms.spectral import full_spectral_bisection
from asunder.column_generation.master import compute_f_star
from asunder.solvers import get_default_solver
from asunder.utils.graph import partition_vector_to_2d_matrix

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
    """Solve pricing heuristically via selected clustering backend.

    Returns:
        Tuple ``(reduced_cost, partition_matrix)``.
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
    """Solve pricing exactly as a binary ILP with transitivity constraints."""
    if ConcreteModel is None:
        raise ImportError("pyomo is required for ILP subproblem. Ensure base dependencies are installed.")
    solver = get_default_solver() if solver is None else solver
    I = np.shape(A)[0]

    model = ConcreteModel()
    model.I = RangeSet(0, I - 1)
    model.z = Var(model.I, model.I, domain=Binary)
    model.DiagonalUnity = Constraint(model.I, rule=lambda mdl, i: mdl.z[i, i] == 1)

    def z_symmetry_rule(mdl, i, j):
        if i < j:
            return mdl.z[i, j] == mdl.z[j, i]
        return Constraint.Skip

    model.ZSymmetry = Constraint(model.I, model.I, rule=z_symmetry_rule)

    def transitivity_rule(mdl, i, j, k):
        if len(set([i, j, k])) == 3:
            return mdl.z[i, j] + mdl.z[i, k] - mdl.z[j, k] <= 1
        return Constraint.Skip

    model.Transitivity = Constraint(model.I, model.I, model.I, rule=transitivity_rule)

    def sub_objective_rule(mdl):
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
    """Run in-package custom pricing heuristics (spectral/modified Louvain)."""
    assert algo in {"spectral", "full_louvain", "one_level_louvain"}
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
        else:
            z_sol = partition_vector_to_2d_matrix(louvain_model.labels_)
        metric = louvain_model.obj_val_
    else:
        z_sol, metric = full_spectral_bisection(A, a, m, dualW, refinement=False, verbose=verbose)
        if refine and refine_params is not None:
            z_refine = refine_params["refine_func"](A=A, partition=z_sol, **refine_params["kwargs"])
            if z_refine is not None:
                z_sol = z_refine

    sub_obj_val = metric - constant_terms
    return sub_obj_val, z_sol
