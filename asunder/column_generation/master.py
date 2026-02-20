"""Master problem and score utilities for CSD decomposition."""

from __future__ import annotations

import numpy as np

from asunder.solvers import get_default_solver

try:
    from pyomo.environ import (
        Binary,
        ConcreteModel,
        Constraint,
        NonNegativeReals,
        Objective,
        RangeSet,
        Set,
        Suffix,
        TerminationCondition,
        Var,
        maximize,
        value,
    )
except Exception:  # pragma: no cover - optional dependency
    ConcreteModel = None


def _require_pyomo():
    if ConcreteModel is None:
        raise ImportError("pyomo is required for master/subproblem optimization. Ensure base dependencies are installed.")


def compute_f_star(A, a, m, z, gamma=1.0, algo="louvain"):
    """Compute column score used by the restricted master objective."""
    metric = np.sum((A / m - np.outer(a, a) / (m * m)) * z)
    return metric


def solve_master_problem(
    A,
    a,
    m,
    Z_star,
    f_stars,
    cannot_link=None,
    must_link=None,
    worthy_edges=None,
    extract_dual=False,
    verbose=False,
    solver=None,
):
    """Solve the restricted master problem for the current column pool.

    Returns:
        ``(lambda_sol, master_obj)`` when ``extract_dual=False``.
        ``(lambda_sol, duals, master_obj)`` when ``extract_dual=True``.
    """
    _require_pyomo()
    cannot_link = [] if cannot_link is None else cannot_link
    must_link = [] if must_link is None else must_link
    solver = get_default_solver() if solver is None else solver

    model = ConcreteModel()
    I = np.shape(Z_star[0])[0]
    model.I = RangeSet(0, I - 1)
    model.C = Set(initialize=list(range(len(Z_star))))
    if extract_dual:
        model.lmbd = Var(model.C, domain=NonNegativeReals, bounds=(0, None), initialize=0)
    else:
        model.lmbd = Var(model.C, domain=Binary, initialize=0)

    def one_column_rule(mdl):
        return sum(mdl.lmbd[c] for c in mdl.C) == 1

    model.OneColumn = Constraint(rule=one_column_rule)

    if worthy_edges:
        worthy_edges = tuple(tuple(sorted(edge)) for edge in worthy_edges)

        def worthy_edge_rule(mdl, i, j):
            if ((i, j) in worthy_edges) or ((j, i) in worthy_edges):
                return Constraint.Skip
            return sum(mdl.lmbd[c] * Z_star[c][i, j] for c in mdl.C) == 1

        all_edges = np.argwhere(np.triu(A, k=0) >= 1).tolist()
        model.WorthyEdges = Constraint(all_edges, rule=worthy_edge_rule)
    else:
        all_edges = []

    if cannot_link:
        cannot_link_pairs = tuple(tuple(sorted(pair)) for pair in cannot_link)

        def cannot_link_rule(mdl, i, j):
            return sum(mdl.lmbd[c] * Z_star[c][i, j] for c in mdl.C) == 0

        model.CannotLink = Constraint(cannot_link_pairs, rule=cannot_link_rule)
    else:
        cannot_link_pairs = []

    if must_link:
        must_link_pairs = tuple(tuple(sorted(pair)) for pair in must_link)

        def must_link_rule(mdl, i, j):
            return sum(mdl.lmbd[c] * Z_star[c][i, j] for c in mdl.C) == 1

        model.MustLink = Constraint(must_link_pairs, rule=must_link_rule)
    else:
        must_link_pairs = []

    def master_objective_function(mdl):
        return sum(f_stars[c] * mdl.lmbd[c] for c in mdl.C)

    model.OBJ = Objective(rule=master_objective_function, sense=maximize)
    if extract_dual:
        model.dual = Suffix(direction=Suffix.IMPORT)

    res = solver.solve(model, tee=bool(verbose is True))
    lambda_sol = [value(model.lmbd[c]) for c in model.C]
    master_obj_val = value(model.OBJ)
    if res.solver.termination_condition == TerminationCondition.infeasible:
        master_obj_val = None

    if not extract_dual:
        return lambda_sol, master_obj_val

    duals = {"mu_dual": model.dual.get(model.OneColumn, 0)}
    if cannot_link:
        tau_dual = np.zeros((I, I))
        for (i, j) in cannot_link_pairs:
            tau_dual[i, j] = model.dual.get(model.CannotLink[i, j], 0)
        duals["tau_dual"] = tau_dual

    if must_link:
        gamma_dual = np.zeros((I, I))
        for (i, j) in must_link_pairs:
            gamma_dual[i, j] = model.dual.get(model.MustLink[i, j], 0)
        duals["gamma_dual"] = gamma_dual

    if worthy_edges:
        pi_dual = np.zeros((I, I))
        for (i, j) in all_edges:
            if not ((i, j) in worthy_edges or (j, i) in worthy_edges):
                pi_dual[i, j] = model.dual.get(model.WorthyEdges[i, j], 0)
        duals["pi_dual"] = pi_dual

    return lambda_sol, duals, master_obj_val
