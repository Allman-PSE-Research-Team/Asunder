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
    """
    Internal helper for the require pyomo check.

    Raises
    ------
    ImportError
        If Pyomo is not importable in the current environment.
    """
    if ConcreteModel is None:
        raise ImportError("pyomo is required for master/subproblem optimization. Ensure base dependencies are installed.")


def compute_f_star(A, a, m, z, gamma=1.0):
    """
    Compute column/partiton score used by the restricted master objective.
    
    Parameters
    ----------
    A : np.ndarray of int | float, shape (N, N)
        Adjacency / weight matrix.
    a : np.ndarray of int | float, shape (N,)
        Degree-like vector; defaults to row sums of the symmetrized adjacency.
    m : float
        Twice the total weight in the graph.
    z : ndarray of int | float, shape (N, N)
        Graph partition.
    gamma : float
        Resolution parameter.
    
    Returns
    -------
    metric: float
        Modularity score.
    """
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
    """
    Solve the restricted master problem for the current column pool.
    
    Parameters
    ----------
    A : np.ndarray of int | float, shape (N, N)
        Adjacency / weight matrix.
    a : np.ndarray of int | float, shape (N,)
        Degree-like vector; defaults to row sums of the symmetrized adjacency.
    m : float
        Twice the total weight in the graph.
    Z_star : list[ndarray of int]
        List of columns.
    f_stars : list[float]
        Objective values computed using the existing columns.
    cannot_link : list[tuple[int, int]] or None
        List of node pairs that must not be together.
    must_link : list[tuple[int, int]] or None
        List of node pairs that must be together.
    worthy_edges : list[tuple[int, int]] or None
        List of edges that are allowed to connect different communities
    extract_dual : bool
        Boolean that determines whether we extract duals from the master problem or not.
    verbose : int or bool
        Controls the level of detail in the printed output.
        ``-1``: No output
        ``False`` | ``0``: Minimal output
        ``True`` | ``1``: Detailed output
    solver : Any
        Solver object.
    
    Returns
    -------
    lambda_sol: list or ndarray of float
        A list/vector which sums to ``1`` that indicates what weight is assigned to each column (and by implication, what columns are active).
    duals: dict[str, ndarray or float]
        Dual values computed from the master problem. This could be a 1D array, 2D array or a float.
    master_obj_val: float
        The objective value of the master problem.
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
        """
        Ensures a convex combination of columns. Ideally, only one column would be active.
        """
        return sum(mdl.lmbd[c] for c in mdl.C) == 1

    model.OneColumn = Constraint(rule=one_column_rule)

    if worthy_edges:
        worthy_edges = tuple(tuple(sorted(edge)) for edge in worthy_edges)

        def worthy_edge_rule(mdl, i, j):
            """
            Enforce worthy-edge consistency for a candidate edge pair.
            """
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
            """
            Enforce a cannot-link pair in the master problem.
            """
            return sum(mdl.lmbd[c] * Z_star[c][i, j] for c in mdl.C) == 0

        model.CannotLink = Constraint(cannot_link_pairs, rule=cannot_link_rule)
    else:
        cannot_link_pairs = []

    if must_link:
        must_link_pairs = tuple(tuple(sorted(pair)) for pair in must_link)

        def must_link_rule(mdl, i, j):
            """
            Enforce a must-link pair in the master problem.
            """
            return sum(mdl.lmbd[c] * Z_star[c][i, j] for c in mdl.C) == 1

        model.MustLink = Constraint(must_link_pairs, rule=must_link_rule)
    else:
        must_link_pairs = []

    def master_objective_function(mdl):
        """
        Master objective function.
        """
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
