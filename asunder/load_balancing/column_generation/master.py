"""Master problem and score utilities for CSD decomposition."""

from __future__ import annotations

import numpy as np

from asunder.base.column_generation.master import _require_pyomo
from asunder.load_balancing.utils.balance import resolve_balance_bounds
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

def solve_master_problem(
    A,
    a,
    m,
    Z_star,
    f_stars,
    LB=True, 
    R=1, 
    K=2, 
    R_bounds=None,
    cannot_link=None,
    must_link=None,
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
    LB : bool
        Whether load balancing constraints are activated or not.
    R : int
        Width of the allowed cluster-size range. Also corresponds to the load balance tightness (smaller R implies tighter load balance).
        For a selected cluster count, the lower and upper bounds are computed from the corresponding
        balanced range rule.
    K : int | None
        Number of communities.
    R_bounds : tuple[int, int] | None
        Minimum and maximum number of nodes per community (community size constraint).
    cannot_link : list[tuple[int, int]] or None
        List of node pairs that must not be together.
    must_link : list[tuple[int, int]] or None
        List of node pairs that must be together.
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

    if LB:
        if K is None:
            raise ValueError("K is required when load-balancing constraints are active.")
        if R == 0 and R_bounds is None and I % K != 0:
            raise ValueError(
                "Infeasible R and K combination, given the number of nodes."
            )
        R_min, R_max = resolve_balance_bounds(I, K, R, R_bounds)

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

    if LB:
        model.Rmin = Constraint(
            model.I, rule=lambda m, i: R_min <= sum(
                sum(m.lmbd[c] * Z_star[c][i, j] for c in m.C) for j in m.I
            )
        )

        model.Rmax = Constraint(
            model.I, rule=lambda m, i: R_max >= sum(
                sum(m.lmbd[c] * Z_star[c][i, j] for c in m.C) for j in m.I
            )
        )

    def master_objective_function(mdl):
        """
        Master objective function.
        """
        return sum(f_stars[c] * mdl.lmbd[c] for c in mdl.C)

    model.OBJ = Objective(rule=master_objective_function, sense=maximize)
    if extract_dual:
        model.dual = Suffix(direction=Suffix.IMPORT)

    res = solver.solve(model, tee=bool(verbose is True))
    if res.solver.termination_condition == TerminationCondition.infeasible:
        return (None, None, None) if extract_dual else (None, None)
    lambda_sol = [value(model.lmbd[c]) for c in model.C]
    master_obj_val = value(model.OBJ)

    if not extract_dual:
        return lambda_sol, master_obj_val

    duals = {"mu_dual": model.dual.get(model.OneColumn, 0)}

    if LB:
        # Build tau_dual array from the Rmin constraint.
        tau_dual = np.zeros((I,))
        for j in model.I:
            tau_dual[j] = model.dual.get(model.Rmin[j], 0)
        # Build pi_dual array from the Rmax constraint.
        pi_dual = np.zeros((I,))
        for j in model.I:
            pi_dual[j] = model.dual.get(model.Rmax[j], 0)
        duals["tau_dual"] = tau_dual
        duals["pi_dual"] = pi_dual

    if cannot_link:
        rho_dual = np.zeros((I, I))
        for (i, j) in cannot_link_pairs:
            rho_val = model.dual.get(model.CannotLink[i, j], 0)
            rho_dual[i, j] = rho_val
        duals["rho_dual"] = rho_dual

    if must_link:
        gamma_dual = np.zeros((I, I))
        for (i, j) in must_link_pairs:
            gamma_dual[i, j] = model.dual.get(model.MustLink[i, j], 0)
        duals["gamma_dual"] = gamma_dual

    return lambda_sol, duals, master_obj_val
