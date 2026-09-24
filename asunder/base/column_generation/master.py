"""Master problem and score utilities for CSD decomposition."""

from __future__ import annotations

import numpy as np
from scipy import sparse

from asunder.base.utils.matrix import matrix_scalar, structural_edge_pairs
from asunder.solvers import get_default_solver
from asunder.types import MatrixLike

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


def compute_f_star(
    A: MatrixLike,
    a: np.ndarray,
    m: float,
    z: MatrixLike,
    gamma: float = 1.0,
) -> float:
    """
    Compute the column/partition score used by the restricted master objective.
    
    Parameters
    ----------
    A : ndarray or scipy.sparse.spmatrix, shape (N, N)
        Adjacency / weight matrix.
    a : np.ndarray of int | float, shape (N,)
        Degree-like vector; defaults to row sums of the symmetrized adjacency.
    m : float
        Twice the total weight in the graph.
    z : ndarray or scipy.sparse.spmatrix, shape (N, N)
        Hard or fractional co-association matrix.
    gamma : float
        Resolution parameter.
    
    Returns
    -------
    metric: float
        Modularity score.

    Raises
    ------
    ValueError
        If the graph volume is nonpositive or the matrix dimensions do not
        match the degree vector.

    Notes
    -----
    Dense terms are reduced one row at a time using float64 arithmetic and
    O(N) scratch space. Boolean columns are not promoted to full floating-point
    matrices. Sparse terms retain their sparse operations.
    """
    if  m <= 0:
        raise ValueError("Graph must have an edge and a positive edge sum.")
    a = np.asarray(a, dtype=float).reshape(-1)
    if np.shape(A) != (a.size, a.size) or np.shape(z) != (a.size, a.size):
        raise ValueError("A and z must be square matrices matching the length of a.")
    dense_adjacency = None
    if sparse.issparse(A):
        adjacency_term = float(A.multiply(z).sum())
    elif sparse.issparse(z):
        adjacency_term = float(z.multiply(np.asarray(A)).sum())
    else:
        dense_adjacency = np.asarray(A)
        adjacency_term = 0.0
    if sparse.issparse(z):
        null_term = float(a @ (z @ a))
    else:
        null_term = 0.0
        for index, row in enumerate(np.asarray(z)):
            # Casting a single row avoids the N x N promotion performed by
            # dense Boolean-matrix/float-vector multiplication.
            row_values = np.asarray(row, dtype=np.float64)
            null_term += float(a[index] * np.dot(row_values, a))
            if dense_adjacency is not None:
                adjacency_row = np.asarray(dense_adjacency[index], dtype=np.float64)
                adjacency_term += float(np.dot(adjacency_row, row_values))
    return adjacency_term / m - float(gamma) * null_term / (m * m)


def solve_master_problem(
    A: MatrixLike,
    a: np.ndarray,
    m: float,
    Z_star: list[MatrixLike],
    f_stars: list[float],
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
    A : numpy.ndarray or scipy.sparse.csr_matrix, shape (N, N)
        Adjacency or weight matrix.
    a : np.ndarray of int | float, shape (N,)
        Degree-like vector; defaults to row sums of the symmetrized adjacency.
    m : float
        Twice the total weight in the graph.
    Z_star : list[numpy.ndarray or scipy.sparse.csr_matrix]
        Binary co-association columns. Dense and CSR columns may coexist.
    f_stars : list[float]
        Objective values computed using the existing columns.
    cannot_link : list[tuple[int, int]] or None
        List of node pairs that must not be together.
    must_link : list[tuple[int, int]] or None
        List of node pairs that must be together.
    worthy_edges : list[tuple[int, int]] or None
        Edges allowed to connect different communities. ``None`` disables the
        edge rule; an empty collection requires every structural edge to stay
        within a community.
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
    lambda_sol : list or ndarray of float
        A list/vector which sums to ``1`` that indicates what weight is assigned to each column (and by implication, what columns are active).
    duals : dict[str, numpy.ndarray, scipy.sparse.csr_matrix, or float]
        Dual values computed from the master problem. Pairwise duals are
        returned as CSR matrices; scalar and one-dimensional values are also
        supported by downstream pricing.
    master_obj_val : float
        The objective value of the master problem.

    Raises
    ------
    RuntimeError
        If the solver terminates without optimality for a reason other than
        infeasibility.
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

    if worthy_edges is not None:
        worthy_edges = tuple(tuple(sorted(edge)) for edge in worthy_edges)
        worthy_edge_set = set(worthy_edges)

        def worthy_edge_rule(mdl, i, j):
            """
            Enforce worthy-edge consistency for a candidate edge pair.
            """
            if tuple(sorted((int(i), int(j)))) in worthy_edge_set:
                return Constraint.Skip
            return sum(
                mdl.lmbd[c] * matrix_scalar(Z_star[c], i, j) for c in mdl.C
            ) == 1

        all_edges = structural_edge_pairs(A)
        model.WorthyEdges = Constraint(all_edges, rule=worthy_edge_rule)
    else:
        all_edges = []

    if cannot_link:
        cannot_link_pairs = tuple(tuple(sorted(pair)) for pair in cannot_link)

        def cannot_link_rule(mdl, i, j):
            """
            Enforce a cannot-link pair in the master problem.
            """
            return sum(
                mdl.lmbd[c] * matrix_scalar(Z_star[c], i, j) for c in mdl.C
            ) == 0

        model.CannotLink = Constraint(cannot_link_pairs, rule=cannot_link_rule)
    else:
        cannot_link_pairs = []

    if must_link:
        must_link_pairs = tuple(tuple(sorted(pair)) for pair in must_link)

        def must_link_rule(mdl, i, j):
            """
            Enforce a must-link pair in the master problem.
            """
            return sum(
                mdl.lmbd[c] * matrix_scalar(Z_star[c], i, j) for c in mdl.C
            ) == 1

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
    condition = res.solver.termination_condition
    if condition == TerminationCondition.infeasible:
        return (None, None, None) if extract_dual else (None, None)
    if condition != TerminationCondition.optimal:
        raise RuntimeError(f"Master solve ended without optimality: {condition}")
    
    lambda_sol = [value(model.lmbd[c]) for c in model.C]
    master_obj_val = value(model.OBJ)

    if not extract_dual:
        return lambda_sol, master_obj_val

    duals = {"mu_dual": model.dual.get(model.OneColumn, 0)}
    if cannot_link:
        rows, columns, values = [], [], []
        for (i, j) in cannot_link_pairs:
            dual = model.dual.get(model.CannotLink[i, j], 0)
            if dual:
                rows.append(i)
                columns.append(j)
                values.append(dual)
        duals["tau_dual"] = sparse.csr_matrix(
            (values, (rows, columns)), shape=(I, I), dtype=float
        )

    if must_link:
        rows, columns, values = [], [], []
        for (i, j) in must_link_pairs:
            dual = model.dual.get(model.MustLink[i, j], 0)
            if dual:
                rows.append(i)
                columns.append(j)
                values.append(dual)
        duals["gamma_dual"] = sparse.csr_matrix(
            (values, (rows, columns)), shape=(I, I), dtype=float
        )

    if worthy_edges is not None:
        rows, columns, values = [], [], []
        for (i, j) in all_edges:
            if tuple(sorted((int(i), int(j)))) not in worthy_edge_set:
                dual = model.dual.get(model.WorthyEdges[i, j], 0)
                if dual:
                    rows.append(i)
                    columns.append(j)
                    values.append(dual)
        duals["pi_dual"] = sparse.csr_matrix(
            (values, (rows, columns)), shape=(I, I), dtype=float
        )

    return lambda_sol, duals, master_obj_val
