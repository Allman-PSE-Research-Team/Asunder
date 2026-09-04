"""Subproblem routines for CSD decomposition."""

from __future__ import annotations

import numpy as np

from asunder.base.algorithms.community import (
    run_igraph,
    run_igraph_spinglass,
    run_leidenalg,
    run_lpa,
    run_modularity,
    run_signed_louvain,
)
from asunder.base.algorithms.louvain_modified import ModifiedLouvain
from asunder.base.algorithms.RCCS import search_partition_by_reduced_cost
from asunder.base.algorithms.spectral import full_spectral_bisection
from asunder.base.column_generation.master import compute_f_star
from asunder.base.column_generation.pricing import (
    build_dual_weight_matrix,
    compute_reduced_cost,
)
from asunder.base.utils.graph import partition_vector_to_2d_matrix
from asunder.solvers import get_default_solver

try:
    from pyomo.environ import (
        Binary,
        ConcreteModel,
        ConstraintList,
        Objective,
        RangeSet,
        Set,
        TerminationCondition,
        Var,
        maximize,
        value,
    )
except Exception:  # pragma: no cover - optional dependency
    ConcreteModel = None


def _validate_resolution_support(
    *,
    algo: str,
    package: str | None,
    gamma: float,
) -> None:
    """Reject non-default resolution for algorithms that cannot use it."""

    supported = {
        ("networkx", "louvain"),
        ("networkx", "greedy"),
        ("sknetwork", "louvain"),
        ("sknetwork", "leiden"),
        ("igraph", "leiden"),
        ("igraph", "multilevel"),
        ("igraph", "cpm_leiden"),
        ("leidenalg", "leiden"),
        ("leidenalg", "signed_leiden"),
        ("leidenalg", "cpm_leiden"),
        (None, "signed_louvain"),
    }
    if gamma != 1.0 and (package, algo) not in supported:
        raise ValueError(
            f"{package or 'internal'}:{algo} does not support a non-default "
            "modularity resolution."
        )


_ZERO_ROW_SUM_BACKENDS = {
    ("sknetwork", "louvain"),
    ("sknetwork", "leiden"),
    ("igraph", "cpm_leiden"),
    ("leidenalg", "signed_leiden"),
    ("leidenalg", "cpm_leiden"),
    ("leidenalg", "signed_surprise_leiden"),
    (None, "signed_louvain"),
    (None, "spinglass"),
}


def _validate_zero_row_sum_support(
    *,
    algo: str,
    package: str | None,
    exact_rc: bool,
) -> None:
    """Reject zero-row-sum augmentation for incompatible heuristic paths."""

    if (package, algo) not in _ZERO_ROW_SUM_BACKENDS:
        supported = ", ".join(
            f"{backend or 'internal'}:{algorithm}"
            for backend, algorithm in sorted(
                _ZERO_ROW_SUM_BACKENDS,
                key=lambda item: (item[0] or "", item[1]),
            )
        )
        raise ValueError(
            "Zero-row-sum augmentation is not supported for "
            f"{package or 'internal'}:{algo}. Supported choices are: {supported}."
        )
    if not exact_rc:
        raise ValueError(
            "Zero-row-sum augmentation requires exact_rc=True because signed "
            "backends do not report the original reduced-cost objective."
        )


def heuristic_subproblem(
    A,
    a,
    m,
    duals,
    algo="louvain",
    package="networkx",
    verbose=False,
    gamma=1,
    exact_rc=True,
    seed=42,
    use_zero_row_sum=False,
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
    verbose : int or bool
        Controls the level of detail in the printed output.
        ``-1``: No output
        ``False`` | ``0``: Minimal output
        ``True`` | ``1``: Detailed output
    gamma : float
        Resolution parameter which controls the scale and size of the detected clusters.
        Algorithms that cannot apply it reject values other than ``1``.
    exact_rc : bool, default=True
        Recompute the returned partition's reduced cost using the original
        graph and duals.
    seed : int or None
        Random seed value
    use_zero_row_sum : bool, default=False
        If ``True``, replace the pairwise dual matrix by its zero-row-sum
        form before constructing the augmented adjacency. This experimental
        option is limited to scikit-network Louvain/Leiden and the supported
        signed heuristics. Scikit-network inputs must remain nonnegative;
        signed heuristics receive the un-clipped augmented adjacency. Exact
        reduced-cost evaluation is required.
    Returns
    -------
    Any
        Computed result.
    """
    gamma = float(gamma)
    _validate_resolution_support(algo=algo, package=package, gamma=gamma)
    if use_zero_row_sum:
        _validate_zero_row_sum_support(
            algo=algo,
            package=package,
            exact_rc=exact_rc,
        )
    dualW, constant_terms = build_dual_weight_matrix(A, duals)

    modA = A - (m * dualW)
    if use_zero_row_sum:
        diagonal = np.diag_indices_from(modA)
        modA[diagonal] += m * dualW.sum(axis=1)
        if package == "sknetwork" and np.min(modA) < -1e-12:
            raise ValueError(
                "Zero-row-sum augmentation produced negative weights. "
                "scikit-network Louvain/Leiden can use this option only when "
                "the augmented adjacency is nonnegative; select a signed "
                "heuristic or disable use_zero_row_sum."
            )
    mod_a = modA.sum(axis=0)
    mod_m = np.sum(mod_a)
    # negative weights will basically lead to a separation of nodes on that edge. The issue however
    # is that when m or 2m becomes negative, modularity logic flips completely and tightly connected
    # components are treated as bad splits.
    modA_positive = modA.copy()
    modA_positive[modA_positive < 0] = 0
    graph_modA = modA
    if use_zero_row_sum and package != "sknetwork":
        # NetworkX/igraph count an undirected loop twice in node degree.
        # Their loop-edge weight must therefore be half the matrix diagonal.
        graph_modA = modA.copy()
        graph_modA[np.diag_indices_from(graph_modA)] *= 0.5

    if package == "igraph" and algo not in {"greedy", "leiden"}:
        zii, metric = run_igraph(
            graph_modA if algo == "cpm_leiden" else modA_positive,
            algo=algo,
            resolution=gamma,
        )
    elif package == "leidenalg":
        zii, metric = run_leidenalg(
            graph_modA
            if algo.startswith("signed") or algo == "cpm_leiden"
            else modA_positive,
            algo=algo,
            seed=seed,
            resolution=gamma,
            verbose=verbose,
        )
    elif algo == "spinglass":
        zii = run_igraph_spinglass(graph_modA)
        metric = compute_f_star(modA, mod_a, mod_m, zii, gamma=gamma)
    elif algo == "signed_louvain":
        zii, metric = run_signed_louvain(
            graph_modA,
            seed=seed,
            resolution=gamma,
        )
    elif algo == "lpa" and package == "sknetwork":
        zii, metric = run_lpa(modA_positive)
    else:
        zii, metric = run_modularity(
            modA_positive,
            algo=algo,
            package=package,
            resolution=gamma,
            verbose=verbose,
            seed=seed
        )
        if metric is None:
            mod_a_p = modA_positive.sum(axis=0)
            mod_m_p = np.sum(mod_a_p)
            modB_p = (modA_positive / mod_m_p) - gamma * np.outer(mod_a_p, mod_a_p) / (mod_m_p**2)
            metric = np.sum(modB_p * zii)
    # TODO: algo param may be necessary if igraph algorithms require a different quality function.
    if exact_rc:
        sub_obj_val = compute_reduced_cost(
            A,
            a,
            m,
            zii,
            duals,
            gamma=gamma,
        )
    else:
        sub_obj_val = metric - constant_terms
    return sub_obj_val, zii


def solve_subproblem(
    A,
    a,
    m,
    duals,
    use_augmented_adjacency=False,
    verbose=False,
    solver=None,
    gamma=1.0,
):
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
    gamma : float, default=1.0
        Modularity resolution parameter.
    
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
    pairs = [(i, j) for i in range(I) for j in range(i + 1, I)]
    model.P = Set(initialize=pairs, dimen=2)
    model.z = Var(model.P, domain=Binary, initialize=0)
    # model.DiagonalUnity = Constraint(model.I, rule=lambda mdl, i: mdl.z[i, i] == 1)

    def zpair(i, j):
        if i == j:
            return 1.0
        return model.z[min(i, j), max(i, j)]

    model.T = Set(
        dimen=3,
        initialize=[(i, j, k) for i in range(I) for j in range(i + 1, I) for k in range(j + 1, I)]
    )

    model.Transitivity = ConstraintList()

    for i, j, k in model.T:
        model.Transitivity.add(zpair(i, j) + zpair(i, k) - zpair(j, k) <= 1)
        model.Transitivity.add(zpair(i, j) + zpair(j, k) - zpair(i, k) <= 1)
        model.Transitivity.add(zpair(i, k) + zpair(j, k) - zpair(i, j) <= 1)


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
                (
                    (modA[i, j] / mod_m)
                    - gamma * ((mod_a[i] * mod_a[j]) / (mod_m**2))
                )
                * zpair(i, j)
                for i in mdl.I
                for j in mdl.I
            )
        else:
            M = sum(
                (
                    (A[i, j] / m)
                    - gamma * ((a[i] * a[j]) / (m**2))
                    - dualW[i, j]
                )
                * zpair(i, j)
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
    z_sol = np.array([[value(zpair(i, j)) for j in model.I] for i in model.I])
    return value(model.OBJ), z_sol


def custom_heuristic_subproblem(
    A,
    a,
    m,
    duals,
    algo="full_louvain",
    verbose=False,
    max_iterations=50,
    tolerance=1e-8,
    seed=42,
    gamma=1.0,
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
    max_iterations : int
        Maximum number of iterations.
    tolerance : float
        Tolerance value.
    seed : int or None
        Random seed value
    gamma : float, default=1.0
        Modularity resolution. Modified Louvain supports non-default values;
        the other custom heuristics reject them.
    
    Returns
    -------
    Any
        Computed result.
    """
    if float(gamma) != 1.0 and algo not in {
        "full_louvain",
        "one_level_louvain",
    }:
        raise ValueError(
            f"The custom {algo!r} pricing heuristic does not support a "
            "non-default modularity resolution."
        )
    assert algo in {"spectral", "full_louvain", "one_level_louvain", "RCCS"}
    dualW, constant_terms = build_dual_weight_matrix(A, duals)

    if "louvain" in algo:
        louvain_model = ModifiedLouvain(
            resolution=gamma,
            random_state=seed,
        )
        if algo.startswith("full_"):
            louvain_model.fit(A, duals)
        else:
            louvain_model.fit_modified_one_level(A, duals, max_iter=max_iterations, tol=tolerance)
        z_sol = partition_vector_to_2d_matrix(louvain_model.labels_)
        metric = louvain_model.obj_val_
    else:
        if algo == "RCCS":
            res = search_partition_by_reduced_cost(adjacency=A, duals=duals, random_seed=seed)
            best_labels = res["best_labels"]
            z_sol = partition_vector_to_2d_matrix(best_labels)
            metric = res["best_reduced_cost"] + constant_terms # for normalization sake
        else:
            z_sol, metric = full_spectral_bisection(A, a, m, dualW, refinement=True, verbose=verbose, max_outer_passes=max_iterations, tol=tolerance)

    if metric is None:
        modularity_contribution = compute_f_star(
            A,
            a,
            m,
            z_sol,
            gamma=gamma,
        )
        dual_contribution = (dualW * z_sol).sum()
        metric = modularity_contribution - dual_contribution
    sub_obj_val = metric - constant_terms
    return sub_obj_val, z_sol
