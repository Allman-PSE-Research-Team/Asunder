"""Projection repairs for load-balanced partitions."""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

from asunder.base.algorithms.modular_VFD import (
    _build_components,
    _component_sum_matrix_B,
    _normalize_pair,
    _objective_B_from_comp_assignment,
    _range_bounds_from_KR,
    _symmetrize_unitdiag,
)
from asunder.base.utils.graph import partition_vector_to_2d_matrix
from asunder.solvers import get_default_solver


def _component_sum_matrix_from_node_matrix(M: np.ndarray, comp: Dict[str, Any]) -> np.ndarray:
    """Aggregate a node-level matrix to components by summation."""
    M = np.asarray(M, dtype=float)
    N = M.shape[0]
    cid = np.asarray(comp["cid"], dtype=int)
    C = int(comp["C"])
    if C == 0:
        return np.zeros((0, 0), dtype=float)

    P = np.zeros((N, C), dtype=float)
    P[np.arange(N), cid] = 1.0
    return P.T @ M @ P


def _build_gvec_from_components(comp: Dict[str, Any], comp2g: np.ndarray, N: int) -> np.ndarray:
    gvec = np.empty(N, dtype=int)
    for c, nodes in enumerate(comp["comps"]):
        gvec[nodes] = int(comp2g[c])
    return gvec


def _projection_score(pair_weights: Sequence[Tuple[int, int, float]], comp2g: np.ndarray) -> float:
    return float(sum(weight for c, d, weight in pair_weights if int(comp2g[c]) == int(comp2g[d])))


def project_partition_ilp(
    *,
    wz: np.ndarray,
    A: np.ndarray,
    a: np.ndarray,
    m: float,
    K: int,
    R: int = 1,
    R_bounds: Optional[Tuple[int, int]] = None,
    must_link: Sequence[Tuple[int, int]] = (),
    cannot_link: Sequence[Tuple[int, int]] = (),
    seed: int | None = 42,
    solver=None,
) -> Optional[Tuple[np.ndarray, Dict[str, Any]]]:
    """
    Project ``wz`` onto the exact-``K`` load-balanced feasible partition set.

    The model maximizes pairwise agreement with ``wz`` while enforcing hard
    must-link, cannot-link, and load-balance constraints.
    """
    try:
        from pyomo.environ import Binary, ConcreteModel, ConstraintList, Objective, RangeSet, Set, Var, maximize, value
    except Exception:
        return None

    A = np.asarray(A, dtype=float)
    a = np.asarray(a, dtype=float).reshape(-1)
    wz = np.asarray(wz, dtype=float)
    if wz.ndim == 1:
        wz = partition_vector_to_2d_matrix(wz)

    N = int(A.shape[0])
    K = int(K)
    if A.shape != (N, N) or wz.shape != (N, N) or a.shape[0] != N:
        raise ValueError("A and wz must be (N,N), and a must be (N,).")
    if K <= 0:
        raise ValueError("K must be positive.")

    if R_bounds is None:
        r_min, r_max = _range_bounds_from_KR(N, K, int(R))
    else:
        r_min, r_max = int(R_bounds[0]), int(R_bounds[1])
        if r_min > r_max:
            raise ValueError("R_bounds must satisfy r_min <= r_max.")

    if not (K * r_min <= N <= K * r_max):
        return None

    must_link = [_normalize_pair(i, j) for i, j in (must_link or [])]
    cannot_link = [_normalize_pair(i, j) for i, j in (cannot_link or [])]

    comp = _build_components(N, must_link, cannot_link)
    if comp is None:
        return None

    Cn = int(comp["C"])
    if Cn == 0:
        return partition_vector_to_2d_matrix(np.zeros(N, dtype=int)), {
            "r_min": int(r_min),
            "r_max": int(r_max),
            "K_used": 0,
            "requested_K": int(K),
            "feasibility_fallback": "projection_ilp",
            "seed": int(seed or 0),
        }

    csz = np.asarray(comp["csz"], dtype=int)
    if int(csz.max()) > r_max or K > Cn:
        return None

    cid = np.asarray(comp["cid"], dtype=int)
    cannot_comp_pairs = sorted(
        {
            tuple(sorted((int(cid[int(i)]), int(cid[int(j)]))))
            for i, j in cannot_link
            if int(cid[int(i)]) != int(cid[int(j)])
        }
    )

    W_B = _component_sum_matrix_B(A, a, m, comp)
    C_wz = _component_sum_matrix_from_node_matrix(_symmetrize_unitdiag(wz), comp)
    pair_weights = [
        (c, d, float(C_wz[c, d] + C_wz[d, c]))
        for c in range(Cn)
        for d in range(c + 1, Cn)
        if float(C_wz[c, d] + C_wz[d, c]) != 0.0
    ]

    if solver is None:
        try:
            solver = get_default_solver()
        except Exception:
            return None

    model = ConcreteModel()
    model.C = RangeSet(0, Cn - 1)
    model.G = RangeSet(0, K - 1)
    model.P = Set(initialize=list(range(len(pair_weights))))

    model.x = Var(model.C, model.G, domain=Binary)
    model.y = Var(model.P, model.G, domain=Binary)
    model.constraints = ConstraintList()

    for c in range(Cn):
        model.constraints.add(sum(model.x[c, g] for g in model.G) == 1)

    for g in range(K):
        group_size = sum(int(csz[c]) * model.x[c, g] for c in model.C)
        model.constraints.add(group_size >= int(r_min))
        model.constraints.add(group_size <= int(r_max))

    for c, d in cannot_comp_pairs:
        for g in range(K):
            model.constraints.add(model.x[c, g] + model.x[d, g] <= 1)

    for p, (c, d, _) in enumerate(pair_weights):
        for g in range(K):
            model.constraints.add(model.y[p, g] <= model.x[c, g])
            model.constraints.add(model.y[p, g] <= model.x[d, g])
            model.constraints.add(model.y[p, g] >= model.x[c, g] + model.x[d, g] - 1)

    model.obj = Objective(
        expr=sum(weight * model.y[p, g] for p, (_, _, weight) in enumerate(pair_weights) for g in model.G),
        sense=maximize,
    )

    try:
        result = solver.solve(model, tee=False)
    except Exception:
        return None

    term = getattr(getattr(result, "solver", None), "termination_condition", None)
    term_name = str(term).lower()
    if "infeasible" in term_name or "unbounded" in term_name:
        return None

    comp2g = -np.ones(Cn, dtype=int)
    for c in range(Cn):
        assigned = []
        for g in range(K):
            val = value(model.x[c, g], exception=False)
            if val is not None and float(val) > 0.5:
                assigned.append(g)
        if len(assigned) != 1:
            return None
        comp2g[c] = int(assigned[0])

    Q = _objective_B_from_comp_assignment(W_B, comp2g, K)
    if not np.isfinite(Q):
        return None

    projection_score = _projection_score(pair_weights, comp2g)
    Z = partition_vector_to_2d_matrix(_build_gvec_from_components(comp, comp2g, N))
    meta = {
        "r_min": int(r_min),
        "r_max": int(r_max),
        "K_used": int(K),
        "requested_K": int(K),
        "objective_B_sum": float(Q),
        "objective_total": float(Q),
        "projection_wz_score": float(projection_score),
        "feasibility_fallback": "projection_ilp",
        "solver_termination_condition": str(term),
        "seed": int(seed or 0),
    }
    return Z, meta
