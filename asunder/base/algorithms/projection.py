"""Feasibility projections for partition matrices."""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

from asunder.base.utils.graph import (
    partition_matrix_to_vector,
    partition_vector_to_2d_matrix,
)
from asunder.base.utils.partition_generation import _build_components_links_only
from asunder.solvers import get_default_solver


def _validate_pairs(
    pairs: Sequence[Tuple[int, int]] | None,
    N: int,
    *,
    relation_name: str,
    reject_self: bool,
) -> list[tuple[int, int]]:
    normalized = []
    for pair in pairs or ():
        if len(pair) != 2:
            raise ValueError(f"Each {relation_name} entry must contain two nodes.")
        i, j = int(pair[0]), int(pair[1])
        if not (0 <= i < N and 0 <= j < N):
            raise ValueError(
                f"{relation_name} pair {(i, j)} contains a node outside 0..{N - 1}."
            )
        if i == j:
            if reject_self:
                return [(-1, -1)]
            continue
        normalized.append((i, j) if i < j else (j, i))
    return sorted(set(normalized))


def _component_pair_weights(wz: np.ndarray, comp: Dict[str, Any]) -> dict[tuple[int, int], float]:
    cid = np.asarray(comp["cid"], dtype=int)
    C = int(comp["C"])
    weights = {(c, d): 0.0 for c in range(C) for d in range(c + 1, C)}
    for i in range(wz.shape[0]):
        ci = int(cid[i])
        for j in range(i + 1, wz.shape[0]):
            cj = int(cid[j])
            if ci == cj:
                continue
            c, d = (ci, cj) if ci < cj else (cj, ci)
            weights[(c, d)] += float(wz[i, j] + wz[j, i] - 1.0)
    return weights


def _partition_from_component_matrix(
    comp: Dict[str, Any],
    same_component: dict[tuple[int, int], bool],
    N: int,
) -> np.ndarray:
    C = int(comp["C"])
    parent = np.arange(C, dtype=int)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return int(x)

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for (c, d), is_same in same_component.items():
        if is_same:
            union(c, d)

    labels = np.empty(N, dtype=int)
    for c, nodes in enumerate(comp["comps"]):
        label = find(c)
        for node in nodes:
            labels[int(node)] = label
    return partition_vector_to_2d_matrix(labels)


def project_partition_pairwise_ilp(
    wz: np.ndarray,
    *,
    must_link: Sequence[Tuple[int, int]] = (),
    cannot_link: Sequence[Tuple[int, int]] = (),
    solver=None,
) -> Optional[Tuple[np.ndarray, Dict[str, Any]]]:
    """
    Project ``wz`` onto the nearest partition satisfying pairwise constraints.

    This is a squared Frobenius projection over binary partition matrices. Since
    the repaired partition ``Z`` is binary with a fixed unit diagonal, minimizing
    ``||Z - wz||_F^2`` is equivalent to maximizing
    ``sum_{i < j} (wz[i, j] + wz[j, i] - 1) * Z[i, j]``.

    Parameters
    ----------
    wz : ndarray
        Input partition vector or square co-association matrix.
    must_link : sequence of tuple[int, int]
        Node pairs that must belong to the same block.
    cannot_link : sequence of tuple[int, int]
        Node pairs that must belong to different blocks.
    solver : Any
        Optional Pyomo solver. If omitted, Asunder's default solver is used.

    Returns
    -------
    tuple[ndarray, dict] or None
        Projected partition matrix and metadata, or ``None`` if the projection
        cannot be solved or the pairwise constraints are infeasible.
    """
    try:
        from pyomo.environ import (
            Binary,
            ConcreteModel,
            ConstraintList,
            Objective,
            Param,
            RangeSet,
            Set,
            Var,
            maximize,
            value,
        )
    except Exception:
        return None

    wz = np.asarray(wz, dtype=float)
    if wz.ndim == 1:
        wz = partition_vector_to_2d_matrix(wz)
    if wz.ndim != 2 or wz.shape[0] != wz.shape[1]:
        raise ValueError("wz must be a partition vector or square matrix.")

    N = int(wz.shape[0])
    must_link = _validate_pairs(must_link, N, relation_name="must-link", reject_self=False)
    cannot_link = _validate_pairs(cannot_link, N, relation_name="cannot-link", reject_self=True)
    if cannot_link == [(-1, -1)]:
        return None

    comp = _build_components_links_only(N, must_link, cannot_link)
    if comp is None:
        return None

    C = int(comp["C"])
    if C == 0:
        return np.zeros((0, 0), dtype=int), {
            "K_used": 0,
            "C": 0,
            "objective": 0.0,
            "projection_distance": 0.0,
            "feasibility_projection": "pairwise_ilp",
        }
    if C == 1:
        Z = np.ones((N, N), dtype=int)
        return Z, {
            "K_used": 1,
            "C": 1,
            "objective": 0.0,
            "projection_distance": float(np.sum((Z.astype(float) - wz) ** 2)),
            "feasibility_projection": "pairwise_ilp",
        }

    weights = _component_pair_weights(wz, comp)
    pairs = sorted(weights)
    cid = np.asarray(comp["cid"], dtype=int)
    cannot_comp_pairs = sorted(
        {
            tuple(sorted((int(cid[i]), int(cid[j]))))
            for i, j in cannot_link
            if int(cid[i]) != int(cid[j])
        }
    )

    if solver is None:
        try:
            solver = get_default_solver()
        except Exception:
            return None

    model = ConcreteModel()
    model.C = RangeSet(0, C - 1)
    model.P = Set(initialize=pairs, dimen=2)
    model.weight = Param(model.P, initialize=weights)
    model.z = Var(model.P, domain=Binary, initialize=0)
    model.constraints = ConstraintList()

    def zpair(i: int, j: int):
        if i == j:
            return 1.0
        return model.z[i, j] if i < j else model.z[j, i]

    for c, d in cannot_comp_pairs:
        model.constraints.add(zpair(c, d) == 0)

    for i in range(C):
        for j in range(i + 1, C):
            for k in range(j + 1, C):
                model.constraints.add(zpair(i, j) + zpair(i, k) - zpair(j, k) <= 1)
                model.constraints.add(zpair(i, j) + zpair(j, k) - zpair(i, k) <= 1)
                model.constraints.add(zpair(i, k) + zpair(j, k) - zpair(i, j) <= 1)

    model.obj = Objective(
        expr=sum(model.weight[c, d] * model.z[c, d] for c, d in model.P),
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

    same_component = {}
    for c, d in pairs:
        val = value(model.z[c, d], exception=False)
        if val is None:
            return None
        same_component[(int(c), int(d))] = float(val) > 0.5

    Z = _partition_from_component_matrix(comp, same_component, N)
    objective = float(sum(weight for pair, weight in weights.items() if same_component.get(pair, False)))
    meta = {
        "K_used": int(np.unique(partition_matrix_to_vector(Z)).size),
        "C": C,
        "objective": objective,
        "projection_distance": float(np.sum((Z.astype(float) - wz) ** 2)),
        "feasibility_projection": "pairwise_ilp",
        "solver_termination_condition": str(term),
    }
    return Z, meta
