from __future__ import annotations

import itertools
from types import SimpleNamespace

import numpy as np

from asunder.base.algorithms.projection import project_partition_pairwise_ilp


class _EnumeratingBinarySolver:
    """Small solver stub for projection models with binary pair variables."""

    def __init__(self):
        self.calls = 0

    def solve(self, model, tee=False):
        from pyomo.environ import value

        self.calls += 1
        pairs = list(model.P)
        best_obj = None
        best_bits = None

        for bits in itertools.product((0, 1), repeat=len(pairs)):
            for pair, bit in zip(pairs, bits):
                model.z[pair].set_value(bit)
            if not self._is_feasible(model, value):
                continue
            obj = float(value(model.obj.expr))
            if best_obj is None or obj > best_obj + 1e-12:
                best_obj = obj
                best_bits = bits

        if best_bits is None:
            return SimpleNamespace(solver=SimpleNamespace(termination_condition="infeasible"))

        for pair, bit in zip(pairs, best_bits):
            model.z[pair].set_value(bit)
        return SimpleNamespace(solver=SimpleNamespace(termination_condition="optimal"))

    @staticmethod
    def _is_feasible(model, value):
        for constraint in model.constraints.values():
            body = float(value(constraint.body))
            if constraint.lower is not None and body < float(value(constraint.lower)) - 1e-8:
                return False
            if constraint.upper is not None and body > float(value(constraint.upper)) + 1e-8:
                return False
        return True


def test_pairwise_projection_uses_signed_frobenius_objective():
    """Pairs below 0.5 are penalized, not weakly rewarded."""
    wz = np.full((3, 3), 0.4, dtype=float)
    np.fill_diagonal(wz, 1.0)
    solver = _EnumeratingBinarySolver()

    out = project_partition_pairwise_ilp(wz, solver=solver)

    assert out is not None
    z, meta = out
    assert solver.calls == 1
    assert np.array_equal(z, np.eye(3, dtype=int))
    assert meta["K_used"] == 3
    assert meta["feasibility_projection"] == "pairwise_ilp"


def test_pairwise_projection_enforces_must_and_cannot_links():
    """Pairwise constraints are hard constraints around the distance objective."""
    wz = np.ones((4, 4), dtype=float)
    solver = _EnumeratingBinarySolver()

    out = project_partition_pairwise_ilp(
        wz,
        must_link=[(1, 2)],
        cannot_link=[(0, 1)],
        solver=solver,
    )

    assert out is not None
    z, meta = out
    assert solver.calls == 1
    assert z[1, 2] == 1
    assert z[0, 1] == 0
    assert z[0, 2] == 0
    assert meta["K_used"] in {2, 3}


def test_pairwise_projection_rejects_conflicting_pairwise_constraints_before_solving():
    solver = _EnumeratingBinarySolver()

    out = project_partition_pairwise_ilp(
        np.ones((3, 3), dtype=float),
        must_link=[(0, 1)],
        cannot_link=[(0, 1)],
        solver=solver,
    )

    assert out is None
    assert solver.calls == 0
