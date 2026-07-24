import numpy as np
import pytest

from asunder.base.column_generation.pricing import (
    build_dual_weight_matrix,
    compute_reduced_cost,
)
from asunder.load_balancing.algorithms.qmetis import (
    QMETISApproximationWarning,
)
from asunder.load_balancing.column_generation import subproblem as lb_subproblem


def test_build_dual_weight_matrix_handles_fractional_duals():
    adjacency = np.zeros((3, 3))
    one_dimensional = np.array([0.1, 0.2, 0.4])
    asymmetric = np.array(
        [
            [0.0, 0.3, 0.0],
            [0.1, 0.0, 0.5],
            [0.0, 0.2, 0.0],
        ]
    )

    dual_weight, constant = build_dual_weight_matrix(
        adjacency,
        {
            "node": one_dimensional,
            "pair": asymmetric,
            "constant": np.float64(0.125),
        },
    )

    expected = 0.5 * (
        one_dimensional[:, None] + one_dimensional[None, :]
    ) + 0.5 * (asymmetric + asymmetric.T)
    assert np.allclose(dual_weight, expected)
    assert constant == pytest.approx(0.125)


def test_qmetis_pricing_uses_bound_envelope_and_exact_float_score(monkeypatch):
    adjacency = np.array(
        [
            [0.0, 1.0, 0.1, 0.0],
            [1.0, 0.0, 0.5, 0.0],
            [0.1, 0.5, 0.0, 1.0],
            [0.0, 0.0, 1.0, 0.0],
        ]
    )
    degrees = adjacency.sum(axis=1)
    total = float(degrees.sum())
    duals = {
        "mu_dual": 0.25,
        "tau_dual": np.array([0.01, 0.02, 0.03, 0.04]),
        "pi_dual": np.array([-0.01, 0.0, 0.01, 0.02]),
    }
    candidate = np.equal.outer([0, 0, 1, 1], [0, 0, 1, 1]).astype(int)
    observed = {}

    def fake_run_qmetis(weights, K, **kwargs):
        observed["weights"] = np.asarray(weights)
        observed["K"] = K
        observed.update(kwargs)
        return candidate, 9876.5

    monkeypatch.setattr(lb_subproblem, "run_qmetis", fake_run_qmetis)
    reduced_cost, result = lb_subproblem.qmetis_pricing_subproblem(
        adjacency,
        degrees,
        total,
        duals,
        K=2,
        R=2,
        seed=9,
    )

    assert np.array_equal(result, candidate)
    assert observed["K"] == 2
    assert observed["balance_epsilon"] == pytest.approx(0.5)
    assert observed["seed"] == 9
    assert np.all(observed["weights"] >= 0)
    assert reduced_cost == pytest.approx(
        compute_reduced_cost(adjacency, degrees, total, candidate, duals)
    )


def test_qmetis_pricing_falls_back_when_every_adjusted_edge_is_clipped(
    monkeypatch,
):
    adjacency = np.zeros((4, 4))

    def unexpected_qmetis_call(*args, **kwargs):
        raise AssertionError("Native QMETIS should not receive an edgeless graph.")

    monkeypatch.setattr(lb_subproblem, "run_qmetis", unexpected_qmetis_call)
    reduced_cost, candidate = lb_subproblem.qmetis_pricing_subproblem(
        adjacency,
        np.zeros(4),
        1.0,
        {"mu_dual": 0.0},
        K=2,
        R=0,
    )

    assert reduced_cost == pytest.approx(0.0)
    assert np.array_equal(candidate.sum(axis=1), np.full(4, 2))


def test_qmetis_pricing_labels_contracted_diagonal_as_approximate(
    monkeypatch,
):
    adjacency = np.array(
        [
            [4.0, 1.0, 0.0, 0.0],
            [1.0, 2.0, 1.0, 0.0],
            [0.0, 1.0, 6.0, 1.0],
            [0.0, 0.0, 1.0, 2.0],
        ]
    )
    degrees = adjacency.sum(axis=1)
    total = float(degrees.sum())
    candidate = np.equal.outer([0, 0, 1, 1], [0, 0, 1, 1]).astype(int)
    observed = {}

    def fake_run_qmetis(weights, K, **kwargs):
        observed["weights"] = np.asarray(weights)
        return candidate, 0.0

    monkeypatch.setattr(lb_subproblem, "run_qmetis", fake_run_qmetis)
    with pytest.warns(
        QMETISApproximationWarning,
        match="contracted internal-edge mass",
    ):
        reduced_cost, result = lb_subproblem.qmetis_pricing_subproblem(
            adjacency,
            degrees,
            total,
            {"mu_dual": 0.0},
            K=2,
            R=2,
        )

    assert np.array_equal(result, candidate)
    assert np.all(np.diag(observed["weights"]) == 0)
    assert reduced_cost == pytest.approx(
        compute_reduced_cost(
            adjacency,
            degrees,
            total,
            candidate,
            {"mu_dual": 0.0},
        )
    )
