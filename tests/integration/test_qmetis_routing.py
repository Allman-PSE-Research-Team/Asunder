import importlib

import networkx as nx
import numpy as np
import pytest

from asunder.base.column_generation.pricing import compute_reduced_cost
from asunder.load_balancing.column_generation import subproblem as pricing_module
from asunder.load_balancing.column_generation.subproblem import (
    qmetis_pricing_subproblem,
)
from asunder.types import DecompositionResult


@pytest.mark.parametrize(
    ("memory_options", "expected_limit"),
    [({}, 512 * 1024**2), ({"max_dense_working_bytes": 1024}, 1024),
     ({"max_dense_working_bytes": None}, None)],
)
def test_load_balancer_routes_qmetis_without_public_backend_parameters(
    monkeypatch, memory_options, expected_limit,
):
    lb_module = importlib.import_module(
        "asunder.load_balancing.column_generation.LB"
    )
    captured = {}
    partition = np.equal.outer([0, 0, 1, 1], [0, 0, 1, 1]).astype(int)

    def fake_decomposition(
        A,
        *,
        config,
        master_fn,
        subproblem_fn,
        **kwargs,
    ):
        captured["config"] = config
        captured["master_fn"] = master_fn
        captured["subproblem_fn"] = subproblem_fn
        return DecompositionResult(
            records=[],
            final_partition=partition,
            final_master_obj=0.0,
            metadata={},
        )

    monkeypatch.setattr(lb_module, "run_csd_decomposition", fake_decomposition)

    result = lb_module.LoadBalancer(
        nx.path_graph(4),
        K=2,
        R=0,
        resolution=1.25,
        algorithm="qmetis",
        disable_tqdm=True,
        verbose=-1,
        **memory_options,
    )

    assert captured["subproblem_fn"] is qmetis_pricing_subproblem
    pricing_params = captured["config"].subproblem_params
    assert {key: pricing_params[key] for key in ("K", "R", "R_bounds")} == {
        "K": 2,
        "R": 0,
        "R_bounds": None,
    }
    assert np.array_equal(captured["config"].node_weights, np.ones(4))
    assert np.array_equal(result.final_partition, partition)
    assert captured["config"].resolution == 1.25
    assert captured["config"].max_dense_working_bytes == expected_limit
    assert result.metadata["resolution"] == 1.25
    assert "qmetis_release" in result.metadata


@pytest.mark.parametrize(
    ("weights", "K", "bounds", "expected_count"),
    [
        ([1] * 6, 2, (2, 2), 3),
        ([1] * 6, 5, (3, 3), 2),
        ([4] * 3, 5, (3, 12), 3),
        ([4] * 3, 2, (12, 12), 1),
    ],
)
def test_explicit_bounds_clamp_qmetis_hint_and_preserve_exact_scoring(
    monkeypatch, weights, K, bounds, expected_count,
):
    n = len(weights)
    A = nx.to_numpy_array(nx.cycle_graph(n))
    a = A.sum(axis=1)
    m = float(A.sum())
    duals = {"mu_dual": 0.17}
    calls = []

    def run_qmetis(adjacency, K, **kwargs):
        calls.append((K, kwargs))
        labels = np.arange(n) * K // n
        return np.equal.outer(labels, labels).astype(int), 123.0

    monkeypatch.setattr(pricing_module, "run_qmetis", run_qmetis)
    reduced_cost, partition = qmetis_pricing_subproblem(
        A, a, m, duals, K=K, R=1, R_bounds=bounds,
        balance_weights=np.asarray(weights), gamma=1.25,
    )

    assert len(np.unique(partition, axis=0)) == expected_count
    assert reduced_cost == pytest.approx(
        compute_reduced_cost(A, a, m, partition, duals, gamma=1.25)
    )
    if expected_count == 1:
        assert calls == []
    else:
        assert len(calls) == 1
        count, kwargs = calls[0]
        assert count == expected_count
        assert np.array_equal(kwargs["node_weights"], weights)
        assert kwargs["balance_epsilon"] == pytest.approx(
            max(0.0, bounds[1] / (sum(weights) / count) - 1.0)
        )
