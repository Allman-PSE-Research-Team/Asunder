import importlib

import networkx as nx
import numpy as np

from asunder.load_balancing.column_generation.subproblem import (
    qmetis_pricing_subproblem,
)
from asunder.types import DecompositionResult


def test_load_balancer_routes_qmetis_without_public_backend_parameters(monkeypatch):
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
        algorithm="qmetis",
        disable_tqdm=True,
        verbose=-1,
    )

    assert captured["subproblem_fn"] is qmetis_pricing_subproblem
    assert captured["config"].subproblem_params == {
        "K": 2,
        "R": 0,
        "R_bounds": None,
    }
    assert np.array_equal(result.final_partition, partition)
    assert "qmetis_release" in result.metadata
