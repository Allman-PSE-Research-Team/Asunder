from types import SimpleNamespace

import networkx as nx
import numpy as np
import pytest
from scipy import sparse

from asunder.base.algorithms import community
from asunder.base.column_generation.pricing import compute_reduced_cost
from asunder.base.column_generation.subproblem import heuristic_subproblem


def activate(monkeypatch, function, *, active=True, per_function=True):
    function.backends = {"example"}
    priorities = SimpleNamespace(
        algos=[] if per_function or not active else ["example"],
        get=lambda name, default: ["example"]
        if active and per_function and name == "leiden_communities" else default,
    )
    config = dict(nx.config)
    config["backend_priority"] = priorities
    monkeypatch.setattr(nx, "config", SimpleNamespace(**config))
    monkeypatch.setattr(nx.community, "leiden_communities", function, raising=False)


@pytest.mark.parametrize("csr", [False, True])
@pytest.mark.parametrize("metric_api", [False, True])
def test_leiden_pricing_routes_weights_resolution_seed_and_rescores(monkeypatch, csr, metric_api):
    """
    Checks dispatch, weight clipping, parameter forwarding, and exact rescoring.
    """
    observed = {}

    def backend(graph, *, weight, resolution, seed, backend):
        observed.update(graph=graph, weight=weight, resolution=resolution, seed=seed, backend=backend)
        return [{0, 1}, {2}]

    def modern_backend(graph, *, weight, resolution, seed, backend, metric):
        observed["metric"] = metric
        return legacy_backend(graph, weight=weight, resolution=resolution, seed=seed, backend=backend)

    legacy_backend = backend
    activate(monkeypatch, modern_backend if metric_api else backend)
    adjacency = np.array([[0., 2., .25], [2., 0., .5], [.25, .5, 0.]])
    degrees = adjacency.sum(axis=1)
    total = float(degrees.sum())
    pair_dual = np.zeros((3, 3))
    pair_dual[0, 2] = pair_dual[2, 0] = 1.
    duals = {"pair": sparse.csr_matrix(pair_dual) if csr else pair_dual, "constant": .125}
    matrix = sparse.csr_matrix(adjacency) if csr else adjacency
    rc, partition = heuristic_subproblem(
        matrix, degrees, total, duals, algo="leiden", package="networkx",
        gamma=1.7, seed=19, column_storage="csr",
    )
    graph = observed["graph"]
    assert set(graph) == {0, 1, 2}
    assert not graph.has_edge(0, 2)
    assert graph[1][2]["weight"] == .5
    assert observed["resolution"] == 1.7
    assert observed["seed"] == 19
    assert observed["weight"] == "weight"
    assert observed["backend"] == "example"
    if metric_api:
        assert observed["metric"] == "modularity"
    assert sparse.issparse(partition)
    np.testing.assert_array_equal(partition.toarray(), np.equal.outer([0, 0, 1], [0, 0, 1]))
    assert rc == pytest.approx(compute_reduced_cost(matrix, degrees, total, partition, duals, gamma=1.7))


@pytest.mark.parametrize("per_function", [True, False])
def test_leiden_accepts_function_or_global_activation(monkeypatch, per_function):
    """
    Check if leiden works with a package-wide or a function specific backend activation.
    """
    def backend(graph, **kwargs):
        return [{0}]

    activate(monkeypatch, backend, per_function=per_function)
    assert community._run_networkx_leiden(None, resolution=1, seed=0) == [{0}]
