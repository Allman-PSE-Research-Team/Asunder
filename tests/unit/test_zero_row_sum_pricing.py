import numpy as np
import pytest

from asunder.base.column_generation import subproblem as subproblem_module
from asunder.base.column_generation.subproblem import heuristic_subproblem


def test_zero_row_sum_scikit_network_receives_degree_preserving_adjacency(
    monkeypatch,
):
    adjacency = np.array([[0.0, 2.0], [2.0, 0.0]])
    dual_weight = np.array([[0.0, 0.1], [0.1, 0.0]])
    observed = {}

    def fake_run_modularity(modified_A, **kwargs):
        observed["adjacency"] = modified_A.copy()
        observed["kwargs"] = kwargs
        return np.eye(2, dtype=int), 0.0

    monkeypatch.setattr(subproblem_module, "run_modularity", fake_run_modularity)

    heuristic_subproblem(
        adjacency,
        adjacency.sum(axis=1),
        float(adjacency.sum()),
        {"pair_dual": dual_weight},
        algo="louvain",
        package="sknetwork",
        use_zero_row_sum=True,
    )

    expected = np.array([[0.4, 1.6], [1.6, 0.4]])
    np.testing.assert_allclose(observed["adjacency"], expected)
    np.testing.assert_allclose(
        observed["adjacency"].sum(axis=1),
        adjacency.sum(axis=1),
    )


def test_zero_row_sum_scikit_network_rejects_negative_adjacency():
    adjacency = np.array([[0.0, 1.0], [1.0, 0.0]])
    dual_weight = np.array([[0.0, 0.75], [0.75, 0.0]])

    with pytest.raises(ValueError, match="produced negative weights"):
        heuristic_subproblem(
            adjacency,
            adjacency.sum(axis=1),
            float(adjacency.sum()),
            {"pair_dual": dual_weight},
            algo="leiden",
            package="sknetwork",
            use_zero_row_sum=True,
        )


def test_zero_row_sum_rejects_unsupported_backend():
    adjacency = np.array([[0.0, 1.0], [1.0, 0.0]])

    with pytest.raises(ValueError, match="not supported"):
        heuristic_subproblem(
            adjacency,
            adjacency.sum(axis=1),
            float(adjacency.sum()),
            {},
            algo="louvain",
            package="networkx",
            use_zero_row_sum=True,
        )


def test_zero_row_sum_requires_exact_reduced_cost():
    adjacency = np.array([[0.0, 1.0], [1.0, 0.0]])

    with pytest.raises(ValueError, match="requires exact_rc=True"):
        heuristic_subproblem(
            adjacency,
            adjacency.sum(axis=1),
            float(adjacency.sum()),
            {},
            algo="louvain",
            package="sknetwork",
            exact_rc=False,
            use_zero_row_sum=True,
        )


def test_zero_row_sum_signed_backend_receives_unclipped_graph_weights(monkeypatch):
    adjacency = np.array([[0.0, 1.0], [1.0, 0.0]])
    dual_weight = np.array([[0.0, 0.75], [0.75, 0.0]])
    observed = {}

    def fake_run_leidenalg(modified_A, **kwargs):
        observed["adjacency"] = modified_A.copy()
        observed["kwargs"] = kwargs
        return np.eye(2, dtype=int), 0.0

    monkeypatch.setattr(subproblem_module, "run_leidenalg", fake_run_leidenalg)

    heuristic_subproblem(
        adjacency,
        adjacency.sum(axis=1),
        float(adjacency.sum()),
        {"pair_dual": dual_weight},
        algo="signed_leiden",
        package="leidenalg",
        use_zero_row_sum=True,
    )

    # Graph packages count loop edges twice in weighted degree, so the
    # diagonal is halved at the adapter boundary while signed edges remain.
    expected = np.array([[0.75, -0.5], [-0.5, 0.75]])
    np.testing.assert_allclose(observed["adjacency"], expected)
