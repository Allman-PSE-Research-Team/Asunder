from types import SimpleNamespace

import networkx as nx
import numpy as np
import pytest

import asunder.nlbnp.algorithms.refinement as refinement_module
import asunder.nlbnp.workflow as workflow_module
from asunder.base.algorithms.community import probability_to_integer_labels
from asunder.nlbnp import (
    CorePeripheryPartition,
    NonlinearBranchAndPrice,
    run_nonlinear_branch_and_price,
)
from asunder.nlbnp.algorithms.refinement import (
    refine_partition_linear_group,
    refine_partition_with_cp,
)


def _fake_cp_result(labels, **metadata):
    labels = np.asarray(labels, dtype=int)
    return SimpleNamespace(
        node_labels=labels,
        to_metadata=lambda: {"core_labels": labels.copy(), **metadata},
    )


def _master(A, a, m, Z_star, f_stars, extract_dual=False, **_):
    lambda_sol = [1.0] + [0.0] * (len(Z_star) - 1)
    obj = float(f_stars[0] if f_stars else 0.0)
    if extract_dual:
        return lambda_sol, {"mu_dual": 0.0}, obj
    return lambda_sol, obj


def _subproblem(A, a, m, duals, **_):
    return 0.0, np.eye(A.shape[0], dtype=int)


def test_nonlinear_branch_and_price_accepts_labeled_graph_constraints():
    G = nx.Graph()
    G.add_edges_from([("a", "b"), ("b", "c"), ("c", "d")])

    result = NonlinearBranchAndPrice(
        G,
        worthy_edges=[("a", "b"), ("b", "c"), ("c", "d")],
        must_link=[("a", "b")],
        cannot_link=[("a", "d")],
        nonlinear_nodes=["a", "b"],
        master_fn=_master,
        subproblem_fn=_subproblem,
        use_refined_column=False,
        final_master_solve=False,
        disable_tqdm=True,
        verbose=-1,
    )

    assert result.records
    assert result.final_partition.shape == (4, 4)
    assert result.metadata["label_node_map"] == {"a": 0, "b": 1, "c": 2, "d": 3}
    assert result.metadata["worthy_edges"] == [(0, 1), (1, 2), (2, 3)]
    assert result.metadata["user_must_link"] == [(0, 1)]
    assert result.metadata["user_cannot_link"] == [(0, 3)]
    assert result.metadata["eligible_nodes"] == [2, 3]
    assert result.metadata["contract_graph"] is True
    partition = result.final_partition
    assert partition[0, 1] == 1
    assert partition[0, 3] == 0
    for i in range(partition.shape[0]):
        for j in range(partition.shape[0]):
            for k in range(partition.shape[0]):
                if partition[i, j] and partition[j, k]:
                    assert partition[i, k]
    assert "community_map_labels" in result.metadata


def test_nonlinear_branch_and_price_can_derive_worthy_edges_from_attribute():
    G = nx.Graph()
    G.add_node("x", node_kind="nonlinear")
    G.add_edge("x", "y", edge_kind="integer")
    G.add_edge("y", "z", edge_kind="continuous")

    result = run_nonlinear_branch_and_price(
        G,
        worthy_edge_attr="edge_kind",
        worthy_edge_value="integer",
        nonlinear_node_attr="node_kind",
        nonlinear_node_value="nonlinear",
        master_fn=_master,
        subproblem_fn=_subproblem,
        use_refined_column=False,
        final_master_solve=False,
        disable_tqdm=True,
        verbose=-1,
    )

    assert result.metadata["worthy_edges"] == [(0, 1)]


def test_nonlinear_branch_and_price_accepts_adjacency_matrix():
    A = np.array(
        [
            [0, 1, 0],
            [1, 0, 1],
            [0, 1, 0],
        ],
        dtype=float,
    )

    result = NonlinearBranchAndPrice(
        A,
        worthy_edges=[(1, 2)],
        nonlinear_nodes=[0],
        master_fn=_master,
        subproblem_fn=_subproblem,
        use_refined_column=False,
        final_master_solve=False,
        disable_tqdm=True,
        verbose=-1,
    )

    assert result.final_partition.shape == (3, 3)
    assert result.metadata["node_label_map"] == {0: 0, 1: 1, 2: 2}
    assert result.metadata["worthy_edges"] == [(1, 2)]


def test_refine_partition_with_cp_merges_linear_periphery_and_preserves_core(monkeypatch):
    partition = np.array([0, 0, 1, 1])
    core_labels = np.array([1, 0, 1, 0])

    monkeypatch.setattr(
        refinement_module,
        "_detect_core_periphery",
        lambda A, **kwargs: _fake_cp_result(core_labels, primary_fit=1.0),
    )

    refined = refine_partition_with_cp(
        np.eye(4),
        partition,
        nonlinear_nodes=[0, 2],
    )

    assert np.array_equal(
        refined,
        np.array(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 1],
                [0, 0, 1, 0],
                [0, 1, 0, 1],
            ]
        ),
    )


def test_confidence_refinement_merges_existing_groups_and_adds_low_confidence_nodes(
    monkeypatch,
):
    partition = np.array([0, 0, 1, 2, 2])
    probabilities = np.array(
        [
            [0.9, 0.1, 0.0],
            [0.6, 0.4, 0.0],
            [0.0, 0.9, 0.1],
            [0.0, 0.1, 0.9],
            [0.0, 0.1, 0.9],
        ]
    )
    monkeypatch.setattr(
        refinement_module,
        "labels_to_probabilities",
        lambda *args, **kwargs: SimpleNamespace(toarray=lambda: probabilities),
    )
    converted = {}

    def fake_probability_to_labels(values, **kwargs):
        converted["values"] = values
        converted["kwargs"] = kwargs
        return np.array([0, -1, 1, 2, 2])

    monkeypatch.setattr(
        refinement_module,
        "probability_to_integer_labels",
        fake_probability_to_labels,
    )

    refined = refine_partition_linear_group(
        nx.to_numpy_array(nx.path_graph(5)),
        partition,
        nonlinear_nodes=[0],
        threshold=0.8,
    )

    assert np.all(refined[1:, 1:] == 1)
    assert np.all(refined[0, 1:] == 0)
    assert converted["values"] is probabilities
    assert converted["kwargs"]["method"] == "threshold"


def test_reformulated_cardinality_reports_cannot_link_inside_maximum_set():
    A = nx.to_numpy_array(nx.disjoint_union(nx.path_graph(2), nx.path_graph(2)))

    result = NonlinearBranchAndPrice(
        A,
        worthy_edges=[(0, 1)],
        nonlinear_nodes=[0, 1],
        cannot_link=[(2, 3)],
    )

    assert result.final_partition is None
    assert result.metadata["status"] == "infeasible"
    assert (
        result.metadata["infeasible_reason"]
        == "cannot_link_inside_maximum_eligible_set"
    )
    assert result.metadata["eligible_nodes"] == [2, 3]


def test_core_periphery_partition_merges_linear_periphery_and_splits_original_core(
    monkeypatch,
):
    captured = {}

    def fake_detect(A, **kwargs):
        captured.update(kwargs)
        return _fake_cp_result(
            [1, 1, 0, 1, 1, 0],
            algorithm="SPEC",
            target_space="contracted",
            primary_fit=1.0,
            must_group_role="core",
        )

    monkeypatch.setattr(workflow_module, "_detect_core_periphery", fake_detect)
    G = nx.Graph()
    G.add_nodes_from(
        [
            ("nonlinear-a", {"role": "nonlinear"}),
            ("a", {}),
            ("linear-only-a", {}),
            ("nonlinear-b", {"role": "nonlinear"}),
            ("b", {}),
            ("linear-only-b", {}),
        ]
    )
    G.add_edges_from(
        [
            ("nonlinear-a", "a", {"kind": "continuous"}),
            ("nonlinear-a", "linear-only-a", {"kind": "integer"}),
            ("linear-only-a", "nonlinear-b", {"kind": "integer"}),
            ("nonlinear-b", "b", {"kind": "continuous"}),
            ("nonlinear-a", "linear-only-b", {"kind": "integer"}),
            ("linear-only-b", "nonlinear-b", {"kind": "integer"}),
        ]
    )

    labels, metadata = CorePeripheryPartition(
        G,
        must_link_edge_attr="kind",
        must_link_edge_value="continuous",
        must_group_node_attr="role",
        must_group_node_value="nonlinear",
    )

    assert np.array_equal(labels, np.array([1, 1, 0, 2, 2, 0]))
    assert captured["must_link"] == [(0, 1), (3, 4)]
    assert captured["must_group"] == [0, 3]
    assert metadata["community_map_labels"] == {
        "nonlinear-a": 1,
        "a": 1,
        "linear-only-a": 0,
        "nonlinear-b": 2,
        "b": 2,
        "linear-only-b": 0,
    }
    assert metadata["communities_labels"] == [
        ["linear-only-a", "linear-only-b"],
        ["nonlinear-a", "a"],
        ["nonlinear-b", "b"],
    ]
    assert metadata["component_graph_space"] == "original"
    assert metadata["n_linear_only"] == 2
    assert metadata["n_core"] == 4
    assert metadata["n_periphery"] == 2
    assert metadata["n_independent_nodes"] == 4
    assert metadata["linear_only_node_indices"].tolist() == [2, 5]
    assert metadata["independent_components"] == [{0, 1}, {3, 4}]


def test_core_periphery_allows_disconnected_nonlinear_block_nodes_to_split(monkeypatch):
    """must_group constrains detection, not final core-side connectivity."""

    def fake_detect(A, **kwargs):
        labels = np.array([1, 0, 1])
        return _fake_cp_result(
            labels,
            algorithm="SPEC",
            primary_fit=0.0,
            must_group_role="core",
        )

    monkeypatch.setattr(workflow_module, "_detect_core_periphery", fake_detect)
    G = nx.Graph()
    G.add_nodes_from(["a", "b", "c"])
    G.add_edge("a", "b")

    labels, _ = CorePeripheryPartition(
        G,
        must_group=["a", "c"],
    )

    assert labels[0] != labels[2]
    assert labels[1] == 0


def test_core_periphery_rejects_nonlinear_block_on_periphery(monkeypatch):
    monkeypatch.setattr(
        workflow_module,
        "_detect_core_periphery",
        lambda A, **kwargs: _fake_cp_result(
            [0, 1, 0],
            algorithm="SPEC",
            must_group_role="periphery",
        ),
    )

    with pytest.raises(RuntimeError, match="nonlinear block was not detected in the core"):
        CorePeripheryPartition(nx.path_graph(3), must_group=[0, 2])


@pytest.mark.parametrize("algorithm", ["SPEC", "GA", "KL"])
def test_core_periphery_algorithms_preserve_grouping_blocks(algorithm):
    """Every CP backend returns one binary assignment per grouping block."""
    A = nx.to_numpy_array(nx.path_graph(4), dtype=float)

    result = workflow_module._detect_core_periphery(
        A,
        must_link=[(0, 3)],
        must_group=[1, 2],
        algorithm=algorithm,
        prob_method="threshold",
        threshold=0.5,
        seed=7,
        kl_max_iter=2,
        ga_population_size=8,
        ga_generations=2,
    )

    assert result.node_labels[0] == result.node_labels[3]
    assert result.node_labels[1] == result.node_labels[2]


@pytest.mark.parametrize(
    "worthy_edges",
    [None, (), ((0, 2),)],
)
def test_nonlinear_branch_and_price_requires_structural_worthy_edges(worthy_edges):
    A = nx.to_numpy_array(nx.path_graph(3))

    with pytest.raises(ValueError, match="worthy structural edge|not nonzero structural"):
        NonlinearBranchAndPrice(
            A,
            worthy_edges=worthy_edges,
            nonlinear_nodes=[0],
        )


def test_stage_one_refinement_flags_require_a_callable():
    A = nx.to_numpy_array(nx.path_graph(2))

    with pytest.raises(ValueError, match="requires refine_params"):
        NonlinearBranchAndPrice(
            A,
            worthy_edges=[(0, 1)],
            nonlinear_nodes=[0],
            refine_post_loop=True,
        )


def test_core_periphery_partition_real_spectral_path():
    G = nx.Graph()
    G.add_edges_from(
        [
            ("core", "a"),
            ("core", "b"),
            ("core", "c"),
            ("a", "a2"),
            ("b", "b2"),
            ("c", "c2"),
        ]
    )

    labels, metadata = CorePeripheryPartition(
        G,
        cp_algorithm="SPEC",
        prob_method="threshold",
        threshold=0.5,
    )

    assert labels.shape == (G.number_of_nodes(),)
    assert metadata["core_labels"].shape == labels.shape
    assert metadata["algorithm"] == "SPEC"
    assert metadata["n_communities"] == np.unique(labels).size


def test_core_periphery_partition_handles_single_node_graph():
    with pytest.raises(RuntimeError, match="requires nonempty nonlinear-core"):
        CorePeripheryPartition(nx.empty_graph(1))
