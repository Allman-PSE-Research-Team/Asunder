import networkx as nx
import numpy as np

import asunder.nlbnp.algorithms.refinement as refinement_module
import asunder.nlbnp.workflow as workflow_module
from asunder.nlbnp import CorePeripheryPartition, NonlinearBranchAndPrice, run_nonlinear_branch_and_price
from asunder.nlbnp.algorithms.refinement import refine_partition_with_cp
from asunder.types import DecompositionResult


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
        worthy_edges=[("a", "b")],
        must_link=[("a", "b")],
        cannot_link=[("a", "d")],
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
    assert result.metadata["worthy_edges"] == [(0, 1)]
    assert result.metadata["must_link"] == [(0, 1)]
    assert result.metadata["cannot_link"] == [(0, 3)]
    assert result.records[0].columns[0][0, 3] == 0
    assert "community_map_labels" in result.metadata


def test_nonlinear_branch_and_price_can_derive_worthy_edges_from_attribute():
    G = nx.Graph()
    G.add_edge("x", "y", edge_kind="nonlinear")
    G.add_edge("y", "z", edge_kind="linear")

    result = run_nonlinear_branch_and_price(
        G,
        worthy_edge_attr="edge_kind",
        worthy_edge_value="nonlinear",
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


def test_refine_partition_with_cp_merges_core_and_preserves_periphery(monkeypatch):
    partition = np.array([0, 0, 1, 1])
    core_labels = np.array([1, 0, 1, 0])

    monkeypatch.setattr(
        refinement_module,
        "_detect_core_periphery",
        lambda A, **kwargs: (core_labels, {"core_score": 1.0}),
    )

    refined = refine_partition_with_cp(np.eye(4), partition)

    assert np.array_equal(
        refined,
        np.array(
            [
                [1, 0, 1, 0],
                [0, 1, 0, 0],
                [1, 0, 1, 0],
                [0, 0, 0, 1],
            ]
        ),
    )


def test_nonlinear_branch_and_price_accepts_core_periphery_refinement_hook(monkeypatch):
    captured = {}

    def fake_run(A, config, **kwargs):
        captured["config"] = config
        return DecompositionResult([], np.eye(A.shape[0], dtype=int), None)

    monkeypatch.setattr(workflow_module, "run_csd_decomposition", fake_run)
    refine_params = {
        "refine_func": refine_partition_with_cp,
        "kwargs": {
            "unworthy_edges": [(0, 1)],
            "nonlinear_nodes": [0],
            "cp_algorithm": "KL",
        },
    }

    result = NonlinearBranchAndPrice(
        np.eye(3),
        refine_params=refine_params,
    )

    cfg = captured["config"]
    assert cfg.refine_params["refine_func"] is refine_partition_with_cp
    assert cfg.refine_params["kwargs"]["unworthy_edges"] == [(0, 1)]
    assert cfg.refine_params["kwargs"]["nonlinear_nodes"] == [0]
    assert cfg.refine_params["kwargs"]["cp_algorithm"] == "KL"
    assert result.final_partition.shape == (3, 3)


def test_core_periphery_partition_splits_connected_periphery_components(monkeypatch):
    captured = {}

    def fake_detect(A, **kwargs):
        captured.update(kwargs)
        return np.array([1, 0, 0, 0, 0]), {
            "algorithm": "SPEC",
            "continuous_labels": np.zeros(5),
            "continuous_score": 0.0,
            "integer_labels": np.zeros(5, dtype=int),
            "core_score": 1.0,
        }

    monkeypatch.setattr(workflow_module, "_detect_core_periphery", fake_detect)
    G = nx.Graph()
    G.add_nodes_from(
        [
            ("core", {"role": "nonlinear"}),
            ("a", {}),
            ("b", {}),
            ("c", {}),
            ("d", {}),
        ]
    )
    G.add_edges_from(
        [
            ("core", "a", {"kind": "continuous"}),
            ("a", "b", {"kind": "integer"}),
            ("core", "c", {"kind": "continuous"}),
            ("c", "d", {"kind": "integer"}),
        ]
    )

    labels, metadata = CorePeripheryPartition(
        G,
        unworthy_edge_attr="kind",
        unworthy_edge_value="continuous",
        nonlinear_node_attr="role",
        nonlinear_node_value="nonlinear",
    )

    assert np.array_equal(labels, np.array([0, 1, 1, 2, 2]))
    assert captured["unworthy_edges"] == [(0, 1), (0, 3)]
    assert captured["nonlinear_nodes"] == [0]
    assert metadata["community_map_labels"] == {"core": 0, "a": 1, "b": 1, "c": 2, "d": 2}
    assert metadata["communities_labels"] == [["core"], ["a", "b"], ["c", "d"]]


def test_nonlinear_branch_and_price_refine_false_disables_refined_columns():
    A = np.array([[0, 1], [1, 0]], dtype=float)

    result = NonlinearBranchAndPrice(
        A,
        refine=False,
        master_fn=_master,
        subproblem_fn=_subproblem,
        final_master_solve=False,
        disable_tqdm=True,
        verbose=-1,
    )

    assert result.records
    assert result.final_partition.shape == (2, 2)


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
    labels, metadata = CorePeripheryPartition(nx.empty_graph(1))

    assert np.array_equal(labels, np.array([0]))
    assert metadata["core_labels"].tolist() == [1]
    assert metadata["n_communities"] == 1
