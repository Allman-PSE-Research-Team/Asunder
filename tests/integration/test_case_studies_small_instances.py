import math
import os
from collections import defaultdict

import numpy as np
import pytest

import asunder.evaluation.runner as runner
from asunder.algorithms.community import refine_partition_linear_group
from asunder.case_studies.circle_cutting import build_circle_cutting_graph
from asunder.case_studies.cpcong import build_cpcong_graph
from asunder.column_generation.decomposition import CSD_decomposition
from asunder.column_generation.master import compute_f_star, solve_master_problem
from asunder.column_generation.subproblem import custom_heuristic_subproblem
from asunder.evaluation.metrics import permuted_accuracy
from asunder.evaluation.runner import run_evaluation
from asunder.solvers import set_default_solver
from asunder.utils.graph import partition_matrix_to_vector
from asunder.utils.partition_generation import make_simple_partition


def _assert_metric_payload(payload, *, expect_gap=False, require_nonnegative_gap=False):
    expected_keys = {"NMI", "ARI", "VI", "Accuracy", "time"}
    if expect_gap:
        expected_keys.add("Gap")
    assert set(payload.keys()) == expected_keys
    assert 0.0 <= payload["NMI"] <= 1.0
    assert -1.0 <= payload["ARI"] <= 1.0
    assert payload["VI"] >= -1e-8
    assert 0.0 <= payload["Accuracy"] <= 1.0
    assert payload["time"] >= 0.0
    assert math.isfinite(payload["time"])
    if expect_gap:
        assert math.isfinite(payload["Gap"])
        if require_nonnegative_gap:
            assert payload["Gap"] >= 0.0


def test_cpcong_small_instance_builder_shape_and_tags():
    K, J, T = 2, 3, 5
    G, constraint_labels, var_to_constraints = build_cpcong_graph(K=K, J=J, T=T)

    expected_nodes = (J * T) + (K * J * T) + (K * T) + (K * T)
    assert G.number_of_nodes() == expected_nodes
    assert G.number_of_edges() > 0
    assert constraint_labels["L"] == "Lead Time (Nonlinear)"
    assert constraint_labels["CB"] == "Cummulative Build"
    assert var_to_constraints

    node_constraints = {attrs["constraint"] for _, attrs in G.nodes(data=True)}
    assert "L" in node_constraints
    assert "CU" in node_constraints

    for _, _, attrs in G.edges(data=True):
        assert attrs["weight"] >= 1
        assert attrs["var_type"] in {"integer", "continuous"}


def test_circle_cutting_small_instance_builder_shape_and_tags():
    num_circles, num_rectangles = 4, 5
    dims = ["x", "y"]
    G, constraint_labels, var_to_constraints = build_circle_cutting_graph(
        num_circles=num_circles,
        num_rectangles=num_rectangles,
        dimensions=dims,
    )

    expected_nodes = num_circles + (num_rectangles * (num_circles * (num_circles - 1) // 2)) + (
        num_rectangles * num_circles
    )
    assert G.number_of_nodes() == expected_nodes
    assert G.number_of_edges() > 0
    assert constraint_labels["CA"] == "Circle Assignment"
    assert constraint_labels["NO"] == "Non-overlap"
    assert var_to_constraints

    node_constraints = {attrs["constraint"] for _, attrs in G.nodes(data=True)}
    assert "CA" in node_constraints
    assert "NO" in node_constraints
    assert "BP" in node_constraints

    for _, _, attrs in G.edges(data=True):
        assert attrs["weight"] >= 1
        assert attrs["var_type"] in {"integer", "continuous"}


def test_run_evaluation_cpcong_small_instance_spec_smoke():
    results = run_evaluation(
        problem="cpcong",
        build_params={"K": 2, "J": 3, "T": 5},
        style="CP",
        algos=["SPEC"],
        repeat=1,
    )
    assert set(results.keys()) == {"SPEC"}
    _assert_metric_payload(results["SPEC"])


def test_run_evaluation_circcut_small_instance_spec_smoke():
    results = run_evaluation(
        problem="circcut",
        build_params={"num_circles": 4, "num_rectangles": 5, "dimensions": ["x", "y"]},
        style="CP",
        algos=["SPEC"],
        repeat=1,
    )
    assert set(results.keys()) == {"SPEC"}
    _assert_metric_payload(results["SPEC"])


def test_run_evaluation_cd_refine_handles_partition_tuple_without_solver(monkeypatch):
    def fake_csd_decomposition(A, a, m, mp_function, sp_function, **kwargs):
        return [{"z_sol": np.eye(A.shape[0], dtype=int)}]

    monkeypatch.setattr(runner, "CSD_decomposition", fake_csd_decomposition)

    results = run_evaluation(
        problem="circcut",
        build_params={"num_circles": 3, "num_rectangles": 1, "dimensions": ["x", "y"]},
        style="CD_Refine",
        algos=["spectral"],
        repeat=1,
    )
    assert set(results.keys()) == {"spectral"}
    _assert_metric_payload(results["spectral"], expect_gap=True, require_nonnegative_gap=False)


@pytest.mark.solver
def test_run_evaluation_cd_refine_cpcong_calls_partitioned_gt(monkeypatch):
    _configure_solver_or_skip()
    calls = {"count": 0}
    original = runner.partititon_periphery_components

    def wrapped(A, labels, return_sparse=False):
        calls["count"] += 1
        return original(A, labels, return_sparse=return_sparse)

    monkeypatch.setattr(runner, "partititon_periphery_components", wrapped)

    results = run_evaluation(
        problem="cpcong",
        build_params={"K": 2, "J": 3, "T": 5},
        style="CD_Refine",
        algos=["spectral"],
        repeat=1,
    )
    assert set(results.keys()) == {"spectral"}
    _assert_metric_payload(results["spectral"], expect_gap=True, require_nonnegative_gap=True)
    assert calls["count"] == 1


@pytest.mark.solver
def test_run_evaluation_cd_refine_circcut_calls_partitioned_gt(monkeypatch):
    _configure_solver_or_skip()
    calls = {"count": 0}
    original = runner.partititon_periphery_components

    def wrapped(A, labels, return_sparse=False):
        calls["count"] += 1
        return original(A, labels, return_sparse=return_sparse)

    monkeypatch.setattr(runner, "partititon_periphery_components", wrapped)

    results = run_evaluation(
        problem="circcut",
        build_params={"num_circles": 4, "num_rectangles": 5, "dimensions": ["x", "y"]},
        style="CD_Refine",
        algos=["spectral"],
        repeat=1,
    )
    assert set(results.keys()) == {"spectral"}
    _assert_metric_payload(results["spectral"], expect_gap=True, require_nonnegative_gap=True)
    assert calls["count"] == 1


def _configure_solver_or_skip():
    require_solver = os.environ.get("ASUNDER_REQUIRE_SOLVER_TESTS", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }

    try:
        import pyomo.environ as _  # noqa: F401
        from pyomo.opt import SolverFactory
    except Exception as exc:
        if require_solver:
            pytest.fail(f"Pyomo solver stack unavailable but required: {exc}")
        pytest.skip(f"Pyomo solver stack unavailable: {exc}")

    for name in ("appsi_highs", "highs", "glpk", "cbc", "gurobi_direct"):
        try:
            solver = SolverFactory(name)
            if solver is not None and solver.available(False):
                set_default_solver(solver)
                break
        except Exception:
            continue
    else:
        if require_solver:
            pytest.fail(
                "No available Pyomo solver for real CD_Refine quality test. "
                "Set up one of: appsi_highs, highs, glpk, cbc, gurobi_direct."
            )
        pytest.skip("No available Pyomo solver for real CD_Refine quality test.")

    try:
        import sknetwork  # noqa: F401
    except Exception as exc:
        if require_solver:
            pytest.fail(f"scikit-network is required for refinement tests but unavailable: {exc}")
        pytest.skip(f"scikit-network unavailable for package refinement tests: {exc}")


def _assert_real_decomposition_quality(A, labels_gt, results):
    assert results, "CSD_decomposition should return iteration records."
    final_z = results[-1]["z_sol"]
    assert final_z is not None

    a = A.sum(axis=1)
    m = a.sum()
    baseline = np.ones((A.shape[0], A.shape[0]), dtype=int)
    baseline_score = compute_f_star(A, a, m, baseline, algo="louvain")
    best_score = max(results[-1]["f_stars"])
    assert best_score > baseline_score + 1e-8

    priced_columns = [row["sub_obj_val"] for row in results if row["sub_obj_val"] is not None]
    assert priced_columns and any(v > 1e-8 for v in priced_columns)

    labels_sol = partition_matrix_to_vector(final_z)
    n_clusters = np.unique(labels_sol).size
    assert 1 < n_clusters < A.shape[0]

    acc = permuted_accuracy(labels_gt, labels_sol)[0]
    assert acc >= 0.5


def _tracked_package_refine(counter):
    def _refine(A, partition, **kwargs):
        counter["count"] += 1
        return refine_partition_linear_group(A=A, partition=partition, **kwargs)

    return _refine


@pytest.mark.solver
def test_cd_refine_real_decomposition_quality_cpcong_small_instance():
    _configure_solver_or_skip()

    G, _, _ = build_cpcong_graph(K=2, J=3, T=5)
    labels_gt = np.array([attr["constraint"] == "CU" for _, attr in G.nodes(data=True)], dtype=int)

    A = runner.nx.to_numpy_array(G)
    n = A.shape[0]

    labels_gt, _ = runner.partititon_periphery_components(A, labels_gt, return_sparse=False)

    node_labels = list(G.nodes())
    label_node_map = {label: i for i, label in enumerate(node_labels)}
    worthy_edges = [
        (label_node_map[i], label_node_map[j])
        for i, j, attr in G.edges(data=True)
        if attr.get("var_type") == "integer"
    ]

    ifc_params = {"generator": make_simple_partition, "num": 1, "args": {"N": n}}
    additional_constraints = defaultdict(lambda: None, {"worthy_edges": worthy_edges})
    a = A.sum(axis=1)
    m = a.sum()
    refine_counter = {"count": 0}
    refine_params = {
        "refine_func": _tracked_package_refine(refine_counter),
        "kwargs": {"prob_method": "DBSCAN", "verbose": False},
    }

    results = CSD_decomposition(
        A,
        a,
        m,
        solve_master_problem,
        custom_heuristic_subproblem,
        additional_constraints=additional_constraints,
        algo="spectral",
        package=None,
        extract_dual=True,
        ifc_params=ifc_params,
        refine_in_subproblem=False,
        refine_params=refine_params,
        use_refined_column=True,
        final_master_solve=False,
        max_iterations=20,
        tolerance=1e-8,
        verbose=-1,
    )

    _assert_real_decomposition_quality(A, labels_gt, results)
    assert refine_counter["count"] >= 1


@pytest.mark.solver
def test_cd_refine_real_decomposition_quality_circcut_small_instance():
    _configure_solver_or_skip()

    G, _, _ = build_circle_cutting_graph(num_circles=4, num_rectangles=5, dimensions=["x", "y"])
    labels_gt = np.array([attr["constraint"] == "CA" for _, attr in G.nodes(data=True)], dtype=int)
    A = runner.nx.to_numpy_array(G)
    n = A.shape[0]

    labels_gt, _ = runner.partititon_periphery_components(A, labels_gt, return_sparse=False)

    node_labels = list(G.nodes())
    label_node_map = {label: i for i, label in enumerate(node_labels)}
    worthy_edges = [
        (label_node_map[i], label_node_map[j])
        for i, j, attr in G.edges(data=True)
        if attr.get("var_type") == "integer"
    ]

    ifc_params = {"generator": make_simple_partition, "num": 1, "args": {"N": n}}
    additional_constraints = defaultdict(lambda: None, {"worthy_edges": worthy_edges})
    a = A.sum(axis=1)
    m = a.sum()
    refine_counter = {"count": 0}
    refine_params = {
        "refine_func": _tracked_package_refine(refine_counter),
        "kwargs": {"prob_method": "gaussian_mixture", "verbose": False},
    }

    results = CSD_decomposition(
        A,
        a,
        m,
        solve_master_problem,
        custom_heuristic_subproblem,
        additional_constraints=additional_constraints,
        algo="spectral",
        package=None,
        extract_dual=True,
        ifc_params=ifc_params,
        refine_in_subproblem=False,
        refine_params=refine_params,
        use_refined_column=True,
        final_master_solve=False,
        max_iterations=20,
        tolerance=1e-8,
        verbose=-1,
    )

    _assert_real_decomposition_quality(A, labels_gt, results)
    assert refine_counter["count"] >= 1
