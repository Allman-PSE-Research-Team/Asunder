import functools
from types import SimpleNamespace

import networkx as nx
import numpy as np
import pytest

import asunder.load_balancing.column_generation.LB as lb_module
from asunder.load_balancing.algorithms.VFD import refine_partition
from asunder.orchestrator import run_csd_decomposition
from asunder.types import DecompositionResult, IterationRecord


def _one_hot_master(A, a, m, Z_star, f_stars, extract_dual=False, **_):
    selected = [1.0] + [0.0] * (len(Z_star) - 1)
    if extract_dual:
        return selected, {"mu_dual": 0.0}, float(f_stars[0])
    return selected, float(f_stars[0])


def _zero_pricing(A, a, m, duals, **_):
    return 0.0, np.eye(A.shape[0], dtype=int)


@pytest.fixture
def workflow(monkeypatch):
    """Exercise the real workflow without an external optimization backend."""
    configs = []

    def run(A, *, a, m, config, **_):
        configs.append(config)
        return run_csd_decomposition(
            A, a=a, m=m, config=config,
            master_fn=_one_hot_master, subproblem_fn=_zero_pricing,
        )

    monkeypatch.setattr(lb_module, "run_csd_decomposition", run)
    monkeypatch.setattr(
        lb_module, "refine_partition",
        functools.partial(
            refine_partition, restarts=1, local_iters=0, clustering_methods=(),
            wz_is_C_node=True, tabu_max_steps=0, shake_rounds=0,
        ),
    )
    return configs


@pytest.mark.parametrize("generator", ["random", "ordered"])
@pytest.mark.parametrize(("K", "bounds", "expected_count"), [(2, (2, 2), 3), (3, (3, 3), 2)])
@pytest.mark.parametrize("refine", [False, True])
def test_explicit_bounds_allow_counts_above_and_below_k(
    workflow, generator, K, bounds, expected_count, refine,
):
    result = lb_module.LoadBalancer(
        nx.cycle_graph(6), K=K, R_bounds=bounds, ifc_generator=generator,
        refine=refine, max_iterations=1, disable_tqdm=True,
    )

    assert len(result.metadata["community_balance_weights"]) == expected_count
    assert set(result.metadata["community_balance_weights"].values()) == {bounds[0]}


def test_default_refinement_compares_counts_by_modularity(workflow):
    graph = nx.disjoint_union_all([nx.complete_graph(3) for _ in range(3)])
    result = lb_module.LoadBalancer(
        graph, K=2, R_bounds=(2, 5), max_iterations=1, disable_tqdm=True,
    )

    assert result.metadata["final_partition_source"] == "post_loop_refinement"
    assert len(result.metadata["community_balance_weights"]) == 3
    assert result.metadata["modularity"] > result.records[0].master_obj_val


@pytest.mark.parametrize("generator", ["random", "ordered"])
@pytest.mark.parametrize("contract", [False, True])
def test_explicit_bounds_preserve_weighted_links(workflow, generator, contract):
    graph = nx.path_graph(4)
    nx.set_node_attributes(graph, 2, "load")
    result = lb_module.LoadBalancer(
        graph, K=3, R_bounds=(4, 4), node_weight_attr="load",
        must_link=[(0, 1)], cannot_link=[(0, 2)], contract_graph=contract,
        ifc_generator=generator, max_iterations=1, disable_tqdm=True,
    )

    assert sorted(result.metadata["community_balance_weights"].values()) == [4, 4]
    assert result.final_partition[0, 1] == 1
    assert result.final_partition[0, 2] == 0


@pytest.mark.parametrize("bounds", [(4, 4), (None, 4), (4, None)])
def test_explicit_bounds_accept_single_contracted_component(workflow, bounds):
    result = lb_module.LoadBalancer(
        nx.path_graph(4), K=2, R_bounds=bounds,
        must_link=[(0, 1), (1, 2), (2, 3)], contract_graph=True,
        max_iterations=1, disable_tqdm=True,
    )

    assert np.all(result.final_partition == 1)
    assert result.metadata["final_partition_source"] == "contracted_trivial"


@pytest.mark.parametrize("bounds", [None, (1, 3)])
def test_single_contracted_component_keeps_other_feasibility_checks(workflow, bounds):
    with pytest.raises(RuntimeError, match="integral feasible partition"):
        lb_module.LoadBalancer(
            nx.path_graph(4), K=2, R_bounds=bounds,
            must_link=[(0, 1), (1, 2), (2, 3)], contract_graph=True,
            max_iterations=1, disable_tqdm=True,
        )


@pytest.mark.parametrize("bounds", [(0, 0), (4, 4), (7, 10)])
def test_explicit_bounds_reject_empty_count_interval(bounds):
    with pytest.raises(ValueError, match="no feasible positive community count"):
        lb_module.LoadBalancer(nx.cycle_graph(6), R_bounds=bounds, disable_tqdm=True)


def test_zero_lower_bound_caps_counts_by_nodes_with_weighted_loads(workflow):
    graph = nx.path_graph(4)
    nx.set_node_attributes(graph, 2, "load")
    result = lb_module.LoadBalancer(
        graph, K=8, R_bounds=(0, 4), node_weight_attr="load",
        max_iterations=1, disable_tqdm=True,
    )

    assert all(0 < load <= 4 for load in result.metadata["community_balance_weights"].values())
    assert workflow[0].ifc_params["args"]["K"] == 2
    assert workflow[0].ifc_params["args"]["max_K_increase"] == 2


def test_kr_only_keeps_existing_search_defaults(workflow):
    result = lb_module.LoadBalancer(
        nx.cycle_graph(6), K=2, R=0, max_iterations=1, disable_tqdm=True,
    )

    assert len(result.metadata["community_balance_weights"]) == 2
    assert workflow[0].ifc_params["args"]["K"] == 2
    assert workflow[0].ifc_params["args"]["max_K_increase"] == 0
    assert "K_search_radius" not in workflow[0].refine_params["kwargs"]


def test_builtin_cpp_receives_in_loop_and_post_loop_settings(monkeypatch):
    from asunder.load_balancing.algorithms import VFD_cpp

    calls = []

    @functools.wraps(VFD_cpp.refine_partition)
    def record(**kwargs):
        calls.append(kwargs)
        return np.equal.outer(np.arange(6) // 2, np.arange(6) // 2)

    def improving_pricing(A, a, m, duals, **_):
        return 1.0, np.eye(len(a), dtype=int)

    def run(A, *, a, m, config, **_):
        return run_csd_decomposition(
            A, a=a, m=m, config=config, master_fn=_one_hot_master,
            subproblem_fn=improving_pricing,
        )

    monkeypatch.setattr(VFD_cpp, "refine_partition", record)
    monkeypatch.setattr(lb_module, "run_csd_decomposition", run)
    graph = nx.path_graph(6)
    nx.set_node_attributes(graph, 2, "load")
    lb_module.LoadBalancer(
        graph, K=2, R_bounds=(4, 4), resolution=1.25, node_weight_attr="load",
        must_link=[(0, 1)], cannot_link=[(0, 3)], refinement_backend="cpp",
        refine_params={"kwargs": {"shake_rounds": 2}}, seed=7,
        max_iterations=1, disable_tqdm=True,
    )
    assert [kwargs["shake_rounds"] for kwargs in calls] == [0, 2]
    for kwargs in calls:
        assert (kwargs["K"], kwargs["K_search_radius"], kwargs["R_bounds"]) == (2, 1, (4, 4))
        assert kwargs["gamma"] == 1.25 and kwargs["w_coassoc"] == 0.0
        assert kwargs["seed"] == 7 and kwargs["clustering_seeds"] == (7,)
        assert kwargs["must_link"] == [(0, 1)] and kwargs["cannot_link"] == [(0, 3)]
        assert np.array_equal(kwargs["balance_weights"], np.full(6, 2))


def test_cpp_selector_obeys_disabled_refinement(workflow):
    lb_module.LoadBalancer(
        nx.path_graph(6), K=2, R=0, refine=False, refinement_backend="cpp",
        max_iterations=1, disable_tqdm=True,
    )
    assert workflow[0].refine_params == {}


def test_unknown_refinement_backend_is_rejected():
    with pytest.raises(ValueError, match="refinement_backend"):
        lb_module.LoadBalancer(nx.path_graph(4), refinement_backend="gpu")


@pytest.mark.parametrize("requested_radius", [None, 0])
def test_custom_refiner_receives_bounds_default_without_overwriting_limit(workflow, requested_radius):
    observed = []

    def refiner(A, partition, K_search_radius=0, seed=None):
        observed.append(K_search_radius)
        return partition

    kwargs = {} if requested_radius is None else {"K_search_radius": requested_radius}
    lb_module.LoadBalancer(
        nx.cycle_graph(6), K=2, R_bounds=(2, 2),
        refinement_backend="cpp", refine_params={"refine_func": refiner, "kwargs": kwargs},
        max_iterations=1, disable_tqdm=True,
    )

    assert observed and set(observed) == {1 if requested_radius is None else requested_radius}
    assert kwargs == ({} if requested_radius is None else {"K_search_radius": requested_radius})


@pytest.mark.parametrize(
    ("K", "bounds", "time_limit", "expected_calls", "expected_count"),
    [
        (2, (1, 3), None, [2, 3, 4, 5, 6], 3),
        (10, (2, 2), 15.0, [3], 3),
        (2, (1, 3), 5.0, [2, 3, 4], 3),
        (2, None, 5.0, [2], 2),
    ],
)
def test_projection_searches_counts_and_shares_time_budget(
    monkeypatch, K, bounds, time_limit, expected_calls, expected_count,
):
    graph = nx.cycle_graph(6)
    A = nx.to_numpy_array(graph)
    labels = np.repeat(np.arange(3), 2)
    target = np.equal.outer(labels, labels).astype(int)
    columns = [target, np.eye(6, dtype=int)]
    wz = 0.8 * columns[0] + 0.2 * columns[1]
    result = DecompositionResult(
        records=[IterationRecord(lambda_sol=[0.8, 0.2], columns=columns, sub_obj_val=1.0)],
        final_partition=None, final_master_obj=None,
    )
    solver = SimpleNamespace(type="appsi_highs", options={"time_limit": 99.0})
    elapsed = [0.0]
    calls = []
    limits = []

    def project(*, K, **_):
        calls.append(K)
        limits.append(solver.options["time_limit"])
        elapsed[0] += 2.0
        labels = np.arange(6) * K // 6
        z = np.equal.outer(labels, labels).astype(int)
        return z, {
            "K_used": K, "requested_K": K,
            "projection_distance": float(np.sum((z - wz) ** 2)),
        }

    monkeypatch.setattr(lb_module, "get_default_solver", lambda: solver)
    monkeypatch.setattr(lb_module, "project_partition_ilp", project)
    monkeypatch.setattr(lb_module.time, "perf_counter", lambda: elapsed[0])
    repaired, metadata = lb_module._project_after_failed_post_loop_refinement(
        result=result, A=A, a=A.sum(axis=1), m=float(A.sum()), K=K, R=1,
        R_bounds=bounds, must_link=[], cannot_link=[], balance_weights=np.ones(6, dtype=int),
        seed=42, projection_time_limit=time_limit,
    )

    assert calls == expected_calls
    assert len(np.unique(repaired, axis=0)) == expected_count
    assert metadata["K_used"] == expected_count
    assert metadata["requested_K"] == K
    assert metadata["projection_time_limit"] == time_limit
    assert metadata["projection_time_limit_applied"] == (time_limit is not None)
    assert solver.options == {"time_limit": 99.0}
    if time_limit is not None:
        assert limits == [time_limit - 2 * index for index in range(len(calls))]
