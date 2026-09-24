import networkx as nx
import numpy as np
import pytest

from asunder import CSDDecompositionConfig, refine_partition_modular_vfd, run_csd_decomposition
from asunder.base.algorithms.modular_VFD import modular_very_fortunate_descent
from asunder.base.column_generation.decomposition import CSD_decomposition
from asunder.load_balancing.utils.partition_generation import (
    check_balance,
    make_partitions_random,
)


def _one_hot_master(A, a, m, Z_star, f_stars, extract_dual=False, **_):
    selected = [1.0] + [0.0] * (len(Z_star) - 1)
    if extract_dual:
        return selected, {"mu_dual": 0.0}, float(f_stars[0])
    return selected, float(f_stars[0])


def _identity_generator(N, **_):
    return [np.eye(N, dtype=int)]


def _two_equal_groups_generator(N, **_):
    labels = np.arange(N) >= (N // 2)
    return [np.equal.outer(labels, labels).astype(int)]


def test_pricing_candidate_that_violates_cannot_link_is_not_final_solution():
    A = nx.to_numpy_array(nx.path_graph(3), dtype=float)

    def violating_pricing(A, a, m, duals, **_):
        return 0.0, np.ones_like(A, dtype=int)

    config = CSDDecompositionConfig(
        cannot_link=[(0, 2)],
        ifc_params={"generator": _identity_generator, "num": 1, "args": {"N": 3}},
        refine_post_loop=False,
        final_master_solve=False,
        max_iterations=1,
        disable_tqdm=True,
        verbose=-1,
    )
    result = run_csd_decomposition(
        A,
        config=config,
        master_fn=_one_hot_master,
        subproblem_fn=violating_pricing,
    )

    assert result.records[-1].partition_source == "pricing_candidate"
    assert result.records[-1].z_sol[0, 2] == 1
    assert result.final_partition[0, 2] == 0
    assert result.metadata["final_partition_source"] == "one_hot_relaxed_master"


def test_post_loop_refinement_that_violates_pairwise_constraints_is_discarded():
    A = nx.to_numpy_array(nx.path_graph(3), dtype=float)

    def pricing(A, a, m, duals, **_):
        return 0.0, np.eye(A.shape[0], dtype=int)

    def violating_refiner(A, partition, **_):
        return np.ones_like(A, dtype=int)

    config = CSDDecompositionConfig(
        cannot_link=[(0, 2)],
        ifc_params={"generator": _identity_generator, "num": 1, "args": {"N": 3}},
        refine_params={"refine_func": violating_refiner},
        use_refined_column=False,
        refine_post_loop=True,
        final_master_solve=False,
        max_iterations=1,
        disable_tqdm=True,
        verbose=-1,
    )
    result = run_csd_decomposition(
        A,
        config=config,
        master_fn=_one_hot_master,
        subproblem_fn=pricing,
    )

    assert len(result.records) == 1
    assert result.final_partition[0, 2] == 0


def test_unbalanced_post_loop_refinement_falls_back_to_master_partition():
    A = nx.to_numpy_array(nx.path_graph(4), dtype=float)

    def pricing(A, a, m, duals, **_):
        return 0.0, np.eye(A.shape[0], dtype=int)

    def unbalanced_refiner(A, partition, **_):
        return np.ones_like(A, dtype=int)

    config = CSDDecompositionConfig(
        additional_constraints={"LB": True, "K": 2, "R": 0},
        ifc_params={
            "generator": _two_equal_groups_generator,
            "num": 1,
            "args": {"N": 4},
        },
        refine_params={"refine_func": unbalanced_refiner},
        use_refined_column=False,
        refine_post_loop=True,
        final_master_solve=False,
        max_iterations=1,
        disable_tqdm=True,
        verbose=-1,
    )
    result = run_csd_decomposition(
        A,
        config=config,
        master_fn=_one_hot_master,
        subproblem_fn=pricing,
    )

    assert result.metadata["final_partition_source"] == "one_hot_relaxed_master"
    assert np.all(result.final_partition.sum(axis=1) == 2)


def test_disabling_in_loop_refinement_avoids_refiner_work_for_final_master():
    A = nx.to_numpy_array(nx.path_graph(4), dtype=float)
    calls = []

    def pricing(A, a, m, duals, **_):
        return 0.0, np.eye(A.shape[0], dtype=int)

    def tracking_refiner(A, partition, **_):
        calls.append(1)
        return partition

    config = CSDDecompositionConfig(
        ifc_params={"generator": _identity_generator, "num": 1, "args": {"N": 4}},
        refine_params={"refine_func": tracking_refiner},
        use_refined_column=False,
        refine_post_loop=False,
        final_master_solve=True,
        max_iterations=1,
        disable_tqdm=True,
        verbose=-1,
    )
    run_csd_decomposition(
        A,
        config=config,
        master_fn=_one_hot_master,
        subproblem_fn=pricing,
    )

    assert calls == []


def test_weighted_initial_generation_and_balance_check_use_node_mass():
    weights = np.array([3, 1, 2, 2])
    columns = make_partitions_random(
        N=4,
        K=2,
        R=0,
        node_weights=weights,
        max_K_increase=0,
        seed=7,
    )

    assert columns
    minimum, maximum, balanced = check_balance(
        columns[0],
        K=2,
        R=0,
        node_weights=weights,
    )
    assert balanced
    assert minimum == maximum == 4


def test_zero_k_increase_never_silently_changes_requested_community_count():
    columns = make_partitions_random(
        N=6,
        K=2,
        R=0,
        R_bounds=(1, 1),
        max_K_increase=0,
        seed=7,
    )

    assert columns == []


@pytest.mark.parametrize("cap", [4096, None])
def test_modular_vfd_adapter_supports_weights_constraints_and_resolution(cap, monkeypatch):
    from asunder.base.algorithms import modular_VFD as vfd_module

    constructor = vfd_module.partition_vector_to_2d_matrix
    observed_caps = []

    def capture_cap(labels, **options):
        observed_caps.append(options.get("max_dense_working_bytes", "missing"))
        return constructor(labels, **options)

    monkeypatch.setattr(vfd_module, "partition_vector_to_2d_matrix", capture_cap)
    A = nx.to_numpy_array(nx.cycle_graph(4), dtype=float)
    weights = np.array([3, 1, 2, 2])
    kwargs = dict(
        A=A,
        a=A.sum(axis=1),
        m=float(A.sum()),
        K=2,
        R=0,
        must_link=[(0, 1)],
        cannot_link=[(0, 2)],
        balance_weights=weights,
        gamma=1.5,
        restarts=1,
        local_iters=0,
        clustering_Ks=(2,),
        clustering_methods=(),
        wz_is_C_node=True,
        tabu_max_steps=0,
        shake_rounds=0,
        max_dense_working_bytes=cap,
    )
    direct = modular_very_fortunate_descent(wz=np.eye(4), **kwargs)
    adapted = refine_partition_modular_vfd(partition=np.eye(4), **kwargs)

    assert direct is not None
    assert adapted is not None
    z, metadata = direct
    assert metadata["resolution"] == 1.5
    assert metadata["total_balance_weight"] == 8
    assert z[0, 1] == 1
    assert z[0, 2] == 0
    assert np.all((z @ weights) == 4)
    assert np.array_equal(adapted, z)
    assert observed_caps and all(value == cap for value in observed_caps)
    with pytest.raises(MemoryError):
        refine_partition_modular_vfd(partition=np.eye(4), **{**kwargs, "max_dense_working_bytes": 1})


def test_contracted_refinement_preserves_identity_based_component_provenance():
    class IdentityToken:
        pass

    tokens = tuple(IdentityToken() for _ in range(4))
    captured = []
    A = nx.to_numpy_array(nx.path_graph(4), dtype=float)
    labels = np.array([0, 0, 1, 2])
    initial = np.equal.outer(labels, labels).astype(int)

    def pricing(A, a, m, duals, **_):
        return 0.0, np.eye(A.shape[0], dtype=int)

    def refiner(A, partition, component_members, **_):
        captured.append(component_members)
        return partition

    CSD_decomposition(
        A,
        A.sum(axis=1),
        float(A.sum()),
        _one_hot_master,
        pricing,
        columns=[initial],
        f_stars=[0.0],
        must_link=[(0, 1)],
        contract_graph=True,
        refine_params={
            "refine_func": refiner,
            "kwargs": {"component_members": tuple((token,) for token in tokens)},
        },
        refine_post_loop=True,
        final_master_solve=False,
        max_iterations=1,
        disable_tqdm=True,
        verbose=-1,
    )

    assert captured
    flattened = tuple(member for group in captured[0] for member in group)
    assert all(actual is expected for actual, expected in zip(flattened, tokens))
