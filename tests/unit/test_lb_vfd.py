import inspect

import networkx as nx
import numpy as np
import pytest

from asunder.base.algorithms.modular_VFD import modular_very_fortunate_descent
from asunder.load_balancing.algorithms.VFD import (
    refine_partition,
    very_fortunate_descent,
)


def _cycle_adjacency(size=4):
    return nx.to_numpy_array(nx.cycle_graph(size), dtype=float)


def _small_search_kwargs():
    return {
        "restarts": 1,
        "local_iters": 0,
        "clustering_Ks": (2,),
        "clustering_methods": (),
        "wz_is_C_node": True,
        "tabu_max_steps": 0,
        "shake_rounds": 0,
    }


def test_lb_vfd_defaults_are_immutable():
    parameters = inspect.signature(very_fortunate_descent).parameters
    assert parameters["must_link"].default == ()
    assert parameters["cannot_link"].default == ()


def test_lb_vfd_enforces_weighted_balance_links_and_resolution():
    A = _cycle_adjacency()
    weights = np.array([3, 1, 2, 2])

    out = very_fortunate_descent(
        wz=np.eye(4),
        A=A,
        a=A.sum(axis=1),
        m=float(A.sum()),
        K=2,
        R=0,
        must_link=((0, 1),),
        cannot_link=((0, 2),),
        balance_weights=weights,
        gamma=1.5,
        seed=None,
        **_small_search_kwargs(),
    )

    assert out is not None
    z, metadata = out
    assert z[0, 1] == 1
    assert z[0, 2] == 0
    assert np.all(z @ weights == 4)
    assert metadata["K_used"] == 2
    assert metadata["r_min"] == metadata["r_max"] == 4
    assert metadata["resolution"] == 1.5
    assert metadata["total_balance_weight"] == 8
    assert metadata["seed"] is None

    expected = np.sum((A - 1.5 * np.outer(A.sum(axis=1), A.sum(axis=1)) / A.sum()) * z)
    assert np.isclose(metadata["objective_B_sum"], expected)


@pytest.mark.parametrize(
    ("K", "R_bounds", "expected_K"),
    ((2, (2, 2), 3), (3, (3, 3), 2)),
)
def test_lb_vfd_k_search_radius_checks_both_directions(K, R_bounds, expected_K):
    A = _cycle_adjacency(6)
    common = dict(
        wz=np.eye(6),
        A=A,
        a=A.sum(axis=1),
        m=float(A.sum()),
        K=K,
        R=0,
        R_bounds=R_bounds,
        **_small_search_kwargs(),
    )

    assert very_fortunate_descent(K_search_radius=0, **common) is None
    result = very_fortunate_descent(K_search_radius=1, **common)
    assert result is not None
    _, metadata = result
    assert metadata["K_used"] == expected_K


def test_modular_vfd_k_search_radius_checks_both_directions():
    A = _cycle_adjacency(6)
    common = dict(
        wz=np.eye(6),
        A=A,
        a=A.sum(axis=1),
        m=float(A.sum()),
        K=3,
        R=0,
        R_bounds=(3, 3),
        use_K_constraint=True,
        **_small_search_kwargs(),
    )

    assert modular_very_fortunate_descent(K_search_radius=0, **common) is None
    result = modular_very_fortunate_descent(K_search_radius=1, **common)
    assert result is not None
    _, metadata = result
    assert metadata["K_used"] == 2


def test_modular_vfd_explicit_candidate_ks_override_search_window():
    A = _cycle_adjacency(6)
    input_labels = np.array([0, 0, 0, 1, 1, 1])
    coassociation = np.equal.outer(input_labels, input_labels).astype(float)

    result = modular_very_fortunate_descent(
        wz=coassociation,
        A=A,
        a=A.sum(axis=1),
        m=float(A.sum()),
        K=2,
        K_search_radius=0,
        candidate_Ks=(3,),
        **_small_search_kwargs(),
    )

    assert result is not None
    _, metadata = result
    assert metadata["K_used"] == 3


def test_lb_refinement_adapter_derives_contracted_objective_inputs():
    A = _cycle_adjacency(3)
    weights = np.array([2, 1, 1])

    refined = refine_partition(
        A=A,
        partition=np.eye(3),
        K=2,
        R=0,
        balance_weights=weights,
        **_small_search_kwargs(),
    )

    assert refined is not None
    assert np.all(refined @ weights == 2)


def test_lb_vfd_and_modular_vfd_match_under_equivalent_forced_settings():
    A = _cycle_adjacency()
    common = dict(
        wz=np.eye(4),
        A=A,
        a=A.sum(axis=1),
        m=float(A.sum()),
        K=2,
        R=0,
        must_link=((0, 1),),
        cannot_link=((0, 2),),
        balance_weights=np.ones(4, dtype=int),
        gamma=1.0,
        seed=9,
        **_small_search_kwargs(),
    )

    lb_out = very_fortunate_descent(**common)
    modular_out = modular_very_fortunate_descent(
        **common,
        use_K_constraint=True,
        candidate_Ks=None,
    )

    assert lb_out is not None
    assert modular_out is not None
    assert np.array_equal(lb_out[0], modular_out[0])


def test_lb_and_modular_vfd_split_a_donor_block_to_repair_minimum_loads():
    A = _cycle_adjacency()
    initial_labels = np.array([0, 0, 0, 1])
    coassociation = np.equal.outer(initial_labels, initial_labels).astype(float)
    common = dict(
        wz=coassociation,
        A=A,
        a=A.sum(axis=1),
        m=float(A.sum()),
        K=2,
        R=1,
        R_bounds=(2, 3),
        **_small_search_kwargs(),
    )

    lb_out = very_fortunate_descent(**common)
    modular_out = modular_very_fortunate_descent(
        **common,
        use_K_constraint=True,
    )

    assert lb_out is not None
    assert modular_out is not None
    assert sorted(lb_out[0].sum(axis=1).tolist()) == [2, 2, 2, 2]
    assert sorted(modular_out[0].sum(axis=1).tolist()) == [2, 2, 2, 2]
