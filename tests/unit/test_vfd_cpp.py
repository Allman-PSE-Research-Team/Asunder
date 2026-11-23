"""Verify native action gains, feasible search results, and Asunder integration."""

import inspect

import numpy as np
import pytest
from threadpoolctl import threadpool_limits

from asunder.load_balancing.algorithms import VFD_cpp as adapter
from asunder.load_balancing.algorithms import _vfd_wrapper as _native
from asunder.load_balancing.algorithms.VFD import (
    refine_partition as python_refine,
)
from asunder.load_balancing.algorithms.VFD import (
    very_fortunate_descent as python_vfd,
)
from asunder.load_balancing.algorithms.VFD_cpp import refine_partition, very_fortunate_descent

pytestmark = pytest.mark.native


@pytest.fixture(scope="module", autouse=True)
def require_native_vfd():
    if not _native._library_path().is_file():
        pytest.skip("Build the optional VFD library with python tools/build_vfd.py")
    _native.load_library()


def objective(A, labels, gamma=1.0):
    """Compute the original node-level B objective independently of the kernel."""
    a = A.sum(axis=1)
    B = A - gamma * np.outer(a, a) / a.sum()
    return float((B * np.equal.outer(labels, labels)).sum())


def graph(seed=0, n=18):
    """Generate a small connected, weighted, undirected test graph."""
    rng = np.random.default_rng(seed)
    upper = np.triu(rng.random((n, n)) * (rng.random((n, n)) < 0.22), 1)
    A = upper + upper.T
    for i in range(n):
        A[i, (i + 1) % n] = A[(i + 1) % n, i] = 1
    return A


def assert_feasible(Z, metadata, A, weights=None, must=(), cannot=()):
    """Check a returned partition against original node-level constraints."""
    labels = metadata["labels"]
    if weights is None:
        weights = np.ones(len(labels), dtype=np.int64)
    np.testing.assert_array_equal(Z, np.equal.outer(labels, labels))
    assert len(np.unique(labels)) == metadata["K_used"]
    loads = [
        sum(int(weights[i]) for i in np.flatnonzero(labels == g)) for g in range(metadata["K_used"])
    ]
    assert all(metadata["r_min"] <= load <= metadata["r_max"] for load in loads)
    np.testing.assert_array_equal(loads, metadata["community_loads"])
    for i, j in must:
        assert labels[i] == labels[j]
    for i, j in cannot:
        assert labels[i] != labels[j]
    np.testing.assert_allclose(
        metadata["objective_B_sum"],
        objective(A, labels, metadata["resolution"]),
        atol=1e-9,
        rtol=1e-9,
    )


@pytest.mark.parametrize("seed", range(30))
@pytest.mark.parametrize("kind", ["move", "swap", "eject"])
def test_action_gains(seed, kind):
    """Compare native gains with a separately recomputed Python objective."""
    rng = np.random.default_rng(seed)
    raw_B = rng.normal(size=(12, 12))
    raw_C = rng.random((12, 12))
    B = np.asfortranarray((raw_B + raw_B.T) / 2)
    C = np.asfortranarray((raw_C + raw_C.T) / 2)
    np.fill_diagonal(C, 1)
    blocks = [[0, 1], [2], [3, 4], [5], [6, 7], [8], [9, 10], [11]]
    labels = np.repeat(np.arange(4), 3)
    first, second, destination = (
        (0, -1, 2) if kind == "move" else ((0, 2, 0) if kind == "swap" else (0, 2, 3))
    )
    output = _native._action_delta(B, C, labels.tolist(), blocks, first, second, destination)
    after = labels.copy()
    after[blocks[first]] = destination if second < 0 else labels[blocks[second][0]]
    if second >= 0:
        after[blocks[second]] = destination
    old_mask = np.equal.outer(labels, labels)
    new_mask = np.equal.outer(after, after)
    expected = [float((matrix * new_mask).sum() - (matrix * old_mask).sum()) for matrix in (B, C)]
    np.testing.assert_allclose(output[:2], expected, atol=1e-10)
    np.testing.assert_allclose(output[2:], expected, atol=1e-10)


@pytest.mark.parametrize("seed", range(8))
@pytest.mark.parametrize("gamma", [0.0, 1.0, 1.7])
def test_weighted_constraints_and_state(seed, gamma):
    """Audit native caches while checking original weighted and pair constraints."""
    A = graph(seed)
    weights = np.array([1, 2, 1] * 6)
    must, cannot = [(0, 1), (6, 7)], [(0, 6), (2, 8), (3, 9)]
    result = refine_partition(
        A,
        np.arange(18) % 3,
        K=3,
        R_bounds=(6, 10),
        balance_weights=weights,
        must_link=must,
        cannot_link=cannot,
        gamma=gamma,
        seed=seed,
        wz_is_C_node=True,
        restarts=3,
        local_iters=5,
        tabu_max_steps=15,
        shake_rounds=2,
        check_invariants=True,
        return_metadata=True,
    )
    assert result is not None
    assert_feasible(*result, A, weights, must, cannot)
    assert result[1]["native_search_seconds"] > 0
    run = result[1]["native_runs"][0]
    assert run["tabu_steps"] > 0
    assert run["moves"] + run["swaps"] + run["ejections"] > 0
    assert run["native_seed"] == result[1]["native_seed"]
    assert run["found"]


def test_masks_beyond_machine_word():
    """Verify cannot-link membership bits beyond component positions 63 and 127."""
    n = 150
    A = graph(4, n)
    cannot = [(0, 64), (1, 129), (63, 149), (65, 130)]
    result = refine_partition(
        A,
        np.arange(n) % 5,
        K=5,
        R_bounds=(28, 32),
        cannot_link=cannot,
        wz_is_C_node=True,
        restarts=2,
        local_iters=2,
        tabu_max_steps=8,
        shake_rounds=1,
        check_invariants=True,
        return_metadata=True,
    )
    assert result is not None
    assert_feasible(*result, A, cannot=cannot)


@pytest.mark.parametrize("n,K,R", [(8, 2, 0), (9, 3, 0), (10, 3, 1), (7, 2, 1)])
def test_python_comparison_on_constant_objective(n, K, R):
    """Compare both implementations on cases where every partition has equal B score."""
    A = np.ones((n, n))
    labels = np.arange(n) % K
    kwargs = dict(
        K=K,
        R=R,
        gamma=1.0,
        wz_is_C_node=True,
        restarts=2,
        local_iters=2,
        tabu_max_steps=5,
        shake_rounds=1,
    )
    py = python_vfd(np.equal.outer(labels, labels), A, A.sum(1), A.sum(), **kwargs)
    native = very_fortunate_descent(np.equal.outer(labels, labels), A, A.sum(1), A.sum(), **kwargs)
    assert py is not None and native is not None
    assert native[1]["r_min"] == py[1]["r_min"]
    assert native[1]["r_max"] == py[1]["r_max"]
    assert native[1]["objective_B_sum"] == pytest.approx(py[1]["objective_B_sum"])
    assert_feasible(*native, A)


def test_clustering_and_fallback_paths():
    """Exercise shared clustering preprocessing and the empty-ensemble fallback."""
    A = graph(3, 12)
    for methods in [("kmeans", "gmm", "spectral"), ("gmm_diag",), ()]:
        with threadpool_limits(limits=1):
            out = refine_partition(
                A,
                np.arange(12) % 3,
                K=3,
                R=0,
                clustering_Ks=(3,),
                clustering_seeds=(0,),
                clustering_methods=methods,
                restarts=2,
                local_iters=2,
                shake_rounds=1,
                tabu_max_steps=5,
                return_metadata=True,
            )
        assert out is not None
        assert_feasible(*out, A)


def test_splitting_and_construction_only():
    """Ensure coarse fingerprints can be split without separating must-links."""
    A = graph(1, 12)
    for rounds in [0, 2]:
        out = refine_partition(
            A,
            np.zeros(12, dtype=int),
            K=3,
            R=0,
            must_link=[(0, 1)],
            wz_is_C_node=True,
            restarts=2,
            local_iters=2,
            tabu_max_steps=5,
            shake_rounds=rounds,
            check_invariants=True,
            return_metadata=True,
        )
        assert out is not None
        assert_feasible(*out, A, must=[(0, 1)])


def test_repeatability_and_k_search():
    """Check repeatability within one build and fixed bounds during K search."""
    A = graph(5, 12)
    kwargs = dict(
        K=3,
        R_bounds=(2, 6),
        K_search_radius=1,
        restarts=2,
        local_iters=3,
        tabu_max_steps=10,
        shake_rounds=2,
        wz_is_C_node=True,
        return_metadata=True,
    )
    one = refine_partition(A, np.arange(12) % 3, **kwargs)
    two = refine_partition(A, np.arange(12) % 3, **kwargs)
    assert one is not None and two is not None
    np.testing.assert_array_equal(one[0], two[0])
    assert one[1]["K_used"] in (2, 3, 4)
    assert [run["K_used"] for run in one[1]["native_runs"]] == [2, 3, 4]
    assert_feasible(*one, A)


def test_infeasible_and_empty_cases():
    """Check contradictory constraints, excessive component loads and empty inputs."""
    A = graph(0, 8)
    labels = np.arange(8) % 2
    assert (
        refine_partition(A, labels, must_link=[(0, 1)], cannot_link=[(0, 1)], wz_is_C_node=True)
        is None
    )
    assert (
        refine_partition(
            A, labels, must_link=[(0, 1), (1, 2), (2, 3), (3, 4)], R=0, wz_is_C_node=True
        )
        is None
    )
    assert refine_partition(A, labels, K=9, R_bounds=(1, 8), wz_is_C_node=True) is None
    assert refine_partition(A, labels, R_bounds=(0, 0), wz_is_C_node=True) is None
    assert refine_partition(A, labels, cannot_link=[(0, 0)], wz_is_C_node=True) is None
    empty = refine_partition(np.empty((0, 0)), np.empty(0, dtype=int))
    assert empty.shape == (0, 0)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"balance_weights": [0] * 8},
        {"balance_weights": [1.5] * 8},
        {"R_bounds": (5, 2)},
        {"gamma": -1},
        {"restarts": 0},
        {"local_iters": -1},
        {"local_iters": 1.5},
        {"m": 0},
        {"must_link": [(0, 9)]},
        {"clustering_methods": ("unknown",)},
        {"w_coassoc": float("nan")},
    ],
)
def test_invalid_inputs(kwargs):
    """Require Python exceptions for invalid weights, bounds and search options."""
    with pytest.raises(ValueError):
        refine_partition(graph(0, 8), np.arange(8) % 2, wz_is_C_node=True, **kwargs)


def test_native_boundary_validation():
    """Require safe exceptions for malformed low-level indices and blocks."""
    B = np.asfortranarray(np.eye(4))
    with pytest.raises(ValueError):
        _native._action_delta(B, B, [0, 0, 1, 1], [[0, 5], [1, 2, 3]], 0, -1, 1)
    with pytest.raises(ValueError):
        _native._action_delta(B, B, [0, 0, 1, 1], [[0, 1], [2, 3]], 0, 0, 1)
    bad = B.copy(order="F")
    bad[0, 1] = 1
    with pytest.raises(ValueError):
        _native._action_delta(bad, B, [0, 0, 1, 1], [[0, 1], [2, 3]], 0, -1, 1)


def test_refinement_hook_contract():
    """Confirm the original refinement signature and matrix-only return convention."""
    expected = inspect.signature(python_refine).parameters
    actual = inspect.signature(refine_partition).parameters
    for name, parameter in expected.items():
        assert name in actual
        assert parameter.default == actual[name].default
    A = graph(0, 8)
    Z = refine_partition(
        A,
        np.arange(8) % 2,
        K=2,
        R=0,
        wz_is_C_node=True,
        restarts=1,
        local_iters=2,
        tabu_max_steps=5,
        shake_rounds=1,
    )
    assert isinstance(Z, np.ndarray) and Z.shape == A.shape


def test_sparse_constraint_representation(monkeypatch):
    """Check the shared set-based cannot-link path used for many sparse components."""
    original = adapter._build_components

    def force_sets(N, must_link, cannot_link, node_weights=None):
        """Build small test components using the large sparse-constraint representation."""
        return original(
            N, must_link, cannot_link, node_weights, bitmask_C_max=0, dense_deg_threshold=1e9
        )

    monkeypatch.setattr(adapter, "_build_components", force_sets)
    A = graph(2, 18)
    cannot = [(0, 6), (1, 13), (7, 17)]
    result = refine_partition(
        A,
        np.arange(18) % 3,
        K=3,
        R=0,
        cannot_link=cannot,
        wz_is_C_node=True,
        restarts=2,
        local_iters=3,
        tabu_max_steps=10,
        shake_rounds=1,
        check_invariants=True,
        return_metadata=True,
    )
    assert result is not None
    assert_feasible(*result, A, cannot=cannot)


def test_sparse_inputs_and_memory_guard():
    """Verify guarded dense conversion of sparse adjacency and partition columns."""
    from scipy import sparse

    A = graph(7, 12)
    partition = np.equal.outer(np.arange(12) % 3, np.arange(12) % 3)
    kwargs = dict(
        K=3, R=0, wz_is_C_node=True, restarts=2, local_iters=2, tabu_max_steps=5, shake_rounds=1
    )
    dense = refine_partition(A, partition, **kwargs)
    sparse_out = refine_partition(sparse.csr_matrix(A), sparse.csr_matrix(partition), **kwargs)
    assert dense is not None and sparse_out is not None
    np.testing.assert_array_equal(sparse_out, dense)
    with pytest.raises(MemoryError):
        refine_partition(sparse.csr_matrix(A), partition, max_dense_working_bytes=1, **kwargs)


def test_label_identifiers_preserve_dtype():
    """Preserve string labels and distinct integer labels above float64 precision."""
    A = graph(0, 8)
    labels = np.arange(8) % 2
    kwargs = dict(K=2, R=0, wz_is_C_node=True, restarts=1, shake_rounds=0)
    expected = refine_partition(A, labels, **kwargs)
    for encoded in (labels + np.int64(2**60), np.where(labels == 0, "alpha", "beta")):
        output = refine_partition(A, encoded, **kwargs)
        np.testing.assert_array_equal(output, expected)


@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize("rounds", [0, 1, 3])
def test_zero_lower_bound_preserves_nonempty_groups(seed, rounds):
    """Keep all requested communities occupied through tabu search and shaking."""
    A = np.ones((4, 4)) - np.eye(4)
    result = refine_partition(
        A,
        np.arange(4),
        K=2,
        R_bounds=(0, 4),
        gamma=0,
        w_coassoc=0,
        seed=seed,
        restarts=1,
        local_iters=2,
        tabu_max_steps=5,
        shake_rounds=rounds,
        wz_is_C_node=True,
        check_invariants=True,
        return_metadata=True,
    )
    assert result is not None
    assert_feasible(*result, A)
    assert result[1]["objective_B_sum"] == pytest.approx(6.0)


@pytest.mark.parametrize("large", [2**53 + 1, 2**62 + 1, 2**63 - 6])
@pytest.mark.parametrize("explicit_bounds", [False, True])
def test_exact_large_integer_loads(large, explicit_bounds):
    """Preserve large int64 loads through contraction, bounds, and target creation."""
    A = np.ones((3, 3))
    weights = np.array([large, 2, 3], dtype=np.int64)
    total = sum(map(int, weights))
    result = refine_partition(
        A,
        np.zeros(3),
        K=1,
        R=0,
        R_bounds=(total, total) if explicit_bounds else None,
        balance_weights=weights,
        must_link=[(0, 1)],
        wz_is_C_node=True,
        restarts=1,
        shake_rounds=0,
        check_invariants=True,
        return_metadata=True,
    )
    assert result is not None
    assert result[1]["total_balance_weight"] == total
    assert result[1]["r_min"] == result[1]["r_max"] == total
    assert_feasible(*result, A, weights, must=[(0, 1)])


def test_numpy_constraint_and_method_sequences():
    """Accept NumPy pair arrays and clustering-method arrays without truth coercion."""
    A = graph(0, 8)
    must = np.array([[0, 1], [2, 3]])
    cannot = np.array([[0, 2]])
    result = refine_partition(
        A,
        np.arange(8) % 2,
        K=2,
        R=0,
        must_link=must,
        cannot_link=cannot,
        clustering_methods=np.array(["kmeans"]),
        clustering_Ks=(2,),
        clustering_seeds=(0,),
        restarts=1,
        shake_rounds=0,
        return_metadata=True,
    )
    assert result is not None
    assert_feasible(*result, A, must=must, cannot=cannot)


@pytest.mark.parametrize("name", ["must_link", "cannot_link"])
@pytest.mark.parametrize("pair", [(0.5, 1.5), (-0.5, 1), (0, float("inf")), (0, "1"), (0,)])
def test_invalid_pair_indices(name, pair):
    """Reject fractional, nonnumeric, nonfinite, and malformed node pairs."""
    with pytest.raises(ValueError):
        refine_partition(graph(0, 8), np.arange(8), wz_is_C_node=True, **{name: [pair]})


@pytest.mark.parametrize(
    "weights",
    [
        [2**63, 1, 1],
        [2**63 - 1, 1, 1],
        [2**100, 1, 1],
        [1, 2.5, 1],
    ],
)
def test_invalid_large_or_fractional_weights(weights):
    """Reject loads that exceed int64 capacity or have fractional units."""
    with pytest.raises(ValueError):
        refine_partition(np.ones((3, 3)), np.zeros(3), K=1, balance_weights=weights)


def test_large_integer_bounds_detect_infeasibility_exactly():
    """Avoid treating adjacent large integers as equal while checking load bounds."""
    weight = 2**53 + 1
    assert (
        refine_partition(
            np.ones((1, 1)),
            np.zeros(1),
            K=1,
            R_bounds=(weight + 1, weight + 1),
            balance_weights=[weight],
            wz_is_C_node=True,
        )
        is None
    )


def test_native_search_at_int64_total_limit():
    """Audit moves and swaps with an exact total load of INT64_MAX."""
    A = np.ones((4, 4)) - np.eye(4)
    weights = np.array([2**62 - 1, 2**62 - 2, 1, 1], dtype=np.int64)
    result = refine_partition(
        A,
        np.arange(4),
        K=2,
        R_bounds=(0, 2**62 + 1),
        balance_weights=weights,
        wz_is_C_node=True,
        restarts=2,
        local_iters=2,
        tabu_max_steps=5,
        shake_rounds=2,
        check_invariants=True,
        return_metadata=True,
    )
    assert result is not None
    assert result[1]["total_balance_weight"] == 2**63 - 1
    assert_feasible(*result, A, weights)
