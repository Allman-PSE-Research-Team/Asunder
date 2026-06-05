import functools
from types import SimpleNamespace

import networkx as nx
import numpy as np
import pytest
from scipy.sparse import csr_matrix

import asunder.load_balancing.column_generation.LB as lb_module
from asunder.base.algorithms.community import run_modularity
from asunder.base.algorithms.core_periphery import (
    EnhancedGeneticBE,
    FullContinuousGeneticBE,
    detect_continuous_KL,
    find_core_advanced,
)
from asunder.base.column_generation.decomposition import CSD_decomposition
from asunder.base.column_generation.master import compute_f_star
from asunder.base.column_generation.subproblem import solve_subproblem
from asunder.base.utils.graph import expand_z_matrix
from asunder.load_balancing.utils.partition_generation import (
    assign_from_order_with_links_range,
    make_partitions_random,
)
from asunder.types import DecompositionResult


def _small_graph():
    A = np.array([[0.0, 1.0], [1.0, 0.0]])
    return A, A.sum(axis=1), float(A.sum()), np.eye(2, dtype=int)


def _master_ok(A, a, m, Z_star, f_stars, extract_dual=False, **kwargs):
    lambdas = [1.0] + [0.0] * (len(Z_star) - 1)
    if extract_dual:
        return lambdas, {"mu_dual": 0.0}, float(f_stars[0] if f_stars else 0.0)
    return lambdas, float(f_stars[0] if f_stars else 0.0)


def _subproblem_eye(A, a, m, duals, **kwargs):
    return 0.0, np.eye(A.shape[0], dtype=int)


def test_load_balancer_normalizes_external_node_labels(monkeypatch):
    """Regression coverage for public graph labels in LoadBalancer constraints."""
    captured = {}

    def fake_decomposition(A, a, m, config, master_fn, subproblem_fn, **kwargs):
        captured.update(vars(config))
        return DecompositionResult([], np.eye(A.shape[0], dtype=int), None)

    monkeypatch.setattr(lb_module, "run_csd_decomposition", fake_decomposition)

    G = nx.path_graph(["z", "a", "m", "b"])
    result = lb_module.LoadBalancer(
        G,
        K=2,
        R=1,
        must_link=[("z", "a")],
        cannot_link=[("m", "b")],
        disable_tqdm=True,
    )

    assert captured["must_link"] == [(0, 1)]
    assert captured["cannot_link"] == [(2, 3)]
    assert set(result.metadata["community_map_labels"]) == {"z", "a", "m", "b"}


def test_load_balancer_normalizes_nonconsecutive_integer_labels(monkeypatch):
    """Regression coverage for graph labels that are integers but not row indices."""
    captured = {}

    def fake_decomposition(A, a, m, config, master_fn, subproblem_fn, **kwargs):
        captured.update(vars(config))
        return DecompositionResult([], np.eye(A.shape[0], dtype=int), None)

    monkeypatch.setattr(lb_module, "run_csd_decomposition", fake_decomposition)

    G = nx.path_graph([10, 20, 30, 40])
    lb_module.LoadBalancer(G, K=2, R=1, must_link=[(10, 20)], disable_tqdm=True)

    assert captured["must_link"] == [(0, 1)]


def test_load_balancer_bounds_and_generator_validation(monkeypatch):
    """Regression coverage for R_bounds normalization and generator validation."""
    captured = {}

    def fake_decomposition(A, a, m, config, master_fn, subproblem_fn, **kwargs):
        captured.update(vars(config))
        return DecompositionResult([], np.eye(A.shape[0], dtype=int), None)

    monkeypatch.setattr(lb_module, "run_csd_decomposition", fake_decomposition)
    G = nx.path_graph(4)

    lb_module.LoadBalancer(G, K=2, R=0, R_bounds=(None, 3), disable_tqdm=True)
    assert captured["additional_constraints"]["R_bounds"] == (1, 3)

    with pytest.raises(ValueError, match="Cardinality bounds"):
        lb_module.LoadBalancer(G, K=2, R=0, R_bounds=(4, 2), disable_tqdm=True)

    with pytest.raises(ValueError, match="ifc_generator"):
        lb_module.LoadBalancer(G, K=2, R=0, ifc_generator="typo", disable_tqdm=True)


def test_final_master_infeasibility_records_no_partition():
    """Regression coverage for np.argmax(None) silently selecting stale columns."""
    A, a, m, Z = _small_graph()

    def master_final_infeasible(A, a, m, Z_star, f_stars, extract_dual=False, **kwargs):
        if extract_dual:
            return [1.0], {"mu_dual": 0.0}, 0.0
        return None, None

    out = CSD_decomposition(
        A,
        a,
        m,
        master_final_infeasible,
        _subproblem_eye,
        columns=[Z],
        f_stars=[0.0],
        extract_dual=True,
        final_master_solve=True,
        max_iterations=0,
        disable_tqdm=True,
        verbose=-1,
    )

    assert out[-1]["lambda_sol"] is None
    assert out[-1]["master_obj_val"] is None
    assert out[-1]["z_sol"] is None


def test_contract_final_master_infeasibility_expands_none():
    """Regression coverage for contracted final infeasibility with z_sol=None."""
    A, a, m, Z = _small_graph()

    def master_final_infeasible(A, a, m, Z_star, f_stars, extract_dual=False, **kwargs):
        if extract_dual:
            return [1.0], {"mu_dual": 0.0}, 0.0
        return None, None

    out = CSD_decomposition(
        A,
        a,
        m,
        master_final_infeasible,
        _subproblem_eye,
        columns=[Z],
        f_stars=[0.0],
        must_link=[(0, 1)],
        additional_constraints=dict(),
        contract_graph=True,
        extract_dual=True,
        final_master_solve=True,
        max_iterations=0,
        disable_tqdm=True,
        verbose=-1,
    )

    assert out[-1]["z_sol"] is None
    assert expand_z_matrix(None, np.array([0, 0])) is None


def test_decomposition_accepts_wrapped_heuristic_callables():
    """Regression coverage for callable objects and partials lacking __name__."""
    A, a, m, Z = _small_graph()

    class CallableSubproblem:
        def __call__(self, A, a, m, duals, **kwargs):
            return 0.0, np.eye(A.shape[0], dtype=int)

    for hook in (functools.partial(_subproblem_eye), CallableSubproblem()):
        out = CSD_decomposition(
            A,
            a,
            m,
            _master_ok,
            hook,
            columns=[Z],
            f_stars=[0.0],
            extract_dual=True,
            final_master_solve=False,
            max_iterations=0,
            disable_tqdm=True,
            verbose=-1,
        )
        assert out[-1]["z_sol"].shape == A.shape


class _FakeSolver:
    """Tiny Pyomo solver stub that lets exact pricing build and evaluate."""

    def solve(self, model, tee=False):
        for value in model.z.values():
            value.set_value(1)
        return SimpleNamespace(solver=SimpleNamespace(termination_condition=None))


def test_exact_pricing_subproblem_builds_without_diagonal_z_variables():
    """Regression coverage for z[i, i] lookups in the exact pricing model."""
    A, a, m, _ = _small_graph()

    obj, z = solve_subproblem(A, a, m, {}, solver=_FakeSolver(), verbose=-1)

    assert np.isfinite(obj)
    assert np.all(np.diag(z) == 1)
    assert np.all(z == z.T)


def test_zero_edge_graphs_raise_clear_value_error():
    """Regression coverage for NaN-producing zero-edge objective evaluation."""
    A = np.zeros((3, 3), dtype=float)

    with pytest.raises(ValueError, match="positive edge sum"):
        compute_f_star(A, A.sum(axis=1), 0.0, np.eye(3, dtype=int))


def test_run_modularity_default_call_succeeds():
    """Regression coverage for run_modularity(refine=True, refine_params=None)."""
    A, _, _, _ = _small_graph()

    z, metric = run_modularity(A)

    assert z.shape == A.shape
    assert np.isfinite(metric)


def test_core_periphery_seed_none_uses_local_rng():
    """Regression coverage for seed=None after moving away from global np.random."""
    A = np.array(
        [
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
        ]
    )

    EnhancedGeneticBE(A, pop_size=4, generations=1, tournament_size=2, seed=None).run()
    FullContinuousGeneticBE(A, pop_size=4, generations=1, tournament_size=2, seed=None).run()
    detect_continuous_KL(csr_matrix(A), must_links=[], nonlinear_nodes=[], max_iter=1, seed=None)


def test_find_core_advanced_validates_labels():
    """Advanced core orientation validates and returns one label per node."""
    A = np.array([[0, 1], [1, 0]], dtype=float)
    labels, score = find_core_advanced(A, [0, 1])
    assert labels.shape == (2,)
    assert np.isfinite(score)

    with pytest.raises(ValueError, match="one entry"):
        find_core_advanced(A, [0])


def test_load_balanced_sparse_component_fallbacks_above_bitmask_limit():
    """Regression coverage for set-backed components when C > 4096."""
    N = 4097
    r_bounds = (N, N)

    random_parts = make_partitions_random(N=N, K=1, R=0, R_bounds=r_bounds, seed=1)
    assert random_parts
    assert random_parts[0].shape == (N, N)

    ordered = assign_from_order_with_links_range(
        list(range(N)),
        N=N,
        K=1,
        R=0,
        R_bounds=r_bounds,
        seed=1,
    )
    assert ordered is not None
    g, meta = ordered
    assert g.shape == (N,)
    assert meta["K_used"] == 1
