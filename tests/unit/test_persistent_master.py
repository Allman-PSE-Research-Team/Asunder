from types import SimpleNamespace

import numpy as np
import pytest
from scipy import sparse

from asunder.base.column_generation import decomposition as decomposition_module
from asunder.base.column_generation import persistent_master as persistent_module
from asunder.base.column_generation.decomposition import CSD_decomposition
from asunder.base.column_generation.master import solve_master_problem as base_master
from asunder.base.column_generation.pricing import build_dual_weight_matrix
from asunder.load_balancing.column_generation.master import (
    solve_master_problem as load_balancing_master,
)
from asunder.solvers import create_solver


def _partition(labels):
    labels = np.asarray(labels)
    return np.equal.outer(labels, labels)


def _path_adjacency(n_nodes=4):
    adjacency = np.zeros((n_nodes, n_nodes), dtype=float)
    adjacency[np.arange(n_nodes - 1), np.arange(1, n_nodes)] = 1.0
    return adjacency + adjacency.T


def _assert_dual_solution_valid(adjacency, columns, scores, result):
    lambda_sol, duals, objective = result
    dual_weight, constant = build_dual_weight_matrix(adjacency, duals)
    reduced_costs = []
    for column, score in zip(columns, scores):
        if sparse.issparse(dual_weight):
            dual_contribution = float(dual_weight.multiply(column).sum())
        elif sparse.issparse(column):
            dual_contribution = float(column.multiply(dual_weight).sum())
        else:
            dual_contribution = float(np.sum(dual_weight * column))
        reduced_costs.append(float(score) - dual_contribution - constant)
    assert max(reduced_costs) <= 1e-7
    for weight, reduced_cost in zip(lambda_sol, reduced_costs):
        if weight > 1e-7:
            assert reduced_cost == pytest.approx(0.0, abs=1e-7)
    assert objective == pytest.approx(float(np.dot(lambda_sol, scores)))


def _assert_master_results_compatible(adjacency, columns, scores, actual, expected):
    actual_lambda, actual_duals, actual_objective = actual
    expected_lambda, expected_duals, expected_objective = expected
    np.testing.assert_allclose(actual_lambda, expected_lambda, atol=1e-8)
    assert actual_objective == pytest.approx(expected_objective)
    assert actual_duals.keys() == expected_duals.keys()
    for name in actual_duals:
        actual_value = actual_duals[name]
        expected_value = expected_duals[name]
        if sparse.issparse(actual_value) or sparse.issparse(expected_value):
            assert sparse.issparse(actual_value)
            assert sparse.issparse(expected_value)
            actual_value = sparse.csr_matrix(actual_value).toarray()
            expected_value = sparse.csr_matrix(expected_value).toarray()
        actual_value = np.asarray(actual_value)
        expected_value = np.asarray(expected_value)
        assert actual_value.shape == expected_value.shape
        assert np.all(np.isfinite(actual_value))
        assert np.all(np.isfinite(expected_value))
    _assert_dual_solution_valid(adjacency, columns, scores, actual)
    _assert_dual_solution_valid(adjacency, columns, scores, expected)


@pytest.fixture(scope="module")
def gurobi_direct_solver():
    try:
        from pyomo.environ import ConcreteModel, Objective, Var, maximize

        solver = create_solver("gurobi_direct", manage_env=True)
        if not solver.available(False):
            pytest.skip("The Gurobi direct solver is unavailable.")
        probe = ConcreteModel()
        probe.x = Var(bounds=(0, 1))
        probe.objective = Objective(expr=probe.x, sense=maximize)
        solver.solve(probe, tee=False)
    except Exception as exc:
        if "solver" in locals() and hasattr(solver, "close"):
            solver.close()
        pytest.skip(f"A usable Gurobi license is unavailable: {exc}")
    try:
        yield solver
    finally:
        solver.close()


@pytest.mark.solver
def test_base_persistent_master_matches_rebuild_before_and_after_append(
    gurobi_direct_solver,
):
    adjacency = _path_adjacency(5)
    strengths = adjacency.sum(axis=1)
    volume = float(strengths.sum())
    columns = [
        _partition([0, 1, 1, 2, 2]),
        _partition([0, 0, 0, 0, 0]),
        _partition([0, 1, 2, 3, 4]),
    ]
    scores = [1.0, 4.0, 5.0]
    cannot_link = [(0, 2)]
    must_link = [(3, 4)]
    worthy_edges = [(0, 1), (2, 3), (3, 4)]
    master_kwargs = {
        "worthy_edges": worthy_edges,
        "solver": gurobi_direct_solver,
    }

    session = persistent_module.create_persistent_master_session(
        base_master,
        adjacency,
        columns,
        scores,
        cannot_link=cannot_link,
        must_link=must_link,
        additional_constraints=master_kwargs,
        verbose=-1,
    )
    try:
        rebuilt = base_master(
            adjacency,
            strengths,
            volume,
            columns,
            scores,
            cannot_link=cannot_link,
            must_link=must_link,
            extract_dual=True,
            verbose=-1,
            **master_kwargs,
        )
        _assert_master_results_compatible(
            adjacency,
            columns,
            scores,
            session.solve(extract_dual=True),
            rebuilt,
        )

        columns.append(sparse.csr_matrix(_partition([0, 1, 1, 1, 1])))
        scores.append(3.0)
        session.sync(columns, scores)
        assert session.column_count == 4
        rebuilt = base_master(
            adjacency,
            strengths,
            volume,
            columns,
            scores,
            cannot_link=cannot_link,
            must_link=must_link,
            extract_dual=True,
            verbose=-1,
            **master_kwargs,
        )
        _assert_master_results_compatible(
            adjacency,
            columns,
            scores,
            session.solve(extract_dual=True),
            rebuilt,
        )

        persistent_integer = session.solve(extract_dual=False)
        rebuilt_integer = base_master(
            adjacency,
            strengths,
            volume,
            columns,
            scores,
            cannot_link=cannot_link,
            must_link=must_link,
            extract_dual=False,
            verbose=-1,
            **master_kwargs,
        )
        np.testing.assert_allclose(
            persistent_integer[0], rebuilt_integer[0], atol=1e-8
        )
        assert persistent_integer[1] == pytest.approx(rebuilt_integer[1])
    finally:
        session.close()


@pytest.mark.solver
def test_load_balancing_persistent_master_matches_rebuild_after_append(
    gurobi_direct_solver,
):
    adjacency = _path_adjacency(5)
    strengths = adjacency.sum(axis=1)
    volume = float(strengths.sum())
    columns = [
        _partition([0, 0, 1, 1, 1]),
        _partition([0, 1, 0, 1, 0]),
        _partition([0, 0, 0, 0, 0]),
    ]
    scores = [1.0, 5.0, 6.0]
    cannot_link = [(0, 2)]
    must_link = [(0, 1)]
    master_kwargs = {
        "LB": True,
        "R": 1,
        "K": 2,
        "R_bounds": (2, 3),
        "balance_weights": np.ones(5),
        "solver": gurobi_direct_solver,
    }

    session = persistent_module.create_persistent_master_session(
        load_balancing_master,
        adjacency,
        columns,
        scores,
        cannot_link=cannot_link,
        must_link=must_link,
        additional_constraints=master_kwargs,
        verbose=-1,
    )
    try:
        rebuilt = load_balancing_master(
            adjacency,
            strengths,
            volume,
            columns,
            scores,
            cannot_link=cannot_link,
            must_link=must_link,
            extract_dual=True,
            verbose=-1,
            **master_kwargs,
        )
        _assert_master_results_compatible(
            adjacency,
            columns,
            scores,
            session.solve(extract_dual=True),
            rebuilt,
        )

        columns.append(_partition([0, 0, 1, 0, 1]))
        scores.append(3.0)
        session.sync(columns, scores)
        rebuilt = load_balancing_master(
            adjacency,
            strengths,
            volume,
            columns,
            scores,
            cannot_link=cannot_link,
            must_link=must_link,
            extract_dual=True,
            verbose=-1,
            **master_kwargs,
        )
        _assert_master_results_compatible(
            adjacency,
            columns,
            scores,
            session.solve(extract_dual=True),
            rebuilt,
        )
    finally:
        session.close()


class _ModelUpdatingSolver:
    def __init__(self):
        self.model = None
        self.set_instance_calls = 0
        self.added_columns = []
        self.updated_variables = []
        self.solve_calls = 0
        self.close_calls = 0

    def set_instance(self, model, **_):
        self.model = model
        self.set_instance_calls += 1

    def add_column(self, model, variable, objective_coefficient, constraints, coefficients):
        self.added_columns.append(variable.index())
        model.OBJ.expr += objective_coefficient * variable
        for constraint, coefficient in zip(constraints, coefficients):
            lower, body, upper = constraint.to_bounded_expression()
            constraint.set_value((lower, body + coefficient * variable, upper))

    def update_var(self, variable):
        self.updated_variables.append(variable)

    def solve(self, **_):
        self.solve_calls += 1
        for variable in self.model.lmbd.values():
            variable.set_value(0.0)
        self.model.lmbd[max(self.model.C)].set_value(1.0)
        return SimpleNamespace(
            solver=SimpleNamespace(termination_condition="optimal")
        )

    def close(self):
        self.close_calls += 1


def test_persistent_session_appends_only_suffix_then_switches_to_binary(monkeypatch):
    solver = _ModelUpdatingSolver()
    monkeypatch.setattr(
        persistent_module,
        "_configured_gurobi_solver",
        lambda source_solver=None: solver,
    )
    adjacency = _path_adjacency()
    columns = [_partition([0, 1, 2, 3]), _partition([0, 0, 1, 1])]
    scores = [1.0, 2.0]
    all_edges = [(int(i), int(i + 1)) for i in range(3)]
    session = persistent_module.PersistentMasterSession(
        adjacency,
        columns,
        scores,
        kind="base",
        worthy_edges=all_edges,
    )

    session.sync(columns, scores)
    assert solver.set_instance_calls == 1
    assert solver.added_columns == []

    columns.extend([_partition([0, 1, 0, 1]), _partition([0, 1, 1, 0])])
    scores.extend([3.0, 4.0])
    session.sync(columns, scores)
    assert solver.set_instance_calls == 1
    assert solver.added_columns == [2, 3]
    assert session.column_count == 4

    _, duals, _ = session.solve(extract_dual=True)
    assert "pi_dual" in duals
    assert sparse.isspmatrix_csr(duals["pi_dual"])
    assert duals["pi_dual"].nnz == 0
    assert solver.updated_variables == []

    lambda_sol, objective = session.solve(extract_dual=False)
    assert lambda_sol == [0.0, 0.0, 0.0, 1.0]
    assert objective == pytest.approx(4.0)
    assert len(solver.updated_variables) == 4
    assert all(variable.is_binary() for variable in session.model.lmbd.values())
    with pytest.raises(RuntimeError, match="after the integer solve"):
        session.sync(columns + [_partition([0, 0, 0, 0])], scores + [5.0])

    session.close()
    session.close()
    assert solver.close_calls == 1


class _SpySession:
    def __init__(self, columns, *, close_error=None, infeasible=False):
        self.columns = list(columns)
        self.sync_sizes = []
        self.appended_batches = []
        self.solve_modes = []
        self.close_calls = 0
        self.close_error = close_error
        self.infeasible = infeasible

    def sync(self, columns, f_stars):
        assert len(columns) == len(f_stars)
        self.sync_sizes.append(len(columns))
        self.appended_batches.append(list(columns[len(self.columns) :]))
        self.columns = list(columns)

    def solve(self, *, extract_dual):
        self.solve_modes.append(extract_dual)
        if self.infeasible:
            return (None, None, None) if extract_dual else (None, None)
        if extract_dual:
            return (
                [1.0] + [0.0] * (len(self.columns) - 1),
                {"mu_dual": 0.0},
                0.0,
            )
        return [0.0] * (len(self.columns) - 1) + [1.0], 1.0

    def close(self):
        self.close_calls += 1
        if self.close_error is not None:
            raise self.close_error


def test_decomposition_syncs_all_new_columns_and_closes_one_session(monkeypatch):
    adjacency = _path_adjacency()
    strengths = adjacency.sum(axis=1)
    initial = _partition([0, 0, 0, 0])
    priced = _partition([0, 1, 2, 3])
    in_loop_refined = _partition([0, 0, 1, 1])
    stopped_pricing = _partition([0, 1, 0, 1])
    post_loop_refined = _partition([0, 1, 1, 0])
    sessions = []

    def create_session(mp_function, A, columns, f_stars, **kwargs):
        assert mp_function is base_master
        assert A.shape == adjacency.shape
        assert len(columns) == len(f_stars) == 1
        session = _SpySession(columns)
        sessions.append(session)
        return session

    monkeypatch.setattr(
        persistent_module,
        "create_persistent_master_session",
        create_session,
    )
    pricing_calls = 0

    def subproblem(A, a, m, duals, **kwargs):
        nonlocal pricing_calls
        pricing_calls += 1
        if pricing_calls == 1:
            return 1.0, priced
        return 0.0, stopped_pricing

    refinement_calls = 0

    def refine(A, partition, **kwargs):
        nonlocal refinement_calls
        refinement_calls += 1
        return in_loop_refined if refinement_calls == 1 else post_loop_refined

    raw = CSD_decomposition(
        adjacency,
        strengths,
        float(strengths.sum()),
        base_master,
        subproblem,
        columns=[initial],
        f_stars=[0.0],
        refine_params={"refine_func": refine},
        use_refined_column=True,
        refine_post_loop=True,
        final_master_solve=True,
        persistent_master=True,
        additional_constraints={
            "solver": SimpleNamespace(type="gurobi_direct", options={})
        },
        check_flat_pricing=False,
        max_iterations=3,
        disable_tqdm=True,
        verbose=-1,
    )

    assert len(sessions) == 1
    session = sessions[0]
    assert session.sync_sizes == [1, 3, 4]
    assert [len(batch) for batch in session.appended_batches] == [0, 2, 1]
    assert np.array_equal(session.appended_batches[1][0], priced)
    assert np.array_equal(session.appended_batches[1][1], in_loop_refined)
    assert np.array_equal(session.appended_batches[2][0], post_loop_refined)
    assert session.solve_modes == [True, True, False]
    assert session.close_calls == 1
    assert raw[-1]["partition_source"] == "integer_master"
    assert np.array_equal(raw[-1]["z_sol"], post_loop_refined)
    assert [len(record["columns"]) for record in raw] == [1, 3, 4, 4]
    assert [len(record["f_stars"]) for record in raw] == [1, 3, 4, 4]
    assert raw[0]["columns"]._items is raw[-1]["columns"]._items
    assert raw[0]["f_stars"]._items is raw[-1]["f_stars"]._items
    assert raw[0]["columns"][-1] is raw[0]["columns"][0]
    with pytest.raises(IndexError):
        raw[0]["columns"][1]


def test_decomposition_preserves_work_error_and_resets_owner_when_close_fails(
    monkeypatch,
):
    adjacency = _path_adjacency()
    strengths = adjacency.sum(axis=1)
    session = _SpySession(
        [_partition([0, 0, 0, 0])],
        close_error=RuntimeError("close failed"),
    )
    monkeypatch.setattr(
        persistent_module,
        "create_persistent_master_session",
        lambda *args, **kwargs: session,
    )

    def failing_subproblem(A, a, m, duals, **kwargs):
        raise ValueError("pricing failed")

    with pytest.raises(ValueError, match="pricing failed"):
        CSD_decomposition(
            adjacency,
            strengths,
            float(strengths.sum()),
            base_master,
            failing_subproblem,
            columns=[_partition([0, 0, 0, 0])],
            f_stars=[0.0],
            refine_post_loop=False,
            final_master_solve=False,
            persistent_master=True,
            additional_constraints={
                "solver": SimpleNamespace(type="gurobi_direct", options={})
            },
            disable_tqdm=True,
            verbose=-1,
        )

    assert session.close_calls == 1
    assert decomposition_module._PERSISTENT_SESSION_HOLDER.get() is None


def test_persistent_master_rejects_custom_master_even_on_trivial_contraction():
    adjacency = np.array([[0.0, 1.0], [1.0, 0.0]])

    def custom_master(*args, **kwargs):
        raise AssertionError("custom master must not be called")

    with pytest.raises(ValueError, match="built-in base and load-balancing"):
        CSD_decomposition(
            adjacency,
            adjacency.sum(axis=1),
            float(adjacency.sum()),
            custom_master,
            lambda *args, **kwargs: (0.0, np.eye(2, dtype=bool)),
            columns=[np.ones((2, 2), dtype=bool)],
            f_stars=[0.0],
            must_link=[(0, 1)],
            contract_graph=True,
            refine_post_loop=False,
            final_master_solve=False,
            persistent_master=True,
            disable_tqdm=True,
            verbose=-1,
        )


def test_persistent_master_rejects_unsupported_solver():
    adjacency = _path_adjacency()
    with pytest.raises(ValueError, match="gurobi_direct or gurobi_persistent"):
        persistent_module.create_persistent_master_session(
            base_master,
            adjacency,
            [_partition([0, 0, 1, 1])],
            [0.0],
            cannot_link=[],
            must_link=[],
            additional_constraints={
                "solver": SimpleNamespace(type="appsi_highs", options={})
            },
            verbose=-1,
        )
