"""Run-local persistent restricted-master implementations."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

import numpy as np
from scipy import sparse

from asunder.base.utils.matrix import matrix_scalar, structural_edge_pairs
from asunder.load_balancing.utils.balance import resolve_balance_bounds
from asunder.solvers import create_solver, get_default_solver
from asunder.types import MatrixLike

try:
    from pyomo.environ import (
        Binary,
        ConcreteModel,
        Constraint,
        NonNegativeReals,
        Objective,
        RangeSet,
        Set,
        Suffix,
        TerminationCondition,
        Var,
        maximize,
        value,
    )
except Exception:  # pragma: no cover - optional dependency
    ConcreteModel = None


MasterKind = Literal["base", "load_balancing"]


def validate_persistent_solver(source_solver: Any = None) -> Any:
    """Return a supported source solver or raise before model construction."""
    source_solver = get_default_solver() if source_solver is None else source_solver
    solver_type = str(
        getattr(source_solver, "type", getattr(source_solver, "name", ""))
    ).lower()
    if solver_type not in {"gurobi_direct", "gurobi_persistent"}:
        raise ValueError(
            "persistent_master=True currently requires a configured "
            "gurobi_direct or gurobi_persistent solver."
        )
    return source_solver


def _configured_gurobi_solver(source_solver: Any = None):
    """Create an owned persistent solver from a configured Gurobi solver."""
    source_solver = validate_persistent_solver(source_solver)
    options = dict(getattr(source_solver, "options", {}) or {})
    return create_solver(
        "gurobi_persistent",
        manage_env=True,
        options=options,
    )


class PersistentMasterSession:
    """Incrementally maintain one base or load-balancing master model."""

    def __init__(
        self,
        A: MatrixLike,
        columns: Sequence[MatrixLike],
        f_stars: Sequence[float],
        *,
        kind: MasterKind,
        cannot_link: Sequence[tuple[int, int]] = (),
        must_link: Sequence[tuple[int, int]] = (),
        worthy_edges: Sequence[tuple[int, int]] | None = None,
        LB: bool = True,
        R: int = 1,
        K: int | None = 2,
        R_bounds: tuple[float, float] | None = None,
        balance_weights: Sequence[float] | None = None,
        solver: Any = None,
        verbose: int | bool = False,
    ) -> None:
        if ConcreteModel is None:
            raise ImportError(
                "pyomo is required for persistent master optimization."
            )
        if not columns:
            raise ValueError("At least one master column is required.")
        if len(columns) != len(f_stars):
            raise ValueError("columns and f_stars must have the same length.")
        if kind not in {"base", "load_balancing"}:
            raise ValueError("Unknown persistent master kind.")

        self.kind = kind
        self.A = A
        self.n_nodes = int(A.shape[0])
        self.columns = list(columns)
        self.f_stars = [float(score) for score in f_stars]
        self.cannot_link = tuple(tuple(sorted(pair)) for pair in cannot_link)
        self.must_link = tuple(tuple(sorted(pair)) for pair in must_link)
        self.verbose = verbose
        self._integer_mode = False
        self._closed = False

        self.worthy_edge_set: set[tuple[int, int]] | None = None
        self.worthy_edges_enabled = kind == "base" and worthy_edges is not None
        self.unworthy_edges: tuple[tuple[int, int], ...] = ()
        if self.worthy_edges_enabled:
            self.worthy_edge_set = {
                tuple(sorted((int(i), int(j)))) for i, j in worthy_edges
            }
            self.unworthy_edges = tuple(
                pair
                for pair in structural_edge_pairs(A)
                if tuple(sorted(pair)) not in self.worthy_edge_set
            )

        self.LB = bool(LB) if kind == "load_balancing" else False
        self.balance_weights: np.ndarray | None = None
        self.R_min: float | None = None
        self.R_max: float | None = None
        if self.LB:
            if K is None:
                raise ValueError(
                    "K is required when load-balancing constraints are active."
                )
            if balance_weights is None:
                weights = np.ones(self.n_nodes, dtype=float)
            else:
                weights = np.asarray(balance_weights, dtype=float)
                if weights.shape != (self.n_nodes,):
                    raise ValueError(
                        "balance_weights must contain one value per node."
                    )
                if not np.all(np.isfinite(weights)):
                    raise ValueError(
                        "balance_weights must contain only finite values."
                    )
                if np.any(weights <= 0):
                    raise ValueError("balance_weights must contain positive values.")
            total_weight = float(np.sum(weights))
            if R == 0 and R_bounds is None and total_weight % K != 0:
                raise ValueError(
                    "Infeasible R and K combination, given the total node weight."
                )
            self.R_min, self.R_max = resolve_balance_bounds(
                total_weight,
                K,
                R,
                R_bounds,
            )
            self.balance_weights = weights

        self.model = self._build_model()
        self.solver = _configured_gurobi_solver(solver)
        try:
            self.solver.set_instance(
                self.model,
                skip_trivial_constraints=False,
            )
        except Exception:
            self.close()
            raise

    @property
    def column_count(self) -> int:
        """Return the number of columns currently attached to the model."""
        return len(self.columns)

    def _column_loads(self, column: MatrixLike) -> np.ndarray:
        """Return each node's community load for one column."""
        if self.balance_weights is None:
            raise RuntimeError("Load coefficients requested without balance weights.")
        return np.asarray(column @ self.balance_weights, dtype=float).reshape(-1)

    def _build_model(self):
        """Build the initial continuous restricted master."""
        model = ConcreteModel()
        model.I = RangeSet(0, self.n_nodes - 1)
        model.C = Set(initialize=range(len(self.columns)), ordered=True)
        model.lmbd = Var(
            model.C,
            domain=NonNegativeReals,
            bounds=(0, None),
            initialize=0,
        )
        model.OneColumn = Constraint(
            expr=sum(model.lmbd[c] for c in model.C) == 1
        )

        if self.unworthy_edges:
            model.WorthyEdges = Constraint(
                self.unworthy_edges,
                rule=lambda mdl, i, j: sum(
                    mdl.lmbd[c] * matrix_scalar(self.columns[c], i, j)
                    for c in mdl.C
                )
                == 1,
            )
        if self.cannot_link:
            model.CannotLink = Constraint(
                self.cannot_link,
                rule=lambda mdl, i, j: sum(
                    mdl.lmbd[c] * matrix_scalar(self.columns[c], i, j)
                    for c in mdl.C
                )
                == 0,
            )
        if self.must_link:
            model.MustLink = Constraint(
                self.must_link,
                rule=lambda mdl, i, j: sum(
                    mdl.lmbd[c] * matrix_scalar(self.columns[c], i, j)
                    for c in mdl.C
                )
                == 1,
            )
        if self.LB:
            initial_loads = [self._column_loads(column) for column in self.columns]
            model.Rmin = Constraint(
                model.I,
                rule=lambda mdl, i: self.R_min
                <= sum(
                    mdl.lmbd[c] * float(initial_loads[c][i]) for c in mdl.C
                ),
            )
            model.Rmax = Constraint(
                model.I,
                rule=lambda mdl, i: self.R_max
                >= sum(
                    mdl.lmbd[c] * float(initial_loads[c][i]) for c in mdl.C
                ),
            )

        model.OBJ = Objective(
            expr=sum(
                self.f_stars[c] * model.lmbd[c] for c in model.C
            ),
            sense=maximize,
        )
        model.dual = Suffix(direction=Suffix.IMPORT)
        return model

    def _column_coefficients(
        self,
        column: MatrixLike,
    ) -> tuple[list[Any], list[float]]:
        """Return existing rows and one new column's coefficients."""
        constraints: list[Any] = [self.model.OneColumn]
        coefficients = [1.0]
        if self.unworthy_edges:
            constraints.extend(
                self.model.WorthyEdges[i, j] for i, j in self.unworthy_edges
            )
            coefficients.extend(
                float(matrix_scalar(column, i, j))
                for i, j in self.unworthy_edges
            )
        if self.cannot_link:
            constraints.extend(
                self.model.CannotLink[i, j] for i, j in self.cannot_link
            )
            coefficients.extend(
                float(matrix_scalar(column, i, j))
                for i, j in self.cannot_link
            )
        if self.must_link:
            constraints.extend(
                self.model.MustLink[i, j] for i, j in self.must_link
            )
            coefficients.extend(
                float(matrix_scalar(column, i, j)) for i, j in self.must_link
            )
        if self.LB:
            loads = self._column_loads(column)
            constraints.extend(self.model.Rmin[i] for i in self.model.I)
            coefficients.extend(float(loads[i]) for i in self.model.I)
            constraints.extend(self.model.Rmax[i] for i in self.model.I)
            coefficients.extend(float(loads[i]) for i in self.model.I)
        return constraints, coefficients

    def sync(
        self,
        columns: Sequence[MatrixLike],
        f_stars: Sequence[float],
    ) -> None:
        """Append columns that are not yet present in the persistent model."""
        if self._closed:
            raise RuntimeError("The persistent master session is closed.")
        if self._integer_mode:
            raise RuntimeError("Columns cannot be appended after the integer solve.")
        if len(columns) != len(f_stars):
            raise ValueError("columns and f_stars must have the same length.")
        if len(columns) < self.column_count:
            raise ValueError("The persistent master column pool cannot shrink.")
        for index in range(self.column_count):
            if (
                columns[index] is not self.columns[index]
                or float(f_stars[index]) != self.f_stars[index]
            ):
                raise ValueError(
                    "The persistent master requires an append-only column pool; "
                    "existing columns and scores cannot be replaced."
                )

        for index in range(self.column_count, len(columns)):
            column = columns[index]
            if column.shape != (self.n_nodes, self.n_nodes):
                raise ValueError(
                    "Every persistent master column must match the adjacency shape."
                )
            score = float(f_stars[index])
            self.columns.append(column)
            self.f_stars.append(score)
            self.model.C.add(index)
            variable = self.model.lmbd[index]
            constraints, coefficients = self._column_coefficients(column)
            self.solver.add_column(
                self.model,
                variable,
                score,
                constraints,
                coefficients,
            )

    def _extract_base_duals(self) -> dict[str, Any]:
        """Extract duals using the base master's public representation."""
        duals: dict[str, Any] = {
            "mu_dual": self.model.dual.get(self.model.OneColumn, 0)
        }
        for name, pairs, key in (
            ("CannotLink", self.cannot_link, "tau_dual"),
            ("MustLink", self.must_link, "gamma_dual"),
        ):
            if not pairs:
                continue
            rows, columns, values = [], [], []
            component = getattr(self.model, name)
            for i, j in pairs:
                dual = self.model.dual.get(component[i, j], 0)
                if dual:
                    rows.append(i)
                    columns.append(j)
                    values.append(dual)
            duals[key] = sparse.csr_matrix(
                (values, (rows, columns)),
                shape=(self.n_nodes, self.n_nodes),
                dtype=float,
            )
        if self.worthy_edges_enabled:
            rows, columns, values = [], [], []
            if self.unworthy_edges:
                for i, j in self.unworthy_edges:
                    dual = self.model.dual.get(self.model.WorthyEdges[i, j], 0)
                    if dual:
                        rows.append(i)
                        columns.append(j)
                        values.append(dual)
            duals["pi_dual"] = sparse.csr_matrix(
                (values, (rows, columns)),
                shape=(self.n_nodes, self.n_nodes),
                dtype=float,
            )
        return duals

    def _extract_load_balancing_duals(self) -> dict[str, Any]:
        """Extract duals using the LB master's public representation."""
        duals: dict[str, Any] = {
            "mu_dual": self.model.dual.get(self.model.OneColumn, 0)
        }
        if self.LB:
            tau_node = np.array(
                [self.model.dual.get(self.model.Rmin[i], 0) for i in self.model.I],
                dtype=float,
            )
            pi_node = np.array(
                [self.model.dual.get(self.model.Rmax[i], 0) for i in self.model.I],
                dtype=float,
            )
            weights = np.asarray(self.balance_weights, dtype=float)
            duals["tau_dual"] = 0.5 * (
                tau_node[:, None] * weights[None, :]
                + weights[:, None] * tau_node[None, :]
            )
            duals["pi_dual"] = 0.5 * (
                pi_node[:, None] * weights[None, :]
                + weights[:, None] * pi_node[None, :]
            )
        for name, pairs, key in (
            ("CannotLink", self.cannot_link, "rho_dual"),
            ("MustLink", self.must_link, "gamma_dual"),
        ):
            if not pairs:
                continue
            values = np.zeros((self.n_nodes, self.n_nodes), dtype=float)
            component = getattr(self.model, name)
            for i, j in pairs:
                values[i, j] = self.model.dual.get(component[i, j], 0)
            duals[key] = values
        return duals

    def solve(self, *, extract_dual: bool):
        """Solve the synchronized continuous or final integer master."""
        if self._closed:
            raise RuntimeError("The persistent master session is closed.")
        if extract_dual and self._integer_mode:
            raise RuntimeError("The persistent master has entered integer mode.")
        if not extract_dual and not self._integer_mode:
            self.model.del_component(self.model.dual)
            for variable in self.model.lmbd.values():
                variable.domain = Binary
                self.solver.update_var(variable)
            self._integer_mode = True

        if extract_dual:
            self.model.dual.clear()

        result = self.solver.solve(tee=bool(self.verbose is True))
        condition = result.solver.termination_condition

        if condition == TerminationCondition.infeasible:
            return (None, None, None) if extract_dual else (None, None)

        if condition != TerminationCondition.optimal:
            raise RuntimeError(f"Master solve ended without optimality: {condition}")

        variables = list(self.model.lmbd.values())
        if any(variable.stale or variable.value is None for variable in variables):
            raise RuntimeError(
                "The persistent master solve did not return a usable solution "
                f"({condition})."
            )
        try:
            lambda_sol = [value(self.model.lmbd[c]) for c in self.model.C]
            master_obj_val = value(self.model.OBJ)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                "The persistent master solve did not return a usable solution."
            ) from exc
        if not extract_dual:
            return lambda_sol, master_obj_val
        duals = (
            self._extract_base_duals()
            if self.kind == "base"
            else self._extract_load_balancing_duals()
        )
        return lambda_sol, duals, master_obj_val

    def close(self) -> None:
        """Release the owned Gurobi model and environment."""
        if self._closed:
            return
        self._closed = True
        solver = getattr(self, "solver", None)
        if solver is not None:
            solver.close()


def create_persistent_master_session(
    mp_function: Any,
    A: MatrixLike,
    columns: Sequence[MatrixLike],
    f_stars: Sequence[float],
    *,
    cannot_link: Sequence[tuple[int, int]],
    must_link: Sequence[tuple[int, int]],
    additional_constraints: dict[str, Any],
    verbose: int | bool,
) -> PersistentMasterSession:
    """Create the persistent equivalent of a supported built-in master."""
    from asunder.base.column_generation.master import solve_master_problem as base_master
    from asunder.load_balancing.column_generation.master import (
        solve_master_problem as load_balancing_master,
    )

    options = dict(additional_constraints)
    if mp_function is base_master:
        supported = {"worthy_edges", "solver"}
        unexpected = set(options) - supported
        if unexpected:
            names = ", ".join(sorted(unexpected))
            raise TypeError(
                f"The base master does not accept persistent options: {names}."
            )
        return PersistentMasterSession(
            A,
            columns,
            f_stars,
            kind="base",
            cannot_link=cannot_link,
            must_link=must_link,
            worthy_edges=options.get("worthy_edges"),
            solver=options.get("solver"),
            verbose=verbose,
        )
    if mp_function is load_balancing_master:
        supported = {
            "LB",
            "R",
            "K",
            "R_bounds",
            "balance_weights",
            "solver",
        }
        unexpected = set(options) - supported
        if unexpected:
            names = ", ".join(sorted(unexpected))
            raise TypeError(
                "The load-balancing master does not accept persistent options: "
                f"{names}."
            )
        return PersistentMasterSession(
            A,
            columns,
            f_stars,
            kind="load_balancing",
            cannot_link=cannot_link,
            must_link=must_link,
            LB=options.get("LB", True),
            R=options.get("R", 1),
            K=options.get("K", 2),
            R_bounds=options.get("R_bounds"),
            balance_weights=options.get("balance_weights"),
            solver=options.get("solver"),
            verbose=verbose,
        )
    raise ValueError(
        "persistent_master=True supports only Asunder's built-in base and "
        "load-balancing master functions."
    )


def validate_persistent_master_function(mp_function: Any) -> None:
    """Reject unsupported custom masters before decomposition side effects."""
    from asunder.base.column_generation.master import solve_master_problem as base_master
    from asunder.load_balancing.column_generation.master import (
        solve_master_problem as load_balancing_master,
    )

    if mp_function not in {base_master, load_balancing_master}:
        raise ValueError(
            "persistent_master=True supports only Asunder's built-in base and "
            "load-balancing master functions."
        )
