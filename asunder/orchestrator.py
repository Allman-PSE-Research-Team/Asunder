"""High-level decomposition orchestrator."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

import numpy as np

from asunder.base.column_generation.decomposition import CSD_decomposition
from asunder.base.column_generation.master import solve_master_problem
from asunder.base.column_generation.subproblem import heuristic_subproblem
from asunder.base.utils.graph import (
    partition_satisfies_pairwise_constraints,
    validate_partition_matrix,
)
from asunder.base.utils.matrix import (
    DEFAULT_MAX_DENSE_WORKING_BYTES,
    matrix_storage,
    structural_edge_pairs,
)
from asunder.config import CSDDecompositionConfig
from asunder.types import (
    DecompositionResult,
    IterationRecord,
    MasterProblemFn,
    MatrixLike,
    SubproblemFn,
)


def _one_hot_master_partition(
    item: dict[str, Any], *, max_dense_working_bytes=DEFAULT_MAX_DENSE_WORKING_BYTES,
) -> MatrixLike | None:
    """Recover an integral selected column from a master record, if present."""
    lambda_sol = item.get("lambda_sol")
    columns = item.get("columns") or []
    if lambda_sol is None or len(lambda_sol) != len(columns) or not columns:
        return None
    values = np.asarray(lambda_sol, dtype=float)
    if not np.all(np.isfinite(values)):
        return None
    selected = int(np.argmax(values))
    if not np.isclose(values[selected], 1.0, atol=1e-7, rtol=0):
        return None
    if np.any(np.delete(values, selected) > 1e-7):
        return None
    partition = columns[selected]
    node2comp = item.get("node2comp")
    if node2comp is not None:
        from asunder.base.utils.graph import expand_z_matrix

        partition = expand_z_matrix(partition, node2comp, max_dense_working_bytes=max_dense_working_bytes)
    return partition


def _validated_final_candidate(
    partition: MatrixLike,
    *,
    n_nodes: int,
    must_link,
    cannot_link,
    additional_constraints,
    max_dense_working_bytes=DEFAULT_MAX_DENSE_WORKING_BYTES,
) -> MatrixLike | None:
    """Validate structural, pairwise, and load-balancing constraints."""
    try:
        candidate = validate_partition_matrix(
            partition,
            n_nodes,
            name="final partition",
            max_dense_working_bytes=max_dense_working_bytes,
        )
    except ValueError:
        return None
    if not partition_satisfies_pairwise_constraints(
        candidate,
        must_link=must_link,
        cannot_link=cannot_link,
    ):
        return None
    constraints = additional_constraints or {}
    if constraints.get("LB"):
        from asunder.load_balancing.utils.balance import (
            partition_satisfies_balance_constraints,
        )

        if not partition_satisfies_balance_constraints(
            candidate,
            K=constraints.get("K"),
            R=constraints.get("R", 0),
            R_bounds=constraints.get("R_bounds"),
            balance_weights=constraints.get("balance_weights"),
        ):
            return None
    return candidate


def _select_final_partition(
    raw: list[dict[str, Any]],
    *,
    n_nodes: int,
    must_link,
    cannot_link,
    additional_constraints,
    max_dense_working_bytes=DEFAULT_MAX_DENSE_WORKING_BYTES,
    A: MatrixLike | None = None,
) -> tuple[MatrixLike | None, str | None]:
    """Select only a genuine integral solution, never a pricing candidate."""
    worthy_edges = (additional_constraints or {}).get("worthy_edges")
    if worthy_edges is not None:
        if A is None:
            raise ValueError("A is required to validate final edge constraints.")
        worthy = {tuple(sorted(edge)) for edge in worthy_edges}
        must_link = [
            *(must_link or ()),
            *(pair for pair in structural_edge_pairs(A) if pair not in worthy),
        ]
    solution_sources = {
        "contracted_trivial",
        "integer_master",
        "post_loop_refinement",
    }
    for item in reversed(raw):
        if item.get("partition_source") in solution_sources and item.get("z_sol") is not None:
            candidate = _validated_final_candidate(
                item["z_sol"],
                n_nodes=n_nodes,
                must_link=must_link,
                cannot_link=cannot_link,
                additional_constraints=additional_constraints,
                max_dense_working_bytes=max_dense_working_bytes,
            )
            if candidate is not None:
                return candidate, item["partition_source"]
    for item in reversed(raw):
        partition = _one_hot_master_partition(item, max_dense_working_bytes=max_dense_working_bytes)
        if partition is not None:
            candidate = _validated_final_candidate(
                partition,
                n_nodes=n_nodes,
                must_link=must_link,
                cannot_link=cannot_link,
                additional_constraints=additional_constraints,
                max_dense_working_bytes=max_dense_working_bytes,
            )
            if candidate is not None:
                return candidate, "one_hot_relaxed_master"
    return None, None


class CSDDecomposition:
    """
    High-level driver that wires configuration to master/subproblem hooks.
    
    Parameters
    ----------
    config : CSDDecompositionConfig | None
        Column generation configuration.
    master_fn : MasterProblemFn
        Master problem function.
    subproblem_fn : SubproblemFn
        Subproblem function.
    """

    def __init__(
        self,
        config: CSDDecompositionConfig | None = None,
        master_fn: MasterProblemFn = solve_master_problem,
        subproblem_fn: SubproblemFn = heuristic_subproblem,
    ) -> None:
        self.config = config or CSDDecompositionConfig()
        self.master_fn = master_fn
        self.subproblem_fn = subproblem_fn

    def run(self, A: MatrixLike, a: np.ndarray | None = None, m: float | None = None, **overrides: Any) -> DecompositionResult:
        """
        Execute decomposition and return typed iteration records.
        
        Parameters
        ----------
        A : numpy.ndarray or scipy.sparse.csr_matrix, shape (N, N)
            Adjacency or weight matrix. Sparse input is normalized to CSR and
            preserved through sparse-compatible stages.
        a : np.ndarray of int | float, shape (N,)
            Degree-like vector; defaults to row sums of the symmetrized adjacency.
        m : float
            Twice the total weight in the graph.
        **overrides : Any
            Additional keyword arguments.
        
        Returns
        -------
        DecompositionResult
            Computed decomposition result object.
        """
        if a is None:
            a = np.asarray(A.sum(axis=1)).reshape(-1)
        if m is None:
            m = float(np.sum(a))
        cfg = asdict(self.config)
        cfg.update(overrides)
        verbose = int(cfg.pop("verbose", 1))
        cfg["verbose"] = -1 if verbose <= 0 else verbose
        raw = CSD_decomposition(A, a, m, self.master_fn, self.subproblem_fn, **cfg)
        if raw is None:
            return DecompositionResult(records=[], final_partition=None, final_master_obj=None, metadata={"status": "infeasible"})
        records = []
        node2comp = None
        for item in raw:
            if item.get("node2comp") is not None:
                node2comp = np.asarray(item["node2comp"], dtype=int).copy()
            duals = {
                k: v
                for k, v in item.items()
                if k
                not in {
                    "lambda_sol",
                    "master_obj_val",
                    "z_sol",
                    "heuristic_col",
                    "sub_obj_val",
                    "columns",
                    "f_stars",
                    "node2comp",
                    "partition_source",
                    "storage_metadata",
                }
            }
            records.append(
                IterationRecord(
                    lambda_sol=item.get("lambda_sol"),
                    duals=duals,
                    master_obj_val=item.get("master_obj_val"),
                    z_sol=item.get("z_sol"),
                    heuristic_col=item.get("heuristic_col"),
                    sub_obj_val=item.get("sub_obj_val"),
                    columns=item.get("columns", []),
                    f_stars=item.get("f_stars", []),
                    partition_source=item.get("partition_source"),
                )
            )
        final = records[-1] if records else None
        final_partition, final_partition_source = _select_final_partition(
            raw,
            A=A,
            n_nodes=A.shape[0],
            must_link=cfg.get("must_link"),
            cannot_link=cfg.get("cannot_link"),
            additional_constraints=cfg.get("additional_constraints"),
            max_dense_working_bytes=cfg["max_dense_working_bytes"],
        )
        metadata = {
            "n_iterations": len(records),
            "resolution": float(cfg["resolution"]),
            "final_partition_source": final_partition_source,
            "status": "ok" if final_partition is not None else "no_integral_partition",
        }
        if raw:
            metadata.update(raw[-1].get("storage_metadata", {}))
        metadata["final_partition_storage"] = (
            None if final_partition is None else matrix_storage(final_partition)
        )
        if node2comp is not None:
            metadata["node2comp"] = node2comp
        return DecompositionResult(
            records=records,
            final_partition=final_partition,
            final_master_obj=(final.master_obj_val if final else None),
            metadata=metadata,
        )


def run_csd_decomposition(
    A: MatrixLike,
    a: np.ndarray | None = None,
    m: float | None = None,
    config: CSDDecompositionConfig | None = None,
    master_fn: MasterProblemFn = solve_master_problem,
    subproblem_fn: SubproblemFn = heuristic_subproblem,
    **kwargs: Any,
) -> DecompositionResult:
    """
    Convenience wrapper for one-shot decomposition runs.
    
    Parameters
    ----------
    A : numpy.ndarray or scipy.sparse.csr_matrix, shape (N, N)
        Adjacency or weight matrix. Sparse input is normalized to CSR and
        preserved through sparse-compatible stages.
    a : np.ndarray of int | float, shape (N,)
        Degree-like vector; defaults to row sums of the symmetrized adjacency.
    m : float
        Twice the total weight in the graph.
    config : CSDDecompositionConfig | None
        Column generation configuration.
    master_fn : MasterProblemFn
        Master problem function.
    subproblem_fn : SubproblemFn
        Subproblem function.
    **kwargs : Any
        Additional keyword arguments.
    
    Returns
    -------
    DecompositionResult
        Computed decomposition result object.
    """
    orchestrator = CSDDecomposition(config=config, master_fn=master_fn, subproblem_fn=subproblem_fn)
    return orchestrator.run(A, a=a, m=m, **kwargs)
