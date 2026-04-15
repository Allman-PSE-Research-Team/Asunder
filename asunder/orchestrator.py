"""High-level decomposition orchestrator."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

import numpy as np

from asunder.base.column_generation.decomposition import CSD_decomposition
from asunder.base.column_generation.master import solve_master_problem
from asunder.base.column_generation.subproblem import heuristic_subproblem
from asunder.config import CSDDecompositionConfig
from asunder.types import DecompositionResult, IterationRecord, MasterProblemFn, SubproblemFn


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

    def run(self, A: np.ndarray, a: np.ndarray | None = None, m: float | None = None, **overrides: Any) -> DecompositionResult:
        """
        Execute decomposition and return typed iteration records.
        
        Parameters
        ----------
        A : np.ndarray of int | float, shape (N, N)
            Adjacency / weight matrix.
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
            a = A.sum(axis=1)
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
        for item in raw:
            duals = {
                k: v
                for k, v in item.items()
                if k
                not in {"lambda_sol", "master_obj_val", "z_sol", "heuristic_col", "sub_obj_val", "columns", "f_stars"}
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
                )
            )
        final = records[-1] if records else None
        return DecompositionResult(
            records=records,
            final_partition=(final.z_sol if final else None),
            final_master_obj=(final.master_obj_val if final else None),
            metadata={"n_iterations": len(records)},
        )


def run_csd_decomposition(
    A: np.ndarray,
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
    A : np.ndarray of int | float, shape (N, N)
        Adjacency / weight matrix.
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
