"""High-level decomposition orchestrator."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

import numpy as np

from asunder.column_generation.decomposition import CSD_decomposition
from asunder.column_generation.master import solve_master_problem
from asunder.column_generation.subproblem import heuristic_subproblem
from asunder.config import CSDDecompositionConfig
from asunder.types import DecompositionResult, IterationRecord, MasterProblemFn, SubproblemFn


class CSDDecomposition:
    """High-level driver that wires configuration to master/subproblem hooks."""

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
        """Execute decomposition and return typed iteration records."""
        if a is None:
            a = A.sum(axis=1)
        if m is None:
            m = float(np.sum(a))
        cfg = asdict(self.config)
        cfg.update(overrides)
        verbosity = int(cfg.pop("verbosity", 1))
        cfg["verbose"] = -1 if verbosity <= 0 else verbosity
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
    """Convenience wrapper for one-shot decomposition runs."""
    orchestrator = CSDDecomposition(config=config, master_fn=master_fn, subproblem_fn=subproblem_fn)
    return orchestrator.run(A, a=a, m=m, **kwargs)
