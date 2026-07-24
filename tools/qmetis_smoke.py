"""Installed-wheel smoke tests for bundled QMETIS and LoadBalancer routing."""

from __future__ import annotations

import networkx as nx
import numpy as np

from asunder.load_balancing import LoadBalancer
from asunder.load_balancing.algorithms.qmetis import (
    bundled_qmetis_release,
    run_qmetis,
)
from asunder.load_balancing.column_generation.subproblem import (
    qmetis_pricing_subproblem,
)
from asunder.solvers import create_solver, set_default_solver


def main() -> int:
    """Exercise native QMETIS and end-to-end LoadBalancer routing."""

    adjacency = nx.to_numpy_array(nx.path_graph(8), dtype=float)
    partition, modularity = run_qmetis(
        adjacency,
        2,
        balance_epsilon=0.25,
        seed=7,
    )
    if partition.shape != adjacency.shape:
        raise RuntimeError(f"Unexpected QMETIS partition shape {partition.shape}.")
    if not np.isfinite(modularity):
        raise RuntimeError(f"QMETIS returned non-finite modularity {modularity}.")

    a = adjacency.sum(axis=1)
    m = float(a.sum())
    reduced_cost, priced_partition = qmetis_pricing_subproblem(
        adjacency,
        a,
        m,
        {
            "mu_dual": 0.125,
            "tau_dual": np.linspace(0.0, 0.05, adjacency.shape[0]),
            "pi_dual": np.zeros(adjacency.shape[0]),
        },
        K=2,
        R=2,
        seed=7,
    )
    if priced_partition.shape != adjacency.shape or not np.isfinite(reduced_cost):
        raise RuntimeError("QMETIS pricing adapter smoke test failed.")

    solver = create_solver("appsi_highs")
    if not solver.available(exception_flag=False):
        raise RuntimeError("The HiGHS solver required by the release smoke test is unavailable.")
    set_default_solver(solver)
    result = LoadBalancer(
        nx.path_graph(8),
        K=2,
        R=2,
        algorithm="qmetis",
        ifc_generator="ordered",
        refine_post_loop=True,
        max_iterations=3,
        disable_tqdm=True,
        verbose=-1,
    )
    if result.final_partition is None:
        raise RuntimeError("LoadBalancer QMETIS smoke test returned no partition.")
    sizes = np.asarray(result.final_partition).sum(axis=1)
    if sizes.min() < 3 or sizes.max() > 5:
        raise RuntimeError(f"LoadBalancer returned invalid sizes {sizes.tolist()}.")
    if not result.metadata.get("qmetis_release"):
        raise RuntimeError("Bundled QMETIS release metadata is missing.")

    print(
        "qmetis_smoke_ok",
        bundled_qmetis_release(),
        float(modularity),
        float(reduced_cost),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
