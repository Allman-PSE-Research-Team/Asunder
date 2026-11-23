"""Exercise VFD from an installed Asunder distribution, outside the source tree."""

from __future__ import annotations

import argparse

import networkx as nx
import numpy as np

from asunder.load_balancing import LoadBalancer
from asunder.load_balancing.algorithms import VFD, VFD_cpp
from asunder.solvers import create_solver, set_default_solver


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python-only", action="store_true", help="Check a source installation.")
    args = parser.parse_args()

    graph = nx.path_graph(4)
    weights = [2, 1, 1, 2]
    nx.set_node_attributes(graph, dict(enumerate(weights)), "load")
    labels = np.array([0, 0, 1, 1])
    expected = np.equal.outer(labels, labels)
    constraints = dict(K=2, R=0, must_link=[(0, 1)], cannot_link=[(0, 2)])
    inputs = dict(
        A=nx.to_numpy_array(graph), partition=labels, balance_weights=weights,
        wz_is_C_node=True, seed=7, **constraints,
    )
    if args.python_only:
        partition = VFD.refine_partition(**inputs)
        try:
            VFD_cpp.refine_partition(**inputs)
        except RuntimeError as exc:
            if "python tools/build_vfd.py" not in str(exc):
                raise
        else:
            raise RuntimeError("Source installation unexpectedly provided a native VFD library.")
    else:
        partition, meta = VFD_cpp.refine_partition(
            **inputs, return_metadata=True, check_invariants=True
        )
        if meta["backend"] != "cpp" or not any(run["found"] for run in meta["native_runs"]):
            raise RuntimeError("Native VFD did not report a successful search.")
    if not np.array_equal(partition, expected):
        raise RuntimeError(f"Unexpected direct VFD partition: {partition}")

    set_default_solver(create_solver("appsi_highs"))
    result = LoadBalancer(
        graph, **constraints, node_weight_attr="load", contract_graph=True,
        ifc_generator="ordered", refinement_backend="python" if args.python_only else "cpp",
        max_iterations=1, seed=7, disable_tqdm=True,
    )
    if not any(record.partition_source == "post_loop_refinement" for record in result.records):
        raise RuntimeError("The contracted LoadBalancer run did not return a refined column.")
    if not np.array_equal(result.final_partition, expected):
        raise RuntimeError(f"Unexpected contracted LoadBalancer partition: {result.final_partition}")
    print("vfd_smoke_ok")


if __name__ == "__main__":
    main()
