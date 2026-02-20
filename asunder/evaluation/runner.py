"""Benchmark runner for case studies and algorithms."""

from __future__ import annotations

import time
from collections import defaultdict

import networkx as nx
import numpy as np
from scipy.sparse import csr_matrix

from asunder.algorithms.community import (
    probability_to_integer_labels,
    refine_partition_linear_group,
)
from asunder.algorithms.core_periphery import (
    FullContinuousGeneticBE,
    detect_continuous_KL,
    find_core,
    partititon_periphery_components,
    spectral_continuous_cp_detection,
)
from asunder.case_studies.circle_cutting import build_circle_cutting_graph
from asunder.case_studies.cpcong import build_cpcong_graph
from asunder.column_generation.decomposition import CSD_decomposition
from asunder.column_generation.master import solve_master_problem
from asunder.column_generation.subproblem import custom_heuristic_subproblem, heuristic_subproblem
from asunder.evaluation.metrics import (
    ari_sklearn,
    nmi_sklearn,
    optimality_gap,
    permuted_accuracy,
    vi_sklearn,
)
from asunder.utils.graph import partition_matrix_to_vector, partition_vector_to_2d_matrix
from asunder.utils.partition_generation import make_simple_partition

NX_ALGOS = ["louvain", "leiden", "greedy", "girvan_newman"]
IGRAPH_ALGOS = ["infomap", "lpa", "multilevel", "voronoi", "walktrap"]


def run_evaluation(problem="cpcong", build_params=None, style="CP", algos=None, repeat=3):
    """Run benchmark evaluations for CP (Core-Periphery) and CD_Refine (Community Detection + Refinement to identify linear group) workflows.

    Args:
        problem: Benchmark instance family (``"cpcong"`` or ``"circcut"``).
        build_params: Parameters passed to the chosen case-study graph builder.
        style: Evaluation mode, either ``"CP"`` or ``"CD_Refine"``.
        algos: Algorithms to evaluate for the selected style.
        repeat: Number of repeated runs for stochastic methods.

    Returns:
        Dictionary mapping algorithm names to metric/time summaries.
    """
    build_params = build_params or {"J": 3, "T": 5, "K": 2}
    algos = algos or (["KL", "GA", "SPEC"] if style == "CP" else NX_ALGOS)

    if problem == "cpcong":
        G, _, _ = build_cpcong_graph(**build_params)
        nonlinear_tag = "L"
        core_tag = "CU"
    elif problem == "circcut":
        G, _, _ = build_circle_cutting_graph(**build_params)
        nonlinear_tag = "NO"
        core_tag = "CA"
    else:
        raise NotImplementedError(f"{problem} is either misspelled or has not been implemented.")

    labels_gt = np.array([attr["constraint"] == core_tag for _, attr in G.nodes(data=True)], dtype=int)
    A = nx.to_numpy_array(G)
    n = A.shape[0]
    a = A.sum(axis=1)
    m = a.sum()

    node_labels = list(G.nodes())
    label_node_map = {label: i for i, label in enumerate(node_labels)}
    nonlinear_nodes = [label_node_map[node] for node, attr in G.nodes(data=True) if attr.get("constraint") == nonlinear_tag]

    results = {}
    if style == "CP":
        unworthy_edges = [
            (label_node_map[i], label_node_map[j])
            for i, j, attr in G.edges(data=True)
            if attr.get("var_type") == "continuous"
        ]
        for algo in algos:
            algo_res = {"NMI": -np.inf, "ARI": -np.inf, "VI": np.inf, "Accuracy": -np.inf}
            runs = []
            for _ in range(repeat):
                start = time.time()
                if algo.upper() == "KL":
                    labels_blk, _ = detect_continuous_KL(csr_matrix(A), unworthy_edges, nonlinear_nodes, max_iter=50)
                    integer_labels = probability_to_integer_labels(np.array(labels_blk).reshape(-1, 1), method="gaussian_mixture")
                    labels = find_core(A, integer_labels)
                elif algo.upper() == "GA":
                    ga = FullContinuousGeneticBE(
                        A,
                        must_links=unworthy_edges,
                        nonlinear_nodes=nonlinear_nodes,
                        pop_size=50,
                        generations=100,
                        seed=42,
                    )
                    raw_labels, _ = ga.run()
                    integer_labels = probability_to_integer_labels(
                        np.array(list(raw_labels.values())).reshape(-1, 1), method="gaussian_mixture"
                    )
                    labels = find_core(A, integer_labels)
                elif algo.upper() == "SPEC":
                    labels_blk, _ = spectral_continuous_cp_detection(
                        csr_matrix(A), unworthy_edges, nonlinear_nodes, True
                    )
                    integer_labels = probability_to_integer_labels(labels_blk.reshape(-1, 1), method="gaussian_mixture")
                    labels = find_core(A, integer_labels)
                else:
                    raise NotImplementedError(f"Unsupported CP algorithm: {algo}")
                end = time.time()

                acc = permuted_accuracy(labels_gt, labels)[0]
                algo_res["NMI"] = max(algo_res["NMI"], nmi_sklearn(labels_gt, labels))
                algo_res["ARI"] = max(algo_res["ARI"], ari_sklearn(labels_gt, labels))
                algo_res["VI"] = min(algo_res["VI"], vi_sklearn(labels_gt, labels))
                algo_res["Accuracy"] = max(algo_res["Accuracy"], acc)
                runs.append(end - start)

            algo_res["time"] = float(np.mean(runs))
            results[algo] = algo_res
        return results

    if style == "CD_Refine":
        cp_partition = partititon_periphery_components(A, labels_gt, return_sparse=False)
        labels_gt = cp_partition[0] if isinstance(cp_partition, tuple) else cp_partition
        worthy_edges = [
            (label_node_map[i], label_node_map[j])
            for i, j, attr in G.edges(data=True)
            if attr.get("var_type") == "integer"
        ]
        for algo in algos:
            package = "networkx" if algo in NX_ALGOS else ("igraph" if algo in IGRAPH_ALGOS else None)
            ifc_params = {"generator": make_simple_partition, "num": 1, "args": {"N": n}}
            refine_params = {
                "refine_func": refine_partition_linear_group,
                "kwargs": {"prob_method": "DBSCAN" if problem == "cpcong" else "gaussian_mixture", "verbose": False},
            }
            additional_constraints = defaultdict(lambda: None, {"worthy_edges": worthy_edges})
            start = time.time()
            cg_results = CSD_decomposition(
                A,
                a,
                m,
                solve_master_problem,
                custom_heuristic_subproblem if algo in ["full_louvain", "spectral", "one_level_louvain"] else heuristic_subproblem,
                additional_constraints=additional_constraints,
                algo=algo,
                package=package,
                extract_dual=True,
                ifc_params=ifc_params,
                refine_in_subproblem=False,
                refine_params=refine_params,
                use_refined_column=True,
                final_master_solve=False,
                max_iterations=None,
                tolerance=1e-8,
                verbose=-1,
            )
            end = time.time()
            labels = partition_matrix_to_vector(cg_results[-1]["z_sol"])
            results[algo] = {
                "NMI": nmi_sklearn(labels_gt, labels),
                "ARI": ari_sklearn(labels_gt, labels),
                "VI": vi_sklearn(labels_gt, labels),
                "Accuracy": permuted_accuracy(labels_gt, labels)[0],
                "time": end - start,
                "Gap": optimality_gap(A, a, m, partition_vector_to_2d_matrix(labels_gt), cg_results[-1]["z_sol"])
            }
        return results

    raise NotImplementedError("Invalid style entered.")
