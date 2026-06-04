"""Benchmark runner for case studies and algorithms."""

from __future__ import annotations

import time
from collections import defaultdict

import networkx as nx
import numpy as np

from asunder.base.algorithms.core_periphery import (
    partition_periphery_components,
)
from asunder.base.column_generation.decomposition import CSD_decomposition
from asunder.base.column_generation.master import solve_master_problem
from asunder.base.column_generation.subproblem import (
    custom_heuristic_subproblem,
    heuristic_subproblem,
)
from asunder.base.evaluation.metrics import (
    ari_sklearn,
    nmi_sklearn,
    optimality_gap,
    permuted_accuracy,
    vi_sklearn,
)
from asunder.base.utils.graph import partition_matrix_to_vector, partition_vector_to_2d_matrix
from asunder.base.utils.partition_generation import make_simple_partition
from asunder.nlbnp.algorithms.refinement import refine_partition_linear_group
from asunder.nlbnp.algorithms.core_periphery import _detect_core_periphery
from asunder.nlbnp.case_studies.circle_cutting import build_circle_cutting_graph
from asunder.nlbnp.case_studies.cpcong import build_cpcong_graph

NX_ALGOS = ["louvain", "leiden", "greedy", "girvan_newman"]
IGRAPH_ALGOS = ["infomap", "lpa", "multilevel", "voronoi", "walktrap"]


def run_evaluation(problem="cpcong", build_params=None, style="CP", algos=None, repeat=3):
    """
    Run benchmark evaluations for CP (Core-Periphery) and CD_Refine (Community Detection + Refinement to identify linear group) workflows.
    
    Parameters
    ----------
    problem : str
        Shortcode for problem being evaluated.
    build_params : dict
        Parameters used to build graph instance.
    style : str
        Problem style.
    algos : list[str]
        List of algorithms to be evaluated.
    repeat : int
        Number of times to run the evaluation.
    
    Returns
    -------
    results : dict
        Dictionary of evaluation results keyed by algorithm name. Each
        ``results[algo]`` entry is itself a dictionary with the following
        key-value pairs:

        NMI : float
            Normalized mutual information between ``labels_gt`` and the
            predicted labels.

        ARI : float
            Adjusted Rand index between ``labels_gt`` and the predicted
            labels.

        VI : float
            Variation of information between ``labels_gt`` and the
            predicted labels.

        Accuracy : float
            Best permutation-invariant clustering accuracy.

        time : float
            Runtime for the algorithm evaluation, measured in
            seconds.

        Gap : float
            Percentage optimality gap between the ground-truth partition
            and the returned solution.
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
                start = time.perf_counter()
                labels, _ = _detect_core_periphery(
                    A,
                    unworthy_edges=unworthy_edges,
                    nonlinear_nodes=nonlinear_nodes,
                    algorithm=algo,
                    prob_method="gaussian_mixture",
                )
                end = time.perf_counter()

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
        cp_partition = partition_periphery_components(A, labels_gt)
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
            start = time.perf_counter()
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
            end = time.perf_counter()
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
