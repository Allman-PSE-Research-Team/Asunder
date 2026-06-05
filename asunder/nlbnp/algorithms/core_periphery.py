"""Core-periphery detection helpers for NLBNP workflows."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from scipy import sparse

from asunder.base.algorithms.community import probability_to_integer_labels
from asunder.base.algorithms.core_periphery import (
    FullContinuousGeneticBE,
    detect_continuous_KL,
    find_core_advanced,
    spectral_continuous_cp_detection,
)


def _detect_core_periphery(
    A: np.ndarray,
    *,
    unworthy_edges: Sequence[tuple[int, int]] | None = None,
    nonlinear_nodes: Sequence[int] | None = None,
    algorithm: str = "SPEC",
    prob_method: str = "gaussian_mixture",
    threshold: float = 0.8,
    verbose: bool = False,
    seed: int | None = 42,
    kl_max_iter: int = 50,
    ga_population_size: int = 50,
    ga_generations: int = 100,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Detect a constrained binary core-periphery partition.

    Parameters
    ----------
    A : ndarray of float, shape (N, N)
        Graph adjacency/weight matrix.
    unworthy_edges : sequence of tuple[int, int], optional
        Node pairs that must remain together because the edge between them
        cannot connect separate communities.
    nonlinear_nodes : sequence of int, optional
        Nodes that represent nonlinear constraints and should remain together.
    algorithm : {"SPEC", "GA", "KL"}
        Core-periphery detection algorithm.
    prob_method : {"threshold", "gaussian_mixture", "DBSCAN"}
        Method used to convert continuous coreness values to discrete labels.
    threshold : float
        Threshold used when ``prob_method="threshold"``.
    verbose : bool
        Controls probability conversion output.
    seed : int or None
        Random seed used by stochastic algorithms.
    kl_max_iter : int
        Maximum iterations for the KL algorithm.
    ga_population_size : int
        Population size for the genetic algorithm.
    ga_generations : int
        Number of genetic algorithm generations.

    Returns
    -------
    core_labels : ndarray of int, shape (N,)
        Binary labels where ``1`` identifies core nodes.
    metadata : dict
        Detection metadata including continuous labels, intermediate integer
        labels, the selected algorithm, and the final core score.
    """
    matrix = np.asarray(A, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("A must be a square adjacency matrix.")

    n_nodes = matrix.shape[0]
    if n_nodes == 0:
        raise ValueError("A must contain at least one node.")
    edge_pairs = [] if unworthy_edges is None else [(int(i), int(j)) for i, j in unworthy_edges]
    nonlinear = [] if nonlinear_nodes is None else [int(node) for node in nonlinear_nodes]
    for source, target in edge_pairs:
        if not (0 <= source < n_nodes and 0 <= target < n_nodes):
            raise ValueError("unworthy_edges contains a node index outside the adjacency matrix.")
    if any(not 0 <= node < n_nodes for node in nonlinear):
        raise ValueError("nonlinear_nodes contains a node index outside the adjacency matrix.")

    algorithm_name = algorithm.upper()
    if algorithm_name not in {"SPEC", "GA", "KL"}:
        raise ValueError(f"Unsupported CP algorithm: {algorithm}. Expected one of: SPEC, GA, KL.")
    if n_nodes == 1:
        labels = np.ones(1, dtype=int)
        return labels, {
            "algorithm": algorithm_name,
            "continuous_labels": labels.astype(float),
            "continuous_score": 0.0,
            "integer_labels": labels.copy(),
            "core_score": 0.0,
        }

    if algorithm_name == "KL":
        continuous_labels, continuous_score = detect_continuous_KL(
            sparse.csr_matrix(matrix),
            edge_pairs,
            nonlinear,
            max_iter=kl_max_iter,
            seed=seed,
        )
    elif algorithm_name == "GA":
        ga = FullContinuousGeneticBE(
            matrix,
            must_links=edge_pairs,
            nonlinear_nodes=nonlinear,
            pop_size=ga_population_size,
            generations=ga_generations,
            seed=seed,
        )
        raw_labels, continuous_score = ga.run()
        continuous_labels = np.asarray([raw_labels[node] for node in range(n_nodes)], dtype=float)
    elif algorithm_name == "SPEC":
        continuous_labels, continuous_score = spectral_continuous_cp_detection(
            sparse.csr_matrix(matrix),
            edge_pairs,
            nonlinear,
            True,
        )
    continuous_labels = np.asarray(continuous_labels, dtype=float).reshape(-1)
    if continuous_labels.shape[0] != n_nodes:
        raise ValueError(
            f"Core-periphery algorithm returned {continuous_labels.shape[0]} labels for {n_nodes} nodes."
        )

    integer_labels = probability_to_integer_labels(
        continuous_labels.reshape(-1, 1),
        method=prob_method,
        threshold=threshold,
        verbose=verbose,
    )
    core_labels, core_score = find_core_advanced(matrix, integer_labels)
    return np.asarray(core_labels, dtype=int), {
        "algorithm": algorithm_name,
        "continuous_labels": continuous_labels,
        "continuous_score": float(continuous_score),
        "integer_labels": np.asarray(integer_labels, dtype=int),
        "core_score": float(core_score),
    }
