"""Core-periphery detection helpers for NLBNP workflows."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture

from asunder.base.algorithms.core_periphery import (
    CorePeripheryResult,
    CorePeripheryTarget,
    FullContinuousGeneticBE,
    detect_continuous_KL,
    find_core_advanced,
    normalized_BE_score,
    spectral_continuous_cp_detection,
)


def _cluster_blocks(
    result: CorePeripheryResult,
    *,
    method: str,
    threshold: float,
    seed: int | None,
    verbose: bool,
) -> np.ndarray:
    n_blocks = len(result.blocks)
    if n_blocks == 1:
        return np.zeros(1, dtype=int)

    normalized_method = method.lower()
    if normalized_method not in {"threshold", "gaussian_mixture", "dbscan"}:
        raise ValueError("prob_method must be one of: threshold, gaussian_mixture, DBSCAN.")
    if result.spectral_rank == 2:
        if normalized_method != "gaussian_mixture":
            raise ValueError("spectral_rank=2 requires prob_method='gaussian_mixture'.")
        features = result.block_embedding
    else:
        if result.block_scores is None:
            raise ValueError("Rank-1 detection did not return block coreness scores.")
        features = result.block_scores.reshape(-1, 1)

    if normalized_method == "threshold":
        labels = (features[:, 0] >= threshold).astype(int)
    elif normalized_method == "gaussian_mixture":
        labels = GaussianMixture(n_components=2, random_state=seed).fit_predict(features)
    else:
        span = np.ptp(features, axis=0)
        scaled = (features - features.min(axis=0)) / np.where(span > 1e-12, span, 1.0)
        labels = DBSCAN().fit_predict(scaled)

    if verbose:
        print("Block features are:\n", features)
        print("Unoriented block labels are:\n", labels)
    return np.asarray(labels, dtype=int)


def _orient_and_score(
    A: np.ndarray,
    result: CorePeripheryResult,
    integer_block_labels: np.ndarray,
    must_group: Sequence[int],
) -> CorePeripheryResult:
    if result.target_space == "contracted":
        block_labels, _ = find_core_advanced(
            result.contracted_adjacency,
            integer_block_labels,
            include_diagonal=True,
        )
        node_labels = block_labels[result.node_to_block]
    else:
        provisional_node_labels = integer_block_labels[result.node_to_block]
        node_labels, _ = find_core_advanced(A, provisional_node_labels)
        block_labels = np.asarray(
            [node_labels[block[0]] for block in result.blocks],
            dtype=int,
        )

    result.block_labels = np.asarray(block_labels, dtype=int)
    result.node_labels = np.asarray(node_labels, dtype=int)
    result.contracted_be_score = normalized_BE_score(
        result.contracted_adjacency,
        result.block_labels,
        include_diagonal=True,
    )
    result.original_be_score = normalized_BE_score(A, result.node_labels)
    result.primary_be_score = (
        result.contracted_be_score
        if result.target_space == "contracted"
        else result.original_be_score
    )
    if must_group:
        role = int(result.node_labels[int(must_group[0])])
        result.must_group_role = "core" if role == 1 else "periphery"
    return result


def _detect_core_periphery(
    A: np.ndarray,
    *,
    must_link: Sequence[tuple[int, int]] | None = None,
    must_group: Sequence[int] | None = None,
    algorithm: str = "SPEC",
    target: CorePeripheryTarget = "contracted",
    spectral_rank: int = 1,
    prob_method: str = "gaussian_mixture",
    threshold: float = 0.8,
    verbose: bool = False,
    seed: int | None = 42,
    kl_max_iter: int = 50,
    ga_population_size: int = 50,
    ga_generations: int = 100,
) -> CorePeripheryResult:
    """Detect and orient a constrained binary core-periphery partition.

    Parameters
    ----------
    A : ndarray of float, shape (N, N)
        Symmetric adjacency or weight matrix.
    must_link : sequence of tuple of int, optional
        Node pairs placed in common detection blocks.
    must_group : sequence of int, optional
        Generic set placed in one detection block. Its detected role is
        reported without assuming that it represents a particular node type.
    algorithm : {"SPEC", "GA", "KL"}, default="SPEC"
        Continuous core-periphery backend.
    target : {"contracted", "original"}, default="contracted"
        Space optimized by the backend and used for primary fit and label
        orientation.
    spectral_rank : {1, 2}, default=1
        Spectral approximation rank. Non-default ranks apply only to SPEC;
        rank two requires ``prob_method="gaussian_mixture"``.
    prob_method : {"threshold", "gaussian_mixture", "DBSCAN"}
        Conversion from scalar scores or a rank-two embedding to candidate
        discrete groups.
    threshold : float, default=0.8
        Rank-one cutoff when ``prob_method="threshold"``.
    verbose : bool, default=False
        Print block features and unoriented labels.
    seed : int or None, default=42
        Random seed for stochastic optimization and clustering.
    kl_max_iter : int, default=50
        Maximum KL block-coordinate passes.
    ga_population_size : int, default=50
        GA population size.
    ga_generations : int, default=100
        GA generation count.

    Returns
    -------
    CorePeripheryResult
        Typed continuous and discrete results. ``node_labels`` uses ``1`` for
        core, and all named primary scores correspond to ``target``.
    """

    matrix = np.asarray(A, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("A must be a square adjacency matrix.")
    if matrix.shape[0] == 0:
        raise ValueError("A must contain at least one node.")
    edge_pairs = [] if must_link is None else [(int(i), int(j)) for i, j in must_link]
    grouped_nodes = [] if must_group is None else [int(node) for node in must_group]

    algorithm_name = algorithm.upper()
    if algorithm_name not in {"SPEC", "GA", "KL"}:
        raise ValueError(f"Unsupported CP algorithm: {algorithm}. Expected one of: SPEC, GA, KL.")
    if algorithm_name != "SPEC" and spectral_rank != 1:
        raise ValueError("spectral_rank applies only when algorithm='SPEC'.")

    if algorithm_name == "KL":
        result = detect_continuous_KL(
            matrix,
            must_link=edge_pairs,
            must_group=grouped_nodes,
            target=target,
            max_iter=kl_max_iter,
            seed=seed,
        )
    elif algorithm_name == "GA":
        result = FullContinuousGeneticBE(
            matrix,
            must_link=edge_pairs,
            must_group=grouped_nodes,
            target=target,
            pop_size=ga_population_size,
            generations=ga_generations,
            seed=seed,
        ).run()
    else:
        result = spectral_continuous_cp_detection(
            matrix,
            must_link=edge_pairs,
            must_group=grouped_nodes,
            target=target,
            spectral_rank=spectral_rank,
        )

    if matrix.shape[0] == 1:
        result.block_labels = np.ones(1, dtype=int)
        result.node_labels = np.ones(1, dtype=int)
        result.contracted_be_score = 0.0
        result.original_be_score = 0.0
        result.primary_be_score = 0.0
        if grouped_nodes:
            result.must_group_role = "core"
        return result

    integer_block_labels = _cluster_blocks(
        result,
        method=prob_method,
        threshold=threshold,
        seed=seed,
        verbose=verbose,
    )
    return _orient_and_score(matrix, result, integer_block_labels, grouped_nodes)
