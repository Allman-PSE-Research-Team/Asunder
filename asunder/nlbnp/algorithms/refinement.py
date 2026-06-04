"""Refinement routines specific to nonlinear branch-and-price workflows."""

from __future__ import annotations

import numpy as np

from asunder.base.algorithms.community import (
    labels_to_probabilities,
    probability_to_integer_labels,
)
from asunder.base.utils.graph import partition_matrix_to_vector, partition_vector_to_2d_matrix
from asunder.nlbnp.algorithms.core_periphery import _detect_core_periphery


def refine_partition_linear_group(
    A, partition, *, p=1, prob_method="threshold", threshold=0.8, verbose=False, seed=42
):
    """
    Refine a partition by separating a linear-only group from the remaining nodes.

    Parameters
    ----------
    A : ndarray of float, shape (N, N)
        Graph adjacency/weight matrix.
    partition : ndarray of int, shape (N,) or (N, N)
        Predicted community labels or a 2D partition matrix.
    p : int
        Order of the norm. Defaults to the L1 norm.
    prob_method : str
        One of ``"threshold"``, ``"gaussian_mixture"``, or ``"DBSCAN"``.
    threshold : float
        Value below which a node is reassigned to the linear-only group.
    verbose : bool
        Controls the verbosity of the output.

    Returns
    -------
    ndarray of int, shape (N, N)
        Refined 2D partition matrix.
    """
    labels = partition_matrix_to_vector(partition) if partition.ndim == 2 else partition.copy()
    probs = labels_to_probabilities(A, labels, p=p).toarray()
    refined_labels = probability_to_integer_labels(
        probs, method=prob_method, threshold=threshold, verbose=verbose
    )
    return partition_vector_to_2d_matrix(refined_labels)

def refine_partition_with_cp(
    A,
    partition,
    *,
    unworthy_edges=None,
    nonlinear_nodes=None,
    cp_algorithm="SPEC",
    prob_method="gaussian_mixture",
    threshold=0.8,
    verbose=False,
    seed=42,
):
    """
    Refine a partition by detecting and merging a standalone core community.

    Existing periphery community assignments are preserved. All detected core
    nodes are assigned to one shared community.

    Parameters
    ----------
    A : ndarray of float, shape (N, N)
        Graph adjacency/weight matrix.
    partition : ndarray of int, shape (N,) or (N, N)
        Predicted community labels or a 2D partition matrix.
    unworthy_edges : list[tuple[int, int]] or None
        Node pairs that must be linked because the edges between them cannot connect nodes in different communities.
    nonlinear_nodes : list[int] | None
        Nodes that correspond to nonlinear constraints and so, should be merged.
    cp_algorithm : str
        Core periphery algorithm to be used. Should be one of:
        ``"SPEC"``: Continuous spectral core periphery detection
        ``"GA"``: Continuous genetic search for BE objective
        ``"KL"``: Continuous Kernighan-Lin algorithm
    prob_method : str
        One of ``"threshold"``, ``"gaussian_mixture"``, or ``"DBSCAN"``.
    threshold : float
        Value below which a node is reassigned to the linear-only group.
    verbose : bool
        Controls the verbosity of the output.
    seed : int | None
        Random seed value.

    Returns
    -------
    ndarray of int, shape (N, N)
        Refined 2D partition matrix.
    """
    partition_array = np.asarray(partition)
    partition_labels = (
        partition_matrix_to_vector(partition_array) if partition_array.ndim == 2 else partition_array.copy()
    )
    if partition_labels.ndim != 1 or partition_labels.shape[0] != np.asarray(A).shape[0]:
        raise ValueError("partition must contain one label per adjacency-matrix node.")

    cp_labels, _ = _detect_core_periphery(
        A,
        unworthy_edges=unworthy_edges,
        nonlinear_nodes=nonlinear_nodes,
        algorithm=cp_algorithm,
        prob_method=prob_method,
        threshold=threshold,
        verbose=verbose,
        seed=seed,
    )
    refined_partition = np.where(cp_labels == 1, -1, partition_labels)
    return partition_vector_to_2d_matrix(refined_partition)
