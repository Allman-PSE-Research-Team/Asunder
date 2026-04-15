"""Refinement routines specific to nonlinear branch-and-price workflows."""

from __future__ import annotations

from asunder.base.algorithms.community import (
    labels_to_probabilities,
    probability_to_integer_labels,
)
from asunder.base.utils.graph import partition_matrix_to_vector, partition_vector_to_2d_matrix


def refine_partition_linear_group(
    A, partition, *, p=1, prob_method="threshold", threshold=0.8, verbose=False
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
