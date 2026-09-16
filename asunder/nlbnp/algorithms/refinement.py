"""Refinement routines specific to nonlinear branch-and-price workflows."""

from __future__ import annotations

import numpy as np
from scipy import sparse

from asunder.base.algorithms.community import (
    labels_to_probabilities,
    probability_to_integer_labels,
)
from asunder.base.utils.graph import (
    partition_matrix_to_vector,
    partition_vector_to_2d_matrix,
    validate_partition_matrix,
)
from asunder.base.utils.matrix import (
    DEFAULT_MAX_DENSE_WORKING_BYTES,
    checked_to_dense,
)
from asunder.nlbnp.algorithms.core_periphery import (
    _detect_core_periphery,
    _nlbnp_linear_only_mask,
)
from asunder.nlbnp.algorithms.linear_group import (
    merge_linear_only_communities,
    required_together_components,
)
from asunder.types import MatrixLike


def _hard_partition_labels(partition, n_nodes: int) -> tuple[np.ndarray, np.ndarray]:
    if sparse.issparse(partition):
        matrix = validate_partition_matrix(partition, n_nodes, name="partition")
        return partition_matrix_to_vector(matrix), matrix
    values = np.asarray(partition)
    if values.ndim == 2:
        matrix = validate_partition_matrix(values, n_nodes, name="partition")
        return partition_matrix_to_vector(matrix), matrix
    if values.ndim != 1 or values.shape != (n_nodes,):
        raise ValueError("partition must contain one label per adjacency-matrix node.")
    labels = values.copy()
    return labels, partition_vector_to_2d_matrix(labels)


def refine_partition_linear_group(
    A: MatrixLike,
    partition: MatrixLike,
    *,
    nonlinear_nodes=None,
    worthy_edges=None,
    must_link=None,
    cannot_link=None,
    p=1,
    prob_method="threshold",
    threshold=0.8,
    verbose=False,
    seed=42,
):
    """
    Refine a hard partition by forming exactly one linear-only group.

    Existing communities containing no designated nonlinear node are merged,
    using the largest only as a deterministic label anchor. Eligible
    required-together components containing a low-confidence node are added to
    that same group. Components containing a nonlinear node are never moved.

    Parameters
    ----------
    A : numpy.ndarray or scipy.sparse.spmatrix, shape (N, N)
        Graph adjacency/weight matrix.
    partition : numpy.ndarray or scipy.sparse.spmatrix, shape (N,) or (N, N)
        Predicted community labels or a 2D co-association matrix. Sparse input
        must be two-dimensional.
    nonlinear_nodes : sequence of int or None
        Designated nonlinear nodes. When omitted, place 
        low-confidence nodes in one new group without identifying
        existing pure-linear communities.
    worthy_edges : sequence of tuple or None
        Edges allowed to cross communities. When supplied, all other
        structural edges define required-together components.
    must_link, cannot_link : sequence of tuple or None
        Pairwise constraints preserved by the refinement.
    p : int
        Order of the norm. Defaults to the L1 norm.
    prob_method : str
        One of ``"threshold"``, ``"gaussian_mixture"``, or ``"DBSCAN"``.
    threshold : float
        Value below which a node is reassigned to the linear-only group.
    verbose : bool
        Controls the verbosity of the output.
    seed : int or None
        Random seed used by probability clustering.

    Returns
    -------
    ndarray of int, shape (N, N), or None
        Refined partition, or ``None`` when no feasible linear-only group can
        be formed.
    """
    matrix = A
    labels, hard_partition = _hard_partition_labels(partition, matrix.shape[0])
    probabilities = labels_to_probabilities(A, labels, p=p).toarray()
    proposed_labels = probability_to_integer_labels(
        probabilities,
        method=prob_method,
        threshold=threshold,
        verbose=verbose,
        seed=seed,
    )
    low_confidence = proposed_labels == -1
    if nonlinear_nodes is None:
        refined_labels = labels.copy()
        refined_labels[low_confidence] = int(labels.max(initial=-1)) + 1
        return partition_vector_to_2d_matrix(refined_labels)

    nonlinear = {int(node) for node in nonlinear_nodes}
    selected_nodes: set[int] = set()
    for component in required_together_components(
        matrix,
        worthy_edges=worthy_edges,
        must_link=must_link,
    ):
        if nonlinear.isdisjoint(component) and np.any(low_confidence[list(component)]):
            selected_nodes.update(component)
    return merge_linear_only_communities(
        hard_partition,
        tuple(nonlinear),
        additional_nodes=tuple(sorted(selected_nodes)),
        must_link=must_link,
        cannot_link=cannot_link,
    )


def refine_partition_with_cp(
    A: MatrixLike,
    partition: MatrixLike,
    *,
    must_link=None,
    nonlinear_nodes=None,
    cannot_link=None,
    cp_algorithm="SPEC",
    target="contracted",
    spectral_rank=1,
    prob_method="gaussian_mixture",
    threshold=0.8,
    verbose=False,
    seed=42,
    max_dense_working_bytes=DEFAULT_MAX_DENSE_WORKING_BYTES,
):
    """
    Refine a partition by detecting and merging the linear-only periphery.

    Existing assignments on the nonlinear/core side are preserved. All nodes
    in the complementary periphery are assigned to one shared linear-only
    community. In the intended NLBNP use, ``nonlinear_nodes`` identifies the
    nonlinear detection block and validates that it lies on the core side.

    Parameters
    ----------
    A : numpy.ndarray or scipy.sparse.spmatrix, shape (N, N)
        Graph adjacency or weight matrix. Sparse input crosses a guarded dense
        core-periphery boundary.
    partition : numpy.ndarray or scipy.sparse.spmatrix, shape (N,) or (N, N)
        Predicted community labels or a 2D co-association matrix. Sparse input
        must be two-dimensional.
    must_link : list[tuple[int, int]] or None
        Node pairs constrained to one core-periphery block.
    nonlinear_nodes : list[int] | None
        Designated nonlinear nodes constrained to one binary detection side.
    cannot_link : list[tuple[int, int]] or None
        Pairs that must remain separated after merging linear-only groups.
    cp_algorithm : str
        Core periphery algorithm to be used. Should be one of:
        ``"SPEC"``: Continuous spectral core periphery detection
        ``"GA"``: Continuous genetic search for BE objective
        ``"KL"``: Continuous Kernighan-Lin algorithm
    target : {"contracted", "original"}
        Space containing the core-periphery structure of interest.
    spectral_rank : {1, 2}
        Spectral approximation rank when ``cp_algorithm="SPEC"``.
    prob_method : str
        One of ``"threshold"``, ``"gaussian_mixture"``, or ``"DBSCAN"``.
    threshold : float
        Value used when converting continuous core scores to binary roles.
    verbose : bool
        Controls the verbosity of the output.
    seed : int | None
        Random seed value.
    max_dense_working_bytes : int or None, default=536870912
        Maximum operation-specific dense working-set estimate at the
        core-periphery boundary. ``None`` disables the guard.

    Returns
    -------
    ndarray of int, shape (N, N) or None
        Refined partition, or ``None`` when the required merge violates a
        pairwise constraint.
    """
    partition_labels, hard_partition = _hard_partition_labels(
        partition,
        A.shape[0],
    )

    dense_adjacency = checked_to_dense(
        A,
        working_arrays=4.0,
        max_dense_working_bytes=max_dense_working_bytes,
        operation="core-periphery cardinality refinement",
    )

    cp_result = _detect_core_periphery(
        dense_adjacency,
        must_link=must_link,
        must_group=nonlinear_nodes,
        algorithm=cp_algorithm,
        target=target,
        spectral_rank=spectral_rank,
        prob_method=prob_method,
        threshold=threshold,
        verbose=verbose,
        seed=seed,
    )
    if cp_result.node_labels is None:
        raise RuntimeError("Core-periphery detection did not return binary node labels.")
    cp_labels = cp_result.node_labels
    linear_only_mask = _nlbnp_linear_only_mask(
        cp_labels,
        nonlinear_nodes=nonlinear_nodes,
    )
    if nonlinear_nodes is None:
        refined_partition = np.where(linear_only_mask, -1, partition_labels)
        return partition_vector_to_2d_matrix(refined_partition)
    return merge_linear_only_communities(
        hard_partition,
        nonlinear_nodes,
        additional_nodes=np.flatnonzero(linear_only_mask),
        must_link=must_link,
        cannot_link=cannot_link,
    )
