"""Structural helpers for the NLBNP linear-only community."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from asunder.base.utils.graph import (
    normalize_node_pairs,
    partition_matrix_to_vector,
    partition_satisfies_pairwise_constraints,
    partition_vector_to_2d_matrix,
    validate_partition_matrix,
)
from asunder.base.utils.matrix import (
    DEFAULT_MAX_DENSE_WORKING_BYTES,
    is_symmetric,
    matrix_scalar,
    normalize_adjacency,
    structural_edge_pairs,
)
from asunder.types import MatrixLike


def _validate_node_indices(nodes: Sequence[int] | None, n_nodes: int, *, name: str) -> tuple[int, ...]:
    values = () if nodes is None else nodes
    normalized = tuple(dict.fromkeys(int(node) for node in values))
    if any(node < 0 or node >= n_nodes for node in normalized):
        raise ValueError(f"{name} contains a node index outside 0..{n_nodes - 1}.")
    return normalized


def structural_edges(A: MatrixLike) -> tuple[tuple[int, int], ...]:
    """Return nonzero, off-diagonal undirected edges in deterministic order.

    Parameters
    ----------
    A : numpy.ndarray or scipy.sparse.spmatrix, shape (N, N)
        Symmetric adjacency matrix.

    Returns
    -------
    tuple of tuple of int
        Structural edges represented once as ``(source, target)`` pairs.

    Raises
    ------
    ValueError
        If ``A`` is not square and symmetric.
    """

    matrix = normalize_adjacency(A)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("A must be a square adjacency matrix.")
    if not is_symmetric(matrix, atol=1e-10):
        raise ValueError("A must be symmetric.")
    return structural_edge_pairs(matrix)


def unworthy_edges(
    A: MatrixLike,
    worthy_edges: Sequence[tuple[int, int]],
) -> tuple[tuple[int, int], ...]:
    """Return structural edges not explicitly allowed to cross communities.

    Parameters
    ----------
    A : numpy.ndarray or scipy.sparse.spmatrix, shape (N, N)
        Symmetric adjacency matrix.
    worthy_edges : sequence of tuple of int
        Nonempty structural edges allowed to cross communities.

    Returns
    -------
    tuple of tuple of int
        Structural edges that must remain within a community.

    Raises
    ------
    ValueError
        If no worthy edge is supplied or a supplied pair is not a structural
        edge in ``A``.
    """

    structural = structural_edges(A)
    n_nodes = A.shape[0]
    worthy = tuple(
        normalize_node_pairs(
            worthy_edges, n_nodes, relation_name="worthy-edge", reject_self=True
        )
    )
    if not worthy:
        raise ValueError("NLBNP requires at least one worthy structural edge.")
    non_structural = tuple(sorted(set(worthy).difference(structural)))
    if non_structural:
        raise ValueError(
            "worthy_edges contains pairs that are not nonzero structural edges "
            f"in A: {non_structural}."
        )
    worthy_set = set(worthy)
    return tuple(edge for edge in structural if edge not in worthy_set)


def _components_from_edges(
    n_nodes: int,
    edges: Sequence[tuple[int, int]],
) -> tuple[tuple[int, ...], ...]:
    parent = np.arange(n_nodes, dtype=int)

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = int(parent[node])
        return node

    def union(source: int, target: int) -> None:
        source_root = find(source)
        target_root = find(target)
        if source_root != target_root:
            parent[target_root] = source_root

    for source, target in edges:
        union(int(source), int(target))

    groups: dict[int, list[int]] = {}
    for node in range(n_nodes):
        groups.setdefault(find(node), []).append(node)
    return tuple(sorted((tuple(nodes) for nodes in groups.values()), key=lambda nodes: nodes[0]))


def required_together_components(
    A: MatrixLike,
    *,
    worthy_edges: Sequence[tuple[int, int]] | None = None,
    must_link: Sequence[tuple[int, int]] | None = None,
) -> tuple[tuple[int, ...], ...]:
    """Build components induced by active edge rules and explicit must-links."""

    n_nodes = A.shape[0]
    links = normalize_node_pairs(
        must_link,
        n_nodes,
        relation_name="must-link",
    )
    edge_links = () if worthy_edges is None else unworthy_edges(A, worthy_edges)
    return _components_from_edges(n_nodes, [*edge_links, *links])


@dataclass(frozen=True)
class LinearGroupFeasibility:
    """Result of the maximum feasible linear-only-set calculation.

    ``y`` marks the maximum eligible node set and ``K_max`` is its size.
    ``components`` records the required-together components used in the
    calculation. The derived pair lists encode the exact set for Asunder's
    ordinary pairwise machinery. ``conflicting_cannot_link`` is nonempty when
    the requested relationships are infeasible.
    """

    y: np.ndarray
    eligible_nodes: tuple[int, ...]
    K_max: int
    components: tuple[tuple[int, ...], ...]
    unworthy_edges: tuple[tuple[int, int], ...]
    derived_must_link: tuple[tuple[int, int], ...]
    derived_cannot_link: tuple[tuple[int, int], ...]
    conflicting_cannot_link: tuple[tuple[int, int], ...]


def compute_max_feasible_linear_group(
    A: MatrixLike,
    *,
    worthy_edges: Sequence[tuple[int, int]],
    nonlinear_nodes: Sequence[int],
    must_link: Sequence[tuple[int, int]] | None = None,
    cannot_link: Sequence[tuple[int, int]] | None = None,
) -> LinearGroupFeasibility:
    """Compute the maximum eligible NLBNP linear-only node set.

    Components induced by unworthy edges and explicit must-links are atomic.
    Every component without a nonlinear node is eligible.  The returned
    pairwise constraints encode the union of all eligible components using a
    single deterministic representative.

    Parameters
    ----------
    A : numpy.ndarray or scipy.sparse.spmatrix, shape (N, N)
        Symmetric constraint-graph adjacency matrix.
    worthy_edges : sequence of tuple
        Nonempty structural edges allowed to cross final communities.
    nonlinear_nodes : sequence of int
        Node indices governed by nonlinear constraints.
    must_link, cannot_link : sequence of tuple, optional
        User-supplied pairwise constraints.

    Returns
    -------
    LinearGroupFeasibility
        Maximum-set membership, cardinality, components, derived pairwise
        constraints, and any cannot-link conflicts. This helper reports
        conflicts in its result; the top-level workflow converts them into a
        structured infeasible decomposition result.
    """

    matrix = normalize_adjacency(A)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("A must be a square adjacency matrix.")
    n_nodes = matrix.shape[0]
    nonlinear = _validate_node_indices(
        nonlinear_nodes,
        n_nodes,
        name="nonlinear_nodes",
    )
    if not nonlinear:
        raise ValueError("nonlinear_nodes must contain at least one node.")

    must_links = normalize_node_pairs(
        must_link,
        n_nodes,
        relation_name="must-link",
    )
    cannot_links = normalize_node_pairs(
        cannot_link,
        n_nodes,
        relation_name="cannot-link",
        reject_self=True,
    )
    edge_links = unworthy_edges(matrix, worthy_edges)
    components = _components_from_edges(n_nodes, [*edge_links, *must_links])
    nonlinear_set = set(nonlinear)
    eligible = tuple(
        node
        for component in components
        if nonlinear_set.isdisjoint(component)
        for node in component
    )
    eligible_set = set(eligible)
    y = np.fromiter(
        (node in eligible_set for node in range(n_nodes)),
        dtype=bool,
        count=n_nodes,
    )
    y.setflags(write=False)

    if eligible:
        anchor = eligible[0]
        derived_must_link = tuple((anchor, node) for node in eligible[1:])
        derived_cannot_link = tuple(
            (anchor, node) for node in range(n_nodes) if node not in eligible_set
        )
    else:
        derived_must_link = ()
        derived_cannot_link = ()

    component_by_node = np.empty(n_nodes, dtype=int)
    for component_index, component in enumerate(components):
        component_by_node[list(component)] = component_index
    conflicts = tuple(
        pair
        for pair in cannot_links
        if (
            pair[0] in eligible_set and pair[1] in eligible_set
        )
        or component_by_node[pair[0]] == component_by_node[pair[1]]
    )

    return LinearGroupFeasibility(
        y=y,
        eligible_nodes=eligible,
        K_max=len(eligible),
        components=components,
        unworthy_edges=edge_links,
        derived_must_link=derived_must_link,
        derived_cannot_link=derived_cannot_link,
        conflicting_cannot_link=conflicts,
    )


def linear_only_communities(
    partition: MatrixLike,
    nonlinear_nodes: Sequence[int],
    *,
    max_dense_working_bytes=DEFAULT_MAX_DENSE_WORKING_BYTES,
) -> tuple[tuple[int, ...], ...]:
    """Return nonempty communities containing no designated nonlinear node.

    Parameters
    ----------
    partition : numpy.ndarray or scipy.sparse.spmatrix
        Square hard co-association matrix.
    nonlinear_nodes : sequence of int
        Designated nonlinear-node indices.
    max_dense_working_bytes : int or None, default=536870912
        Dense validation workspace limit; ``None`` disables it.

    Returns
    -------
    tuple of tuple of int
        Original-node indices of each linear-only community.
    """

    matrix = validate_partition_matrix(partition, name="partition", max_dense_working_bytes=max_dense_working_bytes)
    nonlinear = set(
        _validate_node_indices(
            nonlinear_nodes,
            matrix.shape[0],
            name="nonlinear_nodes",
        )
    )
    labels = partition_matrix_to_vector(matrix)
    communities = []
    for label in np.unique(labels):
        members = tuple(int(node) for node in np.flatnonzero(labels == label))
        if nonlinear.isdisjoint(members):
            communities.append(members)
    return tuple(communities)


def partition_satisfies_edge_constraints(
    A: MatrixLike,
    partition: MatrixLike,
    worthy_edges: Sequence[tuple[int, int]] | None,
) -> bool:
    """Check every active unworthy-edge co-assignment constraint."""

    if worthy_edges is None:
        return True
    return all(
        matrix_scalar(partition, source, target) == 1
        for source, target in unworthy_edges(A, worthy_edges)
    )


def partition_satisfies_nlbnp_cardinality(
    partition: MatrixLike,
    nonlinear_nodes: Sequence[int],
    *,
    expected_nodes: Sequence[int] | None = None,
    max_dense_working_bytes=DEFAULT_MAX_DENSE_WORKING_BYTES,
) -> bool:
    """Check that exactly one linear-only community exists.

    Parameters
    ----------
    partition : numpy.ndarray or scipy.sparse.spmatrix
        Square hard co-association matrix.
    nonlinear_nodes : sequence of int
        Designated nonlinear-node indices.
    expected_nodes : sequence of int or None, default=None
        Required membership of the linear-only group, when supplied.
    max_dense_working_bytes : int or None, default=536870912
        Dense validation workspace limit; ``None`` disables it.

    Returns
    -------
    bool
        Whether exactly one qualifying group with the required members exists.
    """

    communities = linear_only_communities(partition, nonlinear_nodes, max_dense_working_bytes=max_dense_working_bytes)
    if len(communities) != 1:
        return False
    return expected_nodes is None or set(communities[0]) == set(expected_nodes)


def merge_linear_only_communities(
    partition: MatrixLike,
    nonlinear_nodes: Sequence[int],
    *,
    additional_nodes: Sequence[int] | None = None,
    must_link: Sequence[tuple[int, int]] | None = None,
    cannot_link: Sequence[tuple[int, int]] | None = None,
    max_dense_working_bytes=DEFAULT_MAX_DENSE_WORKING_BYTES,
) -> np.ndarray | None:
    """Merge existing pure-linear communities and additional eligible nodes.

    Parameters
    ----------
    partition : numpy.ndarray or scipy.sparse.spmatrix
        Square hard co-association matrix.
    nonlinear_nodes : sequence of int
        Designated nonlinear-node indices.
    additional_nodes : sequence of int or None, default=None
        Eligible nodes to join the merged linear-only group.
    must_link, cannot_link : sequence of tuple of int or None, default=None
        Pairwise constraints the merged partition must preserve.
    max_dense_working_bytes : int or None, default=536870912
        Dense validation and output-construction limit; ``None`` disables it.

    Returns
    -------
    numpy.ndarray or None
        Dense Boolean partition, or ``None`` if the merge is infeasible.
    """

    matrix = validate_partition_matrix(partition, name="partition", max_dense_working_bytes=max_dense_working_bytes)
    labels = partition_matrix_to_vector(matrix)
    nonlinear = set(
        _validate_node_indices(
            nonlinear_nodes,
            matrix.shape[0],
            name="nonlinear_nodes",
        )
    )
    pure_communities = linear_only_communities(matrix, tuple(nonlinear), max_dense_working_bytes=max_dense_working_bytes)
    target_nodes = {
        node for community in pure_communities for node in community
    }
    target_nodes.update(
        _validate_node_indices(
            additional_nodes,
            matrix.shape[0],
            name="additional_nodes",
        )
    )
    if not target_nodes or not nonlinear.isdisjoint(target_nodes):
        return None

    if pure_communities:
        anchor_community = max(
            pure_communities,
            key=lambda community: (len(community), -community[0]),
        )
        target_label = labels[anchor_community[0]]
    else:
        target_label = int(labels.max(initial=-1)) + 1

    refined_labels = labels.copy()
    refined_labels[list(sorted(target_nodes))] = target_label
    refined = partition_vector_to_2d_matrix(refined_labels, max_dense_working_bytes=max_dense_working_bytes)
    if not partition_satisfies_pairwise_constraints(
        refined,
        must_link=must_link,
        cannot_link=cannot_link,
    ):
        return None
    return refined
