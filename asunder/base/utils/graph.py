"""Graph and partition utilities."""

from __future__ import annotations

from typing import Any

import networkx as nx
import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components

from asunder.base.utils.matrix import (
    DEFAULT_MAX_DENSE_WORKING_BYTES,
    DEFAULT_SPARSE_COLUMN_DENSITY_THRESHOLD,
    checked_to_dense,
    choose_column_storage,
    ensure_dense_working_set,
    is_symmetric,
    matrix_scalar,
    normalize_adjacency,
    structural_edge_pairs,
    validate_storage_options,
)
from asunder.types import MatrixLike


def get_optimization_params_from_graph(n=None, graph_edges=None, G=None):
    """
    Build adjacency, degree vector, and volume from graph input.
    
    Parameters
    ----------
    n : int
        Number of nodes.
    graph_edges : list[sequence[int]]
        List of graph edges.
    G : nx.Graph
        NetworkX graph.
    
    Returns
    -------
    adjacency_matrix : np.ndarray of int | float, shape (n, n)
        Adjacency / weight matrix.
    degree_matrix : np.ndarray of int | float, shape (n,)
        Degree-like vector; defaults to row sums of the symmetrized adjacency.
    m : float
        Twice the total weight in the graph.
    """
    if G is not None:
        adjacency_matrix = nx.to_numpy_array(G)
    else:
        adjacency_matrix = np.zeros(shape=(n, n))
        for source, sink in graph_edges:
            adjacency_matrix[source, sink] = 1
            adjacency_matrix[sink, source] = 1

    degree_matrix = adjacency_matrix.sum(axis=0)

    m = np.sum(degree_matrix)

    return adjacency_matrix, degree_matrix, m

def group_nodes_by_community(z_matrix: MatrixLike):
    """
    Extract community map and node groups from a partition matrix.
    
    Parameters
    ----------
    z_matrix : numpy.ndarray or scipy.sparse.csr_matrix, shape (N, N)
        Binary co-association matrix.
    
    Returns
    -------
    community_map: dict
        Map from node index to community index
    
    communities: list
        List of different communities
    """
    labels = partition_matrix_to_vector(z_matrix)
    communities = []
    community_map = {}
    for n_community, label in enumerate(np.unique(labels), start=1):
        group = frozenset[Any](np.flatnonzero(labels == label))
        communities.append(group)
        community_map.update({int(node): n_community for node in group})
    return community_map, communities

def map_community_labels(community_map, label_map):
    """
    Remap integer community ids to external node labels.
    
    Parameters
    ----------
    community_map: dict
        Map from node index to community index
    label_map : Any
        Map from node index to node label in Graph.
    
    Returns
    -------
    dict[str, int]
        map from node label to community index
    """
    return {label_map[idx]: community for idx, community in community_map.items()}
def partition_vector_to_2d_matrix(
    partition,
    *,
    storage="dense",
    sparse_column_density_threshold=DEFAULT_SPARSE_COLUMN_DENSITY_THRESHOLD,
    max_dense_working_bytes=DEFAULT_MAX_DENSE_WORKING_BYTES,
) -> MatrixLike:
    """
    Convert a 1D label vector to a binary co-association matrix.
    
    Parameters
    ----------
    partition : array of int, shape (n,)
        1D label vector.
    storage : {"dense", "csr", "auto"}, default="dense"
        Physical representation. Automatic selection uses CSR only when its
        density meets the threshold and its estimated storage is smaller.
    sparse_column_density_threshold : float, default=0.20
        Maximum density at which automatic selection uses CSR storage.
    max_dense_working_bytes : int or None, default=536870912
        Maximum estimated package-created dense working set. ``None`` disables
        the guard.
    
    Returns
    -------
    z : ndarray of bool or scipy.sparse.csr_matrix, shape (n, n)
        Binary co-association matrix.
    """
    validate_storage_options(storage, sparse_column_density_threshold, max_dense_working_bytes)
    threshold = float(sparse_column_density_threshold)
    if not np.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("sparse_column_density_threshold must be between 0 and 1.")

    labels = np.asarray(partition)
    if labels.ndim != 1:
        raise ValueError("partition must be a one-dimensional label vector.")
    n_nodes = labels.size
    if n_nodes:
        _, inverse, counts = np.unique(labels, return_inverse=True, return_counts=True)
        nnz = sum(int(count) ** 2 for count in counts)
        density = nnz / float(n_nodes * n_nodes)
    else:
        inverse = counts = np.empty(0, dtype=int)
        nnz = 0
        density = 0.0
    index_dtype = np.int32 if max(n_nodes, nnz) <= np.iinfo(np.int32).max else np.int64
    csr_bytes = nnz * (1 + np.dtype(index_dtype).itemsize) + (n_nodes + 1) * np.dtype(index_dtype).itemsize
    resolved = storage
    if storage == "auto":
        resolved = "csr" if density <= threshold and csr_bytes < n_nodes * n_nodes else "dense"

    if resolved == "dense":
        ensure_dense_working_set(
            (n_nodes, n_nodes),
            dtype=np.bool_,
            max_dense_working_bytes=max_dense_working_bytes,
            operation="dense partition construction",
        )
        return np.equal.outer(labels, labels)

    if not n_nodes:
        return sparse.csr_matrix((0, 0), dtype=bool)
    # Fill final CSR buffers directly; auxiliary grouping arrays are O(N).
    indptr = np.empty(n_nodes + 1, dtype=index_dtype)
    indptr[0] = 0
    np.cumsum(counts[inverse], out=indptr[1:])
    indices = np.empty(nnz, dtype=index_dtype)
    ordered = np.argsort(inverse, kind="stable") # Time: O(NlogN)
    boundaries = np.concatenate(([0], np.cumsum(counts)))
    for row, group in enumerate(inverse):
        indices[indptr[row]:indptr[row + 1]] = ordered[boundaries[group]:boundaries[group + 1]]
    return sparse.csr_matrix(
        (np.ones(nnz, dtype=bool), indices, indptr),
        shape=(n_nodes, n_nodes),
        copy=False,
    )

def partition_matrix_to_vector(Z: MatrixLike) -> np.ndarray:
    """
    Convert a symmetric 2D partition matrix Z into a 1D membership vector.
    
    Parameters
    ----------
    Z : numpy.ndarray or scipy.sparse.spmatrix, shape (n, n)
        Binary co-association matrix.
    
    Returns
    -------
    labels : array of int, shape (n,)
        1D label vector.
    """
    if sparse.issparse(Z):
        matrix = sparse.csr_matrix(Z, dtype=bool)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError("Z must be a square matrix.")
        _, labels = connected_components(matrix, directed=False)
        return labels.astype(int, copy=False)

    Z = np.asarray(Z)
    N = Z.shape[0]
    labels = -np.ones(N, dtype=int)
    current_label = 0

    for i in range(N):
        if labels[i] == -1:
            labels[i] = current_label
            for j in range(i+1, N):
                if Z[i, j] == 1:
                    labels[j] = current_label
            current_label += 1
    return labels


def normalize_node_pairs(pairs, n_nodes, *, relation_name="node-pair", reject_self=False):
    """Validate, normalize, and deduplicate node-index pairs.

    Pairs are returned in sorted endpoint order.  Self-pairs are discarded for
    reflexive relations such as must-links and rejected for irreflexive
    relations such as cannot-links.
    """
    normalized = set()
    for pair in pairs or []:
        if len(pair) != 2:
            raise ValueError(f"Each {relation_name} entry must contain two nodes.")
        source, target = int(pair[0]), int(pair[1])
        if not (0 <= source < n_nodes and 0 <= target < n_nodes):
            raise ValueError(
                f"{relation_name} pair {(source, target)} contains a node "
                f"outside 0..{n_nodes - 1}."
            )
        if source == target:
            if reject_self:
                raise ValueError(
                    f"{relation_name} pair {(source, target)} cannot contain "
                    "the same node twice."
                )
            continue
        normalized.add(tuple(sorted((source, target))))
    return sorted(normalized)


def validate_partition_matrix(
    partition: MatrixLike,
    n_nodes=None,
    *,
    name="partition",
    atol=1e-8,
    storage="preserve",
    sparse_column_density_threshold=DEFAULT_SPARSE_COLUMN_DENSITY_THRESHOLD,
    max_dense_working_bytes=DEFAULT_MAX_DENSE_WORKING_BYTES,
) -> MatrixLike:
    """Return a validated binary co-association matrix.

    Besides shape, symmetry, and unit-diagonal checks, this verifies that the
    matrix represents an equivalence relation.  The latter prevents malformed
    custom columns from silently entering a restricted master problem.

    Automatic storage preserves the input representation. An already Boolean
    dense input may be returned without copying and is never modified.
    """
    if storage not in {"preserve", "auto", "dense", "csr"}:
        raise ValueError("storage must be 'preserve', 'auto', 'dense', or 'csr'.")
    validate_storage_options(
        "auto" if storage == "preserve" else storage,
        sparse_column_density_threshold,
        max_dense_working_bytes,
    )
    input_sparse = sparse.issparse(partition)
    matrix = (
        sparse.csr_matrix(partition, copy=True)
        if input_sparse
        else np.asarray(partition)
    )
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"{name} must be a square matrix.")
    if n_nodes is not None and matrix.shape != (int(n_nodes), int(n_nodes)):
        raise ValueError(
            f"{name} must have shape {(int(n_nodes), int(n_nodes))}, "
            f"not {matrix.shape}."
        )
    if input_sparse:
        matrix.sum_duplicates()
        matrix.eliminate_zeros()
        matrix.sort_indices()
        if not np.all(np.isfinite(matrix.data)):
            raise ValueError(f"{name} contains NaN or infinity.")
        rounded_data = np.rint(matrix.data)
        if not np.allclose(matrix.data, rounded_data, atol=atol, rtol=0) or np.any(
            (rounded_data != 0) & (rounded_data != 1)
        ):
            raise ValueError(f"{name} must be binary.")
        matrix.data = rounded_data.astype(bool)
        matrix.eliminate_zeros()
        if (matrix != matrix.T).nnz:
            raise ValueError(f"{name} must be symmetric.")
        if not np.all(matrix.diagonal()):
            raise ValueError(f"{name} must have a unit diagonal.")
        labels = partition_matrix_to_vector(matrix)
        counts = np.bincount(labels) if labels.size else np.empty(0, dtype=int)
        if matrix.nnz != int(np.dot(counts, counts)):
            raise ValueError(f"{name} must define a transitive co-association relation.")
        canonical = matrix.astype(bool, copy=False)
    else:
        size = matrix.shape[0]
        needs_copy = matrix.dtype != np.bool_
        ensure_dense_working_set(
            (size,), dtype=float, working_arrays=8,
            extra_bytes=size * size if needs_copy else 0,
            max_dense_working_bytes=max_dense_working_bytes,
            operation=f"dense validation for {name}",
        )
        canonical = np.empty(matrix.shape, dtype=bool) if needs_copy else matrix
        for index, row in enumerate(matrix):
            try:
                finite = np.all(np.isfinite(row))
            except TypeError as exc:
                raise ValueError(f"{name} must contain numeric values.") from exc
            if not finite:
                raise ValueError(f"{name} contains NaN or infinity.")
            if not np.allclose(row, matrix[:, index], atol=atol, rtol=0):
                raise ValueError(f"{name} must be symmetric.")
            if not np.isclose(row[index], 1.0, atol=atol, rtol=0):
                raise ValueError(f"{name} must have a unit diagonal.")
            if needs_copy:
                rounded = np.rint(row)
                if not np.allclose(row, rounded, atol=atol, rtol=0) or np.any(
                    (rounded != 0) & (rounded != 1)
                ):
                    raise ValueError(f"{name} must be binary.")
                canonical[index] = rounded.astype(bool)
        labels = partition_matrix_to_vector(canonical)
        for index, row in enumerate(canonical):
            if not np.array_equal(row, labels == labels[index]):
                raise ValueError(f"{name} must define a transitive co-association relation.")

    if storage == "preserve":
        return canonical
    resolved = choose_column_storage(
        canonical,
        column_storage=storage,
        sparse_column_density_threshold=sparse_column_density_threshold,
    )
    if resolved == "csr":
        return sparse.csr_matrix(canonical, dtype=bool)
    if sparse.issparse(canonical):
        return checked_to_dense(
            canonical,
            dtype=bool,
            max_dense_working_bytes=max_dense_working_bytes,
            operation=f"dense storage for {name}",
        )
    return np.asarray(canonical, dtype=bool)


def partition_satisfies_pairwise_constraints(
    partition: MatrixLike,
    *,
    must_link=None,
    cannot_link=None,
    atol=1e-8,
):
    """Check hard must-link and cannot-link values in a partition matrix."""
    for source, target in must_link or []:
        if not np.isclose(
            matrix_scalar(partition, source, target), 1.0, atol=atol, rtol=0
        ):
            return False
    for source, target in cannot_link or []:
        if not np.isclose(
            matrix_scalar(partition, source, target), 0.0, atol=atol, rtol=0
        ):
            return False
    return True


def contract_node_pairs(
    pairs,
    node2comp,
    *,
    relation_name="node-pair",
    reject_internal=False,
):
    """Map original-node pairs to unique contracted-component pairs.

    Parameters
    ----------
    pairs : iterable[tuple[int, int]] or None
        Original-node pairs to map.
    node2comp : ndarray of int, shape (N,)
        Mapping from original node index to component index.
    relation_name : str
        Human-readable relation name used in validation errors.
    reject_internal : bool
        If ``True``, reject a pair whose endpoints contract into the same
        component. This is required for cannot-link constraints.

    Returns
    -------
    list[tuple[int, int]]
        Sorted, deduplicated component-level pairs.

    Raises
    ------
    ValueError
        If a pair contains an invalid node index or becomes an internal pair
        while ``reject_internal`` is enabled.
    """
    mapping = np.asarray(node2comp, dtype=int)
    if mapping.ndim != 1:
        raise ValueError("node2comp must be a one-dimensional array.")

    contracted = set()
    for pair in pairs or []:
        if len(pair) != 2:
            raise ValueError(f"Each {relation_name} entry must contain two nodes.")
        source, target = (int(pair[0]), int(pair[1]))
        if not (0 <= source < mapping.size and 0 <= target < mapping.size):
            raise ValueError(
                f"{relation_name} pair {(source, target)} contains a node "
                f"outside 0..{mapping.size - 1}."
            )
        component_pair = tuple(
            sorted((int(mapping[source]), int(mapping[target])))
        )
        if component_pair[0] == component_pair[1]:
            if reject_internal:
                raise ValueError(
                    f"{relation_name} pair {(source, target)} lies inside "
                    f"contracted component {component_pair[0]}."
                )
            continue
        contracted.add(component_pair)
    return sorted(contracted)


def contract_partition_matrix(
    partition: MatrixLike,
    node2comp: np.ndarray,
    *,
    atol: float = 1e-8,
    max_dense_working_bytes=DEFAULT_MAX_DENSE_WORKING_BYTES,
) -> MatrixLike:
    """Convert a component-consistent partition to contracted dimensions.

    A warm-start partition is representable after contraction only when every
    original node in a component has the same relationship to every other
    component. In particular, all nodes within a component must be together.

    Parameters
    ----------
    partition : numpy.ndarray or scipy.sparse.spmatrix
        Original ``(N, N)`` or already-contracted ``(C, C)`` co-association
        matrix.
    node2comp : ndarray of int, shape (N,)
        Mapping from original nodes to ``C`` contracted components.
    atol : float
        Absolute tolerance used for component-consistency checks.
    max_dense_working_bytes : int or None, default=536870912
        Limit for newly allocated dense validation and contraction workspaces.

    Returns
    -------
    numpy.ndarray or scipy.sparse.csr_matrix
        A copy of the partition with shape ``(C, C)``. Input storage is
        preserved, with sparse input normalized to CSR.

    Raises
    ------
    ValueError
        If the matrix has an incompatible shape, is asymmetric, lacks a unit
        diagonal, or separates nodes belonging to one contracted component.
    """
    mapping = np.asarray(node2comp, dtype=int)
    if mapping.ndim != 1 or (mapping.size and np.any(mapping < 0)):
        raise ValueError("node2comp must contain nonnegative component indices.")

    n_nodes = mapping.size
    n_components = int(mapping.max()) + 1 if n_nodes else 0
    input_sparse = sparse.issparse(partition)
    matrix = validate_partition_matrix(
        partition,
        name="Warm-start partition",
        atol=atol,
        max_dense_working_bytes=max_dense_working_bytes,
    )
    if matrix.shape == (n_components, n_components):
        if not input_sparse:
            ensure_dense_working_set(
                matrix.shape, dtype=matrix.dtype,
                max_dense_working_bytes=max_dense_working_bytes,
                operation="dense contracted partition copy",
            )
        contracted = matrix.copy()
    elif matrix.shape == (n_nodes, n_nodes):
        component_nodes = [
            np.flatnonzero(mapping == component)
            for component in range(n_components)
        ]
        labels = partition_matrix_to_vector(matrix)
        component_labels = np.empty(n_components, dtype=int)
        for component, nodes in enumerate(component_nodes):
            node_labels = labels[nodes]
            if np.any(node_labels != node_labels[0]):
                raise ValueError(
                    "Warm-start partition is not constant across "
                    f"contracted component {component}."
                )
            component_labels[component] = node_labels[0]
        contracted = partition_vector_to_2d_matrix(
            component_labels,
            storage="csr" if input_sparse else "dense",
            max_dense_working_bytes=max_dense_working_bytes,
        )
    else:
        raise ValueError(
            "Warm-start partition must have shape "
            f"{(n_nodes, n_nodes)} or {(n_components, n_components)}, "
            f"not {matrix.shape}."
        )

    return contracted


def contract_adj_matrix_new(
    A: MatrixLike,
    worthy_edges=None,
    must_link=None,
    keep_self_loops=True,
    degree_preserving=True,   # if True -> diag = 2 * intra_sum, else diag = intra_sum
    *,
    max_dense_working_bytes=DEFAULT_MAX_DENSE_WORKING_BYTES,
) -> tuple[MatrixLike, np.ndarray]:
    """
    Contract A according to connected components induced by rule-graph (G_ml),
    and optionally keep self-loops to encode intra-block connectivity strength.

    Parameters
    ----------
    A : numpy.ndarray or scipy.sparse.spmatrix, shape (n, n)
        Graph adjacency (assumed symmetric, no self-loops.)
    worthy_edges : set[tuple[int,int]] or None
        Edges that can connect different communities. ``None`` disables
        edge-induced contraction; an empty collection contracts every
        structural edge component.
    must_link : iterable[tuple[int,int]] or None
        Extra links to force-merge nodes into the same component.
    keep_self_loops : bool
        If True, store intra-community weight on the diagonal of the coarse matrix.
    degree_preserving : bool
        If True, we set diag(C,C) = 2 * intra_sum_C so that
        vol(supernode C) = sum_{i in C} deg(i). If False, diag = intra_sum_C.
    max_dense_working_bytes : int or None, default=536870912
        Limit for dense contraction workspaces. CSR contraction stays sparse.

    Returns
    -------
    A_sup : numpy.ndarray or scipy.sparse.csr_matrix, shape (k, k)
        Contracted adjacency. Sparse input produces CSR output.
    node2comp : np.ndarray[int]
        Mapping from original node to supernode id.
    """
    input_sparse = sparse.issparse(A)
    if not input_sparse:
        A = checked_to_dense(
            A, max_dense_working_bytes=max_dense_working_bytes,
            operation="dense adjacency contraction input",
        )
    A = normalize_adjacency(A)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be a square matrix.")
    finite = (
        np.all(np.isfinite(A.data)) if input_sparse
        else all(np.all(np.isfinite(row)) for row in A)
    )
    if not finite:
        raise ValueError("A must contain only finite values.")
    if not is_symmetric(A, atol=1e-10):
        raise ValueError("A must be symmetric.")
    n = A.shape[0]

    # build the merge graph G_ml that defines which nodes are contracted
    edges = structural_edge_pairs(A)
    G_ml = nx.Graph()
    G_ml.add_nodes_from(range(n))

    if worthy_edges is not None:
        wset = {tuple(sorted((int(i), int(j)))) for i, j in worthy_edges}
        for (i, j) in edges:
            if i == j:
                continue
            if tuple(sorted((int(i), int(j)))) in wset:
                pass
            else:
                # unworthy edges cannot connect items in different communities
                G_ml.add_edge(i, j)
    else:
        pass

    validated_must_links = contract_node_pairs(
        must_link,
        np.arange(n),
        relation_name="must-link",
    )
    G_ml.add_edges_from(validated_must_links)

    components = list(nx.connected_components(G_ml))
    num_super = len(components)

    # maps
    comp2nodes = [np.fromiter(sorted(c), dtype=int) for c in components]
    node2comp = np.empty(n, dtype=int)
    for cid, nodes in enumerate(comp2nodes):
        node2comp[nodes] = cid

    if input_sparse:
        membership = sparse.csr_matrix(
            (np.ones(n), (np.arange(n), node2comp)),
            shape=(n, num_super),
        )
    else:
        ensure_dense_working_set(
            (num_super, num_super), dtype=float,
            extra_bytes=2 * n * num_super * np.dtype(float).itemsize,
            max_dense_working_bytes=max_dense_working_bytes,
            operation="dense adjacency contraction",
        )
        membership = np.zeros((n, num_super), dtype=float)
        membership[np.arange(n), node2comp] = 1.0
    # P.T @ A @ P exactly preserves every component's summed strength,
    # including pre-existing diagonal mass.  Reconstructing from the upper
    # triangle would incorrectly double original self-loops.
    A_sup = membership.T @ A @ membership
    if input_sparse:
        A_sup = sparse.csr_matrix(A_sup)
    if not keep_self_loops:
        if input_sparse:
            A_sup.setdiag(0)
            A_sup.eliminate_zeros()
        else:
            np.fill_diagonal(A_sup, 0)
    elif not degree_preserving:
        original_diagonal = A.diagonal() if input_sparse else np.diag(A)
        diagonal_mass = np.bincount(
            node2comp,
            weights=original_diagonal,
            minlength=num_super,
        )
        current = A_sup.diagonal().copy() if input_sparse else np.diag(A_sup).copy()
        if input_sparse:
            A_sup.setdiag(0.5 * (current + diagonal_mass))
            A_sup.eliminate_zeros()
        else:
            np.fill_diagonal(A_sup, 0.5 * (current + diagonal_mass))

    return (A_sup, node2comp)

def expand_z_matrix(
    z: MatrixLike | np.ndarray | None,
    node2comp: np.ndarray | None,
    dim: int = 2,
    *,
    max_dense_working_bytes=DEFAULT_MAX_DENSE_WORKING_BYTES,
) -> MatrixLike | np.ndarray | None:
    """
    Expand a supernode-level partition back to original node dimension.
    
    Parameters
    ----------
    z : numpy.ndarray, scipy.sparse.csr_matrix, or None
        Contracted 1D label vector or binary co-association matrix.
    node2comp : np.ndarray[int]
        Mapping from original node to supernode id.
    dim : int
        Dimension of input array (`1` or `2`).
    max_dense_working_bytes : int or None, default=536870912
        Limit for a newly allocated dense original-size partition. CSR input
        stays sparse. ``None`` disables the guard.
    
    Returns
    -------
    z_full : numpy.ndarray, scipy.sparse.csr_matrix, or None
        Expanded 1D label vector or binary co-association matrix. Sparse
        matrix input produces CSR output.
    """
    # returns z if node2comp or z is empty.
    if node2comp is None or z is None:
        return z
    n = len(node2comp)
    if dim == 2:
        comp_idx = np.asarray(node2comp, dtype=int)
        if sparse.issparse(z):
            z_full = sparse.csr_matrix(z)[comp_idx, :][:, comp_idx]
        else:
            ensure_dense_working_set(
                (n, n), dtype=z.dtype,
                extra_bytes=comp_idx.nbytes,
                max_dense_working_bytes=max_dense_working_bytes,
                operation="dense original-size partition expansion",
            )
            z_full = z[np.ix_(comp_idx, comp_idx)]  # shape = (n, n)
    elif dim ==  1:
        z_full = np.array([z[node2comp[i]] for i in range(n)])
    return z_full

def z_hamming_upper(Z1: MatrixLike, Z2: MatrixLike) -> float:
    """
    Compute Hamming distance on strict upper-triangle partition entries.
    
    Parameters
    ----------
    Z1 : numpy.ndarray or scipy.sparse.spmatrix
        Binary co-association partition matrix.
    Z2 : numpy.ndarray or scipy.sparse.spmatrix
        Binary co-association partition matrix.
    
    Returns
    -------
    float
        Computed hamming distance.
    """
    # TODO: for a cheaper test (when N is large), sample a fixed set of upper-tri pairs once and reuse it.
    if Z1.shape != Z2.shape or len(Z1.shape) != 2 or Z1.shape[0] != Z1.shape[1]:
        raise ValueError("Partition matrices must be square and have equal shapes.")
    n = Z1.shape[0]
    pairs = n * (n - 1) // 2
    if pairs == 0:
        return 0.0
    if sparse.issparse(Z1) and sparse.issparse(Z2):
        left = sparse.csr_matrix(Z1, dtype=bool)
        right = sparse.csr_matrix(Z2, dtype=bool)
        mismatches = (left != right).nnz // 2
        return float(mismatches / pairs)
    # A mixed comparison uses at most one sparse row and one Boolean row of
    # scratch; never convert the entire dense operand into CSR.
    left = sparse.csr_matrix(Z1) if sparse.issparse(Z1) else np.asarray(Z1)
    right = sparse.csr_matrix(Z2) if sparse.issparse(Z2) else np.asarray(Z2)
    mismatches = 0
    for row in range(n - 1):
        lrow = left.getrow(row).toarray().ravel() if sparse.issparse(left) else left[row]
        rrow = right.getrow(row).toarray().ravel() if sparse.issparse(right) else right[row]
        mismatches += np.count_nonzero(lrow[row + 1:] != rrow[row + 1:])
    return float(mismatches / pairs)

def sufficiently_different(
    Z_new: MatrixLike,
    Z_pool: list[MatrixLike],
    dist_min: float,
) -> bool:
    """
    Check whether a candidate partition differs sufficiently from a pool.
    
    Parameters
    ----------
    Z_new : numpy.ndarray or scipy.sparse.spmatrix
        Candidate partition.
    Z_pool : list[numpy.ndarray or scipy.sparse.spmatrix]
        Pool of existing partitions; representations may be mixed.
    dist_min : float
        Distance threshold.
    
    Returns
    -------
    bool
        `True` if the candidate partition does not differ sufficiently from the pool. 
    """
    if not Z_pool:
        return True
    dmin = min(z_hamming_upper(Z_new, Z) for Z in Z_pool)
    return dmin >= dist_min


def proportions_to_partition(r: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """
    Convert per-node probabilities into a binary co-association matrix.
    
    Parameters
    ----------
    r : np.ndarray
        Per-node probabilities.
    threshold : float
        Threshold probability value.
    
    Returns
    -------
    ndarray of int, shape (n, n)
        Binary co-association matrix.
    """
    labels = (np.asarray(r) > threshold).astype(int)
    return np.equal.outer(labels, labels)
