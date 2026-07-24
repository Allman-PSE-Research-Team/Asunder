"""Graph and partition utilities."""

from __future__ import annotations

from typing import Any

import networkx as nx
import numpy as np


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

def group_nodes_by_community(z_matrix):
    """
    Extract community map and node groups from a partition matrix.
    
    Parameters
    ----------
    z_matrix : ndarray of int, shape (N, N)
        Partition mat.
    
    Returns
    -------
    community_map: dict
        Map from node index to community index
    
    communities: list
        List of different communities
    """
    communities = []
    community_map = {}
    n_community = 1
    seen = set()
    for var in range(z_matrix.shape[0]):
        if var in seen:
            continue
        group = frozenset[Any](np.where(z_matrix[var] == 1)[0])
        communities.append(group)
        for i in group:
            if i not in community_map:
                community_map[i] = n_community
        seen |= group
        n_community += 1
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
def partition_vector_to_2d_matrix(partition):
    """
    Convert a 1D label vector to a binary co-association matrix.
    
    Parameters
    ----------
    partition : array of int, shape (n,)
        1D label vector.
    
    Returns
    -------
    z : ndarray of int, shape (n, n)
        Binary co-association matrix.
    """
    try:
        z = (partition[:, None] == partition[None, :]).astype(int)
    except Exception:
        n = len(partition)
        z = np.zeros((n, n), dtype=int)

        for i in range(n):
            for j in range(n):
                if partition[i] == partition[j]:
                    z[i, j] = 1
    return z

def partition_matrix_to_vector(Z):
    """
    Convert a symmetric 2D partition matrix Z into a 1D membership vector.
    
    Parameters
    ----------
    Z : ndarray of int, shape (n, n)
        Binary co-association matrix.
    
    Returns
    -------
    labels : array of int, shape (n,)
        1D label vector.
    """
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


def contract_partition_matrix(partition, node2comp, *, atol=1e-8):
    """Convert a component-consistent partition to contracted dimensions.

    A warm-start partition is representable after contraction only when every
    original node in a component has the same relationship to every other
    component. In particular, all nodes within a component must be together.

    Parameters
    ----------
    partition : ndarray
        Original ``(N, N)`` or already-contracted ``(C, C)`` co-association
        matrix.
    node2comp : ndarray of int, shape (N,)
        Mapping from original nodes to ``C`` contracted components.
    atol : float
        Absolute tolerance used for component-consistency checks.

    Returns
    -------
    ndarray
        A copy of the partition with shape ``(C, C)``.

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
    matrix = np.asarray(partition)
    if matrix.shape == (n_components, n_components):
        contracted = matrix.copy()
    elif matrix.shape == (n_nodes, n_nodes):
        contracted = np.empty(
            (n_components, n_components),
            dtype=matrix.dtype,
        )
        component_nodes = [
            np.flatnonzero(mapping == component)
            for component in range(n_components)
        ]
        for source, source_nodes in enumerate(component_nodes):
            for target, target_nodes in enumerate(component_nodes):
                block = matrix[np.ix_(source_nodes, target_nodes)]
                representative = block.flat[0]
                if not np.allclose(block, representative, atol=atol, rtol=0):
                    raise ValueError(
                        "Warm-start partition is not constant across "
                        f"contracted components {source} and {target}."
                    )
                contracted[source, target] = representative
    else:
        raise ValueError(
            "Warm-start partition must have shape "
            f"{(n_nodes, n_nodes)} or {(n_components, n_components)}, "
            f"not {matrix.shape}."
        )

    if not np.allclose(contracted, contracted.T, atol=atol, rtol=0):
        raise ValueError("Warm-start partition must be symmetric.")
    if not np.allclose(
        np.diag(contracted),
        np.ones(n_components),
        atol=atol,
        rtol=0,
    ):
        raise ValueError(
            "Warm-start partition must keep every contracted component "
            "together and have a unit diagonal."
        )
    return contracted


def contract_adj_matrix_new(
    A,
    worthy_edges=None,
    must_link=None,
    keep_self_loops=True,
    degree_preserving=True,   # if True -> diag = 2 * intra_sum, else diag = intra_sum
):
    """
    Contract A according to connected components induced by rule-graph (G_ml),
    and optionally keep self-loops to encode intra-block connectivity strength.

    Parameters
    ----------
    A : ndarray, shape (n, n)
        Graph adjacency (assumed symmetric, no self-loops.)
    worthy_edges : set[tuple[int,int]] or None
        The edges that can connect different communities.
    must_link : iterable[tuple[int,int]] or None
        Extra links to force-merge nodes into the same component.
    keep_self_loops : bool
        If True, store intra-community weight on the diagonal of the coarse matrix.
    degree_preserving : bool
        If True, we set diag(C,C) = 2 * intra_sum_C so that
        vol(supernode C) = sum_{i in C} deg(i). If False, diag = intra_sum_C.

    Returns
    -------
    A_sup : ndarray, shape (k, k)
        Contracted adjacency.
    node2comp : np.ndarray[int]
        Mapping from original node to supernode id.
    """
    A = np.asarray(A)
    n = A.shape[0]

    # build the merge graph G_ml that defines which nodes are contracted
    edges = np.argwhere(np.triu(A) != 0).tolist()
    G_ml = nx.Graph()
    G_ml.add_nodes_from(range(n))

    if worthy_edges:
        wset = set(worthy_edges)
        for (i, j) in edges:
            if i == j:
                continue
            if (i, j) in wset or (j, i) in wset:
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

    # build contracted adjacency while tracking intra weights
    A_sup = np.zeros((num_super, num_super), dtype=A.dtype)
    intra_sum = np.zeros(num_super, dtype=float)

    for i, j in edges:
        wij = A[i, j]
        ci, cj = node2comp[i], node2comp[j]
        if ci == cj:
            intra_sum[ci] += wij
        else:
            A_sup[ci, cj] += wij
            A_sup[cj, ci] += wij

    if keep_self_loops:
        diag_vals = 2 * intra_sum if degree_preserving else intra_sum
        A_sup[np.arange(num_super), np.arange(num_super)] = diag_vals

    return (A_sup, node2comp)

def expand_z_matrix(
        z,
        node2comp,
        dim=2
):
    """
    Expand a supernode-level partition back to original node dimension.
    
    Parameters
    ----------
    z : np.ndarray[int]
        contracted 1D label vector or binary co-association matrix.
    node2comp : np.ndarray[int]
        Mapping from original node to supernode id.
    dim : int
        Dimension of input array (`1` or `2`).
    
    Returns
    -------
    z_full : np.ndarray[int]
        Expanded 1D label vector or binary co-association matrix.
    """
    # returns z if node2comp or z is empty.
    if node2comp is None or z is None:
        return z
    n = len(node2comp)
    if dim == 2:
        comp_idx = np.array([node2comp[i] for i in range(n)])  # shape = (n,)
        z_full = z[np.ix_(comp_idx, comp_idx)]  # shape = (n, n)
    elif dim ==  1:
        z_full = np.array([z[node2comp[i]] for i in range(n)])
    return z_full

def z_hamming_upper(Z1: np.ndarray, Z2: np.ndarray) -> float:
    """
    Compute Hamming distance on strict upper-triangle partition entries.
    
    Parameters
    ----------
    Z1 : np.ndarray
        Binary co-association partition matrix.
    Z2 : np.ndarray
        Binary co-association partition matrix.
    
    Returns
    -------
    float
        Computed hamming distance.
    """
    # TODO: for a cheaper test (when N is large), sample a fixed set of upper-tri pairs once and reuse it.
    n = Z1.shape[0]
    iu = np.triu_indices(n, k=1)
    return float(np.mean(Z1[iu] != Z2[iu]))

def sufficiently_different(Z_new: np.ndarray, Z_pool: list, dist_min: float) -> bool:
    """
    Check whether a candidate partition differs sufficiently from a pool.
    
    Parameters
    ----------
    Z_new : np.ndarray
        Candidate partition.
    Z_pool : list
        Pool of existing partitions.
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
    return np.equal.outer(labels, labels).astype(int)
