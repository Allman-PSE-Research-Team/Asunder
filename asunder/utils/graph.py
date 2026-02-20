"""Graph and partition utilities."""

from __future__ import annotations

import networkx as nx
import numpy as np


def get_optimization_params_from_graph(n=None, graph_edges=None, G=None):
    """Build adjacency, degree vector, and volume from graph input."""
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
    """Extract community map and node groups from a partition matrix."""
    communities = []
    community_map = {}
    n_community = 1
    seen = set()
    for var in range(z_matrix.shape[0]):
        if var in seen:
            continue
        group = frozenset(np.where(z_matrix[var] == 1)[0])
        communities.append(group)
        for i in group:
            if i not in community_map:
                community_map[i] = n_community
        seen |= group
        n_community += 1
    return community_map, communities

def map_community_labels(community_map, label_map):
    """Remap integer community ids to external node labels."""
    return {label_map[idx]: community for idx, community in community_map.items()}
def partition_vector_to_2d_matrix(partition):
    """Convert a 1D label vector to a binary co-association matrix."""
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
    Z_ij = 1 if nodes i and j are in the same community, else 0.
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
def contract_adj_matrix_new(
    A,
    worthy_edges=None,
    must_link=None,
    keep_self_loops=True,
    return_stats=False,
    degree_preserving=True,   # if True -> diag = 2 * intra_sum, else diag = intra_sum
):
    """
    Contract A according to connected components induced by your rule-graph (G_ml),
    and optionally keep self-loops to encode intra-block connectivity strength.

    Parameters
    ----------
    A : (n, n) ndarray
        (Assumed symmetric, no self-loops.)
    worthy_edges : set[tuple[int,int]] or None
        The edges that can connect different communities.
    must_link : iterable[tuple[int,int]] or None
        Extra links to force-merge nodes into the same component.
    keep_self_loops : bool
        If True, store intra-community weight on the diagonal of the coarse matrix.
    return_stats : bool
        If True, return a dict of per-supernode stats.
    degree_preserving : bool
        If True, we set diag(C,C) = 2 * intra_sum_C so that
        vol(supernode C) = sum_{i in C} deg(i). If False, diag = intra_sum_C.

    Returns
    -------
    A_sup : (k, k) ndarray
        Contracted adjacency.
    node2comp : np.ndarray[int]
        Mapping from original node to supernode id.
    comp2nodes : list[np.ndarray]
        Reverse mapping: nodes in each supernode.
    stats : dict or None
        Per-supernode stats (only if return_stats=True).
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

    if must_link is not None:
        G_ml.add_edges_from(must_link or [])

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
    """Expand a supernode-level partition back to original node dimension."""
    # returns z if node2comp is empty.
    if node2comp is None:
        return z
    n = len(node2comp)
    if dim == 2:
        comp_idx = np.array([node2comp[i] for i in range(n)])  # shape = (n,)
        z_full = z[np.ix_(comp_idx, comp_idx)]  # shape = (n, n)
    elif dim ==  1:
        z_full = np.array([z[node2comp[i]] for i in range(n)])
    return z_full

def z_hamming_upper(Z1: np.ndarray, Z2: np.ndarray) -> float:
    """Compute Hamming distance on strict upper-triangle partition entries."""
    # for a cheaper test (when N is large), sample a fixed set of upper-tri pairs once and reuse it.
    n = Z1.shape[0]
    iu = np.triu_indices(n, k=1)
    return float(np.mean(Z1[iu] != Z2[iu]))

def sufficiently_different(Z_new: np.ndarray, Z_pool: list, dist_min: float) -> bool:
    """Check whether a candidate partition differs sufficiently from a pool."""
    if not Z_pool:
        return True
    dmin = min(z_hamming_upper(Z_new, Z) for Z in Z_pool)
    return dmin >= dist_min
def contract_adj_matrix_cp(
    A,
    unworthy_edges=None,
    nonlinear_nodes=None,
    keep_self_loops=True,
    return_stats=False,
    degree_preserving=True,   # if True -> diag = 2 * intra_sum, else diag = intra_sum
):
    """
    Contract A according to connected components induced by your rule-graph (G_ml),
    and optionally keep self-loops to encode intra-block connectivity strength.

    Parameters
    ----------
    A : (n, n) ndarray
        (Assumed symmetric, no self-loops.)
    unworthy_edges : set[tuple[int,int]] or None
        The edges that cannot connect different communities.
    nonlinear_nodes : set[int]
        Nonlinear nodes.
    keep_self_loops : bool
        If True, store intra-community weight on the diagonal of the coarse matrix.
    return_stats : bool
        If True, return a dict of per-supernode stats.
    degree_preserving : bool
        If True, we set diag(C,C) = 2 * intra_sum_C so that
        vol(supernode C) = sum_{i in C} deg(i). If False, diag = intra_sum_C.

    Returns
    -------
    A_sup : (k, k) ndarray
        Contracted adjacency.
    node2comp : np.ndarray[int]
        Mapping from original node to supernode id.
    comp2nodes : list[np.ndarray]
        Reverse mapping: nodes in each supernode.
    stats : dict or None
        Per-supernode stats (only if return_stats=True).
    """
    A = np.asarray(A)
    n = A.shape[0]

    # build the merge graph G_ml that defines which nodes are contracted
    edges = np.argwhere(np.triu(A) != 0).tolist()
    G_ml = nx.Graph()
    G_ml.add_nodes_from(range(n))

    if unworthy_edges is not None:
        G_ml.add_edges_from(unworthy_edges or [])

    # if you have nonlinear_nodes links too, add those:
    if nonlinear_nodes is not None:
        nonlinear_nodes = list(nonlinear_nodes)
        if nonlinear_nodes:
            rep = nonlinear_nodes[0]
            for v in nonlinear_nodes[1:]:
                G_ml.add_edge(v, rep)

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

    # stats
    stats = None
    if return_stats:
        sizes = np.array([len(nodes) for nodes in comp2nodes])
        # density computed on undirected simple graph assumption
        density = np.zeros(num_super)
        for c in range(num_super):
            s = sizes[c]
            density[c] = 0.0 if s <= 1 else (2 * intra_sum[c]) / (s * (s - 1) + 1e-12)
        stats = dict(
            size=sizes,
            intra_weight=intra_sum,
            volume=A_sup.sum(axis=1),  # degree/strength of supernodes (with self-loops)
            density=density,
        )

    return (A_sup, node2comp, comp2nodes, stats) if return_stats else (A_sup, node2comp)
def proportions_to_partition(r: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Convert per-node probabilities into a binary co-association matrix."""
    labels = (np.asarray(r) > threshold).astype(int)
    return np.equal.outer(labels, labels).astype(int)
