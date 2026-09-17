"""Community-detection helpers and wrappers."""

from __future__ import annotations

import itertools

import networkx as nx
import numpy as np
from scipy import sparse

from asunder.base.algorithms.signed_louvain import community_detection as cd
from asunder.base.algorithms.signed_louvain import util as slouvain_util
from asunder.base.utils.graph import partition_vector_to_2d_matrix
from asunder.base.utils.matrix import checked_to_dense
from asunder.types import MatrixLike


def _import_sknetwork():
    """
    Internal helper for importing sknetwork modules.
    
    Returns
    -------
    tuple of modules and methods
        Clustering modules and helper functions from sknetwork.
    """
    from sknetwork.clustering import Leiden, Louvain, PropagationClustering, get_modularity
    from sknetwork.linalg import normalize
    from sknetwork.utils import get_membership

    return Louvain, Leiden, PropagationClustering, get_modularity, normalize, get_membership


def _import_igraph():
    """
    Internal helper for importing igraph.
    
    Returns
    -------
    module
        igraph module
    """
    import igraph as ig

    return ig


def _import_leidenalg():
    """
    Internal helper for importing leidenalg.
    
    Returns
    -------
    module
        leidenalg module
    """
    import leidenalg as la

    return la


def _igraph_from_matrix(matrix: MatrixLike):
    """Build an undirected igraph graph without a dense Python-list copy."""
    ig = _import_igraph()
    if sparse.issparse(matrix):
        upper = sparse.triu(matrix, k=0, format="coo")
    else:
        upper = sparse.coo_matrix(np.triu(np.asarray(matrix)))
    keep = upper.data != 0
    rows = upper.row[keep]
    columns = upper.col[keep]
    weights = np.asarray(upper.data[keep], dtype=float)
    graph = ig.Graph(
        n=matrix.shape[0],
        edges=list(zip(rows.tolist(), columns.tolist())),
        directed=False,
    )
    graph.es["weight"] = weights.tolist()
    return graph


def _partition_from_labels(
    labels,
    *,
    column_storage="dense",
    sparse_column_density_threshold=0.20,
    max_dense_working_bytes=512 * 1024**2,
):
    """Construct a hard column using the requested storage policy."""
    return partition_vector_to_2d_matrix(
        np.asarray(labels),
        storage=column_storage,
        sparse_column_density_threshold=sparse_column_density_threshold,
        max_dense_working_bytes=max_dense_working_bytes,
    )

def labels_to_probabilities(
    A: MatrixLike,
    labels: np.ndarray,
    p: int = 1,
) -> sparse.spmatrix:
    """
    Convert hard labels into row-normalized membership probabilities.
    
    Parameters
    ----------
    A : numpy.ndarray or scipy.sparse.spmatrix, shape (N, N)
        Graph adjacency/weight matrix.
    labels : ndarray of int, shape (N,)
        [Predicted] community labels for each node in a given graph.
    p : int
        Order of the norm.
    
    Returns
    -------
    scipy.sparse.spmatrix, shape (N, K)
        Normalized matrix with K community-assignment confidence scores for
        each node.
    """
    _, _, _, _, normalize, get_membership = _import_sknetwork()
    if sparse.issparse(A):
        A = sparse.csr_matrix(A, dtype=float)
    else:
        A = sparse.csr_matrix(np.asarray(A), dtype=float)
    M = get_membership(labels)
    return normalize(A @ M, p=p)


def probability_to_integer_labels(
    probabilities,
    method="threshold",
    threshold=0.8,
    verbose=False,
    seed=42,
):
    """Convert soft memberships to labels with a low-confidence group.

    Nodes assigned to the low-confidence group receive label ``-1``. All
    other nodes receive the label of their largest membership value. For
    clustering methods, the cluster with the lowest mean maximum-membership
    confidence is treated as the low-confidence group.
    
    Parameters
    ----------
    probabilities : ndarray of float, shape (N, K) or (N,)
        Per-node membership values. For a two-dimensional input, each row
        contains the memberships of one node. A one-dimensional input contains
        one confidence value per node.
    method : {"threshold", "gaussian_mixture", "DBSCAN"}, default="threshold"
        Rule used to identify the low-confidence nodes.
    threshold : float, default=0.8
        Confidence below which a node is assigned to the low-confidence group.
        This is also the fallback when DBSCAN clustering finds fewer than
        two clusters.
    verbose : bool, default=False
        Whether to print the maximum membership values.
    seed : int or None, default=42
        Random seed used by Gaussian-mixture clustering.
    
    Returns
    -------
    ndarray of int, shape (N,)
        Hard community labels. Low-confidence nodes have label ``-1``.

    Raises
    ------
    ValueError
        If ``method`` is unsupported or ``probabilities`` has an invalid
        shape.
    """
    from sklearn.cluster import DBSCAN
    from sklearn.mixture import GaussianMixture

    if method not in {"threshold", "gaussian_mixture", "DBSCAN"}:
        raise ValueError(
            "method must be one of 'threshold', 'gaussian_mixture', or 'DBSCAN'."
        )
    values = np.asarray(probabilities, dtype=float)
    if values.ndim not in {1, 2} or values.shape[0] == 0:
        raise ValueError("probabilities must have shape (N,) or (N, K) with N > 0.")

    if values.ndim == 2:
        p = np.max(values, axis=1).reshape((-1, 1))
        partition = np.argmax(values, axis=1).astype(int)
    else:
        p = values.reshape((-1, 1))
        partition = np.zeros(values.shape[0], dtype=int)
    scaled_probabilities = (values - p.min()) / (p.max() - p.min() + 1e-12)
    scaled_probabilities[scaled_probabilities < 0] = 0

    if verbose:
        print("Probability values are:\n", p)

    if method == "threshold":
        low_confidence = p.reshape(-1) < threshold
    elif method == "gaussian_mixture":
        gmm = GaussianMixture(n_components=2, random_state=seed)
        labels_gmm = gmm.fit_predict(p)
        low_cluster = int(np.argmin(gmm.means_.reshape(-1)))
        if verbose:
            print("Labels from GMM are:\n", labels_gmm)
        low_confidence = labels_gmm == low_cluster
    else:
        labels_dbscan = DBSCAN().fit_predict(
            scaled_probabilities if (np.std(np.unique(p)) < 0.25) else values
        )
        clusters = np.unique(labels_dbscan)
        if clusters.size < 2:
            low_confidence = p.reshape(-1) < threshold
        else:
            confidence = p.reshape(-1)
            low_cluster = min(
                clusters,
                key=lambda cluster: float(confidence[labels_dbscan == cluster].mean()),
            )
            low_confidence = labels_dbscan == low_cluster

    partition[low_confidence] = -1
    return partition

def best_girvan_newman_partition(G, max_levels=10):
    """
    Search Girvan-Newman levels and return the best modularity partition.
    
    Parameters
    ----------
    G : nx.Graph
        Input NetworkX graph.
    max_levels : int
        Maximum number of levels to check in the Girvan-Newman search process.
    
    Returns
    -------
    best_communities: tuple[list[int or str]] or None
        Iterable with communities reflecting the best modularity found during the search process.
    best_mod: float
        Best modularity obtained during the search process.
    """
    comp = nx.community.girvan_newman(G)
    best_mod = -1.0
    best_communities = None
    for communities in itertools.islice(comp, max_levels):
        communities_list = tuple(sorted(c) for c in communities)
        mod = nx.community.modularity(G, communities_list)
        if mod > best_mod:
            best_mod = mod
            best_communities = communities_list
    return best_communities, best_mod


def run_modularity(
    modified_A: MatrixLike,
    algo="louvain",
    package="networkx",
    seed=42,
    resolution=1,
    verbose=False,
    *,
    column_storage="dense",
    sparse_column_density_threshold=0.20,
    max_dense_working_bytes=512 * 1024**2,
) -> tuple[MatrixLike, float]:
    """
    Run modularity-style community detection and return ``(partition, score)``.
    
    Parameters
    ----------
    modified_A : numpy.ndarray or scipy.sparse.spmatrix, shape (N, N)
        Augmented adjacency / weight matrix reflecting the original adjacency / weight matrix with dual-modified weights. Negative weights are not allowed.
        The original adjacency / weight matrix can also be parsed.
    algo : str
        Algorithm to be used for modularity based community detection.
    package : str
        Python package to be used for modularity based community detection.
    seed : int | None
        Random seed value.
    resolution : int or float
        Resolution parameter (gamma) used in modularity based methods.
    verbose : bool
        Controls the verbosity of the output. Default is False.
    column_storage : {"auto", "dense", "csr"}, default="dense"
        Physical storage used for the returned hard partition.
    sparse_column_density_threshold : float, default=0.20
        Maximum density at which automatic storage uses CSR.
    max_dense_working_bytes : int or None, default=536870912
        Maximum estimated dense working set for partition construction.

    Returns
    -------
    zii : ndarray of bool or scipy.sparse.csr_matrix, shape (N, N)
        2D graph partition.
    metric: float
        Modularity score of ``zii`` computed using the provided adjacency / weight matrix.
    """
    minimum = modified_A.min() if sparse.issparse(modified_A) else np.min(modified_A)
    assert minimum >= 0, "Adjacency / weight matrix includes negative values." # TODO: may need to drop this.
    assert algo in ["louvain", "leiden", "greedy", "girvan_newman"]
    modG = (
        nx.from_scipy_sparse_array(sparse.csr_matrix(modified_A), edge_attribute="weight")
        if sparse.issparse(modified_A)
        else nx.from_numpy_array(modified_A.astype([("weight", "float")]))
    )
    metric = None

    if package == "igraph":
        ig_graph = _igraph_from_matrix(modified_A)
    else:
        ig_graph = None

    if algo == "louvain":
        if package == "networkx":
            communities = nx.community.louvain_communities(modG, weight="weight", resolution=resolution, seed=seed)
        elif package == "sknetwork":
            Louvain, _, _, _, _, _ = _import_sknetwork()
            partition = Louvain(resolution=resolution).fit_predict(modified_A)
            communities = {}
            for i, val in enumerate(partition):
                communities.setdefault(val, set()).add(int(i))
            communities = communities.values()
        else:
            raise NotImplementedError(f"Invalid package: {package}")
    elif algo == "leiden":
        if package == "sknetwork":
            _, Leiden, _, _, _, _ = _import_sknetwork()
            partition = Leiden(resolution=resolution).fit_predict(modified_A)
            communities = {}
            for i, val in enumerate(partition):
                communities.setdefault(val, set()).add(int(i))
            communities = communities.values()
        elif package == "igraph":
            clustering = ig_graph.community_leiden(
                objective_function="modularity", weights="weight", resolution=resolution
            )
            communities = {}
            for i, val in enumerate(clustering.membership):
                communities.setdefault(val, set()).add(int(i))
            communities = communities.values()
        else:
            raise NotImplementedError(f"Invalid package: {package}")
    elif algo == "greedy":
        if package == "networkx":
            communities = nx.community.greedy_modularity_communities(
                modG, weight="weight", resolution=resolution
            )
        elif package == "igraph":
            clustering = ig_graph.community_fastgreedy(weights="weight").as_clustering()
            communities = {}
            for i, val in enumerate(clustering.membership):
                communities.setdefault(val, set()).add(int(i))
            communities = communities.values()
            resolution = None
        else:
            raise NotImplementedError(f"Invalid package: {package}")
    else:
        communities, _ = best_girvan_newman_partition(modG, max_levels=modified_A.shape[0])
        resolution = None

    oneD_z = np.zeros(shape=(modified_A.shape[0]), dtype=np.int64)
    for i, community in enumerate(communities):
        for node in community:
            oneD_z[node] = i

    zii = _partition_from_labels(
        oneD_z,
        column_storage=column_storage,
        sparse_column_density_threshold=sparse_column_density_threshold,
        max_dense_working_bytes=max_dense_working_bytes,
    )
    metric = nx.community.modularity(modG, communities, resolution=resolution if resolution is not None else 1)
    return zii, metric


def run_lpa(
    modified_A: MatrixLike,
    *,
    column_storage="dense",
    sparse_column_density_threshold=0.20,
    max_dense_working_bytes=512 * 1024**2,
) -> tuple[MatrixLike, float]:
    """
    Run label propagation clustering and return ``(partition, modularity)``.
    
    Parameters
    ----------
    modified_A : numpy.ndarray or scipy.sparse.spmatrix, shape (N, N)
        Augmented adjacency / weight matrix reflecting the original adjacency / weight matrix with dual-modified weights. Negative weights are not allowed.
        The original adjacency / weight matrix can also be parsed.
    column_storage : {"auto", "dense", "csr"}, default="dense"
        Physical storage used for the returned hard partition.
    sparse_column_density_threshold : float, default=0.20
        Maximum density at which automatic storage uses CSR.
    max_dense_working_bytes : int or None, default=536870912
        Maximum estimated dense working set for partition construction.
    
    Returns
    -------
    zii : ndarray of bool or scipy.sparse.csr_matrix, shape (N, N)
        2D graph partition.
    float
        Modularity score of ``zii`` computed using the provided adjacency / weight matrix.
    """
    _, _, PropagationClustering, get_modularity, _, _ = _import_sknetwork()
    algorithm = PropagationClustering()
    partition = algorithm.fit_predict(modified_A)

    communities = {}
    for i, val in enumerate(partition):
        communities.setdefault(val, set()).add(int(i))
    oneD_z = np.zeros(shape=(modified_A.shape[0]), dtype=np.int64)
    for i, community in enumerate(communities.values()):
        for node in community:
            oneD_z[node] = i
    zii = _partition_from_labels(
        oneD_z,
        column_storage=column_storage,
        sparse_column_density_threshold=sparse_column_density_threshold,
        max_dense_working_bytes=max_dense_working_bytes,
    )
    return zii, get_modularity(modified_A, partition.astype(int))


def run_igraph_spinglass(
    modified_A: MatrixLike,
    *,
    column_storage="dense",
    sparse_column_density_threshold=0.20,
    max_dense_working_bytes=512 * 1024**2,
) -> MatrixLike:
    """
    Run igraph spinglass community detection and return a partition matrix.
    
    Parameters
    ----------
    modified_A : numpy.ndarray or scipy.sparse.spmatrix, shape (N, N)
        Augmented adjacency / weight matrix reflecting the original adjacency / weight matrix with dual-modified weights. Negative weights are not allowed.
        The original adjacency / weight matrix can also be parsed.
    column_storage : {"auto", "dense", "csr"}, default="dense"
        Physical storage used for the returned hard partition.
    sparse_column_density_threshold : float, default=0.20
        Maximum density at which automatic storage uses CSR.
    max_dense_working_bytes : int or None, default=536870912
        Maximum estimated dense working set for partition construction.
    
    Returns
    -------
    ndarray of bool or scipy.sparse.csr_matrix, shape (N, N)
        2D graph partition.
    """
    ig_graph = _igraph_from_matrix(modified_A)
    clustering = ig_graph.community_spinglass(
        weights="weight", implementation="neg", lambda_=0.0, spins=500, start_temp=1.0, stop_temp=0.01, cool_fact=0.99
    )
    oneD_z = clustering.membership
    return _partition_from_labels(
        oneD_z,
        column_storage=column_storage,
        sparse_column_density_threshold=sparse_column_density_threshold,
        max_dense_working_bytes=max_dense_working_bytes,
    )


def run_igraph(
    modified_A: MatrixLike,
    algo="infomap",
    resolution=1,
    *,
    column_storage="dense",
    sparse_column_density_threshold=0.20,
    max_dense_working_bytes=512 * 1024**2,
) -> tuple[MatrixLike, float]:
    """
    Run selected igraph community algorithm and return ``(partition, score)``.
    
    Parameters
    ----------
    modified_A : numpy.ndarray or scipy.sparse.spmatrix, shape (N, N)
        Augmented adjacency / weight matrix reflecting the original adjacency / weight matrix with dual-modified weights. Negative weights are not allowed.
        The original adjacency / weight matrix can also be parsed.
    algo : str
        Algorithm to be used for modularity based community detection.
    resolution : int or float
        Resolution parameter (gamma) used in computing the modularity metric.
    column_storage : {"auto", "dense", "csr"}, default="dense"
        Physical storage used for the returned hard partition.
    sparse_column_density_threshold : float, default=0.20
        Maximum density at which automatic storage uses CSR.
    max_dense_working_bytes : int or None, default=536870912
        Maximum estimated dense working set for partition construction.
    
    Returns
    -------
    zii : ndarray of bool or scipy.sparse.csr_matrix, shape (N, N)
        2D graph partition.
    metric : float
        Modularity score of ``zii`` computed using the provided adjacency / weight matrix.
    """
    ig_graph = _igraph_from_matrix(modified_A)
    if algo == "infomap":
        clustering = ig_graph.community_infomap(edge_weights="weight")
    elif algo == "lpa":
        clustering = ig_graph.community_label_propagation(weights="weight")
    elif algo == "multilevel":
        clustering = ig_graph.community_multilevel(weights="weight", resolution=resolution)
    elif algo == "voronoi":
        clustering = ig_graph.community_voronoi(weights="weight")
    elif algo == "walktrap":
        clustering = ig_graph.community_walktrap(weights="weight").as_clustering()
    elif algo == "cpm_leiden":
        #  Resolution-limit free and handles negative weights
        clustering = ig_graph.community_leiden(objective_function="CPM", weights='weight', resolution=resolution)
    else:
        raise NotImplementedError("Invalid Igraph Algorithm")
    oneD_z = clustering.membership
    zii = _partition_from_labels(
        oneD_z,
        column_storage=column_storage,
        sparse_column_density_threshold=sparse_column_density_threshold,
        max_dense_working_bytes=max_dense_working_bytes,
    )
    metric = ig_graph.modularity(clustering, weights="weight", resolution=resolution if  algo == "multilevel" else 1)
    return zii, metric

def run_leidenalg(
    modified_A: MatrixLike,
    algo="leiden",
    seed=42,
    resolution=1,
    verbose=False,
    *,
    column_storage="dense",
    sparse_column_density_threshold=0.20,
    max_dense_working_bytes=512 * 1024**2,
) -> tuple[MatrixLike, float]:
    """
    Run Leiden algorithm to optimize varying quality functions and return ``(partition, score)``.

    Parameters
    ----------
    modified_A : numpy.ndarray or scipy.sparse.spmatrix, shape (N, N)
        Augmented adjacency / weight matrix reflecting the original adjacency / weight matrix with dual-modified weights.
        The original adjacency / weight matrix can also be parsed.
    algo : str
        Algorithm to be used for community detection. It describes the quality being maximized and whether signed graphs are allowed or not.
    seed : int | None
        Random seed value.
    resolution : int or float
        Resolution parameter (gamma) used.
    verbose : bool
        Controls the verbosity of the output. Default is False.
    column_storage : {"auto", "dense", "csr"}, default="dense"
        Physical storage used for the returned hard partition.
    sparse_column_density_threshold : float, default=0.20
        Maximum density at which automatic storage uses CSR.
    max_dense_working_bytes : int or None, default=536870912
        Maximum estimated dense working set for partition construction.

    Returns
    -------
    zii : ndarray of bool or scipy.sparse.csr_matrix, shape (N, N)
        2D graph partition.
    metric : float
        Quality of ``zii`` computed using the provided adjacency / weight matrix. This could be modularity, CPM or surprise.
    """
    assert algo in ['leiden', 'signed_leiden', 'cpm_leiden', 'surprise_leiden', 'signed_surprise_leiden']
    la = _import_leidenalg()
    # create modified graph
    ig_graph = _igraph_from_matrix(modified_A)

    metric = None
    if algo == "leiden":
        partition = la.find_partition(
            ig_graph,
            la.RBConfigurationVertexPartition,
            resolution_parameter=resolution,
            seed=seed,
            weights='weight'
        )
        metric = partition.quality() / modified_A.sum()
    elif algo == "signed_leiden":
        # uses multiplex community detection to handle negative weights
        g_pos = ig_graph.subgraph_edges(ig_graph.es.select(weight_gt=0), delete_vertices=False)
        g_neg = ig_graph.subgraph_edges(ig_graph.es.select(weight_lt=0), delete_vertices=False)
        g_neg.es['weight'] = [-w for w in g_neg.es['weight']]

        layer_partitions = la.find_partition_multiplex(
            graphs=[g_pos, g_neg],
            partition_type=la.RBConfigurationVertexPartition,
            layer_weights=[1, -1],
            weights='weight',
            resolution_parameter=resolution,
            seed=seed
        )
        partition = layer_partitions[0]

        part_pos = la.RBConfigurationVertexPartition(g_pos, weights='weight', initial_membership=layer_partitions[0], resolution_parameter=resolution)
        part_neg = la.RBConfigurationVertexPartition(g_neg, weights='weight', initial_membership=layer_partitions[0], resolution_parameter=resolution)
        m_pos = 2 * sum(g_pos.es['weight'])
        m_neg = 2 * sum(g_neg.es['weight'])
        Q_pos = (1 * part_pos.quality() / m_pos) if m_pos > 0 else 0
        Q_neg = (1 * part_neg.quality() / m_neg) if m_neg > 0 else 0
        metric = Q_pos - Q_neg
    elif algo == "cpm_leiden":
        #  Constant Potts Model (CPM). Resolution-limit free and handles negative weights.
        partition = la.find_partition(
            ig_graph,
            la.CPMVertexPartition,
            resolution_parameter=resolution,
            seed=seed,
            weights='weight'
        )
        metric = partition.quality()
    elif algo == "surprise_leiden":
        partition = la.find_partition(
            ig_graph,
            la.SurpriseVertexPartition,
            seed=seed,
            weights='weight'
        )
        metric = partition.quality()
    elif algo == "signed_surprise_leiden":
        # uses multiplex community detection to handle negative weights
        g_pos = ig_graph.subgraph_edges(ig_graph.es.select(weight_gt=0), delete_vertices=False)
        g_neg = ig_graph.subgraph_edges(ig_graph.es.select(weight_lt=0), delete_vertices=False)
        g_neg.es['weight'] = [-w for w in g_neg.es['weight']]

        layer_partitions = la.find_partition_multiplex(
            graphs=[g_pos, g_neg],
            partition_type=la.SurpriseVertexPartition,
            layer_weights=[1, -1],
            weights='weight',
            seed=seed
        )
        part_pos = la.SurpriseVertexPartition(g_pos, weights='weight', initial_membership=layer_partitions[0])
        part_neg = la.SurpriseVertexPartition(g_neg, weights='weight', initial_membership=layer_partitions[0])
        partition = layer_partitions[0]
        metric = (1 * part_pos.quality()) + (-1 * part_neg.quality())
    else:
        raise(NotImplementedError("Invalid Leidenalg Algorithm."))

    # process result to get partitions
    oneD_z = np.array(
        partition if algo.startswith("signed") else partition.membership
    )
    zii = _partition_from_labels(
        oneD_z,
        column_storage=column_storage,
        sparse_column_density_threshold=sparse_column_density_threshold,
        max_dense_working_bytes=max_dense_working_bytes,
    )
    return zii, metric


def run_signed_louvain(
    modified_A,
    seed=42,
    resolution=1.0,
    *,
    column_storage="dense",
    sparse_column_density_threshold=0.20,
    max_dense_working_bytes=512 * 1024**2,
):
    """
    Run signed Louvain on positive/negative layers and return ``(partition, score)``.
    
    Parameters
    ----------
    modified_A : numpy.ndarray or scipy.sparse.spmatrix, shape (N, N)
        Signed adjacency or dual-adjusted weight matrix. Negative weights are
        supported. Sparse input crosses a guarded dense boundary before graph
        extraction and positive/negative layer construction.
    seed : int or None
        Random seed value
    resolution : float, default=1.0
        Resolution applied to both signed Louvain layers.
    column_storage : {"auto", "dense", "csr"}, default="dense"
        Physical storage used for the returned hard partition.
    sparse_column_density_threshold : float, default=0.20
        Maximum density at which automatic storage uses CSR.
    max_dense_working_bytes : int or None, default=536870912
        Maximum estimated dense conversion, edge-extraction, and partition
        construction workspace, including additional work for dense input.

    Returns
    -------
    zii : ndarray of bool or scipy.sparse.csr_matrix, shape (N, N)
        2D graph partition.
    float
        Modularity score of ``zii`` computed using the provided adjacency / weight matrix.
    """
    modified_A = checked_to_dense(
        modified_A, working_arrays=3.0,
        max_dense_working_bytes=max_dense_working_bytes,
        operation="internal signed-Louvain pricing",
    )
    n_nodes = modified_A.shape[0]
    edges = [(i, j, modified_A[i, j]) for i, j in zip(*np.where(np.triu(modified_A) != 0))]
    graph = slouvain_util.build_nx_graph(n_nodes, edges)
    posgraph, neggraph = slouvain_util.build_subgraphs(graph, weight="weight")
    communities, status = cd.best_partition(
        layers=[posgraph, neggraph],
        layer_weights=[1.0, -1.0],
        resolutions=[resolution, resolution],
        masks=[False, True],
        k=2,
        initial_membership=None,
        weight="weight",
        pass_max=40,
        return_dendogram=False,
        silent=True,
        random_state=seed
    )
    oneD_z = np.zeros(shape=(modified_A.shape[0]), dtype=np.int64)
    for node, community in communities.items():
        oneD_z[node] = community
    zii = _partition_from_labels(
        oneD_z,
        column_storage=column_storage,
        sparse_column_density_threshold=sparse_column_density_threshold,
        max_dense_working_bytes=max_dense_working_bytes,
    )
    return zii, status.modularity()
