"""Community-detection helpers and wrappers."""

from __future__ import annotations

import itertools

import networkx as nx
import numpy as np
from scipy import sparse

from asunder.base.algorithms.signed_louvain import community_detection as cd
from asunder.base.algorithms.signed_louvain import util as slouvain_util
from asunder.base.utils.graph import group_nodes_by_community


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

def labels_to_probabilities(A, labels, p=1):
    """
    Convert hard labels into row-normalized membership probabilities.
    
    Parameters
    ----------
    A : ndarray of float, shape (N, N)
        Graph adjacency/weight matrix.
    labels : ndarray of int, shape (N,)
        [Predicted] community labels for each node in a given graph.
    p : int
        Order of the norm.
    
    Returns
    -------
    ndarray of float, shape (N, K)
        Normalized matrix with K community assignment confidence scores for each node.
    """
    _, _, _, _, normalize, get_membership = _import_sknetwork()
    if not sparse.isspmatrix_csr(A):
        A = sparse.csr_matrix(np.asarray(A), dtype=float)
    M = get_membership(labels)
    return normalize(A @ M, p=p)


def probability_to_integer_labels(probabilities, method="threshold", threshold=0.8, verbose=False):
    """
    Heuristic map from probability / soft memberships values to integer labels using a configurable rule. It is used to sunder a core-like community from every other node based on the observation that such core-like nodes have low membership scores across every community, given their central role.
    These nodes are core-like and not exactly core nodes because they do not exhibit the typical dense connection one expects from core nodes. They, in fact, are not adjacent to one another.
    
    Parameters
    ----------
    probabilities : ndarray of float, shape (N,K) or (N,)
        2D probabilities reflect the confidence that each node n belongs to community k. 
        1D probabilities reflect the confidence that a node is in one of two groups, typically a core and a periphery group.
    method : str
        One of "threshold," "gaussian_mixture," and "DBSCAN." Determines whether clustering algorithms are required to process probabilities or if thresholding is sufficient.
    threshold : float
        Value (between 0 and 1) below which a node is assumed to be in the core-like group.
    verbose : bool
        Controls the verbosity of the output. Default is False.
    
    Returns
    -------
    ndarray of int, shape (N,)
        Integer community assignment of the nodes reflecting a bipartition.
    """
    from sklearn.cluster import DBSCAN
    from sklearn.mixture import GaussianMixture

    assert method in ["threshold", "gaussian_mixture", "DBSCAN"]

    if probabilities.ndim == 2:
        p = np.max(probabilities, axis=1).reshape((-1, 1))
    else:
        p = probabilities.reshape((-1, 1))
    scaled_probabilities = (probabilities - p.min()) / (p.max() - p.min() + 1e-12)
    scaled_probabilities[scaled_probabilities < 0] = 0

    if verbose:
        print("Probability values are:\n", p)
    partition = np.zeros(shape=(probabilities.shape[0],))

    if method == "threshold":
        for i in range(probabilities.shape[0]):
            if np.max(probabilities[i]) < threshold:
                partition[i] = -1
            else:
                partition[i] = np.argmax(probabilities[i])
    elif method == "gaussian_mixture":
        gmm = GaussianMixture(n_components=2, random_state=42)
        labels_gmm = gmm.fit_predict(p)
        core_cluster = np.argmin(gmm.means_)
        if verbose:
            print("Labels from GMM are:\n", labels_gmm)
        for i in range(probabilities.shape[0]):
            if labels_gmm[i] == core_cluster:
                partition[i] = -1
            else:
                partition[i] = np.argmax(probabilities[i])
    else:
        labels_dbscan = DBSCAN().fit_predict(
            scaled_probabilities if (np.std(np.unique(p)) < 0.25) else probabilities
        )
        v, c = np.unique(labels_dbscan, return_counts=True)
        lv = v[np.argmin(c)]
        for i in range(probabilities.shape[0]):
            if labels_dbscan[i] == lv:
                partition[i] = -1
            else:
                partition[i] = np.argmax(probabilities[i])
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


def run_modularity(modified_A, algo="louvain", package="networkx", resolution=1, verbose=False, refine=True, refine_params=None):
    """
    Run modularity-style community detection and return ``(partition, score)``.
    
    Parameters
    ----------
    modified_A : ndarray of float, shape (N, N)
        Augmented adjacency / weight matrix reflecting the original adjacency / weight matrix with dual-modified weights. Negative weights are not allowed.
        The original adjacency / weight matrix can also be parsed.
    algo : str
        Algorithm to be used for modularity based community detection.
    package : str
        Python package to be used for modularity based community detection.
    resolution : int or float
        Resolution parameter (gamma) used in modularity based methods.
    verbose : bool
        Controls the verbosity of the output. Default is False.
    refine : bool
        Enables/disables refinement procedure.
    refine_params : dict[str, Any]
        Refinement parameters, typically including the refinement function and any associated arguments.
    
    Returns
    -------
    zii: ndarray of int, shape (N, N)
        2D graph partititon.
    metric: float
        Modularity score of ``zii`` computed using the provided adjacency / weight matrix.
    """
    assert modified_A.min() >= 0, "Adjacency / weight matrix includes negative values."
    assert algo in ["louvain", "leiden", "greedy", "girvan_newman"]
    modG = nx.from_numpy_array(modified_A.astype([("weight", "float")]))
    metric = None

    if package in {"igraph", "leidenalg"}:
        ig = _import_igraph()
        ig_graph = ig.Graph.Weighted_Adjacency(modified_A.tolist(), mode="UNDIRECTED", attr="weight")
    else:
        ig_graph = None

    if algo == "louvain":
        if package == "networkx":
            communities = nx.community.louvain_communities(modG, weight="weight", resolution=resolution)
        elif package == "sknetwork":
            Louvain, _, _, _, _, _ = _import_sknetwork()
            partition = Louvain().fit_predict(modified_A)
            communities = {}
            for i, val in enumerate(partition):
                communities.setdefault(val, set()).add(int(i))
            communities = communities.values()
        else:
            raise NotImplementedError(f"Invalid package: {package}")
    elif algo == "leiden":
        if package == "leidenalg":
            la = _import_leidenalg()
            partition = la.find_partition(
                ig_graph, la.ModularityVertexPartition, seed=42, weights="weight"
            )
            # TODO:Test negative weight handling
            # g_pos = ig_graph.subgraph_edges(ig_graph.es.select(weight_gt=0), delete_vertices=False)
            # g_neg = ig_graph.subgraph_edges(ig_graph.es.select(weight_lt=0), delete_vertices=False)
            # g_neg.es['weight'] = [-w for w in g_neg.es['weight']]
            # part_pos = la.ModularityVertexPartition(g_pos, weights='weight')
            # part_neg = la.ModularityVertexPartition(g_neg, weights='weight')

            # optimiser = la.Optimiser()
            # diff = optimiser.optimise_partition_multiplex([part_pos, part_neg], layer_weights=[1, -1])
            # partition = part_pos
            membership = {}
            for node, comm in zip(ig_graph.vs.indices, partition.membership):
                membership.setdefault(comm, []).append(node)
            communities = list(membership.values())
        elif package == "sknetwork":
            _, Leiden, _, _, _, _ = _import_sknetwork()
            partition = Leiden().fit_predict(modified_A)
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
        else:
            raise NotImplementedError(f"Invalid package: {package}")
    else:
        communities, _ = best_girvan_newman_partition(modG, max_levels=modified_A.shape[0])

    oneD_z = np.zeros(shape=(modified_A.shape[0]), dtype=np.int64)
    for i, community in enumerate(communities):
        for node in community:
            oneD_z[node] = i

    if refine:
        if refine_params is None:
            raise ValueError("refine_params is required when refine=True")
        zii = refine_params["refine_func"](A=modified_A, partition=oneD_z, **refine_params["kwargs"])
        _, communities = group_nodes_by_community(np.array(zii))
        # TODO: Recompute metric or decide to leave as None.
    else:
        zii = np.equal.outer(oneD_z, oneD_z).astype(int)
        metric = nx.community.modularity(modG, communities)
    return zii, metric


def run_lpa(modified_A, refine=True):
    """
    Run label propagation clustering and return ``(partition, modularity)``.
    
    Parameters
    ----------
    modified_A : ndarray of float, shape (N, N)
        Augmented adjacency / weight matrix reflecting the original adjacency / weight matrix with dual-modified weights. Negative weights are not allowed.
        The original adjacency / weight matrix can also be parsed.
    refine : bool
        Enables/disables refinement procedure.
    
    Returns
    -------
    zii: ndarray of int, shape (N, N)
        2D graph partititon.
    float
        Modularity score of ``zii`` computed using the provided adjacency / weight matrix.
    """
    _, _, PropagationClustering, get_modularity, _, _ = _import_sknetwork()
    algorithm = PropagationClustering()
    if refine:
        result = algorithm.fit_predict_proba(modified_A)
        partition = np.zeros(shape=(result.shape[0],))
        for i in range(result.shape[0]):
            partition[i] = -1 if np.max(result[i]) < 0.8 else np.argmax(result[i])
    else:
        partition = algorithm.fit_predict(modified_A)

    communities = {}
    for i, val in enumerate(partition):
        communities.setdefault(val, set()).add(int(i))
    oneD_z = np.zeros(shape=(modified_A.shape[0]), dtype=np.int64)
    for i, community in enumerate(communities.values()):
        for node in community:
            oneD_z[node] = i
    zii = np.equal.outer(oneD_z, oneD_z).astype(int)
    return zii, get_modularity(modified_A, partition.astype(int))


def run_igraph_spinglass(modified_A):
    """
    Run igraph spinglass community detection and return a partition matrix.
    
    Parameters
    ----------
    modified_A : ndarray of float, shape (N, N)
        Augmented adjacency / weight matrix reflecting the original adjacency / weight matrix with dual-modified weights. Negative weights are not allowed.
        The original adjacency / weight matrix can also be parsed.
    
    Returns
    -------
    ndarray of int, shape (N, N)
        2D graph partititon.
    """
    ig = _import_igraph()
    ig_graph = ig.Graph.Weighted_Adjacency(modified_A.tolist(), mode="UNDIRECTED", attr="weight")
    clustering = ig_graph.community_spinglass(
        weights="weight", implementation="neg", lambda_=0.0, spins=500, start_temp=1.0, stop_temp=0.01, cool_fact=0.99
    )
    oneD_z = clustering.membership
    return np.equal.outer(oneD_z, oneD_z).astype(int)


def run_igraph(modified_A, algo="infomap", resolution=1):
    """
    Run selected igraph community algorithm and return ``(partition, score)``.
    
    Parameters
    ----------
    modified_A : ndarray of float, shape (N, N)
        Augmented adjacency / weight matrix reflecting the original adjacency / weight matrix with dual-modified weights. Negative weights are not allowed.
        The original adjacency / weight matrix can also be parsed.
    algo : str
        Algorithm to be used for modularity based community detection.
    resolution : int or float
        Resolution parameter (gamma) used in computing the modularity metric.
    
    Returns
    -------
    zii: ndarray of int, shape (N, N)
        2D graph partititon.
    metric: float
        Modularity score of ``zii`` computed using the provided adjacency / weight matrix.
    """
    ig = _import_igraph()
    ig_graph = ig.Graph.Weighted_Adjacency(modified_A.tolist(), mode="UNDIRECTED", attr="weight")
    if algo == "infomap":
        clustering = ig_graph.community_infomap(edge_weights="weight")
    elif algo == "lpa":
        clustering = ig_graph.community_label_propagation(weights="weight")
    elif algo == "multilevel":
        clustering = ig_graph.community_multilevel(weights="weight", resolution=resolution)
    elif algo == "voronoi":
        clustering = ig_graph.community_voronoi(weights="weight", radius=resolution)
    elif algo == "walktrap":
        clustering = ig_graph.community_walktrap(weights="weight").as_clustering()
    else:
        raise NotImplementedError("Invalid Igraph Algorithm")
    oneD_z = clustering.membership
    zii = np.equal.outer(oneD_z, oneD_z).astype(int)
    metric = ig_graph.modularity(clustering, weights="weight", resolution=resolution)
    return zii, metric


def run_signed_louvain(modified_A):
    """
    Run signed Louvain on positive/negative layers and return ``(partition, score)``.
    
    Parameters
    ----------
    modified_A : ndarray of float, shape (N, N)
        Augmented adjacency / weight matrix reflecting the original adjacency / weight matrix with dual-modified weights. Negative weights are not allowed.
        The original adjacency / weight matrix can also be parsed.
    
    Returns
    -------
    zii: ndarray of int, shape (N, N)
        2D graph partititon.
    float
        Modularity score of ``zii`` computed using the provided adjacency / weight matrix.
    """
    n_nodes = modified_A.shape[0]
    edges = [(i, j, modified_A[i, j]) for i, j in zip(*np.where(np.triu(modified_A) != 0))]
    graph = slouvain_util.build_nx_graph(n_nodes, edges)
    posgraph, neggraph = slouvain_util.build_subgraphs(graph, weight="weight")
    communities, status = cd.best_partition(
        layers=[posgraph, neggraph],
        layer_weights=[1.0, -1.0],
        resolutions=[1.0, 1.0],
        masks=[False, True],
        k=2,
        initial_membership=None,
        weight="weight",
        random_state=None,
        pass_max=40,
        return_dendogram=False,
        silent=True,
    )
    oneD_z = np.zeros(shape=(modified_A.shape[0]), dtype=np.int64)
    for node, community in communities.items():
        oneD_z[node] = community
    zii = np.equal.outer(oneD_z, oneD_z).astype(int)
    return zii, status.modularity()
