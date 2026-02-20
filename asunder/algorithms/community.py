"""Community-detection helpers and wrappers."""

from __future__ import annotations

import itertools

import networkx as nx
import numpy as np
from scipy import sparse

from asunder.algorithms.signed_louvain import community_detection as cd
from asunder.algorithms.signed_louvain import util as slouvain_util
from asunder.utils.graph import (
    group_nodes_by_community,
    partition_matrix_to_vector,
    partition_vector_to_2d_matrix,
)


def _import_sknetwork():
    from sknetwork.clustering import Leiden, Louvain, PropagationClustering, get_modularity
    from sknetwork.linalg import normalize
    from sknetwork.utils import get_membership

    return Louvain, Leiden, PropagationClustering, get_modularity, normalize, get_membership


def _import_igraph():
    import igraph as ig

    return ig


def _import_leidenalg():
    import leidenalg as la

    return la

def labels_to_probabilities(A, labels, p=1):
    """Convert hard labels into row-normalized membership probabilities."""
    _, _, _, _, normalize, get_membership = _import_sknetwork()
    if not sparse.isspmatrix_csr(A):
        A = sparse.csr_matrix(np.asarray(A), dtype=float)
    M = get_membership(labels)
    return normalize(A @ M, p=p)


def probability_to_integer_labels(probabilities, method="threshold", threshold=0.8, verbose=False):
    """Map soft memberships to integer labels using a configurable rule."""
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


def refine_partition_linear_group(
    A, partition, *, p=1, prob_method="threshold", threshold=0.8, verbose=False
):
    """Refine a partition by reassigning low-confidence nodes."""
    labels = partition_matrix_to_vector(partition) if partition.ndim == 2 else partition.copy()
    probs = labels_to_probabilities(A, labels, p=p).toarray()
    refined_labels = probability_to_integer_labels(
        probs, method=prob_method, threshold=threshold, verbose=verbose
    )
    return partition_vector_to_2d_matrix(refined_labels)


def best_girvan_newman_partition(G, max_levels=10):
    """Search Girvan-Newman levels and return the best modularity partition."""
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
    """Run modularity-style community detection and return ``(partition, score)``."""
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
    else:
        zii = np.equal.outer(oneD_z, oneD_z).astype(int)
    metric = nx.community.modularity(modG, communities)
    return zii, metric


def run_lpa(modified_A, refine=True):
    """Run label propagation clustering and return ``(partition, modularity)``."""
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
    """Run igraph spinglass community detection and return a partition matrix."""
    ig = _import_igraph()
    ig_graph = ig.Graph.Weighted_Adjacency(modified_A.tolist(), mode="UNDIRECTED", attr="weight")
    clustering = ig_graph.community_spinglass(
        weights="weight", implementation="neg", lambda_=0.0, spins=500, start_temp=1.0, stop_temp=0.01, cool_fact=0.99
    )
    oneD_z = clustering.membership
    return np.equal.outer(oneD_z, oneD_z).astype(int)


def run_igraph(modified_A, algo="infomap", resolution=1):
    """Run selected igraph community algorithm and return ``(partition, score)``."""
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
    """Run signed Louvain on positive/negative layers and return ``(partition, score)``."""
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
