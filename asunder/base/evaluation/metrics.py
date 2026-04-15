"""Evaluation metrics for structure detection."""
from __future__ import annotations

from typing import Dict, Tuple

import networkx as nx
import numpy as np
from sklearn.metrics import (
    adjusted_rand_score,
    mutual_info_score,
    normalized_mutual_info_score,
)

from asunder.base.column_generation.master import compute_f_star
from asunder.base.utils.graph import partition_matrix_to_vector


def _relabel_consecutive(labels: np.ndarray) -> np.ndarray:
    """
    Relabel a label vector to consecutive integer IDs.

    The returned labels preserve the equality pattern of the input labels,
    but remap the unique label values to ``0, 1, ..., k-1`` in sorted order.

    Parameters
    ----------
    labels : numpy.ndarray
        One-dimensional array of cluster or class labels.

    Returns
    -------
    numpy.ndarray
        Integer array with the same shape as ``labels`` whose values are
        consecutive nonnegative integers.

    See Also
    --------
    numpy.unique : Returns the sorted unique elements of an array and,
        optionally, the inverse mapping used here.

    Notes
    -----
    This is useful before constructing contingency tables or other
    label-based statistics that assume compact integer indexing.
    """
    _, inv = np.unique(labels, return_inverse=True)
    return inv.astype(int)

def _contingency(labels_a: np.ndarray, labels_b: np.ndarray) -> np.ndarray:
    """
    Construct the contingency table between two label assignments.

    Each entry ``M[i, j]`` contains the number of samples assigned to
    cluster ``i`` in ``labels_a`` and cluster ``j`` in ``labels_b``.

    Parameters
    ----------
    labels_a : numpy.ndarray
        One-dimensional array of labels for the first partition.
    labels_b : numpy.ndarray
        One-dimensional array of labels for the second partition.

    Returns
    -------
    numpy.ndarray
        Two-dimensional integer contingency matrix of shape
        ``(n_clusters_a, n_clusters_b)``.

    Raises
    ------
    ValueError
        Raised implicitly by NumPy operations if the input arrays are not
        broadcast-compatible for paired indexing.

    See Also
    --------
    _relabel_consecutive : Relabels arbitrary labels to consecutive integers.

    Notes
    -----
    The two label arrays are first relabeled to consecutive integers so they
    can be used directly as row and column indices.
    """
    labels_a = _relabel_consecutive(np.asarray(labels_a))
    labels_b = _relabel_consecutive(np.asarray(labels_b))
    ka = int(labels_a.max()) + 1
    kb = int(labels_b.max()) + 1
    M = np.zeros((ka, kb), dtype=np.int64)
    np.add.at(M, (labels_a, labels_b), 1)
    return M

def _entropy_from_counts(counts: np.ndarray, *, log_base: float = 2.0) -> float:
    """
    Compute the entropy of a discrete distribution from raw counts.

    Parameters
    ----------
    counts : numpy.ndarray
        Array of nonnegative counts.
    log_base : float, optional
        Base of the logarithm used in the entropy calculation. The default is
        ``2.0``, which yields entropy in bits.

    Returns
    -------
    float
        Entropy of the normalized count distribution. Returns ``0.0`` when
        the total count is nonpositive.

    See Also
    --------
    numpy.log : Natural logarithm used to compute the entropy.

    Notes
    -----
    Zero-probability entries are excluded from the summation to avoid
    undefined logarithms.
    """
    total = float(np.sum(counts))
    if total <= 0:
        return 0.0
    p = counts.astype(float) / total
    p = p[p > 0]
    log = np.log(p) / np.log(log_base)
    return float(-np.sum(p * log))

def _mutual_information(cont: np.ndarray, *, log_base: float = 2.0) -> float:
    """
    Compute mutual information from a contingency table.

    Parameters
    ----------
    cont : numpy.ndarray
        Two-dimensional contingency matrix whose entries are joint counts.
    log_base : float, optional
        Base of the logarithm used in the mutual information calculation.
        The default is ``2.0``, which yields mutual information in bits.

    Returns
    -------
    float
        Mutual information implied by the contingency table. Returns ``0.0``
        when the total count is nonpositive.

    See Also
    --------
    _contingency : Builds the contingency table used as input here.

    Notes
    -----
    Only positive joint probabilities are included in the summation.
    The computation follows

    .. math::

       I(X; Y) = \\sum_{i,j} p_{ij} \\log\\left(\\frac{p_{ij}}{p_i p_j}\\right).

    """
    n = float(cont.sum())
    if n <= 0:
        return 0.0
    pij = cont.astype(float) / n
    pi = pij.sum(axis=1, keepdims=True)
    pj = pij.sum(axis=0, keepdims=True)
    mask = pij > 0
    log = np.log(pij[mask] / (pi @ pj)[mask]) / np.log(log_base)
    return float(np.sum(pij[mask] * log))

def optimality_gap(A, a, m, z_gt, z_sol, tol=1e-10):
    """
    Compute relative optimality gap (percent) between reference and candidate.
    
    Parameters
    ----------
    A : np.ndarray of int | float, shape (N, N)
        Adjacency / weight matrix.
    a : np.ndarray of int | float, shape (N,)
        Degree-like vector; defaults to row sums of the symmetrized adjacency.
    m : float
        Twice the total weight in the graph.
    z_gt : ndarray of int, shape (N, N)
        Reference solution, typically the ground-truth or best-known solution.
    z_sol : ndarray of int, shape (N, N)
        Candidate solution being evaluated.
    tol : float
        Small positive constant added to the denominator to avoid division by
        zero. The default is ``1e-10``.
    
    Returns
    -------
    float
        Percentage optimality gap, computed as

        .. math::

           100 \\times \\frac{f(z_{gt}) - f(z_{sol})}{f(z_{sol}) + \\mathrm{tol}}.
    """
    best_sol = compute_f_star(A, a, m, z_sol)
    best_bound = compute_f_star(A, a, m, z_gt)
    opt_gap = (best_bound - best_sol) * 100 / (best_sol + tol)
    return opt_gap

def nmi(labels_gt: np.ndarray, labels_sol: np.ndarray, *, log_base: float = 2.0) -> float:
    """
    Newman's normalized mutual information: NMI = 2 I(X;Y) / (H(X) + H(Y)).
    
    Parameters
    ----------
    labels_gt : ndarray of int, shape (N,)
        Reference solution, typically the ground-truth or best-known solution.
    labels_sol : ndarray of int, shape (N,)
        Candidate solution being evaluated.
    log_base : float
        Base of logarithm.
    
    Returns
    -------
    float
        Computed NMI.
    """
    cont = _contingency(labels_gt, labels_sol)
    Hx = _entropy_from_counts(cont.sum(axis=1), log_base=log_base)
    Hy = _entropy_from_counts(cont.sum(axis=0), log_base=log_base)
    I = _mutual_information(cont, log_base=log_base)
    denom = Hx + Hy
    if denom == 0.0:
        return 1.0 if np.array_equal(labels_gt, labels_sol) else 0.0
    return float(2.0 * I / denom)

def ari(labels_gt: np.ndarray, labels_sol: np.ndarray) -> float:
    """
    Adjusted Rand Index from the contingency table.
    
    Parameters
    ----------
    labels_gt : ndarray of int, shape (N,)
        Reference solution, typically the ground-truth or best-known solution.
    labels_sol : ndarray of int, shape (N,)
        Candidate solution being evaluated.
    
    Returns
    -------
    float
        Computed ARI.
    """
    cont = _contingency(labels_gt, labels_sol).astype(np.int64)
    n = int(cont.sum())

    def comb2(x: np.ndarray) -> np.ndarray:
        """
        Comb2.
        
        Parameters
        ----------
        x : np.ndarray
            Input parameter.
        
        Returns
        -------
        np.ndarray
            Computed result.
        """
        x = x.astype(np.int64)
        return x * (x - 1) // 2

    sum_ij = int(comb2(cont).sum())
    sum_i = int(comb2(cont.sum(axis=1)).sum())
    sum_j = int(comb2(cont.sum(axis=0)).sum())
    total = n * (n - 1) // 2
    if total == 0:
        return 1.0

    expected = (sum_i * sum_j) / total
    max_index = 0.5 * (sum_i + sum_j)
    denom = max_index - expected
    if denom == 0:
        return 1.0
    return float((sum_ij - expected) / denom)

def vi(labels_gt: np.ndarray, labels_sol: np.ndarray, *, log_base: float = 2.0) -> float:
    """
    Variation of Information: VI = H(X) + H(Y) - 2 I(X;Y).
    
    Parameters
    ----------
    labels_gt : ndarray of int, shape (N,)
        Reference solution, typically the ground-truth or best-known solution.
    labels_sol : ndarray of int, shape (N,)
        Candidate solution being evaluated.
    log_base : float
        Base of logarithm.
    
    Returns
    -------
    float
        Computed VI.
    """
    cont = _contingency(labels_gt, labels_sol)
    Hx = _entropy_from_counts(cont.sum(axis=1), log_base=log_base)
    Hy = _entropy_from_counts(cont.sum(axis=0), log_base=log_base)
    I = _mutual_information(cont, log_base=log_base)
    return float(Hx + Hy - 2.0 * I)

def nmi_sklearn(labels_gt, labels_sol) -> float:
    """
    Compute normalized mutual information via scikit-learn.
    
    Parameters
    ----------
    labels_gt : ndarray of int, shape (N,)
        Reference solution, typically the ground-truth or best-known solution.
    labels_sol : ndarray of int, shape (N,)
        Candidate solution being evaluated.
    
    Returns
    -------
    float
        Computed NMI.
    """
    return float(normalized_mutual_info_score(labels_gt, labels_sol, average_method="arithmetic"))

def ari_sklearn(labels_gt, labels_sol) -> float:
    """
    Compute adjusted Rand index via scikit-learn.
    
    Parameters
    ----------
    labels_gt : ndarray of int, shape (N,)
        Reference solution, typically the ground-truth or best-known solution.
    labels_sol : ndarray of int, shape (N,)
        Candidate solution being evaluated.
    
    Returns
    -------
    float
        Computed ARI.
    """
    return float(adjusted_rand_score(labels_gt, labels_sol))

def vi_sklearn(labels_gt, labels_sol, log_base=2.0) -> float:
    """
    Compute variation of information using sklearn mutual information.
    
    Parameters
    ----------
    labels_gt : ndarray of int, shape (N,)
        Reference solution, typically the ground-truth or best-known solution.
    labels_sol : ndarray of int, shape (N,)
        Candidate solution being evaluated.
    log_base : float
        Base of logarithm.
    
    Returns
    -------
    float
        Computed variation of information.
    """
    mi = mutual_info_score(labels_gt, labels_sol)
    # sklearn MI uses natural log; convert if you want bits
    mi = mi / np.log(log_base)
    # entropies from marginals
    def H(x):
        """
        H.
        
        Parameters
        ----------
        x : Any
            Input parameter.
        
        Returns
        -------
        Any
            Computed result.
        """
        _, c = np.unique(x, return_counts=True)
        p = c / c.sum()
        return float(-(p * (np.log(p) / np.log(log_base))).sum())
    return float(H(labels_gt) + H(labels_sol) - 2.0 * mi)

def permuted_accuracy(
    z_gt: np.ndarray, z_sol: np.ndarray
) -> Tuple[float, Dict[int, int]]:
    """
    Maximum fraction of correctly classified nodes under label permutation.
    
    Parameters
    ----------
    z_gt : np.ndarray of int
        Ground truth partition (1D / 2D).
    z_sol : np.ndarray of int
        Predicted partition (1D / 2D).
    
    Returns
    -------
    Tuple[float, Dict[int, int]]
        Accuracy score and label mapping from ground truth to solution.
    """
    # Accept either label vectors or partition matrices.
    gt = partition_matrix_to_vector(z_gt) if np.asarray(z_gt).ndim == 2 else np.asarray(z_gt)
    sol = partition_matrix_to_vector(z_sol) if np.asarray(z_sol).ndim == 2 else np.asarray(z_sol)
    cont = _contingency(gt, sol)
    kg, ks = cont.shape
    n = int(cont.sum())

    G = nx.Graph()
    left = [f"g{u}" for u in range(kg)]
    right = [f"s{v}" for v in range(ks)]
    G.add_nodes_from(left, bipartite=0)
    G.add_nodes_from(right, bipartite=1)

    for i in range(kg):
        for j in range(ks):
            w = int(cont[i, j])
            if w > 0:
                G.add_edge(left[i], right[j], weight=w)

    matching = nx.algorithms.matching.max_weight_matching(G, maxcardinality=False, weight="weight")
    matched_sum = 0
    mapping: Dict[int, int] = {}
    for u, v in matching:
        if u[0] == "g":
            gi = int(u[1:])
            sj = int(v[1:])
        else:
            gi = int(v[1:])
            sj = int(u[1:])
        w = int(cont[gi, sj])
        matched_sum += w
        mapping[sj] = gi

    acc = matched_sum / n if n > 0 else 1.0
    return float(acc), mapping
