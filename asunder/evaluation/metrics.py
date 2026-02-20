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

from asunder.column_generation.master import compute_f_star
from asunder.utils.graph import partition_matrix_to_vector


def _relabel_consecutive(labels: np.ndarray) -> np.ndarray:
    _, inv = np.unique(labels, return_inverse=True)
    return inv.astype(int)

def _contingency(labels_a: np.ndarray, labels_b: np.ndarray) -> np.ndarray:
    labels_a = _relabel_consecutive(np.asarray(labels_a))
    labels_b = _relabel_consecutive(np.asarray(labels_b))
    ka = int(labels_a.max()) + 1
    kb = int(labels_b.max()) + 1
    M = np.zeros((ka, kb), dtype=np.int64)
    np.add.at(M, (labels_a, labels_b), 1)
    return M

def _entropy_from_counts(counts: np.ndarray, *, log_base: float = 2.0) -> float:
    total = float(np.sum(counts))
    if total <= 0:
        return 0.0
    p = counts.astype(float) / total
    p = p[p > 0]
    log = np.log(p) / np.log(log_base)
    return float(-np.sum(p * log))

def _mutual_information(cont: np.ndarray, *, log_base: float = 2.0) -> float:
    n = float(cont.sum())
    if n <= 0:
        return 0.0
    pij = cont.astype(float) / n
    pi = pij.sum(axis=1, keepdims=True)
    pj = pij.sum(axis=0, keepdims=True)
    mask = pij > 0
    log = np.log(pij[mask] / (pi @ pj)[mask]) / np.log(log_base)
    return float(np.sum(pij[mask] * log))

def optimality_gap(A, a, m, z_gt, z_sol, algo="modularity", tol=1e-10):
    """Compute relative optimality gap (percent) between reference and candidate."""
    best_sol = compute_f_star(A, a, m, z_sol, algo=algo)
    best_bound = compute_f_star(A, a, m, z_gt, algo=algo)
    opt_gap = (best_bound - best_sol) * 100 / (best_sol + tol)
    return opt_gap

def nmi(labels_gt: np.ndarray, labels_sol: np.ndarray, *, log_base: float = 2.0) -> float:
    """
    Newman's normalized mutual information: NMI = 2 I(X;Y) / (H(X) + H(Y)).
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
    """
    cont = _contingency(labels_gt, labels_sol).astype(np.int64)
    n = int(cont.sum())

    def comb2(x: np.ndarray) -> np.ndarray:
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
    """
    cont = _contingency(labels_gt, labels_sol)
    Hx = _entropy_from_counts(cont.sum(axis=1), log_base=log_base)
    Hy = _entropy_from_counts(cont.sum(axis=0), log_base=log_base)
    I = _mutual_information(cont, log_base=log_base)
    return float(Hx + Hy - 2.0 * I)

def nmi_sklearn(labels_gt, labels_sol) -> float:
    """Compute normalized mutual information via scikit-learn."""
    return float(normalized_mutual_info_score(labels_gt, labels_sol, average_method="arithmetic"))

def ari_sklearn(labels_gt, labels_sol) -> float:
    """Compute adjusted Rand index via scikit-learn."""
    return float(adjusted_rand_score(labels_gt, labels_sol))

def vi_sklearn(labels_gt, labels_sol, log_base=2.0) -> float:
    """Compute variation of information using sklearn mutual information."""
    mi = mutual_info_score(labels_gt, labels_sol)
    # sklearn MI uses natural log; convert if you want bits
    mi = mi / np.log(log_base)
    # entropies from marginals
    def H(x):
        _, c = np.unique(x, return_counts=True)
        p = c / c.sum()
        return float(-(p * (np.log(p) / np.log(log_base))).sum())
    return float(H(labels_gt) + H(labels_sol) - 2.0 * mi)

def permuted_accuracy(
    z_gt: np.ndarray, z_sol: np.ndarray
) -> Tuple[float, Dict[int, int]]:
    """
    Maximum fraction of correctly classified nodes under label permutation.
    Uses maximum-weight bipartite matching on the contingency table.
    Returns: (best_accuracy, mapping sol_label -> gt_label for matched labels).
    Unmatched sol labels are absent from the mapping and count as incorrect.
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
