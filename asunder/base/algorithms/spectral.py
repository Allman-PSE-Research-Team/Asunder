"""Dual-adjusted spectral bisection and partition refinement routines."""
from __future__ import annotations

import numpy as np

from asunder.base.utils.graph import (
    group_nodes_by_community,
    partition_matrix_to_vector,
    partition_vector_to_2d_matrix,
)


def make_dual_adjusted_matrix(A, a, m, dualW, symmetrize=True):
    """
    Build the dual-adjusted modularity matrix used by the subproblem.

    Parameters
    ----------
    A : array_like of float, shape (N, N)
        Adjacency or weighted adjacency matrix.
    a : array_like of float, shape (N,)
        Degree-like vector used in the modularity term.
    m : float
        Positive graph normalization constant.
    dualW : array_like of float, shape (N, N)
        Matrix of dual contributions aligned with the partition matrix.
    symmetrize : bool, optional
        If True, replace the matrix by its symmetric part. This is appropriate
        because the partition matrix is symmetric.

    Returns
    -------
    B : np.ndarray of float, shape (N, N)
        Dual-adjusted objective matrix.
    """
    if m <= 0:
        raise ValueError("m must be positive.")
    B = (A / m) - (np.outer(a, a) / (m * m)) - dualW
    if symmetrize:
        B = 0.5 * (B + B.T)
    return B


def group_vector_to_matrix(s):
    """
    Convert a two-way group vector into a partition matrix.

    Parameters
    ----------
    s : array_like of int, shape (N,)
        Two-way assignment vector. Positive entries are treated as one group;
        non-positive entries are treated as the other group.

    Returns
    -------
    P : np.ndarray of int, shape (N, N)
        Binary matrix where ``P[i, j] = 1`` when nodes ``i`` and ``j`` are in
        the same group.
    """
    s = np.asarray(s).reshape(-1)
    s = np.where(s > 0, 1, -1)
    return ((np.outer(s, s) + 1) // 2).astype(int)


def matrix_to_group_vector(P):
    """
    Convert a two-community partition matrix into a local ``{-1, 1}`` vector.

    Parameters
    ----------
    P : array_like of int or float, shape (N, N)
        Binary two-community partition matrix.

    Returns
    -------
    s : np.ndarray of int, shape (N,)
        Group vector whose first node's community is labeled ``1`` and whose
        other community is labeled ``-1``.
    """
    P = np.asarray(P)
    if P.ndim != 2 or P.shape[0] != P.shape[1]:
        raise ValueError("P must be a square 2D array.")
    return np.where(P[0, :] > 0.5, 1, -1).astype(int)


def vector_to_communities(s):
    """
    Convert a two-way group vector into nonempty local communities.

    Parameters
    ----------
    s : array_like of int, shape (N,)
        Two-way assignment vector.

    Returns
    -------
    communities : list[set[int]]
        Nonempty sets of local node indices.
    """
    s = np.asarray(s).reshape(-1)
    communities = []
    pos = set(np.flatnonzero(s > 0).tolist())
    neg = set(np.flatnonzero(s <= 0).tolist())
    if pos:
        communities.append(pos)
    if neg:
        communities.append(neg)
    return communities


def relabel_consecutive(labels):
    """
    Relabel community assignments to consecutive integers.

    Parameters
    ----------
    labels : array_like of hashable, shape (N,)
        Community label for each node.

    Returns
    -------
    relabeled : np.ndarray of int, shape (N,)
        Consecutive integer labels preserving equality structure.
    """
    labels = np.asarray(labels)
    mapper = {}
    relabeled = np.empty(labels.shape[0], dtype=int)
    next_label = 0

    for i, label in enumerate(labels.tolist()):
        if label not in mapper:
            mapper[label] = next_label
            next_label += 1
        relabeled[i] = mapper[label]

    return relabeled


def partition_objective(B, z):
    """
    Evaluate the dual-adjusted partition objective.

    Parameters
    ----------
    B : array_like of float, shape (N, N)
        Dual-adjusted objective matrix.
    z : array_like of int or float, shape (N, N)
        Partition matrix.

    Returns
    -------
    objective : float
        Value of ``sum(B * z)``.
    """
    B = np.asarray(B, dtype=float)
    z = np.asarray(z, dtype=float)
    if B.shape != z.shape:
        raise ValueError("B and z must have the same shape.")
    return float(np.sum(B * z))


def restricted_modularity_matrix(B, group):
    """
    Build the row-sum-corrected matrix for splitting one community.

    Parameters
    ----------
    B : array_like of float, shape (N, N)
        Dual-adjusted objective matrix.
    group : Iterable[int]
        Node indices in the community being split.

    Returns
    -------
    B_g : np.ndarray of float, shape (k, k)
        Restricted matrix whose row sums are zero.
    """
    B = np.asarray(B, dtype=float)
    group = list(group)
    B_block = B[np.ix_(group, group)].copy()
    row_sums = np.sum(B_block, axis=1)
    B_block[np.diag_indices_from(B_block)] -= row_sums
    return B_block


def compute_modularity_from_vector(B_g, s):
    """
    Evaluate a local two-way split under a restricted objective matrix.

    Parameters
    ----------
    B_g : array_like of float, shape (k, k)
        Restricted matrix for the community being split.
    s : array_like of int, shape (k,)
        Two-way assignment vector.

    Returns
    -------
    objective : float
        Local split value ``sum(B_g * z_s)``.
    """
    return partition_objective(np.asarray(B_g, dtype=float), group_vector_to_matrix(s))


def flip_vertex(s, vertex):
    """
    Flip one vertex in a two-way group vector.

    Parameters
    ----------
    s : array_like of int, shape (N,)
        Two-way assignment vector.
    vertex : int
        Local vertex index to flip.

    Returns
    -------
    s_new : np.ndarray of int, shape (N,)
        Assignment vector after the flip.
    """
    s_new = np.asarray(s, dtype=int).copy()
    s_new[vertex] = -s_new[vertex]
    return s_new


def refine_two_way_split(B_g, s_init, tol=1e-10):
    """
    Refine a two-way split by Kernighan-Lin-style single-node flips.

    Parameters
    ----------
    B_g : array_like of float, shape (k, k)
        Restricted matrix for the community being split.
    s_init : array_like of int, shape (k,)
        Initial two-way assignment vector.
    tol : float, optional
        Minimum improvement required to accept a sweep.

    Returns
    -------
    s : np.ndarray of int, shape (k,)
        Refined two-way assignment vector.
    value : float
        Local value of the refined split.
    """
    B_g = np.asarray(B_g, dtype=float)
    s = np.where(np.asarray(s_init).reshape(-1) > 0, 1, -1).astype(int)
    current_value = compute_modularity_from_vector(B_g, s)
    k = len(s)

    while True:
        moved = np.zeros(k, dtype=bool)
        temp_s = s.copy()
        temp_value = current_value
        best_s = s.copy()
        best_value = current_value

        for _ in range(k):
            best_delta = -np.inf
            best_vertex = None

            for vertex in range(k):
                if moved[vertex]:
                    continue
                candidate_s = flip_vertex(temp_s, vertex)
                candidate_value = compute_modularity_from_vector(B_g, candidate_s)
                delta = candidate_value - temp_value
                if delta > best_delta:
                    best_delta = delta
                    best_vertex = vertex

            if best_vertex is None:
                break

            temp_s = flip_vertex(temp_s, best_vertex)
            temp_value += best_delta
            moved[best_vertex] = True

            if temp_value > best_value + tol:
                best_s = temp_s.copy()
                best_value = temp_value

        if best_value <= current_value + tol:
            break

        s = best_s
        current_value = best_value

    return s, current_value


def modularity_maximization_matrix(G, B_g=None, initial_partition_matrix=None, tol=1e-10):
    """
    Refine a two-way partition matrix under a local objective.

    Parameters
    ----------
    G : object
        Graph-like object with ``number_of_nodes``. Only the node count is used.
    B_g : array_like of float, shape (N, N)
        Restricted objective matrix for the local split.
    initial_partition_matrix : array_like of int or float, shape (N, N), optional
        Initial two-way partition matrix. If omitted, all vertices start in one
        community.
    tol : float, optional
        Minimum improvement required to accept a move sweep.

    Returns
    -------
    final_partition_matrix : np.ndarray of int, shape (N, N)
        Refined two-way partition matrix.
    """
    if B_g is None:
        raise ValueError("B_g is required.")

    N = G.number_of_nodes() if hasattr(G, "number_of_nodes") else len(G)
    B_g = np.asarray(B_g, dtype=float)
    if B_g.shape != (N, N):
        raise ValueError("B_g shape must match the number of nodes in G.")

    if initial_partition_matrix is None:
        s = np.ones(N, dtype=int)
    else:
        s = matrix_to_group_vector(initial_partition_matrix)

    s, _ = refine_two_way_split(B_g, s, tol=tol)
    return group_vector_to_matrix(s)


def modularity_maximization_matrix_subset(G, B_g, subset, initial_partition_matrix=None, tol=1e-10):
    """
    Refine a two-way split for a specified subset of graph nodes.

    Parameters
    ----------
    G : object
        Graph-like object with ``nodes``. Used only to form a global label map.
    B_g : array_like of float, shape (k, k)
        Restricted matrix ordered according to ``subset``.
    subset : Iterable[int]
        Node indices in the same order used by ``B_g``.
    initial_partition_matrix : array_like of int or float, shape (k, k), optional
        Initial two-way partition matrix for the subset.
    tol : float, optional
        Minimum improvement required to accept a move sweep.

    Returns
    -------
    refined_partition_matrix : np.ndarray of int, shape (k, k)
        Refined partition matrix for the subset.
    global_partition : dict
        Dictionary mapping nodes to ``{-1, 0, 1}``, where nodes outside the
        subset receive ``0``.
    """
    subset = list(subset)
    B_g = np.asarray(B_g, dtype=float)
    if B_g.shape != (len(subset), len(subset)):
        raise ValueError("B_g shape must match the subset length.")

    if initial_partition_matrix is None:
        s = np.ones(len(subset), dtype=int)
    else:
        s = matrix_to_group_vector(initial_partition_matrix)

    s, _ = refine_two_way_split(B_g, s, tol=tol)
    refined_partition_matrix = group_vector_to_matrix(s)

    nodes = list(G.nodes()) if hasattr(G, "nodes") else list(range(max(subset) + 1))
    global_partition = {node: 0 for node in nodes}
    for local_idx, node in enumerate(subset):
        global_partition[node] = int(s[local_idx])

    return refined_partition_matrix, global_partition


def apply_group_split(z, group, s):
    """
    Apply a local two-way split to a global partition matrix.

    Parameters
    ----------
    z : array_like of int or float, shape (N, N)
        Current global partition matrix.
    group : Iterable[int]
        Global node indices being split.
    s : array_like of int, shape (k,)
        Local two-way assignment vector ordered according to ``group``.

    Returns
    -------
    z_new : np.ndarray of int, shape (N, N)
        Updated global partition matrix.
    """
    z_new = np.asarray(z).copy()
    group = list(group)
    z_new[np.ix_(group, group)] = group_vector_to_matrix(s)
    np.fill_diagonal(z_new, 1)
    return z_new.astype(int)


def spec_part_extra_bisect(
    A,
    a,
    m,
    dualW,
    z_curr,
    group,
    refinement=False,
    verbose=False,
    tol=1e-10,
):
    """
    Propose and accept one dual-adjusted spectral bisection if it improves value.

    Parameters
    ----------
    A : array_like of float, shape (N, N)
        Adjacency or weighted adjacency matrix.
    a : array_like of float, shape (N,)
        Degree-like vector used in the modularity term.
    m : float
        Positive graph normalization constant.
    dualW : array_like of float, shape (N, N)
        Matrix of dual contributions aligned with the partition matrix.
    z_curr : array_like of int or float, shape (N, N)
        Current global partition matrix.
    group : Iterable[int]
        Current community to bisect.
    refinement : bool, optional
        If True, run local two-way flip refinement before testing acceptance.
    verbose : bool, optional
        If True, print accepted and rejected split diagnostics.
    tol : float, optional
        Minimum objective improvement required to accept the split.

    Returns
    -------
    gp : np.ndarray of int, shape (k,)
        Local two-way assignment vector ordered according to ``group``.
    sub_obj_val : float
        Objective value after the accepted split, or the current value if the
        split is rejected.
    z_out : np.ndarray of int, shape (N, N)
        Updated global partition matrix if accepted; otherwise ``z_curr``.
    """
    B = make_dual_adjusted_matrix(A, a, m, dualW)
    z_curr = np.asarray(z_curr, dtype=int)
    group = list(group)
    current_obj = partition_objective(B, z_curr)

    if len(group) <= 1:
        return np.ones(len(group), dtype=int), current_obj, z_curr

    B_g = restricted_modularity_matrix(B, group)
    evvals, evvecs = np.linalg.eigh(B_g)
    idx_max = int(np.argmax(evvals))
    leading_value = float(evvals[idx_max])
    evmax = evvecs[:, idx_max]

    # # Alt idea: Grouping by the sum of the eigenvector instead of the sign because the vector is changed by the duals.
    # threshold = np.sum(evmax)

    # gp = (evmax >= threshold).astype(int)
    # gp[gp == 0] = -1

    # # Transform from grouping to binary matrix zii
    # zii = (np.outer(gp, gp) + 1) / 2.0

    gp = np.where(evmax >= 0, 1, -1).astype(int)
    if np.all(gp == gp[0]) or leading_value <= tol:
        if verbose == 1:
            print(f"Rejected group {group}: no positive spectral split.")
        return gp, current_obj, z_curr

    if refinement:
        gp, _ = refine_two_way_split(B_g, gp, tol=tol)
        if np.all(gp == gp[0]):
            if verbose == 1:
                print(f"Rejected group {group}: refinement collapsed the split.")
            return gp, current_obj, z_curr

    z_out = apply_group_split(z_curr, group, gp)
    candidate_obj = partition_objective(B, z_out)

    if candidate_obj <= current_obj + tol:
        if verbose == 1:
            delta = candidate_obj - current_obj
            print(f"Rejected group {group}: delta={delta:.6g}.")
        return gp, current_obj, z_curr

    if verbose == 1:
        delta = candidate_obj - current_obj
        print(f"Accepted group {group}: delta={delta:.6g}.")
    return gp, candidate_obj, z_out


def best_single_node_move(B, labels, current_obj, allow_singletons=True, tol=1e-10):
    """
    Find the best improving single-node move across existing communities.

    Parameters
    ----------
    B : array_like of float, shape (N, N)
        Dual-adjusted objective matrix.
    labels : array_like of int, shape (N,)
        Current community labels.
    current_obj : float
        Objective value of the current labeling.
    allow_singletons : bool, optional
        If True, also test moving a node into a new singleton community.
    tol : float, optional
        Minimum improvement required to accept a move.

    Returns
    -------
    best_labels : np.ndarray of int, shape (N,)
        Labels after the best move, or the original labels if no move improves.
    best_obj : float
        Objective value after the best move.
    improved : bool
        True if an improving move was found.
    """
    B = np.asarray(B, dtype=float)
    labels = relabel_consecutive(labels)
    N = labels.shape[0]
    best_labels = labels.copy()
    best_obj = float(current_obj)
    next_label = int(np.max(labels)) + 1

    for node in range(N):
        old_label = labels[node]
        targets = [label for label in np.unique(labels) if label != old_label]
        if allow_singletons:
            targets.append(next_label)

        for target in targets:
            candidate_labels = labels.copy()
            candidate_labels[node] = target
            candidate_labels = relabel_consecutive(candidate_labels)
            candidate_z = partition_vector_to_2d_matrix(candidate_labels)
            candidate_obj = partition_objective(B, candidate_z)

            if candidate_obj > best_obj + tol:
                best_obj = candidate_obj
                best_labels = candidate_labels

    return best_labels, best_obj, bool(best_obj > current_obj + tol)


def greedy_global_refinement(B, z_init, allow_singletons=True, tol=1e-10, max_moves=None):
    """
    Refine a full partition by greedy single-node moves.

    Parameters
    ----------
    B : array_like of float, shape (N, N)
        Dual-adjusted objective matrix.
    z_init : array_like of int or float, shape (N, N)
        Initial partition matrix.
    allow_singletons : bool, optional
        If True, allow moving a node into a new singleton community.
    tol : float, optional
        Minimum improvement required to accept a move.
    max_moves : int, optional
        Maximum number of accepted moves. If omitted, at most ``N ** 2`` moves
        are accepted.

    Returns
    -------
    z : np.ndarray of int, shape (N, N)
        Refined partition matrix.
    obj : float
        Objective value of the refined partition.
    changed : bool
        True if at least one node move was accepted.
    """
    B = np.asarray(B, dtype=float)
    labels = partition_matrix_to_vector(z_init)
    obj = partition_objective(B, partition_vector_to_2d_matrix(labels))
    N = B.shape[0]
    if max_moves is None:
        max_moves = N * N

    changed = False
    for _ in range(max_moves):
        labels_new, obj_new, improved = best_single_node_move(
            B,
            labels,
            obj,
            allow_singletons=allow_singletons,
            tol=tol,
        )
        if not improved:
            break
        labels = labels_new
        obj = obj_new
        changed = True

    return partition_vector_to_2d_matrix(labels), obj, changed


def full_spectral_bisection(
    A,
    a,
    m,
    dualW,
    refinement=False,
    global_refinement=True,
    allow_singletons=True,
    max_outer_passes=None,
    tol=1e-10,
    verbose=False,
):
    """
    Build a dual-adjusted partition using repeated spectral bisection.

    Parameters
    ----------
    A : array_like of float, shape (N, N)
        Adjacency or weighted adjacency matrix.
    a : array_like of float, shape (N,)
        Degree-like vector used in the modularity term.
    m : float
        Positive graph normalization constant.
    dualW : array_like of float, shape (N, N)
        Matrix of dual contributions aligned with the partition matrix.
    refinement : bool, optional
        If True, refine each proposed two-way split before acceptance.
    global_refinement : bool, optional
        If True, run global single-node-move refinement after each bisection
        pass. This permits correction of earlier splits.
    allow_singletons : bool, optional
        If True, global refinement may move a node into a singleton community.
    max_outer_passes : int, optional
        Maximum number of bisection passes. If omitted, at most ``N`` passes
        are run.
    tol : float, optional
        Minimum improvement required to accept a split or move.
    verbose : bool, optional
        If True, print accepted and rejected move diagnostics.

    Returns
    -------
    z_curr : np.ndarray of int, shape (N, N)
        Final partition matrix.
    sub_obj_val : float
        Final dual-adjusted subproblem objective value.
    """
    B = make_dual_adjusted_matrix(A, a, m, dualW)
    N = B.shape[0]
    z_curr = np.ones((N, N), dtype=int)
    obj = partition_objective(B, z_curr)

    if max_outer_passes is None:
        max_outer_passes = N

    for pass_idx in range(max_outer_passes):
        changed = False
        _, communities = group_nodes_by_community(z_curr)

        for community in communities:
            if len(community) <= 1:
                continue
            _, candidate_obj, z_candidate = spec_part_extra_bisect(
                A,
                a,
                m,
                dualW,
                z_curr=z_curr,
                group=community,
                refinement=refinement,
                verbose=verbose,
                tol=tol,
            )
            if candidate_obj > obj + tol:
                z_curr = z_candidate
                obj = candidate_obj
                changed = True

        if global_refinement:
            z_refined, obj_refined, refined = greedy_global_refinement(
                B,
                z_curr,
                allow_singletons=allow_singletons,
                tol=tol,
            )
            if obj_refined > obj + tol:
                z_curr = z_refined
                obj = obj_refined
                changed = True
                if verbose == 1:
                    print(f"Accepted global refinement on pass {pass_idx}.")
            elif refined:
                z_curr = z_refined
                obj = obj_refined

        if not changed:
            break

    return z_curr.astype(int), float(obj)
    