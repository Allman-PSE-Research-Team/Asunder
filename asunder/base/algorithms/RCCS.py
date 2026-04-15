"""Reduced Cost Community Search: A greedy and local search heuristic for finding commiunities that maximize the reduced cost."""

from __future__ import annotations

import numpy as np

from asunder.base.utils.graph import partition_vector_to_2d_matrix


def _canonicalize_labels(labels: np.ndarray | list[int]) -> np.ndarray:
    """
    Relabel a partition vector to contiguous integer community IDs.

    The relabeling preserves community membership but replaces the original
    labels with ``0, 1, ..., K-1`` in order of first appearance. This is
    useful for normalizing equivalent partitions that use different label
    names.

    Parameters
    ----------
    labels : ndarray or list of int
        One-dimensional array-like of community labels.

    Returns
    -------
    ndarray of int
        Canonicalized label vector with contiguous integer labels.

    Raises
    ------
    ValueError
        If ``labels`` is not one-dimensional.

    Notes
    -----
    Two label vectors that induce the same partition may differ only by label
    names. This function removes that ambiguity.
    """
    arr = np.asarray(labels)
    if arr.ndim != 1:
        raise ValueError("labels must be a 1D array-like")
    out = np.empty(arr.shape[0], dtype=int)
    remap: dict[int, int] = {}
    nxt = 0
    for i, raw in enumerate(arr.tolist()):
        key = int(raw)
        if key not in remap:
            remap[key] = nxt
            nxt += 1
        out[i] = remap[key]
    return out

def _validate_adjacency(adjacency: np.ndarray, symmetrize_inputs: bool) -> np.ndarray:
    """
    Validate and optionally symmetrize an adjacency matrix.

    The matrix must be square, finite, and have strictly positive total
    weight. When ``symmetrize_inputs`` is ``True``, the returned matrix is
    replaced by ``0.5 * (A + A.T)``.

    Parameters
    ----------
    adjacency : ndarray, shape (N, N)
        Candidate adjacency matrix.
    symmetrize_inputs : bool
        Whether to replace the input by its symmetric average.

    Returns
    -------
    ndarray of float
        Validated adjacency matrix.

    Raises
    ------
    ValueError
        If the matrix is not square, contains non-finite values, is not
        symmetric when required, or has non-positive total weight.
    """
    A = np.asarray(adjacency, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("adjacency must be a square 2D NumPy array")
    if not np.all(np.isfinite(A)):
        raise ValueError("adjacency contains non-finite values")
    if symmetrize_inputs:
        A = 0.5 * (A + A.T)
    elif not np.allclose(A, A.T, atol=1e-12):
        raise ValueError("adjacency must be symmetric when symmetrize_inputs=False")
    m = float(A.sum())
    if m <= 0.0:
        raise ValueError("adjacency total weight m must be positive")
    return A


def _parse_duals_to_matrix_and_constant(
    duals: dict[str, object], n: int, symmetrize_inputs: bool
) -> tuple[np.ndarray, float]:
    """
    Convert heterogeneous dual inputs into a pairwise matrix and a constant.

    Scalar dual values contribute only to the constant term. One-dimensional
    dual arrays are interpreted as node-wise quantities and converted into a
    symmetric pairwise contribution via

    ``0.5 * (d_i + d_j)``.

    Two-dimensional dual arrays are interpreted directly as pairwise terms.

    Parameters
    ----------
    duals : dict[str, object]
        Mapping from dual names to scalar, vector, matrix, or ``None`` values.
    n : int
        Expected problem dimension.
    symmetrize_inputs : bool
        Whether to symmetrize two-dimensional dual arrays.

    Returns
    -------
    dualW : ndarray of float
        Aggregated pairwise dual contribution matrix of shape ``(n, n)``.
    constant_terms : float
        Sum of all scalar-valued constant dual contributions.

    Raises
    ------
    ValueError
        If a vector has length different from ``n``, a matrix has shape
        different from ``(n, n)``, a matrix is not symmetric when required,
        or any array contains non-finite values.
    """
    dualW = np.zeros((n, n), dtype=float)
    constant_terms = 0.0

    for _, dual in duals.items():
        if dual is None:
            continue
        if np.isscalar(dual):
            constant_terms += float(dual)
            continue

        arr = np.asarray(dual, dtype=float)
        if not np.all(np.isfinite(arr)):
            raise ValueError("duals contains non-finite array values")

        if arr.ndim == 0:
            constant_terms += float(arr)
        elif arr.ndim == 1:
            if arr.shape[0] != n:
                raise ValueError("1D dual vectors must have length N")
            dualW += 0.5 * (arr[:, None] + arr[None, :])
        elif arr.ndim == 2:
            if arr.shape != (n, n):
                raise ValueError("2D dual arrays must have shape (N, N)")
            if symmetrize_inputs:
                arr = 0.5 * (arr + arr.T)
            elif not np.allclose(arr, arr.T, atol=1e-12):
                raise ValueError("2D dual arrays must be symmetric when symmetrize_inputs=False")
            dualW += arr
        else:
            raise ValueError("dual values must be scalar, 1D vector, or 2D matrix")
    return dualW, float(constant_terms)


def _build_objective_matrix_W(
    adjacency: np.ndarray, duals: dict[str, object], symmetrize_inputs: bool
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """
    Build the reduced-cost objective matrix used by the search heuristic.

    The returned matrix is

    ``W = modularity_base - dualW``,

    where ``modularity_base`` is the dense modularity kernel

    ``A / m - (a a^T) / m^2``.

    Parameters
    ----------
    adjacency : ndarray
        Problem adjacency matrix.
    duals : dict[str, object]
        Dual information to be folded into the reduced-cost objective.
    symmetrize_inputs : bool
        Whether to symmetrize adjacency and matrix-valued dual inputs.

    Returns
    -------
    W : ndarray of float
        Dense reduced-cost objective matrix.
    modularity_base : ndarray of float
        Modularity contribution before subtracting dual terms.
    dualW : ndarray of float
        Dense pairwise matrix derived from dual inputs.
    constant_terms : float
        Scalar dual contribution independent of the partition.
    m : float
        Total adjacency weight.
    """
    A = _validate_adjacency(adjacency, symmetrize_inputs=symmetrize_inputs)
    n = A.shape[0]
    dualW, constant_terms = _parse_duals_to_matrix_and_constant(
        duals, n, symmetrize_inputs=symmetrize_inputs
    )
    a = A.sum(axis=1)
    m = float(A.sum())
    modularity_base = (A / m) - (np.outer(a, a) / (m * m))
    W = modularity_base - dualW
    return W, modularity_base, dualW, constant_terms, m


def _score_labels_from_W(labels: np.ndarray, W: np.ndarray, constant_terms: float) -> float:
    """
    Score a partition exactly from a precomputed objective matrix.

    The score is the sum of all within-community entries of ``W`` minus the
    provided constant term.

    Parameters
    ----------
    labels : ndarray
        Community label vector.
    W : ndarray
        Dense reduced-cost objective matrix.
    constant_terms : float
        Partition-independent scalar contribution to subtract.

    Returns
    -------
    float
        Exact reduced-cost value for the given partition.
    """
    score_no_const = 0.0
    n_comm = int(labels.max()) + 1 if labels.size else 0
    for c in range(n_comm):
        idx = np.where(labels == c)[0]
        if idx.size == 0:
            continue
        score_no_const += float(W[np.ix_(idx, idx)].sum())
    return score_no_const - constant_terms


def _build_members_and_S(labels: np.ndarray, W: np.ndarray) -> tuple[list[set[int]], np.ndarray]:
    """
    Build community membership sets and node-to-community score sums.

    For each community ``c``, the matrix ``S`` stores

    ``S[i, c] = sum_{j in c} W[i, j]``.

    This structure supports efficient move-gain evaluation.

    Parameters
    ----------
    labels : ndarray
        Community label vector.
    W : ndarray
        Dense reduced-cost objective matrix.

    Returns
    -------
    members : list of set of int
        Community membership sets indexed by community ID.
    S : ndarray of float
        Node-to-community score-sum matrix of shape ``(N, K)``.
    """
    n = labels.shape[0]
    n_comm = int(labels.max()) + 1 if labels.size else 0
    members: list[set[int]] = []
    S = np.zeros((n, n_comm), dtype=float)
    for c in range(n_comm):
        idx = np.where(labels == c)[0]
        members.append(set(int(i) for i in idx.tolist()))
        if idx.size > 0:
            S[:, c] = W[:, idx].sum(axis=1)
    return members, S


def _compute_move_gain(
    node: int, from_comm: int, to_comm: int, S: np.ndarray, W_diag: np.ndarray
) -> float:
    """
    Compute the exact objective gain of moving one node between communities.

    The move gain is evaluated from the cached node-to-community sums in
    ``S`` and the diagonal of ``W`` without recomputing the full objective.

    Parameters
    ----------
    node : int
        Node index to move.
    from_comm : int
        Current community of the node.
    to_comm : int
        Target community of the node. This may refer to an existing community
        or to a newly created singleton community when handled upstream.
    S : ndarray
        Node-to-community score-sum matrix.
    W_diag : ndarray
        Diagonal of the dense objective matrix ``W``.

    Returns
    -------
    float
        Exact score change obtained by performing the move.
    """
    if from_comm == to_comm:
        return 0.0
    s_to = 0.0 if to_comm >= S.shape[1] else float(S[node, to_comm])
    s_from_excl = float(S[node, from_comm] - W_diag[node])
    return 2.0 * (s_to - s_from_excl)


def _rebuild_structures_after_empty_comm(
    labels: np.ndarray, W: np.ndarray
) -> tuple[np.ndarray, list[set[int]], np.ndarray]:
    """
    Rebuild community data structures after a community becomes empty.

    This function canonicalizes labels, reconstructs membership sets, and
    recomputes the cached node-to-community score sums.

    Parameters
    ----------
    labels : ndarray
        Current label vector, possibly with a missing community index.
    W : ndarray
        Dense reduced-cost objective matrix.

    Returns
    -------
    labels : ndarray
        Canonicalized label vector.
    members : list of set of int
        Rebuilt community membership sets.
    S : ndarray
        Rebuilt node-to-community score-sum matrix.
    """
    labels = _canonicalize_labels(labels)
    members, S = _build_members_and_S(labels, W)
    return labels, members, S


def _tabu_sweep(
    labels: np.ndarray,
    current_score: float,
    W: np.ndarray,
    constant_terms: float,
    *,
    max_tabu_steps: int,
    tabu_tenure: int,
    allow_non_improving_every: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, float, dict[str, object]]:
    """
    Run one tabu-search sweep over single-node moves.

    At each step, the method evaluates moves from every node to every existing
    community, plus a candidate new singleton community. Improving moves are
    preferred. Periodic non-improving moves may also be accepted, subject to
    the tabu restrictions and aspiration rule.

    Parameters
    ----------
    labels : ndarray
        Initial partition for the sweep.
    current_score : float
        Exact reduced-cost score of ``labels``.
    W : ndarray
        Dense reduced-cost objective matrix.
    constant_terms : float
        Partition-independent scalar contribution.
    max_tabu_steps : int
        Maximum number of tabu iterations in the sweep.
    tabu_tenure : int
        Number of steps for which the reverse move is tabu.
    allow_non_improving_every : int
        Frequency at which the best available non-improving move may be taken.
        If zero, non-improving moves are never forced.
    rng : numpy.random.Generator
        Random number generator.

    Returns
    -------
    best_labels : ndarray
        Best partition encountered during the sweep.
    best_score : float
        Best score encountered during the sweep.
    meta : dict[str, object]
        Sweep diagnostics, including number of steps, moves, non-improving
        moves, and best score reached.

    Notes
    -----
    The running score is periodically corrected against an exact recomputation
    to guard against numerical drift.
    """
    n = labels.shape[0]
    W_diag = np.diag(W)
    members, S = _build_members_and_S(labels, W)
    n_comm = len(members)

    best_labels = labels.copy()
    best_score = float(current_score)
    score = float(current_score)

    tabu_until: dict[tuple[int, int], int] = {}
    n_moves = 0
    n_non_improving = 0

    for step in range(1, max_tabu_steps + 1):
        best_improving: tuple[float, int, int, int] | None = None
        best_any: tuple[float, int, int, int] | None = None

        for i in range(n):
            src = int(labels[i])
            # Existing target communities + new singleton community.
            for dst in range(n_comm + 1):
                if dst == src:
                    continue
                gain = _compute_move_gain(i, src, dst, S, W_diag)
                cand_score = score + gain
                is_tabu = bool(dst < n_comm and tabu_until.get((i, dst), 0) > step)
                aspiration = cand_score > best_score + 1e-12
                if is_tabu and not aspiration:
                    continue

                cand = (gain, i, src, dst)
                if best_any is None or gain > best_any[0] + 1e-15:
                    best_any = cand
                if gain > 1e-12 and (best_improving is None or gain > best_improving[0] + 1e-15):
                    best_improving = cand

        chosen: tuple[float, int, int, int] | None = None
        if best_improving is not None:
            chosen = best_improving
        elif (
            best_any is not None
            and allow_non_improving_every > 0
            and step % allow_non_improving_every == 0
        ):
            chosen = best_any

        if chosen is None:
            break

        gain, i, src, dst = chosen
        if dst == n_comm:
            members.append(set())
            S = np.concatenate([S, np.zeros((n, 1), dtype=float)], axis=1)
            n_comm += 1
            dst = n_comm - 1

        members[src].remove(i)
        members[dst].add(i)
        labels[i] = dst
        wi = W[:, i]
        S[:, src] -= wi
        S[:, dst] += wi
        score += gain
        n_moves += 1
        if gain <= 1e-12:
            n_non_improving += 1

        tabu_until[(i, src)] = step + tabu_tenure

        if len(members[src]) == 0:
            labels, members, S = _rebuild_structures_after_empty_comm(labels, W)
            n_comm = len(members)
            tabu_until.clear()

        if score > best_score + 1e-12:
            best_score = score
            best_labels = labels.copy()

        # Numerical drift guard with exact score from W at checkpoints.
        if step % 200 == 0:
            exact_score = _score_labels_from_W(labels, W, constant_terms)
            if abs(exact_score - score) > 1e-8:
                score = exact_score
                labels, members, S = _rebuild_structures_after_empty_comm(labels, W)
                n_comm = len(members)
                tabu_until.clear()

    return best_labels, best_score, {
        "n_steps": step if "step" in locals() else 0,
        "n_moves": n_moves,
        "n_non_improving_moves": n_non_improving,
        "best_score_after_sweep": best_score,
    }


def _greedy_merge_phase(
    labels: np.ndarray, current_score: float, W: np.ndarray, constant_terms: float
) -> tuple[np.ndarray, float, int]:
    """
    Greedily merge communities while the merge gain is positive.

    For each pair of communities, the merge gain is computed from the sum of
    cross-community weights. The best positive merge is applied repeatedly
    until no improving merge remains.

    Parameters
    ----------
    labels : ndarray
        Current partition.
    current_score : float
        Current exact reduced-cost score.
    W : ndarray
        Dense reduced-cost objective matrix.
    constant_terms : float
        Partition-independent scalar contribution.

    Returns
    -------
    labels : ndarray
        Partition after greedy merging.
    score : float
        Exact score after the final merge sequence.
    merge_count : int
        Number of merges applied.
    """
    labels = _canonicalize_labels(labels)
    merge_count = 0
    score = float(current_score)

    while True:
        n_comm = int(labels.max()) + 1 if labels.size else 0
        if n_comm <= 1:
            break
        members = [np.where(labels == c)[0] for c in range(n_comm)]

        best_gain = 0.0
        best_pair: tuple[int, int] | None = None
        for a in range(n_comm):
            ia = members[a]
            if ia.size == 0:
                continue
            for b in range(a + 1, n_comm):
                ib = members[b]
                if ib.size == 0:
                    continue
                cross = float(W[np.ix_(ia, ib)].sum())
                gain = 2.0 * cross
                if gain > best_gain + 1e-12:
                    best_gain = gain
                    best_pair = (a, b)
        if best_pair is None:
            break

        a, b = best_pair
        labels[labels == b] = a
        labels = _canonicalize_labels(labels)
        score += best_gain
        merge_count += 1

    score = _score_labels_from_W(labels, W, constant_terms)
    return labels, score, merge_count


def _attempt_split_weakest_community(
    labels: np.ndarray,
    current_score: float,
    W: np.ndarray,
    constant_terms: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, float, bool, dict[str, object]]:
    """
    Attempt an improving split of the weakest community.

    The weakest community is chosen by lowest average internal cohesion.
    A two-way split is then seeded by the least compatible node pair inside
    that community and completed greedily. The split is accepted only if it
    strictly improves the exact reduced-cost score.

    Parameters
    ----------
    labels : ndarray
        Current partition.
    current_score : float
        Current exact reduced-cost score.
    W : ndarray
        Dense reduced-cost objective matrix.
    constant_terms : float
        Partition-independent scalar contribution.
    rng : numpy.random.Generator
        Random number generator for tie-breaking and ordering.

    Returns
    -------
    labels : ndarray
        Updated partition if the split is accepted, otherwise the input
        partition.
    score : float
        Updated score if the split is accepted, otherwise the input score.
    split_applied : bool
        Whether an improving split was found and applied.
    meta : dict[str, object]
        Diagnostic information describing the accepted split or the reason no
        split was applied.
    """
    labels = _canonicalize_labels(labels)
    n_comm = int(labels.max()) + 1 if labels.size else 0
    candidate_comms: list[tuple[float, int, np.ndarray]] = []
    for c in range(n_comm):
        idx = np.where(labels == c)[0]
        if idx.size < 3:
            continue
        coh = float(W[np.ix_(idx, idx)].sum()) / float(idx.size * idx.size)
        candidate_comms.append((coh, c, idx))
    if not candidate_comms:
        return labels, current_score, False, {"reason": "no_splittable_community"}

    candidate_comms.sort(key=lambda x: x[0])  # weakest cohesion first
    _, c_weak, idx = candidate_comms[0]

    # Farthest-seed split: pick pair with minimum pairwise affinity.
    min_pair = None
    min_w = np.inf
    for p in range(idx.size):
        i = int(idx[p])
        for q in range(p + 1, idx.size):
            j = int(idx[q])
            wij = float(W[i, j])
            if wij < min_w:
                min_w = wij
                min_pair = (i, j)
    if min_pair is None:
        return labels, current_score, False, {"reason": "seed_pair_not_found"}

    s1, s2 = min_pair
    g1 = {s1}
    g2 = {s2}
    remaining = [int(i) for i in idx.tolist() if int(i) not in {s1, s2}]
    rng.shuffle(remaining)

    for node in remaining:
        gain1 = 2.0 * float(np.sum([W[node, j] for j in g1])) + float(W[node, node])
        gain2 = 2.0 * float(np.sum([W[node, j] for j in g2])) + float(W[node, node])
        if gain1 > gain2 + 1e-12:
            g1.add(node)
        elif gain2 > gain1 + 1e-12:
            g2.add(node)
        else:
            (g1 if rng.random() < 0.5 else g2).add(node)

    if not g1 or not g2:
        return labels, current_score, False, {"reason": "degenerate_split"}

    new_labels = labels.copy()
    new_comm = n_comm
    for node in g2:
        new_labels[node] = new_comm
    new_labels = _canonicalize_labels(new_labels)
    new_score = _score_labels_from_W(new_labels, W, constant_terms)

    if new_score > current_score + 1e-12:
        return new_labels, new_score, True, {"split_comm": int(c_weak), "new_size": int(len(g2))}
    return labels, current_score, False, {"reason": "no_improvement"}


def _shake_partition(
    labels: np.ndarray,
    W: np.ndarray,
    rng: np.random.Generator,
    shake_fraction: float,
) -> np.ndarray:
    """
    Randomly perturb weakly attached nodes to diversify the search.

    Nodes are ranked by their internal attachment to their current community.
    A fraction of the weakest nodes is reassigned uniformly at random to a
    different existing community.

    Parameters
    ----------
    labels : ndarray
        Current partition.
    W : ndarray
        Dense reduced-cost objective matrix.
    rng : numpy.random.Generator
        Random number generator.
    shake_fraction : float
        Fraction of nodes to perturb.

    Returns
    -------
    ndarray of int
        Canonicalized perturbed partition.
    """
    labels = _canonicalize_labels(labels)
    n = labels.shape[0]
    n_comm = int(labels.max()) + 1 if labels.size else 0
    if n <= 1 or n_comm <= 1:
        return labels

    members, S = _build_members_and_S(labels, W)
    W_diag = np.diag(W)
    weak_scores = []
    for i in range(n):
        c = int(labels[i])
        internal_wo_self = float(S[i, c] - W_diag[i])
        weak_scores.append((internal_wo_self, i))
    weak_scores.sort(key=lambda t: t[0])

    n_shake = max(1, int(round(shake_fraction * n)))
    picked = [i for _, i in weak_scores[:n_shake]]

    for i in picked:
        src = int(labels[i])
        choices = [c for c in range(n_comm) if c != src]
        if not choices:
            continue
        dst = int(rng.choice(np.array(choices, dtype=int)))
        labels[i] = dst

    return _canonicalize_labels(labels)


def compute_modularity_reduced_cost(
    adjacency: np.ndarray,
    duals: dict[str, object],
    labels: np.ndarray | list[int],
    *,
    symmetrize_inputs: bool = True,
    return_details: bool = False,
) -> float | dict[str, object]:
    """
    Compute the exact modularity-based reduced cost of a partition.

    The reduced cost is evaluated as the modularity contribution of the
    induced co-clustering matrix minus the pairwise dual contribution and
    minus all partition-independent constant dual terms.

    Parameters
    ----------
    adjacency : ndarray
        Adjacency matrix of the graph.
    duals : dict[str, object]
        Dual information represented as scalars, vectors, matrices, or
        ``None`` values.
    labels : ndarray or list of int
        Community label vector defining the partition.
    symmetrize_inputs : bool, default=True
        Whether to symmetrize adjacency and matrix-valued dual inputs.
    return_details : bool, default=False
        Whether to return a diagnostics dictionary instead of only the reduced
        cost value.

    Returns
    -------
    float or dict[str, object]
        If ``return_details`` is ``False``, the exact reduced cost.
        Otherwise, a dictionary containing the reduced cost, modularity term,
        dual partition term, constant terms, total adjacency weight, and the
        aggregated pairwise dual matrix.

    Raises
    ------
    ValueError
        If the adjacency matrix is invalid or if its dimension is inconsistent
        with the label vector.

    Notes
    -----
    This function is the exact scorer and can be used independently of the
    search heuristic.
    """

    labels_arr = _canonicalize_labels(labels)
    n = labels_arr.shape[0]

    A = _validate_adjacency(adjacency, symmetrize_inputs=symmetrize_inputs)
    if A.shape[0] != n:
        raise ValueError("labels length must equal adjacency dimension")

    dualW, constant_terms = _parse_duals_to_matrix_and_constant(
        duals, n, symmetrize_inputs=symmetrize_inputs
    )

    a = A.sum(axis=1)
    m = float(A.sum())
    Z = partition_vector_to_2d_matrix(labels_arr)
    modularity_base = (A / m) - (np.outer(a, a) / (m * m))

    modularity_term = float(np.sum(modularity_base * Z))
    dual_partition_term = float(np.sum(dualW * Z))
    reduced_cost = modularity_term - dual_partition_term - float(constant_terms)

    if not return_details:
        return reduced_cost
    return {
        "reduced_cost": float(reduced_cost),
        "modularity_term": float(modularity_term),
        "dual_partition_term": float(dual_partition_term),
        "constant_terms": float(constant_terms),
        "m": float(m),
        "dualW": dualW,
    }


def search_partition_by_reduced_cost(
    adjacency: np.ndarray,
    duals: dict[str, object],
    current_labels: np.ndarray | list[int],
    *,
    n_restarts: int = 12,
    max_local_passes: int = 25,
    max_tabu_steps: int = 5000,
    tabu_tenure: int = 7,
    allow_non_improving_every: int = 20,
    shake_rounds: int = 3,
    shake_fraction: float = 0.06,
    split_trigger_no_improve_passes: int = 3,
    random_seed: int | None = None,
) -> dict[str, object]:
    """
    Heuristically search for a partition with high reduced cost.

    The search uses a multistart strategy with a warm start from the supplied
    partition, followed by repeated rounds of tabu-based node moves, greedy
    community merges, occasional community splitting, and shake-based
    diversification.

    Parameters
    ----------
    adjacency : ndarray, shape (N, N)
        Adjacency matrix of the graph.
    duals : dict[str, object]
        Dual information represented as scalars, vectors, matrices, or
        ``None`` values.
    current_labels : ndarray or list of int
        Initial partition used both as the initial incumbent and as a seed for
        some restarts.
    n_restarts : int, default=12
        Number of restart trajectories.
    max_local_passes : int, default=25
        Maximum number of local search passes per shake round.
    max_tabu_steps : int, default=5000
        Maximum number of tabu iterations per tabu sweep.
    tabu_tenure : int, default=7
        Tabu tenure used in the node-move search.
    allow_non_improving_every : int, default=20
        Frequency at which the best non-improving move may be accepted.
    shake_rounds : int, default=3
        Number of diversification rounds after the initial round.
    shake_fraction : float, default=0.06
        Fraction of nodes perturbed in each shake step.
    split_trigger_no_improve_passes : int, default=3
        Number of consecutive non-improving passes required before attempting
        a split.
    random_seed : int, default=0
        Seed for the internal random number generator.

    Returns
    -------
    dict[str, object]
        Dictionary containing the best label vector found, its reduced cost,
        the initial reduced cost, the improvement, detailed restart history,
        the number of restarts, and the effective search configuration.

    Raises
    ------
    ValueError
        If any search parameter is outside its allowed range or if the initial
        labels are inconsistent with the adjacency dimension.

    Notes
    -----
    The objective is fully defined by ``adjacency`` and ``duals``. The
    ``current_labels`` argument serves only as a warm start and baseline for
    the heuristic search.
    """
    if n_restarts <= 0:
        raise ValueError("n_restarts must be >= 1")
    if max_local_passes <= 0:
        raise ValueError("max_local_passes must be >= 1")
    if max_tabu_steps <= 0:
        raise ValueError("max_tabu_steps must be >= 1")
    if tabu_tenure < 0:
        raise ValueError("tabu_tenure must be >= 0")
    if allow_non_improving_every < 0:
        raise ValueError("allow_non_improving_every must be >= 0")
    if shake_rounds < 0:
        raise ValueError("shake_rounds must be >= 0")
    if not (0.0 <= shake_fraction <= 1.0):
        raise ValueError("shake_fraction must be in [0, 1]")
    if split_trigger_no_improve_passes <= 0:
        raise ValueError("split_trigger_no_improve_passes must be >= 1")

    rng = np.random.default_rng(random_seed)

    if current_labels is None:
        current_labels = np.arange(adjacency.shape[0])

    labels0 = _canonicalize_labels(current_labels)
    n = labels0.shape[0]

    W, _, _, constant_terms, _ = _build_objective_matrix_W(
        adjacency, duals, symmetrize_inputs=True
    )
    if W.shape[0] != n:
        raise ValueError("current_labels length must equal adjacency dimension")

    initial_reduced_cost = float(
        compute_modularity_reduced_cost(
            adjacency, duals, labels0, symmetrize_inputs=True, return_details=False
        )
    )

    global_best_labels = labels0.copy()
    global_best_score = float(initial_reduced_cost)
    history: list[dict[str, object]] = []

    for restart in range(n_restarts):
        if restart == 0:
            seed_type = "current"
            labels_seed = labels0.copy()
        elif restart == 1:
            seed_type = "singleton"
            labels_seed = np.arange(n, dtype=int)
        else:
            seed_type = "perturbed"
            labels_seed = labels0.copy()
            n_pert = max(1, int(round(0.15 * n)))
            idx = rng.choice(n, size=n_pert, replace=False)
            k = int(labels_seed.max()) + 1
            for i in idx.tolist():
                labels_seed[i] = int(rng.integers(0, max(k, 1)))
            labels_seed = _canonicalize_labels(labels_seed)

        restart_log: dict[str, object] = {
            "restart": restart,
            "seed_type": seed_type,
            "rounds": [],
        }

        current = labels_seed.copy()
        current_score = float(
            compute_modularity_reduced_cost(
                adjacency, duals, current, symmetrize_inputs=True, return_details=False
            )
        )
        best_restart_labels = current.copy()
        best_restart_score = current_score

        for round_idx in range(shake_rounds + 1):
            no_improve_passes = 0
            round_log: dict[str, object] = {
                "round": round_idx,
                "passes": [],
                "start_score": current_score,
            }

            for pass_idx in range(max_local_passes):
                before = current_score
                current, current_score, tabu_meta = _tabu_sweep(
                    labels=current,
                    current_score=current_score,
                    W=W,
                    constant_terms=constant_terms,
                    max_tabu_steps=max_tabu_steps,
                    tabu_tenure=tabu_tenure,
                    allow_non_improving_every=allow_non_improving_every,
                    rng=rng,
                )
                # Exact authority check with function 1.
                current_score = float(
                    compute_modularity_reduced_cost(
                        adjacency, duals, current, symmetrize_inputs=True, return_details=False
                    )
                )

                current, current_score, merge_count = _greedy_merge_phase(
                    current, current_score, W, constant_terms
                )
                current_score = float(
                    compute_modularity_reduced_cost(
                        adjacency, duals, current, symmetrize_inputs=True, return_details=False
                    )
                )

                split_applied = False
                split_meta: dict[str, object] = {}
                if no_improve_passes + 1 >= split_trigger_no_improve_passes:
                    current, current_score, split_applied, split_meta = _attempt_split_weakest_community(
                        current, current_score, W, constant_terms, rng
                    )
                    if split_applied:
                        current_score = float(
                            compute_modularity_reduced_cost(
                                adjacency, duals, current, symmetrize_inputs=True, return_details=False
                            )
                        )

                improved_this_pass = current_score > before + 1e-12
                if improved_this_pass:
                    no_improve_passes = 0
                else:
                    no_improve_passes += 1

                if current_score > best_restart_score + 1e-12:
                    best_restart_score = current_score
                    best_restart_labels = current.copy()

                round_log["passes"].append(
                    {
                        "pass": pass_idx,
                        "score_before": before,
                        "score_after": current_score,
                        "improved": improved_this_pass,
                        "tabu": tabu_meta,
                        "merge_count": merge_count,
                        "split_applied": split_applied,
                        "split_meta": split_meta,
                    }
                )

                if (
                    tabu_meta.get("n_moves", 0) == 0
                    and merge_count == 0
                    and not split_applied
                    and no_improve_passes >= split_trigger_no_improve_passes
                ):
                    break

            round_log["end_score"] = current_score
            round_log["best_restart_score_so_far"] = best_restart_score
            round_log["n_communities_end"] = int(_canonicalize_labels(current).max()) + 1
            restart_log["rounds"].append(round_log)

            if round_idx < shake_rounds:
                current = _shake_partition(current, W, rng, shake_fraction=shake_fraction)
                current_score = float(
                    compute_modularity_reduced_cost(
                        adjacency, duals, current, symmetrize_inputs=True, return_details=False
                    )
                )

        if best_restart_score > global_best_score + 1e-12:
            global_best_score = best_restart_score
            global_best_labels = best_restart_labels.copy()

        restart_log["restart_best_reduced_cost"] = best_restart_score
        restart_log["restart_best_n_communities"] = int(_canonicalize_labels(best_restart_labels).max()) + 1
        history.append(restart_log)

    return {
        "best_labels": _canonicalize_labels(global_best_labels),
        "best_reduced_cost": float(global_best_score),
        "initial_reduced_cost": float(initial_reduced_cost),
        "improvement": float(global_best_score - initial_reduced_cost),
        "history": history,
        "n_restarts": int(n_restarts),
        "config": {
            "max_local_passes": int(max_local_passes),
            "max_tabu_steps": int(max_tabu_steps),
            "tabu_tenure": int(tabu_tenure),
            "allow_non_improving_every": int(allow_non_improving_every),
            "shake_rounds": int(shake_rounds),
            "shake_fraction": float(shake_fraction),
            "split_trigger_no_improve_passes": int(split_trigger_no_improve_passes),
            "random_seed": int(random_seed),
        },
    }