"""Initial feasible partition generators and helpers."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import math
import networkx as nx
import numpy as np

from asunder.base.utils.graph import partition_vector_to_2d_matrix
from asunder.base.utils.partition_generation import _component_order_from_node_order
from asunder.base.algorithms.modular_VFD import (
    _range_bounds_from_KR, _feasible_K_range, 
    _build_components, _target_sizes_from_bounds
)

# ---------------------------------------------
# Ordered assignment with hard links + (r_min,r_max)
# ---------------------------------------------

def assign_from_order_with_links_range(
    order_idx: List[int],
    N: int,
    K: int,
    R: int,
    R_bounds: tuple|None = None,
    must_link: Optional[List[Tuple[int, int]]] = None,
    cannot_link: Optional[List[Tuple[int, int]]] = None,
    max_K_increase: int = 50,
    max_restarts: int = 8,
    branch: int = 6,
    max_attempts: int = 100,
    seed: int | None = None,
    w_contig: float = 1.0,
    w_switch: float = 0.25,
    w_target: float = 0.15,
) -> Optional[Tuple[np.ndarray, Dict[str, int]]]:
    """
    Assign nodes to balanced clusters from an ordered node sequence.

    Must-link constraints are first contracted into components, and
    cannot-link constraints are enforced as hard conflicts between
    components. The assignment is built from the component order induced by
    ``order_idx`` while searching for a feasible number of used clusters and
    respecting the size bounds implied by ``K`` and ``R``.

    Parameters
    ----------
    order_idx : array_like of int
        Ordered node indices used to induce the component assignment order.
        Entries are interpreted as node ids in ``{0, ..., N - 1}``.
    N : int
        Number of nodes.
    K : int
        Target number of clusters. Larger feasible values may be considered
        up to ``K + max_K_increase``.
    R : int
        Width of the allowed cluster-size range. For a selected cluster count,
        the lower and upper bounds are computed from the corresponding
        balanced range rule.
    R_bounds : tuple[int, int] | None
        Minimum and maximum number of nodes per community (community size constraint).
    must_link : iterable of tuple of int, optional
        Pairs of node indices that must be assigned to the same cluster.
    cannot_link : iterable of tuple of int, optional
        Pairs of node indices that must not be assigned to the same cluster.
    max_K_increase : int, default=50
        Maximum increase above ``K`` allowed when searching for a feasible
        cluster count.
    max_restarts : int, default=8
        Maximum number of randomized restarts used during assignment.
    branch : int, default=6
        Maximum number of candidate partial assignments retained or explored
        at each branching step.
    max_attempts : int, default=100
        Maximum number of assignment attempts before declaring failure.
    seed : int, default=None
        Random seed used for restart and tie-breaking behavior.
    w_contig : float, default=1.0
        Weight for preserving contiguity with respect to the supplied order.
    w_switch : float, default=0.25
        Weight for discouraging unnecessary cluster-label switching.
    w_target : float, default=0.15
        Weight for favoring assignments close to target cluster sizes.

    Returns
    -------
    tuple of numpy.ndarray and dict[str, int] or None
        On success, returns ``(gvec, meta)``. ``gvec`` is an integer array of
        shape ``(N,)`` whose entry ``gvec[i]`` is the cluster label assigned to
        node ``i``. ``meta`` contains ``"r_min"``, ``"r_max"``, and
        ``"K_used"``. Returns ``None`` if no assignment satisfying the hard
        link constraints and size bounds is found.

    Notes
    -----
    This function returns the node-label assignment vector, not the
    co-clustering matrix. Convert ``gvec`` separately when a ``Z`` column is
    required.
    """
    must_link = [] if must_link is None else list(must_link)
    cannot_link = [] if cannot_link is None else list(cannot_link)

    if R_bounds is None:
        r_min, r_max = _range_bounds_from_KR(N, K, R)
    else:
        r_min, r_max = R_bounds

    if N == 0:
        return np.zeros(0, dtype=int), {"r_min": r_min, "r_max": r_max, "K_used": 0}

    k_lo, k_hi = _feasible_K_range(N, r_min, r_max)
    if k_lo > k_hi:
        return None

    K0 = min(max(K, k_lo), k_hi)
    K_end = min(k_hi, K0 + max_K_increase)
    K_candidates = list(range(K0, K_end + 1))

    rng = np.random.default_rng(seed)

    comp = _build_components(N, must_link, cannot_link)
    if comp is None:
        return None

    C = comp["C"]
    cid = comp["cid"]
    comps = comp["comps"]
    csz = comp["csz"]
    forb_mask = comp["forb_mask"]

    if C == 0:
        return np.zeros(N, dtype=int), {"r_min": r_min, "r_max": r_max, "K_used": 0}
    if int(csz.max(initial=0)) > r_max:
        return None

    comp_order = _component_order_from_node_order(order_idx, cid, C)
    pos_of_comp = np.empty(C, dtype=int)
    for t, c in enumerate(comp_order):
        pos_of_comp[int(c)] = t

    def build_gvec(comp2g: np.ndarray) -> np.ndarray:
        gvec = np.empty(N, dtype=int)
        for c in range(C):
            gvec[comps[c]] = int(comp2g[c])
        return gvec

    def try_one(K_used: int, run_seed: int) -> Optional[np.ndarray]:
        target = _target_sizes_from_bounds(N, K_used, r_min, r_max)
        used = np.zeros(K_used, dtype=int)
        deficit = np.full(K_used, r_min, dtype=int)  # deficit[g] = max(0, r_min - used[g])
        deficit_sum = int(deficit.sum())

        in_mask = [0] * K_used
        last_pos = np.full(K_used, -10**9, dtype=int)
        last_g = -1

        comp2g = -np.ones(C, dtype=int)

        local_rng = np.random.default_rng(run_seed)

        frames = []  # each: (c, cand_list, next_i, assigned_g, prev_last_pos, prev_last_g, sc, prev_def)
        t = 0
        attempts = 0
        assigned_sum = 0

        while True:
            if attempts > max_attempts:
                return None

            if t == C:
                if deficit_sum == 0:
                    return build_gvec(comp2g)
                return None

            c = int(comp_order[t])
            sc = int(csz[c])
            pc = int(pos_of_comp[c])
            cf = int(forb_mask[c])

            if len(frames) <= t:
                feas = []
                for g in range(K_used):
                    if used[g] + sc > r_max:
                        continue
                    if (cf & int(in_mask[g])) != 0:
                        continue

                    contig = 0 if last_pos[g] <= -10**8 else abs(pc - int(last_pos[g]))
                    switch = 0 if (last_g == -1 or last_g == g) else 1
                    after = used[g] + sc
                    tgt_pen = abs(int(after) - int(target[g]))

                    score = (
                        w_contig * contig +
                        w_switch * switch +
                        w_target * tgt_pen +
                        1e-12 * float(local_rng.random())
                    )
                    feas.append((score, g))

                if not feas:
                    # backtrack
                    if t == 0:
                        return None
                    t -= 1
                    (c0, cand0, idx0, g0, prev_lp, prev_lg, sc0, prev_def) = frames[t]
                    # undo assignment of frame t
                    used[g0] -= sc0
                    assigned_sum -= sc0
                    in_mask[g0] ^= (1 << c0)
                    last_pos[g0] = prev_lp
                    last_g = prev_lg
                    new_def = max(0, r_min - int(used[g0]))
                    deficit_sum += (new_def - int(deficit[g0]))
                    deficit[g0] = new_def
                    comp2g[c0] = -1
                    frames[t] = (c0, cand0, idx0, -1, prev_lp, prev_lg, sc0, prev_def)
                    continue

                feas.sort(key=lambda x: x[0])
                cand = [g for _, g in feas[: min(branch, len(feas))]]
                frames.append((c, cand, 0, -1, 0, 0, sc, 0))

            c0, cand0, idx0, g0, prev_lp, prev_lg, sc0, prev_def = frames[t]
            if idx0 >= len(cand0):
                # backtrack
                frames.pop()
                if t == 0:
                    return None
                t -= 1
                (c1, cand1, idx1, g1, prev_lp1, prev_lg1, sc1, prev_def1) = frames[t]
                used[g1] -= sc1
                assigned_sum -= sc1
                in_mask[g1] ^= (1 << c1)
                last_pos[g1] = prev_lp1
                last_g = prev_lg1
                new_def = max(0, r_min - int(used[g1]))
                deficit_sum += (new_def - int(deficit[g1]))
                deficit[g1] = new_def
                comp2g[c1] = -1
                frames[t] = (c1, cand1, idx1, -1, prev_lp1, prev_lg1, sc1, prev_def1)
                continue

            g = int(cand0[idx0])
            frames[t] = (c0, cand0, idx0 + 1, g, int(last_pos[g]), int(last_g), sc0, int(deficit[g]))

            # apply assignment
            prev_def_g = int(deficit[g])
            used[g] += sc0
            assigned_sum += sc0
            in_mask[g] |= (1 << c0)
            last_pos[g] = pc
            last_g = g
            comp2g[c0] = g

            new_def_g = max(0, r_min - int(used[g]))
            deficit_sum += (new_def_g - prev_def_g)
            deficit[g] = new_def_g

            rem_nodes = N - assigned_sum
            if deficit_sum > rem_nodes:
                # undo and continue trying next candidate
                used[g] -= sc0
                assigned_sum -= sc0
                in_mask[g] ^= (1 << c0)
                last_pos[g] = frames[t][4]
                last_g = frames[t][5]
                deficit_sum += (prev_def_g - int(deficit[g]))
                deficit[g] = prev_def_g
                comp2g[c0] = -1
                frames[t] = (c0, cand0, idx0 + 1, -1, frames[t][4], frames[t][5], sc0, frames[t][7])
                attempts += 1
                continue

            t += 1
            attempts += 1

    for K_used in K_candidates:
        # global feasibility already ensured by K_candidates; still guard component max
        if int(csz.max(initial=0)) > r_max:
            continue
        for r in range(max_restarts):
            gvec = try_one(K_used, run_seed=int(seed + 31 * (K_used + 1) + 997 * r))
            if gvec is not None:
                return gvec, {"r_min": r_min, "r_max": r_max, "K_used": K_used}

    return None


# ---------------------------------------------
# make_partitions: pass R and handle infeasible orders
# ---------------------------------------------

def make_partitions(
    G,
    K: int,
    R: int,
    R_bounds: tuple|None = None,
    must_link: list|None = None,
    cannot_link: list|None = None,
    n_cols: int = 15,
    seed: int | None = None,
    nodes=None,
):
    """
    Generate graph-informed feasible partition columns.

    Candidate node orderings are constructed from the graph and converted into
    balanced feasible partitions using must-link and cannot-link constraints.
    Each successful partition is returned as a binary co-clustering matrix.

    Parameters
    ----------
    G : networkx.Graph
        Graph whose nodes define the partitioning problem and whose structure
        is used to generate candidate orderings.
    K : int
        Target number of clusters.
    R : int
        Width of the allowed cluster-size range.
    R_bounds : tuple[int, int] | None
        Minimum and maximum number of nodes per community (community size constraint).
    must_link : iterable of tuple, optional
        Pairs of node indices that must be assigned to the same cluster.
    cannot_link : iterable of tuple, optional
        Pairs of node indices that must be assigned to different clusters.
    n_cols : int, default=15
        Maximum number of feasible partition columns to generate.
    seed : int, default=None
        Random seed used for randomized or noisy graph orderings.
    nodes : sequence, optional
        Node ordering used to map graph node labels to integer indices. If
        ``None``, the graph's node iteration order is used.

    Returns
    -------
    Z_star : list of numpy.ndarray
        List of binary ``N x N`` co-clustering matrices. For each matrix
        ``Z``, ``Z[i, j] = 1`` if nodes ``i`` and ``j`` are assigned to the
        same cluster, and ``0`` otherwise.
    """

    rng = np.random.default_rng(seed)

    if must_link == None:
        must_link = []
    if cannot_link == None:
        cannot_link = []

    if nodes is None:
        nodes = sorted(G.nodes())
    N = len(nodes)
    node_to_idx = {u: i for i, u in enumerate(nodes)}

    def build_from_label_order(name: str, order_labels: List[Any]):
        order_idx = [node_to_idx[u] for u in order_labels]

        if must_link != [] or cannot_link != []:
            # For more “contiguous” order, increase w_contig and reduce branch (fewer “jumps”).
            # To succeed more often under heavy cannot-link density, increase max_K_increase, branch, and/or max_restarts.
            out = assign_from_order_with_links_range(
                order_idx, N, K, R, R_bounds=R_bounds,
                must_link=must_link,
                cannot_link=cannot_link,
                seed=seed,
                # max_attempts=1000, If we are not trying enough with the default, it could be increased
            )
            if out is None:
                return None
            g, meta = out
        else:
            if R_bounds is None:
                r_min, r_max = _range_bounds_from_KR(N, K, R)
            else:
                r_min, r_max = R_bounds
            k_lo, k_hi = _feasible_K_range(N, r_min, r_max)
            if k_lo > k_hi:
                return None
            K_used = min(max(K, k_lo), k_hi)
            target = _target_sizes_from_bounds(N, K_used, r_min, r_max)

            g = -np.ones(N, dtype=int)
            pos = 0
            for c, sz in enumerate(target):
                for u in order_idx[pos:pos + int(sz)]:
                    g[u] = c
                pos += int(sz)
            meta = {"r_min": r_min, "r_max": r_max, "K_used": K_used}

        Z = partition_vector_to_2d_matrix(g)
        return Z, {"name": name, "order_labels": order_labels, "g": g, **meta}

    cols = []
    seen = set()



    # 1) spectral (Fiedler)
    if N <= 1:
        f = np.zeros(N)
    else:
        L = nx.laplacian_matrix(G, nodelist=nodes).toarray()

        w, v = np.linalg.eigh(L)
        f = v[:, 1]

    order = [nodes[i] for i in np.argsort(f)]
    out = build_from_label_order("spectral_fiedler", order)
    if out is not None:
        Z, meta = out
        key = meta["g"].tobytes()
        cols.append((Z, meta)); seen.add(key)

    # 2) BFS from max-degree node
    start = max(nodes, key=lambda u: G.degree(u)) if N else None
    if start is not None:
        order = list(nx.bfs_tree(G, start).nodes())
        # cover remaining components
        visited = set(order)
        remaining = [u for u in nodes if u not in visited]
        remaining.sort(key=lambda u: G.degree(u), reverse=True)
        for s in remaining:
            for u in nx.bfs_tree(G, s).nodes():
                if u not in visited:
                    visited.add(u)
                    order.append(u)
        out = build_from_label_order("bfs_from_max_degree", order)
        if out is not None:
            Z, meta = out
            key = meta["g"].tobytes()
            if key not in seen:
                cols.append((Z, meta)); seen.add(key)
    # 3) DFS
    if start is not None:
        order = list(nx.dfs_preorder_nodes(G, start))
        visited = set(order)
        remaining = [u for u in nodes if u not in visited]
        remaining.sort(key=lambda u: G.degree(u), reverse=True)
        for s in remaining:
            for u in nx.dfs_preorder_nodes(G, s):
                if u not in visited:
                    visited.add(u)
                    order.append(u)
        out = build_from_label_order("dfs_from_max_degree", order)
        if out is not None:
            Z, meta = out
            key = meta["g"].tobytes()
            if key not in seen:
                cols.append((Z, meta)); seen.add(key)

    # 4) degree sorted
    order = sorted(nodes, key=lambda u: (G.degree(u), u), reverse=True)
    out = build_from_label_order("degree_sorted", order)
    if out is not None:
        Z, meta = out
        key = meta["g"].tobytes()
        if key not in seen:
            cols.append((Z, meta)); seen.add(key)
    # 5) noisy spectral fill
    noise_std = 1e-6 if N else 0.0
    tries = 0
    while len(cols) < n_cols and tries < 20 * n_cols:
        tries += 1
        f_noisy = f + rng.normal(0.0, noise_std, size=f.shape)
        order = [nodes[i] for i in np.argsort(f_noisy)]
        out = build_from_label_order(f"spectral_noisy_{tries}", order)
        if out is None:
            continue
        Z, meta = out
        key = meta["g"].tobytes()
        if key not in seen:
            cols.append((Z, meta)); seen.add(key)

    Z_star = [Z for Z, _ in cols]
    info = [meta for _, meta in cols]
    return Z_star


# ---------------------------------------
# MRV with restarts under (r_min, r_max)
# ---------------------------------------

def make_one_feasible_partition_mrv_restarts(
    N, K, R,
    R_bounds=None,
    must_link=None,
    cannot_link=None,
    seed=None,
    max_tries=200,
    jitter_top=2,
    max_K_increase=50,
    return_Z=True,
):
    """
    Construct one feasible partition using MRV search with restarts.

    Must-link constraints are compressed into components and cannot-link
    constraints are enforced between components. The search repeatedly chooses
    the unassigned component with the fewest feasible cluster choices and
    assigns it using a randomized best-fit rule. Failed partial assignments are
    discarded and retried until a feasible partition is found or the restart
    limit is reached.

    Parameters
    ----------
    N : int
        Number of nodes.
    K : int
        Target number of clusters.
    R : int
        Width of the allowed cluster-size range.
    R_bounds : tuple[int, int] | None
        Minimum and maximum number of nodes per community (community size constraint).
    must_link : iterable of tuple of int, optional
        Pairs of node indices that must be assigned to the same cluster.
    cannot_link : iterable of tuple of int, optional
        Pairs of node indices that must be assigned to different clusters.
    seed : int, default=None
        Random seed used for tie-breaking and restart randomness.
    max_tries : int, default=200
        Maximum number of restart attempts.
    jitter_top : int, default=2
        Number of top-ranked feasible groups considered during randomized
        best-fit assignment. Larger values increase randomness.
    max_K_increase : int, default=50
        Maximum increase above ``K`` allowed when searching for a feasible
        cluster count.
    return_Z : bool, default=True
        If ``True``, return a binary co-clustering matrix. If ``False``,
        return an integer assignment vector.

    Returns
    -------
    numpy.ndarray or None
        A feasible partition, or ``None`` if no feasible assignment is found.
        If ``return_Z`` is ``True``, the result is an ``N x N`` binary matrix
        ``Z`` with ``Z[i, j] = 1`` when nodes ``i`` and ``j`` are co-clustered.
        If ``return_Z`` is ``False``, the result is an integer vector ``g`` of
        shape ``(N,)``.
    """
    rng = np.random.default_rng(seed)
    must_link = [] if must_link is None else list(must_link)
    cannot_link = [] if cannot_link is None else list(cannot_link)

    if R_bounds is None:
        r_min, r_max = _range_bounds_from_KR(N, K, R)
    else:
        r_min, r_max = R_bounds
    if N == 0:
        g0 = np.zeros(0, dtype=int)
        return partition_vector_to_2d_matrix(g0) if return_Z else g0

    comp = _build_components(N, must_link, cannot_link)
    if comp is None:
        return None

    use_bitmask = comp["use_bitmask"]
    comp_bit = comp["comp_bit"]
    C = comp["C"]
    comps = comp["comps"]
    csz = comp["csz"]
    forb_mask = comp["forb_mask"]

    if int(csz.max(initial=0)) > r_max:
        return None

    k_lo, k_hi = _feasible_K_range(N, r_min, r_max)
    if k_lo > k_hi:
        return None

    # if r_min > 0, empty groups are not allowed; also components are indivisible
    if r_min > 0:
        k_hi = min(k_hi, C)

    if k_lo > k_hi:
        return None

    K0 = min(max(K, k_lo), k_hi)
    K_end = min(k_hi, K0 + max_K_increase)
    K_candidates = list(range(K0, K_end + 1))

    if use_bitmask:
        deg = np.array([m.bit_count() for m in forb_mask], dtype=int)
    else:
        deg = np.array([len(s) for s in forb_mask], dtype=int)

    def build_gvec(comp2g):
        gvec = np.empty(N, dtype=int)
        for c in range(C):
            gvec[comps[c]] = int(comp2g[c])
        return gvec

    for K_used in K_candidates:
        # quick sanity: N must fit in [K_used*r_min, K_used*r_max]
        if not (K_used * r_min <= N <= K_used * r_max):
            continue

        for _ in range(max_tries):
            sz = np.zeros(K_used, dtype=int)             # current group sizes
            deficit = np.maximum(0, r_min - sz)          # needed to hit r_min
            deficit_sum = int(deficit.sum())

            comp2g = -np.ones(C, dtype=int)
            # in_mask = [0] * K_used
            in_mask = ([0] * K_used) if use_bitmask else [set() for _ in range(K_used)]
            unassigned = set(range(C))
            rem_nodes = int(N)

            def feas_groups(c):
                sc = int(csz[c])
                cf = int(forb_mask[c])
                F = []
                for g in range(K_used):
                    if sz[g] + sc > r_max:
                        continue

                    if use_bitmask:
                        if (cf & int(in_mask[g])) != 0:
                            continue
                    else:
                        if forb_mask[c] & in_mask[g]:
                            continue

                    # forward-check min-feasibility
                    old_def = int(deficit[g])
                    new_def = max(0, r_min - int(sz[g] + sc))
                    def2 = deficit_sum - old_def + new_def
                    if def2 <= (rem_nodes - sc):
                        F.append(g)
                return F

            ok = True
            while unassigned:
                # MRV pick
                best_c, best_F, best_key = None, None, None
                for c in unassigned:
                    F = feas_groups(c)
                    if not F:
                        ok = False
                        break
                    key = (len(F), -int(csz[c]), -int(deg[c]), float(rng.random()))
                    if best_key is None or key < best_key:
                        best_c, best_F, best_key = c, F, key
                if not ok:
                    break

                c = int(best_c)
                F = list(best_F)
                sc = int(csz[c])

                # choose group: prioritize filling deficits, then best-fit to r_max
                scored = sorted(
                    F,
                    key=lambda g: (
                        0 if sz[g] < r_min else 1,
                        (r_max - int(sz[g] + sc)),
                        -int(deficit[g]),
                    ),
                )
                if jitter_top and len(scored) > 1:
                    top = scored[: min(jitter_top, len(scored))]
                    g = int(rng.choice(top))
                else:
                    g = int(scored[0])

                # assign
                old_def = int(deficit[g])
                sz[g] += sc
                rem_nodes -= sc

                # in_mask[g] |= (1 << c)
                if use_bitmask:
                    in_mask[g] |= comp_bit[c]
                else:
                    in_mask[g].add(c)

                comp2g[c] = g
                unassigned.remove(c)

                new_def = max(0, r_min - int(sz[g]))
                deficit[g] = new_def
                deficit_sum += (new_def - old_def)

            if ok and deficit_sum == 0:
                gvec = build_gvec(comp2g)
                return partition_vector_to_2d_matrix(gvec) if return_Z else gvec

    return None


# ---------------------------------------
# Random-ish partition generator (range)
# ---------------------------------------

def make_partitions_random(
    N, K, R, R_bounds=None,
    must_link=None,
    cannot_link=None,
    seed=None,
    return_Z=True,
    max_K_increase=50,
):
    """
    Generate random feasible partitions subject to link and size constraints.

    Must-link constraints are contracted into components, cannot-link
    constraints are enforced as hard incompatibilities, and feasible balanced
    assignments are attempted for ``K`` and larger candidate values up to
    ``K + max_K_increase``.

    Parameters
    ----------
    N : int
        Number of nodes.
    K : int
        Target number of clusters.
    R : int
        Width of the allowed cluster-size range.
    R_bounds : tuple[int, int] | None
        Minimum and maximum number of nodes per community (community size constraint).
    must_link : iterable of tuple of int, optional
        Pairs of node indices that must be assigned to the same cluster.
    cannot_link : iterable of tuple of int, optional
        Pairs of node indices that must be assigned to different clusters.
    seed : int, default=None
        Random seed used for shuffled component orders.
    return_Z : bool, default=True
        If ``True``, return binary co-clustering matrices. If ``False``,
        return integer assignment vectors.
    max_K_increase : int, default=50
        Maximum increase above ``K`` allowed when searching for a feasible
        cluster count.

    Returns
    -------
    list of numpy.ndarray
        Feasible partitions. If ``return_Z`` is ``True``, each entry is an
        ``N x N`` binary matrix ``Z`` with ``Z[i, j] = 1`` when nodes ``i`` and
        ``j`` are co-clustered. If ``return_Z`` is ``False``, each entry is an
        integer assignment vector ``g`` of shape ``(N,)``.

        Returns an empty list if no feasible partition is found.
    """
    rng = np.random.default_rng(seed)
    must_link = [] if must_link is None else list(must_link)
    cannot_link = [] if cannot_link is None else list(cannot_link)

    if R_bounds is None:
        r_min, r_max = _range_bounds_from_KR(N, K, R)
    else:
        r_min, r_max = R_bounds
    if N == 0:
        return [partition_vector_to_2d_matrix(np.zeros(0, dtype=int))] if return_Z else []

    comp = _build_components(N, must_link, cannot_link)
    if comp is None:
        return []
    use_bitmask = comp["use_bitmask"]
    comp_bit = comp["comp_bit"]
    C = comp["C"]
    comps = comp["comps"]
    csz = comp["csz"]
    forb_mask = comp["forb_mask"]

    if int(csz.max(initial=0)) > r_max:
        return []

    k_lo, k_hi = _feasible_K_range(N, r_min, r_max)
    if k_lo > k_hi:
        return []

    if r_min > 0:
        k_hi = min(k_hi, C)
    if k_lo > k_hi:
        return []

    K0 = min(max(K, k_lo), k_hi)
    K_end = min(k_hi, K0 + max_K_increase)
    K_candidates = list(range(K0, K_end + 1))

    def build_gvec(comp2g):
        gvec = np.empty(N, dtype=int)
        for c in range(C):
            gvec[comps[c]] = int(comp2g[c])
        return gvec

    def assign_greedy(order, K_used, jitter_top=2):
        sz = np.zeros(K_used, dtype=int)
        deficit = np.maximum(0, r_min - sz)
        deficit_sum = int(deficit.sum())
        rem_nodes = int(N)

        in_mask = [0] * K_used
        comp2g = -np.ones(C, dtype=int)

        for c in order:
            c = int(c)
            sc = int(csz[c])
            cf = int(forb_mask[c])

            feas = []
            for g in range(K_used):
                if sz[g] + sc > r_max:
                    continue
                if (cf & int(in_mask[g])) != 0:
                    continue
                old_def = int(deficit[g])
                new_def = max(0, r_min - int(sz[g] + sc))
                def2 = deficit_sum - old_def + new_def
                if def2 <= (rem_nodes - sc):
                    feas.append(g)

            if not feas:
                return None

            scored = sorted(
                feas,
                key=lambda g: (
                    0 if sz[g] < r_min else 1,
                    (r_max - int(sz[g] + sc)),
                    -int(deficit[g]),
                ),
            )
            if jitter_top and len(scored) > 1:
                top = scored[: min(jitter_top, len(scored))]
                g = int(rng.choice(top))
            else:
                g = int(scored[0])

            old_def = int(deficit[g])
            sz[g] += sc
            rem_nodes -= sc
            in_mask[g] |= (1 << c)
            comp2g[c] = g

            new_def = max(0, r_min - int(sz[g]))
            deficit[g] = new_def
            deficit_sum += (new_def - old_def)

        if deficit_sum != 0:
            return None
        return build_gvec(comp2g)

    parts = []

    # try a couple K_used values near K (and larger if needed)
    for K_used in K_candidates:
        if not (K_used * r_min <= N <= K_used * r_max):
            continue

        order = np.argsort(csz)[::-1]  # largest first
        g1 = assign_greedy(order, K_used, jitter_top=0)
        if g1 is not None:
            parts.append(("largest_first_bestfit", g1, K_used))

        order2 = order.copy()
        rng.shuffle(order2)
        g2 = assign_greedy(order2, K_used, jitter_top=2)
        if g2 is not None:
            parts.insert(0, ("shuffle_then_bestfit", g2, K_used)) # more random like.

        if parts:
            break  # keep closest K_used solutions

    # last-ditch MRV (range-aware now)
    g_last = make_one_feasible_partition_mrv_restarts(
        N=N, K=K, R=R, R_bounds=R_bounds,
        must_link=must_link,
        cannot_link=cannot_link,
        seed=seed + 1,
        max_tries=500,
        jitter_top=2,
        return_Z=False,
    )

    if not parts and g_last is None:
        return []

    if return_Z:
        out = [partition_vector_to_2d_matrix(g) for _, g, _ in parts]
        if g_last is not None:
            out.append(partition_vector_to_2d_matrix(g_last))
        return out

    out = [{"name": name, "g": g, "K_used": K_used, "r_min": r_min, "r_max": r_max}
           for name, g, K_used in parts]
    if g_last is not None:
        out.append({"name": "mrv_last_ditch", "g": g_last, "r_min": r_min, "r_max": r_max})
    return out


def check_balance(Z, K, R, R_bounds=None):
    """
    Check whether a co-clustering matrix satisfies balance bounds.

    The row sums of ``Z`` are interpreted as cluster sizes. A partition is
    balanced if every row sum lies between ``R_min`` and ``R_max``, where
    ``R_min = max(1, floor(N / K - R / 2 + 1 / 2))`` and
    ``R_max = R_min + R``.

    Parameters
    ----------
    Z : numpy.ndarray
        Binary ``N x N`` co-clustering matrix. ``Z[i, j] = 1`` indicates that
        nodes ``i`` and ``j`` are assigned to the same cluster.
    K : int
        Target number of clusters.
    R : int
        Width of the allowed cluster-size range.

    Returns
    -------
    min_size : scalar
        Minimum row sum of ``Z``.
    max_size : scalar
        Maximum row sum of ``Z``.
    is_balanced : bool
        ``True`` if all row sums are within the computed balance bounds;
        otherwise ``False``.
    """
    N = Z.shape[0]

    if R_bounds is None:
        R_min = max(1, math.floor((N/K - R/2) + 1/2))
        R_max = R_min + R
    else:
        R_min, R_max = R_bounds

    rs = Z.sum(axis=1)
    return rs.min(), rs.max(), (rs.min() >= R_min and rs.max() <= R_max)