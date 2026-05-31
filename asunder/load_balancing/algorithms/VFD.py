"""Very Fortunate Descent (VFD):  A local search algorithm for partition refinement under pairwise and balance constraints."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from asunder.base.branch_and_price.symmetry_detection import weighted_constraint_orbits
from asunder.base.algorithms.modular_VFD import (
    _build_coassociation_matrix, _range_bounds_from_KR, 
    _feasible_K_range, _normalize_pair, 
    _build_components, _target_sizes_from_bounds,
    _fingerprint_blocks_from_rounded_rows, _symmetrize_unitdiag,
    _ensure_at_least_K_blocks, _component_sum_matrix_B, 
    _component_matrices_from_node_matrix,
)
from asunder.base.utils.graph import partition_vector_to_2d_matrix

def very_fortunate_descent_legacy(
    wz: np.ndarray,
    A: np.ndarray,
    a: np.ndarray,
    m: float,
    K: int = None,
    R: int = None,
    R_bounds: tuple|None = None,
    must_link: Sequence[Tuple[int, int]] = [],
    cannot_link: Sequence[Tuple[int, int]] = [],
    seed: int | None = None,
    fingerprint_decimals: int = 6,
    allow_block_splitting: bool = True,
    max_K_increase: int = 0, # If 0, keep K_used == K
    restarts: int = 6,
    local_iters: int = 60,
    w_coassoc: float = 0.05,   # tie-break toward co-association cohesion
    clustering_Ks: Sequence[int] = None,
    clustering_seeds: Sequence[int] = (0, 1, 2),
    clustering_methods: Sequence[str] = ("kmeans", "gmm", "spectral"),
    wz_is_C_node: bool= False,
) -> Optional[Tuple[np.ndarray, Dict[str, Any]]]:
    """
    Legacy function for building a feasible decomposition column from co-association structure and local search.

    Parameters
    ----------
    wz : numpy.ndarray
        Input node-level matrix.
    A : numpy.ndarray
        Original adjacency matrix.
    a : numpy.ndarray
        Degree-like vector used in the modularity construction.
    m : float
        Modularity scaling constant.
    K : int or None
        Baseline number of communities.
    R : int or None
        Width of the allowed cluster-size range. Also corresponds to the load balance tightness (smaller R implies tighter load balance).
        For a selected cluster count, the lower and upper bounds are computed from the corresponding
        balanced range rule.
    R_bounds : tuple or None
        Lower and upper limits on the number of nodes in communities.
    must_link : sequence of tuple of int
        Must-link pairs.
    cannot_link : sequence of tuple of int
        Cannot-link pairs.
    seed : int, default=0
        Random seed.
    fingerprint_decimals : int, default=6
        Decimal rounding used to form fingerprint blocks.
    allow_block_splitting : bool, default=True
        If True, allow refinement of coarse fingerprint blocks.
    max_K_increase : int, default=0
        Maximum increase above the baseline K when the K-constraint is active.
    restarts : int, default=6
        Number of constructive restarts.
    local_iters : int, default=60
        Base local-search iteration parameter.
    w_coassoc : float, default=0.05
        Weight of the co-association cohesion term in local decisions.
    clustering_Ks : sequence of int or None, default=None
        Community counts used to build the co-association matrix.
    clustering_seeds : sequence of int, default=(0, 1, 2)
        Seeds used during co-association construction.
    clustering_methods : sequence of str, default=("kmeans", "gmm", "spectral")
        Clustering methods used during co-association construction.
    wz_is_C_node : bool, default=False

    Returns
    -------
    tuple of (numpy.ndarray, dict) or None
        A pair ``(Z_col, meta)`` if a feasible column is found, else ``None``.
    """

    rng = np.random.default_rng(seed)

    wz = np.asarray(wz, dtype=float)
    N = int(wz.shape[0])
    if N == 0:
        g0 = np.zeros(0, dtype=int)
        return partition_vector_to_2d_matrix(g0), {"r_min": 0, "r_max": 0, "K_used": 0}

    sym = _symmetrize_unitdiag(wz)

    if R_bounds is None and K is not None and R is not None:
        r_min, r_max = _range_bounds_from_KR(N, K, R)
    elif R_bounds is not None:
        r_min, r_max = R_bounds
    else:
        raise ValueError("The provided combination of load balancing parameters are invalid.")

    k_lo, k_hi = _feasible_K_range(N, r_min, r_max)

    if k_lo > k_hi:
        return None

    if clustering_Ks is None:
        clustering_Ks = (max(2, k_lo), min(k_hi, K), min(k_hi, K + 2))

    must_link = [_normalize_pair(i, j) for (i, j) in (must_link or [])]
    cannot_link = [_normalize_pair(i, j) for (i, j) in (cannot_link or [])]

    comp = _build_components(N, must_link, cannot_link)
    if comp is None:
        return None

    Cn = int(comp["C"])
    if Cn == 0:
        g0 = np.zeros(N, dtype=int)
        return partition_vector_to_2d_matrix(g0), {"r_min": r_min, "r_max": r_max, "K_used": 0}

    csz = np.asarray(comp["csz"], dtype=int)
    if int(csz.max()) > r_max:
        return None

    # Co-association always (fallback to sym(wz) if sklearn unavailable)
    if not wz_is_C_node:
        C_node = _build_coassociation_matrix(
            sym,
            n_components_list=clustering_Ks,
            seeds=clustering_seeds,
            methods=clustering_methods
        )
    else:
        C_node = wz

    if C_node is None:
        C_node = sym.copy()
    C_node = _symmetrize_unitdiag(C_node)

    C_comp = _component_matrices_from_node_matrix(C_node, comp) # average co-association between components
    W_B = _component_sum_matrix_B(A, a, m, comp) # sum modularity matrix between components

    use_bitmask = bool(comp["use_bitmask"])
    forb = comp["forb_mask"]
    comp_bit = comp["comp_bit"]
    comps = comp["comps"]

    def build_gvec(comp2g: np.ndarray) -> np.ndarray:
        g = np.empty(N, dtype=int)
        for c in range(Cn):
            g[comps[c]] = int(comp2g[c])
        return g

    def group_ok_add(block_forb: int, block_mask: int, g: int, in_mask: List[int]) -> bool:
        return (block_forb & int(in_mask[g])) == 0

    def group_ok_add_set(block_forb_set: set, g: int, in_set: List[set]) -> bool:
        return len(block_forb_set & in_set[g]) == 0

    def block_internal_sum(M: np.ndarray, b: List[int]) -> float:
        idx = np.asarray(b, dtype=int)
        return float(M[np.ix_(idx, idx)].sum())

    best = None

    K0 = min(max(K, k_lo), k_hi)
    K_end = min(k_hi, K0 + int(max_K_increase))
    for K_used in range(K0, K_end + 1):
        if not (K_used * r_min <= N <= K_used * r_max):
            continue
        if r_min > 0 and K_used > Cn:
            continue

        target = _target_sizes_from_bounds(N, K_used, r_min, r_max)

        # Build fingerprint blocks once per K_used; splitting can depend on K_used
        base_blocks = _fingerprint_blocks_from_rounded_rows(C_comp, comp, fingerprint_decimals=fingerprint_decimals, r_max=r_max)
        base_blocks = _ensure_at_least_K_blocks(base_blocks, K_used, csz)

        for rr in range(int(restarts)):
            # Copy blocks per restart (they can be split on demand)
            blocks = [list(b) for b in base_blocks]
            Bn = len(blocks)

            if Bn < K_used:
                continue

            # Precompute per-block data
            b_size = np.array([int(sum(int(csz[c]) for c in b)) for b in blocks], dtype=int)
            b_ncomp = np.array([len(b) for b in blocks], dtype=int)

            if use_bitmask:
                b_mask = []
                b_forb = []
                for b in blocks:
                    cm = 0
                    fm = 0
                    for c in b:
                        cm |= int(comp_bit[c])
                    for c in b:
                        fm |= int(forb[c])
                    b_mask.append(cm)
                    b_forb.append(fm)
                b_mask = np.asarray(b_mask, dtype=object)
                b_forb = np.asarray(b_forb, dtype=object)
            else:
                b_mask = None
                b_forb_sets = []
                for b in blocks:
                    s = set(b)
                    f = set()
                    for c in b:
                        f |= set(forb[c])
                    b_forb_sets.append(f)

            b_intB = np.array([block_internal_sum(W_B, b) for b in blocks], dtype=float)
            b_intC = np.array([block_internal_sum(C_comp, b) for b in blocks], dtype=float)

            # Group state
            gsz = np.zeros(K_used, dtype=int)
            ncomp_g = np.zeros(K_used, dtype=int)
            comp2g = -np.ones(Cn, dtype=int)
            members_b: List[List[int]] = [[] for _ in range(K_used)]

            # Membership representation over components
            if use_bitmask:
                in_mask = [0] * K_used
            else:
                in_set = [set() for _ in range(K_used)]

            # For fast gains: sum of columns for current group membership
            sumB = [np.zeros(Cn, dtype=float) for _ in range(K_used)]
            sumC = [np.zeros(Cn, dtype=float) for _ in range(K_used)]

            # For cohesion: total sum over (c,d) in group of C_comp[c,d]
            totC = np.zeros(K_used, dtype=float)

            def add_block_to_group(bi: int, g: int):
                idx = np.asarray(blocks[bi], dtype=int)

                # update cohesion total: + 2*cross + internal
                cross = float(sumC[g][idx].sum())
                totC[g] += 2.0 * cross + float(b_intC[bi])

                # assign
                for c in idx:
                    comp2g[int(c)] = g
                members_b[g].append(bi)
                gsz[g] += int(b_size[bi])
                ncomp_g[g] += int(b_ncomp[bi])

                if use_bitmask:
                    in_mask[g] |= int(b_mask[bi])
                else:
                    in_set[g] |= set(idx.tolist())

                # update column sums
                for c in idx:
                    sumB[g] += W_B[:, int(c)]
                    sumC[g] += C_comp[:, int(c)]

            def remove_block_from_group(bi: int, g: int):
                idx = np.asarray(blocks[bi], dtype=int)

                cross = float((sumC[g][idx].sum()) - float(b_intC[bi]))  # includes internal; remove later
                # totC removal uses current state; recompute using formula:
                # current totC = sum_{u,v in G} C[u,v]
                # removing block: totC -= 2*cross_to_rest + internal
                # cross_to_rest = sum_{c in block} sumC[g][c] - internal
                cross_to_rest = float(sumC[g][idx].sum()) - float(b_intC[bi])
                totC[g] -= 2.0 * cross_to_rest + float(b_intC[bi])

                for c in idx:
                    comp2g[int(c)] = -1
                members_b[g].remove(bi)
                gsz[g] -= int(b_size[bi])
                ncomp_g[g] -= int(b_ncomp[bi])

                if use_bitmask:
                    in_mask[g] ^= int(b_mask[bi])
                else:
                    in_set[g] -= set(idx.tolist())

                for c in idx:
                    sumB[g] -= W_B[:, int(c)]
                    sumC[g] -= C_comp[:, int(c)]

            def cohesion(g: int) -> float:
                k = int(ncomp_g[g])
                if k <= 1:
                    return 0.0
                pair_sum = (float(totC[g]) - float(k)) / 2.0
                return (2.0 * pair_sum) / float(k * (k - 1))

            def delta_move_block(bi: int, g_from: int, g_to: int) -> float:
                idx = np.asarray(blocks[bi], dtype=int)
                s_to = float(sumB[g_to][idx].sum())
                s_fr = float(sumB[g_from][idx].sum())
                return 2.0 * (s_to - (s_fr - float(b_intB[bi])))

            def feasible_add_block(bi: int, g: int) -> bool:
                if gsz[g] + int(b_size[bi]) > r_max:
                    return False
                if use_bitmask:
                    return group_ok_add(int(b_forb[bi]), int(b_mask[bi]), g, in_mask)
                return group_ok_add_set(b_forb_sets[bi], g, in_set)

            def ensure_nonempty_seeding(order_blocks: List[int]) -> bool:
                # seed first K_used groups with one block each (empty group always feasible wrt cannot-link)
                used_blocks = set()
                gi = 0
                for bi in order_blocks:
                    if gi >= K_used:
                        break
                    if int(b_size[bi]) > r_max:
                        continue
                    add_block_to_group(int(bi), int(gi))
                    used_blocks.add(int(bi))
                    gi += 1
                return gi == K_used, used_blocks

            # Constructive assignment
            order = list(np.argsort(b_size)[::-1])
            ok_seed, seeded = ensure_nonempty_seeding(order)
            if not ok_seed:
                continue

            for bi in order:
                bi = int(bi)
                if bi in seeded:
                    continue

                # feasible destinations with forward check for r_min
                F = []
                rem_nodes = int(N - gsz.sum())
                deficit_sum = int(np.maximum(0, r_min - gsz).sum())

                for g in range(K_used):
                    if not feasible_add_block(bi, g):
                        continue
                    old_def = max(0, r_min - int(gsz[g]))
                    new_def = max(0, r_min - int(gsz[g] + b_size[bi]))
                    def2 = deficit_sum - old_def + new_def
                    if def2 <= (rem_nodes - int(b_size[bi])):
                        F.append(g)

                if not F:
                    # optional split if block too constraining
                    if allow_block_splitting and len(blocks[bi]) > 1:
                        # split into singleton component blocks
                        comps_b = blocks[bi]
                        # replace bi with first singleton, append others
                        blocks[bi] = [comps_b[0]]
                        for c in comps_b[1:]:
                            blocks.append([c])
                        # rebuild and restart this restart
                        ok_seed = False
                    else:
                        ok_seed = False
                    break

                idx = np.asarray(blocks[bi], dtype=int)
                scored = []
                for g in F:
                    gainB = float(2.0 * sumB[g][idx].sum())
                    gainC = float(sumC[g][idx].sum())
                    tgt_pen = abs(int(gsz[g] + b_size[bi]) - int(target[g]))
                    fill = 0 if gsz[g] < r_min else 1
                    scored.append((fill, -(gainB + w_coassoc * gainC), tgt_pen, float(rng.random()), g))
                scored.sort()
                g_best = int(scored[0][-1])
                add_block_to_group(bi, g_best)

            if not ok_seed:
                continue

            # Repair deficits using cohesion-driven donors; split blocks only if needed
            def repair_min_sizes() -> bool:
                max_steps = 20000
                step = 0
                while step < max_steps:
                    step += 1
                    if step > math.sqrt(max_steps):
                        print(f"`repair_min_sizes` has been running for {step - 1} steps.")
                    under = [g for g in range(K_used) if gsz[g] < r_min]
                    if not under:
                        return True
                    g_need = min(under, key=lambda g: int(gsz[g]))
                    donors = [g for g in range(K_used) if gsz[g] > r_min]
                    if not donors:
                        return False
                    donors.sort(key=lambda g: cohesion(g))

                    moved = False
                    for g_from in donors:
                        # pick weakest-attached block from low-cohesion donor
                        cand_blocks = list(members_b[g_from])
                        rng.shuffle(cand_blocks)

                        def block_attachment(bi: int) -> float:
                            idx = np.asarray(blocks[bi], dtype=int)
                            # attachment to donor excluding internal block
                            cross_to_group = float(sumC[g_from][idx].sum())
                            cross_to_rest = cross_to_group - float(b_intC[bi])
                            denom = max(1, int(ncomp_g[g_from]) - int(b_ncomp[bi]))
                            return cross_to_rest / float(max(1, int(b_ncomp[bi]) * denom))

                        cand_blocks.sort(key=lambda bi: (block_attachment(int(bi)), int(b_size[int(bi)])))

                        for bi in cand_blocks:
                            bi = int(bi)
                            if gsz[g_from] - int(b_size[bi]) < r_min:
                                continue
                            if not feasible_add_block(bi, g_need):
                                # attempt to split and try moving a piece
                                if allow_block_splitting and len(blocks[bi]) > 1:
                                    comps_b = blocks[bi]
                                    # remove whole block, split into singleton blocks staying in donor, then move one singleton
                                    remove_block_from_group(bi, g_from)
                                    blocks[bi] = [comps_b[0]]
                                    add_block_to_group(bi, g_from)
                                    for c in comps_b[1:]:
                                        blocks.append([c])
                                        new_bi = len(blocks) - 1
                                        # append new block into donor immediately
                                        # compute minimal per-block fields on the fly for singleton
                                        # singleton internal sums:
                                        # b_intB/B and b_intC/C for singleton are just diagonal entries
                                        # keep consistent arrays by using python lists for these when splitting
                                        # Simplify: disallow splitting after construction unless you restart
                                    return False
                                continue

                            # move block
                            remove_block_from_group(bi, g_from)
                            add_block_to_group(bi, g_need)
                            moved = True
                            break

                        if moved:
                            break

                    if not moved:
                        return False
                return False

            if not repair_min_sizes():
                continue

            # Local search: modularity-improving block moves, biased to take from low cohesion groups
            for _ in range(int(local_iters)):
                improved = False
                g_order = list(range(K_used))
                g_order.sort(key=lambda g: cohesion(g))  # low cohesion first

                for g_from in g_order:
                    cand_blocks = list(members_b[g_from])
                    rng.shuffle(cand_blocks)

                    for bi in cand_blocks:
                        bi = int(bi)
                        if gsz[g_from] - int(b_size[bi]) < r_min:
                            continue

                        best_val = 1e-12
                        best_g = -1
                        for g_to in range(K_used):
                            if g_to == g_from:
                                continue
                            if not feasible_add_block(bi, g_to):
                                continue
                            if gsz[g_to] + int(b_size[bi]) > r_max:
                                continue

                            dB = delta_move_block(bi, g_from, g_to)

                            idx = np.asarray(blocks[bi], dtype=int)
                            dC = 2.0 * (
                                float(sumC[g_to][idx].sum()) - (float(sumC[g_from][idx].sum()) - float(b_intC[bi]))
                            )

                            val = float(dB + w_coassoc * dC)
                            if val > best_val:
                                best_val = val
                                best_g = int(g_to)

                        if best_g >= 0:
                            remove_block_from_group(bi, g_from)
                            add_block_to_group(bi, best_g)
                            improved = True

                if not improved:
                    break

            # Score and keep best
            Q = 0.0
            for g in range(K_used):
                # modularity sum within group
                comps_g = np.where(comp2g == g)[0]
                if comps_g.size == 0:
                    Q = -np.inf
                    break
                Q += float(W_B[np.ix_(comps_g, comps_g)].sum())

            if not np.isfinite(Q):
                continue

            gvec = build_gvec(comp2g)
            Z = partition_vector_to_2d_matrix(gvec)
            meta = {
                "r_min": r_min,
                "r_max": r_max,
                "K_used": K_used,
                "objective_B_sum": float(Q),
                "fingerprint_decimals": int(fingerprint_decimals),
                "allow_block_splitting": bool(allow_block_splitting),
                "seed": int(seed),
            }

            if best is None or meta["objective_B_sum"] > best[1]["objective_B_sum"]:
                best = (Z, meta)
    if best is None:
        # rerun with more adventurous Ks
        if clustering_Ks == range(K, K + 8, 2):
            return best
        return very_fortunate_descent_legacy(
            wz, A, a, m, K, R, R_bounds,
            must_link, cannot_link,
            seed, fingerprint_decimals,
            allow_block_splitting, max_K_increase,
            restarts, local_iters, w_coassoc,
            clustering_Ks = range(K, K + 8, 2),
            clustering_seeds = clustering_seeds,
            clustering_methods = clustering_methods,
            wz_is_C_node = wz_is_C_node
        )
    return best

def very_fortunate_descent(
    wz: np.ndarray,
    A: np.ndarray,
    a: np.ndarray,
    m: float,
    # constraint parameters
    K: int = None,
    R: int = None,
    R_bounds: tuple|None = None,
    must_link: Sequence[Tuple[int, int]] = [],
    cannot_link: Sequence[Tuple[int, int]] = [],
    seed: int | None = None,
    fingerprint_decimals: int = 6,
    allow_block_splitting: bool = True,
    max_K_increase: int = 0, # If 0, keep K_used == K
    # greediness-defining parameters
    restarts: int = 6,
    local_iters: int = 60,
    # tie-break toward co-association cohesion
    w_coassoc: float = 0.05,
    # clustering parameters
    clustering_Ks: Sequence[int] = None,
    clustering_seeds: Sequence[int] = (0, 1, 2),
    clustering_methods: Sequence[str] = ("kmeans", "gmm", "spectral"),
    # use wz as the cluster identifier
    wz_is_C_node: bool= False,
    # maximum number of tabu steps
    tabu_max_steps: int = 60,
    # perturb solution slightly
    shake_rounds: int = 3,
    orbit_fallback: bool = False,
) -> Optional[Tuple[np.ndarray, Dict[str, Any]]]:
    """
    Builds a feasible decomposition column from co-association structure and tabu local search.
    This version has escape mechanisms for common traps: it can swap for improvements, 
    eject one item to unblock moves, use tabu steps to leave local optima, split coarse 
    fingerprint blocks as needed, and shake/re-optimize to find missed solution basins.

    Parameters
    ----------
    wz : numpy.ndarray
        Input node-level matrix.
    A : numpy.ndarray
        Original adjacency matrix.
    a : numpy.ndarray
        Degree-like vector used in the modularity construction.
    m : float
        Modularity scaling constant.
    K : int or None
        Number of communities.
    R : int or None
        Width of the allowed cluster-size range. Also corresponds to the load balance tightness (smaller R implies tighter load balance).
        For a selected cluster count, the lower and upper bounds are computed from the corresponding
        balanced range rule.
    R_bounds : tuple or None
        Lower and upper limits on the number of nodes in communities.
    must_link : sequence of tuple of int
        Must-link pairs.
    cannot_link : sequence of tuple of int
        Cannot-link pairs.
    seed : int, default=0
        Random seed.
    fingerprint_decimals : int, default=6
        Decimal rounding used to form fingerprint blocks.
    allow_block_splitting : bool, default=True
        If True, allow refinement of coarse fingerprint blocks.
    max_K_increase : int, default=0
        Maximum increase above the baseline K when the K-constraint is active.
    restarts : int, default=6
        Number of constructive restarts.
    local_iters : int, default=60
        Base local-search iteration parameter.
    w_coassoc : float, default=0.05
        Weight of the co-association cohesion term in local decisions.
    clustering_Ks : sequence of int or None, default=None
        Community counts used to build the co-association matrix.
    clustering_seeds : sequence of int, default=(0, 1, 2)
        Seeds used during co-association construction.
    clustering_methods : sequence of str, default=("kmeans", "gmm", "spectral")
        Clustering methods used during co-association construction.
    wz_is_C_node : bool, default=False
        If True, treat ``wz`` directly as the node-level co-association matrix.
    tabu_max_steps : int, default=60
        Maximum tabu-search steps.
    shake_rounds : int, default=3
        Number of perturb-and-improve rounds.
    orbit_fallback : bool, default=False
        If True, build a fallback co-association proxy from orbits when needed.

    Returns
    -------
    tuple of (numpy.ndarray, dict) or None
        A pair ``(Z_col, meta)`` if a feasible column is found, else ``None``.
    """

    rng = np.random.default_rng(seed)

    wz = np.asarray(wz, dtype=float)
    N = int(wz.shape[0])
    if N == 0:
        g0 = np.zeros(0, dtype=int)
        return partition_vector_to_2d_matrix(g0), {"r_min": 0, "r_max": 0, "K_used": 0}

    sym = _symmetrize_unitdiag(wz)

    # r_min, r_max = _range_bounds_from_KR(N, K, R)
    if R_bounds is None and K is not None and R is not None:
        r_min, r_max = _range_bounds_from_KR(N, K, R)
    elif R_bounds is not None:
        r_min, r_max = R_bounds
    else:
        raise ValueError("The provided combination of load balancing parameters are invalid.")

    k_lo, k_hi = _feasible_K_range(N, r_min, r_max)

    if k_lo > k_hi:
        return None

    if clustering_Ks is None:
        clustering_Ks = (max(2, k_lo), min(k_hi, K), min(k_hi, K + 2))

    must_link = [_normalize_pair(i, j) for (i, j) in (must_link or [])]
    cannot_link = [_normalize_pair(i, j) for (i, j) in (cannot_link or [])]

    comp = _build_components(N, must_link, cannot_link)
    if comp is None:
        return None

    Cn = int(comp["C"])
    if Cn == 0:
        g0 = np.zeros(N, dtype=int)
        return partition_vector_to_2d_matrix(g0), {"r_min": r_min, "r_max": r_max, "K_used": 0}

    csz = np.asarray(comp["csz"], dtype=int)
    if int(csz.max()) > r_max:
        return None

    # Co-association always (fallback to sym(wz) if sklearn unavailable)
    if not wz_is_C_node:
        C_node = _build_coassociation_matrix(
            sym,
            n_components_list=clustering_Ks,
            seeds=clustering_seeds,
            methods=clustering_methods
        )
    else:
        C_node = wz

    if C_node is None:
        if orbit_fallback:
            symmetry = weighted_constraint_orbits(A)
            C_node = partition_vector_to_2d_matrix(symmetry.rep)
        else:
            C_node = sym.copy()
    C_node = _symmetrize_unitdiag(C_node)

    C_comp = _component_matrices_from_node_matrix(C_node, comp) # average co-association between components
    W_B = _component_sum_matrix_B(A, a, m, comp) # sum modularity matrix between components

    use_bitmask = bool(comp["use_bitmask"])
    forb = comp["forb_mask"]
    comp_bit = comp["comp_bit"]
    comps = comp["comps"]

    def build_gvec(comp2g: np.ndarray) -> np.ndarray:
        g = np.empty(N, dtype=int)
        for c in range(Cn):
            g[comps[c]] = int(comp2g[c])
        return g

    def group_ok_add(block_forb: int, block_mask: int, g: int, in_mask: List[int]) -> bool:
        return (block_forb & int(in_mask[g])) == 0

    def group_ok_add_set(block_forb_set: set, g: int, in_set: List[set]) -> bool:
        return len(block_forb_set & in_set[g]) == 0

    def block_internal_sum(M: np.ndarray, b: List[int]) -> float:
        idx = np.asarray(b, dtype=int)
        return float(M[np.ix_(idx, idx)].sum())

    best = None

    K0 = min(max(K, k_lo), k_hi)
    K_end = min(k_hi, K0 + int(max_K_increase))
    for K_used in range(K0, K_end + 1):
        if not (K_used * r_min <= N <= K_used * r_max):
            continue
        if r_min > 0 and K_used > Cn:
            continue

        target = _target_sizes_from_bounds(N, K_used, r_min, r_max)

        # Build fingerprint blocks once per K_used; splitting can depend on K_used
        base_blocks = _fingerprint_blocks_from_rounded_rows(C_comp, comp, fingerprint_decimals=fingerprint_decimals, r_max=r_max)
        base_blocks = _ensure_at_least_K_blocks(base_blocks, K_used, csz)

        for rr in range(int(restarts)):
            # Copy blocks per restart (they can be split on demand)
            blocks = [list(b) for b in base_blocks]
            Bn = len(blocks)

            if Bn < K_used:
                continue

            # Precompute per-block data
            b_size = np.array([int(sum(int(csz[c]) for c in b)) for b in blocks], dtype=int)
            b_ncomp = np.array([len(b) for b in blocks], dtype=int)

            if use_bitmask:
                b_mask = []
                b_forb = []
                for b in blocks:
                    cm = 0
                    fm = 0
                    for c in b:
                        cm |= int(comp_bit[c])
                    for c in b:
                        fm |= int(forb[c])
                    b_mask.append(cm)
                    b_forb.append(fm)
                b_mask = np.asarray(b_mask, dtype=object)
                b_forb = np.asarray(b_forb, dtype=object)
            else:
                b_mask = None
                b_forb_sets = []
                for b in blocks:
                    s = set(b)
                    f = set()
                    for c in b:
                        f |= set(forb[c])
                    b_forb_sets.append(f)

            b_intB = np.array([block_internal_sum(W_B, b) for b in blocks], dtype=float)
            b_intC = np.array([block_internal_sum(C_comp, b) for b in blocks], dtype=float)

            # Group state
            gsz = np.zeros(K_used, dtype=int)
            ncomp_g = np.zeros(K_used, dtype=int)
            comp2g = -np.ones(Cn, dtype=int)
            members_b: List[List[int]] = [[] for _ in range(K_used)]

            # Membership representation over components
            if use_bitmask:
                in_mask = [0] * K_used
            else:
                in_set = [set() for _ in range(K_used)]

            # For fast gains: sum of columns for current group membership
            sumB = [np.zeros(Cn, dtype=float) for _ in range(K_used)]
            sumC = [np.zeros(Cn, dtype=float) for _ in range(K_used)]

            # For cohesion: total sum over (c,d) in group of C_comp[c,d]
            totC = np.zeros(K_used, dtype=float)
            totB = np.zeros(K_used, dtype=float) # NEW: modularity internal sums
            block2g = [-1] * len(blocks) # NEW: block -> group id

            def add_block_to_group(bi: int, g: int):
                idx = np.asarray(blocks[bi], dtype=int)

                # internal-sum updates use sums BEFORE adding
                crossC = float(sumC[g][idx].sum())
                totC[g] += 2.0 * crossC + float(b_intC[bi])

                crossB = float(sumB[g][idx].sum())
                totB[g] += 2.0 * crossB + float(b_intB[bi])

                for c in idx:
                    comp2g[int(c)] = g
                members_b[g].append(bi)
                block2g[bi] = g
                gsz[g] += int(b_size[bi])
                ncomp_g[g] += int(b_ncomp[bi])

                if use_bitmask:
                    in_mask[g] |= int(b_mask[bi])
                else:
                    in_set[g] |= set(idx.tolist())

                for c in idx:
                    sumB[g] += W_B[:, int(c)]
                    sumC[g] += C_comp[:, int(c)]

            def remove_block_from_group(bi: int, g: int):
                idx = np.asarray(blocks[bi], dtype=int)

                # sums currently include the block itself; remove cross-to-rest + internal
                crossC_to_rest = float(sumC[g][idx].sum()) - float(b_intC[bi])
                totC[g] -= 2.0 * crossC_to_rest + float(b_intC[bi])

                crossB_to_rest = float(sumB[g][idx].sum()) - float(b_intB[bi])
                totB[g] -= 2.0 * crossB_to_rest + float(b_intB[bi])

                for c in idx:
                    comp2g[int(c)] = -1
                members_b[g].remove(bi)
                block2g[bi] = -1
                gsz[g] -= int(b_size[bi])
                ncomp_g[g] -= int(b_ncomp[bi])

                if use_bitmask:
                    in_mask[g] ^= int(b_mask[bi])
                else:
                    in_set[g] -= set(idx.tolist())

                for c in idx:
                    sumB[g] -= W_B[:, int(c)]
                    sumC[g] -= C_comp[:, int(c)]

            def cohesion(g: int) -> float:
                k = int(ncomp_g[g])
                if k <= 1:
                    return 0.0
                pair_sum = (float(totC[g]) - float(k)) / 2.0
                return (2.0 * pair_sum) / float(k * (k - 1))

            def delta_move_block(bi: int, g_from: int, g_to: int) -> float:
                idx = np.asarray(blocks[bi], dtype=int)
                s_to = float(sumB[g_to][idx].sum())
                s_fr = float(sumB[g_from][idx].sum())
                return 2.0 * (s_to - (s_fr - float(b_intB[bi])))

            def feasible_add_block(bi: int, g: int) -> bool:
                if gsz[g] + int(b_size[bi]) > r_max:
                    return False
                if use_bitmask:
                    return group_ok_add(int(b_forb[bi]), int(b_mask[bi]), g, in_mask)
                return group_ok_add_set(b_forb_sets[bi], g, in_set)

            def ensure_nonempty_seeding(order_blocks: List[int]) -> bool:
                # seed first K_used groups with one block each (empty group always feasible wrt cannot-link)
                used_blocks = set()
                gi = 0
                for bi in order_blocks:
                    if gi >= K_used:
                        break
                    if int(b_size[bi]) > r_max:
                        continue
                    add_block_to_group(int(bi), int(gi))
                    used_blocks.add(int(bi))
                    gi += 1
                return gi == K_used, used_blocks

            # Constructive assignment
            order = list(np.argsort(b_size)[::-1])
            ok_seed, seeded = ensure_nonempty_seeding(order)
            if not ok_seed:
                continue

            for bi in order:
                bi = int(bi)
                if bi in seeded:
                    continue

                # feasible destinations with forward check for r_min
                F = []
                rem_nodes = int(N - gsz.sum())
                deficit_sum = int(np.maximum(0, r_min - gsz).sum())

                for g in range(K_used):
                    if not feasible_add_block(bi, g):
                        continue
                    old_def = max(0, r_min - int(gsz[g]))
                    new_def = max(0, r_min - int(gsz[g] + b_size[bi]))
                    def2 = deficit_sum - old_def + new_def
                    if def2 <= (rem_nodes - int(b_size[bi])):
                        F.append(g)

                if not F:
                    # optional split if block too constraining
                    if allow_block_splitting and len(blocks[bi]) > 1:
                        # split into singleton component blocks
                        comps_b = blocks[bi]
                        # replace bi with first singleton, append others
                        blocks[bi] = [comps_b[0]]
                        for c in comps_b[1:]:
                            blocks.append([c])
                        # rebuild and restart this restart
                        ok_seed = False
                    else:
                        ok_seed = False
                    break

                idx = np.asarray(blocks[bi], dtype=int)
                scored = []
                for g in F:
                    gainB = float(2.0 * sumB[g][idx].sum())
                    gainC = float(sumC[g][idx].sum())
                    tgt_pen = abs(int(gsz[g] + b_size[bi]) - int(target[g]))
                    fill = 0 if gsz[g] < r_min else 1
                    scored.append((fill, -(gainB + w_coassoc * gainC), tgt_pen, float(rng.random()), g))
                scored.sort()
                g_best = int(scored[0][-1])
                add_block_to_group(bi, g_best)

            if not ok_seed:
                continue

            # Repair deficits using cohesion-driven donors; split blocks only if needed
            def repair_min_sizes() -> bool:
                max_steps = 20000
                step = 0
                while step < max_steps:
                    step += 1
                    if step > math.sqrt(max_steps):
                        print(f"`repair_min_sizes` has been running for {step - 1} steps.")
                    under = [g for g in range(K_used) if gsz[g] < r_min]
                    if not under:
                        return True
                    g_need = min(under, key=lambda g: int(gsz[g]))
                    donors = [g for g in range(K_used) if gsz[g] > r_min]
                    if not donors:
                        return False
                    donors.sort(key=lambda g: cohesion(g))

                    moved = False
                    for g_from in donors:
                        # pick weakest-attached block from low-cohesion donor
                        cand_blocks = list(members_b[g_from])
                        rng.shuffle(cand_blocks)

                        def block_attachment(bi: int) -> float:
                            idx = np.asarray(blocks[bi], dtype=int)
                            # attachment to donor excluding internal block
                            cross_to_group = float(sumC[g_from][idx].sum())
                            cross_to_rest = cross_to_group - float(b_intC[bi])
                            denom = max(1, int(ncomp_g[g_from]) - int(b_ncomp[bi]))
                            return cross_to_rest / float(max(1, int(b_ncomp[bi]) * denom))

                        cand_blocks.sort(key=lambda bi: (block_attachment(int(bi)), int(b_size[int(bi)])))

                        for bi in cand_blocks:
                            bi = int(bi)
                            if gsz[g_from] - int(b_size[bi]) < r_min:
                                continue
                            if not feasible_add_block(bi, g_need):
                                # attempt to split and try moving a piece
                                if allow_block_splitting and len(blocks[bi]) > 1:
                                    comps_b = blocks[bi]
                                    # remove whole block, split into singleton blocks staying in donor, then move one singleton
                                    remove_block_from_group(bi, g_from)
                                    blocks[bi] = [comps_b[0]]
                                    add_block_to_group(bi, g_from)
                                    for c in comps_b[1:]:
                                        blocks.append([c])
                                        new_bi = len(blocks) - 1
                                        # append new block into donor immediately
                                        # compute minimal per-block fields on the fly for singleton
                                        # singleton internal sums:
                                        # b_intB/B and b_intC/C for singleton are just diagonal entries
                                        # keep consistent arrays by using python lists for these when splitting
                                        # Simplify: disallow splitting after construction unless you restart
                                    return False
                                continue

                            # move block
                            remove_block_from_group(bi, g_from)
                            add_block_to_group(bi, g_need)
                            moved = True
                            break

                        if moved:
                            break

                    if not moved:
                        return False
                return False

            if not repair_min_sizes():
                continue

            # -----------------------------
            # Strong local search (tabu + swaps + ejections + on-demand splitting) + perturb-and-improve
            # -----------------------------

            def delta_move_generic(sumX, b_intX, bi: int, g_from: int, g_to: int) -> float:
                idx = np.asarray(blocks[bi], dtype=int)
                s_to = float(sumX[g_to][idx].sum())
                s_fr = float(sumX[g_from][idx].sum())
                return 2.0 * (s_to - (s_fr - float(b_intX[bi])))

            def attachment_to_group(bi: int, g: int) -> float:
                idx = np.asarray(blocks[bi], dtype=int)
                cross = float(sumC[g][idx].sum())
                cross_to_rest = cross - float(b_intC[bi])
                denom = max(1, int(ncomp_g[g]) - int(b_ncomp[bi]))
                return cross_to_rest / float(max(1, int(b_ncomp[bi]) * denom))

            def cross_sum(M: np.ndarray, bi: int, bj: int) -> float:
                ii = np.asarray(blocks[bi], dtype=int)
                jj = np.asarray(blocks[bj], dtype=int)
                return float(M[np.ix_(ii, jj)].sum())

            def feasible_add_block_after_removal(bi: int, g: int, bj_remove: Optional[int]) -> bool:
                # checks feasibility of placing bi into group g, assuming block bj_remove is removed from g (or None)
                sz_after = int(gsz[g]) - (int(b_size[bj_remove]) if bj_remove is not None else 0) + int(b_size[bi])
                if sz_after > r_max:
                    return False
                if sz_after < r_min:
                    return False
                if use_bitmask:
                    mask_excl = int(in_mask[g]) ^ (int(b_mask[bj_remove]) if bj_remove is not None else 0)
                    return (int(b_forb[bi]) & mask_excl) == 0
                else:
                    S = set(in_set[g])
                    if bj_remove is not None:
                        S -= set(blocks[bj_remove])
                    return len(set(b_forb_sets[bi]) & S) == 0

            def feasible_add_block_to_group(bi: int, g: int) -> bool:
                if gsz[g] + int(b_size[bi]) > r_max:
                    return False
                if gsz[g] + int(b_size[bi]) < r_min and (N - int(gsz.sum()) + int(b_size[bi])) < int(np.maximum(0, r_min - gsz).sum()):
                    pass
                if use_bitmask:
                    return (int(b_forb[bi]) & int(in_mask[g])) == 0
                return len(set(b_forb_sets[bi]) & in_set[g]) == 0

            def apply_move(bi: int, g_from: int, g_to: int):
                remove_block_from_group(bi, g_from)
                add_block_to_group(bi, g_to)

            def apply_swap(bi: int, bj: int, g1: int, g2: int):
                # remove both then add both avoids intermediate mask issues
                remove_block_from_group(bi, g1)
                remove_block_from_group(bj, g2)
                add_block_to_group(bi, g2)
                add_block_to_group(bj, g1)

            def apply_ejection(bi: int, g_from: int, g_to: int, bj: int, g_k: int):
                # eject bj from g_to to g_k, then move bi from g_from to g_to
                remove_block_from_group(bj, g_to)
                add_block_to_group(bj, g_k)
                remove_block_from_group(bi, g_from)
                add_block_to_group(bi, g_to)

            def split_block_into_singletons_in_place(bi: int) -> bool:
                # Keeps all pieces in the same group as the original block; only refines the neighborhood.
                if not allow_block_splitting:
                    return False
                if len(blocks[bi]) <= 1:
                    return False
                g = int(block2g[bi])
                if g < 0:
                    return False

                comps_b = list(blocks[bi])
                # Replace block bi with first singleton
                c0 = int(comps_b[0])
                blocks[bi] = [c0]

                def singleton_props(c: int):
                    if use_bitmask:
                        mask = int(comp_bit[c])
                        forb_m = int(forb[c])
                        return mask, forb_m
                    else:
                        return None, set(forb[c])

                # Update properties for bi (now singleton)
                b_size[bi] = int(csz[c0])
                b_ncomp[bi] = 1
                b_intB[bi] = float(W_B[c0, c0])
                b_intC[bi] = float(C_comp[c0, c0])
                if use_bitmask:
                    b_mask[bi], b_forb[bi] = singleton_props(c0)
                else:
                    b_forb_sets[bi] = singleton_props(c0)[1]

                # Append remaining singleton blocks and assign to same group
                # Group membership over components does not change; sums/masks/totals remain consistent.
                pos = members_b[g].index(bi)
                tail = []
                for c in comps_b[1:]:
                    c = int(c)
                    blocks.append([c])
                    new_bi = len(blocks) - 1

                    b_size.append(int(csz[c]))
                    b_ncomp.append(1)
                    b_intB.append(float(W_B[c, c]))
                    b_intC.append(float(C_comp[c, c]))
                    if use_bitmask:
                        mk, fm = singleton_props(c)
                        b_mask.append(mk)
                        b_forb.append(fm)
                    else:
                        b_forb_sets.append(set(forb[c]))

                    block2g.append(g)
                    tail.append(new_bi)

                # Update group block list: replace bi with [bi] + tail at same position
                members_b[g] = members_b[g][:pos] + [bi] + tail + members_b[g][pos + 1:]
                return True

            def candidate_blocks(g: int, L: int) -> List[int]:
                cand = list(members_b[g])
                # weakest attachment first
                cand.sort(key=lambda bi: (attachment_to_group(int(bi), g), int(b_size[int(bi)])))
                return cand[: min(L, len(cand))]

            # Tabu structure: forbid moving a block back to its previous group for some tenure
            tabu = {}  # (bi, forbidden_group) -> expire_step

            def tabu_forbidden(bi: int, g_forbidden: int, step: int) -> bool:
                return tabu.get((int(bi), int(g_forbidden)), -1) > step

            def set_tabu(bi: int, g_forbidden: int, step: int, tenure: int):
                tabu[(int(bi), int(g_forbidden))] = step + tenure

            def current_total() -> float:
                return float(totB.sum() + w_coassoc * totC.sum())

            def current_B_sum() -> float:
                return float(totB.sum())

            # Main improvement routine (tabu)
            def improve_with_tabu(max_steps: int, tenure: int, L_blocks: int, L_groups: int) -> Tuple[float, float, np.ndarray]:
                best_total = current_total()
                best_B = current_B_sum()
                best_comp2g = comp2g.copy()

                no_improve = 0

                for step in range(int(max_steps)):
                    base_total = current_total()

                    # Prefer exploring from low cohesion groups
                    g_order = list(range(K_used))
                    g_order.sort(key=lambda g: cohesion(g))

                    best_move = None  # ("move", bi, g_from, g_to, delta)
                    best_swap = None  # ("swap", bi, bj, g1, g2, delta)
                    best_eject = None # ("eject", bi, g_from, g_to, bj, gk, delta)

                    # --- 1) single-block moves ---
                    for g_from in g_order:
                        for bi in candidate_blocks(g_from, L_blocks):
                            bi = int(bi)
                            if gsz[g_from] - int(b_size[bi]) < r_min:
                                continue

                            # candidate destinations: try most "promising" first via gain proxy
                            dests = list(range(K_used))
                            rng.shuffle(dests)
                            for g_to in dests[: min(L_groups, K_used)]:
                                if g_to == g_from:
                                    continue
                                if not feasible_add_block_to_group(bi, g_to):
                                    continue

                                dB = delta_move_generic(sumB, b_intB, bi, g_from, g_to)
                                dC = delta_move_generic(sumC, b_intC, bi, g_from, g_to)
                                d = float(dB + w_coassoc * dC)

                                # tabu: forbid immediate return to g_from
                                if tabu_forbidden(bi, g_to, step) and (base_total + d) <= best_total:
                                    continue

                                if best_move is None or d > best_move[-1]:
                                    best_move = ("move", bi, g_from, g_to, d)

                    # --- 2) swaps (2-opt) ---
                    # sample group pairs anchored on low cohesion groups
                    for g1 in g_order[: max(1, K_used // 2)]:
                        others = list(range(K_used))
                        rng.shuffle(others)
                        for g2 in others[: min(K_used, 4)]:
                            if g2 == g1:
                                continue
                            if not members_b[g1] or not members_b[g2]:
                                continue

                            cand1 = candidate_blocks(g1, L_blocks)
                            cand2 = candidate_blocks(g2, L_blocks)

                            for bi in cand1:
                                bi = int(bi)
                                for bj in cand2:
                                    bj = int(bj)
                                    # size feasibility after swap
                                    s1 = int(gsz[g1]) - int(b_size[bi]) + int(b_size[bj])
                                    s2 = int(gsz[g2]) - int(b_size[bj]) + int(b_size[bi])
                                    if not (r_min <= s1 <= r_max and r_min <= s2 <= r_max):
                                        continue

                                    # cannot-link feasibility: bi into g2 without bj; bj into g1 without bi
                                    if not feasible_add_block_after_removal(bi, g2, bj):
                                        continue
                                    if not feasible_add_block_after_removal(bj, g1, bi):
                                        continue

                                    # delta for swap using cross sums
                                    STB = cross_sum(W_B, bi, bj)
                                    STC = cross_sum(C_comp, bi, bj)

                                    S_idx = np.asarray(blocks[bi], dtype=int)
                                    T_idx = np.asarray(blocks[bj], dtype=int)

                                    SA = float(sumB[g1][S_idx].sum())
                                    SB = float(sumB[g2][S_idx].sum())
                                    TA = float(sumB[g1][T_idx].sum())
                                    TB = float(sumB[g2][T_idx].sum())
                                    dB = 2.0 * ((TA - SA) + (SB - TB) + float(b_intB[bi]) + float(b_intB[bj]) - 2.0 * STB)

                                    SA = float(sumC[g1][S_idx].sum())
                                    SB = float(sumC[g2][S_idx].sum())
                                    TA = float(sumC[g1][T_idx].sum())
                                    TB = float(sumC[g2][T_idx].sum())
                                    dC = 2.0 * ((TA - SA) + (SB - TB) + float(b_intC[bi]) + float(b_intC[bj]) - 2.0 * STC)

                                    d = float(dB + w_coassoc * dC)

                                    if (tabu_forbidden(bi, g2, step) or tabu_forbidden(bj, g1, step)) and (base_total + d) <= best_total:
                                        continue

                                    if best_swap is None or d > best_swap[-1]:
                                        best_swap = ("swap", bi, bj, g1, g2, d)

                    # --- 3) ejection chains (move + eject) ---
                    for g_from in g_order:
                        for bi in candidate_blocks(g_from, L_blocks):
                            bi = int(bi)
                            if gsz[g_from] - int(b_size[bi]) < r_min:
                                continue

                            dests = list(range(K_used))
                            rng.shuffle(dests)
                            for g_to in dests[: min(L_groups, K_used)]:
                                if g_to == g_from:
                                    continue
                                # if direct feasible, skip (covered by move); here target hard cases (overflow/conflict)
                                if feasible_add_block_to_group(bi, g_to):
                                    continue

                                # try ejecting one block out of g_to to make room and/or remove conflict
                                if not members_b[g_to]:
                                    continue

                                # candidate ejections: weakest attachment in g_to
                                eject_cand = candidate_blocks(g_to, L_blocks)

                                for bj in eject_cand:
                                    bj = int(bj)

                                    # g_to after removing bj then adding bi must satisfy size and cannot-links
                                    if not feasible_add_block_after_removal(bi, g_to, bj):
                                        continue

                                    # choose destination gk for bj
                                    gks = list(range(K_used))
                                    rng.shuffle(gks)
                                    for gk in gks:
                                        if gk == g_to:
                                            continue
                                        # size feasibility for bj into gk
                                        if gsz[gk] + int(b_size[bj]) > r_max:
                                            continue
                                        if gk == g_from:
                                            # bj goes to g_from but bi leaves; check final size bounds
                                            s_from = int(gsz[g_from]) - int(b_size[bi]) + int(b_size[bj])
                                        else:
                                            s_from = int(gsz[g_from]) - int(b_size[bi])
                                        if s_from < r_min:
                                            continue
                                        if gk != g_from and (gsz[g_from] - int(b_size[bi]) < r_min):
                                            continue

                                        # cannot-link feasibility for bj into gk (account for bi leaving if gk==g_from)
                                        if use_bitmask:
                                            mask_gk = int(in_mask[gk])
                                            if gk == g_from:
                                                mask_gk ^= int(b_mask[bi])
                                            if (int(b_forb[bj]) & mask_gk) != 0:
                                                continue
                                        else:
                                            Sgk = set(in_set[gk])
                                            if gk == g_from:
                                                Sgk -= set(blocks[bi])
                                            if len(set(b_forb_sets[bj]) & Sgk) != 0:
                                                continue

                                        # delta: move bj g_to->gk, then move bi g_from->(g_to without bj),
                                        # with correction if gk==g_from.
                                        STB = cross_sum(W_B, bi, bj)
                                        STC = cross_sum(C_comp, bi, bj)

                                        dB_bj = delta_move_generic(sumB, b_intB, bj, g_to, gk)
                                        dC_bj = delta_move_generic(sumC, b_intC, bj, g_to, gk)

                                        S_idx = np.asarray(blocks[bi], dtype=int)

                                        to_crossB = float(sumB[g_to][S_idx].sum()) - STB
                                        fr_crossB = float(sumB[g_from][S_idx].sum()) + (STB if gk == g_from else 0.0)
                                        dB_bi = 2.0 * (to_crossB - (fr_crossB - float(b_intB[bi])))

                                        to_crossC = float(sumC[g_to][S_idx].sum()) - STC
                                        fr_crossC = float(sumC[g_from][S_idx].sum()) + (STC if gk == g_from else 0.0)
                                        dC_bi = 2.0 * (to_crossC - (fr_crossC - float(b_intC[bi])))

                                        d = float((dB_bj + dB_bi) + w_coassoc * (dC_bj + dC_bi))

                                        if (tabu_forbidden(bi, g_to, step) or tabu_forbidden(bj, gk, step)) and (base_total + d) <= best_total:
                                            continue

                                        if best_eject is None or d > best_eject[-1]:
                                            best_eject = ("eject", bi, g_from, g_to, bj, gk, d)

                    # choose best admissible action
                    best_action = None
                    for cand in (best_move, best_swap, best_eject):
                        if cand is None:
                            continue
                        if best_action is None or cand[-1] > best_action[-1]:
                            best_action = cand

                    if best_action is None:
                        # On-demand splitting to expand neighborhood
                        did_split = False
                        if allow_block_splitting:
                            # choose a multi-comp block in a low-cohesion group with weakest attachment
                            for g in g_order:
                                multi = [int(bi) for bi in members_b[g] if len(blocks[int(bi)]) > 1]
                                if not multi:
                                    continue
                                multi.sort(key=lambda bi: attachment_to_group(bi, g))
                                if split_block_into_singletons_in_place(multi[0]):
                                    did_split = True
                                    break
                        if did_split:
                            continue
                        break

                    # apply chosen action (tabu allows non-improving moves; stop if it stagnates)
                    kind = best_action[0]
                    d = float(best_action[-1])

                    if kind == "move":
                        _, bi, g_from, g_to, _ = best_action
                        apply_move(int(bi), int(g_from), int(g_to))
                        set_tabu(int(bi), int(g_from), step, tenure)

                    elif kind == "swap":
                        _, bi, bj, g1, g2, _ = best_action
                        apply_swap(int(bi), int(bj), int(g1), int(g2))
                        set_tabu(int(bi), int(g1), step, tenure)
                        set_tabu(int(bj), int(g2), step, tenure)

                    else:  # "eject"
                        _, bi, g_from, g_to, bj, gk, _ = best_action
                        apply_ejection(int(bi), int(g_from), int(g_to), int(bj), int(gk))
                        set_tabu(int(bi), int(g_from), step, tenure)
                        set_tabu(int(bj), int(g_to), step, tenure)

                    # record best
                    curT = current_total()
                    if curT > best_total + 1e-12:
                        best_total = curT
                        best_B = current_B_sum()
                        best_comp2g = comp2g.copy()
                        no_improve = 0
                    else:
                        no_improve += 1
                        if no_improve > max_steps // 4:
                            break

                return best_total, best_B, best_comp2g

            # Convert numpy arrays to lists for in-place splitting support
            # (only for per-block properties; sums remain numpy arrays)
            b_size = list(b_size)
            b_ncomp = list(b_ncomp)
            b_intB = list(b_intB)
            b_intC = list(b_intC)
            if use_bitmask:
                b_mask = list(b_mask)
                b_forb = list(b_forb)
            else:
                b_forb_sets = list(b_forb_sets)

            # Perturb-and-improve rounds for `shake_rounds` iterations
            # shake_rounds = 3
            shake_moves = max(2, len(blocks) // 25)  # small, controlled perturbation

            best_local_total = current_total()
            best_local_B = current_B_sum()
            best_local_comp2g = comp2g.copy()

            for sround in range(shake_rounds):
                # Tabu parameters (small, stable defaults)
                max_steps = max(tabu_max_steps, int(local_iters) * 5)
                tenure = 7
                L_blocks = 10
                L_groups = min(K_used, 6)

                ttot, tB, comp2g_best = improve_with_tabu(max_steps=max_steps, tenure=tenure, L_blocks=L_blocks, L_groups=L_groups)

                # keep best snapshot
                if tB > best_local_B + 1e-12:
                    best_local_B = float(tB)
                    best_local_total = float(ttot)
                    best_local_comp2g = comp2g_best.copy()

                # perturb current state (shake) unless last round
                if sround == shake_rounds - 1:
                    break

                # pick low-attachment blocks and move randomly (feasible only)
                # do not destroy r_min feasibility
                g_order = list(range(K_used))
                g_order.sort(key=lambda g: cohesion(g))

                moved = 0
                for g_from in g_order:
                    cand = candidate_blocks(g_from, L=20)
                    for bi in cand:
                        bi = int(bi)
                        if moved >= shake_moves:
                            break
                        if gsz[g_from] - int(b_size[bi]) < r_min:
                            continue

                        dests = list(range(K_used))
                        rng.shuffle(dests)
                        for g_to in dests:
                            if g_to == g_from:
                                continue
                            if feasible_add_block_to_group(bi, g_to):
                                apply_move(bi, g_from, g_to)
                                moved += 1
                                break
                    if moved >= shake_moves:
                        break

            # Restore best snapshot for scoring/output
            comp2g = best_local_comp2g.copy()

            # Score and keep best
            Q = float(totB.sum())
            # Q = 0.0
            # for g in range(K_used):
            #     # modularity sum within group
            #     comps_g = np.where(comp2g == g)[0]
            #     if comps_g.size == 0:
            #         Q = -np.inf
            #         break
            #     Q += float(W_B[np.ix_(comps_g, comps_g)].sum())

            if not np.isfinite(Q):
                continue

            gvec = build_gvec(comp2g)
            Z = partition_vector_to_2d_matrix(gvec)
            meta = {
                "r_min": r_min,
                "r_max": r_max,
                "K_used": K_used,
                "objective_B_sum": float(Q),
                "fingerprint_decimals": int(fingerprint_decimals),
                "allow_block_splitting": bool(allow_block_splitting),
                "seed": int(seed),
            }

            if best is None or meta["objective_B_sum"] > best[1]["objective_B_sum"]:
                best = (Z, meta)
    if best is None:
        # rerun with more adventurous Ks
        if clustering_Ks == range(K, K + 8, 2):
            return best
        return very_fortunate_descent(
            wz, A, a, m, K, R, R_bounds,
            must_link, cannot_link,
            seed, fingerprint_decimals,
            allow_block_splitting, max_K_increase,
            restarts, local_iters, w_coassoc,
            clustering_Ks = range(K, K + 8, 2),
            clustering_seeds = clustering_seeds,
            clustering_methods = clustering_methods,
            wz_is_C_node = wz_is_C_node,
            tabu_max_steps=tabu_max_steps,
            shake_rounds=shake_rounds,
            orbit_fallback=orbit_fallback
        )
    return best

def refine_partition(
    A: np.ndarray, 
    partition: np.ndarray,
    a: np.ndarray,
    m: float,
    # constraint parameters
    K: int = None,
    R: int = None,
    R_bounds: tuple|None = None,
    must_link: Sequence[Tuple[int, int]] = [],
    cannot_link: Sequence[Tuple[int, int]] = [],
    seed: int | None = None,
    fingerprint_decimals: int = 6,
    allow_block_splitting: bool = True,
    max_K_increase: int = 0, # If 0, keep K_used == K
    # greediness-defining parameters
    restarts: int = 6,
    local_iters: int = 60,
    # tie-break toward co-association cohesion
    w_coassoc: float = 0.05,
    # clustering parameters
    clustering_Ks: Sequence[int] = None,
    clustering_seeds: Sequence[int] = (0, 1, 2),
    clustering_methods: Sequence[str] = ("kmeans", "gmm", "spectral"),
    # use wz as the cluster identifier
    wz_is_C_node: bool= False,
    # maximum number of tabu steps
    tabu_max_steps: int = 60,
    # perturb solution slightly
    shake_rounds: int = 3,
    orbit_fallback: bool = False,
):
    """
    Wrapper function for ``very_fortunate_descent`` to make it compatible with existing partition refinement function structure.
    """
    # Convert to 2D vector if necessary
    if partition.ndim == 1:
        partition = partition_vector_to_2d_matrix(partition)

    out = very_fortunate_descent(
        wz=partition,
        A=A, a=a, m=m, 
        K=K, R=R, R_bounds=R_bounds,
        must_link=must_link, 
        cannot_link=cannot_link,
        seed=seed, 
        fingerprint_decimals=fingerprint_decimals,
        allow_block_splitting=allow_block_splitting, 
        max_K_increase=max_K_increase,
        restarts=restarts, 
        local_iters=local_iters, 
        w_coassoc=w_coassoc,
        clustering_Ks = clustering_Ks,
        clustering_seeds = clustering_seeds,
        clustering_methods = clustering_methods,
        wz_is_C_node = wz_is_C_node,
        tabu_max_steps=tabu_max_steps,
        shake_rounds=shake_rounds,
        orbit_fallback=orbit_fallback
    )
    if out is None:
        return None
    refined_z, meta = out
    return refined_z