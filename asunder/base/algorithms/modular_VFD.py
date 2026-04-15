"""Very Fortunate Descent (VFD):  A greedy + local search algorithm for partition refinement under pairwise constraints and optionally, balance constraints."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from asunder.base.branch_and_price.symmetry_detection import weighted_constraint_orbits
from asunder.base.utils.graph import partition_vector_to_2d_matrix


class _Feas:
    """
    Incremental feasibility checker for block-to-community operations.

    Parameters
    ----------
    r_min : int
        Minimum allowed community size.
    r_max : int
        Maximum allowed community size.
    use_bitmask : bool
        If True, use bitmask-based cannot-link checks. Otherwise, use sets.
    require_nonempty_groups : bool
        If True, disallow operations that empty a community.
    gsz : numpy.ndarray
        Current community sizes.
    members_b : list of list of int
        Block indices currently assigned to each community.
    in_mask : list of int or None
        Component-membership bitmask per community when bitmask mode is used.
    in_set : list of set of int or None
        Component-membership set per community when set mode is used.
    b_size : list of int
        Size of each block.
    b_mask : list of int or None
        Component-membership bitmask per block when bitmask mode is used.
    b_forb : list of int or None
        Cannot-link bitmask per block when bitmask mode is used.
    b_forb_sets : list of set of int or None
        Cannot-link sets per block when set mode is used.
    b_comp_set : list of set of int or None
        Component-membership sets per block when set mode is used.
    """

    def __init__(
        self,
        *,
        r_min: int,
        r_max: int,
        use_bitmask: bool,
        require_nonempty_groups: bool,
        gsz: np.ndarray,
        members_b: List[List[int]],
        in_mask: Optional[List[int]],
        in_set: Optional[List[set]],
        b_size: List[int],
        b_mask: Optional[List[int]] = None,
        b_forb: Optional[List[int]] = None,
        b_forb_sets: Optional[List[set]] = None,
        b_comp_set: Optional[List[set]] = None,
    ) -> None:
        self.r_min = int(r_min)
        self.r_max = int(r_max)
        self.use_bitmask = bool(use_bitmask)
        self.require_nonempty_groups = bool(require_nonempty_groups)

        self.gsz = gsz
        self.members_b = members_b
        self.in_mask = in_mask
        self.in_set = in_set

        self.b_size = b_size
        self.b_mask = b_mask
        self.b_forb = b_forb
        self.b_forb_sets = b_forb_sets
        self.b_comp_set = b_comp_set

    def can_remove(self, bi: int, g: int) -> bool:
        """
        Check whether block ``bi`` can be removed from community ``g``.

        Parameters
        ----------
        bi : int
            Block index.
        g : int
            Community index.

        Returns
        -------
        bool
            True if removal is feasible.
        """
        if self.require_nonempty_groups and len(self.members_b[g]) <= 1:
            return False
        return (int(self.gsz[g]) - int(self.b_size[bi])) >= self.r_min

    def can_add(self, bi: int, g: int) -> bool:
        """
        Check whether block ``bi`` can be added to community ``g``.

        Parameters
        ----------
        bi : int
            Block index.
        g : int
            Community index.

        Returns
        -------
        bool
            True if insertion is feasible.
        """
        if (int(self.gsz[g]) + int(self.b_size[bi])) > self.r_max:
            return False

        if self.use_bitmask:
            return (int(self.b_forb[bi]) & int(self.in_mask[g])) == 0

        return len(self.b_forb_sets[bi] & self.in_set[g]) == 0

    def can_add_after_removal(self, bi: int, g: int, bj_remove: int) -> bool:
        """
        Check whether block ``bi`` can be added to community ``g`` after removing
        block ``bj_remove`` from that same community.

        Parameters
        ----------
        bi : int
            Candidate entering block.
        g : int
            Community index.
        bj_remove : int
            Candidate leaving block.

        Returns
        -------
        bool
            True if the modified community remains feasible.
        """
        sz_after = int(self.gsz[g]) - int(self.b_size[bj_remove]) + int(self.b_size[bi])
        if not (self.r_min <= sz_after <= self.r_max):
            return False

        if self.use_bitmask:
            mask_excl = int(self.in_mask[g]) ^ int(self.b_mask[bj_remove])
            return (int(self.b_forb[bi]) & mask_excl) == 0

        inter = self.b_forb_sets[bi] & self.in_set[g]
        if not inter:
            return True
        return inter.issubset(self.b_comp_set[bj_remove])


def _symmetrize_unitdiag(M: np.ndarray) -> np.ndarray:
    """
    Symmetrize a matrix and set its diagonal entries to one.

    Parameters
    ----------
    M : numpy.ndarray
        Square input matrix.

    Returns
    -------
    numpy.ndarray
        Symmetric matrix equal to ``0.5 * (M + M.T)`` with unit diagonal.
    """
    M = np.asarray(M, dtype=float)
    S = 0.5 * (M + M.T)
    np.fill_diagonal(S, 1.0)
    return S

def _normalize_pair(i, j):
    """
    Return an ordered pair with the smaller index first.

    Parameters
    ----------
    i : int
        First index.
    j : int
        Second index.

    Returns
    -------
    tuple of int
        Pair ``(min(i, j), max(i, j))``.
    """
    return (i, j) if i < j else (j, i)


def _build_components(
    N: int,
    must_link: List[Tuple[int, int]],
    cannot_link: List[Tuple[int, int]],
    bitmask_C_max=4096,     # switch to sets if C is bigger AND sparse
    dense_deg_threshold=64, # keep bitmask if avg degree is high
) -> Optional[Dict[str, Any]]:
    """
    Build must-link components and component-level cannot-link structure.

    Parameters
    ----------
    N : int
        Number of original nodes.
    must_link : list of tuple of int
        Pairwise constraints requiring the linked nodes to belong to the
        same component.
    cannot_link : list of tuple of int
        Pairwise constraints requiring the linked nodes to belong to
        different components.
    bitmask_C_max : int, optional
        Maximum number of components for which bitmask-based cannot-link
        storage is used unconditionally.
    dense_deg_threshold : int, optional
        Average component-level cannot-link degree threshold above which
        bitmask storage is preferred.

    Returns
    -------
    dict or None
        Component data with keys such as ``"C"``, ``"cid"``, ``"comps"``,
        ``"csz"``, ``"use_bitmask"``, ``"forb_mask"``, and ``"comp_bit"``.
        Returns ``None`` if the link constraints are infeasible.

    Notes
    -----
    Must-link edges are compressed with union-find. Cannot-link edges are
    then lifted to the component level.
    """
    parent = np.arange(N, dtype=int)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i, j in must_link:
        i, j = int(i), int(j)
        if i == j:
            continue
        union(i, j)

    reps = np.array([find(i) for i in range(N)], dtype=int)
    _, cid = np.unique(reps, return_inverse=True)  # cid[node] -> comp id
    C = int(cid.max() + 1) if N else 0

    comps = [[] for _ in range(C)]
    for i in range(N):
        comps[int(cid[i])].append(i)
    csz = np.array([len(c) for c in comps], dtype=int)

    # Infeasible if cannot-link inside a must-link component
    for i, j in cannot_link:
        i, j = int(i), int(j)
        if i == j:
            return None
        if int(cid[i]) == int(cid[j]):
            return None

    # build forb sets first (cheap to build)
    forb_sets = [set() for _ in range(C)]
    for i, j in cannot_link:
        a, b = int(cid[int(i)]), int(cid[int(j)])
        if a == b:
            continue
        forb_sets[a].add(b)
        forb_sets[b].add(a)

    avg_deg = (sum(len(s) for s in forb_sets) / C) if C else 0.0
    use_bitmask = (C <= bitmask_C_max) or (avg_deg >= dense_deg_threshold)

    if use_bitmask:
        # Component-level cannot-link as bitmasks
        forb_mask = [0] * C
        for c in range(C):
            m = 0
            for b in forb_sets[c]:
                m |= (1 << b)
            forb_mask[c] = m
        comp_bit = [1 << c for c in range(C)]  # precompute (removes repeated shifts)
        return {
            "C": C, "cid": cid, "comps": comps, "csz": csz,
            "use_bitmask": True,
            "forb_mask": forb_mask,
            "comp_bit": comp_bit,
        }

    return {
        "C": C, "cid": cid, "comps": comps, "csz": csz,
        "use_bitmask": False,
        "forb_mask": forb_sets,
        "comp_bit": None,
    }


def _build_coassociation_matrix(
    sym_wz: np.ndarray,
    n_components_list: Sequence[int],
    seeds: Sequence[int],
    methods: Sequence[str] = ("kmeans", "gmm", "spectral"),
) -> Optional[np.ndarray]:
    """
    Estimate a co-association matrix from repeated clustering runs.

    Parameters
    ----------
    sym_wz : numpy.ndarray
        Symmetric similarity or affinity matrix.
    n_components_list : sequence of int
        Candidate numbers of clusters to try.
    seeds : sequence of int
        Random seeds for repeated runs.
    methods : sequence of str, optional
        Clustering methods to use. Supported values are ``"kmeans"``,
        ``"gmm"``, and ``"spectral"``.

    Returns
    -------
    numpy.ndarray or None
        Matrix ``C`` in ``[0, 1]`` where ``C[i, j]`` is the fraction of
        successful runs in which nodes ``i`` and ``j`` are co-clustered.
        Returns ``None`` if scikit-learn is unavailable or no run succeeds.
    """
    try:
        from sklearn.cluster import KMeans, SpectralClustering
        from sklearn.mixture import GaussianMixture
    except Exception:
        return None

    X = np.asarray(sym_wz, dtype=float)
    n = X.shape[0]
    if n == 0:
        return np.zeros((0, 0), dtype=float)

    # For spectral with precomputed affinity, ensure diagonal is 1 and nonnegative.
    A = X.copy()
    np.fill_diagonal(A, 1.0)
    A[A < 0.0] = 0.0

    C_cnt = np.zeros((n, n), dtype=np.float64)
    total = 0

    for k in n_components_list:
        if k <= 0 or k > n:
            continue
        for sd in seeds:
            for meth in methods:
                try:
                    if meth == "kmeans":
                        labels = KMeans(n_clusters=k, n_init=10, random_state=int(sd)).fit_predict(X)
                    elif meth == "gmm":
                        labels = GaussianMixture(n_components=k, random_state=int(sd)).fit(X).predict(X)
                    elif meth == "spectral":
                        labels = SpectralClustering(
                            n_clusters=k,
                            affinity="precomputed",
                            random_state=int(sd),
                            assign_labels="kmeans",
                        ).fit_predict(A)
                    else:
                        continue
                except Exception:
                    continue

                total += 1
                # accumulate co-cluster indicator
                eq = (labels[:, None] == labels[None, :]).astype(np.float64)
                C_cnt += eq

    if total == 0:
        return None

    C = C_cnt / float(total)
    np.fill_diagonal(C, 1.0)
    return C


def _component_matrices_from_node_matrix(M: np.ndarray, comp: Dict[str, Any]) -> np.ndarray:
    """
    Aggregate a node-level matrix to the component level by block averaging.

    Parameters
    ----------
    M : numpy.ndarray
        Node-level square matrix.
    comp : dict
        Component structure returned by ``_build_components``.

    Returns
    -------
    numpy.ndarray
        Component-level matrix whose entries are averages over node pairs
        between components. The diagonal is set to one.
    """
    M = np.asarray(M, dtype=float)
    N = M.shape[0]
    cid = np.asarray(comp["cid"], dtype=int)
    csz = np.asarray(comp["csz"], dtype=float)
    C = int(comp["C"])
    if C == 0:
        return np.zeros((0, 0), dtype=float)

    P = np.zeros((N, C), dtype=float)
    P[np.arange(N), cid] = 1.0
    denom = np.outer(csz, csz)
    denom[denom == 0.0] = 1.0

    S = (P.T @ M @ P) / denom
    np.fill_diagonal(S, 1.0)
    return S


def _component_sum_matrix_B(A: np.ndarray, a: np.ndarray, m: float, comp: Dict[str, Any]) -> np.ndarray:
    """
    Aggregate the modularity-style matrix to the component level by summation.

    Parameters
    ----------
    A : numpy.ndarray
        Node-level adjacency or weight matrix of shape ``(N, N)``.
    a : numpy.ndarray
        Node weight vector of shape ``(N,)``.
    m : float
        Normalizing scalar used in ``A - aa^T / m``.
    comp : dict
        Component structure returned by ``_build_components``.

    Returns
    -------
    numpy.ndarray
        Component-level summed matrix ``P.T @ B @ P`` where
        ``B = A - aa^T / m``.

    Raises
    ------
    ValueError
        If the input shapes are inconsistent or if ``m`` is zero.
    """
    A = np.asarray(A, dtype=float)
    a = np.asarray(a, dtype=float).reshape(-1)
    N = A.shape[0]
    if A.shape != (N, N) or a.shape[0] != N:
        raise ValueError("A must be (N,N) and a must be (N,).")
    if m == 0:
        raise ValueError("m must be nonzero.")
    B = A - np.outer(a, a) / float(m)

    cid = np.asarray(comp["cid"], dtype=int)
    C = int(comp["C"])
    if C == 0:
        return np.zeros((0, 0), dtype=float)

    P = np.zeros((N, C), dtype=float)
    P[np.arange(N), cid] = 1.0
    return P.T @ B @ P  # sum between components


def _fingerprint_blocks_from_rounded_rows(
    C_comp: np.ndarray,
    comp: Dict[str, Any],
    fingerprint_decimals: int,
    r_max: Any,
) -> List[List[int]]:
    """
    Form component blocks from rounded co-association fingerprints.

    Parameters
    ----------
    C_comp : numpy.ndarray
        Component-level co-association matrix.
    comp : dict
        Component structure returned by ``_build_components``.
    fingerprint_decimals : int
        Number of decimals used to round component rows before grouping.
    r_max : Any
        Maximum allowed total node count per block. If ``None``, no
        effective size cap is imposed.

    Returns
    -------
    list of list of int
        Blocks of component IDs with matching rounded fingerprints, split
        further to avoid cannot-link conflicts and to respect ``r_max``.
    """
    C = int(comp["C"])
    if C == 0:
        return []

    use_bitmask = bool(comp["use_bitmask"])
    forb = comp["forb_mask"]
    csz = np.asarray(comp["csz"], dtype=int)

    if r_max is None:
        r_max = int(csz.sum())  # no size-based splitting unless a bucket exceeds total size

    Q = np.round(np.asarray(C_comp, dtype=float), decimals=int(fingerprint_decimals))
    np.fill_diagonal(Q, 1.0)

    buckets: Dict[bytes, List[int]] = {}
    for c in range(C):
        buckets.setdefault(Q[c].tobytes(), []).append(c)

    blocks: List[List[int]] = []
    for nodes in buckets.values():
        # Split by cannot-link conflicts (component-level)
        sub: List[List[int]] = []
        for u in nodes:
            placed = False
            for g in sub:
                ok = True
                if use_bitmask:
                    mu = int(forb[u])
                    for v in g:
                        if (mu >> int(v)) & 1:
                            ok = False
                            break
                else:
                    fu = forb[u]
                    for v in g:
                        if v in fu:
                            ok = False
                            break
                if ok:
                    g.append(u)
                    placed = True
                    break
            if not placed:
                sub.append([u])

        # Split to respect r_max in node-count
        for g in sub:
            g.sort(key=lambda x: int(csz[x]), reverse=True)
            cur: List[int] = []
            cur_sz = 0
            for u in g:
                su = int(csz[u])
                if cur and (cur_sz + su > r_max):
                    blocks.append(sorted(cur))
                    cur = [u]
                    cur_sz = su
                else:
                    cur.append(u)
                    cur_sz += su
            if cur:
                blocks.append(sorted(cur))

    blocks.sort(key=lambda b: (-sum(int(csz[c]) for c in b), len(b), b[0]))
    return blocks

def _ensure_at_least_K_blocks(
    blocks: List[List[int]],
    K_used: int,
    csz: np.ndarray,
) -> List[List[int]]:
    """
    Split large blocks until at least ``K_used`` blocks are available.

    Parameters
    ----------
    blocks : list of list of int
        Current blocks of component IDs.
    K_used : int
        Required minimum number of blocks.
    csz : numpy.ndarray
        Component sizes indexed by component ID.

    Returns
    -------
    list of list of int
        Updated block list. If too few splittable blocks exist, the result
        may still contain fewer than ``K_used`` blocks.
    """
    blocks = [list(b) for b in blocks]
    if len(blocks) >= K_used:
        return blocks

    # Only possible if there are enough components
    total_components = sum(len(b) for b in blocks)
    if total_components < K_used:
        return blocks  # infeasibility will be handled later

    def block_size(b: List[int]) -> int:
        return int(sum(int(csz[c]) for c in b))

    while len(blocks) < K_used:
        # pick splittable largest block (len>1)
        splittable = [i for i, b in enumerate(blocks) if len(b) > 1]
        if not splittable:
            break
        i = max(splittable, key=lambda idx: block_size(blocks[idx]))
        b = blocks.pop(i)
        b.sort(key=lambda c: int(csz[c]), reverse=True)
        # split off one component
        blocks.append([b[0]])
        blocks.append(b[1:])
    return blocks

def _range_bounds_from_KR(N: int, K: int, R: int) -> Tuple[int, int]:
    """
    Compute lower and upper block-size bounds from ``(N, K, R)``.

    Parameters
    ----------
    N : int
        Number of nodes.
    K : int
        Number of partitions.
    R : int
        Allowed size range width.

    Returns
    -------
    tuple of int
        Pair ``(r_min, r_max)`` with
        ``r_min = floor(N / K - R / 2 + 1/2)`` and ``r_max = r_min + R``.

    Raises
    ------
    ValueError
        If ``K`` is not positive.
    """
    if K <= 0:
        raise ValueError("K must be positive.")
    r_min = math.floor((N / K) - (R / 2.0) + 0.5)
    r_min = max(1, r_min)
    r_max = r_min + int(R)
    return int(r_min), int(r_max)


def _lb_bounds(N: int, K: int, R: int):
    """
    Compute load-balance bounds implied by ``(N, K, R)``.

    Parameters
    ----------
    N : int
        Number of nodes.
    K : int
        Number of partitions.
    R : int
        Allowed size range width.

    Returns
    -------
    tuple of int
        Pair ``(r_min, r_max)`` used for load-balance checks.
    """
    R_min = max(1, math.floor((N / K - R / 2) + 1 / 2))
    return R_min, R_min + R

def _feasible_K_range(N: int, r_min: int, r_max: int) -> Tuple[int, int]:
    """
    Compute the feasible range of partition counts under size bounds.

    Parameters
    ----------
    N : int
        Number of nodes.
    r_min : int
        Minimum allowed partition size.
    r_max : int
        Maximum allowed partition size.

    Returns
    -------
    tuple of int
        Pair ``(k_lo, k_hi)`` such that any feasible ``K`` must satisfy
        ``k_lo <= K <= k_hi``.
    """
    if N == 0:
        return (0, 0)
    if r_max <= 0:
        return (math.inf, -math.inf)

    k_lo = math.ceil(N / r_max)

    if r_min <= 0:
        k_hi = N
    else:
        k_hi = N // r_min

    return int(k_lo), int(k_hi)

def _target_sizes_from_bounds(N: int, K_used: int, r_min: int, r_max: int) -> np.ndarray:
    """
    Construct a balanced target size vector within prescribed bounds.

    Parameters
    ----------
    N : int
        Total number of nodes.
    K_used : int
        Number of partitions.
    r_min : int
        Minimum target size per partition.
    r_max : int
        Maximum target size per partition.

    Returns
    -------
    numpy.ndarray
        Integer vector of length ``K_used`` whose entries sum to ``N`` and
        lie in ``[r_min, r_max]``.

    Raises
    ------
    ValueError
        If the bounds are infeasible for the given ``N`` and ``K_used``.

    Notes
    -----
    This target vector is intended for scoring or guidance, not as a hard
    partition-equality requirement.
    """
    if K_used == 0:
        return np.zeros(0, dtype=int)

    base = np.full(K_used, r_min, dtype=int)
    extra = N - K_used * r_min
    if extra < 0:
        raise ValueError("Infeasible: N < K_used * r_min")
    cap = r_max - r_min
    if extra > K_used * cap:
        raise ValueError("Infeasible: N > K_used * r_max")

    q, r = divmod(extra, K_used)
    base += q
    if r > 0:
        base[:r] += 1
    return base

def _greedy_split_block_by_cannot(
    block: List[int],
    *,
    use_bitmask: bool,
    forb,
    comp_bit=None,
) -> List[List[int]]:
    """
    Split a block so that no resulting sub-block contains an internal cannot-link conflict.

    Parameters
    ----------
    block : list of int
        Component indices in the block.
    use_bitmask : bool
        If True, use bitmask logic. Otherwise, use sets.
    forb : object
        Cannot-link representation from the component structure.
    comp_bit : object, optional
        Component bitmasks used in bitmask mode.

    Returns
    -------
    list of list of int
        Conflict-free sub-blocks.
    """
    if len(block) <= 1:
        return [list(block)]

    out: List[List[int]] = []

    if use_bitmask:
        out_masks: List[int] = []
        for c in block:
            c = int(c)
            cbit = int(comp_bit[c])
            cforb = int(forb[c])
            placed = False
            for i, mask in enumerate(out_masks):
                if (cforb & mask) == 0:
                    out[i].append(c)
                    out_masks[i] = mask | cbit
                    placed = True
                    break
            if not placed:
                out.append([c])
                out_masks.append(cbit)
        return out

    out_sets: List[set] = []
    for c in block:
        c = int(c)
        cforb = set(forb[c])
        placed = False
        for i, s in enumerate(out_sets):
            if not (cforb & s):
                out[i].append(c)
                s.add(c)
                placed = True
                break
        if not placed:
            out.append([c])
            out_sets.append({c})
    return out


def _make_blocks_conflict_free(
    blocks: List[List[int]],
    *,
    use_bitmask: bool,
    forb,
    comp_bit=None,
) -> List[List[int]]:
    """
    Apply cannot-link-safe splitting to every block.

    Parameters
    ----------
    blocks : list of list of int
        Candidate fingerprint blocks.
    use_bitmask : bool
        If True, use bitmask logic. Otherwise, use sets.
    forb : object
        Cannot-link representation from the component structure.
    comp_bit : object, optional
        Component bitmasks used in bitmask mode.

    Returns
    -------
    list of list of int
        Conflict-free blocks.
    """
    out: List[List[int]] = []
    for block in blocks:
        out.extend(
            _greedy_split_block_by_cannot(
                list(block),
                use_bitmask=use_bitmask,
                forb=forb,
                comp_bit=comp_bit,
            )
        )
    return out


def _resolve_k_control(
    *,
    N: int,
    Cn: int,
    K: Optional[int],
    R: Optional[int],
    use_K_constraint: bool,
    max_K_increase: int,
    clustering_Ks: Sequence[int],
    candidate_Ks: Optional[Sequence[int]],
) -> Tuple[int, int, List[int]]:
    """
    Resolve size bounds and the list of fixed-K subproblems to evaluate.

    Parameters
    ----------
    N : int
        Number of original nodes.
    Cn : int
        Number of must-link components.
    K : int or None
        Baseline number of communities.
    R : int or None
        Range parameter used by ``_range_bounds_from_KR`` when the K-constraint is active.
    use_K_constraint : bool
        If True, use K/R-derived balance bounds and only test K-neighborhood values.
        If False, remove K-derived balance bounds and instead test ``candidate_Ks``.
    max_K_increase : int
        Maximum allowed increase over the baseline K when the K-constraint is active.
    clustering_Ks : sequence of int
        Clustering sizes used to build co-association information.
    candidate_Ks : sequence of int or None
        Explicit K values to test when the K-constraint is inactive.

    Returns
    -------
    r_min : int
        Minimum allowed community size.
    r_max : int
        Maximum allowed community size.
    K_values : list of int
        K values to test in the outer loop.
    """
    if use_K_constraint:
        if K is None or R is None:
            raise ValueError("K and R must be provided when use_K_constraint=True.")

        r_min, r_max = _range_bounds_from_KR(N, K, R)
        k_lo, k_hi = _feasible_K_range(N, r_min, r_max)
        if k_lo > k_hi:
            return int(r_min), int(r_max), []

        K0 = min(max(int(K), int(k_lo)), int(k_hi))
        K_end = min(int(k_hi), K0 + int(max_K_increase))

        K_values = [
            int(k)
            for k in range(K0, K_end + 1)
            if (int(k) * int(r_min) <= N <= int(k) * int(r_max))
        ]
        return int(r_min), int(r_max), K_values

    r_min, r_max = 1, N

    if candidate_Ks is None:
        ks = {int(k) for k in (clustering_Ks or ()) if 1 <= int(k) <= Cn}
        if K is not None:
            lo = max(1, int(K) - 2)
            hi = min(int(Cn), int(K) + int(max_K_increase) + 2)
            ks.update(range(lo, hi + 1))
        if not ks:
            ks.update(range(1, min(int(Cn), 8) + 1))
        candidate_Ks = sorted(ks)

    K_values = sorted({int(k) for k in candidate_Ks if 1 <= int(k) <= int(Cn)})
    return int(r_min), int(r_max), K_values


def _objective_B_from_comp_assignment(W_B: np.ndarray, comp2g: np.ndarray, K_used: int) -> float:
    """
    Recompute the modularity-style objective from a component-to-community assignment.

    Parameters
    ----------
    W_B : numpy.ndarray
        Component-level modularity matrix.
    comp2g : numpy.ndarray
        Component-to-community assignment.
    K_used : int
        Number of active communities.

    Returns
    -------
    float
        Sum of within-community entries of ``W_B``.
    """
    total = 0.0
    for g in range(int(K_used)):
        idx = np.where(comp2g == g)[0]
        if idx.size == 0:
            return -np.inf
        total += float(W_B[np.ix_(idx, idx)].sum())
    return float(total)


def modular_very_fortunate_descent(
    wz: np.ndarray,
    A: np.ndarray,
    a: np.ndarray,
    m: float,
    K: Optional[int],
    R: Optional[int],
    must_link: Sequence[Tuple[int, int]],
    cannot_link: Sequence[Tuple[int, int]],
    seed: int = 0,
    fingerprint_decimals: int = 6,
    allow_block_splitting: bool = True,
    max_K_increase: int = 0,
    use_K_constraint: bool = True,
    candidate_Ks: Optional[Sequence[int]] = None,
    restarts: int = 6,
    local_iters: int = 60,
    w_coassoc: float = 0.05,
    clustering_Ks: Sequence[int] = None,
    clustering_seeds: Sequence[int] = (0, 1, 2),
    clustering_methods: Sequence[str] = ("kmeans", "gmm", "spectral"),
    wz_is_C_node: bool = False,
    tabu_max_steps: int = 60,
    shake_rounds: int = 3,
    orbit_fallback: bool = False,
) -> Optional[Tuple[np.ndarray, Dict[str, Any]]]:
    """
    Build a feasible decomposition column from co-association structure and local search.

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
        Baseline number of communities. Used directly only when ``use_K_constraint=True``.
        When ``use_K_constraint=False``, it is treated only as an optional search hint.
    R : int or None
        Range parameter for ``_range_bounds_from_KR`` when ``use_K_constraint=True``.
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
    use_K_constraint : bool, default=True
        If True, enforce K/R-derived balance bounds.
        If False, ignore K/R-derived balance bounds and search over ``candidate_Ks``.
    candidate_Ks : sequence of int or None, default=None
        K values to test when ``use_K_constraint=False``.
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
        return partition_vector_to_2d_matrix(g0), {
            "r_min": 0,
            "r_max": 0,
            "K_used": 0,
            "use_K_constraint": bool(use_K_constraint),
        }

    sym = _symmetrize_unitdiag(wz)

    must_link = [_normalize_pair(i, j) for (i, j) in (must_link or [])]
    cannot_link = [_normalize_pair(i, j) for (i, j) in (cannot_link or [])]

    comp = _build_components(N, must_link, cannot_link)
    if comp is None:
        return None

    Cn = int(comp["C"])
    if Cn == 0:
        g0 = np.zeros(N, dtype=int)
        return partition_vector_to_2d_matrix(g0), {
            "r_min": 0,
            "r_max": 0,
            "K_used": 0,
            "use_K_constraint": bool(use_K_constraint),
        }

    csz = np.asarray(comp["csz"], dtype=int)
    use_bitmask = bool(comp["use_bitmask"])
    forb = comp["forb_mask"]
    comp_bit = comp["comp_bit"]
    comps = comp["comps"]

    if clustering_Ks is None:
        if use_K_constraint and K is not None:
            clustering_Ks = tuple(
                sorted(
                    {
                        max(2, int(K) - 1),
                        max(2, int(K)),
                        min(N, max(2, int(K) + 2)),
                    }
                )
            )
        else:
            rough = max(2, int(round(np.sqrt(max(2, N)))))
            clustering_Ks = tuple(sorted({1, 2, min(N, rough), min(N, rough + 2)}))

    r_min, r_max, K_candidates = _resolve_k_control(
        N=N,
        Cn=Cn,
        K=K,
        R=R,
        use_K_constraint=use_K_constraint,
        max_K_increase=max_K_increase,
        clustering_Ks=clustering_Ks,
        candidate_Ks=candidate_Ks,
    )

    if not K_candidates:
        return None

    if int(csz.max()) > r_max:
        return None

    if not wz_is_C_node:
        C_node = _build_coassociation_matrix(
            sym,
            n_components_list=clustering_Ks,
            seeds=clustering_seeds,
            methods=clustering_methods,
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
    C_comp = _component_matrices_from_node_matrix(C_node, comp)
    W_B = _component_sum_matrix_B(A, a, m, comp)

    def build_gvec(comp2g: np.ndarray) -> np.ndarray:
        g = np.empty(N, dtype=int)
        for c in range(Cn):
            g[comps[c]] = int(comp2g[c])
        return g

    def block_internal_sum(M: np.ndarray, b: List[int]) -> float:
        idx = np.asarray(b, dtype=int)
        return float(M[np.ix_(idx, idx)].sum())

    best = None

    for K_used in K_candidates:
        K_used = int(K_used)
        if K_used <= 0:
            continue
        if not (K_used * r_min <= N <= K_used * r_max):
            continue
        if K_used > Cn:
            continue

        target = _target_sizes_from_bounds(N, K_used, r_min, r_max) if use_K_constraint else None

        base_blocks = _fingerprint_blocks_from_rounded_rows(
            C_comp,
            comp,
            fingerprint_decimals=fingerprint_decimals,
            r_max=r_max,
        )
        base_blocks = _make_blocks_conflict_free(
            base_blocks,
            use_bitmask=use_bitmask,
            forb=forb,
            comp_bit=comp_bit,
        )
        base_blocks = _ensure_at_least_K_blocks(base_blocks, K_used, csz)

        for _rr in range(int(restarts)):
            blocks = [list(b) for b in base_blocks]
            Bn = len(blocks)
            if Bn < K_used:
                continue

            b_size = [int(sum(int(csz[c]) for c in b)) for b in blocks]
            b_ncomp = [int(len(b)) for b in blocks]

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
                    b_mask.append(int(cm))
                    b_forb.append(int(fm))
                b_forb_sets = None
                b_comp_set = None
            else:
                b_mask = None
                b_forb = None
                b_forb_sets = []
                b_comp_set = []
                for b in blocks:
                    b_comp_set.append(set(int(c) for c in b))
                    f = set()
                    for c in b:
                        f |= set(forb[c])
                    b_forb_sets.append(f)

            b_intB = [float(block_internal_sum(W_B, b)) for b in blocks]
            b_intC = [float(block_internal_sum(C_comp, b)) for b in blocks]

            gsz = np.zeros(K_used, dtype=int)
            ncomp_g = np.zeros(K_used, dtype=int)
            comp2g = -np.ones(Cn, dtype=int)
            members_b: List[List[int]] = [[] for _ in range(K_used)]

            if use_bitmask:
                in_mask = [0] * K_used
                in_set = None
            else:
                in_mask = None
                in_set = [set() for _ in range(K_used)]

            sumB = [np.zeros(Cn, dtype=float) for _ in range(K_used)]
            sumC = [np.zeros(Cn, dtype=float) for _ in range(K_used)]

            totC = np.zeros(K_used, dtype=float)
            totB = np.zeros(K_used, dtype=float)
            block2g = [-1] * len(blocks)

            feas = _Feas(
                r_min=r_min,
                r_max=r_max,
                use_bitmask=use_bitmask,
                require_nonempty_groups=True,
                gsz=gsz,
                members_b=members_b,
                in_mask=in_mask,
                in_set=in_set,
                b_size=b_size,
                b_mask=b_mask,
                b_forb=b_forb,
                b_forb_sets=b_forb_sets,
                b_comp_set=b_comp_set,
            )

            def add_block_to_group(bi: int, g: int) -> None:
                idx = np.asarray(blocks[bi], dtype=int)

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

            def remove_block_from_group(bi: int, g: int) -> None:
                idx = np.asarray(blocks[bi], dtype=int)

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

            def ensure_nonempty_seeding(order_blocks: List[int]) -> Tuple[bool, set]:
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

            order = list(np.argsort(np.asarray(b_size))[::-1])
            ok_seed, seeded = ensure_nonempty_seeding(order)
            if not ok_seed:
                continue

            for bi in order:
                bi = int(bi)
                if bi in seeded:
                    continue

                F = []
                rem_nodes = int(N - gsz.sum())
                deficit_sum = int(np.maximum(0, r_min - gsz).sum())

                for g in range(K_used):
                    if not feas.can_add(bi, g):
                        continue
                    old_def = max(0, r_min - int(gsz[g]))
                    new_def = max(0, r_min - int(gsz[g] + b_size[bi]))
                    def2 = deficit_sum - old_def + new_def
                    if def2 <= (rem_nodes - int(b_size[bi])):
                        F.append(g)

                if not F:
                    if allow_block_splitting and len(blocks[bi]) > 1:
                        comps_b = blocks[bi]
                        blocks[bi] = [comps_b[0]]
                        for c in comps_b[1:]:
                            blocks.append([c])
                        ok_seed = False
                    else:
                        ok_seed = False
                    break

                idx = np.asarray(blocks[bi], dtype=int)
                scored = []
                for g in F:
                    gainB = float(2.0 * sumB[g][idx].sum())
                    gainC = float(sumC[g][idx].sum())
                    tgt_pen = 0 if target is None else abs(int(gsz[g] + b_size[bi]) - int(target[g]))
                    fill = 0 if gsz[g] < r_min else 1
                    scored.append((fill, -(gainB + w_coassoc * gainC), tgt_pen, float(rng.random()), g))
                scored.sort()
                g_best = int(scored[0][-1])
                add_block_to_group(bi, g_best)

            if not ok_seed:
                continue

            def repair_min_sizes() -> bool:
                max_steps = 20000
                step = 0
                while step < max_steps:
                    step += 1
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
                        cand_blocks = list(members_b[g_from])
                        rng.shuffle(cand_blocks)

                        def block_attachment(bi_: int) -> float:
                            idx_ = np.asarray(blocks[bi_], dtype=int)
                            cross_to_group = float(sumC[g_from][idx_].sum())
                            cross_to_rest = cross_to_group - float(b_intC[bi_])
                            denom = max(1, int(ncomp_g[g_from]) - int(b_ncomp[bi_]))
                            return cross_to_rest / float(max(1, int(b_ncomp[bi_]) * denom))

                        cand_blocks.sort(key=lambda bi_: (block_attachment(int(bi_)), int(b_size[int(bi_)])))

                        for bi_ in cand_blocks:
                            bi_ = int(bi_)
                            if not feas.can_remove(bi_, g_from):
                                continue
                            if not feas.can_add(bi_, g_need):
                                if allow_block_splitting and len(blocks[bi_]) > 1:
                                    return False
                                continue

                            remove_block_from_group(bi_, g_from)
                            add_block_to_group(bi_, g_need)
                            moved = True
                            break

                        if moved:
                            break

                    if not moved:
                        return False

                return False

            if not repair_min_sizes():
                continue

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

            def apply_move(bi: int, g_from: int, g_to: int) -> None:
                remove_block_from_group(bi, g_from)
                add_block_to_group(bi, g_to)

            def apply_swap(bi: int, bj: int, g1: int, g2: int) -> None:
                remove_block_from_group(bi, g1)
                remove_block_from_group(bj, g2)
                add_block_to_group(bi, g2)
                add_block_to_group(bj, g1)

            def apply_ejection(bi: int, g_from: int, g_to: int, bj: int, g_k: int) -> None:
                remove_block_from_group(bj, g_to)
                add_block_to_group(bj, g_k)
                remove_block_from_group(bi, g_from)
                add_block_to_group(bi, g_to)

            def split_block_into_singletons_in_place(bi: int) -> bool:
                if not allow_block_splitting:
                    return False
                if len(blocks[bi]) <= 1:
                    return False

                g = int(block2g[bi])
                if g < 0:
                    return False

                comps_b = list(blocks[bi])
                c0 = int(comps_b[0])
                blocks[bi] = [c0]

                b_size[bi] = int(csz[c0])
                b_ncomp[bi] = 1
                b_intB[bi] = float(W_B[c0, c0])
                b_intC[bi] = float(C_comp[c0, c0])

                if use_bitmask:
                    b_mask[bi] = int(comp_bit[c0])
                    b_forb[bi] = int(forb[c0])
                else:
                    b_comp_set[bi] = {c0}
                    b_forb_sets[bi] = set(forb[c0])

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
                        b_mask.append(int(comp_bit[c]))
                        b_forb.append(int(forb[c]))
                    else:
                        b_comp_set.append({c})
                        b_forb_sets.append(set(forb[c]))

                    block2g.append(g)
                    tail.append(new_bi)

                members_b[g] = members_b[g][:pos] + [bi] + tail + members_b[g][pos + 1:]
                return True

            def candidate_blocks(g: int, L: int) -> List[int]:
                cand = list(members_b[g])
                cand.sort(key=lambda bi_: (attachment_to_group(int(bi_), g), int(b_size[int(bi_)])))
                return cand[: min(L, len(cand))]

            tabu = {}

            def tabu_forbidden(bi: int, g_forbidden: int, step: int) -> bool:
                return tabu.get((int(bi), int(g_forbidden)), -1) > step

            def set_tabu(bi: int, g_forbidden: int, step: int, tenure: int) -> None:
                tabu[(int(bi), int(g_forbidden))] = step + tenure

            def current_total() -> float:
                return float(totB.sum() + w_coassoc * totC.sum())

            def current_B_sum() -> float:
                return float(totB.sum())

            def improve_with_tabu(
                max_steps: int,
                tenure: int,
                L_blocks: int,
                L_groups: int,
            ) -> Tuple[float, float, np.ndarray]:
                best_total = current_total()
                best_B = current_B_sum()
                best_comp2g = comp2g.copy()

                no_improve = 0

                for step in range(int(max_steps)):
                    base_total = current_total()
                    g_order = list(range(K_used))
                    g_order.sort(key=lambda g: cohesion(g))

                    best_move = None
                    best_swap = None
                    best_eject = None

                    for g_from in g_order:
                        for bi in candidate_blocks(g_from, L_blocks):
                            bi = int(bi)
                            if not feas.can_remove(bi, g_from):
                                continue

                            dests = list(range(K_used))
                            rng.shuffle(dests)

                            for g_to in dests[: min(L_groups, K_used)]:
                                if g_to == g_from:
                                    continue
                                if not feas.can_add(bi, g_to):
                                    continue

                                dB = delta_move_generic(sumB, b_intB, bi, g_from, g_to)
                                dC = delta_move_generic(sumC, b_intC, bi, g_from, g_to)
                                d = float(dB + w_coassoc * dC)

                                if tabu_forbidden(bi, g_to, step) and (base_total + d) <= best_total:
                                    continue

                                if best_move is None or d > best_move[-1]:
                                    best_move = ("move", bi, g_from, g_to, d)

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

                                    if not feas.can_add_after_removal(bi, g2, bj):
                                        continue
                                    if not feas.can_add_after_removal(bj, g1, bi):
                                        continue

                                    STB = cross_sum(W_B, bi, bj)
                                    STC = cross_sum(C_comp, bi, bj)

                                    S_idx = np.asarray(blocks[bi], dtype=int)
                                    T_idx = np.asarray(blocks[bj], dtype=int)

                                    SA = float(sumB[g1][S_idx].sum())
                                    SB = float(sumB[g2][S_idx].sum())
                                    TA = float(sumB[g1][T_idx].sum())
                                    TB = float(sumB[g2][T_idx].sum())
                                    dB = 2.0 * (
                                        (TA - SA) + (SB - TB)
                                        + float(b_intB[bi]) + float(b_intB[bj]) - 2.0 * STB
                                    )

                                    SA = float(sumC[g1][S_idx].sum())
                                    SB = float(sumC[g2][S_idx].sum())
                                    TA = float(sumC[g1][T_idx].sum())
                                    TB = float(sumC[g2][T_idx].sum())
                                    dC = 2.0 * (
                                        (TA - SA) + (SB - TB)
                                        + float(b_intC[bi]) + float(b_intC[bj]) - 2.0 * STC
                                    )

                                    d = float(dB + w_coassoc * dC)

                                    if (
                                        tabu_forbidden(bi, g2, step)
                                        or tabu_forbidden(bj, g1, step)
                                    ) and (base_total + d) <= best_total:
                                        continue

                                    if best_swap is None or d > best_swap[-1]:
                                        best_swap = ("swap", bi, bj, g1, g2, d)

                    for g_from in g_order:
                        for bi in candidate_blocks(g_from, L_blocks):
                            bi = int(bi)
                            if not feas.can_remove(bi, g_from):
                                continue

                            dests = list(range(K_used))
                            rng.shuffle(dests)

                            for g_to in dests[: min(L_groups, K_used)]:
                                if g_to == g_from:
                                    continue
                                if feas.can_add(bi, g_to):
                                    continue
                                if not members_b[g_to]:
                                    continue

                                eject_cand = candidate_blocks(g_to, L_blocks)

                                for bj in eject_cand:
                                    bj = int(bj)
                                    if not feas.can_add_after_removal(bi, g_to, bj):
                                        continue

                                    gks = list(range(K_used))
                                    rng.shuffle(gks)

                                    for gk in gks:
                                        if gk == g_to:
                                            continue

                                        if gk == g_from:
                                            if not feas.can_add_after_removal(bj, g_from, bi):
                                                continue
                                        else:
                                            if not feas.can_add(bj, gk):
                                                continue

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

                                        if (
                                            tabu_forbidden(bi, g_to, step)
                                            or tabu_forbidden(bj, gk, step)
                                        ) and (base_total + d) <= best_total:
                                            continue

                                        if best_eject is None or d > best_eject[-1]:
                                            best_eject = ("eject", bi, g_from, g_to, bj, gk, d)

                    best_action = None
                    for cand in (best_move, best_swap, best_eject):
                        if cand is None:
                            continue
                        if best_action is None or cand[-1] > best_action[-1]:
                            best_action = cand

                    if best_action is None:
                        did_split = False
                        if allow_block_splitting:
                            for g in g_order:
                                multi = [int(bi) for bi in members_b[g] if len(blocks[int(bi)]) > 1]
                                if not multi:
                                    continue
                                multi.sort(key=lambda bi_: attachment_to_group(bi_, g))
                                if split_block_into_singletons_in_place(multi[0]):
                                    did_split = True
                                    break
                        if did_split:
                            continue
                        break

                    kind = best_action[0]

                    if kind == "move":
                        _, bi, g_from, g_to, _ = best_action
                        apply_move(int(bi), int(g_from), int(g_to))
                        set_tabu(int(bi), int(g_from), step, tenure)

                    elif kind == "swap":
                        _, bi, bj, g1, g2, _ = best_action
                        apply_swap(int(bi), int(bj), int(g1), int(g2))
                        set_tabu(int(bi), int(g1), step, tenure)
                        set_tabu(int(bj), int(g2), step, tenure)

                    else:
                        _, bi, g_from, g_to, bj, gk, _ = best_action
                        apply_ejection(int(bi), int(g_from), int(g_to), int(bj), int(gk))
                        set_tabu(int(bi), int(g_from), step, tenure)
                        set_tabu(int(bj), int(g_to), step, tenure)

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

            shake_moves = max(2, len(blocks) // 25)

            best_local_total = current_total()
            best_local_B = current_B_sum()
            best_local_comp2g = comp2g.copy()

            for sround in range(shake_rounds):
                max_steps = max(tabu_max_steps, int(local_iters) * 5)
                tenure = 7
                L_blocks = 10
                L_groups = min(K_used, 6)

                ttot, tB, comp2g_best = improve_with_tabu(
                    max_steps=max_steps,
                    tenure=tenure,
                    L_blocks=L_blocks,
                    L_groups=L_groups,
                )

                if tB > best_local_B + 1e-12:
                    best_local_B = float(tB)
                    best_local_total = float(ttot)
                    best_local_comp2g = comp2g_best.copy()

                if sround == shake_rounds - 1:
                    break

                g_order = list(range(K_used))
                g_order.sort(key=lambda g: cohesion(g))

                moved = 0
                for g_from in g_order:
                    cand = candidate_blocks(g_from, L=20)
                    for bi in cand:
                        bi = int(bi)
                        if moved >= shake_moves:
                            break
                        if not feas.can_remove(bi, g_from):
                            continue

                        dests = list(range(K_used))
                        rng.shuffle(dests)
                        for g_to in dests:
                            if g_to == g_from:
                                continue
                            if feas.can_add(bi, g_to):
                                apply_move(bi, g_from, g_to)
                                moved += 1
                                break
                    if moved >= shake_moves:
                        break

            comp2g_final = best_local_comp2g.copy()
            Q = _objective_B_from_comp_assignment(W_B, comp2g_final, K_used)
            if not np.isfinite(Q):
                continue

            gvec = build_gvec(comp2g_final)
            Z = partition_vector_to_2d_matrix(gvec)
            meta = {
                "r_min": int(r_min),
                "r_max": int(r_max),
                "K_used": int(K_used),
                "objective_B_sum": float(Q),
                "objective_total": float(best_local_total),
                "use_K_constraint": bool(use_K_constraint),
                "fingerprint_decimals": int(fingerprint_decimals),
                "allow_block_splitting": bool(allow_block_splitting),
                "seed": int(seed),
            }

            if best is None or meta["objective_B_sum"] > best[1]["objective_B_sum"]:
                best = (Z, meta)

    if best is None:
        if use_K_constraint:
            alt_Ks = tuple(sorted({k for k in range(max(2, int(K or 2)), max(3, int(K or 2) + 8), 2)}))
            if tuple(clustering_Ks) == alt_Ks:
                return None
            return modular_very_fortunate_descent(
                wz=wz,
                A=A,
                a=a,
                m=m,
                K=K,
                R=R,
                must_link=must_link,
                cannot_link=cannot_link,
                seed=seed,
                fingerprint_decimals=fingerprint_decimals,
                allow_block_splitting=allow_block_splitting,
                max_K_increase=max_K_increase,
                use_K_constraint=use_K_constraint,
                candidate_Ks=candidate_Ks,
                restarts=restarts,
                local_iters=local_iters,
                w_coassoc=w_coassoc,
                clustering_Ks=alt_Ks,
                clustering_seeds=clustering_seeds,
                clustering_methods=clustering_methods,
                wz_is_C_node=wz_is_C_node,
                tabu_max_steps=tabu_max_steps,
                shake_rounds=shake_rounds,
                orbit_fallback=orbit_fallback,
            )
        return None

    return best


# Usage:
# Z, meta = modular_very_fortunate_descent(
# wz=A,
# A=A,
# a=a,
# m=m,
# K=5,              # optional search hint here
# R=2,              # ignored here
# must_link=[],#unworthy_edges,
# cannot_link=[],
# use_K_constraint=True,
# # candidate_Ks=[4, 5, 6, 7, 8, 9, 10],
# shake_rounds=2
# )