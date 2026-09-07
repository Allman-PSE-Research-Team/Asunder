"""Modular VFD refinement with pairwise, balance, and extensible constraints."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.sparse.csgraph import connected_components

from asunder.base.algorithms.vfd_constraints import (
    VFDAssignmentView,
    VFDComponentMove,
    VFDConstraint,
    VFDConstraintContext,
    VFDTransition,
)
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
    if M.ndim != 2 or M.shape[0] != M.shape[1]:
        raise ValueError("The input matrix must be square.")
    if not np.all(np.isfinite(M)):
        raise ValueError("The input matrix must contain only finite values.")
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
    node_weights: Optional[Sequence[int]] = None,
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
    if N < 0:
        raise ValueError("N must be nonnegative.")
    if node_weights is None:
        weights = np.ones(N, dtype=int)
    else:
        raw_weights = np.asarray(node_weights)
        if raw_weights.shape != (N,):
            raise ValueError("node_weights must contain one value per node.")
        if not np.all(np.isfinite(raw_weights)):
            raise ValueError("node_weights must contain only finite values.")
        if np.any(raw_weights <= 0) or not np.all(raw_weights == np.rint(raw_weights)):
            raise ValueError("node_weights must contain positive integers.")
        weights = np.rint(raw_weights).astype(int)

    for relation_name, pairs, reject_self in (
        ("must-link", must_link, False),
        ("cannot-link", cannot_link, True),
    ):
        for pair in pairs:
            if len(pair) != 2:
                raise ValueError(f"Each {relation_name} entry must contain two nodes.")
            source, target = int(pair[0]), int(pair[1])
            if not (0 <= source < N and 0 <= target < N):
                raise ValueError(
                    f"{relation_name} pair {(source, target)} contains a node "
                    f"outside 0..{N - 1}."
                )
            if reject_self and source == target:
                return None

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
    cweight = np.array([weights[c].sum() for c in comps], dtype=int)

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
            "cweight": cweight,
            "use_bitmask": True,
            "forb_mask": forb_mask,
            "comp_bit": comp_bit,
        }

    return {
        "C": C, "cid": cid, "comps": comps, "csz": csz,
        "cweight": cweight,
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
    n_unique = np.unique(X, axis=0).shape[0]

    # For spectral with precomputed affinity, ensure diagonal is 1 and nonnegative.
    A = X.copy()
    np.fill_diagonal(A, 1.0)
    A[A < 0.0] = 0.0

    C_cnt = np.zeros((n, n), dtype=np.float64)
    total = 0

    for k in n_components_list:
        k = min(k, n_unique)
        if k <= 1 or k > n:
            continue
        for sd in seeds:
            for meth in methods:
                try:
                    if meth == "kmeans":
                        labels = KMeans(n_clusters=k, n_init=10, random_state=int(sd)).fit_predict(X)
                    elif meth == "gmm":
                        labels = GaussianMixture(n_components=k, random_state=int(sd)).fit(X).predict(X)
                    elif meth == "spectral":
                        n_components, _ = connected_components(A, directed=False)
                        if n_components > 1:
                            continue
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


def _component_sum_matrix(M: np.ndarray, comp: Dict[str, Any]) -> np.ndarray:
    """Aggregate a node-level square matrix by summing component blocks."""
    matrix = np.asarray(M, dtype=float)
    node_count = matrix.shape[0]
    if matrix.shape != (node_count, node_count):
        raise ValueError("M must be square.")

    component_count = int(comp["C"])
    if component_count == 0:
        return np.zeros((0, 0), dtype=float)

    membership = np.zeros((node_count, component_count), dtype=float)
    membership[np.arange(node_count), np.asarray(comp["cid"], dtype=int)] = 1.0
    return membership.T @ matrix @ membership


def _component_sum_matrix_B(
    A: np.ndarray,
    a: np.ndarray,
    m: float,
    comp: Dict[str, Any],
    gamma: float = 1.0,
) -> np.ndarray:
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
    B = A - float(gamma) * np.outer(a, a) / float(m)

    cid = np.asarray(comp["cid"], dtype=int)
    C = int(comp["C"])
    if C == 0:
        return np.zeros((0, 0), dtype=float)

    P = np.zeros((N, C), dtype=float)
    P[np.arange(N), cid] = 1.0
    return P.T @ B @ P  # sum between components


def _node_partition_B_sum(
    A: np.ndarray,
    a: np.ndarray,
    m: float,
    z: np.ndarray,
    gamma: float = 1.0,
) -> float:
    """
    Score a node-level partition-like matrix on the unnormalized B scale.

    This matches ``objective_B_sum`` from component assignments, where
    ``B = A - aa^T / m``.
    """
    A = np.asarray(A, dtype=float)
    a = np.asarray(a, dtype=float).reshape(-1)
    z = np.asarray(z, dtype=float)
    N = A.shape[0]
    if A.shape != (N, N) or z.shape != (N, N) or a.shape[0] != N:
        raise ValueError("A, z must be (N,N) and a must be (N,).")
    if m == 0:
        raise ValueError("m must be nonzero.")
    B = A - float(gamma) * np.outer(a, a) / float(m)
    return float(np.sum(B * z))


def _fingerprint_blocks_from_rounded_rows(
    C_comp: np.ndarray,
    comp: Dict[str, Any],
    fingerprint_decimals: int,
    r_max: Any,
    component_weights: Optional[np.ndarray] = None,
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
    csz = np.asarray(
        comp["csz"] if component_weights is None else component_weights,
        dtype=int,
    )

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
    R_bounds: Optional[Tuple[int, int]] = None,
) -> Tuple[int, int, List[int]]:
    """
    Resolve size bounds and the list of fixed-K subproblems to evaluate.

    Parameters
    ----------
    N : int
        Number of original nodes.
    Cn : int
        Number of must-link components.
    K : int or None, default=2
        Baseline number of communities.
    R : int or None, default=1
        Width of the allowed cluster-size range. Also corresponds to the load balance tightness (smaller R implies tighter load balance).
        For a selected cluster count, the lower and upper bounds are computed from the corresponding
        balanced range rule. Used when the K-constraint is active.
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
        if K is None:
            raise ValueError("K must be provided when use_K_constraint=True.")
        if R_bounds is None:
            if R is None:
                raise ValueError(
                    "R or R_bounds must be provided when use_K_constraint=True."
                )
            r_min, r_max = _range_bounds_from_KR(N, K, R)
        else:
            if len(R_bounds) != 2:
                raise ValueError("R_bounds must contain exactly two values.")
            r_min, r_max = int(R_bounds[0]), int(R_bounds[1])
            if r_min < 1 or r_min > r_max:
                raise ValueError("R_bounds must satisfy 1 <= r_min <= r_max.")
        k_lo, k_hi = _feasible_K_range(N, r_min, r_max)
        if k_lo > k_hi:
            return int(r_min), int(r_max), []

        K0 = max(int(K), int(k_lo))
        K_end = min(int(k_hi), int(K) + int(max_K_increase))

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
    K: Optional[int] = 2,
    R: Optional[int] = 1,
    must_link: Sequence[Tuple[int, int]] = (),
    cannot_link: Sequence[Tuple[int, int]] = (),
    seed: Optional[int] = 42,
    fingerprint_decimals: int = 6,
    allow_block_splitting: bool = True,
    max_K_increase: int = 0,
    use_K_constraint: bool = False,
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
    R_bounds: Optional[Tuple[int, int]] = None,
    balance_weights: Optional[Sequence[int]] = None,
    gamma: float = 1.0,
    constraints: Sequence[VFDConstraint] = (),
    component_members: Optional[Sequence[Sequence[Any]]] = None,
    constraint_repair_steps: Optional[int] = None,
) -> Optional[Tuple[np.ndarray, Dict[str, Any]]]:
    """
    Modular function for building a feasible decomposition column from co-association structure and local search.

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
        Width of the allowed cluster-size range. Also corresponds to the load balance tightness (smaller R implies tighter load balance).
        For a selected cluster count, the lower and upper bounds are computed from the corresponding
        balanced range rule. Used when ``use_K_constraint=True``.
    must_link : sequence of tuple of int
        Must-link pairs.
    cannot_link : sequence of tuple of int
        Cannot-link pairs.
    R_bounds : tuple of int or None, default=None
        Explicit inclusive lower and upper community-weight bounds. When
        supplied, these replace the bounds derived from ``K`` and ``R``.
    balance_weights : sequence of int or None, default=None
        Positive integer node weights used by balance constraints. Unit
        weights are used by default.
    gamma : float, default=1.0
        Modularity resolution used by the refinement objective.
    constraints : sequence of VFDConstraint, default=()
        Additional component-local, community-wide, or partition-wide hard
        constraints. Constraint specifications are prepared after must-link
        contraction and rebound for every K candidate and restart.
    component_members : sequence of sequences, optional
        Original member identifiers represented by each row of ``A``. This is
        useful when ``A`` was contracted before ModularVFD was called.
    constraint_repair_steps : int or None, default=None
        Maximum guided feasibility-repair steps for partition-wide
        constraints. ``None`` derives a budget from ``local_iters``.
    seed : int or None, default=42
        Random seed.
    fingerprint_decimals : int, default=6
        Decimal rounding used to form fingerprint blocks.
    allow_block_splitting : bool, default=True
        If True, allow refinement of coarse fingerprint blocks.
    max_K_increase : int, default=0
        Maximum increase above the baseline K when the K-constraint is active.
    use_K_constraint : bool, default=False
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
    if int(restarts) < 1:
        raise ValueError("restarts must be at least 1.")
    if int(max_K_increase) < 0:
        raise ValueError("max_K_increase must be nonnegative.")
    if int(local_iters) < 0 or int(tabu_max_steps) < 0 or int(shake_rounds) < 0:
        raise ValueError(
            "local_iters, tabu_max_steps, and shake_rounds must be nonnegative."
        )
    if int(fingerprint_decimals) < 0:
        raise ValueError("fingerprint_decimals must be nonnegative.")
    if constraint_repair_steps is not None and int(constraint_repair_steps) < 0:
        raise ValueError("constraint_repair_steps must be nonnegative or None.")
    gamma = float(gamma)
    if not np.isfinite(gamma) or gamma < 0:
        raise ValueError("gamma must be a finite nonnegative value.")
    unknown_methods = set(clustering_methods or ()) - {"kmeans", "gmm", "spectral"}
    if unknown_methods:
        raise ValueError(f"Unknown clustering methods: {sorted(unknown_methods)}.")

    rng = np.random.default_rng(seed)

    wz = np.asarray(wz, dtype=float)
    if wz.ndim != 2 or wz.shape[0] != wz.shape[1]:
        raise ValueError("wz must be a square matrix.")
    N = int(wz.shape[0])

    A = np.asarray(A, dtype=float)
    a = np.asarray(a, dtype=float).reshape(-1)
    if A.shape != (N, N) or a.shape != (N,):
        raise ValueError("wz and A must be (N,N), and a must be (N,).")
    if not np.all(np.isfinite(A)) or not np.all(np.isfinite(a)):
        raise ValueError("A and a must contain only finite values.")
    try:
        m = float(m)
    except (TypeError, ValueError):
        raise ValueError("m must be finite and positive for a nonempty graph.") from None
    if not np.isfinite(m) or m < 0 or (N > 0 and m <= 0):
        raise ValueError("m must be finite and positive for a nonempty graph.")

    constraint_specs = () if constraints is None else tuple(constraints)
    for constraint in constraint_specs:
        if not callable(getattr(constraint, "prepare", None)):
            raise TypeError("Each constraint must implement prepare(context).")

    if component_members is None:
        input_component_members = tuple((int(i),) for i in range(N))
    else:
        if len(component_members) != N:
            raise ValueError("component_members must contain one collection per row of A.")
        normalized_members = []
        for row, members in enumerate(component_members):
            try:
                member_tuple = tuple(members)
            except TypeError as exc:
                raise TypeError(
                    f"component_members[{row}] must be an iterable of member identifiers."
                ) from exc
            if not member_tuple:
                raise ValueError("component_members entries must not be empty.")
            normalized_members.append(member_tuple)
        input_component_members = tuple(normalized_members)
        flattened_members = [
            member for members in input_component_members for member in members
        ]
        try:
            distinct_member_count = len(set(flattened_members))
        except TypeError as exc:
            raise TypeError(
                "component_members must contain hashable member identifiers."
            ) from exc
        if distinct_member_count != len(flattened_members):
            raise ValueError("component_members must contain distinct member identifiers.")

    if N == 0:
        constraint_names = []
        if constraint_specs:
            empty_context = VFDConstraintContext(
                input_adjacency=A,
                component_adjacency=np.zeros((0, 0), dtype=float),
                node_to_component=(),
                component_members=(),
                component_weights=(),
                K=0,
                r_min=0,
                r_max=0,
            )
            empty_assignment = VFDAssignmentView(empty_context, ())
            for specification in constraint_specs:
                prepared = specification.prepare(empty_context)
                if not callable(getattr(prepared, "bind", None)):
                    raise TypeError(
                        "Constraint prepare(context) must return an object with bind()."
                    )
                runtime = prepared.bind()
                evaluation = runtime.evaluate_final(empty_assignment)
                if not hasattr(evaluation, "satisfied"):
                    raise TypeError(
                        "Constraint evaluations must be "
                        "VFDConstraintEvaluation instances."
                    )
                if not evaluation.satisfied:
                    return None
                constraint_names.append(
                    str(getattr(runtime, "name", type(runtime).__name__))
                )
        g0 = np.zeros(0, dtype=int)
        return partition_vector_to_2d_matrix(g0), {
            "r_min": 0,
            "r_max": 0,
            "K_used": 0,
            "use_K_constraint": bool(use_K_constraint),
            "constraints": tuple(constraint_names),
            "constraint_repair_steps": 0,
        }

    sym = _symmetrize_unitdiag(wz)
    reference_Q = _node_partition_B_sum(A, a, m, sym, gamma=gamma)

    must_link = [_normalize_pair(i, j) for (i, j) in (must_link or [])]
    cannot_link = [_normalize_pair(i, j) for (i, j) in (cannot_link or [])]

    comp = _build_components(
        N,
        must_link,
        cannot_link,
        node_weights=balance_weights,
    )
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

    cweight = np.asarray(comp["cweight"], dtype=int)
    total_balance_weight = int(cweight.sum())
    use_bitmask = bool(comp["use_bitmask"])
    forb = comp["forb_mask"]
    comp_bit = comp["comp_bit"]
    comps = comp["comps"]
    contracted_component_members = tuple(
        tuple(
            member
            for input_node in component
            for member in input_component_members[int(input_node)]
        )
        for component in comps
    )

    auto_clustering_ks = clustering_Ks is None
    if clustering_Ks is None:
        if use_K_constraint:
            # K/R bounds are resolved below; this value is unused by that path.
            clustering_Ks = ()
        else:
            rough = max(2, int(round(np.sqrt(max(2, N)))))
            clustering_Ks = tuple(sorted({1, 2, min(N, rough), min(N, rough + 2)}))

    r_min, r_max, K_candidates = _resolve_k_control(
        N=total_balance_weight,
        Cn=Cn,
        K=K,
        R=R,
        use_K_constraint=use_K_constraint,
        max_K_increase=max_K_increase,
        clustering_Ks=clustering_Ks,
        candidate_Ks=candidate_Ks,
        R_bounds=R_bounds,
    )

    if not K_candidates:
        return None

    if auto_clustering_ks and use_K_constraint:
        k_lo, k_hi = _feasible_K_range(total_balance_weight, r_min, r_max)
        requested_k = int(K or k_lo)
        clustering_Ks = tuple(
            sorted(
                {
                    int(candidate)
                    for candidate in (
                        max(2, k_lo),
                        min(k_hi, requested_k),
                        min(k_hi, requested_k + 2),
                    )
                    if 1 <= int(candidate) <= min(N, Cn)
                }
            )
        )
        if not clustering_Ks:
            clustering_Ks = (min(N, Cn),)

    if int(cweight.max()) > r_max:
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
    component_adjacency = (
        _component_sum_matrix(A, comp) if constraint_specs else None
    )
    W_B = _component_sum_matrix_B(A, a, m, comp, gamma=gamma)

    def build_gvec(comp2g: np.ndarray) -> np.ndarray:
        g = np.empty(N, dtype=int)
        for c in range(Cn):
            g[comps[c]] = int(comp2g[c])
        return g

    def block_internal_sum(M: np.ndarray, b: List[int]) -> float:
        idx = np.asarray(b, dtype=int)
        return float(M[np.ix_(idx, idx)].sum())

    best_improving = None
    best_feasible = None

    for K_used in K_candidates:
        K_used = int(K_used)
        if K_used <= 0:
            continue
        if not (
            K_used * r_min <= total_balance_weight <= K_used * r_max
        ):
            continue
        if K_used > Cn:
            continue

        target = (
            _target_sizes_from_bounds(
                total_balance_weight,
                K_used,
                r_min,
                r_max,
            )
            if use_K_constraint
            else None
        )

        constraint_context = None
        prepared_constraints = []
        if constraint_specs:
            constraint_context = VFDConstraintContext(
                input_adjacency=A,
                component_adjacency=component_adjacency,
                node_to_component=np.asarray(comp["cid"], dtype=int),
                component_members=contracted_component_members,
                component_weights=cweight,
                K=K_used,
                r_min=r_min if use_K_constraint else None,
                r_max=r_max if use_K_constraint else None,
            )
            for specification in constraint_specs:
                prepared = specification.prepare(constraint_context)
                if not callable(getattr(prepared, "bind", None)):
                    raise TypeError(
                        "Constraint prepare(context) must return an object with bind()."
                    )
                prepared_constraints.append(prepared)
        prepared_constraints = tuple(prepared_constraints)

        base_blocks = _fingerprint_blocks_from_rounded_rows(
            C_comp,
            comp,
            fingerprint_decimals=fingerprint_decimals,
            r_max=r_max,
            component_weights=cweight,
        )
        base_blocks = _make_blocks_conflict_free(
            base_blocks,
            use_bitmask=use_bitmask,
            forb=forb,
            comp_bit=comp_bit,
        )
        base_blocks = _ensure_at_least_K_blocks(base_blocks, K_used, cweight)

        restart_limit = int(restarts)
        restart_index = 0
        while restart_index < restart_limit:
            restart_index += 1
            bound_constraints = []
            fast_local_runtime_ids = set()
            for prepared in prepared_constraints:
                runtime = prepared.bind()
                scope = getattr(runtime, "scope", None)
                if scope not in {"local", "community", "partition"}:
                    raise ValueError(
                        "A bound VFD constraint scope must be 'local', 'community', "
                        "or 'partition'."
                    )
                common_methods = (
                    "block_is_feasible",
                    "evaluate_partial",
                    "evaluate_final",
                )
                if any(
                    not callable(getattr(runtime, method, None))
                    for method in common_methods
                ):
                    raise TypeError(
                        "A bound VFD constraint does not implement the required runtime API."
                    )
                fast_local = scope == "local" and all(
                    callable(getattr(runtime, method, None))
                    for method in (
                        "evaluate_component_transition",
                        "component_transition_applied",
                        "component_proposals",
                    )
                )
                generic_transition = all(
                    callable(getattr(runtime, method, None))
                    for method in (
                        "evaluate_transition",
                        "transition_applied",
                        "repair_proposals",
                    )
                )
                if not fast_local and not generic_transition:
                    raise TypeError(
                        "A bound local constraint must implement the component-only "
                        "transition hooks; other constraints must implement the "
                        "assignment-view transition hooks."
                    )
                if fast_local:
                    fast_local_runtime_ids.add(id(runtime))
                bound_constraints.append(runtime)
            bound_constraints = tuple(bound_constraints)

            def uses_fast_local_api(runtime: Any) -> bool:
                return id(runtime) in fast_local_runtime_ids

            blocks = []
            incompatible_block = False
            for base_block in base_blocks:
                block = tuple(int(component) for component in base_block)
                if all(runtime.block_is_feasible(block) for runtime in bound_constraints):
                    blocks.append(list(block))
                    continue
                if not allow_block_splitting or len(block) <= 1:
                    incompatible_block = True
                    break
                for component in block:
                    singleton = (int(component),)
                    if not all(
                        runtime.block_is_feasible(singleton)
                        for runtime in bound_constraints
                    ):
                        incompatible_block = True
                        break
                    blocks.append([int(component)])
                if incompatible_block:
                    break
            if incompatible_block:
                continue

            Bn = len(blocks)
            if Bn < K_used:
                continue

            b_size = [int(sum(int(cweight[c]) for c in b)) for b in blocks]
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

            community_ids = tuple(range(K_used))

            def assignment_view(labels: Optional[Sequence[int]] = None) -> VFDAssignmentView:
                if constraint_context is None:
                    raise RuntimeError("Constraint views require a prepared constraint context.")
                values = comp2g if labels is None else labels
                return VFDAssignmentView(
                    constraint_context,
                    values,
                    community_ids=community_ids,
                )

            def evaluate_constraint_transition(
                transition: VFDTransition,
                *,
                require_satisfied: bool,
            ) -> Tuple[
                bool,
                Tuple[Tuple[float, ...], ...],
                Optional[VFDAssignmentView],
            ]:
                if not bound_constraints:
                    return True, (), None
                before = None
                after = None
                violation = []
                for runtime in bound_constraints:
                    if uses_fast_local_api(runtime):
                        evaluation = runtime.evaluate_component_transition(transition)
                    else:
                        if before is None:
                            before = assignment_view()
                            after = before.after(transition)
                        evaluation = runtime.evaluate_transition(
                            before,
                            transition,
                            after,
                        )
                    if not hasattr(evaluation, "satisfied") or not hasattr(
                        evaluation, "extendable"
                    ):
                        raise TypeError(
                            "Constraint evaluations must be VFDConstraintEvaluation instances."
                        )
                    if not evaluation.satisfied and not evaluation.extendable:
                        return False, (), after
                    if runtime.scope == "local" and not evaluation.satisfied:
                        return False, (), after
                    if require_satisfied and not evaluation.satisfied:
                        return False, (), after
                    violation.append(
                        tuple(float(value) for value in evaluation.violation)
                    )
                return True, tuple(violation), after

            def evaluate_final_constraints(
                labels: Optional[Sequence[int]] = None,
            ) -> Tuple[
                bool,
                Tuple[Tuple[float, ...], ...],
                Optional[VFDAssignmentView],
            ]:
                if not bound_constraints:
                    return True, (), None
                view = assignment_view(labels)
                violation = []
                feasible = True
                for runtime in bound_constraints:
                    evaluation = runtime.evaluate_final(view)
                    feasible = feasible and evaluation.satisfied
                    violation.append(
                        tuple(float(value) for value in evaluation.violation)
                    )
                return feasible, tuple(violation), view

            def independently_validate_final_constraints(
                labels: Sequence[int],
            ) -> bool:
                """Validate a saved assignment with fresh, unmodified runtimes."""
                if not prepared_constraints:
                    return True
                view = assignment_view(labels)
                for prepared in prepared_constraints:
                    runtime = prepared.bind()
                    evaluation = runtime.evaluate_final(view)
                    if not hasattr(evaluation, "satisfied"):
                        raise TypeError(
                            "Constraint evaluations must be "
                            "VFDConstraintEvaluation instances."
                        )
                    if not evaluation.satisfied:
                        return False
                return True

            def notify_constraint_transition(transition: VFDTransition) -> None:
                if not bound_constraints:
                    return
                after = None
                for runtime in bound_constraints:
                    if uses_fast_local_api(runtime):
                        runtime.component_transition_applied(transition)
                    else:
                        if after is None:
                            after = assignment_view()
                        runtime.transition_applied(transition, after)

            def block_transition(
                bi: int,
                source: Optional[int],
                target_group: Optional[int],
                *,
                phase: str,
                label: str,
            ) -> VFDTransition:
                return VFDTransition.move(
                    tuple(int(component) for component in blocks[int(bi)]),
                    source,
                    target_group,
                    phase=phase,
                    label=label,
                )

            initial_is_extendable = True
            if bound_constraints:
                initial_view = assignment_view()
                for runtime in bound_constraints:
                    evaluation = runtime.evaluate_partial(initial_view)
                    if not evaluation.satisfied and (
                        runtime.scope == "local" or not evaluation.extendable
                    ):
                        initial_is_extendable = False
                        break
            if not initial_is_extendable:
                continue

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
                    if int(b_size[bi]) > r_max or not feas.can_add(int(bi), int(gi)):
                        continue
                    transition = block_transition(
                        int(bi),
                        None,
                        int(gi),
                        phase="construction",
                        label="seed",
                    )
                    allowed, _, _ = evaluate_constraint_transition(
                        transition,
                        require_satisfied=False,
                    )
                    if not allowed:
                        continue
                    add_block_to_group(int(bi), int(gi))
                    notify_constraint_transition(transition)
                    used_blocks.add(int(bi))
                    gi += 1
                return gi == K_used, used_blocks

            order = list(np.argsort(np.asarray(b_size))[::-1])
            ok_seed, seeded = ensure_nonempty_seeding(order)
            if not ok_seed:
                if allow_block_splitting:
                    rejected_multi = next(
                        (
                            int(bi)
                            for bi in order
                            if int(bi) not in seeded and len(blocks[int(bi)]) > 1
                        ),
                        None,
                    )
                    if rejected_multi is not None:
                        rejected_components = list(blocks[rejected_multi])
                        blocks[rejected_multi] = [rejected_components[0]]
                        blocks.extend([component] for component in rejected_components[1:])
                        base_blocks = [list(block) for block in blocks]
                        restart_limit += 1
                continue

            for bi in order:
                bi = int(bi)
                if bi in seeded:
                    continue

                F = []
                rem_nodes = int(total_balance_weight - gsz.sum())
                deficit_sum = int(np.maximum(0, r_min - gsz).sum())

                for g in range(K_used):
                    if not feas.can_add(bi, g):
                        continue
                    old_def = max(0, r_min - int(gsz[g]))
                    new_def = max(0, r_min - int(gsz[g] + b_size[bi]))
                    def2 = deficit_sum - old_def + new_def
                    if def2 > (rem_nodes - int(b_size[bi])):
                        continue
                    transition = block_transition(
                        bi,
                        None,
                        int(g),
                        phase="construction",
                        label="construct",
                    )
                    allowed, _, _ = evaluate_constraint_transition(
                        transition,
                        require_satisfied=False,
                    )
                    if allowed:
                        F.append(g)

                if not F:
                    if allow_block_splitting and len(blocks[bi]) > 1:
                        comps_b = blocks[bi]
                        blocks[bi] = [comps_b[0]]
                        for c in comps_b[1:]:
                            blocks.append([c])
                        base_blocks = [list(block) for block in blocks]
                        restart_limit += 1
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
                transition = block_transition(
                    bi,
                    None,
                    g_best,
                    phase="construction",
                    label="construct",
                )
                add_block_to_group(bi, g_best)
                notify_constraint_transition(transition)

            if not ok_seed:
                continue

            def split_block_for_restart(bi: int) -> bool:
                nonlocal base_blocks, restart_limit
                if not allow_block_splitting or len(blocks[bi]) <= 1:
                    return False
                components = list(blocks[bi])
                blocks[bi] = [components[0]]
                blocks.extend([component] for component in components[1:])
                base_blocks = [list(block) for block in blocks]
                restart_limit += 1
                return True

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
                    split_candidate = None

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
                                if (
                                    split_candidate is None
                                    and allow_block_splitting
                                    and len(blocks[bi_]) > 1
                                ):
                                    split_candidate = bi_
                                continue
                            if not feas.can_add(bi_, g_need):
                                if (
                                    split_candidate is None
                                    and allow_block_splitting
                                    and len(blocks[bi_]) > 1
                                ):
                                    split_candidate = bi_
                                continue

                            transition = block_transition(
                                bi_,
                                int(g_from),
                                int(g_need),
                                phase="construction",
                                label="balance-repair",
                            )
                            allowed, _, _ = evaluate_constraint_transition(
                                transition,
                                require_satisfied=False,
                            )
                            if not allowed:
                                continue

                            remove_block_from_group(bi_, g_from)
                            add_block_to_group(bi_, g_need)
                            notify_constraint_transition(transition)
                            moved = True
                            break

                        if moved:
                            break

                    if not moved:
                        if split_candidate is not None:
                            split_block_for_restart(split_candidate)
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

            def apply_move(
                bi: int,
                g_from: int,
                g_to: int,
                transition: Optional[VFDTransition] = None,
            ) -> None:
                if transition is None:
                    transition = block_transition(
                        bi,
                        g_from,
                        g_to,
                        phase="feasible_search",
                        label="move",
                    )
                remove_block_from_group(bi, g_from)
                add_block_to_group(bi, g_to)
                notify_constraint_transition(transition)

            def apply_swap(
                bi: int,
                bj: int,
                g1: int,
                g2: int,
                transition: Optional[VFDTransition] = None,
            ) -> None:
                if transition is None:
                    transition = VFDTransition.swap(
                        tuple(int(component) for component in blocks[bi]),
                        g1,
                        tuple(int(component) for component in blocks[bj]),
                        g2,
                        phase="feasible_search",
                        label="swap",
                    )
                remove_block_from_group(bi, g1)
                remove_block_from_group(bj, g2)
                add_block_to_group(bi, g2)
                add_block_to_group(bj, g1)
                notify_constraint_transition(transition)

            def apply_ejection(
                bi: int,
                g_from: int,
                g_to: int,
                bj: int,
                g_k: int,
                transition: Optional[VFDTransition] = None,
            ) -> None:
                if transition is None:
                    transition = VFDTransition(
                        (
                            VFDComponentMove(tuple(blocks[bj]), g_to, g_k),
                            VFDComponentMove(tuple(blocks[bi]), g_from, g_to),
                        ),
                        phase="feasible_search",
                        label="ejection",
                    )
                remove_block_from_group(bj, g_to)
                add_block_to_group(bj, g_k)
                remove_block_from_group(bi, g_from)
                add_block_to_group(bi, g_to)
                notify_constraint_transition(transition)

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

                b_size[bi] = int(cweight[c0])
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

                    b_size.append(int(cweight[c]))
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

            def structural_assignment_is_feasible(labels: Sequence[int]) -> bool:
                labels = np.asarray(labels, dtype=int)
                if labels.shape != (Cn,) or np.any(labels < 0) or np.any(labels >= K_used):
                    return False

                loads = np.bincount(
                    labels,
                    weights=cweight,
                    minlength=K_used,
                )
                component_counts = np.bincount(labels, minlength=K_used)
                if np.any(component_counts == 0):
                    return False
                if np.any(loads < r_min) or np.any(loads > r_max):
                    return False

                if use_bitmask:
                    group_masks = [0] * K_used
                    for component, group in enumerate(labels):
                        if int(forb[component]) & int(group_masks[int(group)]):
                            return False
                        group_masks[int(group)] |= int(comp_bit[component])
                else:
                    group_components = [set() for _ in range(K_used)]
                    for component, group in enumerate(labels):
                        if set(forb[component]) & group_components[int(group)]:
                            return False
                        group_components[int(group)].add(int(component))
                return True

            def structural_transition_is_feasible(
                transition: VFDTransition,
            ) -> Tuple[bool, Optional[VFDAssignmentView]]:
                try:
                    after = assignment_view().after(transition)
                except ValueError:
                    return False, None
                return (
                    structural_assignment_is_feasible(
                        after.component_to_community
                    ),
                    after,
                )

            def component_transition_is_applicable(
                transition: VFDTransition,
            ) -> bool:
                """Check whether current fingerprint blocks can realize a transition."""
                targets = {
                    int(component): move.target
                    for move in transition.moves
                    for component in move.components
                }
                for block in blocks:
                    moved = [component for component in block if component in targets]
                    if not moved:
                        continue
                    destinations = {targets[component] for component in moved}
                    requires_split = (
                        len(moved) != len(block) or len(destinations) != 1
                    )
                    if requires_split and (
                        not allow_block_splitting or len(block) <= 1
                    ):
                        return False
                    if None in destinations:
                        return False
                return True

            def repair_transition_evaluation(
                transition: VFDTransition,
            ) -> Tuple[
                bool,
                Tuple[Tuple[float, ...], ...],
                Optional[VFDAssignmentView],
            ]:
                if not component_transition_is_applicable(transition):
                    return False, (), None
                structural_ok, after = structural_transition_is_feasible(transition)
                if not structural_ok or after is None:
                    return False, (), after
                if not bound_constraints:
                    return True, (), after

                before = assignment_view()
                violation = []
                for runtime in bound_constraints:
                    if uses_fast_local_api(runtime):
                        transition_evaluation = (
                            runtime.evaluate_component_transition(transition)
                        )
                    else:
                        transition_evaluation = runtime.evaluate_transition(
                            before,
                            transition,
                            after,
                        )
                    if (
                        runtime.scope == "local"
                        and not transition_evaluation.satisfied
                    ):
                        return False, (), after
                    final_evaluation = runtime.evaluate_final(after)
                    if runtime.scope == "local" and not final_evaluation.satisfied:
                        return False, (), after
                    violation.append(
                        tuple(float(value) for value in final_evaluation.violation)
                    )
                return True, tuple(violation), after

            def normalize_repair_transition(transition: VFDTransition) -> VFDTransition:
                if not isinstance(transition, VFDTransition):
                    raise TypeError("Constraint repair proposals must be VFDTransition instances.")
                return VFDTransition(
                    transition.moves,
                    phase="repair",
                    label=transition.label or "constraint-repair",
                )

            def transition_key(transition: VFDTransition) -> Tuple[Any, ...]:
                return tuple(
                    (
                        tuple(int(component) for component in move.components),
                        move.source,
                        move.target,
                    )
                    for move in transition.moves
                )

            def ordinary_repair_transitions() -> List[VFDTransition]:
                proposals = []
                per_group_limit = min(8, max(1, len(blocks)))

                for source in range(K_used):
                    source_blocks = candidate_blocks(source, per_group_limit)
                    for bi in source_blocks:
                        for target_group in range(K_used):
                            if target_group == source:
                                continue
                            proposals.append(
                                block_transition(
                                    int(bi),
                                    source,
                                    target_group,
                                    phase="repair",
                                    label="constraint-move",
                                )
                            )

                for left_group in range(K_used):
                    left_blocks = candidate_blocks(left_group, per_group_limit)
                    for right_group in range(left_group + 1, K_used):
                        right_blocks = candidate_blocks(right_group, per_group_limit)
                        for left_block in left_blocks:
                            for right_block in right_blocks:
                                proposals.append(
                                    VFDTransition.swap(
                                        tuple(blocks[int(left_block)]),
                                        left_group,
                                        tuple(blocks[int(right_block)]),
                                        right_group,
                                        phase="repair",
                                        label="constraint-swap",
                                    )
                                )

                ejection_limit = min(4, per_group_limit)
                for source in range(K_used):
                    for bi in candidate_blocks(source, ejection_limit):
                        for middle in range(K_used):
                            if middle == source:
                                continue
                            for bj in candidate_blocks(middle, ejection_limit):
                                for target_group in range(K_used):
                                    if target_group == middle:
                                        continue
                                    proposals.append(
                                        VFDTransition(
                                            (
                                                VFDComponentMove(
                                                    tuple(blocks[int(bj)]),
                                                    middle,
                                                    target_group,
                                                ),
                                                VFDComponentMove(
                                                    tuple(blocks[int(bi)]),
                                                    source,
                                                    middle,
                                                ),
                                            ),
                                            phase="repair",
                                            label="constraint-ejection",
                                        )
                                    )
                return proposals

            def apply_atomic_component_transition(transition: VFDTransition) -> bool:
                target_by_component = {
                    int(component): move.target
                    for move in transition.moves
                    for component in move.components
                }

                for bi in list(range(len(blocks))):
                    moved_components = [
                        int(component)
                        for component in blocks[bi]
                        if int(component) in target_by_component
                    ]
                    if not moved_components:
                        continue
                    targets = {target_by_component[component] for component in moved_components}
                    if len(moved_components) != len(blocks[bi]) or len(targets) != 1:
                        if not split_block_into_singletons_in_place(bi):
                            return False

                moving_blocks = []
                for bi, block in enumerate(blocks):
                    moved = [int(component) for component in block if int(component) in target_by_component]
                    if not moved:
                        continue
                    targets = {target_by_component[component] for component in moved}
                    if len(moved) != len(block) or len(targets) != 1 or None in targets:
                        return False
                    moving_blocks.append(
                        (int(bi), int(block2g[bi]), int(next(iter(targets))))
                    )

                for bi, source, _ in moving_blocks:
                    remove_block_from_group(bi, source)
                for bi, _, target_group in moving_blocks:
                    add_block_to_group(bi, target_group)
                notify_constraint_transition(transition)
                return True

            repair_steps_used = 0
            constraints_feasible, current_violation, current_constraint_view = (
                evaluate_final_constraints()
            )
            if bound_constraints and not constraints_feasible:
                repair_budget = (
                    max(200, int(local_iters) * 5)
                    if constraint_repair_steps is None
                    else int(constraint_repair_steps)
                )
                seen_assignments = {tuple(int(group) for group in comp2g)}

                for repair_step in range(repair_budget):
                    repair_steps_used = repair_step + 1
                    proposed = ordinary_repair_transitions()
                    if current_constraint_view is None:
                        current_constraint_view = assignment_view()
                    for runtime in bound_constraints:
                        runtime_proposals = (
                            runtime.component_proposals()
                            if uses_fast_local_api(runtime)
                            else runtime.repair_proposals(current_constraint_view)
                        )
                        proposed.extend(
                            normalize_repair_transition(transition)
                            for transition in runtime_proposals
                        )

                    unique_proposals = []
                    seen_transitions = set()
                    for transition in proposed:
                        transition = normalize_repair_transition(transition)
                        key = transition_key(transition)
                        if key not in seen_transitions:
                            seen_transitions.add(key)
                            unique_proposals.append(transition)

                    best_repair = None
                    for transition in unique_proposals:
                        allowed, violation, after = repair_transition_evaluation(
                            transition
                        )
                        if not allowed or after is None:
                            continue
                        state = tuple(after.component_to_community)
                        if violation > current_violation:
                            continue
                        if violation == current_violation and state in seen_assignments:
                            continue
                        labels_after = np.asarray(state, dtype=int)
                        modularity_after = _objective_B_from_comp_assignment(
                            W_B,
                            labels_after,
                            K_used,
                        )
                        rank = (violation, -float(modularity_after))
                        if best_repair is None or rank < best_repair[0]:
                            best_repair = (rank, transition, state, after)

                    if best_repair is None:
                        split_for_repair = False
                        if allow_block_splitting:
                            multi_blocks = [
                                int(bi)
                                for bi, block in enumerate(blocks)
                                if len(block) > 1 and int(block2g[bi]) >= 0
                            ]
                            multi_blocks.sort(
                                key=lambda bi: attachment_to_group(
                                    bi,
                                    int(block2g[bi]),
                                )
                            )
                            if multi_blocks:
                                split_for_repair = split_block_into_singletons_in_place(
                                    multi_blocks[0]
                                )
                        if split_for_repair:
                            continue
                        break

                    _, selected_transition, selected_state, _ = best_repair
                    if not apply_atomic_component_transition(selected_transition):
                        break
                    seen_assignments.add(tuple(selected_state))
                    constraints_feasible, current_violation, current_constraint_view = (
                        evaluate_final_constraints()
                    )
                    if constraints_feasible:
                        break

                if not constraints_feasible:
                    continue

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
                    best_custom = None

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
                                transition = block_transition(
                                    bi,
                                    g_from,
                                    g_to,
                                    phase="feasible_search",
                                    label="move",
                                )
                                allowed, _, _ = evaluate_constraint_transition(
                                    transition,
                                    require_satisfied=True,
                                )
                                if not allowed:
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

                                    transition = VFDTransition.swap(
                                        tuple(int(component) for component in blocks[bi]),
                                        g1,
                                        tuple(int(component) for component in blocks[bj]),
                                        g2,
                                        phase="feasible_search",
                                        label="swap",
                                    )
                                    allowed, _, _ = evaluate_constraint_transition(
                                        transition,
                                        require_satisfied=True,
                                    )
                                    if not allowed:
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

                                        transition = VFDTransition(
                                            (
                                                VFDComponentMove(
                                                    tuple(int(component) for component in blocks[bj]),
                                                    g_to,
                                                    gk,
                                                ),
                                                VFDComponentMove(
                                                    tuple(int(component) for component in blocks[bi]),
                                                    g_from,
                                                    g_to,
                                                ),
                                            ),
                                            phase="feasible_search",
                                            label="ejection",
                                        )
                                        allowed, _, _ = evaluate_constraint_transition(
                                            transition,
                                            require_satisfied=True,
                                        )
                                        if not allowed:
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

                    if bound_constraints:
                        current_view = None
                        seen_custom = set()
                        for runtime in bound_constraints:
                            if uses_fast_local_api(runtime):
                                runtime_proposals = runtime.component_proposals()
                            else:
                                if current_view is None:
                                    current_view = assignment_view()
                                runtime_proposals = runtime.repair_proposals(
                                    current_view
                                )
                            for proposal in runtime_proposals:
                                if not isinstance(proposal, VFDTransition):
                                    raise TypeError(
                                        "Constraint repair proposals must be "
                                        "VFDTransition instances."
                                    )
                                transition = VFDTransition(
                                    proposal.moves,
                                    phase="feasible_search",
                                    label=proposal.label or "constraint-neighborhood",
                                )
                                key = transition_key(transition)
                                if key in seen_custom:
                                    continue
                                seen_custom.add(key)
                                if not component_transition_is_applicable(transition):
                                    continue
                                structural_ok, after = (
                                    structural_transition_is_feasible(transition)
                                )
                                if not structural_ok or after is None:
                                    continue
                                allowed, _, _ = evaluate_constraint_transition(
                                    transition,
                                    require_satisfied=True,
                                )
                                if not allowed:
                                    continue
                                labels_after = np.asarray(
                                    after.component_to_community,
                                    dtype=int,
                                )
                                total_after = _objective_B_from_comp_assignment(
                                    W_B,
                                    labels_after,
                                    K_used,
                                ) + w_coassoc * _objective_B_from_comp_assignment(
                                    C_comp,
                                    labels_after,
                                    K_used,
                                )
                                delta = float(total_after - base_total)
                                if best_custom is None or delta > best_custom[-1]:
                                    best_custom = ("custom", transition, delta)

                    best_action = None
                    for cand in (best_move, best_swap, best_eject, best_custom):
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

                    elif kind == "eject":
                        _, bi, g_from, g_to, bj, gk, _ = best_action
                        apply_ejection(int(bi), int(g_from), int(g_to), int(bj), int(gk))
                        set_tabu(int(bi), int(g_from), step, tenure)
                        set_tabu(int(bj), int(g_to), step, tenure)

                    else:
                        _, transition, _ = best_action
                        if not apply_atomic_component_transition(transition):
                            break

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
                            if not feas.can_add(bi, g_to):
                                continue
                            transition = block_transition(
                                bi,
                                g_from,
                                g_to,
                                phase="shake",
                                label="shake",
                            )
                            allowed, _, _ = evaluate_constraint_transition(
                                transition,
                                require_satisfied=True,
                            )
                            if allowed:
                                apply_move(bi, g_from, g_to, transition)
                                moved += 1
                                break
                    if moved >= shake_moves:
                        break

            comp2g_final = best_local_comp2g.copy()
            if not structural_assignment_is_feasible(comp2g_final):
                continue
            if not independently_validate_final_constraints(comp2g_final):
                continue
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
                "seed": None if seed is None else int(seed),
                "resolution": gamma,
                "total_balance_weight": total_balance_weight,
                "constraints": tuple(
                    str(getattr(runtime, "name", type(runtime).__name__))
                    for runtime in bound_constraints
                ),
                "constraint_repair_steps": int(repair_steps_used),
            }

            candidate = (Z, meta)
            candidate_Q = float(meta["objective_B_sum"])

            if (
                best_feasible is None
                or candidate_Q > float(best_feasible[1]["objective_B_sum"])
            ):
                best_feasible = candidate

            if candidate_Q > reference_Q + 1e-12:
                if (
                    best_improving is None
                    or candidate_Q > float(best_improving[1]["objective_B_sum"])
                ):
                    best_improving = candidate

    best = best_improving if best_improving is not None else best_feasible
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
                R_bounds=R_bounds,
                balance_weights=balance_weights,
                gamma=gamma,
                constraints=constraint_specs,
                component_members=input_component_members,
                constraint_repair_steps=constraint_repair_steps,
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


def refine_partition_modular_vfd(
    A: np.ndarray,
    partition: np.ndarray,
    *,
    a: Optional[np.ndarray] = None,
    m: Optional[float] = None,
    K: Optional[int] = 2,
    R: Optional[int] = 1,
    R_bounds: Optional[Tuple[int, int]] = None,
    balance_weights: Optional[Sequence[int]] = None,
    must_link: Sequence[Tuple[int, int]] = (),
    cannot_link: Sequence[Tuple[int, int]] = (),
    use_K_constraint: bool = False,
    shake_rounds: int = 3,
    gamma: float = 1.0,
    constraints: Sequence[VFDConstraint] = (),
    component_members: Optional[Sequence[Sequence[Any]]] = None,
    constraint_repair_steps: Optional[int] = None,
    seed: Optional[int] = 42,
    **kwargs: Any,
) -> Optional[np.ndarray]:
    """Adapt :func:`modular_very_fortunate_descent` to CSD refinement hooks.

    The decomposition calls refiners with ``A`` and ``partition`` and expects
    only a partition matrix in return.  This adapter supplies degree and graph
    volume defaults and unwraps ModularVFD's diagnostic metadata.
    """
    adjacency = np.asarray(A, dtype=float)
    candidate = np.asarray(partition)
    if candidate.ndim == 1:
        candidate = partition_vector_to_2d_matrix(candidate)
    strengths = adjacency.sum(axis=1) if a is None else np.asarray(a, dtype=float)
    volume = float(strengths.sum()) if m is None else float(m)
    out = modular_very_fortunate_descent(
        wz=candidate,
        A=adjacency,
        a=strengths,
        m=volume,
        K=K,
        R=R,
        R_bounds=R_bounds,
        balance_weights=balance_weights,
        must_link=must_link,
        cannot_link=cannot_link,
        use_K_constraint=use_K_constraint,
        shake_rounds=shake_rounds,
        gamma=gamma,
        constraints=constraints,
        component_members=component_members,
        constraint_repair_steps=constraint_repair_steps,
        seed=seed,
        **kwargs,
    )
    return None if out is None else out[0]


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
