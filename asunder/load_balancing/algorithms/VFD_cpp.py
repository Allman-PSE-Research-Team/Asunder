"""Load-balancing VFD with shared Python preprocessing and a native C++ search."""

from __future__ import annotations

from time import perf_counter
from typing import Any, Sequence

import numpy as np
from scipy import sparse

from asunder.base.algorithms.modular_VFD import (
    _CLUSTERING_METHODS,
    _build_coassociation_matrix,
    _build_components,
    _component_matrices_from_node_matrix,
    _component_sum_matrix_B,
    _ensure_at_least_K_blocks,
    _fingerprint_blocks_from_rounded_rows,
    _make_blocks_conflict_free,
    _node_partition_B_sum,
    _normalize_pair,
    _symmetrize_unitdiag,
    _target_sizes_from_bounds,
)
from asunder.base.branch_and_price.symmetry_detection import weighted_constraint_orbits
from asunder.base.utils.graph import partition_vector_to_2d_matrix
from asunder.base.utils.matrix import DEFAULT_MAX_DENSE_WORKING_BYTES, checked_to_dense
from asunder.load_balancing.algorithms import _vfd_wrapper as _native


def _integer(value, name, minimum=None, maximum=None):
    """Validate an integral scalar without rounding integer inputs through float64.

    Parameters
    ----------
    value : int or float
        Integer or finite floating-point value equal to an integer.
    name : str
        Parameter name used in errors.
    minimum, maximum : int, optional
        Inclusive limits on the converted integer.

    Returns
    -------
    int
        Exact integer value of the supplied scalar.

    Raises
    ------
    ValueError
        If the scalar is fractional, nonnumeric, nonfinite, or outside the limits.
    """
    if isinstance(value, (int, np.integer)):
        result = int(value)
    elif isinstance(value, (float, np.floating)) and np.isfinite(value) and value == int(value):
        result = int(value)
    else:
        raise ValueError(f"{name} must contain finite integers.")
    if (minimum is not None and result < minimum) or (maximum is not None and result > maximum):
        raise ValueError(f"{name} is outside its supported integer range.")
    return result


def _node_pairs(pairs, name, n):
    """Validate and normalize zero-based node pairs from lists or NumPy arrays.

    Parameters
    ----------
    pairs : sequence of pair of int or None
        Pairwise node constraints.
    name : str
        Constraint name used in validation errors.
    n : int
        Number of original nodes.

    Returns
    -------
    list of tuple of int
        Ordered node pairs with exact integer indices in ``0..n-1``.

    Raises
    ------
    ValueError
        If a pair has the wrong length or contains an invalid node index.
    """
    out = []
    for pair in () if pairs is None else pairs:
        try:
            first, second = pair
        except (TypeError, ValueError):
            raise ValueError(f"Each {name} entry must contain two node indices.") from None
        out.append(
            _normalize_pair(
                _integer(first, name, 0, n - 1),
                _integer(second, name, 0, n - 1),
            )
        )
    return out


def very_fortunate_descent(
    wz: np.ndarray,
    A: np.ndarray,
    a: np.ndarray,
    m: float,
    # constraint parameters
    K: int = 2,
    R: int = 1,
    R_bounds: tuple | None = None,
    must_link: Sequence[tuple[int, int]] = (),
    cannot_link: Sequence[tuple[int, int]] = (),
    seed: int | None = 42,
    fingerprint_decimals: int = 6,
    allow_block_splitting: bool = True,
    K_search_radius: int = 0,  # If 0, keep K_used == K
    # greediness-defining parameters
    restarts: int = 6,
    local_iters: int = 60,
    # tie-break toward co-association cohesion
    w_coassoc: float = 0.05,
    # clustering parameters
    clustering_Ks: Sequence[int] | None = None,
    clustering_seeds: Sequence[int] = (0, 1, 2),
    clustering_methods: Sequence[str] = ("kmeans", "gmm", "spectral"),
    # use wz as the cluster identifier
    wz_is_C_node: bool = False,
    # maximum number of tabu steps
    tabu_max_steps: int = 60,
    # perturb solution slightly
    shake_rounds: int = 3,
    orbit_fallback: bool = False,
    balance_weights: Sequence[int] | None = None,
    gamma: float = 1.0,
    *,
    check_invariants: bool = False,
    max_dense_working_bytes: int | None = DEFAULT_MAX_DENSE_WORKING_BYTES,
) -> tuple[np.ndarray, dict[str, Any]] | None:
    """
    Build a feasible column using shared Python preprocessing and native C++ search.
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
    K : int
        Baseline number of communities. With ``K_search_radius=0``, exactly
        this many communities are required.
    R : int or None
        Width of the allowed community-load range. Also corresponds to the
        load balance tightness (smaller R implies tighter load balance).
        For a selected cluster count, the lower and upper bounds are computed from the corresponding
        balanced range rule.
    R_bounds : tuple or None
        Inclusive lower and upper limits on total node weight in communities.
    must_link : sequence of tuple of int
        Must-link pairs.
    cannot_link : sequence of tuple of int
        Cannot-link pairs.
    seed : int or None, default=42
        Random seed.
    fingerprint_decimals : int, default=6
        Decimal rounding used to form fingerprint blocks.
    allow_block_splitting : bool, default=True
        If True, allow refinement of coarse fingerprint blocks.
    K_search_radius : int, default=0
        Maximum distance below or above the baseline K to search. Every
        candidate uses the load bounds resolved from the original ``K`` and
        ``R`` or from ``R_bounds``.
    restarts : int, default=6
        Number of constructive restarts.
    local_iters : int, default=60
        Each tabu round uses ``max(tabu_max_steps, 5 * local_iters)`` steps.
    w_coassoc : float, default=0.05
        Weight of the co-association cohesion term in local decisions.
    clustering_Ks : sequence of int or None, default=None
        Community counts used to build the co-association matrix.
    clustering_seeds : sequence of int, default=(0, 1, 2)
        Seeds used during co-association construction.
    clustering_methods : sequence of str, default=("kmeans", "gmm", "spectral")
        Co-association ensemble methods. Use ``"gmm_diag"`` for a mixture
        with diagonal covariance matrices.
    wz_is_C_node : bool, default=False
        If True, treat ``wz`` directly as the node-level co-association matrix.
    tabu_max_steps : int, default=60
        Minimum tabu-round budget before applying ``5 * local_iters``.
    shake_rounds : int, default=3
        Number of tabu invocations, with feasible perturbations between them.
        Zero returns constructive/repair search results only.
    orbit_fallback : bool, default=False
        If True, build a fallback co-association proxy from orbits when needed.
    balance_weights : sequence of int or None, default=None
        Positive integer node weights used to enforce community load bounds.
        Unit weights are used by default. Fractional weights are not scaled
        automatically; scale weights and explicit ``R_bounds`` to common
        integer units before calling.
    gamma : float, default=1.0
        Resolution parameter used by the modularity objective.

    check_invariants : bool, default=False
        Recompute native state and action gains after transitions. This is
        expensive and intended for correctness checks, not performance runs.
    max_dense_working_bytes : int or None, default=536870912
        Maximum estimated dense workspace. Sparse inputs are converted once
        through Asunder's guarded boundary, reserving twelve float64 square
        arrays as a conservative estimate. None disables the estimate guard.

    Returns
    -------
    tuple of (numpy.ndarray, dict) or None
        A pair ``(Z_col, meta)`` if a feasible column is found, else ``None``.
        Metadata includes the original B-sum score, node/component labels,
        community loads, preprocessing/search timings and native action counts.
        ``None`` means that this heuristic did not find a feasible assignment.

    Notes
    -----
    The native RNG is mt19937_64. A seed is repeatable within the same compiled
    build, but need not reproduce Python's NumPy RNG trajectory or partition.
    Clustering, component contraction and fingerprint construction use Asunder's
    existing Python helpers. The C++ search releases the GIL. Dense matrices
    retain quadratic memory cost. Adjacency matrices must be undirected.
    ``objective_B_sum`` is unnormalized; divide by ``m`` for modularity.
    """

    started = perf_counter()
    _native.load_library()
    max_index = int(np.iinfo(np.int32).max)
    max_load = int(np.iinfo(np.int64).max)
    restarts = _integer(restarts, "restarts", 1, max_index)
    K_search_radius = _integer(K_search_radius, "K_search_radius", 0)
    local_iters = _integer(local_iters, "local_iters", 0, (max_index - 7) // 5)
    tabu_max_steps = _integer(tabu_max_steps, "tabu_max_steps", 0, max_index - 7)
    shake_rounds = _integer(shake_rounds, "shake_rounds", 0, max_index)
    fingerprint_decimals = _integer(fingerprint_decimals, "fingerprint_decimals", 0)
    if not np.isfinite(w_coassoc):
        raise ValueError("w_coassoc must be finite.")
    K = _integer(K, "K", 1)
    if R is not None:
        R = _integer(R, "R", 0)
    gamma = float(gamma)
    if not np.isfinite(gamma) or gamma < 0:
        raise ValueError("gamma must be a finite nonnegative value.")
    clustering_methods = tuple(() if clustering_methods is None else clustering_methods)
    unknown_methods = set(clustering_methods) - _CLUSTERING_METHODS
    if unknown_methods:
        raise ValueError(f"Unknown clustering methods: {sorted(unknown_methods)}.")

    rng = np.random.default_rng(seed)

    wz = checked_to_dense(
        wz,
        dtype=float,
        working_arrays=12.0,
        max_dense_working_bytes=max_dense_working_bytes,
        operation="C++ VFD evidence preprocessing",
    )
    if wz.ndim != 2 or wz.shape[0] != wz.shape[1]:
        raise ValueError("wz must be a square matrix.")
    N = int(wz.shape[0])
    A = checked_to_dense(
        A,
        dtype=float,
        working_arrays=12.0,
        max_dense_working_bytes=max_dense_working_bytes,
        operation="C++ VFD adjacency preprocessing",
    )
    a = np.asarray(a, dtype=float).reshape(-1)
    if A.shape != (N, N) or a.shape != (N,):
        raise ValueError("wz and A must be (N,N), and a must be (N,).")
    if not np.allclose(A, A.T, rtol=1e-12, atol=1e-12):
        raise ValueError("A must be symmetric for undirected VFD.")
    weights = np.ones(N, dtype=np.int64)
    if balance_weights is not None:
        raw_weights = np.asarray(balance_weights, dtype=object)
        if raw_weights.shape != (N,):
            raise ValueError("balance_weights must have one value per node.")
        values = [_integer(value, "balance_weights", 1, max_load) for value in raw_weights]
        if sum(values) > max_load:
            raise ValueError("Total balance weight exceeds int64 storage.")
        weights = np.asarray(values, dtype=np.int64)
    must_link = _node_pairs(must_link, "must_link", N)
    cannot_link = _node_pairs(cannot_link, "cannot_link", N)
    if not np.all(np.isfinite(A)) or not np.all(np.isfinite(a)):
        raise ValueError("A and a must contain only finite values.")
    try:
        m = float(m)
    except (TypeError, ValueError):
        raise ValueError("m must be a finite positive value.") from None
    if N and (not np.isfinite(m) or m <= 0):
        raise ValueError("m must be a finite positive value.")
    if N == 0:
        g0 = np.zeros(0, dtype=int)
        return partition_vector_to_2d_matrix(g0), {
            "r_min": 0,
            "r_max": 0,
            "K_used": 0,
            "seed": None if seed is None else int(seed),
            "resolution": gamma,
            "total_balance_weight": 0,
            "labels": g0,
            "component_labels": g0.copy(),
            "community_loads": g0.copy(),
            "backend": "cpp",
            "objective_B_sum": 0.0,
            "reference_objective_B_sum": 0.0,
            "improves_reference": False,
            "native_search_seconds": 0.0,
            "preprocessing_seconds": float(perf_counter() - started),
            "binding_seconds": 0.0,
            "total_seconds": float(perf_counter() - started),
            "native_runs": [],
        }

    sym = _symmetrize_unitdiag(wz)

    comp = _build_components(
        N,
        must_link,
        cannot_link,
        node_weights=None,
    )
    if comp is None:
        return None

    Cn = int(comp["C"])

    # The validated positive loads sum to at most INT64_MAX. Aggregate them
    # directly, avoiding float64 coercion in the general component builder.
    cweight = np.zeros(Cn, dtype=np.int64)
    np.add.at(cweight, comp["cid"], weights)
    total_balance_weight = int(cweight.sum())

    if R_bounds is None and R is not None:
        r_min = max(1, (2 * total_balance_weight + (1 - R) * K) // (2 * K))
        r_max = r_min + R
    elif R_bounds is not None:
        raw_bounds = np.asarray(R_bounds, dtype=object)
        if raw_bounds.shape != (2,):
            raise ValueError("R_bounds must contain exactly two values.")
        r_min, r_max = (_integer(value, "R_bounds", 0, max_load) for value in raw_bounds)
        if r_max < r_min:
            raise ValueError("R_bounds must satisfy 0 <= lower <= upper.")
    else:
        raise ValueError("R is required when R_bounds is not provided.")

    if r_max <= 0:
        return None
    if r_max > max_load:
        raise ValueError("Upper load bound exceeds int64 storage.")
    if restarts > max_index - Cn:
        raise ValueError("restarts exceeds the supported native restart count.")
    k_lo = (total_balance_weight + r_max - 1) // r_max
    k_hi = total_balance_weight // max(1, r_min)
    K_start = max(1, k_lo, K - K_search_radius)
    K_end = min(Cn, k_hi, K + K_search_radius)
    K_candidates = range(K_start, K_end + 1)
    if not K_candidates:
        return None

    if clustering_Ks is None:
        clustering_Ks = tuple(
            sorted(
                {
                    max(2, k_lo),
                    min(k_hi, int(K)),
                    min(k_hi, int(K) + 2),
                }
            )
        )

    if int(cweight.max()) > r_max:
        return None

    # Build co-association evidence before reducing matrices to components.
    if not wz_is_C_node:
        C_node = _build_coassociation_matrix(
            sym, n_components_list=clustering_Ks, seeds=clustering_seeds, methods=clustering_methods
        )
    else:
        C_node = wz

    if C_node is None:
        if orbit_fallback:
            symmetry = weighted_constraint_orbits(A)
            C_node = partition_vector_to_2d_matrix(
                symmetry.rep,
                max_dense_working_bytes=max_dense_working_bytes,
            )
        else:
            C_node = sym.copy()
    C_node = _symmetrize_unitdiag(C_node)
    reference_Q = _node_partition_B_sum(A, a, m, sym, gamma=gamma)

    C_comp = _component_matrices_from_node_matrix(
        C_node, comp
    )  # average co-association between components
    W_B = _component_sum_matrix_B(
        A, a, m, comp, gamma=gamma
    )  # sum modularity matrix between components

    W_B = np.require((W_B + W_B.T) * 0.5, dtype=np.float64, requirements=["F", "A"])
    C_comp = np.require((C_comp + C_comp.T) * 0.5, dtype=np.float64, requirements=["F", "A"])
    component_pairs = sorted(
        {_normalize_pair(int(comp["cid"][i]), int(comp["cid"][j])) for i, j in cannot_link}
    )
    fingerprints = _fingerprint_blocks_from_rounded_rows(
        C_comp,
        comp,
        fingerprint_decimals=fingerprint_decimals,
        r_max=r_max,
        component_weights=cweight,
    )
    fingerprints = _make_blocks_conflict_free(
        fingerprints,
        use_bitmask=bool(comp["use_bitmask"]),
        forb=comp["forb_mask"],
        comp_bit=comp["comp_bit"],
    )
    best_run = None
    best_labels = None
    native_runs = []
    native_seconds = 0.0
    boundary_seconds = 0.0
    preprocessing_seconds = perf_counter() - started

    for K_used in K_candidates:
        block_start = perf_counter()
        target = _target_sizes_from_bounds(total_balance_weight, K_used, r_min, r_max)
        blocks = _ensure_at_least_K_blocks(fingerprints, K_used, cweight)
        native_seed = int(rng.integers(0, 2**64, dtype=np.uint64))
        preprocessing_seconds += perf_counter() - block_start
        call_start = perf_counter()
        result = _native.search(
            W_B,
            C_comp,
            cweight,
            component_pairs,
            blocks,
            target,
            K_used,
            r_min,
            r_max,
            restarts,
            local_iters,
            tabu_max_steps,
            shake_rounds,
            float(w_coassoc),
            bool(allow_block_splitting),
            bool(check_invariants),
            native_seed,
        )
        call_seconds = perf_counter() - call_start
        native_seconds += result["native_seconds"]
        boundary_seconds += max(0.0, call_seconds - result["native_seconds"])
        run_stats = {key: value for key, value in result.items() if key != "component_labels"}
        run_stats.update(K_used=int(K_used), native_seed=native_seed)
        native_runs.append(run_stats)
        if result["found"] and (
            best_run is None or result["objective_B_sum"] > best_run["objective_B_sum"]
        ):
            best_run = run_stats
            best_labels = result["component_labels"]

    if best_run is None:
        return None
    labels = best_labels[comp["cid"]]
    Z = partition_vector_to_2d_matrix(labels, max_dense_working_bytes=max_dense_working_bytes)
    community_loads = np.zeros(best_run["K_used"], dtype=np.int64)
    np.add.at(community_loads, best_labels, cweight)
    objective = float(best_run["objective_B_sum"])
    return Z, {
        "r_min": r_min,
        "r_max": r_max,
        "K_used": best_run["K_used"],
        "objective_B_sum": objective,
        "reference_objective_B_sum": float(reference_Q),
        "improves_reference": bool(objective > reference_Q + 1e-12),
        "fingerprint_decimals": fingerprint_decimals,
        "allow_block_splitting": bool(allow_block_splitting),
        "seed": None if seed is None else int(seed),
        "resolution": gamma,
        "total_balance_weight": total_balance_weight,
        "labels": labels,
        "component_labels": best_labels,
        "community_loads": community_loads,
        "backend": "cpp",
        "native_seed": best_run["native_seed"],
        "check_invariants": bool(check_invariants),
        "native_runs": native_runs,
        "preprocessing_seconds": preprocessing_seconds,
        "native_search_seconds": native_seconds,
        "binding_seconds": boundary_seconds,
        "total_seconds": perf_counter() - started,
    }


def refine_partition(
    A: np.ndarray,
    partition: np.ndarray,
    a: np.ndarray | None = None,
    m: float | None = None,
    K: int = 2,
    R: int = 1,
    R_bounds: tuple | None = None,
    must_link: Sequence[tuple[int, int]] = (),
    cannot_link: Sequence[tuple[int, int]] = (),
    seed: int | None = 42,
    fingerprint_decimals: int = 6,
    allow_block_splitting: bool = True,
    K_search_radius: int = 0,
    restarts: int = 6,
    local_iters: int = 60,
    w_coassoc: float = 0.05,
    clustering_Ks: Sequence[int] | None = None,
    clustering_seeds: Sequence[int] = (0, 1, 2),
    clustering_methods: Sequence[str] = ("kmeans", "gmm", "spectral"),
    wz_is_C_node: bool = False,
    tabu_max_steps: int = 60,
    shake_rounds: int = 3,
    orbit_fallback: bool = False,
    balance_weights: Sequence[int] | None = None,
    gamma: float = 1.0,
    *,
    check_invariants: bool = False,
    return_metadata: bool = False,
    max_dense_working_bytes: int | None = DEFAULT_MAX_DENSE_WORKING_BYTES,
):
    """Refine a partition through the C++ load-balancing VFD backend.

    Parameters
    ----------
    A : numpy.ndarray or scipy sparse matrix, shape (N, N)
        Finite symmetric adjacency matrix of an undirected graph.
    partition : array-like or scipy sparse matrix, shape (N,) or (N, N)
        Node labels or a node co-association/partition matrix. Label vectors
        are converted using Asunder's partition-matrix helper.
    a : numpy.ndarray, shape (N,), optional
        Degree-like vector in ``B = A - gamma * outer(a, a) / m``. Defaults
        to ``A.sum(axis=1)``.
    m : float, optional
        Positive modularity normalizer. Defaults to ``a.sum()``. For a
        standard undirected graph this is twice the total edge weight.
    K : int, default=2
        Required number of communities when ``K_search_radius=0``.
    R : int or None, default=1
        Allowed integral load-range width used to derive bounds around the
        original K. Explicit ``R_bounds`` overrides this rule.
    R_bounds : tuple of int, optional
        Inclusive lower and upper community-load bounds. Zero is permitted
        as the lower bound; returned communities must still be nonempty.
    must_link, cannot_link : sequence of pair of int
        Zero-based node pairs required to share or separate communities.
        Must-link components are never split by the native search.
    seed : int or None, default=42
        Seed for deriving per-K native search seeds. ``None`` uses entropy.
    fingerprint_decimals : int, default=6
        Rounding precision for initial component fingerprint blocks.
    allow_block_splitting : bool, default=True
        Permit fingerprint blocks to split into must-link components.
    K_search_radius : int, default=0
        Explore community counts within this distance from K, using bounds
        calculated from the original K or supplied explicit bounds.
    restarts : int, default=6
        Constructive restart count. Splitting can add constructive retries.
    local_iters : int, default=60
        Each tabu round uses ``max(tabu_max_steps, 5 * local_iters)`` steps.
    w_coassoc : float, default=0.05
        Cohesion weight in local-search decisions. Final solutions are
        ranked by the unnormalized B score alone.
    clustering_Ks : sequence of int, optional
        Counts used by Asunder's co-association clustering ensemble.
    clustering_seeds : sequence of int, default=(0, 1, 2)
        Seeds for the Python clustering ensemble.
    clustering_methods : sequence of str, default=("kmeans", "gmm", "spectral")
        Co-association ensemble methods, including ``"gmm_diag"`` for
        diagonal covariance. An empty sequence selects fallback evidence.
    wz_is_C_node : bool, default=False
        Treat the partition matrix directly as co-association evidence,
        skipping the clustering ensemble.
    tabu_max_steps : int, default=60
        Minimum tabu-round budget before applying ``5 * local_iters``.
    shake_rounds : int, default=3
        Number of tabu invocations, with feasible perturbations between
        them. Zero returns constructive/repair search results only.
    orbit_fallback : bool, default=False
        Use Asunder's weighted orbit proxy if ensemble construction fails.
    balance_weights : sequence of int, shape (N,), optional
        Positive integral node loads; defaults to unit weights. Scale
        fractional loads and explicit bounds to common integer units first.
    gamma : float, default=1.0
        Finite nonnegative modularity resolution parameter.
    check_invariants : bool, default=False
        Enable expensive native state and transition-gain checks.
    return_metadata : bool, default=False
        Return ``(Z, metadata)`` for profiling. Keep False when registering
        this function as an Asunder refinement hook.
    max_dense_working_bytes : int or None, default=536870912
        Maximum estimated dense workspace including sparse-to-dense conversion.
        A conservative twelve-array estimate is used; None disables the guard.

    Returns
    -------
    numpy.ndarray of bool, shape (N, N), or tuple or None
        Refined partition matrix by default. With ``return_metadata=True``,
        a pair containing the matrix and timing/constraint/search metadata.
        Returns None when the heuristic cannot find a feasible column.

    Raises
    ------
    ValueError
        If shapes, symmetry, numeric values or search parameters are invalid.
    MemoryError
        If the estimated dense workspace exceeds the caller's memory limit.
    RuntimeError
        If the native library is unavailable or a native invariant fails.

    Notes
    -----
    Select this refiner with ``LoadBalancer(..., refinement_backend="cpp")``.
    Clustering and matrix preparation run in Python/NumPy/scikit-learn.
    The single-threaded native search releases the GIL.
    """
    started = perf_counter()
    _native.load_library()
    A = checked_to_dense(
        A,
        dtype=float,
        working_arrays=12.0,
        max_dense_working_bytes=max_dense_working_bytes,
        operation="C++ VFD refinement adjacency",
    )
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be a square matrix.")
    if a is None:
        a = A.sum(axis=1)
    if m is None:
        m = float(np.asarray(a, dtype=float).sum())
    if sparse.issparse(partition) and partition.ndim == 1:
        partition = checked_to_dense(
            partition,
            dtype=partition.dtype,
            max_dense_working_bytes=max_dense_working_bytes,
            operation="C++ VFD refinement labels",
        )
    elif not sparse.issparse(partition):
        partition = np.asarray(partition)
    if partition.ndim == 1:
        if partition.shape != (A.shape[0],):
            raise ValueError("partition must contain one label per node.")
        partition = partition_vector_to_2d_matrix(
            partition,
            max_dense_working_bytes=max_dense_working_bytes,
        )
    elif partition.shape != A.shape:
        raise ValueError("partition must contain a label vector or a square co-association matrix.")
    conversion_seconds = perf_counter() - started
    out = very_fortunate_descent(
        wz=partition,
        A=A,
        a=a,
        m=m,
        K=K,
        R=R,
        R_bounds=R_bounds,
        must_link=must_link,
        cannot_link=cannot_link,
        seed=seed,
        fingerprint_decimals=fingerprint_decimals,
        allow_block_splitting=allow_block_splitting,
        K_search_radius=K_search_radius,
        restarts=restarts,
        local_iters=local_iters,
        w_coassoc=w_coassoc,
        clustering_Ks=clustering_Ks,
        clustering_seeds=clustering_seeds,
        clustering_methods=clustering_methods,
        wz_is_C_node=wz_is_C_node,
        tabu_max_steps=tabu_max_steps,
        shake_rounds=shake_rounds,
        orbit_fallback=orbit_fallback,
        balance_weights=balance_weights,
        gamma=gamma,
        check_invariants=check_invariants,
        max_dense_working_bytes=max_dense_working_bytes,
    )
    if out is not None:
        out[1]["preprocessing_seconds"] += conversion_seconds
        out[1]["adapter_conversion_seconds"] = float(conversion_seconds)
        out[1]["total_seconds"] = float(perf_counter() - started)
    if out is None or return_metadata:
        return out
    return out[0]
