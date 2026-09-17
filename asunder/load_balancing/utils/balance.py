"""Shared load-balancing bound calculations."""

from __future__ import annotations

import numpy as np

from asunder.base.algorithms.modular_VFD import _range_bounds_from_KR
from asunder.base.utils.graph import partition_matrix_to_vector


def resolve_balance_bounds(
    n_nodes: int | float,
    K: int,
    R: int,
    R_bounds: tuple[int | None, int | None] | None = None,
) -> tuple[int, int]:
    """Resolve and validate integer community-load bounds.

    Parameters
    ----------
    n_nodes : int or float
        Total node weight, which must be integer-valued. Equals the node
        count for unit weights; contraction preserves this total.
    K : int
        Number of requested communities.
    R : int
        Width of the default permitted load range, in node-weight units.
    R_bounds : tuple[int or None, int or None] or None
        Optional explicit integer lower and upper load bounds, in node-weight
        units. A missing endpoint is replaced by one or ``n_nodes``, respectively.

    Returns
    -------
    lower : int
        Inclusive minimum community load.
    upper : int
        Inclusive maximum community load, capped at ``n_nodes``.

    Raises
    ------
    ValueError
        If node, community, range, or bound values are inconsistent.

    Notes
    -----
    Default bounds use half-up rounding about ``n_nodes / K``. Consequently,
    scaling both the weights and ``R`` need not scale the derived bounds
    exactly. To preserve existing bounds, scale explicit ``R_bounds`` instead.
    Node weights and bounds are never scaled automatically.
    """

    # Below, we respect the range parameter: R_max = R_min + R
    # Python rounds using "ties-to-even" (e.g., 1.5->2, 2.5->2) leading to X.5 being rounded to X if X is even.
    # R_min below uses the floor function to simulate the more familiar "half-round-up." half_round_up(x) = ⌊x + 1/2⌋
    # For very large integers, e.g. I >= 2**53+1, float operations can lose precision. In that case, use the integer-only formula:
    # ⌊(I/K - R/2) + 1/2⌋ = ((2*I) - K*(R - 1)) // (2*K)
    # We, however, do not anticipate such issues as a graph that big should only be looked at from afar.

    if not float(n_nodes).is_integer():
        raise ValueError(
            "The total balance weight must be an integer. Scale node weights "
            "and explicit R_bounds to common integer units before calling."
        )
    n_nodes = int(n_nodes)
    if n_nodes < 0:
        raise ValueError("n_nodes must be nonnegative.")
    if K < 1:
        raise ValueError("K must be positive.")
    if R < 0:
        raise ValueError("R must be nonnegative.")

    if R_bounds is None or R_bounds == (None, None):
        R_min, R_max = _range_bounds_from_KR(n_nodes, K, R)
    else:
        raw_R_min, raw_R_max = R_bounds
        R_min = 1 if raw_R_min is None else int(raw_R_min)
        R_max = n_nodes if raw_R_max is None else int(raw_R_max)

    R_min = int(R_min)
    R_max = int(R_max)
    if R_min < 0:
        raise ValueError("The lower community-size bound (R_min) must be nonnegative.")
    if R_min > R_max:
        raise ValueError(
            "Cardinality bounds are improperly defined: R_min must not "
            "exceed R_max."
        )
    if R_max > n_nodes and n_nodes > 0:
        R_max = n_nodes

    return R_min, R_max


def epsilon_for_upper_bound(n_nodes: int, K: int, R_max: int) -> float:
    """Map an allowed maximum part size to QMETIS's upper imbalance slack.

    This expands QMETIS's search envelope; the restricted master and final
    refinement remain responsible for enforcing Asunder's actual bounds.

    Parameters
    ----------
    n_nodes : int
        Number of graph nodes.
    K : int
        Number of requested communities.
    R_max : int
        Inclusive maximum community size allowed by the master problem.

    Returns
    -------
    float
        Nonnegative QMETIS imbalance epsilon satisfying
        ``(1 + epsilon) * (n_nodes / K) == R_max`` when ``R_max`` exceeds the
        average part size.

    Raises
    ------
    ValueError
        If ``n_nodes`` or ``K`` is not positive, or ``R_max`` is negative.
    """

    if n_nodes <= 0:
        raise ValueError("n_nodes must be positive.")
    if K <= 0:
        raise ValueError("K must be positive.")
    if R_max < 0:
        raise ValueError("R_max must be nonnegative.")

    average = n_nodes / K
    return max(0.0, float(R_max) / average - 1.0)


def partition_satisfies_balance_constraints(
    partition,
    *,
    K: int,
    R: int,
    R_bounds: tuple[int | None, int | None] | None = None,
    balance_weights=None,
) -> bool:
    """Return whether a partition has exactly ``K`` weight-balanced groups."""
    matrix = partition
    n_nodes = matrix.shape[0]
    try:
        K = int(K)
        R = int(R)
    except (TypeError, ValueError):
        return False
    if balance_weights is None:
        weights = np.ones(n_nodes, dtype=int)
    else:
        weights = np.asarray(balance_weights)
        if weights.shape != (n_nodes,):
            return False
        if (
            not np.all(np.isfinite(weights))
            or np.any(weights <= 0)
            or not np.all(weights == np.rint(weights))
        ):
            return False
    labels = partition_matrix_to_vector(matrix)
    communities = np.unique(labels)
    if communities.size != K:
        return False
    loads = np.bincount(labels, weights=weights, minlength=K)
    try:
        lower, upper = resolve_balance_bounds(
            float(np.sum(weights)),
            K,
            R,
            R_bounds,
        )
    except ValueError:
        return False
    return bool(np.all((loads >= lower) & (loads <= upper)))
