"""Shared load-balancing bound calculations."""

from __future__ import annotations

from asunder.base.algorithms.modular_VFD import _range_bounds_from_KR


def resolve_balance_bounds(
    n_nodes: int,
    K: int,
    R: int,
    R_bounds: tuple[int | None, int | None] | None = None,
) -> tuple[int, int]:
    """Resolve and validate load-balancing community-size bounds.

    Parameters
    ----------
    n_nodes : int
        Number of graph nodes.
    K : int
        Number of requested communities.
    R : int
        Width of the default permitted size range.
    R_bounds : tuple[int or None, int or None] or None
        Optional explicit lower and upper bounds. A missing endpoint is
        replaced by one or ``n_nodes``, respectively.

    Returns
    -------
    lower : int
        Inclusive minimum community size.
    upper : int
        Inclusive maximum community size, capped at ``n_nodes``.

    Raises
    ------
    ValueError
        If node, community, range, or bound values are inconsistent.
    """

    # Below, we respect the range parameter: R_max = R_min + R
    # Python rounds using "ties-to-even" (e.g., 1.5->2, 2.5->2) leading to X.5 being rounded to X if X is even.
    # R_min below uses the floor function to simulate the more familiar "half-round-up." half_round_up(x) = ⌊x + 1/2⌋
    # For very large integers, e.g. I >= 2**53+1, float operations can lose precision. In that case, use the integer-only formula:
    # ⌊(I/K - R/2) + 1/2⌋ = ((2*I) - K*(R - 1)) // (2*K)
    # We, however, do not anticipate such issues as a graph that big should only be looked at from afar.

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
