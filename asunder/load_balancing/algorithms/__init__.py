"""Public algorithm exports for load-balanced community detection."""

from asunder.load_balancing.algorithms.VFD import (
    very_fortunate_descent,
    very_fortunate_descent_legacy,
)

__all__ = [
    "very_fortunate_descent_legacy",
    "very_fortunate_descent"
]
