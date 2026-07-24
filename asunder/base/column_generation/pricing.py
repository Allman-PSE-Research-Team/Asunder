"""Shared helpers for heuristic column-generation pricing."""

from __future__ import annotations

from numbers import Real
from typing import Any, Mapping

import numpy as np

from asunder.base.column_generation.master import compute_f_star


def build_dual_weight_matrix(
    A: np.ndarray,
    duals: Mapping[str, Any],
) -> tuple[np.ndarray, float]:
    """Convert master dual values into pairwise and constant pricing terms.

    One-dimensional dual arrays are lifted to pairwise terms using the
    half-sum convention already used by Asunder's heuristic pricing routines.
    Two-dimensional arrays are symmetrized, and scalar duals contribute to
    the constant reduced-cost term.

    Parameters
    ----------
    A : numpy.ndarray
        Square adjacency matrix defining the required dual-matrix shape.
    duals : mapping[str, Any]
        Master dual values. Supported values are real scalars, one-dimensional
        node arrays, two-dimensional pair arrays, and ``None``.

    Returns
    -------
    dual_weight : numpy.ndarray
        Symmetric pairwise dual-weight matrix.
    constant : float
        Sum of scalar dual terms.

    Raises
    ------
    ValueError
        If ``A`` is not square, a dual has an incompatible shape, or a value
        is not finite.
    TypeError
        If a dual value has an unsupported type.
    """

    adjacency = np.asarray(A)
    if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError("A must be a square matrix.")

    n_nodes = adjacency.shape[0]
    dual_weight = np.zeros(adjacency.shape, dtype=np.float64)
    constant = 0.0

    for name, dual in duals.items():
        if dual is None:
            continue

        if isinstance(dual, np.ndarray):
            values = np.asarray(dual, dtype=np.float64)
            if not np.all(np.isfinite(values)):
                raise ValueError(f"Dual {name!r} contains NaN or infinity.")

            if values.ndim == 1:
                if values.shape[0] != n_nodes:
                    raise ValueError(
                        f"One-dimensional dual {name!r} must have length {n_nodes}."
                    )
                dual_weight += 0.5 * (values[:, None] + values[None, :])
            elif values.ndim == 2:
                if values.shape != adjacency.shape:
                    raise ValueError(
                        f"Two-dimensional dual {name!r} must have shape "
                        f"{adjacency.shape}."
                    )
                dual_weight += 0.5 * (values + values.T)
            else:
                raise ValueError(f"Dual {name!r} must be scalar, 1D, or 2D.")
        elif isinstance(dual, Real):
            value = float(dual)
            if not np.isfinite(value):
                raise ValueError(f"Dual {name!r} contains NaN or infinity.")
            constant += value
        else:
            raise TypeError(
                f"Dual {name!r} has unsupported type {type(dual).__name__}."
            )

    return dual_weight, constant


def compute_reduced_cost(
    A: np.ndarray,
    a: np.ndarray,
    m: float,
    partition: np.ndarray,
    duals: Mapping[str, Any],
    *,
    gamma: float = 1.0,
) -> float:
    """Evaluate a partition with Asunder's exact reduced-cost objective.

    Parameters
    ----------
    A : numpy.ndarray
        Original floating-point adjacency matrix.
    a : numpy.ndarray
        Degree/strength vector used by the modularity null model.
    m : float
        Total directed graph weight.
    partition : numpy.ndarray
        Candidate binary or fractional co-association matrix.
    duals : mapping[str, Any]
        Master dual values accepted by :func:`build_dual_weight_matrix`.
    gamma : float, default=1.0
        Modularity resolution parameter.

    Returns
    -------
    float
        Exact modularity contribution minus pairwise and constant dual terms.

    Raises
    ------
    ValueError
        If the graph or dual values are invalid.
    TypeError
        If a dual value has an unsupported type.
    """

    dual_weight, constant = build_dual_weight_matrix(A, duals)
    objective = compute_f_star(A, a, m, partition, gamma=gamma)
    return float(objective - np.sum(dual_weight * partition) - constant)
