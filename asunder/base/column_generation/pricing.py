"""Shared helpers for heuristic column-generation pricing."""

from __future__ import annotations

from numbers import Real
from typing import Any, Mapping

import numpy as np
from scipy import sparse

from asunder.base.column_generation.master import compute_f_star
from asunder.base.utils.matrix import (
    DEFAULT_MAX_DENSE_WORKING_BYTES,
    ensure_dense_working_set,
)


def build_dual_weight_matrix(
    A: np.ndarray | sparse.spmatrix,
    duals: Mapping[str, Any],
    *,
    max_dense_working_bytes: int | None = DEFAULT_MAX_DENSE_WORKING_BYTES,
) -> tuple[np.ndarray | sparse.csr_matrix, float]:
    """Convert master dual values into pairwise and constant pricing terms.

    One-dimensional dual arrays are lifted to pairwise terms using the
    half-sum convention already used by Asunder's heuristic pricing routines.
    Two-dimensional arrays are symmetrized, and scalar duals contribute to
    the constant reduced-cost term.

    Parameters
    ----------
    A : numpy.ndarray or scipy.sparse.spmatrix
        Square adjacency matrix defining the required dual-matrix shape.
    duals : mapping[str, Any]
        Master dual values. Supported values are real scalars, one-dimensional
        node arrays, two-dimensional pair arrays, and ``None``.
    max_dense_working_bytes : int or None, default=536870912
        Maximum estimated dense working set if the dual structure requires a
        dense pairwise matrix. ``None`` disables the guard.

    Returns
    -------
    dual_weight : numpy.ndarray or scipy.sparse.csr_matrix
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

    if len(A.shape) != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be a square matrix.")

    n_nodes = A.shape[0]
    matrix_duals = [
        dual
        for dual in duals.values()
        if dual is not None and (sparse.issparse(dual) or isinstance(dual, np.ndarray))
    ]
    requires_dense = not sparse.issparse(A) or any(
        not sparse.issparse(dual) and np.asarray(dual).ndim in {1, 2}
        for dual in matrix_duals
    )
    if requires_dense:
        ensure_dense_working_set(
            A.shape,
            dtype=np.float64,
            working_arrays=2.0,
            max_dense_working_bytes=max_dense_working_bytes,
            operation="dense pricing-dual assembly",
        )
    dual_weight = (
        np.zeros(A.shape, dtype=np.float64)
        if requires_dense
        else sparse.csr_matrix(A.shape, dtype=np.float64)
    )
    constant = 0.0

    for name, dual in duals.items():
        if dual is None:
            continue

        if sparse.issparse(dual):
            values = sparse.csr_matrix(dual, dtype=np.float64)
            if values.shape != A.shape:
                raise ValueError(
                    f"Two-dimensional dual {name!r} must have shape {A.shape}."
                )
            if not np.all(np.isfinite(values.data)):
                raise ValueError(f"Dual {name!r} contains NaN or infinity.")
            symmetric_values = 0.5 * (values + values.T)
            if sparse.issparse(dual_weight):
                dual_weight = dual_weight + symmetric_values
            else:
                dual_weight += symmetric_values.toarray()
        elif isinstance(dual, np.ndarray):
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
                if values.shape != A.shape:
                    raise ValueError(
                        f"Two-dimensional dual {name!r} must have shape "
                        f"{A.shape}."
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

    if sparse.issparse(dual_weight):
        dual_weight = sparse.csr_matrix(dual_weight)
        dual_weight.eliminate_zeros()
    return dual_weight, constant


def compute_reduced_cost(
    A: np.ndarray | sparse.spmatrix,
    a: np.ndarray,
    m: float,
    partition: np.ndarray | sparse.spmatrix,
    duals: Mapping[str, Any],
    *,
    gamma: float = 1.0,
    max_dense_working_bytes: int | None = DEFAULT_MAX_DENSE_WORKING_BYTES,
) -> float:
    """Evaluate a partition with Asunder's exact reduced-cost objective.

    Parameters
    ----------
    A : numpy.ndarray or scipy.sparse.spmatrix
        Original floating-point adjacency matrix.
    a : numpy.ndarray
        Degree/strength vector used by the modularity null model.
    m : float
        Total directed graph weight.
    partition : numpy.ndarray or scipy.sparse.spmatrix
        Candidate binary or fractional co-association matrix.
    duals : mapping[str, Any]
        Master dual values accepted by :func:`build_dual_weight_matrix`.
    gamma : float, default=1.0
        Modularity resolution parameter.
    max_dense_working_bytes : int or None, default=536870912
        Maximum estimated dense working set if dual assembly requires one.

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

    dual_weight, constant = build_dual_weight_matrix(
        A,
        duals,
        max_dense_working_bytes=max_dense_working_bytes,
    )
    objective = compute_f_star(A, a, m, partition, gamma=gamma)
    if sparse.issparse(dual_weight):
        dual_contribution = float(dual_weight.multiply(partition).sum())
    elif sparse.issparse(partition):
        dual_contribution = float(partition.multiply(dual_weight).sum())
    else:
        dual_contribution = float(np.sum(dual_weight * partition))
    return float(objective - dual_contribution - constant)
