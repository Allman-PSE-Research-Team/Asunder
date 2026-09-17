"""Dense and sparse matrix storage helpers."""

from __future__ import annotations

from typing import Literal

import numpy as np
from scipy import sparse

DEFAULT_SPARSE_COLUMN_DENSITY_THRESHOLD = 0.20
DEFAULT_MAX_DENSE_WORKING_BYTES = 512 * 1024**2

ColumnStorage = Literal["auto", "dense", "csr"]


def validate_storage_options(
    column_storage: str,
    sparse_column_density_threshold: float,
    max_dense_working_bytes: int | None,
) -> None:
    """Validate matrix-storage configuration.

    Parameters
    ----------
    column_storage : {"auto", "dense", "csr"}
        Requested representation for binary partition columns.
    sparse_column_density_threshold : float
        Maximum density at which automatic selection uses CSR storage.
    max_dense_working_bytes : int or None
        Maximum operation-specific estimate for package-created dense work
        arrays. ``None`` disables the guard.

    Raises
    ------
    ValueError
        If an option is outside its supported range.
    """
    if column_storage not in {"auto", "dense", "csr"}:
        raise ValueError("column_storage must be 'auto', 'dense', or 'csr'.")
    threshold = float(sparse_column_density_threshold)
    if not np.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("sparse_column_density_threshold must be between 0 and 1.")
    if max_dense_working_bytes is not None and int(max_dense_working_bytes) <= 0:
        raise ValueError("max_dense_working_bytes must be positive or None.")


def matrix_storage(matrix) -> str:
    """Return ``"csr"`` for sparse matrices and ``"dense"`` otherwise."""
    return "csr" if sparse.issparse(matrix) else "dense"


def matrix_density(matrix) -> float:
    """Return the fraction of logically stored nonzero matrix entries."""
    rows, columns = matrix.shape
    size = int(rows) * int(columns)
    if size == 0:
        return 0.0
    if sparse.issparse(matrix):
        return float(matrix.count_nonzero()) / float(size)
    return float(np.count_nonzero(np.asarray(matrix))) / float(size)


def choose_column_storage(
    matrix,
    *,
    column_storage: ColumnStorage,
    sparse_column_density_threshold: float,
) -> Literal["dense", "csr"]:
    """Resolve storage for an existing column, preserving its format in auto mode."""
    if column_storage != "auto":
        return column_storage
    return matrix_storage(matrix)


def estimate_dense_working_bytes(
    shape: tuple[int, ...],
    *,
    dtype=np.float64,
    working_arrays: float = 1.0,
) -> int:
    """Estimate the peak bytes needed by a dense matrix operation.

    Parameters
    ----------
    shape : tuple of int
        Shape of one dense array used by the operation.
    dtype : numpy dtype, optional
        Dense array dtype.
    working_arrays : float, default=1.0
        Estimated number of same-sized arrays simultaneously alive.

    Returns
    -------
    int
        Estimated peak working-set size in bytes.
    """
    entries = int(np.prod(shape, dtype=object))
    return int(np.ceil(entries * np.dtype(dtype).itemsize * float(working_arrays)))


def ensure_dense_working_set(
    shape: tuple[int, ...],
    *,
    dtype=np.float64,
    working_arrays: float = 1.0,
    max_dense_working_bytes: int | None,
    operation: str,
    extra_bytes: int = 0,
) -> int:
    """Reject a package-created dense workspace that exceeds its byte cap.

    Parameters
    ----------
    shape : tuple of int
        Shape of one dense array.
    dtype : numpy dtype, optional
        Array element type.
    working_arrays : float, default=1.0
        Estimated number of simultaneous arrays of this shape and dtype.
    max_dense_working_bytes : int or None
        Maximum estimate in bytes; ``None`` disables the guard.
    operation : str
        Human-readable operation name included in errors.
    extra_bytes : int, default=0
        Additional scratch space with a different shape or dtype.

    Returns
    -------
    int
        Estimated working-set bytes.

    Raises
    ------
    MemoryError
        If the estimate exceeds the configured cap.
    """
    required = int(extra_bytes) + estimate_dense_working_bytes(
        shape,
        dtype=dtype,
        working_arrays=working_arrays,
    )
    if (
        max_dense_working_bytes is not None
        and required > int(max_dense_working_bytes)
    ):
        required_mib = required / 1024**2
        limit_mib = int(max_dense_working_bytes) / 1024**2
        raise MemoryError(
            f"{operation} requires an estimated {required_mib:.1f} MiB dense "
            f"working set, exceeding max_dense_working_bytes={limit_mib:.1f} "
            f"MiB. Set max_dense_working_bytes>={required}, select a "
            "sparse-compatible backend, or set "
            "max_dense_working_bytes=None to disable the guard."
        )
    return required


def checked_to_dense(
    matrix,
    *,
    dtype=np.float64,
    working_arrays: float = 1.0,
    max_dense_working_bytes: int | None = DEFAULT_MAX_DENSE_WORKING_BYTES,
    operation: str = "dense matrix conversion",
):
    """Convert a matrix after checking the estimated dense working set.

    Matching-dtype dense inputs need no conversion allocation. A
    ``working_arrays`` estimate greater than one also describes downstream
    workspace and is checked for dense inputs. Sparse values are cast before
    densification so only the destination dense array is allocated.

    Parameters
    ----------
    matrix : numpy.ndarray or scipy.sparse.spmatrix
        Array to convert.
    dtype : numpy dtype, optional
        Output element type; defaults to float64.
    working_arrays : float, default=1.0
        Estimated simultaneous dense arrays, including downstream workspace.
    max_dense_working_bytes : int or None, default=536870912
        Estimated allocation limit; ``None`` disables the guard.
    operation : str, default="dense matrix conversion"
        Description used in allocation errors.

    Returns
    -------
    numpy.ndarray
        Dense values. Matching-dtype dense input is returned without copying.

    Raises
    ------
    MemoryError
        If conversion or additional workspaces exceed the configured cap.
    """
    if not sparse.issparse(matrix):
        values = np.asarray(matrix)
        if values.dtype != np.dtype(dtype) or working_arrays > 1:
            ensure_dense_working_set(
                values.shape,
                dtype=dtype,
                working_arrays=working_arrays,
                max_dense_working_bytes=max_dense_working_bytes,
                operation=operation,
            )
        return np.asarray(values, dtype=dtype)
    ensure_dense_working_set(
        matrix.shape,
        dtype=dtype,
        working_arrays=working_arrays,
        max_dense_working_bytes=max_dense_working_bytes,
        operation=operation,
    )
    return matrix.astype(dtype, copy=False).toarray()


def normalize_adjacency(A, *, dtype=np.float64):
    """Normalize adjacency input while preserving sparse storage."""
    if sparse.issparse(A):
        matrix = sparse.csr_matrix(A, dtype=dtype, copy=True)
        matrix.sum_duplicates()
        matrix.eliminate_zeros()
        matrix.sort_indices()
        return matrix
    return np.asarray(A, dtype=dtype)


def matrix_scalar(matrix, row: int, column: int) -> float:
    """Read one dense or sparse matrix entry as a Python float."""
    return float(matrix[int(row), int(column)])


def is_symmetric(matrix, *, atol: float = 1e-10) -> bool:
    """Return whether a dense or sparse matrix is symmetric within tolerance."""
    if sparse.issparse(matrix):
        difference = sparse.csr_matrix(matrix - matrix.T)
        difference.eliminate_zeros()
        return difference.nnz == 0 or bool(np.max(np.abs(difference.data)) <= atol)
    values = np.asarray(matrix)
    return all(np.allclose(row, values[:, index], atol=atol, rtol=0) for index, row in enumerate(values))
    # return bool(np.allclose(matrix, np.asarray(matrix).T, atol=atol, rtol=0)) # May be better for small/medium arrays.


def structural_edge_pairs(A) -> tuple[tuple[int, int], ...]:
    """Return nonzero strict-upper-triangle coordinates."""
    if sparse.issparse(A):
        upper = sparse.triu(A, k=1, format="coo")
        return tuple(
            sorted(
                (int(row), int(column))
                for row, column, value in zip(upper.row, upper.col, upper.data)
                if value != 0
            )
        )
    values = np.asarray(A)
    return tuple(
        (row, row + 1 + int(offset))
        for row in range(values.shape[0])
        for offset in np.flatnonzero(values[row, row + 1:])
    )
