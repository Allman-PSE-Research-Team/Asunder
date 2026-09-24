"""Matrix visualization helpers."""

from __future__ import annotations

import matplotlib.pyplot as plt

from asunder.base.utils.matrix import (
    DEFAULT_MAX_DENSE_WORKING_BYTES,
    checked_to_dense,
)
from asunder.types import MatrixLike

try:
    import seaborn as sns
except Exception:  # pragma: no cover - optional dependency
    sns = None


def visualize_adjacency_matrix(
    adj_matrix: MatrixLike,
    cmap: str = "viridis",
    show: bool = True,
    save_path: str | None = None,
    max_dense_working_bytes: int | None = DEFAULT_MAX_DENSE_WORKING_BYTES,
) -> None:
    """
    Display an adjacency matrix as an image.
    
    Parameters
    ----------
    adj_matrix : numpy.ndarray or scipy.sparse.csr_matrix, shape (N, N)
        Adjacency matrix. Sparse input crosses a guarded dense visualization
        boundary.
    cmap : str
        Color map to use.
    show : bool
        Show plot if `True`.
    save_path : str | None
        Path for saving image.
    max_dense_working_bytes : int or None, default=536870912
        Maximum operation-specific dense working-set estimate for sparse
        visualization input. ``None`` disables the guard.
    """
    matrix = checked_to_dense(
        adj_matrix,
        working_arrays=2.0,
        max_dense_working_bytes=max_dense_working_bytes,
        operation="adjacency-matrix visualization",
    )
    plt.imshow(matrix, interpolation="nearest", cmap=cmap)
    plt.colorbar(label="Edge weight")
    plt.xlabel("Node j")
    plt.ylabel("Node i")
    plt.title("Adjacency Matrix (A)")
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()


def visualize_partition_matrix(
    partition_matrix: MatrixLike,
    prefix: str = "",
    show: bool = True,
    save_path: str | None = None,
    use_seaborn: bool = True,
    max_dense_working_bytes: int | None = DEFAULT_MAX_DENSE_WORKING_BYTES,
) -> None:
    """
    Visualize a binary partition/co-association matrix.
    
    Parameters
    ----------
    partition_matrix : numpy.ndarray or scipy.sparse.csr_matrix, shape (N, N)
        Partition matrix. Sparse input crosses a guarded dense visualization
        boundary.
    prefix : str
        Prefix for visualization.
    show : bool
        Show plot if `True`.
    save_path : str | None
        Path for saving image.
    use_seaborn : bool
        Use `seaborn` if `True`.
    max_dense_working_bytes : int or None, default=536870912
        Maximum operation-specific dense working-set estimate for sparse
        visualization input. ``None`` disables the guard.
    """
    matrix = checked_to_dense(
        partition_matrix,
        working_arrays=2.0,
        max_dense_working_bytes=max_dense_working_bytes,
        operation="partition-matrix visualization",
    )
    if use_seaborn and sns is not None:
        sns.heatmap(matrix, cmap="gray")
    else:
        plt.imshow(matrix, cmap="gray", interpolation="nearest")

    plt.title(f"{prefix}Partition Matrix (z)")
    plt.xlabel("Node i")
    plt.ylabel("Node j")
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()
