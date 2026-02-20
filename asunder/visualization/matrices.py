"""Matrix visualization helpers."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np

try:
    import seaborn as sns
except Exception:  # pragma: no cover - optional dependency
    sns = None


def visualize_adjacency_matrix(
    adj_matrix: Any,
    cmap: str = "viridis",
    show: bool = True,
    save_path: str | None = None,
) -> None:
    """Display an adjacency matrix as an image."""
    matrix = np.asarray(adj_matrix)
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
    partition_matrix: Any,
    prefix: str = "",
    show: bool = True,
    save_path: str | None = None,
    use_seaborn: bool = True,
) -> None:
    """Visualize a binary partition/co-association matrix."""
    matrix = np.asarray(partition_matrix)
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
