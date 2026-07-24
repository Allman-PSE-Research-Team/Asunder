"""Python interface to Asunder's bundled QMETIS native library."""

from __future__ import annotations

import json
import os
import platform
import sys
import warnings
from collections import defaultdict
from contextlib import ExitStack
from functools import lru_cache
from importlib import import_module
from importlib.resources import as_file, files
from typing import Any, Hashable, Sequence

import networkx as nx
import numpy as np

QMETIS_IDXTYPEWIDTH = 64
QMETIS_REALTYPEWIDTH = 32
QMETIS_MODULARITY_SCALE = 1_000_000

_RESOURCE_STACK = ExitStack()
_DLL_DIRECTORY_HANDLES: list[Any] = []


class QMETISApproximationWarning(UserWarning):
    """Warn that QMETIS candidate generation omits unsupported graph data."""


def _native_library_name() -> str:
    """Return the bundled QMETIS filename for the running operating system.

    Returns
    -------
    str
        ``libqmetis.so`` on Linux, ``qmetis.dll`` on Windows, or
        ``libqmetis.dylib`` on macOS.

    Raises
    ------
    RuntimeError
        If the running operating system has no supported QMETIS wheel.
    """

    match sys.platform:
        case "linux":
            return "libqmetis.so"
        case "win32":
            return "qmetis.dll"
        case "darwin":
            return "libqmetis.dylib"
        case _:
            raise RuntimeError(
                f"QMETIS is not available on platform {sys.platform!r} "
                f"({platform.machine()!r})."
            )


def _bundle_metadata() -> dict[str, Any]:
    """Read metadata describing the QMETIS binary staged in the wheel.

    Returns
    -------
    dict[str, Any]
        Parsed bundle metadata, or an empty dictionary when running from a
        source installation without a staged native library.
    """

    metadata = files("asunder.load_balancing.algorithms._native").joinpath(
        "_qmetis_bundle.json"
    )
    if not metadata.is_file():
        return {}
    with metadata.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def bundled_qmetis_release() -> str | None:
    """Return the release tag recorded for the bundled QMETIS binary.

    Returns
    -------
    str or None
        Immutable QMETIS release tag, or ``None`` when no bundle metadata is
        present.
    """

    release = _bundle_metadata().get("release_tag")
    return str(release) if release else None


@lru_cache(maxsize=1)
def _import_qmetis():
    """Import and cache the Python wrapper against Asunder's QMETIS library.

    The loader configures the wrapper's library path and integer/real ABI
    widths before importing :mod:`metis`.

    Returns
    -------
    module
        Imported :mod:`metis` wrapper bound to the bundled QMETIS binary.

    Raises
    ------
    ImportError
        If the platform wheel has no native binary or its ABI metadata does
        not match this loader.
    RuntimeError
        If :mod:`metis` was already imported against another native library.
    """

    native_package = "asunder.load_balancing.algorithms._native"
    metadata = _bundle_metadata()
    if metadata:
        bundled_widths = (
            int(metadata.get("idx_width", -1)),
            int(metadata.get("real_width", -1)),
        )
        expected_widths = (QMETIS_IDXTYPEWIDTH, QMETIS_REALTYPEWIDTH)
        if bundled_widths != expected_widths:
            raise ImportError(
                "Bundled QMETIS ABI metadata does not match the Python loader: "
                f"bundle={bundled_widths}, loader={expected_widths}."
            )

    library_resource = files(native_package).joinpath(_native_library_name())
    if not library_resource.is_file():
        raise ImportError(
            "This Asunder installation does not contain a QMETIS native "
            f"library for {sys.platform!r}/{platform.machine()!r}. Install a "
            "supported platform wheel from PyPI or build QMETIS for this "
            "platform."
        )

    library_path = _RESOURCE_STACK.enter_context(as_file(library_resource))
    native_dir = os.fspath(library_path.parent)

    if sys.platform == "win32" and hasattr(os, "add_dll_directory"):
        _DLL_DIRECTORY_HANDLES.append(os.add_dll_directory(native_dir))

    expected_dll = os.fspath(library_path)
    loaded = sys.modules.get("metis")
    configured_dll = os.environ.get("METIS_DLL")
    if loaded is not None and configured_dll != expected_dll:
        raise RuntimeError(
            "The metis wrapper was imported before Asunder configured its "
            "bundled QMETIS library. Import and use Asunder's QMETIS support "
            "before importing another METIS library in this process."
        )

    os.environ["METIS_DLL"] = expected_dll
    os.environ["METIS_IDXTYPEWIDTH"] = str(QMETIS_IDXTYPEWIDTH)
    os.environ["METIS_REALTYPEWIDTH"] = str(QMETIS_REALTYPEWIDTH)

    return import_module("metis")


def quantize_metis_weights(
    weights: np.ndarray,
    *,
    relative_resolution: float = 1e-7,
    safe_total: int = 1 << 50,
) -> tuple[np.ndarray, float]:
    """Quantize loop-free QMETIS edge weights within an ``idx_t`` budget.

    Already integer-valued weights are preserved when their directed total is
    within ``safe_total``. Otherwise, the scale preserves approximately
    ``relative_resolution`` of the largest edge while respecting that budget.
    The released ``idx64-real32`` ABI motivates keeping the total well inside
    signed 64-bit range and double's exact-integer range.

    Parameters
    ----------
    weights : numpy.ndarray
        Square, finite, nonnegative edge-weight matrix. Asymmetric entries are
        averaged. Nonzero diagonal entries are removed because the QMETIS
        adapter cannot represent self-loops or contraction-generated internal
        mass.
    relative_resolution : float, default=1e-7
        Target resolution relative to the largest positive off-diagonal edge.
    safe_total : int, default=2**50
        Maximum sum of the directed quantized adjacency weights.

    Returns
    -------
    quantized : numpy.ndarray
        Symmetric ``int64`` matrix with a zero diagonal.
    scale : float
        Multiplicative scale applied before rounding.

    Warns
    -----
    QMETISApproximationWarning
        If a nonzero diagonal is discarded. For contracted graphs this means
        candidate generation is approximate, although callers can still
        rescore the candidate with the exact objective.

    Raises
    ------
    ValueError
        If the matrix, resolution, or safety budget is invalid; if no positive
        off-diagonal edge remains; or if quantization loses excessive mass.
    OverflowError
        If an individual edge or accumulated adjacency exceeds its integer
        safety limit.
    """

    matrix = np.asarray(weights, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("weights must be a square matrix.")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("weights contain NaN or infinity.")
    if relative_resolution <= 0 or not np.isfinite(relative_resolution):
        raise ValueError("relative_resolution must be a positive finite value.")
    if safe_total <= 0 or safe_total > np.iinfo(np.int64).max:
        raise ValueError("safe_total must be within the positive int64 range.")

    if np.any(matrix < 0):
        raise ValueError("QMETIS edge weights must be nonnegative.")
    matrix = 0.5 * (matrix + matrix.T)
    if np.any(np.diag(matrix) != 0):
        warnings.warn(
            "QMETIS ignores nonzero diagonal weights. If they encode "
            "contracted internal-edge mass, candidate generation is "
            "approximate; rescore the partition with the original graph.",
            QMETISApproximationWarning,
            stacklevel=2,
        )
    np.fill_diagonal(matrix, 0.0)

    rows, columns = np.triu_indices_from(matrix, k=1)
    edge_weights = matrix[rows, columns]
    positive = edge_weights > 0
    if not np.any(positive):
        raise ValueError("At least one positive edge weight is required.")

    positive_weights = edge_weights[positive]
    max_weight = float(positive_weights.max())
    directed_total = 2.0 * float(positive_weights.sum(dtype=np.float64))
    if max_weight < 2**63 and np.all(edge_weights == np.rint(edge_weights)):
        integer_edges = edge_weights.astype(np.int64)
        integer_total = 2 * sum(map(int, integer_edges[positive]))
        if integer_total <= safe_total:
            quantized = np.zeros(matrix.shape, dtype=np.int64)
            quantized[rows, columns] = integer_edges
            quantized[columns, rows] = integer_edges
            return quantized, 1.0

    precision_scale = 1.0 / (relative_resolution * max_weight)
    capacity_scale = float(safe_total) / directed_total
    scale = min(precision_scale, capacity_scale)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Could not determine a safe QMETIS weight scale.")

    rounded = np.rint(edge_weights * scale)
    if float(rounded.max(initial=0.0)) > np.iinfo(np.int64).max:
        raise OverflowError("A quantized edge weight exceeds int64.")

    quantized = np.zeros(matrix.shape, dtype=np.int64)
    quantized[rows, columns] = rounded.astype(np.int64)
    quantized[columns, rows] = quantized[rows, columns]

    quantized_total = int(quantized.sum(dtype=np.int64))
    if quantized_total > safe_total:
        raise OverflowError(
            f"Quantized adjacency total {quantized_total} exceeds the "
            f"safety budget {safe_total}."
        )

    lost = positive & (rounded == 0)
    if np.any(lost):
        lost_fraction = float(edge_weights[lost].sum()) / float(
            positive_weights.sum()
        )
        if lost_fraction > relative_resolution:
            raise ValueError(
                "Quantization would remove too much positive edge weight "
                f"({lost_fraction:.3g} of the total)."
            )

    return quantized, scale


def _integer_weight_graph(adjacency: np.ndarray) -> nx.Graph:
    """Build a loop-free NetworkX graph from an integer adjacency matrix.

    Parameters
    ----------
    adjacency : numpy.ndarray
        Square integer adjacency matrix. Only nonzero upper-triangular entries
        are emitted as undirected edges.

    Returns
    -------
    networkx.Graph
        Graph containing every matrix row as a node and no self-loops.
    """

    graph = nx.Graph()
    graph.add_nodes_from(range(adjacency.shape[0]))
    rows, columns = np.triu_indices_from(adjacency, k=1)
    graph.add_weighted_edges_from(
        (
            int(row),
            int(column),
            int(adjacency[row, column]),
        )
        for row, column in zip(rows, columns)
        if adjacency[row, column] != 0
    )
    return graph


def to_stable_weighted_adjlist(
    H: nx.Graph,
    edge_weight_attr: str | None = None,
    node_weight_attr: str | Sequence[str] | None = None,
):
    """Convert a graph to the deterministic adjacency format used by QMETIS.

    Parameters
    ----------
    H : networkx.Graph
        Graph whose node insertion order defines the integer node mapping.
    edge_weight_attr : str or None
        Edge attribute to emit as an integer weight. If ``None``, emit
        unweighted neighbor indices.
    node_weight_attr : str, sequence of str, or None
        One node-balance attribute or multiple constraint attributes.

    Returns
    -------
    nodes : list
        Nodes in the stable order used by the adjacency representation.
    adjacency : list[list]
        Neighbor indices, optionally paired with integer edge weights.
    node_weights : list or None
        Integer node weights in matching order, or ``None``.
    """

    nodes = list(H.nodes())
    index = {node: position for position, node in enumerate(nodes)}

    adjacency = []
    for node in nodes:
        entries = []
        for neighbor in sorted(H.neighbors(node), key=index.__getitem__):
            neighbor_index = index[neighbor]
            if edge_weight_attr is None:
                entries.append(neighbor_index)
            else:
                weight = int(H[node][neighbor].get(edge_weight_attr, 1))
                entries.append((neighbor_index, weight))
        adjacency.append(entries)

    node_weights = None
    if node_weight_attr is not None:
        if isinstance(node_weight_attr, str):
            node_weights = [
                int(H.nodes[node].get(node_weight_attr, 1)) for node in nodes
            ]
        else:
            attributes = list(node_weight_attr)
            node_weights = [
                tuple(int(H.nodes[node].get(attribute, 1)) for attribute in attributes)
                for node in nodes
            ]

    return nodes, adjacency, node_weights


def _validate_integer_weights(
    G: nx.Graph,
    node_weight_attr: str | Sequence[str] | None,
    edge_weight_attr: str | None,
) -> None:
    """Validate integer node and edge weights before calling QMETIS.

    Parameters
    ----------
    G : networkx.Graph
        Graph containing the attributes to validate.
    node_weight_attr : str, sequence of str, or None
        Node attributes interpreted as balance weights.
    edge_weight_attr : str or None
        Edge attribute interpreted as the QMETIS edge weight.

    Raises
    ------
    ValueError
        If a selected weight is negative or not an integer.
    """

    node_attributes: list[str]
    if node_weight_attr is None:
        node_attributes = []
    elif isinstance(node_weight_attr, str):
        node_attributes = [node_weight_attr]
    else:
        node_attributes = list(node_weight_attr)

    for node, data in G.nodes(data=True):
        for attribute in node_attributes:
            value = data.get(attribute, 1)
            if not isinstance(value, (int, np.integer)) or value < 0:
                raise ValueError(
                    f"Node weight {attribute!r} for node {node!r} must be "
                    "a nonnegative integer."
                )

    if edge_weight_attr is not None:
        for first, second, data in G.edges(data=True):
            value = data.get(edge_weight_attr, 1)
            if not isinstance(value, (int, np.integer)) or value < 0:
                raise ValueError(
                    f"Edge weight {edge_weight_attr!r} for edge "
                    f"{(first, second)!r} must be a nonnegative integer."
                )


def _epsilon_to_ubvec(
    balance_epsilon: float | Sequence[float] | None,
    node_weight_attr: str | Sequence[str] | None,
) -> list[float] | None:
    """Translate imbalance epsilon values into QMETIS ``ubvec`` factors.

    Parameters
    ----------
    balance_epsilon : float, sequence of float, or None
        Nonnegative relative imbalance tolerance. ``0`` requests an upper
        factor of exactly ``1``.
    node_weight_attr : str, sequence of str, or None
        Node-weight constraints used to validate a multi-constraint vector.

    Returns
    -------
    list[float] or None
        QMETIS upper-balance factors ``1 + epsilon``, or ``None``.

    Raises
    ------
    ValueError
        If an epsilon is negative or its vector length does not match the
        number of node-weight constraints.
    """

    if balance_epsilon is None:
        return None

    if isinstance(balance_epsilon, (int, float)):
        if balance_epsilon < 0:
            raise ValueError(
                "balance_epsilon must be nonnegative, e.g. 0.03 for 3%."
            )
        return [1.0 + float(balance_epsilon)]

    ubvec = []
    for epsilon in balance_epsilon:
        if epsilon < 0:
            raise ValueError("Each balance epsilon must be nonnegative.")
        ubvec.append(1.0 + float(epsilon))

    if isinstance(node_weight_attr, Sequence) and not isinstance(
        node_weight_attr, str
    ):
        if len(ubvec) != len(node_weight_attr):
            raise ValueError(
                "For multi-constraint weights, one epsilon is needed per weight."
            )

    return ubvec


def qmetis_load_balanced_partition(
    G: nx.Graph,
    nparts: int,
    balance_epsilon: float | Sequence[float] | None = 0.03,
    node_weight_attr: str | Sequence[str] | None = None,
    edge_weight_attr: str | None = None,
    recursive: bool = False,
    contig: bool = False,
    seed: int | None = None,
    **metis_options: Any,
) -> dict[str, Any]:
    """Partition an integer-weighted NetworkX graph with bundled QMETIS.

    Parameters
    ----------
    G : networkx.Graph
        Loop-free graph to partition.
    nparts : int
        Requested number of nonempty parts.
    balance_epsilon : float, sequence of float, or None
        Relative upper imbalance tolerance for each balance constraint.
    node_weight_attr : str, sequence of str, or None
        Node attributes used only for QMETIS load-balancing constraints.
    edge_weight_attr : str or None
        Integer edge-weight attribute used by the modularity objective.
    recursive : bool, default=False
        Use recursive partitioning instead of direct k-way partitioning.
    contig : bool, default=False
        Request contiguous parts from QMETIS.
    seed : int or None
        QMETIS random seed.
    **metis_options : Any
        Additional options accepted by :func:`metis.part_graph`.

    Returns
    -------
    dict[str, Any]
        Native objective, part IDs, node-to-part mapping, grouped nodes, and
        the effective upper-balance vector.

    Raises
    ------
    ValueError
        If the requested part count or graph weights are invalid.
    ImportError
        If the installed package has no compatible QMETIS binary.
    """

    if nparts < 2:
        raise ValueError("nparts must be at least 2.")
    if G.number_of_nodes() == 0:
        raise ValueError("G must contain at least one node.")
    if nparts > G.number_of_nodes():
        raise ValueError("nparts cannot exceed the number of graph nodes.")

    graph = nx.Graph(G) if G.is_directed() else G.copy()
    nodes, adjacency, node_weights = to_stable_weighted_adjlist(
        graph,
        edge_weight_attr=edge_weight_attr,
        node_weight_attr=node_weight_attr,
    )
    _validate_integer_weights(graph, node_weight_attr, edge_weight_attr)

    qmetis = _import_qmetis()
    ubvec = _epsilon_to_ubvec(balance_epsilon, node_weight_attr)
    options = dict(metis_options)
    if contig:
        options["contig"] = True
    if seed is not None:
        options["seed"] = seed

    objective, part_ids = qmetis.part_graph(
        adjacency,
        nparts=nparts,
        ubvec=ubvec,
        recursive=recursive,
        nodew=node_weights,
        **options,
    )

    node_to_part = dict(zip(nodes, part_ids))
    parts_to_nodes: dict[int, list[Hashable]] = defaultdict(list)
    for node, part in node_to_part.items():
        parts_to_nodes[part].append(node)

    return {
        "obj_val": objective,
        "partition": part_ids,
        "node_to_part": node_to_part,
        "parts_to_nodes": dict(sorted(parts_to_nodes.items())),
        "ubvec": ubvec,
    }


def run_qmetis(
    modified_A: np.ndarray,
    K: int,
    balance_epsilon: float | Sequence[float] | None = None,
    node_weight_attr: str | Sequence[str] | None = None,
    edge_weight_attr: str | None = "weight",
    seed: int | None = None,
    relative_resolution: float = 1e-7,
    safe_total: int = 1 << 50,
    **metis_options: Any,
) -> tuple[np.ndarray, float]:
    """Partition a nonnegative matrix using bundled modularity QMETIS.

    Parameters
    ----------
    modified_A : numpy.ndarray
        Square edge-weight matrix supplied to the QMETIS heuristic.
        Nonzero diagonal entries are discarded with
        :class:`QMETISApproximationWarning`.
    K : int
        Number of requested parts.
    balance_epsilon : float, sequence of float, or None
        Relative upper imbalance tolerance.
    node_weight_attr : str, sequence of str, or None
        Optional node-balance attributes.
    edge_weight_attr : str or None, default="weight"
        Integer graph edge attribute passed to QMETIS.
    seed : int or None
        QMETIS random seed.
    relative_resolution : float, default=1e-7
        Relative precision target used during integer quantization.
    safe_total : int, default=2**50
        Maximum directed adjacency sum after quantization.
    **metis_options : Any
        Additional options forwarded to QMETIS.

    Returns
    -------
    partition : numpy.ndarray
        Binary co-association matrix for the generated partition.
    modularity : float
        Native QMETIS modularity after undoing its fixed return-value scale.

    Warns
    -----
    QMETISApproximationWarning
        If nonzero diagonal weights are omitted.
    """

    quantized, _scale = quantize_metis_weights(
        modified_A,
        relative_resolution=relative_resolution,
        safe_total=safe_total,
    )
    graph = _integer_weight_graph(quantized)
    result = qmetis_load_balanced_partition(
        graph,
        nparts=K,
        balance_epsilon=balance_epsilon,
        node_weight_attr=node_weight_attr,
        edge_weight_attr=edge_weight_attr,
        seed=seed,
        **metis_options,
    )

    labels = result["partition"]
    partition = np.equal.outer(labels, labels).astype(int)
    modularity = result["obj_val"] / QMETIS_MODULARITY_SCALE
    return partition, modularity
