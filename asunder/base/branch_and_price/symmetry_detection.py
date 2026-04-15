"""Automorphism based symmetry detection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Hashable, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    import scipy.sparse as sp  # optional
except Exception:
    sp = None


@dataclass
class OrbitsResult:

    """
    Symmetry-orbit data for the original graph vertices.

    Attributes
    ----------
    rep : numpy.ndarray
        Array where ``rep[i]`` is the representative vertex for the orbit
        containing vertex ``i``.
    orbits : list of list of int
        Orbit partition of the original vertices.
    generators : list of list of int
        Automorphism generators of the transformed graph.
    """
    rep: np.ndarray              # rep[i] = smallest vertex id in i's orbit
    orbits: List[List[int]]      # list of orbit vertex-lists (original vertices only)
    generators: List[List[int]]  # automorphism generators on the transformed graph


def _iter_upper_weighted_edges(A) -> Iterable[Tuple[int, int, float]]:
    """
    Yield nonzero weighted edges from the upper triangle of a matrix.

    Parameters
    ----------
    A : array-like or scipy.sparse matrix
        Square weighted adjacency matrix.

    Yields
    ------
    tuple of int, int, float
        Edge triplets ``(i, j, w)`` with ``i < j`` and ``A[i, j] != 0``.

    Raises
    ------
    ValueError
        If ``A`` is dense and not square.
    """
    if sp is not None and sp.issparse(A):
        coo = A.tocoo()
        for i, j, w in zip(coo.row, coo.col, coo.data):
            if i < j and w != 0:
                yield int(i), int(j), float(w)
        return

    A = np.asarray(A)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"A must be square; got shape={A.shape}")
    n = A.shape[0]
    for i in range(n - 1):
        row = A[i, i + 1 :]
        js = np.flatnonzero(row)
        for dj in js:
            j = i + 1 + int(dj)
            yield i, j, float(row[dj])


def weighted_constraint_orbits(
    A,
    *,
    vertex_colors: Optional[Sequence[int]] = None,
    weight_to_key: Optional[Callable[[float], Hashable]] = None,
    weight_round_tol: float = 1e-9,
    sh: str = "fl",
) -> OrbitsResult:
    """
    Compute exact vertex orbits of a weighted graph under color-preserving automorphisms.

    Parameters
    ----------
    A : array-like or scipy.sparse matrix
        Square weighted adjacency matrix. Nonzeros in the upper triangle define
        undirected edges.
    vertex_colors : sequence of int, optional
        Vertex color classes to preserve during the automorphism computation.
    weight_to_key : callable, optional
        Function mapping each edge weight to a hashable key used to distinguish
        weight classes. If omitted, near-integer weights are bucketed by
        integer value.
    weight_round_tol : float, optional
        Tolerance used when deciding whether a weight should be treated as an
        integer-valued key.
    sh : str, optional
        BLISS splitting heuristic passed to ``igraph.Graph.automorphism_group``.

    Returns
    -------
    OrbitsResult
        Orbit representatives, orbit lists, and automorphism generators.

    Notes
    -----
    Edge weights are preserved up to the chosen weight bucketing rule.

    Raises
    ------
    ImportError
        If ``python-igraph`` is not available.
    """
    try:
        import igraph as ig
    except Exception as e:
        raise ImportError("python-igraph is required for this implementation (pip install igraph).") from e

    # ---- parse edges + weights ----
    A_arr = A if (sp is not None and sp.issparse(A)) else np.asarray(A)
    n = int(A_arr.shape[0])

    if vertex_colors is None:
        vcols = [0] * n
    else:
        if len(vertex_colors) != n:
            raise ValueError("vertex_colors must have length N")
        vcols = list(map(int, vertex_colors))

    if weight_to_key is None:
        def weight_to_key(w: float) -> Hashable:
            r = int(round(w))
            if abs(w - r) <= weight_round_tol:
                return r
            return w  # exact float bucket (can kill symmetries if weights are noisy)

    edges_ijw: List[Tuple[int, int, Hashable]] = []
    for i, j, w in _iter_upper_weighted_edges(A_arr):
        edges_ijw.append((i, j, weight_to_key(w)))

    m = len(edges_ijw)

    # ---- map weight keys -> compact bucket ids ----
    uniq_keys = sorted({k for _, _, k in edges_ijw}, key=lambda x: (str(type(x)), x))
    key_to_bucket: Dict[Hashable, int] = {k: t for t, k in enumerate(uniq_keys)}

    # ---- build edge-subdivision graph (vertex-colored) ----
    # Original vertices: 0..n-1
    # Edge-vertices: n..n+m-1 with colors determined by weight bucket.
    nv = n + m
    edge_base_color = (max(vcols) + 1) if vcols else 1

    colors: List[int] = [0] * nv
    colors[:n] = vcols

    new_edges: List[Tuple[int, int]] = []
    for k, (i, j, wk) in enumerate(edges_ijw):
        evert = n + k
        colors[evert] = edge_base_color + key_to_bucket[wk]
        new_edges.append((i, evert))
        new_edges.append((evert, j))

    g = ig.Graph(n=nv, edges=new_edges, directed=False)

    # ---- automorphism generators (BLISS) ----
    gens = g.automorphism_group(sh=sh, color=colors)
    gens = [list(map(int, p)) for p in gens]  # each p is a permutation vector of length nv

    # ---- compute orbits on original vertices from generators ----
    parent = list(range(n))
    rank = [0] * n

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        if rank[ra] < rank[rb]:
            ra, rb = rb, ra
        parent[rb] = ra
        if rank[ra] == rank[rb]:
            rank[ra] += 1

    for p in gens:
        if len(p) != nv:
            raise ValueError("Unexpected generator length from igraph automorphism_group")
        for i in range(n):
            j = p[i]
            if j < n:
                union(i, j)

    groups: Dict[int, List[int]] = {}
    for i in range(n):
        r = find(i)
        groups.setdefault(r, []).append(i)

    orbits = [sorted(v) for v in groups.values()]
    orbits.sort(key=lambda xs: (len(xs), xs))

    rep = np.empty(n, dtype=int)
    for xs in orbits:
        r0 = xs[0]
        for i in xs:
            rep[i] = r0

    return OrbitsResult(rep=rep, orbits=orbits, generators=gens)