"""Initial feasible partition generators and helpers."""

from __future__ import annotations

from typing import Any, Dict, Hashable, List, Optional, Sequence, Tuple

import networkx as nx
import numpy as np

from .graph import partition_vector_to_2d_matrix


def _build_components_links_only(
    N: int,
    must_link: Sequence[Tuple[int, int]],
    cannot_link: Sequence[Tuple[int, int]],
) -> Optional[Dict[str, Any]]:
    """
    Internal helper for building components given pairwise constraints.
    
    Parameters
    ----------
    N : int
        Number of nodes.
    must_link : sequence of tuple of int, optional
        Pairwise must-link constraints.
    cannot_link : sequence of tuple of int, optional
        Pairwise cannot-link constraints.
    
    Returns
    -------
    Optional[Dict[str, Any]]
        Computed result.
    """
    parent = np.arange(N, dtype=int)

    def find(x: int) -> int:
        """
        Find.
        """
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        """
        Union.
        """
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i, j in must_link:
        i, j = int(i), int(j)
        if i != j:
            union(i, j)

    reps = np.array([find(i) for i in range(N)], dtype=int)
    _, cid = np.unique(reps, return_inverse=True)  # cid[node] -> component id
    C = int(cid.max() + 1) if N else 0

    comps: List[List[int]] = [[] for _ in range(C)]
    for i in range(N):
        comps[int(cid[i])].append(i)

    # Infeasible if cannot-link inside a must-link component
    for i, j in cannot_link:
        i, j = int(i), int(j)
        if i == j or int(cid[i]) == int(cid[j]):
            return None

    # Component-level conflict graph from cannot-links
    forb: List[set] = [set() for _ in range(C)]
    for i, j in cannot_link:
        a, b = int(cid[int(i)]), int(cid[int(j)])
        if a == b:
            continue
        forb[a].add(b)
        forb[b].add(a)

    return {"C": C, "cid": cid, "comps": comps, "forb": forb}

def _component_order_from_node_order(order_idx: List[int], cid: np.ndarray, C: int) -> List[int]:
       
    """
    Internal helper for building a component visitation order induced by a node 
    visitation order.

    Parameters
    ----------
    order_idx : List[int]
        Node indices in the desired traversal order.
    cid : np.ndarray
        Component ID for each node; `cid[i]` gives the component containing node `i`.
    C : int
        Total number of components.

    Returns
    -------
    comp_order: List[int]
        Component IDs in the order they first appear in `order_idx`, followed by any
        components not encountered (in increasing component ID).
    """
    seen = set()
    comp_order = []
    for v in order_idx:
        c = int(cid[int(v)])
        if c not in seen:
            seen.add(c)
            comp_order.append(c)
    if len(comp_order) < C:
        for c in range(C):
            if c not in seen:
                comp_order.append(c)
    return comp_order

def _greedy_color_fixed_K(
    comp_order: List[int],
    forb: List[set],
    K_used: int,
    rng: np.random.Generator,
    jitter_top: int = 1,
) -> Optional[np.ndarray]:
    """
    Internal helper for greedy coloring given a fixed K.
    
    Parameters
    ----------
    comp_order : List[int]
        Ordered list of must-link component IDs to assign (e.g., in the sequence 
        they will be colored/placed).
    forb : List[set]
        Component-level cannot-link adjacency; `forb[c]` is the set of component IDs 
        that component `c` is forbidden to share a group with.
    K_used : int
        Number of communities.
    rng : np.random.Generator
        Random number generator.
    jitter_top : int
        If `0` or `1`: no real randomness (always pick the single best).
        If `2`: randomly choose between the top `2` candidates.
        Larger values → more diversity across runs but can limit freedom when `K` is fixed.
    
    Returns
    -------
    comp2g : Optional[np.ndarray]
        Component-to-group assignment vector of length `C`. Entry `comp2g[c]` is the 
        group/color ID assigned to component `c`.
    """
    C = len(forb)
    comp2g = -np.ones(C, dtype=int)
    in_group: List[set] = [set() for _ in range(K_used)]

    for c in comp_order:
        feas = [g for g in range(K_used) if not (forb[c] & in_group[g])]
        if not feas:
            return None

        feas.sort(key=lambda g: (len(in_group[g]), g))  # mild spread
        if jitter_top and len(feas) > 1:
            g = int(rng.choice(feas[: min(jitter_top, len(feas))]))
        else:
            g = int(feas[0])

        comp2g[c] = g
        in_group[g].add(c)

    return comp2g

def _greedy_color_unbounded_K(
    comp_order: List[int],
    forb: List[set],
) -> np.ndarray:
    """
    Internal helper for greedy coloring given an unbounded K.
    
    Parameters
    ----------
    comp_order : List[int]
        Ordered list of must-link component IDs to assign (e.g., in the sequence 
        they will be colored/placed).
    forb : List[set]
        Component-level cannot-link adjacency; `forb[c]` is the set of component IDs 
        that component `c` is forbidden to share a group with.
    
    Returns
    -------
    comp2g : Optional[np.ndarray]
        Component-to-group assignment vector of length `C`. Entry `comp2g[c]` is the 
        group/color ID assigned to component `c`.
    """
    C = len(forb)
    comp2g = -np.ones(C, dtype=int)
    in_group: List[set] = []

    for c in comp_order:
        placed = False
        for g in range(len(in_group)):
            if not (forb[c] & in_group[g]):
                comp2g[c] = g
                in_group[g].add(c)
                placed = True
                break
        if not placed:
            comp2g[c] = len(in_group)
            in_group.append({c})

    return comp2g

def assign_from_order_with_links_links_only(
    order_idx: List[int],
    N: int,
    K: Optional[int] = None,
    must_link: Optional[Sequence[Tuple[int, int]]] = None,
    cannot_link: Optional[Sequence[Tuple[int, int]]] = None,
    max_K_increase: int = 50,
    max_restarts: int = 20,
    jitter_top: int = 2,
    seed: int = 0,
) -> Optional[Tuple[np.ndarray, Dict[str, int]]]:
    """
    Returns a partition given an order, pairwise constraints and other parameters.
    
    Parameters
    ----------
    order_idx : List[int]
        Input parameter.
    N : int
        Number of nodes.
    K : int or None, optional
        Target number of communities.
    must_link : sequence of tuple of int, optional
        Pairwise must-link constraints.
    cannot_link : sequence of tuple of int, optional
        Pairwise cannot-link constraints.
    max_K_increase : int, optional
        Maximum allowed increase above the requested `K` when additional
        partitions are needed to satisfy the cannot-link structure. If
        `0`, the routine enforces the requested `K` exactly.
    max_restarts : int
        maximum number of restarts.
    jitter_top : int
        If `0` or `1`: no real randomness (always pick the single best).
        If `2`: randomly choose between the top `2` candidates.
        Larger values → more diversity across runs but can limit freedom when `K` is fixed.
    seed : int, optional
        Random seed used to initialize the sampling procedure.

    Returns
    -------
    Optional[Tuple[np.ndarray, Dict[str, int]]]
        Partition vector and metadata. 
    """
    must_link = [] if must_link is None else list(must_link)
    cannot_link = [] if cannot_link is None else list(cannot_link)

    if N == 0:
        return np.zeros(0, dtype=int), {"K_used": 0, "C": 0}

    comp = _build_components_links_only(N, must_link, cannot_link)
    if comp is None:
        return None

    C = int(comp["C"])
    cid = comp["cid"]
    comps = comp["comps"]
    forb = comp["forb"]

    if C == 0:
        return np.zeros(N, dtype=int), {"K_used": 0, "C": 0}

    comp_order_base = _component_order_from_node_order(order_idx, cid, C)

    def build_gvec(comp2g: np.ndarray) -> np.ndarray:
        """
        Build gvec.
        """
        gvec = np.empty(N, dtype=int)
        for c in range(C):
            gvec[comps[c]] = int(comp2g[c])
        return gvec

    if K is None:
        comp2g = _greedy_color_unbounded_K(comp_order_base, forb)
        gvec = build_gvec(comp2g)
        return gvec, {"K_used": int(comp2g.max() + 1), "C": C}

    rng = np.random.default_rng(seed)
    K = int(K)
    K_hi = min(C, K + int(max_K_increase))
    for K_used in range(max(1, K), max(1, K_hi) + 1):
        for r in range(max_restarts):
            comp_order = list(comp_order_base)
            if r > 0:
                # light shuffle to escape bad early choices
                rng.shuffle(comp_order)
            comp2g = _greedy_color_fixed_K(comp_order, forb, K_used, rng, jitter_top=jitter_top)
            if comp2g is not None:
                return build_gvec(comp2g), {"K_used": K_used, "C": C}

    return None

def _pairs_to_indices(
    pairs: Optional[Sequence[Tuple[Hashable, Hashable]]],
    node_to_idx: Dict[Hashable, int],
) -> Optional[List[Tuple[int, int]]]:
    """
    Internal helper for mapping the names of node pairs to indices.
    
    Parameters
    ----------
    pairs : Optional[Sequence[Tuple[Hashable, Hashable]]]
        Sequence of node pairs.
    node_to_idx : Dict[Hashable, int]
        Node to index map.
    
    Returns
    -------
    Optional[List[Tuple[int, int]]]
        Sequence of node pairs with integer identifiers.
    """
    if pairs is None:
        return None
    out: List[Tuple[int, int]] = []
    for a, b in pairs:
        ia = node_to_idx[a] if a in node_to_idx else int(a)
        ib = node_to_idx[b] if b in node_to_idx else int(b)
        out.append((int(ia), int(ib)))
    return out

def make_partitions_links_only(
    G,
    K: Optional[int] = None,
    must_link=None,
    cannot_link=None,
    n_cols: int = 15,
    seed: int = 42,
    nodes=None,
    max_K_increase: int = 50,
):
    """
    Generates partly ordered feasible partitions subject to only pairwise link constraints.

    Parameters
    ----------
    G : nx.Graph
        NetworkX graph.
    K : int or None, optional
        Target number of communities.
    must_link : sequence of tuple of int, optional
        Pairwise must-link constraints.
    cannot_link : sequence of tuple of int, optional
        Pairwise cannot-link constraints.
    n_cols : int, optional
        Number of columns.
    seed : int, optional
        Random seed used to initialize the sampling procedure.
    nodes : array-like, optional
        List of Graph nodes.
    max_K_increase : int, optional
        Maximum allowed increase above the requested `K`` when additional
        partitions are needed to satisfy the cannot-link structure. If
        ``0``, the routine enforces the requested ``K`` exactly.
    
    Returns
    -------
    Z_star: list(array)
        List of generated partitions.
    info: list(dict)
        List of partition metadata.
    nodes : array-like, optional
        List of Graph nodes.
    """
    rng = np.random.default_rng(seed)

    if nodes is None:
        nodes = sorted(G.nodes())
    N = len(nodes)
    node_to_idx = {u: i for i, u in enumerate(nodes)}

    must_idx = _pairs_to_indices(must_link, node_to_idx)
    cannot_idx = _pairs_to_indices(cannot_link, node_to_idx)

    def build_from_label_order(name: str, order_labels: List[Any]):
        """
        Build partition from label order.
        """
        order_idx = [node_to_idx[u] for u in order_labels]

        out = assign_from_order_with_links_links_only(
            order_idx=order_idx,
            N=N,
            K=K,
            must_link=must_idx,
            cannot_link=cannot_idx,
            max_K_increase=max_K_increase,
            seed=seed,
        )
        if out is None:
            return None

        g, meta = out
        Z = partition_vector_to_2d_matrix(g)  # defined elsewhere
        return Z, {"name": name, "order_labels": order_labels, "g": g, **meta}

    cols = []
    seen = set()

    # 1) spectral (Fiedler)
    if N <= 1:
        f = np.zeros(N)
    else:
        L = nx.laplacian_matrix(G, nodelist=nodes).toarray()
        w, v = np.linalg.eigh(L)
        f = v[:, 1]
    order = [nodes[i] for i in np.argsort(f)]
    out = build_from_label_order("spectral_fiedler", order)
    if out is not None:
        Z, meta = out
        key = meta["g"].tobytes()
        cols.append((Z, meta))
        seen.add(key)

    # 2) BFS from max-degree node
    start = max(nodes, key=lambda u: G.degree(u)) if N else None
    if start is not None:
        order = list(nx.bfs_tree(G, start).nodes())
        visited = set(order)
        remaining = [u for u in nodes if u not in visited]
        remaining.sort(key=lambda u: G.degree(u), reverse=True)
        for s in remaining:
            for u in nx.bfs_tree(G, s).nodes():
                if u not in visited:
                    visited.add(u)
                    order.append(u)
        out = build_from_label_order("bfs_from_max_degree", order)
        if out is not None:
            Z, meta = out
            key = meta["g"].tobytes()
            if key not in seen:
                cols.append((Z, meta))
                seen.add(key)

    # 3) DFS
    if start is not None:
        order = list(nx.dfs_preorder_nodes(G, start))
        visited = set(order)
        remaining = [u for u in nodes if u not in visited]
        remaining.sort(key=lambda u: G.degree(u), reverse=True)
        for s in remaining:
            for u in nx.dfs_preorder_nodes(G, s):
                if u not in visited:
                    visited.add(u)
                    order.append(u)
        out = build_from_label_order("dfs_from_max_degree", order)
        if out is not None:
            Z, meta = out
            key = meta["g"].tobytes()
            if key not in seen:
                cols.append((Z, meta))
                seen.add(key)

    # 4) degree sorted
    order = sorted(nodes, key=lambda u: (G.degree(u), u), reverse=True)
    out = build_from_label_order("degree_sorted", order)
    if out is not None:
        Z, meta = out
        key = meta["g"].tobytes()
        if key not in seen:
            cols.append((Z, meta))
            seen.add(key)

    # 5) noisy spectral fill
    noise_std = 1e-6 if N else 0.0
    tries = 0
    while len(cols) < n_cols and tries < 20 * n_cols:
        tries += 1
        f_noisy = f + rng.normal(0.0, noise_std, size=f.shape)
        order = [nodes[i] for i in np.argsort(f_noisy)]
        out = build_from_label_order(f"spectral_noisy_{tries}", order)
        if out is None:
            continue
        Z, meta = out
        key = meta["g"].tobytes()
        if key not in seen:
            cols.append((Z, meta))
            seen.add(key)

    Z_star = [Z for Z, _ in cols]
    info = [meta for _, meta in cols]
    return Z_star, info, nodes

def make_partitions_random_links_only(
    N: int,
    K: Optional[int] = None,
    must_link=None,
    cannot_link=None,
    seed: int = 42,
    return_Z: bool = True,
    max_K_increase: int = 50,
    n_parts: int = 10,
):
    """
    Generate random feasible partitions subject only to pairwise link constraints.
    
    Parameters
    ----------
    N : int
        Number of nodes.
    K : int or None, optional
        Target number of communities.
    must_link : sequence of tuple of int, optional
        Pairwise must-link constraints.
    cannot_link : sequence of tuple of int, optional
        Pairwise cannot-link constraints.
    seed : int, optional
        Random seed used to initialize the sampling procedure.
    return_Z : bool, optional
        Boolean that determines if the ``(N, N)`` partition matrix ``Z`` is returned or not.
    max_K_increase : int, optional
        Maximum allowed increase above the requested ``K`` when additional
        partitions are needed to satisfy the cannot-link structure. If
        ``0``, the routine enforces the requested ``K`` exactly.
    n_parts : int, optional
        Number of feasible random partitions to generate.
    
    Returns
    -------
    list
        List of feasible random partitions.

        If ``return_Z`` is ``True``, each element is a partition assignment
        matrix, typically of shape ``(N, N)``.

        If ``return_Z`` is ``False``, each element is a dict with keys that refer
        to the name of the partition, a 1D representation of the partition and the number of communities in it.
    """
    rng = np.random.default_rng(seed)
    must_link = [] if must_link is None else list(must_link)
    cannot_link = [] if cannot_link is None else list(cannot_link)

    if N == 0:
        out = [partition_vector_to_2d_matrix(np.zeros(0, dtype=int))] if return_Z else []
        return out

    comp = _build_components_links_only(N, must_link, cannot_link)
    if comp is None:
        return [] if return_Z else []

    C = int(comp["C"])
    comps = comp["comps"]
    forb = comp["forb"]

    def build_gvec(comp2g: np.ndarray) -> np.ndarray:
        """
        Build gvec.
        """
        gvec = np.empty(N, dtype=int)
        for c in range(C):
            gvec[comps[c]] = int(comp2g[c])
        return gvec

    parts_g = []
    seen = set()

    if K is None:
        for _ in range(n_parts * 3):
            comp_order = list(range(C))
            rng.shuffle(comp_order)
            comp2g = _greedy_color_unbounded_K(comp_order, forb)
            g = build_gvec(comp2g)
            key = g.tobytes()
            if key not in seen:
                seen.add(key)
                parts_g.append(("shuffle_unbounded", g, int(comp2g.max() + 1)))
                if len(parts_g) >= n_parts:
                    break
    else:
        K = int(K)
        K_hi = min(C, K + int(max_K_increase))
        for K_used in range(max(1, K), max(1, K_hi) + 1):
            for _ in range(n_parts * 5):
                comp_order = list(range(C))
                rng.shuffle(comp_order)
                comp2g = _greedy_color_fixed_K(comp_order, forb, K_used, rng, jitter_top=2)
                if comp2g is None:
                    continue
                g = build_gvec(comp2g)
                key = g.tobytes()
                if key not in seen:
                    seen.add(key)
                    parts_g.append(("shuffle_fixedK", g, K_used))
                    if len(parts_g) >= n_parts:
                        break
            if len(parts_g) >= n_parts:
                break

        if not parts_g:
            # Last-ditch fallback: try random restarts through existing constructor.
            for _ in range(50):
                order_idx = np.arange(N)
                rng.shuffle(order_idx)
                out = assign_from_order_with_links_links_only(
                    order_idx=order_idx.tolist(),
                    N=N,
                    K=K,
                    must_link=must_link,
                    cannot_link=cannot_link,
                    max_K_increase=max_K_increase,
                    seed=int(rng.integers(0, 10_000_000)),
                )
                if out is not None:
                    g_last, meta = out
                    parts_g.append(("random_last_ditch", g_last, int(meta["K_used"])))
                    break

    if return_Z:
        return [partition_vector_to_2d_matrix(g) for _, g, _ in parts_g]

    return [{"name": name, "g": g, "K_used": K_used} for name, g, K_used in parts_g]
def make_simple_partition(
    N: int,
    cannot_link: Sequence[tuple[int, int]] | None = None, seed=None
):
    """
    Create a single all-ones partition matrix with cannot-link zeroed pairs.
    
    Parameters
    ----------
    N : int
        Number of nodes.
    cannot_link : Sequence[tuple[int, int]] | None
        List of nodes that cannot be linked.
    
    Returns
    -------
    list[array]
        Generated partition.
    """
    cannot_link = [] if cannot_link is None else list(cannot_link)
    initial_z = np.ones((N, N))
    for i, j in cannot_link:
        initial_z[i, j] = 0
        initial_z[j, i] = 0  # enforce symmetry
    return [initial_z]
