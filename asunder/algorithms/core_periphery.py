"""Core-periphery detection algorithms."""

from __future__ import annotations

from typing import Any

import networkx as nx
import numpy as np
import scipy.sparse as sp
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh

AdjLike = np.ndarray | sp.spmatrix | nx.Graph | nx.DiGraph

class UnionFind:
    """Union-find with path compression and union by rank."""

    def __init__(self, n: int) -> None:
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, i: int) -> int:
        if self.parent[i] != i:
            self.parent[i] = self.find(self.parent[i])
        return self.parent[i]

    def union(self, i: int, j: int) -> None:
        ri, rj = self.find(i), self.find(j)
        if ri == rj:
            return
        if self.rank[ri] < self.rank[rj]:
            ri, rj = rj, ri
        self.parent[rj] = ri
        if self.rank[ri] == self.rank[rj]:
            self.rank[ri] += 1

    def components(self) -> tuple[list[list[int]], dict[int, int]]:
        for i in range(len(self.parent)):
            self.find(i)

        root_to_nodes: dict[int, list[int]] = {}
        for idx, root in enumerate(self.parent):
            root_to_nodes.setdefault(root, []).append(idx)

        blocks = list(root_to_nodes.values())
        comp_map: dict[int, int] = {}
        for block_id, block in enumerate(blocks):
            for node in block:
                comp_map[node] = block_id
        return blocks, comp_map


def normalized_BE_score(A: np.ndarray, labels: np.ndarray, linearize: bool = False) -> float:
    """Compute normalized Borgatti-Everett Pearson score."""
    del linearize  # [MODIFIED] retained for backward compatibility

    matrix = np.asarray(A, dtype=float)
    binary_labels = np.asarray(labels, dtype=bool)
    n = matrix.shape[0]

    i, j = np.triu_indices(n, k=1)
    A_vec = matrix[i, j]
    D_vec = (binary_labels[i] | binary_labels[j]).astype(float)

    A_centered = A_vec - A_vec.mean()
    D_centered = D_vec - D_vec.mean()
    den = np.sqrt((A_centered * A_centered).sum() * (D_centered * D_centered).sum())
    if den == 0:
        return 0.0
    return float((A_centered * D_centered).sum() / den)


def find_core(A: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Choose the best orientation for binary core-periphery labels."""
    labels = np.asarray(labels).copy()
    if labels.size == 0:
        return labels
    if np.max(labels) > 1:
        labels = labels - (np.max(labels) - 1)

    score = normalized_BE_score(A, labels)
    labels_inv = (labels == 0).astype(np.int64)
    score_inv = normalized_BE_score(A, labels_inv)
    return labels if score > score_inv else labels_inv


def find_core_advance(A: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, float]:
    """Select the best core-periphery split induced by unique label values."""
    best_score = -np.inf
    best_labels: np.ndarray | None = None

    for index in set(np.asarray(labels).tolist()):
        trial_labels = (labels != index).astype(int)
        trial_score = normalized_BE_score(A, trial_labels)
        if trial_score > best_score:
            best_score = trial_score
            best_labels = trial_labels

    if best_labels is None:
        return np.asarray(labels), float("-inf")
    return best_labels, float(best_score)


class EnhancedGeneticBE:
    """Binary genetic search for Borgatti-Everett objective with must-link blocks."""

    def __init__(
        self,
        A: np.ndarray,
        must_links: list[tuple[int, int]] | None = None,
        pop_size: int = 50,
        generations: int = 100,
        init_mut_rate: float = 0.1,
        elitism_size: int = 2,
        tournament_size: int = 3,
        seed: int | None = None,
    ) -> None:
        if seed is not None:
            np.random.seed(seed)

        self.A = np.asarray(A)
        self.n = self.A.shape[0]
        self.nodes = list(range(self.n))
        self.triu_i, self.triu_j = np.triu_indices(self.n, k=1)

        graph = nx.Graph()
        graph.add_nodes_from(self.nodes)
        graph.add_edges_from(must_links or [])
        self.must_blocks = list(nx.connected_components(graph))

        self.pop_size = pop_size
        self.generations = generations
        self.init_mut_rate = init_mut_rate
        self.elitism_size = elitism_size
        self.tournament_size = tournament_size

    def _enforce_blocks(self, z: np.ndarray) -> None:
        for block in self.must_blocks:
            idxs = np.fromiter(block, dtype=int)
            z[idxs] = int(np.round(z[idxs].mean()))

    def _initialize_population(self) -> list[np.ndarray]:
        population: list[np.ndarray] = []
        for _ in range(self.pop_size):
            z = np.random.randint(0, 2, size=self.n)
            self._enforce_blocks(z)
            population.append(z)
        return population

    def _fitness(self, z: np.ndarray) -> float:
        D = np.logical_or.outer(z, z).astype(int)
        A_vec = self.A[self.triu_i, self.triu_j]
        D_vec = D[self.triu_i, self.triu_j]
        if A_vec.std() == 0 or D_vec.std() == 0:
            return -1.0
        return float(np.corrcoef(A_vec, D_vec)[0, 1])

    def _tournament(self, pop: list[np.ndarray], fits: list[float]) -> np.ndarray:
        idx = np.random.choice(len(pop), self.tournament_size, replace=False)
        best = idx[np.argmax([fits[i] for i in idx])]
        return pop[int(best)]

    def _multi_point_crossover(self, p1: np.ndarray, p2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if np.random.rand() < 0.9:
            a, b = sorted(np.random.choice(range(1, self.n), size=2, replace=False))
            c1 = np.concatenate([p1[:a], p2[a:b], p1[b:]])
            c2 = np.concatenate([p2[:a], p1[a:b], p2[b:]])
        else:
            c1, c2 = p1.copy(), p2.copy()
        self._enforce_blocks(c1)
        self._enforce_blocks(c2)
        return c1, c2

    def _adaptive_mutation(self, z: np.ndarray, gen: int) -> np.ndarray:
        rate = self.init_mut_rate * (1 - gen / max(1, self.generations))
        flips = np.random.rand(self.n) < rate
        z[flips] = 1 - z[flips]
        self._enforce_blocks(z)
        return z

    def run(self) -> tuple[dict[int, int], float]:
        pop = self._initialize_population()
        best_z: np.ndarray | None = None
        best_fit = -np.inf

        for gen in range(self.generations):
            fits = [self._fitness(ind) for ind in pop]
            elite_idx = np.argsort(fits)[-self.elitism_size :]
            elites = [pop[i].copy() for i in elite_idx]

            cur_best = int(np.argmax(fits))
            if fits[cur_best] > best_fit:
                best_fit = float(fits[cur_best])
                best_z = pop[cur_best].copy()

            new_pop = elites.copy()
            while len(new_pop) < self.pop_size:
                p1 = self._tournament(pop, fits)
                p2 = self._tournament(pop, fits)
                c1, c2 = self._multi_point_crossover(p1, p2)
                new_pop.extend([self._adaptive_mutation(c1, gen), self._adaptive_mutation(c2, gen)])
            pop = new_pop[: self.pop_size]

        if best_z is None:
            best_z = np.zeros(self.n, dtype=int)
        labels = {self.nodes[i]: int(best_z[i]) for i in range(self.n)}
        return labels, best_fit


class FullContinuousGeneticBE:
    """Continuous genetic search for BE objective with block constraints."""

    def __init__(
        self,
        A: np.ndarray,
        must_links: list[tuple[int, int]] | None = None,
        nonlinear_nodes: list[int] | None = None,
        pop_size: int = 50,
        generations: int = 100,
        crossover_rate: float = 0.8,
        mutation_rate: float = 0.1,
        tournament_size: int = 3,
        gene_init_scale: float = 1.0,
        seed: int | None = None,
    ) -> None:
        if seed is not None:
            np.random.seed(seed)

        self.A = np.asarray(A)
        self.n = self.A.shape[0]
        self.nodes = list(range(self.n))
        self.triu_i, self.triu_j = np.triu_indices(self.n, k=1)

        links = [] if must_links is None else list(must_links)
        if nonlinear_nodes is not None and len(nonlinear_nodes) > 1:
            base = nonlinear_nodes[0]
            for node in nonlinear_nodes[1:]:
                links.append((base, node))

        graph = nx.Graph()
        graph.add_nodes_from(self.nodes)
        graph.add_edges_from(links)
        self.blocks = [set(component) for component in nx.connected_components(graph)]

        self.pop_size = pop_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.tournament_size = tournament_size
        self.scale = gene_init_scale

    def _enforce_blocks(self, c: np.ndarray) -> None:
        for block in self.blocks:
            idxs = np.fromiter(block, dtype=int)
            c[idxs] = float(np.mean(c[idxs]))

    def _init_population(self) -> list[np.ndarray]:
        population: list[np.ndarray] = []
        for _ in range(self.pop_size):
            c = np.random.rand(self.n) * self.scale
            self._enforce_blocks(c)
            population.append(c)
        return population

    def _fitness(self, c: np.ndarray) -> float:
        D = np.outer(c, c)
        A_vec = self.A[self.triu_i, self.triu_j]
        D_vec = D[self.triu_i, self.triu_j]
        if A_vec.std() == 0 or D_vec.std() == 0:
            return -1.0
        return float(np.corrcoef(A_vec, D_vec)[0, 1])

    def _tournament(self, pop: list[np.ndarray], fits: list[float]) -> np.ndarray:
        idx = np.random.choice(len(pop), self.tournament_size, replace=False)
        best = idx[np.argmax([fits[i] for i in idx])]
        return pop[int(best)]

    def _crossover(self, p1: np.ndarray, p2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if np.random.rand() > self.crossover_rate:
            return p1.copy(), p2.copy()
        alpha = np.random.rand(self.n)
        c1 = alpha * p1 + (1 - alpha) * p2
        c2 = alpha * p2 + (1 - alpha) * p1
        self._enforce_blocks(c1)
        self._enforce_blocks(c2)
        return c1, c2

    def _mutate(self, c: np.ndarray) -> np.ndarray:
        flips = np.random.rand(self.n) < self.mutation_rate
        noise = np.random.normal(scale=self.scale * 0.1, size=self.n)
        c[flips] = c[flips] + noise[flips]
        np.clip(c, 0, self.scale, out=c)
        self._enforce_blocks(c)
        return c

    def run(self) -> tuple[dict[int, float], float]:
        pop = self._init_population()
        best_c: np.ndarray | None = None
        best_fit = -np.inf

        for _ in range(self.generations):
            fits = [self._fitness(ind) for ind in pop]
            idx = int(np.argmax(fits))
            if fits[idx] > best_fit:
                best_fit = float(fits[idx])
                best_c = pop[idx].copy()

            new_pop: list[np.ndarray] = []
            while len(new_pop) < self.pop_size:
                p1 = self._tournament(pop, fits)
                p2 = self._tournament(pop, fits)
                c1, c2 = self._crossover(p1, p2)
                new_pop.append(self._mutate(c1))
                if len(new_pop) < self.pop_size:
                    new_pop.append(self._mutate(c2))
            pop = new_pop

        if best_c is None:
            best_c = np.zeros(self.n, dtype=float)
        coreness = {i: float(best_c[i]) for i in self.nodes}
        return coreness, best_fit


def upper_triangle(mat: csr_matrix) -> tuple[np.ndarray, np.ndarray]:
    """Return upper-triangle index arrays."""
    triu = sp.triu(mat, k=1, format="coo")
    return triu.row, triu.col


def corr_upper(A_vals: np.ndarray, D_vals: np.ndarray) -> float:
    """Compute Pearson correlation between two vectors."""
    if A_vals.std() == 0 or D_vals.std() == 0:
        return -1.0
    return float(np.corrcoef(A_vals, D_vals)[0, 1])


def detect_continuous_KL(
    A_csr: csr_matrix,
    must_links: list[tuple[int, int]],
    nonlinear_nodes: list[int],
    max_iter: int = 100,
    seed: int | None = None,
) -> tuple[np.ndarray, float]:
    """Continuous-coreness BE via constrained block updates."""
    if seed is not None:
        np.random.seed(seed)

    n = A_csr.shape[0]
    tri_i, tri_j = upper_triangle(A_csr)
    A_vec = A_csr[tri_i, tri_j].A1

    uf = UnionFind(n)
    for i, j in must_links:
        uf.union(i, j)
    if nonlinear_nodes:
        base = nonlinear_nodes[0]
        for node in nonlinear_nodes[1:]:
            uf.union(base, node)
    blocks, _ = uf.components()

    c = np.random.rand(n)
    for block in blocks:
        mean_val = c[block].mean()
        c[block] = mean_val
    c /= max(np.linalg.norm(c), 1e-12)

    def current_score(vec: np.ndarray) -> float:
        return corr_upper(A_vec, vec[tri_i] * vec[tri_j])

    best_r = current_score(c)

    def score_with_block(vec: np.ndarray, block: list[int], t_val: float, mask: np.ndarray) -> float:
        d_new = vec[tri_i] * vec[tri_j]
        for ii, (i, j) in enumerate(zip(tri_i[mask], tri_j[mask], strict=False)):
            if i in block and j in block:
                d_new[np.flatnonzero(mask)[ii]] = t_val * t_val
            elif i in block:
                d_new[np.flatnonzero(mask)[ii]] = t_val * vec[j]
            else:
                d_new[np.flatnonzero(mask)[ii]] = t_val * vec[i]
        return corr_upper(A_vec, d_new)

    for _ in range(max_iter):
        improved = False
        for block in blocks:
            block_set = set(block)
            mask = np.isin(tri_i, block) | np.isin(tri_j, block)

            lo, hi = 0.0, 1.0
            for _ in range(20):
                g1 = lo + 0.382 * (hi - lo)
                g2 = lo + 0.618 * (hi - lo)
                if score_with_block(c, block, g1, mask) < score_with_block(c, block, g2, mask):
                    lo = g1
                else:
                    hi = g2
            t_best = 0.5 * (lo + hi)

            new_c = c.copy()
            for idx in block_set:
                new_c[idx] = t_best
            new_c /= max(np.linalg.norm(new_c), 1e-12)
            new_r = current_score(new_c)

            if new_r > best_r:
                c, best_r = new_c, new_r
                improved = True
                break
        if not improved:
            break

    return c, float(best_r)


def spectral_continuous_cp_detection(
    A_csr: csr_matrix,
    must_links: list[tuple[int, int]],
    nonlinear_nodes: list[int],
    normalize: bool = True,
) -> tuple[np.ndarray, float]:
    """Block-aggregated continuous coreness using principal eigenvector."""
    n = A_csr.shape[0]
    uf = UnionFind(n)
    for i, j in must_links:
        uf.union(i, j)
    if nonlinear_nodes:
        base = nonlinear_nodes[0]
        for node in nonlinear_nodes[1:]:
            uf.union(base, node)
    blocks, comp_map = uf.components()
    b = len(blocks)

    B = np.zeros((b, b), dtype=float)
    for p, block in enumerate(blocks):
        for i in block:
            row_start, row_end = A_csr.indptr[i], A_csr.indptr[i + 1]
            for idx in range(row_start, row_end):
                j = int(A_csr.indices[idx])
                q = comp_map[j]
                B[p, q] += float(A_csr.data[idx])
    B = 0.5 * (B + B.T)

    if b <= 50:
        eigvals, eigvecs = np.linalg.eigh(B)
        v = eigvecs[:, np.argmax(eigvals)]
    else:
        _, eigvec = eigsh(B, k=1, which="LA")
        v = eigvec[:, 0]
    v = np.abs(v)

    c = np.array([v[comp_map[i]] for i in range(n)], dtype=float)
    if normalize:
        c = (c - c.min()) / max(1e-12, (c.max() - c.min()))

    iu, ju = np.triu_indices(n, k=1)
    A_vec = A_csr[iu, ju].A1
    D_vec = c[iu] * c[ju]
    A_centered = A_vec - A_vec.mean()
    D_centered = D_vec - D_vec.mean()
    den = np.sqrt((A_centered * A_centered).sum() * (D_centered * D_centered).sum())
    Q = 0.0 if den == 0 else float((A_centered * D_centered).sum() / den)
    return c, Q


def _to_undirected_graph(adj: AdjLike) -> nx.Graph:
    if isinstance(adj, nx.DiGraph):
        graph = adj.to_undirected()
    elif isinstance(adj, nx.Graph):
        graph = adj.copy()
    elif sp.issparse(adj):
        graph = nx.from_scipy_sparse_array(adj, edge_attribute="weight")
    elif isinstance(adj, np.ndarray):
        if adj.ndim != 2 or adj.shape[0] != adj.shape[1]:
            raise ValueError("NumPy adjacency must be a square 2D array.")
        matrix = adj.copy()
        np.fill_diagonal(matrix, 0)
        graph = nx.from_numpy_array(matrix)
    else:
        raise TypeError("Unsupported adjacency type.")
    graph.remove_edges_from(nx.selfloop_edges(graph))
    return graph


def _core_mask_from_partition(
    core_periphery: np.ndarray | list[int],
    core_is: int | None = 0,
) -> np.ndarray:
    arr = np.asarray(core_periphery)
    if arr.ndim == 1:
        core_mask = arr if arr.dtype == bool else (arr != 0)
    elif arr.ndim == 2 and arr.shape[1] == 2:
        if core_is is None or core_is not in (0, 1):
            raise ValueError("core_is must be 0 or 1 for a 2-column partition matrix.")
        core_mask = arr[:, core_is] > 0.5
    else:
        raise ValueError("core_periphery must be length-N vector or (N,2) partition matrix.")
    return core_mask.astype(bool)


def partititon_periphery_components(
    adj: AdjLike,
    core_periphery: np.ndarray | list[int],
    *,
    core_is: int | None = 0,
    return_sparse: bool = False,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Collapse core nodes and split periphery by connected components."""
    del return_sparse  # [MODIFIED] retained for API compatibility

    graph = _to_undirected_graph(adj)
    n_nodes = graph.number_of_nodes()
    if list(graph.nodes()) != list(range(n_nodes)):
        mapping = {node: i for i, node in enumerate(graph.nodes())}
        graph = nx.relabel_nodes(graph, mapping, copy=True)

    core_mask = _core_mask_from_partition(core_periphery, core_is=core_is)
    if core_mask.shape[0] != n_nodes:
        raise ValueError(f"core_periphery length {core_mask.shape[0]} != number of nodes {n_nodes}.")

    core_nodes = np.flatnonzero(core_mask).tolist()
    periph_nodes = np.flatnonzero(~core_mask).tolist()
    periph_subgraph = graph.subgraph(periph_nodes).copy()
    components = list(nx.connected_components(periph_subgraph))

    labels = np.full(n_nodes, -1, dtype=int)
    if core_nodes:
        labels[core_nodes] = 0

    community_node_indices: list[np.ndarray] = [np.array(core_nodes, dtype=int)]
    for k, component in enumerate(components, start=1):
        idx = np.fromiter(component, dtype=int)
        labels[idx] = k
        community_node_indices.append(idx)

    info = {
        "n_core": int(len(core_nodes)),
        "n_periphery": int(len(periph_nodes)),
        "components": components,
        "community_node_indices": community_node_indices,
    }
    return labels, info
