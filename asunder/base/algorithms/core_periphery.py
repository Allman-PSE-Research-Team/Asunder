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
    """
    Union-Find with path compression and union by rank.
    
    Parameters
    ----------
    n : int
        Number of elements.
    """

    def __init__(self, n: int) -> None:
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, i: int) -> int:
        """
        Find operation identifies which set a particular element belongs to.
        
        Parameters
        ----------
        i : int
            Any element i.
        
        Returns
        -------
        int
            Representative element from the set containing i.
        """
        if self.parent[i] != i:
            self.parent[i] = self.find(self.parent[i])
        return self.parent[i]

    def union(self, i: int, j: int) -> None:
        """
        Union operation merges the sets that two elements belong to (by rank).
        
        Parameters
        ----------
        i : int
            Any element i.
        j : int
            Any element j.
        """
        ri, rj = self.find(i), self.find(j)
        if ri == rj:
            return
        if self.rank[ri] < self.rank[rj]:
            ri, rj = rj, ri
        self.parent[rj] = ri
        if self.rank[ri] == self.rank[rj]:
            self.rank[ri] += 1

    def components(self) -> tuple[list[list[int]], dict[int, int]]:
        """
        Computes the components in each block and returns said blocks and a component map.
        
        Returns
        -------
        blocks :  list[list[int]]
            Groups of nodes, each of which is referred to as a block.
        comp_map : dict[int, int]]
            Maps nodes to the block they belong to.
        """
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


def normalized_BE_score(A: np.ndarray, labels: np.ndarray) -> float:
    """
    Compute normalized Borgatti-Everett Pearson score.
    
    Parameters
    ----------
    A : np.ndarray of int or float, shape (N, N)
        Graph adjacency/weight matrix.
    labels : np.ndarray of int, shape (N,)
        Reflects the group that each node n belongs to.
    
    Returns
    -------
    float
        BE score.
    """

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
    """
    Choose the best orientation for binary core-periphery labels.
    
    Parameters
    ----------
    A : np.ndarray of int or float, shape (N, N)
        Graph adjacency/weight matrix.
    labels : np.ndarray of int, shape (N,)
        Original orientation of binary core-periphery labels.
    
    Returns
    -------
    np.ndarray of int, shape (N,)
        Best orientation of binary core-periphery labels.
    """
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
    """
    Select the best core-periphery split induced by unique label values. Used to recover best core-periphery structure from a multi-label graph split.
    
    Parameters
    ----------
    A : np.ndarray of int or float, shape (N, N)
        Graph adjacency / weight matrix.
    labels : np.ndarray of int, shape (N,)
        Original node labels including two or more unique groups.
    
    Returns
    -------
    best_labels : np.ndarray of int, shape (N,)
        Best core-periphery split
    best_score: float
        Best BE score.
    """
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
    """
    Binary genetic search for Borgatti-Everett objective with must-link blocks. Includes:
        - Adaptive Mutation Rate: mutation probability decreases as the population converges or as fitness increases
        - Elitism: the top‑𝑘 individuals are carried over unchanged into the next generation (Goldberg 1989)
        - Multi‑Point Crossover: instead of a single cut, two‑point crossover swaps two segments between parents
        - Must‑link constraints
    
    Parameters
    ----------
    A : np.ndarray of int or float, shape (N, N)
        Graph adjacency / weight matrix.
    must_links : list[tuple[int, int]] | None
        Nodes that must be linked.
    pop_size : int
        Size of the population.
    generations : int
        Number of generations.
    init_mut_rate : float
        Initial mutation rate.
    elitism_size : int
        Number of individuals carried unchanged into next generation.
    tournament_size : int
        Number of individuals per tournament.
    seed : int | None
        Random seed value.
    """

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
        """
        Hard repair: force each must‑link block to use uniform labels.
        
        Parameters
        ----------
        z : np.ndarray of int, shape (N,)
            1D graph partition.
        """
        for block in self.must_blocks:
            idxs = np.fromiter(block, dtype=int)
            z[idxs] = int(np.round(z[idxs].mean()))

    def _initialize_population(self) -> list[np.ndarray]:
        """
        Random population initialization with block repair.
        
        Returns
        -------
        population: list[np.ndarray], shape (N, self.pop_size)
            Entire population.
        """
        population: list[np.ndarray] = []
        for _ in range(self.pop_size):
            z = np.random.randint(0, 2, size=self.n)
            self._enforce_blocks(z)
            population.append(z)
        return population

    def _fitness(self, z: np.ndarray) -> float:
        """
        Pearson correlation between adjacency A, and the core-periphery pattern matrix, D = (z_i OR z_j).
        
        Parameters
        ----------
        z : np.ndarray of int, shape (N,)
            Core membership vector which is 1 if node i is in the core and 0 otherwise.
        
        Returns
        -------
        float
            Pearson correlation score.
        """
        D = np.logical_or.outer(z, z).astype(int)
        A_vec = self.A[self.triu_i, self.triu_j]
        D_vec = D[self.triu_i, self.triu_j]
        if A_vec.std() == 0 or D_vec.std() == 0:
            return -1.0
        return float(np.corrcoef(A_vec, D_vec)[0, 1])

    def _tournament(self, pop: list[np.ndarray], fits: list[float]) -> np.ndarray:
        """
        Random tournament-based selection.
        
        Parameters
        ----------
        pop : list[np.ndarray]
            Entire population.
        fits : list[float]
           Fitness score for each individual in population.
        
        Returns
        -------
        np.ndarray of int, shape (N,)
            Individual tournament winner.
        """
        idx = np.random.choice(len(pop), self.tournament_size, replace=False)
        best = idx[np.argmax([fits[i] for i in idx])]
        return pop[int(best)]

    def _multi_point_crossover(self, p1: np.ndarray, p2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Two‑point crossover (DEAP style): Randomly selects two crossover points and swaps the genetic material between these points in the two parent individuals.
        
        Parameters
        ----------
        p1 : np.ndarray of int, shape (N,)
            Tournament 1 winner (parent 1).
        p2 : np.ndarray of int, shape (N,)
            Tournament 2 winner (parent 2).
        
        Returns
        -------
        c1 : np.ndarray of int, shape (N,)
            Child 1 from parents.
        c2 : np.ndarray of int, shape (N,)
            Child 2 from parents.
        """
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
        """
        Adaptive mutation of input individual ``z``. mutation_rate = init_mut_rate * (1 - gen / total_gens)
        
        Parameters
        ----------
        z : np.ndarray of int, shape (N,)
            Input individual.
        gen : int
            Current generation.
        
        Returns
        -------
        np.ndarray of int, shape (N,)
            mutated individual.
        """
        rate = self.init_mut_rate * (1 - gen / max(1, self.generations))
        flips = np.random.rand(self.n) < rate
        z[flips] = 1 - z[flips]
        self._enforce_blocks(z)
        return z

    def run(self) -> tuple[dict[int, int], float]:
        """
        Run genetic algorithm with elitism.
        
        Returns
        -------
        labels : dict[int, int]
            Binary Core-Periphery assignments.
        best_fit : float
            Best fitness score, corresponding to returned labels.
        """
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
    """
    Continuous genetic search for BE objective with block constraints.
    
    Parameters
    ----------
    A : np.ndarray of int or float, shape (N, N)
        Graph adjacency / weight matrix.
    must_links : list[tuple[int, int]] | None
        Nodes that must be linked.
    nonlinear_nodes : list[int] | None
        Nodes that correspond to nonlinear constraints and so, should be merged.
    pop_size : int
        Size of the population.
    generations : int
        Number of generations.
    crossover_rate : float
        crossover rate below which parents are blended into children.
    mutation_rate : float
        Fixed mutation rate.
    tournament_size : int
        Number of individuals per tournament.
    gene_init_scale : float
        gene initialization scale.
    seed : int | None
        Random seed value.
    """

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
        
        self.block_idxs = [np.fromiter(block, dtype=int) for block in self.blocks]
        self.n_blocks = len(self.block_idxs)

        # Precompute A_vec (and its centered norm) once in __init__
        self.A_vec = self.A[self.triu_i, self.triu_j].astype(np.float64, copy=False)
        self.m = self.A_vec.size

        A_mean = self.A_vec.mean()
        self.A_ctr = self.A_vec - A_mean
        self.A_ctr_norm = float(np.linalg.norm(self.A_ctr))
        self.A_ctr_sum = float(self.A_ctr.sum())  # close to 0, kept for numerical safety


    def _enforce_blocks(self, c: np.ndarray) -> None:
        """
        For each block, set all c_i to the block‐mean.
        
        Parameters
        ----------
        c : np.ndarray of int, shape (N,)
            1D graph partition.
        """
        for block in self.blocks:
            idxs = np.fromiter(block, dtype=int)
            c[idxs] = float(np.mean(c[idxs]))

    def _init_population(self) -> list[np.ndarray]:
        """
        Initialize population with real vectors in [0,scale], one per individual.
        
        Returns
        -------
        population: list[np.ndarray]
            Entire population.
        """
        population: list[np.ndarray] = []
        for _ in range(self.pop_size):
            c = np.random.rand(self.n) * self.scale
            self._enforce_blocks(c)
            population.append(c)
        return population

    def _fitness(self, c: np.ndarray) -> float:
        """
        Pearson correlation score between A_ij and D_ij = c_i * c_j for i<j.
        
        Parameters
        ----------
        c : np.ndarray of int, shape (N,)
            Core membership vector which is 1 if node i is in the core and 0 otherwise.
        
        Returns
        -------
        float
            Pearson correlation score.
        """
        # D = np.outer(c, c)
        # A_vec = self.A[self.triu_i, self.triu_j]
        # D_vec = D[self.triu_i, self.triu_j]
        # if A_vec.std() == 0 or D_vec.std() == 0:
        #     return -1.0
        # return float(np.corrcoef(A_vec, D_vec)[0, 1])

        # Upper-triangle products without forming an n×n matrix
        y = c[self.triu_i] * c[self.triu_j]  # shape (m,)

        if self.A_ctr_norm == 0.0:
            return -1.0

        sum_y = float(y.sum())
        sum_y2 = float(np.dot(y, y))
        mean_y = sum_y / self.m

        # ||y - mean_y|| computed from sums (avoids allocating y - mean_y)
        sy2 = sum_y2 - self.m * mean_y * mean_y
        if sy2 <= 0.0:
            return -1.0

        # Numerator: (A-meanA)·(y-meany) = (A-meanA)·y - meany*sum(A-meanA)
        num = float(np.dot(self.A_ctr, y) - mean_y * self.A_ctr_sum)
        den = self.A_ctr_norm * (sy2 ** 0.5)
        return num / den


    def _tournament(self, pop: list[np.ndarray], fits: list[float]) -> np.ndarray:
        """
        Select one parent by tournament.
        
        Parameters
        ----------
        pop : list[np.ndarray]
            Entire population.
        fits : list[float]
           Fitness score for each individual in population.
        
        Returns
        -------
        np.ndarray of int, shape (N,)
            Individual tournament winner.
        """
        idx = np.random.choice(len(pop), self.tournament_size, replace=False)
        best = idx[np.argmax([fits[i] for i in idx])]
        return pop[int(best)]

    def _crossover(self, p1: np.ndarray, p2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Blend crossover: c' = alpha * p1 + (1-alpha)*p2.
        
        Parameters
        ----------
        p1 : np.ndarray of int, shape (N,)
            Tournament 1 winner (parent 1).
        p2 : np.ndarray of int, shape (N,)
            Tournament 2 winner (parent 2).
        
        Returns
        -------
        c1 : np.ndarray of int, shape (N,)
            Child 1 from parents.
        c2 : np.ndarray of int, shape (N,)
            Child 2 from parents.
        """
        if np.random.rand() > self.crossover_rate:
            return p1.copy(), p2.copy()
        alpha = np.random.rand(self.n)
        c1 = alpha * p1 + (1 - alpha) * p2
        c2 = alpha * p2 + (1 - alpha) * p1
        self._enforce_blocks(c1)
        self._enforce_blocks(c2)
        return c1, c2

    def _mutate(self, c: np.ndarray) -> np.ndarray:
        """
        Gaussian mutation around each gene.
        
        Parameters
        ----------
        c : np.ndarray of int, shape (N,)
            Input individual.
        
        Returns
        -------
        np.ndarray of int, shape (N,)
            mutated individual.
        """
        flips = np.random.rand(self.n) < self.mutation_rate
        noise = np.random.normal(scale=self.scale * 0.1, size=self.n)
        c[flips] = c[flips] + noise[flips]
        np.clip(c, 0, self.scale, out=c)
        self._enforce_blocks(c)
        return c

    def run(self) -> tuple[dict[int, float], float]:
        """
        Execute GA and return continuous coreness and best fitness.
        
        Returns
        -------
        coreness : dict[int, float]
            Continuous coreness score for each node.
        best_fit : float
            Best fitness score, corresponding to returned coreness.
        """

        # TODO: Add a boolean parameter that allows you to configure this to use block level genome like run_de.
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

    def _expand_blocks(self, g: np.ndarray) -> np.ndarray:
        """
        Expand block-level genome into full individual genome.
        
        Parameters
        ----------
        g : np.ndarray of float, shape (n_blocks,)
            Block-level genome g (len = n_blocks)
        
        Returns
        -------
        c : np.ndarray of float, shape (N,)
            Full individual genome c (len = N)
        """
        c = np.empty(self.n, dtype=float)
        for val, idxs in zip(g, self.block_idxs):
            c[idxs] = float(val)
        return c

    def _init_population_blocks(self) -> list[np.ndarray]:
        """
        Initialize population in block space.

        Returns
        -------
        list[np.ndarray]
            Entire population in block space.
        """
        return [np.random.rand(self.n_blocks) * self.scale for _ in range(self.pop_size)]

    def run_de(self, F: float = 0.7, CR: float = 0.9) -> tuple[dict[int, float], float]:
        """
        Run Differential Evolution (DE/rand/1/bin) using a block-level genome.
        
        Returns
        -------
        coreness : dict[int, float]
            Continuous coreness score for each node.
        best_fit : float
            Best fitness score, corresponding to returned coreness.
        """
        pop = self._init_population_blocks()
        fits = [self._fitness(self._expand_blocks(g)) for g in pop]

        best_idx = int(np.argmax(fits))
        best_g = pop[best_idx].copy()
        best_fit = float(fits[best_idx])

        for _ in range(self.generations):
            for i in range(self.pop_size):
                idxs = np.arange(self.pop_size)
                idxs = idxs[idxs != i]
                r1, r2, r3 = np.random.choice(idxs, 3, replace=False)
                a, b, c = pop[int(r1)], pop[int(r2)], pop[int(r3)]

                v = a + F * (b - c)
                np.clip(v, 0.0, self.scale, out=v)

                u = pop[i].copy()
                cross = np.random.rand(self.n_blocks) < CR
                cross[np.random.randint(self.n_blocks)] = True
                u[cross] = v[cross]
                np.clip(u, 0.0, self.scale, out=u)

                fu = self._fitness(self._expand_blocks(u))
                if fu > fits[i]:
                    pop[i] = u
                    fits[i] = fu
                    if fu > best_fit:
                        best_fit = float(fu)
                        best_g = u.copy()

        best_c = self._expand_blocks(best_g)
        coreness = {i: float(best_c[i]) for i in self.nodes}
        return coreness, best_fit


def upper_triangle(mat: csr_matrix) -> tuple[np.ndarray, np.ndarray]:
    """
    Return upper-triangle index arrays.
    
    Parameters
    ----------
    mat : csr_matrix, shape (N, N)
        Input matrix.
    
    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Row and column indices of elements in the upper triangle.
    """
    triu = sp.triu(mat, k=1, format="coo")
    return triu.row, triu.col


def corr_upper(A_vals: np.ndarray, D_vals: np.ndarray) -> float:
    """
    Compute Pearson correlation between two equal-length vectors.
    
    Parameters
    ----------
    A_vals : np.ndarray
        Values from the upper triangle of the adjacency / weight matrix.
    D_vals : np.ndarray
        Values from the upper triangle of the pattern matrix.
    
    Returns
    -------
    float
        Pearson correlation score.
    """
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
    """
    Continuous- BE vcorenessia constrained block updates.
    
    Parameters
    ----------
    A_csr : csr_matrix, shape (N, N)
        Input adjacency / weight matrix as a csr matrix.
    must_links : list[tuple[int, int]]
        Node pairs that must be linked.
    nonlinear_nodes : list[int] | None
        Nodes that correspond to nonlinear constraints and so, should be merged.
    max_iter : int
        Maximum number of iterations to run.
    seed : int | None
        Random seed value.
    
    Returns
    -------
    c : np.ndarray of float, shape (N,)
        Continuous coreness score for each node.
    best_r: float
        Pearson correlation achieved.
    """
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
        """
        Current score.
        
        Parameters
        ----------
        vec : np.ndarray
            Input parameter.
        
        Returns
        -------
        float
            Computed result.
        """
        return corr_upper(A_vec, vec[tri_i] * vec[tri_j])

    best_r = current_score(c)

    def score_with_block(vec: np.ndarray, block: list[int], t_val: float, mask: np.ndarray) -> float:
        """
        Score with block.
        
        Parameters
        ----------
        vec : np.ndarray
            Input parameter.
        block : list[int]
            Input parameter.
        t_val : float
            Input parameter.
        mask : np.ndarray
            Input parameter.
        
        Returns
        -------
        float
            Computed result.
        """
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
    """
    Block-aggregated continuous coreness using principal eigenvector.

    1) Builds blocks from must_links + nonlinear_nodes.
    2) Aggregates A into a block‐adjacency B of size b×b.
    3) Computes principal eigenvector v of B -> one coreness per block.
    4) Expands v to node‐level c[i] = v[block(i)] and normalizes if requested.
    5) Computes Q = corr( A_flat, (c⌣c)_flat ) as the continuous BE score.
    
    Parameters
    ----------
    A_csr : csr_matrix
        Input adjacency / weight matrix as a csr matrix.
    must_links : list[tuple[int, int]]
        Node pairs that must be linked.
    nonlinear_nodes : list[int] | None
        Nodes that correspond to nonlinear constraints and so, should be merged.
    normalize : bool
        Enable/disable normalization.

    Returns
    -------
    c : np.ndarray of float, shape (N,)
        Continuous coreness vector.
    Q: float
        Pearson‐r fit of A vs. c·cᵀ
    """
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
    """
    Convert adjacency input into an undirected NetworkX Graph.

    Supported inputs are NumPy 2D arrays, SciPy sparse matrices,
    ``networkx.Graph`` objects, and ``networkx.DiGraph`` objects. Edge
    weights are preserved if present, and diagonal entries are ignored.
    
    Parameters
    ----------
    adj : AdjLike
        Adjacency / weight matrix-like array.
    
    Returns
    -------
    nx.Graph
        NetworkX graph.
    """
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
    """
    Produce a boolean mask of shape (N,) where True indicates a core node.      
    
    Parameters
    ----------
    core_periphery : np.ndarray | list[int]
        Accepted inputs:
            1D vector length N with values in {0,1,True,False} (1/True => core).
            2D partition matrix (N,2) that is one-hot.
    core_is : int | None
        Index of the 'core' column (default 0).
    
    Returns
    -------
    np.ndarray of bool, shape (N,)
        Boolean core mask.
    """
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
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Build community labels and assignment matrix after collapsing all core nodes
    into a single community (id=0) and splitting the periphery into connected
    components once core nodes are removed.

    Parameters
    ----------
    adj : AdjLike
        Graph adjacency / weight matrix. Treated as undirected for component detection.
    core_periphery : np.ndarray | list[int]
        Either:
          - length-N vector, where nonzero/True => core, 0/False => periphery, or
          - (N,2) one-hot partition matrix [core_col, periphery_col].
    core_is : {0,1} or None, default=0
        If a (N,2) matrix is provided, index of the core column.
        Ignored for 1D vector input.

    Returns
    -------
    labels : np.ndarray, shape (N,)
        Community id for each node. 0 => all core nodes; 1..K => periphery components.
    info : dict
        Metadata including:
          - 'n_core', 'n_periphery'
          - 'components': list of sets with node indices for each periphery component (in order)
          - 'community_node_indices': list where entry j contains indices for community j
    """

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
