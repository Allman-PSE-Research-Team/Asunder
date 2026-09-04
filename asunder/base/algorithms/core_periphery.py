"""Core-periphery detection algorithms."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import networkx as nx
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh

AdjLike = np.ndarray | sp.spmatrix | nx.Graph | nx.DiGraph
CorePeripheryTarget = Literal["contracted", "original"]


class UnionFind:
    """Maintain disjoint node sets using path compression and union by rank.

    Parameters
    ----------
    n : int
        Number of initially independent elements.
    """

    def __init__(self, n: int) -> None:
        """Initialize ``n`` singleton sets."""

        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, node: int) -> int:
        """Return the representative of the set containing ``node``."""

        if self.parent[node] != node:
            self.parent[node] = self.find(self.parent[node])
        return self.parent[node]

    def union(self, source: int, target: int) -> None:
        """Merge the sets containing ``source`` and ``target``."""

        source_root, target_root = self.find(source), self.find(target)
        if source_root == target_root:
            return
        if self.rank[source_root] < self.rank[target_root]:
            source_root, target_root = target_root, source_root
        self.parent[target_root] = source_root
        if self.rank[source_root] == self.rank[target_root]:
            self.rank[source_root] += 1

    def components(self) -> tuple[list[list[int]], dict[int, int]]:
        """Return deterministic blocks and a node-to-block mapping.

        Returns
        -------
        blocks : list of list of int
            Connected constraint blocks in original-node order.
        node_to_block : dict of int to int
            Block index for every original node.
        """

        roots: dict[int, list[int]] = {}
        for node in range(len(self.parent)):
            roots.setdefault(self.find(node), []).append(node)
        blocks = list(roots.values())
        node_to_block = {
            node: block_index
            for block_index, block in enumerate(blocks)
            for node in block
        }
        return blocks, node_to_block


@dataclass
class CorePeripheryResult:
    """Typed result shared by the continuous core-periphery backends.

    Attributes
    ----------
    algorithm : str
        Backend name, such as ``"SPEC"``, ``"GA"``, or ``"KL"``.
    target_space : {"contracted", "original"}
        Space optimized by the detector and used for ``primary_fit``.
    blocks : tuple of tuple of int
        Original-node indices represented by each contracted entity.
    node_to_block : ndarray of int, shape (N,)
        Contracted block index for every original node.
    block_sizes : ndarray of float, shape (b,)
        Number of original nodes in each block.
    contracted_adjacency : ndarray of float, shape (b, b)
        Aggregate adjacency ``B = S.T @ A @ S``.
    block_embedding : ndarray of float, shape (b, rank)
        Block representation used for discrete clustering.
    block_scores : ndarray of float, shape (b,) or None
        Scalar continuous coreness in rank-one modes. Rank two has no scalar
        replacement for its two-dimensional embedding.
    continuous_node_scores : ndarray of float, shape (N,) or None
        ``block_scores`` expanded to original nodes after continuous detection.
    spectral_rank : int
        Retained rank; one for GA and KL.
    eigenproblem : {"ordinary", "generalized"} or None
        Spectral eigenproblem, when the SPEC backend is used.
    eigenvalues : ndarray of float or None
        Retained spectral eigenvalues, ordered by the selected rank rule.
    block_eigenvectors : ndarray of float or None
        Ordinary or generalized block eigenvectors before eigenvalue scaling.
    block_labels, node_labels : ndarray of int or None
        Oriented discrete assignments where ``1`` denotes the core.
    contracted_fit, original_fit : float or None
        Pearson fits of the continuous reconstruction in each space.
    primary_fit : float or None
        Reconstruction fit in ``target_space``.
    contracted_be_score, original_be_score : float or None
        Discrete Borgatti-Everett scores used to assess the oriented labels.
    primary_be_score : float or None
        Discrete score in ``target_space``.
    contracted_reconstruction_fit, original_reconstruction_fit : float
        Explicit aliases for the corresponding continuous fit fields.
    must_group_role : {"core", "periphery"} or None
        Detected role of the optional generic ``must_group`` block.
    """

    algorithm: str
    target_space: CorePeripheryTarget
    blocks: tuple[tuple[int, ...], ...]
    node_to_block: np.ndarray
    block_sizes: np.ndarray
    contracted_adjacency: np.ndarray = field(repr=False)
    block_embedding: np.ndarray
    block_scores: np.ndarray | None
    continuous_node_scores: np.ndarray | None
    spectral_rank: int = 1
    eigenproblem: str | None = None
    eigenvalues: np.ndarray | None = None
    block_eigenvectors: np.ndarray | None = None
    block_labels: np.ndarray | None = None
    node_labels: np.ndarray | None = None
    contracted_fit: float | None = None
    original_fit: float | None = None
    primary_fit: float | None = None
    contracted_be_score: float | None = None
    original_be_score: float | None = None
    primary_be_score: float | None = None
    contracted_reconstruction_fit: float = 0.0
    original_reconstruction_fit: float = 0.0
    must_group_role: str | None = None

    def to_metadata(self) -> dict[str, Any]:
        """Return a copy-safe public metadata dictionary.

        The potentially large ``contracted_adjacency`` remains available on
        the typed result but is omitted from metadata to avoid duplication.
        """

        return {
            "algorithm": self.algorithm,
            "target_space": self.target_space,
            "spectral_rank": self.spectral_rank,
            "eigenproblem": self.eigenproblem,
            "blocks": self.blocks,
            "node_to_block": self.node_to_block.copy(),
            "block_sizes": self.block_sizes.copy(),
            "block_embedding": self.block_embedding.copy(),
            "block_scores": None if self.block_scores is None else self.block_scores.copy(),
            "continuous_node_scores": (
                None if self.continuous_node_scores is None else self.continuous_node_scores.copy()
            ),
            "eigenvalues": None if self.eigenvalues is None else self.eigenvalues.copy(),
            "block_eigenvectors": (
                None if self.block_eigenvectors is None else self.block_eigenvectors.copy()
            ),
            "block_labels": None if self.block_labels is None else self.block_labels.copy(),
            "core_labels": None if self.node_labels is None else self.node_labels.copy(),
            "contracted_fit": self.contracted_fit,
            "original_fit": self.original_fit,
            "primary_fit": self.primary_fit,
            "contracted_be_score": self.contracted_be_score,
            "original_be_score": self.original_be_score,
            "primary_be_score": self.primary_be_score,
            "contracted_reconstruction_fit": self.contracted_reconstruction_fit,
            "original_reconstruction_fit": self.original_reconstruction_fit,
            "must_group_role": self.must_group_role,
        }


@dataclass(frozen=True)
class CorePeripheryContraction:
    """Aggregate quotient graph and its original-node block mapping.

    Attributes
    ----------
    adjacency : ndarray of float, shape (b, b)
        Aggregate block adjacency ``S.T @ A @ S``.
    blocks : tuple of tuple of int
        Original-node indices in each block.
    node_to_block : ndarray of int, shape (N,)
        Block index for each original node.
    block_sizes : ndarray of float, shape (b,)
        Cardinality of each block.
    """

    adjacency: np.ndarray
    blocks: tuple[tuple[int, ...], ...]
    node_to_block: np.ndarray
    block_sizes: np.ndarray


def _validate_target(target: str) -> CorePeripheryTarget:
    if target not in {"contracted", "original"}:
        raise ValueError("target must be either 'contracted' or 'original'.")
    return target


def _as_symmetric_adjacency(A: np.ndarray | sp.spmatrix) -> np.ndarray:
    matrix = A.toarray() if sp.issparse(A) else np.asarray(A, dtype=float)
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("A must be a square adjacency matrix.")
    if matrix.shape[0] == 0:
        raise ValueError("A must contain at least one node.")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("A must contain only finite values.")
    if not np.allclose(matrix, matrix.T, rtol=1e-10, atol=1e-12):
        raise ValueError("A must be symmetric; directed/asymmetric inputs are not supported.")
    return matrix


def _validate_block_constraints(
    n_nodes: int,
    must_link: Sequence[tuple[int, int]] | None,
    must_group: Sequence[int] | None,
) -> tuple[list[tuple[int, int]], list[int]]:
    pairs = [] if must_link is None else [(int(i), int(j)) for i, j in must_link]
    group = [] if must_group is None else list(dict.fromkeys(int(i) for i in must_group))
    for source, target in pairs:
        if not (0 <= source < n_nodes and 0 <= target < n_nodes):
            raise ValueError("must_link contains a node index outside the adjacency matrix.")
    if any(not 0 <= node < n_nodes for node in group):
        raise ValueError("must_group contains a node index outside the adjacency matrix.")
    return pairs, group


def contract_core_periphery_adjacency(
    A: np.ndarray | sp.spmatrix,
    *,
    must_link: Sequence[tuple[int, int]] | None = None,
    must_group: Sequence[int] | None = None,
) -> CorePeripheryContraction:
    """Return the aggregate contraction ``B = S.T @ A @ S``.

    Parameters
    ----------
    A : ndarray or sparse matrix, shape (N, N)
        Finite symmetric adjacency or weight matrix.
    must_link : sequence of tuple of int, optional
        Node pairs merged into common detection blocks. Transitive closures are
        respected.
    must_group : sequence of int, optional
        Nodes merged into one generic core-periphery detection block.

    Returns
    -------
    CorePeripheryContraction
        Aggregate adjacency and deterministic block mappings.

    Notes
    -----
    Contracted row sums preserve the aggregate strength of each block, not the
    individual degrees of the nodes merged into that block. Diagonal entries
    are retained; for a loop-free undirected graph, ``B[p, p]`` is twice the
    total weight of internal edges in block ``p``. No density normalization is
    applied.
    """

    matrix = _as_symmetric_adjacency(A)
    pairs, group = _validate_block_constraints(matrix.shape[0], must_link, must_group)
    union_find = UnionFind(matrix.shape[0])
    for source, target in pairs:
        union_find.union(source, target)
    if len(group) > 1:
        representative = group[0]
        for node in group[1:]:
            union_find.union(representative, node)

    raw_blocks, component_map = union_find.components()
    blocks = tuple(tuple(block) for block in raw_blocks)
    node_to_block = np.fromiter(
        (component_map[node] for node in range(matrix.shape[0])),
        dtype=int,
        count=matrix.shape[0],
    )
    block_sizes = np.fromiter((len(block) for block in blocks), dtype=float)
    contracted = np.empty((len(blocks), len(blocks)), dtype=float)
    for p, block_p in enumerate(blocks):
        for q, block_q in enumerate(blocks):
            contracted[p, q] = float(matrix[np.ix_(block_p, block_q)].sum())

    return CorePeripheryContraction(
        adjacency=contracted,
        blocks=blocks,
        node_to_block=node_to_block,
        block_sizes=block_sizes,
    )


class _BlockPairObjective:
    """Pearson objective compressed to unique block-pair sufficient statistics."""

    def __init__(
        self,
        pairs_i: np.ndarray,
        pairs_j: np.ndarray,
        counts: np.ndarray,
        weight_sums: np.ndarray,
        weight_square_sum: float,
    ) -> None:
        self.pairs_i = pairs_i
        self.pairs_j = pairs_j
        self.counts = counts.astype(float, copy=False)
        self.weight_sums = weight_sums.astype(float, copy=False)
        self.sample_count = float(self.counts.sum())
        self.weight_sum = float(self.weight_sums.sum())
        self.weight_square_sum = float(weight_square_sum)

    def correlation(self, block_values: np.ndarray) -> float:
        values = np.asarray(block_values, dtype=float)[self.pairs_i, self.pairs_j]
        value_sum = float(np.dot(self.counts, values))
        value_square_sum = float(np.dot(self.counts, values * values))
        cross_sum = float(np.dot(self.weight_sums, values))
        if self.sample_count <= 1:
            return 0.0
        weight_ss = self.weight_square_sum - self.weight_sum**2 / self.sample_count
        value_ss = value_square_sum - value_sum**2 / self.sample_count
        if weight_ss <= 0.0 or value_ss <= 0.0:
            return 0.0
        covariance = cross_sum - self.weight_sum * value_sum / self.sample_count
        return float(covariance / np.sqrt(weight_ss * value_ss))


def _block_pair_objectives(
    A: np.ndarray,
    contraction: CorePeripheryContraction,
) -> tuple[_BlockPairObjective, _BlockPairObjective]:
    B = contraction.adjacency
    contracted_i, contracted_j = np.triu_indices(B.shape[0], k=0)
    contracted_weights = B[contracted_i, contracted_j]
    contracted_objective = _BlockPairObjective(
        contracted_i,
        contracted_j,
        np.ones(contracted_i.size, dtype=float),
        contracted_weights,
        float(np.dot(contracted_weights, contracted_weights)),
    )

    original_i: list[int] = []
    original_j: list[int] = []
    counts: list[float] = []
    weight_sums: list[float] = []
    weight_square_sum = 0.0
    for p, block_p in enumerate(contraction.blocks):
        for q in range(p, len(contraction.blocks)):
            block_q = contraction.blocks[q]
            if p == q:
                local_i, local_j = np.triu_indices(len(block_p), k=1)
                values = A[np.ix_(block_p, block_p)][local_i, local_j]
            else:
                values = A[np.ix_(block_p, block_q)].reshape(-1)
            if values.size == 0:
                continue
            original_i.append(p)
            original_j.append(q)
            counts.append(float(values.size))
            weight_sums.append(float(values.sum()))
            weight_square_sum += float(np.dot(values, values))

    original_objective = _BlockPairObjective(
        np.asarray(original_i, dtype=int),
        np.asarray(original_j, dtype=int),
        np.asarray(counts, dtype=float),
        np.asarray(weight_sums, dtype=float),
        weight_square_sum,
    )
    return contracted_objective, original_objective


def normalized_BE_score(
    A: np.ndarray,
    labels: np.ndarray,
    *,
    include_diagonal: bool = False,
) -> float:
    """Return the normalized binary Borgatti-Everett fit.

    The score is the Pearson correlation between adjacency values and the
    ideal pattern ``D[i, j] = labels[i] OR labels[j]``, with ``1`` denoting the
    core. This is statistical normalization, not degree, density, or block-size
    normalization.

    Parameters
    ----------
    A : ndarray of float, shape (N, N)
        Adjacency or weight matrix.
    labels : ndarray, shape (N,)
        Binary core membership labels.
    include_diagonal : bool, default=False
        Include ``i == j`` samples. Contracted-space scoring enables this so
        aggregated internal edges participate consistently.

    Returns
    -------
    float
        Pearson correlation, or ``0`` when either vector is constant or has
        fewer than two samples.
    """

    matrix = np.asarray(A, dtype=float)
    binary_labels = np.asarray(labels, dtype=bool)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("A must be a square adjacency matrix.")
    if binary_labels.shape != (matrix.shape[0],):
        raise ValueError("labels must contain one entry per adjacency-matrix node.")
    i, j = np.triu_indices(matrix.shape[0], k=0 if include_diagonal else 1)
    if i.size <= 1:
        return 0.0
    adjacency_values = matrix[i, j]
    pattern_values = (binary_labels[i] | binary_labels[j]).astype(float)
    adjacency_centered = adjacency_values - adjacency_values.mean()
    pattern_centered = pattern_values - pattern_values.mean()
    denominator = np.linalg.norm(adjacency_centered) * np.linalg.norm(pattern_centered)
    if denominator == 0.0:
        return 0.0
    return float(np.dot(adjacency_centered, pattern_centered) / denominator)


def find_core(
    A: np.ndarray,
    labels: np.ndarray,
    *,
    include_diagonal: bool = False,
) -> np.ndarray:
    """Choose the better core/periphery orientation of binary labels.

    Parameters
    ----------
    A : ndarray of float, shape (N, N)
        Matrix on which orientation is evaluated.
    labels : ndarray, shape (N,)
        Candidate binary labels with arbitrary orientation.
    include_diagonal : bool, default=False
        Whether diagonal samples participate in the BE score.

    Returns
    -------
    ndarray of int, shape (N,)
        Labels oriented so that ``1`` denotes the higher-scoring core.
    """

    labels = np.asarray(labels).copy()
    if labels.size == 0:
        return labels
    if np.max(labels) > 1:
        labels = labels - (np.max(labels) - 1)
    inverted = (labels == 0).astype(int)
    score = normalized_BE_score(A, labels, include_diagonal=include_diagonal)
    inverted_score = normalized_BE_score(A, inverted, include_diagonal=include_diagonal)
    return labels if score > inverted_score else inverted


def find_core_advanced(
    A: np.ndarray,
    labels: np.ndarray,
    *,
    include_diagonal: bool = False,
) -> tuple[np.ndarray, float]:
    """Choose the best BE split induced by candidate cluster labels.

    Each unique input label is considered as the periphery while all remaining
    labels form the core. The split with the greatest normalized BE score is
    returned.

    Parameters
    ----------
    A : ndarray of float, shape (N, N)
        Matrix on which the split is evaluated.
    labels : ndarray, shape (N,)
        One or more unoriented cluster labels.
    include_diagonal : bool, default=False
        Whether diagonal samples participate in the BE score.

    Returns
    -------
    best_labels : ndarray of int, shape (N,)
        Oriented binary labels where ``1`` denotes the core.
    best_score : float
        Normalized BE score of ``best_labels``.
    """

    labels = np.asarray(labels).reshape(-1)
    if labels.shape[0] != np.asarray(A).shape[0]:
        raise ValueError("labels must contain one entry per adjacency-matrix node.")
    best_score = -np.inf
    best_labels: np.ndarray | None = None
    for candidate_periphery in np.unique(labels):
        trial_labels = (labels != candidate_periphery).astype(int)
        score = normalized_BE_score(
            A,
            trial_labels,
            include_diagonal=include_diagonal,
        )
        if score > best_score:
            best_score = score
            best_labels = trial_labels
    if best_labels is None:
        return labels, float("-inf")
    return best_labels, float(best_score)


class EnhancedGeneticBE:
    """Binary genetic search for the BE objective with must-link blocks.

    Parameters
    ----------
    A : ndarray of float, shape (N, N)
        Symmetric adjacency or weight matrix.
    must_link : sequence of tuple of int, optional
        Node pairs constrained to one binary assignment.
    pop_size : int, default=50
        Number of candidate partitions.
    generations : int, default=100
        Number of evolutionary generations.
    init_mut_rate : float, default=0.1
        Initial per-block mutation probability, reduced over time.
    elitism_size : int, default=2
        Best candidates copied unchanged into the next generation.
    tournament_size : int, default=3
        Candidates sampled during parent selection.
    seed : int or None, default=42
        Local random-generator seed.
    """

    def __init__(
        self,
        A: np.ndarray,
        must_link: Sequence[tuple[int, int]] | None = None,
        pop_size: int = 50,
        generations: int = 100,
        init_mut_rate: float = 0.1,
        elitism_size: int = 2,
        tournament_size: int = 3,
        seed: int | None = 42,
    ) -> None:
        self.A = _as_symmetric_adjacency(A)
        self.rng = np.random.default_rng(seed)
        contraction = contract_core_periphery_adjacency(self.A, must_link=must_link)
        self.blocks = contraction.blocks
        self.node_to_block = contraction.node_to_block
        self.n_blocks = len(self.blocks)
        self.pop_size = int(pop_size)
        self.generations = int(generations)
        self.init_mut_rate = float(init_mut_rate)
        self.elitism_size = min(int(elitism_size), self.pop_size)
        self.tournament_size = int(tournament_size)
        if not 1 <= self.tournament_size <= self.pop_size:
            raise ValueError("tournament_size must be between 1 and pop_size.")

    def _fitness(self, block_labels: np.ndarray) -> float:
        """Pearson correlation between adjacency A and the labels."""
        labels = block_labels[self.node_to_block]
        return normalized_BE_score(self.A, labels)

    def _tournament(self, population: list[np.ndarray], fitness: list[float]) -> np.ndarray:
        """
        Random tournament-based selection.
        
        Parameters
        ----------
        population : list[np.ndarray]
            Entire population.
        fitness : list[float]
           Fitness score for each individual in population.
        
        Returns
        -------
        np.ndarray of int, shape (N,)
            Individual tournament winner.
        """
        indices = self.rng.choice(self.pop_size, self.tournament_size, replace=False)
        return population[int(indices[np.argmax([fitness[index] for index in indices])])]

    def run(self) -> tuple[dict[int, int], float]:
        """Run the binary genetic search.

        Returns
        -------
        labels : dict of int to int
            Original-node binary assignments, with ``1`` denoting the core.
        best_fit : float
            Best normalized BE score encountered.
        """

        population = [self.rng.integers(0, 2, self.n_blocks) for _ in range(self.pop_size)]
        best_labels = population[0].copy()
        best_fit = self._fitness(best_labels)
        for generation in range(self.generations):
            fitness = [self._fitness(individual) for individual in population]
            generation_best = int(np.argmax(fitness))
            if fitness[generation_best] > best_fit:
                best_fit = fitness[generation_best]
                best_labels = population[generation_best].copy()
            elite_indices = np.argsort(fitness)[-self.elitism_size :]
            new_population = [population[index].copy() for index in elite_indices]
            mutation_rate = self.init_mut_rate * (1 - generation / max(1, self.generations))
            while len(new_population) < self.pop_size:
                parent_1 = self._tournament(population, fitness)
                parent_2 = self._tournament(population, fitness)
                selector = self.rng.random(self.n_blocks) < 0.5
                child = np.where(selector, parent_1, parent_2)
                flips = self.rng.random(self.n_blocks) < mutation_rate
                child[flips] = 1 - child[flips]
                new_population.append(child)
            population = new_population
        node_labels = best_labels[self.node_to_block]
        return {node: int(label) for node, label in enumerate(node_labels)}, float(best_fit)


def _continuous_result(
    A: np.ndarray,
    contraction: CorePeripheryContraction,
    *,
    algorithm: str,
    target: CorePeripheryTarget,
    block_embedding: np.ndarray,
    block_scores: np.ndarray | None,
    contracted_reconstruction: np.ndarray,
    original_reconstruction: np.ndarray,
    spectral_rank: int = 1,
    eigenproblem: str | None = None,
    eigenvalues: np.ndarray | None = None,
    block_eigenvectors: np.ndarray | None = None,
) -> CorePeripheryResult:
    contracted_objective, original_objective = _block_pair_objectives(A, contraction)
    contracted_reconstruction_fit = contracted_objective.correlation(
        contracted_reconstruction
    )
    original_reconstruction_fit = original_objective.correlation(original_reconstruction)
    continuous_node_scores = (
        None if block_scores is None else block_scores[contraction.node_to_block].astype(float)
    )
    return CorePeripheryResult(
        algorithm=algorithm,
        target_space=target,
        blocks=contraction.blocks,
        node_to_block=contraction.node_to_block,
        block_sizes=contraction.block_sizes,
        contracted_adjacency=contraction.adjacency,
        block_embedding=np.asarray(block_embedding, dtype=float),
        block_scores=None if block_scores is None else np.asarray(block_scores, dtype=float),
        continuous_node_scores=continuous_node_scores,
        spectral_rank=spectral_rank,
        eigenproblem=eigenproblem,
        eigenvalues=eigenvalues,
        block_eigenvectors=block_eigenvectors,
        contracted_fit=contracted_reconstruction_fit,
        original_fit=original_reconstruction_fit,
        primary_fit=(
            contracted_reconstruction_fit
            if target == "contracted"
            else original_reconstruction_fit
        ),
        contracted_reconstruction_fit=contracted_reconstruction_fit,
        original_reconstruction_fit=original_reconstruction_fit,
    )


def _score_reconstruction_matrices(
    block_scores: np.ndarray,
    contraction: CorePeripheryContraction,
    target: CorePeripheryTarget,
) -> tuple[np.ndarray, np.ndarray]:
    original_reconstruction = np.outer(block_scores, block_scores)
    if target == "contracted":
        return original_reconstruction, original_reconstruction
    sizes = contraction.block_sizes
    contracted_reconstruction = sizes[:, None] * original_reconstruction * sizes[None, :]
    return contracted_reconstruction, original_reconstruction


class FullContinuousGeneticBE:
    """Continuous genetic BE search with one gene per constraint block.

    The genome is block-level from initialization onward; node-level values are
    expanded only when results are reported. The selected target determines
    whether fitness samples aggregate contracted pairs or all original
    off-diagonal pairs, including zero-valued nonedges.

    Parameters
    ----------
    A : ndarray or sparse matrix, shape (N, N)
        Symmetric adjacency or weight matrix.
    must_link : sequence of tuple of int, optional
        Pairwise block-equality constraints.
    must_group : sequence of int, optional
        Nodes placed in one generic core-periphery detection block.
    target : {"contracted", "original"}, default="contracted"
        Space whose continuous reconstruction fit is optimized.
    pop_size : int, default=50
        Number of block-level genomes.
    generations : int, default=100
        Number of evolutionary generations.
    crossover_rate : float, default=0.8
        Probability of blend crossover.
    mutation_rate : float, default=0.1
        Per-gene Gaussian mutation probability.
    tournament_size : int, default=3
        Candidates sampled during parent selection.
    gene_init_scale : float, default=1.0
        Upper bound for initialized and clipped continuous scores.
    seed : int or None, default=42
        Local random-generator seed.
    """

    def __init__(
        self,
        A: np.ndarray | sp.spmatrix,
        must_link: Sequence[tuple[int, int]] | None = None,
        must_group: Sequence[int] | None = None,
        *,
        target: CorePeripheryTarget = "contracted",
        pop_size: int = 50,
        generations: int = 100,
        crossover_rate: float = 0.8,
        mutation_rate: float = 0.1,
        tournament_size: int = 3,
        gene_init_scale: float = 1.0,
        seed: int | None = 42,
    ) -> None:
        self.rng = np.random.default_rng(seed)
        self.A = _as_symmetric_adjacency(A)
        self.target = _validate_target(target)
        self.contraction = contract_core_periphery_adjacency(
            self.A,
            must_link=must_link,
            must_group=must_group,
        )
        self.blocks = self.contraction.blocks
        self.n_blocks = len(self.blocks)
        self.pop_size = int(pop_size)
        self.generations = int(generations)
        self.crossover_rate = float(crossover_rate)
        self.mutation_rate = float(mutation_rate)
        self.tournament_size = int(tournament_size)
        self.scale = float(gene_init_scale)
        if self.pop_size < 1:
            raise ValueError("pop_size must be positive.")
        if not 1 <= self.tournament_size <= self.pop_size:
            raise ValueError("tournament_size must be between 1 and pop_size.")
        contracted_objective, original_objective = _block_pair_objectives(
            self.A,
            self.contraction,
        )
        self._objective = (
            contracted_objective if self.target == "contracted" else original_objective
        )

    def _init_population(self) -> list[np.ndarray]:
        """Initialize population with real vectors in [0,scale], one per individual."""
        return [self.rng.random(self.n_blocks) * self.scale for _ in range(self.pop_size)]

    def _fitness(self, block_scores: np.ndarray) -> float:
        """Pearson correlation score."""
        return self._objective.correlation(np.outer(block_scores, block_scores))

    def _tournament(self, population: list[np.ndarray], fitness: list[float]) -> np.ndarray:
        """
        Random tournament-based selection.
        
        Parameters
        ----------
        population : list[np.ndarray]
            Entire population.
        fitness : list[float]
           Fitness score for each individual in population.
        
        Returns
        -------
        np.ndarray of int, shape (N,)
            Individual tournament winner.
        """
        indices = self.rng.choice(len(population), self.tournament_size, replace=False)
        winner = indices[np.argmax([fitness[index] for index in indices])]
        return population[int(winner)]

    def _crossover(
        self,
        parent_1: np.ndarray,
        parent_2: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Blend crossover: c' = alpha * parent1 + (1-alpha)*parent2"""
        if self.rng.random() > self.crossover_rate:
            return parent_1.copy(), parent_2.copy()
        alpha = self.rng.random(self.n_blocks)
        return (
            alpha * parent_1 + (1.0 - alpha) * parent_2,
            alpha * parent_2 + (1.0 - alpha) * parent_1,
        )

    def _mutate(self, scores: np.ndarray) -> np.ndarray:
        """Gaussian mutation around each gene."""
        selected = self.rng.random(self.n_blocks) < self.mutation_rate
        noise = self.rng.normal(scale=self.scale * 0.1, size=self.n_blocks)
        scores[selected] += noise[selected]
        np.clip(scores, 0.0, self.scale, out=scores)
        return scores

    def _make_result(self, block_scores: np.ndarray, *, algorithm: str) -> CorePeripheryResult:
        contracted_reconstruction, original_reconstruction = _score_reconstruction_matrices(
            block_scores,
            self.contraction,
            self.target,
        )
        return _continuous_result(
            self.A,
            self.contraction,
            algorithm=algorithm,
            target=self.target,
            block_embedding=block_scores[:, None],
            block_scores=block_scores,
            contracted_reconstruction=contracted_reconstruction,
            original_reconstruction=original_reconstruction,
        )

    def run(self) -> CorePeripheryResult:
        """Run the genetic search in block space.

        Returns
        -------
        CorePeripheryResult
            Continuous block and node scores with fit diagnostics. Discrete
            labels are added by the NLBNP detection wrapper.
        """

        population = self._init_population()
        fitness = [self._fitness(individual) for individual in population]
        best_index = int(np.argmax(fitness))
        best_scores = population[best_index].copy()
        best_fit = fitness[best_index]
        for _ in range(self.generations):
            new_population: list[np.ndarray] = []
            while len(new_population) < self.pop_size:
                parent_1 = self._tournament(population, fitness)
                parent_2 = self._tournament(population, fitness)
                child_1, child_2 = self._crossover(parent_1, parent_2)
                new_population.append(self._mutate(child_1))
                if len(new_population) < self.pop_size:
                    new_population.append(self._mutate(child_2))
            population = new_population
            fitness = [self._fitness(individual) for individual in population]
            generation_best = int(np.argmax(fitness))
            if fitness[generation_best] > best_fit:
                best_fit = fitness[generation_best]
                best_scores = population[generation_best].copy()
        return self._make_result(best_scores, algorithm="GA")

    def run_de(self, F: float = 0.7, CR: float = 0.9) -> CorePeripheryResult:
        """Run differential evolution using the same block-level objective.

        Parameters
        ----------
        F : float, default=0.7
            Differential mutation factor.
        CR : float, default=0.9
            Binomial crossover probability.

        Returns
        -------
        CorePeripheryResult
            Continuous block and node scores with fit diagnostics.
        """

        if self.pop_size < 4:
            raise ValueError("run_de requires pop_size >= 4.")
        population = self._init_population()
        fitness = [self._fitness(individual) for individual in population]
        best_index = int(np.argmax(fitness))
        best_scores = population[best_index].copy()
        best_fit = fitness[best_index]
        for _ in range(self.generations):
            for index in range(self.pop_size):
                choices = np.delete(np.arange(self.pop_size), index)
                first, second, third = self.rng.choice(choices, 3, replace=False)
                mutant = (
                    population[int(first)]
                    + F * (population[int(second)] - population[int(third)])
                )
                np.clip(mutant, 0.0, self.scale, out=mutant)
                trial = population[index].copy()
                crossover = self.rng.random(self.n_blocks) < CR
                crossover[self.rng.integers(self.n_blocks)] = True
                trial[crossover] = mutant[crossover]
                trial_fit = self._fitness(trial)
                if trial_fit > fitness[index]:
                    population[index] = trial
                    fitness[index] = trial_fit
                    if trial_fit > best_fit:
                        best_fit = trial_fit
                        best_scores = trial.copy()
        return self._make_result(best_scores, algorithm="DE")


def detect_continuous_KL(
    A: np.ndarray | sp.spmatrix,
    must_link: Sequence[tuple[int, int]] | None = None,
    must_group: Sequence[int] | None = None,
    *,
    target: CorePeripheryTarget = "contracted",
    max_iter: int = 100,
    seed: int | None = 42,
) -> CorePeripheryResult:
    """Optimize continuous BE fit by blockwise coordinate updates.

    Parameters
    ----------
    A : ndarray or sparse matrix, shape (N, N)
        Symmetric adjacency or weight matrix.
    must_link : sequence of tuple of int, optional
        Pairwise block-equality constraints.
    must_group : sequence of int, optional
        Nodes placed in one generic core-periphery detection block.
    target : {"contracted", "original"}, default="contracted"
        For ``"contracted"``, optimize over unique entries of the aggregate
        block adjacency including its diagonal. For ``"original"``, optimize
        over every original pair ``i < j``, including zero nonedges.
    max_iter : int, default=100
        Maximum complete passes over the block coordinates.
    seed : int or None, default=42
        Local random-generator seed.

    Returns
    -------
    CorePeripheryResult
        Continuous block and expanded node scores with named fits. Discrete
        labels are added by the NLBNP detection wrapper.

    Notes
    -----
    The implementation is a continuous block-coordinate search rather than the classical discrete
    graph-bisection form of Kernighan-Lin.
    """

    matrix = _as_symmetric_adjacency(A)
    selected_target = _validate_target(target)
    contraction = contract_core_periphery_adjacency(
        matrix,
        must_link=must_link,
        must_group=must_group,
    )
    contracted_objective, original_objective = _block_pair_objectives(matrix, contraction)
    objective = contracted_objective if selected_target == "contracted" else original_objective
    block_scores = np.random.default_rng(seed).random(len(contraction.blocks))

    def fitness(scores: np.ndarray) -> float:
        return objective.correlation(np.outer(scores, scores))

    best_fit = fitness(block_scores)
    for _ in range(max_iter):
        improved = False
        for block_index in range(len(contraction.blocks)):
            low, high = 0.0, 1.0
            for _ in range(20):
                left = low + 0.382 * (high - low)
                right = low + 0.618 * (high - low)
                left_scores = block_scores.copy()
                right_scores = block_scores.copy()
                left_scores[block_index] = left
                right_scores[block_index] = right
                if fitness(left_scores) < fitness(right_scores):
                    low = left
                else:
                    high = right
            candidate = block_scores.copy()
            candidate[block_index] = 0.5 * (low + high)
            candidate_fit = fitness(candidate)
            if candidate_fit > best_fit + 1e-12:
                block_scores = candidate
                best_fit = candidate_fit
                improved = True
        if not improved:
            break

    contracted_reconstruction, original_reconstruction = _score_reconstruction_matrices(
        block_scores,
        contraction,
        selected_target,
    )
    return _continuous_result(
        matrix,
        contraction,
        algorithm="KL",
        target=selected_target,
        block_embedding=block_scores[:, None],
        block_scores=block_scores,
        contracted_reconstruction=contracted_reconstruction,
        original_reconstruction=original_reconstruction,
    )


def _leading_eigenpairs(operator: np.ndarray, rank: int) -> tuple[np.ndarray, np.ndarray]:
    n = operator.shape[0]
    if rank not in {1, 2}:
        raise ValueError("spectral_rank must be 1 or 2.")
    if rank > n:
        raise ValueError(f"spectral_rank={rank} requires at least {rank} contracted blocks.")
    if n <= 50 or rank == n:
        eigenvalues, eigenvectors = np.linalg.eigh(operator)
    else:
        which = "LA" if rank == 1 else "LM"
        eigenvalues, eigenvectors = eigsh(operator, k=rank, which=which)
    if rank == 1:
        selected = np.array([int(np.argmax(eigenvalues))])
    else:
        selected = np.argsort(np.abs(eigenvalues))[-rank:][::-1]
    return eigenvalues[selected], eigenvectors[:, selected]


def spectral_continuous_cp_detection(
    A: np.ndarray | sp.spmatrix,
    must_link: Sequence[tuple[int, int]] | None = None,
    must_group: Sequence[int] | None = None,
    *,
    target: CorePeripheryTarget = "contracted",
    spectral_rank: int = 1,
    normalize: bool = True,
) -> CorePeripheryResult:
    """Detect block-level spectral structure in contracted or original space.

    Parameters
    ----------
    A : ndarray or sparse matrix, shape (N, N)
        Symmetric adjacency or weight matrix.
    must_link : sequence of tuple of int, optional
        Pairwise block-equality constraints.
    must_group : sequence of int, optional
        Nodes placed in one generic core-periphery detection block.
    target : {"contracted", "original"}, default="contracted"
        ``"contracted"`` detects structure among aggregate block entities;
        ``"original"`` approximates the original graph subject to blockwise
        equal coreness.
    spectral_rank : {1, 2}, default=1
        Number of eigenpairs retained. Rank two is intended for subsequent
        two-component Gaussian-mixture clustering.
    normalize : bool, default=True
        Min-max normalize rank-one scalar scores. It does not alter the
        eigenvectors, eigenvalues, embedding, or reconstruction fits.

    Returns
    -------
    CorePeripheryResult
        Block eigensystem, clustering embedding, expanded rank-one scores when
        available, and reconstruction fits in both spaces.

    Notes
    -----
    ``target="contracted"`` solves ``B g = lambda g``. The non-default
    ``target="original"`` solves ``B g = lambda N g``, where ``N`` contains
    block sizes, using the symmetric operator ``N^-1/2 B N^-1/2``. Rank two
    uses the two largest-magnitude eigenpairs and adjacency spectral embedding
    ``G |Lambda|^1/2``.
    """

    matrix = _as_symmetric_adjacency(A)
    selected_target = _validate_target(target)
    contraction = contract_core_periphery_adjacency(
        matrix,
        must_link=must_link,
        must_group=must_group,
    )
    B = contraction.adjacency
    sizes = contraction.block_sizes
    if selected_target == "contracted":
        eigenvalues, eigenvectors = _leading_eigenpairs(B, spectral_rank)
        contracted_reconstruction = (eigenvectors * eigenvalues[None, :]) @ eigenvectors.T
        original_reconstruction = contracted_reconstruction
    else:
        scale = 1.0 / np.sqrt(sizes)
        operator = scale[:, None] * B * scale[None, :]
        eigenvalues, transformed_vectors = _leading_eigenpairs(operator, spectral_rank)
        eigenvectors = scale[:, None] * transformed_vectors
        original_reconstruction = (eigenvectors * eigenvalues[None, :]) @ eigenvectors.T
        contracted_reconstruction = sizes[:, None] * original_reconstruction * sizes[None, :]

    embedding = eigenvectors * np.sqrt(np.abs(eigenvalues))[None, :]
    block_scores: np.ndarray | None = None
    if spectral_rank == 1:
        block_scores = np.abs(eigenvectors[:, 0])
        if normalize:
            span = float(np.ptp(block_scores))
            block_scores = (
                (block_scores - block_scores.min()) / span
                if span > 1e-12
                else np.zeros_like(block_scores)
            )

    return _continuous_result(
        matrix,
        contraction,
        algorithm="SPEC",
        target=selected_target,
        block_embedding=embedding,
        block_scores=block_scores,
        contracted_reconstruction=contracted_reconstruction,
        original_reconstruction=original_reconstruction,
        spectral_rank=spectral_rank,
        eigenproblem="ordinary" if selected_target == "contracted" else "generalized",
        eigenvalues=eigenvalues,
        block_eigenvectors=eigenvectors,
    )


def _to_undirected_graph(adj: AdjLike) -> nx.Graph:
    """Convert supported adjacency inputs to an undirected NetworkX graph.

    Edge weights are retained and self-loops are removed because this helper
    is used only for final connected-component extraction, not CP scoring.
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
    """Return a boolean mask whose true entries identify core nodes.

    One-dimensional inputs treat any nonzero value as core. Two-dimensional
    inputs must be an ``(N, 2)`` assignment matrix, with ``core_is`` selecting
    the core column.
    """

    partition = np.asarray(core_periphery)
    if partition.ndim == 1:
        core_mask = partition if partition.dtype == bool else partition != 0
    elif partition.ndim == 2 and partition.shape[1] == 2:
        if core_is not in (0, 1):
            raise ValueError("core_is must be 0 or 1 for a 2-column partition matrix.")
        core_mask = partition[:, core_is] > 0.5
    else:
        raise ValueError("core_periphery must be length-N vector or (N,2) partition matrix.")
    return core_mask.astype(bool)


def partition_periphery_components(
    adj: AdjLike,
    core_periphery: np.ndarray | list[int],
    *,
    core_is: int | None = 0,
    must_link: Sequence[tuple[int, int]] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return one core community and connected periphery communities.

    Parameters
    ----------
    adj : ndarray, sparse matrix, or NetworkX graph
        Graph used to determine connectivity after core removal. Self-loops do
        not affect the component calculation.
    core_periphery : ndarray or list
        Length-``N`` core indicator, or an ``(N, 2)`` assignment matrix.
    core_is : {0, 1} or None, default=0
        Core column for a two-dimensional input; ignored for vectors.
    must_link : sequence of tuple of int, optional
        Original-node pairs that must share a final community. Periphery pairs
        are added as virtual component edges; core pairs already share
        community zero. A pair crossing the detected boundary is rejected.

    Returns
    -------
    labels : ndarray of int, shape (N,)
        Community zero contains all core nodes. Communities ``1..K`` are the
        deterministic connected components of the remaining periphery.
    info : dict
        Counts, periphery component sets, and original-node index arrays for
        every returned community.

    Notes
    -----
    Only ``must_link`` pairs are added as virtual edges for the final periphery
    split. A ``must_group`` detection constraint must therefore not be passed
    here unless it independently represents final-community membership.
    """

    graph = _to_undirected_graph(adj)
    n_nodes = graph.number_of_nodes()
    if list(graph.nodes()) != list(range(n_nodes)):
        mapping = {node: index for index, node in enumerate(graph.nodes())}
        graph = nx.relabel_nodes(graph, mapping, copy=True)

    core_mask = _core_mask_from_partition(core_periphery, core_is=core_is)
    if core_mask.shape[0] != n_nodes:
        raise ValueError(
            f"core_periphery length {core_mask.shape[0]} != number of nodes {n_nodes}."
        )

    normalized_links = set()
    for source, target in must_link or []:
        source, target = int(source), int(target)
        if not (0 <= source < n_nodes and 0 <= target < n_nodes):
            raise ValueError(f"must_link pair {(source, target)} contains an invalid node index.")
        if core_mask[source] != core_mask[target]:
            raise ValueError(
                f"must_link pair {(source, target)} crosses the detected core-periphery boundary."
            )
        if source != target:
            normalized_links.add(tuple(sorted((source, target))))

    core_nodes = np.flatnonzero(core_mask).tolist()
    periphery_nodes = np.flatnonzero(~core_mask).tolist()
    periphery_graph = graph.subgraph(periphery_nodes).copy()
    periphery_graph.add_edges_from(
        pair for pair in normalized_links if not core_mask[pair[0]]
    )
    components = sorted(
        nx.connected_components(periphery_graph),
        key=lambda component: min(component),
    )

    labels = np.full(n_nodes, -1, dtype=int)
    if core_nodes:
        labels[core_nodes] = 0
    community_node_indices: list[np.ndarray] = [np.asarray(core_nodes, dtype=int)]
    for community, component in enumerate(components, start=1):
        indices = np.asarray(sorted(component), dtype=int)
        labels[indices] = community
        community_node_indices.append(indices)

    return labels, {
        "n_core": len(core_nodes),
        "n_periphery": len(periphery_nodes),
        "components": components,
        "community_node_indices": community_node_indices,
    }
