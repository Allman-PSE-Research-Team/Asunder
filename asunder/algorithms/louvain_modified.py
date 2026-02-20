"""Modified Louvain implementation."""

from typing import Dict, Optional, Tuple

import numpy as np
from scipy import sparse


def to_csr(A) -> sparse.csr_matrix:
    """Convert array-like or sparse to CSR[float]."""
    if sparse.isspmatrix_csr(A):
        return A.astype(float, copy=False)
    if sparse.issparse(A):
        return A.tocsr().astype(float, copy=False)
    return sparse.csr_matrix(np.asarray(A), dtype=float)

def directed_to_undirected(A: sparse.csr_matrix, average: bool = True) -> sparse.csr_matrix:
    """Symmetrize adjacency. If average=True, use 0.5*(A + A.T); else sum."""
    S = A + A.T
    return (0.5 * S) if average else S

def symmetrize(A: sparse.csr_matrix, average: bool = True) -> sparse.csr_matrix:
    """Symmetrize adjacency. If average=True, use 0.5*(A + A.T); else sum."""
    S = A + A.T
    return (0.5 * S) if average else S

def degrees(A: sparse.csr_matrix) -> np.ndarray:
    """Compute node degrees (sum of row)."""
    return np.asarray(A.sum(axis=1)).ravel()

def total_weight(A: sparse.csr_matrix) -> float:
    """Compute total edge weight m for undirected graphs."""
    return float(A.sum()) / 2.0

def get_membership(labels: np.ndarray, n_labels: Optional[int] = None) -> sparse.csr_matrix:
    """Build a CSR membership indicator matrix of shape (n_nodes, n_labels)."""
    labels = np.asarray(labels, dtype=int)
    n = labels.shape[0]
    if n_labels is None:
        if (labels >= 0).any():
            n_labels = int(labels[labels >= 0].max()) + 1
        else:
            n_labels = 0
    if n_labels == 0:
        return sparse.csr_matrix((n, 0), dtype=float)

    rows = np.flatnonzero(labels >= 0)
    cols = labels[rows]
    data = np.ones_like(rows, dtype=float)
    return sparse.csr_matrix((data, (rows, cols)), shape=(n, n_labels))

def normalize_rows_csr(X: sparse.csr_matrix) -> sparse.csr_matrix:
    """Row L1-normalization for CSR. Zero rows remain zero."""
    if X.shape[0] == 0:
        return X
    row_sums = np.asarray(X.sum(axis=1)).ravel()
    inv = np.zeros_like(row_sums, dtype=float)
    nz = row_sums > 0
    inv[nz] = 1.0 / row_sums[nz]
    return X.multiply(inv[:, None])

def reindex_consecutive(labels: np.ndarray) -> np.ndarray:
    """Map arbitrary non-negative labels to 0..k-1 preserving order of first occurrence."""
    uniq, new = np.unique(labels, return_inverse=True)
    return new

class ModifiedLouvain:
    """
    Full Louvain with resolution parameter and multi-level aggregation.
    Also exposes a one-level 'modified fast unfolding' path with dual-adjusted modularity.

    Parameters
    ----------
    resolution : float
        Modularity resolution γ.
    tol_optimization : float
        Minimum modularity increase to keep optimizing at a level.
    tol_aggregation : float
        Minimum modularity increase to perform another aggregation.
    n_aggregations : int
        Maximum number of aggregation levels (-1 means unlimited).
    random_state : Optional[int]
        Seed for node shuffling inside local moving.
    symmetrize_average : bool
        If True, symmetrize directed inputs by averaging; otherwise by summation.

    Attributes
    ----------
    labels_ : np.ndarray
        Final labels for original nodes (reindexed to 0..k-1).
    probs_ : scipy.sparse.csr_matrix
        Row-normalized neighbor-mass per cluster on the ORIGINAL graph.
    aggregate_ : scipy.sparse.csr_matrix
        Last-level aggregated adjacency.
    modularity_ : float
        Final modularity value (Newman-Girvan with resolution γ).
    obj_val_ : Optional[float]
        Subproblem objective.
    """

    def __init__(self,
                 resolution: float = 1.0,
                 tol_optimization: float = 1e-3,
                 tol_aggregation: float = 1e-3,
                 n_aggregations: int = -1,
                 random_state: Optional[int] = None,
                 symmetrize_average: bool = True):
        self.resolution = float(resolution)
        self.tol_optimization = float(tol_optimization)
        self.tol_aggregation = float(tol_aggregation)
        self.n_aggregations = int(n_aggregations)
        self.random_state = int(random_state) if random_state is not None else None
        self.symmetrize_average = bool(symmetrize_average)

        self.labels_: Optional[np.ndarray] = None
        self.probs_: Optional[sparse.csr_matrix] = None
        self.aggregate_: Optional[sparse.csr_matrix] = None
        self.modularity_: Optional[float] = None
        self.obj_val_: Optional[float] = None

    # ----------------------------full Louvain ---------------------------------
    def fit(self,
            input_matrix,
            duals: Dict[str, np.ndarray | float],
            a: Optional[np.ndarray] = None,
            mprime: Optional[float] = None) -> "ModifiedLouvain":
        """
        Run multi-level Louvain using dual-adjusted objective.

        Arguments
        ---------
        input_matrix : array-like or sparse
            Square adjacency. Directed inputs are symmetrized.
        duals : dict
            Keys may include:
              - tau_dual, pi_dual, gamma_dual, core_tau_dual : (n,n) arrays, combined symmetrically into S.
              - mu_dual, core_mu_dual : floats, subtracted at the end from the objective.
            Missing entries are treated as zeros.
        a : Optional[np.ndarray]
            Degree-like vector; defaults to row sums of the symmetrized adjacency.
        mprime : Optional[float]
            Sum of degrees; defaults to sum(a). For undirected graphs, mprime = 2m.

        Returns
        -------
        self
        """
        rng = np.random.RandomState(self.random_state)

        A0 = to_csr(input_matrix)
        A0 = symmetrize(A0, average=self.symmetrize_average)
        n0 = A0.shape[0]

        if a is None:
            a = degrees(A0)
        if mprime is None:
            mprime = float(a.sum())

        S0 = self._assemble_S(duals, n0)

        # Cumulative labels for original nodes
        labels_cum = np.arange(n0, dtype=int)

        # Level state
        A = A0
        S = S0
        level = 0
        prev_obj = -np.inf

        while True:
            # Local moving at current level
            labels_level, obj_level = self._local_move_with_duals(
                A=A, S=S, a=degrees(A), mprime=float(degrees(A).sum()),  # keep m' consistent per level
                gamma=self.resolution, rng=rng
            )

            # Update cumulative labels
            labels_cum = labels_level[labels_cum]

            # Aggregate to next level
            M = get_membership(labels_level)
            A_agg = (M.T @ A) @ M
            S_agg = M.T @ S @ M

            level += 1
            gain = obj_level - prev_obj

            if (self.n_aggregations >= 0 and level >= self.n_aggregations) or (gain <= self.tol_aggregation):
                self.aggregate_ = A_agg
                break

            A, S = A_agg, S_agg
            prev_obj = obj_level

        # Final labels on original nodes
        final_labels = reindex_consecutive(labels_cum)
        self.labels_ = final_labels

        # Final objective value on original nodes
        M_final = get_membership(final_labels)
        S_final = self._assemble_S(duals, n0)  # same shape as original
        Mmat = self._build_M(A0, degrees(A0), float(degrees(A0).sum()), self.resolution, S_final)
        z_block = self._block_sum(Mmat, M_final)  # sum_{i,j in same cluster} M_ij

        self.obj_val_ = z_block

        # Probabilities on ORIGINAL graph
        self.probs_ = normalize_rows_csr(A0 @ M_final)
        return self

    def fit_standard_louvain(self, input_matrix) -> "ModifiedLouvain":
        """
        Run multi-level Louvain on an undirected weighted graph.
        Returns self with labels_ and probs_ defined on original nodes.
        """
        rng = np.random.RandomState(self.random_state)
        A0 = to_csr(input_matrix)
        A0 = symmetrize(A0, average=self.symmetrize_average)

        # Cumulative labels for original nodes
        n0 = A0.shape[0]
        labels_cumulative = np.arange(n0, dtype=int)

        # Current level graph
        A = A0
        level = 0
        prev_Q = -1.0
        while True:
            # Local moving on current level
            labels_level, Q = self._local_move(A, rng)
            # Update cumulative labels for original nodes
            labels_cumulative = labels_level[labels_cumulative]
            # Check aggregation criterion
            gain = Q - prev_Q
            prev_Q = Q

            # Aggregate graph
            A_agg = self._aggregate_graph(A, labels_level)
            level += 1

            if (self.n_aggregations >= 0 and level >= self.n_aggregations) or (gain <= self.tol_aggregation):
                # Stop; finalize on original nodes
                self.aggregate_ = A_agg
                break
            A = A_agg

        # Final reindex for original nodes
        final_labels = reindex_consecutive(labels_cumulative)
        self.labels_ = final_labels
        self.modularity_ = prev_Q
        # Probabilities computed on ORIGINAL adjacency
        M = get_membership(final_labels)
        self.probs_ = normalize_rows_csr(A0 @ M)
        return self

    def predict(self) -> np.ndarray:
        if self.labels_ is None:
            raise RuntimeError("Model not fitted.")
        return self.labels_

    def transform(self) -> sparse.csr_matrix:
        if self.probs_ is None:
            raise RuntimeError("Model not fitted.")
        return self.probs_

    def predict_proba(self) -> np.ndarray:
        return self.transform().toarray()

    # --------------------- public API: modified one-level --------------------

    def fit_modified_one_level(self,
                               input_matrix,
                               duals: Dict[str, np.ndarray],
                               a: Optional[np.ndarray] = None,
                               m: Optional[float] = None,
                               max_iter: int = 100,
                               tol: float = 1e-6) -> "ModifiedLouvain":
        """
        Run a single-level greedy improvement using a dual-adjusted modularity matrix.
        Sets labels_ and probs_ on original nodes, and obj_val_ if dual scalars provided.
        """
        rng = np.random.RandomState(self.random_state)
        A = to_csr(input_matrix)
        A = symmetrize(A, average=self.symmetrize_average)

        if a is None:
            a = degrees(A)
        if m is None:
            m = float(a.sum())

        mm = self._build_modified_modularity(A, a, m, duals)
        labels = self._one_level_fast_unfolding(A, mm, rng, max_iter=max_iter, tol=tol)

        self.labels_ = reindex_consecutive(labels)
        M = get_membership(self.labels_)
        self.probs_ = normalize_rows_csr(A @ M)

        self.obj_val_ = self._block_sum(mm, M)
        self.aggregate_ = None
        self.modularity_ = None
        return self

    # -------------------------- local moving --------------------------------

    def _local_move(self, A: sparse.csr_matrix, rng: np.random.RandomState) -> Tuple[np.ndarray, float]:
        """
        Local moving phase on a single level using the standard Louvain gain with resolution.
        Returns labels for this level and the resulting modularity.
        """
        n = A.shape[0]
        k = degrees(A)
        m = total_weight(A)
        gamma = self.resolution

        labels = np.arange(n, dtype=int)
        tot = k.copy()  # Σ_tot per community
        moved = True

        nodes = np.arange(n, dtype=int)

        while moved:
            moved = False
            rng.shuffle(nodes)
            for i in nodes:
                c_i = labels[i]
                k_i = k[i]

                # Remove i from its community
                tot[c_i] -= k_i
                labels[i] = -1  # temporary mark to avoid counting itself

                # Accumulate weights from i to neighboring communities
                start, end = A.indptr[i], A.indptr[i + 1]
                neigh = A.indices[start:end]
                w = A.data[start:end]
                neigh_comms = {}
                for j, wij in zip(neigh, w):
                    cj = labels[j]
                    if cj == -1:
                        cj = c_i
                    neigh_comms[cj] = neigh_comms.get(cj, 0.0) + wij

                # Find best target community
                best_c = c_i
                best_gain = 0.0
                for c, k_i_in in neigh_comms.items():
                    gain = k_i_in - gamma * k_i * tot[c] / (2.0 * m)
                    if gain > best_gain + 1e-14:
                        best_gain = gain
                        best_c = c

                # Insert i into best community
                labels[i] = best_c
                tot[best_c] += k_i
                if best_c != c_i:
                    moved = True

            # Stop early if modularity improvement is tiny
            Q = self._modularity(A, labels, k, m, gamma)
            if Q < 0:  # guard against numerical anomalies
                break
            if moved is False or self.tol_optimization <= 0:
                break
            # Optionally could check per-iteration ΔQ here; using node-move flag and final Q suffices.

        # Reindex labels of this level
        labels = reindex_consecutive(labels)
        Q = self._modularity(A, labels, k, m, gamma)
        return labels, Q

    # ----------------------- aggregation and metrics -------------------------

    @staticmethod
    def _aggregate_graph(A: sparse.csr_matrix, labels: np.ndarray) -> sparse.csr_matrix:
        """Aggregate adjacency: A' = M^T A M."""
        M = get_membership(labels)
        return (M.T @ A) @ M

    @staticmethod
    def _modularity(A: sparse.csr_matrix, labels: np.ndarray, k: np.ndarray, m: float, gamma: float) -> float:
        """
        Compute Newman-Girvan modularity with resolution γ:
            Q = (1/(2m)) * sum_{i,j} [A_ij - γ k_i k_j / (2m)] [c_i==c_j]
              = sum_c [ (sum_in_c)/(2m) - γ (tot_c/(2m))^2 ]
        sum_in_c is computed in doubled form (each undirected edge counted twice).
        """
        # sum_in (doubled)
        sum_in_dbl = 0.0
        for i in range(A.shape[0]):
            start, end = A.indptr[i], A.indptr[i + 1]
            j_idx = A.indices[start:end]
            w = A.data[start:end]
            mask = (labels[j_idx] == labels[i])
            sum_in_dbl += w[mask].sum()

        # tot per community
        n_labels = int(labels.max()) + 1 if labels.size else 0
        tot = np.zeros(n_labels, dtype=float)
        for c in range(n_labels):
            tot[c] = k[labels == c].sum()

        Q = (sum_in_dbl / (2.0 * m)) - gamma * np.sum((tot / (2.0 * m)) ** 2)
        return float(Q)

    # ----------------------- modified one-level internals --------------------

    @staticmethod
    def _build_modified_modularity(
        A: sparse.csr_matrix,
        a: np.ndarray,
        m: float,
        duals: Dict[str, np.ndarray],
    ) -> np.ndarray:
        """
        Construct dense modified modularity matrix:

            mm = (A/m - (a a^T)/(m^2))
                 - 0.5*(tau + tau^T)
                 - 0.5*(pi + pi^T)
                 - 0.5*(gamma + gamma^T)
                 - 0.5*(core_tau + core_tau^T)
        """
        A_dense = A.toarray().astype(float, copy=False)

        mm = (A_dense / m) - np.outer(a, a) / (m * m)

        def sym_pull(name: str) -> np.ndarray:
            X = duals.get(name, None)
            if X is None:
                return 0.0
            X = np.asarray(X, dtype=float)
            return 0.5 * (X + X.T)

        mm -= sym_pull("tau_dual")
        mm -= sym_pull("pi_dual")
        mm -= sym_pull("gamma_dual")
        mm -= sym_pull("core_tau_dual")
        return mm

    @staticmethod
    def _one_level_fast_unfolding(
        A: sparse.csr_matrix,
        mm: np.ndarray,
        rng: np.random.RandomState,
        max_iter: int = 100,
        tol: float = 1e-6,
    ) -> np.ndarray:
        """
        Greedy local improvement using mm for scoring.
        Candidate communities are taken from adjacency neighbors for efficiency.
        """
        n = A.shape[0]
        labels = np.arange(n, dtype=int)
        communities = {i: {i} for i in range(n)}

        indptr = A.indptr
        indices = A.indices
        nodes = np.arange(n, dtype=int)

        improved = True
        it = 0
        while improved and it < max_iter:
            improved = False
            it += 1
            rng.shuffle(nodes)

            for i in nodes:
                c_cur = labels[i]
                best_c = c_cur
                best_gain = 0.0

                neigh = indices[indptr[i]:indptr[i + 1]]
                if neigh.size == 0:
                    continue
                cand = set(labels[neigh])

                for c in cand:
                    if c == c_cur:
                        continue
                    idx = communities[c]
                    if not idx:
                        continue
                    gain = mm[i, list(idx)].sum()
                    if gain > best_gain + tol:
                        best_gain = gain
                        best_c = c

                if best_c != c_cur:
                    communities[c_cur].remove(i)
                    communities[best_c].add(i)
                    labels[i] = best_c
                    improved = True

        labels = reindex_consecutive(labels)
        return labels

    @staticmethod
    def _assemble_S(duals: Dict[str, np.ndarray | float], n: int) -> np.ndarray:
        """Symmetric penalty matrix S from dual components; zeros if missing."""

        S = np.zeros((n, n), dtype=float)
        for key, dual in duals.items():
            del key
            if isinstance(dual, np.ndarray):
                if dual.ndim == 1:
                    temp_dual = np.zeros((n, n), dtype=float)
                    for i in range(n):
                        for j in range(n):
                            temp_dual[i, j] = 0.5 * (dual[i] + dual[j])
                    S += temp_dual
                elif dual.ndim == 2:
                    if np.array_equal(dual, dual.T):
                        S += dual
                    else:
                        S += (dual + dual.T) / 2
            elif isinstance(dual, float):
                continue
        return S

    @staticmethod
    def _build_M(A: sparse.csr_matrix,
                 a: np.ndarray,
                 mprime: float,
                 gamma: float,
                 S: np.ndarray) -> np.ndarray:
        """
        Dense effective modularity matrix:
            M = A / m' - γ * (a a^T) / (m'^2) - S
        """
        A_dense = A.toarray().astype(float, copy=False)
        M = (A_dense / mprime) - gamma * np.outer(a, a) / (mprime * mprime) - S
        return M

    # def _local_move_with_duals(self,
    #                            A: sparse.csr_matrix,
    #                            S: np.ndarray,
    #                            a: np.ndarray,
    #                            mprime: float,
    #                            gamma: float,
    #                            rng: np.random.RandomState) -> Tuple[np.ndarray, float]:
    #     """
    #     Local moving using the dual-adjusted objective F(C) = sum_{i,j in same cluster} M_ij.
    #     Candidate communities are restricted to adjacency neighbors.
    #     """
    #     n = A.shape[0]
    #     Mmat = self._build_M(A, a, mprime, gamma, S)  # dense (n × n)

    #     labels = np.arange(n, dtype=int)
    #     communities = {i: {i} for i in range(n)}

    #     indptr = A.indptr
    #     indices = A.indices
    #     nodes = np.arange(n, dtype=int)

    #     improved = True
    #     last_obj = self._objective_from_partition(Mmat, labels)

    #     while improved:
    #         improved = False
    #         rng.shuffle(nodes)

    #         for i in nodes:
    #             c_cur = labels[i]

    #             # loss from leaving current community (exclude i)
    #             cur_set = communities[c_cur]
    #             loss_old = 0.0
    #             if len(cur_set) > 1:
    #                 # sum over current community excluding i
    #                 if i in cur_set:
    #                     loss_old = Mmat[i, list(cur_set)].sum() - Mmat[i, i]
    #                 else:
    #                     loss_old = Mmat[i, list(cur_set)].sum()

    #             # candidate communities from adjacency neighbors
    #             neigh = indices[indptr[i]:indptr[i + 1]]
    #             if neigh.size == 0:
    #                 continue
    #             cand = set(labels[neigh])

    #             best_c = c_cur
    #             best_gain = 0.0

    #             for c in cand:
    #                 if c == c_cur:
    #                     continue
    #                 idx = communities[c]
    #                 if not idx:
    #                     continue
    #                 gain_new = Mmat[i, list(idx)].sum()
    #                 net_gain = gain_new - loss_old
    #                 if net_gain > best_gain + self.tol_optimization:
    #                     best_gain = net_gain
    #                     best_c = c

    #             if best_c != c_cur:
    #                 communities[c_cur].remove(i)
    #                 communities[best_c].add(i)
    #                 labels[i] = best_c
    #                 improved = True

    #         # objective after a full sweep
    #         obj_now = self._objective_from_partition(Mmat, labels)
    #         if obj_now - last_obj <= self.tol_optimization:
    #             break
    #         last_obj = obj_now

    #     labels = reindex_consecutive(labels)
    #     final_obj = self._objective_from_partition(Mmat, labels)
    #     return labels, final_obj

    def _local_move_with_duals(self,
                           A: sparse.csr_matrix,
                           S: np.ndarray,
                           a: np.ndarray,
                           mprime: float,
                           gamma: float,
                           rng: np.random.RandomState) -> Tuple[np.ndarray, float]:
        """
        Local moving with dual-adjusted objective, like-for-like with Blondel ΔQ when S == 0.

        Objective block form on the current level:
            F(C) = Q_gamma(C) - sum_{i,j in same community} S_ij,
        with Q_gamma the Newman–Girvan modularity at resolution gamma and m' = sum(a).

        Move scoring (relative to current community c0):
            Δ_mod(c)  = [k_i,in(c)  - gamma * k_i * tot_a[c]  / m']  \
                        - [k_i,in(c0) - gamma * k_i * tot_a[c0] / m']
            Δ_dual(c) = -2 * ( s_i,in(c) - s_i,in(c0) )
            Accept c that maximizes Δ_mod(c) + Δ_dual(c) if strictly > tol_optimization.

        Conventions:
        - k_i,in(c) excludes the explicit self-loop A_ii.
        - tot_a[c] includes self-loops (it is the usual community degree total).
        - s_i,in(c) = sum_j∈c S_ij with S symmetric.
        """
        n = A.shape[0]
        labels = np.arange(n, dtype=int)

        # Community degree totals Σ_tot initialized from node degrees (includes self-loops).
        k = np.asarray(A.sum(axis=1)).ravel()
        tot = k.copy()

        # Community membership sets for S-row summations.
        communities = {c: {c} for c in range(n)}

        indptr, indices, data = A.indptr, A.indices, A.data
        nodes = np.arange(n, dtype=int)

        improved = True
        while improved:
            improved = False
            rng.shuffle(nodes)

            for i in nodes:
                c0 = labels[i]
                k_i = k[i]

                # Temporarily remove i from its community.
                communities[c0].remove(i)
                tot[c0] -= k_i
                labels[i] = -1  # sentinel during accumulation

                # Baseline dual sum s_i,in(c0) excluding i.
                if communities[c0]:
                    idx_c0 = np.fromiter(communities[c0], dtype=int)
                    s0 = float(S[i, idx_c0].sum())
                else:
                    s0 = 0.0

                # Accumulate k_i,in(c) using adjacency neighbors; skip explicit self-loop (j == i).
                start, end = indptr[i], indptr[i + 1]
                k_in = {}  # community -> weight from i to that community (excluding j==i)
                for j, wij in zip(indices[start:end], data[start:end]):
                    if j == i:
                        continue  # exclude A_ii from k_i,in(·)
                    cj = labels[j] if labels[j] != -1 else c0  # only i can be -1; map that back to c0 if it appears
                    k_in[cj] = k_in.get(cj, 0.0) + wij

                # Baseline modularity term for c0 (k_i,in(c0) might be 0 if no neighbors in c0 after removal).
                k_in_c0 = k_in.get(c0, 0.0)
                mod_base = k_in_c0 - gamma * k_i * (tot[c0]) / mprime

                # Evaluate candidate moves among neighbor-touched communities.
                best_c = c0
                best_delta = 0.0
                for c, k_in_c in k_in.items():
                    if c == c0:
                        continue
                    # Δ_mod relative to c0
                    mod_delta = (k_in_c - gamma * k_i * (tot[c]) / mprime) - mod_base
                    # Δ_dual relative to c0
                    if communities[c]:
                        idx_c = np.fromiter(communities[c], dtype=int)
                        s_c = float(S[i, idx_c].sum())
                    else:
                        s_c = 0.0
                    dual_delta = -2.0 * (s_c - s0)

                    delta = mod_delta + dual_delta
                    if delta > best_delta + self.tol_optimization:
                        best_delta = delta
                        best_c = c

                # Reinsert i (commit move only if strictly beneficial).
                labels[i] = best_c
                communities[best_c].add(i)
                tot[best_c] += k_i
                if best_c != c0:
                    improved = True

        # Reindex labels to 0..k-1 and compute level objective for monitoring.
        labels = reindex_consecutive(labels)
        Mmat = self._build_M(A, a, mprime, gamma, S)  # A/m' - gamma*(a a^T)/m'^2 - S
        final_obj = self._objective_from_partition(Mmat, labels)
        return labels, final_obj

    @staticmethod
    def _objective_from_partition(Mmat: np.ndarray, labels: np.ndarray) -> float:
        """sum_{i,j in same cluster} M_ij computed via block sums."""
        # Fast path using sorting/grouping could be added; simple implementation suffices here.
        k = int(labels.max()) + 1 if labels.size else 0
        total = 0.0
        for c in range(k):
            idx = np.flatnonzero(labels == c)
            if idx.size:
                total += Mmat[np.ix_(idx, idx)].sum()
        return float(total)

    @staticmethod
    def _block_sum(mm: np.ndarray, membership: sparse.csr_matrix) -> float:
        total = 0.0
        M_csc = membership.tocsc()  # one-time conversion
        indptr, indices = M_csc.indptr, M_csc.indices
        for c in range(M_csc.shape[1]):
            idx = indices[indptr[c]:indptr[c+1]]  # row indices in column c
            if idx.size:
                total += mm[np.ix_(idx, idx)].sum()
        return float(total)
