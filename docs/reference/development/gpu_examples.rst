:orphan:

.. code-block:: python

   import os

   # nx-cugraph reads this setting while NetworkX is imported.
   os.environ["NX_CUGRAPH_AUTOCONFIG"] = "True"

   import inspect
   import warnings

   import networkx as nx
   import numpy as np
   from scipy import sparse

   from asunder import CSDDecompositionConfig, run_csd_decomposition


   def require_cugraph_louvain_backend() -> None:
       louvain = getattr(nx.community, "louvain_communities", None)
       if not callable(louvain):
           raise RuntimeError(
               "This NetworkX version does not provide louvain_communities."
           )
       required_parameters = {"weight", "resolution", "seed"}
       missing = required_parameters.difference(inspect.signature(louvain).parameters)
       if missing:
           raise RuntimeError(
               "The installed NetworkX Louvain API is incompatible; missing "
               f"parameters: {sorted(missing)}."
           )
       if "cugraph" not in set(getattr(louvain, "backends", ())):
           raise RuntimeError(
               "nx-cugraph is not installed or does not implement "
               "networkx.community.louvain_communities."
           )


   def initial_columns(N: int, *, seed: int | None = None) -> list[np.ndarray]:
       del seed
       return [np.ones((N, N), dtype=bool)]


   def main() -> None:
       require_cugraph_louvain_backend()
       A = sparse.csr_matrix(
           np.array(
               [
                   [0.0, 3.0, 2.0, 0.0, 0.0, 0.0, 0.0],
                   [3.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0],
                   [2.0, 2.0, 0.0, 0.2, 0.0, 0.0, 0.0],
                   [0.0, 0.0, 0.2, 0.0, 2.0, 2.0, 0.0],
                   [0.0, 0.0, 0.0, 2.0, 0.0, 3.0, 0.0],
                   [0.0, 0.0, 0.0, 2.0, 3.0, 0.0, 0.0],
                   [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
               ]
           )
       )
       if np.any(A.diagonal() != 0.0):
           raise ValueError(
               "nx-cugraph Louvain does not support self-loops; "
               "A must have a zero diagonal."
           )

       warnings.warn(
           "NetworkX Louvain is an unsigned pricing heuristic. Asunder clips "
           "negative dual-adjusted weights only while generating a candidate; "
           "exact_rc=True recomputes reduced cost from the original A, gamma, "
           "and all master duals.",
           RuntimeWarning,
           stacklevel=1,
       )
       config = CSDDecompositionConfig(
           algo="louvain",
           package="networkx",
           subproblem_params={"exact_rc": True},
           seed=7,
           resolution=1.0,
           ifc_params={
               "generator": initial_columns,
               "num": 1,
               "args": {"N": A.shape[0]},
           },
           column_storage="auto",
           sparse_column_density_threshold=0.20,
           max_dense_working_bytes=64 * 1024**2,
           refine_post_loop=False,
           final_master_solve=True,
           max_iterations=3,
           disable_tqdm=True,
           verbose=0,
       )
       result = run_csd_decomposition(A, config=config)
       if result.final_partition is None:
           raise RuntimeError(result.metadata["status"])
       partition = result.final_partition
       print(partition.toarray() if sparse.issparse(partition) else partition)
       print(result.metadata)


   if __name__ == "__main__":
       main()

.. code-block:: python

   import inspect
   import warnings
   from typing import Any

   import networkx as nx
   import numpy as np
   from scipy import sparse

   from asunder import CSDDecompositionConfig, MatrixLike, run_csd_decomposition
   from asunder.base.column_generation.pricing import (
       build_dual_weight_matrix,
       compute_reduced_cost,
   )
   from asunder.base.utils import partition_vector_to_2d_matrix
   from asunder.base.utils.matrix import checked_to_dense, ensure_dense_working_set


   def _has_nonzero_diagonal(matrix: MatrixLike) -> bool:
       diagonal = np.asarray(matrix.diagonal()).reshape(-1)
       return bool(np.any(diagonal != 0.0))


   def _unsigned_adjusted_adjacency(
       A: MatrixLike,
       m: float,
       duals: dict[str, Any],
       *,
       max_dense_working_bytes: int | None,
   ) -> MatrixLike:
       if _has_nonzero_diagonal(A):
           raise ValueError(
               "NetworkX cuGraph Leiden does not support self-loops; "
               "A must have a zero diagonal."
           )

       dual_weight, _ = build_dual_weight_matrix(
           A,
           duals,
           max_dense_working_bytes=max_dense_working_bytes,
       )
       if sparse.issparse(A) and sparse.issparse(dual_weight):
           adjusted = sparse.csr_matrix(A - float(m) * dual_weight)
           adjusted.sum_duplicates()
           adjusted.eliminate_zeros()
           adjusted.sort_indices()
       else:
           ensure_dense_working_set(
               A.shape,
               dtype=np.float64,
               working_arrays=4.0,
               max_dense_working_bytes=max_dense_working_bytes,
               operation="NetworkX cuGraph Leiden pricing",
           )
           adjacency = checked_to_dense(
               A,
               max_dense_working_bytes=max_dense_working_bytes,
               operation="NetworkX cuGraph Leiden adjacency conversion",
           )
           pairwise_duals = checked_to_dense(
               dual_weight,
               max_dense_working_bytes=max_dense_working_bytes,
               operation="NetworkX cuGraph Leiden dual conversion",
           )
           adjusted = adjacency - float(m) * pairwise_duals

       if _has_nonzero_diagonal(adjusted):
           raise ValueError(
               "Dual adjustment produced self-loops, which NetworkX cuGraph "
               "Leiden does not support."
           )

       unsigned = adjusted.copy()
       if sparse.issparse(unsigned):
           has_negative_weights = bool(np.any(unsigned.data < 0.0))
           unsigned.data[unsigned.data < 0.0] = 0.0
           unsigned.eliminate_zeros()
       else:
           has_negative_weights = bool(np.any(unsigned < 0.0))
           unsigned[unsigned < 0.0] = 0.0

       if has_negative_weights:
           warnings.warn(
               "NetworkX cuGraph Leiden candidate generation clipped negative "
               "dual-adjusted weights to zero. The returned reduced cost is "
               "still recomputed exactly from the original A, gamma, and all duals.",
               RuntimeWarning,
               stacklevel=2,
           )
       return unsigned


   def _require_networkx_cugraph_leiden():
       leiden = getattr(nx.community, "leiden_communities", None)
       if not callable(leiden):
           raise RuntimeError(
               "This NetworkX version does not provide leiden_communities; "
               "install a compatible NetworkX and nx-cugraph release."
           )
       required_parameters = {"weight", "resolution", "max_level", "seed"}
       missing = required_parameters.difference(inspect.signature(leiden).parameters)
       if missing:
           raise RuntimeError(
               "The installed NetworkX Leiden API is incompatible; missing "
               f"parameters: {sorted(missing)}."
           )
       if "cugraph" not in set(getattr(leiden, "backends", ())):
           raise RuntimeError(
               "nx-cugraph is not installed or does not implement "
               "networkx.community.leiden_communities."
           )
       return leiden


   def _labels_from_communities(
       communities: list[set[int]],
       n_nodes: int,
   ) -> np.ndarray:
       expected = set(range(n_nodes))
       seen: set[int] = set()
       labels = np.full(n_nodes, -1, dtype=np.int64)
       for community_id, community in enumerate(communities):
           nodes = {int(node) for node in community}
           if not nodes.issubset(expected):
               raise RuntimeError("The GPU backend returned an unknown node.")
           if seen.intersection(nodes):
               raise RuntimeError("The GPU backend returned overlapping communities.")
           labels[list(nodes)] = community_id
           seen.update(nodes)
       if seen != expected:
           raise RuntimeError(
               "The GPU backend omitted one or more nodes, including an isolate."
           )
       return labels


   def networkx_cugraph_leiden_pricing(
       A: MatrixLike,
       a: np.ndarray,
       m: float,
       duals: dict[str, Any],
       *,
       gamma: float = 1.0,
       seed: int | None = 42,
       max_level: int | None = None,
       column_storage: str = "auto",
       sparse_column_density_threshold: float = 0.20,
       max_dense_working_bytes: int | None = 512 * 1024**2,
   ) -> tuple[float, MatrixLike]:
       leiden = _require_networkx_cugraph_leiden()
       unsigned = _unsigned_adjusted_adjacency(
           A,
           m,
           duals,
           max_dense_working_bytes=max_dense_working_bytes,
       )
       n_nodes = A.shape[0]
       has_edges = (
           unsigned.nnz > 0
           if sparse.issparse(unsigned)
           else bool(np.any(unsigned != 0.0))
       )

       if not has_edges:
           # Unsigned modularity is undefined on an edgeless graph. Each node,
           # including every isolate, becomes its own candidate community.
           labels = np.arange(n_nodes, dtype=np.int64)
       else:
           graph = (
               nx.from_scipy_sparse_array(
                   sparse.csr_matrix(unsigned),
                   create_using=nx.Graph,
                   edge_attribute="weight",
               )
               if sparse.issparse(unsigned)
               else nx.from_numpy_array(
                   np.asarray(unsigned),
                   create_using=nx.Graph,
                   edge_attr="weight",
               )
           )
           if set(graph.nodes) != set(range(n_nodes)):
               raise RuntimeError("Graph conversion failed to preserve all nodes.")
           communities = list(
               leiden(
                   graph,
                   weight="weight",
                   resolution=float(gamma),
                   max_level=max_level,
                   seed=seed,
                   backend="cugraph",
               )
           )
           labels = _labels_from_communities(communities, n_nodes)

       partition = partition_vector_to_2d_matrix(
           labels,
           storage=column_storage,
           sparse_column_density_threshold=sparse_column_density_threshold,
           max_dense_working_bytes=max_dense_working_bytes,
       )
       reduced_cost = compute_reduced_cost(
           A,
           a,
           m,
           partition,
           duals,
           gamma=float(gamma),
           max_dense_working_bytes=max_dense_working_bytes,
       )
       return reduced_cost, partition


   def initial_columns(N: int, *, seed: int | None = None) -> list[np.ndarray]:
       del seed
       return [np.ones((N, N), dtype=bool)]


   def main() -> None:
       A = sparse.csr_matrix(
           np.array(
               [
                   [0.0, 3.0, 2.0, 0.0, 0.0, 0.0, 0.0],
                   [3.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0],
                   [2.0, 2.0, 0.0, 0.2, 0.0, 0.0, 0.0],
                   [0.0, 0.0, 0.2, 0.0, 2.0, 2.0, 0.0],
                   [0.0, 0.0, 0.0, 2.0, 0.0, 3.0, 0.0],
                   [0.0, 0.0, 0.0, 2.0, 3.0, 0.0, 0.0],
                   [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
               ]
           )
       )
       config = CSDDecompositionConfig(
           seed=7,
           resolution=1.0,
           ifc_params={
               "generator": initial_columns,
               "num": 1,
               "args": {"N": A.shape[0]},
           },
           subproblem_params={"max_level": 100},
           column_storage="auto",
           sparse_column_density_threshold=0.20,
           max_dense_working_bytes=64 * 1024**2,
           refine_post_loop=False,
           final_master_solve=True,
           max_iterations=3,
           disable_tqdm=True,
           verbose=0,
       )
       result = run_csd_decomposition(
           A,
           config=config,
           subproblem_fn=networkx_cugraph_leiden_pricing,
       )
       if result.final_partition is None:
           raise RuntimeError(result.metadata["status"])
       partition = result.final_partition
       print(partition.toarray() if sparse.issparse(partition) else partition)
       print(result.metadata)


   if __name__ == "__main__":
       main()
