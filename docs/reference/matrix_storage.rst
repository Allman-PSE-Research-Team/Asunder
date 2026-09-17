Matrix Storage and Large Graphs
===============================

Asunder partition objects always represent a complete ``N x N``
co-association matrix: entry ``[i, j]`` says whether nodes ``i`` and ``j``
share a community. The physical storage may be either a NumPy array or a
SciPy CSR matrix. CSR storage changes memory use, not the meaning or shape of
the result.

Storage controls
----------------

Reusable decomposition and nonlinear branch-and-price expose three controls:

``column_storage="auto"``
   Preserve each supplied column's representation independently, including
   columns returned by custom callables. Dense and CSR columns may coexist in
   one pool. ``"dense"`` explicitly requests dense Boolean arrays; ``"csr"``
   explicitly requests Boolean CSR.

``sparse_column_density_threshold=0.20``
   For a new column constructed from labels, choose CSR only when at most
   20 percent of entries are nonzero **and** its estimated data, index, and
   row-pointer storage is smaller than dense Boolean storage. This threshold
   does not convert already-supplied columns. CSR construction fills its final
   buffers directly rather than building duplicate full coordinate arrays.

``max_dense_working_bytes=512 * 1024**2``
   Limit an operation's estimated package-created dense working set to 512
   MiB. This is a memory-safety setting, not an input-size limit or a CPU/GPU
   requirement. Raise it on a machine with sufficient available memory, or
   use ``None`` to disable the guard.

The estimate is operation-specific. If an operation expects ``q`` simultaneous
``N x N`` arrays of a dtype occupying ``b`` bytes per entry, it estimates
``q * N**2 * b`` bytes, plus explicitly estimated scratch space where needed.
For example, two ``float64`` arrays use ``16 * N**2``
bytes. Under the default cap that particular operation can reach 5,792 nodes;
a four-array estimate reaches 4,096 nodes, and ModularVFD's conservative
six-array estimate reaches 3,344 nodes. These are not general graph-size
limits. Estimates are safety checks, not measured peak-memory guarantees:
they do not include the entire column pool, solver memory, sparse storage,
Python overhead, library-private workspaces, or other memory in the process.

The guard does not reject or automatically sparsify a dense matrix merely
because the caller already allocated it. It does apply when Asunder would
create additional dense work arrays: the existing allocation and the new
workspace coexist, so supplying a dense input does not waive checks on those
new allocations. Dense dtype conversions also allocate and are checked.
Validation uses bounded row scratch, and an already Boolean dense column need
not be copied. An over-budget dense operation raises ``MemoryError`` with its
estimated requirement and the required configuration value, including in
automatic mode. It never falls back to CSR merely to evade this cap. Sparse
values are cast before densification to avoid allocating two full dense arrays.
Contraction, original-size expansion, and nested construction receive the same
cap setting. ``None`` disables these checks, not actual memory requirements.

Dense hard columns use the Boolean dtype. This reduces their array storage 
by a factor of eight compared with the ``int64`` representation.

Scoring a dense hard or fractional column uses float64 row-wise reductions,
with O(N) scratch rather than an ``N x N`` floating-point copy. Both NLBNP
entry points check the cap before converting an integer/Boolean adjacency to
float64. An existing float64 adjacency needs no conversion allocation; sparse
and NetworkX inputs remain on the CSR normalization path.

CSR input and results
---------------------

SciPy sparse adjacency inputs are normalized to CSR. NetworkX graphs passed
to ``NonlinearBranchAndPrice`` are also converted to CSR by default. Sparse
storage is preserved through NLBNP preprocessing, graph contraction, the
restricted master, and compatible pricing backends.

The final partition and iteration columns may remain CSR:

.. code-block:: python

   from scipy import sparse

   def print_partition_storage(partition):
       if sparse.isspmatrix_csr(partition):
           print("stored entries:", partition.nnz)
       else:
           print("dense entries:", partition.size)

Use :func:`asunder.base.utils.partition_matrix_to_vector` or
:func:`asunder.base.utils.group_nodes_by_community` when labels or community
members are needed. These helpers support CSR directly. Avoid
``np.asarray(partition)`` because SciPy sparse matrices do not become ordinary
two-dimensional arrays that way.

Current dense boundaries and future sparse work
-----------------------------------------------

Accepting CSR adjacency does not make every downstream algorithm sparse.
The table below lists the remaining dense-preferring areas, including ones
that do not need immediate conversion. It is an implementation roadmap, not a
promise that sparse storage will improve every method.

.. list-table:: Dense-preferring components
   :header-rows: 1
   :widths: 24 36 40

   * - Component
     - Current behavior
     - What future sparse support would require
   * - ModularVFD
     - Its public refinement adapter accepts CSR but crosses a guarded dense
       boundary. The search maintains several full pairwise work arrays.
     - Sparse or lazy modularity gains, constraint-state updates, and
       contraction kernels. Merely changing the input type would not remove
       the internal quadratic work.
   * - Load-balancing VFD and ``LoadBalancer``
     - Deliberately use dense Boolean columns and full-pair scoring. This is a
       separate implementation from ModularVFD.
     - A sparse-aware LB master/column pool plus random-access VFD kernels.
       It is most promising when many small communities make co-association
       columns genuinely sparse.
   * - Core-periphery backends
     - The spectral, KL, and genetic backends use dense original or contracted
       BE matrices. CSR input crosses a guarded conversion boundary.
     - Sparse eigensolvers alone are insufficient: BE scoring,
       reconstruction, contraction, and result storage also need sparse or
       blockwise formulations.
   * - Dual-adjusted ``ModifiedLouvain``
     - Standard Louvain remains sparse, but the column-pricing variants build
       dense modularity and dual matrices under a working-set guard.
     - Lazy null-model and dual terms, with incremental community scores that
       do not assemble the complete modified modularity matrix.
   * - Internal signed Louvain
     - Signed pricing crosses a guarded dense boundary to extract graph edges
       before building positive and negative graph layers. It differs from
       default signed Leiden and ordinary third-party Louvain.
     - Sparse signed adjacency and incremental positive/negative community
       scores throughout the internal search.
   * - Custom spectral and RCCS pricing
     - Build a dense dual-adjusted objective. The decomposition wrapper guards
       the boundary; direct low-level use remains intended for inputs that fit
       memory.
     - Sparse or operator-based eigensolvers and lazy reduced-cost evaluation.
   * - QMETIS load-balancing pricing
     - The native graph interface is adjacency-list based, but the pricing
       adapter first constructs and quantizes a dense dual-adjusted weight
       matrix.
     - Generate native edge weights from sparse adjacency and a lazy dual
       representation without materializing every pair.
   * - Pairwise feasibility projections
     - The base projection guards its dense pair-score conversion. The LB
       projection remains a dense-only low-level path.
     - Sparse objective coefficients and constraint generation. The logical
       pairwise variables can still remain quadratic.
   * - Exact pricing and related ILPs
     - Sparse adjacency is accepted, but the mathematical model retains
       pairwise variables and, in places, transitivity constraints. Sparse
       input does not remove that model size.
     - Cutting planes, delayed constraint generation, or a different exact
       formulation rather than only a storage-format change.
   * - Spectral initial-partition helpers
     - Base dense eigendecomposition paths cross a guarded boundary. The LB
       spectral helper still materializes its Laplacian without that guard.
     - Sparse Laplacian operators/eigensolvers throughout the helper and its
       fallback paths.
   * - NLBNP confidence clustering
     - Converts the ``N x K`` membership-confidence matrix to dense storage.
       This is not an ``N x N`` allocation and is usually much smaller.
     - Sparse-aware clustering or row-wise confidence extraction if both
       ``N`` and the number of communities become large.
   * - Matrix visualization
     - Ultimately renders a dense image and therefore uses a guarded
       conversion.
     - Downsampling, tiling, or sparse point rendering rather than a nominal
       CSR input alone.

The default signed-Leiden pricing route, the restricted base master, graph
contraction, objective scoring, and hard-column validation already operate on
CSR without an unconditional dense conversion. Custom initial-column, master,
pricing, and refinement callables must accept the representation selected by
the workflow. Asunder validates and normalizes returned hard columns, but it
does not silently densify inputs for an unknown custom callable.

Use ``custom_heuristic_subproblem`` to pass sparse duals to ModifiedLouvain or
RCCS: it aggregates every dual contribution and guards the dense boundary.
Direct dense-kernel calls reject sparse duals instead of silently ignoring them.
The adapter recomputes the returned reduced cost from the original objective.

Load balancing
--------------

``LoadBalancer`` deliberately keeps dense Boolean columns because its master
and VFD workflow repeatedly access the full pair matrix. It still receives the
eightfold dtype reduction  compared to ``int64``, but adaptive CSR columns are
not enabled for this workflow.

What remains dense
------------------

A fractional co-association matrix can be genuinely dense. Post-loop
refinement keeps aggregation in CSR when every active (nonzero-weight) column
is sparse. If any active column is dense, it accumulates into one ``float64``
result with bounded row-level scratch. The current estimate is
``8 * N**2 + 16 * N`` bytes, not just the output buffer's size. An over-budget
estimate raises; there is no forced CSR fallback. The source columns keep
their own formats, and mixed-format duplicate comparisons also work row by row.

A dense-only refiner must additionally fit beneath its own working-set
estimate. Final storage metadata is calculated from the actual final column
pool and attached only to the final iteration record. Sparse storage does not
change the quadratic decision space of exact pricing or make dense algorithms
intrinsically sparse.
