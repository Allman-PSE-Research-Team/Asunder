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
   Select storage independently for each hard column. ``"dense"`` always
   stores a dense Boolean array and ``"csr"`` always stores Boolean CSR.

``sparse_column_density_threshold=0.20``
   In automatic mode, use CSR when at most 20 percent of the logical entries
   are nonzero. This can produce a column pool containing both representations.

``max_dense_working_bytes=512 * 1024**2``
   Limit an operation's estimated package-created dense working set to 512
   MiB. This is a memory-safety setting, not an input-size limit or a CPU/GPU
   requirement. Raise it on a machine with sufficient available memory, or
   use ``None`` to disable the guard.

The estimate is operation-specific. If an operation expects ``q`` simultaneous
``N x N`` arrays of a dtype occupying ``b`` bytes per entry, it estimates
``q * N**2 * b`` bytes. For example, two ``float64`` arrays use ``16 * N**2``
bytes. Under the default cap that particular operation can reach 5,792 nodes;
a four-array estimate reaches 4,096 nodes, and ModularVFD's conservative
six-array estimate reaches 3,344 nodes. These are not general graph-size
limits. They do not estimate the column pool, solver memory, sparse storage,
Python overhead, or memory used elsewhere in the process.

The guard does not reject or automatically sparsify a dense matrix merely
because the caller already allocated it. It does apply when Asunder would
create additional dense work arrays. When a sparse-compatible route exists,
the operation keeps or falls back to CSR; post-loop aggregation follows this
rule even for a mixed dense/CSR column pool. When a selected backend is
intrinsically dense, an unsafe sparse-to-dense conversion raises
``MemoryError`` with the estimated requirement and configured limit. Setting
the limit to ``None`` disables this check, not the operation's actual memory
requirement.

Dense hard columns use the Boolean dtype. This reduces their array storage 
by a factor of eight compared with the ``int64`` representation.

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

Dense-only boundaries
---------------------

Some algorithms still require dense workspaces. ModularVFD, core-periphery
detection, internal signed Louvain, and the custom dense pricing heuristics use
an explicit checked conversion when given sparse input. The error reports the
estimated peak working set, the configured limit, and alternatives. The
default signed-Leiden pricing path is sparse-compatible.

Custom initial-column, master, pricing, and refinement callables must accept
the representation selected by the workflow. Asunder validates and normalizes
returned hard columns, but it does not silently densify inputs for an unknown
custom callable.

Load balancing
--------------

``LoadBalancer`` deliberately keeps dense Boolean columns because its master
and VFD workflow repeatedly access the full pair matrix. It still receives the
eightfold dtype reduction  compared to ``int64``, but adaptive CSR columns are
not enabled for this workflow.

What remains dense
------------------

A fractional co-association matrix can be genuinely dense. Post-loop
refinement accumulates an all-CSR active pool sparsely and uses measured
density when dense conversion is safe. A mixed pool uses dense accumulation
when it fits the cap and otherwise falls back to CSR. A dense-only refiner must
still fit beneath the configured working-set cap. Sparse storage does not
change the quadratic decision space of the exact pricing formulation or make
a dense algorithm intrinsically sparse.
