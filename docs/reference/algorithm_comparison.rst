Subproblem Algorithm Comparison
===============================

Asunder offers several pricing algorithms that generate candidate partitions
for column generation. Select a method by its algorithm and backend together:
different implementations of the same algorithm can have different weight,
hardware, and memory requirements.

This comparison covers pricing and not initial partitioning, constraint enforcement,
refinement, and the master solver which also contribute to the complete workflow's
cost. See :doc:`matrix_storage` for the current dense conversion boundaries.

Objectives and edge weights
---------------------------

For the community-detection rows, ``package`` and ``algo`` identify the
configuration in ``asunder.CSDDecompositionConfig``; see
:doc:`../getting_started/base_decomposition`. Custom pricing rows
refer to :func:`asunder.base.column_generation.subproblem.custom_heuristic_subproblem`.
The exact ILP is available through
:func:`asunder.base.column_generation.subproblem.solve_subproblem`.
QMETIS is selected with ``LoadBalancer(..., algorithm="qmetis")``.

The objective column describes candidate generation. With the default
``exact_rc=True``, community-based pricing recomputes reduced cost from the
original adjacency and all master duals. Custom pricing and QMETIS also
rescore candidates using the original objective. QMETIS scales and rounds
edge weights only for search; rescoring uses the original values.
Exact ILP optimality requires the solver to finish with an optimal status.
Exact rescoring does not guarantee that a heuristic finds the best pricing column.

``Preserved`` means negative dual-adjusted edge weights reach a signed search
method. ``Clipped`` means negative values become zero during candidate
generation; their contribution remains in exact rescoring. This distinction
matters even for a nonnegative input graph, because master duals can introduce
negative pricing weights. It does not change the input-normalization
requirements of the overall workflow.

.. list-table:: Candidate objectives and edge-weight handling
   :header-rows: 1
   :widths: 17 15 38 15 15

   * - ``algo`` / method
     - ``package`` / backend
     - Search objective
     - Negative pricing weights
     - Fractional edge weights
   * - ``louvain``
     - ``networkx``
     - Maximize modularity
     - Clipped
     - Accepted
   * - ``leiden``
     - ``networkx`` (active backend)
     - Maximize modularity
     - Clipped
     - Accepted
   * - ``greedy``
     - ``networkx``
     - Maximize modularity (greedy)
     - Clipped
     - Accepted
   * - ``girvan_newman``
     - ``networkx``
     - Betweenness splits; best modularity
     - Clipped
     - Scoring only
   * - ``louvain``
     - ``sknetwork``
     - Maximize modularity
     - Clipped
     - Accepted
   * - ``leiden``
     - ``sknetwork``
     - Maximize modularity
     - Clipped
     - Accepted
   * - ``lpa``
     - ``sknetwork``
     - Label propagation
     - Clipped
     - Accepted
   * - ``leiden``
     - ``igraph``
     - Maximize modularity
     - Clipped
     - Accepted
   * - ``greedy``
     - ``igraph``
     - Maximize modularity (greedy)
     - Clipped
     - Accepted
   * - ``multilevel``
     - ``igraph``
     - Maximize modularity
     - Clipped
     - Accepted
   * - ``cpm_leiden``
     - ``igraph``
     - Maximize CPM
     - Preserved
     - Accepted
   * - ``spinglass``
     - ``igraph``
     - Minimize signed spin energy
     - Preserved
     - Accepted
   * - ``infomap``
     - ``igraph``
     - Minimize map equation
     - Clipped
     - Accepted
   * - ``lpa``
     - ``igraph``
     - Label propagation
     - Clipped
     - Accepted
   * - ``voronoi``
     - ``igraph``
     - Voronoi groups; best modularity
     - Clipped
     - Accepted
   * - ``walktrap``
     - ``igraph``
     - Random-walk merges; best modularity
     - Clipped
     - Accepted
   * - ``leiden``
     - ``leidenalg``
     - Maximize RB modularity
     - Clipped
     - Accepted
   * - ``signed_leiden`` (default)
     - ``leidenalg``
     - Maximize signed RB modularity
     - Preserved
     - Accepted
   * - ``cpm_leiden``
     - ``leidenalg``
     - Maximize CPM
     - Preserved
     - Accepted
   * - ``surprise_leiden``
     - ``leidenalg``
     - Maximize surprise
     - Clipped
     - Accepted
   * - ``signed_surprise_leiden``
     - ``leidenalg``
     - Maximize signed surprise
     - Preserved
     - Accepted
   * - ``signed_louvain``
     - ``None`` (internal)
     - Maximize signed modularity
     - Preserved
     - Accepted
   * - ``full_louvain``
     - Custom pricing
     - Reduced cost (Louvain search)
     - Preserved
     - Accepted
   * - ``spectral``
     - Custom pricing
     - Reduced cost (spectral search)
     - Preserved
     - Accepted
   * - ``RCCS``
     - Custom pricing
     - Maximize reduced cost
     - Preserved
     - Accepted
   * - QMETIS pricing
     - Native QMETIS
     - Modularity with load balance
     - Clipped
     - Scaled to integers for search
   * - Exact ILP pricing
     - Pyomo + configured solver
     - Maximize pricing objective
     - Preserved
     - Accepted (solver tolerances)

Some wrappers report modularity even when the candidate generator optimizes a
different criterion, such as Infomap or label propagation. Signed Leiden's
reported layer-normalized score also differs from its multiplex optimization
quality. RB means Reichardt--Bornholdt configuration-model quality; CPM means
Constant Potts Model. The signed variants subtract negative-layer quality
from positive-layer quality. Use the recomputed reduced cost to judge a candidate's value to the
master problem. The `leidenalg reference`_ describes the available quality
functions and multiplex optimization.

Asunder uses NetworkX Girvan--Newman's default, unweighted edge-betweenness
rule. Fractional edge weights affect the modularity comparison between splits
and the final reduced-cost calculation, but do not guide edge removal. See
the `NetworkX Girvan--Newman implementation`_.

Non-default ``resolution`` support also depends on the selected route.
Community-based pricing accepts it for NetworkX Louvain/Leiden/greedy,
scikit-network Louvain/Leiden, igraph Leiden/multilevel/CPM, leidenalg
Leiden/signed Leiden/CPM, and internal signed Louvain. Custom ModifiedLouvain
and QMETIS also support it; custom spectral and RCCS currently require
``resolution=1``. Other community-based routes reject a non-default value.

Edge weights and node loads are separate
----------------------------------------

Edge weights describe relationships and contribute to the partition objective.
Node loads describe the quantity being balanced. Fractional support for one
does not establish fractional support for the other.

.. list-table:: Weight domains by interface
   :header-rows: 1
   :widths: 30 35 35

   * - Interface
     - Edge weights
     - Node loads
   * - Community / custom pricing
     - Real; signed support above
     - Workflow-dependent
   * - QMETIS pricing
     - Scaled to nonnegative integers
     - Positive integers
   * - LoadBalancer / LB-VFD
     - Real; depends on pricing method
     - Positive integers
   * - Shared node_weights
     - Adjacency unchanged
     - Finite real values; hook-specific limits

To express fractional loads in the built-in load-balancing workflow, convert
loads to common integer units and scale explicit load bounds consistently.
See :doc:`load_balancing` for the weighted K/R semantics and
:doc:`qmetis` for the native graph/matrix interfaces and quantization details.

CPU, GPU, and implementation
----------------------------

All listed methods have a CPU execution path. The table describes the current
adapters; an upstream parallel or GPU implementation is usable only when the
adapter can actually reach it. A native library alone does not establish
multithreaded search.

.. list-table:: Execution capabilities
   :header-rows: 1
   :widths: 25 25 25 25

   * - Backend / method
     - Implementation
     - CPU threads
     - GPU
   * - NetworkX Louvain / Leiden
     - Python / selected backend
     - Backend-dependent
     - Via nx-cugraph
   * - NetworkX greedy / Girvan--Newman
     - Python
     - Serial search
     - No adapter support
   * - scikit-network
     - Python + compiled kernels
     - No exposed thread control
     - No adapter support
   * - igraph
     - Native C/C++ bindings
     - No exposed thread control
     - No adapter support
   * - leidenalg
     - C++ bindings
     - No exposed thread control
     - No adapter support
   * - Internal signed Louvain
     - Python / NetworkX
     - Serial search
     - No adapter support
   * - Custom Louvain / RCCS / spectral
     - Python + NumPy/SciPy
     - Numerical kernels may use threads
     - No adapter support
   * - QMETIS
     - Native library via ctypes
     - No exposed thread control
     - No adapter support
   * - Exact pricing / master
     - Pyomo + solver
     - Solver-dependent
     - Gurobi PDHG for LP work

The `NetworkX Louvain documentation`_ lists ``cugraph`` as an additional
backend and documents restrictions, including unsupported self-loops and
different seed behavior. Check backend registration, dependency versions,
hardware, and the actual search graph before using dispatch. GPU execution
does not restore clipped negative weights. Select built-in NetworkX Leiden
with ``package="networkx", algo="leiden"``. Install a compatible NetworkX
version and backend, then activate that backend through
``nx.config.backend_priority.algos`` or its automatic configuration.
For example, after installing nx-cugraph, set
``nx.config.backend_priority.algos = ["cugraph"]``.
Asunder reports a short error if no active backend supports Leiden.
See the `NetworkX Leiden documentation`_.

Complexity and theoretical scalability
--------------------------------------

Let ``n`` be the number of vertices passed to pricing, ``e`` the number of
edges in its search graph, ``p`` the number of search sweeps, and ``d`` a
merge-tree depth. Contraction can reduce ``n``. Dual adjustment can increase
``e`` relative to the original graph, potentially to quadratic size.

Published bounds below describe the cited algorithm or native implementation.
Where a bound has not been established, the table states the dominant 
operations instead of assigning a fixed order.

.. list-table:: Complexity and theoretical scalability
   :header-rows: 1
   :widths: 25 40 35

   * - Method
     - Time / work
     - Main scaling limit
   * - Louvain
     - Repeated sweeps; near-linear sparse average in igraph
       (`igraph algorithms`_)
     - Edge count, passes, levels
   * - Leiden (all backends)
     - Moving, refinement, aggregation; near-linear sparse behavior in igraph
       (`igraph algorithms`_)
     - Edge count, iterations; two layers when signed
   * - Label propagation
     - Scan model: ``O(p * (n + e))``
     - Sweep count; backend bookkeeping excluded
   * - NetworkX greedy
     - CNM bound: ``O(e * d * log(n))`` (`CNM paper`_)
     - Merge-tree depth
   * - igraph greedy
     - Worst case: ``O(e * n * log(n))`` (`igraph algorithms`_)
     - Edge count and merges
   * - NetworkX Girvan--Newman
     - Full-removal core: ``O(n * e**2)``
       (`NetworkX Girvan--Newman implementation`_)
     - Repeated betweenness calculations
   * - igraph Walktrap
     - Worst case: ``O(e * n**2)`` (`Walktrap paper`_)
     - ``O(n**2)`` memory
   * - igraph spinglass
     - Annealing; no overall bound stated
     - Temperatures, sweeps, spins
   * - igraph Infomap
     - Flow search; no overall bound stated
     - Trials and convergence
   * - igraph Voronoi
     - Density / distance search; no overall bound stated
     - Neighborhoods and radius search
   * - Internal signed Louvain
     - ``Theta(n**2)`` preparation + search
     - Dense workspace
   * - Custom ModifiedLouvain
     - ``Theta(n**2)`` preparation + search
     - Dense workspace; passes and levels
   * - Custom RCCS
     - ``Theta(n**2)`` preparation + search
     - Dense workspace; moves and restarts
   * - Custom spectral
     - ``O(k**3)`` per k-node eigendecomposition
     - Dense workspace; repeated splits / refinement
   * - QMETIS pricing
     - ``Theta(n**2)`` preparation + native search
     - Dense adapter workspace and LB columns
   * - Exact ILP pricing
     - ``Theta(n**2)`` binary variables; ``Theta(n**3)`` constraints
     - Cubic model size; potentially exponential solve

The complete workflow also stores partition columns. For community sizes
``s_1, ..., s_K``, one hard co-association column has ``sum(s_i**2)`` ones.
Its CSR storage therefore grows as ``Theta(n + sum(s_i**2))``; one large
community can make the column dense even when the input graph is sparse.
Keeping ``C`` dense columns requires ``Theta(C * n**2)`` array storage,
before master-model and refinement memory. ``LoadBalancer`` currently uses
dense Boolean columns.


Master solver and Gurobi GPU LP support
---------------------------------------

Pricing is only one stage of column generation. As pricing becomes cheaper,
master-model construction or repeated restricted-master solves can account
for a larger share of the total work.

Asunder defaults to Gurobi through Pyomo and can use other compatible solvers
for supported workflows. The built-in base and load-balancing masters reuse
and extend one model by default with ``gurobi_direct``, ``gurobi_persistent``,
or ``cplex_persistent``. This avoids repeated model construction and allows
solver reoptimization. Set ``persistent_master=False`` to rebuild each solve.

Gurobi 13 provides a primal-dual hybrid gradient (PDHG) LP algorithm. GPU
execution requires ``Method=6`` and ``PDHGGPU=1`` plus a GPU-enabled build.
The documented GPU platforms are Linux x86-64 and ARM64 with compatible
NVIDIA hardware; the GPU option is unavailable on Compute Server and Instant
Cloud as of 10/26. See the
`Gurobi PDHG parameters`_ and `Gurobi GPU platforms`_.

GPU PDHG accelerates selected LP work; it does not move all master-model
construction, integer optimization, or Python orchestration to the GPU.
Gurobi documents GPU PDHG for a MIP's root LP relaxation and explains why
simplex remains appropriate for repeated branch-and-bound node relaxations
(`Gurobi LP and MIP discussion`_).

.. _leidenalg reference: https://leidenalg.readthedocs.io/en/stable/reference.html
.. _NetworkX Girvan--Newman implementation: https://networkx.org/documentation/stable/_modules/networkx/algorithms/community/centrality.html
.. _NetworkX Leiden documentation: https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.community.leiden.leiden_communities.html
.. _NetworkX Louvain documentation: https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.community.louvain.louvain_communities.html
.. _igraph algorithms: https://igraph.org/c/html/latest/igraph-Community.html
.. _CNM paper: https://arxiv.org/abs/cond-mat/0408187
.. _Walktrap paper: https://arxiv.org/abs/physics/0512106
.. _Gurobi PDHG parameters: https://docs.gurobi.com/projects/optimizer/en/current/reference/parameters.html#pdhggpu
.. _Gurobi GPU platforms: https://docs.gurobi.com/projects/optimizer/en/current/reference/releasenotes/platforms.html#supported-gpu-platforms
.. _Gurobi LP and MIP discussion: https://www.gurobi.com/resources/blog/using-gpus-to-solve-lps-vs-mips-what-s-the-difference
