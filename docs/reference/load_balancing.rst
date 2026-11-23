Weighted Load Balancing
========================

Start with :doc:`../getting_started/quickstart` for a complete ``LoadBalancer``
example. This page explains the units behind its balance settings.

Choosing the VFD refinement backend
-----------------------------------

``LoadBalancer`` uses Python VFD by default. To select the bundled C++ search,
set ``refinement_backend="cpp"``. This selects refinement independently of the
pricing algorithm and solver. For example, the following call targets three
equally loaded communities and runs two tabu rounds during post-loop refinement.

.. code-block:: python

   from asunder.load_balancing import LoadBalancer

   result = LoadBalancer(
       graph, K=3, R=0, refinement_backend="cpp",
       refine_params={"kwargs": {"shake_rounds": 2}},
   )

Both backends use the workflow's constraints, seeds and balance settings.
In-loop refinement uses construction and repair with ``shake_rounds=0``.
A custom ``refine_params["refine_func"]`` takes precedence over the selector.
The native search uses its own random generator, so partitions and scores can
differ between backends for the same seed.

For direct refinement and timing information, import
:func:`asunder.load_balancing.algorithms.VFD_cpp.refine_partition` and pass
``return_metadata=True``. A feasible result is a pair containing the partition
matrix and metadata such as ``native_search_seconds``, ``objective_B_sum``,
and ``community_loads``. A search that exhausts its budget without finding a
feasible partition returns ``None``. Native search accepts sparse inputs by
converting them at the configured dense-memory limit.

For direct native calls, integer ``balance_weights`` are summed exactly up to
a total of ``2**63 - 1``. Pass integer arrays when exact loads above ``2**53``
matter. The full column-generation workflow uses floating-point weights in
its master problem.

The Linux x86-64, Windows x64 and macOS universal2 release wheels include the
VFD library. For an editable install, run ``python tools/build_vfd.py`` from
the source root with CMake and a C++17 compiler installed. For a local wheel,
run that command before ``python -m build --wheel``. Python refinement works
directly from a source installation.

Counts versus loads
-------------------

With unit node weights, balancing means balancing node counts. With supplied
weights, each community's load is the sum of its members' weights. Contracting
must-linked nodes sums their weights, so total load does not change.

``LoadBalancer``, its VFD backends, ModularVFD, and QMETIS require positive integer
node weights. ModularVFD retains this requirement even when
``use_K_constraint=False``. Fractional node weights are rejected, not silently
rounded or scaled.

The generic ``run_csd_decomposition`` interface still accepts finite real
``node_weights`` for custom hooks. A hook's own restrictions apply when it
consumes those weights. Additional application-specific vectors can be passed
under distinct hook argument names.

The K/R rule
-------------

Load balancing enforces community-load bounds, which reduce to community-size
bounds for unit node weights. Without explicit ``R_bounds``, supplying ``K``
and ``R`` derives these bounds, and initialization and default refinement
target ``K``. The master enforces community loads rather than an exact count,
so the final result is not guaranteed to contain exactly ``K`` communities.

Let :math:`W=\sum_i w_i` be total node weight. ``K`` is the requested number of
communities; ``R`` is an absolute load-window width in the same units as the
weights, not a percentage. Without explicit ``R_bounds``, the rule is:

.. math::

   L = \max\left(1,\left\lfloor \frac{W}{K}-\frac{R}{2}+\frac12\right\rfloor\right),
   \qquad U=L+R.

Each community's load must lie between ``L`` and ``U``, inclusive. The shared
bound resolver also caps ``U`` at ``W``. For example, ``W=120, K=4, R=10``
gives the interval ``[25, 35]``. With unit weights, ``W`` is simply the node
count and the original size-balancing rule is recovered.

``R_bounds=(lower, upper)`` replaces the derived interval. Its endpoints use
the same integer units as node weights. ``R=0`` without explicit bounds asks
for equal loads; divisibility of ``W`` by ``K`` is necessary but does not
guarantee feasibility because nodes and must-link components are indivisible.

Community counts with explicit bounds
--------------------------------------

Supplying ``R_bounds=(L, U)`` makes ``LoadBalancer`` search the community counts
admitted by those bounds. For total load ``W`` and ``N`` nodes, this interval is
``ceil(W / U)`` through ``min(N, floor(W / L))``, inclusive. When ``L=0``, the
upper count is ``N``. An empty interval raises ``ValueError``. Individual node
loads and pairwise constraints can further restrict feasibility.

Initialization searches this interval until it finds a feasible column.
Default VFD refinement compares candidates across the interval by modularity;
``K`` remains a heuristic hint. For example,
``LoadBalancer(graph, R_bounds=(2, 2))`` can produce three communities on a
six-node graph even though the default ``K`` is two.

Broad intervals can increase refinement time because each admitted count is
searched. An explicit ``K_search_radius`` in compatible ``refine_params``
kwargs takes precedence over the wrapper's automatic radius. Direct VFD calls
retain their own search defaults. Calls that omit ``R_bounds`` retain the
existing K/R initialization and refinement behavior.

With explicit bounds, ``projection_repair=True`` also tries admitted counts,
starting nearest the requested ``K``, and retains the smallest projection
distance found. ``projection_time_limit`` is shared across these attempts;
supported solvers receive the remaining time for each solve. Repair metadata
reports the original ``requested_K`` and the actual ``K_used``. The standalone
``project_partition_ilp`` API continues to project onto an exact-count set.

Dense workspace limit
---------------------

``LoadBalancer(..., max_dense_working_bytes=4 * 1024**3)`` raises the estimated
dense working-set cap per operation from its default 512 MiB to 4 GiB. The
setting is forwarded to decomposition and pricing, including QMETIS dual
assembly and reduced-cost scoring. ``None`` disables the guard; actual memory
requirements are unchanged.

Choosing integer units
----------------------

Choose a unit that represents the precision your application needs, such as
milliseconds instead of seconds. Supply the converted positive integer
weights yourself; Asunder does not choose a scale or precision for you.

For example, loads ``[0.25, 0.75, 1.0]`` with bounds ``[0.75, 1.25]`` can be
expressed in hundredths as weights ``[25, 75, 100]`` and
``R_bounds=(75, 125)``. Keep ``K`` unchanged. Convert any custom load thresholds
to the same units, and divide reported community loads by 100 to interpret
them in the original units. Do not scale the adjacency or modularity
resolution merely because node-load units changed.

Choose a scale that represents the intended weights and bounds exactly.
Rounding arbitrary values is an application-level approximation that can
change hard feasibility. Avoid unnecessarily large units that risk overflow
or loss of numerical precision.

Why explicit bounds matter when rescaling
------------------------------------------

The rounded K/R rule is not invariant under a change of units:

- ``W=10, K=3, R=1`` gives ``[3, 4]``.
- Scaling weights and ``R`` by 10 gives ``W=100, K=3, R=10`` and ``[28, 38]``,
  not ``[30, 40]``.

To preserve the original feasible load interval, pass the explicitly scaled
``R_bounds=(30, 40)``. Scaling ``R`` alone is not enough to preserve rounded
bounds. This also applies to custom constraints that interpret node weights.

QMETIS edge quantization is different
--------------------------------------

The :doc:`qmetis` adapter can quantize fractional **edge** weights to guide its
candidate search, then recompute reduced costs with the original data.
That mechanism does not scale node weights or balance bounds. Node weights
affect hard feasibility, which exact objective rescoring cannot restore after
inappropriate rounding.
