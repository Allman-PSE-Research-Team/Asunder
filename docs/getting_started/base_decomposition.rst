Reusable Decomposition Guide
============================

Use ``run_csd_decomposition`` when you need Asunder's column-generation
orchestration but want control over initial columns, the master problem,
pricing, or refinement. If ordinary balanced communities are enough, use
:doc:`quickstart` instead.

Prerequisites
-------------

- Install Asunder and configure a Pyomo-compatible solver as described in
  :doc:`installation`.
- Represent the graph as a square NumPy or SciPy sparse adjacency matrix.
- Decide how to generate at least one feasible starting partition.

The four replaceable pieces
---------------------------

``initial columns``
   Feasible partitions available before the first iteration.

``master problem``
   Chooses among the available partitions and produces dual information for
   pricing.

``pricing problem``
   Searches for another partition with positive reduced cost.

``refinement``
   Optionally improves a generated partition before or after the main loop.

The default master uses Pyomo, and the default pricing backend is signed
Leiden. Start by configuring only the initial columns and replace other pieces
when the application requires it.

Run a minimal decomposition
---------------------------

.. code-block:: python

   import numpy as np

   from asunder import CSDDecompositionConfig, run_csd_decomposition
   from asunder.base.utils import make_partitions_random_links_only

   A = np.array(
       [
           [0.0, 2.0, 1.0, 0.0],
           [2.0, 0.0, 1.0, 0.0],
           [1.0, 1.0, 0.0, 2.0],
           [0.0, 0.0, 2.0, 0.0],
       ]
   )
   must_link = [(0, 1)]
   cannot_link = [(0, 3)]

   config = CSDDecompositionConfig(
       must_link=must_link,
       cannot_link=cannot_link,
       ifc_params={
           "generator": make_partitions_random_links_only,
           "num": 1,
           "args": {
               "N": A.shape[0],
               "K": 2,
               "max_K_increase": 0,
               "n_parts": 1,
           },
       },
       final_master_solve=True,
       max_iterations=3,
       disable_tqdm=True,
       verbose=0,
   )

   result = run_csd_decomposition(A, config=config)
   if result.final_partition is None:
       raise RuntimeError(result.metadata["status"])

   print(result.final_partition)
   print(result.metadata)

The node identifiers are matrix row numbers. When an application starts from a
labeled NetworkX graph, retain a separate row-to-label mapping or use one of
the label-aware application workflows.

Understand the result
---------------------

``result.final_partition`` is an ``N x N`` co-membership matrix, or ``None``
when no integral final partition is available. ``result.final_master_obj`` is
the final master objective when one was solved. ``result.records`` contains an
:class:`~asunder.types.IterationRecord` for each decomposition iteration.
During column generation, each record's ``columns`` and ``f_stars`` are
read-only views of the pool as it stood at that iteration; use ``list(...)``
if an independent mutable list is needed.

The metadata status distinguishes common outcomes:

``"ok"``
   A structurally valid integral partition was selected.

``"infeasible"``
   The decomposition returned no iteration records, commonly because initial
   feasibility or a master solve failed.

``"no_integral_partition"``
   Iterations ran, but no final integral partition was available. Enable a
   final master solve or a suitable post-loop refinement when one is required.

Configuration controls
----------------------

``ifc_params`` defines the initial-column generator, number of columns, and
generator arguments. ``algo`` and ``package`` select the pricing backend.
``resolution`` changes modularity resolution. ``max_iterations``,
``check_flat_pricing``, and ``stopping_window`` control termination.

``refine_params`` accepts a refinement callable and its keyword arguments.
``use_refined_column`` controls in-loop refinement, while
``refine_post_loop`` controls the final pass. Complete callable signatures and
matrix requirements are documented in
:doc:`../reference/development/special_topics`.

Configure ``must_link`` and ``cannot_link`` once on the main workflow.
Repeating these two keys in generator arguments, refinement kwargs, or pricing
parameters raises ``ValueError``. Generators and refiners receive the resolved
pairwise constraints through their declared parameters. Master worthy-edge
rules contribute their implied must-links too.

Supply shared node weights once with ``node_weights=[...]`` on the config or
``run_csd_decomposition``. They default to ones and are summed when nodes are
contracted. Compatible hooks receive the same vector through ``node_weights``
or ``balance_weights``; repeated standard weight arguments must agree. The
vector does not change adjacency weights or the modularity objective.
Built-in LB/VFD algorithms require positive integer weights; generic CSD can
forward finite real weights to custom hooks. See
:doc:`../reference/load_balancing` for explicit scaling and weighted ``K/R``
semantics.

Non-pairwise settings, including balance bounds and custom constraints,
remain configurable through each hook's arguments. ``additional_constraints``
supplies master settings; CSD does not copy balance settings from the master
into the hooks. Top-level workflows such as ``LoadBalancer`` configure those
hooks for you. Additional application-specific weight vectors belong in hook
arguments under distinct names. Search settings remain configurable as before.

For large sparse graphs, ``column_storage="auto"`` stores low-density hard
columns as CSR and dense columns as Boolean arrays. See
:doc:`../reference/matrix_storage` for memory controls and custom-callable
requirements.

Persistent master (optional)
----------------------------

Set ``persistent_master=True`` in the config or pass it to
``run_csd_decomposition`` when rebuilding the restricted master each iteration
becomes costly. Asunder then keeps one master model for the run, adds new
columns to it, and switches to binary variables for the final integer solve.
This is opt-in, uses the same master formulation, and closes the model at the
end of the run. It supports only the built-in base and load-balancing masters
with a configured ``gurobi_direct`` or ``gurobi_persistent`` solver and a
working Gurobi license; custom masters and other solvers are rejected.
Benchmark it against the default rebuild mode: the retained solver model may
use more memory. ``LoadBalancer`` also accepts ``persistent_master=True``.

Custom constraints
------------------

The generic decomposition accepts pairwise constraints directly. ModularVFD
can additionally enforce component-local, community-wide, and partition-wide
constraints during refinement. See
:doc:`../reference/development/extending_modular_vfd` for a complete runnable
example and extension protocols.

A ModularVFD constraint governs only ModularVFD-generated refinements unless
the same rule is also enforced by initial-column generation, pricing, the
master formulation, warm starts, and final validation.

Next steps
----------

- See :doc:`../api/top_level` for configuration and result APIs.
- See :doc:`../reference/development/special_topics` before replacing a
  callable.
- Use :doc:`nlbnp` for the packaged nonlinear branch-and-price workflow.
