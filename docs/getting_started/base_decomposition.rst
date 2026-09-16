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
               "must_link": must_link,
               "cannot_link": cannot_link,
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

For large sparse graphs, ``column_storage="auto"`` stores low-density hard
columns as CSR and dense columns as Boolean arrays. See
:doc:`../reference/matrix_storage` for memory controls and custom-callable
requirements.

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
