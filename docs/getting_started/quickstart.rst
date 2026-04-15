Quickstart Guide
================

This page shows the smallest realistic workflow that exercises the public
decomposition API. The example is intentionally simple and uses a trivial
initial feasible column generator so that the focus stays on the package
structure and call pattern.

Minimal Decomposition Run
-------------------------

.. code-block:: python

   import numpy as np

   from asunder import CSDDecompositionConfig, run_csd_decomposition


   def trivial_ifc_generator(N, **_):
       # One feasible starting column: all nodes placed in a single block.
       return [np.ones((N, N), dtype=int)]


   A = np.array(
       [
           [0, 1, 1, 0],
           [1, 0, 1, 0],
           [1, 1, 0, 1],
           [0, 0, 1, 0],
       ],
       dtype=float,
   )

   cfg = CSDDecompositionConfig(
       ifc_params={
           "generator": trivial_ifc_generator,
           "num": 1,
           "args": {"N": A.shape[0]},
       },
       extract_dual=False,
       final_master_solve=False,
       max_iterations=2,
       verbose=0,
   )

   result = run_csd_decomposition(A, config=cfg)

   print(result.metadata)
   print(result.final_partition)

This example uses the top-level facade. Internally, the top-level API delegates
to the reusable modules in ``asunder.base``.

Important Practical Note
------------------------

The decomposition loop expects an initial feasible column generator unless you
provide an existing column pool. In many real applications, that generator is
problem-specific and deserves deliberate design. The trivial example above is
appropriate only for smoke tests and first experiments.

Using Canonical Namespaces
--------------------------

The canonical reusable imports now live under ``asunder.base``. For example:

.. code-block:: python

   from asunder.base.column_generation.subproblem import heuristic_subproblem
   from asunder.base.algorithms.modular_VFD import modular_very_fortunate_descent

The nonlinear branch-and-price application pieces live under ``asunder.nlbp``:

.. code-block:: python

   from asunder.nlbp.case_studies import build_circle_cutting_graph, run_evaluation
   from asunder.nlbp.algorithms.refinement import refine_partition_linear_group

Built-In NLBP Evaluation
------------------------

If you want to use the packaged nonlinear branch-and-price evaluation flow, the
top-level convenience wrapper remains available:

.. code-block:: python

   from asunder import run_evaluation

   results = run_evaluation(
       problem="cpcong",
       build_params={"K": 2, "J": 3, "T": 5},
       style="CP",
       algos=["SPEC"],
       repeat=1,
   )

   print(results["SPEC"])

Where To Go Next
----------------

- For direct reusable APIs, see ``API -> Base API``.
- For the built-in NLBP workflow, see ``API -> NLBP API``.
- For a fuller explanation of when the package works well as-is versus when you
  should customize it, see :doc:`../learn/guides/problem_fit`.
