Quickstart Guide
================

This page shows the two most useful starting points: the packaged load
balancing workflow and the lower-level decomposition API. Start with load
balancing when you need balanced graph communities. Drop to the decomposition
API when you need to customize the column-generation pieces directly.

Load Balancing Run
------------------

.. code-block:: python

   import networkx as nx

   from asunder.load_balancing import LoadBalancer


   G = nx.Graph()
   G.add_edges_from(
       [
           ("a", "b"),
           ("a", "c"),
           ("b", "c"),
           ("c", "d"),
           ("d", "e"),
           ("e", "f"),
           ("d", "f"),
       ]
   )

   result = LoadBalancer(
       G,
       K=2,
       R=1,
       must_link=[("a", "b")],
       cannot_link=[("a", "f")],
       disable_tqdm=True,
   )

   print(result.final_partition)
   print(result.metadata["modularity"])

``LoadBalancer`` accepts node labels from the input graph in ``must_link`` and
``cannot_link`` constraints, then returns a ``DecompositionResult`` whose
partition and metadata map communities back to those labels. Use ``K`` and ``R`` for near-equal
communities, or use ``R_bounds=(lower, upper)`` when every community must stay
inside explicit size bounds.

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

The canonical reusable imports live under ``asunder.base``. For example:

.. code-block:: python

   from asunder.base.column_generation.subproblem import heuristic_subproblem
   from asunder.base.algorithms.modular_VFD import modular_very_fortunate_descent

The generic nonlinear branch-and-price workflow and application pieces live
under ``asunder.nlbnp``:

.. code-block:: python

   from asunder.nlbnp import CorePeripheryPartition, NonlinearBranchAndPrice
   from asunder.nlbnp.case_studies import build_circle_cutting_graph, run_evaluation
   from asunder.nlbnp.algorithms.refinement import refine_partition_linear_group, refine_partition_with_cp

Use ``NonlinearBranchAndPrice`` when you already have a graph and want the NLBNP
column-generation workflow without a packaged case-study builder:

.. code-block:: python

   import networkx as nx

   from asunder.nlbnp import NonlinearBranchAndPrice


   G = nx.Graph()
   G.add_edge("u", "v", edge_kind="integer")
   G.add_edge("v", "w", edge_kind="continuous")

   result = NonlinearBranchAndPrice(
       G,
       worthy_edge_attr="edge_kind",
       worthy_edge_value="integer",
       algorithm="louvain",
       package="networkx",
       disable_tqdm=True,
   )

   print(result.final_partition)

Use ``CorePeripheryPartition`` when removing the detected core leaves connected
periphery components that are already suitable final communities:

.. code-block:: python

   community_labels, metadata = CorePeripheryPartition(
       G,
       unworthy_edge_attr="edge_kind",
       unworthy_edge_value="continuous",
       cp_algorithm="SPEC",
   )

   print(metadata["community_map_labels"])

Use ``NonlinearBranchAndPrice`` when the community structure is beyond the direct core-periphery logic. Core-periphery detection can be used through the generic
``refine_params`` hook:

.. code-block:: python

   result = NonlinearBranchAndPrice(
       G,
       refine_params={
           "refine_func": refine_partition_with_cp,
           "kwargs": {
               "unworthy_edges": [(1, 2)],
               "nonlinear_nodes": [],
               "cp_algorithm": "SPEC",
           },
       },
       disable_tqdm=True,
   )

The load balancing application pieces live under ``asunder.load_balancing``:

.. code-block:: python

   from asunder.load_balancing import LoadBalancer
   from asunder.load_balancing.utils.partition_generation import make_partitions

Built-In NLBNP Case-Study Evaluation
-----------------------------------

If you want to use the packaged nonlinear branch-and-price benchmark evaluation
flow, the top-level convenience wrapper remains available:

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
- For balanced graph partitioning, see ``API -> Load Balancing API``.
- For the generic and case-study NLBNP workflows, see ``API -> NLBNP API``.
- For a fuller explanation of when the package works well as-is versus when you
  should customize it, see :doc:`../learn/guides/problem_fit`.
