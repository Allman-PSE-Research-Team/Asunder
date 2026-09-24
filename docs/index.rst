Asunder Documentation
=====================

.. image:: ../assets/asunder.gif
   :alt: A banner image for the Asunder package.

Asunder partitions an undirected graph while respecting grouping rules. A node
can represent a task, constraint, asset, or other item; an edge records a
relationship between two nodes; and the result assigns every node to a
community.

For example, the load-balancing workflow can require two nodes to stay
together, prevent another pair from sharing a community, and keep community
sizes nearly equal. This example requires an available Pyomo-compatible
solver. Asunder uses Gurobi by default; see
:doc:`getting_started/installation` for license and alternative-solver setup.

.. code-block:: python

   import networkx as nx

   from asunder.load_balancing import LoadBalancer

   graph = nx.Graph(
       [
           ("a", "b"),
           ("a", "c"),
           ("b", "c"),
           ("c", "d"),
           ("d", "e"),
           ("d", "f"),
           ("e", "f"),
       ]
   )

   result = LoadBalancer(
       graph,
       K=2,
       R=1,
       must_link=[("a", "b")],
       cannot_link=[("a", "f")],
       final_master_solve=True,
       disable_tqdm=True,
   )

   print(result.metadata["community_map_labels"])

Choose a workflow
-----------------

.. list-table:: Start from your goal
   :header-rows: 1
   :widths: 38 28 34

   * - Goal
     - Start with
     - Guide
   * - Create balanced or bounded graph communities
     - ``LoadBalancer``
     - :doc:`getting_started/quickstart`
   * - Assemble custom master, pricing, or refinement logic
     - ``run_csd_decomposition``
     - :doc:`getting_started/base_decomposition`
   * - Use the NLBNP linear-only-separator structural shortcut
     - ``CorePeripheryPartition``
     - :doc:`getting_started/nlbnp`
   * - Run the nonlinear branch-and-price workflow
     - ``NonlinearBranchAndPrice``
     - :doc:`getting_started/nlbnp`
   * - Decide whether Asunder fits a problem
     - Workflow-selection guidance
     - :doc:`learn/guides/problem_fit`

Start here
----------

- Install the package and configure a solver in
  :doc:`getting_started/installation`.
- Learn the basic terminology and package choices in
  :doc:`getting_started/introduction`.
- Follow one complete workflow under :doc:`getting_started/index`.
- Use the :doc:`api/index` only after choosing a workflow.

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   getting_started/index

.. toctree::
   :maxdepth: 2
   :caption: Learn

   learn/index

.. toctree::
   :maxdepth: 2
   :caption: API

   api/index

.. toctree::
   :maxdepth: 2
   :caption: Reference

   reference/index
