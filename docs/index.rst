Asunder Documentation
=====================

.. image:: ../assets/asunder.gif
   :alt: A banner image for the asunder package.

Asunder is a package for constrained structure detection on undirected graphs.
In machine learning terms, this is constrained graph clustering; in other
communities, it is constrained graph partitioning.

The fastest built-in workflow is load-balanced graph partitioning:

.. code-block:: python

   from asunder.load_balancing import LoadBalancer

   result = LoadBalancer(G, K=4, R=1)

Use ``asunder.load_balancing`` when you have a graph and need communities whose
sizes are equal, near-equal, or bounded by explicit lower and upper limits. The
workflow includes initial feasible partition generation, load balancing
constraints, master problem handling, and refinement.

Asunder also provides ``asunder.base`` for reusable decomposition and
column-generation building blocks, plus ``asunder.nlbnp`` for nonlinear
branch-and-price workflows. Use ``CorePeripheryPartition`` when core removal
exposes final connected-component communities, or ``NonlinearBranchAndPrice``
when those components require finer subdivision through column generation.

Start here:

- :doc:`getting_started/quickstart` for load balancing and decomposition
  examples.
- :doc:`learn/guides/problem_fit` to decide between the load balancing,
  reusable base, and nonlinear branch-and-price workflows.
- :doc:`api/load_balancing/index` for the load balancing API reference.
- :doc:`api/nlbnp/index` for the NLBNP workflows and API reference.

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
