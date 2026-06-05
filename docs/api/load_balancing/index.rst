Load Balancing API
==================

The ``asunder.load_balancing`` package contains the built-in workflow for
load-balanced graph partitioning. Start with ``LoadBalancer`` when you want a
complete workflow that accepts a ``networkx.Graph`` and returns a
``DecompositionResult`` with a partition matrix plus label-aware metadata.

.. code-block:: python

   from asunder.load_balancing import LoadBalancer

   result = LoadBalancer(G, K=4, R=1)

.. toctree::
   :maxdepth: 2

   column_generation/index
   algorithms/index
   utils/index
