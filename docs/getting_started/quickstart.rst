Load-Balancing Quickstart
=========================

Use ``LoadBalancer`` when you want a fixed number of graph communities whose
node counts or node loads are nearly equal.

Prerequisites
-------------

- Install Asunder as described in :doc:`installation`.
- Configure an available Pyomo-compatible solver. Gurobi is the default.
- Provide an undirected ``networkx.Graph`` with at least one edge.

Run a complete example
----------------------

This graph contains two dense regions joined by one edge. The example asks for
two nearly equal communities, keeps ``"a"`` and ``"b"`` together, and prevents
``"a"`` and ``"f"`` from sharing a community.

.. code-block:: python

   from collections import defaultdict

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

   groups = defaultdict(list)
   for node, community in result.metadata["community_map_labels"].items():
       groups[community].append(node)

   print(dict(groups))
   print("modularity:", result.metadata["modularity"])

The exact numeric community labels are arbitrary. What matters is which nodes
share a label.

Understand the result
---------------------

``result`` is a :class:`~asunder.types.DecompositionResult`.

``result.final_partition``
   An ``N x N`` binary co-membership matrix in the graph's node iteration
   order. Entry ``[i, j]`` is one when those nodes share a community.

``result.metadata["community_map_labels"]``
   A mapping from the original NetworkX node labels to community numbers.

``result.metadata["community_balance_weights"]``
   The final node count or total node load in each community.

``result.metadata["modularity"]``
   The modularity score at the configured ``resolution``.

``result.records``
   Per-iteration column-generation diagnostics. Most first-time users do not
   need to inspect them.

Balance controls
----------------

.. list-table:: Common options
   :header-rows: 1
   :widths: 25 75

   * - Option
     - Meaning
   * - ``K=4, R=1``
     - Request four communities whose allowed load range has width one.
   * - ``R_bounds=(10, 14)``
     - Require every community to have total load between 10 and 14,
       inclusive.
   * - ``node_weight_attr="load"``
     - Balance a positive integer node attribute instead of node count.
       Missing attribute values default to one.
   * - ``contract_graph=True``
     - Contract must-linked nodes before column generation while preserving
       their summed load.
   * - ``resolution=1.25``
     - Change the modularity resolution used by pricing and scoring.

Search and runtime controls
---------------------------

Signed Leiden is the default pricing heuristic. Select
``algorithm="qmetis"`` to use bundled QMETIS on a supported platform; see
:doc:`../reference/qmetis` for its approximation and platform details.

For large inputs, ``refine=False`` disables VFD refinement entirely.
``use_refined_column=False`` disables refinement inside the main loop, while
``refine_post_loop=False`` disables the final refinement pass.
``check_flat_pricing`` and ``stopping_window`` control early termination when
pricing stops improving.

Failure behavior
----------------

``LoadBalancer`` raises ``ValueError`` for invalid inputs or impossible bound
definitions. It raises ``RuntimeError`` if the search finishes without an
integral feasible partition. In that case, check the pairwise constraints and
bounds first, then consider a larger search budget or
``projection_repair=True`` when an appropriate solver is available.

Next steps
----------

- Use :doc:`base_decomposition` to replace master, pricing, or refinement
  logic.
- Use :doc:`../reference/development/extending_modular_vfd` to add custom hard
  constraints to ModularVFD refinement.
- See :doc:`../api/load_balancing/index` for the full load-balancing API.
