Nonlinear Branch-and-Price Workflows
====================================

The ``asunder.nlbnp`` package offers two entry points for decomposing a
constraint-interaction graph. ``CorePeripheryPartition`` is a structural
shortcut for the NLBNP pattern described below, not a general-purpose
core-periphery partitioning workflow.

.. list-table:: NLBNP entry points
   :header-rows: 1
   :widths: 31 43 26

   * - Entry point
     - Use it when
     - Solver required
   * - ``CorePeripheryPartition``
     - One linear-only separator group can be identified, and excluding it from
       the original graph reveals the desired independent communities.
     - No
   * - ``NonlinearBranchAndPrice``
     - The structural shortcut is insufficient and column generation is
       required.
     - Yes

How the NLBNP structural shortcut works
---------------------------------------

In the intended NLBNP use, graph nodes represent model constraints and edges
represent shared variables or another relevant coupling. The shortcut has two
conceptual stages:

1. Grouping constraints collect the designated nonlinear nodes into one
   detection block on the core side. Its complementary periphery contains only
   linear nodes. Merging that entire periphery supplies the cardinality
   requirement that exactly one community contain only linear nodes.
2. The merged linear-only periphery is temporarily excluded from the
   **original input graph**. Connected components of the remaining nonlinear
   core-side induced subgraph become the independent communities. The
   contracted graph used by detection is not used for this final component
   calculation.

``target="contracted"`` therefore changes the space used to detect and score
the binary separation; it does not change the graph used to recover the final
independent communities. ``must_link`` pairs are also added as virtual edges
when those original-graph components are calculated.

When ``must_group`` designates nonlinear nodes, the workflow verifies that the
detector put them on the core side. It raises ``RuntimeError`` if they instead
appear in the linear-only periphery. Without designated nonlinear nodes, that
role is inferred only from the detected structure and cannot be independently
validated.

Direct structural partitioning
------------------------------

This complete example uses the solver-free spectral path:

.. code-block:: python

   import networkx as nx

   from asunder.nlbnp import CorePeripheryPartition

   graph = nx.Graph()
   graph.add_nodes_from(
       ["nonlinear-a", "nonlinear-b", "linear-a", "local-b", "linear-b"]
   )
   graph.add_edges_from(
       [
           ("nonlinear-a", "linear-b"),
           ("nonlinear-b", "linear-a"),
           ("nonlinear-b", "local-b"),
           ("linear-a", "local-b"),
           ("local-b", "linear-b"),
       ]
   )

   labels, metadata = CorePeripheryPartition(
       graph,
       must_group=["nonlinear-a", "nonlinear-b"],
       cp_algorithm="SPEC",
       prob_method="threshold",
       threshold=0.5,
       seed=42,
   )

   print(dict(zip(graph.nodes(), labels)))
   print("core labels:", metadata["core_labels"])
   print("communities:", metadata["communities_labels"])

The returned one-dimensional ``labels`` vector uses graph iteration order.
The linear-only periphery is merged into community zero. Connected components
of the nonlinear/core side, found after excluding community zero from the
original graph, receive subsequent community labels. No nodes are discarded
from the returned result. Metadata maps the result back to the original graph
labels.

In this example, ``"linear-a"`` and ``"linear-b"`` share community zero even
though they are not adjacent. Excluding them from the original graph leaves
``"nonlinear-a"`` as one component and ``{"nonlinear-b", "local-b"}`` as
another.

``must_link`` keeps node pairs together in both the binary detection decision
and the final connected-component split. In NLBNP, ``must_group`` is used for
the nonlinear core detection block. It does not put those nodes in one final
community: after the linear-only periphery is excluded, disconnected core-side
nodes may belong to different independent communities.

Nonlinear branch and price
--------------------------

This workflow requires the solver setup described in :doc:`installation`.
The example derives worthy edges from application-owned edge metadata:

.. code-block:: python

   import networkx as nx

   from asunder.nlbnp import NonlinearBranchAndPrice

   graph = nx.Graph()
   graph.add_edge("n1", "n2", relationship="nonlinear")
   graph.add_edge("n2", "n3", relationship="linear")
   graph.add_edge("n3", "n4", relationship="linear")
   graph.add_edge("n1", "n4", relationship="nonlinear")

   result = NonlinearBranchAndPrice(
       graph,
       worthy_edge_attr="relationship",
       worthy_edge_value="nonlinear",
       final_master_solve=True,
       disable_tqdm=True,
   )

   if result.final_partition is None:
       raise RuntimeError(result.metadata["status"])

   print(result.metadata["community_map_labels"])
   print(result.final_partition)

``result`` is a :class:`~asunder.types.DecompositionResult`. Its final
partition is an ``N x N`` co-membership matrix, while
``community_map_labels`` maps original NetworkX labels to community numbers.

Input and constraint choices
----------------------------

Both entry points accept an undirected NetworkX graph or a square adjacency
matrix. Pair constraints use graph labels for NetworkX input and integer row
indices for matrix input. ``NonlinearBranchAndPrice`` accepts explicit
``worthy_edges`` or can derive them with ``worthy_edge_attr`` and
``worthy_edge_value``.

When pair constraints and worthy-edge rules are combined, the configured
initial-column generator must satisfy both. Supply a domain-aware generator
through ``ifc_params`` when the default pairwise generator cannot guarantee
that combined feasibility.

The direct structural workflow can derive must-link edges and nonlinear
must-group nodes from graph attributes. Its ``target`` parameter controls
whether the binary structure is optimized in contracted or original graph
space. Final components are always recovered from the original graph. See
:doc:`../reference/nlbnp_inputs` for the exact input contracts and target
semantics.

Failure behavior
----------------

Both workflows reject malformed graph and constraint inputs with
``ValueError``. Direct structural partitioning raises ``RuntimeError`` if
detection does not produce two nonempty binary sides or if a designated
nonlinear node appears in the linear-only periphery. The branch-and-price
workflow returns a result whose ``final_partition`` may be ``None``; inspect
``result.metadata["status"]`` to distinguish infeasibility from a run that
produced no integral final partition.

Next steps
----------

- See :doc:`../reference/nlbnp_inputs` for graph attributes, contraction, and
  packaged case-study schemas.
- See :doc:`../api/nlbnp/index` for complete signatures.
- Use :doc:`base_decomposition` when building a different application on the
  reusable orchestration layer.
