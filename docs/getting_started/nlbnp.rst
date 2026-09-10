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
Column generation enforces the edge rule. You also choose one of three ways to
enforce the requirement that exactly one community contain only linear nodes:

.. list-table:: Cardinality methods
   :header-rows: 1
   :widths: 20 45 35

   * - Method
     - What it does
     - When to use it
   * - ``"reformulated"`` (default)
     - Finds the maximum eligible linear-node set exactly, converts that set to
       must-link and cannot-link pairs, and contracts the resulting must-link
       components automatically.
     - Prefer this when exact constraint enforcement is priority.
   * - ``"confidence"``
     - In Stage 2, starts from a hard edge-feasible Stage 1 partition, merges any
       existing linear-only communities, and adds eligible low-confidence
       components to that one group.
     - Use when boundary linear nodes with low assignment confidence exist in
       the application.
   * - ``"core_periphery"``
     - In Stage 2, starts from a hard edge-feasible Stage 1 partition, detects a
       linear-only periphery, merges it with any existing linear-only
       communities, and preserves the other assignments.
     - Use when the graph contracts into the required binary structure that reveals
       the linear-only group, but column generation is preferred for detecting other
       communities.

The two heuristic methods always produce their authoritative result from a
hard feasible partition after column generation. They are not installed as
general column refiners and do not run on the fractional master combination.

This complete default-method example derives worthy edges and nonlinear nodes
from application-owned metadata:

.. code-block:: python

   import networkx as nx

   from asunder.nlbnp import NonlinearBranchAndPrice

   graph = nx.Graph()
   graph.add_nodes_from(
       [
           ("n1", {"kind": "nonlinear"}),
           ("n2", {"kind": "linear"}),
           ("n3", {"kind": "linear"}),
           ("n4", {"kind": "linear"}),
       ]
   )
   graph.add_edge("n1", "n2", relationship="integer")
   graph.add_edge("n2", "n3", relationship="continuous")
   graph.add_edge("n3", "n4", relationship="continuous")
   graph.add_edge("n1", "n4", relationship="integer")

   result = NonlinearBranchAndPrice(
       graph,
       worthy_edge_attr="relationship",
       worthy_edge_value="integer",
       nonlinear_node_attr="kind",
       nonlinear_node_value="nonlinear",
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
For the reformulated method, metadata also includes the Boolean ``y`` vector,
``eligible_nodes``, and the exact maximum ``K_max``. Derived pairwise
constraints are exposed separately from user-supplied pairs.

How the exact reformulation finds the group
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

An *unworthy edge* is a graph edge not listed in ``worthy_edges``. Such an edge
cannot cross a final community. The exact method forms components using every
unworthy edge plus every explicit must-link. A component is eligible when it
contains no designated nonlinear node. The union of all eligible components is
the maximum feasible linear-only set ``Y``.

Asunder joins all nodes in ``Y`` with a compact set of must-links and separates
``Y`` from every other node with cannot-links. It then uses the ordinary
pairwise machinery, including automatic contraction, rather than adding a new
master formulation. If an explicit cannot-link lies inside ``Y`` (or inside
any other required-together component), the result is immediately reported as
infeasible.

Every cardinality mode requires a nonempty set of worthy edges. Every supplied
pair must also be a nonzero edge in the input graph. Missing, empty, or
non-structural worthy-edge specifications raise ``ValueError`` because without
an active edge-based constraint the workflow is no longer NLBNP.

Stage 1 column refinement
^^^^^^^^^^^^^^^^^^^^^^^^^

Column refinement is optional and belongs entirely to Stage 1. Supply a
general CSD-compatible refiner through ``refine_params``. Then choose whether
it runs on each priced column with ``use_refined_column`` and/or on the final
fractional co-association matrix with ``refine_post_loop``. Neither switch
changes ``cardinality_method``. Enabling either switch without a callable
``refine_func`` raises ``ValueError``.

For example, this configures ModularVFD as both an in-loop and post-loop Stage
1 refiner. The selected cardinality method still runs independently:

.. code-block:: python

   import networkx as nx

   from asunder import refine_partition_modular_vfd
   from asunder.nlbnp import NonlinearBranchAndPrice

   graph = nx.path_graph(4)

   result = NonlinearBranchAndPrice(
       graph,
       worthy_edges=[(0, 1)],
       nonlinear_nodes=[0],
       cardinality_method="reformulated",
       refine_params={
           "refine_func": refine_partition_modular_vfd,
           "kwargs": {"local_iters": 20},
       },
       use_refined_column=True,
       refine_post_loop=True,
       disable_tqdm=True,
   )

The post-loop input is the fractional weighted co-association matrix produced
by the restricted master. A custom refiner must support that input itself;
ModularVFD does. Cardinality refiners instead receive only the hard partition
selected after Stage 1.

Using a heuristic cardinality method
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Only the method-specific options change:

.. code-block:: python

   import networkx as nx

   from asunder.nlbnp import NonlinearBranchAndPrice

   graph = nx.path_graph(4)
   common = {
       "worthy_edges": [(0, 1)],
       "nonlinear_nodes": [0],
       "disable_tqdm": True,
   }

   confidence_result = NonlinearBranchAndPrice(
       graph,
       **common,
       cardinality_method="confidence",
       cardinality_params={"threshold": 0.75},
   )

   cp_result = NonlinearBranchAndPrice(
       graph,
       **common,
       cardinality_method="core_periphery",
       cardinality_params={
           "cp_algorithm": "SPEC",
           "prob_method": "gaussian_mixture",
       },
   )

For confidence refinement, the largest existing linear-only community is only
the deterministic label anchor: all existing linear-only communities are
merged. If one low-confidence node belongs to a required-together component,
the whole eligible component moves. A component containing a nonlinear node is
never moved. ``prob_method="DBSCAN"`` treats the cluster with the lowest mean
maximum-assignment confidence as low confidence; cluster size does not assign
that role.

For core-periphery refinement, unworthy edges and explicit must-links constrain
the detector's grouping blocks. The detected periphery becomes the linear-only
group, residual linear-only communities are merged into it, and assignments on
the nonlinear/core side remain as column generation produced them. This differs
from ``CorePeripheryPartition``: the direct shortcut additionally excludes the
linear-only group from the original graph and replaces the remaining side with
its connected components.

Input and constraint choices
----------------------------

Both entry points accept an undirected NetworkX graph or a square adjacency
matrix. Pair constraints use graph labels for NetworkX input and integer row
indices for matrix input. ``NonlinearBranchAndPrice`` accepts explicit
``worthy_edges`` or can derive them with ``worthy_edge_attr`` and
``worthy_edge_value``.

The default initial-column generator receives explicit pair constraints and
the must-link implications of every unworthy edge. If you replace it through
``ifc_params``, the custom generator is responsible for the same feasibility.

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
nonlinear node appears in the linear-only periphery. The nonlinear 
branch-and-price workflow returns a result whose ``final_partition`` may be
``None``; inspect ``result.metadata["status"]`` to distinguish infeasibility
from a run that produced no integral final partition.

With ``final_master_solve=True``, the final integer master chooses the hard
column. Otherwise, the workflow orders the latest generated columns by their
already stored objective scores and independently validates candidates until
it finds the best feasible one. This avoids another solver call. Application
constraints implemented only inside a custom master still require an integer
master selection or equivalent application-level final validation.

Next steps
----------

- See :doc:`../reference/nlbnp_inputs` for graph attributes, contraction, and
  packaged case-study schemas.
- See :doc:`../api/nlbnp/index` for complete signatures.
- Use :doc:`base_decomposition` when building a different application on the
  reusable orchestration layer.
