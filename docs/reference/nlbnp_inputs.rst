NLBNP Inputs and Graph Contracts
================================

This reference collects detailed input rules for
``CorePeripheryPartition``, ``NonlinearBranchAndPrice``, and the packaged NLBNP
case-study evaluator. Start with :doc:`../getting_started/nlbnp` for runnable
workflow examples.

Accepted graph forms
--------------------

Both high-level NLBNP entry points accept:

- an undirected ``networkx.Graph`` with arbitrary hashable node labels; or
- a square adjacency matrix whose nodes are integer row indices.

For NetworkX input, pair constraints use the original graph labels and result
metadata includes maps between labels and matrix rows. For matrix input, pairs
use integer row indices.

Pair and edge constraints
-------------------------

``must_link``
   Nodes that must share a final community. In the direct NLBNP structural
   workflow, they also share a detection block.

``cannot_link``
   Nodes that must not share a final community. This is available in the
   nonlinear branch-and-price workflow.

``worthy_edges``
   Edges allowed to cross communities in the NLBNP decomposition model.
   ``NonlinearBranchAndPrice`` accepts explicit pairs or derives them from
   ``worthy_edge_attr`` and ``worthy_edge_value``.

When ``worthy_edge_value`` is ``None``, a truthy edge-attribute value selects
the edge. Otherwise, the attribute must equal the configured value.

NLBNP structural grouping
-------------------------

``CorePeripheryPartition`` additionally accepts ``must_group``. This places
the selected nonlinear nodes in one binary detection block but does not require
them to share a final community. The complementary periphery becomes the one
linear-only community. After that periphery is excluded, disconnected core-side
nodes can therefore remain in separate independent communities.

Must-link pairs can be supplied directly or derived with
``must_link_edge_attr`` and ``must_link_edge_value``. Must-group nodes can be
supplied directly or derived with ``must_group_node_attr`` and
``must_group_node_value``.

Contracted and original targets
-------------------------------

The default ``target="contracted"`` mode evaluates binary structure after
forming the must-link and nonlinear must-group detection blocks. If ``S`` maps
original nodes to those blocks, the aggregate adjacency is:

.. math::

   B = S^{\mathsf{T}} A S

Its row sums preserve aggregate block strength, not every original node's
degree. Diagonal entries retain edge weight internal to each block, and block
pairs are not density-normalized.

Use ``target="original"`` when the original adjacency is already the space
whose core-periphery structure matters and contraction should constrain nodes
to equal coreness without changing that objective space.

For its final partition, the direct workflow merges the entire periphery into
the single linear-only community, including periphery nodes that are mutually
disconnected. It then temporarily excludes that group from the **original
adjacency**, computes connected components of the remaining nonlinear/core-side
induced subgraph, and returns one community for each component. This step never
uses the contracted adjacency, although arbitrary ``must_link`` pairs are added
as virtual component edges. No nodes are removed from the final result.

When ``must_group`` designates nonlinear nodes, the workflow requires that they
appear on the core side. A result that places any of them in the periphery would
put nonlinear nodes in the purported linear-only group, so the workflow raises
``RuntimeError``. Without a designated nonlinear block, the node roles cannot
be independently validated. Choose ``NonlinearBranchAndPrice`` when this
structural shortcut is not valid.

Generic branch-and-price input
------------------------------

``NonlinearBranchAndPrice`` does not require the packaged case-study schema.
It can operate directly on adjacency data plus explicit ``worthy_edges``,
``must_link``, and ``cannot_link`` inputs. Optional custom initial-column,
master, pricing, refinement, and additional-constraint configuration is passed
through the reusable decomposition layer.

Packaged case-study schema
--------------------------

The built-in ``run_evaluation`` path is application-specific. Its packaged
case studies use an undirected constraint graph with these fields:

Required
^^^^^^^^

``constraint`` node attribute
   A string tag used for ground-truth roles and to identify designated
   nonlinear nodes in the packaged case studies.

``var_type`` edge attribute
   Either ``"integer"`` or ``"continuous"``. The evaluation runner uses this
   to derive edge subsets for its packaged CP and refinement paths.

Recommended
^^^^^^^^^^^

- node attributes ``type`` and ``details``;
- edge attribute ``weight``; and
- edge attributes ``variables`` and ``var_types`` when the case study tracks
  the originating optimization variables.

These names are conventions of the packaged evaluation workflow, not
requirements of the generic NLBNP or base decomposition APIs.

Result identity
---------------

For NetworkX input, metadata includes ``node_label_map`` and
``label_node_map``. Direct structural results also include
``community_map_labels`` and ``communities_labels``. Nonlinear branch-and-price
results include ``community_map_labels`` when an integral final partition is
available.

Direct structural metadata distinguishes detection roles from final roles:
``core_labels`` contains the binary core/periphery detection, ``n_linear_only``
counts nodes merged into community zero, ``independent_components`` contains
the original-graph core-side components, and ``component_graph_space`` is
``"original"``.

Community numbers are labels, not durable roles, except that the direct
structural workflow reserves community zero for the merged linear-only
periphery.

See :doc:`../api/nlbnp/index` for complete signatures and
:doc:`development/special_topics` for custom callable contracts.
