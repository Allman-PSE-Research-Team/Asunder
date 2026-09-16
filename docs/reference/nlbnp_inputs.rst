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
   ``worthy_edge_attr`` and ``worthy_edge_value``. At least one worthy edge is
   required for every cardinality mode, and every supplied pair must be a
   nonzero structural edge in the input graph. Missing, empty, and non-edge
   specifications raise ``ValueError``.

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

Generic nonlinear branch-and-price input
----------------------------------------

``NonlinearBranchAndPrice`` does not require the packaged case-study schema.
It can operate directly on adjacency data plus explicit ``worthy_edges``,
``nonlinear_nodes``, ``must_link``, and ``cannot_link`` inputs. Nonlinear nodes
can instead be derived from a NetworkX node attribute. Optional custom
initial-column, master, pricing, refinement, and additional-constraint
configuration is passed through the reusable decomposition layer.

Cardinality modes
^^^^^^^^^^^^^^^^^

The default ``cardinality_method="reformulated"`` computes the exact maximum
linear-only set. It forms connected components from unworthy edges and
must-links, keeps every component containing no nonlinear node, and returns
their union as ``eligible_nodes``. A Boolean ``y`` vector and its cardinality
``K_max`` are included in result metadata. Anchor-based must-links join that
union into one community and cannot-links separate it from ineligible nodes;
automatic contraction then applies those exact relationships before column
generation. Explicit ``contract_graph=False`` is available for diagnostics,
but usually wastes work.

An explicit cannot-link inside the maximum eligible set is infeasible, as is a
cannot-link inside any component already joined by an unworthy edge or
must-link. These conditions return a :class:`~asunder.types.DecompositionResult`
with ``status="infeasible"``, a specific ``infeasible_reason``, and no final
partition. Omitting the edge rule is an input error for every NLBNP method.

``cardinality_method="confidence"`` and
``cardinality_method="core_periphery"`` are heuristic alternatives. Both
start their authoritative refinement from an integral column that already
satisfies active pair and edge constraints. Confidence refinement merges all
existing linear-only communities and adds low-confidence eligible components.
Core-periphery refinement treats the detected periphery as the linear-only
group, merges residual linear-only communities into it, and preserves the
other assignments. Their method-specific arguments belong in
``cardinality_params``.

General column refinement is a separate Stage 1 concern.
``refine_params`` is passed to :func:`asunder.run_csd_decomposition`, while
``use_refined_column`` and ``refine_post_loop`` control its in-loop and
post-loop calls. The post-loop callable receives a fractional co-association
matrix and must support that input. Confidence or core-periphery cardinality
refinement then runs as Stage 2 on the selected hard Stage 1 partition; it is
never substituted into ``refine_params`` by the NLBNP wrapper.

When ``final_master_solve`` is false, the wrapper uses stored column objective
values to inspect the best candidate first and stops at the first partition
that passes its independent pairwise, edge, and cardinality validation. A
custom constraint enforced only by a custom master is outside that validator;
use a final integer master solve or add equivalent application-level final
validation when such a constraint must govern selection.

Neither heuristic refinement is the same operation as
``CorePeripheryPartition``. The latter is a solver-free shortcut that discards
the prior partition, excludes the detected linear-only group from the original
graph, and assigns its remaining connected components as the final independent
communities.

Large graphs and accelerators
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Asunder does not enable a GPU backend globally. If a compatible NetworkX GPU
backend is installed and configured by the user, a custom NetworkX pricing
callable may take advantage of it; no Asunder dependency or import change is
required. A GPU-capable NetworkX Leiden implementation can likewise be supplied
as a custom pricing subproblem. Backend availability, supported operations,
and data-transfer behavior remain the responsibility of that callable.

For large Gurobi runs, reduce the problem before tuning the solver: use exact
contraction where valid, keep the initial column pool small, cap
``max_iterations``, enable flat-pricing termination, and avoid
``final_master_solve`` unless an integer master decision is needed. Heuristic
pricing keeps repeated pricing work out of an ILP, but does not remove the
restricted master or its Python-side column pool.

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
