Adding Constraints to ModularVFD
================================

This guide explains how to add application-specific hard constraints to
ModularVFD. It starts with a complete example and concludes with a cover custom
repair logic and the high-performance local protocol.

What ModularVFD does
--------------------

ModularVFD is a partition-refinement heuristic. It starts from a proposed
partition, searches for a partition with a better modularity objective, and
returns only a result that satisfies every configured hard constraint. It
returns ``None`` when it cannot find a feasible result within the configured
search budget.

The easiest entry point for an existing partition is
:func:`~asunder.base.algorithms.modular_VFD.refine_partition_modular_vfd`.
It accepts an adjacency matrix and either a one-dimensional community-label
vector or a two-dimensional co-membership matrix. The lower-level
:func:`~asunder.base.algorithms.modular_VFD.modular_very_fortunate_descent`
also requires the graph strengths ``a`` and volume ``m`` and returns diagnostic
metadata alongside the partition.

Five terms are useful throughout this guide:

``node``
   One application item, normally represented by one row of the input
   adjacency matrix.

``component``
   One or more nodes that ModularVFD must move together. Must-link constraints
   contract their nodes into a single component.

``community``
   A group of components. Community numbers are temporary search labels, not
   durable application roles.

``partition`` or ``assignment``
   The complete collection of communities.

``transition``
   One atomic change to an assignment, such as a move or swap. Every part of
   an atomic transition is checked together.

Before writing a custom constraint, use the built-in arguments when they fit:

- pass node pairs through ``must_link`` and ``cannot_link``;
- enable fixed-K load balancing with ``use_K_constraint=True`` and ``K`` plus
  either ``R`` or ``R_bounds``; and
- pass ``balance_weights`` when balance means load rather than node count.

First complete example
----------------------

Suppose each node has an application-defined category and no community may mix
categories. A community predicate is enough because the rule can be checked
one community at a time:

.. code-block:: python

   import numpy as np

   from asunder.base.algorithms import (
       CommunityPredicateConstraint,
       refine_partition_modular_vfd,
   )
   from asunder.base.utils import partition_matrix_to_vector

   node_names = ("api-a", "api-b", "db-a", "db-b")
   node_category = {
       "api-a": "application",
       "api-b": "application",
       "db-a": "database",
       "db-b": "database",
   }

   A = np.array(
       [
           [0.0, 3.0, 0.0, 0.0],
           [3.0, 0.0, 1.0, 0.0],
           [0.0, 1.0, 0.0, 3.0],
           [0.0, 0.0, 3.0, 0.0],
       ]
   )
   initial_labels = np.array([0, 0, 1, 1])

   def is_category_homogeneous(community):
       categories = {node_category[node] for node in community.nodes}
       return len(categories) <= 1

   homogeneous = CommunityPredicateConstraint(
       is_category_homogeneous,
       name="each community has one category",
   )

   refined = refine_partition_modular_vfd(
       A,
       initial_labels,
       candidate_Ks=(2,),
       constraints=(homogeneous,),
       # Map adjacency rows to the identifiers used by node_category.
       component_members=tuple((node,) for node in node_names),
       # The supplied partition is already useful co-membership data.
       wz_is_C_node=True,
       # Keep this tutorial run short; increase these budgets for real inputs.
       restarts=1,
       local_iters=5,
       tabu_max_steps=5,
       shake_rounds=0,
       seed=42,
   )

   if refined is None:
       raise RuntimeError("No feasible refinement was found")

   refined_labels = partition_matrix_to_vector(refined)
   for community_id in np.unique(refined_labels):
       members = {
           node_names[index]
           for index in np.flatnonzero(refined_labels == community_id)
       }
       assert len({node_category[node] for node in members}) == 1

``refined`` is a binary co-membership matrix: entry ``[i, j]`` is one when
nodes ``i`` and ``j`` share a community. The ``candidate_Ks=(2,)`` argument
keeps this example to one fixed community count. Supply other candidate counts
when the number of communities is part of the search.

Choosing the constraint type
----------------------------

Choose the narrowest type that contains all the information needed by the
rule. Narrow checks avoid rebuilding or scanning unrelated communities.

.. list-table:: Constraint selection
   :header-rows: 1
   :widths: 24 40 36

   * - Type
     - Use it when
     - Examples
   * - Built-in arguments
     - The rule is pairwise or ordinary fixed-K load balancing.
     - Must-link, cannot-link, community-size or load bounds.
   * - ``CommunityPredicateConstraint``
     - Every community can be checked independently.
     - Allowed member types, homogeneity, capacity, connectivity.
   * - ``QuantifiedCommunityConstraint``
     - The rule counts communities matching another predicate.
     - Exactly one special community, at least one qualifying community.
   * - ``PartitionPredicateConstraint``
     - The rule compares communities or examines the assignment as a whole.
     - Load spread, relationships between communities, existential rules.
   * - ``VFDLocalConstraint``
     - A frequently evaluated rule needs an incremental cache for speed.
     - Large conflict sets, cached capacities, specialized state machines.

All custom constraints use the same ``constraints`` argument. Pass several
constraint objects in one tuple when a refinement must satisfy several rules.

Community predicates
--------------------

A :class:`~asunder.base.algorithms.vfd_constraints.CommunityPredicateConstraint`
receives a read-only
:class:`~asunder.base.algorithms.vfd_constraints.VFDCommunityView`. The most
commonly useful properties are:

``community.nodes``
   The flattened original node identifiers in the community.

``community.components``
   The internal component identifiers in the community.

``community.component_members``
   The original node identifiers grouped by component.

``community.total_weight``
   The sum of its component weights.

``community.component_adjacency``
   Its induced component-level adjacency matrix. Accessing this property
   materializes the matrix, so use it only when the rule needs topology.

``community.is_empty``
   Whether the community currently contains no component. Empty communities
   are skipped unless the constraint is created with ``include_empty=True``.

A predicate returns ``True`` when the community is allowed and ``False`` when
it is forbidden. Keep predicates pure: their result should depend only on the
view and immutable application data.

Construction is incremental. If a community that is currently incomplete may
fail the final predicate but become valid after receiving more components,
supply a ``partial_predicate``. For example, a final rule requiring exactly two
components must permit sizes zero and one during construction:

.. code-block:: python

   from asunder.base.algorithms import (
       CommunityPredicateConstraint,
       VFDConstraintEvaluation,
   )

   def has_two_components(community):
       return len(community.components) == 2

   def can_reach_two_components(community):
       size = len(community.components)
       if size == 2:
           return VFDConstraintEvaluation.ok()
       return VFDConstraintEvaluation.violated(
           abs(2 - size),
           extendable=size < 2,
           reason="community size has not reached two",
       )

   pairs = CommunityPredicateConstraint(
       has_two_components,
       partial_predicate=can_reach_two_components,
       name="two components per community",
   )

Without a ``partial_predicate``, a false community predicate rejects that
partial placement. This is appropriate for rules such as maximum capacity or
forbidden category mixing, which cannot be repaired merely by adding another
member.

Counting qualifying communities
-------------------------------

Use :class:`~asunder.base.algorithms.vfd_constraints.QuantifiedCommunityConstraint`
when the requirement is a count of communities satisfying an arbitrary
community predicate. This example requires exactly one nonempty community
containing only marked nodes:

.. code-block:: python

   from asunder.base.algorithms import QuantifiedCommunityConstraint

   marked = {"api-a", "api-b"}

   def is_marked_only(community):
       members = set(community.nodes)
       return bool(members) and members <= marked

   exactly_one_marked = QuantifiedCommunityConstraint(
       is_marked_only,
       exactly=1,
       name="exactly one marked-only community",
   )

The predicate is application-defined. It may inspect a node attribute,
geography, capacity class, or any other metadata. Use ``minimum=1`` for "at
least one", ``maximum=2`` for "at most two", or ``minimum`` and ``maximum``
together for an inclusive range. ``exactly`` cannot be combined with either
bound.

Although the qualifying test examines one community, the count is a
partition-wide rule. A temporarily incorrect count remains repairable during
construction by default. Supply ``partial_evaluator`` only when application
knowledge can prove that a partial assignment can no longer reach the required
count.

Whole-partition predicates
--------------------------

Use :class:`~asunder.base.algorithms.vfd_constraints.PartitionPredicateConstraint`
when a rule compares communities or otherwise depends on the whole assignment.
Its callback receives a read-only
:class:`~asunder.base.algorithms.vfd_constraints.VFDAssignmentView`.

Useful assignment properties and methods include:

``assignment.iter_communities(include_empty=False)``
   Iterate over nonempty community views.

``assignment.community(community_id)``
   Inspect one community.

``assignment.component_community(component)``
   Find a component's current community, or ``None`` while it is unassigned.

``assignment.unassigned_components`` and ``assignment.is_complete``
   Inspect construction progress.

``assignment.partition_matrix()``
   Materialize a full co-membership matrix. Avoid this in frequently evaluated
   predicates when the lazy properties provide enough information.

The simplest form is a Boolean predicate:

.. code-block:: python

   from asunder.base.algorithms import PartitionPredicateConstraint

   allowed_spread = 3

   def communities(assignment):
       return tuple(assignment.iter_communities(include_empty=False))

   def load_spread(assignment):
       loads = [community.total_weight for community in communities(assignment)]
       return 0 if not loads else max(loads) - min(loads)

   spread_constraint = PartitionPredicateConstraint(
       lambda assignment: load_spread(assignment) <= allowed_spread,
       name="bounded load spread",
   )

This illustrates an R-only spread rule. The ordinary built-in load-balancing
path is faster when K and its size or load bounds are known. Also note that
ModularVFD currently searches a fixed K at a time: a partition predicate can
check the spread for that K, but it does not itself create or delete
communities.

Guided repair and violation scores
----------------------------------

After constructing an assignment, ModularVFD tries to repair unsatisfied
community and partition constraints before improving the objective. A
``violation`` callback tells the repair search how far it is from feasibility:

- return one or more finite, nonnegative numbers;
- return all zeros exactly when the predicate returns ``True``;
- always return the same number of values; and
- keep the score deterministic.

Smaller tuples are better. Multiple constraints are compared
lexicographically in the order supplied through ``constraints``. Modularity
breaks ties between equal violation scores.

The load-spread rule can provide both a violation score and a focused repair
proposal:

.. code-block:: python

   from asunder.base.algorithms import (
       PartitionPredicateConstraint,
       VFDConstraintEvaluation,
       VFDTransition,
   )

   allowed_spread = 3

   def communities(assignment):
       return tuple(assignment.iter_communities(include_empty=False))

   def load_spread(assignment):
       loads = [community.total_weight for community in communities(assignment)]
       return 0 if not loads else max(loads) - min(loads)

   def spread_is_allowed(assignment):
       return load_spread(assignment) <= allowed_spread

   def spread_violation(assignment):
       return (max(0, load_spread(assignment) - allowed_spread),)

   def propose_spread_repair(assignment):
       groups = communities(assignment)
       if len(groups) < 2:
           return ()

       heavy = max(groups, key=lambda group: group.total_weight)
       light = min(groups, key=lambda group: group.total_weight)
       if not heavy.components:
           return ()

       weights = assignment.context.component_weights
       component = min(heavy.components, key=weights.__getitem__)
       return (
           VFDTransition.move(
               (component,),
               source=heavy.community_id,
               target=light.community_id,
               phase="repair",
               label="move a light component out of the heaviest community",
           ),
       )

   spread_constraint = PartitionPredicateConstraint(
       spread_is_allowed,
       violation=spread_violation,
       # Current spread cannot prove an incomplete assignment impossible.
       partial_evaluator=lambda assignment: VFDConstraintEvaluation.ok(),
       repair_proposals=propose_spread_repair,
       name="bounded load spread",
   )

Custom proposals supplement ModularVFD's ordinary move, swap, and ejection
neighborhoods. A proposal may still be rejected because of another hard
constraint, an empty-community result, or an invalid source assignment.

Use :meth:`~asunder.base.algorithms.vfd_constraints.VFDTransition.swap` for a
two-way atomic swap. For a larger coordinated change, construct one
:class:`~asunder.base.algorithms.vfd_constraints.VFDTransition` from several
:class:`~asunder.base.algorithms.vfd_constraints.VFDComponentMove` objects.
The whole transition is checked before it commits, so an atomic swap is not
mistaken for two temporarily infeasible sequential moves.

``constraint_repair_steps`` controls the guided-repair budget. ``None`` derives
a bounded budget from ``local_iters``. If repair fails across the configured
restarts and candidate K values, ModularVFD returns ``None``.

Node identity, must-link contraction, and provenance
----------------------------------------------------

Constraint callbacks operate on components. A component is normally one input
row, but an internal must-link contraction can combine several rows. The
``component_members`` argument tells ModularVFD which application identifiers
are represented by each input row.

For an uncontracted labeled graph, supply one identifier per row:

.. code-block:: python

   node_order = ("north-a", "north-b", "central", "south")
   component_members = tuple((node,) for node in node_order)

If ``component_members`` is omitted, ModularVFD uses integer row numbers. When
the adjacency was contracted before calling ModularVFD, list every original
identifier represented by each contracted row:

.. code-block:: python

   component_members = (
       ("north-a", "north-b"),
       ("central",),
       ("south-a", "south-b", "south-c"),
   )

Pass this value as ``component_members=component_members`` when refining the
corresponding three-row contracted adjacency.

If ModularVFD performs another must-link contraction, it combines these member
groups automatically. Never assume that one component represents one original
node.

The immutable
:class:`~asunder.base.algorithms.vfd_constraints.VFDConstraintContext` also
provides the input and component adjacencies, input-row-to-component mapping,
component weights, candidate K, and active balance bounds. Community IDs may
be permuted without changing a partition, so select special communities by
their contents rather than by permanent numeric IDs.

Advanced: the incremental local protocol
----------------------------------------

Most extensions should use one of the predicate wrappers above. Implement the
:class:`~asunder.base.algorithms.vfd_constraints.VFDLocalConstraint` protocol
only when profiling shows that constructing community views or rescanning
members is too expensive.

A local constraint has three layers:

1. The specification stores user configuration and implements
   ``prepare(context)``.
2. The prepared object stores immutable data for one contracted problem and
   implements ``bind()``.
3. The bound runtime owns mutable state for one restart.

ModularVFD prepares the specification for each candidate K and creates a fresh
bound runtime for every restart. The following capacity constraint is a
minimal structural example:

.. code-block:: python

   from asunder.base.algorithms import VFDConstraintEvaluation

   class CapacityConstraint:
       def __init__(self, capacity):
           self.capacity = float(capacity)

       def prepare(self, context):
           return PreparedCapacity(context, self.capacity)


   class PreparedCapacity:
       def __init__(self, context, capacity):
           self.context = context
           self.capacity = capacity

       def bind(self):
           return BoundCapacity(self.context, self.capacity)


   class BoundCapacity:
       scope = "local"
       name = "incremental community capacity"

       def __init__(self, context, capacity):
           self.context = context
           self.capacity = capacity
           self.loads = [0.0] * context.K

       def _loads_after(self, transition):
           loads = self.loads.copy()
           for move in transition.moves:
               weight = sum(
                   self.context.component_weights[component]
                   for component in move.components
               )
               if move.source is not None:
                   loads[move.source] -= weight
               if move.target is not None:
                   loads[move.target] += weight
           return loads

       def block_is_feasible(self, components):
           weight = sum(
               self.context.component_weights[component]
               for component in components
           )
           return weight <= self.capacity

       def evaluate_partial(self, assignment):
           # Transition checks already enforce capacity during construction.
           return VFDConstraintEvaluation.ok()

       def evaluate_component_transition(self, transition):
           excess = sum(
               max(0.0, load - self.capacity)
               for load in self._loads_after(transition)
           )
           if excess == 0:
               return VFDConstraintEvaluation.ok()
           return VFDConstraintEvaluation.violated(
               excess,
               extendable=False,
               reason="community capacity exceeded",
           )

       def component_transition_applied(self, transition):
           self.loads = self._loads_after(transition)

       def evaluate_final(self, assignment):
           excess = sum(
               max(0.0, community.total_weight - self.capacity)
               for community in assignment.iter_communities()
           )
           if excess == 0:
               return VFDConstraintEvaluation.ok()
           return VFDConstraintEvaluation.violated(
               excess,
               extendable=False,
               reason="final community capacity exceeded",
           )

       def component_proposals(self):
           return ()

The bound runtime receives component transitions directly on the candidate hot
path, avoiding assignment-view materialization. Its methods have distinct
roles:

``block_is_feasible(components)``
   Reject components only when they can never move together. Do not use this
   for a failure that depends on the destination community.

``evaluate_partial(assignment)``
   Decide whether the current construction can still become feasible.

``evaluate_component_transition(transition)``
   Check the complete atomic change against cached state.

``component_transition_applied(transition)``
   Update cached state after, and only after, an accepted commit.

``evaluate_final(assignment)``
   Perform a pure correctness check independent of mutable cache state.

``component_proposals()``
   Return no proposals or a small collection of focused atomic transitions.

Final candidates are checked using a fresh runtime. Consequently,
``evaluate_final`` must derive correctness from the supplied assignment rather
than trusting counters left by the search.

Performance guidance
--------------------

The pairwise-only path does not materialize whole-assignment views. Additional
constraint work is performed only when custom constraints are configured.

For large inputs:

- prefer a community predicate over a partition predicate when unrelated
  communities do not matter;
- avoid ``assignment.partition_matrix()`` unless the rule truly needs the full
  matrix;
- cache counts, loads, conflict masks, or other sufficient statistics in a
  local runtime after profiling demonstrates a need;
- update caches from atomic transitions instead of rescanning the partition;
- return a small, focused set of repair proposals; and
- set a finite ``constraint_repair_steps`` when evaluation or repair proposal
  generation is expensive.

Testing and debugging a constraint
----------------------------------

Give every constraint a descriptive ``name``; it appears in ModularVFD's
diagnostic metadata. Test at least these cases:

- a feasible input remains feasible;
- an infeasible but repairable input is repaired;
- an impossible constraint returns ``None``;
- must-linked nodes expose the expected combined ``component_members``;
- a proposed swap is checked atomically; and
- a stateful local constraint starts with fresh state on every restart.

Use a fixed ``seed`` and small explicit ``candidate_Ks`` while developing a
constraint. Validate the returned partition directly with the same
application rule rather than checking only that ModularVFD returned a value.

Integration boundaries
----------------------

The ``constraints`` argument governs refinement performed by
``modular_very_fortunate_descent`` and ``refine_partition_modular_vfd``. It
does not automatically add the same rule to Asunder's master problem, pricing
problem, initial-column generator, warm starts, or other column-generation
stages.

Therefore, a constraint used only by ModularVFD guarantees that ModularVFD's
returned refinement satisfies it, but other columns can still enter the pool.
Complete end-to-end enforcement remains future work and ideally includes:

- preserving ``component_members`` and other application data through every
  contraction outside ModularVFD;
- applying the rule to initial-column generation, pricing, master
  formulations, warm starts, and final solution validation;
- adding solver-backed formulations or specialized repair neighborhoods when
  generic guided repair is too weak; and
- returning explicit named community-role metadata if a future model needs
  durable roles rather than permutation-invariant content predicates.

API reference
-------------

The complete callback signatures and view properties are listed in
:doc:`../../api/base/algorithms/vfd_constraints`. The ModularVFD function
signatures are listed in :doc:`../../api/base/algorithms/modular_VFD`.
