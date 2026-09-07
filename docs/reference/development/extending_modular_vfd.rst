Extending ModularVFD Constraints
================================

ModularVFD accepts more than pairwise ``must_link`` and ``cannot_link``
constraints. The ``constraints`` argument supports hard constraints at three
cost levels: component-local transitions, affected-community predicates, and
partition-wide predicates. Choose the narrowest level that can express the
rule. Doing so keeps the ordinary pairwise-only path fast and avoids rebuilding
the full partition for a check that only depends on one or two communities.

The constraint objects described here affect refinement performed by
``modular_very_fortunate_descent`` and ``refine_partition_modular_vfd``. They do
not automatically change the master problem, pricing problem, initial-column
generator, or other column-generation stages. See `Integration boundaries`_
before relying on a constraint as a model-wide guarantee.

Choosing a constraint tier
--------------------------

``VFDLocalConstraint``
   Use the local protocol when feasibility can be updated from component moves
   and small cached state. The built-in cannot-link and balance checks remain
   in the optimized internal implementation of this tier and are composed with
   the same atomic transition dispatcher. This is the preferred extension
   point for large inputs.

``CommunityPredicateConstraint``
   Use a community predicate when a rule must inspect all members of a changed
   community but does not depend on unrelated communities. Examples include an
   allow-list, a capacity derived from member metadata, homogeneous membership,
   or connectivity within each community. Only communities touched by an
   atomic transition are evaluated.

``PartitionPredicateConstraint`` and ``QuantifiedCommunityConstraint``
   Use a partition-wide constraint when feasibility depends on several
   communities together. ``QuantifiedCommunityConstraint`` is the convenient
   form for minimum, maximum, or exact counts of communities matching a
   predicate. Use ``PartitionPredicateConstraint`` for other relationships or
   a custom violation score.

All three forms are passed through one argument:

.. code-block:: python

   Z, metadata = modular_very_fortunate_descent(
       wz,
       A,
       a,
       m,
       constraints=(local_rule, community_rule, partition_rule),
   )

The built-in pairwise constraints remain the default. Load balancing remains
optional through ``use_K_constraint=True`` and does not need to be reproduced
as a custom constraint.

Component identity and provenance
---------------------------------

ModularVFD first contracts every must-link component. Constraint callbacks
therefore operate on *components*, not necessarily on individual input rows.
The immutable :class:`~asunder.base.algorithms.vfd_constraints.VFDConstraintContext`
contains the input and component adjacencies, the input-node-to-component map,
component weights, K and balance bounds, and the original members represented
by every component.

When ModularVFD receives the original adjacency, it creates that provenance (source reference)
itself. When another module contracts the graph first, pass ``component_members``
so predicates can still recover application metadata:

.. code-block:: python

   component_members = (
       ("north-a", "north-b"),
       ("central",),
       ("south-a", "south-b", "south-c"),
   )

   Z, metadata = modular_very_fortunate_descent(
       wz_contracted,
       A_contracted,
       a_contracted,
       m_contracted,
       component_members=component_members,
       constraints=(rule,),
   )

Do not assume that a component contains one original node, and do not treat a
temporary community number as a durable role. Community numbers can be
permuted without changing a partition. Select communities by their contents or
other predicates instead.

Affected-community predicates
-----------------------------

A community predicate receives a lazy
:class:`~asunder.base.algorithms.vfd_constraints.VFDCommunityView`. A predicate
should be pure: its return value must depend only on the supplied view and
immutable application data. ModularVFD checks it for the communities affected
by construction, move, swap, ejection, and shake transitions.

For example, an application can reject communities that mix two arbitrary
categories. The categories belong to the application; ModularVFD does not
assign meaning to them:

.. code-block:: python

   from asunder.base.algorithms import CommunityPredicateConstraint

   category = {
       "north-a": "field",
       "north-b": "field",
       "central": "office",
       "south-a": "field",
       "south-b": "field",
       "south-c": "field",
   }

   def is_category_homogeneous(community):
       kinds = {
           category[node]
           for component in community.component_members
           for node in component
       }
       return len(kinds) <= 1

   homogeneous = CommunityPredicateConstraint(is_category_homogeneous)

Pass ``partial_predicate`` when an incomplete community may temporarily fail
the final predicate but can become valid as unassigned components arrive. A
``violation`` callback supplies guided-repair distance, ``repair_proposals``
adds focused atomic changes, and ``co_movement_predicate`` can reject a
fingerprint block whose components can never travel together. Do not use the
co-movement hook for a failure that depends on the destination community.

The exact view properties are listed in the :doc:`../../api/base/algorithms/vfd_constraints`
reference. A constraint that can update a counter from a transition should use
the local protocol instead of repeatedly traversing ``component_members``.

Counting qualifying communities
-------------------------------

Use :class:`~asunder.base.algorithms.vfd_constraints.QuantifiedCommunityConstraint`
when the requirement is stated as a count of communities satisfying an
arbitrary predicate. For example, require exactly one nonempty community whose
members are all marked by the application:

.. code-block:: python

   from asunder.base.algorithms import QuantifiedCommunityConstraint

   marked = {"north-a", "north-b", "south-c"}

   def is_all_marked(community):
       members = {
           node
           for component in community.component_members
           for node in component
       }
       return bool(members) and members <= marked

   exactly_one_marked = QuantifiedCommunityConstraint(
       is_all_marked,
       exactly=1,
   )

The predicate is generic: it could inspect a node attribute, a capacity class,
geography, or any other application-owned metadata. ``minimum=1`` expresses
"at least one". ``maximum=2`` expresses "at most two". Supply ``minimum`` and
``maximum`` together for a range; ``exactly`` is mutually exclusive with those
bounds.

The count is partition-wide even though the qualifying test is
community-local. During construction, ModularVFD treats a temporarily
unsatisfied count as repairable. Supply ``partial_evaluator`` when application
knowledge can identify partial assignments that cannot reach the requested
range. Before objective refinement and before returning the result, the full
partition must satisfy the count. Empty communities are ignored by default;
set ``include_empty=True`` only when emptiness is part of the predicate.

Custom partition predicates and repair
--------------------------------------

Use :class:`~asunder.base.algorithms.vfd_constraints.PartitionPredicateConstraint`
for a relationship that cannot be reduced to a count of independent community
predicates. A partition constraint evaluates a lazy
:class:`~asunder.base.algorithms.vfd_constraints.VFDAssignmentView` and returns
a :class:`~asunder.base.algorithms.vfd_constraints.VFDConstraintEvaluation`.

The evaluation has two jobs:

- report whether a partial assignment remains extendable; and
- provide a tuple of nonnegative violation values for a complete assignment.

An all-zero tuple means feasible. Smaller tuples are better, compared
lexicographically in constraint declaration order. A violation callback must
always return the same number of finite, nonnegative values, and it must return
all zeros exactly when its predicate is satisfied. Make the score
deterministic. A useful score first measures the number of missing or excess
structures and then their magnitude.

This example limits the load spread across nonempty communities. It supplies a
global violation score and a small set of repair proposals. The example is
illustrative--the built-in balance constraint is more efficient for ordinary
load balancing:

.. code-block:: python

   from asunder.base.algorithms import (
       PartitionPredicateConstraint,
       VFDConstraintEvaluation,
       VFDTransition,
   )

   allowed_spread = 3

   def communities(assignment):
       return tuple(assignment.iter_communities(include_empty=False))

   def spread(assignment):
       loads = [community.total_weight for community in communities(assignment)]
       return 0 if not loads else max(loads) - min(loads)

   def spread_is_allowed(assignment):
       return spread(assignment) <= allowed_spread

   def spread_violation(assignment):
       return (max(0, spread(assignment) - allowed_spread),)

   def spread_repairs(assignment):
       groups = communities(assignment)
       if len(groups) < 2:
           return ()
       heavy = max(groups, key=lambda community: community.total_weight)
       light = min(groups, key=lambda community: community.total_weight)
       if not heavy.components:
           return ()

       weights = assignment.context.component_weights
       component = min(heavy.components, key=weights.__getitem__)
       return (
           VFDTransition.move(
               (component,),
               heavy.community_id,
               light.community_id,
               phase="repair",
               label="reduce load spread",
           ),
       )

   spread_constraint = PartitionPredicateConstraint(
       spread_is_allowed,
       violation=spread_violation,
       # Do not reject an incomplete construction solely for its current spread.
       partial_evaluator=lambda assignment: VFDConstraintEvaluation.ok(),
       repair_proposals=spread_repairs,
       name="bounded load spread",
   )

A custom constraint may also propose atomic
:class:`~asunder.base.algorithms.vfd_constraints.VFDTransition` objects for
repair. Each transition contains one or more
:class:`~asunder.base.algorithms.vfd_constraints.VFDComponentMove` objects.
This permits a swap or multi-component change that is feasible as a whole even
when applying its individual moves sequentially would temporarily violate the
constraint. ``VFDTransition.swap(...)`` constructs a two-way atomic swap;
construct ``VFDTransition`` from several ``VFDComponentMove`` objects for a
larger coordinated repair. Treat assignment and community views as read-only; ModularVFD
notifies stateful bound handlers only after an accepted transition commits.

After construction, ModularVFD searches normal move, swap, and ejection
neighborhoods plus any custom repair proposals. It prefers a lower violation
tuple and uses modularity to break ties; tabu tracking can admit unseen
equal-violation states. ``constraint_repair_steps`` sets the repair budget.
Leaving it as ``None`` derives a bounded budget from ``local_iters``. If no
feasible assignment is found across the configured restarts and candidate K
values, ModularVFD returns ``None`` rather than returning a partition that
violates a hard constraint.

Once feasibility has been reached, the same constraint-provided atomic
proposals also augment the ordinary objective-refinement neighborhood. Their
completed results must satisfy every hard constraint, just like built-in move,
swap, and ejection candidates.

Local protocol and performance
------------------------------

Implement :class:`~asunder.base.algorithms.vfd_constraints.VFDLocalConstraint`
when a predicate over whole communities would be needlessly expensive. Its
``prepare(context)`` method creates immutable problem-specific data, and the
prepared object's ``bind()`` method must return fresh runtime state. ModularVFD
binds each constraint again for every candidate K and restart, preventing
cached state from leaking between searches.

A bound local handler implements the following small lifecycle:

``block_is_feasible(components)``
   Reject a set of components only when it is intrinsically incompatible with
   co-movement.

``evaluate_partial(assignment)``
   Decide whether construction can still be completed feasibly.

``evaluate_component_transition(transition)``
   Evaluate the complete atomic change from cached state and the component
   transition alone. Do not emulate a swap as two sequential moves. This hook
   deliberately receives no assignment view, keeping the candidate hot path
   independent of partition materialization.

``component_transition_applied(transition)``
   Update incremental state after, and only after, an accepted commit.

``evaluate_final(assignment)``
   Perform a pure correctness check on a complete candidate.

``component_proposals()``
   Return no proposals or a focused iterable of atomic transitions derived
   from cached state. ModularVFD considers them during guided repair and
   feasible objective refinement without constructing an assignment view.

The bound handler can reject intrinsically incompatible component blocks,
check a proposed atomic transition, validate a partial or final assignment,
and update its cache after an accepted transition. A block should be declared
intrinsically infeasible only when its components can never move together.
Placement-specific failures belong in transition checks. This distinction
allows fingerprint blocks to remain intact when possible and to split only
when required.

For large inputs:

- cache counts, loads, conflict masks, or other sufficient statistics per
  community;
- update caches from ``VFDTransition`` instead of scanning the partition;
- reserve partition-wide predicates for requirements that genuinely need
  them;
- return a small, focused set of custom repair transitions rather than
  enumerating every possible multi-component move; and
- choose a finite ``constraint_repair_steps`` budget when predicate evaluation
  or repair proposal generation is expensive.

Final assignments are independently revalidated, even for stateful local
handlers. An extension must therefore implement final validation as a pure
correctness check rather than relying solely on its cache.

Integration boundaries
----------------------

These APIs make local, community-wide, and partition-wide hard constraints
enforceable inside ModularVFD. They do not yet make a custom rule a global
column-generation model constraint. Complete end-to-end enforcement remains
future work and ideally includes:

- preserving ``component_members`` and other predicate data through every
  contraction performed outside ModularVFD;
- applying the same rule to initial-column generators, pricing algorithms,
  master formulations, warm starts, and final solution validation;
- adding solver-backed formulations or specialized repair neighborhoods when
  generic guided repair is too weak; and
- returning explicit, named community-role metadata if a future model needs
  durable roles rather than permutation-invariant content predicates.

A constraint used only by ModularVFD guarantees that ModularVFD's returned
refinement satisfies it. Other columns can still enter the pool unless the
other pipeline stages enforce the same rule.
