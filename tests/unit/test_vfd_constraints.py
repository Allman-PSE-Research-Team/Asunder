import numpy as np
import pytest

from asunder.base.algorithms.vfd_constraints import (
    CommunityPredicateConstraint,
    PartitionPredicateConstraint,
    QuantifiedCommunityConstraint,
    VFDAssignmentView,
    VFDConstraint,
    VFDConstraintContext,
    VFDConstraintEvaluation,
    VFDTransition,
)


def _context(component_members=(("a", "b"), ("c",)), K=2):
    component_count = len(component_members)
    node_count = sum(len(members) for members in component_members)
    node_to_component = tuple(
        component
        for component, members in enumerate(component_members)
        for _ in members
    )
    return VFDConstraintContext(
        input_adjacency=np.eye(node_count),
        component_adjacency=np.eye(component_count),
        node_to_component=node_to_component,
        component_members=component_members,
        component_weights=tuple(len(members) for members in component_members),
        K=K,
        r_min=1,
        r_max=node_count,
    )


def test_context_preserves_hashable_provenance_and_read_only_matrices():
    context = _context((("alpha", 7), (("tuple", "id"),)))

    assert context.component_members == (("alpha", 7), (("tuple", "id"),))
    assert context.component_count == 2
    assert context.original_node_count == 3
    with pytest.raises(ValueError):
        context.component_adjacency[0, 0] = 2


def test_context_rejects_duplicate_or_unhashable_provenance():
    with pytest.raises(ValueError, match="distinct"):
        _context((("same",), ("same",)))
    with pytest.raises(TypeError, match="hashable"):
        _context((([],), ("ok",)))


def test_assignment_and_community_views_are_lazy_projections():
    context = _context()
    assignment = VFDAssignmentView(context, (0, 1))

    left = assignment.community(0)
    assert left.components == (0,)
    assert left.component_members == (("a", "b"),)
    assert left.nodes == ("a", "b")
    assert left.total_weight == 2
    assert assignment.nonempty_community_ids == (0, 1)
    assert np.array_equal(
        assignment.partition_matrix(),
        np.array([[1, 1, 0], [1, 1, 0], [0, 0, 1]]),
    )
    assert assignment.projected((1,)).community_ids == (1,)


def test_atomic_swap_validates_sources_against_the_pre_transition_assignment():
    context = _context((("a",), ("b",), ("c",)))
    assignment = VFDAssignmentView(context, (0, 1, 1))
    swap = VFDTransition.swap((0,), 0, (1,), 1, phase="repair")

    result = assignment.after(swap)

    assert swap.affected_communities == (0, 1)
    assert result.component_to_community == (1, 0, 1)
    with pytest.raises(ValueError, match="source"):
        assignment.after(VFDTransition.move((0,), 1, 0))


def test_evaluation_normalises_scores_and_combines_lexicographically():
    first = VFDConstraintEvaluation((1, 0), extendable=True, reason="first")
    second = VFDConstraintEvaluation(2, extendable=False, reason="second")

    combined = VFDConstraintEvaluation.combine((first, second))

    assert combined.violation == (1.0, 0.0, 2.0)
    assert not combined.satisfied
    assert not combined.extendable
    assert combined.reason == "first; second"
    with pytest.raises(ValueError, match="nonnegative"):
        VFDConstraintEvaluation(-1)


def test_community_predicate_checks_only_affected_communities():
    context = _context()
    visited = []

    def at_most_two_nodes(community):
        visited.append(community.community_id)
        return len(community.nodes) <= 2

    constraint = CommunityPredicateConstraint(
        at_most_two_nodes,
        violation=lambda community: max(len(community.nodes) - 2, 0),
        name="capacity",
    )
    runtime = constraint.prepare(context).bind()
    before = VFDAssignmentView(context, (0, 1))
    transition = VFDTransition.move((1,), 1, 0)
    after = before.after(transition)

    evaluation = runtime.evaluate_transition(before, transition, after)

    assert visited == [0]
    assert evaluation.violation == (1.0,)
    assert not evaluation.extendable


def test_community_predicate_supports_partial_and_intrinsic_block_checks():
    context = _context()
    constraint = CommunityPredicateConstraint(
        lambda community: len(community.nodes) <= 1,
        partial_predicate=lambda community: len(community.nodes) <= 3,
        co_movement_predicate=lambda components, _context: len(components) == 1,
    )
    runtime = constraint.bind(context)
    before = VFDAssignmentView(context, (0, -1))
    transition = VFDTransition.move((1,), None, 0, phase="construction")

    assert runtime.evaluate_transition(before, transition, before.after(transition)).satisfied
    assert not runtime.block_is_feasible((0, 1))


def test_explicit_evaluation_dimensions_are_preserved_when_satisfied():
    context = _context()
    assignment = VFDAssignmentView(context, (0, 1))
    community = CommunityPredicateConstraint(
        lambda view: (
            VFDConstraintEvaluation((0.0, 0.0))
            if view.community_id == 0
            else VFDConstraintEvaluation((1.0, 0.0))
        )
    ).bind(context)
    partition = PartitionPredicateConstraint(
        lambda _view: VFDConstraintEvaluation((0.0, 0.0))
    ).bind(context)

    assert community.evaluate_final(assignment).violation == (1.0, 0.0)
    assert partition.evaluate_final(assignment).violation == (0.0, 0.0)


def test_quantified_community_constraint_supports_exact_and_bounded_counts():
    context = _context((("a",), ("b",), ("c",)), K=3)
    assignment = VFDAssignmentView(context, (0, 1, 2))

    def contains_marked_node(community):
        return bool({"a", "b"} & set(community.nodes))

    exact = QuantifiedCommunityConstraint(contains_marked_node, exactly=1).bind(context)
    bounded = QuantifiedCommunityConstraint(
        contains_marked_node, minimum=1, maximum=2
    ).bind(context)

    assert exact.evaluate_final(assignment).violation == (0.0, 1.0)
    assert exact.evaluate_partial(assignment).extendable
    assert bounded.evaluate_final(assignment).satisfied
    repaired = assignment.after(VFDTransition.move((1,), 1, 0, phase="repair"))
    assert exact.evaluate_final(repaired).satisfied


def test_partition_predicate_supports_partial_evaluation_and_repair_proposals():
    context = _context()
    assignment = VFDAssignmentView(context, (0, 0))
    proposed = VFDTransition.move((1,), 0, 1, phase="repair")
    constraint = PartitionPredicateConstraint(
        lambda view: len(view.nonempty_community_ids) == 2,
        violation=lambda view: abs(2 - len(view.nonempty_community_ids)),
        partial_evaluator=lambda view: len(view.unassigned_components) <= 1,
        repair_proposals=lambda _view: (proposed,),
        name="two communities",
    )
    prepared = constraint.prepare(context)
    first_runtime = prepared.bind()
    second_runtime = prepared.bind()

    assert isinstance(constraint, VFDConstraint)
    assert first_runtime is not second_runtime
    assert first_runtime.evaluate_final(assignment).violation == (1.0,)
    assert first_runtime.evaluate_partial(assignment).satisfied
    assert tuple(first_runtime.repair_proposals(assignment)) == (proposed,)
    assert first_runtime.evaluate_final(assignment.after(proposed)).satisfied


def test_partition_partial_evaluator_can_prove_assignment_unextendable():
    context = _context()
    constraint = PartitionPredicateConstraint(
        lambda _view: False,
        partial_evaluator=lambda _view: VFDConstraintEvaluation.violated(
            3, extendable=False
        ),
    ).bind(context)

    evaluation = constraint.evaluate_partial(VFDAssignmentView(context, (-1, -1)))

    assert evaluation.violation == (3.0,)
    assert not evaluation.extendable


def test_partition_partial_evaluator_is_construction_only():
    context = _context()
    partial_calls = []

    def reject_partial(_assignment):
        partial_calls.append(True)
        return VFDConstraintEvaluation.violated(1, extendable=False)

    runtime = PartitionPredicateConstraint(
        lambda view: len(view.nonempty_community_ids) == 2,
        partial_evaluator=reject_partial,
    ).bind(context)
    before = VFDAssignmentView(context, (0, 0))
    repair = VFDTransition.move((1,), 0, 1, phase="repair")

    evaluation = runtime.evaluate_transition(before, repair, before.after(repair))

    assert evaluation.satisfied
    assert partial_calls == []
