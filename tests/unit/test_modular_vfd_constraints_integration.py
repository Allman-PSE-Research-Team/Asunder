import inspect

import networkx as nx
import numpy as np
import pytest

from asunder.base.algorithms.modular_VFD import modular_very_fortunate_descent
from asunder.base.algorithms.vfd_constraints import (
    CommunityPredicateConstraint,
    PartitionPredicateConstraint,
    QuantifiedCommunityConstraint,
    VFDAssignmentView,
    VFDBoundLocalConstraint,
    VFDConstraintEvaluation,
    VFDTransition,
)


def _run_modular(wz, *, constraints=(), component_members=None, **kwargs):
    adjacency = nx.to_numpy_array(nx.cycle_graph(wz.shape[0]), dtype=float)
    search_kwargs = {
        "candidate_Ks": (2,),
        "restarts": 2,
        "local_iters": 0,
        "clustering_Ks": (2,),
        "clustering_methods": (),
        "wz_is_C_node": True,
        "tabu_max_steps": 0,
        "shake_rounds": 0,
    }
    search_kwargs.update(kwargs)
    return modular_very_fortunate_descent(
        wz=np.asarray(wz, dtype=float),
        A=adjacency,
        a=adjacency.sum(axis=1),
        m=float(adjacency.sum()),
        constraints=constraints,
        component_members=component_members,
        **search_kwargs,
    )


def _labels_from_partition(partition):
    return np.argmax(partition, axis=1)


def test_modular_vfd_defaults_match_lb_where_applicable_without_enabling_balance():
    parameters = inspect.signature(modular_very_fortunate_descent).parameters

    assert parameters["K"].default == 2
    assert parameters["R"].default == 1
    assert parameters["seed"].default == 42
    assert parameters["must_link"].default == ()
    assert parameters["cannot_link"].default == ()
    assert parameters["use_K_constraint"].default is False


def test_constraint_free_path_does_not_materialize_assignment_views(monkeypatch):
    def fail_if_constructed(*_args, **_kwargs):
        raise AssertionError("constraint-free search constructed an assignment view")

    monkeypatch.setattr(VFDAssignmentView, "__init__", fail_if_constructed)

    result = _run_modular(np.eye(4))

    assert result is not None


def test_fast_local_runtime_uses_component_only_hot_path(monkeypatch):
    events = []
    runtimes = []

    class Runtime:
        scope = "local"
        name = "fast local"

        def __init__(self, context):
            self.context = context
            runtimes.append(self)

        def block_is_feasible(self, _components):
            return True

        def evaluate_partial(self, _assignment):
            return VFDConstraintEvaluation.ok()

        def evaluate_component_transition(self, transition):
            events.append(("evaluate", transition.phase))
            return VFDConstraintEvaluation.ok()

        def component_transition_applied(self, transition):
            events.append(("applied", transition.phase))

        def evaluate_final(self, _assignment):
            return VFDConstraintEvaluation.ok()

        def component_proposals(self):
            return ()

    class Prepared:
        def __init__(self, context):
            self.context = context

        def bind(self):
            return Runtime(self.context)

    class Constraint:
        def prepare(self, context):
            return Prepared(context)

    def fail_if_projected(*_args, **_kwargs):
        raise AssertionError("fast local candidate path projected an assignment")

    monkeypatch.setattr(VFDAssignmentView, "after", fail_if_projected)
    result = _run_modular(np.eye(4), constraints=(Constraint(),))

    assert result is not None
    assert runtimes and isinstance(runtimes[0], VFDBoundLocalConstraint)
    assert any(event == "evaluate" for event, _phase in events)
    assert any(event == "applied" for event, _phase in events)


def test_community_constraint_uses_external_member_provenance_and_splits_blocks():
    labels = np.array([0, 0, 1, 1])
    coassociation = np.equal.outer(labels, labels).astype(float)
    members = (("alpha",), ("beta",), ("gamma",), ("delta",))
    constraint = CommunityPredicateConstraint(
        lambda community: not {"alpha", "beta"}.issubset(community.nodes),
        name="alpha and beta cannot share a community",
    )

    result = _run_modular(
        coassociation,
        constraints=(constraint,),
        component_members=members,
    )

    assert result is not None
    partition, metadata = result
    assert partition[0, 1] == 0
    assert metadata["constraints"] == ("alpha and beta cannot share a community",)


def test_extendable_community_evaluation_is_allowed_during_construction():
    def exactly_two(community):
        return len(community.components) == 2

    def partial_size(community):
        size = len(community.components)
        if size == 2:
            return VFDConstraintEvaluation.ok()
        return VFDConstraintEvaluation.violated(
            abs(2 - size),
            extendable=size < 2,
        )

    result = _run_modular(
        np.eye(4),
        constraints=(
            CommunityPredicateConstraint(
                exactly_two,
                partial_predicate=partial_size,
            ),
        ),
    )

    assert result is not None
    partition, _ = result
    labels = _labels_from_partition(partition)
    assert sorted(np.bincount(labels)) == [2, 2]


def test_external_provenance_is_flattened_through_internal_must_link_contraction():
    seen_component_members = []

    def capture_context(_components, context):
        seen_component_members.append(context.component_members)
        return True

    constraint = CommunityPredicateConstraint(
        lambda _community: True,
        co_movement_predicate=capture_context,
    )
    members = (("alpha", "alpha-copy"), ("beta",), ("gamma",), ("delta",))

    result = _run_modular(
        np.eye(4),
        constraints=(constraint,),
        component_members=members,
        must_link=((0, 1),),
    )

    assert result is not None
    assert seen_component_members
    assert any(
        set(component) == {"alpha", "alpha-copy", "beta"}
        for component in seen_component_members[0]
    )


def test_quantified_constraint_enforces_exactly_one_generic_qualifying_community():
    labels = np.array([0, 0, 1, 1])
    coassociation = np.equal.outer(labels, labels).astype(float)
    members = (("a",), ("b",), ("c",), ("d",))
    allowed = {"a", "b"}
    constraint = QuantifiedCommunityConstraint(
        lambda community: bool(community.nodes)
        and all(node in allowed for node in community.nodes),
        exactly=1,
        name="exactly one qualifying community",
    )

    result = _run_modular(
        coassociation,
        constraints=(constraint,),
        component_members=members,
    )

    assert result is not None
    partition, _ = result
    output_labels = _labels_from_partition(partition)
    qualifying = 0
    for group in np.unique(output_labels):
        nodes = {members[index][0] for index in np.flatnonzero(output_labels == group)}
        qualifying += bool(nodes) and nodes <= allowed
    assert qualifying == 1


@pytest.mark.parametrize(
    "bounds",
    (
        {"minimum": 1},
        {"maximum": 1},
        {"minimum": 1, "maximum": 1},
    ),
)
def test_quantified_minimum_and_maximum_bounds_run_inside_modular_vfd(bounds):
    labels = np.array([0, 0, 1, 1])
    coassociation = np.equal.outer(labels, labels).astype(float)
    allowed = {0, 1}

    def qualifies(community):
        nodes = set(community.nodes)
        return bool(nodes) and nodes <= allowed

    result = _run_modular(
        coassociation,
        constraints=(QuantifiedCommunityConstraint(qualifies, **bounds),),
    )

    assert result is not None
    partition, _ = result
    output_labels = _labels_from_partition(partition)
    count = sum(
        bool(nodes := set(np.flatnonzero(output_labels == group)))
        and nodes <= allowed
        for group in np.unique(output_labels)
    )
    if "minimum" in bounds:
        assert count >= bounds["minimum"]
    if "maximum" in bounds:
        assert count <= bounds["maximum"]


def test_partition_constraint_can_supply_atomic_guided_repair():
    labels = np.array([0, 1, 0, 1])
    coassociation = np.equal.outer(labels, labels).astype(float)
    proposal_calls = []

    def together(assignment: VFDAssignmentView):
        return assignment.component_community(0) == assignment.component_community(1)

    def propose(assignment: VFDAssignmentView):
        proposal_calls.append(assignment.component_to_community)
        left_group = assignment.component_community(0)
        right_group = assignment.component_community(1)
        if left_group is None or right_group is None or left_group == right_group:
            return ()
        donor = next(
            component
            for component in assignment.community(left_group).components
            if component != 0
        )
        return (
            VFDTransition.swap(
                (1,),
                right_group,
                (donor,),
                left_group,
                phase="repair",
                label="put required components together",
            ),
        )

    constraint = PartitionPredicateConstraint(
        together,
        violation=lambda assignment: 0.0 if together(assignment) else 1.0,
        repair_proposals=propose,
        name="components 0 and 1 together",
    )

    result = _run_modular(
        coassociation,
        constraints=(constraint,),
        constraint_repair_steps=10,
    )

    assert result is not None
    partition, metadata = result
    assert partition[0, 1] == 1
    assert proposal_calls
    assert metadata["constraint_repair_steps"] >= 1


def test_partition_constraint_can_be_repaired_by_the_normal_neighborhood():
    labels = np.array([0, 1, 0, 1])
    coassociation = np.equal.outer(labels, labels).astype(float)

    def together(assignment):
        return assignment.component_community(0) == assignment.component_community(1)

    result = _run_modular(
        coassociation,
        constraints=(
            PartitionPredicateConstraint(
                together,
                violation=lambda assignment: 0.0 if together(assignment) else 1.0,
            ),
        ),
        constraint_repair_steps=10,
    )

    assert result is not None
    partition, metadata = result
    assert partition[0, 1] == 1
    assert metadata["constraint_repair_steps"] >= 1


def test_community_constraint_can_be_repaired_over_multiple_transitions():
    initial_labels = np.repeat(np.arange(3), 2)
    coassociation = np.equal.outer(initial_labels, initial_labels).astype(float)
    categories = {
        0: "a",
        1: "b",
        2: "a",
        3: "b",
        4: "a",
        5: "b",
    }

    def homogeneous(community):
        return len({categories[node] for node in community.nodes}) <= 1

    repair_calls = []

    def propose_next_repair(assignment):
        repair_calls.append(assignment.component_to_community)
        pairs = ((1, 3), (2, 0), (5, 1))
        for component, same_category_component in pairs:
            source = assignment.component_community(component)
            target = assignment.component_community(same_category_component)
            paired_component = component - 1 if component in {1, 5} else 3
            if (
                source is not None
                and target is not None
                and source != target
                and assignment.component_community(paired_component) != source
            ):
                continue
            if source is not None and target is not None and source != target:
                return (
                    VFDTransition.move(
                        (component,),
                        source,
                        target,
                        phase="repair",
                    ),
                )
        return ()

    constraint = CommunityPredicateConstraint(
        homogeneous,
        partial_predicate=lambda _community: True,
        violation=lambda community: 0.0 if homogeneous(community) else 1.0,
        repair_proposals=propose_next_repair,
        name="homogeneous categories",
    )

    result = _run_modular(
        coassociation,
        constraints=(constraint,),
        candidate_Ks=(3,),
        clustering_Ks=(3,),
        constraint_repair_steps=12,
    )

    assert result is not None
    partition, metadata = result
    labels = _labels_from_partition(partition)
    for group in np.unique(labels):
        members = np.flatnonzero(labels == group)
        assert len({categories[int(node)] for node in members}) == 1
    assert metadata["constraint_repair_steps"] >= 2
    assert len(repair_calls) >= 2


def test_saved_best_is_revalidated_with_a_fresh_runtime():
    final_notification_counts = []
    prepared_ks = []
    runtimes = []

    class Runtime:
        scope = "partition"
        name = "runtime isolation"

        def __init__(self, context):
            self.context = context
            self.notifications = 0
            runtimes.append(self)

        def block_is_feasible(self, _components):
            return True

        def evaluate_partial(self, _assignment):
            return VFDConstraintEvaluation.ok()

        def evaluate_transition(self, _before, _transition, _after):
            return VFDConstraintEvaluation.ok()

        def evaluate_final(self, _assignment):
            final_notification_counts.append(self.notifications)
            return VFDConstraintEvaluation.ok()

        def transition_applied(self, _transition, _assignment):
            self.notifications += 1

        def repair_proposals(self, _assignment):
            return ()

    class Prepared:
        def __init__(self, context):
            self.context = context

        def bind(self):
            return Runtime(self.context)

    class Constraint:
        def prepare(self, context):
            prepared_ks.append(context.K)
            return Prepared(context)

    result = _run_modular(
        np.eye(4),
        constraints=(Constraint(),),
        candidate_Ks=(2, 3),
        clustering_Ks=(2, 3),
    )

    assert result is not None
    assert set(prepared_ks) == {2, 3}
    assert len({id(runtime) for runtime in runtimes}) == len(runtimes)
    assert any(count > 0 for count in final_notification_counts)
    assert 0 in final_notification_counts


def test_nonextendable_partial_partition_prunes_construction():
    partial_calls = []

    def impossible_partial(assignment):
        partial_calls.append(assignment.component_to_community)
        return VFDConstraintEvaluation.violated(1.0, extendable=False)

    result = _run_modular(
        np.eye(4),
        constraints=(
            PartitionPredicateConstraint(
                lambda _assignment: False,
                partial_evaluator=impossible_partial,
            ),
        ),
    )

    assert result is None
    assert partial_calls
    assert all(set(labels) == {-1} for labels in partial_calls)


def test_constraint_proposals_augment_feasible_objective_search():
    proposal_calls = []

    def proposals(assignment):
        proposal_calls.append(assignment.component_to_community)
        left = assignment.component_community(0)
        right = assignment.component_community(1)
        if left is None or right is None or left == right:
            return ()
        return (VFDTransition.swap((0,), left, (1,), right),)

    result = _run_modular(
        np.eye(4),
        constraints=(
            PartitionPredicateConstraint(
                lambda _assignment: True,
                repair_proposals=proposals,
            ),
        ),
        local_iters=1,
        tabu_max_steps=1,
        shake_rounds=1,
    )

    assert result is not None
    assert proposal_calls


def test_impossible_partition_constraint_returns_none():
    result = _run_modular(
        np.eye(4),
        constraints=(PartitionPredicateConstraint(lambda _assignment: False),),
        constraint_repair_steps=0,
    )

    assert result is None


def test_empty_input_still_validates_partition_constraints():
    empty = np.zeros((0, 0), dtype=float)

    assert _run_modular(
        empty,
        constraints=(PartitionPredicateConstraint(lambda _assignment: False),),
    ) is None

    valid = _run_modular(
        empty,
        constraints=(PartitionPredicateConstraint(lambda _assignment: True),),
    )
    assert valid is not None
    partition, metadata = valid
    assert partition.shape == (0, 0)
    assert metadata["K_used"] == 0
