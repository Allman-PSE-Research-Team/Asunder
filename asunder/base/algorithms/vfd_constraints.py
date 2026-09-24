"""Public constraint interfaces for Modular Very Fortunate Descent.

The classes in this module deliberately do not depend on the VFD search
implementation.  Constraints are prepared against immutable contraction data
once, then bound to a fresh runtime for every restart.  This keeps the common
local-constraint path inexpensive while still allowing community- and
partition-wide predicates to inspect lazy assignment views.
"""

from __future__ import annotations

from collections.abc import Hashable
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Protocol, Sequence, runtime_checkable

import numpy as np

Violation = float | Sequence[float]


def _readonly_matrix(value: Any, name: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"{name} must be a square matrix.")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must contain only finite values.")
    readonly = matrix.view()
    readonly.flags.writeable = False
    return readonly


def _normalise_violation(value: Violation | None) -> tuple[float, ...]:
    if value is None:
        return ()
    if isinstance(value, bool):
        raise TypeError("A constraint violation must be numeric, not boolean.")
    if np.isscalar(value):
        values = (float(value),)
    else:
        values = tuple(float(item) for item in value)
    if any(not np.isfinite(item) or item < 0 for item in values):
        raise ValueError("Constraint violations must be finite and nonnegative.")
    return values


@dataclass(frozen=True)
class VFDConstraintContext:
    """Immutable graph and contraction data supplied to VFD constraints.

    ``component_members[c]`` contains the original node identifiers represented
    by search component ``c``.  Consequently it remains useful when the matrix
    passed to ModularVFD was already contracted by a caller.
    """

    input_adjacency: Any
    component_adjacency: Any
    node_to_component: Sequence[int]
    component_members: Sequence[Sequence[Hashable]]
    component_weights: Sequence[float]
    K: int
    r_min: float | None = None
    r_max: float | None = None

    def __post_init__(self) -> None:
        input_adjacency = _readonly_matrix(self.input_adjacency, "input_adjacency")
        component_adjacency = _readonly_matrix(
            self.component_adjacency, "component_adjacency"
        )
        component_members = tuple(tuple(group) for group in self.component_members)
        component_weights = tuple(float(weight) for weight in self.component_weights)
        node_to_component = tuple(int(component) for component in self.node_to_component)
        K = int(self.K)

        component_count = component_adjacency.shape[0]
        if len(component_members) != component_count:
            raise ValueError("component_members must contain one entry per component.")
        if len(component_weights) != component_count:
            raise ValueError("component_weights must contain one value per component.")
        if len(node_to_component) != input_adjacency.shape[0]:
            raise ValueError(
                "node_to_component must contain one entry per input-adjacency row."
            )
        if any(not members for members in component_members):
            raise ValueError("Every component must contain at least one original node.")
        flattened = [node for members in component_members for node in members]
        try:
            distinct_member_count = len(set(flattened))
        except TypeError as exc:
            raise TypeError("component_members must contain hashable node identifiers.") from exc
        if len(flattened) != distinct_member_count:
            raise ValueError("component_members must contain distinct node identifiers.")
        if any(not np.isfinite(weight) or weight <= 0 for weight in component_weights):
            raise ValueError("component_weights must contain finite positive values.")
        if any(component < 0 or component >= component_count for component in node_to_component):
            raise ValueError("node_to_component contains an invalid component index.")
        if K < 0 or (K == 0 and component_count != 0):
            raise ValueError(
                "K must be positive unless the constraint context is empty."
            )
        r_min = None if self.r_min is None else float(self.r_min)
        r_max = None if self.r_max is None else float(self.r_max)
        if r_min is not None and (not np.isfinite(r_min) or r_min < 0):
            raise ValueError("r_min must be finite and nonnegative when supplied.")
        if r_max is not None and (not np.isfinite(r_max) or r_max < 0):
            raise ValueError("r_max must be finite and nonnegative when supplied.")
        if r_min is not None and r_max is not None and r_min > r_max:
            raise ValueError("r_min cannot exceed r_max.")

        object.__setattr__(self, "input_adjacency", input_adjacency)
        object.__setattr__(self, "component_adjacency", component_adjacency)
        object.__setattr__(self, "node_to_component", node_to_component)
        object.__setattr__(self, "component_members", component_members)
        object.__setattr__(self, "component_weights", component_weights)
        object.__setattr__(self, "K", K)
        object.__setattr__(self, "r_min", r_min)
        object.__setattr__(self, "r_max", r_max)

    @property
    def component_count(self) -> int:
        """Number of components assigned by the VFD search."""

        return len(self.component_members)

    @property
    def original_node_count(self) -> int:
        """Number of provenance identifiers in ``component_members``."""

        return sum(len(members) for members in self.component_members)

    @property
    def input_node_count(self) -> int:
        """Number of rows in the adjacency supplied to ModularVFD."""

        return len(self.node_to_component)

    @property
    def all_components(self) -> tuple[int, ...]:
        """All component identifiers in deterministic order."""

        return tuple(range(self.component_count))


@dataclass(frozen=True)
class VFDComponentMove:
    """Move one or more components between communities atomically."""

    components: Sequence[int]
    source: int | None
    target: int | None

    def __post_init__(self) -> None:
        components = tuple(int(component) for component in self.components)
        source = None if self.source is None else int(self.source)
        target = None if self.target is None else int(self.target)
        if not components:
            raise ValueError("A component move must contain at least one component.")
        if len(components) != len(set(components)) or any(component < 0 for component in components):
            raise ValueError("Move components must be distinct nonnegative integers.")
        if source is not None and source < 0:
            raise ValueError("Move source must be nonnegative or None.")
        if target is not None and target < 0:
            raise ValueError("Move target must be nonnegative or None.")
        if source == target:
            raise ValueError("Move source and target must differ.")
        object.__setattr__(self, "components", components)
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "target", target)


@dataclass(frozen=True)
class VFDTransition:
    """A complete atomic assignment transition.

    ``source=None`` denotes assignment of an unassigned component during
    construction.  The representation permits ``target=None`` for consumers
    that model unassignment, but ModularVFD's guided repair accepts only
    transitions whose resulting partition is complete.
    """

    moves: Sequence[VFDComponentMove]
    phase: str = "feasible_search"
    label: str | None = None

    def __post_init__(self) -> None:
        moves = tuple(self.moves)
        if not moves:
            raise ValueError("A transition must contain at least one move.")
        if any(not isinstance(move, VFDComponentMove) for move in moves):
            raise TypeError("Transition moves must be VFDComponentMove instances.")
        moved_components = [component for move in moves for component in move.components]
        if len(moved_components) != len(set(moved_components)):
            raise ValueError("A transition cannot move the same component more than once.")
        phase = str(self.phase)
        if not phase:
            raise ValueError("Transition phase cannot be empty.")
        object.__setattr__(self, "moves", moves)
        object.__setattr__(self, "phase", phase)

    @property
    def components(self) -> tuple[int, ...]:
        """All moved components in transition order."""

        return tuple(component for move in self.moves for component in move.components)

    @property
    def affected_communities(self) -> tuple[int, ...]:
        """Source and target communities affected by the transition."""

        return tuple(
            sorted(
                {
                    community
                    for move in self.moves
                    for community in (move.source, move.target)
                    if community is not None
                }
            )
        )

    @classmethod
    def move(
        cls,
        components: Sequence[int],
        source: int | None,
        target: int | None,
        *,
        phase: str = "feasible_search",
        label: str | None = None,
    ) -> VFDTransition:
        """Construct a one-way component transition."""

        return cls((VFDComponentMove(components, source, target),), phase=phase, label=label)

    @classmethod
    def swap(
        cls,
        left_components: Sequence[int],
        left_community: int,
        right_components: Sequence[int],
        right_community: int,
        *,
        phase: str = "feasible_search",
        label: str | None = None,
    ) -> VFDTransition:
        """Construct an atomic two-way swap."""

        return cls(
            (
                VFDComponentMove(left_components, left_community, right_community),
                VFDComponentMove(right_components, right_community, left_community),
            ),
            phase=phase,
            label=label,
        )


@dataclass(frozen=True)
class VFDConstraintEvaluation:
    """Feasibility and violation information returned by a constraint.

    A constraint is satisfied when every element of ``violation`` is zero.
    During partial construction or repair, a nonzero violation may remain
    admissible when ``extendable`` is true.
    """

    violation: Violation = (0.0,)
    extendable: bool = True
    reason: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "violation", _normalise_violation(self.violation))
        object.__setattr__(self, "extendable", bool(self.extendable))

    @property
    def satisfied(self) -> bool:
        """Whether the violation tuple is identically zero."""

        return all(value == 0 for value in self.violation)

    @property
    def feasible(self) -> bool:
        """Alias for ``satisfied`` for hard-constraint callers."""

        return self.satisfied

    @classmethod
    def ok(cls) -> VFDConstraintEvaluation:
        """Create a satisfied evaluation."""

        return cls((0.0,))

    @classmethod
    def violated(
        cls,
        violation: Violation = 1.0,
        *,
        extendable: bool = True,
        reason: str | None = None,
    ) -> VFDConstraintEvaluation:
        """Create an unsatisfied evaluation."""

        result = cls(violation, extendable=extendable, reason=reason)
        if result.satisfied:
            raise ValueError("A violated evaluation must have a positive violation.")
        return result

    @classmethod
    def combine(
        cls, evaluations: Iterable[VFDConstraintEvaluation]
    ) -> VFDConstraintEvaluation:
        """Concatenate evaluations in declaration order for lexicographic ranking."""

        evaluations = tuple(evaluations)
        return cls(
            tuple(value for result in evaluations for value in result.violation),
            extendable=all(result.extendable for result in evaluations),
            reason="; ".join(result.reason for result in evaluations if result.reason) or None,
        )


class VFDAssignmentView:
    """Read-only, lazily projected view of a component assignment."""

    __slots__ = ("_community_ids", "_context", "_labels", "_views")

    def __init__(
        self,
        context: VFDConstraintContext,
        component_to_community: Sequence[int],
        community_ids: Sequence[int] | None = None,
    ) -> None:
        labels = tuple(int(label) for label in component_to_community)
        if len(labels) != context.component_count:
            raise ValueError("component_to_community must contain one label per component.")
        if any(label < -1 or label >= context.K for label in labels):
            raise ValueError("Assignment labels must be -1 or valid community indices.")
        if community_ids is None:
            selected = tuple(range(context.K))
        else:
            selected = tuple(dict.fromkeys(int(group) for group in community_ids))
            if any(group < 0 or group >= context.K for group in selected):
                raise ValueError("community_ids contains an invalid community index.")
        self._context = context
        self._labels = labels
        self._community_ids = selected
        self._views: dict[int, VFDCommunityView] = {}

    @property
    def context(self) -> VFDConstraintContext:
        """Constraint context associated with this assignment."""

        return self._context

    @property
    def component_to_community(self) -> tuple[int, ...]:
        """Immutable component labels; ``-1`` means unassigned."""

        return self._labels

    @property
    def community_ids(self) -> tuple[int, ...]:
        """Community IDs projected by this view."""

        return self._community_ids

    @property
    def unassigned_components(self) -> tuple[int, ...]:
        """Components that have not yet been assigned."""

        return tuple(component for component, group in enumerate(self._labels) if group == -1)

    @property
    def is_complete(self) -> bool:
        """Whether every component has been assigned."""

        return -1 not in self._labels

    @property
    def nonempty_community_ids(self) -> tuple[int, ...]:
        """Nonempty communities among the projected IDs."""

        occupied = set(self._labels)
        return tuple(group for group in self._community_ids if group in occupied)

    def component_community(self, component: int) -> int | None:
        """Return a component's community, or ``None`` when unassigned."""

        component = int(component)
        if component < 0 or component >= self._context.component_count:
            raise IndexError("component index out of range")
        group = self._labels[component]
        return None if group == -1 else group

    def community(self, community_id: int) -> VFDCommunityView:
        """Return a lazy view of one community."""

        community_id = int(community_id)
        if community_id < 0 or community_id >= self._context.K:
            raise IndexError("community index out of range")
        if community_id not in self._views:
            self._views[community_id] = VFDCommunityView(self, community_id)
        return self._views[community_id]

    def iter_communities(self, *, include_empty: bool = True) -> Iterable[VFDCommunityView]:
        """Iterate over projected communities in deterministic order."""

        for community_id in self._community_ids:
            community = self.community(community_id)
            if include_empty or not community.is_empty:
                yield community

    def projected(self, community_ids: Sequence[int]) -> VFDAssignmentView:
        """Create a view restricted to selected community projections."""

        return VFDAssignmentView(self._context, self._labels, community_ids)

    def affected(self, transition: VFDTransition) -> VFDAssignmentView:
        """Create a view restricted to a transition's affected communities."""

        return self.projected(transition.affected_communities)

    def after(self, transition: VFDTransition) -> VFDAssignmentView:
        """Return the assignment produced by applying an atomic transition."""

        labels = list(self._labels)
        for move in transition.moves:
            if move.source is not None and move.source >= self._context.K:
                raise ValueError("Transition source is outside the configured community range.")
            if move.target is not None and move.target >= self._context.K:
                raise ValueError("Transition target is outside the configured community range.")
            for component in move.components:
                if component >= self._context.component_count:
                    raise ValueError("Transition contains an unknown component.")
                expected = -1 if move.source is None else move.source
                if labels[component] != expected:
                    raise ValueError("Transition source does not match the current assignment.")
        for move in transition.moves:
            target = -1 if move.target is None else move.target
            for component in move.components:
                labels[component] = target
        return VFDAssignmentView(self._context, labels, self._community_ids)

    def partition_matrix(self, *, component_level: bool = False) -> np.ndarray:
        """Materialize a component- or input-node-level co-membership matrix."""

        if component_level:
            labels = np.asarray(self._labels, dtype=int)
        else:
            component_labels = np.asarray(self._labels, dtype=int)
            labels = component_labels[np.asarray(self._context.node_to_component, dtype=int)]
        assigned = labels >= 0
        matrix = np.equal.outer(labels, labels) & np.logical_and.outer(assigned, assigned)
        matrix = matrix.astype(int)
        matrix.flags.writeable = False
        return matrix


class VFDCommunityView:
    """Lazy read-only projection of one community."""

    __slots__ = ("_assignment", "_community_id", "_components", "_nodes")

    def __init__(self, assignment: VFDAssignmentView, community_id: int) -> None:
        self._assignment = assignment
        self._community_id = int(community_id)
        self._components: tuple[int, ...] | None = None
        self._nodes: tuple[Hashable, ...] | None = None

    @property
    def context(self) -> VFDConstraintContext:
        """Constraint context associated with this community."""

        return self._assignment.context

    @property
    def assignment(self) -> VFDAssignmentView:
        """Owning assignment view."""

        return self._assignment

    @property
    def community_id(self) -> int:
        """Transient search community index."""

        return self._community_id

    @property
    def components(self) -> tuple[int, ...]:
        """Component IDs assigned to this community."""

        if self._components is None:
            self._components = tuple(
                component
                for component, group in enumerate(self._assignment.component_to_community)
                if group == self._community_id
            )
        return self._components

    @property
    def component_members(self) -> tuple[tuple[Hashable, ...], ...]:
        """Original members of each assigned component."""

        return tuple(self.context.component_members[component] for component in self.components)

    @property
    def nodes(self) -> tuple[Hashable, ...]:
        """Original node IDs represented by this community."""

        if self._nodes is None:
            self._nodes = tuple(node for members in self.component_members for node in members)
        return self._nodes

    @property
    def total_weight(self) -> float:
        """Total component weight of this community."""

        return float(sum(self.context.component_weights[c] for c in self.components))

    @property
    def is_empty(self) -> bool:
        """Whether this community contains no assigned components."""

        return not self.components

    @property
    def component_adjacency(self) -> np.ndarray:
        """Materialize this community's induced component adjacency."""

        indices = np.asarray(self.components, dtype=int)
        matrix = np.array(
            self.context.component_adjacency[np.ix_(indices, indices)], copy=True
        )
        matrix.flags.writeable = False
        return matrix


EvaluationLike = bool | VFDConstraintEvaluation
CommunityPredicate = Callable[[VFDCommunityView], EvaluationLike]
PartitionPredicate = Callable[[VFDAssignmentView], EvaluationLike]
CommunityViolation = Callable[[VFDCommunityView], Violation]
PartitionViolation = Callable[[VFDAssignmentView], Violation]
PartialEvaluator = Callable[[VFDAssignmentView], EvaluationLike]
RepairProposalCallback = Callable[[VFDAssignmentView], Iterable[VFDTransition]]
CoMovementPredicate = Callable[[tuple[int, ...], VFDConstraintContext], bool]


@runtime_checkable
class VFDBoundConstraint(Protocol):
    """Common runtime surface consumed by ModularVFD."""

    context: VFDConstraintContext
    name: str
    scope: str

    def block_is_feasible(self, components: Sequence[int]) -> bool: ...

    def evaluate_partial(self, assignment: VFDAssignmentView) -> VFDConstraintEvaluation: ...

    def evaluate_transition(
        self,
        before: VFDAssignmentView,
        transition: VFDTransition,
        after: VFDAssignmentView,
    ) -> VFDConstraintEvaluation: ...

    def evaluate_final(self, assignment: VFDAssignmentView) -> VFDConstraintEvaluation: ...

    def transition_applied(
        self, transition: VFDTransition, assignment: VFDAssignmentView
    ) -> None: ...

    def repair_proposals(self, assignment: VFDAssignmentView) -> Iterable[VFDTransition]: ...


@runtime_checkable
class VFDPreparedConstraint(Protocol):
    """Constraint prepared once for a particular contracted problem."""

    context: VFDConstraintContext

    def bind(self) -> VFDBoundConstraint: ...


@runtime_checkable
class VFDConstraint(Protocol):
    """Constraint accepted by ModularVFD's public ``constraints`` argument."""

    def prepare(
        self, context: VFDConstraintContext
    ) -> VFDPreparedConstraint | VFDPreparedLocalConstraint: ...


@runtime_checkable
class VFDBoundLocalConstraint(Protocol):
    """Fresh per-restart runtime for an incrementally maintained constraint.

    The component-only hooks let ModularVFD avoid constructing assignment
    views on its hot candidate path.  ``evaluate_final`` remains a pure check
    over an explicit assignment for independent result validation.
    """

    context: VFDConstraintContext
    name: str
    scope: str

    def block_is_feasible(self, components: Sequence[int]) -> bool: ...

    def evaluate_partial(
        self, assignment: VFDAssignmentView
    ) -> VFDConstraintEvaluation: ...

    def evaluate_component_transition(
        self, transition: VFDTransition
    ) -> VFDConstraintEvaluation: ...

    def component_transition_applied(self, transition: VFDTransition) -> None: ...

    def evaluate_final(
        self, assignment: VFDAssignmentView
    ) -> VFDConstraintEvaluation: ...

    def component_proposals(self) -> Iterable[VFDTransition]: ...


@runtime_checkable
class VFDPreparedLocalConstraint(Protocol):
    """Prepared form of a local constraint."""

    context: VFDConstraintContext

    def bind(self) -> VFDBoundLocalConstraint: ...


@runtime_checkable
class VFDLocalConstraint(VFDConstraint, Protocol):
    """Extension protocol for fast component-transition constraints."""

    def prepare(self, context: VFDConstraintContext) -> VFDPreparedLocalConstraint: ...


def _coerce_evaluation(
    value: EvaluationLike,
    *,
    false_extendable: bool,
    reason: str | None = None,
) -> VFDConstraintEvaluation:
    if isinstance(value, VFDConstraintEvaluation):
        return value
    if not isinstance(value, (bool, np.bool_)):
        raise TypeError("Constraint predicates must return bool or VFDConstraintEvaluation.")
    if bool(value):
        return VFDConstraintEvaluation.ok()
    return VFDConstraintEvaluation.violated(
        1.0, extendable=false_extendable, reason=reason
    )


def _callback_proposals(
    callback: RepairProposalCallback | None, assignment: VFDAssignmentView
) -> tuple[VFDTransition, ...]:
    if callback is None:
        return ()
    proposals = tuple(callback(assignment))
    if any(not isinstance(proposal, VFDTransition) for proposal in proposals):
        raise TypeError("Repair proposal callbacks must yield VFDTransition instances.")
    return proposals


class _BoundConstraintBase:
    scope = "partition"

    def __init__(self, context: VFDConstraintContext, name: str) -> None:
        self.context = context
        self.name = name

    def block_is_feasible(self, components: Sequence[int]) -> bool:
        del components
        return True

    def transition_applied(
        self, transition: VFDTransition, assignment: VFDAssignmentView
    ) -> None:
        del transition, assignment


@dataclass(frozen=True)
class CommunityPredicateConstraint:
    """Hard predicate evaluated only for affected communities."""

    predicate: CommunityPredicate
    partial_predicate: CommunityPredicate | None = None
    violation: CommunityViolation | None = None
    repair_proposals_callback: RepairProposalCallback | None = field(
        default=None, repr=False
    )
    co_movement_predicate: CoMovementPredicate | None = field(default=None, repr=False)
    include_empty: bool = False
    name: str | None = None

    def __init__(
        self,
        predicate: CommunityPredicate,
        *,
        partial_predicate: CommunityPredicate | None = None,
        violation: CommunityViolation | None = None,
        repair_proposals: RepairProposalCallback | None = None,
        co_movement_predicate: CoMovementPredicate | None = None,
        include_empty: bool = False,
        name: str | None = None,
    ) -> None:
        if not callable(predicate):
            raise TypeError("predicate must be callable.")
        for callback_name, callback in (
            ("partial_predicate", partial_predicate),
            ("violation", violation),
            ("repair_proposals", repair_proposals),
            ("co_movement_predicate", co_movement_predicate),
        ):
            if callback is not None and not callable(callback):
                raise TypeError(f"{callback_name} must be callable when supplied.")
        object.__setattr__(self, "predicate", predicate)
        object.__setattr__(self, "partial_predicate", partial_predicate)
        object.__setattr__(self, "violation", violation)
        object.__setattr__(self, "repair_proposals_callback", repair_proposals)
        object.__setattr__(self, "co_movement_predicate", co_movement_predicate)
        object.__setattr__(self, "include_empty", bool(include_empty))
        object.__setattr__(self, "name", name)

    def prepare(self, context: VFDConstraintContext) -> VFDPreparedConstraint:
        """Prepare the predicate for a contracted problem."""

        return _PreparedCommunityConstraint(self, context)

    def bind(self, context: VFDConstraintContext) -> VFDBoundConstraint:
        """Convenience shorthand for ``prepare(context).bind()``."""

        return self.prepare(context).bind()


class _PreparedCommunityConstraint:
    def __init__(
        self, specification: CommunityPredicateConstraint, context: VFDConstraintContext
    ) -> None:
        self.specification = specification
        self.context = context

    def bind(self) -> VFDBoundConstraint:
        return _BoundCommunityConstraint(self.specification, self.context)


class _BoundCommunityConstraint(_BoundConstraintBase):
    scope = "community"

    def __init__(
        self, specification: CommunityPredicateConstraint, context: VFDConstraintContext
    ) -> None:
        super().__init__(context, specification.name or "community predicate")
        self.specification = specification

    def block_is_feasible(self, components: Sequence[int]) -> bool:
        predicate = self.specification.co_movement_predicate
        if predicate is None:
            return True
        components = tuple(int(component) for component in components)
        return bool(predicate(components, self.context))

    def _evaluate(
        self, assignment: VFDAssignmentView, *, partial: bool
    ) -> VFDConstraintEvaluation:
        predicate = (
            self.specification.partial_predicate
            if partial and self.specification.partial_predicate is not None
            else self.specification.predicate
        )
        evaluations: list[VFDConstraintEvaluation] = []
        use_custom_violation = self.specification.violation is not None and (
            not partial or self.specification.partial_predicate is None
        )
        for community in assignment.iter_communities(
            include_empty=self.specification.include_empty
        ):
            raw_result = predicate(community)
            explicitly_evaluated = isinstance(raw_result, VFDConstraintEvaluation)
            result = _coerce_evaluation(
                raw_result,
                false_extendable=False,
                reason=f"{self.name} failed for community {community.community_id}",
            )
            if use_custom_violation:
                violation = _normalise_violation(self.specification.violation(community))
                if result.satisfied != all(value == 0 for value in violation):
                    raise ValueError(
                        "A community violation callback must be zero exactly when "
                        "its predicate is satisfied."
                    )
                result = VFDConstraintEvaluation(
                    violation,
                    extendable=result.extendable,
                    reason=result.reason,
                )
            elif result.satisfied and not explicitly_evaluated:
                result = VFDConstraintEvaluation((0.0,))
            evaluations.append(result)
        if not evaluations:
            return VFDConstraintEvaluation((0.0,))
        dimensions = {len(result.violation) for result in evaluations}
        if len(dimensions) != 1:
            raise ValueError("Community violation callbacks must use a stable dimension.")
        dimension = dimensions.pop()
        violation = tuple(
            sum(result.violation[index] for result in evaluations)
            for index in range(dimension)
        )
        return VFDConstraintEvaluation(
            violation,
            extendable=all(result.extendable for result in evaluations),
            reason="; ".join(result.reason for result in evaluations if result.reason)
            or None,
        )

    def evaluate_partial(self, assignment: VFDAssignmentView) -> VFDConstraintEvaluation:
        return self._evaluate(assignment, partial=True)

    def evaluate_transition(
        self,
        before: VFDAssignmentView,
        transition: VFDTransition,
        after: VFDAssignmentView,
    ) -> VFDConstraintEvaluation:
        del before
        projected = after.projected(transition.affected_communities)
        return self._evaluate(projected, partial=transition.phase == "construction")

    def evaluate_final(self, assignment: VFDAssignmentView) -> VFDConstraintEvaluation:
        return self._evaluate(assignment, partial=False)

    def repair_proposals(self, assignment: VFDAssignmentView) -> Iterable[VFDTransition]:
        return _callback_proposals(
            self.specification.repair_proposals_callback, assignment
        )


@dataclass(frozen=True)
class QuantifiedCommunityConstraint:
    """Require a bounded number of nonempty communities to match a predicate."""

    predicate: CommunityPredicate
    minimum: int | None = None
    maximum: int | None = None
    exactly: int | None = None
    partial_evaluator: PartialEvaluator | None = field(default=None, repr=False)
    repair_proposals_callback: RepairProposalCallback | None = field(
        default=None, repr=False
    )
    include_empty: bool = False
    name: str | None = None

    def __init__(
        self,
        predicate: CommunityPredicate,
        *,
        minimum: int | None = None,
        maximum: int | None = None,
        exactly: int | None = None,
        partial_evaluator: PartialEvaluator | None = None,
        repair_proposals: RepairProposalCallback | None = None,
        include_empty: bool = False,
        name: str | None = None,
    ) -> None:
        if not callable(predicate):
            raise TypeError("predicate must be callable.")
        if exactly is not None and (minimum is not None or maximum is not None):
            raise ValueError("exactly cannot be combined with minimum or maximum.")
        if exactly is not None:
            minimum = maximum = exactly
        if minimum is None and maximum is None:
            raise ValueError("Supply minimum, maximum, or exactly.")
        minimum = None if minimum is None else int(minimum)
        maximum = None if maximum is None else int(maximum)
        exactly = None if exactly is None else int(exactly)
        if minimum is not None and minimum < 0:
            raise ValueError("minimum must be nonnegative.")
        if maximum is not None and maximum < 0:
            raise ValueError("maximum must be nonnegative.")
        if minimum is not None and maximum is not None and minimum > maximum:
            raise ValueError("minimum cannot exceed maximum.")
        if partial_evaluator is not None and not callable(partial_evaluator):
            raise TypeError("partial_evaluator must be callable when supplied.")
        if repair_proposals is not None and not callable(repair_proposals):
            raise TypeError("repair_proposals must be callable when supplied.")
        object.__setattr__(self, "predicate", predicate)
        object.__setattr__(self, "minimum", minimum)
        object.__setattr__(self, "maximum", maximum)
        object.__setattr__(self, "exactly", exactly)
        object.__setattr__(self, "partial_evaluator", partial_evaluator)
        object.__setattr__(self, "repair_proposals_callback", repair_proposals)
        object.__setattr__(self, "include_empty", bool(include_empty))
        object.__setattr__(self, "name", name)

    def prepare(self, context: VFDConstraintContext) -> VFDPreparedConstraint:
        """Prepare the quantifier for a contracted problem."""

        return _PreparedQuantifiedConstraint(self, context)

    def bind(self, context: VFDConstraintContext) -> VFDBoundConstraint:
        """Convenience shorthand for ``prepare(context).bind()``."""

        return self.prepare(context).bind()


class _PreparedQuantifiedConstraint:
    def __init__(
        self, specification: QuantifiedCommunityConstraint, context: VFDConstraintContext
    ) -> None:
        self.specification = specification
        self.context = context

    def bind(self) -> VFDBoundConstraint:
        return _BoundQuantifiedConstraint(self.specification, self.context)


class _BoundQuantifiedConstraint(_BoundConstraintBase):
    scope = "partition"

    def __init__(
        self, specification: QuantifiedCommunityConstraint, context: VFDConstraintContext
    ) -> None:
        super().__init__(context, specification.name or "quantified community predicate")
        self.specification = specification

    def _count(self, assignment: VFDAssignmentView) -> int:
        return sum(
            _coerce_evaluation(
                self.specification.predicate(community), false_extendable=True
            ).satisfied
            for community in assignment.iter_communities(
                include_empty=self.specification.include_empty
            )
        )

    def _evaluate_current(
        self, assignment: VFDAssignmentView, *, extendable: bool
    ) -> VFDConstraintEvaluation:
        count = self._count(assignment)
        below = (
            0.0
            if self.specification.minimum is None
            else float(max(self.specification.minimum - count, 0))
        )
        above = (
            0.0
            if self.specification.maximum is None
            else float(max(count - self.specification.maximum, 0))
        )
        reason = None if below == above == 0 else f"{self.name} matched {count} communities"
        return VFDConstraintEvaluation(
            (below, above),
            extendable=extendable or below == above == 0,
            reason=reason,
        )

    def evaluate_partial(self, assignment: VFDAssignmentView) -> VFDConstraintEvaluation:
        if self.specification.partial_evaluator is not None:
            return _coerce_evaluation(
                self.specification.partial_evaluator(assignment),
                false_extendable=False,
                reason=f"{self.name} cannot be extended",
            )
        return self._evaluate_current(assignment, extendable=True)

    def evaluate_transition(
        self,
        before: VFDAssignmentView,
        transition: VFDTransition,
        after: VFDAssignmentView,
    ) -> VFDConstraintEvaluation:
        del before
        if transition.phase == "construction":
            return self.evaluate_partial(after)
        return self.evaluate_final(after)

    def evaluate_final(self, assignment: VFDAssignmentView) -> VFDConstraintEvaluation:
        return self._evaluate_current(assignment, extendable=False)

    def repair_proposals(self, assignment: VFDAssignmentView) -> Iterable[VFDTransition]:
        return _callback_proposals(
            self.specification.repair_proposals_callback, assignment
        )


@dataclass(frozen=True)
class PartitionPredicateConstraint:
    """Hard predicate over a lazily materialized whole assignment."""

    predicate: PartitionPredicate
    violation: PartitionViolation | None = None
    partial_evaluator: PartialEvaluator | None = field(default=None, repr=False)
    repair_proposals_callback: RepairProposalCallback | None = field(
        default=None, repr=False
    )
    name: str | None = None

    def __init__(
        self,
        predicate: PartitionPredicate,
        *,
        violation: PartitionViolation | None = None,
        partial_evaluator: PartialEvaluator | None = None,
        repair_proposals: RepairProposalCallback | None = None,
        name: str | None = None,
    ) -> None:
        if not callable(predicate):
            raise TypeError("predicate must be callable.")
        for callback_name, callback in (
            ("violation", violation),
            ("partial_evaluator", partial_evaluator),
            ("repair_proposals", repair_proposals),
        ):
            if callback is not None and not callable(callback):
                raise TypeError(f"{callback_name} must be callable when supplied.")
        object.__setattr__(self, "predicate", predicate)
        object.__setattr__(self, "violation", violation)
        object.__setattr__(self, "partial_evaluator", partial_evaluator)
        object.__setattr__(self, "repair_proposals_callback", repair_proposals)
        object.__setattr__(self, "name", name)

    def prepare(self, context: VFDConstraintContext) -> VFDPreparedConstraint:
        """Prepare the predicate for a contracted problem."""

        return _PreparedPartitionConstraint(self, context)

    def bind(self, context: VFDConstraintContext) -> VFDBoundConstraint:
        """Convenience shorthand for ``prepare(context).bind()``."""

        return self.prepare(context).bind()


class _PreparedPartitionConstraint:
    def __init__(
        self, specification: PartitionPredicateConstraint, context: VFDConstraintContext
    ) -> None:
        self.specification = specification
        self.context = context

    def bind(self) -> VFDBoundConstraint:
        return _BoundPartitionConstraint(self.specification, self.context)


class _BoundPartitionConstraint(_BoundConstraintBase):
    scope = "partition"

    def __init__(
        self, specification: PartitionPredicateConstraint, context: VFDConstraintContext
    ) -> None:
        super().__init__(context, specification.name or "partition predicate")
        self.specification = specification

    def _evaluate_current(
        self, assignment: VFDAssignmentView, *, extendable: bool
    ) -> VFDConstraintEvaluation:
        raw_result = self.specification.predicate(assignment)
        explicitly_evaluated = isinstance(raw_result, VFDConstraintEvaluation)
        result = _coerce_evaluation(
            raw_result,
            false_extendable=extendable,
            reason=f"{self.name} is not satisfied",
        )
        result_extendable = (
            result.extendable
            if explicitly_evaluated
            else result.satisfied or extendable
        )
        if self.specification.violation is not None:
            violation = _normalise_violation(self.specification.violation(assignment))
            if result.satisfied != all(value == 0 for value in violation):
                raise ValueError(
                    "A partition violation callback must be zero exactly when "
                    "its predicate is satisfied."
                )
            result = VFDConstraintEvaluation(
                violation,
                extendable=result_extendable,
                reason=result.reason,
            )
        elif result.satisfied and not explicitly_evaluated:
            result = VFDConstraintEvaluation((0.0,))
        elif result.extendable != result_extendable:
            result = VFDConstraintEvaluation(
                result.violation,
                extendable=result_extendable,
                reason=result.reason,
            )
        return result

    def evaluate_partial(self, assignment: VFDAssignmentView) -> VFDConstraintEvaluation:
        if self.specification.partial_evaluator is not None:
            return _coerce_evaluation(
                self.specification.partial_evaluator(assignment),
                false_extendable=False,
                reason=f"{self.name} cannot be extended",
            )
        return self._evaluate_current(assignment, extendable=True)

    def evaluate_transition(
        self,
        before: VFDAssignmentView,
        transition: VFDTransition,
        after: VFDAssignmentView,
    ) -> VFDConstraintEvaluation:
        del before
        if transition.phase == "construction":
            return self.evaluate_partial(after)
        return self.evaluate_final(after)

    def evaluate_final(self, assignment: VFDAssignmentView) -> VFDConstraintEvaluation:
        return self._evaluate_current(assignment, extendable=False)

    def repair_proposals(self, assignment: VFDAssignmentView) -> Iterable[VFDTransition]:
        return _callback_proposals(
            self.specification.repair_proposals_callback, assignment
        )


__all__ = [
    "CommunityPredicateConstraint",
    "PartitionPredicateConstraint",
    "QuantifiedCommunityConstraint",
    "VFDBoundConstraint",
    "VFDBoundLocalConstraint",
    "VFDCommunityView",
    "VFDComponentMove",
    "VFDConstraint",
    "VFDConstraintContext",
    "VFDConstraintEvaluation",
    "VFDLocalConstraint",
    "VFDPreparedConstraint",
    "VFDPreparedLocalConstraint",
    "VFDAssignmentView",
    "VFDTransition",
]
