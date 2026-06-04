"""Cohort assignment helpers."""

import itertools

from vptune.data import Candidate, CohortAssignment, CohortConstraint
from vptune.errors import MaterializationError
from vptune.identities import stable_hash, to_json_value


def constraint_families(
    constraint: CohortConstraint,
    family_names: tuple[str, ...],
) -> tuple[str, ...]:
    """Return families covered by a cohort constraint.

    Raises:
        MaterializationError: If the constraint names unknown families.
    """
    if constraint.families:
        missing = tuple(
            family for family in constraint.families if family not in family_names
        )

        if missing:
            message = f"cohort constraint names unknown families: {missing}"
            raise MaterializationError(message)

        return constraint.families

    return family_names


def validate_cohort_constraints(
    constraints: tuple[CohortConstraint, ...],
    family_names: tuple[str, ...],
) -> None:
    """Validate cohort constraints against run families.

    Raises:
        MaterializationError: If any constraint field is unsupported.
    """
    names = tuple(constraint.name for constraint in constraints)

    if len(set(names)) != len(names):
        message = "cohort constraint names must be unique"
        raise MaterializationError(message)

    for constraint in constraints:
        if not constraint.settings_keys:
            message = f"cohort constraint has no settings keys: {constraint.name}"
            raise MaterializationError(message)

        if not constraint.assignments:
            message = f"cohort constraint has no assignments: {constraint.name}"
            raise MaterializationError(message)

        if constraint.dependency_inheritance != "covered_families":
            message = f"unsupported dependency inheritance: {constraint.name}"
            raise MaterializationError(message)

        if constraint.selection_aggregation != "sum_selection_score_seconds":
            message = f"unsupported cohort selection aggregation: {constraint.name}"
            raise MaterializationError(message)

        constraint_families(constraint, family_names)

        for assignment in constraint.assignments:
            if set(assignment) != set(constraint.settings_keys):
                message = f"cohort assignment keys differ: {constraint.name}"
                raise MaterializationError(message)


def cohort_assignments(
    constraints: tuple[CohortConstraint, ...],
    family_names: tuple[str, ...],
) -> tuple[CohortAssignment, ...]:
    """Return compatible cohort assignments.

    Raises:
        MaterializationError: If constraints cannot form assignments.
    """
    validate_cohort_constraints(constraints, family_names)

    if not constraints:
        return (
            CohortAssignment(
                assignment_id="default",
                values={},
                constraints=(),
                covered_families=(),
            ),
        )

    assignments = []

    for entries in itertools.product(
        *(constraint.assignments for constraint in constraints)
    ):
        values = {}
        covered_families = set()
        valid = True

        for constraint, entry in zip(constraints, entries, strict=True):
            covered_families.update(constraint_families(constraint, family_names))

            for key, value in entry.items():
                if key in values and values[key] != value:
                    valid = False
                    break

                values[key] = value

            if not valid:
                break

        if not valid:
            continue

        signature = {
            "constraints": tuple(constraint.name for constraint in constraints),
            "values": dict(values),
            "covered_families": tuple(sorted(covered_families)),
        }
        assignments.append(
            CohortAssignment(
                assignment_id=stable_hash(signature),
                values=dict(values),
                constraints=tuple(constraint.name for constraint in constraints),
                covered_families=tuple(sorted(covered_families)),
            )
        )

    if not assignments:
        message = "cohort constraints have no compatible assignments"
        raise MaterializationError(message)

    return tuple(assignments)


def candidate_matches_assignment(
    candidate: Candidate,
    assignment: CohortAssignment,
    constraints: tuple[CohortConstraint, ...],
    family_names: tuple[str, ...],
) -> bool:
    """Return whether a candidate belongs to a cohort assignment."""
    matched_constraint = False

    for constraint in constraints:
        if candidate.family not in constraint_families(constraint, family_names):
            continue

        matched_constraint = True

        for key in constraint.settings_keys:
            if candidate.settings.get(key) != assignment.values[key]:
                return False

    if candidate.cohort_assignment and to_json_value(
        candidate.cohort_assignment
    ) != to_json_value(assignment.signature()):
        return False

    if not matched_constraint and candidate.cohort_assignment:
        return to_json_value(candidate.cohort_assignment) == to_json_value(
            assignment.signature()
        )

    return True
