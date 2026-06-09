"""Pilot adapter boundary helpers."""

import dataclasses
from collections.abc import Mapping, Sequence
from typing import Any

from vptune.axes.candidates import topological_families
from vptune.core.data import (
    Candidate,
    CheckRecord,
    CohortConstraint,
    Family,
    Plan,
    PlanValidator,
    Problem,
    Target,
    TuningRun,
)
from vptune.core.identities import to_json_value
from vptune.errors import MaterializationError
from vptune.tuning.schemas import selected_plan_validation_input_signature

PILOT_ACCEPTANCE_FAMILIES = (
    "capability_gradient",
    "retain_kl_backward",
    "kfac_metric",
    "capability_hvp",
    "hessian_ritz",
    "chart_retain_curvature",
    "contact_training_step",
)


@dataclasses.dataclass(frozen=True, slots=True)
class PilotReadiness:
    """Readiness of a selected plan for pilot consumption."""

    status: str
    required_families: tuple[str, ...]
    missing_families: tuple[str, ...]
    failed_families: tuple[str, ...]
    stale_families: tuple[str, ...]
    selected_rows: Mapping[str, Mapping[str, Any]]

    def passed(self) -> bool:
        """Return whether all required families are ready."""
        return self.status == "passed"


def lower(
    *,
    target: Target,
    families: Sequence[Family],
    problems: Sequence[Problem],
    run_id: str,
    cohort_constraints: Sequence[CohortConstraint] = (),
    plan_validators: Mapping[str, PlanValidator] | None = None,
    validator_identities: Mapping[str, Mapping[str, Any]] | None = None,
) -> TuningRun:
    """Lower pilot-declared families into a package tuning run.

    Returns:
        Tuning run with families in dependency order.

    Raises:
        MaterializationError: If families and problems disagree.
    """
    ordered_families = topological_families(families)
    problems_by_family = {problem.operator.family: problem for problem in problems}
    family_names = tuple(family.name for family in ordered_families)

    if len(problems_by_family) != len(problems):
        message = "pilot lowering requires unique problem families"
        raise MaterializationError(message)

    if set(problems_by_family) != set(family_names):
        message = "pilot lowering families must match problem families"
        raise MaterializationError(message)

    for family in ordered_families:
        problem = problems_by_family[family.name]

        if problem.operator != family.operator:
            message = f"pilot lowering operator mismatch: {family.name}"
            raise MaterializationError(message)

    selected_validators = (
        {} if plan_validators is None else validators(family_names, plan_validators)
    )
    selected_validator_identities = (
        {}
        if validator_identities is None
        else {
            family: dict(identity) for family, identity in validator_identities.items()
        }
    )

    if not selected_validators and selected_validator_identities:
        message = "pilot validator identities require validators"
        raise MaterializationError(message)

    if selected_validators and set(selected_validator_identities) != set(family_names):
        message = "pilot validator identities must match required families"
        raise MaterializationError(message)

    return TuningRun(
        target=target,
        families=ordered_families,
        problems=tuple(problems),
        cohort_constraints=tuple(cohort_constraints),
        validators=selected_validators,
        validator_identities=selected_validator_identities,
        run_id=run_id,
    )


def acceptance_family_names() -> tuple[str, ...]:
    """Return pilot acceptance family names."""
    return PILOT_ACCEPTANCE_FAMILIES


def acceptance_readiness(plan: Plan) -> PilotReadiness:
    """Return readiness for the pilot acceptance family set."""
    return readiness(plan, PILOT_ACCEPTANCE_FAMILIES)


def require_acceptance_families(families: Sequence[str | Family]) -> None:
    """Require the pilot acceptance family set.

    Raises:
        MaterializationError: If the family set differs from pilot acceptance.
    """
    names = tuple(
        family.name if isinstance(family, Family) else family for family in families
    )

    if set(names) != set(PILOT_ACCEPTANCE_FAMILIES):
        message = "pilot acceptance families differ"
        raise MaterializationError(message)


def readiness(plan: Plan, required_families: Sequence[str]) -> PilotReadiness:
    """Return pilot readiness for required selected families."""
    required = _required_with_dependencies(plan, tuple(required_families))
    missing = tuple(
        family
        for family in required
        if family not in plan.selected or family not in plan.records
    )
    present = tuple(family for family in required if family not in missing)
    failed = tuple(
        family
        for family in present
        if plan.records[family].status != "passed"
        or not plan.records[family].reference_passed
    )
    stale = tuple(family for family in present if _selected_row_stale(plan, family))
    status = "passed" if not missing and not failed and not stale else "failed"

    return PilotReadiness(
        status=status,
        required_families=required,
        missing_families=missing,
        failed_families=failed,
        stale_families=stale,
        selected_rows={family: plan.records[family].row_key() for family in present},
    )


def selected_settings(
    plan: Plan,
    required_families: Sequence[str],
    *,
    validation_records: Sequence[CheckRecord] = (),
) -> dict[str, dict[str, Any]]:
    """Return selected settings for pilot consumers.

    Raises:
        MaterializationError: If the selected plan is not ready.
    """
    state = readiness(plan, required_families)

    if not state.passed():
        message = (
            "pilot selected settings require ready families: "
            f"missing={state.missing_families}, "
            f"failed={state.failed_families}, "
            f"stale={state.stale_families}"
        )
        raise MaterializationError(message)

    selected_validation_records = (
        tuple(validation_records) if validation_records else plan.validation_records
    )
    _require_selected_plan_validation(plan, selected_validation_records)

    return {
        family: {
            "candidate_id": plan.selected[family].candidate_id,
            "settings": dict(plan.selected[family].settings),
            "dependency_identities": dict(plan.selected[family].dependency_identities),
            "record": plan.records[family].row_key(),
        }
        for family in required_families
    }


def validators(
    required_families: Sequence[str],
    plan_validators: Mapping[str, PlanValidator],
) -> Mapping[str, PlanValidator]:
    """Return selected-plan validators after coverage validation.

    Raises:
        MaterializationError: If validator keys differ from required families.
    """
    required = tuple(required_families)

    if set(plan_validators) != set(required):
        message = "pilot validators must match required families"
        raise MaterializationError(message)

    return {family: plan_validators[family] for family in required}


def _require_selected_plan_validation(
    plan: Plan,
    validation_records: Sequence[CheckRecord],
) -> None:
    if not plan.validation_required:
        return

    validation_order = plan.validation_order or tuple(plan.selected)

    if set(validation_order) != set(plan.selected):
        message = "pilot selected settings validation order differs from selection"
        raise MaterializationError(message)

    if tuple(record.family for record in validation_records) != tuple(validation_order):
        message = "pilot selected settings require selected-plan validation rows"
        raise MaterializationError(message)

    input_signature = selected_plan_validation_input_signature(plan)

    for record in validation_records:
        candidate = plan.selected[record.family]

        if to_json_value(record.input_signature) != to_json_value(input_signature):
            message = "pilot selected settings validation input signature differs"
            raise MaterializationError(message)

        _require_validation_record_matches_candidate(record, candidate)


def _selected_row_stale(plan: Plan, family: str) -> bool:
    candidate = plan.selected[family]
    record = plan.records[family]

    return (
        record.family != family
        or record.candidate_id != candidate.candidate_id
        or candidate.admission_status != "passed"
        or to_json_value(record.candidate_settings) != to_json_value(candidate.settings)
        or to_json_value(record.dependency_identities)
        != to_json_value(candidate.dependency_identities)
        or to_json_value(record.cohort_assignment)
        != to_json_value(candidate.cohort_assignment)
        or record.generator_id != candidate.generator_id
        or record.generator_version != candidate.generator_version
    )


def _require_validation_record_matches_candidate(
    record: CheckRecord,
    candidate: Candidate,
) -> None:
    if record.name != "selected_plan_validation":
        message = "pilot selected settings validation row name differs"
        raise MaterializationError(message)

    if record.status != "passed":
        message = "pilot selected settings validation did not pass"
        raise MaterializationError(message)

    if record.candidate_id != candidate.candidate_id:
        message = "pilot selected settings validation candidate differs"
        raise MaterializationError(message)

    if to_json_value(record.candidate_settings) != to_json_value(candidate.settings):
        message = "pilot selected settings validation settings differ"
        raise MaterializationError(message)

    if to_json_value(record.dependency_identities) != to_json_value(
        candidate.dependency_identities
    ):
        message = "pilot selected settings validation dependencies differ"
        raise MaterializationError(message)

    if to_json_value(record.cohort_assignment) != to_json_value(
        candidate.cohort_assignment
    ):
        message = "pilot selected settings validation cohort differs"
        raise MaterializationError(message)

    if record.generator_id != candidate.generator_id:
        message = "pilot selected settings validation generator differs"
        raise MaterializationError(message)

    if record.generator_version != candidate.generator_version:
        message = "pilot selected settings validation generator version differs"
        raise MaterializationError(message)


def _required_with_dependencies(
    plan: Plan,
    required_families: tuple[str, ...],
) -> tuple[str, ...]:
    required = []

    def visit(family: str) -> None:
        if family in required:
            return

        required.append(family)
        candidate = plan.selected.get(family)

        if candidate is None:
            return

        for dependency in candidate.dependency_identities:
            visit(dependency)

    for family in required_families:
        visit(family)

    return tuple(required)
