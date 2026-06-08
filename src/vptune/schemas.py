"""Schema validation for saved records."""

import dataclasses
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from vptune.cohorts import candidate_matches_assignment, cohort_assignments
from vptune.data import (
    PACKAGE_VERSION,
    SCHEMA_VERSION,
    Candidate,
    CheckRecord,
    CohortAssignment,
    CohortConstraint,
    FullSizeRecord,
    Materializer,
    Measurement,
    Plan,
    ReplayContext,
    SelectionPolicy,
)
from vptune.errors import MaterializationError, RecordFormatError, StaleRecordError
from vptune.identities import (
    canonical_json,
    to_json_value,
)
from vptune.selection_core import (
    full_size_agreement_satisfied,
    memory_stable,
    select_accepted_family,
    select_complete_cohort,
)

REQUIRED_COMMON_FIELDS = (
    "record_type",
    "schema_version",
    "package_version",
    "input_signature",
    "candidate_settings",
    "status",
    "generator_id",
    "generator_version",
)
REQUIRED_TYPE_FIELDS = {
    "candidate": (
        "family",
        "candidate_id",
        "changed_axes",
        "admission_error",
        "migration_source_id",
        "dependency_identities",
        "cohort_assignment",
    ),
    "reference": (
        "family",
        "candidate_id",
        "name",
        "thresholds",
        "measurements",
        "error_type",
        "error",
        "dependency_identities",
        "cohort_assignment",
    ),
    "full_size": (
        "family",
        "candidate_id",
        "timing_samples",
        "memory_samples",
        "output_signature",
        "error_type",
        "error",
        "reference_passed",
        "dependency_identities",
        "cohort_assignment",
    ),
}
REQUIRED_PLAN_SUMMARY_FIELDS = (
    "selected",
    "candidate_rows",
    "records",
    "full_size_records",
    "check_records",
    "validation_records",
    "validation_required",
    "validation_order",
    "validator_identities",
    "dependencies_by_family",
    "cohort_assignment",
    "cohort_constraints",
    "selected_dependency_identities",
    "materializer_identities",
    "target_identity",
    "runtime_identities",
    "adapter_identities",
    "policy",
)
REQUIRED_VALIDATION_SUMMARY_FIELDS = ("records",)
JSON_MATCH_FIELDS = (
    "changed_axes",
    "input_signature",
    "candidate_settings",
    "dependency_identities",
    "cohort_assignment",
)


@dataclasses.dataclass(frozen=True, slots=True)
class _ValidationReplayIdentity:
    required: bool
    validator_identities: Mapping[str, Mapping[str, Any]]


@dataclasses.dataclass(frozen=True, slots=True)
class _ReplayRows:
    full_size_by_key: Mapping[str, FullSizeRecord]
    ordered_full_size: tuple[FullSizeRecord, ...]
    ordered_checks: tuple[CheckRecord, ...]
    ordered_validation: tuple[CheckRecord, ...]


def validate_json_record(record: Mapping[str, Any]) -> None:
    """Validate common saved-record fields.

    Raises:
        StaleRecordError: If schema or package versions changed.
        RecordFormatError: If required fields are missing or record type is invalid.
    """
    for field in REQUIRED_COMMON_FIELDS:
        if field not in record:
            message = f"record field is missing: {field}"
            raise RecordFormatError(message)

    if record["schema_version"] != SCHEMA_VERSION:
        message = "record schema version changed"
        raise StaleRecordError(message)

    if record["package_version"] != PACKAGE_VERSION:
        message = "record package version changed"
        raise StaleRecordError(message)

    if record["record_type"] not in {"candidate", "reference", "full_size", "summary"}:
        message = f"record type is invalid: {record['record_type']}"
        raise RecordFormatError(message)

    _validate_record_type_fields(record)


def _validate_record_type_fields(record: Mapping[str, Any]) -> None:
    record_type = str(record["record_type"])
    fields = REQUIRED_TYPE_FIELDS.get(record_type, ())

    if record_type == "summary":
        if record["generator_id"] == "selected_plan_validation":
            fields = REQUIRED_VALIDATION_SUMMARY_FIELDS
        else:
            fields = REQUIRED_PLAN_SUMMARY_FIELDS

    for field in fields:
        if field not in record:
            message = f"{record_type} record field is missing: {field}"
            raise RecordFormatError(message)

    if record_type in {"candidate", "reference", "full_size"}:
        _validate_row_input_signature(record)

    if record_type == "summary":
        _validate_summary_identity_fields(record)


def _validate_row_input_signature(record: Mapping[str, Any]) -> None:
    signature = _effective_row_input_signature(record)

    _validate_tuning_input_signature(signature)


def _effective_row_input_signature(record: Mapping[str, Any]) -> Mapping[str, Any]:
    input_signature = record["input_signature"]

    if not isinstance(input_signature, Mapping):
        message = "record input_signature must be a mapping"
        raise RecordFormatError(message)

    if (
        record["record_type"] == "reference"
        and record.get("name") == "selected_plan_validation"
    ):
        nested = input_signature.get("input_signature")

        if not isinstance(nested, Mapping):
            message = "selected-plan validation input signature is missing"
            raise RecordFormatError(message)

        return nested

    parent = input_signature.get("parent_input_signature")

    if isinstance(parent, Mapping):
        return parent

    return input_signature


def _validate_tuning_input_signature(signature: Mapping[str, Any]) -> None:
    if "run_id" in signature:
        family_signatures = tuple(
            value
            for key, value in signature.items()
            if key not in {"run_id", "cohort_assignment"}
        )

        if not family_signatures:
            message = "run input signature has no family signatures"
            raise RecordFormatError(message)

        for family_signature in family_signatures:
            if not isinstance(family_signature, Mapping):
                message = "run family input signature must be a mapping"
                raise RecordFormatError(message)

            _validate_problem_input_signature(family_signature)

        return

    _validate_problem_input_signature(signature)


def _validate_problem_input_signature(signature: Mapping[str, Any]) -> None:
    operator = signature.get("operator")
    target = signature.get("target")
    adapter = signature.get("adapter")

    if not isinstance(operator, Mapping):
        message = "input signature operator is missing"
        raise RecordFormatError(message)

    if not isinstance(target, Mapping) or not target:
        message = "input signature target is missing"
        raise RecordFormatError(message)

    if not isinstance(target.get("environment"), Mapping):
        message = "input signature target environment is missing"
        raise RecordFormatError(message)

    if not isinstance(adapter, Mapping):
        message = "input signature adapter identity is missing"
        raise RecordFormatError(message)

    _validate_adapter_identity(adapter, "input signature adapter")


def _validate_adapter_identity(identity: Mapping[str, Any], label: str) -> None:
    if not identity.get("adapter_id") or not identity.get("adapter_version"):
        message = f"{label} id and version are required"
        raise RecordFormatError(message)


def _validate_summary_identity_fields(record: Mapping[str, Any]) -> None:
    if record["generator_id"] == "selected_plan_validation":
        input_signature = record["input_signature"]

        if not isinstance(input_signature, Mapping):
            message = "selected-plan validation summary input signature is missing"
            raise RecordFormatError(message)

        nested = input_signature.get("input_signature")

        if not isinstance(nested, Mapping):
            message = "selected-plan validation summary input signature is missing"
            raise RecordFormatError(message)

        _validate_tuning_input_signature(nested)

        return

    target_identity = record["target_identity"]

    if not isinstance(target_identity, Mapping) or not target_identity:
        message = "plan summary target_identity is missing"
        raise RecordFormatError(message)

    if not isinstance(target_identity.get("environment"), Mapping):
        message = "plan summary target environment is missing"
        raise RecordFormatError(message)

    selected = _selected_family_names(record)
    _validate_summary_family_identities(
        "runtime",
        record["runtime_identities"],
        selected,
    )
    _validate_summary_family_identities(
        "adapter",
        record["adapter_identities"],
        selected,
    )
    _validate_summary_family_identities(
        "materializer",
        record["materializer_identities"],
        selected,
    )


def _validate_summary_family_identities(
    label: str,
    identities: Any,
    selected: set[str],
) -> None:
    if not isinstance(identities, Mapping) or set(identities) != selected:
        message = f"plan summary {label} identities are missing"
        raise RecordFormatError(message)

    for family, identity in identities.items():
        if not isinstance(identity, Mapping) or not identity:
            message = f"plan summary {label} identity is missing: {family}"
            raise RecordFormatError(message)

        if label == "adapter":
            _validate_adapter_identity(identity, f"plan summary adapter {family}")


def _record_mapping_field(
    record: Mapping[str, Any],
    field: str,
    label: str,
) -> Mapping[Any, Any]:
    try:
        value = record[field]
    except KeyError as error:
        message = f"{label} field is missing"
        raise RecordFormatError(message) from error

    if not isinstance(value, Mapping):
        message = f"{label} must be a mapping"
        raise RecordFormatError(message)

    return value


def _selected_candidate_records(record: Mapping[str, Any]) -> Mapping[Any, Any]:
    return _record_mapping_field(
        record,
        "selected",
        "plan summary selected candidates",
    )


def _selected_family_names(record: Mapping[str, Any]) -> set[str]:
    return {str(family) for family in _selected_candidate_records(record)}


def plan_input_signature_from_json(record: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return the input signature from a saved plan summary.

    Raises:
        RecordFormatError: If the input signature is malformed.
    """
    try:
        input_signature = record["input_signature"]
    except KeyError as error:
        message = "plan summary input_signature field is missing"
        raise RecordFormatError(message) from error

    if not isinstance(input_signature, Mapping):
        message = "plan summary input_signature must be a mapping"
        raise RecordFormatError(message)

    return input_signature


def plan_validation_order_from_json(record: Mapping[str, Any]) -> tuple[str, ...]:
    """Return the validation order from a saved plan summary.

    Raises:
        RecordFormatError: If the validation order is malformed.
    """
    try:
        value = record["validation_order"]
    except KeyError as error:
        message = "plan summary validation_order field is missing"
        raise RecordFormatError(message) from error

    if isinstance(value, str) or not isinstance(value, Sequence):
        message = "plan summary validation_order must be a sequence"
        raise RecordFormatError(message)

    return tuple(str(family) for family in value)


def plan_validation_required_from_json(record: Mapping[str, Any]) -> bool:
    """Return the validation-required flag from a saved plan summary.

    Raises:
        RecordFormatError: If the validation-required flag is malformed.
    """
    try:
        value = record["validation_required"]
    except KeyError as error:
        message = "plan summary validation_required field is missing"
        raise RecordFormatError(message) from error

    if value is not True and value is not False:
        message = "plan summary validation_required must be a bool"
        raise RecordFormatError(message)

    return value


def record_current(
    record: Mapping[str, Any],
    *,
    record_type: str,
    family: str,
    candidate_id: str,
    check_name: str = "",
    status: str | None = None,
    input_signature: Mapping[str, Any],
    candidate_settings: Mapping[str, Any],
    thresholds: Mapping[str, float] | None = None,
    dependency_identities: Mapping[str, Mapping[str, Any]] | None = None,
    cohort_assignment: Mapping[str, Any] | None = None,
    changed_axes: Sequence[str] = (),
    generator_id: str,
    generator_version: str,
) -> bool:
    """Return whether a saved row matches current identity."""
    try:
        validate_json_record(record)
    except (RecordFormatError, StaleRecordError):
        return False

    expected = _expected_record_fields(
        record_type=record_type,
        family=family,
        candidate_id=candidate_id,
        check_name=check_name,
        status=status,
        input_signature=input_signature,
        candidate_settings=candidate_settings,
        thresholds=thresholds,
        dependency_identities=dependency_identities,
        cohort_assignment=cohort_assignment,
        changed_axes=changed_axes,
        generator_id=generator_id,
        generator_version=generator_version,
    )

    return _record_fields_current(record, expected)


def _expected_record_fields(
    *,
    record_type: str,
    family: str,
    candidate_id: str,
    check_name: str,
    status: str | None,
    input_signature: Mapping[str, Any],
    candidate_settings: Mapping[str, Any],
    thresholds: Mapping[str, float] | None,
    dependency_identities: Mapping[str, Mapping[str, Any]] | None,
    cohort_assignment: Mapping[str, Any] | None,
    changed_axes: Sequence[str],
    generator_id: str,
    generator_version: str,
) -> dict[str, Any]:
    expected = _common_expected_record_fields(
        record_type=record_type,
        family=family,
        candidate_id=candidate_id,
        input_signature=input_signature,
        candidate_settings=candidate_settings,
        dependency_identities=dependency_identities,
        cohort_assignment=cohort_assignment,
        generator_id=generator_id,
        generator_version=generator_version,
    )

    if record_type == "reference":
        reference_expected = {**expected, "name": check_name}

        if status is not None:
            reference_expected["status"] = status

        if thresholds is not None:
            reference_expected["thresholds"] = dict(thresholds)

        return reference_expected

    if record_type == "candidate":
        candidate_expected = {**expected, "changed_axes": tuple(changed_axes)}

        if status is not None:
            candidate_expected["status"] = status

        return candidate_expected

    if status is not None:
        return {**expected, "status": status}

    return expected


def _common_expected_record_fields(
    *,
    record_type: str,
    family: str,
    candidate_id: str,
    input_signature: Mapping[str, Any],
    candidate_settings: Mapping[str, Any],
    dependency_identities: Mapping[str, Mapping[str, Any]] | None,
    cohort_assignment: Mapping[str, Any] | None,
    generator_id: str,
    generator_version: str,
) -> dict[str, Any]:
    return {
        "record_type": record_type,
        "family": family,
        "candidate_id": candidate_id,
        "input_signature": dict(input_signature),
        "candidate_settings": dict(candidate_settings),
        "dependency_identities": _dependency_identity_record(dependency_identities),
        "cohort_assignment": {}
        if cohort_assignment is None
        else dict(cohort_assignment),
        "generator_id": generator_id,
        "generator_version": generator_version,
    }


def _record_fields_current(
    record: Mapping[str, Any],
    expected: Mapping[str, Any],
) -> bool:
    return all(
        _record_field_current(record, key, value) for key, value in expected.items()
    )


def _record_field_current(
    record: Mapping[str, Any],
    key: str,
    expected: Any,
) -> bool:
    if key in JSON_MATCH_FIELDS:
        return to_json_value(record.get(key)) == to_json_value(expected)

    return record.get(key) == expected


def _dependency_identity_record(
    dependency_identities: Mapping[str, Mapping[str, Any]] | None,
) -> dict[str, dict[str, Any]]:
    if dependency_identities is None:
        return {}

    return {
        str(family): dict(identity)
        for family, identity in sorted(dependency_identities.items())
    }


def candidate_record_to_json(
    candidate: Candidate,
    input_signature: Mapping[str, Any],
) -> dict[str, Any]:
    """Return a JSON row for a candidate."""
    return {
        "record_type": "candidate",
        "schema_version": SCHEMA_VERSION,
        "package_version": PACKAGE_VERSION,
        "input_signature": dict(input_signature),
        "candidate_settings": dict(candidate.settings),
        "dependency_identities": _dependency_identity_record(
            candidate.dependency_identities
        ),
        "cohort_assignment": dict(candidate.cohort_assignment),
        "status": candidate.admission_status,
        "generator_id": candidate.generator_id,
        "generator_version": candidate.generator_version,
        "family": candidate.family,
        "candidate_id": candidate.candidate_id,
        "changed_axes": candidate.changed_axes,
        "admission_error": candidate.admission_error,
        "migration_source_id": candidate.migration_source_id,
    }


def candidate_from_signature(record: Mapping[str, Any]) -> Candidate:
    """Return a candidate from a saved candidate signature."""
    return _candidate_from_record_fields(
        record,
        settings_field="settings",
        status_field="admission_status",
        label="candidate signature",
    )


def _candidate_from_record_fields(
    record: Mapping[str, Any],
    *,
    settings_field: str,
    status_field: str,
    label: str,
) -> Candidate:
    try:
        return Candidate(
            family=str(record["family"]),
            candidate_id=str(record["candidate_id"]),
            settings=dict(record[settings_field]),
            changed_axes=tuple(str(axis) for axis in record["changed_axes"]),
            dependency_identities={
                str(key): dict(value)
                for key, value in dict(record["dependency_identities"]).items()
            },
            cohort_assignment=dict(record["cohort_assignment"]),
            admission_status=str(record[status_field]),
            admission_error=(
                None
                if record.get("admission_error") is None
                else str(record["admission_error"])
            ),
            generator_id=str(record["generator_id"]),
            generator_version=str(record["generator_version"]),
            migration_source_id=(
                None
                if record.get("migration_source_id") is None
                else str(record["migration_source_id"])
            ),
        )
    except (KeyError, TypeError, ValueError) as error:
        message = f"{label} is invalid: {error}"
        raise RecordFormatError(message) from error


def candidate_record_from_json(record: Mapping[str, Any]) -> Candidate:
    """Return a candidate from a saved candidate row.

    Raises:
        StaleRecordError: If the saved candidate row differs from its fields.
        RecordFormatError: If the row is not a candidate record.
    """
    validate_json_record(record)

    if record["record_type"] != "candidate":
        message = "candidate replay requires a candidate record"
        raise RecordFormatError(message)

    candidate = _candidate_from_record_fields(
        record,
        settings_field="candidate_settings",
        status_field="status",
        label="candidate record",
    )

    if not record_current(
        record,
        record_type="candidate",
        family=candidate.family,
        candidate_id=candidate.candidate_id,
        input_signature=record["input_signature"],
        candidate_settings=candidate.settings,
        status=candidate.admission_status,
        dependency_identities=candidate.dependency_identities,
        cohort_assignment=candidate.cohort_assignment,
        changed_axes=candidate.changed_axes,
        generator_id=candidate.generator_id,
        generator_version=candidate.generator_version,
    ):
        message = "candidate record fields differ"
        raise StaleRecordError(message)

    return candidate


def check_record_to_json(record: CheckRecord) -> dict[str, Any]:
    """Return a JSON row for a reference check."""
    payload = dataclasses.asdict(record)
    payload["record_type"] = "reference"

    return payload


def check_record_from_json(record: Mapping[str, Any]) -> CheckRecord:
    """Return a check record from a JSON row.

    Raises:
        StaleRecordError: If the reference row differs from its fields.
        RecordFormatError: If the row is malformed or is not a reference record.
    """
    validate_json_record(record)

    if record["record_type"] != "reference":
        message = "check replay requires a reference record"
        raise RecordFormatError(message)

    try:
        payload = _check_record_payload(record)
        check_record = CheckRecord(**payload)
        current = check_record_current(check_record)
    except (KeyError, TypeError, ValueError) as error:
        message = f"reference record is invalid: {error}"
        raise RecordFormatError(message) from error

    if not current:
        message = "reference record fields differ"
        raise StaleRecordError(message)

    return check_record


def _check_record_payload(record: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(record)
    payload.pop("record_type")
    payload["input_signature"] = dict(payload["input_signature"])
    payload["candidate_settings"] = dict(payload["candidate_settings"])
    payload["thresholds"] = dict(payload["thresholds"])
    payload["measurements"] = dict(payload["measurements"])
    payload["dependency_identities"] = {
        str(key): dict(identity)
        for key, identity in dict(payload["dependency_identities"]).items()
    }
    payload["cohort_assignment"] = dict(payload["cohort_assignment"])

    return payload


def check_record_current(record: CheckRecord) -> bool:
    """Return whether a reference row stores matching direct fields."""
    return record_current(
        {
            **dataclasses.asdict(record),
            "record_type": "reference",
        },
        record_type="reference",
        family=record.family,
        candidate_id=record.candidate_id,
        check_name=record.name,
        status=record.status,
        input_signature=record.input_signature,
        candidate_settings=record.candidate_settings,
        thresholds=record.thresholds,
        dependency_identities=record.dependency_identities,
        cohort_assignment=record.cohort_assignment,
        generator_id=record.generator_id,
        generator_version=record.generator_version,
    )


def full_size_record_to_json(record: FullSizeRecord) -> dict[str, Any]:
    """Return a JSON row for a full-size result."""
    payload = dataclasses.asdict(record)
    payload["record_type"] = "full_size"

    return payload


def measurement_from_json(record: Mapping[str, Any]) -> Measurement:
    """Return a measurement from a JSON row.

    Raises:
        RecordFormatError: If the measurement row is malformed.
    """
    try:
        return Measurement(**dict(record))
    except (KeyError, TypeError, ValueError) as error:
        message = f"measurement record is invalid: {error}"
        raise RecordFormatError(message) from error


def full_size_record_from_json(record: Mapping[str, Any]) -> FullSizeRecord:
    """Return a full-size record from a JSON row.

    Raises:
        RecordFormatError: If the row is malformed.
        StaleRecordError: If the full-size row differs from its fields.
    """
    validate_json_record(record)

    try:
        payload = dict(record)
        payload.pop("record_type")
        payload["timing_samples"] = tuple(
            measurement_from_json(sample) for sample in payload["timing_samples"]
        )
        payload["memory_samples"] = tuple(
            measurement_from_json(sample) for sample in payload["memory_samples"]
        )
        full_size_record = FullSizeRecord(**payload)
    except MaterializationError as error:
        message = f"full-size record is invalid: {error}"
        raise RecordFormatError(message) from error
    except (KeyError, TypeError, ValueError) as error:
        message = f"full-size record is invalid: {error}"
        raise RecordFormatError(message) from error

    if not full_size_record_current(full_size_record):
        message = "full-size record fields differ"
        raise StaleRecordError(message)

    return full_size_record


def full_size_record_current(record: FullSizeRecord) -> bool:
    """Return whether a full-size row stores matching direct fields."""
    return record_current(
        {
            **dataclasses.asdict(record),
            "record_type": "full_size",
        },
        record_type="full_size",
        family=record.family,
        candidate_id=record.candidate_id,
        status=record.status,
        input_signature=record.input_signature,
        candidate_settings=record.candidate_settings,
        dependency_identities=record.dependency_identities,
        cohort_assignment=record.cohort_assignment,
        generator_id=record.generator_id,
        generator_version=record.generator_version,
    )


def plan_to_json(plan: Plan) -> dict[str, Any]:
    """Return a JSON row for a selected plan."""
    return plan.to_record()


def selected_plan_validation_summary_record(
    plan: Plan,
    records: Sequence[CheckRecord],
) -> dict[str, Any]:
    """Return the selected-plan validation summary row."""
    input_signature = selected_plan_validation_input_signature(plan)
    record_keys = tuple(record.row_key() for record in records)
    candidate_settings = {
        "validation_order": plan.validation_order,
        "records": record_keys,
    }
    status = (
        "passed" if all(record.status == "passed" for record in records) else "failed"
    )

    return {
        "record_type": "summary",
        "schema_version": SCHEMA_VERSION,
        "package_version": PACKAGE_VERSION,
        "input_signature": input_signature,
        "candidate_settings": candidate_settings,
        "status": status,
        "generator_id": "selected_plan_validation",
        "generator_version": PACKAGE_VERSION,
        "records": record_keys,
    }


def selected_plan_validation_input_signature(plan: Plan) -> dict[str, Any]:
    """Return direct fields that selected-plan validation checks."""
    return {
        "input_signature": dict(plan.input_signature),
        "selected": {
            family: candidate.signature()
            for family, candidate in sorted(plan.selected.items())
        },
        "records": {
            family: record.row_key() for family, record in sorted(plan.records.items())
        },
        "validation_order": plan.validation_order,
    }


def selected_plan_validation_summary_current(
    record: Mapping[str, Any],
    plan: Plan,
    records: Sequence[CheckRecord],
) -> bool:
    """Return whether a selected-plan validation summary matches its rows."""
    try:
        validate_json_record(record)
    except (RecordFormatError, StaleRecordError):
        return False

    if record.get("record_type") != "summary":
        return False

    expected = selected_plan_validation_summary_record(plan, records)

    for field in (
        "record_type",
        "schema_version",
        "package_version",
        "input_signature",
        "candidate_settings",
        "status",
        "generator_id",
        "generator_version",
        "records",
    ):
        if to_json_value(record.get(field)) != to_json_value(expected[field]):
            return False

    return True


def _validate_selected_plan_validation_rows(
    plan: Plan,
    summary: Mapping[str, Any],
    records: Sequence[CheckRecord],
) -> None:
    validation_order = plan.validation_order or tuple(plan.selected)
    expected_input_signature = selected_plan_validation_input_signature(plan)

    if tuple(record.family for record in records) != tuple(validation_order):
        message = "plan replay selected-plan validation order differs"
        raise RecordFormatError(message)

    if set(validation_order) != set(plan.selected):
        message = "plan replay selected-plan validation families differ"
        raise RecordFormatError(message)

    for record in records:
        _validate_selected_plan_validation_row(
            plan.selected[record.family],
            expected_input_signature,
            record,
        )

    if summary.get("status") != "passed" or any(
        record.status != "passed" for record in records
    ):
        message = "plan replay selected-plan validation did not pass"
        raise RecordFormatError(message)


def _validate_selected_plan_validation_row(
    candidate: Candidate,
    expected_input_signature: Mapping[str, Any],
    record: CheckRecord,
) -> None:
    if record.name != "selected_plan_validation":
        message = "plan replay validation row name differs"
        raise RecordFormatError(message)

    if to_json_value(record.input_signature) != to_json_value(expected_input_signature):
        message = "plan replay validation row input signature differs"
        raise StaleRecordError(message)

    if record.candidate_id != candidate.candidate_id:
        message = "plan replay validation row candidate differs"
        raise StaleRecordError(message)

    if to_json_value(record.candidate_settings) != to_json_value(candidate.settings):
        message = "plan replay validation row settings differ"
        raise StaleRecordError(message)

    if to_json_value(record.dependency_identities) != to_json_value(
        candidate.dependency_identities
    ):
        message = "plan replay validation row dependencies differ"
        raise StaleRecordError(message)

    if to_json_value(record.cohort_assignment) != to_json_value(
        candidate.cohort_assignment
    ):
        message = "plan replay validation row cohort assignment differs"
        raise StaleRecordError(message)

    if record.generator_id != candidate.generator_id:
        message = "plan replay validation row generator differs"
        raise StaleRecordError(message)

    if record.generator_version != candidate.generator_version:
        message = "plan replay validation row generator version differs"
        raise StaleRecordError(message)


def plan_record_current(record: Mapping[str, Any], plan: Plan) -> bool:
    """Return whether a saved plan matches the current selected plan."""
    try:
        validate_json_record(record)
    except (RecordFormatError, StaleRecordError):
        return False

    expected = plan.to_record()

    for field in (
        "record_type",
        "input_signature",
        "candidate_settings",
        "selected",
        "candidate_rows",
        "records",
        "full_size_records",
        "check_records",
        "validation_records",
        "validation_required",
        "validation_order",
        "validator_identities",
        "dependencies_by_family",
        "cohort_assignment",
        "cohort_constraints",
        "selected_dependency_identities",
        "materializer_identities",
        "target_identity",
        "runtime_identities",
        "adapter_identities",
        "policy",
        "status",
        "generator_id",
        "generator_version",
        "schema_version",
        "package_version",
    ):
        if to_json_value(record.get(field)) != to_json_value(expected[field]):
            return False

    return True


def _record_by_row_key(records: Sequence[Any], label: str) -> dict[str, Any]:
    by_key = {canonical_json(record.row_key()): record for record in records}

    if len(by_key) != len(records):
        message = f"plan replay has duplicate {label} rows"
        raise RecordFormatError(message)

    return by_key


def _check_record_by_lookup_key(
    records: Sequence[CheckRecord],
    label: str,
) -> dict[str, CheckRecord]:
    by_key = {_check_row_lookup_key(record.row_key()): record for record in records}

    if len(by_key) != len(records):
        message = f"plan replay has duplicate {label} rows"
        raise RecordFormatError(message)

    return by_key


def _check_row_lookup_key(row_key: Mapping[str, Any]) -> str:
    lookup = dict(row_key)
    lookup.pop("status", None)

    return canonical_json(lookup)


def _record_sequence_field(
    record: Mapping[str, Any],
    field: str,
    label: str,
) -> tuple[Any, ...]:
    try:
        value = record[field]
    except KeyError as error:
        message = f"{label} field is missing"
        raise RecordFormatError(message) from error

    if isinstance(value, str) or not isinstance(value, Sequence):
        message = f"{label} must be a sequence"
        raise RecordFormatError(message)

    return tuple(value)


def _records_in_saved_order(
    row_keys: Sequence[Any],
    records_by_key: Mapping[str, Any],
    label: str,
) -> tuple[Any, ...]:
    ordered = []

    for value in row_keys:
        row_key = canonical_json(value)
        record = records_by_key.get(row_key)

        if record is None:
            message = f"plan replay missing {label} row: {value}"
            raise RecordFormatError(message)

        ordered.append(record)

    if set(records_by_key) != {canonical_json(value) for value in row_keys}:
        message = f"plan replay received extra {label} rows"
        raise RecordFormatError(message)

    return tuple(ordered)


def _check_records_in_saved_order(
    row_keys: Sequence[Any],
    records_by_key: Mapping[str, CheckRecord],
    label: str,
) -> tuple[CheckRecord, ...]:
    ordered = []
    row_lookup_keys = []

    for value in row_keys:
        if not isinstance(value, Mapping):
            message = f"plan replay {label} row key must be a mapping"
            raise RecordFormatError(message)

        row_key = _check_row_lookup_key(value)
        row_lookup_keys.append(row_key)
        record = records_by_key.get(row_key)

        if record is None:
            message = f"plan replay missing {label} row: {value}"
            raise RecordFormatError(message)

        if to_json_value(record.row_key()) != to_json_value(value):
            message = f"plan replay stale {label} row: {value}"
            raise StaleRecordError(message)

        ordered.append(record)

    if set(records_by_key) != set(row_lookup_keys):
        message = f"plan replay received extra {label} rows"
        raise RecordFormatError(message)

    return tuple(ordered)


def selection_policy_from_json(record: Mapping[str, Any]) -> SelectionPolicy:
    """Return a selection policy parsed from saved JSON.

    Raises:
        RecordFormatError: If the saved policy fields are malformed.
    """
    if not isinstance(record, Mapping):
        message = "plan replay selection policy must be a mapping"
        raise RecordFormatError(message)

    policy = dict(record)
    expected = {field.name for field in dataclasses.fields(SelectionPolicy)}

    if set(policy) != expected:
        message = "plan replay selection policy fields differ"
        raise RecordFormatError(message)

    try:
        return SelectionPolicy(**policy)
    except MaterializationError as error:
        message = f"plan replay selection policy is invalid: {error}"
        raise RecordFormatError(message) from error


def cohort_assignment_from_json(
    record: Mapping[str, Any] | None,
) -> CohortAssignment | None:
    """Return a cohort assignment parsed from saved JSON.

    Raises:
        RecordFormatError: If the saved assignment fields are malformed.
    """
    if record is None:
        return None

    for field in ("assignment_id", "values", "constraints", "covered_families"):
        if field not in record:
            message = f"plan replay cohort assignment field is missing: {field}"
            raise RecordFormatError(message)

    values = record["values"]

    if not isinstance(values, Mapping):
        message = "plan replay cohort assignment values must be a mapping"
        raise RecordFormatError(message)

    constraints = record["constraints"]

    if isinstance(constraints, str) or not isinstance(constraints, Sequence):
        message = "plan replay cohort assignment constraints must be a sequence"
        raise RecordFormatError(message)

    covered_families = record["covered_families"]

    if isinstance(covered_families, str) or not isinstance(
        covered_families,
        Sequence,
    ):
        message = "plan replay cohort assignment covered_families must be a sequence"
        raise RecordFormatError(message)

    return CohortAssignment(
        assignment_id=str(record["assignment_id"]),
        values=dict(values),
        constraints=tuple(str(name) for name in constraints),
        covered_families=tuple(str(family) for family in covered_families),
    )


def _cohort_constraints_from_json(records: Any) -> tuple[CohortConstraint, ...]:
    if isinstance(records, str) or not isinstance(records, Sequence):
        message = "plan replay cohort constraints must be a sequence"
        raise RecordFormatError(message)

    try:
        return tuple(_cohort_constraint_from_json(record) for record in records)
    except (KeyError, TypeError, ValueError, MaterializationError) as error:
        message = f"plan replay cohort constraints are invalid: {error}"
        raise RecordFormatError(message) from error


def _cohort_constraint_from_json(record: Any) -> CohortConstraint:
    if not isinstance(record, Mapping):
        message = "plan replay cohort constraints must contain mappings"
        raise RecordFormatError(message)

    settings_keys = _record_sequence_field(
        record,
        "settings_keys",
        "plan replay cohort constraint settings_keys",
    )
    assignments = _cohort_constraint_assignments_from_json(
        _record_sequence_field(
            record,
            "assignments",
            "plan replay cohort constraint assignments",
        ),
    )
    families = _record_sequence_field(
        record,
        "families",
        "plan replay cohort constraint families",
    )

    return CohortConstraint(
        name=str(record["name"]),
        settings_keys=tuple(str(key) for key in settings_keys),
        assignments=assignments,
        families=tuple(str(family) for family in families),
        dependency_inheritance=str(record["dependency_inheritance"]),
        selection_aggregation=str(record["selection_aggregation"]),
    )


def _cohort_constraint_assignments_from_json(
    records: Sequence[Any],
) -> tuple[Mapping[str, Any], ...]:
    assignments = []

    for assignment in records:
        if not isinstance(assignment, Mapping):
            message = "plan replay cohort constraint assignments must contain mappings"
            raise RecordFormatError(message)

        assignments.append(dict(assignment))

    return tuple(assignments)


def _validation_replay_identity(
    record: Mapping[str, Any],
    replay_context: ReplayContext,
) -> _ValidationReplayIdentity:
    saved_required = plan_validation_required_from_json(record)
    required = saved_required or replay_context.validation_required
    validator_identities = {
        str(family): dict(identity)
        for family, identity in replay_context.validator_identities.items()
    }

    if required and (
        not validator_identities
        or any(not identity for identity in validator_identities.values())
    ):
        message = "plan replay validator identities are missing"
        raise RecordFormatError(message)

    return _ValidationReplayIdentity(
        required=required,
        validator_identities=validator_identities,
    )


def _validate_ordered_replay_rows(
    ordered_full_size: Sequence[FullSizeRecord],
    ordered_checks: Sequence[CheckRecord],
) -> None:
    _validate_reference_linkage(ordered_full_size, ordered_checks)


def _row_candidate_key(
    row: FullSizeRecord | CheckRecord,
) -> str:
    return _result_candidate_key(row)


def _reference_matches_full_size(
    check: CheckRecord,
    record: FullSizeRecord,
) -> bool:
    return (
        check.status == "passed"
        and to_json_value(check.input_signature)
        == to_json_value(record.input_signature)
        and to_json_value(check.candidate_settings)
        == to_json_value(record.candidate_settings)
        and to_json_value(check.dependency_identities)
        == to_json_value(record.dependency_identities)
        and check.generator_id == record.generator_id
        and check.generator_version == record.generator_version
    )


def _validate_reference_linkage(
    full_size_records: Sequence[FullSizeRecord],
    check_records: Sequence[CheckRecord],
) -> None:
    checks_by_key = {}

    for check in check_records:
        checks_by_key.setdefault(_row_candidate_key(check), []).append(check)

    for record in full_size_records:
        if (
            record.status != "passed"
            or not record.reference_passed
            or not full_size_agreement_satisfied(record)
        ):
            continue

        if not any(
            _reference_matches_full_size(check, record)
            for check in checks_by_key.get(_row_candidate_key(record), ())
        ):
            message = (
                "plan replay accepted full-size row has no matching passed reference: "
                f"{record.family}/{record.candidate_id}"
            )
            raise RecordFormatError(message)


def _selected_record_keys(
    record: Mapping[str, Any],
    selected: Mapping[str, Candidate],
    full_size_by_key: Mapping[str, FullSizeRecord],
) -> dict[str, str]:
    selected_record_keys = {
        str(family): canonical_json(row_key)
        for family, row_key in _record_mapping_field(
            record,
            "records",
            "plan summary selected records",
        ).items()
    }

    if set(selected_record_keys) != set(selected):
        message = "plan replay selected families differ from record families"
        raise RecordFormatError(message)

    missing_records = tuple(
        family
        for family, row_key in selected_record_keys.items()
        if row_key not in full_size_by_key
    )

    if missing_records:
        message = f"plan replay missing full-size records: {missing_records}"
        raise RecordFormatError(message)

    return selected_record_keys


def selected_records_from_json(
    record: Mapping[str, Any],
    selected: Mapping[str, Candidate],
    full_size_records: Sequence[FullSizeRecord],
) -> dict[str, FullSizeRecord]:
    """Return selected full-size rows from a saved plan summary."""
    full_size_by_key = _record_by_row_key(full_size_records, "full-size")
    selected_record_keys = _selected_record_keys(record, selected, full_size_by_key)

    return _selected_records(selected, selected_record_keys, full_size_by_key)


def _validate_materializers(
    selected: Mapping[str, Candidate],
    materializers: Mapping[str, Materializer],
) -> None:
    missing_materializers = tuple(
        family for family in selected if family not in materializers
    )

    if missing_materializers:
        message = f"plan replay missing materializers: {missing_materializers}"
        raise RecordFormatError(message)


def _materializer_identities(
    materializers: Mapping[str, Materializer],
) -> dict[str, dict[str, Any]]:
    return {
        family: dict(materializer.identity())
        for family, materializer in sorted(materializers.items())
    }


def _validate_replay_context(
    record: Mapping[str, Any],
    replay_context: ReplayContext,
    materializers: Mapping[str, Materializer],
    ordered_full_size: Sequence[FullSizeRecord],
    ordered_checks: Sequence[CheckRecord],
) -> None:
    selected_families = tuple(_selected_family_names(record))
    records = _record_mapping_field(
        record,
        "records",
        "plan summary selected records",
    )
    selected_record_keys = {canonical_json(row_key) for row_key in records.values()}

    _validate_replay_run_identity(record, replay_context)
    _validate_replay_materializers(record, replay_context, materializers)
    _validate_replay_target_identity(record, replay_context)
    _validate_replay_family_identities(
        "runtime",
        record["runtime_identities"],
        replay_context.runtime_identities,
        selected_families,
    )
    _validate_replay_family_identities(
        "adapter",
        record["adapter_identities"],
        replay_context.adapter_identities,
        selected_families,
    )

    for row in ordered_full_size:
        expected_signature = replay_context.family_input_signatures.get(row.family)

        if expected_signature is None:
            message = f"plan replay missing family input signature: {row.family}"
            raise RecordFormatError(message)

        if canonical_json(
            row.row_key()
        ) not in selected_record_keys and not _row_matches_signature(
            row, expected_signature
        ):
            continue

        if not _row_matches_signature(row, expected_signature):
            message = f"plan replay family input signature is stale: {row.family}"
            raise StaleRecordError(message)

    for row in ordered_checks:
        expected_signature = replay_context.family_input_signatures.get(row.family)

        if expected_signature is None:
            if row.family not in selected_families:
                continue

            message = f"plan replay missing reference input signature: {row.family}"
            raise RecordFormatError(message)

        if not _row_matches_signature(row, expected_signature):
            continue


def _row_matches_signature(
    row: FullSizeRecord | CheckRecord,
    expected_signature: Mapping[str, Any],
) -> bool:
    return to_json_value(row.input_signature) == to_json_value(dict(expected_signature))


def _validate_replay_run_identity(
    record: Mapping[str, Any],
    replay_context: ReplayContext,
) -> None:
    if to_json_value(plan_input_signature_from_json(record)) != to_json_value(
        dict(replay_context.input_signature)
    ):
        message = "plan replay input signature is stale"
        raise StaleRecordError(message)

    saved_policy = selection_policy_from_json(record["policy"])

    if to_json_value(dataclasses.asdict(saved_policy)) != to_json_value(
        dataclasses.asdict(replay_context.selection_policy)
    ):
        message = "plan replay selection policy is stale"
        raise StaleRecordError(message)


def _validate_replay_materializers(
    record: Mapping[str, Any],
    replay_context: ReplayContext,
    materializers: Mapping[str, Materializer],
) -> None:
    saved_materializers = dict(record["materializer_identities"])
    context_materializers = {
        family: dict(identity)
        for family, identity in replay_context.materializer_identities.items()
    }
    supplied_materializers = _materializer_identities(materializers)

    if to_json_value(saved_materializers) != to_json_value(context_materializers):
        message = "plan replay materializer context is stale"
        raise StaleRecordError(message)

    if to_json_value(saved_materializers) != to_json_value(supplied_materializers):
        message = "plan replay materializer identities differ"
        raise StaleRecordError(message)


def _validate_replay_target_identity(
    record: Mapping[str, Any],
    replay_context: ReplayContext,
) -> None:
    if to_json_value(record["target_identity"]) != to_json_value(
        dict(replay_context.target_identity)
    ):
        message = "plan replay target identity is stale"
        raise StaleRecordError(message)

    if not record["target_identity"] or not replay_context.target_identity:
        message = "plan replay target identity is missing"
        raise RecordFormatError(message)


def _validate_replay_family_identities(
    label: str,
    record_identities: Mapping[str, Any],
    context_identities: Mapping[str, Mapping[str, Any]],
    selected_families: tuple[str, ...],
) -> None:
    current = {
        family: dict(identity) for family, identity in context_identities.items()
    }

    if to_json_value(record_identities) != to_json_value(current):
        message = f"plan replay {label} identities are stale"
        raise StaleRecordError(message)

    identities = {
        family: dict(identity) for family, identity in dict(record_identities).items()
    }

    if set(identities) != set(selected_families) or any(
        not identity for identity in identities.values()
    ):
        message = f"plan replay {label} identities are missing"
        raise RecordFormatError(message)


def _replay_select_family(
    records: Sequence[tuple[Candidate, FullSizeRecord]],
    *,
    input_signature: Mapping[str, Any],
    policy: SelectionPolicy,
) -> tuple[Candidate, FullSizeRecord]:
    accepted = tuple(
        (candidate, record)
        for candidate, record in records
        if record.status == "passed"
        and record.reference_passed
        and to_json_value(record.input_signature) == to_json_value(input_signature)
        and full_size_agreement_satisfied(record)
        and memory_stable(record)
    )

    if not accepted:
        message = "plan replay family has no accepted rows"
        raise RecordFormatError(message)

    autobatch_selected = _replay_autobatch_selected(
        tuple(record for _, record in records),
        tuple(record for _, record in accepted),
    )

    if autobatch_selected is not None:
        for candidate, record in accepted:
            if to_json_value(record.row_key()) == to_json_value(
                autobatch_selected.row_key()
            ):
                return candidate, record

    return select_accepted_family(accepted, policy=policy)


def _replay_autobatch_selected(
    records: Sequence[FullSizeRecord],
    accepted: Sequence[FullSizeRecord],
) -> FullSizeRecord | None:
    selected = tuple(record for record in records if _is_autobatch_selected(record))

    if not selected:
        return None

    if len(selected) != 1:
        message = "plan replay has multiple Autobatch-selected rows"
        raise RecordFormatError(message)

    accepted_keys = {canonical_json(record.row_key()) for record in accepted}

    if canonical_json(selected[0].row_key()) not in accepted_keys:
        message = "plan replay Autobatch-selected row is stale"
        raise StaleRecordError(message)

    return selected[0]


def _is_autobatch_selected(record: FullSizeRecord) -> bool:
    metadata = dict(record.selection_metadata)

    return metadata.get("source") == "autobatch" and metadata.get("selected") is True


def _selected_records(
    selected: Mapping[str, Candidate],
    selected_record_keys: Mapping[str, str],
    full_size_by_key: Mapping[str, FullSizeRecord],
) -> dict[str, FullSizeRecord]:
    records = {
        family: full_size_by_key[selected_record_keys[family]] for family in selected
    }

    for family, candidate in selected.items():
        _validate_selected_record(family, candidate, records[family])

    return records


def _validate_selected_record(
    family: str,
    candidate: Candidate,
    selected_record: FullSizeRecord,
) -> None:
    if selected_record.family != family:
        message = f"plan replay selected record family differs: {family}"
        raise RecordFormatError(message)

    if selected_record.candidate_id != candidate.candidate_id:
        message = f"plan replay selected record candidate differs: {family}"
        raise RecordFormatError(message)

    if selected_record.generator_id != candidate.generator_id:
        message = f"plan replay selected record generator differs: {family}"
        raise StaleRecordError(message)

    if selected_record.generator_version != candidate.generator_version:
        message = f"plan replay selected record generator version differs: {family}"
        raise StaleRecordError(message)

    if (
        selected_record.status != "passed"
        or not selected_record.reference_passed
        or not full_size_agreement_satisfied(selected_record)
    ):
        message = f"plan replay selected record did not pass: {family}"
        raise RecordFormatError(message)

    if to_json_value(selected_record.candidate_settings) != to_json_value(
        candidate.settings
    ):
        message = f"plan replay selected record settings differ: {family}"
        raise RecordFormatError(message)

    if to_json_value(selected_record.dependency_identities) != to_json_value(
        candidate.dependency_identities
    ):
        message = f"plan replay selected record dependencies differ: {family}"
        raise StaleRecordError(message)

    if to_json_value(selected_record.cohort_assignment) != to_json_value(
        candidate.cohort_assignment
    ):
        message = f"plan replay selected record cohort differs: {family}"
        raise StaleRecordError(message)


def _dependency_identity(
    family: str,
    candidate: Candidate,
    record: FullSizeRecord,
    materializer_identity: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "family": family,
        "candidate_id": candidate.candidate_id,
        "candidate_settings": dict(candidate.settings),
        "full_size_row": record.row_key(),
        "materializer_identity": dict(materializer_identity),
    }


def _validate_dependency_identities(
    selected: Mapping[str, Candidate],
    selected_records: Mapping[str, FullSizeRecord],
    dependencies_by_family: Mapping[str, tuple[str, ...]],
    materializers: Mapping[str, Materializer],
) -> None:
    if set(dependencies_by_family) != set(selected):
        message = "plan replay dependency families differ"
        raise RecordFormatError(message)

    for family, candidate in selected.items():
        dependencies = dependencies_by_family[family]

        if set(candidate.dependency_identities) != set(dependencies):
            message = f"plan replay selected dependencies differ: {family}"
            raise StaleRecordError(message)

        if set(selected_records[family].dependency_identities) != set(dependencies):
            message = f"plan replay record dependencies differ: {family}"
            raise StaleRecordError(message)

        for dependency in dependencies:
            if dependency not in selected:
                message = f"plan replay dependency is missing: {dependency}"
                raise RecordFormatError(message)

            expected_identity = _dependency_identity(
                dependency,
                selected[dependency],
                selected_records[dependency],
                materializers[dependency].identity(),
            )

            if to_json_value(candidate.dependency_identities[dependency]) != (
                to_json_value(expected_identity)
            ):
                message = f"plan replay dependency identity is stale: {family}"
                raise StaleRecordError(message)

            if to_json_value(
                selected_records[family].dependency_identities[dependency]
            ) != (to_json_value(expected_identity)):
                message = f"plan replay record dependency identity is stale: {family}"
                raise StaleRecordError(message)


def _candidate_record_key(candidate: Candidate) -> str:
    return canonical_json({
        "family": candidate.family,
        "candidate_id": candidate.candidate_id,
        "settings": dict(candidate.settings),
        "dependency_identities": {
            family: dict(identity)
            for family, identity in sorted(candidate.dependency_identities.items())
        },
        "cohort_assignment": dict(candidate.cohort_assignment),
        "generator_id": candidate.generator_id,
        "generator_version": candidate.generator_version,
    })


def _result_candidate_key(record: FullSizeRecord | CheckRecord) -> str:
    return canonical_json({
        "family": record.family,
        "candidate_id": record.candidate_id,
        "settings": dict(record.candidate_settings),
        "dependency_identities": {
            family: dict(identity)
            for family, identity in sorted(record.dependency_identities.items())
        },
        "cohort_assignment": dict(record.cohort_assignment),
        "generator_id": record.generator_id,
        "generator_version": record.generator_version,
    })


def _candidate_records_by_key(
    candidate_records: Sequence[Mapping[str, Any]],
) -> dict[str, tuple[Candidate, Mapping[str, Any]]]:
    records = {}

    for record in candidate_records:
        candidate = candidate_record_from_json(record)
        key = _candidate_record_key(candidate)

        if key in records:
            message = f"plan replay has duplicate candidate row: {key}"
            raise RecordFormatError(message)

        records[key] = (candidate, record)

    return records


def _validate_candidate_records(
    selected: Mapping[str, Candidate],
    selected_records: Mapping[str, FullSizeRecord],
    full_size_records: Sequence[FullSizeRecord],
    check_records: Sequence[CheckRecord],
    candidate_records: Sequence[Mapping[str, Any]],
    summary_candidate_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Candidate]:
    by_key = _candidate_records_by_key(candidate_records)
    expected_by_key = {
        _candidate_record_key(candidate_from_signature(candidate_record)): (
            candidate_record
        )
        for candidate_record in summary_candidate_rows
    }
    full_size_by_key = {
        _result_candidate_key(record): record for record in full_size_records
    }
    checks_by_key = {}

    for record in check_records:
        checks_by_key.setdefault(_result_candidate_key(record), []).append(record)

    check_keys = {_result_candidate_key(record) for record in check_records}

    if len(full_size_by_key) != len(full_size_records):
        message = "plan replay has duplicate full-size candidate identities"
        raise RecordFormatError(message)

    if set(by_key) != set(expected_by_key):
        message = "plan replay candidate rows differ from plan summary"
        raise RecordFormatError(message)

    if not set(full_size_by_key) | check_keys <= set(expected_by_key):
        message = "plan replay candidate rows differ from result rows"
        raise RecordFormatError(message)

    for key, (row_candidate, row) in by_key.items():
        full_size_record = full_size_by_key.get(key)

        if full_size_record is None:
            if key not in checks_by_key:
                continue

            _validate_candidate_against_checks(
                row_candidate,
                row,
                tuple(checks_by_key[key]),
            )
            continue

        _validate_candidate_against_full_size(row_candidate, row, full_size_record)

    for family, candidate in selected.items():
        row_candidate, row = by_key[_candidate_record_key(candidate)]

        if to_json_value(row_candidate.signature()) != to_json_value(
            candidate.signature()
        ):
            message = f"plan replay candidate row differs: {family}"
            raise StaleRecordError(message)

        if _candidate_record_key(row_candidate) != _result_candidate_key(
            selected_records[family]
        ):
            message = f"plan replay selected candidate row is stale: {family}"
            raise StaleRecordError(message)

    return {key: candidate for key, (candidate, _) in by_key.items()}


def _validate_candidate_against_checks(
    candidate: Candidate,
    row: Mapping[str, Any],
    check_records: tuple[CheckRecord, ...],
) -> None:
    if any(_candidate_matches_check(candidate, row, check) for check in check_records):
        return

    message = (
        f"plan replay candidate row does not match reference rows: {candidate.family}"
    )
    raise StaleRecordError(message)


def _candidate_matches_check(
    candidate: Candidate,
    row: Mapping[str, Any],
    check: CheckRecord,
) -> bool:
    return (
        check.status == "passed"
        and to_json_value(row["input_signature"])
        == to_json_value(check.input_signature)
        and to_json_value(candidate.settings) == to_json_value(check.candidate_settings)
        and to_json_value(candidate.dependency_identities)
        == to_json_value(check.dependency_identities)
        and to_json_value(candidate.cohort_assignment)
        == to_json_value(check.cohort_assignment)
        and candidate.generator_id == check.generator_id
        and candidate.generator_version == check.generator_version
    )


def _validate_candidate_against_full_size(
    candidate: Candidate,
    row: Mapping[str, Any],
    full_size_record: FullSizeRecord,
) -> None:
    if to_json_value(candidate.settings) != to_json_value(
        full_size_record.candidate_settings
    ):
        message = f"plan replay candidate settings differ: {candidate.family}"
        raise StaleRecordError(message)

    if to_json_value(candidate.dependency_identities) != to_json_value(
        full_size_record.dependency_identities
    ):
        message = f"plan replay candidate dependencies differ: {candidate.family}"
        raise StaleRecordError(message)

    if candidate.generator_id != full_size_record.generator_id:
        message = f"plan replay candidate generator differs: {candidate.family}"
        raise StaleRecordError(message)

    if candidate.generator_version != full_size_record.generator_version:
        message = f"plan replay candidate generator version differs: {candidate.family}"
        raise StaleRecordError(message)

    if to_json_value(candidate.cohort_assignment) != to_json_value(
        full_size_record.cohort_assignment
    ):
        message = f"plan replay candidate cohort differs: {candidate.family}"
        raise StaleRecordError(message)

    if to_json_value(row["input_signature"]) != to_json_value(
        full_size_record.input_signature
    ):
        message = f"plan replay candidate input signature is stale: {candidate.family}"
        raise StaleRecordError(message)


def _assignment_records(
    assignment: CohortAssignment | None,
    constraints: tuple[CohortConstraint, ...],
    family_names: tuple[str, ...],
    ordered_full_size: Sequence[FullSizeRecord],
    candidates_by_key: Mapping[str, Candidate],
    replay_context: ReplayContext,
) -> tuple[tuple[Candidate, FullSizeRecord], ...]:
    pairs = []

    for record in ordered_full_size:
        expected_signature = replay_context.family_input_signatures[record.family]

        if to_json_value(record.input_signature) != to_json_value(expected_signature):
            continue

        candidate = candidates_by_key[_result_candidate_key(record)]

        if assignment is not None and not candidate_matches_assignment(
            candidate,
            assignment,
            constraints,
            family_names,
        ):
            continue

        if (
            record.status == "passed"
            and record.reference_passed
            and full_size_agreement_satisfied(record)
            and memory_stable(record)
        ):
            pairs.append((candidate, record))

    return tuple(pairs)


def _replay_cohort_dependency_identities_match(
    cohort: Mapping[str, tuple[Candidate, FullSizeRecord]],
    dependencies_by_family: Mapping[str, tuple[str, ...]],
    materializers: Mapping[str, Materializer],
) -> bool:
    for family, (candidate, record) in cohort.items():
        dependencies = dependencies_by_family[family]

        if set(candidate.dependency_identities) != set(dependencies):
            return False

        if set(record.dependency_identities) != set(dependencies):
            return False

        for dependency in dependencies:
            selected_dependency = cohort.get(dependency)

            if selected_dependency is None:
                return False

            expected_identity = _dependency_identity(
                dependency,
                selected_dependency[0],
                selected_dependency[1],
                materializers[dependency].identity(),
            )

            if to_json_value(candidate.dependency_identities[dependency]) != (
                to_json_value(expected_identity)
            ):
                return False

            if to_json_value(record.dependency_identities[dependency]) != (
                to_json_value(expected_identity)
            ):
                return False

    return True


def _replay_family_order(
    family_names: tuple[str, ...],
    dependencies_by_family: Mapping[str, tuple[str, ...]],
) -> tuple[str, ...]:
    remaining = list(family_names)
    ordered = []

    while remaining:
        progressed = False

        for family in tuple(remaining):
            dependencies = dependencies_by_family.get(family)

            if dependencies is None:
                message = f"plan replay dependency record missing: {family}"
                raise RecordFormatError(message)

            if all(dependency in ordered for dependency in dependencies):
                ordered.append(family)
                remaining.remove(family)
                progressed = True

        if not progressed:
            message = "plan replay dependency graph has a cycle or missing dependency"
            raise RecordFormatError(message)

    return tuple(ordered)


def _replay_row_matches_selected_dependencies(
    candidate: Candidate,
    record: FullSizeRecord,
    cohort: Mapping[str, tuple[Candidate, FullSizeRecord]],
    dependencies_by_family: Mapping[str, tuple[str, ...]],
    materializers: Mapping[str, Materializer],
) -> bool:
    dependencies = dependencies_by_family[candidate.family]

    if set(candidate.dependency_identities) != set(dependencies):
        return False

    if set(record.dependency_identities) != set(dependencies):
        return False

    for dependency in dependencies:
        selected_dependency = cohort.get(dependency)

        if selected_dependency is None:
            return False

        expected_identity = _dependency_identity(
            dependency,
            selected_dependency[0],
            selected_dependency[1],
            materializers[dependency].identity(),
        )

        if to_json_value(candidate.dependency_identities[dependency]) != to_json_value(
            expected_identity
        ):
            return False

        if to_json_value(record.dependency_identities[dependency]) != to_json_value(
            expected_identity
        ):
            return False

    return True


def _validate_recomputed_cohort_selection(
    selected_records: Mapping[str, FullSizeRecord],
    ordered_full_size: Sequence[FullSizeRecord],
    replay_context: ReplayContext,
    dependencies_by_family: Mapping[str, tuple[str, ...]],
    materializers: Mapping[str, Materializer],
    candidates_by_key: Mapping[str, Candidate],
    cohort_assignment: CohortAssignment | None,
    cohort_constraints: tuple[CohortConstraint, ...],
) -> None:
    if not selected_records:
        return

    family_names = tuple(selected_records)
    ordered_families = _replay_family_order(family_names, dependencies_by_family)
    cohorts = []
    state_by_cohort = {}

    for assignment in cohort_assignments(cohort_constraints, family_names):
        pairs = _assignment_records(
            assignment,
            cohort_constraints,
            family_names,
            ordered_full_size,
            candidates_by_key,
            replay_context,
        )
        cohort = {}

        for family in ordered_families:
            records = tuple(
                (candidate, record)
                for candidate, record in pairs
                if candidate.family == family
                and _replay_row_matches_selected_dependencies(
                    candidate,
                    record,
                    cohort,
                    dependencies_by_family,
                    materializers,
                )
            )

            if not records:
                break

            candidate, selected_record = _replay_select_family(
                records,
                input_signature=replay_context.family_input_signatures[family],
                policy=replay_context.selection_policy,
            )
            cohort[family] = (candidate, selected_record)

        if set(cohort) == set(
            family_names
        ) and _replay_cohort_dependency_identities_match(
            cohort,
            dependencies_by_family,
            materializers,
        ):
            cohorts.append(cohort)
            state_by_cohort[id(cohort)] = assignment

    selected_cohort = select_complete_cohort(
        cohorts,
        families=family_names,
        policy=replay_context.selection_policy,
    )
    selected_assignment = state_by_cohort[id(selected_cohort)]

    if cohort_constraints and cohort_assignment is None:
        message = "plan replay selected cohort assignment is missing"
        raise StaleRecordError(message)

    if cohort_assignment is not None and to_json_value(
        selected_assignment.signature()
    ) != to_json_value(cohort_assignment.signature()):
        message = "plan replay selected cohort assignment is stale"
        raise StaleRecordError(message)

    for family, (_, record) in selected_cohort.items():
        if to_json_value(record.row_key()) != to_json_value(
            selected_records[family].row_key()
        ):
            message = f"plan replay selected cohort row is stale: {family}"
            raise StaleRecordError(message)


def selected_candidates_from_json(record: Mapping[str, Any]) -> dict[str, Candidate]:
    """Return selected candidates from a saved plan summary.

    Raises:
        RecordFormatError: If selected-candidate fields are malformed.
    """
    candidates = {}

    for family, candidate_record in _selected_candidate_records(record).items():
        if not isinstance(candidate_record, Mapping):
            message = f"plan replay selected candidate is invalid: {family}"
            raise RecordFormatError(message)

        candidates[str(family)] = candidate_from_signature(candidate_record)

    return candidates


def _candidate_rows_from_record(record: Mapping[str, Any]) -> tuple[Candidate, ...]:
    return tuple(
        candidate_from_signature(candidate_record)
        for candidate_record in _candidate_row_records_from_json(record)
    )


def _candidate_row_records_from_json(
    record: Mapping[str, Any],
) -> tuple[Mapping[str, Any], ...]:
    try:
        rows = record["candidate_rows"]
    except KeyError as error:
        message = "plan summary candidate_rows field is missing"
        raise RecordFormatError(message) from error

    if isinstance(rows, str) or not isinstance(rows, Sequence):
        message = "plan summary candidate_rows must be a sequence"
        raise RecordFormatError(message)

    candidate_rows = []

    for candidate_record in rows:
        if not isinstance(candidate_record, Mapping):
            message = "plan summary candidate_rows must contain mappings"
            raise RecordFormatError(message)

        candidate_rows.append(candidate_record)

    return tuple(candidate_rows)


def _replay_rows(
    record: Mapping[str, Any],
    full_size_records: Sequence[FullSizeRecord],
    check_records: Sequence[CheckRecord],
    validation_records: Sequence[CheckRecord],
) -> _ReplayRows:
    full_size_by_key = _record_by_row_key(full_size_records, "full-size")
    check_by_key = _check_record_by_lookup_key(check_records, "reference")
    ordered_full_size = _records_in_saved_order(
        _record_sequence_field(
            record,
            "full_size_records",
            "plan summary full-size records",
        ),
        full_size_by_key,
        "full-size",
    )
    ordered_checks = _check_records_in_saved_order(
        _record_sequence_field(
            record,
            "check_records",
            "plan summary reference records",
        ),
        check_by_key,
        "reference",
    )
    ordered_validation = _plan_validation_records(record, validation_records)

    _validate_ordered_replay_rows(
        ordered_full_size,
        ordered_checks,
    )

    return _ReplayRows(
        full_size_by_key=full_size_by_key,
        ordered_full_size=ordered_full_size,
        ordered_checks=ordered_checks,
        ordered_validation=ordered_validation,
    )


def _dependencies_from_record(record: Mapping[str, Any]) -> dict[str, tuple[str, ...]]:
    records = _record_mapping_field(
        record,
        "dependencies_by_family",
        "plan summary dependencies",
    )

    try:
        return {
            str(family): tuple(str(dependency) for dependency in dependencies)
            for family, dependencies in records.items()
        }
    except TypeError as error:
        message = f"plan replay dependencies are invalid: {error}"
        raise RecordFormatError(message) from error


def _replayed_plan(
    record: Mapping[str, Any],
    replay_context: ReplayContext,
    selected: Mapping[str, Candidate],
    selected_records: Mapping[str, FullSizeRecord],
    rows: _ReplayRows,
    materializers: Mapping[str, Materializer],
    candidate_rows: tuple[Candidate, ...],
    validation_identity: _ValidationReplayIdentity,
    dependencies_by_family: Mapping[str, tuple[str, ...]],
    cohort_assignment: CohortAssignment | None,
    cohort_constraints: tuple[CohortConstraint, ...],
    run_dir: Path | None,
) -> Plan:
    return Plan(
        selected=dict(selected),
        records=dict(selected_records),
        input_signature=dict(replay_context.input_signature),
        policy=replay_context.selection_policy,
        candidate_rows=candidate_rows,
        full_size_records=rows.ordered_full_size,
        check_records=rows.ordered_checks,
        validation_records=rows.ordered_validation,
        materializers={family: materializers[family] for family in selected},
        validation_order=plan_validation_order_from_json(record),
        dependencies_by_family=dict(dependencies_by_family),
        cohort_assignment=cohort_assignment,
        cohort_constraints=cohort_constraints,
        target_identity=dict(replay_context.target_identity),
        runtime_identities={
            family: dict(identity)
            for family, identity in replay_context.runtime_identities.items()
        },
        adapter_identities={
            family: dict(identity)
            for family, identity in replay_context.adapter_identities.items()
        },
        validation_required=validation_identity.required,
        validator_identities=dict(validation_identity.validator_identities),
        run_dir=run_dir,
    )


def _validate_replay_validation_summary(
    plan: Plan,
    replay_context: ReplayContext,
    validation_summary: Mapping[str, Any] | None,
    validation_records: Sequence[CheckRecord],
) -> None:
    if validation_records and validation_summary is None:
        message = "plan replay validation records require a validation summary"
        raise RecordFormatError(message)

    if (replay_context.validation_required or plan.validation_required) and (
        validation_summary is None
    ):
        message = "plan replay requires selected-plan validation"
        raise RecordFormatError(message)

    if validation_summary is not None and not validation_records:
        message = "plan replay validation summary requires validation records"
        raise RecordFormatError(message)

    if validation_summary is None:
        return

    validate_json_record(validation_summary)
    _record_sequence_field(
        validation_summary,
        "records",
        "selected-plan validation summary records",
    )

    if not selected_plan_validation_summary_current(
        validation_summary,
        plan,
        validation_records,
    ):
        message = "plan replay selected-plan validation summary is stale"
        raise StaleRecordError(message)

    ordered_validation = _summary_validation_records(
        validation_summary,
        validation_records,
    )
    _validate_selected_plan_validation_rows(
        plan,
        validation_summary,
        ordered_validation,
    )


def _validate_replay_validation_order(
    record: Mapping[str, Any],
    replay_context: ReplayContext,
) -> None:
    if replay_context.validation_order and plan_validation_order_from_json(
        record
    ) != tuple(replay_context.validation_order):
        message = "plan replay validation order is stale"
        raise StaleRecordError(message)


def plan_from_json(
    record: Mapping[str, Any],
    *,
    replay_context: ReplayContext,
    full_size_records: Sequence[FullSizeRecord],
    check_records: Sequence[CheckRecord],
    candidate_records: Sequence[Mapping[str, Any]],
    materializers: Mapping[str, Materializer],
    validation_summary: Mapping[str, Any] | None = None,
    validation_records: Sequence[CheckRecord] = (),
    run_dir: Path | None = None,
) -> Plan:
    """Return a selected plan from saved JSON rows.

    Raises:
        StaleRecordError: If the summary does not match the rebuilt plan.
        RecordFormatError: If the record is not a summary or required rows are missing.
    """
    validate_json_record(record)

    if record["record_type"] != "summary":
        message = "plan replay requires a summary record"
        raise RecordFormatError(message)

    selected = selected_candidates_from_json(record)
    rows = _replay_rows(record, full_size_records, check_records, validation_records)
    _validate_replay_context(
        record,
        replay_context,
        materializers,
        rows.ordered_full_size,
        rows.ordered_checks,
    )
    _validate_materializers(selected, materializers)
    selected_records = _selected_records(
        selected,
        _selected_record_keys(
            record,
            selected,
            rows.full_size_by_key,
        ),
        rows.full_size_by_key,
    )
    candidates_by_key = _validate_candidate_records(
        selected,
        selected_records,
        rows.ordered_full_size,
        rows.ordered_checks,
        candidate_records,
        _candidate_row_records_from_json(record),
    )
    candidate_rows = tuple(
        candidates_by_key[_candidate_record_key(candidate)]
        for candidate in _candidate_rows_from_record(record)
    )
    dependencies_by_family = _dependencies_from_record(record)
    validation_identity = _validation_replay_identity(record, replay_context)
    cohort_assignment = cohort_assignment_from_json(record["cohort_assignment"])
    cohort_constraints = _cohort_constraints_from_json(record["cohort_constraints"])
    _validate_dependency_identities(
        selected,
        selected_records,
        dependencies_by_family,
        materializers,
    )
    _validate_recomputed_cohort_selection(
        selected_records,
        rows.ordered_full_size,
        replay_context,
        dependencies_by_family,
        materializers,
        candidates_by_key,
        cohort_assignment,
        cohort_constraints,
    )

    plan = _replayed_plan(
        record,
        replay_context,
        selected,
        selected_records,
        rows,
        materializers,
        candidate_rows,
        validation_identity,
        dependencies_by_family,
        cohort_assignment,
        cohort_constraints,
        run_dir,
    )

    if not plan_record_current(record, plan):
        message = "plan replay summary is stale"
        raise StaleRecordError(message)

    _validate_replay_validation_order(record, replay_context)
    _validate_replay_validation_summary(
        plan,
        replay_context,
        validation_summary,
        validation_records,
    )

    return plan


def _plan_validation_records(
    record: Mapping[str, Any],
    validation_records: Sequence[CheckRecord],
) -> tuple[CheckRecord, ...]:
    row_keys = _record_sequence_field(
        record,
        "validation_records",
        "plan summary selected-plan validation records",
    )

    if not row_keys:
        return ()

    return _check_records_in_saved_order(
        row_keys,
        _check_record_by_lookup_key(validation_records, "selected-plan validation"),
        "selected-plan validation",
    )


def _summary_validation_records(
    summary: Mapping[str, Any],
    validation_records: Sequence[CheckRecord],
) -> tuple[CheckRecord, ...]:
    return _check_records_in_saved_order(
        _record_sequence_field(
            summary,
            "records",
            "selected-plan validation summary records",
        ),
        _check_record_by_lookup_key(validation_records, "selected-plan validation"),
        "selected-plan validation",
    )
