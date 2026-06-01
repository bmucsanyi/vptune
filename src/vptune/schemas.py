"""Schema validation for saved records."""

import dataclasses
import itertools
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

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
from vptune.errors import StaleRecordError, VPTuneError
from vptune.identities import (
    owner_hash,
    record_content_hash,
    stable_hash,
    to_json_value,
)

REQUIRED_COMMON_FIELDS = (
    "record_type",
    "schema_version",
    "package_version",
    "owner_hash",
    "input_signature",
    "candidate_settings",
    "status",
    "generator_id",
    "generator_version",
)


@dataclasses.dataclass(frozen=True, slots=True)
class _ValidationReplayIdentity:
    required: bool
    validator_identities: Mapping[str, Mapping[str, Any]]


def validate_json_record(record: Mapping[str, Any]) -> None:
    """Validate common saved-record fields.

    Raises:
        StaleRecordError: If schema or package versions changed.
        VPTuneError: If required fields are missing or record type is invalid.
    """
    for field in REQUIRED_COMMON_FIELDS:
        if field not in record:
            message = f"record field is missing: {field}"
            raise VPTuneError(message)

    if record["schema_version"] != SCHEMA_VERSION:
        message = "record schema version changed"
        raise StaleRecordError(message)

    if record["package_version"] != PACKAGE_VERSION:
        message = "record package version changed"
        raise StaleRecordError(message)

    if record["record_type"] not in {"candidate", "reference", "full_size", "summary"}:
        message = f"record type is invalid: {record['record_type']}"
        raise VPTuneError(message)


def check_owner_payload(
    *,
    record_type: str,
    family: str,
    candidate_id: str,
    check_name: str = "",
    input_signature: Mapping[str, Any],
    candidate_settings: Mapping[str, Any],
    candidate_spec_hash: str = "",
    thresholds: Mapping[str, Any] | None = None,
    dependency_identities: Mapping[str, Mapping[str, Any]] | None = None,
    changed_axes: Sequence[str] = (),
    generator_id: str,
    generator_version: str,
) -> dict[str, Any]:
    """Return the owner-hash payload shared by check and full-size rows."""
    return {
        "record_type": record_type,
        "family": family,
        "candidate_id": candidate_id,
        "check_name": check_name,
        "input_signature": dict(input_signature),
        "candidate_settings": dict(candidate_settings),
        "candidate_spec_hash": candidate_spec_hash,
        "thresholds": {} if thresholds is None else dict(thresholds),
        "dependency_identities": _dependency_identity_record(dependency_identities),
        "changed_axes": tuple(changed_axes),
        "generator_id": generator_id,
        "generator_version": generator_version,
    }


def compute_record_owner_hash(
    *,
    record_type: str,
    family: str,
    candidate_id: str,
    check_name: str = "",
    input_signature: Mapping[str, Any],
    candidate_settings: Mapping[str, Any],
    candidate_spec_hash: str = "",
    thresholds: Mapping[str, Any] | None = None,
    dependency_identities: Mapping[str, Mapping[str, Any]] | None = None,
    changed_axes: Sequence[str] = (),
    generator_id: str,
    generator_version: str,
) -> str:
    """Return the owner hash for a saved row."""
    payload = check_owner_payload(
        record_type=record_type,
        family=family,
        candidate_id=candidate_id,
        check_name=check_name,
        input_signature=input_signature,
        candidate_settings=candidate_settings,
        candidate_spec_hash=candidate_spec_hash,
        thresholds=thresholds,
        dependency_identities=dependency_identities,
        changed_axes=changed_axes,
        generator_id=generator_id,
        generator_version=generator_version,
    )

    return owner_hash(record_type, payload)


def record_current(
    record: Mapping[str, Any],
    *,
    record_type: str,
    family: str,
    candidate_id: str,
    check_name: str = "",
    input_signature: Mapping[str, Any],
    candidate_settings: Mapping[str, Any],
    candidate_spec_hash: str = "",
    thresholds: Mapping[str, Any] | None = None,
    dependency_identities: Mapping[str, Mapping[str, Any]] | None = None,
    changed_axes: Sequence[str] = (),
    generator_id: str,
    generator_version: str,
) -> bool:
    """Return whether a saved row matches current identity."""
    try:
        validate_json_record(record)
    except VPTuneError:
        return False

    expected = compute_record_owner_hash(
        record_type=record_type,
        family=family,
        candidate_id=candidate_id,
        check_name=check_name,
        input_signature=input_signature,
        candidate_settings=candidate_settings,
        candidate_spec_hash=candidate_spec_hash,
        thresholds=thresholds,
        dependency_identities=dependency_identities,
        changed_axes=changed_axes,
        generator_id=generator_id,
        generator_version=generator_version,
    )

    return (
        record.get("record_type") == record_type
        and record.get("family") == family
        and record.get("candidate_id") == candidate_id
        and (record_type != "reference" or record.get("name") == check_name)
        and (
            record_type != "candidate"
            or to_json_value(record.get("changed_axes"))
            == to_json_value(tuple(changed_axes))
        )
        and to_json_value(record.get("input_signature"))
        == to_json_value(dict(input_signature))
        and to_json_value(record.get("candidate_settings"))
        == to_json_value(dict(candidate_settings))
        and str(record.get("candidate_spec_hash", "")) == candidate_spec_hash
        and to_json_value(record.get("dependency_identities"))
        == to_json_value(_dependency_identity_record(dependency_identities))
        and record.get("generator_id") == generator_id
        and record.get("generator_version") == generator_version
        and record.get("owner_hash") == expected
    )


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
    row_hash = compute_record_owner_hash(
        record_type="candidate",
        family=candidate.family,
        candidate_id=candidate.candidate_id,
        input_signature=input_signature,
        candidate_settings=candidate.settings,
        candidate_spec_hash=candidate.candidate_spec_hash(),
        dependency_identities=candidate.dependency_identities,
        changed_axes=candidate.changed_axes,
        generator_id=candidate.generator_id,
        generator_version=candidate.generator_version,
    )

    payload = {
        "record_type": "candidate",
        "schema_version": SCHEMA_VERSION,
        "package_version": PACKAGE_VERSION,
        "owner_hash": row_hash,
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
        "candidate_spec_hash": candidate.candidate_spec_hash(),
    }
    payload["content_hash"] = record_content_hash(payload)

    return payload


def candidate_from_signature(record: Mapping[str, Any]) -> Candidate:
    """Return a candidate from a saved candidate signature.

    Raises:
        StaleRecordError: If the selected candidate spec hash differs.
    """
    candidate = Candidate(
        family=str(record["family"]),
        candidate_id=str(record["candidate_id"]),
        settings=dict(record["settings"]),
        changed_axes=tuple(str(axis) for axis in record["changed_axes"]),
        dependency_identities={
            str(key): dict(value)
            for key, value in dict(record["dependency_identities"]).items()
        },
        cohort_assignment=dict(record["cohort_assignment"]),
        admission_status=str(record["admission_status"]),
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

    if str(record["candidate_spec_hash"]) != candidate.candidate_spec_hash():
        message = "selected candidate spec hash is stale"
        raise StaleRecordError(message)

    return candidate


def candidate_record_from_json(record: Mapping[str, Any]) -> Candidate:
    """Return a candidate from a saved candidate row.

    Raises:
        StaleRecordError: If the saved candidate owner hash is stale.
        VPTuneError: If the row is not a candidate record.
    """
    validate_json_record(record)

    if record["record_type"] != "candidate":
        message = "candidate replay requires a candidate record"
        raise VPTuneError(message)

    _require_content_hash(record, "candidate")

    candidate = Candidate(
        family=str(record["family"]),
        candidate_id=str(record["candidate_id"]),
        settings=dict(record["candidate_settings"]),
        changed_axes=tuple(str(axis) for axis in record["changed_axes"]),
        dependency_identities={
            str(key): dict(value)
            for key, value in dict(record["dependency_identities"]).items()
        },
        cohort_assignment=dict(record["cohort_assignment"]),
        admission_status=str(record["status"]),
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

    if str(record["candidate_spec_hash"]) != candidate.candidate_spec_hash():
        message = "candidate spec hash is stale"
        raise StaleRecordError(message)

    if not record_current(
        record,
        record_type="candidate",
        family=candidate.family,
        candidate_id=candidate.candidate_id,
        input_signature=record["input_signature"],
        candidate_settings=candidate.settings,
        candidate_spec_hash=candidate.candidate_spec_hash(),
        dependency_identities=candidate.dependency_identities,
        changed_axes=candidate.changed_axes,
        generator_id=candidate.generator_id,
        generator_version=candidate.generator_version,
    ):
        message = "candidate record owner hash is stale"
        raise StaleRecordError(message)

    payload = dict(record)
    content_hash = str(payload.pop("content_hash"))

    if content_hash != record_content_hash(payload):
        message = "candidate record content hash is stale"
        raise StaleRecordError(message)

    return candidate


def check_record_to_json(record: CheckRecord) -> dict[str, Any]:
    """Return a JSON row for a reference check."""
    payload = dataclasses.asdict(record)
    payload["record_type"] = "reference"
    payload["content_hash"] = record_content_hash(payload)

    return payload


def check_record_from_json(record: Mapping[str, Any]) -> CheckRecord:
    """Return a check record from a JSON row.

    Raises:
        StaleRecordError: If the owner hash does not match the row.
        VPTuneError: If the row is not a reference record.
    """
    validate_json_record(record)

    if record["record_type"] != "reference":
        message = "check replay requires a reference record"
        raise VPTuneError(message)

    payload = dict(record)
    payload.pop("record_type")
    _require_content_hash(payload, "reference")

    check_record = CheckRecord(**payload)

    if not check_record_current(check_record):
        message = "reference record owner hash is stale"
        raise StaleRecordError(message)

    if not check_record_content_current(check_record):
        message = "reference record content hash is stale"
        raise StaleRecordError(message)

    return check_record


def check_record_current(record: CheckRecord) -> bool:
    """Return whether a reference row owner hash matches its row identity."""
    expected = compute_record_owner_hash(
        record_type="reference",
        family=record.family,
        candidate_id=record.candidate_id,
        check_name=record.name,
        input_signature=record.input_signature,
        candidate_settings=record.candidate_settings,
        candidate_spec_hash=record.candidate_spec_hash,
        thresholds=record.thresholds,
        dependency_identities=record.dependency_identities,
        generator_id=record.generator_id,
        generator_version=record.generator_version,
    )

    return record.owner_hash == expected


def check_record_content_current(record: CheckRecord) -> bool:
    """Return whether a reference row content hash matches its content."""
    return record.content_hash == record.computed_content_hash()


def full_size_record_to_json(record: FullSizeRecord) -> dict[str, Any]:
    """Return a JSON row for a full-size result."""
    payload = dataclasses.asdict(record)
    payload["record_type"] = "full_size"
    payload["content_hash"] = record_content_hash(payload)

    return payload


def measurement_from_json(record: Mapping[str, Any]) -> Measurement:
    """Return a measurement from a JSON row."""
    return Measurement(**dict(record))


def full_size_record_from_json(record: Mapping[str, Any]) -> FullSizeRecord:
    """Return a full-size record from a JSON row.

    Raises:
        StaleRecordError: If the owner hash does not match the row.
    """
    validate_json_record(record)
    payload = dict(record)
    payload.pop("record_type")
    _require_content_hash(payload, "full-size")
    payload["timing_samples"] = tuple(
        measurement_from_json(sample) for sample in payload["timing_samples"]
    )
    payload["memory_samples"] = tuple(
        measurement_from_json(sample) for sample in payload["memory_samples"]
    )

    full_size_record = FullSizeRecord(**payload)

    if not full_size_record_current(full_size_record):
        message = "full-size record owner hash is stale"
        raise StaleRecordError(message)

    if not full_size_record_content_current(full_size_record):
        message = "full-size record content hash is stale"
        raise StaleRecordError(message)

    return full_size_record


def full_size_record_current(record: FullSizeRecord) -> bool:
    """Return whether a full-size row owner hash matches its row identity."""
    expected = compute_record_owner_hash(
        record_type="full_size",
        family=record.family,
        candidate_id=record.candidate_id,
        input_signature=record.input_signature,
        candidate_settings=record.candidate_settings,
        candidate_spec_hash=record.candidate_spec_hash,
        dependency_identities=record.dependency_identities,
        generator_id=record.generator_id,
        generator_version=record.generator_version,
    )

    return record.owner_hash == expected


def full_size_record_content_current(record: FullSizeRecord) -> bool:
    """Return whether a full-size row content hash matches its content."""
    return record.content_hash == record.computed_content_hash()


def plan_to_json(plan: Plan) -> dict[str, Any]:
    """Return a JSON row for a selected plan."""
    return plan.to_record()


def selected_plan_validation_summary_record(
    plan: Plan,
    records: Sequence[CheckRecord],
) -> dict[str, Any]:
    """Return the selected-plan validation summary row."""
    input_signature = {
        "plan": plan.owner_hash(),
        "input_signature": dict(plan.input_signature),
    }
    candidate_settings = {
        "validation_order": plan.validation_order,
        "records": tuple(record.owner_hash for record in records),
    }
    status = (
        "passed" if all(record.status == "passed" for record in records) else "failed"
    )

    return {
        "record_type": "summary",
        "schema_version": SCHEMA_VERSION,
        "package_version": PACKAGE_VERSION,
        "owner_hash": compute_record_owner_hash(
            record_type="summary",
            family="selected_plan_validation",
            candidate_id=plan.owner_hash(),
            input_signature=input_signature,
            candidate_settings=candidate_settings,
            generator_id="selected_plan_validation",
            generator_version=PACKAGE_VERSION,
        ),
        "input_signature": input_signature,
        "candidate_settings": candidate_settings,
        "status": status,
        "generator_id": "selected_plan_validation",
        "generator_version": PACKAGE_VERSION,
        "records": tuple(record.owner_hash for record in records),
    }


def selected_plan_validation_summary_current(
    record: Mapping[str, Any],
    plan: Plan,
    records: Sequence[CheckRecord],
) -> bool:
    """Return whether a selected-plan validation summary matches its rows."""
    try:
        validate_json_record(record)
    except VPTuneError:
        return False

    if record.get("record_type") != "summary":
        return False

    expected = selected_plan_validation_summary_record(plan, records)

    for field in (
        "record_type",
        "schema_version",
        "package_version",
        "owner_hash",
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
    expected_input_signature = {
        "plan": plan.owner_hash(),
        "input_signature": dict(plan.input_signature),
    }

    if tuple(record.family for record in records) != tuple(validation_order):
        message = "plan replay selected-plan validation order differs"
        raise VPTuneError(message)

    if set(validation_order) != set(plan.selected):
        message = "plan replay selected-plan validation families differ"
        raise VPTuneError(message)

    for record in records:
        _validate_selected_plan_validation_row(
            plan.selected[record.family],
            expected_input_signature,
            record,
        )

    if any(not check_record_current(record) for record in records):
        message = "plan replay has stale selected-plan validation rows"
        raise StaleRecordError(message)

    if any(not check_record_content_current(record) for record in records):
        message = "plan replay has changed selected-plan validation rows"
        raise StaleRecordError(message)

    if summary.get("status") != "passed" or any(
        record.status != "passed" for record in records
    ):
        message = "plan replay selected-plan validation did not pass"
        raise VPTuneError(message)


def _validate_selected_plan_validation_row(
    candidate: Candidate,
    expected_input_signature: Mapping[str, Any],
    record: CheckRecord,
) -> None:
    if record.name != "selected_plan_validation":
        message = "plan replay validation row name differs"
        raise VPTuneError(message)

    if to_json_value(record.input_signature) != to_json_value(expected_input_signature):
        message = "plan replay validation row input signature differs"
        raise StaleRecordError(message)

    if record.candidate_id != candidate.candidate_id:
        message = "plan replay validation row candidate differs"
        raise StaleRecordError(message)

    if record.candidate_spec_hash != candidate.candidate_spec_hash():
        message = "plan replay validation row candidate spec differs"
        raise StaleRecordError(message)

    if to_json_value(record.candidate_settings) != to_json_value(candidate.settings):
        message = "plan replay validation row settings differ"
        raise StaleRecordError(message)

    if to_json_value(record.dependency_identities) != to_json_value(
        candidate.dependency_identities
    ):
        message = "plan replay validation row dependencies differ"
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
    except VPTuneError:
        return False

    expected = plan.to_record()

    for field in (
        "record_type",
        "owner_hash",
        "input_signature",
        "candidate_settings",
        "selected",
        "records",
        "full_size_records",
        "full_size_record_content_hashes",
        "check_records",
        "check_record_content_hashes",
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


def _record_by_owner_hash(records: Sequence[Any], label: str) -> dict[str, Any]:
    by_hash = {record.owner_hash: record for record in records}

    if len(by_hash) != len(records):
        message = f"plan replay has duplicate {label} owner hashes"
        raise VPTuneError(message)

    return by_hash


def _records_in_saved_order(
    owner_hashes: Sequence[Any],
    records_by_hash: Mapping[str, Any],
    label: str,
) -> tuple[Any, ...]:
    ordered = []

    for value in owner_hashes:
        record_hash = str(value)
        record = records_by_hash.get(record_hash)

        if record is None:
            message = f"plan replay missing {label} row: {record_hash}"
            raise VPTuneError(message)

        ordered.append(record)

    if set(records_by_hash) != {str(value) for value in owner_hashes}:
        message = f"plan replay received extra {label} rows"
        raise VPTuneError(message)

    return tuple(ordered)


def _content_hashes_in_saved_order(
    content_hashes: Sequence[Any],
    records: Sequence[Any],
    label: str,
) -> None:
    if len(content_hashes) != len(records):
        message = f"plan replay {label} content-hash count differs"
        raise VPTuneError(message)

    for expected, row in zip(content_hashes, records, strict=True):
        if str(expected) != row.computed_content_hash():
            message = f"plan replay {label} row content changed: {row.owner_hash}"
            raise StaleRecordError(message)


def _require_content_hash(payload: Mapping[str, Any], label: str) -> None:
    if "content_hash" not in payload:
        message = f"{label} record content hash is missing"
        raise StaleRecordError(message)


def _selection_policy_from_json(record: Mapping[str, Any]) -> SelectionPolicy:
    policy = dict(record)
    expected = {field.name for field in dataclasses.fields(SelectionPolicy)}

    if set(policy) != expected:
        message = "plan replay selection policy fields differ"
        raise VPTuneError(message)

    return SelectionPolicy(**policy)


def _cohort_assignment_from_json(
    record: Mapping[str, Any] | None,
) -> CohortAssignment | None:
    if record is None:
        return None

    return CohortAssignment(
        assignment_id=str(record["assignment_id"]),
        values=dict(record["values"]),
        constraints=tuple(str(name) for name in record["constraints"]),
        covered_families=tuple(str(family) for family in record["covered_families"]),
    )


def _cohort_constraints_from_json(
    records: Sequence[Mapping[str, Any]],
) -> tuple[CohortConstraint, ...]:
    return tuple(
        CohortConstraint(
            name=str(record["name"]),
            settings_keys=tuple(str(key) for key in record["settings_keys"]),
            assignments=tuple(
                dict(assignment) for assignment in tuple(record["assignments"])
            ),
            families=tuple(str(family) for family in record["families"]),
            dependency_inheritance=str(record["dependency_inheritance"]),
            selection_aggregation=str(record["selection_aggregation"]),
        )
        for record in records
    )


def _validation_replay_identity(
    record: Mapping[str, Any],
    replay_context: ReplayContext,
) -> _ValidationReplayIdentity:
    return _ValidationReplayIdentity(
        required=bool(record["validation_required"]),
        validator_identities={
            str(family): dict(identity)
            for family, identity in replay_context.validator_identities.items()
        },
    )


def _validate_ordered_replay_rows(
    record: Mapping[str, Any],
    ordered_full_size: Sequence[FullSizeRecord],
    ordered_checks: Sequence[CheckRecord],
) -> None:
    if any(not full_size_record_current(row) for row in ordered_full_size):
        message = "plan replay has stale full-size rows"
        raise StaleRecordError(message)

    if any(not full_size_record_content_current(row) for row in ordered_full_size):
        message = "plan replay has changed full-size row content"
        raise StaleRecordError(message)

    if any(not check_record_current(row) for row in ordered_checks):
        message = "plan replay has stale reference rows"
        raise StaleRecordError(message)

    if any(not check_record_content_current(row) for row in ordered_checks):
        message = "plan replay has changed reference row content"
        raise StaleRecordError(message)

    _content_hashes_in_saved_order(
        tuple(record["full_size_record_content_hashes"]),
        ordered_full_size,
        "full-size",
    )
    _content_hashes_in_saved_order(
        tuple(record["check_record_content_hashes"]),
        ordered_checks,
        "reference",
    )
    _validate_reference_linkage(ordered_full_size, ordered_checks)


def _row_candidate_key(
    row: FullSizeRecord | CheckRecord,
) -> tuple[str, str, str]:
    return row.family, row.candidate_id, row.candidate_spec_hash


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
        if record.status != "passed" or not record.reference_passed:
            continue

        if not any(
            _reference_matches_full_size(check, record)
            for check in checks_by_key.get(_row_candidate_key(record), ())
        ):
            message = (
                "plan replay accepted full-size row has no matching passed reference: "
                f"{record.family}/{record.candidate_id}"
            )
            raise VPTuneError(message)


def _selected_record_hashes(
    record: Mapping[str, Any],
    selected: Mapping[str, Candidate],
    full_size_by_hash: Mapping[str, FullSizeRecord],
) -> dict[str, str]:
    selected_record_hashes = {
        str(family): str(row_hash)
        for family, row_hash in dict(record["records"]).items()
    }

    if set(selected_record_hashes) != set(selected):
        message = "plan replay selected families differ from record families"
        raise VPTuneError(message)

    missing_records = tuple(
        family
        for family, row_hash in selected_record_hashes.items()
        if row_hash not in full_size_by_hash
    )

    if missing_records:
        message = f"plan replay missing full-size records: {missing_records}"
        raise VPTuneError(message)

    return selected_record_hashes


def _validate_materializers(
    selected: Mapping[str, Candidate],
    materializers: Mapping[str, Materializer],
) -> None:
    missing_materializers = tuple(
        family for family in selected if family not in materializers
    )

    if missing_materializers:
        message = f"plan replay missing materializers: {missing_materializers}"
        raise VPTuneError(message)


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
) -> None:
    selected_families = tuple(str(family) for family in dict(record["selected"]))

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
            raise VPTuneError(message)

        if to_json_value(row.input_signature) != to_json_value(
            dict(expected_signature)
        ):
            message = f"plan replay family input signature is stale: {row.family}"
            raise StaleRecordError(message)


def _validate_replay_run_identity(
    record: Mapping[str, Any],
    replay_context: ReplayContext,
) -> None:
    if to_json_value(record["input_signature"]) != to_json_value(
        dict(replay_context.input_signature)
    ):
        message = "plan replay input signature is stale"
        raise StaleRecordError(message)

    if to_json_value(record["policy"]) != to_json_value(
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
        raise VPTuneError(message)


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
        raise VPTuneError(message)


def _validate_recomputed_selection(
    selected: Mapping[str, Candidate],
    selected_records: Mapping[str, FullSizeRecord],
    ordered_full_size: Sequence[FullSizeRecord],
    replay_context: ReplayContext,
) -> None:
    for family in selected:
        records = tuple(
            record for record in ordered_full_size if record.family == family
        )

        if not records:
            message = f"plan replay has no full-size rows for family: {family}"
            raise VPTuneError(message)

        expected_signature = replay_context.family_input_signatures.get(family)

        if expected_signature is None:
            message = f"plan replay missing family input signature: {family}"
            raise VPTuneError(message)

        recomputed_record = _replay_select_family(
            records,
            input_signature=expected_signature,
            policy=replay_context.selection_policy,
        )

        if recomputed_record.owner_hash != selected_records[family].owner_hash:
            message = f"plan replay selected row is stale: {family}"
            raise StaleRecordError(message)


def _replay_select_family(
    records: Sequence[FullSizeRecord],
    *,
    input_signature: Mapping[str, Any],
    policy: SelectionPolicy,
) -> FullSizeRecord:
    accepted = tuple(
        record
        for record in records
        if record.status == "passed"
        and record.reference_passed
        and to_json_value(record.input_signature) == to_json_value(input_signature)
        and _replay_memory_stable(record)
    )

    if not accepted:
        message = "plan replay family has no accepted rows"
        raise VPTuneError(message)

    fastest = min(record.median_elapsed_seconds() for record in accepted)
    near_fastest = tuple(
        record
        for record in accepted
        if record.median_elapsed_seconds() <= fastest * policy.near_fastest_multiplier
    )

    return min(near_fastest, key=lambda record: record.peak_reserved_mib())


def _replay_memory_stable(record: FullSizeRecord) -> bool:
    samples_by_device: dict[tuple[int, str], list[Measurement]] = {}

    for sample in record.memory_samples:
        key = (sample.rank, sample.device)
        samples_by_device.setdefault(key, []).append(sample)

    return all(
        _replay_memory_samples_stable(tuple(samples))
        for samples in samples_by_device.values()
    )


def _replay_memory_samples_stable(samples: tuple[Measurement, ...]) -> bool:
    if len(samples) <= 1:
        return True

    first = samples[0]

    return all(
        sample.post_allocated_mib <= first.post_allocated_mib
        and sample.post_reserved_mib <= first.post_reserved_mib
        for sample in samples[1:]
    )


def _selected_records(
    selected: Mapping[str, Candidate],
    selected_record_hashes: Mapping[str, str],
    full_size_by_hash: Mapping[str, FullSizeRecord],
) -> dict[str, FullSizeRecord]:
    records = {
        family: full_size_by_hash[selected_record_hashes[family]] for family in selected
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
        raise VPTuneError(message)

    if selected_record.candidate_id != candidate.candidate_id:
        message = f"plan replay selected record candidate differs: {family}"
        raise VPTuneError(message)

    if selected_record.candidate_spec_hash != candidate.candidate_spec_hash():
        message = f"plan replay selected record candidate spec differs: {family}"
        raise StaleRecordError(message)

    if selected_record.generator_id != candidate.generator_id:
        message = f"plan replay selected record generator differs: {family}"
        raise StaleRecordError(message)

    if selected_record.generator_version != candidate.generator_version:
        message = f"plan replay selected record generator version differs: {family}"
        raise StaleRecordError(message)

    if selected_record.status != "passed" or not selected_record.reference_passed:
        message = f"plan replay selected record did not pass: {family}"
        raise VPTuneError(message)

    if to_json_value(selected_record.candidate_settings) != to_json_value(
        candidate.settings
    ):
        message = f"plan replay selected record settings differ: {family}"
        raise VPTuneError(message)

    if to_json_value(selected_record.dependency_identities) != to_json_value(
        candidate.dependency_identities
    ):
        message = f"plan replay selected record dependencies differ: {family}"
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
        "candidate_spec_hash": candidate.candidate_spec_hash(),
        "full_size_owner_hash": record.owner_hash,
        "full_size_content_hash": record.computed_content_hash(),
        "full_size_input_signature": dict(record.input_signature),
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
        raise VPTuneError(message)

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
                raise VPTuneError(message)

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


def _candidate_record_key(candidate: Candidate) -> tuple[str, str, str]:
    return (
        candidate.family,
        candidate.candidate_id,
        candidate.candidate_spec_hash(),
    )


def _candidate_records_by_key(
    candidate_records: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, str, str], tuple[Candidate, Mapping[str, Any]]]:
    records = {}

    for record in candidate_records:
        candidate = candidate_record_from_json(record)
        key = _candidate_record_key(candidate)

        if key in records:
            message = f"plan replay has duplicate candidate row: {key}"
            raise VPTuneError(message)

        records[key] = (candidate, record)

    return records


def _validate_candidate_records(
    selected: Mapping[str, Candidate],
    selected_records: Mapping[str, FullSizeRecord],
    full_size_records: Sequence[FullSizeRecord],
    check_records: Sequence[CheckRecord],
    candidate_records: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, str, str], Candidate]:
    by_key = _candidate_records_by_key(candidate_records)
    full_size_by_key = {
        (record.family, record.candidate_id, record.candidate_spec_hash): record
        for record in full_size_records
    }
    checks_by_key = {}

    for record in check_records:
        checks_by_key.setdefault(
            (record.family, record.candidate_id, record.candidate_spec_hash),
            [],
        ).append(record)

    check_keys = {
        (record.family, record.candidate_id, record.candidate_spec_hash)
        for record in check_records
    }

    if len(full_size_by_key) != len(full_size_records):
        message = "plan replay has duplicate full-size candidate identities"
        raise VPTuneError(message)

    if set(by_key) != set(full_size_by_key) | check_keys:
        message = "plan replay candidate rows differ from result rows"
        raise VPTuneError(message)

    for key, (row_candidate, row) in by_key.items():
        full_size_record = full_size_by_key.get(key)

        if full_size_record is None:
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

        if _candidate_record_key(row_candidate) != (
            selected_records[family].family,
            selected_records[family].candidate_id,
            selected_records[family].candidate_spec_hash,
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
        to_json_value(row["input_signature"]) == to_json_value(check.input_signature)
        and to_json_value(candidate.settings) == to_json_value(check.candidate_settings)
        and to_json_value(candidate.dependency_identities)
        == to_json_value(check.dependency_identities)
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

    if to_json_value(row["input_signature"]) != to_json_value(
        full_size_record.input_signature
    ):
        message = f"plan replay candidate input signature is stale: {candidate.family}"
        raise StaleRecordError(message)


def _constraint_families(
    constraint: CohortConstraint,
    family_names: tuple[str, ...],
) -> tuple[str, ...]:
    if constraint.families:
        return constraint.families

    return family_names


def _candidate_matches_assignment(
    candidate: Candidate,
    assignment: CohortAssignment,
    constraints: tuple[CohortConstraint, ...],
    family_names: tuple[str, ...],
) -> bool:
    matched_constraint = False

    for constraint in constraints:
        if candidate.family not in _constraint_families(constraint, family_names):
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


def _cohort_assignments(
    constraints: tuple[CohortConstraint, ...],
    family_names: tuple[str, ...],
) -> tuple[CohortAssignment, ...]:
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
            covered_families.update(_constraint_families(constraint, family_names))

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

    return tuple(assignments)


def _assignment_records(
    assignment: CohortAssignment | None,
    constraints: tuple[CohortConstraint, ...],
    family_names: tuple[str, ...],
    ordered_full_size: Sequence[FullSizeRecord],
    candidates_by_key: Mapping[tuple[str, str, str], Candidate],
    replay_context: ReplayContext,
) -> tuple[tuple[Candidate, FullSizeRecord], ...]:
    pairs = []

    for record in ordered_full_size:
        expected_signature = replay_context.family_input_signatures[record.family]

        if to_json_value(record.input_signature) != to_json_value(expected_signature):
            continue

        candidate = candidates_by_key[
            record.family, record.candidate_id, record.candidate_spec_hash
        ]

        if assignment is not None and not _candidate_matches_assignment(
            candidate,
            assignment,
            constraints,
            family_names,
        ):
            continue

        if (
            record.status == "passed"
            and record.reference_passed
            and _replay_memory_stable(record)
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


def _replay_select_cohort(
    cohorts: Sequence[Mapping[str, tuple[Candidate, FullSizeRecord]]],
    policy: SelectionPolicy,
) -> Mapping[str, tuple[Candidate, FullSizeRecord]]:
    if not cohorts:
        message = "plan replay has no complete cohorts"
        raise VPTuneError(message)

    fastest = min(
        sum(record.median_elapsed_seconds() for _, record in cohort.values())
        for cohort in cohorts
    )
    near_fastest = tuple(
        cohort
        for cohort in cohorts
        if sum(record.median_elapsed_seconds() for _, record in cohort.values())
        <= fastest * policy.near_fastest_multiplier
    )

    return min(
        near_fastest,
        key=lambda cohort: sum(
            record.peak_reserved_mib() for _, record in cohort.values()
        ),
    )


def _validate_recomputed_cohort_selection(
    selected_records: Mapping[str, FullSizeRecord],
    ordered_full_size: Sequence[FullSizeRecord],
    replay_context: ReplayContext,
    dependencies_by_family: Mapping[str, tuple[str, ...]],
    materializers: Mapping[str, Materializer],
    candidates_by_key: Mapping[tuple[str, str, str], Candidate],
    cohort_assignment: CohortAssignment | None,
    cohort_constraints: tuple[CohortConstraint, ...],
) -> None:
    family_names = tuple(selected_records)
    cohorts = []
    state_by_cohort = {}

    for assignment in _cohort_assignments(cohort_constraints, family_names):
        pairs = _assignment_records(
            assignment,
            cohort_constraints,
            family_names,
            ordered_full_size,
            candidates_by_key,
            replay_context,
        )
        cohort = {}

        for family in family_names:
            records = tuple(
                record for candidate, record in pairs if candidate.family == family
            )

            if not records:
                break

            selected_record = _replay_select_family(
                records,
                input_signature=replay_context.family_input_signatures[family],
                policy=replay_context.selection_policy,
            )
            candidate = candidates_by_key[
                selected_record.family,
                selected_record.candidate_id,
                selected_record.candidate_spec_hash,
            ]
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

    selected_cohort = _replay_select_cohort(cohorts, replay_context.selection_policy)
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
        if record.owner_hash != selected_records[family].owner_hash:
            message = f"plan replay selected cohort row is stale: {family}"
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
        VPTuneError: If the record is not a summary or required rows are missing.
    """
    validate_json_record(record)

    if record["record_type"] != "summary":
        message = "plan replay requires a summary record"
        raise VPTuneError(message)

    selected_rows = dict(record["selected"])
    selected = {
        str(family): candidate_from_signature(dict(candidate_record))
        for family, candidate_record in selected_rows.items()
    }
    full_size_by_hash = _record_by_owner_hash(full_size_records, "full-size")
    check_by_hash = _record_by_owner_hash(check_records, "reference")
    ordered_full_size = _records_in_saved_order(
        tuple(record["full_size_records"]),
        full_size_by_hash,
        "full-size",
    )
    ordered_checks = _records_in_saved_order(
        tuple(record["check_records"]),
        check_by_hash,
        "reference",
    )

    _validate_ordered_replay_rows(
        record,
        ordered_full_size,
        ordered_checks,
    )
    _validate_replay_context(
        record,
        replay_context,
        materializers,
        ordered_full_size,
    )
    selected_record_hashes = _selected_record_hashes(
        record,
        selected,
        full_size_by_hash,
    )
    _validate_materializers(selected, materializers)
    selected_records = _selected_records(
        selected,
        selected_record_hashes,
        full_size_by_hash,
    )
    candidates_by_key = _validate_candidate_records(
        selected,
        selected_records,
        ordered_full_size,
        ordered_checks,
        candidate_records,
    )
    dependencies_by_family = {
        str(family): tuple(str(dependency) for dependency in dependencies)
        for family, dependencies in dict(record["dependencies_by_family"]).items()
    }
    validation_identity = _validation_replay_identity(record, replay_context)
    cohort_assignment = _cohort_assignment_from_json(record["cohort_assignment"])
    cohort_constraints = _cohort_constraints_from_json(
        tuple(record["cohort_constraints"])
    )
    _validate_dependency_identities(
        selected,
        selected_records,
        dependencies_by_family,
        materializers,
    )
    if not cohort_constraints:
        _validate_recomputed_selection(
            selected,
            selected_records,
            ordered_full_size,
            replay_context,
        )
    _validate_recomputed_cohort_selection(
        selected_records,
        ordered_full_size,
        replay_context,
        dependencies_by_family,
        materializers,
        candidates_by_key,
        cohort_assignment,
        cohort_constraints,
    )

    plan = Plan(
        selected=selected,
        records=selected_records,
        input_signature=dict(replay_context.input_signature),
        policy=replay_context.selection_policy,
        full_size_records=ordered_full_size,
        check_records=ordered_checks,
        materializers={family: materializers[family] for family in selected},
        validation_order=tuple(str(family) for family in record["validation_order"]),
        dependencies_by_family=dependencies_by_family,
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

    if not plan_record_current(record, plan):
        message = "plan replay summary is stale"
        raise StaleRecordError(message)

    if validation_records and validation_summary is None:
        message = "plan replay validation records require a validation summary"
        raise VPTuneError(message)

    if (replay_context.validation_required or plan.validation_required) and (
        validation_summary is None
    ):
        message = "plan replay requires selected-plan validation"
        raise VPTuneError(message)

    if replay_context.validation_order and tuple(record["validation_order"]) != tuple(
        replay_context.validation_order
    ):
        message = "plan replay validation order is stale"
        raise StaleRecordError(message)

    if validation_summary is not None and not validation_records:
        message = "plan replay validation summary requires validation records"
        raise VPTuneError(message)

    if validation_summary is not None and not selected_plan_validation_summary_current(
        validation_summary,
        plan,
        validation_records,
    ):
        message = "plan replay selected-plan validation summary is stale"
        raise StaleRecordError(message)

    if validation_summary is not None:
        _validate_selected_plan_validation_rows(
            plan,
            validation_summary,
            validation_records,
        )

    return plan
