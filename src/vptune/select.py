"""Candidate selection."""

from collections.abc import Mapping, Sequence

from vptune.data import Candidate, FullSizeRecord, SelectionPolicy
from vptune.errors import NoPassedCandidateError
from vptune.identities import to_json_value
from vptune.selection_core import (
    memory_stable,
    select_accepted_family,
    select_complete_cohort,
)


def record_accepted(
    record: FullSizeRecord, input_signature: Mapping[str, object]
) -> bool:
    """Return whether a full-size record can enter selection."""
    return (
        record.status == "passed"
        and record.reference_passed
        and to_json_value(record.input_signature) == to_json_value(input_signature)
        and memory_stable(record)
    )


def record_matches_candidate(candidate: Candidate, record: FullSizeRecord) -> bool:
    """Return whether a record belongs to the paired candidate."""
    return (
        record.family == candidate.family
        and record.candidate_id == candidate.candidate_id
        and to_json_value(record.candidate_settings)
        == to_json_value(candidate.settings)
        and to_json_value(record.dependency_identities)
        == to_json_value(candidate.dependency_identities)
        and to_json_value(record.cohort_assignment)
        == to_json_value(candidate.cohort_assignment)
        and record.generator_id == candidate.generator_id
        and record.generator_version == candidate.generator_version
    )


def select_family(
    records: Sequence[tuple[Candidate, FullSizeRecord]],
    *,
    input_signature: Mapping[str, object],
    policy: SelectionPolicy,
) -> tuple[Candidate, FullSizeRecord]:
    """Select the best candidate for one family.

    Returns:
        Selected candidate and full-size record.

    Raises:
        NoPassedCandidateError: If no records are accepted.
    """
    accepted = tuple(
        (candidate, record)
        for candidate, record in records
        if record_matches_candidate(candidate, record)
        and record_accepted(record, input_signature)
    )

    if not accepted:
        message = "candidate family has no accepted rows"
        raise NoPassedCandidateError(message)

    return select_accepted_family(accepted, policy=policy)


def select_cohort(
    cohorts: Sequence[Mapping[str, tuple[Candidate, FullSizeRecord]]],
    *,
    families: Sequence[str],
    policy: SelectionPolicy,
) -> Mapping[str, tuple[Candidate, FullSizeRecord]]:
    """Select a complete family cohort.

    Returns:
        Selected cohort.
    """
    return select_complete_cohort(cohorts, families=families, policy=policy)
