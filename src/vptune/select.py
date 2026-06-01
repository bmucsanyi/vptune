"""Candidate selection."""

from collections.abc import Mapping, Sequence

from vptune.data import Candidate, FullSizeRecord, Measurement, SelectionPolicy
from vptune.errors import NoPassedCandidateError
from vptune.identities import to_json_value
from vptune.schemas import full_size_record_content_current, full_size_record_current


def memory_stable(record: FullSizeRecord) -> bool:
    """Return whether post-call memory does not grow after the first sample."""
    samples_by_device = {}

    for sample in record.memory_samples:
        key = (sample.rank, sample.device)
        samples_by_device.setdefault(key, []).append(sample)

    return all(
        _memory_samples_stable(tuple(samples)) for samples in samples_by_device.values()
    )


def _memory_samples_stable(samples: tuple[Measurement, ...]) -> bool:
    if len(samples) <= 1:
        return True

    first = samples[0]

    return all(
        sample.post_allocated_mib <= first.post_allocated_mib
        and sample.post_reserved_mib <= first.post_reserved_mib
        for sample in samples[1:]
    )


def record_accepted(
    record: FullSizeRecord, input_signature: Mapping[str, object]
) -> bool:
    """Return whether a full-size record can enter selection."""
    return (
        record.status == "passed"
        and record.reference_passed
        and to_json_value(record.input_signature) == to_json_value(input_signature)
        and full_size_record_current(record)
        and (not record.content_hash or full_size_record_content_current(record))
        and memory_stable(record)
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
    validate_selection_policy(policy)
    accepted = tuple(
        (candidate, record)
        for candidate, record in records
        if record_accepted(record, input_signature)
    )

    if not accepted:
        message = "candidate family has no accepted rows"
        raise NoPassedCandidateError(message)

    fastest = min(record.median_elapsed_seconds() for _, record in accepted)
    near_fastest = tuple(
        (candidate, record)
        for candidate, record in accepted
        if record.median_elapsed_seconds() <= fastest * policy.near_fastest_multiplier
    )

    return min(near_fastest, key=lambda item: item[1].peak_reserved_mib())


def select_cohort(
    cohorts: Sequence[Mapping[str, tuple[Candidate, FullSizeRecord]]],
    *,
    families: Sequence[str],
    policy: SelectionPolicy,
) -> Mapping[str, tuple[Candidate, FullSizeRecord]]:
    """Select a complete family cohort.

    Returns:
        Selected cohort.

    Raises:
        NoPassedCandidateError: If no complete cohort is available.
    """
    validate_selection_policy(policy)
    family_set = set(families)
    complete = tuple(cohort for cohort in cohorts if set(cohort) == family_set)

    if not complete:
        message = "candidate selection has no complete cohort"
        raise NoPassedCandidateError(message)

    fastest = min(cohort_median_elapsed(cohort) for cohort in complete)
    near_fastest = tuple(
        cohort
        for cohort in complete
        if cohort_median_elapsed(cohort) <= fastest * policy.near_fastest_multiplier
    )

    return min(near_fastest, key=cohort_peak_reserved)


def cohort_median_elapsed(
    cohort: Mapping[str, tuple[Candidate, FullSizeRecord]],
) -> float:
    """Return summed family median elapsed seconds."""
    return sum(record.median_elapsed_seconds() for _, record in cohort.values())


def cohort_peak_reserved(
    cohort: Mapping[str, tuple[Candidate, FullSizeRecord]],
) -> float:
    """Return summed family peak reserved memory."""
    return sum(record.peak_reserved_mib() for _, record in cohort.values())


def validate_selection_policy(policy: SelectionPolicy) -> None:
    """Validate supported selection policy fields.

    Raises:
        RuntimeError: If a policy field requests an unsupported statistic.
    """
    if policy.speed_statistic != "median_elapsed_seconds":
        message = f"unsupported speed statistic: {policy.speed_statistic}"
        raise RuntimeError(message)

    if policy.tie_breaker != "min_peak_reserved_mib":
        message = f"unsupported tie breaker: {policy.tie_breaker}"
        raise RuntimeError(message)

    if policy.cohort_speed_statistic != "sum_median_elapsed_seconds":
        message = f"unsupported cohort speed statistic: {policy.cohort_speed_statistic}"
        raise RuntimeError(message)

    if policy.cohort_tie_breaker != "sum_peak_reserved_mib":
        message = f"unsupported cohort tie breaker: {policy.cohort_tie_breaker}"
        raise RuntimeError(message)
