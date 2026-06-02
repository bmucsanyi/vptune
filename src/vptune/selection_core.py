"""Pure candidate ranking helpers."""

from collections.abc import Mapping, Sequence

from vptune.data import Candidate, FullSizeRecord, Measurement, SelectionPolicy
from vptune.errors import NoPassedCandidateError


def memory_stable(record: FullSizeRecord) -> bool:
    """Return whether post-call memory does not grow after the first sample."""
    samples_by_device = {}

    for sample in record.memory_samples:
        key = (sample.rank, sample.device)
        samples_by_device.setdefault(key, []).append(sample)

    return all(
        _memory_samples_stable(tuple(samples)) for samples in samples_by_device.values()
    )


def select_accepted_family(
    records: Sequence[tuple[Candidate, FullSizeRecord]],
    *,
    policy: SelectionPolicy,
) -> tuple[Candidate, FullSizeRecord]:
    """Select one already-accepted candidate row.

    Returns:
        Selected candidate and full-size record.

    Raises:
        NoPassedCandidateError: If no records are supplied.
    """
    validate_selection_policy(policy)

    if not records:
        message = "candidate family has no accepted rows"
        raise NoPassedCandidateError(message)

    fastest = min(record.median_elapsed_seconds() for _, record in records)
    near_fastest = tuple(
        (candidate, record)
        for candidate, record in records
        if record.median_elapsed_seconds() <= fastest * policy.near_fastest_multiplier
    )

    return min(near_fastest, key=lambda item: item[1].peak_reserved_mib())


def select_complete_cohort(
    cohorts: Sequence[Mapping[str, tuple[Candidate, FullSizeRecord]]],
    *,
    families: Sequence[str],
    policy: SelectionPolicy,
) -> Mapping[str, tuple[Candidate, FullSizeRecord]]:
    """Select one complete already-accepted family cohort.

    Returns:
        Selected cohort.

    Raises:
        NoPassedCandidateError: If no complete cohort is supplied.
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
        RuntimeError: If the policy requests an unsupported field.
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


def _memory_samples_stable(samples: tuple[Measurement, ...]) -> bool:
    if len(samples) <= 1:
        return True

    first = samples[0]

    return all(
        sample.post_allocated_mib <= first.post_allocated_mib
        and sample.post_reserved_mib <= first.post_reserved_mib
        for sample in samples[1:]
    )
