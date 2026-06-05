"""Pure candidate ranking helpers."""

from collections.abc import Mapping, Sequence

from vptune.data import Candidate, FullSizeRecord, Measurement, SelectionPolicy
from vptune.errors import NoPassedCandidateError

FULL_SIZE_AGREEMENT_KEY = "full_size_agreement_passed"
BASELINE_ATTENTION_FRONTENDS = {
    "pytorch_sdpa_direct",
    "patched_eager",
    "transformers_eager",
    "transformers_sdpa",
}
COMPILED_SPEED_STATISTIC = "compile_amortized_steady_state_seconds"
ACCEPTED_STATUS = "passed_current_reference_full_size_agreement_stable_memory"


def memory_stable(record: FullSizeRecord) -> bool:
    """Return whether post-call memory does not grow after the first sample."""
    samples_by_device = {}

    for sample in record.memory_samples:
        key = (sample.rank, sample.device)
        samples_by_device.setdefault(key, []).append(sample)

    return all(
        _memory_samples_stable(tuple(samples)) for samples in samples_by_device.values()
    )


def full_size_agreement_satisfied(record: FullSizeRecord) -> bool:
    """Return whether required full-size agreement passed."""
    if not _requires_full_size_agreement(record.candidate_settings):
        return True

    return record.selection_metadata.get(FULL_SIZE_AGREEMENT_KEY) is True


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

    fastest = min(selection_score_seconds(record, policy) for _, record in records)
    near_fastest = tuple(
        (candidate, record)
        for candidate, record in records
        if selection_score_seconds(record, policy)
        <= fastest * policy.near_fastest_multiplier
    )

    return min(
        near_fastest,
        key=lambda item: selection_memory_mib(item[1], policy),
    )


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

    fastest = min(cohort_selection_score(cohort, policy) for cohort in complete)
    near_fastest = tuple(
        cohort
        for cohort in complete
        if cohort_selection_score(cohort, policy)
        <= fastest * policy.near_fastest_multiplier
    )

    return min(near_fastest, key=lambda cohort: cohort_memory_mib(cohort, policy))


def cohort_selection_score(
    cohort: Mapping[str, tuple[Candidate, FullSizeRecord]],
    policy: SelectionPolicy,
) -> float:
    """Return summed family selection score."""
    return sum(selection_score_seconds(record, policy) for _, record in cohort.values())


def selection_score_seconds(record: FullSizeRecord, policy: SelectionPolicy) -> float:
    """Return row speed score in seconds."""
    if record.candidate_settings.get("compile.enabled") != "true":
        return _eager_score_seconds(record, policy)

    metadata = dict(record.selection_metadata)

    if _is_distributed_record(record):
        compile_time = _metadata_float(metadata, "global_compile_time_seconds")
        steady_time = _metadata_float(metadata, "global_steady_elapsed_seconds")
    else:
        compile_time = _metadata_float(metadata, "compile_time_seconds")
        steady_time = _metadata_float(metadata, "steady_elapsed_seconds")

    recompiles = _metadata_int(metadata, "recompile_count")

    return (
        ((1 + recompiles) * compile_time) / policy.compile_call_horizon
    ) + steady_time


def selection_memory_mib(record: FullSizeRecord, policy: SelectionPolicy) -> float:
    """Return row memory score in MiB.

    Raises:
        RuntimeError: If the reduction is unsupported or memory samples are missing.
    """
    if policy.rank_memory_reduction == "max_peak_allocated":
        return _max_peak_allocated_mib(record)

    if policy.rank_memory_reduction == "max_peak_reserved":
        return record.peak_reserved_mib()

    if policy.rank_memory_reduction == "sum_peak_reserved":
        return _sum_peak_reserved_mib(record)

    message = f"unsupported rank memory reduction: {policy.rank_memory_reduction}"
    raise RuntimeError(message)


def cohort_memory_mib(
    cohort: Mapping[str, tuple[Candidate, FullSizeRecord]],
    policy: SelectionPolicy,
) -> float:
    """Return summed family memory score."""
    return sum(selection_memory_mib(record, policy) for _, record in cohort.values())


def validate_selection_policy(policy: SelectionPolicy) -> None:
    """Validate supported selection policy fields.

    Raises:
        RuntimeError: If the policy requests an unsupported field.
    """
    if policy.speed_statistic != "median_elapsed_seconds":
        message = f"unsupported speed statistic: {policy.speed_statistic}"
        raise RuntimeError(message)

    if policy.compiled_speed_statistic != COMPILED_SPEED_STATISTIC:
        message = (
            f"unsupported compiled speed statistic: {policy.compiled_speed_statistic}"
        )
        raise RuntimeError(message)

    if policy.distributed_speed_statistic != "global_elapsed_seconds":
        message = (
            "unsupported distributed speed statistic: "
            f"{policy.distributed_speed_statistic}"
        )
        raise RuntimeError(message)

    if policy.rank_memory_reduction not in {
        "max_peak_allocated",
        "max_peak_reserved",
        "sum_peak_reserved",
    }:
        message = f"unsupported rank memory reduction: {policy.rank_memory_reduction}"
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

    if policy.accepted_status != ACCEPTED_STATUS:
        message = f"unsupported accepted status: {policy.accepted_status}"
        raise RuntimeError(message)

    if policy.compile_call_horizon <= 0:
        message = "compile_call_horizon must be positive"
        raise RuntimeError(message)


def _eager_score_seconds(
    record: FullSizeRecord,
    policy: SelectionPolicy,
) -> float:
    if _is_distributed_record(record):
        metadata = dict(record.selection_metadata)

        return _metadata_float(metadata, policy.distributed_speed_statistic)

    return record.median_elapsed_seconds()


def _memory_samples_stable(samples: tuple[Measurement, ...]) -> bool:
    if len(samples) <= 1:
        return True

    first = samples[0]

    return all(
        sample.post_allocated_mib <= first.post_allocated_mib
        and sample.post_reserved_mib <= first.post_reserved_mib
        for sample in samples[1:]
    )


def _requires_full_size_agreement(settings: Mapping[str, object]) -> bool:
    return (
        _attention_requires_full_size_agreement(settings)
        or _compile_requires_full_size_agreement(settings)
        or _fusion_requires_full_size_agreement(settings)
        or settings.get("tp.loss_parallel") == "true"
        or settings.get("context_parallel.enabled") == "true"
    )


def _attention_requires_full_size_agreement(
    settings: Mapping[str, object],
) -> bool:
    frontend = settings.get("attention.frontend")

    if frontend is not None and frontend not in BASELINE_ATTENTION_FRONTENDS:
        return True

    if settings.get("attention.partition") in {"packed_tokens", "blockwise_queries"}:
        return True

    kernel = settings.get("attention.sdpa_kernel")

    if kernel is not None and kernel not in {"math", "priority_list"}:
        return True

    if kernel != "priority_list":
        return False

    priority = settings.get("attention.sdpa_priority_list")

    if not isinstance(priority, tuple):
        return True

    return any(entry != "math" for entry in priority)


def _compile_requires_full_size_agreement(settings: Mapping[str, object]) -> bool:
    return (
        settings.get("compile.cuda_graphs") == "true"
        or settings.get("compile.mode") == "max-autotune"
    )


def _fusion_requires_full_size_agreement(settings: Mapping[str, object]) -> bool:
    return any(
        key.startswith("fusion.") and value != "model_default"
        for key, value in settings.items()
    )


def _is_distributed_record(record: FullSizeRecord) -> bool:
    strategy = record.candidate_settings.get("distributed.strategy")

    return strategy is not None and strategy != "single_gpu"


def _max_peak_allocated_mib(record: FullSizeRecord) -> float:
    if not record.memory_samples:
        message = "passed full-size record has no memory samples"
        raise RuntimeError(message)

    return max(sample.peak_allocated_mib for sample in record.memory_samples)


def _sum_peak_reserved_mib(record: FullSizeRecord) -> float:
    if not record.memory_samples:
        message = "passed full-size record has no memory samples"
        raise RuntimeError(message)

    groups = {}

    for sample in record.memory_samples:
        key = (sample.rank, sample.device)
        current = groups.get(key)

        if current is None or sample.peak_reserved_mib > current:
            groups[key] = sample.peak_reserved_mib

    return sum(groups.values())


def _metadata_float(metadata: Mapping[str, object], key: str) -> float:
    value = metadata.get(key)

    if not isinstance(value, int | float):
        message = f"row selection metadata missing {key}"
        raise TypeError(message)

    return float(value)


def _metadata_int(metadata: Mapping[str, object], key: str) -> int:
    value = metadata.get(key)

    if not isinstance(value, int):
        message = f"row selection metadata missing {key}"
        raise TypeError(message)

    if value < 0:
        message = f"compiled row selection metadata has negative {key}"
        raise RuntimeError(message)

    return value
