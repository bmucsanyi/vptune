"""Runtime entry points."""

import dataclasses
import itertools
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from vptune.autobatch_bridge import find_autobatch_value
from vptune.candidates import topological_families
from vptune.data import (
    AutobatchDomain,
    Batch,
    Candidate,
    CandidateOperation,
    CheckRecord,
    CohortAssignment,
    CohortConstraint,
    Family,
    FullSizeRecord,
    Materializer,
    Measurement,
    Plan,
    PlanValidationContext,
    PlanValidator,
    Problem,
    ReferenceResult,
    RuntimeConfig,
    SelectionPolicy,
    Target,
    TuningRun,
)
from vptune.errors import MaterializationError, NoPassedCandidateError
from vptune.identities import stable_hash, to_json_value
from vptune.io import write_record
from vptune.measure import (
    MemoryBackend,
    OperationMeasurementError,
    default_memory_backend,
    failed_record,
    measure_once,
    run_candidate,
)
from vptune.schemas import (
    candidate_record_to_json,
    check_record_to_json,
    compute_record_owner_hash,
    full_size_record_to_json,
    plan_to_json,
    selected_plan_validation_summary_record,
)
from vptune.select import record_accepted, select_cohort, select_family
from vptune.tensor_tree import TensorTree, tree_signature

DTYPE_SETTING_KEYS = (
    "model_dtype",
    "compute_dtype",
    "accumulation_dtype",
    "storage_dtype",
)


@dataclasses.dataclass(frozen=True, slots=True)
class _RunCohortState:
    assignment: CohortAssignment
    cohort: Mapping[str, tuple[Candidate, FullSizeRecord]]
    input_signature: Mapping[str, Any]
    materializers: Mapping[str, Any]


@dataclasses.dataclass(frozen=True, slots=True)
class _AssignmentResult:
    state: _RunCohortState | None
    full_size_records: tuple[FullSizeRecord, ...]
    check_records: tuple[CheckRecord, ...]


@dataclasses.dataclass(frozen=True, slots=True)
class _ProbeResult:
    candidates: tuple[Candidate, ...]
    full_size_records: tuple[FullSizeRecord, ...]
    check_records: tuple[CheckRecord, ...]
    input_signature: dict[str, Any]
    materializer: Materializer
    runtime_identity: Mapping[str, Any]
    autobatch_selected_id: str | None = None


@dataclasses.dataclass(slots=True)
class _AutobatchProbeState:
    candidate: Candidate
    samples: list[Measurement] = dataclasses.field(default_factory=list)
    output_signature: Mapping[str, Any] | None = None
    error_type: str | None = None
    error: str | None = None


@dataclasses.dataclass(frozen=True, slots=True)
class _AutobatchReferenceOutcome:
    check_records: tuple[CheckRecord, ...]
    full_size_record: FullSizeRecord | None
    passed: bool


@dataclasses.dataclass(frozen=True, slots=True)
class _AutobatchReferenceRows:
    passed_values: tuple[int, ...]
    full_size_records: tuple[FullSizeRecord, ...]
    check_records: tuple[CheckRecord, ...]


def _reference_input(problem: Problem) -> tuple[Batch, TensorTree]:
    family = problem.operator.family

    return (
        problem.data.reference_batch(family, "tree_close"),
        problem.vectors.reference_vectors(family),
    )


def _probe_inputs(problem: Problem) -> tuple[tuple[Batch, TensorTree], ...]:
    batches = tuple(problem.data.probe_batches(problem.operator.family))
    vectors = tuple(problem.vectors.probe_vectors(problem.operator.family))

    if not batches:
        message = "problem data provider must return probe batches"
        raise RuntimeError(message)

    if not vectors:
        message = "problem vector provider must return probe vectors"
        raise RuntimeError(message)

    if len(batches) != len(vectors):
        message = "probe batch count must match probe vector count"
        raise RuntimeError(message)

    return tuple(zip(batches, vectors, strict=True))


def _runtime(problem: Problem) -> RuntimeConfig:
    return problem.runtime


def _domain_settings_product(
    domains: tuple[AutobatchDomain, ...],
) -> tuple[tuple[tuple[AutobatchDomain, int, Mapping[str, Any]], ...], ...]:
    if not domains:
        return ((),)

    entries = tuple(
        tuple(
            (domain, value, domain.settings_for_value(value)) for value in domain.values
        )
        for domain in domains
    )

    return tuple(itertools.product(*entries))


def _merge_domain_settings(
    candidate: Candidate,
    entries: tuple[tuple[AutobatchDomain, int, Mapping[str, Any]], ...],
) -> tuple[str, dict[str, Any], tuple[str, ...]]:
    settings = dict(candidate.settings)
    suffixes = []
    changed_axes = list(candidate.changed_axes)

    for domain, value, generated_settings in entries:
        suffixes.append(f"{domain.axis_name}={value}")

        if domain.axis_name not in changed_axes:
            changed_axes.append(domain.axis_name)

        for key, generated_value in generated_settings.items():
            if key in settings:
                message = (
                    f"autobatch domain setting collides with candidate setting: {key}"
                )
                raise MaterializationError(message)

            settings[key] = generated_value

    suffix = "|".join(suffixes)

    return suffix, settings, tuple(changed_axes)


def _expand_autobatch_domains(
    candidates: tuple[Candidate, ...],
    domains: tuple[AutobatchDomain, ...],
) -> tuple[Candidate, ...]:
    if not domains:
        return candidates

    expanded = []

    for candidate in candidates:
        for entries in _domain_settings_product(domains):
            suffix, settings, changed_axes = _merge_domain_settings(candidate, entries)
            expanded.append(
                dataclasses.replace(
                    candidate,
                    candidate_id=f"{candidate.candidate_id}|{suffix}",
                    settings=settings,
                    changed_axes=changed_axes,
                )
            )

    return tuple(expanded)


def _candidate_rows(runtime: RuntimeConfig) -> tuple[Candidate, ...]:
    candidates = _expand_autobatch_domains(
        runtime.candidates,
        runtime.autobatch_domains,
    )

    if not candidates:
        message = "runtime candidates must be non-empty"
        raise MaterializationError(message)

    candidate_ids = tuple(candidate.candidate_id for candidate in candidates)

    if len(set(candidate_ids)) != len(candidate_ids):
        message = "candidate ids must be unique"
        raise MaterializationError(message)

    axis_registry = runtime.axis_registry

    if axis_registry is not None:
        candidates = tuple(axis_registry.admit(candidate) for candidate in candidates)

    for candidate in candidates:
        if candidate.admission_status == "pending":
            message = f"candidate admission is pending: {candidate.candidate_id}"
            raise MaterializationError(message)

    return candidates


def _target_admission_error(candidate: Candidate, target: Target) -> str | None:
    errors = []

    for key in DTYPE_SETTING_KEYS:
        if key not in candidate.settings:
            continue

        value = candidate.settings[key]

        if not isinstance(value, str):
            errors.append(f"candidate setting must be a string: {key}")
        elif value not in target.allowed_dtypes:
            errors.append(f"candidate dtype is not allowed by target: {key}={value}")

    attention_impl = candidate.settings.get("attention_impl")

    if attention_impl is not None:
        if not isinstance(attention_impl, str):
            errors.append("candidate setting must be a string: attention_impl")
        elif attention_impl not in target.allowed_attention_impls:
            errors.append(
                f"candidate attention implementation is not allowed: {attention_impl}"
            )

    sharding = candidate.settings.get("sharding")

    if sharding is not None:
        if not isinstance(sharding, str):
            errors.append("candidate setting must be a string: sharding")
        elif sharding not in target.allowed_sharding_modes:
            errors.append(f"candidate sharding mode is not allowed: {sharding}")

    if errors:
        return "; ".join(errors)

    return None


def _admit_target(candidate: Candidate, target: Target) -> Candidate:
    """Return candidate with target admission applied."""
    if candidate.admission_status == "failed":
        return candidate

    error = _target_admission_error(candidate, target)

    if error is None:
        return candidate

    return dataclasses.replace(
        candidate,
        admission_status="failed",
        admission_error=error,
    )


def _single_autobatch_domain(runtime: RuntimeConfig) -> AutobatchDomain | None:
    if not runtime.autobatch_domains:
        return None

    if len(runtime.autobatch_domains) != 1:
        message = "runtime can declare one AutobatchDomain"
        raise MaterializationError(message)

    if len(runtime.candidates) != 1:
        message = "AutobatchDomain requires one base candidate"
        raise MaterializationError(message)

    return runtime.autobatch_domains[0]


def _candidate_for_domain_value(
    candidates: tuple[Candidate, ...],
    domain: AutobatchDomain,
    value: int,
) -> Candidate:
    settings = domain.settings_for_value(value)
    matches = tuple(
        candidate
        for candidate in candidates
        if all(candidate.settings.get(key) == item for key, item in settings.items())
    )

    if len(matches) != 1:
        message = f"autobatch domain value must map to one candidate: {value}"
        raise MaterializationError(message)

    return matches[0]


def _autobatch_cache_key(
    problem: Problem,
    domain: AutobatchDomain,
    input_signature: Mapping[str, Any],
) -> tuple[str, str, str, str, str]:
    return (
        "vptune",
        problem.operator.family,
        stable_hash(domain.signature()),
        stable_hash(input_signature),
        stable_hash(domain.cache_key_payload),
    )


def _measured_operation(
    runtime: RuntimeConfig,
    candidate: Candidate,
    inputs: tuple[tuple[Batch, TensorTree], ...],
) -> CandidateOperation:
    def operation() -> TensorTree:
        return tuple(
            runtime.operation_factory(candidate, batch, vector)()
            for batch, vector in inputs
        )

    return operation


def _check_record(
    candidate: Candidate,
    input_signature: dict[str, Any],
    result: ReferenceResult,
) -> CheckRecord:
    return CheckRecord(
        family=candidate.family,
        candidate_id=candidate.candidate_id,
        name=result.name,
        status="passed",
        input_signature=input_signature,
        candidate_settings=dict(candidate.settings),
        thresholds=dict(result.thresholds),
        measurements=dict(result.measurements),
        generator_id=candidate.generator_id,
        generator_version=candidate.generator_version,
        owner_hash=compute_record_owner_hash(
            record_type="reference",
            family=candidate.family,
            candidate_id=candidate.candidate_id,
            check_name=result.name,
            input_signature=input_signature,
            candidate_settings=candidate.settings,
            candidate_spec_hash=candidate.candidate_spec_hash(),
            thresholds=result.thresholds,
            dependency_identities=candidate.dependency_identities,
            generator_id=candidate.generator_id,
            generator_version=candidate.generator_version,
        ),
        candidate_spec_hash=candidate.candidate_spec_hash(),
        dependency_identities=dict(candidate.dependency_identities),
        cohort_assignment=dict(candidate.cohort_assignment),
    )


def _check_records(
    candidate: Candidate,
    input_signature: dict[str, Any],
    result: ReferenceResult,
) -> tuple[CheckRecord, ...]:
    records = []

    for child in result.child_results:
        child_records = _check_records(
            child.candidate,
            dict(child.input_signature),
            child.result,
        )
        records.extend(child_records)

    child_owner_hashes = tuple(record.owner_hash for record in records)

    if child_owner_hashes:
        result = dataclasses.replace(
            result,
            measurements={
                **dict(result.measurements),
                "child_reference_owner_hashes": child_owner_hashes,
            },
            child_results=(),
        )

    records.append(_check_record(candidate, input_signature, result))

    return tuple(records)


def _child_reference_candidates(
    result: ReferenceResult,
) -> tuple[tuple[Candidate, Mapping[str, Any]], ...]:
    candidates = []

    for child in result.child_results:
        candidates.append((child.candidate, child.input_signature))
        candidates.extend(_child_reference_candidates(child.result))

    return tuple(candidates)


def _write_child_reference_candidates(
    run_dir: Path | None,
    result: ReferenceResult,
) -> None:
    if run_dir is None:
        return

    for child_candidate, child_input_signature in _child_reference_candidates(result):
        _write_candidate(run_dir, dict(child_input_signature), child_candidate)


def _failed_check_record(
    candidate: Candidate,
    input_signature: dict[str, Any],
    *,
    error_type: str,
    error: str,
) -> CheckRecord:
    return CheckRecord(
        family=candidate.family,
        candidate_id=candidate.candidate_id,
        name="tree_close",
        status="failed",
        input_signature=input_signature,
        candidate_settings=dict(candidate.settings),
        thresholds={},
        measurements={},
        generator_id=candidate.generator_id,
        generator_version=candidate.generator_version,
        owner_hash=compute_record_owner_hash(
            record_type="reference",
            family=candidate.family,
            candidate_id=candidate.candidate_id,
            check_name="tree_close",
            input_signature=input_signature,
            candidate_settings=candidate.settings,
            candidate_spec_hash=candidate.candidate_spec_hash(),
            thresholds={},
            dependency_identities=candidate.dependency_identities,
            generator_id=candidate.generator_id,
            generator_version=candidate.generator_version,
        ),
        candidate_spec_hash=candidate.candidate_spec_hash(),
        dependency_identities=dict(candidate.dependency_identities),
        cohort_assignment=dict(candidate.cohort_assignment),
        error_type=error_type,
        error=error,
    )


def _passed_full_size_record_from_samples(
    candidate: Candidate,
    input_signature: dict[str, Any],
    state: _AutobatchProbeState,
) -> FullSizeRecord:
    if state.output_signature is None:
        message = (
            f"autobatch candidate has no output signature: {candidate.candidate_id}"
        )
        raise MaterializationError(message)

    samples = tuple(state.samples)

    return FullSizeRecord(
        family=candidate.family,
        candidate_id=candidate.candidate_id,
        status="passed",
        input_signature=dict(input_signature),
        candidate_settings=dict(candidate.settings),
        generator_id=candidate.generator_id,
        generator_version=candidate.generator_version,
        owner_hash=compute_record_owner_hash(
            record_type="full_size",
            family=candidate.family,
            candidate_id=candidate.candidate_id,
            input_signature=input_signature,
            candidate_settings=candidate.settings,
            candidate_spec_hash=candidate.candidate_spec_hash(),
            dependency_identities=candidate.dependency_identities,
            generator_id=candidate.generator_id,
            generator_version=candidate.generator_version,
        ),
        candidate_spec_hash=candidate.candidate_spec_hash(),
        timing_samples=samples,
        memory_samples=samples,
        output_signature=dict(state.output_signature),
        dependency_identities=dict(candidate.dependency_identities),
        cohort_assignment=dict(candidate.cohort_assignment),
        reference_passed=True,
    )


def _record_from_autobatch_state(
    candidate: Candidate,
    input_signature: dict[str, Any],
    state: _AutobatchProbeState,
) -> FullSizeRecord | None:
    if state.error_type is not None and state.error is not None:
        return failed_record(
            candidate,
            input_signature,
            error_type=state.error_type,
            error=state.error,
            reference_passed=True,
            samples=tuple(state.samples),
        )

    if state.samples:
        return _passed_full_size_record_from_samples(candidate, input_signature, state)

    return None


def _write_candidate(
    run_dir: Path, input_signature: dict[str, Any], candidate: Candidate
) -> None:
    write_record(
        run_dir
        / "candidates"
        / candidate.family
        / candidate.candidate_id
        / f"{candidate.candidate_spec_hash()}.json",
        candidate_record_to_json(candidate, input_signature),
    )


def _write_check(run_dir: Path, record: CheckRecord) -> None:
    write_record(
        run_dir
        / "references"
        / record.family
        / record.candidate_id
        / _record_candidate_spec_hash(record)
        / f"{record.name}.json",
        check_record_to_json(record),
    )


def _write_full_size(run_dir: Path, record: FullSizeRecord) -> None:
    write_record(
        run_dir
        / "full_size"
        / record.family
        / record.candidate_id
        / f"{_record_candidate_spec_hash(record)}.json",
        full_size_record_to_json(record),
    )


def _write_summary(run_dir: Path, plan: Plan) -> None:
    write_record(run_dir / "summaries" / "tuning.json", plan_to_json(plan))


def _record_candidate_spec_hash(record: CheckRecord | FullSizeRecord) -> str:
    return record.candidate_spec_hash


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


def _with_dependency_identities(
    problem: Problem,
    dependencies: tuple[str, ...],
    selected: dict[str, Candidate],
    records: dict[str, FullSizeRecord],
    materializers: dict[str, Any],
) -> Problem:
    if not dependencies:
        return problem

    runtime = _runtime(problem)
    dependency_identities = {
        dependency: _dependency_identity(
            dependency,
            selected[dependency],
            records[dependency],
            materializers[dependency].identity(),
        )
        for dependency in dependencies
    }
    candidates = tuple(
        dataclasses.replace(
            candidate,
            dependency_identities={
                **dict(candidate.dependency_identities),
                **dependency_identities,
            },
        )
        for candidate in runtime.candidates
    )

    return dataclasses.replace(
        problem,
        runtime=dataclasses.replace(runtime, candidates=candidates),
    )


def tune(
    problem: Problem,
    *,
    run_dir: Path | None = None,
    memory_backend: MemoryBackend | None = None,
    clock: Callable[[], float] = time.perf_counter,
) -> Plan:
    """Tune a single-family problem.

    Returns:
        Selected plan.
    """
    probe = _probe_problem(
        problem,
        run_dir=run_dir,
        memory_backend=memory_backend,
        clock=clock,
    )
    selected, selected_record = _select_probe_result(
        probe,
        input_signature=probe.input_signature,
        policy=problem.target.selection_policy,
    )

    plan = Plan(
        selected={problem.operator.family: selected},
        records={problem.operator.family: selected_record},
        input_signature=probe.input_signature,
        policy=problem.target.selection_policy,
        full_size_records=probe.full_size_records,
        check_records=probe.check_records,
        materializers={problem.operator.family: probe.materializer},
        validation_order=(problem.operator.family,),
        dependencies_by_family={problem.operator.family: ()},
        target_identity=problem.target.signature(),
        runtime_identities={problem.operator.family: probe.runtime_identity},
        adapter_identities={problem.operator.family: dict(problem.adapter_identity)},
        run_dir=run_dir,
    )

    if run_dir is not None:
        _write_summary(run_dir, plan)

    return plan


def _select_probe_result(
    probe: _ProbeResult,
    *,
    input_signature: Mapping[str, object],
    policy: SelectionPolicy,
) -> tuple[Candidate, FullSizeRecord]:
    if probe.autobatch_selected_id is None:
        return select_family(
            tuple(
                (candidate, record)
                for candidate, record in zip(
                    probe.candidates,
                    probe.full_size_records,
                    strict=True,
                )
            ),
            input_signature=input_signature,
            policy=policy,
        )

    candidates = {candidate.candidate_id: candidate for candidate in probe.candidates}
    records = {record.candidate_id: record for record in probe.full_size_records}
    candidate = candidates.get(probe.autobatch_selected_id)
    record = records.get(probe.autobatch_selected_id)

    if candidate is None or record is None:
        message = f"autobatch selected row is missing: {probe.autobatch_selected_id}"
        raise NoPassedCandidateError(message)

    if not record_accepted(record, input_signature):
        message = (
            f"autobatch selected row was not accepted: {probe.autobatch_selected_id}"
        )
        raise NoPassedCandidateError(message)

    return candidate, record


def _probe_problem(
    problem: Problem,
    *,
    run_dir: Path | None,
    memory_backend: MemoryBackend | None,
    clock: Callable[[], float],
) -> _ProbeResult:
    runtime = _runtime(problem)
    domain = _single_autobatch_domain(runtime)

    if domain is not None:
        return _probe_problem_with_autobatch(
            problem,
            domain,
            run_dir=run_dir,
            memory_backend=memory_backend,
            clock=clock,
        )

    candidates = tuple(
        _admit_target(candidate, problem.target)
        for candidate in _candidate_rows(runtime)
    )
    reference_batch, reference_vector = _reference_input(problem)
    probe_inputs = _probe_inputs(problem)

    input_signature = problem.input_signature()
    backend = (
        default_memory_backend(problem.target.devices)
        if memory_backend is None
        else memory_backend
    )
    records = []
    check_records = []

    for candidate in candidates:
        if run_dir is not None:
            _write_candidate(run_dir, input_signature, candidate)

        if candidate.admission_status == "failed":
            check_record = _failed_check_record(
                candidate,
                input_signature,
                error_type="AdmissionError",
                error=candidate.admission_error
                or f"candidate admission failed: {candidate.candidate_id}",
            )
            check_records.append(check_record)
            if run_dir is not None:
                _write_check(run_dir, check_record)

            records.append(
                failed_record(
                    candidate,
                    input_signature,
                    error_type="AdmissionError",
                    error=candidate.admission_error
                    or f"candidate admission failed: {candidate.candidate_id}",
                    reference_passed=False,
                )
            )
            if run_dir is not None:
                _write_full_size(run_dir, records[-1])

            continue

        try:
            reference_result = runtime.reference_check(
                candidate,
                reference_batch,
                reference_vector,
            )
        except RuntimeError as error:
            check_record = _failed_check_record(
                candidate,
                input_signature,
                error_type=type(error).__name__,
                error=str(error),
            )
            check_records.append(check_record)
            if run_dir is not None:
                _write_check(run_dir, check_record)

            records.append(
                failed_record(
                    candidate,
                    input_signature,
                    error_type=type(error).__name__,
                    error=str(error),
                    reference_passed=False,
                )
            )
            if run_dir is not None:
                _write_full_size(run_dir, records[-1])

            continue

        candidate_check_records = _check_records(
            candidate,
            input_signature,
            reference_result,
        )
        check_records.extend(candidate_check_records)
        _write_child_reference_candidates(run_dir, reference_result)
        if run_dir is not None:
            for check_record in candidate_check_records:
                _write_check(run_dir, check_record)

        operation = _measured_operation(runtime, candidate, probe_inputs)
        records.append(
            run_candidate(
                candidate,
                input_signature,
                operation,
                timing_policy=problem.target.timing_policy,
                memory_backend=backend,
                clock=clock,
                reference_passed=True,
            )
        )
        if run_dir is not None:
            _write_full_size(run_dir, records[-1])

    return _ProbeResult(
        candidates=candidates,
        full_size_records=tuple(records),
        check_records=tuple(check_records),
        input_signature=input_signature,
        materializer=runtime.materializer,
        runtime_identity=runtime.identity(),
    )


def _probe_problem_with_autobatch(
    problem: Problem,
    domain: AutobatchDomain,
    *,
    run_dir: Path | None,
    memory_backend: MemoryBackend | None,
    clock: Callable[[], float],
) -> _ProbeResult:
    runtime = _runtime(problem)
    candidates = tuple(
        _admit_target(candidate, problem.target)
        for candidate in _candidate_rows(runtime)
    )
    probe_inputs = _probe_inputs(problem)
    input_signature = problem.input_signature()
    backend = (
        default_memory_backend(problem.target.devices)
        if memory_backend is None
        else memory_backend
    )
    check_records = []
    records = []
    value_to_candidate = {
        value: _candidate_for_domain_value(candidates, domain, value)
        for value in domain.values
    }
    reference_rows = _autobatch_reference_rows(
        problem,
        runtime,
        domain,
        value_to_candidate,
        input_signature,
        run_dir,
    )
    records.extend(reference_rows.full_size_records)
    check_records.extend(reference_rows.check_records)

    states = {
        value: _AutobatchProbeState(value_to_candidate[value])
        for value in reference_rows.passed_values
    }

    if reference_rows.passed_values:
        selected_value = find_autobatch_value(
            lambda value: _probe_autobatch_value(
                value,
                states,
                runtime,
                probe_inputs,
                backend,
                clock,
            ),
            values=reference_rows.passed_values,
            goal=domain.goal,
            cache_key=_autobatch_cache_key(problem, domain, input_signature),
            warmup_steps=domain.warmup_steps,
            measure_steps=domain.measure_steps,
            devices=domain.devices,
        )
        selected_candidate_id = value_to_candidate[selected_value].candidate_id
    else:
        selected_candidate_id = None

    for value in reference_rows.passed_values:
        candidate = value_to_candidate[value]
        record = _record_from_autobatch_state(
            candidate,
            input_signature,
            states[value],
        )

        if record is None:
            continue

        records.append(record)
        if run_dir is not None:
            _write_full_size(run_dir, record)

    return _ProbeResult(
        candidates=candidates,
        full_size_records=tuple(records),
        check_records=tuple(check_records),
        input_signature=input_signature,
        materializer=runtime.materializer,
        runtime_identity=runtime.identity(),
        autobatch_selected_id=selected_candidate_id,
    )


def _autobatch_reference_rows(
    problem: Problem,
    runtime: RuntimeConfig,
    domain: AutobatchDomain,
    value_to_candidate: Mapping[int, Candidate],
    input_signature: dict[str, Any],
    run_dir: Path | None,
) -> _AutobatchReferenceRows:
    reference_batch, reference_vector = _reference_input(problem)
    passed_values = []
    records = []
    check_records = []

    for value in domain.values:
        candidate = value_to_candidate[value]

        if run_dir is not None:
            _write_candidate(run_dir, input_signature, candidate)

        outcome = _autobatch_reference_outcome(
            runtime,
            candidate,
            reference_batch,
            reference_vector,
            input_signature,
            run_dir,
        )
        check_records.extend(outcome.check_records)
        if run_dir is not None:
            for check_record in outcome.check_records:
                _write_check(run_dir, check_record)

        if outcome.full_size_record is not None:
            records.append(outcome.full_size_record)
            if run_dir is not None:
                _write_full_size(run_dir, outcome.full_size_record)

        if outcome.passed:
            passed_values.append(value)

    return _AutobatchReferenceRows(
        passed_values=tuple(passed_values),
        full_size_records=tuple(records),
        check_records=tuple(check_records),
    )


def _autobatch_reference_outcome(
    runtime: RuntimeConfig,
    candidate: Candidate,
    reference_batch: Batch,
    reference_vector: TensorTree,
    input_signature: dict[str, Any],
    run_dir: Path | None,
) -> _AutobatchReferenceOutcome:
    if candidate.admission_status == "failed":
        error = candidate.admission_error or (
            f"candidate admission failed: {candidate.candidate_id}"
        )

        return _AutobatchReferenceOutcome(
            check_records=(
                _failed_check_record(
                    candidate,
                    input_signature,
                    error_type="AdmissionError",
                    error=error,
                ),
            ),
            full_size_record=failed_record(
                candidate,
                input_signature,
                error_type="AdmissionError",
                error=error,
                reference_passed=False,
            ),
            passed=False,
        )

    try:
        reference_result = runtime.reference_check(
            candidate,
            reference_batch,
            reference_vector,
        )
    except RuntimeError as error:
        return _AutobatchReferenceOutcome(
            check_records=(
                _failed_check_record(
                    candidate,
                    input_signature,
                    error_type=type(error).__name__,
                    error=str(error),
                ),
            ),
            full_size_record=failed_record(
                candidate,
                input_signature,
                error_type=type(error).__name__,
                error=str(error),
                reference_passed=False,
            ),
            passed=False,
        )

    _write_child_reference_candidates(run_dir, reference_result)

    return _AutobatchReferenceOutcome(
        check_records=_check_records(candidate, input_signature, reference_result),
        full_size_record=None,
        passed=True,
    )


def _probe_autobatch_value(
    value: int,
    states: Mapping[int, _AutobatchProbeState],
    runtime: RuntimeConfig,
    probe_inputs: tuple[tuple[Batch, TensorTree], ...],
    memory_backend: MemoryBackend,
    clock: Callable[[], float],
) -> None:
    state = states.get(value)

    if state is None:
        message = f"autobatch selected unknown value: {value}"
        raise MaterializationError(message)

    operation = _measured_operation(runtime, state.candidate, probe_inputs)

    try:
        _, samples, output = measure_once(
            operation,
            memory_backend=memory_backend,
            clock=clock,
        )
    except OperationMeasurementError as error:
        state.samples.extend(error.samples)
        state.error_type = error.error_type
        state.error = error.error
        raise RuntimeError(error.error) from error
    except RuntimeError as error:
        state.error_type = type(error).__name__
        state.error = str(error)
        raise

    state.samples.extend(samples)
    state.output_signature = tree_signature(output)


def _constraint_families(
    constraint: CohortConstraint,
    family_names: tuple[str, ...],
) -> tuple[str, ...]:
    if constraint.families:
        missing = tuple(
            family for family in constraint.families if family not in family_names
        )

        if missing:
            message = f"cohort constraint names unknown families: {missing}"
            raise MaterializationError(message)

        return constraint.families

    return family_names


def _validate_cohort_constraints(
    constraints: tuple[CohortConstraint, ...],
    family_names: tuple[str, ...],
) -> None:
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

        if constraint.selection_aggregation != "sum_median_elapsed_seconds":
            message = f"unsupported cohort selection aggregation: {constraint.name}"
            raise MaterializationError(message)

        _constraint_families(constraint, family_names)

        for assignment in constraint.assignments:
            if set(assignment) != set(constraint.settings_keys):
                message = f"cohort assignment keys differ: {constraint.name}"
                raise MaterializationError(message)


def _cohort_assignments(
    constraints: tuple[CohortConstraint, ...],
    family_names: tuple[str, ...],
) -> tuple[CohortAssignment, ...]:
    _validate_cohort_constraints(constraints, family_names)

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

    if not assignments:
        message = "cohort constraints have no compatible assignments"
        raise MaterializationError(message)

    return tuple(assignments)


def _candidate_matches_assignment(
    candidate: Candidate,
    assignment: CohortAssignment,
    constraints: tuple[CohortConstraint, ...],
    family_names: tuple[str, ...],
) -> bool:
    for constraint in constraints:
        if candidate.family not in _constraint_families(constraint, family_names):
            continue

        for key in constraint.settings_keys:
            if candidate.settings.get(key) != assignment.values[key]:
                return False

    return True


def _problem_for_assignment(
    problem: Problem,
    assignment: CohortAssignment,
    constraints: tuple[CohortConstraint, ...],
    family_names: tuple[str, ...],
) -> Problem:
    runtime = _runtime(problem)
    candidates = tuple(
        dataclasses.replace(candidate, cohort_assignment=assignment.signature())
        for candidate in runtime.candidates
        if _candidate_matches_assignment(
            candidate, assignment, constraints, family_names
        )
    )

    return dataclasses.replace(
        problem,
        runtime=dataclasses.replace(runtime, candidates=candidates),
    )


def _prerequisite_failed_records(
    *,
    problem: Problem,
    family: Family,
    assignment: CohortAssignment,
    selected: Mapping[str, Candidate],
    records: Mapping[str, FullSizeRecord],
    materializers: Mapping[str, Any],
    constraints: tuple[CohortConstraint, ...],
    family_names: tuple[str, ...],
    run_dir: Path,
) -> tuple[FullSizeRecord, ...]:
    available_dependencies = tuple(
        dependency for dependency in family.dependencies if dependency in selected
    )
    problem_with_dependencies = _with_dependency_identities(
        problem,
        available_dependencies,
        dict(selected),
        dict(records),
        dict(materializers),
    )
    problem_for_assignment = _problem_for_assignment(
        problem_with_dependencies,
        assignment,
        constraints,
        family_names,
    )

    return _write_prerequisite_failed_records(problem_for_assignment, run_dir)


def _write_prerequisite_failed_records(
    problem: Problem,
    run_dir: Path,
) -> tuple[FullSizeRecord, ...]:
    input_signature = problem.input_signature()
    records = []

    for candidate in _candidate_rows(_runtime(problem)):
        admitted = _admit_target(candidate, problem.target)
        _write_candidate(run_dir, input_signature, admitted)
        record = failed_record(
            admitted,
            input_signature,
            error_type="PrerequisiteFailed",
            error="candidate prerequisites failed",
            reference_passed=False,
        )
        records.append(record)
        _write_full_size(run_dir, record)

    return tuple(records)


def _tune_cohort_assignment(
    *,
    run: TuningRun,
    assignment: CohortAssignment,
    ordered_families: tuple[Family, ...],
    family_names: tuple[str, ...],
    problems_by_family: Mapping[str, Problem],
    run_dir: Path,
    memory_backend: MemoryBackend | None,
    clock: Callable[[], float],
) -> _AssignmentResult:
    selected = {}
    records = {}
    materializers = {}
    full_size_records = ()
    check_records = ()
    blocked_families = ()
    input_signature: dict[str, Any] = {
        "run_id": run.run_id,
        "cohort_assignment": assignment.signature(),
    }

    for index, family in enumerate(ordered_families):
        if any(dependency not in selected for dependency in family.dependencies):
            blocked_families = ordered_families[index:]
            break

        problem = problems_by_family[family.name]

        if problem.operator != family.operator:
            message = f"family operator differs from problem operator: {family.name}"
            raise MaterializationError(message)

        problem_with_dependencies = _with_dependency_identities(
            problem,
            family.dependencies,
            selected,
            records,
            materializers,
        )
        problem_for_assignment = _problem_for_assignment(
            problem_with_dependencies,
            assignment,
            run.cohort_constraints,
            family_names,
        )

        try:
            probe = _probe_problem(
                problem_for_assignment,
                run_dir=run_dir,
                memory_backend=memory_backend,
                clock=clock,
            )
        except MaterializationError:
            blocked_families = ordered_families[index + 1 :]
            break

        full_size_records = (*full_size_records, *probe.full_size_records)
        check_records = (*check_records, *probe.check_records)
        input_signature[problem_for_assignment.operator.family] = dict(
            probe.input_signature
        )

        try:
            selected_candidate, selected_record = _select_probe_result(
                probe,
                input_signature=probe.input_signature,
                policy=problem_for_assignment.target.selection_policy,
            )
        except NoPassedCandidateError:
            blocked_families = ordered_families[index + 1 :]
            break

        selected[family.name] = selected_candidate
        records[family.name] = selected_record
        materializers[family.name] = probe.materializer

    for family in blocked_families:
        full_size_records = (
            *full_size_records,
            *_prerequisite_failed_records(
                problem=problems_by_family[family.name],
                family=family,
                assignment=assignment,
                selected=selected,
                records=records,
                materializers=materializers,
                constraints=run.cohort_constraints,
                family_names=family_names,
                run_dir=run_dir,
            ),
        )

    if set(selected) != set(family_names):
        return _AssignmentResult(
            state=None,
            full_size_records=full_size_records,
            check_records=check_records,
        )

    return _AssignmentResult(
        state=_RunCohortState(
            assignment=assignment,
            cohort={
                family: (
                    selected[family],
                    records[family],
                )
                for family in family_names
            },
            input_signature=input_signature,
            materializers=materializers,
        ),
        full_size_records=full_size_records,
        check_records=check_records,
    )


def tune_run(
    run: TuningRun,
    *,
    run_dir: Path,
    memory_backend: MemoryBackend | None = None,
    clock: Callable[[], float] = time.perf_counter,
) -> Plan:
    """Tune a multi-family run.

    The first implementation requires adapters to lower a `TuningRun` into concrete
    `Problem` objects. The function exists as the stable package entry point.

    Returns:
        Selected plan.

    Raises:
        MaterializationError: If the run has no lowered problems.
        NoPassedCandidateError: If no complete cohort has accepted rows.
    """
    if not run.problems:
        message = f"run adapter is required for TuningRun: {run.run_id} at {run_dir}"
        raise MaterializationError(message)

    ordered_families = topological_families(run.families)
    problems_by_family = {problem.operator.family: problem for problem in run.problems}
    family_names = tuple(family.name for family in ordered_families)

    if len(problems_by_family) != len(run.problems):
        message = "problem operator families must be unique"
        raise MaterializationError(message)

    if set(problems_by_family) != set(family_names):
        message = "run families must match lowered problem families"
        raise MaterializationError(message)

    _validate_run_validators(run, family_names)

    cohort_states = []
    full_size_records = ()
    check_records = ()

    for assignment in _cohort_assignments(run.cohort_constraints, family_names):
        result = _tune_cohort_assignment(
            run=run,
            assignment=assignment,
            ordered_families=ordered_families,
            family_names=family_names,
            problems_by_family=problems_by_family,
            run_dir=run_dir,
            memory_backend=memory_backend,
            clock=clock,
        )
        full_size_records = (*full_size_records, *result.full_size_records)
        check_records = (*check_records, *result.check_records)

        if result.state is not None:
            cohort_states.append(result.state)

    selected_cohort = select_cohort(
        tuple(state.cohort for state in cohort_states),
        families=family_names,
        policy=run.target.selection_policy,
    )
    selected_state = next(
        state for state in cohort_states if state.cohort == selected_cohort
    )
    selected = {family: selected_cohort[family][0] for family in family_names}
    records = {family: selected_cohort[family][1] for family in family_names}
    materializers = {
        family: selected_state.materializers[family] for family in family_names
    }
    input_signature = dict(selected_state.input_signature)

    if not full_size_records:
        message = "run has no measured candidate rows"
        raise NoPassedCandidateError(message)

    plan = Plan(
        selected=selected,
        records=records,
        input_signature=input_signature,
        policy=run.target.selection_policy,
        full_size_records=full_size_records,
        check_records=check_records,
        materializers=materializers,
        validation_order=family_names,
        dependencies_by_family={
            family.name: family.dependencies for family in ordered_families
        },
        cohort_assignment=selected_state.assignment,
        cohort_constraints=run.cohort_constraints,
        target_identity=run.target.signature(),
        runtime_identities={
            family: problems_by_family[family].runtime.identity()
            for family in family_names
        },
        adapter_identities={
            family: dict(problems_by_family[family].adapter_identity)
            for family in family_names
        },
        validation_required=bool(run.validators),
        validator_identities={
            family: dict(run.validator_identities[family]) for family in run.validators
        },
        run_dir=run_dir,
    )

    _write_summary(run_dir, plan)

    if run.validators:
        validate_plan(plan, run.validators, run_dir=run_dir)

    return plan


def _validate_run_validators(run: TuningRun, family_names: tuple[str, ...]) -> None:
    if not run.validators:
        if run.validator_identities:
            message = "run validator identities require validators"
            raise MaterializationError(message)

        return

    if set(run.validators) != set(family_names):
        message = "run validators must match run families"
        raise MaterializationError(message)

    if set(run.validator_identities) != set(family_names):
        message = "run validator identities must match run families"
        raise MaterializationError(message)

    empty = tuple(
        family for family in family_names if not run.validator_identities[family]
    )

    if empty:
        message = f"run validator identities must be non-empty: {empty}"
        raise MaterializationError(message)


def materialize(plan: Plan, *, family: str | None = None) -> Any:
    """Return selected materialization data for a plan.

    Raises:
        MaterializationError: If the requested family cannot be materialized.
    """
    if family is None:
        if len(plan.selected) != 1:
            message = "family is required for multi-family plans"
            raise MaterializationError(message)

        family = next(iter(plan.selected))

    candidate = plan.selected.get(family)

    if candidate is None:
        message = f"selected family is missing: {family}"
        raise MaterializationError(message)

    record = plan.records.get(family)

    if record is None:
        message = f"selected record is missing: {family}"
        raise MaterializationError(message)

    selected_materializer = plan.materializers.get(family)

    if selected_materializer is None:
        message = f"selected family has no materializer: {family}"
        raise MaterializationError(message)

    return selected_materializer(candidate, record)


def _validate_plan_dependency_identities(plan: Plan) -> None:
    dependencies_by_family = plan.selected_dependencies_by_family()

    if set(dependencies_by_family) != set(plan.selected):
        message = "selected-plan dependencies must name every selected family"
        raise MaterializationError(message)

    for family, candidate in plan.selected.items():
        dependencies = dependencies_by_family[family]
        record = plan.records[family]

        if set(candidate.dependency_identities) != set(dependencies):
            message = f"selected candidate dependencies differ: {family}"
            raise MaterializationError(message)

        if set(record.dependency_identities) != set(dependencies):
            message = f"selected record dependencies differ: {family}"
            raise MaterializationError(message)

        for dependency in dependencies:
            if dependency not in plan.selected:
                message = f"selected dependency is missing: {dependency}"
                raise MaterializationError(message)

            dependency_materializer = plan.materializers.get(dependency)

            if dependency_materializer is None:
                message = f"selected dependency has no materializer: {dependency}"
                raise MaterializationError(message)

            expected_identity = _dependency_identity(
                dependency,
                plan.selected[dependency],
                plan.records[dependency],
                dependency_materializer.identity(),
            )

            if to_json_value(candidate.dependency_identities[dependency]) != (
                to_json_value(expected_identity)
            ):
                message = f"selected candidate dependency identity differs: {family}"
                raise MaterializationError(message)

            if to_json_value(record.dependency_identities[dependency]) != (
                to_json_value(expected_identity)
            ):
                message = f"selected record dependency identity differs: {family}"
                raise MaterializationError(message)


def _validation_record(
    candidate: Candidate,
    input_signature: dict[str, Any],
    result: ReferenceResult,
) -> CheckRecord:
    record = CheckRecord(
        family=candidate.family,
        candidate_id=candidate.candidate_id,
        name="selected_plan_validation",
        status="passed",
        input_signature=input_signature,
        candidate_settings=dict(candidate.settings),
        thresholds=dict(result.thresholds),
        measurements=dict(result.measurements),
        generator_id=candidate.generator_id,
        generator_version=candidate.generator_version,
        owner_hash=compute_record_owner_hash(
            record_type="reference",
            family=candidate.family,
            candidate_id=candidate.candidate_id,
            check_name="selected_plan_validation",
            input_signature=input_signature,
            candidate_settings=candidate.settings,
            candidate_spec_hash=candidate.candidate_spec_hash(),
            thresholds=result.thresholds,
            dependency_identities=candidate.dependency_identities,
            generator_id=candidate.generator_id,
            generator_version=candidate.generator_version,
        ),
        candidate_spec_hash=candidate.candidate_spec_hash(),
        dependency_identities=dict(candidate.dependency_identities),
        cohort_assignment=dict(candidate.cohort_assignment),
    )

    return dataclasses.replace(record, content_hash=record.computed_content_hash())


def _failed_validation_record(
    candidate: Candidate,
    input_signature: dict[str, Any],
    *,
    error_type: str,
    error: str,
) -> CheckRecord:
    record = CheckRecord(
        family=candidate.family,
        candidate_id=candidate.candidate_id,
        name="selected_plan_validation",
        status="failed",
        input_signature=input_signature,
        candidate_settings=dict(candidate.settings),
        thresholds={},
        measurements={},
        generator_id=candidate.generator_id,
        generator_version=candidate.generator_version,
        owner_hash=compute_record_owner_hash(
            record_type="reference",
            family=candidate.family,
            candidate_id=candidate.candidate_id,
            check_name="selected_plan_validation",
            input_signature=input_signature,
            candidate_settings=candidate.settings,
            candidate_spec_hash=candidate.candidate_spec_hash(),
            thresholds={},
            dependency_identities=candidate.dependency_identities,
            generator_id=candidate.generator_id,
            generator_version=candidate.generator_version,
        ),
        candidate_spec_hash=candidate.candidate_spec_hash(),
        dependency_identities=dict(candidate.dependency_identities),
        cohort_assignment=dict(candidate.cohort_assignment),
        error_type=error_type,
        error=error,
    )

    return dataclasses.replace(record, content_hash=record.computed_content_hash())


def _write_validation(run_dir: Path, record: CheckRecord) -> None:
    write_record(
        run_dir
        / "references"
        / record.family
        / record.candidate_id
        / _record_candidate_spec_hash(record)
        / f"{record.name}.json",
        check_record_to_json(record),
    )


def _write_validation_summary(
    run_dir: Path,
    plan: Plan,
    records: tuple[CheckRecord, ...],
) -> None:
    write_record(
        run_dir / "summaries" / "selected_plan_validation.json",
        selected_plan_validation_summary_record(plan, records),
    )


def validate_plan(
    plan: Plan,
    validators: Mapping[str, PlanValidator],
    *,
    run_dir: Path | None = None,
) -> tuple[CheckRecord, ...]:
    """Validate every selected family in a materialized plan.

    Returns:
        Selected-plan validation records.

    Raises:
        MaterializationError: If validators differ from selected families.
        RuntimeError: If a selected family validator fails validation.
    """
    selected_families = set(plan.selected)

    if set(validators) != selected_families:
        message = "selected-plan validators must match selected families"
        raise MaterializationError(message)

    _validate_plan_dependency_identities(plan)

    input_signature = {
        "plan": plan.owner_hash(),
        "input_signature": dict(plan.input_signature),
    }
    records = []

    validation_order = plan.validation_order or tuple(plan.selected)

    if set(validation_order) != selected_families:
        message = "selected-plan validation order must match selected families"
        raise MaterializationError(message)

    materialized = {
        family: materialize(plan, family=family) for family in validation_order
    }

    for family in validation_order:
        candidate = plan.selected[family]
        selected_record = plan.records[family]
        selected_impl = materialized[family]
        dependency_names = plan.dependencies_by_family.get(
            family,
            tuple(candidate.dependency_identities),
        )
        dependencies = {
            dependency: materialized[dependency] for dependency in dependency_names
        }
        context = PlanValidationContext(
            family=family,
            selected=selected_impl,
            dependencies=dependencies,
            materialized=materialized,
        )

        try:
            result = validators[family](candidate, selected_record, context)
        except RuntimeError as error:
            record = _failed_validation_record(
                candidate,
                input_signature,
                error_type=type(error).__name__,
                error=str(error),
            )
            records.append(record)

            if run_dir is not None:
                _write_validation(run_dir, record)
                _write_validation_summary(
                    run_dir,
                    plan,
                    tuple(records),
                )

            raise

        record = _validation_record(candidate, input_signature, result)
        records.append(record)

        if run_dir is not None:
            _write_validation(run_dir, record)

    if run_dir is not None:
        _write_validation_summary(
            run_dir,
            plan,
            tuple(records),
        )

    return tuple(records)
