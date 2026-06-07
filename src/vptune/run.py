"""Runtime entry points."""

import dataclasses
import itertools
import time
from collections.abc import Callable, Mapping, MutableMapping
from pathlib import Path
from typing import Any

from vptune.admission import admit_call_core_settings
from vptune.autobatch_bridge import find_autobatch_value
from vptune.candidates import AxisTable, axis_table, topological_families
from vptune.cohorts import candidate_matches_assignment, cohort_assignments
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
    OperatorSpec,
    Plan,
    PlanValidationContext,
    PlanValidator,
    Problem,
    ReferenceCheck,
    ReferenceResult,
    ReplayContext,
    RuntimeConfig,
    SelectionPolicy,
    Target,
    TimingPolicy,
    TuningRun,
)
from vptune.errors import (
    AdmissionError,
    MaterializationError,
    NoPassedCandidateError,
    ReferenceFailedError,
)
from vptune.identities import canonical_json, stable_hash
from vptune.io import read_record, write_record, write_record_exclusive
from vptune.measure import (
    MemoryBackend,
    OperationMeasurementError,
    default_memory_backend,
    failed_record,
    measure_once,
    run_candidate,
)
from vptune.runtime import (
    CompositionChild,
    composition_runtime_config,
    deferred_runtime_finite_checks,
    standard_runtime_with_matrix_free_bindings,
)
from vptune.schemas import (
    candidate_from_signature,
    candidate_record_from_json,
    candidate_record_to_json,
    check_record_from_json,
    check_record_to_json,
    full_size_record_from_json,
    full_size_record_to_json,
    plan_from_json,
    plan_to_json,
    selected_plan_validation_input_signature,
    selected_plan_validation_summary_record,
)
from vptune.select import (
    record_accepted,
    record_matches_candidate,
    select_cohort,
    select_family,
)
from vptune.selection_core import selection_memory_mib, selection_score_seconds
from vptune.tensor_tree import TensorTree, tree_signature

DTYPE_SETTING_KEYS = (
    "dtype.parameter_storage",
    "dtype.model_compute",
    "dtype.vector",
    "dtype.intermediate",
    "dtype.output",
    "dtype.metric_factor",
    "storage_dtype",
)


@dataclasses.dataclass(frozen=True, slots=True)
class _RunCohortState:
    assignment: CohortAssignment
    cohort: Mapping[str, tuple[Candidate, FullSizeRecord]]
    input_signature: Mapping[str, Any]
    materializers: Mapping[str, Any]
    runtime_identities: Mapping[str, Mapping[str, Any]]


@dataclasses.dataclass(frozen=True, slots=True)
class _RunProblemIndex:
    ordered_families: tuple[Family, ...]
    problems_by_family: Mapping[str, Problem]
    family_names: tuple[str, ...]


@dataclasses.dataclass(frozen=True, slots=True)
class _AssignmentResult:
    state: _RunCohortState | None
    candidate_rows: tuple[Candidate, ...]
    full_size_records: tuple[FullSizeRecord, ...]
    check_records: tuple[CheckRecord, ...]


@dataclasses.dataclass(frozen=True, slots=True)
class _PrerequisiteRows:
    candidates: tuple[Candidate, ...]
    full_size_records: tuple[FullSizeRecord, ...]


@dataclasses.dataclass(frozen=True, slots=True)
class _ProbeResult:
    candidates: tuple[Candidate, ...]
    candidate_rows: tuple[Candidate, ...]
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
    selection_metadata: Mapping[str, Any] = dataclasses.field(default_factory=dict)
    error_type: str | None = None
    error: str | None = None


@dataclasses.dataclass(frozen=True, slots=True)
class _ReferenceOutcome:
    candidates: tuple[Candidate, ...]
    check_records: tuple[CheckRecord, ...]
    full_size_record: FullSizeRecord | None
    passed: bool
    cached: bool = False


@dataclasses.dataclass(frozen=True, slots=True)
class _RunDirCache:
    candidates_by_input_key: Mapping[str, tuple[Candidate, ...]]
    checks_by_row_key: Mapping[str, CheckRecord]
    check_records: tuple[CheckRecord, ...]
    full_size_records: tuple[FullSizeRecord, ...]


@dataclasses.dataclass(frozen=True, slots=True)
class _AutobatchReferenceRows:
    candidates: tuple[Candidate, ...]
    passed_values: tuple[int, ...]
    full_size_records: tuple[FullSizeRecord, ...]
    check_records: tuple[CheckRecord, ...]


@dataclasses.dataclass(frozen=True, slots=True)
class _BalancedGroupRows:
    retained: tuple[Candidate, ...]
    candidate_rows: tuple[Candidate, ...]
    check_records: tuple[CheckRecord, ...]


@dataclasses.dataclass(frozen=True, slots=True)
class _CandidateProbeRows:
    candidate_rows: tuple[Candidate, ...]
    full_size_records: tuple[FullSizeRecord, ...]
    check_records: tuple[CheckRecord, ...]


@dataclasses.dataclass(frozen=True, slots=True)
class _ProbeContext:
    runtime: RuntimeConfig
    reference_batch: Batch
    reference_vector: TensorTree
    probe_inputs: tuple[tuple[Batch, TensorTree], ...]
    input_signature: dict[str, Any]
    timing_policy: TimingPolicy
    selection_policy: SelectionPolicy
    memory_backend: MemoryBackend
    clock: Callable[[], float]
    run_dir: Path | None
    run_cache: _RunDirCache


def _reference_input(problem: Problem) -> tuple[Batch, TensorTree]:
    family = problem.operator.family

    return (
        problem.data.reference_batch(family, problem.runtime.reference_check_name),
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
    identity = problem.runtime.identity()
    adapter_id = identity.get("adapter_id")

    if adapter_id is None:
        return problem.runtime

    for key in ("adapter_id", "adapter_version"):
        if problem.adapter_identity.get(key) != identity.get(key):
            message = f"problem adapter identity does not match runtime identity: {key}"
            raise MaterializationError(message)

    return problem.runtime


def _memory_backend(
    devices: tuple[str, ...],
    memory_backend: MemoryBackend | None,
) -> MemoryBackend:
    if memory_backend is None:
        return default_memory_backend(devices)

    return memory_backend


def _input_signature(problem: Problem, memory_backend: MemoryBackend) -> dict[str, Any]:
    signature = problem.input_signature()
    signature["measurement"] = {
        "memory_backend": dict(memory_backend.identity()),
    }

    return signature


def _run_input_signature(**fields: Any) -> dict[str, Any]:
    return dict(fields)


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

    candidates = tuple(
        _admit_runtime_bindings(candidate, runtime.signature)
        for candidate in candidates
    )

    for candidate in candidates:
        if candidate.admission_status == "pending":
            message = f"candidate admission is pending: {candidate.candidate_id}"
            raise MaterializationError(message)

    return candidates


def _target_admission_error(candidate: Candidate, target: Target) -> str | None:
    errors = []

    for key in DTYPE_SETTING_KEYS:
        error = _allowed_target_string_error(
            candidate.settings,
            key,
            target.allowed_dtypes,
            "dtype",
        )

        if error is not None:
            errors.append(error)

    for key, allowed_values, label in (
        (
            "attention.frontend",
            target.allowed_attention_frontends,
            "attention frontend",
        ),
        ("attention.sdpa_kernel", target.allowed_sdpa_kernels, "SDPA kernel"),
        (
            "distributed.strategy",
            target.allowed_sharding_modes,
            "distributed strategy",
        ),
    ):
        error = _allowed_target_string_error(
            candidate.settings,
            key,
            allowed_values,
            label,
        )

        if error is not None:
            errors.append(error)

    if errors:
        return "; ".join(errors)

    return None


def _allowed_target_string_error(
    settings: Mapping[str, Any],
    key: str,
    allowed_values: tuple[str, ...],
    label: str,
) -> str | None:
    value = settings.get(key)

    if value is None:
        return None

    if not isinstance(value, str):
        return f"candidate setting must be a string: {key}"

    if value not in allowed_values:
        return f"candidate {label} is not allowed by target: {key}={value}"

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


def _admit_runtime_bindings(
    candidate: Candidate,
    runtime_signature: Mapping[str, Any],
) -> Candidate:
    if candidate.admission_status == "failed":
        return candidate

    error = _runtime_binding_admission_error(candidate, runtime_signature)

    if error is None:
        return candidate

    return dataclasses.replace(
        candidate,
        admission_status="failed",
        admission_error=error,
    )


def _runtime_binding_admission_error(
    candidate: Candidate,
    runtime_signature: Mapping[str, Any],
) -> str | None:
    if runtime_signature.get("runtime") != "standard":
        return None

    settings = candidate.settings

    return _first_runtime_binding_error((
        _fusion_binding_error(settings, runtime_signature),
        _batch_layout_binding_error(settings, runtime_signature),
        _parameter_surface_binding_error(settings, runtime_signature),
        _lm_head_binding_error(settings, runtime_signature),
        _mmap_binding_error(settings, runtime_signature),
        _intermediate_residency_binding_error(settings, runtime_signature),
        _activation_hook_binding_error(settings, runtime_signature),
        _checkpoint_context_binding_error(settings, runtime_signature),
        _teacher_objective_binding_error(settings, runtime_signature),
        _call_core_settings_binding_error(settings),
        _module_call_binding_error(settings, runtime_signature),
    ))


def _first_runtime_binding_error(errors: tuple[str | None, ...]) -> str | None:
    for error in errors:
        if error is not None:
            return error

    return None


def _fusion_binding_error(
    settings: Mapping[str, Any],
    runtime_signature: Mapping[str, Any],
) -> str | None:
    fused_values = {
        "fusion.norm": {"fused_rmsnorm", "fused_layernorm"},
        "fusion.mlp": {"fused_mlp"},
        "fusion.rope": {"fused_rope"},
        "fusion.logits": {"fused_logits_projection"},
        "fusion.loss": {"fused_ce", "fused_kl"},
    }

    for key, values in fused_values.items():
        if settings.get(key) not in values:
            continue

        if not _runtime_callback_present(runtime_signature, "fusion_rewriter"):
            return f"{key} requires a registered fused implementation"

        if runtime_signature.get("module") is not True:
            return "fused rows require a module"

    return None


def _batch_layout_binding_error(
    settings: Mapping[str, Any],
    runtime_signature: Mapping[str, Any],
) -> str | None:
    uses_batch_layout = (
        settings.get("input.batch_layout")
        in {"packed_with_inverse_permutation", "variable_length"}
        or settings.get("input.length_grouping") == "exact_length_bucket"
        or settings.get("schedule.per_token") == "packed"
    )

    if uses_batch_layout and not _runtime_callback_present(
        runtime_signature,
        "batch_layout",
    ):
        return "declared input layout requires a batch_layout binding"

    return None


def _parameter_surface_binding_error(
    settings: Mapping[str, Any],
    runtime_signature: Mapping[str, Any],
) -> str | None:
    for key in ("layout.params", "layout.vector", "layout.output"):
        value = settings.get(key)

        if value == "per_layer_flat" and not _runtime_has_parameter_groups(
            runtime_signature,
            "layer_groups",
        ):
            return f"{key}=per_layer_flat requires declared layer_groups"

        if value == "per_block_flat" and not _runtime_has_parameter_groups(
            runtime_signature,
            "block_groups",
        ):
            return f"{key}=per_block_flat requires declared block_groups"

    if "chunk.layer_block_size" not in settings:
        return None

    if _runtime_has_parameter_groups(runtime_signature, "layer_groups"):
        return None

    return "chunk.layer_block_size requires declared layer_groups"


def _runtime_has_parameter_groups(
    runtime_signature: Mapping[str, Any],
    group_field: str,
) -> bool:
    surface = runtime_signature.get("parameter_surface")

    if not isinstance(surface, Mapping):
        return False

    groups = surface.get(group_field)

    if isinstance(groups, tuple):
        return bool(groups)

    if isinstance(groups, list):
        return bool(groups)

    return False


def _lm_head_binding_error(
    settings: Mapping[str, Any],
    runtime_signature: Mapping[str, Any],
) -> str | None:
    if "chunk.lm_head_weight_chunk_bytes" in settings and not _runtime_callback_present(
        runtime_signature, "lm_head_chunker"
    ):
        return "chunk.lm_head_weight_chunk_bytes requires an LM-head chunker binding"

    return None


def _mmap_binding_error(
    settings: Mapping[str, Any],
    runtime_signature: Mapping[str, Any],
) -> str | None:
    if _runtime_callback_present(runtime_signature, "mmap_residency"):
        return None

    for key in ("memory.vector_residency", "memory.factor_residency"):
        if settings.get(key) == "mmap_cpu":
            return f"{key}=mmap_cpu requires memory-mapped tensor metadata"

    return None


def _intermediate_residency_binding_error(
    settings: Mapping[str, Any],
    runtime_signature: Mapping[str, Any],
) -> str | None:
    if "memory.intermediate_residency" not in settings:
        return None

    operator = runtime_signature.get("operator")
    operator_kind = operator.get("kind") if isinstance(operator, Mapping) else None

    if operator_kind in {"composition", "ggnvp"}:
        return None

    return "memory.intermediate_residency requires named intermediate boundaries"


def _activation_hook_binding_error(
    settings: Mapping[str, Any],
    runtime_signature: Mapping[str, Any],
) -> str | None:
    if settings.get("activation.offload") != "custom_saved_tensor_hooks":
        return None

    pack_hook_id = settings.get("activation.pack_hook")
    unpack_hook_id = settings.get("activation.unpack_hook")

    if not isinstance(pack_hook_id, str) or not isinstance(unpack_hook_id, str):
        return "custom_saved_tensor_hooks requires activation pack and unpack hooks"

    pack_hooks = runtime_signature.get("activation_pack_hooks")
    unpack_hooks = runtime_signature.get("activation_unpack_hooks")

    if pack_hook_id not in _runtime_callback_ids(pack_hooks):
        return f"activation pack hook is not registered: {pack_hook_id}"

    if unpack_hook_id not in _runtime_callback_ids(unpack_hooks):
        return f"activation unpack hook is not registered: {unpack_hook_id}"

    return None


def _checkpoint_context_binding_error(
    settings: Mapping[str, Any],
    runtime_signature: Mapping[str, Any],
) -> str | None:
    if settings.get("checkpoint.context_fn") != "declared_context_pair":
        return None

    context_id = settings.get("checkpoint.context_fn_callable")

    if not isinstance(context_id, str):
        return "checkpoint.context_fn=declared_context_pair requires context id"

    checkpoint_contexts = runtime_signature.get("checkpoint_contexts")

    if context_id not in _runtime_callback_ids(checkpoint_contexts):
        return f"checkpoint context is not registered: {context_id}"

    return None


def _teacher_objective_binding_error(
    settings: Mapping[str, Any],
    runtime_signature: Mapping[str, Any],
) -> str | None:
    if settings.get(
        "teacher_outputs"
    ) == "recomputed_with_equality_check" and not _runtime_callback_present(
        runtime_signature, "teacher_objective"
    ):
        return "recomputed teacher outputs require a teacher objective"

    return None


def _runtime_callback_present(
    runtime_signature: Mapping[str, Any],
    key: str,
) -> bool:
    value = runtime_signature.get(key)

    return value is not None and value is not False


def _runtime_callback_ids(value: Any) -> tuple[str, ...]:
    if not isinstance(value, tuple):
        return ()

    return tuple(
        item["id"]
        for item in value
        if isinstance(item, Mapping) and isinstance(item.get("id"), str)
    )


def _call_core_settings_binding_error(settings: Mapping[str, Any]) -> str | None:
    try:
        admit_call_core_settings(settings)
    except AdmissionError as error:
        return str(error)

    return None


def _module_call_binding_error(
    settings: Mapping[str, Any],
    runtime_signature: Mapping[str, Any],
) -> str | None:
    if settings.get("call.path") != "stateful_module":
        return None

    if runtime_signature.get("module") is not True:
        return "call.path=stateful_module requires a module"

    if runtime_signature.get("module_call") is None:
        return "stateful_module requires a ModuleCallSpec binding"

    return None


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
    operations = tuple(
        runtime.operation_factory(candidate, batch, vector) for batch, vector in inputs
    )

    def operation() -> TensorTree:
        with deferred_runtime_finite_checks():
            return tuple(candidate_operation() for candidate_operation in operations)

    return operation


def _full_size_check(
    runtime: RuntimeConfig,
    candidate: Candidate,
    inputs: tuple[tuple[Batch, TensorTree], ...],
) -> Callable[[TensorTree, tuple[Measurement, ...]], Mapping[str, Any]] | None:
    full_size_check = runtime.full_size_check

    if full_size_check is None:
        return None

    def check(
        output: TensorTree,
        samples: tuple[Measurement, ...],
    ) -> Mapping[str, Any]:
        return full_size_check(candidate, inputs, output, samples)

    return check


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
        dependency_identities=dict(candidate.dependency_identities),
        cohort_assignment=dict(candidate.cohort_assignment),
    )


def _check_records(
    candidate: Candidate,
    input_signature: dict[str, Any],
    result: ReferenceResult,
) -> tuple[CheckRecord, ...]:
    records = []
    child_descriptors = []

    for child in result.child_results:
        child_records = _check_records(
            child.candidate,
            dict(child.input_signature),
            child.result,
        )
        records.extend(child_records)
        child_descriptors.append({
            "child_name": child.name,
            "row": child_records[-1].row_key(),
        })

    if child_descriptors:
        result = dataclasses.replace(
            result,
            measurements={
                **dict(result.measurements),
                "child_reference_rows": tuple(child_descriptors),
            },
            child_results=(),
        )

    records.append(_check_record(candidate, input_signature, result))

    return tuple(records)


def _with_parent_input_signature(
    result: ReferenceResult,
    input_signature: Mapping[str, Any],
) -> ReferenceResult:
    child_results = tuple(
        dataclasses.replace(
            child,
            input_signature={
                **dict(child.input_signature),
                "parent_input_signature": dict(input_signature),
            },
            result=_with_parent_input_signature(child.result, input_signature),
        )
        for child in result.child_results
    )

    return dataclasses.replace(result, child_results=child_results)


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

    record = FullSizeRecord(
        family=candidate.family,
        candidate_id=candidate.candidate_id,
        status="passed",
        input_signature=dict(input_signature),
        candidate_settings=dict(candidate.settings),
        generator_id=candidate.generator_id,
        generator_version=candidate.generator_version,
        timing_samples=samples,
        memory_samples=samples,
        output_signature=dict(state.output_signature),
        selection_metadata=dict(state.selection_metadata),
        dependency_identities=dict(candidate.dependency_identities),
        cohort_assignment=dict(candidate.cohort_assignment),
        reference_passed=True,
    )
    return record


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
    _write_unique_record(
        run_dir
        / "candidates"
        / candidate.family
        / candidate.candidate_id
        / "candidate.json",
        candidate_record_to_json(candidate, input_signature),
    )


def _write_check(run_dir: Path, record: CheckRecord) -> None:
    _write_unique_record(
        run_dir
        / "references"
        / record.family
        / record.candidate_id
        / f"{record.name}.json",
        check_record_to_json(record),
    )


def _write_full_size(run_dir: Path, record: FullSizeRecord) -> None:
    _write_unique_record(
        run_dir / "full_size" / record.family / record.candidate_id / "result.json",
        full_size_record_to_json(record),
    )


def _write_unique_record(path: Path, payload: dict[str, Any]) -> None:
    try:
        write_record_exclusive(path, payload)
    except FileExistsError:
        if _record_payload_matches(path, payload):
            return
    else:
        return

    for index in itertools.count(1):
        candidate_path = path.with_name(f"{path.stem}-{index:06d}{path.suffix}")

        try:
            write_record_exclusive(candidate_path, payload)
        except FileExistsError:
            if _record_payload_matches(candidate_path, payload):
                return
        else:
            return


def _record_payload_matches(path: Path, payload: dict[str, Any]) -> bool:
    return canonical_json(read_record(path)) == canonical_json(payload)


def _write_summary(run_dir: Path, plan: Plan) -> None:
    write_record(run_dir / "summaries" / "tuning.json", plan_to_json(plan))


def _run_dir_cache(run_dir: Path | None) -> _RunDirCache:
    if run_dir is None:
        return _RunDirCache({}, {}, (), ())

    check_records = tuple(
        check_record_from_json(read_record(path))
        for path in sorted((run_dir / "references").rglob("*.json"))
    )
    full_size_records = tuple(
        full_size_record_from_json(read_record(path))
        for path in sorted((run_dir / "full_size").rglob("*.json"))
    )
    candidates_by_input_key = {}

    for path in sorted((run_dir / "candidates").rglob("*.json")):
        record = read_record(path)
        candidate = candidate_record_from_json(record)
        key = _candidate_input_key(candidate, dict(record["input_signature"]))
        candidates_by_input_key.setdefault(key, []).append(candidate)

    checks_by_row_key = {
        canonical_json(record.row_key()): record for record in check_records
    }

    return _RunDirCache(
        {key: tuple(rows) for key, rows in candidates_by_input_key.items()},
        checks_by_row_key,
        check_records,
        full_size_records,
    )


def _candidate_input_key(
    candidate: Candidate,
    input_signature: Mapping[str, Any],
) -> str:
    return canonical_json({
        "family": candidate.family,
        "candidate_id": candidate.candidate_id,
        "input_signature": dict(input_signature),
        "settings": dict(candidate.settings),
        "dependency_identities": {
            family: dict(identity)
            for family, identity in sorted(candidate.dependency_identities.items())
        },
        "cohort_assignment": dict(candidate.cohort_assignment),
        "generator_id": candidate.generator_id,
        "generator_version": candidate.generator_version,
    })


def _check_matches_candidate(
    candidate: Candidate,
    record: CheckRecord,
    input_signature: Mapping[str, Any],
) -> bool:
    return (
        record.status == "passed"
        and record.family == candidate.family
        and record.candidate_id == candidate.candidate_id
        and canonical_json(record.input_signature) == canonical_json(input_signature)
        and canonical_json(record.candidate_settings)
        == canonical_json(candidate.settings)
        and canonical_json(record.dependency_identities)
        == canonical_json(candidate.dependency_identities)
        and canonical_json(record.cohort_assignment)
        == canonical_json(candidate.cohort_assignment)
        and record.generator_id == candidate.generator_id
        and record.generator_version == candidate.generator_version
    )


def _cached_reference_outcome(
    run_cache: _RunDirCache,
    candidate: Candidate,
    input_signature: dict[str, Any],
    *,
    include_full_size: bool,
) -> _ReferenceOutcome | None:
    parent_records = tuple(
        record
        for record in run_cache.check_records
        if _check_matches_candidate(candidate, record, input_signature)
    )

    if not parent_records:
        return None

    parent_record = parent_records[-1]
    child_rows = _cached_child_reference_rows(run_cache, parent_record)

    if child_rows is None:
        return None

    full_size_record = None

    if include_full_size:
        full_size_record = _cached_full_size_record(
            run_cache,
            candidate,
            input_signature,
        )

    return _ReferenceOutcome(
        candidates=child_rows[0],
        check_records=(*child_rows[1], parent_record),
        full_size_record=full_size_record,
        passed=True,
        cached=True,
    )


def _cached_child_reference_rows(
    run_cache: _RunDirCache,
    parent_record: CheckRecord,
) -> tuple[tuple[Candidate, ...], tuple[CheckRecord, ...]] | None:
    descriptors = parent_record.measurements.get("child_reference_rows", ())

    if not isinstance(descriptors, tuple | list):
        return None

    candidates = []
    check_records = []

    for descriptor in descriptors:
        child = _cached_child_reference_descriptor(run_cache, descriptor)

        if child is None:
            return None

        child_candidate, child_record = child
        child_rows = _cached_child_reference_rows(run_cache, child_record)

        if child_rows is None:
            return None

        candidates.append(child_candidate)
        candidates.extend(child_rows[0])
        check_records.extend(child_rows[1])
        check_records.append(child_record)

    return tuple(candidates), tuple(check_records)


def _cached_child_reference_descriptor(
    run_cache: _RunDirCache,
    descriptor: object,
) -> tuple[Candidate, CheckRecord] | None:
    match descriptor:
        case {"row": dict() as row_key}:
            pass
        case _:
            return None

    child_record = run_cache.checks_by_row_key.get(canonical_json(row_key))

    if child_record is None or child_record.status != "passed":
        return None

    child_candidate = _cached_candidate_for_check(run_cache, child_record)

    if child_candidate is None:
        return None

    return child_candidate, child_record


def _cached_candidate_for_check(
    run_cache: _RunDirCache,
    record: CheckRecord,
) -> Candidate | None:
    candidate = Candidate(
        family=record.family,
        candidate_id=record.candidate_id,
        settings=dict(record.candidate_settings),
        dependency_identities={
            family: dict(identity)
            for family, identity in record.dependency_identities.items()
        },
        cohort_assignment=dict(record.cohort_assignment),
        admission_status="passed",
        generator_id=record.generator_id,
        generator_version=record.generator_version,
    )
    key = _candidate_input_key(candidate, record.input_signature)
    candidates = run_cache.candidates_by_input_key.get(key, ())

    if not candidates:
        return None

    return candidates[-1]


def _cached_full_size_record(
    run_cache: _RunDirCache,
    candidate: Candidate,
    input_signature: Mapping[str, Any],
) -> FullSizeRecord | None:
    records = tuple(
        record
        for record in run_cache.full_size_records
        if record.reference_passed
        and record_matches_candidate(candidate, record)
        and canonical_json(record.input_signature) == canonical_json(input_signature)
    )

    if not records:
        return None

    passed_records = tuple(record for record in records if record.status == "passed")

    if passed_records:
        return passed_records[-1]

    return records[-1]


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


def _with_selected_composition_children(
    problem: Problem,
    dependencies: tuple[str, ...],
    selected: dict[str, Candidate],
    records: dict[str, FullSizeRecord],
    materializers: dict[str, Any],
    problems_by_family: Mapping[str, Problem],
) -> Problem:
    if problem.operator.kind != "composition":
        return problem

    child_order = _composition_child_order(problem.operator)

    if set(dependencies) != set(child_order):
        return problem

    runtime = _runtime(problem)
    children = tuple(
        _selected_composition_child(
            child,
            selected,
            records,
            materializers,
            problems_by_family,
        )
        for child in child_order
    )
    components = {child.name: child.component for child in children}
    anchor_components = {child.name: child.anchor_component for child in children}
    component_signature = {
        child: _dependency_identity(
            child,
            selected[child],
            records[child],
            materializers[child].identity(),
        )
        for child in child_order
    }
    bound_runtime = composition_runtime_config(
        problem.operator,
        components=components,
        anchor_components=anchor_components,
        children=children,
        candidates=runtime.candidates,
        thresholds=_composition_thresholds(runtime),
        component_signature=component_signature,
        anchor_component_signature=component_signature,
        axis_registry=runtime.axis_registry,
        numeric_bound_fields=_composition_numeric_bound_fields(runtime),
    )

    return dataclasses.replace(problem, runtime=bound_runtime)


def _with_run_dependencies(
    problem: Problem,
    dependencies: tuple[str, ...],
    selected: dict[str, Candidate],
    records: dict[str, FullSizeRecord],
    materializers: dict[str, Any],
    problems_by_family: Mapping[str, Problem],
) -> Problem:
    problem_with_dependencies = _with_dependency_identities(
        problem,
        dependencies,
        selected,
        records,
        materializers,
    )
    problem_with_dependencies = _with_selected_matrix_free_metric(
        problem_with_dependencies,
        dependencies,
        selected,
        records,
        materializers,
        problems_by_family,
    )

    return _with_selected_composition_children(
        problem_with_dependencies,
        dependencies,
        selected,
        records,
        materializers,
        problems_by_family,
    )


def _with_selected_matrix_free_metric(
    problem: Problem,
    dependencies: tuple[str, ...],
    selected: dict[str, Candidate],
    records: dict[str, FullSizeRecord],
    materializers: dict[str, Any],
    problems_by_family: Mapping[str, Problem],
) -> Problem:
    product = _matrix_free_metric_dependency(problem.operator)

    if product is None:
        return problem

    if dependencies != (product,):
        return problem

    selected_component = materializers[product](selected[product], records[product])

    if not callable(selected_component):
        message = (
            f"matrix_free dependency materializer returned non-callable: {product}"
        )
        raise MaterializationError(message)

    component = _selected_matrix_free_component(
        product,
        problems_by_family[product],
        selected_component,
    )
    binding_signature = {
        product: _dependency_identity(
            product,
            selected[product],
            records[product],
            materializers[product].identity(),
        )
    }
    runtime = standard_runtime_with_matrix_free_bindings(
        _runtime(problem),
        bindings={product: component},
        binding_signature=binding_signature,
    )

    return dataclasses.replace(problem, runtime=runtime)


def _selected_matrix_free_component(
    product: str,
    child_problem: Problem,
    selected_component: Callable[[Batch, TensorTree], TensorTree],
) -> Callable[[Batch, TensorTree], TensorTree]:
    def component(batch: Batch, vector: TensorTree) -> TensorTree:
        return selected_component(
            _matrix_free_dependency_batch(product, child_problem, batch),
            vector,
        )

    return component


def _matrix_free_dependency_batch(
    product: str,
    child_problem: Problem,
    batch: Batch,
) -> Batch:
    child_static_batch = child_problem.data.reference_batch(
        product,
        "matrix_free_dependency",
    )

    return {**dict(child_static_batch), **dict(batch)}


def _matrix_free_metric_dependency(operator: OperatorSpec) -> str | None:
    if operator.kind not in {
        "metric",
        "sqrt_metric",
        "inverse_sqrt_metric",
        "metric_inner",
        "inverse_metric",
        "inverse_metric_inner",
    }:
        return None

    representation = operator.semantics.get("representation")

    if not isinstance(representation, Mapping):
        return None

    if representation.get("kind") != "matrix_free":
        return None

    product = representation.get("operator")

    if not isinstance(product, str) or not product:
        message = "matrix_free metric requires a named sibling product"
        raise MaterializationError(message)

    return product


def _selected_composition_child(
    child: str,
    selected: dict[str, Candidate],
    records: dict[str, FullSizeRecord],
    materializers: dict[str, Any],
    problems_by_family: Mapping[str, Problem],
) -> CompositionChild:
    child_problem = problems_by_family[child]
    selected_component = materializers[child](selected[child], records[child])

    if not callable(selected_component):
        message = f"composition child materializer returned non-callable: {child}"
        raise MaterializationError(message)

    component = _composition_child_component(child, child_problem, selected_component)
    reference_check = _composition_child_reference_check(child, child_problem)

    return CompositionChild(
        name=child,
        candidate=selected[child],
        component=component,
        anchor_component=component,
        reference_check=reference_check,
        input_signature=records[child].input_signature,
    )


def _composition_child_component(
    child: str,
    child_problem: Problem,
    selected_component: Callable[[Batch, TensorTree], TensorTree],
) -> Callable[[Batch, TensorTree], TensorTree]:
    def component(batch: Batch, vector: TensorTree) -> TensorTree:
        return selected_component(
            _composition_child_batch(child, child_problem, batch),
            vector,
        )

    return component


def _composition_child_reference_check(
    child: str,
    child_problem: Problem,
) -> ReferenceCheck:
    child_runtime = _runtime(child_problem)

    def reference_check(
        candidate: Candidate,
        batch: Batch,
        vector: TensorTree,
    ) -> ReferenceResult:
        return child_runtime.reference_check(
            candidate,
            _composition_child_batch(child, child_problem, batch),
            vector,
        )

    return reference_check


def _composition_child_batch(
    child: str,
    child_problem: Problem,
    batch: Batch,
) -> Batch:
    child_static_batch = child_problem.data.reference_batch(
        child,
        "composition_child",
    )

    return {**dict(child_static_batch), **dict(batch)}


def _composition_child_order(operator: OperatorSpec) -> tuple[str, ...]:
    children = operator.semantics.get("children")

    if not isinstance(children, tuple):
        message = "composition operator must declare tuple children"
        raise MaterializationError(message)

    return children


def _composition_thresholds(runtime: RuntimeConfig) -> Mapping[str, float]:
    thresholds = runtime.signature.get("thresholds")

    if not isinstance(thresholds, Mapping):
        message = "composition runtime thresholds are missing"
        raise MaterializationError(message)

    return {
        key: value
        for key, value in thresholds.items()
        if isinstance(key, str) and isinstance(value, int | float)
    }


def _composition_numeric_bound_fields(runtime: RuntimeConfig) -> Mapping[str, Any]:
    fields = runtime.signature.get("numeric_bound_fields")

    if fields is None:
        return {}

    if not isinstance(fields, Mapping):
        message = "composition numeric bound fields are invalid"
        raise MaterializationError(message)

    return fields


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
    if problem.target.search_policy.strategy == "admission":
        plan = Plan(
            selected={},
            records={},
            input_signature=probe.input_signature,
            policy=problem.target.selection_policy,
            candidate_rows=probe.candidate_rows,
            target_identity=problem.target.signature(),
            runtime_identities={problem.operator.family: probe.runtime_identity},
            adapter_identities={
                problem.operator.family: dict(problem.adapter_identity)
            },
            run_dir=run_dir,
        )

        if run_dir is not None:
            _write_summary(run_dir, plan)

        return plan

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
        candidate_rows=probe.candidate_rows,
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


def _require_autobatch_search_strategy(target: Target) -> None:
    if target.search_policy.strategy in {"exhaustive", "fast"}:
        return

    message = (
        "autobatch domain search requires exhaustive or fast strategy: "
        f"{target.search_policy.strategy}"
    )
    raise MaterializationError(message)


def _measured_candidates_for_search(
    candidates: tuple[Candidate, ...],
    target: Target,
) -> tuple[Candidate, ...]:
    strategy = target.search_policy.strategy

    if strategy == "exhaustive":
        return candidates

    if strategy == "smoke":
        return _smoke_candidate_rows(candidates)

    message = (
        "search strategy requires a strategy-specific result shape: "
        f"{target.search_policy.strategy}"
    )
    raise MaterializationError(message)


def _smoke_candidate_rows(candidates: tuple[Candidate, ...]) -> tuple[Candidate, ...]:
    passed_baselines = tuple(
        candidate
        for candidate in candidates
        if not candidate.changed_axes and candidate.admission_status == "passed"
    )

    if len(passed_baselines) != 1:
        message = (
            "smoke search requires exactly one admitted baseline row "
            "with no changed axes"
        )
        raise MaterializationError(message)

    table = axis_table()
    selected = [passed_baselines[0]]
    seen_groups = set()

    for candidate in candidates:
        if not candidate.changed_axes or candidate.admission_status != "passed":
            continue

        group_key = _candidate_class_c_groups(candidate, table)

        if group_key in seen_groups:
            continue

        seen_groups.add(group_key)
        selected.append(candidate)

    return tuple(selected)


def _candidate_class_c_groups(
    candidate: Candidate,
    axis_table: AxisTable,
) -> tuple[str, ...]:
    by_key = axis_table.by_key()
    groups = []

    for axis_key in candidate.changed_axes:
        axis = by_key.get(axis_key)

        if axis is None:
            message = (
                f"smoke search changed axis has no Class C axis table group: {axis_key}"
            )
            raise MaterializationError(message)

        groups.append(axis.class_c_group)

    _add_merged_class_c_groups(groups, candidate.settings)

    return tuple(sorted(set(groups)))


def _add_merged_class_c_groups(
    groups: list[str],
    settings: Mapping[str, Any],
) -> None:
    if settings.get("attention.partition") == "packed_tokens":
        groups.append("input_schedule")

    boundary = settings.get("compile.boundary")

    if boundary == "attention_module":
        groups.append("attention_dispatch")

    if boundary in {
        "loss_closure",
        "gradient_closure",
        "jvp_closure",
        "vjp_closure",
        "hvp_single_vector",
        "hvp_batched_vectors",
        "ggn_jvp",
        "ggn_loss_hessian_product",
        "ggn_vjp",
        "ggn_full_product",
        "fisher_score_grad",
        "sampled_fisher_score_grad",
        "empirical_fisher_example_grad",
        "metric_multiply",
        "inverse_metric_solve",
        "composition_child",
    }:
        groups.append("ad_lowering")

    if any(
        key.startswith("fusion.") and value != "model_default"
        for key, value in settings.items()
    ):
        groups.append("ad_lowering")

    if any(key.startswith("dtensor.") for key in settings):
        groups.append("ad_lowering")

    if settings.get("fsdp.mp_policy.reduce_dtype") not in {None, "fp32"}:
        groups.append("numeric_backend")

    if settings.get("inverse_metric.solve_path") in {
        "factorized_solve",
        "woodbury_low_rank_solve",
    }:
        groups.append("metric_storage")

    if settings.get("inverse_metric.preconditioner") not in {None, "none"}:
        groups.append("metric_storage")


def _admission_probe_result(
    problem: Problem,
    *,
    run_dir: Path | None,
    memory_backend: MemoryBackend | None,
) -> _ProbeResult:
    runtime = _runtime(problem)
    candidates = tuple(
        _admit_target(candidate, problem.target)
        for candidate in _candidate_rows(runtime)
    )
    backend = _memory_backend(problem.target.devices, memory_backend)
    input_signature = _input_signature(problem, backend)

    if run_dir is not None:
        for candidate in candidates:
            _write_candidate(run_dir, input_signature, candidate)

    return _ProbeResult(
        candidates=candidates,
        candidate_rows=candidates,
        full_size_records=(),
        check_records=(),
        input_signature=input_signature,
        materializer=runtime.materializer,
        runtime_identity=runtime.identity(),
    )


def _probe_candidate_rows(
    context: _ProbeContext,
    candidates: tuple[Candidate, ...],
    compile_call_horizons: tuple[int, ...] = (),
) -> _CandidateProbeRows:
    candidate_rows = []
    records = []
    check_records = []

    for candidate in candidates:
        outcome = _reference_outcome(
            context.runtime,
            candidate,
            context.reference_batch,
            context.reference_vector,
            context.input_signature,
            context.run_dir,
            context.run_cache,
            include_full_size=True,
        )
        candidate_rows.extend(outcome.candidates)
        check_records.extend(outcome.check_records)
        if context.run_dir is not None and not outcome.cached:
            for check_record in outcome.check_records:
                _write_check(context.run_dir, check_record)

        if outcome.full_size_record is not None:
            records.append(outcome.full_size_record)
            if context.run_dir is not None and not outcome.cached:
                _write_full_size(context.run_dir, records[-1])

            continue

        operation = _measured_operation(
            context.runtime, candidate, context.probe_inputs
        )
        record = run_candidate(
            candidate,
            context.input_signature,
            operation,
            timing_policy=context.timing_policy,
            memory_backend=context.memory_backend,
            clock=context.clock,
            reference_passed=True,
            full_size_check=_full_size_check(
                context.runtime,
                candidate,
                context.probe_inputs,
            ),
        )
        record = _record_with_compile_horizon_scores(
            record,
            context.selection_policy,
            compile_call_horizons,
        )
        records.append(record)
        if context.run_dir is not None:
            _write_full_size(context.run_dir, records[-1])

    return _CandidateProbeRows(
        candidate_rows=tuple(candidate_rows),
        full_size_records=tuple(records),
        check_records=tuple(check_records),
    )


def _record_with_compile_horizon_scores(
    record: FullSizeRecord,
    policy: SelectionPolicy,
    compile_call_horizons: tuple[int, ...],
) -> FullSizeRecord:
    if not compile_call_horizons:
        return record

    if record.status != "passed":
        return record

    if record.candidate_settings.get("compile.enabled") != "true":
        return record

    scores = {
        str(horizon): selection_score_seconds(
            record,
            dataclasses.replace(policy, compile_call_horizon=horizon),
        )
        for horizon in compile_call_horizons
    }

    return dataclasses.replace(
        record,
        selection_metadata={
            **dict(record.selection_metadata),
            "compile_amortized_seconds_by_horizon": scores,
        },
    )


def _probe_fast_candidate_rows(
    context: _ProbeContext,
    candidates: tuple[Candidate, ...],
) -> tuple[tuple[Candidate, ...], _CandidateProbeRows]:
    eager_candidates = _fast_eager_candidate_rows(candidates)
    eager_rows = _probe_candidate_rows(
        context,
        candidates=eager_candidates,
    )
    top_eager = _near_fastest_candidates(
        eager_candidates,
        eager_rows.full_size_records,
        input_signature=context.input_signature,
        policy=context.selection_policy,
    )
    compile_candidates = _fast_compile_candidate_rows(candidates, top_eager)
    compile_rows = _probe_candidate_rows(
        context,
        candidates=compile_candidates,
    )

    return (
        (*eager_candidates, *compile_candidates),
        _CandidateProbeRows(
            candidate_rows=(*eager_rows.candidate_rows, *compile_rows.candidate_rows),
            full_size_records=(
                *eager_rows.full_size_records,
                *compile_rows.full_size_records,
            ),
            check_records=(*eager_rows.check_records, *compile_rows.check_records),
        ),
    )


def _probe_balanced_candidate_rows(
    context: _ProbeContext,
    candidates: tuple[Candidate, ...],
    *,
    retained_top_count: int | None,
    compile_call_horizons: tuple[int, ...] = (),
) -> tuple[tuple[Candidate, ...], _CandidateProbeRows]:
    if retained_top_count is None:
        message = "balanced search requires retained_top_count"
        raise MaterializationError(message)

    baseline = _single_admitted_baseline(candidates, strategy="balanced")
    grouped = _balanced_group_candidates(candidates)
    baseline_rows = _probe_candidate_rows(
        context,
        candidates=(baseline,),
        compile_call_horizons=compile_call_horizons,
    )
    retained = {}
    group_candidate_rows = []
    group_check_records = []

    for group_key, group_candidates in grouped.items():
        group_rows = _probe_balanced_group_rows(
            context,
            candidates=group_candidates,
            retained_top_count=retained_top_count,
        )
        retained[group_key] = group_rows.retained
        group_candidate_rows.extend(group_rows.candidate_rows)
        group_check_records.extend(group_rows.check_records)

    cross_candidates = _balanced_cross_candidate_rows(baseline, retained)
    synthetic_cross = tuple(
        candidate
        for candidate in cross_candidates
        if candidate.generator_id == "balanced"
    )

    if context.run_dir is not None:
        for candidate in synthetic_cross:
            _write_candidate(context.run_dir, context.input_signature, candidate)

    cross_rows = _probe_candidate_rows(
        context,
        candidates=cross_candidates,
    )
    top_cross = _near_fastest_candidates(
        cross_candidates,
        cross_rows.full_size_records,
        input_signature=context.input_signature,
        policy=context.selection_policy,
    )
    compile_candidates = _fast_compile_candidate_rows(candidates, top_cross)
    compile_rows = _probe_candidate_rows(
        context,
        candidates=compile_candidates,
    )

    return (
        (baseline, *cross_candidates, *compile_candidates),
        _CandidateProbeRows(
            candidate_rows=(
                *baseline_rows.candidate_rows,
                *group_candidate_rows,
                *synthetic_cross,
                *cross_rows.candidate_rows,
                *compile_rows.candidate_rows,
            ),
            full_size_records=(
                *baseline_rows.full_size_records,
                *cross_rows.full_size_records,
                *compile_rows.full_size_records,
            ),
            check_records=(
                *baseline_rows.check_records,
                *group_check_records,
                *cross_rows.check_records,
                *compile_rows.check_records,
            ),
        ),
    )


def _probe_thorough_candidate_rows(
    context: _ProbeContext,
    candidates: tuple[Candidate, ...],
    *,
    retained_top_count: int | None,
    compile_call_horizons: tuple[int, ...],
) -> tuple[tuple[Candidate, ...], _CandidateProbeRows]:
    balanced_candidates, balanced_rows = _probe_balanced_candidate_rows(
        context,
        candidates=candidates,
        retained_top_count=retained_top_count,
        compile_call_horizons=compile_call_horizons,
    )
    top_balanced = _near_fastest_candidates(
        balanced_candidates,
        balanced_rows.full_size_records,
        input_signature=context.input_signature,
        policy=context.selection_policy,
    )
    expanded_candidates = _thorough_expanded_candidate_rows(
        candidates,
        top_balanced,
        measured_candidates=balanced_candidates,
    )
    expanded_rows = _probe_candidate_rows(
        context,
        candidates=expanded_candidates,
        compile_call_horizons=compile_call_horizons,
    )

    return (
        (*balanced_candidates, *expanded_candidates),
        _CandidateProbeRows(
            candidate_rows=(
                *balanced_rows.candidate_rows,
                *expanded_rows.candidate_rows,
            ),
            full_size_records=(
                *balanced_rows.full_size_records,
                *expanded_rows.full_size_records,
            ),
            check_records=(
                *balanced_rows.check_records,
                *expanded_rows.check_records,
            ),
        ),
    )


def _thorough_expanded_candidate_rows(
    candidates: tuple[Candidate, ...],
    top_candidates: tuple[Candidate, ...],
    *,
    measured_candidates: tuple[Candidate, ...],
) -> tuple[Candidate, ...]:
    measured_ids = {candidate.candidate_id for candidate in measured_candidates}
    compile_candidates = _fast_compile_candidate_rows(candidates, top_candidates)
    group_candidates = _thorough_group_expansion_candidates(candidates, top_candidates)
    expanded = []

    for candidate in (*compile_candidates, *group_candidates):
        if candidate.candidate_id in measured_ids:
            continue

        if candidate in expanded:
            continue

        expanded.append(candidate)

    return tuple(expanded)


def _thorough_group_expansion_candidates(
    candidates: tuple[Candidate, ...],
    top_candidates: tuple[Candidate, ...],
) -> tuple[Candidate, ...]:
    table = axis_table()
    top_groups = set()

    for candidate in top_candidates:
        if candidate.changed_axes:
            top_groups.update(_candidate_class_c_groups(candidate, table))

    if not top_groups:
        return ()

    expanded = []

    for candidate in candidates:
        if candidate.admission_status != "passed" or not candidate.changed_axes:
            continue

        groups = set(_candidate_class_c_groups(candidate, table))

        if groups & top_groups:
            expanded.append(candidate)

    return tuple(expanded)


def _probe_balanced_group_rows(
    context: _ProbeContext,
    candidates: tuple[Candidate, ...],
    *,
    retained_top_count: int,
) -> _BalancedGroupRows:
    if not context.probe_inputs:
        message = "balanced search requires at least one full-size probe input"
        raise MaterializationError(message)

    contenders = []
    candidate_rows = []
    check_records = []
    records_by_id = {}

    for candidate in candidates:
        outcome = _reference_outcome(
            context.runtime,
            candidate,
            context.reference_batch,
            context.reference_vector,
            context.input_signature,
            context.run_dir,
            context.run_cache,
            include_full_size=True,
        )
        candidate_rows.extend(outcome.candidates)
        check_records.extend(outcome.check_records)

        if context.run_dir is not None and not outcome.cached:
            for check_record in outcome.check_records:
                _write_check(context.run_dir, check_record)

        if outcome.full_size_record is not None:
            records_by_id[candidate.candidate_id] = outcome.full_size_record
            contenders.append(candidate)
        elif outcome.passed:
            contenders.append(candidate)

    for stage_index, probe_input in enumerate(context.probe_inputs):
        if not contenders:
            break

        stage_records = []

        for candidate in contenders:
            previous = records_by_id.get(candidate.candidate_id)

            if previous is not None and _record_has_probe_stage(previous, stage_index):
                stage_records.append((candidate, previous))
                continue

            operation = _measured_operation(context.runtime, candidate, (probe_input,))
            record = run_candidate(
                candidate,
                context.input_signature,
                operation,
                timing_policy=context.timing_policy,
                memory_backend=context.memory_backend,
                clock=context.clock,
                reference_passed=True,
                full_size_check=_full_size_check(
                    context.runtime,
                    candidate,
                    (probe_input,),
                ),
            )
            record = _balanced_accumulated_record(
                previous,
                record,
            )
            records_by_id[candidate.candidate_id] = record
            stage_records.append((candidate, record))

            if context.run_dir is not None:
                _write_full_size(context.run_dir, record)

        contenders = list(
            _balanced_stage_survivors(
                tuple(stage_records),
                input_signature=context.input_signature,
                policy=context.selection_policy,
                retained_top_count=retained_top_count,
            )
        )

        if len(contenders) <= retained_top_count:
            break

    retained = _balanced_top_candidates(
        tuple(contenders),
        records_by_id,
        input_signature=context.input_signature,
        policy=context.selection_policy,
        retained_top_count=retained_top_count,
    )

    return _BalancedGroupRows(
        retained=retained,
        candidate_rows=tuple(candidate_rows),
        check_records=tuple(check_records),
    )


def _record_has_probe_stage(record: FullSizeRecord, stage_index: int) -> bool:
    return (
        len(record.timing_samples) > stage_index
        and len(record.memory_samples) > stage_index
    )


def _balanced_accumulated_record(
    previous: FullSizeRecord | None,
    current: FullSizeRecord,
) -> FullSizeRecord:
    if previous is None:
        return current

    return dataclasses.replace(
        current,
        timing_samples=(
            *previous.timing_samples,
            *current.timing_samples,
        ),
        memory_samples=(
            *previous.memory_samples,
            *current.memory_samples,
        ),
    )


def _balanced_stage_survivors(
    records: tuple[tuple[Candidate, FullSizeRecord], ...],
    *,
    input_signature: Mapping[str, object],
    policy: SelectionPolicy,
    retained_top_count: int,
) -> tuple[Candidate, ...]:
    accepted = _balanced_accepted(records, input_signature=input_signature)

    if not accepted:
        return ()

    ordered = _balanced_ordered_candidates(accepted, policy=policy)
    keep_count = max(retained_top_count, (len(ordered) + 1) // 2)

    return tuple(candidate for candidate, _ in ordered[:keep_count])


def _balanced_top_candidates(
    candidates: tuple[Candidate, ...],
    records_by_id: Mapping[str, FullSizeRecord],
    *,
    input_signature: Mapping[str, object],
    policy: SelectionPolicy,
    retained_top_count: int,
) -> tuple[Candidate, ...]:
    records = tuple(
        (candidate, records_by_id[candidate.candidate_id])
        for candidate in candidates
        if candidate.candidate_id in records_by_id
    )
    accepted = _balanced_accepted(records, input_signature=input_signature)
    ordered = _balanced_ordered_candidates(accepted, policy=policy)

    return tuple(candidate for candidate, _ in ordered[:retained_top_count])


def _balanced_accepted(
    records: tuple[tuple[Candidate, FullSizeRecord], ...],
    *,
    input_signature: Mapping[str, object],
) -> tuple[tuple[Candidate, FullSizeRecord], ...]:
    return tuple(
        (candidate, record)
        for candidate, record in records
        if record_matches_candidate(candidate, record)
        and record_accepted(record, input_signature)
    )


def _balanced_ordered_candidates(
    records: tuple[tuple[Candidate, FullSizeRecord], ...],
    *,
    policy: SelectionPolicy,
) -> tuple[tuple[Candidate, FullSizeRecord], ...]:
    return tuple(
        sorted(
            records,
            key=lambda item: (
                selection_score_seconds(item[1], policy),
                selection_memory_mib(item[1], policy),
            ),
        )
    )


def _balanced_group_candidates(
    candidates: tuple[Candidate, ...],
) -> dict[tuple[str, ...], tuple[Candidate, ...]]:
    groups = {}
    table = axis_table()

    for candidate in candidates:
        if (
            candidate.admission_status != "passed"
            or not candidate.changed_axes
            or candidate.settings.get("compile.enabled") == "true"
        ):
            continue

        _validate_balanced_delta(candidate)
        group_key = _candidate_class_c_groups(candidate, table)
        groups.setdefault(group_key, []).append(candidate)

    return {group: tuple(rows) for group, rows in groups.items()}


def _validate_balanced_delta(candidate: Candidate) -> None:
    changed = set(candidate.changed_axes)
    extra = tuple(key for key in candidate.settings if key not in changed)

    if extra:
        message = (
            "balanced search group rows must put every setting key in changed_axes: "
            f"{candidate.candidate_id}"
        )
        raise MaterializationError(message)


def _balanced_cross_candidate_rows(
    baseline: Candidate,
    retained: Mapping[tuple[str, ...], tuple[Candidate, ...]],
) -> tuple[Candidate, ...]:
    retained_groups = tuple(rows for rows in retained.values() if rows)

    if not retained_groups:
        return ()

    crossed = []

    for combination in itertools.product(*retained_groups):
        if len(combination) == 1:
            crossed.append(combination[0])
            continue

        crossed.append(_balanced_cross_candidate(baseline, combination))

    return tuple(crossed)


def _balanced_cross_candidate(
    baseline: Candidate,
    combination: tuple[Candidate, ...],
) -> Candidate:
    settings = dict(baseline.settings)
    changed_axes = []

    for candidate in combination:
        _validate_cross_context(baseline, candidate)
        settings.update(candidate.settings)
        changed_axes.extend(candidate.changed_axes)

    candidate_ids = tuple(candidate.candidate_id for candidate in combination)

    return Candidate(
        family=baseline.family,
        candidate_id=f"balanced:{'+'.join(candidate_ids)}",
        settings=settings,
        changed_axes=tuple(dict.fromkeys(changed_axes)),
        dependency_identities=dict(baseline.dependency_identities),
        cohort_assignment=dict(baseline.cohort_assignment),
        admission_status="passed",
        generator_id="balanced",
        migration_source_id="+".join(candidate_ids),
    )


def _validate_cross_context(baseline: Candidate, candidate: Candidate) -> None:
    if candidate.family != baseline.family:
        message = "balanced cross row family differs from baseline"
        raise MaterializationError(message)

    if candidate.dependency_identities != baseline.dependency_identities:
        message = "balanced cross row dependency identities differ from baseline"
        raise MaterializationError(message)

    if candidate.cohort_assignment != baseline.cohort_assignment:
        message = "balanced cross row cohort assignment differs from baseline"
        raise MaterializationError(message)


def _fast_eager_candidate_rows(
    candidates: tuple[Candidate, ...],
) -> tuple[Candidate, ...]:
    _single_admitted_baseline(candidates, strategy="fast")

    return tuple(
        candidate for candidate in candidates if _fast_eager_candidate(candidate)
    )


def _single_admitted_baseline(
    candidates: tuple[Candidate, ...],
    *,
    strategy: str,
) -> Candidate:
    passed_baselines = tuple(
        candidate
        for candidate in candidates
        if not candidate.changed_axes and candidate.admission_status == "passed"
    )

    if len(passed_baselines) != 1:
        message = (
            f"{strategy} search requires exactly one admitted baseline row "
            "with no changed axes"
        )
        raise MaterializationError(message)

    return passed_baselines[0]


def _fast_eager_candidate(candidate: Candidate) -> bool:
    if candidate.admission_status != "passed":
        return False

    if candidate.settings.get("compile.enabled") == "true":
        return False

    if not candidate.changed_axes:
        return True

    return all(
        _fast_eager_axis_allowed(axis, candidate) for axis in candidate.changed_axes
    )


def _fast_eager_axis_allowed(axis_key: str, candidate: Candidate) -> bool:
    if axis_key == "compile.enabled":
        return candidate.settings.get("compile.enabled") == "false"

    if axis_key.startswith("compile."):
        return False

    prefix = axis_key.split(".", 1)[0]

    return prefix in {
        "gradient",
        "jvp",
        "vjp",
        "hvp",
        "ggn",
        "fisher",
        "sampled_fisher",
        "empirical_fisher",
        "composition",
        "vectorization",
        "dtype",
        "attention",
    }


def _near_fastest_candidates(
    candidates: tuple[Candidate, ...],
    records: tuple[FullSizeRecord, ...],
    *,
    input_signature: Mapping[str, object],
    policy: SelectionPolicy,
) -> tuple[Candidate, ...]:
    accepted = tuple(
        (candidate, record)
        for candidate, record in zip(candidates, records, strict=True)
        if record_matches_candidate(candidate, record)
        and record_accepted(record, input_signature)
    )

    if not accepted:
        return ()

    fastest = min(selection_score_seconds(record, policy) for _, record in accepted)

    return tuple(
        candidate
        for candidate, record in accepted
        if selection_score_seconds(record, policy)
        <= fastest * policy.near_fastest_multiplier
    )


def _fast_compile_candidate_rows(
    candidates: tuple[Candidate, ...],
    top_eager: tuple[Candidate, ...],
) -> tuple[Candidate, ...]:
    top_settings = {
        canonical_json(_settings_without_compile(candidate.settings))
        for candidate in top_eager
    }

    if not top_settings:
        return ()

    return tuple(
        candidate
        for candidate in candidates
        if candidate.admission_status == "passed"
        and candidate.settings.get("compile.enabled") == "true"
        and canonical_json(_settings_without_compile(candidate.settings))
        in top_settings
    )


def _settings_without_compile(settings: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in settings.items()
        if not key.startswith("compile.") and key != "memory.output_buffers"
    }


def _timing_policy_for_search(target: Target) -> TimingPolicy:
    if target.search_policy.strategy != "thorough":
        return target.timing_policy

    repeat_count = target.search_policy.variance_repeat_count

    if repeat_count is None:
        message = "thorough search requires variance_repeat_count"
        raise MaterializationError(message)

    return dataclasses.replace(
        target.timing_policy,
        short_measured_calls=max(
            target.timing_policy.short_measured_calls,
            repeat_count,
        ),
        medium_measured_calls=max(
            target.timing_policy.medium_measured_calls,
            repeat_count,
        ),
        long_measured_calls=max(
            target.timing_policy.long_measured_calls,
            repeat_count,
        ),
    )


def _validate_thorough_horizon(target: Target) -> None:
    if target.search_policy.strategy != "thorough":
        return

    horizon = target.selection_policy.compile_call_horizon

    if horizon in target.search_policy.compile_call_horizons:
        return

    message = "thorough search selection horizon must be declared"
    raise MaterializationError(message)


def _probe_problem(
    problem: Problem,
    *,
    run_dir: Path | None,
    memory_backend: MemoryBackend | None,
    clock: Callable[[], float],
) -> _ProbeResult:
    if problem.target.search_policy.strategy == "admission":
        return _admission_probe_result(
            problem,
            run_dir=run_dir,
            memory_backend=memory_backend,
        )

    runtime = _runtime(problem)
    domain = _single_autobatch_domain(runtime)

    if domain is not None:
        _require_autobatch_search_strategy(problem.target)

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
    if problem.target.search_policy.strategy == "smoke":
        probe_inputs = probe_inputs[:1]

    _validate_thorough_horizon(problem.target)
    backend = _memory_backend(problem.target.devices, memory_backend)
    input_signature = _input_signature(problem, backend)
    timing_policy = _timing_policy_for_search(problem.target)
    run_cache = _run_dir_cache(run_dir)
    context = _ProbeContext(
        runtime=runtime,
        reference_batch=reference_batch,
        reference_vector=reference_vector,
        probe_inputs=probe_inputs,
        input_signature=input_signature,
        timing_policy=timing_policy,
        selection_policy=problem.target.selection_policy,
        memory_backend=backend,
        clock=clock,
        run_dir=run_dir,
        run_cache=run_cache,
    )
    candidate_rows = list(candidates)

    if run_dir is not None:
        for candidate in candidates:
            _write_candidate(run_dir, input_signature, candidate)

    if problem.target.search_policy.strategy == "fast":
        measured_candidates, probed = _probe_fast_candidate_rows(
            context,
            candidates=candidates,
        )
    elif problem.target.search_policy.strategy == "balanced":
        measured_candidates, probed = _probe_balanced_candidate_rows(
            context,
            candidates=candidates,
            retained_top_count=problem.target.search_policy.retained_top_count,
        )
    elif problem.target.search_policy.strategy == "thorough":
        measured_candidates, probed = _probe_thorough_candidate_rows(
            context,
            candidates=candidates,
            retained_top_count=problem.target.search_policy.retained_top_count,
            compile_call_horizons=problem.target.search_policy.compile_call_horizons,
        )
    else:
        measured_candidates = _measured_candidates_for_search(
            candidates, problem.target
        )
        probed = _probe_candidate_rows(
            context,
            candidates=measured_candidates,
        )

    candidate_rows.extend(probed.candidate_rows)

    return _ProbeResult(
        candidates=measured_candidates,
        candidate_rows=tuple(candidate_rows),
        full_size_records=probed.full_size_records,
        check_records=probed.check_records,
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
    backend = _memory_backend(problem.target.devices, memory_backend)
    input_signature = _input_signature(problem, backend)
    run_cache = _run_dir_cache(run_dir)
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
        run_cache,
    )
    cached_probe = _cached_autobatch_probe_result(
        candidates=candidates,
        reference_rows=reference_rows,
        value_to_candidate=value_to_candidate,
        input_signature=input_signature,
        materializer=runtime.materializer,
        runtime_identity=runtime.identity(),
    )

    if cached_probe is not None:
        return cached_probe

    records.extend(
        record
        for record in reference_rows.full_size_records
        if not record.reference_passed
    )
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
            objective=domain.objective,
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

        if candidate.candidate_id == selected_candidate_id:
            record = dataclasses.replace(
                record,
                selection_metadata={
                    **dict(record.selection_metadata),
                    "source": "autobatch",
                    "selected": True,
                    "axis_name": domain.axis_name,
                    "value": value,
                    "objective": domain.objective,
                },
            )

        records.append(record)
        if run_dir is not None:
            _write_full_size(run_dir, record)

    return _ProbeResult(
        candidates=candidates,
        candidate_rows=(*candidates, *reference_rows.candidates),
        full_size_records=tuple(records),
        check_records=tuple(reference_rows.check_records),
        input_signature=input_signature,
        materializer=runtime.materializer,
        runtime_identity=runtime.identity(),
        autobatch_selected_id=selected_candidate_id,
    )


def _cached_autobatch_probe_result(
    *,
    candidates: tuple[Candidate, ...],
    reference_rows: _AutobatchReferenceRows,
    value_to_candidate: Mapping[int, Candidate],
    input_signature: dict[str, Any],
    materializer: Materializer,
    runtime_identity: Mapping[str, Any],
) -> _ProbeResult | None:
    cached_records = _autobatch_cached_records_by_value(
        reference_rows.full_size_records,
        value_to_candidate,
        reference_rows.passed_values,
        input_signature,
    )
    selected_candidate_id = _cached_autobatch_selected_candidate_id(
        cached_records,
        reference_rows.passed_values,
    )

    if selected_candidate_id is None:
        return None

    return _ProbeResult(
        candidates=candidates,
        candidate_rows=(*candidates, *reference_rows.candidates),
        full_size_records=reference_rows.full_size_records,
        check_records=tuple(reference_rows.check_records),
        input_signature=input_signature,
        materializer=materializer,
        runtime_identity=runtime_identity,
        autobatch_selected_id=selected_candidate_id,
    )


def _autobatch_cached_records_by_value(
    records: tuple[FullSizeRecord, ...],
    value_to_candidate: Mapping[int, Candidate],
    passed_values: tuple[int, ...],
    input_signature: Mapping[str, Any],
) -> dict[int, FullSizeRecord]:
    records_by_value = {}

    for value in passed_values:
        candidate = value_to_candidate[value]
        matching_records = tuple(
            record
            for record in records
            if record.reference_passed
            and record_matches_candidate(candidate, record)
            and canonical_json(record.input_signature)
            == canonical_json(input_signature)
        )

        if matching_records:
            records_by_value[value] = matching_records[-1]

    return records_by_value


def _cached_autobatch_selected_candidate_id(
    records_by_value: Mapping[int, FullSizeRecord],
    passed_values: tuple[int, ...],
) -> str | None:
    if set(records_by_value) != set(passed_values):
        return None

    selected = tuple(
        record
        for record in records_by_value.values()
        if record.selection_metadata.get("source") == "autobatch"
        and record.selection_metadata.get("selected") is True
    )

    if len(selected) != 1:
        return None

    return selected[0].candidate_id


def _autobatch_reference_rows(
    problem: Problem,
    runtime: RuntimeConfig,
    domain: AutobatchDomain,
    value_to_candidate: Mapping[int, Candidate],
    input_signature: dict[str, Any],
    run_dir: Path | None,
    run_cache: _RunDirCache,
) -> _AutobatchReferenceRows:
    reference_batch, reference_vector = _reference_input(problem)
    candidates = []
    passed_values = []
    records = []
    check_records = []

    for value in domain.values:
        candidate = value_to_candidate[value]

        if run_dir is not None:
            _write_candidate(run_dir, input_signature, candidate)

        outcome = _reference_outcome(
            runtime,
            candidate,
            reference_batch,
            reference_vector,
            input_signature,
            run_dir,
            run_cache,
            include_full_size=True,
        )
        candidates.extend(outcome.candidates)
        check_records.extend(outcome.check_records)
        if run_dir is not None and not outcome.cached:
            for check_record in outcome.check_records:
                _write_check(run_dir, check_record)

        if outcome.full_size_record is not None:
            records.append(outcome.full_size_record)
            if run_dir is not None and not outcome.cached:
                _write_full_size(run_dir, outcome.full_size_record)

        if outcome.passed:
            passed_values.append(value)

    return _AutobatchReferenceRows(
        candidates=tuple(candidates),
        passed_values=tuple(passed_values),
        full_size_records=tuple(records),
        check_records=tuple(check_records),
    )


def _reference_outcome(
    runtime: RuntimeConfig,
    candidate: Candidate,
    reference_batch: Batch,
    reference_vector: TensorTree,
    input_signature: dict[str, Any],
    run_dir: Path | None,
    run_cache: _RunDirCache,
    *,
    include_full_size: bool,
) -> _ReferenceOutcome:
    if candidate.admission_status == "failed":
        error = candidate.admission_error or (
            f"candidate admission failed: {candidate.candidate_id}"
        )

        return _ReferenceOutcome(
            candidates=(),
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

    cached_outcome = _cached_reference_outcome(
        run_cache,
        candidate,
        input_signature,
        include_full_size=include_full_size,
    )

    if cached_outcome is not None:
        return cached_outcome

    try:
        reference_result = runtime.reference_check(
            candidate,
            reference_batch,
            reference_vector,
        )
    except ReferenceFailedError as error:
        return _ReferenceOutcome(
            candidates=(),
            check_records=(
                _failed_check_record(
                    candidate,
                    input_signature,
                    error_type="ReferenceFailed",
                    error=str(error),
                ),
            ),
            full_size_record=failed_record(
                candidate,
                input_signature,
                error_type="ReferenceFailed",
                error=str(error),
                reference_passed=False,
            ),
            passed=False,
        )
    except RuntimeError as error:
        return _ReferenceOutcome(
            candidates=(),
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

    reference_result = _with_parent_input_signature(reference_result, input_signature)
    child_candidates = tuple(
        child_candidate
        for child_candidate, _ in _child_reference_candidates(reference_result)
    )
    _write_child_reference_candidates(run_dir, reference_result)

    return _ReferenceOutcome(
        candidates=child_candidates,
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
    check = _full_size_check(runtime, state.candidate, probe_inputs)

    try:
        if check is not None:
            state.selection_metadata = {
                **dict(state.selection_metadata),
                **dict(check(output, tuple(state.samples))),
            }
    except RuntimeError as error:
        state.error_type = type(error).__name__
        state.error = str(error)
        raise


def _problem_for_assignment(
    problem: Problem,
    assignment: CohortAssignment,
    constraints: tuple[CohortConstraint, ...],
    family_names: tuple[str, ...],
) -> Problem:
    runtime = _runtime(problem)
    candidates = tuple(
        _candidate_for_assignment(candidate, assignment)
        for candidate in runtime.candidates
        if candidate_matches_assignment(
            candidate, assignment, constraints, family_names
        )
    )

    return dataclasses.replace(
        problem,
        runtime=dataclasses.replace(runtime, candidates=candidates),
    )


def _candidate_for_assignment(
    candidate: Candidate,
    assignment: CohortAssignment,
) -> Candidate:
    if candidate.family in assignment.covered_families:
        return dataclasses.replace(candidate, cohort_assignment=assignment.signature())

    return candidate


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
) -> _PrerequisiteRows:
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
) -> _PrerequisiteRows:
    input_signature = problem.input_signature()
    candidates = []
    records = []

    for candidate in _candidate_rows(_runtime(problem)):
        admitted = _admit_target(candidate, problem.target)
        candidates.append(admitted)
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

    return _PrerequisiteRows(
        candidates=tuple(candidates),
        full_size_records=tuple(records),
    )


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
    probe_cache: MutableMapping[str, _ProbeResult],
) -> _AssignmentResult:
    selected = {}
    records = {}
    materializers = {}
    candidate_rows = ()
    full_size_records = ()
    check_records = ()
    input_signature = _run_input_signature(
        run_id=run.run_id,
        cohort_assignment=assignment.signature(),
    )
    runtime_identities = {}

    for family in ordered_families:
        if any(dependency not in selected for dependency in family.dependencies):
            prerequisite = _prerequisite_failed_records(
                problem=problems_by_family[family.name],
                family=family,
                assignment=assignment,
                selected=selected,
                records=records,
                materializers=materializers,
                constraints=run.cohort_constraints,
                family_names=family_names,
                run_dir=run_dir,
            )
            candidate_rows = (*candidate_rows, *prerequisite.candidates)
            full_size_records = (*full_size_records, *prerequisite.full_size_records)
            continue

        if problems_by_family[family.name].operator != family.operator:
            message = f"family operator differs from problem operator: {family.name}"
            raise MaterializationError(message)

        problem_for_assignment = _problem_for_assignment(
            _with_run_dependencies(
                problems_by_family[family.name],
                family.dependencies,
                selected,
                records,
                materializers,
                problems_by_family,
            ),
            assignment,
            run.cohort_constraints,
            family_names,
        )
        if not _runtime(problem_for_assignment).candidates:
            continue

        probe_key = _probe_cache_key(problem_for_assignment, memory_backend)
        probe = probe_cache.get(probe_key)

        if probe is None:
            probe = _probe_problem(
                problem_for_assignment,
                run_dir=run_dir,
                memory_backend=memory_backend,
                clock=clock,
            )
            probe_cache[probe_key] = probe
            candidate_rows = (*candidate_rows, *probe.candidate_rows)
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
            continue

        selected[family.name] = selected_candidate
        records[family.name] = selected_record
        materializers[family.name] = probe.materializer
        runtime_identities[family.name] = probe.runtime_identity

    if set(selected) != set(family_names):
        return _AssignmentResult(
            state=None,
            candidate_rows=candidate_rows,
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
            runtime_identities=runtime_identities,
        ),
        candidate_rows=candidate_rows,
        full_size_records=full_size_records,
        check_records=check_records,
    )


def _probe_cache_key(
    problem: Problem,
    memory_backend: MemoryBackend | None,
) -> str:
    backend = _memory_backend(problem.target.devices, memory_backend)

    return canonical_json({
        "input_signature": _input_signature(problem, backend),
        "candidates": tuple(
            candidate.signature() for candidate in _runtime(problem).candidates
        ),
    })


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
        NoPassedCandidateError: If no measured rows are produced.
    """
    index = _run_problem_index(run, run_dir)
    _validate_run_validators(run, index.family_names)

    if run.target.search_policy.strategy == "admission":
        return _tune_run_admission(
            run=run,
            index=index,
            run_dir=run_dir,
            memory_backend=memory_backend,
        )

    cohort_states = []
    candidate_rows = ()
    full_size_records = ()
    check_records = ()
    probe_cache = {}

    for assignment in cohort_assignments(run.cohort_constraints, index.family_names):
        result = _tune_cohort_assignment(
            run=run,
            assignment=assignment,
            ordered_families=index.ordered_families,
            family_names=index.family_names,
            problems_by_family=index.problems_by_family,
            run_dir=run_dir,
            memory_backend=memory_backend,
            clock=clock,
            probe_cache=probe_cache,
        )
        full_size_records = (*full_size_records, *result.full_size_records)
        candidate_rows = (*candidate_rows, *result.candidate_rows)
        check_records = (*check_records, *result.check_records)

        if result.state is not None:
            cohort_states.append(result.state)

    selected_cohort = select_cohort(
        tuple(state.cohort for state in cohort_states),
        families=index.family_names,
        policy=run.target.selection_policy,
    )
    selected_state = next(
        state for state in cohort_states if state.cohort == selected_cohort
    )
    selected = {family: selected_cohort[family][0] for family in index.family_names}
    records = {family: selected_cohort[family][1] for family in index.family_names}
    materializers = {
        family: selected_state.materializers[family] for family in index.family_names
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
        candidate_rows=candidate_rows,
        full_size_records=full_size_records,
        check_records=check_records,
        materializers=materializers,
        validation_order=index.family_names,
        dependencies_by_family={
            family.name: family.dependencies for family in index.ordered_families
        },
        cohort_assignment=selected_state.assignment,
        cohort_constraints=run.cohort_constraints,
        target_identity=run.target.signature(),
        runtime_identities={
            family: dict(selected_state.runtime_identities[family])
            for family in index.family_names
        },
        adapter_identities={
            family: dict(index.problems_by_family[family].adapter_identity)
            for family in index.family_names
        },
        validation_required=len(run.validators) > 0,
        validator_identities={
            family: dict(run.validator_identities[family]) for family in run.validators
        },
        run_dir=run_dir,
    )

    _write_summary(run_dir, plan)

    if run.validators:
        plan = dataclasses.replace(
            plan,
            validation_records=validate_plan(plan, run.validators, run_dir=run_dir),
        )
        _write_summary(run_dir, plan)

    return plan


def _tune_run_admission(
    *,
    run: TuningRun,
    index: _RunProblemIndex,
    run_dir: Path,
    memory_backend: MemoryBackend | None,
) -> Plan:
    candidate_rows = ()
    input_signature = _run_input_signature(
        run_id=run.run_id,
        search_strategy="admission",
    )

    for assignment in cohort_assignments(run.cohort_constraints, index.family_names):
        for family in index.ordered_families:
            problem = _problem_for_assignment(
                index.problems_by_family[family.name],
                assignment,
                run.cohort_constraints,
                index.family_names,
            )
            probe = _admission_probe_result(
                problem,
                run_dir=run_dir,
                memory_backend=memory_backend,
            )
            candidate_rows = (*candidate_rows, *probe.candidate_rows)
            input_signature[f"{assignment.assignment_id}:{family.name}"] = dict(
                probe.input_signature
            )

    plan = Plan(
        selected={},
        records={},
        input_signature=input_signature,
        policy=run.target.selection_policy,
        candidate_rows=candidate_rows,
        target_identity=run.target.signature(),
        runtime_identities={
            family: index.problems_by_family[family].runtime.identity()
            for family in index.family_names
        },
        adapter_identities={
            family: dict(index.problems_by_family[family].adapter_identity)
            for family in index.family_names
        },
        run_dir=run_dir,
    )

    _write_summary(run_dir, plan)

    return plan


def _run_problem_index(run: TuningRun, run_dir: Path) -> _RunProblemIndex:
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

    return _RunProblemIndex(
        ordered_families=ordered_families,
        problems_by_family=problems_by_family,
        family_names=family_names,
    )


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


def materialize(
    plan: Plan,
    *,
    name: str | None = None,
    family: str | None = None,
) -> Any:
    """Return selected materialization data for a plan."""
    return plan.materialize(family, name=name)


def load_tuned_plan(
    run_dir: Path,
    problem: Problem,
    *,
    memory_backend: MemoryBackend | None = None,
) -> Plan:
    """Load and replay a saved single-family plan for a problem.

    Returns:
        Replayed selected plan.

    """
    runtime = _runtime(problem)
    family = problem.operator.family

    return load_plan(
        run_dir,
        replay_context=_replay_context_for_problem(
            problem,
            runtime=runtime,
            memory_backend=memory_backend,
        ),
        materializers={family: runtime.materializer},
    )


def load_tuned_run(
    run_dir: Path,
    run: TuningRun,
    *,
    memory_backend: MemoryBackend | None = None,
) -> Plan:
    """Load and replay a saved multi-family run.

    Returns:
        Replayed selected plan.
    """
    index = _run_problem_index(run, run_dir)
    _validate_run_validators(run, index.family_names)
    summary = read_record(run_dir / "summaries" / "tuning.json")
    selected = {
        str(family): candidate_from_signature(dict(candidate_record))
        for family, candidate_record in dict(summary["selected"]).items()
    }
    selected_records = _selected_records_for_summary(run_dir, summary)
    materializers = {
        family: _runtime(index.problems_by_family[family]).materializer
        for family in index.family_names
    }

    return load_plan(
        run_dir,
        replay_context=_replay_context_for_run(
            run,
            ordered_families=index.ordered_families,
            problems_by_family=index.problems_by_family,
            family_names=index.family_names,
            selected=selected,
            selected_records=selected_records,
            materializers=materializers,
            summary=summary,
            memory_backend=memory_backend,
        ),
        materializers=materializers,
    )


def _selected_records_for_summary(
    run_dir: Path,
    summary: Mapping[str, Any],
) -> dict[str, FullSizeRecord]:
    full_size_records = tuple(
        full_size_record_from_json(read_record(path))
        for path in sorted((run_dir / "full_size").rglob("*.json"))
    )
    full_size_by_key = {
        canonical_json(record.row_key()): record for record in full_size_records
    }
    selected_records = {}

    for family, row_key in dict(summary["records"]).items():
        record = full_size_by_key.get(canonical_json(row_key))

        if record is None:
            message = f"run replay selected full-size row is missing: {family}"
            raise MaterializationError(message)

        selected_records[str(family)] = record

    return selected_records


def _replay_context_for_run(
    run: TuningRun,
    *,
    ordered_families: tuple[Family, ...],
    problems_by_family: Mapping[str, Problem],
    family_names: tuple[str, ...],
    selected: Mapping[str, Candidate],
    selected_records: Mapping[str, FullSizeRecord],
    materializers: Mapping[str, Materializer],
    summary: Mapping[str, Any],
    memory_backend: MemoryBackend | None,
) -> ReplayContext:
    selected_so_far = {}
    records_so_far = {}
    family_input_signatures = {}
    runtime_identities = {}
    assignment = _cohort_assignment_from_record(summary["cohort_assignment"])

    for family in ordered_families:
        problem = problems_by_family[family.name]

        if problem.operator != family.operator:
            message = f"family operator differs from problem operator: {family.name}"
            raise MaterializationError(message)

        problem_with_dependencies = _with_dependency_identities(
            problem,
            family.dependencies,
            selected_so_far,
            records_so_far,
            dict(materializers),
        )
        problem_with_dependencies = _with_selected_composition_children(
            problem_with_dependencies,
            family.dependencies,
            selected_so_far,
            records_so_far,
            dict(materializers),
            problems_by_family,
        )
        problem_for_assignment = _problem_for_assignment(
            problem_with_dependencies,
            assignment,
            run.cohort_constraints,
            family_names,
        )
        backend = _memory_backend(problem_for_assignment.target.devices, memory_backend)
        family_input_signatures[family.name] = _input_signature(
            problem_for_assignment,
            backend,
        )
        runtime_identities[family.name] = _runtime(problem_for_assignment).identity()
        selected_so_far[family.name] = selected[family.name]
        records_so_far[family.name] = selected_records[family.name]

    return ReplayContext(
        input_signature={
            "run_id": run.run_id,
            "cohort_assignment": assignment.signature(),
            **family_input_signatures,
        },
        family_input_signatures=family_input_signatures,
        materializer_identities={
            family: dict(materializers[family].identity()) for family in family_names
        },
        selection_policy=run.target.selection_policy,
        target_identity=run.target.signature(),
        runtime_identities=runtime_identities,
        adapter_identities={
            family: dict(problems_by_family[family].adapter_identity)
            for family in family_names
        },
        validator_identities={
            family: dict(run.validator_identities[family]) for family in run.validators
        },
        validation_required=len(run.validators) > 0,
        validation_order=family_names,
    )


def _cohort_assignment_from_record(record: Mapping[str, Any]) -> CohortAssignment:
    return CohortAssignment(
        assignment_id=str(record["assignment_id"]),
        values=dict(record["values"]),
        constraints=tuple(str(name) for name in record["constraints"]),
        covered_families=tuple(str(family) for family in record["covered_families"]),
    )


def _replay_context_for_problem(
    problem: Problem,
    *,
    runtime: RuntimeConfig,
    memory_backend: MemoryBackend | None,
) -> ReplayContext:
    backend = _memory_backend(problem.target.devices, memory_backend)
    input_signature = _input_signature(problem, backend)
    family = problem.operator.family

    return ReplayContext(
        input_signature=input_signature,
        family_input_signatures={family: input_signature},
        materializer_identities={family: dict(runtime.materializer.identity())},
        selection_policy=problem.target.selection_policy,
        target_identity=problem.target.signature(),
        runtime_identities={family: runtime.identity()},
        adapter_identities={family: dict(problem.adapter_identity)},
        validation_order=(family,),
    )


def load_plan(
    run_dir: Path,
    *,
    replay_context: ReplayContext,
    materializers: Mapping[str, Materializer],
) -> Plan:
    """Load and replay a saved plan from a run directory.

    Returns:
        Replayed selected plan.
    """
    summary = read_record(run_dir / "summaries" / "tuning.json")
    full_size_records = tuple(
        full_size_record_from_json(read_record(path))
        for path in sorted((run_dir / "full_size").rglob("*.json"))
    )
    reference_rows = tuple(
        check_record_from_json(read_record(path))
        for path in sorted((run_dir / "references").rglob("*.json"))
    )
    check_records = tuple(
        record for record in reference_rows if record.name != "selected_plan_validation"
    )
    validation_records = tuple(
        record for record in reference_rows if record.name == "selected_plan_validation"
    )
    candidate_records = tuple(
        read_record(path) for path in sorted((run_dir / "candidates").rglob("*.json"))
    )
    validation_summary_path = run_dir / "summaries" / "selected_plan_validation.json"
    validation_summary = (
        read_record(validation_summary_path)
        if validation_summary_path.exists()
        else None
    )

    return plan_from_json(
        summary,
        replay_context=replay_context,
        full_size_records=full_size_records,
        check_records=check_records,
        candidate_records=candidate_records,
        materializers=materializers,
        validation_summary=validation_summary,
        validation_records=validation_records,
        run_dir=run_dir,
    )


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
        dependency_identities=dict(candidate.dependency_identities),
        cohort_assignment=dict(candidate.cohort_assignment),
    )
    return record


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
        dependency_identities=dict(candidate.dependency_identities),
        cohort_assignment=dict(candidate.cohort_assignment),
        error_type=error_type,
        error=error,
    )
    return record


def _write_validation(run_dir: Path, record: CheckRecord) -> None:
    write_record(
        run_dir
        / "references"
        / record.family
        / record.candidate_id
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
        ReferenceFailedError: If a selected family validator fails reference validation.
        RuntimeError: If a selected family validator fails validation.
    """
    selected_families = set(plan.selected)

    if set(validators) != selected_families:
        message = "selected-plan validators must match selected families"
        raise MaterializationError(message)

    plan.validate_dependency_identities()

    input_signature = selected_plan_validation_input_signature(plan)
    records = []

    validation_order = plan.validation_order or tuple(plan.selected)

    if set(validation_order) != selected_families:
        message = "selected-plan validation order must match selected families"
        raise MaterializationError(message)

    materialized = {}

    for family in validation_order:
        candidate = plan.selected[family]
        selected_record = plan.records[family]
        dependency_names = plan.dependencies_by_family.get(
            family,
            tuple(candidate.dependency_identities),
        )
        missing_dependencies = tuple(
            dependency
            for dependency in dependency_names
            if dependency not in materialized
        )

        if missing_dependencies:
            message = (
                "selected-plan validation order must place dependencies first: "
                f"{family}"
            )
            raise MaterializationError(message)

        selected_impl = materialize(plan, family=family)
        materialized[family] = selected_impl
        dependencies = {
            dependency: materialized[dependency] for dependency in dependency_names
        }
        context = PlanValidationContext(
            family=family,
            selected=selected_impl,
            dependencies=dependencies,
            materialized=dict(materialized),
        )

        try:
            result = validators[family](candidate, selected_record, context)
        except ReferenceFailedError as error:
            record = _failed_validation_record(
                candidate,
                input_signature,
                error_type="ReferenceFailed",
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
