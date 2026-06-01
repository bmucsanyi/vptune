"""Core data objects."""

import dataclasses
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

import torch

from vptune.identities import module_identity, record_content_hash, stable_hash
from vptune.tensor_tree import TensorTree

SCHEMA_VERSION = 1
PACKAGE_VERSION = "0.0.1"


Batch = Mapping[str, Any]
ParameterTree = dict[str, torch.Tensor]
BufferTree = dict[str, torch.Tensor]


class ScalarObjective(Protocol):
    """Callable protocol for scalar objectives."""

    def __call__(
        self,
        params: ParameterTree,
        buffers: BufferTree,
        batch: Batch,
        context: "ObjectiveContext",
    ) -> torch.Tensor:
        """Return a scalar tensor."""


class FunctionObjective(Protocol):
    """Callable protocol for tensor-tree objectives."""

    def __call__(
        self,
        params: ParameterTree,
        buffers: BufferTree,
        batch: Batch,
        context: "ObjectiveContext",
    ) -> TensorTree:
        """Return a tensor tree."""


class DataProvider(Protocol):
    """Protocol for reference and probe data."""

    def signature(self) -> Mapping[str, Any]:
        """Return stable data identity."""

    def reference_batch(self, family: str, check_name: str) -> Batch:
        """Return a deterministic reference batch."""

    def probe_batches(self, family: str) -> Sequence[Batch]:
        """Return full-size probe batches."""


class VectorProvider(Protocol):
    """Protocol for reference and probe vectors."""

    def signature(self) -> Mapping[str, Any]:
        """Return stable vector identity."""

    def reference_vectors(self, family: str) -> TensorTree:
        """Return deterministic reference vectors."""

    def probe_vectors(self, family: str) -> Sequence[TensorTree]:
        """Return full-size probe vectors."""


class CandidateOperation(Protocol):
    """Zero-argument operation produced for one candidate."""

    def __call__(self) -> TensorTree:
        """Run the candidate operation."""


class OperationFactory(Protocol):
    """Create a measured operation for one candidate."""

    def __call__(
        self,
        candidate: "Candidate",
        batch: Batch,
        vector: TensorTree,
    ) -> CandidateOperation:
        """Return the operation to measure."""


class ReferenceCheck(Protocol):
    """Check one candidate against an anchor."""

    def __call__(
        self,
        candidate: "Candidate",
        batch: Batch,
        vector: TensorTree,
    ) -> "ReferenceResult":
        """Return reference measurements or raise on failure."""


class CandidateAdmitter(Protocol):
    """Apply admission rules to a candidate row."""

    def admit(self, candidate: "Candidate") -> "Candidate":
        """Return candidate with admission status set."""

    def signature(self) -> Mapping[str, Any]:
        """Return stable admission identity."""


class Materializer(Protocol):
    """Build a selected implementation for one family."""

    def identity(self) -> Mapping[str, Any]:
        """Return stable materializer identity."""

    def __call__(
        self,
        candidate: "Candidate",
        record: "FullSizeRecord",
    ) -> Any:
        """Return the selected implementation."""


class MaterializerCallback(Protocol):
    """Callable wrapped by an identity-bearing materializer."""

    def __call__(
        self,
        candidate: "Candidate",
        record: "FullSizeRecord",
    ) -> Any:
        """Return the selected implementation."""


@dataclasses.dataclass(frozen=True, slots=True)
class CallableMaterializer:
    """Identity-bearing wrapper for selected implementation builders."""

    materializer_id: str
    materializer_version: str
    settings: Mapping[str, Any]
    callback: MaterializerCallback

    def identity(self) -> Mapping[str, Any]:
        """Return stable materializer identity."""
        return {
            "materializer_id": self.materializer_id,
            "materializer_version": self.materializer_version,
            "settings": dict(self.settings),
        }

    def __call__(
        self,
        candidate: "Candidate",
        record: "FullSizeRecord",
    ) -> Any:
        """Return the selected implementation."""
        return self.callback(candidate, record)


@dataclasses.dataclass(frozen=True, slots=True)
class PlanValidationContext:
    """Materialized selected implementations for plan validation."""

    family: str
    selected: Any
    dependencies: Mapping[str, Any]
    materialized: Mapping[str, Any]


class PlanValidator(Protocol):
    """Validate one selected family after plan materialization."""

    def __call__(
        self,
        candidate: "Candidate",
        record: "FullSizeRecord",
        context: PlanValidationContext,
    ) -> "ReferenceResult":
        """Return selected-plan validation measurements."""


@dataclasses.dataclass(frozen=True, slots=True)
class AutobatchDomain:
    """Explicit Autobatch-backed axis domain."""

    axis_name: str
    values: tuple[int, ...]
    settings_by_value: Mapping[int, Mapping[str, Any]]
    value_to_settings_id: str
    admission_identity: Mapping[str, Any]
    goal: str
    warmup_steps: int
    measure_steps: int
    devices: tuple[int, ...]
    cache_key_payload: Mapping[str, Any]

    def signature(self) -> dict[str, Any]:
        """Return stable domain identity."""
        return {
            "axis_name": self.axis_name,
            "values": self.values,
            "settings_by_value": {
                str(value): dict(self.settings_by_value[value]) for value in self.values
            },
            "value_to_settings_id": self.value_to_settings_id,
            "admission_identity": dict(self.admission_identity),
            "goal": self.goal,
            "warmup_steps": self.warmup_steps,
            "measure_steps": self.measure_steps,
            "devices": self.devices,
            "cache_key_payload": dict(self.cache_key_payload),
        }

    def settings_for_value(self, value: int) -> dict[str, Any]:
        """Return settings generated for one domain value.

        Raises:
            RuntimeError: If the domain value has no declared settings.
        """
        if value not in self.settings_by_value:
            message = f"autobatch domain value has no settings: {value}"
            raise RuntimeError(message)

        return dict(self.settings_by_value[value])


@dataclasses.dataclass(frozen=True, slots=True)
class RuntimeConfig:
    """Typed runtime settings for one problem."""

    candidates: tuple["Candidate", ...]
    operation_factory: OperationFactory
    reference_check: ReferenceCheck
    materializer: Materializer
    axis_registry: CandidateAdmitter | None
    signature: Mapping[str, Any]
    autobatch_domains: tuple[AutobatchDomain, ...] = ()

    def identity(self) -> dict[str, Any]:
        """Return stable runtime identity."""
        axis_signature = (
            None if self.axis_registry is None else self.axis_registry.signature()
        )

        return {
            **dict(self.signature),
            "axis_registry": axis_signature,
            "materializer": dict(self.materializer.identity()),
            "autobatch_domains": tuple(
                domain.signature() for domain in self.autobatch_domains
            ),
        }


@dataclasses.dataclass(frozen=True, slots=True)
class ObjectiveContext:
    """Runtime context passed to objective callables."""

    family: str
    candidate_id: str
    settings: Mapping[str, Any]


@dataclasses.dataclass(frozen=True, slots=True)
class TimingPolicy:
    """Policy for warmup and measured-call counts."""

    short_seconds: float = 60.0
    medium_seconds: float = 600.0
    short_warmups: int = 2
    short_measured_calls: int = 5
    medium_warmups: int = 1
    medium_measured_calls: int = 3
    long_warmups: int = 0
    long_measured_calls: int = 1

    def plan(self, elapsed_seconds: float) -> tuple[int, int]:
        """Return warmup and measured-call counts."""
        if elapsed_seconds < self.short_seconds:
            return self.short_warmups, self.short_measured_calls

        if elapsed_seconds < self.medium_seconds:
            return self.medium_warmups, self.medium_measured_calls

        return self.long_warmups, self.long_measured_calls


@dataclasses.dataclass(frozen=True, slots=True)
class SelectionPolicy:
    """Policy for candidate selection."""

    near_fastest_multiplier: float = 1.05
    speed_statistic: str = "median_elapsed_seconds"
    tie_breaker: str = "min_peak_reserved_mib"
    cohort_speed_statistic: str = "sum_median_elapsed_seconds"
    cohort_tie_breaker: str = "sum_peak_reserved_mib"


@dataclasses.dataclass(frozen=True, slots=True)
class CohortConstraint:
    """Constraint requiring covered families to share setting assignments."""

    name: str
    settings_keys: tuple[str, ...]
    assignments: tuple[Mapping[str, Any], ...]
    families: tuple[str, ...] = ()
    dependency_inheritance: str = "covered_families"
    selection_aggregation: str = "sum_median_elapsed_seconds"

    def signature(self) -> dict[str, Any]:
        """Return stable cohort-constraint identity."""
        return {
            "name": self.name,
            "settings_keys": self.settings_keys,
            "assignments": tuple(dict(assignment) for assignment in self.assignments),
            "families": self.families,
            "dependency_inheritance": self.dependency_inheritance,
            "selection_aggregation": self.selection_aggregation,
        }


@dataclasses.dataclass(frozen=True, slots=True)
class CohortAssignment:
    """One concrete cohort assignment."""

    assignment_id: str
    values: Mapping[str, Any]
    constraints: tuple[str, ...]
    covered_families: tuple[str, ...]

    def signature(self) -> dict[str, Any]:
        """Return stable cohort-assignment identity."""
        return {
            "assignment_id": self.assignment_id,
            "values": dict(self.values),
            "constraints": self.constraints,
            "covered_families": self.covered_families,
        }


@dataclasses.dataclass(frozen=True, slots=True)
class Target:
    """Execution target and policy identity."""

    devices: tuple[str, ...]
    accelerator: str
    allowed_dtypes: tuple[str, ...]
    allowed_attention_impls: tuple[str, ...]
    allowed_sharding_modes: tuple[str, ...]
    timing_policy: TimingPolicy
    selection_policy: SelectionPolicy
    determinism_policy: Mapping[str, Any]
    environment_capture: Mapping[str, Any]

    def signature(self) -> dict[str, Any]:
        """Return a stable target identity."""
        return {
            "devices": self.devices,
            "accelerator": self.accelerator,
            "allowed_dtypes": self.allowed_dtypes,
            "allowed_attention_impls": self.allowed_attention_impls,
            "allowed_sharding_modes": self.allowed_sharding_modes,
            "timing_policy": dataclasses.asdict(self.timing_policy),
            "selection_policy": dataclasses.asdict(self.selection_policy),
            "determinism_policy": dict(self.determinism_policy),
            "environment": dict(self.environment_capture),
        }


@dataclasses.dataclass(frozen=True, slots=True)
class ParameterSurface:
    """Ordered parameter surface."""

    names: tuple[str, ...]
    shapes: tuple[tuple[int, ...], ...]
    trainable: tuple[bool, ...]
    buffer_policy: str = "include"
    tied_weights_policy: str = "preserve"
    parametrization_policy: str = "active"

    def signature(self) -> dict[str, Any]:
        """Return a stable parameter-surface identity."""
        return dataclasses.asdict(self)


def parameter_surface(
    model: torch.nn.Module,
    *,
    include: Callable[[str, torch.nn.Parameter], bool] | None = None,
    buffers: str = "include",
    tied_weights: str = "preserve",
) -> ParameterSurface:
    """Return a parameter surface from a module.

    Raises:
        RuntimeError: If the tied-weights policy is unsupported.
    """
    if tied_weights not in {"preserve", "deduplicate"}:
        message = f"unsupported tied-weights policy: {tied_weights}"
        raise RuntimeError(message)

    remove_duplicate = tied_weights == "deduplicate"
    pairs = tuple(
        (name, parameter)
        for name, parameter in model.named_parameters(remove_duplicate=remove_duplicate)
        if include is None or include(name, parameter)
    )

    return ParameterSurface(
        names=tuple(name for name, _ in pairs),
        shapes=tuple(tuple(parameter.shape) for _, parameter in pairs),
        trainable=tuple(bool(parameter.requires_grad) for _, parameter in pairs),
        buffer_policy=buffers,
        tied_weights_policy=tied_weights,
    )


@dataclasses.dataclass(frozen=True, slots=True)
class OperatorSpec:
    """Mathematical operator declaration."""

    family: str
    kind: str
    objective_id: str
    aggregation: str
    semantics: Mapping[str, Any] = dataclasses.field(default_factory=dict)
    parameter_surface: Mapping[str, Any] = dataclasses.field(default_factory=dict)
    data_axis: str = "batch"
    output_shape: Mapping[str, Any] = dataclasses.field(default_factory=dict)
    dtype_policy: Mapping[str, Any] = dataclasses.field(default_factory=dict)
    anchor_family: str = "builtin"
    randomness: Mapping[str, Any] = dataclasses.field(default_factory=dict)
    thresholds: Mapping[str, float] = dataclasses.field(default_factory=dict)

    def signature(self) -> dict[str, Any]:
        """Return a stable operator identity."""
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True, slots=True)
class Family:
    """One family in a tuning DAG."""

    name: str
    operator: OperatorSpec
    dependencies: tuple[str, ...] = ()
    candidate_generator: str = "default"
    materialization_rule: str = "selected_operator"


@dataclasses.dataclass(frozen=True, slots=True)
class Candidate:
    """One candidate settings row."""

    family: str
    candidate_id: str
    settings: Mapping[str, Any]
    changed_axes: tuple[str, ...] = ()
    dependency_identities: Mapping[str, Mapping[str, Any]] = dataclasses.field(
        default_factory=dict
    )
    cohort_assignment: Mapping[str, Any] = dataclasses.field(default_factory=dict)
    admission_status: str = "pending"
    admission_error: str | None = None
    generator_id: str = "default"
    generator_version: str = PACKAGE_VERSION
    migration_source_id: str | None = None

    def candidate_spec_hash(self) -> str:
        """Return stable candidate spec hash."""
        return stable_hash({
            "family": self.family,
            "candidate_id": self.candidate_id,
            "settings": dict(self.settings),
            "changed_axes": self.changed_axes,
            "dependency_identities": {
                family: dict(identity)
                for family, identity in sorted(self.dependency_identities.items())
            },
            "cohort_assignment": dict(self.cohort_assignment),
            "admission_status": self.admission_status,
            "admission_error": self.admission_error,
            "generator_id": self.generator_id,
            "generator_version": self.generator_version,
            "migration_source_id": self.migration_source_id,
        })

    def signature(self) -> dict[str, Any]:
        """Return candidate identity fields."""
        return {
            "family": self.family,
            "candidate_id": self.candidate_id,
            "settings": dict(self.settings),
            "changed_axes": self.changed_axes,
            "dependency_identities": {
                family: dict(identity)
                for family, identity in sorted(self.dependency_identities.items())
            },
            "cohort_assignment": dict(self.cohort_assignment),
            "admission_status": self.admission_status,
            "admission_error": self.admission_error,
            "generator_id": self.generator_id,
            "generator_version": self.generator_version,
            "migration_source_id": self.migration_source_id,
            "candidate_spec_hash": self.candidate_spec_hash(),
        }


@dataclasses.dataclass(frozen=True, slots=True)
class Measurement:
    """One timing and memory sample."""

    elapsed_seconds: float
    peak_allocated_mib: float
    peak_reserved_mib: float
    post_allocated_mib: float
    post_reserved_mib: float
    device_memory_used_mib: float | None = None
    rank: int = 0
    device: str = "cpu"

    def to_record(self) -> dict[str, Any]:
        """Return a JSON record."""
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True, slots=True)
class CheckRecord:
    """Reference check record."""

    family: str
    candidate_id: str
    name: str
    status: str
    input_signature: Mapping[str, Any]
    candidate_settings: Mapping[str, Any]
    thresholds: Mapping[str, float]
    measurements: Mapping[str, Any]
    generator_id: str
    generator_version: str
    owner_hash: str
    candidate_spec_hash: str
    dependency_identities: Mapping[str, Mapping[str, Any]] = dataclasses.field(
        default_factory=dict
    )
    cohort_assignment: Mapping[str, Any] = dataclasses.field(default_factory=dict)
    error_type: str | None = None
    error: str | None = None
    schema_version: int = SCHEMA_VERSION
    package_version: str = PACKAGE_VERSION
    content_hash: str = ""

    def computed_content_hash(self) -> str:
        """Return the current saved-content hash."""
        payload = dataclasses.asdict(self)
        payload["record_type"] = "reference"

        return record_content_hash(payload)


@dataclasses.dataclass(frozen=True, slots=True)
class ReferenceResult:
    """Reference check measurements."""

    name: str
    thresholds: Mapping[str, float]
    measurements: Mapping[str, Any]
    child_results: tuple["ReferenceChildResult", ...] = ()


@dataclasses.dataclass(frozen=True, slots=True)
class ReferenceChildResult:
    """Reference result for a child candidate."""

    candidate: Candidate
    input_signature: Mapping[str, Any]
    result: ReferenceResult


@dataclasses.dataclass(frozen=True, slots=True)
class FullSizeRecord:
    """Full-size candidate result."""

    family: str
    candidate_id: str
    status: str
    input_signature: Mapping[str, Any]
    candidate_settings: Mapping[str, Any]
    generator_id: str
    generator_version: str
    owner_hash: str
    candidate_spec_hash: str
    timing_samples: tuple[Measurement, ...] = ()
    memory_samples: tuple[Measurement, ...] = ()
    output_signature: Mapping[str, Any] = dataclasses.field(default_factory=dict)
    dependency_identities: Mapping[str, Mapping[str, Any]] = dataclasses.field(
        default_factory=dict
    )
    cohort_assignment: Mapping[str, Any] = dataclasses.field(default_factory=dict)
    reference_passed: bool = True
    error_type: str | None = None
    error: str | None = None
    schema_version: int = SCHEMA_VERSION
    package_version: str = PACKAGE_VERSION
    content_hash: str = ""

    def computed_content_hash(self) -> str:
        """Return the current saved-content hash."""
        payload = dataclasses.asdict(self)
        payload["record_type"] = "full_size"

        return record_content_hash(payload)

    def median_elapsed_seconds(self) -> float:
        """Return median elapsed seconds.

        Raises:
            RuntimeError: If the record has no timing samples.
        """
        values = sorted(sample.elapsed_seconds for sample in self.timing_samples)

        if not values:
            message = "passed full-size record has no timing samples"
            raise RuntimeError(message)

        midpoint = len(values) // 2

        if len(values) % 2 == 1:
            return values[midpoint]

        return 0.5 * (values[midpoint - 1] + values[midpoint])

    def peak_reserved_mib(self) -> float:
        """Return max peak reserved memory.

        Raises:
            RuntimeError: If the record has no memory samples.
        """
        if not self.memory_samples:
            message = "passed full-size record has no memory samples"
            raise RuntimeError(message)

        return max(sample.peak_reserved_mib for sample in self.memory_samples)


@dataclasses.dataclass(frozen=True, slots=True)
class Problem:
    """Single-family tuning problem."""

    model: torch.nn.Module
    params: ParameterSurface
    data: DataProvider
    operator: OperatorSpec
    vectors: VectorProvider
    target: Target
    runtime: RuntimeConfig
    anchor_policy: Mapping[str, Any] = dataclasses.field(default_factory=dict)
    replay_policy: Mapping[str, Any] = dataclasses.field(default_factory=dict)
    adapter_identity: Mapping[str, Any] = dataclasses.field(
        default_factory=lambda: {
            "adapter_id": "vptune.core",
            "adapter_version": PACKAGE_VERSION,
        }
    )

    def input_signature(self) -> dict[str, Any]:
        """Return the declared problem identity."""
        return {
            "model": module_identity(self.model),
            "params": self.params.signature(),
            "data": dict(self.data.signature()),
            "operator": self.operator.signature(),
            "vectors": dict(self.vectors.signature()),
            "target": self.target.signature(),
            "anchor_policy": dict(self.anchor_policy),
            "runtime": self.runtime.identity(),
            "replay_policy": dict(self.replay_policy),
            "adapter": dict(self.adapter_identity),
        }


@dataclasses.dataclass(frozen=True, slots=True)
class TuningRun:
    """Multi-family tuning run."""

    target: Target
    families: tuple[Family, ...]
    problems: tuple[Problem, ...] = ()
    validators: Mapping[str, PlanValidator] = dataclasses.field(default_factory=dict)
    validator_identities: Mapping[str, Mapping[str, Any]] = dataclasses.field(
        default_factory=dict
    )
    cohort_constraints: tuple[CohortConstraint, ...] = ()
    run_id: str = "default"


@dataclasses.dataclass(frozen=True, slots=True)
class ReplayContext:
    """Current identities required to replay a saved plan."""

    input_signature: Mapping[str, Any]
    family_input_signatures: Mapping[str, Mapping[str, Any]]
    materializer_identities: Mapping[str, Mapping[str, Any]]
    selection_policy: SelectionPolicy
    target_identity: Mapping[str, Any] = dataclasses.field(default_factory=dict)
    runtime_identities: Mapping[str, Mapping[str, Any]] = dataclasses.field(
        default_factory=dict
    )
    adapter_identities: Mapping[str, Mapping[str, Any]] = dataclasses.field(
        default_factory=dict
    )
    validator_identities: Mapping[str, Mapping[str, Any]] = dataclasses.field(
        default_factory=dict
    )
    validation_required: bool = False
    validation_order: tuple[str, ...] = ()

    def signature(self) -> dict[str, Any]:
        """Return stable replay identity."""
        return {
            "input_signature": dict(self.input_signature),
            "family_input_signatures": {
                family: dict(signature)
                for family, signature in sorted(self.family_input_signatures.items())
            },
            "materializer_identities": {
                family: dict(identity)
                for family, identity in sorted(self.materializer_identities.items())
            },
            "selection_policy": dataclasses.asdict(self.selection_policy),
            "target_identity": dict(self.target_identity),
            "runtime_identities": {
                family: dict(identity)
                for family, identity in sorted(self.runtime_identities.items())
            },
            "adapter_identities": {
                family: dict(identity)
                for family, identity in sorted(self.adapter_identities.items())
            },
            "validator_identities": {
                family: dict(identity)
                for family, identity in sorted(self.validator_identities.items())
            },
            "validation_required": self.validation_required,
            "validation_order": self.validation_order,
        }


@dataclasses.dataclass(frozen=True, slots=True)
class Plan:
    """Selected candidates and replay identity."""

    selected: Mapping[str, Candidate]
    records: Mapping[str, FullSizeRecord]
    input_signature: Mapping[str, Any]
    policy: SelectionPolicy
    full_size_records: tuple[FullSizeRecord, ...] = ()
    check_records: tuple[CheckRecord, ...] = ()
    materializers: Mapping[str, Materializer] = dataclasses.field(default_factory=dict)
    validation_order: tuple[str, ...] = ()
    dependencies_by_family: Mapping[str, tuple[str, ...]] = dataclasses.field(
        default_factory=dict
    )
    cohort_assignment: CohortAssignment | None = None
    cohort_constraints: tuple[CohortConstraint, ...] = ()
    target_identity: Mapping[str, Any] = dataclasses.field(default_factory=dict)
    runtime_identities: Mapping[str, Mapping[str, Any]] = dataclasses.field(
        default_factory=dict
    )
    adapter_identities: Mapping[str, Mapping[str, Any]] = dataclasses.field(
        default_factory=dict
    )
    validation_required: bool = False
    validator_identities: Mapping[str, Mapping[str, Any]] = dataclasses.field(
        default_factory=dict
    )
    run_dir: Path | None = None
    schema_version: int = SCHEMA_VERSION
    package_version: str = PACKAGE_VERSION

    def materializer_identities(self) -> dict[str, dict[str, Any]]:
        """Return selected materializer identities by family.

        Raises:
            RuntimeError: If a selected family has no materializer.
        """
        missing = tuple(
            family for family in self.selected if family not in self.materializers
        )

        if missing:
            message = f"selected families are missing materializers: {missing}"
            raise RuntimeError(message)

        return {
            family: dict(self.materializers[family].identity())
            for family in sorted(self.selected)
        }

    def selected_dependency_identities(self) -> dict[str, dict[str, dict[str, Any]]]:
        """Return selected dependency identities by family."""
        return {
            family: {
                dependency: dict(identity)
                for dependency, identity in sorted(
                    candidate.dependency_identities.items()
                )
            }
            for family, candidate in sorted(self.selected.items())
        }

    def selected_dependencies_by_family(self) -> dict[str, tuple[str, ...]]:
        """Return dependency names for every selected family."""
        return {
            family: tuple(self.dependencies_by_family.get(family, ()))
            for family in sorted(self.selected)
        }

    def selected_runtime_identities(self) -> dict[str, dict[str, Any]]:
        """Return runtime identities for selected families."""
        return {
            family: dict(self.runtime_identities.get(family, {}))
            for family in sorted(self.selected)
        }

    def selected_adapter_identities(self) -> dict[str, dict[str, Any]]:
        """Return adapter identities for selected families."""
        return {
            family: dict(self.adapter_identities.get(family, {}))
            for family in sorted(self.selected)
        }

    def selected_validator_identities(self) -> dict[str, dict[str, Any]]:
        """Return validator identities for selected families."""
        return {
            family: dict(self.validator_identities.get(family, {}))
            for family in sorted(self.validator_identities)
        }

    def owner_hash(self) -> str:
        """Return owner hash for the selected plan."""
        payload = {
            "selected": {
                family: candidate.signature()
                for family, candidate in sorted(self.selected.items())
            },
            "records": {
                family: record.owner_hash
                for family, record in sorted(self.records.items())
            },
            "full_size_records": tuple(
                record.owner_hash for record in self.full_size_records
            ),
            "full_size_record_content_hashes": tuple(
                record.computed_content_hash() for record in self.full_size_records
            ),
            "check_records": tuple(record.owner_hash for record in self.check_records),
            "check_record_content_hashes": tuple(
                record.computed_content_hash() for record in self.check_records
            ),
            "validation_required": self.validation_required,
            "validation_order": self.validation_order,
            "validator_identities": self.selected_validator_identities(),
            "dependencies_by_family": self.selected_dependencies_by_family(),
            "cohort_assignment": None
            if self.cohort_assignment is None
            else self.cohort_assignment.signature(),
            "cohort_constraints": tuple(
                constraint.signature() for constraint in self.cohort_constraints
            ),
            "selected_dependency_identities": self.selected_dependency_identities(),
            "materializer_identities": self.materializer_identities(),
            "target_identity": dict(self.target_identity),
            "runtime_identities": self.selected_runtime_identities(),
            "adapter_identities": self.selected_adapter_identities(),
            "input_signature": dict(self.input_signature),
            "policy": dataclasses.asdict(self.policy),
            "schema_version": self.schema_version,
            "package_version": self.package_version,
        }

        return stable_hash(payload)

    def to_record(self) -> dict[str, Any]:
        """Return the saved plan record."""
        return {
            "record_type": "summary",
            "schema_version": self.schema_version,
            "package_version": self.package_version,
            "owner_hash": self.owner_hash(),
            "input_signature": dict(self.input_signature),
            "candidate_settings": {
                family: dict(candidate.settings)
                for family, candidate in sorted(self.selected.items())
            },
            "status": "passed",
            "generator_id": "plan",
            "generator_version": self.package_version,
            "selected": {
                family: candidate.signature()
                for family, candidate in sorted(self.selected.items())
            },
            "records": {
                family: record.owner_hash
                for family, record in sorted(self.records.items())
            },
            "full_size_records": tuple(
                record.owner_hash for record in self.full_size_records
            ),
            "full_size_record_content_hashes": tuple(
                record.computed_content_hash() for record in self.full_size_records
            ),
            "check_records": tuple(record.owner_hash for record in self.check_records),
            "check_record_content_hashes": tuple(
                record.computed_content_hash() for record in self.check_records
            ),
            "validation_required": self.validation_required,
            "validation_order": self.validation_order,
            "validator_identities": self.selected_validator_identities(),
            "dependencies_by_family": self.selected_dependencies_by_family(),
            "cohort_assignment": None
            if self.cohort_assignment is None
            else self.cohort_assignment.signature(),
            "cohort_constraints": tuple(
                constraint.signature() for constraint in self.cohort_constraints
            ),
            "selected_dependency_identities": self.selected_dependency_identities(),
            "materializer_identities": self.materializer_identities(),
            "target_identity": dict(self.target_identity),
            "runtime_identities": self.selected_runtime_identities(),
            "adapter_identities": self.selected_adapter_identities(),
            "policy": dataclasses.asdict(self.policy),
        }
