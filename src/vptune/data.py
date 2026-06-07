"""Core data objects."""

import dataclasses
import itertools
import math
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

import torch

from vptune.errors import MaterializationError
from vptune.identities import (
    device_signature,
    module_identity,
    to_json_value,
)
from vptune.tensor_tree import TensorTree

SCHEMA_VERSION = 1
PACKAGE_VERSION = "1.0.0"
MIN_VARIANCE_REPEAT_COUNT = 2
MIN_LINEAR_DOMAIN_VALUES = 3


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


@dataclasses.dataclass(frozen=True, slots=True)
class ModuleCallSpec:
    """Explicit module call binding for stateful module rows."""

    positional_batch_keys: tuple[str, ...] = ()
    keyword_batch_keys: Mapping[str, str] = dataclasses.field(default_factory=dict)
    output_fields: Mapping[str, tuple[str | int, ...]] = dataclasses.field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        """Validate declared batch and output paths."""
        _require_string_tuple(self.positional_batch_keys, "positional_batch_keys")
        _require_string_mapping(self.keyword_batch_keys, "keyword_batch_keys")
        _require_output_field_paths(self.output_fields)

    def signature(self) -> Mapping[str, Any]:
        """Return stable module-call identity."""
        return {
            "positional_batch_keys": tuple(self.positional_batch_keys),
            "keyword_batch_keys": dict(sorted(self.keyword_batch_keys.items())),
            "output_fields": {
                key: tuple(path) for key, path in sorted(self.output_fields.items())
            },
        }


def _require_string_tuple(value: Any, label: str) -> None:
    if isinstance(value, tuple) and all(isinstance(item, str) for item in value):
        return

    message = f"{label} must be a tuple of strings"
    raise MaterializationError(message)


def _require_string_mapping(value: Any, label: str) -> None:
    if isinstance(value, Mapping) and all(
        isinstance(key, str) and isinstance(item, str) for key, item in value.items()
    ):
        return

    message = f"{label} must map strings to strings"
    raise MaterializationError(message)


def _selected_name(family: str | None, name: str | None) -> str | None:
    if family is not None and name is not None and family != name:
        message = "materialize name and family selectors differ"
        raise MaterializationError(message)

    if name is not None:
        return name

    return family


def _require_output_field_paths(
    value: Mapping[str, tuple[str | int, ...]],
) -> None:
    for key, path in value.items():
        if not isinstance(key, str):
            message = "output field names must be strings"
            raise MaterializationError(message)

        if not isinstance(path, tuple) or not path:
            message = f"output field path must be a nonempty tuple: {key}"
            raise MaterializationError(message)

        if not all(isinstance(item, (str, int)) for item in path):
            message = f"output field path items must be strings or integers: {key}"
            raise MaterializationError(message)


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


class OperationFactoryCallback(Protocol):
    """Callable payload for an identity-bearing operation factory."""

    def __call__(
        self,
        candidate: "Candidate",
        batch: Batch,
        vector: TensorTree,
    ) -> CandidateOperation:
        """Return the operation to measure."""


class OperationFactory(Protocol):
    """Create a measured operation for one candidate."""

    def __call__(
        self,
        candidate: "Candidate",
        batch: Batch,
        vector: TensorTree,
    ) -> CandidateOperation:
        """Return the operation to measure."""


class RuntimeOperationFactory(OperationFactory, Protocol):
    """Identity-bearing operation factory stored in a runtime config."""

    def identity(self) -> Mapping[str, Any]:
        """Return stable operation-factory identity."""


class ReferenceCheckCallback(Protocol):
    """Callable payload for an identity-bearing reference check."""

    def __call__(
        self,
        candidate: "Candidate",
        batch: Batch,
        vector: TensorTree,
    ) -> "ReferenceResult":
        """Return reference measurements or raise on failure."""


class ReferenceCheck(Protocol):
    """Check one candidate against an anchor."""

    def __call__(
        self,
        candidate: "Candidate",
        batch: Batch,
        vector: TensorTree,
    ) -> "ReferenceResult":
        """Return reference measurements or raise on failure."""


class RuntimeReferenceCheck(ReferenceCheck, Protocol):
    """Identity-bearing reference check stored in a runtime config."""

    def identity(self) -> Mapping[str, Any]:
        """Return stable reference-check identity."""


@dataclasses.dataclass(frozen=True, slots=True)
class CallableOperationFactory:
    """Identity-bearing operation factory for callback payloads."""

    factory_id: str
    factory_version: str
    settings: Mapping[str, Any]
    callback_identity: Mapping[str, Any]
    callback: OperationFactoryCallback

    def identity(self) -> Mapping[str, Any]:
        """Return stable operation-factory identity."""
        return {
            "factory_id": self.factory_id,
            "factory_version": self.factory_version,
            "settings": dict(self.settings),
            "callback": dict(self.callback_identity),
        }

    def __call__(
        self,
        candidate: "Candidate",
        batch: Batch,
        vector: TensorTree,
    ) -> CandidateOperation:
        """Return the operation to measure."""
        return self.callback(candidate, batch, vector)


@dataclasses.dataclass(frozen=True, slots=True)
class CallableReferenceCheck:
    """Identity-bearing reference check for callback payloads."""

    check_id: str
    check_version: str
    settings: Mapping[str, Any]
    callback_identity: Mapping[str, Any]
    callback: ReferenceCheckCallback

    def identity(self) -> Mapping[str, Any]:
        """Return stable reference-check identity."""
        return {
            "check_id": self.check_id,
            "check_version": self.check_version,
            "settings": dict(self.settings),
            "callback": dict(self.callback_identity),
        }

    def __call__(
        self,
        candidate: "Candidate",
        batch: Batch,
        vector: TensorTree,
    ) -> "ReferenceResult":
        """Return reference measurements or raise on failure."""
        return self.callback(candidate, batch, vector)


class FullSizeCheck(Protocol):
    """Check a measured full-size output before selection."""

    def identity(self) -> Mapping[str, Any]:
        """Return stable full-size check identity."""

    def __call__(
        self,
        candidate: "Candidate",
        inputs: tuple[tuple[Batch, TensorTree], ...],
        output: TensorTree,
        samples: tuple["Measurement", ...],
    ) -> Mapping[str, Any]:
        """Return selection metadata for the measured output."""


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
    """Callable payload for an identity-bearing materializer."""

    def __call__(
        self,
        candidate: "Candidate",
        record: "FullSizeRecord",
    ) -> Any:
        """Return the selected implementation."""


@dataclasses.dataclass(frozen=True, slots=True)
class CallableMaterializer:
    """Identity-bearing materializer for callback payloads."""

    materializer_id: str
    materializer_version: str
    settings: Mapping[str, Any]
    callback_identity: Mapping[str, Any]
    callback: MaterializerCallback

    def identity(self) -> Mapping[str, Any]:
        """Return stable materializer identity."""
        return {
            "materializer_id": self.materializer_id,
            "materializer_version": self.materializer_version,
            "settings": dict(self.settings),
            "callback": dict(self.callback_identity),
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
    min_value: int
    max_value: int
    initial_value: int
    growth: str
    values: tuple[int, ...]
    settings_by_value: Mapping[int, Mapping[str, Any]]
    value_to_settings_id: str
    admission_identity: Mapping[str, Any]
    objective: str
    failure_signals: tuple[str, ...]
    termination: str
    warmup_steps: int
    measure_steps: int
    devices: tuple[int, ...]
    cache_key_payload: Mapping[str, Any]

    def __post_init__(self) -> None:
        """Validate the finite integer domain."""
        _require_nonempty_string(self.axis_name, "axis_name")
        _require_positive_integer(self.min_value, "min_value")
        _require_positive_integer(self.max_value, "max_value")
        _require_positive_integer(self.initial_value, "initial_value")
        _require_autobatch_growth(self.growth)
        _require_autobatch_objective(self.objective)
        _require_autobatch_failure_signals(self.failure_signals)
        _require_autobatch_termination(self.termination)
        _require_autobatch_objective_termination(self.objective, self.termination)
        _require_autobatch_values(
            self.values,
            min_value=self.min_value,
            max_value=self.max_value,
            initial_value=self.initial_value,
            growth=self.growth,
        )
        _require_autobatch_settings(self.values, self.settings_by_value)
        _require_nonempty_string(self.value_to_settings_id, "value_to_settings_id")
        _require_nonnegative_integer(self.warmup_steps, "warmup_steps")
        _require_positive_integer(self.measure_steps, "measure_steps")
        _require_autobatch_devices(self.devices)

    def signature(self) -> dict[str, Any]:
        """Return stable domain identity."""
        return {
            "axis_name": self.axis_name,
            "min_value": self.min_value,
            "max_value": self.max_value,
            "initial_value": self.initial_value,
            "growth": self.growth,
            "values": self.values,
            "settings_by_value": {
                str(value): dict(self.settings_by_value[value]) for value in self.values
            },
            "value_to_settings_id": self.value_to_settings_id,
            "admission_identity": dict(self.admission_identity),
            "objective": self.objective,
            "failure_signals": self.failure_signals,
            "termination": self.termination,
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


def _require_nonempty_string(value: str, field: str) -> None:
    if not isinstance(value, str) or not value:
        message = f"AutobatchDomain {field} must be a nonempty string"
        raise RuntimeError(message)


def _require_positive_integer(value: int, field: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        message = f"AutobatchDomain {field} must be a positive integer"
        raise RuntimeError(message)


def _require_nonnegative_integer(value: int, field: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        message = f"AutobatchDomain {field} must be a nonnegative integer"
        raise RuntimeError(message)


def _require_autobatch_growth(growth: str) -> None:
    if growth not in {"doubling", "linear_step", "declared_sequence"}:
        message = f"AutobatchDomain growth is unsupported: {growth}"
        raise RuntimeError(message)


def _require_autobatch_objective(objective: str) -> None:
    if objective not in {"largest_passing", "fastest_passing"}:
        message = f"AutobatchDomain objective is unsupported: {objective}"
        raise RuntimeError(message)


def _require_autobatch_failure_signals(signals: tuple[str, ...]) -> None:
    expected = (
        "backend_rejection",
        "oom",
        "reference_failure",
        "runtime_failure",
    )

    if signals != expected:
        message = "AutobatchDomain failure_signals must match the supported signals"
        raise RuntimeError(message)


def _require_autobatch_termination(termination: str) -> None:
    if termination not in {"exhausted_declared_values", "bracketed_failure_frontier"}:
        message = f"AutobatchDomain termination is unsupported: {termination}"
        raise RuntimeError(message)


def _require_autobatch_objective_termination(
    objective: str,
    termination: str,
) -> None:
    if objective == "fastest_passing" and termination != "exhausted_declared_values":
        message = "fastest_passing requires exhausted_declared_values termination"
        raise RuntimeError(message)

    if objective == "largest_passing" and termination != "bracketed_failure_frontier":
        message = "largest_passing requires bracketed_failure_frontier termination"
        raise RuntimeError(message)


def _require_autobatch_values(
    values: tuple[int, ...],
    *,
    min_value: int,
    max_value: int,
    initial_value: int,
    growth: str,
) -> None:
    if not values:
        message = "AutobatchDomain values must be nonempty"
        raise RuntimeError(message)

    previous = None

    for value in values:
        _require_positive_integer(value, "values item")

        if previous is not None and value <= previous:
            message = "AutobatchDomain values must be strictly increasing"
            raise RuntimeError(message)

        previous = value

    if values[0] != min_value:
        message = "AutobatchDomain min_value must match the first value"
        raise RuntimeError(message)

    if values[0] != initial_value:
        message = "AutobatchDomain initial_value must match the first value"
        raise RuntimeError(message)

    if values[-1] != max_value:
        message = "AutobatchDomain max_value must match the last value"
        raise RuntimeError(message)

    if growth == "doubling":
        _require_doubling_values(values)
    elif growth == "linear_step":
        _require_linear_values(values)


def _require_doubling_values(values: tuple[int, ...]) -> None:
    for left, right in itertools.pairwise(values):
        if right != left * 2:
            message = "AutobatchDomain doubling values must double each step"
            raise RuntimeError(message)


def _require_linear_values(values: tuple[int, ...]) -> None:
    if len(values) < MIN_LINEAR_DOMAIN_VALUES:
        return

    step = values[1] - values[0]

    for left, right in itertools.pairwise(values[1:]):
        if right - left != step:
            message = "AutobatchDomain linear_step values must use one step size"
            raise RuntimeError(message)


def _require_autobatch_settings(
    values: tuple[int, ...],
    settings_by_value: Mapping[int, Mapping[str, Any]],
) -> None:
    if set(settings_by_value) != set(values):
        message = "AutobatchDomain settings_by_value must cover every value"
        raise RuntimeError(message)

    for value in values:
        settings = settings_by_value[value]

        if not settings:
            message = f"AutobatchDomain value has empty settings: {value}"
            raise RuntimeError(message)


def _require_autobatch_devices(devices: tuple[int, ...]) -> None:
    if not devices:
        message = "AutobatchDomain devices must be nonempty"
        raise RuntimeError(message)

    seen = set()

    for device in devices:
        _require_nonnegative_integer(device, "devices item")

        if device in seen:
            message = "AutobatchDomain devices must not contain duplicates"
            raise RuntimeError(message)

        seen.add(device)


@dataclasses.dataclass(frozen=True, slots=True)
class RuntimeConfig:
    """Typed runtime settings for one problem."""

    candidates: tuple["Candidate", ...]
    operation_factory: RuntimeOperationFactory
    reference_check: RuntimeReferenceCheck
    materializer: Materializer
    axis_registry: CandidateAdmitter | None
    signature: Mapping[str, Any]
    autobatch_domains: tuple[AutobatchDomain, ...] = ()
    full_size_check: FullSizeCheck | None = None
    reference_check_name: str = "tree_close"

    def __post_init__(self) -> None:
        """Validate runtime callables that affect replay identity."""
        _require_identity_method(self.operation_factory, "operation_factory")
        _require_identity_method(self.reference_check, "reference_check")

    def identity(self) -> dict[str, Any]:
        """Return stable runtime identity."""
        axis_signature = (
            None if self.axis_registry is None else self.axis_registry.signature()
        )
        full_size_check_identity = (
            None if self.full_size_check is None else self.full_size_check.identity()
        )

        return {
            **dict(self.signature),
            "operation_factory": _identity_payload(
                self.operation_factory.identity(),
                "operation_factory.identity",
            ),
            "reference_check": _identity_payload(
                self.reference_check.identity(),
                "reference_check.identity",
            ),
            "axis_registry": axis_signature,
            "materializer": dict(self.materializer.identity()),
            "autobatch_domains": tuple(
                domain.signature() for domain in self.autobatch_domains
            ),
            "full_size_check": full_size_check_identity,
            "reference_check_name": self.reference_check_name,
        }


def _require_identity_method(value: Any, name: str) -> None:
    identity = getattr(value, "identity", None)

    if not callable(identity):
        message = f"RuntimeConfig {name} must expose identity()"
        raise MaterializationError(message)

    _identity_payload(identity(), f"{name}.identity")


def _identity_payload(value: Mapping[str, Any], name: str) -> Any:
    try:
        return to_json_value(value)
    except TypeError as error:
        message = f"{name} must be JSON-compatible"
        raise MaterializationError(message) from error


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

    def __post_init__(self) -> None:
        """Validate timing policy fields.

        Raises:
            RuntimeError: If thresholds or call counts are invalid.
        """
        thresholds = (
            ("short_seconds", self.short_seconds),
            ("medium_seconds", self.medium_seconds),
        )

        for name, value in thresholds:
            if (
                isinstance(value, bool)
                or not isinstance(value, int | float)
                or not math.isfinite(value)
                or value < 0
            ):
                message = f"{name} must be finite and nonnegative"
                raise RuntimeError(message)

        if self.medium_seconds < self.short_seconds:
            message = "medium_seconds must be at least short_seconds"
            raise RuntimeError(message)

        warmups = (
            ("short_warmups", self.short_warmups),
            ("medium_warmups", self.medium_warmups),
            ("long_warmups", self.long_warmups),
        )

        for name, value in warmups:
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                message = f"{name} must be a nonnegative integer"
                raise RuntimeError(message)

        measured_calls = (
            ("short_measured_calls", self.short_measured_calls),
            ("medium_measured_calls", self.medium_measured_calls),
            ("long_measured_calls", self.long_measured_calls),
        )

        for name, value in measured_calls:
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                message = f"{name} must be a positive integer"
                raise RuntimeError(message)

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
    compiled_speed_statistic: str = "compile_amortized_steady_state_seconds"
    distributed_speed_statistic: str = "global_elapsed_seconds"
    rank_memory_reduction: str = "max_peak_reserved"
    tie_breaker: str = "min_peak_reserved_mib"
    cohort_speed_statistic: str = "sum_median_elapsed_seconds"
    cohort_tie_breaker: str = "sum_peak_reserved_mib"
    accepted_status: str = "passed_current_reference_full_size_agreement_stable_memory"
    compile_call_horizon: int = 1

    def __post_init__(self) -> None:
        """Validate policy fields.

        Raises:
            RuntimeError: If numeric policy fields are invalid.
        """
        if (
            isinstance(self.near_fastest_multiplier, bool)
            or not isinstance(self.near_fastest_multiplier, int | float)
            or not math.isfinite(self.near_fastest_multiplier)
            or self.near_fastest_multiplier < 1.0
        ):
            message = "near_fastest_multiplier must be finite and at least 1.0"
            raise RuntimeError(message)

        if (
            not isinstance(self.compile_call_horizon, int)
            or isinstance(self.compile_call_horizon, bool)
            or self.compile_call_horizon <= 0
        ):
            message = "compile_call_horizon must be positive"
            raise RuntimeError(message)


@dataclasses.dataclass(frozen=True, slots=True)
class SearchPolicy:
    """Policy for candidate search strategy."""

    strategy: str
    retained_top_count: int | None = None
    compile_call_horizons: tuple[int, ...] = ()
    variance_repeat_count: int | None = None

    def __post_init__(self) -> None:
        """Validate the search strategy.

        Raises:
            RuntimeError: If the search strategy is unsupported by the spec.
        """
        if self.strategy not in {
            "admission",
            "smoke",
            "fast",
            "balanced",
            "thorough",
            "exhaustive",
        }:
            message = f"unsupported search strategy: {self.strategy}"
            raise RuntimeError(message)

        if self.retained_top_count is not None and (
            not isinstance(self.retained_top_count, int)
            or isinstance(self.retained_top_count, bool)
            or self.retained_top_count <= 0
        ):
            message = "retained_top_count must be positive"
            raise RuntimeError(message)

        if (
            self.strategy in {"balanced", "thorough"}
            and self.retained_top_count is None
        ):
            message = f"{self.strategy} search requires retained_top_count"
            raise RuntimeError(message)

        if any(
            not isinstance(horizon, int) or isinstance(horizon, bool) or horizon <= 0
            for horizon in self.compile_call_horizons
        ):
            message = "compile_call_horizons must be positive"
            raise RuntimeError(message)

        if self.strategy == "thorough" and not self.compile_call_horizons:
            message = "thorough search requires compile_call_horizons"
            raise RuntimeError(message)

        if self.variance_repeat_count is not None and (
            not isinstance(self.variance_repeat_count, int)
            or isinstance(self.variance_repeat_count, bool)
            or self.variance_repeat_count < MIN_VARIANCE_REPEAT_COUNT
        ):
            message = (
                f"variance_repeat_count must be at least {MIN_VARIANCE_REPEAT_COUNT}"
            )
            raise RuntimeError(message)

        if self.strategy == "thorough" and self.variance_repeat_count is None:
            message = "thorough search requires variance_repeat_count"
            raise RuntimeError(message)


@dataclasses.dataclass(frozen=True, slots=True)
class CohortConstraint:
    """Constraint requiring covered families to share setting assignments."""

    name: str
    settings_keys: tuple[str, ...]
    assignments: tuple[Mapping[str, Any], ...]
    families: tuple[str, ...] = ()
    dependency_inheritance: str = "covered_families"
    selection_aggregation: str = "sum_median_elapsed_seconds"

    def __post_init__(self) -> None:
        """Validate supported cohort constraint modes.

        Raises:
            RuntimeError: If a cohort mode is unsupported.
        """
        if self.dependency_inheritance != "covered_families":
            message = (
                "unsupported cohort dependency inheritance: "
                f"{self.dependency_inheritance}"
            )
            raise RuntimeError(message)

        if self.selection_aggregation != "sum_median_elapsed_seconds":
            message = (
                "unsupported cohort selection aggregation: "
                f"{self.selection_aggregation}"
            )
            raise RuntimeError(message)

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
    allowed_attention_frontends: tuple[str, ...]
    allowed_sdpa_kernels: tuple[str, ...]
    allowed_sharding_modes: tuple[str, ...]
    timing_policy: TimingPolicy
    selection_policy: SelectionPolicy
    search_policy: SearchPolicy
    determinism_policy: Mapping[str, Any]
    environment_capture: Mapping[str, Any]

    def signature(self) -> dict[str, Any]:
        """Return a stable target identity."""
        return {
            "devices": self.devices,
            "device_signatures": tuple(
                device_signature(device) for device in self.devices
            ),
            "accelerator": self.accelerator,
            "allowed_dtypes": self.allowed_dtypes,
            "allowed_attention_frontends": self.allowed_attention_frontends,
            "allowed_sdpa_kernels": self.allowed_sdpa_kernels,
            "allowed_sharding_modes": self.allowed_sharding_modes,
            "timing_policy": dataclasses.asdict(self.timing_policy),
            "selection_policy": dataclasses.asdict(self.selection_policy),
            "search_policy": dataclasses.asdict(self.search_policy),
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
    layer_groups: tuple[tuple[str, ...], ...] = ()
    block_groups: tuple[tuple[str, ...], ...] = ()

    def __post_init__(self) -> None:
        """Validate declared parameter-surface fields.

        Raises:
            RuntimeError: If a declared policy or group is invalid.
        """
        if self.parametrization_policy != "active":
            message = "parametrization_policy must be active"
            raise RuntimeError(message)

        _validate_parameter_groups(self.names, self.layer_groups, "layer_groups")
        _validate_parameter_groups(self.names, self.block_groups, "block_groups")

    def signature(self) -> dict[str, Any]:
        """Return a stable parameter-surface identity."""
        return dataclasses.asdict(self)


def parameter_surface(
    model: torch.nn.Module,
    *,
    include: Callable[[str, torch.nn.Parameter], bool] | None = None,
    buffers: str = "include",
    tied_weights: str = "preserve",
    layer_groups: Sequence[Sequence[str]] = (),
    block_groups: Sequence[Sequence[str]] = (),
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
        trainable=tuple(parameter.requires_grad for _, parameter in pairs),
        buffer_policy=buffers,
        tied_weights_policy=tied_weights,
        layer_groups=_normalize_parameter_groups(layer_groups),
        block_groups=_normalize_parameter_groups(block_groups),
    )


def _normalize_parameter_groups(
    groups: Sequence[Sequence[str]],
) -> tuple[tuple[str, ...], ...]:
    return tuple(tuple(group) for group in groups)


def _validate_parameter_groups(
    names: tuple[str, ...],
    groups: tuple[tuple[str, ...], ...],
    label: str,
) -> None:
    if not groups:
        return

    if any(not group for group in groups):
        message = f"{label} must not contain empty groups"
        raise RuntimeError(message)

    flat = tuple(name for group in groups for name in group)

    if set(flat) != set(names):
        message = f"{label} must cover every parameter name exactly once"
        raise RuntimeError(message)

    if len(flat) != len(set(flat)):
        message = f"{label} must not contain duplicate parameter names"
        raise RuntimeError(message)


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
    batch_inputs: Mapping[str, tuple[str, ...]] = dataclasses.field(
        default_factory=lambda: {"reference": (), "operation": ()}
    )
    anchor_family: str = "builtin"
    randomness: Mapping[str, Any] = dataclasses.field(default_factory=dict)
    thresholds: Mapping[str, float] = dataclasses.field(default_factory=dict)

    def signature(self) -> dict[str, Any]:
        """Return a stable operator identity."""
        values = dataclasses.asdict(self)
        batch_inputs = values.pop("batch_inputs")
        signature = {key: to_json_value(value) for key, value in values.items()}
        signature["batch_inputs"] = {
            phase: list(keys) for phase, keys in sorted(batch_inputs.items())
        }

        return signature


@dataclasses.dataclass(frozen=True, slots=True)
class Family:
    """One family in a tuning DAG."""

    name: str
    operator: OperatorSpec
    dependencies: tuple[str, ...] = ()
    candidate_generator: str = "default"
    materialization_rule: str = "selected_operator"

    def __post_init__(self) -> None:
        """Derive dependencies from operator semantics.

        Raises:
            MaterializationError: If operator dependencies are invalid.
        """
        dependencies = operator_dependencies(self.operator)

        if not dependencies:
            return

        if self.dependencies:
            message = "family dependencies are derived from operator semantics"
            raise MaterializationError(message)

        object.__setattr__(self, "dependencies", dependencies)


def operator_dependencies(operator: OperatorSpec) -> tuple[str, ...]:
    """Return product dependencies declared by an operator."""
    if operator.kind == "composition":
        return _composition_dependencies(operator)

    dependency = _matrix_free_metric_dependency(operator)

    if dependency is None:
        return ()

    return (dependency,)


def _composition_dependencies(operator: OperatorSpec) -> tuple[str, ...]:
    children = operator.semantics.get("children")

    if not isinstance(children, Sequence) or isinstance(children, str):
        message = "composition operator must declare ordered children"
        raise MaterializationError(message)

    child_order = tuple(children)

    if not child_order:
        message = "composition operator must declare ordered children"
        raise MaterializationError(message)

    for child in child_order:
        if not isinstance(child, str) or not child:
            message = "composition children must be non-empty strings"
            raise MaterializationError(message)

    return child_order


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
    dependency_identities: Mapping[str, Mapping[str, Any]] = dataclasses.field(
        default_factory=dict
    )
    cohort_assignment: Mapping[str, Any] = dataclasses.field(default_factory=dict)
    error_type: str | None = None
    error: str | None = None
    schema_version: int = SCHEMA_VERSION
    package_version: str = PACKAGE_VERSION

    def row_key(self) -> dict[str, Any]:
        """Return direct fields that identify this reference row."""
        return {
            "family": self.family,
            "candidate_id": self.candidate_id,
            "name": self.name,
            "status": self.status,
            "input_signature": dict(self.input_signature),
            "candidate_settings": dict(self.candidate_settings),
            "thresholds": dict(self.thresholds),
            "dependency_identities": {
                family: dict(identity)
                for family, identity in sorted(self.dependency_identities.items())
            },
            "cohort_assignment": dict(self.cohort_assignment),
            "generator_id": self.generator_id,
            "generator_version": self.generator_version,
        }


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

    name: str
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
    timing_samples: tuple[Measurement, ...] = ()
    memory_samples: tuple[Measurement, ...] = ()
    output_signature: Mapping[str, Any] = dataclasses.field(default_factory=dict)
    selection_metadata: Mapping[str, Any] = dataclasses.field(default_factory=dict)
    dependency_identities: Mapping[str, Mapping[str, Any]] = dataclasses.field(
        default_factory=dict
    )
    cohort_assignment: Mapping[str, Any] = dataclasses.field(default_factory=dict)
    reference_passed: bool = True
    error_type: str | None = None
    error: str | None = None
    schema_version: int = SCHEMA_VERSION
    package_version: str = PACKAGE_VERSION

    def row_key(self) -> dict[str, Any]:
        """Return direct fields that identify this full-size row."""
        return {
            "family": self.family,
            "candidate_id": self.candidate_id,
            "status": self.status,
            "input_signature": dict(self.input_signature),
            "candidate_settings": dict(self.candidate_settings),
            "dependency_identities": {
                family: dict(identity)
                for family, identity in sorted(self.dependency_identities.items())
            },
            "cohort_assignment": dict(self.cohort_assignment),
            "generator_id": self.generator_id,
            "generator_version": self.generator_version,
        }

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
    candidate_rows: tuple[Candidate, ...] = ()
    full_size_records: tuple[FullSizeRecord, ...] = ()
    check_records: tuple[CheckRecord, ...] = ()
    validation_records: tuple[CheckRecord, ...] = ()
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

    def selected_candidate(self, family: str | None = None) -> Candidate:
        """Return the selected candidate for one family."""
        selected_family = self._selected_family(family)

        return self.selected[selected_family]

    def materialize(
        self,
        family: str | None = None,
        *,
        name: str | None = None,
    ) -> Any:
        """Return the materialized selected implementation.

        Raises:
            MaterializationError: If the family or materializer is missing.
        """
        self.validate_dependency_identities()
        selected_family = self._selected_family(_selected_name(family, name))
        candidate = self.selected[selected_family]
        record = self.records.get(selected_family)

        if record is None:
            message = f"selected record is missing: {selected_family}"
            raise MaterializationError(message)

        materializer = self.materializers.get(selected_family)

        if materializer is None:
            message = f"selected materializer is missing: {selected_family}"
            raise MaterializationError(message)

        return materializer(candidate, record)

    def validate_dependency_identities(self) -> None:
        """Validate selected dependency identity records.

        Raises:
            MaterializationError: If selected dependency identities are stale.
        """
        dependencies_by_family = self.selected_dependencies_by_family()

        if set(dependencies_by_family) != set(self.selected):
            message = "selected-plan dependencies must name every selected family"
            raise MaterializationError(message)

        for family, candidate in self.selected.items():
            dependencies = dependencies_by_family[family]
            record = self.records.get(family)

            if record is None:
                message = f"selected record is missing: {family}"
                raise MaterializationError(message)

            if set(candidate.dependency_identities) != set(dependencies):
                message = f"selected candidate dependencies differ: {family}"
                raise MaterializationError(message)

            if set(record.dependency_identities) != set(dependencies):
                message = f"selected record dependencies differ: {family}"
                raise MaterializationError(message)

            for dependency in dependencies:
                self._validate_dependency_identity(
                    family,
                    dependency,
                    candidate,
                    record,
                )

    def _validate_dependency_identity(
        self,
        family: str,
        dependency: str,
        candidate: Candidate,
        record: FullSizeRecord,
    ) -> None:
        if dependency not in self.selected:
            message = f"selected dependency is missing: {dependency}"
            raise MaterializationError(message)

        dependency_materializer = self.materializers.get(dependency)

        if dependency_materializer is None:
            message = f"selected dependency has no materializer: {dependency}"
            raise MaterializationError(message)

        dependency_record = self.records.get(dependency)

        if dependency_record is None:
            message = f"selected dependency has no record: {dependency}"
            raise MaterializationError(message)

        expected_identity = {
            "family": dependency,
            "candidate_id": self.selected[dependency].candidate_id,
            "candidate_settings": dict(self.selected[dependency].settings),
            "full_size_row": dependency_record.row_key(),
            "materializer_identity": dict(dependency_materializer.identity()),
        }

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

    def _selected_family(self, family: str | None) -> str:
        if family is not None:
            if family not in self.selected:
                message = f"selected family is missing: {family}"
                raise MaterializationError(message)

            return family

        if len(self.selected) != 1:
            message = "family is required for multi-family plans"
            raise MaterializationError(message)

        return next(iter(self.selected))

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

    def candidate_rows_for_record(self) -> tuple[Candidate, ...]:
        """Return candidate rows saved in the plan summary."""
        if self.candidate_rows:
            return self.candidate_rows

        candidates = {}

        for record in (*self.full_size_records, *self.check_records):
            candidate = self._candidate_for_result_record(record)
            signature = to_json_value(candidate.signature())
            candidates.setdefault(str(signature), candidate)

        return tuple(candidates[key] for key in sorted(candidates))

    def _candidate_for_result_record(
        self,
        record: FullSizeRecord | CheckRecord,
    ) -> Candidate:
        selected_candidate = self.selected.get(record.family)

        if (
            selected_candidate is not None
            and selected_candidate.candidate_id == record.candidate_id
            and to_json_value(selected_candidate.settings)
            == to_json_value(record.candidate_settings)
        ):
            return selected_candidate

        return Candidate(
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

    def to_record(self) -> dict[str, Any]:
        """Return the saved plan record."""
        return {
            "record_type": "summary",
            "schema_version": self.schema_version,
            "package_version": self.package_version,
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
            "candidate_rows": tuple(
                candidate.signature() for candidate in self.candidate_rows_for_record()
            ),
            "records": {
                family: record.row_key()
                for family, record in sorted(self.records.items())
            },
            "full_size_records": tuple(
                record.row_key() for record in self.full_size_records
            ),
            "check_records": tuple(record.row_key() for record in self.check_records),
            "validation_records": tuple(
                record.row_key() for record in self.validation_records
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
