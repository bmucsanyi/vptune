"""Typed public front door."""

import dataclasses
import itertools
import math
import time
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from operator import itemgetter
from pathlib import Path
from typing import Any, TypeGuard

import torch

from vptune import operators as _operator_builders
from vptune.candidates import (
    AxisDescriptor,
    axis_manifest,
    settings_product,
    standard_axis_descriptors,
    standard_axis_registry,
)
from vptune.checks import STANDARD_THRESHOLDS
from vptune.data import (
    PACKAGE_VERSION,
    Batch,
    BufferTree,
    Candidate,
    Family,
    FunctionObjective,
    ModuleCallSpec,
    OperatorSpec,
    ParameterSurface,
    ParameterTree,
    Plan,
    ReplayContext,
    ScalarObjective,
    SelectionPolicy,
    TensorTree,
    TimingPolicy,
    operator_dependencies,
    parameter_surface,
)
from vptune.data import CohortConstraint as _LowerCohortConstraint
from vptune.data import (
    Problem as _LowerProblem,
)
from vptune.data import (
    SearchPolicy as _SearchPolicy,
)
from vptune.data import (
    Target as _LowerTarget,
)
from vptune.data import (
    TuningRun as _LowerTuningRun,
)
from vptune.errors import AdmissionError, MaterializationError
from vptune.identities import (
    module_identity,
    stable_hash,
    tensor_signature,
    to_json_value,
)
from vptune.io import read_record
from vptune.measure import MemoryBackend
from vptune.run import load_plan as _load_plan
from vptune.run import tune as _tune_problem
from vptune.run import tune_run as _tune_run
from vptune.runtime import (
    composition_runtime_config,
    standard_operation_factory,
    standard_runtime_config,
)
from vptune.tensor_tree import tree_leaves

MIN_CLASS_LOGIT_DIMS = 2
SQUARE_MATRIX_DIMS = 2
SAMPLE_TABLE_MIN_DIMS = 2
WEIGHTED_COMBINE_TERM_DIMS = 2


@dataclasses.dataclass(frozen=True, slots=True)
class Model:
    """Typed model binding for public operators."""

    module: torch.nn.Module
    parameters: ParameterSurface
    call: ModuleCallSpec
    parameter_values: ParameterTree
    buffers: BufferTree

    def signature(self) -> Mapping[str, Any]:
        """Return stable model identity fields."""
        return {
            "module": module_identity(self.module),
            "parameters": self.parameters.signature(),
            "call": self.call.signature(),
        }


@dataclasses.dataclass(frozen=True, slots=True)
class Metric:
    """Typed metric declaration."""

    kind: str
    representation: Mapping[str, Any]
    batch: Mapping[str, Any]
    identity_fields: Mapping[str, Any]
    product_name: str | None = None

    def signature(self) -> Mapping[str, Any]:
        """Return stable metric identity fields."""
        return {
            "kind": self.kind,
            "representation": dict(self.representation),
            "inputs": dict(self.identity_fields),
            "product_name": self.product_name,
        }


@dataclasses.dataclass(frozen=True, slots=True)
class Damping:
    """Typed damping declaration."""

    kind: str
    value: float | Mapping[str, float]
    policy: str | None = None

    def signature(self) -> Mapping[str, Any]:
        """Return stable damping identity fields."""
        return {
            "kind": self.kind,
            "value": self.value,
            "policy": self.policy,
        }


@dataclasses.dataclass(frozen=True, slots=True)
class Output:
    """Typed model-output declaration."""

    field: str

    def signature(self) -> Mapping[str, Any]:
        """Return stable output identity fields."""
        return {"field": self.field}


@dataclasses.dataclass(frozen=True, slots=True)
class Case:
    """Typed reference or probe input."""

    batch: Batch | None = None
    vector: TensorTree | None = None


@dataclasses.dataclass(frozen=True, slots=True)
class DeterminismPolicy:
    """Typed deterministic-execution policy."""

    fields: Mapping[str, Any] = dataclasses.field(default_factory=dict)

    def signature(self) -> Mapping[str, Any]:
        """Return stable deterministic-execution policy fields."""
        return dict(self.fields)


@dataclasses.dataclass(frozen=True, slots=True)
class EnvironmentPolicy:
    """Typed target environment-capture policy."""

    fields: Mapping[str, Any] = dataclasses.field(default_factory=dict)

    def signature(self) -> Mapping[str, Any]:
        """Return stable target environment-capture fields."""
        return dict(self.fields)


@dataclasses.dataclass(frozen=True, slots=True)
class SearchStrategy:
    """Typed public search strategy."""

    policy: _SearchPolicy

    def signature(self) -> Mapping[str, Any]:
        """Return stable search strategy fields."""
        return dataclasses.asdict(self.policy)


@dataclasses.dataclass(frozen=True, slots=True)
class SearchSpace:
    """Typed public search space."""

    axes: Mapping[str, Sequence[Any]] = dataclasses.field(default_factory=dict)
    components: Sequence[Any] = ()

    def __post_init__(self) -> None:
        """Normalize axis values for deterministic candidate generation.

        Raises:
            MaterializationError: If an axis key or value list is invalid.
        """
        normalized = {}

        for key, values in self.axes.items():
            _require_nonempty_string(key, "search-space axis")
            normalized_values = tuple(values)

            if not normalized_values:
                message = f"search-space axis has no values: {key}"
                raise MaterializationError(message)

            normalized[key] = normalized_values

        object.__setattr__(self, "axes", normalized)
        object.__setattr__(self, "components", tuple(self.components))

    def candidate_settings(
        self, operator: "Operator"
    ) -> Mapping[str, Mapping[str, Any]]:
        """Return concrete candidate settings for one operator.

        Returns:
            Candidate settings keyed by candidate id.
        """
        default_settings = dict(operator.default_settings)
        fixed_settings = _operator_search_space_fixed_settings(self, operator)
        axes = _operator_search_space_axes(self, operator)
        axis_registry = _search_space_axis_registry(self)

        if not axes:
            return {
                f"{operator.spec.family}:typed_reference": _merge_search_settings(
                    default_settings,
                    fixed_settings,
                )
            }

        candidates = settings_product(
            operator.spec.family,
            axes,
            axis_registry=axis_registry,
            generator_id="vptune.public.space",
            generator_version=PACKAGE_VERSION,
        )

        return {
            candidate.candidate_id: _merge_search_settings(
                default_settings,
                fixed_settings,
                candidate.settings,
            )
            for candidate in candidates
        }

    def with_attention(self, adapter_space: Any) -> "SearchSpace":
        """Return a search space extended with adapter-owned attention axes."""
        return self._with_adapter_component(adapter_space, "attention")

    def with_distributed(self, adapter_space: Any) -> "SearchSpace":
        """Return a search space extended with adapter-owned distributed axes."""
        return self._with_adapter_component(adapter_space, "distributed")

    def _with_adapter_component(self, component: Any, name: str) -> "SearchSpace":
        if not hasattr(component, "axes_for") or not hasattr(
            component,
            "axis_descriptors",
        ):
            message = f"{name} adapter space is unsupported"
            raise MaterializationError(message)

        return dataclasses.replace(self, components=(*self.components, component))

    def signature(self) -> Mapping[str, Any]:
        """Return stable search-space fields."""
        return {
            "axes": {
                key: tuple(values)
                for key, values in sorted(self.axes.items(), key=itemgetter(0))
            },
            "components": tuple(
                _component_signature(component) for component in self.components
            ),
        }


@dataclasses.dataclass(frozen=True, slots=True)
class CohortConstraint:
    """Public cohort rule resolved against a search space."""

    name: str
    settings_keys: tuple[str, ...]

    def __post_init__(self) -> None:
        """Validate cohort rule fields.

        Raises:
            MaterializationError: If the rule fields are invalid.
        """
        _require_nonempty_string(self.name, "cohort name")

        if not self.settings_keys:
            message = "cohort rule requires settings keys"
            raise MaterializationError(message)

        if len(set(self.settings_keys)) != len(self.settings_keys):
            message = "cohort rule settings keys must be unique"
            raise MaterializationError(message)

        for key in self.settings_keys:
            _require_nonempty_string(key, "cohort settings key")

    def lower(self, space: SearchSpace) -> _LowerCohortConstraint:
        """Lower this public rule to an assignment-bearing constraint.

        Returns:
            Lower-layer cohort constraint.
        """
        return _LowerCohortConstraint(
            name=self.name,
            settings_keys=self.settings_keys,
            assignments=_cohort_assignments_from_space(self.settings_keys, space),
        )


@dataclasses.dataclass(frozen=True, slots=True)
class AD:
    """Public autodiff-path search component."""

    paths: Sequence[str]

    def __post_init__(self) -> None:
        """Validate autodiff path values.

        Raises:
            MaterializationError: If a path is invalid.
        """
        _require_component_values(self.paths, "AD paths")

        for value in self.paths:
            if value not in _AD_PATH_VALUES:
                message = f"AD path is unsupported: {value}"
                raise MaterializationError(message)

    def axes_for(self, operator: "Operator") -> Mapping[str, Sequence[Any]]:
        """Return operator-specific autodiff axes."""
        axis_key = _AD_AXIS_BY_OPERATOR_KIND.get(operator.spec.kind)

        if axis_key is None:
            return {}

        values = _ad_values_for_axis(axis_key, self.paths)

        return {axis_key: values}


@dataclasses.dataclass(frozen=True, slots=True)
class Precision:
    """Public precision search component."""

    model: Sequence[str] = ()
    accumulation: Sequence[str] = ()

    def __post_init__(self) -> None:
        """Validate precision values."""
        _require_optional_component_values(self.model, "Precision model")
        _require_optional_component_values(
            self.accumulation,
            "Precision accumulation",
        )

    def axes_for(self, operator: "Operator") -> Mapping[str, Sequence[Any]]:
        """Return precision axes."""
        _ = operator
        axes = {}

        if self.model:
            axes["dtype.model_compute"] = _axis_values_for_domain(
                "dtype.model_compute",
                self.model,
                "Precision model",
            )

        if self.accumulation:
            axes["dtype.accumulation"] = _axis_values_for_domain(
                "dtype.accumulation",
                self.accumulation,
                "Precision accumulation",
            )

        return axes


@dataclasses.dataclass(frozen=True, slots=True)
class Layout:
    """Public layout search component."""

    params: Sequence[str] = ()
    vector: Sequence[str] = ()
    output: Sequence[str] = ()

    def __post_init__(self) -> None:
        """Validate layout values."""
        _require_optional_component_values(self.params, "Layout params")
        _require_optional_component_values(self.vector, "Layout vector")
        _require_optional_component_values(self.output, "Layout output")

    def axes_for(self, operator: "Operator") -> Mapping[str, Sequence[Any]]:
        """Return layout axes."""
        _ = operator
        axes = {}

        if self.params:
            axes["layout.params"] = _axis_values_for_domain(
                "layout.params",
                self.params,
                "Layout params",
            )

        if self.vector:
            axes["layout.vector"] = _axis_values_for_domain(
                "layout.vector",
                self.vector,
                "Layout vector",
            )

        if self.output:
            axes["layout.output"] = _axis_values_for_domain(
                "layout.output",
                self.output,
                "Layout output",
            )

        return axes


@dataclasses.dataclass(frozen=True, slots=True)
class Vectorization:
    """Public vectorization search component."""

    modes: Sequence[str]
    batch_size: Sequence[int] = ()
    vmap_chunk_size: Sequence[int] = ()
    randomness: Sequence[str] = ()
    in_dims: Sequence[Mapping[str, Any]] = ()

    def __post_init__(self) -> None:
        """Validate vectorization values.

        Raises:
            MaterializationError: If a vectorization setting is invalid.
        """
        modes = _require_component_values(self.modes, "Vectorization modes")

        if "manual_batch" in modes and not self.batch_size:
            message = "Vectorization manual_batch requires batch_size"
            raise MaterializationError(message)

        if "vmap" in modes and (
            not self.vmap_chunk_size or not self.randomness or not self.in_dims
        ):
            message = (
                "Vectorization vmap requires vmap_chunk_size, randomness, and in_dims"
            )
            raise MaterializationError(message)

    def axes_for(self, operator: "Operator") -> Mapping[str, Sequence[Any]]:
        """Return vectorization axes."""
        _ = operator
        axes = {
            "vectorization.mode": _axis_values_for_domain(
                "vectorization.mode",
                self.modes,
                "Vectorization modes",
            )
        }

        if self.batch_size:
            axes["vectorization.batch_size"] = tuple(self.batch_size)

        if self.vmap_chunk_size:
            axes["vectorization.vmap_chunk_size"] = tuple(self.vmap_chunk_size)

        if self.randomness:
            axes["vectorization.randomness"] = _axis_values_for_domain(
                "vectorization.randomness",
                self.randomness,
                "Vectorization randomness",
            )

        if self.in_dims:
            axes["vectorization.in_dims"] = tuple(dict(value) for value in self.in_dims)

        return axes


@dataclasses.dataclass(frozen=True, slots=True)
class Compile:
    """Public compile search component."""

    enabled: Sequence[bool]
    boundaries: Sequence[str] = ()
    backend: str = "inductor"
    mode: str | None = "default"
    fullgraph: bool = False
    dynamic: bool | None = None
    compiled_autograd: bool = False
    epilogue_fusion: bool = False
    shape_padding: bool = False
    cuda_graphs: bool = False
    cache_state: str = "warm_cache"

    def __post_init__(self) -> None:
        """Validate compile component values.

        Raises:
            MaterializationError: If a compile setting is invalid.
        """
        values = tuple(self.enabled)

        if not values:
            message = "Compile enabled values must be nonempty"
            raise MaterializationError(message)

        if any(not isinstance(value, bool) for value in values):
            message = "Compile enabled values must be booleans"
            raise MaterializationError(message)

        if len(set(values)) != len(values):
            message = "Compile enabled values must be unique"
            raise MaterializationError(message)

        if len(values) > 1:
            message = "Compile cannot mix enabled and disabled rows in one component"
            raise MaterializationError(message)

        if values == (True,) and not self.boundaries:
            message = "Compile enabled rows require boundaries"
            raise MaterializationError(message)

    def axes_for(self, operator: "Operator") -> Mapping[str, Sequence[Any]]:
        """Return compile axes."""
        _ = operator
        enabled_value = self.enabled[0]

        if not enabled_value:
            return {"compile.enabled": ("false",)}

        return {
            "compile.enabled": ("true",),
            "compile.boundary": _axis_values_for_domain(
                "compile.boundary",
                self.boundaries,
                "Compile boundaries",
            ),
            "compile.backend": (self.backend,),
            "compile.mode": (self.mode,),
            "compile.fullgraph": (_bool_setting(self.fullgraph),),
            "compile.dynamic": (
                None if self.dynamic is None else _bool_setting(self.dynamic),
            ),
            "compile.compiled_autograd": (_bool_setting(self.compiled_autograd),),
            "compile.options.epilogue_fusion": (_bool_setting(self.epilogue_fusion),),
            "compile.options.shape_padding": (_bool_setting(self.shape_padding),),
            "compile.cuda_graphs": (_bool_setting(self.cuda_graphs),),
            "compile.cache_state": (self.cache_state,),
        }


@dataclasses.dataclass(frozen=True, slots=True)
class Memory:
    """Public memory search component."""

    vector_residency: Sequence[str] = ()

    def __post_init__(self) -> None:
        """Validate memory component values."""
        _require_optional_component_values(
            self.vector_residency,
            "Memory vector_residency",
        )

    def axes_for(self, operator: "Operator") -> Mapping[str, Sequence[Any]]:
        """Return memory axes."""
        _ = operator

        if not self.vector_residency:
            return {}

        return {
            "memory.vector_residency": _axis_values_for_domain(
                "memory.vector_residency",
                self.vector_residency,
                "Memory vector_residency",
            )
        }


_AD_AXIS_BY_OPERATOR_KIND = {
    "gradient": "gradient.path",
    "jvp": "jvp.path",
    "vjp": "vjp.path",
    "hvp": "hvp.path",
    "ggnvp": "ggn.jvp_path",
    "fisher_vp": "fisher.score_grad_path",
    "sampled_fisher_vp": "sampled_fisher.score_grad_path",
    "empirical_fisher_vp": "empirical_fisher.grad_path",
    "per_example_gradient": "per_example_gradient.grad_path",
}
_AD_PATH_VALUES = {
    "autograd_grad_outputs",
    "autograd_functional_hvp",
    "autograd_functional_vhp",
    "backward_materialized_grad",
    "forward_ad_dual",
    "jvp_grad",
    "linearize_grad",
    "reverse_over_reverse",
    "torch_autograd_grad",
    "torch_autograd_grad_loop",
    "torch_func_grad",
    "torch_func_grad_and_value",
    "torch_func_jvp",
    "torch_func_linearize",
    "torch_func_vjp",
    "vmap_grad",
}


def _operator_search_space_axes(
    space: SearchSpace,
    operator: "Operator",
) -> Mapping[str, Sequence[Any]]:
    axes = {key: tuple(values) for key, values in space.axes.items()}

    for component in space.components:
        component_axes = _component_axes_for_operator(component, operator)

        for key, values in component_axes.items():
            normalized = tuple(values)

            if key in axes and axes[key] != normalized:
                message = f"search-space axis is declared twice: {key}"
                raise MaterializationError(message)

            axes[key] = normalized

    return axes


def _operator_search_space_fixed_settings(
    space: SearchSpace,
    operator: "Operator",
) -> Mapping[str, Any]:
    settings = {}

    for component in space.components:
        component_settings = _component_settings_for_operator(component, operator)
        settings = _merge_search_settings(settings, component_settings)

    return settings


def _component_axes_for_operator(
    component: Any,
    operator: "Operator",
) -> Mapping[str, Sequence[Any]]:
    if not hasattr(component, "axes_for"):
        message = f"search-space component is unsupported: {type(component).__name__}"
        raise MaterializationError(message)

    axes = component.axes_for(operator)

    if not isinstance(axes, Mapping):
        message = (
            f"search-space component returned invalid axes: {type(component).__name__}"
        )
        raise MaterializationError(message)

    return axes


def _component_settings_for_operator(
    component: Any,
    operator: "Operator",
) -> Mapping[str, Any]:
    settings_for = getattr(component, "settings_for", None)

    if settings_for is None:
        return {}

    settings = settings_for(operator)

    if not isinstance(settings, Mapping):
        message = (
            "search-space component returned invalid settings: "
            f"{type(component).__name__}"
        )
        raise MaterializationError(message)

    return settings


def _search_space_axis_registry(space: SearchSpace) -> Any:
    descriptors = _search_space_axis_descriptors(space)
    descriptor_keys = _descriptor_setting_keys(descriptors)
    excluded = tuple(
        axis.name
        for axis in standard_axis_descriptors()
        if descriptor_keys.intersection(_descriptor_setting_keys((axis,)))
    )
    registry = standard_axis_registry(exclude=excluded)

    for descriptor in descriptors:
        registry.register(descriptor)

    return registry


def _search_space_axis_descriptors(space: SearchSpace) -> tuple[AxisDescriptor, ...]:
    descriptors = []

    for component in space.components:
        axis_descriptors = getattr(component, "axis_descriptors", None)

        if axis_descriptors is None:
            continue

        for descriptor in axis_descriptors():
            if not isinstance(descriptor, AxisDescriptor):
                message = (
                    "search-space component returned invalid axis descriptor: "
                    f"{type(component).__name__}"
                )
                raise MaterializationError(message)

            descriptors.append(descriptor)

    return tuple(descriptors)


def _descriptor_setting_keys(
    descriptors: Sequence[AxisDescriptor],
) -> set[str]:
    keys = set()

    for descriptor in descriptors:
        keys.update(descriptor.settings_keys)
        keys.update(descriptor.optional_settings_keys)

    return keys


def _merge_search_settings(
    *parts: Mapping[str, Any],
) -> dict[str, Any]:
    settings = {}

    for part in parts:
        for key, value in part.items():
            if key in settings and settings[key] != value:
                message = f"search-space setting is declared twice: {key}"
                raise MaterializationError(message)

            settings[key] = value

    return settings


def _component_signature(component: Any) -> Mapping[str, Any]:
    if not dataclasses.is_dataclass(component):
        message = (
            "search-space component is not dataclass-backed: "
            f"{type(component).__name__}"
        )
        raise MaterializationError(message)

    return {
        "kind": type(component).__name__,
        "fields": to_json_value(dataclasses.asdict(component)),
    }


def _require_component_values(values: Sequence[str], field: str) -> tuple[str, ...]:
    normalized = tuple(values)

    if not normalized:
        message = f"{field} must be nonempty"
        raise MaterializationError(message)

    _require_unique_string_values(normalized, field)

    return normalized


def _require_optional_component_values(values: Sequence[str], field: str) -> None:
    normalized = tuple(values)

    if normalized:
        _require_unique_string_values(normalized, field)


def _require_unique_string_values(values: tuple[str, ...], field: str) -> None:
    for value in values:
        _require_nonempty_string(value, field)

    if len(set(values)) != len(values):
        message = f"{field} values must be unique"
        raise MaterializationError(message)


def _ad_values_for_axis(
    axis_key: str,
    paths: Sequence[str],
) -> tuple[str, ...]:
    descriptor = _axis_descriptor(axis_key)
    values = tuple(value for value in paths if value in descriptor.value_domain)

    if not values:
        message = f"AD paths do not apply to {axis_key}"
        raise MaterializationError(message)

    return values


def _axis_values_for_domain(
    axis_key: str,
    values: Sequence[Any],
    field: str,
) -> tuple[Any, ...]:
    normalized = tuple(values)

    if not normalized:
        message = f"{field} must be nonempty"
        raise MaterializationError(message)

    descriptor = _axis_descriptor(axis_key)

    if descriptor.value_domain:
        invalid = tuple(
            value for value in normalized if value not in descriptor.value_domain
        )

        if invalid:
            message = f"{field} values are invalid for {axis_key}: {invalid}"
            raise MaterializationError(message)

    return normalized


def _axis_descriptor(axis_key: str) -> Any:
    descriptor = axis_manifest().by_key().get(axis_key)

    if descriptor is None:
        message = f"search-space axis is unknown: {axis_key}"
        raise MaterializationError(message)

    return descriptor


def _bool_setting(value: bool) -> str:
    return "true" if value else "false"


@dataclasses.dataclass(frozen=True, slots=True)
class Target:
    """Typed public execution target."""

    devices: tuple[str, ...]
    accelerator: str
    allowed_dtypes: tuple[str, ...]
    allowed_attention_frontends: tuple[str, ...]
    allowed_sdpa_kernels: tuple[str, ...]
    allowed_sharding_modes: tuple[str, ...]
    timing_policy: TimingPolicy
    selection_policy: SelectionPolicy
    determinism_policy: DeterminismPolicy
    environment_policy: EnvironmentPolicy

    def lower(self, search: SearchStrategy) -> _LowerTarget:
        """Return the lower-layer target used by the tuning engine."""
        return _LowerTarget(
            devices=self.devices,
            accelerator=self.accelerator,
            allowed_dtypes=self.allowed_dtypes,
            allowed_attention_frontends=self.allowed_attention_frontends,
            allowed_sdpa_kernels=self.allowed_sdpa_kernels,
            allowed_sharding_modes=self.allowed_sharding_modes,
            timing_policy=self.timing_policy,
            selection_policy=self.selection_policy,
            search_policy=search.policy,
            determinism_policy=self.determinism_policy.signature(),
            environment_capture=self.environment_policy.signature(),
        )

    def signature(self) -> Mapping[str, Any]:
        """Return stable public target fields."""
        return {
            "devices": self.devices,
            "accelerator": self.accelerator,
            "allowed_dtypes": self.allowed_dtypes,
            "allowed_attention_frontends": self.allowed_attention_frontends,
            "allowed_sdpa_kernels": self.allowed_sdpa_kernels,
            "allowed_sharding_modes": self.allowed_sharding_modes,
            "timing_policy": dataclasses.asdict(self.timing_policy),
            "selection_policy": dataclasses.asdict(self.selection_policy),
            "determinism_policy": self.determinism_policy.signature(),
            "environment": self.environment_policy.signature(),
        }


@dataclasses.dataclass(frozen=True, slots=True)
class Loss:
    """Typed scalar-loss declaration."""

    kind: str
    output: str
    objective: ScalarObjective | None
    identity_fields: Mapping[str, Any]
    hessian_factor: torch.Tensor | None = None
    hessian_matvec: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None

    def signature(self) -> Mapping[str, Any]:
        """Return stable loss identity fields."""
        return {
            "kind": self.kind,
            "output": self.output,
            "identity": dict(self.identity_fields),
        }


@dataclasses.dataclass(frozen=True, slots=True)
class Likelihood:
    """Typed predictive-likelihood declaration."""

    kind: str
    output: str
    fields: Mapping[str, Any]

    def signature(self) -> Mapping[str, Any]:
        """Return stable likelihood identity fields."""
        return {
            "kind": self.kind,
            "output": self.output,
            "fields": dict(self.fields),
        }


@dataclasses.dataclass(frozen=True, slots=True)
class SampleSource:
    """Typed sampled-Fisher sample source declaration."""

    kind: str
    count: int
    identity_fields: Mapping[str, Any]
    table: torch.Tensor | None = None

    def signature(self) -> Mapping[str, Any]:
        """Return stable sample-source identity fields."""
        return {
            "kind": self.kind,
            "count": self.count,
            "identity": dict(self.identity_fields),
        }

    def runtime_sample_source(self) -> str:
        """Return the lower sampled-Fisher sample source value.

        Raises:
            MaterializationError: If the source kind is not lowered.
        """
        if self.kind == "fixed_seed":
            return "fixed_seed_and_count"

        if self.kind == "table":
            return "fixed_sample_table"

        message = f"sample source kind is not lowered: {self.kind}"
        raise MaterializationError(message)


@dataclasses.dataclass(frozen=True, slots=True)
class Combine:
    """Base type for public composition expressions."""


@dataclasses.dataclass(frozen=True, slots=True)
class ScaledIdentity(Combine):
    """Scaled identity leaf in a public composition expression."""

    coefficient: float


@dataclasses.dataclass(frozen=True, slots=True)
class Source(Combine):
    """Batch-to-vector source leaf in a public composition expression."""

    child: str


@dataclasses.dataclass(frozen=True, slots=True)
class Compose(Combine):
    """Sequential public composition expression."""

    terms: tuple[Combine | str, ...]


@dataclasses.dataclass(frozen=True, slots=True)
class LinearCombination(Combine):
    """Weighted-sum public composition expression."""

    terms: tuple[tuple[float, Combine | str], ...]


@dataclasses.dataclass(frozen=True, slots=True)
class _ScalarLossObjective:
    callback: ScalarObjective
    output: str
    version: str

    def identity(self) -> Mapping[str, Any]:
        """Return stable scalar objective identity fields."""
        return {
            "kind": "loss.from_scalar",
            "output": self.output,
            "version": self.version,
        }

    def __call__(
        self,
        params: ParameterTree,
        buffers: BufferTree,
        batch: Batch,
        context: Any,
    ) -> torch.Tensor:
        """Evaluate the wrapped scalar objective.

        Returns:
            Scalar loss tensor.
        """
        return self.callback(params, buffers, batch, context)


@dataclasses.dataclass(frozen=True, slots=True)
class _SoftmaxCrossEntropyObjective:
    model: Model
    output: str
    labels: str
    mask: str | None
    reduction: str
    denominator: str

    def identity(self) -> Mapping[str, Any]:
        """Return stable scalar objective identity fields."""
        return {
            "kind": "loss.softmax_cross_entropy",
            "output": self.output,
            "labels": self.labels,
            "mask": self.mask,
            "reduction": self.reduction,
            "denominator": self.denominator,
            "model": self.model.signature(),
        }

    def __call__(
        self,
        params: ParameterTree,
        buffers: BufferTree,
        batch: Batch,
        context: Any,
    ) -> torch.Tensor:
        """Evaluate softmax cross entropy over the declared model output.

        Returns:
            Scalar CE loss tensor.
        """
        _ = context
        output = _call_model(self.model, params, buffers, batch)
        logits = _single_tensor_output(
            _select_model_output(self.model, output, self.output),
            self.output,
        )
        labels = _batch_long_tensor(batch, self.labels)
        losses = _softmax_cross_entropy_losses(logits, labels)
        mask = _loss_mask(batch, self.mask, labels.shape)

        if mask is not None:
            mask = mask.to(device=losses.device, dtype=losses.dtype)
            losses = losses * mask

        return _reduce_losses(
            losses,
            mask=mask,
            reduction=self.reduction,
            denominator=self.denominator,
        )


@dataclasses.dataclass(frozen=True, slots=True)
class _KLLossObjective:
    model: Model
    output: str
    target: str
    mask: str | None
    reduction: str
    denominator: str

    def identity(self) -> Mapping[str, Any]:
        """Return stable scalar objective identity fields."""
        return {
            "kind": "loss.kl",
            "output": self.output,
            "target": self.target,
            "mask": self.mask,
            "reduction": self.reduction,
            "denominator": self.denominator,
            "model": self.model.signature(),
        }

    def __call__(
        self,
        params: ParameterTree,
        buffers: BufferTree,
        batch: Batch,
        context: Any,
    ) -> torch.Tensor:
        """Evaluate KL divergence over the declared model output.

        Returns:
            Scalar KL loss tensor.
        """
        _ = context
        output = _call_model(self.model, params, buffers, batch)
        logits = _single_tensor_output(
            _select_model_output(self.model, output, self.output),
            self.output,
        )
        target = _batch_tensor(batch, self.target)
        losses = _kl_token_losses(logits, target)
        mask = _loss_mask(batch, self.mask, losses.shape)

        if mask is not None:
            mask = mask.to(device=losses.device, dtype=losses.dtype)
            losses = losses * mask

        return _reduce_losses(
            losses,
            mask=mask,
            reduction=self.reduction,
            denominator=self.denominator,
        )


@dataclasses.dataclass(frozen=True, slots=True)
class _MSELossObjective:
    model: Model
    output: str
    target: str
    mask: str | None
    reduction: str
    denominator: str

    def identity(self) -> Mapping[str, Any]:
        """Return stable scalar objective identity fields."""
        return {
            "kind": "loss.mse",
            "output": self.output,
            "target": self.target,
            "mask": self.mask,
            "reduction": self.reduction,
            "denominator": self.denominator,
            "model": self.model.signature(),
        }

    def __call__(
        self,
        params: ParameterTree,
        buffers: BufferTree,
        batch: Batch,
        context: Any,
    ) -> torch.Tensor:
        """Evaluate squared error over the declared model output.

        Returns:
            Scalar squared-error loss tensor.
        """
        _ = context
        output = _call_model(self.model, params, buffers, batch)
        prediction = _single_tensor_output(
            _select_model_output(self.model, output, self.output),
            self.output,
        )
        target = _batch_tensor(batch, self.target)
        losses = _mse_element_losses(prediction, target)
        mask = _mse_mask(batch, self.mask, prediction.shape)

        if mask is not None:
            losses = losses * _mse_element_mask(mask, prediction)

        return _reduce_mse_losses(
            losses,
            mask=mask,
            reduction=self.reduction,
            denominator=self.denominator,
        )


@dataclasses.dataclass(frozen=True, slots=True)
class _SoftmaxCrossEntropyPerExampleObjective:
    model: Model
    output: str
    labels: str
    mask: str | None
    reduction: str
    denominator: str

    def identity(self) -> Mapping[str, Any]:
        """Return stable per-example objective identity fields."""
        return {
            "kind": "loss.softmax_cross_entropy.per_example",
            "output": self.output,
            "labels": self.labels,
            "mask": self.mask,
            "reduction": self.reduction,
            "denominator": self.denominator,
            "model": self.model.signature(),
        }

    def __call__(
        self,
        params: ParameterTree,
        buffers: BufferTree,
        batch: Batch,
        context: Any,
    ) -> torch.Tensor:
        """Evaluate per-example CE losses.

        Returns:
            One loss value per leading batch example.
        """
        _ = context
        output = _call_model(self.model, params, buffers, batch)
        logits = _single_tensor_output(
            _select_model_output(self.model, output, self.output),
            self.output,
        )
        labels = _batch_long_tensor(batch, self.labels)
        losses = _softmax_cross_entropy_losses(logits, labels)
        mask = _loss_mask(batch, self.mask, labels.shape)

        return _per_example_reduce_losses(
            losses,
            mask=mask,
            reduction=self.reduction,
            denominator=self.denominator,
        )


@dataclasses.dataclass(frozen=True, slots=True)
class _GaussianScoreGradientBatch:
    model: Model
    output: str
    target: str
    noise: float

    def __call__(self, batch: Batch) -> Batch:
        """Add Gaussian score-gradient rows to a batch.

        Returns:
            Batch with score_gradients and num_examples fields.

        Raises:
            MaterializationError: If the batch does not match the likelihood.
        """
        with torch.enable_grad():
            output = _call_model(
                self.model,
                self.model.parameter_values,
                self.model.buffers,
                batch,
            )
            prediction = _single_tensor_output(
                _select_model_output(self.model, output, self.output),
                self.output,
            )
            target = _batch_tensor(batch, self.target).to(
                device=prediction.device,
                dtype=prediction.dtype,
            )

            if target.shape != prediction.shape:
                message = "gaussian likelihood target shape must match output"
                raise MaterializationError(message)

            if prediction.ndim == 0:
                message = "gaussian likelihood requires a leading example axis"
                raise MaterializationError(message)

            residual = target - prediction
            scores = -0.5 * (residual / prediction.new_tensor(self.noise)).square()
            score_rows = scores.reshape(scores.shape[0], -1).sum(dim=1)
            score_gradients = _score_gradient_rows(
                score_rows,
                self.model.parameter_values,
            )

        return {
            **batch,
            "score_gradients": score_gradients,
            "num_examples": score_gradients.shape[0],
        }


@dataclasses.dataclass(frozen=True, slots=True)
class _GaussianScoreTermsObjective:
    model: Model
    output: str
    target: str
    noise: float

    def identity(self) -> Mapping[str, Any]:
        """Return stable Gaussian score identity fields."""
        return {
            "kind": "likelihood.gaussian.score_terms",
            "output": self.output,
            "target": self.target,
            "noise": self.noise,
            "model": self.model.signature(),
        }

    def __call__(
        self,
        params: ParameterTree,
        buffers: BufferTree,
        batch: Batch,
        context: Any,
    ) -> torch.Tensor:
        """Return one score term per example.

        Raises:
            MaterializationError: If the batch shape is incompatible.
        """
        _ = context
        output = _call_model(self.model, params, buffers, batch)
        prediction = _single_tensor_output(
            _select_model_output(self.model, output, self.output),
            self.output,
        )
        target = _batch_tensor(batch, self.target).to(
            device=prediction.device,
            dtype=prediction.dtype,
        )

        if target.shape != prediction.shape:
            message = "gaussian likelihood target shape must match output"
            raise MaterializationError(message)

        if prediction.ndim == 0:
            message = "gaussian likelihood requires a leading example axis"
            raise MaterializationError(message)

        residual = target - prediction
        scores = -0.5 * (residual / prediction.new_tensor(self.noise)).square()

        return scores.reshape(scores.shape[0], -1).sum(dim=1)


@dataclasses.dataclass(frozen=True, slots=True)
class _SampledFisherScoreGradientBatch:
    model: Model
    likelihood: Likelihood
    samples: SampleSource

    def __call__(self, batch: Batch) -> Batch:
        """Add sampled score-gradient rows to a batch.

        Returns:
            Batch with sampled_score_gradients and denominator fields.
        """
        with torch.enable_grad():
            output = _call_model(
                self.model,
                self.model.parameter_values,
                self.model.buffers,
                batch,
            )
            prediction = _single_tensor_output(
                _select_model_output(self.model, output, self.likelihood.output),
                self.likelihood.output,
            )
            scores, denominator_fields = _sampled_likelihood_scores(
                prediction,
                self.likelihood,
                self.samples,
            )
            score_gradients = _score_gradient_rows(
                scores.reshape(-1),
                self.model.parameter_values,
            )

        return {
            **batch,
            **denominator_fields,
            "sampled_score_gradients": score_gradients,
        }


def _score_gradient_rows(
    scores: torch.Tensor,
    params: ParameterTree,
) -> torch.Tensor:
    leaves = tree_leaves(params)

    if not leaves:
        message = "score-gradient lowering requires parameter leaves"
        raise MaterializationError(message)

    rows = []

    for index, score in enumerate(scores):
        grads = torch.autograd.grad(
            score,
            leaves,
            retain_graph=index + 1 < scores.shape[0],
            allow_unused=True,
        )
        rows.append(
            torch.cat(
                tuple(
                    torch.zeros_like(param).reshape(-1)
                    if grad is None
                    else grad.detach().reshape(-1)
                    for grad, param in zip(grads, leaves, strict=True)
                )
            )
        )

    return torch.stack(tuple(rows))


@dataclasses.dataclass(frozen=True, slots=True)
class _ModelFieldObjective:
    model: Model
    field: str

    def identity(self) -> Mapping[str, Any]:
        """Return stable function objective identity fields."""
        return {
            "kind": "output",
            "field": self.field,
            "model": self.model.signature(),
        }

    def __call__(
        self,
        params: ParameterTree,
        buffers: BufferTree,
        batch: Batch,
        context: Any,
    ) -> TensorTree:
        """Evaluate the declared model output.

        Returns:
            Selected tensor output.
        """
        _ = context
        output = _call_model(self.model, params, buffers, batch)

        return _select_model_output(self.model, output, self.field)


@dataclasses.dataclass(frozen=True, slots=True)
class _SoftmaxCrossEntropyLossHessianBatch:
    model: Model
    output: str
    labels: str
    mask: str | None
    reduction: str
    denominator: str

    def __call__(self, batch: Batch) -> Batch:
        output = _call_model(
            self.model,
            self.model.parameter_values,
            self.model.buffers,
            batch,
        )
        logits = _single_tensor_output(
            _select_model_output(self.model, output, self.output),
            self.output,
        )
        labels = _batch_long_tensor(batch, self.labels)
        _require_softmax_cross_entropy_shapes(logits, labels)
        mask = _loss_mask(batch, self.mask, labels.shape)
        loss_hessian = _softmax_cross_entropy_loss_hessian(
            logits,
            mask=mask,
            reduction=self.reduction,
            denominator=self.denominator,
        )

        return {**batch, "loss_hessian": loss_hessian}


@dataclasses.dataclass(frozen=True, slots=True)
class _KLLossHessianBatch:
    model: Model
    output: str
    target: str
    mask: str | None
    reduction: str
    denominator: str

    def __call__(self, batch: Batch) -> Batch:
        output = _call_model(
            self.model,
            self.model.parameter_values,
            self.model.buffers,
            batch,
        )
        logits = _single_tensor_output(
            _select_model_output(self.model, output, self.output),
            self.output,
        )
        target = _batch_tensor(batch, self.target)
        _require_same_shape(logits, target, "loss.kl target")
        mask = _loss_mask(batch, self.mask, logits.shape[:-1])
        loss_hessian = _kl_loss_hessian(
            logits,
            target,
            mask=mask,
            reduction=self.reduction,
            denominator=self.denominator,
        )

        return {**batch, "loss_hessian": loss_hessian}


@dataclasses.dataclass(frozen=True, slots=True)
class _MSELossHessianBatch:
    model: Model
    output: str
    target: str
    mask: str | None
    reduction: str
    denominator: str

    def __call__(self, batch: Batch) -> Batch:
        output = _call_model(
            self.model,
            self.model.parameter_values,
            self.model.buffers,
            batch,
        )
        prediction = _single_tensor_output(
            _select_model_output(self.model, output, self.output),
            self.output,
        )
        target = _batch_tensor(batch, self.target)
        _require_same_shape(prediction, target, "loss.mse target")
        mask = _mse_mask(batch, self.mask, prediction.shape)
        loss_hessian = _mse_loss_hessian(
            prediction,
            mask=mask,
            reduction=self.reduction,
            denominator=self.denominator,
        )

        return {**batch, "loss_hessian": loss_hessian}


@dataclasses.dataclass(frozen=True, slots=True)
class _DeclaredPSDLossHessianBatch:
    model: Model
    output: str
    factor: torch.Tensor

    def __call__(self, batch: Batch) -> Batch:
        output = _call_model(
            self.model,
            self.model.parameter_values,
            self.model.buffers,
            batch,
        )
        value = _single_tensor_output(
            _select_model_output(self.model, output, self.output),
            self.output,
        )
        loss_hessian = _declared_psd_loss_hessian(self.factor, value)

        return {**batch, "loss_hessian": loss_hessian}


@dataclasses.dataclass(frozen=True, slots=True)
class _DeclaredPSDMatrixFreeLossHessianBatch:
    model: Model
    output: str
    matvec: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]

    def __call__(self, batch: Batch) -> Batch:
        output = _call_model(
            self.model,
            self.model.parameter_values,
            self.model.buffers,
            batch,
        )
        value = _single_tensor_output(
            _select_model_output(self.model, output, self.output),
            self.output,
        )
        loss_hessian = _declared_psd_matrix_free_loss_hessian(self.matvec, value)

        return {**batch, "loss_hessian": loss_hessian}


@dataclasses.dataclass(frozen=True, slots=True)
class Operator:
    """Typed operator that executes through the standard runtime."""

    model: Model
    spec: Any
    call_inputs: tuple[str, ...]
    default_settings: Mapping[str, Any]
    metric: Metric | None = None
    scalar_objectives: Mapping[str, ScalarObjective] = dataclasses.field(
        default_factory=dict
    )
    function_objectives: Mapping[str, FunctionObjective] = dataclasses.field(
        default_factory=dict
    )
    batch_transform: Callable[[Batch], Batch] | None = None
    changed_axes: tuple[str, ...] | None = None
    plan: Plan | None = None
    bound_batch: Batch | None = None

    def __call__(self, *call_inputs: Any) -> Any:
        """Run the operator through its declared reference row.

        Returns:
            Operator output.

        Raises:
            AdmissionError: If the declared reference row is rejected.
            MaterializationError: If the call inputs or runtime row are invalid.
        """
        if len(call_inputs) != len(self.call_inputs):
            message = f"{self.spec.kind} expects {len(self.call_inputs)} inputs"
            raise MaterializationError(message)

        dependencies = operator_dependencies(self.spec)

        if dependencies and self.plan is None:
            dependency_label = "child" if self.spec.kind == "composition" else "sibling"
            message = (
                f"{self.spec.kind} requires selected {dependency_label} rows "
                "from vp.tune: "
                f"{dependencies}"
            )
            raise MaterializationError(message)

        batch, vector = _runtime_call_inputs(self.call_inputs, call_inputs)
        batch = _prepared_operator_batch(self, batch)

        if self.plan is not None:
            selected = self.plan.materialize(name=self.spec.family)

            return selected(batch, vector)

        candidate = Candidate(
            family=self.spec.family,
            candidate_id="typed_reference",
            settings=dict(self.default_settings),
            changed_axes=(
                tuple(self.default_settings)
                if self.changed_axes is None
                else self.changed_axes
            ),
            generator_id="vptune.typed_public",
            generator_version=PACKAGE_VERSION,
        )
        admitted = standard_axis_registry().admit(candidate)

        if admitted.admission_status != "passed":
            message = admitted.admission_error or "typed reference row was rejected"
            raise AdmissionError(message)

        operation_factory = standard_operation_factory(
            self.spec,
            params=self.model.parameter_values,
            buffers=self.model.buffers,
            parameter_surface=self.model.parameters,
            scalar_objectives=self.scalar_objectives,
            function_objectives=self.function_objectives,
            module=self.model.module,
            module_call=self.model.call,
        )

        return operation_factory(admitted, batch, vector)()

    def tune(
        self,
        *,
        data: Iterable[Batch] | None = None,
        vectors: Iterable[TensorTree] | None = None,
        target: Target,
        space: SearchSpace,
        search: SearchStrategy,
        run_dir: Path | None = None,
        reference: Case | None = None,
        probes: Sequence[tuple[Batch, TensorTree]] | None = None,
        memory_backend: MemoryBackend | None = None,
        clock: Callable[[], float] = time.perf_counter,
    ) -> "Operator":
        """Tune this operator and return a new operator carrying the plan.

        Returns:
            Tuned operator.

        Raises:
            MaterializationError: If tuning does not select a row.
        """
        tuning_problem = _operator_problem(
            self,
            data=data,
            vectors=vectors,
            target=target,
            space=space,
            search=search,
            reference=reference,
            probes=probes,
        )
        selected_plan = autotune(
            tuning_problem,
            run_dir=run_dir,
            memory_backend=memory_backend,
            clock=clock,
        )

        if self.spec.family not in selected_plan.selected:
            message = "operator.tune requires a search strategy that selects a row"
            raise MaterializationError(message)

        return dataclasses.replace(self, plan=selected_plan)

    def load(
        self,
        run_dir: Path,
        *,
        memory_backend: MemoryBackend | None = None,
    ) -> "Operator":
        """Load a saved tuning plan for this operator.

        Returns:
            Tuned operator.
        """
        selected_plan = _load_operator_plan(
            self,
            run_dir,
            memory_backend=memory_backend,
        )

        return dataclasses.replace(self, plan=selected_plan)

    def bind(self, *, batch: Batch) -> "Operator":
        """Bind a batch for repeated vector calls.

        Returns:
            Operator with the batch input fixed.

        Raises:
            MaterializationError: If the operator has no batch input.
        """
        if "batch" not in self.call_inputs:
            message = f"{self.spec.kind} has no batch input to bind"
            raise MaterializationError(message)

        return dataclasses.replace(
            self,
            call_inputs=tuple(
                input_name for input_name in self.call_inputs if input_name != "batch"
            ),
            bound_batch=batch,
        )


def torch_model(
    module: torch.nn.Module,
    *,
    parameters: ParameterSurface,
    call: ModuleCallSpec,
) -> Model:
    """Build a typed model binding.

    Returns:
        Typed model binding.

    Raises:
        MaterializationError: If the parameter surface does not match the module.
    """
    remove_duplicate = parameters.tied_weights_policy == "deduplicate"
    named_params = dict(module.named_parameters(remove_duplicate=remove_duplicate))
    missing = tuple(name for name in parameters.names if name not in named_params)

    if missing:
        message = f"parameter surface names are missing from module: {missing}"
        raise MaterializationError(message)

    values = _module_parameter_values(named_params, parameters.names)
    buffers = _module_buffers(module, parameters.buffer_policy)

    return Model(
        module=module,
        parameters=parameters,
        call=call,
        parameter_values=values,
        buffers=buffers,
    )


def parameters(
    module: torch.nn.Module,
    *,
    include: Callable[[str, torch.nn.Parameter], bool] | None = None,
    buffers: str = "include",
    tied: str = "preserve",
) -> ParameterSurface:
    """Build the typed parameter surface.

    Returns:
        Parameter surface for the module.

    Raises:
        MaterializationError: If a closed-set policy is invalid.
    """
    if buffers not in {"include", "exclude"}:
        message = f"buffers policy is unsupported: {buffers}"
        raise MaterializationError(message)

    return parameter_surface(
        module,
        include=include,
        buffers=buffers,
        tied_weights=tied,
    )


def module_call(
    *,
    args: Sequence[str],
    kwargs: Mapping[str, str],
    output: str,
) -> ModuleCallSpec:
    """Build a typed module-call binding.

    Returns:
        Module call specification.

    Raises:
        MaterializationError: If the output field is invalid.
    """
    if not isinstance(output, str) or not output:
        message = "module_call output must be a nonempty string"
        raise MaterializationError(message)

    return ModuleCallSpec(
        positional_batch_keys=tuple(args),
        keyword_batch_keys=dict(kwargs),
        output_fields={output: (output,)},
    )


def output(field: str) -> Output:
    """Build a typed model-output declaration.

    Returns:
        Output declaration.

    Raises:
        MaterializationError: If the field is invalid.
    """
    if not isinstance(field, str) or not field:
        message = "output field must be a nonempty string"
        raise MaterializationError(message)

    return Output(field)


def case(*, batch: Batch | None = None, vector: TensorTree | None = None) -> Case:
    """Build a typed reference or probe input.

    Returns:
        Case declaration.
    """
    if vector is not None:
        _checked_tensor_tree(vector, "case vector")

    return Case(
        batch=None if batch is None else dict(batch),
        vector=vector,
    )


def _module_buffers(module: torch.nn.Module, policy: str) -> BufferTree:
    if policy == "include":
        return dict(module.named_buffers())

    if policy == "exclude":
        return {}

    message = f"buffers policy is unsupported: {policy}"
    raise MaterializationError(message)


def _module_parameter_values(
    named_params: Mapping[str, torch.Tensor],
    names: Sequence[str],
) -> ParameterTree:
    return {name: named_params[name] for name in names}


def _call_model(
    model: Model,
    params: ParameterTree,
    buffers: BufferTree,
    batch: Batch,
) -> object:
    args = tuple(_batch_value(batch, key) for key in model.call.positional_batch_keys)
    kwargs = {
        argument_name: _batch_value(batch, batch_key)
        for argument_name, batch_key in model.call.keyword_batch_keys.items()
    }

    return torch.func.functional_call(
        model.module,
        (params, buffers),
        args,
        kwargs,
    )


def _batch_value(batch: Batch, key: str) -> Any:
    if key in batch:
        return batch[key]

    message = f"batch field is missing: {key}"
    raise MaterializationError(message)


def _select_model_output(model: Model, output: object, field: str) -> TensorTree:
    if isinstance(output, Mapping):
        for key, value in output.items():
            if key == field:
                return _checked_tensor_tree(value, f"model output field {field}")

    if hasattr(output, field):
        return _checked_tensor_tree(
            getattr(output, field),
            f"model output field {field}",
        )

    if isinstance(output, torch.Tensor) and tuple(model.call.output_fields) == (field,):
        return output

    message = f"model output field is missing: {field}"
    raise MaterializationError(message)


def _checked_tensor_tree(value: object, name: str) -> TensorTree:
    if _is_tensor_tree(value):
        return value

    message = f"{name} must be a tensor tree"
    raise MaterializationError(message)


def _is_tensor_tree(value: object) -> TypeGuard[TensorTree]:
    if isinstance(value, torch.Tensor):
        return True

    if isinstance(value, tuple):
        return all(_is_tensor_tree(item) for item in value)

    if isinstance(value, dict):
        return all(
            isinstance(key, str) and _is_tensor_tree(item)
            for key, item in value.items()
        )

    return False


class _LossNamespace:
    @staticmethod
    def softmax_cross_entropy(
        *,
        output: str,
        labels: str,
        mask: str | None = None,
        reduction: str = "token_mean",
        denominator: str = "num_tokens",
    ) -> Loss:
        """Build a typed softmax-cross-entropy loss.

        Returns:
            Loss declaration with package-owned CE objective lowering.

        Raises:
            MaterializationError: If a closed-set field is invalid.
        """
        _require_nonempty_string(output, "loss output")
        _require_nonempty_string(labels, "loss labels")

        if mask is not None:
            _require_nonempty_string(mask, "loss mask")

        _require_loss_reduction(reduction)

        if denominator != "num_tokens":
            message = f"softmax_cross_entropy denominator is unsupported: {denominator}"
            raise MaterializationError(message)

        return Loss(
            kind="softmax_cross_entropy",
            output=output,
            objective=None,
            identity_fields={
                "kind": "loss.softmax_cross_entropy",
                "output": output,
                "labels": labels,
                "mask": mask,
                "reduction": reduction,
                "denominator": denominator,
            },
        )

    @staticmethod
    def kl(
        *,
        output: str,
        target: str,
        mask: str | None = None,
        reduction: str = "token_mean",
        denominator: str = "num_tokens",
    ) -> Loss:
        """Build a typed softmax-KL loss.

        Returns:
            Loss declaration with package-owned KL objective lowering.
        """
        _require_nonempty_string(output, "loss output")
        _require_nonempty_string(target, "loss target")

        if mask is not None:
            _require_nonempty_string(mask, "loss mask")

        _require_loss_reduction(reduction)
        _require_token_mean_denominator(reduction, denominator)

        return Loss(
            kind="kl",
            output=output,
            objective=None,
            identity_fields={
                "kind": "loss.kl",
                "output": output,
                "target": target,
                "mask": mask,
                "reduction": reduction,
                "denominator": denominator,
            },
        )

    @staticmethod
    def mse(
        *,
        output: str,
        target: str,
        mask: str | None = None,
        reduction: str = "mean",
        denominator: str = "num_elements",
    ) -> Loss:
        """Build a typed squared-error loss.

        Returns:
            Loss declaration with package-owned MSE objective lowering.
        """
        _require_nonempty_string(output, "loss output")
        _require_nonempty_string(target, "loss target")

        if mask is not None:
            _require_nonempty_string(mask, "loss mask")

        _require_loss_reduction(reduction)
        _require_mse_denominator(reduction, denominator)

        return Loss(
            kind="mse",
            output=output,
            objective=None,
            identity_fields={
                "kind": "loss.mse",
                "output": output,
                "target": target,
                "mask": mask,
                "reduction": reduction,
                "denominator": denominator,
            },
        )

    @staticmethod
    def from_scalar(
        fn: ScalarObjective,
        *,
        output: str,
        version: str,
    ) -> Loss:
        """Build a typed loss from a scalar objective.

        Returns:
            Scalar loss declaration.

        Raises:
            MaterializationError: If the output or version is invalid.
        """
        if not isinstance(output, str) or not output:
            message = "loss output must be a nonempty string"
            raise MaterializationError(message)

        if not isinstance(version, str) or not version:
            message = "loss version must be a nonempty string"
            raise MaterializationError(message)

        objective = _ScalarLossObjective(fn, output, version)

        return Loss(
            kind="from_scalar",
            output=output,
            objective=objective,
            identity_fields=objective.identity(),
        )

    @staticmethod
    def declared_psd(
        *,
        output: str,
        factors: torch.Tensor,
    ) -> Loss:
        """Build a typed declared-PSD output-space loss.

        Returns:
            Loss declaration with a dense output-space PSD factor.

        Raises:
            MaterializationError: If the output or factor declaration is invalid.
        """
        _require_nonempty_string(output, "loss output")
        _require_tensor(factors, "loss.declared_psd factors")

        if factors.ndim != SQUARE_MATRIX_DIMS:
            message = "loss.declared_psd factors must be a matrix"
            raise MaterializationError(message)

        _require_finite_public_tensor(factors, "loss.declared_psd factors")

        return Loss(
            kind="declared_psd",
            output=output,
            objective=None,
            identity_fields={
                "kind": "loss.declared_psd",
                "output": output,
                "factors": tensor_signature(factors),
            },
            hessian_factor=factors,
        )

    @staticmethod
    def declared_psd_matrix_free(
        *,
        output: str,
        matvec: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        version: str,
    ) -> Loss:
        """Build a typed matrix-free declared-PSD output-space loss.

        Returns:
            Loss declaration with a caller-owned output-space Hessian matvec.

        Raises:
            MaterializationError: If the declaration is invalid.
        """
        _require_nonempty_string(output, "loss output")

        if not callable(matvec):
            message = "loss.declared_psd_matrix_free matvec must be callable"
            raise MaterializationError(message)

        _require_nonempty_string(version, "loss version")

        return Loss(
            kind="declared_psd_matrix_free",
            output=output,
            objective=None,
            identity_fields={
                "kind": "loss.declared_psd_matrix_free",
                "output": output,
                "matvec": _qualified_callable_name(matvec),
                "version": version,
            },
            hessian_matvec=matvec,
        )


class _MetricNamespace:
    @staticmethod
    def dense(*, matrix: torch.Tensor) -> Metric:
        """Build a dense typed metric.

        Returns:
            Dense metric declaration.
        """
        _require_tensor(matrix, "metric.dense matrix")

        return Metric(
            kind="dense_matrix",
            representation={"kind": "dense_matrix"},
            batch={"metric_matrix": matrix},
            identity_fields={"metric_matrix": tensor_signature(matrix)},
        )

    @staticmethod
    def diagonal(*, diag: TensorTree) -> Metric:
        """Build a diagonal typed metric.

        Returns:
            Diagonal metric declaration.
        """
        return Metric(
            kind="diagonal_tree",
            representation={"kind": "diagonal_tree"},
            batch={"metric_diagonal": diag},
            identity_fields={"metric_diagonal": _tree_signature(diag)},
        )

    @staticmethod
    def ekfac(
        *,
        eigvecs_a: Mapping[str, torch.Tensor],
        eigvecs_g: Mapping[str, torch.Tensor],
        corrected_eigenvalues: Mapping[str, torch.Tensor],
    ) -> Metric:
        """Build an EKFAC typed metric.

        Returns:
            EKFAC metric declaration.
        """
        _require_same_keys(eigvecs_a, eigvecs_g, "ekfac eigvec keys")
        _require_same_keys(eigvecs_a, corrected_eigenvalues, "ekfac eigenvalue keys")

        return Metric(
            kind="ekfac_factors",
            representation={"kind": "ekfac_factors"},
            batch={
                "ekfac_eigvecs_a": dict(eigvecs_a),
                "ekfac_eigvecs_g": dict(eigvecs_g),
                "ekfac_corrected_eigenvalues": dict(corrected_eigenvalues),
            },
            identity_fields={
                "ekfac_eigvecs_a": _tree_signature(eigvecs_a),
                "ekfac_eigvecs_g": _tree_signature(eigvecs_g),
                "ekfac_corrected_eigenvalues": _tree_signature(corrected_eigenvalues),
            },
        )

    @staticmethod
    def low_rank(
        *,
        factor: TensorTree,
        diagonal: TensorTree,
    ) -> Metric:
        """Build a low-rank typed metric.

        Returns:
            Low-rank metric declaration.
        """
        return Metric(
            kind="low_rank_factors",
            representation={"kind": "low_rank_factors"},
            batch={"low_rank_factors": {"basis": factor, "diagonal": diagonal}},
            identity_fields={
                "factor": _tree_signature(factor),
                "diagonal": _tree_signature(diagonal),
            },
        )

    @staticmethod
    def block_diagonal(*, blocks: Mapping[str, torch.Tensor]) -> Metric:
        """Build a block-diagonal typed metric.

        Returns:
            Block-diagonal metric declaration.

        Raises:
            MaterializationError: If the block mapping is empty or invalid.
        """
        if not blocks:
            message = "metric.block_diagonal blocks must be nonempty"
            raise MaterializationError(message)

        for key, block in blocks.items():
            _require_nonempty_string(key, "metric.block_diagonal block key")
            _require_tensor(block, f"metric.block_diagonal block {key}")

        return Metric(
            kind="block_diagonal",
            representation={
                "kind": "block_diagonal",
                "block_names": tuple(blocks),
            },
            batch={"metric_blocks": tuple(blocks.values())},
            identity_fields={"metric_blocks": _tree_signature(blocks)},
        )

    @staticmethod
    def kfac(
        *,
        factors: Mapping[str, Mapping[str, torch.Tensor]],
        dampings: Mapping[str, float] | None = None,
    ) -> Metric:
        """Build a KFAC typed metric.

        Returns:
            KFAC metric declaration.

        Raises:
            MaterializationError: If factor declarations are invalid.
        """
        if dampings is not None:
            message = "metric.kfac dampings are not lowered"
            raise MaterializationError(message)

        if not factors:
            message = "metric.kfac factors must be nonempty"
            raise MaterializationError(message)

        blocks = []
        factor_batch = {}

        for parameter_name, pair in factors.items():
            _require_nonempty_string(parameter_name, "metric.kfac parameter")
            left, right = _kfac_factor_pair(pair, parameter_name)
            left_key = f"{parameter_name}.left_factor"
            right_key = f"{parameter_name}.right_factor"
            factor_batch[left_key] = left
            factor_batch[right_key] = right
            blocks.append({
                "parameter": parameter_name,
                "left_factor": left_key,
                "right_factor": right_key,
            })

        return Metric(
            kind="kfac_factors",
            representation={"kind": "kfac_factors", "blocks": tuple(blocks)},
            batch={"kfac_factors": factor_batch},
            identity_fields={"kfac_factors": _tree_signature(factors)},
        )

    @staticmethod
    def ggn_derived(*, factors: Mapping[str, Any]) -> Metric:
        """Build a GGN-derived typed metric.

        Returns:
            GGN-derived metric declaration.
        """
        return Metric(
            kind="ggn_derived_factors",
            representation={"kind": "ggn_derived_factors"},
            batch={"ggn_factors": dict(factors)},
            identity_fields={"ggn_factors": _tree_signature(factors)},
        )

    @staticmethod
    def matrix_free(*, operator: Operator) -> Metric:
        """Build a matrix-free metric from a named sibling product.

        Returns:
            Matrix-free metric declaration.

        Raises:
            MaterializationError: If the sibling product is anonymous.
        """
        if operator.spec.family == operator.spec.kind:
            message = "matrix_free metric requires a named operator"
            raise MaterializationError(message)

        if operator.spec.kind not in {
            "ggnvp",
            "fisher_vp",
            "sampled_fisher_vp",
            "empirical_fisher_vp",
        }:
            message = "matrix_free metric requires a GGN or Fisher operator"
            raise MaterializationError(message)

        return Metric(
            kind="matrix_free",
            representation={"kind": "matrix_free", "operator": operator.spec.family},
            batch={},
            identity_fields={"operator": operator.spec.family},
            product_name=operator.spec.family,
        )


class _DampingNamespace:
    @staticmethod
    def scalar(lam: float) -> Damping:
        """Build scalar damping.

        Returns:
            Scalar damping declaration.
        """
        _require_nonnegative_float(lam, "scalar damping")

        return Damping("scalar", float(lam))

    @staticmethod
    def per_group(values: Mapping[str, float]) -> Damping:
        """Build per-group damping.

        Returns:
            Per-group damping declaration.

        Raises:
            MaterializationError: If the mapping is empty or invalid.
        """
        if not values:
            message = "per_group damping requires values"
            raise MaterializationError(message)

        result = {}

        for key, value in values.items():
            if not isinstance(key, str):
                message = "per_group damping keys must be strings"
                raise MaterializationError(message)

            _require_nonnegative_float(value, "per_group damping")
            result[key] = float(value)

        return Damping("per_group", result)

    @staticmethod
    def kfac_pi(lam: float, *, policy: str = "trace_norm") -> Damping:
        """Build KFAC pi damping.

        Returns:
            KFAC pi damping declaration.

        Raises:
            MaterializationError: If the damping or policy is invalid.
        """
        _require_nonnegative_float(lam, "kfac_pi damping")

        if policy not in {"trace_norm", "equal"}:
            message = f"kfac_pi policy is unsupported: {policy}"
            raise MaterializationError(message)

        return Damping("kfac_pi", float(lam), policy=policy)

    @staticmethod
    def eigenvalue_floor(lam: float) -> Damping:
        """Build EKFAC eigenvalue-floor damping.

        Returns:
            EKFAC eigenvalue-floor damping declaration.
        """
        _require_nonnegative_float(lam, "eigenvalue_floor damping")

        return Damping("eigenvalue_floor", float(lam))


loss = _LossNamespace()


class _LikelihoodNamespace:
    @staticmethod
    def categorical(
        *,
        output: str,
        labels: str,
        sample_space: str = "terms",
        denominator: str = "num_tokens",
        label_policy: str = "explicit",
    ) -> Likelihood:
        """Build a categorical likelihood declaration.

        Returns:
            Categorical likelihood declaration.

        Raises:
            MaterializationError: If a closed-set field is invalid.
        """
        _require_nonempty_string(output, "likelihood output")
        _require_nonempty_string(labels, "likelihood labels")

        if sample_space != "terms":
            message = f"categorical sample_space is unsupported: {sample_space}"
            raise MaterializationError(message)

        if denominator != "num_tokens":
            message = f"categorical denominator is unsupported: {denominator}"
            raise MaterializationError(message)

        if label_policy != "explicit":
            message = f"categorical label_policy is unsupported: {label_policy}"
            raise MaterializationError(message)

        return Likelihood(
            kind="categorical",
            output=output,
            fields={
                "labels": labels,
                "sample_space": sample_space,
                "denominator": denominator,
                "label_policy": label_policy,
            },
        )

    @staticmethod
    def gaussian(
        *,
        output: str,
        target: str,
        noise: float,
        sample_space: str = "terms",
        denominator: str = "num_examples",
    ) -> Likelihood:
        """Build a Gaussian likelihood declaration.

        Returns:
            Gaussian likelihood declaration.

        Raises:
            MaterializationError: If a closed-set field is invalid.
        """
        _require_nonempty_string(output, "likelihood output")
        _require_nonempty_string(target, "likelihood target")
        _require_positive_float(noise, "gaussian noise")

        if sample_space != "terms":
            message = f"gaussian sample_space is unsupported: {sample_space}"
            raise MaterializationError(message)

        if denominator != "num_examples":
            message = f"gaussian denominator is unsupported: {denominator}"
            raise MaterializationError(message)

        return Likelihood(
            kind="gaussian",
            output=output,
            fields={
                "target": target,
                "noise": float(noise),
                "sample_space": sample_space,
                "denominator": denominator,
            },
        )


likelihood = _LikelihoodNamespace()


class _SamplesNamespace:
    @staticmethod
    def fixed_seed(seed: int, count: int) -> SampleSource:
        """Build a fixed-seed sampled-Fisher source.

        Returns:
            Sample source declaration.

        Raises:
            MaterializationError: If the seed or count is invalid.
        """
        if not isinstance(seed, int) or isinstance(seed, bool):
            message = "samples.fixed_seed seed must be an integer"
            raise MaterializationError(message)

        if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
            message = "samples.fixed_seed count must be a positive integer"
            raise MaterializationError(message)

        return SampleSource(
            kind="fixed_seed",
            count=count,
            identity_fields={"seed": seed, "count": count},
        )

    @staticmethod
    def table(*, table: torch.Tensor, identity: Any) -> SampleSource:
        """Build a fixed-table sampled-Fisher source.

        Returns:
            Sample source declaration.

        Raises:
            MaterializationError: If the table or identity is invalid.
        """
        _require_tensor(table, "samples.table table")

        if table.ndim < SAMPLE_TABLE_MIN_DIMS:
            message = "samples.table table must have example and sample axes"
            raise MaterializationError(message)

        if table.shape[1] <= 0:
            message = "samples.table sample axis must be nonempty"
            raise MaterializationError(message)

        try:
            identity_value = to_json_value(identity)
        except TypeError as error:
            raise MaterializationError(str(error)) from error

        return SampleSource(
            kind="table",
            count=int(table.shape[1]),
            identity_fields={
                "identity": identity_value,
                "table": tensor_signature(table),
            },
            table=table,
        )


samples = _SamplesNamespace()
metric = _MetricNamespace()
damping = _DampingNamespace()


def compose(*terms: Combine | str) -> Compose:
    """Build a sequential public composition expression.

    Returns:
        Sequential composition expression.

    Raises:
        MaterializationError: If the expression terms are invalid.
    """
    if not terms:
        message = "compose requires at least one term"
        raise MaterializationError(message)

    for term in terms:
        _require_combine_term(term)

    _require_source_positions(Compose(tuple(terms)))

    return Compose(tuple(terms))


def linear_combination(*weighted: tuple[float, Combine | str]) -> LinearCombination:
    """Build a weighted-sum public composition expression.

    Returns:
        Linear-combination expression.

    Raises:
        MaterializationError: If a weighted expression term is invalid.
    """
    if not weighted:
        message = "linear_combination requires at least one term"
        raise MaterializationError(message)

    normalized = []

    for item in weighted:
        if not isinstance(item, tuple) or len(item) != WEIGHTED_COMBINE_TERM_DIMS:
            message = "linear_combination terms must be coefficient-expression pairs"
            raise MaterializationError(message)

        coefficient, term = item
        normalized.append((_float_coefficient(coefficient), term))
        _require_combine_term(term)

    expression = LinearCombination(tuple(normalized))
    _require_source_positions(expression)

    return expression


def scaled_identity(coefficient: float) -> ScaledIdentity:
    """Build a scaled identity leaf.

    Returns:
        Scaled identity expression leaf.
    """
    return ScaledIdentity(_float_coefficient(coefficient))


def source(child: str) -> Source:
    """Build a batch-to-vector source leaf.

    Returns:
        Source expression leaf.
    """
    _require_nonempty_string(child, "source child")

    return Source(child)


def _float_coefficient(value: float) -> float:
    if isinstance(value, int | float) and not isinstance(value, bool):
        return float(value)

    message = "composition coefficient must be a number"
    raise MaterializationError(message)


def _require_combine_term(term: Combine | str) -> None:
    if isinstance(term, str):
        _require_nonempty_string(term, "composition child")

        return

    if isinstance(term, ScaledIdentity):
        return

    if isinstance(term, Source):
        _require_nonempty_string(term.child, "source child")

        return

    if isinstance(term, Compose):
        if not term.terms:
            message = "compose requires at least one term"
            raise MaterializationError(message)

        for nested in term.terms:
            _require_combine_term(nested)

        return

    if isinstance(term, LinearCombination):
        if not term.terms:
            message = "linear_combination requires at least one term"
            raise MaterializationError(message)

        for coefficient, nested in term.terms:
            _float_coefficient(coefficient)
            _require_combine_term(nested)

        return

    message = f"composition expression term is unsupported: {type(term).__name__}"
    raise MaterializationError(message)


def _require_source_positions(term: Combine | str) -> None:
    if isinstance(term, Compose):
        for nested in term.terms[:-1]:
            if _combine_contains_source(nested):
                message = "source may only seed the innermost compose argument"
                raise MaterializationError(message)

        for nested in term.terms:
            _require_source_positions(nested)

        return

    if isinstance(term, LinearCombination):
        for _, nested in term.terms:
            _require_source_positions(nested)


def _require_source_input_space(term: Combine | str) -> None:
    if not _combine_contains_source(term):
        return

    if not _combine_requires_external_vector(term):
        return

    message = "source-valued composition terms must be source-seeded"
    raise MaterializationError(message)


def _combine_requires_external_vector(term: Combine | str) -> bool:
    if isinstance(term, str | ScaledIdentity):
        return True

    if isinstance(term, Source):
        return False

    if isinstance(term, Compose):
        return _combine_requires_external_vector(term.terms[-1])

    if isinstance(term, LinearCombination):
        return any(
            _combine_requires_external_vector(nested) for _, nested in term.terms
        )

    message = f"composition expression term is unsupported: {type(term).__name__}"
    raise MaterializationError(message)


def _combine_contains_source(term: Combine | str) -> bool:
    if isinstance(term, Source):
        return True

    if isinstance(term, Compose):
        return any(_combine_contains_source(nested) for nested in term.terms)

    if isinstance(term, LinearCombination):
        return any(_combine_contains_source(nested) for _, nested in term.terms)

    return False


def _combine_child_names(term: Combine | str) -> tuple[str, ...]:
    if isinstance(term, str):
        return (term,)

    if isinstance(term, Source):
        return (term.child,)

    if isinstance(term, ScaledIdentity):
        return ()

    if isinstance(term, Compose):
        return tuple(
            child for nested in term.terms for child in _combine_child_names(nested)
        )

    if isinstance(term, LinearCombination):
        return tuple(
            child for _, nested in term.terms for child in _combine_child_names(nested)
        )

    message = f"composition expression term is unsupported: {type(term).__name__}"
    raise MaterializationError(message)


def _combine_signature(term: Combine | str) -> Mapping[str, Any]:
    if isinstance(term, str):
        return {"kind": "child", "name": term}

    if isinstance(term, Source):
        return {"kind": "source", "child": term.child}

    if isinstance(term, ScaledIdentity):
        return {"kind": "scaled_identity", "coefficient": term.coefficient}

    if isinstance(term, Compose):
        return {
            "kind": "compose",
            "terms": tuple(_combine_signature(nested) for nested in term.terms),
        }

    if isinstance(term, LinearCombination):
        return {
            "kind": "linear_combination",
            "terms": tuple(
                {
                    "coefficient": coefficient,
                    "term": _combine_signature(nested),
                }
                for coefficient, nested in term.terms
            ),
        }

    message = f"composition expression term is unsupported: {type(term).__name__}"
    raise MaterializationError(message)


class _SearchNamespace:
    @staticmethod
    def admission() -> SearchStrategy:
        """Build an admission-only search strategy.

        Returns:
            Admission-only search strategy.
        """
        return _search_strategy("admission")

    @staticmethod
    def smoke() -> SearchStrategy:
        """Build a smoke search strategy.

        Returns:
            Smoke search strategy.
        """
        return _search_strategy("smoke")

    @staticmethod
    def fast() -> SearchStrategy:
        """Build a fast search strategy.

        Returns:
            Fast search strategy.
        """
        return _search_strategy("fast")

    @staticmethod
    def balanced(
        *,
        retain: int,
        compile_horizons: Sequence[int] = (),
    ) -> SearchStrategy:
        """Build a balanced search strategy.

        Returns:
            Balanced search strategy.
        """
        return _search_strategy(
            "balanced",
            retained_top_count=retain,
            compile_call_horizons=_int_tuple(compile_horizons, "compile_horizons"),
        )

    @staticmethod
    def thorough(
        *,
        retain: int,
        compile_horizons: Sequence[int],
        variance_repeats: int,
    ) -> SearchStrategy:
        """Build a thorough search strategy.

        Returns:
            Thorough search strategy.
        """
        return _search_strategy(
            "thorough",
            retained_top_count=retain,
            compile_call_horizons=_int_tuple(compile_horizons, "compile_horizons"),
            variance_repeat_count=variance_repeats,
        )

    @staticmethod
    def exhaustive() -> SearchStrategy:
        """Build an exhaustive search strategy.

        Returns:
            Exhaustive search strategy.
        """
        return _search_strategy("exhaustive")


search = _SearchNamespace()


class _SpaceNamespace:
    @staticmethod
    def standard(
        *,
        autodiff: AD | None = None,
        vectorization: Vectorization | None = None,
        precision: Precision | None = None,
        compile: Compile | None = None,
        layout: Layout | None = None,
        memory: Memory | None = None,
    ) -> SearchSpace:
        """Build the standard package-owned search space.

        Returns:
            Standard search space with the operator reference row.
        """
        components = tuple(
            component
            for component in (
                autodiff,
                vectorization,
                precision,
                compile,
                layout,
                memory,
            )
            if component is not None
        )

        return SearchSpace(components=components)


space = _SpaceNamespace()


class _CohortNamespace:
    @staticmethod
    def layout_coherence(settings_keys: Sequence[str]) -> CohortConstraint:
        """Build a cohort rule that pins settings across covered products.

        Returns:
            Public cohort rule.
        """
        keys = tuple(settings_keys)
        signature = {"kind": "layout_coherence", "settings_keys": keys}

        return CohortConstraint(
            name=f"layout_coherence:{stable_hash(signature)}",
            settings_keys=keys,
        )


cohort = _CohortNamespace()


def cuda(
    device: int | str = 0,
    accelerator: str = "cuda",
    *,
    timing: TimingPolicy | None = None,
    selection: SelectionPolicy | None = None,
    determinism: DeterminismPolicy | Mapping[str, Any] | None = None,
    environment: EnvironmentPolicy | Mapping[str, Any] | None = None,
) -> Target:
    """Build a CUDA target for public tuning.

    Returns:
        Public CUDA target.
    """
    _require_nonempty_string(accelerator, "accelerator")

    return Target(
        devices=(_cuda_device_name(device),),
        accelerator=accelerator,
        allowed_dtypes=_target_dtype_values(),
        allowed_attention_frontends=_target_axis_values(("attention.frontend",)),
        allowed_sdpa_kernels=_target_axis_values(("attention.sdpa_kernel",)),
        allowed_sharding_modes=_target_axis_values(("distributed.strategy",)),
        timing_policy=TimingPolicy() if timing is None else timing,
        selection_policy=SelectionPolicy() if selection is None else selection,
        determinism_policy=_determinism_policy(determinism),
        environment_policy=_environment_policy(environment),
    )


def problem(
    product: Operator,
    *,
    data: Iterable[Batch],
    vectors: Iterable[TensorTree],
    target: Target,
    space: SearchSpace,
    search: SearchStrategy,
) -> _LowerProblem:
    """Build a public single-product tuning problem.

    Returns:
        Lower-layer tuning problem for the typed product.
    """
    return _operator_problem(
        product,
        data=data,
        vectors=vectors,
        target=target,
        space=space,
        search=search,
        reference=None,
        probes=None,
    )


def autotune(
    problem: _LowerProblem,
    *,
    run_dir: Path | None = None,
    memory_backend: MemoryBackend | None = None,
    clock: Callable[[], float] = time.perf_counter,
) -> Plan:
    """Run search for a public tuning problem.

    Returns:
        Selected plan.
    """
    return _tune_problem(
        problem,
        run_dir=run_dir,
        memory_backend=memory_backend,
        clock=clock,
    )


def _is_product_data_mapping(
    data: Iterable[Batch] | Mapping[str, Iterable[Batch]],
) -> TypeGuard[Mapping[str, Iterable[Batch]]]:
    return isinstance(data, Mapping)


def _is_batch_data_iterable(
    data: Iterable[Batch] | Mapping[str, Iterable[Batch]],
) -> TypeGuard[Iterable[Batch]]:
    return not isinstance(data, Mapping)


def _required_product_data(
    data: Iterable[Batch] | Mapping[str, Iterable[Batch]],
    family: str,
) -> Iterable[Batch]:
    if _is_product_data_mapping(data):
        product_data = data.get(family)

        if product_data is None:
            message = f"vp.tune data is missing product: {family}"
            raise MaterializationError(message)

        return product_data

    if _is_batch_data_iterable(data):
        return data

    message = "vp.tune data must be an iterable or a product data mapping"
    raise MaterializationError(message)


def _is_product_vector_mapping(
    vectors: Iterable[TensorTree] | Mapping[str, Iterable[TensorTree]],
) -> TypeGuard[Mapping[str, Iterable[TensorTree]]]:
    return isinstance(vectors, Mapping)


def _is_vector_iterable(
    vectors: Iterable[TensorTree] | Mapping[str, Iterable[TensorTree]],
) -> TypeGuard[Iterable[TensorTree]]:
    return not isinstance(vectors, Mapping)


def _required_product_vectors(
    vectors: Iterable[TensorTree] | Mapping[str, Iterable[TensorTree]],
    family: str,
    product_count: int,
) -> Iterable[TensorTree]:
    if _is_product_vector_mapping(vectors):
        product_vectors = vectors.get(family)

        if product_vectors is None:
            message = f"vp.tune vectors are missing product: {family}"
            raise MaterializationError(message)

        return product_vectors

    if _is_vector_iterable(vectors):
        if product_count != 1:
            message = "vp.tune vectors must be keyed by product for multiple products"
            raise MaterializationError(message)

        return vectors

    message = "vp.tune vectors must be an iterable or a product vector mapping"
    raise MaterializationError(message)


def _cohort_assignments_from_space(
    settings_keys: tuple[str, ...],
    space: SearchSpace,
) -> tuple[Mapping[str, Any], ...]:
    domains = []

    for key in settings_keys:
        values = space.axes.get(key)

        if values is None:
            message = f"cohort axis is not declared in search space: {key}"
            raise MaterializationError(message)

        domains.append(tuple(values))

    return tuple(
        dict(zip(settings_keys, values, strict=True))
        for values in itertools.product(*domains)
    )


def _lower_cohort_constraints(
    constraints: Sequence[CohortConstraint],
    space: SearchSpace,
) -> tuple[_LowerCohortConstraint, ...]:
    return tuple(constraint.lower(space) for constraint in constraints)


def _require_unique_product_names(products: Sequence[Operator]) -> None:
    names = tuple(product.spec.family for product in products)

    if len(set(names)) != len(names):
        message = "vp.tune product names must be unique"
        raise MaterializationError(message)


def _require_product_model(product: Operator, model: Model) -> None:
    if to_json_value(product.model.signature()) != to_json_value(model.signature()):
        message = f"vp.tune product model differs: {product.spec.family}"
        raise MaterializationError(message)


def _require_selected_products(plan: Plan, products: Sequence[Operator]) -> None:
    missing = tuple(
        product.spec.family
        for product in products
        if product.spec.family not in plan.selected
    )

    if missing:
        message = f"vp.tune selected no row for products: {missing}"
        raise MaterializationError(message)


def tune(
    *,
    products: Sequence[Operator],
    model: Model,
    data: Iterable[Batch] | Mapping[str, Iterable[Batch]],
    vectors: Iterable[TensorTree] | Mapping[str, Iterable[TensorTree]],
    target: Target,
    space: SearchSpace,
    search: SearchStrategy,
    cohort_constraints: Sequence[CohortConstraint] = (),
    run_dir: Path | None = None,
    memory_backend: MemoryBackend | None = None,
    clock: Callable[[], float] = time.perf_counter,
) -> "Run":
    """Tune public products and return tuned operators by name.

    Returns:
        Run containing tuned products.

    Raises:
        MaterializationError: If the product set is not lowered by this path.
    """
    if not products:
        message = "vp.tune requires at least one product"
        raise MaterializationError(message)

    _require_unique_product_names(products)

    for product in products:
        _require_product_model(product, model)

    product_data_source = tuple(data) if _is_batch_data_iterable(data) else data

    if len(products) != 1 or cohort_constraints:
        return _tune_public_run(
            products=products,
            data=product_data_source,
            vectors=vectors,
            target=target,
            space=space,
            search=search,
            cohort_constraints=cohort_constraints,
            run_dir=run_dir,
            memory_backend=memory_backend,
            clock=clock,
        )

    product = products[0]

    if operator_dependencies(product.spec):
        message = (
            f"vp.tune requires dependency products for {product.spec.family}: "
            f"{operator_dependencies(product.spec)}"
        )
        raise MaterializationError(message)

    product_data = _required_product_data(product_data_source, product.spec.family)
    product_vectors = _required_product_vectors(vectors, product.spec.family, 1)
    tuned = product.tune(
        data=product_data,
        vectors=product_vectors,
        target=target,
        space=space,
        search=search,
        run_dir=run_dir,
        memory_backend=memory_backend,
        clock=clock,
    )
    selected_plan = _operator_selected_plan(tuned)

    return Run(plan=selected_plan, operators={product.spec.family: tuned})


def _tune_public_run(
    *,
    products: Sequence[Operator],
    data: Iterable[Batch] | Mapping[str, Iterable[Batch]],
    vectors: Iterable[TensorTree] | Mapping[str, Iterable[TensorTree]],
    target: Target,
    space: SearchSpace,
    search: SearchStrategy,
    cohort_constraints: Sequence[CohortConstraint],
    run_dir: Path | None,
    memory_backend: MemoryBackend | None,
    clock: Callable[[], float],
) -> "Run":
    if run_dir is None:
        message = "vp.tune multi-product or cohort runs require run_dir"
        raise MaterializationError(message)

    lower_run = _LowerTuningRun(
        target=target.lower(search),
        families=tuple(
            Family(name=product.spec.family, operator=product.spec)
            for product in products
        ),
        problems=tuple(
            _operator_problem(
                product,
                data=_required_product_data(data, product.spec.family),
                vectors=_required_product_vectors(
                    vectors,
                    product.spec.family,
                    len(products),
                ),
                target=target,
                space=space,
                search=search,
                reference=None,
                probes=None,
            )
            for product in products
        ),
        cohort_constraints=_lower_cohort_constraints(cohort_constraints, space),
    )
    selected_plan = _tune_run(
        lower_run,
        run_dir=run_dir,
        memory_backend=memory_backend,
        clock=clock,
    )
    _require_selected_products(selected_plan, products)

    return Run(
        plan=selected_plan,
        operators={
            product.spec.family: dataclasses.replace(product, plan=selected_plan)
            for product in products
        },
    )


def gradient(model: Model, loss: Loss, name: str | None = None) -> Operator:
    """Build a typed gradient operator.

    Returns:
        Typed gradient operator.
    """
    return _typed_gradient(model, loss, name)


def hvp(model: Model, loss: Loss, name: str | None = None) -> Operator:
    """Build a typed HVP operator.

    Returns:
        Typed HVP operator.
    """
    return _typed_hvp(model, loss, name)


def ggnvp(model: Model, loss: Loss, name: str | None = None) -> Operator:
    """Build a typed GGN vector product.

    Returns:
        Typed GGN operator.
    """
    return _typed_ggnvp(model, loss, name)


def fisher_vp(
    model: Model,
    likelihood: Likelihood,
    name: str | None = None,
) -> Operator:
    """Build a typed Fisher vector product.

    Returns:
        Typed Fisher operator.
    """
    return _typed_fisher_vp(model, likelihood, name)


def sampled_fisher_vp(
    model: Model,
    likelihood: Likelihood,
    name: str | None = None,
    *,
    samples: SampleSource,
) -> Operator:
    """Build a typed sampled-Fisher vector product.

    Returns:
        Typed sampled-Fisher operator.
    """
    return _typed_sampled_fisher_vp(model, likelihood, samples, name)


def jvp(model: Model, output: Output, name: str | None = None) -> Operator:
    """Build a typed JVP operator.

    Returns:
        Typed JVP operator.
    """
    return _typed_jvp(model, output, name)


def vjp(model: Model, output: Output, name: str | None = None) -> Operator:
    """Build a typed VJP operator.

    Returns:
        Typed VJP operator.
    """
    return _typed_vjp(model, output, name)


def per_example_gradient(
    model: Model,
    loss: Loss,
    name: str | None = None,
) -> Operator:
    """Build a typed per-example-gradient operator.

    Returns:
        Typed per-example-gradient operator.
    """
    return _typed_per_example_gradient(model, loss, name)


def empirical_fisher_vp(
    model: Model,
    loss: Loss,
    name: str | None = None,
) -> Operator:
    """Build a typed empirical-Fisher vector product.

    Returns:
        Typed empirical-Fisher operator.
    """
    return _typed_empirical_fisher_vp(model, loss, name)


def composition(
    model: Model,
    name: str | None = None,
    *,
    children: Sequence[str],
    combine: Combine | str,
) -> Operator:
    """Build a typed composition operator.

    Returns:
        Typed composition operator.
    """
    child_order = _public_composition_children(children)
    _require_combine_term(combine)
    _require_source_positions(combine)
    _require_source_input_space(combine)
    _require_combine_children(combine, child_order)
    family = _operator_family(name, "composition")
    spec = _operator_builders.composition(
        family,
        "typed_composition",
        aggregation="none",
        children=child_order,
    )
    call_inputs = _composition_call_inputs(combine)
    spec = dataclasses.replace(
        spec,
        semantics={
            **dict(spec.semantics),
            "public_children": child_order,
            "combine": _combine_signature(combine),
            "call_inputs": call_inputs,
        },
    )

    return Operator(
        model=model,
        spec=spec,
        call_inputs=call_inputs,
        default_settings={
            "composition.execution": "stream_child_outputs",
            "composition.child_evaluation": "selected_child_rows",
            "composition.validation": "validate_composed_output",
        },
        changed_axes=(
            "composition.execution",
            "composition.child_evaluation",
            "composition.validation",
        ),
    )


def _public_composition_children(children: Sequence[str]) -> tuple[str, ...]:
    child_order = tuple(children)

    if not child_order:
        message = "composition children must be nonempty"
        raise MaterializationError(message)

    for child in child_order:
        _require_nonempty_string(child, "composition child")

    if len(set(child_order)) != len(child_order):
        message = "composition children must be unique"
        raise MaterializationError(message)

    return child_order


def _require_combine_children(
    combine: Combine | str,
    children: tuple[str, ...],
) -> None:
    child_names = _combine_child_names(combine)

    if set(child_names) == set(children):
        return

    message = "composition combine leaves must match children"
    raise MaterializationError(message)


def _composition_call_inputs(combine: Combine | str) -> tuple[str, ...]:
    if _combine_contains_source(combine):
        return ("batch",)

    return ("batch", "vector")


def _typed_gradient(model: Model, typed_loss: Loss, name: str | None) -> Operator:
    family = _operator_family(name, "gradient")
    spec = _operator_builders.gradient(
        family,
        "typed_loss",
        aggregation="sum",
    )

    return Operator(
        model=model,
        spec=spec,
        call_inputs=("batch",),
        default_settings={"gradient.path": "torch_autograd_grad"},
        scalar_objectives={"typed_loss": _loss_scalar_objective(model, typed_loss)},
    )


def _typed_hvp(model: Model, typed_loss: Loss, name: str | None) -> Operator:
    family = _operator_family(name, "hvp")
    spec = _operator_builders.hvp(
        family,
        "typed_loss",
        aggregation="sum",
    )

    return Operator(
        model=model,
        spec=spec,
        call_inputs=("batch", "vector"),
        default_settings={"hvp.path": "reverse_over_reverse"},
        scalar_objectives={"typed_loss": _loss_scalar_objective(model, typed_loss)},
    )


def _typed_ggnvp(model: Model, typed_loss: Loss, name: str | None) -> Operator:
    family = _operator_family(name, "ggn")
    spec = _operator_builders.ggnvp(
        family,
        typed_loss.output,
        aggregation="sum",
    )

    return Operator(
        model=model,
        spec=spec,
        call_inputs=("batch", "vector"),
        default_settings={
            "ggn.jvp_path": "torch_func_jvp",
            "ggn.loss_hessian_path": "autodiff_loss_hvp",
            "ggn.loss_hessian_kernel": "dense_global",
            "ggn.vjp_path": "torch_func_vjp",
            **_torch_func_settings(requires_forward_ad=True),
        },
        function_objectives={
            typed_loss.output: _ModelFieldObjective(model, typed_loss.output)
        },
        batch_transform=_loss_hessian_batch_transform(model, typed_loss),
        changed_axes=(
            "ggn.jvp_path",
            "ggn.loss_hessian_path",
            "ggn.loss_hessian_kernel",
            "ggn.vjp_path",
        ),
    )


def _typed_fisher_vp(
    model: Model,
    typed_likelihood: Likelihood,
    name: str | None,
) -> Operator:
    if typed_likelihood.kind == "categorical":
        message = "exact categorical Fisher is represented by GGNVP"
        raise MaterializationError(message)

    if typed_likelihood.kind != "gaussian":
        message = f"likelihood kind is not lowered: {typed_likelihood.kind}"
        raise MaterializationError(message)

    family = _operator_family(name, "fisher")
    spec = _operator_builders.fisher_vp(
        family,
        "typed_gaussian_score",
        aggregation="mean_per_example",
        distribution="explicit_score_gradients",
        label_policy="explicit_scores",
        sample_space=_likelihood_string_field(typed_likelihood, "sample_space"),
        score_reduction="none",
        denominator=_likelihood_string_field(typed_likelihood, "denominator"),
    )

    return Operator(
        model=model,
        spec=spec,
        call_inputs=("batch", "vector"),
        default_settings={
            "fisher.expectation_path": "explicit_full_expectation_score_rows",
            "fisher.accumulation": "materialize_score_gradients",
        },
        function_objectives={
            "typed_gaussian_score": _GaussianScoreTermsObjective(
                model=model,
                output=typed_likelihood.output,
                target=_likelihood_string_field(typed_likelihood, "target"),
                noise=_likelihood_float_field(typed_likelihood, "noise"),
            )
        },
        batch_transform=_likelihood_score_gradient_batch_transform(
            model,
            typed_likelihood,
        ),
        changed_axes=(
            "fisher.expectation_path",
            "fisher.accumulation",
        ),
    )


def _typed_sampled_fisher_vp(
    model: Model,
    typed_likelihood: Likelihood,
    sample_source: SampleSource,
    name: str | None,
) -> Operator:
    if typed_likelihood.kind not in {"categorical", "gaussian"}:
        message = f"likelihood kind is not lowered: {typed_likelihood.kind}"
        raise MaterializationError(message)

    family = _operator_family(name, "sampled_fisher")
    spec = _operator_builders.sampled_fisher_vp(
        family,
        "typed_sampled_scores",
        aggregation="mean_per_example",
        distribution="explicit_score_gradients",
        label_policy="sampled_labels",
        sample_count=sample_source.count,
        sample_source=sample_source.runtime_sample_source(),
        sampling_bound={"kind": "disabled"},
        score_reduction="none",
        denominator=_sampled_fisher_denominator(typed_likelihood),
    )
    spec = _operator_with_sample_source_identity(
        spec,
        typed_likelihood,
        sample_source,
    )

    return Operator(
        model=model,
        spec=spec,
        call_inputs=("batch", "vector"),
        default_settings={
            "sampled_fisher.accumulation": "materialize_score_gradients",
            "sampled_fisher.sample_source": sample_source.runtime_sample_source(),
            "sampled_fisher.exact_fisher_check": "disabled",
        },
        batch_transform=_SampledFisherScoreGradientBatch(
            model,
            typed_likelihood,
            sample_source,
        ),
        changed_axes=(
            "sampled_fisher.accumulation",
            "sampled_fisher.sample_source",
            "sampled_fisher.exact_fisher_check",
        ),
    )


def _typed_per_example_gradient(
    model: Model,
    typed_loss: Loss,
    name: str | None,
) -> Operator:
    family = _operator_family(name, "per_example_gradient")
    spec = _operator_builders.per_example_gradient(
        family,
        "typed_per_example_loss",
        aggregation="sum",
        example_loss_reduction="per_example",
    )

    return Operator(
        model=model,
        spec=spec,
        call_inputs=("batch",),
        default_settings={
            "per_example_gradient.grad_path": "torch_autograd_grad_loop",
            "per_example_gradient.accumulation": "stacked_leading_axis",
        },
        function_objectives={
            "typed_per_example_loss": _loss_per_example_objective(model, typed_loss)
        },
        changed_axes=(
            "per_example_gradient.grad_path",
            "per_example_gradient.accumulation",
        ),
    )


def _typed_empirical_fisher_vp(
    model: Model,
    typed_loss: Loss,
    name: str | None,
) -> Operator:
    family = _operator_family(name, "empirical_fisher")
    spec = _operator_builders.empirical_fisher_vp(
        family,
        "typed_per_example_loss",
        aggregation="mean_per_example",
        example_loss_reduction="per_example",
        denominator="num_examples",
    )

    return Operator(
        model=model,
        spec=spec,
        call_inputs=("batch", "vector"),
        default_settings={
            "empirical_fisher.grad_path": "torch_autograd_grad_loop",
            "schedule.per_example": "loop",
        },
        function_objectives={
            "typed_per_example_loss": _loss_per_example_objective(model, typed_loss)
        },
        changed_axes=(
            "empirical_fisher.grad_path",
            "schedule.per_example",
        ),
    )


def _loss_scalar_objective(model: Model, typed_loss: Loss) -> ScalarObjective:
    if typed_loss.kind == "from_scalar":
        if typed_loss.objective is None:
            message = "loss.from_scalar requires a scalar objective"
            raise MaterializationError(message)

        return typed_loss.objective

    if typed_loss.kind == "softmax_cross_entropy":
        return _SoftmaxCrossEntropyObjective(
            model=model,
            output=typed_loss.output,
            labels=_loss_string_field(typed_loss, "labels"),
            mask=_loss_optional_string_field(typed_loss, "mask"),
            reduction=_loss_string_field(typed_loss, "reduction"),
            denominator=_loss_string_field(typed_loss, "denominator"),
        )

    if typed_loss.kind == "kl":
        return _KLLossObjective(
            model=model,
            output=typed_loss.output,
            target=_loss_string_field(typed_loss, "target"),
            mask=_loss_optional_string_field(typed_loss, "mask"),
            reduction=_loss_string_field(typed_loss, "reduction"),
            denominator=_loss_string_field(typed_loss, "denominator"),
        )

    if typed_loss.kind == "mse":
        return _MSELossObjective(
            model=model,
            output=typed_loss.output,
            target=_loss_string_field(typed_loss, "target"),
            mask=_loss_optional_string_field(typed_loss, "mask"),
            reduction=_loss_string_field(typed_loss, "reduction"),
            denominator=_loss_string_field(typed_loss, "denominator"),
        )

    message = f"loss kind is not lowered: {typed_loss.kind}"
    raise MaterializationError(message)


def _loss_per_example_objective(
    model: Model,
    typed_loss: Loss,
) -> FunctionObjective:
    if typed_loss.kind == "softmax_cross_entropy":
        return _SoftmaxCrossEntropyPerExampleObjective(
            model=model,
            output=typed_loss.output,
            labels=_loss_string_field(typed_loss, "labels"),
            mask=_loss_optional_string_field(typed_loss, "mask"),
            reduction=_loss_string_field(typed_loss, "reduction"),
            denominator=_loss_string_field(typed_loss, "denominator"),
        )

    message = f"loss kind has no per-example lowering: {typed_loss.kind}"
    raise MaterializationError(message)


def _loss_hessian_batch_transform(
    model: Model,
    typed_loss: Loss,
) -> Callable[[Batch], Batch]:
    if typed_loss.kind == "softmax_cross_entropy":
        return _SoftmaxCrossEntropyLossHessianBatch(
            model=model,
            output=typed_loss.output,
            labels=_loss_string_field(typed_loss, "labels"),
            mask=_loss_optional_string_field(typed_loss, "mask"),
            reduction=_loss_string_field(typed_loss, "reduction"),
            denominator=_loss_string_field(typed_loss, "denominator"),
        )

    if typed_loss.kind == "kl":
        return _KLLossHessianBatch(
            model=model,
            output=typed_loss.output,
            target=_loss_string_field(typed_loss, "target"),
            mask=_loss_optional_string_field(typed_loss, "mask"),
            reduction=_loss_string_field(typed_loss, "reduction"),
            denominator=_loss_string_field(typed_loss, "denominator"),
        )

    if typed_loss.kind == "mse":
        return _MSELossHessianBatch(
            model=model,
            output=typed_loss.output,
            target=_loss_string_field(typed_loss, "target"),
            mask=_loss_optional_string_field(typed_loss, "mask"),
            reduction=_loss_string_field(typed_loss, "reduction"),
            denominator=_loss_string_field(typed_loss, "denominator"),
        )

    if typed_loss.kind == "declared_psd":
        if typed_loss.hessian_factor is None:
            message = "loss.declared_psd requires factors"
            raise MaterializationError(message)

        return _DeclaredPSDLossHessianBatch(
            model=model,
            output=typed_loss.output,
            factor=typed_loss.hessian_factor,
        )

    if typed_loss.kind == "declared_psd_matrix_free":
        if typed_loss.hessian_matvec is None:
            message = "loss.declared_psd_matrix_free requires matvec"
            raise MaterializationError(message)

        return _DeclaredPSDMatrixFreeLossHessianBatch(
            model=model,
            output=typed_loss.output,
            matvec=typed_loss.hessian_matvec,
        )

    message = f"loss kind has no output-Hessian lowering: {typed_loss.kind}"
    raise MaterializationError(message)


def _likelihood_score_gradient_batch_transform(
    model: Model,
    typed_likelihood: Likelihood,
) -> Callable[[Batch], Batch]:
    if typed_likelihood.kind == "gaussian":
        return _GaussianScoreGradientBatch(
            model=model,
            output=typed_likelihood.output,
            target=_likelihood_string_field(typed_likelihood, "target"),
            noise=_likelihood_float_field(typed_likelihood, "noise"),
        )

    message = f"likelihood kind has no score-gradient lowering: {typed_likelihood.kind}"
    raise MaterializationError(message)


def _sampled_fisher_denominator(typed_likelihood: Likelihood) -> str:
    denominator = _likelihood_string_field(typed_likelihood, "denominator")

    if typed_likelihood.kind == "categorical":
        if denominator != "num_tokens":
            message = f"categorical denominator is unsupported: {denominator}"
            raise MaterializationError(message)

        return "num_tokens"

    if typed_likelihood.kind == "gaussian":
        if denominator != "num_examples":
            message = f"gaussian denominator is unsupported: {denominator}"
            raise MaterializationError(message)

        return "num_examples"

    message = (
        f"likelihood kind has no sampled-Fisher denominator: {typed_likelihood.kind}"
    )
    raise MaterializationError(message)


def _sampled_likelihood_scores(
    prediction: torch.Tensor,
    typed_likelihood: Likelihood,
    sample_source: SampleSource,
) -> tuple[torch.Tensor, dict[str, int]]:
    if typed_likelihood.kind == "gaussian":
        return _sampled_gaussian_scores(prediction, typed_likelihood, sample_source)

    if typed_likelihood.kind == "categorical":
        return _sampled_categorical_scores(prediction, sample_source)

    message = f"likelihood kind has no sampled-score lowering: {typed_likelihood.kind}"
    raise MaterializationError(message)


def _sampled_gaussian_scores(
    prediction: torch.Tensor,
    typed_likelihood: Likelihood,
    sample_source: SampleSource,
) -> tuple[torch.Tensor, dict[str, int]]:
    if prediction.ndim == 0:
        message = "gaussian sampled Fisher requires a leading example axis"
        raise MaterializationError(message)

    samples_tensor = _gaussian_sample_table(
        prediction,
        typed_likelihood,
        sample_source,
    )
    noise = prediction.new_tensor(_likelihood_float_field(typed_likelihood, "noise"))
    residual = (samples_tensor - prediction.unsqueeze(1)) / noise
    scores = -0.5 * residual.square().reshape(
        prediction.shape[0],
        sample_source.count,
        -1,
    ).sum(dim=-1)

    return scores, {"num_examples": prediction.shape[0]}


def _gaussian_sample_table(
    prediction: torch.Tensor,
    typed_likelihood: Likelihood,
    sample_source: SampleSource,
) -> torch.Tensor:
    expected_shape = (prediction.shape[0], sample_source.count, *prediction.shape[1:])

    if sample_source.kind == "table":
        if sample_source.table is None:
            message = "samples.table requires a sample tensor"
            raise MaterializationError(message)

        table = sample_source.table.to(device=prediction.device, dtype=prediction.dtype)

        if tuple(table.shape) != expected_shape:
            message = "gaussian sample table shape must be (examples, samples, ...)"
            raise MaterializationError(message)

        return table

    if sample_source.kind == "fixed_seed":
        seed = _sample_source_int_field(sample_source, "seed")
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        noise = prediction.new_tensor(
            _likelihood_float_field(typed_likelihood, "noise")
        )
        eps = torch.randn(
            expected_shape,
            dtype=prediction.dtype,
            device="cpu",
            generator=generator,
        ).to(device=prediction.device)

        return prediction.detach().unsqueeze(1) + noise * eps

    message = f"sample source kind is not lowered: {sample_source.kind}"
    raise MaterializationError(message)


def _sampled_categorical_scores(
    logits: torch.Tensor,
    sample_source: SampleSource,
) -> tuple[torch.Tensor, dict[str, int]]:
    if logits.ndim < MIN_CLASS_LOGIT_DIMS:
        message = "categorical sampled Fisher logits must include a class axis"
        raise MaterializationError(message)

    samples_tensor = _categorical_sample_table(logits, sample_source)
    log_probs = torch.log_softmax(logits, dim=-1)
    expanded = log_probs.unsqueeze(1).expand(
        logits.shape[0],
        sample_source.count,
        *logits.shape[1:],
    )
    gathered = torch.gather(expanded, -1, samples_tensor.unsqueeze(-1)).squeeze(-1)

    return gathered, {"num_tokens": int(torch.tensor(logits.shape[:-1]).prod().item())}


def _categorical_sample_table(
    logits: torch.Tensor,
    sample_source: SampleSource,
) -> torch.Tensor:
    expected_shape = (logits.shape[0], sample_source.count, *logits.shape[1:-1])

    if sample_source.kind == "table":
        if sample_source.table is None:
            message = "samples.table requires a sample tensor"
            raise MaterializationError(message)

        table = sample_source.table.to(device=logits.device)

        if table.dtype != torch.long:
            message = "categorical sample table must have dtype torch.long"
            raise MaterializationError(message)

        if tuple(table.shape) != expected_shape:
            message = "categorical sample table shape must be (examples, samples, ...)"
            raise MaterializationError(message)

        _require_categorical_sample_range(table, logits.shape[-1])

        return table

    if sample_source.kind == "fixed_seed":
        seed = _sample_source_int_field(sample_source, "seed")
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        probabilities = torch.softmax(logits.detach(), dim=-1)
        flat_probabilities = probabilities.reshape(-1, logits.shape[-1]).cpu()
        flat_samples = torch.multinomial(
            flat_probabilities,
            sample_source.count,
            replacement=True,
            generator=generator,
        )
        samples = flat_samples.reshape(
            *logits.shape[:-1],
            sample_source.count,
        ).movedim(-1, 1)

        return samples.to(device=logits.device)

    message = f"sample source kind is not lowered: {sample_source.kind}"
    raise MaterializationError(message)


def _require_categorical_sample_range(table: torch.Tensor, class_count: int) -> None:
    if torch.any(table < 0) or torch.any(table >= class_count):
        message = "categorical sample table contains an invalid class index"
        raise MaterializationError(message)


def _sample_source_int_field(sample_source: SampleSource, field: str) -> int:
    value = sample_source.identity_fields.get(field)

    if not isinstance(value, int) or isinstance(value, bool):
        message = f"sample source {field} must be an integer"
        raise MaterializationError(message)

    return value


def _likelihood_string_field(
    typed_likelihood: Likelihood,
    field: str,
) -> str:
    value = typed_likelihood.fields.get(field)

    if not isinstance(value, str) or not value:
        message = f"likelihood {field} must be a nonempty string"
        raise MaterializationError(message)

    return value


def _likelihood_float_field(
    typed_likelihood: Likelihood,
    field: str,
) -> float:
    value = typed_likelihood.fields.get(field)

    if not isinstance(value, int | float) or isinstance(value, bool):
        message = f"likelihood {field} must be a number"
        raise MaterializationError(message)

    return float(value)


def _loss_string_field(typed_loss: Loss, field: str) -> str:
    value = typed_loss.identity_fields.get(field)

    if not isinstance(value, str) or not value:
        message = f"loss {field} must be a nonempty string"
        raise MaterializationError(message)

    return value


def _loss_optional_string_field(typed_loss: Loss, field: str) -> str | None:
    value = typed_loss.identity_fields.get(field)

    if value is None:
        return None

    if not isinstance(value, str) or not value:
        message = f"loss {field} must be a nonempty string"
        raise MaterializationError(message)

    return value


def _softmax_cross_entropy_losses(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    _require_softmax_cross_entropy_shapes(logits, labels)
    class_count = logits.shape[-1]

    return torch.nn.functional.cross_entropy(
        logits.reshape(-1, class_count),
        labels.reshape(-1),
        reduction="none",
    ).reshape(labels.shape)


def _require_softmax_cross_entropy_shapes(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> None:
    if logits.ndim < MIN_CLASS_LOGIT_DIMS:
        message = "softmax_cross_entropy logits must include a class axis"
        raise MaterializationError(message)

    expected_label_shape = tuple(logits.shape[:-1])

    if tuple(labels.shape) != expected_label_shape:
        message = "softmax_cross_entropy labels must match logits without class axis"
        raise MaterializationError(message)


def _softmax_cross_entropy_loss_hessian(
    logits: torch.Tensor,
    *,
    mask: torch.Tensor | None,
    reduction: str,
    denominator: str,
) -> torch.Tensor:
    class_count = logits.shape[-1]
    probabilities = torch.softmax(logits, dim=-1)
    blocks = torch.diag_embed(probabilities) - (
        probabilities[..., :, None] * probabilities[..., None, :]
    )

    if mask is not None:
        blocks = (
            blocks
            * mask.to(
                device=logits.device,
                dtype=logits.dtype,
            )[..., None, None]
        )

    scale = _softmax_cross_entropy_loss_hessian_scale(
        logits,
        mask=mask,
        reduction=reduction,
        denominator=denominator,
    )
    flat_blocks = (blocks * scale).reshape(-1, class_count, class_count)

    return torch.block_diag(*tuple(flat_blocks))


def _kl_token_losses(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if logits.ndim < MIN_CLASS_LOGIT_DIMS:
        message = "loss.kl logits must include a class axis"
        raise MaterializationError(message)

    _require_same_shape(logits, target, "loss.kl target")
    _require_finite_public_tensor(logits, "loss.kl logits")
    _require_finite_public_tensor(target, "loss.kl target")

    if bool(torch.any(target < 0.0)):
        message = "loss.kl target must be nonnegative"
        raise MaterializationError(message)

    positive = target > 0.0
    safe_target = torch.where(positive, target, torch.ones_like(target))
    target_log = torch.where(positive, torch.log(safe_target), torch.zeros_like(target))
    log_probabilities = torch.log_softmax(logits, dim=-1)

    return (target * (target_log - log_probabilities)).sum(dim=-1)


def _kl_loss_hessian(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    mask: torch.Tensor | None,
    reduction: str,
    denominator: str,
) -> torch.Tensor:
    _kl_token_losses(logits, target)
    probabilities = torch.softmax(logits, dim=-1)
    blocks = torch.diag_embed(probabilities) - (
        probabilities[..., :, None] * probabilities[..., None, :]
    )
    target_mass = target.sum(dim=-1)
    blocks = blocks * target_mass[..., None, None]

    if mask is not None:
        blocks = (
            blocks
            * mask.to(
                device=logits.device,
                dtype=logits.dtype,
            )[..., None, None]
        )

    scale = _softmax_cross_entropy_loss_hessian_scale(
        logits,
        mask=mask,
        reduction=reduction,
        denominator=denominator,
    )
    class_count = logits.shape[-1]
    flat_blocks = (blocks * scale).reshape(-1, class_count, class_count)

    return torch.block_diag(*tuple(flat_blocks))


def _mse_element_losses(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    _require_same_shape(prediction, target, "loss.mse target")
    _require_finite_public_tensor(prediction, "loss.mse output")
    _require_finite_public_tensor(target, "loss.mse target")

    return (prediction - target).square()


def _mse_loss_hessian(
    prediction: torch.Tensor,
    *,
    mask: torch.Tensor | None,
    reduction: str,
    denominator: str,
) -> torch.Tensor:
    _require_mse_denominator(reduction, denominator)
    weights = torch.ones_like(prediction)

    if mask is not None:
        weights = weights * _mse_element_mask(mask, prediction)

    scale = _mse_hessian_scale(
        prediction,
        mask=mask,
        reduction=reduction,
    )

    return torch.diag((2.0 * scale * weights).reshape(-1))


def _declared_psd_loss_hessian(
    factor: torch.Tensor,
    output: torch.Tensor,
) -> torch.Tensor:
    _require_finite_public_tensor(output, "loss.declared_psd output")

    if factor.shape[0] != output.numel():
        message = "loss.declared_psd factors must match flattened output"
        raise MaterializationError(message)

    loss_hessian = factor @ factor.T
    _require_finite_public_tensor(loss_hessian, "loss.declared_psd Hessian")

    return loss_hessian


def _declared_psd_matrix_free_loss_hessian(
    matvec: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    output: torch.Tensor,
) -> torch.Tensor:
    _require_finite_public_tensor(output, "loss.declared_psd_matrix_free output")

    if not torch.is_floating_point(output):
        message = "loss.declared_psd_matrix_free output must be floating point"
        raise MaterializationError(message)

    output_numel = output.numel()

    if output_numel < 1:
        message = "loss.declared_psd_matrix_free output must be nonempty"
        raise MaterializationError(message)

    basis = torch.eye(output_numel, dtype=output.dtype, device=output.device)
    columns = []

    for flat_basis in basis:
        tangent = flat_basis.reshape_as(output)
        product = matvec(output, tangent)
        _require_tensor(product, "loss.declared_psd_matrix_free matvec result")
        _require_same_shape(
            product,
            output,
            "loss.declared_psd_matrix_free matvec result",
        )
        _require_finite_public_tensor(
            product,
            "loss.declared_psd_matrix_free matvec result",
        )
        columns.append(product.reshape(-1))

    loss_hessian = torch.stack(tuple(columns), dim=1)
    _require_finite_public_tensor(
        loss_hessian,
        "loss.declared_psd_matrix_free Hessian",
    )
    _require_declared_psd_matrix_free_certificate(loss_hessian)

    return loss_hessian


def _require_declared_psd_matrix_free_certificate(loss_hessian: torch.Tensor) -> None:
    if not torch.equal(loss_hessian, loss_hessian.T):
        message = "loss.declared_psd_matrix_free matvec must be symmetric"
        raise MaterializationError(message)

    eigenvalues = torch.linalg.eigvalsh(loss_hessian)
    smallest_ritz = eigenvalues[0]
    residual_norm = loss_hessian.new_tensor(0.0)

    if bool((smallest_ritz - residual_norm) < 0.0):
        message = "GGN requires a PSD output-space metric"
        raise MaterializationError(message)


def _softmax_cross_entropy_loss_hessian_scale(
    logits: torch.Tensor,
    *,
    mask: torch.Tensor | None,
    reduction: str,
    denominator: str,
) -> torch.Tensor:
    token_values = logits[..., 0]

    if reduction == "sum":
        return logits.new_tensor(1.0)

    if reduction == "mean":
        return logits.new_tensor(1.0 / float(token_values.numel()))

    if reduction == "token_mean":
        if denominator != "num_tokens":
            message = f"token_mean denominator is unsupported: {denominator}"
            raise MaterializationError(message)

        count = _token_count(token_values, mask)

        return count.to(device=logits.device, dtype=logits.dtype).reciprocal()

    message = f"loss reduction is unsupported: {reduction}"
    raise MaterializationError(message)


def _reduce_losses(
    losses: torch.Tensor,
    *,
    mask: torch.Tensor | None,
    reduction: str,
    denominator: str,
) -> torch.Tensor:
    if reduction == "sum":
        return losses.sum()

    if reduction == "mean":
        return losses.mean()

    if reduction == "token_mean":
        if denominator != "num_tokens":
            message = f"token_mean denominator is unsupported: {denominator}"
            raise MaterializationError(message)

        count = _token_count(losses, mask)

        return losses.sum() / count

    message = f"loss reduction is unsupported: {reduction}"
    raise MaterializationError(message)


def _reduce_mse_losses(
    losses: torch.Tensor,
    *,
    mask: torch.Tensor | None,
    reduction: str,
    denominator: str,
) -> torch.Tensor:
    _require_mse_denominator(reduction, denominator)

    if reduction == "sum":
        return losses.sum()

    if reduction == "mean":
        return losses.mean()

    if reduction == "token_mean":
        token_losses = losses if losses.ndim == 1 else losses.sum(dim=-1)
        count = _token_count(token_losses, mask)

        return token_losses.sum() / count

    message = f"loss reduction is unsupported: {reduction}"
    raise MaterializationError(message)


def _mse_hessian_scale(
    prediction: torch.Tensor,
    *,
    mask: torch.Tensor | None,
    reduction: str,
) -> torch.Tensor:
    if reduction == "sum":
        return prediction.new_tensor(1.0)

    if reduction == "mean":
        return prediction.new_tensor(1.0 / float(prediction.numel()))

    if reduction == "token_mean":
        token_values = prediction if prediction.ndim == 1 else prediction[..., 0]
        count = _token_count(token_values, mask)

        return count.to(device=prediction.device, dtype=prediction.dtype).reciprocal()

    message = f"loss reduction is unsupported: {reduction}"
    raise MaterializationError(message)


def _mse_mask(
    batch: Batch,
    key: str | None,
    output_shape: torch.Size,
) -> torch.Tensor | None:
    if len(output_shape) < 1:
        message = "loss.mse output must have at least one dimension"
        raise MaterializationError(message)

    expected = output_shape if len(output_shape) == 1 else output_shape[:-1]

    return _loss_mask(batch, key, expected)


def _mse_element_mask(mask: torch.Tensor, prediction: torch.Tensor) -> torch.Tensor:
    value = mask.to(device=prediction.device, dtype=prediction.dtype)

    if tuple(value.shape) == tuple(prediction.shape):
        return value

    return value.unsqueeze(-1)


def _token_count(losses: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    if mask is None:
        return losses.new_tensor(losses.numel())

    count = mask.sum()

    if bool(count <= 0):
        message = "softmax_cross_entropy token count must be positive"
        raise MaterializationError(message)

    return count


def _per_example_reduce_losses(
    losses: torch.Tensor,
    *,
    mask: torch.Tensor | None,
    reduction: str,
    denominator: str,
) -> torch.Tensor:
    if losses.ndim == 0:
        message = "per-example loss requires a leading example axis"
        raise MaterializationError(message)

    flat_losses = losses.reshape(losses.shape[0], -1)

    if mask is None:
        flat_mask = None
    else:
        flat_mask = mask.to(device=losses.device, dtype=losses.dtype).reshape(
            losses.shape[0],
            -1,
        )
        flat_losses = flat_losses * flat_mask

    if reduction == "sum":
        return flat_losses.sum(dim=1)

    if reduction == "mean":
        if flat_mask is None:
            return flat_losses.mean(dim=1)

        counts = flat_mask.sum(dim=1)
        _require_positive_counts(counts, "per-example mean denominator")

        return flat_losses.sum(dim=1) / counts

    if reduction == "token_mean":
        if denominator != "num_tokens":
            message = f"token_mean denominator is unsupported: {denominator}"
            raise MaterializationError(message)

        if flat_mask is None:
            counts = losses.new_full((losses.shape[0],), flat_losses.shape[1])
        else:
            counts = flat_mask.sum(dim=1)

        _require_positive_counts(counts, "per-example token denominator")

        return flat_losses.sum(dim=1) / counts

    message = f"loss reduction is unsupported: {reduction}"
    raise MaterializationError(message)


def _require_positive_counts(counts: torch.Tensor, name: str) -> None:
    if torch.any(counts <= 0):
        message = f"{name} must be positive"
        raise MaterializationError(message)


def _single_tensor_output(value: TensorTree, name: str) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value

    message = f"{name} output must be a tensor"
    raise MaterializationError(message)


def _batch_tensor(batch: Batch, key: str) -> torch.Tensor:
    value = _batch_value(batch, key)

    if isinstance(value, torch.Tensor):
        return value

    message = f"batch field must be a tensor: {key}"
    raise MaterializationError(message)


def _batch_long_tensor(batch: Batch, key: str) -> torch.Tensor:
    value = _batch_value(batch, key)

    if not isinstance(value, torch.Tensor):
        message = f"batch field must be a tensor: {key}"
        raise MaterializationError(message)

    if value.dtype != torch.long:
        message = f"batch field must have dtype torch.long: {key}"
        raise MaterializationError(message)

    return value


def _loss_mask(
    batch: Batch,
    key: str | None,
    expected_shape: torch.Size,
) -> torch.Tensor | None:
    if key is None:
        return None

    value = _batch_value(batch, key)

    if not isinstance(value, torch.Tensor):
        message = f"loss mask field must be a tensor: {key}"
        raise MaterializationError(message)

    if tuple(value.shape) != tuple(expected_shape):
        message = f"loss mask field shape mismatch: {key}"
        raise MaterializationError(message)

    if value.dtype == torch.bool:
        return value

    if value.is_floating_point():
        return value

    message = f"loss mask field must be bool or floating point: {key}"
    raise MaterializationError(message)


def _typed_jvp(model: Model, typed_output: Output, name: str | None) -> Operator:
    family = _operator_family(name, "jvp")
    spec = _operator_builders.jvp(
        family,
        typed_output.field,
        aggregation="sum",
    )

    return Operator(
        model=model,
        spec=spec,
        call_inputs=("batch", "vector"),
        default_settings={
            "jvp.path": "torch_func_jvp",
            **_torch_func_settings(requires_forward_ad=True),
        },
        function_objectives={
            typed_output.field: _ModelFieldObjective(model, typed_output.field)
        },
        changed_axes=("jvp.path",),
    )


def _typed_vjp(model: Model, typed_output: Output, name: str | None) -> Operator:
    family = _operator_family(name, "vjp")
    spec = _operator_builders.vjp(
        family,
        typed_output.field,
        aggregation="sum",
    )

    return Operator(
        model=model,
        spec=spec,
        call_inputs=("batch", "vector"),
        default_settings={
            "vjp.path": "torch_func_vjp",
            **_torch_func_settings(requires_forward_ad=False),
        },
        function_objectives={
            typed_output.field: _ModelFieldObjective(model, typed_output.field)
        },
        changed_axes=("vjp.path",),
    )


def _torch_func_settings(*, requires_forward_ad: bool) -> dict[str, Any]:
    return {
        "contains_autograd_call": False,
        "contains_backward_call": False,
        "uses_out_variant": False,
        "uses_data_dependent_control_flow": False,
        "uses_item": False,
        "has_dynamic_shape_output": False,
        "vectorization.randomness": "error",
        "requires_forward_ad": requires_forward_ad,
        "forward_ad_supported": True,
    }


def metric_vp(model: Model, metric: Metric, name: str | None = None) -> Operator:
    """Build a typed metric-vector product.

    Returns:
        Metric-vector operator.
    """
    family = _operator_family(name, "metric")
    spec = _operator_builders.metric(
        family,
        "typed_metric",
        aggregation="sum",
        representation=metric.representation,
    )

    return Operator(
        model=model,
        spec=spec,
        call_inputs=_metric_vector_call_inputs(metric),
        default_settings=_metric_default_settings(metric),
        metric=metric,
    )


def sqrt_metric_vp(model: Model, metric: Metric, name: str | None = None) -> Operator:
    """Build a typed metric square-root product.

    Returns:
        Metric square-root operator.
    """
    family = _operator_family(name, "sqrt_metric")
    spec = _operator_builders.sqrt_metric(
        family,
        "typed_metric",
        aggregation="sum",
        representation=metric.representation,
    )

    return Operator(
        model=model,
        spec=spec,
        call_inputs=_metric_vector_call_inputs(metric),
        default_settings=_sqrt_metric_default_settings(model, metric),
        metric=metric,
    )


def inverse_metric_vp(
    model: Model,
    metric: Metric,
    name: str | None = None,
    *,
    damping: Damping,
    tol: float | None = None,
) -> Operator:
    """Build a typed inverse metric-vector product.

    Returns:
        Inverse metric-vector operator.
    """
    tolerance = _typed_inverse_tol(metric, tol)
    damping_value = _damping_value(metric, damping)
    _require_matrix_free_positive_damping(metric, damping_value)
    family = _operator_family(name, "inverse_metric")
    spec = _operator_builders.inverse_metric(
        family,
        "typed_metric",
        aggregation="sum",
        representation=metric.representation,
        damping=damping_value,
        tol=tolerance,
    )
    spec = _operator_with_damping_identity(spec, metric, damping)

    return Operator(
        model=model,
        spec=spec,
        call_inputs=_metric_vector_call_inputs(metric),
        default_settings=_inverse_metric_default_settings(model, metric),
        metric=metric,
    )


def inverse_sqrt_metric_vp(
    model: Model,
    metric: Metric,
    name: str | None = None,
    *,
    damping: Damping,
    tol: float | None = None,
) -> Operator:
    """Build a typed inverse metric square-root product.

    Returns:
        Inverse metric square-root operator.
    """
    _reject_inverse_sqrt_tol(tol)
    damping_value = _damping_value(metric, damping)
    _require_matrix_free_positive_damping(metric, damping_value)
    family = _operator_family(name, "inverse_sqrt_metric")
    spec = _operator_builders.inverse_sqrt_metric(
        family,
        "typed_metric",
        aggregation="sum",
        representation=metric.representation,
        damping=damping_value,
    )
    spec = _operator_with_damping_identity(spec, metric, damping)

    return Operator(
        model=model,
        spec=spec,
        call_inputs=_metric_vector_call_inputs(metric),
        default_settings=_sqrt_metric_default_settings(model, metric),
        metric=metric,
    )


def metric_inner_vp(
    model: Model,
    metric: Metric,
    name: str | None = None,
    *,
    as_norm: bool = False,
) -> Operator:
    """Build a typed metric inner product.

    Returns:
        Metric inner-product operator.
    """
    family = _operator_family(name, "metric_inner")
    spec = _operator_builders.metric_inner(
        family,
        "typed_metric",
        aggregation="sum",
        representation=metric.representation,
        as_norm=as_norm,
    )

    return Operator(
        model=model,
        spec=spec,
        call_inputs=_metric_inner_call_inputs(metric),
        default_settings=_metric_inner_default_settings(model, metric, as_norm),
        metric=metric,
    )


def inverse_metric_inner_vp(
    model: Model,
    metric: Metric,
    name: str | None = None,
    *,
    damping: Damping,
    as_norm: bool = False,
    tol: float | None = None,
) -> Operator:
    """Build a typed inverse metric inner product.

    Returns:
        Inverse metric inner-product operator.
    """
    tolerance = _typed_inverse_tol(metric, tol)
    damping_value = _damping_value(metric, damping)
    _require_matrix_free_positive_damping(metric, damping_value)
    family = _operator_family(name, "inverse_metric_inner")
    spec = _operator_builders.inverse_metric_inner(
        family,
        "typed_metric",
        aggregation="sum",
        representation=metric.representation,
        damping=damping_value,
        as_norm=as_norm,
        tol=tolerance,
    )
    spec = _operator_with_damping_identity(spec, metric, damping)

    return Operator(
        model=model,
        spec=spec,
        call_inputs=_metric_inner_call_inputs(metric),
        default_settings=_inverse_metric_inner_default_settings(model, metric, as_norm),
        metric=metric,
    )


@dataclasses.dataclass(frozen=True, slots=True)
class Run:
    """Typed multi-product tuning result."""

    plan: Plan
    operators: Mapping[str, Operator]

    def __getitem__(self, name: str) -> Operator:
        """Return one tuned operator by name."""
        return self.operators[name]

    def __iter__(self) -> Iterator[str]:
        """Iterate product names.

        Returns:
            Product-name iterator.
        """
        return iter(self.operators)

    def __contains__(self, name: object) -> bool:
        """Return whether a product exists."""
        return name in self.operators

    def __len__(self) -> int:
        """Return the product count."""
        return len(self.operators)


@dataclasses.dataclass(frozen=True, slots=True)
class _PublicDataProvider:
    operator: Operator
    batches: tuple[Batch, ...]
    reference: Case | None
    probes: tuple[tuple[Batch, TensorTree], ...]

    def signature(self) -> Mapping[str, Any]:
        """Return stable public data identity fields."""
        return {
            "kind": "public_data",
            "batches": tuple(_tree_signature(batch) for batch in self.batches),
            "reference": None
            if self.reference is None or self.reference.batch is None
            else _tree_signature(self.reference.batch),
            "probe_batches": tuple(_tree_signature(batch) for batch, _ in self.probes),
            "operator": self.operator.spec.signature(),
        }

    def reference_batch(self, family: str, check_name: str) -> Batch:
        """Return the reference batch for the lower tuning engine.

        Raises:
            MaterializationError: If data for the requested family is unavailable.
        """
        _ = check_name
        self._require_family(family)

        if self.reference is not None and self.reference.batch is not None:
            return _prepared_operator_batch(self.operator, self.reference.batch)

        if self.batches:
            return _prepared_operator_batch(self.operator, self.batches[0])

        if _operator_requires_batch(self.operator):
            message = f"public tune data is required for {self.operator.spec.family}"
            raise MaterializationError(message)

        return _prepared_operator_batch(self.operator, {})

    def probe_batches(self, family: str) -> Sequence[Batch]:
        """Return probe batches for the lower tuning engine.

        Raises:
            MaterializationError: If probe data for the family is unavailable.
        """
        self._require_family(family)

        if self.probes:
            return tuple(
                _prepared_operator_batch(self.operator, batch)
                for batch, _ in self.probes
            )

        if self.batches:
            return tuple(
                _prepared_operator_batch(self.operator, batch) for batch in self.batches
            )

        if _operator_requires_batch(self.operator):
            message = f"public tune data is required for {self.operator.spec.family}"
            raise MaterializationError(message)

        return (_prepared_operator_batch(self.operator, {}),)

    def _require_family(self, family: str) -> None:
        if family != self.operator.spec.family:
            message = f"public data requested unknown family: {family}"
            raise MaterializationError(message)


@dataclasses.dataclass(frozen=True, slots=True)
class _PublicVectorProvider:
    operator: Operator
    vectors: tuple[TensorTree, ...]
    reference: Case | None
    probes: tuple[tuple[Batch, TensorTree], ...]

    def signature(self) -> Mapping[str, Any]:
        """Return stable public vector identity fields."""
        return {
            "kind": "public_vectors",
            "vectors": tuple(_tree_signature(vector) for vector in self.vectors),
            "reference": None
            if self.reference is None or self.reference.vector is None
            else _tree_signature(self.reference.vector),
            "probe_vectors": tuple(
                _tree_signature(vector) for _, vector in self.probes
            ),
            "operator": self.operator.spec.signature(),
        }

    def reference_vectors(self, family: str) -> TensorTree:
        """Return reference vectors for the lower tuning engine.

        Raises:
            MaterializationError: If vectors for the family are unavailable.
        """
        self._require_family(family)

        if self.reference is not None and self.reference.vector is not None:
            return self.reference.vector

        if self.vectors:
            return self.vectors[0]

        if _operator_requires_vector(self.operator):
            message = (
                f"public tune vectors are required for {self.operator.spec.family}"
            )
            raise MaterializationError(message)

        return {}

    def probe_vectors(self, family: str) -> Sequence[TensorTree]:
        """Return probe vectors for the lower tuning engine.

        Raises:
            MaterializationError: If probe vectors for the family are unavailable.
        """
        self._require_family(family)

        if self.probes:
            return tuple(vector for _, vector in self.probes)

        if self.vectors:
            return self.vectors

        if _operator_requires_vector(self.operator):
            message = (
                f"public tune vectors are required for {self.operator.spec.family}"
            )
            raise MaterializationError(message)

        return ({},)

    def _require_family(self, family: str) -> None:
        if family != self.operator.spec.family:
            message = f"public vectors requested unknown family: {family}"
            raise MaterializationError(message)


def _operator_family(name: str | None, default: str) -> str:
    if name is not None:
        return name

    return default


def _operator_problem(
    operator: Operator,
    *,
    data: Iterable[Batch] | None,
    vectors: Iterable[TensorTree] | None,
    target: Target,
    space: SearchSpace,
    search: SearchStrategy,
    reference: Case | None,
    probes: Sequence[tuple[Batch, TensorTree]] | None,
) -> _LowerProblem:
    if operator.spec.kind == "composition":
        return _composition_problem(
            operator,
            data=data,
            vectors=vectors,
            target=target,
            space=space,
            search=search,
            reference=reference,
            probes=probes,
        )

    probe_inputs = () if probes is None else tuple(probes)
    data_provider = _PublicDataProvider(
        operator=operator,
        batches=() if data is None else tuple(data),
        reference=reference,
        probes=probe_inputs,
    )
    vector_provider = _PublicVectorProvider(
        operator=operator,
        vectors=() if vectors is None else tuple(vectors),
        reference=reference,
        probes=probe_inputs,
    )
    axis_registry = _search_space_axis_registry(space)
    candidates = _public_candidate_rows(operator, space, axis_registry)
    runtime = standard_runtime_config(
        operator.spec,
        params=operator.model.parameter_values,
        buffers=operator.model.buffers,
        candidates=candidates,
        thresholds=_operator_thresholds(operator),
        objective_signature=_operator_objective_signature(operator),
        axis_registry=axis_registry,
        parameter_surface=operator.model.parameters,
        scalar_objectives=operator.scalar_objectives,
        function_objectives=operator.function_objectives,
        module=operator.model.module,
        module_call=operator.model.call,
    )

    return _LowerProblem(
        model=operator.model.module,
        params=operator.model.parameters,
        data=data_provider,
        operator=operator.spec,
        vectors=vector_provider,
        target=target.lower(search),
        runtime=runtime,
        anchor_policy={},
        replay_policy={},
        adapter_identity={
            "adapter_id": "vptune.public",
            "adapter_version": PACKAGE_VERSION,
        },
    )


def _composition_problem(
    operator: Operator,
    *,
    data: Iterable[Batch] | None,
    vectors: Iterable[TensorTree] | None,
    target: Target,
    space: SearchSpace,
    search: SearchStrategy,
    reference: Case | None,
    probes: Sequence[tuple[Batch, TensorTree]] | None,
) -> _LowerProblem:
    probe_inputs = () if probes is None else tuple(probes)
    data_provider = _PublicDataProvider(
        operator=operator,
        batches=() if data is None else tuple(data),
        reference=reference,
        probes=probe_inputs,
    )
    vector_provider = _PublicVectorProvider(
        operator=operator,
        vectors=() if vectors is None else tuple(vectors),
        reference=reference,
        probes=probe_inputs,
    )
    axis_registry = _search_space_axis_registry(space)
    candidates = _public_candidate_rows(operator, space, axis_registry)
    components = {
        child: _unbound_composition_child(child)
        for child in _composition_runtime_children(operator)
    }
    runtime = composition_runtime_config(
        operator.spec,
        components=components,
        anchor_components=components,
        candidates=candidates,
        thresholds=_operator_thresholds(operator),
        component_signature={"state": "unbound_public_children"},
        anchor_component_signature={"state": "unbound_public_children"},
        axis_registry=axis_registry,
    )

    return _LowerProblem(
        model=operator.model.module,
        params=operator.model.parameters,
        data=data_provider,
        operator=operator.spec,
        vectors=vector_provider,
        target=target.lower(search),
        runtime=runtime,
    )


def _public_candidate_rows(
    operator: Operator,
    space: SearchSpace,
    axis_registry: Any,
) -> tuple[Candidate, ...]:
    return tuple(
        Candidate(
            family=operator.spec.family,
            candidate_id=candidate_id,
            settings=dict(settings),
            changed_axes=_public_changed_axes(settings, axis_registry),
            generator_id="vptune.public.problem",
            generator_version=PACKAGE_VERSION,
        )
        for candidate_id, settings in space.candidate_settings(operator).items()
    )


def _public_changed_axes(
    settings: Mapping[str, Any],
    axis_registry: Any,
) -> tuple[str, ...]:
    axes = set()

    for key in settings:
        owner = axis_registry.owners.get(key)

        if owner is not None:
            axes.add(owner)
            continue

        optional_owners = axis_registry.optional_owners.get(key)

        if optional_owners is not None:
            axes.update(optional_owners)
            continue

        axes.add(key)

    return tuple(sorted(axes))


def _composition_runtime_children(operator: Operator) -> tuple[str, ...]:
    children = operator.spec.semantics.get("children")

    if isinstance(children, Sequence) and not isinstance(children, str):
        return tuple(str(child) for child in children)

    message = "composition operator must declare children"
    raise MaterializationError(message)


def _unbound_composition_child(
    child: str,
) -> Callable[[Batch, TensorTree], TensorTree]:
    def component(_: Batch, __: TensorTree) -> TensorTree:
        message = f"composition child row is not selected: {child}"
        raise MaterializationError(message)

    return component


def _load_operator_plan(
    operator: Operator,
    run_dir: Path,
    *,
    memory_backend: MemoryBackend | None,
) -> Plan:
    _ = memory_backend
    summary = read_record(run_dir / "summaries" / "tuning.json")
    input_signature = _single_operator_saved_input_signature(summary, operator)
    _require_saved_operator_matches_live(input_signature, operator)
    runtime = _operator_runtime_config(operator)
    replay_context = ReplayContext(
        input_signature=input_signature,
        family_input_signatures={operator.spec.family: input_signature},
        materializer_identities={
            operator.spec.family: dict(runtime.materializer.identity())
        },
        selection_policy=_selection_policy_from_saved(summary),
        target_identity=dict(summary["target_identity"]),
        runtime_identities={operator.spec.family: runtime.identity()},
        adapter_identities={
            operator.spec.family: {
                "adapter_id": "vptune.public",
                "adapter_version": PACKAGE_VERSION,
            }
        },
        validation_required=bool(summary["validation_required"]),
        validation_order=tuple(str(name) for name in summary["validation_order"]),
    )

    return _load_plan(
        run_dir,
        replay_context=replay_context,
        materializers={operator.spec.family: runtime.materializer},
    )


def _single_operator_saved_input_signature(
    summary: Mapping[str, Any],
    operator: Operator,
) -> Mapping[str, Any]:
    selected = dict(summary["selected"])

    if tuple(selected) != (operator.spec.family,):
        message = "Operator.load requires a single-product run for this operator"
        raise MaterializationError(message)

    input_signature = dict(summary["input_signature"])

    if "operator" not in input_signature:
        message = "Operator.load requires a single-product tuning summary"
        raise MaterializationError(message)

    return input_signature


def _require_saved_operator_matches_live(
    input_signature: Mapping[str, Any],
    operator: Operator,
) -> None:
    expected = {
        "model": module_identity(operator.model.module),
        "params": operator.model.parameters.signature(),
        "operator": operator.spec.signature(),
    }

    for key, value in expected.items():
        if to_json_value(input_signature.get(key)) != to_json_value(value):
            message = f"Operator.load saved {key} identity differs"
            raise MaterializationError(message)


def _operator_runtime_config(operator: Operator) -> Any:
    return standard_runtime_config(
        operator.spec,
        params=operator.model.parameter_values,
        buffers=operator.model.buffers,
        candidates=(),
        thresholds=_operator_thresholds(operator),
        objective_signature=_operator_objective_signature(operator),
        axis_registry=standard_axis_registry(),
        parameter_surface=operator.model.parameters,
        scalar_objectives=operator.scalar_objectives,
        function_objectives=operator.function_objectives,
        module=operator.model.module,
        module_call=operator.model.call,
    )


def _operator_selected_plan(operator: Operator) -> Plan:
    if operator.plan is None:
        message = f"operator has no selected plan: {operator.spec.family}"
        raise MaterializationError(message)

    return operator.plan


def _operator_thresholds(operator: Operator) -> Mapping[str, float]:
    if operator.spec.thresholds:
        return dict(operator.spec.thresholds)

    thresholds = {
        "max_abs_diff": STANDARD_THRESHOLDS["max_abs_diff"],
        "max_rel_diff": STANDARD_THRESHOLDS["max_rel_diff"],
    }

    if operator.spec.kind in {"gradient", "jvp", "hvp"}:
        thresholds.update({
            "directional_abs_diff": STANDARD_THRESHOLDS["directional_abs_diff"],
            "directional_rel_diff": STANDARD_THRESHOLDS["directional_rel_diff"],
        })

    if operator.spec.kind == "vjp":
        thresholds["inner_abs_diff"] = STANDARD_THRESHOLDS["inner_abs_diff"]

    if operator.spec.kind == "ggnvp":
        thresholds["inner_abs_diff"] = STANDARD_THRESHOLDS["inner_abs_diff"]

    metric_representation = operator.spec.semantics.get("representation", {})
    matrix_free_metric = (
        operator.spec.kind == "metric"
        and metric_representation.get("kind") == "matrix_free"
    )

    if operator.spec.kind == "ggnvp" or (
        operator.spec.kind == "metric" and not matrix_free_metric
    ):
        thresholds["symmetry_max_abs_diff"] = STANDARD_THRESHOLDS[
            "symmetry_max_abs_diff"
        ]
        thresholds["psd_violation"] = STANDARD_THRESHOLDS["psd_violation"]

    if operator.spec.kind in {"inverse_metric", "inverse_metric_inner"}:
        thresholds["inverse_residual"] = _operator_inverse_residual_threshold(operator)
        damping = operator.spec.semantics.get("damping")

        if isinstance(damping, float | int) and damping > 0.0:
            thresholds["damping_min"] = float(damping)

    return thresholds


def _operator_objective_signature(operator: Operator) -> Mapping[str, Any]:
    return {
        "kind": "typed_public_operator",
        "operator": operator.spec.signature(),
        "model": operator.model.signature(),
        "scalar_objectives": {
            key: _callable_signature(value)
            for key, value in sorted(operator.scalar_objectives.items())
        },
        "function_objectives": {
            key: _callable_signature(value)
            for key, value in sorted(operator.function_objectives.items())
        },
    }


def _callable_signature(value: Any) -> Any:
    identity = getattr(value, "identity", None)

    if callable(identity):
        return identity()

    signature = getattr(value, "signature", None)

    if callable(signature):
        return signature()

    message = f"typed callable lacks identity: {type(value).__name__}"
    raise MaterializationError(message)


def _qualified_callable_name(value: Callable[..., Any]) -> str:
    module = getattr(value, "__module__", None)
    qualname = getattr(value, "__qualname__", None)

    if isinstance(module, str) and module and isinstance(qualname, str) and qualname:
        return f"{module}.{qualname}"

    message = "typed callable must expose module and qualname"
    raise MaterializationError(message)


def _selection_policy_from_saved(summary: Mapping[str, Any]) -> SelectionPolicy:
    policy = dict(summary["policy"])
    expected = {field.name for field in dataclasses.fields(SelectionPolicy)}

    if set(policy) != expected:
        message = "saved selection policy fields differ"
        raise MaterializationError(message)

    return SelectionPolicy(**policy)


def _operator_requires_batch(operator: Operator) -> bool:
    return "batch" in operator.call_inputs or operator.bound_batch is not None


def _operator_requires_vector(operator: Operator) -> bool:
    return operator.spec.kind == "gradient" or any(
        name in {"vector", "left", "right"} for name in operator.call_inputs
    )


def _prepared_operator_batch(operator: Operator, batch: Batch) -> Batch:
    result = dict(batch)

    if operator.bound_batch is not None:
        result = {**operator.bound_batch, **result}

    if operator.metric is not None:
        result = {**operator.metric.batch, **result}

    if operator.batch_transform is not None:
        result = operator.batch_transform(result)

    return result


def _search_strategy(
    strategy: str,
    *,
    retained_top_count: int | None = None,
    compile_call_horizons: tuple[int, ...] = (),
    variance_repeat_count: int | None = None,
) -> SearchStrategy:
    try:
        policy = _SearchPolicy(
            strategy=strategy,
            retained_top_count=retained_top_count,
            compile_call_horizons=compile_call_horizons,
            variance_repeat_count=variance_repeat_count,
        )
    except RuntimeError as error:
        raise MaterializationError(str(error)) from error

    return SearchStrategy(policy)


def _int_tuple(values: Sequence[int], name: str) -> tuple[int, ...]:
    if isinstance(values, str | bytes | bytearray):
        message = f"{name} must be a sequence of integers"
        raise MaterializationError(message)

    result = tuple(values)

    for value in result:
        if not isinstance(value, int) or isinstance(value, bool):
            message = f"{name} entries must be integers"
            raise MaterializationError(message)

    return result


def _cuda_device_name(device: int | str) -> str:
    if isinstance(device, bool):
        message = "cuda device must be an integer index or cuda device string"
        raise MaterializationError(message)

    if isinstance(device, int):
        if device < 0:
            message = "cuda device index must be nonnegative"
            raise MaterializationError(message)

        return f"cuda:{device}"

    if isinstance(device, str):
        if device == "cuda" or device.startswith("cuda:"):
            return device

        message = f"cuda device string is unsupported: {device}"
        raise MaterializationError(message)

    message = "cuda device must be an integer index or cuda device string"
    raise MaterializationError(message)


def _determinism_policy(
    policy: DeterminismPolicy | Mapping[str, Any] | None,
) -> DeterminismPolicy:
    if policy is None:
        return DeterminismPolicy()

    if isinstance(policy, DeterminismPolicy):
        return policy

    if isinstance(policy, Mapping):
        return DeterminismPolicy(dict(policy))

    message = "determinism policy must be a mapping"
    raise MaterializationError(message)


def _environment_policy(
    policy: EnvironmentPolicy | Mapping[str, Any] | None,
) -> EnvironmentPolicy:
    if policy is None:
        return EnvironmentPolicy()

    if isinstance(policy, EnvironmentPolicy):
        return policy

    if isinstance(policy, Mapping):
        return EnvironmentPolicy(dict(policy))

    message = "environment policy must be a mapping"
    raise MaterializationError(message)


def _target_dtype_values() -> tuple[str, ...]:
    return _target_axis_values((
        "dtype.parameter_storage",
        "dtype.model_compute",
        "dtype.vector",
        "dtype.intermediate",
        "dtype.output",
        "dtype.metric_factor",
    ))


def _target_axis_values(axis_keys: Sequence[str]) -> tuple[str, ...]:
    axes = axis_manifest().by_key()
    values = ()

    for axis_key in axis_keys:
        axis = axes.get(axis_key)

        if axis is None:
            message = f"target axis is not registered: {axis_key}"
            raise MaterializationError(message)

        values = (*values, *_string_axis_values(axis_key, axis.value_domain))

    return _unique_strings(values)


def _string_axis_values(axis_key: str, values: Sequence[Any]) -> tuple[str, ...]:
    strings = ()

    for value in values:
        if not isinstance(value, str):
            message = f"target axis has a non-string value: {axis_key}"
            raise MaterializationError(message)

        strings = (*strings, value)

    return strings


def _unique_strings(values: Sequence[str]) -> tuple[str, ...]:
    unique = []

    for value in values:
        if value not in unique:
            unique.append(value)

    return tuple(unique)


def _runtime_call_inputs(
    names: tuple[str, ...],
    values: tuple[Any, ...],
) -> tuple[Batch, TensorTree]:
    if names == ("batch",):
        return values[0], {}

    if names == ("vector",):
        return {}, values[0]

    if names == ("left", "right"):
        return {}, (values[0], values[1])

    if names == ("batch", "vector"):
        return values[0], values[1]

    if names == ("batch", "left", "right"):
        return values[0], (values[1], values[2])

    message = f"typed call input shape is unsupported: {names}"
    raise MaterializationError(message)


def _metric_vector_call_inputs(metric: Metric) -> tuple[str, ...]:
    if metric.kind == "matrix_free":
        return ("batch", "vector")

    return ("vector",)


def _metric_inner_call_inputs(metric: Metric) -> tuple[str, ...]:
    if metric.kind == "matrix_free":
        return ("batch", "left", "right")

    return ("left", "right")


def _metric_default_settings(metric: Metric) -> dict[str, Any]:
    if metric.kind == "dense_matrix":
        return {"metric.multiply_path": "dense_matmul"}

    if metric.kind == "matrix_free":
        return {
            "metric.multiply_path": "streaming_multiply",
            "metric.accumulation": "streaming",
        }

    if metric.kind == "block_diagonal":
        return {
            "metric.multiply_path": "streaming_multiply",
            "metric.accumulation": "streaming",
        }

    return {
        "metric.multiply_path": "factorized_multiply",
        "metric.accumulation": "materialized_blocks",
    }


def _inverse_metric_default_settings(model: Model, metric: Metric) -> dict[str, Any]:
    if metric.kind == "dense_matrix":
        return {"inverse_metric.solve_path": "dense_solve"}

    if metric.kind == "matrix_free":
        return {
            "inverse_metric.solve_path": "conjugate_gradient",
            "inverse_metric.iteration_budget": _active_parameter_count(model),
            "inverse_metric.preconditioner": "none",
            **_metric_default_settings(metric),
        }

    if metric.kind == "block_diagonal":
        return {"inverse_metric.solve_path": "blockwise_solve"}

    if metric.kind == "low_rank_factors":
        return {"inverse_metric.solve_path": "woodbury_low_rank_solve"}

    return {"inverse_metric.solve_path": "factorized_solve"}


def _sqrt_metric_default_settings(model: Model, metric: Metric) -> dict[str, Any]:
    if metric.kind == "matrix_free":
        return {
            "sqrt_metric.factor_path": "matrix_free_lanczos",
            "sqrt_metric.lanczos_iterations": _active_parameter_count(model),
            **_metric_default_settings(metric),
        }

    if metric.kind in {"dense_matrix", "block_diagonal"}:
        return {"sqrt_metric.factor_path": "cholesky_factor"}

    return {"sqrt_metric.factor_path": "closed_form_factor_square_root"}


def _metric_inner_default_settings(
    model: Model,
    metric: Metric,
    as_norm: bool,
) -> dict[str, Any]:
    if as_norm:
        return {
            "metric_inner.reduction_path": "sqrt_apply_reduce",
            "metric_inner.multi_rhs": "single_column",
            **_sqrt_metric_default_settings(model, metric),
        }

    if metric.kind == "matrix_free":
        return {
            "metric_inner.reduction_path": "multiply_then_reduce",
            "metric_inner.multi_rhs": "single_column",
            **_metric_default_settings(metric),
        }

    if metric.kind == "dense_matrix":
        return {
            "metric_inner.reduction_path": "multiply_then_reduce",
            "metric_inner.multi_rhs": "single_column",
            "metric.multiply_path": "dense_matmul",
        }

    if metric.kind == "block_diagonal":
        return {
            "metric_inner.reduction_path": "multiply_then_reduce",
            "metric_inner.multi_rhs": "single_column",
            "metric.multiply_path": "blockwise_multiply",
            "metric.accumulation": "materialized_blocks",
        }

    return {
        "metric_inner.reduction_path": "factored_gram",
        "metric_inner.multi_rhs": "single_column",
    }


def _inverse_metric_inner_default_settings(
    model: Model,
    metric: Metric,
    as_norm: bool,
) -> dict[str, Any]:
    if as_norm:
        return {
            "inverse_metric_inner.reduction_path": "sqrt_apply_reduce",
            "inverse_metric_inner.multi_rhs": "single_column",
            **_sqrt_metric_default_settings(model, metric),
        }

    if metric.kind == "matrix_free":
        return {
            "inverse_metric_inner.reduction_path": "solve_then_reduce",
            "inverse_metric_inner.multi_rhs": "single_column",
            **_inverse_metric_default_settings(model, metric),
        }

    if metric.kind == "block_diagonal":
        return {
            "inverse_metric_inner.reduction_path": "solve_then_reduce",
            "inverse_metric_inner.multi_rhs": "single_column",
            "inverse_metric.solve_path": "blockwise_solve",
        }

    return {
        "inverse_metric_inner.reduction_path": "solve_then_reduce",
        "inverse_metric_inner.multi_rhs": "single_column",
        **_inverse_metric_default_settings(model, metric),
    }


def _active_parameter_count(model: Model) -> int:
    count = sum(tensor.numel() for tensor in model.parameter_values.values())

    if count < 1:
        message = "matrix_free metric requires at least one active parameter"
        raise MaterializationError(message)

    return count


def _require_matrix_free_positive_damping(metric: Metric, damping: float) -> None:
    if metric.kind != "matrix_free":
        return

    if damping > 0.0:
        return

    message = "matrix_free inverse metric requires positive damping"
    raise MaterializationError(message)


def _damping_value(metric: Metric, damping: Damping) -> float:
    if damping.kind == "scalar":
        return _float_damping(damping.value, "scalar damping")

    if damping.kind == "per_group":
        _per_group_damping_values(metric, damping)

        return 0.0

    if damping.kind == "eigenvalue_floor":
        if metric.kind != "ekfac_factors":
            message = "eigenvalue_floor damping requires an EKFAC metric"
            raise MaterializationError(message)

        return _float_damping(damping.value, "eigenvalue_floor damping")

    if damping.kind == "kfac_pi":
        if metric.kind != "kfac_factors":
            message = "kfac_pi damping requires a KFAC metric"
            raise MaterializationError(message)

        return _float_damping(damping.value, "kfac_pi damping")

    message = f"damping kind is not lowered: {damping.kind}"
    raise MaterializationError(message)


def _per_group_damping_values(
    metric: Metric,
    damping: Damping,
) -> dict[str, float]:
    if metric.kind not in {"block_diagonal", "kfac_factors"}:
        message = "per_group damping requires a block-diagonal or KFAC metric"
        raise MaterializationError(message)

    values = _per_group_damping_mapping(damping.value)

    expected = _per_group_metric_names(metric)
    actual = tuple(values)

    if set(actual) != set(expected):
        message = "per_group damping keys must match metric block names"
        raise MaterializationError(message)

    result = {}

    for key in expected:
        result[key] = _float_damping(values[key], "per_group damping")

    return result


def _per_group_damping_mapping(
    value: float | Mapping[str, float],
) -> dict[str, float]:
    if not isinstance(value, Mapping):
        message = "per_group damping values must be a mapping"
        raise MaterializationError(message)

    result = {}

    for key, damping in value.items():
        if not isinstance(key, str):
            message = "per_group damping keys must be strings"
            raise MaterializationError(message)

        if (
            not isinstance(damping, float | int)
            or isinstance(damping, bool)
            or damping < 0.0
        ):
            message = "per_group damping must be nonnegative"
            raise MaterializationError(message)

        result[key] = float(damping)

    return result


def _per_group_metric_names(metric: Metric) -> tuple[str, ...]:
    if metric.kind == "block_diagonal":
        value = metric.representation.get("block_names")

        if (
            not isinstance(value, tuple)
            or not value
            or any(not isinstance(item, str) for item in value)
        ):
            message = (
                "block-diagonal metric requires named blocks for per_group damping"
            )
            raise MaterializationError(message)

        return value

    blocks = metric.representation.get("blocks")

    if not isinstance(blocks, tuple) or not blocks:
        message = "KFAC metric requires named blocks for per_group damping"
        raise MaterializationError(message)

    names = []

    for block in blocks:
        if not isinstance(block, Mapping):
            message = "KFAC block descriptor must be a mapping"
            raise MaterializationError(message)

        name = block.get("parameter")

        if not isinstance(name, str):
            message = "KFAC block parameter must be a string"
            raise MaterializationError(message)

        names.append(name)

    return tuple(names)


def _operator_with_damping_identity(
    spec: OperatorSpec,
    metric: Metric,
    damping: Damping,
) -> OperatorSpec:
    if damping.kind == "per_group":
        damping_value = _per_group_damping_values(metric, damping)
    else:
        damping_value = damping.value

    semantics = {
        **dict(spec.semantics),
        "damping_kind": damping.kind,
        "damping_value": damping_value,
        "damping": damping_value,
    }

    if damping.policy is not None:
        semantics["damping_policy"] = damping.policy

    return dataclasses.replace(spec, semantics=semantics)


def _operator_with_sample_source_identity(
    spec: OperatorSpec,
    typed_likelihood: Likelihood,
    sample_source: SampleSource,
) -> OperatorSpec:
    semantics = {
        **dict(spec.semantics),
        "likelihood": typed_likelihood.signature(),
        "sample_source_identity": sample_source.signature(),
    }

    return dataclasses.replace(spec, semantics=semantics)


def _operator_inverse_residual_threshold(operator: Operator) -> float:
    value = operator.spec.semantics.get("tol")

    if isinstance(value, float):
        return value

    return STANDARD_THRESHOLDS["inverse_residual"]


def _typed_inverse_tol(metric: Metric, tol: float | None) -> float | None:
    if tol is None:
        return None

    if not math.isfinite(tol) or tol <= 0.0:
        message = "typed inverse tol must be positive and finite"
        raise MaterializationError(message)

    if metric.kind != "matrix_free":
        message = "typed inverse tol requires a matrix_free conjugate_gradient row"
        raise MaterializationError(message)

    return tol


def _reject_inverse_sqrt_tol(tol: float | None) -> None:
    if tol is None:
        return

    message = "inverse_sqrt_metric tol has no lowered residual rule"
    raise MaterializationError(message)


def _require_nonempty_string(value: Any, name: str) -> None:
    if isinstance(value, str) and value:
        return

    message = f"{name} must be a nonempty string"
    raise MaterializationError(message)


def _require_loss_reduction(reduction: str) -> None:
    if reduction in {"sum", "mean", "token_mean"}:
        return

    message = f"loss reduction is unsupported: {reduction}"
    raise MaterializationError(message)


def _require_token_mean_denominator(reduction: str, denominator: str) -> None:
    if denominator == "num_tokens":
        return

    message = f"{reduction} denominator is unsupported: {denominator}"
    raise MaterializationError(message)


def _require_mse_denominator(reduction: str, denominator: str) -> None:
    if reduction == "token_mean":
        if denominator == "num_tokens":
            return

        message = f"token_mean denominator is unsupported: {denominator}"
        raise MaterializationError(message)

    if denominator == "num_elements":
        return

    message = f"{reduction} denominator is unsupported: {denominator}"
    raise MaterializationError(message)


def _require_same_shape(
    left: torch.Tensor,
    right: torch.Tensor,
    name: str,
) -> None:
    if tuple(left.shape) == tuple(right.shape):
        return

    message = f"{name} shape must match"
    raise MaterializationError(message)


def _require_finite_public_tensor(tensor: torch.Tensor, name: str) -> None:
    if bool(torch.isfinite(tensor).all()):
        return

    message = f"{name} must be finite"
    raise MaterializationError(message)


def _float_damping(value: float | Mapping[str, float], name: str) -> float:
    if not isinstance(value, float | int) or isinstance(value, bool):
        message = f"{name} must be a number"
        raise MaterializationError(message)

    _require_nonnegative_float(value, name)

    return float(value)


def _require_nonnegative_float(value: float, name: str) -> None:
    if not isinstance(value, float | int) or isinstance(value, bool) or value < 0.0:
        message = f"{name} must be nonnegative"
        raise MaterializationError(message)


def _require_positive_float(value: float, name: str) -> None:
    if not isinstance(value, float | int) or isinstance(value, bool) or value <= 0.0:
        message = f"{name} must be positive"
        raise MaterializationError(message)


def _require_tensor(value: Any, name: str) -> None:
    if isinstance(value, torch.Tensor):
        return

    message = f"{name} must be a tensor"
    raise MaterializationError(message)


def _require_square_tensor(value: torch.Tensor, name: str) -> None:
    if value.ndim == SQUARE_MATRIX_DIMS and value.shape[0] == value.shape[1]:
        return

    message = f"{name} must be a square matrix"
    raise MaterializationError(message)


def _kfac_factor_pair(
    value: Mapping[str, torch.Tensor],
    parameter_name: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not isinstance(value, Mapping):
        message = f"metric.kfac factors for {parameter_name} must be a mapping"
        raise MaterializationError(message)

    left = value.get("left_factor")
    right = value.get("right_factor")

    if not isinstance(left, torch.Tensor):
        message = f"metric.kfac {parameter_name} left_factor must be a tensor"
        raise MaterializationError(message)

    if not isinstance(right, torch.Tensor):
        message = f"metric.kfac {parameter_name} right_factor must be a tensor"
        raise MaterializationError(message)

    _require_square_tensor(left, f"metric.kfac {parameter_name} left_factor")
    _require_square_tensor(right, f"metric.kfac {parameter_name} right_factor")

    return left, right


def _require_same_keys(
    left: Mapping[str, torch.Tensor],
    right: Mapping[str, torch.Tensor],
    name: str,
) -> None:
    if set(left) == set(right):
        return

    message = f"{name} must match"
    raise MaterializationError(message)


def _tree_signature(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return tensor_signature(value)

    if isinstance(value, Mapping):
        return {
            str(key): _tree_signature(nested)
            for key, nested in sorted(value.items(), key=lambda item: str(item[0]))
        }

    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return tuple(_tree_signature(item) for item in value)

    return value


__all__ = [
    "Case",
    "Combine",
    "Compose",
    "Damping",
    "DeterminismPolicy",
    "EnvironmentPolicy",
    "Likelihood",
    "LinearCombination",
    "Loss",
    "Metric",
    "Model",
    "Operator",
    "Output",
    "Run",
    "SampleSource",
    "ScaledIdentity",
    "SearchSpace",
    "SearchStrategy",
    "SelectionPolicy",
    "Source",
    "Target",
    "TimingPolicy",
    "autotune",
    "case",
    "compose",
    "composition",
    "cuda",
    "damping",
    "empirical_fisher_vp",
    "fisher_vp",
    "ggnvp",
    "gradient",
    "hvp",
    "inverse_metric_inner_vp",
    "inverse_metric_vp",
    "inverse_sqrt_metric_vp",
    "jvp",
    "likelihood",
    "linear_combination",
    "loss",
    "metric",
    "metric_inner_vp",
    "metric_vp",
    "module_call",
    "output",
    "parameters",
    "per_example_gradient",
    "problem",
    "sampled_fisher_vp",
    "samples",
    "scaled_identity",
    "search",
    "source",
    "space",
    "sqrt_metric_vp",
    "torch_model",
    "tune",
    "vjp",
]
