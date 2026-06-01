"""Candidate axes, admission, and DAG helpers."""

import dataclasses
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from vptune.admission import (
    CHECKPOINT_FIELDS,
    FUNCTIONAL_CALL_FIELDS,
    TORCH_FUNC_FIELDS,
    admit_checkpoint,
    admit_functional_call,
    admit_torch_func,
)
from vptune.data import Candidate, Family
from vptune.errors import AdmissionError

AdmissionRule = Callable[[Candidate], tuple[bool, str | None]]

DTYPE_VALUES = ("bfloat16", "float16", "float32")
ATTENTION_IMPL_VALUES = (
    "eager",
    "sdpa_math",
    "sdpa_flash",
    "sdpa_memory_efficient",
    "transformers_eager",
    "transformers_sdpa",
    "transformers_flash_attention_2",
    "patched_eager",
)
OPERATOR_PATH_VALUES = (
    "autograd_grad",
    "torch_func_jvp",
    "torch_func_vjp",
    "reverse_over_reverse",
    "jvp_grad",
    "dense_ggn",
    "jvp_hessian_vjp",
    "dense_score_outer",
    "categorical_exact",
    "score_gradient_loop",
    "dense_empirical_fisher",
    "per_example_gradient_loop",
    "per_example_gradient_vmap",
    "dense_metric",
    "dense_inverse_metric",
    "vhp",
    "sequential_composition",
)
TORCH_FUNC_OPERATOR_PATHS = (
    "torch_func_jvp",
    "torch_func_vjp",
    "jvp_grad",
    "per_example_gradient_vmap",
)
FORWARD_AD_OPERATOR_PATHS = ("torch_func_jvp", "jvp_grad")
VMAP_OPERATOR_PATHS = ("per_example_gradient_vmap",)
PARAMETER_LAYOUT_VALUES = ("flat_cpu", "flat_cuda", "tensor_tree", "dtensor")
SHARDING_VALUES = (
    "single_device",
    "fsdp2",
    "tensor_parallel",
    "sequence_parallel",
    "context_parallel",
)
METRIC_RESIDENCY_VALUES = ("device", "cpu", "mmap_cpu", "staged_device")
CHECKPOINT_VALUES = ("disabled", "non_reentrant")
CHECKPOINT_POLICY_VALUES = (
    "disabled",
    "non_reentrant_deterministic",
    "non_reentrant_no_rng_preservation",
)
ACTIVE_CHECKPOINT_VALUES = ("non_reentrant",)
ACTIVE_CHECKPOINT_POLICY_VALUES = (
    "non_reentrant_deterministic",
    "non_reentrant_no_rng_preservation",
)
MATMUL_PRECISION_VALUES = ("highest", "high", "medium")


@dataclasses.dataclass(frozen=True, slots=True)
class AxisDescriptor:
    """One tunable axis registered by core or an adapter."""

    name: str
    settings_keys: tuple[str, ...]
    allowed_values: tuple[Any, ...]
    adapter_id: str = "core"
    adapter_version: str = "0.0.1"
    admission_rule: AdmissionRule | None = None
    identity: Mapping[str, Any] = dataclasses.field(default_factory=dict)

    def admit(self, candidate: Candidate) -> tuple[bool, str | None]:
        """Run the admission rule for a candidate.

        Returns:
            Admission status and optional failure reason.
        """
        value_error = _axis_value_error(self, candidate)

        if value_error is not None:
            return False, value_error

        if self.admission_rule is None:
            return True, None

        return self.admission_rule(candidate)

    def signature(self) -> dict[str, Any]:
        """Return stable axis identity."""
        return {
            "name": self.name,
            "settings_keys": self.settings_keys,
            "allowed_values": self.allowed_values,
            "adapter_id": self.adapter_id,
            "adapter_version": self.adapter_version,
            "has_admission_rule": self.admission_rule is not None,
            "identity": dict(self.identity),
        }


@dataclasses.dataclass(slots=True)
class AxisRegistry:
    """Registry that prevents overlapping setting ownership."""

    axes: dict[str, AxisDescriptor] = dataclasses.field(default_factory=dict)
    owners: dict[str, str] = dataclasses.field(default_factory=dict)

    def register(self, axis: AxisDescriptor) -> None:
        """Register an axis descriptor.

        Raises:
            AdmissionError: If the axis has no validator, or if the axis or any
                owned setting key is duplicated.
        """
        if not axis.allowed_values and axis.admission_rule is None:
            message = (
                f"axis must declare allowed values or an admission rule: {axis.name}"
            )
            raise AdmissionError(message)

        if axis.name in self.axes:
            message = f"axis is already registered: {axis.name}"
            raise AdmissionError(message)

        for key in axis.settings_keys:
            owner = self.owners.get(key)

            if owner is not None:
                message = f"setting key has multiple axis owners: {key}"
                raise AdmissionError(message)

        self.axes[axis.name] = axis

        for key in axis.settings_keys:
            self.owners[key] = axis.name

    def admit(self, candidate: Candidate) -> Candidate:
        """Return candidate with admission status set.

        Returns:
            Candidate with updated admission status.

        Raises:
            AdmissionError: If a changed axis is unknown.
        """
        axis_names = set(candidate.changed_axes)

        for key in candidate.settings:
            owner = self.owners.get(key)

            if owner is None:
                message = f"candidate setting key has no axis owner: {key}"
                raise AdmissionError(message)

            axis_names.add(owner)

        for axis_name in sorted(axis_names):
            axis = self.axes.get(axis_name)

            if axis is None:
                message = f"candidate axis is unknown: {axis_name}"
                raise AdmissionError(message)

            passed, reason = axis.admit(candidate)

            if not passed:
                return dataclasses.replace(
                    candidate,
                    admission_status="failed",
                    admission_error=reason or f"axis rejected: {axis_name}",
                )

        return dataclasses.replace(candidate, admission_status="passed")

    def signature(self) -> dict[str, Any]:
        """Return stable registry identity."""
        return {
            "axes": {
                name: axis.signature() for name, axis in sorted(self.axes.items())
            },
            "owners": dict(sorted(self.owners.items())),
        }


def topological_families(families: Sequence[Family]) -> tuple[Family, ...]:
    """Return families in dependency order.

    Raises:
        RuntimeError: If names are duplicated, missing, or cyclic.
    """
    by_name = {family.name: family for family in families}

    if len(by_name) != len(families):
        message = "family names must be unique"
        raise RuntimeError(message)

    ordered = []
    visiting = set()
    visited = set()

    def visit(name: str) -> None:
        if name in visited:
            return

        if name in visiting:
            message = f"family DAG has a cycle at {name}"
            raise RuntimeError(message)

        family = by_name.get(name)

        if family is None:
            message = f"family dependency is missing: {name}"
            raise RuntimeError(message)

        visiting.add(name)

        for dependency in family.dependencies:
            visit(dependency)

        visiting.remove(name)
        visited.add(name)
        ordered.append(family)

    for family in families:
        visit(family.name)

    return tuple(ordered)


def _axis_value_error(axis: AxisDescriptor, candidate: Candidate) -> str | None:
    missing = tuple(key for key in axis.settings_keys if key not in candidate.settings)

    if missing:
        return f"candidate missing axis setting: {axis.name}"

    if len(axis.settings_keys) == 1:
        value = candidate.settings[axis.settings_keys[0]]
    else:
        value = {key: candidate.settings[key] for key in axis.settings_keys}

    if not axis.allowed_values or any(
        value == allowed for allowed in axis.allowed_values
    ):
        return None

    return f"candidate axis value is not allowed: {axis.name}"


def _positive_int_axis(*keys: str) -> AdmissionRule:
    def admit(candidate: Candidate) -> tuple[bool, str | None]:
        for key in keys:
            value = candidate.settings[key]

            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                return False, f"candidate axis must be a positive integer: {key}"

        return True, None

    return admit


def _bool_axis(*keys: str) -> AdmissionRule:
    def admit(candidate: Candidate) -> tuple[bool, str | None]:
        for key in keys:
            if not isinstance(candidate.settings[key], bool):
                return False, f"candidate axis must be boolean: {key}"

        return True, None

    return admit


def _operator_path_axis() -> AdmissionRule:
    def admit(candidate: Candidate) -> tuple[bool, str | None]:
        value = candidate.settings["operator_path"]

        if value not in TORCH_FUNC_OPERATOR_PATHS:
            return True, None

        missing = tuple(
            field for field in TORCH_FUNC_FIELDS if field not in candidate.settings
        )

        if missing:
            return False, f"torch.func admission fields missing: {missing}"

        if (
            value in FORWARD_AD_OPERATOR_PATHS
            and candidate.settings["requires_forward_ad"] is not True
        ):
            return False, f"operator path requires forward AD: {value}"

        if value in VMAP_OPERATOR_PATHS:
            vmap_error = _vmap_path_error(candidate.settings)

            if vmap_error is not None:
                return False, vmap_error

        try:
            admit_torch_func(candidate.settings)
        except AdmissionError as error:
            return False, str(error)

        return True, None

    return admit


def _vmap_path_error(settings: Mapping[str, Any]) -> str | None:
    if settings["requires_forward_ad"] is True:
        return "per_example_gradient_vmap does not use forward AD"

    chunk_error = _vmap_chunk_size_error(settings)

    if chunk_error is not None:
        return chunk_error

    return _vmap_batch_in_dims_error(settings)


def _vmap_chunk_size_error(settings: Mapping[str, Any]) -> str | None:
    if "vmap_chunk_size" not in settings:
        return "per_example_gradient_vmap requires vmap_chunk_size"

    chunk_size = settings["vmap_chunk_size"]

    if (
        not isinstance(chunk_size, int)
        or isinstance(chunk_size, bool)
        or chunk_size < 1
    ):
        return "vmap_chunk_size must be a positive integer"

    return None


def _vmap_batch_in_dims_error(settings: Mapping[str, Any]) -> str | None:
    if "vmap_batch_in_dims" not in settings:
        return "per_example_gradient_vmap requires vmap_batch_in_dims"

    return _vmap_batch_in_dims_value_error(settings["vmap_batch_in_dims"])


def _vmap_batch_in_dims_value_error(in_dims: Any) -> str | None:
    if not isinstance(in_dims, Mapping) or not in_dims:
        return "vmap_batch_in_dims must be a nonempty mapping"

    for key, value in in_dims.items():
        if not isinstance(key, str) or not key:
            return "vmap_batch_in_dims keys must be nonempty strings"

        if value is not None and (
            not isinstance(value, int) or isinstance(value, bool)
        ):
            return "vmap_batch_in_dims values must be integers or None"

    return None


def _vmap_batch_in_dims_axis() -> AdmissionRule:
    def admit(candidate: Candidate) -> tuple[bool, str | None]:
        error = _vmap_batch_in_dims_value_error(
            candidate.settings["vmap_batch_in_dims"]
        )

        return (True, None) if error is None else (False, error)

    return admit


def _torch_func_axis() -> AdmissionRule:
    def admit(candidate: Candidate) -> tuple[bool, str | None]:
        try:
            admit_torch_func(candidate.settings)
        except AdmissionError as error:
            return False, str(error)

        return True, None

    return admit


def _functional_call_axis() -> AdmissionRule:
    def admit(candidate: Candidate) -> tuple[bool, str | None]:
        try:
            admit_functional_call(candidate.settings)
        except AdmissionError as error:
            return False, str(error)

        return True, None

    return admit


def _checkpoint_axis(
    *, setting_key: str, active_values: tuple[str, ...]
) -> AdmissionRule:
    def admit(candidate: Candidate) -> tuple[bool, str | None]:
        value = candidate.settings[setting_key]

        if value not in active_values:
            return True, None

        return _admit_checkpoint_fields(candidate)

    return admit


def _checkpoint_fields_axis() -> AdmissionRule:
    def admit(candidate: Candidate) -> tuple[bool, str | None]:
        return _admit_checkpoint_fields(candidate)

    return admit


def _admit_checkpoint_fields(candidate: Candidate) -> tuple[bool, str | None]:
    missing = tuple(
        field for field in CHECKPOINT_FIELDS if field not in candidate.settings
    )

    if missing:
        return False, f"checkpoint admission fields missing: {missing}"

    try:
        admit_checkpoint(candidate.settings)
    except AdmissionError as error:
        return False, str(error)

    return True, None


def standard_axis_descriptors() -> tuple[AxisDescriptor, ...]:
    """Return standard core axis descriptors."""
    return (
        AxisDescriptor("model_dtype", ("model_dtype",), DTYPE_VALUES),
        AxisDescriptor("compute_dtype", ("compute_dtype",), DTYPE_VALUES),
        AxisDescriptor("accumulation_dtype", ("accumulation_dtype",), DTYPE_VALUES),
        AxisDescriptor(
            "batch_size",
            ("batch_size",),
            (),
            admission_rule=_positive_int_axis("batch_size"),
        ),
        AxisDescriptor(
            "row_batch_size",
            ("row_batch_size",),
            (),
            admission_rule=_positive_int_axis("row_batch_size"),
        ),
        AxisDescriptor(
            "token_budget",
            ("token_budget",),
            (),
            admission_rule=_positive_int_axis("token_budget"),
        ),
        AxisDescriptor(
            "attention_element_budget",
            ("attention_element_budget",),
            (),
            admission_rule=_positive_int_axis("attention_element_budget"),
        ),
        AxisDescriptor(
            "sequence_block",
            ("sequence_block",),
            (),
            admission_rule=_positive_int_axis("sequence_block"),
        ),
        AxisDescriptor(
            "checkpoint",
            ("checkpoint",),
            CHECKPOINT_VALUES,
            admission_rule=_checkpoint_axis(
                setting_key="checkpoint",
                active_values=ACTIVE_CHECKPOINT_VALUES,
            ),
        ),
        AxisDescriptor(
            "operator_path",
            ("operator_path",),
            OPERATOR_PATH_VALUES,
            admission_rule=_operator_path_axis(),
        ),
        AxisDescriptor(
            "torch_func_admission",
            TORCH_FUNC_FIELDS,
            (),
            admission_rule=_torch_func_axis(),
        ),
        AxisDescriptor(
            "functional_call_admission",
            FUNCTIONAL_CALL_FIELDS,
            (),
            admission_rule=_functional_call_axis(),
        ),
        AxisDescriptor(
            "checkpoint_admission",
            CHECKPOINT_FIELDS,
            (),
            admission_rule=_checkpoint_fields_axis(),
        ),
        AxisDescriptor(
            "vmap_chunk_size",
            ("vmap_chunk_size",),
            (),
            admission_rule=_positive_int_axis("vmap_chunk_size"),
        ),
        AxisDescriptor(
            "vmap_batch_in_dims",
            ("vmap_batch_in_dims",),
            (),
            admission_rule=_vmap_batch_in_dims_axis(),
        ),
        AxisDescriptor(
            "parameter_layout",
            ("parameter_layout",),
            PARAMETER_LAYOUT_VALUES,
        ),
        AxisDescriptor("sharding", ("sharding",), SHARDING_VALUES),
        AxisDescriptor("attention_impl", ("attention_impl",), ATTENTION_IMPL_VALUES),
        AxisDescriptor("storage_dtype", ("storage_dtype",), DTYPE_VALUES),
        AxisDescriptor(
            "metric_residency", ("metric_residency",), METRIC_RESIDENCY_VALUES
        ),
        AxisDescriptor(
            "checkpoint_policy",
            ("checkpoint_policy",),
            CHECKPOINT_POLICY_VALUES,
            admission_rule=_checkpoint_axis(
                setting_key="checkpoint_policy",
                active_values=ACTIVE_CHECKPOINT_POLICY_VALUES,
            ),
        ),
        AxisDescriptor(
            "matmul_precision", ("matmul_precision",), MATMUL_PRECISION_VALUES
        ),
        AxisDescriptor(
            "allow_tf32", ("allow_tf32",), (), admission_rule=_bool_axis("allow_tf32")
        ),
        AxisDescriptor(
            "allow_bf16_reduced_precision_reduction",
            ("allow_bf16_reduced_precision_reduction",),
            (),
            admission_rule=_bool_axis("allow_bf16_reduced_precision_reduction"),
        ),
    )


def standard_axis_registry() -> AxisRegistry:
    """Return a registry populated with standard core axes."""
    registry = AxisRegistry()

    for axis in standard_axis_descriptors():
        registry.register(axis)

    return registry


def settings_product(
    family: str,
    axes: Mapping[str, Sequence[Any]],
    *,
    axis_registry: AxisRegistry | None = None,
    generator_id: str = "grid",
    generator_version: str = "0.0.1",
) -> tuple[Candidate, ...]:
    """Create candidates from an ordered grid of axis values.

    Returns:
        Candidate grid in deterministic order.
    """
    items = tuple(axes.items())
    candidates = []

    def build(index: int, settings: dict[str, Any], changed: tuple[str, ...]) -> None:
        if index == len(items):
            candidate_id = f"{family}:{len(candidates)}"
            candidates.append(
                Candidate(
                    family=family,
                    candidate_id=candidate_id,
                    settings=dict(settings),
                    changed_axes=changed,
                    generator_id=generator_id,
                    generator_version=generator_version,
                )
            )

            return

        axis_name, values = items[index]

        for value in values:
            changed_settings = _settings_for_axis_value(
                axis_name,
                value,
                axis_registry,
            )
            settings.update(changed_settings)
            build(index + 1, settings, (*changed, axis_name))

        for key in changed_settings:
            del settings[key]

    build(0, {}, ())

    return tuple(candidates)


def _settings_for_axis_value(
    axis_name: str,
    value: Any,
    axis_registry: AxisRegistry | None,
) -> dict[str, Any]:
    if axis_registry is None:
        return {axis_name: value}

    axis = axis_registry.axes.get(axis_name)

    if axis is None:
        message = f"candidate axis is unknown: {axis_name}"
        raise AdmissionError(message)

    if len(axis.settings_keys) == 1:
        return {axis.settings_keys[0]: value}

    if not isinstance(value, Mapping):
        message = f"multi-key axis value must be a mapping: {axis_name}"
        raise AdmissionError(message)

    if set(value) != set(axis.settings_keys):
        message = f"multi-key axis value keys differ from axis settings: {axis_name}"
        raise AdmissionError(message)

    return {key: value[key] for key in axis.settings_keys}
