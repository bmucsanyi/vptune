"""Transformers adapter admission helpers."""

import contextlib
import dataclasses
import math
from collections.abc import Callable, Iterator, Mapping, Sequence
from typing import Any, Protocol

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from vptune.attention import (
    AttentionSemantics,
    MappingAttentionLocation,
)
from vptune.attention import (
    check_patched_attention_output_reference as _check_patched_attention_output,
)
from vptune.attention import (
    check_patched_attention_vjp_reference as _check_patched_attention_vjp,
)
from vptune.candidates import AxisDescriptor
from vptune.checks import tree_error_measurements, validate_thresholds
from vptune.data import (
    PACKAGE_VERSION,
    Batch,
    BufferTree,
    CallableMaterializer,
    CallableOperationFactory,
    CallableReferenceCheck,
    Candidate,
    CandidateAdmitter,
    CandidateOperation,
    FullSizeCheck,
    FullSizeRecord,
    FunctionObjective,
    Measurement,
    ModuleCallSpec,
    OperationFactory,
    OperatorSpec,
    ParameterSurface,
    ParameterTree,
    ReferenceCheck,
    ReferenceResult,
    RuntimeConfig,
    ScalarObjective,
    TensorTree,
)
from vptune.errors import AdmissionError, MaterializationError
from vptune.identities import module_identity
from vptune.runtime import (
    _compile_backend,
    _compile_bool,
    _compile_mode,
    _compile_optional_bool,
    _compile_options,
    _compiled_autograd_patch,
    _validate_compile_cache_state,
    standard_operation_factory,
    standard_reference_check,
)
from vptune.tensor_tree import tree_signature

EAGER_ATTENTION_FRONTENDS = (
    "transformers_eager",
    "paged|eager",
)
SDPA_ATTENTION_FRONTENDS = (
    "transformers_sdpa",
    "paged|sdpa",
)
FLASH_ATTENTION_FRONTENDS = (
    "transformers_flash_attention_2",
    "transformers_flash_attention_3",
    "transformers_flash_attention_4",
    "paged|flash_attention_2",
    "paged|flash_attention_3",
    "paged|flash_attention_4",
)
CUSTOM_ATTENTION_FRONTENDS = (
    "transformers_flex_attention",
    "registered_transformers_attention",
)
TRANSFORMERS_ATTENTION_FRONTENDS = (
    "transformers_eager",
    "transformers_sdpa",
    "transformers_flash_attention_2",
    "transformers_flash_attention_3",
    "transformers_flash_attention_4",
    "transformers_flex_attention",
    "paged|eager",
    "paged|sdpa",
    "paged|flash_attention_2",
    "paged|flash_attention_3",
    "paged|flash_attention_4",
    "registered_transformers_attention",
)
PUBLIC_ATTENTION_FRONTEND_ALIASES = {
    "eager": "transformers_eager",
    "sdpa": "transformers_sdpa",
    "flash_attention_2": "transformers_flash_attention_2",
    "flash_attention_3": "transformers_flash_attention_3",
    "flash_attention_4": "transformers_flash_attention_4",
    "flex_attention": "transformers_flex_attention",
}
SDPA_KERNELS = (
    "math",
    "flash_attention",
    "efficient_attention",
    "cudnn_attention",
    "overrideable",
    "priority_list",
)
NON_MATH_SDPA_KERNELS = tuple(kernel for kernel in SDPA_KERNELS if kernel != "math")
SDPA_KERNEL_BACKENDS = {
    "math": SDPBackend.MATH,
    "flash_attention": SDPBackend.FLASH_ATTENTION,
    "efficient_attention": SDPBackend.EFFICIENT_ATTENTION,
    "cudnn_attention": SDPBackend.CUDNN_ATTENTION,
    "overrideable": SDPBackend.OVERRIDEABLE,
}
FLASH_ATTENTION_DTYPES = ("fp16", "bf16")
LOAD_TIME_ATTENTION_FRONTENDS = {
    "transformers_eager": "eager",
    "transformers_sdpa": "sdpa",
    "transformers_flash_attention_2": "flash_attention_2",
    "transformers_flash_attention_3": "flash_attention_3",
    "transformers_flash_attention_4": "flash_attention_4",
    "transformers_flex_attention": "flex_attention",
    "paged|eager": "paged|eager",
    "paged|sdpa": "paged|sdpa",
    "paged|flash_attention_2": "paged|flash_attention_2",
    "paged|flash_attention_3": "paged|flash_attention_3",
    "paged|flash_attention_4": "paged|flash_attention_4",
}
TRANSFORMERS_RUNTIME_SETTINGS = (
    "attention.frontend",
    "attention.sdpa_kernel",
    "attention.sdpa_priority_list",
    "attention.custom_kernel_id",
    "attention.mask_formatter_id",
    "output_attentions",
    "module_mode",
    "dropout_p",
)
COMPILE_RUNTIME_SETTINGS = (
    "compile.enabled",
    "compile.boundary",
    "compile.backend",
    "compile.mode",
    "compile.fullgraph",
    "compile.dynamic",
    "compile.compiled_autograd",
    "compile.options.epilogue_fusion",
    "compile.options.shape_padding",
    "compile.cuda_graphs",
    "compile.cache_state",
)
TRANSFORMERS_ADAPTER_COMPILE_BOUNDARIES = ("transformer_block", "attention_module")


@dataclasses.dataclass(frozen=True, slots=True)
class _ForwardPatch:
    module: torch.nn.Module
    original_forward: Callable[..., Any] | None
    compiled_forward: Callable[..., Any]


@dataclasses.dataclass(frozen=True, slots=True)
class TransformersAttentionPolicy:
    """Semantic identity and admission policy for Transformers attention."""

    model_config_hash: str
    use_cache: bool
    softcap: Mapping[str, Any]
    mask_semantics: str
    causal_policy: str
    backend_numeric_policy: Mapping[str, Any]
    determinism: Mapping[str, Any]
    padding_limit: int | None
    forced_kernel_available: bool
    forced_kernel_failure_reason: str | None = None

    def signature(self) -> dict[str, Any]:
        """Return stable policy identity."""
        return {
            "model_config_hash": self.model_config_hash,
            "use_cache": self.use_cache,
            "softcap": dict(self.softcap),
            "mask_semantics": self.mask_semantics,
            "causal_policy": self.causal_policy,
            "backend_numeric_policy": dict(self.backend_numeric_policy),
            "determinism": dict(self.determinism),
            "padding_limit": self.padding_limit,
            "forced_kernel_available": self.forced_kernel_available,
            "forced_kernel_failure_reason": self.forced_kernel_failure_reason,
        }


@dataclasses.dataclass(frozen=True, slots=True)
class TransformersAttentionSpace:
    """Public search-space component for Transformers attention frontends."""

    frontends: tuple[str, ...]
    policy: TransformersAttentionPolicy
    sdpa_kernel: str
    module_mode: str
    dropout_p: float
    sdpa_priority_list: tuple[str, ...] = ()
    attention_custom_kernel_id: str | None = None
    mask_formatter_id: str | None = None

    def __post_init__(self) -> None:
        """Validate frontend and admission fields.

        Raises:
            AdmissionError: If an attention field is invalid.
        """
        normalized = _normalize_public_attention_frontends(self.frontends)

        if self.sdpa_kernel not in SDPA_KERNELS:
            message = f"unsupported SDPA kernel: {self.sdpa_kernel}"
            raise AdmissionError(message)

        if self.module_mode not in {"train", "eval"}:
            message = f"unsupported module_mode: {self.module_mode}"
            raise AdmissionError(message)

        if not isinstance(self.dropout_p, float | int) or self.dropout_p < 0.0:
            message = "dropout_p must be a nonnegative number"
            raise AdmissionError(message)

        if self.module_mode == "eval" and not math.isclose(
            float(self.dropout_p),
            0.0,
        ):
            message = "eval attention rows require dropout_p=0.0"
            raise AdmissionError(message)

        if self.sdpa_kernel == "priority_list":
            _require_sdpa_priority_list(self.sdpa_priority_list)
        elif self.sdpa_priority_list:
            message = "attention.sdpa_priority_list requires priority_list"
            raise AdmissionError(message)

        if "registered_transformers_attention" in normalized:
            if self.attention_custom_kernel_id is None:
                message = "registered attention requires attention_custom_kernel_id"
                raise AdmissionError(message)

            if self.mask_formatter_id is None:
                message = "registered attention requires mask_formatter_id"
                raise AdmissionError(message)

        object.__setattr__(self, "frontends", normalized)
        object.__setattr__(self, "dropout_p", float(self.dropout_p))
        object.__setattr__(
            self,
            "sdpa_priority_list",
            tuple(self.sdpa_priority_list),
        )

    def axes_for(self, operator: object) -> Mapping[str, Sequence[Any]]:
        """Return adapter attention axes for a public operator."""
        _ = operator

        return {"transformers_attention_frontend": _attention_axis_values(self)}

    def axis_descriptors(self) -> tuple[AxisDescriptor, ...]:
        """Return adapter descriptors registered by this component."""
        return (_attention_space_axis(self),)

    def settings_for(self, operator: object) -> Mapping[str, Any]:
        """Return fixed admission settings for generated attention rows."""
        _ = operator
        settings = dict[str, Any]()
        settings["module_mode"] = self.module_mode
        settings["dropout_p"] = self.dropout_p

        if self.sdpa_kernel == "priority_list":
            settings["attention.sdpa_priority_list"] = self.sdpa_priority_list

        if self.attention_custom_kernel_id is not None:
            settings["attention.custom_kernel_id"] = self.attention_custom_kernel_id

        if self.mask_formatter_id is not None:
            settings["attention.mask_formatter_id"] = self.mask_formatter_id

        return settings


def attention_space(
    *,
    frontends: Sequence[str],
    sdpa_kernel: str = "math",
    module_mode: str = "eval",
    dropout_p: float = 0.0,
    sdpa_priority_list: Sequence[str] = (),
    attention_custom_kernel_id: str | None = None,
    mask_formatter_id: str | None = None,
    policy: TransformersAttentionPolicy | None = None,
) -> TransformersAttentionSpace:
    """Build a public search-space component for Transformers attention.

    Returns:
        Transformers attention search-space component.
    """
    attention_policy = _public_attention_policy() if policy is None else policy

    return TransformersAttentionSpace(
        frontends=tuple(frontends),
        policy=attention_policy,
        sdpa_kernel=sdpa_kernel,
        module_mode=module_mode,
        dropout_p=dropout_p,
        sdpa_priority_list=tuple(sdpa_priority_list),
        attention_custom_kernel_id=attention_custom_kernel_id,
        mask_formatter_id=mask_formatter_id,
    )


@dataclasses.dataclass(frozen=True, slots=True)
class TransformersModelIdentity:
    """Serializable identity for a Transformers model adapter."""

    transformers_version: str
    model_config_hash: str
    module: Mapping[str, Any]
    source_revision: str | None
    dtype_policy: Mapping[str, Any]
    tokenizer_identity: Mapping[str, Any]
    adapter_rules: Mapping[str, Any]

    def signature(self) -> dict[str, Any]:
        """Return stable model identity."""
        return {
            "adapter_id": "vptune.transformers",
            "adapter_version": PACKAGE_VERSION,
            "transformers_version": self.transformers_version,
            "model_config_hash": self.model_config_hash,
            "module": dict(self.module),
            "source_revision": self.source_revision,
            "dtype_policy": dict(self.dtype_policy),
            "tokenizer_identity": dict(self.tokenizer_identity),
            "adapter_rules": dict(self.adapter_rules),
        }


class TransformersModelLoader(Protocol):
    """Object with a Transformers-style loader."""

    def from_pretrained(
        self,
        model_name_or_path: str,
        *,
        revision: str,
        torch_dtype: torch.dtype,
        attn_implementation: str,
        use_cache: bool,
    ) -> torch.nn.Module:
        """Return a loaded model."""


class TransformersAttentionConfigurable(Protocol):
    """Object with a Transformers attention switch method."""

    def set_attn_implementation(self, attn_implementation: str) -> None:
        """Set the active Transformers attention implementation."""


class TransformersMaskConfigurable(Protocol):
    """Object with a Transformers attention-mask formatter switch method."""

    def set_attention_mask_formatter(self, mask_formatter_id: str) -> None:
        """Set the active Transformers attention-mask formatter."""


class TransformersRegistry(Protocol):
    """Object with a Transformers-style register method."""

    def register(self, name: str, function: Callable[..., Any]) -> None:
        """Register a named callable."""


def load_transformers_model(
    model_cls: TransformersModelLoader,
    *,
    model_name_or_path: str,
    revision: str,
    torch_dtype: torch.dtype,
    attention_frontend: str,
    attention_custom_kernel_id: str | None,
    use_cache: bool,
) -> torch.nn.Module:
    """Load a Transformers model with explicit execution settings.

    Returns:
        Loaded model.
    """
    return model_cls.from_pretrained(
        model_name_or_path,
        revision=revision,
        torch_dtype=torch_dtype,
        attn_implementation=transformers_attn_implementation(
            attention_frontend,
            attention_custom_kernel_id,
        ),
        use_cache=use_cache,
    )


def transformers_attn_implementation(
    attention_frontend: str,
    attention_custom_kernel_id: str | None,
) -> str:
    """Return the Transformers loader value for an attention frontend.

    Returns:
        `from_pretrained(attn_implementation=...)` value.

    Raises:
        AdmissionError: If the frontend is not a load-time Transformers frontend.
    """
    if attention_frontend == "registered_transformers_attention":
        if not attention_custom_kernel_id:
            message = (
                "registered_transformers_attention requires attention_custom_kernel_id"
            )
            raise AdmissionError(message)

        return attention_custom_kernel_id

    attn_implementation = LOAD_TIME_ATTENTION_FRONTENDS.get(attention_frontend)

    if attn_implementation is None:
        message = (
            f"attention frontend is not load-time selectable: {attention_frontend}"
        )
        raise AdmissionError(message)

    return attn_implementation


def set_transformers_attention_implementation(
    model: TransformersAttentionConfigurable,
    *,
    attention_frontend: str,
    attention_custom_kernel_id: str | None,
) -> str:
    """Set the active Transformers attention implementation.

    Returns:
        Transformers attention implementation string sent to the model.
    """
    attn_implementation = transformers_attn_implementation(
        attention_frontend,
        attention_custom_kernel_id,
    )
    model.set_attn_implementation(attn_implementation)

    return attn_implementation


def set_transformers_attention_mask_formatter(
    model: TransformersMaskConfigurable,
    *,
    mask_formatter_id: str,
) -> str:
    """Set the active Transformers attention-mask formatter.

    Returns:
        Mask formatter id sent to the model.
    """
    model.set_attention_mask_formatter(mask_formatter_id)

    return mask_formatter_id


def register_transformers_attention(
    attention_interface: TransformersRegistry,
    mask_interface: TransformersRegistry,
    *,
    attention_custom_kernel_id: str,
    attention_function: Callable[..., Any],
    mask_formatter_id: str,
    mask_function: Callable[..., Any],
) -> dict[str, Any]:
    """Register a custom Transformers attention implementation and mask.

    Returns:
        Registration identity.

    Raises:
        AdmissionError: If the declared attention and mask ids cannot execute together.
    """
    settings = {
        "attention.custom_kernel_id": attention_custom_kernel_id,
        "attention.mask_formatter_id": mask_formatter_id,
    }
    error = _registered_attention_ids_error(settings)

    if error is not None:
        raise AdmissionError(error)

    attention_interface.register(attention_custom_kernel_id, attention_function)
    mask_interface.register(mask_formatter_id, mask_function)

    return {
        "attention_custom_kernel_id": attention_custom_kernel_id,
        "mask_formatter_id": mask_formatter_id,
    }


def transformers_operation_factory(
    operator: OperatorSpec,
    *,
    model: Any,
    params: ParameterTree,
    buffers: BufferTree,
    module_call: ModuleCallSpec,
    parameter_surface: ParameterSurface | None = None,
    scalar_objectives: Mapping[str, ScalarObjective] | None = None,
    function_objectives: Mapping[str, FunctionObjective] | None = None,
    attention_custom_kernel_id: str | None = None,
    transformer_block_paths: Sequence[str] = (),
    attention_module_paths: Sequence[str] = (),
) -> OperationFactory:
    """Return a Transformers-backed standard operation factory."""
    standard_factory = standard_operation_factory(
        operator,
        params=params,
        buffers=buffers,
        parameter_surface=parameter_surface,
        scalar_objectives=scalar_objectives,
        function_objectives=function_objectives,
        module=model,
        module_call=module_call,
    )

    def factory(
        candidate: Candidate,
        batch: Batch,
        vector: TensorTree,
    ) -> CandidateOperation:
        _configure_transformers_runtime(
            model,
            candidate,
            attention_custom_kernel_id=attention_custom_kernel_id,
        )
        standard_candidate = _standard_candidate(candidate)
        forward_patches = _transformers_forward_patches(
            model,
            candidate.settings,
            transformer_block_paths=transformer_block_paths,
            attention_module_paths=attention_module_paths,
        )

        with _transformers_sdpa_kernel_context(candidate.settings):
            operation = standard_factory(standard_candidate, batch, vector)

        if _transformers_compile_cache_state(candidate.settings) == "warm_cache":
            with (
                _installed_transformers_forward_patches(forward_patches),
                _transformers_sdpa_kernel_context(candidate.settings),
            ):
                operation()

        def transformers_operation() -> TensorTree:
            with (
                _installed_transformers_forward_patches(forward_patches),
                _transformers_sdpa_kernel_context(candidate.settings),
            ):
                return operation()

        return transformers_operation

    return factory


def transformers_reference_check(
    operator: OperatorSpec,
    *,
    model: Any,
    params: ParameterTree,
    buffers: BufferTree,
    module_call: ModuleCallSpec,
    thresholds: Mapping[str, float],
    parameter_surface: ParameterSurface | None = None,
    numeric_bound_fields: Mapping[str, Any] | None = None,
    scalar_objectives: Mapping[str, ScalarObjective] | None = None,
    function_objectives: Mapping[str, FunctionObjective] | None = None,
    attention_custom_kernel_id: str | None = None,
    transformer_block_paths: Sequence[str] = (),
    attention_module_paths: Sequence[str] = (),
) -> ReferenceCheck:
    """Return a Transformers-backed standard reference check."""
    standard_check = standard_reference_check(
        operator,
        params=params,
        buffers=buffers,
        thresholds=thresholds,
        parameter_surface=parameter_surface,
        numeric_bound_fields=numeric_bound_fields,
        scalar_objectives=scalar_objectives,
        function_objectives=function_objectives,
        module=model,
        module_call=module_call,
    )

    def check(
        candidate: Candidate,
        batch: Batch,
        vector: TensorTree,
    ) -> ReferenceResult:
        _configure_transformers_runtime(
            model,
            candidate,
            attention_custom_kernel_id=attention_custom_kernel_id,
        )

        standard_candidate = _standard_candidate(candidate)

        forward_patches = _transformers_forward_patches(
            model,
            candidate.settings,
            transformer_block_paths=transformer_block_paths,
            attention_module_paths=attention_module_paths,
        )

        with (
            _installed_transformers_forward_patches(forward_patches),
            _transformers_sdpa_kernel_context(candidate.settings),
        ):
            return standard_check(standard_candidate, batch, vector)

    return check


def check_patched_attention_output_reference(
    reference: Callable[..., TensorTree],
    patched: Callable[..., TensorTree],
    args: Sequence[Any],
    *,
    thresholds: Mapping[str, float],
) -> ReferenceResult:
    """Return adapter reference result for patched attention output.

    Returns:
        Reference result with output error measurements.
    """
    measurements = _check_patched_attention_output(
        reference,
        patched,
        args,
        thresholds=thresholds,
    )

    return ReferenceResult(
        "patched_attention_output",
        dict(thresholds),
        measurements,
    )


def check_patched_attention_vjp_reference(
    reference: Callable[..., torch.Tensor],
    patched: Callable[..., torch.Tensor],
    args: Sequence[Any],
    cotangent: torch.Tensor,
    differentiable_arg_indices: Sequence[int],
    *,
    thresholds: Mapping[str, float],
) -> ReferenceResult:
    """Return adapter reference result for patched attention VJP.

    Returns:
        Reference result with VJP error measurements.
    """
    measurements = _check_patched_attention_vjp(
        reference,
        patched,
        args,
        cotangent,
        differentiable_arg_indices,
        thresholds=thresholds,
    )

    return ReferenceResult(
        "patched_attention_vjp",
        dict(thresholds),
        measurements,
    )


@dataclasses.dataclass(frozen=True, slots=True)
class TransformersFullSizeCheck:
    """Full-size check for Transformers-backed rows."""

    operation_factory: OperationFactory
    reference_check: ReferenceCheck
    thresholds: Mapping[str, float]

    def __post_init__(self) -> None:
        """Validate output comparison thresholds.

        Raises:
            MaterializationError: If required thresholds are missing.
        """
        for key in ("max_abs_diff", "max_rel_diff"):
            if key not in self.thresholds:
                message = f"Transformers full-size check requires threshold: {key}"
                raise MaterializationError(message)

    def identity(self) -> Mapping[str, Any]:
        """Return stable full-size check identity."""
        return {
            "full_size_check": "vptune.transformers.full_size",
            "thresholds": {
                "max_abs_diff": self.thresholds["max_abs_diff"],
                "max_rel_diff": self.thresholds["max_rel_diff"],
            },
        }

    def __call__(
        self,
        candidate: Candidate,
        inputs: tuple[tuple[Batch, TensorTree], ...],
        output: TensorTree,
        samples: tuple[Measurement, ...],
    ) -> Mapping[str, Any]:
        """Validate measured full-size outputs against rerun and reference checks.

        Returns:
            Selection metadata for the checked row.
        """
        del samples

        outputs = _full_size_output_tuple(output, len(inputs))
        max_abs = 0.0
        max_rel = 0.0

        for (batch, vector), observed in zip(inputs, outputs, strict=True):
            rerun = self.operation_factory(candidate, batch, vector)()
            measurements = tree_error_measurements(observed, rerun)
            validate_thresholds(measurements, self._output_thresholds())
            self.reference_check(candidate, batch, vector)
            max_abs = max(max_abs, float(measurements["max_abs_diff"]))
            max_rel = max(max_rel, float(measurements["max_rel_diff"]))

        return {
            "transformers_full_size_max_abs_diff": max_abs,
            "transformers_full_size_max_rel_diff": max_rel,
            "transformers_full_size_checked_inputs": len(inputs),
        }

    def _output_thresholds(self) -> dict[str, float]:
        return {
            "max_abs_diff": self.thresholds["max_abs_diff"],
            "max_rel_diff": self.thresholds["max_rel_diff"],
        }


def transformers_full_size_check(
    *,
    operation_factory: OperationFactory,
    reference_check: ReferenceCheck,
    thresholds: Mapping[str, float],
) -> FullSizeCheck:
    """Return a Transformers full-size checker."""
    return TransformersFullSizeCheck(
        operation_factory=operation_factory,
        reference_check=reference_check,
        thresholds=dict(thresholds),
    )


def transformers_runtime_config(
    operator: OperatorSpec,
    *,
    model: Any,
    params: ParameterTree,
    buffers: BufferTree,
    candidates: Sequence[Candidate],
    thresholds: Mapping[str, float],
    objective_signature: Mapping[str, Any],
    module_call: ModuleCallSpec,
    axis_registry: CandidateAdmitter | None,
    parameter_surface: ParameterSurface | None = None,
    numeric_bound_fields: Mapping[str, Any] | None = None,
    scalar_objectives: Mapping[str, ScalarObjective] | None = None,
    function_objectives: Mapping[str, FunctionObjective] | None = None,
    attention_custom_kernel_id: str | None = None,
    transformer_block_paths: Sequence[str] = (),
    attention_module_paths: Sequence[str] = (),
) -> RuntimeConfig:
    """Return a runtime config for Transformers-backed standard operators."""
    operation_factory = transformers_operation_factory(
        operator,
        model=model,
        params=params,
        buffers=buffers,
        module_call=module_call,
        parameter_surface=parameter_surface,
        scalar_objectives=scalar_objectives,
        function_objectives=function_objectives,
        attention_custom_kernel_id=attention_custom_kernel_id,
        transformer_block_paths=transformer_block_paths,
        attention_module_paths=attention_module_paths,
    )
    reference_check = transformers_reference_check(
        operator,
        model=model,
        params=params,
        buffers=buffers,
        module_call=module_call,
        thresholds=thresholds,
        parameter_surface=parameter_surface,
        numeric_bound_fields=numeric_bound_fields,
        scalar_objectives=scalar_objectives,
        function_objectives=function_objectives,
        attention_custom_kernel_id=attention_custom_kernel_id,
        transformer_block_paths=transformer_block_paths,
        attention_module_paths=attention_module_paths,
    )
    runtime_signature = {
        "runtime": "transformers",
        "operator": operator.signature(),
        "model": module_identity(model),
        "params": tree_signature(params),
        "buffers": tree_signature(buffers),
        "parameter_surface": (
            None if parameter_surface is None else parameter_surface.signature()
        ),
        "thresholds": dict(thresholds),
        "numeric_bound_fields": {}
        if numeric_bound_fields is None
        else dict(numeric_bound_fields),
        "objective": dict(objective_signature),
        "module_call": module_call.signature(),
        "transformer_block_paths": tuple(transformer_block_paths),
        "attention_module_paths": tuple(attention_module_paths),
    }
    operation_factory = CallableOperationFactory(
        "vptune.transformers_operation_factory",
        PACKAGE_VERSION,
        runtime_signature,
        {"callback": "vptune.adapters.transformers.transformers_operation_factory"},
        operation_factory,
    )
    reference_check = CallableReferenceCheck(
        "vptune.transformers_reference_check",
        PACKAGE_VERSION,
        runtime_signature,
        {"callback": "vptune.adapters.transformers.transformers_reference_check"},
        reference_check,
    )
    full_size_check = transformers_full_size_check(
        operation_factory=operation_factory,
        reference_check=reference_check,
        thresholds=thresholds,
    )
    materializer = CallableMaterializer(
        "vptune.transformers_runtime",
        PACKAGE_VERSION,
        {"operation_factory": dict(operation_factory.identity())},
        {"callback": "_materialize_transformers_selected"},
        lambda candidate, record: _materialize_transformers_selected(
            operation_factory,
            candidate,
            record,
        ),
    )

    return RuntimeConfig(
        candidates=tuple(candidates),
        operation_factory=operation_factory,
        reference_check=reference_check,
        materializer=materializer,
        axis_registry=axis_registry,
        reference_check_name="standard_anchor",
        signature=runtime_signature,
        full_size_check=full_size_check,
    )


def _materialize_transformers_selected(
    operation_factory: OperationFactory,
    candidate: Candidate,
    record: FullSizeRecord,
) -> Any:
    if (
        record.family != candidate.family
        or record.candidate_id != candidate.candidate_id
    ):
        message = "selected record does not match selected Transformers candidate"
        raise MaterializationError(message)

    def selected(batch: Batch, vector: TensorTree) -> TensorTree:
        return operation_factory(candidate, batch, vector)()

    return selected


def _full_size_output_tuple(
    output: TensorTree,
    expected_count: int,
) -> tuple[Any, ...]:
    if not isinstance(output, tuple):
        message = "Transformers full-size output must be a tuple"
        raise MaterializationError(message)

    if len(output) != expected_count:
        message = "Transformers full-size output count differs from inputs"
        raise MaterializationError(message)

    return tuple(output)


def _configure_transformers_runtime(
    model: Any,
    candidate: Candidate,
    *,
    attention_custom_kernel_id: str | None,
) -> None:
    _require_transformers_runtime_row_settings(candidate.settings)
    attention_frontend = candidate.settings.get("attention.frontend")

    if isinstance(attention_frontend, str):
        set_transformers_attention_implementation(
            model,
            attention_frontend=attention_frontend,
            attention_custom_kernel_id=_runtime_attention_custom_kernel_id(
                candidate.settings,
                attention_custom_kernel_id,
            ),
        )
        mask_formatter_id = _runtime_attention_mask_formatter_id(candidate.settings)

        if mask_formatter_id is not None:
            _set_transformers_attention_mask_formatter(model, mask_formatter_id)

    module_mode = candidate.settings.get("module_mode")

    if module_mode is None:
        return

    if module_mode == "eval":
        model.eval()

        return

    if module_mode == "train":
        model.train()

        return

    message = f"module_mode is unsupported: {module_mode}"
    raise AdmissionError(message)


def _require_transformers_runtime_row_settings(settings: Mapping[str, Any]) -> None:
    if "use_cache" in settings:
        message = "use_cache is a Transformers model-load setting"
        raise AdmissionError(message)

    if settings.get("output_attentions") is True:
        message = "output_attentions=True requires an attention-weights output surface"
        raise AdmissionError(message)


def _runtime_attention_custom_kernel_id(
    settings: Mapping[str, Any],
    configured_id: str | None,
) -> str | None:
    row_id = settings.get("attention.custom_kernel_id")
    mask_id = settings.get("attention.mask_formatter_id")

    if settings.get("attention.frontend") != "registered_transformers_attention":
        if row_id is not None or mask_id is not None:
            message = (
                "registered attention ids apply only to "
                "registered_transformers_attention"
            )
            raise AdmissionError(message)

        return configured_id

    error = _registered_attention_ids_error(settings)

    if error is not None:
        raise AdmissionError(error)

    if configured_id is not None and configured_id != row_id:
        message = (
            "runtime attention_custom_kernel_id must match attention.custom_kernel_id"
        )
        raise AdmissionError(message)

    return row_id


def _runtime_attention_mask_formatter_id(
    settings: Mapping[str, Any],
) -> str | None:
    if settings.get("attention.frontend") != "registered_transformers_attention":
        return None

    error = _registered_attention_ids_error(settings)

    if error is not None:
        raise AdmissionError(error)

    value = settings["attention.mask_formatter_id"]

    if not isinstance(value, str):
        message = "attention.mask_formatter_id must be a string"
        raise AdmissionError(message)

    return value


def _set_transformers_attention_mask_formatter(
    model: Any,
    mask_formatter_id: str,
) -> None:
    setter = getattr(model, "set_attention_mask_formatter", None)

    if not callable(setter):
        message = (
            "registered_transformers_attention requires set_attention_mask_formatter"
        )
        raise AdmissionError(message)

    setter(mask_formatter_id)


@contextlib.contextmanager
def _transformers_sdpa_kernel_context(
    settings: Mapping[str, Any],
) -> Iterator[None]:
    if settings.get("attention.frontend") not in SDPA_ATTENTION_FRONTENDS:
        yield

        return

    kernel = settings.get("attention.sdpa_kernel")

    if kernel == "priority_list":
        priority_list = _sdpa_priority_list(settings)

        with sdpa_kernel(priority_list, set_priority=True):
            yield

        return

    backend = _sdpa_backend(kernel)

    with sdpa_kernel(backend, set_priority=False):
        yield


def _sdpa_priority_list(settings: Mapping[str, Any]) -> list[SDPBackend]:
    priority = settings.get("attention.sdpa_priority_list")

    if not isinstance(priority, Sequence) or isinstance(priority, str) or not priority:
        message = "attention.sdpa_priority_list must be a non-empty sequence"
        raise AdmissionError(message)

    return [_sdpa_backend(value) for value in priority]


def _sdpa_backend(value: object) -> SDPBackend:
    if not isinstance(value, str):
        message = "attention.sdpa_kernel must be a string"
        raise AdmissionError(message)

    backend = SDPA_KERNEL_BACKENDS.get(value)

    if backend is None:
        message = f"unsupported SDPA kernel: {value}"
        raise AdmissionError(message)

    return backend


def _standard_candidate(candidate: Candidate) -> Candidate:
    settings = {
        key: value
        for key, value in candidate.settings.items()
        if key not in TRANSFORMERS_RUNTIME_SETTINGS
    }

    if _transformers_adapter_compile_boundary(candidate.settings) is not None:
        settings = {
            key: value
            for key, value in settings.items()
            if key not in COMPILE_RUNTIME_SETTINGS
        }

    return dataclasses.replace(candidate, settings=settings)


def _transformers_forward_patches(
    model: torch.nn.Module,
    settings: Mapping[str, Any],
    *,
    transformer_block_paths: Sequence[str],
    attention_module_paths: Sequence[str],
) -> tuple[_ForwardPatch, ...]:
    paths = _transformers_compile_paths(
        settings,
        transformer_block_paths=transformer_block_paths,
        attention_module_paths=attention_module_paths,
    )

    return tuple(_transformers_forward_patch(model, path, settings) for path in paths)


def _transformers_compile_paths(
    settings: Mapping[str, Any],
    *,
    transformer_block_paths: Sequence[str],
    attention_module_paths: Sequence[str],
) -> tuple[str, ...]:
    boundary = _transformers_adapter_compile_boundary(settings)

    if boundary is None:
        return ()

    paths = (
        transformer_block_paths
        if boundary == "transformer_block"
        else attention_module_paths
    )
    result = tuple(paths)

    if not result:
        message = f"compile.boundary={boundary} requires declared module paths"
        raise MaterializationError(message)

    if not all(isinstance(path, str) and path for path in result):
        message = f"compile.boundary={boundary} module paths must be non-empty strings"
        raise MaterializationError(message)

    if len(set(result)) != len(result):
        message = f"compile.boundary={boundary} module paths must be unique"
        raise MaterializationError(message)

    _validate_compile_cache_state(settings)

    return result


def _transformers_adapter_compile_boundary(settings: Mapping[str, Any]) -> str | None:
    if settings.get("compile.enabled") != "true":
        return None

    boundary = settings.get("compile.boundary")

    if boundary in TRANSFORMERS_ADAPTER_COMPILE_BOUNDARIES:
        return boundary

    return None


def _transformers_compile_cache_state(settings: Mapping[str, Any]) -> str | None:
    if _transformers_adapter_compile_boundary(settings) is None:
        return None

    value = settings.get("compile.cache_state")

    if value not in {"cold_compile", "warm_cache"}:
        message = "compile.cache_state must be cold_compile or warm_cache"
        raise MaterializationError(message)

    return value


def _transformers_forward_patch(
    model: torch.nn.Module,
    path: str,
    settings: Mapping[str, Any],
) -> _ForwardPatch:
    module = model.get_submodule(path)
    original_forward = _instance_forward(module)
    compiled_forward = _compile_transformers_forward(settings, module.forward)

    return _ForwardPatch(
        module=module,
        original_forward=original_forward,
        compiled_forward=compiled_forward,
    )


def _instance_forward(module: torch.nn.Module) -> Callable[..., Any] | None:
    if "forward" not in module.__dict__:
        return None

    value = module.__dict__["forward"]

    if not callable(value):
        message = "module instance forward must be callable"
        raise MaterializationError(message)

    return value


def _compile_transformers_forward(
    settings: Mapping[str, Any],
    forward: Callable[..., Any],
) -> Callable[..., Any]:
    compiled_autograd = _compile_bool(settings, "compile.compiled_autograd")

    def build_compiled() -> Callable[..., Any]:
        return torch.compile(
            forward,
            backend=_compile_backend(settings),
            mode=_compile_mode(settings),
            fullgraph=_compile_bool(settings, "compile.fullgraph"),
            dynamic=_compile_optional_bool(settings, "compile.dynamic"),
            options=_compile_options(settings),
        )

    if compiled_autograd:
        with _compiled_autograd_patch():
            compiled = build_compiled()
    else:
        compiled = build_compiled()

    def compiled_forward(*args: Any, **kwargs: Any) -> Any:
        if compiled_autograd:
            with _compiled_autograd_patch():
                return compiled(*args, **kwargs)

        return compiled(*args, **kwargs)

    return compiled_forward


@contextlib.contextmanager
def _installed_transformers_forward_patches(
    patches: tuple[_ForwardPatch, ...],
) -> Iterator[None]:
    for patch in patches:
        patch.module.forward = patch.compiled_forward

    try:
        yield
    finally:
        for patch in reversed(patches):
            if patch.original_forward is not None:
                patch.module.forward = patch.original_forward
            elif "forward" in patch.module.__dict__:
                del patch.module.forward


def transformers_attention_location(
    *,
    query_key: str,
    key_key: str,
    value_key: str,
    output_key: str,
    mask_key: str | None,
    inverse_permutation_key: str | None,
    query_block_size_key: str | None,
    dropout_p: float,
    is_causal: bool,
    scale: float | None,
    enable_gqa: bool,
    causal_policy: str,
    sliding_window_policy: str,
    padding_policy: str,
    mask_convention: str,
    dropout_rng: Mapping[str, Any],
    qkv_layout: str,
    head_layout: str,
    scale_source: str,
    use_cache: bool,
    output_attentions: bool,
    rope_parameters: Mapping[str, Any],
    position_id_policy: Mapping[str, Any],
    score_softcap: float | None,
    final_logit_softcap: float | None,
) -> MappingAttentionLocation:
    """Return a core attention-location descriptor for Transformers rows.

    Returns:
        Core attention-location descriptor.
    """
    return MappingAttentionLocation(
        semantics=AttentionSemantics(
            causal_policy=causal_policy,
            sliding_window_policy=sliding_window_policy,
            padding_policy=padding_policy,
            mask_convention=mask_convention,
            dropout_rng=dropout_rng,
            qkv_layout=qkv_layout,
            head_layout=head_layout,
            scale_source=scale_source,
            use_cache=use_cache,
            output_attentions=output_attentions,
            rope_parameters=rope_parameters,
            position_id_policy=position_id_policy,
            score_softcap=score_softcap,
            final_logit_softcap=final_logit_softcap,
        ),
        query_key=query_key,
        key_key=key_key,
        value_key=value_key,
        output_key=output_key,
        mask_key=mask_key,
        inverse_permutation_key=inverse_permutation_key,
        query_block_size_key=query_block_size_key,
        dropout_p=dropout_p,
        is_causal=is_causal,
        scale=scale,
        enable_gqa=enable_gqa,
    )


def transformers_model_identity(
    model: torch.nn.Module,
    *,
    transformers_version: str,
    model_config_hash: str,
    source_revision: str | None,
    dtype_policy: Mapping[str, Any],
    tokenizer_identity: Mapping[str, Any],
    adapter_rules: Mapping[str, Any],
) -> TransformersModelIdentity:
    """Return explicit identity for a Transformers model adapter.

    Raises:
        AdmissionError: If an identity field is missing.
    """
    for key, value in (
        ("transformers_version", transformers_version),
        ("model_config_hash", model_config_hash),
    ):
        if not isinstance(value, str) or not value:
            message = f"{key} must be a non-empty string"
            raise AdmissionError(message)

    if source_revision is not None and not source_revision:
        message = "source_revision must be None or a non-empty string"
        raise AdmissionError(message)

    for key, value in (
        ("dtype_policy", dtype_policy),
        ("tokenizer_identity", tokenizer_identity),
        ("adapter_rules", adapter_rules),
    ):
        if not isinstance(value, Mapping) or not value:
            message = f"{key} must be a non-empty mapping"
            raise AdmissionError(message)

    return TransformersModelIdentity(
        transformers_version=transformers_version,
        model_config_hash=model_config_hash,
        module=module_identity(model),
        source_revision=source_revision,
        dtype_policy=dtype_policy,
        tokenizer_identity=tokenizer_identity,
        adapter_rules=adapter_rules,
    )


def transformers_attention_axis(
    frontends: Sequence[str],
    *,
    policy: TransformersAttentionPolicy,
) -> AxisDescriptor:
    """Return an attention frontend axis for Transformers models.

    Raises:
        AdmissionError: If a frontend name is unsupported.
    """
    unsupported = tuple(
        frontend
        for frontend in frontends
        if frontend not in TRANSFORMERS_ATTENTION_FRONTENDS
    )

    if unsupported:
        message = f"unsupported Transformers attention frontends: {unsupported}"
        raise AdmissionError(message)

    return AxisDescriptor(
        name="transformers_attention_frontend",
        settings_keys=("attention.frontend",),
        allowed_values=tuple(frontends),
        optional_settings_keys=(
            "attention.sdpa_priority_list",
            "attention.custom_kernel_id",
            "attention.mask_formatter_id",
            "output_attentions",
            "module_mode",
            "dropout_p",
        ),
        adapter_id="vptune.transformers",
        adapter_version=PACKAGE_VERSION,
        admission_rule=lambda candidate: admit_transformers_attention(
            candidate,
            policy=policy,
        ),
        identity=policy.signature(),
    )


def transformers_manifest_attention_axis() -> AxisDescriptor:
    """Return the Transformers attention descriptor for the package manifest."""
    return transformers_attention_axis(
        TRANSFORMERS_ATTENTION_FRONTENDS,
        policy=_transformers_manifest_attention_policy(),
    )


def _attention_space_axis(space: TransformersAttentionSpace) -> AxisDescriptor:
    optional_keys = (
        "module_mode",
        "dropout_p",
        "attention.sdpa_priority_list",
        "attention.custom_kernel_id",
        "attention.mask_formatter_id",
        "output_attentions",
    )

    return AxisDescriptor(
        name="transformers_attention_frontend",
        settings_keys=("attention.frontend", "attention.sdpa_kernel"),
        allowed_values=_attention_axis_values(space),
        optional_settings_keys=optional_keys,
        adapter_id="vptune.transformers",
        adapter_version=PACKAGE_VERSION,
        admission_rule=lambda candidate: admit_transformers_attention(
            candidate,
            policy=space.policy,
        ),
        identity={
            "policy": space.policy.signature(),
            "frontends": space.frontends,
            "sdpa_kernel": space.sdpa_kernel,
        },
    )


def _attention_axis_values(
    space: TransformersAttentionSpace,
) -> tuple[Mapping[str, Any], ...]:
    return tuple(
        {
            "attention.frontend": frontend,
            "attention.sdpa_kernel": (
                space.sdpa_kernel if frontend in SDPA_ATTENTION_FRONTENDS else None
            ),
        }
        for frontend in space.frontends
    )


def _normalize_public_attention_frontends(
    frontends: Sequence[str],
) -> tuple[str, ...]:
    if not frontends:
        message = "attention frontends must be nonempty"
        raise AdmissionError(message)

    normalized = []

    for frontend in frontends:
        if not isinstance(frontend, str) or not frontend:
            message = "attention frontend must be a nonempty string"
            raise AdmissionError(message)

        value = PUBLIC_ATTENTION_FRONTEND_ALIASES.get(frontend, frontend)

        if value not in TRANSFORMERS_ATTENTION_FRONTENDS:
            message = f"unsupported Transformers attention frontend: {frontend}"
            raise AdmissionError(message)

        if value in normalized:
            message = f"duplicate Transformers attention frontend: {frontend}"
            raise AdmissionError(message)

        normalized.append(value)

    return tuple(normalized)


def _require_sdpa_priority_list(priority_list: Sequence[str]) -> None:
    if (
        not isinstance(priority_list, Sequence)
        or isinstance(priority_list, str)
        or not priority_list
    ):
        message = "attention.sdpa_priority_list must be a nonempty sequence"
        raise AdmissionError(message)

    for kernel in priority_list:
        if not isinstance(kernel, str):
            message = "attention.sdpa_priority_list entries must be strings"
            raise AdmissionError(message)

        if kernel not in SDPA_KERNELS or kernel == "priority_list":
            message = f"invalid SDPA priority-list entry: {kernel}"
            raise AdmissionError(message)


def _public_attention_policy() -> TransformersAttentionPolicy:
    return TransformersAttentionPolicy(
        model_config_hash="public_attention_space",
        use_cache=False,
        softcap={},
        mask_semantics="boolean_keep_mask",
        causal_policy="causal",
        backend_numeric_policy={"backend": "sdpa"},
        determinism={"deterministic": True},
        padding_limit=1,
        forced_kernel_available=True,
    )


def _transformers_manifest_attention_policy() -> TransformersAttentionPolicy:
    return TransformersAttentionPolicy(
        model_config_hash="manifest",
        use_cache=False,
        softcap={"logit_softcap": 30.0},
        mask_semantics="boolean_keep_mask",
        causal_policy="causal",
        backend_numeric_policy={"backend": "sdpa"},
        determinism={"deterministic": True},
        padding_limit=1,
        forced_kernel_available=True,
    )


def admit_transformers_attention(
    candidate: Candidate,
    *,
    policy: TransformersAttentionPolicy,
) -> tuple[bool, str | None]:
    """Return whether a Transformers attention candidate is admitted."""
    attention_frontend = candidate.settings.get("attention.frontend")

    if not isinstance(attention_frontend, str):
        return False, "attention.frontend must be a string"

    if attention_frontend not in TRANSFORMERS_ATTENTION_FRONTENDS:
        return (
            False,
            f"unsupported Transformers attention frontend: {attention_frontend}",
        )

    error = _attention_error(candidate, policy, attention_frontend)

    if error is None:
        return True, None

    return False, error


def _attention_error(
    candidate: Candidate,
    policy: TransformersAttentionPolicy,
    attention_frontend: str,
) -> str | None:
    error = _core_attention_setting_error(candidate.settings)

    if error is None:
        error = _sdpa_kernel_error(candidate.settings, policy, attention_frontend)

    if error is None and attention_frontend in FLASH_ATTENTION_FRONTENDS:
        error = _flash_attention_frontend_error(candidate, policy, attention_frontend)

    if (
        error is None
        and candidate.settings.get("output_attentions") is True
        and attention_frontend not in EAGER_ATTENTION_FRONTENDS
    ):
        error = "output_attentions requires eager attention"

    if error is None:
        error = _eval_dropout_error(candidate.settings)

    if error is None:
        error = _registered_attention_error(candidate.settings, attention_frontend)

    if error is None:
        error = _policy_error(policy)

    return error


def _core_attention_setting_error(settings: Mapping[str, Any]) -> str | None:
    for key in (
        "attention.partition",
        "attention.padding",
        "chunk.sequence_position_block_size",
    ):
        if key in settings:
            return f"{key} is owned by the core attention executor"

    return None


def _flash_attention_frontend_error(
    candidate: Candidate,
    policy: TransformersAttentionPolicy,
    attention_frontend: str,
) -> str | None:
    if policy.padding_limit is None or policy.padding_limit <= 0:
        return f"{attention_frontend} requires a positive padding limit"

    dtype_error = _flash_attention_dtype_error(
        candidate.settings,
        attention_frontend,
    )

    if dtype_error is not None:
        return dtype_error

    return _forced_kernel_error(policy)


def _sdpa_kernel_error(
    settings: Mapping[str, Any],
    policy: TransformersAttentionPolicy,
    attention_frontend: str,
) -> str | None:
    if attention_frontend not in SDPA_ATTENTION_FRONTENDS:
        return _unexpected_sdpa_kernel_error(settings)

    value_error = _sdpa_kernel_value_error(settings)

    if value_error is not None:
        return value_error

    return _admitted_sdpa_kernel_error(
        settings,
        policy,
        settings["attention.sdpa_kernel"],
    )


def _unexpected_sdpa_kernel_error(settings: Mapping[str, Any]) -> str | None:
    if settings.get("attention.sdpa_kernel") is not None:
        return "attention.sdpa_kernel applies only to SDPA frontends"

    return None


def _sdpa_kernel_value_error(settings: Mapping[str, Any]) -> str | None:
    sdpa_kernel = settings.get("attention.sdpa_kernel")

    if not isinstance(sdpa_kernel, str):
        return "attention.sdpa_kernel must be a string"

    if sdpa_kernel not in SDPA_KERNELS:
        return f"unsupported SDPA kernel: {sdpa_kernel}"

    return None


def _admitted_sdpa_kernel_error(
    settings: Mapping[str, Any],
    policy: TransformersAttentionPolicy,
    sdpa_kernel: object,
) -> str | None:
    if sdpa_kernel == "priority_list":
        return _sdpa_priority_list_error(settings, policy)

    if sdpa_kernel == "math":
        return None

    return _non_math_sdpa_kernel_error(settings, policy, sdpa_kernel)


def _sdpa_priority_list_error(
    settings: Mapping[str, Any],
    policy: TransformersAttentionPolicy,
) -> str | None:
    priority_list = settings.get("attention.sdpa_priority_list")

    if (
        not isinstance(priority_list, Sequence)
        or isinstance(priority_list, str)
        or not priority_list
    ):
        return "attention.sdpa_priority_list must be a non-empty sequence"

    for kernel in priority_list:
        if not isinstance(kernel, str):
            return "attention.sdpa_priority_list entries must be strings"

        if kernel not in SDPA_KERNELS or kernel == "priority_list":
            return f"invalid SDPA priority-list entry: {kernel}"

    if "flash_attention" in priority_list:
        dtype_error = _flash_attention_dtype_error(
            settings,
            "attention.sdpa_kernel=priority_list",
        )

        if dtype_error is not None:
            return dtype_error

    if any(kernel != "math" for kernel in priority_list):
        return _forced_kernel_error(policy)

    return None


def _non_math_sdpa_kernel_error(
    settings: Mapping[str, Any],
    policy: TransformersAttentionPolicy,
    sdpa_kernel: object,
) -> str | None:
    if sdpa_kernel == "flash_attention":
        dtype_error = _flash_attention_dtype_error(
            settings,
            "attention.sdpa_kernel=flash_attention",
        )

        if dtype_error is not None:
            return dtype_error

    if sdpa_kernel not in NON_MATH_SDPA_KERNELS:
        return None

    return _forced_kernel_error(policy)


def _forced_kernel_error(policy: TransformersAttentionPolicy) -> str | None:
    if not policy.forced_kernel_available:
        return (
            policy.forced_kernel_failure_reason
            or "forced attention kernel is unavailable"
        )

    return None


def _flash_attention_dtype_error(
    settings: Mapping[str, Any],
    attention_frontend: str,
) -> str | None:
    dtype = settings.get("dtype.model_compute", settings.get("dtype.parameter_storage"))

    if dtype not in FLASH_ATTENTION_DTYPES:
        return f"{attention_frontend} requires float16 or bfloat16 effective dtype"

    return None


def _eval_dropout_error(settings: Mapping[str, Any]) -> str | None:
    module_mode = settings.get("module_mode")

    if module_mode not in {"train", "eval"}:
        return "attention rows require module_mode train or eval"

    if module_mode != "eval":
        return None

    dropout_p = settings.get("dropout_p")

    if dropout_p is None:
        return "eval attention references require recorded dropout_p"

    if not isinstance(dropout_p, int | float):
        return "dropout_p must be numeric"

    if math.isclose(dropout_p, 0.0, rel_tol=0.0, abs_tol=0.0):
        return None

    return "eval attention references require dropout_p=0.0"


def _registered_attention_error(
    settings: Mapping[str, Any],
    attention_frontend: str,
) -> str | None:
    if attention_frontend != "registered_transformers_attention":
        return None

    return _registered_attention_ids_error(settings)


def _registered_attention_ids_error(settings: Mapping[str, Any]) -> str | None:
    error = _first_attention_error((
        _required_string_setting(settings, "attention.custom_kernel_id"),
        _required_string_setting(settings, "attention.mask_formatter_id"),
    ))

    if error is not None:
        return error

    if (
        settings["attention.custom_kernel_id"]
        != settings["attention.mask_formatter_id"]
    ):
        return (
            "registered Transformers attention requires matching attention and mask ids"
        )

    return None


def _first_attention_error(errors: Sequence[str | None]) -> str | None:
    for error in errors:
        if error is not None:
            return error

    return None


def _required_string_setting(settings: Mapping[str, Any], key: str) -> str | None:
    value = settings.get(key)

    if not isinstance(value, str) or not value:
        return f"{key} must be a non-empty string"

    return None


def _required_bool_setting(settings: Mapping[str, Any], key: str) -> str | None:
    if not isinstance(settings.get(key), bool):
        return f"{key} must be a bool"

    return None


def _policy_error(policy: TransformersAttentionPolicy) -> str | None:
    if not policy.mask_semantics:
        return "mask semantics must be recorded"

    if not policy.causal_policy:
        return "causal policy must be recorded"

    if not policy.backend_numeric_policy:
        return "backend numeric policy must be recorded"

    if not policy.determinism:
        return "determinism policy must be recorded"

    return None
