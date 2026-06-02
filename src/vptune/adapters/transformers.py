"""Transformers adapter admission helpers."""

import dataclasses
import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Protocol

import torch

from vptune.candidates import AxisDescriptor
from vptune.checks import tree_error_measurements, validate_thresholds
from vptune.data import PACKAGE_VERSION, Candidate
from vptune.errors import AdmissionError
from vptune.identities import module_identity
from vptune.tensor_tree import TensorTree

EAGER_ATTENTION_FRONTENDS = (
    "transformers_eager",
    "patched_eager",
    "paged|eager",
)
SDPA_ATTENTION_FRONTENDS = (
    "transformers_sdpa",
    "paged|sdpa",
    "pytorch_sdpa_direct",
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
PACKED_ATTENTION_FRONTENDS = ("packed_exact",)
BLOCKWISE_ATTENTION_FRONTENDS = ("blockwise_exact",)
TRANSFORMERS_ATTENTION_FRONTENDS = (
    *EAGER_ATTENTION_FRONTENDS,
    *SDPA_ATTENTION_FRONTENDS,
    *FLASH_ATTENTION_FRONTENDS,
    *CUSTOM_ATTENTION_FRONTENDS,
    *PACKED_ATTENTION_FRONTENDS,
    *BLOCKWISE_ATTENTION_FRONTENDS,
)
SDPA_KERNELS = (
    "math",
    "flash_attention",
    "efficient_attention",
    "cudnn_attention",
    "overrideable",
    "priority_list",
)
NON_MATH_SDPA_KERNELS = tuple(kernel for kernel in SDPA_KERNELS if kernel != "math")
FLASH_ATTENTION_DTYPES = ("float16", "bfloat16")
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


def load_transformers_model(
    model_cls: TransformersModelLoader,
    *,
    model_name_or_path: str,
    revision: str,
    torch_dtype: torch.dtype,
    attention_frontend: str,
    attention_custom_kernel_id: str | None = None,
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
            attention_custom_kernel_id=attention_custom_kernel_id,
        ),
        use_cache=use_cache,
    )


def transformers_attn_implementation(
    attention_frontend: str,
    *,
    attention_custom_kernel_id: str | None = None,
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
            "attention.sdpa_kernel",
            "attention.sdpa_priority_list",
            "attention.custom_kernel_id",
            "attention.mask_formatter_id",
            "model_dtype",
            "compute_dtype",
            "output_attentions",
            "module_mode",
            "dropout_p",
            "enable_gqa",
            "query_heads",
            "key_heads",
            "value_heads",
            "patched_attention_id",
            "patched_attention_semantics",
            "packed_attention_id",
            "packed_attention_semantics",
            "packed_target_row_axis",
            "packed_target_row_count",
            "blockwise_attention_id",
            "blockwise_attention_semantics",
            "attention_block_size",
            "blockwise_preserves_softcap",
            "blockwise_preserves_mask",
        ),
        adapter_id="vptune.transformers",
        adapter_version=PACKAGE_VERSION,
        admission_rule=lambda candidate: admit_transformers_attention(
            candidate,
            policy=policy,
        ),
        identity=policy.signature(),
    )


def transformers_cache_axis(
    *,
    policy: TransformersAttentionPolicy,
) -> AxisDescriptor:
    """Return a cache-flag axis for Transformers models."""
    return AxisDescriptor(
        name="transformers_cache",
        settings_keys=("use_cache",),
        allowed_values=(True, False),
        adapter_id="vptune.transformers",
        adapter_version=PACKAGE_VERSION,
        admission_rule=lambda candidate: admit_transformers_cache(
            candidate,
            policy=policy,
        ),
        identity=policy.signature(),
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


def admit_transformers_cache(
    candidate: Candidate,
    *,
    policy: TransformersAttentionPolicy,
) -> tuple[bool, str | None]:
    """Return whether a Transformers cache candidate is admitted."""
    error = _required_bool_setting(candidate.settings, "use_cache")

    if error is not None:
        return False, error

    use_cache = candidate.settings["use_cache"]
    if use_cache != policy.use_cache:
        return (
            False,
            f"use_cache={use_cache} differs from policy use_cache={policy.use_cache}",
        )

    return True, None


def check_patched_attention_reference(
    reference: Callable[..., TensorTree],
    patched: Callable[..., TensorTree],
    args: Sequence[Any],
    *,
    thresholds: Mapping[str, float],
) -> dict[str, float]:
    """Validate patched attention output against a reference output.

    Returns:
        Error measurements for the patched output.
    """
    reference_output = reference(*args)
    patched_output = patched(*args)
    measurements = tree_error_measurements(patched_output, reference_output)
    validate_thresholds(measurements, thresholds)

    return measurements


def check_patched_attention_vjp_reference(
    reference: Callable[..., torch.Tensor],
    patched: Callable[..., torch.Tensor],
    args: Sequence[Any],
    cotangent: torch.Tensor,
    differentiable_arg_indices: Sequence[int],
    *,
    thresholds: Mapping[str, float],
) -> dict[str, float]:
    """Validate patched attention VJP against a reference VJP.

    Returns:
        Error measurements for the patched VJP.
    """
    indices = _validate_differentiable_arg_indices(
        differentiable_arg_indices,
        len(args),
    )
    reference_vjp = _attention_vjp(reference, args, cotangent, indices)
    patched_vjp = _attention_vjp(patched, args, cotangent, indices)
    measurements = tree_error_measurements(patched_vjp, reference_vjp)
    validate_thresholds(measurements, thresholds)

    return measurements


def _attention_error(
    candidate: Candidate,
    policy: TransformersAttentionPolicy,
    attention_frontend: str,
) -> str | None:
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
        error = _gqa_error(candidate.settings)

    if error is None:
        error = _patched_attention_error(candidate.settings, attention_frontend)

    if error is None:
        error = _packed_attention_error(candidate.settings, policy, attention_frontend)

    if error is None:
        error = _blockwise_attention_error(candidate.settings, attention_frontend)

    if error is None:
        error = _registered_attention_error(candidate.settings, attention_frontend)

    if error is None:
        error = _policy_error(policy)

    return error


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
    dtype = settings.get("compute_dtype", settings.get("model_dtype"))

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


def _gqa_error(settings: Mapping[str, Any]) -> str | None:
    if settings.get("enable_gqa") is not True:
        return None

    query_heads = settings.get("query_heads")
    key_heads = settings.get("key_heads")
    value_heads = settings.get("value_heads")

    if (
        not isinstance(query_heads, int)
        or not isinstance(key_heads, int)
        or not isinstance(value_heads, int)
    ):
        return "GQA admission requires integer head counts"

    if key_heads <= 0 or value_heads <= 0:
        return "GQA key and value head counts must be positive"

    if key_heads != value_heads:
        return "GQA requires key_heads equal to value_heads"

    if query_heads % key_heads != 0:
        return "GQA requires query_heads divisible by key_heads"

    return None


def _patched_attention_error(
    settings: Mapping[str, Any],
    attention_frontend: str,
) -> str | None:
    if attention_frontend != "patched_eager":
        return None

    return _semantic_attention_error(
        settings,
        id_key="patched_attention_id",
        semantics_key="patched_attention_semantics",
    )


def _packed_attention_error(
    settings: Mapping[str, Any],
    policy: TransformersAttentionPolicy,
    attention_frontend: str,
) -> str | None:
    if attention_frontend != "packed_exact":
        return None

    return _first_attention_error((
        _semantic_attention_error(
            settings,
            id_key="packed_attention_id",
            semantics_key="packed_attention_semantics",
        ),
        _required_string_setting(settings, "packed_target_row_axis"),
        _required_positive_int_setting(settings, "packed_target_row_count"),
        _required_true_setting(settings, "packed_preserves_softcap"),
        _required_true_setting(settings, "packed_preserves_mask"),
        _required_string_match(
            settings,
            "packed_mask_semantics",
            policy.mask_semantics,
        ),
        _required_string_match(
            settings,
            "packed_causal_policy",
            policy.causal_policy,
        ),
    ))


def _blockwise_attention_error(
    settings: Mapping[str, Any],
    attention_frontend: str,
) -> str | None:
    if attention_frontend != "blockwise_exact":
        return None

    return _first_attention_error((
        _semantic_attention_error(
            settings,
            id_key="blockwise_attention_id",
            semantics_key="blockwise_attention_semantics",
        ),
        _required_positive_int_setting(settings, "attention_block_size"),
        _required_true_setting(settings, "blockwise_preserves_softcap"),
        _required_true_setting(settings, "blockwise_preserves_mask"),
    ))


def _registered_attention_error(
    settings: Mapping[str, Any],
    attention_frontend: str,
) -> str | None:
    if attention_frontend != "registered_transformers_attention":
        return None

    return _first_attention_error((
        _required_string_setting(settings, "attention.custom_kernel_id"),
        _required_string_setting(settings, "attention.mask_formatter_id"),
    ))


def _semantic_attention_error(
    settings: Mapping[str, Any],
    *,
    id_key: str,
    semantics_key: str,
) -> str | None:
    return _first_attention_error((
        _required_string_setting(settings, id_key),
        _required_mapping_setting(settings, semantics_key),
    ))


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


def _required_mapping_setting(settings: Mapping[str, Any], key: str) -> str | None:
    value = settings.get(key)

    if not isinstance(value, Mapping) or not value:
        return f"{key} must be a non-empty mapping"

    return None


def _required_positive_int_setting(
    settings: Mapping[str, Any],
    key: str,
) -> str | None:
    value = settings.get(key)

    if not isinstance(value, int) or value < 1:
        return f"{key} must be a positive integer"

    return None


def _required_bool_setting(settings: Mapping[str, Any], key: str) -> str | None:
    if not isinstance(settings.get(key), bool):
        return f"{key} must be a bool"

    return None


def _required_true_setting(settings: Mapping[str, Any], key: str) -> str | None:
    if settings.get(key) is not True:
        return f"{key} must be true"

    return None


def _required_string_match(
    settings: Mapping[str, Any],
    key: str,
    expected: str,
) -> str | None:
    value = settings.get(key)

    if value != expected:
        return f"{key} must match policy value {expected}"

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


def _validate_differentiable_arg_indices(
    indices: Sequence[int],
    arg_count: int,
) -> tuple[int, ...]:
    if not indices:
        message = "differentiable_arg_indices must be non-empty"
        raise AdmissionError(message)

    unique = tuple(dict.fromkeys(indices))

    if len(unique) != len(indices):
        message = "differentiable_arg_indices must not contain duplicates"
        raise AdmissionError(message)

    for index in unique:
        if index < 0 or index >= arg_count:
            message = "differentiable_arg_indices contains an out-of-range index"
            raise AdmissionError(message)

    return unique


def _attention_vjp(
    function: Callable[..., torch.Tensor],
    args: Sequence[Any],
    cotangent: torch.Tensor,
    indices: Sequence[int],
) -> tuple[torch.Tensor, ...]:
    active_args = []
    call_args = list(args)

    for index in indices:
        argument = args[index]

        if not isinstance(argument, torch.Tensor):
            message = "differentiable attention arguments must be tensors"
            raise AdmissionError(message)

        if not argument.is_floating_point():
            message = "differentiable attention arguments must be floating tensors"
            raise AdmissionError(message)

        active = argument.detach().clone().requires_grad_(True)
        active_args.append(active)
        call_args[index] = active

    output = function(*call_args)

    if not isinstance(output, torch.Tensor):
        message = "attention VJP reference functions must return a tensor"
        raise AdmissionError(message)

    scalar = (output * cotangent).sum()
    gradients = torch.autograd.grad(
        scalar,
        tuple(active_args),
        allow_unused=True,
    )

    return tuple(
        torch.zeros_like(argument) if gradient is None else gradient.detach()
        for argument, gradient in zip(active_args, gradients, strict=True)
    )
