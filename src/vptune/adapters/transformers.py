"""Transformers adapter admission helpers."""

import dataclasses
import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import torch

from vptune.candidates import AxisDescriptor
from vptune.checks import tree_error_measurements, validate_thresholds
from vptune.data import PACKAGE_VERSION, Candidate
from vptune.errors import AdmissionError
from vptune.identities import module_identity
from vptune.tensor_tree import TensorTree

EAGER_ATTENTION_IMPLS = (
    "eager",
    "transformers_eager",
    "patched_eager",
)
SDPA_ATTENTION_IMPLS = (
    "sdpa_math",
    "sdpa_flash",
    "sdpa_memory_efficient",
    "transformers_sdpa",
)
FLASH_ATTENTION_IMPLS = (
    "sdpa_flash",
    "transformers_flash_attention_2",
)
PACKED_ATTENTION_IMPLS = ("packed_target_rows",)
BLOCKWISE_ATTENTION_IMPLS = ("blockwise_second_derivative",)
TRANSFORMERS_ATTENTION_IMPLS = (
    *EAGER_ATTENTION_IMPLS,
    *SDPA_ATTENTION_IMPLS,
    "transformers_flash_attention_2",
    *PACKED_ATTENTION_IMPLS,
    *BLOCKWISE_ATTENTION_IMPLS,
)
FLASH_ATTENTION_DTYPES = ("float16", "bfloat16")


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
    implementations: Sequence[str],
    *,
    policy: TransformersAttentionPolicy,
) -> AxisDescriptor:
    """Return an attention-implementation axis for Transformers models.

    Raises:
        AdmissionError: If an implementation name is unsupported.
    """
    unsupported = tuple(
        implementation
        for implementation in implementations
        if implementation not in TRANSFORMERS_ATTENTION_IMPLS
    )

    if unsupported:
        message = f"unsupported Transformers attention implementations: {unsupported}"
        raise AdmissionError(message)

    return AxisDescriptor(
        name="transformers_attention",
        settings_keys=("attention_impl",),
        allowed_values=tuple(implementations),
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
    attention_impl = candidate.settings.get("attention_impl")

    if not isinstance(attention_impl, str):
        return False, "attention_impl must be a string"

    if attention_impl not in TRANSFORMERS_ATTENTION_IMPLS:
        return (
            False,
            f"unsupported Transformers attention implementation: {attention_impl}",
        )

    error = _attention_error(candidate, policy, attention_impl)

    if error is None:
        return True, None

    return False, error


def admit_transformers_cache(
    candidate: Candidate,
    *,
    policy: TransformersAttentionPolicy,
) -> tuple[bool, str | None]:
    """Return whether a Transformers cache candidate is admitted."""
    use_cache = candidate.settings.get("use_cache")

    if not isinstance(use_cache, bool):
        return False, "use_cache must be a bool"

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
    attention_impl: str,
) -> str | None:
    error = None

    if attention_impl == "transformers_flash_attention_2":
        error = _flash_attention2_error(candidate, policy)
    elif attention_impl == "sdpa_flash":
        error = _sdpa_flash_error(candidate, policy)

    if (
        error is None
        and candidate.settings.get("output_attentions") is True
        and attention_impl not in EAGER_ATTENTION_IMPLS
    ):
        error = "output_attentions requires eager attention"

    if error is None:
        error = _eval_dropout_error(candidate.settings)

    if error is None:
        error = _gqa_error(candidate.settings)

    if error is None:
        error = _patched_attention_error(candidate.settings, attention_impl)

    if error is None:
        error = _packed_attention_error(candidate.settings, attention_impl)

    if error is None:
        error = _blockwise_attention_error(candidate.settings, attention_impl)

    if error is None:
        error = _policy_error(policy)

    return error


def _flash_attention2_error(
    candidate: Candidate,
    policy: TransformersAttentionPolicy,
) -> str | None:
    if policy.padding_limit is None or policy.padding_limit <= 0:
        return "FlashAttention-2 requires a positive padding limit"

    dtype_error = _flash_attention_dtype_error(
        candidate.settings,
        "transformers_flash_attention_2",
    )

    if dtype_error is not None:
        return dtype_error

    return _forced_kernel_error(policy)


def _sdpa_flash_error(
    candidate: Candidate,
    policy: TransformersAttentionPolicy,
) -> str | None:
    dtype_error = _flash_attention_dtype_error(candidate.settings, "sdpa_flash")

    if dtype_error is not None:
        return dtype_error

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
    attention_impl: str,
) -> str | None:
    dtype = settings.get("model_dtype")

    if dtype not in FLASH_ATTENTION_DTYPES:
        return f"{attention_impl} requires float16 or bfloat16 model_dtype"

    return None


def _eval_dropout_error(settings: Mapping[str, Any]) -> str | None:
    if settings.get("module_mode") != "eval":
        return None

    dropout_p = settings.get("dropout_p")

    if dropout_p is None:
        return None

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
    attention_impl: str,
) -> str | None:
    if attention_impl != "patched_eager":
        return None

    return _semantic_attention_error(
        settings,
        id_key="patched_attention_id",
        semantics_key="patched_attention_semantics",
    )


def _packed_attention_error(
    settings: Mapping[str, Any],
    attention_impl: str,
) -> str | None:
    if attention_impl != "packed_target_rows":
        return None

    return _first_attention_error((
        _semantic_attention_error(
            settings,
            id_key="packed_attention_id",
            semantics_key="packed_attention_semantics",
        ),
        _required_string_setting(settings, "packed_target_row_axis"),
        _required_positive_int_setting(settings, "packed_target_row_count"),
    ))


def _blockwise_attention_error(
    settings: Mapping[str, Any],
    attention_impl: str,
) -> str | None:
    if attention_impl != "blockwise_second_derivative":
        return None

    return _first_attention_error((
        _semantic_attention_error(
            settings,
            id_key="blockwise_attention_id",
            semantics_key="blockwise_attention_semantics",
        ),
        _required_positive_int_setting(settings, "attention_block_size"),
        _required_bool_setting(settings, "blockwise_preserves_softcap"),
        _required_bool_setting(settings, "blockwise_preserves_mask"),
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
