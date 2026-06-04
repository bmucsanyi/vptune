"""Core attention execution."""

import contextlib
import dataclasses
import math
from collections.abc import Callable, Iterator, Mapping, Sequence
from typing import Any, Protocol

import torch
from torch.nn import functional
from torch.nn.attention import SDPBackend, sdpa_kernel

from vptune.checks import tree_error_measurements, validate_thresholds
from vptune.errors import AdmissionError
from vptune.tensor_tree import TensorTree

SDPA_BACKENDS = {
    "math": SDPBackend.MATH,
    "flash_attention": SDPBackend.FLASH_ATTENTION,
    "efficient_attention": SDPBackend.EFFICIENT_ATTENTION,
    "cudnn_attention": SDPBackend.CUDNN_ATTENTION,
    "overrideable": SDPBackend.OVERRIDEABLE,
}
CORE_ATTENTION_FRONTENDS = (
    "pytorch_sdpa_direct",
    "patched_eager",
    "packed_exact",
    "blockwise_exact",
)
ATTENTION_PARTITIONS = (
    "full",
    "packed_tokens",
    "blockwise_queries",
    "segmented_forward_ad",
)
ATTENTION_PADDING = ("dense_padded", "unpadded_packed")
MIN_QUERY_MASK_DIMS = 2


@dataclasses.dataclass(frozen=True, slots=True)
class AttentionSemantics:
    """Declared semantics for one attention location."""

    causal_policy: str
    sliding_window_policy: str
    padding_policy: str
    mask_convention: str
    dropout_rng: Mapping[str, Any]
    qkv_layout: str
    head_layout: str
    scale_source: str
    use_cache: bool
    output_attentions: bool
    rope_parameters: Mapping[str, Any]
    position_id_policy: Mapping[str, Any]
    score_softcap: float | None
    final_logit_softcap: float | None

    def signature(self) -> dict[str, Any]:
        """Return serializable attention semantics."""
        return {
            "causal_policy": self.causal_policy,
            "sliding_window_policy": self.sliding_window_policy,
            "padding_policy": self.padding_policy,
            "mask_convention": self.mask_convention,
            "dropout_rng": dict(self.dropout_rng),
            "qkv_layout": self.qkv_layout,
            "head_layout": self.head_layout,
            "scale_source": self.scale_source,
            "use_cache": self.use_cache,
            "output_attentions": self.output_attentions,
            "rope_parameters": dict(self.rope_parameters),
            "position_id_policy": dict(self.position_id_policy),
            "score_softcap": self.score_softcap,
            "final_logit_softcap": self.final_logit_softcap,
        }


@dataclasses.dataclass(frozen=True, slots=True)
class AttentionInputs:
    """Inputs for one attention call."""

    query: torch.Tensor
    key: torch.Tensor
    value: torch.Tensor
    attn_mask: torch.Tensor | None
    dropout_p: float
    is_causal: bool
    scale: float | None
    score_softcap: float | None
    enable_gqa: bool
    inverse_permutation: torch.Tensor | None
    query_block_size: int | None


@dataclasses.dataclass(frozen=True, slots=True)
class AttentionSettings:
    """Execution settings for core attention."""

    frontend: str
    sdpa_kernel: str | None
    sdpa_priority: tuple[str, ...]
    partition: str
    padding: str

    def signature(self) -> dict[str, Any]:
        """Return serializable settings identity."""
        return {
            "frontend": self.frontend,
            "sdpa_kernel": self.sdpa_kernel,
            "sdpa_priority": self.sdpa_priority,
            "partition": self.partition,
            "padding": self.padding,
        }


class AttentionLocation(Protocol):
    """Descriptor that binds model data to core attention inputs."""

    def inputs(self, batch: Mapping[str, Any]) -> AttentionInputs:
        """Read attention inputs from a batch."""

    def output(
        self, attention_output: torch.Tensor, _: Mapping[str, Any]
    ) -> TensorTree:
        """Write attention output into the declared output tree."""

    def signature(self) -> Mapping[str, Any]:
        """Return descriptor identity."""


@dataclasses.dataclass(frozen=True, slots=True)
class MappingAttentionLocation:
    """Attention descriptor backed by batch mapping keys."""

    semantics: AttentionSemantics
    query_key: str
    key_key: str
    value_key: str
    output_key: str
    mask_key: str | None
    inverse_permutation_key: str | None
    query_block_size_key: str | None
    dropout_p: float
    is_causal: bool
    scale: float | None
    enable_gqa: bool

    def inputs(self, batch: Mapping[str, Any]) -> AttentionInputs:
        """Read attention inputs from a batch.

        Returns:
            Attention inputs.
        """
        query = _tensor_from_batch(batch, self.query_key)
        key = _tensor_from_batch(batch, self.key_key)
        value = _tensor_from_batch(batch, self.value_key)
        mask = _tensor_from_batch(batch, self.mask_key) if self.mask_key else None
        inverse_permutation = (
            _tensor_from_batch(batch, self.inverse_permutation_key)
            if self.inverse_permutation_key
            else None
        )
        query_block_size = (
            _positive_int_from_batch(batch, self.query_block_size_key)
            if self.query_block_size_key
            else None
        )

        return AttentionInputs(
            query=query,
            key=key,
            value=value,
            attn_mask=mask,
            dropout_p=self.dropout_p,
            is_causal=self.is_causal,
            scale=self.scale,
            score_softcap=self.semantics.score_softcap,
            enable_gqa=self.enable_gqa,
            inverse_permutation=inverse_permutation,
            query_block_size=query_block_size,
        )

    def output(
        self, attention_output: torch.Tensor, _: Mapping[str, Any]
    ) -> TensorTree:
        """Write attention output into the declared output tree.

        Returns:
            Declared output tree.
        """
        return {self.output_key: attention_output}

    def signature(self) -> dict[str, Any]:
        """Return descriptor identity."""
        return {
            "semantics": self.semantics.signature(),
            "query_key": self.query_key,
            "key_key": self.key_key,
            "value_key": self.value_key,
            "output_key": self.output_key,
            "mask_key": self.mask_key,
            "inverse_permutation_key": self.inverse_permutation_key,
            "query_block_size_key": self.query_block_size_key,
            "dropout_p": self.dropout_p,
            "is_causal": self.is_causal,
            "scale": self.scale,
            "enable_gqa": self.enable_gqa,
        }


def execute_attention(
    location: AttentionLocation,
    batch: Mapping[str, Any],
    settings: AttentionSettings,
) -> TensorTree:
    """Run attention through a descriptor.

    Returns:
        Declared attention output tree.
    """
    output = run_attention(location.inputs(batch), settings)

    return location.output(output, batch)


def check_patched_attention_output_reference(
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


def run_attention(
    inputs: AttentionInputs,
    settings: AttentionSettings,
) -> torch.Tensor:
    """Run a core attention frontend.

    Returns:
        Attention output tensor.

    Raises:
        AdmissionError: If settings cannot execute.
    """
    _validate_attention_settings(settings)

    if settings.frontend == "pytorch_sdpa_direct":
        return _sdpa_attention(inputs, settings)

    if settings.frontend == "patched_eager":
        return exact_attention(inputs)

    if settings.frontend == "packed_exact":
        return _packed_exact_attention(inputs, settings)

    if settings.frontend == "blockwise_exact":
        return _blockwise_exact_attention(inputs, settings)

    message = f"unsupported core attention frontend: {settings.frontend}"
    raise AdmissionError(message)


def exact_attention(inputs: AttentionInputs) -> torch.Tensor:
    """Run exact eager scaled dot-product attention.

    Returns:
        Attention output tensor.
    """
    _validate_attention_inputs(inputs)
    key, value = _gqa_key_value(inputs)
    scores = _attention_scores(
        inputs.query,
        key,
        inputs.scale,
        inputs.score_softcap,
    )
    scores = scores + _attention_bias(
        inputs.query, key, inputs.attn_mask, inputs.is_causal
    )
    weights = torch.softmax(scores, dim=-1)

    if inputs.dropout_p > 0.0:
        weights = torch.dropout(weights, inputs.dropout_p, train=True)

    return weights @ value


def _sdpa_attention(
    inputs: AttentionInputs,
    settings: AttentionSettings,
) -> torch.Tensor:
    if inputs.score_softcap is not None:
        message = "pytorch_sdpa_direct does not support attention score softcap"
        raise AdmissionError(message)

    kernel, priority = _sdpa_kernel_selection(settings)

    with _sdpa_kernel_context(kernel, priority):
        return functional.scaled_dot_product_attention(
            inputs.query,
            inputs.key,
            inputs.value,
            attn_mask=inputs.attn_mask,
            dropout_p=inputs.dropout_p,
            is_causal=inputs.is_causal,
            scale=inputs.scale,
            enable_gqa=inputs.enable_gqa,
        )


def _packed_exact_attention(
    inputs: AttentionInputs,
    settings: AttentionSettings,
) -> torch.Tensor:
    if settings.partition != "packed_tokens":
        message = "packed_exact requires attention.partition=packed_tokens"
        raise AdmissionError(message)

    if settings.padding != "unpadded_packed":
        message = "packed_exact requires attention.padding=unpadded_packed"
        raise AdmissionError(message)

    if inputs.inverse_permutation is None:
        message = "packed_exact requires inverse token permutation"
        raise AdmissionError(message)

    output = exact_attention(inputs)

    return output.index_select(-2, inputs.inverse_permutation)


def _blockwise_exact_attention(
    inputs: AttentionInputs,
    settings: AttentionSettings,
) -> torch.Tensor:
    if settings.partition != "blockwise_queries":
        message = "blockwise_exact requires attention.partition=blockwise_queries"
        raise AdmissionError(message)

    if inputs.query_block_size is None:
        message = "blockwise_exact requires query block size"
        raise AdmissionError(message)

    blocks = []
    query_length = inputs.query.size(-2)

    for start in range(0, query_length, inputs.query_block_size):
        stop = min(start + inputs.query_block_size, query_length)
        blocks.append(_block_attention(inputs, start, stop))

    return torch.cat(blocks, dim=-2)


def _block_attention(
    inputs: AttentionInputs,
    start: int,
    stop: int,
) -> torch.Tensor:
    query = inputs.query.narrow(-2, start, stop - start)
    mask = _block_attention_mask(
        inputs.attn_mask,
        inputs.is_causal,
        start,
        stop,
        inputs.key.size(-2),
        inputs.query.device,
    )
    block_inputs = dataclasses.replace(
        inputs,
        query=query,
        attn_mask=mask,
        is_causal=False,
    )

    return exact_attention(block_inputs)


def _block_attention_mask(
    attn_mask: torch.Tensor | None,
    is_causal: bool,
    start: int,
    stop: int,
    key_length: int,
    device: torch.device,
) -> torch.Tensor | None:
    if attn_mask is not None and is_causal:
        message = "causal attention with explicit mask is not supported by SDPA"
        raise AdmissionError(message)

    mask = _query_block_mask(attn_mask, start, stop)

    if not is_causal:
        return mask

    causal = _causal_block_mask(start, stop, key_length, device)

    if mask is None:
        return causal

    if mask.dtype == torch.bool:
        return mask & causal

    return mask.masked_fill(causal.logical_not(), float("-inf"))


def _query_block_mask(
    attn_mask: torch.Tensor | None,
    start: int,
    stop: int,
) -> torch.Tensor | None:
    if attn_mask is None:
        return None

    if attn_mask.dim() >= MIN_QUERY_MASK_DIMS and attn_mask.size(-2) >= stop:
        return attn_mask.narrow(-2, start, stop - start)

    return attn_mask


def _causal_block_mask(
    start: int,
    stop: int,
    key_length: int,
    device: torch.device,
) -> torch.Tensor:
    query_positions = torch.arange(start, stop, device=device).unsqueeze(-1)
    key_positions = torch.arange(key_length, device=device).unsqueeze(0)

    return key_positions <= query_positions


def _attention_scores(
    query: torch.Tensor,
    key: torch.Tensor,
    scale: float | None,
    score_softcap: float | None,
) -> torch.Tensor:
    scale_factor = 1.0 / math.sqrt(query.size(-1)) if scale is None else scale
    scores = query @ key.transpose(-2, -1) * scale_factor

    return apply_softcap(scores, score_softcap, field_name="attention score")


def apply_final_logit_softcap(
    logits: torch.Tensor,
    softcap: float | None,
) -> torch.Tensor:
    """Apply the declared final-logit softcap.

    Returns:
        Softcapped logits, or unchanged logits when no softcap is declared.
    """
    return apply_softcap(logits, softcap, field_name="final logit")


def apply_softcap(
    values: torch.Tensor,
    softcap: float | None,
    *,
    field_name: str,
) -> torch.Tensor:
    """Apply a tanh softcap to tensor values.

    Returns:
        Softcapped values, or unchanged values when no softcap is declared.

    Raises:
        AdmissionError: If a declared softcap is nonpositive.
    """
    if softcap is None:
        return values

    if softcap <= 0.0:
        message = f"{field_name} softcap must be positive"
        raise AdmissionError(message)

    return torch.tanh(values / softcap) * softcap


def _attention_bias(
    query: torch.Tensor,
    key: torch.Tensor,
    attn_mask: torch.Tensor | None,
    is_causal: bool,
) -> torch.Tensor:
    if attn_mask is not None and is_causal:
        message = "causal attention with explicit mask is not supported by SDPA"
        raise AdmissionError(message)

    bias = torch.zeros(
        query.size(-2),
        key.size(-2),
        dtype=query.dtype,
        device=query.device,
    )

    if is_causal:
        causal_mask = torch.ones_like(bias, dtype=torch.bool).tril(diagonal=0)
        bias = bias.masked_fill(causal_mask.logical_not(), float("-inf"))

    if attn_mask is None:
        return bias

    if attn_mask.dtype == torch.bool:
        return bias.masked_fill(attn_mask.logical_not(), float("-inf"))

    return bias + attn_mask


def _gqa_key_value(inputs: AttentionInputs) -> tuple[torch.Tensor, torch.Tensor]:
    if not inputs.enable_gqa:
        return inputs.key, inputs.value

    query_heads = inputs.query.size(-3)
    key_heads = inputs.key.size(-3)
    value_heads = inputs.value.size(-3)

    if key_heads != value_heads:
        message = "GQA requires key and value head counts to match"
        raise AdmissionError(message)

    if key_heads <= 0 or query_heads % key_heads != 0:
        message = "GQA requires query heads divisible by key heads"
        raise AdmissionError(message)

    repeats = query_heads // key_heads

    return (
        inputs.key.repeat_interleave(repeats, -3),
        inputs.value.repeat_interleave(repeats, -3),
    )


def _sdpa_kernel_selection(
    settings: AttentionSettings,
) -> tuple[str | None, tuple[str, ...]]:
    if settings.sdpa_kernel is None:
        message = "pytorch_sdpa_direct requires attention.sdpa_kernel"
        raise AdmissionError(message)

    if settings.sdpa_kernel == "priority_list":
        if not settings.sdpa_priority:
            message = "attention.sdpa_kernel=priority_list requires backend order"
            raise AdmissionError(message)

        return None, settings.sdpa_priority

    if settings.sdpa_priority:
        message = "sdpa priority order applies only to priority_list"
        raise AdmissionError(message)

    return settings.sdpa_kernel, ()


@contextlib.contextmanager
def _sdpa_kernel_context(
    kernel: str | None,
    priority: tuple[str, ...],
) -> Iterator[None]:
    if priority:
        with sdpa_kernel(
            [_sdpa_backend(name) for name in priority],
            set_priority=True,
        ):
            yield

        return

    if kernel is None:
        message = "attention.sdpa_kernel is required"
        raise AdmissionError(message)

    with sdpa_kernel(_sdpa_backend(kernel), set_priority=False):
        yield


def _sdpa_backend(name: str) -> SDPBackend:
    backend = SDPA_BACKENDS.get(name)

    if backend is None:
        message = f"unknown SDPA backend: {name}"
        raise AdmissionError(message)

    return backend


def _validate_attention_settings(settings: AttentionSettings) -> None:
    if settings.frontend not in CORE_ATTENTION_FRONTENDS:
        message = f"unsupported core attention frontend: {settings.frontend}"
        raise AdmissionError(message)

    if settings.partition not in ATTENTION_PARTITIONS:
        message = f"unsupported attention partition: {settings.partition}"
        raise AdmissionError(message)

    if settings.padding not in ATTENTION_PADDING:
        message = f"unsupported attention padding: {settings.padding}"
        raise AdmissionError(message)

    if settings.frontend != "pytorch_sdpa_direct" and settings.sdpa_kernel is not None:
        message = "attention.sdpa_kernel applies only to pytorch_sdpa_direct"
        raise AdmissionError(message)

    _validate_attention_partition_settings(settings)


def _validate_attention_partition_settings(settings: AttentionSettings) -> None:
    if settings.partition == "segmented_forward_ad":
        message = "attention.partition=segmented_forward_ad has no core lowering"
        raise AdmissionError(message)

    if settings.frontend in {"pytorch_sdpa_direct", "patched_eager"}:
        if settings.partition != "full":
            message = f"{settings.frontend} requires attention.partition=full"
            raise AdmissionError(message)

        if settings.padding != "dense_padded":
            message = f"{settings.frontend} requires attention.padding=dense_padded"
            raise AdmissionError(message)

        return

    if settings.frontend == "blockwise_exact" and settings.padding != "dense_padded":
        message = "blockwise_exact requires attention.padding=dense_padded"
        raise AdmissionError(message)


def _validate_attention_inputs(inputs: AttentionInputs) -> None:
    if inputs.dropout_p < 0.0 or inputs.dropout_p > 1.0:
        message = "attention dropout probability must be between 0 and 1"
        raise AdmissionError(message)


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


def _tensor_from_batch(
    batch: Mapping[str, Any],
    key: str | None,
) -> torch.Tensor:
    if key is None:
        message = "batch tensor key is required"
        raise AdmissionError(message)

    value = batch[key]

    if not isinstance(value, torch.Tensor):
        message = f"batch value is not a tensor: {key}"
        raise AdmissionError(message)

    return value


def _positive_int_from_batch(
    batch: Mapping[str, Any],
    key: str | None,
) -> int:
    if key is None:
        message = "batch integer key is required"
        raise AdmissionError(message)

    value = batch[key]

    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        message = f"batch value is not a positive integer: {key}"
        raise AdmissionError(message)

    return value
