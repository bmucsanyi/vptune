"""Checkpoint execution helpers for adapter runtimes."""

from collections.abc import Callable, Sequence
from typing import Any

import torch
from torch.utils.checkpoint import checkpoint, noop_context_fn

from vptune.admission import admit_checkpoint
from vptune.data import Candidate, CandidateOperation, TensorTree
from vptune.errors import AdmissionError

ACTIVE_CHECKPOINT_SETTINGS = (
    "checkpoint_non_reentrant_by_layer",
    "checkpoint_selective",
)


def checkpoint_operation(
    candidate: Candidate,
    function: Callable[..., TensorTree],
    args: Sequence[Any],
    *,
    policy_key: str,
) -> CandidateOperation:
    """Return direct or checkpointed execution for an adapter operation.

    Raises:
        AdmissionError: If the candidate has missing or rejected checkpoint fields.
    """
    setting = _checkpoint_setting(candidate, policy_key)
    offload = _activation_offload(candidate)

    if setting == "none":
        return _with_activation_offload(
            candidate, _direct_operation(function, args), offload
        )

    if setting not in ACTIVE_CHECKPOINT_SETTINGS:
        message = f"checkpoint setting is unsupported: {setting}"
        raise AdmissionError(message)

    admit_checkpoint(candidate.settings)
    context_fn = _checkpoint_context_fn(candidate)

    def operation() -> TensorTree:
        return checkpoint(
            function,
            *args,
            use_reentrant=False,
            preserve_rng_state=_checkpoint_bool(
                candidate,
                "checkpoint.preserve_rng_state",
            ),
            determinism_check=candidate.settings["checkpoint.determinism_check"],
            context_fn=context_fn,
            early_stop=_checkpoint_bool(candidate, "checkpoint.early_stop"),
        )

    return _with_activation_offload(candidate, operation, offload)


def _checkpoint_setting(candidate: Candidate, policy_key: str) -> str:
    if policy_key not in candidate.settings:
        message = f"checkpoint policy key is missing: {policy_key}"
        raise AdmissionError(message)

    setting = candidate.settings[policy_key]

    if not isinstance(setting, str):
        message = f"checkpoint setting must be a string: {policy_key}"
        raise AdmissionError(message)

    return setting


def _direct_operation(
    function: Callable[..., TensorTree],
    args: Sequence[Any],
) -> CandidateOperation:
    def operation() -> TensorTree:
        return function(*args)

    return operation


def _with_activation_offload(
    candidate: Candidate,
    operation: CandidateOperation,
    offload: str,
) -> CandidateOperation:
    if offload == "none":
        return operation

    pack_hook, unpack_hook = _saved_tensor_hooks(candidate, offload)

    def wrapped() -> TensorTree:
        with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
            return operation()

    return wrapped


def _activation_offload(candidate: Candidate) -> str:
    value = candidate.settings.get("activation.offload")

    if value not in {"none", "saved_tensor_hooks_cpu", "custom_saved_tensor_hooks"}:
        message = "activation.offload is invalid"
        raise AdmissionError(message)

    return value


def _saved_tensor_hooks(
    candidate: Candidate,
    offload: str,
) -> tuple[Callable[[torch.Tensor], Any], Callable[[Any], torch.Tensor]]:
    if offload == "saved_tensor_hooks_cpu":
        return _cpu_pack_hook, _cpu_unpack_hook

    pack_hook = candidate.settings.get("activation.pack_hook")
    unpack_hook = candidate.settings.get("activation.unpack_hook")

    if not callable(pack_hook) or not callable(unpack_hook):
        message = "custom_saved_tensor_hooks requires activation pack and unpack hooks"
        raise AdmissionError(message)

    return pack_hook, unpack_hook


def _cpu_pack_hook(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.device]:
    return tensor.detach().cpu(), tensor.device


def _cpu_unpack_hook(packed: tuple[torch.Tensor, torch.device]) -> torch.Tensor:
    tensor, device = packed

    return tensor.to(device)


def _checkpoint_context_fn(candidate: Candidate) -> Callable[[], Any]:
    context_fn = candidate.settings["checkpoint.context_fn"]

    if context_fn == "none":
        return noop_context_fn

    callable_context_fn = candidate.settings.get("checkpoint.context_fn_callable")

    if not callable(callable_context_fn):
        message = "checkpoint.context_fn=declared_context_pair requires callable"
        raise AdmissionError(message)

    return callable_context_fn


def _checkpoint_bool(candidate: Candidate, key: str) -> bool:
    value = candidate.settings[key]

    if value == "true":
        return True

    if value == "false":
        return False

    message = f"{key} must be false or true"
    raise AdmissionError(message)
