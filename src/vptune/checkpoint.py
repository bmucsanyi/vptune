"""Checkpoint execution helpers for adapter runtimes."""

from collections.abc import Callable, Sequence
from typing import Any

from torch.utils.checkpoint import checkpoint, noop_context_fn

from vptune.admission import admit_checkpoint
from vptune.data import Candidate, CandidateOperation, TensorTree
from vptune.errors import AdmissionError

ACTIVE_CHECKPOINT_SETTINGS = (
    "non_reentrant",
    "non_reentrant_preserve_rng",
    "non_reentrant_deterministic",
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

    if setting == "disabled":
        return _direct_operation(function, args)

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
            preserve_rng_state=candidate.settings["preserve_rng_state"],
            determinism_check=candidate.settings["determinism_check"],
            context_fn=context_fn,
            early_stop=candidate.settings["early_stop"],
        )

    return operation


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


def _checkpoint_context_fn(candidate: Candidate) -> Callable[[], Any]:
    context_fn = candidate.settings["context_fn"]

    if context_fn is None:
        return noop_context_fn

    if not callable(context_fn):
        message = "checkpoint context_fn must be callable or None"
        raise AdmissionError(message)

    return context_fn
