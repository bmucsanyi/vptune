"""Admission helpers for framework-sensitive candidates."""

from collections.abc import Mapping
from typing import Any

from vptune.errors import AdmissionError

FUNCTIONAL_CALL_FIELDS = (
    "parameter_keys",
    "buffer_keys",
    "tie_weights",
    "strict",
    "parametrization_policy",
    "mutates_state",
    "mutated_parameter_keys",
    "mutated_buffer_keys",
    "module_mode",
)
TORCH_FUNC_FIELDS = (
    "contains_autograd_call",
    "contains_backward_call",
    "uses_out_variant",
    "uses_data_dependent_control_flow",
    "uses_item",
    "has_dynamic_shape_output",
    "vectorization.randomness",
    "requires_forward_ad",
    "forward_ad_supported",
)
TORCH_FUNC_BOOL_FIELDS = (
    "contains_autograd_call",
    "contains_backward_call",
    "uses_out_variant",
    "uses_data_dependent_control_flow",
    "uses_item",
    "has_dynamic_shape_output",
    "requires_forward_ad",
    "forward_ad_supported",
)
FORWARD_AD_FIELDS = ("requires_forward_ad", "forward_ad_supported")
CHECKPOINT_FIELDS = (
    "checkpoint.use_reentrant",
    "checkpoint.preserve_rng_state",
    "checkpoint.determinism_check",
    "checkpoint.context_fn",
    "checkpoint.early_stop",
    "checkpoint.moves_to_new_device",
    "checkpoint.uses_global_state",
)


def require_fields(settings: Mapping[str, Any], fields: tuple[str, ...]) -> None:
    """Raise when settings miss a required field.

    Raises:
        AdmissionError: If any required field is missing.
    """
    missing = tuple(field for field in fields if field not in settings)

    if missing:
        message = f"admission settings missing fields: {', '.join(missing)}"
        raise AdmissionError(message)


def admit_functional_call(settings: Mapping[str, Any]) -> None:
    """Validate functional_call admission fields.

    Raises:
        AdmissionError: If the candidate is not admissible.
    """
    require_fields(settings, FUNCTIONAL_CALL_FIELDS)
    _require_string_tuple(settings["parameter_keys"], "parameter_keys")
    _require_string_tuple(settings["buffer_keys"], "buffer_keys")
    _require_bool(settings["tie_weights"], "tie_weights")
    _require_bool(settings["strict"], "strict")
    _require_bool(settings["mutates_state"], "mutates_state")
    _require_string_tuple(settings["mutated_parameter_keys"], "mutated_parameter_keys")
    _require_string_tuple(settings["mutated_buffer_keys"], "mutated_buffer_keys")

    if settings["parametrization_policy"] != "active":
        message = "parametrization_policy must be active"
        raise AdmissionError(message)

    if settings["module_mode"] not in {"train", "eval"}:
        message = "module_mode must be train or eval"
        raise AdmissionError(message)

    mutated_parameter_keys = settings["mutated_parameter_keys"]
    mutated_buffer_keys = settings["mutated_buffer_keys"]

    if settings["mutates_state"]:
        if not mutated_parameter_keys and not mutated_buffer_keys:
            message = "functional_call mutation requires declared mutated keys"
            raise AdmissionError(message)

        _require_subset(
            mutated_parameter_keys,
            settings["parameter_keys"],
            "mutated_parameter_keys",
            "parameter_keys",
        )
        _require_subset(
            mutated_buffer_keys,
            settings["buffer_keys"],
            "mutated_buffer_keys",
            "buffer_keys",
        )
    elif mutated_parameter_keys or mutated_buffer_keys:
        message = "mutated keys require mutates_state=True"
        raise AdmissionError(message)


def _require_string_tuple(value: Any, label: str) -> None:
    if not isinstance(value, tuple) or not all(isinstance(item, str) for item in value):
        message = f"{label} must be a tuple of strings"
        raise AdmissionError(message)


def _require_bool(value: Any, label: str) -> None:
    if not isinstance(value, bool):
        message = f"{label} must be a bool"
        raise AdmissionError(message)


def _require_subset(
    values: tuple[str, ...],
    allowed: tuple[str, ...],
    values_label: str,
    allowed_label: str,
) -> None:
    unexpected = tuple(value for value in values if value not in allowed)

    if unexpected:
        message = f"{values_label} must be contained in {allowed_label}: {unexpected}"
        raise AdmissionError(message)


def admit_torch_func(settings: Mapping[str, Any]) -> None:
    """Validate torch.func transform admission fields.

    Raises:
        AdmissionError: If the candidate is not admissible.
    """
    require_fields(settings, TORCH_FUNC_FIELDS)

    for field in TORCH_FUNC_BOOL_FIELDS:
        _require_bool(settings[field], field)

    rejected_flags = (
        "contains_autograd_call",
        "contains_backward_call",
        "uses_out_variant",
        "uses_data_dependent_control_flow",
        "uses_item",
        "has_dynamic_shape_output",
    )

    for flag in rejected_flags:
        if settings[flag]:
            message = f"torch.func candidate rejected by flag: {flag}"
            raise AdmissionError(message)

    if settings["vectorization.randomness"] not in {"error", "same", "different"}:
        message = "vectorization.randomness is invalid"
        raise AdmissionError(message)

    if settings["requires_forward_ad"] and not settings["forward_ad_supported"]:
        message = "forward AD is required but unsupported"
        raise AdmissionError(message)


def admit_forward_ad(settings: Mapping[str, Any]) -> None:
    """Validate forward AD admission fields.

    Raises:
        AdmissionError: If the candidate cannot use forward AD.
    """
    require_fields(settings, FORWARD_AD_FIELDS)
    _require_bool(settings["requires_forward_ad"], "requires_forward_ad")
    _require_bool(settings["forward_ad_supported"], "forward_ad_supported")

    if settings["requires_forward_ad"] is not True:
        message = "candidate must declare requires_forward_ad=True"
        raise AdmissionError(message)

    if settings["forward_ad_supported"] is not True:
        message = "forward AD is required but unsupported"
        raise AdmissionError(message)


def admit_checkpoint(settings: Mapping[str, Any]) -> None:
    """Validate checkpoint admission fields.

    Raises:
        AdmissionError: If the candidate is not admissible.
    """
    require_fields(settings, CHECKPOINT_FIELDS)
    _require_bool_token(
        settings["checkpoint.use_reentrant"], "checkpoint.use_reentrant"
    )
    _require_bool_token(
        settings["checkpoint.preserve_rng_state"],
        "checkpoint.preserve_rng_state",
    )
    _require_bool_token(settings["checkpoint.early_stop"], "checkpoint.early_stop")
    _require_bool_token(
        settings["checkpoint.moves_to_new_device"],
        "checkpoint.moves_to_new_device",
    )
    _require_bool_token(
        settings["checkpoint.uses_global_state"],
        "checkpoint.uses_global_state",
    )

    if settings["checkpoint.use_reentrant"] == "true":
        message = "checkpoint candidates must use non-reentrant checkpointing"
        raise AdmissionError(message)

    if settings["checkpoint.determinism_check"] not in {"default", "none"}:
        message = "checkpoint.determinism_check must be default or none"
        raise AdmissionError(message)

    if settings["checkpoint.context_fn"] not in {"none", "declared_context_pair"}:
        message = "checkpoint.context_fn must be none or declared_context_pair"
        raise AdmissionError(message)

    if settings["checkpoint.moves_to_new_device"] == "true":
        message = "checkpoint candidate moves tensors to an undeclared device"
        raise AdmissionError(message)

    if settings["checkpoint.uses_global_state"] == "true":
        message = "checkpoint candidate depends on global mutable state"
        raise AdmissionError(message)


def _require_bool_token(value: Any, label: str) -> None:
    if value not in {"false", "true"}:
        message = f"{label} must be false or true"
        raise AdmissionError(message)
