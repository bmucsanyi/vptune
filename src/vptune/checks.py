"""Reference check and threshold helpers."""

import math
from collections.abc import Mapping
from typing import Any

import torch

from vptune.errors import ReferenceFailedError
from vptune.tensor_tree import TensorTree, tree_l2_norm, tree_max_abs, tree_sub

STANDARD_THRESHOLDS = {
    "max_abs_diff": 1e-4,
    "max_rel_diff": 1e-3,
    "value_abs_diff": 1e-4,
    "multiply_max_abs_diff": 1e-4,
    "multiply_max_rel_diff": 1e-3,
    "inverse_max_abs_diff": 1e-4,
    "inverse_max_rel_diff": 1e-3,
    "inner_abs_diff": 1e-4,
    "inverse_residual": 1e-4,
    "symmetry_max_abs_diff": 1e-4,
    "psd_violation": 1e-12,
    "directional_abs_diff": 1e-3,
    "directional_rel_diff": 1e-2,
}
THRESHOLD_PAIRS = (
    ("max_abs_diff", "max_rel_diff"),
    ("directional_abs_diff", "directional_rel_diff"),
    ("multiply_max_abs_diff", "multiply_max_rel_diff"),
    ("inverse_max_abs_diff", "inverse_max_rel_diff"),
)
MIN_THRESHOLD_FIELDS = ("damping_min",)
NUMERIC_ERROR_BOUND_FIELDS = (
    "k",
    "epsilon",
    "C_op",
    "S_row",
    "output_norm_floor",
)


def validate_thresholds(
    measurements: Mapping[str, Any],
    thresholds: Mapping[str, float],
) -> None:
    """Raise when any thresholded measurement fails.

    Raises:
        ReferenceFailedError: If a measurement is missing, nonfinite, or too large.
    """
    if not thresholds:
        message = "thresholds must be nonempty"
        raise ReferenceFailedError(message)

    paired = set()

    for absolute_key, relative_key in THRESHOLD_PAIRS:
        if absolute_key not in thresholds or relative_key not in thresholds:
            continue

        absolute_value = finite_value(measurements, absolute_key)
        relative_value = finite_value(measurements, relative_key)
        absolute_threshold = finite_value(thresholds, absolute_key)
        relative_threshold = finite_value(thresholds, relative_key)

        if absolute_value > absolute_threshold and relative_value > relative_threshold:
            message = f"measurement exceeds threshold: {absolute_key}/{relative_key}"
            raise ReferenceFailedError(message)

        paired.add(absolute_key)
        paired.add(relative_key)

    for key in thresholds:
        if key in paired:
            continue

        value = finite_value(measurements, key)
        threshold = finite_value(thresholds, key)

        if key in MIN_THRESHOLD_FIELDS:
            if value < threshold:
                message = f"measurement below threshold: {key}"
                raise ReferenceFailedError(message)
        elif value > threshold:
            message = f"measurement exceeds threshold: {key}"
            raise ReferenceFailedError(message)


def numeric_error_bound_measurements(
    settings: Mapping[str, Any],
    bound_fields: Mapping[str, Any],
    reference: TensorTree,
) -> dict[str, float]:
    """Return derived numeric error-bound measurements.

    Raises:
        ReferenceFailedError: If required fields are missing or invalid.
    """
    if not uses_reduction_degrading_setting(settings):
        return {}

    missing = tuple(
        field for field in NUMERIC_ERROR_BOUND_FIELDS if field not in bound_fields
    )

    if missing:
        message = f"numeric error bound fields are missing: {missing}"
        raise ReferenceFailedError(message)

    k = finite_bound_value(bound_fields, "k")
    epsilon = finite_bound_value(bound_fields, "epsilon")
    c_op = finite_bound_value(bound_fields, "C_op")
    s_row = finite_bound_value(bound_fields, "S_row")
    output_norm_floor = finite_bound_value(bound_fields, "output_norm_floor")

    if k <= 0.0:
        message = "numeric error bound k must be positive"
        raise ReferenceFailedError(message)

    if epsilon <= 0.0:
        message = "numeric error bound epsilon must be positive"
        raise ReferenceFailedError(message)

    if c_op < 0.0:
        message = "numeric error bound C_op must be nonnegative"
        raise ReferenceFailedError(message)

    if s_row < 0.0:
        message = "numeric error bound S_row must be nonnegative"
        raise ReferenceFailedError(message)

    if output_norm_floor <= 0.0:
        message = "numeric error bound output_norm_floor must be positive"
        raise ReferenceFailedError(message)

    product = k * epsilon

    if product >= 1.0:
        message = "numeric error bound requires k * epsilon < 1"
        raise ReferenceFailedError(message)

    gamma = product / (1.0 - product)
    absolute_bound = c_op * gamma * s_row
    reference_norm = float(tree_l2_norm(reference).detach().cpu())
    denominator = max(reference_norm, output_norm_floor)
    relative_bound = absolute_bound / denominator

    return {
        "numeric_error_bound_abs": absolute_bound,
        "numeric_error_bound_rel": relative_bound,
        "numeric_reference_l2_norm": reference_norm,
    }


def validate_numeric_error_bound(
    measurements: Mapping[str, Any],
    thresholds: Mapping[str, float],
    bound_measurements: Mapping[str, Any],
) -> None:
    """Raise when measured error or the derived bound violates thresholds.

    Raises:
        ReferenceFailedError: If the derived bound rejects the row.
    """
    if not bound_measurements:
        return

    measured_abs = finite_value(measurements, "max_abs_diff")
    measured_rel = finite_value(measurements, "max_rel_diff")
    bound_abs = finite_value(bound_measurements, "numeric_error_bound_abs")
    bound_rel = finite_value(bound_measurements, "numeric_error_bound_rel")
    threshold_abs = finite_value(thresholds, "max_abs_diff")
    threshold_rel = finite_value(thresholds, "max_rel_diff")

    if measured_abs > bound_abs and measured_rel > bound_rel:
        message = "numeric error exceeds derived bound"
        raise ReferenceFailedError(message)

    if bound_abs > threshold_abs and bound_rel > threshold_rel:
        message = "numeric error bound exceeds threshold"
        raise ReferenceFailedError(message)


def uses_reduction_degrading_setting(settings: Mapping[str, Any]) -> bool:
    """Return whether a row uses a reduction-degrading numeric setting."""
    if settings.get("dtype.accumulation") in {"bf16", "fp16"}:
        return True

    if settings.get("numeric.float32_matmul_precision") in {"high", "medium"}:
        return True

    if settings.get("numeric.bf16_reduced_precision_reduction") == "true":
        return True

    if settings.get("numeric.fp16_reduced_precision_reduction") == "true":
        return True

    return any(
        key.endswith(".reduce_dtype") and value in {"bf16", "fp16"}
        for key, value in settings.items()
    )


def finite_value(values: Mapping[str, Any], key: str) -> float:
    """Return a finite numeric field.

    Raises:
        ReferenceFailedError: If the field is missing, nonnumeric, or nonfinite.
    """
    value = values.get(key)

    if not isinstance(value, int | float):
        message = f"measurement is missing or nonnumeric: {key}"
        raise ReferenceFailedError(message)

    result = float(value)

    if not math.isfinite(result):
        message = f"measurement is nonfinite: {key}"
        raise ReferenceFailedError(message)

    return result


def finite_bound_value(values: Mapping[str, Any], key: str) -> float:
    """Return a finite numeric error-bound field.

    Raises:
        ReferenceFailedError: If the field is missing, nonnumeric, or nonfinite.
    """
    value = values.get(key)

    if not isinstance(value, int | float):
        message = f"numeric error bound field is missing or nonnumeric: {key}"
        raise ReferenceFailedError(message)

    result = float(value)

    if not math.isfinite(result):
        message = f"numeric error bound field is nonfinite: {key}"
        raise ReferenceFailedError(message)

    return result


def tree_error_measurements(
    observed: TensorTree,
    reference: TensorTree,
    *,
    denominator: torch.Tensor | None = None,
) -> dict[str, float]:
    """Return absolute and relative max-error measurements."""
    diff = tree_sub(observed, reference)
    absolute = tree_max_abs(diff)

    if denominator is None:
        denominator = tree_max_abs(reference)

    if float(denominator.detach().cpu()) <= 0.0:
        relative = torch.where(
            absolute <= 0.0,
            torch.zeros_like(absolute),
            torch.full_like(absolute, math.inf),
        )
    else:
        relative = absolute / denominator

    return {
        "max_abs_diff": float(absolute.detach().cpu()),
        "max_rel_diff": float(relative.detach().cpu()),
    }
