"""Reference check and threshold helpers."""

import math
from collections.abc import Mapping
from typing import Any

import torch

from vptune.errors import ReferenceFailedError
from vptune.tensor_tree import TensorTree, tree_max_abs, tree_sub

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
DTYPE_ABS_FLOORS = {
    "bfloat16": 0.25 * float(torch.finfo(torch.bfloat16).eps),
    "float16": 0.25 * float(torch.finfo(torch.float16).eps),
}
DTYPE_FLOOR_FIELDS = ("max_abs_diff", "directional_abs_diff")
DTYPE_FLOOR_SETTINGS = ("model_dtype", "compute_dtype", "accumulation_dtype")


def thresholds_for_measurements(
    measurements: Mapping[str, Any],
    settings: Mapping[str, Any],
) -> dict[str, float]:
    """Return active thresholds for present measurement fields.

    Raises:
        ReferenceFailedError: If no thresholded measurement field is present.
    """
    thresholds = {
        name: threshold
        for name, threshold in STANDARD_THRESHOLDS.items()
        if name in measurements
    }
    apply_dtype_floors(thresholds, settings)

    if not thresholds:
        message = "reference check has no thresholded measurements"
        raise ReferenceFailedError(message)

    return thresholds


def apply_dtype_floors(
    thresholds: dict[str, float],
    settings: Mapping[str, Any],
) -> None:
    """Apply dtype absolute-threshold floors in place."""
    floors = []

    for setting in DTYPE_FLOOR_SETTINGS:
        dtype_name = settings.get(setting)

        if not isinstance(dtype_name, str):
            continue

        floor = DTYPE_ABS_FLOORS.get(dtype_name)

        if floor is not None:
            floors.append(floor)

    if not floors:
        return

    floor = max(floors)

    for field in DTYPE_FLOOR_FIELDS:
        if field in thresholds:
            thresholds[field] = max(thresholds[field], floor)


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

        if value > threshold:
            message = f"measurement exceeds threshold: {key}"
            raise ReferenceFailedError(message)


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


def assert_tree_close(
    observed: TensorTree,
    reference: TensorTree,
    *,
    settings: Mapping[str, Any] | None = None,
    thresholds: Mapping[str, float],
) -> dict[str, float]:
    """Validate tree agreement and return measurements.

    Returns:
        Error measurements.
    """
    safe_settings = {} if settings is None else settings
    measurements = tree_error_measurements(observed, reference)
    active_thresholds = dict(thresholds)
    apply_dtype_floors(active_thresholds, safe_settings)
    validate_thresholds(measurements, active_thresholds)

    return measurements
