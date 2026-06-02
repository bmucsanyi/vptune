"""Test helpers for vptune assertions."""

from collections.abc import Mapping
from typing import Any

from vptune.checks import (
    STANDARD_THRESHOLDS,
    apply_dtype_floors,
    tree_error_measurements,
    validate_thresholds,
)
from vptune.errors import ReferenceFailedError
from vptune.tensor_tree import TensorTree


def thresholds_for_measurements(
    measurements: Mapping[str, Any],
    settings: Mapping[str, Any],
) -> dict[str, float]:
    """Return active thresholds for present measurement fields.

    Raises:
        ReferenceFailedError: If no thresholded field is present.
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
    settings_map = {} if settings is None else settings
    measurements = tree_error_measurements(observed, reference)
    active_thresholds = dict(thresholds)
    apply_dtype_floors(active_thresholds, settings_map)
    validate_thresholds(measurements, active_thresholds)

    return measurements
