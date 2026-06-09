"""Test helpers for vptune assertions."""

from collections.abc import Mapping
from typing import Any

from vptune.core.tensor_tree import TensorTree
from vptune.engine.checks import (
    STANDARD_THRESHOLDS,
    tree_error_measurements,
    validate_thresholds,
)
from vptune.errors import ReferenceFailedError


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
    del settings

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
    del settings
    measurements = tree_error_measurements(observed, reference)
    active_thresholds = dict(thresholds)
    validate_thresholds(measurements, active_thresholds)

    return measurements
