"""Operator constructors."""

from collections.abc import Mapping
from typing import Any

from vptune.data import OperatorSpec


def gradient(
    family: str,
    objective_id: str,
    *,
    aggregation: str,
    randomness: Mapping[str, Any] | None = None,
    thresholds: Mapping[str, float] | None = None,
) -> OperatorSpec:
    """Return a gradient operator spec."""
    return OperatorSpec(
        family,
        "gradient",
        objective_id,
        aggregation=aggregation,
        randomness={} if randomness is None else dict(randomness),
        thresholds={} if thresholds is None else dict(thresholds),
    )


def jvp(
    family: str,
    objective_id: str,
    *,
    aggregation: str,
    randomness: Mapping[str, Any] | None = None,
    thresholds: Mapping[str, float] | None = None,
) -> OperatorSpec:
    """Return a JVP operator spec."""
    return OperatorSpec(
        family,
        "jvp",
        objective_id,
        aggregation=aggregation,
        randomness={} if randomness is None else dict(randomness),
        thresholds={} if thresholds is None else dict(thresholds),
    )


def vjp(
    family: str,
    objective_id: str,
    *,
    aggregation: str,
    randomness: Mapping[str, Any] | None = None,
    thresholds: Mapping[str, float] | None = None,
) -> OperatorSpec:
    """Return a VJP operator spec."""
    return OperatorSpec(
        family,
        "vjp",
        objective_id,
        aggregation=aggregation,
        randomness={} if randomness is None else dict(randomness),
        thresholds={} if thresholds is None else dict(thresholds),
    )


def hvp(
    family: str,
    objective_id: str,
    *,
    aggregation: str,
    randomness: Mapping[str, Any] | None = None,
    thresholds: Mapping[str, float] | None = None,
) -> OperatorSpec:
    """Return an HVP operator spec."""
    return OperatorSpec(
        family,
        "hvp",
        objective_id,
        aggregation=aggregation,
        randomness={} if randomness is None else dict(randomness),
        thresholds={} if thresholds is None else dict(thresholds),
    )


def ggnvp(
    family: str,
    objective_id: str,
    *,
    aggregation: str,
    randomness: Mapping[str, Any] | None = None,
    thresholds: Mapping[str, float] | None = None,
) -> OperatorSpec:
    """Return a GGNVP operator spec."""
    return OperatorSpec(
        family,
        "ggnvp",
        objective_id,
        aggregation=aggregation,
        randomness={} if randomness is None else dict(randomness),
        thresholds={} if thresholds is None else dict(thresholds),
    )


def fisher_vp(
    family: str,
    objective_id: str,
    *,
    aggregation: str,
    distribution: str,
    label_policy: str,
    expectation: str,
    sample_space: str,
    loss_reduction: str,
    denominator: str,
    logits_axis: int | None = None,
    sample_count: int | None = None,
    seed: int | None = None,
    randomness: Mapping[str, Any] | None = None,
    thresholds: Mapping[str, float] | None = None,
) -> OperatorSpec:
    """Return a FisherVP operator spec."""
    raw_semantics = {
        "distribution": distribution,
        "label_policy": label_policy,
        "expectation": expectation,
        "sample_space": sample_space,
        "loss_reduction": loss_reduction,
        "denominator": denominator,
        "sample_count": sample_count,
        "seed": seed,
        "logits_axis": logits_axis,
    }
    semantics = {
        key: value for key, value in raw_semantics.items() if value is not None
    }

    return OperatorSpec(
        family,
        "fisher_vp",
        objective_id,
        aggregation=aggregation,
        semantics=semantics,
        randomness={} if randomness is None else dict(randomness),
        thresholds={} if thresholds is None else dict(thresholds),
    )


def empirical_fisher_vp(
    family: str,
    objective_id: str,
    *,
    aggregation: str,
    randomness: Mapping[str, Any] | None = None,
    thresholds: Mapping[str, float] | None = None,
) -> OperatorSpec:
    """Return an empirical FisherVP operator spec."""
    return OperatorSpec(
        family,
        "empirical_fisher_vp",
        objective_id,
        aggregation=aggregation,
        randomness={} if randomness is None else dict(randomness),
        thresholds={} if thresholds is None else dict(thresholds),
    )


def metric(
    family: str,
    objective_id: str,
    *,
    aggregation: str,
    randomness: Mapping[str, Any] | None = None,
    thresholds: Mapping[str, float] | None = None,
) -> OperatorSpec:
    """Return a metric operator spec."""
    return OperatorSpec(
        family,
        "metric",
        objective_id,
        aggregation=aggregation,
        randomness={} if randomness is None else dict(randomness),
        thresholds={} if thresholds is None else dict(thresholds),
    )


def inverse_metric(
    family: str,
    objective_id: str,
    *,
    aggregation: str,
    randomness: Mapping[str, Any] | None = None,
    thresholds: Mapping[str, float] | None = None,
) -> OperatorSpec:
    """Return an inverse-metric operator spec."""
    return OperatorSpec(
        family,
        "inverse_metric",
        objective_id,
        aggregation=aggregation,
        randomness={} if randomness is None else dict(randomness),
        thresholds={} if thresholds is None else dict(thresholds),
    )


def composition(
    family: str,
    objective_id: str,
    *,
    aggregation: str,
    randomness: Mapping[str, Any] | None = None,
    thresholds: Mapping[str, float] | None = None,
) -> OperatorSpec:
    """Return a composed operator spec."""
    return OperatorSpec(
        family,
        "composition",
        objective_id,
        aggregation=aggregation,
        randomness={} if randomness is None else dict(randomness),
        thresholds={} if thresholds is None else dict(thresholds),
    )
