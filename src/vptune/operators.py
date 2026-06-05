"""Operator constructors."""

from collections.abc import Mapping, Sequence
from typing import Any

from vptune.data import OperatorSpec
from vptune.errors import MaterializationError


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
        batch_inputs={"reference": (), "operation": ()},
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
        batch_inputs={"reference": ("tangent_vector",), "operation": ()},
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
        batch_inputs={"reference": ("symmetry_vector",), "operation": ()},
        randomness={} if randomness is None else dict(randomness),
        thresholds={} if thresholds is None else dict(thresholds),
    )


def ggnvp(
    family: str,
    objective_id: str,
    *,
    aggregation: str,
    loss_geometry: str,
    randomness: Mapping[str, Any] | None = None,
    thresholds: Mapping[str, float] | None = None,
) -> OperatorSpec:
    """Return a GGNVP operator spec."""
    if loss_geometry == "psd_metric":
        batch_inputs = {
            "reference": ("loss_hessian", "symmetry_vector"),
            "operation": ("loss_hessian",),
        }
    else:
        batch_inputs = {
            "reference": ("loss_hessian",),
            "operation": ("loss_hessian",),
        }

    return OperatorSpec(
        family,
        "ggnvp",
        objective_id,
        aggregation=aggregation,
        semantics={"loss_geometry": loss_geometry},
        batch_inputs=batch_inputs,
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
    sample_space: str,
    score_reduction: str,
    denominator: str,
    randomness: Mapping[str, Any] | None = None,
    thresholds: Mapping[str, float] | None = None,
) -> OperatorSpec:
    """Return a FisherVP operator spec.

    Raises:
        MaterializationError: If exact categorical Fisher is requested.
    """
    if distribution == "categorical":
        message = "exact categorical Fisher is represented by GGNVP"
        raise MaterializationError(message)

    raw_semantics = {
        "distribution": distribution,
        "label_policy": label_policy,
        "sample_space": sample_space,
        "score_reduction": score_reduction,
        "denominator": denominator,
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
        batch_inputs={"reference": (), "operation": ()},
        randomness={} if randomness is None else dict(randomness),
        thresholds={} if thresholds is None else dict(thresholds),
    )


def sampled_fisher_vp(
    family: str,
    objective_id: str,
    *,
    aggregation: str,
    distribution: str,
    label_policy: str,
    sample_count: int,
    sample_source: str,
    sampling_bound: Mapping[str, Any],
    score_reduction: str,
    denominator: str,
    randomness: Mapping[str, Any] | None = None,
    thresholds: Mapping[str, float] | None = None,
) -> OperatorSpec:
    """Return a sampled FisherVP operator spec.

    Raises:
        MaterializationError: If sample count is invalid.
    """
    if sample_count <= 0:
        message = "sampled Fisher sample_count must be positive"
        raise MaterializationError(message)

    if sample_source not in {"fixed_sample_table", "fixed_seed_and_count"}:
        message = f"sampled Fisher sample_source is unsupported: {sample_source}"
        raise MaterializationError(message)

    semantics = {
        "distribution": distribution,
        "label_policy": label_policy,
        "sample_count": sample_count,
        "sample_source": sample_source,
        "sampling_bound": dict(sampling_bound),
        "score_reduction": score_reduction,
        "denominator": denominator,
    }

    return OperatorSpec(
        family,
        "sampled_fisher_vp",
        objective_id,
        aggregation=aggregation,
        semantics=semantics,
        batch_inputs={"reference": (), "operation": ()},
        randomness={} if randomness is None else dict(randomness),
        thresholds={} if thresholds is None else dict(thresholds),
    )


def empirical_fisher_vp(
    family: str,
    objective_id: str,
    *,
    aggregation: str,
    example_loss_reduction: str,
    denominator: str,
    randomness: Mapping[str, Any] | None = None,
    thresholds: Mapping[str, float] | None = None,
) -> OperatorSpec:
    """Return an empirical FisherVP operator spec."""
    semantics = {
        "example_loss_reduction": example_loss_reduction,
        "denominator": denominator,
    }

    return OperatorSpec(
        family,
        "empirical_fisher_vp",
        objective_id,
        aggregation=aggregation,
        semantics=semantics,
        batch_inputs={"reference": (), "operation": ()},
        randomness={} if randomness is None else dict(randomness),
        thresholds={} if thresholds is None else dict(thresholds),
    )


def metric(
    family: str,
    objective_id: str,
    *,
    aggregation: str,
    representation: Mapping[str, Any],
    randomness: Mapping[str, Any] | None = None,
    thresholds: Mapping[str, float] | None = None,
) -> OperatorSpec:
    """Return a metric operator spec."""
    representation_fields = _metric_representation_inputs(representation)

    return OperatorSpec(
        family,
        "metric",
        objective_id,
        aggregation=aggregation,
        semantics={"representation": dict(representation)},
        batch_inputs={
            "reference": representation_fields,
            "operation": representation_fields,
        },
        randomness={} if randomness is None else dict(randomness),
        thresholds={} if thresholds is None else dict(thresholds),
    )


def inverse_metric(
    family: str,
    objective_id: str,
    *,
    aggregation: str,
    representation: Mapping[str, Any],
    damping: float,
    randomness: Mapping[str, Any] | None = None,
    thresholds: Mapping[str, float] | None = None,
) -> OperatorSpec:
    """Return an inverse-metric operator spec.

    Raises:
        MaterializationError: If damping is negative.
    """
    if damping < 0.0:
        message = "inverse metric damping must be nonnegative"
        raise MaterializationError(message)

    representation_fields = _metric_representation_inputs(representation)

    return OperatorSpec(
        family,
        "inverse_metric",
        objective_id,
        aggregation=aggregation,
        semantics={"damping": damping, "representation": dict(representation)},
        batch_inputs={
            "reference": representation_fields,
            "operation": representation_fields,
        },
        randomness={} if randomness is None else dict(randomness),
        thresholds={} if thresholds is None else dict(thresholds),
    )


def _metric_representation_inputs(representation: Mapping[str, Any]) -> tuple[str, ...]:
    kind = representation.get("kind")

    if kind == "dense_matrix":
        return ("metric_matrix",)

    if kind == "diagonal_tree":
        return ("metric_diagonal",)

    if kind == "block_diagonal":
        return ("metric_blocks",)

    if kind == "kfac_factors":
        return ("kfac_factors",)

    if kind == "low_rank_factors":
        return ("low_rank_factors",)

    if kind == "ggn_derived_factors":
        return ("ggn_factors",)

    message = "metric representation kind is unsupported"
    raise MaterializationError(message)


def composition(
    family: str,
    objective_id: str,
    *,
    aggregation: str,
    children: Sequence[str],
    randomness: Mapping[str, Any] | None = None,
    thresholds: Mapping[str, float] | None = None,
) -> OperatorSpec:
    """Return a composed operator spec."""
    child_order = _composition_children(children)

    return OperatorSpec(
        family,
        "composition",
        objective_id,
        aggregation=aggregation,
        semantics={"children": child_order},
        randomness={} if randomness is None else dict(randomness),
        thresholds={} if thresholds is None else dict(thresholds),
    )


def _composition_children(children: Sequence[str]) -> tuple[str, ...]:
    child_order = tuple(children)

    if not child_order:
        message = "composition children must be non-empty"
        raise MaterializationError(message)

    for child in child_order:
        if not isinstance(child, str) or not child:
            message = "composition children must be non-empty strings"
            raise MaterializationError(message)

    if len(set(child_order)) != len(child_order):
        message = "composition children must be unique"
        raise MaterializationError(message)

    return child_order
