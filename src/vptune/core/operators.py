"""Operator constructors."""

import math
from collections.abc import Mapping, Sequence
from typing import Any

from vptune.core.data import OperatorSpec
from vptune.errors import MaterializationError

AGGREGATIONS = ("sum", "mean", "mean_per_example", "none")
FISHER_DISTRIBUTIONS = ("explicit_score_gradients",)
FISHER_LABEL_POLICIES = ("explicit_scores",)
SAMPLED_FISHER_LABEL_POLICIES = ("sampled_labels",)
FISHER_SAMPLE_SPACES = ("terms",)
FISHER_SCORE_REDUCTIONS = ("none",)
FISHER_DENOMINATORS = ("one", "num_examples", "batch_normalization")
SAMPLED_FISHER_DENOMINATORS = (
    "one",
    "num_examples",
    "num_tokens",
    "batch_normalization",
)
EMPIRICAL_FISHER_EXAMPLE_LOSS_REDUCTIONS = ("per_example",)
METRIC_REPRESENTATION_INPUTS = {
    "dense_matrix": ("metric_matrix",),
    "diagonal_tree": ("metric_diagonal",),
    "block_diagonal": ("metric_blocks",),
    "kfac_factors": ("kfac_factors",),
    "ekfac_factors": (
        "ekfac_eigvecs_a",
        "ekfac_eigvecs_g",
        "ekfac_corrected_eigenvalues",
    ),
    "low_rank_factors": ("low_rank_factors",),
    "ggn_derived_factors": ("ggn_factors",),
    "matrix_free": (),
}


def gradient(
    family: str,
    objective_id: str,
    *,
    aggregation: str,
    randomness: Mapping[str, Any] | None = None,
    thresholds: Mapping[str, float] | None = None,
) -> OperatorSpec:
    """Return a gradient operator spec."""
    _require_value(aggregation, AGGREGATIONS, "aggregation")

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
    _require_value(aggregation, AGGREGATIONS, "aggregation")

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
    _require_value(aggregation, AGGREGATIONS, "aggregation")

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
    _require_value(aggregation, AGGREGATIONS, "aggregation")

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
    randomness: Mapping[str, Any] | None = None,
    thresholds: Mapping[str, float] | None = None,
) -> OperatorSpec:
    """Return a GGNVP operator spec."""
    _require_value(aggregation, AGGREGATIONS, "aggregation")

    return OperatorSpec(
        family,
        "ggnvp",
        objective_id,
        aggregation=aggregation,
        semantics={},
        batch_inputs={
            "reference": ("loss_hessian", "symmetry_vector"),
            "operation": ("loss_hessian",),
        },
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
    _require_value(aggregation, AGGREGATIONS, "aggregation")

    if distribution == "categorical":
        message = "exact categorical Fisher is represented by GGNVP"
        raise MaterializationError(message)

    _require_value(distribution, FISHER_DISTRIBUTIONS, "distribution")
    _require_value(label_policy, FISHER_LABEL_POLICIES, "label_policy")
    _require_value(sample_space, FISHER_SAMPLE_SPACES, "sample_space")
    _require_value(score_reduction, FISHER_SCORE_REDUCTIONS, "score_reduction")
    _require_value(denominator, FISHER_DENOMINATORS, "denominator")

    semantics = {
        "distribution": distribution,
        "label_policy": label_policy,
        "sample_space": sample_space,
        "score_reduction": score_reduction,
        "denominator": denominator,
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
    _require_value(aggregation, AGGREGATIONS, "aggregation")
    _require_value(distribution, FISHER_DISTRIBUTIONS, "distribution")
    _require_value(label_policy, SAMPLED_FISHER_LABEL_POLICIES, "label_policy")
    _require_value(score_reduction, FISHER_SCORE_REDUCTIONS, "score_reduction")
    _require_value(denominator, SAMPLED_FISHER_DENOMINATORS, "denominator")

    if isinstance(sample_count, bool) or sample_count <= 0:
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
        "sampling_bound": _sampling_bound(sampling_bound),
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
    _require_value(aggregation, AGGREGATIONS, "aggregation")
    _require_value(
        example_loss_reduction,
        EMPIRICAL_FISHER_EXAMPLE_LOSS_REDUCTIONS,
        "example_loss_reduction",
    )
    _require_value(denominator, FISHER_DENOMINATORS, "denominator")

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


def per_example_gradient(
    family: str,
    objective_id: str,
    *,
    aggregation: str,
    example_loss_reduction: str,
    randomness: Mapping[str, Any] | None = None,
    thresholds: Mapping[str, float] | None = None,
) -> OperatorSpec:
    """Return a per-example gradient operator spec."""
    _require_value(aggregation, AGGREGATIONS, "aggregation")
    _require_value(
        example_loss_reduction,
        EMPIRICAL_FISHER_EXAMPLE_LOSS_REDUCTIONS,
        "example_loss_reduction",
    )

    return OperatorSpec(
        family,
        "per_example_gradient",
        objective_id,
        aggregation=aggregation,
        semantics={"example_loss_reduction": example_loss_reduction},
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
    _require_value(aggregation, AGGREGATIONS, "aggregation")
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


def sqrt_metric(
    family: str,
    objective_id: str,
    *,
    aggregation: str,
    representation: Mapping[str, Any],
    randomness: Mapping[str, Any] | None = None,
    thresholds: Mapping[str, float] | None = None,
) -> OperatorSpec:
    """Return a metric square-root operator spec."""
    return _sqrt_metric_operator(
        family,
        objective_id,
        kind="sqrt_metric",
        aggregation=aggregation,
        representation=representation,
        damping=None,
        tol=None,
        randomness=randomness,
        thresholds=thresholds,
    )


def inverse_sqrt_metric(
    family: str,
    objective_id: str,
    *,
    aggregation: str,
    representation: Mapping[str, Any],
    damping: float = 0.0,
    tol: float | None = None,
    randomness: Mapping[str, Any] | None = None,
    thresholds: Mapping[str, float] | None = None,
) -> OperatorSpec:
    """Return an inverse metric square-root operator spec."""
    return _sqrt_metric_operator(
        family,
        objective_id,
        kind="inverse_sqrt_metric",
        aggregation=aggregation,
        representation=representation,
        damping=damping,
        tol=tol,
        randomness=randomness,
        thresholds=thresholds,
    )


def metric_inner(
    family: str,
    objective_id: str,
    *,
    aggregation: str,
    representation: Mapping[str, Any],
    as_norm: bool = False,
    randomness: Mapping[str, Any] | None = None,
    thresholds: Mapping[str, float] | None = None,
) -> OperatorSpec:
    """Return a metric inner-product operator spec."""
    return _metric_inner_operator(
        family,
        objective_id,
        kind="metric_inner",
        aggregation=aggregation,
        representation=representation,
        damping=None,
        as_norm=as_norm,
        tol=None,
        randomness=randomness,
        thresholds=thresholds,
    )


def inverse_metric_inner(
    family: str,
    objective_id: str,
    *,
    aggregation: str,
    representation: Mapping[str, Any],
    damping: float = 0.0,
    as_norm: bool = False,
    tol: float | None = None,
    randomness: Mapping[str, Any] | None = None,
    thresholds: Mapping[str, float] | None = None,
) -> OperatorSpec:
    """Return an inverse metric inner-product operator spec."""
    return _metric_inner_operator(
        family,
        objective_id,
        kind="inverse_metric_inner",
        aggregation=aggregation,
        representation=representation,
        damping=damping,
        as_norm=as_norm,
        tol=tol,
        randomness=randomness,
        thresholds=thresholds,
    )


def _sqrt_metric_operator(
    family: str,
    objective_id: str,
    *,
    kind: str,
    aggregation: str,
    representation: Mapping[str, Any],
    damping: float | None,
    tol: float | None,
    randomness: Mapping[str, Any] | None,
    thresholds: Mapping[str, float] | None,
) -> OperatorSpec:
    _require_value(aggregation, AGGREGATIONS, "aggregation")

    if damping is not None and damping < 0.0:
        message = "metric square-root damping must be nonnegative"
        raise MaterializationError(message)

    _require_metric_tol(tol)
    representation_fields = _metric_representation_inputs(representation)
    semantics = _sqrt_metric_semantics(representation, damping)

    if tol is not None:
        semantics["tol"] = tol

    return OperatorSpec(
        family,
        kind,
        objective_id,
        aggregation=aggregation,
        semantics=semantics,
        batch_inputs={
            "reference": representation_fields,
            "operation": representation_fields,
        },
        randomness={} if randomness is None else dict(randomness),
        thresholds={} if thresholds is None else dict(thresholds),
    )


def _metric_inner_operator(
    family: str,
    objective_id: str,
    *,
    kind: str,
    aggregation: str,
    representation: Mapping[str, Any],
    damping: float | None,
    as_norm: bool,
    tol: float | None,
    randomness: Mapping[str, Any] | None,
    thresholds: Mapping[str, float] | None,
) -> OperatorSpec:
    _require_value(aggregation, AGGREGATIONS, "aggregation")

    if damping is not None and damping < 0.0:
        message = "metric inner-product damping must be nonnegative"
        raise MaterializationError(message)

    _require_metric_tol(tol)
    representation_fields = _metric_representation_inputs(representation)
    semantics = _metric_inner_semantics(representation, damping, as_norm)

    if tol is not None:
        semantics["tol"] = tol

    return OperatorSpec(
        family,
        kind,
        objective_id,
        aggregation=aggregation,
        semantics=semantics,
        batch_inputs={
            "reference": representation_fields,
            "operation": representation_fields,
        },
        randomness={} if randomness is None else dict(randomness),
        thresholds={} if thresholds is None else dict(thresholds),
    )


def _sqrt_metric_semantics(
    representation: Mapping[str, Any],
    damping: float | None,
) -> dict[str, Any]:
    if damping is None:
        return {"representation": dict(representation)}

    return {
        "representation": dict(representation),
        "damping": damping,
        "damping_kind": "scalar",
        "damping_value": damping,
    }


def _metric_inner_semantics(
    representation: Mapping[str, Any],
    damping: float | None,
    as_norm: bool,
) -> dict[str, Any]:
    if damping is None:
        return {
            "representation": dict(representation),
            "as_norm": as_norm,
        }

    return {
        "representation": dict(representation),
        "damping": damping,
        "damping_kind": "scalar",
        "damping_value": damping,
        "as_norm": as_norm,
    }


def inverse_metric(
    family: str,
    objective_id: str,
    *,
    aggregation: str,
    representation: Mapping[str, Any],
    damping: float,
    tol: float | None = None,
    randomness: Mapping[str, Any] | None = None,
    thresholds: Mapping[str, float] | None = None,
) -> OperatorSpec:
    """Return an inverse-metric operator spec.

    Raises:
        MaterializationError: If damping is negative.
    """
    _require_value(aggregation, AGGREGATIONS, "aggregation")

    if damping < 0.0:
        message = "inverse metric damping must be nonnegative"
        raise MaterializationError(message)

    _require_metric_tol(tol)
    representation_fields = _metric_representation_inputs(representation)
    semantics = {
        "damping": damping,
        "damping_kind": "scalar",
        "damping_value": damping,
        "representation": dict(representation),
    }

    if tol is not None:
        semantics["tol"] = tol

    return OperatorSpec(
        family,
        "inverse_metric",
        objective_id,
        aggregation=aggregation,
        semantics=semantics,
        batch_inputs={
            "reference": representation_fields,
            "operation": representation_fields,
        },
        randomness={} if randomness is None else dict(randomness),
        thresholds={} if thresholds is None else dict(thresholds),
    )


def _require_metric_tol(tol: float | None) -> None:
    if tol is None:
        return

    if math.isfinite(tol) and tol > 0.0:
        return

    message = "inverse metric tolerance must be positive and finite"
    raise MaterializationError(message)


def _metric_representation_inputs(representation: Mapping[str, Any]) -> tuple[str, ...]:
    kind = representation.get("kind")

    if not isinstance(kind, str):
        message = "metric representation kind is unsupported"
        raise MaterializationError(message)

    fields = METRIC_REPRESENTATION_INPUTS.get(kind)

    if fields is not None:
        return fields

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
    _require_value(aggregation, AGGREGATIONS, "aggregation")
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


def _require_value(value: str, allowed: Sequence[str], field: str) -> None:
    if value not in allowed:
        message = f"{field} is unsupported: {value}"
        raise MaterializationError(message)


def _sampling_bound(sampling_bound: Mapping[str, Any]) -> dict[str, object]:
    if sampling_bound.get("kind") == "disabled":
        return {"kind": "disabled"}

    if sampling_bound.get("kind") in {
        "matrix_bernstein",
        "hutchinson_relative_variance",
    }:
        return {
            "kind": str(sampling_bound["kind"]),
            "failure_probability": _sampling_bound_probability(
                sampling_bound,
                "failure_probability",
            ),
            "norm_floor": _sampling_bound_float(sampling_bound, "norm_floor"),
        }

    if sampling_bound.get("kind") != "abs_or_rel":
        message = (
            "sampled Fisher sampling_bound.kind must be disabled, abs_or_rel, "
            "matrix_bernstein, or hutchinson_relative_variance"
        )
        raise MaterializationError(message)

    return {
        "kind": "abs_or_rel",
        "max_abs_diff": _sampling_bound_float(sampling_bound, "max_abs_diff"),
        "max_rel_diff": _sampling_bound_float(sampling_bound, "max_rel_diff"),
        "norm_floor": _sampling_bound_float(sampling_bound, "norm_floor"),
    }


def _sampling_bound_float(sampling_bound: Mapping[str, Any], key: str) -> float:
    value = sampling_bound.get(key)

    if not isinstance(value, int | float) or isinstance(value, bool):
        message = f"sampled Fisher sampling_bound.{key} must be numeric"
        raise MaterializationError(message)

    result = float(value)

    if not math.isfinite(result) or result < 0.0:
        message = f"sampled Fisher sampling_bound.{key} must be finite and nonnegative"
        raise MaterializationError(message)

    return result


def _sampling_bound_probability(
    sampling_bound: Mapping[str, Any],
    key: str,
) -> float:
    result = _sampling_bound_float(sampling_bound, key)

    if 0.0 < result < 1.0:
        return result

    message = f"sampled Fisher sampling_bound.{key} must be in (0, 1)"
    raise MaterializationError(message)
