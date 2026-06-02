"""Runtime builders for package-owned operator anchors."""

import dataclasses
import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import torch

from vptune.admission import (
    FUNCTIONAL_CALL_FIELDS,
    TORCH_FUNC_FIELDS,
    admit_forward_ad,
    admit_functional_call,
    admit_torch_func,
)
from vptune.anchors import (
    dense_metric_inner,
    dense_metric_inverse_multiply,
    dense_metric_multiply,
    empirical_fisher_vp_dense_anchor,
    finite_difference_hvp,
    finite_difference_jvp,
    fisher_vp_dense_anchor,
    forward_ad_jvp_anchor,
    gradient_anchor,
    hvp_anchor,
    hvp_jvp_grad_anchor,
    hvp_reverse_over_reverse_anchor,
    jvp_anchor,
    vjp_anchor,
    vjp_dot_identity_error,
)
from vptune.candidates import standard_axis_registry
from vptune.checks import (
    apply_dtype_floors,
    tree_error_measurements,
    validate_thresholds,
)
from vptune.data import (
    PACKAGE_VERSION,
    Batch,
    BufferTree,
    CallableMaterializer,
    Candidate,
    CandidateAdmitter,
    CandidateOperation,
    DataProvider,
    FullSizeRecord,
    FunctionObjective,
    Materializer,
    ObjectiveContext,
    OperationFactory,
    OperatorSpec,
    ParameterSurface,
    ParameterTree,
    Problem,
    ReferenceCheck,
    ReferenceChildResult,
    ReferenceResult,
    RuntimeConfig,
    ScalarObjective,
    Target,
    VectorProvider,
)
from vptune.errors import AdmissionError, MaterializationError, ReferenceFailedError
from vptune.tensor_tree import (
    TensorTree,
    tree_dot,
    tree_from_leaves,
    tree_leaves,
    tree_map,
    tree_map2,
    tree_signature,
)

GRADIENT_PATH = "autograd_grad"
JVP_PATH = "torch_func_jvp"
JVP_FORWARD_AD_PATH = "forward_ad_jvp"
VJP_PATH = "torch_func_vjp"
HVP_REFERENCE_PATH = "reverse_over_reverse"
HVP_FUNCTIONAL_PATH = "functional_hvp"
HVP_JVP_GRAD_PATH = "jvp_grad"
VHP_PATH = "vhp"
GGN_DENSE_PATH = "dense_ggn"
GGN_JVP_HESSIAN_VJP_PATH = "jvp_hessian_vjp"
FISHER_DENSE_PATH = "dense_score_outer"
FISHER_CATEGORICAL_EXACT_PATH = "categorical_exact"
FISHER_CATEGORICAL_MC_PATH = "categorical_monte_carlo"
FISHER_SCORE_GRADIENT_LOOP_PATH = "score_gradient_loop"
EMPIRICAL_FISHER_DENSE_PATH = "dense_empirical_fisher"
EMPIRICAL_FISHER_GRADIENT_LOOP_PATH = "per_example_gradient_loop"
EMPIRICAL_FISHER_GRADIENT_VMAP_PATH = "per_example_gradient_vmap"
METRIC_DENSE_PATH = "dense_metric"
INVERSE_METRIC_DENSE_PATH = "dense_inverse_metric"
COMPOSITION_PATH = "sequential_composition"
RUNTIME_DTYPE_SETTINGS = ("model_dtype", "compute_dtype")
BACKEND_SETTINGS = (
    "matmul_precision",
    "allow_tf32",
    "allow_bf16_reduced_precision_reduction",
)
SUPPORTED_STANDARD_SETTINGS = (
    *RUNTIME_DTYPE_SETTINGS,
    "operator_path",
    *BACKEND_SETTINGS,
    *TORCH_FUNC_FIELDS,
    *FUNCTIONAL_CALL_FIELDS,
    "vmap_chunk_size",
    "vmap_batch_in_dims",
)
MATRIX_DIMS = 2
STANDARD_ANCHOR_PATHS = {
    "gradient": GRADIENT_PATH,
    "jvp": JVP_PATH,
    "vjp": VJP_PATH,
    "hvp": HVP_REFERENCE_PATH,
    "ggnvp": GGN_JVP_HESSIAN_VJP_PATH,
    "fisher_vp": FISHER_SCORE_GRADIENT_LOOP_PATH,
    "empirical_fisher_vp": EMPIRICAL_FISHER_GRADIENT_LOOP_PATH,
    "metric": METRIC_DENSE_PATH,
    "inverse_metric": INVERSE_METRIC_DENSE_PATH,
}
SINGLE_OPERATOR_PATHS = {
    "gradient": GRADIENT_PATH,
    "vjp": VJP_PATH,
    "metric": METRIC_DENSE_PATH,
    "inverse_metric": INVERSE_METRIC_DENSE_PATH,
}


@dataclasses.dataclass(frozen=True, slots=True)
class CompositionChild:
    """Child operator reference used by sequential composition."""

    name: str
    candidate: Candidate
    component: Callable[[Batch, TensorTree], TensorTree]
    anchor_component: Callable[[Batch, TensorTree], TensorTree]
    reference_check: ReferenceCheck
    input_signature: Mapping[str, Any]


def composition_operation_factory(
    operator: OperatorSpec,
    *,
    order: Sequence[str],
    components: Mapping[str, Callable[[Batch, TensorTree], TensorTree]],
) -> OperationFactory:
    """Return an operation factory for sequential operator composition."""
    order_tuple = _composition_order(order)
    component_map = dict(components)
    _require_composition_components(order_tuple, component_map)

    def factory(
        candidate: Candidate,
        batch: Batch,
        vector: TensorTree,
    ) -> CandidateOperation:
        _require_candidate_family(operator, candidate)

        def operation() -> TensorTree:
            _require_supported_standard_settings(operator, candidate)
            _require_path(
                operator.kind,
                _required_candidate_operator_path(candidate),
                (COMPOSITION_PATH,),
            )
            runtime_batch = _runtime_batch(batch, candidate.settings)
            result = _runtime_vector(vector, candidate.settings)

            def run_components() -> TensorTree:
                component_result = result

                for component_name in order_tuple:
                    component_result = component_map[component_name](
                        runtime_batch,
                        component_result,
                    )

                return component_result

            return _run_with_backend_settings(candidate.settings, run_components)

        return operation

    return factory


def composition_reference_check(
    operator: OperatorSpec,
    *,
    order: Sequence[str],
    components: Mapping[str, Callable[[Batch, TensorTree], TensorTree]],
    anchor_components: Mapping[str, Callable[[Batch, TensorTree], TensorTree]],
    thresholds: Mapping[str, float],
    children: Sequence[CompositionChild] = (),
) -> ReferenceCheck:
    """Return a reference check for sequential operator composition.

    Raises:
        MaterializationError: If thresholds are empty.
    """
    if not thresholds:
        message = "composition reference thresholds are required"
        raise MaterializationError(message)

    order_tuple = _composition_order(order)
    component_map = dict(components)
    anchor_component_map = dict(anchor_components)
    child_map = _composition_child_map(order_tuple, children)
    _require_composition_components(order_tuple, component_map)
    _require_composition_components(order_tuple, anchor_component_map)

    def check(
        candidate: Candidate,
        batch: Batch,
        vector: TensorTree,
    ) -> ReferenceResult:
        try:
            anchor_candidate = dataclasses.replace(
                candidate,
                settings=_anchor_settings(candidate, COMPOSITION_PATH),
            )
            candidate_output, anchor_output, component_errors, child_results = (
                _composition_reference_outputs(
                    operator,
                    candidate,
                    anchor_candidate,
                    batch,
                    vector,
                    order_tuple,
                    component_map,
                    anchor_component_map,
                    child_map,
                )
            )
        except (MaterializationError, ReferenceFailedError) as error:
            raise ReferenceFailedError(str(error)) from error

        measurements = tree_error_measurements(candidate_output, anchor_output)
        _merge_component_measurements(measurements, component_errors)
        measurements.update(
            _semantic_measurements(operator, batch, vector, candidate_output)
        )
        effective_thresholds = dict(thresholds)
        apply_dtype_floors(effective_thresholds, candidate.settings)
        _require_thresholds_for_measurements(measurements, effective_thresholds)
        validate_thresholds(measurements, effective_thresholds)

        return ReferenceResult(
            "composition_anchor",
            effective_thresholds,
            measurements,
            child_results=child_results,
        )

    return check


def _composition_reference_outputs(
    operator: OperatorSpec,
    candidate: Candidate,
    anchor_candidate: Candidate,
    batch: Batch,
    vector: TensorTree,
    order: tuple[str, ...],
    components: Mapping[str, Callable[[Batch, TensorTree], TensorTree]],
    anchor_components: Mapping[str, Callable[[Batch, TensorTree], TensorTree]],
    children: Mapping[str, CompositionChild],
) -> tuple[
    TensorTree,
    TensorTree,
    dict[str, dict[str, float]],
    tuple[ReferenceChildResult, ...],
]:
    _require_candidate_family(operator, candidate)
    _require_candidate_family(operator, anchor_candidate)
    _require_supported_standard_settings(operator, candidate)
    _require_supported_standard_settings(operator, anchor_candidate)
    _require_path(
        operator.kind,
        _required_candidate_operator_path(candidate),
        (COMPOSITION_PATH,),
    )
    _require_path(
        operator.kind,
        _required_candidate_operator_path(anchor_candidate),
        (COMPOSITION_PATH,),
    )
    candidate_batch = _runtime_batch(batch, candidate.settings)
    anchor_batch = _runtime_batch(batch, anchor_candidate.settings)
    candidate_result = _runtime_vector(vector, candidate.settings)
    anchor_result = _runtime_vector(vector, anchor_candidate.settings)
    component_errors = {}
    child_results = []

    for component_name in order:
        child = children.get(component_name)

        if child is not None:
            child_input_signature = {
                **dict(child.input_signature),
                "component": component_name,
                "component_input": tree_signature(candidate_result),
            }
            child_result = child.reference_check(
                child.candidate,
                candidate_batch,
                candidate_result,
            )
            child_results.append(
                ReferenceChildResult(
                    child.candidate,
                    child_input_signature,
                    child_result,
                )
            )

        candidate_result = components[component_name](
            candidate_batch,
            candidate_result,
        )
        anchor_result = anchor_components[component_name](
            anchor_batch,
            anchor_result,
        )
        component_errors[component_name] = tree_error_measurements(
            candidate_result,
            anchor_result,
        )

    return candidate_result, anchor_result, component_errors, tuple(child_results)


def _composition_child_map(
    order: tuple[str, ...],
    children: Sequence[CompositionChild],
) -> dict[str, CompositionChild]:
    child_map = {child.name: child for child in children}

    if len(child_map) != len(children):
        message = "composition child names must be unique"
        raise MaterializationError(message)

    unknown = tuple(name for name in child_map if name not in order)

    if unknown:
        message = f"composition child names are not in the order: {unknown}"
        raise MaterializationError(message)

    return child_map


def _merge_component_measurements(
    measurements: dict[str, Any],
    component_errors: Mapping[str, Mapping[str, float]],
) -> None:
    if not component_errors:
        return

    max_abs = float(measurements["max_abs_diff"])
    max_rel = float(measurements["max_rel_diff"])

    for errors in component_errors.values():
        max_abs = max(max_abs, float(errors["max_abs_diff"]))
        max_rel = max(max_rel, float(errors["max_rel_diff"]))

    measurements["max_abs_diff"] = max_abs
    measurements["max_rel_diff"] = max_rel
    measurements["component_errors"] = {
        name: dict(errors) for name, errors in component_errors.items()
    }


def _require_thresholds_for_measurements(
    measurements: Mapping[str, Any],
    thresholds: Mapping[str, float],
) -> None:
    exact_required = ("psd_violation",)
    missing = tuple(
        key
        for key in measurements
        if key.endswith(("_diff", "_residual")) or key in exact_required
        if key not in thresholds
    )

    if missing:
        message = f"reference thresholds are missing measurements: {missing}"
        raise ReferenceFailedError(message)


def _require_vhp_reference_policy(
    candidate: Candidate,
    batch: Batch,
    thresholds: Mapping[str, float],
) -> None:
    if candidate.settings.get("operator_path") != VHP_PATH:
        return

    required_thresholds = (
        "symmetry_max_abs_diff",
        "directional_abs_diff",
        "directional_rel_diff",
    )
    missing_thresholds = tuple(
        threshold for threshold in required_thresholds if threshold not in thresholds
    )

    if missing_thresholds:
        message = f"vhp reference thresholds are missing: {missing_thresholds}"
        raise ReferenceFailedError(message)

    _batch_tree(batch, "symmetry_vector")


def _semantic_measurements(
    operator: OperatorSpec,
    batch: Batch,
    vector: TensorTree,
    output: TensorTree,
) -> dict[str, float]:
    if operator.kind == "ggnvp":
        if _ggn_loss_geometry(operator) != "psd_metric":
            return {}

        loss_hessian = _batch_tensor(batch, "loss_hessian")

        return {
            "symmetry_max_abs_diff": _matrix_symmetry_error(loss_hessian),
            "psd_violation": _matrix_psd_violation(loss_hessian),
        }

    if operator.kind == "metric":
        matrix = _batch_tensor(batch, "metric")

        return {
            "symmetry_max_abs_diff": _matrix_symmetry_error(matrix),
            "psd_violation": _matrix_psd_violation(matrix),
        }

    if operator.kind == "inverse_metric":
        matrix = _batch_tensor(batch, "metric")
        vector_tensor = _flatten_vector(vector)
        output_tensor = _flatten_vector(output)

        return {
            "symmetry_max_abs_diff": _matrix_symmetry_error(matrix),
            "psd_violation": _matrix_psd_violation(matrix),
            "inverse_residual": _inverse_residual(
                matrix,
                output_tensor,
                vector_tensor,
            ),
        }

    return {}


def _first_order_reference_measurements(
    operator: OperatorSpec,
    candidate: Candidate,
    batch: Batch,
    vector: TensorTree,
    candidate_output: TensorTree,
    *,
    params: ParameterTree,
    buffers: BufferTree,
    scalar_objectives: Mapping[str, ScalarObjective],
    function_objectives: Mapping[str, FunctionObjective],
) -> dict[str, float]:
    if operator.kind == "gradient":
        scalar = _scalar_objective(operator, scalar_objectives)
        context = ObjectiveContext(
            family=operator.family,
            candidate_id=candidate.candidate_id,
            settings=dict(candidate.settings),
        )

        def scalar_function(active_params: ParameterTree) -> torch.Tensor:
            return scalar(active_params, buffers, batch, context)

        finite_difference = finite_difference_jvp(scalar_function, params, vector)
        directional = tree_dot(candidate_output, vector)
        errors = tree_error_measurements(directional, finite_difference)

        return {
            "directional_abs_diff": errors["max_abs_diff"],
            "directional_rel_diff": errors["max_rel_diff"],
        }

    if operator.kind == "jvp":
        function = _function_objective(operator, function_objectives)
        context = ObjectiveContext(
            family=operator.family,
            candidate_id=candidate.candidate_id,
            settings=dict(candidate.settings),
        )

        def tensor_function(active_params: ParameterTree) -> TensorTree:
            return function(active_params, buffers, batch, context)

        finite_difference = finite_difference_jvp(tensor_function, params, vector)
        errors = tree_error_measurements(candidate_output, finite_difference)

        return {
            "directional_abs_diff": errors["max_abs_diff"],
            "directional_rel_diff": errors["max_rel_diff"],
        }

    if operator.kind == "vjp":
        function = _function_objective(operator, function_objectives)
        tangent = _batch_tree(batch, "tangent_vector")
        context = ObjectiveContext(
            family=operator.family,
            candidate_id=candidate.candidate_id,
            settings=dict(candidate.settings),
        )

        def tensor_function(active_params: ParameterTree) -> TensorTree:
            return function(active_params, buffers, batch, context)

        return {
            "inner_abs_diff": float(
                vjp_dot_identity_error(tensor_function, params, tangent, vector).item()
            )
        }

    return {}


def _hvp_finite_difference_measurements(
    operator: OperatorSpec,
    candidate: Candidate,
    batch: Batch,
    vector: TensorTree,
    candidate_output: TensorTree,
    *,
    params: ParameterTree,
    buffers: BufferTree,
    scalar_objectives: Mapping[str, ScalarObjective],
    anchor_candidate: Candidate,
    candidate_factory: OperationFactory,
) -> dict[str, float]:
    if operator.kind != "hvp":
        return {}

    scalar = _scalar_objective(operator, scalar_objectives)
    context = ObjectiveContext(
        family=operator.family,
        candidate_id=candidate.candidate_id,
        settings=dict(candidate.settings),
    )

    def scalar_function(active_params: ParameterTree) -> torch.Tensor:
        return scalar(active_params, buffers, batch, context)

    finite_difference = finite_difference_hvp(
        scalar_function,
        params,
        vector,
    )
    errors = tree_error_measurements(candidate_output, finite_difference)

    measurements = {
        "directional_abs_diff": errors["max_abs_diff"],
        "directional_rel_diff": errors["max_rel_diff"],
    }

    symmetry_vector = _batch_tree(batch, "symmetry_vector")
    anchor_symmetry = candidate_factory(
        anchor_candidate,
        batch,
        symmetry_vector,
    )()
    left = tree_dot(
        _runtime_vector(symmetry_vector, candidate.settings),
        candidate_output,
    )
    right = tree_dot(vector, anchor_symmetry)
    measurements["symmetry_max_abs_diff"] = float((left - right).abs().item())

    return measurements


def _ggn_inner_product_measurements(
    operator: OperatorSpec,
    candidate: Candidate,
    batch: Batch,
    vector: TensorTree,
    candidate_output: TensorTree,
    anchor_candidate: Candidate,
    candidate_factory: OperationFactory,
) -> dict[str, float]:
    if operator.kind != "ggnvp" or _ggn_loss_geometry(operator) != "psd_metric":
        return {}

    symmetry_vector = _batch_tree(batch, "symmetry_vector")
    anchor_symmetry = candidate_factory(
        anchor_candidate,
        batch,
        symmetry_vector,
    )()
    left = tree_dot(
        _runtime_vector(symmetry_vector, candidate.settings),
        candidate_output,
    )
    right = tree_dot(vector, anchor_symmetry)

    return {"inner_abs_diff": float((left - right).abs().item())}


def _matrix_symmetry_error(matrix: torch.Tensor) -> float:
    if matrix.ndim != MATRIX_DIMS or matrix.shape[0] != matrix.shape[1]:
        message = "metric matrix must be square"
        raise MaterializationError(message)

    _require_finite_tensor(matrix, "metric matrix")

    return float((matrix - matrix.T).abs().max().item())


def _matrix_psd_violation(matrix: torch.Tensor) -> float:
    if matrix.ndim != MATRIX_DIMS or matrix.shape[0] != matrix.shape[1]:
        message = "metric matrix must be square"
        raise MaterializationError(message)

    _require_finite_tensor(matrix, "metric matrix")

    min_eigenvalue = torch.linalg.eigvalsh(matrix).min()

    return float(torch.clamp(-min_eigenvalue, min=0.0).item())


def _inverse_residual(
    matrix: torch.Tensor,
    inverse_result: torch.Tensor,
    vector: torch.Tensor,
) -> float:
    _require_finite_tensor(matrix, "metric matrix")
    _require_finite_tensor(inverse_result, "inverse result")
    _require_finite_tensor(vector, "metric vector")

    residual = (matrix @ inverse_result.reshape(-1) - vector.reshape(-1)).norm()
    denominator = vector.reshape(-1).norm()

    if math.isclose(float(denominator.item()), 0.0, rel_tol=0.0, abs_tol=0.0):
        return float(residual.item())

    return float((residual / denominator).item())


def _require_finite_tensor(tensor: torch.Tensor, name: str) -> None:
    if not torch.isfinite(tensor).all().item():
        message = f"{name} contains nonfinite values"
        raise MaterializationError(message)


def _require_finite_tree(tree: TensorTree, name: str) -> None:
    for leaf in tree_leaves(tree):
        _require_finite_tensor(leaf, name)


def composition_runtime_config(
    operator: OperatorSpec,
    *,
    order: Sequence[str],
    components: Mapping[str, Callable[[Batch, TensorTree], TensorTree]],
    anchor_components: Mapping[str, Callable[[Batch, TensorTree], TensorTree]],
    children: Sequence[CompositionChild] = (),
    candidates: Sequence[Candidate],
    thresholds: Mapping[str, float],
    component_signature: Mapping[str, Any],
    anchor_component_signature: Mapping[str, Any],
    axis_registry: CandidateAdmitter | None,
) -> RuntimeConfig:
    """Return runtime config for sequential operator composition."""
    operation_factory = composition_operation_factory(
        operator,
        order=order,
        components=components,
    )
    reference_check = composition_reference_check(
        operator,
        order=order,
        components=components,
        anchor_components=anchor_components,
        thresholds=thresholds,
        children=children,
    )

    return RuntimeConfig(
        candidates=tuple(candidates),
        operation_factory=operation_factory,
        reference_check=reference_check,
        materializer=_standard_materializer(operation_factory),
        axis_registry=axis_registry,
        signature={
            "runtime": "composition",
            "operator": operator.signature(),
            "order": _composition_order(order),
            "thresholds": dict(thresholds),
            "components": {
                "candidate": dict(component_signature),
                "anchor": dict(anchor_component_signature),
            },
            "children": tuple(child.name for child in children),
        },
    )


@dataclasses.dataclass(frozen=True, slots=True)
class StandardExecution:
    """Inputs for one standard operator execution."""

    operator: OperatorSpec
    candidate: Candidate
    path: str
    batch: Batch
    vector: TensorTree
    params: ParameterTree
    buffers: BufferTree
    context: ObjectiveContext
    scalar_objectives: Mapping[str, ScalarObjective]
    function_objectives: Mapping[str, FunctionObjective]


def standard_operation_factory(
    operator: OperatorSpec,
    *,
    params: ParameterTree,
    buffers: BufferTree,
    scalar_objectives: Mapping[str, ScalarObjective] | None = None,
    function_objectives: Mapping[str, FunctionObjective] | None = None,
) -> OperationFactory:
    """Return an operation factory for package-owned standard operators."""
    scalar_map = {} if scalar_objectives is None else dict(scalar_objectives)
    function_map = {} if function_objectives is None else dict(function_objectives)

    def factory(
        candidate: Candidate,
        batch: Batch,
        vector: TensorTree,
    ) -> CandidateOperation:
        _require_candidate_family(operator, candidate)
        _require_supported_standard_settings(operator, candidate)
        path = _operator_path(operator, candidate)
        _require_batch_inputs(operator, candidate, batch, phase="operation")
        runtime_params = _runtime_params(params, candidate.settings)
        runtime_buffers = _runtime_buffers(buffers, candidate.settings)
        runtime_batch = _runtime_batch(batch, candidate.settings)
        runtime_vector = _runtime_vector(vector, candidate.settings)
        context = ObjectiveContext(
            family=operator.family,
            candidate_id=candidate.candidate_id,
            settings=dict(candidate.settings),
        )
        execution = StandardExecution(
            operator=operator,
            candidate=candidate,
            path=path,
            batch=runtime_batch,
            vector=runtime_vector,
            params=runtime_params,
            buffers=runtime_buffers,
            context=context,
            scalar_objectives=scalar_map,
            function_objectives=function_map,
        )

        def operation() -> TensorTree:
            return _run_with_backend_settings(
                candidate.settings,
                lambda: _run_standard_operation(execution),
            )

        return operation

    return factory


def standard_reference_check(
    operator: OperatorSpec,
    *,
    params: ParameterTree,
    buffers: BufferTree,
    thresholds: Mapping[str, float],
    scalar_objectives: Mapping[str, ScalarObjective] | None = None,
    function_objectives: Mapping[str, FunctionObjective] | None = None,
) -> ReferenceCheck:
    """Return a reference check backed by package-owned anchors.

    Raises:
        MaterializationError: If thresholds are empty.
    """
    if not thresholds:
        message = "reference thresholds are required"
        raise MaterializationError(message)

    scalar_map = {} if scalar_objectives is None else dict(scalar_objectives)
    function_map = {} if function_objectives is None else dict(function_objectives)
    candidate_factory = standard_operation_factory(
        operator,
        params=params,
        buffers=buffers,
        scalar_objectives=scalar_map,
        function_objectives=function_map,
    )

    def check(
        candidate: Candidate,
        batch: Batch,
        vector: TensorTree,
    ) -> ReferenceResult:
        try:
            anchor_candidate, candidate_output, anchor_output = (
                _standard_reference_outputs(
                    operator,
                    candidate,
                    batch,
                    vector,
                    thresholds,
                    candidate_factory,
                )
            )
        except ReferenceFailedError:
            raise
        except RuntimeError as error:
            raise ReferenceFailedError(str(error)) from error

        try:
            measurements = _standard_reference_measurements(
                operator,
                candidate,
                anchor_candidate,
                batch,
                vector,
                candidate_output,
                anchor_output,
                candidate_factory,
                params=params,
                buffers=buffers,
                scalar_objectives=scalar_map,
                function_objectives=function_map,
            )
        except ReferenceFailedError:
            raise
        except RuntimeError as error:
            raise ReferenceFailedError(str(error)) from error

        effective_thresholds = dict(thresholds)
        apply_dtype_floors(effective_thresholds, candidate.settings)
        _require_thresholds_for_measurements(measurements, effective_thresholds)
        validate_thresholds(measurements, effective_thresholds)

        return ReferenceResult(
            "standard_anchor",
            effective_thresholds,
            measurements,
        )

    return check


def _standard_reference_outputs(
    operator: OperatorSpec,
    candidate: Candidate,
    batch: Batch,
    vector: TensorTree,
    thresholds: Mapping[str, float],
    candidate_factory: OperationFactory,
) -> tuple[Candidate, TensorTree, TensorTree]:
    anchor_candidate = _anchor_candidate(operator, candidate)
    _require_batch_inputs(operator, candidate, batch, phase="reference")
    _require_batch_inputs(operator, anchor_candidate, batch, phase="reference")
    _require_vhp_reference_policy(candidate, batch, thresholds)
    candidate_output = candidate_factory(candidate, batch, vector)()
    anchor_output = candidate_factory(anchor_candidate, batch, vector)()

    return anchor_candidate, candidate_output, anchor_output


def _require_batch_inputs(
    operator: OperatorSpec,
    candidate: Candidate,
    batch: Batch,
    *,
    phase: str,
) -> None:
    required = _required_batch_inputs(operator, candidate, phase)
    missing = tuple(key for key in required if key not in batch)

    if missing:
        message = (
            f"batch inputs missing for {operator.family}/{candidate.candidate_id}/"
            f"{phase}: {missing}"
        )
        raise MaterializationError(message)


def _required_batch_inputs(
    operator: OperatorSpec,
    candidate: Candidate,
    phase: str,
) -> tuple[str, ...]:
    declared = operator.batch_inputs.get(phase)

    if declared is None:
        message = f"operator batch_inputs must declare {phase}"
        raise MaterializationError(message)

    path_inputs = _candidate_batch_inputs(operator, candidate)

    return tuple(dict.fromkeys((*declared, *path_inputs)))


def _candidate_batch_inputs(
    operator: OperatorSpec,
    candidate: Candidate,
) -> tuple[str, ...]:
    path = _operator_path(operator, candidate)

    if operator.kind == "fisher_vp":
        return _fisher_batch_inputs(operator, path)

    if operator.kind == "empirical_fisher_vp":
        return _empirical_fisher_batch_inputs(operator, path)

    return ()


def _fisher_batch_inputs(operator: OperatorSpec, path: str) -> tuple[str, ...]:
    if path not in {
        FISHER_DENSE_PATH,
        FISHER_CATEGORICAL_EXACT_PATH,
        FISHER_CATEGORICAL_MC_PATH,
        FISHER_SCORE_GRADIENT_LOOP_PATH,
    }:
        return ()

    inputs = ()

    if path == FISHER_DENSE_PATH:
        inputs = ("score_gradients",)

    return (*inputs, *_fisher_denominator_batch_inputs(operator, path))


def _fisher_denominator_batch_inputs(
    operator: OperatorSpec,
    path: str,
) -> tuple[str, ...]:
    denominator = _operator_semantic(operator, "denominator")

    if denominator == "batch_normalization":
        return ("normalization",)

    if denominator == "num_examples" and path not in {
        FISHER_CATEGORICAL_EXACT_PATH,
        FISHER_CATEGORICAL_MC_PATH,
    }:
        return ("num_examples",)

    return ()


def _empirical_fisher_batch_inputs(
    operator: OperatorSpec,
    path: str,
) -> tuple[str, ...]:
    if path not in {
        EMPIRICAL_FISHER_DENSE_PATH,
        EMPIRICAL_FISHER_GRADIENT_LOOP_PATH,
        EMPIRICAL_FISHER_GRADIENT_VMAP_PATH,
    }:
        return ()

    inputs = ()

    if path == EMPIRICAL_FISHER_DENSE_PATH:
        inputs = ("per_example_gradients",)

    if _operator_semantic(operator, "denominator") == "batch_normalization":
        return (*inputs, "normalization")

    return inputs


def _standard_reference_measurements(
    operator: OperatorSpec,
    candidate: Candidate,
    anchor_candidate: Candidate,
    batch: Batch,
    vector: TensorTree,
    candidate_output: TensorTree,
    anchor_output: TensorTree,
    candidate_factory: OperationFactory,
    *,
    params: ParameterTree,
    buffers: BufferTree,
    scalar_objectives: Mapping[str, ScalarObjective],
    function_objectives: Mapping[str, FunctionObjective],
) -> dict[str, Any]:
    measurements = tree_error_measurements(candidate_output, anchor_output)
    _augment_ggn_dense_cross_check(
        operator,
        candidate,
        batch,
        vector,
        candidate_output,
        candidate_factory,
        measurements,
    )
    measurements.update(
        _semantic_measurements(operator, batch, vector, candidate_output)
    )
    measurements.update(
        _ggn_inner_product_measurements(
            operator,
            candidate,
            batch,
            vector,
            candidate_output,
            anchor_candidate,
            candidate_factory,
        )
    )
    measurements.update(
        _first_order_reference_measurements(
            operator,
            candidate,
            batch,
            vector,
            candidate_output,
            params=params,
            buffers=buffers,
            scalar_objectives=scalar_objectives,
            function_objectives=function_objectives,
        )
    )
    measurements.update(
        _hvp_finite_difference_measurements(
            operator,
            candidate,
            batch,
            vector,
            candidate_output,
            params=params,
            buffers=buffers,
            scalar_objectives=scalar_objectives,
            anchor_candidate=anchor_candidate,
            candidate_factory=candidate_factory,
        )
    )

    return measurements


def _augment_ggn_dense_cross_check(
    operator: OperatorSpec,
    candidate: Candidate,
    batch: Batch,
    vector: TensorTree,
    candidate_output: TensorTree,
    candidate_factory: OperationFactory,
    measurements: dict[str, Any],
) -> None:
    if operator.kind != "ggnvp":
        return

    dense_candidate = dataclasses.replace(
        candidate,
        settings=_anchor_settings(candidate, GGN_DENSE_PATH),
    )
    dense_output = candidate_factory(dense_candidate, batch, vector)()
    errors = tree_error_measurements(candidate_output, dense_output)
    measurements["max_abs_diff"] = max(
        float(measurements["max_abs_diff"]),
        float(errors["max_abs_diff"]),
    )
    measurements["max_rel_diff"] = max(
        float(measurements["max_rel_diff"]),
        float(errors["max_rel_diff"]),
    )
    measurements["dense_anchor_errors"] = dict(errors)


def standard_runtime_config(
    operator: OperatorSpec,
    *,
    params: ParameterTree,
    buffers: BufferTree,
    candidates: Sequence[Candidate],
    thresholds: Mapping[str, float],
    objective_signature: Mapping[str, Any],
    axis_registry: CandidateAdmitter | None,
    scalar_objectives: Mapping[str, ScalarObjective] | None = None,
    function_objectives: Mapping[str, FunctionObjective] | None = None,
) -> RuntimeConfig:
    """Return runtime config for package-owned standard operators."""
    operation_factory = standard_operation_factory(
        operator,
        params=params,
        buffers=buffers,
        scalar_objectives=scalar_objectives,
        function_objectives=function_objectives,
    )
    reference_check = standard_reference_check(
        operator,
        params=params,
        buffers=buffers,
        thresholds=thresholds,
        scalar_objectives=scalar_objectives,
        function_objectives=function_objectives,
    )
    materializer = _standard_materializer(operation_factory, operator)

    return RuntimeConfig(
        candidates=tuple(candidates),
        operation_factory=operation_factory,
        reference_check=reference_check,
        materializer=materializer,
        axis_registry=axis_registry,
        signature={
            "runtime": "standard",
            "operator": operator.signature(),
            "params": tree_signature(params),
            "buffers": tree_signature(buffers),
            "thresholds": dict(thresholds),
            "objective": dict(objective_signature),
        },
    )


def standard_problem(
    *,
    model: torch.nn.Module,
    parameter_surface: ParameterSurface,
    parameter_values: ParameterTree,
    buffers: BufferTree,
    operator: OperatorSpec,
    data: DataProvider,
    vectors: VectorProvider,
    target: Target,
    candidates: Mapping[str, Mapping[str, Any]],
    thresholds: Mapping[str, float],
    objective_signature: Mapping[str, Any],
    scalar_objectives: Mapping[str, ScalarObjective] | None = None,
    function_objectives: Mapping[str, FunctionObjective] | None = None,
) -> Problem:
    """Return a standard PyTorch tuning problem from explicit settings.

    Raises:
        MaterializationError: If candidate settings are empty.
    """
    if not candidates:
        message = "problem candidate_settings are required"
        raise MaterializationError(message)

    candidate_rows = tuple(
        Candidate(
            family=operator.family,
            candidate_id=candidate_id,
            settings=dict(settings),
            changed_axes=tuple(settings),
            generator_id="vptune.problem",
            generator_version=PACKAGE_VERSION,
        )
        for candidate_id, settings in candidates.items()
    )
    runtime = standard_runtime_config(
        operator,
        params=parameter_values,
        buffers=buffers,
        candidates=candidate_rows,
        thresholds=thresholds,
        objective_signature=objective_signature,
        axis_registry=standard_axis_registry(),
        scalar_objectives=scalar_objectives,
        function_objectives=function_objectives,
    )

    return Problem(
        model=model,
        params=parameter_surface,
        data=data,
        operator=operator,
        vectors=vectors,
        target=target,
        runtime=runtime,
        anchor_policy={},
        replay_policy={},
        adapter_identity={
            "adapter_id": "vptune.core",
            "adapter_version": PACKAGE_VERSION,
        },
    )


def _run_standard_operation(execution: StandardExecution) -> TensorTree:
    runner = STANDARD_RUNNERS.get(execution.operator.kind)

    if runner is None:
        message = (
            "standard runtime does not support operator kind: "
            f"{execution.operator.kind}"
        )
        raise MaterializationError(message)

    return runner(execution)


def _composition_order(order: Sequence[str]) -> tuple[str, ...]:
    """Return validated composition order.

    Raises:
        MaterializationError: If the order is empty or contains duplicate names.
    """
    order_result = tuple(order)

    if not order_result:
        message = "composition order must be non-empty"
        raise MaterializationError(message)

    if len(set(order_result)) != len(order_result):
        message = "composition order contains duplicate component names"
        raise MaterializationError(message)

    return order_result


def _require_composition_components(
    order: tuple[str, ...],
    components: Mapping[str, Callable[[Batch, TensorTree], TensorTree]],
) -> None:
    if set(components) != set(order):
        message = "composition components must match composition order"
        raise MaterializationError(message)


def _run_gradient(execution: StandardExecution) -> TensorTree:
    _require_single_operator_path(execution)
    scalar = _scalar_objective(execution.operator, execution.scalar_objectives)

    def scalar_function(active_params: ParameterTree) -> torch.Tensor:
        return scalar(
            active_params,
            execution.buffers,
            execution.batch,
            execution.context,
        )

    return gradient_anchor(scalar_function, execution.params)


def _run_jvp(execution: StandardExecution) -> TensorTree:
    _require_path(
        execution.operator.kind,
        execution.path,
        (JVP_PATH, JVP_FORWARD_AD_PATH),
    )
    function = _function_objective(execution.operator, execution.function_objectives)

    def tensor_function(active_params: ParameterTree) -> TensorTree:
        return function(
            active_params,
            execution.buffers,
            execution.batch,
            execution.context,
        )

    if execution.path == JVP_FORWARD_AD_PATH:
        return forward_ad_jvp_anchor(
            tensor_function,
            execution.params,
            execution.vector,
        )

    return jvp_anchor(tensor_function, execution.params, execution.vector)


def _run_vjp(execution: StandardExecution) -> TensorTree:
    _require_single_operator_path(execution)
    function = _function_objective(execution.operator, execution.function_objectives)

    def tensor_function(active_params: ParameterTree) -> TensorTree:
        return function(
            active_params,
            execution.buffers,
            execution.batch,
            execution.context,
        )

    return vjp_anchor(tensor_function, execution.params, execution.vector)


def _run_hvp(execution: StandardExecution) -> TensorTree:
    _require_path(
        execution.operator.kind,
        execution.path,
        (HVP_REFERENCE_PATH, HVP_FUNCTIONAL_PATH, HVP_JVP_GRAD_PATH, VHP_PATH),
    )
    scalar = _scalar_objective(execution.operator, execution.scalar_objectives)

    def scalar_function(active_params: ParameterTree) -> torch.Tensor:
        return scalar(
            active_params,
            execution.buffers,
            execution.batch,
            execution.context,
        )

    if execution.path == HVP_REFERENCE_PATH:
        return hvp_reverse_over_reverse_anchor(
            scalar_function,
            execution.params,
            execution.vector,
        )

    if execution.path == HVP_FUNCTIONAL_PATH:
        return hvp_anchor(scalar_function, execution.params, execution.vector)

    if execution.path == VHP_PATH:
        return _run_hvp_vhp_path(execution, scalar)

    return hvp_jvp_grad_anchor(scalar_function, execution.params, execution.vector)


def _run_hvp_vhp_path(
    execution: StandardExecution,
    scalar: ScalarObjective,
) -> TensorTree:
    parameter_items = tuple(execution.params.items())
    vector_leaves = _matching_vector_leaves(execution.params, execution.vector)
    parameter_leaves = tuple(tensor for _, tensor in parameter_items)

    def scalar_function(*active_leaves: torch.Tensor) -> torch.Tensor:
        active_params = {
            name: active
            for (name, _), active in zip(parameter_items, active_leaves, strict=True)
        }

        return scalar(
            active_params,
            execution.buffers,
            execution.batch,
            execution.context,
        )

    _, result_leaves = torch.autograd.functional.vhp(
        scalar_function,
        parameter_leaves,
        vector_leaves,
    )

    return tree_from_leaves(execution.params, result_leaves)


def _run_ggnvp(execution: StandardExecution) -> TensorTree:
    _require_path(
        execution.operator.kind,
        execution.path,
        (GGN_DENSE_PATH, GGN_JVP_HESSIAN_VJP_PATH),
    )
    _ggn_loss_geometry(execution.operator)

    if execution.path == GGN_JVP_HESSIAN_VJP_PATH:
        return _run_ggnvp_jvp_hessian_vjp(execution)

    function = _function_objective(execution.operator, execution.function_objectives)
    parameter_items = tuple(execution.params.items())
    parameter_leaves = tuple(tensor for _, tensor in parameter_items)
    vector_leaves = _matching_vector_leaves(execution.params, execution.vector)
    vector_tensor = torch.cat(tuple(leaf.reshape(-1) for leaf in vector_leaves))
    loss_hessian = _batch_tensor(execution.batch, "loss_hessian")

    def tensor_function(*active_leaves: torch.Tensor) -> torch.Tensor:
        active_params = {
            name: active
            for (name, _), active in zip(parameter_items, active_leaves, strict=True)
        }
        output = function(
            active_params,
            execution.buffers,
            execution.batch,
            execution.context,
        )

        if not isinstance(output, torch.Tensor):
            message = "dense GGNVP requires tensor function output"
            raise MaterializationError(message)

        return output.reshape(-1)

    output = tensor_function(*parameter_leaves)
    _require_loss_hessian_shape(loss_hessian, output.numel())
    _require_finite_tensor(loss_hessian, "loss_hessian")
    _require_finite_tensor(vector_tensor, "GGN vector")
    jacobian = _dense_jacobian_tree(
        tensor_function,
        parameter_leaves,
        output.numel(),
    )
    _require_finite_tensor(jacobian, "GGN jacobian")
    result = jacobian.T @ (loss_hessian @ (jacobian @ vector_tensor.reshape(-1)))
    _require_finite_tensor(result, "GGN result")

    return _wrap_flat_vector(execution.params, result)


def _run_ggnvp_jvp_hessian_vjp(execution: StandardExecution) -> TensorTree:
    function = _function_objective(execution.operator, execution.function_objectives)
    loss_hessian = _batch_tensor(execution.batch, "loss_hessian")

    def tensor_function(active_params: ParameterTree) -> TensorTree:
        return function(
            active_params,
            execution.buffers,
            execution.batch,
            execution.context,
        )

    output = tensor_function(execution.params)
    output_jvp = jvp_anchor(tensor_function, execution.params, execution.vector)
    flat_output_jvp = _flatten_vector(output_jvp)
    _require_loss_hessian_shape(loss_hessian, flat_output_jvp.numel())
    _require_finite_tensor(loss_hessian, "loss_hessian")
    _require_finite_tensor(flat_output_jvp, "GGN output JVP")
    output_cotangent = _wrap_flat_vector(
        output,
        loss_hessian @ flat_output_jvp.reshape(-1),
    )
    _require_finite_tree(output_cotangent, "GGN output cotangent")

    result = vjp_anchor(tensor_function, execution.params, output_cotangent)
    _require_finite_tree(result, "GGN result")

    return result


def _dense_jacobian_tree(
    function: Callable[..., torch.Tensor],
    parameter_leaves: tuple[torch.Tensor, ...],
    output_numel: int,
) -> torch.Tensor:
    jacobian = torch.autograd.functional.jacobian(function, parameter_leaves)
    jacobian_leaves = (jacobian,) if isinstance(jacobian, torch.Tensor) else jacobian
    parts = tuple(
        part.reshape(output_numel, parameter.numel())
        for part, parameter in zip(jacobian_leaves, parameter_leaves, strict=True)
    )

    return torch.cat(parts, dim=1)


def _require_loss_hessian_shape(
    loss_hessian: torch.Tensor,
    output_numel: int,
) -> None:
    if loss_hessian.shape != (output_numel, output_numel):
        message = "loss_hessian shape must match flattened function output"
        raise MaterializationError(message)


def _run_fisher_vp(execution: StandardExecution) -> TensorTree:
    _require_path(
        execution.operator.kind,
        execution.path,
        (
            FISHER_DENSE_PATH,
            FISHER_CATEGORICAL_EXACT_PATH,
            FISHER_CATEGORICAL_MC_PATH,
            FISHER_SCORE_GRADIENT_LOOP_PATH,
        ),
    )

    if execution.path == FISHER_CATEGORICAL_EXACT_PATH:
        score_gradients = _categorical_fisher_gradient_matrix(execution)
    elif execution.path == FISHER_CATEGORICAL_MC_PATH:
        score_gradients = _categorical_monte_carlo_fisher_gradient_matrix(execution)
    elif execution.path == FISHER_SCORE_GRADIENT_LOOP_PATH:
        _require_explicit_score_fisher_semantics(execution.operator)
        score_gradients = _per_example_gradient_matrix(execution)
    else:
        _require_valid_fisher_semantics(execution.operator)
        score_gradients = _batch_tensor(execution.batch, "score_gradients")

    vector_leaves = _matching_vector_leaves(execution.params, execution.vector)
    vector_tensor = torch.cat(tuple(leaf.reshape(-1) for leaf in vector_leaves))
    _require_finite_tensor(score_gradients, "score_gradients")
    _require_finite_tensor(vector_tensor, "Fisher vector")
    result = fisher_vp_dense_anchor(
        score_gradients,
        vector_tensor,
        normalization=_fisher_normalization(execution),
    )
    _require_finite_tensor(result, "Fisher result")

    return _wrap_flat_vector(execution.params, result)


def _categorical_fisher_gradient_matrix(execution: StandardExecution) -> torch.Tensor:
    _require_fisher_semantics(
        execution.operator,
        {
            "distribution": "categorical",
            "label_policy": "model_distribution",
            "expectation": "exact",
            "sample_space": "classes",
            "loss_reduction": "log_prob",
        },
    )
    function = _function_objective(execution.operator, execution.function_objectives)
    parameter_items = tuple(execution.params.items())
    active_leaves = tuple(
        tensor.detach().clone().requires_grad_(True) for _, tensor in parameter_items
    )

    def logits_function(*leaves: torch.Tensor) -> torch.Tensor:
        active_params = {
            name: leaf for (name, _), leaf in zip(parameter_items, leaves, strict=True)
        }
        logits = function(
            active_params,
            execution.buffers,
            execution.batch,
            execution.context,
        )

        return _categorical_logits_matrix(execution.operator, logits)

    logits_matrix = logits_function(*active_leaves)
    _require_finite_tensor(logits_matrix, "categorical Fisher logits")
    log_probs = torch.log_softmax(logits_matrix, dim=-1)
    probabilities = log_probs.exp().detach()
    gradient_rows = []

    for row in range(log_probs.shape[0]):
        for label in range(log_probs.shape[1]):
            term = log_probs[row, label]
            gradient_result = torch.autograd.grad(
                term,
                active_leaves,
                retain_graph=row < log_probs.shape[0] - 1
                or label < log_probs.shape[1] - 1,
                allow_unused=True,
            )
            gradients = tuple(
                torch.zeros_like(leaf) if gradient is None else gradient.detach()
                for leaf, gradient in zip(
                    active_leaves,
                    gradient_result,
                    strict=True,
                )
            )
            weight = torch.sqrt(probabilities[row, label])
            gradient_rows.append(
                weight
                * torch.cat(tuple(gradient.reshape(-1) for gradient in gradients))
            )

    return torch.stack(gradient_rows)


def _categorical_monte_carlo_fisher_gradient_matrix(
    execution: StandardExecution,
) -> torch.Tensor:
    _require_fisher_semantics(
        execution.operator,
        {
            "distribution": "categorical",
            "label_policy": "model_distribution",
            "expectation": "monte_carlo",
            "sample_space": "classes",
            "loss_reduction": "log_prob",
        },
    )
    sample_count = _operator_semantic_positive_int(execution.operator, "sample_count")
    seed = _operator_semantic_int(execution.operator, "seed")
    function = _function_objective(execution.operator, execution.function_objectives)
    parameter_items = tuple(execution.params.items())
    active_leaves = tuple(
        tensor.detach().clone().requires_grad_(True) for _, tensor in parameter_items
    )

    def logits_function(*leaves: torch.Tensor) -> torch.Tensor:
        active_params = {
            name: leaf for (name, _), leaf in zip(parameter_items, leaves, strict=True)
        }
        logits = function(
            active_params,
            execution.buffers,
            execution.batch,
            execution.context,
        )

        return _categorical_logits_matrix(execution.operator, logits)

    logits_matrix = logits_function(*active_leaves)
    _require_finite_tensor(logits_matrix, "Monte Carlo Fisher logits")
    log_probs = torch.log_softmax(logits_matrix, dim=-1)
    probabilities = log_probs.exp().detach()
    generator = torch.Generator(device=probabilities.device)
    generator.manual_seed(seed)
    labels = torch.multinomial(
        probabilities,
        sample_count,
        replacement=True,
        generator=generator,
    )

    return _monte_carlo_fisher_rows(log_probs, labels, active_leaves, sample_count)


def _monte_carlo_fisher_rows(
    log_probs: torch.Tensor,
    labels: torch.Tensor,
    active_leaves: tuple[torch.Tensor, ...],
    sample_count: int,
) -> torch.Tensor:
    gradient_rows = []

    for row in range(labels.shape[0]):
        for sample_index in range(sample_count):
            term = log_probs[row, int(labels[row, sample_index])]
            gradient_result = torch.autograd.grad(
                term,
                active_leaves,
                retain_graph=row < labels.shape[0] - 1
                or sample_index < sample_count - 1,
                allow_unused=True,
            )
            gradients = tuple(
                torch.zeros_like(leaf) if gradient is None else gradient.detach()
                for leaf, gradient in zip(
                    active_leaves,
                    gradient_result,
                    strict=True,
                )
            )
            gradient_rows.append(
                torch.cat(tuple(gradient.reshape(-1) for gradient in gradients))
                / math.sqrt(float(sample_count))
            )

    return torch.stack(gradient_rows)


def _categorical_logits_matrix(
    operator: OperatorSpec,
    logits: TensorTree,
) -> torch.Tensor:
    if not isinstance(logits, torch.Tensor):
        message = "categorical Fisher requires tensor logits"
        raise MaterializationError(message)

    if logits.ndim < MATRIX_DIMS:
        message = "categorical Fisher logits must include data and class axes"
        raise MaterializationError(message)

    axis = _operator_semantic_int(operator, "logits_axis")

    if axis < 0:
        axis += logits.ndim

    if axis < 0 or axis >= logits.ndim:
        message = "categorical Fisher logits_axis is out of range"
        raise MaterializationError(message)

    moved = torch.movedim(logits, axis, -1)

    return moved.reshape(-1, moved.shape[-1])


def _run_empirical_fisher_vp(execution: StandardExecution) -> TensorTree:
    _require_path(
        execution.operator.kind,
        execution.path,
        (
            EMPIRICAL_FISHER_DENSE_PATH,
            EMPIRICAL_FISHER_GRADIENT_LOOP_PATH,
            EMPIRICAL_FISHER_GRADIENT_VMAP_PATH,
        ),
    )

    if execution.path == EMPIRICAL_FISHER_GRADIENT_LOOP_PATH:
        per_example_gradients = _per_example_gradient_matrix(execution)
    elif execution.path == EMPIRICAL_FISHER_GRADIENT_VMAP_PATH:
        per_example_gradients = _per_example_gradient_matrix_vmap(execution)
    else:
        per_example_gradients = _batch_tensor(execution.batch, "per_example_gradients")

    vector_leaves = _matching_vector_leaves(execution.params, execution.vector)
    vector_tensor = torch.cat(tuple(leaf.reshape(-1) for leaf in vector_leaves))
    _require_finite_tensor(per_example_gradients, "per_example_gradients")
    _require_finite_tensor(vector_tensor, "empirical Fisher vector")
    result = empirical_fisher_vp_dense_anchor(
        per_example_gradients,
        vector_tensor,
        normalization=_empirical_fisher_normalization(
            execution.batch,
            execution.operator,
            per_example_gradients,
        ),
    )
    _require_finite_tensor(result, "empirical Fisher result")

    return _wrap_flat_vector(execution.params, result)


def _per_example_gradient_matrix(execution: StandardExecution) -> torch.Tensor:
    function = _function_objective(execution.operator, execution.function_objectives)
    parameter_items = tuple(execution.params.items())
    active_leaves = tuple(
        tensor.detach().clone().requires_grad_(True) for _, tensor in parameter_items
    )

    def tensor_function(*leaves: torch.Tensor) -> torch.Tensor:
        active_params = {
            name: leaf for (name, _), leaf in zip(parameter_items, leaves, strict=True)
        }
        output = function(
            active_params,
            execution.buffers,
            execution.batch,
            execution.context,
        )

        if not isinstance(output, torch.Tensor):
            message = "per-example gradient loop requires tensor objective output"
            raise MaterializationError(message)

        return output.reshape(-1)

    terms = tensor_function(*active_leaves)

    if terms.numel() == 0:
        message = "per-example gradient loop requires at least one objective term"
        raise MaterializationError(message)

    gradient_rows = []

    for index in range(terms.numel()):
        if not terms[index].requires_grad:
            gradients = tuple(torch.zeros_like(leaf) for leaf in active_leaves)
        else:
            gradient_result = torch.autograd.grad(
                terms[index],
                active_leaves,
                retain_graph=index < terms.numel() - 1,
                allow_unused=True,
            )
            gradients = tuple(
                torch.zeros_like(leaf) if gradient is None else gradient.detach()
                for leaf, gradient in zip(
                    active_leaves,
                    gradient_result,
                    strict=True,
                )
            )

        gradient_rows.append(
            torch.cat(tuple(gradient.reshape(-1) for gradient in gradients))
        )

    return torch.stack(gradient_rows)


def _per_example_gradient_matrix_vmap(execution: StandardExecution) -> torch.Tensor:
    try:
        admit_torch_func(execution.candidate.settings)
    except AdmissionError as error:
        raise MaterializationError(str(error)) from error

    function = _function_objective(execution.operator, execution.function_objectives)
    parameter_items = tuple(execution.params.items())
    active_params = {
        name: tensor.detach().clone().requires_grad_(True)
        for name, tensor in parameter_items
    }
    batched_batch, batch_in_dims = _vmap_batch(execution.batch, execution.candidate)
    chunk_size = _vmap_chunk_size(execution.candidate.settings)

    def single_loss(
        active_params: ParameterTree,
        single_tensor_batch: Batch,
    ) -> torch.Tensor:
        output = function(
            active_params,
            execution.buffers,
            single_tensor_batch,
            execution.context,
        )

        if not isinstance(output, torch.Tensor):
            message = "per-example vmap requires tensor objective output"
            raise MaterializationError(message)

        terms = output.reshape(-1)

        if terms.numel() != 1:
            message = "per-example vmap objective must return one scalar per example"
            raise MaterializationError(message)

        return terms[0]

    gradients = _torch_func_vmap(
        _torch_func_grad(single_loss),
        in_dims=(None, batch_in_dims),
        randomness=str(execution.candidate.settings["vmap_randomness"]),
        chunk_size=chunk_size,
    )(active_params, batched_batch)
    row_count = _vmap_batch_size(batched_batch, batch_in_dims)
    pieces = []

    for name, _ in parameter_items:
        gradient = gradients[name]
        pieces.append(gradient.reshape(row_count, -1))

    return torch.cat(tuple(pieces), dim=1)


def _vmap_batch(
    batch: Batch,
    candidate: Candidate,
) -> tuple[dict[str, Any], dict[str, int | None]]:
    raw_in_dims = candidate.settings.get("vmap_batch_in_dims")

    if not isinstance(raw_in_dims, Mapping):
        message = "per-example vmap requires vmap_batch_in_dims"
        raise MaterializationError(message)

    if set(raw_in_dims) != set(batch):
        message = "vmap_batch_in_dims must cover every batch key"
        raise MaterializationError(message)

    result = {}
    in_dims = {}
    expected_size = None

    for key, value in batch.items():
        raw_dim = raw_in_dims[key]

        if raw_dim is None:
            result[key] = value
            in_dims[key] = None
            continue

        if not isinstance(raw_dim, int) or isinstance(raw_dim, bool):
            message = "vmap_batch_in_dims values must be integers or None"
            raise MaterializationError(message)

        if not isinstance(value, torch.Tensor):
            message = f"per-example vmap mapped batch value must be a tensor: {key}"
            raise MaterializationError(message)

        dim = raw_dim

        if dim < 0:
            dim += value.ndim

        if dim < 0 or dim >= value.ndim:
            message = f"vmap_batch_in_dims axis is out of range: {key}"
            raise MaterializationError(message)

        leading_size = value.shape[dim]

        if expected_size is None:
            expected_size = leading_size
        elif leading_size != expected_size:
            message = "per-example vmap batch leading dimensions differ"
            raise MaterializationError(message)

        result[key] = value
        in_dims[key] = raw_dim

    if expected_size is None or expected_size == 0:
        message = "per-example vmap requires a nonempty mapped batch"
        raise MaterializationError(message)

    return result, in_dims


def _vmap_batch_size(
    batch: Mapping[str, Any],
    in_dims: Mapping[str, int | None],
) -> int:
    for key, value in batch.items():
        dim = in_dims[key]

        if isinstance(value, torch.Tensor) and dim is not None:
            if dim < 0:
                dim += value.ndim

            return value.shape[dim]

    message = "per-example vmap requires a nonempty batch"
    raise MaterializationError(message)


def _vmap_chunk_size(settings: Mapping[str, Any]) -> int | None:
    chunk_size = settings.get("vmap_chunk_size")

    if chunk_size is None:
        message = "per_example_gradient_vmap requires vmap_chunk_size"
        raise MaterializationError(message)

    if not isinstance(chunk_size, int) or chunk_size < 1:
        message = "vmap_chunk_size must be a positive integer"
        raise MaterializationError(message)

    return chunk_size


def _torch_func_grad(function: Callable[..., torch.Tensor]) -> Callable[..., Any]:
    return torch.func.grad(function)


def _torch_func_vmap(function: Callable[..., Any], **kwargs: Any) -> Callable[..., Any]:
    return torch.func.vmap(function, **kwargs)


def _run_metric(execution: StandardExecution) -> TensorTree:
    _require_single_operator_path(execution)
    matrix = _batch_tensor(execution.batch, "metric")
    vector_tensor = _flatten_vector(execution.vector)
    _require_finite_tensor(matrix, "metric matrix")
    _require_finite_tensor(vector_tensor, "metric vector")
    result = dense_metric_multiply(matrix, vector_tensor)
    _require_finite_tensor(result, "metric result")

    return _wrap_flat_vector(execution.vector, result)


def _run_inverse_metric(execution: StandardExecution) -> TensorTree:
    _require_single_operator_path(execution)
    matrix = _batch_tensor(execution.batch, "metric")
    vector_tensor = _flatten_vector(execution.vector)
    _require_finite_tensor(matrix, "metric matrix")
    _require_finite_tensor(vector_tensor, "inverse metric vector")
    result = dense_metric_inverse_multiply(matrix, vector_tensor)
    _require_finite_tensor(result, "inverse metric result")

    return _wrap_flat_vector(execution.vector, result)


@dataclasses.dataclass(frozen=True, slots=True)
class StandardMetricOperator:
    """Materialized dense metric selected by the standard runtime."""

    candidate: Candidate
    record: FullSizeRecord
    default_operation: str = "multiply"

    def __call__(self, batch: Batch, vector: TensorTree) -> TensorTree:
        """Return the selected metric-vector product.

        Raises:
            MaterializationError: If the default metric operation is unsupported.
        """
        if self.default_operation == "multiply":
            return self.multiply(batch, vector)

        if self.default_operation == "inverse_multiply":
            return self.inverse_multiply(batch, vector)

        message = f"unsupported metric default operation: {self.default_operation}"
        raise MaterializationError(message)

    def multiply(self, batch: Batch, vector: TensorTree) -> TensorTree:
        """Return metric-vector product."""

        def callback() -> TensorTree:
            matrix, runtime_vector = self._matrix_and_vector(batch, vector)
            result = dense_metric_multiply(matrix, _flatten_vector(runtime_vector))
            _require_finite_tensor(result, "metric result")

            return _wrap_flat_vector(runtime_vector, result)

        return _run_with_backend_settings(self.candidate.settings, callback)

    def inverse_multiply(self, batch: Batch, vector: TensorTree) -> TensorTree:
        """Return inverse metric-vector product."""

        def callback() -> TensorTree:
            matrix, runtime_vector = self._matrix_and_vector(batch, vector)
            result = dense_metric_inverse_multiply(
                matrix,
                _flatten_vector(runtime_vector),
            )
            _require_finite_tensor(result, "inverse metric result")

            return _wrap_flat_vector(runtime_vector, result)

        return _run_with_backend_settings(self.candidate.settings, callback)

    def inner(
        self,
        batch: Batch,
        left: TensorTree,
        right: TensorTree,
    ) -> torch.Tensor:
        """Return metric inner product."""

        def callback() -> torch.Tensor:
            runtime_batch = _runtime_batch(batch, self.candidate.settings)
            runtime_left = _runtime_vector(left, self.candidate.settings)
            runtime_right = _runtime_vector(right, self.candidate.settings)
            matrix = _batch_tensor(runtime_batch, "metric")
            left_tensor = _flatten_vector(runtime_left)
            right_tensor = _flatten_vector(runtime_right)
            _require_finite_tensor(matrix, "metric matrix")
            _require_finite_tensor(left_tensor, "metric inner left vector")
            _require_finite_tensor(right_tensor, "metric inner right vector")
            result = dense_metric_inner(matrix, left_tensor, right_tensor)
            _require_finite_tensor(result, "metric inner result")

            return result

        return _run_with_backend_settings(self.candidate.settings, callback)

    def _matrix_and_vector(
        self,
        batch: Batch,
        vector: TensorTree,
    ) -> tuple[torch.Tensor, TensorTree]:
        runtime_batch = _runtime_batch(batch, self.candidate.settings)
        runtime_vector = _runtime_vector(vector, self.candidate.settings)
        matrix = _batch_tensor(runtime_batch, "metric")
        vector_tensor = _flatten_vector(runtime_vector)
        _require_finite_tensor(matrix, "metric matrix")
        _require_finite_tensor(vector_tensor, "metric vector")

        return matrix, runtime_vector


@dataclasses.dataclass(frozen=True, slots=True)
class KFACMetricBlock:
    """One Kronecker-factored metric block for a matrix parameter."""

    parameter_name: str
    left_factor_key: str
    right_factor_key: str


@dataclasses.dataclass(frozen=True, slots=True)
class KFACMetricOperator:
    """Metric operations backed by Kronecker-factored blocks."""

    blocks: tuple[KFACMetricBlock, ...]

    def __call__(self, batch: Batch, vector: TensorTree) -> TensorTree:
        """Return metric-vector product."""
        return self.multiply(batch, vector)

    def multiply(self, batch: Batch, vector: TensorTree) -> TensorTree:
        """Return KFAC metric-vector product."""
        vector_map = _kfac_vector_map(vector)
        result = {}

        for block in self.blocks:
            left = _kfac_factor(batch, block.left_factor_key)
            right = _kfac_factor(batch, block.right_factor_key)
            value = _kfac_vector_leaf(vector_map, block)
            _require_kfac_shapes(block, left, right, value)
            product = left @ value @ right.T
            _require_finite_tensor(
                product, f"KFAC metric result {block.parameter_name}"
            )
            result[block.parameter_name] = product

        return result

    def inverse_multiply(self, batch: Batch, vector: TensorTree) -> TensorTree:
        """Return inverse KFAC metric-vector product."""
        vector_map = _kfac_vector_map(vector)
        result = {}

        for block in self.blocks:
            left = _kfac_factor(batch, block.left_factor_key)
            right = _kfac_factor(batch, block.right_factor_key)
            value = _kfac_vector_leaf(vector_map, block)
            _require_kfac_shapes(block, left, right, value)
            left_solved = torch.linalg.solve(left, value)
            product = torch.linalg.solve(right, left_solved.T).T
            _require_finite_tensor(
                product,
                f"inverse KFAC metric result {block.parameter_name}",
            )
            result[block.parameter_name] = product

        return result

    def inner(
        self,
        batch: Batch,
        left: TensorTree,
        right: TensorTree,
    ) -> torch.Tensor:
        """Return KFAC metric inner product."""
        return tree_dot(left, self.multiply(batch, right))


def _kfac_vector_map(vector: TensorTree) -> dict[str, TensorTree]:
    if type(vector) is not dict:
        message = "KFAC vector must be a tensor-tree mapping"
        raise MaterializationError(message)

    return dict(vector)


def _kfac_factor(batch: Batch, key: str) -> torch.Tensor:
    value = batch.get(key)

    if not isinstance(value, torch.Tensor):
        message = f"KFAC factor is missing or not a tensor: {key}"
        raise MaterializationError(message)

    _require_finite_tensor(value, f"KFAC factor {key}")

    if value.ndim != MATRIX_DIMS or value.shape[0] != value.shape[1]:
        message = f"KFAC factor must be square: {key}"
        raise MaterializationError(message)

    return value


def _kfac_vector_leaf(
    vector: Mapping[str, TensorTree],
    block: KFACMetricBlock,
) -> torch.Tensor:
    value = vector.get(block.parameter_name)

    if not isinstance(value, torch.Tensor):
        message = f"KFAC vector leaf is missing or not a tensor: {block.parameter_name}"
        raise MaterializationError(message)

    _require_finite_tensor(value, f"KFAC vector {block.parameter_name}")

    if value.ndim != MATRIX_DIMS:
        message = f"KFAC vector leaf must be a matrix: {block.parameter_name}"
        raise MaterializationError(message)

    return value


def _require_kfac_shapes(
    block: KFACMetricBlock,
    left: torch.Tensor,
    right: torch.Tensor,
    value: torch.Tensor,
) -> None:
    expected = (left.shape[0], right.shape[0])

    if tuple(value.shape) != expected:
        message = (
            f"KFAC vector leaf shape mismatch for {block.parameter_name}: "
            f"{tuple(value.shape)} != {expected}"
        )
        raise MaterializationError(message)


STANDARD_RUNNERS = {
    "gradient": _run_gradient,
    "jvp": _run_jvp,
    "vjp": _run_vjp,
    "hvp": _run_hvp,
    "ggnvp": _run_ggnvp,
    "fisher_vp": _run_fisher_vp,
    "empirical_fisher_vp": _run_empirical_fisher_vp,
    "metric": _run_metric,
    "inverse_metric": _run_inverse_metric,
}


def _standard_materializer(
    operation_factory: OperationFactory,
    operator: OperatorSpec | None = None,
) -> Materializer:
    def callback(candidate: Candidate, record: FullSizeRecord) -> Any:
        if (
            record.family != candidate.family
            or record.candidate_id != candidate.candidate_id
        ):
            message = "selected record does not match selected candidate"
            raise MaterializationError(message)

        if operator is not None and operator.kind == "metric":
            return StandardMetricOperator(candidate, record)

        if operator is not None and operator.kind == "inverse_metric":
            return StandardMetricOperator(
                candidate,
                record,
                default_operation="inverse_multiply",
            )

        def selected(batch: Batch, vector: TensorTree) -> TensorTree:
            return operation_factory(candidate, batch, vector)()

        return selected

    return CallableMaterializer(
        "vptune.standard_runtime",
        "0.0.1",
        {"operation_factory": "standard_operation_factory"},
        callback,
    )


def _operator_path(operator: OperatorSpec, candidate: Candidate) -> str:
    path = candidate.settings.get("operator_path")
    singleton_path = SINGLE_OPERATOR_PATHS.get(operator.kind)

    if singleton_path is None:
        return _required_candidate_operator_path(candidate)

    if path is not None:
        message = f"operator_path is not a setting for {operator.kind}"
        raise MaterializationError(message)

    return singleton_path


def _required_candidate_operator_path(candidate: Candidate) -> str:
    path = candidate.settings.get("operator_path")

    if not isinstance(path, str):
        message = f"candidate operator_path is required: {candidate.candidate_id}"
        raise MaterializationError(message)

    return path


def _require_supported_standard_settings(
    operator: OperatorSpec,
    candidate: Candidate,
) -> None:
    unsupported = tuple(
        key for key in candidate.settings if key not in SUPPORTED_STANDARD_SETTINGS
    )

    if unsupported:
        message = f"standard runtime settings are unsupported: {unsupported}"
        raise MaterializationError(message)

    path = _operator_path(operator, candidate)

    for key in ("vmap_chunk_size", "vmap_batch_in_dims"):
        if key in candidate.settings and path != EMPIRICAL_FISHER_GRADIENT_VMAP_PATH:
            message = f"{key} is only supported by per_example_gradient_vmap"
            raise MaterializationError(message)

    if path in {
        JVP_PATH,
        VJP_PATH,
        HVP_JVP_GRAD_PATH,
        EMPIRICAL_FISHER_GRADIENT_VMAP_PATH,
    }:
        try:
            admit_torch_func(candidate.settings)
        except AdmissionError as error:
            raise MaterializationError(str(error)) from error

    if path in {JVP_PATH, JVP_FORWARD_AD_PATH, HVP_JVP_GRAD_PATH}:
        try:
            admit_forward_ad(candidate.settings)
        except AdmissionError as error:
            raise MaterializationError(str(error)) from error

    if any(field in candidate.settings for field in FUNCTIONAL_CALL_FIELDS):
        try:
            admit_functional_call(candidate.settings)
        except AdmissionError as error:
            raise MaterializationError(str(error)) from error


def _runtime_params(
    params: ParameterTree,
    settings: Mapping[str, Any],
) -> ParameterTree:
    dtype = _runtime_compute_dtype(settings)

    if dtype is None:
        return params

    return {key: tensor.to(dtype=dtype) for key, tensor in params.items()}


def _runtime_buffers(
    buffers: BufferTree,
    settings: Mapping[str, Any],
) -> BufferTree:
    dtype = _runtime_compute_dtype(settings)

    if dtype is None:
        return buffers

    return {key: tensor.to(dtype=dtype) for key, tensor in buffers.items()}


def _runtime_batch(batch: Batch, settings: Mapping[str, Any]) -> Batch:
    dtype = _runtime_compute_dtype(settings)

    if dtype is None:
        return batch

    return {key: _runtime_batch_value(value, dtype) for key, value in batch.items()}


def _runtime_vector(vector: TensorTree, settings: Mapping[str, Any]) -> TensorTree:
    dtype = _runtime_compute_dtype(settings)

    if dtype is None:
        return vector

    return tree_map(lambda tensor: tensor.to(dtype=dtype), vector)


def _runtime_batch_value(value: Any, dtype: torch.dtype) -> Any:
    if isinstance(value, torch.Tensor):
        if value.is_floating_point():
            return value.to(dtype=dtype)

        return value

    if isinstance(value, dict):
        return {key: _runtime_batch_value(child, dtype) for key, child in value.items()}

    if isinstance(value, tuple):
        return tuple(_runtime_batch_value(child, dtype) for child in value)

    return value


def _runtime_compute_dtype(settings: Mapping[str, Any]) -> torch.dtype | None:
    compute_dtype = _dtype_setting(settings, "compute_dtype")

    if compute_dtype is not None:
        return compute_dtype

    return _dtype_setting(settings, "model_dtype")


def _dtype_setting(settings: Mapping[str, Any], key: str) -> torch.dtype | None:
    dtype_name = settings.get(key)

    if dtype_name is None:
        return None

    if not isinstance(dtype_name, str):
        message = f"{key} must be a string"
        raise MaterializationError(message)

    if dtype_name == "bfloat16":
        return torch.bfloat16

    if dtype_name == "float16":
        return torch.float16

    if dtype_name == "float32":
        return torch.float32

    message = f"{key} is unsupported by standard runtime: {dtype_name}"
    raise MaterializationError(message)


def _run_with_backend_settings(
    settings: Mapping[str, Any],
    callback: Callable[[], Any],
) -> Any:
    matmul_precision = _matmul_precision_setting(settings)
    allow_tf32 = _bool_setting(settings, "allow_tf32")
    allow_bf16_reduction = _bool_setting(
        settings,
        "allow_bf16_reduced_precision_reduction",
    )
    previous_matmul_precision = torch.get_float32_matmul_precision()
    previous_allow_tf32 = torch.backends.cuda.matmul.allow_tf32
    previous_allow_bf16_reduction = (
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
    )

    if matmul_precision is not None:
        torch.set_float32_matmul_precision(matmul_precision)

    if allow_tf32 is not None:
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32

    if allow_bf16_reduction is not None:
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = (
            allow_bf16_reduction
        )

    try:
        return callback()
    finally:
        torch.set_float32_matmul_precision(previous_matmul_precision)
        torch.backends.cuda.matmul.allow_tf32 = previous_allow_tf32
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = (
            previous_allow_bf16_reduction
        )


def _matmul_precision_setting(settings: Mapping[str, Any]) -> str | None:
    value = settings.get("matmul_precision")

    if value is None:
        return None

    if not isinstance(value, str):
        message = "matmul_precision must be a string"
        raise MaterializationError(message)

    if value not in {"highest", "high", "medium"}:
        message = f"matmul_precision is unsupported by standard runtime: {value}"
        raise MaterializationError(message)

    return value


def _bool_setting(settings: Mapping[str, Any], key: str) -> bool | None:
    value = settings.get(key)

    if value is None:
        return None

    if not isinstance(value, bool):
        message = f"{key} must be boolean"
        raise MaterializationError(message)

    return value


def _anchor_candidate(operator: OperatorSpec, candidate: Candidate) -> Candidate:
    path = _anchor_path(operator)
    settings = _anchor_settings(candidate, path)

    if operator.kind in SINGLE_OPERATOR_PATHS:
        settings.pop("operator_path", None)

    return dataclasses.replace(
        candidate,
        settings=settings,
    )


def _anchor_settings(candidate: Candidate, path: str) -> dict[str, Any]:
    settings = dict(candidate.settings)

    for key in (
        *RUNTIME_DTYPE_SETTINGS,
        *BACKEND_SETTINGS,
        *TORCH_FUNC_FIELDS,
        *FUNCTIONAL_CALL_FIELDS,
        "vmap_chunk_size",
        "vmap_batch_in_dims",
    ):
        settings.pop(key, None)

    settings["operator_path"] = path
    settings.update(_anchor_admission_settings(path))

    return settings


def _anchor_admission_settings(path: str) -> dict[str, Any]:
    if path == JVP_PATH:
        return _torch_func_anchor_settings(requires_forward_ad=True)

    if path == VJP_PATH:
        return _torch_func_anchor_settings(requires_forward_ad=False)

    if path == HVP_JVP_GRAD_PATH:
        return _torch_func_anchor_settings(requires_forward_ad=True)

    return {}


def _torch_func_anchor_settings(*, requires_forward_ad: bool) -> dict[str, Any]:
    return {
        "contains_autograd_call": False,
        "contains_backward_call": False,
        "uses_out_variant": False,
        "uses_data_dependent_control_flow": False,
        "uses_item": False,
        "has_dynamic_shape_output": False,
        "vmap_randomness": "error",
        "requires_forward_ad": requires_forward_ad,
        "forward_ad_supported": True,
    }


def _anchor_path(operator: OperatorSpec) -> str:
    if operator.kind == "fisher_vp":
        return _fisher_anchor_path(operator)

    path = STANDARD_ANCHOR_PATHS.get(operator.kind)

    if path is not None:
        return path

    message = f"standard anchor does not support operator kind: {operator.kind}"
    raise MaterializationError(message)


def _fisher_anchor_path(operator: OperatorSpec) -> str:
    distribution = _operator_semantic(operator, "distribution")
    expectation = _operator_semantic(operator, "expectation")
    label_policy = _operator_semantic(operator, "label_policy")

    if distribution == "categorical" and label_policy == "model_distribution":
        if expectation == "exact":
            return FISHER_CATEGORICAL_EXACT_PATH

        if expectation == "monte_carlo":
            return FISHER_CATEGORICAL_MC_PATH

    if distribution == "explicit_score_gradients" and expectation == "explicit_rows":
        return FISHER_SCORE_GRADIENT_LOOP_PATH

    message = "standard Fisher anchor does not support declared semantics"
    raise MaterializationError(message)


def _require_candidate_family(operator: OperatorSpec, candidate: Candidate) -> None:
    if candidate.family != operator.family:
        message = (
            f"candidate family does not match operator family: "
            f"{candidate.family} != {operator.family}"
        )
        raise MaterializationError(message)


def _require_path(
    operator_kind: str,
    path: str,
    allowed_paths: tuple[str, ...],
) -> None:
    if path in allowed_paths:
        return

    message = f"operator path {path} does not support {operator_kind}"
    raise MaterializationError(message)


def _require_single_operator_path(execution: StandardExecution) -> None:
    path = SINGLE_OPERATOR_PATHS[execution.operator.kind]

    _require_path(execution.operator.kind, execution.path, (path,))


def _scalar_objective(
    operator: OperatorSpec,
    scalar_objectives: Mapping[str, ScalarObjective],
) -> ScalarObjective:
    objective = scalar_objectives.get(operator.objective_id)

    if objective is None:
        message = f"scalar objective is missing: {operator.objective_id}"
        raise MaterializationError(message)

    return objective


def _function_objective(
    operator: OperatorSpec,
    function_objectives: Mapping[str, FunctionObjective],
) -> FunctionObjective:
    objective = function_objectives.get(operator.objective_id)

    if objective is None:
        message = f"function objective is missing: {operator.objective_id}"
        raise MaterializationError(message)

    return objective


def _flatten_vector(vector: TensorTree) -> torch.Tensor:
    leaves = tree_leaves(vector)

    if not leaves:
        message = "dense standard operator requires at least one tensor leaf"
        raise MaterializationError(message)

    return torch.cat(tuple(leaf.reshape(-1) for leaf in leaves))


def _matching_vector_leaves(
    params: ParameterTree, vector: TensorTree
) -> tuple[torch.Tensor, ...]:
    checked = tree_map2(
        lambda param, tangent: tangent.reshape_as(param), params, vector
    )

    return tree_leaves(checked)


def _wrap_flat_vector(template: TensorTree, result: torch.Tensor) -> TensorTree:
    leaves = []
    offset = 0

    for leaf in tree_leaves(template):
        width = leaf.numel()
        leaves.append(result[offset : offset + width].reshape_as(leaf))
        offset += width

    if offset != result.numel():
        message = "dense standard operator output length differs from vector tree"
        raise MaterializationError(message)

    return tree_from_leaves(template, tuple(leaves))


def _batch_tensor(batch: Batch, key: str) -> torch.Tensor:
    value = batch.get(key)

    if not isinstance(value, torch.Tensor):
        message = f"batch tensor is missing: {key}"
        raise MaterializationError(message)

    return value


def _batch_tree(batch: Batch, key: str) -> TensorTree:
    value = batch.get(key)

    if isinstance(value, torch.Tensor):
        return value

    if isinstance(value, tuple):
        tree_leaves(value)

        return value

    if isinstance(value, dict):
        tree_leaves(value)

        return value

    message = f"batch tensor tree is missing: {key}"
    raise MaterializationError(message)


def _normalization(batch: Batch, operator: OperatorSpec) -> float:
    value = batch.get("normalization")

    if not isinstance(value, int | float):
        message = "batch normalization is missing"
        raise MaterializationError(message)

    normalization = float(value)

    if normalization <= 0.0:
        message = "batch normalization must be positive"
        raise MaterializationError(message)

    if operator.aggregation == "sum" and not math.isclose(
        normalization,
        1.0,
        rel_tol=0.0,
        abs_tol=0.0,
    ):
        message = "sum aggregation requires normalization=1.0"
        raise MaterializationError(message)

    return normalization


def _empirical_fisher_normalization(
    batch: Batch,
    operator: OperatorSpec,
    per_example_gradients: torch.Tensor,
) -> float:
    _require_empirical_fisher_semantics(operator)
    denominator = _operator_semantic(operator, "denominator")

    if denominator == "num_examples":
        normalization = float(per_example_gradients.shape[0])
    elif denominator == "one":
        normalization = 1.0
    elif denominator == "batch_normalization":
        normalization = _normalization(batch, operator)
    else:
        message = f"empirical Fisher denominator is unsupported: {denominator}"
        raise MaterializationError(message)

    if operator.aggregation == "sum" and not math.isclose(
        normalization,
        1.0,
        rel_tol=0.0,
        abs_tol=0.0,
    ):
        message = "sum aggregation requires empirical Fisher denominator one"
        raise MaterializationError(message)

    return normalization


def _require_empirical_fisher_semantics(operator: OperatorSpec) -> None:
    loss_reduction = _operator_semantic(operator, "loss_reduction")

    if loss_reduction != "per_example":
        message = f"empirical Fisher loss_reduction is unsupported: {loss_reduction}"
        raise MaterializationError(message)


def _fisher_normalization(execution: StandardExecution) -> float:
    denominator = _operator_semantic(execution.operator, "denominator")

    if denominator == "num_examples":
        if execution.path in {
            FISHER_CATEGORICAL_EXACT_PATH,
            FISHER_CATEGORICAL_MC_PATH,
        }:
            logits = _function_objective(
                execution.operator,
                execution.function_objectives,
            )(
                execution.params,
                execution.buffers,
                execution.batch,
                execution.context,
            )
            logits_matrix = _categorical_logits_matrix(execution.operator, logits)

            return float(logits_matrix.shape[0])

        value = execution.batch.get("num_examples")

        if not isinstance(value, int | float):
            message = "num_examples denominator requires batch num_examples"
            raise MaterializationError(message)

        if float(value) <= 0.0:
            message = "num_examples denominator must be positive"
            raise MaterializationError(message)

        return float(value)

    if denominator == "one":
        return 1.0

    if denominator == "batch_normalization":
        return _normalization(execution.batch, execution.operator)

    message = f"Fisher denominator is unsupported: {denominator}"
    raise MaterializationError(message)


def _require_fisher_semantics(
    operator: OperatorSpec,
    required: Mapping[str, str],
) -> None:
    for key, expected in required.items():
        actual = _operator_semantic(operator, key)

        if actual != expected:
            message = f"Fisher semantic field mismatch: {key}"
            raise MaterializationError(message)


def _require_valid_fisher_semantics(operator: OperatorSpec) -> None:
    distribution = _operator_semantic(operator, "distribution")
    expectation = _operator_semantic(operator, "expectation")

    if distribution == "categorical" and expectation == "exact":
        _require_fisher_semantics(
            operator,
            {
                "distribution": "categorical",
                "label_policy": "model_distribution",
                "expectation": "exact",
                "sample_space": "classes",
                "loss_reduction": "log_prob",
            },
        )

        return

    if distribution == "categorical" and expectation == "monte_carlo":
        _require_fisher_semantics(
            operator,
            {
                "distribution": "categorical",
                "label_policy": "model_distribution",
                "expectation": "monte_carlo",
                "sample_space": "classes",
                "loss_reduction": "log_prob",
            },
        )

        return

    if distribution == "explicit_score_gradients":
        _require_explicit_score_fisher_semantics(operator)

        return

    message = f"Fisher distribution is unsupported: {distribution}"
    raise MaterializationError(message)


def _require_explicit_score_fisher_semantics(operator: OperatorSpec) -> None:
    _require_fisher_semantics(
        operator,
        {
            "distribution": "explicit_score_gradients",
            "label_policy": "explicit_scores",
            "expectation": "explicit_rows",
            "sample_space": "terms",
            "loss_reduction": "none",
        },
    )


def _ggn_loss_geometry(operator: OperatorSpec) -> str:
    value = _operator_semantic(operator, "loss_geometry")

    if value not in {"psd_metric", "linear_map"}:
        message = f"GGNVP loss_geometry is unsupported: {value}"
        raise MaterializationError(message)

    return value


def _operator_semantic(operator: OperatorSpec, key: str) -> str:
    value = operator.semantics.get(key)

    if not isinstance(value, str):
        message = f"operator semantic field is missing: {key}"
        raise MaterializationError(message)

    return value


def _operator_semantic_int(operator: OperatorSpec, key: str) -> int:
    value = operator.semantics.get(key)

    if not isinstance(value, int) or isinstance(value, bool):
        message = f"operator semantic field is missing: {key}"
        raise MaterializationError(message)

    return value


def _operator_semantic_positive_int(operator: OperatorSpec, key: str) -> int:
    value = _operator_semantic_int(operator, key)

    if value < 1:
        message = f"operator semantic field must be positive: {key}"
        raise MaterializationError(message)

    return value
