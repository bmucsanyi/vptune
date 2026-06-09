"""Derivative operator lowerings for the standard runtime.

Gradient, JVP, VJP, HVP, and VHP paths across torch.func, eager
autograd, forward AD, and linearize, plus per-example gradient
matrices and streaming gradient rows.
"""

import dataclasses
from collections.abc import Callable, Iterator, Mapping, Sequence
from typing import Any

import torch

from vptune.core.data import (
    Batch,
    BufferTree,
    Candidate,
    CandidateOperation,
    FunctionObjective,
    ObjectiveContext,
    OperationFactory,
    OperatorSpec,
    ParameterSurface,
    ParameterTree,
    ScalarObjective,
)
from vptune.core.tensor_tree import (
    TensorTree,
    tree_from_leaves,
    tree_leaves,
    tree_map,
    tree_map2,
)
from vptune.engine import (
    fisher,
    layout,
    runtime,
    runtime_values,
    vectorization,
)
from vptune.engine.anchors import (
    finite_difference_hvp,
    forward_ad_jvp_anchor,
    gradient_anchor,
    hvp_anchor,
    hvp_jvp_grad_anchor,
    hvp_reverse_over_reverse_anchor,
    jvp_anchor,
)
from vptune.errors import (
    MaterializationError,
    ReferenceFailedError,
)


def require_vhp_reference_policy(
    candidate: Candidate,
    batch: Batch,
    thresholds: Mapping[str, float],
) -> None:
    """Validate vhp reference policy.

    Raises:
        ReferenceFailedError: If the declared inputs are invalid.
    """
    if candidate.settings.get("hvp.path") != "autograd_functional_vhp":
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

    runtime_values.batch_tree(batch, "symmetry_vector")


def hvp_finite_difference_measurements(
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
    parameter_surface: ParameterSurface | None,
) -> dict[str, float]:
    """Return the hvp finite difference measurements.

    Returns:
        The hvp finite difference measurements.
    """
    if operator.kind != "hvp":
        return {}

    scalar = runtime_values.scalar_objective(operator, scalar_objectives)
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
    errors = layout.layout_aware_tree_error_measurements(
        candidate,
        candidate_output,
        finite_difference,
    )

    measurements = {
        "directional_abs_diff": errors["max_abs_diff"],
        "directional_rel_diff": errors["max_rel_diff"],
    }

    symmetry_vector = runtime_values.batch_tree(batch, "symmetry_vector")
    runtime_values.require_min_probe_norm(vector, "hvp reference vector")
    runtime_values.require_min_probe_norm(symmetry_vector, "symmetry_vector")
    anchor_symmetry = candidate_factory(
        anchor_candidate,
        batch,
        symmetry_vector,
    )()
    left = layout.layout_aware_tree_dot(
        candidate.settings,
        runtime.runtime_vector(
            symmetry_vector,
            candidate.settings,
            candidate_output,
            parameter_surface,
        ),
        candidate_output,
    )
    right = layout.layout_aware_tree_dot(
        candidate.settings,
        runtime.runtime_vector(
            vector,
            candidate.settings,
            anchor_symmetry,
            parameter_surface,
        ),
        anchor_symmetry,
    )
    measurements["symmetry_max_abs_diff"] = float((left - right).abs().item())

    return measurements


def prepare_gradient_execution(
    execution: runtime_values.StandardExecution,
) -> runtime_values.StandardExecution:
    """Prepare gradient execution.

    Returns:
        The gradient execution result.

    Raises:
        MaterializationError: If the declared inputs are invalid.
    """
    schedule = execution.candidate.settings.get("gradient.graph_schedule")

    if schedule is None or schedule == "rebuild_per_call":
        return execution

    if schedule != "build_once":
        message = f"gradient.graph_schedule is unsupported: {schedule}"
        raise MaterializationError(message)

    return dataclasses.replace(
        execution,
        prepared_gradient=_gradient_operation_by_path(execution),
    )


def prepare_jvp_execution(
    execution: runtime_values.StandardExecution,
) -> runtime_values.StandardExecution:
    """Prepare jvp execution.

    Returns:
        The jvp execution result.

    Raises:
        MaterializationError: If the declared inputs are invalid.
    """
    reuse = execution.candidate.settings.get("jvp.linearize_reuse")

    if reuse is None or reuse == "none":
        return execution

    if reuse != "reuse_at_same_primal":
        message = f"jvp.linearize_reuse is unsupported: {reuse}"
        raise MaterializationError(message)

    if execution.path != runtime_values.JVP_LINEARIZE_PATH:
        message = "reuse_at_same_primal requires torch_func_linearize"
        raise MaterializationError(message)

    _, jvp_function = torch.func.linearize(
        jvp_tensor_function(execution),
        execution.params,
    )

    return dataclasses.replace(execution, linearized_jvp=jvp_function)


def prepare_vjp_execution(
    execution: runtime_values.StandardExecution,
) -> runtime_values.StandardExecution:
    """Prepare vjp execution.

    Returns:
        The vjp execution result.

    Raises:
        MaterializationError: If the declared inputs are invalid.
    """
    reuse = execution.candidate.settings.get("vjp.closure_reuse")

    if reuse is None or reuse == "none":
        return execution

    if reuse != "reuse_vjp_closure_at_same_primal":
        message = f"vjp.closure_reuse is unsupported: {reuse}"
        raise MaterializationError(message)

    if execution.path != runtime_values.VJP_PATH:
        message = "reuse_vjp_closure_at_same_primal requires torch_func_vjp"
        raise MaterializationError(message)

    pullback = vjp_pullback(
        vjp_tensor_function(execution),
        execution.params,
    )

    def closure(cotangent: TensorTree) -> TensorTree:
        (result,) = pullback(cotangent)

        return result

    return dataclasses.replace(execution, vjp_closure=closure)


def prepare_hvp_execution(
    execution: runtime_values.StandardExecution,
) -> runtime_values.StandardExecution:
    """Prepare hvp execution.

    Returns:
        The hvp execution result.

    Raises:
        MaterializationError: If the declared inputs are invalid.
    """
    reuse = execution.candidate.settings.get("hvp.gradient_reuse")

    if reuse is None or reuse == "recompute_gradient":
        return execution

    if reuse != "reuse_gradient_closure":
        message = f"hvp.gradient_reuse is unsupported: {reuse}"
        raise MaterializationError(message)

    if execution.path != runtime_values.HVP_LINEARIZE_GRAD_PATH:
        message = "reuse_gradient_closure requires linearize_grad"
        raise MaterializationError(message)

    gradient_function = torch.func.grad(hvp_scalar_function(execution))
    _, hvp_function = torch.func.linearize(gradient_function, execution.params)

    return dataclasses.replace(execution, linearized_hvp=hvp_function)


def run_gradient(execution: runtime_values.StandardExecution) -> TensorTree:
    """Run gradient.

    Returns:
        The gradient result.
    """
    if execution.compiled_inner is not None:
        return execution.compiled_inner()

    return run_gradient_by_path(execution)


def run_gradient_by_path(execution: runtime_values.StandardExecution) -> TensorTree:
    """Run gradient by path.

    Returns:
        The gradient by path result.
    """
    if execution.prepared_gradient is not None:
        return execution.prepared_gradient()

    return _gradient_operation_by_path(execution)()


def _gradient_operation_by_path(
    execution: runtime_values.StandardExecution,
) -> CandidateOperation:
    runtime_values.require_path(
        execution.operator.kind,
        execution.path,
        (
            runtime_values.GRADIENT_PATH,
            runtime_values.GRADIENT_TORCH_FUNC_PATH,
            runtime_values.GRADIENT_TORCH_FUNC_VALUE_PATH,
            runtime_values.GRADIENT_BACKWARD_MATERIALIZED_PATH,
        ),
    )
    scalar_function = hvp_scalar_function(execution)

    if execution.path == runtime_values.GRADIENT_PATH:

        def operation() -> TensorTree:
            return gradient_anchor(scalar_function, execution.params)

        return operation

    if execution.path == runtime_values.GRADIENT_TORCH_FUNC_PATH:
        gradient_function = torch.func.grad(scalar_function)

        def operation() -> TensorTree:
            result = gradient_function(execution.params)
            runtime_values.require_finite_tree(result, "gradient result")

            return result

        return operation

    if execution.path == runtime_values.GRADIENT_TORCH_FUNC_VALUE_PATH:
        gradient_function = torch.func.grad_and_value(scalar_function)

        def operation() -> TensorTree:
            result, value = gradient_function(execution.params)
            runtime_values.require_finite_tree(result, "gradient result")
            _require_gradient_value_reuse(execution, value)

            return result

        return operation

    def operation() -> TensorTree:
        return _run_materialized_gradient(execution)

    return operation


def _require_gradient_value_reuse(
    execution: runtime_values.StandardExecution,
    value: torch.Tensor,
) -> None:
    reuse = execution.candidate.settings.get("gradient.value_reuse")

    if reuse is None or reuse == "gradient_only":
        return

    if reuse != "gradient_and_primal_value":
        message = f"gradient.value_reuse is unsupported: {reuse}"
        raise MaterializationError(message)

    if execution.path != runtime_values.GRADIENT_TORCH_FUNC_VALUE_PATH:
        message = "gradient_and_primal_value requires torch_func_grad_and_value"
        raise MaterializationError(message)

    runtime_values.require_finite_tensor(value, "gradient primal value")


def _run_materialized_gradient(
    execution: runtime_values.StandardExecution,
) -> TensorTree:
    scalar_function = hvp_scalar_function(execution)
    active_params = runtime_values.grad_enabled_params(execution.params)
    value = scalar_function(active_params)
    value.backward()
    result = runtime_values.parameter_grad_tree(active_params)
    runtime_values.require_finite_tree(result, "gradient result")

    return result


def run_jvp(execution: runtime_values.StandardExecution) -> TensorTree:
    """Run jvp.

    Returns:
        The jvp result.
    """
    if execution.compiled_inner is not None:
        return execution.compiled_inner()

    return run_jvp_by_path(execution)


def run_jvp_by_path(execution: runtime_values.StandardExecution) -> TensorTree:
    """Run jvp by path.

    Returns:
        The jvp by path result.
    """
    return vectorization.run_single_vectorized_by_path(
        execution,
        (
            runtime_values.JVP_PATH,
            runtime_values.JVP_FORWARD_AD_PATH,
            runtime_values.JVP_LINEARIZE_PATH,
        ),
        _run_jvp_single_vector,
        _run_jvp_single_vector,
        vectorization.run_jvp_vector_vmap,
    )


def _run_jvp_single_vector(execution: runtime_values.StandardExecution) -> TensorTree:
    tensor_function = jvp_tensor_function(execution)

    if execution.path == runtime_values.JVP_FORWARD_AD_PATH:
        return forward_ad_jvp_anchor(
            tensor_function,
            execution.params,
            execution.vector,
        )

    if execution.path == runtime_values.JVP_LINEARIZE_PATH:
        if execution.linearized_jvp is not None:
            return execution.linearized_jvp(execution.vector)

        _, jvp_function = torch.func.linearize(tensor_function, execution.params)

        return jvp_function(execution.vector)

    return jvp_anchor(tensor_function, execution.params, execution.vector)


def jvp_tensor_function(
    execution: runtime_values.StandardExecution,
) -> Callable[[ParameterTree], TensorTree]:
    """Return the tensor function for JVP paths.

    Returns:
        The tensor function for JVP paths.
    """
    function = runtime_values.function_objective(
        execution.operator, execution.function_objectives
    )

    def tensor_function(active_params: ParameterTree) -> TensorTree:
        return runtime.call_function_objective(execution, function, active_params)

    return tensor_function


def run_vjp(execution: runtime_values.StandardExecution) -> TensorTree:
    """Run vjp.

    Returns:
        The vjp result.
    """
    if execution.compiled_inner is not None:
        return execution.compiled_inner()

    return run_vjp_by_path(execution)


def run_vjp_by_path(execution: runtime_values.StandardExecution) -> TensorTree:
    """Run vjp by path.

    Returns:
        The vjp by path result.
    """
    return vectorization.run_single_vectorized_by_path(
        execution,
        (
            runtime_values.VJP_PATH,
            runtime_values.VJP_AUTOGRAD_OUTPUTS_PATH,
            runtime_values.VJP_BACKWARD_MATERIALIZED_PATH,
        ),
        _run_vjp_single_vector,
        _run_vjp_single_vector,
        vectorization.run_vjp_vector_vmap,
    )


def _run_vjp_single_vector(execution: runtime_values.StandardExecution) -> TensorTree:
    tensor_function = vjp_tensor_function(execution)

    if execution.path == runtime_values.VJP_PATH:
        if execution.vjp_closure is not None:
            return execution.vjp_closure(execution.vector)

        pullback = vjp_pullback(
            tensor_function,
            execution.params,
        )
        (result,) = pullback(execution.vector)

        return result

    return _run_autograd_vjp(execution, tensor_function)


def vjp_tensor_function(
    execution: runtime_values.StandardExecution,
) -> Callable[[ParameterTree], TensorTree]:
    """Return the tensor function for VJP paths.

    Returns:
        The tensor function for VJP paths.
    """
    if _uses_stateful_module_call(execution):
        return _stateful_module_tensor_function(execution)

    function = runtime_values.function_objective(
        execution.operator, execution.function_objectives
    )

    def tensor_function(active_params: ParameterTree) -> TensorTree:
        return runtime.call_function_objective(execution, function, active_params)

    return tensor_function


def vjp_pullback(
    tensor_function: Callable[[ParameterTree], TensorTree],
    params: ParameterTree,
) -> Callable[[TensorTree], tuple[TensorTree]]:
    """Return the VJP pullback for the declared path.

    Returns:
        The VJP pullback for the declared path.
    """
    vjp_result = torch.func.vjp(tensor_function, params, has_aux=False)

    return vjp_result[1]


def _run_autograd_vjp(
    execution: runtime_values.StandardExecution,
    tensor_function: Callable[[ParameterTree], TensorTree],
) -> TensorTree:
    if execution.path == runtime_values.VJP_AUTOGRAD_OUTPUTS_PATH:
        return autograd_grad_outputs_vjp(
            tensor_function,
            execution.params,
            execution.vector,
        )

    return _backward_materialized_vjp(
        tensor_function,
        execution.params,
        execution.vector,
    )


def _backward_materialized_vjp(
    tensor_function: Callable[[ParameterTree], TensorTree],
    params: ParameterTree,
    cotangent: TensorTree,
) -> TensorTree:
    active_params = runtime_values.grad_enabled_params(params)
    output = tensor_function(active_params)
    output_leaves = tree_leaves(output)
    cotangent_leaves = tree_leaves(
        tree_map2(lambda out, cotangent: cotangent.reshape_as(out), output, cotangent)
    )

    torch.autograd.backward(output_leaves, grad_tensors=cotangent_leaves)
    result = runtime_values.parameter_grad_tree(active_params)
    runtime_values.require_finite_tree(result, "VJP result")

    return result


def autograd_grad_outputs_vjp(
    tensor_function: Callable[[ParameterTree], TensorTree],
    params: ParameterTree,
    cotangent: TensorTree,
) -> TensorTree:
    """Return the autograd VJP for declared grad outputs.

    Returns:
        The autograd VJP for declared grad outputs.
    """
    active_params = runtime_values.grad_enabled_params(params)
    output = tensor_function(active_params)
    output_leaves = tree_leaves(output)
    cotangent_leaves = tree_leaves(
        tree_map2(lambda out, cotangent: cotangent.reshape_as(out), output, cotangent)
    )
    gradients = torch.autograd.grad(
        output_leaves,
        tuple(active_params.values()),
        grad_outputs=cotangent_leaves,
        allow_unused=True,
    )
    result = tree_from_leaves(
        active_params,
        tuple(
            torch.zeros_like(param) if gradient is None else gradient.detach()
            for param, gradient in zip(active_params.values(), gradients, strict=True)
        ),
    )
    runtime_values.require_finite_tree(result, "VJP result")

    return result


def run_hvp(execution: runtime_values.StandardExecution) -> TensorTree:
    """Run hvp.

    Returns:
        The hvp result.
    """
    if execution.compiled_inner is not None:
        return execution.compiled_inner()

    return run_hvp_by_path(execution)


def run_hvp_by_path(execution: runtime_values.StandardExecution) -> TensorTree:
    """Run hvp by path.

    Returns:
        The hvp by path result.
    """
    runtime_values.require_path(
        execution.operator.kind,
        execution.path,
        (
            runtime_values.HVP_REFERENCE_PATH,
            runtime_values.HVP_FUNCTIONAL_PATH,
            runtime_values.HVP_JVP_GRAD_PATH,
            runtime_values.HVP_FORWARD_AD_PATH,
            runtime_values.HVP_LINEARIZE_GRAD_PATH,
            runtime_values.VHP_PATH,
        ),
    )

    return vectorization.run_by_vectorization_mode(
        execution,
        single_vector=run_hvp_single_vector,
        single_loop=vectorization.run_hvp_vector_single_loop,
        manual_batch=vectorization.run_hvp_vector_manual_batch,
        vmap=vectorization.run_hvp_vector_vmap,
    )


def run_hvp_single_vector(execution: runtime_values.StandardExecution) -> TensorTree:
    """Run the HVP path for a single vector.

    Returns:
        The HVP path for a single vector.
    """
    scalar_function = hvp_scalar_function(execution)

    if execution.path == runtime_values.HVP_REFERENCE_PATH:
        if _hvp_row_batch_size(execution.candidate.settings) is None:
            result = hvp_reverse_over_reverse_anchor(
                scalar_function,
                execution.params,
                execution.vector,
            )
        else:
            result = _run_hvp_row_batched_reverse(execution, scalar_function)
    elif execution.path == runtime_values.HVP_FUNCTIONAL_PATH:
        result = hvp_anchor(scalar_function, execution.params, execution.vector)
    elif execution.path == runtime_values.VHP_PATH:
        result = _run_hvp_vhp_path(execution)
    elif execution.path == runtime_values.HVP_FORWARD_AD_PATH:
        result = _run_hvp_forward_ad_path(execution)
    elif execution.path == runtime_values.HVP_LINEARIZE_GRAD_PATH:
        if execution.linearized_hvp is not None:
            result = execution.linearized_hvp(execution.vector)
        else:
            gradient_function = torch.func.grad(scalar_function)
            _, hvp_function = torch.func.linearize(
                gradient_function,
                execution.params,
            )
            result = hvp_function(execution.vector)
    else:
        result = hvp_jvp_grad_anchor(
            scalar_function,
            execution.params,
            execution.vector,
        )

    return result


def _run_hvp_row_batched_reverse(
    execution: runtime_values.StandardExecution,
    scalar_function: Callable[[ParameterTree], torch.Tensor],
) -> TensorTree:
    batch_size = _hvp_row_batch_size(execution.candidate.settings)

    if batch_size is None:
        message = "batch.hvp_row_batch_size is required"
        raise MaterializationError(message)

    active_params = runtime_values.grad_enabled_params(execution.params)
    parameter_leaves = tuple(active_params.values())
    value = scalar_function(active_params)
    gradient_leaves = torch.autograd.grad(
        value,
        parameter_leaves,
        create_graph=True,
        allow_unused=True,
    )
    gradient_flat = _flat_gradient_row(
        tuple(
            torch.zeros_like(leaf) if gradient is None else gradient
            for leaf, gradient in zip(parameter_leaves, gradient_leaves, strict=True)
        )
    )
    vector_tensor = runtime.parameter_order_vector(execution)
    result = torch.zeros_like(vector_tensor)

    for start in range(0, gradient_flat.numel(), batch_size):
        stop = min(start + batch_size, gradient_flat.numel())

        for row_index in range(start, stop):
            component = gradient_flat[row_index]

            if not component.requires_grad:
                continue

            row_gradients = torch.autograd.grad(
                component,
                parameter_leaves,
                retain_graph=True,
                allow_unused=True,
            )
            row = _flat_gradient_row(
                tuple(
                    torch.zeros_like(leaf) if gradient is None else gradient
                    for leaf, gradient in zip(
                        parameter_leaves,
                        row_gradients,
                        strict=True,
                    )
                )
            )
            result[row_index] = runtime.dot_runtime(
                execution.candidate.settings,
                row,
                vector_tensor,
            )

    runtime_values.require_finite_tensor(result, "row-batched HVP result")

    return runtime_values.wrap_flat_parameter_tree(active_params, result)


def hvp_scalar_function(
    execution: runtime_values.StandardExecution,
) -> Callable[[ParameterTree], torch.Tensor]:
    """Return the scalar loss function for HVP paths.

    Returns:
        The scalar loss function for HVP paths.
    """
    if execution.compiled_scalar_function is not None:
        return execution.compiled_scalar_function

    if _uses_stateful_module_call(execution):
        return _stateful_module_scalar_function(execution)

    scalar = runtime_values.scalar_objective(
        execution.operator, execution.scalar_objectives
    )

    def scalar_function(active_params: ParameterTree) -> torch.Tensor:
        settings = execution.candidate.settings

        return scalar(
            runtime.model_compute_tree(active_params, settings),
            runtime.model_compute_tree(execution.buffers, settings),
            runtime.model_compute_batch(execution.batch, settings),
            execution.context,
        )

    return scalar_function


def _run_hvp_forward_ad_path(
    execution: runtime_values.StandardExecution,
) -> TensorTree:
    scalar_function = hvp_scalar_function(execution)

    def gradient_function(active_params: ParameterTree) -> TensorTree:
        active_leaves = tree_leaves(active_params)
        value = scalar_function(active_params)
        gradients = torch.autograd.grad(
            value,
            active_leaves,
            allow_unused=True,
            create_graph=True,
        )

        return tree_from_leaves(
            active_params,
            tuple(
                torch.zeros_like(leaf) if gradient is None else gradient
                for leaf, gradient in zip(active_leaves, gradients, strict=True)
            ),
        )

    primal_params = runtime_values.grad_enabled_params(execution.params)
    vector_leaves = runtime_values.matching_vector_leaves(
        execution.params, execution.vector
    )

    with torch.autograd.forward_ad.dual_level():
        dual_params = {
            name: torch.autograd.forward_ad.make_dual(param, vector)
            for (name, param), vector in zip(
                primal_params.items(),
                vector_leaves,
                strict=True,
            )
        }
        dual_gradients = gradient_function(dual_params)

        def tangent_leaf(output: torch.Tensor) -> torch.Tensor:
            primal, tangent = torch.autograd.forward_ad.unpack_dual(output)

            if tangent is None:
                return torch.zeros_like(primal)

            return tangent

        result = tree_map(tangent_leaf, dual_gradients)

    runtime_values.require_finite_tree(result, "HVP result")

    return result


def _run_hvp_vhp_path(
    execution: runtime_values.StandardExecution,
) -> TensorTree:
    parameter_items = tuple(execution.params.items())
    vector_leaves = runtime_values.matching_vector_leaves(
        execution.params, execution.vector
    )
    parameter_leaves = tuple(tensor for _, tensor in parameter_items)
    compiled_scalar_function = hvp_scalar_function(execution)

    def scalar_function(*active_leaves: torch.Tensor) -> torch.Tensor:
        active_params = {
            name: active
            for (name, _), active in zip(parameter_items, active_leaves, strict=True)
        }

        return compiled_scalar_function(active_params)

    _, result_leaves = torch.autograd.functional.vhp(
        scalar_function,
        parameter_leaves,
        vector_leaves,
    )

    return tree_from_leaves(execution.params, result_leaves)


def run_per_example_gradient(
    execution: runtime_values.StandardExecution,
) -> TensorTree:
    """Run per example gradient.

    Returns:
        The per example gradient result.

    Raises:
        MaterializationError: If the declared inputs are invalid.
    """
    runtime_values.require_path(
        execution.operator.kind,
        execution.path,
        runtime_values.PER_EXAMPLE_GRADIENT_PATHS,
    )
    accumulation = execution.candidate.settings.get("per_example_gradient.accumulation")

    if accumulation == "stacked_leading_axis":
        matrix = fisher.score_gradient_matrix_from_operator_row(execution)
    elif accumulation == "blockwise_stacked":
        matrix = _per_example_gradient_matrix_blockwise(execution)
    else:
        message = (
            "per_example_gradient.accumulation is required for per_example_gradient"
        )
        raise MaterializationError(message)

    if matrix.ndim != runtime_values.MATRIX_DIMS:
        message = "per-example gradient output must be a matrix"
        raise MaterializationError(message)

    runtime_values.require_finite_tensor(matrix, "per-example gradient output")

    return runtime_values.wrap_flat_vector_batch(execution.params, matrix)


def _streaming_gradient_rows_loop(
    execution: runtime_values.StandardExecution,
) -> Iterator[torch.Tensor]:
    function = runtime_values.function_objective(
        execution.operator, execution.function_objectives
    )
    parameter_items = tuple(execution.params.items())
    active_leaves = tuple(
        tensor.detach().clone().requires_grad_(True) for _, tensor in parameter_items
    )

    def tensor_function(*leaves: torch.Tensor) -> torch.Tensor:
        active_params = {
            name: leaf for (name, _), leaf in zip(parameter_items, leaves, strict=True)
        }
        output = runtime.call_function_objective(execution, function, active_params)

        if not isinstance(output, torch.Tensor):
            message = "streaming gradient loop requires tensor objective output"
            raise MaterializationError(message)

        return output.reshape(-1)

    terms = tensor_function(*active_leaves)
    runtime_values.require_nonempty_per_example_terms(terms, "streaming gradient loop")

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

        yield _flat_gradient_row(gradients)


def _streaming_gradient_rows_torch_func(
    execution: runtime_values.StandardExecution,
) -> Iterator[torch.Tensor]:
    function = runtime_values.function_objective(
        execution.operator, execution.function_objectives
    )
    parameter_items = tuple(execution.params.items())
    active_params = {
        name: tensor.detach().clone().requires_grad_(True)
        for name, tensor in parameter_items
    }
    terms = _per_example_terms(function, active_params, execution)

    for index in range(terms.numel()):

        def single_loss(
            active: ParameterTree,
            term_index: int = index,
        ) -> torch.Tensor:
            active_terms = _per_example_terms(function, active, execution)

            return active_terms[term_index]

        gradients = torch.func.grad(single_loss)(active_params)
        yield torch.cat(
            tuple(gradients[name].reshape(-1) for name, _ in parameter_items)
        )


def _streaming_gradient_rows_backward(
    execution: runtime_values.StandardExecution,
) -> Iterator[torch.Tensor]:
    function = runtime_values.function_objective(
        execution.operator, execution.function_objectives
    )
    parameter_items = tuple(execution.params.items())
    active_params = {
        name: tensor.detach().clone().requires_grad_(True)
        for name, tensor in parameter_items
    }
    terms = _per_example_terms(function, active_params, execution)

    for index in range(terms.numel()):
        for param in active_params.values():
            param.grad = None

        terms[index].backward(retain_graph=index < terms.numel() - 1)
        gradients = tuple(
            torch.zeros_like(param) if param.grad is None else param.grad.detach()
            for param in active_params.values()
        )

        yield _flat_gradient_row(gradients)


def _path_builder_map(
    rows: Sequence[
        tuple[tuple[str, ...], Callable[[runtime_values.StandardExecution], Any]]
    ],
) -> dict[str, Callable[[runtime_values.StandardExecution], Any]]:
    result = {}

    for paths, builder in rows:
        for path in paths:
            result[path] = builder

    return result


STREAMING_GRADIENT_ROW_BUILDERS = _path_builder_map((
    (runtime_values.STREAMING_GRADIENT_LOOP_PATHS, _streaming_gradient_rows_loop),
    (
        runtime_values.STREAMING_GRADIENT_TORCH_FUNC_PATHS,
        _streaming_gradient_rows_torch_func,
    ),
    (
        runtime_values.STREAMING_GRADIENT_BACKWARD_PATHS,
        _streaming_gradient_rows_backward,
    ),
    (
        runtime_values.STREAMING_GRADIENT_VMAP_PATHS,
        vectorization.streaming_gradient_rows_vmap,
    ),
))


def _flat_gradient_row(gradients: Sequence[torch.Tensor]) -> torch.Tensor:
    return torch.cat(tuple(gradient.reshape(-1) for gradient in gradients))


def per_example_sliced_executions(
    execution: runtime_values.StandardExecution,
    label: str,
    batch_size: int,
) -> Iterator[runtime_values.StandardExecution]:
    """Yield per-example sliced executions for manual schedules.

    Yields:
        The per-example sliced execution for each example slice.
    """
    batch, batch_in_dims = vectorization.per_example_batch_in_dims(
        execution.batch,
        label,
    )
    example_count = runtime_values.per_example_batch_size(
        batch,
        batch_in_dims,
        label,
    )

    for start in range(0, example_count, batch_size):
        stop = min(start + batch_size, example_count)
        subbatch = runtime_values.per_example_batch_slice(
            batch, batch_in_dims, start, stop
        )
        yield dataclasses.replace(execution, batch=subbatch)


def _per_example_gradient_matrix_blockwise(
    execution: runtime_values.StandardExecution,
) -> torch.Tensor:
    return per_example_gradient_matrix_batched(
        execution,
        "per-example gradient blockwise stacking",
        _per_example_block_size(execution),
    )


def per_example_gradient_matrix_batched(
    execution: runtime_values.StandardExecution,
    label: str,
    batch_size: int,
) -> torch.Tensor:
    """Run the batched per-example gradient matrix path.

    Returns:
        The batched per-example gradient matrix path.
    """
    rows = [
        vectorization.per_example_gradient_matrix_without_manual_batch(subexecution)
        for subexecution in per_example_sliced_executions(execution, label, batch_size)
    ]

    return torch.cat(tuple(rows), dim=0)


def _per_example_block_size(execution: runtime_values.StandardExecution) -> int:
    key = "batch.per_example_block_size"

    return runtime_values.required_positive_int_setting(
        execution.candidate.settings,
        key,
        f"{key} must be a positive integer",
    )


def _per_example_gradient_matrix(
    execution: runtime_values.StandardExecution,
) -> torch.Tensor:
    function = runtime_values.function_objective(
        execution.operator, execution.function_objectives
    )
    parameter_items = tuple(execution.params.items())
    active_leaves = tuple(
        tensor.detach().clone().requires_grad_(True) for _, tensor in parameter_items
    )

    def tensor_function(*leaves: torch.Tensor) -> torch.Tensor:
        active_params = {
            name: leaf for (name, _), leaf in zip(parameter_items, leaves, strict=True)
        }
        output = runtime.call_function_objective(execution, function, active_params)

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


def _per_example_gradient_matrix_torch_func(
    execution: runtime_values.StandardExecution,
) -> torch.Tensor:
    function = runtime_values.function_objective(
        execution.operator, execution.function_objectives
    )
    parameter_items = tuple(execution.params.items())
    active_params = {
        name: tensor.detach().clone().requires_grad_(True)
        for name, tensor in parameter_items
    }
    terms = _per_example_terms(function, active_params, execution)
    gradient_rows = []

    for index in range(terms.numel()):

        def single_loss(
            active: ParameterTree,
            term_index: int = index,
        ) -> torch.Tensor:
            active_terms = _per_example_terms(function, active, execution)

            return active_terms[term_index]

        gradients = torch.func.grad(single_loss)(active_params)
        pieces = tuple(gradients[name].reshape(-1) for name, _ in parameter_items)
        gradient_rows.append(torch.cat(pieces))

    return torch.stack(gradient_rows)


def _per_example_gradient_matrix_backward(
    execution: runtime_values.StandardExecution,
) -> torch.Tensor:
    function = runtime_values.function_objective(
        execution.operator, execution.function_objectives
    )
    parameter_items = tuple(execution.params.items())
    active_params = {
        name: tensor.detach().clone().requires_grad_(True)
        for name, tensor in parameter_items
    }
    terms = _per_example_terms(function, active_params, execution)
    gradient_rows = []

    for index in range(terms.numel()):
        for param in active_params.values():
            param.grad = None

        terms[index].backward(retain_graph=index < terms.numel() - 1)
        pieces = tuple(
            torch.zeros_like(param).reshape(-1)
            if param.grad is None
            else param.grad.detach().reshape(-1)
            for param in active_params.values()
        )
        gradient_rows.append(torch.cat(pieces))

    return torch.stack(gradient_rows)


def _per_example_terms(
    function: FunctionObjective,
    active_params: ParameterTree,
    execution: runtime_values.StandardExecution,
) -> torch.Tensor:
    output = runtime.call_function_objective(execution, function, active_params)

    if not isinstance(output, torch.Tensor):
        message = "per-example gradient path requires tensor objective output"
        raise MaterializationError(message)

    terms = output.reshape(-1)

    runtime_values.require_nonempty_per_example_terms(
        terms, "per-example gradient path"
    )

    return terms


def per_example_gradient_matrix_from_builders(
    execution: runtime_values.StandardExecution,
    builders: Mapping[str, Callable[[runtime_values.StandardExecution], torch.Tensor]],
    message: str,
) -> torch.Tensor:
    """Run the per-example gradient matrix from declared row builders.

    Returns:
        The per-example gradient matrix from declared row builders.

    Raises:
        MaterializationError: If the declared inputs are invalid.
    """
    builder = builders.get(execution.path)

    if builder is None:
        raise MaterializationError(message)

    return builder(execution)


PER_EXAMPLE_GRADIENT_WITHOUT_MANUAL_BUILDERS = _path_builder_map((
    (runtime_values.PER_EXAMPLE_GRADIENT_LOOP_PATHS, _per_example_gradient_matrix),
    (
        runtime_values.PER_EXAMPLE_GRADIENT_TORCH_FUNC_PATHS,
        _per_example_gradient_matrix_torch_func,
    ),
    (
        runtime_values.PER_EXAMPLE_GRADIENT_BACKWARD_PATHS,
        _per_example_gradient_matrix_backward,
    ),
    (
        runtime_values.PER_EXAMPLE_GRADIENT_VMAP_PATHS,
        vectorization.per_example_gradient_matrix_vmap,
    ),
))


def per_example_gradient_spec_runtime_path(candidate: Candidate) -> str | None:
    """Return the per example gradient spec runtime path.

    Returns:
        The per example gradient spec runtime path.

    Raises:
        MaterializationError: If the declared inputs are invalid.
    """
    grad_key = runtime_values.SPEC_PATH_KEYS["per_example_gradient"]
    accumulation_key = "per_example_gradient.accumulation"
    grad_path = candidate.settings.get(grad_key)
    accumulation = candidate.settings.get(accumulation_key)

    if accumulation not in {"stacked_leading_axis", "blockwise_stacked"}:
        message = (
            f"per_example_gradient.accumulation value is not lowered: {accumulation}"
        )
        raise MaterializationError(message)

    if grad_key not in candidate.settings:
        return None

    if not isinstance(grad_path, str):
        message = "per_example_gradient.grad_path must be a string"
        raise MaterializationError(message)

    path_map = runtime_values.SPEC_PATH_TO_RUNTIME["per_example_gradient"]
    path = path_map.get(grad_path)

    if path is None:
        message = f"{grad_key} value is not lowered by standard runtime: {grad_path}"
        raise MaterializationError(message)

    return path


def _uses_stateful_module_call(execution: runtime_values.StandardExecution) -> bool:
    return execution.candidate.settings.get("call.path") == "stateful_module"


def _stateful_module_scalar_function(
    execution: runtime_values.StandardExecution,
) -> Callable[[ParameterTree], torch.Tensor]:
    def scalar_function(active_params: ParameterTree) -> torch.Tensor:
        output = _call_stateful_module(execution, active_params)

        if not isinstance(output, torch.Tensor) or output.ndim != 0:
            message = "stateful module scalar objective must return a scalar tensor"
            raise MaterializationError(message)

        return output

    return scalar_function


def _stateful_module_tensor_function(
    execution: runtime_values.StandardExecution,
) -> Callable[[ParameterTree], TensorTree]:
    def tensor_function(active_params: ParameterTree) -> TensorTree:
        output = _call_stateful_module(execution, active_params)

        return runtime_values.checked_function_output(
            execution.candidate.settings,
            output,
            "stateful module output",
        )

    return tensor_function


def _call_stateful_module(
    execution: runtime_values.StandardExecution,
    active_params: ParameterTree,
) -> object:
    if execution.module is None or execution.module_call is None:
        message = "stateful module execution requires module and module_call"
        raise MaterializationError(message)

    settings = execution.candidate.settings
    model_params = runtime.stateful_model_tree(active_params, settings)
    model_buffers = runtime.stateful_model_tree(execution.buffers, settings)
    model_batch = runtime.stateful_model_batch(execution.batch, settings)
    slots = runtime_values.replace_module_state(
        execution.module, model_params, model_buffers
    )

    try:
        if execution.compiled_model_forward is None:
            output = runtime_values.invoke_stateful_module(
                execution.module,
                execution.module_call,
                model_batch,
            )
        else:
            output = execution.compiled_model_forward(model_batch)
    finally:
        runtime_values.restore_module_state(slots)

    return runtime_values.select_stateful_module_output(
        output,
        execution.module_call,
        execution.candidate.settings,
    )


def require_per_example_batch_size_settings(
    path: str | None,
    settings: Mapping[str, Any],
) -> None:
    """Validate per example batch size settings."""
    runtime_values.require_per_example_batch_size_setting(
        path,
        settings,
        "batch.fisher_sample_batch_size",
        runtime_values.FISHER_SAMPLE_VMAP_PATHS,
        runtime_values.FISHER_SAMPLE_MANUAL_PER_EXAMPLE_PATHS,
    )
    runtime_values.require_per_example_batch_size_setting(
        path,
        settings,
        "batch.empirical_example_batch_size",
        {runtime_values.EMPIRICAL_FISHER_GRADIENT_VMAP_PATH},
        set(runtime_values.EMPIRICAL_FISHER_PER_EXAMPLE_MANUAL_BATCH_PATHS),
    )
    _require_per_example_gradient_block_size_setting(path, settings)


def _require_per_example_gradient_block_size_setting(
    path: str | None,
    settings: Mapping[str, Any],
) -> None:
    key = "batch.per_example_block_size"
    accumulation = settings.get("per_example_gradient.accumulation")

    if key not in settings:
        if (
            accumulation == "blockwise_stacked"
            and path in runtime_values.PER_EXAMPLE_GRADIENT_PATHS
        ):
            message = f"{key} is required for blockwise_stacked"
            raise MaterializationError(message)

        return

    if (
        accumulation != "blockwise_stacked"
        or path not in runtime_values.PER_EXAMPLE_GRADIENT_PATHS
    ):
        message = f"{key} requires per_example_gradient.accumulation=blockwise_stacked"
        raise MaterializationError(message)

    runtime_values.required_positive_int_setting(
        settings,
        key,
        f"{key} must be a positive integer",
    )


def require_gradient_graph_schedule_settings(
    operator: OperatorSpec,
    settings: Mapping[str, Any],
) -> None:
    """Validate gradient graph schedule settings.

    Raises:
        MaterializationError: If the declared inputs are invalid.
    """
    graph_schedule = settings.get("gradient.graph_schedule")

    if graph_schedule is None:
        return

    if operator.kind != "gradient":
        message = "gradient.graph_schedule applies only to gradient rows"
        raise MaterializationError(message)

    if graph_schedule not in {"build_once", "rebuild_per_call"}:
        message = f"gradient.graph_schedule is unsupported: {graph_schedule}"
        raise MaterializationError(message)


def require_gradient_value_reuse_settings(
    operator: OperatorSpec,
    path: str,
    settings: Mapping[str, Any],
) -> None:
    """Validate gradient value reuse settings."""
    runtime_values.require_path_coupled_reuse_setting(
        operator,
        path,
        settings,
        operator_kind="gradient",
        setting_key="gradient.value_reuse",
        default_value="gradient_only",
        required_value="gradient_and_primal_value",
        required_path=runtime_values.GRADIENT_TORCH_FUNC_VALUE_PATH,
        path_message="gradient_and_primal_value requires torch_func_grad_and_value",
    )


def require_jvp_linearize_reuse_settings(
    operator: OperatorSpec,
    path: str,
    settings: Mapping[str, Any],
) -> None:
    """Validate jvp linearize reuse settings."""
    runtime_values.require_path_coupled_reuse_setting(
        operator,
        path,
        settings,
        operator_kind="jvp",
        setting_key="jvp.linearize_reuse",
        default_value="none",
        required_value="reuse_at_same_primal",
        required_path=runtime_values.JVP_LINEARIZE_PATH,
        path_message="reuse_at_same_primal requires torch_func_linearize",
    )


def require_vjp_closure_reuse_settings(
    operator: OperatorSpec,
    path: str,
    settings: Mapping[str, Any],
) -> None:
    """Validate vjp closure reuse settings."""
    runtime_values.require_path_coupled_reuse_setting(
        operator,
        path,
        settings,
        operator_kind="vjp",
        setting_key="vjp.closure_reuse",
        default_value="none",
        required_value="reuse_vjp_closure_at_same_primal",
        required_path=runtime_values.VJP_PATH,
        path_message="reuse_vjp_closure_at_same_primal requires torch_func_vjp",
    )


def _hvp_row_batch_size(settings: Mapping[str, Any]) -> int | None:
    key = "batch.hvp_row_batch_size"
    return runtime_values.optional_positive_int_setting(
        settings,
        key,
        f"{key} must be a positive integer",
    )


def require_hvp_row_batch_size_settings(
    operator: OperatorSpec,
    path: str,
    settings: Mapping[str, Any],
) -> None:
    """Validate hvp row batch size settings.

    Raises:
        MaterializationError: If the declared inputs are invalid.
    """
    batch_size = _hvp_row_batch_size(settings)

    if batch_size is None:
        return

    if operator.kind != "hvp":
        message = "batch.hvp_row_batch_size applies only to HVP rows"
        raise MaterializationError(message)

    if path != runtime_values.HVP_REFERENCE_PATH:
        message = "batch.hvp_row_batch_size requires reverse_over_reverse"
        raise MaterializationError(message)


def require_hvp_reuse_settings(
    operator: OperatorSpec,
    path: str,
    settings: Mapping[str, Any],
) -> None:
    """Validate hvp reuse settings.

    Raises:
        MaterializationError: If the declared inputs are invalid.
    """
    graph_schedule = settings.get("hvp.graph_schedule")
    primal_reuse = settings.get("hvp.primal_reuse")
    gradient_reuse = settings.get("hvp.gradient_reuse")

    if operator.kind != "hvp":
        if graph_schedule is not None:
            message = "hvp.graph_schedule applies only to HVP rows"
            raise MaterializationError(message)

        if primal_reuse is not None:
            message = "hvp.primal_reuse applies only to HVP rows"
            raise MaterializationError(message)

        if gradient_reuse is not None:
            message = "hvp.gradient_reuse applies only to HVP rows"
            raise MaterializationError(message)

        return

    if graph_schedule == "retain_graph_across_vectors":
        _require_hvp_reverse_reuse_settings(path, settings)

        if primal_reuse != "reuse_primal":
            message = (
                "retain_graph_across_vectors requires hvp.primal_reuse=reuse_primal"
            )
            raise MaterializationError(message)
    elif graph_schedule not in {None, "rebuild_graph_per_vector"}:
        message = f"hvp.graph_schedule is unsupported: {graph_schedule}"
        raise MaterializationError(message)

    if primal_reuse == "reuse_primal":
        _require_hvp_reverse_reuse_settings(path, settings)
    elif primal_reuse not in {None, "recompute_primal"}:
        message = f"hvp.primal_reuse is unsupported: {primal_reuse}"
        raise MaterializationError(message)

    if gradient_reuse in {None, "recompute_gradient"}:
        return

    if gradient_reuse != "reuse_gradient_closure":
        message = f"hvp.gradient_reuse is unsupported: {gradient_reuse}"
        raise MaterializationError(message)

    if path != runtime_values.HVP_LINEARIZE_GRAD_PATH:
        message = "reuse_gradient_closure requires linearize_grad"
        raise MaterializationError(message)


def _require_hvp_reverse_reuse_settings(
    path: str,
    settings: Mapping[str, Any],
) -> None:
    if path != runtime_values.HVP_REFERENCE_PATH:
        message = "HVP graph and primal reuse require reverse_over_reverse"
        raise MaterializationError(message)

    if settings.get("vectorization.mode") != "single_loop":
        message = "HVP graph and primal reuse require vectorization.mode=single_loop"
        raise MaterializationError(message)

    if "vectorization.in_dims" not in settings:
        message = "HVP graph and primal reuse require vectorization.in_dims"
        raise MaterializationError(message)


def jvp_output_template(execution: runtime_values.StandardExecution) -> TensorTree:
    """Return the jvp output template.

    Returns:
        The jvp output template.
    """
    function = runtime_values.function_objective(
        execution.operator,
        execution.function_objectives,
    )

    def callback() -> TensorTree:
        return runtime.call_function_objective(execution, function, execution.params)

    return runtime.run_with_backend_settings(
        execution.candidate.settings,
        lambda: runtime_values.run_with_call_grad_mode(
            execution.candidate.settings, callback
        ),
    )


def per_example_gradient_anchor_settings(
    operator: OperatorSpec,
    path: str,
) -> dict[str, Any]:
    """Return the per example gradient anchor settings.

    Returns:
        The per example gradient anchor settings.
    """
    if operator.kind != "per_example_gradient":
        return {}

    if path != runtime_values.PER_EXAMPLE_GRADIENT_LOOP_PATH:
        return {}

    return {"per_example_gradient.accumulation": "stacked_leading_axis"}
