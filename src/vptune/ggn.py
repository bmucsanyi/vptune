"""GGNVP lowerings for the standard runtime.

Dense and matrix-free Gauss-Newton vector products, loss-Hessian
paths, PSD enforcement, and the closed-form softmax CE/KL products.
"""

import dataclasses
from collections.abc import Callable, Mapping
from typing import Any

import torch

from vptune import derivatives, layout, runtime, runtime_values, vectorization
from vptune.anchors import (
    jvp_anchor,
)
from vptune.data import (
    Batch,
    Candidate,
    OperationFactory,
    OperatorSpec,
    ParameterSurface,
    ParameterTree,
)
from vptune.errors import (
    MaterializationError,
)
from vptune.tensor_tree import (
    TensorTree,
)


def ggn_inner_product_measurements(
    operator: OperatorSpec,
    candidate: Candidate,
    batch: Batch,
    vector: TensorTree,
    candidate_output: TensorTree,
    anchor_candidate: Candidate,
    candidate_factory: OperationFactory,
    parameter_surface: ParameterSurface | None,
) -> dict[str, float]:
    """Return the ggn inner product measurements.

    Returns:
        The ggn inner product measurements.
    """
    if operator.kind != "ggnvp":
        return {}

    symmetry_vector = runtime_values.batch_tree(batch, "symmetry_vector")
    runtime_values.require_min_probe_norm(vector, "ggn reference vector")
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

    return {"inner_abs_diff": float((left - right).abs().item())}


def prepare_ggn_compile_boundary_execution(
    execution: runtime_values.StandardExecution,
    settings: Mapping[str, Any],
    boundary: str,
) -> runtime_values.StandardExecution | None:
    """Prepare ggn compile boundary execution.

    Returns:
        The ggn compile boundary execution result.
    """
    if execution.operator.kind != "ggnvp":
        return None

    if boundary == "ggn_loss_hessian_product":
        return _prepare_ggn_loss_product_compile_boundary(
            execution,
            settings,
            lambda output, output_jvp: _ggn_loss_hessian_product_by_path(
                execution,
                output,
                output_jvp,
            ),
        )

    if boundary == "ggn_jvp":
        return _prepare_ggn_jvp_compile_boundary(
            execution,
            settings,
            lambda: _ggn_output_and_jvp_by_path(
                execution,
                _ggn_tensor_function(execution),
            ),
        )

    if boundary == "ggn_vjp":
        return _prepare_ggn_vjp_compile_boundary(
            execution,
            settings,
            lambda output_cotangent: _run_ggnvp_vjp_by_path(
                execution,
                _ggn_tensor_function(execution),
                output_cotangent,
            ),
        )

    return None


def _prepare_ggn_loss_product_compile_boundary(
    execution: runtime_values.StandardExecution,
    settings: Mapping[str, Any],
    builder: Callable[[TensorTree, TensorTree], TensorTree],
) -> runtime_values.StandardExecution:
    runtime.require_compiled_execution(execution, settings)
    warm_inputs = (
        _ggn_loss_product_warm_inputs(execution)
        if settings.get("compile.cache_state") == "warm_cache"
        else None
    )
    compiled_loss_product = _compiled_ggn_loss_product_operation(
        settings,
        builder,
        None if warm_inputs is None else warm_inputs[0],
        None if warm_inputs is None else warm_inputs[1],
    )

    return dataclasses.replace(
        execution,
        compiled_ggn_loss_product=compiled_loss_product,
    )


def _prepare_ggn_jvp_compile_boundary(
    execution: runtime_values.StandardExecution,
    settings: Mapping[str, Any],
    builder: Callable[[], tuple[TensorTree, TensorTree]],
) -> runtime_values.StandardExecution:
    runtime.require_compiled_execution(execution, settings)
    compiled_ggn_jvp = _compiled_ggn_jvp_operation(settings, builder)

    return dataclasses.replace(
        execution,
        compiled_ggn_jvp=compiled_ggn_jvp,
    )


def _prepare_ggn_vjp_compile_boundary(
    execution: runtime_values.StandardExecution,
    settings: Mapping[str, Any],
    builder: Callable[[TensorTree], TensorTree],
) -> runtime_values.StandardExecution:
    runtime.require_compiled_execution(execution, settings)
    warm_output_cotangent = (
        _ggn_vjp_warm_input(execution)
        if settings.get("compile.cache_state") == "warm_cache"
        else None
    )
    compiled_vjp = _compiled_ggn_vjp_operation(
        settings,
        builder,
        warm_output_cotangent,
    )

    return dataclasses.replace(
        execution,
        compiled_ggn_vjp=compiled_vjp,
    )


def _compiled_ggn_loss_product_operation(
    settings: Mapping[str, Any],
    operation: Callable[[TensorTree, TensorTree], TensorTree],
    warm_output: TensorTree | None,
    warm_output_jvp: TensorTree | None,
) -> Callable[[TensorTree, TensorTree], TensorTree]:
    compiled = runtime.compiled_callable(
        settings,
        operation,
        use_backend_settings=False,
    )
    runtime.warm_compiled_cache(
        settings,
        compiled,
        warm_output,
        warm_output_jvp,
        error_message="GGN loss-product warm cache requires warm inputs",
    )

    return compiled


def _compiled_ggn_jvp_operation(
    settings: Mapping[str, Any],
    operation: Callable[[], tuple[TensorTree, TensorTree]],
) -> Callable[[], tuple[TensorTree, TensorTree]]:
    compiled = runtime.compiled_callable(
        settings,
        operation,
        use_backend_settings=True,
    )
    runtime.warm_compiled_cache(settings, compiled)

    return compiled


def _compiled_ggn_vjp_operation(
    settings: Mapping[str, Any],
    operation: Callable[[TensorTree], TensorTree],
    warm_output_cotangent: TensorTree | None,
) -> Callable[[TensorTree], TensorTree]:
    compiled = runtime.compiled_callable(
        settings,
        operation,
        use_backend_settings=False,
    )
    runtime.warm_compiled_cache(
        settings,
        compiled,
        warm_output_cotangent,
        error_message="GGN VJP warm cache requires warm output cotangent",
    )

    return compiled


def ggn_compile_boundary_supported(
    boundary: str,
    settings: Mapping[str, Any],
) -> bool:
    """Return the ggn compile boundary supported.

    Returns:
        The ggn compile boundary supported.
    """
    if boundary == "ggn_full_product":
        return True

    if settings.get("vectorization.mode") in {"single_loop", "manual_batch", "vmap"}:
        return False

    if boundary == "ggn_jvp":
        return settings.get("ggn.jvp_path") in {
            "torch_func_jvp",
            "forward_ad_dual",
            "torch_func_linearize",
        }

    if boundary == "ggn_loss_hessian_product":
        return settings.get("ggn.jvp_path") in {
            "torch_func_jvp",
            "forward_ad_dual",
            "torch_func_linearize",
        }

    if boundary != "ggn_vjp":
        return False

    return settings.get("ggn.vjp_path") in {
        "torch_func_vjp",
        "autograd_grad_outputs",
    }


def ggn_declared_batch_inputs(
    operator: OperatorSpec,
    candidate: Candidate,
    declared: tuple[str, ...],
    phase: str,
) -> tuple[str, ...]:
    """Return the ggn declared batch inputs.

    Returns:
        The ggn declared batch inputs.
    """
    if operator.kind != "ggnvp":
        return declared

    if phase != "operation":
        return declared

    if candidate.settings.get("ggn.loss_hessian_path") != "closed_form_softmax_ce_kl":
        return declared

    return tuple(key for key in declared if key != "loss_hessian")


def augment_ggn_dense_cross_check(
    operator: OperatorSpec,
    candidate: Candidate,
    batch: Batch,
    vector: TensorTree,
    candidate_output: TensorTree,
    candidate_factory: OperationFactory,
    measurements: dict[str, Any],
) -> None:
    """Augment ggn dense cross check."""
    if operator.kind != "ggnvp":
        return

    dense_candidate = dataclasses.replace(
        candidate,
        settings=runtime.anchor_settings(
            operator, candidate, runtime_values.GGN_DENSE_PATH
        ),
    )
    dense_output = candidate_factory(dense_candidate, batch, vector)()
    errors = layout.layout_aware_tree_error_measurements(
        candidate,
        candidate_output,
        dense_output,
    )
    measurements["max_abs_diff"] = max(
        float(measurements["max_abs_diff"]),
        float(errors["max_abs_diff"]),
    )
    measurements["max_rel_diff"] = max(
        float(measurements["max_rel_diff"]),
        float(errors["max_rel_diff"]),
    )
    measurements["dense_anchor_errors"] = dict(errors)


def run_ggnvp(execution: runtime_values.StandardExecution) -> TensorTree:
    """Run ggnvp.

    Returns:
        The ggnvp result.
    """
    if (
        execution.compiled_inner is not None
        and execution.candidate.settings.get("compile.boundary") == "ggn_full_product"
    ):
        return execution.compiled_inner()

    return run_ggnvp_by_path(execution)


def run_ggnvp_by_path(execution: runtime_values.StandardExecution) -> TensorTree:
    """Run ggnvp by path.

    Returns:
        The ggnvp by path result.
    """
    return vectorization.run_single_vectorized_by_path(
        execution,
        (
            runtime_values.GGN_DENSE_PATH,
            runtime_values.GGN_JVP_HESSIAN_VJP_PATH,
            runtime_values.GGN_FORWARD_AD_HESSIAN_VJP_PATH,
            runtime_values.GGN_LINEARIZE_HESSIAN_VJP_PATH,
        ),
        _run_ggnvp_single_vector,
        _run_ggnvp_single_loop_vector,
        _run_ggnvp_vector_vmap,
    )


def _run_ggnvp_single_vector(execution: runtime_values.StandardExecution) -> TensorTree:
    if execution.path in {
        runtime_values.GGN_JVP_HESSIAN_VJP_PATH,
        runtime_values.GGN_FORWARD_AD_HESSIAN_VJP_PATH,
        runtime_values.GGN_LINEARIZE_HESSIAN_VJP_PATH,
    }:
        return _run_ggnvp_jvp_hessian_vjp(execution)

    function = runtime_values.function_objective(
        execution.operator, execution.function_objectives
    )
    parameter_items = tuple(execution.params.items())
    parameter_leaves = tuple(tensor for _, tensor in parameter_items)
    vector_tensor = runtime.parameter_order_vector(execution)

    def tensor_function(*active_leaves: torch.Tensor) -> torch.Tensor:
        active_params = {
            name: active
            for (name, _), active in zip(parameter_items, active_leaves, strict=True)
        }
        output = runtime.call_function_objective(execution, function, active_params)

        if not isinstance(output, torch.Tensor):
            message = "dense GGNVP requires tensor function output"
            raise MaterializationError(message)

        return output.reshape(-1)

    output = tensor_function(*parameter_leaves)
    runtime_values.require_finite_tensor(vector_tensor, "GGN vector")
    ggn_batch_size = _ggn_batch_size(execution.candidate.settings)

    if ggn_batch_size is not None:
        result = _dense_ggnvp_batched_rows(
            execution,
            tensor_function,
            parameter_leaves,
            vector_tensor,
            ggn_batch_size,
        )

        return runtime_values.wrap_flat_vector(execution.params, result)

    jacobian = runtime_values.dense_jacobian_tree(
        tensor_function,
        parameter_leaves,
        output.numel(),
    )
    runtime_values.require_finite_tensor(jacobian, "GGN jacobian")
    output_vector = runtime.parameter_blocked_matrix_vector_product(
        jacobian,
        vector_tensor.reshape(-1),
        execution.candidate.settings,
        execution.parameter_surface,
    )
    output_cotangent = _ggn_loss_hessian_product_by_path(
        execution,
        output,
        runtime_values.wrap_flat_vector(output, output_vector),
    )
    loss_vector = runtime_values.flatten_vector(output_cotangent)
    result = _jacobian_transpose_product(
        jacobian,
        loss_vector,
        execution.candidate.settings,
        execution.parameter_surface,
    )
    runtime_values.require_finite_tensor(result, "GGN result")

    return runtime_values.wrap_flat_vector(execution.params, result)


def _dense_ggnvp_batched_rows(
    execution: runtime_values.StandardExecution,
    tensor_function: Callable[..., torch.Tensor],
    parameter_leaves: tuple[torch.Tensor, ...],
    vector_tensor: torch.Tensor,
    batch_size: int,
) -> torch.Tensor:
    settings = execution.candidate.settings
    active_leaves = tuple(
        parameter.detach().clone().requires_grad_(True)
        for parameter in parameter_leaves
    )
    output = tensor_function(*active_leaves)
    output_width = output.numel()
    parameter_width = vector_tensor.numel()
    output_vector = torch.empty(
        output_width,
        dtype=output.dtype,
        device=output.device,
    )

    for start in range(0, output_width, batch_size):
        stop = min(start + batch_size, output_width)
        jacobian_block = runtime_values.dense_jacobian_row_block(
            output,
            active_leaves,
            parameter_width,
            start,
            stop,
        )
        output_vector[start:stop] = runtime.parameter_blocked_matrix_vector_product(
            jacobian_block,
            vector_tensor.reshape(-1),
            settings,
            execution.parameter_surface,
        )

    output_cotangent = _ggn_loss_hessian_product_by_path(
        execution,
        output,
        runtime_values.wrap_flat_vector(output, output_vector),
    )
    loss_vector = runtime_values.flatten_vector(output_cotangent)
    result = torch.zeros(
        parameter_width,
        dtype=loss_vector.dtype,
        device=loss_vector.device,
    )

    for start in range(0, output_width, batch_size):
        stop = min(start + batch_size, output_width)
        jacobian_block = runtime_values.dense_jacobian_row_block(
            output,
            active_leaves,
            parameter_width,
            start,
            stop,
        )
        result = result + _jacobian_transpose_product(
            jacobian_block,
            loss_vector[start:stop],
            settings,
            execution.parameter_surface,
        )

    runtime_values.require_finite_tensor(result, "GGN batched result")

    return result


def _run_ggnvp_single_loop_vector(
    execution: runtime_values.StandardExecution,
) -> TensorTree:
    return _run_ggnvp_single_vector(
        dataclasses.replace(
            execution,
            compiled_inner=None,
            compiled_ggn_loss_product=None,
            compiled_ggn_vjp=None,
        )
    )


def _run_ggnvp_vector_vmap(execution: runtime_values.StandardExecution) -> TensorTree:
    if execution.path not in runtime_values.GGN_VECTOR_VMAP_PATHS:
        message = "vectorization.mode=vmap requires a torch.func GGN JVP path"
        raise MaterializationError(message)

    if execution.candidate.settings.get("ggn.vjp_path") != "torch_func_vjp":
        message = "vectorization.mode=vmap requires ggn.vjp_path=torch_func_vjp"
        raise MaterializationError(message)

    tensor_function = _ggn_tensor_function(execution)

    if execution.path == runtime_values.GGN_LINEARIZE_HESSIAN_VJP_PATH:
        output, jvp_function = torch.func.linearize(tensor_function, execution.params)
    else:
        output = tensor_function(execution.params)

        def jvp_function(vector: TensorTree) -> TensorTree:
            return jvp_anchor(tensor_function, execution.params, vector)

    runtime_values.require_finite_tree(output, "GGN output")
    _require_ggn_loss_hessian_vector_vmap_inputs(execution, output)
    pullback = derivatives.vjp_pullback(tensor_function, execution.params)

    def ggn_function(vector: TensorTree) -> TensorTree:
        output_jvp = jvp_function(vector)
        output_jvp = _runtime_intermediate_residency_tree(
            output_jvp,
            execution.candidate.settings,
            execution.intermediate_transform,
        )
        output_cotangent = _ggn_loss_hessian_product_unchecked(
            execution,
            output,
            output_jvp,
        )
        output_cotangent = _runtime_intermediate_residency_tree(
            output_cotangent,
            execution.candidate.settings,
            execution.intermediate_transform,
        )
        (result,) = pullback(output_cotangent)

        return result

    result = vectorization.run_vector_vmap(execution, ggn_function)
    runtime_values.require_finite_tree(result, "GGN result")

    return result


def _require_ggn_loss_hessian_vector_vmap_inputs(
    execution: runtime_values.StandardExecution,
    output: TensorTree,
) -> None:
    if (
        execution.candidate.settings.get("ggn.loss_hessian_path")
        == "closed_form_softmax_ce_kl"
    ):
        return

    flat_output = runtime_values.flatten_vector(output)
    loss_hessian = runtime_values.batch_tensor(execution.batch, "loss_hessian")
    runtime_values.require_loss_hessian_shape(loss_hessian, flat_output.numel())
    runtime_values.require_finite_tensor(loss_hessian, "loss_hessian")


def _run_ggnvp_jvp_hessian_vjp(
    execution: runtime_values.StandardExecution,
) -> TensorTree:
    tensor_function = _ggn_tensor_function(execution)
    output, output_jvp = _ggn_output_and_jvp(execution, tensor_function)
    output_jvp = _runtime_intermediate_residency_tree(
        output_jvp,
        execution.candidate.settings,
        execution.intermediate_transform,
    )

    if _ggn_recomputes_jvp(execution.candidate.settings):
        _, output_jvp = _ggn_output_and_jvp(execution, tensor_function)
        output_jvp = _runtime_intermediate_residency_tree(
            output_jvp,
            execution.candidate.settings,
            execution.intermediate_transform,
        )

    output_cotangent = _ggn_loss_hessian_product(
        execution,
        output,
        output_jvp,
    )
    output_cotangent = _runtime_intermediate_residency_tree(
        output_cotangent,
        execution.candidate.settings,
        execution.intermediate_transform,
    )

    if _ggn_recomputes_output_cotangent(execution.candidate.settings):
        output_cotangent = _ggn_loss_hessian_product(
            execution,
            output,
            output_jvp,
        )
        output_cotangent = _runtime_intermediate_residency_tree(
            output_cotangent,
            execution.candidate.settings,
            execution.intermediate_transform,
        )

    runtime_values.require_finite_tree(output_cotangent, "GGN output cotangent")

    result = _run_ggnvp_vjp(execution, tensor_function, output_cotangent)
    runtime_values.require_finite_tree(result, "GGN result")

    return result


def _ggn_tensor_function(
    execution: runtime_values.StandardExecution,
) -> Callable[[ParameterTree], TensorTree]:
    function = runtime_values.function_objective(
        execution.operator, execution.function_objectives
    )

    def tensor_function(active_params: ParameterTree) -> TensorTree:
        return runtime.call_function_objective(execution, function, active_params)

    return tensor_function


def _ggn_loss_product_warm_inputs(
    execution: runtime_values.StandardExecution,
) -> tuple[TensorTree, TensorTree]:
    tensor_function = _ggn_tensor_function(execution)
    output, output_jvp = _ggn_output_and_jvp_by_path(execution, tensor_function)
    output_jvp = _runtime_intermediate_residency_tree(
        output_jvp,
        execution.candidate.settings,
        execution.intermediate_transform,
    )

    return output, output_jvp


def _ggn_vjp_warm_input(execution: runtime_values.StandardExecution) -> TensorTree:
    output, output_jvp = _ggn_loss_product_warm_inputs(execution)

    output_cotangent = _ggn_loss_hessian_product_by_path(execution, output, output_jvp)

    return _runtime_intermediate_residency_tree(
        output_cotangent,
        execution.candidate.settings,
        execution.intermediate_transform,
    )


def _ggn_output_and_jvp(
    execution: runtime_values.StandardExecution,
    tensor_function: Callable[[ParameterTree], TensorTree],
) -> tuple[TensorTree, TensorTree]:
    if (
        execution.compiled_ggn_jvp is not None
        and execution.candidate.settings.get("compile.boundary") == "ggn_jvp"
    ):
        return execution.compiled_ggn_jvp()

    return _ggn_output_and_jvp_by_path(execution, tensor_function)


def _ggn_output_and_jvp_by_path(
    execution: runtime_values.StandardExecution,
    tensor_function: Callable[[ParameterTree], TensorTree],
) -> tuple[TensorTree, TensorTree]:
    if execution.path == runtime_values.GGN_LINEARIZE_HESSIAN_VJP_PATH:
        output, jvp_function = torch.func.linearize(tensor_function, execution.params)

        return output, jvp_function(execution.vector)

    if execution.path == runtime_values.GGN_FORWARD_AD_HESSIAN_VJP_PATH:
        return _forward_ad_output_and_jvp(
            tensor_function,
            execution.params,
            execution.vector,
        )

    jvp_result = torch.func.jvp(
        tensor_function,
        (execution.params,),
        (execution.vector,),
    )

    return jvp_result[0], jvp_result[1]


def _forward_ad_output_and_jvp(
    function: Callable[[ParameterTree], TensorTree],
    params: ParameterTree,
    vector: TensorTree,
) -> tuple[TensorTree, TensorTree]:
    vector_params = runtime_values.parameter_tree_from_tensor_tree(vector, "GGN vector")

    with torch.autograd.forward_ad.dual_level():
        dual_params = {
            name: torch.autograd.forward_ad.make_dual(params[name], vector_params[name])
            for name in params
        }
        dual_output = function(dual_params)

        return _split_forward_ad_dual_tree(dual_output)


def _split_forward_ad_dual_tree(tree: TensorTree) -> tuple[TensorTree, TensorTree]:
    if isinstance(tree, torch.Tensor):
        primal, tangent = torch.autograd.forward_ad.unpack_dual(tree)

        if tangent is None:
            return primal, torch.zeros_like(primal)

        return primal, tangent

    if isinstance(tree, tuple):
        pairs = tuple(_split_forward_ad_dual_tree(child) for child in tree)

        return (
            tuple(pair[0] for pair in pairs),
            tuple(pair[1] for pair in pairs),
        )

    if isinstance(tree, dict):
        pairs = {key: _split_forward_ad_dual_tree(value) for key, value in tree.items()}

        return (
            {key: pair[0] for key, pair in pairs.items()},
            {key: pair[1] for key, pair in pairs.items()},
        )

    message = "forward AD output must be a tensor tree"
    raise MaterializationError(message)


def _ggn_recomputes_jvp(settings: Mapping[str, Any]) -> bool:
    return settings.get("ggn.jvp_reuse") == "recompute_jvp"


def _ggn_recomputes_output_cotangent(settings: Mapping[str, Any]) -> bool:
    return settings.get("ggn.cotangent_reuse") == "recompute_output_cotangent"


def _ggn_loss_hessian_product(
    execution: runtime_values.StandardExecution,
    output: TensorTree,
    output_jvp: TensorTree,
) -> TensorTree:
    if (
        execution.compiled_ggn_loss_product is not None
        and execution.candidate.settings.get("compile.boundary")
        == "ggn_loss_hessian_product"
    ):
        return execution.compiled_ggn_loss_product(output, output_jvp)

    return _ggn_loss_hessian_product_by_path(execution, output, output_jvp)


def _ggn_loss_hessian_product_by_path(
    execution: runtime_values.StandardExecution,
    output: TensorTree,
    output_jvp: TensorTree,
) -> TensorTree:
    path = execution.candidate.settings.get("ggn.loss_hessian_path")

    if path == "closed_form_softmax_ce_kl":
        return _ggn_closed_form_softmax_ce_kl_product(
            execution.candidate.settings,
            output,
            output_jvp,
        )

    if path == "autodiff_loss_hvp":
        return _ggn_autodiff_loss_hvp(execution, output, output_jvp, validate=True)

    message = f"ggn.loss_hessian_path is unsupported: {path}"
    raise MaterializationError(message)


def _ggn_loss_hessian_product_unchecked(
    execution: runtime_values.StandardExecution,
    output: TensorTree,
    output_jvp: TensorTree,
) -> TensorTree:
    path = execution.candidate.settings.get("ggn.loss_hessian_path")

    if path == "closed_form_softmax_ce_kl":
        return _ggn_closed_form_softmax_ce_kl_product_unchecked(
            execution.candidate.settings,
            output,
            output_jvp,
        )

    if path == "autodiff_loss_hvp":
        return _ggn_autodiff_loss_hvp(execution, output, output_jvp, validate=False)

    message = f"ggn.loss_hessian_path is unsupported: {path}"
    raise MaterializationError(message)


def _ggn_autodiff_loss_hvp(
    execution: runtime_values.StandardExecution,
    output: TensorTree,
    output_jvp: TensorTree,
    *,
    validate: bool,
) -> TensorTree:
    flat_output = runtime_values.flatten_vector(output)
    flat_output_jvp = runtime_values.flatten_vector(output_jvp)
    loss_hessian = runtime_values.batch_tensor(execution.batch, "loss_hessian")

    if validate:
        runtime_values.require_loss_hessian_shape(loss_hessian, flat_output_jvp.numel())
        runtime_values.require_finite_tensor(loss_hessian, "loss_hessian")
        runtime_values.require_finite_tensor(flat_output_jvp, "GGN output JVP")

    def output_loss(flat_value: torch.Tensor) -> torch.Tensor:
        hessian_value = layout.matmul_runtime(
            execution.candidate.settings,
            loss_hessian,
            flat_value,
        )

        return 0.5 * runtime.dot_runtime(
            execution.candidate.settings,
            flat_value,
            hessian_value,
        )

    product = torch.func.jvp(
        torch.func.grad(output_loss),
        (flat_output,),
        (flat_output_jvp,),
    )[1]

    return runtime_values.wrap_flat_vector(
        output,
        product,
    )


def _ggn_closed_form_softmax_ce_kl_product(
    settings: Mapping[str, Any],
    output: TensorTree,
    output_jvp: TensorTree,
) -> TensorTree:
    if not isinstance(output, torch.Tensor) or not isinstance(output_jvp, torch.Tensor):
        message = "closed-form CE/KL GGN requires tensor logits and tensor JVP"
        raise MaterializationError(message)

    if output.shape != output_jvp.shape:
        message = "closed-form CE/KL logits and JVP shapes must match"
        raise MaterializationError(message)

    if output.ndim == 0:
        message = "closed-form CE/KL logits must have a class dimension"
        raise MaterializationError(message)

    runtime_values.require_finite_tensor(output, "GGN logits")
    runtime_values.require_finite_tensor(output_jvp, "GGN output JVP")
    token_block_size = runtime_values.token_block_size(settings)

    if token_block_size is not None:
        return _softmax_ce_kl_product_token_loop(
            settings,
            output,
            output_jvp,
            token_block_size,
        )

    return _softmax_ce_kl_product_without_token_loop(settings, output, output_jvp)


def _ggn_closed_form_softmax_ce_kl_product_unchecked(
    settings: Mapping[str, Any],
    output: TensorTree,
    output_jvp: TensorTree,
) -> TensorTree:
    if not isinstance(output, torch.Tensor) or not isinstance(output_jvp, torch.Tensor):
        message = "closed-form CE/KL GGN requires tensor logits and tensor JVP"
        raise MaterializationError(message)

    if output.shape != output_jvp.shape:
        message = "closed-form CE/KL logits and JVP shapes must match"
        raise MaterializationError(message)

    if output.ndim == 0:
        message = "closed-form CE/KL logits must have a class dimension"
        raise MaterializationError(message)

    token_block_size = runtime_values.token_block_size(settings)

    if token_block_size is not None:
        return _softmax_ce_kl_product_token_loop(
            settings,
            output,
            output_jvp,
            token_block_size,
        )

    return _softmax_ce_kl_product_without_token_loop(settings, output, output_jvp)


def _softmax_ce_kl_product_token_loop(
    settings: Mapping[str, Any],
    logits: torch.Tensor,
    tangent: torch.Tensor,
    block_size: int,
) -> torch.Tensor:
    class_count = logits.shape[-1]
    flat_logits = logits.reshape(-1, class_count)
    flat_tangent = tangent.reshape(-1, class_count)
    outputs = []

    for start in range(0, flat_logits.shape[0], block_size):
        stop = min(start + block_size, flat_logits.shape[0])
        logits_block = flat_logits[start:stop]
        tangent_block = flat_tangent[start:stop]
        outputs.append(
            _softmax_ce_kl_product_without_token_loop(
                settings,
                logits_block,
                tangent_block,
            )
        )

    return torch.cat(tuple(outputs), dim=0).reshape_as(logits)


def _softmax_ce_kl_product_without_token_loop(
    settings: Mapping[str, Any],
    logits: torch.Tensor,
    tangent: torch.Tensor,
) -> torch.Tensor:
    kernel = settings.get("ggn.loss_hessian_kernel")

    if kernel == "dense_global":
        return _softmax_ce_kl_product_dense_global(settings, logits, tangent)

    if kernel == "streaming_global":
        return _softmax_ce_kl_product_streaming_global(settings, logits, tangent)

    if kernel == "two_pass_chunked_global":
        block_size = runtime_values.class_block_size_with_exact_global_normalization(
            settings
        )

        return _softmax_ce_kl_product_two_pass_chunked_global(
            settings,
            logits,
            tangent,
            block_size,
        )

    message = f"ggn.loss_hessian_kernel is unsupported: {kernel}"
    raise MaterializationError(message)


def _softmax_ce_kl_product_dense_global(
    settings: Mapping[str, Any],
    logits: torch.Tensor,
    tangent: torch.Tensor,
) -> torch.Tensor:
    probabilities = torch.softmax(logits, dim=-1)
    runtime_probabilities = runtime.accumulation_tensor(probabilities, settings)
    runtime_tangent = runtime.accumulation_tensor(tangent, settings)
    mean_tangent = (runtime_probabilities * runtime_tangent).sum(
        dim=-1,
        keepdim=True,
    )

    return runtime_probabilities * (runtime_tangent - mean_tangent)


def _softmax_ce_kl_product_streaming_global(
    settings: Mapping[str, Any],
    logits: torch.Tensor,
    tangent: torch.Tensor,
) -> torch.Tensor:
    max_logits = logits.max(dim=-1, keepdim=True).values
    unnormalized = runtime.accumulation_tensor((logits - max_logits).exp(), settings)
    runtime_tangent = runtime.accumulation_tensor(tangent, settings)
    denominator = unnormalized.sum(dim=-1, keepdim=True)
    probabilities = unnormalized / denominator
    mean_tangent = (probabilities * runtime_tangent).sum(dim=-1, keepdim=True)

    return probabilities * (runtime_tangent - mean_tangent)


def _softmax_ce_kl_product_two_pass_chunked_global(
    settings: Mapping[str, Any],
    logits: torch.Tensor,
    tangent: torch.Tensor,
    block_size: int,
) -> torch.Tensor:
    chunks = torch.split(logits, block_size, dim=-1)
    tangent_chunks = torch.split(tangent, block_size, dim=-1)
    max_logits = chunks[0].max(dim=-1, keepdim=True).values

    for chunk in chunks[1:]:
        max_logits = torch.maximum(max_logits, chunk.max(dim=-1, keepdim=True).values)

    accumulator_template = runtime.accumulation_tensor(max_logits, settings)
    denominator = torch.zeros_like(accumulator_template)
    weighted_tangent_sum = torch.zeros_like(accumulator_template)

    for chunk, tangent_chunk in zip(chunks, tangent_chunks, strict=True):
        unnormalized = runtime.accumulation_tensor((chunk - max_logits).exp(), settings)
        runtime_tangent_chunk = runtime.accumulation_tensor(tangent_chunk, settings)
        denominator = denominator + unnormalized.sum(dim=-1, keepdim=True)
        weighted_tangent_sum = weighted_tangent_sum + (
            unnormalized * runtime_tangent_chunk
        ).sum(dim=-1, keepdim=True)

    mean_tangent = weighted_tangent_sum / denominator
    outputs = []

    for chunk, tangent_chunk in zip(chunks, tangent_chunks, strict=True):
        unnormalized = runtime.accumulation_tensor((chunk - max_logits).exp(), settings)
        runtime_tangent_chunk = runtime.accumulation_tensor(tangent_chunk, settings)
        probabilities = unnormalized / denominator
        outputs.append(probabilities * (runtime_tangent_chunk - mean_tangent))

    return torch.cat(tuple(outputs), dim=-1)


def _run_ggnvp_vjp(
    execution: runtime_values.StandardExecution,
    tensor_function: Callable[[ParameterTree], TensorTree],
    output_cotangent: TensorTree,
) -> TensorTree:
    block_size = runtime_values.output_cotangent_block_size(
        execution.candidate.settings
    )

    if block_size is not None:

        def run_block(block: TensorTree) -> TensorTree:
            return _run_ggnvp_vjp_single_cotangent(execution, tensor_function, block)

        return _run_output_cotangent_blocks(
            execution,
            output_cotangent,
            block_size,
            run_block,
        )

    return _run_ggnvp_vjp_single_cotangent(
        execution,
        tensor_function,
        output_cotangent,
    )


def _run_ggnvp_vjp_single_cotangent(
    execution: runtime_values.StandardExecution,
    tensor_function: Callable[[ParameterTree], TensorTree],
    output_cotangent: TensorTree,
) -> TensorTree:
    if (
        execution.compiled_ggn_vjp is not None
        and execution.candidate.settings.get("compile.boundary") == "ggn_vjp"
    ):
        return execution.compiled_ggn_vjp(output_cotangent)

    return _run_ggnvp_vjp_by_path(execution, tensor_function, output_cotangent)


def _run_output_cotangent_blocks(
    execution: runtime_values.StandardExecution,
    output_cotangent: TensorTree,
    block_size: int,
    run_block: Callable[[TensorTree], TensorTree],
) -> TensorTree:
    blocks = runtime_values.cotangent_blocks(output_cotangent, block_size)
    result = run_block(blocks[0])

    for block in blocks[1:]:
        result = runtime.tree_add_runtime(
            execution.candidate.settings,
            result,
            run_block(block),
        )

    return result


def _run_ggnvp_vjp_by_path(
    execution: runtime_values.StandardExecution,
    tensor_function: Callable[[ParameterTree], TensorTree],
    output_cotangent: TensorTree,
) -> TensorTree:
    path = execution.candidate.settings.get("ggn.vjp_path")

    if path == "torch_func_vjp":
        pullback = derivatives.vjp_pullback(tensor_function, execution.params)
        (result,) = pullback(output_cotangent)

        return result

    if path == "autograd_grad_outputs":
        return derivatives.autograd_grad_outputs_vjp(
            tensor_function,
            execution.params,
            output_cotangent,
        )

    message = f"ggn.vjp_path is unsupported: {path}"
    raise MaterializationError(message)


def _jacobian_transpose_product(
    jacobian: torch.Tensor,
    cotangent: torch.Tensor,
    settings: Mapping[str, Any],
    parameter_surface: ParameterSurface | None,
) -> torch.Tensor:
    ranges = runtime_values.parameter_column_ranges(
        jacobian.shape[1], settings, parameter_surface
    )

    if ranges is None:
        return layout.matmul_runtime(settings, jacobian.T, cotangent)

    if jacobian.ndim != runtime_values.MATRIX_DIMS:
        message = "GGN jacobian must be two-dimensional"
        raise MaterializationError(message)

    if cotangent.ndim != 1:
        message = "GGN cotangent must be one-dimensional"
        raise MaterializationError(message)

    if jacobian.shape[0] != cotangent.numel():
        message = "GGN jacobian rows must match cotangent width"
        raise MaterializationError(message)

    chunks = []

    for start, stop in ranges:
        chunks.append(
            layout.matmul_runtime(settings, jacobian[:, start:stop].T, cotangent)
        )

    return torch.cat(tuple(chunks))


def ggn_spec_runtime_path(candidate: Candidate) -> str | None:
    """Return the ggn spec runtime path.

    Returns:
        The ggn spec runtime path.

    Raises:
        MaterializationError: If the declared inputs are invalid.
    """
    settings = candidate.settings
    jvp_key = runtime_values.SPEC_PATH_KEYS["ggnvp"]

    if (
        settings.get("ggn.loss_hessian_kernel") == "dense_global"
        and jvp_key not in settings
    ):
        return runtime_values.GGN_DENSE_PATH

    if jvp_key not in settings:
        return None

    value = settings[jvp_key]
    path_map = runtime_values.SPEC_PATH_TO_RUNTIME["ggnvp"]
    path = path_map.get(value)

    if path is None:
        message = f"{jvp_key} value is not lowered by standard runtime: {value}"
        raise MaterializationError(message)

    return path


def _runtime_intermediate_residency_tree(
    tree: TensorTree,
    settings: Mapping[str, Any],
    intermediate_transform: runtime_values.IntermediateTransform | None = None,
) -> TensorTree:
    residency = settings.get("memory.intermediate_residency")
    result = tree

    if residency is not None:
        result = runtime.tree_residency(
            tree, residency, "memory.intermediate_residency"
        )

    if intermediate_transform is None:
        return result

    return intermediate_transform(result)


def require_ggn_vjp_path_settings(
    operator: OperatorSpec,
    path: str,
    settings: Mapping[str, Any],
) -> None:
    """Validate ggn vjp path settings.

    Raises:
        MaterializationError: If the declared inputs are invalid.
    """
    if operator.kind != "ggnvp":
        return

    value = settings.get("ggn.vjp_path")

    if path == runtime_values.GGN_DENSE_PATH:
        if value is not None:
            message = "ggn.vjp_path is not used with dense_global"
            raise MaterializationError(message)

        return

    if value is None:
        message = "ggn.vjp_path is required for JVP-Hessian-VJP rows"
        raise MaterializationError(message)

    if value not in {
        "torch_func_vjp",
        "autograd_grad_outputs",
    }:
        message = f"ggn.vjp_path is unsupported: {value}"
        raise MaterializationError(message)


def require_ggn_loss_hessian_settings(
    operator: OperatorSpec,
    settings: Mapping[str, Any],
) -> None:
    """Validate ggn loss hessian settings.

    Raises:
        MaterializationError: If the declared inputs are invalid.
    """
    if operator.kind != "ggnvp":
        return

    path = settings.get("ggn.loss_hessian_path")
    kernel = settings.get("ggn.loss_hessian_kernel")
    chunk_key = "chunk.class_block_size_with_exact_global_normalization"

    if path is None and kernel is None:
        if chunk_key in settings:
            message = f"{chunk_key} requires two_pass_chunked_global"
            raise MaterializationError(message)

        return

    if path is None or kernel is None:
        message = "GGN loss-Hessian rows require path and kernel settings"
        raise MaterializationError(message)

    if path == "closed_form_softmax_ce_kl":
        if kernel not in {
            "dense_global",
            "streaming_global",
            "two_pass_chunked_global",
        }:
            message = f"ggn.loss_hessian_kernel is unsupported: {kernel}"
            raise MaterializationError(message)

        if kernel == "two_pass_chunked_global":
            runtime_values.class_block_size_with_exact_global_normalization(settings)
        elif chunk_key in settings:
            message = f"{chunk_key} requires two_pass_chunked_global"
            raise MaterializationError(message)

        return

    if path != "autodiff_loss_hvp":
        message = f"ggn.loss_hessian_path is not lowered by standard runtime: {path}"
        raise MaterializationError(message)

    if kernel != "dense_global":
        message = (
            f"ggn.loss_hessian_kernel is not lowered by standard runtime: {kernel}"
        )
        raise MaterializationError(message)

    if chunk_key in settings:
        message = f"{chunk_key} requires two_pass_chunked_global"
        raise MaterializationError(message)


def require_ggn_batch_size_settings(
    operator: OperatorSpec,
    path: str,
    settings: Mapping[str, Any],
) -> None:
    """Validate ggn batch size settings.

    Raises:
        MaterializationError: If the declared inputs are invalid.
    """
    batch_size = _ggn_batch_size(settings)

    if batch_size is None:
        return

    if operator.kind != "ggnvp":
        message = "batch.ggn_batch_size applies only to GGNVP rows"
        raise MaterializationError(message)

    if path != runtime_values.GGN_DENSE_PATH:
        message = "batch.ggn_batch_size requires dense GGN"
        raise MaterializationError(message)


def _ggn_batch_size(settings: Mapping[str, Any]) -> int | None:
    key = "batch.ggn_batch_size"
    return runtime_values.optional_positive_int_setting(
        settings,
        key,
        f"{key} must be a positive integer",
    )


def require_ggn_reuse_settings(
    operator: OperatorSpec,
    path: str,
    settings: Mapping[str, Any],
) -> None:
    """Validate ggn reuse settings.

    Raises:
        MaterializationError: If the declared inputs are invalid.
    """
    jvp_reuse = settings.get("ggn.jvp_reuse")
    cotangent_reuse = settings.get("ggn.cotangent_reuse")

    if operator.kind != "ggnvp":
        if jvp_reuse is not None:
            message = "ggn.jvp_reuse applies only to GGNVP rows"
            raise MaterializationError(message)

        if cotangent_reuse is not None:
            message = "ggn.cotangent_reuse applies only to GGNVP rows"
            raise MaterializationError(message)

        return

    if path == runtime_values.GGN_DENSE_PATH:
        if jvp_reuse is not None:
            message = "ggn.jvp_reuse is not used with dense_global"
            raise MaterializationError(message)

        if cotangent_reuse is not None:
            message = "ggn.cotangent_reuse is not used with dense_global"
            raise MaterializationError(message)

        return

    if jvp_reuse not in {None, "reuse_jvp", "recompute_jvp"}:
        message = f"ggn.jvp_reuse is unsupported: {jvp_reuse}"
        raise MaterializationError(message)

    if cotangent_reuse not in {
        None,
        "reuse_output_cotangent",
        "recompute_output_cotangent",
    }:
        message = f"ggn.cotangent_reuse is unsupported: {cotangent_reuse}"
        raise MaterializationError(message)


def ggn_anchor_settings(
    operator: OperatorSpec,
    path: str,
) -> dict[str, Any]:
    """Return the ggn anchor settings.

    Returns:
        The ggn anchor settings.
    """
    if operator.kind != "ggnvp":
        return {}

    if path == runtime_values.GGN_DENSE_PATH:
        return {
            "ggn.loss_hessian_path": "autodiff_loss_hvp",
            "ggn.loss_hessian_kernel": "dense_global",
        }

    if path == runtime_values.GGN_JVP_HESSIAN_VJP_PATH:
        return {
            "ggn.loss_hessian_path": "autodiff_loss_hvp",
            "ggn.loss_hessian_kernel": "dense_global",
            "ggn.vjp_path": "torch_func_vjp",
        }

    return {}
