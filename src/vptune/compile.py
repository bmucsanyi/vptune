"""Compile lowerings for the standard runtime.

torch.compile setup, backend and mode parsing, compile boundaries,
compiled autograd, CUDA graph rows, cache warming, and compiled
callable preparation for every operator family.
"""

import dataclasses
import importlib
from collections.abc import Callable, Mapping
from typing import Any

import torch

from vptune import (
    derivatives,
    fisher,
    ggn,
    metrics,
    runtime,
    runtime_values,
)
from vptune.data import (
    Batch,
    Candidate,
    CandidateOperation,
    OperatorSpec,
    ParameterTree,
)
from vptune.errors import (
    CompileSetupError,
)
from vptune.tensor_tree import (
    TensorTree,
)


def _call_compiled_body(callback: Callable[..., Any], *args: Any) -> Any:
    with (
        runtime_values.deferred_runtime_finite_checks(),
        runtime_values.disabled_backend_settings(),
    ):
        return callback(*args)


def _call_compiled_operation(
    settings: Mapping[str, Any],
    callback: Callable[..., Any],
    *args: Any,
) -> Any:
    return runtime.run_with_backend_settings(
        settings,
        lambda: _call_compiled_body(callback, *args),
    )


def prepare_compile_boundary_execution(
    execution: runtime_values.StandardExecution,
) -> runtime_values.StandardExecution:
    """Prepare compile boundary execution.

    Returns:
        The compile boundary execution result.
    """
    settings = execution.candidate.settings
    boundary = settings.get("compile.boundary")

    if settings.get("compile.enabled") != "true" or not isinstance(boundary, str):
        return execution

    return _prepare_enabled_compile_boundary_execution(execution, settings, boundary)


def _prepare_enabled_compile_boundary_execution(
    execution: runtime_values.StandardExecution,
    settings: Mapping[str, Any],
    boundary: str,
) -> runtime_values.StandardExecution:
    special_builder = {
        "model_forward": lambda: _prepare_model_forward_compile_boundary(
            execution,
            settings,
        ),
        "loss_closure": lambda: _prepare_loss_closure_compile_boundary(
            execution,
            settings,
        ),
    }.get(boundary)

    if special_builder is not None:
        return special_builder()

    if boundary == "bound_operator_vector_step":
        return _prepare_bound_operator_vector_step_compile_boundary(
            execution,
            settings,
        )

    inner_builders = {
        ("gradient", "gradient_closure"): lambda: derivatives.run_gradient_by_path(
            execution
        ),
        ("jvp", "jvp_closure"): lambda: derivatives.run_jvp_by_path(execution),
        ("vjp", "vjp_closure"): lambda: derivatives.run_vjp_by_path(execution),
        ("hvp", "hvp_single_vector"): lambda: derivatives.run_hvp_single_vector(
            execution
        ),
        ("hvp", "hvp_batched_vectors"): lambda: derivatives.run_hvp_by_path(execution),
        ("ggnvp", "ggn_full_product"): lambda: ggn.run_ggnvp_by_path(execution),
        ("metric", "metric_multiply"): lambda: metrics.metric_multiply_by_path(
            execution.operator,
            execution.batch,
            execution.vector,
            execution.path,
            settings,
        ),
        ("inverse_metric", "inverse_metric_solve"): lambda: (
            metrics.run_inverse_metric_by_mode(execution)
        ),
        (
            "sqrt_metric",
            "metric_sqrt_multiply",
        ): lambda: metrics.metric_square_root_apply(
            execution,
            inverse=False,
            adjoint=False,
        ),
        (
            "inverse_sqrt_metric",
            "metric_sqrt_multiply",
        ): lambda: metrics.metric_square_root_apply(
            execution,
            inverse=True,
            adjoint=False,
        ),
        ("metric_inner", "metric_inner_reduce"): lambda: metrics.run_metric_inner(
            execution
        ),
        (
            "inverse_metric_inner",
            "inverse_metric_inner_reduce",
        ): lambda: metrics.run_inverse_metric_inner(execution),
    }
    builder = inner_builders.get((execution.operator.kind, boundary))

    if builder is not None:
        return _prepare_inner_compile_boundary(execution, settings, builder)

    score_builder = fisher.score_matrix_compile_boundary_builder(execution, boundary)

    if score_builder is not None:
        return fisher.prepare_score_matrix_compile_boundary(
            execution,
            settings,
            score_builder,
        )

    ggn_execution = ggn.prepare_ggn_compile_boundary_execution(
        execution,
        settings,
        boundary,
    )

    if ggn_execution is not None:
        return ggn_execution

    return execution


def require_compiled_execution(
    execution: runtime_values.StandardExecution,
    settings: Mapping[str, Any],
) -> None:
    """Validate that the execution carries the compiled inner operation."""
    _require_compile_boundary(execution.operator, settings)

    if compile_bool(settings, "compile.compiled_autograd"):
        _require_compiled_autograd_operator(execution.operator)


def _prepare_inner_compile_boundary(
    execution: runtime_values.StandardExecution,
    settings: Mapping[str, Any],
    builder: CandidateOperation,
) -> runtime_values.StandardExecution:
    require_compiled_execution(execution, settings)
    compiled_inner = compiled_operation(settings, builder)

    return dataclasses.replace(
        execution,
        compiled_inner=compiled_inner,
    )


def _prepare_bound_operator_vector_step_compile_boundary(
    execution: runtime_values.StandardExecution,
    settings: Mapping[str, Any],
) -> runtime_values.StandardExecution:
    require_compiled_execution(execution, settings)
    step_execution = dataclasses.replace(execution, compiled_vector_step=None)

    def vector_step(vector: TensorTree) -> TensorTree:
        vector_execution = runtime.execution_with_vector(step_execution, vector)

        return runtime.run_standard_operation(vector_execution)

    compiled_vector_step = compiled_bound_vector_step(
        settings,
        vector_step,
        execution.vector,
    )

    return dataclasses.replace(
        execution,
        compiled_vector_step=compiled_vector_step,
    )


def _prepare_model_forward_compile_boundary(
    execution: runtime_values.StandardExecution,
    settings: Mapping[str, Any],
) -> runtime_values.StandardExecution:
    require_compiled_execution(execution, settings)

    if execution.module is None or execution.module_call is None:
        message = "compile.boundary=model_forward requires module_call"
        raise CompileSetupError(message)

    module = execution.module
    module_call = execution.module_call

    def model_forward(batch: Batch) -> object:
        return runtime_values.invoke_stateful_module(
            module,
            module_call,
            batch,
        )

    compiled_model_forward = _compiled_model_forward(settings, model_forward)

    if settings.get("compile.cache_state") == "warm_cache":
        _call_compiled_model_forward(
            execution,
            compiled_model_forward,
            execution.params,
        )

    return dataclasses.replace(
        execution,
        compiled_model_forward=compiled_model_forward,
    )


def _prepare_loss_closure_compile_boundary(
    execution: runtime_values.StandardExecution,
    settings: Mapping[str, Any],
) -> runtime_values.StandardExecution:
    require_compiled_execution(execution, settings)
    scalar_function = derivatives.hvp_scalar_function(execution)
    compiled_scalar_function = _compiled_scalar_function(
        settings,
        scalar_function,
        execution.params,
    )

    return dataclasses.replace(
        execution,
        compiled_scalar_function=compiled_scalar_function,
    )


def compile_operation(
    operator: OperatorSpec,
    settings: Mapping[str, Any],
    operation: CandidateOperation,
) -> CandidateOperation:
    """Return the operation compiled per the declared boundary.

    Returns:
        The operation compiled per the declared boundary.

    Raises:
        CompileSetupError: If the declared inputs are invalid.
    """
    enabled = settings.get("compile.enabled")

    if enabled is None or enabled == "false":
        return operation

    if enabled != "true":
        message = f"compile.enabled is unsupported: {enabled}"
        raise CompileSetupError(message)

    _require_compile_boundary(operator, settings)

    compiled_autograd = compile_bool(settings, "compile.compiled_autograd")
    if compiled_autograd:
        _require_compiled_autograd_operator(operator)

    if _compile_boundary_runs_inside_operator(operator.kind, settings):
        return operation

    return compiled_operation(settings, operation)


def _compile_boundary_runs_inside_operator(
    operator_kind: str,
    settings: Mapping[str, Any],
) -> bool:
    if settings.get("compile.boundary") in {
        "model_forward",
        "bound_operator_vector_step",
    }:
        return True

    if operator_kind in runtime_values.SCORE_MATRIX_COMPILE_ROWS:
        return (
            settings.get("compile.boundary")
            == runtime_values.SCORE_MATRIX_COMPILE_ROWS[operator_kind].boundary
        )

    return (operator_kind, settings.get("compile.boundary")) in {
        ("gradient", "loss_closure"),
        ("gradient", "gradient_closure"),
        ("jvp", "jvp_closure"),
        ("vjp", "vjp_closure"),
        ("hvp", "loss_closure"),
        ("hvp", "hvp_single_vector"),
        ("hvp", "hvp_batched_vectors"),
        ("ggnvp", "ggn_full_product"),
        ("ggnvp", "ggn_jvp"),
        ("ggnvp", "ggn_loss_hessian_product"),
        ("ggnvp", "ggn_vjp"),
        ("metric", "metric_multiply"),
        ("sqrt_metric", "metric_sqrt_multiply"),
        ("inverse_sqrt_metric", "metric_sqrt_multiply"),
        ("metric_inner", "metric_inner_reduce"),
        ("inverse_metric", "inverse_metric_solve"),
        ("inverse_metric_inner", "inverse_metric_inner_reduce"),
        ("composition", "composition_child"),
    }


def compiled_operation(
    settings: Mapping[str, Any],
    operation: CandidateOperation,
) -> CandidateOperation:
    """Compile a candidate operation per its declared compile settings.

    Applies the declared backend settings and warms the compile cache
    according to the declared cache state.

    Returns:
        The compiled candidate operation.
    """
    compiled = compiled_callable(
        settings,
        operation,
        use_backend_settings=True,
    )
    warm_compiled_cache(settings, compiled)

    return compiled


def compiled_bound_vector_step(
    settings: Mapping[str, Any],
    operation: Callable[[TensorTree], TensorTree],
    warm_vector: TensorTree,
) -> Callable[[TensorTree], TensorTree]:
    """Return the compiled bound vector step.

    Returns:
        The compiled bound vector step.
    """
    compiled = compiled_callable(
        settings,
        operation,
        use_backend_settings=True,
    )
    warm_compiled_cache(settings, compiled, warm_vector)

    return compiled


def validate_compile_cache_state(settings: Mapping[str, Any]) -> None:
    """Validate the declared compile cache state.

    Raises:
        CompileSetupError: If the declared inputs are invalid.
    """
    if settings.get("compile.cache_state") in {"cold_compile", "warm_cache"}:
        return

    message = "compile.cache_state must be cold_compile or warm_cache"
    raise CompileSetupError(message)


def _compiled_model_forward(
    settings: Mapping[str, Any],
    operation: Callable[[Batch], object],
) -> Callable[[Batch], object]:
    compiled_function = compiled_callable(
        settings,
        operation,
        use_backend_settings=False,
    )
    validate_compile_cache_state(settings)

    return compiled_function


def _compiled_scalar_function(
    settings: Mapping[str, Any],
    operation: Callable[[ParameterTree], torch.Tensor],
    warm_params: ParameterTree,
) -> Callable[[ParameterTree], torch.Tensor]:
    compiled_function = compiled_callable(
        settings,
        operation,
        use_backend_settings=False,
    )
    warm_compiled_cache(settings, compiled_function, warm_params)

    return compiled_function


def compiled_callable(
    settings: Mapping[str, Any],
    operation: Callable[..., Any],
    *,
    use_backend_settings: bool,
) -> Callable[..., Any]:
    """Return the compiled callable for declared compile settings.

    Returns:
        The compiled callable for declared compile settings.
    """
    compiled_autograd = compile_bool(settings, "compile.compiled_autograd")
    compiled = _compile_with_runtime_settings(
        settings,
        operation,
        compiled_autograd=compiled_autograd,
    )

    def compiled_function(*args: Any) -> Any:
        if compiled_autograd:
            with compiled_autograd_patch():
                return _call_compiled_callable(
                    settings,
                    compiled,
                    args,
                    use_backend_settings=use_backend_settings,
                )

        return _call_compiled_callable(
            settings,
            compiled,
            args,
            use_backend_settings=use_backend_settings,
        )

    return compiled_function


def _compile_with_runtime_settings(
    settings: Mapping[str, Any],
    operation: Callable[..., Any],
    *,
    compiled_autograd: bool,
) -> Callable[..., Any]:
    if compiled_autograd:
        with compiled_autograd_patch():
            return _torch_compile_with_runtime_settings(settings, operation)

    return _torch_compile_with_runtime_settings(settings, operation)


def _torch_compile_with_runtime_settings(
    settings: Mapping[str, Any],
    operation: Callable[..., Any],
) -> Callable[..., Any]:
    return torch.compile(
        operation,
        backend=compile_backend(settings),
        mode=compile_mode(settings),
        fullgraph=compile_bool(settings, "compile.fullgraph"),
        dynamic=compile_optional_bool(settings, "compile.dynamic"),
        options=compile_options(settings),
    )


def _call_compiled_callable(
    settings: Mapping[str, Any],
    compiled: Callable[..., Any],
    args: tuple[Any, ...],
    *,
    use_backend_settings: bool,
) -> Any:
    if use_backend_settings:
        return _call_compiled_operation(settings, compiled, *args)

    return runtime_values.call_with_deferred_finite_checks(compiled, *args)


def warm_compiled_cache(
    settings: Mapping[str, Any],
    compiled: Callable[..., Any],
    *warm_args: Any,
    error_message: str | None = None,
) -> None:
    """Warm the compile cache per the declared cache state.

    Raises:
        CompileSetupError: If the declared inputs are invalid.
    """
    cache_state = settings.get("compile.cache_state")
    validate_compile_cache_state(settings)

    if cache_state != "warm_cache":
        return

    if error_message is not None and any(arg is None for arg in warm_args):
        raise CompileSetupError(error_message)

    compiled(*warm_args)


def _require_compile_boundary(
    operator: OperatorSpec,
    settings: Mapping[str, Any],
) -> None:
    boundary = settings.get("compile.boundary")

    if not isinstance(boundary, str):
        message = "compile.boundary is required"
        raise CompileSetupError(message)

    if _compile_boundary_supported(operator.kind, boundary, settings):
        return

    message = f"compile.boundary={boundary} is not lowered for {operator.kind}"
    raise CompileSetupError(message)


def _compile_boundary_supported(
    operator_kind: str,
    boundary: str,
    settings: Mapping[str, Any],
) -> bool:
    direct = _direct_compile_boundary_supported(operator_kind, boundary, settings)

    if direct is not None:
        return direct

    if operator_kind == "hvp":
        return _hvp_compile_boundary_supported(boundary, settings)

    if operator_kind == "ggnvp":
        return ggn.ggn_compile_boundary_supported(boundary, settings)

    if operator_kind in {
        "fisher_vp",
        "sampled_fisher_vp",
        "empirical_fisher_vp",
        "per_example_gradient",
    }:
        return fisher.score_matrix_compile_boundary_supported(
            operator_kind,
            boundary,
            settings,
        )

    boundaries = {
        "gradient": "gradient_closure",
        "jvp": "jvp_closure",
        "vjp": "vjp_closure",
        "metric": "metric_multiply",
        "sqrt_metric": "metric_sqrt_multiply",
        "inverse_sqrt_metric": "metric_sqrt_multiply",
        "metric_inner": "metric_inner_reduce",
        "inverse_metric": "inverse_metric_solve",
        "inverse_metric_inner": "inverse_metric_inner_reduce",
        "composition": "composition_child",
    }

    return boundaries.get(operator_kind) == boundary


def _direct_compile_boundary_supported(
    operator_kind: str,
    boundary: str,
    settings: Mapping[str, Any],
) -> bool | None:
    if boundary == "whole_operator":
        return True

    if boundary == "model_forward":
        return settings.get("call.path") == "stateful_module"

    if boundary == "loss_closure":
        return operator_kind in {"gradient", "hvp"}

    if boundary == "bound_operator_vector_step":
        return operator_kind in runtime_values.BOUND_OPERATOR_VECTOR_STEP_FAMILIES

    return None


def _hvp_compile_boundary_supported(
    boundary: str,
    settings: Mapping[str, Any],
) -> bool:
    vectorized = settings.get("vectorization.mode") in {"single_loop", "vmap"}

    if boundary == "hvp_single_vector":
        return not vectorized

    if boundary == "hvp_batched_vectors":
        return vectorized

    return False


def compiled_autograd_patch() -> Any:
    """Return the compiled-autograd config patch context.

    Returns:
        The compiled-autograd config patch context.
    """
    config = importlib.import_module("torch._dynamo.config")

    return config.patch({"compiled_autograd": True})


def _require_compiled_autograd_operator(operator: OperatorSpec) -> None:
    if operator.kind in {
        "gradient",
        "vjp",
        "hvp",
        "ggnvp",
        "fisher_vp",
        "sampled_fisher_vp",
        "empirical_fisher_vp",
    }:
        return

    message = (
        "compile.compiled_autograd=true requires a backward or higher-order "
        f"operator, got {operator.kind}"
    )
    raise CompileSetupError(message)


def compile_backend(settings: Mapping[str, Any]) -> str:
    """Return the validated compile backend id from settings.

    Returns:
        The validated compile backend id from settings.

    Raises:
        CompileSetupError: If the declared inputs are invalid.
    """
    value = settings.get("compile.backend")

    if not isinstance(value, str):
        message = "compile.backend is required"
        raise CompileSetupError(message)

    if value == "inductor":
        return value

    if value == "registered_backend":
        message = "compile.backend requires a concrete PyTorch compiler backend id"
        raise CompileSetupError(message)

    if _is_registered_compile_backend(value):
        return value

    message = f"compile.backend is not registered with PyTorch: {value}"
    raise CompileSetupError(message)


def _is_registered_compile_backend(value: str) -> bool:
    try:
        backends = torch.compiler.list_backends()
    except AttributeError as error:
        message = "torch.compiler.list_backends is required"
        raise CompileSetupError(message) from error

    return value in set(backends)


def compile_mode(settings: Mapping[str, Any]) -> str | None:
    """Return the validated compile mode from settings.

    Returns:
        The validated compile mode from settings.

    Raises:
        CompileSetupError: If the declared inputs are invalid.
    """
    value = settings.get("compile.mode")

    if value is None:
        return None

    if value in {"default", "max-autotune"}:
        return value

    message = f"compile.mode is unsupported: {value}"
    raise CompileSetupError(message)


def compile_bool(settings: Mapping[str, Any], key: str) -> bool:
    """Return the declared boolean compile setting.

    Returns:
        The declared boolean compile setting.

    Raises:
        CompileSetupError: If the declared inputs are invalid.
    """
    value = settings.get(key)

    if value == "true":
        return True

    if value == "false":
        return False

    message = f"{key} must be true or false"
    raise CompileSetupError(message)


def compile_optional_bool(settings: Mapping[str, Any], key: str) -> bool | None:
    """Return the declared optional boolean compile setting.

    Returns:
        The declared optional boolean compile setting.

    Raises:
        CompileSetupError: If the declared inputs are invalid.
    """
    value = settings.get(key)

    if value is None:
        return None

    if value == "true":
        return True

    if value == "false":
        return False

    message = f"{key} must be None, true, or false"
    raise CompileSetupError(message)


def compile_options(settings: Mapping[str, Any]) -> dict[str, Any] | None:
    """Return the validated compile options mapping.

    Returns:
        The validated compile options mapping.
    """
    options = {}

    if compile_bool(settings, "compile.options.epilogue_fusion"):
        options["epilogue_fusion"] = True

    if compile_bool(settings, "compile.options.shape_padding"):
        options["shape_padding"] = True

    if compile_bool(settings, "compile.cuda_graphs"):
        options["triton.cudagraphs"] = True

    if not options:
        return None

    return options


def candidate_without_compile_settings(candidate: Candidate) -> Candidate:
    """Return the candidate without compile settings.

    Returns:
        The candidate without compile settings.
    """
    return dataclasses.replace(
        candidate,
        settings={
            key: value
            for key, value in candidate.settings.items()
            if key not in runtime_values.COMPILE_SETTING_KEYS
        },
    )


def _call_compiled_model_forward(
    execution: runtime_values.StandardExecution,
    compiled_model_forward: Callable[[Batch], object],
    active_params: ParameterTree,
) -> object:
    if execution.module is None:
        message = "compile.boundary=model_forward requires module"
        raise CompileSetupError(message)

    settings = execution.candidate.settings
    model_params = runtime.stateful_model_tree(active_params, settings)
    model_buffers = runtime.stateful_model_tree(execution.buffers, settings)
    model_batch = runtime.stateful_model_batch(execution.batch, settings)
    slots = runtime_values.replace_module_state(
        execution.module, model_params, model_buffers
    )

    try:
        return compiled_model_forward(model_batch)
    finally:
        runtime_values.restore_module_state(slots)
