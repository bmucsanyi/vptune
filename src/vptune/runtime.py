"""Runtime builders for package-owned operator anchors."""

import dataclasses
import importlib
import inspect
import math
from collections.abc import Callable, Mapping, Sequence
from itertools import starmap
from typing import Any

import torch
from torch.utils.checkpoint import checkpoint, noop_context_fn

from vptune import (
    derivatives,
    fisher,
    ggn,
    layout,
    metrics,
    runtime_values,
    vectorization,
)
from vptune.admission import (
    FUNCTIONAL_CALL_FIELDS,
    TORCH_FUNC_FIELDS,
    admit_checkpoint,
)
from vptune.anchors import (
    finite_difference_jvp,
    vjp_dot_identity_error,
)
from vptune.checks import (
    tree_error_measurements,
    validate_thresholds,
)
from vptune.data import (
    PACKAGE_VERSION,
    Batch,
    BufferTree,
    CallableMaterializer,
    CallableOperationFactory,
    CallableReferenceCheck,
    Candidate,
    CandidateAdmitter,
    CandidateOperation,
    FullSizeRecord,
    FunctionObjective,
    Materializer,
    ModuleCallSpec,
    ObjectiveContext,
    OperationFactory,
    OperatorSpec,
    ParameterSurface,
    ParameterTree,
    ReferenceCheck,
    ReferenceResult,
    RuntimeConfig,
    RuntimeOperationFactory,
    RuntimeReferenceCheck,
    ScalarObjective,
)
from vptune.errors import (
    AdmissionError,
    CompileSetupError,
    MaterializationError,
    ReferenceFailedError,
)
from vptune.identities import stable_hash, to_json_value
from vptune.tensor_tree import (
    TensorTree,
    tree_add_foreach,
    tree_dot,
    tree_dot_foreach,
    tree_map,
    tree_map2,
    tree_mul_foreach,
    tree_signature,
)

MMapResidency = Callable[[torch.Tensor, str], torch.Tensor]


def checkpoint_operation(
    candidate: Candidate,
    function: Callable[..., TensorTree],
    args: Sequence[Any],
    *,
    policy_key: str,
    activation_pack_hooks: runtime_values.ActivationPackHooks | None = None,
    activation_unpack_hooks: runtime_values.ActivationUnpackHooks | None = None,
    checkpoint_contexts: runtime_values.CheckpointContextFns | None = None,
) -> CandidateOperation:
    """Return direct or checkpointed execution for an adapter operation.

    Raises:
        AdmissionError: If the candidate has missing or rejected checkpoint fields.
    """
    setting = _checkpoint_setting(candidate, policy_key)
    offload = _activation_offload(candidate)

    if setting == "none":
        return _with_activation_offload(
            candidate,
            runtime_values.direct_operation(function, args),
            offload,
            activation_pack_hooks,
            activation_unpack_hooks,
        )

    if setting not in runtime_values.ACTIVE_CHECKPOINT_SETTINGS:
        message = f"checkpoint setting is unsupported: {setting}"
        raise AdmissionError(message)

    admit_checkpoint(candidate.settings)
    context_fn = _checkpoint_context_fn(candidate, checkpoint_contexts)

    def operation() -> TensorTree:
        return checkpoint(
            function,
            *args,
            use_reentrant=False,
            preserve_rng_state=_checkpoint_bool(
                candidate,
                "checkpoint.preserve_rng_state",
            ),
            determinism_check=candidate.settings["checkpoint.determinism_check"],
            context_fn=context_fn,
            early_stop=_checkpoint_bool(candidate, "checkpoint.early_stop"),
        )

    return _with_activation_offload(
        candidate,
        operation,
        offload,
        activation_pack_hooks,
        activation_unpack_hooks,
    )


def _checkpoint_setting(candidate: Candidate, policy_key: str) -> str:
    if policy_key not in candidate.settings:
        message = f"checkpoint policy key is missing: {policy_key}"
        raise AdmissionError(message)

    setting = candidate.settings[policy_key]

    if not isinstance(setting, str):
        message = f"checkpoint setting must be a string: {policy_key}"
        raise AdmissionError(message)

    return setting


def _with_activation_offload(
    candidate: Candidate,
    operation: CandidateOperation,
    offload: str,
    activation_pack_hooks: runtime_values.ActivationPackHooks | None,
    activation_unpack_hooks: runtime_values.ActivationUnpackHooks | None,
) -> CandidateOperation:
    if offload == "none":
        return operation

    pack_hook, unpack_hook = _saved_tensor_hooks(
        candidate,
        offload,
        activation_pack_hooks,
        activation_unpack_hooks,
    )

    def wrapped() -> TensorTree:
        with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
            return operation()

    return wrapped


def _activation_offload(candidate: Candidate) -> str:
    value = candidate.settings.get("activation.offload")

    if value not in {"none", "saved_tensor_hooks_cpu", "custom_saved_tensor_hooks"}:
        message = "activation.offload is invalid"
        raise AdmissionError(message)

    return value


def _saved_tensor_hooks(
    candidate: Candidate,
    offload: str,
    activation_pack_hooks: runtime_values.ActivationPackHooks | None,
    activation_unpack_hooks: runtime_values.ActivationUnpackHooks | None,
) -> tuple[Callable[[torch.Tensor], Any], Callable[[Any], torch.Tensor]]:
    if offload == "saved_tensor_hooks_cpu":
        return _cpu_pack_hook, _cpu_unpack_hook

    pack_hook_id = candidate.settings.get("activation.pack_hook")
    unpack_hook_id = candidate.settings.get("activation.unpack_hook")

    if not isinstance(pack_hook_id, str) or not isinstance(unpack_hook_id, str):
        message = "custom_saved_tensor_hooks requires activation pack and unpack hooks"
        raise AdmissionError(message)

    if activation_pack_hooks is None or pack_hook_id not in activation_pack_hooks:
        message = f"activation pack hook is not registered: {pack_hook_id}"
        raise AdmissionError(message)

    if activation_unpack_hooks is None or unpack_hook_id not in activation_unpack_hooks:
        message = f"activation unpack hook is not registered: {unpack_hook_id}"
        raise AdmissionError(message)

    return activation_pack_hooks[pack_hook_id], activation_unpack_hooks[unpack_hook_id]


def _cpu_pack_hook(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.device]:
    return tensor.detach().cpu(), tensor.device


def _cpu_unpack_hook(packed: tuple[torch.Tensor, torch.device]) -> torch.Tensor:
    tensor, device = packed

    return tensor.to(device)


def _checkpoint_context_fn(
    candidate: Candidate,
    checkpoint_contexts: runtime_values.CheckpointContextFns | None,
) -> Callable[[], Any]:
    context_fn = candidate.settings["checkpoint.context_fn"]

    if context_fn == "none":
        return noop_context_fn

    context_id = candidate.settings.get("checkpoint.context_fn_callable")

    if not isinstance(context_id, str):
        message = "checkpoint.context_fn=declared_context_pair requires context id"
        raise AdmissionError(message)

    if checkpoint_contexts is None or context_id not in checkpoint_contexts:
        message = f"checkpoint context is not registered: {context_id}"
        raise AdmissionError(message)

    return checkpoint_contexts[context_id]


def _checkpoint_bool(candidate: Candidate, key: str) -> bool:
    value = candidate.settings[key]

    if value == "true":
        return True

    if value == "false":
        return False

    message = f"{key} must be false or true"
    raise AdmissionError(message)


def require_thresholds_for_measurements(
    measurements: Mapping[str, Any],
    thresholds: Mapping[str, float],
) -> None:
    """Validate that every measurement has a declared threshold.

    Raises:
        ReferenceFailedError: If the declared inputs are invalid.
    """
    exact_required = ("psd_violation", "damping_min", "condition_number_max")
    missing = tuple(
        key
        for key in measurements
        if key.endswith(("_diff", "_residual")) or key in exact_required
        if key not in thresholds
    )

    if missing:
        message = f"reference thresholds are missing measurements: {missing}"
        raise ReferenceFailedError(message)


def semantic_measurements(
    operator: OperatorSpec,
    batch: Batch,
    vector: TensorTree,
    output: TensorTree,
) -> dict[str, float]:
    """Return semantic measurements for an operator output.

    Returns:
        The semantic measurements for an operator output.
    """
    if operator.kind == "ggnvp":
        loss_hessian = runtime_values.batch_tensor(batch, "loss_hessian")

        return {
            "symmetry_max_abs_diff": runtime_values.matrix_symmetry_error(loss_hessian),
            "psd_violation": runtime_values.matrix_psd_violation(loss_hessian),
        }

    if operator.kind == "metric":
        if metrics.metric_representation_kind(operator) == "matrix_free":
            return {}

        matrix = metrics.metric_dense_matrix(operator, batch, vector)

        return {
            "symmetry_max_abs_diff": runtime_values.matrix_symmetry_error(matrix),
            "psd_violation": runtime_values.matrix_psd_violation(matrix),
        }

    if operator.kind == "inverse_metric":
        if metrics.metric_representation_kind(operator) == "matrix_free":
            return {}

        matrix = metrics.metric_dense_matrix(operator, batch, vector)
        inverse_matrix = metrics.inverse_metric_matrix(operator, matrix, batch, vector)
        vector_tensor = runtime_values.flatten_vector(vector)
        output_tensor = runtime_values.flatten_vector(output)
        measurements = {
            "symmetry_max_abs_diff": runtime_values.matrix_symmetry_error(
                inverse_matrix
            ),
            "psd_violation": runtime_values.matrix_psd_violation(inverse_matrix),
            "inverse_residual": runtime_values.inverse_residual(
                inverse_matrix,
                output_tensor,
                vector_tensor,
            ),
            "condition_number_max": runtime_values.matrix_condition_number(
                inverse_matrix
            ),
        }
        damping = metrics.inverse_metric_min_damping(operator)

        if damping > 0.0:
            measurements["damping_min"] = damping

        return measurements

    return {}


def _matrix_free_inverse_reference_measurements(
    operator: OperatorSpec,
    candidate: Candidate,
    batch: Batch,
    vector: TensorTree,
    output: TensorTree,
) -> dict[str, float]:
    if operator.kind != "inverse_metric":
        return {}

    if metrics.metric_representation_kind(operator) != "matrix_free":
        return {}

    damping = metrics.inverse_metric_damping_payload(operator)
    flat_output = runtime_values.flatten_vector(output)
    applied = metrics.metric_apply_flat(
        operator,
        batch,
        output,
        flat_output,
        damping,
        runtime_values.METRIC_STREAMING_PATH,
        candidate.settings,
    )
    residual = (applied - runtime_values.flatten_vector(vector)).norm()
    denominator = runtime_values.flatten_vector(vector).norm()

    if math.isclose(float(denominator.item()), 0.0, rel_tol=0.0, abs_tol=0.0):
        inverse_residual = float(residual.item())
    else:
        inverse_residual = float((residual / denominator).item())

    return {
        "inverse_residual": inverse_residual,
        "damping_min": metrics.minimum_inverse_metric_damping(damping),
    }


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
        scalar = runtime_values.scalar_objective(operator, scalar_objectives)
        context = ObjectiveContext(
            family=operator.family,
            candidate_id=candidate.candidate_id,
            settings=dict(candidate.settings),
        )

        def scalar_function(active_params: ParameterTree) -> torch.Tensor:
            return scalar(active_params, buffers, batch, context)

        finite_difference = finite_difference_jvp(scalar_function, params, vector)
        directional = layout.layout_aware_tree_dot(
            candidate.settings,
            candidate_output,
            vector,
        )
        errors = tree_error_measurements(directional, finite_difference)

        return {
            "directional_abs_diff": errors["max_abs_diff"],
            "directional_rel_diff": errors["max_rel_diff"],
        }

    if operator.kind == "jvp":
        function = runtime_values.function_objective(operator, function_objectives)
        context = ObjectiveContext(
            family=operator.family,
            candidate_id=candidate.candidate_id,
            settings=dict(candidate.settings),
        )

        def tensor_function(active_params: ParameterTree) -> TensorTree:
            output = function(active_params, buffers, batch, context)

            return runtime_values.checked_function_output(
                candidate.settings,
                output,
                "function objective output",
            )

        finite_difference = finite_difference_jvp(tensor_function, params, vector)
        errors = layout.layout_aware_tree_error_measurements(
            candidate,
            candidate_output,
            finite_difference,
        )

        return {
            "directional_abs_diff": errors["max_abs_diff"],
            "directional_rel_diff": errors["max_rel_diff"],
        }

    if operator.kind == "vjp":
        function = runtime_values.function_objective(operator, function_objectives)
        tangent = runtime_values.batch_tree(batch, "tangent_vector")
        runtime_values.require_min_probe_norm(tangent, "tangent_vector")
        runtime_values.require_min_probe_norm(vector, "cotangent_vector")
        context = ObjectiveContext(
            family=operator.family,
            candidate_id=candidate.candidate_id,
            settings=dict(candidate.settings),
        )

        def tensor_function(active_params: ParameterTree) -> TensorTree:
            output = function(active_params, buffers, batch, context)

            return runtime_values.checked_function_output(
                candidate.settings,
                output,
                "function objective output",
            )

        return {
            "inner_abs_diff": float(
                vjp_dot_identity_error(tensor_function, params, tangent, vector).item()
            )
        }

    return {}


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
    return run_with_backend_settings(
        settings,
        lambda: _call_compiled_body(callback, *args),
    )


def execution_with_vector(
    execution: runtime_values.StandardExecution,
    vector: TensorTree,
    **changes: Any,
) -> runtime_values.StandardExecution:
    """Return the execution with a replaced vector.

    Returns:
        The execution with a replaced vector.
    """
    return dataclasses.replace(
        execution,
        vector=vector,
        flat_parameter_vector=None,
        flat_parameter_vector_batch=None,
        **changes,
    )


def standard_operation_factory(
    operator: OperatorSpec,
    *,
    params: ParameterTree,
    buffers: BufferTree,
    parameter_surface: ParameterSurface | None = None,
    scalar_objectives: Mapping[str, ScalarObjective] | None = None,
    function_objectives: Mapping[str, FunctionObjective] | None = None,
    module: torch.nn.Module | None = None,
    module_call: ModuleCallSpec | None = None,
    teacher_objective: FunctionObjective | None = None,
    batch_layout: Callable[[Candidate, Batch], Batch] | None = None,
    lm_head_chunker: Callable[[Candidate, Batch], Batch] | None = None,
    fusion_rewriter: (
        Callable[[torch.nn.Module, Candidate], torch.nn.Module] | None
    ) = None,
    mmap_residency: Callable[[torch.Tensor, str], torch.Tensor] | None = None,
    manual_recompute: (
        Callable[
            [Candidate, CandidateOperation, tuple[torch.Tensor, ...]],
            CandidateOperation,
        ]
        | None
    ) = None,
    activation_pack_hooks: runtime_values.ActivationPackHooks | None = None,
    activation_unpack_hooks: runtime_values.ActivationUnpackHooks | None = None,
    checkpoint_contexts: runtime_values.CheckpointContextFns | None = None,
    intermediate_transform: runtime_values.IntermediateTransform | None = None,
) -> OperationFactory:
    """Return an operation factory for package-owned standard operators."""
    scalar_map = {} if scalar_objectives is None else dict(scalar_objectives)
    function_map = {} if function_objectives is None else dict(function_objectives)
    pack_hook_map = {} if activation_pack_hooks is None else dict(activation_pack_hooks)
    unpack_hook_map = (
        {} if activation_unpack_hooks is None else dict(activation_unpack_hooks)
    )
    checkpoint_context_map = (
        {} if checkpoint_contexts is None else dict(checkpoint_contexts)
    )
    mmap_residency_callback = mmap_residency

    def factory(
        candidate: Candidate,
        batch: Batch,
        vector: TensorTree,
    ) -> CandidateOperation:
        runtime_values.require_candidate_family(operator, candidate)
        require_supported_standard_settings(
            operator,
            candidate,
            parameter_surface,
            mmap_residency_callback,
            fusion_rewriter,
            batch_layout,
            lm_head_chunker,
            pack_hook_map,
            unpack_hook_map,
            checkpoint_context_map,
        )
        _require_recomputed_teacher_objective(candidate.settings, teacher_objective)
        runtime_module = runtime_values.runtime_fusion_module(
            module, candidate, fusion_rewriter
        )
        path = runtime_path(operator, candidate)
        transformed_batch = _runtime_declared_batch_transforms(
            batch,
            candidate,
            batch_layout,
            lm_head_chunker,
        )
        _require_batch_inputs(operator, candidate, transformed_batch, phase="operation")
        runtime_params = _runtime_params(params, candidate.settings, parameter_surface)
        runtime_buffers = _runtime_buffers(buffers, candidate.settings)
        prepared_batch = runtime_batch(
            transformed_batch,
            candidate.settings,
            move_input_residency=_move_input_residency_outside_measured_call(
                candidate.settings
            ),
            mmap_residency=mmap_residency_callback,
        )
        prepared_vector = runtime_vector(
            vector,
            candidate.settings,
            runtime_params,
            parameter_surface,
            mmap_residency=mmap_residency_callback,
        )
        context = ObjectiveContext(
            family=operator.family,
            candidate_id=candidate.candidate_id,
            settings=dict(candidate.settings),
        )
        execution = runtime_values.StandardExecution(
            operator=operator,
            candidate=candidate,
            path=path,
            batch=prepared_batch,
            vector=prepared_vector,
            params=runtime_params,
            buffers=runtime_buffers,
            parameter_surface=parameter_surface,
            context=context,
            scalar_objectives=scalar_map,
            function_objectives=function_map,
            module=runtime_module,
            module_call=module_call,
            teacher_objective=teacher_objective,
            batch_layout=batch_layout,
            lm_head_chunker=lm_head_chunker,
            mmap_residency=mmap_residency_callback,
            manual_recompute=manual_recompute,
            activation_pack_hooks=pack_hook_map,
            activation_unpack_hooks=unpack_hook_map,
            checkpoint_contexts=checkpoint_context_map,
            intermediate_transform=intermediate_transform,
        )
        _require_stateful_module_execution(execution)
        execution = _loss_scaled_execution(execution)
        _require_finite_execution_inputs(execution)
        execution = _prepare_standard_execution(execution)
        execution = _prepare_flat_vector_execution(execution)
        execution = _prepare_compile_boundary_execution(execution)
        output_buffer = _standard_output_buffer(execution)

        def operation() -> TensorTree:
            operation_execution = _execution_with_inside_input_residency(execution)

            return run_with_backend_settings(
                candidate.settings,
                lambda: runtime_values.run_with_call_grad_mode(
                    candidate.settings,
                    lambda: runtime_values.runtime_output_to_buffer(
                        runtime_output(
                            _run_with_buffer_mutation_check(
                                operation_execution,
                                lambda: _run_standard_operation(operation_execution),
                            ),
                            candidate.settings,
                            parameter_surface,
                        ),
                        output_buffer,
                    ),
                ),
            )

        activated_operation = _activation_operation(execution, operation)

        return compile_operation(
            operator,
            candidate.settings,
            activated_operation,
        )

    return factory


def _prepare_standard_execution(
    execution: runtime_values.StandardExecution,
) -> runtime_values.StandardExecution:
    if execution.operator.kind == "gradient":
        return derivatives.prepare_gradient_execution(execution)

    if execution.operator.kind == "jvp":
        return derivatives.prepare_jvp_execution(execution)

    if execution.operator.kind == "vjp":
        return derivatives.prepare_vjp_execution(execution)

    if execution.operator.kind == "hvp":
        return derivatives.prepare_hvp_execution(execution)

    return execution


def _prepare_compile_boundary_execution(
    execution: runtime_values.StandardExecution,
) -> runtime_values.StandardExecution:
    settings = execution.candidate.settings
    boundary = settings.get("compile.boundary")

    if settings.get("compile.enabled") != "true" or not isinstance(boundary, str):
        return execution

    return _prepare_enabled_compile_boundary_execution(execution, settings, boundary)


def _prepare_flat_vector_execution(
    execution: runtime_values.StandardExecution,
) -> runtime_values.StandardExecution:
    if (
        execution.operator.kind
        not in runtime_values.PARAMETER_VECTOR_CACHE_OPERATOR_KINDS
    ):
        return execution

    if _uses_rectangular_square_root_input(execution):
        if "vectorization.mode" in execution.candidate.settings:
            message = (
                "rectangular closed-form square-root does not support vectorization"
            )
            raise MaterializationError(message)

        return execution

    mode = execution.candidate.settings.get("vectorization.mode")

    if mode == "vmap":
        return dataclasses.replace(
            execution,
            flat_parameter_vector_batch=vectorization.build_flat_vector_batch(
                execution
            ),
        )

    if mode in {"single_loop", "manual_batch"}:
        return execution

    return dataclasses.replace(
        execution,
        flat_parameter_vector=_build_parameter_order_vector(execution),
    )


def _uses_rectangular_square_root_input(
    execution: runtime_values.StandardExecution,
) -> bool:
    if execution.operator.kind != "sqrt_metric":
        return False

    if execution.path != runtime_values.SQRT_METRIC_CLOSED_FORM_PATH:
        return False

    return metrics.metric_representation_kind(execution.operator) in {
        "low_rank_factors",
        "ggn_derived_factors",
    }


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

    if _compile_bool(settings, "compile.compiled_autograd"):
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
        vector_execution = execution_with_vector(step_execution, vector)

        return _run_standard_operation(vector_execution)

    compiled_vector_step = _compiled_bound_vector_step(
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


def _activation_operation(
    execution: runtime_values.StandardExecution,
    operation: CandidateOperation,
) -> CandidateOperation:
    settings = execution.candidate.settings

    if not runtime_values.has_activation_settings(settings):
        return operation

    if settings.get("activation.recompute") == "manual_recompute":
        if execution.manual_recompute is None:
            message = "manual_recompute requires a declared recompute callback"
            raise MaterializationError(message)

        recomputed_operation = execution.manual_recompute(
            execution.candidate,
            operation,
            _activation_tensor_args(execution),
        )

        return _with_activation_offload(
            execution.candidate,
            recomputed_operation,
            _activation_offload(execution.candidate),
            execution.activation_pack_hooks,
            execution.activation_unpack_hooks,
        )

    def function(*_: torch.Tensor) -> TensorTree:
        return operation()

    try:
        return checkpoint_operation(
            execution.candidate,
            function,
            _activation_tensor_args(execution),
            policy_key="activation.recompute",
            activation_pack_hooks=execution.activation_pack_hooks,
            activation_unpack_hooks=execution.activation_unpack_hooks,
            checkpoint_contexts=execution.checkpoint_contexts,
        )
    except AdmissionError as error:
        raise MaterializationError(str(error)) from error


def _activation_tensor_args(
    execution: runtime_values.StandardExecution,
) -> tuple[torch.Tensor, ...]:
    return (
        *runtime_values.tensor_args(execution.params),
        *runtime_values.tensor_args(execution.buffers),
        *runtime_values.tensor_args(execution.batch),
        *runtime_values.tensor_args(execution.vector),
    )


def _require_finite_execution_inputs(
    execution: runtime_values.StandardExecution,
) -> None:
    for name, value in (
        ("parameters", execution.params),
        ("buffers", execution.buffers),
        ("batch", execution.batch),
        ("vector", execution.vector),
    ):
        runtime_values.require_finite_nested_tensors(value, name)


def _execution_with_inside_input_residency(
    execution: runtime_values.StandardExecution,
) -> runtime_values.StandardExecution:
    if _move_input_residency_outside_measured_call(execution.candidate.settings):
        return execution

    return dataclasses.replace(
        execution,
        batch=runtime_batch_input_residency(
            execution.batch,
            execution.candidate.settings,
        ),
    )


def _move_input_residency_outside_measured_call(
    settings: Mapping[str, Any],
) -> bool:
    value = settings.get("input.host_to_device")

    if value is None or value == "outside_measured_call":
        return True

    if value == "inside_measured_call":
        return False

    message = f"input.host_to_device is unsupported: {value}"
    raise MaterializationError(message)


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

    compiled_autograd = _compile_bool(settings, "compile.compiled_autograd")
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


def _compiled_bound_vector_step(
    settings: Mapping[str, Any],
    operation: Callable[[TensorTree], TensorTree],
    warm_vector: TensorTree,
) -> Callable[[TensorTree], TensorTree]:
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
    compiled_autograd = _compile_bool(settings, "compile.compiled_autograd")
    compiled = _compile_with_runtime_settings(
        settings,
        operation,
        compiled_autograd=compiled_autograd,
    )

    def compiled_function(*args: Any) -> Any:
        if compiled_autograd:
            with _compiled_autograd_patch():
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
        with _compiled_autograd_patch():
            return _torch_compile_with_runtime_settings(settings, operation)

    return _torch_compile_with_runtime_settings(settings, operation)


def _torch_compile_with_runtime_settings(
    settings: Mapping[str, Any],
    operation: Callable[..., Any],
) -> Callable[..., Any]:
    return torch.compile(
        operation,
        backend=_compile_backend(settings),
        mode=_compile_mode(settings),
        fullgraph=_compile_bool(settings, "compile.fullgraph"),
        dynamic=_compile_optional_bool(settings, "compile.dynamic"),
        options=_compile_options(settings),
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


def _compiled_autograd_patch() -> Any:
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


def _compile_backend(settings: Mapping[str, Any]) -> str:
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


def _compile_mode(settings: Mapping[str, Any]) -> str | None:
    value = settings.get("compile.mode")

    if value is None:
        return None

    if value in {"default", "max-autotune"}:
        return value

    message = f"compile.mode is unsupported: {value}"
    raise CompileSetupError(message)


def _compile_bool(settings: Mapping[str, Any], key: str) -> bool:
    value = settings.get(key)

    if value == "true":
        return True

    if value == "false":
        return False

    message = f"{key} must be true or false"
    raise CompileSetupError(message)


def _compile_optional_bool(settings: Mapping[str, Any], key: str) -> bool | None:
    value = settings.get(key)

    if value is None:
        return None

    if value == "true":
        return True

    if value == "false":
        return False

    message = f"{key} must be None, true, or false"
    raise CompileSetupError(message)


def _compile_options(settings: Mapping[str, Any]) -> dict[str, Any] | None:
    options = {}

    if _compile_bool(settings, "compile.options.epilogue_fusion"):
        options["epilogue_fusion"] = True

    if _compile_bool(settings, "compile.options.shape_padding"):
        options["shape_padding"] = True

    if _compile_bool(settings, "compile.cuda_graphs"):
        options["triton.cudagraphs"] = True

    if not options:
        return None

    return options


def standard_reference_check(
    operator: OperatorSpec,
    *,
    params: ParameterTree,
    buffers: BufferTree,
    thresholds: Mapping[str, float],
    parameter_surface: ParameterSurface | None = None,
    numeric_bound_fields: Mapping[str, Any] | None = None,
    scalar_objectives: Mapping[str, ScalarObjective] | None = None,
    function_objectives: Mapping[str, FunctionObjective] | None = None,
    module: torch.nn.Module | None = None,
    module_call: ModuleCallSpec | None = None,
    teacher_objective: FunctionObjective | None = None,
    batch_layout: Callable[[Candidate, Batch], Batch] | None = None,
    lm_head_chunker: Callable[[Candidate, Batch], Batch] | None = None,
    fusion_rewriter: (
        Callable[[torch.nn.Module, Candidate], torch.nn.Module] | None
    ) = None,
    mmap_residency: Callable[[torch.Tensor, str], torch.Tensor] | None = None,
    manual_recompute: (
        Callable[
            [Candidate, CandidateOperation, tuple[torch.Tensor, ...]],
            CandidateOperation,
        ]
        | None
    ) = None,
    activation_pack_hooks: runtime_values.ActivationPackHooks | None = None,
    activation_unpack_hooks: runtime_values.ActivationUnpackHooks | None = None,
    checkpoint_contexts: runtime_values.CheckpointContextFns | None = None,
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
    bound_fields = {} if numeric_bound_fields is None else dict(numeric_bound_fields)
    candidate_factory = standard_operation_factory(
        operator,
        params=params,
        buffers=buffers,
        parameter_surface=parameter_surface,
        scalar_objectives=scalar_map,
        function_objectives=function_map,
        module=module,
        module_call=module_call,
        teacher_objective=teacher_objective,
        batch_layout=batch_layout,
        lm_head_chunker=lm_head_chunker,
        fusion_rewriter=fusion_rewriter,
        mmap_residency=mmap_residency,
        manual_recompute=manual_recompute,
        activation_pack_hooks=activation_pack_hooks,
        activation_unpack_hooks=activation_unpack_hooks,
        checkpoint_contexts=checkpoint_contexts,
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
                parameter_surface=parameter_surface,
            )
        except ReferenceFailedError:
            raise
        except RuntimeError as error:
            raise ReferenceFailedError(str(error)) from error

        effective_thresholds = reference_thresholds_for_operator(operator, thresholds)
        require_thresholds_for_measurements(measurements, effective_thresholds)
        validate_thresholds(measurements, effective_thresholds)
        runtime_values.apply_numeric_error_bound(
            measurements,
            effective_thresholds,
            candidate.settings,
            bound_fields,
            anchor_output,
        )

        return ReferenceResult(
            "standard_anchor",
            effective_thresholds,
            measurements,
        )

    return check


def reference_thresholds_for_operator(
    operator: OperatorSpec,
    thresholds: Mapping[str, float],
) -> dict[str, float]:
    """Return the reference thresholds for an operator.

    Returns:
        The reference thresholds for an operator.
    """
    effective_thresholds = dict(thresholds)

    if operator.kind in {"inverse_metric", "inverse_metric_inner"}:
        tolerance = metrics.inverse_metric_tolerance(operator)

        if tolerance is not None:
            effective_thresholds["inverse_residual"] = tolerance

    return effective_thresholds


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
    derivatives.require_vhp_reference_policy(candidate, batch, thresholds)
    candidate_output = candidate_factory(candidate, batch, vector)()

    if (
        operator.kind
        in {"metric", "inverse_metric", "sqrt_metric", "inverse_sqrt_metric"}
        and metrics.metric_representation_kind(operator) == "matrix_free"
    ):
        anchor_output = candidate_output
    elif operator.kind in {"metric", "inverse_metric"}:
        anchor_output = metrics.metric_reference_output(operator, batch, vector)
    else:
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

    declared = ggn.ggn_declared_batch_inputs(operator, candidate, declared, phase)
    path_inputs = _candidate_batch_inputs(operator, candidate)
    teacher_inputs = _teacher_output_batch_inputs(candidate)

    return tuple(dict.fromkeys((*declared, *path_inputs, *teacher_inputs)))


def _teacher_output_batch_inputs(candidate: Candidate) -> tuple[str, ...]:
    if "teacher_outputs" not in candidate.settings:
        return ()

    return ("teacher_outputs",)


def _candidate_batch_inputs(
    operator: OperatorSpec,
    candidate: Candidate,
) -> tuple[str, ...]:
    path = runtime_path(operator, candidate)

    if operator.kind == "fisher_vp":
        return fisher.fisher_batch_inputs(operator, path)

    if operator.kind == "sampled_fisher_vp":
        return fisher.sampled_fisher_batch_inputs(operator, path)

    if operator.kind == "empirical_fisher_vp":
        return fisher.empirical_fisher_batch_inputs(operator, path)

    return ()


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
    parameter_surface: ParameterSurface | None,
) -> dict[str, Any]:
    measurements = layout.layout_aware_tree_error_measurements(
        candidate,
        candidate_output,
        anchor_output,
    )
    ggn.augment_ggn_dense_cross_check(
        operator,
        candidate,
        batch,
        vector,
        candidate_output,
        candidate_factory,
        measurements,
    )
    measurements.update(
        semantic_measurements(operator, batch, vector, candidate_output)
    )
    measurements.update(
        _matrix_free_inverse_reference_measurements(
            operator,
            candidate,
            batch,
            vector,
            candidate_output,
        )
    )
    measurements.update(
        metrics.inverse_metric_inner_reference_measurements(
            operator,
            candidate,
            batch,
            vector,
            params,
        )
    )
    measurements.update(
        ggn.ggn_inner_product_measurements(
            operator,
            candidate,
            batch,
            vector,
            candidate_output,
            anchor_candidate,
            candidate_factory,
            parameter_surface,
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
        derivatives.hvp_finite_difference_measurements(
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
            parameter_surface=parameter_surface,
        )
    )

    return measurements


def _runtime_callable_identity(
    value: Callable[..., Any] | None,
    name: str,
) -> Any:
    if value is None:
        return None

    return _callable_identity(value, name)


def _runtime_callable_map_identity(
    values: Mapping[str, Callable[..., Any]] | None,
    name: str,
) -> tuple[dict[str, Any], ...]:
    if values is None:
        return ()

    return tuple(
        {
            "id": key,
            "identity": _callable_identity(callback, f"{name}.{key}"),
        }
        for key, callback in sorted(values.items())
    )


def _callable_identity(value: Callable[..., Any], name: str) -> Any:
    explicit_identity = _explicit_callable_identity(value)

    if explicit_identity is not None:
        return explicit_identity

    module = getattr(value, "__module__", None)
    qualname = getattr(value, "__qualname__", None)

    if not isinstance(module, str) or not isinstance(qualname, str):
        message = f"{name} must provide identity() or signature()"
        raise MaterializationError(message)

    try:
        source = inspect.getsource(value)
    except (OSError, TypeError) as error:
        message = f"{name} must provide identity() or signature()"
        raise MaterializationError(message) from error

    return {
        "kind": "python_callable",
        "module": module,
        "qualname": qualname,
        "source_hash": stable_hash({"source": source}),
        "defaults": _json_identity(getattr(value, "__defaults__", None), name),
        "kwdefaults": _json_identity(getattr(value, "__kwdefaults__", None), name),
        "closure": _callable_closure_identity(value, name),
    }


def _explicit_callable_identity(value: Callable[..., Any]) -> Any | None:
    identity = getattr(value, "identity", None)

    if callable(identity):
        return {
            "kind": "explicit_identity",
            "value": _json_identity(identity(), "callable.identity"),
        }

    signature = getattr(value, "signature", None)

    if callable(signature):
        return {
            "kind": "explicit_signature",
            "value": _json_identity(signature(), "callable.signature"),
        }

    return None


def _json_identity(value: Any, name: str) -> Any:
    try:
        return to_json_value(value)
    except TypeError as error:
        message = f"{name} must be JSON-compatible"
        raise MaterializationError(message) from error


def _callable_closure_identity(value: Callable[..., Any], name: str) -> tuple[Any, ...]:
    closure = getattr(value, "__closure__", None)

    if closure is None:
        return ()

    if closure:
        message = f"{name} closes over runtime state; provide identity() or signature()"
        raise MaterializationError(message)

    return ()


def _keyword_map(**kwargs: Any) -> dict[str, Any]:
    return kwargs


def standard_runtime_config(
    operator: OperatorSpec,
    *,
    params: ParameterTree,
    buffers: BufferTree,
    candidates: Sequence[Candidate],
    thresholds: Mapping[str, float],
    objective_signature: Mapping[str, Any],
    axis_registry: CandidateAdmitter | None,
    parameter_surface: ParameterSurface | None = None,
    numeric_bound_fields: Mapping[str, Any] | None = None,
    scalar_objectives: Mapping[str, ScalarObjective] | None = None,
    function_objectives: Mapping[str, FunctionObjective] | None = None,
    module: torch.nn.Module | None = None,
    module_call: ModuleCallSpec | None = None,
    teacher_objective: FunctionObjective | None = None,
    batch_layout: Callable[[Candidate, Batch], Batch] | None = None,
    lm_head_chunker: Callable[[Candidate, Batch], Batch] | None = None,
    fusion_rewriter: (
        Callable[[torch.nn.Module, Candidate], torch.nn.Module] | None
    ) = None,
    mmap_residency: Callable[[torch.Tensor, str], torch.Tensor] | None = None,
    manual_recompute: (
        Callable[
            [Candidate, CandidateOperation, tuple[torch.Tensor, ...]],
            CandidateOperation,
        ]
        | None
    ) = None,
    activation_pack_hooks: runtime_values.ActivationPackHooks | None = None,
    activation_unpack_hooks: runtime_values.ActivationUnpackHooks | None = None,
    checkpoint_contexts: runtime_values.CheckpointContextFns | None = None,
) -> RuntimeConfig:
    """Return runtime config for package-owned standard operators."""
    bound_fields = {} if numeric_bound_fields is None else dict(numeric_bound_fields)
    mmap_residency_callback = mmap_residency
    runtime_bindings = _keyword_map(
        params=params,
        buffers=buffers,
        parameter_surface=parameter_surface,
        scalar_objectives=scalar_objectives,
        function_objectives=function_objectives,
        module=module,
        module_call=module_call,
        teacher_objective=teacher_objective,
        batch_layout=batch_layout,
        lm_head_chunker=lm_head_chunker,
        fusion_rewriter=fusion_rewriter,
        mmap_residency=mmap_residency_callback,
        manual_recompute=manual_recompute,
        activation_pack_hooks=activation_pack_hooks,
        activation_unpack_hooks=activation_unpack_hooks,
        checkpoint_contexts=checkpoint_contexts,
    )
    operation_factory = standard_operation_factory(operator, **runtime_bindings)
    reference_check = standard_reference_check(
        operator,
        thresholds=thresholds,
        numeric_bound_fields=bound_fields,
        **runtime_bindings,
    )
    runtime_signature = {
        "runtime": "standard",
        "operator": operator.signature(),
        "params": tree_signature(params),
        "buffers": tree_signature(buffers),
        "parameter_surface": (
            None if parameter_surface is None else parameter_surface.signature()
        ),
        "thresholds": dict(thresholds),
        "numeric_bound_fields": bound_fields,
        "objective": dict(objective_signature),
        "module": module is not None,
        "module_call": None if module_call is None else module_call.signature(),
        "teacher_objective": _runtime_callable_identity(
            teacher_objective,
            "teacher_objective",
        ),
        "batch_layout": _runtime_callable_identity(batch_layout, "batch_layout"),
        "lm_head_chunker": _runtime_callable_identity(
            lm_head_chunker,
            "lm_head_chunker",
        ),
        "fusion_rewriter": _runtime_callable_identity(
            fusion_rewriter,
            "fusion_rewriter",
        ),
        "mmap_residency": _runtime_callable_identity(
            mmap_residency,
            "mmap_residency",
        ),
        "manual_recompute": _runtime_callable_identity(
            manual_recompute,
            "manual_recompute",
        ),
        "activation_pack_hooks": _runtime_callable_map_identity(
            activation_pack_hooks,
            "activation_pack_hooks",
        ),
        "activation_unpack_hooks": _runtime_callable_map_identity(
            activation_unpack_hooks,
            "activation_unpack_hooks",
        ),
        "checkpoint_contexts": _runtime_callable_map_identity(
            checkpoint_contexts,
            "checkpoint_contexts",
        ),
    }
    operation_factory = CallableOperationFactory(
        "vptune.standard_operation_factory",
        PACKAGE_VERSION,
        runtime_signature,
        {"callback": "vptune.runtime.standard_operation_factory"},
        operation_factory,
    )
    reference_check = CallableReferenceCheck(
        "vptune.standard_reference_check",
        PACKAGE_VERSION,
        runtime_signature,
        {"callback": "vptune.runtime.standard_reference_check"},
        reference_check,
    )
    materializer = standard_materializer(
        operation_factory,
        operator,
        mmap_residency_callback,
    )

    return RuntimeConfig(
        candidates=tuple(candidates),
        operation_factory=operation_factory,
        reference_check=reference_check,
        materializer=materializer,
        axis_registry=axis_registry,
        reference_check_name="standard_anchor",
        signature=runtime_signature,
    )


def _standard_changed_axes(
    settings: Mapping[str, Any],
    axis_registry: Any,
) -> tuple[str, ...]:
    axes = set()

    for key in settings:
        owner = axis_registry.owners.get(key)

        if owner is not None:
            axes.add(owner)
            continue

        optional_owners = axis_registry.optional_owners.get(key)

        if optional_owners is not None:
            axes.update(optional_owners)
            continue

        axes.add(key)

    return tuple(sorted(axes))


def _run_standard_operation(execution: runtime_values.StandardExecution) -> TensorTree:
    if execution.compiled_vector_step is not None:
        return execution.compiled_vector_step(execution.vector)

    require_loss_scaling_settings(execution.operator, execution.candidate.settings)
    runner = STANDARD_RUNNERS.get(execution.operator.kind)

    if runner is None:
        message = (
            "standard runtime does not support operator kind: "
            f"{execution.operator.kind}"
        )
        raise MaterializationError(message)

    execution = _execution_with_recomputed_teacher_outputs(execution)

    if _uses_microbatch_accumulation(execution):
        return _run_microbatch_accumulate(execution)

    result = runner(execution)
    result = loss_scaled_output_source(
        execution.operator,
        execution.candidate.settings,
        result,
    )

    return loss_unscaled_output(
        execution.operator,
        execution.candidate.settings,
        result,
    )


def _loss_scaled_execution(
    execution: runtime_values.StandardExecution,
) -> runtime_values.StandardExecution:
    scale = loss_scale(execution.candidate.settings)

    if scale is None:
        return execution

    if execution.operator.kind in {"gradient", "hvp"}:
        return dataclasses.replace(
            execution,
            scalar_objectives=_scaled_scalar_objectives(execution, scale),
        )

    if execution.operator.kind in {"jvp", "vjp"}:
        return dataclasses.replace(
            execution,
            function_objectives=_scaled_function_objectives(execution, scale),
        )

    if execution.operator.kind == "ggnvp":
        return dataclasses.replace(
            execution,
            batch=runtime_values.scaled_loss_hessian_batch(execution.batch, scale),
        )

    return execution


def _scaled_scalar_objectives(
    execution: runtime_values.StandardExecution,
    scale: float,
) -> Mapping[str, ScalarObjective]:
    objective_id = execution.operator.objective_id
    objective = runtime_values.scalar_objective(
        execution.operator, execution.scalar_objectives
    )
    objectives = dict(execution.scalar_objectives)

    def scaled(
        params: ParameterTree,
        buffers: BufferTree,
        batch: Batch,
        context: ObjectiveContext,
    ) -> torch.Tensor:
        return objective(params, buffers, batch, context) * scale

    objectives[objective_id] = scaled

    return objectives


def _scaled_function_objectives(
    execution: runtime_values.StandardExecution,
    scale: float,
) -> Mapping[str, FunctionObjective]:
    objective_id = execution.operator.objective_id
    objective = runtime_values.function_objective(
        execution.operator, execution.function_objectives
    )
    objectives = dict(execution.function_objectives)

    def scaled(
        params: ParameterTree,
        buffers: BufferTree,
        batch: Batch,
        context: ObjectiveContext,
    ) -> TensorTree:
        output = objective(params, buffers, batch, context)
        output = runtime_values.checked_function_output(
            execution.candidate.settings,
            output,
            "function objective output",
        )

        return tree_map(
            lambda tensor: tensor * scale,
            output,
        )

    objectives[objective_id] = scaled

    return objectives


def loss_scaled_output_source(
    operator: OperatorSpec,
    settings: Mapping[str, Any],
    result: TensorTree,
) -> TensorTree:
    """Return the loss-scaled output source for an execution.

    Returns:
        The loss-scaled output source for an execution.
    """
    scale = loss_scale(settings)

    if scale is None:
        return result

    if operator.kind in {"metric", "inverse_metric", "composition"}:
        return tree_scale_runtime(settings, result, scale)

    return result


def loss_unscaled_output(
    operator: OperatorSpec,
    settings: Mapping[str, Any],
    result: TensorTree,
) -> TensorTree:
    """Return the output unscaled by the declared loss-scale law.

    Returns:
        The output unscaled by the declared loss-scale law.
    """
    scale = loss_scale(settings)

    if scale is None:
        return result

    degree = _loss_unscale_degree(operator, settings)

    return tree_scale_runtime(settings, result, 1.0 / (scale**degree))


def require_loss_scaling_settings(
    operator: OperatorSpec,
    settings: Mapping[str, Any],
) -> None:
    """Validate the declared loss-scaling settings.

    Raises:
        MaterializationError: If the declared inputs are invalid.
    """
    mode = settings.get("numeric.loss_scaling")
    has_scale = "numeric.loss_scale" in settings
    has_degree = "numeric.loss_unscale_degree" in settings

    if mode is None:
        if has_scale or has_degree:
            message = "numeric.loss_scaling is required for loss-scale fields"
            raise MaterializationError(message)

        return

    if mode == "none":
        if has_scale or has_degree:
            message = "numeric.loss_scaling=none forbids loss-scale fields"
            raise MaterializationError(message)

        return

    if mode != "static_scale_with_exact_unscale":
        message = f"numeric.loss_scaling is unsupported: {mode}"
        raise MaterializationError(message)

    loss_scale(settings)
    _loss_unscale_degree(operator, settings)


def loss_scale(settings: Mapping[str, Any]) -> float | None:
    """Return the declared numeric loss scale.

    Returns:
        The declared numeric loss scale.

    Raises:
        MaterializationError: If the declared inputs are invalid.
    """
    mode = settings.get("numeric.loss_scaling")

    if mode is None or mode == "none":
        return None

    value = settings.get("numeric.loss_scale")

    if isinstance(value, bool) or not isinstance(value, int | float) or value <= 0.0:
        message = "numeric.loss_scale must be a positive float"
        raise MaterializationError(message)

    return float(value)


def _loss_unscale_degree(
    operator: OperatorSpec,
    settings: Mapping[str, Any],
) -> int:
    value = settings.get("numeric.loss_unscale_degree")

    if isinstance(value, bool) or not isinstance(value, int):
        message = "numeric.loss_unscale_degree must be an integer"
        raise MaterializationError(message)

    expected = _expected_loss_unscale_degree(operator)

    if value != expected:
        message = (
            "numeric.loss_unscale_degree does not match operator: "
            f"{value} != {expected}"
        )
        raise MaterializationError(message)

    return value


def _expected_loss_unscale_degree(operator: OperatorSpec) -> int:
    if operator.kind in {
        "fisher_vp",
        "sampled_fisher_vp",
        "empirical_fisher_vp",
    }:
        return 2

    return 1


def _uses_microbatch_accumulation(execution: runtime_values.StandardExecution) -> bool:
    return (
        execution.candidate.settings.get("schedule.gradient_accumulation")
        == "microbatch_accumulate"
    )


def _run_microbatch_accumulate(
    execution: runtime_values.StandardExecution,
) -> TensorTree:
    if execution.operator.aggregation != "sum":
        message = "microbatch_accumulate requires sum aggregation"
        raise MaterializationError(message)

    batch, batch_in_dims = vectorization.microbatch_in_dims(execution.batch)
    batch_size = runtime_values.per_example_batch_size(
        batch,
        batch_in_dims,
        "microbatch accumulation",
    )
    microbatch_size = _data_microbatch_size(execution.candidate.settings)
    accumulated = None

    for start in range(0, batch_size, microbatch_size):
        stop = min(start + microbatch_size, batch_size)
        subbatch = runtime_values.per_example_batch_slice(
            batch, batch_in_dims, start, stop
        )
        subexecution = dataclasses.replace(
            execution,
            batch=subbatch,
            candidate=dataclasses.replace(
                execution.candidate,
                settings=_single_step_microbatch_settings(execution.candidate.settings),
            ),
        )
        subresult = _run_standard_operation(subexecution)
        accumulated = (
            subresult
            if accumulated is None
            else tree_add_runtime(
                execution.candidate.settings,
                accumulated,
                subresult,
            )
        )

    if accumulated is None:
        message = "microbatch accumulation requires a nonempty batch"
        raise MaterializationError(message)

    runtime_values.require_finite_tree(accumulated, "microbatch result")

    return accumulated


def _single_step_microbatch_settings(settings: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(settings)
    result["schedule.gradient_accumulation"] = "single_step"
    result.pop("batch.data_microbatch_size", None)

    return result


def parameter_blocked_matrix_vector_product(
    matrix: torch.Tensor,
    vector: torch.Tensor,
    settings: Mapping[str, Any],
    parameter_surface: ParameterSurface | None,
    ranges: tuple[tuple[int, int], ...] | None = None,
) -> torch.Tensor:
    """Return the parameter-blocked matrix-vector product.

    Returns:
        The parameter-blocked matrix-vector product.

    Raises:
        MaterializationError: If the declared inputs are invalid.
    """
    column_ranges = (
        runtime_values.parameter_column_ranges(
            vector.numel(), settings, parameter_surface
        )
        if ranges is None
        else ranges
    )

    if column_ranges is None:
        return layout.matmul_runtime(settings, matrix, vector)

    if matrix.ndim != runtime_values.MATRIX_DIMS:
        message = "parameter-block matrix must be two-dimensional"
        raise MaterializationError(message)

    if vector.ndim != 1:
        message = "parameter-block vector must be one-dimensional"
        raise MaterializationError(message)

    if matrix.shape[1] != vector.numel():
        message = "parameter-block matrix columns must match vector width"
        raise MaterializationError(message)

    chunks = []

    for start, stop in column_ranges:
        chunks.append(
            layout.matmul_runtime(settings, matrix[:, start:stop], vector[start:stop])
        )

    result = chunks[0]

    for chunk in chunks[1:]:
        result = result + chunk

    return result


def parameter_order_vector(
    execution: runtime_values.StandardExecution,
) -> torch.Tensor:
    """Return the vector flattened in parameter order.

    Returns:
        The vector flattened in parameter order.
    """
    if execution.flat_parameter_vector is not None:
        return execution.flat_parameter_vector

    return _build_parameter_order_vector(execution)


def _build_parameter_order_vector(
    execution: runtime_values.StandardExecution,
) -> torch.Tensor:
    vector_leaves = runtime_values.matching_vector_leaves(
        execution.params, execution.vector
    )
    vector_tensor = torch.cat(tuple(leaf.reshape(-1) for leaf in vector_leaves))
    runtime_values.require_finite_tensor(vector_tensor, "streaming Fisher vector")

    return vector_tensor


STANDARD_RUNNERS = {
    "gradient": derivatives.run_gradient,
    "jvp": derivatives.run_jvp,
    "vjp": derivatives.run_vjp,
    "hvp": derivatives.run_hvp,
    "ggnvp": ggn.run_ggnvp,
    "fisher_vp": fisher.run_fisher_vp,
    "sampled_fisher_vp": fisher.run_sampled_fisher_vp,
    "empirical_fisher_vp": fisher.run_empirical_fisher_vp,
    "per_example_gradient": derivatives.run_per_example_gradient,
    "metric": metrics.run_metric,
    "sqrt_metric": metrics.run_sqrt_metric,
    "inverse_sqrt_metric": metrics.run_sqrt_metric,
    "metric_inner": metrics.run_metric_inner,
    "inverse_metric": metrics.run_inverse_metric,
    "inverse_metric_inner": metrics.run_inverse_metric_inner,
}


def standard_materializer(
    operation_factory: RuntimeOperationFactory,
    operator: OperatorSpec | None = None,
    mmap_residency: Callable[[torch.Tensor, str], torch.Tensor] | None = None,
) -> Materializer:
    """Return the standard materializer for an operation factory.

    Returns:
        The standard materializer for an operation factory.
    """
    mmap_residency_callback = mmap_residency

    def callback(candidate: Candidate, record: FullSizeRecord) -> Any:
        if (
            record.family != candidate.family
            or record.candidate_id != candidate.candidate_id
        ):
            message = "selected record does not match selected candidate"
            raise MaterializationError(message)

        if operator is not None and operator.kind == "metric":
            return metrics.StandardMetricOperator(
                candidate,
                record,
                operator,
                metrics.metric_representation(operator),
                mmap_residency=mmap_residency_callback,
            )

        if operator is not None and operator.kind == "inverse_metric":
            return metrics.StandardMetricOperator(
                candidate,
                record,
                operator,
                metrics.metric_representation(operator),
                default_operation="inverse_multiply",
                damping=metrics.inverse_metric_damping_payload(operator),
                inverse_path=runtime_path(operator, candidate),
                mmap_residency=mmap_residency_callback,
            )

        if candidate.settings.get("compile.boundary") == "bound_operator_vector_step":
            eager_candidate = _candidate_without_compile_settings(candidate)
            compiled_vector_step = None
            bound_batch_signature = None

            def selected(batch: Batch, vector: TensorTree) -> TensorTree:
                nonlocal bound_batch_signature, compiled_vector_step

                current_batch_signature = runtime_values.batch_signature(batch)

                if bound_batch_signature is None:
                    bound_batch_signature = current_batch_signature
                elif current_batch_signature != bound_batch_signature:
                    message = (
                        "compile.boundary=bound_operator_vector_step requires a "
                        "fixed batch signature"
                    )
                    raise MaterializationError(message)

                if compiled_vector_step is None:
                    fixed_batch = dict(batch)

                    def vector_step(step_vector: TensorTree) -> TensorTree:
                        return operation_factory(
                            eager_candidate,
                            fixed_batch,
                            step_vector,
                        )()

                    compiled_vector_step = _compiled_bound_vector_step(
                        candidate.settings,
                        vector_step,
                        vector,
                    )

                return compiled_vector_step(vector)

            return selected

        def selected(batch: Batch, vector: TensorTree) -> TensorTree:
            return operation_factory(candidate, batch, vector)()

        return selected

    return CallableMaterializer(
        "vptune.standard_runtime",
        PACKAGE_VERSION,
        {"operation_factory": dict(operation_factory.identity())},
        {"callback": "standard_materializer.callback"},
        callback,
    )


def _candidate_without_compile_settings(candidate: Candidate) -> Candidate:
    return dataclasses.replace(
        candidate,
        settings={
            key: value
            for key, value in candidate.settings.items()
            if key not in runtime_values.COMPILE_SETTING_KEYS
        },
    )


def standard_runtime_with_matrix_free_bindings(
    runtime: RuntimeConfig,
    *,
    bindings: Mapping[str, Callable[[Batch, TensorTree], TensorTree]],
    binding_signature: Mapping[str, Any],
) -> RuntimeConfig:
    """Return a standard runtime bound to selected matrix-free products."""
    if not bindings:
        return runtime

    binding_map = dict(bindings)
    runtime_signature = {
        **dict(runtime.signature),
        "matrix_free_bindings": dict(binding_signature),
    }
    operation_factory = _matrix_free_bound_operation_factory(
        runtime.operation_factory,
        binding_map,
        runtime_signature,
    )
    reference_check = _matrix_free_bound_reference_check(
        runtime.reference_check,
        binding_map,
        runtime_signature,
    )
    materializer = _matrix_free_bound_materializer(
        runtime.materializer,
        binding_map,
        runtime_signature,
    )

    return dataclasses.replace(
        runtime,
        operation_factory=operation_factory,
        reference_check=reference_check,
        materializer=materializer,
        signature=runtime_signature,
    )


def _matrix_free_bound_operation_factory(
    operation_factory: RuntimeOperationFactory,
    bindings: Mapping[str, Callable[[Batch, TensorTree], TensorTree]],
    runtime_signature: Mapping[str, Any],
) -> RuntimeOperationFactory:
    def factory(
        candidate: Candidate,
        batch: Batch,
        vector: TensorTree,
    ) -> CandidateOperation:
        operation = operation_factory(candidate, batch, vector)

        def bound_operation() -> TensorTree:
            return runtime_values.run_with_matrix_free_runtime_bindings(
                bindings, operation
            )

        return bound_operation

    return CallableOperationFactory(
        "vptune.matrix_free_bound_operation_factory",
        PACKAGE_VERSION,
        runtime_signature,
        {"base": dict(operation_factory.identity())},
        factory,
    )


def _matrix_free_bound_reference_check(
    reference_check: RuntimeReferenceCheck,
    bindings: Mapping[str, Callable[[Batch, TensorTree], TensorTree]],
    runtime_signature: Mapping[str, Any],
) -> RuntimeReferenceCheck:
    def check(
        candidate: Candidate,
        batch: Batch,
        vector: TensorTree,
    ) -> ReferenceResult:
        return runtime_values.run_with_matrix_free_runtime_bindings(
            bindings,
            lambda: reference_check(candidate, batch, vector),
        )

    return CallableReferenceCheck(
        "vptune.matrix_free_bound_reference_check",
        PACKAGE_VERSION,
        runtime_signature,
        {"base": dict(reference_check.identity())},
        check,
    )


def _matrix_free_bound_materializer(
    materializer: Materializer,
    bindings: Mapping[str, Callable[[Batch, TensorTree], TensorTree]],
    runtime_signature: Mapping[str, Any],
) -> Materializer:
    def callback(candidate: Candidate, record: FullSizeRecord) -> Any:
        selected = materializer(candidate, record)

        if isinstance(selected, metrics.StandardMetricOperator):
            return dataclasses.replace(selected, matrix_free_operators=dict(bindings))

        if callable(selected):

            def bound_selected(batch: Batch, vector: TensorTree) -> TensorTree:
                return runtime_values.run_with_matrix_free_runtime_bindings(
                    bindings,
                    lambda: selected(batch, vector),
                )

            return bound_selected

        return selected

    return CallableMaterializer(
        "vptune.matrix_free_bound_materializer",
        PACKAGE_VERSION,
        runtime_signature,
        {"base": dict(materializer.identity())},
        callback,
    )


def runtime_path(operator: OperatorSpec, candidate: Candidate) -> str:
    """Return the runtime path.

    Returns:
        The runtime path.

    Raises:
        MaterializationError: If the declared inputs are invalid.
    """
    spec_path = _spec_runtime_path(operator, candidate)

    if spec_path is not None:
        if "operator_path" in candidate.settings:
            message = "candidate cannot mix operator_path with SPEC path keys"
            raise MaterializationError(message)

        return spec_path

    if "operator_path" in candidate.settings:
        message = f"operator_path is not a setting for {operator.kind}"
        raise MaterializationError(message)

    if operator.kind in runtime_values.SPEC_REQUIRED_PATH_OPERATORS:
        key = runtime_values.SPEC_PATH_KEYS[operator.kind]
        message = f"{key} is required for {operator.kind}"
        raise MaterializationError(message)

    message = f"standard runtime has no SPEC path key for {operator.kind}"
    raise MaterializationError(message)


def _spec_runtime_path(operator: OperatorSpec, candidate: Candidate) -> str | None:
    special_paths = {
        "ggnvp": ggn.ggn_spec_runtime_path,
        "fisher_vp": fisher.fisher_spec_runtime_path,
        "sampled_fisher_vp": fisher.sampled_fisher_spec_runtime_path,
        "empirical_fisher_vp": fisher.empirical_fisher_spec_runtime_path,
        "per_example_gradient": derivatives.per_example_gradient_spec_runtime_path,
    }
    special_path = special_paths.get(operator.kind)

    if special_path is not None:
        return special_path(candidate)

    key = runtime_values.SPEC_PATH_KEYS.get(operator.kind)

    if key is None or key not in candidate.settings:
        return None

    value = candidate.settings[key]
    path_map = runtime_values.SPEC_PATH_TO_RUNTIME[operator.kind]
    path = path_map.get(value)

    if path is None:
        message = f"{key} value is not lowered by standard runtime: {value}"
        raise MaterializationError(message)

    return path


def require_supported_standard_settings(
    operator: OperatorSpec,
    candidate: Candidate,
    parameter_surface: ParameterSurface | None = None,
    mmap_residency: Callable[[torch.Tensor, str], torch.Tensor] | None = None,
    fusion_rewriter: (
        Callable[[torch.nn.Module, Candidate], torch.nn.Module] | None
    ) = None,
    batch_layout: Callable[[Candidate, Batch], Batch] | None = None,
    lm_head_chunker: Callable[[Candidate, Batch], Batch] | None = None,
    activation_pack_hooks: runtime_values.ActivationPackHooks | None = None,
    activation_unpack_hooks: runtime_values.ActivationUnpackHooks | None = None,
    checkpoint_contexts: runtime_values.CheckpointContextFns | None = None,
) -> None:
    """Validate that declared settings are supported by the standard runtime.

    Raises:
        MaterializationError: If the declared inputs are invalid.
    """
    unsupported = tuple(
        key
        for key in candidate.settings
        if key not in runtime_values.SUPPORTED_STANDARD_SETTINGS
    )

    if unsupported:
        message = f"standard runtime settings are unsupported: {unsupported}"
        raise MaterializationError(message)

    path = runtime_path(operator, candidate)
    layout.require_dtype_runtime_settings(candidate.settings)
    _require_teacher_output_settings(candidate.settings)
    _require_input_schedule_settings(operator, path, candidate.settings, batch_layout)
    _require_input_residency_settings(candidate.settings)
    _require_memory_residency_settings(
        operator,
        candidate.settings,
        mmap_residency,
    )
    _require_memory_recompute_settings(operator, candidate.settings)
    runtime_values.require_output_buffer_settings(candidate.settings)
    runtime_values.require_fusion_settings(
        operator, candidate.settings, fusion_rewriter
    )
    runtime_values.require_call_runtime_settings(candidate.settings)
    runtime_values.require_stateful_module_path_settings(
        operator, path, candidate.settings
    )
    derivatives.require_gradient_graph_schedule_settings(operator, candidate.settings)
    ggn.require_ggn_loss_hessian_settings(operator, candidate.settings)
    ggn.require_ggn_batch_size_settings(operator, path, candidate.settings)
    ggn.require_ggn_vjp_path_settings(operator, path, candidate.settings)
    runtime_values.require_output_cotangent_block_settings(
        operator, path, candidate.settings
    )
    runtime_values.require_lm_head_chunking_settings(
        candidate.settings, lm_head_chunker
    )
    runtime_values.require_parameter_block_size_settings(
        operator,
        path,
        candidate.settings,
        parameter_surface,
    )
    ggn.require_ggn_reuse_settings(operator, path, candidate.settings)
    derivatives.require_hvp_reuse_settings(operator, path, candidate.settings)
    derivatives.require_hvp_row_batch_size_settings(operator, path, candidate.settings)
    vectorization.require_vectorization_mode_settings(
        operator.kind, path, candidate.settings
    )
    _require_activation_runtime_settings(
        candidate.settings,
        activation_pack_hooks,
        activation_unpack_hooks,
        checkpoint_contexts,
    )
    vectorization.require_vectorization_setting_keys(
        operator.kind, path, candidate.settings
    )
    runtime_values.require_transform_admission_settings(
        operator, path, candidate.settings
    )
    derivatives.require_gradient_value_reuse_settings(
        operator, path, candidate.settings
    )
    derivatives.require_jvp_linearize_reuse_settings(operator, path, candidate.settings)
    derivatives.require_vjp_closure_reuse_settings(operator, path, candidate.settings)
    metrics.require_metric_runtime_settings(operator, path, candidate.settings)
    metrics.require_inverse_metric_factor_reuse_settings(
        operator,
        path,
        candidate.settings,
    )
    metrics.require_inverse_metric_multi_rhs_settings(
        operator, path, candidate.settings
    )
    layout.require_layout_runtime_settings(candidate.settings)


def _require_teacher_output_settings(settings: Mapping[str, Any]) -> None:
    value = settings.get("teacher_outputs")

    if value is None:
        return

    if value in {"precomputed_cpu", "precomputed_cpu_pinned", "precomputed_gpu"}:
        return

    if value == "recomputed_with_equality_check":
        return

    message = f"teacher_outputs is unsupported: {value}"
    raise MaterializationError(message)


def _require_recomputed_teacher_objective(
    settings: Mapping[str, Any],
    teacher_objective: FunctionObjective | None,
) -> None:
    if settings.get("teacher_outputs") != "recomputed_with_equality_check":
        return

    if teacher_objective is None:
        message = "recomputed teacher outputs require a teacher objective"
        raise MaterializationError(message)


def _require_stateful_module_execution(
    execution: runtime_values.StandardExecution,
) -> None:
    if execution.candidate.settings.get("call.path") != "stateful_module":
        return

    if execution.module is None:
        message = "call.path=stateful_module requires a module"
        raise MaterializationError(message)

    if execution.module_call is None:
        message = "call.path=stateful_module requires module_call"
        raise MaterializationError(message)

    runtime_values.require_module_state_names(
        execution.module,
        execution.params,
        execution.buffers,
    )


def stateful_model_tree(
    values: dict[str, torch.Tensor],
    settings: Mapping[str, Any],
) -> dict[str, torch.Tensor]:
    """Return the tree prepared for a stateful module call.

    Returns:
        The tree prepared for a stateful module call.
    """
    return model_compute_tree(values, settings)


def stateful_model_batch(
    batch: Batch,
    settings: Mapping[str, Any],
) -> Batch:
    """Return the batch prepared for a stateful module call.

    Returns:
        The batch prepared for a stateful module call.
    """
    return model_compute_batch(batch, settings)


def _call_compiled_model_forward(
    execution: runtime_values.StandardExecution,
    compiled_model_forward: Callable[[Batch], object],
    active_params: ParameterTree,
) -> object:
    if execution.module is None:
        message = "compile.boundary=model_forward requires module"
        raise CompileSetupError(message)

    settings = execution.candidate.settings
    model_params = stateful_model_tree(active_params, settings)
    model_buffers = stateful_model_tree(execution.buffers, settings)
    model_batch = stateful_model_batch(execution.batch, settings)
    slots = runtime_values.replace_module_state(
        execution.module, model_params, model_buffers
    )

    try:
        return compiled_model_forward(model_batch)
    finally:
        runtime_values.restore_module_state(slots)


def model_compute_tree(
    values: dict[str, torch.Tensor],
    settings: Mapping[str, Any],
) -> dict[str, torch.Tensor]:
    """Return the tree cast to the declared model compute dtype.

    Returns:
        The tree cast to the declared model compute dtype.
    """
    dtype = layout.dtype_setting(settings, "dtype.model_compute")

    return layout.runtime_named_tensor_dtype(values, dtype)


def model_compute_batch(
    batch: Batch,
    settings: Mapping[str, Any],
) -> Batch:
    """Return the batch cast to the declared model compute dtype.

    Returns:
        The batch cast to the declared model compute dtype.
    """
    dtype = layout.dtype_setting(settings, "dtype.model_compute")

    if dtype is None:
        return batch

    return {
        key: runtime_values.runtime_batch_value(value, dtype)
        for key, value in batch.items()
    }


def _require_input_schedule_settings(
    operator: OperatorSpec,
    path: str | None,
    settings: Mapping[str, Any],
    batch_layout_callback: Callable[[Candidate, Batch], Batch] | None,
) -> None:
    per_example = settings.get("schedule.per_example")

    if per_example is None:
        if path in runtime_values.VMAP_RUNTIME_PATHS:
            message = "vmap_grad rows require schedule.per_example=vmap"
            raise MaterializationError(message)
    else:
        runtime_values.require_batch_data_axis(operator, "schedule.per_example")
        runtime_values.require_per_example_schedule(path, per_example)

    derivatives.require_per_example_batch_size_settings(path, settings)

    runtime_values.require_per_token_schedule(operator, settings, batch_layout_callback)

    batch_layout = settings.get("input.batch_layout")

    if batch_layout not in {
        None,
        "dense_padded",
        "packed_with_inverse_permutation",
        "variable_length",
    }:
        message = f"input.batch_layout is unsupported: {batch_layout}"
        raise MaterializationError(message)

    if batch_layout not in {None, "dense_padded"} and batch_layout_callback is None:
        message = f"input.batch_layout requires input-layout binding: {batch_layout}"
        raise MaterializationError(message)

    length_grouping = settings.get("input.length_grouping")

    if length_grouping not in {None, "none", "exact_length_bucket"}:
        message = f"input.length_grouping is unsupported: {length_grouping}"
        raise MaterializationError(message)

    if length_grouping not in {None, "none"} and batch_layout_callback is None:
        message = (
            f"input.length_grouping requires input-order restoration: {length_grouping}"
        )
        raise MaterializationError(message)

    gradient_accumulation = settings.get("schedule.gradient_accumulation")

    if gradient_accumulation in {None, "single_step"}:
        if "batch.data_microbatch_size" in settings:
            message = (
                "batch.data_microbatch_size requires "
                "schedule.gradient_accumulation=microbatch_accumulate"
            )
            raise MaterializationError(message)

        return

    if gradient_accumulation != "microbatch_accumulate":
        message = (
            f"schedule.gradient_accumulation is unsupported: {gradient_accumulation}"
        )
        raise MaterializationError(message)

    runtime_values.require_batch_data_axis(
        operator,
        "schedule.gradient_accumulation=microbatch_accumulate",
    )

    if (
        settings.get("compile.enabled") == "true"
        and settings.get("compile.boundary") == "loss_closure"
    ):
        message = "loss_closure compile boundary is incompatible with microbatching"
        raise MaterializationError(message)

    _data_microbatch_size(settings)


def _data_microbatch_size(settings: Mapping[str, Any]) -> int:
    key = "batch.data_microbatch_size"

    return runtime_values.required_positive_int_setting(
        settings,
        key,
        f"{key} must be a positive integer",
    )


def _require_input_residency_settings(settings: Mapping[str, Any]) -> None:
    residency = settings.get("input.residency")
    host_to_device = settings.get("input.host_to_device")

    if residency is None and host_to_device is None:
        return

    if residency not in {"cpu_staged", "cpu_pinned", "gpu"}:
        message = f"input.residency is unsupported: {residency}"
        raise MaterializationError(message)

    if host_to_device not in {"outside_measured_call", "inside_measured_call"}:
        message = f"input.host_to_device is unsupported: {host_to_device}"
        raise MaterializationError(message)


def _require_memory_residency_settings(
    operator: OperatorSpec,
    settings: Mapping[str, Any],
    mmap_residency: Callable[[torch.Tensor, str], torch.Tensor] | None,
) -> None:
    vector_residency = settings.get("memory.vector_residency")
    factor_residency = settings.get("memory.factor_residency")
    intermediate_residency = settings.get("memory.intermediate_residency")

    if vector_residency is not None:
        _require_runtime_residency(
            vector_residency,
            "memory.vector_residency",
            allow_mmap=mmap_residency is not None,
        )

    if factor_residency is not None:
        _require_runtime_residency(
            factor_residency,
            "memory.factor_residency",
            allow_mmap=mmap_residency is not None,
        )

    if intermediate_residency is None:
        return

    if operator.kind in {"composition", "ggnvp"}:
        _require_runtime_residency(
            intermediate_residency,
            "memory.intermediate_residency",
            allow_mmap=False,
        )

        if (
            operator.kind == "composition"
            and settings.get("composition.execution") == "fuse_adjacent_children"
        ):
            message = (
                "memory.intermediate_residency requires visible composition child "
                "boundaries"
            )
            raise MaterializationError(message)

        return

    if intermediate_residency in {"gpu", "cpu_staged", "cpu_pinned"}:
        message = "memory.intermediate_residency requires named intermediate boundaries"
        raise MaterializationError(message)

    message = f"memory.intermediate_residency is unsupported: {intermediate_residency}"
    raise MaterializationError(message)


def _require_memory_recompute_settings(
    operator: OperatorSpec,
    settings: Mapping[str, Any],
) -> None:
    for key in (
        "memory.primal_outputs",
        "memory.jvp_outputs",
        "memory.output_cotangents",
    ):
        value = settings.get(key)

        if value is None or value == "retain":
            continue

        if value == "recompute":
            _require_memory_recompute_lowering(operator, settings, key)
            continue

        message = f"{key} is unsupported: {value}"
        raise MaterializationError(message)


def _require_memory_recompute_lowering(
    operator: OperatorSpec,
    settings: Mapping[str, Any],
    key: str,
) -> None:
    if (
        key == "memory.primal_outputs"
        and operator.kind == "hvp"
        and settings.get("hvp.path") == runtime_values.HVP_REFERENCE_PATH
        and settings.get("hvp.primal_reuse") == "recompute_primal"
    ):
        return

    if (
        key == "memory.jvp_outputs"
        and operator.kind == "ggnvp"
        and settings.get("ggn.jvp_reuse") == "recompute_jvp"
    ):
        return

    if (
        key == "memory.output_cotangents"
        and operator.kind == "ggnvp"
        and settings.get("ggn.cotangent_reuse") == "recompute_output_cotangent"
    ):
        return

    message = f"{key}=recompute requires matching package-owned recompute settings"
    raise MaterializationError(message)


def _require_runtime_residency(
    value: Any,
    key: str,
    *,
    allow_mmap: bool,
) -> None:
    if value in {"cpu_staged", "cpu_pinned", "gpu"}:
        return

    if value == "mmap_cpu":
        if allow_mmap:
            return

        message = f"{key}=mmap_cpu requires memory-mapped tensor metadata"
        raise MaterializationError(message)

    message = f"{key} is unsupported: {value}"
    raise MaterializationError(message)


def _require_activation_runtime_settings(
    settings: Mapping[str, Any],
    activation_pack_hooks: runtime_values.ActivationPackHooks | None,
    activation_unpack_hooks: runtime_values.ActivationUnpackHooks | None,
    checkpoint_contexts: runtime_values.CheckpointContextFns | None,
) -> None:
    if not runtime_values.has_activation_settings(settings):
        return

    recompute = settings.get("activation.recompute")
    offload = settings.get("activation.offload")

    if recompute not in {
        "none",
        "checkpoint_non_reentrant_by_layer",
        "checkpoint_selective",
        "manual_recompute",
    }:
        message = f"activation.recompute is unsupported: {recompute}"
        raise MaterializationError(message)

    if offload not in {"none", "saved_tensor_hooks_cpu", "custom_saved_tensor_hooks"}:
        message = f"activation.offload is unsupported: {offload}"
        raise MaterializationError(message)

    if (
        recompute == "checkpoint_selective"
        and settings.get("checkpoint.context_fn") != "declared_context_pair"
    ):
        message = (
            "checkpoint_selective requires checkpoint.context_fn=declared_context_pair"
        )
        raise MaterializationError(message)

    if recompute in {"checkpoint_non_reentrant_by_layer", "checkpoint_selective"}:
        _admit_checkpoint_runtime(settings)
        _require_checkpoint_context_binding(settings, checkpoint_contexts)

        if offload == "custom_saved_tensor_hooks":
            _require_activation_hook_binding(
                settings,
                activation_pack_hooks,
                activation_unpack_hooks,
            )

        return

    _require_disabled_checkpoint_settings(settings, recompute)

    if offload == "custom_saved_tensor_hooks":
        _require_activation_hook_binding(
            settings,
            activation_pack_hooks,
            activation_unpack_hooks,
        )


def _admit_checkpoint_runtime(settings: Mapping[str, Any]) -> None:
    try:
        admit_checkpoint(settings)
    except AdmissionError as error:
        raise MaterializationError(str(error)) from error


def _require_checkpoint_context_binding(
    settings: Mapping[str, Any],
    checkpoint_contexts: runtime_values.CheckpointContextFns | None,
) -> None:
    if settings.get("checkpoint.context_fn") != "declared_context_pair":
        return

    context_id = settings.get("checkpoint.context_fn_callable")

    if not isinstance(context_id, str):
        message = "checkpoint.context_fn=declared_context_pair requires context id"
        raise MaterializationError(message)

    if checkpoint_contexts is None or context_id not in checkpoint_contexts:
        message = f"checkpoint context is not registered: {context_id}"
        raise MaterializationError(message)


def _require_activation_hook_binding(
    settings: Mapping[str, Any],
    activation_pack_hooks: runtime_values.ActivationPackHooks | None,
    activation_unpack_hooks: runtime_values.ActivationUnpackHooks | None,
) -> None:
    pack_hook_id = settings.get("activation.pack_hook")
    unpack_hook_id = settings.get("activation.unpack_hook")

    if not isinstance(pack_hook_id, str) or not isinstance(unpack_hook_id, str):
        message = "custom_saved_tensor_hooks requires activation pack and unpack hooks"
        raise MaterializationError(message)

    if activation_pack_hooks is None or pack_hook_id not in activation_pack_hooks:
        message = f"activation pack hook is not registered: {pack_hook_id}"
        raise MaterializationError(message)

    if activation_unpack_hooks is None or unpack_hook_id not in activation_unpack_hooks:
        message = f"activation unpack hook is not registered: {unpack_hook_id}"
        raise MaterializationError(message)


def _require_disabled_checkpoint_settings(
    settings: Mapping[str, Any],
    recompute: str,
) -> None:
    disabled = {
        "checkpoint.use_reentrant": "false",
        "checkpoint.early_stop": "false",
        "checkpoint.preserve_rng_state": "false",
        "checkpoint.determinism_check": "none",
        "checkpoint.context_fn": "none",
        "checkpoint.moves_to_new_device": "false",
        "checkpoint.uses_global_state": "false",
    }

    for key, value in disabled.items():
        if key in settings and settings[key] != value:
            message = f"activation.recompute={recompute} requires {key}={value}"
            raise MaterializationError(message)


def tree_dot_runtime(
    settings: Mapping[str, Any],
    left: TensorTree,
    right: TensorTree,
) -> torch.Tensor:
    """Return the runtime dot product of two tensor trees.

    Returns:
        the runtime dot product of two tensor trees.
    """
    left = runtime_intermediate_tree(left, settings)
    right = runtime_intermediate_tree(right, settings)
    left = _accumulation_tree(left, settings)
    right = _accumulation_tree(right, settings)

    if layout.layout_vector_ops(settings) == "foreach":
        return tree_dot_foreach(left, right)

    return tree_dot(left, right)


def tree_add_runtime(
    settings: Mapping[str, Any],
    left: TensorTree,
    right: TensorTree,
) -> TensorTree:
    """Return the elementwise sum of two tensor trees.

    Returns:
        The elementwise sum of two tensor trees.
    """
    left = runtime_intermediate_tree(left, settings)
    right = runtime_intermediate_tree(right, settings)
    left = _accumulation_tree(left, settings)
    right = _accumulation_tree(right, settings)

    if layout.layout_vector_ops(settings) == "foreach":
        return tree_add_foreach(left, right)

    return tree_map2(torch.add, left, right)


def dot_runtime(
    settings: Mapping[str, Any],
    left: torch.Tensor,
    right: torch.Tensor,
) -> torch.Tensor:
    """Return the runtime dot product of two tensors.

    Returns:
        the runtime dot product of two tensors.
    """
    left = runtime_intermediate_tensor(left, settings)
    right = runtime_intermediate_tensor(right, settings)

    return torch.dot(
        accumulation_tensor(left, settings),
        accumulation_tensor(right, settings),
    )


def tree_scale_runtime(
    settings: Mapping[str, Any],
    tree: TensorTree,
    scale: float,
) -> TensorTree:
    """Return the tree scaled by a runtime scalar.

    Returns:
        The tree scaled by a runtime scalar.
    """
    tree = runtime_intermediate_tree(tree, settings)

    if layout.layout_vector_ops(settings) == "foreach":
        return tree_mul_foreach(tree, scale)

    return tree_map(lambda tensor: tensor * scale, tree)


def runtime_intermediate_tree(
    tree: TensorTree,
    settings: Mapping[str, Any],
) -> TensorTree:
    """Return an intermediate tree in the declared residency and dtype.

    Returns:
        an intermediate tree in the declared residency and dtype.
    """
    return tree_map(lambda tensor: runtime_intermediate_tensor(tensor, settings), tree)


def runtime_intermediate_tensor(
    tensor: torch.Tensor,
    settings: Mapping[str, Any],
) -> torch.Tensor:
    """Return an intermediate tensor in the declared residency and dtype.

    Returns:
        an intermediate tensor in the declared residency and dtype.
    """
    dtype = layout.dtype_setting(settings, "dtype.intermediate")

    if dtype is None or not tensor.is_floating_point():
        return tensor

    return tensor.to(dtype=dtype)


def _runtime_params(
    params: ParameterTree,
    settings: Mapping[str, Any],
    parameter_surface: ParameterSurface | None,
) -> ParameterTree:
    runtime_values.require_parameter_surface_runtime_settings(
        parameter_surface, settings
    )
    dtype = layout.parameter_dtype(settings)
    result = layout.runtime_named_tensor_dtype(params, dtype)
    result = layout.runtime_named_tensor_contiguity(result, settings)

    if settings.get("layout.params") == "flat_contiguous":
        layout.require_alias_safe_parameter_layout(result, settings)

        return runtime_values.wrap_flat_parameter_tree(
            result,
            runtime_values.flatten_vector(result).contiguous(),
        )

    if settings.get("layout.params") in {"per_layer_flat", "per_block_flat"}:
        layout.require_alias_safe_parameter_layout(result, settings)

    return layout.runtime_grouped_parameter_layout(
        result,
        settings,
        "layout.params",
        parameter_surface,
    )


def _runtime_buffers(
    buffers: BufferTree,
    settings: Mapping[str, Any],
) -> BufferTree:
    dtype = layout.parameter_dtype(settings)

    if dtype is None:
        return layout.runtime_named_tensor_contiguity(buffers, settings)

    return layout.runtime_named_tensor_contiguity(
        {key: tensor.to(dtype=dtype) for key, tensor in buffers.items()},
        settings,
    )


def runtime_batch(
    batch: Batch,
    settings: Mapping[str, Any],
    *,
    move_input_residency: bool = True,
    mmap_residency: Callable[[torch.Tensor, str], torch.Tensor] | None = None,
) -> Batch:
    """Return the runtime batch.

    Returns:
        The runtime batch.
    """
    dtype = layout.batch_dtype(settings)
    metric_factor_dtype = layout.dtype_setting(settings, "dtype.metric_factor")
    metric_factor_residency = settings.get("memory.factor_residency")

    if (
        dtype is None
        and metric_factor_dtype is None
        and metric_factor_residency is None
    ):
        return layout.runtime_batch_after_contiguity(
            batch,
            settings,
            move_input_residency=move_input_residency,
        )

    result = (
        batch
        if dtype is None
        else {
            key: runtime_values.runtime_batch_value(value, dtype)
            for key, value in batch.items()
        }
    )

    if metric_factor_dtype is None and metric_factor_residency is None:
        return layout.runtime_batch_after_contiguity(
            result,
            settings,
            move_input_residency=move_input_residency,
        )

    if metric_factor_dtype is not None:
        result = {
            key: metrics.runtime_metric_factor_value(key, value, metric_factor_dtype)
            for key, value in result.items()
        }

    if metric_factor_residency is not None:
        result = {
            key: metrics.runtime_metric_factor_residency_value(
                key,
                value,
                metric_factor_residency,
                mmap_residency,
            )
            for key, value in result.items()
        }

    return layout.runtime_batch_after_contiguity(
        result,
        settings,
        move_input_residency=move_input_residency,
    )


def _runtime_declared_batch_transforms(
    batch: Batch,
    candidate: Candidate,
    batch_layout: Callable[[Candidate, Batch], Batch] | None,
    lm_head_chunker: Callable[[Candidate, Batch], Batch] | None,
) -> Batch:
    result = batch

    if layout.uses_declared_batch_layout(candidate.settings):
        if batch_layout is None:
            message = "input batch layout requires declared binding"
            raise MaterializationError(message)

        result = batch_layout(candidate, result)

    if "chunk.lm_head_weight_chunk_bytes" in candidate.settings:
        if lm_head_chunker is None:
            message = "chunk.lm_head_weight_chunk_bytes requires LM-head weight binding"
            raise MaterializationError(message)

        result = lm_head_chunker(candidate, result)

    return result


def runtime_batch_input_residency(
    batch: Batch,
    settings: Mapping[str, Any],
) -> Batch:
    """Return the batch moved to the declared input residency.

    Returns:
        The batch moved to the declared input residency.
    """
    residency = settings.get("input.residency")

    if residency is None:
        return batch

    result = dict(batch)

    for key, value in batch.items():
        if key != "teacher_outputs":
            result[key] = _input_residency_value(value, residency)

    return result


def _input_residency_value(value: Any, residency: Any) -> Any:
    return runtime_values.runtime_nested_tensor_value(
        value,
        lambda tensor: _input_residency_tensor(tensor, residency),
    )


def _input_residency_tensor(tensor: torch.Tensor, residency: Any) -> torch.Tensor:
    return _residency_tensor(tensor, residency, "input.residency")


def tree_residency(tree: TensorTree, residency: Any, key: str) -> TensorTree:
    """Return the tree moved to the declared residency.

    Returns:
        The tree moved to the declared residency.
    """
    return tree_map(lambda tensor: _residency_tensor(tensor, residency, key), tree)


def _residency_tensor(
    tensor: torch.Tensor,
    residency: Any,
    key: str,
    mmap_residency: Callable[[torch.Tensor, str], torch.Tensor] | None = None,
) -> torch.Tensor:
    if residency == "cpu_staged":
        return tensor.to(device=torch.device("cpu"))

    if residency == "cpu_pinned":
        cpu_tensor = tensor.to(device=torch.device("cpu"))

        try:
            return cpu_tensor.pin_memory()
        except RuntimeError as error:
            raise MaterializationError(str(error)) from error

    if residency == "gpu":
        if not torch.cuda.is_available():
            message = f"{key}=gpu requires CUDA"
            raise MaterializationError(message)

        return tensor.to(device=torch.device("cuda"))

    if residency == "mmap_cpu":
        if mmap_residency is None:
            message = f"{key}=mmap_cpu requires memory-mapped tensor metadata"
            raise MaterializationError(message)

        return mmap_residency(tensor, key)

    message = f"{key} is unsupported: {residency}"
    raise MaterializationError(message)


def runtime_residency_tensor(
    tensor: torch.Tensor,
    residency: Any,
    key: str,
    mmap_residency: Callable[[torch.Tensor, str], torch.Tensor] | None,
) -> torch.Tensor:
    """Return a tensor moved to the declared residency.

    Returns:
        a tensor moved to the declared residency.
    """
    if mmap_residency is None or residency != "mmap_cpu":
        return _residency_tensor(tensor, residency, key)

    return _residency_tensor(tensor, residency, key, mmap_residency)


def _execution_with_recomputed_teacher_outputs(
    execution: runtime_values.StandardExecution,
) -> runtime_values.StandardExecution:
    if execution.candidate.settings.get("teacher_outputs") != (
        "recomputed_with_equality_check"
    ):
        return execution

    if execution.teacher_objective is None:
        message = "recomputed teacher outputs require a teacher objective"
        raise MaterializationError(message)

    fixed_outputs = execution.batch.get("teacher_outputs")
    require_teacher_output_tree(fixed_outputs)
    settings = execution.candidate.settings
    recomputed_outputs = execution.teacher_objective(
        model_compute_tree(execution.params, settings),
        model_compute_tree(execution.buffers, settings),
        model_compute_batch(execution.batch, settings),
        execution.context,
    )
    _require_teacher_outputs_match(fixed_outputs, recomputed_outputs)
    batch = dict(execution.batch)
    batch["teacher_outputs"] = recomputed_outputs

    return dataclasses.replace(execution, batch=batch)


def require_teacher_output_tree(value: Any) -> None:
    """Validate the declared teacher output tree."""
    runtime_values.runtime_nested_tensor_value(
        value,
        lambda tensor: tensor,
        error_message="teacher_outputs batch field must be a tensor tree",
    )


def _require_teacher_outputs_match(fixed: Any, recomputed: Any) -> None:
    if _teacher_outputs_equal(fixed, recomputed):
        return

    message = "recomputed teacher outputs do not match fixed teacher_outputs"
    raise MaterializationError(message)


def _teacher_outputs_equal(left: Any, right: Any) -> bool:
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        return torch.equal(left, right)

    if isinstance(left, dict) and isinstance(right, dict):
        if set(left) != set(right):
            return False

        return all(_teacher_outputs_equal(left[key], right[key]) for key in left)

    if isinstance(left, tuple) and isinstance(right, tuple):
        if len(left) != len(right):
            return False

        return all(starmap(_teacher_outputs_equal, zip(left, right, strict=True)))

    return False


def runtime_vector(
    vector: TensorTree,
    settings: Mapping[str, Any],
    template: TensorTree | None = None,
    parameter_surface: ParameterSurface | None = None,
    mmap_residency: Callable[[torch.Tensor, str], torch.Tensor] | None = None,
) -> TensorTree:
    """Return the runtime vector.

    Returns:
        The runtime vector.
    """
    dtype = layout.dtype_setting(settings, "dtype.vector")

    if dtype is None:
        result = vector
    else:
        result = tree_map(lambda tensor: tensor.to(dtype=dtype), vector)

    result = _runtime_vector_residency(result, settings, mmap_residency)
    result = layout.runtime_vector_layout(
        result,
        settings,
        vector if template is None else template,
        parameter_surface,
    )

    return layout.runtime_tree_contiguity(result, settings)


def _runtime_vector_residency(
    vector: TensorTree,
    settings: Mapping[str, Any],
    mmap_residency: Callable[[torch.Tensor, str], torch.Tensor] | None,
) -> TensorTree:
    residency = settings.get("memory.vector_residency")

    if residency is None:
        return vector

    return tree_map(
        lambda tensor: runtime_residency_tensor(
            tensor,
            residency,
            "memory.vector_residency",
            mmap_residency,
        ),
        vector,
    )


def runtime_output(
    output: TensorTree,
    settings: Mapping[str, Any],
    parameter_surface: ParameterSurface | None = None,
) -> TensorTree:
    """Return the runtime output tree for declared output settings.

    Returns:
        The runtime output tree for declared output settings.
    """
    dtype = layout.dtype_setting(settings, "dtype.output")

    if dtype is not None:
        output = tree_map(lambda tensor: tensor.to(dtype=dtype), output)

    if layout.layout_output(settings) == "flat_contiguous":
        return runtime_values.flatten_vector(output).contiguous()

    return layout.runtime_grouped_output_layout(output, settings, parameter_surface)


def _standard_output_buffer(
    execution: runtime_values.StandardExecution,
) -> TensorTree | None:
    if execution.candidate.settings.get("memory.output_buffers") != "preallocated":
        return None

    template = _standard_output_template(execution)
    runtime_template = runtime_output(
        template,
        execution.candidate.settings,
        execution.parameter_surface,
    )

    return tree_map(torch.empty_like, runtime_template)


def _standard_output_template(
    execution: runtime_values.StandardExecution,
) -> TensorTree:
    kind = execution.operator.kind

    if kind == "jvp":
        return derivatives.jvp_output_template(execution)

    if kind in {
        "gradient",
        "vjp",
        "hvp",
        "ggnvp",
        "fisher_vp",
        "sampled_fisher_vp",
        "empirical_fisher_vp",
    }:
        return execution.params

    if kind in {"metric", "inverse_metric", "composition"}:
        return execution.vector

    message = f"memory.output_buffers=preallocated lacks output template for {kind}"
    raise MaterializationError(message)


def accumulation_tensor(
    tensor: torch.Tensor,
    settings: Mapping[str, Any],
) -> torch.Tensor:
    """Return the accumulation tensor for the declared accumulation dtype.

    Returns:
        the accumulation tensor for the declared accumulation dtype.
    """
    dtype = layout.dtype_setting(settings, "dtype.accumulation")

    if dtype is None:
        return tensor

    return tensor.to(dtype=dtype)


def _accumulation_tree(tree: TensorTree, settings: Mapping[str, Any]) -> TensorTree:
    dtype = layout.dtype_setting(settings, "dtype.accumulation")

    if dtype is None:
        return tree

    return tree_map(lambda tensor: tensor.to(dtype=dtype), tree)


def run_with_backend_settings(
    settings: Mapping[str, Any],
    callback: Callable[[], Any],
) -> Any:
    """Run with backend settings.

    Returns:
        The with backend settings result.
    """
    if not runtime_values.BACKEND_SETTINGS_ENABLED[0]:
        return callback()

    matmul_precision = layout.matmul_precision_setting(settings)
    autocast_setting = runtime_values.autocast_setting(settings)
    allow_bf16_reduction = runtime_values.bool_string_setting(
        settings,
        "numeric.bf16_reduced_precision_reduction",
    )
    allow_fp16_reduction = runtime_values.bool_string_setting(
        settings,
        "numeric.fp16_reduced_precision_reduction",
    )
    deterministic_algorithms = runtime_values.bool_string_setting(
        settings,
        "numeric.deterministic_algorithms",
    )
    previous_matmul_precision = torch.get_float32_matmul_precision()
    previous_allow_bf16_reduction = (
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
    )
    previous_allow_fp16_reduction = (
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction
    )
    previous_deterministic_algorithms = torch.are_deterministic_algorithms_enabled()

    if matmul_precision is not None:
        torch.set_float32_matmul_precision(matmul_precision)

    if allow_bf16_reduction is not None:
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = (
            allow_bf16_reduction
        )

    if allow_fp16_reduction is not None:
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = (
            allow_fp16_reduction
        )

    if deterministic_algorithms is not None:
        torch.use_deterministic_algorithms(deterministic_algorithms)

    try:
        if autocast_setting is None:
            return callback()

        device_type, dtype = autocast_setting

        with torch.autocast(device_type=device_type, dtype=dtype):
            return callback()
    finally:
        torch.set_float32_matmul_precision(previous_matmul_precision)
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = (
            previous_allow_bf16_reduction
        )
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = (
            previous_allow_fp16_reduction
        )
        torch.use_deterministic_algorithms(previous_deterministic_algorithms)


def _run_with_buffer_mutation_check(
    execution: runtime_values.StandardExecution,
    callback: CandidateOperation,
) -> TensorTree:
    mode = execution.candidate.settings.get("call.buffer_mutation")

    if mode is None:
        return callback()

    if mode == "declared_and_restored":
        return _run_with_declared_state_restore(execution, callback)

    if mode != "forbidden":
        message = f"call.buffer_mutation is unsupported: {mode}"
        raise MaterializationError(message)

    before = runtime_values.buffer_snapshot(execution.buffers)
    result = callback()
    runtime_values.require_buffers_unchanged(before, execution.buffers)

    return result


def _run_with_declared_state_restore(
    execution: runtime_values.StandardExecution,
    callback: CandidateOperation,
) -> TensorTree:
    settings = execution.candidate.settings
    parameter_snapshot = runtime_values.declared_tensor_snapshot(
        execution.params,
        settings["mutated_parameter_keys"],
        "parameter",
    )
    buffer_snapshot = runtime_values.declared_tensor_snapshot(
        execution.buffers,
        settings["mutated_buffer_keys"],
        "buffer",
    )

    try:
        return callback()
    finally:
        runtime_values.restore_declared_tensors(execution.params, parameter_snapshot)
        runtime_values.restore_declared_tensors(execution.buffers, buffer_snapshot)


def _anchor_candidate(operator: OperatorSpec, candidate: Candidate) -> Candidate:
    path = _anchor_path(operator)
    settings = anchor_settings(operator, candidate, path)

    return dataclasses.replace(
        candidate,
        settings=settings,
    )


def anchor_settings(
    operator: OperatorSpec,
    candidate: Candidate,
    path: str,
) -> dict[str, Any]:
    """Return the anchor settings for a reference check.

    Returns:
        The anchor settings for a reference check.

    Raises:
        MaterializationError: If the declared inputs are invalid.
    """
    settings = dict(candidate.settings)

    for key in (
        *runtime_values.RUNTIME_DTYPE_SETTINGS,
        *runtime_values.BACKEND_SETTINGS,
        *runtime_values.SPEC_PATH_KEYS.values(),
        *runtime_values.SPEC_ADDITIONAL_RUNTIME_SETTINGS,
        *FUNCTIONAL_CALL_FIELDS,
        *TORCH_FUNC_FIELDS,
        *runtime_values.LOSS_SCALING_SETTINGS,
        "vectorization.vmap_chunk_size",
        "vectorization.in_dims",
    ):
        settings.pop(key, None)

    spec_value = runtime_values.spec_path_value_for_runtime_path(operator.kind, path)

    if spec_value is None:
        if operator.kind != "ggnvp" or path != runtime_values.GGN_DENSE_PATH:
            message = f"anchor path has no SPEC mapping: {path}"
            raise MaterializationError(message)
    else:
        settings[runtime_values.SPEC_PATH_KEYS[operator.kind]] = spec_value

    settings.update(fisher.fisher_anchor_settings(operator, path))
    settings.update(fisher.sampled_fisher_anchor_settings(operator, candidate, path))
    settings.update(derivatives.per_example_gradient_anchor_settings(operator, path))
    settings.update(ggn.ggn_anchor_settings(operator, path))
    settings.update(metrics.metric_inner_anchor_settings(operator, candidate, path))
    settings.update(
        metrics.inverse_metric_inner_anchor_settings(operator, candidate, path)
    )

    settings.update(_anchor_admission_settings(path))

    return settings


def _anchor_admission_settings(path: str) -> dict[str, Any]:
    if path == runtime_values.JVP_PATH:
        return _torch_func_anchor_settings(requires_forward_ad=True)

    if path == runtime_values.VJP_PATH:
        return _torch_func_anchor_settings(requires_forward_ad=False)

    if path == runtime_values.HVP_JVP_GRAD_PATH:
        return _torch_func_anchor_settings(requires_forward_ad=True)

    if path == runtime_values.GGN_JVP_HESSIAN_VJP_PATH:
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
        "vectorization.randomness": "error",
        "requires_forward_ad": requires_forward_ad,
        "forward_ad_supported": True,
    }


def _anchor_path(operator: OperatorSpec) -> str:
    if operator.kind == "fisher_vp":
        return fisher.fisher_anchor_path(operator)

    path = runtime_values.STANDARD_ANCHOR_PATHS.get(operator.kind)

    if path is not None:
        return path

    message = f"standard anchor does not support operator kind: {operator.kind}"
    raise MaterializationError(message)


def call_function_objective(
    execution: runtime_values.StandardExecution,
    function: FunctionObjective,
    params: ParameterTree,
    batch: Batch | None = None,
) -> TensorTree:
    """Call the declared function objective for an execution.

    Returns:
        The the declared function objective for an execution.
    """
    active_batch = execution.batch if batch is None else batch
    settings = execution.candidate.settings
    output = function(
        model_compute_tree(params, settings),
        model_compute_tree(execution.buffers, settings),
        model_compute_batch(active_batch, settings),
        execution.context,
    )

    return runtime_values.checked_function_output(
        execution.candidate.settings,
        output,
        "function objective output",
    )
