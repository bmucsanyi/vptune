"""Runtime builders for package-owned operator anchors."""

import dataclasses
import importlib
import inspect
import math
from collections.abc import Callable, Iterator, Mapping, Sequence
from itertools import starmap
from typing import Any

import torch
from torch.utils.checkpoint import checkpoint, noop_context_fn

from vptune import ggn, metrics, runtime_values
from vptune.admission import (
    FUNCTIONAL_CALL_FIELDS,
    TORCH_FUNC_FIELDS,
    admit_checkpoint,
    admit_torch_func,
)
from vptune.anchors import (
    finite_difference_hvp,
    finite_difference_jvp,
    forward_ad_jvp_anchor,
    gradient_anchor,
    hvp_anchor,
    hvp_jvp_grad_anchor,
    hvp_reverse_over_reverse_anchor,
    jvp_anchor,
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
    ReferenceChildResult,
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
    tree_from_leaves,
    tree_leaves,
    tree_map,
    tree_map2,
    tree_mul_foreach,
    tree_signature,
)

FISHER_PER_EXAMPLE_MANUAL_BATCH_PATHS = (
    runtime_values.FISHER_MANUAL_BATCH_PATHS_BY_KIND["fisher_vp"]
)
FISHER_SCORE_GRADIENT_PRODUCT_PATHS = (
    runtime_values.FISHER_STREAMING_PRODUCT_PATHS_BY_KIND["fisher_vp"]
)
SAMPLED_FISHER_PER_EXAMPLE_MANUAL_BATCH_PATHS = (
    runtime_values.FISHER_MANUAL_BATCH_PATHS_BY_KIND["sampled_fisher_vp"]
)
SAMPLED_FISHER_SCORE_GRADIENT_PRODUCT_PATHS = (
    runtime_values.FISHER_STREAMING_PRODUCT_PATHS_BY_KIND["sampled_fisher_vp"]
)
EMPIRICAL_FISHER_GRADIENT_PRODUCT_PATHS = (
    runtime_values.FISHER_STREAMING_PRODUCT_PATHS_BY_KIND["empirical_fisher_vp"]
)


FISHER_VECTOR_VMAP_PATHS = runtime_values.FISHER_VECTOR_VMAP_PATHS_BY_KIND["fisher_vp"]
SAMPLED_FISHER_VECTOR_VMAP_PATHS = runtime_values.FISHER_VECTOR_VMAP_PATHS_BY_KIND[
    "sampled_fisher_vp"
]
EMPIRICAL_FISHER_VECTOR_VMAP_PATHS = runtime_values.FISHER_VECTOR_VMAP_PATHS_BY_KIND[
    "empirical_fisher_vp"
]
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


def composition_operation_factory(
    operator: OperatorSpec,
    *,
    components: Mapping[str, Callable[[Batch, TensorTree], TensorTree]],
    fused_components: Mapping[
        tuple[str, ...],
        Callable[[Batch, TensorTree], TensorTree],
    ]
    | None = None,
    children: Sequence[runtime_values.CompositionChild] = (),
) -> OperationFactory:
    """Return an operation factory for sequential operator composition."""
    expression = _operator_composition_expression(operator)
    sequential_order = _composition_expression_sequential_order(expression)
    order_tuple = (
        sequential_order
        if sequential_order is not None
        else _operator_composition_order(operator)
    )
    component_map = dict(components)
    fused_component_map = {} if fused_components is None else dict(fused_components)
    child_map = _composition_child_map(order_tuple, children)
    _require_composition_components(order_tuple, component_map)
    expression = None if sequential_order is not None else expression

    def factory(
        candidate: Candidate,
        batch: Batch,
        vector: TensorTree,
    ) -> CandidateOperation:
        runtime_values.require_candidate_family(operator, candidate)
        _require_supported_standard_settings(operator, candidate)
        runtime_values.require_path(
            operator.kind,
            runtime_path(operator, candidate),
            (runtime_values.COMPOSITION_PATH,),
        )
        executable_components = _composition_execution_components(
            candidate.settings,
            order_tuple,
            component_map,
            child_map,
        )
        fused_component = _fused_composition_component(
            candidate.settings,
            order_tuple,
            fused_component_map,
        )
        _require_loss_scaling_settings(operator, candidate.settings)
        _require_composition_execution_settings(candidate.settings)
        executable_components = _compile_composition_child_components(
            candidate.settings,
            order_tuple,
            executable_components,
            batch,
            vector,
        )
        output_buffer = _composition_output_buffer(candidate.settings, vector)

        def operation() -> TensorTree:
            prepared_batch = runtime_batch(batch, candidate.settings)
            result = runtime_vector(vector, candidate.settings)

            def run_components() -> TensorTree:
                if expression is not None:
                    return _run_composition_expression(
                        candidate.settings,
                        expression,
                        executable_components,
                        prepared_batch,
                        result,
                    )

                return _run_composition_components(
                    candidate.settings,
                    order_tuple,
                    executable_components,
                    fused_component,
                    prepared_batch,
                    result,
                )

            def run_scaled_components() -> TensorTree:
                component_output = run_components()
                scaled_output = _loss_scaled_output_source(
                    operator,
                    candidate.settings,
                    component_output,
                )

                return _loss_unscaled_output(
                    operator,
                    candidate.settings,
                    scaled_output,
                )

            return run_with_backend_settings(
                candidate.settings,
                lambda: runtime_values.run_with_call_grad_mode(
                    candidate.settings,
                    lambda: runtime_values.runtime_output_to_buffer(
                        _runtime_output(
                            run_scaled_components(),
                            candidate.settings,
                        ),
                        output_buffer,
                    ),
                ),
            )

        return _compile_operation(operator, candidate.settings, operation)

    return factory


def composition_reference_check(
    operator: OperatorSpec,
    *,
    components: Mapping[str, Callable[[Batch, TensorTree], TensorTree]],
    anchor_components: Mapping[str, Callable[[Batch, TensorTree], TensorTree]],
    fused_components: Mapping[
        tuple[str, ...],
        Callable[[Batch, TensorTree], TensorTree],
    ]
    | None = None,
    thresholds: Mapping[str, float],
    numeric_bound_fields: Mapping[str, Any] | None = None,
    children: Sequence[runtime_values.CompositionChild] = (),
) -> ReferenceCheck:
    """Return a reference check for sequential operator composition.

    Raises:
        MaterializationError: If thresholds are empty.
    """
    if not thresholds:
        message = "composition reference thresholds are required"
        raise MaterializationError(message)

    expression = _operator_composition_expression(operator)
    sequential_order = _composition_expression_sequential_order(expression)
    order_tuple = (
        sequential_order
        if sequential_order is not None
        else _operator_composition_order(operator)
    )
    component_map = dict(components)
    anchor_component_map = dict(anchor_components)
    fused_component_map = {} if fused_components is None else dict(fused_components)
    bound_fields = {} if numeric_bound_fields is None else dict(numeric_bound_fields)
    child_map = _composition_child_map(order_tuple, children)
    _require_composition_components(order_tuple, component_map)
    _require_composition_components(order_tuple, anchor_component_map)
    expression = None if sequential_order is not None else expression

    def check(
        candidate: Candidate,
        batch: Batch,
        vector: TensorTree,
    ) -> ReferenceResult:
        try:
            anchor_candidate = dataclasses.replace(
                candidate,
                settings=anchor_settings(
                    operator, candidate, runtime_values.COMPOSITION_PATH
                ),
            )
            candidate_components, anchor_components_for_row = (
                _composition_reference_components(
                    candidate.settings,
                    order_tuple,
                    component_map,
                    anchor_component_map,
                    child_map,
                )
            )
            reference_children = _composition_reference_children(
                candidate.settings,
                order_tuple,
                child_map,
            )
            candidate_output, anchor_output, component_errors, child_results = (
                _composition_reference_outputs(
                    operator,
                    candidate,
                    anchor_candidate,
                    batch,
                    vector,
                    order_tuple,
                    candidate_components,
                    anchor_components_for_row,
                    fused_component_map,
                    reference_children,
                    expression,
                )
            )
        except (MaterializationError, ReferenceFailedError) as error:
            raise ReferenceFailedError(str(error)) from error

        measurements = tree_error_measurements(candidate_output, anchor_output)
        runtime_values.merge_component_measurements(measurements, component_errors)
        measurements.update(
            _semantic_measurements(operator, batch, vector, candidate_output)
        )
        effective_thresholds = _reference_thresholds_for_operator(operator, thresholds)
        _require_thresholds_for_measurements(measurements, effective_thresholds)
        validate_thresholds(measurements, effective_thresholds)
        runtime_values.apply_numeric_error_bound(
            measurements,
            effective_thresholds,
            candidate.settings,
            bound_fields,
            anchor_output,
        )

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
    fused_components: Mapping[
        tuple[str, ...],
        Callable[[Batch, TensorTree], TensorTree],
    ],
    children: Mapping[str, runtime_values.CompositionChild],
    expression: Mapping[str, Any] | None,
) -> tuple[
    TensorTree,
    TensorTree,
    dict[str, dict[str, float]],
    tuple[ReferenceChildResult, ...],
]:
    runtime_values.require_candidate_family(operator, candidate)
    runtime_values.require_candidate_family(operator, anchor_candidate)
    _require_supported_standard_settings(operator, candidate)
    _require_supported_standard_settings(operator, anchor_candidate)
    _require_composition_execution_settings(candidate.settings)
    _require_loss_scaling_settings(operator, candidate.settings)
    runtime_values.require_path(
        operator.kind,
        runtime_path(operator, candidate),
        (runtime_values.COMPOSITION_PATH,),
    )
    runtime_values.require_path(
        operator.kind,
        runtime_path(operator, anchor_candidate),
        (runtime_values.COMPOSITION_PATH,),
    )
    candidate_batch = runtime_batch(batch, candidate.settings)
    anchor_batch = runtime_batch(batch, anchor_candidate.settings)
    candidate_result = runtime_vector(vector, candidate.settings)
    anchor_result = runtime_vector(vector, anchor_candidate.settings)
    fused_component = _fused_composition_component(
        candidate.settings,
        order,
        fused_components,
    )
    fused_result = None
    component_errors = {}
    child_results = []

    if expression is not None:
        (
            candidate_result,
            anchor_result,
            component_errors,
            child_results,
        ) = _composition_expression_reference_outputs(
            candidate.settings,
            expression,
            candidate_batch,
            anchor_batch,
            candidate_result,
            anchor_result,
            components,
            anchor_components,
            children,
        )
        candidate_result = _loss_scaled_output_source(
            operator,
            candidate.settings,
            candidate_result,
        )
        candidate_result = _loss_unscaled_output(
            operator,
            candidate.settings,
            candidate_result,
        )
        candidate_result = _runtime_output(candidate_result, candidate.settings)
        anchor_result = _runtime_output(anchor_result, anchor_candidate.settings)

        return candidate_result, anchor_result, component_errors, tuple(child_results)

    if fused_component is not None:
        fused_result = fused_component(candidate_batch, candidate_result)

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
                    component_name,
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

    if fused_result is not None:
        component_errors["fused_composition"] = tree_error_measurements(
            fused_result,
            candidate_result,
        )
        candidate_result = fused_result

    candidate_result = _loss_scaled_output_source(
        operator,
        candidate.settings,
        candidate_result,
    )
    candidate_result = _loss_unscaled_output(
        operator,
        candidate.settings,
        candidate_result,
    )
    candidate_result = _runtime_output(candidate_result, candidate.settings)
    anchor_result = _runtime_output(anchor_result, anchor_candidate.settings)

    return candidate_result, anchor_result, component_errors, tuple(child_results)


def _composition_execution_components(
    settings: Mapping[str, Any],
    order: tuple[str, ...],
    components: Mapping[str, Callable[[Batch, TensorTree], TensorTree]],
    children: Mapping[str, runtime_values.CompositionChild],
) -> Mapping[str, Callable[[Batch, TensorTree], TensorTree]]:
    if settings.get("composition.execution") == "fuse_adjacent_children":
        return components

    mode = _composition_child_evaluation(settings)

    if mode == "inline_child_lowering":
        return components

    child_components = {name: child.component for name, child in children.items()}
    _require_composition_components(order, child_components)

    return child_components


def _fused_composition_component(
    settings: Mapping[str, Any],
    order: tuple[str, ...],
    fused_components: Mapping[
        tuple[str, ...],
        Callable[[Batch, TensorTree], TensorTree],
    ],
) -> Callable[[Batch, TensorTree], TensorTree] | None:
    if settings.get("composition.execution") != "fuse_adjacent_children":
        return None

    component = fused_components.get(order)

    if component is None:
        message = "fuse_adjacent_children requires a fused component for child order"
        raise MaterializationError(message)

    return component


def _operator_composition_expression(
    operator: OperatorSpec,
) -> Mapping[str, Any] | None:
    expression = operator.semantics.get("combine")

    if expression is None:
        return None

    if not isinstance(expression, Mapping):
        message = "composition combine expression must be a mapping"
        raise MaterializationError(message)

    _require_composition_expression(expression)

    return expression


def _composition_expression_sequential_order(
    expression: Mapping[str, Any] | None,
) -> tuple[str, ...] | None:
    if expression is None:
        return None

    kind = _composition_expression_kind(expression)

    if kind == "child":
        return (_composition_expression_child(expression),)

    if kind != "compose":
        return None

    terms = _composition_expression_terms(expression)

    if not all(_composition_expression_kind(term) == "child" for term in terms):
        return None

    return tuple(_composition_expression_child(term) for term in reversed(terms))


def _require_composition_expression(expression: Mapping[str, Any]) -> None:
    kind = _composition_expression_kind(expression)

    if kind in {"child", "source"}:
        _composition_expression_child(expression)

        return

    if kind == "scaled_identity":
        _composition_expression_coefficient(expression)

        return

    if kind == "compose":
        for term in _composition_expression_terms(expression):
            _require_composition_expression(term)

        return

    if kind == "linear_combination":
        for term in metrics.composition_expression_weighted_terms(expression):
            _require_composition_expression(term["term"])

        return

    message = f"composition expression kind is unsupported: {kind}"
    raise MaterializationError(message)


def _composition_expression_kind(expression: Mapping[str, Any]) -> str:
    kind = expression.get("kind")

    if isinstance(kind, str) and kind:
        return kind

    message = "composition expression kind must be a non-empty string"
    raise MaterializationError(message)


def _composition_expression_child(expression: Mapping[str, Any]) -> str:
    child = expression.get("name")

    if child is None:
        child = expression.get("child")

    if isinstance(child, str) and child:
        return child

    message = "composition expression child must be a non-empty string"
    raise MaterializationError(message)


def _composition_expression_coefficient(expression: Mapping[str, Any]) -> float:
    coefficient = expression.get("coefficient")

    if isinstance(coefficient, int | float) and not isinstance(coefficient, bool):
        return float(coefficient)

    message = "composition expression coefficient must be a number"
    raise MaterializationError(message)


def _composition_expression_terms(
    expression: Mapping[str, Any],
) -> tuple[Mapping[str, Any], ...]:
    terms = expression.get("terms")

    if (
        isinstance(terms, Sequence)
        and not isinstance(terms, str)
        and terms
        and all(isinstance(term, Mapping) for term in terms)
    ):
        return tuple(terms)

    message = "composition expression terms must be non-empty mappings"
    raise MaterializationError(message)


def _run_composition_expression(
    settings: Mapping[str, Any],
    expression: Mapping[str, Any],
    components: Mapping[str, Callable[[Batch, TensorTree], TensorTree]],
    batch: Batch,
    vector: TensorTree,
) -> TensorTree:
    if settings.get("composition.execution") == "fuse_adjacent_children":
        message = "fuse_adjacent_children requires a sequential compose expression"
        raise MaterializationError(message)

    result, _, _ = _evaluate_composition_expression(
        settings,
        expression,
        components,
        batch,
        vector,
        children={},
        run_child_checks=False,
    )

    return result


def _composition_expression_reference_outputs(
    settings: Mapping[str, Any],
    expression: Mapping[str, Any],
    candidate_batch: Batch,
    anchor_batch: Batch,
    candidate_vector: TensorTree,
    anchor_vector: TensorTree,
    components: Mapping[str, Callable[[Batch, TensorTree], TensorTree]],
    anchor_components: Mapping[str, Callable[[Batch, TensorTree], TensorTree]],
    children: Mapping[str, runtime_values.CompositionChild],
) -> tuple[
    TensorTree,
    TensorTree,
    dict[str, dict[str, float]],
    tuple[ReferenceChildResult, ...],
]:
    run_child_checks = _composition_validation(settings) == "validate_each_child"
    candidate_result, candidate_outputs, child_results = (
        _evaluate_composition_expression(
            settings,
            expression,
            components,
            candidate_batch,
            candidate_vector,
            children=children,
            run_child_checks=run_child_checks,
        )
    )
    anchor_result, anchor_outputs, _ = _evaluate_composition_expression(
        settings,
        expression,
        anchor_components,
        anchor_batch,
        anchor_vector,
        children={},
        run_child_checks=False,
    )
    component_errors = _composition_expression_component_errors(
        candidate_outputs,
        anchor_outputs,
    )

    return candidate_result, anchor_result, component_errors, child_results


def _evaluate_composition_expression(
    settings: Mapping[str, Any],
    expression: Mapping[str, Any],
    components: Mapping[str, Callable[[Batch, TensorTree], TensorTree]],
    batch: Batch,
    vector: TensorTree,
    *,
    children: Mapping[str, runtime_values.CompositionChild],
    run_child_checks: bool,
) -> tuple[
    TensorTree,
    tuple[tuple[str, TensorTree], ...],
    tuple[ReferenceChildResult, ...],
]:
    leaf_outputs = []
    child_results = []

    def evaluate(term: Mapping[str, Any], current_vector: TensorTree) -> TensorTree:
        kind = _composition_expression_kind(term)

        if kind == "child":
            child = _composition_expression_child(term)
            result = components[child](batch, current_vector)
            leaf_outputs.append((child, result))

            if run_child_checks and child in children:
                child_results.append(
                    _composition_expression_child_result(
                        child,
                        children[child],
                        batch,
                        current_vector,
                    )
                )

            return result

        if kind == "source":
            child = _composition_expression_child(term)
            result = components[child](batch, {})
            leaf_outputs.append((f"source:{child}", result))

            if run_child_checks and child in children:
                child_results.append(
                    _composition_expression_child_result(
                        child,
                        children[child],
                        batch,
                        {},
                    )
                )

            return result

        if kind == "scaled_identity":
            return _tree_scale_runtime(
                settings,
                current_vector,
                _composition_expression_coefficient(term),
            )

        if kind == "compose":
            terms = _composition_expression_terms(term)
            result = current_vector
            execution_terms = tuple(reversed(terms))

            for index, nested in enumerate(execution_terms):
                result = evaluate(nested, result)
                result = _composition_intermediate_residency(
                    settings,
                    result,
                    is_last=index == len(execution_terms) - 1,
                )

            return result

        if kind == "linear_combination":
            result = None

            for weighted in metrics.composition_expression_weighted_terms(term):
                term_output = evaluate(weighted["term"], current_vector)
                scaled = _tree_scale_runtime(
                    settings,
                    term_output,
                    weighted["coefficient"],
                )

                if result is None:
                    result = scaled
                else:
                    result = tree_add_runtime(settings, result, scaled)

            if result is None:
                message = "linear composition produced no terms"
                raise MaterializationError(message)

            return result

        message = f"composition expression kind is unsupported: {kind}"
        raise MaterializationError(message)

    result = evaluate(expression, vector)

    return result, tuple(leaf_outputs), tuple(child_results)


def _composition_expression_child_result(
    name: str,
    child: runtime_values.CompositionChild,
    batch: Batch,
    vector: TensorTree,
) -> ReferenceChildResult:
    child_input_signature = {
        **dict(child.input_signature),
        "component": name,
        "component_input": tree_signature(vector),
    }
    child_result = child.reference_check(child.candidate, batch, vector)

    return ReferenceChildResult(
        name,
        child.candidate,
        child_input_signature,
        child_result,
    )


def _composition_expression_component_errors(
    candidate_outputs: tuple[tuple[str, TensorTree], ...],
    anchor_outputs: tuple[tuple[str, TensorTree], ...],
) -> dict[str, dict[str, float]]:
    if len(candidate_outputs) != len(anchor_outputs):
        message = "composition candidate and anchor component counts differ"
        raise MaterializationError(message)

    errors = {}

    for index, (candidate, anchor) in enumerate(
        zip(candidate_outputs, anchor_outputs, strict=True)
    ):
        candidate_name, candidate_output = candidate
        anchor_name, anchor_output = anchor

        if candidate_name != anchor_name:
            message = "composition candidate and anchor component names differ"
            raise MaterializationError(message)

        errors[f"{index}:{candidate_name}"] = tree_error_measurements(
            candidate_output,
            anchor_output,
        )

    return errors


def _run_composition_components(
    settings: Mapping[str, Any],
    order: tuple[str, ...],
    components: Mapping[str, Callable[[Batch, TensorTree], TensorTree]],
    fused_component: Callable[[Batch, TensorTree], TensorTree] | None,
    batch: Batch,
    vector: TensorTree,
) -> TensorTree:
    def vector_runner(selected_vector: TensorTree) -> TensorTree:
        return _run_composition_single_vector(
            settings,
            order,
            components,
            fused_component,
            batch,
            selected_vector,
        )

    return _run_tensor_tree_by_vectorization_mode(vector, settings, vector_runner)


def _run_composition_single_vector(
    settings: Mapping[str, Any],
    order: tuple[str, ...],
    components: Mapping[str, Callable[[Batch, TensorTree], TensorTree]],
    fused_component: Callable[[Batch, TensorTree], TensorTree] | None,
    batch: Batch,
    vector: TensorTree,
) -> TensorTree:
    if fused_component is not None:
        return fused_component(batch, vector)

    result = vector

    for index, component_name in enumerate(order):
        result = components[component_name](batch, result)
        result = _composition_intermediate_residency(
            settings,
            result,
            is_last=index == len(order) - 1,
        )

    return result


def _composition_intermediate_residency(
    settings: Mapping[str, Any],
    result: TensorTree,
    *,
    is_last: bool,
) -> TensorTree:
    if is_last:
        return result

    residency = settings.get("memory.intermediate_residency")

    if residency is None:
        return result

    return tree_residency(result, residency, "memory.intermediate_residency")


def _compile_composition_child_components(
    settings: Mapping[str, Any],
    order: tuple[str, ...],
    components: Mapping[str, Callable[[Batch, TensorTree], TensorTree]],
    batch: Batch,
    vector: TensorTree,
) -> Mapping[str, Callable[[Batch, TensorTree], TensorTree]]:
    if settings.get("compile.boundary") != "composition_child":
        return components

    if settings.get("composition.execution") == "fuse_adjacent_children":
        message = "compile.boundary=composition_child requires child calls"
        raise CompileSetupError(message)

    _validate_compile_cache_state(settings)
    compiled_components = {}
    warm_result = _composition_child_warm_vector(settings, vector)
    warm_batch = _composition_child_warm_batch(settings, batch)

    for name in order:
        compiled_component = _compiled_composition_component(
            settings,
            components[name],
        )
        compiled_components[name] = compiled_component

        if warm_batch is not None and warm_result is not None:
            warm_result = compiled_component(warm_batch, warm_result)

    return compiled_components


def _composition_child_warm_batch(
    settings: Mapping[str, Any],
    batch: Batch,
) -> Batch | None:
    if settings.get("compile.cache_state") != "warm_cache":
        return None

    return runtime_batch(batch, settings)


def _composition_child_warm_vector(
    settings: Mapping[str, Any],
    vector: TensorTree,
) -> TensorTree | None:
    if settings.get("compile.cache_state") != "warm_cache":
        return None

    warm_vector = runtime_vector(vector, settings)
    mode = settings.get("vectorization.mode")

    if mode == "single_loop":
        vector_in_dims = vector_tree_in_dims(warm_vector, settings)

        return runtime_values.vector_tree_select(warm_vector, vector_in_dims, 0)

    if mode == "manual_batch":
        vector_in_dims = vector_tree_in_dims(warm_vector, settings)
        vector_count = runtime_values.vector_tree_batch_size(
            warm_vector, vector_in_dims
        )
        batch_size = runtime_values.manual_vector_batch_size(settings)

        return runtime_values.vector_tree_slice(
            warm_vector,
            vector_in_dims,
            0,
            min(batch_size, vector_count),
        )

    return warm_vector


def _compiled_composition_component(
    settings: Mapping[str, Any],
    component: Callable[[Batch, TensorTree], TensorTree],
) -> Callable[[Batch, TensorTree], TensorTree]:
    return compiled_callable(settings, component, use_backend_settings=False)


def _composition_reference_components(
    settings: Mapping[str, Any],
    order: tuple[str, ...],
    components: Mapping[str, Callable[[Batch, TensorTree], TensorTree]],
    anchor_components: Mapping[str, Callable[[Batch, TensorTree], TensorTree]],
    children: Mapping[str, runtime_values.CompositionChild],
) -> tuple[
    Mapping[str, Callable[[Batch, TensorTree], TensorTree]],
    Mapping[str, Callable[[Batch, TensorTree], TensorTree]],
]:
    mode = _composition_child_evaluation(settings)

    if mode == "inline_child_lowering":
        return components, anchor_components

    child_components = {name: child.component for name, child in children.items()}
    child_anchor_components = {
        name: child.anchor_component for name, child in children.items()
    }
    _require_composition_components(order, child_components)
    _require_composition_components(order, child_anchor_components)

    return child_components, child_anchor_components


def _composition_reference_children(
    settings: Mapping[str, Any],
    order: tuple[str, ...],
    children: Mapping[str, runtime_values.CompositionChild],
) -> Mapping[str, runtime_values.CompositionChild]:
    validation = _composition_validation(settings)

    if validation == "validate_composed_output":
        return {}

    _require_composition_children(order, children)

    return children


def _require_composition_execution_settings(settings: Mapping[str, Any]) -> None:
    execution = settings.get("composition.execution")

    if (
        execution == "materialize_each_child"
        and settings.get("composition.child_evaluation") != "selected_child_rows"
    ):
        message = (
            "materialize_each_child requires "
            "composition.child_evaluation=selected_child_rows"
        )
        raise MaterializationError(message)

    if execution != "compile_whole_composition":
        return

    if settings.get("compile.enabled") != "true":
        message = "compile_whole_composition requires compile.enabled=true"
        raise MaterializationError(message)


def _composition_child_evaluation(settings: Mapping[str, Any]) -> str:
    value = settings.get("composition.child_evaluation")

    if value not in {"selected_child_rows", "inline_child_lowering"}:
        message = "composition.child_evaluation is invalid"
        raise MaterializationError(message)

    return value


def _composition_validation(settings: Mapping[str, Any]) -> str:
    value = settings.get("composition.validation")

    if value not in {"validate_each_child", "validate_composed_output"}:
        message = "composition.validation is invalid"
        raise MaterializationError(message)

    return value


def _composition_child_map(
    order: tuple[str, ...],
    children: Sequence[runtime_values.CompositionChild],
) -> dict[str, runtime_values.CompositionChild]:
    child_map = {child.name: child for child in children}

    if len(child_map) != len(children):
        message = "composition child names must be unique"
        raise MaterializationError(message)

    unknown = tuple(name for name in child_map if name not in order)

    if unknown:
        message = f"composition child names are not in the order: {unknown}"
        raise MaterializationError(message)

    return child_map


def _require_composition_children(
    order: tuple[str, ...],
    children: Mapping[str, runtime_values.CompositionChild],
) -> None:
    if set(children) != set(order):
        message = "composition children must match composition order"
        raise MaterializationError(message)


def _require_thresholds_for_measurements(
    measurements: Mapping[str, Any],
    thresholds: Mapping[str, float],
) -> None:
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


def _require_vhp_reference_policy(
    candidate: Candidate,
    batch: Batch,
    thresholds: Mapping[str, float],
) -> None:
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


def _semantic_measurements(
    operator: OperatorSpec,
    batch: Batch,
    vector: TensorTree,
    output: TensorTree,
) -> dict[str, float]:
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
        directional = layout_aware_tree_dot(
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
        errors = layout_aware_tree_error_measurements(
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
    parameter_surface: ParameterSurface | None,
) -> dict[str, float]:
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
    errors = layout_aware_tree_error_measurements(
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
    left = layout_aware_tree_dot(
        candidate.settings,
        runtime_vector(
            symmetry_vector,
            candidate.settings,
            candidate_output,
            parameter_surface,
        ),
        candidate_output,
    )
    right = layout_aware_tree_dot(
        candidate.settings,
        runtime_vector(
            vector,
            candidate.settings,
            anchor_symmetry,
            parameter_surface,
        ),
        anchor_symmetry,
    )
    measurements["symmetry_max_abs_diff"] = float((left - right).abs().item())

    return measurements


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


def composition_runtime_config(
    operator: OperatorSpec,
    *,
    components: Mapping[str, Callable[[Batch, TensorTree], TensorTree]],
    anchor_components: Mapping[str, Callable[[Batch, TensorTree], TensorTree]],
    fused_components: Mapping[
        tuple[str, ...],
        Callable[[Batch, TensorTree], TensorTree],
    ]
    | None = None,
    children: Sequence[runtime_values.CompositionChild] = (),
    candidates: Sequence[Candidate],
    thresholds: Mapping[str, float],
    component_signature: Mapping[str, Any],
    anchor_component_signature: Mapping[str, Any],
    axis_registry: CandidateAdmitter | None,
    numeric_bound_fields: Mapping[str, Any] | None = None,
) -> RuntimeConfig:
    """Return runtime config for sequential operator composition."""
    bound_fields = {} if numeric_bound_fields is None else dict(numeric_bound_fields)
    operation_factory = composition_operation_factory(
        operator,
        components=components,
        fused_components=fused_components,
        children=children,
    )
    reference_check = composition_reference_check(
        operator,
        components=components,
        anchor_components=anchor_components,
        fused_components=fused_components,
        thresholds=thresholds,
        numeric_bound_fields=bound_fields,
        children=children,
    )
    fused_component_keys = (
        ()
        if fused_components is None
        else tuple(tuple(key) for key in sorted(fused_components))
    )
    runtime_signature = {
        "runtime": "composition",
        "operator": operator.signature(),
        "order": _operator_composition_order(operator),
        "thresholds": dict(thresholds),
        "numeric_bound_fields": bound_fields,
        "components": {
            "candidate": dict(component_signature),
            "anchor": dict(anchor_component_signature),
        },
        "fused_components": fused_component_keys,
        "children": tuple(child.name for child in children),
    }
    operation_factory = CallableOperationFactory(
        "vptune.composition_operation_factory",
        PACKAGE_VERSION,
        runtime_signature,
        {"callback": "vptune.runtime.composition_operation_factory"},
        operation_factory,
    )
    reference_check = CallableReferenceCheck(
        "vptune.composition_reference_check",
        PACKAGE_VERSION,
        runtime_signature,
        {"callback": "vptune.runtime.composition_reference_check"},
        reference_check,
    )

    return RuntimeConfig(
        candidates=tuple(candidates),
        operation_factory=operation_factory,
        reference_check=reference_check,
        materializer=_standard_materializer(operation_factory),
        axis_registry=axis_registry,
        reference_check_name="composition_anchor",
        signature=runtime_signature,
    )


def _execution_with_vector(
    execution: runtime_values.StandardExecution,
    vector: TensorTree,
    **changes: Any,
) -> runtime_values.StandardExecution:
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
        _require_supported_standard_settings(
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
                        _runtime_output(
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

        return _compile_operation(
            operator,
            candidate.settings,
            activated_operation,
        )

    return factory


def _prepare_standard_execution(
    execution: runtime_values.StandardExecution,
) -> runtime_values.StandardExecution:
    if execution.operator.kind == "gradient":
        return _prepare_gradient_execution(execution)

    if execution.operator.kind == "jvp":
        return _prepare_jvp_execution(execution)

    if execution.operator.kind == "vjp":
        return _prepare_vjp_execution(execution)

    if execution.operator.kind == "hvp":
        return _prepare_hvp_execution(execution)

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
            flat_parameter_vector_batch=_build_flat_vector_batch(execution),
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
        ("gradient", "gradient_closure"): lambda: _run_gradient_by_path(execution),
        ("jvp", "jvp_closure"): lambda: _run_jvp_by_path(execution),
        ("vjp", "vjp_closure"): lambda: _run_vjp_by_path(execution),
        ("hvp", "hvp_single_vector"): lambda: _run_hvp_single_vector(execution),
        ("hvp", "hvp_batched_vectors"): lambda: _run_hvp_by_path(execution),
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

    score_builder = _score_matrix_compile_boundary_builder(execution, boundary)

    if score_builder is not None:
        return _prepare_score_matrix_compile_boundary(
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
        vector_execution = _execution_with_vector(step_execution, vector)

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
    scalar_function = _hvp_scalar_function(execution)
    compiled_scalar_function = _compiled_scalar_function(
        settings,
        scalar_function,
        execution.params,
    )

    return dataclasses.replace(
        execution,
        compiled_scalar_function=compiled_scalar_function,
    )


def _prepare_score_matrix_compile_boundary(
    execution: runtime_values.StandardExecution,
    settings: Mapping[str, Any],
    builder: Callable[[], torch.Tensor],
) -> runtime_values.StandardExecution:
    require_compiled_execution(execution, settings)
    compiled_score_matrix = _compiled_tensor_operation(settings, builder)

    return dataclasses.replace(
        execution,
        compiled_score_matrix=compiled_score_matrix,
    )


def _score_matrix_compile_boundary_builder(
    execution: runtime_values.StandardExecution,
    boundary: str,
) -> Callable[[], torch.Tensor] | None:
    row = _score_matrix_compile_row(execution.operator.kind, boundary)

    if row is None:
        return None

    return lambda: _score_gradient_matrix_from_row(
        execution,
        row,
        use_compiled=False,
    )


def _score_matrix_compile_row(
    operator_kind: str,
    boundary: object,
) -> runtime_values.ScoreMatrixCompileRow | None:
    row = runtime_values.SCORE_MATRIX_COMPILE_ROWS.get(operator_kind)

    if row is None or boundary != row.boundary:
        return None

    return row


def _prepare_gradient_execution(
    execution: runtime_values.StandardExecution,
) -> runtime_values.StandardExecution:
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


def _prepare_jvp_execution(
    execution: runtime_values.StandardExecution,
) -> runtime_values.StandardExecution:
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
        _jvp_tensor_function(execution),
        execution.params,
    )

    return dataclasses.replace(execution, linearized_jvp=jvp_function)


def _prepare_vjp_execution(
    execution: runtime_values.StandardExecution,
) -> runtime_values.StandardExecution:
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
        _vjp_tensor_function(execution),
        execution.params,
    )

    def closure(cotangent: TensorTree) -> TensorTree:
        (result,) = pullback(cotangent)

        return result

    return dataclasses.replace(execution, vjp_closure=closure)


def _prepare_hvp_execution(
    execution: runtime_values.StandardExecution,
) -> runtime_values.StandardExecution:
    reuse = execution.candidate.settings.get("hvp.gradient_reuse")

    if reuse is None or reuse == "recompute_gradient":
        return execution

    if reuse != "reuse_gradient_closure":
        message = f"hvp.gradient_reuse is unsupported: {reuse}"
        raise MaterializationError(message)

    if execution.path != runtime_values.HVP_LINEARIZE_GRAD_PATH:
        message = "reuse_gradient_closure requires linearize_grad"
        raise MaterializationError(message)

    gradient_function = torch.func.grad(_hvp_scalar_function(execution))
    _, hvp_function = torch.func.linearize(gradient_function, execution.params)

    return dataclasses.replace(execution, linearized_hvp=hvp_function)


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
        batch=_runtime_batch_input_residency(
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


def _compile_operation(
    operator: OperatorSpec,
    settings: Mapping[str, Any],
    operation: CandidateOperation,
) -> CandidateOperation:
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


def _validate_compile_cache_state(settings: Mapping[str, Any]) -> None:
    if settings.get("compile.cache_state") in {"cold_compile", "warm_cache"}:
        return

    message = "compile.cache_state must be cold_compile or warm_cache"
    raise CompileSetupError(message)


def _compiled_tensor_operation(
    settings: Mapping[str, Any],
    operation: Callable[[], torch.Tensor],
) -> Callable[[], torch.Tensor]:
    compiled = compiled_callable(
        settings,
        operation,
        use_backend_settings=False,
    )
    warm_compiled_cache(settings, compiled)

    return compiled


def _compiled_model_forward(
    settings: Mapping[str, Any],
    operation: Callable[[Batch], object],
) -> Callable[[Batch], object]:
    compiled_function = compiled_callable(
        settings,
        operation,
        use_backend_settings=False,
    )
    _validate_compile_cache_state(settings)

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
    _validate_compile_cache_state(settings)

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
        return _score_matrix_compile_boundary_supported(
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


def _score_matrix_compile_boundary_supported(
    operator_kind: str,
    boundary: str,
    settings: Mapping[str, Any],
) -> bool:
    row = runtime_values.SCORE_MATRIX_COMPILE_ROWS[operator_kind]

    if boundary != row.boundary:
        return False

    if (
        operator_kind == "per_example_gradient"
        and settings.get("per_example_gradient.accumulation") != "stacked_leading_axis"
    ):
        return False

    return settings.get(row.path_key) in runtime_values.SCORE_MATRIX_COMPILE_PATH_VALUES


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

        effective_thresholds = _reference_thresholds_for_operator(operator, thresholds)
        _require_thresholds_for_measurements(measurements, effective_thresholds)
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


def _reference_thresholds_for_operator(
    operator: OperatorSpec,
    thresholds: Mapping[str, float],
) -> dict[str, float]:
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
    _require_vhp_reference_policy(candidate, batch, thresholds)
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
        return _fisher_batch_inputs(operator, path)

    if operator.kind == "sampled_fisher_vp":
        return _sampled_fisher_batch_inputs(operator, path)

    if operator.kind == "empirical_fisher_vp":
        return _empirical_fisher_batch_inputs(operator, path)

    return ()


def _fisher_batch_inputs(operator: OperatorSpec, path: str) -> tuple[str, ...]:
    if path not in {
        runtime_values.FISHER_DENSE_PATH,
        runtime_values.FISHER_SCORE_GRADIENT_LOOP_PATH,
    }:
        return ()

    inputs = ()

    if path == runtime_values.FISHER_DENSE_PATH:
        inputs = ("score_gradients",)

    return (*inputs, *_fisher_denominator_batch_inputs(operator))


def _sampled_fisher_batch_inputs(operator: OperatorSpec, path: str) -> tuple[str, ...]:
    if path not in {
        runtime_values.SAMPLED_FISHER_DENSE_PATH,
        runtime_values.SAMPLED_FISHER_SCORE_GRADIENT_LOOP_PATH,
        runtime_values.SAMPLED_FISHER_SCORE_GRADIENT_VMAP_PATH,
    }:
        return ()

    inputs = ()

    if path == runtime_values.SAMPLED_FISHER_DENSE_PATH:
        inputs = ("sampled_score_gradients",)

    return (*inputs, *_fisher_denominator_batch_inputs(operator))


def _fisher_denominator_batch_inputs(operator: OperatorSpec) -> tuple[str, ...]:
    denominator = runtime_values.operator_semantic(operator, "denominator")

    if denominator == "batch_normalization":
        return ("normalization",)

    if denominator == "num_examples":
        return ("num_examples",)

    return ()


def _empirical_fisher_batch_inputs(
    operator: OperatorSpec,
    path: str,
) -> tuple[str, ...]:
    if path not in {
        runtime_values.EMPIRICAL_FISHER_DENSE_PATH,
        runtime_values.EMPIRICAL_FISHER_GRADIENT_LOOP_PATH,
        runtime_values.EMPIRICAL_FISHER_GRADIENT_VMAP_PATH,
    }:
        return ()

    inputs = ()

    if path == runtime_values.EMPIRICAL_FISHER_DENSE_PATH:
        inputs = ("per_example_gradients",)

    if (
        runtime_values.operator_semantic(operator, "denominator")
        == "batch_normalization"
    ):
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
    parameter_surface: ParameterSurface | None,
) -> dict[str, Any]:
    measurements = layout_aware_tree_error_measurements(
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
        _semantic_measurements(operator, batch, vector, candidate_output)
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
            parameter_surface=parameter_surface,
        )
    )

    return measurements


def layout_aware_tree_error_measurements(
    candidate: Candidate,
    candidate_output: TensorTree,
    anchor_output: TensorTree,
) -> dict[str, float]:
    """Return layout-aware tree error measurements.

    Returns:
        The layout-aware tree error measurements.
    """
    if candidate.settings.get("layout.output") != "flat_contiguous":
        return tree_error_measurements(candidate_output, anchor_output)

    return tree_error_measurements(
        runtime_values.flatten_vector(candidate_output),
        runtime_values.flatten_vector(anchor_output),
    )


def layout_aware_tree_dot(
    settings: Mapping[str, Any],
    left: TensorTree,
    right: TensorTree,
) -> torch.Tensor:
    """Return the layout-aware dot product of two trees.

    Returns:
        The layout-aware dot product of two trees.
    """
    if settings.get("layout.output") != "flat_contiguous":
        return tree_dot_runtime(settings, left, right)

    return dot_runtime(
        settings,
        runtime_values.flatten_vector(left),
        runtime_values.flatten_vector(right),
    )


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
    materializer = _standard_materializer(
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

    _require_loss_scaling_settings(execution.operator, execution.candidate.settings)
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
    result = _loss_scaled_output_source(
        execution.operator,
        execution.candidate.settings,
        result,
    )

    return _loss_unscaled_output(
        execution.operator,
        execution.candidate.settings,
        result,
    )


def _loss_scaled_execution(
    execution: runtime_values.StandardExecution,
) -> runtime_values.StandardExecution:
    scale = _loss_scale(execution.candidate.settings)

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


def _loss_scaled_output_source(
    operator: OperatorSpec,
    settings: Mapping[str, Any],
    result: TensorTree,
) -> TensorTree:
    scale = _loss_scale(settings)

    if scale is None:
        return result

    if operator.kind in {"metric", "inverse_metric", "composition"}:
        return _tree_scale_runtime(settings, result, scale)

    return result


def _loss_unscaled_output(
    operator: OperatorSpec,
    settings: Mapping[str, Any],
    result: TensorTree,
) -> TensorTree:
    scale = _loss_scale(settings)

    if scale is None:
        return result

    degree = _loss_unscale_degree(operator, settings)

    return _tree_scale_runtime(settings, result, 1.0 / (scale**degree))


def _require_loss_scaling_settings(
    operator: OperatorSpec,
    settings: Mapping[str, Any],
) -> None:
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

    _loss_scale(settings)
    _loss_unscale_degree(operator, settings)


def _loss_scale(settings: Mapping[str, Any]) -> float | None:
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


def _operator_composition_order(operator: OperatorSpec) -> tuple[str, ...]:
    if operator.kind != "composition":
        message = f"operator is not a composition: {operator.kind}"
        raise MaterializationError(message)

    children = operator.semantics.get("children")

    if not isinstance(children, Sequence) or isinstance(children, str):
        message = "composition operator must declare ordered children"
        raise MaterializationError(message)

    return _composition_order(children)


def _require_composition_components(
    order: tuple[str, ...],
    components: Mapping[str, Callable[[Batch, TensorTree], TensorTree]],
) -> None:
    if set(components) != set(order):
        message = "composition components must match composition order"
        raise MaterializationError(message)


def _run_gradient(execution: runtime_values.StandardExecution) -> TensorTree:
    if execution.compiled_inner is not None:
        return execution.compiled_inner()

    return _run_gradient_by_path(execution)


def _run_gradient_by_path(execution: runtime_values.StandardExecution) -> TensorTree:
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
    scalar_function = _hvp_scalar_function(execution)

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

    batch, batch_in_dims = _microbatch_in_dims(execution.batch)
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


def _microbatch_in_dims(batch: Batch) -> tuple[dict[str, Any], dict[str, int | None]]:
    return _per_example_batch_in_dims(batch, "microbatch accumulation")


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
    scalar_function = _hvp_scalar_function(execution)
    active_params = runtime_values.grad_enabled_params(execution.params)
    value = scalar_function(active_params)
    value.backward()
    result = runtime_values.parameter_grad_tree(active_params)
    runtime_values.require_finite_tree(result, "gradient result")

    return result


def _run_jvp(execution: runtime_values.StandardExecution) -> TensorTree:
    if execution.compiled_inner is not None:
        return execution.compiled_inner()

    return _run_jvp_by_path(execution)


def run_by_vectorization_mode(
    execution: runtime_values.StandardExecution,
    *,
    single_vector: Callable[[runtime_values.StandardExecution], TensorTree],
    single_loop: Callable[[runtime_values.StandardExecution], TensorTree],
    manual_batch: Callable[[runtime_values.StandardExecution], TensorTree],
    vmap: Callable[[runtime_values.StandardExecution], TensorTree],
) -> TensorTree:
    """Run the declared vectorization mode with the supplied runners.

    Returns:
        Run the declared vectorization mode with the supplied runners.
    """
    mode = execution.candidate.settings.get("vectorization.mode")

    if mode == "single_loop":
        return single_loop(execution)

    if mode == "manual_batch":
        return manual_batch(execution)

    if mode == "vmap":
        return vmap(execution)

    return single_vector(execution)


def run_single_vectorized_by_path(
    execution: runtime_values.StandardExecution,
    paths: tuple[str, ...],
    single_vector: Callable[[runtime_values.StandardExecution], TensorTree],
    single_loop_vector: Callable[[runtime_values.StandardExecution], TensorTree],
    vmap: Callable[[runtime_values.StandardExecution], TensorTree],
) -> TensorTree:
    """Run a single vectorized call for the declared path.

    Returns:
        The a single vectorized call for the declared path.
    """
    runtime_values.require_path(execution.operator.kind, execution.path, paths)

    def single_loop(loop_execution: runtime_values.StandardExecution) -> TensorTree:
        return run_vector_single_loop(loop_execution, single_loop_vector)

    def manual_batch(batch_execution: runtime_values.StandardExecution) -> TensorTree:
        return run_vector_manual_batches(batch_execution, single_loop)

    return run_by_vectorization_mode(
        execution,
        single_vector=single_vector,
        single_loop=single_loop,
        manual_batch=manual_batch,
        vmap=vmap,
    )


def _run_jvp_by_path(execution: runtime_values.StandardExecution) -> TensorTree:
    return run_single_vectorized_by_path(
        execution,
        (
            runtime_values.JVP_PATH,
            runtime_values.JVP_FORWARD_AD_PATH,
            runtime_values.JVP_LINEARIZE_PATH,
        ),
        _run_jvp_single_vector,
        _run_jvp_single_vector,
        _run_jvp_vector_vmap,
    )


def _run_jvp_single_vector(execution: runtime_values.StandardExecution) -> TensorTree:
    tensor_function = _jvp_tensor_function(execution)

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


def _run_jvp_vector_vmap(execution: runtime_values.StandardExecution) -> TensorTree:
    if execution.path not in runtime_values.JVP_VECTOR_VMAP_PATHS:
        message = "vectorization.mode=vmap requires a torch.func JVP path"
        raise MaterializationError(message)

    if execution.path == runtime_values.JVP_LINEARIZE_PATH:
        if execution.linearized_jvp is not None:
            jvp_function = execution.linearized_jvp
        else:
            _, jvp_function = torch.func.linearize(
                _jvp_tensor_function(execution),
                execution.params,
            )

        return run_vector_vmap(execution, jvp_function)

    tensor_function = _jvp_tensor_function(execution)

    def jvp_function(vector: TensorTree) -> TensorTree:
        return jvp_anchor(
            tensor_function,
            execution.params,
            vector,
        )

    return run_vector_vmap(execution, jvp_function)


def _jvp_tensor_function(
    execution: runtime_values.StandardExecution,
) -> Callable[[ParameterTree], TensorTree]:
    function = runtime_values.function_objective(
        execution.operator, execution.function_objectives
    )

    def tensor_function(active_params: ParameterTree) -> TensorTree:
        return call_function_objective(execution, function, active_params)

    return tensor_function


def _run_vjp(execution: runtime_values.StandardExecution) -> TensorTree:
    if execution.compiled_inner is not None:
        return execution.compiled_inner()

    return _run_vjp_by_path(execution)


def _run_vjp_by_path(execution: runtime_values.StandardExecution) -> TensorTree:
    return run_single_vectorized_by_path(
        execution,
        (
            runtime_values.VJP_PATH,
            runtime_values.VJP_AUTOGRAD_OUTPUTS_PATH,
            runtime_values.VJP_BACKWARD_MATERIALIZED_PATH,
        ),
        _run_vjp_single_vector,
        _run_vjp_single_vector,
        _run_vjp_vector_vmap,
    )


def _run_vjp_single_vector(execution: runtime_values.StandardExecution) -> TensorTree:
    tensor_function = _vjp_tensor_function(execution)

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


def _run_vjp_vector_vmap(execution: runtime_values.StandardExecution) -> TensorTree:
    if execution.path not in runtime_values.VJP_VECTOR_VMAP_PATHS:
        message = "vectorization.mode=vmap requires torch_func_vjp"
        raise MaterializationError(message)

    closure = execution.vjp_closure

    if closure is not None:

        def vjp_function(vector: TensorTree) -> TensorTree:
            return closure(vector)

    else:
        pullback = vjp_pullback(
            _vjp_tensor_function(execution),
            execution.params,
        )

        def vjp_function(vector: TensorTree) -> TensorTree:
            (result,) = pullback(vector)

            return result

    return run_vector_vmap(execution, vjp_function)


def _vjp_tensor_function(
    execution: runtime_values.StandardExecution,
) -> Callable[[ParameterTree], TensorTree]:
    if _uses_stateful_module_call(execution):
        return _stateful_module_tensor_function(execution)

    function = runtime_values.function_objective(
        execution.operator, execution.function_objectives
    )

    def tensor_function(active_params: ParameterTree) -> TensorTree:
        return call_function_objective(execution, function, active_params)

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


def _run_hvp(execution: runtime_values.StandardExecution) -> TensorTree:
    if execution.compiled_inner is not None:
        return execution.compiled_inner()

    return _run_hvp_by_path(execution)


def _run_hvp_by_path(execution: runtime_values.StandardExecution) -> TensorTree:
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

    return run_by_vectorization_mode(
        execution,
        single_vector=_run_hvp_single_vector,
        single_loop=_run_hvp_vector_single_loop,
        manual_batch=_run_hvp_vector_manual_batch,
        vmap=_run_hvp_vector_vmap,
    )


def _run_hvp_single_vector(execution: runtime_values.StandardExecution) -> TensorTree:
    scalar_function = _hvp_scalar_function(execution)

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
    vector_tensor = parameter_order_vector(execution)
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
            result[row_index] = dot_runtime(
                execution.candidate.settings,
                row,
                vector_tensor,
            )

    runtime_values.require_finite_tensor(result, "row-batched HVP result")

    return runtime_values.wrap_flat_parameter_tree(active_params, result)


def _run_hvp_vector_single_loop(
    execution: runtime_values.StandardExecution,
) -> TensorTree:
    if _hvp_uses_reverse_reuse(execution.candidate.settings):
        return _run_hvp_reused_reverse_vectors(execution)

    return run_vector_single_loop(execution, _run_hvp_single_vector)


def _run_hvp_vector_manual_batch(
    execution: runtime_values.StandardExecution,
) -> TensorTree:
    return run_vector_manual_batches(execution, _run_hvp_vector_single_loop)


def run_vector_manual_batches(
    execution: runtime_values.StandardExecution,
    runner: Callable[[runtime_values.StandardExecution], TensorTree],
) -> TensorTree:
    """Run a vector operation over declared manual batches.

    Returns:
        Run a vector operation over declared manual batches.
    """

    def vector_runner(vector: TensorTree) -> TensorTree:
        return runner(_execution_with_vector(execution, vector))

    return _run_tensor_tree_vector_manual_batches(
        execution.vector,
        execution.candidate.settings,
        vector_runner,
    )


def _run_tensor_tree_vector_manual_batches(
    vector_tree: TensorTree,
    settings: Mapping[str, Any],
    runner: Callable[[TensorTree], TensorTree],
) -> TensorTree:
    vector_in_dims = vector_tree_in_dims(
        vector_tree,
        settings,
    )
    vector_count = runtime_values.vector_tree_batch_size(vector_tree, vector_in_dims)
    batch_size = runtime_values.manual_vector_batch_size(settings)
    results = []

    for start in range(0, vector_count, batch_size):
        stop = min(start + batch_size, vector_count)
        vector = runtime_values.vector_tree_slice(
            vector_tree, vector_in_dims, start, stop
        )
        result = runner(vector)
        results.append(result)

    return runtime_values.cat_tensor_trees(tuple(results), 0)


def run_vector_vmap(
    execution: runtime_values.StandardExecution,
    runner: Callable[[TensorTree], TensorTree],
) -> TensorTree:
    """Run a vector operation under torch.func.vmap.

    Returns:
        The a vector operation under torch.func.vmap.
    """
    return _run_tensor_tree_vector_vmap(
        execution.vector,
        execution.candidate.settings,
        runner,
    )


def _run_tensor_tree_by_vectorization_mode(
    vector_tree: TensorTree,
    settings: Mapping[str, Any],
    runner: Callable[[TensorTree], TensorTree],
) -> TensorTree:
    mode = settings.get("vectorization.mode")

    if mode == "single_loop":
        return _run_tensor_tree_vector_single_loop(vector_tree, settings, runner)

    if mode == "manual_batch":
        return _run_tensor_tree_vector_manual_batches(vector_tree, settings, runner)

    if mode == "vmap":
        return _run_tensor_tree_vector_vmap(vector_tree, settings, runner)

    return runner(vector_tree)


def _run_tensor_tree_vector_vmap(
    vector_tree: TensorTree,
    settings: Mapping[str, Any],
    runner: Callable[[TensorTree], TensorTree],
) -> TensorTree:
    vector_in_dims = vector_tree_in_dims(
        vector_tree,
        settings,
    )

    return torch_func_vmap(
        runner,
        in_dims=(vector_in_dims,),
        randomness=settings["vectorization.randomness"],
        chunk_size=vmap_chunk_size(settings),
    )(vector_tree)


def run_vector_single_loop(
    execution: runtime_values.StandardExecution,
    runner: Callable[[runtime_values.StandardExecution], TensorTree],
) -> TensorTree:
    """Run a vector operation one vector at a time.

    Returns:
        Run a vector operation one vector at a time.
    """

    def vector_runner(vector: TensorTree) -> TensorTree:
        return runner(_execution_with_vector(execution, vector))

    return _run_tensor_tree_vector_single_loop(
        execution.vector,
        execution.candidate.settings,
        vector_runner,
    )


def _run_tensor_tree_vector_single_loop(
    vector_tree: TensorTree,
    settings: Mapping[str, Any],
    runner: Callable[[TensorTree], TensorTree],
) -> TensorTree:
    def vector_runner(vector: TensorTree, _: bool) -> TensorTree:
        return runner(vector)

    return _run_tensor_tree_vector_single_loop_with_last(
        vector_tree,
        settings,
        vector_runner,
    )


def _run_tensor_tree_vector_single_loop_with_last(
    vector_tree: TensorTree,
    settings: Mapping[str, Any],
    runner: Callable[[TensorTree, bool], TensorTree],
) -> TensorTree:
    vector_in_dims = vector_tree_in_dims(
        vector_tree,
        settings,
    )
    vector_count = runtime_values.vector_tree_batch_size(vector_tree, vector_in_dims)
    results = []

    for index in range(vector_count):
        vector = runtime_values.vector_tree_select(vector_tree, vector_in_dims, index)
        result = runner(vector, index == vector_count - 1)
        results.append(result)

    return runtime_values.stack_tensor_trees(tuple(results), 0)


def _hvp_uses_reverse_reuse(settings: Mapping[str, Any]) -> bool:
    return (
        settings.get("hvp.graph_schedule") == "retain_graph_across_vectors"
        or settings.get("hvp.primal_reuse") == "reuse_primal"
    )


def _run_hvp_reused_reverse_vectors(
    execution: runtime_values.StandardExecution,
) -> TensorTree:
    scalar_function = _hvp_scalar_function(execution)
    active_params = runtime_values.grad_enabled_params(execution.params)
    value = scalar_function(active_params)
    retain_gradient_graph = (
        execution.candidate.settings.get("hvp.graph_schedule")
        == "retain_graph_across_vectors"
    )

    if retain_gradient_graph:
        return _run_hvp_reused_gradient_graph_vectors(
            execution,
            active_params,
            value,
        )

    return _run_hvp_reused_primal_vectors(
        execution,
        active_params,
        value,
    )


def _run_hvp_reused_gradient_graph_vectors(
    execution: runtime_values.StandardExecution,
    active_params: ParameterTree,
    value: torch.Tensor,
) -> TensorTree:
    gradient_tree = _hvp_gradient_tree(
        active_params,
        value,
        retain_graph=True,
    )

    def vector_runner(vector: TensorTree, is_last: bool) -> TensorTree:
        return _hvp_from_gradient_tree(
            active_params,
            gradient_tree,
            vector,
            execution.candidate.settings,
            retain_graph=not is_last,
        )

    return _run_tensor_tree_vector_single_loop_with_last(
        execution.vector,
        execution.candidate.settings,
        vector_runner,
    )


def _run_hvp_reused_primal_vectors(
    execution: runtime_values.StandardExecution,
    active_params: ParameterTree,
    value: torch.Tensor,
) -> TensorTree:
    def vector_runner(vector: TensorTree, is_last: bool) -> TensorTree:
        gradient_tree = _hvp_gradient_tree(
            active_params,
            value,
            retain_graph=not is_last,
        )

        return _hvp_from_gradient_tree(
            active_params,
            gradient_tree,
            vector,
            execution.candidate.settings,
            retain_graph=False,
        )

    return _run_tensor_tree_vector_single_loop_with_last(
        execution.vector,
        execution.candidate.settings,
        vector_runner,
    )


def _hvp_gradient_tree(
    active_params: ParameterTree,
    value: torch.Tensor,
    *,
    retain_graph: bool,
) -> TensorTree:
    leaves = tuple(active_params.values())
    gradient_leaves = torch.autograd.grad(
        value,
        leaves,
        create_graph=True,
        retain_graph=retain_graph,
        allow_unused=True,
    )

    return tree_from_leaves(
        active_params,
        tuple(
            torch.zeros_like(leaf) if gradient is None else gradient
            for leaf, gradient in zip(leaves, gradient_leaves, strict=True)
        ),
    )


def _hvp_from_gradient_tree(
    active_params: ParameterTree,
    gradient_tree: TensorTree,
    vector: TensorTree,
    settings: Mapping[str, Any],
    *,
    retain_graph: bool,
) -> TensorTree:
    leaves = tuple(active_params.values())
    dot = tree_dot_runtime(settings, gradient_tree, vector)
    hvp_leaves = torch.autograd.grad(
        dot,
        leaves,
        retain_graph=retain_graph,
        allow_unused=True,
    )

    return tree_from_leaves(
        active_params,
        tuple(
            torch.zeros_like(leaf) if hvp is None else hvp.detach()
            for leaf, hvp in zip(leaves, hvp_leaves, strict=True)
        ),
    )


def _run_hvp_vector_vmap(execution: runtime_values.StandardExecution) -> TensorTree:
    if execution.path not in runtime_values.HVP_VECTOR_VMAP_PATHS:
        message = "vectorization.mode=vmap requires linearize_grad HVP"
        raise MaterializationError(message)

    if execution.linearized_hvp is not None:
        hvp_function = execution.linearized_hvp
    else:
        gradient_function = torch.func.grad(_hvp_scalar_function(execution))
        _, hvp_function = torch.func.linearize(
            gradient_function,
            execution.params,
        )

    return run_vector_vmap(execution, hvp_function)


def _hvp_scalar_function(
    execution: runtime_values.StandardExecution,
) -> Callable[[ParameterTree], torch.Tensor]:
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
            _model_compute_tree(active_params, settings),
            _model_compute_tree(execution.buffers, settings),
            _model_compute_batch(execution.batch, settings),
            execution.context,
        )

    return scalar_function


def _run_hvp_forward_ad_path(
    execution: runtime_values.StandardExecution,
) -> TensorTree:
    scalar_function = _hvp_scalar_function(execution)

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
    compiled_scalar_function = _hvp_scalar_function(execution)

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


def _run_fisher_vp(execution: runtime_values.StandardExecution) -> TensorTree:
    return _run_fisher_family_vp(execution, FISHER_FAMILY_RUNTIME_ROWS["fisher_vp"])


def _run_fisher_family_vp(
    execution: runtime_values.StandardExecution,
    row: Mapping[str, Any],
) -> TensorTree:
    def single_vector(row_execution: runtime_values.StandardExecution) -> TensorTree:
        row["precheck"](row_execution)

        return _run_fisher_family_vp_single_vector(row_execution, row)

    def vmap(row_execution: runtime_values.StandardExecution) -> TensorTree:
        row["precheck"](row_execution)

        return _run_fisher_family_vp_vector_vmap(row_execution, row)

    return run_single_vectorized_by_path(
        execution,
        row["paths"],
        single_vector,
        single_vector,
        vmap,
    )


def _run_fisher_family_vp_single_vector(
    execution: runtime_values.StandardExecution,
    row: Mapping[str, Any],
) -> TensorTree:
    if execution.path in row["streaming_paths"]:
        row["require_streaming"](execution)
        result = _streaming_score_gradient_product(
            execution,
            row["streaming_normalization"](execution),
            row["streaming_label"],
        )
        row["check_result"](execution, result)

        return runtime_values.wrap_flat_vector(execution.params, result)

    if execution.path == row["blockwise_path"]:
        row["require_matrix"](execution)

        return _run_blockwise_score_matrix_product(
            execution,
            row["block_batch_key"],
            row["blockwise_normalization"](execution),
            row["matrix_batch_key"],
        )

    row["require_matrix"](execution)
    score_matrix = runtime_values.batch_tensor(execution.batch, row["matrix_batch_key"])
    score_matrix = _loss_scaled_score_matrix(execution, score_matrix)

    return _run_score_matrix_product_single_vector(
        execution,
        score_matrix,
        row["matrix_normalization"](execution, score_matrix),
        row["vector_label"],
        row["result_label"],
        row["check_result"],
    )


def _run_fisher_family_vp_vector_vmap(
    execution: runtime_values.StandardExecution,
    row: Mapping[str, Any],
) -> TensorTree:
    if execution.path == row["blockwise_path"]:
        row["require_matrix"](execution)
        result = _blockwise_score_matrix_product_batch_vmap(
            execution,
            row["block_batch_key"],
            row["blockwise_normalization"](execution),
            row["blockwise_label"],
        )
        row["check_result"](execution, result)

        return runtime_values.wrap_flat_vector_batch(execution.params, result)

    if execution.path in row["streaming_paths"]:
        row["require_streaming"](execution)

        return _run_streaming_score_gradient_product_vmap(
            execution,
            row["streaming_normalization"](execution),
            row["streaming_label"],
        )

    score_matrix = _score_fisher_matrix_for_product(
        execution,
        row["streaming_paths"],
        row["dense_path"],
        row["matrix_batch_key"],
        row["streaming_matrix"],
        row["require_streaming"],
        row["require_matrix"],
        row["vmap_error_message"],
    )
    result = _score_matrix_product_batch_vmap(
        execution,
        score_matrix,
        row["matrix_normalization"](execution, score_matrix),
        row["streaming_label"],
    )
    row["check_result"](execution, result)

    return runtime_values.wrap_flat_vector_batch(execution.params, result)


def _score_fisher_matrix_for_product(
    execution: runtime_values.StandardExecution,
    streaming_paths: tuple[str, ...],
    dense_path: str,
    dense_batch_key: str,
    streaming_matrix: Callable[[runtime_values.StandardExecution], torch.Tensor],
    require_streaming: Callable[[runtime_values.StandardExecution], None],
    require_dense: Callable[[runtime_values.StandardExecution], None],
    error_message: str,
) -> torch.Tensor:
    if execution.path in streaming_paths:
        require_streaming(execution)

        return _loss_scaled_score_matrix(execution, streaming_matrix(execution))

    if execution.path == dense_path:
        require_dense(execution)

        return _loss_scaled_score_matrix(
            execution,
            runtime_values.batch_tensor(execution.batch, dense_batch_key),
        )

    raise MaterializationError(error_message)


def _require_explicit_score_fisher_execution(
    execution: runtime_values.StandardExecution,
) -> None:
    _require_explicit_score_fisher_semantics(execution.operator)


def _require_valid_fisher_execution(
    execution: runtime_values.StandardExecution,
) -> None:
    _require_valid_fisher_semantics(execution.operator)


def _fisher_score_matrix_normalization(
    execution: runtime_values.StandardExecution,
    score_matrix: torch.Tensor,
) -> float:
    _ = score_matrix

    return _fisher_normalization(execution)


def _skip_score_fisher_requirement(execution: runtime_values.StandardExecution) -> None:
    _ = execution


def _run_sampled_fisher_vp(execution: runtime_values.StandardExecution) -> TensorTree:
    return _run_fisher_family_vp(
        execution,
        FISHER_FAMILY_RUNTIME_ROWS["sampled_fisher_vp"],
    )


def _sampled_fisher_score_matrix_normalization(
    execution: runtime_values.StandardExecution,
    score_matrix: torch.Tensor,
) -> float:
    _ = score_matrix

    return _sampled_fisher_normalization(execution)


def _run_empirical_fisher_vp(execution: runtime_values.StandardExecution) -> TensorTree:
    return _run_fisher_family_vp(
        execution,
        FISHER_FAMILY_RUNTIME_ROWS["empirical_fisher_vp"],
    )


def _score_gradient_matrix_from_builders(
    execution: runtime_values.StandardExecution,
    compile_boundary: str,
    paths: tuple[str, ...],
    error_message: str,
    *,
    use_compiled: bool = True,
) -> torch.Tensor:
    if (
        use_compiled
        and execution.compiled_score_matrix is not None
        and execution.candidate.settings.get("compile.boundary") == compile_boundary
    ):
        return execution.compiled_score_matrix()

    if _uses_manual_per_example_schedule(execution):
        return _per_example_gradient_matrix_manual_batches(execution)

    if execution.path not in paths:
        raise MaterializationError(error_message)

    return _per_example_gradient_matrix_from_builders(
        execution,
        PER_EXAMPLE_GRADIENT_WITHOUT_MANUAL_BUILDERS,
        error_message,
    )


def _score_gradient_matrix_from_operator_row(
    execution: runtime_values.StandardExecution,
) -> torch.Tensor:
    row = runtime_values.SCORE_MATRIX_COMPILE_ROWS.get(execution.operator.kind)

    if row is None:
        message = f"score-gradient matrix is not lowered for {execution.operator.kind}"
        raise MaterializationError(message)

    return _score_gradient_matrix_from_row(execution, row)


def _score_gradient_matrix_from_row(
    execution: runtime_values.StandardExecution,
    row: runtime_values.ScoreMatrixCompileRow,
    *,
    use_compiled: bool = True,
) -> torch.Tensor:
    return _score_gradient_matrix_from_builders(
        execution,
        row.boundary,
        row.paths,
        row.message,
        use_compiled=use_compiled,
    )


def _empirical_fisher_blockwise_normalization(
    execution: runtime_values.StandardExecution,
) -> float:
    blocks = runtime_values.batch_tensor_blocks(
        execution.batch,
        "per_example_gradient_blocks",
    )

    return _empirical_fisher_normalization(
        execution.batch,
        execution.operator,
        blocks[0],
    )


def _empirical_fisher_score_matrix_normalization(
    execution: runtime_values.StandardExecution,
    score_matrix: torch.Tensor,
) -> float:
    return _empirical_fisher_normalization(
        execution.batch,
        execution.operator,
        score_matrix,
    )


def _run_per_example_gradient(
    execution: runtime_values.StandardExecution,
) -> TensorTree:
    runtime_values.require_path(
        execution.operator.kind,
        execution.path,
        runtime_values.PER_EXAMPLE_GRADIENT_PATHS,
    )
    accumulation = execution.candidate.settings.get("per_example_gradient.accumulation")

    if accumulation == "stacked_leading_axis":
        matrix = _score_gradient_matrix_from_operator_row(execution)
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


def _run_blockwise_score_matrix_product(
    execution: runtime_values.StandardExecution,
    batch_key: str,
    normalization: float,
    label: str,
) -> TensorTree:
    blocks = _loss_scaled_score_blocks(
        execution,
        runtime_values.batch_tensor_blocks(execution.batch, batch_key),
    )
    vector_tensor = parameter_order_vector(execution)
    result = _blockwise_score_matrix_product(
        blocks,
        vector_tensor,
        normalization,
        label,
        execution.candidate.settings,
    )

    return runtime_values.wrap_flat_vector(execution.params, result)


def _blockwise_score_matrix_product(
    blocks: tuple[torch.Tensor, ...],
    vector: torch.Tensor,
    normalization: float,
    label: str,
    settings: Mapping[str, Any],
) -> torch.Tensor:
    first_block = blocks[0]
    offset = first_block.shape[1]
    score_dot = matmul_runtime(settings, first_block, vector[:offset])

    for block in blocks[1:]:
        width = block.shape[1]
        stop = offset + width

        if stop > vector.numel():
            message = f"{label} block columns exceed vector length"
            raise MaterializationError(message)

        score_dot = score_dot + matmul_runtime(settings, block, vector[offset:stop])
        offset = stop

    if offset != vector.numel():
        message = f"{label} block columns must match vector length"
        raise MaterializationError(message)

    pieces = tuple(matmul_runtime(settings, block.T, score_dot) for block in blocks)
    result = torch.cat(pieces) / normalization
    runtime_values.require_finite_tensor(result, f"{label} blockwise result")

    return result


def _skip_score_matrix_result_check(
    execution: runtime_values.StandardExecution,
    result: torch.Tensor,
) -> None:
    _ = execution, result


def _run_score_matrix_product_single_vector(
    execution: runtime_values.StandardExecution,
    score_gradients: torch.Tensor,
    normalization: float,
    vector_label: str,
    result_label: str,
    check_result: Callable[[runtime_values.StandardExecution, torch.Tensor], None],
) -> TensorTree:
    vector_tensor = parameter_order_vector(execution)
    runtime_values.require_finite_tensor(score_gradients, "score_gradients")
    runtime_values.require_finite_tensor(vector_tensor, vector_label)
    result = _score_matrix_product(
        score_gradients,
        vector_tensor,
        normalization,
        execution.candidate.settings,
        execution.parameter_surface,
    )
    runtime_values.require_finite_tensor(result, result_label)
    check_result(execution, result)

    return runtime_values.wrap_flat_vector(execution.params, result)


def _score_matrix_product(
    score_gradients: torch.Tensor,
    vector: torch.Tensor,
    normalization: float,
    settings: Mapping[str, Any],
    parameter_surface: ParameterSurface | None,
) -> torch.Tensor:
    _require_score_matrix_vector_shape(score_gradients, vector, "score_gradients")
    ranges = runtime_values.parameter_column_ranges(
        vector.numel(), settings, parameter_surface
    )

    if ranges is not None:
        return _parameter_blocked_score_matrix_product(
            score_gradients,
            vector,
            normalization,
            settings,
            ranges,
        )

    score_dot = matmul_runtime(settings, score_gradients, vector)

    return matmul_runtime(settings, score_gradients.T, score_dot) / normalization


def _parameter_blocked_score_matrix_product(
    score_gradients: torch.Tensor,
    vector: torch.Tensor,
    normalization: float,
    settings: Mapping[str, Any],
    ranges: tuple[tuple[int, int], ...],
) -> torch.Tensor:
    _require_score_matrix_vector_shape(score_gradients, vector, "score_gradients")
    score_dot = parameter_blocked_matrix_vector_product(
        score_gradients,
        vector,
        settings,
        None,
        ranges,
    )
    result_chunks = []

    for start, stop in ranges:
        score_block = score_gradients[:, start:stop]
        result_chunks.append(matmul_runtime(settings, score_block.T, score_dot))

    result = torch.cat(tuple(result_chunks)) / normalization
    runtime_values.require_finite_tensor(result, "score matrix parameter-block result")

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
        return matmul_runtime(settings, matrix, vector)

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
            matmul_runtime(settings, matrix[:, start:stop], vector[start:stop])
        )

    result = chunks[0]

    for chunk in chunks[1:]:
        result = result + chunk

    return result


def _require_score_matrix_vector_shape(
    score_gradients: torch.Tensor,
    vector: torch.Tensor,
    label: str,
) -> None:
    if score_gradients.ndim != runtime_values.MATRIX_DIMS:
        message = f"{label} must be a two-dimensional tensor"
        raise MaterializationError(message)

    if vector.ndim != 1:
        message = "Fisher vector must flatten to a one-dimensional tensor"
        raise MaterializationError(message)

    if score_gradients.shape[0] == 0:
        message = f"{label} must have at least one row"
        raise MaterializationError(message)

    if score_gradients.shape[1] != vector.numel():
        message = f"{label} column count must match vector width"
        raise MaterializationError(message)


def _streaming_score_gradient_product(
    execution: runtime_values.StandardExecution,
    normalization: float,
    label: str,
) -> torch.Tensor:
    vector_tensor = parameter_order_vector(execution)

    return _streaming_score_gradient_product_for_vector(
        execution,
        vector_tensor,
        normalization,
        label,
    )


def _streaming_score_gradient_product_for_vector(
    execution: runtime_values.StandardExecution,
    vector_tensor: torch.Tensor,
    normalization: float,
    label: str,
) -> torch.Tensor:

    if _uses_manual_per_example_schedule(execution):
        result = _streaming_score_gradient_product_manual_batches(
            execution,
            vector_tensor,
        )
    else:
        result = _streaming_score_gradient_product_without_manual_batch(
            execution,
            vector_tensor,
        )

    result = result / normalization
    runtime_values.require_finite_tensor(result, f"{label} streaming result")

    return result


def _run_streaming_score_gradient_product_vmap(
    execution: runtime_values.StandardExecution,
    normalization: float,
    label: str,
) -> TensorTree:
    vector_batch = _flat_vector_batch(execution)
    chunk_size = vmap_chunk_size(execution.candidate.settings)
    result = torch.zeros_like(vector_batch)

    for row in _streaming_gradient_rows(execution):
        result = _accumulate_streaming_gradient_row_batch(
            execution,
            result,
            row,
            vector_batch,
            chunk_size,
        )

    result = result / normalization
    runtime_values.require_finite_tensor(result, f"{label} streaming batched result")

    return runtime_values.wrap_flat_vector_batch(execution.params, result)


def _streaming_score_gradient_product_manual_batches(
    execution: runtime_values.StandardExecution,
    vector_tensor: torch.Tensor,
) -> torch.Tensor:
    result = torch.zeros_like(vector_tensor)

    for subexecution in _per_example_sliced_executions(
        execution,
        "per-example manual batching",
        _per_example_manual_batch_size(execution),
    ):
        result = result + _streaming_score_gradient_product_without_manual_batch(
            subexecution,
            vector_tensor,
        )

    return result


def _streaming_score_gradient_product_without_manual_batch(
    execution: runtime_values.StandardExecution,
    vector_tensor: torch.Tensor,
) -> torch.Tensor:
    result = torch.zeros_like(vector_tensor)

    for row in _streaming_gradient_rows_without_manual_batch(execution):
        result = _accumulate_streaming_gradient_row(
            execution,
            result,
            row,
            vector_tensor,
        )

    return result


def _streaming_gradient_rows(
    execution: runtime_values.StandardExecution,
) -> Iterator[torch.Tensor]:
    if _uses_manual_per_example_schedule(execution):
        for subexecution in _per_example_sliced_executions(
            execution,
            "per-example manual batching",
            _per_example_manual_batch_size(execution),
        ):
            yield from _streaming_gradient_rows_without_manual_batch(subexecution)

        return

    yield from _streaming_gradient_rows_without_manual_batch(execution)


def _streaming_gradient_rows_without_manual_batch(
    execution: runtime_values.StandardExecution,
) -> Iterator[torch.Tensor]:
    compiled_rows = _compiled_streaming_gradient_rows(execution)

    if compiled_rows is not None:
        yield from compiled_rows

        return

    builder = STREAMING_GRADIENT_ROW_BUILDERS.get(execution.path)

    if builder is not None:
        yield from builder(execution)
        return

    message = "streaming score-gradient product requires a score-gradient path"
    raise MaterializationError(message)


def _compiled_streaming_gradient_rows(
    execution: runtime_values.StandardExecution,
) -> Iterator[torch.Tensor] | None:
    if execution.compiled_score_matrix is None:
        return None

    boundary = execution.candidate.settings.get("compile.boundary")

    if _score_matrix_compile_row(execution.operator.kind, boundary) is None:
        return None

    score_gradients = execution.compiled_score_matrix()

    if score_gradients.ndim != runtime_values.MATRIX_DIMS:
        message = "compiled score-gradient rows must be a matrix"
        raise MaterializationError(message)

    runtime_values.require_finite_tensor(
        score_gradients, "compiled score-gradient rows"
    )

    return (row.reshape(-1) for row in score_gradients)


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
        output = call_function_objective(execution, function, active_params)

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


def _streaming_gradient_rows_vmap(
    execution: runtime_values.StandardExecution,
) -> Iterator[torch.Tensor]:
    gradients, parameter_items, row_count = _per_example_gradient_tree_vmap(execution)
    pieces = []

    for name, _ in parameter_items:
        gradient = gradients[name].reshape(row_count, -1)
        pieces.append(gradient)

    for index in range(row_count):
        yield torch.cat(tuple(piece[index] for piece in pieces))


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
    (runtime_values.STREAMING_GRADIENT_VMAP_PATHS, _streaming_gradient_rows_vmap),
))


def _accumulate_streaming_gradient_row_batch(
    execution: runtime_values.StandardExecution,
    result: torch.Tensor,
    row: torch.Tensor,
    vector_batch: torch.Tensor,
    chunk_size: int | None,
) -> torch.Tensor:
    scale = _loss_scale(execution.candidate.settings)

    if scale is not None:
        row = row * scale

    def product(flat_vector: torch.Tensor) -> torch.Tensor:
        return row * dot_runtime(execution.candidate.settings, row, flat_vector)

    return result + torch_func_vmap(
        product,
        in_dims=0,
        randomness=execution.candidate.settings["vectorization.randomness"],
        chunk_size=chunk_size,
    )(vector_batch)


def _accumulate_streaming_gradient_row(
    execution: runtime_values.StandardExecution,
    result: torch.Tensor,
    row: torch.Tensor,
    vector_tensor: torch.Tensor,
) -> torch.Tensor:
    scale = _loss_scale(execution.candidate.settings)

    if scale is not None:
        row = row * scale

    return result + row * dot_runtime(execution.candidate.settings, row, vector_tensor)


def _flat_gradient_row(gradients: Sequence[torch.Tensor]) -> torch.Tensor:
    return torch.cat(tuple(gradient.reshape(-1) for gradient in gradients))


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


def _score_matrix_product_batch_vmap(
    execution: runtime_values.StandardExecution,
    score_gradients: torch.Tensor,
    normalization: float,
    label: str,
) -> torch.Tensor:
    vector_batch = _flat_vector_batch(execution)
    _require_score_matrix_product_inputs(
        score_gradients,
        vector_batch,
        f"{label} score_gradients",
    )
    chunk_size = vmap_chunk_size(execution.candidate.settings)

    def product(flat_vector: torch.Tensor) -> torch.Tensor:
        return _score_matrix_product(
            score_gradients,
            flat_vector,
            normalization,
            execution.candidate.settings,
            execution.parameter_surface,
        )

    result = torch_func_vmap(
        product,
        in_dims=0,
        randomness=execution.candidate.settings["vectorization.randomness"],
        chunk_size=chunk_size,
    )(vector_batch)
    runtime_values.require_finite_tensor(result, f"{label} batched result")

    return result


def _blockwise_score_matrix_product_batch_vmap(
    execution: runtime_values.StandardExecution,
    batch_key: str,
    normalization: float,
    label: str,
) -> torch.Tensor:
    blocks = _loss_scaled_score_blocks(
        execution,
        runtime_values.batch_tensor_blocks(execution.batch, batch_key),
    )
    vector_batch = _flat_vector_batch(execution)
    _require_blockwise_score_matrix_product_inputs(blocks, vector_batch, label)
    chunk_size = vmap_chunk_size(execution.candidate.settings)

    def product(flat_vector: torch.Tensor) -> torch.Tensor:
        return _blockwise_score_matrix_product_unchecked(
            blocks,
            flat_vector,
            normalization,
            execution.candidate.settings,
        )

    result = torch_func_vmap(
        product,
        in_dims=0,
        randomness=execution.candidate.settings["vectorization.randomness"],
        chunk_size=chunk_size,
    )(vector_batch)
    runtime_values.require_finite_tensor(result, f"{label} blockwise batched result")

    return result


def _blockwise_score_matrix_product_unchecked(
    blocks: tuple[torch.Tensor, ...],
    vector: torch.Tensor,
    normalization: float,
    settings: Mapping[str, Any],
) -> torch.Tensor:
    first_block = blocks[0]
    offset = first_block.shape[1]
    score_dot = matmul_runtime(settings, first_block, vector[:offset])

    for block in blocks[1:]:
        width = block.shape[1]
        stop = offset + width
        score_dot = score_dot + matmul_runtime(settings, block, vector[offset:stop])
        offset = stop

    pieces = tuple(matmul_runtime(settings, block.T, score_dot) for block in blocks)

    return torch.cat(pieces) / normalization


def _flat_vector_batch(execution: runtime_values.StandardExecution) -> torch.Tensor:
    if execution.flat_parameter_vector_batch is not None:
        return execution.flat_parameter_vector_batch

    return _build_flat_vector_batch(execution)


def _build_flat_vector_batch(
    execution: runtime_values.StandardExecution,
) -> torch.Tensor:
    vector_in_dims = vector_tree_in_dims(
        execution.vector,
        execution.candidate.settings,
    )

    return runtime_values.flatten_vector_batch(
        execution.params, execution.vector, vector_in_dims
    )


def _require_score_matrix_product_inputs(
    score_gradients: torch.Tensor,
    vector_batch: torch.Tensor,
    label: str,
) -> None:
    if score_gradients.ndim != runtime_values.MATRIX_DIMS:
        message = f"{label} must be a two-dimensional tensor"
        raise MaterializationError(message)

    if vector_batch.ndim != runtime_values.MATRIX_DIMS:
        message = "vectorized Fisher vectors must flatten to a matrix"
        raise MaterializationError(message)

    if score_gradients.shape[0] == 0:
        message = f"{label} must have at least one row"
        raise MaterializationError(message)

    if score_gradients.shape[1] != vector_batch.shape[1]:
        message = f"{label} column count must match vector width"
        raise MaterializationError(message)

    runtime_values.require_finite_tensor(score_gradients, label)
    runtime_values.require_finite_tensor(vector_batch, "vectorized Fisher vectors")


def _require_blockwise_score_matrix_product_inputs(
    blocks: tuple[torch.Tensor, ...],
    vector_batch: torch.Tensor,
    label: str,
) -> None:
    if vector_batch.ndim != runtime_values.MATRIX_DIMS:
        message = "vectorized Fisher vectors must flatten to a matrix"
        raise MaterializationError(message)

    if blocks[0].shape[0] == 0:
        message = f"{label} blocks must have at least one row"
        raise MaterializationError(message)

    width = sum(block.shape[1] for block in blocks)

    if width != vector_batch.shape[1]:
        message = f"{label} block columns must match vector width"
        raise MaterializationError(message)

    runtime_values.require_finite_tensor(vector_batch, "vectorized Fisher vectors")


def _uses_manual_per_example_schedule(
    execution: runtime_values.StandardExecution,
) -> bool:
    if execution.candidate.settings.get("schedule.per_example") != "manual_batch":
        return False

    return execution.path in runtime_values.FISHER_MANUAL_PER_EXAMPLE_PATHS


def _per_example_sliced_executions(
    execution: runtime_values.StandardExecution,
    label: str,
    batch_size: int,
) -> Iterator[runtime_values.StandardExecution]:
    batch, batch_in_dims = _per_example_batch_in_dims(
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


def _per_example_gradient_matrix_manual_batches(
    execution: runtime_values.StandardExecution,
) -> torch.Tensor:
    return _per_example_gradient_matrix_batched(
        execution,
        "per-example manual batching",
        _per_example_manual_batch_size(execution),
    )


def _per_example_gradient_matrix_blockwise(
    execution: runtime_values.StandardExecution,
) -> torch.Tensor:
    return _per_example_gradient_matrix_batched(
        execution,
        "per-example gradient blockwise stacking",
        _per_example_block_size(execution),
    )


def _per_example_gradient_matrix_batched(
    execution: runtime_values.StandardExecution,
    label: str,
    batch_size: int,
) -> torch.Tensor:
    rows = [
        _per_example_gradient_matrix_without_manual_batch(subexecution)
        for subexecution in _per_example_sliced_executions(execution, label, batch_size)
    ]

    return torch.cat(tuple(rows), dim=0)


def _per_example_gradient_matrix_without_manual_batch(
    execution: runtime_values.StandardExecution,
) -> torch.Tensor:
    return _per_example_gradient_matrix_from_builders(
        execution,
        PER_EXAMPLE_GRADIENT_WITHOUT_MANUAL_BUILDERS,
        (
            "schedule.per_example=manual_batch is incompatible with path: "
            f"{execution.path}"
        ),
    )


def _per_example_manual_batch_size(execution: runtime_values.StandardExecution) -> int:
    if execution.operator.kind in {"fisher_vp", "sampled_fisher_vp"}:
        key = "batch.fisher_sample_batch_size"
    elif execution.operator.kind == "empirical_fisher_vp":
        key = "batch.empirical_example_batch_size"
    else:
        message = "per-example manual batching requires a Fisher-family operator"
        raise MaterializationError(message)

    return runtime_values.required_positive_int_setting(
        execution.candidate.settings,
        key,
        f"{key} must be a positive integer",
    )


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
        output = call_function_objective(execution, function, active_params)

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


def _loss_scaled_score_matrix(
    execution: runtime_values.StandardExecution,
    matrix: torch.Tensor,
) -> torch.Tensor:
    scale = _loss_scale(execution.candidate.settings)

    if scale is None:
        return matrix

    return matrix * scale


def _loss_scaled_score_blocks(
    execution: runtime_values.StandardExecution,
    blocks: tuple[torch.Tensor, ...],
) -> tuple[torch.Tensor, ...]:
    scale = _loss_scale(execution.candidate.settings)

    if scale is None:
        return blocks

    return tuple(block * scale for block in blocks)


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
    output = call_function_objective(execution, function, active_params)

    if not isinstance(output, torch.Tensor):
        message = "per-example gradient path requires tensor objective output"
        raise MaterializationError(message)

    terms = output.reshape(-1)

    runtime_values.require_nonempty_per_example_terms(
        terms, "per-example gradient path"
    )

    return terms


def _per_example_gradient_matrix_vmap(
    execution: runtime_values.StandardExecution,
) -> torch.Tensor:
    gradients, parameter_items, row_count = _per_example_gradient_tree_vmap(execution)
    pieces = []

    for name, _ in parameter_items:
        gradient = gradients[name]
        pieces.append(gradient.reshape(row_count, -1))

    return torch.cat(tuple(pieces), dim=1)


def _per_example_gradient_matrix_from_builders(
    execution: runtime_values.StandardExecution,
    builders: Mapping[str, Callable[[runtime_values.StandardExecution], torch.Tensor]],
    message: str,
) -> torch.Tensor:
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
    (runtime_values.PER_EXAMPLE_GRADIENT_VMAP_PATHS, _per_example_gradient_matrix_vmap),
))


def _per_example_gradient_tree_vmap(
    execution: runtime_values.StandardExecution,
) -> tuple[ParameterTree, tuple[tuple[str, torch.Tensor], ...], int]:
    try:
        admit_torch_func(execution.candidate.settings)
    except AdmissionError as error:
        raise MaterializationError(str(error)) from error

    function = runtime_values.function_objective(
        execution.operator, execution.function_objectives
    )
    parameter_items = tuple(execution.params.items())
    active_params = {
        name: tensor.detach().clone().requires_grad_(True)
        for name, tensor in parameter_items
    }
    batched_batch, batch_in_dims = _per_example_vmap_batch(execution.batch)
    chunk_size = _per_example_vmap_chunk_size(execution)

    def single_loss(
        active_params: ParameterTree,
        single_tensor_batch: Batch,
    ) -> torch.Tensor:
        output = call_function_objective(
            execution,
            function,
            active_params,
            single_tensor_batch,
        )

        if not isinstance(output, torch.Tensor):
            message = "per-example vmap requires tensor objective output"
            raise MaterializationError(message)

        terms = output.reshape(-1)

        if terms.numel() != 1:
            message = "per-example vmap objective must return one scalar per example"
            raise MaterializationError(message)

        return terms[0]

    gradients = torch_func_vmap(
        _torch_func_grad(single_loss),
        in_dims=(None, batch_in_dims),
        randomness=str(execution.candidate.settings["vectorization.randomness"]),
        chunk_size=chunk_size,
    )(active_params, batched_batch)
    row_count = _vmap_batch_size(batched_batch, batch_in_dims)

    return gradients, parameter_items, row_count


def _per_example_vmap_batch(
    batch: Batch,
) -> tuple[dict[str, Any], dict[str, int | None]]:
    return _per_example_batch_in_dims(batch, "per-example vmap")


def _per_example_batch_in_dims(
    batch: Batch,
    label: str,
) -> tuple[dict[str, Any], dict[str, int | None]]:
    result = {}
    in_dims = {}
    expected_size = None

    for key, value in batch.items():
        result[key] = value

        if not isinstance(value, torch.Tensor):
            in_dims[key] = None
            continue

        if value.ndim == 0:
            message = f"{label} tensor batch field is scalar: {key}"
            raise MaterializationError(message)

        leading_size = value.shape[0]

        if expected_size is None:
            expected_size = leading_size
        elif leading_size != expected_size:
            message = f"{label} batch leading dimensions differ"
            raise MaterializationError(message)

        in_dims[key] = 0

    if expected_size is None or expected_size == 0:
        message = f"{label} requires a nonempty mapped batch"
        raise MaterializationError(message)

    return result, in_dims


def _per_example_vmap_chunk_size(
    execution: runtime_values.StandardExecution,
) -> int | None:
    if execution.operator.kind in {"fisher_vp", "sampled_fisher_vp"}:
        key = "batch.fisher_sample_batch_size"
    elif execution.operator.kind == "empirical_fisher_vp":
        key = "batch.empirical_example_batch_size"
    elif execution.operator.kind == "per_example_gradient":
        return None
    else:
        message = "per-example vmap chunk size requires a Fisher-family operator"
        raise MaterializationError(message)

    return runtime_values.optional_positive_int_setting(
        execution.candidate.settings,
        key,
        f"{key} must be a positive integer",
    )


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


def vmap_chunk_size(settings: Mapping[str, Any]) -> int:
    """Return the declared vmap chunk size.

    Returns:
        the declared vmap chunk size.
    """
    key = "vectorization.vmap_chunk_size"

    return runtime_values.required_positive_int_setting(
        settings,
        key,
        "vectorization.vmap_chunk_size must be a positive integer",
        missing_message=(
            "vectorization.mode=vmap requires vectorization.vmap_chunk_size"
        ),
    )


def vector_tree_in_dims(
    vector: TensorTree,
    settings: Mapping[str, Any],
) -> Any:
    """Return vmap in_dims for the vector tree.

    Returns:
        vmap in_dims for the vector tree.

    Raises:
        MaterializationError: If the declared inputs are invalid.
    """
    raw_in_dims = settings.get("vectorization.in_dims")

    if raw_in_dims is None:
        message = "vectorized vector inputs require vectorization.in_dims"
        raise MaterializationError(message)

    return validate_vector_tree_in_dims(vector, raw_in_dims)


def validate_vector_tree_in_dims(vector: TensorTree, raw_in_dims: Any) -> Any:
    """Validate vmap in_dims against the vector tree.

    Returns:
        Validate vmap in_dims against the vector tree.

    Raises:
        MaterializationError: If the declared inputs are invalid.
    """
    if isinstance(vector, torch.Tensor):
        return runtime_values.validate_vector_tensor_in_dim(vector, raw_in_dims)

    if runtime_values.is_tensor_tree_dict(vector):
        return _validate_vector_dict_in_dims(vector, raw_in_dims)

    if runtime_values.is_tensor_tree_tuple(vector):
        return _validate_vector_tuple_in_dims(vector, raw_in_dims)

    message = f"unsupported vector tree node: {type(vector).__name__}"
    raise MaterializationError(message)


def _validate_vector_dict_in_dims(
    vector: dict[str, TensorTree],
    raw_in_dims: Any,
) -> dict[str, Any]:
    if not isinstance(raw_in_dims, Mapping):
        message = "vectorization.in_dims must match the vector tree"
        raise MaterializationError(message)

    if set(raw_in_dims) != set(vector):
        message = "vectorization.in_dims must cover every vector key"
        raise MaterializationError(message)

    return {
        key: validate_vector_tree_in_dims(vector[key], raw_in_dims[key])
        for key in vector
    }


def _validate_vector_tuple_in_dims(
    vector: tuple[TensorTree, ...],
    raw_in_dims: Any,
) -> tuple[Any, ...]:
    if not isinstance(raw_in_dims, tuple):
        message = "vectorization.in_dims must match the vector tree"
        raise MaterializationError(message)

    if len(raw_in_dims) != len(vector):
        message = "vectorization.in_dims must cover every vector element"
        raise MaterializationError(message)

    result = []

    for value, in_dim in zip(vector, raw_in_dims, strict=True):
        result.append(validate_vector_tree_in_dims(value, in_dim))

    return tuple(result)


def _torch_func_grad(function: Callable[..., torch.Tensor]) -> Callable[..., Any]:
    return torch.func.grad(function)


def torch_func_vmap(function: Callable[..., Any], **kwargs: Any) -> Callable[..., Any]:
    """Return torch.func.vmap configured from the declared settings.

    Returns:
        torch.func.vmap configured from the declared settings.
    """
    return torch.func.vmap(function, **kwargs)


STANDARD_RUNNERS = {
    "gradient": _run_gradient,
    "jvp": _run_jvp,
    "vjp": _run_vjp,
    "hvp": _run_hvp,
    "ggnvp": ggn.run_ggnvp,
    "fisher_vp": _run_fisher_vp,
    "sampled_fisher_vp": _run_sampled_fisher_vp,
    "empirical_fisher_vp": _run_empirical_fisher_vp,
    "per_example_gradient": _run_per_example_gradient,
    "metric": metrics.run_metric,
    "sqrt_metric": metrics.run_sqrt_metric,
    "inverse_sqrt_metric": metrics.run_sqrt_metric,
    "metric_inner": metrics.run_metric_inner,
    "inverse_metric": metrics.run_inverse_metric,
    "inverse_metric_inner": metrics.run_inverse_metric_inner,
}


def _standard_materializer(
    operation_factory: RuntimeOperationFactory,
    operator: OperatorSpec | None = None,
    mmap_residency: Callable[[torch.Tensor, str], torch.Tensor] | None = None,
) -> Materializer:
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
        {"callback": "_standard_materializer.callback"},
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
        "fisher_vp": _fisher_spec_runtime_path,
        "sampled_fisher_vp": _sampled_fisher_spec_runtime_path,
        "empirical_fisher_vp": _empirical_fisher_spec_runtime_path,
        "per_example_gradient": _per_example_gradient_spec_runtime_path,
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


def _fisher_spec_runtime_path(candidate: Candidate) -> str | None:
    _require_fisher_expectation_path(candidate.settings)

    return _score_fisher_spec_runtime_path(
        candidate.settings,
        accumulation_key=runtime_values.SPEC_PATH_KEYS["fisher_vp"],
        score_path_key="fisher.score_grad_path",
        dense_path=runtime_values.FISHER_DENSE_PATH,
        block_path=runtime_values.FISHER_BLOCKWISE_SCORE_MATRIX_PATH,
        streaming_paths=runtime_values.FISHER_STREAMING_PATH_BY_SCORE_GRAD,
    )


def _require_fisher_expectation_path(settings: Mapping[str, Any]) -> None:
    expectation_key = "fisher.expectation_path"

    if expectation_key not in settings:
        message = "fisher.expectation_path is required for FisherVP rows"
        raise MaterializationError(message)

    if settings[expectation_key] != "explicit_full_expectation_score_rows":
        message = "fisher.expectation_path is unsupported"
        raise MaterializationError(message)


def _score_fisher_spec_runtime_path(
    settings: Mapping[str, Any],
    *,
    accumulation_key: str,
    score_path_key: str,
    dense_path: str,
    block_path: str,
    streaming_paths: Mapping[str, str],
) -> str | None:
    accumulation = settings.get(accumulation_key)

    if accumulation == "materialize_score_gradients":
        return _score_fisher_non_streaming_path(
            settings,
            score_path_key,
            "materialize_score_gradients",
            dense_path,
        )

    if accumulation == "blockwise_score_matrix":
        return _score_fisher_non_streaming_path(
            settings,
            score_path_key,
            "blockwise_score_matrix",
            block_path,
        )

    if accumulation not in {None, "streaming_dot_accumulate"}:
        message = f"{accumulation_key} value is not lowered: {accumulation}"
        raise MaterializationError(message)

    if accumulation is None:
        return None

    return _score_fisher_streaming_path(settings, score_path_key, streaming_paths)


def _score_fisher_non_streaming_path(
    settings: Mapping[str, Any],
    score_path_key: str,
    accumulation: str,
    path: str,
) -> str:
    if score_path_key in settings:
        message = f"{score_path_key} is not used with {accumulation}"
        raise MaterializationError(message)

    return path


def _score_fisher_streaming_path(
    settings: Mapping[str, Any],
    score_path_key: str,
    streaming_paths: Mapping[str, str],
) -> str:
    score_path = settings.get(score_path_key)

    if score_path_key not in settings:
        message = f"{score_path_key} is required for streaming rows"
        raise MaterializationError(message)

    path = streaming_paths.get(score_path) if isinstance(score_path, str) else None

    if path is not None:
        return path

    message = f"{score_path_key} value is not lowered by standard runtime: {score_path}"
    raise MaterializationError(message)


def _sampled_fisher_spec_runtime_path(candidate: Candidate) -> str | None:
    return _score_fisher_spec_runtime_path(
        candidate.settings,
        accumulation_key=runtime_values.SPEC_PATH_KEYS["sampled_fisher_vp"],
        score_path_key="sampled_fisher.score_grad_path",
        dense_path=runtime_values.SAMPLED_FISHER_DENSE_PATH,
        block_path=runtime_values.SAMPLED_FISHER_BLOCKWISE_SCORE_MATRIX_PATH,
        streaming_paths=runtime_values.SAMPLED_FISHER_STREAMING_PATH_BY_SCORE_GRAD,
    )


def _empirical_fisher_spec_runtime_path(candidate: Candidate) -> str | None:
    grad_key = runtime_values.SPEC_PATH_KEYS["empirical_fisher_vp"]
    accumulation_key = "empirical_fisher.accumulation"
    grad_path = candidate.settings.get(grad_key)
    accumulation = candidate.settings.get(accumulation_key)

    if accumulation == "materialize_per_example_gradients":
        if grad_key in candidate.settings:
            message = (
                "empirical_fisher.grad_path is not used with "
                "materialize_per_example_gradients"
            )
            raise MaterializationError(message)

        return runtime_values.EMPIRICAL_FISHER_DENSE_PATH

    if accumulation == "blockwise_gradient_matrix":
        if grad_key in candidate.settings:
            message = (
                "empirical_fisher.grad_path is not used with blockwise_gradient_matrix"
            )
            raise MaterializationError(message)

        return runtime_values.EMPIRICAL_FISHER_BLOCKWISE_GRADIENT_MATRIX_PATH

    if accumulation not in {None, "streaming_dot_accumulate"}:
        message = f"empirical_fisher.accumulation value is not lowered: {accumulation}"
        raise MaterializationError(message)

    if grad_key not in candidate.settings:
        return None

    if not isinstance(grad_path, str):
        message = "empirical_fisher.grad_path must be a string"
        raise MaterializationError(message)

    path_map = runtime_values.SPEC_PATH_TO_RUNTIME["empirical_fisher_vp"]
    path = path_map.get(grad_path)

    if path is None:
        message = f"{grad_key} value is not lowered by standard runtime: {grad_path}"
        raise MaterializationError(message)

    return path


def _per_example_gradient_spec_runtime_path(candidate: Candidate) -> str | None:
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


def _require_supported_standard_settings(
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
    unsupported = tuple(
        key
        for key in candidate.settings
        if key not in runtime_values.SUPPORTED_STANDARD_SETTINGS
    )

    if unsupported:
        message = f"standard runtime settings are unsupported: {unsupported}"
        raise MaterializationError(message)

    path = runtime_path(operator, candidate)
    _require_dtype_runtime_settings(candidate.settings)
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
    _require_gradient_graph_schedule_settings(operator, candidate.settings)
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
    _require_hvp_reuse_settings(operator, path, candidate.settings)
    _require_hvp_row_batch_size_settings(operator, path, candidate.settings)
    _require_vectorization_mode_settings(operator.kind, path, candidate.settings)
    _require_activation_runtime_settings(
        candidate.settings,
        activation_pack_hooks,
        activation_unpack_hooks,
        checkpoint_contexts,
    )
    _require_vectorization_setting_keys(operator.kind, path, candidate.settings)
    runtime_values.require_transform_admission_settings(
        operator, path, candidate.settings
    )
    _require_gradient_value_reuse_settings(operator, path, candidate.settings)
    _require_jvp_linearize_reuse_settings(operator, path, candidate.settings)
    _require_vjp_closure_reuse_settings(operator, path, candidate.settings)
    metrics.require_metric_runtime_settings(operator, path, candidate.settings)
    metrics.require_inverse_metric_factor_reuse_settings(
        operator,
        path,
        candidate.settings,
    )
    metrics.require_inverse_metric_multi_rhs_settings(
        operator, path, candidate.settings
    )
    _require_layout_runtime_settings(candidate.settings)


def _require_vectorization_mode_settings(
    operator_kind: str,
    path: str | None,
    settings: Mapping[str, Any],
) -> None:
    mode_key = "vectorization.mode"

    if mode_key not in settings:
        return

    mode = settings[mode_key]

    if mode == "manual_batch":
        if not runtime_values.supports_vector_loop(operator_kind, path):
            message = (
                "vectorization.mode=manual_batch requires a supported vector-product "
                "path"
            )
            raise MaterializationError(message)

        runtime_values.manual_vector_batch_size(settings)
        _require_vectorization_in_dims_setting(settings)

        return

    if mode == "vmap":
        if _supports_vector_vmap(operator_kind, path):
            vmap_chunk_size(settings)
            _require_vectorization_in_dims_setting(settings)

            return

        if operator_kind == "hvp":
            message = "vectorization.mode=vmap requires linearize_grad HVP"
            raise MaterializationError(message)

        message = "vectorization.mode=vmap requires vector-axis vmap lowering"
        raise MaterializationError(message)

    if mode == "single_loop":
        if not runtime_values.supports_vector_loop(operator_kind, path):
            message = (
                "vectorization.mode=single_loop requires a supported vector-product "
                "path"
            )
            raise MaterializationError(message)

        _require_vectorization_in_dims_setting(settings)

        return

    message = f"vectorization.mode is unsupported: {mode}"
    raise MaterializationError(message)


def _require_vectorization_setting_keys(
    operator_kind: str,
    path: str | None,
    settings: Mapping[str, Any],
) -> None:
    mode = settings.get("vectorization.mode")

    if "vectorization.vmap_chunk_size" in settings:
        if mode != "vmap":
            message = "vectorization.vmap_chunk_size is only supported by vmap rows"
            raise MaterializationError(message)

        if not _supports_vector_vmap(
            operator_kind,
            path,
        ):
            message = "vectorization.vmap_chunk_size is only supported by vmap rows"
            raise MaterializationError(message)

    if "vectorization.batch_size" in settings and mode != "manual_batch":
        message = "vectorization.batch_size is only supported by manual_batch rows"
        raise MaterializationError(message)

    if "vectorization.in_dims" not in settings:
        return

    if runtime_values.supports_vector_loop(operator_kind, path) and mode in {
        "single_loop",
        "manual_batch",
    }:
        return

    if _supports_vector_vmap(operator_kind, path) and mode == "vmap":
        return

    message = "vectorization.in_dims is unsupported for this operator path"
    raise MaterializationError(message)


def _supports_vector_vmap(operator_kind: str, path: str | None) -> bool:
    return path in runtime_values.VECTOR_VMAP_RUNTIME_PATHS.get(operator_kind, ())


def _require_vectorization_in_dims_setting(settings: Mapping[str, Any]) -> None:
    if "vectorization.in_dims" in settings:
        return

    message = "vectorized rows require vectorization.in_dims"
    raise MaterializationError(message)


def _require_dtype_runtime_settings(settings: Mapping[str, Any]) -> None:
    model_compute = settings.get("dtype.model_compute")
    autodiff_compute = settings.get("dtype.autodiff_compute")

    if model_compute is None or autodiff_compute is None:
        return

    if model_compute == autodiff_compute:
        return

    if settings.get("call.path") == "stateful_module":
        return

    message = (
        "split model and autodiff compute dtypes require call.path=stateful_module"
    )
    raise MaterializationError(message)


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
    model_params = _stateful_model_tree(active_params, settings)
    model_buffers = _stateful_model_tree(execution.buffers, settings)
    model_batch = _stateful_model_batch(execution.batch, settings)
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


def _stateful_model_tree(
    values: dict[str, torch.Tensor],
    settings: Mapping[str, Any],
) -> dict[str, torch.Tensor]:
    return _model_compute_tree(values, settings)


def _stateful_model_batch(
    batch: Batch,
    settings: Mapping[str, Any],
) -> Batch:
    return _model_compute_batch(batch, settings)


def _call_compiled_model_forward(
    execution: runtime_values.StandardExecution,
    compiled_model_forward: Callable[[Batch], object],
    active_params: ParameterTree,
) -> object:
    if execution.module is None:
        message = "compile.boundary=model_forward requires module"
        raise CompileSetupError(message)

    settings = execution.candidate.settings
    model_params = _stateful_model_tree(active_params, settings)
    model_buffers = _stateful_model_tree(execution.buffers, settings)
    model_batch = _stateful_model_batch(execution.batch, settings)
    slots = runtime_values.replace_module_state(
        execution.module, model_params, model_buffers
    )

    try:
        return compiled_model_forward(model_batch)
    finally:
        runtime_values.restore_module_state(slots)


def _model_compute_tree(
    values: dict[str, torch.Tensor],
    settings: Mapping[str, Any],
) -> dict[str, torch.Tensor]:
    dtype = _dtype_setting(settings, "dtype.model_compute")

    return _runtime_named_tensor_dtype(values, dtype)


def _model_compute_batch(
    batch: Batch,
    settings: Mapping[str, Any],
) -> Batch:
    dtype = _dtype_setting(settings, "dtype.model_compute")

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

    _require_per_example_batch_size_settings(path, settings)

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


def _require_per_example_batch_size_settings(
    path: str | None,
    settings: Mapping[str, Any],
) -> None:
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


def _require_gradient_graph_schedule_settings(
    operator: OperatorSpec,
    settings: Mapping[str, Any],
) -> None:
    graph_schedule = settings.get("gradient.graph_schedule")

    if graph_schedule is None:
        return

    if operator.kind != "gradient":
        message = "gradient.graph_schedule applies only to gradient rows"
        raise MaterializationError(message)

    if graph_schedule not in {"build_once", "rebuild_per_call"}:
        message = f"gradient.graph_schedule is unsupported: {graph_schedule}"
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


def _require_gradient_value_reuse_settings(
    operator: OperatorSpec,
    path: str,
    settings: Mapping[str, Any],
) -> None:
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


def _require_jvp_linearize_reuse_settings(
    operator: OperatorSpec,
    path: str,
    settings: Mapping[str, Any],
) -> None:
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


def _require_vjp_closure_reuse_settings(
    operator: OperatorSpec,
    path: str,
    settings: Mapping[str, Any],
) -> None:
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


def _require_layout_runtime_settings(settings: Mapping[str, Any]) -> None:
    flatten_order = settings.get("layout.flatten_order")

    if flatten_order is not None and flatten_order != "canonical_parameter_order":
        message = f"layout.flatten_order is unsupported: {flatten_order}"
        raise MaterializationError(message)

    _layout_tree_input(settings, "layout.params")
    _layout_tree_input(settings, "layout.vector")
    _layout_output(settings)
    _layout_single_value(
        settings,
        "layout.aliasing",
        "preserve_tied_weight_aliases",
    )
    _layout_single_value(
        settings,
        "layout.parametrizations",
        "preserve_active_parametrizations",
    )
    layout_vector_ops(settings)


def _layout_tree_input(settings: Mapping[str, Any], key: str) -> None:
    value = settings.get(key)

    if value is None or value == "parameter_tree":
        return

    if value in {"flat_contiguous", "per_layer_flat", "per_block_flat"}:
        return

    message = f"{key}={value} requires tree reconstruction support"
    raise MaterializationError(message)


def _layout_output(settings: Mapping[str, Any]) -> str:
    value = settings.get("layout.output")

    if value is None or value == "parameter_tree":
        return "parameter_tree"

    if value == "flat_contiguous":
        return "flat_contiguous"

    if value in {"per_layer_flat", "per_block_flat"}:
        return "parameter_tree"

    message = f"layout.output is unsupported: {value}"
    raise MaterializationError(message)


def _layout_single_value(
    settings: Mapping[str, Any],
    key: str,
    expected: str,
) -> None:
    value = settings.get(key)

    if value is None or value == expected:
        return

    message = f"{key} is unsupported: {value}"
    raise MaterializationError(message)


def layout_vector_ops(settings: Mapping[str, Any]) -> str:
    """Return the declared layout.vector_ops setting value.

    Returns:
        the declared layout.vector_ops setting value.

    Raises:
        MaterializationError: If the declared inputs are invalid.
    """
    value = settings.get("layout.vector_ops")

    if value is None or value == "python_loop":
        return "python_loop"

    if value == "foreach":
        return "foreach"

    message = f"layout.vector_ops is unsupported: {value}"
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

    if layout_vector_ops(settings) == "foreach":
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

    if layout_vector_ops(settings) == "foreach":
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


def matmul_runtime(
    settings: Mapping[str, Any],
    left: torch.Tensor,
    right: torch.Tensor,
) -> torch.Tensor:
    """Return the runtime matrix product under the declared precision.

    Returns:
        the runtime matrix product under the declared precision.
    """
    left = runtime_intermediate_tensor(left, settings)
    right = runtime_intermediate_tensor(right, settings)

    return accumulation_tensor(left, settings) @ accumulation_tensor(
        right,
        settings,
    )


def _tree_scale_runtime(
    settings: Mapping[str, Any],
    tree: TensorTree,
    scale: float,
) -> TensorTree:
    tree = runtime_intermediate_tree(tree, settings)

    if layout_vector_ops(settings) == "foreach":
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
    dtype = _dtype_setting(settings, "dtype.intermediate")

    if dtype is None or not tensor.is_floating_point():
        return tensor

    return tensor.to(dtype=dtype)


def _hvp_row_batch_size(settings: Mapping[str, Any]) -> int | None:
    key = "batch.hvp_row_batch_size"
    return runtime_values.optional_positive_int_setting(
        settings,
        key,
        f"{key} must be a positive integer",
    )


def _require_hvp_row_batch_size_settings(
    operator: OperatorSpec,
    path: str,
    settings: Mapping[str, Any],
) -> None:
    batch_size = _hvp_row_batch_size(settings)

    if batch_size is None:
        return

    if operator.kind != "hvp":
        message = "batch.hvp_row_batch_size applies only to HVP rows"
        raise MaterializationError(message)

    if path != runtime_values.HVP_REFERENCE_PATH:
        message = "batch.hvp_row_batch_size requires reverse_over_reverse"
        raise MaterializationError(message)


def _require_hvp_reuse_settings(
    operator: OperatorSpec,
    path: str,
    settings: Mapping[str, Any],
) -> None:
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


def _runtime_params(
    params: ParameterTree,
    settings: Mapping[str, Any],
    parameter_surface: ParameterSurface | None,
) -> ParameterTree:
    runtime_values.require_parameter_surface_runtime_settings(
        parameter_surface, settings
    )
    dtype = _parameter_dtype(settings)
    result = _runtime_named_tensor_dtype(params, dtype)
    result = _runtime_named_tensor_contiguity(result, settings)

    if settings.get("layout.params") == "flat_contiguous":
        _require_alias_safe_parameter_layout(result, settings)

        return runtime_values.wrap_flat_parameter_tree(
            result,
            runtime_values.flatten_vector(result).contiguous(),
        )

    if settings.get("layout.params") in {"per_layer_flat", "per_block_flat"}:
        _require_alias_safe_parameter_layout(result, settings)

    return _runtime_grouped_parameter_layout(
        result,
        settings,
        "layout.params",
        parameter_surface,
    )


def _runtime_grouped_parameter_layout(
    tree: ParameterTree,
    settings: Mapping[str, Any],
    key: str,
    parameter_surface: ParameterSurface | None,
) -> ParameterTree:
    groups = _parameter_layout_groups(settings, key, parameter_surface)

    if groups is None:
        return tree

    return runtime_values.wrap_grouped_parameter_tree(tree, groups, key)


def _parameter_layout_groups(
    settings: Mapping[str, Any],
    key: str,
    parameter_surface: ParameterSurface | None,
) -> tuple[tuple[str, ...], ...] | None:
    layout = settings.get(key)

    if layout in {None, "parameter_tree", "flat_contiguous"}:
        return None

    if layout == "per_layer_flat":
        return runtime_values.declared_parameter_groups(
            parameter_surface, "layer_groups", key
        )

    if layout == "per_block_flat":
        return runtime_values.declared_parameter_groups(
            parameter_surface, "block_groups", key
        )

    return None


def _runtime_grouped_output_layout(
    tree: TensorTree,
    settings: Mapping[str, Any],
    parameter_surface: ParameterSurface | None,
) -> TensorTree:
    groups = _parameter_layout_groups(settings, "layout.output", parameter_surface)

    if groups is None:
        return tree

    result = runtime_values.parameter_tree_from_tensor_tree(
        tree,
        "grouped output layout",
    )

    return runtime_values.wrap_grouped_parameter_tree(result, groups, "layout.output")


def _runtime_buffers(
    buffers: BufferTree,
    settings: Mapping[str, Any],
) -> BufferTree:
    dtype = _parameter_dtype(settings)

    if dtype is None:
        return _runtime_named_tensor_contiguity(buffers, settings)

    return _runtime_named_tensor_contiguity(
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
    dtype = _batch_dtype(settings)
    metric_factor_dtype = _dtype_setting(settings, "dtype.metric_factor")
    metric_factor_residency = settings.get("memory.factor_residency")

    if (
        dtype is None
        and metric_factor_dtype is None
        and metric_factor_residency is None
    ):
        return _runtime_batch_after_contiguity(
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
        return _runtime_batch_after_contiguity(
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

    return _runtime_batch_after_contiguity(
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

    if _uses_declared_batch_layout(candidate.settings):
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


def _uses_declared_batch_layout(settings: Mapping[str, Any]) -> bool:
    return (
        settings.get("input.batch_layout")
        in {"packed_with_inverse_permutation", "variable_length"}
        or settings.get("input.length_grouping") == "exact_length_bucket"
        or settings.get("schedule.per_token") == "packed"
    )


def _runtime_batch_after_contiguity(
    batch: Batch,
    settings: Mapping[str, Any],
    *,
    move_input_residency: bool,
) -> Batch:
    result = _runtime_batch_contiguity(batch, settings)

    if move_input_residency:
        result = _runtime_batch_input_residency(result, settings)

    return _runtime_batch_teacher_outputs(result, settings)


def _runtime_batch_input_residency(
    batch: Batch,
    settings: Mapping[str, Any],
) -> Batch:
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


def _runtime_batch_teacher_outputs(
    batch: Batch,
    settings: Mapping[str, Any],
) -> Batch:
    value = settings.get("teacher_outputs")

    if value is None:
        return batch

    if "teacher_outputs" not in batch:
        message = "teacher_outputs batch field is required"
        raise MaterializationError(message)

    result = dict(batch)

    if value == "precomputed_cpu":
        result["teacher_outputs"] = _teacher_outputs_to_device(
            batch["teacher_outputs"],
            torch.device("cpu"),
        )

        return result

    if value == "precomputed_cpu_pinned":
        result["teacher_outputs"] = _teacher_outputs_pin_cpu(batch["teacher_outputs"])

        return result

    if value == "precomputed_gpu":
        if not torch.cuda.is_available():
            message = "precomputed_gpu teacher outputs require CUDA"
            raise MaterializationError(message)

        result["teacher_outputs"] = _teacher_outputs_to_device(
            batch["teacher_outputs"],
            torch.device("cuda"),
        )

        return result

    if value == "recomputed_with_equality_check":
        _require_teacher_output_tree(batch["teacher_outputs"])

        return result

    message = f"teacher_outputs is unsupported: {value}"
    raise MaterializationError(message)


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
    _require_teacher_output_tree(fixed_outputs)
    settings = execution.candidate.settings
    recomputed_outputs = execution.teacher_objective(
        _model_compute_tree(execution.params, settings),
        _model_compute_tree(execution.buffers, settings),
        _model_compute_batch(execution.batch, settings),
        execution.context,
    )
    _require_teacher_outputs_match(fixed_outputs, recomputed_outputs)
    batch = dict(execution.batch)
    batch["teacher_outputs"] = recomputed_outputs

    return dataclasses.replace(execution, batch=batch)


def _teacher_outputs_to_device(value: Any, device: torch.device) -> TensorTree:
    return runtime_values.runtime_nested_tensor_value(
        value,
        lambda tensor: tensor.to(device=device),
        error_message="teacher_outputs batch field must be a tensor tree",
    )


def _require_teacher_output_tree(value: Any) -> None:
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


def _teacher_outputs_pin_cpu(value: Any) -> TensorTree:
    cpu_value = _teacher_outputs_to_device(value, torch.device("cpu"))

    return tree_map(runtime_values.pin_cpu_tensor, cpu_value)


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
    dtype = _dtype_setting(settings, "dtype.vector")

    if dtype is None:
        result = vector
    else:
        result = tree_map(lambda tensor: tensor.to(dtype=dtype), vector)

    result = _runtime_vector_residency(result, settings, mmap_residency)
    result = _runtime_vector_layout(
        result,
        settings,
        vector if template is None else template,
        parameter_surface,
    )

    return _runtime_tree_contiguity(result, settings)


def _runtime_vector_layout(
    vector: TensorTree,
    settings: Mapping[str, Any],
    template: TensorTree,
    parameter_surface: ParameterSurface | None,
) -> TensorTree:
    layout = settings.get("layout.vector")

    if layout is None or layout == "parameter_tree":
        return vector

    if isinstance(vector, torch.Tensor):
        flat_vector = vector.reshape(-1).contiguous()
    else:
        flat_vector = runtime_values.flatten_vector_like(template, vector).contiguous()

    if layout == "flat_contiguous":
        return runtime_values.wrap_flat_vector(template, flat_vector)

    wrapped = runtime_values.wrap_flat_vector(template, flat_vector)
    parameter_tree = runtime_values.parameter_tree_from_tensor_tree(
        wrapped,
        f"layout.vector={layout}",
    )

    return _runtime_grouped_parameter_layout(
        parameter_tree,
        settings,
        "layout.vector",
        parameter_surface,
    )


def _runtime_named_tensor_dtype(
    tree: dict[str, torch.Tensor],
    dtype: torch.dtype | None,
) -> dict[str, torch.Tensor]:
    if dtype is None:
        return tree

    return runtime_values.runtime_named_tensor_map_preserve_alias(
        tree,
        lambda tensor: tensor.to(dtype=dtype),
    )


def _require_alias_safe_parameter_layout(
    params: ParameterTree,
    settings: Mapping[str, Any],
) -> None:
    if not runtime_values.preserves_parameter_aliases(settings):
        return

    if not runtime_values.has_parameter_aliases(params):
        return

    message = "non-tree parameter layout cannot preserve tied parameter aliases"
    raise MaterializationError(message)


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


def _runtime_output(
    output: TensorTree,
    settings: Mapping[str, Any],
    parameter_surface: ParameterSurface | None = None,
) -> TensorTree:
    dtype = _dtype_setting(settings, "dtype.output")

    if dtype is not None:
        output = tree_map(lambda tensor: tensor.to(dtype=dtype), output)

    if _layout_output(settings) == "flat_contiguous":
        return runtime_values.flatten_vector(output).contiguous()

    return _runtime_grouped_output_layout(output, settings, parameter_surface)


def _standard_output_buffer(
    execution: runtime_values.StandardExecution,
) -> TensorTree | None:
    if execution.candidate.settings.get("memory.output_buffers") != "preallocated":
        return None

    template = _standard_output_template(execution)
    runtime_template = _runtime_output(
        template,
        execution.candidate.settings,
        execution.parameter_surface,
    )

    return tree_map(torch.empty_like, runtime_template)


def _composition_output_buffer(
    settings: Mapping[str, Any],
    vector: TensorTree,
) -> TensorTree | None:
    if settings.get("memory.output_buffers") != "preallocated":
        return None

    template = runtime_vector(vector, settings)
    runtime_template = _runtime_output(template, settings)

    return tree_map(torch.empty_like, runtime_template)


def _standard_output_template(
    execution: runtime_values.StandardExecution,
) -> TensorTree:
    kind = execution.operator.kind

    if kind == "jvp":
        return _jvp_output_template(execution)

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


def _jvp_output_template(execution: runtime_values.StandardExecution) -> TensorTree:
    function = runtime_values.function_objective(
        execution.operator,
        execution.function_objectives,
    )

    def callback() -> TensorTree:
        return call_function_objective(execution, function, execution.params)

    return run_with_backend_settings(
        execution.candidate.settings,
        lambda: runtime_values.run_with_call_grad_mode(
            execution.candidate.settings, callback
        ),
    )


def accumulation_tensor(
    tensor: torch.Tensor,
    settings: Mapping[str, Any],
) -> torch.Tensor:
    """Return the accumulation tensor for the declared accumulation dtype.

    Returns:
        the accumulation tensor for the declared accumulation dtype.
    """
    dtype = _dtype_setting(settings, "dtype.accumulation")

    if dtype is None:
        return tensor

    return tensor.to(dtype=dtype)


def _accumulation_tree(tree: TensorTree, settings: Mapping[str, Any]) -> TensorTree:
    dtype = _dtype_setting(settings, "dtype.accumulation")

    if dtype is None:
        return tree

    return tree_map(lambda tensor: tensor.to(dtype=dtype), tree)


def _runtime_tree_contiguity(
    tree: TensorTree,
    settings: Mapping[str, Any],
) -> TensorTree:
    if not _layout_contiguity_enabled(settings):
        return tree

    return tree_map(lambda tensor: tensor.contiguous(), tree)


def _runtime_named_tensor_contiguity(
    tree: dict[str, torch.Tensor],
    settings: Mapping[str, Any],
) -> dict[str, torch.Tensor]:
    if not _layout_contiguity_enabled(settings):
        return tree

    return runtime_values.runtime_named_tensor_map_preserve_alias(
        tree,
        lambda tensor: tensor.contiguous(),
    )


def _runtime_batch_contiguity(
    batch: Batch,
    settings: Mapping[str, Any],
) -> Batch:
    if not _layout_contiguity_enabled(settings):
        return batch

    return {
        key: runtime_values.runtime_nested_tensor_value(
            value, lambda tensor: tensor.contiguous()
        )
        for key, value in batch.items()
    }


def _layout_contiguity_enabled(settings: Mapping[str, Any]) -> bool:
    value = settings.get("layout.contiguity")

    if value is None or value == "preserve_existing_strides":
        return False

    if value == "contiguous":
        return True

    message = f"layout.contiguity is unsupported: {value}"
    raise MaterializationError(message)


def _parameter_dtype(settings: Mapping[str, Any]) -> torch.dtype | None:
    autodiff_dtype = _dtype_setting(settings, "dtype.autodiff_compute")

    if autodiff_dtype is not None:
        return autodiff_dtype

    return _dtype_setting(settings, "dtype.parameter_storage")


def _batch_dtype(settings: Mapping[str, Any]) -> torch.dtype | None:
    autodiff_dtype = _dtype_setting(settings, "dtype.autodiff_compute")

    if autodiff_dtype is not None:
        return autodiff_dtype

    return _dtype_setting(settings, "dtype.intermediate")


def _dtype_setting(settings: Mapping[str, Any], key: str) -> torch.dtype | None:
    dtype_name = settings.get(key)

    if dtype_name is None:
        return None

    if not isinstance(dtype_name, str):
        message = f"{key} must be a string"
        raise MaterializationError(message)

    if dtype_name == "bf16":
        return torch.bfloat16

    if dtype_name == "fp16":
        return torch.float16

    if dtype_name == "fp32":
        return torch.float32

    if dtype_name == "fp8_when_supported":
        return _fp8_dtype()

    message = f"{key} is unsupported by standard runtime: {dtype_name}"
    raise MaterializationError(message)


def _fp8_dtype() -> torch.dtype:
    if hasattr(torch, "float8_e4m3fn"):
        return torch.float8_e4m3fn

    message = "fp8_when_supported requires PyTorch FP8 dtype support"
    raise MaterializationError(message)


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

    matmul_precision = _matmul_precision_setting(settings)
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


def _matmul_precision_setting(settings: Mapping[str, Any]) -> str | None:
    key = "numeric.float32_matmul_precision"
    value = settings.get(key)

    if value is None:
        return None

    if not isinstance(value, str):
        message = f"{key} must be a string"
        raise MaterializationError(message)

    if value not in {"highest", "high", "medium"}:
        message = f"{key} is unsupported by standard runtime: {value}"
        raise MaterializationError(message)

    return value


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

    settings.update(_fisher_anchor_settings(operator, path))
    settings.update(_sampled_fisher_anchor_settings(operator, candidate, path))
    settings.update(_per_example_gradient_anchor_settings(operator, path))
    settings.update(ggn.ggn_anchor_settings(operator, path))
    settings.update(metrics.metric_inner_anchor_settings(operator, candidate, path))
    settings.update(
        metrics.inverse_metric_inner_anchor_settings(operator, candidate, path)
    )

    settings.update(_anchor_admission_settings(path))

    return settings


def _fisher_anchor_settings(
    operator: OperatorSpec,
    path: str,
) -> dict[str, Any]:
    if operator.kind != "fisher_vp":
        return {}

    if path != runtime_values.FISHER_SCORE_GRADIENT_LOOP_PATH:
        return {}

    return {
        "fisher.expectation_path": "explicit_full_expectation_score_rows",
        "fisher.score_grad_path": "torch_autograd_grad_loop",
    }


def _sampled_fisher_anchor_settings(
    operator: OperatorSpec,
    candidate: Candidate,
    path: str,
) -> dict[str, Any]:
    if operator.kind != "sampled_fisher_vp":
        return {}

    if path != runtime_values.SAMPLED_FISHER_SCORE_GRADIENT_LOOP_PATH:
        return {}

    sample_source_key = "sampled_fisher.sample_source"
    exact_check_key = "sampled_fisher.exact_fisher_check"
    settings = {
        "sampled_fisher.score_grad_path": "torch_autograd_grad_loop",
    }

    for key in (sample_source_key, exact_check_key):
        if key not in candidate.settings:
            message = f"{key} is required for sampled Fisher anchors"
            raise MaterializationError(message)

        settings[key] = candidate.settings[key]

    return settings


def _per_example_gradient_anchor_settings(
    operator: OperatorSpec,
    path: str,
) -> dict[str, Any]:
    if operator.kind != "per_example_gradient":
        return {}

    if path != runtime_values.PER_EXAMPLE_GRADIENT_LOOP_PATH:
        return {}

    return {"per_example_gradient.accumulation": "stacked_leading_axis"}


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
        return _fisher_anchor_path(operator)

    path = runtime_values.STANDARD_ANCHOR_PATHS.get(operator.kind)

    if path is not None:
        return path

    message = f"standard anchor does not support operator kind: {operator.kind}"
    raise MaterializationError(message)


def _fisher_anchor_path(operator: OperatorSpec) -> str:
    distribution = runtime_values.operator_semantic(operator, "distribution")

    if distribution == "explicit_score_gradients":
        return runtime_values.FISHER_SCORE_GRADIENT_LOOP_PATH

    message = "standard Fisher anchor does not support declared semantics"
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
        _model_compute_tree(params, settings),
        _model_compute_tree(execution.buffers, settings),
        _model_compute_batch(active_batch, settings),
        execution.context,
    )

    return runtime_values.checked_function_output(
        execution.candidate.settings,
        output,
        "function objective output",
    )


def _empirical_fisher_normalization(
    batch: Batch,
    operator: OperatorSpec,
    per_example_gradients: torch.Tensor,
) -> float:
    return _empirical_fisher_normalization_from_count(
        batch,
        operator,
        per_example_gradients.shape[0],
    )


def _empirical_fisher_streaming_normalization(
    execution: runtime_values.StandardExecution,
) -> float:
    batch, batch_in_dims = _per_example_batch_in_dims(
        execution.batch,
        "empirical Fisher streaming",
    )
    example_count = runtime_values.per_example_batch_size(
        batch,
        batch_in_dims,
        "empirical Fisher streaming",
    )

    return _empirical_fisher_normalization_from_count(
        execution.batch,
        execution.operator,
        example_count,
    )


def _empirical_fisher_normalization_from_count(
    batch: Batch,
    operator: OperatorSpec,
    example_count: int,
) -> float:
    _require_empirical_fisher_semantics(operator)
    denominator = runtime_values.operator_semantic(operator, "denominator")

    if denominator == "num_examples":
        normalization = float(example_count)
    elif denominator == "one":
        normalization = 1.0
    elif denominator == "batch_normalization":
        normalization = runtime_values.normalization(batch, operator)
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
    example_loss_reduction = runtime_values.operator_semantic(
        operator, "example_loss_reduction"
    )

    if example_loss_reduction != "per_example":
        message = (
            "empirical Fisher example_loss_reduction is unsupported: "
            f"{example_loss_reduction}"
        )
        raise MaterializationError(message)


def _fisher_normalization(execution: runtime_values.StandardExecution) -> float:
    denominator = runtime_values.operator_semantic(execution.operator, "denominator")

    if denominator == "num_examples":
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
        return runtime_values.normalization(execution.batch, execution.operator)

    message = f"Fisher denominator is unsupported: {denominator}"
    raise MaterializationError(message)


def _sampled_fisher_normalization(execution: runtime_values.StandardExecution) -> float:
    denominator = runtime_values.operator_semantic(execution.operator, "denominator")
    sample_count = runtime_values.operator_semantic_positive_int(
        execution.operator, "sample_count"
    )

    if denominator == "num_examples":
        value = execution.batch.get("num_examples")

        if not isinstance(value, int | float):
            message = "num_examples denominator requires batch num_examples"
            raise MaterializationError(message)

        if float(value) <= 0.0:
            message = "num_examples denominator must be positive"
            raise MaterializationError(message)

        return float(value) * float(sample_count)

    if denominator == "num_tokens":
        value = execution.batch.get("num_tokens")

        if not isinstance(value, int | float):
            message = "num_tokens denominator requires batch num_tokens"
            raise MaterializationError(message)

        if float(value) <= 0.0:
            message = "num_tokens denominator must be positive"
            raise MaterializationError(message)

        return float(value) * float(sample_count)

    if denominator == "one":
        return 1.0

    if denominator == "batch_normalization":
        return runtime_values.normalization(execution.batch, execution.operator)

    message = f"sampled Fisher denominator is unsupported: {denominator}"
    raise MaterializationError(message)


def _check_sampled_fisher_exact_bound(
    execution: runtime_values.StandardExecution,
    result: torch.Tensor,
) -> None:
    exact_check = execution.candidate.settings.get("sampled_fisher.exact_fisher_check")

    if exact_check == "disabled":
        return

    if exact_check != "enabled_with_sampling_bound":
        message = f"sampled_fisher.exact_fisher_check is unsupported: {exact_check}"
        raise MaterializationError(message)

    exact = runtime_values.batch_tensor(execution.batch, "exact_fisher_vp").reshape(-1)
    runtime_values.require_finite_tensor(exact, "exact FisherVP reference")

    if exact.numel() != result.numel():
        message = "exact_fisher_vp must match sampled Fisher result width"
        raise MaterializationError(message)

    bound = _sampled_fisher_sampling_bound(execution, result)
    difference = (result.reshape(-1) - exact).norm()
    exact_norm = exact.norm()
    floor = torch.tensor(
        bound["norm_floor"],
        dtype=exact_norm.dtype,
        device=exact_norm.device,
    )
    relative = difference / torch.maximum(exact_norm, floor)
    abs_error = float(difference.item())
    rel_error = float(relative.item())

    if abs_error <= bound["max_abs_diff"] or rel_error <= bound["max_rel_diff"]:
        return

    message = (
        "sampled Fisher exact-Fisher comparison exceeded bound: "
        f"abs={abs_error}, rel={rel_error}"
    )
    raise MaterializationError(message)


def _sampled_fisher_sampling_bound(
    execution: runtime_values.StandardExecution,
    result: torch.Tensor,
) -> dict[str, float]:
    raw = execution.operator.semantics.get("sampling_bound")

    if not isinstance(raw, Mapping):
        message = "sampled Fisher sampling_bound must be a mapping"
        raise MaterializationError(message)

    if raw.get("kind") == "matrix_bernstein":
        return _sampled_fisher_matrix_bernstein_bound(execution, raw)

    if raw.get("kind") == "hutchinson_relative_variance":
        return _sampled_fisher_hutchinson_relative_bound(execution, result, raw)

    if raw.get("kind") != "abs_or_rel":
        message = (
            "sampled Fisher sampling_bound.kind must be abs_or_rel, "
            "matrix_bernstein, or hutchinson_relative_variance"
        )
        raise MaterializationError(message)

    return {
        "max_abs_diff": runtime_values.sampling_bound_float(raw, "max_abs_diff"),
        "max_rel_diff": runtime_values.sampling_bound_float(raw, "max_rel_diff"),
        "norm_floor": runtime_values.sampling_bound_float(raw, "norm_floor"),
    }


def _sampled_fisher_contributions(
    execution: runtime_values.StandardExecution,
) -> torch.Tensor:
    score_matrix = _sampled_fisher_score_matrix_for_bound(execution)
    vector = parameter_order_vector(execution)
    normalization = _sampled_fisher_normalization(execution)
    score_dot = matmul_runtime(execution.candidate.settings, score_matrix, vector)
    contributions = score_matrix * score_dot.unsqueeze(1) / normalization
    runtime_values.require_finite_tensor(
        contributions, "sampled Fisher bound contributions"
    )

    return contributions


def _sampled_fisher_score_matrix_for_bound(
    execution: runtime_values.StandardExecution,
) -> torch.Tensor:
    if "sampled_score_gradients" in execution.batch:
        matrix = runtime_values.batch_tensor(execution.batch, "sampled_score_gradients")
    else:
        matrix = _score_gradient_matrix_from_operator_row(execution)

    return _loss_scaled_score_matrix(execution, matrix)


def _sampled_fisher_centered_contributions(
    execution: runtime_values.StandardExecution,
) -> torch.Tensor:
    contributions = _sampled_fisher_contributions(execution)

    if contributions.shape[0] < runtime_values.MIN_SAMPLED_FISHER_FORMULA_BOUND_SAMPLES:
        message = "sampled Fisher formula bounds require at least two samples"
        raise MaterializationError(message)

    centered = contributions - contributions.mean(dim=0, keepdim=True)
    runtime_values.require_finite_tensor(
        centered, "sampled Fisher centered bound contributions"
    )

    return centered


def _sampled_fisher_matrix_bernstein_bound(
    execution: runtime_values.StandardExecution,
    raw: Mapping[str, Any],
) -> dict[str, float]:
    centered = _sampled_fisher_centered_contributions(execution)
    failure_probability = runtime_values.sampling_bound_probability(
        raw, "failure_probability"
    )
    log_term = math.log(2.0 / failure_probability)
    variance = float(centered.square().sum().item())
    row_norms = centered.norm(dim=1)
    range_bound = float(row_norms.max().item())
    max_abs_diff = math.sqrt(2.0 * variance * log_term)
    max_abs_diff += (2.0 / 3.0) * range_bound * log_term

    return {
        "max_abs_diff": max_abs_diff,
        "max_rel_diff": 0.0,
        "norm_floor": runtime_values.sampling_bound_float(raw, "norm_floor"),
    }


def _sampled_fisher_hutchinson_relative_bound(
    execution: runtime_values.StandardExecution,
    result: torch.Tensor,
    raw: Mapping[str, Any],
) -> dict[str, float]:
    centered = _sampled_fisher_centered_contributions(execution)
    failure_probability = runtime_values.sampling_bound_probability(
        raw, "failure_probability"
    )
    norm_floor = runtime_values.sampling_bound_float(raw, "norm_floor")
    result_norm = float(result.reshape(-1).norm().item())
    denominator = max(result_norm, norm_floor)

    if denominator <= 0.0:
        message = "sampled Fisher hutchinson bound requires positive norm scale"
        raise MaterializationError(message)

    relative_variance = float(centered.square().sum().item()) / (denominator**2)
    max_rel_diff = math.sqrt(relative_variance / failure_probability)

    return {
        "max_abs_diff": 0.0,
        "max_rel_diff": max_rel_diff,
        "norm_floor": norm_floor,
    }


def _require_sampled_fisher_semantics(
    execution: runtime_values.StandardExecution,
) -> None:
    operator = execution.operator
    runtime_values.operator_semantic_positive_int(operator, "sample_count")
    operator_sample_source = runtime_values.operator_semantic(operator, "sample_source")
    row_sample_source = execution.candidate.settings.get("sampled_fisher.sample_source")

    if row_sample_source not in {"fixed_sample_table", "fixed_seed_and_count"}:
        message = "sampled_fisher.sample_source is required"
        raise MaterializationError(message)

    if row_sample_source != operator_sample_source:
        message = "sampled_fisher.sample_source differs from operator"
        raise MaterializationError(message)

    exact_check = execution.candidate.settings.get("sampled_fisher.exact_fisher_check")

    if exact_check is None:
        message = "sampled_fisher.exact_fisher_check is required"
        raise MaterializationError(message)

    if exact_check not in {"disabled", "enabled_with_sampling_bound"}:
        message = f"sampled_fisher.exact_fisher_check is unsupported: {exact_check}"
        raise MaterializationError(message)

    if exact_check == "enabled_with_sampling_bound":
        raw = operator.semantics.get("sampling_bound")

        if not isinstance(raw, Mapping):
            message = "sampled Fisher sampling_bound must be a mapping"
            raise MaterializationError(message)

        if raw.get("kind") not in {
            "abs_or_rel",
            "matrix_bernstein",
            "hutchinson_relative_variance",
        }:
            message = (
                "sampled Fisher exact-Fisher check requires declared sampling_bound"
            )
            raise MaterializationError(message)

    score_reduction = runtime_values.operator_semantic(operator, "score_reduction")

    if score_reduction != "none":
        message = f"sampled Fisher score_reduction is unsupported: {score_reduction}"
        raise MaterializationError(message)


def _require_fisher_semantics(
    operator: OperatorSpec,
    required: Mapping[str, str],
) -> None:
    for key, expected in required.items():
        actual = runtime_values.operator_semantic(operator, key)

        if actual != expected:
            message = f"Fisher semantic field mismatch: {key}"
            raise MaterializationError(message)


def _require_valid_fisher_semantics(operator: OperatorSpec) -> None:
    distribution = runtime_values.operator_semantic(operator, "distribution")

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
            "sample_space": "terms",
            "score_reduction": "none",
        },
    )


FISHER_FAMILY_RUNTIME_ROWS = {
    "fisher_vp": {
        "paths": runtime_values.FISHER_VECTOR_VMAP_PATHS_BY_KIND["fisher_vp"],
        "streaming_paths": runtime_values.FISHER_STREAMING_PRODUCT_PATHS_BY_KIND[
            "fisher_vp"
        ],
        "blockwise_path": runtime_values.FISHER_BLOCKWISE_PATH_BY_KIND["fisher_vp"],
        "dense_path": runtime_values.FISHER_DENSE_PATH_BY_KIND["fisher_vp"],
        "block_batch_key": "score_gradient_blocks",
        "matrix_batch_key": "score_gradients",
        "streaming_matrix": _score_gradient_matrix_from_operator_row,
        "streaming_normalization": _fisher_normalization,
        "blockwise_normalization": _fisher_normalization,
        "matrix_normalization": _fisher_score_matrix_normalization,
        "streaming_label": "Fisher",
        "vector_label": "Fisher vector",
        "result_label": "Fisher result",
        "blockwise_label": "score_gradients",
        "require_streaming": _require_explicit_score_fisher_execution,
        "require_matrix": _require_valid_fisher_execution,
        "check_result": _skip_score_matrix_result_check,
        "vmap_error_message": (
            "Fisher vector vmap requires a score-gradient matrix path"
        ),
        "precheck": _skip_score_fisher_requirement,
    },
    "sampled_fisher_vp": {
        "paths": runtime_values.FISHER_VECTOR_VMAP_PATHS_BY_KIND["sampled_fisher_vp"],
        "streaming_paths": runtime_values.FISHER_STREAMING_PRODUCT_PATHS_BY_KIND[
            "sampled_fisher_vp"
        ],
        "blockwise_path": runtime_values.FISHER_BLOCKWISE_PATH_BY_KIND[
            "sampled_fisher_vp"
        ],
        "dense_path": runtime_values.FISHER_DENSE_PATH_BY_KIND["sampled_fisher_vp"],
        "block_batch_key": "sampled_score_gradient_blocks",
        "matrix_batch_key": "sampled_score_gradients",
        "streaming_matrix": _score_gradient_matrix_from_operator_row,
        "streaming_normalization": _sampled_fisher_normalization,
        "blockwise_normalization": _sampled_fisher_normalization,
        "matrix_normalization": _sampled_fisher_score_matrix_normalization,
        "streaming_label": "sampled Fisher",
        "vector_label": "sampled Fisher vector",
        "result_label": "sampled Fisher result",
        "blockwise_label": "sampled_score_gradients",
        "require_streaming": _skip_score_fisher_requirement,
        "require_matrix": _skip_score_fisher_requirement,
        "check_result": _check_sampled_fisher_exact_bound,
        "vmap_error_message": (
            "sampled Fisher vector vmap requires a score-gradient matrix path"
        ),
        "precheck": _require_sampled_fisher_semantics,
    },
    "empirical_fisher_vp": {
        "paths": runtime_values.FISHER_VECTOR_VMAP_PATHS_BY_KIND["empirical_fisher_vp"],
        "streaming_paths": runtime_values.FISHER_STREAMING_PRODUCT_PATHS_BY_KIND[
            "empirical_fisher_vp"
        ],
        "blockwise_path": runtime_values.FISHER_BLOCKWISE_PATH_BY_KIND[
            "empirical_fisher_vp"
        ],
        "dense_path": runtime_values.FISHER_DENSE_PATH_BY_KIND["empirical_fisher_vp"],
        "block_batch_key": "per_example_gradient_blocks",
        "matrix_batch_key": "per_example_gradients",
        "streaming_matrix": _score_gradient_matrix_from_operator_row,
        "streaming_normalization": _empirical_fisher_streaming_normalization,
        "blockwise_normalization": _empirical_fisher_blockwise_normalization,
        "matrix_normalization": _empirical_fisher_score_matrix_normalization,
        "streaming_label": "empirical Fisher",
        "vector_label": "empirical Fisher vector",
        "result_label": "empirical Fisher result",
        "blockwise_label": "per_example_gradients",
        "require_streaming": _skip_score_fisher_requirement,
        "require_matrix": _skip_score_fisher_requirement,
        "check_result": _skip_score_matrix_result_check,
        "vmap_error_message": (
            "empirical Fisher vector vmap requires a score-gradient matrix path"
        ),
        "precheck": _skip_score_fisher_requirement,
    },
}
