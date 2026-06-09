"""Composition lowerings for the standard runtime.

Sequential, linear-combination, scaled-identity, and source
composition execution, child lowering and validation modes, and
composition reference checks.
"""

import dataclasses
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import torch

from vptune import memory, metrics, runtime, runtime_values, vectorization
from vptune.checks import (
    tree_error_measurements,
    validate_thresholds,
)
from vptune.data import (
    PACKAGE_VERSION,
    Batch,
    CallableOperationFactory,
    CallableReferenceCheck,
    Candidate,
    CandidateAdmitter,
    CandidateOperation,
    OperationFactory,
    OperatorSpec,
    ReferenceCheck,
    ReferenceChildResult,
    ReferenceResult,
    RuntimeConfig,
)
from vptune.errors import (
    CompileSetupError,
    MaterializationError,
    ReferenceFailedError,
)
from vptune.tensor_tree import (
    TensorTree,
    tree_map,
    tree_signature,
)


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
        runtime.require_supported_standard_settings(operator, candidate)
        runtime_values.require_path(
            operator.kind,
            runtime.runtime_path(operator, candidate),
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
        runtime.require_loss_scaling_settings(operator, candidate.settings)
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
            prepared_batch = runtime.runtime_batch(batch, candidate.settings)
            result = runtime.runtime_vector(vector, candidate.settings)

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
                scaled_output = runtime.loss_scaled_output_source(
                    operator,
                    candidate.settings,
                    component_output,
                )

                return runtime.loss_unscaled_output(
                    operator,
                    candidate.settings,
                    scaled_output,
                )

            return runtime.run_with_backend_settings(
                candidate.settings,
                lambda: runtime_values.run_with_call_grad_mode(
                    candidate.settings,
                    lambda: runtime_values.runtime_output_to_buffer(
                        runtime.runtime_output(
                            run_scaled_components(),
                            candidate.settings,
                        ),
                        output_buffer,
                    ),
                ),
            )

        return runtime.compile_operation(operator, candidate.settings, operation)

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
                settings=runtime.anchor_settings(
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
            runtime.semantic_measurements(operator, batch, vector, candidate_output)
        )
        effective_thresholds = runtime.reference_thresholds_for_operator(
            operator, thresholds
        )
        runtime.require_thresholds_for_measurements(measurements, effective_thresholds)
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
    runtime.require_supported_standard_settings(operator, candidate)
    runtime.require_supported_standard_settings(operator, anchor_candidate)
    _require_composition_execution_settings(candidate.settings)
    runtime.require_loss_scaling_settings(operator, candidate.settings)
    runtime_values.require_path(
        operator.kind,
        runtime.runtime_path(operator, candidate),
        (runtime_values.COMPOSITION_PATH,),
    )
    runtime_values.require_path(
        operator.kind,
        runtime.runtime_path(operator, anchor_candidate),
        (runtime_values.COMPOSITION_PATH,),
    )
    candidate_batch = runtime.runtime_batch(batch, candidate.settings)
    anchor_batch = runtime.runtime_batch(batch, anchor_candidate.settings)
    candidate_result = runtime.runtime_vector(vector, candidate.settings)
    anchor_result = runtime.runtime_vector(vector, anchor_candidate.settings)
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
        candidate_result = runtime.loss_scaled_output_source(
            operator,
            candidate.settings,
            candidate_result,
        )
        candidate_result = runtime.loss_unscaled_output(
            operator,
            candidate.settings,
            candidate_result,
        )
        candidate_result = runtime.runtime_output(candidate_result, candidate.settings)
        anchor_result = runtime.runtime_output(anchor_result, anchor_candidate.settings)

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

    candidate_result = runtime.loss_scaled_output_source(
        operator,
        candidate.settings,
        candidate_result,
    )
    candidate_result = runtime.loss_unscaled_output(
        operator,
        candidate.settings,
        candidate_result,
    )
    candidate_result = runtime.runtime_output(candidate_result, candidate.settings)
    anchor_result = runtime.runtime_output(anchor_result, anchor_candidate.settings)

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
            return runtime.tree_scale_runtime(
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
                scaled = runtime.tree_scale_runtime(
                    settings,
                    term_output,
                    weighted["coefficient"],
                )

                if result is None:
                    result = scaled
                else:
                    result = runtime.tree_add_runtime(settings, result, scaled)

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

    return vectorization.run_tensor_tree_by_vectorization_mode(
        vector, settings, vector_runner
    )


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

    return memory.tree_residency(result, residency, "memory.intermediate_residency")


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

    runtime.validate_compile_cache_state(settings)
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

    return runtime.runtime_batch(batch, settings)


def _composition_child_warm_vector(
    settings: Mapping[str, Any],
    vector: TensorTree,
) -> TensorTree | None:
    if settings.get("compile.cache_state") != "warm_cache":
        return None

    warm_vector = runtime.runtime_vector(vector, settings)
    mode = settings.get("vectorization.mode")

    if mode == "single_loop":
        vector_in_dims = vectorization.vector_tree_in_dims(warm_vector, settings)

        return runtime_values.vector_tree_select(warm_vector, vector_in_dims, 0)

    if mode == "manual_batch":
        vector_in_dims = vectorization.vector_tree_in_dims(warm_vector, settings)
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
    return runtime.compiled_callable(settings, component, use_backend_settings=False)


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
        materializer=runtime.standard_materializer(operation_factory),
        axis_registry=axis_registry,
        reference_check_name="composition_anchor",
        signature=runtime_signature,
    )


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


def _composition_output_buffer(
    settings: Mapping[str, Any],
    vector: TensorTree,
) -> TensorTree | None:
    if settings.get("memory.output_buffers") != "preallocated":
        return None

    template = runtime.runtime_vector(vector, settings)
    runtime_template = runtime.runtime_output(template, settings)

    return tree_map(torch.empty_like, runtime_template)
