"""Fisher-family lowerings for the standard runtime.

FisherVP, sampled FisherVP, and empirical FisherVP paths, score
matrices, streaming score-gradient products, and their runtime
validation helpers.
"""

import dataclasses
import math
from collections.abc import Callable, Iterator, Mapping
from typing import Any

import torch

from vptune import derivatives, runtime, runtime_values, vectorization
from vptune.data import (
    Batch,
    Candidate,
    OperatorSpec,
    ParameterSurface,
)
from vptune.errors import (
    MaterializationError,
)
from vptune.tensor_tree import (
    TensorTree,
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


def prepare_score_matrix_compile_boundary(
    execution: runtime_values.StandardExecution,
    settings: Mapping[str, Any],
    builder: Callable[[], torch.Tensor],
) -> runtime_values.StandardExecution:
    """Prepare score matrix compile boundary.

    Returns:
        The score matrix compile boundary result.
    """
    runtime.require_compiled_execution(execution, settings)
    compiled_score_matrix = _compiled_tensor_operation(settings, builder)

    return dataclasses.replace(
        execution,
        compiled_score_matrix=compiled_score_matrix,
    )


def score_matrix_compile_boundary_builder(
    execution: runtime_values.StandardExecution,
    boundary: str,
) -> Callable[[], torch.Tensor] | None:
    """Return the score matrix compile boundary builder.

    Returns:
        The score matrix compile boundary builder.
    """
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


def _compiled_tensor_operation(
    settings: Mapping[str, Any],
    operation: Callable[[], torch.Tensor],
) -> Callable[[], torch.Tensor]:
    compiled = runtime.compiled_callable(
        settings,
        operation,
        use_backend_settings=False,
    )
    runtime.warm_compiled_cache(settings, compiled)

    return compiled


def score_matrix_compile_boundary_supported(
    operator_kind: str,
    boundary: str,
    settings: Mapping[str, Any],
) -> bool:
    """Return the score matrix compile boundary supported.

    Returns:
        The score matrix compile boundary supported.
    """
    row = runtime_values.SCORE_MATRIX_COMPILE_ROWS[operator_kind]

    if boundary != row.boundary:
        return False

    if (
        operator_kind == "per_example_gradient"
        and settings.get("per_example_gradient.accumulation") != "stacked_leading_axis"
    ):
        return False

    return settings.get(row.path_key) in runtime_values.SCORE_MATRIX_COMPILE_PATH_VALUES


def fisher_batch_inputs(operator: OperatorSpec, path: str) -> tuple[str, ...]:
    """Return the fisher batch inputs.

    Returns:
        The fisher batch inputs.
    """
    if path not in {
        runtime_values.FISHER_DENSE_PATH,
        runtime_values.FISHER_SCORE_GRADIENT_LOOP_PATH,
    }:
        return ()

    inputs = ()

    if path == runtime_values.FISHER_DENSE_PATH:
        inputs = ("score_gradients",)

    return (*inputs, *_fisher_denominator_batch_inputs(operator))


def sampled_fisher_batch_inputs(operator: OperatorSpec, path: str) -> tuple[str, ...]:
    """Return the sampled fisher batch inputs.

    Returns:
        The sampled fisher batch inputs.
    """
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


def empirical_fisher_batch_inputs(
    operator: OperatorSpec,
    path: str,
) -> tuple[str, ...]:
    """Return the empirical fisher batch inputs.

    Returns:
        The empirical fisher batch inputs.
    """
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


def run_fisher_vp(execution: runtime_values.StandardExecution) -> TensorTree:
    """Run fisher vp.

    Returns:
        The fisher vp result.
    """
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

    return vectorization.run_single_vectorized_by_path(
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


def run_sampled_fisher_vp(execution: runtime_values.StandardExecution) -> TensorTree:
    """Run sampled fisher vp.

    Returns:
        The sampled fisher vp result.
    """
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


def run_empirical_fisher_vp(execution: runtime_values.StandardExecution) -> TensorTree:
    """Run empirical fisher vp.

    Returns:
        The empirical fisher vp result.
    """
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

    return derivatives.per_example_gradient_matrix_from_builders(
        execution,
        derivatives.PER_EXAMPLE_GRADIENT_WITHOUT_MANUAL_BUILDERS,
        error_message,
    )


def score_gradient_matrix_from_operator_row(
    execution: runtime_values.StandardExecution,
) -> torch.Tensor:
    """Return the score gradient matrix from operator row.

    Returns:
        The score gradient matrix from operator row.

    Raises:
        MaterializationError: If the declared inputs are invalid.
    """
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
    vector_tensor = runtime.parameter_order_vector(execution)
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
    score_dot = runtime.matmul_runtime(settings, first_block, vector[:offset])

    for block in blocks[1:]:
        width = block.shape[1]
        stop = offset + width

        if stop > vector.numel():
            message = f"{label} block columns exceed vector length"
            raise MaterializationError(message)

        score_dot = score_dot + runtime.matmul_runtime(
            settings, block, vector[offset:stop]
        )
        offset = stop

    if offset != vector.numel():
        message = f"{label} block columns must match vector length"
        raise MaterializationError(message)

    pieces = tuple(
        runtime.matmul_runtime(settings, block.T, score_dot) for block in blocks
    )
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
    vector_tensor = runtime.parameter_order_vector(execution)
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

    score_dot = runtime.matmul_runtime(settings, score_gradients, vector)

    return (
        runtime.matmul_runtime(settings, score_gradients.T, score_dot) / normalization
    )


def _parameter_blocked_score_matrix_product(
    score_gradients: torch.Tensor,
    vector: torch.Tensor,
    normalization: float,
    settings: Mapping[str, Any],
    ranges: tuple[tuple[int, int], ...],
) -> torch.Tensor:
    _require_score_matrix_vector_shape(score_gradients, vector, "score_gradients")
    score_dot = runtime.parameter_blocked_matrix_vector_product(
        score_gradients,
        vector,
        settings,
        None,
        ranges,
    )
    result_chunks = []

    for start, stop in ranges:
        score_block = score_gradients[:, start:stop]
        result_chunks.append(runtime.matmul_runtime(settings, score_block.T, score_dot))

    result = torch.cat(tuple(result_chunks)) / normalization
    runtime_values.require_finite_tensor(result, "score matrix parameter-block result")

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
    vector_tensor = runtime.parameter_order_vector(execution)

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
    chunk_size = vectorization.vmap_chunk_size(execution.candidate.settings)
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

    for subexecution in derivatives.per_example_sliced_executions(
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
        for subexecution in derivatives.per_example_sliced_executions(
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

    builder = derivatives.STREAMING_GRADIENT_ROW_BUILDERS.get(execution.path)

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


def _accumulate_streaming_gradient_row_batch(
    execution: runtime_values.StandardExecution,
    result: torch.Tensor,
    row: torch.Tensor,
    vector_batch: torch.Tensor,
    chunk_size: int | None,
) -> torch.Tensor:
    scale = runtime.loss_scale(execution.candidate.settings)

    if scale is not None:
        row = row * scale

    def product(flat_vector: torch.Tensor) -> torch.Tensor:
        return row * runtime.dot_runtime(execution.candidate.settings, row, flat_vector)

    return result + vectorization.torch_func_vmap(
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
    scale = runtime.loss_scale(execution.candidate.settings)

    if scale is not None:
        row = row * scale

    return result + row * runtime.dot_runtime(
        execution.candidate.settings, row, vector_tensor
    )


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
    chunk_size = vectorization.vmap_chunk_size(execution.candidate.settings)

    def product(flat_vector: torch.Tensor) -> torch.Tensor:
        return _score_matrix_product(
            score_gradients,
            flat_vector,
            normalization,
            execution.candidate.settings,
            execution.parameter_surface,
        )

    result = vectorization.torch_func_vmap(
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
    chunk_size = vectorization.vmap_chunk_size(execution.candidate.settings)

    def product(flat_vector: torch.Tensor) -> torch.Tensor:
        return _blockwise_score_matrix_product_unchecked(
            blocks,
            flat_vector,
            normalization,
            execution.candidate.settings,
        )

    result = vectorization.torch_func_vmap(
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
    score_dot = runtime.matmul_runtime(settings, first_block, vector[:offset])

    for block in blocks[1:]:
        width = block.shape[1]
        stop = offset + width
        score_dot = score_dot + runtime.matmul_runtime(
            settings, block, vector[offset:stop]
        )
        offset = stop

    pieces = tuple(
        runtime.matmul_runtime(settings, block.T, score_dot) for block in blocks
    )

    return torch.cat(pieces) / normalization


def _flat_vector_batch(execution: runtime_values.StandardExecution) -> torch.Tensor:
    if execution.flat_parameter_vector_batch is not None:
        return execution.flat_parameter_vector_batch

    return vectorization.build_flat_vector_batch(execution)


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


def _per_example_gradient_matrix_manual_batches(
    execution: runtime_values.StandardExecution,
) -> torch.Tensor:
    return derivatives.per_example_gradient_matrix_batched(
        execution,
        "per-example manual batching",
        _per_example_manual_batch_size(execution),
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


def _loss_scaled_score_matrix(
    execution: runtime_values.StandardExecution,
    matrix: torch.Tensor,
) -> torch.Tensor:
    scale = runtime.loss_scale(execution.candidate.settings)

    if scale is None:
        return matrix

    return matrix * scale


def _loss_scaled_score_blocks(
    execution: runtime_values.StandardExecution,
    blocks: tuple[torch.Tensor, ...],
) -> tuple[torch.Tensor, ...]:
    scale = runtime.loss_scale(execution.candidate.settings)

    if scale is None:
        return blocks

    return tuple(block * scale for block in blocks)


def fisher_spec_runtime_path(candidate: Candidate) -> str | None:
    """Return the fisher spec runtime path.

    Returns:
        The fisher spec runtime path.
    """
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


def sampled_fisher_spec_runtime_path(candidate: Candidate) -> str | None:
    """Return the sampled fisher spec runtime path.

    Returns:
        The sampled fisher spec runtime path.
    """
    return _score_fisher_spec_runtime_path(
        candidate.settings,
        accumulation_key=runtime_values.SPEC_PATH_KEYS["sampled_fisher_vp"],
        score_path_key="sampled_fisher.score_grad_path",
        dense_path=runtime_values.SAMPLED_FISHER_DENSE_PATH,
        block_path=runtime_values.SAMPLED_FISHER_BLOCKWISE_SCORE_MATRIX_PATH,
        streaming_paths=runtime_values.SAMPLED_FISHER_STREAMING_PATH_BY_SCORE_GRAD,
    )


def empirical_fisher_spec_runtime_path(candidate: Candidate) -> str | None:
    """Return the empirical fisher spec runtime path.

    Returns:
        The empirical fisher spec runtime path.

    Raises:
        MaterializationError: If the declared inputs are invalid.
    """
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


def fisher_anchor_settings(
    operator: OperatorSpec,
    path: str,
) -> dict[str, Any]:
    """Return the fisher anchor settings.

    Returns:
        The fisher anchor settings.
    """
    if operator.kind != "fisher_vp":
        return {}

    if path != runtime_values.FISHER_SCORE_GRADIENT_LOOP_PATH:
        return {}

    return {
        "fisher.expectation_path": "explicit_full_expectation_score_rows",
        "fisher.score_grad_path": "torch_autograd_grad_loop",
    }


def sampled_fisher_anchor_settings(
    operator: OperatorSpec,
    candidate: Candidate,
    path: str,
) -> dict[str, Any]:
    """Return the sampled fisher anchor settings.

    Returns:
        The sampled fisher anchor settings.

    Raises:
        MaterializationError: If the declared inputs are invalid.
    """
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


def fisher_anchor_path(operator: OperatorSpec) -> str:
    """Return the fisher anchor path.

    Returns:
        The fisher anchor path.

    Raises:
        MaterializationError: If the declared inputs are invalid.
    """
    distribution = runtime_values.operator_semantic(operator, "distribution")

    if distribution == "explicit_score_gradients":
        return runtime_values.FISHER_SCORE_GRADIENT_LOOP_PATH

    message = "standard Fisher anchor does not support declared semantics"
    raise MaterializationError(message)


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
    batch, batch_in_dims = vectorization.per_example_batch_in_dims(
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
    vector = runtime.parameter_order_vector(execution)
    normalization = _sampled_fisher_normalization(execution)
    score_dot = runtime.matmul_runtime(
        execution.candidate.settings, score_matrix, vector
    )
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
        matrix = score_gradient_matrix_from_operator_row(execution)

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
        "streaming_matrix": score_gradient_matrix_from_operator_row,
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
        "streaming_matrix": score_gradient_matrix_from_operator_row,
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
        "streaming_matrix": score_gradient_matrix_from_operator_row,
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
