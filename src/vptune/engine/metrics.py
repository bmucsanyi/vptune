"""Metric representation lowerings for the standard runtime.

Multiply, inverse, square-root, and inner-product paths for dense,
diagonal, block-diagonal, KFAC, EKFAC, low-rank, GGN-derived, and
matrix-free metric representations, with their damping and
preconditioner helpers and dispatch tables.
"""

import dataclasses
import math
from collections.abc import Callable, Mapping, Sequence
from itertools import starmap
from typing import Any

import torch

from vptune.core.data import (
    Batch,
    Candidate,
    FullSizeRecord,
    ObjectiveContext,
    OperatorSpec,
    ParameterTree,
)
from vptune.core.tensor_tree import (
    TensorTree,
    tree_add_foreach,
    tree_add_scalar_foreach,
    tree_elementwise_div_foreach,
    tree_elementwise_mul_foreach,
    tree_map,
    tree_map2,
)
from vptune.engine import (
    layout,
    memory,
    runtime,
    runtime_values,
    vectorization,
)
from vptune.engine.anchors import (
    dense_metric_inverse_multiply,
    dense_metric_multiply,
)
from vptune.errors import (
    MaterializationError,
)


def composition_expression_weighted_terms(
    expression: Mapping[str, Any],
) -> tuple[Mapping[str, Any], ...]:
    """Return the composition expression weighted terms.

    Returns:
        The composition expression weighted terms.

    Raises:
        MaterializationError: If the declared inputs are invalid.
    """
    terms = expression.get("terms")

    if not isinstance(terms, Sequence) or isinstance(terms, str) or not terms:
        message = "linear composition terms must be non-empty mappings"
        raise MaterializationError(message)

    normalized = []

    for term in terms:
        if not isinstance(term, Mapping):
            message = "linear composition term must be a mapping"
            raise MaterializationError(message)

        coefficient = term.get("coefficient")
        nested = term.get("term")

        if not isinstance(coefficient, int | float) or isinstance(coefficient, bool):
            message = "linear composition coefficient must be a number"
            raise MaterializationError(message)

        if not isinstance(nested, Mapping):
            message = "linear composition term expression must be a mapping"
            raise MaterializationError(message)

        normalized.append({"coefficient": float(coefficient), "term": nested})

    return tuple(normalized)


def inverse_metric_inner_reference_measurements(
    operator: OperatorSpec,
    candidate: Candidate,
    batch: Batch,
    vector: TensorTree,
    params: ParameterTree,
) -> dict[str, float]:
    """Return the inverse metric inner reference measurements.

    Returns:
        The inverse metric inner reference measurements.
    """
    if operator.kind != "inverse_metric_inner":
        return {}

    if (
        candidate.settings.get("inverse_metric_inner.reduction_path")
        != "solve_then_reduce"
    ):
        return {}

    _, right = _metric_inner_vectors(vector)
    inverse_path = _inverse_metric_runtime_path_from_settings(candidate.settings)
    execution = _inverse_metric_inner_reference_execution(
        operator,
        candidate,
        inverse_path,
        batch,
        right,
        params,
    )
    right_matrix = _inverse_metric_inner_right_matrix(execution, right)
    solution_matrix = _inverse_metric_inner_solution_matrix(
        execution,
        right,
        inverse_path,
    )
    measurements = {
        "inverse_residual": _inverse_metric_inner_residual(
            operator,
            candidate,
            batch,
            params,
            right_matrix,
            solution_matrix,
        )
    }
    damping = inverse_metric_min_damping(operator)

    if damping > 0.0:
        measurements["damping_min"] = damping

    if metric_representation_kind(operator) != "matrix_free":
        inverse_matrix = inverse_metric_matrix(
            operator,
            metric_dense_matrix(operator, batch, params),
            batch,
            params,
        )
        measurements["condition_number_max"] = runtime_values.matrix_condition_number(
            inverse_matrix
        )

    return measurements


def _inverse_metric_inner_reference_execution(
    operator: OperatorSpec,
    candidate: Candidate,
    path: str,
    batch: Batch,
    vector: TensorTree,
    params: ParameterTree,
) -> "runtime_values.StandardExecution":
    return runtime_values.StandardExecution(
        operator=operator,
        candidate=candidate,
        path=path,
        batch=batch,
        vector=vector,
        params=params,
        buffers={},
        parameter_surface=None,
        context=ObjectiveContext(
            family=operator.family,
            candidate_id=candidate.candidate_id,
            settings=dict(candidate.settings),
        ),
        scalar_objectives={},
        function_objectives={},
    )


def _inverse_metric_inner_right_matrix(
    execution: "runtime_values.StandardExecution",
    right: TensorTree,
) -> torch.Tensor:
    block_mode = _metric_inner_block_mode(
        execution,
        "inverse_metric_inner.multi_rhs",
    )

    if block_mode is None:
        return runtime_values.flatten_vector(right).unsqueeze(0)

    return _metric_inner_flat_block(execution, right, 1)


def _inverse_metric_inner_solution_matrix(
    execution: "runtime_values.StandardExecution",
    right: TensorTree,
    inverse_path: str,
) -> torch.Tensor:
    block_mode = _metric_inner_block_mode(
        execution,
        "inverse_metric_inner.multi_rhs",
    )

    if block_mode is None:
        product = _inverse_metric_solve_by_path(execution)

        return runtime_values.flatten_vector(product).unsqueeze(0)

    right_execution = _metric_inner_side_execution(
        execution,
        right,
        1,
        path=inverse_path,
    )
    product = _run_inverse_metric_rhs_batch(right_execution)

    return _metric_inner_flat_leading_block(execution, product)


def _inverse_metric_inner_residual(
    operator: OperatorSpec,
    candidate: Candidate,
    batch: Batch,
    params: ParameterTree,
    right_matrix: torch.Tensor,
    solution_matrix: torch.Tensor,
) -> float:
    damping = inverse_metric_damping_payload(operator)

    if metric_representation_kind(operator) == "matrix_free":
        metric_path = _metric_runtime_path_from_settings(candidate.settings)
        applied = _metric_apply_flat_batch(
            operator,
            batch,
            params,
            solution_matrix,
            damping,
            metric_path,
            candidate.settings,
        )
    else:
        inverse_matrix = inverse_metric_matrix(
            operator,
            metric_dense_matrix(operator, batch, params),
            batch,
            params,
        )
        applied = layout.matmul_runtime(
            candidate.settings,
            solution_matrix,
            inverse_matrix.T,
        )

    residual = applied - right_matrix
    residual_norm = torch.linalg.vector_norm(residual, dim=1)
    denominator = torch.linalg.vector_norm(right_matrix, dim=1)
    scaled = torch.where(
        denominator == 0,
        residual_norm,
        residual_norm / denominator,
    )

    return float(torch.max(scaled).item())


def inverse_metric_matrix(
    operator: OperatorSpec,
    matrix: torch.Tensor,
    batch: Batch,
    template: TensorTree | None = None,
) -> torch.Tensor:
    """Return the inverse metric matrix.

    Returns:
        The inverse metric matrix.
    """
    if _inverse_metric_damping_kind(operator) == "per_group":
        return _per_group_damped_metric_matrix(operator, matrix, batch, template)

    return _damped_metric_matrix(matrix, _inverse_metric_damping(operator))


def _per_group_damped_metric_matrix(
    operator: OperatorSpec,
    matrix: torch.Tensor,
    batch: Batch,
    template: TensorTree | None,
) -> torch.Tensor:
    kind = metric_representation_kind(operator)

    if kind == "diagonal_tree":
        damping = runtime_values.flatten_vector(
            _diagonal_group_damping_tree(
                operator,
                runtime_values.batch_tree(batch, "metric_diagonal"),
            )
        )
        damped = matrix + torch.diag(damping)
    elif kind == "block_diagonal":
        blocks = _metric_blocks(batch)
        damped = torch.block_diag(
            *starmap(
                _damped_metric_matrix,
                zip(blocks, _block_metric_dampings(operator, blocks), strict=True),
            )
        )
    elif kind == "kfac_factors":
        damped = _kfac_per_group_damped_metric_matrix(operator, batch)
    elif kind == "ekfac_factors":
        damped = _ekfac_per_group_damped_metric_matrix(operator, batch, template)
    elif kind in {
        "dense_matrix",
        "low_rank_factors",
        "ggn_derived_factors",
    }:
        damping = _per_group_damping_vector(
            operator,
            matrix.shape[0],
            dtype=matrix.dtype,
            device=matrix.device,
        )
        damped = matrix + torch.diag(damping)
    else:
        message = f"per_group damping is not lowered for metric kind: {kind}"
        raise MaterializationError(message)

    if damped.shape != matrix.shape:
        message = "per_group damping matrix does not match metric matrix shape"
        raise MaterializationError(message)

    runtime_values.require_finite_tensor(damped, "per-group damped metric matrix")

    return damped


def _kfac_per_group_damped_metric_matrix(
    operator: OperatorSpec,
    batch: Batch,
) -> torch.Tensor:
    factor_batch = _kfac_factor_batch(batch)
    damped_blocks = []

    for block in _kfac_blocks(operator):
        left = _kfac_factor(factor_batch, block.left_factor_key)
        right = _kfac_factor(factor_batch, block.right_factor_key)
        dense_block = torch.kron(left, right)
        damping = _inverse_metric_group_damping(operator, block.parameter_name)
        damped_blocks.append(_damped_metric_matrix(dense_block, damping))

    return torch.block_diag(*damped_blocks)


def _ekfac_per_group_damped_metric_matrix(
    operator: OperatorSpec,
    batch: Batch,
    template: TensorTree | None,
) -> torch.Tensor:
    if template is None:
        message = "EKFAC per_group damping requires a vector template"
        raise MaterializationError(message)

    damped_blocks = []

    for key, value in _ekfac_vector_map(template).items():
        eigvecs_a, eigvecs_g, eigenvalues = _ekfac_factors(batch, key, value)
        basis = torch.kron(eigvecs_a, eigvecs_g)
        damping = _inverse_metric_group_damping(operator, key)
        spectrum = eigenvalues.reshape(-1) + damping
        block = basis @ torch.diag(spectrum) @ basis.T
        damped_blocks.append(block)

    return torch.block_diag(*damped_blocks)


def metric_reference_output(
    operator: OperatorSpec,
    batch: Batch,
    vector: TensorTree,
) -> TensorTree:
    """Return the metric reference output.

    Returns:
        The metric reference output.

    Raises:
        MaterializationError: If the declared inputs are invalid.
    """
    matrix = metric_dense_matrix(operator, batch, vector)
    flat_vector = runtime_values.flatten_vector(vector)
    runtime_values.require_finite_tensor(matrix, "metric matrix")
    runtime_values.require_finite_tensor(flat_vector, "metric vector")

    if operator.kind == "metric":
        result = dense_metric_multiply(matrix, flat_vector)
    elif operator.kind == "inverse_metric":
        result = dense_metric_inverse_multiply(
            inverse_metric_matrix(operator, matrix, batch, vector),
            flat_vector,
        )
    else:
        message = f"metric reference output is not supported for {operator.kind}"
        raise MaterializationError(message)

    runtime_values.require_finite_tensor(result, "metric reference result")

    return runtime_values.wrap_flat_vector(vector, result)


def metric_dense_matrix(
    operator: OperatorSpec,
    batch: Batch,
    vector: TensorTree,
) -> torch.Tensor:
    """Return the metric dense matrix.

    Returns:
        The metric dense matrix.

    Raises:
        MaterializationError: If the declared inputs are invalid.
    """
    kind = metric_representation_kind(operator)

    if kind == "dense_matrix":
        matrix = runtime_values.batch_tensor(batch, "metric_matrix")
    elif kind == "diagonal_tree":
        diagonal = runtime_values.flatten_vector(_metric_diagonal_tree(batch, vector))
        matrix = torch.diag(diagonal)
    elif kind == "block_diagonal":
        matrix = torch.block_diag(*_metric_blocks(batch))
    elif kind == "kfac_factors":
        matrix = _kfac_dense_matrix(operator, batch)
    elif kind == "ekfac_factors":
        matrix = _ekfac_dense_matrix(batch, vector)
    elif kind == "low_rank_factors":
        basis, diagonal = _low_rank_factors(batch, vector)
        matrix = basis @ basis.T + torch.diag(diagonal)
    elif kind == "ggn_derived_factors":
        jacobian, loss_hessian = _ggn_metric_factors(batch, vector)
        matrix = jacobian.T @ loss_hessian @ jacobian
    else:
        message = f"metric representation has no dense reference: {kind}"
        raise MaterializationError(message)

    runtime_values.require_finite_tensor(matrix, "metric matrix")

    return matrix


def _metric_diagonal_tree(batch: Batch, vector: TensorTree) -> TensorTree:
    diagonal = tree_map2(
        lambda diag, template: diag.reshape_as(template),
        runtime_values.batch_tree(batch, "metric_diagonal"),
        vector,
    )
    runtime_values.require_finite_tree(diagonal, "metric diagonal")

    return diagonal


def _diagonal_metric_multiply(
    operator: OperatorSpec,
    batch: Batch,
    vector: TensorTree,
    settings: Mapping[str, Any],
) -> TensorTree:
    _ = operator
    diagonal = _metric_diagonal_tree(batch, vector)
    result = _tree_elementwise_mul_runtime(settings, diagonal, vector)
    runtime_values.require_finite_tree(result, "metric result")

    return result


def _diagonal_inverse_metric_multiply(
    operator: OperatorSpec,
    batch: Batch,
    vector: TensorTree,
    settings: Mapping[str, Any],
) -> TensorTree:
    diagonal = _metric_diagonal_tree(batch, vector)
    denominator = _diagonal_inverse_denominator(operator, diagonal, settings)
    result = _tree_elementwise_div_runtime(settings, vector, denominator)
    runtime_values.require_finite_tree(result, "inverse metric result")

    return result


def _diagonal_inverse_denominator(
    operator: OperatorSpec,
    diagonal: TensorTree,
    settings: Mapping[str, Any],
) -> TensorTree:
    if _inverse_metric_damping_kind(operator) != "per_group":
        return _tree_add_scalar_runtime(
            settings,
            diagonal,
            _inverse_metric_damping(operator),
        )

    runtime_diagonal = runtime.runtime_intermediate_tree(diagonal, settings)
    damping_tree = _diagonal_group_damping_tree(operator, runtime_diagonal)

    if layout.layout_vector_ops(settings) == "foreach":
        return tree_add_foreach(runtime_diagonal, damping_tree)

    return tree_map2(torch.add, runtime_diagonal, damping_tree)


def _diagonal_group_damping_tree(
    operator: OperatorSpec,
    diagonal: TensorTree,
) -> TensorTree:
    values = _inverse_metric_damping_values(operator)

    if not runtime_values.is_tensor_tree_dict(diagonal) or not diagonal:
        message = "diagonal per_group damping requires named tensor leaves"
        raise MaterializationError(message)

    if set(diagonal) != set(values):
        message = "diagonal per_group damping keys must match metric leaves"
        raise MaterializationError(message)

    result = {}

    for key, value in diagonal.items():
        if not isinstance(key, str) or not isinstance(value, torch.Tensor):
            message = "diagonal per_group damping requires named tensor leaves"
            raise MaterializationError(message)

        result[key] = torch.full_like(value, values[key])

    return result


def _diagonal_inverse_metric_multiply_batch(
    execution: "runtime_values.StandardExecution",
) -> TensorTree:
    denominator = runtime_values.flatten_vector(
        _diagonal_inverse_denominator(
            execution.operator,
            _metric_diagonal_tree(execution.batch, execution.params),
            execution.candidate.settings,
        )
    )
    vector_batch = _flat_inverse_metric_vector_batch(execution)
    result = vector_batch / denominator
    runtime_values.require_finite_tensor(
        result, "batched diagonal inverse metric result"
    )

    return runtime_values.wrap_flat_vector_batch(execution.params, result)


def _metric_blocks(batch: Batch) -> tuple[torch.Tensor, ...]:
    value = batch.get("metric_blocks")

    if not isinstance(value, tuple) or not value:
        message = "metric blocks are missing"
        raise MaterializationError(message)

    for block in value:
        if not isinstance(block, torch.Tensor):
            message = "metric block must be a tensor"
            raise MaterializationError(message)

        if block.ndim != runtime_values.MATRIX_DIMS or block.shape[0] != block.shape[1]:
            message = "metric block must be square"
            raise MaterializationError(message)

        runtime_values.require_finite_tensor(block, "metric block")

    return value


def _block_diagonal_metric_multiply(
    operator: OperatorSpec,
    batch: Batch,
    vector: TensorTree,
    settings: Mapping[str, Any],
) -> TensorTree:
    _ = operator
    result = _block_diagonal_apply(
        _metric_blocks(batch),
        runtime_values.flatten_vector(vector),
        settings,
        "metric block multiply",
    )

    return runtime_values.wrap_flat_vector(vector, result)


def _block_diagonal_inverse_metric_multiply(
    operator: OperatorSpec,
    batch: Batch,
    vector: TensorTree,
    settings: Mapping[str, Any],
) -> TensorTree:
    _ = settings
    blocks = _metric_blocks(batch)
    result = _block_diagonal_solve(
        blocks,
        runtime_values.flatten_vector(vector),
        _block_metric_dampings(operator, blocks),
    )

    return runtime_values.wrap_flat_vector(vector, result)


def _block_diagonal_inverse_metric_multiply_batch(
    execution: "runtime_values.StandardExecution",
) -> TensorTree:
    blocks = _metric_blocks(execution.batch)
    vector_batch = _flat_inverse_metric_vector_batch(execution)
    result = _block_diagonal_solve_batch(
        blocks,
        vector_batch,
        _block_metric_dampings(execution.operator, blocks),
    )

    return runtime_values.wrap_flat_vector_batch(execution.params, result)


def _block_metric_dampings(
    operator: OperatorSpec,
    blocks: tuple[torch.Tensor, ...],
) -> tuple[float, ...]:
    if _inverse_metric_damping_kind(operator) == "per_group":
        names = _block_diagonal_group_names(operator)

        if len(names) != len(blocks):
            message = "per_group damping keys do not match metric block count"
            raise MaterializationError(message)

        return tuple(_inverse_metric_group_damping(operator, name) for name in names)

    damping = _inverse_metric_damping(operator)

    return tuple(damping for _ in blocks)


def _block_diagonal_group_names(operator: OperatorSpec) -> tuple[str, ...]:
    representation = metric_representation(operator)
    value = representation.get("block_names")

    if (
        not isinstance(value, tuple)
        or not value
        or any(not isinstance(name, str) for name in value)
    ):
        message = "block-diagonal per_group damping requires named metric blocks"
        raise MaterializationError(message)

    return value


def _block_diagonal_apply(
    blocks: tuple[torch.Tensor, ...],
    vector: torch.Tensor,
    settings: Mapping[str, Any],
    name: str,
) -> torch.Tensor:
    parts = []
    offset = 0

    for block in blocks:
        width = block.shape[1]
        part = vector[offset : offset + width]

        if part.numel() != width:
            message = "metric blocks do not match vector length"
            raise MaterializationError(message)

        parts.append(layout.matmul_runtime(settings, block, part))
        offset += width

    if offset != vector.numel():
        message = "metric blocks do not match vector length"
        raise MaterializationError(message)

    result = torch.cat(tuple(parts))
    runtime_values.require_finite_tensor(result, name)

    return result


def _block_diagonal_solve(
    blocks: tuple[torch.Tensor, ...],
    vector: torch.Tensor,
    dampings: tuple[float, ...],
) -> torch.Tensor:
    result = _block_diagonal_solve_batch(
        blocks,
        vector.unsqueeze(0),
        dampings,
    )[0]
    runtime_values.require_finite_tensor(result, "inverse metric block solve")

    return result


def _block_diagonal_solve_batch(
    blocks: tuple[torch.Tensor, ...],
    vector_batch: torch.Tensor,
    dampings: tuple[float, ...],
) -> torch.Tensor:
    if vector_batch.ndim != runtime_values.MATRIX_DIMS:
        message = "batched block inverse vectors must flatten to a matrix"
        raise MaterializationError(message)

    parts = []
    offset = 0

    for block, damping in zip(blocks, dampings, strict=True):
        width = block.shape[1]
        part = vector_batch[:, offset : offset + width]

        if part.shape[1] != width:
            message = "metric blocks do not match vector width"
            raise MaterializationError(message)

        solved = torch.linalg.solve(
            _damped_metric_matrix(block, damping),
            part.T,
        ).T
        parts.append(solved)
        offset += width

    if offset != vector_batch.shape[1]:
        message = "metric blocks do not match vector width"
        raise MaterializationError(message)

    result = torch.cat(tuple(parts), dim=1)
    runtime_values.require_finite_tensor(result, "batched inverse metric block solve")

    return result


def _low_rank_factors(
    batch: Batch,
    vector: TensorTree,
) -> tuple[torch.Tensor, torch.Tensor]:
    value = batch.get("low_rank_factors")

    if not isinstance(value, Mapping):
        message = "low_rank_factors must be a mapping"
        raise MaterializationError(message)

    basis = value.get("basis")
    diagonal = value.get("diagonal")
    width = runtime_values.flatten_vector(vector).numel()

    if not isinstance(basis, torch.Tensor) or basis.ndim != runtime_values.MATRIX_DIMS:
        message = "low_rank_factors.basis must be a two-dimensional tensor"
        raise MaterializationError(message)

    if basis.shape[0] != width:
        message = "low_rank_factors.basis row count must match vector length"
        raise MaterializationError(message)

    if not isinstance(diagonal, torch.Tensor) or diagonal.ndim != 1:
        message = "low_rank_factors.diagonal must be a one-dimensional tensor"
        raise MaterializationError(message)

    if diagonal.numel() != width:
        message = "low_rank_factors.diagonal length must match vector length"
        raise MaterializationError(message)

    runtime_values.require_finite_tensor(basis, "low-rank basis")
    runtime_values.require_finite_tensor(diagonal, "low-rank diagonal")

    return basis, diagonal


def _low_rank_metric_multiply(
    operator: OperatorSpec,
    batch: Batch,
    vector: TensorTree,
    settings: Mapping[str, Any],
) -> TensorTree:
    _ = operator
    flat_vector = runtime.runtime_intermediate_tensor(
        runtime_values.flatten_vector(vector), settings
    )
    basis, diagonal = _low_rank_factors(batch, vector)
    basis = runtime.runtime_intermediate_tensor(basis, settings)
    diagonal = runtime.runtime_intermediate_tensor(diagonal, settings)
    basis_projection = layout.matmul_runtime(settings, basis.T, flat_vector)
    result = (
        runtime.accumulation_tensor(diagonal, settings)
        * runtime.accumulation_tensor(flat_vector, settings)
    ) + layout.matmul_runtime(settings, basis, basis_projection)
    runtime_values.require_finite_tensor(result, "low-rank metric result")

    return runtime_values.wrap_flat_vector(vector, result)


def _low_rank_inverse_metric_multiply(
    operator: OperatorSpec,
    batch: Batch,
    vector: TensorTree,
    settings: Mapping[str, Any],
) -> TensorTree:
    _ = settings
    result = _low_rank_inverse_metric_flat_batch(
        operator,
        batch,
        vector,
        runtime_values.flatten_vector(vector).unsqueeze(0),
    )[0]
    runtime_values.require_finite_tensor(result, "low-rank inverse metric result")

    return runtime_values.wrap_flat_vector(vector, result)


def _low_rank_inverse_metric_flat_batch(
    operator: OperatorSpec,
    batch: Batch,
    template: TensorTree,
    vector_batch: torch.Tensor,
) -> torch.Tensor:
    basis, diagonal = _low_rank_factors(batch, template)
    base_diagonal = _damped_diagonal_vector(operator, template, diagonal)

    return _low_rank_plus_diagonal_inverse_flat_batch(
        basis,
        base_diagonal,
        vector_batch,
        "low-rank inverse metric",
    )


def _low_rank_plus_diagonal_inverse_flat_batch(
    basis: torch.Tensor,
    base_diagonal: torch.Tensor,
    vector_batch: torch.Tensor,
    name: str,
) -> torch.Tensor:
    _require_positive_spectrum(base_diagonal, f"{name} base diagonal")
    inverse_base_vectors = vector_batch / base_diagonal
    inverse_base_basis = basis / base_diagonal.unsqueeze(1)
    inner = (
        torch.eye(
            basis.shape[1],
            dtype=basis.dtype,
            device=basis.device,
        )
        + basis.T @ inverse_base_basis
    )
    correction = (
        inverse_base_basis @ torch.linalg.solve(inner, basis.T @ inverse_base_vectors.T)
    ).T
    result = inverse_base_vectors - correction
    runtime_values.require_finite_tensor(result, f"batched {name} result")

    return result


def _low_rank_inverse_metric_multiply_batch(
    execution: "runtime_values.StandardExecution",
) -> TensorTree:
    result = _low_rank_inverse_metric_flat_batch(
        execution.operator,
        execution.batch,
        execution.params,
        _flat_inverse_metric_vector_batch(execution),
    )

    return runtime_values.wrap_flat_vector_batch(execution.params, result)


def _kfac_metric_multiply(
    operator: OperatorSpec,
    batch: Batch,
    vector: TensorTree,
    settings: Mapping[str, Any],
) -> TensorTree:
    metric = KFACMetricOperator(_kfac_blocks(operator), settings=settings)

    return metric.multiply(_kfac_factor_batch(batch), vector)


def _kfac_inverse_metric_multiply(
    operator: OperatorSpec,
    batch: Batch,
    vector: TensorTree,
    settings: Mapping[str, Any],
) -> TensorTree:
    _ = settings
    metric = KFACMetricOperator(
        _kfac_blocks(operator),
        damping=inverse_metric_damping_payload(operator),
        damping_kind=_inverse_metric_damping_kind(operator),
        damping_policy=_inverse_metric_damping_policy(operator),
    )

    return metric.inverse_multiply(_kfac_factor_batch(batch), vector)


def _kfac_inverse_metric_multiply_batch(
    execution: "runtime_values.StandardExecution",
) -> TensorTree:
    batch = _kfac_factor_batch(execution.batch)
    vector_map = _kfac_vector_map(execution.vector)
    vector_in_dims = vectorization.vector_tree_in_dims(
        execution.vector,
        execution.candidate.settings,
    )
    vector_count = runtime_values.vector_tree_batch_size(
        execution.vector, vector_in_dims
    )
    result = {}

    if not isinstance(vector_in_dims, Mapping):
        message = "KFAC vectorization.in_dims must be a mapping"
        raise MaterializationError(message)

    for block in _kfac_blocks(execution.operator):
        left = _kfac_factor(batch, block.left_factor_key)
        right = _kfac_factor(batch, block.right_factor_key)
        value = _kfac_batched_vector_leaf(
            vector_map,
            vector_in_dims,
            block,
            left,
            right,
            vector_count,
        )
        product = _kfac_inverse_product_batch(
            execution.operator,
            block.parameter_name,
            left,
            right,
            value,
        )
        runtime_values.require_finite_tensor(
            product,
            f"batched inverse KFAC metric result {block.parameter_name}",
        )
        result[block.parameter_name] = product

    return result


def _kfac_square_root_apply(
    execution: "runtime_values.StandardExecution",
    *,
    inverse: bool,
) -> TensorTree:
    batch = _kfac_factor_batch(execution.batch)
    vector_map = _kfac_vector_map(execution.vector)
    result = {}

    for block in _kfac_blocks(execution.operator):
        left = _kfac_factor(batch, block.left_factor_key)
        right = _kfac_factor(batch, block.right_factor_key)
        value = _kfac_vector_leaf(vector_map, block)
        _require_kfac_shapes(block, left, right, value)

        if inverse:
            product = _kfac_inverse_square_root_product(
                execution.operator,
                block.parameter_name,
                left,
                right,
                value,
            )
        else:
            product = _kfac_square_root_product(left, right, value)

        runtime_values.require_finite_tensor(
            product,
            f"KFAC square-root metric result {block.parameter_name}",
        )
        result[block.parameter_name] = product

    return result


def _kfac_square_root_product(
    left: torch.Tensor,
    right: torch.Tensor,
    value: torch.Tensor,
) -> torch.Tensor:
    left_eigenvalues, left_eigenvectors = torch.linalg.eigh(left)
    right_eigenvalues, right_eigenvectors = torch.linalg.eigh(right)
    _require_positive_spectrum(left_eigenvalues, "KFAC left spectrum")
    _require_positive_spectrum(right_eigenvalues, "KFAC right spectrum")

    return _kfac_eigenbasis_scale(
        left_eigenvectors,
        right_eigenvectors,
        torch.sqrt(left_eigenvalues)[:, None] * torch.sqrt(right_eigenvalues)[None, :],
        value,
    )


def _kfac_inverse_square_root_product(
    operator: OperatorSpec,
    parameter_name: str,
    left: torch.Tensor,
    right: torch.Tensor,
    value: torch.Tensor,
) -> torch.Tensor:
    left_eigenvalues, left_eigenvectors = torch.linalg.eigh(left)
    right_eigenvalues, right_eigenvectors = torch.linalg.eigh(right)
    damping_kind = _inverse_metric_damping_kind(operator)
    damping = _resolved_group_damping(
        inverse_metric_damping_payload(operator),
        damping_kind,
        parameter_name,
    )
    resolved_kind = _resolved_group_damping_kind(damping_kind)

    if resolved_kind == "scalar":
        spectrum = left_eigenvalues[:, None] * right_eigenvalues[None, :] + damping
        _require_positive_spectrum(spectrum.reshape(-1), "KFAC inverse spectrum")
        scale = torch.rsqrt(spectrum)
    elif resolved_kind == "kfac_pi":
        left_shift, right_shift = _kfac_pi_shifts(
            left,
            right,
            damping,
            _inverse_metric_damping_policy(operator),
        )
        left_spectrum = left_eigenvalues + left_shift
        right_spectrum = right_eigenvalues + right_shift
        _require_positive_spectrum(left_spectrum, "KFAC pi left spectrum")
        _require_positive_spectrum(right_spectrum, "KFAC pi right spectrum")
        scale = (
            torch.rsqrt(left_spectrum)[:, None] * torch.rsqrt(right_spectrum)[None, :]
        )
    else:
        message = (
            f"KFAC inverse square root does not lower damping kind: {damping_kind}"
        )
        raise MaterializationError(message)

    return _kfac_eigenbasis_scale(
        left_eigenvectors,
        right_eigenvectors,
        scale,
        value,
    )


def _kfac_eigenbasis_scale(
    left_eigenvectors: torch.Tensor,
    right_eigenvectors: torch.Tensor,
    scale: torch.Tensor,
    value: torch.Tensor,
) -> torch.Tensor:
    rotated = left_eigenvectors.T @ value @ right_eigenvectors
    scaled = scale * rotated

    return left_eigenvectors @ scaled @ right_eigenvectors.T


def _kfac_dense_matrix(operator: OperatorSpec, batch: Batch) -> torch.Tensor:
    factor_batch = _kfac_factor_batch(batch)
    dense_blocks = []

    for block in _kfac_blocks(operator):
        left = _kfac_factor(factor_batch, block.left_factor_key)
        right = _kfac_factor(factor_batch, block.right_factor_key)
        dense_blocks.append(torch.kron(left, right))

    matrix = torch.block_diag(*dense_blocks)
    runtime_values.require_finite_tensor(matrix, "KFAC dense matrix")

    return matrix


def _kfac_blocks(
    operator: OperatorSpec,
) -> tuple["runtime_values.KFACMetricBlock", ...]:
    representation = metric_representation(operator)
    raw_blocks = representation.get("blocks")

    if not isinstance(raw_blocks, tuple) or not raw_blocks:
        message = "kfac_factors representation requires non-empty blocks"
        raise MaterializationError(message)

    blocks = []

    for raw_block in raw_blocks:
        if not isinstance(raw_block, Mapping):
            message = "KFAC block descriptor must be a mapping"
            raise MaterializationError(message)

        parameter_name = raw_block.get("parameter")
        left_factor_key = raw_block.get("left_factor")
        right_factor_key = raw_block.get("right_factor")

        if not isinstance(parameter_name, str):
            message = "KFAC block parameter must be a string"
            raise MaterializationError(message)

        if not isinstance(left_factor_key, str):
            message = "KFAC block left_factor must be a string"
            raise MaterializationError(message)

        if not isinstance(right_factor_key, str):
            message = "KFAC block right_factor must be a string"
            raise MaterializationError(message)

        blocks.append(
            runtime_values.KFACMetricBlock(
                parameter_name,
                left_factor_key,
                right_factor_key,
            )
        )

    return tuple(blocks)


def _kfac_factor_batch(batch: Batch) -> Batch:
    value = batch.get("kfac_factors")

    if not isinstance(value, Mapping):
        message = "kfac_factors must be a mapping"
        raise MaterializationError(message)

    return value


def _ekfac_dense_matrix(batch: Batch, vector: TensorTree) -> torch.Tensor:
    blocks = []

    for key, value in _ekfac_vector_map(vector).items():
        eigvecs_a, eigvecs_g, eigenvalues = _ekfac_factors(batch, key, value)
        basis = torch.kron(eigvecs_a, eigvecs_g)
        block = basis @ torch.diag(eigenvalues.reshape(-1)) @ basis.T
        blocks.append(block)

    matrix = torch.block_diag(*blocks)
    runtime_values.require_finite_tensor(matrix, "EKFAC dense matrix")

    return matrix


def _ekfac_metric_multiply(
    operator: OperatorSpec,
    batch: Batch,
    vector: TensorTree,
    settings: Mapping[str, Any],
) -> TensorTree:
    _ = operator

    return _ekfac_apply(batch, vector, settings, inverse=False, square_root=False)


def _ekfac_inverse_metric_multiply(
    operator: OperatorSpec,
    batch: Batch,
    vector: TensorTree,
    settings: Mapping[str, Any],
) -> TensorTree:
    _ = settings

    return _ekfac_apply(
        batch,
        vector,
        {},
        inverse=True,
        square_root=False,
        damping=inverse_metric_damping_payload(operator),
        damping_kind=_inverse_metric_damping_kind(operator),
    )


def _ekfac_inverse_metric_multiply_batch(
    execution: "runtime_values.StandardExecution",
) -> TensorTree:
    vector_batch = _flat_inverse_metric_vector_batch(execution)
    matrix = _ekfac_dense_matrix(execution.batch, execution.params)
    result = torch.linalg.solve(
        inverse_metric_matrix(
            execution.operator,
            matrix,
            execution.batch,
            execution.params,
        ),
        vector_batch.T,
    ).T
    runtime_values.require_finite_tensor(result, "batched inverse EKFAC metric result")

    return runtime_values.wrap_flat_vector_batch(execution.params, result)


def _ekfac_square_root_apply(
    execution: "runtime_values.StandardExecution",
    *,
    inverse: bool,
) -> TensorTree:
    return _ekfac_apply(
        execution.batch,
        execution.vector,
        execution.candidate.settings,
        inverse=inverse,
        square_root=True,
        damping=(
            inverse_metric_damping_payload(execution.operator) if inverse else 0.0
        ),
        damping_kind=(
            _inverse_metric_damping_kind(execution.operator) if inverse else "scalar"
        ),
    )


def _ekfac_apply(
    batch: Batch,
    vector: TensorTree,
    settings: Mapping[str, Any],
    *,
    inverse: bool,
    square_root: bool,
    damping: float | Mapping[str, float] = 0.0,
    damping_kind: str = "scalar",
) -> TensorTree:
    result = {}

    for key, value in _ekfac_vector_map(vector).items():
        eigvecs_a, eigvecs_g, eigenvalues = _ekfac_factors(batch, key, value)
        leaf_damping = (
            _resolved_group_damping(damping, damping_kind, key) if inverse else 0.0
        )
        result[key] = _ekfac_apply_leaf(
            eigvecs_a,
            eigvecs_g,
            eigenvalues,
            value,
            settings,
            inverse=inverse,
            square_root=square_root,
            damping=leaf_damping,
        )

    return result


def _ekfac_apply_leaf(
    eigvecs_a: torch.Tensor,
    eigvecs_g: torch.Tensor,
    eigenvalues: torch.Tensor,
    value: torch.Tensor,
    settings: Mapping[str, Any],
    *,
    inverse: bool,
    square_root: bool,
    damping: float,
) -> torch.Tensor:
    spectrum = eigenvalues + damping if inverse else eigenvalues
    _require_positive_spectrum(spectrum.reshape(-1), "EKFAC spectrum")
    rotated = layout.matmul_runtime(
        settings,
        layout.matmul_runtime(settings, eigvecs_a.T, value),
        eigvecs_g,
    )

    if square_root:
        factors = torch.rsqrt(spectrum) if inverse else torch.sqrt(spectrum)
    elif inverse:
        factors = torch.reciprocal(spectrum)
    else:
        factors = spectrum

    scaled = factors * rotated
    result = layout.matmul_runtime(
        settings,
        layout.matmul_runtime(settings, eigvecs_a, scaled),
        eigvecs_g.T,
    )
    runtime_values.require_finite_tensor(result, "EKFAC metric result")

    return result


def _ekfac_vector_map(vector: TensorTree) -> dict[str, torch.Tensor]:
    if type(vector) is not dict:
        message = "EKFAC vector must be a tensor-tree mapping"
        raise MaterializationError(message)

    result = {}

    for key, value in vector.items():
        if not isinstance(key, str) or not isinstance(value, torch.Tensor):
            message = "EKFAC vector leaves must be named tensors"
            raise MaterializationError(message)

        if value.ndim != runtime_values.MATRIX_DIMS:
            message = f"EKFAC vector leaf must be a matrix: {key}"
            raise MaterializationError(message)

        runtime_values.require_finite_tensor(value, f"EKFAC vector {key}")
        result[key] = value

    return result


def _ekfac_factor_map(batch: Batch, key: str) -> Mapping[str, torch.Tensor]:
    value = batch.get(key)

    if not isinstance(value, Mapping):
        message = f"{key} must be a mapping"
        raise MaterializationError(message)

    return value


def _ekfac_factors(
    batch: Batch,
    key: str,
    value: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    eigvecs_a = _ekfac_factor_tensor(
        _ekfac_factor_map(batch, "ekfac_eigvecs_a"),
        key,
        "ekfac_eigvecs_a",
    )
    eigvecs_g = _ekfac_factor_tensor(
        _ekfac_factor_map(batch, "ekfac_eigvecs_g"),
        key,
        "ekfac_eigvecs_g",
    )
    eigenvalues = _ekfac_factor_tensor(
        _ekfac_factor_map(batch, "ekfac_corrected_eigenvalues"),
        key,
        "ekfac_corrected_eigenvalues",
    )

    if (
        eigvecs_a.ndim != runtime_values.MATRIX_DIMS
        or eigvecs_a.shape[0] != eigvecs_a.shape[1]
    ):
        message = f"EKFAC eigvecs_a must be square for {key}"
        raise MaterializationError(message)

    if (
        eigvecs_g.ndim != runtime_values.MATRIX_DIMS
        or eigvecs_g.shape[0] != eigvecs_g.shape[1]
    ):
        message = f"EKFAC eigvecs_g must be square for {key}"
        raise MaterializationError(message)

    if tuple(value.shape) != (eigvecs_a.shape[0], eigvecs_g.shape[0]):
        message = f"EKFAC vector leaf shape mismatch for {key}"
        raise MaterializationError(message)

    if tuple(eigenvalues.shape) != tuple(value.shape):
        message = f"EKFAC corrected eigenvalues shape mismatch for {key}"
        raise MaterializationError(message)

    return eigvecs_a, eigvecs_g, eigenvalues


def _ekfac_factor_tensor(
    values: Mapping[str, torch.Tensor],
    key: str,
    label: str,
) -> torch.Tensor:
    tensor = values.get(key)

    if not isinstance(tensor, torch.Tensor):
        message = f"{label} is missing or not a tensor: {key}"
        raise MaterializationError(message)

    runtime_values.require_finite_tensor(tensor, f"EKFAC factor {label}.{key}")

    return tensor


def _ggn_metric_factors(
    batch: Batch,
    vector: TensorTree,
) -> tuple[torch.Tensor, torch.Tensor]:
    value = batch.get("ggn_factors")

    if not isinstance(value, Mapping):
        message = "ggn_factors must be a mapping"
        raise MaterializationError(message)

    jacobian = value.get("jacobian")
    loss_hessian = value.get("loss_hessian")
    width = runtime_values.flatten_vector(vector).numel()

    if (
        not isinstance(jacobian, torch.Tensor)
        or jacobian.ndim != runtime_values.MATRIX_DIMS
    ):
        message = "ggn_factors.jacobian must be a two-dimensional tensor"
        raise MaterializationError(message)

    if jacobian.shape[1] != width:
        message = "ggn_factors.jacobian column count must match vector length"
        raise MaterializationError(message)

    if (
        not isinstance(loss_hessian, torch.Tensor)
        or loss_hessian.ndim != runtime_values.MATRIX_DIMS
        or loss_hessian.shape[0] != loss_hessian.shape[1]
    ):
        message = "ggn_factors.loss_hessian must be a square matrix"
        raise MaterializationError(message)

    if loss_hessian.shape[0] != jacobian.shape[0]:
        message = "ggn_factors.loss_hessian shape must match jacobian rows"
        raise MaterializationError(message)

    runtime_values.require_finite_tensor(jacobian, "GGN metric jacobian")
    runtime_values.require_finite_tensor(loss_hessian, "GGN metric loss hessian")

    return jacobian, loss_hessian


def _ggn_metric_multiply(
    operator: OperatorSpec,
    batch: Batch,
    vector: TensorTree,
    settings: Mapping[str, Any],
) -> TensorTree:
    _ = operator
    flat_vector = runtime_values.flatten_vector(vector)
    jacobian, loss_hessian = _ggn_metric_factors(batch, vector)
    output_vector = layout.matmul_runtime(settings, jacobian, flat_vector)
    loss_vector = layout.matmul_runtime(settings, loss_hessian, output_vector)
    result = layout.matmul_runtime(settings, jacobian.T, loss_vector)
    runtime_values.require_finite_tensor(result, "GGN-derived metric result")

    return runtime_values.wrap_flat_vector(vector, result)


def _ggn_derived_inverse_metric_multiply(
    operator: OperatorSpec,
    batch: Batch,
    vector: TensorTree,
    settings: Mapping[str, Any],
) -> TensorTree:
    _ = settings
    flat_vector = runtime_values.flatten_vector(vector)
    jacobian, loss_hessian = _ggn_metric_factors(batch, vector)
    result = _ggn_derived_inverse_metric_flat_rhs(
        operator,
        jacobian,
        loss_hessian,
        flat_vector[:, None],
    ).squeeze(1)
    runtime_values.require_finite_tensor(result, "GGN-derived inverse metric result")

    return runtime_values.wrap_flat_vector(vector, result)


def _ggn_derived_inverse_metric_multiply_batch(
    execution: "runtime_values.StandardExecution",
) -> TensorTree:
    vector_batch = _flat_inverse_metric_vector_batch(execution)
    jacobian, loss_hessian = _ggn_metric_factors(execution.batch, execution.params)
    result = _ggn_derived_inverse_metric_flat_rhs(
        execution.operator,
        jacobian,
        loss_hessian,
        vector_batch.T,
    ).T
    runtime_values.require_finite_tensor(
        result, "batched GGN-derived inverse metric result"
    )

    return runtime_values.wrap_flat_vector_batch(execution.params, result)


def _ggn_derived_inverse_metric_flat_rhs(
    operator: OperatorSpec,
    jacobian: torch.Tensor,
    loss_hessian: torch.Tensor,
    rhs: torch.Tensor,
) -> torch.Tensor:
    if rhs.ndim != runtime_values.MATRIX_DIMS or rhs.shape[0] != jacobian.shape[1]:
        message = "GGN-derived inverse RHS shape must match parameter width"
        raise MaterializationError(message)

    eigenvalues, eigenvectors = torch.linalg.eigh(loss_hessian)
    _require_nonnegative_spectrum(eigenvalues, "GGN-derived loss Hessian")
    sqrt_loss_hessian = (
        eigenvectors @ torch.diag(torch.sqrt(eigenvalues)) @ eigenvectors.T
    )
    factor = sqrt_loss_hessian @ jacobian
    base_diagonal = _inverse_metric_damping_vector(
        operator,
        jacobian.shape[1],
        dtype=jacobian.dtype,
        device=jacobian.device,
    )
    result = _low_rank_plus_diagonal_inverse_flat_batch(
        factor.T,
        base_diagonal,
        rhs.T,
        "GGN-derived inverse metric",
    ).T
    runtime_values.require_finite_tensor(
        result, "GGN-derived inverse metric flat result"
    )

    return result


def _damped_metric_matrix(matrix: torch.Tensor, damping: float) -> torch.Tensor:
    if damping <= 0.0:
        return matrix

    if matrix.ndim != runtime_values.MATRIX_DIMS or matrix.shape[0] != matrix.shape[1]:
        message = "metric matrix must be square"
        raise MaterializationError(message)

    identity = torch.eye(matrix.shape[0], dtype=matrix.dtype, device=matrix.device)

    return matrix + damping * identity


def _inverse_metric_damping(operator: OperatorSpec) -> float:
    value = operator.semantics.get("damping")

    if not isinstance(value, float):
        message = "inverse metric damping must be a float"
        raise MaterializationError(message)

    if value < 0.0:
        message = "inverse metric damping must be nonnegative"
        raise MaterializationError(message)

    return value


def _damped_diagonal_vector(
    operator: OperatorSpec,
    template: TensorTree,
    diagonal: torch.Tensor,
) -> torch.Tensor:
    damping = _inverse_metric_damping_vector(
        operator,
        diagonal.numel(),
        dtype=diagonal.dtype,
        device=diagonal.device,
    )

    if damping.numel() != runtime_values.flatten_vector(template).numel():
        message = "per_group damping width does not match vector width"
        raise MaterializationError(message)

    return diagonal + damping


def _inverse_metric_damping_vector(
    operator: OperatorSpec,
    width: int,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    if _inverse_metric_damping_kind(operator) == "per_group":
        return _per_group_damping_vector(operator, width, dtype=dtype, device=device)

    damping = _inverse_metric_damping(operator)

    return torch.full((width,), damping, dtype=dtype, device=device)


def _per_group_damping_vector(
    operator: OperatorSpec,
    width: int,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    groups = _inverse_metric_damping_groups(operator)
    values = _inverse_metric_damping_values(operator)
    result = torch.empty((width,), dtype=dtype, device=device)
    expected_start = 0

    for group in groups:
        name = group["name"]
        start = group["start"]
        stop = group["stop"]

        if start != expected_start or stop <= start or stop > width:
            message = "per_group damping groups must partition the vector"
            raise MaterializationError(message)

        result[start:stop] = values[name]
        expected_start = stop

    if expected_start != width:
        message = "per_group damping groups must cover the vector"
        raise MaterializationError(message)

    return result


def _inverse_metric_damping_groups(
    operator: OperatorSpec,
) -> tuple[Mapping[str, Any], ...]:
    value = operator.semantics.get("damping_groups")

    if not isinstance(value, tuple) or not value:
        message = "per_group damping requires parameter-surface group identity"
        raise MaterializationError(message)

    groups = []

    for item in value:
        if not isinstance(item, Mapping):
            message = "per_group damping group identity must be a mapping"
            raise MaterializationError(message)

        name = item.get("name")
        start = item.get("start")
        stop = item.get("stop")

        if not isinstance(name, str) or not name:
            message = "per_group damping group name must be a string"
            raise MaterializationError(message)

        if (
            not isinstance(start, int)
            or isinstance(start, bool)
            or not isinstance(stop, int)
            or isinstance(stop, bool)
        ):
            message = "per_group damping group bounds must be integers"
            raise MaterializationError(message)

        groups.append({"name": name, "start": start, "stop": stop})

    if {group["name"] for group in groups} != set(
        _inverse_metric_damping_values(operator)
    ):
        message = "per_group damping group names must match damping values"
        raise MaterializationError(message)

    return tuple(groups)


def _inverse_metric_damping_values(operator: OperatorSpec) -> Mapping[str, float]:
    value = operator.semantics.get("damping")

    if not isinstance(value, Mapping) or not value:
        message = "inverse metric per_group damping must be a nonempty mapping"
        raise MaterializationError(message)

    result = {}

    for key, damping in value.items():
        if not isinstance(key, str):
            message = "inverse metric per_group damping keys must be strings"
            raise MaterializationError(message)

        if (
            not isinstance(damping, float | int)
            or isinstance(damping, bool)
            or damping < 0.0
        ):
            message = "inverse metric per_group damping values must be nonnegative"
            raise MaterializationError(message)

        result[key] = float(damping)

    return result


def _inverse_metric_group_damping(operator: OperatorSpec, group: str) -> float:
    values = _inverse_metric_damping_values(operator)
    value = values.get(group)

    if value is None:
        message = f"inverse metric per_group damping is missing group: {group}"
        raise MaterializationError(message)

    return value


def inverse_metric_min_damping(operator: OperatorSpec) -> float:
    """Return the inverse metric min damping.

    Returns:
        The inverse metric min damping.
    """
    if _inverse_metric_damping_kind(operator) == "per_group":
        return min(_inverse_metric_damping_values(operator).values())

    return _inverse_metric_damping(operator)


def inverse_metric_damping_payload(
    operator: OperatorSpec,
) -> float | Mapping[str, float]:
    """Return the inverse metric damping payload.

    Returns:
        The inverse metric damping payload.
    """
    if _inverse_metric_damping_kind(operator) == "per_group":
        return _inverse_metric_damping_values(operator)

    return _inverse_metric_damping(operator)


def minimum_inverse_metric_damping(damping: float | Mapping[str, float]) -> float:
    """Return the inverse metric damping.

    Returns:
        The inverse metric damping.
    """
    if isinstance(damping, Mapping):
        return min(_required_group_damping_mapping(damping).values())

    return damping


def _inverse_metric_damping_product(
    operator: OperatorSpec,
    flat_vector: torch.Tensor,
    damping: float | Mapping[str, float],
) -> torch.Tensor:
    if isinstance(damping, Mapping):
        values = _per_group_damping_vector(
            operator,
            flat_vector.numel(),
            dtype=flat_vector.dtype,
            device=flat_vector.device,
        )

        return values * flat_vector

    return damping * flat_vector


def _resolved_group_damping(
    damping: float | Mapping[str, float],
    damping_kind: str,
    group: str,
) -> float:
    if damping_kind == "per_group":
        values = _required_group_damping_mapping(damping)
        value = values.get(group)

        if value is None:
            message = f"per_group damping is missing group: {group}"
            raise MaterializationError(message)

        return value

    if not isinstance(damping, float):
        message = "inverse metric damping must be a float"
        raise MaterializationError(message)

    return damping


def _required_group_damping_mapping(
    damping: float | Mapping[str, float],
) -> dict[str, float]:
    if not isinstance(damping, Mapping):
        message = "per_group damping payload must be a mapping"
        raise MaterializationError(message)

    result = {}

    for key, value in damping.items():
        if not isinstance(key, str):
            message = "per_group damping keys must be strings"
            raise MaterializationError(message)

        if not isinstance(value, float | int) or isinstance(value, bool) or value < 0.0:
            message = "per_group damping values must be nonnegative"
            raise MaterializationError(message)

        result[key] = float(value)

    return result


def _resolved_group_damping_kind(damping_kind: str) -> str:
    if damping_kind == "per_group":
        return "scalar"

    return damping_kind


def inverse_metric_tolerance(operator: OperatorSpec) -> float | None:
    """Return the inverse metric tolerance.

    Returns:
        The inverse metric tolerance.

    Raises:
        MaterializationError: If the declared inputs are invalid.
    """
    value = operator.semantics.get("tol")

    if value is None:
        return None

    if not isinstance(value, float) or not math.isfinite(value) or value <= 0.0:
        message = "inverse metric tol must be positive and finite"
        raise MaterializationError(message)

    return value


def _inverse_metric_damping_kind(operator: OperatorSpec) -> str:
    value = operator.semantics.get("damping_kind")

    if not isinstance(value, str):
        message = "inverse metric damping_kind must be a string"
        raise MaterializationError(message)

    return value


def _inverse_metric_damping_policy(operator: OperatorSpec) -> str | None:
    value = operator.semantics.get("damping_policy")

    if value is None:
        return None

    if not isinstance(value, str):
        message = "inverse metric damping_policy must be a string"
        raise MaterializationError(message)

    return value


def run_metric(execution: runtime_values.StandardExecution) -> TensorTree:
    """Run metric.

    Returns:
        The metric result.
    """
    runtime_values.require_path(
        execution.operator.kind,
        execution.path,
        (
            runtime_values.METRIC_DENSE_PATH,
            runtime_values.METRIC_FACTORIZED_PATH,
            runtime_values.METRIC_BLOCKWISE_PATH,
            runtime_values.METRIC_STREAMING_PATH,
        ),
    )
    _require_metric_accumulation_settings(
        execution.path,
        execution.candidate.settings,
    )
    if execution.compiled_inner is None:
        result = metric_multiply_by_path(
            execution.operator,
            execution.batch,
            execution.vector,
            execution.path,
            execution.candidate.settings,
        )
    else:
        result = execution.compiled_inner()

    runtime_values.require_finite_tree(result, "metric result")

    return result


def run_sqrt_metric(execution: runtime_values.StandardExecution) -> TensorTree:
    """Run sqrt metric.

    Returns:
        The sqrt metric result.
    """
    runtime_values.require_path(
        execution.operator.kind,
        execution.path,
        (
            runtime_values.SQRT_METRIC_CLOSED_FORM_PATH,
            runtime_values.SQRT_METRIC_CHOLESKY_PATH,
            runtime_values.SQRT_METRIC_EIGENBASIS_PATH,
            runtime_values.SQRT_METRIC_LANCZOS_PATH,
        ),
    )
    if execution.compiled_inner is None:
        result = metric_square_root_apply(
            execution,
            inverse=execution.operator.kind == "inverse_sqrt_metric",
            adjoint=False,
        )
    else:
        result = execution.compiled_inner()

    runtime_values.require_finite_tree(result, "metric square-root result")

    return result


def run_metric_inner(execution: runtime_values.StandardExecution) -> TensorTree:
    """Run metric inner.

    Returns:
        The metric inner result.
    """
    runtime_values.require_path(
        execution.operator.kind,
        execution.path,
        (
            runtime_values.METRIC_INNER_MULTIPLY_REDUCE_PATH,
            runtime_values.METRIC_INNER_FACTORED_GRAM_PATH,
            runtime_values.METRIC_INNER_SQRT_REDUCE_PATH,
        ),
    )
    _require_metric_inner_norm_path(execution, "metric_inner.reduction_path")
    if execution.compiled_inner is None:
        left, right = _metric_inner_vectors(execution.vector)
        result = _metric_inner_by_path(execution, left, right)
    else:
        result = _metric_inner_compiled_tensor(
            execution.compiled_inner(),
            "metric inner result",
        )

    runtime_values.require_finite_tensor(result, "metric inner result")

    return result


def run_inverse_metric_inner(
    execution: runtime_values.StandardExecution,
) -> TensorTree:
    """Run inverse metric inner.

    Returns:
        The inverse metric inner result.
    """
    runtime_values.require_path(
        execution.operator.kind,
        execution.path,
        (
            runtime_values.INVERSE_METRIC_INNER_SOLVE_REDUCE_PATH,
            runtime_values.INVERSE_METRIC_INNER_FACTORED_GRAM_PATH,
            runtime_values.INVERSE_METRIC_INNER_SQRT_REDUCE_PATH,
        ),
    )
    _require_metric_inner_norm_path(execution, "inverse_metric_inner.reduction_path")
    if execution.compiled_inner is None:
        left, right = _metric_inner_vectors(execution.vector)
        result = _inverse_metric_inner_by_path(execution, left, right)
    else:
        result = _metric_inner_compiled_tensor(
            execution.compiled_inner(),
            "inverse metric inner result",
        )

    runtime_values.require_finite_tensor(result, "inverse metric inner result")

    return result


def _metric_inner_compiled_tensor(result: TensorTree, name: str) -> torch.Tensor:
    if not isinstance(result, torch.Tensor):
        message = f"{name} must be a tensor"
        raise MaterializationError(message)

    return result


def _require_metric_inner_norm_path(
    execution: runtime_values.StandardExecution,
    reduction_key: str,
) -> None:
    if execution.operator.semantics.get("as_norm") is not True:
        return

    if execution.candidate.settings.get(reduction_key) == "sqrt_apply_reduce":
        return

    message = f"{reduction_key}=sqrt_apply_reduce is required when as_norm=True"
    raise MaterializationError(message)


def _metric_inner_vectors(vector: TensorTree) -> tuple[TensorTree, TensorTree]:
    if (
        not isinstance(vector, tuple)
        or len(vector) != runtime_values.METRIC_INNER_VECTOR_COUNT
    ):
        message = "metric inner vector input must be a (left, right) tuple"
        raise MaterializationError(message)

    left, right = vector

    return left, right


def _metric_inner_by_path(
    execution: runtime_values.StandardExecution,
    left: TensorTree,
    right: TensorTree,
) -> torch.Tensor:
    if execution.path == runtime_values.METRIC_INNER_MULTIPLY_REDUCE_PATH:
        return _metric_inner_multiply_then_reduce(execution, left, right)

    if execution.path == runtime_values.METRIC_INNER_FACTORED_GRAM_PATH:
        return _metric_inner_factored_gram(execution, left, right)

    if execution.path == runtime_values.METRIC_INNER_SQRT_REDUCE_PATH:
        return _metric_inner_sqrt_apply_reduce(
            execution,
            left,
            right,
            inverse=False,
        )

    message = f"metric inner path is not lowered: {execution.path}"
    raise MaterializationError(message)


def _inverse_metric_inner_by_path(
    execution: runtime_values.StandardExecution,
    left: TensorTree,
    right: TensorTree,
) -> torch.Tensor:
    if execution.path == runtime_values.INVERSE_METRIC_INNER_SOLVE_REDUCE_PATH:
        return _inverse_metric_inner_solve_then_reduce(execution, left, right)

    if execution.path == runtime_values.INVERSE_METRIC_INNER_FACTORED_GRAM_PATH:
        return _inverse_metric_inner_factored_gram(execution, left, right)

    if execution.path == runtime_values.INVERSE_METRIC_INNER_SQRT_REDUCE_PATH:
        return _metric_inner_sqrt_apply_reduce(
            execution,
            left,
            right,
            inverse=True,
        )

    message = f"inverse metric inner path is not lowered: {execution.path}"
    raise MaterializationError(message)


def _metric_inner_multiply_then_reduce(
    execution: runtime_values.StandardExecution,
    left: TensorTree,
    right: TensorTree,
) -> torch.Tensor:
    metric_path = _metric_runtime_path_from_settings(execution.candidate.settings)

    return _metric_inner_multiply_reduce_by_metric_path(
        execution,
        left,
        right,
        metric_path,
        "metric_inner.multi_rhs",
    )


def _inverse_metric_inner_solve_then_reduce(
    execution: runtime_values.StandardExecution,
    left: TensorTree,
    right: TensorTree,
) -> torch.Tensor:
    inverse_path = _inverse_metric_runtime_path_from_settings(
        execution.candidate.settings
    )

    return _metric_inner_inverse_reduce_by_path(
        execution,
        left,
        right,
        inverse_path,
        "inverse_metric_inner.multi_rhs",
    )


def _metric_inner_factored_gram(
    execution: runtime_values.StandardExecution,
    left: TensorTree,
    right: TensorTree,
) -> torch.Tensor:
    return _metric_inner_factorized_multiply_reduce(execution, left, right)


def _inverse_metric_inner_factored_gram(
    execution: runtime_values.StandardExecution,
    left: TensorTree,
    right: TensorTree,
) -> torch.Tensor:
    return _metric_inner_inverse_reduce_by_path(
        execution,
        left,
        right,
        runtime_values.INVERSE_METRIC_FACTORIZED_PATH,
        "inverse_metric_inner.multi_rhs",
    )


def _metric_inner_factorized_multiply_reduce(
    execution: runtime_values.StandardExecution,
    left: TensorTree,
    right: TensorTree,
) -> torch.Tensor:
    return _metric_inner_multiply_reduce_by_metric_path(
        execution,
        left,
        right,
        runtime_values.METRIC_FACTORIZED_PATH,
        "metric_inner.multi_rhs",
    )


def _metric_inner_multiply_reduce_by_metric_path(
    execution: runtime_values.StandardExecution,
    left: TensorTree,
    right: TensorTree,
    metric_path: str,
    multi_rhs_key: str,
) -> torch.Tensor:
    def left_matrix_builder(left_block: TensorTree) -> torch.Tensor:
        return _metric_inner_flat_block(execution, left_block, 0)

    def right_matrix_builder(right_block: TensorTree) -> torch.Tensor:
        right_matrix = _metric_inner_flat_block(execution, right_block, 1)

        return _metric_apply_flat_batch(
            execution.operator,
            execution.batch,
            execution.params,
            right_matrix,
            0.0,
            metric_path,
            execution.candidate.settings,
        )

    def right_vector_product(right_vector: TensorTree) -> TensorTree:
        return metric_multiply_by_path(
            execution.operator,
            execution.batch,
            right_vector,
            metric_path,
            execution.candidate.settings,
        )

    return _metric_inner_reduce_by_product_builders(
        execution,
        left,
        right,
        multi_rhs_key,
        left_matrix_builder=left_matrix_builder,
        right_matrix_builder=right_matrix_builder,
        right_vector_product=right_vector_product,
        left_vector_product=None,
    )


def _metric_inner_inverse_reduce_by_path(
    execution: runtime_values.StandardExecution,
    left: TensorTree,
    right: TensorTree,
    inverse_path: str,
    multi_rhs_key: str,
) -> torch.Tensor:
    solve_execution = dataclasses.replace(
        execution,
        path=inverse_path,
        vector=right,
    )

    def left_matrix_builder(left_block: TensorTree) -> torch.Tensor:
        return _metric_inner_flat_block(execution, left_block, 0)

    def right_matrix_builder(right_block: TensorTree) -> torch.Tensor:
        right_execution = _metric_inner_side_execution(
            solve_execution,
            right_block,
            1,
            path=inverse_path,
        )
        product = _run_inverse_metric_rhs_batch(right_execution)

        return _metric_inner_flat_leading_block(execution, product)

    def right_vector_product(right_vector: TensorTree) -> TensorTree:
        right_execution = dataclasses.replace(
            solve_execution,
            vector=right_vector,
        )

        return _inverse_metric_solve_by_path(right_execution)

    return _metric_inner_reduce_by_product_builders(
        execution,
        left,
        right,
        multi_rhs_key,
        left_matrix_builder=left_matrix_builder,
        right_matrix_builder=right_matrix_builder,
        right_vector_product=right_vector_product,
        left_vector_product=None,
    )


def _metric_inner_sqrt_apply_reduce(
    execution: runtime_values.StandardExecution,
    left: TensorTree,
    right: TensorTree,
    *,
    inverse: bool,
) -> torch.Tensor:
    multi_rhs_key = (
        "inverse_metric_inner.multi_rhs" if inverse else "metric_inner.multi_rhs"
    )
    sqrt_path = _sqrt_metric_runtime_path_from_settings(execution.candidate.settings)

    def left_matrix_builder(left_block: TensorTree) -> torch.Tensor:
        return _metric_square_root_apply_flat_batch(
            execution,
            left_block,
            0,
            sqrt_path,
            inverse=inverse,
        )

    def right_matrix_builder(right_block: TensorTree) -> torch.Tensor:
        return _metric_square_root_apply_flat_batch(
            execution,
            right_block,
            1,
            sqrt_path,
            inverse=inverse,
        )

    def factor_vector(vector: TensorTree) -> TensorTree:
        return _metric_square_root_apply_vector(
            execution,
            vector,
            sqrt_path,
            inverse=inverse,
            adjoint=True,
        )

    return _metric_inner_reduce_by_product_builders(
        execution,
        left,
        right,
        multi_rhs_key,
        left_matrix_builder=left_matrix_builder,
        right_matrix_builder=right_matrix_builder,
        right_vector_product=factor_vector,
        left_vector_product=factor_vector,
    )


def _metric_inner_reduce_by_product_builders(
    execution: runtime_values.StandardExecution,
    left: TensorTree,
    right: TensorTree,
    multi_rhs_key: str,
    *,
    left_matrix_builder: Callable[[TensorTree], torch.Tensor],
    right_matrix_builder: Callable[[TensorTree], torch.Tensor],
    right_vector_product: Callable[[TensorTree], TensorTree],
    left_vector_product: Callable[[TensorTree], TensorTree] | None,
) -> torch.Tensor:
    block_mode = _metric_inner_block_mode(execution, multi_rhs_key)

    if block_mode is not None:
        left_matrix = left_matrix_builder(left)

        if block_mode == "manual_batch":
            return _metric_inner_manual_block_reduce(
                execution,
                left_matrix,
                right,
                right_matrix_builder,
            )

        if block_mode == "vmap":
            return _metric_inner_vmap_block_reduce(
                execution,
                left_matrix,
                right,
                1,
                right_vector_product,
            )

        right_matrix = right_matrix_builder(right)

        return _metric_inner_reduce_matrices(execution, left_matrix, right_matrix)

    if left_vector_product is None:
        right_product = right_vector_product(right)

        return runtime.tree_dot_runtime(
            execution.candidate.settings, left, right_product
        )

    left_product = left_vector_product(left)
    right_product = right_vector_product(right)

    return runtime.tree_dot_runtime(
        execution.candidate.settings, left_product, right_product
    )


def _metric_inner_block_mode(
    execution: runtime_values.StandardExecution,
    key: str,
) -> str | None:
    if key not in execution.candidate.settings:
        message = f"{key} is required"
        raise MaterializationError(message)

    value = execution.candidate.settings[key]

    if value == "single_column":
        if "vectorization.mode" in execution.candidate.settings:
            message = f"vectorization.mode requires {key}=block"
            raise MaterializationError(message)

        return None

    if value != "block":
        message = f"{key} is unsupported: {value}"
        raise MaterializationError(message)

    mode = execution.candidate.settings.get("vectorization.mode")

    if mode == "single_loop":
        return mode

    if mode == "manual_batch":
        runtime_values.manual_vector_batch_size(execution.candidate.settings)

        return mode

    if mode == "vmap":
        vectorization.vmap_chunk_size(execution.candidate.settings)

        return mode

    message = (
        f"{key}=block requires vectorization.mode=single_loop, manual_batch, or vmap"
    )
    raise MaterializationError(message)


def _metric_inner_vmap_block_reduce(
    execution: runtime_values.StandardExecution,
    left_matrix: torch.Tensor,
    right: TensorTree,
    side: int,
    right_product: Callable[[TensorTree], TensorTree],
) -> torch.Tensor:
    right_in_dims = _metric_inner_vector_in_dims(
        execution.candidate.settings,
        right,
        side,
    )
    chunk_size = vectorization.vmap_chunk_size(execution.candidate.settings)

    def flat_right_product(right_vector: TensorTree) -> torch.Tensor:
        return runtime_values.flatten_vector(
            runtime_values.call_with_deferred_finite_checks(right_product, right_vector)
        )

    product_matrix = vectorization.torch_func_vmap(
        flat_right_product,
        in_dims=(right_in_dims,),
        randomness=execution.candidate.settings["vectorization.randomness"],
        chunk_size=chunk_size,
    )(right)
    runtime_values.require_finite_tensor(
        product_matrix, "metric inner vmap block result"
    )

    return _metric_inner_reduce_matrices(execution, left_matrix, product_matrix)


def _metric_inner_manual_block_reduce(
    execution: runtime_values.StandardExecution,
    left_matrix: torch.Tensor,
    right: TensorTree,
    right_matrix_builder: Callable[[TensorTree], torch.Tensor],
) -> torch.Tensor:
    parts = []

    for right_chunk in _metric_inner_manual_right_chunks(execution, right):
        right_matrix = right_matrix_builder(right_chunk)
        parts.append(
            _metric_inner_reduce_matrices(execution, left_matrix, right_matrix)
        )

    result = torch.cat(tuple(parts), dim=1)
    runtime_values.require_finite_tensor(
        result, "metric inner manual-batch block result"
    )

    return result


def _metric_inner_manual_right_chunks(
    execution: runtime_values.StandardExecution,
    right: TensorTree,
) -> tuple[TensorTree, ...]:
    right_in_dims = _metric_inner_vector_in_dims(
        execution.candidate.settings,
        right,
        1,
    )
    right_count = runtime_values.vector_tree_batch_size(right, right_in_dims)
    batch_size = runtime_values.manual_vector_batch_size(execution.candidate.settings)
    chunks = []

    for start in range(0, right_count, batch_size):
        stop = min(start + batch_size, right_count)
        chunks.append(
            runtime_values.vector_tree_slice(right, right_in_dims, start, stop)
        )

    return tuple(chunks)


def _metric_inner_side_execution(
    execution: runtime_values.StandardExecution,
    vector: TensorTree,
    side: int,
    *,
    path: str,
) -> runtime_values.StandardExecution:
    settings = _metric_inner_side_settings(execution.candidate.settings, side)
    candidate = dataclasses.replace(execution.candidate, settings=settings)

    return dataclasses.replace(
        execution,
        candidate=candidate,
        path=path,
        vector=vector,
    )


def _metric_inner_side_settings(
    settings: Mapping[str, Any],
    side: int,
) -> Mapping[str, Any]:
    raw_in_dims = settings.get("vectorization.in_dims")

    if not isinstance(raw_in_dims, tuple):
        return settings

    if len(raw_in_dims) != runtime_values.METRIC_INNER_VECTOR_COUNT:
        message = "metric inner vectorization.in_dims must cover left and right"
        raise MaterializationError(message)

    result = dict(settings)
    result["vectorization.in_dims"] = raw_in_dims[side]

    return result


def _metric_inner_flat_leading_block(
    execution: runtime_values.StandardExecution,
    vector: TensorTree,
) -> torch.Tensor:
    return runtime_values.flatten_vector_batch(
        execution.params,
        vector,
        _leading_vector_batch_in_dims(execution.params),
    )


def _leading_vector_batch_in_dims(template: TensorTree) -> Any:
    if isinstance(template, torch.Tensor):
        return 0

    if runtime_values.is_tensor_tree_dict(template):
        return {key: _leading_vector_batch_in_dims(template[key]) for key in template}

    if runtime_values.is_tensor_tree_tuple(template):
        return tuple(_leading_vector_batch_in_dims(value) for value in template)

    message = f"unsupported tensor tree node: {type(template).__name__}"
    raise MaterializationError(message)


def _metric_inner_flat_block(
    execution: runtime_values.StandardExecution,
    vector: TensorTree,
    side: int,
) -> torch.Tensor:
    in_dims = _metric_inner_vector_in_dims(
        execution.candidate.settings,
        vector,
        side,
    )

    return runtime_values.flatten_vector_batch(execution.params, vector, in_dims)


def _metric_inner_vector_in_dims(
    settings: Mapping[str, Any],
    vector: TensorTree,
    side: int,
) -> Any:
    raw_in_dims = settings.get("vectorization.in_dims")

    if isinstance(raw_in_dims, tuple):
        if len(raw_in_dims) != runtime_values.METRIC_INNER_VECTOR_COUNT:
            message = "metric inner vectorization.in_dims must cover left and right"
            raise MaterializationError(message)

        raw_in_dims = raw_in_dims[side]

    return vectorization.validate_vector_tree_in_dims(vector, raw_in_dims)


def _metric_inner_reduce_matrices(
    execution: runtime_values.StandardExecution,
    left_matrix: torch.Tensor,
    right_matrix: torch.Tensor,
) -> torch.Tensor:
    if (
        left_matrix.ndim != runtime_values.MATRIX_DIMS
        or right_matrix.ndim != runtime_values.MATRIX_DIMS
    ):
        message = "metric inner block operands must flatten to matrices"
        raise MaterializationError(message)

    if left_matrix.shape[1] != right_matrix.shape[1]:
        message = "metric inner block widths differ"
        raise MaterializationError(message)

    result = layout.matmul_runtime(
        execution.candidate.settings, left_matrix, right_matrix.T
    )
    runtime_values.require_finite_tensor(result, "metric inner block result")

    return result


def _sqrt_metric_runtime_path_from_settings(settings: Mapping[str, Any]) -> str:
    return runtime_values.runtime_path_from_settings(
        settings,
        "sqrt_metric",
        "sqrt_metric.factor_path",
    )


def _metric_square_root_apply_vector(
    execution: runtime_values.StandardExecution,
    vector: TensorTree,
    path: str,
    *,
    inverse: bool,
    adjoint: bool,
) -> TensorTree:
    sqrt_execution = dataclasses.replace(
        execution,
        path=path,
        vector=vector,
    )

    return metric_square_root_apply(sqrt_execution, inverse=inverse, adjoint=adjoint)


def _metric_square_root_apply_flat_batch(
    execution: runtime_values.StandardExecution,
    vector: TensorTree,
    side: int,
    path: str,
    *,
    inverse: bool,
) -> torch.Tensor:
    flat_vectors = _metric_inner_flat_block(execution, vector, side)
    rows = []

    for flat_vector in flat_vectors:
        vector_tree = runtime_values.wrap_flat_vector(execution.params, flat_vector)
        result = _metric_square_root_apply_vector(
            execution,
            vector_tree,
            path,
            inverse=inverse,
            adjoint=True,
        )
        rows.append(runtime_values.flatten_vector(result))

    matrix = torch.stack(tuple(rows), dim=0)
    runtime_values.require_finite_tensor(matrix, "metric square-root block result")

    return matrix


def metric_square_root_apply(
    execution: runtime_values.StandardExecution,
    *,
    inverse: bool,
    adjoint: bool,
) -> TensorTree:
    """Return the metric square root apply.

    Returns:
        The metric square root apply.
    """
    path = execution.path
    vector = execution.vector

    if path == runtime_values.SQRT_METRIC_CLOSED_FORM_PATH:
        return _closed_form_metric_square_root_apply(
            execution,
            inverse=inverse,
            adjoint=adjoint,
        )

    if path == runtime_values.SQRT_METRIC_LANCZOS_PATH:
        _require_metric_representation(execution.operator, ("matrix_free",))
        flat_result = _lanczos_matrix_free_metric_square_root_product(
            execution,
            runtime_values.flatten_vector(vector),
            inverse=inverse,
        )
        runtime_values.require_finite_tensor(flat_result, "metric square-root result")

        return runtime_values.wrap_flat_vector(vector, flat_result)

    matrix = metric_dense_matrix(execution.operator, execution.batch, vector)
    flat_vector = runtime_values.flatten_vector(vector)

    if path == runtime_values.SQRT_METRIC_EIGENBASIS_PATH:
        # The symmetric root is applied in factored form; the adjoint of a
        # symmetric factor is the factor itself.
        flat_result = _eigenbasis_metric_square_root_product(
            execution,
            matrix,
            flat_vector,
            inverse=inverse,
        )
    else:
        factor = _metric_square_root_factor_matrix(
            execution,
            matrix,
            inverse=inverse,
            path=path,
        )
        flat_result = factor.T @ flat_vector if adjoint else factor @ flat_vector

    runtime_values.require_finite_tensor(flat_result, "metric square-root result")

    return runtime_values.wrap_flat_vector(vector, flat_result)


def _closed_form_metric_square_root_apply(
    execution: runtime_values.StandardExecution,
    *,
    inverse: bool,
    adjoint: bool,
) -> TensorTree:
    representation = metric_representation_kind(execution.operator)

    if representation == "ekfac_factors":
        return _ekfac_square_root_apply(execution, inverse=inverse)

    if representation == "kfac_factors":
        return _kfac_square_root_apply(execution, inverse=inverse)

    if representation == "low_rank_factors":
        return _low_rank_square_root_apply(
            execution,
            inverse=inverse,
            adjoint=adjoint,
        )

    if representation == "ggn_derived_factors":
        return _ggn_derived_square_root_apply(
            execution,
            inverse=inverse,
            adjoint=adjoint,
        )

    _require_metric_representation(execution.operator, ("diagonal_tree",))
    diagonal = _metric_diagonal_tree(execution.batch, execution.vector)

    if inverse:
        denominator = runtime_values.flatten_vector(
            _diagonal_inverse_denominator(
                execution.operator,
                diagonal,
                execution.candidate.settings,
            )
        )
        factors = torch.rsqrt(denominator)
    else:
        flat_diagonal = runtime_values.flatten_vector(diagonal)
        _require_positive_spectrum(flat_diagonal, "diagonal metric square root")
        factors = torch.sqrt(flat_diagonal)

    flat_result = factors * runtime_values.flatten_vector(execution.vector)
    runtime_values.require_finite_tensor(
        flat_result, "closed-form metric square-root result"
    )

    return runtime_values.wrap_flat_vector(execution.vector, flat_result)


def _rectangular_square_root_apply(
    execution: runtime_values.StandardExecution,
    *,
    adjoint: bool,
    name: str,
    widths: tuple[int, int],
    width_labels: tuple[str, str],
    products: Sequence[Callable[[torch.Tensor], torch.Tensor]],
) -> TensorTree:
    branch = 0 if adjoint else 1
    role = ("adjoint ", "")[branch]
    flat_vector = runtime_values.flatten_vector(execution.vector)

    if flat_vector.numel() != widths[branch]:
        message = f"{name} {role}input must match {width_labels[branch]}"
        raise MaterializationError(message)

    flat_result = products[branch](flat_vector)
    runtime_values.require_finite_tensor(flat_result, f"{name} {role}result")

    if adjoint:
        return flat_result

    return runtime_values.wrap_flat_vector(execution.params, flat_result)


def _low_rank_square_root_apply(
    execution: runtime_values.StandardExecution,
    *,
    inverse: bool,
    adjoint: bool,
) -> TensorTree:
    basis, diagonal = _low_rank_factors(execution.batch, execution.params)

    if inverse:
        base_diagonal = _damped_diagonal_vector(
            execution.operator,
            execution.params,
            diagonal,
        )
        flat_result = _low_rank_plus_diagonal_inverse_square_root_flat_apply(
            basis,
            base_diagonal,
            runtime_values.flatten_vector(execution.vector),
            adjoint=adjoint,
        )

        return runtime_values.wrap_flat_vector(execution.params, flat_result)

    _require_nonnegative_spectrum(diagonal, "low-rank diagonal square root")
    rank = basis.shape[1]
    width = diagonal.numel()
    root_diagonal = torch.sqrt(diagonal)

    def product(flat_vector: torch.Tensor) -> torch.Tensor:
        return basis @ flat_vector[:rank] + root_diagonal * flat_vector[rank:]

    return _rectangular_square_root_apply(
        execution,
        adjoint=adjoint,
        name="low-rank square-root",
        widths=(width, rank + width),
        width_labels=("parameter width", "rank plus parameter width"),
        products=(
            lambda flat_vector: torch.cat((
                basis.T @ flat_vector,
                root_diagonal * flat_vector,
            )),
            product,
        ),
    )


def _low_rank_plus_diagonal_inverse_square_root_flat_apply(
    basis: torch.Tensor,
    base_diagonal: torch.Tensor,
    vector: torch.Tensor,
    *,
    adjoint: bool,
) -> torch.Tensor:
    _require_positive_spectrum(base_diagonal, "low-rank inverse square root base")
    root_base = torch.sqrt(base_diagonal)
    scaled_basis = basis / root_base.unsqueeze(1)

    if adjoint:
        base_scaled = vector / root_base
        result = _identity_plus_low_rank_inverse_square_root_apply(
            scaled_basis,
            base_scaled,
        )
    else:
        reduced = _identity_plus_low_rank_inverse_square_root_apply(
            scaled_basis,
            vector,
        )
        result = reduced / root_base

    runtime_values.require_finite_tensor(result, "low-rank inverse square-root result")

    return result


def _identity_plus_low_rank_inverse_square_root_apply(
    basis: torch.Tensor,
    vector: torch.Tensor,
) -> torch.Tensor:
    if basis.shape[0] != vector.numel():
        message = "inverse square-root basis width differs from vector"
        raise MaterializationError(message)

    if basis.shape[1] == 0:
        return vector

    gram = basis.T @ basis
    eigenvalues, eigenvectors = torch.linalg.eigh(gram)
    _require_nonnegative_spectrum(eigenvalues, "low-rank inverse square-root Gram")
    positive = eigenvalues > 0

    if not torch.any(positive):
        return vector

    active_values = eigenvalues[positive]
    active_vectors = eigenvectors[:, positive]
    orthonormal = basis @ active_vectors / torch.sqrt(active_values).unsqueeze(0)
    coefficients = orthonormal.T @ vector
    scales = torch.rsqrt(1.0 + active_values) - 1.0
    result = vector + orthonormal @ (scales * coefficients)
    runtime_values.require_finite_tensor(
        result, "low-rank inverse square-root core result"
    )

    return result


def _ggn_derived_square_root_apply(
    execution: runtime_values.StandardExecution,
    *,
    inverse: bool,
    adjoint: bool,
) -> TensorTree:
    jacobian, loss_hessian = _ggn_metric_factors(execution.batch, execution.params)
    loss_root = _psd_square_root(loss_hessian, "GGN-derived loss Hessian")

    if inverse:
        factor_basis = (loss_root @ jacobian).T
        diagonal = _inverse_metric_damping_vector(
            execution.operator,
            jacobian.shape[1],
            dtype=jacobian.dtype,
            device=jacobian.device,
        )
        flat_result = _low_rank_plus_diagonal_inverse_square_root_flat_apply(
            factor_basis,
            diagonal,
            runtime_values.flatten_vector(execution.vector),
            adjoint=adjoint,
        )

        return runtime_values.wrap_flat_vector(execution.params, flat_result)

    return _rectangular_square_root_apply(
        execution,
        adjoint=adjoint,
        name="GGN-derived square-root",
        widths=(jacobian.shape[1], jacobian.shape[0]),
        width_labels=("parameter width", "output width"),
        products=(
            lambda flat_vector: loss_root @ (jacobian @ flat_vector),
            lambda flat_vector: jacobian.T @ (loss_root @ flat_vector),
        ),
    )


def _psd_square_root(matrix: torch.Tensor, label: str) -> torch.Tensor:
    eigenvalues, eigenvectors = torch.linalg.eigh(matrix)
    _require_nonnegative_spectrum(eigenvalues, label)

    result = eigenvectors @ torch.diag(torch.sqrt(eigenvalues)) @ eigenvectors.T
    runtime_values.require_finite_tensor(result, f"{label} square root")

    return result


def _metric_square_root_factor_matrix(
    execution: runtime_values.StandardExecution,
    matrix: torch.Tensor,
    *,
    inverse: bool,
    path: str,
) -> torch.Tensor:
    if inverse:
        factor_matrix = torch.linalg.inv(
            inverse_metric_matrix(
                execution.operator,
                matrix,
                execution.batch,
                execution.vector,
            )
        )
    else:
        factor_matrix = matrix

    if path == runtime_values.SQRT_METRIC_CHOLESKY_PATH:
        runtime_values.require_positive_definite_matrix(
            factor_matrix, "Cholesky square root"
        )

        return torch.linalg.cholesky(factor_matrix)

    message = f"metric square-root path is not lowered: {path}"
    raise MaterializationError(message)


def _eigenbasis_metric_square_root_product(
    execution: runtime_values.StandardExecution,
    matrix: torch.Tensor,
    flat_vector: torch.Tensor,
    *,
    inverse: bool,
) -> torch.Tensor:
    if inverse:
        damped = inverse_metric_matrix(
            execution.operator,
            matrix,
            execution.batch,
            execution.vector,
        )
        eigenvalues, eigenvectors = torch.linalg.eigh(damped)
        _require_positive_spectrum(eigenvalues, "eigenbasis square root")
        scales = torch.rsqrt(eigenvalues)
    else:
        eigenvalues, eigenvectors = torch.linalg.eigh(matrix)
        _require_positive_spectrum(eigenvalues, "eigenbasis square root")
        scales = torch.sqrt(eigenvalues)

    return eigenvectors @ (scales * (eigenvectors.T @ flat_vector))


def _lanczos_metric_square_root_product(
    execution: runtime_values.StandardExecution,
    matrix: torch.Tensor,
    vector: torch.Tensor,
    *,
    inverse: bool,
) -> torch.Tensor:
    iterations, transform = _lanczos_sqrt_transform(execution, inverse=inverse)

    return _lanczos_matrix_function_product(matrix, vector, iterations, transform)


def _lanczos_matrix_free_metric_square_root_product(
    execution: runtime_values.StandardExecution,
    vector: torch.Tensor,
    *,
    inverse: bool,
) -> torch.Tensor:
    iterations, transform = _lanczos_sqrt_transform(execution, inverse=inverse)
    damping = inverse_metric_damping_payload(execution.operator) if inverse else 0.0

    def apply(flat_vector: torch.Tensor) -> torch.Tensor:
        return metric_apply_flat(
            execution.operator,
            execution.batch,
            execution.vector,
            flat_vector,
            damping,
            runtime_values.METRIC_STREAMING_PATH,
            execution.candidate.settings,
        )

    result, residual = _lanczos_matrix_function_product_with_residual_from_apply(
        apply,
        vector,
        iterations,
        transform,
    )
    _require_inverse_sqrt_lanczos_tolerance(execution, result, residual, inverse)

    return result


def _require_inverse_sqrt_lanczos_tolerance(
    execution: runtime_values.StandardExecution,
    result: torch.Tensor,
    residual: torch.Tensor,
    inverse: bool,
) -> None:
    if not inverse:
        return

    tolerance = inverse_metric_tolerance(execution.operator)

    if tolerance is None:
        return

    result_norm = result.norm()

    if torch.equal(result_norm, torch.zeros_like(result_norm)):
        if float(residual.item()) <= tolerance:
            return

        message = "inverse_sqrt_metric Lanczos residual exceeded tol"
        raise MaterializationError(message)

    scaled_residual = residual / result_norm

    if float(scaled_residual.item()) <= tolerance:
        return

    message = "inverse_sqrt_metric Lanczos residual exceeded tol"
    raise MaterializationError(message)


def _lanczos_sqrt_transform(
    execution: runtime_values.StandardExecution,
    *,
    inverse: bool,
) -> tuple[int, Callable[[torch.Tensor], torch.Tensor]]:
    iterations = _sqrt_metric_lanczos_iterations(execution.candidate.settings)

    def transform(values: torch.Tensor) -> torch.Tensor:
        if inverse:
            _require_positive_spectrum(values, "Lanczos square-root spectrum")

            return torch.rsqrt(values)

        _require_nonnegative_spectrum(values, "Lanczos square-root spectrum")

        return torch.sqrt(values)

    return iterations, transform


def _lanczos_matrix_function_product(
    matrix: torch.Tensor,
    vector: torch.Tensor,
    iterations: int,
    transform: Callable[[torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    return _lanczos_matrix_function_product_from_apply(
        lambda flat_vector: matrix @ flat_vector,
        vector,
        iterations,
        transform,
    )


def _lanczos_matrix_function_product_from_apply(
    apply: Callable[[torch.Tensor], torch.Tensor],
    vector: torch.Tensor,
    iterations: int,
    transform: Callable[[torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    result, _ = _lanczos_matrix_function_product_with_residual_from_apply(
        apply,
        vector,
        iterations,
        transform,
    )

    return result


def _lanczos_matrix_function_product_with_residual_from_apply(
    apply: Callable[[torch.Tensor], torch.Tensor],
    vector: torch.Tensor,
    iterations: int,
    transform: Callable[[torch.Tensor], torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    norm = vector.norm()

    if torch.equal(norm, torch.zeros_like(norm)):
        residual = torch.zeros((), dtype=vector.dtype, device=vector.device)

        return torch.zeros_like(vector), residual

    basis = []
    alphas = []
    betas = []
    current = vector / norm
    previous = torch.zeros_like(current)
    beta = torch.zeros((), dtype=vector.dtype, device=vector.device)

    for iteration in range(iterations):
        basis.append(current)
        residual = apply(current)
        alpha = current @ residual
        residual = residual - alpha * current - beta * previous
        beta = residual.norm()
        alphas.append(alpha)

        if torch.equal(beta, torch.zeros_like(beta)):
            break

        if iteration == iterations - 1:
            break

        betas.append(beta)
        previous = current
        current = residual / beta

    q_matrix = torch.stack(tuple(basis), dim=1)
    tri = _lanczos_tridiagonal(alphas, betas)
    eigenvalues, eigenvectors = torch.linalg.eigh(tri)
    projected = eigenvectors @ (transform(eigenvalues) * eigenvectors[0] * norm)
    result = q_matrix @ projected
    residual = torch.abs(beta * projected[-1])

    return result, residual


def _lanczos_tridiagonal(
    alphas: Sequence[torch.Tensor],
    betas: Sequence[torch.Tensor],
) -> torch.Tensor:
    tri = torch.diag(torch.stack(tuple(alphas)))

    if betas:
        off_diagonal = torch.stack(tuple(betas))
        tri = tri + torch.diag(off_diagonal, diagonal=1)
        tri = tri + torch.diag(off_diagonal, diagonal=-1)

    return tri


def _sqrt_metric_lanczos_iterations(settings: Mapping[str, Any]) -> int:
    return runtime_values.required_positive_int_setting(
        settings,
        "sqrt_metric.lanczos_iterations",
        "sqrt_metric.lanczos_iterations must be a positive integer",
    )


def _require_positive_spectrum(values: torch.Tensor, label: str) -> None:
    if torch.any(values <= 0):
        message = f"{label} requires positive eigenvalues"
        raise MaterializationError(message)


def _require_nonnegative_spectrum(values: torch.Tensor, label: str) -> None:
    if torch.any(values < 0):
        message = f"{label} requires nonnegative eigenvalues"
        raise MaterializationError(message)


def _require_metric_accumulation_settings(
    metric_path: str,
    settings: Mapping[str, Any],
) -> None:
    value = settings.get("metric.accumulation")

    if metric_path == runtime_values.METRIC_DENSE_PATH:
        if value is not None:
            message = "metric.accumulation applies only to non-dense metric paths"
            raise MaterializationError(message)

        return

    if value is None:
        message = "metric.accumulation is required for non-dense metric paths"
        raise MaterializationError(message)

    expected = (
        "streaming"
        if metric_path == runtime_values.METRIC_STREAMING_PATH
        else "materialized_blocks"
    )

    if value != expected:
        message = f"metric.accumulation must be {expected} for this path"
        raise MaterializationError(message)


def metric_multiply_by_path(
    operator: OperatorSpec,
    batch: Batch,
    vector: TensorTree,
    metric_path: str,
    settings: Mapping[str, Any],
    matrix_free_operators: Mapping[str, Callable[[Batch, TensorTree], TensorTree]]
    | None = None,
) -> TensorTree:
    """Return the metric multiply by path.

    Returns:
        The metric multiply by path.

    Raises:
        MaterializationError: If the declared inputs are invalid.
    """
    if metric_representation_kind(operator) == "matrix_free":
        if metric_path != runtime_values.METRIC_STREAMING_PATH:
            message = "matrix_free metric requires streaming_multiply"
            raise MaterializationError(message)

        return _matrix_free_metric_multiply(
            operator,
            batch,
            vector,
            matrix_free_operators,
        )

    if metric_path == runtime_values.METRIC_FACTORIZED_PATH:
        return _factorized_metric_multiply(operator, batch, vector, settings)

    if metric_path == runtime_values.METRIC_BLOCKWISE_PATH:
        _require_metric_representation(operator, ("block_diagonal",))

        return _block_diagonal_metric_multiply(operator, batch, vector, settings)

    if metric_path == runtime_values.METRIC_STREAMING_PATH:
        return _streaming_metric_multiply(operator, batch, vector, settings)

    if metric_path == runtime_values.METRIC_DENSE_PATH:
        _require_metric_representation(operator, ("dense_matrix",))
        matrix = metric_dense_matrix(operator, batch, vector)
        vector_tensor = runtime_values.flatten_vector(vector)
        runtime_values.require_finite_tensor(matrix, "metric matrix")
        runtime_values.require_finite_tensor(vector_tensor, "metric vector")
        flat_result = layout.matmul_runtime(settings, matrix, vector_tensor)
        runtime_values.require_finite_tensor(flat_result, "metric result")

        return runtime_values.wrap_flat_vector(vector, flat_result)

    message = f"metric path is not lowered: {metric_path}"
    raise MaterializationError(message)


def _matrix_free_metric_multiply(
    operator: OperatorSpec,
    batch: Batch,
    vector: TensorTree,
    matrix_free_operators: Mapping[str, Callable[[Batch, TensorTree], TensorTree]]
    | None,
) -> TensorTree:
    product = _matrix_free_metric_product(operator)
    bindings = matrix_free_operators

    if bindings is None:
        bindings = runtime_values.MATRIX_FREE_RUNTIME_BINDINGS.get()

    selected = None if bindings is None else bindings.get(product)

    if selected is None:
        message = f"matrix_free metric requires selected sibling product: {product}"
        raise MaterializationError(message)

    result = selected(batch, vector)
    runtime_values.require_finite_tree(result, "matrix-free metric result")

    return result


def _factorized_metric_multiply(
    operator: OperatorSpec,
    batch: Batch,
    vector: TensorTree,
    settings: Mapping[str, Any],
) -> TensorTree:
    return _metric_multiply_by_kind(
        operator,
        batch,
        vector,
        settings,
        runners=FACTORIZED_METRIC_MULTIPLY_BY_KIND,
        error_prefix="factorized metric path is not lowered for representation",
    )


def _streaming_metric_multiply(
    operator: OperatorSpec,
    batch: Batch,
    vector: TensorTree,
    settings: Mapping[str, Any],
) -> TensorTree:
    return _metric_multiply_by_kind(
        operator,
        batch,
        vector,
        settings,
        runners=STREAMING_METRIC_MULTIPLY_BY_KIND,
        error_prefix="metric streaming path is not lowered for representation",
    )


def _streaming_low_rank_metric_multiply(
    operator: OperatorSpec,
    batch: Batch,
    vector: TensorTree,
    settings: Mapping[str, Any],
) -> TensorTree:
    _ = operator
    flat_vector = runtime.runtime_intermediate_tensor(
        runtime_values.flatten_vector(vector), settings
    )
    basis, diagonal = _low_rank_factors(batch, vector)
    diagonal = runtime.runtime_intermediate_tensor(diagonal, settings)
    result = runtime.accumulation_tensor(
        diagonal, settings
    ) * runtime.accumulation_tensor(flat_vector, settings)

    for index in range(basis.shape[1]):
        column = runtime.runtime_intermediate_tensor(basis[:, index], settings)
        projection = runtime.dot_runtime(settings, column, flat_vector)
        result = result + runtime.accumulation_tensor(column, settings) * projection

    runtime_values.require_finite_tensor(result, "streaming low-rank metric result")

    return runtime_values.wrap_flat_vector(vector, result)


def _streaming_kfac_metric_multiply(
    operator: OperatorSpec,
    batch: Batch,
    vector: TensorTree,
    settings: Mapping[str, Any],
) -> TensorTree:
    def product(
        _: runtime_values.KFACMetricBlock,
        left: torch.Tensor,
        right: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        left_product = layout.matmul_runtime(settings, left, value)

        return layout.matmul_runtime(settings, left_product, right.T)

    return _kfac_block_results(
        _kfac_blocks(operator),
        _kfac_factor_batch(batch),
        vector,
        product,
        "streaming KFAC metric result",
    )


def _streaming_ggn_metric_multiply(
    operator: OperatorSpec,
    batch: Batch,
    vector: TensorTree,
    settings: Mapping[str, Any],
) -> TensorTree:
    _ = operator
    flat_vector = runtime.runtime_intermediate_tensor(
        runtime_values.flatten_vector(vector), settings
    )
    jacobian, loss_hessian = _ggn_metric_factors(batch, vector)
    output_vector = torch.stack(
        tuple(
            runtime.dot_runtime(
                settings,
                runtime.runtime_intermediate_tensor(row, settings),
                flat_vector,
            )
            for row in jacobian
        )
    )
    loss_vector = torch.stack(
        tuple(
            runtime.dot_runtime(
                settings,
                runtime.runtime_intermediate_tensor(row, settings),
                output_vector,
            )
            for row in loss_hessian
        )
    )
    result = torch.zeros_like(flat_vector)

    for row, weight in zip(jacobian, loss_vector, strict=True):
        result = result + (
            runtime.accumulation_tensor(
                runtime.runtime_intermediate_tensor(row, settings), settings
            )
            * runtime.accumulation_tensor(weight, settings)
        )

    runtime_values.require_finite_tensor(result, "streaming GGN-derived metric result")

    return runtime_values.wrap_flat_vector(vector, result)


InverseMetricBatchRunner = Callable[[runtime_values.StandardExecution], TensorTree]


def _metric_multiply_by_kind(
    operator: OperatorSpec,
    batch: Batch,
    vector: TensorTree,
    settings: Mapping[str, Any],
    *,
    runners: Mapping[str, runtime_values.MetricMultiplyRunner],
    error_prefix: str,
) -> TensorTree:
    kind = metric_representation_kind(operator)
    runner = runners.get(kind)

    if runner is None:
        message = f"{error_prefix}: {kind}"
        raise MaterializationError(message)

    return runner(operator, batch, vector, settings)


FACTORIZED_METRIC_MULTIPLY_BY_KIND = {
    "diagonal_tree": _diagonal_metric_multiply,
    "low_rank_factors": _low_rank_metric_multiply,
    "kfac_factors": _kfac_metric_multiply,
    "ekfac_factors": _ekfac_metric_multiply,
    "ggn_derived_factors": _ggn_metric_multiply,
}

FACTORIZED_INVERSE_METRIC_BY_KIND = {
    "diagonal_tree": _diagonal_inverse_metric_multiply,
    "low_rank_factors": _low_rank_inverse_metric_multiply,
    "kfac_factors": _kfac_inverse_metric_multiply,
    "ekfac_factors": _ekfac_inverse_metric_multiply,
    "ggn_derived_factors": _ggn_derived_inverse_metric_multiply,
}

INVERSE_METRIC_MULTIPLY_BY_PATH = {
    runtime_values.INVERSE_METRIC_FACTORIZED_PATH: (
        FACTORIZED_INVERSE_METRIC_BY_KIND,
        "factorized inverse path is not lowered for representation",
    ),
    runtime_values.INVERSE_METRIC_BLOCKWISE_PATH: (
        {"block_diagonal": _block_diagonal_inverse_metric_multiply},
        "metric representation kind is not supported by path",
    ),
    runtime_values.INVERSE_METRIC_WOODBURY_PATH: (
        {"low_rank_factors": _low_rank_inverse_metric_multiply},
        "metric representation kind is not supported by path",
    ),
}

BLOCK_PRECONDITIONER_BY_KIND = {
    "block_diagonal": _block_diagonal_inverse_metric_multiply,
    "kfac_factors": _kfac_inverse_metric_multiply,
}

FACTORIZED_PRECONDITIONER_BY_KIND = {
    "diagonal_tree": _diagonal_inverse_metric_multiply,
    "low_rank_factors": _low_rank_inverse_metric_multiply,
    "kfac_factors": _kfac_inverse_metric_multiply,
    "ggn_derived_factors": _ggn_derived_inverse_metric_multiply,
}

FACTORIZED_INVERSE_METRIC_BATCH_BY_KIND = {
    "diagonal_tree": _diagonal_inverse_metric_multiply_batch,
    "low_rank_factors": _low_rank_inverse_metric_multiply_batch,
    "kfac_factors": _kfac_inverse_metric_multiply_batch,
    "ekfac_factors": _ekfac_inverse_metric_multiply_batch,
    "ggn_derived_factors": _ggn_derived_inverse_metric_multiply_batch,
}

STREAMING_METRIC_MULTIPLY_BY_KIND = {
    "diagonal_tree": _diagonal_metric_multiply,
    "block_diagonal": _block_diagonal_metric_multiply,
    "low_rank_factors": _streaming_low_rank_metric_multiply,
    "kfac_factors": _streaming_kfac_metric_multiply,
    "ekfac_factors": _ekfac_metric_multiply,
    "ggn_derived_factors": _streaming_ggn_metric_multiply,
}


def run_inverse_metric(execution: runtime_values.StandardExecution) -> TensorTree:
    """Run inverse metric.

    Returns:
        The inverse metric result.
    """
    runtime_values.require_path(
        execution.operator.kind,
        execution.path,
        (
            runtime_values.INVERSE_METRIC_DENSE_PATH,
            runtime_values.INVERSE_METRIC_CG_PATH,
            runtime_values.INVERSE_METRIC_CHOLESKY_PATH,
            runtime_values.INVERSE_METRIC_EIGH_PATH,
            runtime_values.INVERSE_METRIC_SVD_PATH,
            runtime_values.INVERSE_METRIC_FACTORIZED_PATH,
            runtime_values.INVERSE_METRIC_BLOCKWISE_PATH,
            runtime_values.INVERSE_METRIC_WOODBURY_PATH,
        ),
    )

    if execution.compiled_inner is None:
        result = run_inverse_metric_by_mode(execution)
    else:
        result = execution.compiled_inner()

    runtime_values.require_finite_tree(result, "inverse metric result")

    return result


def run_inverse_metric_by_mode(
    execution: runtime_values.StandardExecution,
) -> TensorTree:
    """Run inverse metric by mode.

    Returns:
        The inverse metric by mode result.
    """
    return vectorization.run_by_vectorization_mode(
        execution,
        single_vector=_inverse_metric_solve_by_path,
        single_loop=_run_inverse_metric_vector_single_loop,
        manual_batch=_run_inverse_metric_vector_manual_batch,
        vmap=_reject_inverse_metric_vector_vmap,
    )


def _run_inverse_metric_vector_manual_batch(
    execution: runtime_values.StandardExecution,
) -> TensorTree:
    return vectorization.run_vector_manual_batches(
        execution,
        _run_inverse_metric_vector_single_loop,
    )


def _reject_inverse_metric_vector_vmap(
    execution: runtime_values.StandardExecution,
) -> TensorTree:
    _ = execution
    message = "inverse_metric does not lower vectorization.mode=vmap"
    raise MaterializationError(message)


def _run_inverse_metric_vector_single_loop(
    execution: runtime_values.StandardExecution,
) -> TensorTree:
    settings = execution.candidate.settings

    if (
        settings.get("inverse_metric.multi_rhs") == "block"
        or settings.get("inverse_metric.factor_reuse") == "reuse_factor_across_rhs"
    ):
        return _run_inverse_metric_rhs_batch(execution)

    return vectorization.run_vector_single_loop(
        execution, _inverse_metric_solve_by_path
    )


def _run_inverse_metric_rhs_batch(
    execution: runtime_values.StandardExecution,
) -> TensorTree:
    row = INVERSE_METRIC_BATCH_BY_PATH.get(execution.path)

    if row is None:
        message = "block inverse metric RHS requires a batched solve path"
        raise MaterializationError(message)

    runner, representations = row

    if representations is not None:
        _require_metric_representation(execution.operator, representations)

    return runner(execution)


def _run_inverse_metric_reused_dense_factor_batch(
    execution: runtime_values.StandardExecution,
) -> TensorTree:
    _require_metric_representation(execution.operator, ("dense_matrix",))
    inverse_matrix = inverse_metric_matrix(
        execution.operator,
        metric_dense_matrix(execution.operator, execution.batch, execution.vector),
        execution.batch,
        execution.vector,
    )
    vector_batch = _flat_inverse_metric_vector_batch(execution)
    runtime_values.require_finite_tensor(inverse_matrix, "metric matrix")
    runtime_values.require_finite_tensor(vector_batch, "inverse metric vector batch")
    result = _dense_inverse_metric_solve_batch(
        inverse_matrix,
        vector_batch,
        execution.path,
    )
    runtime_values.require_finite_tensor(result, "inverse metric batched result")

    return runtime_values.wrap_flat_vector_batch(execution.params, result)


def _factorized_inverse_metric_multiply_batch(
    execution: runtime_values.StandardExecution,
) -> TensorTree:
    kind = metric_representation_kind(execution.operator)
    runner = FACTORIZED_INVERSE_METRIC_BATCH_BY_KIND.get(kind)

    if runner is not None:
        return runner(execution)

    message = f"factorized inverse batch path is not lowered for representation: {kind}"
    raise MaterializationError(message)


def _flat_inverse_metric_vector_batch(
    execution: runtime_values.StandardExecution,
) -> torch.Tensor:
    vector_in_dims = vectorization.vector_tree_in_dims(
        execution.vector,
        execution.candidate.settings,
    )

    return runtime_values.flatten_vector_batch(
        execution.params, execution.vector, vector_in_dims
    )


def _inverse_metric_solve_by_path(
    execution: runtime_values.StandardExecution,
) -> TensorTree:
    if execution.path == runtime_values.INVERSE_METRIC_CG_PATH:
        return _conjugate_gradient_inverse_metric_multiply(execution)

    runner_row = INVERSE_METRIC_MULTIPLY_BY_PATH.get(execution.path)

    if runner_row is not None:
        runners, error_prefix = runner_row

        return _metric_multiply_by_kind(
            execution.operator,
            execution.batch,
            execution.vector,
            execution.candidate.settings,
            runners=runners,
            error_prefix=error_prefix,
        )

    _require_metric_representation(execution.operator, ("dense_matrix",))
    inverse_matrix = inverse_metric_matrix(
        execution.operator,
        metric_dense_matrix(execution.operator, execution.batch, execution.vector),
        execution.batch,
        execution.vector,
    )
    vector_tensor = runtime_values.flatten_vector(execution.vector)
    runtime_values.require_finite_tensor(inverse_matrix, "metric matrix")
    runtime_values.require_finite_tensor(vector_tensor, "inverse metric vector")
    result = _dense_inverse_metric_solve(
        inverse_matrix,
        vector_tensor,
        execution.path,
    )
    runtime_values.require_finite_tensor(result, "inverse metric result")

    return runtime_values.wrap_flat_vector(execution.vector, result)


def _conjugate_gradient_inverse_metric_multiply(
    execution: runtime_values.StandardExecution,
) -> TensorTree:
    budget = _inverse_metric_iteration_budget(execution.candidate.settings)
    preconditioner = _inverse_metric_preconditioner(execution.candidate.settings)
    metric_path = _metric_runtime_path_from_settings(execution.candidate.settings)
    _require_metric_accumulation_settings(metric_path, execution.candidate.settings)
    damping = inverse_metric_damping_payload(execution.operator)
    _require_positive_matrix_free_damping(execution.operator, damping)
    tolerance = inverse_metric_tolerance(execution.operator)
    solution = _conjugate_gradient_inverse_metric_batch_solve(
        execution,
        execution.vector,
        runtime_values.flatten_vector(execution.vector).unsqueeze(0),
        budget,
        preconditioner,
        metric_path,
        damping,
        tolerance,
    )
    result = solution[0]

    runtime_values.require_finite_tensor(result, "conjugate gradient result")

    return runtime_values.wrap_flat_vector(execution.vector, result)


def _require_positive_matrix_free_damping(
    operator: OperatorSpec,
    damping: float | Mapping[str, float],
) -> None:
    if metric_representation_kind(operator) != "matrix_free":
        return

    if minimum_inverse_metric_damping(damping) > 0.0:
        return

    message = "matrix_free conjugate_gradient requires positive damping"
    raise MaterializationError(message)


def _conjugate_gradient_inverse_metric_multiply_batch(
    execution: runtime_values.StandardExecution,
) -> TensorTree:
    budget = _inverse_metric_iteration_budget(execution.candidate.settings)
    preconditioner = _inverse_metric_preconditioner(execution.candidate.settings)
    metric_path = _metric_runtime_path_from_settings(execution.candidate.settings)
    _require_metric_accumulation_settings(metric_path, execution.candidate.settings)
    damping = inverse_metric_damping_payload(execution.operator)
    _require_positive_matrix_free_damping(execution.operator, damping)
    tolerance = inverse_metric_tolerance(execution.operator)
    vector_batch = _flat_inverse_metric_vector_batch(execution)
    solution = _conjugate_gradient_inverse_metric_batch_solve(
        execution,
        execution.params,
        vector_batch,
        budget,
        preconditioner,
        metric_path,
        damping,
        tolerance,
    )
    runtime_values.require_finite_tensor(solution, "batched conjugate gradient result")

    return runtime_values.wrap_flat_vector_batch(execution.params, solution)


INVERSE_METRIC_BATCH_BY_PATH = {
    **dict.fromkeys(
        runtime_values.INVERSE_METRIC_DIRECT_SOLVE_PATHS,
        (_run_inverse_metric_reused_dense_factor_batch, None),
    ),
    runtime_values.INVERSE_METRIC_CG_PATH: (
        _conjugate_gradient_inverse_metric_multiply_batch,
        None,
    ),
    runtime_values.INVERSE_METRIC_FACTORIZED_PATH: (
        _factorized_inverse_metric_multiply_batch,
        None,
    ),
    runtime_values.INVERSE_METRIC_BLOCKWISE_PATH: (
        _block_diagonal_inverse_metric_multiply_batch,
        ("block_diagonal",),
    ),
    runtime_values.INVERSE_METRIC_WOODBURY_PATH: (
        _low_rank_inverse_metric_multiply_batch,
        ("low_rank_factors",),
    ),
}

INVERSE_METRIC_FACTOR_REUSE_PATHS = tuple(INVERSE_METRIC_BATCH_BY_PATH)


def _conjugate_gradient_inverse_metric_batch_solve(
    execution: runtime_values.StandardExecution,
    template: TensorTree,
    vectors: torch.Tensor,
    budget: int,
    preconditioner: str,
    metric_path: str,
    damping: float | Mapping[str, float],
    tolerance: float | None,
) -> torch.Tensor:
    solution = torch.zeros_like(vectors)
    residual = vectors - _metric_apply_flat_batch(
        execution.operator,
        execution.batch,
        template,
        solution,
        damping,
        metric_path,
        execution.candidate.settings,
    )

    if runtime_values.batched_cg_residual_satisfies_tolerance(
        residual, vectors, tolerance
    ):
        return solution

    preconditioned = _apply_inverse_metric_preconditioner_batch(
        execution.operator,
        execution.batch,
        template,
        residual,
        preconditioner,
        execution.candidate.settings,
        runtime_values.MATRIX_FREE_RUNTIME_BINDINGS.get(),
    )
    direction = preconditioned
    residual_dot = _batched_dot_runtime(
        execution.candidate.settings,
        residual,
        preconditioned,
    )

    for _ in range(budget):
        matrix_direction = _metric_apply_flat_batch(
            execution.operator,
            execution.batch,
            template,
            direction,
            damping,
            metric_path,
            execution.candidate.settings,
        )
        step = runtime_values.zero_numerator_divide(
            residual_dot,
            _batched_dot_runtime(
                execution.candidate.settings,
                direction,
                matrix_direction,
            ),
        )
        solution = solution + step[:, None] * direction
        residual = residual - step[:, None] * matrix_direction

        if runtime_values.batched_cg_residual_satisfies_tolerance(
            residual, vectors, tolerance
        ):
            break

        preconditioned = _apply_inverse_metric_preconditioner_batch(
            execution.operator,
            execution.batch,
            template,
            residual,
            preconditioner,
            execution.candidate.settings,
            runtime_values.MATRIX_FREE_RUNTIME_BINDINGS.get(),
        )
        next_residual_dot = _batched_dot_runtime(
            execution.candidate.settings,
            residual,
            preconditioned,
        )
        direction = preconditioned + runtime_values.zero_numerator_divide(
            next_residual_dot,
            residual_dot,
        )[:, None] * (direction)
        residual_dot = next_residual_dot

    runtime_values.require_finite_tensor(solution, "batched conjugate gradient result")

    return solution


def _metric_runtime_path_from_settings(settings: Mapping[str, Any]) -> str:
    return runtime_values.runtime_path_from_settings(
        settings,
        "metric",
        "metric.multiply_path",
    )


def _inverse_metric_runtime_path_from_settings(settings: Mapping[str, Any]) -> str:
    return runtime_values.runtime_path_from_settings(
        settings,
        "inverse_metric",
        "inverse_metric.solve_path",
    )


def metric_apply_flat(
    operator: OperatorSpec,
    batch: Batch,
    template: TensorTree,
    flat_vector: torch.Tensor,
    damping: float | Mapping[str, float],
    metric_path: str,
    settings: Mapping[str, Any],
) -> torch.Tensor:
    """Return the metric apply flat.

    Returns:
        The metric apply flat.
    """
    vector = runtime_values.wrap_flat_vector(template, flat_vector)
    result = metric_multiply_by_path(operator, batch, vector, metric_path, settings)
    flat_result = runtime_values.flatten_vector(
        result
    ) + _inverse_metric_damping_product(
        operator,
        flat_vector,
        damping,
    )
    runtime_values.require_finite_tensor(flat_result, "metric apply result")

    return flat_result


def _metric_apply_flat_batch(
    operator: OperatorSpec,
    batch: Batch,
    template: TensorTree,
    flat_batch: torch.Tensor,
    damping: float | Mapping[str, float],
    metric_path: str,
    settings: Mapping[str, Any],
) -> torch.Tensor:
    def runner(flat_vector: torch.Tensor) -> torch.Tensor:
        return metric_apply_flat(
            operator,
            batch,
            template,
            flat_vector,
            damping,
            metric_path,
            settings,
        )

    return runtime_values.map_flat_batch(
        flat_batch, runner, "batched metric apply result"
    )


def _apply_inverse_metric_preconditioner(
    operator: OperatorSpec,
    batch: Batch,
    template: TensorTree,
    residual: torch.Tensor,
    preconditioner: str,
    settings: Mapping[str, Any],
    matrix_free_operators: Mapping[str, Callable[[Batch, TensorTree], TensorTree]]
    | None,
) -> torch.Tensor:
    residual_tree = runtime_values.wrap_flat_vector(template, residual)

    if preconditioner == "none":
        result = residual
    elif preconditioner == "diagonal":
        diagonal = torch.diag(
            inverse_metric_matrix(
                operator,
                metric_dense_matrix(operator, batch, template),
                batch,
                template,
            )
        )
        result = residual / diagonal
    elif preconditioner == "block_diagonal":
        result = runtime_values.flatten_vector(
            _block_or_kfac_preconditioner(operator, batch, residual_tree)
        )
    elif preconditioner == "factorized_metric":
        result = runtime_values.flatten_vector(
            _factorized_metric_preconditioner(operator, batch, residual_tree, settings)
        )
    elif preconditioner == "matrix_free":
        result = runtime_values.flatten_vector(
            _matrix_free_preconditioner(
                batch,
                residual_tree,
                settings,
                matrix_free_operators,
            )
        )
    else:
        message = f"inverse metric preconditioner is unsupported: {preconditioner}"
        raise MaterializationError(message)

    runtime_values.require_finite_tensor(result, "inverse metric preconditioner result")

    return result


def _apply_inverse_metric_preconditioner_batch(
    operator: OperatorSpec,
    batch: Batch,
    template: TensorTree,
    residual_batch: torch.Tensor,
    preconditioner: str,
    settings: Mapping[str, Any],
    matrix_free_operators: Mapping[str, Callable[[Batch, TensorTree], TensorTree]]
    | None,
) -> torch.Tensor:
    def runner(residual: torch.Tensor) -> torch.Tensor:
        return _apply_inverse_metric_preconditioner(
            operator,
            batch,
            template,
            residual,
            preconditioner,
            settings,
            matrix_free_operators,
        )

    return runtime_values.map_flat_batch(
        residual_batch,
        runner,
        "batched inverse metric preconditioner result",
    )


def _block_or_kfac_preconditioner(
    operator: OperatorSpec,
    batch: Batch,
    vector: TensorTree,
) -> TensorTree:
    return _metric_multiply_by_kind(
        operator,
        batch,
        vector,
        {},
        runners=BLOCK_PRECONDITIONER_BY_KIND,
        error_prefix="block preconditioner is not lowered for representation",
    )


def _factorized_metric_preconditioner(
    operator: OperatorSpec,
    batch: Batch,
    vector: TensorTree,
    settings: Mapping[str, Any],
) -> TensorTree:
    return _metric_multiply_by_kind(
        operator,
        batch,
        vector,
        settings,
        runners=FACTORIZED_PRECONDITIONER_BY_KIND,
        error_prefix="factorized preconditioner is not lowered for representation",
    )


def _matrix_free_preconditioner(
    batch: Batch,
    vector: TensorTree,
    settings: Mapping[str, Any],
    matrix_free_operators: Mapping[str, Callable[[Batch, TensorTree], TensorTree]]
    | None,
) -> TensorTree:
    product = _inverse_metric_preconditioner_product(settings)
    bindings = matrix_free_operators

    if bindings is None:
        bindings = runtime_values.MATRIX_FREE_RUNTIME_BINDINGS.get()

    operation = None if bindings is None else bindings.get(product)

    if operation is None:
        message = (
            f"matrix_free preconditioner requires selected sibling product: {product}"
        )
        raise MaterializationError(message)

    return operation(batch, vector)


def _inverse_metric_iteration_budget(settings: Mapping[str, Any]) -> int:
    return runtime_values.required_positive_int_setting(
        settings,
        "inverse_metric.iteration_budget",
        "inverse_metric.iteration_budget must be a positive integer",
    )


def _inverse_metric_preconditioner(settings: Mapping[str, Any]) -> str:
    value = settings.get("inverse_metric.preconditioner")

    if not isinstance(value, str):
        message = "inverse_metric.preconditioner is required"
        raise MaterializationError(message)

    if value == "matrix_free":
        _inverse_metric_preconditioner_product(settings)

        return value

    if "inverse_metric.preconditioner_product" in settings:
        message = (
            "inverse_metric.preconditioner_product applies only to matrix_free "
            "preconditioner"
        )
        raise MaterializationError(message)

    if value not in {"none", "diagonal", "block_diagonal", "factorized_metric"}:
        message = f"inverse_metric.preconditioner is unsupported: {value}"
        raise MaterializationError(message)

    return value


def _inverse_metric_preconditioner_product(settings: Mapping[str, Any]) -> str:
    value = settings.get("inverse_metric.preconditioner_product")

    if isinstance(value, str) and value:
        return value

    message = (
        "inverse_metric.preconditioner=matrix_free requires "
        "inverse_metric.preconditioner_product"
    )
    raise MaterializationError(message)


def _dense_inverse_metric_solve(
    matrix: torch.Tensor,
    vector: torch.Tensor,
    path: str,
) -> torch.Tensor:
    flat_vector = vector.reshape(-1)
    result = _dense_inverse_metric_solve_batch(
        matrix,
        flat_vector.unsqueeze(0),
        path,
    )[0]

    return result.reshape_as(vector)


def _dense_inverse_metric_solve_batch(
    matrix: torch.Tensor,
    vector_batch: torch.Tensor,
    path: str,
) -> torch.Tensor:
    if vector_batch.ndim != runtime_values.MATRIX_DIMS:
        message = "batched inverse metric vectors must flatten to a matrix"
        raise MaterializationError(message)

    rhs = vector_batch.T

    if path == runtime_values.INVERSE_METRIC_DENSE_PATH:
        return torch.linalg.solve(matrix, rhs).T

    if path == runtime_values.INVERSE_METRIC_CHOLESKY_PATH:
        factor = torch.linalg.cholesky(matrix)

        return torch.cholesky_solve(rhs, factor).T

    if path == runtime_values.INVERSE_METRIC_EIGH_PATH:
        eigenvalues, eigenvectors = torch.linalg.eigh(matrix)
        coefficients = eigenvectors.T @ rhs

        return (eigenvectors @ (coefficients / eigenvalues[:, None])).T

    if path == runtime_values.INVERSE_METRIC_SVD_PATH:
        left, singular_values, right_h = torch.linalg.svd(matrix, full_matrices=False)
        coefficients = left.T @ rhs

        return (right_h.T @ (coefficients / singular_values[:, None])).T

    message = f"batched inverse metric solve path is not lowered: {path}"
    raise MaterializationError(message)


@dataclasses.dataclass(frozen=True, slots=True)
class StandardMetricOperator:
    """Materialized metric selected by the standard runtime."""

    candidate: Candidate
    record: FullSizeRecord
    operator: OperatorSpec
    representation: Mapping[str, Any]
    default_operation: str = "multiply"
    damping: float | Mapping[str, float] = 0.0
    inverse_path: str | None = None
    mmap_residency: Callable[[torch.Tensor, str], torch.Tensor] | None = None
    matrix_free_operators: Mapping[str, Callable[[Batch, TensorTree], TensorTree]] = (
        dataclasses.field(default_factory=dict)
    )

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
        return self._run_metric_runtime(batch, vector, self._multiply_runtime)

    def inverse_multiply(self, batch: Batch, vector: TensorTree) -> TensorTree:
        """Return inverse metric-vector product."""
        return self._run_metric_runtime(batch, vector, self._inverse_runtime)

    def _run_metric_runtime(
        self,
        batch: Batch,
        vector: TensorTree,
        operation: Callable[[Batch, TensorTree], TensorTree],
    ) -> TensorTree:
        return runtime.run_with_backend_settings(
            self.candidate.settings,
            lambda: operation(*self._runtime_inputs(batch, vector)),
        )

    def _multiply_runtime(self, batch: Batch, vector: TensorTree) -> TensorTree:
        metric_operator = self._operator_spec("metric")
        path = runtime.runtime_path(metric_operator, self.candidate)
        result = metric_multiply_by_path(
            metric_operator,
            batch,
            vector,
            path,
            self.candidate.settings,
            self.matrix_free_operators,
        )
        runtime_values.require_finite_tree(result, "metric result")

        return result

    def _inverse_runtime(self, batch: Batch, vector: TensorTree) -> TensorTree:
        if self.inverse_path is None:
            message = "inverse_multiply requires an inverse_metric selection"
            raise MaterializationError(message)

        inverse_operator = self._operator_for_inverse()

        if self.inverse_path == runtime_values.INVERSE_METRIC_CG_PATH:
            execution = runtime_values.StandardExecution(
                inverse_operator,
                self.candidate,
                self.inverse_path,
                batch,
                vector,
                {},
                {},
                None,
                ObjectiveContext(
                    family=self.record.family,
                    candidate_id=self.candidate.candidate_id,
                    settings=dict(self.candidate.settings),
                ),
                {},
                {},
            )
            result = runtime_values.run_with_matrix_free_runtime_bindings(
                self.matrix_free_operators,
                lambda: _conjugate_gradient_inverse_metric_multiply(execution),
            )
        elif self.inverse_path in INVERSE_METRIC_MULTIPLY_BY_PATH:
            runners, error_prefix = INVERSE_METRIC_MULTIPLY_BY_PATH[self.inverse_path]
            result = _metric_multiply_by_kind(
                inverse_operator,
                batch,
                vector,
                self.candidate.settings,
                runners=runners,
                error_prefix=error_prefix,
            )
        else:
            _require_metric_representation(inverse_operator, ("dense_matrix",))
            matrix = self._dense_matrix(batch, vector)
            inverse_matrix = inverse_metric_matrix(
                inverse_operator,
                matrix,
                batch,
                vector,
            )
            flat_result = _dense_inverse_metric_solve(
                inverse_matrix,
                runtime_values.flatten_vector(vector),
                self.inverse_path,
            )
            runtime_values.require_finite_tensor(flat_result, "inverse metric result")
            result = runtime_values.wrap_flat_vector(vector, flat_result)

        runtime_values.require_finite_tree(result, "inverse metric result")

        return result

    def inner(
        self,
        batch: Batch,
        left: TensorTree,
        right: TensorTree,
    ) -> torch.Tensor:
        """Return metric inner product."""

        def callback() -> torch.Tensor:
            runtime_batch, runtime_left = self._runtime_inputs(batch, left)
            runtime_right = runtime.runtime_vector(right, self.candidate.settings)
            left_tensor = runtime_values.flatten_vector(runtime_left)
            runtime_values.require_finite_tensor(
                left_tensor, "metric inner left vector"
            )
            metric_right = self.multiply(runtime_batch, runtime_right)
            result = runtime.tree_dot_runtime(
                self.candidate.settings,
                runtime_left,
                metric_right,
            )
            runtime_values.require_finite_tensor(result, "metric inner result")

            return result

        return runtime.run_with_backend_settings(self.candidate.settings, callback)

    def _runtime_inputs(
        self,
        batch: Batch,
        vector: TensorTree,
    ) -> tuple[Batch, TensorTree]:
        runtime_batch = runtime.runtime_batch(
            batch,
            self.candidate.settings,
            mmap_residency=self.mmap_residency,
        )
        runtime_vector = runtime.runtime_vector(
            vector,
            self.candidate.settings,
            mmap_residency=self.mmap_residency,
        )
        vector_tensor = runtime_values.flatten_vector(runtime_vector)
        runtime_values.require_finite_tensor(vector_tensor, "metric vector")

        return runtime_batch, runtime_vector

    def _dense_matrix(self, batch: Batch, vector: TensorTree) -> torch.Tensor:
        operator = self._operator_spec("metric")

        return metric_dense_matrix(operator, batch, vector)

    def _operator_for_inverse(self) -> OperatorSpec:
        damping_kind = _inverse_metric_damping_kind(self.operator)
        damping_value = inverse_metric_damping_payload(self.operator)
        semantics = {
            "damping": damping_value,
            "damping_kind": damping_kind,
            "damping_value": damping_value,
        }
        damping_groups = self.operator.semantics.get("damping_groups")

        if damping_groups is not None:
            semantics["damping_groups"] = damping_groups

        return self._operator_spec("inverse_metric", semantics)

    def _operator_spec(
        self,
        kind: str,
        semantics: Mapping[str, Any] | None = None,
    ) -> OperatorSpec:
        full_semantics = {"representation": dict(self.representation)}

        if semantics is not None:
            full_semantics.update(semantics)

        return dataclasses.replace(
            self.operator,
            kind=kind,
            semantics=full_semantics,
        )


@dataclasses.dataclass(frozen=True, slots=True)
class KFACMetricOperator:
    """Metric operations backed by Kronecker-factored blocks."""

    blocks: tuple[runtime_values.KFACMetricBlock, ...]
    damping: float | Mapping[str, float] = 0.0
    damping_kind: str = "scalar"
    damping_policy: str | None = None
    settings: Mapping[str, Any] = dataclasses.field(default_factory=dict)

    def __call__(self, batch: Batch, vector: TensorTree) -> TensorTree:
        """Return metric-vector product."""
        return self.multiply(batch, vector)

    def multiply(self, batch: Batch, vector: TensorTree) -> TensorTree:
        """Return KFAC metric-vector product."""

        def product(
            _: runtime_values.KFACMetricBlock,
            left: torch.Tensor,
            right: torch.Tensor,
            value: torch.Tensor,
        ) -> torch.Tensor:
            return layout.matmul_runtime(
                self.settings,
                layout.matmul_runtime(self.settings, left, value),
                right.T,
            )

        return _kfac_block_results(
            self.blocks,
            batch,
            vector,
            product,
            "KFAC metric result",
        )

    def inverse_multiply(self, batch: Batch, vector: TensorTree) -> TensorTree:
        """Return inverse KFAC metric-vector product."""

        def product(
            block: runtime_values.KFACMetricBlock,
            left: torch.Tensor,
            right: torch.Tensor,
            value: torch.Tensor,
        ) -> torch.Tensor:
            damping = _resolved_group_damping(
                self.damping,
                self.damping_kind,
                block.parameter_name,
            )

            return _kfac_inverse_product(
                left,
                right,
                value,
                damping,
                _resolved_group_damping_kind(self.damping_kind),
                self.damping_policy,
            )

        return _kfac_block_results(
            self.blocks,
            batch,
            vector,
            product,
            "inverse KFAC metric result",
        )

    def inner(
        self,
        batch: Batch,
        left: TensorTree,
        right: TensorTree,
    ) -> torch.Tensor:
        """Return KFAC metric inner product."""
        return runtime.tree_dot_runtime(
            self.settings, left, self.multiply(batch, right)
        )


def _kfac_block_results(
    blocks: Sequence[runtime_values.KFACMetricBlock],
    batch: Batch,
    vector: TensorTree,
    product: Callable[
        [runtime_values.KFACMetricBlock, torch.Tensor, torch.Tensor, torch.Tensor],
        torch.Tensor,
    ],
    label_prefix: str,
) -> dict[str, torch.Tensor]:
    vector_map = _kfac_vector_map(vector)
    result = {}

    for block in blocks:
        left = _kfac_factor(batch, block.left_factor_key)
        right = _kfac_factor(batch, block.right_factor_key)
        value = _kfac_vector_leaf(vector_map, block)
        _require_kfac_shapes(block, left, right, value)
        block_result = product(block, left, right, value)
        runtime_values.require_finite_tensor(
            block_result, f"{label_prefix} {block.parameter_name}"
        )
        result[block.parameter_name] = block_result

    return result


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

    runtime_values.require_finite_tensor(value, f"KFAC factor {key}")

    if value.ndim != runtime_values.MATRIX_DIMS or value.shape[0] != value.shape[1]:
        message = f"KFAC factor must be square: {key}"
        raise MaterializationError(message)

    return value


def _kfac_inverse_product(
    left: torch.Tensor,
    right: torch.Tensor,
    value: torch.Tensor,
    damping: float,
    damping_kind: str,
    damping_policy: str | None,
) -> torch.Tensor:
    return _kfac_inverse_product_with_damping(
        left,
        right,
        value,
        damping,
        damping_kind,
        damping_policy,
    )


def _kfac_inverse_product_batch(
    operator: OperatorSpec,
    parameter_name: str,
    left: torch.Tensor,
    right: torch.Tensor,
    value: torch.Tensor,
) -> torch.Tensor:
    damping_kind = _inverse_metric_damping_kind(operator)
    damping = _resolved_group_damping(
        inverse_metric_damping_payload(operator),
        damping_kind,
        parameter_name,
    )
    damping_kind = _resolved_group_damping_kind(damping_kind)
    damping_policy = _inverse_metric_damping_policy(operator)

    return _kfac_inverse_product_with_damping(
        left,
        right,
        value,
        damping,
        damping_kind,
        damping_policy,
    )


def _kfac_inverse_product_with_damping(
    left: torch.Tensor,
    right: torch.Tensor,
    value: torch.Tensor,
    damping: float,
    damping_kind: str,
    damping_policy: str | None,
) -> torch.Tensor:
    if damping_kind == "kfac_pi":
        left_shift, right_shift = _kfac_pi_shifts(
            left,
            right,
            damping,
            damping_policy,
        )

        return _kfac_factored_inverse_product(
            left
            + left_shift
            * torch.eye(
                left.shape[0],
                dtype=left.dtype,
                device=left.device,
            ),
            right
            + right_shift
            * torch.eye(
                right.shape[0],
                dtype=right.dtype,
                device=right.device,
            ),
            value,
        )

    if damping_kind != "scalar":
        message = f"KFAC inverse does not lower damping kind: {damping_kind}"
        raise MaterializationError(message)

    if damping <= 0.0:
        return _kfac_factored_inverse_product(left, right, value)

    left_eigenvalues, left_eigenvectors = torch.linalg.eigh(left)
    right_eigenvalues, right_eigenvectors = torch.linalg.eigh(right)
    rotated = left_eigenvectors.T @ value @ right_eigenvectors
    denominator = left_eigenvalues[:, None] * right_eigenvalues[None, :] + damping
    solved = rotated / denominator

    return left_eigenvectors @ solved @ right_eigenvectors.T


def _kfac_factored_inverse_product(
    left: torch.Tensor,
    right: torch.Tensor,
    value: torch.Tensor,
) -> torch.Tensor:
    left_solved = torch.linalg.solve(left, value)

    return torch.linalg.solve(right, left_solved.mT).mT


def _kfac_pi_shifts(
    left: torch.Tensor,
    right: torch.Tensor,
    damping: float,
    policy: str | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    root = torch.sqrt(left.new_tensor(damping))

    if policy == "equal":
        return root, root

    if policy != "trace_norm":
        message = f"kfac_pi policy is unsupported: {policy}"
        raise MaterializationError(message)

    left_mean = torch.trace(left) / left.shape[0]
    right_mean = torch.trace(right) / right.shape[0]
    _require_positive_spectrum(
        torch.stack((left_mean, right_mean)),
        "KFAC pi factor traces",
    )
    ratio = torch.sqrt(left_mean / right_mean)

    return ratio * root, root / ratio


def _kfac_batched_vector_leaf(
    vector: Mapping[str, TensorTree],
    in_dims: Mapping[str, Any],
    block: runtime_values.KFACMetricBlock,
    left: torch.Tensor,
    right: torch.Tensor,
    vector_count: int,
) -> torch.Tensor:
    value = vector.get(block.parameter_name)
    in_dim = in_dims.get(block.parameter_name)

    if not isinstance(value, torch.Tensor):
        message = f"KFAC vector leaf is missing or not a tensor: {block.parameter_name}"
        raise MaterializationError(message)

    runtime_values.require_finite_tensor(value, f"KFAC vector {block.parameter_name}")

    if in_dim is None:
        _require_kfac_shapes(block, left, right, value)

        return value.expand(vector_count, *value.shape)

    if not isinstance(in_dim, int) or isinstance(in_dim, bool):
        message = "KFAC vectorization.in_dims values must be integers or None"
        raise MaterializationError(message)

    dim = runtime_values.normalized_vector_dim(value, in_dim)
    expected = (left.shape[0], right.shape[0])
    unbatched_shape = value.shape[:dim] + value.shape[dim + 1 :]

    if tuple(unbatched_shape) != expected:
        message = (
            f"KFAC vector leaf shape mismatch for {block.parameter_name}: "
            f"{tuple(unbatched_shape)} != {expected}"
        )
        raise MaterializationError(message)

    if value.shape[dim] != vector_count:
        message = "KFAC vectorized dimensions differ"
        raise MaterializationError(message)

    return value.movedim(dim, 0)


def _kfac_vector_leaf(
    vector: Mapping[str, TensorTree],
    block: runtime_values.KFACMetricBlock,
) -> torch.Tensor:
    value = vector.get(block.parameter_name)

    if not isinstance(value, torch.Tensor):
        message = f"KFAC vector leaf is missing or not a tensor: {block.parameter_name}"
        raise MaterializationError(message)

    runtime_values.require_finite_tensor(value, f"KFAC vector {block.parameter_name}")

    if value.ndim != runtime_values.MATRIX_DIMS:
        message = f"KFAC vector leaf must be a matrix: {block.parameter_name}"
        raise MaterializationError(message)

    return value


def _require_kfac_shapes(
    block: runtime_values.KFACMetricBlock,
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


def require_metric_runtime_settings(
    operator: OperatorSpec,
    path: str,
    settings: Mapping[str, Any],
) -> None:
    """Validate metric runtime settings."""
    if operator.kind in {"sqrt_metric", "inverse_sqrt_metric"}:
        _require_sqrt_metric_operator_settings(operator, path, settings)
        return

    if operator.kind == "metric":
        _require_metric_block_schedule(operator, settings, "metric.block_schedule")
        _require_metric_accumulation_settings(path, settings)
        return

    if operator.kind == "metric_inner":
        _require_metric_inner_runtime_settings(operator, path, settings)
        return

    if operator.kind == "inverse_metric":
        _require_inverse_metric_operator_settings(operator, path, settings)
        return

    if operator.kind == "inverse_metric_inner":
        _require_inverse_metric_inner_runtime_settings(operator, path, settings)
        return

    _reject_stray_metric_runtime_settings(settings)


def _require_sqrt_metric_operator_settings(
    operator: OperatorSpec,
    path: str | None,
    settings: Mapping[str, Any],
) -> None:
    _require_sqrt_metric_runtime_settings(path, settings)
    _require_inverse_sqrt_tolerance_path(operator, path)

    if path == runtime_values.SQRT_METRIC_LANCZOS_PATH:
        _require_metric_representation(operator, ("matrix_free",))
        metric_path = _metric_runtime_path_from_settings(settings)
        _require_metric_accumulation_settings(metric_path, settings)

        if "metric.block_schedule" in settings:
            message = "metric.block_schedule does not apply to Lanczos sqrt rows"
            raise MaterializationError(message)

        if "inverse_metric.block_schedule" in settings:
            message = (
                "inverse_metric.block_schedule does not apply to Lanczos sqrt rows"
            )
            raise MaterializationError(message)

        if _has_inverse_metric_settings(settings):
            message = "inverse metric settings do not apply to sqrt rows"
            raise MaterializationError(message)

        return

    if _has_metric_runtime_settings(settings) or _has_inverse_metric_settings(settings):
        message = "metric solve and multiply settings do not apply to sqrt rows"
        raise MaterializationError(message)


def _require_inverse_sqrt_tolerance_path(
    operator: OperatorSpec,
    path: str | None,
) -> None:
    if operator.kind != "inverse_sqrt_metric":
        return

    if inverse_metric_tolerance(operator) is None:
        return

    if path == runtime_values.SQRT_METRIC_LANCZOS_PATH:
        return

    message = "inverse_sqrt_metric tol requires matrix_free_lanczos"
    raise MaterializationError(message)


def _require_metric_inner_runtime_settings(
    operator: OperatorSpec,
    path: str,
    settings: Mapping[str, Any],
) -> None:
    if path == runtime_values.METRIC_INNER_MULTIPLY_REDUCE_PATH:
        metric_path = _metric_runtime_path_from_settings(settings)
        _require_metric_block_schedule(operator, settings, "metric.block_schedule")
        _require_metric_accumulation_settings(metric_path, settings)
        return

    if path == runtime_values.METRIC_INNER_SQRT_REDUCE_PATH:
        sqrt_path = _sqrt_metric_runtime_path_from_settings(settings)
        _require_sqrt_metric_runtime_settings(sqrt_path, settings)

        if _has_metric_multiply_settings(settings):
            message = "metric multiply settings apply only to multiply_then_reduce"
            raise MaterializationError(message)

        return

    if _has_metric_multiply_settings(settings) or _has_sqrt_metric_settings(settings):
        message = "metric multiply settings apply only to multiply_then_reduce"
        raise MaterializationError(message)


def _require_inverse_metric_operator_settings(
    operator: OperatorSpec,
    path: str,
    settings: Mapping[str, Any],
) -> None:
    _require_inverse_metric_tolerance_path(operator, path)
    _require_inverse_metric_iteration_budget_settings(path, settings)
    _require_metric_block_schedule(
        operator,
        settings,
        "inverse_metric.block_schedule",
    )

    if path == runtime_values.INVERSE_METRIC_CG_PATH:
        metric_path = _metric_runtime_path_from_settings(settings)
        _require_metric_accumulation_settings(metric_path, settings)
        return

    if _has_metric_multiply_settings(settings):
        message = (
            "metric settings apply only to metric rows and conjugate_gradient "
            "inverse_metric rows"
        )
        raise MaterializationError(message)


def _require_inverse_metric_inner_runtime_settings(
    operator: OperatorSpec,
    path: str,
    settings: Mapping[str, Any],
) -> None:
    _require_inverse_metric_inner_tolerance_reduction(operator, path)

    if path == runtime_values.INVERSE_METRIC_INNER_SOLVE_REDUCE_PATH:
        _require_inverse_metric_inner_solve_settings(operator, settings)
        return

    if path == runtime_values.INVERSE_METRIC_INNER_SQRT_REDUCE_PATH:
        sqrt_path = _sqrt_metric_runtime_path_from_settings(settings)
        _require_sqrt_metric_runtime_settings(sqrt_path, settings)

        if _has_inverse_metric_settings(settings):
            message = "inverse metric solve settings apply only to solve_then_reduce"
            raise MaterializationError(message)

        return

    if _has_inverse_metric_settings(settings):
        message = "inverse metric solve settings apply only to solve_then_reduce"
        raise MaterializationError(message)

    if _has_sqrt_metric_settings(settings):
        message = "sqrt metric settings apply only to sqrt_apply_reduce"
        raise MaterializationError(message)


def _require_inverse_metric_inner_solve_settings(
    operator: OperatorSpec,
    settings: Mapping[str, Any],
) -> None:
    inverse_path = _inverse_metric_runtime_path_from_settings(settings)
    _require_inverse_metric_inner_tolerance_path(operator, inverse_path)
    _require_inverse_metric_iteration_budget_settings(inverse_path, settings)
    _require_metric_block_schedule(
        operator,
        settings,
        "inverse_metric.block_schedule",
    )

    if inverse_path == runtime_values.INVERSE_METRIC_CG_PATH:
        metric_path = _metric_runtime_path_from_settings(settings)
        _require_metric_accumulation_settings(metric_path, settings)
    elif _has_metric_multiply_settings(settings):
        message = (
            "metric multiply settings apply only to conjugate_gradient "
            "solve_then_reduce"
        )
        raise MaterializationError(message)

    if _has_sqrt_metric_settings(settings):
        message = "sqrt metric settings apply only to sqrt_apply_reduce"
        raise MaterializationError(message)


def _require_inverse_metric_tolerance_path(
    operator: OperatorSpec,
    path: str,
) -> None:
    if inverse_metric_tolerance(operator) is None:
        return

    if path == runtime_values.INVERSE_METRIC_CG_PATH:
        return

    message = "inverse metric tol requires conjugate_gradient"
    raise MaterializationError(message)


def _require_inverse_metric_inner_tolerance_reduction(
    operator: OperatorSpec,
    path: str,
) -> None:
    if inverse_metric_tolerance(operator) is None:
        return

    if path == runtime_values.INVERSE_METRIC_INNER_SOLVE_REDUCE_PATH:
        return

    message = "inverse_metric_inner tol requires solve_then_reduce"
    raise MaterializationError(message)


def _require_inverse_metric_inner_tolerance_path(
    operator: OperatorSpec,
    path: str,
) -> None:
    if inverse_metric_tolerance(operator) is None:
        return

    if path == runtime_values.INVERSE_METRIC_CG_PATH:
        return

    message = "inverse_metric_inner tol requires conjugate_gradient"
    raise MaterializationError(message)


def _reject_stray_metric_runtime_settings(settings: Mapping[str, Any]) -> None:
    if not (
        _has_metric_runtime_settings(settings)
        or _has_inverse_metric_settings(settings)
        or _has_sqrt_metric_settings(settings)
    ):
        return

    message = (
        "metric settings apply only to metric rows and conjugate_gradient "
        "inverse_metric rows"
    )
    raise MaterializationError(message)


def _has_metric_multiply_settings(settings: Mapping[str, Any]) -> bool:
    return "metric.multiply_path" in settings or "metric.accumulation" in settings


def _has_metric_runtime_settings(settings: Mapping[str, Any]) -> bool:
    return (
        _has_metric_multiply_settings(settings)
        or "metric.block_schedule" in settings
        or "inverse_metric.block_schedule" in settings
    )


def _has_inverse_metric_settings(settings: Mapping[str, Any]) -> bool:
    return (
        "inverse_metric.solve_path" in settings
        or "inverse_metric.iteration_budget" in settings
        or "inverse_metric.preconditioner" in settings
        or "inverse_metric.preconditioner_product" in settings
        or "inverse_metric.block_schedule" in settings
    )


def _has_sqrt_metric_settings(settings: Mapping[str, Any]) -> bool:
    return (
        "sqrt_metric.factor_path" in settings
        or "sqrt_metric.lanczos_iterations" in settings
    )


def _require_sqrt_metric_runtime_settings(
    path: str | None,
    settings: Mapping[str, Any],
) -> None:
    if path == runtime_values.SQRT_METRIC_LANCZOS_PATH:
        _sqrt_metric_lanczos_iterations(settings)

        return

    if "sqrt_metric.lanczos_iterations" in settings:
        message = "sqrt_metric.lanczos_iterations requires matrix_free_lanczos"
        raise MaterializationError(message)

    _sqrt_metric_runtime_path_from_settings(settings)


def require_inverse_metric_factor_reuse_settings(
    operator: OperatorSpec,
    path: str | None,
    settings: Mapping[str, Any],
) -> None:
    """Validate inverse metric factor reuse settings.

    Raises:
        MaterializationError: If the declared inputs are invalid.
    """
    value = settings.get("inverse_metric.factor_reuse")

    if value is None:
        return

    if operator.kind != "inverse_metric":
        message = "inverse_metric.factor_reuse applies only to inverse_metric rows"
        raise MaterializationError(message)

    if value == "refactor_each_rhs":
        return

    if value == "reuse_factor_across_rhs":
        if settings.get("vectorization.mode") not in {"single_loop", "manual_batch"}:
            message = "reuse_factor_across_rhs requires vectorized inverse metric input"
            raise MaterializationError(message)

        if path not in INVERSE_METRIC_FACTOR_REUSE_PATHS:
            message = "reuse_factor_across_rhs requires a factor-reuse solve path"
            raise MaterializationError(message)

        return

    message = f"inverse_metric.factor_reuse is unsupported: {value}"
    raise MaterializationError(message)


def require_inverse_metric_multi_rhs_settings(
    operator: OperatorSpec,
    path: str | None,
    settings: Mapping[str, Any],
) -> None:
    """Validate inverse metric multi rhs settings.

    Raises:
        MaterializationError: If the declared inputs are invalid.
    """
    value = settings.get("inverse_metric.multi_rhs")
    mode = settings.get("vectorization.mode")

    if value is None:
        if operator.kind == "inverse_metric" and mode in {
            "single_loop",
            "manual_batch",
        }:
            message = "vectorized inverse_metric rows require inverse_metric.multi_rhs"
            raise MaterializationError(message)

        return

    if operator.kind != "inverse_metric":
        message = "inverse_metric.multi_rhs applies only to inverse_metric rows"
        raise MaterializationError(message)

    if value == "single_column":
        return

    if value != "block":
        message = f"inverse_metric.multi_rhs is unsupported: {value}"
        raise MaterializationError(message)

    if mode not in {"single_loop", "manual_batch"}:
        message = "inverse_metric.multi_rhs=block requires stacked vectors"
        raise MaterializationError(message)

    if path not in INVERSE_METRIC_FACTOR_REUSE_PATHS:
        message = "inverse_metric.multi_rhs=block requires a batched solve path"
        raise MaterializationError(message)


def _require_inverse_metric_iteration_budget_settings(
    path: str | None,
    settings: Mapping[str, Any],
) -> None:
    if "inverse_metric.iteration_budget" not in settings:
        return

    if path != runtime_values.INVERSE_METRIC_CG_PATH:
        message = "inverse_metric.iteration_budget applies only to conjugate_gradient"
        raise MaterializationError(message)

    _inverse_metric_iteration_budget(settings)


def _require_metric_block_schedule(
    operator: OperatorSpec,
    settings: Mapping[str, Any],
    axis_key: str,
) -> None:
    value = settings.get(axis_key)

    if value is None:
        return

    representation = metric_representation(operator)
    kind = metric_representation_kind(operator)

    if kind not in {"block_diagonal", "kfac_factors"}:
        message = f"{axis_key} requires blocks or KFAC factors"
        raise MaterializationError(message)

    schedule = representation.get("block_schedule")

    if not isinstance(schedule, str):
        message = f"{axis_key} requires representation.block_schedule"
        raise MaterializationError(message)

    if value != schedule:
        message = f"{axis_key} must match representation.block_schedule"
        raise MaterializationError(message)


def _batched_dot_runtime(
    settings: Mapping[str, Any],
    left: torch.Tensor,
    right: torch.Tensor,
) -> torch.Tensor:
    left = runtime.runtime_intermediate_tensor(left, settings)
    right = runtime.runtime_intermediate_tensor(right, settings)
    left_accumulation = runtime.accumulation_tensor(left, settings)
    right_accumulation = runtime.accumulation_tensor(right, settings)

    return torch.sum(left_accumulation * right_accumulation, dim=1)


def _tree_elementwise_mul_runtime(
    settings: Mapping[str, Any],
    left: TensorTree,
    right: TensorTree,
) -> TensorTree:
    left = runtime.runtime_intermediate_tree(left, settings)
    right = runtime.runtime_intermediate_tree(right, settings)

    if layout.layout_vector_ops(settings) == "foreach":
        return tree_elementwise_mul_foreach(left, right)

    return tree_map2(torch.mul, left, right)


def _tree_elementwise_div_runtime(
    settings: Mapping[str, Any],
    left: TensorTree,
    right: TensorTree,
) -> TensorTree:
    left = runtime.runtime_intermediate_tree(left, settings)
    right = runtime.runtime_intermediate_tree(right, settings)

    if layout.layout_vector_ops(settings) == "foreach":
        return tree_elementwise_div_foreach(left, right)

    return tree_map2(torch.div, left, right)


def _tree_add_scalar_runtime(
    settings: Mapping[str, Any],
    tree: TensorTree,
    scalar: float,
) -> TensorTree:
    tree = runtime.runtime_intermediate_tree(tree, settings)

    if layout.layout_vector_ops(settings) == "foreach":
        return tree_add_scalar_foreach(tree, scalar)

    return tree_map(lambda tensor: tensor + scalar, tree)


def runtime_metric_factor_value(
    key: str,
    value: Any,
    dtype: torch.dtype,
) -> Any:
    """Return the runtime metric factor value.

    Returns:
        The runtime metric factor value.
    """
    if key not in runtime_values.METRIC_FACTOR_BATCH_KEYS:
        return value

    return runtime_values.runtime_batch_value(value, dtype)


def runtime_metric_factor_residency_value(
    key: str,
    value: Any,
    residency: Any,
    mmap_residency: Callable[[torch.Tensor, str], torch.Tensor] | None,
) -> Any:
    """Return the runtime metric factor residency value.

    Returns:
        The runtime metric factor residency value.
    """
    if key not in runtime_values.METRIC_FACTOR_BATCH_KEYS:
        return value

    return runtime_values.runtime_nested_tensor_value(
        value,
        lambda tensor: memory.runtime_residency_tensor(
            tensor,
            residency,
            "memory.factor_residency",
            mmap_residency,
        ),
    )


def metric_inner_anchor_settings(
    operator: OperatorSpec,
    candidate: Candidate,
    path: str,
) -> dict[str, Any]:
    """Return the metric inner anchor settings.

    Returns:
        The metric inner anchor settings.
    """
    if operator.kind != "metric_inner":
        return {}

    if path == runtime_values.METRIC_INNER_MULTIPLY_REDUCE_PATH:
        settings = _metric_multiply_anchor_settings(operator, candidate)
    elif path == runtime_values.METRIC_INNER_SQRT_REDUCE_PATH:
        settings = _sqrt_metric_anchor_settings(operator, candidate)
    else:
        settings = {}

    settings["metric_inner.multi_rhs"] = _inner_multi_rhs_anchor_value(
        candidate,
        "metric_inner.multi_rhs",
    )

    return settings


def inverse_metric_inner_anchor_settings(
    operator: OperatorSpec,
    candidate: Candidate,
    path: str,
) -> dict[str, Any]:
    """Return the inverse metric inner anchor settings.

    Returns:
        The inverse metric inner anchor settings.
    """
    if operator.kind != "inverse_metric_inner":
        return {}

    if path == runtime_values.INVERSE_METRIC_INNER_SOLVE_REDUCE_PATH:
        settings = _inverse_metric_anchor_settings(operator, candidate)
    elif path == runtime_values.INVERSE_METRIC_INNER_SQRT_REDUCE_PATH:
        settings = _sqrt_metric_anchor_settings(operator, candidate)
    else:
        settings = {}

    settings["inverse_metric_inner.multi_rhs"] = _inner_multi_rhs_anchor_value(
        candidate,
        "inverse_metric_inner.multi_rhs",
    )

    return settings


def _inner_multi_rhs_anchor_value(candidate: Candidate, key: str) -> str:
    value = candidate.settings.get(key)

    if isinstance(value, str):
        return value

    return "single_column"


def _metric_multiply_anchor_settings(
    operator: OperatorSpec,
    candidate: Candidate,
) -> dict[str, Any]:
    value = candidate.settings.get("metric.multiply_path")

    if not isinstance(value, str):
        value = _metric_multiply_anchor_value(operator)

    settings = {"metric.multiply_path": value}

    if value == "dense_matmul":
        return settings

    accumulation = candidate.settings.get("metric.accumulation")

    if not isinstance(accumulation, str):
        accumulation = (
            "streaming" if value == "streaming_multiply" else "materialized_blocks"
        )

    settings["metric.accumulation"] = accumulation

    return settings


def _metric_multiply_anchor_value(operator: OperatorSpec) -> str:
    kind = metric_representation_kind(operator)

    if kind == "dense_matrix":
        return "dense_matmul"

    if kind == "block_diagonal":
        return "blockwise_multiply"

    if kind == "matrix_free":
        return "streaming_multiply"

    return "factorized_multiply"


def _inverse_metric_anchor_settings(
    operator: OperatorSpec,
    candidate: Candidate,
) -> dict[str, Any]:
    value = candidate.settings.get("inverse_metric.solve_path")

    if not isinstance(value, str):
        value = _inverse_metric_anchor_value(operator)

    settings = {"inverse_metric.solve_path": value}

    for key in (
        "inverse_metric.iteration_budget",
        "inverse_metric.preconditioner",
        "inverse_metric.preconditioner_product",
        "inverse_metric.block_schedule",
        "inverse_metric.multi_rhs",
    ):
        if key in candidate.settings:
            settings[key] = candidate.settings[key]

    if value == "conjugate_gradient":
        settings.update(_metric_multiply_anchor_settings(operator, candidate))

    return settings


def _inverse_metric_anchor_value(operator: OperatorSpec) -> str:
    kind = metric_representation_kind(operator)

    if kind == "dense_matrix":
        return "dense_solve"

    if kind == "matrix_free":
        return "conjugate_gradient"

    if kind == "block_diagonal":
        return "blockwise_solve"

    if kind == "low_rank_factors":
        return "woodbury_low_rank_solve"

    return "factorized_solve"


def _sqrt_metric_anchor_settings(
    operator: OperatorSpec,
    candidate: Candidate,
) -> dict[str, Any]:
    value = candidate.settings.get("sqrt_metric.factor_path")

    if not isinstance(value, str):
        return {}

    settings = {"sqrt_metric.factor_path": value}

    if "sqrt_metric.lanczos_iterations" in candidate.settings:
        settings["sqrt_metric.lanczos_iterations"] = candidate.settings[
            "sqrt_metric.lanczos_iterations"
        ]

    if value == "matrix_free_lanczos":
        settings.update(_metric_multiply_anchor_settings(operator, candidate))

    return settings


def _require_metric_representation(
    operator: OperatorSpec,
    allowed: tuple[str, ...],
) -> None:
    kind = metric_representation_kind(operator)

    if kind not in allowed:
        message = f"metric representation kind is not supported by path: {kind}"
        raise MaterializationError(message)


def metric_representation_kind(operator: OperatorSpec) -> str:
    """Return the metric representation kind.

    Returns:
        The metric representation kind.

    Raises:
        MaterializationError: If the declared inputs are invalid.
    """
    representation = metric_representation(operator)
    kind = representation.get("kind")

    if not isinstance(kind, str):
        message = "metric representation kind is required"
        raise MaterializationError(message)

    return kind


def metric_representation(operator: OperatorSpec) -> Mapping[str, Any]:
    """Return the metric representation.

    Returns:
        The metric representation.

    Raises:
        MaterializationError: If the declared inputs are invalid.
    """
    representation = operator.semantics.get("representation")

    if not isinstance(representation, Mapping):
        message = "metric representation is required"
        raise MaterializationError(message)

    return representation


def _matrix_free_metric_product(operator: OperatorSpec) -> str:
    representation = metric_representation(operator)
    product = representation.get("operator")

    if not isinstance(product, str) or not product:
        message = "matrix_free metric requires a named sibling product"
        raise MaterializationError(message)

    return product
