"""Layout and dtype lowerings for the standard runtime.

Flat, per-layer, and per-block vector layouts, contiguity, tied
parameter aliases, parametrization preservation, foreach vector
ops, and dtype and matmul precision handling.
"""

from collections.abc import Mapping
from typing import Any

import torch

from vptune import memory, runtime, runtime_values
from vptune.checks import (
    tree_error_measurements,
)
from vptune.data import (
    Batch,
    Candidate,
    ParameterSurface,
    ParameterTree,
)
from vptune.errors import (
    MaterializationError,
)
from vptune.tensor_tree import (
    TensorTree,
    tree_map,
)


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
        return runtime.tree_dot_runtime(settings, left, right)

    return runtime.dot_runtime(
        settings,
        runtime_values.flatten_vector(left),
        runtime_values.flatten_vector(right),
    )


def require_dtype_runtime_settings(settings: Mapping[str, Any]) -> None:
    """Validate dtype runtime settings.

    Raises:
        MaterializationError: If the declared inputs are invalid.
    """
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


def require_layout_runtime_settings(settings: Mapping[str, Any]) -> None:
    """Validate layout runtime settings.

    Raises:
        MaterializationError: If the declared inputs are invalid.
    """
    flatten_order = settings.get("layout.flatten_order")

    if flatten_order is not None and flatten_order != "canonical_parameter_order":
        message = f"layout.flatten_order is unsupported: {flatten_order}"
        raise MaterializationError(message)

    _layout_tree_input(settings, "layout.params")
    _layout_tree_input(settings, "layout.vector")
    layout_output(settings)
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


def layout_output(settings: Mapping[str, Any]) -> str:
    """Return the layout output.

    Returns:
        The layout output.

    Raises:
        MaterializationError: If the declared inputs are invalid.
    """
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


def matmul_runtime(
    settings: Mapping[str, Any],
    left: torch.Tensor,
    right: torch.Tensor,
) -> torch.Tensor:
    """Return the runtime matrix product under the declared precision.

    Returns:
        the runtime matrix product under the declared precision.
    """
    left = runtime.runtime_intermediate_tensor(left, settings)
    right = runtime.runtime_intermediate_tensor(right, settings)

    return runtime.accumulation_tensor(left, settings) @ runtime.accumulation_tensor(
        right,
        settings,
    )


def runtime_grouped_parameter_layout(
    tree: ParameterTree,
    settings: Mapping[str, Any],
    key: str,
    parameter_surface: ParameterSurface | None,
) -> ParameterTree:
    """Return the runtime grouped parameter layout.

    Returns:
        The runtime grouped parameter layout.
    """
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


def runtime_grouped_output_layout(
    tree: TensorTree,
    settings: Mapping[str, Any],
    parameter_surface: ParameterSurface | None,
) -> TensorTree:
    """Return the runtime grouped output layout.

    Returns:
        The runtime grouped output layout.
    """
    groups = _parameter_layout_groups(settings, "layout.output", parameter_surface)

    if groups is None:
        return tree

    result = runtime_values.parameter_tree_from_tensor_tree(
        tree,
        "grouped output layout",
    )

    return runtime_values.wrap_grouped_parameter_tree(result, groups, "layout.output")


def uses_declared_batch_layout(settings: Mapping[str, Any]) -> bool:
    """Return the uses declared batch layout.

    Returns:
        The uses declared batch layout.
    """
    return (
        settings.get("input.batch_layout")
        in {"packed_with_inverse_permutation", "variable_length"}
        or settings.get("input.length_grouping") == "exact_length_bucket"
        or settings.get("schedule.per_token") == "packed"
    )


def runtime_batch_after_contiguity(
    batch: Batch,
    settings: Mapping[str, Any],
    *,
    move_input_residency: bool,
) -> Batch:
    """Return the runtime batch after contiguity.

    Returns:
        The runtime batch after contiguity.
    """
    result = _runtime_batch_contiguity(batch, settings)

    if move_input_residency:
        result = memory.runtime_batch_input_residency(result, settings)

    return _runtime_batch_teacher_outputs(result, settings)


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
        runtime.require_teacher_output_tree(batch["teacher_outputs"])

        return result

    message = f"teacher_outputs is unsupported: {value}"
    raise MaterializationError(message)


def _teacher_outputs_to_device(value: Any, device: torch.device) -> TensorTree:
    return runtime_values.runtime_nested_tensor_value(
        value,
        lambda tensor: tensor.to(device=device),
        error_message="teacher_outputs batch field must be a tensor tree",
    )


def _teacher_outputs_pin_cpu(value: Any) -> TensorTree:
    cpu_value = _teacher_outputs_to_device(value, torch.device("cpu"))

    return tree_map(runtime_values.pin_cpu_tensor, cpu_value)


def runtime_vector_layout(
    vector: TensorTree,
    settings: Mapping[str, Any],
    template: TensorTree,
    parameter_surface: ParameterSurface | None,
) -> TensorTree:
    """Return the runtime vector layout.

    Returns:
        The runtime vector layout.
    """
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

    return runtime_grouped_parameter_layout(
        parameter_tree,
        settings,
        "layout.vector",
        parameter_surface,
    )


def runtime_named_tensor_dtype(
    tree: dict[str, torch.Tensor],
    dtype: torch.dtype | None,
) -> dict[str, torch.Tensor]:
    """Return the runtime named tensor dtype.

    Returns:
        The runtime named tensor dtype.
    """
    if dtype is None:
        return tree

    return runtime_values.runtime_named_tensor_map_preserve_alias(
        tree,
        lambda tensor: tensor.to(dtype=dtype),
    )


def require_alias_safe_parameter_layout(
    params: ParameterTree,
    settings: Mapping[str, Any],
) -> None:
    """Validate alias safe parameter layout.

    Raises:
        MaterializationError: If the declared inputs are invalid.
    """
    if not runtime_values.preserves_parameter_aliases(settings):
        return

    if not runtime_values.has_parameter_aliases(params):
        return

    message = "non-tree parameter layout cannot preserve tied parameter aliases"
    raise MaterializationError(message)


def runtime_tree_contiguity(
    tree: TensorTree,
    settings: Mapping[str, Any],
) -> TensorTree:
    """Return the runtime tree contiguity.

    Returns:
        The runtime tree contiguity.
    """
    if not _layout_contiguity_enabled(settings):
        return tree

    return tree_map(lambda tensor: tensor.contiguous(), tree)


def runtime_named_tensor_contiguity(
    tree: dict[str, torch.Tensor],
    settings: Mapping[str, Any],
) -> dict[str, torch.Tensor]:
    """Return the runtime named tensor contiguity.

    Returns:
        The runtime named tensor contiguity.
    """
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


def parameter_dtype(settings: Mapping[str, Any]) -> torch.dtype | None:
    """Return the parameter dtype.

    Returns:
        The parameter dtype.
    """
    autodiff_dtype = dtype_setting(settings, "dtype.autodiff_compute")

    if autodiff_dtype is not None:
        return autodiff_dtype

    return dtype_setting(settings, "dtype.parameter_storage")


def batch_dtype(settings: Mapping[str, Any]) -> torch.dtype | None:
    """Return the batch dtype.

    Returns:
        The batch dtype.
    """
    autodiff_dtype = dtype_setting(settings, "dtype.autodiff_compute")

    if autodiff_dtype is not None:
        return autodiff_dtype

    return dtype_setting(settings, "dtype.intermediate")


def dtype_setting(settings: Mapping[str, Any], key: str) -> torch.dtype | None:
    """Return the dtype setting.

    Returns:
        The dtype setting.

    Raises:
        MaterializationError: If the declared inputs are invalid.
    """
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


def matmul_precision_setting(settings: Mapping[str, Any]) -> str | None:
    """Return the matmul precision setting.

    Returns:
        The matmul precision setting.

    Raises:
        MaterializationError: If the declared inputs are invalid.
    """
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
