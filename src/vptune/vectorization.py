"""Vectorization mode lowerings for the standard runtime.

Single-loop, manual-batch, and torch.func.vmap execution of vector
operations, with in_dims validation and flat vector batch builders.
"""

from collections.abc import Callable, Iterator, Mapping
from typing import Any

import torch

from vptune import derivatives, runtime, runtime_values
from vptune.admission import (
    admit_torch_func,
)
from vptune.anchors import (
    jvp_anchor,
)
from vptune.data import (
    Batch,
    ParameterTree,
)
from vptune.errors import (
    AdmissionError,
    MaterializationError,
)
from vptune.tensor_tree import (
    TensorTree,
    tree_from_leaves,
)


def microbatch_in_dims(batch: Batch) -> tuple[dict[str, Any], dict[str, int | None]]:
    """Return the microbatch in dims.

    Returns:
        The microbatch in dims.
    """
    return per_example_batch_in_dims(batch, "microbatch accumulation")


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


def run_jvp_vector_vmap(execution: runtime_values.StandardExecution) -> TensorTree:
    """Run jvp vector vmap.

    Returns:
        The jvp vector vmap result.

    Raises:
        MaterializationError: If the declared inputs are invalid.
    """
    if execution.path not in runtime_values.JVP_VECTOR_VMAP_PATHS:
        message = "vectorization.mode=vmap requires a torch.func JVP path"
        raise MaterializationError(message)

    if execution.path == runtime_values.JVP_LINEARIZE_PATH:
        if execution.linearized_jvp is not None:
            jvp_function = execution.linearized_jvp
        else:
            _, jvp_function = torch.func.linearize(
                derivatives.jvp_tensor_function(execution),
                execution.params,
            )

        return run_vector_vmap(execution, jvp_function)

    tensor_function = derivatives.jvp_tensor_function(execution)

    def jvp_function(vector: TensorTree) -> TensorTree:
        return jvp_anchor(
            tensor_function,
            execution.params,
            vector,
        )

    return run_vector_vmap(execution, jvp_function)


def run_vjp_vector_vmap(execution: runtime_values.StandardExecution) -> TensorTree:
    """Run vjp vector vmap.

    Returns:
        The vjp vector vmap result.

    Raises:
        MaterializationError: If the declared inputs are invalid.
    """
    if execution.path not in runtime_values.VJP_VECTOR_VMAP_PATHS:
        message = "vectorization.mode=vmap requires torch_func_vjp"
        raise MaterializationError(message)

    closure = execution.vjp_closure

    if closure is not None:

        def vjp_function(vector: TensorTree) -> TensorTree:
            return closure(vector)

    else:
        pullback = derivatives.vjp_pullback(
            derivatives.vjp_tensor_function(execution),
            execution.params,
        )

        def vjp_function(vector: TensorTree) -> TensorTree:
            (result,) = pullback(vector)

            return result

    return run_vector_vmap(execution, vjp_function)


def run_hvp_vector_single_loop(
    execution: runtime_values.StandardExecution,
) -> TensorTree:
    """Run hvp vector single loop.

    Returns:
        The hvp vector single loop result.
    """
    if _hvp_uses_reverse_reuse(execution.candidate.settings):
        return _run_hvp_reused_reverse_vectors(execution)

    return run_vector_single_loop(execution, derivatives.run_hvp_single_vector)


def run_hvp_vector_manual_batch(
    execution: runtime_values.StandardExecution,
) -> TensorTree:
    """Run hvp vector manual batch.

    Returns:
        The hvp vector manual batch result.
    """
    return run_vector_manual_batches(execution, run_hvp_vector_single_loop)


def run_vector_manual_batches(
    execution: runtime_values.StandardExecution,
    runner: Callable[[runtime_values.StandardExecution], TensorTree],
) -> TensorTree:
    """Run a vector operation over declared manual batches.

    Returns:
        Run a vector operation over declared manual batches.
    """

    def vector_runner(vector: TensorTree) -> TensorTree:
        return runner(runtime.execution_with_vector(execution, vector))

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


def run_tensor_tree_by_vectorization_mode(
    vector_tree: TensorTree,
    settings: Mapping[str, Any],
    runner: Callable[[TensorTree], TensorTree],
) -> TensorTree:
    """Run tensor tree by vectorization mode.

    Returns:
        The tensor tree by vectorization mode result.
    """
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
        return runner(runtime.execution_with_vector(execution, vector))

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
    scalar_function = derivatives.hvp_scalar_function(execution)
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
    dot = runtime.tree_dot_runtime(settings, gradient_tree, vector)
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


def run_hvp_vector_vmap(execution: runtime_values.StandardExecution) -> TensorTree:
    """Run hvp vector vmap.

    Returns:
        The hvp vector vmap result.

    Raises:
        MaterializationError: If the declared inputs are invalid.
    """
    if execution.path not in runtime_values.HVP_VECTOR_VMAP_PATHS:
        message = "vectorization.mode=vmap requires linearize_grad HVP"
        raise MaterializationError(message)

    if execution.linearized_hvp is not None:
        hvp_function = execution.linearized_hvp
    else:
        gradient_function = torch.func.grad(derivatives.hvp_scalar_function(execution))
        _, hvp_function = torch.func.linearize(
            gradient_function,
            execution.params,
        )

    return run_vector_vmap(execution, hvp_function)


def streaming_gradient_rows_vmap(
    execution: runtime_values.StandardExecution,
) -> Iterator[torch.Tensor]:
    """Return the streaming gradient rows vmap.

    Yields:
        The streaming gradient rows vmap items.
    """
    gradients, parameter_items, row_count = _per_example_gradient_tree_vmap(execution)
    pieces = []

    for name, _ in parameter_items:
        gradient = gradients[name].reshape(row_count, -1)
        pieces.append(gradient)

    for index in range(row_count):
        yield torch.cat(tuple(piece[index] for piece in pieces))


def build_flat_vector_batch(
    execution: runtime_values.StandardExecution,
) -> torch.Tensor:
    """Build the flat vector batch for vectorized execution.

    Returns:
        The flat vector batch for vectorized execution.
    """
    vector_in_dims = vector_tree_in_dims(
        execution.vector,
        execution.candidate.settings,
    )

    return runtime_values.flatten_vector_batch(
        execution.params, execution.vector, vector_in_dims
    )


def per_example_gradient_matrix_without_manual_batch(
    execution: runtime_values.StandardExecution,
) -> torch.Tensor:
    """Return the per example gradient matrix without manual batch.

    Returns:
        The per example gradient matrix without manual batch.
    """
    return derivatives.per_example_gradient_matrix_from_builders(
        execution,
        derivatives.PER_EXAMPLE_GRADIENT_WITHOUT_MANUAL_BUILDERS,
        (
            "schedule.per_example=manual_batch is incompatible with path: "
            f"{execution.path}"
        ),
    )


def per_example_gradient_matrix_vmap(
    execution: runtime_values.StandardExecution,
) -> torch.Tensor:
    """Return the per example gradient matrix vmap.

    Returns:
        The per example gradient matrix vmap.
    """
    gradients, parameter_items, row_count = _per_example_gradient_tree_vmap(execution)
    pieces = []

    for name, _ in parameter_items:
        gradient = gradients[name]
        pieces.append(gradient.reshape(row_count, -1))

    return torch.cat(tuple(pieces), dim=1)


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
        output = runtime.call_function_objective(
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
    return per_example_batch_in_dims(batch, "per-example vmap")


def per_example_batch_in_dims(
    batch: Batch,
    label: str,
) -> tuple[dict[str, Any], dict[str, int | None]]:
    """Return vmap in_dims for a per-example batch.

    Returns:
        The vmap in_dims for a per-example batch.

    Raises:
        MaterializationError: If the declared inputs are invalid.
    """
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


def require_vectorization_mode_settings(
    operator_kind: str,
    path: str | None,
    settings: Mapping[str, Any],
) -> None:
    """Validate vectorization mode settings.

    Raises:
        MaterializationError: If the declared inputs are invalid.
    """
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


def require_vectorization_setting_keys(
    operator_kind: str,
    path: str | None,
    settings: Mapping[str, Any],
) -> None:
    """Validate vectorization setting keys.

    Raises:
        MaterializationError: If the declared inputs are invalid.
    """
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
