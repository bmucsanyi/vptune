"""Activation and memory lowerings for the standard runtime.

Checkpointing with RNG preservation, manual recompute, saved-tensor
offload hooks, residency movement, and output buffers.
"""

import dataclasses
from collections.abc import Callable, Mapping, Sequence
from itertools import starmap
from typing import Any

import torch
from torch.utils.checkpoint import checkpoint, noop_context_fn

from vptune.axes.admission import (
    admit_checkpoint,
)
from vptune.core.data import (
    Batch,
    Candidate,
    CandidateOperation,
    FunctionObjective,
    OperatorSpec,
)
from vptune.core.tensor_tree import (
    TensorTree,
    tree_map,
)
from vptune.engine import (
    derivatives,
    runtime,
    runtime_values,
)
from vptune.errors import (
    AdmissionError,
    MaterializationError,
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


def activation_operation(
    execution: runtime_values.StandardExecution,
    operation: CandidateOperation,
) -> CandidateOperation:
    """Return the activation operation.

    Returns:
        The activation operation.

    Raises:
        MaterializationError: If the declared inputs are invalid.
    """
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


def execution_with_inside_input_residency(
    execution: runtime_values.StandardExecution,
) -> runtime_values.StandardExecution:
    """Return the execution with inside input residency.

    Returns:
        The execution with inside input residency.
    """
    if move_input_residency_outside_measured_call(execution.candidate.settings):
        return execution

    return dataclasses.replace(
        execution,
        batch=runtime_batch_input_residency(
            execution.batch,
            execution.candidate.settings,
        ),
    )


def move_input_residency_outside_measured_call(
    settings: Mapping[str, Any],
) -> bool:
    """Return the move input residency outside measured call.

    Returns:
        The move input residency outside measured call.

    Raises:
        MaterializationError: If the declared inputs are invalid.
    """
    value = settings.get("input.host_to_device")

    if value is None or value == "outside_measured_call":
        return True

    if value == "inside_measured_call":
        return False

    message = f"input.host_to_device is unsupported: {value}"
    raise MaterializationError(message)


def require_recomputed_teacher_objective(
    settings: Mapping[str, Any],
    teacher_objective: FunctionObjective | None,
) -> None:
    """Validate recomputed teacher objective.

    Raises:
        MaterializationError: If the declared inputs are invalid.
    """
    if settings.get("teacher_outputs") != "recomputed_with_equality_check":
        return

    if teacher_objective is None:
        message = "recomputed teacher outputs require a teacher objective"
        raise MaterializationError(message)


def require_input_residency_settings(settings: Mapping[str, Any]) -> None:
    """Validate input residency settings.

    Raises:
        MaterializationError: If the declared inputs are invalid.
    """
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


def require_memory_residency_settings(
    operator: OperatorSpec,
    settings: Mapping[str, Any],
    mmap_residency: Callable[[torch.Tensor, str], torch.Tensor] | None,
) -> None:
    """Validate memory residency settings.

    Raises:
        MaterializationError: If the declared inputs are invalid.
    """
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


def require_memory_recompute_settings(
    operator: OperatorSpec,
    settings: Mapping[str, Any],
) -> None:
    """Validate memory recompute settings.

    Raises:
        MaterializationError: If the declared inputs are invalid.
    """
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


def require_activation_runtime_settings(
    settings: Mapping[str, Any],
    activation_pack_hooks: runtime_values.ActivationPackHooks | None,
    activation_unpack_hooks: runtime_values.ActivationUnpackHooks | None,
    checkpoint_contexts: runtime_values.CheckpointContextFns | None,
) -> None:
    """Validate activation runtime settings.

    Raises:
        MaterializationError: If the declared inputs are invalid.
    """
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


def execution_with_recomputed_teacher_outputs(
    execution: runtime_values.StandardExecution,
) -> runtime_values.StandardExecution:
    """Return the execution with recomputed teacher outputs.

    Returns:
        The execution with recomputed teacher outputs.

    Raises:
        MaterializationError: If the declared inputs are invalid.
    """
    if execution.candidate.settings.get("teacher_outputs") != (
        "recomputed_with_equality_check"
    ):
        return execution

    if execution.teacher_objective is None:
        message = "recomputed teacher outputs require a teacher objective"
        raise MaterializationError(message)

    fixed_outputs = execution.batch.get("teacher_outputs")
    runtime.require_teacher_output_tree(fixed_outputs)
    settings = execution.candidate.settings
    recomputed_outputs = execution.teacher_objective(
        runtime.model_compute_tree(execution.params, settings),
        runtime.model_compute_tree(execution.buffers, settings),
        runtime.model_compute_batch(execution.batch, settings),
        execution.context,
    )
    _require_teacher_outputs_match(fixed_outputs, recomputed_outputs)
    batch = dict(execution.batch)
    batch["teacher_outputs"] = recomputed_outputs

    return dataclasses.replace(execution, batch=batch)


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


def runtime_vector_residency(
    vector: TensorTree,
    settings: Mapping[str, Any],
    mmap_residency: Callable[[torch.Tensor, str], torch.Tensor] | None,
) -> TensorTree:
    """Return the runtime vector residency.

    Returns:
        The runtime vector residency.
    """
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


def standard_output_buffer(
    execution: runtime_values.StandardExecution,
) -> TensorTree | None:
    """Return the standard output buffer.

    Returns:
        The standard output buffer.
    """
    if execution.candidate.settings.get("memory.output_buffers") != "preallocated":
        return None

    template = _standard_output_template(execution)
    runtime_template = runtime.runtime_output(
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
