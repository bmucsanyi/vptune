"""Distributed adapter helpers."""

import dataclasses
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import torch

from vptune.candidates import AxisDescriptor
from vptune.data import PACKAGE_VERSION, Candidate, Measurement
from vptune.errors import AdmissionError, MaterializationError
from vptune.identities import to_json_value

DISTRIBUTED_STRATEGIES = (
    "single_gpu",
    "fsdp2",
    "hsdp",
    "tensor_parallel",
    "sequence_parallel",
    "context_parallel",
    "hybrid",
)
LAYOUT_DISTRIBUTED_STRATEGIES = (
    "tensor_parallel",
    "sequence_parallel",
    "context_parallel",
)
DTENSOR_PLACEMENT_KEYS = (
    "dtensor.params_placement",
    "dtensor.vector_placement",
    "dtensor.logits_placement",
    "dtensor.tangent_placement",
    "dtensor.cotangent_placement",
    "dtensor.output_placement",
)


def build_device_mesh(
    init_device_mesh: Callable[..., Any],
    *,
    device_type: str,
    mesh_shape: tuple[int, ...],
    mesh_dim_names: tuple[str, ...],
) -> Any:
    """Build a PyTorch DeviceMesh.

    Returns:
        DeviceMesh returned by the supplied PyTorch entry point.
    """
    return init_device_mesh(
        device_type,
        mesh_shape,
        mesh_dim_names=mesh_dim_names,
    )


def resolve_process_group_backend(
    backend: str,
    *,
    is_ucc_available: Callable[[], bool],
) -> str:
    """Return the PyTorch process-group backend name.

    Returns:
        Backend name passed to `init_process_group`.

    Raises:
        AdmissionError: If the declared backend is unsupported or unavailable.
    """
    if backend in {"nccl", "gloo"}:
        return backend

    if backend == "ucc_when_available":
        if is_ucc_available():
            return "ucc"

        message = "ucc_when_available requires torch.distributed UCC support"
        raise AdmissionError(message)

    message = f"unsupported distributed process-group backend: {backend}"
    raise AdmissionError(message)


def initialize_process_group(
    init_process_group: Callable[..., Any],
    *,
    backend: str,
    init_method: str | None,
    timeout: Any,
    world_size: int,
    rank: int,
    store: Any,
    pg_options: Any,
    device_id: Any,
) -> Any:
    """Initialize the PyTorch distributed process group.

    Returns:
        Return value from the supplied `init_process_group` entry point.
    """
    return init_process_group(
        backend=backend,
        init_method=init_method,
        timeout=timeout,
        world_size=world_size,
        rank=rank,
        store=store,
        pg_options=pg_options,
        device_id=device_id,
    )


def apply_fsdp2(
    fully_shard: Callable[..., torch.nn.Module],
    module: torch.nn.Module,
    *,
    mesh: Any,
    reshard_after_forward: bool | int | None,
    shard_placement_fn: Callable[..., Any] | None,
    mp_policy: Any,
    offload_policy: Any,
    ignored_params: Sequence[torch.nn.Parameter],
    dp_mesh_dims: Any,
) -> torch.nn.Module:
    """Apply PyTorch FSDP2 to a module.

    Returns:
        Module returned by the supplied `fully_shard` entry point.
    """
    return fully_shard(
        module,
        mesh=mesh,
        reshard_after_forward=reshard_after_forward,
        shard_placement_fn=shard_placement_fn,
        mp_policy=mp_policy,
        offload_policy=offload_policy,
        ignored_params=tuple(ignored_params),
        dp_mesh_dims=dp_mesh_dims,
    )


def named_modules_for_distributed_wrap(
    module: torch.nn.Module,
    module_names: Sequence[str],
) -> tuple[torch.nn.Module, ...]:
    """Return declared submodules for distributed wrapping.

    Returns:
        Submodules in the declared order.

    Raises:
        AdmissionError: If a declared module name is missing.
    """
    named_modules = dict(module.named_modules())
    selected = []

    for module_name in module_names:
        if module_name not in named_modules:
            message = f"declared distributed module is missing: {module_name}"
            raise AdmissionError(message)

        selected.append(named_modules[module_name])

    return tuple(selected)


def apply_fsdp2_group(
    fully_shard: Callable[..., torch.nn.Module],
    modules: list[torch.nn.Module],
    *,
    mesh: Any,
    reshard_after_forward: bool | int | None,
    shard_placement_fn: Callable[..., Any] | None,
    mp_policy: Any,
    offload_policy: Any,
    ignored_params: Sequence[torch.nn.Parameter],
    dp_mesh_dims: Any,
) -> torch.nn.Module:
    """Apply PyTorch FSDP2 to a declared module group.

    Returns:
        Module returned by the supplied `fully_shard` entry point.
    """
    return fully_shard(
        modules,
        mesh=mesh,
        reshard_after_forward=reshard_after_forward,
        shard_placement_fn=shard_placement_fn,
        mp_policy=mp_policy,
        offload_policy=offload_policy,
        ignored_params=tuple(ignored_params),
        dp_mesh_dims=dp_mesh_dims,
    )


def build_fsdp_mixed_precision_policy(
    mixed_precision_policy: Callable[..., Any],
    *,
    param_dtype: torch.dtype | None,
    reduce_dtype: torch.dtype | None,
    output_dtype: torch.dtype | None,
    cast_forward_inputs: bool,
) -> Any:
    """Build a PyTorch FSDP2 mixed-precision policy.

    Returns:
        Policy returned by the supplied constructor.
    """
    return mixed_precision_policy(
        param_dtype=param_dtype,
        reduce_dtype=reduce_dtype,
        output_dtype=output_dtype,
        cast_forward_inputs=cast_forward_inputs,
    )


def build_fsdp_offload_policy(
    offload_policy: Callable[[], Any],
    cpu_offload_policy: Callable[..., Any],
    *,
    offload: str,
    pin_memory: bool,
) -> Any:
    """Build a PyTorch FSDP2 offload policy.

    Returns:
        Policy returned by the supplied constructor.

    Raises:
        AdmissionError: If the declared offload policy is unsupported.
    """
    if offload == "none":
        return offload_policy()

    if offload == "cpu":
        return cpu_offload_policy(pin_memory=pin_memory)

    message = f"unsupported FSDP2 offload policy: {offload}"
    raise AdmissionError(message)


def build_fsdp_dp_mesh_dims(
    data_parallel_mesh_dims: Callable[..., Any],
    *,
    shard: str | tuple[str, ...] | None,
    replicate: str | tuple[str, ...] | None,
) -> Any:
    """Build PyTorch FSDP2 data-parallel mesh dimensions.

    Returns:
        Data-parallel mesh dimension object returned by the supplied constructor.
    """
    return data_parallel_mesh_dims(shard=shard, replicate=replicate)


def apply_tensor_parallel(
    parallelize_module: Callable[..., torch.nn.Module],
    module: torch.nn.Module,
    *,
    device_mesh: Any,
    parallelize_plan: Mapping[str, Any],
    src_data_rank: int,
) -> torch.nn.Module:
    """Apply PyTorch tensor parallelism to a module.

    Returns:
        Module returned by the supplied `parallelize_module` entry point.
    """
    return parallelize_module(
        module,
        device_mesh=device_mesh,
        parallelize_plan=parallelize_plan,
        src_data_rank=src_data_rank,
    )


def build_colwise_parallel(
    colwise_parallel: Callable[..., Any],
    *,
    input_layouts: Any,
    output_layouts: Any,
    use_local_output: bool,
) -> Any:
    """Build a PyTorch column-wise tensor-parallel style.

    Returns:
        Parallel style returned by the supplied constructor.
    """
    return colwise_parallel(
        input_layouts=input_layouts,
        output_layouts=output_layouts,
        use_local_output=use_local_output,
    )


def build_rowwise_parallel(
    rowwise_parallel: Callable[..., Any],
    *,
    input_layouts: Any,
    output_layouts: Any,
    use_local_output: bool,
) -> Any:
    """Build a PyTorch row-wise tensor-parallel style.

    Returns:
        Parallel style returned by the supplied constructor.
    """
    return rowwise_parallel(
        input_layouts=input_layouts,
        output_layouts=output_layouts,
        use_local_output=use_local_output,
    )


def build_sequence_parallel(
    sequence_parallel: Callable[..., Any],
    *,
    sequence_dim: int,
    use_local_output: bool,
) -> Any:
    """Build a PyTorch sequence-parallel style.

    Returns:
        Parallel style returned by the supplied constructor.
    """
    return sequence_parallel(
        sequence_dim=sequence_dim,
        use_local_output=use_local_output,
    )


def build_prepare_module_input(
    prepare_module_input: Callable[..., Any],
    *,
    input_layouts: Any,
    desired_input_layouts: Any,
    input_kwarg_layouts: Mapping[str, Any],
    desired_input_kwarg_layouts: Mapping[str, Any],
    use_local_output: bool,
) -> Any:
    """Build a PyTorch input-layout preparation style.

    Returns:
        Parallel style returned by the supplied constructor.
    """
    return prepare_module_input(
        input_layouts=input_layouts,
        desired_input_layouts=desired_input_layouts,
        input_kwarg_layouts=input_kwarg_layouts,
        desired_input_kwarg_layouts=desired_input_kwarg_layouts,
        use_local_output=use_local_output,
    )


def build_prepare_module_output(
    prepare_module_output: Callable[..., Any],
    *,
    output_layouts: Any,
    desired_output_layouts: Any,
    use_local_output: bool,
) -> Any:
    """Build a PyTorch output-layout preparation style.

    Returns:
        Parallel style returned by the supplied constructor.
    """
    return prepare_module_output(
        output_layouts=output_layouts,
        desired_output_layouts=desired_output_layouts,
        use_local_output=use_local_output,
    )


def run_with_loss_parallel(
    loss_parallel: Callable[[], Any],
    operation: Callable[[], Any],
) -> Any:
    """Run an operation inside PyTorch tensor-parallel loss context.

    Returns:
        Operation result.
    """
    with loss_parallel():
        return operation()


def build_dtensor_placement(
    replicate: Callable[[], Any],
    shard: Callable[[int], Any],
    partial: Callable[[str], Any],
    *,
    placement: str,
    shard_dim: int | None,
    reduce_op: str | None,
) -> Any:
    """Build one DTensor placement object.

    Returns:
        Placement object returned by the supplied PyTorch placement constructor.

    Raises:
        AdmissionError: If the placement fields contradict the placement kind.
    """
    if placement == "replicate":
        if shard_dim is not None or reduce_op is not None:
            message = "replicate placement requires no shard_dim or reduce_op"
            raise AdmissionError(message)

        return replicate()

    if placement == "shard_dim":
        if shard_dim is None or reduce_op is not None:
            message = "shard_dim placement requires shard_dim and no reduce_op"
            raise AdmissionError(message)

        return shard(shard_dim)

    if placement == "partial":
        if shard_dim is not None or reduce_op is None:
            message = "partial placement requires reduce_op and no shard_dim"
            raise AdmissionError(message)

        return partial(reduce_op)

    message = f"unsupported DTensor placement: {placement}"
    raise AdmissionError(message)


def redistribute_dtensor(
    dtensor: Any,
    *,
    device_mesh: Any,
    placements: Sequence[Any],
    async_op: bool,
    forward_dtype: torch.dtype | None,
    backward_dtype: torch.dtype | None,
) -> Any:
    """Redistribute a DTensor at a declared operator boundary.

    Returns:
        DTensor returned by `redistribute`.
    """
    return dtensor.redistribute(
        device_mesh=device_mesh,
        placements=tuple(placements),
        async_op=async_op,
        forward_dtype=forward_dtype,
        backward_dtype=backward_dtype,
    )


def apply_context_parallel(
    context_parallel: Callable[..., Any],
    mesh: Any,
    *,
    buffers: Sequence[torch.Tensor],
    buffer_seq_dims: Sequence[int],
    no_restore_buffers: Sequence[torch.Tensor],
) -> Any:
    """Apply PyTorch context parallelism.

    Returns:
        Return value from the supplied `context_parallel` entry point.
    """
    return context_parallel(
        mesh,
        buffers=tuple(buffers),
        buffer_seq_dims=tuple(buffer_seq_dims),
        no_restore_buffers=tuple(no_restore_buffers),
    )


def collective_all_gather_into_tensor(
    all_gather_into_tensor: Callable[..., Any],
    output_tensor: torch.Tensor,
    input_tensor: torch.Tensor,
    *,
    group: Any,
    async_op: bool,
) -> Any:
    """Run `torch.distributed.all_gather_into_tensor`.

    Returns:
        Return value from the supplied collective entry point.
    """
    return all_gather_into_tensor(
        output_tensor,
        input_tensor,
        group=group,
        async_op=async_op,
    )


def collective_reduce_scatter_tensor(
    reduce_scatter_tensor: Callable[..., Any],
    output_tensor: torch.Tensor,
    input_tensor: torch.Tensor,
    *,
    op: Any,
    group: Any,
    async_op: bool,
) -> Any:
    """Run `torch.distributed.reduce_scatter_tensor`.

    Returns:
        Return value from the supplied collective entry point.
    """
    return reduce_scatter_tensor(
        output_tensor,
        input_tensor,
        op=op,
        group=group,
        async_op=async_op,
    )


def collective_all_to_all_single(
    all_to_all_single: Callable[..., Any],
    output_tensor: torch.Tensor,
    input_tensor: torch.Tensor,
    *,
    output_split_sizes: Sequence[int],
    input_split_sizes: Sequence[int],
    group: Any,
    async_op: bool,
) -> Any:
    """Run `torch.distributed.all_to_all_single`.

    Returns:
        Return value from the supplied collective entry point.
    """
    return all_to_all_single(
        output_tensor,
        input_tensor,
        output_split_sizes=output_split_sizes,
        input_split_sizes=input_split_sizes,
        group=group,
        async_op=async_op,
    )


def wait_collective(work: Any) -> Any:
    """Wait for an async distributed work handle.

    Returns:
        Return value from `work.wait()`.
    """
    return work.wait()


@dataclasses.dataclass(frozen=True, slots=True)
class RankStatus:
    """Status reported by one distributed rank."""

    rank: int
    status: str
    device: str
    error_type: str | None = None
    error: str | None = None

    def to_record(self) -> dict[str, Any]:
        """Return JSON-compatible rank status."""
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True, slots=True)
class RankSelectedSettings:
    """Selected settings reported by one distributed rank."""

    rank: int
    settings: Mapping[str, Any]

    def to_record(self) -> dict[str, Any]:
        """Return JSON-compatible selected settings."""
        return {"rank": self.rank, "settings": dict(self.settings)}


@dataclasses.dataclass(frozen=True, slots=True)
class DistributedAdmissionPolicy:
    """Admission identity for distributed candidates."""

    candidate_generator_version: str
    device_mesh: Mapping[str, Any]
    rank_count: int
    per_rank_placements: tuple[Mapping[str, Any], ...]
    communication: Mapping[str, Any]
    fsdp2: Mapping[str, Any]
    dtensor: Mapping[str, Any]
    tensor_parallel: Mapping[str, Any]
    sequence_parallel: Mapping[str, Any]
    context_parallel: Mapping[str, Any]

    def signature(self) -> dict[str, Any]:
        """Return stable distributed admission identity."""
        return {
            "adapter_id": "vptune.distributed",
            "adapter_version": PACKAGE_VERSION,
            "candidate_generator_version": self.candidate_generator_version,
            "device_mesh": dict(self.device_mesh),
            "rank_count": self.rank_count,
            "per_rank_placements": tuple(
                dict(placement) for placement in self.per_rank_placements
            ),
            "communication": dict(self.communication),
            "fsdp2": dict(self.fsdp2),
            "dtensor": dict(self.dtensor),
            "tensor_parallel": dict(self.tensor_parallel),
            "sequence_parallel": dict(self.sequence_parallel),
            "context_parallel": dict(self.context_parallel),
        }


def distributed_strategy_axis(
    strategies: Sequence[str],
    *,
    policy: DistributedAdmissionPolicy,
) -> AxisDescriptor:
    """Return a distributed strategy axis.

    Raises:
        AdmissionError: If a strategy is unsupported.
    """
    unsupported = tuple(
        strategy for strategy in strategies if strategy not in DISTRIBUTED_STRATEGIES
    )

    if unsupported:
        message = f"unsupported distributed strategies: {unsupported}"
        raise AdmissionError(message)

    return AxisDescriptor(
        name="distributed.strategy",
        settings_keys=("distributed.strategy",),
        allowed_values=tuple(strategies),
        optional_settings_keys=(
            "distributed.launch",
            "distributed.process_group_backend",
            "distributed.local_rank_binding",
            "distributed.mesh_shape",
            "distributed.mesh_dim_names",
            "dtensor.params_placement",
            "dtensor.vector_placement",
            "dtensor.logits_placement",
            "dtensor.tangent_placement",
            "dtensor.cotangent_placement",
            "dtensor.output_placement",
            "dtensor.redistribute_schedule",
            "dtensor.module_class",
            "dtensor.to_local_grad_placement",
            "dtensor.from_local_check",
            "dtensor.uneven_shard_handling",
            "dtensor.async_local_tensor_handling",
            "dtensor.higher_order_diff_status",
            "fsdp.hook_entry_points",
            "fsdp.hook_entry_policy",
            "fsdp.wrap_granularity",
            "fsdp.forward_prefetch",
            "fsdp.backward_prefetch",
            "fsdp.reshard_after_forward",
            "fsdp.shard_placement_fn",
            "fsdp.mp_policy.param_dtype",
            "fsdp.mp_policy.reduce_dtype",
            "fsdp.mp_policy.output_dtype",
            "fsdp.mp_policy.cast_forward_inputs",
            "fsdp.offload_policy",
            "fsdp.ignored_params",
            "fsdp.dp_mesh_dims",
            "fsdp.bypasses_hooks",
            "fsdp.bottom_up_order",
            "fsdp.mutated_modules",
            "fsdp.collectives",
            "tp.plan",
            "tp.qkv_projection",
            "tp.output_projection",
            "tp.mlp_up_gate",
            "tp.mlp_down",
            "tp.embedding",
            "tp.lm_head",
            "tp.prepare_module_input",
            "tp.prepare_module_output",
            "tp.loss_parallel",
            "sequence_parallel.enabled",
            "sequence_parallel.norm_modules",
            "sequence_parallel.output_placement_policy",
            "context_parallel.enabled",
            "context_parallel.rotate_method",
            "context_parallel.sequence_dim",
            "comm.overlap",
            "comm.prefetch",
            "comm.collective_bucket_size",
        ),
        adapter_id="vptune.distributed",
        adapter_version=PACKAGE_VERSION,
        admission_rule=lambda candidate: admit_distributed_candidate(
            candidate,
            policy=policy,
        ),
        identity=policy.signature(),
    )


def admit_distributed_candidate(
    candidate: Candidate,
    *,
    policy: DistributedAdmissionPolicy,
) -> tuple[bool, str | None]:
    """Return whether a distributed candidate is admitted."""
    strategy = candidate.settings.get("distributed.strategy")

    if strategy in {"fsdp2", "hsdp"}:
        return _admit_fsdp2(candidate.settings, policy)

    if strategy in LAYOUT_DISTRIBUTED_STRATEGIES:
        return _admit_layout_sharding(candidate.settings, policy)

    return False, f"unsupported distributed strategy: {strategy}"


def _admit_fsdp2(
    settings: Mapping[str, Any],
    policy: DistributedAdmissionPolicy,
) -> tuple[bool, str | None]:
    validation_error = _fsdp2_validation_error(settings, policy)

    if validation_error is not None:
        return False, validation_error

    return True, None


def _fsdp2_validation_error(
    settings: Mapping[str, Any],
    policy: DistributedAdmissionPolicy,
) -> str | None:
    return _first_error((
        _non_empty_string_sequence(settings, "fsdp.hook_entry_points"),
        _fsdp2_policy_error(settings, policy),
        _fsdp2_hook_bypass_error(settings),
        _required_bool(settings, "fsdp.bypasses_hooks"),
        _required_bool(settings, "fsdp.bottom_up_order"),
        _string_sequence(settings, "fsdp.mutated_modules"),
        _non_empty_mapping(settings, "fsdp.collectives"),
    ))


def _fsdp2_policy_error(
    settings: Mapping[str, Any],
    policy: DistributedAdmissionPolicy,
) -> str | None:
    return _policy_fields_error(
        settings,
        policy.fsdp2,
        (
            "fsdp.hook_entry_policy",
            "fsdp.wrap_granularity",
            "fsdp.forward_prefetch",
            "fsdp.backward_prefetch",
            "fsdp.reshard_after_forward",
            "fsdp.shard_placement_fn",
            "fsdp.mp_policy.param_dtype",
            "fsdp.mp_policy.reduce_dtype",
            "fsdp.mp_policy.output_dtype",
            "fsdp.mp_policy.cast_forward_inputs",
            "fsdp.offload_policy",
        ),
    )


def _policy_fields_error(
    settings: Mapping[str, Any],
    policy: Mapping[str, Any],
    keys: Sequence[str],
) -> str | None:
    for key in keys:
        policy_error = _allowed_policy(settings, key, policy)

        if policy_error is not None:
            return policy_error

    return None


def _fsdp2_hook_bypass_error(settings: Mapping[str, Any]) -> str | None:
    if settings.get("fsdp.bypasses_hooks") is not False:
        return "fsdp2 candidates must not bypass FSDP hooks"

    return None


def _first_error(errors: Sequence[str | None]) -> str | None:
    for error in errors:
        if error is not None:
            return error

    return None


def _admit_layout_sharding(
    settings: Mapping[str, Any],
    policy: DistributedAdmissionPolicy,
) -> tuple[bool, str | None]:
    validation_error = _layout_validation_error(settings, policy)

    if validation_error is not None:
        return False, validation_error

    return True, None


def _layout_validation_error(
    settings: Mapping[str, Any],
    policy: DistributedAdmissionPolicy,
) -> str | None:
    return _first_error((
        _layout_common_error(settings, policy),
        _layout_mode_error(settings, policy),
    ))


def _layout_common_error(
    settings: Mapping[str, Any],
    policy: DistributedAdmissionPolicy,
) -> str | None:
    placement_error = _layout_placement_error(settings)

    if placement_error is not None:
        return placement_error

    policy_error = _policy_fields_error(
        settings,
        policy.dtensor,
        (
            "dtensor.module_class",
            "dtensor.to_local_grad_placement",
            "dtensor.from_local_check",
            "dtensor.uneven_shard_handling",
            "dtensor.async_local_tensor_handling",
            "dtensor.redistribute_schedule",
        ),
    )

    if policy_error is not None:
        return policy_error

    return _higher_order_diff_status_error(settings, policy)


def _higher_order_diff_status_error(
    settings: Mapping[str, Any],
    policy: DistributedAdmissionPolicy,
) -> str | None:
    status = settings.get("dtensor.higher_order_diff_status")

    if not isinstance(status, Mapping):
        return "dtensor.higher_order_diff_status must be a mapping"

    expected = _higher_order_diff_status_keys(settings)

    if set(status) != set(expected):
        return "dtensor.higher_order_diff_status must cover every placement slot"

    allowed = policy.dtensor.get("allowed_higher_order_diff_status")

    if not isinstance(allowed, tuple):
        return "higher_order_diff_status policy must declare allowed values"

    for slot, value in status.items():
        if not isinstance(value, str) or value not in allowed:
            return (
                f"dtensor.higher_order_diff_status is not allowed for {slot}: {value}"
            )

    return None


def _higher_order_diff_status_keys(settings: Mapping[str, Any]) -> tuple[str, ...]:
    return tuple(key for key in DTENSOR_PLACEMENT_KEYS if key in settings)


def _layout_placement_error(settings: Mapping[str, Any]) -> str | None:
    for key in DTENSOR_PLACEMENT_KEYS:
        placement_error = _required_string(settings, key)

        if placement_error is not None:
            return placement_error

    return None


def _layout_mode_error(
    settings: Mapping[str, Any],
    policy: DistributedAdmissionPolicy,
) -> str | None:
    strategy = settings.get("distributed.strategy")

    if strategy == "tensor_parallel":
        return _policy_fields_error(
            settings,
            policy.tensor_parallel,
            (
                "tp.plan",
                "tp.qkv_projection",
                "tp.output_projection",
                "tp.mlp_up_gate",
                "tp.mlp_down",
                "tp.embedding",
                "tp.lm_head",
                "tp.prepare_module_input",
                "tp.prepare_module_output",
                "tp.loss_parallel",
            ),
        )

    if strategy == "sequence_parallel":
        return _policy_fields_error(
            settings,
            policy.sequence_parallel,
            (
                "sequence_parallel.enabled",
                "sequence_parallel.norm_modules",
                "sequence_parallel.output_placement_policy",
            ),
        )

    if strategy == "context_parallel":
        return _policy_fields_error(
            settings,
            policy.context_parallel,
            (
                "context_parallel.enabled",
                "context_parallel.rotate_method",
                "context_parallel.sequence_dim",
            ),
        )

    return f"unsupported layout distributed strategy: {strategy}"


def _non_empty_string_sequence(
    settings: Mapping[str, Any],
    key: str,
) -> str | None:
    value = settings.get(key)

    if not isinstance(value, tuple) or not value:
        return f"{key} must be a non-empty tuple of strings"

    if not all(isinstance(item, str) and item for item in value):
        return f"{key} must be a non-empty tuple of strings"

    return None


def _required_string(settings: Mapping[str, Any], key: str) -> str | None:
    value = settings.get(key)

    if not isinstance(value, str) or not value:
        return f"{key} must be a non-empty string"

    return None


def _string_sequence(settings: Mapping[str, Any], key: str) -> str | None:
    value = settings.get(key)

    if not isinstance(value, tuple):
        return f"{key} must be a tuple of strings"

    if not all(isinstance(item, str) and item for item in value):
        return f"{key} must be a tuple of strings"

    return None


def _non_empty_mapping(settings: Mapping[str, Any], key: str) -> str | None:
    value = settings.get(key)

    if not isinstance(value, Mapping) or not value:
        return f"{key} must be a non-empty mapping"

    return None


def _required_bool(settings: Mapping[str, Any], key: str) -> str | None:
    if not isinstance(settings.get(key), bool):
        return f"{key} must be a bool"

    return None


def _allowed_policy(
    settings: Mapping[str, Any],
    key: str,
    policy: Mapping[str, Any],
) -> str | None:
    value = settings.get(key)
    allowed = policy.get(f"allowed_{key}")

    if value is None:
        return f"{key} must be declared"

    if isinstance(value, str) and len(value) == 0:
        return f"{key} must be declared"

    if not isinstance(allowed, tuple) or value not in allowed:
        return f"{key} is not allowed by distributed admission policy: {value}"

    return None


def reduce_rank_statuses(statuses: Sequence[RankStatus]) -> dict[str, Any]:
    """Return global status from rank-local statuses.

    Raises:
        RuntimeError: If no rank status is supplied.
    """
    if not statuses:
        message = "distributed status reduction requires rank statuses"
        raise RuntimeError(message)

    failed = tuple(status for status in statuses if status.status != "passed")

    if failed:
        return {
            "status": "failed",
            "rank_statuses": tuple(status.to_record() for status in statuses),
            "failed_ranks": tuple(status.rank for status in failed),
        }

    return {
        "status": "passed",
        "rank_statuses": tuple(status.to_record() for status in statuses),
        "failed_ranks": (),
    }


def require_rank_selected_settings_agree(
    rank_settings: Sequence[RankSelectedSettings],
) -> Mapping[str, Any]:
    """Return selected settings when every rank agrees.

    Raises:
        MaterializationError: If rank settings are empty or disagree.
    """
    if not rank_settings:
        message = "distributed selected settings require rank reports"
        raise MaterializationError(message)

    first = rank_settings[0].settings

    for report in rank_settings[1:]:
        if to_json_value(report.settings) != to_json_value(first):
            message = "distributed ranks selected different settings"
            raise MaterializationError(message)

    return dict(first)


def distributed_record(
    *,
    identity: Mapping[str, Any],
    expected_rank_count: int,
    rank_statuses: Sequence[RankStatus],
    rank_memory_samples: Sequence[Measurement],
    rank_selected_settings: Sequence[RankSelectedSettings],
    global_parameter_surface: Mapping[str, Any],
) -> dict[str, Any]:
    """Return a distributed row record payload."""
    _require_distributed_identity(identity)
    _require_matching_rank_sets(
        expected_rank_count,
        rank_statuses,
        rank_memory_samples,
        rank_selected_settings,
    )
    selected_settings = require_rank_selected_settings_agree(rank_selected_settings)
    global_status = reduce_rank_statuses(rank_statuses)

    return {
        "identity": dict(identity),
        "status": global_status["status"],
        "rank_count": len(rank_statuses),
        "rank_statuses": global_status["rank_statuses"],
        "failed_ranks": global_status["failed_ranks"],
        "rank_memory_samples": tuple(
            sample.to_record() for sample in rank_memory_samples
        ),
        "global_elapsed_seconds": max(
            sample.elapsed_seconds for sample in rank_memory_samples
        ),
        "global_peak_allocated_mib": max(
            sample.peak_allocated_mib for sample in rank_memory_samples
        ),
        "global_peak_reserved_mib": max(
            sample.peak_reserved_mib for sample in rank_memory_samples
        ),
        "global_post_allocated_mib": max(
            sample.post_allocated_mib for sample in rank_memory_samples
        ),
        "global_post_reserved_mib": max(
            sample.post_reserved_mib for sample in rank_memory_samples
        ),
        "global_parameter_surface": dict(global_parameter_surface),
        "selected_settings": dict(selected_settings),
        "rank_selected_settings": tuple(
            report.to_record() for report in rank_selected_settings
        ),
    }


def _require_matching_rank_sets(
    expected_rank_count: int,
    rank_statuses: Sequence[RankStatus],
    rank_memory_samples: Sequence[Measurement],
    rank_selected_settings: Sequence[RankSelectedSettings],
) -> None:
    if expected_rank_count < 1:
        message = "distributed records require a positive expected rank count"
        raise MaterializationError(message)

    status_ranks = _rank_set(tuple(status.rank for status in rank_statuses), "status")
    memory_ranks = _rank_set(
        tuple(sample.rank for sample in rank_memory_samples),
        "memory",
    )
    selected_ranks = _rank_set(
        tuple(report.rank for report in rank_selected_settings),
        "selected settings",
    )

    if status_ranks != memory_ranks or status_ranks != selected_ranks:
        message = "distributed rank sets differ across status, memory, and settings"
        raise MaterializationError(message)

    if len(status_ranks) != expected_rank_count:
        message = "distributed rank set differs from expected rank count"
        raise MaterializationError(message)

    expected_ranks = set(range(expected_rank_count))

    if status_ranks != expected_ranks:
        message = "distributed rank set must be contiguous from zero"
        raise MaterializationError(message)


def _rank_set(ranks: Sequence[int], label: str) -> set[int]:
    if not ranks:
        message = f"distributed {label} ranks are required"
        raise MaterializationError(message)

    rank_set = set(ranks)

    if len(rank_set) != len(ranks):
        message = f"distributed {label} ranks contain duplicates"
        raise MaterializationError(message)

    return rank_set


def _require_distributed_identity(identity: Mapping[str, Any]) -> None:
    required = (
        "adapter_id",
        "adapter_version",
        "device_mesh",
        "placements",
        "communication",
    )
    missing = tuple(key for key in required if key not in identity)

    if missing:
        message = f"distributed identity missing fields: {missing}"
        raise MaterializationError(message)

    if identity["adapter_id"] != "vptune.distributed":
        message = "distributed identity adapter_id differs"
        raise MaterializationError(message)

    for key in ("device_mesh", "communication"):
        if not isinstance(identity[key], Mapping) or not identity[key]:
            message = f"distributed identity {key} must be a non-empty mapping"
            raise MaterializationError(message)

    placements = identity["placements"]

    if not isinstance(placements, tuple) or not placements:
        message = "distributed identity placements must be a non-empty tuple"
        raise MaterializationError(message)


def distributed_identity(
    *,
    device_mesh: Mapping[str, Any],
    placements: Sequence[Mapping[str, Any]],
    communication: Mapping[str, Any],
) -> dict[str, Any]:
    """Return distributed adapter identity."""
    return {
        "adapter_id": "vptune.distributed",
        "adapter_version": PACKAGE_VERSION,
        "device_mesh": dict(device_mesh),
        "placements": tuple(dict(placement) for placement in placements),
        "communication": dict(communication),
    }
