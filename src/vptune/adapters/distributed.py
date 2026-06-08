"""Distributed adapter helpers."""

import dataclasses
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Protocol, TypeGuard

import torch

from vptune.candidates import (
    FSDP_RESHARD_AFTER_FORWARD_DOMAIN,
    INTEGER_DOMAIN,
    AxisDescriptor,
    AxisRegistry,
)
from vptune.checks import tree_error_measurements, validate_thresholds
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
    FullSizeCheck,
    FullSizeRecord,
    FunctionObjective,
    Measurement,
    ModuleCallSpec,
    OperationFactory,
    OperatorSpec,
    ParameterSurface,
    ParameterTree,
    ReferenceCheck,
    ReferenceResult,
    RuntimeConfig,
    RuntimeOperationFactory,
    ScalarObjective,
)
from vptune.errors import AdmissionError, MaterializationError
from vptune.identities import module_identity, to_json_value
from vptune.runtime import standard_operation_factory, standard_reference_check
from vptune.tensor_tree import TensorTree, tree_signature

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
TENSOR_PARALLEL_STYLE_KEYS = (
    "tp.qkv_projection",
    "tp.output_projection",
    "tp.mlp_up_gate",
    "tp.mlp_down",
    "tp.embedding",
    "tp.lm_head",
)
DISTRIBUTED_RUNTIME_SETTING_PREFIXES = (
    "distributed.",
    "dtensor.",
    "fsdp.",
    "tp.",
    "sequence_parallel.",
    "context_parallel.",
    "comm.",
)
DISTRIBUTED_LAYOUT_VALUES = ("per_shard", "dtensor")
DISTRIBUTED_STRATEGY_ADMISSION_SETTINGS = (
    "dtensor.module_class",
    "dtensor.to_local_grad_placement",
    "dtensor.from_local_check",
    "dtensor.uneven_shard_handling",
    "dtensor.async_local_tensor_handling",
    "dtensor.higher_order_diff_status",
    "fsdp.hook_entry_points",
    "fsdp.hook_entry_policy",
    "fsdp.forward_prefetch",
    "fsdp.backward_prefetch",
    "fsdp.bypasses_hooks",
    "fsdp.bottom_up_order",
    "fsdp.mutated_modules",
    "fsdp.collectives",
)
DISTRIBUTED_COMMON_STRING_DOMAIN_FIELDS = (
    ("distributed.launch", ("single_process", "torchrun")),
    ("distributed.process_group_backend", ("gloo", "nccl", "ucc_when_available")),
    ("distributed.local_rank_binding", ("cuda_local_rank", "explicit_device_map")),
)
FSDP2_POLICY_FIELDS = (
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
)
DTENSOR_POLICY_FIELDS = (
    "dtensor.module_class",
    "dtensor.to_local_grad_placement",
    "dtensor.from_local_check",
    "dtensor.uneven_shard_handling",
    "dtensor.async_local_tensor_handling",
    "dtensor.redistribute_schedule",
)
TENSOR_PARALLEL_POLICY_FIELDS = (
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
)
SEQUENCE_PARALLEL_POLICY_FIELDS = (
    "sequence_parallel.enabled",
    "sequence_parallel.norm_modules",
    "sequence_parallel.output_placement_policy",
)
CONTEXT_PARALLEL_POLICY_FIELDS = (
    "context_parallel.enabled",
    "context_parallel.rotate_method",
    "context_parallel.sequence_dim",
)
LAYOUT_MODE_POLICY_FIELDS = {
    "tensor_parallel": ("tensor_parallel", TENSOR_PARALLEL_POLICY_FIELDS),
    "sequence_parallel": ("sequence_parallel", SEQUENCE_PARALLEL_POLICY_FIELDS),
    "context_parallel": ("context_parallel", CONTEXT_PARALLEL_POLICY_FIELDS),
}
COMMUNICATION_STRING_DOMAIN_FIELDS = (
    ("comm.overlap", ("none", "all_gather_overlap", "reduce_scatter_overlap", "both")),
    ("comm.prefetch", ("none", "forward", "backward", "both")),
)


class DistributedStrategyApplier(Protocol):
    """Apply one distributed strategy row to a module."""

    def __call__(
        self,
        module: torch.nn.Module,
        candidate: Candidate,
    ) -> torch.nn.Module:
        """Return the rank-local module used by the measured operation."""


class DistributedRankReporter(Protocol):
    """Report rank-local status, memory, settings, and compile timing."""

    def __call__(
        self,
        candidate: Candidate,
        samples: tuple[Measurement, ...],
    ) -> "DistributedRankReport":
        """Return rank reports for the measured candidate."""


@dataclasses.dataclass(frozen=True, slots=True)
class DistributedProcessGroupBindings:
    """Runtime inputs for process-group initialization."""

    init_process_group: Callable[..., Any]
    is_ucc_available: Callable[[], bool]
    init_method: str | None
    timeout: Any
    world_size: int
    rank: int
    store: Any
    pg_options: Any
    device_for_rank: Callable[[str, int], Any]


@dataclasses.dataclass(frozen=True, slots=True)
class DistributedMeshBindings:
    """Runtime inputs for DeviceMesh construction."""

    init_device_mesh: Callable[..., Any]
    device_type: str


@dataclasses.dataclass(frozen=True, slots=True)
class DistributedPlacementBindings:
    """Runtime inputs for DTensor placement construction."""

    replicate: Callable[[], Any]
    shard: Callable[[int], Any]
    partial: Callable[[str], Any]
    placement_specs: Mapping[str, Mapping[str, Any]]


@dataclasses.dataclass(frozen=True, slots=True)
class DistributedFSDPBindings:
    """Runtime inputs for FSDP2 and HSDP lowering."""

    fully_shard: Callable[..., torch.nn.Module]
    configure_forward_prefetch: Callable[[torch.nn.Module, str], None]
    configure_backward_prefetch: Callable[[torch.nn.Module, str], None]
    mixed_precision_policy: Callable[..., Any]
    offload_policy: Callable[[], Any]
    cpu_offload_policy: Callable[..., Any]
    data_parallel_mesh_dims: Callable[..., Any]
    shard_placement_fns: Mapping[str, Callable[..., Any]]
    ignored_params: Mapping[str, torch.nn.Parameter]
    cpu_offload_pin_memory: bool
    hsdp_replicate_mesh_dims: str | tuple[str, ...] | None


@dataclasses.dataclass(frozen=True, slots=True)
class DistributedTensorParallelBindings:
    """Runtime inputs for tensor-parallel lowering."""

    parallelize_module: Callable[..., torch.nn.Module]
    colwise_parallel: Callable[..., Any]
    rowwise_parallel: Callable[..., Any]
    prepare_module_input: Callable[..., Any]
    prepare_module_output: Callable[..., Any]
    loss_parallel: Callable[[], Any]
    module_paths_by_plan: Mapping[str, Mapping[str, str]]
    style_specs: Mapping[str, Mapping[str, Any]]
    prepare_input_specs: Mapping[str, Mapping[str, Any]]
    prepare_output_specs: Mapping[str, Mapping[str, Any]]
    src_data_rank: int


@dataclasses.dataclass(frozen=True, slots=True)
class DistributedSequenceParallelBindings:
    """Runtime inputs for sequence-parallel module styles."""

    sequence_parallel: Callable[..., Any]
    sequence_dim: int


@dataclasses.dataclass(frozen=True, slots=True)
class DistributedContextParallelBindings:
    """Runtime inputs for context-parallel execution."""

    context_parallel: Callable[..., Any]
    buffers: tuple[torch.Tensor, ...]
    no_restore_buffers: tuple[torch.Tensor, ...]


@dataclasses.dataclass(frozen=True, slots=True)
class DistributedCommunicationBindings:
    """Runtime inputs for distributed communication scheduling."""

    configure: Callable[[Mapping[str, Any]], Any]


@dataclasses.dataclass(frozen=True, slots=True)
class DistributedStrategyBindings:
    """Explicit runtime inputs for one distributed strategy row."""

    process_group: DistributedProcessGroupBindings | None
    mesh: DistributedMeshBindings
    placements: DistributedPlacementBindings
    fsdp: DistributedFSDPBindings | None
    tensor_parallel: DistributedTensorParallelBindings | None
    sequence_parallel: DistributedSequenceParallelBindings | None
    context_parallel: DistributedContextParallelBindings | None
    communication: DistributedCommunicationBindings | None
    hybrid_order: tuple[str, ...]


@dataclasses.dataclass(frozen=True, slots=True)
class _DistributedRedistribution:
    settings: Mapping[str, Any]
    mesh: Any
    placements: Mapping[str, Any]

    def before_forward_params(self, params: ParameterTree) -> ParameterTree:
        return self._scheduled_tree(
            params,
            "before_forward",
            "dtensor.params_placement",
        )

    def before_forward_batch(self, batch: Batch) -> Batch:
        return self._scheduled_batch_slots(batch, "before_forward", ("logits",))

    def before_forward_vector(self, vector: TensorTree) -> TensorTree:
        return self._scheduled_tree(
            vector,
            "before_forward",
            "dtensor.vector_placement",
        )

    def before_backward_vector(self, vector: TensorTree) -> TensorTree:
        return self._scheduled_tree(
            vector,
            "before_backward",
            "dtensor.cotangent_placement",
        )

    def between_operator_parts(self, tree: TensorTree) -> TensorTree:
        return self._scheduled_tree(
            tree,
            "between_operator_parts",
            "dtensor.tangent_placement",
        )

    def before_output(self, output: TensorTree) -> TensorTree:
        return self._scheduled_tree(
            output,
            "before_output",
            "dtensor.output_placement",
        )

    def redistribute(self, value: Any, placement_key: str) -> Any:
        if placement_key not in self.placements:
            message = f"dtensor redistribution requires {placement_key}"
            raise MaterializationError(message)

        if not hasattr(value, "redistribute"):
            message = f"dtensor redistribution requires a DTensor for {placement_key}"
            raise MaterializationError(message)

        return redistribute_dtensor(
            value,
            device_mesh=self.mesh,
            placements=(self.placements[placement_key],),
            async_op=False,
            forward_dtype=None,
            backward_dtype=None,
        )

    def _batch_slots(self, batch: Batch, slots: tuple[str, ...]) -> Batch:
        result = dict(batch)

        for slot in slots:
            if slot in result:
                result[slot] = self.redistribute(
                    result[slot], "dtensor.logits_placement"
                )

        return result

    def _scheduled_tree(
        self,
        value: Any,
        schedule: str,
        placement_key: str,
    ) -> Any:
        if self.settings.get("dtensor.redistribute_schedule") != schedule:
            return value

        return _distributed_tree_map(
            lambda item: self.redistribute(item, placement_key),
            value,
        )

    def _scheduled_batch_slots(
        self,
        batch: Batch,
        schedule: str,
        slots: tuple[str, ...],
    ) -> Batch:
        if self.settings.get("dtensor.redistribute_schedule") != schedule:
            return batch

        return self._batch_slots(batch, slots)


def _distributed_tree_map(fn: Callable[[Any], Any], value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _distributed_tree_map(fn, child) for key, child in value.items()}

    if isinstance(value, tuple):
        return tuple(_distributed_tree_map(fn, child) for child in value)

    return fn(value)


class _BoundDistributedStrategyApplier:
    def __init__(self, bindings: DistributedStrategyBindings) -> None:
        self.bindings = bindings

    def __call__(
        self,
        module: torch.nn.Module,
        candidate: Candidate,
    ) -> torch.nn.Module:
        return _apply_distributed_strategy(module, candidate.settings, self.bindings)


def distributed_strategy_applier(
    bindings: DistributedStrategyBindings,
) -> DistributedStrategyApplier:
    """Return an applier that lowers distributed row settings."""
    return _BoundDistributedStrategyApplier(bindings)


def build_device_mesh(
    init_device_mesh: Callable[..., Any],
    *,
    device_type: str,
    mesh_shape: tuple[int, ...],
    mesh_dim_names: tuple[str, ...],
) -> Any:
    """Build a PyTorch DeviceMesh.

    Returns:
        DeviceMesh.
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
        Process group initialization result.
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
    module: torch.nn.Module | list[torch.nn.Module],
    *,
    mesh: Any,
    reshard_after_forward: bool | int | None,
    shard_placement_fn: Callable[..., Any] | None,
    mp_policy: Any,
    offload_policy: Any,
    ignored_params: Sequence[torch.nn.Parameter],
    dp_mesh_dims: Any,
) -> torch.nn.Module:
    """Apply PyTorch FSDP2 to a module or declared module group.

    Returns:
        FSDP-wrapped module.
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
        FSDP-wrapped module group.
    """
    return apply_fsdp2(
        fully_shard,
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
        Mixed-precision policy.
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
        Data-parallel mesh dimensions.
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
        Tensor-parallel module.
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
        Column-wise parallel style.
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
        Row-wise parallel style.
    """
    return rowwise_parallel(
        input_layouts=input_layouts,
        output_layouts=output_layouts,
        use_local_output=use_local_output,
    )


TENSOR_PARALLEL_STYLE_BUILDERS = {
    "colwise": build_colwise_parallel,
    "rowwise": build_rowwise_parallel,
}


def build_sequence_parallel(
    sequence_parallel: Callable[..., Any],
    *,
    sequence_dim: int,
    use_local_output: bool,
) -> Any:
    """Build a PyTorch sequence-parallel style.

    Returns:
        Sequence-parallel style.
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
        Input preparation style.
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
        Output preparation style.
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
        Redistributed DTensor.
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
    rotate_method: str,
    buffers: Sequence[torch.Tensor],
    buffer_seq_dims: Sequence[int],
    no_restore_buffers: Sequence[torch.Tensor],
) -> Any:
    """Apply PyTorch context parallelism.

    Returns:
        Context-parallel application result.
    """
    return context_parallel(
        mesh,
        rotate_method=rotate_method,
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
        Collective result.
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
        Collective result.
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
        Collective result.
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
        Wait result.
    """
    return work.wait()


def _apply_distributed_strategy(
    module: torch.nn.Module,
    settings: Mapping[str, Any],
    bindings: DistributedStrategyBindings,
) -> torch.nn.Module:
    _initialize_distributed_process_group(settings, bindings)
    mesh = _build_declared_mesh(settings, bindings.mesh)
    placements = _build_declared_placements(settings, bindings.placements)
    _configure_distributed_communication(settings, bindings)
    strategy = _required_string_setting(settings, "distributed.strategy")

    return _apply_declared_distributed_stage(
        strategy,
        module,
        settings,
        mesh,
        placements,
        bindings,
        appliers=DISTRIBUTED_STRATEGY_APPLIERS,
        error_prefix="unsupported distributed strategy",
    )


def _initialize_distributed_process_group(
    settings: Mapping[str, Any],
    bindings: DistributedStrategyBindings,
) -> None:
    launch = _required_string_setting(settings, "distributed.launch")

    if launch == "single_process":
        return

    if launch != "torchrun":
        message = f"unsupported distributed launch: {launch}"
        raise MaterializationError(message)

    process_group = _required_binding(
        bindings.process_group,
        "distributed.launch=torchrun",
    )
    backend = resolve_process_group_backend(
        _required_string_setting(settings, "distributed.process_group_backend"),
        is_ucc_available=process_group.is_ucc_available,
    )
    device_id = process_group.device_for_rank(
        _required_string_setting(settings, "distributed.local_rank_binding"),
        process_group.rank,
    )
    initialize_process_group(
        process_group.init_process_group,
        backend=backend,
        init_method=process_group.init_method,
        timeout=process_group.timeout,
        world_size=process_group.world_size,
        rank=process_group.rank,
        store=process_group.store,
        pg_options=process_group.pg_options,
        device_id=device_id,
    )


def _build_declared_mesh(
    settings: Mapping[str, Any],
    bindings: DistributedMeshBindings,
) -> Any:
    return build_device_mesh(
        bindings.init_device_mesh,
        device_type=bindings.device_type,
        mesh_shape=_tuple_of_ints(settings, "distributed.mesh_shape"),
        mesh_dim_names=_tuple_of_strings(settings, "distributed.mesh_dim_names"),
    )


def _build_declared_placements(
    settings: Mapping[str, Any],
    bindings: DistributedPlacementBindings,
) -> dict[str, Any]:
    placements = {}

    for key in DTENSOR_PLACEMENT_KEYS:
        if key in settings:
            spec = bindings.placement_specs.get(key, {})
            placements[key] = build_dtensor_placement(
                bindings.replicate,
                bindings.shard,
                bindings.partial,
                placement=_required_string_setting(settings, key),
                shard_dim=_optional_int(spec.get("shard_dim"), f"{key}.shard_dim"),
                reduce_op=_optional_string(spec.get("reduce_op"), f"{key}.reduce_op"),
            )

    return placements


def _configure_distributed_communication(
    settings: Mapping[str, Any],
    bindings: DistributedStrategyBindings,
) -> None:
    communication_settings = {
        key: settings[key] for key in settings if key.startswith("comm.")
    }

    if not communication_settings:
        return

    communication = _required_binding(
        bindings.communication,
        "comm.* distributed scheduling",
    )
    communication.configure(communication_settings)


def _apply_declared_fsdp(
    module: torch.nn.Module,
    settings: Mapping[str, Any],
    mesh: Any,
    bindings: DistributedStrategyBindings,
) -> torch.nn.Module:
    fsdp = _required_binding(bindings.fsdp, "FSDP2 distributed strategy")
    mp_policy = build_fsdp_mixed_precision_policy(
        fsdp.mixed_precision_policy,
        param_dtype=_distributed_dtype_setting(settings, "fsdp.mp_policy.param_dtype"),
        reduce_dtype=_distributed_dtype_setting(
            settings, "fsdp.mp_policy.reduce_dtype"
        ),
        output_dtype=_distributed_dtype_setting(
            settings, "fsdp.mp_policy.output_dtype"
        ),
        cast_forward_inputs=_bool_string_setting(
            settings,
            "fsdp.mp_policy.cast_forward_inputs",
        ),
    )
    offload_policy = build_fsdp_offload_policy(
        fsdp.offload_policy,
        fsdp.cpu_offload_policy,
        offload=_required_string_setting(settings, "fsdp.offload_policy"),
        pin_memory=fsdp.cpu_offload_pin_memory,
    )
    dp_mesh_dims = build_fsdp_dp_mesh_dims(
        fsdp.data_parallel_mesh_dims,
        shard=_required_declared_setting(settings, "fsdp.dp_mesh_dims"),
        replicate=_fsdp_replicate_mesh_dims(settings, fsdp),
    )
    ignored_params = _declared_ignored_params(settings, fsdp)
    reshard_after_forward = _fsdp_reshard_after_forward(settings, fsdp)
    shard_placement_fn = _fsdp_shard_placement_fn(settings, fsdp)
    wrap_granularity = _required_string_setting(settings, "fsdp.wrap_granularity")

    if wrap_granularity == "root":
        target = module
    else:
        target = list(
            named_modules_for_distributed_wrap(
                module,
                _fsdp_hook_module_names(settings),
            )
        )

    sharded = apply_fsdp2(
        fsdp.fully_shard,
        target,
        mesh=mesh,
        reshard_after_forward=reshard_after_forward,
        shard_placement_fn=shard_placement_fn,
        mp_policy=mp_policy,
        offload_policy=offload_policy,
        ignored_params=ignored_params,
        dp_mesh_dims=dp_mesh_dims,
    )

    return _configure_fsdp_prefetch(sharded, settings, fsdp)


def _apply_declared_tensor_parallel(
    module: torch.nn.Module,
    settings: Mapping[str, Any],
    mesh: Any,
    placements: Mapping[str, Any],
    bindings: DistributedStrategyBindings,
) -> torch.nn.Module:
    tensor_parallel = _required_binding(
        bindings.tensor_parallel,
        "tensor-parallel distributed strategy",
    )
    plan = _tensor_parallel_plan(settings, placements, tensor_parallel)

    return _apply_tensor_parallel_plan(module, mesh, tensor_parallel, plan)


def _apply_declared_sequence_parallel(
    module: torch.nn.Module,
    settings: Mapping[str, Any],
    mesh: Any,
    placements: Mapping[str, Any],
    bindings: DistributedStrategyBindings,
) -> torch.nn.Module:
    tensor_parallel = _required_binding(
        bindings.tensor_parallel,
        "sequence-parallel tensor plan",
    )
    sequence_parallel = _required_binding(
        bindings.sequence_parallel,
        "sequence-parallel distributed strategy",
    )
    plan = _tensor_parallel_plan(settings, placements, tensor_parallel)

    if _required_string_setting(settings, "sequence_parallel.enabled") != "true":
        message = "sequence_parallel strategy requires sequence_parallel.enabled=true"
        raise MaterializationError(message)

    sequence_style = build_sequence_parallel(
        sequence_parallel.sequence_parallel,
        sequence_dim=sequence_parallel.sequence_dim,
        use_local_output=_sequence_parallel_use_local_output(settings),
    )

    for module_name in _tuple_of_strings(settings, "sequence_parallel.norm_modules"):
        plan[module_name] = sequence_style

    return _apply_tensor_parallel_plan(module, mesh, tensor_parallel, plan)


def _apply_tensor_parallel_plan(
    module: torch.nn.Module,
    mesh: Any,
    bindings: DistributedTensorParallelBindings,
    plan: Mapping[str, Any],
) -> torch.nn.Module:
    return apply_tensor_parallel(
        bindings.parallelize_module,
        module,
        device_mesh=mesh,
        parallelize_plan=plan,
        src_data_rank=bindings.src_data_rank,
    )


def _sequence_parallel_use_local_output(settings: Mapping[str, Any]) -> bool:
    policy = _required_string_setting(
        settings,
        "sequence_parallel.output_placement_policy",
    )

    if policy == "preserve_sequence_shard":
        return False

    if policy == "redistribute_to_declared_output":
        return True

    message = f"unsupported sequence-parallel output policy: {policy}"
    raise MaterializationError(message)


def _apply_declared_context_parallel(
    module: torch.nn.Module,
    settings: Mapping[str, Any],
    mesh: Any,
    placements: Mapping[str, Any],
    bindings: DistributedStrategyBindings,
) -> torch.nn.Module:
    context_parallel = _required_binding(
        bindings.context_parallel,
        "context-parallel distributed strategy",
    )

    if _required_string_setting(settings, "context_parallel.enabled") != "true":
        message = "context_parallel strategy requires context_parallel.enabled=true"
        raise MaterializationError(message)

    sequence_dim = _required_int_setting(settings, "context_parallel.sequence_dim")
    apply_context_parallel(
        context_parallel.context_parallel,
        mesh,
        rotate_method=_required_string_setting(
            settings,
            "context_parallel.rotate_method",
        ),
        buffers=context_parallel.buffers,
        buffer_seq_dims=tuple(sequence_dim for _ in context_parallel.buffers),
        no_restore_buffers=context_parallel.no_restore_buffers,
    )

    return _apply_declared_tensor_parallel(
        module,
        settings,
        mesh,
        placements,
        bindings,
    )


def _apply_declared_hybrid(
    module: torch.nn.Module,
    settings: Mapping[str, Any],
    mesh: Any,
    placements: Mapping[str, Any],
    bindings: DistributedStrategyBindings,
) -> torch.nn.Module:
    result = module

    for stage in bindings.hybrid_order:
        result = _apply_declared_distributed_stage(
            stage,
            result,
            settings,
            mesh,
            placements,
            bindings,
            appliers=HYBRID_STAGE_APPLIERS,
            error_prefix="unsupported hybrid distributed stage",
        )

    if not bindings.hybrid_order:
        message = "hybrid distributed strategy requires a non-empty hybrid_order"
        raise MaterializationError(message)

    return result


def _apply_declared_distributed_stage(
    stage: str,
    module: torch.nn.Module,
    settings: Mapping[str, Any],
    mesh: Any,
    placements: Mapping[str, Any],
    bindings: DistributedStrategyBindings,
    *,
    appliers: Mapping[str, Callable[..., torch.nn.Module]],
    error_prefix: str,
) -> torch.nn.Module:
    applier = appliers.get(stage)

    if applier is None:
        message = f"{error_prefix}: {stage}"
        raise MaterializationError(message)

    return applier(module, settings, mesh, placements, bindings)


def _apply_single_gpu_stage(
    module: torch.nn.Module,
    settings: Mapping[str, Any],
    mesh: Any,
    placements: Mapping[str, Any],
    bindings: DistributedStrategyBindings,
) -> torch.nn.Module:
    _ = settings, mesh, placements, bindings

    return module


def _apply_fsdp_stage(
    module: torch.nn.Module,
    settings: Mapping[str, Any],
    mesh: Any,
    placements: Mapping[str, Any],
    bindings: DistributedStrategyBindings,
) -> torch.nn.Module:
    _ = placements

    return _apply_declared_fsdp(module, settings, mesh, bindings)


DISTRIBUTED_STRATEGY_APPLIERS = {
    "single_gpu": _apply_single_gpu_stage,
    "fsdp2": _apply_fsdp_stage,
    "hsdp": _apply_fsdp_stage,
    "tensor_parallel": _apply_declared_tensor_parallel,
    "sequence_parallel": _apply_declared_sequence_parallel,
    "context_parallel": _apply_declared_context_parallel,
    "hybrid": _apply_declared_hybrid,
}
HYBRID_STAGE_APPLIERS = {
    "tensor_parallel": _apply_declared_tensor_parallel,
    "fsdp2": _apply_fsdp_stage,
    "sequence_parallel": _apply_declared_sequence_parallel,
    "context_parallel": _apply_declared_context_parallel,
}


def _tensor_parallel_plan(
    settings: Mapping[str, Any],
    placements: Mapping[str, Any],
    bindings: DistributedTensorParallelBindings,
) -> dict[str, Any]:
    plan_id = _required_string_setting(settings, "tp.plan")
    module_paths = bindings.module_paths_by_plan.get(plan_id)

    if module_paths is None:
        message = f"tp.plan has no declared module path map: {plan_id}"
        raise MaterializationError(message)

    plan = {}

    for key in TENSOR_PARALLEL_STYLE_KEYS:
        value = _required_string_setting(settings, key)

        if value == "replicated":
            continue

        module_path = _required_module_path(module_paths, key)
        plan[module_path] = _tensor_parallel_style(
            key,
            value,
            placements,
            bindings,
        )

    _add_prepare_module_input_style(settings, plan, placements, bindings)
    _add_prepare_module_output_style(settings, plan, placements, bindings)

    return plan


def _tensor_parallel_style(
    key: str,
    value: str,
    placements: Mapping[str, Any],
    bindings: DistributedTensorParallelBindings,
) -> Any:
    if value == "vocab_sharded":
        value = "colwise"

    spec = _required_mapping(bindings.style_specs, key)
    builder = TENSOR_PARALLEL_STYLE_BUILDERS.get(value)
    constructor = {
        "colwise": bindings.colwise_parallel,
        "rowwise": bindings.rowwise_parallel,
    }.get(value)

    if builder is not None and constructor is not None:
        return builder(
            constructor,
            input_layouts=_placement_spec_value(
                spec,
                "input_layouts",
                placements,
                key,
            ),
            output_layouts=_placement_spec_value(
                spec,
                "output_layouts",
                placements,
                key,
            ),
            use_local_output=_required_bool_value(
                spec.get("use_local_output"),
                f"{key}.use_local_output",
            ),
        )

    message = f"unsupported tensor-parallel style for {key}: {value}"
    raise MaterializationError(message)


def _add_prepare_module_input_style(
    settings: Mapping[str, Any],
    plan: dict[str, Any],
    placements: Mapping[str, Any],
    bindings: DistributedTensorParallelBindings,
) -> None:
    _add_prepare_module_style(
        settings,
        plan,
        placements,
        "tp.prepare_module_input",
        bindings.prepare_input_specs,
        build_prepare_module_input,
        bindings.prepare_module_input,
        ("input_layouts", "desired_input_layouts"),
        ("input_kwarg_layouts", "desired_input_kwarg_layouts"),
    )


def _add_prepare_module_output_style(
    settings: Mapping[str, Any],
    plan: dict[str, Any],
    placements: Mapping[str, Any],
    bindings: DistributedTensorParallelBindings,
) -> None:
    _add_prepare_module_style(
        settings,
        plan,
        placements,
        "tp.prepare_module_output",
        bindings.prepare_output_specs,
        build_prepare_module_output,
        bindings.prepare_module_output,
        ("output_layouts", "desired_output_layouts"),
        (),
    )


def _add_prepare_module_style(
    settings: Mapping[str, Any],
    plan: dict[str, Any],
    placements: Mapping[str, Any],
    setting_key: str,
    specs: Mapping[str, Mapping[str, Any]],
    style_builder: Callable[..., Any],
    constructor: Callable[..., Any],
    placement_fields: tuple[str, ...],
    placement_mapping_fields: tuple[str, ...],
) -> None:
    value = _required_string_setting(settings, setting_key)
    spec = _required_mapping(specs, value)
    module_path = _required_string_value(
        spec.get("module_path"),
        f"{setting_key}.module_path",
    )
    kwargs = {
        field: _placement_spec_value(spec, field, placements, setting_key)
        for field in placement_fields
    }

    for field in placement_mapping_fields:
        kwargs[field] = _placement_spec_mapping(spec, field, placements, setting_key)

    kwargs["use_local_output"] = _required_bool_value(
        spec.get("use_local_output"),
        f"{setting_key}.use_local_output",
    )
    plan[module_path] = style_builder(constructor, **kwargs)


def _placement_spec_value(
    spec: Mapping[str, Any],
    field: str,
    placements: Mapping[str, Any],
    context: str,
) -> Any:
    placement_key = _required_string_value(spec.get(field), f"{context}.{field}")

    if placement_key not in placements:
        message = f"{context}.{field} references undeclared placement: {placement_key}"
        raise MaterializationError(message)

    return placements[placement_key]


def _placement_spec_mapping(
    spec: Mapping[str, Any],
    field: str,
    placements: Mapping[str, Any],
    context: str,
) -> dict[str, Any]:
    value = _required_mapping_value(spec.get(field), f"{context}.{field}")
    result = {}

    for key, placement_key in value.items():
        if not isinstance(key, str) or not key:
            message = f"{context}.{field} keys must be non-empty strings"
            raise MaterializationError(message)

        if not isinstance(placement_key, str) or placement_key not in placements:
            message = (
                f"{context}.{field} references undeclared placement: {placement_key}"
            )
            raise MaterializationError(message)

        result[key] = placements[placement_key]

    return result


def _required_module_path(module_paths: Mapping[str, str], key: str) -> str:
    value = module_paths.get(key)

    if not isinstance(value, str) or not value:
        message = f"tp.plan missing module path for {key}"
        raise MaterializationError(message)

    return value


def _fsdp_replicate_mesh_dims(
    settings: Mapping[str, Any],
    bindings: DistributedFSDPBindings,
) -> str | tuple[str, ...] | None:
    strategy = _required_string_setting(settings, "distributed.strategy")

    if strategy != "hsdp":
        return None

    if bindings.hsdp_replicate_mesh_dims is None:
        message = "hsdp requires declared replicated mesh dimensions"
        raise MaterializationError(message)

    return bindings.hsdp_replicate_mesh_dims


def _declared_ignored_params(
    settings: Mapping[str, Any],
    bindings: DistributedFSDPBindings,
) -> tuple[torch.nn.Parameter, ...]:
    names = _tuple_of_strings(settings, "fsdp.ignored_params")
    params = []

    for name in names:
        param = bindings.ignored_params.get(name)

        if param is None:
            message = f"fsdp.ignored_params has no declared parameter: {name}"
            raise MaterializationError(message)

        params.append(param)

    return tuple(params)


def _configure_fsdp_prefetch(
    module: torch.nn.Module,
    settings: Mapping[str, Any],
    bindings: DistributedFSDPBindings,
) -> torch.nn.Module:
    for key, enabled_value, configure, label in (
        (
            "fsdp.forward_prefetch",
            "next-forward",
            bindings.configure_forward_prefetch,
            "forward",
        ),
        (
            "fsdp.backward_prefetch",
            "backward-pre",
            bindings.configure_backward_prefetch,
            "backward",
        ),
    ):
        value = _required_string_setting(settings, key)

        if value == enabled_value:
            configure(module, value)
        elif value != "disabled":
            message = f"unsupported FSDP {label} prefetch setting: {value}"
            raise MaterializationError(message)

    return module


def _fsdp_reshard_after_forward(
    settings: Mapping[str, Any],
    _: DistributedFSDPBindings,
) -> bool | int | None:
    key = "fsdp.reshard_after_forward"
    value = settings.get(key)

    if value == "true":
        return True

    if value == "false":
        return False

    if isinstance(value, int) and not isinstance(value, bool) and value > 0:
        return value

    message = f"unsupported FSDP2 reshard setting: {value}"
    raise MaterializationError(message)


def _fsdp_shard_placement_fn(
    settings: Mapping[str, Any],
    bindings: DistributedFSDPBindings,
) -> Callable[..., Any] | None:
    value = _required_string_setting(settings, "fsdp.shard_placement_fn")

    if value == "none":
        return None

    shard_placement_fn = bindings.shard_placement_fns.get(value)

    if shard_placement_fn is None:
        message = f"fsdp.shard_placement_fn has no declared function: {value}"
        raise MaterializationError(message)

    return shard_placement_fn


def _fsdp_hook_module_names(settings: Mapping[str, Any]) -> tuple[str, ...]:
    names = []

    for hook in _tuple_of_strings(settings, "fsdp.hook_entry_points"):
        if hook == "root.forward":
            names.append("")
        elif hook.endswith(".forward"):
            names.append(hook.removesuffix(".forward"))
        else:
            names.append(hook)

    return tuple(names)


def _distributed_loss_parallel(
    settings: Mapping[str, Any],
    operation: CandidateOperation,
    loss_parallel: Callable[[], Any] | None,
) -> CandidateOperation:
    if settings.get("tp.loss_parallel") != "true":
        return operation

    if loss_parallel is None:
        message = "tp.loss_parallel=true requires a loss_parallel binding"
        raise MaterializationError(message)

    def wrapped() -> TensorTree:
        return run_with_loss_parallel(loss_parallel, operation)

    return wrapped


def _strategy_applier_and_loss_parallel(
    *,
    strategy_applier: DistributedStrategyApplier | None,
    strategy_bindings: DistributedStrategyBindings | None,
    loss_parallel: Callable[[], Any] | None,
) -> tuple[DistributedStrategyApplier, Callable[[], Any] | None]:
    if strategy_applier is not None and strategy_bindings is not None:
        message = "distributed runtime accepts strategy_applier or strategy_bindings"
        raise MaterializationError(message)

    if strategy_applier is None and strategy_bindings is None:
        message = "distributed runtime requires strategy_applier or strategy_bindings"
        raise MaterializationError(message)

    if strategy_bindings is None:
        if strategy_applier is None:
            message = "distributed runtime requires strategy_applier"
            raise MaterializationError(message)

        return strategy_applier, loss_parallel

    if loss_parallel is not None:
        message = "loss_parallel is read from strategy_bindings"
        raise MaterializationError(message)

    bound_loss_parallel = None

    if strategy_bindings.tensor_parallel is not None:
        bound_loss_parallel = strategy_bindings.tensor_parallel.loss_parallel

    return distributed_strategy_applier(strategy_bindings), bound_loss_parallel


def _required_string_setting(settings: Mapping[str, Any], key: str) -> str:
    return _required_string_value(settings.get(key), key)


def _required_declared_setting(settings: Mapping[str, Any], key: str) -> Any:
    if key not in settings:
        message = f"{key} must be declared"
        raise MaterializationError(message)

    return settings[key]


def _required_int_setting(settings: Mapping[str, Any], key: str) -> int:
    return _required_checked_value(
        settings.get(key),
        key,
        _positive_int_value,
        "a positive integer",
    )


def _tuple_of_ints(settings: Mapping[str, Any], key: str) -> tuple[int, ...]:
    return _required_checked_value(
        settings.get(key),
        key,
        _positive_int_tuple_value,
        "a non-empty tuple of positive integers",
    )


def _tuple_of_strings(settings: Mapping[str, Any], key: str) -> tuple[str, ...]:
    return _required_checked_value(
        settings.get(key),
        key,
        _string_tuple_value,
        "a tuple of strings",
    )


def _required_binding(value: Any, name: str) -> Any:
    if value is None:
        message = f"{name} requires runtime bindings"
        raise MaterializationError(message)

    return value


def _required_mapping(
    values: Mapping[str, Mapping[str, Any]],
    key: str,
) -> Mapping[str, Any]:
    value = values.get(key)

    if value is None:
        message = f"{key} requires a declared mapping"
        raise MaterializationError(message)

    return value


def _required_mapping_value(value: Any, name: str) -> Mapping[str, Any]:
    return _required_checked_value(value, name, _mapping_value, "a mapping")


def _required_string_value(value: Any, name: str) -> str:
    return _required_checked_value(
        value,
        name,
        _non_empty_string_value,
        "a non-empty string",
    )


def _required_bool_value(value: Any, name: str) -> bool:
    return _required_checked_value(value, name, _bool_value, "a bool")


def _optional_int(value: Any, name: str) -> int | None:
    if value is None:
        return None

    return _required_checked_value(value, name, _int_value, "an integer")


def _optional_string(value: Any, name: str) -> str | None:
    if value is None:
        return None

    return _required_string_value(value, name)


def _required_checked_value(
    value: Any,
    name: str,
    validate: Callable[[Any], bool],
    value_message: str,
) -> Any:
    if not validate(value):
        message = f"{name} must be {value_message}"
        raise MaterializationError(message)

    return value


def _bool_string_setting(settings: Mapping[str, Any], key: str) -> bool:
    value = _required_string_setting(settings, key)

    if value == "true":
        return True

    if value == "false":
        return False

    message = f"{key} must be false or true"
    raise MaterializationError(message)


def _distributed_dtype_setting(
    settings: Mapping[str, Any],
    key: str,
) -> torch.dtype | None:
    value = settings.get(key)

    if value is None:
        return None

    if value == "fp32":
        return torch.float32

    if value == "bf16":
        return torch.bfloat16

    if value == "fp16":
        return torch.float16

    message = f"{key} is unsupported by distributed runtime: {value}"
    raise MaterializationError(message)


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
class RankCompileTiming:
    """Compile timing reported by one distributed rank."""

    rank: int
    compile_time_seconds: float
    steady_elapsed_seconds: float
    recompile_count: int

    def to_record(self) -> dict[str, Any]:
        """Return JSON-compatible compile timing."""
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True, slots=True)
class DistributedRankReport:
    """Rank reports collected after a measured distributed row."""

    rank_statuses: tuple[RankStatus, ...]
    rank_memory_samples: tuple[Measurement, ...]
    rank_selected_settings: tuple[RankSelectedSettings, ...]
    rank_compile_timings: tuple[RankCompileTiming, ...] = ()


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


def distributed_manifest_axis_descriptors() -> tuple[AxisDescriptor, ...]:
    """Return distributed descriptors for the full package manifest."""
    return distributed_axis_descriptors(
        DISTRIBUTED_STRATEGIES,
        policy=_distributed_manifest_policy(),
    )


def _distributed_manifest_policy() -> DistributedAdmissionPolicy:
    return DistributedAdmissionPolicy(
        candidate_generator_version=PACKAGE_VERSION,
        device_mesh={"shape": (2,), "names": ("data",)},
        rank_count=2,
        per_rank_placements=(
            {"rank": 0, "device": "cuda:0"},
            {"rank": 1, "device": "cuda:1"},
        ),
        communication={"backend": "nccl"},
        fsdp2={
            "allowed_fsdp.hook_entry_policy": ("root-forward", "layer-forward"),
            "allowed_fsdp.wrap_granularity": (
                "root",
                "transformer_block",
                "block_group",
            ),
            "allowed_fsdp.forward_prefetch": ("disabled", "next-forward"),
            "allowed_fsdp.backward_prefetch": ("disabled", "backward-pre"),
            "allowed_fsdp.reshard_after_forward": ("false", "true", 1),
            "allowed_fsdp.shard_placement_fn": ("none", "declared_fn"),
            "allowed_fsdp.mp_policy.param_dtype": ("fp32", "bf16", "fp16"),
            "allowed_fsdp.mp_policy.reduce_dtype": ("fp32", "bf16", "fp16"),
            "allowed_fsdp.mp_policy.output_dtype": ("fp32", "bf16", "fp16"),
            "allowed_fsdp.mp_policy.cast_forward_inputs": ("false", "true"),
            "allowed_fsdp.offload_policy": ("none", "cpu"),
        },
        dtensor={
            "allowed_dtensor.module_class": ("torch.nn.Linear",),
            "allowed_dtensor.to_local_grad_placement": ("redistribute",),
            "allowed_dtensor.from_local_check": ("strict",),
            "allowed_dtensor.uneven_shard_handling": ("reject",),
            "allowed_dtensor.async_local_tensor_handling": ("sync",),
            "allowed_dtensor.redistribute_schedule": (
                "none",
                "before_forward",
                "before_backward",
                "between_operator_parts",
                "before_output",
            ),
            "allowed_higher_order_diff_status": ("supported", "rejected"),
        },
        tensor_parallel={
            "allowed_tp.plan": ("registered",),
            "allowed_tp.qkv_projection": ("colwise", "rowwise", "replicated"),
            "allowed_tp.output_projection": ("rowwise", "colwise", "replicated"),
            "allowed_tp.mlp_up_gate": ("colwise", "rowwise", "replicated"),
            "allowed_tp.mlp_down": ("rowwise", "colwise", "replicated"),
            "allowed_tp.embedding": ("replicated", "rowwise", "colwise"),
            "allowed_tp.lm_head": ("replicated", "vocab_sharded"),
            "allowed_tp.prepare_module_input": ("registered",),
            "allowed_tp.prepare_module_output": ("registered",),
            "allowed_tp.loss_parallel": ("false", "true"),
        },
        sequence_parallel={
            "allowed_sequence_parallel.enabled": ("false", "true"),
            "allowed_sequence_parallel.norm_modules": ("registered",),
            "allowed_sequence_parallel.output_placement_policy": (
                "preserve_sequence_shard",
                "redistribute_to_declared_output",
            ),
        },
        context_parallel={
            "allowed_context_parallel.enabled": ("false", "true"),
            "allowed_context_parallel.rotate_method": ("all_gather", "all_to_all"),
            "allowed_context_parallel.sequence_dim": (0, 1),
        },
    )


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
        optional_settings_keys=DISTRIBUTED_STRATEGY_ADMISSION_SETTINGS,
        adapter_id="vptune.distributed",
        adapter_version=PACKAGE_VERSION,
        admission_rule=lambda candidate: admit_distributed_candidate(
            candidate,
            policy=policy,
        ),
        identity=policy.signature(),
    )


@dataclasses.dataclass(frozen=True, slots=True)
class DistributedSpace:
    """Public search-space component for distributed strategy axes."""

    strategies: tuple[str, ...]
    policy: DistributedAdmissionPolicy

    def __post_init__(self) -> None:
        """Validate distributed strategies.

        Raises:
            AdmissionError: If a strategy is invalid.
        """
        if not self.strategies:
            message = "distributed strategy must be nonempty"
            raise AdmissionError(message)

        unsupported = tuple(
            strategy
            for strategy in self.strategies
            if strategy not in DISTRIBUTED_STRATEGIES
        )

        if unsupported:
            message = f"unsupported distributed strategies: {unsupported}"
            raise AdmissionError(message)

        if len(set(self.strategies)) != len(self.strategies):
            message = "distributed strategies must be unique"
            raise AdmissionError(message)

        object.__setattr__(self, "strategies", tuple(self.strategies))

    def axes_for(self, operator: object) -> Mapping[str, Sequence[Any]]:
        """Return distributed axes for a public operator."""
        _ = operator

        return {"distributed.strategy": self.strategies}

    def axis_descriptors(self) -> tuple[AxisDescriptor, ...]:
        """Return adapter descriptors registered by this component."""
        return distributed_axis_descriptors(self.strategies, policy=self.policy)

    def settings_for(self, operator: object) -> Mapping[str, Any]:
        """Return fixed distributed settings for generated rows."""
        _ = operator

        return _distributed_space_common_settings(self.policy)


def space(
    *,
    strategy: Sequence[str],
    policy: DistributedAdmissionPolicy | None = None,
) -> DistributedSpace:
    """Build a public search-space component for distributed strategies.

    Returns:
        Distributed search-space component.
    """
    distributed_policy = _distributed_manifest_policy() if policy is None else policy

    return DistributedSpace(tuple(strategy), distributed_policy)


def _distributed_space_common_settings(
    policy: DistributedAdmissionPolicy,
) -> Mapping[str, Any]:
    shape = policy.device_mesh.get("shape")
    names = policy.device_mesh.get("names")
    backend = policy.communication.get("backend")

    if not isinstance(shape, tuple) or not shape:
        message = "distributed policy device_mesh.shape must be a tuple"
        raise AdmissionError(message)

    if not isinstance(names, tuple) or not names:
        message = "distributed policy device_mesh.names must be a tuple"
        raise AdmissionError(message)

    if not isinstance(backend, str) or not backend:
        message = "distributed policy communication.backend must be a string"
        raise AdmissionError(message)

    return {
        "distributed.launch": "torchrun" if policy.rank_count > 1 else "single_process",
        "distributed.process_group_backend": backend,
        "distributed.local_rank_binding": "cuda_local_rank",
        "distributed.mesh_shape": shape,
        "distributed.mesh_dim_names": names,
    }


def _distributed_axis_row(
    axis_key: str,
    allowed_values: tuple[Any, ...],
    rule_factory: Callable[[str], Callable[[Candidate], tuple[bool, str | None]]]
    | None = None,
) -> tuple[Any, ...]:
    return axis_key, allowed_values, rule_factory


def _distributed_axis_rows(
    axis_keys: tuple[str, ...],
    allowed_values: tuple[Any, ...],
    rule_factory: Callable[[str], Callable[[Candidate], tuple[bool, str | None]]]
    | None = None,
) -> tuple[tuple[Any, ...], ...]:
    return tuple(
        _distributed_axis_row(axis_key, allowed_values, rule_factory)
        for axis_key in axis_keys
    )


def _distributed_value_axis(
    value_message: str,
    validate: Callable[[Any], bool],
) -> Callable[[str], Callable[[Candidate], tuple[bool, str | None]]]:
    def build(axis_key: str) -> Callable[[Candidate], tuple[bool, str | None]]:
        def admit(candidate: Candidate) -> tuple[bool, str | None]:
            value = candidate.settings[axis_key]

            if validate(value):
                return True, None

            return False, f"{axis_key} must be {value_message}"

        return admit

    return build


def _positive_int_value(value: Any) -> TypeGuard[int]:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _nonnegative_int_value(value: Any) -> TypeGuard[int]:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _positive_int_tuple_value(value: Any) -> TypeGuard[tuple[int, ...]]:
    return (
        isinstance(value, tuple)
        and bool(value)
        and all(_positive_int_value(item) for item in value)
    )


def _non_empty_string_value(value: Any) -> TypeGuard[str]:
    return isinstance(value, str) and bool(value)


def _string_tuple_value(value: Any) -> TypeGuard[tuple[str, ...]]:
    return isinstance(value, tuple) and all(
        _non_empty_string_value(item) for item in value
    )


def _non_empty_string_tuple_value(value: Any) -> TypeGuard[tuple[str, ...]]:
    return isinstance(value, tuple) and bool(value) and _string_tuple_value(value)


POSITIVE_INT_AXIS = _distributed_value_axis("a positive integer", _positive_int_value)
NONNEGATIVE_INT_AXIS = _distributed_value_axis(
    "a non-negative integer", _nonnegative_int_value
)
POSITIVE_INT_TUPLE_AXIS = _distributed_value_axis(
    "a positive integer tuple", _positive_int_tuple_value
)
NON_EMPTY_STRING_AXIS = _distributed_value_axis(
    "a non-empty string", _non_empty_string_value
)
STRING_TUPLE_AXIS = _distributed_value_axis("a tuple of strings", _string_tuple_value)
NON_EMPTY_STRING_TUPLE_AXIS = _distributed_value_axis(
    "a non-empty tuple of strings", _non_empty_string_tuple_value
)
DTENSOR_PLACEMENT_VALUES = ("replicate", "shard_dim", "partial")
DTENSOR_REDISTRIBUTE_SCHEDULE_VALUES = (
    "none",
    "before_forward",
    "before_backward",
    "between_operator_parts",
    "before_output",
)
DISTRIBUTED_AXIS_ROWS = (
    _distributed_axis_row("distributed.launch", ("single_process", "torchrun")),
    _distributed_axis_row(
        "distributed.process_group_backend",
        ("nccl", "gloo", "ucc_when_available"),
    ),
    _distributed_axis_row(
        "distributed.local_rank_binding",
        ("cuda_local_rank", "explicit_device_map"),
    ),
    _distributed_axis_row("distributed.mesh_shape", (), POSITIVE_INT_TUPLE_AXIS),
    _distributed_axis_row(
        "distributed.mesh_dim_names",
        (),
        NON_EMPTY_STRING_TUPLE_AXIS,
    ),
    *_distributed_axis_rows(DTENSOR_PLACEMENT_KEYS, DTENSOR_PLACEMENT_VALUES),
    _distributed_axis_row(
        "dtensor.redistribute_schedule",
        DTENSOR_REDISTRIBUTE_SCHEDULE_VALUES,
    ),
    _distributed_axis_row(
        "fsdp.wrap_granularity",
        ("root", "transformer_block", "block_group"),
    ),
    _distributed_axis_row(
        "fsdp.reshard_after_forward",
        FSDP_RESHARD_AFTER_FORWARD_DOMAIN,
    ),
    _distributed_axis_row("fsdp.shard_placement_fn", ("none", "declared_fn")),
    *_distributed_axis_rows(
        (
            "fsdp.mp_policy.param_dtype",
            "fsdp.mp_policy.reduce_dtype",
            "fsdp.mp_policy.output_dtype",
        ),
        ("fp32", "bf16", "fp16"),
    ),
    _distributed_axis_row("fsdp.mp_policy.cast_forward_inputs", ("false", "true")),
    _distributed_axis_row("fsdp.offload_policy", ("none", "cpu")),
    _distributed_axis_row("fsdp.ignored_params", (), STRING_TUPLE_AXIS),
    _distributed_axis_row("fsdp.dp_mesh_dims", (), NON_EMPTY_STRING_TUPLE_AXIS),
    _distributed_axis_row("tp.plan", (), NON_EMPTY_STRING_AXIS),
    _distributed_axis_row("tp.qkv_projection", ("colwise", "rowwise", "replicated")),
    _distributed_axis_row(
        "tp.output_projection",
        ("rowwise", "colwise", "replicated"),
    ),
    _distributed_axis_row("tp.mlp_up_gate", ("colwise", "rowwise", "replicated")),
    _distributed_axis_row("tp.mlp_down", ("rowwise", "colwise", "replicated")),
    _distributed_axis_row("tp.embedding", ("replicated", "rowwise", "colwise")),
    _distributed_axis_row("tp.lm_head", ("replicated", "vocab_sharded")),
    *_distributed_axis_rows(
        ("tp.prepare_module_input", "tp.prepare_module_output"),
        (),
        NON_EMPTY_STRING_AXIS,
    ),
    _distributed_axis_row("tp.loss_parallel", ("false", "true")),
    _distributed_axis_row("sequence_parallel.enabled", ("false", "true")),
    _distributed_axis_row(
        "sequence_parallel.norm_modules",
        (),
        NON_EMPTY_STRING_TUPLE_AXIS,
    ),
    _distributed_axis_row(
        "sequence_parallel.output_placement_policy",
        ("preserve_sequence_shard", "redistribute_to_declared_output"),
    ),
    _distributed_axis_row("context_parallel.enabled", ("false", "true")),
    _distributed_axis_row(
        "context_parallel.rotate_method",
        ("all_gather", "all_to_all"),
    ),
    _distributed_axis_row(
        "context_parallel.sequence_dim",
        (),
        NONNEGATIVE_INT_AXIS,
    ),
    _distributed_axis_row(
        "comm.overlap",
        ("none", "all_gather_overlap", "reduce_scatter_overlap", "both"),
    ),
    _distributed_axis_row("comm.prefetch", ("none", "forward", "backward", "both")),
    _distributed_axis_row("comm.collective_bucket_size", (), POSITIVE_INT_AXIS),
)


def distributed_axis_descriptors(
    strategies: Sequence[str],
    *,
    policy: DistributedAdmissionPolicy,
) -> tuple[AxisDescriptor, ...]:
    """Return adapter-owned distributed axis descriptors."""
    return (
        distributed_strategy_axis(strategies, policy=policy),
        *_distributed_axes(DISTRIBUTED_AXIS_ROWS),
    )


def distributed_axis_registry(
    strategies: Sequence[str],
    *,
    policy: DistributedAdmissionPolicy,
) -> AxisRegistry:
    """Return a registry populated with distributed adapter axes."""
    registry = AxisRegistry()

    for axis in distributed_axis_descriptors(strategies, policy=policy):
        registry.register(axis)

    return registry


def distributed_axis_manifest(
    strategies: Sequence[str],
    *,
    policy: DistributedAdmissionPolicy,
) -> tuple[AxisDescriptor, ...]:
    """Return the distributed adapter axis descriptors."""
    return distributed_axis_descriptors(strategies, policy=policy)


def _distributed_axis(
    axis_key: str,
    allowed_values: tuple[Any, ...],
    *,
    admission_rule: Callable[[Candidate], tuple[bool, str | None]] | None = None,
) -> AxisDescriptor:
    rule = _distributed_axis_rule(axis_key, admission_rule)

    return AxisDescriptor(
        name=axis_key,
        settings_keys=(axis_key,),
        allowed_values=allowed_values,
        adapter_id="vptune.distributed",
        adapter_version=PACKAGE_VERSION,
        admission_rule=rule,
    )


def _distributed_axes(rows: Sequence[tuple[Any, ...]]) -> tuple[AxisDescriptor, ...]:
    return tuple(
        _distributed_axis(
            axis_key,
            allowed_values,
            admission_rule=None if rule_factory is None else rule_factory(axis_key),
        )
        for axis_key, allowed_values, rule_factory in rows
    )


def _distributed_axis_rule(
    axis_key: str,
    admission_rule: Callable[[Candidate], tuple[bool, str | None]] | None,
) -> Callable[[Candidate], tuple[bool, str | None]]:
    def admit(candidate: Candidate) -> tuple[bool, str | None]:
        if "distributed.strategy" not in candidate.settings:
            return False, f"{axis_key} requires distributed.strategy"

        if admission_rule is None:
            return True, None

        return admission_rule(candidate)

    return admit


def admit_distributed_candidate(
    candidate: Candidate,
    *,
    policy: DistributedAdmissionPolicy,
) -> tuple[bool, str | None]:
    """Return whether a distributed candidate is admitted."""
    strategy = candidate.settings.get("distributed.strategy")

    if strategy not in DISTRIBUTED_STRATEGIES:
        return False, f"unsupported distributed strategy: {strategy}"

    validation_error = _first_error((
        _distributed_common_validation_error(candidate.settings),
        _strategy_validation_error(candidate.settings, policy),
    ))

    if validation_error is not None:
        return False, validation_error

    return True, None


def _strategy_validation_error(
    settings: Mapping[str, Any],
    policy: DistributedAdmissionPolicy,
) -> str | None:
    strategy = settings.get("distributed.strategy")

    if strategy == "single_gpu":
        return None

    if strategy in {"fsdp2", "hsdp"}:
        return _fsdp2_validation_error(settings, policy)

    if strategy in LAYOUT_DISTRIBUTED_STRATEGIES:
        return _layout_validation_error(settings, policy)

    if strategy == "hybrid":
        return _hybrid_validation_error(settings, policy)

    return f"unsupported distributed strategy: {strategy}"


def _distributed_common_validation_error(settings: Mapping[str, Any]) -> str | None:
    return _first_error((
        *(
            _required_string_domain_error(settings, key, values)
            for key, values in DISTRIBUTED_COMMON_STRING_DOMAIN_FIELDS
        ),
        _required_value_error(
            settings,
            "distributed.mesh_shape",
            _positive_int_tuple_value,
            "a non-empty tuple of positive integers",
        ),
        _mesh_dim_names_error(settings),
        _communication_settings_error(settings),
    ))


def _required_string_domain_error(
    settings: Mapping[str, Any],
    key: str,
    values: tuple[str, ...],
) -> str | None:
    return _string_domain_value_error(settings.get(key), key, values)


def _mesh_dim_names_error(settings: Mapping[str, Any]) -> str | None:
    value = settings.get("distributed.mesh_dim_names")
    mesh_shape = settings.get("distributed.mesh_shape")

    if not _non_empty_string_tuple_value(value):
        return "distributed.mesh_dim_names must be a non-empty tuple of strings"

    if isinstance(mesh_shape, tuple) and len(value) != len(mesh_shape):
        return "distributed.mesh_dim_names must match distributed.mesh_shape"

    return None


def _communication_settings_error(settings: Mapping[str, Any]) -> str | None:
    return _first_error((
        *(
            _optional_string_domain_error(
                settings,
                key,
                values,
            )
            for key, values in COMMUNICATION_STRING_DOMAIN_FIELDS
        ),
        _optional_value_error(
            settings,
            "comm.collective_bucket_size",
            _positive_int_value,
            "a positive integer",
        ),
    ))


def _optional_string_domain_error(
    settings: Mapping[str, Any],
    key: str,
    values: Sequence[str],
) -> str | None:
    if key not in settings:
        return None

    return _string_domain_value_error(settings.get(key), key, values)


def _string_domain_value_error(
    value: Any,
    key: str,
    values: Sequence[str],
) -> str | None:
    if not isinstance(value, str) or value not in values:
        return f"{key} is unsupported: {value}"

    return None


def _fsdp2_validation_error(
    settings: Mapping[str, Any],
    policy: DistributedAdmissionPolicy,
) -> str | None:
    return _first_error((
        _required_value_rows_error(settings, FSDP2_PRE_POLICY_VALUE_ROWS),
        _policy_fields_error(settings, policy.fsdp2, FSDP2_POLICY_FIELDS),
        _required_value_rows_error(settings, FSDP2_POST_POLICY_VALUE_ROWS),
    ))


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


def _first_error(errors: Sequence[str | None]) -> str | None:
    for error in errors:
        if error is not None:
            return error

    return None


def _hybrid_validation_error(
    settings: Mapping[str, Any],
    policy: DistributedAdmissionPolicy,
) -> str | None:
    return _first_error((
        _fsdp2_validation_error(settings, policy),
        _layout_common_error(settings, policy),
        _policy_fields_error(
            settings,
            policy.tensor_parallel,
            TENSOR_PARALLEL_POLICY_FIELDS,
        ),
    ))


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

    policy_error = _policy_fields_error(settings, policy.dtensor, DTENSOR_POLICY_FIELDS)

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
        placement_error = _required_string_domain_value_error(
            settings,
            key,
            DTENSOR_PLACEMENT_VALUES,
        )

        if placement_error is not None:
            return placement_error

    return None


def _layout_mode_error(
    settings: Mapping[str, Any],
    policy: DistributedAdmissionPolicy,
) -> str | None:
    strategy = settings.get("distributed.strategy")

    if not isinstance(strategy, str):
        return f"unsupported layout distributed strategy: {strategy}"

    row = LAYOUT_MODE_POLICY_FIELDS.get(strategy)

    if row is not None:
        policy_name, fields = row

        return _policy_fields_error(
            settings,
            getattr(policy, policy_name),
            fields,
        )

    return f"unsupported layout distributed strategy: {strategy}"


def _int_value(value: Any) -> bool:
    return isinstance(value, int)


def _mapping_value(value: Any) -> bool:
    return isinstance(value, Mapping)


def _false_value(value: Any) -> bool:
    return value is False


def _bool_value(value: Any) -> bool:
    return isinstance(value, bool)


def _non_empty_mapping_value(value: Any) -> bool:
    return isinstance(value, Mapping) and bool(value)


def _required_value_error(
    settings: Mapping[str, Any],
    key: str,
    validate: Callable[[Any], bool],
    value_message: str,
) -> str | None:
    value = settings.get(key)

    if not validate(value):
        return f"{key} must be {value_message}"

    return None


def _optional_value_error(
    settings: Mapping[str, Any],
    key: str,
    validate: Callable[[Any], bool],
    value_message: str,
) -> str | None:
    if key not in settings:
        return None

    return _required_value_error(settings, key, validate, value_message)


def _required_string_domain_value_error(
    settings: Mapping[str, Any],
    key: str,
    values: Sequence[str],
) -> str | None:
    value_error = _required_value_error(
        settings,
        key,
        _non_empty_string_value,
        "a non-empty string",
    )

    if value_error is not None:
        return value_error

    value = settings.get(key)

    if value not in values:
        return f"{key} is unsupported: {value}"

    return None


def _required_value_rows_error(
    settings: Mapping[str, Any],
    rows: Sequence[tuple[str, Callable[[Any], bool], str]],
) -> str | None:
    for key, validate, value_message in rows:
        row_error = _required_value_error(settings, key, validate, value_message)

        if row_error is not None:
            return row_error

    return None


FSDP2_PRE_POLICY_VALUE_ROWS = (
    (
        "fsdp.hook_entry_points",
        _non_empty_string_tuple_value,
        "a non-empty tuple of strings",
    ),
)
FSDP2_POST_POLICY_VALUE_ROWS = (
    ("fsdp.ignored_params", _string_tuple_value, "a tuple of strings"),
    (
        "fsdp.dp_mesh_dims",
        _non_empty_string_tuple_value,
        "a non-empty tuple of strings",
    ),
    ("fsdp.bypasses_hooks", _false_value, "false"),
    ("fsdp.bottom_up_order", _bool_value, "a bool"),
    ("fsdp.mutated_modules", _string_tuple_value, "a tuple of strings"),
    ("fsdp.collectives", _non_empty_mapping_value, "a non-empty mapping"),
)


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

    if not isinstance(allowed, tuple) or not _policy_value_allowed(value, allowed):
        return f"{key} is not allowed by distributed admission policy: {value}"

    return None


def _policy_value_allowed(value: Any, allowed: tuple[Any, ...]) -> bool:
    if any(value == item for item in allowed):
        return True

    if INTEGER_DOMAIN not in allowed:
        return False

    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def reduce_rank_statuses(statuses: Sequence[RankStatus]) -> dict[str, Any]:
    """Return global status from rank-local statuses.

    Raises:
        MaterializationError: If no rank status is supplied.
    """
    if not statuses:
        message = "distributed status reduction requires rank statuses"
        raise MaterializationError(message)

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
    rank_compile_timings: Sequence[RankCompileTiming] = (),
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
    selection_metadata = _distributed_selection_metadata(
        selected_settings,
        rank_memory_samples,
        rank_compile_timings,
    )

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
        "selection_metadata": selection_metadata,
        "rank_compile_timings": tuple(
            timing.to_record() for timing in rank_compile_timings
        ),
    }


def distributed_operation_factory(
    operator: OperatorSpec,
    *,
    model: torch.nn.Module,
    params: ParameterTree,
    buffers: BufferTree,
    module_call: ModuleCallSpec,
    strategy_applier: DistributedStrategyApplier | None = None,
    strategy_bindings: DistributedStrategyBindings | None = None,
    loss_parallel: Callable[[], Any] | None = None,
    parameter_surface: ParameterSurface | None = None,
    scalar_objectives: Mapping[str, ScalarObjective] | None = None,
    function_objectives: Mapping[str, FunctionObjective] | None = None,
) -> OperationFactory:
    """Return a distributed standard-operation factory."""
    resolved_applier, resolved_loss_parallel = _strategy_applier_and_loss_parallel(
        strategy_applier=strategy_applier,
        strategy_bindings=strategy_bindings,
        loss_parallel=loss_parallel,
    )

    def factory(
        candidate: Candidate,
        batch: Batch,
        vector: TensorTree,
    ) -> CandidateOperation:
        distributed_model = resolved_applier(model, candidate)
        redistribution = _distributed_redistribution(
            candidate.settings,
            strategy_bindings,
        )
        runtime_params = redistribution.before_forward_params(params)
        runtime_batch = redistribution.before_forward_batch(batch)
        runtime_vector = redistribution.before_forward_vector(vector)
        runtime_vector = redistribution.before_backward_vector(runtime_vector)
        standard_factory = standard_operation_factory(
            operator,
            params=runtime_params,
            buffers=buffers,
            parameter_surface=parameter_surface,
            scalar_objectives=scalar_objectives,
            function_objectives=function_objectives,
            module=distributed_model,
            module_call=module_call,
            intermediate_transform=redistribution.between_operator_parts,
        )
        operation = standard_factory(
            _standard_candidate(candidate),
            runtime_batch,
            runtime_vector,
        )
        distributed_operation = _distributed_loss_parallel(
            candidate.settings,
            operation,
            resolved_loss_parallel,
        )

        def wrapped() -> TensorTree:
            return redistribution.before_output(distributed_operation())

        return wrapped

    return factory


def _distributed_redistribution(
    settings: Mapping[str, Any],
    strategy_bindings: DistributedStrategyBindings | None,
) -> _DistributedRedistribution:
    schedule = settings.get("dtensor.redistribute_schedule")

    if schedule in {None, "none"}:
        return _DistributedRedistribution(settings, None, {})

    if strategy_bindings is None:
        message = "dtensor.redistribute_schedule requires strategy bindings"
        raise MaterializationError(message)

    mesh = _build_declared_mesh(settings, strategy_bindings.mesh)
    placements = _build_declared_placements(settings, strategy_bindings.placements)

    return _DistributedRedistribution(settings, mesh, placements)


def distributed_reference_check(
    operator: OperatorSpec,
    *,
    reference_model: torch.nn.Module,
    params: ParameterTree,
    buffers: BufferTree,
    module_call: ModuleCallSpec,
    thresholds: Mapping[str, float],
    parameter_surface: ParameterSurface | None = None,
    numeric_bound_fields: Mapping[str, Any] | None = None,
    scalar_objectives: Mapping[str, ScalarObjective] | None = None,
    function_objectives: Mapping[str, FunctionObjective] | None = None,
) -> ReferenceCheck:
    """Return a single-device reference check for distributed rows."""
    standard_check = standard_reference_check(
        operator,
        params=params,
        buffers=buffers,
        thresholds=thresholds,
        parameter_surface=parameter_surface,
        numeric_bound_fields=numeric_bound_fields,
        scalar_objectives=scalar_objectives,
        function_objectives=function_objectives,
        module=reference_model,
        module_call=module_call,
    )

    def check(
        candidate: Candidate,
        batch: Batch,
        vector: TensorTree,
    ) -> ReferenceResult:
        return standard_check(_standard_candidate(candidate), batch, vector)

    return check


@dataclasses.dataclass(frozen=True, slots=True)
class DistributedFullSizeCheck:
    """Full-size check and rank reduction for distributed rows."""

    operation_factory: OperationFactory
    reference_check: ReferenceCheck
    thresholds: Mapping[str, float]
    identity_payload: Mapping[str, Any]
    expected_rank_count: int
    global_parameter_surface: Mapping[str, Any]
    rank_reporter: DistributedRankReporter

    def __post_init__(self) -> None:
        """Validate output comparison thresholds.

        Raises:
            MaterializationError: If a required threshold is missing.
        """
        for key in ("max_abs_diff", "max_rel_diff"):
            if key not in self.thresholds:
                message = f"distributed full-size check requires threshold: {key}"
                raise MaterializationError(message)

    def identity(self) -> Mapping[str, Any]:
        """Return stable full-size check identity."""
        return {
            "full_size_check": "vptune.distributed.full_size",
            "identity": dict(self.identity_payload),
            "expected_rank_count": self.expected_rank_count,
            "global_parameter_surface": dict(self.global_parameter_surface),
            "thresholds": {
                "max_abs_diff": self.thresholds["max_abs_diff"],
                "max_rel_diff": self.thresholds["max_rel_diff"],
            },
        }

    def __call__(
        self,
        candidate: Candidate,
        inputs: tuple[tuple[Batch, TensorTree], ...],
        output: TensorTree,
        samples: tuple[Measurement, ...],
    ) -> Mapping[str, Any]:
        """Return distributed selection metadata for the measured row."""
        outputs = _full_size_output_tuple(output, len(inputs))
        max_abs = 0.0
        max_rel = 0.0

        for (batch, vector), observed in zip(inputs, outputs, strict=True):
            rerun = self.operation_factory(candidate, batch, vector)()
            measurements = tree_error_measurements(observed, rerun)
            validate_thresholds(measurements, self._output_thresholds())
            self.reference_check(candidate, batch, vector)
            max_abs = max(max_abs, float(measurements["max_abs_diff"]))
            max_rel = max(max_rel, float(measurements["max_rel_diff"]))

        report = self.rank_reporter(candidate, samples)
        record = distributed_record(
            identity=self.identity_payload,
            expected_rank_count=self.expected_rank_count,
            rank_statuses=report.rank_statuses,
            rank_memory_samples=report.rank_memory_samples,
            rank_selected_settings=report.rank_selected_settings,
            global_parameter_surface=self.global_parameter_surface,
            rank_compile_timings=report.rank_compile_timings,
        )

        return {
            **dict(record["selection_metadata"]),
            "distributed_status": record["status"],
            "distributed_rank_count": record["rank_count"],
            "distributed_failed_ranks": record["failed_ranks"],
            "distributed_full_size_max_abs_diff": max_abs,
            "distributed_full_size_max_rel_diff": max_rel,
        }

    def _output_thresholds(self) -> dict[str, float]:
        return {
            "max_abs_diff": self.thresholds["max_abs_diff"],
            "max_rel_diff": self.thresholds["max_rel_diff"],
        }


def distributed_full_size_check(
    *,
    operation_factory: OperationFactory,
    reference_check: ReferenceCheck,
    thresholds: Mapping[str, float],
    identity: Mapping[str, Any],
    expected_rank_count: int,
    global_parameter_surface: Mapping[str, Any],
    rank_reporter: DistributedRankReporter,
) -> FullSizeCheck:
    """Return a distributed full-size checker."""
    return DistributedFullSizeCheck(
        operation_factory=operation_factory,
        reference_check=reference_check,
        thresholds=dict(thresholds),
        identity_payload=dict(identity),
        expected_rank_count=expected_rank_count,
        global_parameter_surface=dict(global_parameter_surface),
        rank_reporter=rank_reporter,
    )


def distributed_runtime_config(
    operator: OperatorSpec,
    *,
    model: torch.nn.Module,
    reference_model: torch.nn.Module,
    rank_reporter: DistributedRankReporter,
    identity: Mapping[str, Any],
    expected_rank_count: int,
    global_parameter_surface: Mapping[str, Any],
    params: ParameterTree,
    buffers: BufferTree,
    candidates: Sequence[Candidate],
    thresholds: Mapping[str, float],
    objective_signature: Mapping[str, Any],
    module_call: ModuleCallSpec,
    axis_registry: CandidateAdmitter | None,
    strategy_applier: DistributedStrategyApplier | None = None,
    strategy_bindings: DistributedStrategyBindings | None = None,
    loss_parallel: Callable[[], Any] | None = None,
    parameter_surface: ParameterSurface | None = None,
    numeric_bound_fields: Mapping[str, Any] | None = None,
    scalar_objectives: Mapping[str, ScalarObjective] | None = None,
    function_objectives: Mapping[str, FunctionObjective] | None = None,
) -> RuntimeConfig:
    """Return a distributed runtime config for standard operators."""
    standard_kwargs = {
        "params": params,
        "buffers": buffers,
        "parameter_surface": parameter_surface,
        "scalar_objectives": scalar_objectives,
        "function_objectives": function_objectives,
    }
    operation_factory = distributed_operation_factory(
        operator,
        model=model,
        module_call=module_call,
        strategy_applier=strategy_applier,
        strategy_bindings=strategy_bindings,
        loss_parallel=loss_parallel,
        **standard_kwargs,
    )
    reference_check = distributed_reference_check(
        operator,
        reference_model=reference_model,
        module_call=module_call,
        thresholds=thresholds,
        numeric_bound_fields=numeric_bound_fields,
        **standard_kwargs,
    )
    runtime_signature = {
        "runtime": "distributed",
        "operator": operator.signature(),
        "model": module_identity(model),
        "reference_model": module_identity(reference_model),
        "params": tree_signature(params),
        "buffers": tree_signature(buffers),
        "parameter_surface": (
            None if parameter_surface is None else parameter_surface.signature()
        ),
        "thresholds": dict(thresholds),
        "numeric_bound_fields": {}
        if numeric_bound_fields is None
        else dict(numeric_bound_fields),
        "objective": dict(objective_signature),
        "module_call": module_call.signature(),
        "distributed": dict(identity),
        "expected_rank_count": expected_rank_count,
        "global_parameter_surface": dict(global_parameter_surface),
    }
    operation_factory = CallableOperationFactory(
        "vptune.distributed_operation_factory",
        PACKAGE_VERSION,
        runtime_signature,
        {"callback": "vptune.adapters.distributed.distributed_operation_factory"},
        operation_factory,
    )
    reference_check = CallableReferenceCheck(
        "vptune.distributed_reference_check",
        PACKAGE_VERSION,
        runtime_signature,
        {"callback": "vptune.adapters.distributed.distributed_reference_check"},
        reference_check,
    )
    full_size_check = distributed_full_size_check(
        operation_factory=operation_factory,
        reference_check=reference_check,
        thresholds=thresholds,
        identity=identity,
        expected_rank_count=expected_rank_count,
        global_parameter_surface=global_parameter_surface,
        rank_reporter=rank_reporter,
    )
    materializer = distributed_materializer(operation_factory)

    return RuntimeConfig(
        candidates=tuple(candidates),
        operation_factory=operation_factory,
        reference_check=reference_check,
        materializer=materializer,
        axis_registry=axis_registry,
        signature=runtime_signature,
        full_size_check=full_size_check,
        reference_check_name="standard_anchor",
    )


def distributed_materializer(
    operation_factory: RuntimeOperationFactory,
) -> CallableMaterializer:
    """Return a materializer for selected distributed rows."""
    return CallableMaterializer(
        "vptune.distributed_runtime",
        PACKAGE_VERSION,
        {"operation_factory": dict(operation_factory.identity())},
        {"callback": "_materialize_distributed_selected"},
        lambda candidate, record: _materialize_distributed_selected(
            operation_factory,
            candidate,
            record,
        ),
    )


def _materialize_distributed_selected(
    operation_factory: OperationFactory,
    candidate: Candidate,
    record: FullSizeRecord,
) -> Any:
    if (
        record.family != candidate.family
        or record.candidate_id != candidate.candidate_id
    ):
        message = "selected record does not match selected distributed candidate"
        raise MaterializationError(message)

    def selected(batch: Batch, vector: TensorTree) -> TensorTree:
        return operation_factory(candidate, batch, vector)()

    return selected


def _full_size_output_tuple(
    output: TensorTree,
    expected_count: int,
) -> tuple[Any, ...]:
    if not isinstance(output, tuple):
        message = "distributed full-size output must be a tuple"
        raise MaterializationError(message)

    if len(output) != expected_count:
        message = "distributed full-size output count differs from inputs"
        raise MaterializationError(message)

    return tuple(output)


def _standard_candidate(candidate: Candidate) -> Candidate:
    settings = {
        key: value
        for key, value in candidate.settings.items()
        if not _is_distributed_runtime_setting(key, value)
    }

    return dataclasses.replace(candidate, settings=settings)


def _is_distributed_runtime_setting(key: str, value: Any) -> bool:
    if any(key.startswith(prefix) for prefix in DISTRIBUTED_RUNTIME_SETTING_PREFIXES):
        return True

    return key in {"layout.params", "layout.vector", "layout.output"} and (
        value in DISTRIBUTED_LAYOUT_VALUES
    )


def _distributed_selection_metadata(
    selected_settings: Mapping[str, Any],
    rank_memory_samples: Sequence[Measurement],
    rank_compile_timings: Sequence[RankCompileTiming],
) -> dict[str, Any]:
    metadata = {
        "global_elapsed_seconds": max(
            sample.elapsed_seconds for sample in rank_memory_samples
        ),
    }
    compiled = selected_settings.get("compile.enabled") == "true"

    if compiled:
        _require_compile_timing_rank_set(rank_memory_samples, rank_compile_timings)
        metadata.update({
            "global_compile_time_seconds": max(
                timing.compile_time_seconds for timing in rank_compile_timings
            ),
            "global_steady_elapsed_seconds": max(
                timing.steady_elapsed_seconds for timing in rank_compile_timings
            ),
            "recompile_count": max(
                timing.recompile_count for timing in rank_compile_timings
            ),
        })

        return metadata

    if rank_compile_timings:
        message = "rank compile timings apply only to compiled distributed rows"
        raise MaterializationError(message)

    return metadata


def _require_compile_timing_rank_set(
    rank_memory_samples: Sequence[Measurement],
    rank_compile_timings: Sequence[RankCompileTiming],
) -> None:
    memory_ranks = _rank_set(
        tuple(sample.rank for sample in rank_memory_samples),
        "memory",
    )
    compile_ranks = _rank_set(
        tuple(timing.rank for timing in rank_compile_timings),
        "compile timing",
    )

    if memory_ranks != compile_ranks:
        message = "distributed compile timing ranks differ from memory ranks"
        raise MaterializationError(message)


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
