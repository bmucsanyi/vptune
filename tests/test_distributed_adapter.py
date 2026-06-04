import pytest
import torch

import vptune.ext as vpx
from vptune import (
    AdmissionError,
    Candidate,
    MaterializationError,
    Measurement,
)
from vptune.adapters.distributed import (
    DistributedAdmissionPolicy,
    RankSelectedSettings,
    RankStatus,
    admit_distributed_candidate,
    apply_context_parallel,
    apply_fsdp2,
    apply_fsdp2_group,
    apply_tensor_parallel,
    build_colwise_parallel,
    build_device_mesh,
    build_dtensor_placement,
    build_fsdp_dp_mesh_dims,
    build_fsdp_mixed_precision_policy,
    build_fsdp_offload_policy,
    build_prepare_module_input,
    build_prepare_module_output,
    build_rowwise_parallel,
    build_sequence_parallel,
    collective_all_gather_into_tensor,
    collective_all_to_all_single,
    collective_reduce_scatter_tensor,
    distributed_identity,
    distributed_record,
    distributed_strategy_axis,
    initialize_process_group,
    named_modules_for_distributed_wrap,
    redistribute_dtensor,
    reduce_rank_statuses,
    require_rank_selected_settings_agree,
    resolve_process_group_backend,
    run_with_loss_parallel,
    wait_collective,
)


def distributed_policy(
    *,
    hook_entry_policy: str = "root-forward",
    output_layout: str = "rowwise",
    offload: str = "none",
) -> DistributedAdmissionPolicy:
    return DistributedAdmissionPolicy(
        candidate_generator_version="1",
        device_mesh={"shape": (2,), "names": ("data",)},
        rank_count=2,
        per_rank_placements=(
            {"rank": 0, "device": "cuda:0"},
            {"rank": 1, "device": "cuda:1"},
        ),
        communication={"backend": "nccl"},
        fsdp2={
            "allowed_fsdp.hook_entry_policy": (hook_entry_policy,),
            "allowed_fsdp.wrap_granularity": ("root", "transformer_block"),
            "allowed_fsdp.forward_prefetch": ("disabled", "next-forward"),
            "allowed_fsdp.backward_prefetch": ("disabled", "backward-pre"),
            "allowed_fsdp.reshard_after_forward": ("true", "false"),
            "allowed_fsdp.shard_placement_fn": ("none", "declared_fn"),
            "allowed_fsdp.mp_policy.param_dtype": ("fp32", "bf16", "fp16"),
            "allowed_fsdp.mp_policy.reduce_dtype": ("fp32", "bf16", "fp16"),
            "allowed_fsdp.mp_policy.output_dtype": ("fp32", "bf16", "fp16"),
            "allowed_fsdp.mp_policy.cast_forward_inputs": ("false", "true"),
            "allowed_fsdp.offload_policy": (offload,),
        },
        dtensor={
            "allowed_dtensor.module_class": ("torch.nn.Linear",),
            "allowed_dtensor.to_local_grad_placement": ("redistribute",),
            "allowed_dtensor.from_local_check": ("strict",),
            "allowed_dtensor.uneven_shard_handling": ("reject",),
            "allowed_dtensor.async_local_tensor_handling": ("sync",),
            "allowed_dtensor.redistribute_schedule": ("none", "before_forward"),
            "allowed_higher_order_diff_status": ("supported",),
        },
        tensor_parallel={
            "allowed_tp.plan": ("linear",),
            "allowed_tp.qkv_projection": ("colwise", "replicated"),
            "allowed_tp.output_projection": (output_layout,),
            "allowed_tp.mlp_up_gate": ("colwise", "replicated"),
            "allowed_tp.mlp_down": ("rowwise", "replicated"),
            "allowed_tp.embedding": ("replicated",),
            "allowed_tp.lm_head": ("replicated", "vocab_sharded"),
            "allowed_tp.prepare_module_input": ("declared",),
            "allowed_tp.prepare_module_output": ("declared",),
            "allowed_tp.loss_parallel": ("false", "true"),
        },
        sequence_parallel={
            "allowed_sequence_parallel.enabled": ("true",),
            "allowed_sequence_parallel.norm_modules": (("norm",),),
            "allowed_sequence_parallel.output_placement_policy": (
                "preserve_sequence_shard",
            ),
        },
        context_parallel={
            "allowed_context_parallel.enabled": ("true",),
            "allowed_context_parallel.rotate_method": ("all_gather",),
            "allowed_context_parallel.sequence_dim": (1,),
        },
    )


def valid_fsdp_settings() -> dict[str, object]:
    return {
        "distributed.strategy": "fsdp2",
        "fsdp.hook_entry_points": ("root.forward",),
        "fsdp.hook_entry_policy": "root-forward",
        "fsdp.wrap_granularity": "root",
        "fsdp.forward_prefetch": "disabled",
        "fsdp.backward_prefetch": "disabled",
        "fsdp.reshard_after_forward": "true",
        "fsdp.shard_placement_fn": "none",
        "fsdp.mp_policy.param_dtype": "fp32",
        "fsdp.mp_policy.reduce_dtype": "fp32",
        "fsdp.mp_policy.output_dtype": "fp32",
        "fsdp.mp_policy.cast_forward_inputs": "false",
        "fsdp.offload_policy": "none",
        "fsdp.ignored_params": (),
        "fsdp.dp_mesh_dims": ("data",),
        "fsdp.bypasses_hooks": False,
        "fsdp.bottom_up_order": True,
        "fsdp.mutated_modules": (),
        "fsdp.collectives": {"all_gather": True, "reduce_scatter": True},
    }


def valid_layout_settings() -> dict[str, object]:
    return {
        "distributed.strategy": "tensor_parallel",
        "dtensor.params_placement": "shard_dim",
        "dtensor.vector_placement": "replicate",
        "dtensor.logits_placement": "replicate",
        "dtensor.tangent_placement": "replicate",
        "dtensor.cotangent_placement": "replicate",
        "dtensor.output_placement": "replicate",
        "dtensor.redistribute_schedule": "none",
        "dtensor.module_class": "torch.nn.Linear",
        "dtensor.to_local_grad_placement": "redistribute",
        "dtensor.from_local_check": "strict",
        "dtensor.uneven_shard_handling": "reject",
        "dtensor.async_local_tensor_handling": "sync",
        "dtensor.higher_order_diff_status": {
            "dtensor.params_placement": "supported",
            "dtensor.vector_placement": "supported",
            "dtensor.logits_placement": "supported",
            "dtensor.tangent_placement": "supported",
            "dtensor.cotangent_placement": "supported",
            "dtensor.output_placement": "supported",
        },
        "tp.plan": "linear",
        "tp.qkv_projection": "colwise",
        "tp.output_projection": "rowwise",
        "tp.mlp_up_gate": "colwise",
        "tp.mlp_down": "rowwise",
        "tp.embedding": "replicated",
        "tp.lm_head": "replicated",
        "tp.prepare_module_input": "declared",
        "tp.prepare_module_output": "declared",
        "tp.loss_parallel": "false",
    }


def valid_sequence_parallel_settings() -> dict[str, object]:
    return {
        **valid_layout_settings(),
        "distributed.strategy": "sequence_parallel",
        "sequence_parallel.enabled": "true",
        "sequence_parallel.norm_modules": ("norm",),
        "sequence_parallel.output_placement_policy": "preserve_sequence_shard",
    }


def valid_context_parallel_settings() -> dict[str, object]:
    return {
        **valid_layout_settings(),
        "distributed.strategy": "context_parallel",
        "context_parallel.enabled": "true",
        "context_parallel.rotate_method": "all_gather",
        "context_parallel.sequence_dim": 1,
    }


def valid_distributed_identity() -> dict[str, object]:
    return distributed_identity(
        device_mesh={"shape": (2,), "names": ("data",)},
        placements=({"parameter": "weight", "placement": "shard0"},),
        communication={"backend": "nccl"},
    )


def test_resolve_process_group_backend_uses_declared_backend() -> None:
    assert (
        resolve_process_group_backend(
            "nccl",
            is_ucc_available=lambda: False,
        )
        == "nccl"
    )
    assert (
        resolve_process_group_backend(
            "gloo",
            is_ucc_available=lambda: False,
        )
        == "gloo"
    )
    assert (
        resolve_process_group_backend(
            "ucc_when_available",
            is_ucc_available=lambda: True,
        )
        == "ucc"
    )

    with pytest.raises(AdmissionError, match="UCC support"):
        resolve_process_group_backend(
            "ucc_when_available",
            is_ucc_available=lambda: False,
        )

    with pytest.raises(AdmissionError, match="unsupported"):
        resolve_process_group_backend(
            "mpi",
            is_ucc_available=lambda: True,
        )


def test_initialize_process_group_forwards_declared_arguments() -> None:
    calls = []
    result = object()
    timeout = object()
    store = object()
    pg_options = object()
    device_id = object()

    def init_process_group(
        *,
        backend: str,
        init_method: str | None,
        timeout: object,
        world_size: int,
        rank: int,
        store: object,
        pg_options: object,
        device_id: object,
    ) -> object:
        calls.append({
            "backend": backend,
            "init_method": init_method,
            "timeout": timeout,
            "world_size": world_size,
            "rank": rank,
            "store": store,
            "pg_options": pg_options,
            "device_id": device_id,
        })

        return result

    initialized = initialize_process_group(
        init_process_group,
        backend="nccl",
        init_method="env://",
        timeout=timeout,
        world_size=8,
        rank=3,
        store=store,
        pg_options=pg_options,
        device_id=device_id,
    )

    assert initialized is result
    assert calls == [
        {
            "backend": "nccl",
            "init_method": "env://",
            "timeout": timeout,
            "world_size": 8,
            "rank": 3,
            "store": store,
            "pg_options": pg_options,
            "device_id": device_id,
        }
    ]


def test_build_device_mesh_forwards_declared_mesh_arguments() -> None:
    calls = []
    mesh = object()

    def init_device_mesh(
        device_type: str,
        mesh_shape: tuple[int, ...],
        *,
        mesh_dim_names: tuple[str, ...],
    ) -> object:
        calls.append({
            "device_type": device_type,
            "mesh_shape": mesh_shape,
            "mesh_dim_names": mesh_dim_names,
        })

        return mesh

    result = build_device_mesh(
        init_device_mesh,
        device_type="cuda",
        mesh_shape=(2, 4),
        mesh_dim_names=("dp", "tp"),
    )

    assert result is mesh
    assert calls == [
        {
            "device_type": "cuda",
            "mesh_shape": (2, 4),
            "mesh_dim_names": ("dp", "tp"),
        }
    ]


def test_apply_fsdp2_forwards_declared_fully_shard_arguments() -> None:
    calls = []
    module = torch.nn.Linear(2, 2)
    sharded = torch.nn.Sequential(module)
    mesh = object()
    mp_policy = object()
    offload_policy = object()
    ignored = tuple(module.parameters())

    def shard_placement_fn(_: torch.nn.Module) -> object:
        return object()

    def fully_shard(
        module: torch.nn.Module,
        *,
        mesh: object,
        reshard_after_forward: bool | int | None,
        shard_placement_fn: object,
        mp_policy: object,
        offload_policy: object,
        ignored_params: tuple[torch.nn.Parameter, ...],
        dp_mesh_dims: tuple[int, ...],
    ) -> torch.nn.Module:
        calls.append({
            "module": module,
            "mesh": mesh,
            "reshard_after_forward": reshard_after_forward,
            "shard_placement_fn": shard_placement_fn,
            "mp_policy": mp_policy,
            "offload_policy": offload_policy,
            "ignored_params": ignored_params,
            "dp_mesh_dims": dp_mesh_dims,
        })

        return sharded

    result = apply_fsdp2(
        fully_shard,
        module,
        mesh=mesh,
        reshard_after_forward=2,
        shard_placement_fn=shard_placement_fn,
        mp_policy=mp_policy,
        offload_policy=offload_policy,
        ignored_params=ignored,
        dp_mesh_dims=(0,),
    )

    assert result is sharded
    assert calls == [
        {
            "module": module,
            "mesh": mesh,
            "reshard_after_forward": 2,
            "shard_placement_fn": shard_placement_fn,
            "mp_policy": mp_policy,
            "offload_policy": offload_policy,
            "ignored_params": ignored,
            "dp_mesh_dims": (0,),
        }
    ]


def test_named_modules_for_distributed_wrap_returns_declared_order() -> None:
    first = torch.nn.Linear(2, 2)
    nested = torch.nn.Sequential(torch.nn.Linear(2, 2))
    model = torch.nn.Sequential(first, nested)

    selected = named_modules_for_distributed_wrap(model, ("1.0", "", "0"))

    assert selected == (nested[0], model, first)

    with pytest.raises(AdmissionError, match="missing"):
        named_modules_for_distributed_wrap(model, ("2",))


def test_apply_fsdp2_group_forwards_declared_module_group() -> None:
    calls = []
    first = torch.nn.Linear(2, 2)
    second = torch.nn.Linear(2, 2)
    sharded = torch.nn.Sequential(first, second)
    modules = list(named_modules_for_distributed_wrap(sharded, ("0", "1")))
    mesh = object()
    mp_policy = object()
    offload_policy = object()
    ignored = tuple(first.parameters())

    def shard_placement_fn(_: torch.nn.Module) -> object:
        return object()

    def fully_shard(
        modules: list[torch.nn.Module],
        *,
        mesh: object,
        reshard_after_forward: bool | int | None,
        shard_placement_fn: object,
        mp_policy: object,
        offload_policy: object,
        ignored_params: tuple[torch.nn.Parameter, ...],
        dp_mesh_dims: tuple[int, ...],
    ) -> torch.nn.Module:
        calls.append({
            "modules": modules,
            "mesh": mesh,
            "reshard_after_forward": reshard_after_forward,
            "shard_placement_fn": shard_placement_fn,
            "mp_policy": mp_policy,
            "offload_policy": offload_policy,
            "ignored_params": ignored_params,
            "dp_mesh_dims": dp_mesh_dims,
        })

        return sharded

    result = apply_fsdp2_group(
        fully_shard,
        modules,
        mesh=mesh,
        reshard_after_forward=True,
        shard_placement_fn=shard_placement_fn,
        mp_policy=mp_policy,
        offload_policy=offload_policy,
        ignored_params=ignored,
        dp_mesh_dims=("dp",),
    )

    assert result is sharded
    assert calls == [
        {
            "modules": modules,
            "mesh": mesh,
            "reshard_after_forward": True,
            "shard_placement_fn": shard_placement_fn,
            "mp_policy": mp_policy,
            "offload_policy": offload_policy,
            "ignored_params": ignored,
            "dp_mesh_dims": ("dp",),
        }
    ]


def test_build_fsdp_mixed_precision_policy_forwards_declared_fields() -> None:
    calls = []
    policy = object()

    def mixed_precision_policy(
        *,
        param_dtype: torch.dtype | None,
        reduce_dtype: torch.dtype | None,
        output_dtype: torch.dtype | None,
        cast_forward_inputs: bool,
    ) -> object:
        calls.append({
            "param_dtype": param_dtype,
            "reduce_dtype": reduce_dtype,
            "output_dtype": output_dtype,
            "cast_forward_inputs": cast_forward_inputs,
        })

        return policy

    result = build_fsdp_mixed_precision_policy(
        mixed_precision_policy,
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
        output_dtype=torch.float16,
        cast_forward_inputs=False,
    )

    assert result is policy
    assert calls == [
        {
            "param_dtype": torch.bfloat16,
            "reduce_dtype": torch.float32,
            "output_dtype": torch.float16,
            "cast_forward_inputs": False,
        }
    ]


def test_build_fsdp_offload_policy_forwards_declared_mode() -> None:
    calls = []
    no_offload = object()
    cpu_offload = object()

    def offload_policy() -> object:
        calls.append({"kind": "none"})

        return no_offload

    def cpu_offload_policy(*, pin_memory: bool) -> object:
        calls.append({"kind": "cpu", "pin_memory": pin_memory})

        return cpu_offload

    assert (
        build_fsdp_offload_policy(
            offload_policy,
            cpu_offload_policy,
            offload="none",
            pin_memory=False,
        )
        is no_offload
    )
    assert (
        build_fsdp_offload_policy(
            offload_policy,
            cpu_offload_policy,
            offload="cpu",
            pin_memory=True,
        )
        is cpu_offload
    )
    assert calls == [
        {"kind": "none"},
        {"kind": "cpu", "pin_memory": True},
    ]

    with pytest.raises(AdmissionError, match="unsupported"):
        build_fsdp_offload_policy(
            offload_policy,
            cpu_offload_policy,
            offload="disk",
            pin_memory=True,
        )


def test_build_fsdp_dp_mesh_dims_forwards_declared_dimensions() -> None:
    calls = []
    dims = object()

    def data_parallel_mesh_dims(
        *,
        shard: str | tuple[str, ...] | None,
        replicate: str | tuple[str, ...] | None,
    ) -> object:
        calls.append({"shard": shard, "replicate": replicate})

        return dims

    result = build_fsdp_dp_mesh_dims(
        data_parallel_mesh_dims,
        shard=("dp_shard", "expert"),
        replicate="dp_replicate",
    )

    assert result is dims
    assert calls == [
        {
            "shard": ("dp_shard", "expert"),
            "replicate": "dp_replicate",
        }
    ]


def test_apply_tensor_parallel_forwards_declared_parallelize_arguments() -> None:
    calls = []
    module = torch.nn.Linear(2, 2)
    parallelized = torch.nn.Sequential(module)
    mesh = object()
    plan = {"layers.0": object()}

    def parallelize_module(
        module: torch.nn.Module,
        device_mesh: object,
        parallelize_plan: dict[str, object],
        *,
        src_data_rank: int,
    ) -> torch.nn.Module:
        calls.append({
            "module": module,
            "device_mesh": device_mesh,
            "parallelize_plan": parallelize_plan,
            "src_data_rank": src_data_rank,
        })

        return parallelized

    result = apply_tensor_parallel(
        parallelize_module,
        module,
        device_mesh=mesh,
        parallelize_plan=plan,
        src_data_rank=3,
    )

    assert result is parallelized
    assert calls == [
        {
            "module": module,
            "device_mesh": mesh,
            "parallelize_plan": plan,
            "src_data_rank": 3,
        }
    ]


def test_build_tensor_parallel_styles_forward_declared_arguments() -> None:
    calls = []
    input_layout = object()
    output_layout = object()
    colwise_result = object()
    rowwise_result = object()
    sequence_result = object()

    def colwise_parallel(
        *,
        input_layouts: object,
        output_layouts: object,
        use_local_output: bool,
    ) -> object:
        calls.append({
            "kind": "colwise",
            "input_layouts": input_layouts,
            "output_layouts": output_layouts,
            "use_local_output": use_local_output,
        })

        return colwise_result

    def rowwise_parallel(
        *,
        input_layouts: object,
        output_layouts: object,
        use_local_output: bool,
    ) -> object:
        calls.append({
            "kind": "rowwise",
            "input_layouts": input_layouts,
            "output_layouts": output_layouts,
            "use_local_output": use_local_output,
        })

        return rowwise_result

    def sequence_parallel(
        *,
        sequence_dim: int,
        use_local_output: bool,
    ) -> object:
        calls.append({
            "kind": "sequence",
            "sequence_dim": sequence_dim,
            "use_local_output": use_local_output,
        })

        return sequence_result

    assert (
        build_colwise_parallel(
            colwise_parallel,
            input_layouts=input_layout,
            output_layouts=output_layout,
            use_local_output=False,
        )
        is colwise_result
    )
    assert (
        build_rowwise_parallel(
            rowwise_parallel,
            input_layouts=input_layout,
            output_layouts=output_layout,
            use_local_output=True,
        )
        is rowwise_result
    )
    assert (
        build_sequence_parallel(
            sequence_parallel,
            sequence_dim=2,
            use_local_output=False,
        )
        is sequence_result
    )
    assert calls == [
        {
            "kind": "colwise",
            "input_layouts": input_layout,
            "output_layouts": output_layout,
            "use_local_output": False,
        },
        {
            "kind": "rowwise",
            "input_layouts": input_layout,
            "output_layouts": output_layout,
            "use_local_output": True,
        },
        {
            "kind": "sequence",
            "sequence_dim": 2,
            "use_local_output": False,
        },
    ]


def test_build_prepare_module_styles_forward_declared_arguments() -> None:
    calls = []
    input_layouts = (object(), None)
    desired_input_layouts = (object(), None)
    input_kwarg_layouts = {"attention_mask": object()}
    desired_input_kwarg_layouts = {"attention_mask": object()}
    output_layouts = object()
    desired_output_layouts = object()
    input_result = object()
    output_result = object()

    def prepare_module_input(
        *,
        input_layouts: tuple[object | None, ...],
        desired_input_layouts: tuple[object | None, ...],
        input_kwarg_layouts: dict[str, object],
        desired_input_kwarg_layouts: dict[str, object],
        use_local_output: bool,
    ) -> object:
        calls.append({
            "kind": "input",
            "input_layouts": input_layouts,
            "desired_input_layouts": desired_input_layouts,
            "input_kwarg_layouts": input_kwarg_layouts,
            "desired_input_kwarg_layouts": desired_input_kwarg_layouts,
            "use_local_output": use_local_output,
        })

        return input_result

    def prepare_module_output(
        *,
        output_layouts: object,
        desired_output_layouts: object,
        use_local_output: bool,
    ) -> object:
        calls.append({
            "kind": "output",
            "output_layouts": output_layouts,
            "desired_output_layouts": desired_output_layouts,
            "use_local_output": use_local_output,
        })

        return output_result

    assert (
        build_prepare_module_input(
            prepare_module_input,
            input_layouts=input_layouts,
            desired_input_layouts=desired_input_layouts,
            input_kwarg_layouts=input_kwarg_layouts,
            desired_input_kwarg_layouts=desired_input_kwarg_layouts,
            use_local_output=False,
        )
        is input_result
    )
    assert (
        build_prepare_module_output(
            prepare_module_output,
            output_layouts=output_layouts,
            desired_output_layouts=desired_output_layouts,
            use_local_output=True,
        )
        is output_result
    )
    assert calls == [
        {
            "kind": "input",
            "input_layouts": input_layouts,
            "desired_input_layouts": desired_input_layouts,
            "input_kwarg_layouts": input_kwarg_layouts,
            "desired_input_kwarg_layouts": desired_input_kwarg_layouts,
            "use_local_output": False,
        },
        {
            "kind": "output",
            "output_layouts": output_layouts,
            "desired_output_layouts": desired_output_layouts,
            "use_local_output": True,
        },
    ]


def test_run_with_loss_parallel_executes_inside_context() -> None:
    events = []

    class LossParallel:
        def __enter__(self) -> None:
            events.append("enter")

        def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
            events.append("exit")

    def loss_parallel() -> LossParallel:
        return LossParallel()

    def operation() -> str:
        events.append("operation")

        return "result"

    assert run_with_loss_parallel(loss_parallel, operation) == "result"
    assert events == ["enter", "operation", "exit"]


def test_build_dtensor_placement_constructs_declared_placement() -> None:
    calls = []
    replicate_result = object()
    shard_result = object()
    partial_result = object()

    def replicate() -> object:
        calls.append({"kind": "replicate"})

        return replicate_result

    def shard(dim: int) -> object:
        calls.append({"kind": "shard", "dim": dim})

        return shard_result

    def partial(reduce_op: str) -> object:
        calls.append({"kind": "partial", "reduce_op": reduce_op})

        return partial_result

    assert (
        build_dtensor_placement(
            replicate,
            shard,
            partial,
            placement="replicate",
            shard_dim=None,
            reduce_op=None,
        )
        is replicate_result
    )
    assert (
        build_dtensor_placement(
            replicate,
            shard,
            partial,
            placement="shard_dim",
            shard_dim=1,
            reduce_op=None,
        )
        is shard_result
    )
    assert (
        build_dtensor_placement(
            replicate,
            shard,
            partial,
            placement="partial",
            shard_dim=None,
            reduce_op="sum",
        )
        is partial_result
    )
    assert calls == [
        {"kind": "replicate"},
        {"kind": "shard", "dim": 1},
        {"kind": "partial", "reduce_op": "sum"},
    ]


@pytest.mark.parametrize(
    ("placement", "shard_dim", "reduce_op"),
    [
        ("replicate", 0, None),
        ("replicate", None, "sum"),
        ("shard_dim", None, None),
        ("shard_dim", 0, "sum"),
        ("partial", 0, "sum"),
        ("partial", None, None),
        ("unknown", None, None),
    ],
)
def test_build_dtensor_placement_rejects_contradictory_fields(
    placement: str,
    shard_dim: int | None,
    reduce_op: str | None,
) -> None:
    def replicate() -> object:
        return object()

    def shard(_: int) -> object:
        return object()

    def partial(_: str) -> object:
        return object()

    with pytest.raises(AdmissionError):
        build_dtensor_placement(
            replicate,
            shard,
            partial,
            placement=placement,
            shard_dim=shard_dim,
            reduce_op=reduce_op,
        )


def test_redistribute_dtensor_forwards_declared_arguments() -> None:
    dtensor = RecordingDTensor()
    mesh = object()
    placements = (object(), object())

    result = redistribute_dtensor(
        dtensor,
        device_mesh=mesh,
        placements=placements,
        async_op=True,
        forward_dtype=torch.bfloat16,
        backward_dtype=torch.float32,
    )

    assert result is dtensor.result
    assert dtensor.calls == [
        {
            "device_mesh": mesh,
            "placements": placements,
            "async_op": True,
            "forward_dtype": torch.bfloat16,
            "backward_dtype": torch.float32,
        }
    ]


def test_apply_context_parallel_forwards_declared_arguments() -> None:
    calls = []
    mesh = object()
    result = object()
    first = torch.tensor([1.0])
    second = torch.tensor([2.0])

    def context_parallel(
        mesh: object,
        *,
        buffers: tuple[torch.Tensor, ...],
        buffer_seq_dims: tuple[int, ...],
        no_restore_buffers: tuple[torch.Tensor, ...],
    ) -> object:
        calls.append({
            "mesh": mesh,
            "buffers": buffers,
            "buffer_seq_dims": buffer_seq_dims,
            "no_restore_buffers": no_restore_buffers,
        })

        return result

    applied = apply_context_parallel(
        context_parallel,
        mesh,
        buffers=(first, second),
        buffer_seq_dims=(1, 1),
        no_restore_buffers=(second,),
    )

    assert applied is result
    assert calls == [
        {
            "mesh": mesh,
            "buffers": (first, second),
            "buffer_seq_dims": (1, 1),
            "no_restore_buffers": (second,),
        }
    ]


def test_collective_all_gather_into_tensor_forwards_declared_arguments() -> None:
    calls = []
    output = torch.empty(4)
    input_tensor = torch.ones(2)
    group = object()
    work = object()

    def all_gather_into_tensor(
        output_tensor: torch.Tensor,
        input_tensor: torch.Tensor,
        *,
        group: object,
        async_op: bool,
    ) -> object:
        calls.append({
            "output_tensor": output_tensor,
            "input_tensor": input_tensor,
            "group": group,
            "async_op": async_op,
        })

        return work

    result = collective_all_gather_into_tensor(
        all_gather_into_tensor,
        output,
        input_tensor,
        group=group,
        async_op=True,
    )

    assert result is work
    assert calls == [
        {
            "output_tensor": output,
            "input_tensor": input_tensor,
            "group": group,
            "async_op": True,
        }
    ]


def test_collective_reduce_scatter_tensor_forwards_declared_arguments() -> None:
    calls = []
    output = torch.empty(2)
    input_tensor = torch.ones(4)
    group = object()
    op = object()
    work = object()

    def reduce_scatter_tensor(
        output_tensor: torch.Tensor,
        input_tensor: torch.Tensor,
        *,
        op: object,
        group: object,
        async_op: bool,
    ) -> object:
        calls.append({
            "output_tensor": output_tensor,
            "input_tensor": input_tensor,
            "op": op,
            "group": group,
            "async_op": async_op,
        })

        return work

    result = collective_reduce_scatter_tensor(
        reduce_scatter_tensor,
        output,
        input_tensor,
        op=op,
        group=group,
        async_op=False,
    )

    assert result is work
    assert calls == [
        {
            "output_tensor": output,
            "input_tensor": input_tensor,
            "op": op,
            "group": group,
            "async_op": False,
        }
    ]


def test_collective_all_to_all_single_forwards_declared_arguments() -> None:
    calls = []
    output = torch.empty(4)
    input_tensor = torch.ones(4)
    group = object()
    output_split_sizes = [1, 3]
    input_split_sizes = [2, 2]
    work = object()

    def all_to_all_single(
        output_tensor: torch.Tensor,
        input_tensor: torch.Tensor,
        *,
        output_split_sizes: list[int],
        input_split_sizes: list[int],
        group: object,
        async_op: bool,
    ) -> object:
        calls.append({
            "output_tensor": output_tensor,
            "input_tensor": input_tensor,
            "output_split_sizes": output_split_sizes,
            "input_split_sizes": input_split_sizes,
            "group": group,
            "async_op": async_op,
        })

        return work

    result = collective_all_to_all_single(
        all_to_all_single,
        output,
        input_tensor,
        output_split_sizes=output_split_sizes,
        input_split_sizes=input_split_sizes,
        group=group,
        async_op=True,
    )

    assert result is work
    assert calls == [
        {
            "output_tensor": output,
            "input_tensor": input_tensor,
            "output_split_sizes": output_split_sizes,
            "input_split_sizes": input_split_sizes,
            "group": group,
            "async_op": True,
        }
    ]


def test_wait_collective_waits_on_work_handle() -> None:
    work = RecordingWork("done")

    assert wait_collective(work) == "done"
    assert work.waited is True


class RecordingDTensor:
    def __init__(self) -> None:
        self.calls = []
        self.result = object()

    def redistribute(
        self,
        *,
        device_mesh: object,
        placements: tuple[object, ...],
        async_op: bool,
        forward_dtype: torch.dtype | None,
        backward_dtype: torch.dtype | None,
    ) -> object:
        self.calls.append({
            "device_mesh": device_mesh,
            "placements": placements,
            "async_op": async_op,
            "forward_dtype": forward_dtype,
            "backward_dtype": backward_dtype,
        })

        return self.result


class RecordingWork:
    def __init__(self, result: object) -> None:
        self.result = result
        self.waited = False

    def wait(self) -> object:
        self.waited = True

        return self.result


def test_distributed_strategy_axis_validates_modes() -> None:
    policy = distributed_policy()
    axis = distributed_strategy_axis(("fsdp2", "tensor_parallel"), policy=policy)

    assert axis.adapter_id == "vptune.distributed"
    assert axis.allowed_values == ("fsdp2", "tensor_parallel")
    assert axis.signature()["has_admission_rule"] is True

    with pytest.raises(AdmissionError):
        distributed_strategy_axis(("single_device",), policy=policy)


def test_distributed_strategy_axis_records_admission_identity() -> None:
    first = distributed_strategy_axis(("fsdp2",), policy=distributed_policy())
    second = distributed_strategy_axis(
        ("fsdp2",),
        policy=distributed_policy(hook_entry_policy="layer-forward"),
    )

    assert first.signature()["identity"]["adapter_id"] == "vptune.distributed"
    assert first.signature()["identity"] != second.signature()["identity"]


def test_distributed_strategy_axis_owns_optional_admission_fields() -> None:
    registry = vpx.AxisRegistry()
    registry.register(
        distributed_strategy_axis(
            ("fsdp2", "tensor_parallel"),
            policy=distributed_policy(),
        )
    )
    fsdp = Candidate("family", "fsdp", valid_fsdp_settings())
    tensor_parallel = Candidate("family", "tensor-parallel", valid_layout_settings())

    assert registry.admit(fsdp).admission_status == "passed"
    assert registry.admit(tensor_parallel).admission_status == "passed"


def test_fsdp2_admission_requires_hook_entry_and_rejects_bypass() -> None:
    policy = distributed_policy()
    valid = Candidate("family", "valid", valid_fsdp_settings())
    missing_hook = Candidate(
        "family",
        "missing-hook",
        {**valid_fsdp_settings(), "fsdp.hook_entry_points": ()},
    )
    bypass = Candidate(
        "family",
        "bypass",
        {**valid_fsdp_settings(), "fsdp.bypasses_hooks": True},
    )

    assert admit_distributed_candidate(valid, policy=policy) == (True, None)
    assert admit_distributed_candidate(missing_hook, policy=policy)[0] is False
    assert admit_distributed_candidate(bypass, policy=policy)[0] is False


def test_fsdp2_admission_requires_policy_axes() -> None:
    policy = distributed_policy()
    valid = Candidate("family", "valid", valid_fsdp_settings())
    missing_prefetch_settings = dict(valid_fsdp_settings())
    missing_prefetch_settings.pop("fsdp.forward_prefetch")
    missing_prefetch = Candidate(
        "family",
        "missing-prefetch",
        missing_prefetch_settings,
    )
    offload_policy_mismatch = Candidate(
        "family",
        "offload-policy-mismatch",
        {**valid_fsdp_settings(), "fsdp.offload_policy": "cpu"},
    )
    hook_policy_mismatch = Candidate(
        "family",
        "hook-policy-mismatch",
        {**valid_fsdp_settings(), "fsdp.hook_entry_policy": "layer-forward"},
    )

    assert admit_distributed_candidate(valid, policy=policy) == (True, None)
    assert admit_distributed_candidate(missing_prefetch, policy=policy)[0] is False
    assert (
        admit_distributed_candidate(offload_policy_mismatch, policy=policy)[0] is False
    )
    assert admit_distributed_candidate(hook_policy_mismatch, policy=policy)[0] is False


def test_dtensor_admission_requires_gradient_placement_policy() -> None:
    policy = distributed_policy()
    valid = Candidate("family", "valid", valid_layout_settings())
    invalid = Candidate(
        "family",
        "invalid",
        {**valid_layout_settings(), "dtensor.to_local_grad_placement": "drop"},
    )

    assert admit_distributed_candidate(valid, policy=policy) == (True, None)
    assert admit_distributed_candidate(invalid, policy=policy)[0] is False


def test_dtensor_admission_records_module_class_and_higher_order_diff() -> None:
    policy = distributed_policy()
    valid = Candidate("family", "valid", valid_layout_settings())
    missing_status = dict(valid_layout_settings())
    missing_status["dtensor.higher_order_diff_status"] = {
        "dtensor.params_placement": "supported"
    }
    unsupported_status = dict(valid_layout_settings())
    unsupported_status["dtensor.higher_order_diff_status"] = {
        "dtensor.params_placement": "supported",
        "dtensor.vector_placement": "supported",
        "dtensor.logits_placement": "supported",
        "dtensor.tangent_placement": "supported",
        "dtensor.cotangent_placement": "supported",
        "dtensor.output_placement": "unsupported",
    }
    unsupported_module = Candidate(
        "family",
        "unsupported-module",
        {**valid_layout_settings(), "dtensor.module_class": "torch.nn.Conv2d"},
    )

    assert admit_distributed_candidate(valid, policy=policy) == (True, None)
    assert (
        admit_distributed_candidate(
            Candidate("family", "missing-status", missing_status),
            policy=policy,
        )[0]
        is False
    )
    assert (
        admit_distributed_candidate(
            Candidate("family", "unsupported-status", unsupported_status),
            policy=policy,
        )[0]
        is False
    )
    assert admit_distributed_candidate(unsupported_module, policy=policy)[0] is False


def test_dtensor_admission_requires_higher_order_diff_status_per_slot() -> None:
    settings = {
        **valid_layout_settings(),
        "dtensor.higher_order_diff_status": {
            "dtensor.params_placement": "supported",
            "dtensor.vector_placement": "supported",
            "dtensor.logits_placement": "supported",
            "dtensor.tangent_placement": "supported",
            "dtensor.cotangent_placement": "supported",
            "dtensor.output_placement": "supported",
        },
    }
    collapsed_status = {
        **settings,
        "dtensor.higher_order_diff_status": {"shard(0)": "supported"},
    }

    assert admit_distributed_candidate(
        Candidate("family", "slots", settings),
        policy=distributed_policy(),
    ) == (True, None)
    assert (
        admit_distributed_candidate(
            Candidate("family", "collapsed", collapsed_status),
            policy=distributed_policy(),
        )[0]
        is False
    )


def test_tensor_parallel_admission_requires_output_layout_propagation() -> None:
    policy = distributed_policy()
    valid = Candidate("family", "valid", valid_layout_settings())
    missing_layout = Candidate(
        "family",
        "missing-layout",
        {**valid_layout_settings(), "tp.output_projection": ""},
    )

    assert admit_distributed_candidate(valid, policy=policy) == (True, None)
    assert admit_distributed_candidate(missing_layout, policy=policy)[0] is False


def test_layout_admission_uses_mode_specific_fields() -> None:
    policy = distributed_policy()
    tensor_with_sequence_only = Candidate(
        "family",
        "tensor-with-sequence-only",
        {
            **valid_layout_settings(),
            "tp.output_projection": "",
            "sequence_parallel.enabled": "true",
            "sequence_parallel.norm_modules": ("norm",),
            "sequence_parallel.output_placement_policy": "preserve_sequence_shard",
        },
    )
    sequence = Candidate(
        "family",
        "sequence",
        valid_sequence_parallel_settings(),
    )
    sequence_missing_axis = Candidate(
        "family",
        "sequence-missing-axis",
        {**valid_sequence_parallel_settings(), "sequence_parallel.norm_modules": ""},
    )
    context = Candidate("family", "context", valid_context_parallel_settings())
    context_missing_layout = Candidate(
        "family",
        "context-missing-layout",
        {**valid_context_parallel_settings(), "context_parallel.rotate_method": ""},
    )

    assert (
        admit_distributed_candidate(tensor_with_sequence_only, policy=policy)[0]
        is False
    )
    assert admit_distributed_candidate(sequence, policy=policy) == (True, None)
    assert admit_distributed_candidate(sequence_missing_axis, policy=policy)[0] is False
    assert admit_distributed_candidate(context, policy=policy) == (True, None)
    assert (
        admit_distributed_candidate(context_missing_layout, policy=policy)[0] is False
    )


def test_reduce_rank_statuses_records_global_failure() -> None:
    passed = RankStatus(rank=0, status="passed", device="cuda:0")
    failed = RankStatus(
        rank=1,
        status="failed",
        device="cuda:1",
        error_type="RuntimeError",
        error="collective failed",
    )
    global_status = reduce_rank_statuses((passed, failed))

    assert global_status["status"] == "failed"
    assert global_status["failed_ranks"] == (1,)

    with pytest.raises(RuntimeError):
        reduce_rank_statuses(())


def test_distributed_identity_records_mesh_and_communication() -> None:
    identity = distributed_identity(
        device_mesh={"shape": (2,), "names": ("data",)},
        placements=({"parameter": "weight", "placement": "shard0"},),
        communication={"backend": "nccl"},
    )

    assert identity["adapter_id"] == "vptune.distributed"
    assert identity["device_mesh"]["shape"] == (2,)
    assert identity["communication"] == {"backend": "nccl"}


def test_distributed_selected_settings_must_match_across_ranks() -> None:
    selected = require_rank_selected_settings_agree((
        RankSelectedSettings(rank=0, settings={"distributed.strategy": "fsdp2"}),
        RankSelectedSettings(rank=1, settings={"distributed.strategy": "fsdp2"}),
    ))

    assert selected == {"distributed.strategy": "fsdp2"}

    with pytest.raises(MaterializationError):
        require_rank_selected_settings_agree((
            RankSelectedSettings(rank=0, settings={"distributed.strategy": "fsdp2"}),
            RankSelectedSettings(
                rank=1,
                settings={"distributed.strategy": "tensor_parallel"},
            ),
        ))

    with pytest.raises(MaterializationError):
        require_rank_selected_settings_agree(())


def test_distributed_record_contains_memory_surface_and_settings() -> None:
    record = distributed_record(
        identity=valid_distributed_identity(),
        expected_rank_count=2,
        rank_statuses=(
            RankStatus(rank=0, status="passed", device="cuda:0"),
            RankStatus(rank=1, status="passed", device="cuda:1"),
        ),
        rank_memory_samples=(
            Measurement(
                elapsed_seconds=1.0,
                peak_allocated_mib=11.0,
                peak_reserved_mib=19.0,
                post_allocated_mib=3.0,
                post_reserved_mib=4.0,
                rank=0,
                device="cuda:0",
            ),
            Measurement(
                elapsed_seconds=1.2,
                peak_allocated_mib=13.0,
                peak_reserved_mib=23.0,
                post_allocated_mib=5.0,
                post_reserved_mib=6.0,
                rank=1,
                device="cuda:1",
            ),
        ),
        rank_selected_settings=(
            RankSelectedSettings(rank=0, settings={"distributed.strategy": "fsdp2"}),
            RankSelectedSettings(rank=1, settings={"distributed.strategy": "fsdp2"}),
        ),
        global_parameter_surface={"names": ("weight",), "shapes": ((2, 2),)},
    )

    assert record["status"] == "passed"
    assert record["rank_count"] == 2
    assert record["failed_ranks"] == ()
    assert record["global_elapsed_seconds"] == pytest.approx(1.2)
    assert record["global_peak_allocated_mib"] == pytest.approx(13.0)
    assert record["global_peak_reserved_mib"] == pytest.approx(23.0)
    assert record["global_post_allocated_mib"] == pytest.approx(5.0)
    assert record["global_post_reserved_mib"] == pytest.approx(6.0)
    assert record["selected_settings"] == {"distributed.strategy": "fsdp2"}
    assert record["global_parameter_surface"] == {
        "names": ("weight",),
        "shapes": ((2, 2),),
    }
    assert record["rank_memory_samples"][1]["device"] == "cuda:1"


def test_distributed_record_requires_matching_rank_sets() -> None:
    with pytest.raises(MaterializationError):
        distributed_record(
            identity=valid_distributed_identity(),
            expected_rank_count=1,
            rank_statuses=(RankStatus(rank=0, status="passed", device="cuda:0"),),
            rank_memory_samples=(
                Measurement(
                    elapsed_seconds=1.0,
                    peak_allocated_mib=1.0,
                    peak_reserved_mib=1.0,
                    post_allocated_mib=0.0,
                    post_reserved_mib=0.0,
                    rank=1,
                    device="cuda:1",
                ),
            ),
            rank_selected_settings=(
                RankSelectedSettings(
                    rank=0, settings={"distributed.strategy": "fsdp2"}
                ),
            ),
            global_parameter_surface={},
        )

    with pytest.raises(MaterializationError):
        distributed_record(
            identity=valid_distributed_identity(),
            expected_rank_count=1,
            rank_statuses=(RankStatus(rank=0, status="passed", device="cuda:0"),),
            rank_memory_samples=(),
            rank_selected_settings=(
                RankSelectedSettings(
                    rank=0, settings={"distributed.strategy": "fsdp2"}
                ),
            ),
            global_parameter_surface={},
        )


def test_distributed_record_requires_expected_rank_count() -> None:
    with pytest.raises(MaterializationError, match="expected rank count"):
        distributed_record(
            identity=valid_distributed_identity(),
            expected_rank_count=2,
            rank_statuses=(RankStatus(rank=0, status="passed", device="cuda:0"),),
            rank_memory_samples=(
                Measurement(
                    elapsed_seconds=1.0,
                    peak_allocated_mib=1.0,
                    peak_reserved_mib=1.0,
                    post_allocated_mib=0.0,
                    post_reserved_mib=0.0,
                    rank=0,
                    device="cuda:0",
                ),
            ),
            rank_selected_settings=(
                RankSelectedSettings(
                    rank=0, settings={"distributed.strategy": "fsdp2"}
                ),
            ),
            global_parameter_surface={},
        )


def test_distributed_record_requires_identity_fields() -> None:
    with pytest.raises(MaterializationError, match="identity missing fields"):
        distributed_record(
            identity={},
            expected_rank_count=1,
            rank_statuses=(RankStatus(rank=0, status="passed", device="cuda:0"),),
            rank_memory_samples=(
                Measurement(
                    elapsed_seconds=1.0,
                    peak_allocated_mib=1.0,
                    peak_reserved_mib=1.0,
                    post_allocated_mib=0.0,
                    post_reserved_mib=0.0,
                    rank=0,
                    device="cuda:0",
                ),
            ),
            rank_selected_settings=(
                RankSelectedSettings(
                    rank=0, settings={"distributed.strategy": "fsdp2"}
                ),
            ),
            global_parameter_surface={},
        )


def test_distributed_record_requires_zero_based_rank_set() -> None:
    with pytest.raises(MaterializationError, match="contiguous from zero"):
        distributed_record(
            identity=valid_distributed_identity(),
            expected_rank_count=2,
            rank_statuses=(
                RankStatus(rank=1, status="passed", device="cuda:1"),
                RankStatus(rank=2, status="passed", device="cuda:2"),
            ),
            rank_memory_samples=(
                Measurement(
                    elapsed_seconds=1.0,
                    peak_allocated_mib=1.0,
                    peak_reserved_mib=1.0,
                    post_allocated_mib=0.0,
                    post_reserved_mib=0.0,
                    rank=1,
                    device="cuda:1",
                ),
                Measurement(
                    elapsed_seconds=1.0,
                    peak_allocated_mib=1.0,
                    peak_reserved_mib=1.0,
                    post_allocated_mib=0.0,
                    post_reserved_mib=0.0,
                    rank=2,
                    device="cuda:2",
                ),
            ),
            rank_selected_settings=(
                RankSelectedSettings(
                    rank=1, settings={"distributed.strategy": "fsdp2"}
                ),
                RankSelectedSettings(
                    rank=2, settings={"distributed.strategy": "fsdp2"}
                ),
            ),
            global_parameter_surface={},
        )

    with pytest.raises(MaterializationError, match="positive expected rank count"):
        distributed_record(
            identity=valid_distributed_identity(),
            expected_rank_count=0,
            rank_statuses=(RankStatus(rank=0, status="passed", device="cuda:0"),),
            rank_memory_samples=(
                Measurement(
                    elapsed_seconds=1.0,
                    peak_allocated_mib=1.0,
                    peak_reserved_mib=1.0,
                    post_allocated_mib=0.0,
                    post_reserved_mib=0.0,
                    rank=0,
                    device="cuda:0",
                ),
            ),
            rank_selected_settings=(
                RankSelectedSettings(
                    rank=0, settings={"distributed.strategy": "fsdp2"}
                ),
            ),
            global_parameter_surface={},
        )
