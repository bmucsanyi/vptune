import dataclasses
import datetime
import queue
from pathlib import Path
from typing import Any, Self

import pytest
import torch
import torch.multiprocessing as mp

from vptune import AdmissionError, MaterializationError
from vptune.adapters import distributed as distributed_module
from vptune.adapters.distributed import (
    DistributedAdmissionPolicy,
    DistributedCommunicationBindings,
    DistributedContextParallelBindings,
    DistributedFSDPBindings,
    DistributedMeshBindings,
    DistributedPlacementBindings,
    DistributedProcessGroupBindings,
    DistributedRankReport,
    DistributedSequenceParallelBindings,
    DistributedStrategyBindings,
    DistributedTensorParallelBindings,
    RankCompileTiming,
    RankSelectedSettings,
    RankStatus,
    admit_distributed_candidate,
    build_dtensor_placement,
    collective_all_gather_into_tensor,
    distributed_axis_registry,
    distributed_identity,
    distributed_operation_factory,
    distributed_record,
    distributed_reference_check,
    distributed_runtime_config,
    distributed_strategy_applier,
    distributed_strategy_axis,
    initialize_process_group,
    named_modules_for_distributed_wrap,
    reduce_rank_statuses,
    require_rank_selected_settings_agree,
    resolve_process_group_backend,
    run_with_loss_parallel,
    wait_collective,
)
from vptune.axes.candidates import INTEGER_DOMAIN
from vptune.core import operators as ops
from vptune.ext import (
    Batch,
    BufferTree,
    Candidate,
    FullSizeRecord,
    Measurement,
    ModuleCallSpec,
    ObjectiveContext,
    ParameterTree,
    TensorTree,
)


class TinyDistributedScalarModule(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.w = torch.nn.Parameter(torch.tensor([2.0], dtype=torch.float64))

    def forward(self, scale: torch.Tensor) -> torch.Tensor:
        return (self.w * scale).sum()


class RecordingStrategyApplier:
    def __init__(self) -> None:
        self.calls = []

    def __call__(
        self,
        module: torch.nn.Module,
        candidate: Candidate,
    ) -> torch.nn.Module:
        self.calls.append(dict(candidate.settings))

        return module


class RecordingRankReporter:
    def __init__(self) -> None:
        self.calls = []

    def __call__(
        self,
        candidate: Candidate,
        samples: tuple[Measurement, ...],
    ) -> DistributedRankReport:
        self.calls.append((candidate.candidate_id, samples))

        return DistributedRankReport(
            rank_statuses=(RankStatus(rank=0, status="passed", device="cpu"),),
            rank_memory_samples=samples,
            rank_selected_settings=(
                RankSelectedSettings(rank=0, settings=dict(candidate.settings)),
            ),
        )


def distributed_policy(
    *,
    hook_entry_policy: str = "root-forward",
    fsdp_wrap_granularity: tuple[str, ...] = ("root", "transformer_block"),
    fsdp_reshard_after_forward: tuple[Any, ...] = (
        "true",
        "false",
        INTEGER_DOMAIN,
    ),
    output_layout: str = "rowwise",
    offload: str = "none",
    context_rotate_method: tuple[str, ...] = ("all_gather",),
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
            "allowed_fsdp.wrap_granularity": fsdp_wrap_granularity,
            "allowed_fsdp.forward_prefetch": ("disabled", "next-forward"),
            "allowed_fsdp.backward_prefetch": ("disabled", "backward-pre"),
            "allowed_fsdp.reshard_after_forward": fsdp_reshard_after_forward,
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
                "redistribute_to_declared_output",
            ),
        },
        context_parallel={
            "allowed_context_parallel.enabled": ("true",),
            "allowed_context_parallel.rotate_method": context_rotate_method,
            "allowed_context_parallel.sequence_dim": (1,),
        },
    )


def distributed_base_settings() -> dict[str, object]:
    return {
        "distributed.launch": "torchrun",
        "distributed.process_group_backend": "gloo",
        "distributed.local_rank_binding": "explicit_device_map",
        "distributed.mesh_shape": (2,),
        "distributed.mesh_dim_names": ("data",),
    }


def distributed_single_process_settings() -> dict[str, object]:
    return {
        "distributed.launch": "single_process",
        "distributed.process_group_backend": "gloo",
        "distributed.local_rank_binding": "explicit_device_map",
        "distributed.mesh_shape": (1,),
        "distributed.mesh_dim_names": ("data",),
    }


def valid_fsdp_settings() -> dict[str, object]:
    return {
        **distributed_base_settings(),
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
        **distributed_base_settings(),
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


def distributed_stateful_gradient_settings() -> dict[str, object]:
    return {
        **valid_fsdp_settings(),
        "gradient.path": "torch_autograd_grad",
        "call.path": "stateful_module",
        "call.params": "module_params",
        "call.buffers": "module_buffers",
        "call.tied_weights": "preserve_alias_groups",
        "call.parametrizations": "preserve_parametrizations",
        "call.buffer_mutation": "forbidden",
        "call.grad_mode": "grad_enabled",
        "call.return_type": "raw_tensor_tree",
    }


def distributed_bindings(
    events: list[dict[str, object]],
) -> DistributedStrategyBindings:
    return DistributedStrategyBindings(
        process_group=distributed_process_group_bindings(events),
        mesh=distributed_mesh_bindings(events),
        placements=distributed_placement_bindings(events),
        fsdp=distributed_fsdp_bindings(events),
        tensor_parallel=distributed_tensor_parallel_bindings(events),
        sequence_parallel=distributed_sequence_parallel_bindings(events),
        context_parallel=distributed_context_parallel_bindings(events),
        communication=DistributedCommunicationBindings(
            configure=lambda settings: events.append({
                "kind": "communication",
                "settings": dict(settings),
            })
        ),
        hybrid_order=("tensor_parallel", "fsdp2"),
    )


def distributed_process_group_bindings(
    events: list[dict[str, object]],
) -> DistributedProcessGroupBindings:
    def device_for_rank(binding: str, rank: int) -> str:
        events.append({"kind": "device_for_rank", "binding": binding, "rank": rank})

        return f"{binding}:{rank}"

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
    ) -> None:
        events.append({
            "kind": "process_group",
            "backend": backend,
            "init_method": init_method,
            "timeout": timeout,
            "world_size": world_size,
            "rank": rank,
            "store": store,
            "pg_options": pg_options,
            "device_id": device_id,
        })

    return DistributedProcessGroupBindings(
        init_process_group=init_process_group,
        is_ucc_available=lambda: False,
        init_method="env://",
        timeout="timeout",
        world_size=2,
        rank=1,
        store="store",
        pg_options="pg_options",
        device_for_rank=device_for_rank,
    )


def distributed_mesh_bindings(
    events: list[dict[str, object]],
) -> DistributedMeshBindings:
    mesh = object()

    def init_device_mesh(
        device_type: str,
        mesh_shape: tuple[int, ...],
        *,
        mesh_dim_names: tuple[str, ...],
    ) -> object:
        events.append({
            "kind": "mesh",
            "device_type": device_type,
            "mesh_shape": mesh_shape,
            "mesh_dim_names": mesh_dim_names,
        })

        return mesh

    return DistributedMeshBindings(
        init_device_mesh=init_device_mesh,
        device_type="cuda",
    )


def distributed_placement_bindings(
    events: list[dict[str, object]],
) -> DistributedPlacementBindings:
    def replicate() -> str:
        events.append({"kind": "placement", "placement": "replicate"})

        return "replicate"

    def shard(dim: int) -> str:
        events.append({"kind": "placement", "placement": "shard", "dim": dim})

        return f"shard-{dim}"

    def partial(reduce_op: str) -> str:
        events.append({
            "kind": "placement",
            "placement": "partial",
            "reduce_op": reduce_op,
        })

        return f"partial-{reduce_op}"

    return DistributedPlacementBindings(
        replicate=replicate,
        shard=shard,
        partial=partial,
        placement_specs={
            "dtensor.params_placement": {"shard_dim": 0},
            "dtensor.vector_placement": {},
            "dtensor.logits_placement": {},
            "dtensor.tangent_placement": {},
            "dtensor.cotangent_placement": {},
            "dtensor.output_placement": {},
        },
    )


def distributed_fsdp_bindings(
    events: list[dict[str, object]],
) -> DistributedFSDPBindings:
    sharded = torch.nn.Identity()

    def fully_shard(target: object, **kwargs: object) -> torch.nn.Module:
        events.append({"kind": "fully_shard", "target": target, **kwargs})

        return sharded

    def configure_forward_prefetch(
        module: torch.nn.Module,
        policy: str,
    ) -> None:
        events.append({
            "kind": "forward_prefetch",
            "module": module,
            "policy": policy,
        })

    def configure_backward_prefetch(
        module: torch.nn.Module,
        policy: str,
    ) -> None:
        events.append({
            "kind": "backward_prefetch",
            "module": module,
            "policy": policy,
        })

    def mixed_precision_policy(**kwargs: object) -> dict[str, object]:
        events.append({"kind": "mixed_precision", **kwargs})

        return {"mixed_precision": kwargs}

    def offload_policy() -> dict[str, object]:
        events.append({"kind": "offload", "offload": "none"})

        return {"offload": "none"}

    def cpu_offload_policy(*, pin_memory: bool) -> dict[str, object]:
        events.append({
            "kind": "offload",
            "offload": "cpu",
            "pin_memory": pin_memory,
        })

        return {"offload": "cpu", "pin_memory": pin_memory}

    def data_parallel_mesh_dims(
        *,
        shard: object,
        replicate: object,
    ) -> dict[str, object]:
        events.append({
            "kind": "dp_mesh_dims",
            "shard": shard,
            "replicate": replicate,
        })

        return {"shard": shard, "replicate": replicate}

    def shard_placement_fn(param: object) -> str:
        events.append({"kind": "shard_placement_fn", "param": param})

        return "shard-placement"

    return DistributedFSDPBindings(
        fully_shard=fully_shard,
        configure_forward_prefetch=configure_forward_prefetch,
        configure_backward_prefetch=configure_backward_prefetch,
        mixed_precision_policy=mixed_precision_policy,
        offload_policy=offload_policy,
        cpu_offload_policy=cpu_offload_policy,
        data_parallel_mesh_dims=data_parallel_mesh_dims,
        shard_placement_fns={"declared_fn": shard_placement_fn},
        ignored_params={},
        cpu_offload_pin_memory=True,
        hsdp_replicate_mesh_dims="replicate",
    )


def distributed_tensor_parallel_bindings(
    events: list[dict[str, object]],
) -> DistributedTensorParallelBindings:
    def parallelize_module(
        module: torch.nn.Module,
        device_mesh: object,
        parallelize_plan: dict[str, object],
        *,
        src_data_rank: int,
    ) -> torch.nn.Module:
        events.append({
            "kind": "parallelize_module",
            "module": module,
            "device_mesh": device_mesh,
            "parallelize_plan": dict(parallelize_plan),
            "src_data_rank": src_data_rank,
        })

        return module

    def colwise_parallel(**kwargs: object) -> tuple[str, dict[str, object]]:
        events.append({"kind": "colwise", **kwargs})

        return "colwise", kwargs

    def rowwise_parallel(**kwargs: object) -> tuple[str, dict[str, object]]:
        events.append({"kind": "rowwise", **kwargs})

        return "rowwise", kwargs

    def sequence_parallel(**kwargs: object) -> tuple[str, dict[str, object]]:
        events.append({"kind": "sequence", **kwargs})

        return "sequence", kwargs

    def prepare_module_input(**kwargs: object) -> tuple[str, dict[str, object]]:
        events.append({"kind": "prepare_input", **kwargs})

        return "prepare_input", kwargs

    def prepare_module_output(**kwargs: object) -> tuple[str, dict[str, object]]:
        events.append({"kind": "prepare_output", **kwargs})

        return "prepare_output", kwargs

    class LossParallel:
        def __enter__(self) -> None:
            events.append({"kind": "loss_parallel_enter"})

        def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
            events.append({"kind": "loss_parallel_exit"})

    def loss_parallel() -> LossParallel:
        return LossParallel()

    return DistributedTensorParallelBindings(
        parallelize_module=parallelize_module,
        colwise_parallel=colwise_parallel,
        rowwise_parallel=rowwise_parallel,
        prepare_module_input=prepare_module_input,
        prepare_module_output=prepare_module_output,
        loss_parallel=loss_parallel,
        module_paths_by_plan={
            "linear": {
                "tp.qkv_projection": "qkv",
                "tp.output_projection": "out",
                "tp.mlp_up_gate": "up",
                "tp.mlp_down": "down",
                "tp.embedding": "embed",
                "tp.lm_head": "lm_head",
            }
        },
        style_specs={
            "tp.qkv_projection": {
                "input_layouts": "dtensor.vector_placement",
                "output_layouts": "dtensor.tangent_placement",
                "use_local_output": False,
            },
            "tp.output_projection": {
                "input_layouts": "dtensor.tangent_placement",
                "output_layouts": "dtensor.output_placement",
                "use_local_output": True,
            },
            "tp.mlp_up_gate": {
                "input_layouts": "dtensor.vector_placement",
                "output_layouts": "dtensor.output_placement",
                "use_local_output": False,
            },
            "tp.mlp_down": {
                "input_layouts": "dtensor.output_placement",
                "output_layouts": "dtensor.output_placement",
                "use_local_output": True,
            },
            "tp.embedding": {
                "input_layouts": "dtensor.vector_placement",
                "output_layouts": "dtensor.output_placement",
                "use_local_output": True,
            },
            "tp.lm_head": {
                "input_layouts": "dtensor.output_placement",
                "output_layouts": "dtensor.logits_placement",
                "use_local_output": False,
            },
        },
        prepare_input_specs={
            "declared": {
                "module_path": "prepare_in",
                "input_layouts": "dtensor.vector_placement",
                "desired_input_layouts": "dtensor.tangent_placement",
                "input_kwarg_layouts": {"mask": "dtensor.logits_placement"},
                "desired_input_kwarg_layouts": {"mask": "dtensor.cotangent_placement"},
                "use_local_output": False,
            }
        },
        prepare_output_specs={
            "declared": {
                "module_path": "prepare_out",
                "output_layouts": "dtensor.tangent_placement",
                "desired_output_layouts": "dtensor.output_placement",
                "use_local_output": True,
            }
        },
        src_data_rank=0,
    )


def distributed_sequence_parallel_bindings(
    events: list[dict[str, object]],
) -> DistributedSequenceParallelBindings:
    def sequence_parallel(**kwargs: object) -> tuple[str, dict[str, object]]:
        events.append({"kind": "sequence", **kwargs})

        return "sequence", kwargs

    return DistributedSequenceParallelBindings(
        sequence_parallel=sequence_parallel,
        sequence_dim=1,
    )


def distributed_context_parallel_bindings(
    events: list[dict[str, object]],
) -> DistributedContextParallelBindings:
    def context_parallel(mesh: object, **kwargs: object) -> object:
        events.append({"kind": "context_parallel", "mesh": mesh, **kwargs})

        return object()

    return DistributedContextParallelBindings(
        context_parallel=context_parallel,
        buffers=(torch.tensor([1.0]), torch.tensor([2.0])),
        no_restore_buffers=(torch.tensor([2.0]),),
    )


def tensor_dict(tree: object) -> dict[str, torch.Tensor]:
    assert isinstance(tree, dict)
    result = {}

    for key, value in tree.items():
        assert isinstance(key, str)
        assert isinstance(value, torch.Tensor)
        result[key] = value

    return result


def gloo_all_gather_worker(
    rank: int,
    world_size: int,
    init_file: str,
    result_queue: Any,
) -> None:
    try:
        initialize_process_group(
            torch.distributed.init_process_group,
            backend="gloo",
            init_method=f"file://{init_file}",
            timeout=datetime.timedelta(seconds=20),
            world_size=world_size,
            rank=rank,
            store=None,
            pg_options=None,
            device_id=None,
        )
        local = torch.tensor([float(rank + 1)], dtype=torch.float32)
        gathered = torch.empty(world_size, dtype=torch.float32)
        collective_all_gather_into_tensor(
            torch.distributed.all_gather_into_tensor,
            gathered,
            local,
            group=None,
            async_op=False,
        )
        result_queue.put((
            rank,
            "passed",
            tuple(float(value) for value in gathered.tolist()),
            float(gathered.sum().item()),
        ))
    except (OSError, RuntimeError, ValueError) as error:
        result_queue.put((rank, "failed", type(error).__name__, str(error)))
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


def nccl_all_gather_worker(
    rank: int,
    world_size: int,
    init_file: str,
    result_queue: Any,
) -> None:
    try:
        values, total = run_nccl_all_gather(rank, world_size, init_file)
        result_queue.put((
            rank,
            "passed",
            values,
            total,
        ))
    except (OSError, RuntimeError, ValueError) as error:
        result_queue.put((rank, "failed", type(error).__name__, str(error)))
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


def run_nccl_all_gather(
    rank: int,
    world_size: int,
    init_file: str,
) -> tuple[tuple[float, ...], float]:
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    initialize_process_group(
        torch.distributed.init_process_group,
        backend="nccl",
        init_method=f"file://{init_file}",
        timeout=datetime.timedelta(seconds=20),
        world_size=world_size,
        rank=rank,
        store=None,
        pg_options=None,
        device_id=device,
    )
    local = torch.tensor([float(rank + 1)], dtype=torch.float32, device=device)
    gathered = torch.empty(world_size, dtype=torch.float32, device=device)
    collective_all_gather_into_tensor(
        torch.distributed.all_gather_into_tensor,
        gathered,
        local,
        group=None,
        async_op=False,
    )
    torch.cuda.synchronize(device)

    return (
        tuple(float(value) for value in gathered.cpu().tolist()),
        float(gathered.sum().cpu().item()),
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


def test_named_modules_for_distributed_wrap_returns_declared_order() -> None:
    first = torch.nn.Linear(2, 2)
    nested = torch.nn.Sequential(torch.nn.Linear(2, 2))
    model = torch.nn.Sequential(first, nested)

    selected = named_modules_for_distributed_wrap(model, ("1.0", "", "0"))

    assert selected == (nested[0], model, first)

    with pytest.raises(AdmissionError, match="missing"):
        named_modules_for_distributed_wrap(model, ("2",))


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


def test_distributed_redistribution_runs_before_output_schedule() -> None:
    events = []
    bindings = distributed_bindings(events)
    settings = {
        **valid_layout_settings(),
        "dtensor.redistribute_schedule": "before_output",
    }
    plan = distributed_module._distributed_redistribution(settings, bindings)
    output = RecordingDTensor()

    result = plan.before_output({"output": output})

    assert result == {"output": output.result}
    assert len(output.calls) == 1
    assert output.calls[0]["placements"] == ("replicate",)
    assert output.calls[0]["async_op"] is False
    assert output.calls[0]["forward_dtype"] is None
    assert output.calls[0]["backward_dtype"] is None


def test_distributed_redistribution_runs_before_forward_schedule() -> None:
    events = []
    bindings = distributed_bindings(events)
    settings = {
        **valid_layout_settings(),
        "dtensor.redistribute_schedule": "before_forward",
    }
    plan = distributed_module._distributed_redistribution(settings, bindings)
    param = RecordingDTensor()
    params = recording_parameter_tree(param)
    logits = RecordingDTensor()
    label = object()
    batch = {"logits": logits, "labels": label}
    vector_value = RecordingDTensor()
    vector = recording_tensor_tree(vector_value)

    redistributed_params = plan.before_forward_params(params)
    redistributed_batch = plan.before_forward_batch(batch)
    redistributed_vector = plan.before_forward_vector(vector)

    assert redistributed_params == {"w": param.result}
    assert redistributed_batch == {
        "logits": logits.result,
        "labels": label,
    }
    assert redistributed_vector == {"w": vector_value.result}
    assert param.calls[0]["placements"] == ("shard-0",)
    assert logits.calls[0]["placements"] == ("replicate",)
    assert vector_value.calls[0]["placements"] == ("replicate",)


@pytest.mark.parametrize(
    ("schedule", "method_name"),
    [
        ("before_backward", "before_backward_vector"),
        ("between_operator_parts", "between_operator_parts"),
    ],
)
def test_distributed_redistribution_runs_vector_boundary_schedules(
    schedule: str,
    method_name: str,
) -> None:
    bindings = distributed_bindings([])
    settings = {
        **valid_layout_settings(),
        "dtensor.redistribute_schedule": schedule,
    }
    plan = distributed_module._distributed_redistribution(settings, bindings)
    vector_value = RecordingDTensor()
    vector = recording_tensor_tree(vector_value)
    method = getattr(plan, method_name)

    result = method(vector)

    assert result == {"w": vector_value.result}
    assert vector_value.calls[0]["placements"] == ("replicate",)


def test_distributed_redistribution_requires_bindings_for_active_schedule() -> None:
    settings = {
        **valid_layout_settings(),
        "dtensor.redistribute_schedule": "before_output",
    }

    with pytest.raises(MaterializationError, match="requires strategy bindings"):
        distributed_module._distributed_redistribution(settings, None)


def test_gloo_process_group_all_gather_matches_logical_rank_output(
    tmp_path: Path,
) -> None:
    if not torch.distributed.is_available():
        pytest.skip("torch.distributed is unavailable")

    if not torch.distributed.is_gloo_available():
        pytest.skip("gloo backend is unavailable")

    world_size = 2
    init_file = str(tmp_path / "gloo_init")
    context = mp.get_context("spawn")
    result_queue = context.Queue()
    processes = tuple(
        context.Process(
            target=gloo_all_gather_worker,
            args=(rank, world_size, init_file, result_queue),
        )
        for rank in range(world_size)
    )

    for process in processes:
        process.start()

    try:
        results = [result_queue.get(timeout=30) for _ in range(world_size)]
    except queue.Empty as error:
        for process in processes:
            if process.is_alive():
                process.terminate()

        message = "distributed worker did not report"
        raise AssertionError(message) from error
    finally:
        for process in processes:
            process.join(timeout=30)

            if process.is_alive():
                process.terminate()
                process.join(timeout=5)

    assert all(process.exitcode == 0 for process in processes)
    assert tuple(sorted(results)) == (
        (0, "passed", (1.0, 2.0), 3.0),
        (1, "passed", (1.0, 2.0), 3.0),
    )


def test_nccl_process_group_single_rank_all_gather_matches_logical_rank_output(
    tmp_path: Path,
) -> None:
    if not torch.distributed.is_available():
        pytest.skip("torch.distributed is unavailable")

    if not torch.distributed.is_nccl_available():
        pytest.skip("nccl backend is unavailable")

    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for NCCL")

    if torch.distributed.is_initialized():
        pytest.skip("process group already initialized")

    torch.cuda.set_device(0)
    device = torch.device("cuda", 0)
    initialize_process_group(
        torch.distributed.init_process_group,
        backend="nccl",
        init_method=f"file://{tmp_path / 'nccl_single_init'}",
        timeout=datetime.timedelta(seconds=20),
        world_size=1,
        rank=0,
        store=None,
        pg_options=None,
        device_id=device,
    )

    try:
        local = torch.tensor([1.0], dtype=torch.float32, device=device)
        gathered = torch.empty(1, dtype=torch.float32, device=device)
        collective_all_gather_into_tensor(
            torch.distributed.all_gather_into_tensor,
            gathered,
            local,
            group=None,
            async_op=False,
        )
        torch.cuda.synchronize(device)

        assert tuple(float(value) for value in gathered.cpu().tolist()) == (1.0,)
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


def test_nccl_process_group_all_gather_matches_logical_rank_output(
    tmp_path: Path,
) -> None:
    if not torch.distributed.is_available():
        pytest.skip("torch.distributed is unavailable")

    if not torch.distributed.is_nccl_available():
        pytest.skip("nccl backend is unavailable")

    if torch.cuda.device_count() < 2:
        pytest.skip("two CUDA devices are required for two-rank NCCL")

    world_size = 2
    init_file = str(tmp_path / "nccl_init")
    context = mp.get_context("spawn")
    result_queue = context.Queue()
    processes = tuple(
        context.Process(
            target=nccl_all_gather_worker,
            args=(rank, world_size, init_file, result_queue),
        )
        for rank in range(world_size)
    )

    for process in processes:
        process.start()

    try:
        results = [result_queue.get(timeout=30) for _ in range(world_size)]
    except queue.Empty as error:
        for process in processes:
            if process.is_alive():
                process.terminate()

        message = "distributed worker did not report"
        raise AssertionError(message) from error
    finally:
        for process in processes:
            process.join(timeout=30)

            if process.is_alive():
                process.terminate()
                process.join(timeout=5)

    assert all(process.exitcode == 0 for process in processes)
    assert tuple(sorted(results)) == (
        (0, "passed", (1.0, 2.0), 3.0),
        (1, "passed", (1.0, 2.0), 3.0),
    )


def test_wait_collective_waits_on_work_handle() -> None:
    work = RecordingWork("done")

    assert wait_collective(work) == "done"
    assert work.waited is True


class RecordingDTensor(torch.Tensor):
    def __new__(cls) -> Self:
        return torch.Tensor._make_subclass(cls, torch.zeros(1), False)

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


def recording_parameter_tree(tensor: torch.Tensor) -> ParameterTree:
    return {"w": tensor}


def recording_tensor_tree(tensor: torch.Tensor) -> TensorTree:
    return {"w": tensor}


def test_distributed_strategy_axis_rejects_unsupported_modes() -> None:
    policy = distributed_policy()

    with pytest.raises(AdmissionError):
        distributed_strategy_axis(("single_device",), policy=policy)


def test_distributed_adapter_registry_admits_owned_axes_and_strategy_fields() -> None:
    registry = distributed_axis_registry(
        ("fsdp2", "tensor_parallel"),
        policy=distributed_policy(),
    )
    fsdp = Candidate("family", "fsdp", valid_fsdp_settings())
    tensor_parallel = Candidate("family", "tensor-parallel", valid_layout_settings())
    orphan_fsdp = Candidate(
        "family",
        "orphan-fsdp",
        {"fsdp.wrap_granularity": "root"},
    )

    assert registry.admit(fsdp).admission_status == "passed"
    assert registry.admit(tensor_parallel).admission_status == "passed"
    assert registry.admit(orphan_fsdp).admission_status == "failed"


def test_distributed_adapter_registry_admits_direct_fsdp_reshard_group_sizes() -> None:
    registry = distributed_axis_registry(
        ("fsdp2",),
        policy=distributed_policy(),
    )
    direct_group_size = Candidate(
        "family",
        "direct-group-size",
        {**valid_fsdp_settings(), "fsdp.reshard_after_forward": 3},
    )
    old_marker = Candidate(
        "family",
        "old-marker",
        {
            **valid_fsdp_settings(),
            "fsdp.reshard_after_forward": "positive_integer_group_size",
        },
    )
    zero_group_size = Candidate(
        "family",
        "zero-group-size",
        {**valid_fsdp_settings(), "fsdp.reshard_after_forward": 0},
    )
    bool_group_size = Candidate(
        "family",
        "bool-group-size",
        {**valid_fsdp_settings(), "fsdp.reshard_after_forward": True},
    )
    policy_without_integer_domain = distributed_axis_registry(
        ("fsdp2",),
        policy=distributed_policy(fsdp_reshard_after_forward=("true", "false")),
    )

    assert registry.admit(direct_group_size).admission_status == "passed"
    assert registry.admit(old_marker).admission_status == "failed"
    assert registry.admit(zero_group_size).admission_status == "failed"
    assert registry.admit(bool_group_size).admission_status == "failed"
    assert (
        policy_without_integer_domain.admit(direct_group_size).admission_status
        == "failed"
    )


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


def test_distributed_admission_accepts_single_gpu_and_hybrid() -> None:
    policy = distributed_policy()
    single_gpu = Candidate(
        "family",
        "single",
        {
            **distributed_single_process_settings(),
            "distributed.strategy": "single_gpu",
        },
    )
    hybrid = Candidate(
        "family",
        "hybrid",
        {
            **valid_fsdp_settings(),
            **valid_layout_settings(),
            "distributed.strategy": "hybrid",
        },
    )
    hybrid_missing_tensor_plan = Candidate(
        "family",
        "hybrid-missing-tp-plan",
        {
            **valid_fsdp_settings(),
            **valid_layout_settings(),
            "distributed.strategy": "hybrid",
            "tp.plan": "",
        },
    )

    assert admit_distributed_candidate(single_gpu, policy=policy) == (True, None)
    assert admit_distributed_candidate(hybrid, policy=policy) == (True, None)
    assert (
        admit_distributed_candidate(hybrid_missing_tensor_plan, policy=policy)[0]
        is False
    )


def test_distributed_admission_requires_runtime_identity_fields() -> None:
    policy = distributed_policy()
    missing_launch_settings = dict(valid_fsdp_settings())
    missing_launch_settings.pop("distributed.launch")
    missing_mesh_name = Candidate(
        "family",
        "missing-mesh-name",
        {
            **valid_layout_settings(),
            "distributed.mesh_shape": (2, 1),
            "distributed.mesh_dim_names": ("data",),
        },
    )

    assert (
        admit_distributed_candidate(
            Candidate("family", "missing-launch", missing_launch_settings),
            policy=policy,
        )[0]
        is False
    )
    assert admit_distributed_candidate(missing_mesh_name, policy=policy)[0] is False


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


def test_fsdp2_admission_validates_declared_sets_and_communication() -> None:
    policy = distributed_policy()
    invalid_ignored_params = Candidate(
        "family",
        "bad-ignored-params",
        {**valid_fsdp_settings(), "fsdp.ignored_params": ("w", 1)},
    )
    invalid_dp_dims = Candidate(
        "family",
        "bad-dp-dims",
        {**valid_fsdp_settings(), "fsdp.dp_mesh_dims": ()},
    )
    invalid_overlap = Candidate(
        "family",
        "bad-overlap",
        {**valid_fsdp_settings(), "comm.overlap": "async_magic"},
    )
    invalid_bucket = Candidate(
        "family",
        "bad-bucket",
        {**valid_fsdp_settings(), "comm.collective_bucket_size": 0},
    )

    assert (
        admit_distributed_candidate(invalid_ignored_params, policy=policy)[0] is False
    )
    assert admit_distributed_candidate(invalid_dp_dims, policy=policy)[0] is False
    assert admit_distributed_candidate(invalid_overlap, policy=policy)[0] is False
    assert admit_distributed_candidate(invalid_bucket, policy=policy)[0] is False


def test_dtensor_admission_requires_gradient_placement_policy() -> None:
    policy = distributed_policy()
    valid = Candidate("family", "valid", valid_layout_settings())
    invalid = Candidate(
        "family",
        "invalid",
        {**valid_layout_settings(), "dtensor.to_local_grad_placement": "drop"},
    )
    invalid_placement = Candidate(
        "family",
        "invalid-placement",
        {**valid_layout_settings(), "dtensor.params_placement": "scatter"},
    )

    assert admit_distributed_candidate(valid, policy=policy) == (True, None)
    assert admit_distributed_candidate(invalid, policy=policy)[0] is False
    assert admit_distributed_candidate(invalid_placement, policy=policy)[0] is False


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


def test_distributed_strategy_applier_lowers_fsdp2_row_settings() -> None:
    events = []
    model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Linear(2, 2))
    settings = {
        **valid_fsdp_settings(),
        **distributed_base_settings(),
        "fsdp.hook_entry_points": ("0.forward",),
        "fsdp.wrap_granularity": "transformer_block",
        "fsdp.reshard_after_forward": 3,
        "fsdp.shard_placement_fn": "declared_fn",
        "fsdp.mp_policy.param_dtype": "bf16",
        "fsdp.mp_policy.reduce_dtype": "fp16",
        "fsdp.mp_policy.output_dtype": "fp32",
        "fsdp.mp_policy.cast_forward_inputs": "true",
        "fsdp.offload_policy": "cpu",
        "comm.overlap": "both",
        "comm.prefetch": "forward",
        "comm.collective_bucket_size": 1024,
    }
    applier = distributed_strategy_applier(distributed_bindings(events))

    result = applier(model, Candidate("gradient", "fsdp", settings))

    assert isinstance(result, torch.nn.Module)
    assert events[0] == {
        "kind": "device_for_rank",
        "binding": "explicit_device_map",
        "rank": 1,
    }
    assert events[1]["kind"] == "process_group"
    assert events[1]["backend"] == "gloo"
    assert events[1]["device_id"] == "explicit_device_map:1"
    assert events[2] == {
        "kind": "mesh",
        "device_type": "cuda",
        "mesh_shape": (2,),
        "mesh_dim_names": ("data",),
    }
    assert {
        "kind": "communication",
        "settings": {
            "comm.overlap": "both",
            "comm.prefetch": "forward",
            "comm.collective_bucket_size": 1024,
        },
    } in events
    assert {
        "kind": "mixed_precision",
        "param_dtype": torch.bfloat16,
        "reduce_dtype": torch.float16,
        "output_dtype": torch.float32,
        "cast_forward_inputs": True,
    } in events
    assert {
        "kind": "offload",
        "offload": "cpu",
        "pin_memory": True,
    } in events
    assert {
        "kind": "dp_mesh_dims",
        "shard": ("data",),
        "replicate": None,
    } in events
    fully_shard_call = next(event for event in events if event["kind"] == "fully_shard")

    assert fully_shard_call["target"] == [model[0]]
    assert fully_shard_call["reshard_after_forward"] == 3
    assert fully_shard_call["shard_placement_fn"]


def test_distributed_strategy_applier_lowers_fsdp2_prefetch_settings() -> None:
    events = []
    model = torch.nn.Sequential(torch.nn.Linear(2, 2))
    settings = {
        **valid_fsdp_settings(),
        **distributed_base_settings(),
        "fsdp.forward_prefetch": "next-forward",
        "fsdp.backward_prefetch": "backward-pre",
    }
    candidate = Candidate("gradient", "fsdp-prefetch", settings)
    applier = distributed_strategy_applier(distributed_bindings(events))

    assert admit_distributed_candidate(candidate, policy=distributed_policy()) == (
        True,
        None,
    )
    result = applier(model, candidate)

    assert isinstance(result, torch.nn.Module)
    sharded = next(
        event["module"] for event in events if event["kind"] == "forward_prefetch"
    )
    assert {
        "kind": "forward_prefetch",
        "module": sharded,
        "policy": "next-forward",
    } in events
    assert {
        "kind": "backward_prefetch",
        "module": sharded,
        "policy": "backward-pre",
    } in events


def test_distributed_strategy_applier_lowers_fsdp2_block_group_wrap() -> None:
    events = []
    model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Linear(2, 2))
    settings = {
        **valid_fsdp_settings(),
        **distributed_base_settings(),
        "fsdp.hook_entry_points": ("0.forward",),
        "fsdp.wrap_granularity": "block_group",
    }
    candidate = Candidate("gradient", "fsdp-block-group", settings)
    policy = distributed_policy(fsdp_wrap_granularity=("block_group",))
    applier = distributed_strategy_applier(distributed_bindings(events))

    assert admit_distributed_candidate(candidate, policy=policy) == (True, None)
    result = applier(model, candidate)

    assert isinstance(result, torch.nn.Module)
    fully_shard_call = next(event for event in events if event["kind"] == "fully_shard")
    assert fully_shard_call["target"] == [model[0]]


def test_distributed_strategy_applier_lowers_cuda_local_rank_binding() -> None:
    events = []
    model = torch.nn.Sequential(torch.nn.Linear(2, 2))
    settings = {
        **valid_fsdp_settings(),
        **distributed_base_settings(),
        "distributed.local_rank_binding": "cuda_local_rank",
    }
    applier = distributed_strategy_applier(distributed_bindings(events))

    result = applier(model, Candidate("gradient", "fsdp", settings))

    assert isinstance(result, torch.nn.Module)
    assert {
        "kind": "device_for_rank",
        "binding": "cuda_local_rank",
        "rank": 1,
    } in events


def test_distributed_strategy_applier_lowers_context_parallel_row_settings() -> None:
    events = []
    model = torch.nn.Sequential(torch.nn.Linear(2, 2))
    settings = {
        **valid_context_parallel_settings(),
        **distributed_base_settings(),
        "comm.overlap": "all_gather_overlap",
        "comm.prefetch": "both",
        "comm.collective_bucket_size": 2048,
    }
    applier = distributed_strategy_applier(distributed_bindings(events))

    result = applier(model, Candidate("gradient", "context", settings))

    assert result is model
    assert {"kind": "placement", "placement": "shard", "dim": 0} in events
    assert {"kind": "placement", "placement": "replicate"} in events
    context_call = next(
        event for event in events if event["kind"] == "context_parallel"
    )
    parallelize_call = next(
        event for event in events if event["kind"] == "parallelize_module"
    )

    assert context_call["rotate_method"] == "all_gather"
    assert context_call["buffer_seq_dims"] == (1, 1)
    assert parallelize_call["src_data_rank"] == 0
    assert set(parallelize_call["parallelize_plan"]) == {
        "qkv",
        "out",
        "up",
        "down",
        "prepare_in",
        "prepare_out",
    }
    qkv_style = parallelize_call["parallelize_plan"]["qkv"]
    prepare_style = parallelize_call["parallelize_plan"]["prepare_in"]

    assert qkv_style[1]["input_layouts"] == "replicate"
    assert qkv_style[1]["output_layouts"] == "replicate"
    assert prepare_style[1]["input_kwarg_layouts"] == {"mask": "replicate"}


def test_distributed_strategy_applier_lowers_context_parallel_all_to_all() -> None:
    events = []
    model = torch.nn.Sequential(torch.nn.Linear(2, 2))
    settings = {
        **valid_context_parallel_settings(),
        **distributed_base_settings(),
        "context_parallel.rotate_method": "all_to_all",
    }
    candidate = Candidate("gradient", "context-all-to-all", settings)
    policy = distributed_policy(context_rotate_method=("all_to_all",))
    applier = distributed_strategy_applier(distributed_bindings(events))

    assert admit_distributed_candidate(candidate, policy=policy) == (True, None)
    result = applier(model, candidate)

    assert result is model
    context_call = next(
        event for event in events if event["kind"] == "context_parallel"
    )
    assert context_call["rotate_method"] == "all_to_all"


def test_distributed_strategy_applier_lowers_reduce_scatter_overlap() -> None:
    events = []
    model = torch.nn.Sequential(torch.nn.Linear(2, 2))
    settings = {
        **valid_fsdp_settings(),
        **distributed_base_settings(),
        "comm.overlap": "reduce_scatter_overlap",
    }
    applier = distributed_strategy_applier(distributed_bindings(events))

    result = applier(model, Candidate("gradient", "fsdp", settings))

    assert isinstance(result, torch.nn.Module)
    assert {
        "kind": "communication",
        "settings": {"comm.overlap": "reduce_scatter_overlap"},
    } in events


def test_distributed_strategy_applier_lowers_sequence_parallel_modules() -> None:
    events = []
    model = torch.nn.Sequential(torch.nn.Linear(2, 2))
    settings = {
        **valid_sequence_parallel_settings(),
        **distributed_single_process_settings(),
    }
    applier = distributed_strategy_applier(distributed_bindings(events))

    result = applier(model, Candidate("gradient", "sequence", settings))

    assert result is model
    assert {"kind": "sequence", "sequence_dim": 1, "use_local_output": False} in events
    parallelize_call = next(
        event for event in events if event["kind"] == "parallelize_module"
    )

    assert "norm" in parallelize_call["parallelize_plan"]


def test_distributed_strategy_applier_lowers_sequence_parallel_output_policy() -> None:
    events = []
    model = torch.nn.Sequential(torch.nn.Linear(2, 2))
    settings = {
        **valid_sequence_parallel_settings(),
        **distributed_single_process_settings(),
        "sequence_parallel.output_placement_policy": "redistribute_to_declared_output",
    }
    applier = distributed_strategy_applier(distributed_bindings(events))

    result = applier(model, Candidate("gradient", "sequence", settings))

    assert result is model
    assert {"kind": "sequence", "sequence_dim": 1, "use_local_output": True} in events


def test_distributed_strategy_applier_rejects_undeclared_tp_layout_key() -> None:
    events = []
    bindings = distributed_bindings(events)
    tensor_parallel = bindings.tensor_parallel
    assert tensor_parallel is not None
    style_specs = dict(tensor_parallel.style_specs)
    style_specs["tp.qkv_projection"] = {
        **dict(style_specs["tp.qkv_projection"]),
        "input_layouts": "dtensor.missing_placement",
    }
    applier = distributed_strategy_applier(
        dataclasses.replace(
            bindings,
            tensor_parallel=dataclasses.replace(
                tensor_parallel,
                style_specs=style_specs,
            ),
        )
    )
    settings = {
        **valid_layout_settings(),
        **distributed_single_process_settings(),
    }

    with pytest.raises(MaterializationError, match="undeclared placement"):
        applier(
            torch.nn.Sequential(torch.nn.Linear(2, 2)),
            Candidate("g", "tp", settings),
        )


def test_distributed_operation_factory_wraps_loss_parallel_from_bindings() -> None:
    events = []
    model = TinyDistributedScalarModule()
    factory = distributed_operation_factory(
        ops.gradient("gradient", "loss", aggregation="sum"),
        model=model,
        params=dict(model.named_parameters()),
        buffers=dict(model.named_buffers()),
        module_call=ModuleCallSpec(positional_batch_keys=("scale",)),
        strategy_bindings=distributed_bindings(events),
    )
    settings = {
        **distributed_stateful_gradient_settings(),
        **distributed_single_process_settings(),
        "distributed.strategy": "single_gpu",
        "tp.loss_parallel": "true",
    }
    candidate = Candidate(
        "gradient",
        "distributed-loss-parallel",
        settings,
        admission_status="passed",
    )
    output = factory(
        candidate,
        {"scale": torch.tensor([4.0], dtype=torch.float64)},
        {"w": torch.tensor([1.0], dtype=torch.float64)},
    )()
    output_map = tensor_dict(output)

    torch.testing.assert_close(
        output_map["w"], torch.tensor([4.0], dtype=torch.float64)
    )
    assert {"kind": "loss_parallel_enter"} in events
    assert {"kind": "loss_parallel_exit"} in events


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


def test_distributed_operation_factory_applies_strategy_and_runs_module() -> None:
    model = TinyDistributedScalarModule()
    applier = RecordingStrategyApplier()
    factory = distributed_operation_factory(
        ops.gradient("gradient", "loss", aggregation="sum"),
        model=model,
        strategy_applier=applier,
        params=dict(model.named_parameters()),
        buffers=dict(model.named_buffers()),
        module_call=ModuleCallSpec(positional_batch_keys=("scale",)),
    )
    candidate = Candidate(
        "gradient",
        "distributed-gradient",
        distributed_stateful_gradient_settings(),
        admission_status="passed",
    )
    output = factory(
        candidate,
        {"scale": torch.tensor([4.0], dtype=torch.float64)},
        {"w": torch.tensor([1.0], dtype=torch.float64)},
    )()
    output_map = tensor_dict(output)

    assert applier.calls == [dict(candidate.settings)]
    torch.testing.assert_close(
        output_map["w"], torch.tensor([4.0], dtype=torch.float64)
    )


def test_distributed_operation_factory_delegates_dtensor_layout_to_strategy() -> None:
    model = TinyDistributedScalarModule()
    applier = RecordingStrategyApplier()
    factory = distributed_operation_factory(
        ops.gradient("gradient", "loss", aggregation="sum"),
        model=model,
        strategy_applier=applier,
        params=dict(model.named_parameters()),
        buffers=dict(model.named_buffers()),
        module_call=ModuleCallSpec(positional_batch_keys=("scale",)),
    )
    settings = {
        **valid_layout_settings(),
        "gradient.path": "torch_autograd_grad",
        "call.path": "stateful_module",
        "call.params": "module_params",
        "call.buffers": "module_buffers",
        "call.tied_weights": "preserve_alias_groups",
        "call.parametrizations": "preserve_parametrizations",
        "call.buffer_mutation": "forbidden",
        "call.grad_mode": "grad_enabled",
        "call.return_type": "raw_tensor_tree",
        "layout.params": "dtensor",
        "layout.vector": "per_shard",
        "layout.output": "dtensor",
    }
    candidate = Candidate(
        "gradient",
        "distributed-dtensor-layout",
        settings,
        admission_status="passed",
    )
    output = factory(
        candidate,
        {"scale": torch.tensor([4.0], dtype=torch.float64)},
        {"w": torch.tensor([1.0], dtype=torch.float64)},
    )()
    output_map = tensor_dict(output)

    assert applier.calls == [settings]
    torch.testing.assert_close(
        output_map["w"], torch.tensor([4.0], dtype=torch.float64)
    )


def test_distributed_reference_check_uses_single_device_anchor() -> None:
    reference_model = TinyDistributedScalarModule()

    def scalar_objective(
        params: ParameterTree,
        buffers: BufferTree,
        batch: Batch,
        context: ObjectiveContext,
    ) -> torch.Tensor:
        assert buffers == {}
        assert context.family == "gradient"

        return (params["w"] * batch["scale"]).sum()

    check = distributed_reference_check(
        ops.gradient("gradient", "loss", aggregation="sum"),
        reference_model=reference_model,
        params=dict(reference_model.named_parameters()),
        buffers=dict(reference_model.named_buffers()),
        module_call=ModuleCallSpec(positional_batch_keys=("scale",)),
        thresholds={
            "max_abs_diff": 0.0,
            "max_rel_diff": 0.0,
            "directional_abs_diff": 1e-9,
            "directional_rel_diff": 1e-9,
        },
        scalar_objectives={"loss": scalar_objective},
    )
    result = check(
        Candidate(
            "gradient",
            "distributed-gradient",
            distributed_stateful_gradient_settings(),
            admission_status="passed",
        ),
        {"scale": torch.tensor([4.0], dtype=torch.float64)},
        {"w": torch.tensor([1.0], dtype=torch.float64)},
    )

    assert result.measurements["max_abs_diff"] == pytest.approx(0.0)


def test_distributed_runtime_config_records_rank_selection_metadata() -> None:
    model = TinyDistributedScalarModule()
    reference_model = TinyDistributedScalarModule()
    applier = RecordingStrategyApplier()
    reporter = RecordingRankReporter()

    def scalar_objective(
        params: ParameterTree,
        buffers: BufferTree,
        batch: Batch,
        context: ObjectiveContext,
    ) -> torch.Tensor:
        assert buffers == {}
        assert context.family == "gradient"

        return (params["w"] * batch["scale"]).sum()

    candidate = Candidate(
        "gradient",
        "distributed-gradient",
        distributed_stateful_gradient_settings(),
        admission_status="passed",
    )
    runtime = distributed_runtime_config(
        ops.gradient("gradient", "loss", aggregation="sum"),
        model=model,
        reference_model=reference_model,
        strategy_applier=applier,
        rank_reporter=reporter,
        identity=valid_distributed_identity(),
        expected_rank_count=1,
        global_parameter_surface={"names": ("w",), "shapes": ((1,),)},
        params=dict(model.named_parameters()),
        buffers=dict(model.named_buffers()),
        candidates=(candidate,),
        thresholds={
            "max_abs_diff": 0.0,
            "max_rel_diff": 0.0,
            "directional_abs_diff": 1e-9,
            "directional_rel_diff": 1e-9,
        },
        objective_signature={"case": "distributed-runtime"},
        module_call=ModuleCallSpec(positional_batch_keys=("scale",)),
        axis_registry=None,
        scalar_objectives={"loss": scalar_objective},
    )
    batch = {"scale": torch.tensor([4.0], dtype=torch.float64)}
    vector = {"w": torch.tensor([1.0], dtype=torch.float64)}
    output = runtime.operation_factory(candidate, batch, vector)()
    samples = (
        Measurement(
            elapsed_seconds=2.0,
            peak_allocated_mib=3.0,
            peak_reserved_mib=5.0,
            post_allocated_mib=1.0,
            post_reserved_mib=2.0,
            rank=0,
            device="cpu",
        ),
    )

    assert runtime.full_size_check is not None
    metadata = runtime.full_size_check(
        candidate,
        ((batch, vector),),
        (output,),
        samples,
    )
    record = FullSizeRecord(
        family="gradient",
        candidate_id="distributed-gradient",
        status="passed",
        input_signature={},
        candidate_settings=dict(candidate.settings),
        generator_id=candidate.generator_id,
        generator_version=candidate.generator_version,
    )
    selected = runtime.materializer(candidate, record)
    selected_map = tensor_dict(selected(batch, vector))

    assert reporter.calls == [("distributed-gradient", samples)]
    assert metadata["global_elapsed_seconds"] == pytest.approx(2.0)
    assert metadata["distributed_rank_count"] == 1
    assert metadata["distributed_status"] == "passed"
    assert runtime.identity()["full_size_check"] is not None
    torch.testing.assert_close(
        selected_map["w"], torch.tensor([4.0], dtype=torch.float64)
    )


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
    assert record["selection_metadata"] == {
        "global_elapsed_seconds": pytest.approx(1.2)
    }
    assert record["rank_compile_timings"] == ()
    assert record["rank_memory_samples"][1]["device"] == "cuda:1"


def test_distributed_record_contains_compiled_selection_metadata() -> None:
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
            RankSelectedSettings(
                rank=0,
                settings={
                    "compile.enabled": "true",
                    "distributed.strategy": "fsdp2",
                },
            ),
            RankSelectedSettings(
                rank=1,
                settings={
                    "compile.enabled": "true",
                    "distributed.strategy": "fsdp2",
                },
            ),
        ),
        global_parameter_surface={"names": ("weight",), "shapes": ((2, 2),)},
        rank_compile_timings=(
            RankCompileTiming(
                rank=0,
                compile_time_seconds=9.0,
                steady_elapsed_seconds=0.8,
                recompile_count=1,
            ),
            RankCompileTiming(
                rank=1,
                compile_time_seconds=12.0,
                steady_elapsed_seconds=0.7,
                recompile_count=2,
            ),
        ),
    )

    assert record["selection_metadata"] == {
        "global_elapsed_seconds": pytest.approx(1.2),
        "global_compile_time_seconds": pytest.approx(12.0),
        "global_steady_elapsed_seconds": pytest.approx(0.8),
        "recompile_count": 2,
    }
    assert record["rank_compile_timings"][1]["rank"] == 1


def test_distributed_record_rejects_bad_compile_timing_rank_sets() -> None:
    with pytest.raises(MaterializationError, match="compile timing ranks"):
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
                    rank=0,
                    device="cuda:0",
                ),
            ),
            rank_selected_settings=(
                RankSelectedSettings(
                    rank=0,
                    settings={
                        "compile.enabled": "true",
                        "distributed.strategy": "fsdp2",
                    },
                ),
            ),
            global_parameter_surface={},
            rank_compile_timings=(
                RankCompileTiming(
                    rank=1,
                    compile_time_seconds=1.0,
                    steady_elapsed_seconds=1.0,
                    recompile_count=0,
                ),
            ),
        )


def test_distributed_record_rejects_compile_timing_on_eager_rows() -> None:
    with pytest.raises(MaterializationError, match="compiled distributed rows"):
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
                    rank=0,
                    device="cuda:0",
                ),
            ),
            rank_selected_settings=(
                RankSelectedSettings(
                    rank=0,
                    settings={"distributed.strategy": "fsdp2"},
                ),
            ),
            global_parameter_surface={},
            rank_compile_timings=(
                RankCompileTiming(
                    rank=0,
                    compile_time_seconds=1.0,
                    steady_elapsed_seconds=1.0,
                    recompile_count=0,
                ),
            ),
        )


def test_distributed_record_requires_compile_timing_for_compiled_rows() -> None:
    with pytest.raises(MaterializationError, match="compile timing"):
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
                    rank=0,
                    device="cuda:0",
                ),
            ),
            rank_selected_settings=(
                RankSelectedSettings(
                    rank=0,
                    settings={
                        "compile.enabled": "true",
                        "distributed.strategy": "fsdp2",
                    },
                ),
            ),
            global_parameter_surface={},
        )


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
