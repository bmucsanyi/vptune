import pytest

from vptune import AdmissionError, Candidate, MaterializationError, Measurement
from vptune.adapters.distributed import (
    DistributedAdmissionPolicy,
    RankSelectedSettings,
    RankStatus,
    admit_distributed_candidate,
    distributed_identity,
    distributed_record,
    distributed_sharding_axis,
    reduce_rank_statuses,
    require_rank_selected_settings_agree,
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
            "allowed_fsdp_hook_entry_policy": (hook_entry_policy,),
            "allowed_fsdp_sharding_granularity": ("root", "layerwise"),
            "allowed_fsdp_forward_prefetch": ("disabled", "next-forward"),
            "allowed_fsdp_backward_prefetch": ("disabled", "backward-pre"),
            "allowed_fsdp_reshard_after_forward": ("always", "never"),
            "allowed_fsdp_mixed_precision": ("none", "bf16", "fp16"),
            "allowed_fsdp_offload": (offload,),
        },
        dtensor={
            "allowed_to_local_grad_placement": ("redistribute",),
            "allowed_from_local_check": ("strict",),
            "allowed_uneven_shard_handling": ("reject",),
            "allowed_async_local_tensor_handling": ("sync",),
        },
        tensor_parallel={"allowed_tp_output_layout": (output_layout,)},
        sequence_parallel={
            "allowed_sp_sequence_axis": ("tokens",),
            "allowed_sp_output_layout": ("sequence",),
        },
        context_parallel={
            "allowed_cp_context_axis": ("tokens",),
            "allowed_cp_output_layout": ("context",),
        },
    )


def valid_fsdp_settings() -> dict[str, object]:
    return {
        "sharding": "fsdp2",
        "fsdp_hook_entry_points": ("root.forward",),
        "fsdp_hook_entry_policy": "root-forward",
        "fsdp_sharding_granularity": "root",
        "fsdp_forward_prefetch": "disabled",
        "fsdp_backward_prefetch": "disabled",
        "fsdp_reshard_after_forward": "always",
        "fsdp_mixed_precision": "none",
        "fsdp_offload": "none",
        "fsdp_bypasses_hooks": False,
        "fsdp_bottom_up_order": True,
        "fsdp_mutated_modules": (),
        "fsdp_collectives": {"all_gather": True, "reduce_scatter": True},
    }


def valid_layout_settings() -> dict[str, object]:
    return {
        "sharding": "tensor_parallel",
        "input_placements": ("shard(0)",),
        "output_placements": ("replicate",),
        "to_local_grad_placement": "redistribute",
        "from_local_check": "strict",
        "uneven_shard_handling": "reject",
        "async_local_tensor_handling": "sync",
        "tp_output_layout": "rowwise",
    }


def valid_sequence_parallel_settings() -> dict[str, object]:
    return {
        **valid_layout_settings(),
        "sharding": "sequence_parallel",
        "sp_sequence_axis": "tokens",
        "sp_output_layout": "sequence",
    }


def valid_context_parallel_settings() -> dict[str, object]:
    return {
        **valid_layout_settings(),
        "sharding": "context_parallel",
        "cp_context_axis": "tokens",
        "cp_output_layout": "context",
    }


def test_distributed_sharding_axis_validates_modes() -> None:
    policy = distributed_policy()
    axis = distributed_sharding_axis(("fsdp2", "tensor_parallel"), policy=policy)

    assert axis.adapter_id == "vptune.distributed"
    assert axis.allowed_values == ("fsdp2", "tensor_parallel")
    assert axis.signature()["has_admission_rule"] is True

    with pytest.raises(AdmissionError):
        distributed_sharding_axis(("single_device",), policy=policy)


def test_distributed_sharding_axis_records_admission_identity() -> None:
    first = distributed_sharding_axis(("fsdp2",), policy=distributed_policy())
    second = distributed_sharding_axis(
        ("fsdp2",),
        policy=distributed_policy(hook_entry_policy="layer-forward"),
    )

    assert first.signature()["identity"]["adapter_id"] == "vptune.distributed"
    assert first.signature()["identity"] != second.signature()["identity"]


def test_fsdp2_admission_requires_hook_entry_and_rejects_bypass() -> None:
    policy = distributed_policy()
    valid = Candidate("family", "valid", valid_fsdp_settings())
    missing_hook = Candidate(
        "family",
        "missing-hook",
        {**valid_fsdp_settings(), "fsdp_hook_entry_points": ()},
    )
    bypass = Candidate(
        "family",
        "bypass",
        {**valid_fsdp_settings(), "fsdp_bypasses_hooks": True},
    )

    assert admit_distributed_candidate(valid, policy=policy) == (True, None)
    assert admit_distributed_candidate(missing_hook, policy=policy)[0] is False
    assert admit_distributed_candidate(bypass, policy=policy)[0] is False


def test_fsdp2_admission_requires_policy_axes() -> None:
    policy = distributed_policy()
    valid = Candidate("family", "valid", valid_fsdp_settings())
    missing_prefetch_settings = dict(valid_fsdp_settings())
    missing_prefetch_settings.pop("fsdp_forward_prefetch")
    missing_prefetch = Candidate(
        "family",
        "missing-prefetch",
        missing_prefetch_settings,
    )
    offload_policy_mismatch = Candidate(
        "family",
        "offload-policy-mismatch",
        {**valid_fsdp_settings(), "fsdp_offload": "cpu"},
    )
    hook_policy_mismatch = Candidate(
        "family",
        "hook-policy-mismatch",
        {**valid_fsdp_settings(), "fsdp_hook_entry_policy": "layer-forward"},
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
        {**valid_layout_settings(), "to_local_grad_placement": "drop"},
    )

    assert admit_distributed_candidate(valid, policy=policy) == (True, None)
    assert admit_distributed_candidate(invalid, policy=policy)[0] is False


def test_tensor_parallel_admission_requires_output_layout_propagation() -> None:
    policy = distributed_policy()
    valid = Candidate("family", "valid", valid_layout_settings())
    missing_layout = Candidate(
        "family",
        "missing-layout",
        {**valid_layout_settings(), "tp_output_layout": ""},
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
            "tp_output_layout": "",
            "sp_sequence_axis": "tokens",
            "sp_output_layout": "sequence",
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
        {**valid_sequence_parallel_settings(), "sp_sequence_axis": ""},
    )
    context = Candidate("family", "context", valid_context_parallel_settings())
    context_missing_layout = Candidate(
        "family",
        "context-missing-layout",
        {**valid_context_parallel_settings(), "cp_output_layout": ""},
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
        RankSelectedSettings(rank=0, settings={"sharding": "fsdp2"}),
        RankSelectedSettings(rank=1, settings={"sharding": "fsdp2"}),
    ))

    assert selected == {"sharding": "fsdp2"}

    with pytest.raises(MaterializationError):
        require_rank_selected_settings_agree((
            RankSelectedSettings(rank=0, settings={"sharding": "fsdp2"}),
            RankSelectedSettings(
                rank=1,
                settings={"sharding": "tensor_parallel"},
            ),
        ))

    with pytest.raises(MaterializationError):
        require_rank_selected_settings_agree(())


def test_distributed_record_contains_memory_surface_and_settings() -> None:
    record = distributed_record(
        identity=distributed_identity(
            device_mesh={"shape": (2,), "names": ("data",)},
            placements=({"parameter": "weight", "placement": "shard0"},),
            communication={"backend": "nccl"},
        ),
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
            RankSelectedSettings(rank=0, settings={"sharding": "fsdp2"}),
            RankSelectedSettings(rank=1, settings={"sharding": "fsdp2"}),
        ),
        global_parameter_surface={"names": ("weight",), "shapes": ((2, 2),)},
    )

    assert record["status"] == "passed"
    assert record["rank_count"] == 2
    assert record["failed_ranks"] == ()
    assert record["global_peak_allocated_mib"] == pytest.approx(13.0)
    assert record["global_peak_reserved_mib"] == pytest.approx(23.0)
    assert record["global_post_allocated_mib"] == pytest.approx(5.0)
    assert record["global_post_reserved_mib"] == pytest.approx(6.0)
    assert record["selected_settings"] == {"sharding": "fsdp2"}
    assert record["global_parameter_surface"] == {
        "names": ("weight",),
        "shapes": ((2, 2),),
    }
    assert record["rank_memory_samples"][1]["device"] == "cuda:1"


def test_distributed_record_requires_matching_rank_sets() -> None:
    with pytest.raises(MaterializationError):
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
                    rank=1,
                    device="cuda:1",
                ),
            ),
            rank_selected_settings=(
                RankSelectedSettings(rank=0, settings={"sharding": "fsdp2"}),
            ),
            global_parameter_surface={},
        )

    with pytest.raises(MaterializationError):
        distributed_record(
            identity={},
            expected_rank_count=1,
            rank_statuses=(RankStatus(rank=0, status="passed", device="cuda:0"),),
            rank_memory_samples=(),
            rank_selected_settings=(
                RankSelectedSettings(rank=0, settings={"sharding": "fsdp2"}),
            ),
            global_parameter_surface={},
        )


def test_distributed_record_requires_expected_rank_count() -> None:
    with pytest.raises(MaterializationError, match="expected rank count"):
        distributed_record(
            identity={},
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
                RankSelectedSettings(rank=0, settings={"sharding": "fsdp2"}),
            ),
            global_parameter_surface={},
        )

    with pytest.raises(MaterializationError, match="positive expected rank count"):
        distributed_record(
            identity={},
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
                RankSelectedSettings(rank=0, settings={"sharding": "fsdp2"}),
            ),
            global_parameter_surface={},
        )
