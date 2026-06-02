"""Distributed adapter helpers."""

import dataclasses
from collections.abc import Mapping, Sequence
from typing import Any

from vptune.candidates import AxisDescriptor
from vptune.data import PACKAGE_VERSION, Candidate, Measurement
from vptune.errors import AdmissionError, MaterializationError
from vptune.identities import to_json_value

DISTRIBUTED_SHARDING_MODES = (
    "fsdp2",
    "tensor_parallel",
    "sequence_parallel",
    "context_parallel",
)
LAYOUT_SHARDING_MODES = (
    "tensor_parallel",
    "sequence_parallel",
    "context_parallel",
)


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


def distributed_sharding_axis(
    modes: Sequence[str],
    *,
    policy: DistributedAdmissionPolicy,
) -> AxisDescriptor:
    """Return a distributed sharding axis.

    Raises:
        AdmissionError: If a sharding mode is unsupported.
    """
    unsupported = tuple(
        mode for mode in modes if mode not in DISTRIBUTED_SHARDING_MODES
    )

    if unsupported:
        message = f"unsupported distributed sharding modes: {unsupported}"
        raise AdmissionError(message)

    return AxisDescriptor(
        name="distributed_sharding",
        settings_keys=("sharding",),
        allowed_values=tuple(modes),
        optional_settings_keys=(
            "fsdp_hook_entry_points",
            "fsdp_hook_entry_policy",
            "fsdp_sharding_granularity",
            "fsdp_forward_prefetch",
            "fsdp_backward_prefetch",
            "fsdp_reshard_after_forward",
            "fsdp_mixed_precision",
            "fsdp_offload",
            "fsdp_bypasses_hooks",
            "fsdp_bottom_up_order",
            "fsdp_mutated_modules",
            "fsdp_collectives",
            "input_placements",
            "output_placements",
            "dtensor_module_class",
            "to_local_grad_placement",
            "from_local_check",
            "uneven_shard_handling",
            "async_local_tensor_handling",
            "higher_order_diff_status",
            "tp_output_layout",
            "sp_sequence_axis",
            "sp_output_layout",
            "cp_context_axis",
            "cp_output_layout",
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
    sharding = candidate.settings.get("sharding")

    if sharding == "fsdp2":
        return _admit_fsdp2(candidate.settings, policy)

    if sharding in LAYOUT_SHARDING_MODES:
        return _admit_layout_sharding(candidate.settings, policy)

    return False, f"unsupported distributed sharding mode: {sharding}"


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
        _non_empty_string_sequence(settings, "fsdp_hook_entry_points"),
        _fsdp2_policy_error(settings, policy),
        _fsdp2_hook_bypass_error(settings),
        _required_bool(settings, "fsdp_bottom_up_order"),
        _string_sequence(settings, "fsdp_mutated_modules"),
        _non_empty_mapping(settings, "fsdp_collectives"),
    ))


def _fsdp2_policy_error(
    settings: Mapping[str, Any],
    policy: DistributedAdmissionPolicy,
) -> str | None:
    return _policy_fields_error(
        settings,
        policy.fsdp2,
        (
            "fsdp_hook_entry_policy",
            "fsdp_sharding_granularity",
            "fsdp_forward_prefetch",
            "fsdp_backward_prefetch",
            "fsdp_reshard_after_forward",
            "fsdp_mixed_precision",
            "fsdp_offload",
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
    if settings.get("fsdp_bypasses_hooks") is not False:
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
            "dtensor_module_class",
            "to_local_grad_placement",
            "from_local_check",
            "uneven_shard_handling",
            "async_local_tensor_handling",
        ),
    )

    if policy_error is not None:
        return policy_error

    return _higher_order_diff_status_error(settings, policy)


def _higher_order_diff_status_error(
    settings: Mapping[str, Any],
    policy: DistributedAdmissionPolicy,
) -> str | None:
    status = settings.get("higher_order_diff_status")

    if not isinstance(status, Mapping):
        return "higher_order_diff_status must be a mapping"

    expected = _higher_order_diff_status_keys(settings)

    if set(status) != set(expected):
        return (
            "higher_order_diff_status must cover every input and output placement slot"
        )

    allowed = policy.dtensor.get("allowed_higher_order_diff_status")

    if not isinstance(allowed, tuple):
        return "higher_order_diff_status policy must declare allowed values"

    for slot, value in status.items():
        if not isinstance(value, str) or value not in allowed:
            return f"higher_order_diff_status is not allowed for {slot}: {value}"

    return None


def _higher_order_diff_status_keys(settings: Mapping[str, Any]) -> tuple[str, ...]:
    input_placements = tuple(settings["input_placements"])
    output_placements = tuple(settings["output_placements"])

    return tuple(
        f"input_placements[{index}]" for index, _ in enumerate(input_placements)
    ) + tuple(
        f"output_placements[{index}]" for index, _ in enumerate(output_placements)
    )


def _layout_placement_error(settings: Mapping[str, Any]) -> str | None:
    for key in ("input_placements", "output_placements"):
        placement_error = _non_empty_string_sequence(settings, key)

        if placement_error is not None:
            return placement_error

    return None


def _layout_mode_error(
    settings: Mapping[str, Any],
    policy: DistributedAdmissionPolicy,
) -> str | None:
    sharding = settings.get("sharding")

    if sharding == "tensor_parallel":
        return _allowed_policy(
            settings,
            "tp_output_layout",
            policy.tensor_parallel,
        )

    if sharding == "sequence_parallel":
        return _policy_fields_error(
            settings,
            policy.sequence_parallel,
            ("sp_sequence_axis", "sp_output_layout"),
        )

    if sharding == "context_parallel":
        return _policy_fields_error(
            settings,
            policy.context_parallel,
            ("cp_context_axis", "cp_output_layout"),
        )

    return f"unsupported layout sharding mode: {sharding}"


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

    if not isinstance(value, str) or not value:
        return f"{key} must be a non-empty string"

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
