import dataclasses
import math
from collections.abc import Callable, Hashable, Mapping, Sequence
from pathlib import Path
from typing import Any

import autobatch
import pytest
import torch

import vptune as vp
import vptune.adapters as vpa
from vptune import autobatch_bridge
from vptune.adapters.curvlinops import curvlinops_operation_factory
from vptune.checks import (
    ReferenceFailedError,
    assert_tree_close,
    thresholds_for_measurements,
    validate_thresholds,
)
from vptune.data import FullSizeRecord, Measurement
from vptune.identities import module_identity, stable_hash
from vptune.io import read_record, write_record
from vptune.measure import (
    CPUMemoryBackend,
    measure_once,
    measure_operation,
    run_candidate,
)
from vptune.schemas import compute_record_owner_hash, record_current
from vptune.select import memory_stable, select_cohort, select_family
from vptune.tensor_tree import tree_from_leaves, tree_leaves, tree_map, tree_signature


class OneBatchData:
    @staticmethod
    def signature() -> Mapping[str, object]:
        return {"case": "one_batch"}

    @staticmethod
    def reference_batch(
        family: str,
        check_name: str,
    ) -> Mapping[str, object]:
        return {"family": family, "check": check_name, "source": "reference"}

    @staticmethod
    def probe_batches(family: str) -> tuple[Mapping[str, object], ...]:
        return ({"family": family, "source": "probe"},)


class OneVectorProvider:
    @staticmethod
    def signature() -> Mapping[str, object]:
        return {"case": "one_vector"}

    @staticmethod
    def reference_vectors(family: str) -> torch.Tensor:
        assert family

        return torch.tensor([1.0])

    @staticmethod
    def probe_vectors(family: str) -> tuple[torch.Tensor, ...]:
        assert family

        return (torch.tensor([1.0]),)


class TwoProbeData:
    @staticmethod
    def signature() -> Mapping[str, object]:
        return {"case": "two_probe"}

    @staticmethod
    def reference_batch(
        family: str,
        check_name: str,
    ) -> Mapping[str, object]:
        return {"family": family, "check": check_name, "source": "reference"}

    @staticmethod
    def probe_batches(family: str) -> tuple[Mapping[str, object], ...]:
        return (
            {"family": family, "source": "probe", "index": 0},
            {"family": family, "source": "probe", "index": 1},
        )


class TwoVectorProvider:
    @staticmethod
    def signature() -> Mapping[str, object]:
        return {"case": "two_vector"}

    @staticmethod
    def reference_vectors(family: str) -> torch.Tensor:
        assert family

        return torch.tensor([1.0])

    @staticmethod
    def probe_vectors(family: str) -> tuple[torch.Tensor, ...]:
        assert family

        return (torch.tensor([1.0]), torch.tensor([2.0]))


class SequenceClock:
    def __init__(self, values: tuple[float, ...]) -> None:
        self.values = values
        self.index = 0

    def __call__(self) -> float:
        value = self.values[self.index]
        self.index += 1

        return value


def reference_passed() -> vp.ReferenceResult:
    return vp.ReferenceResult(
        "tree_close",
        {"max_abs_diff": 1e-6},
        {"max_abs_diff": 0.0},
    )


def cpu_target(timing_policy: vp.TimingPolicy | None = None) -> vp.Target:
    policy = vp.TimingPolicy() if timing_policy is None else timing_policy

    return vp.Target(
        devices=("cpu",),
        accelerator="cpu",
        allowed_dtypes=("float64", "float32", "bfloat16", "float16"),
        allowed_attention_impls=(),
        allowed_sharding_modes=("single_device",),
        timing_policy=policy,
        selection_policy=vp.SelectionPolicy(),
        determinism_policy={},
        environment_capture={"runtime": "test"},
    )


def materialize_candidate_impl(
    candidate: vp.Candidate,
    record: vp.FullSizeRecord,
) -> vp.CandidateOperation:
    def operation() -> torch.Tensor:
        return torch.tensor([float(candidate.settings.get("scale", 1.0))])

    assert record.candidate_id == candidate.candidate_id

    return operation


materialize_candidate = vp.CallableMaterializer(
    "tests.materialize_candidate",
    "1",
    {},
    materialize_candidate_impl,
)


def replay_context_for_plan(
    plan: vp.Plan,
    *,
    validation_required: bool = False,
) -> vp.ReplayContext:
    return vp.ReplayContext(
        input_signature=plan.input_signature,
        family_input_signatures={
            family: record.input_signature for family, record in plan.records.items()
        },
        materializer_identities=plan.materializer_identities(),
        selection_policy=plan.policy,
        target_identity=plan.target_identity,
        runtime_identities=plan.selected_runtime_identities(),
        adapter_identities=plan.selected_adapter_identities(),
        validator_identities=plan.selected_validator_identities(),
        validation_required=validation_required,
        validation_order=plan.validation_order,
    )


def saved_plan_rows(
    run_dir: Path,
    plan: vp.Plan,
) -> tuple[tuple[vp.FullSizeRecord, ...], tuple[vp.CheckRecord, ...]]:
    full_size_rows = tuple(
        vp.full_size_record_from_json(
            read_record(
                run_dir
                / "full_size"
                / record.family
                / record.candidate_id
                / f"{record.candidate_spec_hash}.json"
            )
        )
        for record in plan.full_size_records
    )
    check_rows = tuple(
        vp.check_record_from_json(
            read_record(
                run_dir
                / "references"
                / record.family
                / record.candidate_id
                / record.candidate_spec_hash
                / f"{record.name}.json"
            )
        )
        for record in plan.check_records
    )

    return full_size_rows, check_rows


def saved_candidate_rows(
    run_dir: Path,
    plan: vp.Plan,
) -> tuple[Mapping[str, object], ...]:
    return tuple(
        read_record(
            run_dir
            / "candidates"
            / record.family
            / record.candidate_id
            / f"{record.candidate_spec_hash}.json"
        )
        for record in plan.full_size_records
    )


def candidate_records_for_plan(plan: vp.Plan) -> tuple[Mapping[str, object], ...]:
    def candidate_for_record(record: vp.FullSizeRecord) -> vp.Candidate:
        selected_candidate = plan.selected.get(record.family)

        if (
            selected_candidate is not None
            and selected_candidate.candidate_spec_hash() == record.candidate_spec_hash
        ):
            return selected_candidate

        return vp.Candidate(
            family=record.family,
            candidate_id=record.candidate_id,
            settings=dict(record.candidate_settings),
            dependency_identities={
                family: dict(identity)
                for family, identity in record.dependency_identities.items()
            },
            cohort_assignment=dict(record.cohort_assignment),
            admission_status="passed",
            generator_id=record.generator_id,
            generator_version=record.generator_version,
        )

    return tuple(
        vp.candidate_record_to_json(
            candidate_for_record(record), record.input_signature
        )
        for record in plan.full_size_records
    )


def test_candidate_record_round_trips_migration_source_id() -> None:
    candidate = vp.Candidate(
        "family",
        "row",
        {"scale": 1.0},
        admission_status="passed",
        migration_source_id="pilot-row-1",
    )
    row = vp.candidate_record_to_json(candidate, {"case": "migration-source"})
    replayed = vp.candidate_record_from_json(row)

    assert row["migration_source_id"] == "pilot-row-1"
    assert replayed.migration_source_id == "pilot-row-1"
    assert replayed.candidate_spec_hash() == candidate.candidate_spec_hash()


def test_owner_hash_changes_on_identity_inputs() -> None:
    first = {
        "parameter_order": ("weight", "bias"),
        "data": {"slice": "a"},
        "target": {"device": "cpu"},
        "generator_version": "1",
    }
    second = {
        "parameter_order": ("bias", "weight"),
        "data": {"slice": "a"},
        "target": {"device": "cpu"},
        "generator_version": "1",
    }

    assert stable_hash(first) == stable_hash(dict(first))
    assert stable_hash(first) != stable_hash(second)


def test_adapter_namespace_exports_adapter_helpers() -> None:
    assert vpa.CurvLinOpsAdmitter
    assert vpa.CurvLinOpsFisherMCSemantics
    assert vpa.RankStatus
    assert vpa.PilotReadiness
    assert vpa.transformers_attention_axis


def test_module_identity_records_tied_parameters_and_devices() -> None:
    class TiedModule(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            shared = torch.nn.Parameter(torch.ones(2))
            self.first = shared
            self.second = shared
            self.register_buffer("scale", torch.ones(1))

    identity = module_identity(TiedModule())

    assert identity["tied_parameter_groups"] == (("first", "second"),)
    assert identity["parameters"][0]["device"] == "cpu"
    assert identity["buffers"][0]["device"] == "cpu"

    preserved = vp.parameter_surface(TiedModule(), tied_weights="preserve")
    deduplicated = vp.parameter_surface(TiedModule(), tied_weights="deduplicate")

    assert preserved.names == ("first", "second")
    assert deduplicated.names == ("first",)

    with pytest.raises(RuntimeError):
        vp.parameter_surface(TiedModule(), tied_weights="unknown")


def test_threshold_logic() -> None:
    thresholds = thresholds_for_measurements(
        {"max_abs_diff": 1e-4, "max_rel_diff": 2.0},
        {"model_dtype": "float16"},
    )

    assert thresholds["max_abs_diff"] >= 0.25 * float(torch.finfo(torch.float16).eps)
    validate_thresholds({"max_abs_diff": 1e-5, "max_rel_diff": 10.0}, thresholds)
    validate_thresholds({"max_abs_diff": 10.0, "max_rel_diff": 1e-5}, thresholds)

    with pytest.raises(ReferenceFailedError):
        validate_thresholds({"max_abs_diff": 10.0, "max_rel_diff": 10.0}, thresholds)

    with pytest.raises(ReferenceFailedError):
        validate_thresholds({"max_abs_diff": math.nan}, {"max_abs_diff": 1e-4})

    assert_tree_close(
        torch.tensor([1.0]),
        torch.tensor([1.0]),
        thresholds={"max_abs_diff": 1e-6, "max_rel_diff": 1e-6},
    )

    with pytest.raises(ReferenceFailedError):
        assert_tree_close(
            torch.tensor([1.0]),
            torch.tensor([0.0]),
            thresholds={"max_abs_diff": 1e-6, "max_rel_diff": 1e-6},
        )

    with pytest.raises(RuntimeError):
        assert_tree_close(
            torch.zeros(1, 2),
            torch.zeros(2),
            thresholds={"max_abs_diff": 1e-6, "max_rel_diff": 1e-6},
        )

    assert_tree_close(
        torch.tensor([1.0], dtype=torch.float32),
        torch.tensor([1.0 + 1e-5], dtype=torch.float32),
        settings={"model_dtype": "float16"},
        thresholds={"max_abs_diff": 0.0, "max_rel_diff": 0.0},
    )


def test_gradient_jvp_vjp_hvp_anchors() -> None:
    params = torch.tensor([0.2, -0.3, 0.5], dtype=torch.float64)
    vector = torch.tensor([0.7, -0.2, 0.1], dtype=torch.float64)
    cotangent = torch.tensor([1.3, -0.4, 0.8], dtype=torch.float64)

    def scalar(input_params: torch.Tensor) -> torch.Tensor:
        return input_params.pow(3).sum() + 0.5 * input_params.pow(2).sum()

    def function(input_params: torch.Tensor) -> torch.Tensor:
        return torch.stack((
            input_params[0] * input_params[1],
            input_params[1].sin(),
            input_params[2].pow(2),
        ))

    gradient = vp.gradient_anchor(scalar, params)
    expected_gradient = 3.0 * params.pow(2) + params

    assert torch.allclose(gradient, expected_gradient)
    assert torch.allclose(
        vp.jvp_anchor(function, params, vector),
        vp.finite_difference_jvp(function, params, vector, epsilon=1e-6),
        atol=1e-6,
    )
    assert vp.vjp_dot_identity_error(function, params, vector, cotangent) < 1e-12
    assert torch.allclose(
        vp.hvp_reverse_over_reverse_anchor(scalar, params, vector),
        vp.finite_difference_hvp(scalar, params, vector, epsilon=1e-6),
        atol=1e-6,
    )
    assert torch.allclose(
        vp.vhp_anchor(scalar, params, vector),
        vp.hvp_reverse_over_reverse_anchor(scalar, params, vector),
        atol=1e-12,
    )


def test_tree_gradient_and_hvp_return_zero_for_disconnected_leaves() -> None:
    params = {
        "active": torch.tensor([2.0], dtype=torch.float64),
        "disconnected": torch.tensor([3.0], dtype=torch.float64),
    }
    vector = {
        "active": torch.tensor([0.5], dtype=torch.float64),
        "disconnected": torch.tensor([7.0], dtype=torch.float64),
    }

    def scalar(tree: dict[str, torch.Tensor]) -> torch.Tensor:
        return tree["active"].pow(2).sum()

    gradient = vp.gradient_anchor(scalar, params)
    hvp = vp.hvp_reverse_over_reverse_anchor(scalar, params, vector)

    assert isinstance(gradient, dict)
    assert isinstance(hvp, dict)
    assert torch.allclose(gradient["active"], torch.tensor([4.0], dtype=torch.float64))
    assert torch.allclose(
        gradient["disconnected"],
        torch.zeros(1, dtype=torch.float64),
    )
    assert torch.allclose(hvp["active"], torch.tensor([1.0], dtype=torch.float64))
    assert torch.allclose(hvp["disconnected"], torch.zeros(1, dtype=torch.float64))


def test_tensor_tree_preserves_mapping_insertion_order() -> None:
    tree = {
        "second": torch.tensor([2.0]),
        "first": torch.tensor([1.0]),
    }
    mapped = tree_map(lambda tensor: tensor + 1.0, tree)
    rebuilt = tree_from_leaves(
        tree,
        (torch.tensor([20.0]), torch.tensor([10.0])),
    )
    signature = tree_signature(tree)

    assert isinstance(mapped, dict)
    assert isinstance(rebuilt, dict)
    assert tuple(mapped) == ("second", "first")
    assert [float(leaf.item()) for leaf in tree_leaves(tree)] == [2.0, 1.0]
    assert torch.equal(tree_leaves(rebuilt)[0], torch.tensor([20.0]))
    assert torch.equal(tree_leaves(rebuilt)[1], torch.tensor([10.0]))
    assert signature["items"][0]["key"] == "second"
    assert signature["items"][1]["key"] == "first"

    with pytest.raises(RuntimeError, match="too few"):
        tree_from_leaves(tree, (torch.tensor([20.0]),))

    with pytest.raises(RuntimeError, match="too many"):
        tree_from_leaves(
            tree,
            (
                torch.tensor([20.0]),
                torch.tensor([10.0]),
                torch.tensor([30.0]),
            ),
        )


def test_dense_ggn_fisher_and_metric_anchors() -> None:
    params = torch.tensor([0.3, -0.2], dtype=torch.float64)
    vector = torch.tensor([0.4, -0.7], dtype=torch.float64)
    jacobian = torch.tensor([[1.0, 2.0], [-1.0, 1.0]], dtype=torch.float64)
    loss_hessian = torch.diag(torch.tensor([3.0, 5.0], dtype=torch.float64))

    def function(input_params: torch.Tensor) -> torch.Tensor:
        return torch.stack((
            input_params[0] + 2.0 * input_params[1],
            -input_params[0] + input_params[1],
        ))

    expected_ggn = jacobian.T @ (loss_hessian @ (jacobian @ vector))

    assert torch.allclose(vp.dense_jacobian_anchor(function, params), jacobian)
    assert torch.allclose(
        vp.ggnvp_dense_anchor(function, loss_hessian, params, vector),
        expected_ggn,
    )

    score_gradients = torch.tensor(
        [[1.0, 0.0], [2.0, -1.0], [0.5, 3.0]],
        dtype=torch.float64,
    )
    expected_fisher = score_gradients.T @ (score_gradients @ vector) / 3.0

    assert torch.allclose(
        vp.fisher_vp_dense_anchor(
            score_gradients,
            vector,
            normalization=3.0,
        ),
        expected_fisher,
    )
    assert torch.allclose(
        vp.empirical_fisher_vp_dense_anchor(
            score_gradients,
            vector,
            normalization=3.0,
        ),
        expected_fisher,
    )

    metric = torch.tensor([[4.0, 1.0], [1.0, 3.0]], dtype=torch.float64)
    inverse_product = vp.dense_metric_inverse_multiply(metric, vector)

    assert torch.allclose(vp.dense_metric_multiply(metric, vector), metric @ vector)
    assert torch.allclose(vp.dense_metric_multiply(metric, inverse_product), vector)
    assert torch.allclose(
        vp.dense_metric_inner(metric, vector, vector),
        vector @ (metric @ vector),
    )
    assert vp.dense_metric_inverse_residual(metric, inverse_product, vector) < 1e-12


def test_measurement_timing_policy() -> None:
    calls = {"count": 0}

    def operation() -> torch.Tensor:
        calls["count"] += 1

        return torch.tensor(float(calls["count"]))

    class FakeClock:
        def __init__(self) -> None:
            self.value = 0.0

        def __call__(self) -> float:
            current = self.value
            self.value += 10.0

            return current

    samples, output, probe = measure_operation(
        operation,
        timing_policy=vp.TimingPolicy(),
        memory_backend=CPUMemoryBackend(),
        clock=FakeClock(),
    )

    assert len(samples) == 5
    assert len(probe) == 1
    assert calls["count"] == 8
    assert torch.equal(output, torch.tensor(8.0))
    assert all(sample.device == "cpu" for sample in samples)


def test_measurement_reuses_long_probe_as_measured_sample() -> None:
    calls = {"count": 0}

    def operation() -> torch.Tensor:
        calls["count"] += 1

        return torch.tensor(float(calls["count"]))

    samples, output, probe = measure_operation(
        operation,
        timing_policy=vp.TimingPolicy(),
        memory_backend=CPUMemoryBackend(),
        clock=SequenceClock((0.0, 700.0)),
    )

    assert len(samples) == 1
    assert len(probe) == 1
    assert calls["count"] == 1
    assert torch.equal(output, torch.tensor(1.0))
    assert math.isclose(samples[0].elapsed_seconds, 700.0)


def test_run_candidate_records_runtime_failures() -> None:
    candidate = vp.Candidate("family", "row", {})

    def operation() -> torch.Tensor:
        message = "failed row"
        raise RuntimeError(message)

    record = run_candidate(
        candidate,
        {"case": "runtime"},
        operation,
        timing_policy=vp.TimingPolicy(),
        memory_backend=CPUMemoryBackend(),
        clock=SequenceClock((0.0, 2.0)),
    )

    assert record.status == "failed"
    assert record.error_type == "RuntimeError"
    assert record.error == "failed row"
    assert record.reference_passed
    assert record.input_signature == {"case": "runtime"}
    assert record.timing_samples[0].elapsed_seconds == pytest.approx(2.0)
    assert record.memory_samples[0].elapsed_seconds == pytest.approx(2.0)


def test_run_candidate_records_cuda_oom_failures() -> None:
    candidate = vp.Candidate("family", "row", {})

    def operation() -> torch.Tensor:
        message = "cuda oom"
        raise torch.cuda.OutOfMemoryError(message)

    record = run_candidate(
        candidate,
        {"case": "oom"},
        operation,
        timing_policy=vp.TimingPolicy(),
        memory_backend=CPUMemoryBackend(),
        clock=SequenceClock((0.0, 3.0)),
    )

    assert record.status == "failed"
    assert record.error_type == "OutOfMemoryError"
    assert record.error == "cuda oom"
    assert record.reference_passed
    assert record.input_signature == {"case": "oom"}
    assert record.timing_samples[0].elapsed_seconds == pytest.approx(3.0)
    assert record.memory_samples[0].elapsed_seconds == pytest.approx(3.0)


def test_measurement_cleans_memory_backend_after_runtime_failure() -> None:
    class RecordingBackend:
        def __init__(self) -> None:
            self.cleanup_calls = 0

        def prepare(self) -> None:
            pass

        @staticmethod
        def sample() -> tuple[Measurement, ...]:
            return (
                Measurement(
                    elapsed_seconds=0.0,
                    peak_allocated_mib=0.0,
                    peak_reserved_mib=0.0,
                    post_allocated_mib=0.0,
                    post_reserved_mib=0.0,
                ),
            )

        def synchronize(self) -> None:
            pass

        def cleanup(self) -> None:
            self.cleanup_calls += 1

    backend = RecordingBackend()

    def operation() -> torch.Tensor:
        message = "failed measurement"
        raise RuntimeError(message)

    with pytest.raises(RuntimeError):
        measure_once(operation, memory_backend=backend)

    assert backend.cleanup_calls == 2


def _record(
    candidate: vp.Candidate,
    *,
    elapsed: tuple[float, ...],
    reserved: tuple[float, ...],
    input_signature: Mapping[str, object],
    status: str = "passed",
    reference_passed: bool = True,
) -> FullSizeRecord:
    samples = tuple(
        Measurement(
            elapsed_seconds=time,
            peak_allocated_mib=memory,
            peak_reserved_mib=memory,
            post_allocated_mib=0.0,
            post_reserved_mib=0.0,
        )
        for time, memory in zip(elapsed, reserved, strict=True)
    )

    return FullSizeRecord(
        family=candidate.family,
        candidate_id=candidate.candidate_id,
        status=status,
        input_signature=input_signature,
        candidate_settings=candidate.settings,
        generator_id=candidate.generator_id,
        generator_version=candidate.generator_version,
        owner_hash=compute_record_owner_hash(
            record_type="full_size",
            family=candidate.family,
            candidate_id=candidate.candidate_id,
            input_signature=input_signature,
            candidate_settings=candidate.settings,
            candidate_spec_hash=candidate.candidate_spec_hash(),
            dependency_identities=candidate.dependency_identities,
            generator_id=candidate.generator_id,
            generator_version=candidate.generator_version,
        ),
        candidate_spec_hash=candidate.candidate_spec_hash(),
        timing_samples=samples,
        memory_samples=samples,
        dependency_identities=dict(candidate.dependency_identities),
        reference_passed=reference_passed,
    )


def _check_record(
    candidate: vp.Candidate,
    *,
    input_signature: Mapping[str, object],
) -> vp.CheckRecord:
    record = vp.CheckRecord(
        family=candidate.family,
        candidate_id=candidate.candidate_id,
        name="tree_close",
        status="passed",
        input_signature=input_signature,
        candidate_settings=dict(candidate.settings),
        thresholds={"max_abs_diff": 1e-6},
        measurements={"max_abs_diff": 0.0},
        generator_id=candidate.generator_id,
        generator_version=candidate.generator_version,
        owner_hash=compute_record_owner_hash(
            record_type="reference",
            family=candidate.family,
            candidate_id=candidate.candidate_id,
            check_name="tree_close",
            input_signature=input_signature,
            candidate_settings=candidate.settings,
            candidate_spec_hash=candidate.candidate_spec_hash(),
            thresholds={"max_abs_diff": 1e-6},
            dependency_identities=candidate.dependency_identities,
            generator_id=candidate.generator_id,
            generator_version=candidate.generator_version,
        ),
        candidate_spec_hash=candidate.candidate_spec_hash(),
        dependency_identities=dict(candidate.dependency_identities),
    )

    return dataclasses.replace(record, content_hash=record.computed_content_hash())


def _identity_kwargs(
    families: tuple[str, ...] = ("family",),
) -> dict[str, Any]:
    return {
        "target_identity": {"target": "test"},
        "runtime_identities": {
            family: {"runtime": f"test.{family}"} for family in families
        },
        "adapter_identities": {
            family: {"adapter_id": "tests", "adapter_version": "1"}
            for family in families
        },
    }


def _current_record(record: FullSizeRecord) -> FullSizeRecord:
    return dataclasses.replace(record, content_hash=record.computed_content_hash())


def test_within_family_selection() -> None:
    signature = {"case": "current"}
    policy = vp.SelectionPolicy()
    fast = vp.Candidate("family", "fast", {})
    near = vp.Candidate("family", "near", {})
    slow = vp.Candidate("family", "slow", {})

    selected, _ = select_family(
        (
            (
                fast,
                _record(
                    fast,
                    elapsed=(10.0, 10.0),
                    reserved=(9.0, 9.0),
                    input_signature=signature,
                ),
            ),
            (
                near,
                _record(
                    near,
                    elapsed=(10.4, 10.4),
                    reserved=(1.0, 1.0),
                    input_signature=signature,
                ),
            ),
            (
                slow,
                _record(
                    slow,
                    elapsed=(11.0, 11.0),
                    reserved=(0.1, 0.1),
                    input_signature=signature,
                ),
            ),
        ),
        input_signature=signature,
        policy=policy,
    )

    assert selected == near


def test_selection_rejects_unsupported_policy_fields() -> None:
    candidate = vp.Candidate("family", "row", {})

    with pytest.raises(RuntimeError):
        select_family(
            (
                (
                    candidate,
                    _record(
                        candidate,
                        elapsed=(1.0,),
                        reserved=(1.0,),
                        input_signature={},
                    ),
                ),
            ),
            input_signature={},
            policy=vp.SelectionPolicy(speed_statistic="mean_elapsed_seconds"),
        )


def test_selection_rejects_invalid_rows() -> None:
    signature = {"case": "current"}
    stale = vp.Candidate("family", "stale", {})
    failed = vp.Candidate("family", "failed", {})

    with pytest.raises(vp.NoPassedCandidateError):
        select_family(
            (
                (
                    stale,
                    _record(
                        stale,
                        elapsed=(1.0,),
                        reserved=(1.0,),
                        input_signature={"case": "old"},
                    ),
                ),
                (
                    failed,
                    _record(
                        failed,
                        elapsed=(),
                        reserved=(),
                        input_signature=signature,
                        status="failed",
                    ),
                ),
            ),
            input_signature=signature,
            policy=vp.SelectionPolicy(),
        )


def test_selection_rejects_stale_owner_hash() -> None:
    signature = {"case": "current"}
    candidate = vp.Candidate("family", "row", {})
    stale_owner = dataclasses.replace(
        _record(
            candidate,
            elapsed=(1.0,),
            reserved=(1.0,),
            input_signature=signature,
        ),
        owner_hash="stale",
    )

    with pytest.raises(vp.NoPassedCandidateError):
        select_family(
            ((candidate, stale_owner),),
            input_signature=signature,
            policy=vp.SelectionPolicy(),
        )


def test_selection_uses_json_normalized_signatures_and_content_hashes() -> None:
    candidate = vp.Candidate("family", "row", {})
    tuple_signature = {"shape": (1, 2)}
    list_signature = {"shape": [1, 2]}
    record = _current_record(
        _record(
            candidate,
            elapsed=(1.0,),
            reserved=(1.0,),
            input_signature=tuple_signature,
        )
    )
    selected, _ = select_family(
        ((candidate, record),),
        input_signature=list_signature,
        policy=vp.SelectionPolicy(),
    )

    assert selected == candidate

    stale_content = dataclasses.replace(
        record,
        output_signature={"changed": True},
    )

    with pytest.raises(vp.NoPassedCandidateError):
        select_family(
            ((candidate, stale_content),),
            input_signature=list_signature,
            policy=vp.SelectionPolicy(),
        )


def test_memory_stability() -> None:
    candidate = vp.Candidate("family", "row", {})
    stable = _record(
        candidate,
        elapsed=(1.0, 1.0),
        reserved=(1.0, 1.0),
        input_signature={},
    )
    unstable = FullSizeRecord(
        family=candidate.family,
        candidate_id=candidate.candidate_id,
        status="passed",
        input_signature={},
        candidate_settings={},
        generator_id="default",
        generator_version="0.0.1",
        owner_hash="owner",
        candidate_spec_hash=candidate.candidate_spec_hash(),
        timing_samples=stable.timing_samples,
        memory_samples=(
            Measurement(1.0, 1.0, 1.0, 1.0, 1.0),
            Measurement(1.0, 1.0, 1.0, 1.0, 2.0),
        ),
    )

    assert memory_stable(stable)
    assert not memory_stable(unstable)

    mixed_device_unstable = dataclasses.replace(
        stable,
        memory_samples=(
            Measurement(1.0, 1.0, 1.0, 100.0, 100.0, rank=0, device="cuda:0"),
            Measurement(1.0, 1.0, 1.0, 1.0, 1.0, rank=0, device="cuda:1"),
            Measurement(1.0, 1.0, 1.0, 2.0, 2.0, rank=0, device="cuda:1"),
        ),
    )

    assert not memory_stable(mixed_device_unstable)


def test_cohort_selection_prefers_lower_memory_near_fastest() -> None:
    signature = {}
    policy = vp.SelectionPolicy()
    first_a = vp.Candidate("a", "first-a", {"backend": "first"})
    first_b = vp.Candidate("b", "first-b", {"backend": "first"})
    second_a = vp.Candidate("a", "second-a", {"backend": "second"})
    second_b = vp.Candidate("b", "second-b", {"backend": "second"})
    cohort = select_cohort(
        (
            {
                "a": (
                    first_a,
                    _record(
                        first_a,
                        elapsed=(10.0,),
                        reserved=(10.0,),
                        input_signature=signature,
                    ),
                ),
                "b": (
                    first_b,
                    _record(
                        first_b,
                        elapsed=(10.0,),
                        reserved=(10.0,),
                        input_signature=signature,
                    ),
                ),
            },
            {
                "a": (
                    second_a,
                    _record(
                        second_a,
                        elapsed=(10.4,),
                        reserved=(1.0,),
                        input_signature=signature,
                    ),
                ),
                "b": (
                    second_b,
                    _record(
                        second_b,
                        elapsed=(10.4,),
                        reserved=(1.0,),
                        input_signature=signature,
                    ),
                ),
            },
        ),
        families=("a", "b"),
        policy=policy,
    )

    assert cohort["a"][0] == second_a


def test_cohort_selection_rejects_incomplete_cohort() -> None:
    signature = {}
    first_a = vp.Candidate("a", "first-a", {"backend": "first"})
    second_a = vp.Candidate("a", "second-a", {"backend": "second"})
    second_b = vp.Candidate("b", "second-b", {"backend": "second"})

    with pytest.raises(vp.NoPassedCandidateError):
        select_cohort(
            (
                {
                    "a": (
                        first_a,
                        _record(
                            first_a,
                            elapsed=(1.0,),
                            reserved=(1.0,),
                            input_signature=signature,
                        ),
                    )
                },
                {
                    "a": (
                        second_a,
                        _record(
                            second_a,
                            elapsed=(1.0,),
                            reserved=(1.0,),
                            input_signature=signature,
                        ),
                    ),
                    "b": (
                        second_b,
                        _record(
                            second_b,
                            elapsed=(1.0,),
                            reserved=(1.0,),
                            input_signature=signature,
                        ),
                    ),
                },
            ),
            families=("a", "b", "c"),
            policy=vp.SelectionPolicy(),
        )


def test_record_current_rejects_stale_generator_version() -> None:
    record = {
        "record_type": "full_size",
        "schema_version": 1,
        "package_version": "0.0.1",
        "owner_hash": "owner",
        "input_signature": {},
        "candidate_settings": {},
        "status": "passed",
        "generator_id": "gen",
        "generator_version": "1",
    }

    assert not record_current(
        record,
        record_type="full_size",
        family="family",
        candidate_id="row",
        input_signature={},
        candidate_settings={},
        generator_id="gen",
        generator_version="2",
    )


def test_row_owner_hash_includes_row_and_check_identity() -> None:
    first = compute_record_owner_hash(
        record_type="reference",
        family="family",
        candidate_id="row-a",
        check_name="first",
        input_signature={},
        candidate_settings={"axis": "same"},
        thresholds={"max_abs_diff": 1e-6},
        generator_id="gen",
        generator_version="1",
    )
    second = compute_record_owner_hash(
        record_type="reference",
        family="family",
        candidate_id="row-b",
        check_name="first",
        input_signature={},
        candidate_settings={"axis": "same"},
        thresholds={"max_abs_diff": 1e-6},
        generator_id="gen",
        generator_version="1",
    )
    third = compute_record_owner_hash(
        record_type="reference",
        family="family",
        candidate_id="row-a",
        check_name="second",
        input_signature={},
        candidate_settings={"axis": "same"},
        thresholds={"max_abs_diff": 1e-6},
        generator_id="gen",
        generator_version="1",
    )

    assert first != second
    assert first != third


def test_admission_helpers() -> None:
    vp.admit_functional_call({
        "parameter_keys": ("weight",),
        "buffer_keys": ("running",),
        "tie_weights": True,
        "strict": False,
        "parametrization_policy": "active",
        "mutates_state": False,
        "mutated_parameter_keys": (),
        "mutated_buffer_keys": (),
        "module_mode": "eval",
    })

    with pytest.raises(vp.AdmissionError):
        vp.admit_torch_func({
            "contains_autograd_call": True,
            "contains_backward_call": False,
            "uses_out_variant": False,
            "uses_data_dependent_control_flow": False,
            "uses_item": False,
            "has_dynamic_shape_output": False,
            "vmap_randomness": "error",
            "requires_forward_ad": False,
            "forward_ad_supported": False,
        })

    with pytest.raises(vp.AdmissionError):
        vp.admit_checkpoint({
            "use_reentrant": True,
            "preserve_rng_state": True,
            "determinism_check": "default",
            "context_fn": None,
            "early_stop": True,
            "moves_to_new_device": False,
            "uses_global_state": False,
        })


def checkpoint_fields() -> dict[str, object]:
    return {
        "use_reentrant": False,
        "preserve_rng_state": True,
        "determinism_check": "default",
        "context_fn": None,
        "early_stop": True,
        "moves_to_new_device": False,
        "uses_global_state": False,
    }


def test_checkpoint_operation_preserves_rng_state() -> None:
    values = []
    vector = torch.tensor([1.0, 2.0], requires_grad=True)
    candidate = vp.Candidate(
        "family",
        "row",
        {"checkpoint": "non_reentrant", **checkpoint_fields()},
        admission_status="passed",
    )

    def function(value: torch.Tensor) -> torch.Tensor:
        noise = torch.rand_like(value)
        values.append(noise.detach().clone())

        return (value * noise).sum()

    torch.manual_seed(17)
    output = vp.checkpoint_operation(
        candidate,
        function,
        (vector,),
        policy_key="checkpoint",
    )()
    assert isinstance(output, torch.Tensor)
    output.backward()

    assert vector.grad is not None
    assert len(values) == 2
    assert torch.equal(values[0], values[1])


def test_checkpoint_operation_rejects_unadmitted_fields_before_execution() -> None:
    calls = []
    vector = torch.tensor([1.0], requires_grad=True)
    candidate = vp.Candidate(
        "family",
        "row",
        {
            "checkpoint_policy": "non_reentrant_deterministic",
            **checkpoint_fields(),
            "use_reentrant": True,
        },
    )

    def function(value: torch.Tensor) -> torch.Tensor:
        calls.append(value.detach().clone())

        return value.square()

    with pytest.raises(vp.AdmissionError):
        vp.checkpoint_operation(
            candidate,
            function,
            (vector,),
            policy_key="checkpoint_policy",
        )()

    assert calls == []


def test_adapter_runtime_executes_checkpoint_without_standard_runtime_support(
    tmp_path: Path,
) -> None:
    model = torch.nn.Linear(1, 1)
    settings = {
        "operator_path": "autograd_grad",
        "checkpoint": "non_reentrant",
        **checkpoint_fields(),
    }
    candidate = vp.Candidate(
        "family",
        "checkpoint-row",
        settings,
        admission_status="passed",
    )

    def adapter_operation_factory(
        candidate: vp.Candidate,
        batch: Mapping[str, object],
        vector: vp.TensorTree,
    ) -> vp.CandidateOperation:
        assert batch["family"] == "family"
        assert isinstance(vector, torch.Tensor)

        def function(value: torch.Tensor) -> torch.Tensor:
            return value * 3.0

        return vp.checkpoint_operation(
            candidate,
            function,
            (vector,),
            policy_key="checkpoint",
        )

    def reference_check(
        candidate: vp.Candidate,
        batch: Mapping[str, object],
        vector: vp.TensorTree,
    ) -> vp.ReferenceResult:
        output = adapter_operation_factory(candidate, batch, vector)()
        assert isinstance(vector, torch.Tensor)
        assert isinstance(output, torch.Tensor)
        assert torch.equal(output, vector * 3.0)

        return reference_passed()

    runtime = vp.RuntimeConfig(
        candidates=(candidate,),
        operation_factory=adapter_operation_factory,
        reference_check=reference_check,
        materializer=materialize_candidate,
        axis_registry=None,
        signature={"runtime": "checkpoint-adapter"},
    )
    problem = vp.Problem(
        model=model,
        params=vp.parameter_surface(model),
        data=OneBatchData(),
        operator=vp.gradient("family", "objective", aggregation="sum"),
        vectors=OneVectorProvider(),
        target=cpu_target(
            vp.TimingPolicy(
                short_seconds=0.0,
                medium_seconds=0.0,
                long_warmups=0,
                long_measured_calls=1,
            )
        ),
        runtime=runtime,
    )
    plan = vp.tune(
        problem,
        run_dir=tmp_path,
        memory_backend=CPUMemoryBackend(),
        clock=SequenceClock((0.0, 1.0)),
    )

    assert plan.selected["family"].candidate_id == "checkpoint-row"

    def scalar_objective(
        params: vp.ParameterTree,
        buffers: vp.BufferTree,
        batch: vp.Batch,
        context: vp.ObjectiveContext,
    ) -> torch.Tensor:
        assert not buffers
        assert batch
        assert context.family == "family"

        return params["weight"].sum()

    standard_factory = vp.standard_operation_factory(
        vp.gradient("family", "objective", aggregation="sum"),
        params=dict(model.named_parameters()),
        buffers={},
        scalar_objectives={"objective": scalar_objective},
    )

    with pytest.raises(vp.MaterializationError):
        standard_factory(candidate, {"family": "family"}, torch.tensor([1.0]))()


def test_transformers_attention_admission_axis() -> None:
    policy = vpa.TransformersAttentionPolicy(
        model_config_hash="model",
        use_cache=False,
        softcap={"logit_softcap": 30.0},
        mask_semantics="boolean_keep_mask",
        causal_policy="causal",
        backend_numeric_policy={"backend": "flash_attention_2"},
        determinism={"deterministic": True},
        padding_limit=4096,
        forced_kernel_available=True,
    )
    axis = vpa.transformers_attention_axis(
        (
            "sdpa_math",
            "sdpa_flash",
            "patched_eager",
            "transformers_eager",
            "transformers_sdpa",
            "transformers_flash_attention_2",
        ),
        policy=policy,
    )
    flash = vp.Candidate(
        "family",
        "flash",
        {
            "attention_impl": "transformers_flash_attention_2",
            "model_dtype": "bfloat16",
        },
    )
    float_flash = vp.Candidate(
        "family",
        "float-flash",
        {
            "attention_impl": "transformers_flash_attention_2",
            "model_dtype": "float32",
        },
    )
    attentions = vp.Candidate(
        "family",
        "attentions",
        {"attention_impl": "transformers_sdpa", "output_attentions": True},
    )
    math_attention = vp.Candidate(
        "family",
        "math",
        {"attention_impl": "sdpa_math"},
    )
    math_attentions = vp.Candidate(
        "family",
        "math-attentions",
        {"attention_impl": "sdpa_math", "output_attentions": True},
    )
    sdpa_flash = vp.Candidate(
        "family",
        "sdpa-flash",
        {
            "attention_impl": "sdpa_flash",
            "model_dtype": "bfloat16",
        },
    )
    float_sdpa_flash = vp.Candidate(
        "family",
        "float-sdpa-flash",
        {
            "attention_impl": "sdpa_flash",
            "model_dtype": "float32",
        },
    )
    eval_dropout = vp.Candidate(
        "family",
        "dropout",
        {
            "attention_impl": "transformers_eager",
            "module_mode": "eval",
            "dropout_p": 0.1,
        },
    )
    bad_gqa = vp.Candidate(
        "family",
        "gqa",
        {
            "attention_impl": "transformers_eager",
            "enable_gqa": True,
            "query_heads": 5,
            "key_value_heads": 2,
        },
    )
    valid_gqa = vp.Candidate(
        "family",
        "valid-gqa",
        {
            "attention_impl": "transformers_eager",
            "enable_gqa": True,
            "query_heads": 8,
            "key_heads": 2,
            "value_heads": 2,
        },
    )
    mismatched_gqa = vp.Candidate(
        "family",
        "mismatched-gqa",
        {
            "attention_impl": "transformers_eager",
            "enable_gqa": True,
            "query_heads": 8,
            "key_heads": 2,
            "value_heads": 4,
        },
    )
    patched = vp.Candidate(
        "family",
        "patched",
        {
            "attention_impl": "patched_eager",
            "output_attentions": True,
            "patched_attention_id": "gemma-softcap-eager",
            "patched_attention_semantics": {"logit_softcap": 30.0},
        },
    )
    untracked_patch = vp.Candidate(
        "family",
        "untracked-patch",
        {"attention_impl": "patched_eager"},
    )

    assert axis.admit(flash) == (True, None)
    assert axis.admit(float_flash)[0] is False
    assert axis.admit(attentions)[0] is False
    assert axis.admit(math_attention) == (True, None)
    assert axis.admit(math_attentions)[0] is False
    assert axis.admit(sdpa_flash) == (True, None)
    assert axis.admit(float_sdpa_flash)[0] is False
    assert axis.admit(eval_dropout)[0] is False
    assert axis.admit(bad_gqa)[0] is False
    assert axis.admit(valid_gqa) == (True, None)
    assert axis.admit(mismatched_gqa)[0] is False
    assert axis.admit(patched) == (True, None)
    assert axis.admit(untracked_patch)[0] is False
    assert axis.signature()["identity"]["softcap"] == {"logit_softcap": 30.0}

    no_padding_policy = dataclasses.replace(policy, padding_limit=None)
    no_padding_axis = vpa.transformers_attention_axis(
        ("transformers_flash_attention_2",),
        policy=no_padding_policy,
    )
    unavailable_policy = dataclasses.replace(
        policy,
        forced_kernel_available=False,
        forced_kernel_failure_reason="kernel unavailable",
    )
    unavailable_axis = vpa.transformers_attention_axis(
        ("transformers_flash_attention_2",),
        policy=unavailable_policy,
    )

    assert no_padding_axis.admit(flash)[0] is False
    assert unavailable_axis.admit(flash) == (False, "kernel unavailable")
    assert (
        vpa.admit_transformers_attention(
            vp.Candidate(
                "family",
                "unknown",
                {"attention_impl": "unknown"},
            ),
            policy=policy,
        )[0]
        is False
    )

    with pytest.raises(vp.AdmissionError):
        vpa.transformers_attention_axis(("unknown",), policy=policy)


def test_curvlinops_operation_factory_handles_flat_and_tensor_list_vectors() -> None:
    class FakeLinearOperator:
        def __matmul__(
            self,
            vector: torch.Tensor | list[torch.Tensor],
        ) -> torch.Tensor | list[torch.Tensor]:
            if isinstance(vector, torch.Tensor):
                return vector * 2.0

            return [value + 1.0 for value in vector]

    calls = []

    def linear_operator_factory(
        candidate: vp.Candidate,
        batch: Mapping[str, object],
    ) -> FakeLinearOperator:
        calls.append((candidate.candidate_id, batch["source"]))

        return FakeLinearOperator()

    factory = curvlinops_operation_factory(linear_operator_factory)
    ordered_factory = curvlinops_operation_factory(
        linear_operator_factory,
        parameter_names=("weight", "bias"),
    )
    candidate = vp.Candidate("family", "row", {}, admission_status="passed")
    flat = factory(candidate, {"source": "flat"}, torch.tensor([1.0, 2.0]))()
    listed = factory(
        candidate,
        {"source": "list"},
        (torch.tensor([1.0]), torch.tensor([2.0])),
    )()
    mapped = ordered_factory(
        candidate,
        {"source": "dict"},
        {"bias": torch.tensor([2.0]), "weight": torch.tensor([1.0])},
    )()

    assert isinstance(flat, torch.Tensor)
    assert torch.equal(flat, torch.tensor([2.0, 4.0]))
    assert isinstance(listed, tuple)
    assert torch.equal(listed[0], torch.tensor([2.0]))
    assert torch.equal(listed[1], torch.tensor([3.0]))

    def require_mapping(tree: vp.TensorTree) -> dict[str, vp.TensorTree]:
        if type(tree) is not dict:
            message = "expected mapping tensor tree"
            raise TypeError(message)

        return dict(tree)

    mapped_values = require_mapping(mapped)
    weight = mapped_values["weight"]
    bias = mapped_values["bias"]
    assert isinstance(weight, torch.Tensor)
    assert isinstance(bias, torch.Tensor)
    assert torch.equal(weight, torch.tensor([2.0]))
    assert torch.equal(bias, torch.tensor([3.0]))
    assert calls == [("row", "flat"), ("row", "list"), ("row", "dict")]

    with pytest.raises(TypeError):
        factory(candidate, {"source": "bad"}, {"weight": torch.tensor([1.0])})()

    with pytest.raises(RuntimeError):
        ordered_factory(
            candidate,
            {"source": "bad"},
            {"wrong": torch.tensor([1.0])},
        )()


def test_axis_registry_admits_grid_and_records_failed_admission() -> None:
    model = torch.nn.Linear(1, 1)
    registry = vp.AxisRegistry()

    def admit_dtype(candidate: vp.Candidate) -> tuple[bool, str | None]:
        if candidate.settings["dtype"] == "float16":
            return False, "float16 disabled"

        return True, None

    registry.register(
        vp.AxisDescriptor(
            "dtype",
            ("dtype",),
            ("float32", "float16"),
            admission_rule=admit_dtype,
        )
    )

    with pytest.raises(vp.AdmissionError):
        registry.register(vp.AxisDescriptor("other", ("dtype",), ("float64",)))

    candidates = vp.settings_product(
        "family",
        {"dtype": ("float32", "float16")},
        generator_id="grid",
        generator_version="1",
    )

    def reference_check(
        candidate: vp.Candidate,
        batch: Mapping[str, object],
        vector: vp.TensorTree,
    ) -> vp.ReferenceResult:
        assert candidate.settings["dtype"] == "float32"
        assert batch["family"] == "family"
        assert batch["source"] == "reference"
        assert isinstance(vector, torch.Tensor)
        assert torch.equal(vector, torch.tensor([1.0]))

        return reference_passed()

    def operation_factory(
        candidate: vp.Candidate,
        batch: Mapping[str, object],
        vector: vp.TensorTree,
    ) -> vp.CandidateOperation:
        assert candidate.settings["dtype"] == "float32"
        assert batch["family"] == "family"
        assert batch["source"] == "probe"
        assert isinstance(vector, torch.Tensor)

        return vp.constant_operation(vector)

    problem = vp.Problem(
        model=model,
        params=vp.parameter_surface(model),
        data=OneBatchData(),
        operator=vp.hvp("family", "objective", aggregation="sum"),
        vectors=OneVectorProvider(),
        target=cpu_target(
            vp.TimingPolicy(
                short_seconds=0.0,
                medium_seconds=0.0,
                long_warmups=0,
                long_measured_calls=1,
            )
        ),
        runtime=vp.RuntimeConfig(
            candidates,
            operation_factory,
            reference_check,
            materialize_candidate,
            registry,
            {"generator": "grid"},
        ),
    )
    plan = vp.tune(
        problem,
        memory_backend=CPUMemoryBackend(),
        clock=SequenceClock((0.0, 1.0)),
    )

    assert tuple(candidate.candidate_id for candidate in candidates) == (
        "family:0",
        "family:1",
    )
    assert plan.selected["family"].settings == {"dtype": "float32"}
    assert plan.full_size_records[1].status == "failed"
    assert plan.full_size_records[1].error_type == "AdmissionError"

    invalid = vp.Candidate(
        "family",
        "invalid",
        {"dtype": "float64"},
        changed_axes=("dtype",),
    )

    assert registry.admit(invalid).admission_status == "failed"


def test_settings_product_expands_registry_multi_key_axis() -> None:
    registry = vp.AxisRegistry()
    registry.register(
        vp.AxisDescriptor(
            "pair",
            ("left", "right"),
            ({"left": 1, "right": 2},),
        )
    )
    candidates = vp.settings_product(
        "family",
        {"pair": ({"left": 1, "right": 2},)},
        axis_registry=registry,
    )

    assert candidates[0].settings == {"left": 1, "right": 2}
    assert candidates[0].changed_axes == ("pair",)
    assert registry.admit(candidates[0]).admission_status == "passed"

    with pytest.raises(vp.AdmissionError):
        vp.settings_product(
            "family",
            {"pair": ({"left": 1},)},
            axis_registry=registry,
        )


def test_standard_axis_registry_validates_core_axes() -> None:
    registry = vp.standard_axis_registry()
    torch_func_fields = {
        "contains_autograd_call": False,
        "contains_backward_call": False,
        "uses_out_variant": False,
        "uses_data_dependent_control_flow": False,
        "uses_item": False,
        "has_dynamic_shape_output": False,
        "vmap_randomness": "error",
        "requires_forward_ad": True,
        "forward_ad_supported": True,
    }
    checkpoint_fields = {
        "use_reentrant": False,
        "preserve_rng_state": True,
        "determinism_check": "default",
        "context_fn": None,
        "early_stop": True,
        "moves_to_new_device": False,
        "uses_global_state": False,
    }
    functional_call_fields = {
        "parameter_keys": ("weight",),
        "buffer_keys": ("running",),
        "tie_weights": True,
        "strict": False,
        "parametrization_policy": "active",
        "mutates_state": False,
        "mutated_parameter_keys": (),
        "mutated_buffer_keys": (),
        "module_mode": "eval",
    }
    candidate = vp.Candidate(
        "family",
        "row",
        {
            "model_dtype": "bfloat16",
            "batch_size": 4,
            "attention_impl": "sdpa_math",
            "sharding": "single_device",
            "operator_path": "reverse_over_reverse",
            "allow_tf32": True,
        },
    )
    bad_budget = vp.Candidate("family", "bad-budget", {"batch_size": 0})
    bad_flag = vp.Candidate("family", "bad-flag", {"allow_tf32": 1})
    bad_dtype = vp.Candidate("family", "bad-dtype", {"model_dtype": "float64"})
    bad_path = vp.Candidate(
        "family",
        "bad-path",
        {"operator_path": "reverse_over_forward"},
    )
    missing_torch_func_fields = vp.Candidate(
        "family",
        "missing-torch-func-fields",
        {"operator_path": "torch_func_jvp"},
    )
    valid_torch_func = vp.Candidate(
        "family",
        "valid-torch-func",
        {"operator_path": "torch_func_jvp", **torch_func_fields},
    )
    valid_vmap = vp.Candidate(
        "family",
        "valid-vmap",
        {
            "operator_path": "per_example_gradient_vmap",
            **torch_func_fields,
            "requires_forward_ad": False,
            "vmap_chunk_size": 2,
            "vmap_batch_in_dims": {"x": 0, "normalization": None},
        },
    )
    missing_vmap_in_dims = vp.Candidate(
        "family",
        "missing-vmap-in-dims",
        {
            "operator_path": "per_example_gradient_vmap",
            **torch_func_fields,
            "requires_forward_ad": False,
            "vmap_chunk_size": 2,
        },
    )
    invalid_vmap_in_dims = vp.Candidate(
        "family",
        "invalid-vmap-in-dims",
        {
            "operator_path": "per_example_gradient_vmap",
            **torch_func_fields,
            "requires_forward_ad": False,
            "vmap_chunk_size": 2,
            "vmap_batch_in_dims": {"x": "0"},
        },
    )
    forward_ad_vmap = vp.Candidate(
        "family",
        "forward-ad-vmap",
        {
            "operator_path": "per_example_gradient_vmap",
            **torch_func_fields,
            "vmap_chunk_size": 2,
            "vmap_batch_in_dims": {"x": 0},
        },
    )
    missing_functional_call_fields = vp.Candidate(
        "family",
        "missing-functional-call-fields",
        {"tie_weights": True},
    )
    valid_functional_call = vp.Candidate(
        "family",
        "valid-functional-call",
        functional_call_fields,
    )
    missing_mutation_policy = vp.Candidate(
        "family",
        "missing-mutation-policy",
        {**functional_call_fields, "mutates_state": True},
    )
    mutating_functional_call = vp.Candidate(
        "family",
        "mutating-functional-call",
        {
            **functional_call_fields,
            "mutates_state": True,
            "mutated_buffer_keys": ("running",),
        },
    )
    missing_checkpoint_fields = vp.Candidate(
        "family",
        "missing-checkpoint-fields",
        {"checkpoint": "non_reentrant"},
    )
    valid_checkpoint = vp.Candidate(
        "family",
        "valid-checkpoint",
        {"checkpoint": "non_reentrant", **checkpoint_fields},
    )
    reentrant_checkpoint = vp.Candidate(
        "family",
        "reentrant-checkpoint",
        {"checkpoint": "non_reentrant", **checkpoint_fields, "use_reentrant": True},
    )
    valid_checkpoint_policy = vp.Candidate(
        "family",
        "valid-checkpoint-policy",
        {
            "checkpoint_policy": "non_reentrant_deterministic",
            **checkpoint_fields,
        },
    )

    assert registry.admit(candidate).admission_status == "passed"
    assert registry.admit(bad_budget).admission_status == "failed"
    assert registry.admit(bad_flag).admission_status == "failed"
    assert registry.admit(bad_dtype).admission_status == "failed"
    assert registry.admit(bad_path).admission_status == "failed"
    assert registry.admit(missing_torch_func_fields).admission_status == "failed"
    assert registry.admit(valid_torch_func).admission_status == "passed"
    assert registry.admit(valid_vmap).admission_status == "passed"
    assert registry.admit(missing_vmap_in_dims).admission_status == "failed"
    assert registry.admit(invalid_vmap_in_dims).admission_status == "failed"
    assert registry.admit(forward_ad_vmap).admission_status == "failed"
    assert registry.admit(missing_functional_call_fields).admission_status == "failed"
    assert registry.admit(valid_functional_call).admission_status == "passed"
    assert registry.admit(missing_mutation_policy).admission_status == "failed"
    assert registry.admit(mutating_functional_call).admission_status == "passed"
    assert registry.admit(missing_checkpoint_fields).admission_status == "failed"
    assert registry.admit(valid_checkpoint).admission_status == "passed"
    assert registry.admit(reentrant_checkpoint).admission_status == "failed"
    assert registry.admit(valid_checkpoint_policy).admission_status == "passed"
    assert registry.axes["model_dtype"].admit(bad_dtype)[0] is False

    with pytest.raises(vp.AdmissionError):
        vp.AxisRegistry().register(vp.AxisDescriptor("bad", ("x",), ()))


def test_problem_signature_includes_axis_registry_identity() -> None:
    model = torch.nn.Linear(1, 1)
    first_registry = vp.AxisRegistry()
    second_registry = vp.AxisRegistry()
    first_registry.register(
        vp.AxisDescriptor(
            "axis",
            ("axis",),
            ("value",),
            adapter_id="adapter",
            adapter_version="1",
        )
    )
    second_registry.register(
        vp.AxisDescriptor(
            "axis",
            ("axis",),
            ("value",),
            adapter_id="adapter",
            adapter_version="2",
        )
    )

    def reference_check(
        candidate: vp.Candidate,
        batch: Mapping[str, object],
        vector: vp.TensorTree,
    ) -> vp.ReferenceResult:
        assert candidate
        assert batch
        assert vector

        return reference_passed()

    def operation_factory(
        candidate: vp.Candidate,
        batch: Mapping[str, object],
        vector: vp.TensorTree,
    ) -> vp.CandidateOperation:
        assert candidate
        assert batch

        return vp.constant_operation(vector)

    first = vp.Problem(
        model=model,
        params=vp.parameter_surface(model),
        data=OneBatchData(),
        operator=vp.hvp("family", "objective", aggregation="sum"),
        vectors=OneVectorProvider(),
        target=cpu_target(),
        runtime=vp.RuntimeConfig(
            (),
            operation_factory,
            reference_check,
            materialize_candidate,
            first_registry,
            {"generator": "registry"},
        ),
    )
    second = dataclasses.replace(
        first,
        runtime=dataclasses.replace(first.runtime, axis_registry=second_registry),
    )

    assert first_registry.signature() != second_registry.signature()
    assert first.input_signature() != second.input_signature()


def test_target_admission_rejects_disallowed_settings() -> None:
    model = torch.nn.Linear(1, 1)
    candidates = (
        vp.Candidate(
            "family",
            "allowed",
            {"model_dtype": "float32"},
            admission_status="passed",
        ),
        vp.Candidate(
            "family",
            "blocked",
            {"model_dtype": "bfloat16"},
            admission_status="passed",
        ),
    )

    def reference_check(
        candidate: vp.Candidate,
        batch: Mapping[str, object],
        vector: vp.TensorTree,
    ) -> vp.ReferenceResult:
        assert candidate.candidate_id == "allowed"
        assert batch["source"] == "reference"
        assert isinstance(vector, torch.Tensor)

        return reference_passed()

    def operation_factory(
        candidate: vp.Candidate,
        batch: Mapping[str, object],
        vector: vp.TensorTree,
    ) -> vp.CandidateOperation:
        assert candidate.candidate_id == "allowed"
        assert batch["source"] == "probe"
        assert isinstance(vector, torch.Tensor)

        return vp.constant_operation(vector)

    target = vp.Target(
        devices=("cpu",),
        accelerator="cpu",
        allowed_dtypes=("float32",),
        allowed_attention_impls=(),
        allowed_sharding_modes=("single_device",),
        timing_policy=vp.TimingPolicy(
            short_seconds=0.0,
            medium_seconds=0.0,
            long_warmups=0,
            long_measured_calls=1,
        ),
        selection_policy=vp.SelectionPolicy(),
        determinism_policy={},
        environment_capture={"runtime": "test"},
    )
    problem = vp.Problem(
        model=model,
        params=vp.parameter_surface(model),
        data=OneBatchData(),
        operator=vp.hvp("family", "objective", aggregation="sum"),
        vectors=OneVectorProvider(),
        target=target,
        runtime=vp.RuntimeConfig(
            candidates,
            operation_factory,
            reference_check,
            materialize_candidate,
            None,
            {"generator": "target-admission"},
        ),
    )
    plan = vp.tune(
        problem,
        memory_backend=CPUMemoryBackend(),
        clock=SequenceClock((0.0, 1.0)),
    )

    assert plan.selected["family"].candidate_id == "allowed"
    assert plan.full_size_records[1].status == "failed"
    assert plan.full_size_records[1].error_type == "AdmissionError"


def test_autobatch_bridge_selects_candidate_by_positive_index_domain() -> None:
    calls = []
    candidates = (
        vp.Candidate("family", "first", {}, admission_status="passed"),
        vp.Candidate("family", "second", {}, admission_status="passed"),
    )

    def fake_find(
        probe: Callable[[int], None],
        *,
        values: Sequence[int],
        goal: autobatch.Goal,
        cache_key: Hashable,
        warmup_steps: int,
        measure_steps: int,
        devices: list[int],
    ) -> int:
        assert values == (1, 2)
        assert cache_key == ("case", "bridge")
        assert warmup_steps == 1
        assert measure_steps == 2
        assert devices == [0]
        assert goal == autobatch.Goal.fastest_step()
        probe(2)

        return 2

    selected = vp.select_fastest_candidate_with_autobatch(
        candidates,
        lambda candidate: calls.append(candidate.candidate_id),
        cache_key=("case", "bridge"),
        warmup_steps=1,
        measure_steps=2,
        devices=[0],
        find=fake_find,
    )

    assert selected.candidate_id == "second"
    assert calls == ["second"]


def test_tune_delegates_autobatch_domain_to_autobatch_find(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []
    find_calls = []
    model = torch.nn.Linear(1, 1)

    def reference_check(
        candidate: vp.Candidate,
        batch: Mapping[str, object],
        vector: vp.TensorTree,
    ) -> vp.ReferenceResult:
        assert candidate.settings["batch_size"] in {1, 2}
        assert batch["source"] == "reference"
        assert isinstance(vector, torch.Tensor)

        return reference_passed()

    def operation_factory(
        candidate: vp.Candidate,
        batch: Mapping[str, object],
        vector: vp.TensorTree,
    ) -> vp.CandidateOperation:
        assert batch["source"] == "probe"
        assert isinstance(vector, torch.Tensor)

        def operation() -> torch.Tensor:
            calls.append(candidate.candidate_id)

            return vector * float(candidate.settings["batch_size"])

        return operation

    def fake_find(
        probe: Callable[[int], None],
        *,
        values: Sequence[int],
        goal: autobatch.Goal,
        cache_key: Hashable,
        warmup_steps: int,
        measure_steps: int,
        devices: list[int],
    ) -> int:
        assert values == (1, 2)
        assert goal == autobatch.Goal.fastest_step()
        assert isinstance(cache_key, tuple)
        assert warmup_steps == 0
        assert measure_steps == 1
        assert devices == [0]
        find_calls.append(tuple(values))
        probe(1)
        probe(2)

        return 2

    monkeypatch.setattr(autobatch_bridge.autobatch, "find", fake_find)

    target = cpu_target(
        vp.TimingPolicy(
            short_seconds=0.0,
            medium_seconds=0.0,
            long_warmups=0,
            long_measured_calls=1,
        )
    )
    problem = vp.Problem(
        model=model,
        params=vp.parameter_surface(model),
        data=OneBatchData(),
        operator=vp.gradient("family", "objective", aggregation="sum"),
        vectors=OneVectorProvider(),
        target=target,
        runtime=vp.RuntimeConfig(
            (
                vp.Candidate(
                    "family",
                    "base",
                    {},
                    admission_status="passed",
                ),
            ),
            operation_factory,
            reference_check,
            materialize_candidate,
            None,
            {"generator": "autobatch-domain"},
            (
                vp.AutobatchDomain(
                    axis_name="batch_size",
                    values=(1, 2),
                    settings_by_value={
                        1: {"batch_size": 1},
                        2: {"batch_size": 2},
                    },
                    value_to_settings_id="tests.batch_size_settings",
                    admission_identity={"case": "test"},
                    goal="fastest_step",
                    warmup_steps=0,
                    measure_steps=1,
                    devices=(0,),
                    cache_key_payload={"case": "autobatch-domain"},
                ),
            ),
        ),
    )
    plan = vp.tune(
        problem,
        memory_backend=CPUMemoryBackend(),
        clock=SequenceClock((0.0, 2.0, 2.0, 3.0)),
    )

    assert find_calls == [(1, 2)]
    assert tuple(record.candidate_id for record in plan.full_size_records) == (
        "base|batch_size=1",
        "base|batch_size=2",
    )
    assert calls == ["base|batch_size=1", "base|batch_size=2"]
    assert plan.selected["family"].candidate_id == "base|batch_size=2"
    assert plan.selected["family"].settings["batch_size"] == 2


def test_tune_uses_explicit_candidates_and_reference_checks(tmp_path: Path) -> None:
    calls = {"slow": 0, "bad": 0, "fast": 0}
    model = torch.nn.Linear(1, 1)
    candidates = (
        vp.Candidate(
            "family",
            "slow",
            {"scale": 1.0},
            admission_status="passed",
        ),
        vp.Candidate(
            "family",
            "bad",
            {"scale": 0.0},
            admission_status="passed",
        ),
        vp.Candidate(
            "family",
            "fast",
            {"scale": 2.0},
            admission_status="passed",
        ),
    )

    def reference_check(
        candidate: vp.Candidate,
        batch: Mapping[str, object],
        vector: vp.TensorTree,
    ) -> vp.ReferenceResult:
        assert batch["family"] == "family"
        assert batch["source"] == "reference"
        assert isinstance(vector, torch.Tensor)
        assert torch.equal(vector, torch.tensor([1.0]))

        if candidate.candidate_id == "bad":
            message = "bad row"
            raise vp.ReferenceFailedError(message)

        return reference_passed()

    def operation_factory(
        candidate: vp.Candidate,
        batch: Mapping[str, object],
        vector: vp.TensorTree,
    ) -> vp.CandidateOperation:
        assert batch["family"] == "family"
        assert batch["source"] == "probe"
        assert isinstance(vector, torch.Tensor)

        def operation() -> torch.Tensor:
            calls[candidate.candidate_id] += 1

            if candidate.candidate_id == "fast":
                return vector * 2.0

            return vector

        return operation

    target = cpu_target(
        vp.TimingPolicy(
            short_seconds=0.0,
            medium_seconds=0.0,
            long_warmups=0,
            long_measured_calls=1,
        )
    )
    problem = vp.Problem(
        model=model,
        params=vp.parameter_surface(model),
        data=OneBatchData(),
        operator=vp.hvp("family", "objective", aggregation="sum"),
        vectors=OneVectorProvider(),
        target=target,
        runtime=vp.RuntimeConfig(
            candidates,
            operation_factory,
            reference_check,
            materialize_candidate,
            None,
            {"generator": "unit_test"},
        ),
    )
    plan = vp.tune(
        problem,
        run_dir=tmp_path,
        memory_backend=CPUMemoryBackend(),
        clock=SequenceClock((0.0, 10.0, 10.0, 15.0)),
    )

    assert plan.selected["family"].candidate_id == "fast"
    assert len(plan.full_size_records) == 3
    assert calls == {"slow": 1, "bad": 0, "fast": 1}
    assert plan.full_size_records[1].status == "failed"
    assert not plan.full_size_records[1].reference_passed
    assert len(plan.check_records) == 3
    assert plan.check_records[0].status == "passed"
    assert plan.check_records[1].status == "failed"
    selected_operator = vp.materialize(plan, family="family")

    assert torch.equal(selected_operator(), torch.tensor([2.0]))

    saved_full_size, saved_checks = saved_plan_rows(tmp_path, plan)
    replayed = vp.plan_from_json(
        read_record(tmp_path / "summaries" / "tuning.json"),
        replay_context=replay_context_for_plan(plan),
        full_size_records=saved_full_size,
        check_records=saved_checks,
        candidate_records=saved_candidate_rows(tmp_path, plan),
        materializers={"family": materialize_candidate},
    )

    assert replayed.owner_hash() == plan.owner_hash()

    def validator(
        candidate: vp.Candidate,
        record: vp.FullSizeRecord,
        context: vp.PlanValidationContext,
    ) -> vp.ReferenceResult:
        assert candidate.candidate_id == "fast"
        assert record.candidate_id == "fast"
        assert context.family == "family"
        assert torch.equal(context.selected(), torch.tensor([2.0]))

        return vp.ReferenceResult(
            "selected_plan_validation",
            {"max_abs_diff": 1e-6},
            {"max_abs_diff": 0.0},
        )

    validation_records = vp.validate_plan(plan, {"family": validator})

    assert validation_records[0].status == "passed"
    assert validation_records[0].name == "selected_plan_validation"


def test_tune_records_reference_runtime_failures() -> None:
    model = torch.nn.Linear(1, 1)
    candidates = (
        vp.Candidate("family", "bad", {}, admission_status="passed"),
        vp.Candidate("family", "good", {}, admission_status="passed"),
    )

    def reference_check(
        candidate: vp.Candidate,
        batch: Mapping[str, object],
        vector: vp.TensorTree,
    ) -> vp.ReferenceResult:
        assert batch["source"] == "reference"
        assert isinstance(vector, torch.Tensor)

        if candidate.candidate_id == "bad":
            message = "reference runtime failed"
            raise RuntimeError(message)

        return reference_passed()

    def operation_factory(
        candidate: vp.Candidate,
        batch: Mapping[str, object],
        vector: vp.TensorTree,
    ) -> vp.CandidateOperation:
        assert candidate.candidate_id == "good"
        assert batch["source"] == "probe"
        assert isinstance(vector, torch.Tensor)

        return vp.constant_operation(vector)

    problem = vp.Problem(
        model=model,
        params=vp.parameter_surface(model),
        data=OneBatchData(),
        operator=vp.hvp("family", "objective", aggregation="sum"),
        vectors=OneVectorProvider(),
        target=cpu_target(
            vp.TimingPolicy(
                short_seconds=0.0,
                medium_seconds=0.0,
                long_warmups=0,
                long_measured_calls=1,
            )
        ),
        runtime=vp.RuntimeConfig(
            candidates,
            operation_factory,
            reference_check,
            materialize_candidate,
            None,
            {"generator": "reference-runtime"},
        ),
    )
    plan = vp.tune(
        problem,
        memory_backend=CPUMemoryBackend(),
        clock=SequenceClock((0.0, 1.0)),
    )

    assert plan.selected["family"].candidate_id == "good"
    assert plan.check_records[0].status == "failed"
    assert plan.check_records[0].error_type == "RuntimeError"
    assert plan.full_size_records[0].status == "failed"
    assert not plan.full_size_records[0].reference_passed


def test_tune_writes_records_and_produced_rows_are_current(tmp_path: Path) -> None:
    model = torch.nn.Linear(1, 1)
    candidate = vp.Candidate(
        "family",
        "row",
        {"scale": 1.0},
        admission_status="passed",
    )

    def reference_check(
        candidate: vp.Candidate,
        batch: Mapping[str, object],
        vector: vp.TensorTree,
    ) -> vp.ReferenceResult:
        assert candidate.candidate_id == "row"
        assert batch["source"] == "reference"
        assert isinstance(vector, torch.Tensor)

        return reference_passed()

    def operation_factory(
        candidate: vp.Candidate,
        batch: Mapping[str, object],
        vector: vp.TensorTree,
    ) -> vp.CandidateOperation:
        assert candidate.candidate_id == "row"
        assert batch["source"] == "probe"

        return vp.constant_operation(vector)

    problem = vp.Problem(
        model=model,
        params=vp.parameter_surface(model),
        data=OneBatchData(),
        operator=vp.hvp("family", "objective", aggregation="sum"),
        vectors=OneVectorProvider(),
        target=cpu_target(
            vp.TimingPolicy(
                short_seconds=0.0,
                medium_seconds=0.0,
                long_warmups=0,
                long_measured_calls=1,
            )
        ),
        runtime=vp.RuntimeConfig(
            (candidate,),
            operation_factory,
            reference_check,
            materialize_candidate,
            None,
            {"generator": "write-test"},
        ),
    )
    plan = vp.tune(
        problem,
        run_dir=tmp_path,
        memory_backend=CPUMemoryBackend(),
        clock=SequenceClock((0.0, 1.0)),
    )
    candidate_hash = candidate.candidate_spec_hash()
    full_size_row = read_record(
        tmp_path / "full_size" / "family" / "row" / f"{candidate_hash}.json"
    )
    reference_row = read_record(
        tmp_path / "references" / "family" / "row" / candidate_hash / "tree_close.json"
    )
    summary = read_record(tmp_path / "summaries" / "tuning.json")

    assert read_record(
        tmp_path / "candidates" / "family" / "row" / f"{candidate_hash}.json"
    )
    candidate_row = read_record(
        tmp_path / "candidates" / "family" / "row" / f"{candidate_hash}.json"
    )
    replayed_candidate = vp.candidate_record_from_json(candidate_row)

    assert replayed_candidate == candidate

    stale_candidate_row = dict(candidate_row)
    stale_candidate_row["owner_hash"] = "stale"

    with pytest.raises(vp.StaleRecordError):
        vp.candidate_record_from_json(stale_candidate_row)

    changed_candidate_row = dict(candidate_row)
    changed_candidate_row["status"] = "failed"

    with pytest.raises(vp.StaleRecordError):
        vp.candidate_record_from_json(changed_candidate_row)

    assert record_current(
        full_size_row,
        record_type="full_size",
        family=candidate.family,
        candidate_id=candidate.candidate_id,
        input_signature=problem.input_signature(),
        candidate_settings=candidate.settings,
        candidate_spec_hash=candidate.candidate_spec_hash(),
        dependency_identities=candidate.dependency_identities,
        generator_id=candidate.generator_id,
        generator_version=candidate.generator_version,
    )
    assert record_current(
        reference_row,
        record_type="reference",
        family=candidate.family,
        candidate_id=candidate.candidate_id,
        check_name="tree_close",
        input_signature=problem.input_signature(),
        candidate_settings=candidate.settings,
        candidate_spec_hash=candidate.candidate_spec_hash(),
        thresholds={"max_abs_diff": 1e-6},
        dependency_identities=candidate.dependency_identities,
        generator_id=candidate.generator_id,
        generator_version=candidate.generator_version,
    )
    assert vp.plan_record_current(summary, plan)

    stale_summary = dict(summary)
    stale_summary["generator_version"] = "stale"

    assert not vp.plan_record_current(stale_summary, plan)

    stale_reference_row = dict(reference_row)
    stale_reference_row["owner_hash"] = "stale"

    with pytest.raises(vp.StaleRecordError):
        vp.check_record_from_json(stale_reference_row)

    changed_full_size = dict(full_size_row)
    changed_timing = [dict(sample) for sample in changed_full_size["timing_samples"]]
    changed_timing[0]["elapsed_seconds"] = 1000.0
    changed_full_size["timing_samples"] = changed_timing

    with pytest.raises(vp.StaleRecordError):
        vp.full_size_record_from_json(changed_full_size)

    changed_reference = dict(reference_row)
    changed_reference["measurements"] = {"max_abs_diff": 0.5}

    with pytest.raises(vp.StaleRecordError):
        vp.check_record_from_json(changed_reference)

    replayed = vp.plan_from_json(
        summary,
        replay_context=replay_context_for_plan(plan),
        full_size_records=(vp.full_size_record_from_json(full_size_row),),
        check_records=(vp.check_record_from_json(reference_row),),
        candidate_records=(candidate_row,),
        materializers={
            "family": materialize_candidate,
        },
        run_dir=tmp_path,
    )
    replayed_operator = vp.materialize(replayed, family="family")

    assert torch.equal(replayed_operator(), torch.tensor([1.0]))


def test_tune_measures_every_probe_input() -> None:
    calls = []
    model = torch.nn.Linear(1, 1)
    candidate = vp.Candidate("family", "row", {}, admission_status="passed")

    def reference_check(
        candidate: vp.Candidate,
        batch: Mapping[str, object],
        vector: vp.TensorTree,
    ) -> vp.ReferenceResult:
        assert candidate.candidate_id == "row"
        assert batch["source"] == "reference"
        assert isinstance(vector, torch.Tensor)

        return reference_passed()

    def operation_factory(
        candidate: vp.Candidate,
        batch: Mapping[str, object],
        vector: vp.TensorTree,
    ) -> vp.CandidateOperation:
        assert candidate.candidate_id == "row"
        assert isinstance(vector, torch.Tensor)

        def operation() -> torch.Tensor:
            calls.append(batch["index"])

            return vector

        return operation

    problem = vp.Problem(
        model=model,
        params=vp.parameter_surface(model),
        data=TwoProbeData(),
        operator=vp.hvp("family", "objective", aggregation="sum"),
        vectors=TwoVectorProvider(),
        target=cpu_target(
            vp.TimingPolicy(
                short_seconds=0.0,
                medium_seconds=0.0,
                long_warmups=0,
                long_measured_calls=1,
            )
        ),
        runtime=vp.RuntimeConfig(
            (candidate,),
            operation_factory,
            reference_check,
            materialize_candidate,
            None,
            {"generator": "two-probe"},
        ),
    )
    plan = vp.tune(
        problem,
        memory_backend=CPUMemoryBackend(),
        clock=SequenceClock((0.0, 1.0)),
    )

    assert calls == [0, 1]
    assert plan.full_size_records[0].output_signature["type"] == "sequence"


def test_tree_reference_check_compares_candidate_to_anchor() -> None:
    candidate = vp.Candidate(
        "family",
        "row",
        {"mode": "same"},
        admission_status="passed",
    )
    bad_candidate = vp.Candidate(
        "family",
        "bad",
        {"mode": "bad"},
        admission_status="passed",
    )

    def anchor_factory(
        candidate: vp.Candidate,
        batch: Mapping[str, object],
        vector: vp.TensorTree,
    ) -> vp.CandidateOperation:
        assert candidate.candidate_id == "anchor"
        assert batch["family"] == "family"
        assert isinstance(vector, torch.Tensor)

        return vp.constant_operation(vector)

    def candidate_factory(
        candidate: vp.Candidate,
        batch: Mapping[str, object],
        vector: vp.TensorTree,
    ) -> vp.CandidateOperation:
        assert batch["family"] == "family"
        assert isinstance(vector, torch.Tensor)

        if candidate.candidate_id == "bad":
            return vp.constant_operation(vector * 2.0)

        return vp.constant_operation(vector)

    check = vp.tree_reference_check(
        anchor_factory=anchor_factory,
        candidate_factory=candidate_factory,
        thresholds={"max_abs_diff": 1e-6, "max_rel_diff": 1e-6},
        anchor_candidate_id="anchor",
    )
    result = check(candidate, {"family": "family"}, torch.tensor([1.0]))

    assert result.measurements == {"max_abs_diff": 0.0, "max_rel_diff": 0.0}

    with pytest.raises(vp.ReferenceFailedError):
        check(bad_candidate, {"family": "family"}, torch.tensor([1.0]))


def test_records_round_trip_through_json(tmp_path: Path) -> None:
    candidate = vp.Candidate(
        "family",
        "row",
        {"dtype": "float32"},
        admission_status="passed",
    )
    record = _record(
        candidate,
        elapsed=(1.0,),
        reserved=(2.0,),
        input_signature={"case": "json"},
    )
    plan = vp.Plan(
        selected={"family": candidate},
        records={"family": record},
        input_signature={"case": "json"},
        policy=vp.SelectionPolicy(),
        full_size_records=(record,),
        materializers={"family": materialize_candidate},
    )
    row_path = tmp_path / "full_size.json"
    plan_path = tmp_path / "plan.json"

    write_record(row_path, vp.full_size_record_to_json(record))
    write_record(plan_path, vp.plan_to_json(plan))

    loaded = read_record(row_path)
    round_tripped = vp.full_size_record_from_json(loaded)
    loaded_plan = read_record(plan_path)

    assert round_tripped.owner_hash == record.owner_hash
    assert loaded_plan["owner_hash"] == plan.owner_hash()
    assert vp.plan_record_current(loaded_plan, plan)

    stale_row = dict(loaded)
    stale_row["owner_hash"] = "stale"

    with pytest.raises(vp.StaleRecordError):
        vp.full_size_record_from_json(stale_row)

    stale_plan = dict(loaded_plan)
    stale_plan["owner_hash"] = "stale"

    assert not vp.plan_record_current(stale_plan, plan)


def test_plan_replay_requires_all_saved_rows(tmp_path: Path) -> None:
    model = torch.nn.Linear(1, 1)
    candidate = vp.Candidate(
        "family",
        "row",
        {"scale": 1.0},
        admission_status="passed",
    )

    def reference_check(
        candidate: vp.Candidate,
        batch: Mapping[str, object],
        vector: vp.TensorTree,
    ) -> vp.ReferenceResult:
        assert candidate.candidate_id == "row"
        assert batch["source"] == "reference"
        assert isinstance(vector, torch.Tensor)

        return reference_passed()

    def operation_factory(
        candidate: vp.Candidate,
        batch: Mapping[str, object],
        vector: vp.TensorTree,
    ) -> vp.CandidateOperation:
        assert candidate.candidate_id == "row"
        assert batch["source"] == "probe"
        assert isinstance(vector, torch.Tensor)

        return vp.constant_operation(vector)

    problem = vp.Problem(
        model=model,
        params=vp.parameter_surface(model),
        data=OneBatchData(),
        operator=vp.hvp("family", "objective", aggregation="sum"),
        vectors=OneVectorProvider(),
        target=cpu_target(
            vp.TimingPolicy(
                short_seconds=0.0,
                medium_seconds=0.0,
                long_warmups=0,
                long_measured_calls=1,
            )
        ),
        runtime=vp.RuntimeConfig(
            (candidate,),
            operation_factory,
            reference_check,
            materialize_candidate,
            None,
            {"generator": "replay-test"},
        ),
    )
    plan = vp.tune(
        problem,
        run_dir=tmp_path,
        memory_backend=CPUMemoryBackend(),
        clock=SequenceClock((0.0, 1.0)),
    )
    summary = read_record(tmp_path / "summaries" / "tuning.json")

    with pytest.raises(vp.VPTuneError):
        vp.plan_from_json(
            summary,
            replay_context=replay_context_for_plan(plan),
            full_size_records=plan.full_size_records,
            check_records=(),
            candidate_records=candidate_records_for_plan(plan),
            materializers={"family": materialize_candidate},
            run_dir=tmp_path,
        )

    mismatched_summary = dict(summary)
    mismatched_summary["records"] = {}

    with pytest.raises(vp.VPTuneError):
        vp.plan_from_json(
            mismatched_summary,
            replay_context=replay_context_for_plan(plan),
            full_size_records=plan.full_size_records,
            check_records=plan.check_records,
            candidate_records=candidate_records_for_plan(plan),
            materializers={"family": materialize_candidate},
            run_dir=tmp_path,
        )

    with pytest.raises(vp.VPTuneError):
        vp.plan_from_json(
            summary,
            replay_context=replay_context_for_plan(plan),
            full_size_records=plan.full_size_records,
            check_records=plan.check_records,
            candidate_records=candidate_records_for_plan(plan),
            materializers={},
            run_dir=tmp_path,
        )


def test_plan_replay_rejects_stale_context_and_materializer() -> None:
    candidate = vp.Candidate(
        "family",
        "row",
        {"scale": 1.0},
        admission_status="passed",
    )
    record = _current_record(
        _record(
            candidate,
            elapsed=(1.0,),
            reserved=(1.0,),
            input_signature={"case": "replay-context"},
        )
    )
    check = _check_record(candidate, input_signature={"case": "replay-context"})
    plan = vp.Plan(
        selected={"family": candidate},
        records={"family": record},
        input_signature={"case": "replay-context"},
        policy=vp.SelectionPolicy(),
        full_size_records=(record,),
        check_records=(check,),
        materializers={"family": materialize_candidate},
        **_identity_kwargs(),
    )
    summary = vp.plan_to_json(plan)
    context = replay_context_for_plan(plan)
    other_materializer = vp.CallableMaterializer(
        "tests.other_materializer",
        "1",
        {},
        materialize_candidate_impl,
    )

    assert (
        plan.owner_hash()
        != dataclasses.replace(
            plan,
            materializers={"family": other_materializer},
        ).owner_hash()
    )

    with pytest.raises(vp.StaleRecordError):
        vp.plan_from_json(
            summary,
            replay_context=dataclasses.replace(
                context,
                input_signature={"case": "other"},
            ),
            full_size_records=(record,),
            check_records=(check,),
            candidate_records=candidate_records_for_plan(plan),
            materializers={"family": materialize_candidate},
        )

    with pytest.raises(vp.StaleRecordError):
        vp.plan_from_json(
            summary,
            replay_context=dataclasses.replace(
                context,
                family_input_signatures={"family": {"case": "other"}},
            ),
            full_size_records=(record,),
            check_records=(check,),
            candidate_records=candidate_records_for_plan(plan),
            materializers={"family": materialize_candidate},
        )

    with pytest.raises(vp.StaleRecordError):
        vp.plan_from_json(
            summary,
            replay_context=dataclasses.replace(
                context,
                selection_policy=vp.SelectionPolicy(near_fastest_multiplier=1.01),
            ),
            full_size_records=(record,),
            check_records=(check,),
            candidate_records=candidate_records_for_plan(plan),
            materializers={"family": materialize_candidate},
        )

    with pytest.raises(vp.StaleRecordError):
        vp.plan_from_json(
            summary,
            replay_context=dataclasses.replace(
                context,
                materializer_identities={"family": other_materializer.identity()},
            ),
            full_size_records=(record,),
            check_records=(check,),
            candidate_records=candidate_records_for_plan(plan),
            materializers={"family": materialize_candidate},
        )

    with pytest.raises(vp.StaleRecordError):
        vp.plan_from_json(
            summary,
            replay_context=dataclasses.replace(
                context,
                target_identity={"target": "other"},
            ),
            full_size_records=(record,),
            check_records=(check,),
            candidate_records=candidate_records_for_plan(plan),
            materializers={"family": materialize_candidate},
        )

    with pytest.raises(vp.StaleRecordError):
        vp.plan_from_json(
            summary,
            replay_context=dataclasses.replace(
                context,
                runtime_identities={"family": {"runtime": "other"}},
            ),
            full_size_records=(record,),
            check_records=(check,),
            candidate_records=candidate_records_for_plan(plan),
            materializers={"family": materialize_candidate},
        )

    with pytest.raises(vp.StaleRecordError):
        vp.plan_from_json(
            summary,
            replay_context=dataclasses.replace(
                context,
                adapter_identities={"family": {"adapter": "other"}},
            ),
            full_size_records=(record,),
            check_records=(check,),
            candidate_records=candidate_records_for_plan(plan),
            materializers={"family": materialize_candidate},
        )

    with pytest.raises(vp.StaleRecordError):
        vp.plan_from_json(
            summary,
            replay_context=context,
            full_size_records=(record,),
            check_records=(check,),
            candidate_records=candidate_records_for_plan(plan),
            materializers={"family": other_materializer},
        )

    with pytest.raises(vp.VPTuneError):
        vp.plan_from_json(
            summary,
            replay_context=context,
            full_size_records=(record,),
            check_records=(check,),
            candidate_records=(),
            materializers={"family": materialize_candidate},
        )

    stale_candidate_row = dict(candidate_records_for_plan(plan)[0])
    stale_candidate_row["owner_hash"] = "stale"

    with pytest.raises(vp.StaleRecordError):
        vp.plan_from_json(
            summary,
            replay_context=context,
            full_size_records=(record,),
            check_records=(check,),
            candidate_records=(stale_candidate_row,),
            materializers={"family": materialize_candidate},
        )


def test_plan_replay_recomputes_family_selection() -> None:
    fast = vp.Candidate(
        "family",
        "fast",
        {"scale": 1.0},
        admission_status="passed",
    )
    slow = vp.Candidate(
        "family",
        "slow",
        {"scale": 2.0},
        admission_status="passed",
    )
    fast_record = _current_record(
        _record(
            fast,
            elapsed=(1.0,),
            reserved=(10.0,),
            input_signature={"case": "recompute"},
        )
    )
    slow_record = _current_record(
        _record(
            slow,
            elapsed=(2.0,),
            reserved=(1.0,),
            input_signature={"case": "recompute"},
        )
    )
    fast_check = _check_record(fast, input_signature={"case": "recompute"})
    slow_check = _check_record(slow, input_signature={"case": "recompute"})
    plan = vp.Plan(
        selected={"family": slow},
        records={"family": slow_record},
        input_signature={"case": "recompute"},
        policy=vp.SelectionPolicy(),
        full_size_records=(fast_record, slow_record),
        check_records=(fast_check, slow_check),
        materializers={"family": materialize_candidate},
        **_identity_kwargs(),
    )

    with pytest.raises(vp.StaleRecordError):
        vp.plan_from_json(
            vp.plan_to_json(plan),
            replay_context=replay_context_for_plan(plan),
            full_size_records=(fast_record, slow_record),
            check_records=(fast_check, slow_check),
            candidate_records=candidate_records_for_plan(plan),
            materializers={"family": materialize_candidate},
        )

    low_memory = vp.Candidate(
        "family",
        "low-memory",
        {"scale": 1.0},
        admission_status="passed",
    )
    high_memory = vp.Candidate(
        "family",
        "high-memory",
        {"scale": 2.0},
        admission_status="passed",
    )
    low_memory_record = _current_record(
        _record(
            low_memory,
            elapsed=(1.0,),
            reserved=(1.0,),
            input_signature={"case": "recompute-memory"},
        )
    )
    high_memory_record = _current_record(
        _record(
            high_memory,
            elapsed=(1.0,),
            reserved=(10.0,),
            input_signature={"case": "recompute-memory"},
        )
    )
    low_memory_check = _check_record(
        low_memory,
        input_signature={"case": "recompute-memory"},
    )
    high_memory_check = _check_record(
        high_memory,
        input_signature={"case": "recompute-memory"},
    )
    memory_plan = vp.Plan(
        selected={"family": high_memory},
        records={"family": high_memory_record},
        input_signature={"case": "recompute-memory"},
        policy=vp.SelectionPolicy(),
        full_size_records=(low_memory_record, high_memory_record),
        check_records=(low_memory_check, high_memory_check),
        materializers={"family": materialize_candidate},
        **_identity_kwargs(),
    )

    with pytest.raises(vp.StaleRecordError):
        vp.plan_from_json(
            vp.plan_to_json(memory_plan),
            replay_context=replay_context_for_plan(memory_plan),
            full_size_records=(low_memory_record, high_memory_record),
            check_records=(low_memory_check, high_memory_check),
            candidate_records=candidate_records_for_plan(memory_plan),
            materializers={"family": materialize_candidate},
        )


def test_selected_plan_validation_writes_failed_record(tmp_path: Path) -> None:
    candidate = vp.Candidate(
        "family",
        "row",
        {"scale": 1.0},
        admission_status="passed",
    )
    record = _record(
        candidate,
        elapsed=(1.0,),
        reserved=(1.0,),
        input_signature={"case": "validation"},
    )
    record = dataclasses.replace(record, content_hash=record.computed_content_hash())
    check = _check_record(candidate, input_signature={"case": "validation"})
    plan = vp.Plan(
        selected={"family": candidate},
        records={"family": record},
        input_signature={"case": "validation"},
        policy=vp.SelectionPolicy(),
        full_size_records=(record,),
        check_records=(check,),
        materializers={"family": materialize_candidate},
        validation_order=("family",),
        **_identity_kwargs(),
    )

    def validator(
        candidate: vp.Candidate,
        record: vp.FullSizeRecord,
        context: vp.PlanValidationContext,
    ) -> vp.ReferenceResult:
        assert candidate.candidate_id == "row"
        assert record.candidate_id == "row"
        assert torch.equal(context.selected(), torch.tensor([1.0]))
        message = "selected row failed validation"
        raise vp.ReferenceFailedError(message)

    with pytest.raises(vp.ReferenceFailedError):
        vp.validate_plan(plan, {"family": validator}, run_dir=tmp_path)

    failed = read_record(
        tmp_path
        / "references"
        / "family"
        / "row"
        / candidate.candidate_spec_hash()
        / "selected_plan_validation.json"
    )
    summary = read_record(tmp_path / "summaries" / "selected_plan_validation.json")

    assert failed["status"] == "failed"
    assert failed["error_type"] == "ReferenceFailedError"
    assert summary["status"] == "failed"
    assert summary["records"] == [failed["owner_hash"]]
    failed_record = vp.check_record_from_json(failed)

    assert vp.selected_plan_validation_summary_current(
        summary,
        plan,
        (failed_record,),
    )

    with pytest.raises(vp.VPTuneError):
        vp.plan_from_json(
            vp.plan_to_json(plan),
            replay_context=replay_context_for_plan(plan, validation_required=True),
            full_size_records=(record,),
            check_records=(check,),
            candidate_records=candidate_records_for_plan(plan),
            materializers={"family": materialize_candidate},
        )

    with pytest.raises(vp.VPTuneError):
        vp.plan_from_json(
            vp.plan_to_json(plan),
            replay_context=replay_context_for_plan(plan, validation_required=True),
            full_size_records=(record,),
            check_records=(check,),
            candidate_records=candidate_records_for_plan(plan),
            materializers={"family": materialize_candidate},
            validation_summary=summary,
            validation_records=(failed_record,),
        )

    stale_summary = dict(summary)
    stale_summary["records"] = []

    assert not vp.selected_plan_validation_summary_current(
        stale_summary,
        plan,
        (failed_record,),
    )

    with pytest.raises(vp.StaleRecordError):
        vp.plan_from_json(
            vp.plan_to_json(plan),
            replay_context=replay_context_for_plan(plan),
            full_size_records=(record,),
            check_records=(check,),
            candidate_records=candidate_records_for_plan(plan),
            materializers={"family": materialize_candidate},
            validation_summary=stale_summary,
            validation_records=(failed_record,),
        )

    with pytest.raises(vp.VPTuneError):
        vp.plan_from_json(
            vp.plan_to_json(plan),
            replay_context=replay_context_for_plan(plan),
            full_size_records=(record,),
            check_records=(check,),
            candidate_records=candidate_records_for_plan(plan),
            materializers={"family": materialize_candidate},
            validation_records=(failed_record,),
        )

    with pytest.raises(vp.MaterializationError):
        vp.validate_plan(plan, {})


def test_selected_plan_validation_writes_runtime_failure_record(
    tmp_path: Path,
) -> None:
    candidate = vp.Candidate(
        "family",
        "row",
        {"scale": 1.0},
        admission_status="passed",
    )
    record = _record(
        candidate,
        elapsed=(1.0,),
        reserved=(1.0,),
        input_signature={"case": "validation-runtime"},
    )
    plan = vp.Plan(
        selected={"family": candidate},
        records={"family": record},
        input_signature={"case": "validation-runtime"},
        policy=vp.SelectionPolicy(),
        full_size_records=(record,),
        materializers={"family": materialize_candidate},
    )

    def validator(
        candidate: vp.Candidate,
        record: vp.FullSizeRecord,
        context: vp.PlanValidationContext,
    ) -> vp.ReferenceResult:
        assert candidate.candidate_id == "row"
        assert record.candidate_id == "row"
        assert torch.equal(context.selected(), torch.tensor([1.0]))
        message = "validation runtime failed"
        raise RuntimeError(message)

    with pytest.raises(RuntimeError):
        vp.validate_plan(plan, {"family": validator}, run_dir=tmp_path)

    failed = read_record(
        tmp_path
        / "references"
        / "family"
        / "row"
        / candidate.candidate_spec_hash()
        / "selected_plan_validation.json"
    )
    summary = read_record(tmp_path / "summaries" / "selected_plan_validation.json")

    assert failed["status"] == "failed"
    assert failed["error_type"] == "RuntimeError"
    assert summary["status"] == "failed"


def test_operator_constructors_declare_kind_and_aggregation() -> None:
    operator = vp.hvp(
        "curvature",
        "capability",
        aggregation="sum",
        thresholds={"max_rel_diff": 1e-3},
    )

    assert operator.family == "curvature"
    assert operator.kind == "hvp"
    assert operator.objective_id == "capability"
    assert operator.aggregation == "sum"
    assert operator.thresholds == {"max_rel_diff": 1e-3}
    fisher = vp.fisher_vp(
        "metric",
        "retain",
        aggregation="mean",
        distribution="categorical",
        label_policy="model_distribution",
        expectation="exact",
        sample_space="classes",
        loss_reduction="log_prob",
        denominator="num_examples",
        logits_axis=1,
    )

    assert fisher.kind == "fisher_vp"
    assert fisher.semantics["distribution"] == "categorical"
    assert (
        vp.empirical_fisher_vp("metric", "retain", aggregation="mean").kind
        == "empirical_fisher_vp"
    )
    assert vp.ggnvp("metric", "retain", aggregation="mean").kind == "ggnvp"
    assert vp.gradient("grad", "loss", aggregation="sum").kind == "gradient"
    assert vp.jvp("jvp", "function", aggregation="none").kind == "jvp"
    assert vp.vjp("vjp", "function", aggregation="none").kind == "vjp"
    assert vp.metric("metric", "retain", aggregation="mean").kind == "metric"
    assert (
        vp.inverse_metric("inverse_metric", "retain", aggregation="mean").kind
        == "inverse_metric"
    )
    assert (
        vp.composition("compose", "hvp_after_metric", aggregation="none").kind
        == "composition"
    )


def test_tune_run_uses_family_dag_order(tmp_path: Path) -> None:
    calls = []
    model = torch.nn.Linear(1, 1)
    target = cpu_target(
        vp.TimingPolicy(
            short_seconds=0.0,
            medium_seconds=0.0,
            long_warmups=0,
            long_measured_calls=1,
        )
    )
    operator_a = vp.gradient("a", "loss_a", aggregation="sum")
    operator_b = vp.hvp("b", "loss_b", aggregation="sum")

    def make_problem(name: str, operator: vp.OperatorSpec) -> vp.Problem:
        candidate = vp.Candidate(
            name,
            f"{name}:row",
            {"axis": name},
            admission_status="passed",
        )

        def reference_check(
            candidate: vp.Candidate,
            batch: Mapping[str, object],
            vector: vp.TensorTree,
        ) -> vp.ReferenceResult:
            assert batch["family"] == name
            assert batch["source"] == "reference"
            assert candidate.family == name
            assert isinstance(vector, torch.Tensor)
            assert torch.equal(vector, torch.tensor([1.0]))

            return reference_passed()

        def operation_factory(
            candidate: vp.Candidate,
            batch: Mapping[str, object],
            vector: vp.TensorTree,
        ) -> vp.CandidateOperation:
            assert batch["family"] == name
            assert batch["source"] == "probe"
            assert candidate.family == name
            assert isinstance(vector, torch.Tensor)

            def operation() -> torch.Tensor:
                calls.append(name)

                return vector

            return operation

        return vp.Problem(
            model=model,
            params=vp.parameter_surface(model),
            data=OneBatchData(),
            operator=operator,
            vectors=OneVectorProvider(),
            target=target,
            runtime=vp.RuntimeConfig(
                (candidate,),
                operation_factory,
                reference_check,
                materialize_candidate,
                None,
                {"generator": name},
            ),
        )

    run = vp.TuningRun(
        target=target,
        families=(
            vp.Family("b", operator_b, dependencies=("a",)),
            vp.Family("a", operator_a),
        ),
        problems=(make_problem("b", operator_b), make_problem("a", operator_a)),
        run_id="dag",
    )
    plan = vp.tune_run(
        run,
        run_dir=tmp_path,
        memory_backend=CPUMemoryBackend(),
        clock=SequenceClock((0.0, 1.0, 1.0, 2.0)),
    )

    assert calls == ["a", "b"]
    assert tuple(plan.selected) == ("a", "b")
    assert plan.validation_order == ("a", "b")
    assert plan.dependencies_by_family == {"a": (), "b": ("a",)}
    assert len(plan.full_size_records) == 2
    dependency_identity = plan.selected["b"].dependency_identities["a"]
    expected_dependency_identity = {
        "family": "a",
        "candidate_id": "a:row",
        "candidate_spec_hash": plan.selected["a"].candidate_spec_hash(),
        "full_size_owner_hash": plan.records["a"].owner_hash,
        "full_size_content_hash": plan.records["a"].computed_content_hash(),
        "full_size_input_signature": dict(plan.records["a"].input_signature),
        "materializer_identity": materialize_candidate.identity(),
    }

    assert dependency_identity == expected_dependency_identity
    assert plan.records["b"].dependency_identities["a"] == expected_dependency_identity
    assert plan.selected_dependency_identities() == {
        "a": {},
        "b": {"a": expected_dependency_identity},
    }

    saved_full_size, saved_checks = saved_plan_rows(tmp_path, plan)
    replayed = vp.plan_from_json(
        vp.plan_to_json(plan),
        replay_context=replay_context_for_plan(plan),
        full_size_records=saved_full_size,
        check_records=saved_checks,
        candidate_records=candidate_records_for_plan(plan),
        materializers=plan.materializers,
    )

    assert replayed.owner_hash() == plan.owner_hash()

    stale_dependency_identity = dict(expected_dependency_identity)
    stale_dependency_identity["full_size_owner_hash"] = "stale"
    stale_child = dataclasses.replace(
        plan.selected["b"],
        dependency_identities={"a": stale_dependency_identity},
    )
    stale_plan = dataclasses.replace(
        plan,
        selected={"a": plan.selected["a"], "b": stale_child},
    )

    with pytest.raises(vp.StaleRecordError):
        vp.plan_from_json(
            vp.plan_to_json(stale_plan),
            replay_context=replay_context_for_plan(stale_plan),
            full_size_records=saved_full_size,
            check_records=saved_checks,
            candidate_records=candidate_records_for_plan(stale_plan),
            materializers=plan.materializers,
        )

    validation_calls = []

    def validator(
        candidate: vp.Candidate,
        record: vp.FullSizeRecord,
        context: vp.PlanValidationContext,
    ) -> vp.ReferenceResult:
        validation_calls.append(candidate.family)
        assert record.family == candidate.family
        assert torch.equal(context.selected(), torch.tensor([1.0]))

        if candidate.family == "b":
            assert tuple(context.dependencies) == ("a",)
            assert torch.equal(context.dependencies["a"](), torch.tensor([1.0]))

        return reference_passed()

    plan_without_candidate_dependency = dataclasses.replace(
        plan,
        selected={
            "a": plan.selected["a"],
            "b": dataclasses.replace(
                plan.selected["b"],
                dependency_identities={},
            ),
        },
    )

    with pytest.raises(vp.MaterializationError):
        vp.validate_plan(
            plan_without_candidate_dependency, {"a": validator, "b": validator}
        )

    validation_records = tuple(
        vp.check_record_from_json(vp.check_record_to_json(record))
        for record in vp.validate_plan(plan, {"a": validator, "b": validator})
    )
    validation_summary = vp.selected_plan_validation_summary_record(
        plan,
        validation_records,
    )

    replayed_with_validation = vp.plan_from_json(
        vp.plan_to_json(plan),
        replay_context=replay_context_for_plan(plan, validation_required=True),
        full_size_records=saved_full_size,
        check_records=saved_checks,
        candidate_records=candidate_records_for_plan(plan),
        materializers=plan.materializers,
        validation_summary=validation_summary,
        validation_records=validation_records,
    )

    assert replayed_with_validation.owner_hash() == plan.owner_hash()

    with pytest.raises(vp.VPTuneError):
        vp.plan_from_json(
            vp.plan_to_json(plan),
            replay_context=replay_context_for_plan(plan, validation_required=True),
            full_size_records=saved_full_size,
            check_records=saved_checks,
            candidate_records=candidate_records_for_plan(plan),
            materializers=plan.materializers,
            validation_summary=validation_summary,
            validation_records=tuple(reversed(validation_records)),
        )

    assert validation_calls == ["a", "b"]


def test_tune_run_executes_declared_selected_plan_validators(
    tmp_path: Path,
) -> None:
    calls = []
    validation_calls = []
    model = torch.nn.Linear(1, 1)
    target = cpu_target(
        vp.TimingPolicy(
            short_seconds=0.0,
            medium_seconds=0.0,
            long_warmups=0,
            long_measured_calls=1,
        )
    )
    operator = vp.gradient("family", "loss", aggregation="sum")
    candidate = vp.Candidate(
        "family",
        "row",
        {"axis": "value"},
        admission_status="passed",
    )

    def reference_check(
        candidate: vp.Candidate,
        batch: Mapping[str, object],
        vector: vp.TensorTree,
    ) -> vp.ReferenceResult:
        assert candidate.family == "family"
        assert batch["source"] == "reference"
        assert isinstance(vector, torch.Tensor)

        return reference_passed()

    def operation_factory(
        candidate: vp.Candidate,
        batch: Mapping[str, object],
        vector: vp.TensorTree,
    ) -> vp.CandidateOperation:
        assert candidate.family == "family"
        assert batch["source"] == "probe"
        assert isinstance(vector, torch.Tensor)

        def operation() -> torch.Tensor:
            calls.append(candidate.candidate_id)

            return vector

        return operation

    def validator(
        candidate: vp.Candidate,
        record: vp.FullSizeRecord,
        context: vp.PlanValidationContext,
    ) -> vp.ReferenceResult:
        validation_calls.append(candidate.candidate_id)
        assert record.candidate_id == candidate.candidate_id
        assert torch.equal(context.selected(), torch.tensor([1.0]))

        return vp.ReferenceResult(
            "selected_plan_validation",
            {"max_abs_diff": 1e-6},
            {"max_abs_diff": 0.0},
        )

    problem = vp.Problem(
        model=model,
        params=vp.parameter_surface(model),
        data=OneBatchData(),
        operator=operator,
        vectors=OneVectorProvider(),
        target=target,
        runtime=vp.RuntimeConfig(
            (candidate,),
            operation_factory,
            reference_check,
            materialize_candidate,
            None,
            {"generator": "validation-run"},
        ),
    )
    run = vp.TuningRun(
        target=target,
        families=(vp.Family("family", operator),),
        problems=(problem,),
        validators={"family": validator},
        validator_identities={"family": {"validator_id": "tests.validator.v1"}},
        run_id="validation-run",
    )
    plan = vp.tune_run(
        run,
        run_dir=tmp_path,
        memory_backend=CPUMemoryBackend(),
        clock=SequenceClock((0.0, 1.0)),
    )
    saved_full_size, saved_checks = saved_plan_rows(tmp_path, plan)
    validation_record = vp.check_record_from_json(
        read_record(
            tmp_path
            / "references"
            / "family"
            / "row"
            / plan.selected["family"].candidate_spec_hash()
            / "selected_plan_validation.json"
        )
    )
    validation_summary = read_record(
        tmp_path / "summaries" / "selected_plan_validation.json"
    )
    replayed = vp.plan_from_json(
        vp.plan_to_json(plan),
        replay_context=replay_context_for_plan(plan, validation_required=True),
        full_size_records=saved_full_size,
        check_records=saved_checks,
        candidate_records=candidate_records_for_plan(plan),
        materializers=plan.materializers,
        validation_summary=validation_summary,
        validation_records=(validation_record,),
    )
    stale_validator_context = dataclasses.replace(
        replay_context_for_plan(plan, validation_required=True),
        validator_identities={"family": {"validator_id": "tests.validator.v2"}},
    )

    with pytest.raises(vp.StaleRecordError):
        vp.plan_from_json(
            vp.plan_to_json(plan),
            replay_context=stale_validator_context,
            full_size_records=saved_full_size,
            check_records=saved_checks,
            candidate_records=candidate_records_for_plan(plan),
            materializers=plan.materializers,
            validation_summary=validation_summary,
            validation_records=(validation_record,),
        )

    assert calls == ["row"]
    assert validation_calls == ["row"]
    assert plan.validation_required
    assert plan.validator_identities == {
        "family": {"validator_id": "tests.validator.v1"}
    }
    assert validation_summary["status"] == "passed"

    with pytest.raises(vp.VPTuneError):
        vp.plan_from_json(
            vp.plan_to_json(plan),
            replay_context=replay_context_for_plan(plan),
            full_size_records=saved_full_size,
            check_records=saved_checks,
            candidate_records=candidate_records_for_plan(plan),
            materializers=plan.materializers,
        )

    forged_record = dataclasses.replace(
        validation_record,
        name="tree_close",
        owner_hash=compute_record_owner_hash(
            record_type="reference",
            family=validation_record.family,
            candidate_id=validation_record.candidate_id,
            check_name="tree_close",
            input_signature=validation_record.input_signature,
            candidate_settings=validation_record.candidate_settings,
            candidate_spec_hash=validation_record.candidate_spec_hash,
            thresholds=validation_record.thresholds,
            dependency_identities=validation_record.dependency_identities,
            generator_id=validation_record.generator_id,
            generator_version=validation_record.generator_version,
        ),
    )
    forged_record = dataclasses.replace(
        forged_record,
        content_hash=forged_record.computed_content_hash(),
    )
    forged_summary = vp.selected_plan_validation_summary_record(
        plan,
        (forged_record,),
    )

    with pytest.raises(vp.VPTuneError):
        vp.plan_from_json(
            vp.plan_to_json(plan),
            replay_context=replay_context_for_plan(plan, validation_required=True),
            full_size_records=saved_full_size,
            check_records=saved_checks,
            candidate_records=candidate_records_for_plan(plan),
            materializers=plan.materializers,
            validation_summary=forged_summary,
            validation_records=(forged_record,),
        )

    assert replayed.owner_hash() == plan.owner_hash()

    bad_dir = tmp_path / "bad-validator"
    bad_run = dataclasses.replace(
        run,
        validators={"wrong": validator},
        validator_identities={"wrong": {"validator_id": "tests.bad"}},
    )

    with pytest.raises(vp.MaterializationError):
        vp.tune_run(
            bad_run,
            run_dir=bad_dir,
            memory_backend=CPUMemoryBackend(),
            clock=SequenceClock(()),
        )

    assert not (bad_dir / "summaries" / "tuning.json").exists()


def test_tune_run_selects_complete_dtype_cohort(tmp_path: Path) -> None:
    target = cpu_target(
        vp.TimingPolicy(
            short_seconds=0.0,
            medium_seconds=0.0,
            long_warmups=0,
            long_measured_calls=1,
        )
    )
    model = torch.nn.Linear(1, 1)
    operator_a = vp.gradient("a", "loss", aggregation="sum")
    operator_b = vp.gradient("b", "loss", aggregation="sum")
    calls = []

    def make_problem(
        name: str,
        operator: vp.OperatorSpec,
        candidates: tuple[vp.Candidate, ...],
    ) -> vp.Problem:
        def reference_check(
            candidate: vp.Candidate,
            batch: Mapping[str, object],
            vector: vp.TensorTree,
        ) -> vp.ReferenceResult:
            assert candidate.family == name
            assert batch["source"] == "reference"
            assert isinstance(vector, torch.Tensor)

            return reference_passed()

        def operation_factory(
            candidate: vp.Candidate,
            batch: Mapping[str, object],
            vector: vp.TensorTree,
        ) -> vp.CandidateOperation:
            assert batch["source"] == "probe"
            assert isinstance(vector, torch.Tensor)

            def operation() -> torch.Tensor:
                calls.append(candidate.candidate_id)

                return vector

            return operation

        return vp.Problem(
            model=model,
            params=vp.parameter_surface(model),
            data=OneBatchData(),
            operator=operator,
            vectors=OneVectorProvider(),
            target=target,
            runtime=vp.RuntimeConfig(
                candidates,
                operation_factory,
                reference_check,
                materialize_candidate,
                None,
                {"generator": name},
            ),
        )

    run = vp.TuningRun(
        target=target,
        families=(vp.Family("a", operator_a), vp.Family("b", operator_b)),
        problems=(
            make_problem(
                "a",
                operator_a,
                (
                    vp.Candidate(
                        "a",
                        "a-float16",
                        {"model_dtype": "float16"},
                        admission_status="passed",
                    ),
                    vp.Candidate(
                        "a",
                        "a-float32",
                        {"model_dtype": "float32"},
                        admission_status="passed",
                    ),
                ),
            ),
            make_problem(
                "b",
                operator_b,
                (
                    vp.Candidate(
                        "b",
                        "b-float16",
                        {"model_dtype": "float16"},
                        admission_status="passed",
                    ),
                    vp.Candidate(
                        "b",
                        "b-float32",
                        {"model_dtype": "float32"},
                        admission_status="passed",
                    ),
                ),
            ),
        ),
        cohort_constraints=(
            vp.CohortConstraint(
                name="dtype",
                settings_keys=("model_dtype",),
                assignments=(
                    {"model_dtype": "float16"},
                    {"model_dtype": "float32"},
                ),
            ),
        ),
        run_id="dtype-cohort",
    )
    plan = vp.tune_run(
        run,
        run_dir=tmp_path,
        memory_backend=CPUMemoryBackend(),
        clock=SequenceClock((0.0, 1.0, 1.0, 101.0, 101.0, 111.0, 111.0, 121.0)),
    )

    assert calls == ["a-float16", "b-float16", "a-float32", "b-float32"]
    assert plan.selected["a"].candidate_id == "a-float32"
    assert plan.selected["b"].candidate_id == "b-float32"
    assert plan.cohort_assignment is not None
    assert plan.cohort_assignment.values == {"model_dtype": "float32"}


def test_tune_run_uses_generic_multi_key_cohort_constraint(tmp_path: Path) -> None:
    target = cpu_target(
        vp.TimingPolicy(
            short_seconds=0.0,
            medium_seconds=0.0,
            long_warmups=0,
            long_measured_calls=1,
        )
    )
    model = torch.nn.Linear(1, 1)
    operator_a = vp.gradient("a", "loss", aggregation="sum")
    operator_b = vp.gradient("b", "loss", aggregation="sum")
    operator_c = vp.gradient("c", "loss", aggregation="sum")
    calls = []

    def make_problem(
        name: str,
        operator: vp.OperatorSpec,
        candidates: tuple[vp.Candidate, ...],
    ) -> vp.Problem:
        def reference_check(
            candidate: vp.Candidate,
            batch: Mapping[str, object],
            vector: vp.TensorTree,
        ) -> vp.ReferenceResult:
            assert candidate.family == name
            assert batch["source"] == "reference"
            assert isinstance(vector, torch.Tensor)

            return reference_passed()

        def operation_factory(
            candidate: vp.Candidate,
            batch: Mapping[str, object],
            vector: vp.TensorTree,
        ) -> vp.CandidateOperation:
            assert batch["source"] == "probe"
            assert isinstance(vector, torch.Tensor)

            def operation() -> torch.Tensor:
                calls.append(candidate.candidate_id)

                return vector

            return operation

        return vp.Problem(
            model=model,
            params=vp.parameter_surface(model),
            data=OneBatchData(),
            operator=operator,
            vectors=OneVectorProvider(),
            target=target,
            runtime=vp.RuntimeConfig(
                candidates,
                operation_factory,
                reference_check,
                materialize_candidate,
                None,
                {"generator": name},
            ),
        )

    run = vp.TuningRun(
        target=target,
        families=(
            vp.Family("a", operator_a),
            vp.Family("b", operator_b, dependencies=("a",)),
            vp.Family("c", operator_c),
        ),
        problems=(
            make_problem(
                "a",
                operator_a,
                (
                    vp.Candidate(
                        "a",
                        "a-first",
                        {"backend": "first", "chunk": 1},
                        admission_status="passed",
                    ),
                    vp.Candidate(
                        "a",
                        "a-second",
                        {"backend": "second", "chunk": 2},
                        admission_status="passed",
                    ),
                ),
            ),
            make_problem(
                "b",
                operator_b,
                (
                    vp.Candidate(
                        "b",
                        "b-first",
                        {"backend": "first", "chunk": 1},
                        admission_status="passed",
                    ),
                    vp.Candidate(
                        "b",
                        "b-second",
                        {"backend": "second", "chunk": 2},
                        admission_status="passed",
                    ),
                ),
            ),
            make_problem(
                "c",
                operator_c,
                (vp.Candidate("c", "c-row", {}, admission_status="passed"),),
            ),
        ),
        cohort_constraints=(
            vp.CohortConstraint(
                name="backend_chunk",
                settings_keys=("backend", "chunk"),
                assignments=(
                    {"backend": "first", "chunk": 1},
                    {"backend": "second", "chunk": 2},
                ),
                families=("a", "b"),
            ),
        ),
        run_id="generic-cohort",
    )
    plan = vp.tune_run(
        run,
        run_dir=tmp_path,
        memory_backend=CPUMemoryBackend(),
        clock=SequenceClock((
            0.0,
            1.0,
            1.0,
            101.0,
            101.0,
            102.0,
            102.0,
            112.0,
            112.0,
            122.0,
            122.0,
            123.0,
        )),
    )

    assert calls == ["a-first", "b-first", "c-row", "a-second", "b-second", "c-row"]
    assert plan.selected["a"].candidate_id == "a-second"
    assert plan.selected["b"].candidate_id == "b-second"
    assert plan.selected["c"].candidate_id == "c-row"
    assert plan.cohort_assignment is not None
    assert plan.cohort_assignment.values == {"backend": "second", "chunk": 2}
    assert plan.selected["b"].dependency_identities["a"]["candidate_id"] == "a-second"

    saved_full_size, saved_checks = saved_plan_rows(tmp_path, plan)
    candidate_rows = candidate_records_for_plan(plan)
    replayed = vp.plan_from_json(
        vp.plan_to_json(plan),
        replay_context=replay_context_for_plan(plan),
        full_size_records=saved_full_size,
        check_records=saved_checks,
        candidate_records=candidate_rows,
        materializers=plan.materializers,
    )

    assert replayed.owner_hash() == plan.owner_hash()

    candidates_by_key = {
        (
            candidate.family,
            candidate.candidate_id,
            candidate.candidate_spec_hash(),
        ): candidate
        for candidate in (
            vp.candidate_record_from_json(candidate_row)
            for candidate_row in candidate_rows
        )
    }
    first_records = {
        "a": plan.full_size_records[0],
        "b": plan.full_size_records[1],
        "c": plan.full_size_records[2],
    }
    first_selected = {
        family: candidates_by_key[
            record.family, record.candidate_id, record.candidate_spec_hash
        ]
        for family, record in first_records.items()
    }
    first_assignment_record = first_selected["a"].cohort_assignment
    stale_plan = dataclasses.replace(
        plan,
        selected=first_selected,
        records=first_records,
        cohort_assignment=vp.CohortAssignment(
            assignment_id=str(first_assignment_record["assignment_id"]),
            values=dict(first_assignment_record["values"]),
            constraints=tuple(
                str(name) for name in first_assignment_record["constraints"]
            ),
            covered_families=tuple(
                str(family) for family in first_assignment_record["covered_families"]
            ),
        ),
    )

    with pytest.raises(vp.StaleRecordError):
        vp.plan_from_json(
            vp.plan_to_json(stale_plan),
            replay_context=replay_context_for_plan(stale_plan),
            full_size_records=saved_full_size,
            check_records=saved_checks,
            candidate_records=candidate_rows,
            materializers=plan.materializers,
        )


def test_tune_run_writes_prerequisite_failed_descendants(tmp_path: Path) -> None:
    target = cpu_target(
        vp.TimingPolicy(
            short_seconds=0.0,
            medium_seconds=0.0,
            long_warmups=0,
            long_measured_calls=1,
        )
    )
    model = torch.nn.Linear(1, 1)
    operator_a = vp.gradient("a", "loss", aggregation="sum")
    operator_b = vp.gradient("b", "loss", aggregation="sum")
    calls = []

    def make_problem(
        name: str,
        operator: vp.OperatorSpec,
        candidates: tuple[vp.Candidate, ...],
    ) -> vp.Problem:
        def reference_check(
            candidate: vp.Candidate,
            batch: Mapping[str, object],
            vector: vp.TensorTree,
        ) -> vp.ReferenceResult:
            assert batch["source"] == "reference"
            assert isinstance(vector, torch.Tensor)

            if candidate.candidate_id == "a-first":
                message = "reference failed"
                raise RuntimeError(message)

            return reference_passed()

        def operation_factory(
            candidate: vp.Candidate,
            batch: Mapping[str, object],
            vector: vp.TensorTree,
        ) -> vp.CandidateOperation:
            assert batch["source"] == "probe"
            assert isinstance(vector, torch.Tensor)

            def operation() -> torch.Tensor:
                calls.append(candidate.candidate_id)

                return vector

            return operation

        return vp.Problem(
            model=model,
            params=vp.parameter_surface(model),
            data=OneBatchData(),
            operator=operator,
            vectors=OneVectorProvider(),
            target=target,
            runtime=vp.RuntimeConfig(
                candidates,
                operation_factory,
                reference_check,
                materialize_candidate,
                None,
                {"generator": name},
            ),
        )

    run = vp.TuningRun(
        target=target,
        families=(
            vp.Family("a", operator_a),
            vp.Family("b", operator_b, dependencies=("a",)),
        ),
        problems=(
            make_problem(
                "a",
                operator_a,
                (
                    vp.Candidate(
                        "a",
                        "a-first",
                        {"backend": "first"},
                        admission_status="passed",
                    ),
                    vp.Candidate(
                        "a",
                        "a-second",
                        {"backend": "second"},
                        admission_status="passed",
                    ),
                ),
            ),
            make_problem(
                "b",
                operator_b,
                (
                    vp.Candidate(
                        "b",
                        "b-first",
                        {"backend": "first"},
                        admission_status="passed",
                    ),
                    vp.Candidate(
                        "b",
                        "b-second",
                        {"backend": "second"},
                        admission_status="passed",
                    ),
                ),
            ),
        ),
        cohort_constraints=(
            vp.CohortConstraint(
                name="backend",
                settings_keys=("backend",),
                assignments=({"backend": "first"}, {"backend": "second"}),
            ),
        ),
        run_id="blocked-descendant",
    )
    plan = vp.tune_run(
        run,
        run_dir=tmp_path,
        memory_backend=CPUMemoryBackend(),
        clock=SequenceClock((0.0, 1.0, 1.0, 2.0)),
    )

    blocked = tuple(
        record
        for record in plan.full_size_records
        if record.error_type == "PrerequisiteFailed"
    )
    failed_references = tuple(
        record
        for record in plan.full_size_records
        if record.error_type == "RuntimeError"
    )

    assert calls == ["a-second", "b-second"]
    assert plan.selected["a"].candidate_id == "a-second"
    assert plan.selected["b"].candidate_id == "b-second"
    assert tuple(record.candidate_id for record in failed_references) == ("a-first",)
    assert tuple(record.candidate_id for record in blocked) == ("b-first",)
